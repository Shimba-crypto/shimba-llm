"""
train.py -- Training loop for the Shimba LLM (CPU-only).

Optimisations applied
---------------------
1. Gradient accumulation -- effective batch = batch_size × grad_accum_steps
   without storing large intermediate activations simultaneously.
2. torch.compile() -- best effort, silently skipped when unavailable.
3. Cosine learning-rate schedule with linear warmup, *tied to max_iters*
   so the schedule always completes exactly when training stops.
4. Gradient clipping (max_norm=1.0) to stabilise training.
5. Fixed (deterministic) eval batches so val loss is comparable across evals.
6. Checkpoint saved whenever val loss improves; non-finite losses are never
   allowed to become the "best" model.

Anti-catastrophic-forgetting
----------------------------
Fine-tuning a small model on a narrow corpus with a plain cross-entropy
objective reliably destroys what the model already knew.  Four independent
levers are available, all opt-in and all cheap:

  l2sp            L2 pull toward the *base* weights (L2-SP).  Directly
                  penalises drift from the pre-trained solution.
  freeze_layers   Freeze embeddings + the first N transformer blocks.
                  The lower half of the network keeps its features, so the
                  upper layers only have to learn the new format.
  replay          Mix a slice of the original pre-training corpus into
                  every batch, so the old distribution never goes stale.
  forget_guard    Hold out part of the replay corpus and *watch its loss*.
                  If general-domain loss rises more than `--forget_tol`
                  above the loss at the start of training, stop and keep
                  the last checkpoint that had not yet regressed.

Any one of these helps; together they make multi-stage fine-tunes additive
instead than destructive.
"""

import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .model import GPT, GPTConfig
from .data import DataLoader, TextDataset


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Optimisation
    batch_size:               int   = 8
    gradient_accumulation_steps: int = 4       # effective batch = 8×4 = 32
    learning_rate:            float = 3e-4
    max_iters:                int   = 5000
    weight_decay:             float = 0.1
    grad_clip:                float = 1.0

    # Schedule (lr_decay_iters <= 0 means "track max_iters")
    warmup_iters:             int   = -1       # <0 -> 5% of max_iters, min 20
    lr_decay_iters:           int   = -1       # <0 -> max_iters
    min_lr_frac:              float = 0.1      # floor as a fraction of peak LR

    # Evaluation & checkpointing
    eval_interval:            int   = 500
    eval_iters:               int   = 40       # fixed batches per eval
    out_path:                 str   = "model.pth"
    tokenizer_path:           str   = ""

    # Misc
    log_interval:             int   = 50
    seed:                     int   = 42
    use_compile:              bool  = False   # torch.compile; needs a C++ toolchain

    # Anti-forgetting
    l2sp:                     float = 0.0      # L2 pull toward base weights
    freeze_layers:            int   = 0        # freeze embeddings + first N blocks
    forget_tol:               float = 0.0      # 0 disables the forgetting guard

    # Derived / runtime
    min_lr:                   float = field(default=0.0, init=False)
    wall_clock_limit:         float = 0.0      # seconds; 0 = no limit

    # Device: "cpu", "cuda", or "auto" (cuda when available, else cpu)
    device:                   str   = "cpu"


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def resolve_schedule(cfg: TrainConfig) -> None:
    """
    Fill in warmup / decay lengths that were left as -1.

    The old code hard-coded lr_decay_iters=5000 regardless of max_iters, so
    a 2000-iteration fine-tune only ever used the first 40% of the cosine
    (staying at a high LR) while a 20000-iteration run hit the floor at
    5000 and spent 75% of its time at the minimum LR.  Tying the schedule to
    max_iters fixes both ends.
    """
    if cfg.lr_decay_iters <= 0:
        cfg.lr_decay_iters = cfg.max_iters
    if cfg.warmup_iters < 0:
        cfg.warmup_iters = max(20, int(0.05 * cfg.max_iters))
    cfg.min_lr = cfg.learning_rate * cfg.min_lr_frac
    cfg.warmup_iters = min(cfg.warmup_iters, max(1, cfg.lr_decay_iters // 2))


def get_lr(cfg: TrainConfig, it: int) -> float:
    """
    Cosine annealing with linear warmup.

    warmup phase   : 0 -> max_lr over warmup_iters steps
    cosine phase   : max_lr -> min_lr over lr_decay_iters steps
    flat phase     : min_lr thereafter
    """
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / max(1, cfg.warmup_iters)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    span = max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    progress = min(1.0, (it - cfg.warmup_iters) / span)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model_cfg_or_model,
        train_cfg: TrainConfig,
        train_dataset: TextDataset,
        val_dataset: TextDataset,
        replay_dataset: Optional[TextDataset] = None,
        guard_dataset: Optional[TextDataset] = None,
    ):
        torch.manual_seed(train_cfg.seed)
        resolve_schedule(train_cfg)

        # Resolve device: "auto" -> cuda when available, else cpu.
        # Anything else must be usable or we fall back to cpu with a warning.
        req = (train_cfg.device or "cpu").lower()
        if req == "auto":
            req = "cuda" if torch.cuda.is_available() else "cpu"
        if req.startswith("cuda") and not torch.cuda.is_available():
            print(f"[train] CUDA requested but not available -- falling back to cpu")
            req = "cpu"
        self.device = torch.device(req)

        # Accept either a GPTConfig (train from scratch) or an existing GPT
        # model (finetune).
        if isinstance(model_cfg_or_model, GPT):
            self.model = model_cfg_or_model
            self.mcfg = model_cfg_or_model.cfg
        else:
            self.mcfg = model_cfg_or_model
            self.model = GPT(self.mcfg)
        self.model.to(self.device)

        self.tcfg = train_cfg
        if self.device.type == "cuda":
            print(f"[train] using GPU: {torch.cuda.get_device_name(0)}")

        # Data loaders
        self.train_loader = DataLoader(train_dataset, train_cfg.batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_dataset,   train_cfg.batch_size, shuffle=False)
        self.replay_loader = (DataLoader(replay_dataset, train_cfg.batch_size, shuffle=True)
                              if replay_dataset is not None else None)
        self.guard_batches = (DataLoader(guard_dataset, train_cfg.batch_size, shuffle=False)
                              .eval_batches(min(20, max(1, len(guard_dataset) // max(1, train_cfg.batch_size))))
                              if guard_dataset is not None and len(guard_dataset) > 0 else [])

        self.model.train()
        self._try_compile()
        self._apply_freezing()

        # Optimiser (built after freezing so frozen params are excluded)
        self.optimizer = self.model.configure_optimizer(
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        # ── Anti-forgetting state ───────────────────────────────────
        self.base_weights: Optional[list] = None
        if train_cfg.l2sp > 0:
            self.base_weights = [p.detach().clone()
                                 for p in self.model.parameters() if p.requires_grad]

        self.best_val_loss: float = float("inf")
        self.iter_num:      int   = 0
        self.skipped_batches: int = 0
        self.base_guard_loss: Optional[float] = None
        self.replay_prob: float = 0.0
        self._saved_best: bool = False

    # ------------------------------------------------------------------
    def _try_compile(self) -> None:
        """
        torch.compile is OPT-IN (TrainConfig.use_compile) and off by default.

        It was previously gated on `sys.version_info < (3, 12)`, which on a
        modern interpreter silently disabled it -- but the moment that gate
        is removed, inductor tries to build a C++ precompiled header in
        /tmp.  That fails outright on small tmpfs partitions ("No space left
        on device") and costs minutes of compile time for little CPU gain.
        Enable with `--compile` when you know you have a toolchain and
        scratch space.
        """
        if not self.tcfg.use_compile:
            print("[train] torch.compile disabled (pass --compile to enable)")
            return
        try:
            self.model = torch.compile(self.model)
            print("[train] torch.compile enabled")
        except Exception as e:
            print(f"[train] torch.compile skipped: {e}")

    # ------------------------------------------------------------------
    def _apply_freezing(self) -> None:
        """
        Freeze embeddings + the first `n` transformer blocks.

        Freezing the *bottom* of the network is the important part: those
        layers hold the generic language features that a narrow fine-tune
        would otherwise overwrite.  The blocks are left in place (only
        requires_grad is cleared) so state_dict keys stay identical to the
        base checkpoint and the result remains a drop-in replacement.
        """
        n = self.tcfg.freeze_layers
        if n <= 0:
            return
        n = min(n, len(self.model.transformer.h))
        tr = self.model.transformer
        resized = getattr(self.model, "vocab_resized", False)

        # A freshly grown embedding table still holds random rows for the
        # characters that were just added.  Freezing it would lock those in
        # place forever, so only freeze the table when the vocab did not grow.
        if resized:
            print("[train] token embeddings left trainable (vocabulary grew "
                  "during this run)")
            frozen = 0
        else:
            tr.wte.weight.requires_grad_(False)
            frozen = sum(p.numel() for p in tr.wte.parameters())

        tr.wpe.weight.requires_grad_(False)
        for block in list(tr.h)[:n]:
            for p in block.parameters():
                p.requires_grad_(False)

        frozen += sum(p.numel() for p in tr.wpe.parameters())
        frozen += sum(p.numel() for b in list(tr.h)[:n] for p in b.parameters())
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[train] froze position table + first {n}/{len(tr.h)} blocks "
              f"({frozen:,}/{total:,} params = {100*frozen/total:.0f}% locked)")


    @torch.no_grad()
    def _loss_on_batches(self, batches) -> float:
        """Mean loss over a fixed list of batches."""
        if not batches:
            return float("inf")
        total = 0.0
        for x, y in batches:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            v = loss.item()
            if not math.isfinite(v):
                return float("nan")
            total += v
        return total / len(batches)

    @torch.no_grad()
    def _estimate_loss(self) -> dict:
        """Fixed-batch train/val loss, directly comparable between evals."""
        self.model.eval()
        n_train = min(self.tcfg.eval_iters, max(1, len(self.train_loader)))
        losses = {
            "train": self._loss_on_batches(self.train_loader.eval_batches(n_train)),
            "val":   self._loss_on_batches(self.val_loader.eval_batches(n_train)),
        }
        self.model.train()
        return losses

    # ------------------------------------------------------------------
    def _l2sp_penalty(self) -> torch.Tensor:
        """Sum of squared distances from the base weights."""
        if not self.base_weights:
            return torch.zeros((), device=self.device)
        total = None
        for p, base in zip((p for p in self.model.parameters() if p.requires_grad),
                           self.base_weights):
            d = (p - base).pow(2).sum()
            total = d if total is None else total + d
        return total if total is not None else torch.zeros((), device=self.device)

    # ------------------------------------------------------------------
    def _train_step(self) -> float:
        """One training step: mix replay -> accumulate -> clip -> step."""
        tcfg = self.tcfg
        self.optimizer.zero_grad(set_to_none=True)
        accum = 0.0
        n_used = 0

        for _ in range(tcfg.gradient_accumulation_steps):
            x, y = self.train_loader.get_batch()
            if self.replay_loader is not None and self.replay_prob > 0:
                # Interleave a slice of the original corpus into the batch so
                # the general distribution stays in the loss.
                k = max(1, int(round(x.size(0) * self.replay_prob)))
                rx, ry = self.replay_loader.get_batch()
                x = torch.cat([x, rx[:k]], dim=0)
                y = torch.cat([y, ry[:k]], dim=0)
            x, y = x.to(self.device), y.to(self.device)

            _, loss = self.model(x, y)
            if not math.isfinite(loss.item()):
                # Every target in this batch was masked out.  Skip rather
                # than poison the weights.
                self.skipped_batches += 1
                continue

            loss = loss / tcfg.gradient_accumulation_steps
            if tcfg.l2sp > 0:
                loss = loss + tcfg.l2sp * self._l2sp_penalty()
            loss.backward()
            accum += loss.item()
            n_used += 1

        if n_used == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.iter_num += 1
            return float("nan")

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), tcfg.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.iter_num += 1
        return accum / n_used

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main training loop."""
        tcfg = self.tcfg
        t0 = time.time()

        print(f"\n{'='*64}", flush=True)
        print("  Shimba LLM -- Training", flush=True)
        print(f"  max_iters={tcfg.max_iters}  batch={tcfg.batch_size}"
              f"x{tcfg.gradient_accumulation_steps}(accum)  lr={tcfg.learning_rate:.2e}",
              flush=True)
        print(f"  schedule: warmup {tcfg.warmup_iters} -> decay {tcfg.lr_decay_iters}"
              f"  floor {tcfg.min_lr:.2e}", flush=True)
        guards = []
        if tcfg.l2sp > 0:
            guards.append(f"l2sp={tcfg.l2sp:g}")
        if tcfg.freeze_layers > 0:
            guards.append(f"freeze={tcfg.freeze_layers}")
        if self.replay_loader is not None:
            guards.append(f"replay={self.replay_prob:.0%}")
        if tcfg.forget_tol > 0 and self.guard_batches:
            guards.append(f"forget_guard=±{tcfg.forget_tol:g}")
        if guards:
            print(f"  anti-forgetting: {', '.join(guards)}", flush=True)
        print(f"{'='*64}\n", flush=True)

        if self.guard_batches:
            self.model.eval()
            self.base_guard_loss = self._loss_on_batches(self.guard_batches)
            self.model.train()
            print(f"[train] baseline general-corpus loss: {self.base_guard_loss:.4f}",
                  flush=True)

        stopped_early = ""
        while self.iter_num < tcfg.max_iters:
            # Checked at the top so a long evaluation cannot push us past
            # the budget unnoticed.
            if tcfg.wall_clock_limit > 0 and (time.time() - t0) > tcfg.wall_clock_limit:
                stopped_early = f"wall-clock limit of {tcfg.wall_clock_limit:.0f}s reached"
                print(f"\n[train] {stopped_early} at iter {self.iter_num}", flush=True)
                break

            # ── Learning-rate update ──────────────────────────────────
            lr = get_lr(tcfg, self.iter_num)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # ── Eval + checkpoint (never at iter 0: that would save an
            #    untrained model as the "best" checkpoint) ────────────
            if self.iter_num > 0 and self.iter_num % tcfg.eval_interval == 0:
                losses = self._estimate_loss()
                elapsed = time.time() - t0
                val_str = f"{losses['val']:.4f}" if math.isfinite(losses['val']) else "nonfinite"
                tr_str = f"{losses['train']:.4f}" if math.isfinite(losses['train']) else "nonfinite"
                print(f"  iter {self.iter_num:6d}/{tcfg.max_iters}  "
                      f"train={tr_str}  val={val_str}  lr={lr:.2e}  "
                      f"time={elapsed:.0f}s", flush=True)

                if math.isfinite(losses["val"]) and losses["val"] < self.best_val_loss:
                    self.best_val_loss = losses["val"]
                    self.model.save(tcfg.out_path)
                    self._saved_best = True
                    print(f"  [+] checkpoint saved (val_loss={self.best_val_loss:.4f})",
                          flush=True)

                # ── Forgetting guard ─────────────────────────────────
                if tcfg.forget_tol > 0 and self.guard_batches and \
                        self.base_guard_loss is not None:
                    self.model.eval()
                    now = self._loss_on_batches(self.guard_batches)
                    self.model.train()
                    drift = now - self.base_guard_loss
                    flag = ""
                    if drift > tcfg.forget_tol:
                        flag = "  <-- REGRESSION"
                        stopped_early = (f"forgetting guard tripped at iter {self.iter_num} "
                                         f"(general loss {self.base_guard_loss:.4f} -> {now:.4f})")
                    print(f"      general={now:.4f} (baseline {self.base_guard_loss:.4f}, "
                          f"drift {drift:+.4f}){flag}", flush=True)
                    if flag:
                        break

                if self.skipped_batches:
                    print(f"      [skipped {self.skipped_batches} fully-masked batch(es)]",
                          flush=True)

            # ── One training step ────────────────────────────────────
            accum_loss = self._train_step()

            if self.iter_num % tcfg.log_interval == 0:
                ls = f"{accum_loss:.4f}" if math.isfinite(accum_loss) else "skip"
                print(f"  iter {self.iter_num:6d}  loss={ls}  lr={lr:.2e}", flush=True)

        # ── Always leave a usable "best" checkpoint at out_path ─────
        # If max_iters < eval_interval the loop never evaluated, and the old
        # code happily left only a random-init model at out_path.
        if not self._saved_best:
            losses = self._estimate_loss()
            if math.isfinite(losses["val"]):
                self.best_val_loss = losses["val"]
                self.model.save(tcfg.out_path)
                self._saved_best = True
                print(f"  [+] final checkpoint saved (val_loss={self.best_val_loss:.4f})",
                      flush=True)

        # Final checkpoint (last weights) + tokenizer copy
        final_path = _final_path(tcfg.out_path)
        self.model.save(final_path)
        if tcfg.tokenizer_path:
            final_tok = _final_tokenizer_path(tcfg.tokenizer_path)
            if os.path.abspath(final_tok) != os.path.abspath(tcfg.tokenizer_path):
                import shutil
                shutil.copy2(tcfg.tokenizer_path, final_tok)

        total_time = time.time() - t0
        print(f"\n[train] done in {total_time:.0f}s  "
              f"best val_loss={self.best_val_loss:.4f}  "
              f"skipped_batches={self.skipped_batches}")
        if stopped_early:
            print(f"[train] stopped early: {stopped_early}")
        if math.isfinite(self.best_val_loss):
            print(f"[train] BEST model   -> {tcfg.out_path}")
        else:
            print(f"[train] no valid eval loss -- use the final model")
        print(f"[train] final model  -> {final_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _final_path(out_path: str) -> str:
    base, ext = os.path.splitext(out_path)
    return f"{base}_final{ext or '.pth'}"


def _final_tokenizer_path(tok_path: str) -> str:
    base, ext = os.path.splitext(tok_path)
    return f"{base}_final{ext or '.json'}"