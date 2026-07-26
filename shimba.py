#!/usr/bin/env python3
"""
shimba.py — One-file LLM trainer for any platform (CPU/GPU/Colab/Cloud).

Usage:
  # Install:  pip install torch numpy
  # Train:    python shimba.py train --data data.txt --out model.pth
  # Chat:     python shimba.py chat --model model.pth
  # Eval:     python shimba.py benchmark --model model.pth
  # Serve:    python shimba.py serve --model model.pth --port 8000
"""

import os, sys, math, json, time, random, shutil, tempfile
from dataclasses import dataclass
from typing import Optional, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("Need PyTorch. Run: pip install torch")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 512
    n_embd:     int = 256
    n_head:     int = 4
    n_layer:    int = 4
    dropout:    float = 0.1
    bias:       bool  = False


@dataclass
class TrainConfig:
    batch_size:               int   = 8
    gradient_accumulation_steps: int = 4
    learning_rate:            float = 3e-4
    max_iters:                int   = 5000
    weight_decay:             float = 0.1
    grad_clip:                float = 1.0
    warmup_iters:             int   = 200
    lr_decay_iters:           int   = 5000
    min_lr:                   float = 3e-5
    eval_interval:            int   = 500
    eval_iters:               int   = 100
    out_path:                 str   = "model.pth"
    tokenizer_path:           str   = ""
    log_interval:             int   = 50
    seed:                     int   = 42


# ═══════════════════════════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════════════════════════

class CharTokenizer:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.char2idx: dict = {}
        self.idx2char: dict = {}
        self.vocab_size: int = 0

    def build(self, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        vocab = [self.PAD_TOKEN, self.UNK_TOKEN] + chars
        self.char2idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx2char = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)
        print(f"[tok] vocab: {self.vocab_size} ({len(chars)} chars)")
        return self

    def encode(self, text: str) -> List[int]:
        unk = self.char2idx[self.UNK_TOKEN]
        if not hasattr(self, '_trans'):
            self._trans = {ord(c): i for c, i in self.char2idx.items() if len(c) == 1}
            self._unk = unk
        return [self._trans.get(ord(c), self._unk) for c in text]

    def decode(self, ids: List[int]) -> str:
        parts = []
        for i in ids:
            ch = self.idx2char.get(i, self.UNK_TOKEN)
            if ch not in (self.PAD_TOKEN, self.UNK_TOKEN):
                parts.append(ch)
        return "".join(parts)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char2idx": self.char2idx, "vocab_size": self.vocab_size}, f, ensure_ascii=False)
        print(f"[tok] saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok.char2idx = data["char2idx"]
        tok.idx2char = {int(k): v for v, k in data["char2idx"].items()}
        tok.vocab_size = data["vocab_size"]
        print(f"[tok] loaded <- {path} (vocab={tok.vocab_size})")
        return tok

    @staticmethod
    def default_path(model_path: str) -> str:
        return os.path.splitext(model_path)[0] + "_tokenizer.json"


# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════

class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd, self.head_dim = cfg.n_head, cfg.n_embd, cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer("causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        def _r(t): return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = _r(q), _r(k), _r(v)
        try:
            y = F.scaled_dot_product_attention(q, k, v,
                dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        except TypeError:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.act = nn.GELU()
    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)]),
            ln_f=LayerNorm(cfg.n_embd, cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
        for n, p in self.named_parameters():
            if n.endswith(("c_proj.weight", "proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        n_p = sum(p.numel() for p in self.parameters())
        print(f"[model] params: {n_p:,} ({n_p*4/1024**2:.1f}MB)")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizer(self, lr: float, wd: float):
        decay, no_decay = set(), set()
        whitelist, blacklist = (nn.Linear,), (LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"): no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist): decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist): no_decay.add(fpn)
        decay.discard("lm_head.weight")
        pd = {pn: p for pn, p in self.named_parameters()}
        return torch.optim.AdamW([
            {"params": [pd[pn] for pn in sorted(decay)], "weight_decay": wd},
            {"params": [pd[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ], lr=lr, betas=(0.9, 0.95))

    def save(self, path: str):
        torch.save({"config": self.cfg, "state_dict": self.state_dict()}, path)
        print(f"[model] saved -> {path}")

    @classmethod
    def load(cls, path: str):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except:
            ckpt = torch.load(path, map_location="cpu")
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt.get("state_dict", ckpt.get("model")))
        model.eval()
        print(f"[model] loaded <- {path}")
        return model


# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

class TextDataset:
    def __init__(self, tokens: list, block_size: int):
        self.data = torch.tensor(tokens, dtype=torch.int16)
        self.block_size = block_size
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1].to(torch.int64)
        return chunk[:-1], chunk[1:]


def make_splits(tokens: list, block_size: int, val_fraction=0.1) -> Tuple:
    n = len(tokens)
    split = int(n * (1 - val_fraction))
    train_ds = TextDataset(tokens[:split], block_size)
    val_ds = TextDataset(tokens[split:], block_size)
    print(f"[data] train: {len(tokens[:split]):,} val: {len(tokens[split:]):,} tokens")
    return train_ds, val_ds


class DataLoader:
    def __init__(self, dataset: TextDataset, batch_size: int):
        self.dataset, self.batch_size = dataset, batch_size
    def __len__(self):
        return len(self.dataset) // self.batch_size
    def get_batch(self):
        n = len(self.dataset)
        idxs = [random.randint(0, n-1) for _ in range(self.batch_size)]
        xs, ys = zip(*[self.dataset[i] for i in idxs])
        return torch.stack(xs), torch.stack(ys)


def load_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


# ═══════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════

def get_lr(cfg: TrainConfig, it: int) -> float:
    if it < cfg.warmup_iters: return cfg.learning_rate * it / max(1, cfg.warmup_iters)
    if it > cfg.lr_decay_iters: return cfg.min_lr
    p = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * p)) * (cfg.learning_rate - cfg.min_lr)


class Trainer:
    def __init__(self, mcfg: GPTConfig, tcfg: TrainConfig, train_ds: TextDataset, val_ds: TextDataset):
        torch.manual_seed(tcfg.seed)
        self.model = GPT(mcfg)
        self.mcfg, self.tcfg = mcfg, tcfg
        self.train_loader = DataLoader(train_ds, tcfg.batch_size)
        self.val_loader = DataLoader(val_ds, tcfg.batch_size)
        self.model.train()
        if hasattr(torch, "compile"):
            try: self.model = torch.compile(self.model, mode="reduce-overhead")
            except: pass
        self.optimizer = self.model.configure_optimizer(tcfg.learning_rate, tcfg.weight_decay)
        self.best_val_loss = float("inf")
        self.iter_num = 0

    @torch.no_grad()
    def _estimate_loss(self):
        self.model.eval()
        losses = {}
        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            total, count = 0.0, min(self.tcfg.eval_iters, len(loader))
            if count == 0: losses[split] = float("inf"); continue
            for _ in range(count):
                x, y = loader.get_batch()
                _, loss = self.model(x, y)
                total += loss.item()
            losses[split] = total / count
        self.model.train()
        return losses

    def _train_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(self.tcfg.gradient_accumulation_steps):
            x, y = self.train_loader.get_batch()
            _, loss = self.model(x, y)
            (loss / self.tcfg.gradient_accumulation_steps).backward()
            loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
        self.optimizer.step()
        self.iter_num += 1
        return loss_acc

    def run(self):
        t0 = time.time()
        print(f"\n{'='*50}\n  Shimba LLM Training\n{'='*50}\n")
        while self.iter_num < self.tcfg.max_iters:
            lr = get_lr(self.tcfg, self.iter_num)
            for pg in self.optimizer.param_groups: pg["lr"] = lr
            if self.iter_num % self.tcfg.eval_interval == 0:
                losses = self._estimate_loss()
                vs = f"{losses['val']:.4f}" if losses['val'] != float('inf') else "skip"
                print(f"  iter {self.iter_num:5d}  train={losses['train']:.4f}  val={vs}  lr={lr:.2e}  time={time.time()-t0:.0f}s")
                if losses["val"] < self.best_val_loss:
                    self.best_val_loss = losses["val"]
                    self.model.save(self.tcfg.out_path)
                    print(f"  [+] best model (val={self.best_val_loss:.4f})")
            loss = self._train_step()
            if self.iter_num % self.tcfg.log_interval == 0:
                print(f"  iter {self.iter_num:5d}  loss={loss:.4f}  lr={lr:.2e}")
        final = self.tcfg.out_path.replace(".pth", "_final.pth")
        self.model.save(final)
        print(f"\nDone in {time.time()-t0:.0f}s  best val_loss={self.best_val_loss:.4f}")


# ═══════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=200, temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1):
    model.eval()
    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.int64)
    gen_ids = []
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        logits = logits[:, -1, :]
        if repeat_penalty != 1.0 and gen_ids:
            for tid in set(ids + gen_ids):
                if tid < logits.size(-1): logits[0, tid] /= repeat_penalty
        if temperature == 0.0:
            nid = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                k = min(top_k, logits.size(-1)); th = torch.topk(logits, k).values[:, -1:]
                logits = logits.masked_fill(logits < th, float('-inf'))
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cp - F.softmax(sl, dim=-1) > top_p] = float('-inf')
                logits = torch.scatter(logits, 1, si, sl)
            nid = torch.multinomial(F.softmax(logits, dim=-1), 1)
        tok = nid.item()
        gen_ids.append(tok)
        idx = torch.cat([idx, nid], dim=1)
    full = ids + gen_ids
    return tokenizer.decode(full)


@torch.no_grad()
def stream_generate(model, tokenizer, prompt, max_new=200, temperature=0.8, top_k=40, top_p=0.95):
    model.eval()
    ids = tokenizer.encode(prompt) or [0]
    sys.stdout.write(prompt); sys.stdout.flush()
    idx = torch.tensor([ids], dtype=torch.int64)
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k > 0:
            k = min(top_k, logits.size(-1)); th = torch.topk(logits, k).values[:, -1:]
            logits = logits.masked_fill(logits < th, float('-inf'))
        if top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            sl[cp - F.softmax(sl, dim=-1) > top_p] = float('-inf')
            logits = torch.scatter(logits, 1, si, sl)
        nid = torch.multinomial(F.softmax(logits, dim=-1), 1)
        sys.stdout.write(tokenizer.decode([nid.item()]))
        sys.stdout.flush()
        idx = torch.cat([idx, nid], dim=1)
    sys.stdout.write("\n")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════

BENCHMARKS = [
    ("Identity", "User: who are you\nResponse:", ["ai", "assistant", "help"]),
    ("Identity", "User: what is your name\nResponse:", ["name", "ai", "called"]),
    ("Identity", "User: what can you do\nResponse:", ["help", "answer", "assist"]),
    ("Knowledge", "User: what is gravity\nResponse:", ["force", "mass", "earth", "attract"]),
    ("Knowledge", "User: what is machine learning\nResponse:", ["learn", "data", "ai", "model"]),
    ("Knowledge", "User: what is python\nResponse:", ["programming", "language", "code"]),
    ("Math", "User: what is 2 plus 2\nResponse:", ["4", "four"]),
    ("Conversation", "User: hi\nResponse:", ["hello", "hi", "help"]),
    ("Conversation", "User: good morning\nResponse:", ["morning", "day", "good"]),
    ("Conversation", "User: thanks\nResponse:", ["welcome", "glad", "help"]),
]


def score_response(response: str, keywords: list) -> dict:
    r = response.lower(); words = response.split()
    coherence = 1 if len(words) >= 3 and 3 <= (sum(len(w) for w in words)/max(len(words),1)) <= 10 else 0
    hits = sum(1 for kw in keywords if kw in r)
    relevance = 1 if hits >= 2 else (0.5 if hits >= 1 else 0)
    rep = 1
    if len(words) > 6 and len(set(words))/len(words) < 0.4: rep = 0
    return {"coherence": coherence, "relevance": relevance, "repetition": rep, "total": coherence + relevance + rep, "max": 3}


def run_benchmark(model_path: str, temperature=0.7, top_k=40, save_path=None):
    tok = CharTokenizer.load(CharTokenizer.default_path(model_path))
    model = GPT.load(model_path)
    results, by_cat, lines = [], {}, []
    def log(s=""): print(s); lines.append(s)
    log(f"\n{'='*50}\n  Shimba Benchmark\n  Model: {model_path}\n{'='*50}")
    for i, (cat, prompt, kw) in enumerate(BENCHMARKS, 1):
        t0 = time.time()
        response = generate(model, tok, prompt, max_new=80, temperature=temperature, top_k=top_k)
        response = response[len(prompt):].strip()
        scores = score_response(response, kw)
        results.append(scores["total"])
        q = prompt.split("User: ")[1].split("\n")[0]
        log(f"[{i:02d}] {cat:<12} Q: {q}")
        log(f"       {response[:100]}")
        log(f"       Score: {scores['total']}/3  [{time.time()-t0:.1f}s]\n")
        by_cat.setdefault(cat, []).append(scores["total"])
    total, mx = sum(results), len(BENCHMARKS)*3
    pct = total/mx*100
    log(f"{'='*50}\n  TOTAL: {total}/{mx} ({pct:.0f}%)")
    for cat, sc in by_cat.items():
        cp = sum(sc)/(len(sc)*3)*100
        log(f"  {cat:<12} {'█'*int(cp/10)}{'░'*(10-int(cp/10))} {sum(sc):.0f}/{len(sc)*3} ({cp:.0f}%)")
    grade = "A" if pct >= 80 else "B" if pct >= 65 else "C" if pct >= 50 else "D" if pct >= 35 else "F"
    log(f"  Grade: {grade}")
    if save_path:
        with open(save_path, "w") as f: f.write(f"Model: {model_path}\n" + "\n".join(lines))
        print(f"Saved -> {save_path}")


# ═══════════════════════════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════════════════════════

def chat_loop(model_path: str, temperature=0.8, top_k=40):
    tok = CharTokenizer.load(CharTokenizer.default_path(model_path))
    model = GPT.load(model_path)
    hist = ""
    print("\nShimba Chat (type 'quit' to exit, '/reset' to clear)\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if user.lower() in ("quit", "exit"): break
        if user == "/reset": hist = ""; print("Reset."); continue
        if not user: continue
        hist += f"User: {user}\nResponse:"
        response = generate(model, tok, hist, max_new=150, temperature=temperature, top_k=top_k)
        resp = response[len(hist):].split("\n")[0].strip() if "\n" in response[len(hist):] else response[len(hist):].strip()
        print(f"AI: {resp}")
        hist += resp + "\n"


# ═══════════════════════════════════════════════════════════════
# API SERVER — OpenAI-compatible HTTP API
# ═══════════════════════════════════════════════════════════════

class ShimbaAPI:
    def __init__(self, model_path: str, host="0.0.0.0", port=8000):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.tokenizer = CharTokenizer.load(CharTokenizer.default_path(model_path))
        self.model = GPT.load(model_path)
        self.model.eval()

    def completions(self, prompt: str, max_tokens=200, temperature=0.8, top_k=40, top_p=0.95, stream=False, **kw):
        if stream:
            return self._stream_completions(prompt, max_tokens, temperature, top_k, top_p)
        text = generate(self.model, self.tokenizer, prompt, max_tokens, temperature, top_k, top_p)
        return {
            "id": f"cmpl-{random.randint(0,999999)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": os.path.basename(self.model_path),
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(self.tokenizer.encode(prompt)), "completion_tokens": len(self.tokenizer.encode(text)), "total_tokens": 0}
        }

    def _stream_completions(self, prompt, max_tokens, temperature, top_k, top_p):
        ids = self.tokenizer.encode(prompt) or [0]
        idx = torch.tensor([ids], dtype=torch.int64)
        for _ in range(max_tokens):
            logits, _ = self.model(idx[:, -self.model.cfg.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                logits = logits.masked_fill(logits < torch.topk(logits, k).values[:, -1:], float('-inf'))
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cp - F.softmax(sl, dim=-1) > top_p] = float('-inf')
                logits = torch.scatter(logits, 1, si, sl)
            nid = torch.multinomial(F.softmax(logits, dim=-1), 1)
            token = nid.item()
            text = self.tokenizer.decode([token])
            yield f"data: {json.dumps({'choices': [{'text': text, 'index': 0}]})}\n\n"
            idx = torch.cat([idx, nid], dim=1)
        yield "data: [DONE]\n\n"

    def chat_completions(self, messages, max_tokens=200, temperature=0.8, top_k=40, top_p=0.95, stream=False, **kw):
        prompt = self._messages_to_prompt(messages)
        if stream:
            return self._stream_chat(prompt, max_tokens, temperature, top_k, top_p)
        text = generate(self.model, self.tokenizer, prompt, max_tokens, temperature, top_k, top_p)
        if text.startswith(prompt): text = text[len(prompt):]
        return {
            "id": f"chatcmpl-{random.randint(0,999999)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": os.path.basename(self.model_path),
            "choices": [{"message": {"role": "assistant", "content": text.strip()}, "index": 0, "finish_reason": "stop"}],
        }

    def _stream_chat(self, prompt, max_tokens, temperature, top_k, top_p):
        ids = self.tokenizer.encode(prompt) or [0]
        idx = torch.tensor([ids], dtype=torch.int64)
        for _ in range(max_tokens):
            logits, _ = self.model(idx[:, -self.model.cfg.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                logits = logits.masked_fill(logits < torch.topk(logits, k).values[:, -1:], float('-inf'))
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cp - F.softmax(sl, dim=-1) > top_p] = float('-inf')
                logits = torch.scatter(logits, 1, si, sl)
            nid = torch.multinomial(F.softmax(logits, dim=-1), 1)
            token = nid.item()
            text = self.tokenizer.decode([token])
            yield f"data: {json.dumps({'choices': [{'delta': {'content': text}, 'index': 0}]})}\n\n"
            idx = torch.cat([idx, nid], dim=1)
        yield "data: [DONE]\n\n"

    def _messages_to_prompt(self, messages):
        prompt = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant:"
        return prompt

    def list_models(self):
        return {
            "object": "list",
            "data": [{"id": os.path.basename(self.model_path), "object": "model", "created": int(time.time()), "owned_by": "shimba"}]
        }

    def run(self):
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse

        api = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, data, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def _read_body(self):
                n = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(n)) if n else {}

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.end_headers()

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/v1/models":
                    self._send(api.list_models())
                elif parsed.path in ("/", "/health", "/v1"):
                    self._send({"status": "ok", "model": os.path.basename(api.model_path), "docs": "/v1/models, /v1/completions, /v1/chat/completions"})
                else:
                    self._send({"error": "not found"}, 404)

            def do_POST(self):
                parsed = urllib.parse.urlparse(self.path)
                body = self._read_body()

                if parsed.path == "/v1/completions":
                    is_stream = body.pop("stream", False)
                    if is_stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        for chunk in api.completions(stream=True, **body):
                            self.wfile.write(chunk.encode())
                            self.wfile.flush()
                        return
                    result = api.completions(**body)
                    self._send(result)

                elif parsed.path == "/v1/chat/completions":
                    is_stream = body.pop("stream", False)
                    if is_stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        for chunk in api.chat_completions(stream=True, **body):
                            self.wfile.write(chunk.encode())
                            self.wfile.flush()
                        return
                    result = api.chat_completions(**body)
                    self._send(result)
                else:
                    self._send({"error": "not found"}, 404)

            def log_message(self, *a): pass

        server = HTTPServer((api.host, api.port), Handler)
        print(f"\n{'='*50}")
        print(f"  Shimba API Server")
        print(f"  Model: {api.model_path}")
        print(f"  URL:   http://{api.host}:{api.port}")
        print(f"{'='*50}")
        print(f"  Endpoints:")
        print(f"    GET  /                  Health check")
        print(f"    GET  /v1/models         List models")
        print(f"    POST /v1/completions    Text completion")
        print(f"    POST /v1/chat/completions  Chat completion")
        print(f"  Usage:  curl http://{api.host}:{api.port}/v1/completions ")
        print(f"          -d '{{\"prompt\":\"Hello\",\"max_tokens\":50}}'")
        print(f"  SDK:    from shimbasdk import ShimbaClient")
        print(f"          client = ShimbaClient(base_url='http://{api.host}:{api.port}')")
        print(f"{'='*50}\n")
        server.serve_forever()


# ═══════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════

def smoke_test():
    print("="*50 + "\n  Shimba Smoke Test\n" + "="*50)
    tok = CharTokenizer().build("Hello, world! 123 abc")
    e = tok.encode("Hello"); d = tok.decode(e)
    assert d == "Hello", f"tokenizer failed: {d}"
    print(f"[OK] tokenizer vocab={tok.vocab_size}")
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_embd=64, n_head=2, n_layer=2)
    model = GPT(cfg)
    x = torch.randint(0, tok.vocab_size, (2, 16))
    y = torch.randint(0, tok.vocab_size, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, tok.vocab_size)
    print(f"[OK] forward pass loss={loss.item():.4f}")
    # Train 5 steps
    tcfg = TrainConfig(max_iters=5, batch_size=2, eval_interval=5, eval_iters=2, log_interval=5,
                       out_path=os.path.join(tempfile.gettempdir(), "shimba_test.pth"))
    corpus = "".join(random.choices(list(tok.char2idx.keys())[2:], k=2000))
    tokens = tok.encode(corpus)
    tr, va = make_splits(tokens, 32, 0.2)
    trainer = Trainer(cfg, tcfg, tr, va)
    trainer.run()
    print(f"[OK] training done")
    result = generate(model, tok, "Hello", max_new=20, top_k=5)
    print(f"[OK] generation: {result!r}")
    print("\n" + "="*50 + "\n  ALL TESTS PASSED\n" + "="*50)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__); return

    cmd = sys.argv[1]

    if cmd == "test":
        smoke_test()
        return

    if cmd == "train":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--data", required=True)
        p.add_argument("--out", default="model.pth")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--n_embd", type=int, default=256)
        p.add_argument("--n_layer", type=int, default=6)
        p.add_argument("--n_head", type=int, default=8)
        p.add_argument("--block_size", type=int, default=256)
        p.add_argument("--batch_size", type=int, default=8)
        p.add_argument("--max_iters", type=int, default=5000)
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--eval_interval", type=int, default=500)
        p.add_argument("--eval_iters", type=int, default=100)
        p.add_argument("--val_frac", type=float, default=0.1)
        args = p.parse_args(sys.argv[2:])

        tok_path = CharTokenizer.default_path(args.out)
        if args.resume and os.path.exists(tok_path):
            tokenizer = CharTokenizer.load(tok_path)
        else:
            text = load_text(args.data)
            tokenizer = CharTokenizer().build(text)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            tokenizer.save(tok_path)

        text = load_text(args.data)
        tokens = tokenizer.encode(text)
        train_ds, val_ds = make_splits(tokens, args.block_size, args.val_frac)

        mcfg = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size,
                         n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer)
        tcfg = TrainConfig(batch_size=args.batch_size, learning_rate=args.lr,
                          max_iters=args.max_iters, eval_interval=args.eval_interval,
                          eval_iters=args.eval_iters, out_path=args.out, tokenizer_path=tok_path)

        trainer = Trainer(mcfg, tcfg, train_ds, val_ds)
        if args.resume and os.path.exists(args.out):
            ckpt = torch.load(args.out, map_location="cpu")
            trainer.model.load_state_dict(ckpt.get("state_dict", ckpt.get("model")))
            trainer.iter_num = ckpt.get("iter_num", 0)
            print(f"Resumed at iter {trainer.iter_num}")
        trainer.run()
        return

    if cmd == "chat":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--model", required=True)
        p.add_argument("--temp", type=float, default=0.8)
        p.add_argument("--top_k", type=int, default=40)
        args = p.parse_args(sys.argv[2:])
        chat_loop(args.model, args.temp, args.top_k)
        return

    if cmd == "benchmark":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--model", required=True)
        p.add_argument("--temp", type=float, default=0.7)
        p.add_argument("--top_k", type=int, default=40)
        p.add_argument("--save", default=None)
        args = p.parse_args(sys.argv[2:])
        run_benchmark(args.model, args.temp, args.top_k, args.save)
        return

    if cmd == "serve":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--model", required=True)
        p.add_argument("--host", default="0.0.0.0")
        p.add_argument("--port", type=int, default=8000)
        args = p.parse_args(sys.argv[2:])
        api = ShimbaAPI(args.model, host=args.host, port=args.port)
        api.run()
        return

    print(f"Unknown command: {cmd}")
    print(__doc__)


if __name__ == "__main__":
    main()
