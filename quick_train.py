#!/usr/bin/env python3
"""
quick_train.py -- Fast training with checkpoint every iteration and resume support.
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.model import GPT, GPTConfig
from llm.tokenizer import CharTokenizer
from llm.data import make_splits
from llm.train import Trainer, TrainConfig


class FastTrainer(Trainer):
    """Trainer that saves a checkpoint after every training step."""

    def _save_checkpoint(self, iter_num, is_best=False):
        checkpoint = {
            "config": self.model.cfg,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, self.tcfg.out_path)
        if is_best:
            best_path = self.tcfg.out_path.replace(".pth", "_best.pth")
            torch.save(checkpoint, best_path)
        print(f"  [checkpoint] saved iteration {iter_num}")

    def _train_step(self):
        loss = super()._train_step()
        self._save_checkpoint(self.iter_num, is_best=False)
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to .txt file")
    parser.add_argument("--out", default="model.pth")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--val_frac", type=float, default=0.1)
    args = parser.parse_args()

    tok_path = CharTokenizer.default_path(args.out)
    if args.resume and os.path.exists(tok_path):
        print("[quick_train] Loading existing tokenizer...")
        tokenizer = CharTokenizer.load(tok_path)
    else:
        print("[quick_train] Building tokenizer from corpus...")
        with open(args.data, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = CharTokenizer().build(text)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        tokenizer.save(tok_path)

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    train_ds, val_ds = make_splits(tokens, args.block_size, val_fraction=args.val_frac)

    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=0.0,
        bias=False,
    )

    tok_path = CharTokenizer.default_path(args.out)

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        max_iters=args.max_iters,
        weight_decay=0.0,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        out_path=args.out,
        tokenizer_path=tok_path,
    )

    trainer = FastTrainer(model_cfg, train_cfg, train_ds, val_ds)

    if args.resume and os.path.exists(args.out):
        print(f"[quick_train] Resuming from {args.out}")
        checkpoint = torch.load(args.out, map_location="cpu", weights_only=False)
        trainer.model.load_state_dict(checkpoint["model"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.iter_num = checkpoint["iter_num"]
        trainer.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"  Resumed at iteration {trainer.iter_num}")

    print(f"[quick_train] Starting training for {args.max_iters} iterations...")
    print(f"  Model size: {sum(p.numel() for p in trainer.model.parameters()):,} params")
    print(f"  Saving checkpoint to {args.out} after EVERY iteration")
    print("  Press Ctrl+C to interrupt and resume later with --resume\n")

    try:
        trainer.run()
    except KeyboardInterrupt:
        print("\n[quick_train] Interrupted. Saving final checkpoint...")
        trainer._save_checkpoint(trainer.iter_num, is_best=False)
        sys.exit(0)


if __name__ == "__main__":
    main()
