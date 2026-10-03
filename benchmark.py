#!/usr/bin/env python3
"""
benchmark.py -- Evaluate Shimba LLM models on a fixed prompt suite.

Usage
-----
  python benchmark.py --model a.pth
  python benchmark.py --model a.pth --compare b.pth --compare c.pth
  python benchmark.py --model fast.pth --model reasoner.pth --system concise
  python benchmark.py --model a.pth --system reasoner --reasoning
  python benchmark.py --model a.pth --val_corpus data/base_forget.txt
  python benchmark.py --model a.pth --save results.json

What it reports
---------------
  score      0-3: coherence + relevance + no-repetition (keyword-scored, so
             it is only a rough proxy -- treat it as a smoke test)
  reasoning  for models trained with --reason: does the reply contain a
             well-formed Reasoning: chain and a separate Response: line
  val loss   held-out cross-entropy on --val_corpus, and perplexity. This is
             the objective, architecture-independent number to compare models
             with, and the one that actually reveals forgetting.

Speed
-----
Generation goes through the batched, KV-cached decoder, so all prompts run
together instead of one forward pass per token per prompt.
"""

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_default_device("cpu")

from llm.model import GPT
from llm.tokenizer import CharTokenizer
from llm.generate import generate_batch
from llm.data import TextDataset
import prompts


# ---------------------------------------------------------------------------
# Prompt suite
# ---------------------------------------------------------------------------

BENCHMARKS = [
    ("Identity",     "who are you",
     ["ai", "assistant", "shimba", "bomenater", "help", "model"]),
    ("Identity",     "what is your name",
     ["name", "ai", "shimba", "called", "i am"]),
    ("Identity",     "what can you do",
     ["help", "answer", "question", "assist", "information"]),
    ("Knowledge",    "what is gravity",
     ["force", "mass", "earth", "attract", "pull", "weight"]),
    ("Knowledge",    "what is the speed of light",
     ["light", "speed", "metres", "miles", "fast", "second"]),
    ("Knowledge",    "what is photosynthesis",
     ["plant", "light", "sun", "energy", "oxygen", "carbon"]),
    ("Knowledge",    "what is DNA",
     ["genetic", "gene", "molecule", "cell", "biology", "double"]),
    ("Knowledge",    "who was albert einstein",
     ["physicist", "relativity", "theory", "german", "scientist"]),
    ("Knowledge",    "what is machine learning",
     ["learn", "data", "ai", "model", "train", "pattern"]),
    ("Knowledge",    "what is python",
     ["programming", "language", "code", "software", "script"]),
    ("Math",         "what is 2 plus 2", ["4", "four"]),
    ("Math",         "what is the square root of 144", ["12", "twelve"]),
    ("Math",         "what is pi", ["3.14", "circle", "ratio", "math", "constant"]),
    ("Math",         "what is 10 percent of 250", ["25", "twenty five"]),
    ("Conversation", "hi", ["hello", "hi", "hey", "welcome", "help"]),
    ("Conversation", "good morning", ["morning", "day", "great", "good", "help"]),
    ("Conversation", "thanks", ["welcome", "glad", "help", "anytime", "happy"]),
    ("Conversation", "goodbye", ["bye", "goodbye", "see", "take", "return"]),
    ("Instruction",  "tell me a joke",
     ["why", "what", "joke", "laugh", "funny", "!"]),
    ("Instruction",  "help me write something",
     ["write", "help", "draft", "what", "topic", "like"]),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_response(response: str, keywords: list) -> dict:
    """
    0-3: coherence (0-1) + relevance (0-1) + no-repetition (0-1).

    Coherence is a straight readability check.  The previous version tried to
    add a bonus for punctuation and capitals but clamped the result with
    `min(coherence + 1, 1)`, which can never exceed 1 -- the bonus was dead
    code and never affected any score.
    """
    words = response.split()

    # 1. Coherence -- readable English-ish output
    coherence = 0.0
    if len(words) >= 3:
        avg_word_len = sum(len(w) for w in words) / len(words)
        alpha_ratio = sum(1 for w in words if any(c.isalpha() for c in w)) / len(words)
        if 2 <= avg_word_len <= 12 and alpha_ratio >= 0.6:
            coherence = 1.0

    # 2. Relevance -- keyword hits
    resp_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in resp_lower)
    if hits >= 2:
        relevance = 1.0
    elif hits == 1:
        relevance = 0.5
    else:
        relevance = 0.0

    # 3. Repetition -- looped characters or words
    repetition = 1.0
    for length in (2, 3, 4):
        if len(response) >= length * 4:
            for i in range(len(response) - length * 3):
                chunk = response[i:i + length]
                if response[i + length:i + length * 4] == chunk * 3:
                    repetition = 0.0
                    break
        if repetition == 0.0:
            break
    if len(words) > 6:
        if len(set(words)) / len(words) < 0.4:
            repetition = 0.0

    total = coherence + relevance + repetition
    return {"coherence": coherence, "relevance": relevance,
            "repetition": repetition, "total": total, "max": 3}


def score_reasoning(response: str) -> dict:
    """
    Structure check for models trained with --reason.

    Returns chain (0-1): did the model produce a non-trivial Reasoning: line
    followed by a separate Response: line, and is the chain longer than a
    bare restatement of the question?
    """
    has_both = ("Reasoning:" in response) and ("Response:" in response)
    if not has_both:
        return {"chain": 0.0, "well_formed": False}

    chain = response.split("Response:")[0]
    chain = chain.replace("Reasoning:", "", 1).strip()
    answer = response.split("Response:", 1)[1].strip()
    # A real chain does more than echo the question back.
    ok = len(chain) > 40 and len(answer) > 0 and chain.lower() != answer.lower()
    return {"chain": 1.0 if ok else 0.5, "well_formed": True}


# ---------------------------------------------------------------------------
# Held-out loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def corpus_loss(model, tokenizer, corpus_path: str, block_size: int = None,
                max_windows: int = 400) -> dict:
    """Cross-entropy on a held-out corpus.  Objective and comparable."""
    if not os.path.exists(corpus_path):
        return {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    if len(ids) < block_size + 2:
        return {}

    ds = TextDataset(ids, block_size)
    loader_batches = min(max_windows, len(ds))
    g = torch.Generator().manual_seed(7)
    idxs = torch.randint(0, len(ds), (loader_batches,), generator=g)

    model.eval()
    total, n = 0.0, 0
    dev = next(model.parameters()).device
    for i in idxs.tolist():
        x, y = ds[i]
        x, y = x.to(dev), y.to(dev)
        _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
        v = loss.item()
        if math.isfinite(v):
            total += v
            n += 1
    if n == 0:
        return {}
    loss = total / n
    return {"val_loss": loss, "perplexity": math.exp(min(20.0, loss)),
            "windows": n}


# ---------------------------------------------------------------------------
# Run one model
# ---------------------------------------------------------------------------

def run_benchmark(model_path: str, temperature=0.7, top_k=40, top_p=0.95,
                  max_tokens=90, system="none", reasoning=False,
                  batch_size=10, seed=42, val_corpus=None, verbose=True,
                  device="cpu"):
    tok_path = CharTokenizer.default_path(model_path)
    if not os.path.exists(tok_path):
        print(f"[skip] no tokenizer at {tok_path}")
        return None

    tokenizer = CharTokenizer.load(tok_path)
    model = GPT.load(model_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA requested but not available -- using cpu")
        device = "cpu"
    model.to(device)
    model.eval()

    sys_block = prompts.system_block(system)
    sep = prompts.TURN_SEP
    if reasoning:
        prompts_list = [f"{sys_block}{sep}User: {q}{sep}Reasoning:" if sys_block
                        else f"User: {q}{sep}Reasoning:"
                        for _, q, _ in BENCHMARKS]
    else:
        prompts_list = [f"{sys_block}{sep}User: {q}{sep}Response:" if sys_block
                        else f"User: {q}{sep}Response:"
                        for _, q, _ in BENCHMARKS]

    # Generate the whole suite in one batched, KV-cached pass.
    responses = []
    t0 = time.perf_counter()
    for i in range(0, len(prompts_list), batch_size):
        chunk = prompts_list[i:i + batch_size]
        responses.extend(generate_batch(
            model, tokenizer, chunk,
            max_new_tokens=max_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=1.1,
            stop_str=None if reasoning else "\n\n",
            seed=seed + i,
        ))
    elapsed = time.perf_counter() - t0

    results = []
    by_category = {}
    for (cat, question, keywords), resp in zip(BENCHMARKS, responses):
        resp = (resp or "").strip()
        s = score_response(resp, keywords)
        if reasoning:
            s.update(score_reasoning(resp))
        results.append(s)
        by_category.setdefault(cat, []).append(s)

        if verbose:
            chain = ""
            if reasoning and s.get("well_formed"):
                chain = f"  chain={s['chain']:.1f}"
            print(f"  [{cat:<12}] {question[:34]:<36} {s['total']:.1f}/3{chain}")
            print(f"      -> {resp[:110]!r}")

    n = len(results)
    total = sum(r["total"] for r in results)
    max_total = n * 3
    summary = {
        "model": model_path,
        "name": os.path.basename(model_path).replace(".pth", ""),
        "params": sum(p.numel() for p in model.parameters()),
        "context": model.cfg.block_size,
        "score": round(total, 2),
        "max": max_total,
        "pct": round(100 * total / max_total, 1),
        "elapsed_s": round(elapsed, 1),
        "by_category": {k: round(sum(r["total"] for r in v), 2)
                        for k, v in by_category.items()},
    }
    if reasoning:
        summary["chain_pct"] = round(
            100 * sum(r.get("chain", 0.0) for r in results) / n, 1)
    if val_corpus:
        summary.update(corpus_loss(model, tokenizer, val_corpus,
                                   block_size=model.cfg.block_size))

    if verbose:
        print(f"\n  score {summary['score']}/{max_total} ({summary['pct']}%)  "
              f"{elapsed:.1f}s for {n} prompts")
        for cat, s in summary["by_category"].items():
            k = sum(1 for c, _, _ in BENCHMARKS if c == cat)
            print(f"    {cat:<12} {s:5.1f}/{k*3}")
        if "chain_pct" in summary:
            print(f"    reasoning chains well-formed: {summary['chain_pct']}%")
        if "val_loss" in summary:
            print(f"    val_loss {summary['val_loss']:.4f}  "
                  f"ppl {summary['perplexity']:.2f}")
    return summary


def compare(summaries):
    print(f"\n{'='*66}")
    print("  COMPARISON")
    print(f"{'='*66}")
    print(f"  {'model':<34} {'score':>10} {'pct':>7} {'val loss':>10} {'ppl':>8}")
    print(f"  {'-'*34} {'-'*10} {'-'*7} {'-'*10} {'-'*8}")
    for s in summaries:
        vl = f"{s['val_loss']:.4f}" if "val_loss" in s else "-"
        ppl = f"{s['perplexity']:.1f}" if "perplexity" in s else "-"
        print(f"  {s['name'][:33]:<34} {s['score']:>6.1f}/{s['max']:<3} "
              f"{s['pct']:>6.1f}% {vl:>10} {ppl:>8}")
    print(f"{'='*66}")

    best = max(summaries, key=lambda s: s["pct"])
    print(f"  best keyword score : {best['name']} ({best['pct']}%)")
    with_val = [s for s in summaries if "val_loss" in s]
    if len(with_val) >= 2:
        lowest = min(with_val, key=lambda s: s["val_loss"])
        print(f"  lowest held-out loss: {lowest['name']} "
              f"({lowest['val_loss']:.4f}) -- this is the honest winner")
        base = with_val[0]
        for s in with_val[1:]:
            delta = s["val_loss"] - base["val_loss"]
            verdict = "FORGOT" if delta > 0.05 else "kept"
            print(f"    {s['name'][:30]:<32} vs {base['name'][:20]:<22} "
                  f"{delta:+.4f}  {verdict}")
    print(f"{'='*66}")


def main():
    p = argparse.ArgumentParser(
        prog="python benchmark.py",
        description="Benchmark Shimba LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", action="append", default=[], required=True,
                   help="Model .pth to benchmark (repeat for several)")
    p.add_argument("--compare", action="append", default=[],
                   help="Alias for extra --model")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=90)
    p.add_argument("--batch_size", type=int, default=10,
                   help="Prompts decoded together (lower = less RAM)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--system", default="none", metavar="NAME",
                   help="System-prompt template:\n" + prompts.describe())
    p.add_argument("--reasoning", action="store_true",
                   help="Prompt 'Reasoning:' and score the chain separately")
    p.add_argument("--val_corpus", default=None,
                   help="Held-out .txt for cross-entropy / perplexity")
    p.add_argument("--save", default=None, help="Write results as JSON")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                   help="Inference device (default: cpu; auto = cuda when available)")
    args = p.parse_args()

    if args.system.lower() not in prompts.TEMPLATES:
        print(f"[error] unknown system template '{args.system}'")
        print(prompts.describe())
        sys.exit(1)

    models = args.model + args.compare
    print(f"\n{'='*66}")
    print("  Shimba LLM Benchmark")
    print(f"  models : {len(models)}   temp {args.temp}   top_k {args.top_k}"
          f"   system '{args.system}'"
          f"{'   reasoning' if args.reasoning else ''}")
    print(f"{'='*66}")

    summaries = []
    for m in models:
        print(f"\n--- {m} " + "-" * max(0, 58 - len(m)))
        s = run_benchmark(m, temperature=args.temp, top_k=args.top_k,
                          top_p=args.top_p, max_tokens=args.max_tokens,
                          system=args.system.lower(), reasoning=args.reasoning,
                          batch_size=args.batch_size, seed=args.seed,
                          val_corpus=args.val_corpus,
                          verbose=not args.quiet, device=args.device)
        if s:
            summaries.append(s)

    if len(summaries) > 1:
        compare(summaries)
    elif summaries and args.val_corpus:
        s = summaries[0]
        print(f"\n  val_loss {s.get('val_loss', float('nan')):.4f}  "
              f"perplexity {s.get('perplexity', float('nan')):.2f}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"\n  Results saved -> {args.save}")


if __name__ == "__main__":
    main()