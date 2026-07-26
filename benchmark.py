#!/usr/bin/env python3
"""
benchmark.py — Evaluate a Shimba LLM model on a standard set of prompts.

Usage:
    python benchmark.py --model shimba-10m.pth
    python benchmark.py --model shimba-10m.pth --compare bomenater-bomb.pth
    python benchmark.py --model shimba-10m.pth --save results.txt

Scoring:
    Each response is auto-scored 0-3 on:
      - Coherence   : is it readable English?
      - Relevance   : does it address the prompt?
      - Repetition  : does it loop or stutter?
    Max score: 60 points (20 questions × 3 points)
"""

import os
import sys
import re
import time
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_default_device("cpu")

# ---------------------------------------------------------------------------
# Benchmark prompts
# ---------------------------------------------------------------------------

BENCHMARKS = [
    # Category, Prompt, Keywords expected in good response
    ("Identity",     "User: who are you\nResponse:",
     ["ai", "assistant", "bomenater", "shimba", "help"]),

    ("Identity",     "User: what is your name\nResponse:",
     ["name", "ai", "bomenater", "shimba", "called"]),

    ("Identity",     "User: what can you do\nResponse:",
     ["help", "answer", "question", "assist", "information"]),

    ("Knowledge",    "User: what is gravity\nResponse:",
     ["force", "mass", "earth", "attract", "pull", "weight"]),

    ("Knowledge",    "User: what is the speed of light\nResponse:",
     ["light", "speed", "metres", "miles", "fast", "second"]),

    ("Knowledge",    "User: what is photosynthesis\nResponse:",
     ["plant", "light", "sun", "energy", "oxygen", "carbon"]),

    ("Knowledge",    "User: what is DNA\nResponse:",
     ["genetic", "gene", "molecule", "cell", "biology", "double"]),

    ("Knowledge",    "User: who was albert einstein\nResponse:",
     ["physicist", "relativity", "theory", "german", "scientist"]),

    ("Knowledge",    "User: what is machine learning\nResponse:",
     ["learn", "data", "ai", "model", "train", "pattern"]),

    ("Knowledge",    "User: what is python\nResponse:",
     ["programming", "language", "code", "software", "script"]),

    ("Math",         "User: what is 2 plus 2\nResponse:",
     ["4", "four"]),

    ("Math",         "User: what is the square root of 144\nResponse:",
     ["12", "twelve"]),

    ("Math",         "User: what is pi\nResponse:",
     ["3.14", "circle", "ratio", "math", "constant"]),

    ("Conversation", "User: hi\nResponse:",
     ["hello", "hi", "hey", "welcome", "help", "assist"]),

    ("Conversation", "User: good morning\nResponse:",
     ["morning", "day", "great", "good", "help"]),

    ("Conversation", "User: thanks\nResponse:",
     ["welcome", "glad", "help", "anytime", "happy"]),

    ("Conversation", "User: goodbye\nResponse:",
     ["bye", "goodbye", "see", "take", "return", "anytime"]),

    ("Instruction",  "User: tell me a joke\nResponse:",
     ["why", "what", "joke", "laugh", "funny", "!"]),

    ("Instruction",  "User: help me write something\nResponse:",
     ["write", "help", "draft", "what", "topic", "like"]),

    ("Instruction",  "User: what should I eat for energy\nResponse:",
     ["eat", "food", "energy", "protein", "carb", "fruit",
      "vegetable", "meal"]),
]


# ---------------------------------------------------------------------------
# Auto-scorer
# ---------------------------------------------------------------------------

def score_response(response: str, keywords: list) -> dict:
    """
    Score a response 0-3 on coherence, relevance, repetition.
    Returns dict with individual scores and total.
    """
    resp_lower = response.lower()

    # 1. Coherence — is it readable? Check word/char ratio
    words = response.split()
    if len(words) == 0:
        coherence = 0
    else:
        avg_word_len = sum(len(w) for w in words) / len(words)
        # Good English words average 4-8 chars
        if 3 <= avg_word_len <= 10 and len(words) >= 3:
            coherence = 1
        else:
            coherence = 0

        # Bonus: contains punctuation and capital letters
        has_punct = any(c in response for c in ".!?,")
        has_caps  = any(c.isupper() for c in response[1:])
        if has_punct and has_caps:
            coherence = min(coherence + 1, 1)

    # 2. Relevance — does it contain expected keywords?
    hits = sum(1 for kw in keywords if kw in resp_lower)
    if hits >= 3:
        relevance = 1
    elif hits >= 1:
        relevance = 0.5
    else:
        relevance = 0

    # 3. Repetition — check for repeated substrings
    rep_score = 1
    # Check for character-level repetition (atatatatat)
    for length in [2, 3, 4]:
        for i in range(len(response) - length * 3):
            chunk = response[i:i+length]
            if response[i+length:i+length*4] == chunk * 3:
                rep_score = 0
                break

    # Check for word-level repetition
    if len(words) > 6:
        word_set = set(words)
        if len(word_set) / len(words) < 0.4:  # >60% repeated words
            rep_score = 0

    total = coherence + relevance + rep_score
    return {
        "coherence":  coherence,
        "relevance":  relevance,
        "repetition": rep_score,
        "total":      total,
        "max":        3,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_response(model, tokenizer, prompt: str,
                       max_tokens: int = 80,
                       temperature: float = 0.7,
                       top_k: int = 40) -> str:
    model.eval()
    block_size = model.cfg.block_size
    ids = tokenizer.encode(prompt)
    if not ids:
        return ""

    idx = torch.tensor([ids], dtype=torch.int64)

    generated = []
    for _ in range(max_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :].float() / temperature

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            thresh = torch.topk(logits, k).values[:, -1:]
            logits = logits.masked_fill(logits < thresh, float("-inf"))

        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        tok     = next_id.item()
        generated.append(tok)
        idx = torch.cat([idx, next_id], dim=1)

        # Stop at double newline
        if len(generated) >= 2:
            last2 = tokenizer.decode(generated[-2:])
            if last2 == "\n\n":
                break

    full = tokenizer.decode(ids + generated)
    # Return only the response part
    if "\nResponse:" in prompt:
        return full[len(prompt):].strip()
    return full[len(prompt):].strip()


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

def run_benchmark(model_path: str, temperature: float = 0.7,
                  top_k: int = 40, save_path: str = None):
    from llm.model import GPT
    from llm.tokenizer import CharTokenizer

    print(f"\n{'='*62}")
    print(f"  Shimba LLM Benchmark")
    print(f"  Model : {model_path}")
    print(f"  Temp  : {temperature}  Top-k: {top_k}")
    print(f"{'='*62}\n")

    # Load
    tok_path = CharTokenizer.default_path(model_path)
    if not os.path.exists(tok_path):
        print(f"[error] Tokenizer not found: {tok_path}")
        sys.exit(1)

    tokenizer = CharTokenizer.load(tok_path)
    model     = GPT.load(model_path)
    model.eval()

    results   = []
    by_category = {}
    output_lines = []

    def log(line=""):
        print(line)
        output_lines.append(line)

    for i, (category, prompt, keywords) in enumerate(BENCHMARKS, 1):
        # Generate
        t0       = time.perf_counter()
        response = generate_response(model, tokenizer, prompt,
                                     temperature=temperature, top_k=top_k)
        elapsed  = time.perf_counter() - t0

        # Score
        scores   = score_response(response, keywords)
        results.append(scores["total"])

        # Display
        question = prompt.split("User: ")[1].split("\n")[0]
        log(f"[{i:02d}] {category:<12} | Q: {question}")
        log(f"       Response : {response[:120]}")
        log(f"       Score    : {scores['total']:.1f}/3  "
            f"(coherence={scores['coherence']:.1f}  "
            f"relevance={scores['relevance']:.1f}  "
            f"no-repeat={scores['repetition']:.1f})  "
            f"[{elapsed:.1f}s]")
        log()

        # Track by category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(scores["total"])

    # Summary
    total     = sum(results)
    max_total = len(BENCHMARKS) * 3
    pct       = total / max_total * 100

    log("=" * 62)
    log(f"  RESULTS: {total:.1f} / {max_total}  ({pct:.0f}%)")
    log("=" * 62)
    log()
    log("  By category:")
    for cat, scores in by_category.items():
        cat_total = sum(scores)
        cat_max   = len(scores) * 3
        cat_pct   = cat_total / cat_max * 100
        bar_len   = int(cat_pct / 10)
        bar       = "█" * bar_len + "░" * (10 - bar_len)
        log(f"  {cat:<12} [{bar}] {cat_total:.1f}/{cat_max}  ({cat_pct:.0f}%)")
    log()

    # Grade
    if pct >= 80:
        grade = "A — Excellent"
    elif pct >= 65:
        grade = "B — Good"
    elif pct >= 50:
        grade = "C — Fair"
    elif pct >= 35:
        grade = "D — Poor"
    else:
        grade = "F — Needs more training"

    log(f"  Grade: {grade}")
    log("=" * 62)

    if save_path:
        with open(save_path, "w") as f:
            f.write(f"Model: {model_path}\n")
            f.write("\n".join(output_lines))
        print(f"\n  Results saved → {save_path}")

    return total, max_total


# ---------------------------------------------------------------------------
# Compare two models
# ---------------------------------------------------------------------------

def compare_models(model_a: str, model_b: str, temperature: float, top_k: int):
    print(f"\nComparing:")
    print(f"  A: {model_a}")
    print(f"  B: {model_b}\n")

    score_a, max_a = run_benchmark(model_a, temperature, top_k)
    score_b, max_b = run_benchmark(model_b, temperature, top_k)

    print(f"\n{'='*62}")
    print(f"  COMPARISON")
    print(f"  {model_a:<30} : {score_a:.1f}/{max_a} ({score_a/max_a*100:.0f}%)")
    print(f"  {model_b:<30} : {score_b:.1f}/{max_b} ({score_b/max_b*100:.0f}%)")
    winner = model_a if score_a >= score_b else model_b
    print(f"  Winner: {winner}")
    print(f"{'='*62}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="python benchmark.py",
        description="Benchmark a Shimba LLM model"
    )
    parser.add_argument("--model",    required=True,
                        help="Model .pth file to benchmark")
    parser.add_argument("--compare",  default=None,
                        help="Second model to compare against")
    parser.add_argument("--temp",     type=float, default=0.7)
    parser.add_argument("--top_k",    type=int,   default=40)
    parser.add_argument("--save",     default=None,
                        help="Save results to text file")

    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExamples:")
        print("  python benchmark.py --model shimba-10m.pth")
        print("  python benchmark.py --model shimba-10m.pth --compare bomenater-bomb.pth")
        print("  python benchmark.py --model shimba-10m.pth --save results.txt")
        sys.exit(0)

    args = parser.parse_args()

    if args.compare:
        compare_models(args.model, args.compare, args.temp, args.top_k)
    else:
        run_benchmark(args.model, args.temp, args.top_k, args.save)


if __name__ == "__main__":
    main()
