#!/usr/bin/env python3
"""
gen_spike_data.py -- Build spike-data.txt: Thinking/Response turns only.

- Pool: every chat-format corpus. Exact-dedup on (user, response).
- flafi-dataset.txt turns are kept VERBATIM (real reasoning, never restamped).
- Turns without Thinking: get a short stamped line in flafi style:
      Thinking: they ask <q>. answer with <a>.
  (shape, not real reasoning -- honest limitation, see README note below)
- ChatModelDataset's "Assistant :" is normalized to "Response :"; answers
  are collapsed to one line because continuation lines carry no loss
  (build_chat_loss_mask only scores lines starting with the prefix).
- Drops: empty sides, user > 200 chars, response > 500 chars (a block-256
  char model cannot learn 800-char essays; they become loss noise),
  wordlists/prose (no Thinking/Response lines -- inert under thinking loss).

Usage:
  python3 gen_spike_data.py --out spike-data.txt --seed 7
"""

import argparse
import random
import re

USER_CAP = 200
RESP_CAP = 500

# Every self-identification in the pool corpora (Bomenater, omni fast, flafi)
# is rewritten to Spike. Spike is the model; the rest are other people's names.
NAME_PAT = re.compile(r"bomenater(?:\s+ai)?|omni\s*fast(?:\s+ai)?|flafi",
                      flags=re.IGNORECASE)


def spike_name(m: re.Match) -> str:
    return "spike" if m.group(0).islower() else "Spike"


def normalize_response(resp: str) -> str:
    resp = NAME_PAT.sub(spike_name, resp)
    # "call me Spike, or just Spike" -> "call me Spike" (after substitution)
    resp = re.sub(r"\bspike, or just spike\b", "Spike",
                  resp, flags=re.IGNORECASE)
    resp = re.sub(r"^i am an ai\.?$", "I am Spike.",
                  resp.strip(), flags=re.IGNORECASE)
    return resp


NAME_Q = ("your name", "who are you", "who r u", "what are you",
          "introduce yourself", "what should i call you", "alias")
MAKER_Q = ("who made you", "who created you", "who built you",
           "who trained you", "who trained")


def canonical(user: str):
    """(thinking, response) for identity/maker questions, else None."""
    q = norm(user).lower()
    if any(k in q for k in MAKER_Q):
        return ("they ask who made me. shimba trained me. say that.",
                "Shimba trained me. I am Spike.")
    if any(k in q for k in NAME_Q):
        return ("they ask who i am. that is identity. answer with my name.",
                "I am Spike, a small AI trained by Shimba.")
    return None

# chat files sharing the "User:\nResponse:" block shape
STANDARD = [
    "mega.txt",
    "deepseek_dataset.txt",
    "large_chat_dataset.txt",
    "chat_corpus_clean.txt",
    "chat_corpus.txt",
    "bomenater_50k_dataset.txt",
    "deepslop.txt",
    "deepslop_combined.txt",
    "deepslop_v2.txt",
    "flafi-anonomuse.txt",
]

CHATMODEL = "ChatModelDataset.txt"   # "User :" / "Assistant :" + multi-paragraph
FLAFI_VERBATIM = "flafi-dataset.txt"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def stamp(user: str, resp: str) -> str:
    q = norm(user).lower()[:60]
    a = norm(resp).lower()[:60]
    return f"they ask {q}. answer with {a}."


def parse_standard(text: str):
    """Yield (user, thinking-or-None, response) from User:/Response: blocks."""
    for block in text.replace("\r\n", "\n").split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        if not lines[0].startswith("User:"):
            continue
        user = lines[0][len("User:"):].strip()
        think = None
        rest = lines[1:]
        if rest and rest[0].startswith("Thinking:"):
            think = rest[0][len("Thinking:"):].strip()
            rest = rest[1:]
        if not rest or not rest[0].startswith("Response:"):
            continue
        resp = rest[0][len("Response:"):].strip()
        if len(rest) > 1:  # collapse continuation lines (they carry no loss)
            resp = norm(resp + " " + " ".join(rest[1:]))
        yield user, think, resp


def parse_chatmodel(text: str):
    """Yield (user, None, response) from User :/Assistant : pairs."""
    pat = re.compile(r"User\s*:\s*(.*?)\nAssistant\s*:\s*(.*?)(?=\n\nUser\s*:|\Z)",
                     re.DOTALL)
    for m in pat.finditer(text.replace("\r\n", "\n")):
        yield m.group(1).strip(), None, norm(m.group(2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="spike-data.txt")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    seen = set()
    turns = []  # (rendered_block, has_thinking)
    n_stamp = n_flafi = n_drop = 0

    def add(user, think, resp, verbatim=False):
        nonlocal n_stamp, n_drop
        user = NAME_PAT.sub(spike_name, norm(user))
        resp = normalize_response(norm(resp))
        if not user or not resp:
            n_drop += 1
            return
        if len(user) > USER_CAP or len(resp) > RESP_CAP:
            n_drop += 1
            return
        is_stamp = False
        if verbatim:
            think = normalize_response(norm(think)) if think else stamp(user, resp)
        else:
            hit = canonical(user)
            if hit is not None:
                think, resp = hit
            elif think is None:
                think = stamp(user, resp)
                is_stamp = True
            else:
                think = norm(think)
        key = (user, resp)
        if key in seen:
            return
        seen.add(key)
        if is_stamp:
            n_stamp += 1
        turns.append(
            (f"User: {user}\nThinking: {norm(think)}\nResponse: {resp}",
             True))

    import os
    for path in STANDARD:
        if not os.path.exists(path):
            print(f"[skip] {path} missing")
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        n0 = len(turns)
        for u, t, r in parse_standard(text):
            add(u, t, r)
        del text
        print(f"[pool] {path}: +{len(turns) - n0} turns")

    if os.path.exists(CHATMODEL):
        with open(CHATMODEL, encoding="utf-8", errors="replace") as f:
            text = f.read()
        n0 = len(turns)
        for u, t, r in parse_chatmodel(text):
            add(u, t, r)
        del text
        print(f"[pool] {CHATMODEL}: +{len(turns) - n0} turns")

    if os.path.exists(FLAFI_VERBATIM):
        with open(FLAFI_VERBATIM, encoding="utf-8") as f:
            text = f.read()
        n0 = len(turns)
        for u, t, r in parse_standard(text):
            if t is not None:
                n_flafi += 1
            add(u, t, r, verbatim=True)
        del text
        print(f"[pool] {FLAFI_VERBATIM}: +{len(turns) - n0} turns ({n_flafi} real-thinking)")

    r = random.Random(args.seed)
    r.shuffle(turns)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(b for b, _ in turns) + "\n")

    import os as _os
    print(f"[gen] wrote {args.out} ({_os.path.getsize(args.out):,} bytes, "
          f"{len(turns)} turns, {n_flafi} real-thinking, {n_stamp} stamped, "
          f"{n_drop} dropped)")


if __name__ == "__main__":
    main()
