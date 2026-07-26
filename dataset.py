"""
dataset.py — Convert chat exports into training data and fine-tune.

Usage:
  # 1. Export chats from the web UI (click 📦 button) → all_chats.txt
  # 2. Convert to training format:
  python dataset.py prepare all_chats.txt --out training_data.txt

  # 3. Train on it:
  python shimba.py train --data training_data.txt --out fine_tuned.pth --resume --lr 1e-4

  # 4. Or convert JSON exports:
  python dataset.py prepare chat_export.json --format json --out training_data.txt

  # 5. View stats about your dataset:
  python dataset.py stats training_data.txt
"""

import os, sys, json, re, math
from collections import Counter


def load_chat_txt(path: str):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    conversations = []
    blocks = text.strip().split("### ")
    for block in blocks:
        if not block.strip(): continue
        lines = block.strip().split("\n")
        title = lines[0].strip() if lines else "Chat"
        messages = []
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("User: "):
                messages.append(("user", line[6:]))
            elif line.startswith("Response: "):
                messages.append(("assistant", line[10:]))
        if messages: conversations.append({"title": title, "messages": messages})
    return conversations


def load_chat_json(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    messages = []
    for m in data:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant"):
            messages.append((role, content))
    return [{"title": os.path.basename(path), "messages": messages}]


def to_training_text(conversations):
    lines = []
    for conv in conversations:
        for role, text in conv["messages"]:
            if role == "user":
                lines.append(f"User: {text}")
            elif role == "assistant":
                lines.append(f"Response: {text}")
        lines.append("")  # blank line between conversations
    return "\n".join(lines)


def to_chat_format(conversations):
    """Format for chat fine-tuning: User: ...\nResponse: ...\nUser: ...\nResponse: ..."""
    return to_training_text(conversations)


def to_completion_format(conversations):
    """Format for completion training: just concatenated text."""
    lines = []
    for conv in conversations:
        for role, text in conv["messages"]:
            lines.append(text)
    return "\n".join(lines)


def dataset_stats(path: str):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.strip().split("\n")
    user_msgs = [l for l in lines if l.startswith("User: ")]
    resp_msgs = [l for l in lines if l.startswith("Response: ")]
    words = text.split()
    chars = len(text)
    avg_user = sum(len(l[6:].split()) for l in user_msgs) / max(len(user_msgs), 1)
    avg_resp = sum(len(l[10:].split()) for l in resp_msgs) / max(len(resp_msgs), 1)

    print(f"=== Dataset Stats ===")
    print(f"  File:           {path}")
    print(f"  Characters:     {chars:,}")
    print(f"  Words:          {len(words):,}")
    print(f"  Lines:          {len(lines):,}")
    print(f"  User messages:  {len(user_msgs)}")
    print(f"  Responses:      {len(resp_msgs)}")
    print(f"  Conversations:  {text.count('### ')}")
    print(f"  Avg user len:   {avg_user:.1f} words")
    print(f"  Avg resp len:   {avg_resp:.1f} words")
    print(f"  Est. tokens:    {chars // 3:,} (rough)")


def prepare(args):
    path = args.data
    fmt = args.format or ("json" if path.endswith(".json") else "txt")
    out = args.out or "training_data.txt"
    conversations = []

    if fmt == "json":
        conversations = load_chat_json(path)
    else:
        conversations = load_chat_txt(path)

    text = to_training_text(conversations)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Exported {len(conversations)} conversations -> {out}")
    dataset_stats(out)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Shimba dataset tools")
    sub = p.add_subparsers(dest="cmd")

    pa = sub.add_parser("prepare", help="Convert chat export to training data")
    pa.add_argument("data", help="Chat .txt or .json export")
    pa.add_argument("--format", choices=["txt", "json"], help="Input format (auto-detected)")
    pa.add_argument("--out", default="training_data.txt", help="Output .txt file")

    ps = sub.add_parser("stats", help="Show dataset statistics")
    ps.add_argument("data", help="Training .txt file")

    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "stats":
        dataset_stats(args.data)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
