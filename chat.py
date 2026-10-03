#!/usr/bin/env python3
"""
chat.py -- Interactive chat with a trained Shimba LLM.

Examples
--------
  # quick model, no system prompt
  python chat.py --model models/fast.pth

  # pick a system-prompt template (must match what the model was trained with)
  python chat.py --model models/expert.pth --system expert

  # thinking model: shows the Reasoning: chain then the Response:
  python chat.py --model models/reasoner.pth --system reasoner --reasoning

  # see which templates exist
  python chat.py --list-systems

Commands
--------
  /reset              clear the conversation
  /system [name]      show or switch the system-prompt template
  /systems            list every template
  /reasoning [on|off] toggle reasoning mode mid-chat
  /temp <value>       change sampling temperature
  /info               show model, template and settings
  /exit               quit
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.model import GPT
from llm.tokenizer import CharTokenizer
from llm.generate import generate_stream
import prompts


USER_PREFIX     = "User:"
REASON_PREFIX   = "Reasoning:"
THINK_PREFIX    = "Thinking:"
RESPONSE_PREFIX = "Response:"
TURN_SEP        = "\n\n"


class ChatSession:
    """
    Holds the conversation and renders it in exactly the shape of the
    training corpus, so inference matches what the model was fitted on.
    """

    def __init__(self, model, tokenizer, max_history_tokens=1024,
                 temperature=0.7, top_k=40, top_p=0.95, max_new_tokens=400,
                 system="none", reasoning=False, thinking=False,
                 repetition_penalty=1.1):
        self.model = model
        self.tokenizer = tokenizer
        self.max_history_tokens = max_history_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.system = system
        self.reasoning = reasoning
        self.thinking = thinking
        self.history = []

    # ------------------------------------------------------------------
    def _render(self, pending_user=None) -> str:
        return prompts.compose(self.system, self.history, pending_user,
                               reasoning=self.reasoning,
                               thinking=self.thinking)

    def _trim(self, pending_user=None) -> str:
        """
        Drop whole turns (never raw tokens) until the prompt fits, so turn
        and system boundaries stay intact. The pending user turn is rendered
        but never dropped -- it is not in history yet.
        """
        if len(self.history) <= 1:
            return self._render(pending_user)
        while len(self.history) > 1:
            if len(self.tokenizer.encode(self._render(pending_user))) <= self.max_history_tokens:
                break
            self.history = self.history[1:]
        return self._render(pending_user)

    # ------------------------------------------------------------------
    def stream_response(self, user_input: str):
        """Yield the assistant reply as it is produced."""
        user_text = user_input.strip()
        prompt = self._trim(pending_user=user_text)

        parts = []
        for piece in generate_stream(
            self.model, self.tokenizer, prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            # In reasoning/thinking mode the model must run past the
            # chain to reach the Response: line, so the turn separator
            # is not a stop token.
            stop_str=None if (self.reasoning or self.thinking) else TURN_SEP,
        ):
            parts.append(piece)
            yield piece

        raw = "".join(parts).strip()
        self.history.append(("user", user_text))
        self.history.append(("assistant", raw))

    def get_response(self, user_input: str) -> str:
        return "".join(self.stream_response(user_input))

    def stream_thinking(self, user_input: str):
        """
        Stream a turn live, showing the chain as it is written.

        Yields ('thinking', text) while the model is inside the
        Thinking: block and ('response', text) once it reaches the
        Response: line. With no --thinking the model picks the structure
        itself, so the caller sees whichever it emits; a model that never
        writes Thinking: simply yields only ('response', ...).

        Uses a carriage return to rewrite the chain line in place, so the
        thought updates as it grows instead of scrolling forever.

        Yields incremental deltas: ('thinking', new_chars) while inside the
        chain, ('response', new_chars) once past Response:. The caller
        accumulates; newlines in thinking deltas are the caller's to strip
        so one rewrite stays on one line.
        """
        user_text = user_input.strip()
        prompt = self._trim(pending_user=user_text)

        # When --thinking is forced the prompt already ends with "Thinking:",
        # so the continuation IS the chain -- we start inside it. Otherwise
        # we wait for the model to emit a prefix before labelling anything,
        # so a bare "Response: ..." is never mislabelled as thinking.
        buf = ""
        mode = "thinking" if self.thinking else None
        emitted_think = 0
        emitted_resp = 0
        for piece in generate_stream(
            self.model, self.tokenizer, prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            # Must run past the chain to reach Response:, so the turn
            # separator is not a stop token here.
            stop_str=None,
        ):
            buf += piece

            # Did we cross into the answer?
            if RESPONSE_PREFIX in buf:
                head, _, tail = buf.partition(RESPONSE_PREFIX)
                if mode != "response":
                    mode = "response"
                    yield ("switch", head.strip())
                    emitted_resp = 0
                delta = tail[emitted_resp:]
                emitted_resp = len(tail)
                if delta:
                    yield ("response", delta)
                continue

            if THINK_PREFIX in buf and mode is None:
                mode = "thinking"
                head, _, tail = buf.partition(THINK_PREFIX)
                if head.strip():
                    yield ("response", head)
                emitted_think = 0
                if tail:
                    emitted_think = len(tail)
                    yield ("thinking", tail)
                continue

            if mode == "thinking":
                # Forced mode: buf holds only chain content (the prompt's
                # "Thinking:" is not part of buf). Auto mode: strip a
                # re-emitted prefix if the model repeats it.
                if THINK_PREFIX in buf:
                    _, _, content = buf.partition(THINK_PREFIX)
                else:
                    content = buf
                delta = content[emitted_think:]
                emitted_think = len(content)
                if delta:
                    yield ("thinking", delta)
                continue

            if mode is None:
                # No prefix yet and not forced: hold output until we know
                # which block this is. If the model never emits a prefix,
                # flush as plain response once enough chars accumulate.
                if len(buf) >= 20:
                    mode = "response"
                    head, _, tail = buf.partition(RESPONSE_PREFIX)
                    start = tail if RESPONSE_PREFIX in buf else buf
                    emitted_resp = len(start)
                    yield ("response", start)
                continue

            # Plain response tail with no explicit Response: line.
            delta = buf[emitted_resp:]
            emitted_resp = len(buf)
            if delta:
                yield ("response", delta)

        raw = buf.strip()
        self.history.append(("user", user_text))
        self.history.append(("assistant", raw))

    def reset(self) -> None:
        self.history = []


# ---------------------------------------------------------------------------

def load_model(path: str, device: str = "cpu"):
    """Load a checkpoint written by either the current or the legacy saver."""
    if not os.path.exists(path):
        print(f"[error] Model file not found: {path}")
        sys.exit(1)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA requested but not available -- using cpu")
        device = "cpu"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "config" not in checkpoint:
        print("[error] Checkpoint missing 'config' key")
        sys.exit(1)

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model")
    if state_dict is None:
        print("[error] Checkpoint has no 'state_dict' or 'model' key")
        sys.exit(1)

    model = GPT(checkpoint["config"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[chat] loaded with {len(missing)} missing / "
              f"{len(unexpected)} unexpected tensors")
    model.to(device)
    model.eval()
    if device.startswith("cuda"):
        print(f"[chat] using GPU: {torch.cuda.get_device_name(0)}")
    return model


def split_reasoning(text: str):
    """Split a raw reply into (reasoning, response); either may be ''."""
    if REASON_PREFIX not in text and RESPONSE_PREFIX not in text:
        return "", text.strip()
    reasoning, _, response = text.partition(RESPONSE_PREFIX)
    reasoning = reasoning.replace(REASON_PREFIX, "", 1).strip()
    return reasoning, response.strip()


def split_thinking(text: str):
    """Split a raw reply into (thinking, response); either may be ''."""
    if THINK_PREFIX not in text and RESPONSE_PREFIX not in text:
        return "", text.strip()
    thinking, _, response = text.partition(RESPONSE_PREFIX)
    thinking = thinking.replace(THINK_PREFIX, "", 1).strip()
    # No Response: line yet (model still thinking when cut off) -- the
    # whole thing is chain.
    if not response.strip() and THINK_PREFIX in text:
        return thinking, ""
    return thinking, response.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with a Shimba LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=False,
                        help="Path to model .pth file")
    parser.add_argument("--system", default="none", metavar="NAME",
                        help="System-prompt template. One of:\n" +
                             prompts.describe())
    parser.add_argument("--list-systems", action="store_true",
                        help="List the system-prompt templates and exit")
    parser.add_argument("--reasoning", action="store_true",
                        help="Reasoning mode: prompt 'Reasoning:' and show "
                             "the chain before the answer")
    parser.add_argument("--thinking", action="store_true",
                        help="Force 'Thinking:' and stream the chain live as "
                             "it is written. Omit it and a Thinking-trained "
                             "model still thinks on its own -- the turn is "
                             "left open so the model picks the structure.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="1.0 disables")
    parser.add_argument("--max_tokens", type=int, default=400,
                        help="Max new tokens per response")
    parser.add_argument("--max_history", type=int, default=1024)
    parser.add_argument("--no_stream", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                        help="Inference device (default: cpu; auto = cuda when available)")
    args = parser.parse_args()

    if args.list_systems:
        print("System-prompt templates:\n")
        print(prompts.describe())
        return

    if not args.model:
        parser.error("--model is required (or use --list-systems)")
    if args.system.lower() not in prompts.TEMPLATES:
        print(f"[error] unknown system template '{args.system}'. Choose from:")
        print(prompts.describe())
        sys.exit(1)

    tok_path = CharTokenizer.default_path(args.model)
    if not os.path.exists(tok_path):
        print(f"[error] Tokenizer not found at {tok_path}")
        sys.exit(1)
    tokenizer = CharTokenizer.load(tok_path)

    print(f"[chat] Loading model from {args.model} ...")
    model = load_model(args.model, device=args.device)

    chat = ChatSession(
        model, tokenizer,
        max_history_tokens=args.max_history,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens,
        system=args.system.lower(),
        reasoning=args.reasoning,
        thinking=args.thinking,
        repetition_penalty=args.repetition_penalty,
    )

    print("\n" + "=" * 62)
    print("  Shimba LLM -- Interactive Chat")
    print(f"  Model     : {args.model}")
    print(f"  System    : {args.system.lower()}")
    print(f"  Reasoning : {'on' if args.reasoning else 'off'}")
    print(f"  Thinking  : {'on (forced)' if args.thinking else 'auto'}")
    print(f"  Temp {args.temperature}  top_k {args.top_k}  top_p {args.top_p}")
    print("  Commands: /reset /system [name] /systems /reasoning [on|off]")
    print("            /thinking [on|off] /temp <v> /info /exit")
    print("=" * 62 + "\n")

    while True:
        try:
            raw = input(">>> ").strip()
            if not raw:
                continue

            # ── Slash commands ──────────────────────────────────────
            if raw.startswith("/"):
                cmd, _, arg = raw.partition(" ")
                cmd = cmd.lower()
                arg = arg.strip()

                if cmd in ("/exit", "/quit"):
                    print("Goodbye!")
                    break
                if cmd == "/reset":
                    chat.reset()
                    print("[Conversation reset]")
                elif cmd == "/systems":
                    print(prompts.describe())
                elif cmd == "/system":
                    if not arg:
                        print(f"[system] {chat.system} -> {prompts.get(chat.system)!r}")
                    elif arg.lower() in prompts.TEMPLATES:
                        chat.system = arg.lower()
                        chat.reset()
                        print(f"[system] switched to '{chat.system}'. History cleared.")
                    else:
                        print(f"[error] unknown template '{arg}'. Try /systems")
                elif cmd == "/reasoning":
                    if not arg:
                        state = "on" if chat.reasoning else "off"
                        print(f"[reasoning] {state}")
                    else:
                        chat.reasoning = arg.lower() in ("on", "1", "true", "yes")
                        chat.reset()
                        print(f"[reasoning] {'on' if chat.reasoning else 'off'}. "
                              "History cleared.")
                elif cmd == "/thinking":
                    if not arg:
                        state = "on" if chat.thinking else "auto"
                        print(f"[thinking] {state}")
                    else:
                        chat.thinking = arg.lower() in ("on", "1", "true", "yes")
                        chat.reset()
                        print(f"[thinking] "
                              f"{'on (forced)' if chat.thinking else 'auto'}. "
                              "History cleared.")
                elif cmd == "/temp":
                    try:
                        chat.temperature = float(arg)
                        print(f"[temp] {chat.temperature}")
                    except ValueError:
                        print("[error] usage: /temp 0.7")
                elif cmd == "/info":
                    print(f"  model     : {args.model}")
                    print(f"  params    : {sum(p.numel() for p in model.parameters()):,}")
                    print(f"  context   : {model.cfg.block_size}")
                    print(f"  system    : {chat.system}")
                    print(f"  reasoning : {'on' if chat.reasoning else 'off'}")
                    print(f"  thinking  : {'on (forced)' if chat.thinking else 'auto'}")
                    print(f"  temp {chat.temperature}  top_k {chat.top_k}  "
                          f"top_p {chat.top_p}")
                else:
                    print(f"[error] unknown command '{cmd}'. Try /info")
                continue

            # ── Normal turn ─────────────────────────────────────────
            if args.no_stream:
                reply = chat.get_response(raw)
                chain, answer = split_thinking(reply)
                if chain or chat.thinking:
                    if chain:
                        print(f"\n[Thinking] {chain}")
                    if answer:
                        print(f"\n[Response] {answer}\n")
                    else:
                        print(f"\n[Response] {chain}\n"
                              "(note: no Response: line -- model likely "
                              "not trained on Thinking: data; try "
                              "without --thinking)\n")
                else:
                    print(f"\n{reply}\n")
                continue

            if chat.thinking:
                # Live-thinking display: the chain rewrites ONE line in place
                # as it grows, then the answer streams below it. Thinking
                # deltas are sanitized to a single line so embedded newlines
                # can never flood the terminal with repeats.
                sys.stdout.write("\n")
                think_shown = ""
                resp_shown = ""
                switched = False
                for kind, text in chat.stream_thinking(raw):
                    if kind == "switch":
                        # Chain done -- close the rewrite line, open answer.
                        sys.stdout.write("\r\033[K\n")
                        switched = True
                    elif kind == "thinking":
                        think_shown += text
                        single = think_shown.replace("\n", " ").replace("\r", " ")
                        sys.stdout.write("\r\033[K[Thinking] " + single[-120:])
                    else:
                        resp_shown += text
                        sys.stdout.write(text)
                    sys.stdout.flush()
                if not switched:
                    # Forced Thinking: on a model that never learned it often
                    # never emits Response: within budget -- otherwise this
                    # looks like "no response". Close the rewrite line and
                    # show what was generated as the answer.
                    sys.stdout.write("\r\033[K\n")
                    if think_shown.strip() and not resp_shown.strip():
                        sys.stdout.write("[Response] " + think_shown.strip() + "\n")
                        sys.stdout.write("(note: no Response: line -- model likely "
                                         "not trained on Thinking: data; try "
                                         "without --thinking)\n")
                sys.stdout.write("\n")
                sys.stdout.flush()
                continue

            if chat.reasoning:
                # Buffer the chain so it can be shown dimmed, then the answer.
                sys.stdout.write("\n")
                raw_text = chat.get_response(raw)
                chain, answer = split_reasoning(raw_text)
                if chain:
                    sys.stdout.write(f"[Reasoning] {chain}\n")
                sys.stdout.write(f"[Response] {answer}\n\n")
                sys.stdout.flush()
            else:
                sys.stdout.write("\n")
                for piece in chat.stream_response(raw):
                    sys.stdout.write(piece)
                    sys.stdout.flush()
                sys.stdout.write("\n\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[error] {e}")
            continue


if __name__ == "__main__":
    main()