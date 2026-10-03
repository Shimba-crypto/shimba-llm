#!/usr/bin/env python3
"""
setup.py -- Shimba LLM entry point.

Usage
-----
  python setup.py train    --data <file.txt>        --out <model.pth> [options]
  python setup.py train    --data <folder/>          --out <model.pth> [options]
  python setup.py finetune --model <model.pth>      --data <file.txt>  [options]
  python setup.py generate --prompt "..." --model <model.pth> [options]
  python setup.py help

When --data points to a FOLDER, all .txt files inside it are merged
(recursively) into one corpus before training.

Train options
-------------
  --data          Path to a .txt file OR a folder of .txt files (required)
  --out           Output model path (default: model.pth)
  --block_size    Context window length          (default: 512)
  --n_embd        Embedding dimension            (default: 256)
  --n_head        Number of attention heads      (default: 4)
  --n_layer       Number of transformer layers   (default: 4)
  --dropout       Dropout probability            (default: 0.1)
  --batch_size    Mini-batch size                (default: 8)
  --grad_accum    Gradient accumulation steps    (default: 4)
  --lr            Learning rate                  (default: 3e-4)
  --max_iters     Training iterations            (default: 5000)
  --eval_interval Eval every N iters             (default: 500)
  --eval_iters    Batches per eval               (default: 100)
  --weight_decay  AdamW weight decay             (default: 0.1)
  --val_frac      Fraction of data for validation (default: 0.1)
  --pattern       Glob pattern when --data is a folder (default: *.txt)
  --no_recurse    Do not recurse into subfolders (flag)

Generate options
----------------
  --prompt        Seed text for generation (required)
  --model         Path to saved model .pth file (required)
  --max_tokens    Tokens to generate        (default: 200)
  --temperature   Sampling temperature      (default: 0.8)
  --top_k         Top-k sampling            (default: 40)
  --top_p         Nucleus sampling p        (default: 0.95)
  --stream        Stream output token-by-token (flag)

Finetune options
----------------
  --model         Path to pre-trained model .pth file (required)
  --data          Path to a .txt file OR a folder (required)
  --out           Output model path (default: overwrites input model)
  --lr            Learning rate                  (default: 5e-5)
  --max_iters     Training iterations            (default: 2000)
  --response_only Only compute loss on assistant response tokens (flag)
  All other options same as 'train'.

Notes
-----
  * The tokenizer is saved alongside the model as <model_stem>_tokenizer.json
  * All computation runs on CPU -- no GPU required.
  * Memory usage is kept well below 6 GB for the default hyperparameters.
"""

import sys
import os
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch


def resolve_device(name: str) -> str:
    """Map --device {auto,cpu,cuda} to a concrete torch device string."""
    req = (name or "cpu").lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] CUDA requested but not available -- using cpu")
        return "cpu"
    return req


# ---------------------------------------------------------------------------
# Folder -> merged corpus loader
# ---------------------------------------------------------------------------

def load_data_path(data_path: str, pattern: str = "*.txt", recurse: bool = True) -> str:
    """
    Load text from:
      - a single .txt file, OR
      - a directory: merges all files matching `pattern` (optionally recursive)

    Returns the combined text string.
    """
    if os.path.isfile(data_path):
        return _read_file(data_path)

    if os.path.isdir(data_path):
        # Collect matching files
        if recurse:
            matches = []
            for root, dirs, files in os.walk(data_path):
                # Sort for deterministic ordering
                dirs.sort()
                for fname in sorted(files):
                    if _matches_pattern(fname, pattern):
                        matches.append(os.path.join(root, fname))
        else:
            matches = sorted(glob.glob(os.path.join(data_path, pattern)))

        if not matches:
            print(f"[error] No files matching '{pattern}' found in '{data_path}'")
            sys.exit(1)

        print(f"[data] found {len(matches)} file(s) in '{data_path}'")

        # Read and concatenate with a separator so docs don't bleed together
        sep = "\n\n" + "=" * 60 + "\n\n"
        parts = []
        total_bytes = 0
        for i, fpath in enumerate(matches, 1):
            text = _read_file(fpath, silent=True)
            if text.strip():          # skip empty files
                parts.append(text)
                total_bytes += len(text)
            # Progress every 20 files
            if i % 20 == 0 or i == len(matches):
                print(f"  loaded {i}/{len(matches)} files  "
                      f"({total_bytes / 1_000_000:.1f} MB so far)")

        combined = sep.join(parts)
        print(f"[data] total corpus: {len(combined):,} characters "
              f"across {len(parts)} file(s)")
        return combined

    print(f"[error] --data path does not exist: '{data_path}'")
    sys.exit(1)


def _matches_pattern(filename: str, pattern: str) -> bool:
    """Simple glob-style match on filename only (not full path)."""
    import fnmatch
    return fnmatch.fnmatch(filename.lower(), pattern.lower())


def _read_file(path: str, silent: bool = False) -> str:
    """Read a text file with UTF-8 -> latin-1 fallback."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            text = f.read()
    if not silent:
        print(f"[data] loaded '{path}'  ({len(text):,} characters)")
    return text


# ---------------------------------------------------------------------------
# Sub-command: train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    from llm.model import GPT, GPTConfig
    from llm.tokenizer import CharTokenizer
    from llm.data import make_splits, build_chat_loss_mask
    from llm.train import Trainer, TrainConfig

    # 1. Load data (file or folder)
    text = load_data_path(args.data, pattern=args.pattern, recurse=not args.no_recurse)

    # Thinking: auto-detect.  Turns with Thinking:/Response: lines train
    # with loss on those lines only (User: lines masked out); turns
    # without them train normally.  A dataset may freely mix both.
    # --thinking asserts the corpus must contain Thinking: blocks, so a
    # thinking model is never silently trained on think-less data.
    has_thinking = "\nThinking:" in text
    if args.thinking and not has_thinking:
        print("[error] --thinking passed but the corpus contains no "
              "'Thinking:' lines. Without them this would train as a "
              "non-thinking model. Add Thinking: blocks (see gen_flafi.py) "
              "or drop --thinking.")
        sys.exit(1)
    loss_mask = None
    if has_thinking:
        print("[train] Thinking: blocks detected -- loss on "
              "Thinking:/Response: lines only")
        loss_mask = build_chat_loss_mask(
            text, loss_prefixes=("Thinking:", "Response:"))

    if len(text) < args.block_size * 2:
        print(f"[error] Corpus too short ({len(text):,} chars). "
              f"Need at least {args.block_size * 2:,} characters.")
        sys.exit(1)

    # Warn if corpus is very large (memory check)
    size_mb = len(text) / 1_000_000
    if size_mb > 500:
        print(f"[warn] Large corpus ({size_mb:.0f} MB). "
              f"Token tensor will use ~{size_mb * 2:.0f} MB RAM (int16). "
              f"Consider --block_size 256 or --batch_size 4 to save memory.")

    # 2. Build tokenizer from full corpus
    print("[train] building tokenizer ...")
    tokenizer = CharTokenizer().build(text)
    tok_path  = CharTokenizer.default_path(args.out)

    # Make sure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save(tok_path)

    # 3. Encode corpus
    print("[train] encoding corpus (this may take a moment for large files) ...")
    tokens = tokenizer.encode(text)
    print(f"[train] total tokens: {len(tokens):,}")

    # Free the raw text string -- we only need tokens from here on
    del text

    # 4. Split into train / val
    train_ds, val_ds = make_splits(tokens, args.block_size,
                                   val_fraction=args.val_frac,
                                   loss_mask=loss_mask)
    del tokens   # free encoded list; datasets hold int16 tensors

    # 5. Model config
    model_cfg = GPTConfig(
        vocab_size = tokenizer.vocab_size,
        block_size = args.block_size,
        n_embd     = args.n_embd,
        n_head     = args.n_head,
        n_layer    = args.n_layer,
        dropout    = args.dropout,
        bias       = False,
    )
    print(f"[train] model: {args.n_layer} layers x {args.n_embd} dim x "
          f"{args.n_head} heads, context {args.block_size}, "
          f"{tokenizer.vocab_size} vocab")

    # Never let eval_interval silently exceed max_iters: that disables
    # every in-loop checkpoint (the model only saves after the loop ends,
    # so an interrupted run loses everything).
    if args.eval_interval > args.max_iters:
        print(f"[train] eval_interval {args.eval_interval} > max_iters "
              f"{args.max_iters}: clamping to {max(1, args.max_iters)}")
        args.eval_interval = max(1, args.max_iters)

    # 6. Training config
    train_cfg = TrainConfig(
        batch_size                  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        max_iters                   = args.max_iters,
        weight_decay                = args.weight_decay,
        warmup_iters                = args.warmup,
        lr_decay_iters              = args.lr_decay,
        eval_interval               = args.eval_interval,
        eval_iters                  = args.eval_iters,
        out_path                    = args.out,
        tokenizer_path              = tok_path,
        log_interval                = args.log_interval,
        seed                        = args.seed,
        use_compile                 = args.compile,
        wall_clock_limit            = args.time_budget,
        device                      = resolve_device(args.device),
    )

    # 7. Train
    trainer = Trainer(model_cfg, train_cfg, train_ds, val_ds)
    trainer.run()


# ---------------------------------------------------------------------------
# Sub-command: finetune
# ---------------------------------------------------------------------------

def cmd_finetune(args: argparse.Namespace) -> None:
    from llm.model import GPT, GPTConfig
    from llm.tokenizer import CharTokenizer
    from llm.data import make_splits, build_chat_loss_mask, TextDataset
    from llm.train import Trainer, TrainConfig

    # Default output: overwrite input model
    if args.out is None:
        args.out = args.model

    # 1. Load pre-trained model and tokenizer
    if not os.path.exists(args.model):
        print(f"[error] Model file not found: '{args.model}'")
        sys.exit(1)

    tok_path = CharTokenizer.default_path(args.model)
    if not os.path.exists(tok_path):
        print(f"[error] Tokenizer not found at '{tok_path}'.")
        print("        The tokenizer JSON must be in the same folder as the model.")
        sys.exit(1)

    tokenizer = CharTokenizer.load(tok_path)
    model = GPT.load(args.model)
    old_vocab_size = tokenizer.vocab_size

    # Use model's actual block_size unless overridden
    block_size = args.block_size if args.block_size is not None else model.cfg.block_size

    # Optionally stretch the context window before anything else happens, so
    # the tokenizer, datasets and cache all use the new length.
    if args.extend_context and args.extend_context > model.cfg.block_size:
        print(f"[finetune] extending context {model.cfg.block_size} "
              f"-> {args.extend_context}")
        model.extend_context(args.extend_context)
        block_size = args.extend_context

    # 2. Load fine-tuning data
    text = load_data_path(args.data, pattern=args.pattern, recurse=not args.no_recurse)

    if len(text) < block_size * 2:
        print(f"[error] Corpus too short ({len(text):,} chars). "
              f"Need at least {block_size * 2:,} characters.")
        sys.exit(1)

    # 3. Build loss mask (before tokenizer expansion)
    loss_mask = None
    if args.reason:
        print("[finetune] building reasoning+response loss mask ...")
        loss_mask = build_chat_loss_mask(
            text, loss_prefixes=("Reasoning:", "Thinking:", "Response:"))
        n_response = sum(loss_mask)
        print(f"[finetune] loss tokens (reason+response): {n_response:,} / {len(loss_mask):,} "
              f"({100 * n_response // len(loss_mask):d}%)")
    elif args.thinking:
        print("[finetune] building thinking+response loss mask ...")
        loss_mask = build_chat_loss_mask(
            text, loss_prefixes=("Thinking:", "Response:"))
        n_response = sum(loss_mask)
        print(f"[finetune] loss tokens (thinking+response): {n_response:,} / {len(loss_mask):,} "
              f"({100 * n_response // len(loss_mask):d}%)")
    elif args.response_only:
        print("[finetune] building response-only loss mask ...")
        loss_mask = build_chat_loss_mask(text)
        n_response = sum(loss_mask)
        print(f"[finetune] response tokens: {n_response:,} / {len(loss_mask):,} "
              f"({100 * n_response // len(loss_mask):d}%)")
        print("[finetune] note: windows with no response token are excluded from "
              "sampling automatically, so the loss can no longer go NaN.")

    # 4. Expand tokenizer if new characters are present
    tokenizer.expand(text)

    # 5. Resize model embedding if vocab grew
    if tokenizer.vocab_size != old_vocab_size:
        print(f"[finetune] resizing embeddings {old_vocab_size} -> {tokenizer.vocab_size}")
        model.resize_token_embeddings(tokenizer.vocab_size)

    # 6. Save expanded tokenizer
    out_tok_path = CharTokenizer.default_path(args.out)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save(out_tok_path)

    # 7. Encode corpus
    print("[finetune] encoding corpus ...")
    tokens = tokenizer.encode(text)
    print(f"[finetune] total tokens: {len(tokens):,}")
    del text

    # 8. Split into train / val (with loss mask if available)
    train_ds, val_ds = make_splits(
        tokens, block_size,
        val_fraction=args.val_frac,
        loss_mask=loss_mask,
    )
    del tokens, loss_mask

    # 9. Anti-forgetting: replay + guard corpora
    replay_ds, guard_ds = None, None
    if args.replay or args.forget_guard:
        base_path = args.replay or args.forget_guard
        base_text = load_data_path(base_path, pattern=args.pattern,
                                   recurse=not args.no_recurse)
        base_tokens = tokenizer.encode(base_text)
        del base_text
        replay_frac = 1.0 - args.val_frac
        cut = int(len(base_tokens) * replay_frac)
        replay_ds = TextDataset(base_tokens[:cut], block_size)
        guard_ds = TextDataset(base_tokens[cut:], block_size)
        print(f"[finetune] replay corpus: {len(replay_ds):,} windows   "
              f"guard corpus: {len(guard_ds):,} windows")
        del base_tokens

    if args.eval_interval > args.max_iters:
        print(f"[finetune] eval_interval {args.eval_interval} > max_iters "
              f"{args.max_iters}: clamping to {max(1, args.max_iters)}")
        args.eval_interval = max(1, args.max_iters)

    # 10. Training config
    train_cfg = TrainConfig(
        batch_size                  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        max_iters                   = args.max_iters,
        weight_decay                = args.weight_decay,
        warmup_iters                = args.warmup,
        lr_decay_iters              = args.lr_decay,
        eval_interval               = args.eval_interval,
        eval_iters                  = args.eval_iters,
        out_path                    = args.out,
        tokenizer_path              = out_tok_path,
        log_interval                = args.log_interval,
        seed                        = args.seed,
        l2sp                        = args.l2sp,
        freeze_layers               = args.freeze_layers,
        forget_tol                  = args.forget_tol,
        use_compile                 = args.compile,
        wall_clock_limit            = args.time_budget,
        device                      = resolve_device(args.device),
    )

    # 11. Fine-tune (re-use Trainer with the loaded model)
    trainer = Trainer(model, train_cfg, train_ds, val_ds,
                      replay_dataset=replay_ds, guard_dataset=guard_ds)
    trainer.replay_prob = args.replay_prob
    trainer.run()


# ---------------------------------------------------------------------------
# Sub-command: moae (Model of Actual Experts)
# ---------------------------------------------------------------------------

def cmd_moae(args: argparse.Namespace) -> None:
    import math
    import tempfile
    import torch
    from llm import moae as M
    from llm.model import GPT
    from llm.tokenizer import CharTokenizer
    from llm.data import load_text

    if len(args.models) < 2:
        print("[moae] need at least 2 --models to merge")
        sys.exit(1)

    # 1. Load members
    members = []  # (path, model, tok, sd)
    for p in args.models:
        model, tok, sd = M.load_member(p)
        n = sum(v.numel() for v in sd.values())
        print(f"[moae] member: {p}  "
              f"L{model.cfg.n_layer}/d{model.cfg.n_embd}/"
              f"ctx{model.cfg.block_size}/v{tok.vocab_size}  {n:,} params")
        members.append((p, model, tok, sd))
    cfg = M.check_shapes([(m, t, s) for _, m, t, s in members])
    n_tags = len(args.finetune)

    # 2. Tokenizer union (+ tag chars), remap embeddings by token identity
    if args.no_union_vocab:
        vocabs = {t.vocab_size for _, _, t, _ in members}
        if len(vocabs) != 1:
            print(f"[moae] vocabs differ {sorted(vocabs)} -- drop "
                  f"--no_union_vocab to build the union")
            sys.exit(1)
        union_tok = members[0][2]
        grown = [s for _, _, _, s in members]
    else:
        union_tok = M.union_tokenizer([t for _, _, t, _ in members],
                                      n_tags=n_tags)
        grown = [M.grow_to_union(s, t, union_tok)
                 for _, _, t, s in members]
    cfg.vocab_size = union_tok.vocab_size

    # 3. Dedupe
    kept_idx_paths = M.dedupe([(p, s) for (p, _, _, _), s
                               in zip(members, grown)], args.dedupe)
    keep = {p for p, _ in kept_idx_paths}
    members = [(p, m, t, s) for (p, m, t, _), s in
               zip(members, grown) if p in keep]
    grown = [s for _, s in kept_idx_paths]
    if len(members) < 2:
        print("[moae] dedupe left fewer than 2 members -- "
              "lower --dedupe or pass more models")
        sys.exit(1)

    # 4. Coefficients
    if args.coefs is None:
        coefs = [1.0 / len(members)] * len(members)
    else:
        if len(args.coefs) != len(members):
            print(f"[moae] got {len(args.coefs)} --coefs for "
                  f"{len(members)} members")
            sys.exit(1)
        tot = sum(args.coefs)
        coefs = [c / tot for c in args.coefs]
    print(f"[moae] coefs: {[f'{c:.3f}' for c in coefs]}")

    # 5. Merge
    if args.layer_tune:
        if len(members) != 2 or not args.tune_on:
            print("[moae] --layer_tune needs exactly 2 members and --tune_on")
            sys.exit(1)
        val_toks = union_tok.encode(load_text(args.tune_on))
        merged_sd, alphas = M.tune_layer_coefs(
            grown, val_toks, cfg, iters=args.eval_iters)
        del val_toks
    elif args.mode == "ties":
        merged_sd = M.ties_merge(grown, coefs, density=args.density)
    else:
        merged_sd = M.soup_merge(grown, coefs)

    # 6. Embedding protection: take wte/lm_head from one member wholesale
    if args.embed_from is not None:
        i = args.embed_from
        if not 0 <= i < len(members):
            print(f"[moae] --embed_from {i} out of range "
                  f"(0..{len(members)-1})")
            sys.exit(1)
        merged_sd["transformer.wte.weight"] = \
            grown[i]["transformer.wte.weight"].clone()
        merged_sd["lm_head.weight"] = \
            merged_sd["transformer.wte.weight"].clone()
        print(f"[moae] embeddings taken from member {i} "
              f"({members[i][0]})")

    # 7. Build + save
    merged = M.build_merged_model(cfg, merged_sd)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    merged.save(args.out)
    tok_path = CharTokenizer.default_path(args.out)
    union_tok.save(tok_path)

    # 8. Scorecard: parents + merged, per domain
    domains = []
    if args.tune_on:
        domains.append(("tune", union_tok.encode(load_text(args.tune_on))))
    for spec in args.score:
        name, path = M.parse_scored(spec)
        domains.append((name, union_tok.encode(load_text(path))))
    if domains:
        rows = [(f"parent[{i}] {os.path.basename(p)[:24]}", m, t.vocab_size)
                for i, (p, m, t, _) in enumerate(members)]
        rows.append((f"MERGED -> {os.path.basename(args.out)}", merged, None))
        M.scorecard(rows, domains, cfg.block_size)
    else:
        print("[moae] no --tune_on/--score corpora: skipping scorecard "
              "(you are flying blind -- add at least one)")

    # 9. Optional mix-finetune with domain tags
    if args.finetune:
        texts = [load_text(p) for p in args.finetune]
        if args.balance:
            cut = min(len(t) for t in texts)
            texts = [t[:cut] for t in texts]
            print(f"[moae] balanced finetune corpora to {cut:,} chars each")
        tmpdir = tempfile.mkdtemp(prefix="moae_ft_")
        for i, (src, text) in enumerate(zip(args.finetune, texts)):
            tag = os.path.splitext(os.path.basename(src))[0][:24]
            with open(os.path.join(tmpdir, f"domain{i}_{tag}.txt"),
                      "w", encoding="utf-8") as f:
                f.write(M.tag_text(text, i))
            print(f"[moae] domain {i} tag=U+{0xE000+i:04X} <- {src}")
        ft = argparse.Namespace(
            model=args.out, data=tmpdir,
            out=args.finetune_out or args.out,
            pattern="*.txt", no_recurse=False,
            block_size=None, n_embd=None, n_head=None, n_layer=None,
            dropout=None, batch_size=args.ft_batch, grad_accum=4,
            lr=args.ft_lr, max_iters=args.ft_iters, weight_decay=0.1,
            warmup=-1, lr_decay=-1,
            eval_interval=min(args.ft_eval, max(1, args.ft_iters)),
            eval_iters=40, val_frac=0.1, log_interval=50, seed=args.seed,
            l2sp=0.0, freeze_layers=0, forget_tol=0.0, compile=False,
            wall_clock_limit=0.0, response_only=False, reason=False,
            thinking=False,
            replay=None, replay_prob=0.25, extend_context=0,
        )
        print(f"\n[moae] finetuning merged model on {len(texts)} tagged "
              f"domains -> {ft.out}")
        cmd_finetune(ft)

    print(f"\n[moae] done. merged model -> {args.out}")
    print(f"[moae] tokenizer      -> {tok_path}")


# ---------------------------------------------------------------------------
# Sub-command: generate
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    from llm.model import GPT
    from llm.tokenizer import CharTokenizer
    from llm.generate import generate, stream_generate

    tok_path = CharTokenizer.default_path(args.model)
    if not os.path.exists(tok_path):
        print(f"[error] Tokenizer not found at '{tok_path}'.")
        print("        The tokenizer JSON must be in the same folder as the model.")
        sys.exit(1)
    tokenizer = CharTokenizer.load(tok_path)

    if not os.path.exists(args.model):
        print(f"[error] Model file not found: '{args.model}'")
        sys.exit(1)
    model = GPT.load(args.model)
    device = resolve_device(args.device)
    model.to(device)
    print(f"[generate] device: {device}")

    prompt = args.prompt
    if args.thinking and "Thinking:" not in prompt:
        prompt = prompt.rstrip() + "\nThinking:"

    print(f"\n[generate] prompt     : {prompt!r}")
    print(f"[generate] temperature: {args.temperature}  "
           f"top_k: {args.top_k}  top_p: {args.top_p}  "
           f"max_tokens: {args.max_tokens}")
    print("\n" + "-" * 60)

    if args.stream:
        stream_generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
    else:
        result = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
        print(result)

    print("-" * 60)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def add_common_flags(p: argparse.ArgumentParser) -> None:
    """Flags shared by `train` and `finetune`."""
    g = p.add_argument_group("schedule and runtime")
    g.add_argument("--warmup", type=int, default=-1, metavar="ITERS",
                   help="Linear LR warmup steps (default: 5%% of max_iters, min 20)")
    g.add_argument("--lr_decay", type=int, default=-1, metavar="ITERS",
                   help="Cosine decay endpoint (default: max_iters). Tying this "
                        "to max_iters is what makes the schedule actually finish.")
    g.add_argument("--log_interval", type=int, default=50)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--time_budget", type=float, default=0.0, metavar="SECONDS",
                   help="Stop cleanly after this many seconds (0 = no limit). "
                        "Useful for unattended runs with a hard wall clock.")
    g.add_argument("--compile", action="store_true",
                   help="Enable torch.compile. Needs a C++ toolchain and free "
                        "space in /tmp; it is off by default.")
    g.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                   help="Train/infer device (default: cpu; auto = cuda when available)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python setup.py",
        description="Shimba LLM -- tiny decoder-only transformer (CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── train ────────────────────────────────────────────────────────
    t = sub.add_parser("train", help="Train a new model on text file(s)")
    t.add_argument("--data",          required=True,
                   help="Path to a .txt file OR a folder containing .txt files")
    t.add_argument("--out",           default="model.pth",
                   help="Output model path (default: model.pth)")
    t.add_argument("--pattern",       default="*.txt",
                   help="File glob pattern when --data is a folder (default: *.txt)")
    t.add_argument("--no_recurse",    action="store_true",
                   help="Do not search subfolders (top-level only)")
    # Model
    t.add_argument("--block_size",    type=int,   default=512)
    t.add_argument("--n_embd",        type=int,   default=256)
    t.add_argument("--n_head",        type=int,   default=4)
    t.add_argument("--n_layer",       type=int,   default=4)
    t.add_argument("--dropout",       type=float, default=0.1)
    # Training
    t.add_argument("--batch_size",    type=int,   default=8)
    t.add_argument("--grad_accum",    type=int,   default=4)
    t.add_argument("--lr",            type=float, default=3e-4)
    t.add_argument("--max_iters",     type=int,   default=5000)
    t.add_argument("--weight_decay",  type=float, default=0.1)
    t.add_argument("--eval_interval", type=int,   default=500)
    t.add_argument("--eval_iters",    type=int,   default=40)
    t.add_argument("--val_frac",      type=float, default=0.1)
    t.add_argument("--thinking", action="store_true",
                   help="Require Thinking: blocks in the corpus (fails if "
                        "absent). Without the flag they are auto-detected: "
                        "present -> loss on Thinking:/Response: only; "
                        "absent -> plain full-loss (non-thinking) model.")
    add_common_flags(t)

    # ── finetune ──────────────────────────────────────────────────────
    ft = sub.add_parser("finetune", help="Fine-tune a pre-trained model on new text")
    ft.add_argument("--model",       required=True,
                    help="Path to pre-trained model .pth file")
    ft.add_argument("--data",        required=True,
                    help="Path to a .txt file OR a folder containing .txt files")
    ft.add_argument("--out",         default=None,
                    help="Output model path (default: overwrites input model)")
    ft.add_argument("--pattern",     default="*.txt",
                    help="File glob pattern when --data is a folder (default: *.txt)")
    ft.add_argument("--no_recurse",  action="store_true",
                    help="Do not search subfolders (top-level only)")
    # Model (must match pre-trained model unless extending vocab)
    ft.add_argument("--block_size",  type=int,   default=None)
    ft.add_argument("--n_embd",      type=int,   default=None)
    ft.add_argument("--n_head",      type=int,   default=None)
    ft.add_argument("--n_layer",     type=int,   default=None)
    ft.add_argument("--dropout",     type=float, default=None)
    # Training (lower defaults for fine-tuning)
    ft.add_argument("--batch_size",  type=int,   default=8)
    ft.add_argument("--grad_accum",  type=int,   default=4)
    ft.add_argument("--lr",          type=float, default=5e-5)
    ft.add_argument("--max_iters",   type=int,   default=2000)
    ft.add_argument("--weight_decay",type=float, default=0.1)
    ft.add_argument("--eval_interval", type=int, default=200)
    ft.add_argument("--eval_iters",  type=int,   default=40)
    ft.add_argument("--val_frac",    type=float, default=0.1)
    add_common_flags(ft)
    ft.add_argument("--response_only", default=False, action="store_true",
                    help="Only compute loss on assistant response tokens "
                         "(expects 'User: ...\\nResponse: ...' format)")
    ft.add_argument("--reason", default=False, action="store_true",
                    help="Compute loss on reasoning + response tokens "
                         "(expects 'User: ...\\nReasoning: ...\\nResponse: ...' format)")
    ft.add_argument("--thinking", default=False, action="store_true",
                    help="Compute loss on thinking + response tokens "
                         "(expects 'User: ...\\nThinking: ...\\nResponse: ...' format, "
                         "e.g. flafi-dataset.txt)")
    ft.add_argument("--extend_context", type=int, default=0, metavar="N",
                    help="Stretch the model's context window to N tokens before "
                         "fine-tuning (position embeddings are resampled). Needed "
                         "for reasoning models: the stock checkpoints use 64, "
                         "which is too short to hold a full reasoning chain.")

    # ── Anti-catastrophic-forgetting ──────────────────────────────
    anti = ft.add_argument_group("anti-forgetting")
    anti.add_argument("--replay", metavar="CORPUS", default=None,
                      help="Original pre-training corpus. Mixed into every batch "
                           "so the general distribution never goes stale.")
    anti.add_argument("--replay_prob", type=float, default=0.25, metavar="FRAC",
                      help="Share of each batch drawn from the replay corpus "
                           "(default: 0.25)")
    anti.add_argument("--l2sp", type=float, default=0.0, metavar="LAMBDA",
                      help="L2 pull toward the base weights (L2-SP). Higher = "
                           "stays closer to the pre-trained model. Try 0.05-0.5.")
    anti.add_argument("--freeze_layers", type=int, default=0, metavar="N",
                      help="Freeze embeddings + the first N transformer blocks.")
    anti.add_argument("--forget_guard", metavar="CORPUS", default=None,
                      help="Corpus whose general-domain loss is monitored during "
                           "training. Used with --forget_tol.")
    anti.add_argument("--forget_tol", type=float, default=0.0, metavar="TOL",
                      help="Stop training if the guarded corpus loss rises more "
                           "than TOL above its starting value. 0 disables.")

    # ── moae ─────────────────────────────────────────────────────────
    m = sub.add_parser("moae", help="MOAE: merge trained checkpoints into one, "
                                    "then finetune on mixed data")
    m.add_argument("--models", nargs="+", required=True, metavar="PTH",
                   help="Checkpoints to merge (2+). Same architecture required; "
                        "vocab may differ (fixed by the tokenizer union).")
    m.add_argument("--out", required=True,
                   help="Output merged model path")
    m.add_argument("--mode", choices=["soup", "ties"], default="ties",
                   help="soup = weighted average; ties = trim + sign-elect "
                        "(better for independently trained models)")
    m.add_argument("--density", type=float, default=0.2, metavar="FRAC",
                   help="TIES: fraction of largest-magnitude deltas kept "
                        "per matrix (default: 0.2)")
    m.add_argument("--coefs", type=float, nargs="+", default=None, metavar="C",
                   help="Per-model weights (default: uniform). Normalised.")
    m.add_argument("--dedupe", type=float, default=0.01, metavar="RMS",
                   help="Drop members closer than this RMS distance "
                        "(0 disables; default: 0.01 -- catches re-saves, "
                        "keeps genuinely diverged members)")
    m.add_argument("--no_union_vocab", action="store_true",
                   help="Skip the tokenizer union (requires identical vocabs)")
    m.add_argument("--embed_from", type=int, default=None, metavar="I",
                   help="Take wte/lm_head from member I instead of merging "
                        "them (default: merge)")
    m.add_argument("--tune_on", default=None, metavar="CORPUS",
                   help="Held-out text for --layer_tune and scorecard")
    m.add_argument("--layer_tune", action="store_true",
                   help="2-model merges: greedy per-layer coefficient search "
                        "against --tune_on (needs --tune_on)")
    m.add_argument("--score", nargs="*", default=[], metavar="name=path",
                   help="Extra name=corpus pairs for the final scorecard")
    m.add_argument("--finetune", nargs="*", default=[], metavar="CORPUS",
                   help="After merging, finetune on these corpora, one "
                        "domain-tag char each")
    m.add_argument("--balance", action="store_true",
                   help="Truncate every finetune corpus to the smallest one "
                        "so no domain dominates by size")
    m.add_argument("--ft_iters", type=int, default=1000)
    m.add_argument("--ft_lr", type=float, default=5e-5)
    m.add_argument("--ft_batch", type=int, default=8)
    m.add_argument("--ft_eval", type=int, default=200)
    m.add_argument("--finetune_out", default=None,
                   help="Where the finetuned model goes "
                        "(default: overwrite --out)")
    m.add_argument("--eval_iters", type=int, default=20)
    m.add_argument("--seed", type=int, default=42)

    # ── generate ─────────────────────────────────────────────────────
    g = sub.add_parser("generate", help="Generate text from a trained model")
    g.add_argument("--prompt",      required=True)
    g.add_argument("--model",       required=True)
    g.add_argument("--max_tokens",  type=int,   default=200)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top_k",       type=int,   default=40)
    g.add_argument("--top_p",       type=float, default=0.95)
    g.add_argument("--repetition_penalty", type=float, default=1.1,
                   help="1.0 disables")
    g.add_argument("--seed",        type=int,   default=None,
                   help="Fix the sampling seed for reproducible output")
    g.add_argument("--stream",      action="store_true")
    g.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                   help="Inference device (default: cpu; auto = cuda when available)")
    g.add_argument("--thinking",    action="store_true",
                   help="Force a 'Thinking:' block first: the prompt is "
                        "suffixed with '\\nThinking:' so the model thinks "
                        "out loud before its Response:. Works even on "
                        "models never trained on Thinking: data.")

    # ── help ─────────────────────────────────────────────────────────
    sub.add_parser("help", help="Show detailed help")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExamples:")
        print("  # single file")
        print("  python setup.py train --data corpus.txt --out model.pth")
        print()
        print("  # entire folder of .txt files")
        print("  python setup.py train --data cuad\\CUAD_v1\\full_contract_txt\\Part_I --out model.pth")
        print()
        print("  # finetune a pre-trained model")
        print("  python setup.py finetune --model model.pth --data new_data.txt --out finetuned.pth")
        print()
        print("  # generate")
        print("  python setup.py generate --model model.pth --prompt \"This agreement\"")
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "finetune":
        cmd_finetune(args)
    elif args.command == "moae":
        cmd_moae(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "help":
        parser.print_help()
        print("\n" + __doc__)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()