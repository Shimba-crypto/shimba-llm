#!/home/shimba/Shimba-LLM-Training/venv/bin/python
"""
quantize.py -- Reduce precision or export format of a trained model.

Usage:
    python quantize.py --model model.pth --out model_quantized.pth --dtype int8
    python quantize.py --model model.pth --out model_fp16.pth --dtype float16
    python quantize.py --model model.pth --out model.gguf --dtype gguf
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.model import GPT, GPTConfig

def quantize_int8(model):
    from torch.quantization import quantize_dynamic
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def quantize_float16(model):
    return model.half()

def quantize_gguf(model_path, out_path, quant="f16"):
    from export_gguf import export_to_gguf
    export_to_gguf(model_path, out_path, quant)

def main():
    parser = argparse.ArgumentParser(description="Quantize or export a model")
    parser.add_argument("--model", required=True, help="Input model .pth file")
    parser.add_argument("--out",   required=True, help="Output path")
    parser.add_argument("--dtype", choices=["int8", "float16", "gguf"], default="int8",
                        help="Quantization target (default: int8)")
    parser.add_argument("--gguf-quant", choices=["f32", "f16", "q8"], default="f16",
                        help="GGUF quantization level (default: f16)")
    args = parser.parse_args()

    if args.dtype == "gguf":
        quantize_gguf(args.model, args.out, args.gguf_quant)
        return

    if not os.path.exists(args.model):
        print(f"[error] Model not found: {args.model}")
        sys.exit(1)

    print(f"[quantize] Loading {args.model} ...")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)

    if "config" not in checkpoint:
        print("[error] Checkpoint missing 'config' key.")
        sys.exit(1)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        print("[error] Checkpoint has no 'model' or 'state_dict' key.")
        sys.exit(1)

    cfg = checkpoint["config"]
    model = GPT(cfg)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[quantize] Applying {args.dtype} quantization ...")
    if args.dtype == "int8":
        quantized_model = quantize_int8(model)
    else:
        quantized_model = quantize_float16(model)

    new_checkpoint = {
        "config": cfg,
        "model": quantized_model.state_dict(),
    }
    torch.save(new_checkpoint, args.out)
    print(f"[quantize] Saved quantized model to {args.out}")

    orig_size = os.path.getsize(args.model) / (1024 * 1024)
    new_size = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[quantize] Size: {orig_size:.2f} MB -> {new_size:.2f} MB ({new_size/orig_size*100:.1f}%)")

if __name__ == "__main__":
    main()
