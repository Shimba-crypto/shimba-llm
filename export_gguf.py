"""
export_gguf.py -- Export a .pth model to GGUF format (for llama.cpp).

Usage:
    python export_gguf.py --model deepseek-67.pth --out deepseek-67.gguf

The GGUF format is used by llama.cpp and its bindings.
This exports the model architecture and weights in a compatible format.
"""

import os
import sys
import struct
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGUF_TENSOR_F32 = 0
GGUF_TENSOR_F16 = 1
GGUF_TENSOR_Q4_0 = 2
GGUF_TENSOR_Q4_1 = 3
GGUF_TENSOR_Q8_0 = 8

ARCH = "gpt2"

TENSOR_NAMES = {
    "transformer.wte.weight": "token_embd.weight",
    "transformer.wpe.weight": "position_embd.weight",
    "transformer.ln_f.weight": "output_norm.weight",
    "transformer.ln_f.bias": "output_norm.bias",
    "lm_head.weight": "output.weight",
}

for i in range(128):
    TENSOR_NAMES[f"transformer.h.{i}.ln_1.weight"] = f"blk.{i}.attn_norm.weight"
    TENSOR_NAMES[f"transformer.h.{i}.ln_1.bias"] = f"blk.{i}.attn_norm.bias"
    TENSOR_NAMES[f"transformer.h.{i}.attn.c_attn.weight"] = f"blk.{i}.attn_qkv.weight"
    TENSOR_NAMES[f"transformer.h.{i}.attn.c_proj.weight"] = f"blk.{i}.attn_output.weight"
    TENSOR_NAMES[f"transformer.h.{i}.ln_2.weight"] = f"blk.{i}.ffn_norm.weight"
    TENSOR_NAMES[f"transformer.h.{i}.ln_2.bias"] = f"blk.{i}.ffn_norm.bias"
    TENSOR_NAMES[f"transformer.h.{i}.mlp.fc.weight"] = f"blk.{i}.ffn_gate.weight"
    TENSOR_NAMES[f"transformer.h.{i}.mlp.proj.weight"] = f"blk.{i}.ffn_down.weight"


def gguf_name(torch_key: str) -> str:
    return TENSOR_NAMES.get(torch_key, torch_key)


def write_string(f, s: str):
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def write_value(f, vtype: int, value):
    f.write(struct.pack("<I", vtype))
    if vtype == GGUF_TYPE_UINT8:
        f.write(struct.pack("<B", value))
    elif vtype == GGUF_TYPE_INT8:
        f.write(struct.pack("<b", value))
    elif vtype == GGUF_TYPE_UINT16:
        f.write(struct.pack("<H", value))
    elif vtype == GGUF_TYPE_INT16:
        f.write(struct.pack("<h", value))
    elif vtype == GGUF_TYPE_UINT32:
        f.write(struct.pack("<I", value))
    elif vtype == GGUF_TYPE_INT32:
        f.write(struct.pack("<i", value))
    elif vtype == GGUF_TYPE_FLOAT32:
        f.write(struct.pack("<f", value))
    elif vtype == GGUF_TYPE_BOOL:
        f.write(struct.pack("<?", value))
    elif vtype == GGUF_TYPE_STRING:
        write_string(f, value)
    elif vtype == GGUF_TYPE_UINT64:
        f.write(struct.pack("<Q", value))
    elif vtype == GGUF_TYPE_INT64:
        f.write(struct.pack("<q", value))
    elif vtype == GGUF_TYPE_FLOAT64:
        f.write(struct.pack("<d", value))


def write_array_header(f, arr_type: int, length: int):
    f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
    f.write(struct.pack("<I", arr_type))
    f.write(struct.pack("<Q", length))


def export_to_gguf(model_path: str, output_path: str, quantize: str = "f16"):
    print(f"[gguf] Loading {model_path} ...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    cfg = checkpoint.get("config")
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint

    vocab_size = cfg.vocab_size
    n_embd = cfg.n_embd
    n_head = cfg.n_head
    n_layer = cfg.n_layer
    block_size = cfg.block_size

    # Build tokenizer metadata
    tok_path = model_path.replace(".pth", "_tokenizer.json")
    tokenizer_info = {}
    if os.path.exists(tok_path):
        with open(tok_path) as f:
            tokenizer_info = json.load(f)

    tensor_data = []
    for key, tensor in state_dict.items():
        if tensor.numel() == 0:
            continue
        gguf_key = gguf_name(key)
        if quantize == "f16" and tensor.dim() >= 2:
            tensor = tensor.half()
        elif quantize == "q8":
            if tensor.dim() >= 2:
                tensor = tensor.to(torch.float16)
        tensor_data.append((gguf_key, tensor, key))

    tensor_data.sort(key=lambda x: x[0])

    # Write GGUF file
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensor_data)))

        # Metadata key-value pairs
        kv_count = 8
        f.write(struct.pack("<Q", kv_count))

        write_kv_string(f, "general.architecture", ARCH)
        write_kv_string(f, "general.name", os.path.splitext(os.path.basename(model_path))[0])
        write_kv_int(f, "gpt2.context_length", block_size)
        write_kv_int(f, "gpt2.embedding_length", n_embd)
        write_kv_int(f, "gpt2.num_attention_heads", n_head)
        write_kv_int(f, "gpt2.num_layers", n_layer)
        write_kv_int(f, "gpt2.vocab_size", vocab_size)
        write_kv_int(f, "gpt2.feedforward_length", 4 * n_embd)

        # Tensor info
        offset = f.tell() + len(tensor_data) * (4 + 8 + 4 + 4)
        offset = (offset + 31) & ~31

        for gguf_key, tensor, orig_key in tensor_data:
            name_bytes = gguf_key.encode("utf-8")
            dims = tensor.dim()
            shape = list(tensor.shape)

            if quantize == "f16" and tensor.dtype == torch.float16:
                tensor_type = GGUF_TENSOR_F16
            else:
                tensor_type = GGUF_TENSOR_F32

            nbytes = tensor.nbytes

            f.write(struct.pack("<I", tensor_type))
            f.write(struct.pack("<Q", offset))
            f.write(struct.pack("<I", dims))
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            for s in reversed(shape):
                f.write(struct.pack("<Q", s))

            offset += nbytes
            offset = (offset + 31) & ~31

        # Tensor data
        for gguf_key, tensor, orig_key in tensor_data:
            data = tensor.numpy().tobytes()
            f.write(data)
            pad = (32 - (len(data) % 32)) % 32
            if pad:
                f.write(b"\x00" * pad)

    orig_size = os.path.getsize(model_path) / (1024*1024)
    new_size = os.path.getsize(output_path) / (1024*1024)
    print(f"[gguf] Exported to {output_path}")
    print(f"[gguf] Size: {orig_size:.1f} MB -> {new_size:.1f} MB ({new_size/orig_size*100:.1f}%)")


def write_kv_string(f, key: str, value: str):
    write_key(f, key)
    write_value(f, GGUF_TYPE_STRING, value)


def write_kv_int(f, key: str, value: int):
    write_key(f, key)
    write_value(f, GGUF_TYPE_INT32, value)


def write_key(f, key: str):
    write_string(f, key)


def main():
    parser = argparse.ArgumentParser(description="Export .pth model to GGUF format")
    parser.add_argument("--model", required=True, help="Input .pth file")
    parser.add_argument("--out", default=None, help="Output .gguf file")
    parser.add_argument("--quantize", choices=["f32", "f16", "q8"], default="f16",
                        help="Quantization type (default: f16)")
    args = parser.parse_args()

    if args.out is None:
        args.out = args.model.replace(".pth", ".gguf")

    export_to_gguf(args.model, args.out, args.quantize)


if __name__ == "__main__":
    main()
