"""
afcag.py -- Aura Full CPU and GPU inference.

Allows model inference on CPU, GPU (CUDA), or both.
Automatically detects available hardware and provides
a unified interface for generation.

Usage:
    from afcag import AFCAGModel
    model = AFCAGModel("deepseek-67.pth", device="auto")
    output = model.generate("Hello")
"""

import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm.model import GPT, GPTConfig
from llm.tokenizer import CharTokenizer


def detect_device(prefer: str = "auto") -> torch.device:
    """
    Detect best available device.
    auto  -> CUDA if available else CPU
    cpu   -> CPU
    gpu   -> CUDA (requires CUDA)
    """
    if prefer == "cpu":
        return torch.device("cpu")
    elif prefer == "gpu" or prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    else:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


class AFCAGModel:
    """
    Full CPU + GPU inference wrapper.
    Handles device placement automatically.
    """

    def __init__(self, pth_path: str, device: str = "auto",
                 half: bool = False):
        self.pth_path = pth_path
        self.device = detect_device(device)
        self.half = half

        print(f"[afcag] Loading model on {self.device}...")
        self.model = GPT.load(pth_path)
        self.model.to(self.device)
        self.model.eval()

        if half and self.device.type == "cuda":
            self.model.half()
            print("[afcag] Using half precision (float16)")

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[afcag] Model has {n_params:,} params on {self.device}")

    @torch.no_grad()
    def generate(self, prompt: str, tokenizer,
                 max_new_tokens: int = 200,
                 temperature: float = 0.8,
                 top_k: int = 40,
                 top_p: float = 0.95,
                 repetition_penalty: float = 1.1) -> str:
        from llm.generate import generate as gen_fn
        return gen_fn(
            self.model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    @torch.no_grad()
    def stream_generate(self, prompt: str, tokenizer,
                        max_new_tokens: int = 200,
                        temperature: float = 0.8,
                        top_k: int = 40,
                        top_p: float = 0.95):
        from llm.generate import stream_generate as sg
        sg(
            self.model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def to(self, device: str):
        """Move model to a different device."""
        self.device = detect_device(device)
        self.model.to(self.device)
        return self
