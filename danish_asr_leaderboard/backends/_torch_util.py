"""Tiny torch device/dtype helpers shared by GPU backends (torch imported lazily)."""
from __future__ import annotations


def cuda_ok(device: str) -> bool:
    import torch

    return device == "cuda" and torch.cuda.is_available()


def pipeline_device(device: str) -> int:
    """Device index for the HF ``pipeline`` API (0 = first GPU, -1 = CPU)."""
    return 0 if cuda_ok(device) else -1


def device_map(device: str) -> str:
    return "cuda:0" if cuda_ok(device) else "cpu"


def half_dtype(device: str):
    """float16 on GPU, float32 on CPU."""
    import torch

    return torch.float16 if cuda_ok(device) else torch.float32


def bf16_dtype(device: str):
    """bfloat16 on GPU, float32 on CPU."""
    import torch

    return torch.bfloat16 if cuda_ok(device) else torch.float32
