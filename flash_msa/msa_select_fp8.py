"""FP8 proxy quantization and selection-quality utilities for SM90.

These helpers intentionally keep block ranking in the existing selector while
the native FP8 WGMMA specialization is developed.  They provide the reference
quantization policy and the quality gates shared by both implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


E4M3_MAX = 448.0


@dataclass(frozen=True)
class SelectionAgreement:
    """Set-based agreement between reference and candidate block schedules."""

    recall_at_k: float
    exact_row_fraction: float
    mean_symmetric_difference: float


def quantize_proxy_e4m3_per_head(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[B,H,S,D]`` proxy activations with one FP32 scale per head."""

    if tensor.ndim != 4:
        raise ValueError("proxy tensor must have shape [B,H,S,D]")
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("proxy tensor must use fp16, bf16, or fp32")
    if tensor.device.type != "cuda":
        raise ValueError("FP8 proxy quantization requires CUDA")

    amax = tensor.detach().float().abs().amax(dim=(2, 3), keepdim=True)
    scale = (amax / E4M3_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (tensor.detach().float() / scale).clamp(-E4M3_MAX, E4M3_MAX)
    return quantized.to(torch.float8_e4m3fn), scale


def dequantize_proxy_e4m3(
    tensor: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    """Apply FP32 scales and return an fp16/bf16 proxy tensor."""

    if tensor.dtype != torch.float8_e4m3fn:
        raise TypeError("tensor must use float8_e4m3fn")
    if scale.dtype != torch.float32 or scale.shape != (*tensor.shape[:2], 1, 1):
        raise ValueError("scale must be FP32 with shape [B,H,1,1]")
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("dequantized dtype must be fp16 or bf16")
    return (tensor.float() * scale).to(dtype)


def quantize_proxy_e4m3_per_block(
    tensor: torch.Tensor, *, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize proxy activations with one FP32 scale per token block and head."""

    if tensor.ndim != 4:
        raise ValueError("proxy tensor must have shape [B,H,S,D]")
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("proxy tensor must use fp16, bf16, or fp32")
    if tensor.device.type != "cuda":
        raise ValueError("FP8 proxy quantization requires CUDA")
    if block_size <= 0 or tensor.shape[2] % block_size:
        raise ValueError("sequence length must be divisible by block_size")

    batch, heads, sequence, head_dim = tensor.shape
    blocked = tensor.detach().float().reshape(
        batch, heads, sequence // block_size, block_size, head_dim
    )
    amax = blocked.abs().amax(dim=(3, 4))
    scale = (amax / E4M3_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (blocked / scale[..., None, None]).clamp(-E4M3_MAX, E4M3_MAX)
    return quantized.reshape_as(tensor).to(torch.float8_e4m3fn), scale


def dequantize_proxy_e4m3_per_block(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize a block-scaled E4M3 proxy tensor."""

    if tensor.dtype != torch.float8_e4m3fn:
        raise TypeError("tensor must use float8_e4m3fn")
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("dequantized dtype must be fp16 or bf16")
    batch, heads, sequence, head_dim = tensor.shape
    expected = (batch, heads, sequence // block_size)
    if sequence % block_size or scale.dtype != torch.float32 or scale.shape != expected:
        raise ValueError(f"scale must be FP32 with shape {expected}")
    blocked = tensor.float().reshape(
        batch, heads, sequence // block_size, block_size, head_dim
    )
    return (blocked * scale[..., None, None]).reshape_as(tensor).to(dtype)


def selection_agreement(
    reference: torch.Tensor, candidate: torch.Tensor
) -> SelectionAgreement:
    """Measure unordered Top-K block-set agreement without host-side sorting."""

    if reference.shape != candidate.shape or reference.ndim != 4:
        raise ValueError("selection tensors must have matching [B,H,S,K] shapes")
    if reference.dtype != torch.int32 or candidate.dtype != torch.int32:
        raise TypeError("selection tensors must be int32")
    if reference.shape[-1] == 0:
        raise ValueError("selection tensors must contain at least one block")

    matches = reference.unsqueeze(-1) == candidate.unsqueeze(-2)
    matched_reference = matches.any(dim=-1)
    recall = matched_reference.float().mean()
    exact_rows = matched_reference.all(dim=-1).float().mean()
    symmetric_difference = 2.0 * (
        reference.shape[-1] - matched_reference.sum(dim=-1).float()
    )
    return SelectionAgreement(
        recall_at_k=float(recall.item()),
        exact_row_fraction=float(exact_rows.item()),
        mean_symmetric_difference=float(symmetric_difference.mean().item()),
    )


__all__ = [
    "SelectionAgreement",
    "dequantize_proxy_e4m3",
    "dequantize_proxy_e4m3_per_block",
    "quantize_proxy_e4m3_per_block",
    "quantize_proxy_e4m3_per_head",
    "selection_agreement",
]
