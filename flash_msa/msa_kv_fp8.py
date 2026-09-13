"""Block-scaled E4M3 K/V storage for the fixed-length SM90 forward path.

Milestone 4 changes the representation read by the remote-attention mainloop,
not its arithmetic: stored E4M3 tiles are restored to BF16 before QK/PV WGMMA.
The helpers here define that representation and provide the numerical oracle
used by the native TMA staging implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flash_msa.msa_select_fp8 import (
    dequantize_proxy_e4m3_per_block,
    quantize_proxy_e4m3_per_block,
    quantize_proxy_e4m3_per_block_cutedsl,
)


BLOCK_SIZE = 128


@dataclass(frozen=True)
class FP8KVStorage:
    """E4M3 K/V tensors and independent FP32 scales per 128-token tile."""

    k: torch.Tensor
    v: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    source_dtype: torch.dtype

    @property
    def payload_bytes(self) -> int:
        """Bytes read by an FP8 K/V consumer, including scale metadata."""

        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.k, self.v, self.k_scale, self.v_scale)
        )


@dataclass(frozen=True)
class MixedFP8QKV:
    """Prequantized inputs for FP8-QK/BF16-PV selected attention.

    The originating BF16 tensors remain separate and are used by local
    attention and backward.  This bundle is produced at the Q/K/V projection
    boundary and is never reconstructed inside sparse attention.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor


def prequantize_mixed_qkv_cutedsl(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> MixedFP8QKV:
    """Prequantize Q/K/V once for the mixed FP8-QK/BF16-PV forward path."""

    _validate_kv(k, v)
    if q.ndim != 4 or q.shape[-1] != BLOCK_SIZE or q.shape[2] % BLOCK_SIZE:
        raise ValueError("Q must have shape [B,H,S,128] with S divisible by 128")
    if q.dtype != k.dtype or q.device != k.device:
        raise ValueError("Q and K/V must have matching dtype and device")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("Q and K/V must agree on batch, sequence, and head dimension")
    q8, q_scale = quantize_proxy_e4m3_per_block_cutedsl(q)
    storage = quantize_kv_e4m3_cutedsl(k, v)
    return MixedFP8QKV(
        q=q8,
        k=storage.k,
        v=storage.v,
        q_scale=q_scale,
        k_scale=storage.k_scale,
        v_scale=storage.v_scale,
    )


def _validate_kv(k: torch.Tensor, v: torch.Tensor) -> None:
    if k.shape != v.shape or k.ndim != 4:
        raise ValueError("K and V must have matching [B,H,S,128] shapes")
    if k.shape[-1] != BLOCK_SIZE or k.shape[2] % BLOCK_SIZE:
        raise ValueError("K and V must be [B,H,S,128] with S divisible by 128")
    if k.dtype != v.dtype or k.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("K and V must have matching fp16 or bf16 dtypes")
    if k.device != v.device or k.device.type != "cuda":
        raise ValueError("K and V must be CUDA tensors on the same device")


def _make_storage(
    k: torch.Tensor,
    v: torch.Tensor,
    quantize,
) -> FP8KVStorage:
    _validate_kv(k, v)
    k8, k_scale = quantize(k)
    v8, v_scale = quantize(v)
    return FP8KVStorage(k8, v8, k_scale, v_scale, k.dtype)


def quantize_kv_e4m3_reference(k: torch.Tensor, v: torch.Tensor) -> FP8KVStorage:
    """Reference PyTorch implementation of the block-scaled storage policy."""

    return _make_storage(k, v, quantize_proxy_e4m3_per_block)


def quantize_kv_e4m3_cutedsl(k: torch.Tensor, v: torch.Tensor) -> FP8KVStorage:
    """Quantize K and V with the fused per-tensor CuTe conversion kernel."""

    return _make_storage(k, v, quantize_proxy_e4m3_per_block_cutedsl)


def dequantize_kv_e4m3(storage: FP8KVStorage) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore the stored K/V tensors for oracle and fallback comparisons."""

    k = dequantize_proxy_e4m3_per_block(
        storage.k, storage.k_scale, dtype=storage.source_dtype
    )
    v = dequantize_proxy_e4m3_per_block(
        storage.v, storage.v_scale, dtype=storage.source_dtype
    )
    return k, v


def bf16_kv_payload_bytes(k: torch.Tensor, v: torch.Tensor) -> int:
    """Return the dense source payload size used for traffic comparisons."""

    _validate_kv(k, v)
    return k.numel() * k.element_size() + v.numel() * v.element_size()


__all__ = [
    "FP8KVStorage",
    "MixedFP8QKV",
    "bf16_kv_payload_bytes",
    "dequantize_kv_e4m3",
    "quantize_kv_e4m3_cutedsl",
    "quantize_kv_e4m3_reference",
    "prequantize_mixed_qkv_cutedsl",
]
