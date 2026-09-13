"""FP8 proxy quantization and selection-quality utilities for SM90.

These helpers intentionally keep block ranking in the existing selector while
the native FP8 WGMMA specialization is developed.  They provide the reference
quantization policy and the quality gates shared by both implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from cutlass.cute.runtime import from_dlpack


E4M3_MAX = 448.0
_QUANTIZE_CACHE: dict[tuple, object] = {}


@dataclass(frozen=True)
class SelectionAgreement:
    """Set-based agreement between reference and candidate block schedules."""

    recall_at_k: float
    exact_row_fraction: float
    mean_symmetric_difference: float


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16)


class _QuantizeE4M3PerBlockKernel:
    def __init__(self, *, batch: int, heads: int, sequence: int, head_dim: int) -> None:
        self.batch = int(batch)
        self.heads = int(heads)
        self.sequence = int(sequence)
        self.head_dim = int(head_dim)
        self.num_blocks = self.sequence // 128
        self.values_per_block = 128 * self.head_dim

    @cute.jit
    def __call__(self, source, output, scales, stream: cuda.CUstream):
        @cute.struct
        class SharedStorage:
            warp_max: cute.struct.MemRange[cutlass.Float32, 4]
            scale: cute.struct.MemRange[cutlass.Float32, 1]

        self.shared_storage = SharedStorage
        self.kernel(source, output, scales).launch(
            grid=[self.num_blocks, self.heads, self.batch],
            block=[128, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source: cute.Tensor, output: cute.Tensor, scales: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        block, head, batch = cute.arch.block_idx()
        warp = tidx // Int32(32)
        lane = tidx - warp * Int32(32)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        warp_max = storage.warp_max.get_tensor(cute.make_layout(4))
        shared_scale = storage.scale.get_tensor(cute.make_layout(1))

        local_max = cutlass.Float32(0.0)
        linear = tidx
        while linear < Int32(self.values_per_block):
            row = linear // Int32(self.head_dim)
            dim = linear - row * Int32(self.head_dim)
            value = source[batch, head, block * Int32(128) + row, dim].to(
                cutlass.Float32
            )
            absolute = value
            if absolute < cutlass.Float32(0.0):
                absolute = -absolute
            local_max = cute.arch.fmax(local_max, absolute)
            linear += Int32(128)
        local_max = cute.arch.warp_reduction_max(local_max, threads_in_group=32)
        if lane == 0:
            warp_max[warp] = local_max
        cute.arch.sync_threads()

        if warp == 0:
            block_max = cutlass.Float32(0.0)
            if lane < Int32(4):
                block_max = warp_max[lane]
            block_max = cute.arch.warp_reduction_max(block_max, threads_in_group=32)
            if lane == 0:
                value_scale = block_max / cutlass.Float32(E4M3_MAX)
                if value_scale < cutlass.Float32(1.0e-20):
                    value_scale = cutlass.Float32(1.0e-20)
                shared_scale[0] = value_scale
                scales[batch, head, block] = value_scale
        cute.arch.sync_threads()

        inverse_scale = cutlass.Float32(1.0) / shared_scale[0]
        linear = tidx * Int32(4)
        while linear < Int32(self.values_per_block):
            values = cute.make_rmem_tensor((4,), cutlass.Float32)
            quantized = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
            for element in cutlass.range_constexpr(4):
                element_linear = linear + Int32(element)
                row = element_linear // Int32(self.head_dim)
                dim = element_linear - row * Int32(self.head_dim)
                value = source[
                    batch, head, block * Int32(128) + row, dim
                ].to(cutlass.Float32) * inverse_scale
                value = cute.arch.fmax(value, cutlass.Float32(-E4M3_MAX))
                if value > cutlass.Float32(E4M3_MAX):
                    value = cutlass.Float32(E4M3_MAX)
                values[element] = value
            quantized.store(values.load().to(cutlass.Float8E4M3FN))
            for element in cutlass.range_constexpr(4):
                element_linear = linear + Int32(element)
                row = element_linear // Int32(self.head_dim)
                dim = element_linear - row * Int32(self.head_dim)
                output[
                    batch, head, block * Int32(128) + row, dim
                ] = quantized[element]
            linear += Int32(128 * 4)


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


def quantize_proxy_e4m3_per_block_cutedsl(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused GH200 per-block amax, scale, and E4M3 conversion."""

    if tensor.ndim != 4 or tensor.shape[-1] != 128 or tensor.shape[2] % 128:
        raise ValueError("proxy tensor must be [B,H,S,128] with S divisible by 128")
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("fused FP8 quantization requires fp16 or bf16 input")
    if tensor.device.type != "cuda" or torch.cuda.get_device_capability(tensor.device) != (9, 0):
        raise RuntimeError("fused FP8 quantization requires an SM90 CUDA device")

    source = tensor.detach().contiguous()
    output = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (source.shape[0], source.shape[1], source.shape[2] // 128),
        device=source.device,
        dtype=torch.float32,
    )
    source_t = _to_cute_tensor(source)
    output_t = _to_cute_tensor(output)
    scales_t = _to_cute_tensor(scales)
    stream = cuda.CUstream(torch.cuda.current_stream(source.device).cuda_stream)
    key = (tuple(source.shape), source_t.element_type)
    if key not in _QUANTIZE_CACHE:
        kernel = _QuantizeE4M3PerBlockKernel(
            batch=source.shape[0],
            heads=source.shape[1],
            sequence=source.shape[2],
            head_dim=source.shape[3],
        )
        _QUANTIZE_CACHE[key] = cute.compile(
            kernel, source_t, output_t, scales_t, stream
        )
    _QUANTIZE_CACHE[key](source_t, output_t, scales_t, stream)
    return output, scales


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
    "quantize_proxy_e4m3_per_block_cutedsl",
    "quantize_proxy_e4m3_per_head",
    "selection_agreement",
]
