#!/usr/bin/env python3
"""Profile one deterministic Flash-MSA training case from a headless shell."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from contextlib import contextmanager
from pathlib import Path

import torch

from flash_msa import flash_msa_func


BLOCK_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--n-proxy-heads", type=int, default=4)
    parser.add_argument("--n-proxy-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--documents-per-row", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    positive = (
        args.batch_size,
        args.sequence_length,
        args.top_k,
        args.n_heads,
        args.n_kv_heads,
        args.n_proxy_heads,
        args.n_proxy_kv_heads,
        args.head_dim,
        args.repeats,
    )
    if any(value <= 0 for value in positive) or args.warmup < 0:
        raise ValueError("sizes/repeats must be positive and warmup must be nonnegative")
    if args.sequence_length % BLOCK_SIZE or args.top_k % BLOCK_SIZE:
        raise ValueError("sequence length and top-k must be divisible by 128")
    if args.top_k > args.sequence_length:
        raise ValueError("top-k cannot exceed sequence length")
    if args.head_dim != 128:
        raise ValueError("Flash-MSA currently requires head dimension 128")
    if args.n_heads % args.n_kv_heads or args.n_heads % args.n_proxy_heads:
        raise ValueError("query heads must be divisible by KV and proxy heads")
    if args.n_proxy_heads % args.n_proxy_kv_heads:
        raise ValueError("proxy heads must be divisible by proxy KV heads")
    if args.n_proxy_heads < args.n_kv_heads or args.n_proxy_heads % args.n_kv_heads:
        raise ValueError("proxy heads must be >= and divisible by KV heads")
    if not 0 <= args.documents_per_row <= args.sequence_length:
        raise ValueError("documents per row must be between 0 and sequence length")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")


def make_cu_seqlens(args: argparse.Namespace) -> torch.Tensor | None:
    if args.documents_per_row == 0:
        return None

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    offsets = [0]
    for batch_index in range(args.batch_size):
        if args.documents_per_row > 1:
            boundaries = torch.randperm(
                args.sequence_length - 1, generator=generator
            )[: args.documents_per_row - 1]
            boundaries = (boundaries + 1).sort().values.tolist()
        else:
            boundaries = []
        row_start = batch_index * args.sequence_length
        offsets.extend(row_start + boundary for boundary in boundaries)
        offsets.append(row_start + args.sequence_length)
    return torch.tensor(offsets, device="cuda", dtype=torch.int32)


def make_inputs(args: argparse.Namespace, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    shape = (args.batch_size, args.sequence_length, args.head_dim)

    def tensor(heads: int) -> torch.Tensor:
        value = torch.randn(
            shape[0], heads, shape[1], shape[2], device="cuda", dtype=dtype
        )
        return value.requires_grad_(True)

    return (
        tensor(args.n_proxy_heads),
        tensor(args.n_proxy_kv_heads),
        tensor(args.n_heads),
        tensor(args.n_kv_heads),
        tensor(args.n_kv_heads),
    )


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def clear_grads(inputs: tuple[torch.Tensor, ...]) -> None:
    for tensor in inputs:
        tensor.grad = None


def forward(
    inputs: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with nvtx_range("flash_msa_forward"):
        output, kl_loss = flash_msa_func(
            *inputs,
            args.top_k,
            args.head_dim**-0.5,
            cu_seqlens=cu_seqlens,
        )
    return output, output.float().sum() + kl_loss.float()


def timed(operation) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def environment() -> dict[str, object]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "gpu_memory_bytes": properties.total_memory,
    }


def main() -> None:
    args = parse_args()
    validate(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    inputs = make_inputs(args, getattr(torch, args.dtype))
    cu_seqlens = make_cu_seqlens(args)

    for _ in range(args.warmup):
        _, loss = forward(inputs, args, cu_seqlens)
        with nvtx_range("flash_msa_backward"):
            loss.backward()
        clear_grads(inputs)
    torch.cuda.synchronize()

    forward_ms: list[float] = []
    backward_ms: list[float] = []
    with nvtx_range("flash_msa_measured_iterations"):
        for _ in range(args.repeats):
            holder: list[torch.Tensor] = []

            def run_forward() -> None:
                _, loss = forward(inputs, args, cu_seqlens)
                holder.append(loss)

            forward_ms.append(timed(run_forward))
            with nvtx_range("flash_msa_backward"):
                backward_ms.append(timed(holder.pop().backward))
            clear_grads(inputs)

    result = {
        "environment": environment(),
        "case": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "top_k": args.top_k,
            "n_heads": args.n_heads,
            "n_kv_heads": args.n_kv_heads,
            "n_proxy_heads": args.n_proxy_heads,
            "n_proxy_kv_heads": args.n_proxy_kv_heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "documents_per_row": args.documents_per_row,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "timing_ms": {
            "forward_median": statistics.median(forward_ms),
            "forward_min": min(forward_ms),
            "forward_max": max(forward_ms),
            "backward_median": statistics.median(backward_ms),
            "backward_min": min(backward_ms),
            "backward_max": max(backward_ms),
        },
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
