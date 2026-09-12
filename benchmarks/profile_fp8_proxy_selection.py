#!/usr/bin/env python3
"""Measure E4M3 proxy-selection quality and quantization overhead on GH200."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from flash_msa.msa_select_cutedsl import select_blocks
from flash_msa.msa_select_fp8 import (
    dequantize_proxy_e4m3,
    dequantize_proxy_e4m3_per_block,
    quantize_proxy_e4m3_per_block,
    quantize_proxy_e4m3_per_head,
    selection_agreement,
)
from flash_msa.msa_select_sm90 import select_blocks_fp8_sm90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16384)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--n-proxy-heads", type=int, default=4)
    parser.add_argument("--n-proxy-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def elapsed_ms(operation) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    result = operation()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end), result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("this benchmark requires an SM90 GPU")
    if args.sequence_length % 128 or args.top_k % 128:
        raise ValueError("sequence length and top-k must be divisible by 128")
    if args.top_k > args.sequence_length:
        raise ValueError("top-k cannot exceed sequence length")

    torch.manual_seed(args.seed)
    shape = (args.batch_size, args.sequence_length, args.head_dim)
    q = torch.randn(
        shape[0], args.n_proxy_heads, shape[1], shape[2],
        device="cuda", dtype=torch.bfloat16,
    )
    k = torch.randn(
        shape[0], args.n_proxy_kv_heads, shape[1], shape[2],
        device="cuda", dtype=torch.bfloat16,
    )
    num_blocks = args.sequence_length // 128
    top_k_blocks = args.top_k // 128
    scale = args.head_dim**-0.5

    def select(q_value: torch.Tensor, k_value: torch.Tensor) -> torch.Tensor:
        return select_blocks(
            q_value,
            k_value,
            scale=scale,
            num_blocks=num_blocks,
            top_k_blocks=top_k_blocks,
        )

    reference = select(q, k)
    q8, q_scale = quantize_proxy_e4m3_per_head(q)
    k8, k_scale = quantize_proxy_e4m3_per_head(k)
    q_quantized = dequantize_proxy_e4m3(q8, q_scale, dtype=q.dtype)
    k_quantized = dequantize_proxy_e4m3(k8, k_scale, dtype=k.dtype)
    candidate = select(q_quantized, k_quantized)
    head_agreement = selection_agreement(reference, candidate)
    q8_block, q_block_scale = quantize_proxy_e4m3_per_block(q)
    k8_block, k_block_scale = quantize_proxy_e4m3_per_block(k)
    q_block = dequantize_proxy_e4m3_per_block(
        q8_block, q_block_scale, dtype=q.dtype
    )
    k_block = dequantize_proxy_e4m3_per_block(
        k8_block, k_block_scale, dtype=k.dtype
    )
    block_candidate = select(q_block, k_block)
    block_agreement = selection_agreement(reference, block_candidate)
    native_candidate = select_blocks_fp8_sm90(
        q8_block,
        k8_block,
        q_block_scale,
        k_block_scale,
        scale=scale,
        top_k_blocks=top_k_blocks,
    )
    native_agreement = selection_agreement(reference, native_candidate)

    for _ in range(args.warmup):
        quantize_proxy_e4m3_per_head(q)
        quantize_proxy_e4m3_per_head(k)
        select(q_quantized, k_quantized)
    torch.cuda.synchronize()

    quantize_times = []
    selection_times = []
    native_times = []
    for _ in range(args.repeats):
        quantize_ms, _ = elapsed_ms(
            lambda: (
                quantize_proxy_e4m3_per_head(q),
                quantize_proxy_e4m3_per_head(k),
            )
        )
        select_ms, _ = elapsed_ms(lambda: select(q_quantized, k_quantized))
        native_ms, _ = elapsed_ms(
            lambda: select_blocks_fp8_sm90(
                q8_block,
                k8_block,
                q_block_scale,
                k_block_scale,
                scale=scale,
                top_k_blocks=top_k_blocks,
            )
        )
        quantize_times.append(quantize_ms)
        selection_times.append(select_ms)
        native_times.append(native_ms)

    result = {
        "case": vars(args) | {"json": None},
        "quality": {
            "per_head": vars(head_agreement),
            "per_block": vars(block_agreement),
            "native_fp8_wgmma": vars(native_agreement),
        },
        "timing_ms": {
            "quantize_median": statistics.median(quantize_times),
            "dequantized_selection_median": statistics.median(selection_times),
            "native_fp8_wgmma_median": statistics.median(native_times),
        },
    }
    result["case"].pop("json")
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json is not None:
        args.json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
