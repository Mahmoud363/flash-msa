#!/usr/bin/env python3
"""Profile Milestone 4 quantization, remote attention, and full forward costs."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from flash_msa import flash_msa_func
from flash_msa.msa_kv_fp8 import MixedFP8QKV
from flash_msa.msa_forward_sm90 import wgmma_selected_attention
from flash_msa.msa_kv_fp8 import quantize_kv_e4m3_cutedsl
from flash_msa.msa_select_cutedsl import select_blocks
from flash_msa.msa_select_fp8 import quantize_proxy_e4m3_per_block_cutedsl
from flash_msa.reverse_index_cuda import build_sparse_attention_metadata_cuda


def timed(operation, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=16384)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("SM90 GPU required")
    torch.manual_seed(107)
    sequence = args.sequence_length
    q_proxy = torch.randn(1, 4, sequence, 128, device="cuda", dtype=torch.bfloat16)
    k_proxy = torch.randn(1, 1, sequence, 128, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(1, 16, sequence, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, sequence, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = 128**-0.5
    blocks = select_blocks(
        q_proxy,
        k_proxy,
        scale=scale,
        num_blocks=sequence // 128,
        top_k_blocks=args.top_k // 128,
    )
    metadata = build_sparse_attention_metadata_cuda(
        blocks, backward_query_chunk=32, remote_query_chunk=512
    )
    schedule = metadata.kv_outer_schedule
    assert schedule is not None
    task_meta = schedule.task_meta[: schedule.num_tasks]
    query_indices = schedule.query_indices[: schedule.num_edges]
    storage = quantize_kv_e4m3_cutedsl(k, v)
    q8, q_scale = quantize_proxy_e4m3_per_block_cutedsl(q)
    prequantized = MixedFP8QKV(
        q=q8,
        k=storage.k,
        v=storage.v,
        q_scale=q_scale,
        k_scale=storage.k_scale,
        v_scale=storage.v_scale,
    )

    def remote_bf16():
        return wgmma_selected_attention(
            q, k, v, task_meta, query_indices, n_proxy_heads=4, scale=scale
        )

    def remote_fp8():
        return wgmma_selected_attention(
            q8,
            storage.k,
            storage.v,
            task_meta,
            query_indices,
            n_proxy_heads=4,
            scale=scale,
            q_scale=q_scale,
            k_scale=storage.k_scale,
            v_scale=storage.v_scale,
        )

    def quantize():
        return quantize_kv_e4m3_cutedsl(k, v)

    def quantize_qkv():
        quantize_proxy_e4m3_per_block_cutedsl(q)
        return quantize_kv_e4m3_cutedsl(k, v)

    inputs = (q_proxy, k_proxy, q, k, v)

    def full(storage_backend: str, packed: MixedFP8QKV | None = None):
        os.environ["MSA_FORWARD_BACKEND"] = "sm90"
        os.environ["MSA_SELECT_BACKEND"] = "bf16"
        os.environ["MSA_KV_STORAGE"] = storage_backend
        return flash_msa_func(
            *inputs, args.top_k, scale, prequantized_qkv=packed
        )

    def full_fp8_conversion_inclusive():
        packed_storage = quantize_kv_e4m3_cutedsl(k, v)
        packed_q, packed_q_scale = quantize_proxy_e4m3_per_block_cutedsl(q)
        return full(
            "fp8",
            MixedFP8QKV(
                packed_q,
                packed_storage.k,
                packed_storage.v,
                packed_q_scale,
                packed_storage.k_scale,
                packed_storage.v_scale,
            ),
        )

    for _ in range(args.warmup):
        quantize()
        quantize_qkv()
        remote_bf16()
        remote_fp8()
        full("bf16")
        full("fp8", prequantized)
    torch.cuda.synchronize()
    result = {
        "case": vars(args),
        "tasks": schedule.num_tasks,
        "edges": schedule.num_edges,
        "timing_ms": {
            "quantize_kv": timed(quantize, args.repeats),
            "quantize_qkv": timed(quantize_qkv, args.repeats),
            "remote_bf16": timed(remote_bf16, args.repeats),
            "remote_fp8_prequantized": timed(remote_fp8, args.repeats),
            "full_bf16": timed(lambda: full("bf16"), args.repeats),
            "full_fp8_prequantized": timed(
                lambda: full("fp8", prequantized), args.repeats
            ),
            "full_fp8_conversion_inclusive": timed(
                full_fp8_conversion_inclusive, args.repeats
            ),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
