#!/usr/bin/env python3
"""Measure the normal BF16 SM90 attention and full-forward I/O paths."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from flash_msa import flash_msa_func
from flash_msa.msa_forward_sm90 import wgmma_selected_attention
from flash_msa.msa_select_cutedsl import select_blocks
from flash_msa.reverse_index_cuda import build_sparse_attention_metadata_cuda


def median_ms(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return statistics.median(samples)


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

    def remote():
        return wgmma_selected_attention(
            q, k, v, task_meta, query_indices, n_proxy_heads=4, scale=scale
        )

    inputs = (q_proxy, k_proxy, q, k, v)

    def full():
        os.environ["MSA_FORWARD_BACKEND"] = "sm90"
        os.environ["MSA_SELECT_BACKEND"] = "bf16"
        os.environ["MSA_KV_STORAGE"] = "bf16"
        return flash_msa_func(*inputs, args.top_k, scale)

    for _ in range(args.warmup):
        remote()
        full()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "case": vars(args),
                "tasks": schedule.num_tasks,
                "edges": schedule.num_edges,
                "timing_ms": {
                    "remote_bf16": median_ms(remote, args.repeats),
                    "full_bf16": median_ms(full, args.repeats),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
