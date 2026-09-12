#!/usr/bin/env python3
"""Validate and time the KV-stationary SM90 main-attention backward."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from flash_msa.msa_backward_cutedsl import run_fused_backward
from flash_msa.msa_backward_sm90 import wgmma_kv_row_backward_main
from flash_msa.msa_forward_cutedsl import run_main_forward
from flash_msa.msa_select_cutedsl import compute_proxy_lse, select_blocks
from flash_msa.reverse_index_cuda import build_sparse_attention_metadata_cuda


def median_ms(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=16384)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(241)
    b, hp, hpkv, h, hkv, d = 1, 4, 1, 16, 2, 128
    s, scale = args.sequence_length, d**-0.5
    make = lambda heads: torch.randn(b, heads, s, d, device="cuda", dtype=torch.bfloat16)
    qp, kp, q, k, v, do = make(hp), make(hpkv), make(h), make(hkv), make(hkv), make(h)
    blocks = select_blocks(qp, kp, scale=scale, num_blocks=s // 128,
                           top_k_blocks=args.top_k // 128)
    metadata = build_sparse_attention_metadata_cuda(
        blocks, backward_query_chunk=16, remote_query_chunk=512
    )
    output, lse, _ = run_main_forward(q, k, v, scale=scale, metadata=metadata)
    delta = (output.float() * do.float()).sum(-1)
    schedule = metadata.kv_outer_schedule
    assert schedule is not None

    def native():
        return wgmma_kv_row_backward_main(
            q, k, v, do, lse, delta, schedule.row_ptr,
            schedule.query_indices, n_proxy_heads=hp, scale=scale,
        )

    lse_proxy = compute_proxy_lse(qp, kp, scale=scale, metadata=metadata)

    def baseline():
        return run_fused_backward(
            qp, kp, q, k, v, do, lse, lse_proxy, delta,
            metadata.task_meta, metadata.task_qids,
            scale=scale, grad_kl_scale=0.0,
        )[2:]

    for _ in range(args.warmup):
        native()
        baseline()
    torch.cuda.synchronize()
    result = {
        "case": vars(args),
        "edges": schedule.num_edges,
        "timing_ms": {
            "kv_row_main": median_ms(native, args.repeats),
            "legacy_fused_main_plus_disabled_proxy": median_ms(baseline, args.repeats),
        },
    }
    if args.check:
        actual, reference = native(), baseline()
        result["accuracy"] = {}
        for name, got, expected in zip(("dq", "dk", "dv"), actual, reference):
            gf, ef = got.float(), expected.float()
            result["accuracy"][name] = {
                "cosine": torch.nn.functional.cosine_similarity(
                    gf.flatten(), ef.flatten(), dim=0
                ).item(),
                "max_abs": (gf - ef).abs().max().item(),
                "mean_abs": (gf - ef).abs().mean().item(),
            }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
