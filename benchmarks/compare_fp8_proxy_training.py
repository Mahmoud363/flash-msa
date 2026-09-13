#!/usr/bin/env python3
"""Compare BF16- and FP8-selected Flash-MSA forward/backward numerics."""

from __future__ import annotations

import argparse
import json
import os

import torch

from flash_msa import flash_msa_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=67)
    return parser.parse_args()


def cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            reference.float().reshape(1, -1),
            candidate.float().reshape(1, -1),
        ).item()
    )


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = reference.float().norm().clamp_min(1.0e-20)
    return float(((candidate.float() - reference.float()).norm() / denominator).item())


def run(inputs: tuple[torch.Tensor, ...], top_k: int, backend: str):
    os.environ["MSA_FORWARD_BACKEND"] = "sm90"
    os.environ["MSA_SELECT_BACKEND"] = backend
    values = tuple(value.detach().clone().requires_grad_(True) for value in inputs)
    output, kl_loss = flash_msa_func(*values, top_k, 128**-0.5)
    loss = output.float().square().mean() + kl_loss.float()
    gradients = torch.autograd.grad(loss, values)
    return output.detach(), loss.detach(), tuple(g.detach() for g in gradients)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("this benchmark requires an SM90 GPU")
    if args.sequence_length % 128 or args.top_k % 128:
        raise ValueError("sequence length and top-k must be divisible by 128")
    torch.manual_seed(args.seed)
    shape = (args.batch_size, args.sequence_length, 128)

    def tensor(heads: int) -> torch.Tensor:
        return torch.randn(
            shape[0], heads, shape[1], shape[2], device="cuda", dtype=torch.bfloat16
        )

    inputs = (tensor(4), tensor(1), tensor(16), tensor(2), tensor(2))
    bf16_output, bf16_loss, bf16_gradients = run(inputs, args.top_k, "bf16")
    fp8_output, fp8_loss, fp8_gradients = run(inputs, args.top_k, "fp8")
    names = ("q_proxy", "k_proxy", "q", "k", "v")
    result = {
        "case": vars(args),
        "finite": {
            "output": bool(torch.isfinite(fp8_output).all()),
            "loss": bool(torch.isfinite(fp8_loss)),
            "gradients": all(bool(torch.isfinite(value).all()) for value in fp8_gradients),
        },
        "output": {
            "cosine": cosine(bf16_output, fp8_output),
            "relative_l2": relative_l2(bf16_output, fp8_output),
        },
        "loss": {
            "bf16": float(bf16_loss),
            "fp8": float(fp8_loss),
            "relative_error": float(
                ((fp8_loss - bf16_loss).abs() / bf16_loss.abs().clamp_min(1.0e-20)).item()
            ),
        },
        "gradients": {
            name: {
                "cosine": cosine(reference, candidate),
                "relative_l2": relative_l2(reference, candidate),
            }
            for name, reference, candidate in zip(names, bf16_gradients, fp8_gradients)
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
