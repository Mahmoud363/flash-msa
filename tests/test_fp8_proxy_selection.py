"""GH200 quality tests for the Milestone 3 FP8 proxy path."""

import pytest
import torch

from flash_msa.msa_select_fp8 import (
    dequantize_proxy_e4m3,
    dequantize_proxy_e4m3_per_block,
    quantize_proxy_e4m3_per_block,
    quantize_proxy_e4m3_per_head,
    selection_agreement,
)
from flash_msa.msa_select_cutedsl import select_blocks
from flash_msa.msa_select_sm90 import select_blocks_fp8_sm90


def _require_sm90() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")


def test_per_head_e4m3_quantization_scales_and_roundtrip() -> None:
    _require_sm90()
    torch.manual_seed(71)
    source = torch.randn(2, 3, 257, 128, device="cuda", dtype=torch.bfloat16)
    source[0, 1].mul_(8.0)
    quantized, scale = quantize_proxy_e4m3_per_head(source)
    restored = dequantize_proxy_e4m3(quantized, scale, dtype=source.dtype)

    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert scale.shape == (2, 3, 1, 1)
    assert bool(torch.isfinite(restored).all())
    relative_l2 = (restored.float() - source.float()).norm() / source.float().norm()
    assert float(relative_l2) < 0.04


def test_selection_agreement_is_order_independent() -> None:
    _require_sm90()
    reference = torch.tensor(
        [[[[1, 2, 3], [4, 5, 6]]]], device="cuda", dtype=torch.int32
    )
    candidate = torch.tensor(
        [[[[3, 1, 2], [4, 7, 6]]]], device="cuda", dtype=torch.int32
    )
    agreement = selection_agreement(reference, candidate)
    assert agreement.recall_at_k == pytest.approx(5 / 6)
    assert agreement.exact_row_fraction == pytest.approx(0.5)
    assert agreement.mean_symmetric_difference == pytest.approx(1.0)


def test_per_block_e4m3_quantization_uses_independent_scales() -> None:
    _require_sm90()
    torch.manual_seed(73)
    source = torch.randn(1, 2, 256, 128, device="cuda", dtype=torch.bfloat16)
    source[:, :, 128:].mul_(16.0)
    quantized, scale = quantize_proxy_e4m3_per_block(source)
    restored = dequantize_proxy_e4m3_per_block(
        quantized, scale, dtype=source.dtype
    )

    assert scale.shape == (1, 2, 2)
    assert bool((scale[:, :, 1] > scale[:, :, 0] * 8).all())
    relative_l2 = (restored.float() - source.float()).norm() / source.float().norm()
    assert float(relative_l2) < 0.04


def test_fp8_wgmma_selector_matches_dequantized_reference() -> None:
    _require_sm90()
    torch.manual_seed(79)
    q = torch.randn(1, 4, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, 512, 128, device="cuda", dtype=torch.bfloat16)
    q8, q_scale = quantize_proxy_e4m3_per_block(q)
    k8, k_scale = quantize_proxy_e4m3_per_block(k)
    q_ref = dequantize_proxy_e4m3_per_block(q8, q_scale, dtype=q.dtype)
    k_ref = dequantize_proxy_e4m3_per_block(k8, k_scale, dtype=k.dtype)
    reference = select_blocks(
        q_ref, k_ref, scale=128**-0.5, num_blocks=4, top_k_blocks=2
    )
    candidate = select_blocks_fp8_sm90(
        q8,
        k8,
        q_scale,
        k_scale,
        scale=128**-0.5,
        top_k_blocks=2,
    )
    agreement = selection_agreement(reference, candidate)
    assert agreement.recall_at_k > 0.995
