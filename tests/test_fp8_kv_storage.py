"""Milestone 4 tests for block-scaled E4M3 K/V storage."""

import pytest
import torch

from flash_msa.msa_kv_fp8 import (
    bf16_kv_payload_bytes,
    dequantize_kv_e4m3,
    quantize_kv_e4m3_cutedsl,
    quantize_kv_e4m3_reference,
)


def _require_sm90() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(97)
    k = torch.randn(2, 2, 256, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k[:, :, 128:].mul_(8.0)
    v[:, 1, :128].mul_(4.0)
    return k, v


def test_fp8_kv_reference_roundtrip_and_independent_scales() -> None:
    _require_sm90()
    k, v = _inputs()
    storage = quantize_kv_e4m3_reference(k, v)
    restored_k, restored_v = dequantize_kv_e4m3(storage)

    assert storage.k.dtype == torch.float8_e4m3fn
    assert storage.v.dtype == torch.float8_e4m3fn
    assert storage.k_scale.shape == (2, 2, 2)
    assert storage.v_scale.shape == (2, 2, 2)
    assert bool((storage.k_scale[:, :, 1] > storage.k_scale[:, :, 0] * 4).all())
    assert bool(storage.v_scale[:, 1, 0].mean() > storage.v_scale[:, 0, 0].mean() * 2)
    for source, restored in ((k, restored_k), (v, restored_v)):
        assert bool(torch.isfinite(restored).all())
        relative_l2 = (restored.float() - source.float()).norm() / source.float().norm()
        assert float(relative_l2) < 0.04


def test_fp8_kv_cutedsl_matches_reference_policy() -> None:
    _require_sm90()
    k, v = _inputs()
    expected = quantize_kv_e4m3_reference(k, v)
    actual = quantize_kv_e4m3_cutedsl(k, v)

    torch.testing.assert_close(actual.k_scale, expected.k_scale, rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(actual.v_scale, expected.v_scale, rtol=2e-6, atol=1e-8)
    # The fused kernel multiplies by a reciprocal while the torch oracle uses
    # division, so values exactly on an E4M3 rounding boundary may choose
    # adjacent representable numbers. Compare the reconstructed BF16 tiles.
    expected_k, expected_v = dequantize_kv_e4m3(expected)
    actual_k, actual_v = dequantize_kv_e4m3(actual)
    for reference, candidate in ((expected_k, actual_k), (expected_v, actual_v)):
        relative_l2 = (candidate.float() - reference.float()).norm() / reference.float().norm()
        assert float(relative_l2) < 0.005


def test_fp8_kv_payload_is_about_half_bf16() -> None:
    _require_sm90()
    k, v = _inputs()
    storage = quantize_kv_e4m3_reference(k, v)
    ratio = storage.payload_bytes / bf16_kv_payload_bytes(k, v)

    # Two FP32 scales per 128x128 K/V tile add only 8 bytes to 32 KiB FP8 data.
    assert ratio == pytest.approx(0.5001220703125)
