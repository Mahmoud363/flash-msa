"""Milestone 4 tests for block-scaled E4M3 K/V storage."""

import pytest
import torch

from flash_msa import flash_msa_func, prequantize_mixed_qkv_cutedsl
from flash_msa.msa_kv_fp8 import (
    bf16_kv_payload_bytes,
    dequantize_kv_e4m3,
    quantize_kv_e4m3_cutedsl,
    quantize_kv_e4m3_reference,
)
from flash_msa.msa_forward_sm90 import (
    tma_load_fp8_kv_as_bf16,
    wgmma_selected_attention,
)
from flash_msa.msa_select_fp8 import quantize_proxy_e4m3_per_block_cutedsl


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


def test_fp8_kv_tma_restores_selected_bf16_tiles() -> None:
    _require_sm90()
    k, v = _inputs()
    storage = quantize_kv_e4m3_cutedsl(k, v)
    # Columns are [batch, proxy head, key block, query count, edge offset].
    task_meta = torch.tensor(
        [[0, 0, 1, 1, 0], [1, 3, 0, 1, 1]],
        device="cuda",
        dtype=torch.int32,
    )
    copied_k, copied_v = tma_load_fp8_kv_as_bf16(
        storage.k,
        storage.v,
        storage.k_scale,
        storage.v_scale,
        task_meta,
        n_proxy_heads=4,
    )
    restored_k, restored_v = dequantize_kv_e4m3(storage)
    torch.testing.assert_close(copied_k[0], restored_k[0, 0, 128:256], rtol=0, atol=0)
    torch.testing.assert_close(copied_v[0], restored_v[0, 0, 128:256], rtol=0, atol=0)
    torch.testing.assert_close(copied_k[1], restored_k[1, 1, :128], rtol=0, atol=0)
    torch.testing.assert_close(copied_v[1], restored_v[1, 1, :128], rtol=0, atol=0)


def test_mixed_fp8_qk_bf16_pv_matches_attention_reference() -> None:
    _require_sm90()
    torch.manual_seed(101)
    q = torch.randn(1, 16, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 512, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    storage = quantize_kv_e4m3_cutedsl(k, v)
    q8, q_scale = quantize_proxy_e4m3_per_block_cutedsl(q)
    qids = torch.arange(320, 336, dtype=torch.int32, device="cuda")
    task_meta = torch.tensor(
        [[0, 3, 1, 16, 0]], dtype=torch.int32, device="cuda"
    )
    kwargs = dict(n_proxy_heads=4, scale=128**-0.5)
    reference_out, reference_lse = wgmma_selected_attention(
        q, k, v, task_meta, qids, **kwargs
    )
    actual_out, actual_lse = wgmma_selected_attention(
        q8,
        storage.k,
        storage.v,
        task_meta,
        qids,
        q_scale=q_scale,
        k_scale=storage.k_scale,
        v_scale=storage.v_scale,
        **kwargs,
    )
    # QK is intentionally approximate FP8 while PV and output remain BF16.
    output_cosine = torch.nn.functional.cosine_similarity(
        actual_out.float().reshape(1, -1), reference_out.float().reshape(1, -1)
    )
    assert float(output_cosine) >= 0.999
    torch.testing.assert_close(actual_lse, reference_lse, rtol=3e-3, atol=1.5e-2)


def test_fp8_kv_fixed_length_training_is_stable(monkeypatch) -> None:
    _require_sm90()
    torch.manual_seed(103)

    def tensor(heads: int) -> torch.Tensor:
        return torch.randn(1, heads, 1024, 128, device="cuda", dtype=torch.bfloat16)

    inputs = (tensor(4), tensor(1), tensor(16), tensor(2), tensor(2))

    def run(storage_backend: str):
        monkeypatch.setenv("MSA_FORWARD_BACKEND", "sm90")
        # Hold block selection fixed to isolate K/V storage error.
        monkeypatch.setenv("MSA_SELECT_BACKEND", "bf16")
        monkeypatch.setenv("MSA_KV_STORAGE", storage_backend)
        values = tuple(value.clone().requires_grad_(True) for value in inputs)
        prequantized = (
            prequantize_mixed_qkv_cutedsl(values[2], values[3], values[4])
            if storage_backend == "fp8"
            else None
        )
        output, kl_loss = flash_msa_func(
            *values,
            512,
            128**-0.5,
            prequantized_qkv=prequantized,
        )
        loss = output.float().square().mean() + kl_loss.float()
        gradients = torch.autograd.grad(loss, values)
        return output, loss, gradients

    reference_output, reference_loss, reference_gradients = run("bf16")
    fp8_output, fp8_loss, fp8_gradients = run("fp8")
    assert bool(torch.isfinite(fp8_output).all())
    assert bool(torch.isfinite(fp8_loss))
    assert all(bool(torch.isfinite(gradient).all()) for gradient in fp8_gradients)
    output_cosine = torch.nn.functional.cosine_similarity(
        reference_output.float().reshape(1, -1), fp8_output.float().reshape(1, -1)
    )
    loss_error = (fp8_loss - reference_loss).abs() / reference_loss.abs().clamp_min(1e-20)
    assert float(output_cosine.detach()) >= 0.99
    assert float(loss_error.detach()) <= 0.01
    for reference, candidate in zip(reference_gradients, fp8_gradients):
        gradient_cosine = torch.nn.functional.cosine_similarity(
            reference.float().reshape(1, -1), candidate.float().reshape(1, -1)
        )
        assert float(gradient_cosine.detach()) >= 0.99
