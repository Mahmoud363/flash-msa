"""GH200 quality tests for the Milestone 3 FP8 proxy path."""

import pytest
import torch

from flash_msa import flash_msa_func
from flash_msa.msa_select_fp8 import (
    dequantize_proxy_e4m3,
    dequantize_proxy_e4m3_per_block,
    quantize_proxy_e4m3_per_block,
    quantize_proxy_e4m3_per_block_cutedsl,
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


@pytest.mark.parametrize(
    ("n_proxy_heads", "n_proxy_kv_heads"), [(2, 1), (4, 1), (4, 2), (8, 2)]
)
def test_fp8_wgmma_selector_matches_dequantized_reference(
    n_proxy_heads: int, n_proxy_kv_heads: int
) -> None:
    _require_sm90()
    torch.manual_seed(79)
    q = torch.randn(
        1, n_proxy_heads, 512, 128, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        1, n_proxy_kv_heads, 512, 128, device="cuda", dtype=torch.bfloat16
    )
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


def test_fused_cutedsl_quantizer_matches_reference_policy() -> None:
    _require_sm90()
    torch.manual_seed(83)
    source = torch.randn(1, 2, 256, 128, device="cuda", dtype=torch.bfloat16)
    expected_q, expected_scale = quantize_proxy_e4m3_per_block(source)
    actual_q, actual_scale = quantize_proxy_e4m3_per_block_cutedsl(source)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(
        actual_q.float(), expected_q.float(), rtol=0, atol=0
    )


def test_fp8_training_step_meets_approximate_stability_gates(monkeypatch) -> None:
    _require_sm90()
    torch.manual_seed(89)

    def tensor(heads: int) -> torch.Tensor:
        return torch.randn(1, heads, 1024, 128, device="cuda", dtype=torch.bfloat16)

    inputs = (tensor(4), tensor(1), tensor(16), tensor(2), tensor(2))

    def run(backend: str):
        monkeypatch.setenv("MSA_FORWARD_BACKEND", "sm90")
        monkeypatch.setenv("MSA_SELECT_BACKEND", backend)
        values = tuple(value.clone().requires_grad_(True) for value in inputs)
        output, kl_loss = flash_msa_func(*values, 512, 128**-0.5)
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
    assert float(output_cosine.detach()) >= 0.97
    assert float(loss_error.detach()) <= 0.01
    for reference, candidate in zip(reference_gradients, fp8_gradients):
        gradient_cosine = torch.nn.functional.cosine_similarity(
            reference.float().reshape(1, -1), candidate.float().reshape(1, -1)
        )
        assert float(gradient_cosine.detach()) >= 0.99
