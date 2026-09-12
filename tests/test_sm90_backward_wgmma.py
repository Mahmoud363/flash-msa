"""Focused numerical gates for the Hopper-native backward tensor products."""

import pytest
import torch

from flash_msa.msa_backward_sm90 import (
    wgmma_backward_attention_tile,
    wgmma_backward_dq,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("product", ["dq", "dk", "dv"])
def test_backward_products_use_fp32_wgmma(product: str) -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    torch.manual_seed({"dq": 211, "dk": 223, "dv": 227}[product])
    left = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    right = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    actual = wgmma_backward_dq(left, right)
    reference = left.float() @ right.float()
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)


def test_fused_backward_tile_recomputes_probabilities_and_gradients() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    torch.manual_seed(229)
    scale = 128**-0.5
    q, k, v, grad_out = [
        torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    scores = (q.float() @ k.float().T) * scale
    probability = scores.softmax(dim=-1)
    lse = scores.logsumexp(dim=-1)
    output = probability @ v.float()
    delta = (grad_out.float() * output).sum(dim=-1)
    dq, dk, dv = wgmma_backward_attention_tile(
        q, k, v, grad_out, lse, delta, scale=scale
    )
    ds = probability * (grad_out.float() @ v.float().T - delta[:, None]) * scale
    references = (ds @ k.float(), ds.T @ q.float(), probability.T @ grad_out.float())
    for actual, reference in zip((dq, dk, dv), references):
        torch.testing.assert_close(actual, reference, rtol=5e-3, atol=4e-3)
