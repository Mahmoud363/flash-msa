"""Focused numerical gates for the Hopper-native backward tensor products."""

import pytest
import torch

from flash_msa.msa_backward_sm90 import (
    wgmma_backward_attention_tile,
    wgmma_backward_dq,
    wgmma_kv_row_backward_main,
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


def test_kv_row_backward_matches_dense_causal_attention() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    torch.manual_seed(233)
    batch, heads, kv_heads, proxy_heads, seq_len, dim = 1, 4, 1, 1, 128, 128
    scale = dim**-0.5
    q = torch.randn(batch, heads, seq_len, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, seq_len, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    grad_out = torch.randn_like(q)
    k_heads = k.expand(batch, heads, seq_len, dim).float()
    v_heads = v.expand(batch, heads, seq_len, dim).float()
    scores = torch.einsum("bhsd,bhtd->bhst", q.float(), k_heads) * scale
    scores.masked_fill_(
        torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool).triu(1),
        -torch.inf,
    )
    probability = scores.softmax(dim=-1)
    output = torch.einsum("bhst,bhtd->bhsd", probability, v_heads)
    lse = scores.logsumexp(dim=-1)
    delta = (output * grad_out.float()).sum(dim=-1)
    row_ptr = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
    query_ids = torch.arange(seq_len, device="cuda", dtype=torch.int32)
    actual = wgmma_kv_row_backward_main(
        q, k, v, grad_out, lse, delta, row_ptr, query_ids,
        n_proxy_heads=proxy_heads, scale=scale,
    )
    ds = probability * (
        torch.einsum("bhsd,bhtd->bhst", grad_out.float(), v_heads)
        - delta[..., None]
    ) * scale
    references = (
        torch.einsum("bhst,bhtd->bhsd", ds, k_heads),
        torch.einsum("bhst,bhsd->bhtd", ds, q.float()).sum(dim=1, keepdim=True),
        torch.einsum("bhst,bhsd->bhtd", probability, grad_out.float()).sum(
            dim=1, keepdim=True
        ),
    )
    for result, reference in zip(actual, references):
        torch.testing.assert_close(result, reference, rtol=8e-3, atol=8e-3)
