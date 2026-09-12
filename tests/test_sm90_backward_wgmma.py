"""Focused numerical gates for the Hopper-native backward tensor products."""

import pytest
import torch

from flash_msa import flash_msa_func
from flash_msa.msa_backward_sm90 import (
    wgmma_backward_attention_tile,
    wgmma_backward_dq,
    wgmma_kv_row_backward_main,
)
from flash_msa.reverse_index_cuda import (
    build_document_segment_metadata,
    build_sparse_attention_metadata_cuda,
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
    row_ptr = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    query_ids = torch.empty(1, device="cuda", dtype=torch.int32)
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


def test_segmented_kv_row_backward_predicates_uneven_documents() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    torch.manual_seed(239)
    batch, heads, proxy_heads, seq_len, dim = 2, 4, 1, 128, 128
    scale = dim**-0.5
    documents = torch.empty(batch, seq_len, device="cuda", dtype=torch.int32)
    documents[0, :37], documents[0, 37:91], documents[0, 91:] = 0, 1, 2
    documents[1, :19], documents[1, 19:73], documents[1, 73:] = 3, 4, 5
    segments = build_document_segment_metadata(documents)
    q = torch.randn(batch, heads, seq_len, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 1, seq_len, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    grad_out = torch.randn_like(q)
    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    scores = torch.einsum(
        "bhsd,bhtd->bhst", q_ref, k_ref.expand(-1, heads, -1, -1)
    ) * scale
    causal = torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool).triu(1)
    different_document = documents[:, None, :, None] != documents[:, None, None, :]
    scores = scores.masked_fill(causal[None, None] | different_document, -torch.inf)
    probability = scores.softmax(dim=-1)
    output = torch.einsum(
        "bhst,bhtd->bhsd", probability, v_ref.expand(-1, heads, -1, -1)
    )
    lse = scores.logsumexp(dim=-1).detach()
    output.backward(grad_out.float())
    delta = (output.detach() * grad_out.float()).sum(dim=-1)
    row_ptr = torch.zeros(
        proxy_heads * segments.num_segments + 1, device="cuda", dtype=torch.int32
    )
    actual = wgmma_kv_row_backward_main(
        q, k, v, grad_out, lse, delta, row_ptr,
        torch.empty(1, device="cuda", dtype=torch.int32),
        n_proxy_heads=proxy_heads,
        scale=scale,
        segment_starts=segments.starts,
        segment_lengths=segments.lengths,
        segment_batches=segments.batches,
    )
    for result, reference in zip(actual, (q_ref.grad, k_ref.grad, v_ref.grad)):
        torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-2)


def test_segmented_remote_metadata_is_canonical_reverse_csr() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA test")
    documents = torch.empty(1, 128, device="cuda", dtype=torch.int32)
    documents[0, :37], documents[0, 37:91], documents[0, 91:] = 0, 1, 2
    segments = build_document_segment_metadata(documents)
    selected = torch.zeros((1, 1, 128, 1), device="cuda", dtype=torch.int32)
    metadata = build_sparse_attention_metadata_cuda(
        selected,
        backward_query_chunk=16,
        remote_query_chunk=64,
        document_segments=segments,
    )
    schedule = metadata.kv_outer_schedule
    assert schedule is not None and schedule.segmented
    torch.testing.assert_close(
        schedule.row_ptr,
        torch.tensor([0, 91, 91, 91], device="cuda", dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        schedule.query_indices[: schedule.num_edges].sort().values,
        torch.arange(37, 128, device="cuda", dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    offsets = schedule.task_offsets[: schedule.num_tasks + 1]
    assert bool((offsets[1:] >= offsets[:-1]).all())


def test_autograd_dispatches_sm90_main_backward(monkeypatch) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    monkeypatch.setenv("MSA_FORWARD_BACKEND", "sm90")
    monkeypatch.setenv("MSA_SELECT_BACKEND", "bf16")
    monkeypatch.setenv("MSA_KV_STORAGE", "bf16")
    torch.manual_seed(251)
    shapes = ((1, 4, 512, 128), (1, 1, 512, 128),
              (1, 16, 512, 128), (1, 2, 512, 128), (1, 2, 512, 128))
    values = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for shape in shapes]
    grad_output = torch.randn(1, 512, 16 * 128, device="cuda")

    inputs = [value.detach().clone().requires_grad_(True) for value in values]
    output, _unused_kl = flash_msa_func(*inputs, 256, 128**-0.5)
    loss = (output.float() * grad_output).sum()
    monkeypatch.setenv("MSA_BACKWARD_BACKEND", "sm90")
    loss.backward(retain_graph=True)
    native_grads = tuple(item.grad.detach().clone() for item in inputs)
    for item in inputs:
        item.grad = None
    monkeypatch.setenv("MSA_BACKWARD_BACKEND", "legacy")
    loss.backward()
    legacy_grads = tuple(item.grad.detach() for item in inputs)
    for index, (native, legacy) in enumerate(zip(native_grads, legacy_grads)):
        if index < 2:
            torch.testing.assert_close(native, legacy, rtol=0, atol=0)
        else:
            cosine = torch.nn.functional.cosine_similarity(
                native.float().flatten(), legacy.float().flatten(), dim=0
            )
            assert float(cosine) >= 0.999


@pytest.mark.parametrize("document_masking", [False, True])
def test_autograd_sm90_composes_main_and_proxy_backward(
    monkeypatch, document_masking: bool
) -> None:
    """The optimized main kernel composes with KL-only proxy gradients."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    monkeypatch.setenv("MSA_FORWARD_BACKEND", "sm90")
    monkeypatch.setenv("MSA_SELECT_BACKEND", "bf16")
    monkeypatch.setenv("MSA_KV_STORAGE", "bf16")
    torch.manual_seed(257 + int(document_masking))
    shapes = ((1, 4, 512, 128), (1, 1, 512, 128),
              (1, 16, 512, 128), (1, 2, 512, 128), (1, 2, 512, 128))
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
        for shape in shapes
    ]
    grad_output = torch.randn(1, 512, 16 * 128, device="cuda")
    documents = None
    if document_masking:
        documents = torch.empty(1, 512, device="cuda", dtype=torch.int32)
        documents[0, :93], documents[0, 93:287], documents[0, 287:] = 0, 1, 2

    output, kl_loss = flash_msa_func(
        *inputs, 256, 128**-0.5, document_list=documents
    )
    loss = (output.float() * grad_output).sum() + 1.7 * kl_loss.float()
    monkeypatch.setenv("MSA_BACKWARD_BACKEND", "sm90")
    loss.backward(retain_graph=True)
    native_grads = tuple(item.grad.detach().clone() for item in inputs)
    for item in inputs:
        item.grad = None
    monkeypatch.setenv("MSA_BACKWARD_BACKEND", "legacy")
    loss.backward()
    legacy_grads = tuple(item.grad.detach() for item in inputs)

    for native, legacy in zip(native_grads, legacy_grads):
        cosine = torch.nn.functional.cosine_similarity(
            native.float().flatten(), legacy.float().flatten(), dim=0
        )
        assert float(cosine) >= 0.999
        torch.testing.assert_close(native, legacy, rtol=2e-2, atol=2e-2)
