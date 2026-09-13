"""GH200 tests for the native forward persistent scheduler."""

import pytest
import torch

from flash_msa.msa_forward_sm90 import (
    persistent_claim_work,
    tma_load_selected_kv,
    wgmma_selected_qk,
    wgmma_selected_softmax,
    wgmma_selected_attention,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_tasks", [1, 113, 4097])
def test_every_work_item_is_claimed(num_tasks: int) -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    owners = persistent_claim_work(num_tasks, device="cuda")
    torch.cuda.synchronize()
    assert owners.shape == (num_tasks,)
    assert owners.dtype == torch.int32
    assert bool((owners >= 0).all())
    assert int(owners.max()) < min(
        num_tasks, 2 * torch.cuda.get_device_properties(0).multi_processor_count
    )


def test_empty_worklist() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    assert persistent_claim_work(0, device="cuda").numel() == 0


def test_selected_kv_tiles_are_loaded_with_tma() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    torch.manual_seed(19)
    k = torch.randn(2, 2, 512, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    # [batch, proxy head, key block, query count, edge offset]
    task_meta = torch.tensor(
        [[0, 0, 2, 7, 0], [1, 3, 1, 11, 7]],
        dtype=torch.int32,
        device="cuda",
    )
    copied_k, copied_v = tma_load_selected_kv(
        k, v, task_meta, n_proxy_heads=4
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(copied_k[0], k[0, 0, 256:384], rtol=0, atol=0)
    torch.testing.assert_close(copied_v[0], v[0, 0, 256:384], rtol=0, atol=0)
    torch.testing.assert_close(copied_k[1], k[1, 1, 128:256], rtol=0, atol=0)
    torch.testing.assert_close(copied_v[1], v[1, 1, 128:256], rtol=0, atol=0)


def test_selected_qk_uses_wgmma_fp32_accumulation() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    torch.manual_seed(23)
    q = torch.randn(1, 16, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 512, 128, device="cuda", dtype=torch.bfloat16)
    qids = torch.arange(129, 145, dtype=torch.int32, device="cuda")
    task_meta = torch.tensor(
        [[0, 1, 1, 16, 0]], dtype=torch.int32, device="cuda"
    )
    scores = wgmma_selected_qk(
        q, k, task_meta, qids, n_proxy_heads=4
    )
    torch.cuda.synchronize()
    gathered_q = q[0, 4:8, qids.long()].permute(1, 0, 2).reshape(64, 128)
    reference = gathered_q.float() @ k[0, 0, 128:256].float().T
    torch.testing.assert_close(scores[0], reference, rtol=2e-3, atol=0.2)


def test_selected_qk_softmax_stays_in_fp32() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    torch.manual_seed(29)
    q = torch.randn(1, 16, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 512, 128, device="cuda", dtype=torch.bfloat16)
    qids = torch.arange(257, 273, dtype=torch.int32, device="cuda")
    task_meta = torch.tensor(
        [[0, 2, 0, 16, 0]], dtype=torch.int32, device="cuda"
    )
    scale = 128**-0.5
    probabilities, lse = wgmma_selected_softmax(
        q, k, task_meta, qids, n_proxy_heads=4, scale=scale
    )
    torch.cuda.synchronize()
    gathered_q = q[0, 8:12, qids.long()].permute(1, 0, 2).reshape(64, 128)
    logits = (gathered_q.float() @ k[0, 1, :128].float().T) * scale
    torch.testing.assert_close(
        probabilities[0], logits.softmax(dim=-1), rtol=3e-3, atol=3e-4
    )
    torch.testing.assert_close(
        lse.reshape(-1), logits.logsumexp(dim=-1), rtol=3e-3, atol=3e-3
    )


def test_selected_attention_uses_pv_wgmma() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    torch.manual_seed(31)
    q = torch.randn(1, 16, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 512, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qids = torch.arange(321, 337, dtype=torch.int32, device="cuda")
    task_meta = torch.tensor(
        [[0, 3, 2, 16, 0]], dtype=torch.int32, device="cuda"
    )
    scale = 128**-0.5
    output, lse = wgmma_selected_attention(
        q, k, v, task_meta, qids, n_proxy_heads=4, scale=scale
    )
    torch.cuda.synchronize()
    gathered_q = q[0, 12:16, qids.long()].permute(1, 0, 2).reshape(64, 128)
    logits = (gathered_q.float() @ k[0, 1, 256:384].float().T) * scale
    reference = logits.softmax(dim=-1) @ v[0, 1, 256:384].float()
    torch.testing.assert_close(
        output.reshape(64, 128).float(), reference, rtol=1e-2, atol=2e-2
    )
    torch.testing.assert_close(
        lse.reshape(-1), logits.logsumexp(dim=-1), rtol=3e-3, atol=3e-3
    )


def test_segmented_selected_attention_predicates_document_boundaries() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    torch.manual_seed(33)
    q = torch.randn(1, 16, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 512, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qids = torch.arange(200, 216, dtype=torch.int32, device="cuda")
    task_meta = torch.tensor([[0, 1, 0, 16, 0]], dtype=torch.int32, device="cuda")
    starts = torch.tensor([17], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([43], dtype=torch.int32, device="cuda")
    scale = 128**-0.5
    output, lse = wgmma_selected_attention(
        q,
        k,
        v,
        task_meta,
        qids,
        n_proxy_heads=4,
        scale=scale,
        segment_starts=starts,
        segment_lengths=lengths,
    )
    gathered_q = q[0, 4:8, qids.long()].permute(1, 0, 2).reshape(64, 128)
    logits = (gathered_q.float() @ k[0, 0, 17:60].float().T) * scale
    reference = logits.softmax(dim=-1) @ v[0, 0, 17:60].float()
    torch.testing.assert_close(
        output.reshape(64, 128).float(), reference, rtol=1e-2, atol=2e-2
    )
    torch.testing.assert_close(
        lse.reshape(-1), logits.logsumexp(dim=-1), rtol=3e-3, atol=3e-3
    )


@pytest.mark.parametrize(
    ("n_heads", "n_proxy_heads", "n_kv_heads"), [(8, 4, 2), (16, 2, 2)]
)
def test_selected_attention_derives_query_group_width(
    n_heads: int, n_proxy_heads: int, n_kv_heads: int
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    torch.manual_seed(37 + n_heads)
    q = torch.randn(1, n_heads, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, n_kv_heads, 512, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    main_per_proxy = n_heads // n_proxy_heads
    queries_per_tile = 64 // main_per_proxy
    qids = torch.arange(256, 256 + queries_per_tile, dtype=torch.int32, device="cuda")
    task_meta = torch.tensor(
        [[0, 1, 0, queries_per_tile, 0]], dtype=torch.int32, device="cuda"
    )
    scale = 128**-0.5
    output, lse = wgmma_selected_attention(
        q, k, v, task_meta, qids, n_proxy_heads=n_proxy_heads, scale=scale
    )
    proxy_per_kv = n_proxy_heads // n_kv_heads
    kv_head = 1 // proxy_per_kv
    head_start = main_per_proxy
    gathered_q = q[0, head_start : head_start + main_per_proxy, qids.long()]
    gathered_q = gathered_q.permute(1, 0, 2).reshape(64, 128)
    logits = (gathered_q.float() @ k[0, kv_head, :128].float().T) * scale
    reference = logits.softmax(dim=-1) @ v[0, kv_head, :128].float()
    torch.testing.assert_close(
        output.reshape(64, 128).float(), reference, rtol=1e-2, atol=2e-2
    )
    torch.testing.assert_close(
        lse.reshape(-1), logits.logsumexp(dim=-1), rtol=3e-3, atol=3e-3
    )


def test_kv_work_item_reuses_tile_across_query_groups() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    torch.manual_seed(61)
    q = torch.randn(1, 16, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 512, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qids = torch.arange(300, 340, dtype=torch.int32, device="cuda")
    task_meta = torch.tensor([[0, 1, 1, 40, 0]], dtype=torch.int32, device="cuda")
    scale = 128**-0.5
    output, lse = wgmma_selected_attention(
        q, k, v, task_meta, qids, n_proxy_heads=4, scale=scale
    )
    gathered_q = q[0, 4:8, qids.long()].permute(1, 0, 2)
    logits = torch.einsum(
        "qhd,kd->qhk", gathered_q.float(), k[0, 0, 128:256].float()
    ) * scale
    reference = torch.einsum(
        "qhk,kd->qhd", logits.softmax(dim=-1), v[0, 0, 128:256].float()
    )
    torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(lse, logits.logsumexp(dim=-1), rtol=3e-3, atol=3e-3)


def test_persistent_attention_reclaims_multiple_tasks(monkeypatch) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 test")
    monkeypatch.setenv("MSA_SM90_MAX_CTAS", "1")
    torch.manual_seed(67)
    q = torch.randn(1, 16, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 512, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qids = torch.cat(
        [
            torch.arange(32, 48, device="cuda", dtype=torch.int32),
            torch.arange(192, 208, device="cuda", dtype=torch.int32),
            torch.arange(352, 368, device="cuda", dtype=torch.int32),
        ]
    )
    task_meta = torch.tensor(
        [[0, 0, 0, 16, 0], [0, 1, 2, 16, 16], [0, 3, 1, 16, 32]],
        device="cuda",
        dtype=torch.int32,
    )
    scale = 128**-0.5
    output, lse = wgmma_selected_attention(
        q, k, v, task_meta, qids, n_proxy_heads=4, scale=scale
    )
    for task, (proxy_head, key_block, edge_offset) in enumerate(
        [(0, 0, 0), (1, 2, 16), (3, 1, 32)]
    ):
        task_qids = qids[edge_offset : edge_offset + 16].long()
        gathered_q = q[0, proxy_head * 4 : proxy_head * 4 + 4, task_qids]
        gathered_q = gathered_q.permute(1, 0, 2)
        kv_head = proxy_head // 2
        kv_slice = slice(key_block * 128, (key_block + 1) * 128)
        logits = torch.einsum(
            "qhd,kd->qhk", gathered_q.float(), k[0, kv_head, kv_slice].float()
        ) * scale
        reference = torch.einsum(
            "qhk,kd->qhd", logits.softmax(dim=-1), v[0, kv_head, kv_slice].float()
        )
        actual = slice(edge_offset, edge_offset + 16)
        torch.testing.assert_close(
            output[actual].float(), reference, rtol=1e-2, atol=2e-2
        )
        torch.testing.assert_close(
            lse[actual], logits.logsumexp(dim=-1), rtol=3e-3, atol=3e-3
        )
