"""GH200 tests for the native forward persistent scheduler."""

import pytest
import torch

from flash_msa.msa_forward_sm90 import (
    persistent_claim_work,
    tma_load_selected_kv,
    wgmma_selected_qk,
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
