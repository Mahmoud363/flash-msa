"""GH200 tests for the native forward persistent scheduler."""

import pytest
import torch

from flash_msa.msa_forward_sm90 import persistent_claim_work


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
