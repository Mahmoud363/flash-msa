"""Focused numerical gates for the Hopper-native backward tensor products."""

import pytest
import torch

from flash_msa.msa_backward_sm90 import wgmma_backward_dq


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

