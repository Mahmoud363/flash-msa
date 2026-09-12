"""Native Hopper KV-outer selected-attention building blocks.

The production forward remains in :mod:`sparse_flash_varlen` until the native
consumer passes both correctness and long-context performance gates.  This
module contains the independently testable persistent scheduler used by that
consumer.
"""

from __future__ import annotations

import inspect

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from cutlass._mlir.dialects import nvvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op


_COMPILE_CACHE = {}
_NVVM_ATOMICRMW_HAS_RES = "res" in inspect.signature(nvvm.atomicrmw).parameters


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16)


@dsl_user_op
def _atomic_claim(counter: cute.Tensor, *, loc=None, ip=None) -> Int32:
    """Atomically claim and return one monotonically increasing work index."""

    ptr = counter.iterator
    value = Int32(1).ir_value(loc=loc, ip=ip)
    if _NVVM_ATOMICRMW_HAS_RES:
        old = nvvm.atomicrmw(
            T.i32(), nvvm.AtomicOpKind.ADD, ptr.llvm_ptr, value, loc=loc, ip=ip
        )
    else:
        old = nvvm.atomicrmw(
            nvvm.AtomicOpKind.ADD, ptr.llvm_ptr, value, loc=loc, ip=ip
        )
    return Int32(old)


class _PersistentWorkClaimKernel:
    """Dynamic persistent scheduler shared by the SM90 forward mainloop."""

    def __init__(self, num_tasks: int, num_ctas: int) -> None:
        self.num_tasks = int(num_tasks)
        self.num_ctas = int(num_ctas)

    @cute.jit
    def __call__(
        self,
        counter: cute.Tensor,
        claimed_by: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(counter, claimed_by).launch(
            grid=[self.num_ctas, 1, 1], block=[128, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, counter: cute.Tensor, claimed_by: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        cta_idx, _, _ = cute.arch.block_idx()

        if tidx == 0:
            task_idx = _atomic_claim(counter)
            while task_idx < Int32(self.num_tasks):
                claimed_by[task_idx] = cta_idx
                task_idx = _atomic_claim(counter)


def persistent_claim_work(num_tasks: int, *, device: torch.device | str) -> torch.Tensor:
    """Return the CTA owner of every dynamically claimed work item.

    This is a scheduler validation hook, not part of the public attention API.
    The attention consumer will put its TMA/WGMMA mainloop where ``claimed_by``
    is currently written.
    """

    num_tasks = int(num_tasks)
    if num_tasks < 0:
        raise ValueError("num_tasks must be nonnegative")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("persistent SM90 scheduling requires a CUDA device")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("persistent SM90 scheduling requires compute capability 9.0")
    if num_tasks == 0:
        return torch.empty(0, dtype=torch.int32, device=device)

    counter = torch.zeros(1, dtype=torch.int32, device=device)
    claimed_by = torch.full((num_tasks,), -1, dtype=torch.int32, device=device)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    num_ctas = min(num_tasks, 2 * int(sm_count))
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    counter_cute = _to_cute_tensor(counter)
    claimed_by_cute = _to_cute_tensor(claimed_by)
    key = (num_tasks, num_ctas, counter_cute.element_type, claimed_by_cute.element_type)
    if key not in _COMPILE_CACHE:
        kernel = _PersistentWorkClaimKernel(num_tasks, num_ctas)
        _COMPILE_CACHE[key] = cute.compile(
            kernel, counter_cute, claimed_by_cute, stream
        )
    _COMPILE_CACHE[key](counter_cute, claimed_by_cute, stream)
    return claimed_by


__all__ = ["persistent_claim_work"]
