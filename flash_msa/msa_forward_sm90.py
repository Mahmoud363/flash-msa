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
from cutlass import pipeline
from cutlass.cute.nvgpu import warpgroup


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


class _SelectedKVTmaKernel:
    """Validation kernel for the selected-tile TMA data path."""

    tile_size = 128

    def __init__(self, num_tasks: int, proxy_heads_per_kv: int) -> None:
        self.num_tasks = int(num_tasks)
        self.proxy_heads_per_kv = int(proxy_heads_per_kv)

    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,
        v: cute.Tensor,
        task_meta: cute.Tensor,
        copied_k: cute.Tensor,
        copied_v: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # Reinterpret contiguous [B,H,S,D] storage as [S,D,H,B] so every
        # selected [128,128] token/dimension tile is a legal TMA rectangle.
        batch, n_kv_heads, seq_len, head_dim = k.shape
        kv_layout = cute.make_layout(
            (seq_len, head_dim, n_kv_heads, batch),
            stride=(
                head_dim,
                1,
                seq_len * head_dim,
                n_kv_heads * seq_len * head_dim,
            ),
        )
        k = cute.make_tensor(k.iterator, kv_layout)
        v = cute.make_tensor(v.iterator, kv_layout)
        dtype = k.element_type
        self._dtype = dtype
        smem_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128, dtype
        )
        smem_layout_staged = cute.tile_to_shape(
            smem_atom, (self.tile_size, self.tile_size, 1), (0, 1, 2)
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_k, tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op, k, smem_layout, (self.tile_size, self.tile_size)
        )
        tma_v, tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op, v, smem_layout, (self.tile_size, self.tile_size)
        )

        @cute.struct
        class SharedStorage:
            k_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            v_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            sK: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(smem_layout_staged)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(smem_layout_staged)], 1024
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_k,
            tensor_k,
            tma_v,
            tensor_v,
            task_meta,
            copied_k,
            copied_v,
            smem_layout_staged,
        ).launch(
            grid=[self.num_tasks, 1, 1],
            block=[128, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_k: cute.CopyAtom,
        k: cute.Tensor,
        tma_v: cute.CopyAtom,
        v: cute.Tensor,
        task_meta: cute.Tensor,
        copied_k: cute.Tensor,
        copied_v: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        task_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sK = storage.sK.get_tensor(
            smem_layout_staged.outer, swizzle=smem_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            smem_layout_staged.outer, swizzle=smem_layout_staged.inner
        )

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        k_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_barriers.data_ptr(),
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(
                self._dtype, cute.slice_(smem_layout_staged, (None, None, 0))
            ),
        )
        v_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.v_barriers.data_ptr(),
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(
                self._dtype, cute.slice_(smem_layout_staged, (None, None, 0))
            ),
        )

        batch = task_meta[task_idx, 0]
        proxy_head = task_meta[task_idx, 1]
        key_block = task_meta[task_idx, 2]
        kv_head = proxy_head // Int32(self.proxy_heads_per_kv)

        gK = cute.local_tile(
            k[None, None, kv_head, batch],
            (self.tile_size, self.tile_size),
            (None, 0),
        )
        gV = cute.local_tile(
            v[None, None, kv_head, batch],
            (self.tile_size, self.tile_size),
            (None, 0),
        )
        cta_layout = cute.make_layout(1)
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_k, 0, cta_layout, cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2)
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_v, 0, cta_layout, cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2)
        )

        k_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        v_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        k_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        v_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

        if warp_idx == 0:
            k_pipe.producer_acquire(k_producer)
            cute.copy(
                tma_k,
                tKgK[(None, key_block)],
                tKsK[(None, 0)],
                tma_bar_ptr=k_pipe.producer_get_barrier(k_producer),
            )
            k_pipe.producer_commit(k_producer)
            v_pipe.producer_acquire(v_producer)
            cute.copy(
                tma_v,
                tVgV[(None, key_block)],
                tVsV[(None, 0)],
                tma_bar_ptr=v_pipe.producer_get_barrier(v_producer),
            )
            v_pipe.producer_commit(v_producer)

        k_pipe.consumer_wait(k_consumer)
        v_pipe.consumer_wait(v_consumer)
        linear = tidx
        while linear < Int32(self.tile_size * self.tile_size):
            row = linear // Int32(self.tile_size)
            col = linear - row * Int32(self.tile_size)
            copied_k[task_idx, row, col] = sK[row, col, 0]
            copied_v[task_idx, row, col] = sV[row, col, 0]
            linear += Int32(128)


def tma_load_selected_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    task_meta: torch.Tensor,
    *,
    n_proxy_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy selected fixed-length K/V tiles through Hopper TMA for validation."""

    if k.shape != v.shape or k.ndim != 4 or k.shape[-1] != 128:
        raise ValueError("k and v must have matching [B, Hkv, S, 128] shapes")
    if k.dtype not in (torch.float16, torch.bfloat16) or k.device.type != "cuda":
        raise TypeError("TMA K/V inputs must be CUDA fp16/bf16 tensors")
    if task_meta.device != k.device or task_meta.dtype != torch.int32:
        raise TypeError("task_meta must be an int32 tensor on the K/V device")
    if task_meta.ndim != 2 or task_meta.shape[1] != 5:
        raise ValueError("task_meta must have shape [T, 5]")
    if n_proxy_heads % k.shape[1]:
        raise ValueError("proxy heads must be divisible by KV heads")
    if torch.cuda.get_device_capability(k.device) != (9, 0):
        raise RuntimeError("selected K/V TMA requires compute capability 9.0")

    num_tasks = int(task_meta.shape[0])
    copied_k = torch.empty((num_tasks, 128, 128), dtype=k.dtype, device=k.device)
    copied_v = torch.empty_like(copied_k)
    if num_tasks == 0:
        return copied_k, copied_v

    batch, n_kv_heads, seq_len, head_dim = map(int, k.shape)
    k_cute = _to_cute_tensor(k.detach().contiguous())
    v_cute = _to_cute_tensor(v.detach().contiguous())
    task_cute = _to_cute_tensor(task_meta.contiguous())
    copied_k_cute = _to_cute_tensor(copied_k)
    copied_v_cute = _to_cute_tensor(copied_v)
    stream = cuda.CUstream(torch.cuda.current_stream(k.device).cuda_stream)
    key = (
        "selected_kv_tma",
        num_tasks,
        int(n_proxy_heads) // n_kv_heads,
        k_cute.element_type,
    )
    if key not in _COMPILE_CACHE:
        kernel = _SelectedKVTmaKernel(num_tasks, int(n_proxy_heads) // n_kv_heads)
        _COMPILE_CACHE[key] = cute.compile(
            kernel,
            k_cute,
            v_cute,
            task_cute,
            copied_k_cute,
            copied_v_cute,
            stream,
        )
    _COMPILE_CACHE[key](
        k_cute, v_cute, task_cute, copied_k_cute, copied_v_cute, stream
    )
    return copied_k, copied_v


__all__ = ["persistent_claim_work", "tma_load_selected_kv"]
