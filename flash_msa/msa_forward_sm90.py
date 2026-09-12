"""Native Hopper KV-outer selected-attention building blocks.

The fixed-length SM90 forward can dispatch this consumer directly; focused
validation hooks expose its scheduler, TMA loads, WGMMA math, and epilogue.
"""

from __future__ import annotations

import inspect
import os

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from cutlass._mlir.dialects import nvvm
from cutlass._mlir.dialects import math as _math
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass import pipeline
from cutlass.cute.nvgpu import cpasync, warpgroup
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils


_COMPILE_CACHE = {}
_NVVM_ATOMICRMW_HAS_RES = "res" in inspect.signature(nvvm.atomicrmw).parameters
PRODUCER_REGISTERS = int(os.environ.get("MSA_SM90_PRODUCER_REGISTERS", "40"))
CONSUMER_REGISTERS = int(os.environ.get("MSA_SM90_CONSUMER_REGISTERS", "232"))
Q_PIPELINE_STAGES = int(os.environ.get("MSA_SM90_Q_PIPELINE_STAGES", "2"))
if Q_PIPELINE_STAGES not in (1, 2):
    raise ValueError("MSA_SM90_Q_PIPELINE_STAGES must be 1 or 2")


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16)


@dsl_user_op
def _elem_pointer(tensor: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None):
    return tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)


def _forward_head_tiling(
    n_heads: int, n_kv_heads: int, n_proxy_heads: int
) -> tuple[int, int, int]:
    if n_heads % n_proxy_heads or n_proxy_heads % n_kv_heads:
        raise ValueError("invalid main/proxy/KV head divisibility")
    main_per_proxy = int(n_heads) // int(n_proxy_heads)
    if main_per_proxy > 64 or 64 % main_per_proxy:
        raise ValueError("main heads per proxy must divide the 64-row WGMMA tile")
    return (
        main_per_proxy,
        64 // main_per_proxy,
        int(n_proxy_heads) // int(n_kv_heads),
    )


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


class _FP8SelectedKVTmaKernel:
    """Validate E4M3 TMA loads followed by scaled BF16 tile conversion."""

    tile_size = 128

    def __init__(self, num_tasks: int, proxy_heads_per_kv: int) -> None:
        self.num_tasks = int(num_tasks)
        self.proxy_heads_per_kv = int(proxy_heads_per_kv)

    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,
        v: cute.Tensor,
        k_scale: cute.Tensor,
        v_scale: cute.Tensor,
        task_meta: cute.Tensor,
        copied_k: cute.Tensor,
        copied_v: cute.Tensor,
        stream: cuda.CUstream,
    ):
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
        fp8_dtype = k.element_type
        output_dtype = copied_k.element_type
        self._fp8_dtype = fp8_dtype
        self._output_dtype = output_dtype
        smem_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128, fp8_dtype
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
        self._tma_bytes = cute.size_in_bytes(fp8_dtype, smem_layout)

        @cute.struct
        class SharedStorage:
            k_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            v_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            sK: cute.struct.Align[
                cute.struct.MemRange[fp8_dtype, cute.cosize(smem_layout_staged)],
                1024,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[fp8_dtype, cute.cosize(smem_layout_staged)],
                1024,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_k,
            tensor_k,
            tma_v,
            tensor_v,
            k_scale,
            v_scale,
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
        k_scale: cute.Tensor,
        v_scale: cute.Tensor,
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
            tx_count=self._tma_bytes,
        )
        v_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.v_barriers.data_ptr(),
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self._tma_bytes,
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
        k_value_scale = k_scale[batch, kv_head, key_block]
        v_value_scale = v_scale[batch, kv_head, key_block]
        # SM90 converts packed FP8 lanes; scalar E4M3 extension is not a legal
        # instruction. Keep four adjacent values in registers for each cvt.
        linear = tidx * Int32(4)
        while linear < Int32(self.tile_size * self.tile_size):
            packed_k = cute.make_rmem_tensor((4,), self._fp8_dtype)
            packed_v = cute.make_rmem_tensor((4,), self._fp8_dtype)
            restored_k = cute.make_rmem_tensor((4,), cutlass.Float32)
            restored_v = cute.make_rmem_tensor((4,), cutlass.Float32)
            for element in cutlass.range_constexpr(4):
                element_linear = linear + Int32(element)
                row = element_linear // Int32(self.tile_size)
                col = element_linear - row * Int32(self.tile_size)
                packed_k[element] = sK[row, col, 0]
                packed_v[element] = sV[row, col, 0]
            restored_k.store(packed_k.load().to(cutlass.Float32))
            restored_v.store(packed_v.load().to(cutlass.Float32))
            for element in cutlass.range_constexpr(4):
                element_linear = linear + Int32(element)
                row = element_linear // Int32(self.tile_size)
                col = element_linear - row * Int32(self.tile_size)
                copied_k[task_idx, row, col] = self._output_dtype(
                    restored_k[element] * k_value_scale
                )
                copied_v[task_idx, row, col] = self._output_dtype(
                    restored_v[element] * v_value_scale
                )
            linear += Int32(128 * 4)


def tma_load_fp8_kv_as_bf16(
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    task_meta: torch.Tensor,
    *,
    n_proxy_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TMA-load selected E4M3 tiles and restore them to BF16."""

    if k.shape != v.shape or k.ndim != 4 or k.shape[-1] != 128:
        raise ValueError("k and v must have matching [B,Hkv,S,128] shapes")
    if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
        raise TypeError("FP8 TMA K/V inputs must use float8_e4m3fn")
    expected_scales = (k.shape[0], k.shape[1], k.shape[2] // 128)
    if (
        k_scale.shape != expected_scales
        or v_scale.shape != expected_scales
        or k_scale.dtype != torch.float32
        or v_scale.dtype != torch.float32
    ):
        raise ValueError(f"K/V scales must be FP32 with shape {expected_scales}")
    if task_meta.device != k.device or task_meta.dtype != torch.int32:
        raise TypeError("task_meta must be an int32 tensor on the K/V device")
    if task_meta.ndim != 2 or task_meta.shape[1] != 5:
        raise ValueError("task_meta must have shape [T,5]")
    if n_proxy_heads % k.shape[1]:
        raise ValueError("proxy heads must be divisible by KV heads")
    if torch.cuda.get_device_capability(k.device) != (9, 0):
        raise RuntimeError("FP8 selected K/V TMA requires compute capability 9.0")

    num_tasks = int(task_meta.shape[0])
    copied_k = torch.empty((num_tasks, 128, 128), dtype=torch.bfloat16, device=k.device)
    copied_v = torch.empty_like(copied_k)
    if num_tasks == 0:
        return copied_k, copied_v
    k_cute = _to_cute_tensor(k.detach().contiguous())
    v_cute = _to_cute_tensor(v.detach().contiguous())
    ks_cute = _to_cute_tensor(k_scale.detach().contiguous())
    vs_cute = _to_cute_tensor(v_scale.detach().contiguous())
    task_cute = _to_cute_tensor(task_meta.contiguous())
    copied_k_cute = _to_cute_tensor(copied_k)
    copied_v_cute = _to_cute_tensor(copied_v)
    stream = cuda.CUstream(torch.cuda.current_stream(k.device).cuda_stream)
    key = (
        "selected_fp8_kv_tma",
        num_tasks,
        int(n_proxy_heads) // int(k.shape[1]),
    )
    if key not in _COMPILE_CACHE:
        kernel = _FP8SelectedKVTmaKernel(
            num_tasks, int(n_proxy_heads) // int(k.shape[1])
        )
        _COMPILE_CACHE[key] = cute.compile(
            kernel,
            k_cute,
            v_cute,
            ks_cute,
            vs_cute,
            task_cute,
            copied_k_cute,
            copied_v_cute,
            stream,
        )
    _COMPILE_CACHE[key](
        k_cute,
        v_cute,
        ks_cute,
        vs_cute,
        task_cute,
        copied_k_cute,
        copied_v_cute,
        stream,
    )
    return copied_k, copied_v


class _SelectedQKWgmmaKernel:
    """One selected 64x128 QK tile using warp-specialized TMA/WGMMA."""

    rows = 64
    block = 128

    def __init__(
        self,
        num_tasks: int,
        main_per_proxy: int,
        proxy_per_kv: int,
        *,
        write_debug: bool = False,
        persistent: bool = False,
        num_ctas: int | None = None,
        fp8_kv: bool = False,
        segmented: bool = False,
    ) -> None:
        self.num_tasks = int(num_tasks)
        self.main_per_proxy = int(main_per_proxy)
        self.proxy_per_kv = int(proxy_per_kv)
        self.write_debug = bool(write_debug)
        self.persistent = bool(persistent)
        self.fp8_kv = bool(fp8_kv)
        self.segmented = bool(segmented)
        self.num_ctas = int(num_ctas if num_ctas is not None else num_tasks)
        self.claims_per_cta = (self.num_tasks + self.num_ctas - 1) // self.num_ctas

    @staticmethod
    @cute.jit
    def _gemm_zero(tiled_mma, a, b, acc):
        for k_block in range(cute.size(a, mode=[2]), unroll_full=True):
            tiled_mma.set(warpgroup.Field.ACCUMULATE, k_block != 0)
            cute.gemm(
                tiled_mma,
                acc,
                a[None, None, k_block],
                b[None, None, k_block],
                acc,
            )

    @staticmethod
    def _layout_separate(threshold, source, reference):
        lower = cute.make_layout(())
        upper = cute.make_layout(())
        for index, value in enumerate(reference):
            if cutlass.const_expr(value < threshold):
                lower = cute.append(lower, source[index])
            else:
                upper = cute.append(upper, source[index])
        if cutlass.const_expr(cute.rank(lower) == 1):
            return cute.append(lower, upper)
        return cute.append(cute.append(cute.make_layout(()), lower), upper)

    @cute.jit
    def _layout_acc_mn(self, tiled_mma, accumulator):
        separated = self._layout_separate(
            tiled_mma.shape_mnk[0],
            accumulator[0],
            tiled_mma.tv_layout_C.stride[1],
        )
        values_m, values_n = separated[0], separated[1]
        if cutlass.const_expr(cute.rank(values_m) == 1):
            values_m = cute.append(values_m, accumulator[1])
        else:
            values_m = cute.append(
                cute.append(cute.make_layout(()), values_m), accumulator[1]
            )
        if cutlass.const_expr(cute.rank(values_n) == 1):
            values_n = cute.append(values_n, accumulator[2])
        else:
            values_n = cute.append(
                cute.append(cute.make_layout(()), values_n), accumulator[2]
            )
        if cutlass.const_expr(cute.rank(values_m) == 1):
            return cute.append(values_m, values_n)
        return cute.append(cute.append(cute.make_layout(()), values_m), values_n)

    @cute.jit
    def _reduction_target_n(self, tiled_mma):
        separated = self._layout_separate(
            tiled_mma.shape_mnk[0],
            cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
            tiled_mma.tv_layout_C.stride[0],
        )
        return separated[1]

    @cute.jit
    def _softmax_fp32(self, acc, tiled_mma, row_scale_log2):
        """Normalize one WGMMA score tile and return per-row max/sum."""

        acc_mn = cute.make_tensor(
            acc.iterator, self._layout_acc_mn(tiled_mma, acc.layout)
        )
        row_shape = cute.make_layout(cute.size(acc_mn, mode=[0]))
        row_max = cute.make_rmem_tensor_like(row_shape, cutlass.Float32)
        row_sum = cute.make_rmem_tensor_like(row_shape, cutlass.Float32)
        reduction_target = self._reduction_target_n(tiled_mma)
        reduction_rank = cute.rank(reduction_target)
        for row in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
            row_max[row] = acc_mn[row, 0]
            for col in cutlass.range_constexpr(1, cute.size(acc_mn, mode=[1])):
                row_max[row] = cute.arch.fmax(row_max[row], acc_mn[row, col])
            for reduction in cutlass.range_constexpr(reduction_rank):
                row_max[row] = cute.arch.warp_reduction_max(
                    row_max[row], threads_in_group=reduction_target.shape[reduction]
                )
            scaled_max = row_scale_log2[row] * row_max[row]
            for col in cutlass.range_constexpr(cute.size(acc_mn, mode=[1])):
                acc_mn[row, col] = cute.math.exp2(
                    row_scale_log2[row] * acc_mn[row, col] - scaled_max,
                    fastmath=True,
                )
            row_sum[row] = acc_mn[row, None].load().reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
            for reduction in cutlass.range_constexpr(reduction_rank):
                row_sum[row] = cute.arch.warp_reduction_sum(
                    row_sum[row], threads_in_group=reduction_target.shape[reduction]
                )
            inv_sum = cute.arch.rcp_approx(row_sum[row])
            for col in cutlass.range_constexpr(cute.size(acc_mn, mode=[1])):
                acc_mn[row, col] *= inv_sum
        return row_max, row_sum

    @staticmethod
    def _convert_c_layout_to_a_layout(c_layout, a_layout):
        return cute.make_layout(
            (
                a_layout,
                c_layout.shape[1],
                (c_layout.shape[2], cute.size(c_layout, mode=[0]) // cute.size(a_layout)),
            ),
            stride=(
                c_layout.stride[0],
                c_layout.stride[1],
                (
                    c_layout.stride[2],
                    cute.size(a_layout, mode=[2]) * c_layout.stride[0][2],
                ),
            ),
        )

    @cute.jit
    def _accumulator_to_operand(self, accumulator, operand_layout_tv):
        operand = cute.make_rmem_tensor_like(
            self._convert_c_layout_to_a_layout(
                accumulator.layout, operand_layout_tv.shape[1]
            ),
            self._pv_dtype,
        )
        operand_as_accumulator = cute.make_tensor(operand.iterator, accumulator.layout)
        operand_as_accumulator.store(accumulator.load().to(self._pv_dtype))
        return operand

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        task_meta: cute.Tensor,
        query_indices: cute.Tensor,
        scores: cute.Tensor,
        probabilities: cute.Tensor,
        lse: cute.Tensor,
        output: cute.Tensor,
        work_counter: cute.Tensor,
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
        v_scale: cute.Tensor,
        segment_starts: cute.Tensor,
        segment_lengths: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        batch, n_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape
        q_layout = cute.make_layout(
            (seq_len, head_dim, n_heads, batch),
            stride=(
                head_dim,
                1,
                seq_len * head_dim,
                n_heads * seq_len * head_dim,
            ),
        )
        kv_layout = cute.make_layout(
            (seq_len, head_dim, n_kv_heads, batch),
            stride=(
                head_dim,
                1,
                seq_len * head_dim,
                n_kv_heads * seq_len * head_dim,
            ),
        )
        q = cute.make_tensor(q.iterator, q_layout)
        k = cute.make_tensor(k.iterator, kv_layout)
        v_layout = cute.make_layout(
            (head_dim, seq_len, n_kv_heads, batch),
            stride=(
                1,
                head_dim,
                seq_len * head_dim,
                n_kv_heads * seq_len * head_dim,
            ),
        )
        v = cute.make_tensor(v.iterator, v_layout)
        dtype = q.element_type
        pv_dtype = output.element_type
        q_layout_enum = utils.LayoutEnum.from_tensor(q)
        k_layout_enum = utils.LayoutEnum.from_tensor(k)
        v_layout_enum = utils.LayoutEnum.from_tensor(v)
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            q_layout_enum.sm90_mma_major_mode(),
            k_layout_enum.sm90_mma_major_mode(),
            cutlass.Float32,
            (1, 1, 1),
            (self.rows, self.block),
        )
        pv_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            pv_dtype,
            pv_dtype,
            warpgroup.OperandMajorMode.K,
            v_layout_enum.sm90_mma_major_mode(),
            cutlass.Float32,
            (1, 1, 1),
            (self.rows, self.block),
            warpgroup.OperandSource.RMEM,
        )
        q_smem_staged = sm90_utils.make_smem_layout_a(
            q_layout_enum,
            (self.rows, self.block, head_dim),
            dtype,
            Q_PIPELINE_STAGES,
        )
        q_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=128,
        )
        q_copy_elements = 128 // dtype.width
        q_threads_per_row = self.block // q_copy_elements
        q_gmem_copy = cute.make_tiled_copy_tv(
            q_copy_atom,
            cute.make_ordered_layout(
                (128 // q_threads_per_row, q_threads_per_row), order=(1, 0)
            ),
            cute.make_layout((1, q_copy_elements)),
        )
        self._q_copy_elements = q_copy_elements
        k_smem_staged = sm90_utils.make_smem_layout_b(
            k_layout_enum, (self.rows, self.block, head_dim), dtype, 1
        )
        v_smem_staged = sm90_utils.make_smem_layout_b(
            v_layout_enum, (self.rows, self.block, self.block), pv_dtype, 1
        )
        kv_storage_dtype = k.element_type
        if cutlass.const_expr(self.fp8_kv):
            k_tma_smem_staged = k_smem_staged
            v_tma_smem_staged = sm90_utils.make_smem_layout_b(
                v_layout_enum,
                (self.rows, self.block, self.block),
                kv_storage_dtype,
                1,
            )
        else:
            k_tma_smem_staged = k_smem_staged
            v_tma_smem_staged = v_smem_staged
        k_smem = cute.slice_(k_tma_smem_staged, (None, None, 0))
        tma_k, tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            k,
            k_smem,
            (self.block, head_dim),
        )
        v_smem = cute.slice_(v_tma_smem_staged, (None, None, 0))
        tma_v, tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            v,
            v_smem,
            (self.block, self.block),
        )
        self._dtype = dtype
        self._pv_dtype = pv_dtype
        self._kv_storage_dtype = kv_storage_dtype
        self._tma_k_bytes = cute.size_in_bytes(kv_storage_dtype, k_smem)
        self._tma_v_bytes = cute.size_in_bytes(kv_storage_dtype, v_smem)

        if cutlass.const_expr(self.fp8_kv):
            @cute.struct
            class SharedStorage:
                k_barriers: cute.struct.MemRange[cutlass.Int64, 2]
                v_barriers: cute.struct.MemRange[cutlass.Int64, 2]
                q_barriers: cute.struct.MemRange[
                    cutlass.Int64, 2 * Q_PIPELINE_STAGES
                ]
                task_index: cute.struct.MemRange[cutlass.Int32, 1]
                sQ: cute.struct.Align[
                    cute.struct.MemRange[dtype, cute.cosize(q_smem_staged)], 1024
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[dtype, cute.cosize(k_smem_staged)], 1024
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[pv_dtype, cute.cosize(v_smem_staged)], 1024
                ]
                sV8: cute.struct.Align[
                    cute.struct.MemRange[
                        kv_storage_dtype, cute.cosize(v_tma_smem_staged)
                    ],
                    1024,
                ]
        else:
            @cute.struct
            class SharedStorage:
                k_barriers: cute.struct.MemRange[cutlass.Int64, 2]
                v_barriers: cute.struct.MemRange[cutlass.Int64, 2]
                q_barriers: cute.struct.MemRange[
                    cutlass.Int64, 2 * Q_PIPELINE_STAGES
                ]
                task_index: cute.struct.MemRange[cutlass.Int32, 1]
                sQ: cute.struct.Align[
                    cute.struct.MemRange[dtype, cute.cosize(q_smem_staged)], 1024
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[dtype, cute.cosize(k_smem_staged)], 1024
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[pv_dtype, cute.cosize(v_smem_staged)], 1024
                ]

        self.shared_storage = SharedStorage
        self.kernel(
            q,
            tma_k,
            tensor_k,
            tma_v,
            tensor_v,
            task_meta,
            query_indices,
            scores,
            probabilities,
            lse,
            output,
            work_counter,
            q_scale,
            k_scale,
            v_scale,
            segment_starts,
            segment_lengths,
            scale * cutlass.Float32(1.4426950408889634),
            scale,
            tiled_mma,
            pv_tiled_mma,
            q_smem_staged,
            k_smem_staged,
            v_smem_staged,
            k_tma_smem_staged,
            v_tma_smem_staged,
            q_gmem_copy,
        ).launch(
            grid=[self.num_ctas, 1, 1],
            block=[256, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        tma_k: cute.CopyAtom,
        k: cute.Tensor,
        tma_v: cute.CopyAtom,
        v: cute.Tensor,
        task_meta: cute.Tensor,
        query_indices: cute.Tensor,
        scores: cute.Tensor,
        probabilities: cute.Tensor,
        lse: cute.Tensor,
        output: cute.Tensor,
        work_counter: cute.Tensor,
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
        v_scale: cute.Tensor,
        segment_starts: cute.Tensor,
        segment_lengths: cute.Tensor,
        scale_log2: cutlass.Float32,
        scale: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        q_smem_staged: cute.ComposedLayout,
        k_smem_staged: cute.ComposedLayout,
        v_smem_staged: cute.ComposedLayout,
        k_tma_smem_staged: cute.ComposedLayout,
        v_tma_smem_staged: cute.ComposedLayout,
        q_gmem_copy: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // Int32(128))

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sQ = storage.sQ.get_tensor(
            q_smem_staged.outer, swizzle=q_smem_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_staged.outer, swizzle=k_smem_staged.inner
        )
        sV = storage.sV.get_tensor(
            v_smem_staged.outer, swizzle=v_smem_staged.inner
        )
        if cutlass.const_expr(self.fp8_kv):
            sK_tma = sK
            sV_tma = storage.sV8.get_tensor(
                v_tma_smem_staged.outer, swizzle=v_tma_smem_staged.inner
            )
        else:
            sK_tma = sK
            sV_tma = sV
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        k_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_barriers.data_ptr(),
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self._tma_k_bytes,
        )
        v_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.v_barriers.data_ptr(),
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self._tma_v_bytes,
        )
        q_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        q_pipe = pipeline.PipelineAsync.create(
            barrier_storage=storage.q_barriers.data_ptr(),
            num_stages=Q_PIPELINE_STAGES,
            producer_group=q_group,
            consumer_group=q_group,
        )

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        v_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        v_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, Q_PIPELINE_STAGES
        )
        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, Q_PIPELINE_STAGES
        )
        if warpgroup_idx == 0:
            cute.arch.setmaxregister_decrease(PRODUCER_REGISTERS)
        if warpgroup_idx == 1:
            cute.arch.setmaxregister_increase(CONSUMER_REGISTERS)

        task_index = storage.task_index.get_tensor(cute.make_layout(1))
        # PERSISTENT_TASK_BODY_BEGIN
        for _ in cutlass.range_constexpr(self.claims_per_cta):
            task_idx = block_idx
            if cutlass.const_expr(self.persistent):
                if tidx == 0:
                    task_index[0] = _atomic_claim(work_counter)
                cute.arch.sync_threads()
                task_idx = task_index[0]

            active = task_idx < Int32(self.num_tasks)
            if not active:
                task_idx = Int32(0)

            batch = task_meta[task_idx, 0]
            proxy_head = task_meta[task_idx, 1]
            key_unit = task_meta[task_idx, 2]
            key_block = key_unit
            key_begin = Int32(0)
            key_end = Int32(self.block)
            if cutlass.const_expr(self.segmented):
                segment_start = segment_starts[key_unit]
                key_block = segment_start // Int32(self.block)
                key_begin = segment_start - key_block * Int32(self.block)
                key_end = key_begin + segment_lengths[key_unit]
            query_count = task_meta[task_idx, 3]
            if not active:
                query_count = Int32(0)
            edge_offset = task_meta[task_idx, 4]
            kv_head = proxy_head // Int32(self.proxy_per_kv)

            gK = cute.local_tile(
                k[None, None, kv_head, batch],
                (self.block, self.block),
                (None, 0),
            )
            tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
                tma_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sK_tma, 0, 2),
                cute.group_modes(gK, 0, 2),
            )
            gV = cute.local_tile(
                v[None, None, kv_head, batch],
                (self.block, self.block),
                (0, None),
            )
            tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
                tma_v,
                0,
                cute.make_layout(1),
                cute.group_modes(sV_tma, 0, 2),
                cute.group_modes(gV, 0, 2),
            )
            if warp_idx == 0:
                k_pipe.producer_acquire(producer_state)
                cute.copy(
                    tma_k,
                    tKgK[(None, key_block)],
                    tKsK[(None, 0)],
                    tma_bar_ptr=k_pipe.producer_get_barrier(producer_state),
                )
                k_pipe.producer_commit(producer_state)
                producer_state.advance()
                v_pipe.producer_acquire(v_producer_state)
                cute.copy(
                    tma_v,
                    tVgV[(None, key_block)],
                    tVsV[(None, 0)],
                    tma_bar_ptr=v_pipe.producer_get_barrier(v_producer_state),
                )
                v_pipe.producer_commit(v_producer_state)
                v_producer_state.advance()

            if warpgroup_idx == 1:
                k_pipe.consumer_wait(consumer_state)
                v_pipe.consumer_wait(v_consumer_state)
            if cutlass.const_expr(self.fp8_kv):
                # K remains E4M3 for QK WGMMA. Use the full CTA to expand V
                # into the BF16 PV layout before producer/consumer divergence.
                cute.arch.sync_threads()
                v_convert_linear = tidx * Int32(4)
                while v_convert_linear < Int32(self.block * self.block):
                    packed_v = cute.make_rmem_tensor((4,), self._kv_storage_dtype)
                    float_v = cute.make_rmem_tensor((4,), cutlass.Float32)
                    restored_v = cute.make_rmem_tensor((4,), self._pv_dtype)
                    for element in cutlass.range_constexpr(4):
                        element_linear = v_convert_linear + Int32(element)
                        row = element_linear // Int32(self.block)
                        col = element_linear - row * Int32(self.block)
                        packed_v[element] = sV_tma[row, col, 0]
                    float_v.store(packed_v.load().to(cutlass.Float32))
                    restored_v.store(float_v.load().to(self._pv_dtype))
                    for element in cutlass.range_constexpr(4):
                        element_linear = v_convert_linear + Int32(element)
                        row = element_linear // Int32(self.block)
                        col = element_linear - row * Int32(self.block)
                        sV[row, col, 0] = restored_v[element]
                    v_convert_linear += Int32(256 * 4)
                cute.arch.sync_threads()

            task_scale_log2 = scale_log2
            task_scale = scale
            task_v_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(self.fp8_kv):
                task_scale_log2 *= k_scale[batch, kv_head, key_block]
                task_scale *= k_scale[batch, kv_head, key_block]
                task_v_scale = v_scale[batch, kv_head, key_block]

            queries_per_tile = Int32(self.rows // self.main_per_proxy)
            if warpgroup_idx == 0:
                query_start = Int32(0)
                while query_start < query_count:
                    q_pipe.producer_acquire(q_producer_state)
                    q_stage = cute.slice_(
                        sQ, (None, None, q_producer_state.index)
                    )
                    q_copy = q_gmem_copy.get_slice(tidx)
                    tQsQ = q_copy.partition_D(q_stage)
                    q_coords = cute.make_identity_tensor((self.rows, self.block))
                    tQcQ = q_copy.partition_S(q_coords)
                    for copy_row in cutlass.range_constexpr(tQsQ.shape[1]):
                        row = tQcQ[0, copy_row, 0][0]
                        tile_slot = row // Int32(self.main_per_proxy)
                        query_slot = query_start + tile_slot
                        main_offset = row - tile_slot * Int32(self.main_per_proxy)
                        valid = query_slot < query_count
                        qid = Int32(0)
                        if valid:
                            qid = query_indices[edge_offset + query_slot]
                        head = proxy_head * Int32(self.main_per_proxy) + main_offset
                        if valid:
                            source = cute.make_tensor(
                                cute.make_ptr(
                                    self._dtype,
                                    _elem_pointer(q, (qid, 0, head, batch)).llvm_ptr,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout(self.block),
                            )
                            source_vectors = cute.tiled_divide(
                                source, (self._q_copy_elements,)
                            )
                            for copy_col in cutlass.range_constexpr(tQsQ.shape[2]):
                                vector = (
                                    tQcQ[0, 0, copy_col][1]
                                    // self._q_copy_elements
                                )
                                cute.copy(
                                    q_copy,
                                    source_vectors[None, vector],
                                    tQsQ[None, copy_row, copy_col],
                                )
                        else:
                            for copy_col in cutlass.range_constexpr(tQsQ.shape[2]):
                                tQsQ[None, copy_row, copy_col].fill(self._dtype(0))
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    q_pipe.producer_commit(q_producer_state)
                    q_producer_state.advance()
                    query_start += queries_per_tile

            if warpgroup_idx == 1:
                query_start = Int32(0)
                while query_start < query_count:
                    q_pipe.consumer_wait(q_consumer_state)
                    wg_thread = tidx - Int32(128)
                    thr_mma = tiled_mma.get_slice(wg_thread)
                    tSsQ = thr_mma.partition_A(sQ)
                    tSsK = thr_mma.partition_B(sK)
                    tSrQ = thr_mma.make_fragment_A(tSsQ)
                    tSrK = thr_mma.make_fragment_B(tSsK)
                    acc_shape = thr_mma.partition_shape_C((self.rows, self.block))
                    acc = thr_mma.make_fragment_C(acc_shape)
                    cute.nvgpu.warpgroup.fence()
                    self._gemm_zero(
                        tiled_mma,
                        tSrQ[(None, None, None, q_consumer_state.index)],
                        tSrK[(None, None, None, 0)],
                        acc,
                    )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)
                    if cutlass.const_expr(self.write_debug):
                        if query_start == 0:
                            g_scores = scores[task_idx, None, None]
                            tCg = thr_mma.partition_C(g_scores)
                            for i in cutlass.range(cute.size(acc), unroll_full=True):
                                tCg[i] = acc[i]
                    coordinates = cute.make_identity_tensor((self.rows, self.block))
                    tCoordinates = thr_mma.partition_C(coordinates)
                    coordinates_mn = cute.make_tensor(
                        tCoordinates.iterator,
                        self._layout_acc_mn(tiled_mma, tCoordinates.layout),
                    )
                    acc_mn = cute.make_tensor(
                        acc.iterator, self._layout_acc_mn(tiled_mma, acc.layout)
                    )
                    if cutlass.const_expr(self.segmented):
                        for row in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
                            for col in cutlass.range_constexpr(cute.size(acc_mn, mode=[1])):
                                key_column = coordinates_mn[row, col][1]
                                if key_column < key_begin or key_column >= key_end:
                                    acc_mn[row, col] = cutlass.Float32(-float("inf"))
                    row_layout = cute.make_layout(cute.size(acc_mn, mode=[0]))
                    row_scale_log2 = cute.make_rmem_tensor_like(
                        row_layout, cutlass.Float32
                    )
                    row_attention_scale = cute.make_rmem_tensor_like(
                        row_layout, cutlass.Float32
                    )
                    for row in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
                        q_value_scale = cutlass.Float32(1.0)
                        if cutlass.const_expr(self.fp8_kv):
                            coordinate = coordinates_mn[row, 0]
                            tile_slot = coordinate[0] // Int32(self.main_per_proxy)
                            main_offset = coordinate[0] - tile_slot * Int32(
                                self.main_per_proxy
                            )
                            query_slot = query_start + tile_slot
                            qid = Int32(0)
                            if query_slot < query_count:
                                qid = query_indices[edge_offset + query_slot]
                            head = proxy_head * Int32(self.main_per_proxy) + main_offset
                            q_value_scale = q_scale[
                                batch, head, qid // Int32(self.block)
                            ]
                        row_scale_log2[row] = task_scale_log2 * q_value_scale
                        row_attention_scale[row] = task_scale * q_value_scale
                    row_max, row_sum = self._softmax_fp32(
                        acc, tiled_mma, row_scale_log2
                    )
                    if cutlass.const_expr(self.write_debug):
                        if query_start == 0:
                            g_probabilities = probabilities[task_idx, None, None]
                            tPg = thr_mma.partition_C(g_probabilities)
                            for i in cutlass.range(cute.size(acc), unroll_full=True):
                                tPg[i] = acc[i]

                    lse_flat = cute.make_tensor(
                        lse.iterator,
                        cute.make_layout(lse.shape[0] * lse.shape[1], stride=1),
                    )
                    for row in cutlass.range_constexpr(cute.size(row_max)):
                        coordinate = coordinates_mn[row, 0]
                        if coordinate[1] == 0:
                            lse_flat[
                                (edge_offset + query_start)
                                * Int32(self.main_per_proxy)
                                + coordinate[0]
                            ] = (
                                row_max[row] * row_attention_scale[row]
                                + _math.log(row_sum[row])
                            )

                    pv_thr_mma = pv_tiled_mma.get_slice(wg_thread)
                    tOsV = pv_thr_mma.partition_B(sV)
                    tOrV = pv_thr_mma.make_fragment_B(tOsV)
                    probability_operand = self._accumulator_to_operand(
                        acc, pv_tiled_mma.tv_layout_A
                    )
                    output_shape = pv_thr_mma.partition_shape_C((self.rows, self.block))
                    output_accumulator = pv_thr_mma.make_fragment_C(output_shape)
                    cute.nvgpu.warpgroup.fence()
                    self._gemm_zero(
                        pv_tiled_mma,
                        probability_operand,
                        tOrV[(None, None, None, 0)],
                        output_accumulator,
                    )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)
                    output_flat = cute.make_tensor(
                        output.iterator,
                        cute.make_layout(
                            (output.shape[0] * output.shape[1], self.block),
                            stride=(self.block, 1),
                        ),
                    )
                    output_offset = cute.domain_offset(
                        (
                            (edge_offset + query_start) * Int32(self.main_per_proxy),
                            0,
                        ),
                        output_flat,
                    )
                    g_output = cute.make_tensor(
                        output_offset.iterator,
                        cute.make_layout(
                            (self.rows, self.block), stride=(self.block, 1)
                        ),
                    )
                    tOg = pv_thr_mma.partition_C(g_output)
                    for i in cutlass.range(
                        cute.size(output_accumulator), unroll_full=True
                    ):
                        tOg[i] = self._pv_dtype(
                            output_accumulator[i] * task_v_scale
                        )
                    q_pipe.consumer_release(q_consumer_state)
                    q_consumer_state.advance()
                    query_start += queries_per_tile

            if warpgroup_idx == 1:
                k_pipe.consumer_release(consumer_state)
                v_pipe.consumer_release(v_consumer_state)
                consumer_state.advance()
                v_consumer_state.advance()
            cute.arch.sync_threads()
        # PERSISTENT_TASK_BODY_END


def wgmma_selected_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    task_meta: torch.Tensor,
    query_indices: torch.Tensor,
    *,
    n_proxy_heads: int,
) -> torch.Tensor:
    """Compute selected 64x128 QK score tiles with Hopper WGMMA."""

    if q.ndim != 4 or k.ndim != 4 or q.shape[-1] != 128 or k.shape[-1] != 128:
        raise ValueError("q and k must have shapes [B,H,S,128]")
    if q.dtype != k.dtype or q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q and k must have matching fp16/bf16 dtypes")
    if q.device.type != "cuda" or k.device != q.device:
        raise ValueError("q and k must be on the same CUDA device")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("q and k batch/sequence dimensions must match")
    main_per_proxy, queries_per_tile, proxy_per_kv = _forward_head_tiling(
        int(q.shape[1]), int(k.shape[1]), int(n_proxy_heads)
    )
    if task_meta.dtype != torch.int32 or query_indices.dtype != torch.int32:
        raise TypeError("schedule tensors must be int32")
    num_tasks = int(task_meta.shape[0])
    scores = torch.empty((num_tasks, 64, 128), device=q.device, dtype=torch.float32)
    probabilities = torch.empty_like(scores)
    lse = torch.empty((num_tasks, 64), device=q.device, dtype=torch.float32)
    output = torch.empty(
        (num_tasks * queries_per_tile, main_per_proxy, 128),
        device=q.device,
        dtype=q.dtype,
    )
    if num_tasks == 0:
        return scores
    q_cute = _to_cute_tensor(q.detach().contiguous())
    k_cute = _to_cute_tensor(k.detach().contiguous())
    meta_cute = _to_cute_tensor(task_meta.contiguous())
    qids_cute = _to_cute_tensor(query_indices.contiguous())
    scores_cute = _to_cute_tensor(scores)
    probabilities_cute = _to_cute_tensor(probabilities)
    lse_cute = _to_cute_tensor(lse)
    work_counter = torch.zeros(1, dtype=torch.int32, device=q.device)
    counter_cute = _to_cute_tensor(work_counter)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key = (
        "selected_qk_wgmma",
        num_tasks,
        main_per_proxy,
        proxy_per_kv,
        True,
        q_cute.element_type,
    )
    if key not in _COMPILE_CACHE:
        kernel = _SelectedQKWgmmaKernel(
            num_tasks, main_per_proxy, proxy_per_kv, write_debug=True
        )
        _COMPILE_CACHE[key] = cute.compile(
            kernel,
            q_cute,
            k_cute,
            k_cute,
            meta_cute,
            qids_cute,
            scores_cute,
            probabilities_cute,
            lse_cute,
            _to_cute_tensor(output),
            counter_cute,
            scores_cute,
            scores_cute,
            scores_cute,
            qids_cute,
            qids_cute,
            1.0,
            stream,
        )
    _COMPILE_CACHE[key](
        q_cute,
        k_cute,
        k_cute,
        meta_cute,
        qids_cute,
        scores_cute,
        probabilities_cute,
        lse_cute,
        _to_cute_tensor(output),
        counter_cute,
        scores_cute,
        scores_cute,
        scores_cute,
        qids_cute,
        qids_cute,
        1.0,
        stream,
    )
    return scores


def wgmma_selected_softmax(
    q: torch.Tensor,
    k: torch.Tensor,
    task_meta: torch.Tensor,
    query_indices: torch.Tensor,
    *,
    n_proxy_heads: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 normalized probabilities and LSE from fused QK/softmax."""

    num_tasks = int(task_meta.shape[0])
    main_per_proxy, queries_per_tile, proxy_per_kv = _forward_head_tiling(
        int(q.shape[1]), int(k.shape[1]), int(n_proxy_heads)
    )
    scores = torch.empty((num_tasks, 64, 128), device=q.device, dtype=torch.float32)
    probabilities = torch.empty_like(scores)
    lse = torch.empty(
        (num_tasks * queries_per_tile, main_per_proxy),
        device=q.device,
        dtype=torch.float32,
    )
    output = torch.empty(
        (num_tasks * queries_per_tile, main_per_proxy, 128),
        device=q.device,
        dtype=q.dtype,
    )
    if num_tasks == 0:
        return probabilities, lse
    q_cute = _to_cute_tensor(q.detach().contiguous())
    k_cute = _to_cute_tensor(k.detach().contiguous())
    meta_cute = _to_cute_tensor(task_meta.contiguous())
    qids_cute = _to_cute_tensor(query_indices.contiguous())
    scores_cute = _to_cute_tensor(scores)
    probabilities_cute = _to_cute_tensor(probabilities)
    lse_cute = _to_cute_tensor(lse)
    work_counter = torch.zeros(1, dtype=torch.int32, device=q.device)
    counter_cute = _to_cute_tensor(work_counter)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key = (
        "selected_qk_wgmma",
        num_tasks,
        main_per_proxy,
        proxy_per_kv,
        True,
        q_cute.element_type,
    )
    if key not in _COMPILE_CACHE:
        kernel = _SelectedQKWgmmaKernel(
            num_tasks, main_per_proxy, proxy_per_kv, write_debug=True
        )
        _COMPILE_CACHE[key] = cute.compile(
            kernel,
            q_cute,
            k_cute,
            k_cute,
            meta_cute,
            qids_cute,
            scores_cute,
            probabilities_cute,
            lse_cute,
            _to_cute_tensor(output),
            counter_cute,
            scores_cute,
            scores_cute,
            scores_cute,
            qids_cute,
            qids_cute,
            float(scale),
            stream,
        )
    _COMPILE_CACHE[key](
        q_cute,
        k_cute,
        k_cute,
        meta_cute,
        qids_cute,
        scores_cute,
        probabilities_cute,
        lse_cute,
        _to_cute_tensor(output),
        counter_cute,
        scores_cute,
        scores_cute,
        scores_cute,
        qids_cute,
        qids_cute,
        float(scale),
        stream,
    )
    return probabilities, lse


def wgmma_selected_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    task_meta: torch.Tensor,
    query_indices: torch.Tensor,
    *,
    n_proxy_heads: int,
    scale: float,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    segment_starts: torch.Tensor | None = None,
    segment_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return partial output/LSE using BF16 compute and BF16 or E4M3 K/V storage."""

    if k.shape != v.shape or k.dtype != v.dtype:
        raise ValueError("K/V shapes and dtypes must match")
    fp8_kv = k.dtype == torch.float8_e4m3fn
    if fp8_kv:
        if q.dtype != torch.float8_e4m3fn:
            raise TypeError("mixed FP8 QK requires E4M3 Q and K tensors")
        expected_q_scales = (q.shape[0], q.shape[1], q.shape[2] // 128)
        expected_scales = (k.shape[0], k.shape[1], k.shape[2] // 128)
        if (
            q_scale is None
            or q_scale.shape != expected_q_scales
            or q_scale.dtype != torch.float32
            or k_scale is None
            or v_scale is None
            or k_scale.shape != expected_scales
            or v_scale.shape != expected_scales
            or k_scale.dtype != torch.float32
            or v_scale.dtype != torch.float32
        ):
            raise ValueError(
                "FP8 Q/K/V scales must be FP32 with shapes "
                f"{expected_q_scales} and {expected_scales}"
            )
    elif q.dtype != k.dtype or k.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("BF16/FP16 Q/K/V shapes and dtypes must match")
    segmented = segment_starts is not None or segment_lengths is not None
    if segmented:
        if segment_starts is None or segment_lengths is None:
            raise ValueError("segment_starts and segment_lengths must be provided together")
        if (
            segment_starts.device != q.device
            or segment_lengths.device != q.device
            or segment_starts.dtype != torch.int32
            or segment_lengths.dtype != torch.int32
            or segment_starts.ndim != 1
            or segment_lengths.shape != segment_starts.shape
        ):
            raise ValueError("segment metadata must be matching CUDA int32 vectors")
    num_tasks = int(task_meta.shape[0])
    # Production specialization does not write either diagnostic tensor. Keep
    # one aligned element because the common compiled signature still carries
    # the arguments used by the focused QK/softmax validation specializations.
    scores = torch.empty(1, device=q.device, dtype=torch.float32)
    probabilities = torch.empty_like(scores)
    main_per_proxy, queries_per_tile, proxy_per_kv = _forward_head_tiling(
        int(q.shape[1]), int(k.shape[1]), int(n_proxy_heads)
    )
    num_edges = int(query_indices.shape[0])
    # The final WGMMA tile writes predicated padding rows after the last edge.
    edge_capacity = num_edges + queries_per_tile - 1
    lse = torch.empty((edge_capacity, main_per_proxy), device=q.device, dtype=torch.float32)
    output = torch.empty(
        (edge_capacity, main_per_proxy, 128),
        device=q.device,
        dtype=torch.bfloat16 if fp8_kv else q.dtype,
    )
    if num_tasks == 0:
        return output, lse
    q_cute = _to_cute_tensor(q.detach().contiguous())
    k_cute = _to_cute_tensor(k.detach().contiguous())
    v_cute = _to_cute_tensor(v.detach().contiguous())
    meta_cute = _to_cute_tensor(task_meta.contiguous())
    qids_cute = _to_cute_tensor(query_indices.contiguous())
    scores_cute = _to_cute_tensor(scores)
    probabilities_cute = _to_cute_tensor(probabilities)
    lse_cute = _to_cute_tensor(lse)
    output_cute = _to_cute_tensor(output)
    q_scale_cute = _to_cute_tensor(q_scale) if fp8_kv else scores_cute
    k_scale_cute = _to_cute_tensor(k_scale) if fp8_kv else scores_cute
    v_scale_cute = _to_cute_tensor(v_scale) if fp8_kv else scores_cute
    segment_starts_cute = (
        _to_cute_tensor(segment_starts.contiguous()) if segmented else qids_cute
    )
    segment_lengths_cute = (
        _to_cute_tensor(segment_lengths.contiguous()) if segmented else qids_cute
    )
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    ctas_per_sm = max(1, int(os.environ.get("MSA_SM90_CTAS_PER_SM", "4")))
    max_ctas = int(os.environ.get("MSA_SM90_MAX_CTAS", "0"))
    num_ctas = min(num_tasks, ctas_per_sm * int(sm_count))
    if max_ctas > 0:
        num_ctas = min(num_ctas, max_ctas)
    persistent = num_ctas < num_tasks
    work_counter = (
        torch.zeros(1, dtype=torch.int32, device=q.device) if persistent else None
    )
    counter_cute = _to_cute_tensor(work_counter) if persistent else qids_cute
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key = (
        "selected_qk_wgmma",
        num_tasks,
        main_per_proxy,
        proxy_per_kv,
        False,
        persistent,
        num_ctas,
        q_cute.element_type,
        fp8_kv,
        segmented,
    )
    if key not in _COMPILE_CACHE:
        kernel = _SelectedQKWgmmaKernel(
            num_tasks,
            main_per_proxy,
            proxy_per_kv,
            write_debug=False,
            persistent=persistent,
            num_ctas=num_ctas,
            fp8_kv=fp8_kv,
            segmented=segmented,
        )
        _COMPILE_CACHE[key] = cute.compile(
            kernel,
            q_cute,
            k_cute,
            v_cute,
            meta_cute,
            qids_cute,
            scores_cute,
            probabilities_cute,
            lse_cute,
            output_cute,
            counter_cute,
            q_scale_cute,
            k_scale_cute,
            v_scale_cute,
            segment_starts_cute,
            segment_lengths_cute,
            float(scale),
            stream,
        )
    _COMPILE_CACHE[key](
        q_cute,
        k_cute,
        v_cute,
        meta_cute,
        qids_cute,
        scores_cute,
        probabilities_cute,
        lse_cute,
        output_cute,
        counter_cute,
        q_scale_cute,
        k_scale_cute,
        v_scale_cute,
        segment_starts_cute,
        segment_lengths_cute,
        float(scale),
        stream,
    )
    return output[:num_edges], lse[:num_edges]


__all__ = [
    "persistent_claim_work",
    "tma_load_fp8_kv_as_bf16",
    "tma_load_selected_kv",
    "wgmma_selected_qk",
    "wgmma_selected_softmax",
    "wgmma_selected_attention",
]
