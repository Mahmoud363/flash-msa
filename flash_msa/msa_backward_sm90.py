"""Hopper WGMMA building blocks for the KV-stationary MSA backward.

This module is intentionally separate from the production warp-MMA fallback so
the native backward can be validated one tensor-core product at a time before
its reverse-CSR scheduler is enabled.
"""

from __future__ import annotations

import inspect

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass import pipeline
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as hopper_helpers
from quack import layout_utils, sm90_utils


_COMPILE_CACHE: dict[tuple[object, ...], object] = {}
_NVVM_ATOMICRMW_HAS_RES = "res" in inspect.signature(nvvm.atomicrmw).parameters


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16)


@dsl_user_op
def _atomic_add_fp32(value: Float32, pointer: cute.Pointer, *, loc=None, ip=None) -> None:
    if _NVVM_ATOMICRMW_HAS_RES:
        nvvm.atomicrmw(
            T.f32(), nvvm.AtomicOpKind.FADD, pointer.llvm_ptr, value.ir_value()
        )
    else:
        nvvm.atomicrmw(nvvm.AtomicOpKind.FADD, pointer.llvm_ptr, value.ir_value())


@dsl_user_op
def _elem_pointer(tensor: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None):
    return tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)


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


def _layout_acc_mn(tiled_mma, accumulator):
    separated = _layout_separate(
        tiled_mma.shape_mnk[0],
        accumulator[0],
        tiled_mma.tv_layout_C.stride[1],
    )
    values_m, values_n = separated[0], separated[1]
    if cutlass.const_expr(cute.rank(values_m) == 1):
        values_m = cute.append(values_m, accumulator[1])
    else:
        values_m = cute.append(cute.append(cute.make_layout(()), values_m), accumulator[1])
    if cutlass.const_expr(cute.rank(values_n) == 1):
        values_n = cute.append(values_n, accumulator[2])
    else:
        values_n = cute.append(cute.append(cute.make_layout(()), values_n), accumulator[2])
    if cutlass.const_expr(cute.rank(values_m) == 1):
        return cute.append(values_m, values_n)
    return cute.append(cute.append(cute.make_layout(()), values_m), values_n)


class _BackwardDQWgmmaKernel:
    """Compute one ``dQ = dS @ K`` 64x128 tile with SM90 WGMMA."""

    rows = 64
    key_rows = 64
    head_dim = 128

    @cute.jit
    def __call__(
        self,
        ds: cute.Tensor,
        kt: cute.Tensor,
        dq: cute.Tensor,
        stream: cuda.CUstream,
    ):
        dtype = ds.element_type
        tiled_mma = hopper_helpers.make_trivial_tiled_mma(
            dtype,
            dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            (1, 1, 1),
            (self.rows, self.head_dim),
        )
        sds_layout = sm90_utils.make_smem_layout(
            dtype, LayoutEnum.ROW_MAJOR, (self.rows, self.key_rows), None
        )
        skt_layout = sm90_utils.make_smem_layout(
            dtype, LayoutEnum.ROW_MAJOR, (self.head_dim, self.key_rows), None
        )

        @cute.struct
        class SharedStorage:
            sDS: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sds_layout)], 1024
            ]
            sKT: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(skt_layout)], 1024
            ]

        self._shared_storage = SharedStorage
        self.kernel(ds, kt, dq, tiled_mma, sds_layout, skt_layout).launch(
            grid=[1, 1, 1],
            block=[128, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        ds: cute.Tensor,
        kt: cute.Tensor,
        dq: cute.Tensor,
        tiled_mma: cute.TiledMma,
        sds_layout: cute.ComposedLayout,
        skt_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._shared_storage)
        sDS = storage.sDS.get_tensor(sds_layout.outer, swizzle=sds_layout.inner)
        sKT = storage.sKT.get_tensor(skt_layout.outer, swizzle=skt_layout.inner)

        linear = tidx
        while linear < Int32(self.rows * self.key_rows):
            row = linear // Int32(self.key_rows)
            col = linear - row * Int32(self.key_rows)
            sDS[row, col] = ds[row, col]
            linear += Int32(128)
        linear = tidx
        while linear < Int32(self.head_dim * self.key_rows):
            row = linear // Int32(self.key_rows)
            col = linear - row * Int32(self.key_rows)
            sKT[row, col] = kt[row, col]
            linear += Int32(128)
        cute.arch.sync_threads()

        thr_mma = tiled_mma.get_slice(tidx)
        _, tCrDS, tCrKT = sm90_utils.partition_fragment_ABC(
            thr_mma,
            (self.rows, self.head_dim, self.key_rows),
            sDS,
            sKT,
        )
        acc = sm90_utils.gemm_zero_init(
            tiled_mma, (self.rows, self.head_dim), tCrDS, tCrKT, wg_wait=0
        )
        coordinates = cute.make_identity_tensor((self.rows, self.head_dim))
        coord_mn = cute.make_tensor(
            thr_mma.partition_C(coordinates).iterator,
            _layout_acc_mn(tiled_mma, thr_mma.partition_C(coordinates).layout),
        )
        acc_mn = cute.make_tensor(acc.iterator, _layout_acc_mn(tiled_mma, acc.layout))
        for row in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
            for col in cutlass.range_constexpr(cute.size(acc_mn, mode=[1])):
                coord = coord_mn[row, col]
                dq[coord[0], coord[1]] = acc_mn[row, col]


class _BackwardAttentionTileKernel:
    """Recompute P/dS and form dQ/dK/dV for one 64x64 selected tile."""

    rows = 64
    keys = 64
    dim = 128

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        do: cute.Tensor,
        lse: cute.Tensor,
        delta: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        dtype = q.element_type
        sQ_layout, sK_layout, sV_layout, sdO_layout, sPdS_layout = [
            sm90_utils.make_smem_layout(dtype, LayoutEnum.ROW_MAJOR, shape, None)
            for shape in (
                (self.rows, self.dim),
                (self.keys, self.dim),
                (self.keys, self.dim),
                (self.rows, self.dim),
                (self.rows, self.keys),
            )
        ]
        mma_sdp = hopper_helpers.make_trivial_tiled_mma(
            dtype,
            dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            (1, 1, 1),
            (self.rows, self.keys),
        )
        mma_dkv = hopper_helpers.make_trivial_tiled_mma(
            dtype,
            dtype,
            warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN,
            Float32,
            (1, 1, 1),
            (self.keys, self.dim),
        )
        mma_dq = hopper_helpers.make_trivial_tiled_mma(
            dtype,
            dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            (1, 1, 1),
            (self.rows, self.dim),
        )

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sK_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sV_layout)], 1024]
            sdO: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sdO_layout)], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sPdS_layout)], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sPdS_layout)], 1024]

        self._attention_storage = SharedStorage
        self.attention_kernel(
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq,
            dk,
            dv,
            scale,
            mma_sdp,
            mma_dkv,
            mma_dq,
            sQ_layout,
            sK_layout,
            sV_layout,
            sdO_layout,
            sPdS_layout,
        ).launch(
            grid=[1, 1, 1],
            block=[128, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def attention_kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        do: cute.Tensor,
        lse: cute.Tensor,
        delta: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        scale: Float32,
        mma_sdp: cute.TiledMma,
        mma_dkv: cute.TiledMma,
        mma_dq: cute.TiledMma,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        storage = cutlass.utils.SmemAllocator().allocate(self._attention_storage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)

        linear = tidx
        while linear < Int32(self.rows * self.dim):
            row = linear // Int32(self.dim)
            col = linear - row * Int32(self.dim)
            sQ[row, col] = q[row, col]
            sdO[row, col] = do[row, col]
            sK[row, col] = k[row, col]
            sV[row, col] = v[row, col]
            linear += Int32(128)
        cute.arch.sync_threads()

        thr_sdp = mma_sdp.get_slice(tidx)
        _, q_frag, k_frag = sm90_utils.partition_fragment_ABC(
            thr_sdp, (self.rows, self.keys, self.dim), sQ, sK
        )
        scores = sm90_utils.gemm_zero_init(
            mma_sdp, (self.rows, self.keys), q_frag, k_frag, wg_wait=0
        )
        _, do_frag, v_frag = sm90_utils.partition_fragment_ABC(
            thr_sdp, (self.rows, self.keys, self.dim), sdO, sV
        )
        dp = sm90_utils.gemm_zero_init(
            mma_sdp, (self.rows, self.keys), do_frag, v_frag, wg_wait=0
        )
        coordinates = cute.make_identity_tensor((self.rows, self.keys))
        partitioned_coords = thr_sdp.partition_C(coordinates)
        coord_mn = cute.make_tensor(
            partitioned_coords.iterator, _layout_acc_mn(mma_sdp, partitioned_coords.layout)
        )
        scores_mn = cute.make_tensor(scores.iterator, _layout_acc_mn(mma_sdp, scores.layout))
        dp_mn = cute.make_tensor(dp.iterator, _layout_acc_mn(mma_sdp, dp.layout))
        for row in cutlass.range_constexpr(cute.size(scores_mn, mode=[0])):
            for col in cutlass.range_constexpr(cute.size(scores_mn, mode=[1])):
                coord = coord_mn[row, col]
                probability = cute.math.exp(
                    scores_mn[row, col] * scale - lse[coord[0]], fastmath=True
                )
                sP[coord[0], coord[1]] = q.element_type(probability)
                ds_value = probability * (dp_mn[row, col] - delta[coord[0]]) * scale
                sdS[coord[0], coord[1]] = q.element_type(ds_value)
        cute.arch.sync_threads()

        sPt = layout_utils.transpose_view(sP)
        sdSt = layout_utils.transpose_view(sdS)
        sQt = layout_utils.transpose_view(sQ)
        sdOt = layout_utils.transpose_view(sdO)
        sKt = layout_utils.transpose_view(sK)

        thr_dkv = mma_dkv.get_slice(tidx)
        _, p_frag, do_t_frag = sm90_utils.partition_fragment_ABC(
            thr_dkv, (self.keys, self.dim, self.rows), sPt, sdOt
        )
        acc_dv = sm90_utils.gemm_zero_init(
            mma_dkv, (self.keys, self.dim), p_frag, do_t_frag, wg_wait=0
        )
        _, ds_t_frag, q_t_frag = sm90_utils.partition_fragment_ABC(
            thr_dkv, (self.keys, self.dim, self.rows), sdSt, sQt
        )
        acc_dk = sm90_utils.gemm_zero_init(
            mma_dkv, (self.keys, self.dim), ds_t_frag, q_t_frag, wg_wait=0
        )
        thr_dq = mma_dq.get_slice(tidx)
        _, ds_frag, k_t_frag = sm90_utils.partition_fragment_ABC(
            thr_dq, (self.rows, self.dim, self.keys), sdS, sKt
        )
        acc_dq = sm90_utils.gemm_zero_init(
            mma_dq, (self.rows, self.dim), ds_frag, k_t_frag, wg_wait=0
        )

        for tiled, acc, target, shape in (
            (mma_dq, acc_dq, dq, (self.rows, self.dim)),
            (mma_dkv, acc_dk, dk, (self.keys, self.dim)),
            (mma_dkv, acc_dv, dv, (self.keys, self.dim)),
        ):
            thr = tiled.get_slice(tidx)
            coords = thr.partition_C(cute.make_identity_tensor(shape))
            coords_mn = cute.make_tensor(
                coords.iterator, _layout_acc_mn(tiled, coords.layout)
            )
            values_mn = cute.make_tensor(acc.iterator, _layout_acc_mn(tiled, acc.layout))
            for row in cutlass.range_constexpr(cute.size(values_mn, mode=[0])):
                for col in cutlass.range_constexpr(cute.size(values_mn, mode=[1])):
                    coord = coords_mn[row, col]
                    target[coord[0], coord[1]] = values_mn[row, col]


class _KVRowBackwardKernel:
    """Accumulate main-attention gradients for one reverse-CSR KV row.

    A CTA owns one ``(batch, proxy-head, key-block, key-slice)`` work item,
    keeps its K/V tile resident, and visits every selected query in that CSR
    row in groups large enough to fill a 64-row WGMMA operation.
    """

    rows = 64
    keys = 64
    dim = 128

    def __init__(
        self, batch, n_heads, n_kv_heads, n_proxy_heads, seq_len,
        num_key_units, segmented=False,
    ):
        self.batch = int(batch)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.n_proxy_heads = int(n_proxy_heads)
        self.seq_len = int(seq_len)
        self.num_blocks = self.seq_len // 128
        self.num_key_units = int(num_key_units)
        self.segmented = bool(segmented)
        self.main_per_proxy = self.n_heads // self.n_proxy_heads
        self.proxy_per_kv = self.n_proxy_heads // self.n_kv_heads
        self.queries_per_group = self.rows // self.main_per_proxy
        self.num_rows = (
            self.n_proxy_heads * self.num_key_units
            if self.segmented
            else self.batch * self.n_proxy_heads * self.num_blocks
        )

    @cute.jit
    def __call__(self, q, k, v, grad_o, lse, delta, row_ptr, query_ids,
                 segment_starts, segment_lengths, segment_batches,
                 dq, dk, dv, scale: Float32, stream: cuda.CUstream):
        batch, n_kv_heads, seq_len, head_dim = k.shape
        kv_layout = cute.make_layout(
            (seq_len, head_dim, n_kv_heads, batch),
            stride=(
                head_dim, 1, seq_len * head_dim,
                n_kv_heads * seq_len * head_dim,
            ),
        )
        k = cute.make_tensor(k.iterator, kv_layout)
        v = cute.make_tensor(v.iterator, kv_layout)
        dtype = q.element_type
        layouts = [
            sm90_utils.make_smem_layout(dtype, LayoutEnum.ROW_MAJOR, shape, None)
            for shape in ((64, 128), (64, 128), (64, 128), (64, 128), (64, 64))
        ]
        sQ_layout, sK_layout, sV_layout, sdO_layout, sPdS_layout = layouts
        q_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=128,
        )
        q_copy_elements = 128 // dtype.width
        q_threads_per_row = 128 // q_copy_elements
        q_gmem_copy = cute.make_tiled_copy_tv(
            q_copy_atom,
            cute.make_ordered_layout(
                (128 // q_threads_per_row, q_threads_per_row), order=(1, 0)
            ),
            cute.make_layout((1, q_copy_elements)),
        )
        self._q_copy_elements = q_copy_elements
        kv_smem_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128, dtype
        )
        sKV_layout_staged = cute.tile_to_shape(
            kv_smem_atom, (64, 128, 1), (0, 1, 2)
        )
        sK_layout = cute.slice_(sKV_layout_staged, (None, None, 0))
        sV_layout = sK_layout
        layouts[1], layouts[2] = sK_layout, sV_layout
        tma_k, tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            k,
            sK_layout,
            (64, 128),
        )
        tma_v, tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            v,
            sV_layout,
            (64, 128),
        )
        self._tma_k_bytes = cute.size_in_bytes(dtype, sK_layout)
        self._tma_v_bytes = cute.size_in_bytes(dtype, sV_layout)
        mma_sdp = hopper_helpers.make_trivial_tiled_mma(
            dtype, dtype, warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K, Float32, (1, 1, 1), (64, 64)
        )
        mma_dkv = hopper_helpers.make_trivial_tiled_mma(
            dtype, dtype, warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN, Float32, (1, 1, 1), (64, 128)
        )
        mma_dq = hopper_helpers.make_trivial_tiled_mma(
            dtype, dtype, warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN, Float32, (1, 1, 1), (64, 128)
        )

        @cute.struct
        class SharedStorage:
            k_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            v_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            sQ: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sKV_layout_staged)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(sKV_layout_staged)], 1024
            ]
            sdO: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sdO_layout)], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sPdS_layout)], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(sPdS_layout)], 1024]

        self._kvrow_storage = SharedStorage
        self.kernel(
            q, tma_k, tensor_k, tma_v, tensor_v,
            grad_o, lse, delta, row_ptr, query_ids,
            segment_starts, segment_lengths, segment_batches, dq, dk, dv,
            scale, mma_sdp, mma_dkv, mma_dq, q_gmem_copy,
            sKV_layout_staged, *layouts,
        ).launch(
            grid=[self.num_rows * 2, 1, 1], block=[128, 1, 1],
            smem=SharedStorage.size_in_bytes(), stream=stream
        )

    @cute.kernel
    def kernel(self, q, tma_k, k, tma_v, v,
               grad_o, lse, delta, row_ptr, query_ids,
               segment_starts, segment_lengths, segment_batches, dq, dk, dv,
               scale: Float32, mma_sdp: cute.TiledMma, mma_dkv: cute.TiledMma,
               mma_dq: cute.TiledMma, q_gmem_copy: cute.TiledCopy,
               sKV_layout_staged,
               sQ_layout, sK_layout, sV_layout,
               sdO_layout, sPdS_layout):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        row_idx = block_idx // Int32(2)
        key_slice = block_idx - row_idx * Int32(2)
        edge_begin = row_ptr[row_idx]
        edge_end = row_ptr[row_idx + Int32(1)]

        row_tmp = row_idx // Int32(self.num_blocks)
        key_unit = row_idx - row_tmp * Int32(self.num_blocks)
        batch_idx = row_tmp // Int32(self.n_proxy_heads)
        proxy_head = row_tmp - batch_idx * Int32(self.n_proxy_heads)
        key_block = key_unit
        key_begin = Int32(0)
        key_end = Int32(128)
        local_query_start = key_block * Int32(128)
        local_query_count = Int32(128)
        if cutlass.const_expr(self.segmented):
            proxy_head = row_idx // Int32(self.num_key_units)
            key_unit = row_idx - proxy_head * Int32(self.num_key_units)
            batch_idx = segment_batches[key_unit]
            segment_start = segment_starts[key_unit]
            local_query_start = segment_start
            local_query_count = segment_lengths[key_unit]
            key_block = segment_start // Int32(128)
            key_begin = segment_start - key_block * Int32(128)
            key_end = key_begin + local_query_count
        kv_head = proxy_head // Int32(self.proxy_per_kv)
        key_start = key_block * Int32(128) + key_slice * Int32(64)

        storage = cutlass.utils.SmemAllocator().allocate(self._kvrow_storage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK_staged = storage.sK.get_tensor(
            sKV_layout_staged.outer, swizzle=sKV_layout_staged.inner
        )
        sV_staged = storage.sV.get_tensor(
            sKV_layout_staged.outer, swizzle=sKV_layout_staged.inner
        )
        sK = cute.slice_(sK_staged, (None, None, 0))
        sV = cute.slice_(sV_staged, (None, None, 0))
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)

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
        gK = cute.local_tile(k[None, None, kv_head, batch_idx], (64, 128), (None, 0))
        gV = cute.local_tile(v[None, None, kv_head, batch_idx], (64, 128), (None, 0))
        cta_layout = cute.make_layout(1)
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_k, 0, cta_layout,
            cute.group_modes(sK_staged, 0, 2), cute.group_modes(gK, 0, 2),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_v, 0, cta_layout,
            cute.group_modes(sV_staged, 0, 2), cute.group_modes(gV, 0, 2),
        )
        k_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        v_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        k_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        v_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            k_pipe.producer_acquire(k_producer)
            cute.copy(
                tma_k, tKgK[(None, key_start // Int32(64))], tKsK[(None, 0)],
                tma_bar_ptr=k_pipe.producer_get_barrier(k_producer),
            )
            k_pipe.producer_commit(k_producer)
            v_pipe.producer_acquire(v_producer)
            cute.copy(
                tma_v, tVgV[(None, key_start // Int32(64))], tVsV[(None, 0)],
                tma_bar_ptr=v_pipe.producer_get_barrier(v_producer),
            )
            v_pipe.producer_commit(v_producer)
        k_pipe.consumer_wait(k_consumer)
        v_pipe.consumer_wait(v_consumer)

        thr_dkv = mma_dkv.get_slice(tidx)
        c_dkv = thr_dkv.partition_C(cute.make_identity_tensor((64, 128)))
        acc_dk = cute.make_rmem_tensor(c_dkv.shape, Float32)
        acc_dv = cute.make_rmem_tensor(c_dkv.shape, Float32)
        acc_dk.fill(0.0)
        acc_dv.fill(0.0)
        # Forward always includes the query's own 128-token block, while the
        # compact reverse CSR stores only remote selections.  Visit the local
        # queries first, then the explicit CSR entries.
        row_count = local_query_count + edge_end - edge_begin
        row_work_base = Int32(0)
        while row_work_base < row_count:
            q_copy = q_gmem_copy.get_slice(tidx)
            tQsQ = q_copy.partition_D(sQ)
            tdOsdO = q_copy.partition_D(sdO)
            q_coords = cute.make_identity_tensor((self.rows, 128))
            tQcQ = q_copy.partition_S(q_coords)
            for copy_row in cutlass.range_constexpr(tQsQ.shape[1]):
                qr = tQcQ[0, copy_row, 0][0]
                slot = qr // Int32(self.main_per_proxy)
                head_off = qr - slot * Int32(self.main_per_proxy)
                local_work = row_work_base + slot
                valid = local_work < row_count
                qid = Int32(0)
                if valid:
                    if local_work < local_query_count:
                        qid = local_query_start + local_work
                    else:
                        qid = query_ids[edge_begin + local_work - local_query_count]
                main_head = proxy_head * Int32(self.main_per_proxy) + head_off
                if valid:
                    q_source = cute.make_tensor(
                        cute.make_ptr(
                            q.element_type,
                            _elem_pointer(q, (batch_idx, main_head, qid, 0)).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(128),
                    )
                    do_source = cute.make_tensor(
                        cute.make_ptr(
                            grad_o.element_type,
                            _elem_pointer(
                                grad_o, (batch_idx, main_head, qid, 0)
                            ).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(128),
                    )
                    q_vectors = cute.tiled_divide(q_source, (self._q_copy_elements,))
                    do_vectors = cute.tiled_divide(do_source, (self._q_copy_elements,))
                    for copy_col in cutlass.range_constexpr(tQsQ.shape[2]):
                        vector = tQcQ[0, 0, copy_col][1] // self._q_copy_elements
                        cute.copy(
                            q_copy, q_vectors[None, vector],
                            tQsQ[None, copy_row, copy_col],
                        )
                        cute.copy(
                            q_copy, do_vectors[None, vector],
                            tdOsdO[None, copy_row, copy_col],
                        )
                else:
                    for copy_col in cutlass.range_constexpr(tQsQ.shape[2]):
                        tQsQ[None, copy_row, copy_col].fill(q.element_type(0))
                        tdOsdO[None, copy_row, copy_col].fill(q.element_type(0))
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            thr_sdp = mma_sdp.get_slice(tidx)
            _, q_frag, k_frag = sm90_utils.partition_fragment_ABC(
                thr_sdp, (64, 64, 128), sQ, sK
            )
            scores = sm90_utils.gemm_zero_init(mma_sdp, (64, 64), q_frag, k_frag, wg_wait=0)
            _, do_frag, v_frag = sm90_utils.partition_fragment_ABC(
                thr_sdp, (64, 64, 128), sdO, sV
            )
            dp = sm90_utils.gemm_zero_init(mma_sdp, (64, 64), do_frag, v_frag, wg_wait=0)
            coords = thr_sdp.partition_C(cute.make_identity_tensor((64, 64)))
            coords_mn = cute.make_tensor(coords.iterator, _layout_acc_mn(mma_sdp, coords.layout))
            scores_mn = cute.make_tensor(scores.iterator, _layout_acc_mn(mma_sdp, scores.layout))
            dp_mn = cute.make_tensor(dp.iterator, _layout_acc_mn(mma_sdp, dp.layout))
            for ri in cutlass.range_constexpr(cute.size(scores_mn, mode=[0])):
                for ci in cutlass.range_constexpr(cute.size(scores_mn, mode=[1])):
                    coord = coords_mn[ri, ci]
                    qr, kc = coord[0], coord[1]
                    slot = qr // Int32(self.main_per_proxy)
                    head_off = qr - slot * Int32(self.main_per_proxy)
                    local_work = row_work_base + slot
                    valid = local_work < row_count
                    qid = Int32(0)
                    if valid:
                        if local_work < local_query_count:
                            qid = local_query_start + local_work
                        else:
                            qid = query_ids[edge_begin + local_work - local_query_count]
                    main_head = proxy_head * Int32(self.main_per_proxy) + head_off
                    valid = (
                        valid
                        and key_slice * Int32(64) + kc >= key_begin
                        and key_slice * Int32(64) + kc < key_end
                        and key_start + kc <= qid
                    )
                    probability = Float32(0.0)
                    ds_value = Float32(0.0)
                    if valid:
                        probability = cute.math.exp(
                            scores_mn[ri, ci] * scale - lse[batch_idx, main_head, qid],
                            fastmath=True,
                        )
                        ds_value = probability * (
                            dp_mn[ri, ci] - delta[batch_idx, main_head, qid]
                        ) * scale
                    sP[qr, kc] = q.element_type(probability)
                    sdS[qr, kc] = q.element_type(ds_value)
            cute.arch.sync_threads()

            sPt = layout_utils.transpose_view(sP)
            sdSt = layout_utils.transpose_view(sdS)
            sQt = layout_utils.transpose_view(sQ)
            sdOt = layout_utils.transpose_view(sdO)
            sKt = layout_utils.transpose_view(sK)
            _, p_frag, do_t_frag = sm90_utils.partition_fragment_ABC(
                thr_dkv, (64, 128, 64), sPt, sdOt
            )
            sm90_utils.gemm_w_idx(mma_dkv, acc_dv, p_frag, do_t_frag, zero_init=False, wg_wait=0)
            _, ds_t_frag, q_t_frag = sm90_utils.partition_fragment_ABC(
                thr_dkv, (64, 128, 64), sdSt, sQt
            )
            sm90_utils.gemm_w_idx(mma_dkv, acc_dk, ds_t_frag, q_t_frag, zero_init=False, wg_wait=0)
            thr_dq = mma_dq.get_slice(tidx)
            _, ds_frag, k_t_frag = sm90_utils.partition_fragment_ABC(
                thr_dq, (64, 128, 64), sdS, sKt
            )
            acc_dq = sm90_utils.gemm_zero_init(
                mma_dq, (64, 128), ds_frag, k_t_frag, wg_wait=0
            )
            dq_coords = thr_dq.partition_C(cute.make_identity_tensor((64, 128)))
            dq_coords_mn = cute.make_tensor(
                dq_coords.iterator, _layout_acc_mn(mma_dq, dq_coords.layout)
            )
            dq_values_mn = cute.make_tensor(
                acc_dq.iterator, _layout_acc_mn(mma_dq, acc_dq.layout)
            )
            for ri in cutlass.range_constexpr(cute.size(dq_values_mn, mode=[0])):
                for ci in cutlass.range_constexpr(cute.size(dq_values_mn, mode=[1])):
                    coord = dq_coords_mn[ri, ci]
                    qr = coord[0]
                    slot = qr // Int32(self.main_per_proxy)
                    head_off = qr - slot * Int32(self.main_per_proxy)
                    local_work = row_work_base + slot
                    if local_work < row_count:
                        qid = Int32(0)
                        if local_work < local_query_count:
                            qid = local_query_start + local_work
                        else:
                            qid = query_ids[edge_begin + local_work - local_query_count]
                        main_head = proxy_head * Int32(self.main_per_proxy) + head_off
                        _atomic_add_fp32(
                            dq_values_mn[ri, ci],
                            _elem_pointer(dq, (batch_idx, main_head, qid, coord[1])),
                        )
            cute.arch.sync_threads()
            row_work_base += Int32(self.queries_per_group)

        dkv_coords_mn = cute.make_tensor(
            c_dkv.iterator, _layout_acc_mn(mma_dkv, c_dkv.layout)
        )
        dk_values_mn = cute.make_tensor(acc_dk.iterator, _layout_acc_mn(mma_dkv, acc_dk.layout))
        dv_values_mn = cute.make_tensor(acc_dv.iterator, _layout_acc_mn(mma_dkv, acc_dv.layout))
        for ri in cutlass.range_constexpr(cute.size(dk_values_mn, mode=[0])):
            for ci in cutlass.range_constexpr(cute.size(dk_values_mn, mode=[1])):
                coord = dkv_coords_mn[ri, ci]
                target = (batch_idx, kv_head, key_start + coord[0], coord[1])
                _atomic_add_fp32(dk_values_mn[ri, ci], _elem_pointer(dk, target))
                _atomic_add_fp32(dv_values_mn[ri, ci], _elem_pointer(dv, target))


def wgmma_backward_dq(ds: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Validate the native ``dS @ K`` product on one 64x128 Hopper tile."""

    if ds.shape != (64, 64) or k.shape != (64, 128):
        raise ValueError("dS and K must have shapes [64,64] and [64,128]")
    if ds.device.type != "cuda" or k.device != ds.device:
        raise ValueError("dS and K must be on the same CUDA device")
    if ds.dtype != k.dtype or ds.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("dS and K must have matching fp16/bf16 dtypes")
    if torch.cuda.get_device_capability(ds.device) != (9, 0):
        raise RuntimeError("SM90 WGMMA backward requires compute capability 9.0")

    ds_c = ds.contiguous()
    kt_c = k.transpose(0, 1).contiguous()
    dq = torch.empty((64, 128), device=ds.device, dtype=torch.float32)
    ds_t = _to_cute_tensor(ds_c)
    kt_t = _to_cute_tensor(kt_c)
    dq_t = _to_cute_tensor(dq)
    stream = cuda.CUstream(torch.cuda.current_stream(ds.device).cuda_stream)
    key = ("backward_dq", ds_t.element_type)
    if key not in _COMPILE_CACHE:
        _COMPILE_CACHE[key] = cute.compile(
            _BackwardDQWgmmaKernel(), ds_t, kt_t, dq_t, stream
        )
    _COMPILE_CACHE[key](ds_t, kt_t, dq_t, stream)
    return dq


def wgmma_backward_attention_tile(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute one selected attention tile and return FP32 dQ/dK/dV."""

    matrices = (q, k, v, grad_out)
    if any(tensor.shape != (64, 128) for tensor in matrices):
        raise ValueError("Q/K/V/dO must each have shape [64,128]")
    if lse.shape != (64,) or delta.shape != (64,):
        raise ValueError("LSE and delta must each have shape [64]")
    if any(tensor.device != q.device for tensor in (*matrices[1:], lse, delta)):
        raise ValueError("all backward-tile tensors must share one device")
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise RuntimeError("SM90 WGMMA backward requires a compute capability 9.0 GPU")
    if any(tensor.dtype != q.dtype for tensor in matrices) or q.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        raise TypeError("Q/K/V/dO must have one matching fp16/bf16 dtype")
    if lse.dtype != torch.float32 or delta.dtype != torch.float32:
        raise TypeError("LSE and delta must use FP32")

    q_c, k_c, v_c, do_c = (tensor.contiguous() for tensor in matrices)
    lse_c, delta_c = lse.contiguous(), delta.contiguous()
    dq = torch.empty_like(q_c, dtype=torch.float32)
    dk = torch.empty_like(k_c, dtype=torch.float32)
    dv = torch.empty_like(v_c, dtype=torch.float32)
    args = tuple(
        _to_cute_tensor(tensor)
        for tensor in (q_c, k_c, v_c, do_c, lse_c, delta_c, dq, dk, dv)
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key = ("backward_attention_tile", args[0].element_type)
    if key not in _COMPILE_CACHE:
        _COMPILE_CACHE[key] = cute.compile(
            _BackwardAttentionTileKernel(), *args, float(scale), stream
        )
    _COMPILE_CACHE[key](*args, float(scale), stream)
    return dq, dk, dv


def wgmma_kv_row_backward_main(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    row_ptr: torch.Tensor,
    query_ids: torch.Tensor,
    *,
    n_proxy_heads: int,
    scale: float,
    segment_starts: torch.Tensor | None = None,
    segment_lengths: torch.Tensor | None = None,
    segment_batches: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fixed or segment-aware KV-stationary reverse-CSR backward."""

    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise RuntimeError("KV-row backward requires a compute capability 9.0 GPU")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or grad_out.shape != q.shape:
        raise ValueError("Q/dO and K/V must be matching [B,H,S,D] tensors")
    batch, n_heads, seq_len, dim = q.shape
    if dim != 128 or seq_len % 128:
        raise NotImplementedError("KV-row backward currently requires D=128 and S divisible by 128")
    if any(t.device != q.device for t in (k, v, grad_out, lse, delta, row_ptr, query_ids)):
        raise ValueError("all KV-row backward tensors must share one CUDA device")
    if any(t.dtype != q.dtype for t in (k, v, grad_out)) or q.dtype not in (
        torch.float16, torch.bfloat16
    ):
        raise TypeError("Q/K/V/dO must have one fp16/bf16 dtype")
    if lse.shape != (batch, n_heads, seq_len) or delta.shape != lse.shape:
        raise ValueError("LSE and delta must have shape [B,H,S]")
    if lse.dtype != torch.float32 or delta.dtype != torch.float32:
        raise TypeError("LSE and delta must be FP32")
    if row_ptr.dtype != torch.int32 or query_ids.dtype != torch.int32:
        raise TypeError("reverse-CSR tensors must be int32")
    n_kv_heads = k.shape[1]
    n_proxy_heads = int(n_proxy_heads)
    if n_heads % n_proxy_heads or n_proxy_heads % n_kv_heads:
        raise NotImplementedError("head counts must satisfy H % Hp == 0 and Hp % Hkv == 0")
    main_per_proxy = n_heads // n_proxy_heads
    if 64 % main_per_proxy:
        raise NotImplementedError("main heads per proxy must divide 64")
    segmented = any(
        item is not None
        for item in (segment_starts, segment_lengths, segment_batches)
    )
    if segmented and not all(
        item is not None
        for item in (segment_starts, segment_lengths, segment_batches)
    ):
        raise ValueError("all three segment metadata tensors must be provided together")
    if segmented:
        assert segment_starts is not None
        assert segment_lengths is not None
        assert segment_batches is not None
        if any(
            item.device != q.device or item.dtype != torch.int32 or item.ndim != 1
            for item in (segment_starts, segment_lengths, segment_batches)
        ) or not (
            segment_starts.shape == segment_lengths.shape == segment_batches.shape
        ):
            raise ValueError("segment metadata must be matching CUDA int32 vectors")
        num_key_units = int(segment_starts.numel())
        if num_key_units < 1:
            raise ValueError("segment metadata must contain at least one segment")
    else:
        num_key_units = seq_len // 128
    expected_rows = (
        n_proxy_heads * num_key_units
        if segmented
        else batch * n_proxy_heads * num_key_units
    )
    if row_ptr.numel() != expected_rows + 1:
        raise ValueError(f"row_ptr must have {expected_rows + 1} entries")

    packed = tuple(t.detach().contiguous() for t in (q, k, v, grad_out))
    lse_c = lse.detach().to(torch.float32).contiguous()
    delta_c = delta.detach().to(torch.float32).contiguous()
    row_ptr_c = row_ptr.detach().contiguous()
    query_ids_c = query_ids.detach().contiguous()
    segment_starts_c = (
        segment_starts.detach().contiguous() if segmented else query_ids_c
    )
    segment_lengths_c = (
        segment_lengths.detach().contiguous() if segmented else query_ids_c
    )
    segment_batches_c = (
        segment_batches.detach().contiguous() if segmented else query_ids_c
    )
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    args = tuple(
        _to_cute_tensor(t)
        for t in (
            *packed, lse_c, delta_c, row_ptr_c, query_ids_c,
            segment_starts_c, segment_lengths_c, segment_batches_c,
            dq, dk, dv,
        )
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key = (
        "kv_row_backward_main", args[0].element_type, batch, n_heads,
        n_kv_heads, n_proxy_heads, seq_len, num_key_units, segmented,
    )
    if key not in _COMPILE_CACHE:
        kernel = _KVRowBackwardKernel(
            batch, n_heads, n_kv_heads, n_proxy_heads, seq_len,
            num_key_units, segmented,
        )
        _COMPILE_CACHE[key] = cute.compile(kernel, *args, float(scale), stream)
    _COMPILE_CACHE[key](*args, float(scale), stream)
    return dq, dk, dv


__all__ = [
    "wgmma_backward_attention_tile",
    "wgmma_backward_dq",
    "wgmma_kv_row_backward_main",
]
