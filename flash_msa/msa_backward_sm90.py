"""Hopper WGMMA building blocks for the KV-stationary MSA backward.

This module is intentionally separate from the production warp-MMA fallback so
the native backward can be validated one tensor-core product at a time before
its reverse-CSR scheduler is enabled.
"""

from __future__ import annotations

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as hopper_helpers
from quack import layout_utils, sm90_utils


_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16)


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


__all__ = ["wgmma_backward_attention_tile", "wgmma_backward_dq"]
