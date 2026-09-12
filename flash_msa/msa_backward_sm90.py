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
from quack import sm90_utils


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


__all__ = ["wgmma_backward_dq"]
