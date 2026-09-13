"""Hopper E4M3 WGMMA proxy block selector for the fixed-length path."""

from __future__ import annotations

import os
from typing import Optional

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Int32, cute, pipeline
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils

from flash_msa.reverse_index_cuda import DocumentSegmentMetadata


BLOCK = 128
ROWS = 64
K_STAGES = int(os.environ.get("MSA_FP8_SELECT_K_STAGES", "2"))
if K_STAGES not in (1, 2, 3):
    raise ValueError("MSA_FP8_SELECT_K_STAGES must be 1, 2, or 3")
PRODUCER_REGISTERS = int(os.environ.get("MSA_FP8_SELECT_PRODUCER_REGISTERS", "48"))
CONSUMER_REGISTERS = int(os.environ.get("MSA_FP8_SELECT_CONSUMER_REGISTERS", "224"))
_COMPILE_CACHE: dict[tuple, object] = {}


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16)


class _FP8SelectKernel:
    """Stream causal FP8 proxy K blocks through one WGMMA consumer."""

    def __init__(
        self,
        *,
        batch: int,
        n_proxy_heads: int,
        n_proxy_kv_heads: int,
        seq_len: int,
        top_k_blocks: int,
        segmented: bool = False,
        num_segments: int = 0,
    ) -> None:
        self.batch = int(batch)
        self.n_proxy_heads = int(n_proxy_heads)
        self.proxy_groups = int(n_proxy_heads) // int(n_proxy_kv_heads)
        self.seq_len = int(seq_len)
        self.num_blocks = int(seq_len) // BLOCK
        self.num_query_tiles = int(seq_len) // ROWS
        self.top_k_blocks = int(top_k_blocks)
        self.segmented = bool(segmented)
        self.num_key_units = int(num_segments) if segmented else self.num_blocks

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
    def _row_max(self, acc, tiled_mma):
        acc_mn = cute.make_tensor(
            acc.iterator, self._layout_acc_mn(tiled_mma, acc.layout)
        )
        row_shape = cute.make_layout(cute.size(acc_mn, mode=[0]))
        row_max = cute.make_rmem_tensor_like(row_shape, cutlass.Float32)
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
        return row_max

    @cute.jit
    def _insert_topk(self, values, indices, row, score, block_idx):
        insert_value = score
        insert_index = block_idx
        for slot in cutlass.range_constexpr(self.top_k_blocks):
            old_value = values[row, slot]
            old_index = indices[row, slot]
            if insert_value > old_value or (
                insert_value == old_value and insert_index < old_index
            ):
                values[row, slot] = insert_value
                indices[row, slot] = insert_index
                insert_value = old_value
                insert_index = old_index

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        q_scales: cute.Tensor,
        k_scales: cute.Tensor,
        block_indices: cute.Tensor,
        token_segment_ids: Optional[cute.Tensor],
        doc_first_segment: Optional[cute.Tensor],
        segment_starts: Optional[cute.Tensor],
        segment_lengths: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            q.element_type != cutlass.Float8E4M3FN
            or k.element_type != cutlass.Float8E4M3FN
        ):
            raise TypeError("FP8 selector requires E4M3FN Q/K")
        batch, n_proxy_heads, seq_len, head_dim = q.shape
        _, n_proxy_kv_heads, _, _ = k.shape
        q_layout = cute.make_layout(
            (seq_len, head_dim, n_proxy_heads, batch),
            stride=(
                head_dim,
                1,
                seq_len * head_dim,
                n_proxy_heads * seq_len * head_dim,
            ),
        )
        k_layout = cute.make_layout(
            (seq_len, head_dim, n_proxy_kv_heads, batch),
            stride=(
                head_dim,
                1,
                seq_len * head_dim,
                n_proxy_kv_heads * seq_len * head_dim,
            ),
        )
        q = cute.make_tensor(q.iterator, q_layout)
        k = cute.make_tensor(k.iterator, k_layout)
        dtype = q.element_type
        q_layout_enum = utils.LayoutEnum.from_tensor(q)
        k_layout_enum = utils.LayoutEnum.from_tensor(k)
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            q_layout_enum.sm90_mma_major_mode(),
            k_layout_enum.sm90_mma_major_mode(),
            cutlass.Float32,
            (1, 1, 1),
            (ROWS, BLOCK),
        )
        q_smem_staged = sm90_utils.make_smem_layout_a(
            q_layout_enum, (ROWS, BLOCK, head_dim), dtype, 1
        )
        k_smem_staged = sm90_utils.make_smem_layout_b(
            k_layout_enum, (ROWS, BLOCK, head_dim), dtype, K_STAGES
        )
        q_smem = cute.slice_(q_smem_staged, (None, None, 0))
        k_smem = cute.slice_(k_smem_staged, (None, None, 0))
        tma_q, tensor_q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            q,
            q_smem,
            (ROWS, head_dim),
        )
        tma_k, tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            k,
            k_smem,
            (BLOCK, head_dim),
        )
        self._tma_q_bytes = cute.size_in_bytes(dtype, q_smem)
        self._tma_k_bytes = cute.size_in_bytes(dtype, k_smem)

        @cute.struct
        class SharedStorage:
            q_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            k_barriers: cute.struct.MemRange[cutlass.Int64, 2 * K_STAGES]
            sQ: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(q_smem_staged)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(k_smem_staged)], 1024
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_q,
            tensor_q,
            tma_k,
            tensor_k,
            q_scales,
            k_scales,
            block_indices,
            token_segment_ids,
            doc_first_segment,
            segment_starts,
            segment_lengths,
            softmax_scale,
            tiled_mma,
            q_smem_staged,
            k_smem_staged,
        ).launch(
            grid=[self.num_query_tiles, self.n_proxy_heads, self.batch],
            block=[256, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_q: cute.CopyAtom,
        q: cute.Tensor,
        tma_k: cute.CopyAtom,
        k: cute.Tensor,
        q_scales: cute.Tensor,
        k_scales: cute.Tensor,
        block_indices: cute.Tensor,
        token_segment_ids: Optional[cute.Tensor],
        doc_first_segment: Optional[cute.Tensor],
        segment_starts: Optional[cute.Tensor],
        segment_lengths: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        q_smem_staged: cute.ComposedLayout,
        k_smem_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        query_tile, proxy_head, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // Int32(128))
        query_start = query_tile * Int32(ROWS)
        query_block = query_start // Int32(BLOCK)
        proxy_kv_head = proxy_head // Int32(self.proxy_groups)
        key_end = query_block + Int32(1)
        key_begin = Int32(0)
        if cutlass.const_expr(self.segmented):
            first_query_segment = token_segment_ids[batch, query_start]
            first_document_segment = doc_first_segment[first_query_segment]
            key_begin = segment_starts[first_document_segment] // Int32(BLOCK)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sQ = storage.sQ.get_tensor(
            q_smem_staged.outer, swizzle=q_smem_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_staged.outer, swizzle=k_smem_staged.inner
        )
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        q_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_barriers.data_ptr(),
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self._tma_q_bytes,
        )
        k_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_barriers.data_ptr(),
            num_stages=K_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self._tma_k_bytes,
        )
        q_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        q_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        k_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, K_STAGES
        )
        k_consumer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, K_STAGES
        )

        gQ = cute.local_tile(
            q[None, None, proxy_head, batch], (ROWS, BLOCK), (None, 0)
        )
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        gK = cute.local_tile(
            k[None, None, proxy_kv_head, batch], (BLOCK, BLOCK), (None, 0)
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )

        if warpgroup_idx == 0:
            cute.arch.setmaxregister_decrease(PRODUCER_REGISTERS)
            if warp_idx == 0:
                q_pipe.producer_acquire(q_producer)
                cute.copy(
                    tma_q,
                    tQgQ[(None, query_tile)],
                    tQsQ[(None, q_producer.index)],
                    tma_bar_ptr=q_pipe.producer_get_barrier(q_producer),
                )
                q_pipe.producer_commit(q_producer)
                key_block = key_begin
                while key_block < key_end:
                    k_pipe.producer_acquire(k_producer)
                    cute.copy(
                        tma_k,
                        tKgK[(None, key_block)],
                        tKsK[(None, k_producer.index)],
                        tma_bar_ptr=k_pipe.producer_get_barrier(k_producer),
                    )
                    k_pipe.producer_commit(k_producer)
                    k_producer.advance()
                    key_block += Int32(1)

        if warpgroup_idx == 1:
            cute.arch.setmaxregister_increase(CONSUMER_REGISTERS)
            q_pipe.consumer_wait(q_consumer)
            wg_thread = tidx - Int32(128)
            thr_mma = tiled_mma.get_slice(wg_thread)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
            acc_shape = thr_mma.partition_shape_C((ROWS, BLOCK))
            coordinates = cute.make_identity_tensor((ROWS, BLOCK))
            coordinates_mn = cute.make_tensor(
                thr_mma.partition_C(coordinates).iterator,
                self._layout_acc_mn(
                    tiled_mma, thr_mma.partition_C(coordinates).layout
                ),
            )
            row_count = cute.size(coordinates_mn, mode=[0])
            top_values = cute.make_rmem_tensor(
                (row_count, self.top_k_blocks), cutlass.Float32
            )
            top_indices = cute.make_rmem_tensor(
                (row_count, self.top_k_blocks), Int32
            )
            top_values.fill(-cutlass.Float32.inf)
            top_indices.fill(Int32(self.num_key_units))
            key_block = key_begin
            while key_block < key_end:
                k_pipe.consumer_wait(k_consumer)
                accumulator = thr_mma.make_fragment_C(acc_shape)
                cute.nvgpu.warpgroup.fence()
                self._gemm_zero(
                    tiled_mma,
                    tSrQ[(None, None, None, 0)],
                    tSrK[(None, None, None, k_consumer.index)],
                    accumulator,
                )
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                if cutlass.const_expr(self.segmented):
                    accumulator_mn = cute.make_tensor(
                        accumulator.iterator,
                        self._layout_acc_mn(tiled_mma, accumulator.layout),
                    )
                    for row in cutlass.range_constexpr(
                        cute.size(accumulator_mn, mode=[0])
                    ):
                        query_offset = coordinates_mn[row, 0][0]
                        query_position = query_start + query_offset
                        query_segment = token_segment_ids[batch, query_position]
                        query_document = doc_first_segment[query_segment]
                        block_last = cutlass.min(
                            (key_block + Int32(1)) * Int32(BLOCK) - Int32(1),
                            Int32(self.seq_len - 1),
                        )
                        candidate = token_segment_ids[batch, block_last]
                        if key_block == query_block:
                            candidate = query_segment
                        candidate_start = segment_starts[candidate]
                        candidate_end = candidate_start + segment_lengths[candidate]
                        same_document = doc_first_segment[candidate] == query_document
                        for col in cutlass.range_constexpr(
                            cute.size(accumulator_mn, mode=[1])
                        ):
                            key_offset = coordinates_mn[row, col][1]
                            key_position = key_block * Int32(BLOCK) + key_offset
                            if (
                                not same_document
                                or key_position < candidate_start
                                or key_position >= candidate_end
                                or key_position > query_position
                            ):
                                accumulator_mn[row, col] = -cutlass.Float32.inf
                row_max = self._row_max(accumulator, tiled_mma)
                score_scale = (
                    softmax_scale
                    * q_scales[batch, proxy_head, query_block]
                    * k_scales[batch, proxy_kv_head, key_block]
                )
                for row in cutlass.range_constexpr(cute.size(row_max)):
                    score = row_max[row] * score_scale
                    candidate = key_block
                    valid_candidate = True
                    if cutlass.const_expr(self.segmented):
                        query_offset = coordinates_mn[row, 0][0]
                        query_position = query_start + query_offset
                        query_segment = token_segment_ids[batch, query_position]
                        query_document = doc_first_segment[query_segment]
                        block_last = cutlass.min(
                            (key_block + Int32(1)) * Int32(BLOCK) - Int32(1),
                            Int32(self.seq_len - 1),
                        )
                        candidate = token_segment_ids[batch, block_last]
                        if key_block == query_block:
                            candidate = query_segment
                        valid_candidate = (
                            doc_first_segment[candidate] == query_document
                        )
                        if candidate == query_segment:
                            score = cutlass.Float32.inf
                    elif key_block == query_block:
                        score = cutlass.Float32.inf
                    if valid_candidate:
                        self._insert_topk(
                            top_values, top_indices, row, score, candidate
                        )
                k_pipe.consumer_release(k_consumer)
                k_consumer.advance()
                key_block += Int32(1)

            for row in cutlass.range_constexpr(row_count):
                coordinate = coordinates_mn[row, 0]
                if coordinate[1] == 0:
                    q_position = query_start + coordinate[0]
                    for slot in cutlass.range_constexpr(self.top_k_blocks):
                        block_indices[
                            batch, proxy_head, q_position, slot
                        ] = top_indices[row, slot]
            q_pipe.consumer_release(q_consumer)


def select_blocks_fp8_sm90(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q_scales: torch.Tensor,
    k_scales: torch.Tensor,
    *,
    scale: float,
    top_k_blocks: int,
    document_segments: DocumentSegmentMetadata | None = None,
) -> torch.Tensor:
    """Select causal proxy blocks with E4M3 WGMMA and FP32 ranking."""

    if q_proxy.dtype != torch.float8_e4m3fn or k_proxy.dtype != torch.float8_e4m3fn:
        raise TypeError("q_proxy and k_proxy must use float8_e4m3fn")
    if q_proxy.ndim != 4 or k_proxy.ndim != 4:
        raise ValueError("q_proxy and k_proxy must have shape [B,H,S,D]")
    batch, n_proxy_heads, seq_len, head_dim = q_proxy.shape
    if head_dim != 128 or seq_len % BLOCK:
        raise ValueError("FP8 selector requires D=128 and sequence divisible by 128")
    if q_proxy.shape[0] != k_proxy.shape[0] or q_proxy.shape[2:] != k_proxy.shape[2:]:
        raise ValueError("Q/K batch, sequence, and head dimension must match")
    n_proxy_kv_heads = int(k_proxy.shape[1])
    if n_proxy_heads % n_proxy_kv_heads:
        raise ValueError("proxy query heads must be divisible by proxy KV heads")
    expected_q_scale = (batch, n_proxy_heads, seq_len // BLOCK)
    expected_k_scale = (batch, n_proxy_kv_heads, seq_len // BLOCK)
    if q_scales.shape != expected_q_scale or k_scales.shape != expected_k_scale:
        raise ValueError("FP8 scales must have shape [B,H,S/128]")
    if q_scales.dtype != torch.float32 or k_scales.dtype != torch.float32:
        raise TypeError("FP8 scales must be FP32")
    if torch.cuda.get_device_capability(q_proxy.device) != (9, 0):
        raise RuntimeError("FP8 WGMMA selector requires compute capability 9.0")

    q_c = q_proxy.detach().contiguous()
    k_c = k_proxy.detach().contiguous()
    q_scale_c = q_scales.detach().contiguous()
    k_scale_c = k_scales.detach().contiguous()
    segmented = document_segments is not None
    token_segments_t = None
    doc_first_t = None
    segment_starts_t = None
    segment_lengths_t = None
    if segmented:
        token_segments_t = _to_cute_tensor(
            document_segments.token_segment_ids.contiguous()
        )
        doc_first_t = _to_cute_tensor(
            document_segments.doc_first_segment.contiguous()
        )
        segment_starts_t = _to_cute_tensor(document_segments.starts.contiguous())
        segment_lengths_t = _to_cute_tensor(document_segments.lengths.contiguous())
    output = torch.empty(
        (batch, n_proxy_heads, seq_len, int(top_k_blocks)),
        device=q_proxy.device,
        dtype=torch.int32,
    )
    q_t = _to_cute_tensor(q_c)
    k_t = _to_cute_tensor(k_c)
    qs_t = _to_cute_tensor(q_scale_c)
    ks_t = _to_cute_tensor(k_scale_c)
    output_t = _to_cute_tensor(output)
    stream = cuda.CUstream(torch.cuda.current_stream(q_proxy.device).cuda_stream)
    key = (
        batch,
        n_proxy_heads,
        n_proxy_kv_heads,
        seq_len,
        int(top_k_blocks),
        q_t.element_type,
        segmented,
        0 if document_segments is None else document_segments.num_segments,
    )
    if key not in _COMPILE_CACHE:
        kernel = _FP8SelectKernel(
            batch=batch,
            n_proxy_heads=n_proxy_heads,
            n_proxy_kv_heads=n_proxy_kv_heads,
            seq_len=seq_len,
            top_k_blocks=int(top_k_blocks),
            segmented=segmented,
            num_segments=(
                0 if document_segments is None else document_segments.num_segments
            ),
        )
        _COMPILE_CACHE[key] = cute.compile(
            kernel,
            q_t,
            k_t,
            qs_t,
            ks_t,
            output_t,
            token_segments_t,
            doc_first_t,
            segment_starts_t,
            segment_lengths_t,
            float(scale),
            stream,
        )
    _COMPILE_CACHE[key](
        q_t,
        k_t,
        qs_t,
        ks_t,
        output_t,
        token_segments_t,
        doc_first_t,
        segment_starts_t,
        segment_lengths_t,
        float(scale),
        stream,
    )
    return output


__all__ = ["select_blocks_fp8_sm90"]
