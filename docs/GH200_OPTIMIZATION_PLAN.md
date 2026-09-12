# GH200 Flash-MSA Optimization Plan

## Goal

Build a measurably faster SM90a Flash-MSA training path for NVIDIA GH200 while
preserving document masking, varlen inputs, causal behavior, and forward/backward
numerical correctness.

The work follows two rules:

1. Optimize from measured bottlenecks, not assumed bottlenecks.
2. Land each major milestone as an independently revertible git commit.
3. Design every optimization for both fixed-length and varlen/document-masked
   execution, even when the fixed-length implementation is delivered first.

## Target and non-goals

Target hardware is GH200 (Hopper, compute capability 9.0) with head dimension
128 and block size 128. The first target is training; inference-only changes may
be developed when they share useful infrastructure.

Implementation proceeds in two phases. First, optimize the normal fixed-length
path, where uniform full blocks make performance behavior easier to isolate.
After that path reaches its target, port the same scheduling, data movement, MMA,
and fusion design to varlen/document-masked execution. Fixed-length work must not
remove or bypass the existing varlen implementation.

Blackwell-only TCGen05/UMMA, tensor memory (TMEM), NVFP4, and Cluster Launch
Control are not portable to GH200. Portable ideas from the MiniMax SM100 kernel
include KV-outer traversal, GPU-resident CSR scheduling, persistent work queues,
split-KV reduction, quantized KV storage, and specialized common configurations.

## Required correctness gates

Every optimization must pass:

- sparse eager comparison, forward and backward;
- warmup eager comparison, forward and backward when affected;
- document-masked `cu_seqlens` cases with uneven boundaries;
- batch size greater than one, including explicit batch-row boundaries;
- non-document-masked regression tests;
- finite-value checks and configured output/gradient tolerances;
- BF16 baseline comparison before any FP8-specific tolerance is introduced.

Each fixed-length optimization must also document its varlen adaptation:

- which tensors become segment-aware;
- how partial first/last document blocks are predicated;
- how batch-row boundaries enter scheduling;
- which fast path remains valid for full aligned document segments;
- whether metadata or workspace formats remain shared between both paths.

An optimization is not considered complete project-wide until both paths use it,
but the fixed-length implementation and its later varlen port should be separate
rollback commits.

Performance results must report warmup policy, shape, dtype, Top-K, GPU, software
versions, median latency, dispersion, and peak allocated memory. JIT compilation
must be excluded from steady-state timings.

## Milestone 0: reproducible baseline and profiling

Deliverables:

- a deterministic single-case benchmark for sparse forward and backward;
- optional document masking with reproducible uneven document boundaries;
- NVTX ranges around selection, metadata, forward, and backward where practical;
- CLI recipes for Nsight Systems and Nsight Compute text-mode profiling;
- machine-readable JSON or CSV output with environment metadata;
- baseline GH200 results checked into `docs/performance/`.

Exit criteria:

- repeated measurements are stable enough to rank changes;
- the dominant kernels, launch overhead, and memory traffic are identified;
- one first optimization is selected from evidence.

Rollback point: commit the measurement tooling before changing kernel behavior.

## Milestone 1: remove host synchronization and launch overhead

Audit metadata construction and sparse chunk dispatch for `.cpu()`, `.item()`,
`.tolist()`, Python loops, and repeated small FlashAttention launches. Move work
descriptors and chunk boundaries to GPU-resident data where profitable, reuse
metadata for compatible shapes, and fuse adjacent metadata operations.

Exit criteria: unchanged correctness and a measured end-to-end improvement for
at least two representative document-masked shapes.

Implementation order: first fuse/reduce overhead in the fixed-length path, then
extend the same metadata representation and launch structure to document
segments without regressing its aligned-block fast path.

## Milestone 2: native SM90a KV-outer BF16 forward (complete)

Replace packed remote-KV materialization plus repeated FA3 calls with a Hopper
CuTe kernel that consumes the reverse index directly:

1. claim a KV-centric work item;
2. load a selected KV tile with TMA;
3. process all associated query/head groups with WGMMA;
4. maintain online-softmax state in FP32;
5. merge/store BF16 output and FP32 LSE.

Use warp-specialized producer/consumer roles and tune shared-memory pipeline
depth, tile shape, register pressure, and persistent scheduling.

Exit criteria: correctness parity and a speedup over the FA3-backed baseline at
the target long-context shapes.

Completed on GH200 on 2026-09-12 for the fixed-length path. The accepted
specialization uses a 64x128 query/KV WGMMA tile, one resident TMA K/V stage,
a two-stage asynchronous Q-gather pipeline, 40 producer and 232 consumer
registers, 512 queries per native CSR chunk, and adaptive persistent claiming
above four CTAs per SM. The document-masked path deliberately retains its
existing fallback until the later varlen port.

The final BF16 B=1, Top-K 2048 measurements were 8.920 ms at 16K and 18.586 ms
at 32K, compared with 10.721 ms and 22.562 ms for the FA3-backed path. Forward
speedups are 16.8% and 17.6%, respectively. Fixed-length B=1/B=2 training
parity, document-masked B=1/B=2 fallback parity, and 12 focused scheduler/TMA/
WGMMA/FP32-softmax/epilogue tests passed. Detailed evidence is in
`docs/performance/GH200_BASELINE.md`.

## Milestone 3: FP8 proxy/index branch (complete)

Introduce E4M3 proxy Q/K operands with per-head or per-block scaling. Use Hopper
FP8 WGMMA with FP32 accumulation, and retain FP32 block maxima and Top-K ranking.

Exit criteria: selection agreement/quality is quantified, training correctness
meets an explicitly documented FP8 tolerance, and proxy time improves.

Completed on GH200 on 2026-09-12 for the fixed-length SM90 path. Proxy Q/K are
quantized to E4M3 with one FP32 amax scale per head and 128-token block. Fused
CuTe quantizers feed a 64x128, two-stage-K TMA/WGMMA selector with FP32
accumulation, block maxima, and Top-K ranking. Generated PTX contains
`wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3`.

FP8 selection is intentionally approximate, matching the original design
intent rather than requiring exact BF16 block IDs. The enforced training gates
are finite output/loss/all gradients, output cosine >= 0.97, relative loss
error <= 1%, and cosine >= 0.99 for every proxy/main QKV gradient. Measured
Top-K recall spans 95.2%-97.4% across the retained 8K-32K quality matrix; both
B=1 and B=2 training comparisons pass the numerical gates.

At 16K/Top-K 2K the complete proxy-selection stage is 0.638 ms versus 1.031 ms
for BF16. End-to-end fixed-length forward improves by 3.0% at 16K and 6.1% at
32K. Smaller 8K/Top-K 512 and larger 16K/Top-K 4096 cases also improve. The
document-masked path retains exact BF16 selection until the later varlen port.
Detailed evidence is in `docs/performance/GH200_BASELINE.md`.

## Portable scheduling and I/O follow-up (fixed-length complete)

The Blackwell design's scheduling and data-movement principles are useful on
GH200 even though its TCGen05, TMEM, and Cluster Launch Control mechanisms are
not. Two SM90 implementations were retained:

1. The online merge is destination-centric. One CTA gathers all reverse-CSR
   partials for a destination query/head group, forms the multi-edge
   log-sum-exp weights in shared memory, and reads/writes each FP32 output
   element once. This replaces an edge-sized launch in which most CTAs returned
   after a slot scan and surviving CTAs repeatedly round-tripped output through
   global memory.
2. Arbitrary CSR query tiles are gathered with layout-aware 128-bit
   `cp.async` copies into the existing swizzled shared-memory Q stages. The copy
   partition is derived from the destination layout, so this is valid for BF16
   and E4M3 tiles without assuming that logical rows are physically contiguous.

The retained persistent limit remains four CTAs per SM. A 2/3/4/6-CTA sweep at
16K/Top-K 2K showed no benefit from changing it. Headless Nsight Systems measured
the merge at 1.167 ms versus 2.519 ms before the destination-centric rewrite.
The final BF16 normal-path measurements are 6.434 ms at 16K/Top-K 2K,
10.782 ms at 16K/Top-K 4K, and 13.518 ms at 32K/Top-K 2K. These are
27.5%, 28.8%, and 27.6% faster than the corresponding pre-follow-up results.

This work changes forward scheduling and I/O only. The existing backward kernel
and arithmetic are unchanged; a 16K/Top-K 2K training run measured 6.519 ms
forward and 23.411 ms backward. The fixed-length correctness suite accounts for
27 passing tests, including forward/backward numerical gates. The later varlen
port must make destination IDs segment-aware, predicate partial first/last
document tiles, and preserve batch-row boundaries while reusing the same
destination-gather and layout-partitioned-copy structure.

## Milestone 4: FP8 KV storage with BF16 compute

Store K/V as E4M3, load through TMA, and convert into BF16 shared-memory layouts
for BF16 QK and PV WGMMA. This targets HBM/L2 bandwidth without initially
changing attention arithmetic.

Exit criteria: memory traffic decreases, attention output remains within BF16
baseline tolerance, and end-to-end time improves.

## Milestone 5: mixed FP8 QK and BF16 PV

Run QK with FP8 operands and FP32 accumulation; compute softmax in FP32 and PV
with BF16 operands and FP32 accumulation. Keep backward BF16 initially.

Exit criteria: stable training-step comparisons, documented scaling policy, and
a speedup beyond FP8-storage-only mode.

## Milestone 6: native SM90a backward

Reuse KV-outer metadata and TMA/WGMMA pipelines for dQ, dK, dV, and proxy KL
gradients. Avoid global temporary tensors where online or fused reductions are
possible.

Exit criteria: all gradient gates pass and total forward-plus-backward time
improves on representative training configurations.

## Milestone 7: decode and long-context scheduling

Add reusable plan/run scheduling, split-KV execution, GPU-side partial-result
combination, and specialized short-Q decode paths. Evaluate paged KV through a
custom SM90 implementation rather than the SM100-only FA4 paged path.

Exit criteria: schedule construction is amortized, decode correctness passes,
and long-context latency improves across multiple batch/context regimes.

## Milestone 8: cache and cluster experiments

Only after single-CTA kernels are tuned, evaluate L2 persistence for sink/local
blocks and thread-block clusters with distributed shared memory for popular KV
tiles. Retain these paths only when profiling shows a repeatable benefit after
occupancy costs.

## Benchmark matrix

At minimum, measure:

- batch sizes: 1 and 2;
- sequence lengths: 8K, 16K, and 32K, plus the largest practical long-context
  case once memory scaling is characterized;
- Top-K tokens: 512, 2048, and 4096 where valid;
- masking: none and uneven packed documents;
- phases: forward-only and forward+backward;
- precision: BF16 baseline, FP8 storage, and mixed FP8/BF16 as introduced.

Optimization commits must include the benchmark command and before/after result
in their commit message or an accompanying performance note.

During the fixed-length phase, every benchmark change is measured first without
masking. Document-masked regression tests continue to run after every commit so
normal-path work cannot silently break varlen behavior. A matching varlen
performance matrix is required when the porting phase begins.

The 2K case is retained only as a quick smoke test and launch-overhead stress
case. Optimization decisions must be supported by results at 16K or longer.
