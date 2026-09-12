# GH200 Flash-MSA Optimization Plan

## Goal

Build a measurably faster SM90a Flash-MSA training path for NVIDIA GH200 while
preserving document masking, varlen inputs, causal behavior, and forward/backward
numerical correctness.

The work follows two rules:

1. Optimize from measured bottlenecks, not assumed bottlenecks.
2. Land each major milestone as an independently revertible git commit.

## Target and non-goals

Target hardware is GH200 (Hopper, compute capability 9.0) with head dimension
128 and block size 128. The first target is training; inference-only changes may
be developed when they share useful infrastructure.

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

## Milestone 2: native SM90a KV-outer BF16 forward

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

## Milestone 3: FP8 proxy/index branch

Introduce E4M3 proxy Q/K operands with per-head or per-block scaling. Use Hopper
FP8 WGMMA with FP32 accumulation, and retain FP32 block maxima and Top-K ranking.

Exit criteria: selection agreement/quality is quantified, training correctness
meets an explicitly documented FP8 tolerance, and proxy time improves.

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
- sequence lengths: 2K, 8K, and the largest practical long-context case;
- Top-K tokens: 512, 2048, and 4096 where valid;
- masking: none and uneven packed documents;
- phases: forward-only and forward+backward;
- precision: BF16 baseline, FP8 storage, and mixed FP8/BF16 as introduced.

Optimization commits must include the benchmark command and before/after result
in their commit message or an accompanying performance note.

