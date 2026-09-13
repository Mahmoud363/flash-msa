# GH200 BF16 Baseline

## Configuration

- GPU: NVIDIA GH200 480GB, SM90
- PyTorch: 2.11.0+cu130
- dtype: BF16
- shape: `B=1`, `S=2048`, `Hq=16`, `Hkv=2`, `Hp=4`, `D=128`
- selected tokens: 512
- masking: four reproducible uneven documents

The unprofiled CUDA-event run used five warmups and twenty measured iterations.
Its complete machine-readable result is stored alongside this note.

- forward median: 3.644 ms
- backward median: 3.264 ms
- forward + backward median: 6.907 ms

## Nsight Systems result

The valid trace used three warmups and five measured iterations after a separate
JIT prewarm. Profiling overhead raised wall-clock timings, so kernel proportions
and launch counts—not profiled latency—are the useful results.

- 1,853 `cudaLaunchKernel` calls over eight total warmup/measured iterations;
  approximately 232 launches per training iteration.
- `MSAFusedBackwardMMAKernel`: 38.0% of observed GPU kernel time, approximately
  518 microseconds per invocation.
- PyTorch indexing kernels: 7.5% of GPU kernel time across 96 launches.
- PyTorch elementwise kernels: 5.1% across 64 launches.
- `MSASelectBlocksKernel`: 4.0%, approximately 54 microseconds per invocation.
- Segment metadata scan kernels together account for about 5.3%.
- FA3 forward kernels are individually small but invoked repeatedly for local
  and remote attention work.

The trace supports two immediate conclusions:

1. backward compute is the largest single-kernel target;
2. metadata, packing, and bounded attention generate substantial launch
   fragmentation, so reducing launches is also a first-order target.

## Nsight Compute status

Nsight Compute 2025.3.1 was available through the module system, but the driver
denied access to hardware performance counters with `ERR_NVGPUCTRPERM`. Detailed
tensor-core, occupancy, cache, and stall metrics require the cluster
administrator to enable NVIDIA performance counters on the profiling node.

## Fixed-length 16K optimization baseline

The primary optimization case is fixed-length `B=1`, `S=16384`, Top-K 2048,
with the same head configuration and BF16 dtype. Five warmups and twenty
unprofiled iterations produced:

- forward median: 11.404 ms;
- backward median: 24.416 ms;
- forward + backward median: 35.821 ms.

The corresponding Nsight Systems trace used three warmups and five measured
iterations. Across all eight iterations it recorded 2,453 kernel launches,
approximately 307 launches per iteration. GPU kernel time was distributed as:

- fused MSA backward: 60.8%, approximately 20.41 ms per invocation;
- PyTorch indexing kernels: 11.3%;
- online attention merge kernels: 8.1%;
- remote FA3 forward kernels: 4.2% for the largest variant, with additional
  smaller FA3 variants outside that percentage;
- block selection: 2.8%, approximately 0.94 ms per invocation.

This supersedes the 2K case for optimization decisions. The first fixed-length
work should target backward throughput while avoiding additional indexing and
launch fragmentation. The native KV-outer forward remains the next structural
forward optimization.

## Milestone 1: larger remote-task chunks

Increasing the default remote-task chunk from 256 to 1024 amortizes query/KV
packing, FA3 dispatch, and online-merge launches. The environment variable
`MSA_FLASH_TASKS_PER_CHUNK` can restore a smaller value for memory-constrained
cases.

The same fixed-length 16K/Top-K 2K benchmark produced:

- forward median: 10.963 ms, 3.87% faster than baseline;
- backward median: 23.478 ms, 3.84% faster than baseline;
- combined median: 34.441 ms, 3.85% faster than baseline.

Fixed-length, document-masked, and batch-size-two document-masked correctness
tests passed. The chunk iterator is shared by both fixed-length and varlen paths,
so the launch-amortization change applies to both; varlen may split chunks at
full/partial segment access-mode boundaries.

## Milestone 2: native SM90a KV-outer BF16 forward

The fixed-length production path now consumes the reverse CSR schedule in a
native Hopper CuTe kernel. Each work item selects one KV tile, loads K/V with
TMA, reuses it across all associated query/head groups, executes QK and PV with
WGMMA, computes softmax/LSE in FP32, and stores BF16 partial output plus FP32
LSE for the existing online merge. Producer and consumer warpgroups use 40 and
232 registers, respectively.

The retained pipeline has two shared-memory Q stages. It overlaps gathering the
next arbitrary CSR query group with QK/softmax/PV on the current group. At 16K,
two stages measured 8.920 ms versus 9.343 ms for one stage. K/V remain
single-stage because one tile is intentionally reused for the complete work
item; double-buffering both would add 64 KiB per CTA without overlap inside a
work item and would reduce occupancy. The 64x128 WGMMA tile and 512-query CSR
chunk were the retained tile/work-size configuration.

Dynamic persistent claiming is bounded to four CTAs per SM and activates only
when the task grid exceeds that threshold. Smaller grids retain direct static
assignment, avoiding the counter overhead measured at 16K (9.640 ms when
forced versus 9.410 ms static before Q pipelining). A forced one-CTA test
validates repeated claims and pipeline phase reuse across three heterogeneous
tasks.

Five warmups and twenty CUDA-event iterations on the same GH200 produced:

| Path | Shape | Forward median | Backward median | Combined | Forward speedup |
|---|---:|---:|---:|---:|---:|
| FA3 | 16K / Top-K 2K | 10.721 ms | 23.348 ms | 34.069 ms | baseline |
| native SM90a | 16K / Top-K 2K | 8.920 ms | 23.458 ms | 32.378 ms | 16.8% |
| FA3 | 32K / Top-K 2K | 22.562 ms | 48.137 ms | 70.699 ms | baseline |
| native SM90a | 32K / Top-K 2K | 18.586 ms | 48.790 ms | 67.375 ms | 17.6% |

The backward kernel is unchanged in this forward milestone; its small paired
variation is benchmark noise. Combined time improves by 5.0% at 16K and 4.7%
at 32K.

Correctness gates passed for fixed-length B=1 and B=2 training, including
backward and optimizer updates; document-masked B=1 and B=2 fallback training;
head-group ratios 2, 4, and 8; multiple query groups per KV item; and repeated
persistent claims. The focused SM90 suite reports 12 passed tests.

A headless Nsight Systems 2025.5.1 trace with three warmups and three measured
iterations recorded 773 `cudaLaunchKernel` calls across six iterations,
approximately 129 launches per iteration. The native selected-attention kernel
ran six times, averaged 2.773 ms, and represented 8.8% of observed GPU kernel
time. Nsight Compute hardware counters remain unavailable on this node because
the driver returns `ERR_NVGPUCTRPERM`; this does not affect Nsight Systems CUDA
and NVTX tracing.

## Milestone 3: E4M3 proxy/index selection

The fixed-length SM90 path now automatically quantizes proxy Q/K to E4M3 and
runs block selection on Hopper FP8 tensor cores. Each fused quantization CTA
computes an FP32 amax over one `[128,128]` token/dimension block, writes one
FP32 scale, and emits E4M3 values. The selector uses 64 query rows, 128 key
columns, a two-stage TMA K pipeline, and 48/224 producer/consumer registers.
QK uses FP8 WGMMA with FP32 accumulation; scale application, block maxima, and
Top-K ranking remain FP32.

The generated SM90a PTX contains
`wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3`, confirming that this is
the FP8 tensor-core instruction rather than a storage-only conversion. Set
`MSA_SELECT_BACKEND=bf16` to restore BF16 selection. FA3 and document-masked
execution retain BF16 by default; this milestone does not yet port FP8
selection to variable document segments.

### Approximate-selection policy and numerical gates

FP8 selection is intentionally approximate. Exact-row agreement is reported as
a diagnostic, not a correctness requirement. The enforced training gates are:

- output, scalar loss, and all five input gradients must be finite;
- output cosine similarity versus BF16 selection must be at least 0.97;
- relative scalar-loss error must be at most 1%;
- every proxy-Q/proxy-K/main-Q/main-K/main-V gradient cosine must be at least
  0.99.

Random BF16 proxy inputs with seed 67 produced the following schedule metrics:

| Shape | Top-K | Recall@K | Exact rows | Mean symmetric difference |
|---:|---:|---:|---:|---:|
| 8K | 512 | 95.60% | 82.62% | 0.352 blocks |
| 16K | 2048 | 96.07% | 47.20% | 1.257 blocks |
| 16K | 4096 | 97.37% | 42.19% | 1.684 blocks |
| 32K | 2048 | 95.19% | 38.04% | 1.538 blocks |

The B=1, 8K/Top-K 2K forward/backward comparison measured output cosine
0.9815, relative loss error 0.070%, and gradient cosines from 0.9977 to 0.99995.
The B=2, 4K/Top-K 1K comparison measured output cosine 0.9804, relative loss
error 0.010%, and gradient cosines from 0.9956 to 0.99996. Both satisfy the
documented approximate-training gates.

### Performance

The tuned 16K/Top-K 2K proxy stage takes 0.638 ms including fused Q/K
quantization, versus 1.031 ms for BF16 selection, a 38.1% speedup. Paired
end-to-end CUDA-event measurements produced:

| Shape | Top-K | BF16-select forward | FP8-select forward | Speedup |
|---:|---:|---:|---:|---:|
| 8K | 512 | 2.319 ms | 2.266 ms | 2.3% |
| 16K | 2048 | 8.870 ms | 8.606 ms | 3.0% |
| 16K | 4096 | 15.150 ms | 14.167 ms | 6.5% |
| 32K | 2048 | 18.668 ms | 17.535 ms | 6.1% |

Backward arithmetic remains BF16 and its timings are unchanged within normal
run variation; only the forward selection schedule is approximate.

Headless Nsight Systems 2025.5.1 reports 0.395 ms average for the FP8 selector,
20.6 microseconds for proxy-Q quantization, and 13.7 microseconds for proxy-K
quantization across six traced iterations. The trace contains 773
`cudaLaunchKernel` calls, the same observed total as the final Milestone 2
trace. Generated-code inspection supplies the FP8 WGMMA evidence because
Nsight Compute counters remain blocked by `ERR_NVGPUCTRPERM` on this node.

The full automated suite reports 21 passing tests. Coverage includes exact
fused-quantizer parity, proxy head mappings 2:1/4:1/4:2/8:2, approximate
forward/backward stability, the existing SM90 attention tests, and BF16
document-mask fallback parity at B=2.

## Portable scheduling and I/O follow-up

The fixed-length SM90 path now applies two ideas shared with the Blackwell
kernel without depending on Blackwell-only instructions. Commit `d20c1cf`
replaces edge-oriented partial-output merging with one work item per destination
query/head group. Commit `29d7c29` replaces scalar gathered-Q traffic with
layout-aware 128-bit `cp.async` copies into the swizzled shared-memory Q stages.

The merge launch shrank from roughly 921,600 CTAs to 65,536 CTAs in the
16K/Top-K 2K case. A headless Nsight Systems trace measured its mean kernel time
at 1.167 ms, down from 2.519 ms (53.7%). The selected-attention kernel retains
the previously chosen KV-outer reverse-CSR work items and four-CTAs-per-SM
persistent cap. Sweeping caps of 2, 3, 4, and 6 produced remote-attention
medians of 2.048, 2.065, 2.035, and 2.037 ms, respectively, so four remains the
default.

CUDA-event results below use the normal fixed-length BF16 path. JIT compilation
is excluded. The 16K/2K result uses five warmups and twenty measurements; the
other long-context cases use three warmups and ten measurements.

| Shape | Top-K | Before full forward | Final full forward | Speedup | Final remote attention |
|---:|---:|---:|---:|---:|---:|
| 16K | 2048 | 8.870 ms | 6.434 ms | 27.5% | 2.035 ms |
| 16K | 4096 | 15.150 ms | 10.782 ms | 28.8% | 3.667 ms |
| 32K | 2048 | 18.668 ms | 13.518 ms | 27.6% | 3.912 ms |

The focused benchmark command is:

```bash
python benchmarks/profile_bf16_io.py \
  --sequence-length 32768 --top-k 2048 --warmup 3 --repeats 10
```

An autograd-enabled 16K/Top-K 2K run with five warmups and twenty measurements
reported a 6.519 ms forward median and a 23.411 ms backward median. The
backward implementation is unchanged, so the backward number is a regression
guard rather than an optimization claim. All 27 focused tests are accounted for:
six FP8-storage/mixed-path tests, twelve persistent-scheduler/native-forward
tests, and nine proxy-selection/training tests.

These optimizations currently accelerate the fixed-length normal path. The
document-masked/varlen fallback still passes its regression gates but does not
yet use the destination-centric merge or native SM90 gathered-Q pipeline. Its
port must retain segment predicates and explicit batch-row boundaries for
partial document tiles.
