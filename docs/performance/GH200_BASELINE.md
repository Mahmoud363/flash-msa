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
