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
