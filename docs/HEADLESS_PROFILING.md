# Headless GH200 Profiling

All profiling commands in this document run without a GUI. Run one ordinary
benchmark first so CUDA extensions and CuTe kernels are compiled before a trace.

```bash
conda activate longvu
cd /e/scratch/jureap1/ahmed9/repos/evaluation/flash-msa
python benchmarks/profile_training_case.py \
  --batch-size 1 --sequence-length 2048 --top-k 512 \
  --documents-per-row 4 --warmup 2 --repeats 5 \
  --json baseline.json
```

## Nsight Systems

Capture launch, synchronization, and memory-operation timelines:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true \
  --output=flash_msa_docmask \
  python benchmarks/profile_training_case.py \
    --batch-size 1 --sequence-length 2048 --top-k 512 \
    --documents-per-row 4 --warmup 2 --repeats 2
```

Read the report in a terminal:

```bash
nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum \
  flash_msa_docmask.nsys-rep
```

## Nsight Compute

Start with a light kernel survey:

```bash
ncu --set basic --target-processes all --print-summary per-kernel \
  --kernel-name 'regex:.*(msa|flash|reverse|merge).*' \
  python benchmarks/profile_training_case.py \
    --batch-size 1 --sequence-length 2048 --top-k 512 \
    --documents-per-row 4 --warmup 2 --repeats 1
```

After identifying the dominant kernel, collect the detailed report by replacing
`KERNEL_REGEX` with a narrow regular expression:

```bash
ncu --set full --target-processes all --force-overwrite \
  --kernel-name 'regex:KERNEL_REGEX' --launch-count 1 \
  --export flash_msa_kernel \
  python benchmarks/profile_training_case.py \
    --batch-size 1 --sequence-length 2048 --top-k 512 \
    --documents-per-row 4 --warmup 2 --repeats 1

ncu --import flash_msa_kernel.ncu-rep --page summary
```

The profiler executables may be installed outside `PATH` on clusters. Common
locations are under `/opt/nvidia/nsight-systems/*/bin` and
`/opt/nvidia/nsight-compute/*/ncu`. If neither executable is installed, the JSON
benchmark still provides stable CUDA-event timings, while profiler collection
must be performed on a node image containing the NVIDIA tools.

