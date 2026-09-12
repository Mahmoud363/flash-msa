# Headless GH200 Profiling

All profiling commands in this document run without a GUI. Run one ordinary
benchmark first so CUDA extensions and CuTe kernels are compiled before a trace.

```bash
module load Nsight-Systems/2025.5.1 Nsight-Compute/2025.3.1
source /e/scratch/jureap1/ahmed9/miniforge3/etc/profile.d/conda.sh
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

On this cluster the profilers are environment modules. Load the profiler modules
before reactivating Conda: module loading can change the default Python and clear
toolchain variables, while reactivation restores the `longvu` interpreter and
lets PyTorch discover its CUDA root.

Nsight Compute additionally requires permission to access GPU performance
counters. If it reports `ERR_NVGPUCTRPERM`, ask the cluster administrator to
enable performance counters for the job/node. Nsight Systems CUDA and NVTX
tracing remains usable without those counters.
