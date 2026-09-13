# Flash-MSA
Flash-MSA is written in CuTeDSL for Hopper and Blackwell GPUs (eg H100, B200) on CUDA 13.

These kernels implement training for the MiniMax Sparse Attention paper:
https://arxiv.org/abs/2606.13392

Briefly, MSA is a style of sparse attention fitted to GQA that uses a small proxy attention layer to select blocks of keys to provide to the main attention layer. This offers a massive speedup to inference by slashing the memory-bandwidth bottleneck of loading the full KV cache from HBM.

The proxy heads are trained via a KL-divergence loss between the main attention layer's attention scores over the sparsely selected blocks. The proxy heads are assigned groups of main attention heads to select keys for & average scores over for KL-teaching.

This library also includes MSA warmup kernels, which run the main attention densely and train the proxy attention on the full sequence.

More information is included in the [blog post](https://nanduruganesh.github.io/flash-msa).

# Installation

flash-msa depends on FA3/4 from [flash-attn](https://github.com/dao-ailab/flash-attention). Try to configure your CUDA/Python/Torch versions to match one of the flash-attn wheels for a fast installation, but if you must build from source, set `MAX_JOBS=<max jobs>` to avoid `pip install flash-msa[attn]` bricking your CPU.

You will also need Python headers, e.g. `apt-get install python3.12-dev`, for whichever python version you are using.

```
uv pip install flash-msa
```

From source:

```
python setup.py install
```
or
```
uv pip install -e . --no-build-isolation
```
# Training with the SM90 kernels

The public training APIs consume projected tensors in `(batch, heads, sequence,
128)` layout. `Q_proxy` and `K_proxy` belong to the indexer/proxy branch, while
`Q`, `K`, and `V` belong to the main attention branch. The optimized Hopper path
uses FP8 E4M3 proxy selection with FP32 ranking, native SM90 sparse
forward/backward kernels, BF16 main-attention tensors, and FP32 softmax/LSE
accumulation by default. No environment variables are required to enable it.

## Warm-up stage

Use dense causal warm-up attention while training the proxy/indexer before its
block selections are reliable:

```python
from flash_msa import flash_msa_func_warmup

attn_out, kl_loss = flash_msa_func_warmup(
    Q_proxy,
    K_proxy,
    Q,
    K,
    V,
    top_k,
    head_dim**-0.5,
)

task_loss = loss_fn(attn_out, targets)
loss = task_loss + kl_weight * kl_loss
loss.backward()
```

Warm-up attention is dense rather than top-k sparse; `top_k` is accepted for API
compatibility but does not change its attention pattern. Include `kl_loss` in
the objective during this stage so backward computes the proxy Q/K training
signal.

## Normal sparse-training stage

After warm-up, switch to `flash_msa_func`. The FP8 proxy selector chooses
`top_k / 128` KV blocks and the SM90 kernels perform the sparse main-attention
forward and backward:

```python
from flash_msa import flash_msa_func

attn_out, kl_loss = flash_msa_func(
    Q_proxy,
    K_proxy,
    Q,
    K,
    V,
    top_k,                 # number of selected tokens; must be a multiple of 128
    head_dim**-0.5,
)

task_loss = loss_fn(attn_out, targets)
loss = task_loss + kl_weight * kl_loss
loss.backward()
```

`kl_loss` is a zero-valued autograd placeholder, not a materialized forward KL
value. Including it in the loss activates the on-the-fly proxy KL gradients in
backward; `kl_weight` scales those gradients. Consequently, logging the returned
placeholder does not report the actual KL-divergence value.

To train only the main sparse-attention branch and deactivate proxy KL backward,
leave the placeholder out of the loss:

```python
attn_out, _ = flash_msa_func(
    Q_proxy, K_proxy, Q, K, V, top_k, head_dim**-0.5
)
loss = loss_fn(attn_out, targets)  # no kl_loss term
loss.backward()
```

On the default SM90 path, omitting `kl_loss` also skips proxy LSE and proxy-gradient
work during backward. The FP8 proxy selector is still used in forward to choose
the sparse blocks; only its KL training signal is disabled.

## Packed documents / variable-length sequences

Both stages accept FA4-style cumulative document offsets:

```python
# Dense warm-up stage
attn_out, kl_loss = flash_msa_func_warmup(
    Q_proxy, K_proxy, Q, K, V, top_k, head_dim**-0.5,
    cu_seqlens=cu_seqlens,
)

# Normal sparse-training stage
attn_out, kl_loss = flash_msa_func(
    Q_proxy, K_proxy, Q, K, V, top_k, head_dim**-0.5,
    cu_seqlens=cu_seqlens,
)
```

`cu_seqlens` must be a one-dimensional CUDA `int32` tensor indexing the
flattened `B * S` token dimension, and it must contain every batch-row boundary.
The sparse API also retains the `[B, S]` `document_list` argument; do not pass it
together with `cu_seqlens`. Because Flash-MSA receives already projected Q/K
tensors, the caller must reset RoPE positions at every document boundary.

# Caveats

1. Varlen/document masking uses CUDA int32 cumulative document offsets or a
   `[B, S]` document-ID tensor.
2. Flash-MSA only supports headdims 128, block size 128.
3. Flash-MSA does not currently return fully materialized KL div. loss term in the fwd/bwd (see [blog](https://nanduruganesh.github.io/flash-msa) for explanation).
4. Proxy selection uses FP8 E4M3 on SM90; main attention remains BF16 by
   default. NVFP4 and MXFP4 training are not supported.
5. No support for attn temps / oai-style softmax bias.
6. Proxy Q is grouped by Main KV so Q_p >= KV heads for now.

These are not ridiculous to implement though so if there is demand or if someone makes a PR, I will update the repo to include these features.

# Testing

Test sparse MSA correctness against an eager implementation of MSA: `python tests/test_eager_match.py [args]`

Test warmup MSA correctness against an eager implementation of MSA: `python tests/test_warmup_eager_match.py [args]`

# Integration notes

An MSA training example is implemented in this [Megatron-LM fork](https://github.com/nanduruganesh/Megatron-LM). 

To monitor proxy training, log proxy gradient/update norms rather than the
zero-valued `kl_loss` placeholder. Another option is to materialize and
accumulate the KL divergence in a separate diagnostic pass every N steps, which
avoids adding that overhead to every forward call.

In general if you are going to train with this it is highly recommended to follow tips from [the paper](https://arxiv.org/abs/2606.13392), use MSA warmup before turning on MSA sparse training, and replicate any transformations to the main attention queries and keys (RoPE, QK norm, QK clip, etc) to the proxy queries and keys to improve proxy convergence.

# Inference
See MiniMax's [official repo](https://github.com/MiniMax-AI/MSA) for MSA inference kernels.
