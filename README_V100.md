# vllm-v100: vLLM 0.19.0 with V100 (sm70, fp16) kernels

This branch is upstream vLLM `v0.19.0` (commit `2a69949bd`) plus a few commits that speed up decoding on
NVIDIA V100 (compute capability 7.0) in fp16. Every change sits behind an environment variable that is **off by
default**; with no variable set the code path is the stock one (checked token by token against an unmodified
0.19.0 install).

The kernels are compiled at run time with `torch.utils.cpp_extension.load_inline` (needs `nvcc`, sm70 code only);
vLLM's own compiled libraries are the ones of the 0.19.0 release (install with precompiled binaries, see below).

## Switches

| Variable | Effect |
|---|---|
| `ATTPATT_V100_DECODE=1` | TRITON_ATTN backend: decode-only batches (query length 1) use a CUDA decode kernel (`vllm/attpatt_v100/decode_attention.py`) instead of the Triton kernel, which on sm70 is bound by the shared-memory pipe. |
| `ATTPATT_V100_SPLIT=1` (with `DECODE`) | The decode kernel is the split-KV version (`vllm/attpatt_v100/decode_attention_split.py`): partitions of 512 tokens plus a reduce kernel, so few long sequences still fill the GPU. Its fp32 scratch is reserved when the TRITON_ATTN metadata builder is created (before CUDA graph capture), outside `gpu_memory_utilization`: max_num_seqs x query heads x ceil(max_model_len / 512) x 130 x 4 bytes (about 136 MB for 256 sequences, 32 heads, 16k context). |
| `ATTPATT_V100_MIXED=1` (with `DECODE`) | Mixed prefill + decode batches: the metadata builder asks the runner to put decode requests first; their rows use the CUDA decode kernel and only the prefill rows go to the Triton kernel. |
| `ATTPATT_V100_GEMM=1` | Unquantized linear layers of the Qwen3-4B qkv (K=2560, N=6144) and down (K=9728, N=2560) shapes use a cuBLASLt algorithm chosen per (M, N, K) for every CUDA graph capture size (`vllm/attpatt_v100/gemm.py`), registered as a custom op; every other layer and shape runs `F.linear`. The switch is part of the torch.compile cache key. |

Only the value `1` turns a switch on. Conditions for the decode kernel: fp16 KV cache (`kv_cache_dtype=auto` with
`dtype=float16`), head size 128, 4 query heads per KV head, no ALiBi / sliding window / soft cap / sinks / output
quantisation / multimodal prefix ranges; anything else runs the stock kernel.

## GEMM table

The engine never selects algorithms; it only loads the table. Build it once per GPU model / cuBLASLt / torch
version (about 6 minutes on a V100, one GPU):

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m vllm.attpatt_v100.gemm build
```

The file goes to `$ATTPATT_V100_GEMM_CACHE` (default `$VLLM_CACHE_ROOT/attpatt_v100_gemm`, i.e.
`~/.cache/vllm/attpatt_v100_gemm`). Without it, `ATTPATT_V100_GEMM=1` logs one line and runs `F.linear` everywhere.
An entry is kept only if it was at least 3 % faster than `F.linear` in an interleaved re-measurement and its max-abs
and RMS errors against an fp32 reference were not worse (split-K only with fp32 partial sums); operands that are not
256-byte aligned, non-contiguous inputs, bias, other dtypes and sizes outside the table fall back to `F.linear`.
Measured on V100 at the profiled production batch (graph sizes 168 / 176 / 184): qkv about 15 % and down about 8 %
faster; at batch 256 neither shape gains and both fall back.

## Install (precompiled vLLM libraries, no CUDA build of vLLM)

```bash
git clone <this repository> vllm-v100 && cd vllm-v100
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=2a69949bdadf0e8942b7a1619b229cb475beef20 \
    uv pip install --editable .
```

With a local copy of the 0.19.0 wheel, use `VLLM_PRECOMPILED_WHEEL_LOCATION=<wheel>` instead of
`VLLM_PRECOMPILED_WHEEL_COMMIT`.

Offline machine that already has the vLLM 0.19.0 release wheel installed (every other `vllm/*.py` file of the wheel
is identical to the `v0.19.0` tag; checked on 1470 files): copy only the changed files into a copy of that
environment:

```bash
cp -a <venv with vllm 0.19.0> <new venv>
tools/attpatt_v100/overlay_install.sh <new venv>/lib/python3.X/site-packages
```

## Tests

```bash
CUDA_VISIBLE_DEVICES=<an sm70 GPU> python -m pytest --noconftest -q tests/attpatt_v100
```

`--noconftest` skips vLLM's global `tests/conftest.py`, which needs the full test requirements. With the overlay
install, run the tests from a directory that holds only `tests/` (the source tree's `vllm/` has no compiled
libraries and would shadow the installed package).

## Known limits

- sm70 / fp16 only; the kernels are not tuned or tested on other GPUs.
- GEMM: one cuBLASLt workspace per device, shared by all table GEMMs; this assumes they run on one stream (true for
  vLLM's dense decoder models). Only the two Qwen3-4B shapes are in the table.
- Split-KV scratch is allocated outside `gpu_memory_utilization` (see the table above).
