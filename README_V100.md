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
| `ATTPATT_V100_MIXED=1` (with `DECODE`) | Mixed prefill + decode batches: the metadata builder asks the runner to put decode requests first; their rows use the CUDA decode kernel and only the prefill rows go to the Triton kernel. |

Only the value `1` turns a switch on. Conditions for the decode kernel: fp16 KV cache (`kv_cache_dtype=auto` with
`dtype=float16`), head size 128, 4 query heads per KV head, no ALiBi / sliding window / soft cap / sinks / output
quantisation / multimodal prefix ranges; anything else runs the stock kernel.

## Install (precompiled vLLM libraries, no CUDA build of vLLM)

```bash
git clone <this repository> vllm-v100 && cd vllm-v100
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=2a69949bdadf0e8942b7a1619b229cb475beef20 \
    uv pip install --editable .
```

Offline machines: point `VLLM_PRECOMPILED_WHEEL_LOCATION` at a local copy of the vLLM 0.19.0 wheel (or any zip
holding its `vllm/*.abi3.so`, `vllm/vllm_flash_attn/` and `vllm/third_party/` files) instead of
`VLLM_PRECOMPILED_WHEEL_COMMIT`.

## Tests

```bash
CUDA_VISIBLE_DEVICES=<an sm70 GPU> python -m pytest --noconftest -q tests/attpatt_v100
```

`--noconftest` skips vLLM's global `tests/conftest.py`, which needs the full test requirements.

## Known limits

- sm70 / fp16 only; the kernels are not tuned or tested on other GPUs.
