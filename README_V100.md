# vllm-v100: vLLM 0.33.0 for V100 (sm70, fp16)

This branch is upstream vLLM `v0.33.0` (commit `4ba787ec99`) plus:

- two build changes, so that vLLM still compiles for NVIDIA V100 (compute capability 7.0): `CMakeLists.txt` keeps
  `7.0` in the CUDA 12.8 / 12.9 architecture list (upstream dropped it after 0.19; nvcc 12.8 / 12.9 still target
  sm_70), and requires GCC 13 only when an sm90+ architecture is built (DeepGEMM / DeepSelect; an sm70-only build
  compiles with GCC 11);
- the V100 decode switches of the 0.19 branch (`v0.19.0-v100.*` tags), ported to the v0.33 TRITON_ATTN backend.
  Every switch is **off by default**; with no variable set the code path is the stock one.

Upstream wheels on PyPI do not run on V100 (built with CUDA >= 12.8 for sm75+); this branch is built from source,
see Build below.  The decode kernels are compiled at run time with `torch.utils.cpp_extension.load_inline` (needs
`nvcc`, sm70 code only).

## Switches

| Variable | Effect |
|---|---|
| `ATTPATT_V100_DECODE=1` | TRITON_ATTN backend: decode-only batches (query length 1) use a CUDA decode kernel (`vllm/attpatt_v100/decode_attention.py`) instead of the Triton kernel, which on sm70 is bound by the shared-memory pipe. |
| `ATTPATT_V100_SPLIT=1` (with `DECODE`) | The decode kernel is the split-KV version (`vllm/attpatt_v100/decode_attention_split.py`): partitions of 512 tokens plus a reduce kernel, so few long sequences still fill the GPU. Its fp32 scratch is reserved when the TRITON_ATTN metadata builder is created (before CUDA graph capture), outside `gpu_memory_utilization`: max_num_seqs x query heads x ceil(max_model_len / 512) x 130 x 4 bytes (about 136 MB for 256 sequences, 32 heads, 16k context). |
| `ATTPATT_V100_QK_GROUP=8` (default inside `SPLIT`) | Eight threads cooperate on each K row, reuse it for four query heads, and reduce with a fixed subgroup mask. fp16 KV / fp32 accumulation, same 512-token partitions and CUDA graph buffers. `=1` restores the original one-thread-per-token QK. JIT module names include the group so compiled caches cannot collide. |
| `ATTPATT_V100_MIXED=1` (with `DECODE`) | Mixed prefill + decode batches: the metadata builder asks the runner to put decode requests first; their rows use the CUDA decode kernel and only the prefill rows go to the Triton kernel. |
| `ATTPATT_V100_PREFILL=2` (with `DECODE` + `MIXED`) | Prefill / recompute rows of mixed and prefill batches run on tensor cores through torch's cutlass memory-efficient attention (`torch.ops.aten._efficient_attention_forward`, sm70 kernels in libtorch; `vllm/attpatt_v100/prefill_attention.py`): per request, its KV pages are gathered through the block table (4 KiB per token) and GQA is a stride-0 broadcast, causal mask aligned bottom-right. Host lengths come from `seq_lens_cpu_upper_bound`, so the path is off with speculative decoding. V100, Qwen3-4B, 3 prompts of 30k tokens: 291 -> 2713 prompt tok/s; 2048-token chunk over 32k history 425.6 -> 34.6 ms (32.8 TF/s). `=1` uses the 1Cat-vLLM flash-attention-v100 paged prefill kernel instead (JIT from sources at `$ATTPATT_FA_V100_SRC`, not shipped; 1177 prompt tok/s). |
| `ATTPATT_V100_GEMM=1` | Unquantized linear layers of the Qwen3-4B qkv (K=2560, N=6144) and down (K=9728, N=2560) shapes use a cuBLASLt algorithm chosen per (M, N, K) for every CUDA graph capture size (`vllm/attpatt_v100/gemm.py`), registered as a custom op; every other layer and shape runs `F.linear`. The switch is part of the torch.compile cache key. |
| `ATTPATT_V100_GDN_PREFILL=1` | Independent opt-in for GDN prefill on sm70: fp16 Q/K/V, 128-dimensional key/value heads and fp32 initial states use the existing recurrent update kernel with compact per-sequence states and a single final-state write. No per-token state pool or slot-index table is allocated. Initial states remain unchanged; zero-length rows, external output buffers and CUDA graph replay are supported. Other devices, dtypes and head dimensions retain the selected original backend. The switch joins the compile cache key and must be set before engine creation. |

The on/off switches use `1`; `PREFILL` also accepts `2`, and `QK_GROUP` is a selector accepting `1` or `8`.
Set variables before worker startup. Use a release-specific `TORCH_EXTENSIONS_DIR` and precompile both QK groups
before dispatch; check for abandoned build locks without removing locks held by active compilers.
Conditions for the decode kernel: fp16 KV cache (`kv_cache_dtype=auto` with
`dtype=float16`), head size 128, 4 query heads per KV head, no ALiBi / sliding window / soft cap / sinks / output
quantisation / multimodal prefix ranges / R-SWA; anything else runs the stock kernel.  v0.33 stores K and V of a
token in one cache row (`[blocks, kv_heads, page, 2 * 128]`); the backend passes strided `[blocks, page, kv_heads,
128]` views and the kernels read through the strides.


## Long-context release v0.33.0-v100.3

On one V100-SXM2-32GB, Qwen3-4B, four synthetic 32k contexts, fp16, original sampling and a 32768 generation cap: two 45-second decode windows per variant, with reversed order, measured 80.03 / 80.04 generated token/s original and 97.55 / 97.53 cooperative (about +21.9%). The first measured busy mean power was 233.3 -> 266.6 W. These are synthetic performance probes, not task accuracy results. Both variants captured graph sizes [1,2,4]; running production jobs are not modified. QK changes floating-point reduction order; bitwise parity with the original is not promised. Keep `ATTPATT_V100_QK_GROUP=1` as the rollback path.

## Recurrent GDN prefill

The GDN switch applies to the Triton/FLA `ChunkGatedDeltaRule` prefill path, independently of the ordinary decode-attention switches. v0.33 already peels non-spec decode rows from mixed batches; this switch changes the remaining prefill computation. Pure decode, speculative updates and checkpoint-producing backends retain their original implementations. Unset the switch or use `=0` to restore stock GDN prefill. Qwen3.5-9B and Qwen3.6-35B-A3B have matching GDN dimensions (16 key heads, 32 value heads, K/V 128), but validating one does not establish whole-model MoE throughput or accuracy for the other.

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

## Build and artifacts (GitHub Actions)

`.github/workflows/attpatt-v100-wheel.yml` builds the complete CUDA wheel from
the checked-out source on a GitHub-hosted runner: Python3.13, CUDA12.8.1,
Torch2.14.0+cu126, and SM70 only. It does not reuse or repack an older vLLM
wheel. Pushes to `main`, `v100-*` and `codex/v100-*`, or a manual dispatch,
produce a wheel, SHA256 checksums, source/run provenance, native SM70
verification and CPU test results as one Actions artifact. Native compilation
and the separate CPU workflow can run concurrently. V100 kernel/model tests
require a real GPU and are a separate acceptance step; a hosted CPU runner
does not establish GPU correctness or performance.

Build speed: `MAX_JOBS` is set per run from the runner (one nvcc job per 4 GiB
of RAM, capped by cores; `dist/runner.txt`). The ccache directory and the Rust
build tree are kept in the Actions cache; ccache only replays compiler calls
with byte-identical inputs, so the wheel is still a full source build. A
failed or timed-out build saves its partial cache under a run-specific key.
Each artifact carries `ninja-report.txt` (slowest targets), `ccache-stats.txt`,
`vmstat.txt` (memory during the compile) and `native-libs.sha256`. Artifacts
and logs expire after one day (repository setting).

A Volta / Turing-only `TORCH_CUDA_ARCH_LIST` (every entry below 8.0,
`tools/cuda_arch_targets.py`) does not build the FA2 (sm80+) / FA3 (sm90a)
kernels, which have no usable arch there; the vllm-flash-attn Python files
(e.g. `layers/rotary.py`, used by `RotaryEmbedding.forward_cuda`) are still
installed. Releases up to v100.6 shipped both libraries (about 84 MB) compiled
for nvcc's default arch; vLLM reports them as unavailable when absent.

Build environments live only on disposable Actions runners. Do not accumulate
versioned build virtual environments on inference servers. Install a verified
CI artifact for runtime use; GPU acceptance must use that exact artifact.

## Reproduce the Actions build (sm70 only)

Tested on Ubuntu 22.04, GCC 11.4, CUDA 12.8 toolkit, driver 580, Python 3.13:

```bash
uv venv <venv> --python 3.13
uv pip install -p <venv>/bin/python torch==2.14.0+cu126 --find-links <pytorch cu126 wheel index>  # keeps sm_70
uv pip install -p <venv>/bin/python -r requirements/build/cuda.txt  # torch==2.14.0 is satisfied by +cu126
export CUDA_HOME=/usr/local/cuda-12.8 PATH=<venv>/bin:/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=32 VLLM_VERSION_OVERRIDE=0.33.0+v100.1
python -m build --wheel --no-isolation --skip-dependency-check .
uv pip install -p <venv>/bin/python dist/vllm-0.33.0+v100.1-*.whl --find-links <pytorch cu126 wheel index>
```

Without access to github.com, point CMake at local source trees of its fetched dependencies (revisions are the ones
in `CMakeLists.txt` and `cmake/external_projects/*.cmake`, plus their git submodules): `VLLM_CUTLASS_SRC_DIR`,
`VLLM_FLASH_ATTN_SRC_DIR`, `DEEPGEMM_SRC_DIR`, `DEEPSELECT_SRC_DIR`, `FLASH_KDA_SRC_DIR`, `FLASH_MLA_SRC_DIR`,
`FMHA_SM100_SRC_DIR`, `QUTLASS_SRC_DIR`, `TML_FA4_SRC_DIR`, `TRITON_KERNELS_SRC_DIR`.  For sm70 nothing of them is
compiled, but their files are installed into the wheel.  Triton 3.8 (pulled in by torch 2.14) ships ptxas 12.9,
which still targets sm_70.

## Tests

```bash
CUDA_VISIBLE_DEVICES=<an sm70 GPU> python -m pytest --noconftest -q tests/attpatt_v100
```

`--noconftest` skips vLLM's global `tests/conftest.py`, which needs the full test requirements. Run the tests from
a directory that holds only `tests/` when the source tree has not been built in place (its `vllm/` would shadow the
installed package).  GitHub Actions (`.github/workflows/attpatt-v100-cpu.yml`) runs the CPU part on every push to a
`v100-*` branch: the switch-off paths of `test_switches.py`; switch-on and kernel tests need an sm70 GPU and are
skipped there.

## Known limits

- sm70 / fp16 only; the kernels are not tuned or tested on other GPUs.
- GEMM: one cuBLASLt workspace per device, shared by all table GEMMs; this assumes they run on one stream (true for
  vLLM's dense decoder models). Only the two Qwen3-4B shapes are in the table.
- Split-KV scratch is allocated outside `gpu_memory_utilization` (see the table above).
