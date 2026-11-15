# SPDX-License-Identifier: Apache-2.0
"""Which kernel runs each part of a TRITON_ATTN batch on V100 (vllm/v1/attention/backends/triton_attn.py).  Pure
functions of the layer geometry and the batch, no CUDA, so the rules are unit-tested on the CPU job.

Two independent decisions (plan 2026-11-16 round 1 item 4; before, the prefill branch sat inside the decode condition
and so required G4, D128 and no window):
- decode_ok: the CUDA split / one-block decode kernel can serve the one-token rows (G = 4, a compiled head dim, causal
  sliding window only with the split kernel, no softcap / alibi / sinks);
- prefill_ok: torch's cutlass memory-efficient attention (vllm/attpatt_v100/prefill_attention.py) can serve the prefill
  rows: causal, head dim in PREFILL_CUTLASS_HEAD_DIMS, full or causal sliding window, no softcap / alibi / sinks (it has
  no softcap: a softcap layer would be computed wrongly, silently).  The 1Cat flash-attention-v100 path keeps its old
  scope (decode_ok, D128, no window).
Layers with chunked attention (chunk_lookback != -1, e.g. rnj1: the stock kernel's chunk mask, not a sliding window)
are served by neither kernel.  Per-call conditions (fp16 KV, no fused output quantization, no multimodal prefix
ranges, no R-SWA) apply to both and are checked by the caller with call_ok.
- prefill_kernel: cutlass or Triton for the prefill rows of one batch, from the head dim, the window and the batch's
  longest query chunk (constants below; a batch is decided as a whole by its longest chunk)."""

# Rule constants.  Source: tools/v100_triton/prefill_attn_options.py <out> round1 (AttPatt repository), V100 GPU 6,
# 2026-11-16, output /home/arc/build/triton/impl-r1/work/logs/prefill_round1.json: one request of M query tokens at the
# end of L tokens (M = L = 8192, or M = 2048 ... 16 with L = 16384), ms per call, Triton / cutlass (cutlass includes
# the page gather); both errors equal the fp16 rounding of the fp32 reference in every case.
# - Short windows: D256 w512 Triton faster at every M (M = 8192: 2.27 / 2.61; M = 16: 0.21 / 0.37), D256 w1024 too
#   (3.92 / 4.15 ... 0.27 / 0.44); D128 w1024 (gemma-3-27b TP4, 8 q / 4 kv) cutlass from M = 2 x window on
#   (M = 8192: 2.25 / 1.59; M = 2048: 0.72 / 0.70; M = 1024: 0.47 / 0.52; M = 16: 0.20 / 0.37).  Longer windows
#   follow the full-attention rule (Mistral D128 w4096: cutlass from M = 128 on, 0.64 / 0.59; D256 w4096 about even
#   from M = 512 on).
# - Small chunks over a long history: cutlass pays the gather of the whole history, about 0.37 ms at D128.  Full
#   attention D128 (32 q / 8 kv and G7 28 q / 4 kv): M = 128 cutlass faster (2.09 / 1.56, 2.16 / 1.48), M = 64 Triton
#   (1.14 / 1.36, 1.13 / 1.34); D256 (16 q / 4 kv): M = 256 cutlass (4.24 / 3.40), M = 128 Triton (2.17 / 2.27);
#   D512 (8 q / 2 kv): cutlass at every M down to 16 (4.12 / 3.94).
# - Small head dims (same tool, argument small_d, V100 GPU 0, 2026-11-16, logs/prefill_small_d.json; cutlass error <=
#   Triton's everywhere): D96 (Phi-3-mini, 32 q / 32 kv) and D80 (Phi-2, 32 / 32) M = 128 cutlass (2.20 / 1.85,
#   2.16 / 1.79), M = 64 Triton (1.21 / 1.75, 1.23 / 1.62); D64 (Qwen2-0.5B, 14 q / 2 kv) M = 512 cutlass
#   (2.32 / 1.70), M = 256 Triton (1.54 / 1.62); M = L = 8192: 1.96x / 1.63x / 1.61x faster on cutlass.
PREFILL_SHORT_WINDOW = 1024  # windows up to this many tokens: Triton for D >= 256, and for M < 2 x window

# head dims with numeric tests (test_prefill_attention.py GEOMETRIES) and measured thresholds; other multiples of 8 stay
# on Triton until measured
PREFILL_CUTLASS_HEAD_DIMS = (64, 80, 96, 128, 256, 512)
_MIN_CUTLASS_QUERY = {64: 512, 80: 128, 96: 128, 128: 128, 256: 256, 512: 1}


def _min_cutlass_query(head_dim):
    """chunks with fewer query tokens than this run on Triton"""
    return _MIN_CUTLASS_QUERY[head_dim]


def window_tokens(sliding_window):
    """vLLM (left, right) sliding window -> tokens attended including the query itself (0 = full attention);
    None when the window is not a causal sliding window (left, 0)"""
    left, right = sliding_window
    if (left, right) == (-1, -1):
        return 0
    if right != 0 or left < 0:
        return None
    return left + 1


def decode_ok(*, num_q_heads, num_kv_heads, head_dim, sliding_window, softcap, alibi, sinks, decode_dims,
              split_kernel, chunk_lookback=-1):
    w = window_tokens(sliding_window)
    return (w is not None and (w == 0 or split_kernel) and not softcap and not alibi and not sinks
            and chunk_lookback == -1 and head_dim in decode_dims and num_q_heads == 4 * num_kv_heads)


def prefill_ok(*, kind, head_dim, sliding_window, softcap, alibi, sinks, decode, chunk_lookback=-1):
    """kind: 'cutlass' (ATTPATT_V100_PREFILL=2) or 'flash' (=1); decode: decode_ok of the layer"""
    w = window_tokens(sliding_window)
    if chunk_lookback != -1:
        return False
    if kind == "flash":
        return decode and head_dim == 128 and w == 0
    return (w is not None and head_dim in PREFILL_CUTLASS_HEAD_DIMS and not softcap and not alibi
            and not sinks)


def call_ok(*, kv_fp16, output_scale, mm_prefix, rswa, chunk_lookback=-1):
    return (kv_fp16 and output_scale is None and mm_prefix is None and rswa is None
            and chunk_lookback == -1)


def prefill_kernel(*, kind, head_dim, sliding_window, max_query_len):
    """'cutlass' or 'triton' for the prefill rows of a batch whose layer is prefill_ok; the window / chunk rules were
    measured for the cutlass path and apply to it only (kind 'flash' keeps its own scope, prefill_ok)"""
    if kind != "cutlass":
        return kind
    w = window_tokens(sliding_window)
    if w and w <= PREFILL_SHORT_WINDOW and (head_dim >= 256 or max_query_len < 2 * w):
        return "triton"
    if max_query_len < _min_cutlass_query(head_dim):
        return "triton"
    return "cutlass"
