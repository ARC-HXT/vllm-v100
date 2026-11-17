# SPDX-License-Identifier: Apache-2.0
"""Dense (no KV cache) attention on V100 tensor cores (triton-v100 tl.dot -> HMMA) as a drop-in for
torch.nn.functional.scaled_dot_product_attention in Hugging Face models run outside vLLM, e.g. the diffusion LMs
LLaDA-8B and Dream-7B, whose every denoising step re-runs the whole sequence (bidirectional, no KV cache).
Not used by any vLLM code path.  Enabled per process by ATTPATT_DENSE_ATTN=1 through dense_attention_hook.py
(see README_V100.md), or directly: dense_attention.install().

Why: on V100, SDPA has no flash kernel; LLaDA's attention (a float mask) ran on the math path (fp32 sgemm) and Dream's
on cutlass memory-efficient attention (34 TFLOP/s at D128).  This FlashAttention-2 style kernel reaches 56-63 TFLOP/s
(D128) on LLaDA / Dream / Qwen3-4B shapes, errors equal to the fp16 rounding of an fp32 reference.  Whole model
(fp16, transformers 4.46.2): LLaDA-8B forward 1186.8 -> 962.2 ms (B 8, ~540 tokens) and 1715.0 -> 995.8 ms (B 2,
~2090 tokens); Dream-7B 1.01-1.05x; argmax agreement 99.73-99.98 % (AttPatt journal 2026-11-17).

Served: CUDA fp16 on sm70, 4-D [B, H, T, D] inputs (last dim contiguous), head dim in CONFIGS, dropout 0, no grad
needed, GQA (key heads dividing query heads), is_causal only with T == S, attn_mask None or a key-padding mask in any
of these forms: bool / float [B or 1, 1, 1 or T, S] where every query row either equals the per-key mask or is fully
masked (HF / Dream outer-product padding masks; LLaDA's additive bias).  Fully masked query rows (padding queries)
get attention over the valid keys instead of SDPA's undefined row - those rows are never read by valid positions.
Everything else (bf16 included: V100 has no bf16 tensor cores) goes to the original SDPA unchanged; `stats` counts
calls served and fallbacks by reason.
"""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
MASKED = -1.0e30  # finite: a fully masked row stays finite instead of NaN

# launch configs per head dim, swept on V100 (dense_attn_probe.py, 24 / 48 configs, 7 / 4 shapes each):
# D128 bm128 bn32 w4 s1 was fastest on all 7 shapes (LLaDA-8B 16x1024 / 8x2048 / 4x4096 bidirectional, Dream-7B 8x2048,
# Qwen3-4B 1x8192 both, 4x2048 causal); D256 bm128 bn64 w8 s2 with the output split in two D halves on all 4 shapes.
CONFIGS = {128: dict(bm=128, bn=32, w=4, st=1, dsplit=1), 256: dict(bm=128, bn=64, w=8, st=2, dsplit=2)}

stats = {}


@triton.jit
def _attn_fwd(Q, K, V, O, KB, qk_scale, M, N,
              sqb, sqm, sqh, skb, skn, skh, svb, svn, svh, sob, som, soh, skbb,
              HQ: tl.constexpr, GROUPS: tl.constexpr, HEAD_DIM: tl.constexpr, D_OUT: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr, EVEN_N: tl.constexpr,
              HAS_KB: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    pid_d = tl.program_id(2)
    b = bh // HQ
    h = bh % HQ
    kvh = h // GROUPS
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_o = pid_d * D_OUT + tl.arange(0, D_OUT)
    offs_n0 = tl.arange(0, BLOCK_N)
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :], mask=offs_m[:, None] < M, other=0.0)
    k_base = K + b * skb + kvh * skh
    v_base = V + b * svb + kvh * svh
    kb_base = KB + b * skbb
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D_OUT], tl.float32)
    shift = N - M
    if CAUSAL:
        full_hi = tl.maximum(pid_m * BLOCK_M + shift + 1, 0) // BLOCK_N * BLOCK_N
        hi = tl.minimum((pid_m + 1) * BLOCK_M + shift, N)
    else:
        if EVEN_N:
            full_hi = N
        else:
            full_hi = N // BLOCK_N * BLOCK_N
        hi = N
    for start_n in range(0, full_hi, BLOCK_N):  # key blocks every row sees: no mask
        offs_n = start_n + offs_n0
        k = tl.load(k_base + offs_n[None, :] * skn + offs_d[:, None])
        qk = tl.dot(q, k) * qk_scale
        if HAS_KB:
            qk += tl.load(kb_base + offs_n)[None, :] * 1.4426950408889634
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + offs_n[:, None] * svn + offs_o[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    for start_n in range(full_hi, hi, BLOCK_N):  # causal diagonal / ragged tail
        offs_n = start_n + offs_n0
        k = tl.load(k_base + offs_n[None, :] * skn + offs_d[:, None], mask=offs_n[None, :] < N, other=0.0)
        qk = tl.dot(q, k) * qk_scale
        if HAS_KB:
            qk += tl.load(kb_base + offs_n, mask=offs_n < N, other=0.0)[None, :] * 1.4426950408889634
        ok = offs_n[None, :] < N
        if CAUSAL:
            ok = ok & (offs_n[None, :] <= offs_m[:, None] + shift)
        qk = tl.where(ok, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + offs_n[:, None] * svn + offs_o[None, :], mask=offs_n[:, None] < N, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + b * sob + h * soh + offs_m[:, None] * som + offs_o[None, :], acc.to(tl.float16),
             mask=offs_m[:, None] < M)


def dense_attention(q, k, v, key_bias=None, causal=False, scale=None, cfg=None):
    """q [B, Hq, T, D], k / v [B, Hk, S, D] fp16 (SDPA layout, any strides with the last dim contiguous);
    key_bias: None or fp32 [B, S] added to every score of that key (MASKED for padding).  Returns [B, Hq, T, D] (a
    transposed view of a [B, T, Hq, D] buffer, so the HF `.transpose(1, 2).contiguous()` that follows is free)."""
    B, HQ, T, D = q.shape
    HK, S = k.shape[1], k.shape[2]
    cfg = cfg or CONFIGS[D]
    out = torch.empty(B, T, HQ, D, device=q.device, dtype=torch.float16)
    if key_bias is None:
        kb, skbb = q, 0  # unused pointer
    else:
        kb = key_bias.to(torch.float32).contiguous()
        skbb = kb.stride(0) if kb.shape[0] > 1 else 0
    scale = D ** -0.5 if scale is None else scale
    grid = (triton.cdiv(T, cfg["bm"]), B * HQ, cfg["dsplit"])
    kern = _attn_fwd[grid](
        q, k, v, out, kb, scale * LOG2E, T, S,
        q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1), out.stride(0), out.stride(1), out.stride(2), skbb,
        HQ=HQ, GROUPS=HQ // HK, HEAD_DIM=D, D_OUT=D // cfg["dsplit"], BLOCK_M=cfg["bm"], BLOCK_N=cfg["bn"],
        CAUSAL=bool(causal), EVEN_N=(S % cfg["bn"] == 0), HAS_KB=key_bias is not None,
        num_warps=cfg["w"], num_stages=cfg["st"])
    global _tensor_cores
    if _tensor_cores is None:  # first launch: stock Triton 3.x lowers tl.dot to FFMA on sm70 (no mma.sync)
        try:
            _tensor_cores = "mma.sync" in kern.asm.get("ptx", "")
        except Exception:  # noqa: BLE001 - cannot inspect: assume fine
            _tensor_cores = True
    return out.transpose(1, 2)


_tensor_cores = None  # None: not checked yet; False: this Triton has no sm70 tensor-core tl.dot -> sdpa() falls back


_mask_cache = {}


def key_bias_from_mask(mask, B, T, S):
    """fp32 [B or 1, S] key bias equivalent to `mask` for every query row that is not fully masked, or None if the
    mask is not a key-padding mask.  Cached per mask tensor (all layers of one forward share the same object)."""
    key = (mask.data_ptr(), mask._version, tuple(mask.shape), tuple(mask.stride()), mask.dtype)
    hit = _mask_cache.get(key)
    if hit is not None:
        return hit[0]
    kb = None
    if mask.dim() == 4 and mask.shape[1] == 1 and mask.shape[2] in (1, T) and mask.shape[3] == S \
            and mask.shape[0] in (1, B):
        m = mask[:, 0]  # [b, t, S]
        if mask.dtype == torch.bool:
            valid_row = m.any(-1, keepdim=True)  # [b, t, 1]
            k_ok = m.any(-2)  # [b, S]
            if bool(((m == k_ok[:, None, :]) | ~valid_row).all()):
                kb = torch.where(k_ok, 0.0, MASKED).to(torch.float32)
        elif mask.dtype in (torch.float16, torch.float32, torch.bfloat16):
            mf = m.float()
            lowest = torch.finfo(mask.dtype).min
            kmax = mf.amax(-2)  # [b, S]
            row_dead = (mf <= lowest * 0.5).all(-1, keepdim=True)
            if bool(((mf == kmax[:, None, :]) | row_dead).all()):
                kb = torch.where(kmax <= lowest * 0.5, MASKED, kmax).to(torch.float32)
    if len(_mask_cache) > 8:
        _mask_cache.clear()
    _mask_cache[key] = (kb, mask)  # keep the tensor alive so its data_ptr is not reused while cached
    return kb


_orig_sdpa = None


_caps = {}


def _capability(device):
    i = device.index if device.index is not None else torch.cuda.current_device()
    if i not in _caps:
        _caps[i] = torch.cuda.get_device_capability(i)
    return _caps[i]


def _note(reason):
    stats[reason] = stats.get(reason, 0) + 1


def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kw):
    """torch.nn.functional.scaled_dot_product_attention replacement; falls back to the original for anything the
    Triton kernel does not serve (see module docstring)."""
    def fallback(reason):
        _note("fallback:" + reason)
        return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                          scale=scale, enable_gqa=enable_gqa, **kw)
    if kw:
        return fallback("kwargs")
    if _tensor_cores is False:
        return fallback("triton without sm70 tensor cores (install triton-v100)")
    if not (query.is_cuda and query.dtype == torch.float16 and key.dtype == torch.float16
            and value.dtype == torch.float16):
        return fallback("dtype/device")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return fallback("rank")
    if _capability(query.device) != (7, 0):
        return fallback("not sm70")
    B, HQ, T, D = query.shape
    HK, S = key.shape[1], key.shape[2]
    if D not in CONFIGS or value.shape[-1] != D or key.shape[0] != B or HQ % HK or value.shape[1] != HK:
        return fallback("geometry")
    if dropout_p != 0.0:
        return fallback("dropout")
    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
        return fallback("grad")
    if is_causal and (T != S or attn_mask is not None):
        return fallback("causal shape")
    kb = None
    if attn_mask is not None:
        kb = key_bias_from_mask(attn_mask, B, T, S)
        if kb is None:
            return fallback("mask")
    q = query if query.stride(-1) == 1 else query.contiguous()
    k = key if key.stride(-1) == 1 else key.contiguous()
    v = value if value.stride(-1) == 1 else value.contiguous()
    _note("triton" + ("+mask" if kb is not None else ""))
    return dense_attention(q, k, v, key_bias=kb, causal=is_causal, scale=scale)


def install():
    """patch torch.nn.functional.scaled_dot_product_attention (module code calling F.scaled_dot_product_attention or
    torch.nn.functional.scaled_dot_product_attention at run time picks it up)"""
    global _orig_sdpa
    import torch.nn.functional as F
    if _orig_sdpa is None:
        _orig_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = sdpa
    torch.nn.functional.scaled_dot_product_attention = sdpa


def uninstall():
    import torch.nn.functional as F
    if _orig_sdpa is not None:
        F.scaled_dot_product_attention = _orig_sdpa
        torch.nn.functional.scaled_dot_product_attention = _orig_sdpa
