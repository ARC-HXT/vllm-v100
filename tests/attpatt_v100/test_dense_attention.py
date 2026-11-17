# SPDX-License-Identifier: Apache-2.0
"""vllm/attpatt_v100/dense_attention.py (SDPA drop-in for HF models on V100) and dense_attention_hook.py
(ATTPATT_DENSE_ATTN=1 through a sitecustomize on PYTHONPATH).  GPU part: the kernel against an fp32 reference within
the fp16 rounding budget, and the drop-in against the original SDPA for the LLaDA / Dream call patterns, padding
masks and every fallback reason (V100 + triton-v100).  CPU part: the hook leaves the vllm package unimported and
patches only with the switch set."""
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

triton = pytest.importorskip("triton")

from vllm.attpatt_v100 import dense_attention as DA  # noqa: E402
from vllm.attpatt_v100 import dense_attention_hook as HOOK  # noqa: E402

ON_V100 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)
v100 = pytest.mark.skipif(not ON_V100, reason="needs a V100")
DEV = torch.device("cuda", 0)
ORIG = F.scaled_dot_product_attention

PROBE = r"""
import sys
import torch
import torch.nn.functional as F
print("PROBE", F.scaled_dot_product_attention.__module__, "vllm" in sys.modules)
if len(sys.argv) > 1:  # run one fp16 call on the GPU
    q = torch.randn(1, 8, 256, 128, device="cuda", dtype=torch.float16)
    F.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
"""


def _run_probe(tmp_path, switch, *args):
    site = tmp_path / "site"
    HOOK.write_sitecustomize(str(site))
    env = dict(os.environ, PYTHONPATH=str(site))
    env.pop("ATTPATT_DENSE_ATTN", None)
    if switch is not None:
        env["ATTPATT_DENSE_ATTN"] = switch
    r = subprocess.run([sys.executable, "-c", PROBE, *args], env=env, capture_output=True, text=True, timeout=600,
                       cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]
    line = next(x for x in r.stdout.splitlines() if x.startswith("PROBE"))
    return line.split()[1:], r.stderr


@pytest.mark.cpu
@pytest.mark.parametrize("switch", [None, "0", "true"])
def test_hook_off_unless_switch_is_1(tmp_path, switch):
    (module, vllm_loaded), _ = _run_probe(tmp_path, switch)
    assert module == ORIG.__module__ and vllm_loaded == "False"


@pytest.mark.cpu
def test_hook_patches_after_torch_import_without_importing_vllm(tmp_path):
    (module, vllm_loaded), err = _run_probe(tmp_path, "1")
    assert module == "vllm_attpatt_v100_dense_attention", err
    assert vllm_loaded == "False"
    assert "[attpatt dense attention] no SDPA calls" in err


@v100
def test_hook_serves_fp16_calls_on_v100(tmp_path):
    _, err = _run_probe(tmp_path, "1", "gpu")
    assert "[attpatt dense attention] {'triton': 1}" in err, err


def ref(q, k, v, key_ok=None, causal=False, scale=None):
    """fp32 reference in SDPA layout; key_ok [B, S] bool"""
    g = q.shape[1] // k.shape[1]
    kk, vv = k.float().repeat_interleave(g, 1), v.float().repeat_interleave(g, 1)
    s = torch.einsum("bhtd,bhsd->bhts", q.float(), kk) * (scale or q.shape[-1] ** -0.5)
    if key_ok is not None:
        s = s.masked_fill(~key_ok[:, None, None, :], float("-inf"))
    if causal:
        T, S = s.shape[-2:]
        s = s.masked_fill(torch.ones(T, S, dtype=torch.bool, device=s.device).triu(S - T + 1), float("-inf"))
    return torch.einsum("bhts,bhsd->bhtd", s.softmax(-1), vv)


def budget(r):
    # fp16 output rounding of the reference plus the same margin for the fp16 P in the PV product
    return 2 * (r.half().float() - r).abs().max().item() + 1e-4


def rand(B, H, T, D):
    return torch.randn(B, H, T, D, device=DEV, dtype=torch.float16)


@v100
@pytest.mark.parametrize("B,HQ,HK,T,D,causal", [
    (2, 32, 32, 1024, 128, False),   # LLaDA-8B heads
    (2, 28, 4, 1000, 128, False),    # Dream-7B heads, ragged length
    (1, 32, 8, 777, 128, True),      # Qwen3-4B heads, causal, ragged
    (1, 16, 4, 1024, 256, False),    # D256, output split in two halves
    (1, 16, 4, 900, 256, True),
    (3, 32, 32, 65, 128, False),     # shorter than one query block
])
def test_kernel_matches_fp32(B, HQ, HK, T, D, causal):
    torch.manual_seed(0)
    q, k, v = rand(B, HQ, T, D), rand(B, HK, T, D), rand(B, HK, T, D)
    r = ref(q, k, v, causal=causal)
    out = DA.dense_attention(q, k, v, causal=causal)
    assert out.shape == q.shape
    assert (out.float() - r).abs().max().item() <= budget(r)


@v100
def test_kernel_takes_transposed_views():
    # LLaDA: q.view(B, T, H, D).transpose(1, 2) - last dim contiguous, heads strided
    torch.manual_seed(1)
    B, T, H, D = 2, 512, 32, 128
    q = torch.randn(B, T, H, D, device=DEV, dtype=torch.float16).transpose(1, 2)
    k = torch.randn(B, T, H, D, device=DEV, dtype=torch.float16).transpose(1, 2)
    v = torch.randn(B, T, H, D, device=DEV, dtype=torch.float16).transpose(1, 2)
    r = ref(q, k, v)
    assert (DA.dense_attention(q, k, v).float() - r).abs().max().item() <= budget(r)


def _padding(B, S, lens):
    return torch.arange(S, device=DEV)[None, :] < torch.tensor(lens, device=DEV)[:, None]  # [B, S] bool


@v100
def test_dream_outer_product_bool_mask():
    # Dream generation_utils: logical_and(m[:, None, None, :], m[:, None, :, None]) -> [B, 1, N, N] bool
    torch.manual_seed(2)
    B, HQ, HK, N, D = 3, 28, 4, 640, 128
    m = _padding(B, N, [640, 500, 333])
    mask = torch.logical_and(m[:, None, None, :], m[:, None, :, None])
    q, k, v = rand(B, HQ, N, D), rand(B, HQ, N, D), rand(B, HQ, N, D)  # Dream repeats KV before SDPA
    DA.install()
    try:
        DA.stats.clear()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    finally:
        DA.uninstall()
    assert DA.stats == {"triton+mask": 1}
    r = ref(q, k, v, key_ok=m)
    rows = m  # valid query rows; padding rows are not compared (SDPA leaves them undefined)
    err = (out.float() - r).abs().amax(dim=(1, 3))[rows].max().item()
    assert err <= budget(r)


@v100
def test_llada_additive_float_mask():
    # LLaDA: zeros [1, 1, S, S] + (1 - m) * finfo.min, cast to fp16 with -inf -> fp16 min
    torch.manual_seed(3)
    B, H, S, D = 2, 32, 512, 128
    m = _padding(B, S, [512, 400])
    bias = torch.zeros(1, 1, S, S, device=DEV) + ((1.0 - m.float()) * torch.finfo(torch.float32).min)[:, None, None, :]
    bias = bias.half().nan_to_num(neginf=torch.finfo(torch.float16).min)
    q, k, v = rand(B, H, S, D), rand(B, H, S, D), rand(B, H, S, D)
    DA.install()
    try:
        DA.stats.clear()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    finally:
        DA.uninstall()
    assert DA.stats == {"triton+mask": 1}
    r = ref(q, k, v, key_ok=m)
    assert (out.float() - r).abs().max().item() <= budget(r)


@v100
def test_dropin_matches_original_sdpa_without_mask():
    torch.manual_seed(4)
    q, k, v = rand(2, 32, 1024, 128), rand(2, 32, 1024, 128), rand(2, 32, 1024, 128)
    base = ORIG(q, k, v)
    DA.install()
    try:
        DA.stats.clear()
        out = F.scaled_dot_product_attention(q, k, v)
    finally:
        DA.uninstall()
    assert DA.stats == {"triton": 1}
    r = ref(q, k, v)
    assert (out.float() - r).abs().max().item() <= budget(r)
    assert (base.float() - r).abs().max().item() <= budget(r)  # both within the same budget


@v100
@pytest.mark.parametrize("case,reason", [
    ("fp32", "fallback:dtype/device"),
    ("bf16", "fallback:dtype/device"),  # DDLM loads models in bf16 by default; V100 has no bf16 tensor cores
    ("d64", "fallback:geometry"),
    ("dropout", "fallback:dropout"),
    ("grad", "fallback:grad"),
    ("query_mask", "fallback:mask"),
    ("causal_rect", "fallback:causal shape"),
])
def test_fallbacks_go_to_original(case, reason):
    torch.manual_seed(5)
    q, k, v = rand(1, 8, 256, 128), rand(1, 8, 256, 128), rand(1, 8, 256, 128)
    kw = {}
    if case == "fp32":
        q, k, v = q.float(), k.float(), v.float()
    elif case == "bf16":
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    elif case == "d64":
        q, k, v = rand(1, 8, 256, 64), rand(1, 8, 256, 64), rand(1, 8, 256, 64)
    elif case == "dropout":
        kw = dict(dropout_p=0.1)
    elif case == "grad":
        q.requires_grad_(True)
    elif case == "query_mask":  # a genuine per-query mask (causal pattern) is not a key-padding mask
        kw = dict(attn_mask=torch.ones(256, 256, device=DEV, dtype=torch.bool).tril()[None, None])
    elif case == "causal_rect":
        q = rand(1, 8, 128, 128)
        kw = dict(is_causal=True)
    DA.install()
    try:
        DA.stats.clear()
        F.scaled_dot_product_attention(q, k, v, **kw)
    finally:
        DA.uninstall()
    assert DA.stats == {reason: 1}
