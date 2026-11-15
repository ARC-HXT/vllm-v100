# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the V100 kernel tests: a paged fp16 KV batch in vLLM's flash layout and an fp32 reference."""
import torch

HQ, HKV, D, PAGE, CAP = 32, 8, 128, 16, 16384  # Qwen3-4B heads; block table sized for max_model_len 16384 (32 parts)


def make_batch(lens, seed, cap=CAP, page=PAGE, pool=None,
               head_dim=D, num_q_heads=HQ, num_kv_heads=HKV):
    """Paged fp16 KV in vLLM's flash layout [blocks, page, kv_heads, 128]; every sequence gets physical blocks drawn
    from one random permutation (non-contiguous), block table columns past a sequence's length point at block 0.
    pool=N: only N physical blocks; each sequence draws its own distinct random blocks from them (sequences share
    blocks), so large batches fit in a few hundred MB on a shared card."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nblk = [(n + page - 1) // page for n in lens]
    total = sum(nblk) + 1 if pool is None else pool + 1
    kc = torch.randn(total, page, num_kv_heads, head_dim, generator=g).half().cuda()
    vc = torch.randn(total, page, num_kv_heads, head_dim, generator=g).half().cuda()
    perm = (torch.randperm(total - 1, generator=g) + 1).tolist()
    bt = torch.zeros(len(lens), cap // page, dtype=torch.int32)
    i = 0
    for b, nb in enumerate(nblk):
        if pool is not None:
            bt[b, :nb] = (torch.randperm(pool, generator=g)[:nb] + 1).int()
            continue
        bt[b, :nb] = torch.tensor(perm[i:i + nb], dtype=torch.int32)
        i += nb
    q = torch.randn(len(lens), num_q_heads, head_dim, generator=g).half().cuda()
    return q, kc, vc, bt.cuda(), torch.tensor(lens, dtype=torch.int32).cuda()


def reference(q, kc, vc, bt, lens, scale):
    """fp32 attention of one query token per sequence over its paged KV, 4 query heads per KV head"""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    hq, d = q.shape[1:]
    page, hkv = kc.shape[1:3]
    for b, n in enumerate(lens.tolist()):
        if n == 0:
            out[b].zero_()
            continue
        nb = (n + page - 1) // page
        blocks = bt[b, :nb].long()
        k = kc[blocks].reshape(-1, hkv, d)[:n].float()
        v = vc[blocks].reshape(-1, hkv, d)[:n].float()
        qq = q[b].float().view(hkv, hq // hkv, d)
        s = torch.einsum("hgd,thd->hgt", qq, k) * scale
        out[b] = torch.einsum("hgt,thd->hgd", s.softmax(-1), v).reshape(hq, d)
    return out


def rand_lens(n, lo, hi, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(lo, hi + 1, (n,), generator=g).tolist()
