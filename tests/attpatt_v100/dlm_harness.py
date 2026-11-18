# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU harness for the full-sequence diffusion sampler: runs the real
FullSeqDiffusionSampler over a deterministic toy "model" (logits are a fixed
function of the whole sequence, so prompt and canvas interact like a
bidirectional model), next to the official sampling loops in dlm_official.py.
"""

from types import SimpleNamespace

import numpy as np
import torch

LLADA_VOCAB, LLADA_MASK = 126464, 126336
DREAM_VOCAB, DREAM_MASK = 152064, 151666

# --------------------------------------------------------------------------
# Seeded uniforms on the CPU. The engine's hash_uniform runs a Triton kernel
# (dlm/random.py); the CPU tests use this torch port of the same murmur3 mix
# (sample/gumbel.py murmur3_hash32): one value per element from (seed, pos,
# salt) only, never from the rest of the batch.
# --------------------------------------------------------------------------
_M32 = 0xFFFFFFFF


def _rotl32(v, s):
    return ((v << s) | (v >> (32 - s))) & _M32


def _mix32(h, k):
    k = (k * 0xCC9E2D51) & _M32
    k = _rotl32(k, 15)
    k = (k * 0x1B873593) & _M32
    h = h ^ k
    h = _rotl32(h, 13)
    return (h * 5 + 0xE6546B64) & _M32


def _fmix32(h):
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & _M32
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & _M32
    return h ^ (h >> 16)


def _hash32(seeds, pos, salt, domain=0):
    h = torch.full_like(seeds, domain)
    h = _mix32(h, seeds & _M32)
    h = _mix32(h, (seeds >> 32) & _M32)
    h = _mix32(h, pos & _M32)
    h = _mix32(h, torch.full_like(h, salt & _M32))
    return _fmix32(h ^ 16)


def cpu_hash_uniform(seeds, pos, salt, fp64=False):
    """murmur3_uniform32 / murmur3_uniform64 of sample/gumbel.py."""
    seeds = seeds.to(torch.int64)
    pos = pos.to(torch.int64)
    lo = _hash32(seeds, pos, salt)
    if not fp64:
        # (random32 + 0.5) * 2**-32, rounded once to fp32.
        return ((lo.to(torch.float64) + 0.5) * 2.0**-32).to(torch.float32)
    hi = _hash32(seeds, pos, salt, domain=0x9E3779B9)
    random53 = ((hi << 21) | (lo >> 11)).to(torch.float64)
    u = (random53 + 0.5) * 2.0**-53
    return torch.where(u == 1.0, u - 2.0**-53, u)


def _install_cpu_stubs():
    from vllm.utils import torch_utils
    from vllm.v1.worker.gpu.dlm import builtin_policies, policy

    # No pinned staging without a GPU: async_tensor_h2d to the CPU then
    # returns the host tensor itself.
    if not torch.cuda.is_available():
        torch_utils.PIN_MEMORY = False
    engine_hash = policy.hash_uniform

    def hash_uniform(seeds, pos, salt, fp64=False):
        if seeds.is_cuda:
            return engine_hash(seeds, pos, salt, fp64)
        return cpu_hash_uniform(seeds, pos, salt, fp64)

    policy.hash_uniform = hash_uniform
    builtin_policies.hash_uniform = hash_uniform


_install_cpu_stubs()


class ToyLM:
    """logits(x) = tanh(E[x] + P[pos] + mean over the sequence) @ O, fp32."""

    def __init__(self, vocab: int, seed: int = 0, dim: int = 8, boost=()):
        g = torch.Generator().manual_seed(seed)
        self.vocab = vocab
        self.emb = torch.randn(vocab, dim, generator=g)
        self.out = torch.randn(dim, vocab, generator=g) * 3
        self.pos = torch.randn(1024, dim, generator=g)
        self.bias = torch.zeros(vocab)
        for token_id, value in boost:
            self.bias[token_id] = value
        self.seen: list[torch.Tensor] = []
        self.device = torch.device("cpu")

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """The "final-norm hidden states" [L, dim] of the sequence."""
        h = self.emb[x] + self.pos[: x.shape[0]]
        return torch.tanh(h + h.mean(0, keepdim=True))

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.out + self.bias

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x))

    # Official LLaDA calls model(x, attention_mask=...).logits; official
    # Dream calls self(x, attention_mask, tok_idx).logits. Batch of one.
    def __call__(self, x, *args, **kwargs):
        self.seen.append(x[0].clone())
        return SimpleNamespace(logits=self.logits(x[0])[None])


class _Uva:
    def __init__(self, array: np.ndarray):
        self.np = array
        self.gpu = torch.from_numpy(array)


class _SamplingStates:
    def __init__(self, n: int):
        self.temperature = _Uva(np.zeros(n, dtype=np.float32))
        self.seeds = _Uva(np.zeros(n, dtype=np.int64))

    def add_request(self, slot, params):
        self.temperature.np[slot] = params.temperature
        self.seeds.np[slot] = params.seed if params.seed is not None else 1234 + slot

    def apply_staged_writes(self):
        pass


class _NoLogprobs:
    def add_request(self, slot, params):
        pass

    def apply_staged_writes(self):
        pass


def make_sampler(family: str, max_num_reqs: int = 4, max_model_len: int = 512):
    from vllm.v1.worker.gpu.dlm.sampler import FullSeqDiffusionSampler

    vocab, mask = (
        (LLADA_VOCAB, LLADA_MASK) if family == "llada" else (DREAM_VOCAB, DREAM_MASK)
    )
    req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(
            gpu=torch.zeros(max_num_reqs, max_model_len, dtype=torch.int32)
        ),
        prompt_len=SimpleNamespace(np=np.zeros(max_num_reqs, dtype=np.int32)),
    )
    base = SimpleNamespace(
        sampling_states=_SamplingStates(max_num_reqs),
        logprob_token_ids_state=_NoLogprobs(),
        req_states=req_states,
        logprobs_mode="raw_logprobs",
    )
    model_state = SimpleNamespace(
        model_config=SimpleNamespace(
            full_sequence_diffusion=family,
            hf_config=SimpleNamespace(mask_token_id=mask),
            get_vocab_size=lambda: vocab,
        ),
        model=SimpleNamespace(dlm_logits_shift=1 if family == "dream" else 0),
        device=torch.device("cpu"),
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        dlm_steps={},
    )
    return FullSeqDiffusionSampler(base, model_state)


def normalized_params(family: str, prompt: list[int], max_tokens: int, **kw):
    """SamplingParams after admission (vllm.utils.diffusion)."""
    from vllm.sampling_params import SamplingParams
    from vllm.utils.diffusion import normalize_full_sequence_diffusion_params

    extra = kw.pop("extra_args", {})
    generation_config = kw.pop("generation_config", {})
    params = SamplingParams(max_tokens=max_tokens, extra_args=extra, **kw)
    mask = LLADA_MASK if family == "llada" else DREAM_MASK
    normalize_full_sequence_diffusion_params(
        params,
        family=family,
        prompt_token_ids=prompt,
        mask_token_id=mask,
        generation_config=generation_config,
        max_model_len=8192,
        max_seq_len=4096,
        max_num_scheduled_tokens=8192,
    )
    return params


class Run:
    """Drive one engine-like run of several requests through the sampler."""

    def __init__(self, sampler, models: dict[str, ToyLM], hidden_states=False):
        self.sampler = sampler
        self.models = models
        self.hidden_states = hidden_states
        self.reqs: dict[str, tuple[int, int, int, int]] = {}  # slot, P, G, steps
        self.history: dict[str, list[torch.Tensor]] = {}

    def add(self, req_id: str, slot: int, prompt: list[int], params) -> None:
        mask = self.sampler.mask_id
        full = list(prompt) + [mask] * params.max_tokens
        ids = self.sampler.req_states.all_token_ids.gpu
        ids[slot, : len(full)] = torch.tensor(full, dtype=torch.int32)
        self.sampler.req_states.prompt_len.np[slot] = len(full)
        self.sampler.add_request(slot, params)
        steps = params.extra_args["diffusion_max_steps"]
        self.reqs[req_id] = (slot, len(prompt), params.max_tokens, steps)
        self.history[req_id] = []

    def sequence(self, req_id: str) -> torch.Tensor:
        slot, p, g, _ = self.reqs[req_id]
        return self.sampler.req_states.all_token_ids.gpu[slot, : p + g].long().clone()

    def step(self, req_steps: dict[str, int]):
        """One engine step over the given requests (req_id -> step index).
        Each request's logits stand in for hidden states and compute_logits
        is the identity; with hidden_states=True (one shared model) the
        sampler gets the model's hidden states and its head instead."""
        req_ids = list(req_steps)
        shared = {id(self.models[r]) for r in req_ids}
        model = None
        if self.hidden_states:
            assert len(shared) == 1, "hidden_states needs one shared model"
            model = self.models[req_ids[0]]
        hidden, qsl = [], [0]
        for req_id in req_ids:
            x = self.sequence(req_id)
            self.history[req_id].append(x)
            m = self.models[req_id]
            hidden.append(m.hidden(x) if model is not None else m.logits(x))
            qsl.append(qsl[-1] + x.shape[0])
        batch = SimpleNamespace(
            num_reqs=len(req_ids),
            req_ids=req_ids,
            idx_mapping_np=np.array([self.reqs[r][0] for r in req_ids]),
            query_start_loc_np=np.array(qsl, dtype=np.int32),
        )
        self.sampler.model_state.dlm_steps = dict(req_steps)
        compute_logits = model.head if model is not None else (lambda h: h.clone())
        return self.sampler.sample_hidden_states(
            torch.cat(hidden), batch, compute_logits
        )

    def run_all(self):
        """Run every request to completion, all in the same batch."""
        outputs = {}
        step = 0
        while True:
            live = {r: step for r, v in self.reqs.items() if step < v[3]}
            if not live:
                return outputs
            out = self.step(live)
            for i, req_id in enumerate(live):
                n = int(out.num_sampled[i])
                if n:
                    outputs[req_id] = out.sampled_token_ids[i, :n].long().clone()
            step += 1
