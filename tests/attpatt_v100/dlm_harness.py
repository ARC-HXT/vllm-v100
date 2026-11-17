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

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb[x] + self.pos[: x.shape[0]]
        h = torch.tanh(h + h.mean(0, keepdim=True))
        return h @ self.out + self.bias

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
        max_model_len=4096,
        max_position=4096,
        max_num_scheduled_tokens=8192,
    )
    return params


class Run:
    """Drive one engine-like run of several requests through the sampler."""

    def __init__(self, sampler, models: dict[str, ToyLM]):
        self.sampler = sampler
        self.models = models
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
        """One engine step over the given requests (req_id -> step index)."""
        req_ids = list(req_steps)
        hidden, qsl = [], [0]
        for req_id in req_ids:
            x = self.sequence(req_id)
            self.history[req_id].append(x)
            hidden.append(self.models[req_id].logits(x))
            qsl.append(qsl[-1] + x.shape[0])
        batch = SimpleNamespace(
            num_reqs=len(req_ids),
            req_ids=req_ids,
            idx_mapping_np=np.array([self.reqs[r][0] for r in req_ids]),
            query_start_loc_np=np.array(qsl, dtype=np.int32),
        )
        self.sampler.model_state.dlm_steps = dict(req_steps)
        return self.sampler.sample_hidden_states(
            torch.cat(hidden), batch, lambda h: h.clone()
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
