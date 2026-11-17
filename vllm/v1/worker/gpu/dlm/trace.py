# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step trajectory of full-sequence diffusion requests (opt-in per
request with ``extra_args["diffusion_trace"]``).

For every step and every canvas position j < G of a traced request:

* ``token_logp`` (fp16): log-probability of the canvas token, or of the
  policy's x0 where the position is masked;
* ``entropy`` (fp16): entropy of the full-vocabulary distribution;
* ``margin`` (fp16): top-1 minus top-2 probability;
* ``topk_ids`` (int32) / ``topk_logp`` (fp16): the top
  ``diffusion_trace_topk`` tokens;
* ``x0`` (int32): the policy's predicted token;
* ``canvas`` (int32): the canvas after the step;
* ``flags`` (uint8): bit 0 revealed, bit 1 remasked, bit 2 rewritten.

All distributions are the raw model distribution (fp32 logits, before any
policy-side temperature / top-p). The sampler packs the traced rows of one
step; ``AsyncOutput`` copies them on the copy stream and splits them per
request; the scheduler stacks the steps into ``[steps, G, ...]`` arrays that
reach ``CompletionOutput.dlm_trace`` when the request finishes.
"""

from typing import NamedTuple

import numpy as np
import torch

TRACE_KEY = "diffusion_trace"
TRACE_TOPK_KEY = "diffusion_trace_topk"
DEFAULT_TRACE_TOPK = 20
MAX_TRACE_TOPK = 1024

FLAG_REVEALED = 1
FLAG_REMASKED = 2
FLAG_REWRITTEN = 4

# Per-row fields and their dtypes ([R] or [R, K]).
TRACE_FIELDS: dict[str, torch.dtype] = {
    "token_logp": torch.float16,
    "entropy": torch.float16,
    "margin": torch.float16,
    "topk_ids": torch.int32,
    "topk_logp": torch.float16,
    "x0": torch.int32,
    "canvas": torch.int32,
    "flags": torch.uint8,
}


def trace_row_stats(
    logits: torch.Tensor, tokens: torch.Tensor, topk: int
) -> dict[str, torch.Tensor]:
    """Distribution statistics of fp32 ``logits`` [N, V] for ``tokens`` [N]."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    probs = logp.exp()
    # 0 * log 0 = 0: clamp -inf so masked-out tokens add nothing.
    entropy = -(probs * logp.clamp_min(torch.finfo(logp.dtype).min)).sum(dim=-1)
    del probs
    top_logp, top_ids = torch.topk(logp, max(topk, 2), dim=-1)
    top_p = top_logp[:, :2].exp()
    return {
        "token_logp": logp.gather(-1, tokens.long().unsqueeze(-1)).squeeze(-1),
        "entropy": entropy,
        "margin": top_p[:, 0] - top_p[:, 1],
        "topk_ids": top_ids[:, :topk],
        "topk_logp": top_logp[:, :topk],
    }


class DlmTraceTensors(NamedTuple):
    """The traced rows of one step, packed request after request."""

    fields: dict[str, torch.Tensor]  # name -> [R] or [R, K]
    req_ids: list[str]
    offsets: list[int]  # request i owns rows offsets[i] : offsets[i + 1]
    topks: list[int]  # requested top-k of request i (<= the packed width)

    def to_cpu_nonblocking(self) -> "DlmTraceTensors":
        fields = {
            name: t.to("cpu", non_blocking=True, copy=True)
            for name, t in self.fields.items()
        }
        return self._replace(fields=fields)

    def split(self) -> dict[str, dict[str, np.ndarray]]:
        """Per-request numpy views (call after the copy completed)."""
        arrays = {name: t.numpy() for name, t in self.fields.items()}
        out = {}
        for i, req_id in enumerate(self.req_ids):
            lo, hi = self.offsets[i], self.offsets[i + 1]
            k = self.topks[i]
            per_req = {name: a[lo:hi] for name, a in arrays.items()}
            per_req["topk_ids"] = per_req["topk_ids"][:, :k]
            per_req["topk_logp"] = per_req["topk_logp"][:, :k]
            out[req_id] = per_req
        return out


class DlmTraceAccumulator:
    """Engine-side ``[steps, G, ...]`` arrays of one request, in ordinary
    (pageable) memory, filled one step at a time."""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.arrays: dict[str, np.ndarray] | None = None

    def add(self, step: int, chunk: dict[str, np.ndarray]) -> None:
        if self.arrays is None:
            self.arrays = {
                name: np.zeros((self.steps, *a.shape), dtype=a.dtype)
                for name, a in chunk.items()
            }
        for name, a in chunk.items():
            self.arrays[name][step] = a

    def result(self) -> dict[str, np.ndarray] | None:
        return self.arrays
