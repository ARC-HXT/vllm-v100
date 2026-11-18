# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only unmasking policy for the trajectory replay (GPU tests): the
canvas follows a reference trajectory (teacher forcing), while a built-in
policy decides on the same input what it would have done. That decision is
written to files on the worker (host syncs are fine in a test plugin).

Loaded by FQCN: ``diffusion_policy="tests.attpatt_v100.dlm_replay:ReplayPolicy"``
with ``diffusion_policy_args``:

* ``ref``: [steps][G] canvas after each step of the reference run;
* ``inner`` / ``inner_args``: the built-in policy to evaluate and its args;
* ``out``: file prefix; step k writes ``{out}_{k:04d}.npz`` with the inner
  policy's decision ``revealed`` / ``remasked`` / ``rewritten`` [G], its
  ``x0`` [G], ``confidence`` [G] and every other score output as
  ``extra_<name>`` [G] (fp32).

The canvas follows the reference whatever the transition (reveal, remask or
rewrite). ``ReplayFeaturesPolicy`` is the same for inner policies that read
the hidden states.
"""

from dataclasses import fields, replace

import numpy as np
import torch

from vllm.v1.worker.gpu.dlm.policy import (
    DlmStepDecision,
    UnmaskPolicy,
    get_unmask_policy_cls,
)

_OWN_ARGS = ("ref", "inner", "inner_args", "out")


class ReplayPolicy(UnmaskPolicy):
    name = "test_replay"
    bytes_per_logit = 40  # replaced by the inner policy's in init_request

    def __init__(self) -> None:
        self.inners: dict[str, UnmaskPolicy] = {}

    def _inner(self, name: str) -> UnmaskPolicy:
        if name not in self.inners:
            self.inners[name] = get_unmask_policy_cls(name)()
        return self.inners[name]

    def validate(self, args, config) -> None:
        ref = args.get("ref")
        if not (
            isinstance(ref, list)
            and len(ref) == config.steps
            and all(len(r) == config.gen_length for r in ref)
        ):
            raise ValueError("ref must be [steps][G]")
        inner = self._inner(args["inner"])
        inner_args = dict(args.get("inner_args") or {})
        inner.validate(inner_args, config)
        # The inner policy reads its own args from config.policy_args.
        args.update(inner_args)

    def alloc_state(self, max_num_reqs, max_canvas, device):
        return {"ref": torch.zeros(max_num_reqs, 1, max_canvas, dtype=torch.int32,
                                   device=device)}  # fmt: skip

    def init_request(self, slot, config, state) -> None:
        inner = self._inner(config.policy_args["inner"])
        for key, value in inner.alloc_state(
            state["ref"].shape[0], state["ref"].shape[2], state["ref"].device
        ).items():
            state.setdefault(key, value)
        inner.init_request(slot, config, state)
        # Row chunks as large as the inner policy's (the lm_head GEMM sees the
        # same row counts as in a run of the inner policy alone).
        self.bytes_per_logit = inner.bytes_per_logit
        ref = torch.tensor(config.policy_args["ref"], dtype=torch.int32)
        buf = state["ref"]
        if buf.shape[1] < config.steps:
            grown = buf.new_zeros(buf.shape[0], config.steps, buf.shape[2])
            grown[:, : buf.shape[1]] = buf
            state["ref"] = buf = grown
        buf[slot, : config.steps, : config.gen_length] = ref.to(buf.device)

    def _groups(self, configs) -> dict[str, list[int]]:
        """Requests (indices into configs) by inner policy."""
        groups: dict[str, list[int]] = {}
        for i, config in enumerate(configs):
            groups.setdefault(config.policy_args["inner"], []).append(i)
        return groups

    def score(self, x):
        groups = self._groups(x.configs)
        if len(groups) == 1:
            out = self._inner(next(iter(groups))).score(x)
        else:  # each inner policy scores its own requests' rows
            out = {}
            for name, reqs in groups.items():
                rows_np = np.flatnonzero(np.isin(x.row_req_np, reqs))
                if len(rows_np) == 0:
                    continue
                rows = torch.from_numpy(rows_np).to(x.logits.device)
                sub = replace(
                    x,
                    logits=x.logits[rows],
                    hidden=None if x.hidden is None else x.hidden[rows],
                    row_req=x.row_req[rows],
                    row_pos=x.row_pos[rows],
                    row_slot=x.row_slot[rows],
                    row_token=x.row_token[rows],
                    row_req_np=x.row_req_np[rows_np],
                )
                for key, value in self._inner(name).score(sub).items():
                    if key not in out:
                        out[key] = value.new_zeros((len(x.row_req), *value.shape[1:]))
                    out[key].index_copy_(0, rows, value)
        forced = x.state["ref"][x.row_slot, x.step[x.row_req], x.row_pos]
        out["our_x0"] = out.pop("x0")
        out["x0"] = forced.long()
        return out

    def _inner_select(self, x) -> DlmStepDecision:
        groups = self._groups(x.configs)
        if len(groups) == 1:
            return self._inner(next(iter(groups))).select(x)
        num_reqs = len(x.configs)
        merged = {
            "revealed": torch.zeros_like(x.is_masked),
            "remasked": torch.zeros_like(x.is_masked),
            "rewritten": torch.zeros_like(x.is_masked),
        }
        for name, reqs in groups.items():
            idx = torch.tensor(reqs, device=x.canvas.device)

            def take(v, idx=idx):
                if (
                    isinstance(v, torch.Tensor)
                    and v.dim() > 0
                    and v.shape[0] == num_reqs
                ):
                    return v[idx]
                return v

            sub = replace(
                x,
                **{f.name: take(getattr(x, f.name)) for f in fields(x)
                   if f.name not in ("extras", "configs", "steps_np", "seeds")},
                extras={k: v[idx] for k, v in x.extras.items()},
                configs=[x.configs[i] for i in reqs],
                steps_np=x.steps_np[reqs],
            )  # fmt: skip
            decision = self._inner(name).select(sub)
            for key in merged:
                value = getattr(decision, key)
                if value is not None:
                    merged[key][idx] = value
        return DlmStepDecision(**merged)

    def select(self, x):
        extras = {k: v for k, v in x.extras.items() if k != "our_x0"}
        ours = self._inner_select(replace(x, x0=x.extras["our_x0"], extras=extras))
        width = x.canvas.shape[1]
        nxt = x.state["ref"][x.slots, x.step, :width].long()
        # The reference transition, whatever the inner policy does: reveal
        # (mask -> token), remask (token -> mask), rewrite (token -> token).
        written = ~x.is_masked & x.valid
        revealed = x.is_masked & x.valid & (nxt != x.mask_id)
        remasked = written & (nxt == x.mask_id)
        rewritten = written & (nxt != x.mask_id) & (nxt != x.canvas)
        none = torch.zeros_like(revealed)
        for i, config in enumerate(x.configs):
            g = config.gen_length
            arrays = {
                "revealed": ours.revealed[i, :g],
                "remasked": (ours.remasked if ours.remasked is not None else none)[
                    i, :g
                ],
                "rewritten": (ours.rewritten if ours.rewritten is not None else none)[
                    i, :g
                ],
                "x0": x.extras["our_x0"][i, :g],
                "confidence": x.confidence[i, :g].float(),
                **{f"extra_{k}": v[i, :g].float() for k, v in extras.items()},
            }
            np.savez(
                f"{config.policy_args['out']}_{int(x.steps_np[i]):04d}.npz",
                **{k: v.cpu().numpy() for k, v in arrays.items()},
            )
        return DlmStepDecision(
            revealed=revealed, remasked=remasked, rewritten=rewritten
        )


class ReplayFeaturesPolicy(ReplayPolicy):
    """ReplayPolicy for an inner policy that reads the hidden states."""

    name = "test_replay_features"
    needs_hidden_states = True
