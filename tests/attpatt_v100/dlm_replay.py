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
  policy's ``revealed`` [G], ``x0`` [G] and ``confidence`` [G] (fp32).
"""

from dataclasses import replace

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
    bytes_per_logit = 40  # the inner policy's (Dream's is the largest)

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
        ref = torch.tensor(config.policy_args["ref"], dtype=torch.int32)
        buf = state["ref"]
        if buf.shape[1] < config.steps:
            grown = buf.new_zeros(buf.shape[0], config.steps, buf.shape[2])
            grown[:, : buf.shape[1]] = buf
            state["ref"] = buf = grown
        buf[slot, : config.steps, : config.gen_length] = ref.to(buf.device)

    def score(self, x):
        inner = self._inner(x.configs[0].policy_args["inner"])
        out = inner.score(x)
        forced = x.state["ref"][x.row_slot, x.step[x.row_req], x.row_pos]
        out["our_x0"] = out.pop("x0")
        out["x0"] = forced.long()
        return out

    def select(self, x):
        inner = self._inner(x.configs[0].policy_args["inner"])
        ours = inner.select(replace(x, x0=x.extras["our_x0"]))
        width = x.canvas.shape[1]
        nxt = x.state["ref"][x.slots, x.step, :width].long()
        revealed = x.is_masked & (nxt != x.mask_id)
        for i, config in enumerate(x.configs):
            g = config.gen_length
            np.savez(
                f"{config.policy_args['out']}_{int(x.steps_np[i]):04d}.npz",
                revealed=ours.revealed[i, :g].cpu().numpy(),
                x0=x.extras["our_x0"][i, :g].cpu().numpy(),
                confidence=x.confidence[i, :g].float().cpu().numpy(),
            )
        return DlmStepDecision(revealed=revealed)
