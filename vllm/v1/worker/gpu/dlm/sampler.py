# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampler for full-sequence masked diffusion: one denoising step per call.

Per step and per policy group it (1) gathers each request's canvas from
``req_states.all_token_ids`` (where the prefill path reads it next step),
(2) projects the canvas rows' hidden states to fp32 logits in row chunks and
lets the policy score them, (3) lets the policy select, and (4) writes the
new canvas back. On a request's last step it returns the canvas as its G
output tokens. Every request reports all query tokens as rejected, so
num_computed_tokens never moves. The step index comes from the scheduler;
there is no host synchronization.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.dlm.policy import (
    DlmRequestConfig,
    DlmScoreInputs,
    DlmSelectInputs,
    UnmaskPolicy,
    get_unmask_policy_cls,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import NO_PENALTIES_STATE

logger = init_logger(__name__)

# Transient logits budget of one row chunk (rows * V * policy.bytes_per_logit).
_CHUNK_BYTES = 1 << 30


def _h2d(array: np.ndarray, device: torch.device) -> torch.Tensor:
    if device.type == "cpu":  # CPU tests
        return torch.from_numpy(array)
    return async_tensor_h2d(array, device=device)


_DEFAULT_POLICY = {"llada": "llada_low_confidence", "dream": "dream_maskgit_plus"}


class FullSeqDiffusionSampler:
    # The runner hands this sampler hidden states and compute_logits instead
    # of a [num_logits, V] logits tensor (model_runner.sample).
    consumes_hidden_states = True

    def __init__(self, sampler: Any, model_state: Any) -> None:
        self.sampling_states = sampler.sampling_states
        self.logprob_token_ids_state = sampler.logprob_token_ids_state
        self.req_states = sampler.req_states
        self.logprobs_mode = sampler.logprobs_mode
        self.model_state = model_state
        model_config = model_state.model_config
        self.family = model_config.full_sequence_diffusion
        self.mask_id = int(model_config.hf_config.mask_token_id)
        self.vocab_size = model_config.get_vocab_size()
        self.logits_shift = int(getattr(model_state.model, "dlm_logits_shift", 0))
        self.device = model_state.device
        self.max_num_reqs = model_state.max_num_reqs
        # P + 2G <= max_model_len (checked at admission), so G < max_len / 2.
        self.max_canvas = max(1, model_state.max_model_len // 2)
        self.configs: list[DlmRequestConfig | None] = [None] * self.max_num_reqs
        self.failed = [False] * self.max_num_reqs
        self.policies: dict[str, UnmaskPolicy] = {}
        self.policy_states: dict[str, dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Request state
    # ------------------------------------------------------------------

    def _policy(self, name: str) -> UnmaskPolicy:
        policy = self.policies.get(name)
        if policy is None:
            policy = get_unmask_policy_cls(name)()
            self.policies[name] = policy
            self.policy_states[name] = policy.alloc_state(
                self.max_num_reqs, self.max_canvas, self.device
            )
        return policy

    def add_request(self, req_idx: int, sampling_params: Any) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)
        config = DlmRequestConfig.from_sampling_params(sampling_params)
        self.failed[req_idx] = False
        if config is not None:
            config.prompt_len = (
                int(self.req_states.prompt_len.np[req_idx]) - config.gen_length
            )
            try:
                policy = self._policy(config.policy)
                policy.init_request(req_idx, config, self.policy_states[config.policy])
            except Exception:
                # E.g. a plugin policy the frontend found but this worker
                # cannot load. The request is never stepped and returns no
                # tokens on its last step, which the scheduler finishes as an
                # error; the engine keeps running.
                logger.exception(
                    "Diffusion policy %r failed on this worker", config.policy
                )
                self.failed[req_idx] = True
        self.configs[req_idx] = config

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()

    @property
    def penalties_state(self):
        return NO_PENALTIES_STATE

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def sample_hidden_states(
        self,
        hidden_states: torch.Tensor,
        input_batch: Any,
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        query_start_loc = input_batch.query_start_loc_np
        steps = self.model_state.dlm_steps
        num_sampled = np.zeros(num_reqs, dtype=np.int32)
        groups: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        for batch_idx in range(num_reqs):
            slot = int(input_batch.idx_mapping_np[batch_idx])
            config = self.configs[slot]
            if config is None or self.failed[slot]:
                # Not a diffusion request (the runner's kernel warmup), or one
                # whose policy failed to load (add_request).
                continue
            req_id = input_batch.req_ids[batch_idx]
            step = steps.get(req_id)
            if step is None:
                raise RuntimeError(f"no denoising step scheduled for {req_id}")
            groups[config.policy].append((batch_idx, slot, step))
            if step == config.steps - 1:
                num_sampled[batch_idx] = config.gen_length

        # Fresh per step: the async D2H copy of the previous step may still
        # be reading the last one.
        sampled = torch.zeros(
            num_reqs,
            max(1, int(num_sampled.max(initial=0))),
            dtype=torch.int32,
            device=self.device,
        )
        for name, members in groups.items():
            self._step_group(
                self._policy(name),
                self.policy_states[name],
                members,
                hidden_states,
                query_start_loc,
                compute_logits,
                sampled,
            )

        query_lens = np.diff(query_start_loc[: num_reqs + 1]).astype(np.int32)
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=_h2d(num_sampled, self.device),
            # Every query token is "rejected": num_computed_tokens stays 0.
            num_rejected=_h2d(query_lens, self.device),
        )

    def _step_group(
        self,
        policy: UnmaskPolicy,
        state: dict[str, torch.Tensor],
        members: list[tuple[int, int, int]],
        hidden_states: torch.Tensor,
        query_start_loc: np.ndarray,
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
        sampled: torch.Tensor,
    ) -> None:
        device = self.device
        batch_idx = np.array([m[0] for m in members], dtype=np.int64)
        slots = np.array([m[1] for m in members], dtype=np.int64)
        steps_np = np.array([m[2] for m in members], dtype=np.int64)
        configs = [self.configs[s] for s in slots]
        num_rows = len(members)

        gen_len = np.array([c.gen_length for c in configs], dtype=np.int64)
        prompt_len = np.array([c.prompt_len for c in configs], dtype=np.int64)
        steps_per_block = np.array([c.steps_per_block for c in configs], dtype=np.int64)
        num_steps = np.array([c.steps for c in configs], dtype=np.int64)
        blocks = np.array(
            [c.block_range(int(s)) for c, s in zip(configs, steps_np)], dtype=np.int64
        )
        ranges = np.array(
            [policy.row_range(c, int(s)) for c, s in zip(configs, steps_np)],
            dtype=np.int64,
        )
        ranges = np.clip(ranges, 0, gen_len[:, None])
        width = int(gen_len.max())

        # Canvas of request r = all_token_ids[slot, P : P + G].
        all_token_ids = self.req_states.all_token_ids.gpu
        row_stride = all_token_ids.shape[1]
        pos = np.arange(width, dtype=np.int64)
        flat_idx = slots[:, None] * row_stride + np.minimum(
            prompt_len[:, None] + pos[None, :], row_stride - 1
        )
        valid = (pos[None, :] < gen_len[:, None]).astype(np.int64)

        # Canvas rows the policy scores, flattened request-major.
        counts = ranges[:, 1] - ranges[:, 0]
        total = int(counts.sum())
        row_req = np.repeat(np.arange(num_rows, dtype=np.int64), counts)
        row_pos = (
            np.arange(total, dtype=np.int64)
            - np.repeat(np.cumsum(counts) - counts, counts)
            + np.repeat(ranges[:, 0], counts)
        )
        # Canvas position j reads hidden row P + j - shift of its request.
        hidden_rows = (
            query_start_loc[batch_idx].astype(np.int64)[row_req]
            + prompt_len[row_req]
            + row_pos
            - self.logits_shift
        )

        packed = np.concatenate(
            [
                flat_idx.ravel(),
                valid.ravel(),
                row_req,
                row_pos,
                hidden_rows,
                slots,
                steps_np,
                steps_np % steps_per_block,
                steps_per_block,
                num_steps,
                blocks[:, 0],
                blocks[:, 1],
                batch_idx,
            ]
        )
        dev = _h2d(packed, device)
        sizes = [num_rows * width] * 2 + [total] * 3 + [num_rows] * 8
        (
            flat_idx_t,
            valid_t,
            row_req_t,
            row_pos_t,
            hidden_rows_t,
            slots_t,
            steps_t,
            step_in_block_t,
            steps_per_block_t,
            num_steps_t,
            block_lo_t,
            block_hi_t,
            batch_idx_t,
        ) = torch.split(dev, sizes)
        valid_t = valid_t.view(num_rows, width).bool()

        canvas = all_token_ids.view(-1)[flat_idx_t].view(num_rows, width).long()
        is_masked = (canvas == self.mask_id) & valid_t
        x0 = canvas.clone()
        confidence = torch.full(
            (num_rows, width), float("-inf"), dtype=torch.float32, device=device
        )
        extras: dict[str, torch.Tensor] = {}

        seeds = self.sampling_states.seeds.gpu
        temperature = self.sampling_states.temperature.gpu
        chunk = max(1, _CHUNK_BYTES // (self.vocab_size * policy.bytes_per_logit))
        for start in range(0, total, chunk):
            rows = slice(start, start + chunk)
            hidden = hidden_states[hidden_rows_t[rows]]
            logits = compute_logits(hidden)
            chunk_req = row_req_t[rows]
            out = policy.score(
                DlmScoreInputs(
                    logits=logits,
                    hidden=hidden if policy.needs_hidden_states else None,
                    row_req=chunk_req,
                    row_pos=row_pos_t[rows],
                    row_slot=slots_t[chunk_req],
                    row_req_np=row_req[rows],
                    step=steps_t,
                    slots=slots_t,
                    seeds=seeds,
                    temperature=temperature,
                    configs=configs,
                    steps_np=steps_np,
                    state=state,
                    mask_id=self.mask_id,
                )
            )
            del logits
            flat = chunk_req * width + row_pos_t[rows]
            x0.view(-1).index_copy_(0, flat, out.pop("x0").to(torch.int64))
            confidence.view(-1).index_copy_(
                0, flat, out.pop("confidence").to(torch.float32)
            )
            for key, value in out.items():
                buf = extras.get(key)
                if buf is None:
                    buf = extras[key] = value.new_zeros(
                        (num_rows, width, *value.shape[1:])
                    )
                buf.view(num_rows * width, *value.shape[1:]).index_copy_(0, flat, value)

        decision = policy.select(
            DlmSelectInputs(
                x0=x0,
                confidence=confidence,
                extras=extras,
                canvas=canvas,
                is_masked=is_masked,
                valid=valid_t,
                block_lo=block_lo_t,
                block_hi=block_hi_t,
                step=steps_t,
                step_in_block=step_in_block_t,
                steps_per_block=steps_per_block_t,
                num_steps=num_steps_t,
                slots=slots_t,
                seeds=seeds,
                configs=configs,
                steps_np=steps_np,
                state=state,
                mask_id=self.mask_id,
            )
        )
        new_canvas = torch.where(valid_t, decision.canvas, canvas).to(torch.int32)
        # Positions past G hold the value just read, so the clamped duplicate
        # indices write identical values.
        all_token_ids.view(-1).index_copy_(0, flat_idx_t, new_canvas.view(-1))

        out_width = sampled.shape[1]
        if (steps_np == num_steps - 1).any():
            rows_out = new_canvas[:, :out_width]
            if rows_out.shape[1] < out_width:
                rows_out = torch.nn.functional.pad(
                    rows_out, (0, out_width - rows_out.shape[1])
                )
            sampled.index_copy_(0, batch_idx_t, rows_out)

    # ------------------------------------------------------------------
    # Memory profile
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def dummy_run(
        self,
        hidden_states: torch.Tensor,
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Reach the transient peak of one row chunk of the family's default
        policy: fp32 logits, softmax, and (Dream) a full-vocab sort."""
        policy = self._policy(_DEFAULT_POLICY[self.family])
        chunk = max(1, _CHUNK_BYTES // (self.vocab_size * policy.bytes_per_logit))
        index = torch.zeros(chunk, dtype=torch.int64, device=hidden_states.device)
        logits = compute_logits(hidden_states.index_select(0, index))
        probs = torch.softmax(logits, dim=-1)
        if self.family == "dream":
            values, indices = torch.sort(logits, descending=True)
            del values, indices
        del logits, probs
