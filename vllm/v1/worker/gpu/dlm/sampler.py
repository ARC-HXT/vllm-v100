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

Standard logprobs (``SamplingParams.logprobs``): each position reports the
distribution of the step that wrote its final token (reveal or rewrite; a
position still masked after the last step reports the last step). They are
computed with ``compute_topk_scores`` for the policy's x0 and kept in a
per-slot stash until the last step. The optional per-step trajectory is
described in ``trace.py``.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.dlm.policy import (
    DlmRequestConfig,
    DlmScoreInputs,
    DlmSelectInputs,
    UnmaskPolicy,
    get_unmask_policy_cls,
)
from vllm.v1.worker.gpu.dlm.trace import (
    FLAG_REMASKED,
    FLAG_REVEALED,
    FLAG_REWRITTEN,
    TRACE_FIELDS,
    DlmTraceTensors,
    trace_row_stats,
)
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import NO_PENALTIES_STATE

logger = init_logger(__name__)

# Transient logits budget of one row chunk (rows * V * bytes per logit).
_CHUNK_BYTES = 1 << 30
# Extra transient bytes per logit: trajectory (log-softmax + probs), logprobs.
_TRACE_BYTES_PER_LOGIT = 8
_LOGPROBS_BYTES_PER_LOGIT = 4


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
        # Logprob stash [max_num_reqs * max_canvas, 1 + K], allocated on the
        # first request that asks for logprobs and widened on demand.
        self.lp_ids: torch.Tensor | None = None
        self.lp_logp: torch.Tensor | None = None
        self.lp_rank: torch.Tensor | None = None

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

    def _ensure_lp_stash(self, num_logprobs: int) -> None:
        width = 1 + num_logprobs
        if self.lp_ids is not None and self.lp_ids.shape[1] >= width:
            return
        rows = self.max_num_reqs * self.max_canvas
        ids = torch.zeros(rows, width, dtype=torch.int32, device=self.device)
        logp = torch.full(
            (rows, width), float("-inf"), dtype=torch.float32, device=self.device
        )
        if self.lp_ids is not None:
            old = self.lp_ids.shape[1]
            ids[:, :old] = self.lp_ids
            logp[:, :old] = self.lp_logp
            rank = self.lp_rank
        else:
            rank = torch.zeros(rows, dtype=torch.int32, device=self.device)
        self.lp_ids, self.lp_logp, self.lp_rank = ids, logp, rank

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
            if config.num_logprobs >= 0:
                self._ensure_lp_stash(config.num_logprobs)
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
        lp_slots: list[int] = []  # slots of the requests finishing this step
        want_logprobs = False
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
                lp_slots.append(slot)
                want_logprobs |= config.num_logprobs >= 0

        # Fresh per step: the async D2H copy of the previous step may still
        # be reading the last one.
        sampled = torch.zeros(
            num_reqs,
            max(1, int(num_sampled.max(initial=0))),
            dtype=torch.int32,
            device=self.device,
        )
        trace_parts: list[tuple[list[str], list[int], dict[str, torch.Tensor]]] = []
        for name, members in groups.items():
            self._step_group(
                self._policy(name),
                self.policy_states[name],
                members,
                hidden_states,
                input_batch,
                compute_logits,
                sampled,
                trace_parts,
            )

        logprobs_tensors = None
        if want_logprobs:
            logprobs_tensors = self._gather_logprobs(lp_slots, num_sampled)
        query_lens = np.diff(query_start_loc[: num_reqs + 1]).astype(np.int32)
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=_h2d(num_sampled, self.device),
            # Every query token is "rejected": num_computed_tokens stays 0.
            num_rejected=_h2d(query_lens, self.device),
            dlm_trace=self._pack_trace(trace_parts) if trace_parts else None,
        )

    def _gather_logprobs(
        self, slots: list[int], num_sampled: np.ndarray
    ) -> LogprobsTensors:
        """Stash rows of the finishing requests, in batch order (requests
        without logprobs get rows the scheduler never reads)."""
        assert self.lp_ids is not None
        configs = [self.configs[s] for s in slots]
        rows = np.concatenate(
            [
                s * self.max_canvas + np.arange(c.gen_length, dtype=np.int64)
                for s, c in zip(slots, configs)
            ]
        )
        index = _h2d(rows, self.device)
        cu = np.concatenate(([0], np.cumsum(num_sampled))).tolist()
        return LogprobsTensors(
            logprob_token_ids=self.lp_ids.index_select(0, index),
            logprobs=self.lp_logp.index_select(0, index),
            selected_token_ranks=self.lp_rank.index_select(0, index),
            cu_num_generated_tokens=cu,
        )

    @staticmethod
    def _pack_trace(
        parts: list[tuple[list[str], list[int], dict[str, torch.Tensor]]],
    ) -> DlmTraceTensors:
        req_ids: list[str] = []
        topks: list[int] = []
        offsets = [0]
        for ids, ks, fields in parts:
            req_ids += ids
            topks += ks
            rows = fields["x0"].shape[0]
            # Rows of one part are its requests' G_r, in order.
            offsets += (offsets[-1] + np.cumsum(fields.pop("_rows"))).tolist()
            assert offsets[-1] - offsets[-1 - len(ids)] == rows
        width = max(p[2]["topk_ids"].shape[1] for p in parts)

        def pad(t: torch.Tensor, value: float) -> torch.Tensor:
            return F.pad(t, (0, width - t.shape[1]), value=value)

        fields = {}
        for name in TRACE_FIELDS:
            chunks = [p[2][name] for p in parts]
            if name in ("topk_ids", "topk_logp"):
                chunks = [pad(c, 0) for c in chunks]
            fields[name] = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        return DlmTraceTensors(fields, req_ids, offsets, topks)

    def _step_group(
        self,
        policy: UnmaskPolicy,
        state: dict[str, torch.Tensor],
        members: list[tuple[int, int, int]],
        hidden_states: torch.Tensor,
        input_batch: Any,
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
        sampled: torch.Tensor,
        trace_parts: list,
    ) -> None:
        device = self.device
        query_start_loc = input_batch.query_start_loc_np
        batch_idx = np.array([m[0] for m in members], dtype=np.int64)
        slots = np.array([m[1] for m in members], dtype=np.int64)
        steps_np = np.array([m[2] for m in members], dtype=np.int64)
        configs = [self.configs[s] for s in slots]
        num_rows = len(members)

        gen_len = np.array([c.gen_length for c in configs], dtype=np.int64)
        prompt_len = np.array([c.prompt_len for c in configs], dtype=np.int64)
        steps_per_block = np.array([c.steps_per_block for c in configs], dtype=np.int64)
        num_steps = np.array([c.steps for c in configs], dtype=np.int64)
        traced = np.array([c.trace_topk >= 0 for c in configs])
        trace_topk = max((c.trace_topk for c in configs), default=-1)
        num_logprobs = max((c.num_logprobs for c in configs), default=-1)
        blocks = np.array(
            [c.block_range(int(s)) for c, s in zip(configs, steps_np)], dtype=np.int64
        )
        ranges = np.array(
            [policy.row_range(c, int(s)) for c, s in zip(configs, steps_np)],
            dtype=np.int64,
        )
        # The trajectory covers every canvas position.
        ranges[traced] = [0, np.iinfo(np.int64).max]
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
        # Per-row results of this step that the engine keeps ([R * W, ...]).
        kept: dict[str, torch.Tensor] = {}

        def keep(name: str, flat: torch.Tensor, value: torch.Tensor) -> None:
            buf = kept.get(name)
            if buf is None:
                buf = kept[name] = value.new_zeros((num_rows * width, *value.shape[1:]))
            buf.index_copy_(0, flat, value)

        seeds = self.sampling_states.seeds.gpu
        temperature = self.sampling_states.temperature.gpu
        bytes_per_logit = policy.bytes_per_logit
        if trace_topk >= 0:
            bytes_per_logit += _TRACE_BYTES_PER_LOGIT
        if num_logprobs >= 0:
            bytes_per_logit += _LOGPROBS_BYTES_PER_LOGIT
        chunk = max(1, _CHUNK_BYTES // (self.vocab_size * bytes_per_logit))
        logits_mode = self.logprobs_mode.endswith("logits")
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
            flat = chunk_req * width + row_pos_t[rows]
            chunk_x0 = out.pop("x0").to(torch.int64)
            x0.view(-1).index_copy_(0, flat, chunk_x0)
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
            if num_logprobs >= 0:
                # The token a reveal / rewrite of this row would write.
                scores = compute_topk_scores(
                    logits, num_logprobs, chunk_x0, logits_mode=logits_mode
                )
                keep("lp_ids", flat, scores.logprob_token_ids.to(torch.int32))
                keep("lp_logp", flat, scores.logprobs)
                keep("lp_rank", flat, scores.selected_token_ranks.to(torch.int32))
            if trace_topk >= 0:
                current = canvas.view(-1)[flat]
                tokens = torch.where(current == self.mask_id, chunk_x0, current)
                for key, value in trace_row_stats(logits, tokens, trace_topk).items():
                    keep(key, flat, value.to(TRACE_FIELDS[key]))
            del logits

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
        revealed = decision.revealed & valid_t
        written = revealed
        new_canvas = torch.where(revealed, x0, canvas)
        if decision.rewritten is not None:
            rewritten = decision.rewritten & valid_t
            written = written | rewritten
            new_canvas = torch.where(rewritten, x0, new_canvas)
        if decision.remasked is not None:
            remasked = decision.remasked & valid_t
            new_canvas = torch.where(remasked, self.mask_id, new_canvas)
        new_canvas = new_canvas.to(torch.int32)
        # Positions past G hold the value just read, so the clamped duplicate
        # indices write identical values.
        all_token_ids.view(-1).index_copy_(0, flat_idx_t, new_canvas.view(-1))

        last = steps_np == num_steps - 1
        if num_logprobs >= 0:
            self._stash_logprobs(
                kept, written, new_canvas, valid_t, slots_t, num_steps_t, steps_t
            )
        if traced.any():
            self._trace_rows(
                kept,
                x0,
                new_canvas,
                decision,
                valid_t,
                traced,
                gen_len,
                width,
                [input_batch.req_ids[b] for b in batch_idx[traced]],
                [c.trace_topk for c, t in zip(configs, traced) if t],
                trace_parts,
            )

        out_width = sampled.shape[1]
        if last.any():
            rows_out = new_canvas[:, :out_width]
            if rows_out.shape[1] < out_width:
                rows_out = torch.nn.functional.pad(
                    rows_out, (0, out_width - rows_out.shape[1])
                )
            sampled.index_copy_(0, batch_idx_t, rows_out)

    def _stash_logprobs(
        self,
        kept: dict[str, torch.Tensor],
        written: torch.Tensor,
        new_canvas: torch.Tensor,
        valid: torch.Tensor,
        slots: torch.Tensor,
        num_steps: torch.Tensor,
        steps: torch.Tensor,
    ) -> None:
        """Keep this step's scores where a token was written; on the last
        step also where the position is still masked."""
        assert self.lp_ids is not None
        num_rows, width = written.shape
        last = (steps == num_steps - 1).unsqueeze(1)
        take = (written | (last & (new_canvas == self.mask_id))) & valid
        dest = (
            slots.unsqueeze(1) * self.max_canvas
            + torch.arange(width, device=written.device)
        ).view(-1)
        take = take.view(-1, 1)
        k = kept["lp_ids"].shape[1]
        for stash, name in (
            (self.lp_ids, "lp_ids"),
            (self.lp_logp, "lp_logp"),
        ):
            view = stash[:, :k]
            new = torch.where(take, kept[name], view[dest])
            view[dest] = new
        rank = self.lp_rank
        rank[dest] = torch.where(take.view(-1), kept["lp_rank"], rank[dest])

    def _trace_rows(
        self,
        kept: dict[str, torch.Tensor],
        x0: torch.Tensor,
        new_canvas: torch.Tensor,
        decision: Any,
        valid: torch.Tensor,
        traced: np.ndarray,
        gen_len: np.ndarray,
        width: int,
        req_ids: list[str],
        topks: list[int],
        trace_parts: list,
    ) -> None:
        flags = decision.revealed.to(torch.uint8) * FLAG_REVEALED
        if decision.remasked is not None:
            flags |= decision.remasked.to(torch.uint8) * FLAG_REMASKED
        if decision.rewritten is not None:
            flags |= decision.rewritten.to(torch.uint8) * FLAG_REWRITTEN
        flags = flags * valid
        reqs = np.flatnonzero(traced)
        lens = gen_len[reqs]
        index = np.concatenate(
            [r * width + np.arange(g, dtype=np.int64) for r, g in zip(reqs, lens)]
        )
        index_t = _h2d(index, self.device)
        fields = {
            name: kept[name].index_select(0, index_t)
            for name in ("token_logp", "entropy", "margin", "topk_ids", "topk_logp")
        }
        fields["x0"] = x0.view(-1).index_select(0, index_t).to(torch.int32)
        fields["canvas"] = new_canvas.view(-1).index_select(0, index_t)
        fields["flags"] = flags.view(-1).index_select(0, index_t)
        fields["_rows"] = lens
        trace_parts.append((req_ids, topks, fields))

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
