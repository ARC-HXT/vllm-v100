# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The official LLaDA and Dream unmasking rules as built-in policies.

Semantics follow the official code line by line:
- LLaDA: ML-GSAI/LLaDA generate.py (commit f2683fd9) ``generate`` :96-197.
- Dream: Dream-org/Dream-v0-Instruct-7B generation_utils.py ``_sample``
  :364-455 and ``sample_tokens`` :37-90.

Deliberate differences (plan 2.9 / 2.8):
- Logits are fp32 (the lm_head accumulates into fp32) instead of the model
  dtype; the algorithm is unchanged.
- top-k selection breaks confidence ties by the smaller canvas position
  (``torch.topk`` leaves tie order unspecified).
- Randomness (temperature > 0, ``random`` remasking, Dream ``origin`` and
  ``alg_temp``) uses vLLM's seeded hash noise; it matches the official
  distribution, not its bits. Multinomial without replacement over
  softmax(conf / alg_temp) is drawn as Gumbel-top-k (same distribution).
"""

from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.dlm.policy import (
    DlmRequestConfig,
    DlmScoreInputs,
    DlmSelectInputs,
    DlmStepDecision,
    UnmaskPolicy,
    dlm_noise_pos,
    hash_uniform,
    register_unmask_policy,
)

# Token ids hard-coded in the official LLaDA generate.py (:174, :178).
LLADA_EOS_ID = 126081
LLADA_EOT_ID = 126348

# Domain salts so the hash draws of different uses never coincide (the
# Gumbel sampler keys its draws by vocab id, all < 2**24).
_SALT_RANDOM_REMASK = 1 << 28
_SALT_TRANSFER = (1 << 28) + 1
_SALT_ALG_TEMP = (1 << 28) + 2


def topk_mask(scores: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """[R, W] bool: the k[r] highest scores of each row. Ties go to the
    smaller position (stable descending sort). No host sync."""
    num_rows, width = scores.shape
    order = torch.sort(scores, dim=1, descending=True, stable=True).indices
    rank = torch.empty_like(order)
    rank.scatter_(
        1, order, torch.arange(width, device=scores.device).expand(num_rows, width)
    )
    return rank < k.unsqueeze(1)


def _row_flags(x: DlmScoreInputs, values: list[Any]) -> np.ndarray:
    return np.asarray(values)[x.row_req_np]


def _to_device(a: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    if like.device.type == "cpu":  # CPU tests
        return torch.from_numpy(a)
    return async_tensor_h2d(a, device=like.device)


# --------------------------------------------------------------------------
# LLaDA
# --------------------------------------------------------------------------


class LLaDAPolicy(UnmaskPolicy):
    families: ClassVar[tuple[str, ...]] = ("llada",)
    remasking: ClassVar[str]
    bytes_per_logit = 12  # logits + softmax + slack

    _BOOL_ARGS = ("logits_eos_inf", "confidence_eos_eot_inf")

    def validate(self, args: dict[str, Any], config: DlmRequestConfig) -> None:
        unknown = set(args) - set(self._BOOL_ARGS)
        if unknown:
            raise ValueError(
                f"{self.name}: unsupported diffusion_policy_args {sorted(unknown)} "
                "(cfg_scale and the end-think boost are not supported)"
            )
        for key in self._BOOL_ARGS:
            value = args.setdefault(key, False)
            if not isinstance(value, (bool, int)) or value not in (0, 1):
                raise ValueError(f"{self.name}: {key} must be a boolean")
            args[key] = bool(value)

    def row_range(self, config: DlmRequestConfig, step: int) -> tuple[int, int]:
        # Only the current block can be revealed (generate.py:189-192).
        return config.block_range(step)

    def alloc_state(self, max_num_reqs, max_canvas, device):
        # Masks in the current block when the block started
        # (get_num_transfer_tokens, generate.py:22-40,152-153).
        return {
            "block_masks": torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        }

    def score(self, x: DlmScoreInputs) -> dict[str, torch.Tensor]:
        logits = x.logits
        cfgs = x.configs
        row_step = x.step[x.row_req]
        greedy = [c.temperature == 0.0 for c in cfgs]

        eos_inf = [c.policy_args["logits_eos_inf"] for c in cfgs]
        conf_eos = [c.policy_args["confidence_eos_eot_inf"] for c in cfgs]
        if any(eos_inf) or any(conf_eos):
            # The official code edits logits in place; the engine still
            # needs the raw ones (logprobs, trajectory).
            logits = logits.clone()
        if any(eos_inf):
            # generate.py:173-174
            rows = _to_device(_row_flags(x, eos_inf), logits)
            col = logits[:, LLADA_EOS_ID]
            logits[:, LLADA_EOS_ID] = torch.where(rows, float("-inf"), col)

        # generate.py:176-177: x0 = argmax(add_gumbel_noise(logits, T)).
        x0 = logits.argmax(dim=-1)
        if not all(greedy):
            from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

            # argmax(exp(l) / (-log u)^T) = argmax(l / T + Gumbel): the
            # official float64 Gumbel-max, drawn from seeded hash noise.
            noisy = gumbel_sample(
                logits,
                x.row_slot,
                x.temperature,
                x.seeds,
                dlm_noise_pos(row_step, x.row_pos),
                apply_temperature=True,
                is_drafting=False,
                use_fp64=True,
            )
            row_greedy = _to_device(_row_flags(x, greedy), logits)
            x0 = torch.where(row_greedy, x0, noisy)

        if any(conf_eos):
            # generate.py:178: `logits_with_noise[:, :, 126081] =
            # logits[:, :, 126348] = -inf`, after x0. With T == 0,
            # add_gumbel_noise returns `logits` itself (:14-15), so both ids
            # leave the softmax; with T > 0 only 126348 does.
            rows = _to_device(_row_flags(x, conf_eos), logits)
            rows_greedy = _to_device(
                _row_flags(x, [a and g for a, g in zip(conf_eos, greedy)]), logits
            )
            logits[:, LLADA_EOT_ID] = torch.where(
                rows, float("-inf"), logits[:, LLADA_EOT_ID]
            )
            logits[:, LLADA_EOS_ID] = torch.where(
                rows_greedy, float("-inf"), logits[:, LLADA_EOS_ID]
            )

        if self.remasking == "low_confidence":
            # generate.py:180-183
            probs = F.softmax(logits, dim=-1)
            confidence = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            del probs
        else:
            # generate.py:184-185: uniform random confidence.
            confidence = hash_uniform(
                x.seeds[x.row_slot],
                dlm_noise_pos(row_step, x.row_pos),
                _SALT_RANDOM_REMASK,
            )
        return {"x0": x0, "confidence": confidence}

    def select(self, x: DlmSelectInputs) -> DlmStepDecision:
        width = x.canvas.shape[1]
        pos = torch.arange(width, device=x.canvas.device)
        in_block = (
            x.valid & (pos >= x.block_lo.unsqueeze(1)) & (pos < x.block_hi.unsqueeze(1))
        )
        candidates = x.is_masked & in_block
        # get_num_transfer_tokens on the block's mask count at block start:
        # base = n // steps, the first n % steps steps take one more.
        block_masks = x.state["block_masks"]
        num_masked = torch.where(
            x.step_in_block == 0, candidates.sum(dim=1), block_masks[x.slots]
        )
        block_masks[x.slots] = num_masked
        k = num_masked // x.steps_per_block + (
            x.step_in_block < num_masked % x.steps_per_block
        )
        # generate.py:187-196: confidence of masked positions in the block,
        # -inf elsewhere; reveal the top k.
        confidence = torch.where(candidates, x.confidence, float("-inf"))
        reveal = topk_mask(confidence, k) & candidates
        return DlmStepDecision(revealed=reveal)


@register_unmask_policy
class LLaDALowConfidence(LLaDAPolicy):
    name = "llada_low_confidence"
    remasking = "low_confidence"


@register_unmask_policy
class LLaDARandom(LLaDAPolicy):
    name = "llada_random"
    remasking = "random"


# --------------------------------------------------------------------------
# Dream
# --------------------------------------------------------------------------


class DreamPolicy(UnmaskPolicy):
    families: ClassVar[tuple[str, ...]] = ("dream",)
    alg: ClassVar[str]
    bytes_per_logit = 40  # logits, softmax, and top-p sort (values + ids)

    def validate(self, args: dict[str, Any], config: DlmRequestConfig) -> None:
        unknown = set(args) - {"eps", "alg_temp"}
        if unknown:
            raise ValueError(
                f"{self.name}: unsupported diffusion_policy_args {sorted(unknown)}"
            )
        eps = args.get("eps", 1e-3)
        if not isinstance(eps, (int, float)) or not 0 < eps < 1:
            raise ValueError(f"{self.name}: eps must be in (0, 1)")
        args["eps"] = float(eps)
        alg_temp = args.get("alg_temp")
        if alg_temp is not None:
            if not isinstance(alg_temp, (int, float)) or alg_temp < 0:
                raise ValueError(f"{self.name}: alg_temp must be >= 0")
            alg_temp = float(alg_temp)
        args["alg_temp"] = alg_temp
        if config.block_length != config.gen_length:
            raise ValueError(f"{self.name}: Dream has no semi-AR blocks")

    def alloc_state(self, max_num_reqs, max_canvas, device):
        return {
            "timesteps": torch.zeros(
                max_num_reqs, 1025, dtype=torch.float32, device=device
            )
        }

    def init_request(self, slot, config, state) -> None:
        # generation_utils.py:405, on the device like the official run.
        timesteps = state["timesteps"]
        need = config.steps + 1
        if need > timesteps.shape[1]:
            grown = torch.zeros(
                timesteps.shape[0], need, dtype=timesteps.dtype, device=timesteps.device
            )
            grown[:, : timesteps.shape[1]] = timesteps
            state["timesteps"] = timesteps = grown
        timesteps[slot, :need] = torch.linspace(
            1, config.policy_args["eps"], need, device=timesteps.device
        )

    def score(self, x: DlmScoreInputs) -> dict[str, torch.Tensor]:
        # sample_tokens (generation_utils.py:58-90) on fp32 logits.
        logits = x.logits
        cfgs = x.configs
        temps = [c.temperature for c in cfgs]
        if any(t > 0 for t in temps):
            temp = torch.where(
                x.temperature[x.row_slot] > 0, x.temperature[x.row_slot], 1.0
            )
            logits = logits / temp.unsqueeze(1)
        # vLLM normalizes top_p / top_k away for greedy requests, so at T == 0
        # they are off (official would still renormalize the confidence).
        if any(c.top_p < 1.0 for c in cfgs):
            top_p = _to_device(
                _row_flags(x, [c.top_p for c in cfgs]).astype(np.float32), logits
            )
            logits = _top_p_logits(logits, top_p)
        if any(c.top_k > 0 for c in cfgs):
            vocab = logits.shape[-1]
            top_k = _to_device(
                _row_flags(
                    x, [min(c.top_k, vocab) if c.top_k > 0 else vocab for c in cfgs]
                ).astype(np.int64),
                logits,
            )
            logits = _top_k_logits(logits, top_k)
        probs = torch.softmax(logits, dim=-1)

        confidence, x0 = probs.max(dim=-1)
        if any(t > 0 for t in temps):
            from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

            # Categorical(probs).sample() == argmax(log probs + Gumbel).
            sampled = gumbel_sample(
                logits,
                x.row_slot,
                x.temperature,
                x.seeds,
                dlm_noise_pos(x.step[x.row_req], x.row_pos),
                apply_temperature=False,
                is_drafting=False,
            )
            row_sampled = _to_device(_row_flags(x, [t > 0 for t in temps]), logits)
            x0 = torch.where(row_sampled, sampled, x0)
            confidence = torch.where(
                row_sampled, probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1), confidence
            )
        if self.alg == "topk_margin":
            top2 = torch.topk(probs, 2, dim=-1).values
            confidence = top2[:, 0] - top2[:, 1]
        elif self.alg == "entropy":
            confidence = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return {"x0": x0, "confidence": confidence}

    def select(self, x: DlmSelectInputs) -> DlmStepDecision:
        candidates = x.is_masked & x.valid
        timesteps = x.state["timesteps"]
        t = timesteps[x.slots, x.step]
        s = timesteps[x.slots, x.step + 1]
        last = x.step >= x.num_steps - 1
        num_rows, width = x.canvas.shape
        if self.alg == "origin":
            # generation_utils.py:421-426
            p_transfer = torch.where(last, 1.0, 1 - s / t)
            pos = torch.arange(width, device=x.canvas.device).expand(num_rows, width)
            u = hash_uniform(
                x.seeds[x.slots].unsqueeze(1).expand(num_rows, width),
                dlm_noise_pos(x.step.unsqueeze(1), pos),
                _SALT_TRANSFER,
            )
            reveal = candidates & (u < p_transfer.unsqueeze(1))
        else:
            # generation_utils.py:435-450. num_mask_token is float32 (true
            # division of the int count); int() truncates.
            num_masked = candidates.sum(dim=1)
            k = torch.where(
                last,
                num_masked,
                (num_masked.to(torch.float32) * (1 - s / t)).to(torch.int64),
            )
            confidence = torch.where(candidates, x.confidence, float("-inf"))
            alg_temps = [c.policy_args["alg_temp"] or 0.0 for c in x.configs]
            if any(a > 0 for a in alg_temps):
                pos = torch.arange(width, device=x.canvas.device).expand(
                    num_rows, width
                )
                u = hash_uniform(
                    x.seeds[x.slots].unsqueeze(1).expand(num_rows, width),
                    dlm_noise_pos(x.step.unsqueeze(1), pos),
                    _SALT_ALG_TEMP,
                    fp64=True,
                )
                gumbel = -torch.log(-torch.log(u)).to(torch.float32)
                alg_temp = _to_device(
                    np.asarray(alg_temps, dtype=np.float32), confidence
                )
                use = (alg_temp > 0).unsqueeze(1)
                keyed = (
                    confidence / torch.where(alg_temp > 0, alg_temp, 1.0).unsqueeze(1)
                    + gumbel
                )
                confidence = torch.where(use & candidates, keyed, confidence)
            reveal = topk_mask(confidence, k) & candidates
        return DlmStepDecision(revealed=reveal)


def _top_p_logits(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    # generation_utils.py:37-47 with a per-row top_p ([N]); rows with
    # top_p >= 1 keep every token, as when the official code skips it.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    del sorted_logits
    remove = cumulative_probs > top_p.unsqueeze(1)
    del cumulative_probs
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = 0
    remove &= (top_p < 1.0).unsqueeze(1)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask = mask.scatter_(-1, sorted_indices, remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _top_k_logits(logits: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    # generation_utils.py:49-54 with a per-row k ([N]; k = V keeps all).
    kth = torch.sort(logits, dim=-1, descending=True).values.gather(
        -1, (top_k - 1).unsqueeze(1)
    )
    return logits.masked_fill(logits < kth, torch.finfo(logits.dtype).min)


@register_unmask_policy
class DreamOrigin(DreamPolicy):
    name = "dream_origin"
    alg = "origin"


@register_unmask_policy
class DreamMaskGitPlus(DreamPolicy):
    name = "dream_maskgit_plus"
    alg = "maskgit_plus"


@register_unmask_policy
class DreamTopKMargin(DreamPolicy):
    name = "dream_topk_margin"
    alg = "topk_margin"


@register_unmask_policy
class DreamEntropy(DreamPolicy):
    name = "dream_entropy"
    alg = "entropy"
