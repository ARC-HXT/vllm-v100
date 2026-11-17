# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Official LLaDA / Dream sampling code, copied verbatim for the parity tests.

- LLaDA: ML-GSAI/LLaDA generate.py commit f2683fd9, lines 1-200 (the
  `from transformers import ...` line dropped; nothing else changed).
- Dream: Dream-org/Dream-v0-Instruct-7B generation_utils.py lines 37-90
  (top_p_logits, top_k_logits, sample_tokens) and lines 405-450
  (the denoising loop of DreamGenerationMixin._sample), wrapped in
  `dream_sample` below with `self` -> `model`, `self.device` -> `model.device`
  and the hooks / history removed.
"""
# ruff: noqa
import torch
import numpy as np
import torch.nn.functional as F



def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def contains_token_sequence(tokens, sequence):
    if sequence.numel() == 0 or tokens.numel() < sequence.numel():
        return False
    for start in range(tokens.numel() - sequence.numel() + 1):
        if torch.equal(tokens[start:start + sequence.numel()], sequence):
            return True
    return False


def get_next_sequence_token_id(tokens, position, sequence, context_start):
    max_prefix_length = min(sequence.numel() - 1, position - context_start)
    for prefix_length in range(max_prefix_length, 0, -1):
        if torch.equal(tokens[position - prefix_length:position], sequence[:prefix_length]):
            return sequence[prefix_length].item()
    return sequence[0].item()


def apply_end_think_logit_boost(logits, tokens, candidate_mask_index, context_start,
                                total_gen_length, end_think_token_ids=None,
                                end_think_logit_boost=0., end_think_boost_power=2.):
    """Gradually encourage an end-of-thinking token sequence during generation."""
    if end_think_logit_boost <= 0 or not end_think_token_ids or total_gen_length <= 0:
        return logits
    if any(token_id < 0 or token_id >= logits.shape[-1] for token_id in end_think_token_ids):
        return logits

    sequence = torch.tensor(end_think_token_ids, dtype=torch.long, device=logits.device)
    logits = logits.clone()

    for batch_index in range(tokens.shape[0]):
        if contains_token_sequence(tokens[batch_index, context_start:], sequence):
            continue

        candidate_positions = torch.nonzero(
            candidate_mask_index[batch_index], as_tuple=False
        ).flatten()
        if candidate_positions.numel() == 0:
            continue

        position = candidate_positions[0].item()
        generated_length = position - context_start + 1
        progress = min(max(generated_length / float(total_gen_length), 0.), 1.)
        boost = end_think_logit_boost * (progress ** end_think_boost_power)
        token_id = get_next_sequence_token_id(
            tokens[batch_index], position, sequence, context_start
        )
        logits[batch_index, position, token_id] += boost

    return logits


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False,
             confidence_eos_eot_inf=False, end_think_token_ids=None, end_think_logit_boost=0.,
             end_think_boost_power=2., end_think_context_start=None,
             end_think_total_gen_length=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    base_model = getattr(model, 'module', model)
    model_name = getattr(getattr(base_model, 'config', None), '_name_or_path', '')
    if 'illada' in model_name.lower():
        assert prompt.shape[0] == 1, 'iLLaDA currently does not support padded batch generation.'

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    if end_think_context_start is None:
        end_think_context_start = prompt.shape[1]
    if end_think_total_gen_length is None:
        end_think_total_gen_length = gen_length

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length
            candidate_mask_index = mask_index.clone()
            candidate_mask_index[:, :block_start] = False
            candidate_mask_index[:, block_end:] = False
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits = apply_end_think_logit_boost(
                logits,
                x,
                candidate_mask_index,
                end_think_context_start,
                end_think_total_gen_length,
                end_think_token_ids=end_think_token_ids,
                end_think_logit_boost=end_think_logit_boost,
                end_think_boost_power=end_think_boost_power,
            )

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


import torch.distributions as dists


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0

@torch.no_grad()
def dream_sample(model, input_ids, max_length, mask_token_id, steps, eps=1e-3,
                 alg='origin', alg_temp=None, temperature=0.0, top_p=None, top_k=None):
    self = model
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    attention_mask, tok_idx = 'full', None
    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

    for i in range(steps):
        mask_index = (x == mask_token_id)
        logits = self(x, attention_mask, tok_idx).logits
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)


        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]
    
        if alg == 'origin':
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
            transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
            _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
            x[mask_index] = x0.clone()
        else:
            if alg == 'maskgit_plus':
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            elif alg == 'topk_margin':
                confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
            elif alg == 'entropy':
                confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
            else:
                raise RuntimeError(f"Unknown alg: {alg}")
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
            full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence
            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                else:
                    full_confidence = full_confidence / alg_temp
                    full_confidence = F.softmax(full_confidence, dim=-1)
                    transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                x_[mask_index] = x0.clone()
                row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                x[row_indices,transfer_index] = x_[row_indices,transfer_index]
    return x
