# SPDX-License-Identifier: Apache-2.0
"""Inference-only GDN prefill using the existing fp32 recurrent update on sm70."""

import torch

from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd,
)


def recurrent_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    core_attn_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return per-token outputs and compact states without modifying inputs.

    Args:
        q: Queries [B, T, H, 128], or B=1 for variable-length batches.
        k: Keys with the same shape and dtype as q.
        v: Values [B, T, HV, 128], where HV is a multiple of H.
        g: Float32 log-decay gates [B, T, HV].
        beta: Write gates [B, T, HV].
        initial_state: Float32 states [N, HV, 128, 128], including sequence zero.
        output_final_state: Whether to return the final states.
        cu_seqlens: Device sequence boundaries [N+1], including empty sequences.
        use_qk_l2norm_in_kernel: Normalize Q/K inside the recurrent kernel.
        core_attn_out: Optional contiguous output storage; padding stays untouched.

    """
    out, final = fused_recurrent_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=k.shape[-1] ** -0.5,
        initial_state=initial_state.contiguous(),
        inplace_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        final_state_only=True,
    )
    if core_attn_out is not None:
        target = core_attn_out.view(-1)[: out.numel()].view_as(out)
        target.copy_(out)
        out = target
    return out, final if output_final_state else None
