"""Loss reductions shared by the multimodal GRPO trainers."""

from __future__ import annotations

from typing import Optional

import torch


SUPPORTED_LOSS_TYPES = ("grpo", "bnpo", "dr_grpo")


def validate_loss_type(loss_type: str) -> str:
    """Validate and return a GRPO loss reduction name."""

    if loss_type not in SUPPORTED_LOSS_TYPES:
        supported = ", ".join(f"{name!r}" for name in SUPPORTED_LOSS_TYPES)
        raise ValueError(f"Unknown loss type {loss_type!r}; expected one of: {supported}.")
    return loss_type


def aggregate_grpo_loss(
    per_token_loss: torch.Tensor,
    completion_mask: torch.Tensor,
    loss_type: str = "grpo",
    max_completion_length: Optional[int] = None,
) -> torch.Tensor:
    """Reduce masked token losses according to the selected GRPO objective.

    ``grpo`` preserves the reduction used by this project: each completion is
    normalized by its number of active tokens before averaging the batch. The
    ``bnpo`` reduction normalizes by the number of active tokens in the batch.
    ``dr_grpo`` uses a fixed denominator of ``batch_size * max_completion_length``
    (the Dr. GRPO objective), so shorter completions do not receive a larger
    update solely because they contain fewer tokens.
    """

    if per_token_loss.ndim != 2 or completion_mask.ndim != 2:
        raise ValueError("per_token_loss and completion_mask must both be rank-2 tensors")
    if per_token_loss.shape != completion_mask.shape:
        raise ValueError(
            "per_token_loss and completion_mask must have the same shape, "
            f"got {tuple(per_token_loss.shape)} and {tuple(completion_mask.shape)}"
        )
    if per_token_loss.shape[0] == 0:
        raise ValueError("GRPO loss reduction requires a non-empty batch")

    loss_type = validate_loss_type(loss_type)
    mask = completion_mask.to(dtype=per_token_loss.dtype)
    masked_loss = per_token_loss * mask

    if loss_type == "grpo":
        # Clamp only protects malformed all-padding samples; valid completions
        # retain exactly the historical per-completion mean reduction.
        completion_lengths = mask.sum(dim=-1).clamp_min(1.0)
        return (masked_loss.sum(dim=-1) / completion_lengths).mean()

    if loss_type == "bnpo":
        return masked_loss.sum() / mask.sum().clamp_min(1.0)

    if max_completion_length is None or max_completion_length <= 0:
        raise ValueError("max_completion_length must be positive when loss_type='dr_grpo'")
    denominator = per_token_loss.new_tensor(per_token_loss.shape[0] * max_completion_length)
    return masked_loss.sum() / denominator
