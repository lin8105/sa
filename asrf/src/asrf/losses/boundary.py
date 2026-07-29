"""Masked boundary-regression BCE loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class BoundaryLossOutput:
    loss: Tensor
    valid_frame_count: int
    positive_count: int
    negative_count: int


def masked_boundary_bce(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    *,
    positive_weight: float | Tensor | None = None,
) -> BoundaryLossOutput:
    """Official BCE-with-logits with reciprocal-positive-ratio weighting."""

    if logits.ndim == 3 and logits.shape[1] == 1:
        logits_2d = logits[:, 0]
    elif logits.ndim == 2:
        logits_2d = logits
    else:
        raise ValueError("BRB logits must have shape [B,1,T] or [B,T].")
    target_2d = targets[:, 0] if targets.ndim == 3 and targets.shape[1] == 1 else targets
    mask = torch.as_tensor(valid_mask, device=logits.device, dtype=torch.bool)
    if target_2d.shape != logits_2d.shape or mask.shape != logits_2d.shape:
        raise ValueError("targets and valid_mask must match BRB logits [B,T].")
    valid = mask
    valid_count = int(valid.sum())
    positive_count = int((target_2d[valid] > 0.5).sum())
    negative_count = valid_count - positive_count
    if valid_count == 0:
        return BoundaryLossOutput(logits.sum() * 0.0, 0, 0, 0)
    pos_weight = None if positive_weight is None else torch.as_tensor(positive_weight, device=logits.device, dtype=logits.dtype)
    losses = F.binary_cross_entropy_with_logits(
        logits_2d, target_2d.to(dtype=logits.dtype), pos_weight=pos_weight, reduction="none"
    )
    per_sample: list[Tensor] = []
    for sample_loss, sample_mask in zip(losses, valid):
        if sample_mask.any():
            per_sample.append(sample_loss[sample_mask].mean())
        else:
            per_sample.append(sample_loss.sum() * 0.0)
    return BoundaryLossOutput(torch.stack(per_sample).mean(), valid_count, positive_count, negative_count)


class MaskedBoundaryBCE(nn.Module):
    def __init__(self, positive_weight: float | Tensor | None = None) -> None:
        super().__init__()
        if positive_weight is None:
            self.positive_weight = None  # type: ignore[assignment]
        else:
            self.register_buffer("positive_weight", torch.as_tensor(positive_weight, dtype=torch.float32))

    def forward(self, logits: Tensor, targets: Tensor, valid_mask: Tensor) -> BoundaryLossOutput:
        return masked_boundary_bce(
            logits, targets, valid_mask, positive_weight=self.positive_weight
        )


def reciprocal_positive_weight(positive_count: int, negative_count: int) -> float:
    if positive_count <= 0:
        raise ValueError("positive_count must be positive.")
    return float((positive_count + negative_count) / positive_count)


__all__ = ["BoundaryLossOutput", "MaskedBoundaryBCE", "masked_boundary_bce", "reciprocal_positive_weight"]
