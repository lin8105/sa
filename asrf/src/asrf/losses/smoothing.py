"""TMSE and Gaussian Similarity-weighted TMSE losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SmoothingLossOutput:
    loss: Tensor
    valid_transition_count: int


def _pair_mask(valid_mask: Tensor, temporal_width: int) -> Tensor:
    mask = torch.as_tensor(valid_mask, dtype=torch.bool)
    if mask.ndim != 2 or mask.shape[1] != temporal_width:
        raise ValueError("valid_mask must have shape [B,T].")
    return mask[:, 1:] & mask[:, :-1]


def _reduce_pair_loss(pair_loss: Tensor, pair_mask: Tensor) -> SmoothingLossOutput:
    # Official code averages each trajectory independently, then averages the
    # batch. This also gives a finite zero for a one-frame/no-pair trajectory.
    per_sample: list[Tensor] = []
    for sample_loss, sample_mask in zip(pair_loss, pair_mask):
        if sample_mask.any():
            per_sample.append(sample_loss[:, sample_mask].mean())
        else:
            per_sample.append(sample_loss.sum() * 0.0)
    loss = torch.stack(per_sample).mean() if per_sample else pair_loss.sum() * 0.0
    return SmoothingLossOutput(loss, int(pair_mask.sum()))


def tmse_loss(
    logits: Tensor,
    valid_mask: Tensor,
    *,
    tau: float = 4.0,
) -> SmoothingLossOutput:
    """Compare adjacent class log probabilities and clamp each square at tau²."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B,C,T].")
    pair_mask = _pair_mask(valid_mask, logits.shape[-1]).to(device=logits.device)
    if logits.shape[-1] < 2:
        return SmoothingLossOutput(logits.sum() * 0.0, 0)
    log_prob = F.log_softmax(logits, dim=1)
    difference = log_prob[:, :, 1:] - log_prob[:, :, :-1]
    pair_loss = difference.square().clamp(max=float(tau) ** 2)
    return _reduce_pair_loss(pair_loss, pair_mask)


def gs_tmse_loss(
    logits: Tensor,
    features: Tensor,
    valid_mask: Tensor,
    *,
    tau: float = 4.0,
    sigma: float = 1.0,
) -> SmoothingLossOutput:
    """TMSE weighted by official Gaussian feature similarity.

    ``features`` are intentionally not detached, matching the official code.
    For CITR, callers pass the encoder features: these are the project-local
    analogue of official ASRF's original precomputed feature input.
    """

    if logits.ndim != 3 or features.ndim != 3 or features.shape[0] != logits.shape[0] or features.shape[-1] != logits.shape[-1]:
        raise ValueError("logits/features must be [B,C,T] and [B,D,T] with matching B,T.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    pair_mask = _pair_mask(valid_mask, logits.shape[-1]).to(device=logits.device)
    if logits.shape[-1] < 2:
        return SmoothingLossOutput(logits.sum() * 0.0, 0)
    log_prob = F.log_softmax(logits, dim=1)
    difference = log_prob[:, :, 1:] - log_prob[:, :, :-1]
    temporal_loss = difference.square().clamp(max=float(tau) ** 2)
    feature_difference = features[:, :, 1:] - features[:, :, :-1]
    similarity = torch.exp(
        -torch.linalg.vector_norm(feature_difference, dim=1)
        / (2.0 * float(sigma) ** 2)
    )
    pair_loss = temporal_loss * similarity.unsqueeze(1)
    return _reduce_pair_loss(pair_loss, pair_mask)


class TMSE(nn.Module):
    def __init__(self, tau: float = 4.0) -> None:
        super().__init__()
        self.tau = float(tau)

    def forward(self, logits: Tensor, valid_mask: Tensor) -> SmoothingLossOutput:
        return tmse_loss(logits, valid_mask, tau=self.tau)


class GSTMSE(nn.Module):
    def __init__(self, tau: float = 4.0, sigma: float = 1.0) -> None:
        super().__init__()
        self.tau = float(tau)
        self.sigma = float(sigma)

    def forward(self, logits: Tensor, features: Tensor, valid_mask: Tensor) -> SmoothingLossOutput:
        return gs_tmse_loss(logits, features, valid_mask, tau=self.tau, sigma=self.sigma)


GaussianSimilarityTMSE = GSTMSE

__all__ = ["GSTMSE", "GaussianSimilarityTMSE", "SmoothingLossOutput", "TMSE", "gs_tmse_loss", "tmse_loss"]
