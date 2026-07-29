"""Official ASRF local-maximum boundary selection."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def _one_dimensional_boundary(probabilities: Tensor) -> Tensor:
    if probabilities.ndim == 3:
        if probabilities.shape[0] != 1 or probabilities.shape[1] != 1:
            raise ValueError("A single boundary map must have shape [1,1,T].")
        return probabilities[0, 0]
    if probabilities.ndim == 2:
        if probabilities.shape[0] == 1:
            return probabilities[0]
        raise ValueError("A single boundary map must have shape [T] or [1,T].")
    if probabilities.ndim != 1:
        raise ValueError("Boundary probabilities must have shape [T], [1,T], or [1,1,T].")
    return probabilities


def _select_one(probabilities: Tensor, valid_mask: Tensor, threshold: float) -> list[int]:
    values = _one_dimensional_boundary(probabilities).detach().clone().float()
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=values.device)
    if mask.shape != values.shape:
        raise ValueError("valid_mask must match the boundary temporal width.")
    valid_indices = torch.where(mask)[0]
    if not valid_indices.numel():
        return []
    # Official code treats frame zero as a boundary and uses <, so equality
    # at threshold survives. Invalid/padded positions are always zeroed.
    values[~mask] = 0.0
    values[values < float(threshold)] = 0.0
    length = int(valid_indices[-1]) + 1
    if length == 1:
        return [int(valid_indices[0])] if int(valid_indices[0]) == 0 else []
    peak = torch.zeros_like(values, dtype=torch.bool)
    if mask[0]:
        peak[0] = True
    if length > 2:
        peak[1 : length - 1] = (
            (values[: length - 2] < values[1 : length - 1])
            & (values[2:length] < values[1 : length - 1])
            & mask[1 : length - 1]
        )
    return torch.where(peak & mask)[0].tolist()


def select_boundary_peaks(
    boundary_probabilities: Tensor | Sequence[float],
    valid_mask: Tensor | Sequence[bool] | None = None,
    *,
    threshold: float = 0.5,
) -> list[int] | list[list[int]]:
    """Return sorted unique official-style boundary peaks.

    The implementation is intentionally a strict interior maximum.  Frame
    zero is included as the sequence-start boundary when valid; the final
    frame is not an interior candidate.  For a batch, one list is returned
    per sample.
    """

    probabilities = torch.as_tensor(boundary_probabilities)
    if probabilities.ndim == 1:
        mask = torch.ones_like(probabilities, dtype=torch.bool) if valid_mask is None else torch.as_tensor(valid_mask, dtype=torch.bool)
        return _select_one(probabilities, mask, threshold)
    if probabilities.ndim == 2 and probabilities.shape[0] == 1 and valid_mask is not None and torch.as_tensor(valid_mask).ndim == 1:
        return _select_one(probabilities, torch.as_tensor(valid_mask, dtype=torch.bool), threshold)
    if probabilities.ndim == 3:
        if probabilities.shape[1] != 1:
            raise ValueError("Batched BRB probabilities must have shape [B,1,T].")
        probabilities = probabilities[:, 0]
    if probabilities.ndim != 2:
        raise ValueError("Batched boundary probabilities must have shape [B,T] or [B,1,T].")
    batch_size, temporal_width = probabilities.shape
    if valid_mask is None:
        masks = torch.ones((batch_size, temporal_width), dtype=torch.bool)
    else:
        masks = torch.as_tensor(valid_mask, dtype=torch.bool)
        if masks.shape != probabilities.shape:
            raise ValueError("Batched valid_mask must match [B,T].")
    return [_select_one(probabilities[index], masks[index], threshold) for index in range(batch_size)]


def greedy_score_guided_nms(
    candidate_peaks: Sequence[int],
    peak_probabilities: Sequence[float] | Tensor,
    minimum_distance_frames: int = 0,
) -> list[int]:
    """Apply deterministic greedy score-guided one-dimensional NMS.

    Candidates are processed by descending score, then ascending frame index.
    A selected candidate suppresses candidates strictly closer than
    ``minimum_distance_frames``.  The returned frame indices are chronological;
    no peak location is moved or averaged.  A distance of zero disables
    suppression while still returning the unique candidate frame set in
    chronological order.
    """

    distance = int(minimum_distance_frames)
    if distance < 0:
        raise ValueError("minimum_distance_frames must be non-negative.")
    scores = torch.as_tensor(peak_probabilities, dtype=torch.float64).detach().cpu().tolist()
    peaks = [int(peak) for peak in candidate_peaks]
    if len(peaks) != len(scores):
        raise ValueError("candidate_peaks and peak_probabilities must have equal length.")
    if not peaks:
        return []

    # Keep the highest score for duplicate frame indices.  The original input
    # position is a final deterministic tie-breaker for otherwise identical
    # duplicate records.
    ranked = sorted(zip(peaks, (float(score) for score in scores), range(len(peaks))), key=lambda item: (-item[1], item[0], item[2]))
    unique_ranked: list[tuple[int, float, int]] = []
    seen: set[int] = set()
    for item in ranked:
        if item[0] not in seen:
            unique_ranked.append(item)
            seen.add(item[0])
    if distance == 0:
        return sorted(seen)

    selected: list[int] = []
    for peak, _, _ in unique_ranked:
        if all(abs(peak - retained) >= distance for retained in selected):
            selected.append(peak)
    return sorted(selected)


__all__ = ["greedy_score_guided_nms", "select_boundary_peaks"]
