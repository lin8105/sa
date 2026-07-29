"""Batch-safe refinement API kept outside neural-network forward."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .majority_vote import SegmentVoteDiagnostic, _vote_one
from .peaks import select_boundary_peaks
from .segments import TemporalInterval, construct_segments


@dataclass(frozen=True)
class ASRFRefinementOutput:
    raw_labels: Tensor
    selected_boundaries: tuple[tuple[int, ...], ...]
    intervals: tuple[tuple[TemporalInterval, ...], ...]
    refined_labels: Tensor
    segment_diagnostics: tuple[tuple[SegmentVoteDiagnostic, ...], ...]
    raw_collapsed_sequences: tuple[tuple[int, ...], ...]
    refined_collapsed_sequences: tuple[tuple[int, ...], ...]
    threshold: float
    voting: str


def _final_asb(asb: Tensor) -> Tensor:
    if asb.ndim != 3:
        raise ValueError("ASB probabilities must have shape [B,C,T].")
    return asb


def _final_brb(brb: Tensor) -> Tensor:
    if brb.ndim == 3 and brb.shape[1] == 1:
        return brb[:, 0]
    if brb.ndim == 2:
        return brb
    raise ValueError("BRB probabilities must have shape [B,1,T] or [B,T].")


def _collapse(labels: Tensor) -> tuple[int, ...]:
    values = labels.tolist()
    if not values:
        return ()
    collapsed = [int(values[0])]
    for value in values[1:]:
        if int(value) != collapsed[-1]:
            collapsed.append(int(value))
    return tuple(collapsed)


def refine_asrf_predictions(
    asb_probabilities: Tensor,
    brb_probabilities: Tensor,
    valid_mask: Tensor,
    *,
    threshold: float = 0.5,
    voting: str = "majority",
) -> ASRFRefinementOutput:
    """Select BRB boundaries and refine final ASB labels per valid trajectory."""

    asb = _final_asb(asb_probabilities)
    brb = _final_brb(brb_probabilities)
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=asb.device)
    if mask.shape != asb.shape[:1] + asb.shape[2:] or brb.shape != mask.shape:
        raise ValueError("valid_mask must match the ASB/BRB temporal batch shape.")
    selected = select_boundary_peaks(brb, mask, threshold=threshold)
    assert isinstance(selected, list)
    raw = asb.argmax(dim=1).to(dtype=torch.long)
    refined = torch.full_like(raw, -100)
    intervals_all: list[tuple[TemporalInterval, ...]] = []
    diagnostics_all: list[tuple[SegmentVoteDiagnostic, ...]] = []
    boundary_all: list[tuple[int, ...]] = []
    raw_collapsed_all: list[tuple[int, ...]] = []
    refined_collapsed_all: list[tuple[int, ...]] = []
    for index in range(asb.shape[0]):
        valid_indices = torch.where(mask[index])[0]
        length = int(valid_indices[-1]) + 1 if valid_indices.numel() else 0
        intervals = construct_segments(selected[index], length)
        refined_one, diagnostics = _vote_one(asb[index, :, :length], intervals, voting=voting)
        if length:
            refined[index, :length] = refined_one
        raw_collapsed_all.append(_collapse(raw[index, :length]))
        refined_collapsed_all.append(_collapse(refined_one))
        intervals_all.append(tuple(intervals))
        diagnostics_all.append(tuple(diagnostics))
        boundary_all.append(tuple(int(value) for value in selected[index]))
    return ASRFRefinementOutput(
        raw_labels=raw.masked_fill(~mask, -100),
        selected_boundaries=tuple(boundary_all),
        intervals=tuple(intervals_all),
        refined_labels=refined,
        segment_diagnostics=tuple(diagnostics_all),
        raw_collapsed_sequences=tuple(raw_collapsed_all),
        refined_collapsed_sequences=tuple(refined_collapsed_all),
        threshold=float(threshold),
        voting=voting,
    )


__all__ = ["ASRFRefinementOutput", "refine_asrf_predictions"]
