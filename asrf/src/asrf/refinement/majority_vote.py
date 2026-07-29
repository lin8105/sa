"""Official majority-vote and diagnostic mean-probability refinement."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor

from .segments import TemporalInterval


@dataclass(frozen=True)
class SegmentVoteDiagnostic:
    start: int
    end: int
    duration: int
    class_counts: tuple[int, ...]
    selected_class: int
    majority_fraction: float


def _labels_and_scores(asb: Tensor) -> tuple[Tensor, Tensor | None]:
    if asb.ndim == 1:
        return asb.to(dtype=torch.long), None
    if asb.ndim != 2:
        raise ValueError("ASB input must be labels [T] or scores/probabilities [C,T].")
    return asb.argmax(dim=0).to(dtype=torch.long), asb


def _vote_one(
    asb: Tensor,
    intervals: Sequence[TemporalInterval],
    *,
    voting: str,
) -> tuple[Tensor, list[SegmentVoteDiagnostic]]:
    raw_labels, scores = _labels_and_scores(asb)
    refined = raw_labels.clone()
    diagnostics: list[SegmentVoteDiagnostic] = []
    for interval in intervals:
        segment_labels = raw_labels[interval.start : interval.end]
        counts = torch.bincount(segment_labels, minlength=(scores.shape[0] if scores is not None else int(raw_labels.max().item()) + 1 if raw_labels.numel() else 0))
        modes = torch.where(counts == counts.max())[0].tolist() if counts.numel() else [0]
        if voting == "majority":
            if scores is None or len(modes) == 1:
                chosen = int(modes[0])
            else:
                sums = {mode: float(scores[mode, interval.start : interval.end].sum()) for mode in modes}
                chosen = max(modes, key=lambda mode: (sums[mode], -mode))
        elif voting == "mean_probability":
            if scores is None:
                raise ValueError("mean_probability voting requires class scores/probabilities.")
            chosen = int(scores[:, interval.start : interval.end].mean(dim=1).argmax())
        else:
            raise ValueError("voting must be 'majority' or 'mean_probability'.")
        refined[interval.start : interval.end] = chosen
        diagnostics.append(
            SegmentVoteDiagnostic(
                start=interval.start,
                end=interval.end,
                duration=interval.duration,
                class_counts=tuple(int(value) for value in counts.tolist()),
                selected_class=chosen,
                majority_fraction=float(counts[chosen].item() / max(1, interval.duration)),
            )
        )
    return refined, diagnostics


def majority_vote_refinement(
    asb_probabilities: Tensor,
    intervals: Sequence[TemporalInterval],
) -> tuple[Tensor, list[SegmentVoteDiagnostic]]:
    """Apply official majority voting to one final ASB output."""

    return _vote_one(asb_probabilities, intervals, voting="majority")


def mean_probability_refinement(
    asb_probabilities: Tensor,
    intervals: Sequence[TemporalInterval],
) -> tuple[Tensor, list[SegmentVoteDiagnostic]]:
    """Diagnostic-only segment mean-probability argmax refinement."""

    return _vote_one(asb_probabilities, intervals, voting="mean_probability")


__all__ = [
    "SegmentVoteDiagnostic",
    "majority_vote_refinement",
    "mean_probability_refinement",
]
