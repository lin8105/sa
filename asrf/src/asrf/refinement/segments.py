"""Temporal interval construction for boundary-based refinement."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class TemporalInterval:
    """A half-open interval ``[start, end)`` in valid heatmap columns."""

    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


def construct_segments(boundary_indices: Iterable[int], valid_length: int) -> list[TemporalInterval]:
    """Construct non-overlapping intervals covering exactly ``[0, valid_length)``."""

    if valid_length < 0:
        raise ValueError("valid_length must be non-negative.")
    if valid_length == 0:
        return []
    starts = sorted({int(index) for index in boundary_indices if 0 <= int(index) < valid_length})
    starts = [0] + [index for index in starts if index != 0]
    ends = starts[1:] + [valid_length]
    return [TemporalInterval(start, end) for start, end in zip(starts, ends) if start < end]


__all__ = ["TemporalInterval", "construct_segments"]
