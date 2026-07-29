from __future__ import annotations

from asrf.refinement.segments import TemporalInterval, construct_segments


def test_intervals_cover_valid_range_exactly() -> None:
    intervals = construct_segments([4, 2, 4], 7)
    assert intervals == [TemporalInterval(0, 2), TemporalInterval(2, 4), TemporalInterval(4, 7)]
    assert [index for interval in intervals for index in range(interval.start, interval.end)] == list(range(7))


def test_no_boundaries_and_edge_inputs() -> None:
    assert construct_segments([], 5) == [TemporalInterval(0, 5)]
    assert construct_segments([0, 5, -1, 2], 5) == [TemporalInterval(0, 2), TemporalInterval(2, 5)]


def test_empty_trajectory_has_no_intervals() -> None:
    assert construct_segments([0], 0) == []
