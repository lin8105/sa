from __future__ import annotations

from asrf.refinement.peaks import greedy_score_guided_nms


def test_empty_and_single_candidate() -> None:
    assert greedy_score_guided_nms([], [], 20) == []
    assert greedy_score_guided_nms([12], [0.8], 20) == [12]


def test_equal_score_uses_earlier_frame_tie_break() -> None:
    assert greedy_score_guided_nms([110, 100], [0.9, 0.9], 20) == [100]


def test_exact_distance_is_allowed() -> None:
    assert greedy_score_guided_nms([100, 120], [0.9, 0.8], 20) == [100, 120]


def test_distance_minus_one_suppresses_lower_score() -> None:
    assert greedy_score_guided_nms([100, 119], [0.9, 0.8], 20) == [100]


def test_chain_case_is_greedy_score_guided() -> None:
    assert greedy_score_guided_nms([136, 100, 118], [0.7, 0.9, 0.8], 20) == [100, 136]


def test_unsorted_input_and_chronological_output() -> None:
    assert greedy_score_guided_nms([30, 10, 50], [0.7, 0.9, 0.8], 10) == [10, 30, 50]


def test_zero_distance_reproduces_candidate_peak_set() -> None:
    assert greedy_score_guided_nms([30, 10, 20], [0.7, 0.9, 0.8], 0) == [10, 20, 30]


def test_duplicate_frame_indices_keep_highest_score_deterministically() -> None:
    assert greedy_score_guided_nms([10, 10, 30], [0.4, 0.9, 0.8], 20) == [10, 30]


def test_negative_distance_is_rejected() -> None:
    try:
        greedy_score_guided_nms([1], [0.5], -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative NMS distance must be rejected")
