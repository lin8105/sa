from __future__ import annotations

import torch

from asrf.refinement.refine import refine_asrf_predictions


def _make_asb(labels: list[int]) -> torch.Tensor:
    result = torch.full((1, 2, len(labels)), 0.1)
    for index, label in enumerate(labels):
        result[0, label, index] = 0.9
    return result


def test_end_to_end_no_boundary_collapses_oversegmentation() -> None:
    result = refine_asrf_predictions(
        _make_asb([0, 0, 1, 0, 0]),
        torch.tensor([[[0.9, 0.1, 0.1, 0.1, 0.1]]]),
        torch.ones(1, 5, dtype=torch.bool),
    )
    assert result.raw_labels.tolist() == [[0, 0, 1, 0, 0]]
    assert result.selected_boundaries == ((0,),)
    assert result.refined_labels.tolist() == [[0, 0, 0, 0, 0]]


def test_boundary_preserves_two_segments_and_padding_is_excluded() -> None:
    result = refine_asrf_predictions(
        _make_asb([0, 0, 0, 1, 1]),
        torch.tensor([[[0.9, 0.1, 0.1, 0.8, 0.1]]]),
        torch.tensor([[True, True, True, True, True]]),
    )
    assert result.selected_boundaries == ((0, 3),)
    assert result.refined_labels.tolist() == [[0, 0, 0, 1, 1]]

    padded = refine_asrf_predictions(
        _make_asb([0, 0, 1, 1, 1]),
        torch.tensor([[[0.9, 0.1, 0.8, 0.1, 0.1]]]),
        torch.tensor([[True, True, True, False, False]]),
    )
    assert padded.refined_labels.tolist() == [[0, 0, 0, -100, -100]]


def test_refinement_result_has_exact_intervals_and_length() -> None:
    result = refine_asrf_predictions(
        _make_asb([0, 1, 1, 0]),
        torch.tensor([[[0.9, 0.1, 0.9, 0.1]]]),
        torch.ones(1, 4, dtype=torch.bool),
    )
    assert result.intervals[0][0].start == 0
    assert result.intervals[0][-1].end == 4
    assert result.refined_labels.shape == (1, 4)
