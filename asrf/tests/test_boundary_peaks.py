from __future__ import annotations

import torch

from asrf.refinement.peaks import select_boundary_peaks


def test_strict_interior_peaks_and_first_frame() -> None:
    values = torch.tensor([0.1, 0.6, 0.8, 0.6, 0.5, 0.7, 0.4])
    assert select_boundary_peaks(values, threshold=0.5) == [0, 2, 5]


def test_threshold_equality_is_retained_but_not_sufficient_without_maximum() -> None:
    assert select_boundary_peaks(torch.tensor([0.5, 0.5, 0.4]), threshold=0.5) == [0]
    assert select_boundary_peaks(torch.tensor([0.5, 0.7, 0.5]), threshold=0.7) == [0, 1]


def test_plateaus_and_final_frame_are_not_selected() -> None:
    assert select_boundary_peaks(torch.tensor([0.5, 0.8, 0.8, 0.5]), threshold=0.5) == [0]
    assert select_boundary_peaks(torch.tensor([0.5, 0.4, 0.9]), threshold=0.5) == [0]


def test_mask_and_batch_are_respected() -> None:
    probabilities = torch.tensor([[0.2, 0.9, 0.2, 0.99], [0.2, 0.8, 0.2, 0.1]])
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    assert select_boundary_peaks(probabilities, mask, threshold=0.5) == [[0, 1], [0, 1]]
    assert select_boundary_peaks(torch.zeros(4), torch.zeros(4, dtype=torch.bool)) == []
