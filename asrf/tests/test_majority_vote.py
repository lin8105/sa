from __future__ import annotations

import torch

from asrf.refinement.majority_vote import majority_vote_refinement, mean_probability_refinement
from asrf.refinement.segments import TemporalInterval


def _probabilities(labels: list[int], classes: int = 3) -> torch.Tensor:
    result = torch.full((classes, len(labels)), -2.0)
    for index, label in enumerate(labels):
        result[label, index] = 2.0
    return result.softmax(dim=0)


def test_majority_removes_short_false_run() -> None:
    output, diagnostics = majority_vote_refinement(
        _probabilities([0, 0, 1, 0, 0]), [TemporalInterval(0, 5)]
    )
    assert output.tolist() == [0, 0, 0, 0, 0]
    assert diagnostics[0].majority_fraction == 0.8


def test_supplied_boundary_preserves_true_change() -> None:
    output, _ = majority_vote_refinement(
        _probabilities([0, 0, 0, 1, 1]), [TemporalInterval(0, 3), TemporalInterval(3, 5)]
    )
    assert output.tolist() == [0, 0, 0, 1, 1]


def test_tie_uses_largest_probability_sum_then_first_class() -> None:
    probabilities = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
    output, _ = majority_vote_refinement(probabilities, [TemporalInterval(0, 2)])
    assert output.tolist() == [0, 0]


def test_mean_probability_is_a_separate_diagnostic() -> None:
    probabilities = torch.tensor([[0.51, 0.01, 0.51], [0.49, 0.99, 0.49]])
    majority, _ = majority_vote_refinement(probabilities, [TemporalInterval(0, 3)])
    mean_probability, _ = mean_probability_refinement(probabilities, [TemporalInterval(0, 3)])
    assert majority.tolist() == [0, 0, 0]
    assert mean_probability.tolist() == [1, 1, 1]
