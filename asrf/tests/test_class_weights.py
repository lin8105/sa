from __future__ import annotations

import torch

from asrf.losses.classification import masked_class_weighted_cross_entropy, median_frequency_class_weights


def test_median_frequency_known_values() -> None:
    frequencies, median, weights = median_frequency_class_weights(torch.tensor([2, 4, 8]))
    assert torch.allclose(frequencies, torch.tensor([1 / 7, 2 / 7, 4 / 7], dtype=torch.float64))
    assert torch.isclose(median, torch.tensor(2 / 7, dtype=torch.float64))
    assert torch.allclose(weights, torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64))


def test_weighted_ce_ignores_padding() -> None:
    logits = torch.tensor([[[4.0, -4.0, 100.0], [-4.0, 4.0, -100.0]]])
    targets = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[True, True, False]])
    result = masked_class_weighted_cross_entropy(logits, targets, mask)
    expected = torch.nn.functional.cross_entropy(logits[:, :, :2], targets[:, :2])
    assert torch.isclose(result.loss, expected)
    assert result.valid_frame_count == 2


def test_zero_valid_frames_is_finite() -> None:
    logits = torch.randn(1, 3, 4, requires_grad=True)
    result = masked_class_weighted_cross_entropy(logits, torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.bool))
    assert result.valid_frame_count == 0
    assert result.loss.item() == 0.0
    result.loss.backward()
    assert logits.grad is not None
