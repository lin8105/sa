from __future__ import annotations

import torch

from asrf.models import HeatmapEncoder


def test_heatmap_encoder_preserves_odd_temporal_width() -> None:
    encoder = HeatmapEncoder()
    output = encoder(torch.randn(1, 3, 88, 101))
    assert output.shape == (1, 128, 101)


def test_heatmap_encoder_preserves_even_temporal_width_and_batch() -> None:
    encoder = HeatmapEncoder()
    output = encoder(torch.randn(2, 3, 88, 100))
    assert output.shape == (2, 128, 100)


def test_heatmap_encoder_supports_variable_height_without_temporal_resize() -> None:
    encoder = HeatmapEncoder()
    for temporal_width in (7, 8, 19):
        assert encoder(torch.randn(2, 3, 12, temporal_width)).shape == (2, 128, temporal_width)


def test_heatmap_encoder_gradients_reach_all_convolutional_blocks() -> None:
    encoder = HeatmapEncoder()
    output = encoder(torch.randn(2, 3, 88, 13, requires_grad=True))
    output.square().mean().backward()
    for module in (encoder.conv1, encoder.conv2, encoder.conv3):
        assert module.weight.grad is not None
        assert float(module.weight.grad.abs().sum()) > 0

