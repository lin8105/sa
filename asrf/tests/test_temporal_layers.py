from __future__ import annotations

import torch

from asrf.models.layers import (
    DEFAULT_DILATION_SCHEDULE,
    DilatedResidualLayer,
    RefinementStage,
    temporal_receptive_field,
)


def test_dilated_residual_layer_preserves_shape_and_masks_padding() -> None:
    layer = DilatedResidualLayer(64, dilation=8, dropout=0.5).eval()
    values = torch.randn(2, 64, 17)
    valid_mask = torch.tensor([[True] * 17, [True] * 9 + [False] * 8])
    output = layer(values, valid_mask=valid_mask)
    assert output.shape == values.shape
    assert torch.count_nonzero(output[1, :, 9:]) == 0


def test_temporal_layer_is_deterministic_in_eval_mode() -> None:
    layer = DilatedResidualLayer(8, dilation=2, dropout=0.5).eval()
    values = torch.randn(1, 8, 11)
    assert torch.equal(layer(values), layer(values))


def test_refinement_stage_preserves_width_and_has_gradients() -> None:
    stage = RefinementStage(7, 7, feature_channels=64).train()
    values = torch.randn(2, 7, 13, requires_grad=True)
    output = stage(values)
    assert output.shape == (2, 7, 13)
    output.square().mean().backward()
    assert float(stage.conv_in.weight.grad.abs().sum()) > 0
    assert float(stage.layers.layers[-1].conv_dilated.weight.grad.abs().sum()) > 0
    assert float(stage.conv_out.weight.grad.abs().sum()) > 0


def test_official_dilation_schedule_and_receptive_field() -> None:
    assert DEFAULT_DILATION_SCHEDULE == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    assert temporal_receptive_field(DEFAULT_DILATION_SCHEDULE, kernel_size=3) == 2047


def test_short_temporal_sequences_work_below_receptive_field() -> None:
    stage = RefinementStage(1, 1, feature_channels=64).eval()
    with torch.no_grad():
        assert stage(torch.randn(1, 1, 5)).shape == (1, 1, 5)

