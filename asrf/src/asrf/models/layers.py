"""Official-style temporal layers used by the ASRF architecture."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_DILATION_SCHEDULE: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


@dataclass
class StageOutputs:
    """Logits and probabilities for every initial/refinement stage."""

    logits: list[Tensor]
    probabilities: list[Tensor]


def normalize_valid_mask(valid_mask: Tensor | None, *, batch_size: int, temporal_width: int, device: torch.device) -> Tensor:
    """Validate and return a boolean mask shaped ``[B,T]``."""
    if valid_mask is None:
        return torch.ones((batch_size, temporal_width), dtype=torch.bool, device=device)
    if valid_mask.ndim != 2 or tuple(valid_mask.shape) != (batch_size, temporal_width):
        raise ValueError(
            "valid_mask must have shape [B,T] matching the temporal input; "
            f"got {tuple(valid_mask.shape)} for {(batch_size, temporal_width)}."
        )
    return valid_mask.to(device=device, dtype=torch.bool)


def apply_temporal_mask(values: Tensor, valid_mask: Tensor | None) -> Tensor:
    """Zero invalid temporal positions without changing valid values."""
    if valid_mask is None:
        return values
    return values * valid_mask.to(dtype=values.dtype).unsqueeze(1)


class DilatedResidualLayer(nn.Module):
    """Non-causal dilated residual block matching the official implementation.

    The official block is ``Conv1d(k=3,dilation=d) -> ReLU -> Conv1d(k=1) ->
    Dropout(p=.5) -> residual add``.  A mask is applied before and after the
    block when supplied.  Convolutions near the valid/padded border can see
    zero-valued padded inputs, but invalid outputs are explicitly zeroed.
    """

    def __init__(
        self,
        channels: int,
        *,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd for symmetric padding.")
        if channels <= 0 or dilation <= 0:
            raise ValueError("channels and dilation must be positive.")
        padding = dilation * (kernel_size - 1) // 2
        self.dilation = int(dilation)
        self.kernel_size = int(kernel_size)
        self.conv_dilated = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv_in = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,T], got {tuple(x.shape)}.")
        masked_input = apply_temporal_mask(x, valid_mask)
        out = F.relu(self.conv_dilated(masked_input))
        out = self.conv_in(out)
        out = self.dropout(out)
        return apply_temporal_mask(masked_input + out, valid_mask)


class DilatedResidualStack(nn.Module):
    """A sequence of official-style residual layers at fixed channel width."""

    def __init__(
        self,
        channels: int,
        *,
        dilations: Sequence[int] = DEFAULT_DILATION_SCHEDULE,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dilations = tuple(int(dilation) for dilation in dilations)
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(
                    channels,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for dilation in self.dilations
            ]
        )

    def forward(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, valid_mask=valid_mask)
        return apply_temporal_mask(x, valid_mask)


class RefinementStage(nn.Module):
    """Official SingleStageTCN-style refinement stage returning logits."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        feature_channels: int = 64,
        dilations: Sequence[int] = DEFAULT_DILATION_SCHEDULE,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(input_channels, feature_channels, kernel_size=1)
        self.layers = DilatedResidualStack(
            feature_channels,
            dilations=dilations,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.conv_out = nn.Conv1d(feature_channels, output_channels, kernel_size=1)

    def forward(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        out = apply_temporal_mask(self.conv_in(x), valid_mask)
        out = self.layers(out, valid_mask=valid_mask)
        return apply_temporal_mask(self.conv_out(out), valid_mask)


def temporal_receptive_field(dilations: Sequence[int], kernel_size: int = 3) -> int:
    """Return the theoretical receptive field of a sequential temporal stack."""
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")
    return 1 + (kernel_size - 1) * sum(int(dilation) for dilation in dilations)


__all__ = [
    "DEFAULT_DILATION_SCHEDULE",
    "DilatedResidualLayer",
    "DilatedResidualStack",
    "RefinementStage",
    "StageOutputs",
    "apply_temporal_mask",
    "normalize_valid_mask",
    "temporal_receptive_field",
]
