"""Shared long-term feature extractor from the official ASRF design."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from .layers import (
    DEFAULT_DILATION_SCHEDULE,
    DilatedResidualStack,
    apply_temporal_mask,
)


class LongTermFeatureExtractor(nn.Module):
    """Project 128-channel encoder features into a 64-channel full-resolution stream.

    The projection is followed by ten non-causal dilated residual layers with
    dilation schedule 1,2,...,512 and dropout 0.5. No temporal pooling is
    performed.
    """

    def __init__(
        self,
        input_channels: int = 128,
        output_channels: int = 64,
        *,
        num_layers: int = 10,
        dilations: Sequence[int] = DEFAULT_DILATION_SCHEDULE,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers != len(tuple(dilations)):
            raise ValueError("num_layers must equal the dilation schedule length in this baseline.")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.conv_in = nn.Conv1d(input_channels, output_channels, kernel_size=1)
        self.layers = DilatedResidualStack(
            output_channels,
            dilations=dilations,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, features: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if features.ndim != 3 or features.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected [{features.shape[0] if features.ndim else 'B'},{self.input_channels},T], "
                f"got {tuple(features.shape)}."
            )
        output = apply_temporal_mask(self.conv_in(features), valid_mask)
        return self.layers(output, valid_mask=valid_mask)


__all__ = ["LongTermFeatureExtractor"]

