"""Temporal-width-preserving RGB CITR heatmap encoder.

This is an independent adaptation of the tested RGB heatmap encoder
architecture.  It is project-specific because official ASRF consumes
precomputed 2048-channel I3D features rather than RGB heatmaps.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .layers import normalize_valid_mask


class HeatmapEncoder(nn.Module):
    """Encode ``[B,3,H,T]`` heatmaps into ``[B,128,T]`` features.

    Layer sequence:

    * ``Conv2d(3,16,kernel=(5,5),padding=(2,2)) + BatchNorm + ReLU``;
    * height-only ``MaxPool2d(kernel=(2,1),stride=(2,1))``;
    * ``Conv2d(16,32,kernel=(3,5),padding=(1,2)) + BatchNorm + ReLU``;
    * a second height-only 2x1 max pool;
    * ``Conv2d(32,128,kernel=(3,5),padding=(1,2)) + BatchNorm + ReLU``;
    * mean over height only.

    The height becomes ``floor(floor(H/2)/2)`` (22 for H=88).  Every temporal
    kernel has symmetric padding and every pool has temporal kernel/stride 1,
    so T is preserved exactly for odd and even widths.
    """

    output_channels = 128

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5, 5), padding=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv3 = nn.Conv2d(32, self.output_channels, kernel_size=(3, 5), padding=(1, 2), bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channels)
        self.relu3 = nn.ReLU()

    def forward(self, heatmap: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if heatmap.ndim != 4:
            raise ValueError(f"HeatmapEncoder expects [B,3,H,T], got {tuple(heatmap.shape)}.")
        batch_size, channels, height, temporal_width = heatmap.shape
        if channels != 3:
            raise ValueError(f"HeatmapEncoder expects 3 channels, got {channels}.")
        if batch_size <= 0 or height < 4 or temporal_width <= 0:
            raise ValueError("Batch size and temporal width must be positive; height must be >= 4.")
        mask = normalize_valid_mask(
            valid_mask,
            batch_size=batch_size,
            temporal_width=temporal_width,
            device=heatmap.device,
        )
        mask_4d = mask.to(dtype=heatmap.dtype).unsqueeze(1).unsqueeze(2)
        heatmap = heatmap * mask_4d

        x = self.relu1(self.bn1(self.conv1(heatmap)))
        x = x * mask_4d
        self._assert_width(x, temporal_width, "conv1")
        x = self.pool1(x)
        x = x * mask_4d
        self._assert_width(x, temporal_width, "pool1")
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x * mask_4d
        self._assert_width(x, temporal_width, "conv2")
        x = self.pool2(x)
        x = x * mask_4d
        self._assert_width(x, temporal_width, "pool2")
        x = self.relu3(self.bn3(self.conv3(x)))
        x = x * mask_4d
        self._assert_width(x, temporal_width, "conv3")
        output = x.mean(dim=2)
        expected = (batch_size, self.output_channels, temporal_width)
        if tuple(output.shape) != expected:
            raise RuntimeError(f"Expected encoder output {expected}, got {tuple(output.shape)}.")
        return output

    @staticmethod
    def _assert_width(values: Tensor, expected_width: int, stage: str) -> None:
        if values.shape[-1] != expected_width:
            raise RuntimeError(f"Temporal width changed in {stage}: {values.shape[-1]} != {expected_width}.")


__all__ = ["HeatmapEncoder"]
