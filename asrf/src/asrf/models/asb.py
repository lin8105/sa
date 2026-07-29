"""Action Segmentation Branch (ASB)."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from .layers import (
    DEFAULT_DILATION_SCHEDULE,
    RefinementStage,
    StageOutputs,
    apply_temporal_mask,
)


class ActionSegmentationBranch(nn.Module):
    """Initial class projection followed by three probability-fed refinements."""

    def __init__(
        self,
        input_channels: int = 64,
        num_classes: int = 7,
        *,
        feature_channels: int = 64,
        num_layers: int = 10,
        dilations: Sequence[int] = DEFAULT_DILATION_SCHEDULE,
        kernel_size: int = 3,
        dropout: float = 0.5,
        refinement_stages: int = 3,
    ) -> None:
        super().__init__()
        if refinement_stages < 0:
            raise ValueError("refinement_stages must be non-negative.")
        if num_layers != len(tuple(dilations)):
            raise ValueError("num_layers must equal the dilation schedule length in this baseline.")
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.initial_projection = nn.Conv1d(input_channels, num_classes, kernel_size=1)
        self.refinement_stages = nn.ModuleList(
            [
                RefinementStage(
                    num_classes,
                    num_classes,
                    feature_channels=feature_channels,
                    dilations=dilations,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(refinement_stages)
            ]
        )

    def forward(self, shared_features: Tensor, valid_mask: Tensor | None = None) -> StageOutputs:
        if shared_features.ndim != 3 or shared_features.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected [B,{self.input_channels},T], got {tuple(shared_features.shape)}."
            )
        logits: list[Tensor] = []
        probabilities: list[Tensor] = []

        current_logits = apply_temporal_mask(self.initial_projection(shared_features), valid_mask)
        current_probabilities = apply_temporal_mask(
            current_logits.softmax(dim=1), valid_mask
        )
        logits.append(current_logits)
        probabilities.append(current_probabilities)

        # Official ASRF feeds class probabilities, not logits, to each later stage.
        for stage in self.refinement_stages:
            current_logits = stage(current_probabilities, valid_mask=valid_mask)
            current_probabilities = apply_temporal_mask(
                current_logits.softmax(dim=1), valid_mask
            )
            logits.append(current_logits)
            probabilities.append(current_probabilities)

        return StageOutputs(logits=logits, probabilities=probabilities)


ASB = ActionSegmentationBranch

__all__ = ["ASB", "ActionSegmentationBranch"]
