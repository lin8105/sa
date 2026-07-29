"""Top-level independent ASRF architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from torch import Tensor, nn

from .asb import ActionSegmentationBranch
from .brb import BoundaryRegressionBranch
from .feature_extractor import LongTermFeatureExtractor
from .heatmap_encoder import HeatmapEncoder
from .layers import DEFAULT_DILATION_SCHEDULE, apply_temporal_mask, normalize_valid_mask


@dataclass
class ASRFOutput:
    """All intermediate features and all four stage outputs."""

    encoder_features: Tensor
    shared_features: Tensor
    asb_stage_logits: list[Tensor]
    asb_stage_probabilities: list[Tensor]
    brb_stage_logits: list[Tensor]
    brb_stage_probabilities: list[Tensor]
    valid_mask: Tensor


class ASRFModel(nn.Module):
    """RGB-heatmap ASRF with the official probability-fed branch refinements."""

    def __init__(
        self,
        *,
        num_classes: int = 7,
        encoder_output_channels: int = 128,
        temporal_feature_channels: int = 64,
        num_temporal_layers: int = 10,
        dilation_schedule: Sequence[int] = DEFAULT_DILATION_SCHEDULE,
        kernel_size: int = 3,
        dropout: float = 0.5,
        causal: bool = False,
        asb_refinement_stages: int = 3,
        brb_refinement_stages: int = 3,
    ) -> None:
        super().__init__()
        if causal:
            raise ValueError("The round-2 baseline is non-causal, matching official ASRF.")
        if encoder_output_channels != 128 or temporal_feature_channels != 64:
            raise ValueError("Round-2 ASRF uses encoder=128 and temporal=64 channels.")
        dilations = tuple(int(dilation) for dilation in dilation_schedule)
        if num_temporal_layers != len(dilations):
            raise ValueError("num_temporal_layers must equal dilation_schedule length.")

        self.num_classes = int(num_classes)
        self.encoder = HeatmapEncoder()
        self.feature_extractor = LongTermFeatureExtractor(
            input_channels=encoder_output_channels,
            output_channels=temporal_feature_channels,
            num_layers=num_temporal_layers,
            dilations=dilations,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.asb = ActionSegmentationBranch(
            input_channels=temporal_feature_channels,
            num_classes=num_classes,
            feature_channels=temporal_feature_channels,
            num_layers=num_temporal_layers,
            dilations=dilations,
            kernel_size=kernel_size,
            dropout=dropout,
            refinement_stages=asb_refinement_stages,
        )
        self.brb = BoundaryRegressionBranch(
            input_channels=temporal_feature_channels,
            feature_channels=temporal_feature_channels,
            num_layers=num_temporal_layers,
            dilations=dilations,
            kernel_size=kernel_size,
            dropout=dropout,
            refinement_stages=brb_refinement_stages,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ASRFModel":
        """Build the round-2 model from the architecture YAML mapping."""
        data_config = config.get("data", {})
        model_config = config.get("model", {})
        return cls(
            num_classes=int(model_config.get("num_classes", data_config.get("num_classes", 7))),
            encoder_output_channels=int(model_config.get("encoder_output_channels", 128)),
            temporal_feature_channels=int(model_config.get("temporal_feature_channels", 64)),
            num_temporal_layers=int(model_config.get("num_temporal_layers", 10)),
            dilation_schedule=tuple(model_config.get("dilation_schedule", DEFAULT_DILATION_SCHEDULE)),
            kernel_size=int(model_config.get("kernel_size", 3)),
            dropout=float(model_config.get("dropout", 0.5)),
            causal=bool(model_config.get("causal", False)),
            asb_refinement_stages=int(model_config.get("asb_refinement_stages", 3)),
            brb_refinement_stages=int(model_config.get("brb_refinement_stages", 3)),
        )

    def forward(self, heatmap: Tensor, valid_mask: Tensor | None = None) -> ASRFOutput:
        if heatmap.ndim != 4 or heatmap.shape[1] != 3 or heatmap.shape[2] != 88:
            raise ValueError(f"ASRFModel expects [B,3,88,T], got {tuple(heatmap.shape)}.")
        batch_size, _, _, temporal_width = heatmap.shape
        mask = normalize_valid_mask(
            valid_mask,
            batch_size=batch_size,
            temporal_width=temporal_width,
            device=heatmap.device,
        )
        masked_heatmap = heatmap * mask.to(dtype=heatmap.dtype).unsqueeze(1).unsqueeze(2)
        encoder_features = apply_temporal_mask(
            self.encoder(masked_heatmap, valid_mask=mask), mask
        )
        shared_features = self.feature_extractor(encoder_features, valid_mask=mask)
        asb_outputs = self.asb(shared_features, valid_mask=mask)
        brb_outputs = self.brb(shared_features, valid_mask=mask)
        return ASRFOutput(
            encoder_features=encoder_features,
            shared_features=shared_features,
            asb_stage_logits=asb_outputs.logits,
            asb_stage_probabilities=asb_outputs.probabilities,
            brb_stage_logits=brb_outputs.logits,
            brb_stage_probabilities=brb_outputs.probabilities,
            valid_mask=mask,
        )


ASRF = ASRFModel

__all__ = ["ASRF", "ASRFModel", "ASRFOutput"]
