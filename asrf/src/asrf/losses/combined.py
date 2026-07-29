"""Combined multi-stage ASRF objective for round-3 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from asrf.models.model import ASRFOutput

from .boundary import BoundaryLossOutput, masked_boundary_bce
from .classification import CrossEntropyOutput, masked_class_weighted_cross_entropy
from .smoothing import SmoothingLossOutput, gs_tmse_loss


@dataclass(frozen=True)
class ASRFLossOutput:
    total_loss: Tensor
    asb_loss: Tensor
    asb_ce: Tensor
    asb_smoothing: Tensor
    brb_loss: Tensor
    per_stage_asb_ce: tuple[Tensor, ...]
    per_stage_asb_smoothing: tuple[Tensor, ...]
    per_stage_brb_loss: tuple[Tensor, ...]
    valid_frame_count: int
    valid_transition_count: int
    boundary_positive_count: int


def compute_asrf_loss(
    output: ASRFOutput,
    labels: Tensor,
    boundary_targets: Tensor,
    valid_mask: Tensor,
    *,
    class_weights: Tensor | None = None,
    boundary_positive_weight: float | Tensor | None = None,
    tau: float = 4.0,
    sigma: float = 1.0,
    smoothing_weight: float = 1.0,
    boundary_loss_weight: float = 0.1,
) -> ASRFLossOutput:
    """Compute CE + GS-TMSE per ASB stage and weighted BCE per BRB stage."""

    if len(output.asb_stage_logits) != len(output.brb_stage_logits):
        raise ValueError("ASB and BRB must expose the same number of stages.")
    if not output.asb_stage_logits:
        raise ValueError("At least one ASB/BRB stage is required.")
    ce_results: list[CrossEntropyOutput] = []
    smoothing_results: list[SmoothingLossOutput] = []
    boundary_results: list[BoundaryLossOutput] = []
    for logits in output.asb_stage_logits:
        ce_results.append(
            masked_class_weighted_cross_entropy(
                logits, labels, valid_mask, class_weights=class_weights
            )
        )
        smoothing_results.append(
            gs_tmse_loss(
                logits,
                output.encoder_features,
                valid_mask,
                tau=tau,
                sigma=sigma,
            )
        )
    for logits in output.brb_stage_logits:
        boundary_results.append(
            masked_boundary_bce(
                logits,
                boundary_targets,
                valid_mask,
                positive_weight=boundary_positive_weight,
            )
        )
    asb_ce = sum((item.loss for item in ce_results), output.encoder_features.sum() * 0.0) / len(ce_results)
    asb_smoothing = sum((item.loss for item in smoothing_results), output.encoder_features.sum() * 0.0) / len(smoothing_results)
    brb_loss = sum((item.loss for item in boundary_results), output.encoder_features.sum() * 0.0) / len(boundary_results)
    asb_loss = asb_ce + float(smoothing_weight) * asb_smoothing
    total_loss = asb_loss + float(boundary_loss_weight) * brb_loss
    boundary_positive_count = boundary_results[0].positive_count
    return ASRFLossOutput(
        total_loss=total_loss,
        asb_loss=asb_loss,
        asb_ce=asb_ce,
        asb_smoothing=asb_smoothing,
        brb_loss=brb_loss,
        per_stage_asb_ce=tuple(item.loss for item in ce_results),
        per_stage_asb_smoothing=tuple(item.loss for item in smoothing_results),
        per_stage_brb_loss=tuple(item.loss for item in boundary_results),
        valid_frame_count=int(sum(item.valid_frame_count for item in ce_results[:1])),
        valid_transition_count=int(sum(item.valid_transition_count for item in smoothing_results[:1])),
        boundary_positive_count=int(boundary_positive_count),
    )


class ASRFLoss(nn.Module):
    def __init__(
        self,
        *,
        class_weights: Tensor | None = None,
        boundary_positive_weight: float | Tensor | None = None,
        tau: float = 4.0,
        sigma: float = 1.0,
        smoothing_weight: float = 1.0,
        boundary_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if class_weights is None:
            self.class_weights = None  # type: ignore[assignment]
        else:
            self.register_buffer("class_weights", class_weights.detach().clone().float())
        self.boundary_positive_weight = boundary_positive_weight
        self.tau = float(tau)
        self.sigma = float(sigma)
        self.smoothing_weight = float(smoothing_weight)
        self.boundary_loss_weight = float(boundary_loss_weight)

    def forward(
        self, output: ASRFOutput, labels: Tensor, boundary_targets: Tensor, valid_mask: Tensor
    ) -> ASRFLossOutput:
        return compute_asrf_loss(
            output,
            labels,
            boundary_targets,
            valid_mask,
            class_weights=self.class_weights,
            boundary_positive_weight=self.boundary_positive_weight,
            tau=self.tau,
            sigma=self.sigma,
            smoothing_weight=self.smoothing_weight,
            boundary_loss_weight=self.boundary_loss_weight,
        )


__all__ = ["ASRFLoss", "ASRFLossOutput", "compute_asrf_loss"]
