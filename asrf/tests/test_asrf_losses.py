from __future__ import annotations

import torch

from asrf.losses.boundary import masked_boundary_bce
from asrf.losses.combined import compute_asrf_loss
from asrf.losses.smoothing import gs_tmse_loss, tmse_loss
from asrf.models import ASRFModel


def test_tmse_zero_for_constant_logits_and_positive_for_change() -> None:
    mask = torch.ones(1, 5, dtype=torch.bool)
    constant = torch.zeros(1, 3, 5)
    changing = constant.clone()
    changing[:, 0, 2:] = 4.0
    assert tmse_loss(constant, mask).loss.item() == 0.0
    assert tmse_loss(changing, mask).loss.item() > 0.0


def test_tmse_clamps_at_tau_squared() -> None:
    logits = torch.zeros(1, 2, 3)
    logits[:, 0, 1] = 100.0
    result = tmse_loss(logits, torch.ones(1, 3, dtype=torch.bool), tau=1.0)
    assert result.loss.item() <= 1.0 + 1e-6


def test_gstmse_reduces_penalty_at_strong_feature_change() -> None:
    logits = torch.zeros(1, 2, 4)
    logits[:, 0, 2:] = 3.0
    mask = torch.ones(1, 4, dtype=torch.bool)
    similar = torch.zeros(1, 3, 4)
    different = similar.clone()
    different[:, :, 2:] = 100.0
    similar_loss = gs_tmse_loss(logits, similar, mask).loss
    different_loss = gs_tmse_loss(logits, different, mask).loss
    assert similar_loss > different_loss
    assert torch.isfinite(different_loss)


def test_smoothing_excludes_invalid_adjacent_pairs_and_tiny_probabilities() -> None:
    logits = torch.tensor([[[1000.0, -1000.0, 1000.0], [-1000.0, 1000.0, -1000.0]]])
    mask = torch.tensor([[True, False, True]])
    result = tmse_loss(logits, mask)
    assert result.valid_transition_count == 0
    assert torch.isfinite(result.loss)


def test_brb_positive_weight_and_no_positive_batch_are_finite() -> None:
    logits = torch.zeros(1, 1, 4, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    unweighted = masked_boundary_bce(logits, targets, mask, positive_weight=1.0)
    weighted = masked_boundary_bce(logits, targets, mask, positive_weight=4.0)
    assert weighted.loss > unweighted.loss
    no_positive = masked_boundary_bce(logits, torch.zeros_like(targets), mask, positive_weight=4.0)
    assert torch.isfinite(no_positive.loss)
    no_positive.loss.backward()
    assert logits.grad is not None


def test_combined_loss_returns_all_stage_diagnostics() -> None:
    torch.manual_seed(4)
    model = ASRFModel(dropout=0.0)
    model.eval()
    heatmap = torch.randn(1, 3, 88, 9)
    mask = torch.ones(1, 9, dtype=torch.bool)
    output = model(heatmap, mask)
    labels = torch.randint(0, 7, (1, 9))
    boundaries = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    result = compute_asrf_loss(output, labels, boundaries, mask, boundary_positive_weight=3.0)
    assert len(result.per_stage_asb_ce) == 4
    assert len(result.per_stage_asb_smoothing) == 4
    assert len(result.per_stage_brb_loss) == 4
    assert torch.isfinite(result.total_loss)
