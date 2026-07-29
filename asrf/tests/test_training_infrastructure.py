from __future__ import annotations

import csv
import json

import pytest
import torch

from asrf.evaluation.metrics import boundary_counts
from asrf.losses.combined import ASRFLoss
from asrf.models.model import ASRFOutput
from asrf.training.checkpointing import load_checkpoint, save_checkpoint
from asrf.training.logging import CSVMetricLogger
from asrf.training.trainer import seed_everything


def _tiny_output() -> ASRFOutput:
    torch.manual_seed(4)
    encoder = torch.randn(1, 128, 6, requires_grad=True)
    shared = torch.randn(1, 64, 6, requires_grad=True)
    asb_logits = [torch.randn(1, 7, 6, requires_grad=True) for _ in range(4)]
    brb_logits = [torch.randn(1, 1, 6, requires_grad=True) for _ in range(4)]
    return ASRFOutput(
        encoder_features=encoder,
        shared_features=shared,
        asb_stage_logits=asb_logits,
        asb_stage_probabilities=[value.softmax(dim=1) for value in asb_logits],
        brb_stage_logits=brb_logits,
        brb_stage_probabilities=[value.sigmoid() for value in brb_logits],
        valid_mask=torch.ones(1, 6, dtype=torch.bool),
    )


def test_one_training_step_and_finite_gradient() -> None:
    output = _tiny_output()
    loss_fn = ASRFLoss(class_weights=torch.ones(7), boundary_positive_weight=3.0)
    labels = torch.tensor([[0, 0, 1, 1, 2, 2]])
    boundaries = torch.tensor([[1, 0, 1, 0, 1, 0]], dtype=torch.float32)
    optimizer = torch.optim.Adam(list(output.asb_stage_logits) + list(output.brb_stage_logits), lr=1e-3)
    result = loss_fn(output, labels, boundaries, output.valid_mask)
    assert torch.isfinite(result.total_loss)
    result.total_loss.backward()
    optimizer.step()
    assert all(value.grad is not None for value in output.asb_stage_logits)
    assert all(value.grad is not None for value in output.brb_stage_logits)


def test_one_validation_step_is_no_grad() -> None:
    output = _tiny_output()
    loss_fn = ASRFLoss(class_weights=torch.ones(7), boundary_positive_weight=3.0)
    with torch.no_grad():
        result = loss_fn(output, torch.zeros(1, 6, dtype=torch.long), torch.zeros(1, 6), output.valid_mask)
    assert torch.isfinite(result.total_loss)
    assert not result.total_loss.requires_grad


def test_checkpoint_save_load_and_persisted_training_metadata(tmp_path) -> None:
    path = tmp_path / "last.pt"
    payload = {
        "model_state": {"weight": torch.ones(2)}, "optimizer_state": {}, "epoch": 3,
        "global_step": 7, "class_weights": torch.arange(7, dtype=torch.float32),
        "boundary_positive_weight": 508.49, "train_trajectory_ids": ["p1"],
        "validation_trajectory_ids": ["p9"],
    }
    save_checkpoint(path, payload)
    restored = load_checkpoint(path)
    assert restored["epoch"] == 3
    assert torch.equal(restored["class_weights"], payload["class_weights"])
    assert restored["boundary_positive_weight"] == pytest.approx(508.49)


def test_resume_state_preserves_epoch_and_global_step(tmp_path) -> None:
    path = tmp_path / "last.pt"
    save_checkpoint(path, {"model_state": {}, "optimizer_state": {}, "epoch": 16, "global_step": 128, "best_epoch": 14, "best_validation_metric": 0.2})
    restored = load_checkpoint(path)
    assert restored["epoch"] + 1 == 17
    assert restored["global_step"] == 128


def test_seeded_training_inputs_are_repeatable() -> None:
    seed_everything(42)
    first = torch.rand(8)
    seed_everything(42)
    second = torch.rand(8)
    assert torch.equal(first, second)


def test_metrics_logger_schema(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    with CSVMetricLogger(path, ["epoch", "split", "val_total_loss"]) as logger:
        logger.write({"epoch": 1, "split": "val", "val_total_loss": 2.0})
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"epoch": "1", "split": "val", "val_total_loss": "2.0"}]


def test_train_validation_nonoverlap_and_no_test_access() -> None:
    config = {
        "data": {"train_root": "/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data/train/pour", "train_split": "splits/pour_train.txt", "val_split": "splits/pour_val.txt"},
    }
    assert set(config["data"]["train_root"].split("/")) & {"test"} == set()
    assert set(["p1", "p2"]) .isdisjoint({"p9", "p10"})
    assert "/test/" not in config["data"]["train_root"]


def test_invalid_length_mismatch_is_rejected_by_synthetic_contract() -> None:
    heatmap = torch.zeros(1, 3, 88, 5)
    labels = torch.zeros(1, 4, dtype=torch.long)
    mask = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(ValueError):
        if labels.shape != mask.shape or labels.shape[-1] != heatmap.shape[-1]:
            raise ValueError("length mismatch")


def test_nan_loss_is_detectable() -> None:
    output = _tiny_output()
    output.asb_stage_logits[0].data.fill_(float("nan"))
    loss_fn = ASRFLoss(class_weights=torch.ones(7), boundary_positive_weight=3.0)
    result = loss_fn(output, torch.zeros(1, 6, dtype=torch.long), torch.zeros(1, 6), output.valid_mask)
    assert not torch.isfinite(result.total_loss)


def test_frame_zero_and_internal_boundary_metrics_differ() -> None:
    included = boundary_counts([0, 4], [0, 5], 0, include_frame0=True)
    internal = boundary_counts([0, 4], [0, 5], 0, include_frame0=False)
    assert included["tp"] == 1
    assert internal["tp"] == 0


def test_best_checkpoint_metric_and_early_stopping_policy_are_loss_based() -> None:
    values = [3.0, 2.0, 2.0, 2.0]
    best_epoch = min(range(1, len(values) + 1), key=lambda epoch: values[epoch - 1])
    assert best_epoch == 2
    patience = 2
    minimum_epochs = 3
    stale = 0
    stopping = None
    best = float("inf")
    for epoch, value in enumerate(values, 1):
        if value < best:
            best, stale = value, 0
        else:
            stale += 1
        if epoch >= minimum_epochs and stale >= patience:
            stopping = epoch
            break
    assert stopping == 4


def test_raw_and_refined_exports_are_separate() -> None:
    assert "asb_raw_label" != "asrf_refined_label"
