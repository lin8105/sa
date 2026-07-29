"""Deterministic CPU trainer for the strict pour-only ASRF baseline."""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from asrf.data.collate import collate_fn
from asrf.data.dataset import MultiTaskTrajectoryDataset, TrajectoryDataset
from asrf.data.labels import LabelMapping, load_label_mapping
from asrf.evaluation.metrics import (
    aggregate_trajectory_metrics,
    boundary_counts,
    boundary_indices_from_labels,
    trajectory_metrics,
)
from asrf.losses.classification import (
    TrainingStatistics,
    collect_statistics_for_entries,
    collect_training_statistics,
)
from asrf.losses.combined import ASRFLoss, ASRFLossOutput
from asrf.models import ASRFModel
from asrf.refinement.refine import ASRFRefinementOutput, refine_asrf_predictions
from asrf.refinement.segments import TemporalInterval
from asrf.utils.config import PROJECT_ROOT, load_yaml_config, resolve_repo_path

from .checkpointing import checkpoint_manifest, load_checkpoint, save_checkpoint, sha256_file
from asrf.data.ontology import metadata_for_mapping
from .logging import CSVMetricLogger, append_log
from .transfer import expand_asrf_state_dict, project_asrf_state_dict


OFFICIAL_COMMIT = "9623f1e8d9a1171333a4eeb65d190997b6c44a95"
TOLERANCES = (10, 20, 30, 33)


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _git_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "no_commit_HEAD"


def _as_float(value: Tensor | float | int) -> float:
    return float(value.detach().cpu().item()) if isinstance(value, Tensor) else float(value)


def _gradient_norm(model: nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().pow(2).sum().cpu())
    return math.sqrt(squared)


def _component_gradient_norm(model: nn.Module, prefix: str) -> float:
    squared = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith(prefix) and parameter.grad is not None:
            squared += float(parameter.grad.detach().pow(2).sum().cpu())
    return math.sqrt(squared)


def _boundary_contributions(logits: Tensor, targets: Tensor, mask: Tensor, positive_weight: float) -> tuple[float, float]:
    logits = logits[:, 0] if logits.ndim == 3 else logits
    valid_targets = targets[mask].to(dtype=logits.dtype)
    valid_logits = logits[mask]
    if valid_logits.numel() == 0:
        return 0.0, 0.0
    positive = valid_targets > 0.5
    raw = torch.nn.functional.binary_cross_entropy_with_logits(valid_logits, valid_targets, reduction="none")
    pos_value = float((raw[positive] * positive_weight).mean().detach().cpu()) if positive.any() else 0.0
    neg_value = float(raw[~positive].mean().detach().cpu()) if (~positive).any() else 0.0
    return pos_value, neg_value


def _label_names(mapping: LabelMapping) -> dict[int, str]:
    return {int(value): str(name) for name, value in mapping.items()}


def _stats_to_dict(stats: TrainingStatistics, mapping: LabelMapping) -> dict[str, Any]:
    names = sorted(mapping, key=mapping.get)
    return {
        "class_counts": {name: int(stats.class_counts[mapping[name]]) for name in names},
        "class_frequencies": {name: float(stats.class_frequencies[mapping[name]]) for name in names},
        "median_frequency": float(stats.median_frequency),
        "class_weights": {name: float(stats.class_weights[mapping[name]]) for name in names},
        "segment_counts": {name: int(stats.segment_counts[mapping[name]]) for name in names},
        "total_valid_frames": stats.total_valid_frames,
        "boundary_positive_count": stats.boundary_positive_count,
        "boundary_negative_count": stats.boundary_negative_count,
        "boundary_positive_ratio": stats.boundary_positive_ratio,
        "boundary_positive_weight": stats.boundary_positive_weight,
        "boundary_positive_mass": float(stats.boundary_positive_mass),
        "boundary_negative_mass": float(stats.boundary_negative_mass),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _stage_key(prefix: str, stage: int, metric: str) -> str:
    return f"{prefix}_stage{stage + 1}_{metric}"


class ASRFTrainer:
    """Own datasets, model, objective, checkpoints, and epoch diagnostics."""

    def __init__(self, config: dict[str, Any], *, device: str | torch.device | None = None, resume: str | Path | None = None) -> None:
        self.config = config
        self.seed = int(config["experiment"]["seed"])
        training = config["training"]
        seed_everything(self.seed, deterministic=bool(training.get("deterministic", True)))
        requested_device = device or training.get("device", "cpu")
        self.device = torch.device(requested_device)
        if self.device.type != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {self.device}, but CUDA is unavailable.")

        data = config["data"]
        self.multitask = "dataset_root" in data
        self.boundary_target_config = {
            key: data[key] for key in (
                "boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma",
                "boundary_include_frame_zero", "boundary_include_final_frame",
            ) if key in data
        }
        self.allow_zero_class_weights = bool(data.get("allow_zero_class_weights", False))
        self.train_root = Path(data.get("train_root", data.get("dataset_root", "")))
        if not self.train_root:
            raise ValueError("Configuration must provide data.train_root or data.dataset_root.")
        self.train_split = resolve_repo_path(data["train_split"])
        self.val_split = resolve_repo_path(data["val_split"])
        self.label_path = resolve_repo_path(data["label_config"])
        self.mapping = load_label_mapping(self.label_path)
        if len(self.mapping) != int(config["model"]["num_classes"]):
            raise ValueError("The configured label map and model class count disagree.")
        if self.multitask:
            self.train_root = self.train_root.resolve()
            self.train_dataset = MultiTaskTrajectoryDataset(
                self.train_root, self.train_split, self.label_path,
                expected_height=int(data["heatmap_height"]), allow_test=False,
                boundary_target_config=self.boundary_target_config,
            )
            self.val_dataset = MultiTaskTrajectoryDataset(
                self.train_root, self.val_split, self.label_path,
                expected_height=int(data["heatmap_height"]), allow_test=False,
                boundary_target_config=self.boundary_target_config,
            )
        else:
            self.train_dataset = TrajectoryDataset(self.train_root, self.train_split, self.label_path, expected_height=int(data["heatmap_height"]), boundary_target_config=self.boundary_target_config)
            self.val_dataset = TrajectoryDataset(self.train_root, self.val_split, self.label_path, expected_height=int(data["heatmap_height"]), boundary_target_config=self.boundary_target_config)
        train_ids = set(self.train_dataset.trajectory_ids)
        val_ids = set(self.val_dataset.trajectory_ids)
        overlap = sorted(train_ids & val_ids)
        if overlap:
            raise ValueError(f"Train/validation split overlap: {overlap}")
        if any("test" in str(path).lower() for path in (self.train_root, self.train_split, self.val_split)):
            raise ValueError("Pour baseline configuration may not reference test paths.")
        self.train_loader = DataLoader(self.train_dataset, batch_size=int(data["batch_size"]), shuffle=True, num_workers=int(data["num_workers"]), collate_fn=collate_fn, generator=torch.Generator().manual_seed(self.seed))
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

        if self.multitask:
            self.train_stats = collect_statistics_for_entries(self.train_root, self.train_split, self.mapping, self.boundary_target_config, self.allow_zero_class_weights)
            self.val_stats = collect_statistics_for_entries(self.train_root, self.val_split, self.mapping, self.boundary_target_config, self.allow_zero_class_weights)
        else:
            self.train_stats = collect_training_statistics(self.train_root, self.train_split, self.mapping, self.boundary_target_config, self.allow_zero_class_weights)
            self.val_stats = collect_training_statistics(self.train_root, self.val_split, self.mapping, self.boundary_target_config, self.allow_zero_class_weights)
        model_config = dict(config)
        model_config.setdefault("data", {})["num_classes"] = int(config["model"]["num_classes"])
        self.model = ASRFModel.from_config(model_config).to(self.device)
        self.initialization_metadata: dict[str, Any] = {"mode": "random_initialization"}
        initialization_path = training.get("initialize_from_checkpoint")
        if initialization_path:
            checkpoint_path = resolve_repo_path(initialization_path)
            payload = load_checkpoint(checkpoint_path, map_location=self.device, expected_ontology=True)
            row_mapping = training.get("initialize_class_row_mapping")
            if row_mapping is not None:
                expanded_state, transfer_metadata = project_asrf_state_dict(
                    payload["model_state"], self.model.state_dict(),
                    source_class_rows={int(dest): int(src) for dest, src in row_mapping.items()},
                )
            else:
                expanded_state, transfer_metadata = expand_asrf_state_dict(payload["model_state"], self.model.state_dict())
            self.model.load_state_dict(expanded_state, strict=True)
            self.initialization_metadata = {
                "mode": "expanded_round8_checkpoint",
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                **transfer_metadata,
            }
        loss_config = config["loss"]
        self.criterion = ASRFLoss(
            class_weights=self.train_stats.class_weights.float(),
            boundary_positive_weight=self._configured_boundary_positive_weight(loss_config),
            tau=float(loss_config["tau"]), sigma=float(loss_config["sigma"]),
            smoothing_weight=float(loss_config["smoothing_weight"]),
            boundary_loss_weight=float(loss_config["boundary_loss_weight"]),
        )
        if str(training["optimizer"]).lower() != "adam":
            raise ValueError("This baseline trainer only implements the configured Adam optimizer.")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
        self.output_dir = resolve_repo_path(config["paths"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.best_metric = float("inf")
        self.best_epoch = 0
        self.start_epoch = 1
        self.history: list[dict[str, Any]] = []
        self.task_history: list[dict[str, Any]] = []
        self.class_history: list[dict[str, Any]] = []
        self.resume_path = Path(resume) if resume else None
        if self.resume_path:
            self._resume(self.resume_path)
            self._load_existing_history()
        self._write_static_artifacts()

    def _configured_boundary_positive_weight(self, loss_config: dict[str, Any]) -> float:
        """Resolve the BRB positive weight without changing the official default."""
        weighting = str(loss_config.get("boundary_positive_weighting", "reciprocal_frequency"))
        if weighting in {"reciprocal_frequency", "reciprocal_positive_ratio"}:
            return float(self.train_stats.boundary_positive_weight)
        if weighting in {"none", "unit"}:
            return 1.0
        if weighting == "fixed":
            if "boundary_positive_weight" not in loss_config:
                raise ValueError("fixed boundary weighting requires loss.boundary_positive_weight.")
            value = float(loss_config["boundary_positive_weight"])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("loss.boundary_positive_weight must be a finite positive number.")
            return value
        raise ValueError(f"Unsupported boundary_positive_weighting: {weighting!r}")

    def _write_static_artifacts(self) -> None:
        import yaml

        (self.output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        (self.output_dir / "class_statistics.json").write_text(json.dumps(_stats_to_dict(self.train_stats, self.mapping), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        boundary = {
            "train_split": _stats_to_dict(self.train_stats, self.mapping),
            "validation_split": _stats_to_dict(self.val_stats, self.mapping),
            "frame0_included": True,
            "internal_only_positive_count_train": max(0, self.train_stats.boundary_positive_count - len(self.train_dataset)),
            "configured_positive_weight_train": float(self.criterion.boundary_positive_weight),
            "reciprocal_positive_weight_train": self.train_stats.boundary_positive_weight,
            "target_config": self.boundary_target_config,
            "positive_target_mass_train": float(self.train_stats.boundary_positive_mass),
            "negative_target_mass_train": float(self.train_stats.boundary_negative_mass),
        }
        (self.output_dir / "boundary_statistics.json").write_text(json.dumps(boundary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _resume(self, path: Path) -> None:
        payload = load_checkpoint(path, map_location=self.device, expected_ontology=True)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.global_step = int(payload.get("global_step", 0))
        self.best_metric = float(payload.get("best_validation_metric", float("inf")))
        self.best_epoch = int(payload.get("best_epoch", 0))
        self.start_epoch = int(payload["epoch"]) + 1

    def _load_existing_history(self) -> None:
        path = self.output_dir / "metrics.csv"
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                converted: dict[str, Any] = {"split": row.get("split", "")}
                for key, value in row.items():
                    if key == "split" or value in (None, ""):
                        continue
                    try:
                        converted[key] = float(value)
                        if key == "epoch":
                            converted[key] = int(float(value))
                    except ValueError:
                        converted[key] = value
                self.history.append(converted)

    def _batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        result = dict(batch)
        for key in ("heatmap", "labels", "boundary_targets", "hard_boundary_targets", "valid_mask", "lengths"):
            result[key] = result[key].to(self.device)
        return result

    def _validate_batch(self, batch: dict[str, Any]) -> None:
        heatmap, labels, targets, hard_targets, mask = batch["heatmap"], batch["labels"], batch["boundary_targets"], batch["hard_boundary_targets"], batch["valid_mask"]
        if heatmap.ndim != 4 or heatmap.shape[1:3] != (3, 88):
            raise ValueError(f"Invalid batch heatmap shape {tuple(heatmap.shape)}")
        if labels.shape != mask.shape or targets.shape != mask.shape or hard_targets.shape != mask.shape or labels.shape[-1] != heatmap.shape[-1]:
            raise ValueError("Heatmap, labels, boundary targets, and mask have mismatched temporal shapes.")
        for index, length in enumerate(batch["lengths"].tolist()):
            if not 0 <= int(length) <= heatmap.shape[-1]:
                raise ValueError("Invalid collated length.")
            if not mask[index, :length].all() or mask[index, length:].any():
                raise ValueError("Mask must be a right-padded prefix.")
            if length and not torch.all((labels[index, :length] >= 0) & (labels[index, :length] < len(self.mapping))):
                raise ValueError("Valid labels must be in the configured class range.")
            if labels[index, length:].ne(-100).any():
                raise ValueError("Padded labels must use -100.")
            if targets[index, length:].ne(0).any():
                raise ValueError("Padded boundary targets must be zero.")
            if hard_targets[index, length:].ne(0).any():
                raise ValueError("Padded hard boundary targets must be zero.")

    def _run_epoch(self, epoch: int, *, training: bool) -> dict[str, Any]:
        self.model.train(training)
        loader = self.train_loader if training else self.val_loader
        loss_values: defaultdict[str, list[float]] = defaultdict(list)
        raw_rows: list[dict[str, float | int]] = []
        refined_rows: list[dict[str, float | int]] = []
        trajectory_rows: list[dict[str, Any]] = []
        tolerance_counts: dict[str, dict[str, int]] = {f"{tol}_{scope}": {"tp": 0, "fp": 0, "fn": 0, "predicted_count": 0, "target_count": 0} for tol in TOLERANCES for scope in ("including_frame0", "internal_only")}
        stage_rows: dict[str, list[dict[str, float | int]]] = defaultdict(list)
        stage_boundary_counts: dict[str, list[dict[str, int | float]]] = defaultdict(list)
        grad_norms: list[float] = []
        brb_grad_norms: list[float] = []
        brb_pos_contrib: list[float] = []
        brb_neg_contrib: list[float] = []
        brb_logit_max: list[float] = []
        brb_probability_means: list[float] = []
        predicted_boundary_total = 0
        sample_count = 0
        task_raw_rows: defaultdict[str, list[dict[str, float | int]]] = defaultdict(list)
        task_refined_rows: defaultdict[str, list[dict[str, float | int]]] = defaultdict(list)
        task_boundary_counts: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "predicted_count": 0, "target_count": 0}
        )
        class_confusion: defaultdict[str, defaultdict[str, defaultdict[int, dict[str, int]]]] = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
                )
            )
        )

        for batch in loader:
            batch = self._batch_to_device(batch)
            self._validate_batch(batch)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                output = self.model(batch["heatmap"], valid_mask=batch["valid_mask"])
                loss_output: ASRFLossOutput = self.criterion(output, batch["labels"], batch["boundary_targets"], batch["valid_mask"])
                if not torch.isfinite(loss_output.total_loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {self.global_step}.")
                if training:
                    loss_output.total_loss.backward()
                    gradient_norm = _gradient_norm(self.model)
                    brb_gradient_norm = _component_gradient_norm(self.model, "brb")
                    if not math.isfinite(gradient_norm) or not math.isfinite(brb_gradient_norm):
                        raise FloatingPointError(f"Non-finite gradient at epoch {epoch}, step {self.global_step}.")
                    clip_value = self.config["training"].get("gradient_clip_norm")
                    if clip_value is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip_value))
                    self.optimizer.step()
                    self.global_step += 1
                else:
                    gradient_norm = 0.0
                    brb_gradient_norm = 0.0

            for key, value in {
                "total_loss": loss_output.total_loss, "asb_loss": loss_output.asb_loss,
                "asb_ce": loss_output.asb_ce, "asb_smoothing": loss_output.asb_smoothing,
                "brb_loss": loss_output.brb_loss,
            }.items():
                loss_values[key].append(_as_float(value))
            for stage, value in enumerate(loss_output.per_stage_asb_ce):
                loss_values[_stage_key("asb", stage, "ce")].append(_as_float(value))
            for stage, value in enumerate(loss_output.per_stage_asb_smoothing):
                loss_values[_stage_key("asb", stage, "smoothing")].append(_as_float(value))
            for stage, value in enumerate(loss_output.per_stage_brb_loss):
                loss_values[_stage_key("brb", stage, "loss")].append(_as_float(value))

            final_brb_logits = output.brb_stage_logits[-1]
            for index, length_value in enumerate(batch["lengths"].tolist()):
                length = int(length_value)
                mask = batch["valid_mask"][index, :length]
                truth = batch["labels"][index, :length]
                # Metrics/refinement are diagnostic and must not retain the
                # training graph or backpropagate through argmax/voting.
                final_asb_prob = output.asb_stage_probabilities[-1][index:index + 1, :, :length].detach()
                final_brb_prob = output.brb_stage_probabilities[-1][index:index + 1, :, :length].detach()
                refinement: ASRFRefinementOutput = refine_asrf_predictions(final_asb_prob, final_brb_prob, mask.unsqueeze(0), threshold=float(self.config["refinement"]["boundary_threshold"]), voting=str(self.config["refinement"]["voting"]))
                raw = refinement.raw_labels[0, :length]
                refined = refinement.refined_labels[0, :length]
                raw_metric = trajectory_metrics(raw, truth)
                refined_metric = trajectory_metrics(refined, truth)
                raw_rows.append(raw_metric)
                refined_rows.append(refined_metric)
                task_name = str(batch.get("task_names", ["unknown"])[index])
                task_raw_rows[task_name].append(raw_metric)
                task_refined_rows[task_name].append(refined_metric)
                for variant, prediction in (("raw", raw), ("refined", refined)):
                    for class_id in range(len(self.mapping)):
                        predicted_class = prediction == class_id
                        target_class = truth == class_id
                        class_confusion[task_name][variant][class_id]["tp"] += int((predicted_class & target_class).sum())
                        class_confusion[task_name][variant][class_id]["fp"] += int((predicted_class & ~target_class).sum())
                        class_confusion[task_name][variant][class_id]["fn"] += int((~predicted_class & target_class).sum())
                        class_confusion[task_name][variant][class_id]["support"] += int(target_class.sum())
                truth_boundaries = torch.where(batch["hard_boundary_targets"][index, :length] > 0.5)[0].tolist()
                selected = list(refinement.selected_boundaries[0])
                predicted_boundary_total += len(selected)
                for tolerance in TOLERANCES:
                    for scope, include_frame0 in (("including_frame0", True), ("internal_only", False)):
                        result = boundary_counts(selected, truth_boundaries, tolerance, include_frame0=include_frame0)
                        target_counter = tolerance_counts[f"{tolerance}_{scope}"]
                        for key in ("tp", "fp", "fn", "predicted_count", "target_count"):
                            target_counter[key] += int(result[key])
                        if tolerance == 33 and scope == "internal_only":
                            for key in ("tp", "fp", "fn", "predicted_count", "target_count"):
                                task_boundary_counts[task_name][key] += int(result[key])
                for stage, probs in enumerate(output.asb_stage_probabilities):
                    stage_prediction = probs[index, :, :length].argmax(dim=0)
                    stage_rows[_stage_key("asb", stage, "metrics")].append(trajectory_metrics(stage_prediction, truth))
                for stage, probs in enumerate(output.brb_stage_probabilities):
                    peaks = refinement.selected_boundaries[0] if stage == len(output.brb_stage_probabilities) - 1 else self._select_stage_peaks(probs[index, 0, :length])
                    stage_boundary_counts[_stage_key("brb", stage, "boundary")].append(boundary_counts(peaks, truth_boundaries, 33, include_frame0=False))
                position_mask = batch["valid_mask"][index:index + 1]
                pos, neg = _boundary_contributions(final_brb_logits[index:index + 1], batch["boundary_targets"][index:index + 1], position_mask, float(self.criterion.boundary_positive_weight))
                brb_pos_contrib.append(pos)
                brb_neg_contrib.append(neg)
                brb_logit_max.append(float(final_brb_logits[index, 0, :length].detach().abs().max().cpu()))
                brb_probability_means.append(float(final_brb_prob[0, 0].detach().mean().cpu()))
                trajectory_rows.append({
                    "trajectory_id": batch["trajectory_ids"][index], "loss": _as_float(loss_output.total_loss),
                    "raw": raw_metric, "refined": refined_metric, "selected_boundaries": selected,
                    "truth_boundaries": truth_boundaries, "false_boundary_count": int(boundary_counts(selected, truth_boundaries, 33, include_frame0=False)["fp"]),
                    "missed_boundary_count": int(boundary_counts(selected, truth_boundaries, 33, include_frame0=False)["fn"]),
                    "stage_asb": {str(stage + 1): stage_rows[_stage_key("asb", stage, "metrics")][-1] for stage in range(len(output.asb_stage_probabilities))},
                })
                sample_count += 1
            grad_norms.append(gradient_norm)
            brb_grad_norms.append(brb_gradient_norm)

        metrics: dict[str, Any] = {key: _mean(values) for key, values in loss_values.items()}
        metrics.update({f"raw_{key}": value for key, value in aggregate_trajectory_metrics(raw_rows).items()})
        metrics.update({f"refined_{key}": value for key, value in aggregate_trajectory_metrics(refined_rows).items()})
        metrics["trajectory_count"] = sample_count
        metrics["predicted_boundary_count"] = predicted_boundary_total
        metrics["predicted_boundary_count_mean"] = predicted_boundary_total / sample_count if sample_count else 0.0
        metrics["gradient_norm"] = _mean(grad_norms)
        metrics["brb_gradient_norm"] = _mean(brb_grad_norms)
        metrics["brb_positive_contribution"] = _mean(brb_pos_contrib)
        metrics["brb_negative_contribution"] = _mean(brb_neg_contrib)
        metrics["brb_logit_abs_max"] = max(brb_logit_max, default=0.0)
        metrics["brb_mean_probability"] = _mean(brb_probability_means)
        for key, counter in tolerance_counts.items():
            denominator_p = counter["tp"] + counter["fp"]
            denominator_r = counter["tp"] + counter["fn"]
            precision = counter["tp"] / denominator_p if denominator_p else 0.0
            recall = counter["tp"] / denominator_r if denominator_r else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            metrics[f"boundary_{key}_precision"] = precision
            metrics[f"boundary_{key}_recall"] = recall
            metrics[f"boundary_{key}_f1"] = f1
            metrics[f"boundary_{key}_tp"] = counter["tp"]
            metrics[f"boundary_{key}_fp"] = counter["fp"]
            metrics[f"boundary_{key}_fn"] = counter["fn"]
        for key, rows in stage_rows.items():
            metrics.update({f"{key}_{metric}": value for metric, value in aggregate_trajectory_metrics(rows).items()})
        for key, rows in stage_boundary_counts.items():
            metrics[f"{key}_f1"] = _mean([float(row["f1"]) for row in rows])
            metrics[f"{key}_precision"] = _mean([float(row["precision"]) for row in rows])
            metrics[f"{key}_recall"] = _mean([float(row["recall"]) for row in rows])
            metrics[f"{key}_peak_count"] = _mean([float(row["predicted_count"]) for row in rows])
            metrics[f"{key}_mean_probability"] = metrics["brb_mean_probability"]
        task_metrics: dict[str, dict[str, Any]] = {}
        for task_name in sorted(set(task_raw_rows) | set(task_refined_rows)):
            boundary = task_boundary_counts[task_name]
            precision = boundary["tp"] / (boundary["tp"] + boundary["fp"]) if boundary["tp"] + boundary["fp"] else 0.0
            recall = boundary["tp"] / (boundary["tp"] + boundary["fn"]) if boundary["tp"] + boundary["fn"] else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            task_metrics[task_name] = {
                "trajectory_count": len(task_raw_rows[task_name]),
                "raw": aggregate_trajectory_metrics(task_raw_rows[task_name]),
                "refined": aggregate_trajectory_metrics(task_refined_rows[task_name]),
                "internal_boundary_f1@33": f1,
                "internal_boundary_precision@33": precision,
                "internal_boundary_recall@33": recall,
                "predicted_boundary_count": boundary["predicted_count"],
            }
            safe_task = task_name.replace(" ", "_").replace("/", "_")
            metrics[f"task_{safe_task}_raw_frame_accuracy"] = task_metrics[task_name]["raw"]["frame_accuracy"]
            metrics[f"task_{safe_task}_refined_frame_accuracy"] = task_metrics[task_name]["refined"]["frame_accuracy"]
            metrics[f"task_{safe_task}_internal_boundary_f1@33"] = f1

        class_metrics: list[dict[str, Any]] = []
        names = _label_names(self.mapping)
        for task_name in sorted(class_confusion):
            for variant in ("raw", "refined"):
                for class_id in range(len(self.mapping)):
                    counts = class_confusion[task_name][variant][class_id]
                    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
                    recall = counts["tp"] / counts["support"] if counts["support"] else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                    class_metrics.append({
                        "task": task_name, "variant": variant, "class_id": class_id,
                        "class_name": names[class_id], "tp": counts["tp"], "fp": counts["fp"],
                        "fn": counts["fn"], "support": counts["support"],
                        "precision": precision, "recall": recall, "f1": f1,
                    })
        return {"metrics": metrics, "trajectory_rows": trajectory_rows, "task_metrics": task_metrics, "class_metrics": class_metrics}

    @staticmethod
    def _select_stage_peaks(probabilities: Tensor) -> list[int]:
        from asrf.refinement.peaks import select_boundary_peaks
        return list(select_boundary_peaks(probabilities, threshold=0.5))

    def _checkpoint_payload(self, epoch: int, best_validation_metric: float) -> dict[str, Any]:
        return {
            "model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch, "global_step": self.global_step,
            "best_validation_metric": best_validation_metric, "best_epoch": self.best_epoch,
            "architecture_config": self.config.get("model", {}), "loss_config": self.config.get("loss", {}),
            "label_map": dict(self.mapping), "label_aliases": dict(self.mapping.aliases),
            "ontology_metadata": metadata_for_mapping(self.mapping, self.mapping.aliases, version=self.config.get("ontology_version")),
            "class_weights": self.train_stats.class_weights.detach().cpu(),
            "boundary_positive_weight": float(self.criterion.boundary_positive_weight),
            "reciprocal_boundary_positive_weight": self.train_stats.boundary_positive_weight,
            "boundary_target_config": self.boundary_target_config,
            "initialization": self.initialization_metadata,
            "seed": self.seed,
            "train_trajectory_ids": list(self.train_dataset.trajectory_ids), "validation_trajectory_ids": list(self.val_dataset.trajectory_ids),
            "train_split_entries": list(getattr(self.train_dataset, "entries", self.train_dataset.trajectory_ids)),
            "validation_split_entries": list(getattr(self.val_dataset, "entries", self.val_dataset.trajectory_ids)),
            "official_asrf_reference_commit": OFFICIAL_COMMIT, "code_version": _git_version(),
        }

    def train(self) -> dict[str, Any]:
        start_time = time.time()
        log_path = self.output_dir / "training.log"
        append_log(log_path, f"start_time_epoch={time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
        csv_fields = ["epoch", "split", "learning_rate", "best_validation_metric"]
        base_fields = ["total_loss", "asb_loss", "asb_ce", "asb_smoothing", "brb_loss", "raw_frame_accuracy", "raw_edit_score", "raw_f1@10", "raw_f1@25", "raw_f1@50", "refined_frame_accuracy", "refined_edit_score", "refined_f1@10", "refined_f1@25", "refined_f1@50", "predicted_boundary_count", "predicted_boundary_count_mean", "gradient_norm", "brb_gradient_norm", "brb_positive_contribution", "brb_negative_contribution", "brb_logit_abs_max", "brb_mean_probability", "trajectory_count"]
        csv_fields.extend(base_fields)
        csv_fields.extend(f"boundary_{tol}_{scope}_{metric}" for tol in TOLERANCES for scope in ("including_frame0", "internal_only") for metric in ("precision", "recall", "f1", "tp", "fp", "fn"))
        epochs_since_improvement = 0
        stage1_epochs = int(self.config["training"].get("stage1_epochs", 0))
        stage2_learning_rate = self.config["training"].get("stage2_learning_rate")
        if stage1_epochs > 0:
            for parameter in self.model.encoder.parameters():
                parameter.requires_grad_(False)
        else:
            for parameter in self.model.encoder.parameters():
                parameter.requires_grad_(True)
        with CSVMetricLogger(self.output_dir / "metrics.csv", csv_fields, append=self.resume_path is not None) as logger:
            for epoch in range(self.start_epoch, int(self.config["training"]["max_epochs"]) + 1):
                if stage1_epochs > 0 and epoch == stage1_epochs + 1:
                    for parameter in self.model.encoder.parameters():
                        parameter.requires_grad_(True)
                    if stage2_learning_rate is not None:
                        for group in self.optimizer.param_groups:
                            group["lr"] = float(stage2_learning_rate)
                train_result = self._run_epoch(epoch, training=True)
                val_result = self._run_epoch(epoch, training=False)
                val_metric = float(val_result["metrics"]["total_loss"])
                improved = val_metric < self.best_metric
                if improved:
                    self.best_metric = val_metric
                    self.best_epoch = epoch
                    epochs_since_improvement = 0
                    save_checkpoint(self.output_dir / "best.pt", self._checkpoint_payload(epoch, val_metric))
                else:
                    epochs_since_improvement += 1
                for split, result in (("train", train_result), ("val", val_result)):
                    row = {"epoch": epoch, "split": split, "learning_rate": self.optimizer.param_groups[0]["lr"], "best_validation_metric": self.best_metric}
                    row.update(result["metrics"])
                    logger.write(row)
                    append_log(log_path, json.dumps({"epoch": epoch, "split": split, **row}, sort_keys=True))
                if bool(self.config["training"].get("save_last", True)):
                    save_checkpoint(self.output_dir / "last.pt", self._checkpoint_payload(epoch, self.best_metric))
                self.history.extend([{"epoch": epoch, "split": "train", **train_result["metrics"]}, {"epoch": epoch, "split": "val", **val_result["metrics"]}])
                for task_name, task_result in val_result["task_metrics"].items():
                    self.task_history.append({"epoch": epoch, "task": task_name, **task_result["raw"], **{
                        f"refined_{key}": value for key, value in task_result["refined"].items()
                    }, "internal_boundary_f1@33": task_result["internal_boundary_f1@33"],
                    "internal_boundary_precision@33": task_result["internal_boundary_precision@33"],
                    "internal_boundary_recall@33": task_result["internal_boundary_recall@33"],
                    "predicted_boundary_count": task_result["predicted_boundary_count"]})
                for row in val_result["class_metrics"]:
                    self.class_history.append({"epoch": epoch, **row})
                minimum_epochs = int(self.config["training"]["minimum_epochs"])
                if epoch >= minimum_epochs and epochs_since_improvement >= int(self.config["training"]["early_stopping_patience"]):
                    break
        elapsed = time.time() - start_time
        summary = {
            "start_epoch": self.start_epoch, "stopping_epoch": self.history[-1]["epoch"] if self.history else 0,
            "best_epoch": self.best_epoch, "best_validation_total_loss": self.best_metric,
            "total_optimization_steps": self.global_step, "elapsed_seconds": elapsed,
            "history": self.history, "task_validation_history": self.task_history,
            "class_validation_history": self.class_history,
            "train_trajectory_ids": self.train_dataset.trajectory_ids,
            "validation_trajectory_ids": self.val_dataset.trajectory_ids, "total_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "official_asrf_reference_commit": OFFICIAL_COMMIT,
            "initialization": self.initialization_metadata,
            "stage1_epochs": stage1_epochs,
            "stage2_learning_rate": stage2_learning_rate,
        }
        if (self.output_dir / "best.pt").is_file():
            summary["best_checkpoint"] = checkpoint_manifest(self.output_dir / "best.pt")
        if (self.output_dir / "last.pt").is_file():
            summary["last_checkpoint"] = checkpoint_manifest(self.output_dir / "last.pt")
        (self.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
        if self.task_history:
            task_fields = list(self.task_history[0].keys())
            with (self.output_dir / "per_task_validation_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=task_fields)
                writer.writeheader()
                writer.writerows(self.task_history)
        if self.class_history:
            class_fields = list(self.class_history[0].keys())
            with (self.output_dir / "per_class_validation_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=class_fields)
                writer.writeheader()
                writer.writerows(self.class_history)
        if (self.output_dir / "best.pt").is_file() and (self.output_dir / "last.pt").is_file():
            (self.output_dir / "checkpoint_manifest.json").write_text(
                json.dumps({"best": checkpoint_manifest(self.output_dir / "best.pt"), "last": checkpoint_manifest(self.output_dir / "last.pt")}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self._plot_training_curves()
        append_log(log_path, f"end_time_epoch={time.strftime('%Y-%m-%dT%H:%M:%S%z')} elapsed_seconds={elapsed:.3f}")
        self.export_validation()
        return summary

    def export_validation(self) -> None:
        best_path = self.output_dir / "best.pt"
        if not best_path.is_file():
            return
        payload = load_checkpoint(best_path, map_location=self.device, expected_ontology=True)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        names = _label_names(self.mapping)
        import matplotlib.pyplot as plt

        for batch in self.val_loader:
            batch = self._batch_to_device(batch)
            self._validate_batch(batch)
            with torch.no_grad():
                output = self.model(batch["heatmap"], valid_mask=batch["valid_mask"])
            index = 0
            length = int(batch["lengths"][index])
            heatmap = batch["heatmap"][index, :, :, :length].cpu().numpy()
            truth = batch["labels"][index, :length].cpu()
            asb_stages = [stage[index, :, :length] for stage in output.asb_stage_probabilities]
            brb_stages = [stage[index:index + 1, :, :length] for stage in output.brb_stage_probabilities]
            result = refine_asrf_predictions(asb_stages[-1].unsqueeze(0), brb_stages[-1], torch.ones((1, length), dtype=torch.bool, device=self.device), threshold=float(self.config["refinement"]["boundary_threshold"]), voting="majority")
            raw = result.raw_labels[0, :length].cpu()
            refined = result.refined_labels[0, :length].cpu()
            confidence = asb_stages[-1].max(dim=0).values.cpu().numpy()
            trajectory_id = batch["trajectory_ids"][index]
            target_dir = self.output_dir / "validation" / str(trajectory_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            self._write_prediction_csvs(target_dir, truth, raw, refined, confidence, brb_stages[-1][0, 0].cpu(), batch["boundary_targets"][index, :length].cpu(), result)
            metrics = {"raw": trajectory_metrics(raw, truth), "refined": trajectory_metrics(refined, truth), "selected_boundaries": list(result.selected_boundaries[0]), "truth_boundaries": torch.where(batch["boundary_targets"][index, :length].cpu() > 0.5)[0].tolist(), "stage_asb": {}, "stage_brb": {}}
            for stage, probs in enumerate(asb_stages):
                metrics["stage_asb"][str(stage + 1)] = trajectory_metrics(probs.argmax(dim=0).cpu(), truth)
            truth_boundaries = metrics["truth_boundaries"]
            for stage, probs in enumerate(brb_stages):
                peaks = result.selected_boundaries[0] if stage == len(brb_stages) - 1 else self._select_stage_peaks(probs[0, 0].cpu())
                metrics["stage_brb"][str(stage + 1)] = {"selected_boundaries": peaks, "mean_probability": float(probs[0, 0].mean().cpu()), "boundary_at_33_internal": boundary_counts(peaks, truth_boundaries, 33, include_frame0=False)}
            (target_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
        self._plot_validation(target_dir / "annotation_vs_asrf.png", heatmap, truth.numpy(), raw.numpy(), refined.numpy(), brb_stages[-1][0, 0].cpu().numpy(), confidence, result.selected_boundaries[0], names, trajectory_id)

    def _plot_training_curves(self) -> None:
        """Write compact loss/accuracy curves without affecting training."""
        if not self.history:
            return
        import matplotlib.pyplot as plt
        train = [row for row in self.history if row.get("split") == "train"]
        val = [row for row in self.history if row.get("split") == "val"]
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        for rows, label in ((train, "train"), (val, "validation")):
            epochs = [row["epoch"] for row in rows]
            axes[0].plot(epochs, [row.get("total_loss", 0.0) for row in rows], label=label)
            axes[1].plot(epochs, [row.get("refined_frame_accuracy", 0.0) for row in rows], label=label)
        axes[0].set_ylabel("total loss")
        axes[1].set_ylabel("refined accuracy")
        axes[1].set_xlabel("epoch")
        axes[0].legend()
        axes[1].legend()
        figure.savefig(self.output_dir / "training_curves.png", dpi=120)
        plt.close(figure)

    @staticmethod
    def _write_prediction_csvs(directory: Path, truth: Tensor, raw: Tensor, refined: Tensor, confidence: np.ndarray, boundary_probability: Tensor, boundary_target: Tensor, result: ASRFRefinementOutput) -> None:
        with (directory / "frame_predictions_asb.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "ground_truth", "asb_raw_label", "asrf_refined_label", "asb_confidence"])
            for index in range(len(truth)):
                writer.writerow([index, int(truth[index]), int(raw[index]), int(refined[index]), float(confidence[index])])
        with (directory / "boundary_probabilities.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "boundary_probability", "boundary_target", "selected_peak"])
            selected = set(result.selected_boundaries[0])
            for index, value in enumerate(boundary_probability.tolist()):
                writer.writerow([index, float(value), float(boundary_target[index]), int(index in selected)])
        with (directory / "selected_boundaries.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["boundary_index"])
            for index in result.selected_boundaries[0]:
                writer.writerow([index])
        with (directory / "predicted_segments_asrf.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["start", "end_exclusive", "duration", "selected_class"])
            for interval, diagnostic in zip(result.intervals[0], result.segment_diagnostics[0]):
                writer.writerow([interval.start, interval.end, interval.duration, diagnostic.selected_class])

    @staticmethod
    def _plot_validation(path: Path, heatmap: np.ndarray, truth: np.ndarray, raw: np.ndarray, refined: np.ndarray, boundary: np.ndarray, confidence: np.ndarray, peaks: list[int] | tuple[int, ...], names: dict[int, str], trajectory_id: str) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        length = heatmap.shape[-1]
        palette = plt.get_cmap("tab20").colors
        colors = tuple(palette[index % len(palette)] for index in range(max(7, len(names))))
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, max(7, len(names)) + 0.5, 1), cmap.N)
        figure, axes = plt.subplots(7, 1, figsize=(18, 11), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [5, 0.7, 0.7, 0.7, 1.4, 0.7, 1.0]})
        axes[0].imshow(np.moveaxis(heatmap, 0, -1), origin="upper", aspect="auto", interpolation="nearest", extent=(0, length, heatmap.shape[1], 0))
        axes[0].set_ylabel("CITR\nheatmap")
        for axis, values, label in ((axes[1], truth, "ground truth"), (axes[2], raw, "ASB raw"), (axes[3], refined, "ASRF refined")):
            axis.imshow(values[np.newaxis, :], origin="lower", aspect="auto", interpolation="nearest", extent=(0, length, 0, 1), cmap=cmap, norm=norm)
            axis.set_ylabel(label)
        axes[4].plot(np.arange(length) + 0.5, boundary, color="black", linewidth=0.8)
        for peak in peaks:
            axes[4].axvline(peak, color="red", linewidth=0.8, alpha=0.8)
        axes[4].set_ylim(0.0, 1.0)
        axes[4].set_ylabel("BRB p\npeaks")
        axes[5].imshow(refined[np.newaxis, :], origin="lower", aspect="auto", interpolation="nearest", extent=(0, length, 0, 1), cmap=cmap, norm=norm)
        axes[5].set_ylabel("vote")
        axes[6].plot(np.arange(length) + 0.5, confidence, color="navy", linewidth=0.8)
        axes[6].set_ylim(0.0, 1.0)
        axes[6].set_ylabel("ASB conf.")
        axes[6].set_xlabel("frame index; exact temporal width preserved")
        for axis in axes:
            axis.set_xlim(0, length)
            axis.set_yticks([])
        figure.suptitle(f"{trajectory_id} — display downsampling factor = 1; labels: " + ", ".join(f"{key}:{value}" for key, value in sorted(names.items())))
        figure.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(figure)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


__all__ = ["ASRFTrainer", "OFFICIAL_COMMIT", "seed_everything"]
