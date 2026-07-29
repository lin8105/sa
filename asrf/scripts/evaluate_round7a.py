#!/usr/bin/env python
"""Comprehensive read-only evaluation for ASRF Round 7A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asrf.data.dataset import MultiTaskTrajectoryDataset  # noqa: E402
from asrf.data.labels import LabelMapping, load_label_mapping  # noqa: E402
from asrf.evaluation.metrics import boundary_counts, edit_score, labels_to_segments, segmental_f1  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.refinement.majority_vote import _vote_one  # noqa: E402
from asrf.refinement.peaks import select_boundary_peaks  # noqa: E402
from asrf.refinement.refine import refine_asrf_predictions  # noqa: E402
from asrf.training.checkpointing import load_checkpoint, sha256_file  # noqa: E402
from asrf.utils.config import load_yaml_config, resolve_repo_path  # noqa: E402


OUT_ROOT = REPO_ROOT / "outputs/brb_ablation_round7a"
DATA_ROOT = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
TOLERANCES = (0, 5, 10, 20, 30, 33, 50)
TRANSITIONS = (
    "reach -> grasp", "grasp -> lift", "lift -> transport", "transport -> pour",
    "pour -> pour_recover", "pour_recover -> transport", "transport -> place",
    "place -> release", "place -> wipe", "wipe -> lift", "place -> retreat",
)


def _json(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _names(mapping: LabelMapping) -> dict[int, str]:
    return {int(value): str(name) for name, value in mapping.items()}


def _task_from_entry(entry: str) -> str:
    parts = Path(entry).parts
    if len(parts) < 2:
        return "unknown"
    return "pp" if parts[1] == "pp" else ("pp" if parts[1] == "pick and place" else parts[1])


def _load_model(config: dict[str, Any], checkpoint: Path) -> ASRFModel:
    model = ASRFModel.from_config(config).to("cpu")
    payload = load_checkpoint(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


@torch.no_grad()
def _record(model: ASRFModel, dataset: MultiTaskTrajectoryDataset, index: int) -> dict[str, Any]:
    sample = dataset[index]
    heatmap = sample["heatmap"].unsqueeze(0)
    mask = sample["valid_mask"].unsqueeze(0)
    output = model(heatmap, valid_mask=mask)
    length = int(sample["labels"].numel())
    return {
        "entry": str(sample["trajectory_id"]),
        "task": _task_from_entry(str(sample["trajectory_id"])),
        "truth": sample["labels"].cpu(),
        "targets": sample.get("hard_boundary_targets", sample["boundary_targets"]).cpu(),
        "heatmap": sample["heatmap"].cpu(),
        "asb": output.asb_stage_probabilities[-1][0, :, :length].cpu(),
        "brb": output.brb_stage_probabilities[-1][0, 0, :length].cpu(),
    }


def _truth_boundaries(record: dict[str, Any], *, internal: bool = False) -> list[int]:
    values = torch.where(record["targets"] > 0.5)[0].tolist()
    return [int(value) for value in values if not internal or int(value) != 0]


def _matched_errors(predicted: list[int], target: list[int], tolerance: int) -> list[int]:
    candidates = sorted((abs(p - t), p, t) for p in predicted for t in target if abs(p - t) <= tolerance)
    used_p: set[int] = set()
    used_t: set[int] = set()
    errors: list[int] = []
    for error, p, t in candidates:
        if p not in used_p and t not in used_t:
            used_p.add(p)
            used_t.add(t)
            errors.append(int(error))
    return errors


def _confusion(prediction: torch.Tensor, truth: torch.Tensor, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (truth.numpy(), prediction.numpy()), 1)
    return matrix


def _balanced_accuracy(prediction: torch.Tensor, truth: torch.Tensor, num_classes: int) -> float:
    recalls = []
    for class_id in range(num_classes):
        support = int((truth == class_id).sum())
        if support:
            recalls.append(float(((prediction == class_id) & (truth == class_id)).sum()) / support)
    return float(np.mean(recalls)) if recalls else 0.0


def _semantic_metrics(prediction: torch.Tensor, truth: torch.Tensor, num_classes: int) -> dict[str, Any]:
    truth_segments = labels_to_segments(truth)
    pred_segments = labels_to_segments(prediction)
    true_count = len(truth_segments)
    predicted_count = len(pred_segments)
    return {
        "frame_accuracy": float((prediction == truth).float().mean()),
        "balanced_frame_accuracy": _balanced_accuracy(prediction, truth, num_classes),
        "edit": float(edit_score(prediction, truth)),
        "F1@10": float(segmental_f1(prediction, truth, 0.10)),
        "F1@25": float(segmental_f1(prediction, truth, 0.25)),
        "F1@50": float(segmental_f1(prediction, truth, 0.50)),
        "collapsed_sequence": [int(value) for value in _collapse(prediction)],
        "predicted_segment_count": predicted_count,
        "true_segment_count": true_count,
        "over_segmentation_ratio": max(0, predicted_count - true_count) / max(1, true_count),
        "under_segmentation_ratio": max(0, true_count - predicted_count) / max(1, true_count),
        "confusion_matrix": _confusion(prediction, truth, num_classes).tolist(),
    }


def _class_metrics(rows: list[dict[str, Any]], mapping: LabelMapping, variant: str) -> list[dict[str, Any]]:
    names = _names(mapping)
    counts = {class_id: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for class_id in range(len(mapping))}
    for row in rows:
        prediction = row["variants"][variant]["prediction"]
        truth = row["truth"]
        for class_id in counts:
            p = prediction == class_id
            t = truth == class_id
            counts[class_id]["tp"] += int((p & t).sum())
            counts[class_id]["fp"] += int((p & ~t).sum())
            counts[class_id]["fn"] += int((~p & t).sum())
            counts[class_id]["support"] += int(t.sum())
    result = []
    for class_id in range(len(mapping)):
        value = counts[class_id]
        precision = value["tp"] / (value["tp"] + value["fp"]) if value["tp"] + value["fp"] else 0.0
        recall = value["tp"] / value["support"] if value["support"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.append({"variant": variant, "class_id": class_id, "class": names[class_id], **value, "precision": precision, "recall": recall, "f1": f1})
    return result


def _collapse(values: torch.Tensor) -> tuple[int, ...]:
    sequence = values.tolist()
    if not sequence:
        return ()
    result = [int(sequence[0])]
    for value in sequence[1:]:
        if int(value) != result[-1]:
            result.append(int(value))
    return tuple(result)


def _variants(record: dict[str, Any], threshold: float) -> dict[str, Any]:
    asb = record["asb"]
    truth = record["truth"]
    mask = torch.ones(1, len(truth), dtype=torch.bool)
    official = refine_asrf_predictions(asb.unsqueeze(0), record["brb"].view(1, 1, -1), mask, threshold=0.5, voting="majority")
    calibrated = refine_asrf_predictions(asb.unsqueeze(0), record["brb"].view(1, 1, -1), mask, threshold=threshold, voting="majority")
    oracle_boundaries = _truth_boundaries(record)
    oracle_intervals = __import__("asrf.refinement.segments", fromlist=["construct_segments"]).construct_segments(oracle_boundaries, len(truth))
    oracle_prediction, oracle_diagnostics = _vote_one(asb, oracle_intervals, voting="majority")
    return {
        "raw": {"prediction": asb.argmax(dim=0), "boundaries": []},
        "official": {"prediction": official.refined_labels[0], "boundaries": list(official.selected_boundaries[0]), "refinement": official},
        "calibrated": {"prediction": calibrated.refined_labels[0], "boundaries": list(calibrated.selected_boundaries[0]), "refinement": calibrated},
        "oracle": {"prediction": oracle_prediction, "boundaries": oracle_boundaries, "oracle_diagnostics": oracle_diagnostics},
    }


def _peak_diagnostics(record: dict[str, Any], threshold: float) -> dict[str, Any]:
    probability = record["brb"]
    target = _truth_boundaries(record, internal=True)
    all_peaks = list(select_boundary_peaks(probability, threshold=0.0))
    selected = list(select_boundary_peaks(probability, threshold=threshold))
    internal_selected = [peak for peak in selected if peak != 0]
    counts = boundary_counts(internal_selected, target, 33, include_frame0=False)
    errors = _matched_errors(internal_selected, target, 33)
    segment_starts = target
    duplicate = 0
    near_count = 0
    for boundary in target:
        nearby = [peak for peak in internal_selected if abs(peak - boundary) <= 33]
        near_count += len(nearby)
        duplicate += max(0, len(nearby) - 1)
    false_inside = 0
    for peak in internal_selected:
        if not any(start <= peak < end for start, end in zip([0] + target, target + [len(record["truth"])])):
            false_inside += 1
    def _window_max(radius: int) -> list[float]:
        return [float(probability[max(0, boundary - radius):min(len(probability), boundary + radius + 1)].max()) for boundary in target]
    positive_values = probability[target].tolist() if target else []
    non_boundary_mask = torch.ones(len(probability), dtype=torch.bool)
    non_boundary_mask[torch.tensor(target, dtype=torch.long)] = False if target else non_boundary_mask[torch.tensor([], dtype=torch.long)]
    negative_values = probability[non_boundary_mask].tolist()
    far_mask = torch.ones(len(probability), dtype=torch.bool)
    for boundary in target:
        far_mask[max(0, boundary - 33):min(len(probability), boundary + 34)] = False
    far_values = probability[far_mask].tolist()
    return {
        "total_local_maxima": len(all_peaks), "selected_peaks": len(selected), "selected_peak_indices": selected,
        "true_internal_boundaries": len(target), "matched_predicted_boundaries": int(counts["tp"]),
        "false_positive_peaks": int(counts["fp"]), "missed_boundaries": int(counts["fn"]),
        "peak_precision": float(counts["precision"]), "peak_recall": float(counts["recall"]), "peak_f1": float(counts["f1"]),
        "mean_absolute_matched_boundary_error": float(np.mean(errors)) if errors else 0.0,
        "median_absolute_matched_boundary_error": float(np.median(errors)) if errors else 0.0,
        "duplicate_peaks_around_one_boundary": duplicate, "false_peaks_inside_ground_truth_segments": false_inside,
        "max_probability_within_10": _window_max(10), "max_probability_within_20": _window_max(20), "max_probability_within_33": _window_max(33),
        "probability_at_exact_boundary": positive_values, "mean_true_boundary_probability": float(np.mean(positive_values)) if positive_values else 0.0,
        "mean_non_boundary_probability": float(np.mean(negative_values)) if negative_values else 0.0,
        "mean_far_probability": float(np.mean(far_values)) if far_values else 0.0,
        "positive_negative_probability_separation": (float(np.mean(positive_values)) - float(np.mean(negative_values))) if positive_values and negative_values else 0.0,
        "near_boundary_selected_peak_count": near_count,
    }


def _boundary_summary(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tolerance in TOLERANCES:
        macro = []
        pooled = {"tp": 0, "fp": 0, "fn": 0}
        for row in rows:
            predicted = row["variants"][variant]["boundaries"]
            truth = _truth_boundaries(row, internal=True)
            item = boundary_counts(predicted, truth, tolerance, include_frame0=False)
            macro.append(item)
            for key in pooled:
                pooled[key] += int(item[key])
        def scores(value: dict[str, Any]) -> dict[str, Any]:
            p = value["tp"] / (value["tp"] + value["fp"]) if value["tp"] + value["fp"] else 0.0
            r = value["tp"] / (value["tp"] + value["fn"]) if value["tp"] + value["fn"] else 0.0
            return {**value, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
        result[f"internal_{tolerance}"] = {"macro_trajectory": {key: float(np.mean([float(row[key]) for row in macro])) if macro else 0.0 for key in ("precision", "recall", "f1")}, "pooled": scores(pooled)}
    return result


def _summarize(rows: list[dict[str, Any]], mapping: LabelMapping) -> dict[str, Any]:
    summary: dict[str, Any] = {"trajectory_count": len(rows), "variants": {}, "boundary": {}, "mean_selected_peaks_official": float(np.mean([len(row["variants"]["official"]["boundaries"]) for row in rows])) if rows else 0.0, "mean_true_internal_boundaries": float(np.mean([len(_truth_boundaries(row, internal=True)) for row in rows])) if rows else 0.0}
    for variant in ("raw", "official", "calibrated", "oracle"):
        metrics = [row["variants"][variant]["semantic"] for row in rows]
        keys = ("frame_accuracy", "balanced_frame_accuracy", "edit", "F1@10", "F1@25", "F1@50", "predicted_segment_count", "over_segmentation_ratio", "under_segmentation_ratio")
        macro = {key: float(np.mean([float(item[key]) for item in metrics])) if metrics else 0.0 for key in keys}
        truth_frames = sum(len(row["truth"]) for row in rows)
        correct = sum(int((row["variants"][variant]["prediction"] == row["truth"]).sum()) for row in rows)
        confusion = np.sum([np.asarray(row["variants"][variant]["semantic"]["confusion_matrix"]) for row in rows], axis=0).tolist() if rows else np.zeros((len(mapping), len(mapping)), dtype=int).tolist()
        summary["variants"][variant] = {"macro_trajectory": macro, "pooled_frame_accuracy": correct / max(1, truth_frames), "confusion_matrix": confusion, "per_class": _class_metrics(rows, mapping, variant)}
        summary["boundary"][variant] = _boundary_summary(rows, variant)
    return summary


def _make_records(model: ASRFModel, split: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    split_path = resolve_repo_path(split)
    data = config["data"]
    target_config = {key: data[key] for key in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame") if key in data}
    dataset = MultiTaskTrajectoryDataset(DATA_ROOT, split_path, resolve_repo_path(data["label_config"]), expected_height=88, allow_test=split.startswith("splits/multitask_test"), boundary_target_config=target_config)
    return [_record(model, dataset, index) for index in range(len(dataset))]


def _attach_variant_metrics(rows: list[dict[str, Any]], threshold: float, num_classes: int) -> None:
    for row in rows:
        variants = _variants(row, threshold)
        for name, item in variants.items():
            item["semantic"] = _semantic_metrics(item["prediction"], row["truth"], num_classes)
            item["boundary_diagnostics"] = _peak_diagnostics(row, 0.5) if name == "official" else None
        row["variants"] = variants


def _calibrate(validation_rows: list[dict[str, Any]], output: Path, num_classes: int) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for threshold in THRESHOLDS:
        values = []
        for row in validation_rows:
            refinement = refine_asrf_predictions(row["asb"].unsqueeze(0), row["brb"].view(1, 1, -1), torch.ones(1, len(row["truth"]), dtype=torch.bool), threshold=threshold, voting="majority")
            item = boundary_counts([p for p in refinement.selected_boundaries[0] if p != 0], _truth_boundaries(row, internal=True), 33, include_frame0=False)
            prediction = refinement.refined_labels[0]
            values.append((item, _semantic_metrics(prediction, row["truth"], num_classes)))
        pooled = {key: sum(int(item[0][key]) for item in values) for key in ("tp", "fp", "fn")}
        p = pooled["tp"] / (pooled["tp"] + pooled["fp"]) if pooled["tp"] + pooled["fp"] else 0.0
        r = pooled["tp"] / (pooled["tp"] + pooled["fn"]) if pooled["tp"] + pooled["fn"] else 0.0
        rows.append({"threshold": threshold, "pooled_boundary_precision_33": p, "pooled_boundary_recall_33": r, "pooled_boundary_f1_33": 2 * p * r / (p + r) if p + r else 0.0, "false_positive_peaks": pooled["fp"], "macro_refined_F1@50": float(np.mean([item[1]["F1@50"] for item in values])) if values else 0.0, "macro_refined_accuracy": float(np.mean([item[1]["frame_accuracy"] for item in values])) if values else 0.0})
    chosen = sorted(rows, key=lambda item: (-item["pooled_boundary_f1_33"], item["false_positive_peaks"], -item["macro_refined_F1@50"], -item["macro_refined_accuracy"], -item["threshold"]))[0]["threshold"]
    _write_csv(output / "validation_threshold_calibration.csv", rows)
    _write_json(output / "frozen_postprocessing.json", {"official_threshold": 0.5, "calibrated_threshold": chosen, "selection_data": "validation only", "selection_primary": "pooled internal boundary F1@33", "thresholds": list(THRESHOLDS)})
    return float(chosen), rows


def _epoch_diagnostics(experiment_dir: Path) -> dict[str, Any]:
    metrics_path = experiment_dir / "metrics.csv"
    if not metrics_path.is_file():
        return {}
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "val"]
    def best(metric: str, maximize: bool = True) -> dict[str, Any]:
        values = [(float(row[metric]), int(float(row["epoch"]))) for row in rows if row.get(metric) not in (None, "")]
        if not values:
            return {}
        value, epoch = (max(values) if maximize else min(values))
        return {"epoch": epoch, metric: value}
    return {"primary": best("total_loss", maximize=False), "best_raw_asb_F1@50": best("raw_f1@50"), "best_official_refined_F1@50": best("refined_f1@50"), "best_internal_boundary_F1@33": best("boundary_33_internal_only_f1"), "all_validation_epochs": rows}


def _fixed_threshold_rows(rows: list[dict[str, Any]], experiment: str, split: str) -> list[dict[str, Any]]:
    result = []
    for threshold in THRESHOLDS:
        semantic = []
        boundary = []
        selected_counts = []
        for row in rows:
            refinement = refine_asrf_predictions(row["asb"].unsqueeze(0), row["brb"].view(1, 1, -1), torch.ones(1, len(row["truth"]), dtype=torch.bool), threshold=threshold, voting="majority")
            prediction = refinement.refined_labels[0]
            semantic.append(_semantic_metrics(prediction, row["truth"], num_classes=10))
            boundary.append(boundary_counts([p for p in refinement.selected_boundaries[0] if p != 0], _truth_boundaries(row, internal=True), 33, include_frame0=False))
            selected_counts.append(len(refinement.selected_boundaries[0]))
        tp = sum(int(item["tp"]) for item in boundary); fp = sum(int(item["fp"]) for item in boundary); fn = sum(int(item["fn"]) for item in boundary)
        precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
        result.append({"experiment": experiment, "split": split, "threshold": threshold, "macro_accuracy": float(np.mean([item["frame_accuracy"] for item in semantic])) if semantic else 0.0, "macro_edit": float(np.mean([item["edit"] for item in semantic])) if semantic else 0.0, "macro_F1_50": float(np.mean([item["F1@50"] for item in semantic])) if semantic else 0.0, "boundary_precision_33": precision, "boundary_recall_33": recall, "boundary_F1_33": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "false_positive_peaks": fp, "missed_boundaries": fn, "selected_peaks": sum(selected_counts)})
    return result


def _transition_rows(rows: list[dict[str, Any]], experiment: str, mapping: LabelMapping) -> list[dict[str, Any]]:
    names = _names(mapping)
    result = []
    for variant in ("official",):
        for row in rows:
            truth = row["truth"]
            target = _truth_boundaries(row, internal=True)
            segments = labels_to_segments(truth)
            for segment in segments[1:]:
                previous = names[int(truth[segment.start - 1])]
                current = names[int(truth[segment.start])]
                transition = f"{previous} -> {current}"
                if transition not in TRANSITIONS:
                    continue
                boundary = segment.start
                prob = row["brb"]
                predicted = row["variants"][variant]["boundaries"]
                matched = [p for p in predicted if p != 0 and abs(p - boundary) <= 33]
                errors = [abs(p - boundary) for p in matched[:1]]
                result.append({"experiment": experiment, "transition": transition, "trajectory": row["entry"], "support": 1, "exact_boundary_probability": float(prob[boundary]), "max_probability_within_10": float(prob[max(0, boundary - 10):min(len(prob), boundary + 11)].max()), "max_probability_within_20": float(prob[max(0, boundary - 20):min(len(prob), boundary + 21)].max()), "max_probability_within_33": float(prob[max(0, boundary - 33):min(len(prob), boundary + 34)].max()), "detected_at_threshold_0.50": int(bool(matched)), "localization_error": errors[0] if errors else "", "missed_context": "none" if matched else f"{previous} -> {current}"})
    grouped: list[dict[str, Any]] = []
    for transition in TRANSITIONS:
        subset = [item for item in result if item["transition"] == transition]
        grouped.append({"experiment": experiment, "transition": transition, "support": len(subset), "mean_exact_boundary_probability": float(np.mean([item["exact_boundary_probability"] for item in subset])) if subset else 0.0, "mean_max_probability_within_10": float(np.mean([item["max_probability_within_10"] for item in subset])) if subset else 0.0, "mean_max_probability_within_20": float(np.mean([item["max_probability_within_20"] for item in subset])) if subset else 0.0, "mean_max_probability_within_33": float(np.mean([item["max_probability_within_33"] for item in subset])) if subset else 0.0, "detection_recall_threshold_0.50": float(np.mean([item["detected_at_threshold_0.50"] for item in subset])) if subset else 0.0, "mean_localization_error": float(np.mean([item["localization_error"] for item in subset if item["localization_error"] != ""])) if any(item["localization_error"] != "" for item in subset) else 0.0, "common_missed_transitions": transition if subset and not any(item["detected_at_threshold_0.50"] for item in subset) else "", "common_false_boundary_contexts": ""})
    return grouped


def _distribution_figures(all_rows: dict[str, list[dict[str, Any]]], figures: Path) -> None:
    import matplotlib.pyplot as plt
    figures.mkdir(parents=True, exist_ok=True)
    experiments = list(all_rows)
    fig, axes = plt.subplots(len(experiments), 2, figsize=(12, max(6, 2.5 * len(experiments))), squeeze=False)
    for index, experiment in enumerate(experiments):
        positives = np.concatenate([row["brb"].numpy()[np.asarray(row["targets"]) > 0.5] for row in all_rows[experiment]])
        negatives = np.concatenate([row["brb"].numpy()[np.asarray(row["targets"]) <= 0.5] for row in all_rows[experiment]])
        near = []
        far = []
        for row in all_rows[experiment]:
            p = row["brb"].numpy()
            boundary = _truth_boundaries(row, internal=True)
            near_mask = np.zeros(len(p), dtype=bool)
            for value in boundary:
                near_mask[max(0, value - 10):min(len(p), value + 11)] = True
            near.append(p[near_mask]); far.append(p[~near_mask])
        axes[index, 0].hist(positives, bins=30, range=(0, 1), alpha=0.65, label="exact positive")
        axes[index, 0].hist(negatives, bins=30, range=(0, 1), alpha=0.65, label="negative")
        axes[index, 1].hist(np.concatenate(near), bins=30, range=(0, 1), alpha=0.65, label="within ±10")
        axes[index, 1].hist(np.concatenate(far), bins=30, range=(0, 1), alpha=0.65, label="far")
        axes[index, 0].set_title(experiment); axes[index, 1].set_title(experiment)
        axes[index, 0].legend(fontsize=7); axes[index, 1].legend(fontsize=7)
        for axis in axes[index]: axis.set_xlim(0, 1); axis.set_ylabel("frames")
    axes[-1, 0].set_xlabel("BRB probability"); axes[-1, 1].set_xlabel("BRB probability")
    fig.suptitle("Round 7A BRB probability distributions"); fig.tight_layout(); fig.savefig(figures / "probability_distributions.png", dpi=130); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for experiment in experiments:
        y_true = np.concatenate([row["targets"].numpy()[1:] for row in all_rows[experiment]])
        y_score = np.concatenate([row["brb"].numpy()[1:] for row in all_rows[experiment]])
        order = np.argsort(-y_score); truth = y_true[order]; tp = np.cumsum(truth); fp = np.cumsum(1 - truth)
        axis.plot(fp / np.maximum(1, fp[-1]), tp / np.maximum(1, tp[-1]), label=experiment)
    axis.set_xlabel("normalized false positives"); axis.set_ylabel("recall"); axis.set_title("BRB precision-recall curves (internal frames)"); axis.legend(); fig.tight_layout(); fig.savefig(figures / "precision_recall_curves.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for experiment in experiments:
        f1 = []; false = []; missed = []
        for threshold in THRESHOLDS:
            values = [_peak_diagnostics(row, threshold) for row in all_rows[experiment]]
            f1.append(np.mean([v["peak_f1"] for v in values])); false.append(sum(v["false_positive_peaks"] for v in values)); missed.append(sum(v["missed_boundaries"] for v in values))
        axes[0].plot(THRESHOLDS, f1, marker="o", label=experiment); axes[1].plot(THRESHOLDS, false, marker="o", label=experiment); axes[2].plot(THRESHOLDS, missed, marker="o", label=experiment)
    axes[0].set_title("Boundary F1 vs threshold"); axes[1].set_title("False-positive peaks vs threshold"); axes[2].set_title("Missed boundaries vs threshold")
    for axis in axes: axis.set_xlabel("threshold"); axis.grid(alpha=0.25)
    axes[0].set_ylabel("F1"); axes[1].set_ylabel("count"); axes[2].set_ylabel("count"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "boundary_threshold_diagnostics.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for experiment in experiments:
        semantic = []; boundary = []
        for threshold in THRESHOLDS:
            items = []
            for row in all_rows[experiment]:
                refinement = refine_asrf_predictions(row["asb"].unsqueeze(0), row["brb"].view(1, 1, -1), torch.ones(1, len(row["truth"]), dtype=torch.bool), threshold=threshold)
                items.append((refinement.refined_labels[0], list(refinement.selected_boundaries[0])))
            semantic.append(float(np.mean([_semantic_metrics(pred, row["truth"], 10)["F1@50"] for row, (pred, _) in zip(all_rows[experiment], items)])))
            b = [boundary_counts([p for p in peaks if p != 0], _truth_boundaries(row, internal=True), 33, include_frame0=False) for row, (_, peaks) in zip(all_rows[experiment], items)]
            tp = sum(int(v["tp"]) for v in b); fp = sum(int(v["fp"]) for v in b); fn = sum(int(v["fn"]) for v in b); p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
            boundary.append(2 * p * r / (p + r) if p + r else 0.0)
        axes[0].plot(THRESHOLDS, semantic, marker="o", label=experiment); axes[1].plot(boundary, semantic, marker="o", label=experiment)
    axes[0].set_xlabel("threshold"); axes[0].set_ylabel("semantic F1@50"); axes[0].set_title("Semantic F1@50 vs threshold")
    axes[1].set_xlabel("boundary F1@±33"); axes[1].set_ylabel("semantic F1@50"); axes[1].set_title("Boundary-semantic trade-off")
    for axis in axes: axis.grid(alpha=0.25); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "boundary_semantic_tradeoff.png", dpi=130); plt.close(fig)


def _representative_figures(all_rows: dict[str, list[dict[str, Any]]], figures: Path, mapping: LabelMapping, calibrated: dict[str, float]) -> None:
    import matplotlib.pyplot as plt
    names = _names(mapping)
    desired = ("test/pour/p1", "test/pp/pp_c1", "test/wipe/w1", "test/wipe/w4")
    for experiment, rows in all_rows.items():
        for wanted in desired:
            matches = [row for row in rows if row["entry"] == wanted]
            if not matches:
                continue
            row = matches[0]; variants = row["variants"]
            length = len(row["truth"]); fig, axes = plt.subplots(5, 1, figsize=(18, 9), sharex=True)
            axes[0].imshow(np.moveaxis(row["heatmap"].numpy(), 0, -1), aspect="auto", interpolation="nearest"); axes[0].set_ylabel("heatmap")
            for axis, key in zip(axes[1:], ("truth", "raw", "official", "calibrated")):
                values = row["truth"].numpy() if key == "truth" else variants[key]["prediction"].numpy()
                axis.imshow(values[np.newaxis, :], aspect="auto", interpolation="nearest", cmap="tab10", vmin=0, vmax=max(8, len(mapping) - 1)); axis.set_ylabel(key)
            for peak in variants["official"]["boundaries"]: axes[3].axvline(peak, color="red", linewidth=0.7)
            for peak in variants["calibrated"]["boundaries"]: axes[4].axvline(peak, color="orange", linewidth=0.7)
            for peak in _truth_boundaries(row): axes[1].axvline(peak, color="lime", linewidth=0.7)
            axes[-1].set_xlim(0, length); axes[-1].set_xlabel("frame"); fig.suptitle(f"{experiment} — {wanted} — calibrated={calibrated[experiment]:.2f}"); fig.tight_layout()
            output = figures / "representative" / f"{experiment}_{wanted.replace('/', '_').replace(' ', '_')}.png"; output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=120); plt.close(fig)


def _refinement_effect(rows_by_split: dict[str, list[dict[str, Any]]], experiment: str, output: Path) -> None:
    result = []
    for split, rows in rows_by_split.items():
        for row in rows:
            raw = row["variants"]["raw"]; official = row["variants"]["official"]
            raw_m = raw["semantic"]; official_m = official["semantic"]
            delta = {key: float(official_m[key] - raw_m[key]) for key in ("frame_accuracy", "edit", "F1@10", "F1@25", "F1@50")}
            if all(abs(value) < 1e-12 for value in delta.values()): status = "refinement unchanged"
            elif any(value > 0 for value in delta.values()): status = "refinement improved"
            else: status = "refinement harmed"
            truth = _truth_boundaries(row, internal=True); predicted = [p for p in official["boundaries"] if p != 0]
            counts = boundary_counts(predicted, truth, 33, include_frame0=False)
            cause = "none"
            if status == "refinement harmed":
                cause = "missed boundary causing segment merging" if counts["fn"] else ("false boundary causing unnecessary splitting" if counts["fp"] else "incorrect ASB majority inside a BRB segment")
            result.append({"experiment": experiment, "split": split, "task": row["task"], "trajectory": row["entry"], **{f"delta_{key}": value for key, value in delta.items()}, "classification": status, "cause": cause, "official_false_peaks": counts["fp"], "official_missed_boundaries": counts["fn"], "official_selected_peaks": len(official["boundaries"])})
    _write_csv(output, result)


def _model_comparison(experiment: str, positive_weight: float, best_epoch: int, validation_total_loss: float, val_summary: dict[str, Any], test_summary: dict[str, Any], calibration: float, training_summary: dict[str, Any], checkpoint: Path, effect_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def m(summary: dict[str, Any], variant: str, key: str) -> float:
        return float(summary["variants"][variant]["macro_trajectory"].get(key, 0.0))
    official_boundary = test_summary["boundary"]["official"]["internal_33"]["pooled"]
    return {"experiment": experiment, "positive_weight": positive_weight, "best_epoch": best_epoch, "validation_total_loss": validation_total_loss, "raw_accuracy": m(test_summary, "raw", "frame_accuracy"), "raw_edit": m(test_summary, "raw", "edit"), "raw_F1_50": m(test_summary, "raw", "F1@50"), "official_accuracy": m(test_summary, "official", "frame_accuracy"), "official_edit": m(test_summary, "official", "edit"), "official_F1_50": m(test_summary, "official", "F1@50"), "official_boundary_precision_33": official_boundary["precision"], "official_boundary_recall_33": official_boundary["recall"], "official_boundary_F1_33": official_boundary["f1"], "official_false_peaks": official_boundary["fp"], "official_missed_boundaries": official_boundary["fn"], "calibrated_threshold": calibration, "calibrated_accuracy": m(test_summary, "calibrated", "frame_accuracy"), "calibrated_F1_50": m(test_summary, "calibrated", "F1@50"), "calibrated_boundary_F1_33": test_summary["boundary"]["calibrated"]["internal_33"]["pooled"]["f1"], "oracle_accuracy": m(test_summary, "oracle", "frame_accuracy"), "oracle_F1_50": m(test_summary, "oracle", "F1@50"), "mean_selected_peaks_per_trajectory": test_summary.get("mean_selected_peaks_official", 0.0), "mean_true_boundaries_per_trajectory": test_summary.get("mean_true_internal_boundaries", 0.0), "refinement_improved_count": sum(row["classification"] == "refinement improved" for row in effect_rows), "refinement_harmed_count": sum(row["classification"] == "refinement harmed" for row in effect_rows), "training_duration_s": float(training_summary.get("elapsed_seconds", 0.0)), "checkpoint_sha256": sha256_file(checkpoint), "validation_raw_F1_50": m(val_summary, "raw", "F1@50"), "validation_official_F1_50": m(val_summary, "official", "F1@50"), "validation_boundary_F1_33": val_summary["boundary"]["official"]["internal_33"]["pooled"]["f1"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/brb_ablation_round7a")
    args = parser.parse_args()
    root = REPO_ROOT / args.output_dir
    mapping = load_label_mapping(REPO_ROOT / "configs/labels_multitask.yaml")
    experiments = {"pw_reciprocal": ("configs/brb_ablation_round7a/brb_pw_reciprocal.yaml", "outputs/multitask_baseline/best.pt", 528.4810606060606), "pw_200": ("configs/brb_ablation_round7a/brb_pw_200.yaml", "outputs/brb_ablation_round7a/pw_200/best.pt", 200.0), "pw_100": ("configs/brb_ablation_round7a/brb_pw_100.yaml", "outputs/brb_ablation_round7a/pw_100/best.pt", 100.0), "pw_50": ("configs/brb_ablation_round7a/brb_pw_50.yaml", "outputs/brb_ablation_round7a/pw_50/best.pt", 50.0), "pw_25": ("configs/brb_ablation_round7a/brb_pw_25.yaml", "outputs/brb_ablation_round7a/pw_25/best.pt", 25.0)}
    all_rows_for_figures: dict[str, list[dict[str, Any]]] = {}
    comparison_rows = []
    transition_rows = []
    effect_all = []
    peak_all = []
    fixed_threshold_all = []
    calibrated_values: dict[str, float] = {}
    for experiment, (config_path, checkpoint_path, positive_weight) in experiments.items():
        config = load_yaml_config(config_path); checkpoint = resolve_repo_path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {experiment}: {checkpoint}")
        out = root / experiment; out.mkdir(parents=True, exist_ok=True)
        model = _load_model(config, checkpoint)
        validation = _make_records(model, "splits/multitask_val.txt", config)
        # Threshold calibration is validation-only and is independent of test evaluation.
        calibration, calibration_rows = _calibrate(validation, out, len(mapping)); calibrated_values[experiment] = calibration
        _attach_variant_metrics(validation, calibration, len(mapping))
        validation_summary = _summarize(validation, mapping)
        _write_json(out / "validation_summary.json", validation_summary)
        training_dir = resolve_repo_path(config["paths"]["output_dir"]) if experiment != "pw_reciprocal" else REPO_ROOT / "outputs/multitask_baseline"
        _write_json(out / "validation_epoch_diagnostics.json", _epoch_diagnostics(training_dir))
        test_splits = {"test_pour": "splits/multitask_test_pour.txt", "test_pp": "splits/multitask_test_pp.txt", "test_wipe": "splits/multitask_test_wipe.txt"}
        rows_by_split = {split: _make_records(model, split_path, config) for split, split_path in test_splits.items()}
        all_test = [row for rows in rows_by_split.values() for row in rows]
        _attach_variant_metrics(all_test, calibration, len(mapping))
        for split, rows in rows_by_split.items():
            _attach_variant_metrics(rows, calibration, len(mapping))
            _write_json(out / f"{split}_summary.json", _summarize(rows, mapping))
        all_summary = _summarize(all_test, mapping)
        _write_json(out / "all_test_summary.json", all_summary)
        _write_json(out / "test_wipe_w4_summary.json", _summarize([row for row in rows_by_split["test_wipe"] if row["entry"] == "test/wipe/w4"], mapping))
        all_rows_for_figures[experiment] = all_test
        effects_path = out / "refinement_effect_by_trajectory.csv"; _refinement_effect({"validation": validation, **rows_by_split}, experiment, effects_path)
        with effects_path.open(encoding="utf-8", newline="") as handle: effect_rows = list(csv.DictReader(handle))
        effect_all.extend(effect_rows)
        peak_rows = []
        for split_name, split_rows in (("validation", validation), *rows_by_split.items(), ("all_test", all_test)):
            for row in split_rows:
                for threshold in THRESHOLDS:
                    diagnostic = _peak_diagnostics(row, threshold)
                    peak_rows.append({"experiment": experiment, "split": split_name, "task": row["task"], "trajectory": row["entry"], "threshold": threshold, **{key: (json.dumps(value) if isinstance(value, list) else value) for key, value in diagnostic.items()}})
        _write_csv(out / "peak_diagnostics.csv", peak_rows)
        peak_all.extend(peak_rows)
        transition_rows.extend(_transition_rows(all_test, experiment, mapping))
        threshold_rows = []
        for split_name, split_rows in (("validation", validation), *rows_by_split.items(), ("all_test", all_test)):
            fixed_rows = _fixed_threshold_rows(split_rows, experiment, split_name)
            threshold_rows.extend(fixed_rows)
            fixed_threshold_all.extend(fixed_rows)
        _write_csv(out / "fixed_threshold_metrics.csv", threshold_rows)
        training_dir = resolve_repo_path(config["paths"]["output_dir"]) if experiment != "pw_reciprocal" else root
        training_summary_path = training_dir / "training_summary.json"; training_summary = json.loads(training_summary_path.read_text()) if training_summary_path.is_file() else {"elapsed_seconds": 0}
        diagnostics = _epoch_diagnostics(training_dir) if experiment != "pw_reciprocal" else {"primary": {"epoch": 38, "total_loss": 0.25685593829705167}}
        primary = diagnostics.get("primary", {})
        comparison_rows.append(_model_comparison(experiment, positive_weight, int(primary.get("epoch", 0)), float(primary.get("total_loss", 0.0)), validation_summary, all_summary, calibration, training_summary, checkpoint, effect_rows))
    _write_csv(root / "refinement_effect_by_trajectory.csv", effect_all)
    _write_csv(root / "peak_diagnostics.csv", peak_all)
    _write_csv(root / "fixed_threshold_metrics.csv", fixed_threshold_all)
    _write_csv(root / "per_transition_boundary_metrics.csv", transition_rows)
    _distribution_figures(all_rows_for_figures, root / "figures")
    _representative_figures(all_rows_for_figures, root / "figures", mapping, calibrated_values)
    _write_csv(root / "model_comparison.csv", comparison_rows)
    _write_json(root / "evaluation_manifest.json", {"experiments": list(experiments), "thresholds": list(THRESHOLDS), "boundary_tolerances": list(TOLERANCES), "official_threshold": 0.5, "calibration_selection": "validation only", "raw_asb_uses_brb": False, "official_refinement_uses_predicted_brb_boundaries": True, "oracle_refinement_uses_ground_truth_boundaries": True, "one_to_one_boundary_matching": True, "figures": [str(path.relative_to(REPO_ROOT)) for path in sorted((root / "figures").rglob("*.png"))]})
    print(json.dumps({"experiments": list(experiments), "comparison": str(root / "model_comparison.csv"), "figures": len(list((root / "figures").rglob("*.png")))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
