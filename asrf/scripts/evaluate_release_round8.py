"""Evaluate all Round 8 target-shape models with frozen post-processing."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_round7a as base  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.training.checkpointing import sha256_file  # noqa: E402
from asrf.utils.config import load_yaml_config, resolve_repo_path  # noqa: E402


OUT = REPO_ROOT / "outputs/brb_release_round8"
EXPERIMENTS = {
    "baseline_single_frame": "configs/brb_release_round8/baseline_single_frame.yaml",
    "hard_window_r5": "configs/brb_release_round8/hard_window_r5.yaml",
    "hard_window_r10": "configs/brb_release_round8/hard_window_r10.yaml",
    "hard_window_r20": "configs/brb_release_round8/hard_window_r20.yaml",
    "gaussian_s5": "configs/brb_release_round8/gaussian_s5.yaml",
    "gaussian_s10": "configs/brb_release_round8/gaussian_s10.yaml",
    "gaussian_s20": "configs/brb_release_round8/gaussian_s20.yaml",
}
TEST_SPLITS = {"test_pour": "splits/multitask_test_pour.txt", "test_pp": "splits/multitask_test_pp.txt", "test_wipe": "splits/multitask_test_wipe.txt"}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=base._json) + "\n", encoding="utf-8")


def _fixed_threshold_rows(rows: list[dict[str, Any]], experiment: str, split: str, num_classes: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for threshold in base.THRESHOLDS:
        semantic: list[dict[str, Any]] = []; boundary: list[dict[str, Any]] = []; selected_count = 0
        for row in rows:
            refinement = base.refine_asrf_predictions(row["asb"].unsqueeze(0), row["brb"].view(1, 1, -1), torch.ones(1, len(row["truth"]), dtype=torch.bool), threshold=threshold, voting="majority")
            semantic.append(base._semantic_metrics(refinement.refined_labels[0], row["truth"], num_classes))
            boundary.append(base.boundary_counts([p for p in refinement.selected_boundaries[0] if p != 0], base._truth_boundaries(row, internal=True), 33, include_frame0=False))
            selected_count += len(refinement.selected_boundaries[0])
        tp = sum(int(item["tp"]) for item in boundary); fp = sum(int(item["fp"]) for item in boundary); fn = sum(int(item["fn"]) for item in boundary)
        precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
        result.append({"experiment": experiment, "split": split, "threshold": threshold, "macro_accuracy": float(np.mean([item["frame_accuracy"] for item in semantic])) if semantic else 0.0, "macro_balanced_accuracy": float(np.mean([item["balanced_frame_accuracy"] for item in semantic])) if semantic else 0.0, "macro_edit": float(np.mean([item["edit"] for item in semantic])) if semantic else 0.0, "macro_F1@10": float(np.mean([item["F1@10"] for item in semantic])) if semantic else 0.0, "macro_F1@25": float(np.mean([item["F1@25"] for item in semantic])) if semantic else 0.0, "macro_F1@50": float(np.mean([item["F1@50"] for item in semantic])) if semantic else 0.0, "boundary_precision_33": precision, "boundary_recall_33": recall, "boundary_F1_33": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "false_positive_peaks": fp, "missed_boundaries": fn, "selected_peaks": selected_count})
    return result


def _class_rows(rows: list[dict[str, Any]], variant: str, split: str, mapping: Any) -> list[dict[str, Any]]:
    values = base._class_metrics(rows, mapping, variant)
    for row in values: row["split"] = split
    return values


def _comparison(experiment: str, config: dict[str, Any], val_summary: dict[str, Any], test_summary: dict[str, Any], test_rows: list[dict[str, Any]], calibration: float, training_summary: dict[str, Any], checkpoint: Path, effects: list[dict[str, Any]], validation_diagnostics: dict[str, Any]) -> dict[str, Any]:
    def metric(summary: dict[str, Any], variant: str, key: str) -> float:
        return float(summary["variants"][variant]["macro_trajectory"].get(key, 0.0))
    boundary = test_summary["boundary"]["official"]["internal_33"]["pooled"]
    release = next(row for row in test_summary["variants"]["official"]["per_class"] if row["class"] == "release")
    transition = next((row for row in test_summary.get("per_transition", []) if row["transition"] == "place -> release"), {})
    peak_diagnostics = [base._peak_diagnostics(row, 0.5) for row in test_rows]
    duplicate_peaks = sum(int(item["duplicate_peaks_around_one_boundary"]) for item in peak_diagnostics)
    localization_errors = [float(item["mean_absolute_matched_boundary_error"]) for item in peak_diagnostics if item["matched_predicted_boundaries"]]
    target_mode = config["data"]["boundary_target_mode"]
    configured_weight = config["loss"].get("boundary_positive_weight")
    if configured_weight is None:
        stats_path = resolve_repo_path(config["paths"]["output_dir"]) / "boundary_statistics.json"
        if stats_path.is_file():
            configured_weight = json.loads(stats_path.read_text())["configured_positive_weight_train"]
    return {
        "experiment": experiment, "target_mode": target_mode, "window_radius": config["data"].get("boundary_window_radius", 0), "gaussian_sigma": config["data"].get("boundary_gaussian_sigma", 0.0), "positive_weight": configured_weight,
        "best_epoch": validation_diagnostics.get("primary", {}).get("epoch", 0), "validation_total_loss": validation_diagnostics.get("primary", {}).get("total_loss", 0.0),
        "raw_accuracy": metric(test_summary, "raw", "frame_accuracy"), "raw_edit": metric(test_summary, "raw", "edit"), "raw_F1_50": metric(test_summary, "raw", "F1@50"),
        "official_accuracy": metric(test_summary, "official", "frame_accuracy"), "official_edit": metric(test_summary, "official", "edit"), "official_F1_50": metric(test_summary, "official", "F1@50"),
        "boundary_precision_33": boundary["precision"], "boundary_recall_33": boundary["recall"], "boundary_F1_33": boundary["f1"], "false_peaks": boundary["fp"], "missed_boundaries": boundary["fn"], "duplicate_peaks": duplicate_peaks, "mean_localization_error": float(np.mean(localization_errors)) if localization_errors else 0.0,
        "release_precision": release["precision"], "release_recall": release["recall"], "release_F1": release["f1"], "place_release_boundary_recall": transition.get("detection_recall_threshold_0.50", 0.0),
        "calibrated_threshold": calibration, "calibrated_F1_50": metric(test_summary, "calibrated", "F1@50"), "calibrated_boundary_F1_33": test_summary["boundary"]["calibrated"]["internal_33"]["pooled"]["f1"],
        "oracle_accuracy": metric(test_summary, "oracle", "frame_accuracy"), "oracle_F1_50": metric(test_summary, "oracle", "F1@50"), "refinement_improved_count": sum(row["classification"] == "refinement improved" for row in effects), "refinement_harmed_count": sum(row["classification"] == "refinement harmed" for row in effects), "training_duration_s": float(training_summary.get("elapsed_seconds", 0.0)), "checkpoint_sha256": sha256_file(checkpoint),
        "validation_raw_F1_50": metric(val_summary, "raw", "F1@50"), "validation_refined_F1_50": metric(val_summary, "official", "F1@50"), "validation_boundary_F1_33": val_summary["boundary"]["official"]["internal_33"]["pooled"]["f1"], "validation_best_raw_F1_50_epoch": validation_diagnostics.get("best_raw_asb_F1@50", {}).get("epoch", 0), "validation_best_refined_F1_50_epoch": validation_diagnostics.get("best_official_refined_F1@50", {}).get("epoch", 0), "validation_best_boundary_F1_33_epoch": validation_diagnostics.get("best_internal_boundary_F1@33", {}).get("epoch", 0),
    }


def main() -> int:
    mapping = load_label_mapping(REPO_ROOT / "configs/labels_multitask_release.yaml")
    all_test_rows: dict[str, list[dict[str, Any]]] = {}; comparison: list[dict[str, Any]] = []; all_effects: list[dict[str, Any]] = []; all_peaks: list[dict[str, Any]] = []; all_fixed: list[dict[str, Any]] = []; all_transition: list[dict[str, Any]] = []; class_rows: list[dict[str, Any]] = []; calibrated: dict[str, float] = {}
    for experiment, config_relative in EXPERIMENTS.items():
        config = load_yaml_config(config_relative); output = OUT / experiment; output.mkdir(parents=True, exist_ok=True)
        checkpoint = resolve_repo_path(config["paths"]["output_dir"]) / "best.pt"
        if not checkpoint.is_file(): raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        model = base._load_model(config, checkpoint)
        validation = base._make_records(model, "splits/multitask_val.txt", config)
        threshold, _ = base._calibrate(validation, output, len(mapping)); calibrated[experiment] = threshold
        base._attach_variant_metrics(validation, threshold, len(mapping)); val_summary = base._summarize(validation, mapping); _write_json(output / "validation_summary.json", val_summary)
        train_dir = resolve_repo_path(config["paths"]["output_dir"]); training_summary = json.loads((train_dir / "training_summary.json").read_text()) if (train_dir / "training_summary.json").is_file() else {}
        diagnostics = base._epoch_diagnostics(train_dir); _write_json(output / "validation_epoch_diagnostics.json", diagnostics)
        test_by_split = {name: base._make_records(model, split, config) for name, split in TEST_SPLITS.items()}
        all_test = [row for values in test_by_split.values() for row in values]; base._attach_variant_metrics(all_test, threshold, len(mapping))
        for split, values in test_by_split.items():
            base._attach_variant_metrics(values, threshold, len(mapping)); summary = base._summarize(values, mapping); _write_json(output / f"{split}_summary.json", summary); class_rows.extend(_class_rows(values, "raw", split, mapping)); class_rows.extend(_class_rows(values, "official", split, mapping)); class_rows.extend(_class_rows(values, "calibrated", split, mapping))
        all_summary = base._summarize(all_test, mapping); _write_json(output / "all_test_summary.json", all_summary); _write_json(output / "test_wipe_w4_summary.json", base._summarize([row for row in test_by_split["test_wipe"] if row["entry"] == "test/wipe/w4"], mapping))
        effects_path = output / "refinement_effect_by_trajectory.csv"; base._refinement_effect({"validation": validation, **test_by_split}, experiment, effects_path); effects = list(csv.DictReader(effects_path.open(encoding="utf-8"))); all_effects.extend(effects)
        transition = base._transition_rows(all_test, experiment, mapping); all_transition.extend(transition); all_summary["per_transition"] = transition; _write_json(output / "all_test_summary.json", all_summary)
        fixed = []
        for split_name, rows in (("validation", validation), *test_by_split.items(), ("all_test", all_test)): fixed.extend(_fixed_threshold_rows(rows, experiment, split_name, len(mapping)))
        base._write_csv(output / "fixed_threshold_metrics.csv", fixed); all_fixed.extend(fixed)
        peak_rows = []
        for split_name, rows in (("validation", validation), *test_by_split.items(), ("all_test", all_test)):
            for row in rows:
                for th in base.THRESHOLDS:
                    diagnostic = base._peak_diagnostics(row, th); peak_rows.append({"experiment": experiment, "split": split_name, "task": row["task"], "trajectory": row["entry"], "threshold": th, **{key: json.dumps(value) if isinstance(value, list) else value for key, value in diagnostic.items()}})
        base._write_csv(output / "peak_diagnostics.csv", peak_rows); all_peaks.extend(peak_rows); all_test_rows[experiment] = all_test
        comparison.append(_comparison(experiment, config, val_summary, all_summary, all_test, threshold, training_summary, checkpoint, effects, diagnostics))
    base._write_csv(OUT / "refinement_effect_by_trajectory.csv", all_effects); base._write_csv(OUT / "peak_diagnostics.csv", all_peaks); base._write_csv(OUT / "fixed_threshold_metrics.csv", all_fixed); base._write_csv(OUT / "per_transition_boundary_metrics.csv", all_transition); base._write_csv(OUT / "per_class_metrics.csv", class_rows); base._write_csv(OUT / "model_comparison.csv", comparison)
    base._distribution_figures(all_test_rows, OUT / "figures"); base._representative_figures(all_test_rows, OUT / "figures", mapping, calibrated)
    _write_json(OUT / "evaluation_manifest.json", {"experiments": list(EXPERIMENTS), "thresholds": list(base.THRESHOLDS), "boundary_tolerances": list(base.TOLERANCES), "official_threshold": 0.5, "calibration_selection": "validation only", "raw_asb_uses_brb": False, "official_refinement_uses_predicted_brb_boundaries": True, "oracle_refinement_uses_ground_truth_boundaries": True, "one_to_one_boundary_matching": True, "gaussian_weighting_note": "Reciprocal hard-label weighting is not directly meaningful for soft Gaussian targets; primary Gaussian runs use pos_weight=1 and fixed-50 diagnostics are reserved for secondary runs."})
    print(json.dumps({"experiments": list(EXPERIMENTS), "comparison": str(OUT / "model_comparison.csv"), "figures": len(list((OUT / "figures").rglob("*.png")))}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
