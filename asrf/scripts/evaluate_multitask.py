#!/usr/bin/env python
"""Freeze validation postprocessing, then evaluate the multi-task ASRF model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asrf.data.dataset import MultiTaskTrajectoryDataset  # noqa: E402
from asrf.data.labels import LabelMapping, load_label_mapping  # noqa: E402
from asrf.evaluation.metrics import (  # noqa: E402
    aggregate_trajectory_metrics,
    boundary_counts,
    boundary_indices_from_labels,
    trajectory_metrics,
)
from asrf.models import ASRFModel  # noqa: E402
from asrf.refinement.majority_vote import _vote_one  # noqa: E402
from asrf.refinement.refine import ASRFRefinementOutput, refine_asrf_predictions  # noqa: E402
from asrf.refinement.segments import construct_segments  # noqa: E402
from asrf.training.checkpointing import load_checkpoint, sha256_file  # noqa: E402
from asrf.utils.config import load_yaml_config, resolve_repo_path  # noqa: E402


TOLERANCES = (10, 20, 30, 33)
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.50, 0.951, 0.05))


def _names(mapping: LabelMapping) -> dict[int, str]:
    return {value: name for name, value in mapping.items()}


def _collapsed(values: torch.Tensor) -> list[int]:
    result: list[int] = []
    for value in values.tolist():
        value = int(value)
        if not result or result[-1] != value:
            result.append(value)
    return result


def _label_transitions(values: list[int], names: dict[int, str]) -> list[str]:
    return [f"{names.get(first, str(first))} -> {names.get(second, str(second))}" for first, second in zip(values, values[1:])]


def _record(model: ASRFModel, dataset: MultiTaskTrajectoryDataset, index: int, device: torch.device) -> dict[str, Any]:
    sample = dataset[index]
    heatmap = sample["heatmap"].unsqueeze(0).to(device)
    mask = sample["valid_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(heatmap, valid_mask=mask)
    length = int(sample["heatmap"].shape[-1])
    return {
        "sample": sample,
        "asb": output.asb_stage_probabilities[-1][0, :, :length].cpu(),
        "brb": output.brb_stage_probabilities[-1][0, 0, :length].cpu(),
        "truth": sample["labels"].cpu(),
        "boundary_targets": sample["boundary_targets"].cpu(),
        "task": sample.get("task_name", "unknown"),
        "entry": sample.get("relative_path", sample["trajectory_id"]),
    }


def _oracle_refinement(asb: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, list[int], Any, Any]:
    length = int(truth.numel())
    boundaries = boundary_indices_from_labels(truth, include_frame0=True)
    intervals = construct_segments(boundaries, length)
    refined, diagnostics = _vote_one(asb[:, :length], intervals, voting="majority")
    return refined, boundaries, intervals, diagnostics


def _matched_errors(predicted: list[int], target: list[int], tolerance: int) -> list[int]:
    candidates = sorted((abs(int(p) - int(t)), pi, ti) for pi, p in enumerate(predicted) for ti, t in enumerate(target) if abs(int(p) - int(t)) <= tolerance)
    used_p: set[int] = set()
    used_t: set[int] = set()
    errors: list[int] = []
    for error, pi, ti in candidates:
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            errors.append(error)
    return errors


def _variant_metrics(prediction: torch.Tensor, truth: torch.Tensor, predicted_boundaries: list[int], targets: torch.Tensor) -> dict[str, Any]:
    truth_boundaries = torch.where(targets > 0.5)[0].tolist()
    result: dict[str, Any] = {"metrics": trajectory_metrics(prediction, truth), "predicted_boundary_count": len(predicted_boundaries), "boundaries": predicted_boundaries, "truth_boundaries": truth_boundaries}
    for tolerance in TOLERANCES:
        for scope, include_frame0 in (("including_frame0", True), ("internal_only", False)):
            result[f"boundary_{tolerance}_{scope}"] = boundary_counts(predicted_boundaries, truth_boundaries, tolerance, include_frame0=include_frame0)
    result["matched_boundary_error@33"] = _matched_errors([p for p in predicted_boundaries if p != 0], [t for t in truth_boundaries if t != 0], 33)
    return result


def _class_rows(prediction: torch.Tensor, truth: torch.Tensor, mapping: LabelMapping, *, task: str, variant: str) -> list[dict[str, Any]]:
    rows = []
    for class_name, class_id in sorted(mapping.items(), key=lambda item: item[1]):
        pred = prediction == class_id
        target = truth == class_id
        tp = int((pred & target).sum())
        fp = int((pred & ~target).sum())
        fn = int((~pred & target).sum())
        support = int(target.sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"task": task, "variant": variant, "class": class_name, "class_id": class_id, "tp": tp, "fp": fp, "fn": fn, "support": support, "precision": precision, "recall": recall, "f1": f1})
    return rows


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_trajectory_outputs(root: Path, record: dict[str, Any], variants: dict[str, Any], mapping: LabelMapping, calibrated_threshold: float) -> None:
    sample = record["sample"]
    truth = record["truth"]
    length = int(truth.numel())
    names = _names(mapping)
    target_dir = root / str(record["task"]) / Path(str(record["entry"])).name
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(target_dir / "ground_truth.csv", ["frame", "label_id", "label"], [[i, int(truth[i]), names[int(truth[i])] ] for i in range(length)])
    for name, item in (("raw_asb_predictions", variants["raw"]), ("official_asrf_predictions", variants["official"]), ("calibrated_asrf_predictions", variants["calibrated"]), ("oracle_boundary_predictions", variants["oracle"])):
        prediction = item["prediction"]
        _write_csv(target_dir / f"{name}.csv", ["frame", "label_id", "label"], [[i, int(prediction[i]), names.get(int(prediction[i]), f"class_{int(prediction[i])}")] for i in range(length)])
    _write_csv(target_dir / "brb_probabilities.csv", ["frame", "probability"], [[i, float(record["brb"][i])] for i in range(length)])
    for name, item in (("official_boundaries", variants["official"]), ("calibrated_boundaries", variants["calibrated"]), ("ground_truth_boundaries", variants["oracle"])):
        _write_csv(target_dir / f"{name}.csv", ["boundary_index"], [[value] for value in item["boundaries"]])
    diagnostics = variants["calibrated"].get("refinement")
    if diagnostics is not None:
        rows = []
        for interval, diagnostic in zip(diagnostics.intervals[0], diagnostics.segment_diagnostics[0]):
            rows.append([interval.start, interval.end, interval.duration, json.dumps(diagnostic.class_counts, sort_keys=True), diagnostic.selected_class, diagnostic.majority_fraction])
        _write_csv(target_dir / "segment_diagnostics.csv", ["start", "end_exclusive", "duration", "class_counts", "selected_class", "majority_fraction"], rows)
    keep = {"metrics", "predicted_boundary_count", "boundaries", "truth_boundaries", "matched_boundary_error@33"} | {f"boundary_{tolerance}_{scope}" for tolerance in TOLERANCES for scope in ("including_frame0", "internal_only")}
    metrics = {name: {key: value for key, value in item.items() if key in keep} for name, item in variants.items()}
    metrics["ground_truth_collapsed_sequence"] = _collapsed(truth)
    for name, item in variants.items():
        metrics[name]["collapsed_sequence"] = _collapsed(item["prediction"])
    (target_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_trajectory(target_dir / "annotation_vs_predictions.png", sample["heatmap"].numpy(), truth.numpy(), record["asb"].numpy(), record["brb"].numpy(), variants, names, record["entry"], calibrated_threshold)


def _contiguous_segments(values: np.ndarray) -> list[tuple[int, int, int]]:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if not len(values):
        return []
    starts = [0]
    starts.extend(int(index) for index in np.flatnonzero(values[1:] != values[:-1]) + 1)
    segments: list[tuple[int, int, int]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(values)
        segments.append((start, end, int(values[start])))
    return segments


def _draw_segment_names(axis: Any, values: np.ndarray, names: dict[int, str], *, fontsize: float = 7.0) -> None:
    """Place canonical names in sufficiently wide contiguous colored blocks."""
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    length = len(values)
    for start, end, class_id in _contiguous_segments(values):
        full_name = names.get(class_id, f"class_{class_id}")
        # Keep labels readable at the exact temporal scale; narrow segments
        # remain identifiable through the canonical legend and color block.
        minimum_width = max(80, 22 * len(full_name))
        if end - start < minimum_width:
            continue
        display_name = "pour_rec" if full_name == "pour_recover" and end - start < 600 else full_name
        axis.text((start + end) / 2.0, 0.5, display_name, ha="center", va="center", fontsize=fontsize, color="white", fontweight="bold", clip_on=True, bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.0, "edgecolor": "none"})


def _draw_boundary_lines(axis: Any, variants: dict[str, Any]) -> None:
    for peak in variants["oracle"].get("boundaries", []):
        axis.axvline(peak, color="lime", linewidth=0.7, alpha=0.8)
    for peak in variants["official"].get("boundaries", []):
        axis.axvline(peak, color="red", linewidth=0.7, alpha=0.8)
    for peak in variants["calibrated"].get("boundaries", []):
        axis.axvline(peak, color="orange", linewidth=0.7, alpha=0.8)


def _plot_trajectory(path: Path, heatmap: np.ndarray, truth: np.ndarray, asb: np.ndarray, brb: np.ndarray, variants: dict[str, Any], names: dict[int, str], entry: str, calibrated_threshold: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    length = heatmap.shape[-1]
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:max(10, len(names))])
    norm = BoundaryNorm(np.arange(-0.5, max(9, len(names)) + 0.5, 1), cmap.N)
    figure, axes = plt.subplots(10, 1, figsize=(20, 16), sharex=True, constrained_layout=False, gridspec_kw={"height_ratios": [5] + [0.75] * 8 + [1.2]})
    axes[0].imshow(np.moveaxis(heatmap, 0, -1), origin="upper", aspect="auto", interpolation="nearest", extent=(0, length, heatmap.shape[1], 0))
    axes[0].set_ylabel("CITR")
    panels = [(truth, "ground truth"), (variants["raw"]["prediction"].numpy(), "raw ASB"), (variants["official"]["prediction"].numpy(), "official 0.50"), (variants["calibrated"]["prediction"].numpy(), f"calibrated {calibrated_threshold:.2f}"), (variants["oracle"]["prediction"].numpy(), "oracle")]
    for axis, (values, label) in zip(axes[1:6], panels):
        axis.imshow(np.asarray(values)[np.newaxis, :], origin="lower", aspect="auto", interpolation="nearest", extent=(0, length, 0, 1), cmap=cmap, norm=norm)
        axis.set_ylabel(label)
        _draw_segment_names(axis, np.asarray(values), names)
        axis.set_yticks([])
    for axis in axes[1:6]:
        _draw_boundary_lines(axis, variants)
    axes[6].plot(np.arange(length) + 0.5, brb, linewidth=0.8, color="black")
    axes[6].axhline(0.5, color="gray", linewidth=0.7, linestyle="--", label="BRB threshold 0.50")
    axes[6].axhline(calibrated_threshold, color="orange", linewidth=0.7, linestyle=":", label=f"BRB threshold {calibrated_threshold:.2f}")
    axes[6].set_ylim(0, 1)
    axes[6].set_ylabel("BRB p")
    for peak in variants["official"]["boundaries"]:
        axes[7].axvline(peak, color="red", linewidth=0.7)
    for peak in variants["calibrated"]["boundaries"]:
        axes[7].axvline(peak, color="orange", linewidth=0.7)
    for peak in variants["oracle"].get("boundaries", []):
        axes[7].axvline(peak, color="lime", linewidth=0.7)
    axes[7].set_ylabel("peaks")
    axes[7].set_yticks([])
    confidence = asb.max(axis=0)
    axes[8].plot(np.arange(length) + 0.5, confidence, color="navy", linewidth=0.8)
    axes[8].set_ylim(0, 1)
    axes[8].set_ylabel("confidence")
    axes[9].plot(np.arange(length) + 0.5, asb.argmax(axis=0), color="black", linewidth=0.5)
    axes[9].set_ylabel("ASB id")
    axes[9].set_xlabel("frame index; exact temporal width preserved; display downsampling factor = 1")
    for axis in axes:
        axis.set_xlim(0, length)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.0, label="ground-truth boundaries"),
        Line2D([0], [0], color="red", linewidth=1.0, label="threshold 0.50 peaks"),
        Line2D([0], [0], color="orange", linewidth=1.0, label=f"threshold {calibrated_threshold:.2f} peaks"),
        Line2D([0], [0], color="lime", linewidth=1.0, label="oracle boundaries (non-deployable)"),
        Line2D([0], [0], color="gray", linestyle="--", label="threshold 0.50"),
        Line2D([0], [0], color="orange", linestyle=":", label=f"threshold {calibrated_threshold:.2f}"),
    ]
    figure.subplots_adjust(top=0.87, bottom=0.09, hspace=0.28)
    figure.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, fontsize=8)
    figure.text(0.01, 0.035, "Raw ASB does not use BRB. Official ASRF uses threshold 0.50. Calibrated ASRF uses threshold 0.90 selected on validation only. Oracle uses ground-truth boundaries. Confidence is max ASB probability. ASB ID is final-stage argmax. Canonical names are shown inside sufficiently wide segments; narrow blocks are represented by color and the legend.", fontsize=8, va="bottom")
    figure.suptitle(f"{entry} — canonical labels: " + ", ".join(f"{key}:{value}" for key, value in sorted(names.items())), y=0.995)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _summarize(records: list[dict[str, Any]], mapping: LabelMapping, *, task: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"task": task, "trajectory_count": len(records), "variants": {}, "per_class": []}
    for variant in ("raw", "official", "calibrated", "oracle"):
        rows = [record["variants"][variant]["metrics"] for record in records]
        mean_metrics = aggregate_trajectory_metrics(rows)
        std_metrics = {key: float(np.std([float(row[key]) for row in rows])) if rows else 0.0 for key in ("frame_accuracy", "edit_score", "f1@10", "f1@25", "f1@50")}
        confusion = np.zeros((len(mapping), len(mapping)), dtype=np.int64)
        for record in records:
            truth_values = record["truth"].numpy()
            predicted_values = record["variants"][variant]["prediction"].numpy()
            np.add.at(confusion, (truth_values, predicted_values), 1)
        common_confusions = []
        for true_id in range(len(mapping)):
            for predicted_id in range(len(mapping)):
                if true_id != predicted_id and confusion[true_id, predicted_id]:
                    common_confusions.append({"true_class": _names(mapping)[true_id], "predicted_class": _names(mapping)[predicted_id], "count": int(confusion[true_id, predicted_id])})
        common_confusions.sort(key=lambda row: (-row["count"], row["true_class"], row["predicted_class"]))
        summary["variants"][variant] = {"macro_trajectory": mean_metrics, "trajectory_std": std_metrics, "frame_weighted_accuracy": float(sum(int(record["variants"][variant]["prediction"].eq(record["truth"]).sum()) for record in records) / max(1, sum(len(record["truth"]) for record in records))), "confusion_matrix": confusion.tolist(), "common_confusions": common_confusions[:20]}
        for tolerance in TOLERANCES:
            for scope in ("including_frame0", "internal_only"):
                counters = {key: sum(int(record["variants"][variant][f"boundary_{tolerance}_{scope}"][key]) for record in records) for key in ("tp", "fp", "fn", "predicted_count", "target_count")}
                precision = counters["tp"] / (counters["tp"] + counters["fp"]) if counters["tp"] + counters["fp"] else 0.0
                recall = counters["tp"] / (counters["tp"] + counters["fn"]) if counters["tp"] + counters["fn"] else 0.0
                summary["variants"][variant][f"boundary_{tolerance}_{scope}"] = {**counters, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    for variant in ("raw", "official", "calibrated", "oracle"):
        aggregate: dict[int, dict[str, int]] = {class_id: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for class_id in range(len(mapping))}
        for record in records:
            for row in _class_rows(record["variants"][variant]["prediction"], record["truth"], mapping, task=task, variant=variant):
                counts = aggregate[int(row["class_id"])]
                for key in ("tp", "fp", "fn", "support"):
                    counts[key] += int(row[key])
        for class_name, class_id in sorted(mapping.items(), key=lambda item: item[1]):
            counts = aggregate[class_id]
            precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
            recall = counts["tp"] / counts["support"] if counts["support"] else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            summary["per_class"].append({"task": task, "variant": variant, "class": class_name, "class_id": class_id, **counts, "precision": precision, "recall": recall, "f1": f1})
    return summary


def _calibrate(model: ASRFModel, dataset: MultiTaskTrajectoryDataset, device: torch.device, mapping: LabelMapping, output_dir: Path) -> tuple[float, list[dict[str, Any]]]:
    records = [_record(model, dataset, i, device) for i in range(len(dataset))]
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        by_task: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        total_fp = 0
        for record in records:
            refinement = refine_asrf_predictions(record["asb"].unsqueeze(0), record["brb"].view(1, 1, -1), torch.ones(1, len(record["truth"]), dtype=torch.bool), threshold=threshold)
            item = _variant_metrics(refinement.refined_labels[0], record["truth"], list(refinement.selected_boundaries[0]), record["boundary_targets"])
            item["refinement"] = refinement
            by_task[str(record["task"])].append(item)
        task_f1 = []
        task_refined_f1 = []
        task_refined_acc = []
        for task, items in by_task.items():
            tp = sum(int(item["boundary_33_internal_only"]["tp"]) for item in items)
            fp = sum(int(item["boundary_33_internal_only"]["fp"]) for item in items)
            fn = sum(int(item["boundary_33_internal_only"]["fn"]) for item in items)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            task_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            task_refined_f1.append(float(np.mean([item["metrics"]["f1@50"] for item in items])))
            task_refined_acc.append(float(np.mean([item["metrics"]["frame_accuracy"] for item in items])))
            total_fp += fp
        rows.append({"threshold": threshold, "macro_internal_boundary_f1@33": float(np.mean(task_f1)) if task_f1 else 0.0, "false_positive_boundaries": total_fp, "macro_refined_f1@50": float(np.mean(task_refined_f1)) if task_refined_f1 else 0.0, "macro_refined_accuracy": float(np.mean(task_refined_acc)) if task_refined_acc else 0.0})
    chosen = sorted(rows, key=lambda row: (-row["macro_internal_boundary_f1@33"], row["false_positive_boundaries"], -row["macro_refined_f1@50"], -row["macro_refined_accuracy"], -row["threshold"]))[0]["threshold"]
    with (output_dir / "validation_threshold_calibration.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "frozen_postprocessing.json").write_text(json.dumps({"official_threshold": 0.5, "calibrated_threshold": chosen, "objective": "macro internal-boundary F1@+-33 on multitask validation", "tie_breaking": ["fewer false positive boundaries", "higher macro refined F1@50", "higher macro refined accuracy", "higher threshold"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return float(chosen), records


def _evaluate_test(model: ASRFModel, config: dict[str, Any], mapping: LabelMapping, calibrated_threshold: float, output_dir: Path, device: torch.device) -> dict[str, Any]:
    data_root = Path(config["data"]["dataset_root"])
    split_specs = {"pour": "splits/multitask_test_pour.txt", "pp": "splits/multitask_test_pp.txt", "wipe": "splits/multitask_test_wipe.txt"}
    all_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for task, split in split_specs.items():
        dataset = MultiTaskTrajectoryDataset(data_root, resolve_repo_path(split), resolve_repo_path(config["data"]["label_config"]), expected_height=88, allow_test=True)
        records: list[dict[str, Any]] = []
        for i in range(len(dataset)):
            record = _record(model, dataset, i, device)
            raw_prediction = record["asb"].argmax(dim=0)
            official_refinement = refine_asrf_predictions(record["asb"].unsqueeze(0), record["brb"].view(1, 1, -1), torch.ones(1, len(record["truth"]), dtype=torch.bool), threshold=0.5)
            calibrated_refinement = refine_asrf_predictions(record["asb"].unsqueeze(0), record["brb"].view(1, 1, -1), torch.ones(1, len(record["truth"]), dtype=torch.bool), threshold=calibrated_threshold)
            oracle_prediction, oracle_boundaries, oracle_intervals, oracle_diagnostics = _oracle_refinement(record["asb"], record["truth"])
            variants = {
                "raw": {"prediction": raw_prediction, **_variant_metrics(raw_prediction, record["truth"], [], record["boundary_targets"])},
                "official": {"prediction": official_refinement.refined_labels[0], "refinement": official_refinement, **_variant_metrics(official_refinement.refined_labels[0], record["truth"], list(official_refinement.selected_boundaries[0]), record["boundary_targets"])},
                "calibrated": {"prediction": calibrated_refinement.refined_labels[0], "refinement": calibrated_refinement, **_variant_metrics(calibrated_refinement.refined_labels[0], record["truth"], list(calibrated_refinement.selected_boundaries[0]), record["boundary_targets"])},
                "oracle": {"prediction": oracle_prediction, "boundaries": oracle_boundaries, "intervals": oracle_intervals, "diagnostics": oracle_diagnostics, **_variant_metrics(oracle_prediction, record["truth"], oracle_boundaries, record["boundary_targets"])},
            }
            record["variants"] = variants
            records.append(record)
            _write_trajectory_outputs(output_dir / "test", record, variants, mapping, calibrated_threshold)
        summaries[task] = _summarize(records, mapping, task=task)
        (output_dir / "test" / f"{task}_summary.json").write_text(json.dumps(summaries[task], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        all_records.extend(records)
    summaries["all_tasks"] = _summarize(all_records, mapping, task="all_tasks")
    (output_dir / "test" / "all_tasks_summary.json").write_text(json.dumps(summaries["all_tasks"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    names = _names(mapping)
    sequence_rows = []
    transition_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in all_records:
        truth_sequence = _collapsed(record["truth"])
        row = {"task": record["task"], "trajectory": record["entry"], "ground_truth": truth_sequence}
        for variant in ("raw", "official", "calibrated"):
            sequence = _collapsed(record["variants"][variant]["prediction"])
            row[variant] = sequence
            for transition in _label_transitions(sequence, names):
                transition_counts[f"{variant}:{record['task']}"][transition] += 1
        for transition in _label_transitions(truth_sequence, names):
            transition_counts[f"ground_truth:{record['task']}"][transition] += 1
        sequence_rows.append(row)
    (output_dir / "sequence_bias_diagnostics.json").write_text(json.dumps({"sequences": sequence_rows, "transition_counts": {key: dict(value) for key, value in transition_counts.items()}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summaries


def _pour_comparison(config: dict[str, Any], multitask_model: ASRFModel, calibrated_threshold: float, output_dir: Path, device: torch.device) -> None:
    pour_config = load_yaml_config("configs/pour_asrf_train.yaml")
    mapping = load_label_mapping(resolve_repo_path(pour_config["data"]["label_config"]))
    model = ASRFModel.from_config(pour_config).to(device)
    checkpoint = load_checkpoint(REPO_ROOT / "outputs/pour_baseline/best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = MultiTaskTrajectoryDataset(Path(config["data"]["dataset_root"]), REPO_ROOT / "splits/multitask_test_pour.txt", resolve_repo_path(pour_config["data"]["label_config"]), expected_height=88, allow_test=True)
    rows: list[dict[str, Any]] = []
    multitask_mapping = load_label_mapping(resolve_repo_path(config["data"]["label_config"]))
    for index in range(len(dataset)):
        record = _record(model, dataset, index, device)
        raw = record["asb"].argmax(dim=0)
        refined = refine_asrf_predictions(record["asb"].unsqueeze(0), record["brb"].view(1, 1, -1), torch.ones(1, len(record["truth"]), dtype=torch.bool), threshold=0.5)
        target = record["boundary_targets"]
        for name, prediction in {"pour_only_raw": raw, "pour_only_official": refined.refined_labels[0]}.items():
            variant = _variant_metrics(prediction, record["truth"], list(refined.selected_boundaries[0]) if name.endswith("official") else [], target)
            per_class = _class_rows(prediction, record["truth"], mapping, task="pour", variant=name)
            rows.append({"model": name, "trajectory": record["entry"], **variant["metrics"], "internal_boundary_f1@33": variant["boundary_33_internal_only"]["f1"], "predicted_boundary_count": variant["predicted_boundary_count"], "per_class_recall": json.dumps({row["class"]: row["recall"] for row in per_class}, sort_keys=True)})

    for index in range(len(dataset)):
        record = _record(multitask_model, MultiTaskTrajectoryDataset(Path(config["data"]["dataset_root"]), REPO_ROOT / "splits/multitask_test_pour.txt", resolve_repo_path(config["data"]["label_config"]), expected_height=88, allow_test=True), index, device)
        raw = record["asb"].argmax(dim=0)
        official = refine_asrf_predictions(record["asb"].unsqueeze(0), record["brb"].view(1, 1, -1), torch.ones(1, len(record["truth"]), dtype=torch.bool), threshold=0.5)
        calibrated = refine_asrf_predictions(record["asb"].unsqueeze(0), record["brb"].view(1, 1, -1), torch.ones(1, len(record["truth"]), dtype=torch.bool), threshold=calibrated_threshold)
        for name, prediction, boundaries in (("multitask_raw", raw, []), ("multitask_official", official.refined_labels[0], list(official.selected_boundaries[0])), ("multitask_calibrated", calibrated.refined_labels[0], list(calibrated.selected_boundaries[0]))):
            variant = _variant_metrics(prediction, record["truth"], boundaries, record["boundary_targets"])
            per_class = _class_rows(prediction, record["truth"], multitask_mapping, task="pour", variant=name)
            rows.append({"model": name, "trajectory": record["entry"], **variant["metrics"], "internal_boundary_f1@33": variant["boundary_33_internal_only"]["f1"], "predicted_boundary_count": variant["predicted_boundary_count"], "per_class_recall": json.dumps({row["class"]: row["recall"] for row in per_class}, sort_keys=True)})
    path = output_dir / "pour_only_vs_multitask.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multitask_asrf_train.yaml")
    parser.add_argument("--checkpoint", default="outputs/multitask_baseline/best.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    device = torch.device(args.device)
    output_dir = resolve_repo_path(config["paths"]["output_dir"])
    mapping = load_label_mapping(resolve_repo_path(config["data"]["label_config"]))
    model = ASRFModel.from_config(config).to(device)
    payload = load_checkpoint(resolve_repo_path(args.checkpoint), map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    val_dataset = MultiTaskTrajectoryDataset(Path(config["data"]["dataset_root"]), resolve_repo_path(config["data"]["val_split"]), resolve_repo_path(config["data"]["label_config"]), expected_height=88)
    calibrated_threshold, _ = _calibrate(model, val_dataset, device, mapping, output_dir)
    summaries = _evaluate_test(model, config, mapping, calibrated_threshold, output_dir, device)
    _pour_comparison(config, model, calibrated_threshold, output_dir, device)
    print(json.dumps({"calibrated_threshold": calibrated_threshold, "test_trajectory_counts": {key: value["trajectory_count"] for key, value in summaries.items()}}, indent=2))
    print(f"multitask_best_sha256={sha256_file(resolve_repo_path(args.checkpoint))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
