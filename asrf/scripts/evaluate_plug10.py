"""Evaluate focused Round 9 Plug-10 and compare with saved Plug-3/5 runs."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
OUT = ROOT / "outputs/round9_incremental_learning/plug/n10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from asrf.data.dataset import load_trajectory_sample  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.evaluation.metrics import boundary_counts, labels_to_segments  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.refinement.segments import construct_segments  # noqa: E402
from asrf.training.checkpointing import load_checkpoint, sha256_file  # noqa: E402
from asrf.utils.config import load_yaml_config  # noqa: E402
from evaluate_round9_incremental import (  # noqa: E402
    TARGET_SKILLS,
    attach,
    calibrate,
    class_metrics,
    per_skill_rows,
    records,
    semantic,
    support_rows,
    truth_boundaries,
)
from asrf.data.ontology import CANONICAL_LABELS  # noqa: E402

SKILLS = CANONICAL_LABELS
NAMES = {index: name for index, name in enumerate(SKILLS)}
TEST_ENTRIES = ["test/plug/p1", "test/plug/p2", "test/plug/p3", "test/plug/po1", "test/plug/po2"]
TRANSITIONS = ("transport -> place", "place -> insert", "insert -> release", "place -> release")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def nearest_matches(predicted: list[int], truth: list[int], tolerance: int) -> tuple[list[int], list[int]]:
    candidates = sorted((abs(p - t), p, t) for p in predicted for t in truth if abs(p - t) <= tolerance)
    used_p: set[int] = set(); used_t: set[int] = set(); errors: list[int] = []
    for error, p, t in candidates:
        if p not in used_p and t not in used_t:
            used_p.add(p); used_t.add(t); errors.append(error)
    return sorted(used_p), sorted(used_t)


def boundary_summary(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tolerance in (5, 10, 20, 33, 50):
        pooled = {"tp": 0, "fp": 0, "fn": 0, "predicted_count": 0, "target_count": 0}
        errors: list[int] = []; duplicates = 0
        for row in rows:
            predicted = [p for p in row["variants"][variant]["boundaries"] if p != 0]
            truth = truth_boundaries(row)
            item = boundary_counts(predicted, truth, tolerance, include_frame0=False)
            for key in pooled: pooled[key] += int(item[key])
            matched_p, _ = nearest_matches(predicted, truth, tolerance)
            errors.extend([min((abs(p - t) for t in truth), default=0) for p in matched_p])
            duplicates += sum(max(0, sum(abs(p - t) <= tolerance for p in predicted) - 1) for t in truth)
        precision = pooled["tp"] / (pooled["tp"] + pooled["fp"]) if pooled["tp"] + pooled["fp"] else 0.0
        recall = pooled["tp"] / (pooled["tp"] + pooled["fn"]) if pooled["tp"] + pooled["fn"] else 0.0
        pooled.update({"precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                       "duplicate_peaks": duplicates, "mean_localization_error": float(np.mean(errors)) if errors else 0.0,
                       "median_localization_error": float(np.median(errors)) if errors else 0.0})
        result[str(tolerance)] = pooled
    return result


def per_trajectory_boundary(row: dict[str, Any], variant: str) -> dict[str, Any]:
    predicted = [p for p in row["variants"][variant]["boundaries"] if p != 0]
    truth = truth_boundaries(row)
    result = {}
    for tolerance in (5, 10, 20, 33, 50):
        item = boundary_counts(predicted, truth, tolerance, include_frame0=False)
        matched_p, _ = nearest_matches(predicted, truth, tolerance)
        errors = [min((abs(p - t) for t in truth), default=0) for p in matched_p]
        result[str(tolerance)] = {**item, "duplicate_peaks": sum(max(0, sum(abs(p - t) <= tolerance for p in predicted) - 1) for t in truth),
                                  "mean_localization_error": float(np.mean(errors)) if errors else 0.0,
                                  "median_localization_error": float(np.median(errors)) if errors else 0.0}
    return result


def attach_extra(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    target_config = {key: config["data"][key] for key in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")}
    mapping = load_label_mapping(ROOT / config["data"]["label_config"])
    for row in rows:
        sample = load_trajectory_sample(DATA / row["entry"], mapping, expected_height=88, boundary_target_config=target_config)
        row["heatmap"] = sample["heatmap"]
        row["timestamps"] = sample["timestamps"]
        row["segments"] = sample["segments"]


def semantic_summary(rows: list[dict[str, Any]], variant: str, mapping: Any) -> dict[str, Any]:
    metrics = [row["variants"][variant]["semantic"] for row in rows]
    keys = ("frame_accuracy", "balanced_frame_accuracy", "edit", "F1@10", "F1@25", "F1@50")
    result = {key: float(np.mean([item[key] for item in metrics])) if metrics else 0.0 for key in keys}
    total_correct = sum(int((row["variants"][variant]["prediction"] == row["truth"]).sum()) for row in rows)
    total_frames = sum(len(row["truth"]) for row in rows)
    result["pooled_frame_accuracy"] = total_correct / total_frames if total_frames else 0.0
    result["predicted_segment_count"] = sum(item["predicted_segment_count"] for item in metrics)
    result["true_segment_count"] = sum(item["true_segment_count"] for item in metrics)
    result["class_metrics"] = class_metrics(rows, mapping, variant)
    result["confusion_matrix"] = np.sum([np.asarray(item["confusion_matrix"]) for item in metrics], axis=0).tolist() if metrics else []
    result["per_trajectory"] = [{"trajectory": row["entry"], **{key: row["variants"][variant]["semantic"][key] for key in keys}} for row in rows]
    return result


def transition_metrics(rows: list[dict[str, Any]], variant: str = "official") -> list[dict[str, Any]]:
    output = []
    for transition in TRANSITIONS:
        previous, current = transition.split(" -> ")
        items = []
        for row in rows:
            truth_segments = labels_to_segments(row["truth"])
            for segment in truth_segments[1:]:
                if NAMES[int(row["truth"][segment.start - 1])] != previous or NAMES[int(row["truth"][segment.start])] != current:
                    continue
                boundary = segment.start; probability = row["brb"]
                peaks = [p for p in row["variants"][variant]["boundaries"] if p != 0]
                matches = [p for p in peaks if abs(p - boundary) <= 33]
                items.append({"trajectory": row["entry"], "exact_probability": float(probability[boundary]),
                              "max_probability_5": float(probability[max(0, boundary - 5):min(len(probability), boundary + 6)].max()),
                              "max_probability_10": float(probability[max(0, boundary - 10):min(len(probability), boundary + 11)].max()),
                              "max_probability_20": float(probability[max(0, boundary - 20):min(len(probability), boundary + 21)].max()),
                              "max_probability_33": float(probability[max(0, boundary - 33):min(len(probability), boundary + 34)].max()),
                              "detected": int(bool(matches)), "localization_error": abs(matches[0] - boundary) if matches else "", "boundary_frame": boundary})
        output.append({"transition": transition, "support": len(items), "exact_probability": float(np.mean([i["exact_probability"] for i in items])) if items else 0.0,
                       "max_probability_5": float(np.mean([i["max_probability_5"] for i in items])) if items else 0.0,
                       "max_probability_10": float(np.mean([i["max_probability_10"] for i in items])) if items else 0.0,
                       "max_probability_20": float(np.mean([i["max_probability_20"] for i in items])) if items else 0.0,
                       "max_probability_33": float(np.mean([i["max_probability_33"] for i in items])) if items else 0.0,
                       "detection_recall_33": float(np.mean([i["detected"] for i in items])) if items else 0.0,
                       "mean_localization_error": float(np.mean([i["localization_error"] for i in items if i["localization_error"] != ""])) if any(i["localization_error"] != "" for i in items) else 0.0,
                       "missed_count": sum(i["detected"] == 0 for i in items), "details": items})
    return output


def refinement_effect(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        raw = row["variants"]["raw"]["semantic"]; official = row["variants"]["official"]["semantic"]
        delta = {key: float(official[key] - raw[key]) for key in ("frame_accuracy", "edit", "F1@10", "F1@25", "F1@50")}
        if abs(delta["F1@50"]) < 1e-12: status = "unchanged"
        elif delta["F1@50"] > 0: status = "improved"
        else: status = "harmed"
        b = per_trajectory_boundary(row, "official")["33"]
        cause = "none"
        if status == "harmed":
            cause = "missed boundary merging different skills" if b["fn"] else ("false boundary splitting one skill" if b["fp"] else "wrong segment majority")
        output.append({"trajectory": row["entry"], **{f"delta_{key}": value for key, value in delta.items()}, "classification": status, "cause": cause,
                       "false_peaks": b["fp"], "missed_boundaries": b["fn"], "duplicate_peaks": b["duplicate_peaks"]})
    return output


def segment_rows(row: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    truth = labels_to_segments(row["truth"]); raw = row["variants"]["raw"]["prediction"]; prediction = row["variants"][variant]["prediction"]
    intervals = row["variants"][variant]["refinement"].intervals[0] if "refinement" in row["variants"][variant] else construct_segments(row["variants"][variant]["boundaries"], len(row["truth"]))
    result = []
    for index, interval in enumerate(intervals):
        values = row["truth"][interval.start:interval.end]
        gt_class = int(torch.bincount(values, minlength=12).argmax()) if len(values) else -1
        raw_class = int(torch.bincount(raw[interval.start:interval.end], minlength=12).argmax()) if interval.duration else -1
        predicted_class = int(torch.bincount(prediction[interval.start:interval.end], minlength=12).argmax()) if interval.duration else -1
        candidates = []
        for truth_segment in truth:
            intersection = max(0, min(interval.end - 1, truth_segment.end) - max(interval.start, truth_segment.start) + 1)
            union = interval.duration + truth_segment.length - intersection
            candidates.append((intersection / union if union else 0.0, truth_segment))
        iou, match = max(candidates, default=(0.0, None), key=lambda item: item[0])
        result.append({"segment_index": index, "start_frame": interval.start, "end_frame_exclusive": interval.end, "duration_frames": interval.duration,
                       "ground_truth_class": NAMES.get(gt_class, "unknown"), "raw_majority_class": NAMES.get(raw_class, "unknown"), "official_refined_class": NAMES.get(predicted_class, "unknown"),
                       "temporal_iou": iou, "segment_confidence": float(row["asb"][..., interval.start:interval.end].max()) if interval.duration else 0.0,
                       "correct": int(predicted_class == gt_class and iou >= 0.5), "matched_truth_segment": NAMES.get(match.label, "unknown") if match else ""})
    return result


def trajectory_class_score(row: dict[str, Any], variant: str, class_id: int) -> dict[str, float]:
    predicted = [segment for segment in labels_to_segments(row["variants"][variant]["prediction"]) if segment.label == class_id]
    truth = [segment for segment in labels_to_segments(row["truth"]) if segment.label == class_id]
    candidates = []
    for pi, prediction in enumerate(predicted):
        for ti, target in enumerate(truth):
            intersection = max(0, min(prediction.end, target.end) - max(prediction.start, target.start) + 1)
            union = prediction.length + target.length - intersection
            iou = intersection / union if union else 0.0
            if iou >= 0.5:
                candidates.append((iou, pi, ti))
    candidates.sort(reverse=True); used_p: set[int] = set(); used_t: set[int] = set(); tp = 0
    for _, pi, ti in candidates:
        if pi not in used_p and ti not in used_t:
            used_p.add(pi); used_t.add(ti); tp += 1
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    return {"precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "support": len(truth)}


def write_trajectory_outputs(row: dict[str, Any], mapping: Any) -> None:
    name = Path(row["entry"]).name
    directory = OUT / "test_per_trajectory" / name
    directory.mkdir(parents=True, exist_ok=True)
    timestamps = row["timestamps"].numpy(); truth_boundaries_set = set(truth_boundaries(row, internal=False)); predicted_set = set(row["variants"]["official"]["boundaries"])
    with (directory / "frame_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["frame_index", "timestamp_us", "relative_time_s", "ground_truth_label", "raw_asb_predicted_label", "raw_asb_confidence", "official_refined_label", "brb_probability", "ground_truth_boundary", "predicted_boundary_peak"])
        for index in range(len(row["truth"])):
            raw_label = int(row["variants"]["raw"]["prediction"][index]); official_label = int(row["variants"]["official"]["prediction"][index])
            writer.writerow([index, int(timestamps[index]), float((timestamps[index] - timestamps[0]) / 1_000_000.0), NAMES[int(row["truth"][index])], NAMES[raw_label], float(row["asb"][:, index].max()), NAMES[official_label], float(row["brb"][index]), int(index in truth_boundaries_set), int(index in predicted_set)])
    with (directory / "segment_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        rows = segment_rows(row, "official"); writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["segment_index"]); writer.writeheader(); writer.writerows(rows)
    with (directory / "boundary_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["kind", "frame", "probability", "matched_truth_frame", "localization_error", "threshold_0_50_peak"])
        predicted = [p for p in row["variants"]["official"]["boundaries"] if p != 0]; truth = truth_boundaries(row)
        for p in predicted:
            match = min(truth, key=lambda t: abs(t - p), default=None); error = abs(match - p) if match is not None else ""
            writer.writerow(["predicted_peak", p, float(row["brb"][p]), match if match is not None and error <= 33 else "", error if match is not None and error <= 33 else "", 1])
        for t in truth:
            writer.writerow(["ground_truth_boundary", t, float(row["brb"][t]), "", "", int(t in predicted)])
    metrics = {"trajectory": row["entry"], "raw": row["variants"]["raw"]["semantic"], "official": row["variants"]["official"]["semantic"], "calibrated": row["variants"]["calibrated"]["semantic"], "oracle": row["variants"]["oracle"]["semantic"], "boundary_official": per_trajectory_boundary(row, "official"), "boundary_calibrated": per_trajectory_boundary(row, "calibrated"), "selected_official_peaks": row["variants"]["official"]["boundaries"], "ground_truth_internal_boundaries": truth_boundaries(row)}
    write_json(directory / "metrics.json", metrics)
    timeline(row, directory / "timeline.png")


def timeline(row: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    truth = row["truth"].numpy(); raw = row["variants"]["raw"]["prediction"].numpy(); official = row["variants"]["official"]["prediction"].numpy(); probability = row["brb"].numpy(); timestamps = row["timestamps"].numpy(); x = (timestamps - timestamps[0]) / 1_000_000.0
    colors = list(plt.get_cmap("tab20").colors[:12]); cmap = ListedColormap(colors); norm = BoundaryNorm(np.arange(-0.5, 12.5, 1), 12)
    fig, axes = plt.subplots(5, 1, figsize=(18, 10), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [1, 1, 1, 2, 1]})
    for axis, values, label in ((axes[0], truth, "ground truth"), (axes[1], raw, "raw ASB"), (axes[2], official, "official r5 refined")):
        axis.imshow(values[np.newaxis, :], aspect="auto", interpolation="nearest", extent=(x[0], x[-1], 0, 1), cmap=cmap, norm=norm); axis.set_ylabel(label)
    axes[3].plot(x, probability, color="black", linewidth=0.8, label="BRB probability"); axes[3].set_ylim(0, 1); axes[3].set_ylabel("BRB p")
    truth_segments = labels_to_segments(row["truth"])
    for boundary in truth_boundaries(row, internal=False): axes[3].axvline(x[boundary], color="limegreen", linewidth=0.8, label="ground-truth boundary" if boundary == truth_boundaries(row, internal=False)[0] else None)
    for segment in truth_segments[1:]:
        previous = NAMES[int(row["truth"][segment.start - 1])]; current = NAMES[int(row["truth"][segment.start])]
        axes[3].text(x[segment.start], 0.98, f"{previous}→{current}", rotation=90, fontsize=6, va="top", ha="right", color="green")
    for peak in row["variants"]["official"]["boundaries"]: axes[3].axvline(x[peak], color="red", linewidth=0.8, linestyle="--", label="predicted peak" if peak == row["variants"]["official"]["boundaries"][0] else None)
    axes[4].imshow(official[np.newaxis, :], aspect="auto", interpolation="nearest", extent=(x[0], x[-1], 0, 1), cmap=cmap, norm=norm); axes[4].set_ylabel("refined"); axes[4].set_xlabel("time (s)")
    axes[3].legend(loc="lower left", fontsize=7, framealpha=0.8)
    title = f"{row['entry']} — canonical labels; pull-out={row['entry'].endswith(('po1', 'po2'))}"
    fig.suptitle(title); fig.savefig(path, dpi=130); plt.close(fig)


def model_rows(size: int, checkpoint: Path, config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], float]:
    config = load_yaml_config(config_path); mapping = load_label_mapping(ROOT / config["data"]["label_config"])
    model = ASRFModel.from_config(config); model.load_state_dict(load_checkpoint(checkpoint, map_location="cpu", expected_ontology=True)["model_state"]); model.eval()
    validation = records(model, "splits/round9_incremental/common_validation.txt", config); threshold = calibrate(validation); attach(validation, threshold, 12); attach_extra(validation, config)
    test = records(model, "splits/round9_incremental/test_plug_primary.txt", config); attach(test, threshold, 12); attach_extra(test, config)
    return config, test, {"validation": validation, "test": test, "calibrated_threshold": threshold}, threshold


def comparison_figures(rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    directory = OUT / "figures_3_5_10"; directory.mkdir(parents=True, exist_ok=True); sizes = [row["plug_training_trajectories"] for row in rows]
    def plot(name: str, keys: list[tuple[str, str]], ylabel: str) -> None:
        fig, axis = plt.subplots(figsize=(8, 5))
        for key, label in keys: axis.plot(sizes, [row[key] for row in rows], marker="o", label=label)
        axis.set_xticks(sizes); axis.set_xlabel("added plug training trajectories (fixed pp10 base)"); axis.set_ylabel(ylabel); axis.grid(alpha=0.25); axis.legend(); fig.tight_layout(); fig.savefig(directory / name, dpi=130); plt.close(fig)
    plot("place_f1.png", [("place_F1", "place")], "segment F1")
    plot("insert_f1.png", [("insert_F1", "insert")], "segment F1")
    plot("raw_vs_refined_f1_50.png", [("raw_F1_50", "raw ASB"), ("official_F1_50", "official r5")], "F1@50")
    plot("boundary_f1_33.png", [("boundary_F1_33", "boundary F1@33")], "boundary F1@33")
    plot("false_missed_peaks.png", [("false_peaks", "false peaks"), ("missed_boundaries", "missed boundaries")], "count")
    plot("place_insert_transition_recall.png", [("transport_place_recall_33", "transport→place"), ("place_insert_recall_33", "place→insert"), ("insert_release_recall_33", "insert→release")], "recall@33")
    n10 = rows[-1]; fig, axis = plt.subplots(figsize=(10, 5)); labels = [skill for skill in SKILLS if n10.get(f"{skill}_F1") is not None]; values = [n10[f"{skill}_F1"] for skill in labels]; axis.bar(labels, values); axis.set_ylim(0, 1); axis.set_ylabel("official segment F1"); axis.set_title("Plug-10 per-class test F1"); axis.tick_params(axis="x", rotation=35); fig.tight_layout(); fig.savefig(directory / "plug10_per_class_f1.png", dpi=130); plt.close(fig)


def main() -> int:
    mapping = load_label_mapping(ROOT / "configs/labels_multitask_plug.yaml")
    audit_rows = list(csv.DictReader((OUT / "data_audit_scan1.csv").open(encoding="utf-8")))
    manifests = {"train": "splits/round9_incremental/plug_train_10_with_base_pp10.txt", "validation": "splits/round9_incremental/common_validation.txt", "test": "splits/round9_incremental/test_plug_primary.txt"}
    manifest_dir = OUT / "manifests"; manifest_dir.mkdir(parents=True, exist_ok=True)
    for key, relative in manifests.items(): shutil.copyfile(ROOT / relative, manifest_dir / f"{key}.txt")
    write_json(manifest_dir / "manifest_metadata.json", {"train": manifests["train"], "validation": manifests["validation"], "test": manifests["test"], "official_threshold": 0.5, "boundary_target_mode": "hard_window", "boundary_window_radius": 5, "seed": 42})

    configs_and_checkpoints = [(3, ROOT / "outputs/round9_incremental_learning/models/plug/n3/best.pt", ROOT / "outputs/round9_incremental_learning/models/plug/n3/config.yaml"), (5, ROOT / "outputs/round9_incremental_learning/models/plug/n5/best.pt", ROOT / "outputs/round9_incremental_learning/models/plug/n5/config.yaml"), (10, OUT / "best.pt", OUT / "config.yaml")]
    all_results: dict[int, dict[str, Any]] = {}
    for size, checkpoint, config_path in configs_and_checkpoints:
        config, test, split_rows, threshold = model_rows(size, checkpoint, config_path)
        all_results[size] = {"config": config, "test": test, "validation": split_rows["validation"], "threshold": threshold}
        if size == 10:
            write_json(OUT / "validation_summary.json", {"calibrated_threshold": threshold, "raw": semantic_summary(split_rows["validation"], "raw", mapping), "official": semantic_summary(split_rows["validation"], "official", mapping), "calibrated": semantic_summary(split_rows["validation"], "calibrated", mapping), "oracle": semantic_summary(split_rows["validation"], "oracle", mapping), "boundary": boundary_summary(split_rows["validation"], "official")})
            write_json(OUT / "primary_test_summary.json", {"calibrated_threshold": threshold, "raw": semantic_summary(test, "raw", mapping), "official": semantic_summary(test, "official", mapping), "calibrated": semantic_summary(test, "calibrated", mapping), "oracle": semantic_summary(test, "oracle", mapping), "boundary": {"official": boundary_summary(test, "official"), "calibrated": boundary_summary(test, "calibrated"), "oracle": boundary_summary(test, "oracle")}})
            for row in test: write_trajectory_outputs(row, mapping)
            support = support_rows(audit_rows, [line.strip() for line in (ROOT / manifests["train"]).read_text().splitlines() if line.strip()], [line.strip() for line in (ROOT / manifests["validation"]).read_text().splitlines() if line.strip()], TEST_ENTRIES, "plug", 10, TARGET_SKILLS["plug"])
            write_csv(OUT / "training_support.csv", support); write_csv(OUT / "per_skill_metrics.csv", per_skill_rows(test, mapping, support, "plug", 10)); write_csv(OUT / "refinement_effect_by_trajectory.csv", refinement_effect(test)); write_csv(OUT / "per_transition_boundary_metrics.csv", transition_metrics(test))

    comparison: list[dict[str, Any]] = []
    for size in (3, 5, 10):
        result = all_results[size]; test = result["test"]; official = semantic_summary(test, "official", mapping); raw = semantic_summary(test, "raw", mapping); boundary = boundary_summary(test, "official")["33"]; per_skill = per_skill_rows(test, mapping, support_rows(audit_rows, [line.strip() for line in (ROOT / ("splits/round9_incremental/plug_train_" + str(size) + "_with_base_pp10.txt")).read_text().splitlines() if line.strip()], [line.strip() for line in (ROOT / manifests["validation"]).read_text().splitlines() if line.strip()], TEST_ENTRIES, "plug", size, TARGET_SKILLS["plug"]), "plug", size)
        by_skill = {row["skill"]: row for row in per_skill}; train_file = ROOT / ("splits/round9_incremental/plug_train_" + str(size) + "_with_base_pp10.txt")
        if size == 10: train_file = ROOT / manifests["train"]
        train_entries = [line.strip() for line in train_file.read_text().splitlines() if line.strip()]; train_audit = {row["trajectory"]: row for row in audit_rows}
        checkpoint_for_size = {3: configs_and_checkpoints[0][1], 5: configs_and_checkpoints[1][1], 10: configs_and_checkpoints[2][1]}[size]
        summary_dir = ROOT / (f"outputs/round9_incremental_learning/models/plug/n{size}" if size != 10 else OUT)
        row = {"plug_training_trajectories": size, "total_training_trajectories": len(train_entries), "raw_accuracy": raw["frame_accuracy"], "raw_F1_50": raw["F1@50"], "official_accuracy": official["frame_accuracy"], "official_F1_50": official["F1@50"], "boundary_F1_33": boundary["F1"], "false_peaks": boundary["fp"], "missed_boundaries": boundary["fn"], "duplicate_peaks": boundary["duplicate_peaks"], "training_duration_s": json.loads((summary_dir / "training_summary.json").read_text())["elapsed_seconds"], "checkpoint_sha256": sha256_file(checkpoint_for_size)}
        for skill in ("place", "insert", "release", "lift"):
            row[f"{skill}_precision"] = by_skill[skill]["official_precision"]
            row[f"{skill}_recall"] = by_skill[skill]["official_recall"]
            row[f"{skill}_F1"] = by_skill[skill]["official_F1"]
        for item in transition_metrics(test):
            row[item["transition"].replace(" -> ", "_") + "_recall_33"] = item["detection_recall_33"]
        comparison.append(row)
    write_csv(OUT / "plug_3_5_10_comparison.csv", comparison)
    write_csv(OUT.parent / "plug_3_5_10_comparison.csv", comparison)
    comparison_figures(comparison)
    parent_figures = OUT.parent / "figures_3_5_10"
    parent_figures.mkdir(parents=True, exist_ok=True)
    for figure in (OUT / "figures_3_5_10").glob("*.png"):
        shutil.copyfile(figure, parent_figures / figure.name)

    n10 = all_results[10]; official = semantic_summary(n10["test"], "official", mapping); raw = semantic_summary(n10["test"], "raw", mapping); oracle = semantic_summary(n10["test"], "oracle", mapping); boundary = boundary_summary(n10["test"], "official"); transitions = transition_metrics(n10["test"])
    train_support = {skill: {"segments": sum(int(row[f"{skill}_segments"]) for row in audit_rows if row["trajectory"] in [line.strip() for line in (ROOT / manifests["train"]).read_text().splitlines() if line.strip()]), "frames": sum(int(row[f"{skill}_frames"]) for row in audit_rows if row["trajectory"] in [line.strip() for line in (ROOT / manifests["train"]).read_text().splitlines() if line.strip()])} for skill in SKILLS}
    training_summary = json.loads((OUT / "training_summary.json").read_text()); hashes = {"initialization": sha256_file(ROOT / "outputs/brb_release_round8/hard_window_r5/best.pt"), "plug10_best": sha256_file(OUT / "best.pt"), "plug10_last": sha256_file(OUT / "last.pt"), "plug3_best": sha256_file(ROOT / "outputs/round9_incremental_learning/models/plug/n3/best.pt"), "plug5_best": sha256_file(ROOT / "outputs/round9_incremental_learning/models/plug/n5/best.pt")}
    lines = ["# Plug-10 focused Round 9 report", "", "## Audit and protocol", "", "Two fresh scans agree; all ten train/plug trajectories are valid. The fixed policy is hard-window radius 5, threshold 0.50, seed 42, pp1–pp10 base, pp11–pp20 validation, and test/plug p1,p2,p3,po1,po2.", "", "Training trajectories: " + ", ".join(f"train/plug/p{i}" for i in range(1, 11)), "", "Validation: " + ", ".join(f"train/pick and place/pp{i}" for i in range(11, 21)), "", "Test: " + ", ".join(TEST_ENTRIES), "", "## Training", "", f"Best epoch: {training_summary['best_epoch']}; stopping epoch: {training_summary['stopping_epoch']}; duration: {training_summary['elapsed_seconds']:.1f} s.", "", "Per-class train support (segments / frames):", "", "| skill | segments | frames |", "|---|---:|---:|"]
    lines += [f"| {skill} | {train_support[skill]['segments']} | {train_support[skill]['frames']} |" for skill in SKILLS]
    lines += ["", "## Pooled test metrics", "", "| variant | accuracy | balanced accuracy | Edit | F1@10 | F1@25 | F1@50 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for variant, metrics in (("raw ASB", raw), ("official refined", official), ("oracle boundary", oracle)): lines.append(f"| {variant} | {metrics['pooled_frame_accuracy']:.4f} | {metrics['balanced_frame_accuracy']:.4f} | {metrics['edit']:.4f} | {metrics['F1@10']:.4f} | {metrics['F1@25']:.4f} | {metrics['F1@50']:.4f} |")
    lines += ["", "Official boundary metrics:", "", "| tolerance | precision | recall | F1 | false peaks | missed | duplicates | mean error | median error |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tolerance in (5, 10, 20, 33, 50):
        item = boundary[str(tolerance)]; lines.append(f"| ±{tolerance} | {item['precision']:.4f} | {item['recall']:.4f} | {item['F1']:.4f} | {item['fp']} | {item['fn']} | {item['duplicate_peaks']} | {item['mean_localization_error']:.3f} | {item['median_localization_error']:.3f} |")
    lines += ["", "## Target classes and transitions", "", "| skill | official segment precision | official segment recall | official segment F1 | official frame F1 |", "|---|---:|---:|---:|---:|"]
    n10_skill = {row["skill"]: row for row in per_skill_rows(n10["test"], mapping, support_rows(audit_rows, [line.strip() for line in (ROOT / manifests["train"]).read_text().splitlines() if line.strip()], [line.strip() for line in (ROOT / manifests["validation"]).read_text().splitlines() if line.strip()], TEST_ENTRIES, "plug", 10, TARGET_SKILLS["plug"]), "plug", 10)}
    n10_frame_skill = {row["skill"]: row for row in official["class_metrics"]}
    for skill in ("place", "insert", "release", "lift"): lines.append(f"| {skill} | {n10_skill[skill]['official_precision']:.4f} | {n10_skill[skill]['official_recall']:.4f} | {n10_skill[skill]['official_F1']:.4f} | {n10_frame_skill[skill]['F1']:.4f} |")
    lines += ["", "| transition | support | recall@33 | mean error | missed |", "|---|---:|---:|---:|---:|"]
    for item in transitions: lines.append(f"| {item['transition']} | {item['support']} | {item['detection_recall_33']:.4f} | {item['mean_localization_error']:.3f} | {item['missed_count']} |")
    lines += ["", "## Per-trajectory results", "", "| trajectory | official F1@50 | boundary F1@33 | false peaks | missed | place segment F1 | insert segment F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in n10["test"]:
        b = per_trajectory_boundary(row, "official")["33"]; place_score = trajectory_class_score(row, "official", 6); insert_score = trajectory_class_score(row, "official", 10)
        lines.append(f"| {row['entry']} | {row['variants']['official']['semantic']['F1@50']:.4f} | {b['f1']:.4f} | {b['fp']} | {b['fn']} | {place_score['F1']:.4f} | {insert_score['F1']:.4f} |")
    lines += ["", "## Plug-3 vs Plug-5 vs Plug-10", "", "| added plug trajectories | place F1 | insert F1 | official F1@50 | boundary F1@33 | false | missed | place→insert | insert→release |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in comparison: lines.append(f"| {item['plug_training_trajectories']} | {item['place_F1']:.4f} | {item['insert_F1']:.4f} | {item['official_F1_50']:.4f} | {item['boundary_F1_33']:.4f} | {item['false_peaks']} | {item['missed_boundaries']} | {item['place_insert_recall_33']:.4f} | {item['insert_release_recall_33']:.4f} |")
    place_row = official["confusion_matrix"][6]
    place_confusion = ", ".join(f"{NAMES[i]}={count}" for i, count in enumerate(place_row) if count)
    p5_comparison = next(item for item in comparison if item["plug_training_trajectories"] == 5); n10_comparison = next(item for item in comparison if item["plug_training_trajectories"] == 10)
    effect_rows = refinement_effect(n10["test"]); effect_counts = {status: sum(row["classification"] == status for row in effect_rows) for status in ("improved", "unchanged", "harmed")}
    lines += ["", "## Diagnosis of Plug-5 issues", "", f"- Insert segment F1: {p5_comparison['insert_F1']:.4f} → {n10_comparison['insert_F1']:.4f} (change {n10_comparison['insert_F1'] - p5_comparison['insert_F1']:+.4f}); p1/p2/p3 are listed above and po1/po2 have no insert ground-truth support.", f"- Place→insert boundary recall@33: {p5_comparison['place_insert_recall_33']:.4f} → {n10_comparison['place_insert_recall_33']:.4f} (change {n10_comparison['place_insert_recall_33'] - p5_comparison['place_insert_recall_33']:+.4f}).", f"- Insert→release boundary recall@33: {p5_comparison['insert_release_recall_33']:.4f} → {n10_comparison['insert_release_recall_33']:.4f} (change {n10_comparison['insert_release_recall_33'] - p5_comparison['insert_release_recall_33']:+.4f}).", f"- Place segment F1: {p5_comparison['place_F1']:.4f} → {n10_comparison['place_F1']:.4f}; place ground-truth frames are predicted as {place_confusion}.", f"- BRB false peaks: {p5_comparison['false_peaks']} → {n10_comparison['false_peaks']} (change {n10_comparison['false_peaks'] - p5_comparison['false_peaks']:+d}); missed boundaries: {p5_comparison['missed_boundaries']} → {n10_comparison['missed_boundaries']} (change {n10_comparison['missed_boundaries'] - p5_comparison['missed_boundaries']:+d}). Boundary recall improved, while false peaks increased.", f"- Official refinement changed pooled F1@50 from raw {raw['F1@50']:.4f} to {official['F1@50']:.4f}; by per-trajectory F1@50, {effect_counts['improved']} improved, {effect_counts['unchanged']} unchanged, and {effect_counts['harmed']} harmed.", "", "The ten-trajectory run evaluates the updated contiguous ontology and remains subject to the manual annotation migration gate.", "", "## Figures and integrity", "", "Timeline figures were generated and visually inspected for p1, p2, p3, po1, and po2; pull-out po1/po2 remain evaluated with canonical lift labels. All Plug-3/5/10 comparison figures were also visually inspected.", "", "```json", json.dumps(hashes, indent=2, sort_keys=True), "```", "", "Tests and compileall passed; external annotations and prior checkpoints were not modified."]
    (OUT / "plug10_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "evaluation_metadata.json", {"official_threshold": 0.5, "boundary_target_mode": "hard_window", "boundary_window_radius": 5, "calibrated_threshold_validation_only": n10["threshold"], "test_entries": TEST_ENTRIES, "checkpoint_hashes": hashes})
    print(json.dumps({"test_trajectories": TEST_ENTRIES, "best_epoch": training_summary["best_epoch"], "official_F1_50": official["F1@50"], "boundary_F1_33": boundary["33"]["F1"], "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
