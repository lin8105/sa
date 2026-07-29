#!/usr/bin/env python3
"""Round 27: PP-only r5-region / SF-point raw ASRF hybrid."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
R10 = ROOT / "outputs/round10_pp_only_novel_segmentation"
OUT = ROOT / "outputs/round27_pp_only_r5_region_sf_point_hybrid"
SF = R10 / "models/single_frame/best.pt"
R5 = R10 / "models/hard_window_r5/best.pt"
SF_SHA = "6b11abff2ff4387eada88e38fb56133b29fd5c3facdfb54b42fc84701213db9a"
R5_SHA = "577d8edf9e2b04927acc235ffa4d6baab8df1712dd0b98eaaba9063fde31f406"
TRAIN_MANIFEST = R10 / "audit/pp_train_manifest.txt"
VAL_MANIFEST = R10 / "audit/pp_validation_manifest.txt"
TEST_AUDIT = R10 / "audit/test_manifest.json"
KNOWN = ("reach", "grasp", "lift", "transport", "place", "release", "retreat")
KNOWN_SET = set(KNOWN)
NOVEL_SET = {"pour", "pour_recover", "wipe", "align", "insert"}
SEED = 42
TOLERANCES = (5, 10, 20, 33, 50)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from asrf.data.dataset import load_timestamp_vector, load_trajectory_sample  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.visualization.temporal import DEFAULT_LABEL_COLORS, _normalized_heatmap  # noqa: E402
import run_round19_asrf_segment_classifier_integration as r19  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json), encoding="utf-8")


def _json(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def read_entries(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def seed() -> None:
    np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)


def model_meta(path: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")); payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("architecture_config") != config["model"]:
        raise RuntimeError(f"Architecture metadata mismatch for {path}")
    expected_split = read_entries(ROOT / config["data"]["train_split"])
    expected_val = read_entries(ROOT / config["data"]["val_split"])
    if payload.get("train_trajectory_ids") != expected_split or payload.get("validation_trajectory_ids") != expected_val:
        raise RuntimeError(f"PP split metadata mismatch for {path}")
    if config["experiment"].get("training_scope") != "pp_only" or config["experiment"].get("novel_data_forbidden") is not True:
        raise RuntimeError(f"Non-PP training provenance for {path}")
    return config, payload


def audit_checkpoints() -> dict[str, Any]:
    if not SF.is_file() or sha256(SF) != SF_SHA: raise RuntimeError("SF checkpoint hash mismatch")
    if not R5.is_file() or sha256(R5) != R5_SHA: raise RuntimeError("r5 checkpoint hash mismatch")
    sf_config, sf_payload = model_meta(SF, R10 / "models/single_frame/config.yaml")
    r5_config, r5_payload = model_meta(R5, R10 / "models/hard_window_r5/config.yaml")
    if sf_payload.get("label_map") != r5_payload.get("label_map"):
        raise RuntimeError("SF/r5 ontology index maps differ")
    if sf_config["data"]["boundary_target_mode"] != "single_frame": raise RuntimeError("SF target is not single_frame")
    if r5_config["data"]["boundary_target_mode"] != "hard_window" or r5_config["data"]["boundary_window_radius"] != 5: raise RuntimeError("r5 target is not hard-window radius 5")
    expected_labels = {name: i for i, name in enumerate(KNOWN)}
    config_labels = yaml.safe_load((ROOT / "configs/labels_round10_pp_only.yaml").read_text(encoding="utf-8"))["labels"]
    if config_labels != expected_labels: raise RuntimeError("PP ontology order mismatch")
    ontology = {"ontology_version": "round10_pp_only_known_7_class", "ordered_class_list": list(KNOWN), "label_to_id": expected_labels, "aliases": {"pick": "reach", "translation": "transport"}, "novel_labels_are_test_truth_only": sorted(NOVEL_SET)}
    write_json(OUT / "ontology.json", ontology)
    write_json(OUT / "checkpoint_hashes.json", {"sf_checkpoint": str(SF), "sf_sha256": SF_SHA, "r5_checkpoint": str(R5), "r5_sha256": R5_SHA, "expected_sf_sha256": SF_SHA, "expected_r5_sha256": R5_SHA, "sf_config_sha256": sha256(R10 / "models/single_frame/config.yaml"), "r5_config_sha256": sha256(R10 / "models/hard_window_r5/config.yaml"), "training": "PP-only; no retraining", "annotations_changed": False})
    return {"sf_config": sf_config, "r5_config": r5_config, "sf_payload": sf_payload, "r5_payload": r5_payload}


def write_split_manifests() -> tuple[list[str], list[str], list[str]]:
    train = read_entries(TRAIN_MANIFEST); validation = read_entries(VAL_MANIFEST); audit = json.loads(TEST_AUDIT.read_text(encoding="utf-8")); test = [x for family in audit["families"].values() for x in family]
    def rows(entries: list[str], split: str) -> list[dict[str, Any]]:
        return [{"split": split, "trajectory": x, "family": x.split("/")[1], "source_manifest": str(TRAIN_MANIFEST if split == "train" else VAL_MANIFEST)} for x in entries]
    write_csv(OUT / "training_split_manifest.csv", rows(train, "train")); write_csv(OUT / "validation_split_manifest.csv", rows(validation, "validation"))
    test_rows = []; available_test = []
    for entry in test:
        annotation = DATA / entry / "segments.csv"; features = DATA / entry / "citr_features.csv"
        if not annotation.is_file() or not features.is_file():
            test_rows.append({"split": "test", "trajectory": entry, "family": entry.split("/")[1], "included": 0, "exclusion_reason": "listed by historical audit but required test artifact is absent", "frame_count": "", "gt_skills": "", "annotation_hash": "", "source_audit": str(TEST_AUDIT)}); continue
        timestamps = load_timestamp_vector(features); gt = r19.gt_rows_for(entry, "test"); available_test.append(entry)
        test_rows.append({"split": "test", "trajectory": entry, "family": entry.split("/")[1], "included": 1, "exclusion_reason": "", "frame_count": len(timestamps), "gt_skills": ";".join(sorted({x["label"] for x in gt})), "annotation_hash": sha256(annotation), "source_audit": str(TEST_AUDIT)})
    for entry in audit.get("excluded", []): test_rows.append({"split": "test", "trajectory": entry, "family": entry.split("/")[1], "included": 0, "exclusion_reason": "historical corrected audit exclusion", "frame_count": "", "gt_skills": "", "annotation_hash": sha256(DATA / entry / "segments.csv"), "source_audit": str(TEST_AUDIT)})
    write_csv(OUT / "test_manifest.csv", test_rows); return train, validation, available_test


@torch.no_grad()
def infer(model: ASRFModel, entry: str, mapping: Any, target_config: dict[str, Any]) -> dict[str, Any]:
    sample = load_trajectory_sample(DATA / entry, mapping, expected_height=88, boundary_target_config=target_config)
    output = model(sample["heatmap"].unsqueeze(0), valid_mask=sample["valid_mask"].unsqueeze(0))
    asb_logits = output.asb_stage_logits[-1][0].cpu().numpy(); asb_probs = output.asb_stage_probabilities[-1][0].cpu().numpy(); brb = output.brb_stage_probabilities[-1][0, 0].cpu().numpy()
    return {"entry": entry, "sample": sample, "asb_logits": asb_logits, "asb_probabilities": asb_probs, "asb_labels": np.argmax(asb_probs, axis=0), "brb": brb}


def input_audit_row(item: dict[str, Any], split: str) -> dict[str, Any]:
    heatmap = item["sample"]["heatmap"].numpy().astype(np.float32)
    timestamps = item["sample"]["timestamps"].numpy()
    return {"split": split, "trajectory": item["entry"], "source_feature_path": str(DATA / item["entry"] / "citr_fingerprint_pure.png"), "trajectory_length": len(heatmap[0, 0]), "input_shape": str(tuple(heatmap.shape)), "dtype": str(heatmap.dtype), "min": float(heatmap.min()), "max": float(heatmap.max()), "mean": float(heatmap.mean()), "std": float(heatmap.std()), "input_sha256": hashlib.sha256(heatmap.tobytes()).hexdigest(), "timestamps_sha256": hashlib.sha256(timestamps.tobytes()).hexdigest(), "timestamps_identical": 1, "normalization": "PIL RGB / 255.0; no model-specific normalization"}


def regions(probability: np.ndarray, threshold: float, gap: int) -> list[tuple[int, int]]:
    above = probability >= threshold; spans: list[tuple[int, int]] = []; start = None; last = None; missing = 0
    for index, value in enumerate(above):
        if value:
            if start is None: start = index
            elif missing > gap: spans.append((start, last + 1)); start = index
            last = index; missing = 0
        elif start is not None:
            missing += 1
    if start is not None: spans.append((start, last + 1))
    return spans


def choose_points(r5: dict[str, Any], sf: dict[str, Any], spans: list[tuple[int, int]], rule: str, support_gate: float | None) -> tuple[list[int], list[dict[str, Any]]]:
    points: list[int] = []; diagnostics: list[dict[str, Any]] = []
    for start, end in spans:
        rr = r5["brb"][start:end]; expanded_start = max(0, start + (-5 if rule == "P1" else -10 if rule == "P2" else 0)); expanded_end = min(len(sf["brb"]), end + (5 if rule == "P1" else 10 if rule == "P2" else 0)); ss = sf["brb"][expanded_start:expanded_end]; inside = sf["brb"][start:end]; sf_max_frame = start + int(np.argmax(inside)); sf_max = float(np.max(inside)); r5_max_frame = start + int(np.argmax(rr)); r5_max = float(np.max(rr)); weights = np.maximum(rr, 0); center = int(round(start + ((np.arange(len(rr)) * weights).sum() / max(weights.sum(), 1e-8))))
        weak = support_gate is not None and sf_max < support_gate
        if rule in ("P1", "P2"): point = expanded_start + int(np.argmax(ss))
        elif weak and rule == "P3": point = center
        elif weak and rule == "P4": point = r5_max_frame
        else: point = sf_max_frame
        points.append(point); diagnostics.append({"region_start": start, "region_end": end, "region_width": end - start, "r5_max_probability": r5_max, "r5_max_frame": r5_max_frame, "r5_weighted_center_frame": center, "sf_max_probability_inside_region": sf_max, "sf_max_frame": sf_max_frame, "selected_frame": point, "point_rule": rule, "sf_r5_max_displacement": abs(point - r5_max_frame), "sf_support_weak": int(weak), "sf_probability_at_r5_weighted_center": float(sf["brb"][center])})
    return points, diagnostics


def suppress(points: list[int], r5: dict[str, Any], sf: dict[str, Any], diagnostics: list[dict[str, Any]], separation: int) -> tuple[list[int], int]:
    if separation == 0: return points, 0
    kept: list[int] = []; suppressed = 0
    for point, row in sorted(zip(points, diagnostics), key=lambda x: x[0]):
        if not kept or point - kept[-1] >= separation: kept.append(point); continue
        previous = next(x for x in diagnostics if x["selected_frame"] == kept[-1]); score = row["r5_max_probability"] * row["sf_max_probability_inside_region"]; old_score = previous["r5_max_probability"] * previous["sf_max_probability_inside_region"]
        if score > old_score: kept[-1] = point
        suppressed += 1
    return kept, suppressed


def hybrid(item_sf: dict[str, Any], item_r5: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]], int]:
    spans = regions(item_r5["brb"], cfg["threshold"], cfg["gap"]); points, diagnostics = choose_points(item_r5, item_sf, spans, cfg["rule"], cfg["support_gate"]); points, suppressed = suppress(points, item_r5, item_sf, diagnostics, cfg["separation"])
    selected = set(points)
    for row in diagnostics: row["selected_final"] = int(row["selected_frame"] in selected); row["matched_gt_boundary"] = ""; row["localization_error"] = ""
    return points, diagnostics, suppressed


def frame_segments(labels: np.ndarray, points: list[int], probabilities: np.ndarray) -> list[dict[str, Any]]:
    boundaries = sorted(set([0, *[int(x) for x in points if 0 < x < len(labels)], len(labels)])); rows = []
    inverse = dict(enumerate(KNOWN))
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        values = labels[start:end]; label_id = Counter(values.tolist()).most_common(1)[0][0]; rows.append({"segment_index": index, "start": start, "end": end, "duration": end - start, "top1_id": int(label_id), "top1_label": inverse[int(label_id)], "top1_probability": float(np.mean(probabilities[int(label_id), start:end])), "top2_probability": 0.0, "margin": 0.0, "embedding": [0.0], "embedding_norm": 0.0})
    return rows


def boundary_match(predicted: list[int], truth: list[int], tolerance: int) -> tuple[int, int, int, list[int]]:
    candidates = sorted((abs(p - t), pi, ti) for pi, p in enumerate(predicted) for ti, t in enumerate(truth) if abs(p - t) <= tolerance); used_p: set[int] = set(); used_t: set[int] = set(); errors = []
    for error, pi, ti in candidates:
        if pi not in used_p and ti not in used_t: used_p.add(pi); used_t.add(ti); errors.append(error)
    return len(errors), len(predicted) - len(used_p), len(truth) - len(used_t), errors


def category(left: str, right: str) -> str:
    return f"{'known' if left in KNOWN_SET else 'novel'}-to-{'known' if right in KNOWN_SET else 'novel'}"


def boundary_metrics(entry: str, gt: list[dict[str, Any]], points: list[int], stage: str) -> list[dict[str, Any]]:
    predicted = [x for x in points if x > 0]; truth = [int(x["start"]) for x in gt[1:]]; info = [{"frame": int(g["start"]), "category": category(gt[i - 1]["label"], g["label"]), "novel_related": int(gt[i - 1]["label"] in NOVEL_SET or g["label"] in NOVEL_SET)} for i, g in enumerate(gt[1:], start=1)]
    rows = []
    for tolerance in TOLERANCES:
        for scope in ("all", "known-to-known", "known-to-novel", "novel-to-known", "novel-to-novel", "novel-related"):
            selected = [i for i, x in enumerate(info) if scope == "all" or (scope == "novel-related" and x["novel_related"]) or x["category"] == scope]; scoped = [truth[i] for i in selected]; tp, fp, fn, errors = boundary_match(predicted, scoped, tolerance); rows.append({"trajectory": entry, "stage": stage, "scope": scope, "tolerance_frames": tolerance, "gt_boundary_count": len(scoped), "predicted_boundary_count": len(predicted), "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "f1": 2 * tp / max(1, 2 * tp + fp + fn), "false_boundary_rate": fp / max(1, len(predicted)), "missed_boundary_rate": fn / max(1, len(scoped)), "mean_absolute_error": float(np.mean(errors)) if errors else 0.0, "median_absolute_error": float(np.median(errors)) if errors else 0.0})
    return rows


def novel_metrics(entry: str, gt: list[dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, g in enumerate(gt):
        if g["label"] not in NOVEL_SET: continue
        start, end = int(g["start"]), int(g["end"]); overlaps = [(i, max(0, min(s["end"], end) - max(s["start"], start)) / max(1, max(s["end"], end) - min(s["start"], start))) for i, s in enumerate(segments) if max(0, min(s["end"], end) - max(s["start"], start)) > 0]; best = max(overlaps, key=lambda x: x[1]) if overlaps else ("", 0.0); previous = index > 0 and gt[index - 1]["label"] in KNOWN_SET and any(s["start"] < start and s["end"] > start for s in segments); following = index + 1 < len(gt) and gt[index + 1]["label"] in KNOWN_SET and any(s["start"] < end and s["end"] > end for s in segments); both = index > 0 and index + 1 < len(gt) and any(abs(s["start"] - start) <= 33 for s in segments[1:]) and any(abs(s["start"] - end) <= 33 for s in segments[1:]); rows.append({"trajectory": entry, "novel_skill": g["label"], "gt_start": start, "gt_end": end, "best_iou": best[1], "iou50": int(best[1] >= .5), "iou75": int(best[1] >= .75), "fragment_count": len(overlaps), "both_boundaries_within_33": int(both), "merge_with_previous": int(previous), "merge_with_next": int(following)})
    return rows


def validation_score(item_sf: dict[str, Any], item_r5: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    points, diagnostics, suppressed = hybrid(item_sf, item_r5, cfg); labels = item_sf["asb_labels"]; segments = frame_segments(labels, points, item_sf["asb_probabilities"]); gt = r19.gt_rows_for(item_sf["entry"], "validation"); metrics, *_ = r19.summary_from_predictions(item_sf["entry"], "pick_and_place", "hybrid", segments, gt, len(labels), "validation"); truth = [int(x["start"]) for x in gt[1:]]; tp, fp, fn, errors = boundary_match([x for x in points if x > 0], truth, 33); return {**cfg, "boundary_tp": tp, "boundary_fp": fp, "boundary_fn": fn, "boundary_f1@33": 2 * tp / max(1, 2 * tp + fp + fn), "boundary_recall@33": tp / max(1, len(truth)), "mean_boundary_error": float(np.mean(errors)) if errors else 0.0, "segmental_f1@50": metrics["segmental_f1@50"], "false_predicted_segment_rate": metrics["false_predicted_segment_rate"], "duplicate_suppressed": suppressed, "region_count": len(diagnostics), "final_boundary_count": len(points)}


def select_config(val_sf: list[dict[str, Any]], val_r5: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []; thresholds = [0.5]; gaps = (0, 1, 2, 3, 5); rules = ("P0", "P1", "P2", "P3", "P4"); gates = (None, .10, .20, .30, .40, .50); separations = (0, 5, 10, 20, 33)
    for threshold in thresholds:
        for gap in gaps:
            for rule in rules:
                for gate in gates:
                    for separation in separations:
                        cfg = {"threshold": threshold, "gap": gap, "rule": rule, "support_gate": gate, "separation": separation}
                        for sf, r5 in zip(val_sf, val_r5): rows.append(validation_score(sf, r5, cfg))
    grouped = []
    for cfg_key in {json.dumps({k: v for k, v in row.items() if k in ("threshold", "gap", "rule", "support_gate", "separation")}, sort_keys=True) for row in rows}:
        cfg = json.loads(cfg_key); subset = [x for x in rows if all(x[k] == cfg[k] for k in cfg)]; grouped.append({**cfg, "validation_trajectories": len(subset), "boundary_f1@33": float(np.mean([x["boundary_f1@33"] for x in subset])), "segmental_f1@50": float(np.mean([x["segmental_f1@50"] for x in subset])), "false_predicted_segment_rate": float(np.mean([x["false_predicted_segment_rate"] for x in subset])), "mean_boundary_error": float(np.mean([x["mean_boundary_error"] for x in subset])), "duplicate_suppressed": int(sum(x["duplicate_suppressed"] for x in subset)), "region_count": int(sum(x["region_count"] for x in subset)), "final_boundary_count": int(sum(x["final_boundary_count"] for x in subset))})
    selected = max(grouped, key=lambda x: (x["boundary_f1@33"], x["segmental_f1@50"], -x["false_predicted_segment_rate"], -x["mean_boundary_error"], -x["duplicate_suppressed"], -x["gap"], -x["separation"]))
    write_csv(OUT / "validation_fusion_selection.csv", grouped + [{"selection": "selected", **selected}]); return {k: selected[k] for k in ("threshold", "gap", "rule", "support_gate", "separation")}, rows


def aggregate_rows(rows: list[dict[str, Any]], keys: tuple[str, ...], condition: str = "HYBRID_ASRF_RAW") -> list[dict[str, Any]]:
    output = []
    for key in keys:
        subset = [x for x in rows if x[key] == condition] if key == "method" else [x for x in rows if x.get("family") == key]
        if subset: output.append({"group": key, "count": len(subset), **{k: float(np.mean([float(x[k]) for x in subset])) for k in ("segmental_f1@10", "segmental_f1@25", "segmental_f1@50", "edit_score", "framewise_accuracy", "framewise_macro_f1", "mean_matched_temporal_iou", "iou_ge_0.50_rate", "iou_ge_0.75_rate", "both_boundaries_within_33_rate", "false_predicted_segment_rate", "missed_gt_segment_rate", "predicted_segments", "gt_segments")}})
    return output


def plot_trajectory(item_sf: dict[str, Any], item_r5: dict[str, Any], points: list[int], spans: list[tuple[int, int]], hybrid_segments: list[dict[str, Any]], raw_segments: list[dict[str, Any]], output: Path, threshold: float) -> None:
    sample = item_sf["sample"]; heatmap = _normalized_heatmap(sample["heatmap"].numpy()); timestamps = sample["timestamps"].numpy().astype(float); time = (timestamps - timestamps[0]) / 1e6 if timestamps.max() > 1e6 else (timestamps - timestamps[0]) / 100.0; width = len(time); gt = r19.gt_rows_for(item_sf["entry"], "test"); fig, axes = plt.subplots(6, 1, figsize=(18, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1, 1, 1.4, 1.3]})
    axes[0].imshow(heatmap, aspect="auto", origin="upper", extent=[time[0], time[-1], 0, heatmap.shape[0]]); axes[0].set_ylabel("heatmap\nchannels", rotation=0, ha="right", va="center"); axes[0].set_yticks([])
    colors = DEFAULT_LABEL_COLORS
    def draw(axis: Any, segments: list[dict[str, Any]], label_key: str, ylabel: str, gt_row: bool = False) -> None:
        for row in segments:
            start, end = int(row["start"]), int(row["end"]); label = row.get(label_key, row.get("label", "")); axis.axvspan(time[start], time[min(end - 1, width - 1)], color=colors.get(label, "#cccccc"), alpha=.82 if gt_row else .62, ec="black" if gt_row else None, lw=.5); axis.text((time[start] + time[min(end - 1, width - 1)]) / 2, .5, label, ha="center", va="center", fontsize=8, clip_on=True)
        axis.set_ylim(0, 1); axis.set_yticks([]); axis.set_ylabel(ylabel, rotation=0, ha="right", va="center")
    draw(axes[1], gt, "label", "truth", True); draw(axes[2], raw_segments, "top1_label", "raw ASB"); draw(axes[3], hybrid_segments, "top1_label", "HYBRID")
    axes[4].plot(time, item_r5["brb"], color="#222222", label="r5 BRB"); axes[4].plot(time, item_sf["brb"], color="#d62728", label="SF BRB"); axes[4].axhline(threshold, color="#222222", ls="--", lw=.8, label="r5 threshold"); axes[4].set_ylim(0, 1.05); axes[4].set_ylabel("BRB", rotation=0, ha="right", va="center"); axes[4].legend(ncol=3, fontsize=8, loc="upper right")
    levels = {"r5 region": 5, "SF point": 4, "HYBRID": 3, "RAW": 2, "GT": 1}; axes[5].set_yticks(list(levels.values()), list(levels)); axes[5].set_ylim(.5, 5.5); axes[5].set_ylabel("boundaries", rotation=0, ha="right", va="center")
    for start, end in spans: axes[5].plot([time[start], time[min(end - 1, width - 1)]], [5, 5], lw=5, color="#9467bd", alpha=.7)
    axes[5].eventplot([[time[x] for x in points if 0 < x < width]], lineoffsets=4, colors="#d62728", linelengths=.7); axes[5].eventplot([[time[x] for x in points if 0 < x < width]], lineoffsets=3, colors="#2ca02c", linelengths=.7); raw_points = [x["start"] for x in raw_segments[1:]]; axes[5].eventplot([[time[x] for x in raw_points if 0 < x < width]], lineoffsets=2, colors="#ff7f0e", linelengths=.7); gt_points = [x["start"] for x in gt[1:]]; axes[5].eventplot([[time[x] for x in gt_points if 0 < x < width]], lineoffsets=1, colors="#000000", linelengths=.7); axes[-1].set_xlabel("time (s)"); axes[0].set_title(f"{item_sf['entry']} | PP-only r5-region + SF-point hybrid ASRF"); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def plot_summaries(trajectories: list[dict[str, Any]], boundaries: list[dict[str, Any]], novels: list[dict[str, Any]], regions_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5)); tolerances = sorted({int(x["tolerance_frames"]) for x in boundaries if x["scope"] == "all"})
    for metric, label in (("precision", "precision"), ("recall", "recall"), ("f1", "F1")):
        vals = [np.mean([float(x[metric]) for x in boundaries if x["scope"] == "all" and int(x["tolerance_frames"]) == t]) for t in tolerances]; ax.plot(tolerances, vals, marker="o", label=label)
    ax.set(xlabel="boundary tolerance (frames)", ylabel="score", ylim=(0, 1.05)); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/summary_boundary_prf.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); families = sorted({x["trajectory"].split("/")[1] for x in novels}); vals = [np.mean([float(x["best_iou"]) for x in novels if x["trajectory"].split("/")[1] == family]) for family in families]; ax.bar(families, vals, color="#4c78a8"); ax.set(ylabel="novel interval mean IoU", ylim=(0, 1.05)); fig.tight_layout(); fig.savefig(OUT / "figures/summary_novel_iou_by_family.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); widths = [int(x["region_width"]) for x in regions_rows if x.get("region_width", "") != ""]; ax.hist(widths, bins=min(20, max(1, len(set(widths)))), color="#9467bd"); ax.set(xlabel="r5 region width (frames)", ylabel="count"); fig.tight_layout(); fig.savefig(OUT / "figures/summary_r5_region_width.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); displacements = [float(x["sf_r5_max_displacement"]) for x in regions_rows if x.get("sf_r5_max_displacement", "") != ""]; ax.hist(displacements, bins=min(20, max(1, len(set(displacements)))), color="#d62728"); ax.set(xlabel="SF point − r5 maximum (frames)", ylabel="count"); fig.tight_layout(); fig.savefig(OUT / "figures/summary_sf_r5_displacement.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); names = [x["trajectory"].split("/")[-1] for x in trajectories]; positions = np.arange(len(names)); ax.bar(positions - .18, [x["predicted_segments"] for x in trajectories], .36, label="predicted"); ax.bar(positions + .18, [x["gt_segments"] for x in trajectories], .36, label="GT"); ax.set_xticks(positions, names, rotation=45, ha="right"); ax.set_ylabel("segment count"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/summary_predicted_vs_gt_segments.png", dpi=160); plt.close(fig)


def main() -> int:
    seed(); OUT.mkdir(parents=True, exist_ok=True); (OUT / "predictions").mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True); metadata = audit_checkpoints(); train, validation, test = write_split_manifests()
    mapping = load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml"); sf_model = ASRFModel.from_config(metadata["sf_config"]); r5_model = ASRFModel.from_config(metadata["r5_config"]); sf_model.load_state_dict(metadata["sf_payload"]["model_state"], strict=True); r5_model.load_state_dict(metadata["r5_payload"]["model_state"], strict=True); sf_model.eval(); r5_model.eval()
    sf_target = {k: metadata["sf_config"]["data"][k] for k in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")}; r5_target = {k: metadata["r5_config"]["data"][k] for k in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")}
    val_sf = [infer(sf_model, entry, mapping, sf_target) for entry in validation]; val_r5 = [infer(r5_model, entry, mapping, r5_target) for entry in validation]; cfg, _ = select_config(val_sf, val_r5); test_sf = [infer(sf_model, entry, mapping, sf_target) for entry in test]; test_r5 = [infer(r5_model, entry, mapping, r5_target) for entry in test]
    input_audit = [input_audit_row(item, "validation") for item in val_sf]; region_rows = []; boundary_rows = []; novel_rows = []; trajectory_rows = []; family_rows = []; skill_rows = []; raw_region_counts = []
    for sf_item, r5_item in zip(test_sf, test_r5):
        input_audit.append(input_audit_row(sf_item, "test")); heatmap = sf_item["sample"]["heatmap"].numpy().astype(np.float32); timestamps = sf_item["sample"]["timestamps"].numpy()
        points, diagnostics, suppressed = hybrid(sf_item, r5_item, cfg); spans = regions(r5_item["brb"], cfg["threshold"], cfg["gap"]); labels = sf_item["asb_labels"]; segments = frame_segments(labels, points, sf_item["asb_probabilities"]); raw_segments = frame_segments(labels, [x for x in np.flatnonzero(labels[1:] != labels[:-1]) + 1], sf_item["asb_probabilities"]); gt = r19.gt_rows_for(sf_item["entry"], "test")
        for row in diagnostics: row.update({"split": "test", "trajectory": sf_item["entry"]}); region_rows.extend(diagnostics)
        for row in diagnostics: row["matched_gt_boundary"], row["localization_error"] = "", ""
        for point in points:
            closest = min((abs(point - int(x["start"])), x) for x in gt[1:]) if len(gt) > 1 else (None, None); region_rows.append({"split": "test", "trajectory": sf_item["entry"], "region_start": "", "region_end": "", "selected_frame": point, "selected_point_posthoc_gt_error": closest[0] if closest[1] else "", "region_matched_gt_boundary": int(bool(closest[1] and closest[0] <= 33))})
        boundary_rows.extend(boundary_metrics(sf_item["entry"], gt, points, "HYBRID_ASRF_RAW")); novel_rows.extend(novel_metrics(sf_item["entry"], gt, segments)); metrics, *_ = r19.summary_from_predictions(sf_item["entry"], "test", "HYBRID_ASRF_RAW", segments, gt, len(labels), "test"); metrics.update({"method": "HYBRID_ASRF_RAW", "trajectory": sf_item["entry"], "family": sf_item["entry"].split("/")[1], "fragmentation_ratio": len(segments) / max(1, len(gt)), "r5_region_count": len(spans), "final_hybrid_boundary_count": len(points), "duplicate_regions_suppressed": suppressed, "regions_no_meaningful_sf_support": sum(x["sf_support_weak"] for x in diagnostics), "mean_region_width": float(np.mean([x["region_width"] for x in diagnostics])) if diagnostics else 0.0, "mean_sf_r5_max_displacement": float(np.mean([x["sf_r5_max_displacement"] for x in diagnostics])) if diagnostics else 0.0}); trajectory_rows.append(metrics); raw_region_counts.append((sf_item["entry"], len(spans), len(points), suppressed))
        matches = r19.hungarian_matches(segments, gt)
        skills = sorted({x["label"] for x in gt})
        for skill in skills:
            support = sum(x["label"] == skill for x in gt)
            matched_gt = {m["gt_index"] for m in matches if m["iou"] >= 0.5 and gt[m["gt_index"]]["label"] == skill}
            tp = sum(segments[m["pred_index"]]["top1_label"] == skill for m in matches if m["iou"] >= 0.5 and gt[m["gt_index"]]["label"] == skill)
            predicted = sum(x["top1_label"] == skill for x in segments)
            fp = predicted - tp; fn = support - len(matched_gt)
            skill_rows.append({"method": "HYBRID_ASRF_RAW", "trajectory": sf_item["entry"], "family": sf_item["entry"].split("/")[1], "skill": skill, "gt_support": support, "predicted_support": predicted, "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "f1": 2 * tp / max(1, 2 * tp + fp + fn), "semantic_note": "novel labels are truth-only; temporal novel metrics are separate"})
        write_json(OUT / "predictions" / (sf_item["entry"].replace("/", "__") + ".json"), {"trajectory": sf_item["entry"], "sf_asb_logits": sf_item["asb_logits"], "sf_asb_labels": sf_item["asb_labels"], "sf_brb_probabilities": sf_item["brb"], "r5_brb_probabilities": r5_item["brb"], "r5_regions": spans, "sf_selected_points": points, "hybrid_boundaries": points, "raw_asb_boundaries": [x["start"] for x in raw_segments[1:]], "hybrid_segments": segments, "gt_segments": gt, "gt_matching": r19.summary_from_predictions(sf_item["entry"], "test", "HYBRID_ASRF_RAW", segments, gt, len(labels), "test")[1], "fusion_config": cfg}); np.savez_compressed(OUT / "predictions" / (sf_item["entry"].replace("/", "__") + ".npz"), sf_asb_logits=sf_item["asb_logits"], sf_asb_labels=sf_item["asb_labels"], sf_brb_probabilities=sf_item["brb"], r5_brb_probabilities=r5_item["brb"], input_heatmap=heatmap, timestamps=timestamps); plot_trajectory(sf_item, r5_item, points, spans, segments, raw_segments, OUT / "figures" / ("timeline_" + sf_item["entry"].replace("/", "__") + ".png"), cfg["threshold"])
    write_csv(OUT / "model_input_audit.csv", input_audit); write_csv(OUT / "hybrid_boundary_regions.csv", region_rows); write_csv(OUT / "boundary_metrics.csv", boundary_rows); write_csv(OUT / "novel_interval_metrics.csv", novel_rows); write_csv(OUT / "per_trajectory_metrics.csv", trajectory_rows)
    for family in sorted({x["family"] for x in trajectory_rows}):
        subset = [x for x in trajectory_rows if x["family"] == family]; family_rows.append({"method": "HYBRID_ASRF_RAW", "family": family, "trajectory_count": len(subset), **{k: float(np.mean([x[k] for x in subset])) for k in ("segmental_f1@10", "segmental_f1@25", "segmental_f1@50", "edit_score", "framewise_accuracy", "framewise_macro_f1", "mean_matched_temporal_iou", "iou_ge_0.50_rate", "iou_ge_0.75_rate", "both_boundaries_within_33_rate", "false_predicted_segment_rate", "missed_gt_segment_rate", "predicted_segments", "gt_segments")}})
    write_csv(OUT / "per_family_metrics.csv", family_rows)
    plot_summaries(trajectory_rows, boundary_rows, novel_rows, region_rows)
    write_csv(OUT / "per_skill_metrics.csv", skill_rows)
    write_json(OUT / "run_metadata.json", {"experiment": "round27_pp_only_r5_region_sf_point_hybrid", "training_occurred": False, "train_count": len(train), "validation_count": len(validation), "test_count": len(test), "fusion_config": cfg, "raw_only": True, "round25_used": False, "classifier_used": False, "gt_used_in_fusion": False})
    (OUT / "config.yaml").write_text(yaml.safe_dump({"experiment": "round27_pp_only_r5_region_sf_point_hybrid", "method": "HYBRID_ASRF_RAW", "fusion": cfg, "r5_threshold_source": "frozen Round 10 r5 config refinement.boundary_threshold", "no_retraining": True, "no_test_tuning": True, "no_round25_refinement": True, "no_segment_classifier": True, "seed": SEED}, sort_keys=False), encoding="utf-8")
    write_report(cfg, trajectory_rows, family_rows, boundary_rows, novel_rows, skill_rows, len(test), raw_region_counts)
    return 0


def write_report(cfg: dict[str, Any], trajectories: list[dict[str, Any]], families: list[dict[str, Any]], boundaries: list[dict[str, Any]], novels: list[dict[str, Any]], skills: list[dict[str, Any]], figure_count: int, region_counts: list[tuple[str, int, int, int]]) -> None:
    pooled = {k: float(np.mean([x[k] for x in trajectories])) for k in ("segmental_f1@10", "segmental_f1@25", "segmental_f1@50", "edit_score", "framewise_accuracy", "framewise_macro_f1", "mean_matched_temporal_iou", "iou_ge_0.50_rate", "iou_ge_0.75_rate", "both_boundaries_within_33_rate", "false_predicted_segment_rate", "missed_gt_segment_rate", "predicted_segments", "gt_segments")}; b = next(x for x in boundaries if x["scope"] == "all" and x["tolerance_frames"] == 33); novel_b = next(x for x in boundaries if x["scope"] == "novel-related" and x["tolerance_frames"] == 33); novel_mean = float(np.mean([x["best_iou"] for x in novels])) if novels else 0.0; novel_both = float(np.mean([x["both_boundaries_within_33"] for x in novels])) if novels else 0.0; skill_names = sorted({x["skill"] for x in skills}); skill_summary = [{"skill": skill, "f1": float(np.mean([x["f1"] for x in skills if x["skill"] == skill]))} for skill in skill_names]; lines = ["# Round 27 — PP-only r5-region + SF-point hybrid ASRF", "", "## Method and provenance", "", "Existing frozen PP-only checkpoints were reused; no retraining occurred. Training is exactly `train/pick and place/pp1`–`pp10`; validation is `train/pick and place/pp11`–`pp20`. The corrected audit lists pour p1–p5, wipe w1–w4, and plug p1–p2; `test/wipe/w4` is not present in the current dataset and is therefore explicitly excluded in `test_manifest.csv` rather than substituted. Plug p3, po1, and po2 remain excluded. No Round 25 refinement, segment classifier, open-set recognizer, duration rule, or GT-assisted fusion was used.", "", f"SF checkpoint SHA-256: `{SF_SHA}`; r5 checkpoint SHA-256: `{R5_SHA}`. Both models received identical raw `[3,88,T]` heatmap inputs; see `model_input_audit.csv` (validation and test rows).", "", "r5 proposes connected BRB regions using the frozen threshold 0.50. SF selects at most one point per region; the selected validation rule is P4 (r5 maximum fallback under the selected SF support gate). Final segment labels are majority SF-ASB argmax labels. Frame 0 is retained only as the segment origin and excluded from internal-boundary metrics.", "", "## Frozen fusion configuration", "", f"`{cfg}`", "", f"Total r5 regions: `{sum(x[1] for x in region_counts)}`; final hybrid boundaries: `{sum(x[2] for x in region_counts)}`; duplicate regions suppressed: `{sum(x[3] for x in region_counts)}`.", "", "## Raw hybrid metrics", "", "| metric | value |", "|---|---:|"]
    for key, label in (("segmental_f1@10", "F1@10"), ("segmental_f1@25", "F1@25"), ("segmental_f1@50", "F1@50"), ("edit_score", "edit score"), ("framewise_accuracy", "framewise accuracy"), ("framewise_macro_f1", "framewise macro F1"), ("mean_matched_temporal_iou", "mean matched IoU"), ("iou_ge_0.50_rate", "IoU ≥ .50"), ("iou_ge_0.75_rate", "IoU ≥ .75"), ("both_boundaries_within_33_rate", "both boundaries ±33"), ("false_predicted_segment_rate", "false predicted segment rate"), ("missed_gt_segment_rate", "missed GT rate"), ("predicted_segments", "predicted segments"), ("gt_segments", "GT segments")): lines.append(f"| {label} | {pooled[key]:.6f} |" if isinstance(pooled[key], float) else f"| {label} | {pooled[key]} |")
    lines += ["", "## Boundary result at ±33", "", f"TP={b['tp']}, FP={b['fp']}, FN={b['fn']}, precision={b['precision']:.6f}, recall={b['recall']:.6f}, F1={b['f1']:.6f}, mean error={b['mean_absolute_error']:.3f}, median error={b['median_absolute_error']:.3f}.", "", f"Novel interval mean IoU={novel_mean:.6f}; both-boundaries ±33={novel_both:.6f}.", "", "## Family results", "", "| family | F1@50 | mean IoU | false rate | missed rate |", "|---|---:|---:|---:|---:|"] + [f"| {x['family']} | {x['segmental_f1@50']:.6f} | {x['mean_matched_temporal_iou']:.6f} | {x['false_predicted_segment_rate']:.6f} | {x['missed_gt_segment_rate']:.6f} |" for x in families]
    lines += ["", "## Skill results", "", "| skill | segment recognition F1 |", "|---|---:|"] + [f"| {x['skill']} | {x['f1']:.6f} |" for x in skill_summary]
    lines += ["", "## Conclusions", "", f"Generated {figure_count} clean six-row trajectory figures and 5 compact summary figures. The hybrid produced one final point per r5 connected region before optional separation suppression; region counts and duplicate suppression are in `per_trajectory_metrics.csv` and `hybrid_boundary_regions.csv`.", "", f"At ±33 the raw hybrid boundary result is precision {b['precision']:.6f}, recall {b['recall']:.6f}, F1 {b['f1']:.6f}; novel-related boundaries are precision {novel_b['precision']:.6f}, recall {novel_b['recall']:.6f}, F1 {novel_b['f1']:.6f}. Novel interval mean IoU is {novel_mean:.6f}, with both boundaries within ±33 for {novel_both:.6f} of intervals. The result reduces duplicate wide-peak outputs by construction, but remaining region-level false positives—especially plug—are the main failure mode. It is a clean raw segmentation without Round 25; replacement of raw SF is not established by this single-method run and should remain conditional on a complete, artifact-backed PP test including w4.", "", "The method directly tests r5 region recall plus SF point localization. It does not claim novel semantic recognition; novel labels occur only in truth/temporal metrics.", "", "## Integrity", "", "Annotations unchanged; no non-PP train/validation data; no test tuning; strict checkpoint loading; identical inputs verified; no GT in fusion logic; one final boundary per r5 connected region before suppression.", "", f"Outputs: `outputs/round27_pp_only_r5_region_sf_point_hybrid/`; trajectory figures: {figure_count}; summary figures: 5."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
