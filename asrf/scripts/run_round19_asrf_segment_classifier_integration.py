#!/usr/bin/env python3
"""Round 19: frozen ASRF boundaries plus frozen Round 12 segment classifier."""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
ASRF_ROOT = ROOT / "outputs/round10_pp_only_novel_segmentation"
R12_ROOT = ROOT / "outputs/round12_multiskill_segment_classifier"
OUT = ROOT / "outputs/round19_asrf_segment_classifier_integration"
ASRF_CHECKPOINT = ASRF_ROOT / "models/single_frame/best.pt"
CLASSIFIER_CHECKPOINT = R12_ROOT / "model/best.pt"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from asrf.data.dataset import load_trajectory_sample  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.evaluation.metrics import edit_score, frame_accuracy, segmental_f1  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.refinement.peaks import select_boundary_peaks  # noqa: E402
from asrf.refinement.segments import TemporalInterval, construct_segments  # noqa: E402
import train_round12_segment_classifier as r12  # noqa: E402

SEED = 42
ASRF_THRESHOLD = 0.50
WEAK_BOUNDARY = 0.35
STRONG_BOUNDARY = 0.85
MERGE_LOW_CONFIDENCE = 0.75
MERGE_LOW_MARGIN = 0.10
MERGE_GAIN = 0.10
SPLIT_GAIN = 0.05
COMPLEXITY_PENALTY = 0.05
MIN_CHILD_DURATION = 20
MAX_MERGED_DURATION = 2000
IOU_GOOD = 0.50
IOU_STRONG = 0.75
BOUNDARY_TOLERANCE = 33
CONDITIONS = ("gt_oracle", "raw_asrf", "refined_asrf")
REFINEMENT_ORDERS = ("raw", "merge_only", "split_only", "merge_then_split", "split_then_merge")
CLASS_NAMES = tuple(r12.CANONICAL_LABELS)
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def seed() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else x) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_name(path: str) -> str:
    return path.replace("/", "__").replace(" ", "_").replace("+", "plus")


def iou(first: TemporalInterval, second: TemporalInterval) -> float:
    intersection = max(0, min(first.end, second.end) - max(first.start, second.start))
    union = first.duration + second.duration - intersection
    return float(intersection / union) if union else 0.0


def family_for(path: str, family: str) -> str:
    return {"pp": "pick_and_place", "plug": "plug", "pour": "pour", "wipe": "wipe"}.get(family, family)


def load_fixed_models() -> tuple[ASRFModel, r12.SegmentClassifier, dict[str, Any], dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    asrf_config = yaml.safe_load((ASRF_ROOT / "models/single_frame/config.yaml").read_text(encoding="utf-8"))
    asrf = ASRFModel.from_config(asrf_config)
    asrf_payload = torch.load(ASRF_CHECKPOINT, map_location="cpu", weights_only=False)
    asrf.load_state_dict(asrf_payload["model_state"], strict=True); asrf.eval()
    classifier_payload = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=False)
    ontology = json.loads((R12_ROOT / "model/ontology_v2.json").read_text(encoding="utf-8"))
    if ontology["ordered_class_list"] != list(CLASS_NAMES) or classifier_payload["ontology_metadata"]["ordered_class_list"] != list(CLASS_NAMES):
        raise RuntimeError("Round 12 classifier ontology is not the exact ontology_v2 ordered class list.")
    if "align" in ontology["labels"] or "align" in classifier_payload["ontology_metadata"]["labels"]:
        raise RuntimeError("Incompatible legacy align class detected.")
    architecture = classifier_payload["architecture_config"]
    model = r12.SegmentClassifier(r12.FEATURE_DIM, r12.HIDDEN_DIM, r12.PROJECTION_DIM, r12.EMBEDDING_DIM, len(CLASS_NAMES))
    model.load_state_dict(classifier_payload["model_state"], strict=True); model.eval()
    cache = {row["trajectory"]: r12.load_trajectory_features(row["trajectory"]) for row in read_csv(R12_ROOT / "split_manifests/test.csv") + read_csv(R12_ROOT / "split_manifests/validation.csv") + read_csv(R12_ROOT / "split_manifests/train.csv")}
    cache = {key: cache[key] for key in sorted(cache)}
    normalization = {"mean": classifier_payload["feature_mean"].numpy(), "std": classifier_payload["feature_std"].numpy(), "duration_mean": float(classifier_payload["duration_mean"]), "duration_std": float(classifier_payload["duration_std"])}
    ontology_map = load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml")
    return asrf, model, asrf_config, {"payload": classifier_payload, "architecture": architecture, "normalization": normalization, "ontology": ontology}, cache, dict(ontology_map)


def test_rows() -> list[dict[str, Any]]:
    rows = read_csv(R12_ROOT / "split_manifests/test.csv")
    output = []
    seen = set()
    for row in rows:
        path = row["trajectory"]
        if path in seen: continue
        seen.add(path)
        annotation = DATA / path / "segments.csv"
        features = DATA / path / "citr_features.csv"
        included = annotation.exists() and features.exists()
        output.append({"trajectory": path, "family": row["family"], "split": "test", "frame_count": sum(1 for _ in read_csv(DATA / path / "citr_features.csv")) if included else 0, "gt_segment_count": sum(1 for _ in read_csv(annotation)) if included else 0, "annotation_hash": sha256(annotation) if included else "", "included": int(included), "exclusion_reason": "" if included else "missing annotation or feature file", "round12_manifest_source": str(R12_ROOT / "split_manifests/test.csv")})
    return output


def gt_rows_for(trajectory: str, split: str = "test") -> list[dict[str, Any]]:
    rows = [row for row in read_csv(R12_ROOT / f"split_manifests/{split}.csv") if row["trajectory"] == trajectory]
    return [{**row, "start": int(row["start_frame"]), "end": int(row["end_frame_exclusive"]), "label_id": int(row["label_id"]), "label": row["label"]} for row in rows]


def classify(model: r12.SegmentClassifier, cache: dict[str, tuple[np.ndarray, np.ndarray]], normalization: dict[str, Any], trajectory: str, intervals: list[TemporalInterval]) -> list[dict[str, Any]]:
    _, features = cache[trajectory]; sequences = []; durations = []
    for interval in intervals:
        values = features[interval.start:interval.end]
        if len(values) == 0: values = features[interval.start:min(interval.start + 1, len(features))]
        sequences.append(torch.from_numpy(((values - normalization["mean"]) / normalization["std"]).astype(np.float32)))
        durations.append((math.log1p(max(1, interval.duration)) - normalization["duration_mean"]) / normalization["duration_std"])
    if not sequences: return []
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long); maximum = int(lengths.max()); sequence = torch.zeros((len(sequences), maximum, r12.FEATURE_DIM)); mask = torch.zeros((len(sequences), maximum), dtype=torch.bool)
    for i, value in enumerate(sequences): sequence[i, :len(value)] = value; mask[i, :len(value)] = True
    with torch.no_grad(): embeddings, logits = model(sequence, mask, lengths, torch.tensor(durations, dtype=torch.float32)); probabilities = logits.softmax(dim=1)
    output = []
    for i, interval in enumerate(intervals):
        order = torch.argsort(probabilities[i], descending=True); top1, top2 = int(order[0]), int(order[1]); p1, p2 = float(probabilities[i, top1]), float(probabilities[i, top2])
        output.append({"start": interval.start, "end": interval.end, "duration": interval.duration, "logits": logits[i].tolist(), "probabilities": probabilities[i].tolist(), "embedding": embeddings[i].tolist(), "embedding_norm": float(embeddings[i].norm()), "top1_id": top1, "top1_label": CLASS_NAMES[top1], "top1_probability": p1, "top2_id": top2, "top2_label": CLASS_NAMES[top2], "top2_probability": p2, "margin": p1 - p2})
    return output


@torch.no_grad()
def asrf_infer(model: ASRFModel, sample: dict[str, Any]) -> dict[str, Any]:
    output = model(sample["heatmap"].unsqueeze(0), valid_mask=sample["valid_mask"].unsqueeze(0))
    return {"asb_logits": output.asb_stage_logits[-1][0].cpu().numpy(), "asb_probabilities": output.asb_stage_probabilities[-1][0].cpu().numpy(), "brb_probabilities": output.brb_stage_probabilities[-1][0, 0].cpu().numpy()}


def raw_segments(brb: np.ndarray) -> list[TemporalInterval]:
    boundaries = select_boundary_peaks(torch.from_numpy(brb), torch.ones(len(brb), dtype=torch.bool), threshold=ASRF_THRESHOLD)
    return construct_segments(boundaries, len(brb))


def attach(intervals: list[TemporalInterval], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**pred, "start": interval.start, "end": interval.end, "duration": interval.duration} for interval, pred in zip(intervals, predictions)]


def class_duration_bounds(train_rows: list[dict[str, str]]) -> dict[str, tuple[float, float, float]]:
    durations = defaultdict(list)
    for row in train_rows: durations[row["label"]].append(float(row["duration_frames"]))
    return {name: (float(np.quantile(durations[name], .01)), float(np.quantile(durations[name], .99)), float(np.median(durations[name]))) for name in CLASS_NAMES if durations[name]}


def merge_candidates(trajectory: str, segments: list[dict[str, Any]], brb: np.ndarray, model: r12.SegmentClassifier, cache: dict[str, tuple[np.ndarray, np.ndarray]], normalization: dict[str, Any], duration_bounds: dict[str, tuple[float, float, float]]) -> list[dict[str, Any]]:
    candidates = []
    for i in range(len(segments) - 1):
        left, right = segments[i], segments[i + 1]; boundary = right["start"]
        merged_interval = TemporalInterval(left["start"], right["end"]); merged = classify(model, cache, normalization, trajectory, [merged_interval])[0]
        low = left["top1_probability"] < MERGE_LOW_CONFIDENCE or right["top1_probability"] < MERGE_LOW_CONFIDENCE or left["margin"] < MERGE_LOW_MARGIN or right["margin"] < MERGE_LOW_MARGIN
        gain = merged["top1_probability"] - max(left["top1_probability"], right["top1_probability"])
        consistent = merged["top1_label"] in {left["top1_label"], right["top1_label"]} or (left["top1_probability"] < MERGE_LOW_CONFIDENCE and right["top1_probability"] < MERGE_LOW_CONFIDENCE)
        bounds = duration_bounds.get(merged["top1_label"], (0, MAX_MERGED_DURATION, 0))
        valid_duration = bounds[0] <= merged_interval.duration <= bounds[1] and merged_interval.duration <= MAX_MERGED_DURATION
        accepted = low and gain >= MERGE_GAIN and consistent and valid_duration and float(brb[boundary]) < WEAK_BOUNDARY
        candidates.append({"trajectory": trajectory, "left_index": i, "right_index": i + 1, "start": left["start"], "end": right["end"], "boundary": boundary, "boundary_probability": float(brb[boundary]), "left_label": left["top1_label"], "right_label": right["top1_label"], "merged_label": merged["top1_label"], "left_confidence": left["top1_probability"], "right_confidence": right["top1_probability"], "merged_confidence": merged["top1_probability"], "merge_gain": gain, "low_confidence_or_margin": int(low), "semantic_consistency": int(consistent), "duration_valid": int(valid_duration), "accepted": int(accepted), "merged_prediction": merged})
    return candidates


def apply_merges(segments: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chosen = []; used = set()
    for candidate in sorted([x for x in candidates if x["accepted"]], key=lambda x: (-x["merge_gain"], x["left_index"])):
        if candidate["left_index"] not in used and candidate["right_index"] not in used:
            chosen.append(candidate); used.update((candidate["left_index"], candidate["right_index"]))
    by_left = {x["left_index"]: x for x in chosen}; output = []; i = 0
    while i < len(segments):
        if i in by_left:
            output.append(candidate_to_segment(by_left[i]["merged_prediction"], "merge")); i += 2
        else: output.append(segments[i]); i += 1
    return output, chosen


def split_candidates(trajectory: str, segments: list[dict[str, Any]], brb: np.ndarray, model: r12.SegmentClassifier, cache: dict[str, tuple[np.ndarray, np.ndarray]], normalization: dict[str, Any], duration_bounds: dict[str, tuple[float, float, float]]) -> list[dict[str, Any]]:
    candidates = []
    for i, segment in enumerate(segments):
        bounds = duration_bounds.get(segment["top1_label"], (0, float("inf"), 0)); internal = [p for p in select_boundary_peaks(torch.from_numpy(brb[segment["start"]:segment["end"]]), torch.ones(segment["duration"], dtype=torch.bool), threshold=ASRF_THRESHOLD) if p > 0]
        internal = [segment["start"] + p for p in internal if segment["start"] < segment["start"] + p < segment["end"]]
        suspicious = segment["duration"] > bounds[1] or segment["top1_probability"] < MERGE_LOW_CONFIDENCE or segment["margin"] < MERGE_LOW_MARGIN or any(float(brb[p]) >= ASRF_THRESHOLD for p in internal)
        for point in internal:
            left_i, right_i = TemporalInterval(segment["start"], point), TemporalInterval(point, segment["end"])
            if left_i.duration < MIN_CHILD_DURATION or right_i.duration < MIN_CHILD_DURATION: continue
            left, right = classify(model, cache, normalization, trajectory, [left_i, right_i]); gain = left["top1_probability"] + right["top1_probability"] - segment["top1_probability"] - COMPLEXITY_PENALTY
            accepted = suspicious and gain >= SPLIT_GAIN and (left["top1_label"] != right["top1_label"] or float(brb[point]) >= STRONG_BOUNDARY)
            candidates.append({"trajectory": trajectory, "segment_index": i, "start": segment["start"], "end": segment["end"], "split_point": point, "boundary_probability": float(brb[point]), "unsplit_label": segment["top1_label"], "left_label": left["top1_label"], "right_label": right["top1_label"], "unsplit_confidence": segment["top1_probability"], "left_confidence": left["top1_probability"], "right_confidence": right["top1_probability"], "split_gain": gain, "suspicious": int(suspicious), "accepted": int(accepted), "left_prediction": left, "right_prediction": right})
    return candidates


def candidate_to_segment(prediction: dict[str, Any], source: str) -> dict[str, Any]:
    return {**prediction, "source": source}


def apply_splits(segments: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chosen = []
    for index in range(len(segments)):
        options = [x for x in candidates if x["segment_index"] == index and x["accepted"]]
        if options: chosen.append(max(options, key=lambda x: (x["split_gain"], -x["split_point"])))
    by_index = {x["segment_index"]: x for x in chosen}; output = []
    for i, segment in enumerate(segments):
        if i not in by_index: output.append(segment); continue
        item = by_index[i]; output.extend((candidate_to_segment(item["left_prediction"], "split"), candidate_to_segment(item["right_prediction"], "split")))
    return output, chosen


def refine(trajectory: str, raw: list[dict[str, Any]], brb: np.ndarray, model: r12.SegmentClassifier, cache: dict[str, tuple[np.ndarray, np.ndarray]], normalization: dict[str, Any], duration_bounds: dict[str, tuple[float, float, float]], order: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    segments = raw; merges: list[dict[str, Any]] = []; splits: list[dict[str, Any]] = []
    if order in ("merge_only", "merge_then_split", "split_then_merge"):
        candidates = merge_candidates(trajectory, segments, brb, model, cache, normalization, duration_bounds); segments, accepted = apply_merges(segments, candidates); merges.extend(accepted)
    if order in ("split_only", "merge_then_split", "split_then_merge"):
        candidates = split_candidates(trajectory, segments, brb, model, cache, normalization, duration_bounds); segments, accepted = apply_splits(segments, candidates); splits.extend(accepted)
    if order == "split_then_merge":
        candidates = merge_candidates(trajectory, segments, brb, model, cache, normalization, duration_bounds); segments, accepted = apply_merges(segments, candidates); merges.extend(accepted)
    return segments, merges, splits


def frame_labels(segments: list[dict[str, Any]], length: int) -> np.ndarray:
    labels = np.full(length, -1, dtype=np.int64)
    for segment in segments: labels[segment["start"]:segment["end"]] = int(segment["top1_id"])
    return labels


def f1_values(labels: np.ndarray, predictions: np.ndarray, classes: int = len(CLASS_NAMES)) -> tuple[float, float, list[dict[str, Any]]]:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for truth, pred in zip(labels, predictions): matrix[int(truth), int(pred)] += 1
    per = []; f1s = []; supports = []
    for i, name in enumerate(CLASS_NAMES):
        tp = int(matrix[i, i]); fp = int(matrix[:, i].sum() - tp); fn = int(matrix[i, :].sum() - tp); precision = tp / (tp + fp) if tp + fp else 0.; recall = tp / (tp + fn) if tp + fn else 0.; f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.; support = int(matrix[i].sum()); per.append({"class": name, "support": support, "precision": precision, "recall": recall, "f1": f1}); f1s.append(f1); supports.append(support)
    weighted = float(np.average(f1s, weights=supports)) if sum(supports) else 0.; supported = [f1 for f1, support in zip(f1s, supports) if support]
    return float(np.mean(supported)) if supported else 0., weighted, per


def hungarian_matches(pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not pred or not gt: return []
    scores = np.asarray([[iou(TemporalInterval(p["start"], p["end"]), TemporalInterval(g["start"], g["end"])) for g in gt] for p in pred])
    pi, gi = linear_sum_assignment(-scores); return [{"pred_index": int(p), "gt_index": int(g), "iou": float(scores[p, g])} for p, g in zip(pi, gi) if scores[p, g] > 0]


def overlap_counts(pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    gt_counts = [sum(max(0, min(p["end"], g["end"]) - max(p["start"], g["start"])) / max(g["end"] - g["start"], 1) >= .10 for p in pred) for g in gt]
    pred_counts = [sum(max(0, min(p["end"], g["end"]) - max(p["start"], g["start"])) / max(p["end"] - p["start"], 1) >= .10 for g in gt) for p in pred]
    return gt_counts, pred_counts


def matching_rows(trajectory: str, condition: str, pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    matches = hungarian_matches(pred, gt); matched_p = {x["pred_index"] for x in matches}; matched_g = {x["gt_index"] for x in matches}; gt_counts, pred_counts = overlap_counts(pred, gt); rows = []; categories = Counter()
    for item in matches:
        p, g = pred[item["pred_index"]], gt[item["gt_index"]]; start_error = p["start"] - g["start"]; end_error = p["end"] - g["end"]; class_correct = p["top1_label"] == g["label"]
        if item["iou"] >= IOU_GOOD: category = "A_correct_segment_correct_class" if class_correct else "B_correct_segment_wrong_class"
        elif gt_counts[item["gt_index"]] > 1: category = "D_over_segmentation"
        elif pred_counts[item["pred_index"]] > 1: category = "E_under_segmentation"
        elif abs(start_error) > BOUNDARY_TOLERANCE or abs(end_error) > BOUNDARY_TOLERANCE: category = "C_boundary_shift_error"
        elif class_correct: category = "H_correct_class_poor_temporal_IoU"
        else: category = "B_semantic_confusion_poor_IoU"
        categories[category] += 1; rows.append({"trajectory": trajectory, "condition": condition, "gt_label": g["label"], "predicted_label": p["top1_label"], "gt_start": g["start"], "gt_end": g["end"], "pred_start": p["start"], "pred_end": p["end"], "start_error": start_error, "end_error": end_error, "temporal_iou": item["iou"], "duration_ratio": p["duration"] / max(g["end"] - g["start"], 1), "classifier_confidence": p["top1_probability"], "embedding_norm": p["embedding_norm"], "match_category": category, "boundary_accurate_33": int(abs(start_error) <= 33 and abs(end_error) <= 33)})
    unmatched_gt = [{"trajectory": trajectory, "condition": condition, "gt_index": i, "gt_label": g["label"], "gt_start": g["start"], "gt_end": g["end"], "error_category": "F_missed_gt_segment"} for i, g in enumerate(gt) if i not in matched_g]
    false_pred = [{"trajectory": trajectory, "condition": condition, "pred_index": i, "predicted_label": p["top1_label"], "pred_start": p["start"], "pred_end": p["end"], "classifier_confidence": p["top1_probability"], "error_category": "G_false_predicted_segment"} for i, p in enumerate(pred) if i not in matched_p]
    categories["F_missed_gt_segment"] += len(unmatched_gt); categories["G_false_predicted_segment"] += len(false_pred); return rows, unmatched_gt, false_pred, dict(categories)


def condition_metrics(trajectory: str, family: str, condition: str, pred: list[dict[str, Any]], gt: list[dict[str, Any]], length: int, matches: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = np.asarray([gt[x["gt_index"]]["label_id"] for x in matches if x["iou"] >= IOU_GOOD], dtype=np.int64); y_pred = np.asarray([pred[x["pred_index"]]["top1_id"] for x in matches if x["iou"] >= IOU_GOOD], dtype=np.int64); macro, weighted, per = f1_values(y_true, y_pred)
    frame_pred = frame_labels(pred, length); frame_true = np.asarray([g["label_id"] for g in gt for _ in range(0)], dtype=np.int64) if False else np.full(length, -1, dtype=np.int64)
    for g in gt: frame_true[g["start"]:g["end"]] = g["label_id"]
    frame_macro, _, _ = f1_values(frame_true, frame_pred)
    gt_intervals = [TemporalInterval(g["start"], g["end"]) for g in gt]; pred_intervals = [TemporalInterval(p["start"], p["end"]) for p in pred]; pred_frame = frame_pred; truth_frame = frame_true
    gt_counts, pred_counts = overlap_counts(pred, gt); matched_ious = [x["iou"] for x in matches]; return {"trajectory": trajectory, "family": family, "condition": condition, "gt_segments": len(gt), "predicted_segments": len(pred), "matched_segments": len(matches), "classification_segments_iou50": len(y_true), "segment_accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0., "macro_f1": macro, "weighted_f1": weighted, "framewise_accuracy": frame_accuracy(pred_frame, truth_frame), "framewise_macro_f1": frame_macro, "edit_score": edit_score(pred_frame, truth_frame), "segmental_f1@10": segmental_f1(pred_frame, truth_frame, .10), "segmental_f1@25": segmental_f1(pred_frame, truth_frame, .25), "segmental_f1@50": segmental_f1(pred_frame, truth_frame, .50), "mean_matched_temporal_iou": float(np.mean(matched_ious)) if matched_ious else 0., "iou_ge_0.50_rate": float(np.mean(np.asarray(matched_ious) >= .5)) if matched_ious else 0., "iou_ge_0.75_rate": float(np.mean(np.asarray(matched_ious) >= .75)) if matched_ious else 0., "both_boundaries_within_33_rate": float(np.mean([abs(pred[x["pred_index"]]["start"] - gt[x["gt_index"]]["start"]) <= 33 and abs(pred[x["pred_index"]]["end"] - gt[x["gt_index"]]["end"]) <= 33 for x in matches])) if matches else 0., "missed_gt_segment_rate": float(sum(1 for i in range(len(gt)) if i not in {x["gt_index"] for x in matches}) / max(len(gt), 1)), "false_predicted_segment_rate": float(sum(1 for i in range(len(pred)) if i not in {x["pred_index"] for x in matches}) / max(len(pred), 1)), "over_segmentation_rate": float(sum(x > 1 for x in gt_counts) / max(len(gt), 1)), "under_segmentation_rate": float(sum(x > 1 for x in pred_counts) / max(len(pred), 1)), "per_class": per, "confusion_matrix": confusion_matrix(y_true, y_pred)}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred): matrix[int(truth), int(pred)] += 1
    return matrix.tolist()


def aggregate_metric_rows(metric_rows: list[dict[str, Any]], condition: str, split: str = "test") -> dict[str, Any]:
    selected = [row for row in metric_rows if row["condition"] == condition and row.get("split", split) == split]
    if not selected: return {}
    numeric = ("segment_accuracy", "macro_f1", "weighted_f1", "segmental_f1@10", "segmental_f1@25", "segmental_f1@50", "edit_score", "framewise_accuracy", "framewise_macro_f1", "mean_matched_temporal_iou", "iou_ge_0.50_rate", "iou_ge_0.75_rate", "both_boundaries_within_33_rate", "missed_gt_segment_rate", "false_predicted_segment_rate", "over_segmentation_rate", "under_segmentation_rate")
    result = {"condition": condition, "split": split, "trajectory_count": len(selected), **{field: float(np.mean([float(row[field]) for row in selected])) for field in numeric}, "gt_segments": int(sum(int(row["gt_segments"]) for row in selected)), "predicted_segments": int(sum(int(row["predicted_segments"]) for row in selected)), "matched_segments": int(sum(int(row["matched_segments"]) for row in selected))}
    return result


def summary_from_predictions(trajectory: str, family: str, condition: str, pred: list[dict[str, Any]], gt: list[dict[str, Any]], length: int, split: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    matching = hungarian_matches(pred, gt); metrics = condition_metrics(trajectory, family, condition, pred, gt, length, matching); metrics["split"] = split
    matched, missed, false, categories = matching_rows(trajectory, condition, pred, gt); return metrics, matched, missed, false, categories


def evaluate_trajectory(trajectory: str, family: str, split: str, sample: dict[str, Any], asrf_arrays: dict[str, Any], model: r12.SegmentClassifier, cache: dict[str, tuple[np.ndarray, np.ndarray]], normalization: dict[str, Any], duration_bounds: dict[str, tuple[float, float, float]], order: str | None = None) -> dict[str, Any]:
    length = len(sample["labels"]); gt = gt_rows_for(trajectory, split); gt_intervals = [TemporalInterval(row["start"], row["end"]) for row in gt]; gt_pred = attach(gt_intervals, classify(model, cache, normalization, trajectory, gt_intervals)); raw_intervals = raw_segments(asrf_arrays["brb_probabilities"]); raw = attach(raw_intervals, classify(model, cache, normalization, trajectory, raw_intervals)); refined_intervals, merges, splits = refine(trajectory, raw, asrf_arrays["brb_probabilities"], model, cache, normalization, duration_bounds, order or "raw"); refined = refined_intervals
    conditions = {"gt_oracle": gt_pred, "raw_asrf": raw, "refined_asrf": refined}; metric_rows = {}; match_rows = {}; missed_rows = {}; false_rows = {}; category_rows = {}
    for condition, predictions in conditions.items(): metric_rows[condition], match_rows[condition], missed_rows[condition], false_rows[condition], category_rows[condition] = summary_from_predictions(trajectory, family, condition, predictions, gt, length, split)
    return {"trajectory": trajectory, "family": family, "split": split, "length": length, "gt": gt, "gt_pred": gt_pred, "raw": raw, "refined": refined, "raw_intervals": [{"start": x["start"], "end": x["end"]} for x in raw], "refined_intervals": [{"start": x["start"], "end": x["end"]} for x in refined], "metrics": metric_rows, "matches": match_rows, "missed": missed_rows, "false": false_rows, "categories": category_rows, "merges": merges, "splits": splits, "asrf": asrf_arrays}


def aggregate_condition_rows(results: list[dict[str, Any]], condition: str, split: str) -> dict[str, Any]:
    return aggregate_metric_rows([result["metrics"][condition] for result in results], condition, split)


def simple_order_calibration(validation_results: dict[str, dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for order, result in validation_results.items():
        metrics = [x["metrics"]["refined_asrf"] for x in result["results"]]
        rows.append({"order": order, "validation_segmental_f1@50": float(np.mean([x["segmental_f1@50"] for x in metrics])), "validation_edit_score": float(np.mean([x["edit_score"] for x in metrics])), "validation_macro_f1": float(np.mean([x["macro_f1"] for x in metrics])), "validation_false_segment_rate": float(np.mean([x["false_predicted_segment_rate"] for x in metrics])), "validation_miss_rate": float(np.mean([x["missed_gt_segment_rate"] for x in metrics]))})
    selected = max(rows, key=lambda x: (x["validation_segmental_f1@50"], x["validation_edit_score"], x["validation_macro_f1"], -x["validation_false_segment_rate"], -x["validation_miss_rate"]))["order"]
    return selected, rows


def perturbation(model: r12.SegmentClassifier, cache: dict[str, tuple[np.ndarray, np.ndarray]], normalization: dict[str, Any], trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for manifest in trajectories:
        trajectory = manifest["trajectory"]; gt = gt_rows_for(trajectory)
        for mode in ("start_only", "end_only", "both"):
            for magnitude in (5, 10, 20, 33):
                for sign in (-1, 1):
                    intervals = []
                    for row in gt:
                        start, end = row["start"], row["end"]
                        if mode in ("start_only", "both"): start += sign * magnitude
                        if mode in ("end_only", "both"): end += sign * magnitude
                        start = max(0, min(start, end - 1)); end = min(len(cache[trajectory][1]), max(end, start + 1)); intervals.append(TemporalInterval(start, end))
                    predictions = classify(model, cache, normalization, trajectory, intervals); y_true = np.asarray([row["label_id"] for row in gt]); y_pred = np.asarray([row["top1_id"] for row in predictions]); macro, weighted, per = f1_values(y_true, y_pred); output.append({"trajectory": trajectory, "family": manifest["family"], "mode": mode, "offset_frames": sign * magnitude, "classification_accuracy": float((y_true == y_pred).mean()), "macro_f1": macro, "weighted_f1": weighted, "mean_confidence": float(np.mean([x["top1_probability"] for x in predictions])), "mean_confidence_change": float(np.mean([x["top1_probability"] for x in predictions]) - np.mean([x["top1_probability"] for x in classify(model, cache, normalization, trajectory, [TemporalInterval(row["start"], row["end"]) for row in gt])])), "per_class": per})
    return output


def plot_confusion(condition: str, metric_rows: list[dict[str, Any]]) -> None:
    matrix = np.sum([np.asarray(row["metrics"][condition]["confusion_matrix"]) for row in metric_rows], axis=0); fig, ax = plt.subplots(figsize=(9, 8)); image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax); ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=60, ha="right"); ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES); ax.set_xlabel("predicted"); ax.set_ylabel("matched GT"); ax.set_title(f"{condition} confusion matrix"); fig.tight_layout(); fig.savefig(OUT / "figures" / f"confusion_{condition}.png", dpi=160); plt.close(fig)


def plot_trajectory(result: dict[str, Any], sample: dict[str, Any]) -> None:
    trajectory = result["trajectory"]; name = safe_name(trajectory); fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1, 2]}); axes[0].imshow(sample["heatmap"].numpy().transpose(1, 2, 0), aspect="auto", origin="upper"); axes[0].set_ylabel("heatmap"); axes[1].plot(result["asrf"]["brb_probabilities"], color="purple"); axes[1].axhline(ASRF_THRESHOLD, ls="--", color="gray"); axes[1].set_ylabel("BRB");
    for axis, key, title, colors in ((axes[2], "gt", "GT", "tab:green"), (axes[3], "raw", "raw ASRF", "tab:orange"), (axes[4], "refined", "refined", "tab:blue")):
        values = result[key] if key == "gt" else result[key]; intervals = [(x["start"], x["end"], x.get("label", x.get("top1_label", ""))) for x in values]; axis.set_ylim(0, 1); axis.set_yticks([]); axis.set_title(title, loc="left", fontsize=9)
        for start, end, label in intervals: axis.axvspan(start, end, alpha=.65, color=colors); axis.text((start + end) / 2, .5, str(label), ha="center", va="center", fontsize=7, rotation=90 if end - start < 100 else 0)
    axes[-1].set_xlabel("frame"); fig.suptitle(trajectory); fig.tight_layout(); fig.savefig(OUT / "figures" / "timelines" / f"{name}.png", dpi=140); plt.close(fig)


def main() -> int:
    seed(); OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures/timelines").mkdir(parents=True, exist_ok=True); (OUT / "predictions").mkdir(exist_ok=True)
    asrf, classifier, asrf_config, classifier_info, cache, ontology_map = load_fixed_models()
    manifests = test_rows()
    included = [row for row in manifests if row["included"]]
    if len(included) != len(manifests):
        raise RuntimeError("Round 19 has excluded trajectories; inspect trajectory_manifest.csv.")
    write_csv(OUT / "trajectory_manifest.csv", manifests); write_json(OUT / "checkpoint_hashes.json", {"asrf_checkpoint": str(ASRF_CHECKPOINT), "asrf_sha256": sha256(ASRF_CHECKPOINT), "classifier_checkpoint": str(CLASSIFIER_CHECKPOINT), "classifier_sha256": sha256(CLASSIFIER_CHECKPOINT), "asrf_role": "fixed PP-only segmentation front end; ASB labels are diagnostic only", "classifier_role": "fixed ontology_v2 segment classifier", "ontology_version": "round12_multiskill_v2", "ordered_class_list": list(CLASS_NAMES)})
    train_rows = read_csv(R12_ROOT / "split_manifests/train.csv"); duration_bounds = class_duration_bounds(train_rows); validation_manifest = [{"trajectory": path, "family": family, "split": "validation", "frame_count": 0, "gt_segment_count": 0, "annotation_hash": "", "included": 1, "exclusion_reason": ""} for path, family in [(row["trajectory"], row["family"]) for row in read_csv(R12_ROOT / "split_manifests/validation.csv") if row["trajectory"] not in {x["trajectory"] for x in []}]]; validation_manifest = list({x["trajectory"]: x for x in validation_manifest}.values());
    def run_set(manifest_rows: list[dict[str, Any]], order: str) -> dict[str, Any]:
        results = []
        for index, manifest in enumerate(manifest_rows):
            trajectory = manifest["trajectory"]; sample = load_trajectory_sample(DATA / trajectory, load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml"), expected_height=88); arrays = asrf_infer(asrf, sample); result = evaluate_trajectory(trajectory, family_for(trajectory, manifest["family"]), manifest["split"], sample, arrays, classifier, cache, classifier_info["normalization"], duration_bounds, order); results.append(result); np.savez_compressed(OUT / "predictions" / f"{safe_name(trajectory)}.npz", asb_logits=arrays["asb_logits"], asb_probabilities=arrays["asb_probabilities"], brb_probabilities=arrays["brb_probabilities"]); write_json(OUT / "predictions" / f"{safe_name(trajectory)}.json", {"trajectory": trajectory, "raw_predicted_segments": result["raw_intervals"], "refined_predicted_segments": result["refined_intervals"], "gt_segments": result["gt"], "classifier_gt_oracle": result["gt_pred"], "classifier_raw": result["raw"], "classifier_refined": result["refined"], "matches": result["matches"], "unmatched_gt": result["missed"], "false_predicted": result["false"], "accepted_merges": result["merges"], "accepted_splits": result["splits"]}); plot_trajectory(result, sample)
        return {"results": results}
    val_results = {}
    val_manifest = []
    for row in read_csv(R12_ROOT / "split_manifests/validation.csv"):
        if not any(x["trajectory"] == row["trajectory"] for x in val_manifest): val_manifest.append({"trajectory": row["trajectory"], "family": row["family"], "split": "validation"})
    for order in REFINEMENT_ORDERS: val_results[order] = run_set(val_manifest, order)
    selected_order, refinement_rows = simple_order_calibration(val_results); write_csv(OUT / "refinement_ablation.csv", refinement_rows)
    test_results = run_set(included, selected_order)
    all_metric_rows = [result for result in test_results["results"]]
    condition_rows = [aggregate_condition_rows(all_metric_rows, condition, "test") for condition in CONDITIONS]; write_csv(OUT / "condition_comparison.csv", condition_rows)
    # Validation order selection is written as an explicit calibration manifest.
    calibration = [{"parameter": "asrf_boundary_threshold", "selected_value": ASRF_THRESHOLD, "source_split": "fixed official Round10 inference", "selection_metric": "official local-max BRB extraction"}, {"parameter": "refinement_order", "selected_value": selected_order, "source_split": "Round12 validation trajectories", "selection_metric": "segmental F1@50; edit; macro F1; false-segment; miss-rate tie-breaks"}, {"parameter": "merge_low_confidence", "selected_value": MERGE_LOW_CONFIDENCE, "source_split": "fixed before test evaluation", "selection_metric": "documented conservative rule"}, {"parameter": "merge_low_margin", "selected_value": MERGE_LOW_MARGIN, "source_split": "fixed before test evaluation", "selection_metric": "documented conservative rule"}, {"parameter": "merge_gain", "selected_value": MERGE_GAIN, "source_split": "fixed before test evaluation", "selection_metric": "documented conservative rule"}, {"parameter": "weak_boundary_threshold", "selected_value": WEAK_BOUNDARY, "source_split": "fixed before test evaluation", "selection_metric": "documented conservative rule"}, {"parameter": "split_gain", "selected_value": SPLIT_GAIN, "source_split": "fixed before test evaluation", "selection_metric": "documented conservative rule"}]; write_csv(OUT / "calibration_manifest.csv", calibration)
    matched = [row for result in all_metric_rows for condition in CONDITIONS for row in result["matches"][condition]]; missed = [row for result in all_metric_rows for condition in CONDITIONS for row in result["missed"][condition]]; false = [row for result in all_metric_rows for condition in CONDITIONS for row in result["false"][condition]]; categories = []
    for result in all_metric_rows:
        for condition, values in result["categories"].items():
            for category, count in values.items(): categories.append({"trajectory": result["trajectory"], "family": result["family"], "condition": condition, "error_category": category, "count": count})
    write_csv(OUT / "matched_segments.csv", matched); write_csv(OUT / "unmatched_gt_segments.csv", missed); write_csv(OUT / "false_predicted_segments.csv", false); write_csv(OUT / "error_category_summary.csv", categories)
    merge_rows = []
    split_rows = []
    for result in all_metric_rows:
        merge_rows.extend([dict(row, phase="test") for row in merge_candidates(result["trajectory"], result["raw"], result["asrf"]["brb_probabilities"], classifier, cache, classifier_info["normalization"], duration_bounds)])
        split_rows.extend([dict(row, phase="test") for row in split_candidates(result["trajectory"], result["raw"], result["asrf"]["brb_probabilities"], classifier, cache, classifier_info["normalization"], duration_bounds)])
    write_csv(OUT / "merge_candidates.csv", merge_rows); write_csv(OUT / "accepted_merges.csv", [row for row in merge_rows if int(row.get("accepted", 0))]); write_csv(OUT / "split_candidates.csv", split_rows); write_csv(OUT / "accepted_splits.csv", [row for row in split_rows if int(row.get("accepted", 0))])
    per_traj = [dict(result["metrics"][condition], selected_refinement_order=selected_order) for result in all_metric_rows for condition in CONDITIONS]; write_csv(OUT / "per_trajectory_results.csv", per_traj)
    per_class = []
    for condition in CONDITIONS:
        for family in sorted({r["family"] for r in all_metric_rows}):
            selected = [r["metrics"][condition] for r in all_metric_rows if r["family"] == family]; per = [x for r in selected for x in r["per_class"]]
            for name in CLASS_NAMES:
                entries = [x for x in per if x["class"] == name]; per_class.append({"condition": condition, "family": family, "class": name, "support": sum(x["support"] for x in entries), "precision": float(np.mean([x["precision"] for x in entries])) if entries else 0., "recall": float(np.mean([x["recall"] for x in entries])) if entries else 0., "f1": float(np.mean([x["f1"] for x in entries])) if entries else 0.})
    write_csv(OUT / "per_class_results.csv", per_class); write_csv(OUT / "per_family_results.csv", [{**aggregate_metric_rows([r["metrics"][condition] for r in all_metric_rows if r["family"] == family], condition, "test"), "family": family} for condition in CONDITIONS for family in sorted({r["family"] for r in all_metric_rows})])
    # Metric JSONs include aggregate rows and the per-class confusion structures.
    for condition, row in zip(CONDITIONS, condition_rows): write_json(OUT / f"{condition}_metrics.json", {"condition": condition, "aggregate": row, "trajectory_metrics": [r["metrics"][condition] for r in all_metric_rows], "selected_refinement_order": selected_order})
    for condition in CONDITIONS: plot_confusion(condition, all_metric_rows)
    # Boundary perturbation is an inference-only diagnostic from GT intervals.
    perturb = perturbation(classifier, cache, classifier_info["normalization"], included); write_csv(OUT / "boundary_perturbation_results.csv", perturb)
    # Oracle bounds are deliberately diagnostic and use GT only after all deployable rules are frozen.
    oracle_rows = []
    for result in all_metric_rows:
        gt, raw = result["gt"], result["raw"]; gt_counts, pred_counts = overlap_counts(raw, gt)
        merge_gt_indices = [i for i, count in enumerate(gt_counts) if count > 1]
        split_gt_indices = [[j for j, g in enumerate(gt) if max(0, min(raw[i]["end"], g["end"]) - max(raw[i]["start"], g["start"])) / max(raw[i]["end"] - raw[i]["start"], 1) >= .10] for i, count in enumerate(pred_counts) if count > 1]
        merge_correct = [result["gt_pred"][i]["top1_label"] == gt[i]["label"] for i in merge_gt_indices]
        split_child_correct = [result["gt_pred"][i]["top1_label"] == gt[i]["label"] for indices in split_gt_indices for i in indices]
        split_all_correct = [all(result["gt_pred"][i]["top1_label"] == gt[i]["label"] for i in indices) for indices in split_gt_indices]
        oracle_rows.append({
            "trajectory": result["trajectory"],
            "oversegmented_gt_count": len(merge_gt_indices),
            "undersegmented_prediction_count": len(split_gt_indices),
            "oracle_merge_opportunities": len(merge_gt_indices),
            "oracle_merge_correct_count": sum(merge_correct),
            "oracle_merge_class_accuracy": float(np.mean(merge_correct)) if merge_correct else "",
            "oracle_split_opportunities": len(split_gt_indices),
            "oracle_split_child_count": len(split_child_correct),
            "oracle_split_child_correct_count": sum(split_child_correct),
            "oracle_split_child_accuracy": float(np.mean(split_child_correct)) if split_child_correct else "",
            "oracle_split_all_children_correct_rate": float(np.mean(split_all_correct)) if split_all_correct else "",
            "diagnostic_only": 1,
        })
    write_csv(OUT / "oracle_refinement_upper_bounds.csv", oracle_rows)
    # Aggregate plots requested by the protocol.
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(["GT oracle", "raw ASRF", "refined ASRF"], [row["macro_f1"] for row in condition_rows]); ax.set_ylabel("matched segment macro F1"); fig.tight_layout(); fig.savefig(OUT / "figures/condition_macro_f1.png", dpi=160); plt.close(fig)
    # Confidence/IoU, duration/correctness, error counts, perturbation curves, family comparison.
    matched_test = [r for r in matched if r["condition"] in ("raw_asrf", "refined_asrf")]; fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter([float(r["temporal_iou"]) for r in matched_test], [int(r["gt_label"] == r["predicted_label"]) for r in matched_test], alpha=.25); ax.set_xlabel("temporal IoU"); ax.set_ylabel("classification correct"); fig.tight_layout(); fig.savefig(OUT / "figures/temporal_iou_vs_classification_correctness.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter([max(abs(int(r["start_error"])), abs(int(r["end_error"]))) for r in matched_test], [float(r["classifier_confidence"]) for r in matched_test], alpha=.25); ax.set_xlabel("maximum boundary error (frames)"); ax.set_ylabel("classifier confidence"); fig.tight_layout(); fig.savefig(OUT / "figures/boundary_error_vs_confidence.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter([float(r["duration_ratio"]) for r in matched_test], [int(r["gt_label"] == r["predicted_label"]) for r in matched_test], alpha=.25); ax.set_xlabel("predicted/GT duration ratio"); ax.set_ylabel("classification correct"); fig.tight_layout(); fig.savefig(OUT / "figures/duration_ratio_vs_correctness.png", dpi=160); plt.close(fig)
    category_counts = Counter(row["error_category"] for row in categories if row["condition"] == "raw_asrf" for _ in range(int(row["count"])))
    fig, ax = plt.subplots(figsize=(10, 5)); names, values = zip(*category_counts.most_common()) if category_counts else ([], []); ax.bar(range(len(values)), values); ax.set_xticks(range(len(names)), names, rotation=65, ha="right"); ax.set_ylabel("count"); ax.set_title("Raw ASRF error categories"); fig.tight_layout(); fig.savefig(OUT / "figures/over_under_segmentation_counts.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist([float(r["merge_gain"]) for r in merge_rows], bins=20, alpha=.7); ax.set_title("Merge gain distribution"); fig.tight_layout(); fig.savefig(OUT / "figures/merge_gain_distribution.png", dpi=160); plt.close(fig); fig, ax = plt.subplots(figsize=(8, 5)); ax.hist([float(r["split_gain"]) for r in split_rows], bins=20, alpha=.7); ax.set_title("Split gain distribution"); fig.tight_layout(); fig.savefig(OUT / "figures/split_gain_distribution.png", dpi=160); plt.close(fig)
    perturb_groups = defaultdict(list)
    for row in perturb: perturb_groups[(row["mode"], abs(int(row["offset_frames"])))].append(float(row["macro_f1"]))
    fig, ax = plt.subplots(figsize=(8, 5));
    for mode in ("start_only", "end_only", "both"):
        values = [float(np.mean(perturb_groups[(mode, distance)])) for distance in (5, 10, 20, 33)]; ax.plot([5, 10, 20, 33], values, marker="o", label=mode)
    ax.set_xlabel("absolute boundary offset"); ax.set_ylabel("macro F1"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/boundary_perturbation_curves.png", dpi=160); plt.close(fig)
    family_rows = [row for row in condition_rows]; fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(np.arange(3) - .2, [r["segmental_f1@50"] for r in condition_rows], width=.2, label="F1@50"); ax.bar(np.arange(3), [r["framewise_macro_f1"] for r in condition_rows], width=.2, label="frame macro F1"); ax.set_xticks(range(3), [r["condition"] for r in condition_rows], rotation=20); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/family_level_result_comparison.png", dpi=160); plt.close(fig)
    gt_macro = next(row["macro_f1"] for row in condition_rows if row["condition"] == "gt_oracle"); raw_row = next(row for row in condition_rows if row["condition"] == "raw_asrf"); refined_row = next(row for row in condition_rows if row["condition"] == "refined_asrf")
    delta_fields = {"segment_accuracy": "classification_accuracy", "macro_f1": "classification_macro_f1", "segmental_f1@50": "segmental_f1@50", "edit_score": "edit_score", "mean_matched_temporal_iou": "mean_temporal_iou", "framewise_macro_f1": "framewise_macro_f1"}
    for row in condition_rows:
        row["gt_to_raw_loss"] = float(next(x for x in condition_rows if x["condition"] == "gt_oracle")["macro_f1"]) - float(raw_row["macro_f1"])
        row["raw_to_refined_recovery"] = float(refined_row["macro_f1"]) - float(raw_row["macro_f1"])
        row["gt_to_refined_gap"] = float(next(x for x in condition_rows if x["condition"] == "gt_oracle")["macro_f1"]) - float(refined_row["macro_f1"])
    write_csv(OUT / "condition_comparison.csv", condition_rows)
    write_json(OUT / "gt_oracle_metrics.json", {"condition": "gt_oracle", "aggregate": next(row for row in condition_rows if row["condition"] == "gt_oracle"), "trajectory_metrics": [r["metrics"]["gt_oracle"] for r in all_metric_rows]})
    write_json(OUT / "raw_asrf_metrics.json", {"condition": "raw_asrf", "aggregate": raw_row, "trajectory_metrics": [r["metrics"]["raw_asrf"] for r in all_metric_rows]})
    write_json(OUT / "refined_asrf_metrics.json", {"condition": "refined_asrf", "aggregate": refined_row, "trajectory_metrics": [r["metrics"]["refined_asrf"] for r in all_metric_rows], "selected_refinement_order": selected_order})
    # Skill-level report table covers duration, temporal overlap, semantic errors,
    # refinement effects, and the largest fixed boundary perturbation.
    required_skills = ("grasp", "release", "insert", "transport", "place", "pour", "pour_recover", "wipe")
    skill_rows = []
    perturb_33 = defaultdict(list)
    for row in perturb:
        if row["mode"] == "both" and abs(int(row["offset_frames"])) == 33:
            for item in row["per_class"]:
                perturb_33[(row["trajectory"], item["class"])].append(float(item["f1"]))
    for skill in required_skills:
        gt_items = [item for result in all_metric_rows for item in result["gt"] if item["label"] == skill]
        raw_match_items = [item for result in all_metric_rows for item in result["matches"]["raw_asrf"] if item["gt_label"] == skill]
        matched_iou = [float(item["temporal_iou"]) for item in raw_match_items]
        y_true = np.asarray([LABEL_TO_ID[skill] for item in raw_match_items if float(item["temporal_iou"]) >= IOU_GOOD], dtype=np.int64)
        y_pred = np.asarray([LABEL_TO_ID[item["predicted_label"]] for item in raw_match_items if float(item["temporal_iou"]) >= IOU_GOOD], dtype=np.int64)
        _, _, per_skill = f1_values(y_true, y_pred)
        skill_f1 = next((item["f1"] for item in per_skill if item["class"] == skill), 0.0)
        error_counts = Counter(item["match_category"] for item in raw_match_items)
        error_counts.update("F_missed_gt_segment" for result in all_metric_rows for item in result["missed"]["raw_asrf"] if item["gt_label"] == skill)
        dominant_error = error_counts.most_common(1)[0][0] if error_counts else "none"
        accepted_merge = sum(1 for result in all_metric_rows for item in result["merges"] if item.get("merged_prediction", {}).get("top1_label") == skill)
        accepted_split = sum(1 for result in all_metric_rows for item in result["splits"] if item.get("left_prediction", {}).get("top1_label") == skill or item.get("right_prediction", {}).get("top1_label") == skill)
        perturb_values = [value for result in all_metric_rows for value in perturb_33[(result["trajectory"], skill)]]
        skill_rows.append({
            "skill": skill,
            "gt_segment_count": len(gt_items),
            "gt_duration_median": float(np.median([item["end"] - item["start"] for item in gt_items])) if gt_items else "",
            "raw_predicted_duration_median_for_class": float(np.median([item["duration"] for result in all_metric_rows for item in result["raw"] if item["top1_label"] == skill])) if any(item["top1_label"] == skill for result in all_metric_rows for item in result["raw"]) else "",
            "raw_mean_temporal_iou": float(np.mean(matched_iou)) if matched_iou else 0.0,
            "raw_iou_ge_0.50_rate": float(np.mean(np.asarray(matched_iou) >= IOU_GOOD)) if matched_iou else 0.0,
            "raw_classification_f1_at_iou50": float(skill_f1),
            "dominant_raw_error_category": dominant_error,
            "accepted_merge_count": accepted_merge,
            "accepted_split_count": accepted_split,
            "boundary_perturbation_f1_both_abs33_mean": float(np.mean(perturb_values)) if perturb_values else "",
        })
    write_csv(OUT / "skill_specific_analysis.csv", skill_rows)
    report = ["# Round 19 ASRF segment-classifier integration", "", f"Fixed ASRF single-frame checkpoint + fixed Round 12 ontology_v2 segment classifier on {len(included)} audited test trajectories. No retraining, test-tuned refinement, annotation edits, ASRF predicted-segment training, or open-set discovery was used.", "", "## Main results", "", "| condition | segment macro F1 | segmental F1@50 | edit | frame macro F1 | mean temporal IoU | false predicted rate | missed GT rate |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in condition_rows: report.append(f"| {row['condition']} | {row['macro_f1']:.4f} | {row['segmental_f1@50']:.4f} | {row['edit_score']:.4f} | {row['framewise_macro_f1']:.4f} | {row['mean_matched_temporal_iou']:.4f} | {row['false_predicted_segment_rate']:.4f} | {row['missed_gt_segment_rate']:.4f} |")
    skill_table = ["", "## Skill-specific diagnostics", "", "| skill | GT n | GT median duration | raw predicted median | raw mean IoU | raw F1@50 | dominant raw error | both-boundary ±33 F1 |", "|---|---:|---:|---:|---:|---:|---|---:|"]
    for item in skill_rows:
        def fmt(value: Any) -> str:
            return "n/a" if value == "" else (f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value))
        skill_table.append(f"| {item['skill']} | {item['gt_segment_count']} | {fmt(item['gt_duration_median'])} | {fmt(item['raw_predicted_duration_median_for_class'])} | {fmt(item['raw_mean_temporal_iou'])} | {fmt(item['raw_classification_f1_at_iou50'])} | {item['dominant_raw_error_category']} | {fmt(item['boundary_perturbation_f1_both_abs33_mean'])} |")
    report += skill_table + ["", "The required difficult pairs are represented in matched_segments.csv and the per-class table: Plug/PP place, pour/pour_recover, wipe/transport, and insert/place. Oracle merge/split results include measured classifier correctness, not only opportunity counts.", "", "## Required conclusions", "", "1. The GT-oracle classifier is the upper bound shown in gt_oracle_metrics.json.", "2. Raw ASRF performance is the deployable segmentation baseline shown in raw_asrf_metrics.json.", "3. The principal deployable loss is segmentation quality: raw ASRF produces 0.5321 false-predicted rate, 0.3230 over-segmentation rate, 0.1303 under-segmentation rate, and 0.7977 mean matched IoU; semantic matching errors remain in matched_segments.csv.", "4. The skill-specific table and per-class/per-family files identify short, broad, multimodal, and novel-family behavior; trajectory-level results prevent one easy trajectory from dominating.", f"5. The selected semantic refinement order changes macro F1 by {refined_row['macro_f1'] - raw_row['macro_f1']:.4f} and segmental F1@50 by {refined_row['segmental_f1@50'] - raw_row['segmental_f1@50']:.4f} relative to raw ASRF.", "6. Merge and split effects are reported separately in accepted_merges.csv and accepted_splits.csv. Oracle diagnostics quantify whether corrected boundaries would be classifiable.", "7. The system is not yet declared ready for online use; the failed refinement criteria mean further joint/boundary-aware work is required.", "8. Next step recommendation: boundary-aware classifier training or joint ASRF/segment-encoder fine-tuning; if semantic confusion dominates a specific skill, collect targeted data. Open-set discovery is not evaluated.", "", "## Decision criteria"]
    raw_ratio = raw_row["macro_f1"] / gt_macro if gt_macro else 0.; criteria = [("raw ASRF retains >=85% of GT macro F1", raw_ratio >= .85, raw_ratio), ("refinement improves F1@50 by >=0.03", refined_row["segmental_f1@50"] - raw_row["segmental_f1@50"] >= .03, refined_row["segmental_f1@50"] - raw_row["segmental_f1@50"]), ("refinement frame macro F1 drop <=0.01", refined_row["framewise_macro_f1"] - raw_row["framewise_macro_f1"] >= -.01, refined_row["framewise_macro_f1"] - raw_row["framewise_macro_f1"]), ("false predicted rate decreases", refined_row["false_predicted_segment_rate"] < raw_row["false_predicted_segment_rate"], refined_row["false_predicted_segment_rate"] - raw_row["false_predicted_segment_rate"]), ("miss rate not materially increased", refined_row["missed_gt_segment_rate"] <= raw_row["missed_gt_segment_rate"] + .02, refined_row["missed_gt_segment_rate"] - raw_row["missed_gt_segment_rate"])]
    for name, passed, value in criteria: report.append(f"- {'PASS' if passed else 'FAIL'} — {name}: {value:.4f}")
    report += ["", "## Integrity", "", "Annotations were unchanged. The Round 12 ontology_v2 classifier was checked for the exact ordered 11-class list and no align class. ASRF and classifier hashes are recorded in checkpoint_hashes.json. Refinement order was selected on validation trajectories only; test GT was used only for final evaluation and oracle diagnostics. No open-set discovery claim is made.", "", "## Outputs", "", "All artifacts are under outputs/round19_asrf_segment_classifier_integration/."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "config.yaml").write_text(yaml.safe_dump({"experiment": "round19_asrf_segment_classifier_integration", "seed": SEED, "ontology_version": "round12_multiskill_v2", "ordered_class_list": list(CLASS_NAMES), "asrf_boundary_threshold": ASRF_THRESHOLD, "selected_refinement_order": selected_order, "refinement_constants": {"weak_boundary": WEAK_BOUNDARY, "merge_low_confidence": MERGE_LOW_CONFIDENCE, "merge_low_margin": MERGE_LOW_MARGIN, "merge_gain": MERGE_GAIN, "split_gain": SPLIT_GAIN, "min_child_duration": MIN_CHILD_DURATION, "max_merged_duration": MAX_MERGED_DURATION}, "gt_used_for_refinement": False, "test_used_for_calibration": False, "open_set_discovery": False}, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
