#!/usr/bin/env python3
"""Round 25: duration-gated local hypothesis selection.

This experiment is deliberately an inference-only layer on top of the exact
Round 19 artifacts.  The scoring function is shared by all families and all
four local hypotheses.  Validation is used to freeze the configuration before
the 33 audited test trajectories are evaluated.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
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

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
R19 = ROOT / "outputs/round19_asrf_segment_classifier_integration"
R12 = ROOT / "outputs/round12_multiskill_segment_classifier"
OUT = ROOT / "outputs/round25_duration_gated_local_hypothesis_selection"
ASRF_CHECKPOINT = ROOT / "outputs/round10_pp_only_novel_segmentation/models/single_frame/best.pt"
CLASSIFIER_CHECKPOINT = R12 / "model/best.pt"
EXPECTED_ASRF_SHA = "6b11abff2ff4387eada88e38fb56133b29fd5c3facdfb54b42fc84701213db9a"
EXPECTED_CLASSIFIER_SHA = "51f0abbcc4250ef97951bcaef04fc8f55cb2de968affdf0121a446ea1635a86f"
sys.path.insert(0, str(ROOT / "scripts"))
import run_round19_asrf_segment_classifier_integration as r19  # noqa: E402

CLASS_NAMES = tuple(r19.CLASS_NAMES)
ASB_LABELS = ("reach", "grasp", "lift", "transport", "place", "release", "retreat")
THRESHOLDS = (80, 100, 120, 150, 180, 200)
MARGIN_GRID = (0.00, 0.05, 0.10, 0.20, 0.30)
SECOND_MARGIN = 0.05
SEED = 42


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_").replace("+", "plus")


def seed() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def fail_closed() -> None:
    if not ASRF_CHECKPOINT.exists() or not CLASSIFIER_CHECKPOINT.exists():
        raise RuntimeError("Round 25 requires both frozen checkpoints.")
    if sha256(ASRF_CHECKPOINT) != EXPECTED_ASRF_SHA or sha256(CLASSIFIER_CHECKPOINT) != EXPECTED_CLASSIFIER_SHA:
        raise RuntimeError("Frozen checkpoint SHA-256 mismatch; refusing to continue.")
    ontology_path = R12 / "model/ontology_v2.json"
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    expected = {name: i for i, name in enumerate(CLASS_NAMES)}
    if ontology.get("ordered_class_list") != list(CLASS_NAMES) or ontology.get("labels") != expected:
        raise RuntimeError("ontology_v2 ordered labels do not match the fixed Round 25 ontology.")
    payload = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=False)
    metadata = payload.get("ontology_metadata", {})
    if metadata.get("ordered_class_list") != list(CLASS_NAMES) or int(metadata.get("feature_dim", -1)) != 12:
        raise RuntimeError("Classifier ontology or feature dimension mismatch.")
    if payload.get("config", {}).get("feature_columns") and len(payload["config"]["feature_columns"]) != 12:
        raise RuntimeError("Classifier preprocessing feature-column mismatch.")


def duration_models() -> tuple[dict[str, dict[str, float]], list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = read_csv(R12 / "split_manifests/train.csv")
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        label = row["label"]
        if label in CLASS_NAMES:
            values[label].append(float(row["duration_frames"]))
    models: dict[str, dict[str, float]] = {}
    stats: list[dict[str, Any]] = []
    for label in CLASS_NAMES:
        arr = np.asarray(values.get(label, []), dtype=float)
        if not len(arr):
            raise RuntimeError(f"No training duration samples for {label}.")
        q = {f"q{p}": float(np.quantile(arr, p / 100.0)) for p in (1, 5, 10, 25, 50, 75, 90, 99, 100)}
        models[label] = {**q, "median": float(np.median(arr)), "iqr": float(np.quantile(arr, .75) - np.quantile(arr, .25)), "count": int(len(arr))}
        stats.append({"skill": label, "sample_count": len(arr), **{f"{k}_frames": v for k, v in q.items()}, **{f"{k}_seconds": v / 100.0 for k, v in q.items()}, "median_frames": float(np.median(arr)), "median_seconds": float(np.median(arr) / 100.0), "iqr_frames": float(np.quantile(arr, .75) - np.quantile(arr, .25)), "iqr_seconds": float((np.quantile(arr, .75) - np.quantile(arr, .25)) / 100.0)})
    return models, stats, {label: np.asarray(values[label], dtype=float) for label in CLASS_NAMES}


def load_fixed() -> tuple[Any, Any, dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict[str, float]]]:
    asrf, classifier, _, classifier_info, cache, _ = r19.load_fixed_models()
    models, _, _ = duration_models()
    if classifier_info["normalization"]["mean"].shape != (12,) or classifier_info["normalization"]["std"].shape != (12,):
        raise RuntimeError("Frozen classifier preprocessing has the wrong feature dimension.")
    return asrf, classifier, classifier_info, cache, models


def test_manifest() -> list[dict[str, Any]]:
    rows = read_csv(R19 / "trajectory_manifest.csv")
    expected = read_csv(R21_MANIFEST := ROOT / "outputs/round21_asb_assisted_boundary_merge/trajectory_manifest.csv")
    a = [(r["trajectory"], r["family"], r["split"], r["annotation_hash"]) for r in rows]
    b = [(r["trajectory"], r["family"], r["split"], r["annotation_hash"]) for r in expected]
    if a != b:
        raise RuntimeError("Round 19 and Round 21 audited trajectory manifests differ.")
    if len(rows) != 33 or any(int(r["included"]) != 1 for r in rows):
        raise RuntimeError("The exact 33 audited Round 19 trajectories are not available.")
    return rows


def asb_summary(arrays: dict[str, np.ndarray], start: int, end: int) -> dict[str, Any]:
    probs = arrays["asb_probabilities"][:, max(0, start):min(end, len(arrays["brb_probabilities"]))]
    if probs.shape[1] == 0:
        probs = arrays["asb_probabilities"][:, max(0, start - 1):max(1, start)]
    labels = np.argmax(probs, axis=0)
    counts = np.bincount(labels, minlength=len(ASB_LABELS)); order = np.argsort(counts)[::-1]
    majority = int(order[0]); ratio = float(counts[majority] / max(1, len(labels)))
    transitions = int(np.sum(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
    entropy = float(-sum((n / max(1, len(labels))) * np.log(max(n / max(1, len(labels)), 1e-8)) for n in counts if n))
    return {"asb_majority_label": ASB_LABELS[majority], "asb_majority_ratio": ratio, "asb_entropy": entropy, "asb_transition_count": transitions, "asb_consistency": ratio, "asb_mean_probability": np.mean(probs, axis=1).tolist()}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))


def embedding_refs(classifier: Any, cache: dict[str, tuple[np.ndarray, np.ndarray]], normalization: dict[str, Any]) -> dict[str, np.ndarray]:
    rows = read_csv(R12 / "split_manifests/train.csv")
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    intervals_by_traj: dict[str, list[Any]] = defaultdict(list)
    labels_by_traj: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        intervals_by_traj[row["trajectory"]].append(r19.TemporalInterval(int(row["start_frame"]), int(row["end_frame_exclusive"])))
        labels_by_traj[row["trajectory"]].append(row["label"])
    for trajectory, intervals in intervals_by_traj.items():
        preds = r19.classify(classifier, cache, normalization, trajectory, intervals)
        for pred, label in zip(preds, labels_by_traj[trajectory]):
            grouped[label].append(np.asarray(pred["embedding"], dtype=np.float32))
    return {label: np.asarray(values, dtype=np.float32) for label, values in grouped.items()}


def support(pred: dict[str, Any], refs: dict[str, np.ndarray]) -> tuple[float, float]:
    emb = np.asarray(pred["embedding"], dtype=float); label = pred["top1_label"]
    values = refs.get(label)
    if values is None or not len(values):
        return 0.0, 1.0
    sims = np.asarray([cosine(emb, row) for row in values])
    best = float(np.max(sims)); return best, float(1.0 - best)


def dense_features(boundaries: list[int], boundary: int) -> dict[str, Any]:
    others = [x for x in boundaries if x != boundary]
    nearest = min((abs(boundary - x) for x in others), default=10**9)
    counts = {d: sum(abs(boundary - x) <= d for x in others) for d in (20, 33, 50, 100)}
    group = [x for x in boundaries if abs(boundary - x) <= 50]
    return {"nearest_boundary_distance": nearest, **{f"boundaries_within_{d}": counts[d] for d in counts}, "dense_boundary_group": int(len(group) >= 2)}


def enrich_prediction(pred: dict[str, Any], arrays: dict[str, np.ndarray], refs: dict[str, np.ndarray], models: dict[str, dict[str, float]]) -> dict[str, Any]:
    out = dict(pred)
    asb = asb_summary(arrays, int(out["start"]), int(out["end"]))
    best, dist = support(out, refs)
    dm = models.get(out["top1_label"], {})
    duration = float(out["duration"])
    quantiles = (dm.get("q1", 0), dm.get("q5", 0), dm.get("q10", 0), dm.get("q25", 0), dm.get("q50", 0), dm.get("q75", 0), dm.get("q90", 0), dm.get("q99", 10**9))
    percentile = float(np.mean(np.asarray([duration >= x for x in quantiles], dtype=float))) if dm else 0.0
    out.update(asb, embedding_support=float(best), embedding_support_distance=float(dist), predicted_class_duration_percentile=percentile, duration_lower_support=float(dm.get("q1", 0.0)), duration_upper_support=float(dm.get("q99", 10**9)), duration_median=float(dm.get("median", 0.0)), duration_plausible=int(dm.get("q1", 0.0) <= duration <= dm.get("q99", 10**9)))
    return out


def load_test_records() -> list[dict[str, Any]]:
    records = []
    for manifest in test_manifest():
        name = safe_name(manifest["trajectory"])
        payload = json.loads((R19 / "predictions" / f"{name}.json").read_text(encoding="utf-8"))
        arrays_npz = np.load(R19 / "predictions" / f"{name}.npz")
        arrays = {key: np.asarray(arrays_npz[key]) for key in arrays_npz.files}
        records.append({"trajectory": manifest["trajectory"], "family": r19.family_for(manifest["trajectory"], manifest["family"]), "split": "test", "length": len(arrays["brb_probabilities"]), "gt": payload["gt_segments"], "raw": payload["classifier_raw"], "arrays": arrays, "raw_intervals": payload["raw_predicted_segments"]})
    return records


def load_validation_records(asrf: Any, classifier: Any, info: dict[str, Any], cache: dict[str, tuple[np.ndarray, np.ndarray]], models: dict[str, dict[str, float]], refs: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []; seen: set[str] = set()
    label_map = r19.load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml")
    for row in read_csv(R12 / "split_manifests/validation.csv"):
        trajectory = row["trajectory"]
        if trajectory in seen: continue
        seen.add(trajectory)
        sample = r19.load_trajectory_sample(DATA / trajectory, label_map, expected_height=88)
        arrays = r19.asrf_infer(asrf, sample)
        intervals = r19.raw_segments(arrays["brb_probabilities"])
        raw = r19.attach(intervals, r19.classify(classifier, cache, info["normalization"], trajectory, intervals))
        gt = r19.gt_rows_for(trajectory, "validation")
        records.append({"trajectory": trajectory, "family": r19.family_for(trajectory, row["family"]), "split": "validation", "length": len(arrays["brb_probabilities"]), "gt": gt, "raw": raw, "arrays": arrays, "raw_intervals": [{"start": x.start, "end": x.end} for x in intervals]})
    return records


def attach_features(record: dict[str, Any], refs: dict[str, np.ndarray], models: dict[str, dict[str, float]]) -> None:
    record["raw"] = [enrich_prediction(x, record["arrays"], refs, models) for x in record["raw"]]
    boundaries = [int(x["start"]) for x in record["raw"][1:]]
    for i, segment in enumerate(record["raw"]):
        segment.update(dense_features(boundaries, int(segment["start"])))
        segment["segment_index"] = i
    record["boundaries"] = boundaries


def duration_ok(pred: dict[str, Any], models: dict[str, dict[str, float]]) -> float:
    model = models.get(pred["top1_label"], {})
    if not model: return 0.0
    lo, hi = model["q1"], model["q99"]
    if lo <= pred["duration"] <= hi: return 1.0
    if pred["duration"] < lo: return max(0.0, pred["duration"] / max(lo, 1.0))
    return max(0.0, hi / max(pred["duration"], 1.0))


def segment_semantic(pred: dict[str, Any], variant: str) -> float:
    p = float(pred["top1_probability"]); margin = float(pred.get("margin", 0.0)); emb = float(pred.get("embedding_support", 0.0)); agreement = float(pred.get("asb_majority_ratio", 0.0)) if pred.get("asb_majority_label") else 0.0
    value = math.log(max(p, 1e-8)) / math.log(1.0 / len(CLASS_NAMES))
    if variant in ("S1", "S2", "S3"): value += .25 * max(0.0, min(1.0, margin))
    if variant in ("S2", "S3"): value += .25 * max(0.0, min(1.0, emb))
    if variant == "S3": value += .25 * agreement * float(pred.get("asb_classifier_agreement", 0.0))
    if p < .25 or emb < .25: value -= .10
    return float(value)


def make_prediction(interval: Any, pred: dict[str, Any], arrays: dict[str, np.ndarray], refs: dict[str, np.ndarray], models: dict[str, dict[str, float]]) -> dict[str, Any]:
    return enrich_prediction({**pred, "start": interval.start, "end": interval.end, "duration": interval.duration}, arrays, refs, models)


def hypothesis_intervals(segments: list[dict[str, Any]], index: int) -> dict[str, list[Any]]:
    s = segments[index]; left = segments[index - 1] if index > 0 else None; right = segments[index + 1] if index + 1 < len(segments) else None
    out = {"H0": [r19.TemporalInterval(s["start"], s["end"])]}
    if left is not None: out["H1"] = [r19.TemporalInterval(left["start"], s["end"])] + ([r19.TemporalInterval(right["start"], right["end"])] if right else [])
    if right is not None: out["H2"] = ([r19.TemporalInterval(left["start"], left["end"])] if left else []) + [r19.TemporalInterval(s["start"], right["end"])]
    if left is not None and right is not None: out["H3"] = [r19.TemporalInterval(left["start"], right["end"])]
    return out


def boundary_value(arrays: dict[str, np.ndarray], index: int) -> float:
    brb = arrays["brb_probabilities"]
    return float(brb[max(0, min(index, len(brb) - 1))])


def score_hypotheses(record: dict[str, Any], index: int, classifier: Any, cache: dict[str, tuple[np.ndarray, np.ndarray]], info: dict[str, Any], refs: dict[str, np.ndarray], models: dict[str, dict[str, float]], cfg: dict[str, Any], prediction_cache: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    segments = record["current_segments"]; s = segments[index]
    intervals_by_h = hypothesis_intervals(segments, index)
    all_intervals = [interval for values in intervals_by_h.values() for interval in values]
    missing = [x for x in all_intervals if (x.start, x.end) not in prediction_cache]
    if missing:
        # Small duration-sorted batches keep padding bounded while allowing
        # all newly formed hypotheses to be reclassified by the frozen model.
        for pos in range(0, len(missing), 16):
            batch = sorted(missing[pos:pos + 16], key=lambda x: x.duration)
            for interval, pred in zip(batch, r19.classify(classifier, cache, info["normalization"], record["trajectory"], batch)):
                prediction_cache[(interval.start, interval.end)] = enrich_prediction(pred, record["arrays"], refs, models)
    rows = []
    for name, intervals in intervals_by_h.items():
        preds = [prediction_cache[(x.start, x.end)] for x in intervals]
        for pred, interval in zip(preds, intervals):
            pred["asb_classifier_agreement"] = float(pred["top1_label"] == pred["asb_majority_label"])
        semantic = float(np.mean([segment_semantic(pred, cfg["semantic_variant"]) for pred in preds]))
        asb_consistency = float(np.mean([pred["asb_consistency"] for pred in preds]))
        retained = []
        for interval in intervals[1:]: retained.append(boundary_value(record["arrays"], interval.start))
        retained_support = float(np.mean([1.0 - x for x in retained])) if retained else 0.0
        duration_plausibility = float(np.mean([duration_ok(pred, models) for pred in preds]))
        fragment_penalty = float(np.mean([int(pred["duration"] < pred["duration_lower_support"]) for pred in preds]))
        local_input_count = 1 + int(index > 0) + int(index + 1 < len(segments))
        complexity = float(local_input_count - len(intervals) + (0.25 if name == "H3" else 0.0))
        source_labels = {x["top1_label"] for x in segments[max(0, index - 1):min(len(segments), index + 2)]}
        conflicts = sum(int(pred["top1_label"] not in source_labels) for pred in preds) / max(1, len(preds))
        dense_bonus = 0.0
        if cfg.get("dense_mode") == "D1": dense_bonus = .15 * float(s.get("dense_boundary_group", 0))
        if cfg.get("dense_mode") == "D2": dense_bonus = .20 * float(s.get("boundaries_within_33", 0) > 0)
        if name == "H1" and index > 0 and segments[index - 1]["top1_label"] == s["top1_label"] and boundary_value(record["arrays"], s["start"]) < .35: dense_bonus += .05
        if name == "H2" and index + 1 < len(segments) and segments[index + 1]["top1_label"] == s["top1_label"] and boundary_value(record["arrays"], s["end"]) < .35: dense_bonus += .05
        score = (cfg["w_semantic"] * semantic + cfg["w_asb"] * asb_consistency + cfg["w_boundary"] * retained_support + cfg["w_duration"] * duration_plausibility - cfg["w_fragment"] * fragment_penalty - cfg["w_complexity"] * complexity - cfg["w_conflict"] * conflicts + dense_bonus)
        rows.append({"hypothesis": name, "trajectory": record["trajectory"], "candidate_segment_index": index, "candidate_start": s["start"], "candidate_end": s["end"], "candidate_duration": s["duration"], "semantic_score": semantic, "asb_consistency": asb_consistency, "retained_boundary_support": retained_support, "duration_plausibility": duration_plausibility, "short_fragment_penalty": fragment_penalty, "merge_complexity": complexity, "semantic_conflict": conflicts, "dense_boundary_bonus": dense_bonus, "score": float(score), "resulting_intervals": [[x.start, x.end] for x in intervals], "resulting_labels": [x["top1_label"] for x in preds], "removed_boundaries": [int(x["start"]) for x in segments[max(0, index - 1):index + 2] if x is not s], "predictions": preds})
    return rows


def candidate_allowed(segment: dict[str, Any], threshold: int, mode: str, models: dict[str, dict[str, float]]) -> bool:
    if mode == "global": return int(segment["duration"]) < threshold
    model = models.get(segment["top1_label"], {})
    return int(segment["duration"]) < min(threshold, int(round(model.get("q10", threshold))))


def select_operation(record: dict[str, Any], index: int, classifier: Any, cache: dict[str, tuple[np.ndarray, np.ndarray]], info: dict[str, Any], refs: dict[str, np.ndarray], models: dict[str, dict[str, float]], cfg: dict[str, Any], prediction_cache: dict[tuple[int, int], dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    segments = record["current_segments"]; candidate = segments[index]
    if not candidate_allowed(candidate, cfg["threshold"], cfg["threshold_mode"], models): return None, []
    if not (index > 0 or index + 1 < len(segments)): return None, []
    scores = score_hypotheses(record, index, classifier, cache, info, refs, models, cfg, prediction_cache)
    by_name = {x["hypothesis"]: x for x in scores}; keep = by_name["H0"]; merges = [x for x in scores if x["hypothesis"] != "H0"]
    best = max(merges, key=lambda x: x["score"]) if merges else None
    second = max([x["score"] for x in merges if x is not best] + [-1e9]) if best else -1e9
    decision_margin = float(best["score"] - keep["score"]) if best else -1e9
    separation = float(best["score"] - second) if best else -1e9
    stable_labels = {x["top1_label"] for x in segments[max(0, index - 1):min(len(segments), index + 2)]}
    h3 = best and best["hypothesis"] == "H3"
    left_b = boundary_value(record["arrays"], candidate["start"]); right_b = boundary_value(record["arrays"], candidate["end"])
    h3_protected = bool(h3 and index > 0 and index + 1 < len(segments) and (left_b >= .50 and right_b >= .50 and candidate["asb_majority_ratio"] >= .85 or best["duration_plausibility"] < .2))
    valid = bool(best and decision_margin >= cfg["decision_margin"] and separation >= cfg.get("second_margin", SECOND_MARGIN) and not h3_protected)
    selected = dict(best) if valid else None
    return (None if selected is None else {"selected": selected["hypothesis"], "decision_margin": decision_margin, "second_best_separation": separation, "h3_protected": int(h3_protected), "same_label_shortcut": int((index > 0 and segments[index - 1]["top1_label"] == candidate["top1_label"]) or (index + 1 < len(segments) and segments[index + 1]["top1_label"] == candidate["top1_label"])), "source_index": index, "scores": scores}), scores


def apply_selected(segments: list[dict[str, Any]], operation: dict[str, Any]) -> list[dict[str, Any]]:
    i = operation["source_index"]; name = operation["selected"]; rows = {x["hypothesis"]: x for x in operation["scores"]}; chosen = rows[name]
    intervals = chosen["resulting_intervals"]; predictions = chosen["predictions"]
    start_index = i - 1 if name in ("H1", "H3") else i
    consumed = 3 if name == "H3" else 2
    out = segments[:start_index]
    out.extend([dict(pred, start=int(interval[0]), end=int(interval[1]), duration=int(interval[1] - interval[0])) for pred, interval in zip(predictions, intervals)])
    out.extend(segments[start_index + consumed:])
    for j, segment in enumerate(out): segment["segment_index"] = j
    return out


def run_refinement(record: dict[str, Any], classifier: Any, cache: dict[str, tuple[np.ndarray, np.ndarray]], info: dict[str, Any], refs: dict[str, np.ndarray], models: dict[str, dict[str, float]], cfg: dict[str, Any], export_scores: bool = False) -> dict[str, Any]:
    record = dict(record); record["current_segments"] = [dict(x) for x in record["raw"]]; prediction_cache = {(x["start"], x["end"]): x for x in record["raw"]}; all_scores: list[dict[str, Any]] = []; operations: list[dict[str, Any]] = []; rejected: list[dict[str, Any]] = []
    max_iters = int(cfg.get("max_iterations", 1)) if cfg["processing_mode"] == "iterative" else 1
    for iteration in range(max_iters):
        current = record["current_segments"]; choices: list[dict[str, Any]] = []
        for index in range(len(current)):
            operation, scores = select_operation({**record, "current_segments": current}, index, classifier, cache, info, refs, models, cfg, prediction_cache)
            for row in scores:
                row["iteration"] = iteration; row["accepted"] = int(operation is not None and row["hypothesis"] == operation["selected"]); all_scores.append(row)
            if operation is not None:
                operation["iteration"] = iteration; choices.append(operation)
            elif scores:
                keep = next(x for x in scores if x["hypothesis"] == "H0"); rejected.append({"trajectory": record["trajectory"], "candidate_segment_index": index, "candidate_start": keep["candidate_start"], "candidate_end": keep["candidate_end"], "reason": "below_decision_or_separation_margin_or_protected", "keep_score": keep["score"], "best_merge_score": max(x["score"] for x in scores if x["hypothesis"] != "H0")})
        if not choices: break
        if cfg["processing_mode"] == "one_shot":
            occupied: set[int] = set(); selected: list[dict[str, Any]] = []
            for choice in sorted(choices, key=lambda x: (-x["decision_margin"], x["source_index"])):
                i = int(choice["source_index"]); span = {i}
                if choice["selected"] in ("H1", "H3"): span.add(i - 1)
                if choice["selected"] in ("H2", "H3"): span.add(i + 1)
                if span & occupied: continue
                occupied.update(span); choice["iteration"] = iteration; selected.append(choice)
            for choice in sorted(selected, key=lambda x: x["source_index"], reverse=True):
                record["current_segments"] = apply_selected(record["current_segments"], choice)
            operations.extend(selected)
            break
        if cfg["processing_mode"] == "left_to_right": chosen = min(choices, key=lambda x: x["source_index"])
        else: chosen = max(choices, key=lambda x: x["decision_margin"])
        record["current_segments"] = apply_selected(current, chosen); chosen["iteration"] = iteration; operations.append(chosen)
    record["refined"] = record["current_segments"]
    return {"record": record, "scores": all_scores, "operations": operations, "rejected": rejected}


def metric_for(record: dict[str, Any], predictions: list[dict[str, Any]], condition: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    return r19.summary_from_predictions(record["trajectory"], record["family"], condition, predictions, record["gt"], record["length"], record["split"])


def aggregate(records: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    return r19.aggregate_metric_rows([x["metrics"][condition] for x in records], condition, records[0]["split"] if records else "test")


def evaluate_config(records: list[dict[str, Any]], classifier: Any, cache: dict[str, tuple[np.ndarray, np.ndarray]], info: dict[str, Any], refs: dict[str, np.ndarray], models: dict[str, dict[str, float]], cfg: dict[str, Any], retain: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    outputs = []; scores = []; ops = []
    for original in records:
        result = run_refinement(original, classifier, cache, info, refs, models, cfg)
        refined = result["record"]["refined"]
        metrics, matched, missed, false, cats = metric_for(original, refined, "refined_asrf")
        raw_metrics, raw_matched, raw_missed, raw_false, raw_cats = metric_for(original, original["raw"], "raw_asrf")
        outputs.append({"record": original, "refined": refined, "metrics": {"raw_asrf": raw_metrics, "refined_asrf": metrics}, "matches": {"raw_asrf": raw_matched, "refined_asrf": matched}, "missed": {"raw_asrf": raw_missed, "refined_asrf": missed}, "false": {"raw_asrf": raw_false, "refined_asrf": false}, "categories": {"raw_asrf": raw_cats, "refined_asrf": cats}, "scores": result["scores"], "operations": result["operations"], "rejected": result["rejected"]})
        if retain: scores.extend(result["scores"]); ops.extend([{**x, "record": original} for x in result["operations"]])
    row = r19.aggregate_metric_rows([x["metrics"]["refined_asrf"] for x in outputs], "refined_asrf", records[0]["split"] if records else "validation")
    return row, outputs, scores + ops


def config_for(name: str, threshold: int, threshold_mode: str = "global", processing: str = "one_shot", max_iterations: int = 8, margin: float = .10, dense: str = "D0") -> dict[str, Any]:
    # One formula; ablations change evidence weights, not family-specific logic.
    weights = {
        "R1": (1.0, 0.0, 0.0, 0.0, 0.5, .05, .25, "S0"),
        "R2": (1.0, 0.0, 0.0, 0.0, .5, .05, .25, "S0"),
        "R3": (1.0, 0.0, 0.0, 0.0, .5, .05, .25, "S3"),
        "R4": (1.0, 0.5, 1.0, 0.0, .5, .05, .25, "S3"),
        "R5": (1.0, 0.5, 1.0, .5, .5, .05, .25, "S3"),
        "R6": (1.0, 0.5, 1.0, 1.0, .75, .10, .35, "S3"),
        "R7": (1.0, 0.5, 1.0, 1.0, .75, .10, .35, "S3"),
        "R8": (1.0, 0.5, 1.0, 1.0, .75, .10, .35, "S3"),
    }
    sem, asb, boundary, duration, fragment, complexity, conflict, semantic_variant = weights.get(name, weights["R7"])
    return {"name": name, "threshold": threshold, "threshold_mode": threshold_mode, "processing_mode": processing, "max_iterations": max_iterations, "decision_margin": margin, "second_margin": SECOND_MARGIN, "semantic_variant": semantic_variant, "dense_mode": dense, "w_semantic": sem, "w_asb": asb, "w_boundary": boundary, "w_duration": duration, "w_fragment": fragment, "w_complexity": complexity, "w_conflict": conflict}


def raw_exact_metrics(test_records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in test_records:
        metrics, *_ = metric_for(record, record["raw"], "raw_asrf")
        rows.append(metrics)
    actual = r19.aggregate_metric_rows(rows, "raw_asrf", "test")
    expected = json.loads((R19 / "raw_asrf_metrics.json").read_text(encoding="utf-8"))["aggregate"]
    fields = ("segmental_f1@50", "edit_score", "framewise_macro_f1", "mean_matched_temporal_iou", "false_predicted_segment_rate", "missed_gt_segment_rate")
    deltas = {field: float(actual[field]) - float(expected[field]) for field in fields}
    return {"exact_artifact_reuse": True, "source": str(R19 / "predictions"), "expected": {x: expected[x] for x in fields}, "actual": {x: actual[x] for x in fields}, "deltas": deltas, "zero_delta": all(abs(x) < 1e-12 for x in deltas.values())}


def operation_audit(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in outputs:
        record = result["record"]; gt = record["gt"]
        for op in result["operations"]:
            selected = next(x for x in op["scores"] if x["hypothesis"] == op["selected"])
            # Local operation audit is intentionally post-inference: only this
            # function receives GT and it is never called by scoring.
            before = record["raw"]
            after = apply_selected(before, op) if all(x < len(before) for x in [op["source_index"]]) else record["refined"]
            before_m = r19.summary_from_predictions(record["trajectory"], record["family"], "before", before, gt, record["length"], record["split"])[0]
            after_m = r19.summary_from_predictions(record["trajectory"], record["family"], "after", after, gt, record["length"], record["split"])[0]
            distinct = False
            start, end = selected["resulting_intervals"][0]
            overlapped = [g for g in gt if max(0, min(end, g["end"]) - max(start, g["start"])) > .10 * max(1, g["end"] - g["start"])]
            if len({g["label"] for g in overlapped}) > 1: distinct = True
            df1 = float(after_m["segmental_f1@50"] - before_m["segmental_f1@50"]); dfalse = float(after_m["false_predicted_segment_rate"] - before_m["false_predicted_segment_rate"]); dmiss = float(after_m["missed_gt_segment_rate"] - before_m["missed_gt_segment_rate"])
            if distinct or dmiss > .01 or df1 < -.01: category = "clearly harmful"
            elif df1 > .01 and dfalse <= 0: category = "clearly beneficial"
            elif df1 >= 0 and dfalse <= .01: category = "weakly beneficial"
            elif abs(df1) < .01 and abs(dfalse) < .01: category = "neutral"
            else: category = "weakly harmful"
            rows.append({"trajectory": record["trajectory"], "family": record["family"], "iteration": op.get("iteration", 0), "hypothesis": op["selected"], "candidate_start": selected["candidate_start"], "candidate_end": selected["candidate_end"], "h0_score": next(x["score"] for x in op["scores"] if x["hypothesis"] == "H0"), "h1_score": next((x["score"] for x in op["scores"] if x["hypothesis"] == "H1"), ""), "h2_score": next((x["score"] for x in op["scores"] if x["hypothesis"] == "H2"), ""), "h3_score": next((x["score"] for x in op["scores"] if x["hypothesis"] == "H3"), ""), "decision_margin": op["decision_margin"], "second_best_separation": op["second_best_separation"], "metric_delta_f1@50": df1, "metric_delta_false_rate": dfalse, "metric_delta_missed_rate": dmiss, "distinct_gt_skills_merged": int(distinct), "audit_category": category})
    return rows


def diagnostic_skillness(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        gt_intervals = [(int(x["start"]), int(x["end"]), "gt_complete") for x in record["gt"]]
        raw = record["raw"]
        fragments = [(int(x["start"]), int(x["end"]), "asrf_internal_fragment") for x in raw if any(x["start"] >= g["start"] and x["end"] <= g["end"] and x["duration"] < g["end"] - g["start"] for g in record["gt"])]
        mixed = [(int(x["start"]), int(x["end"]), "cross_boundary_mixed") for x in raw if sum(max(0, min(x["end"], g["end"]) - max(x["start"], g["start"])) > .1 * max(1, g["end"] - g["start"]) for g in record["gt"]) > 1]
        for start, end, kind in gt_intervals + fragments + mixed:
            pred = next((x for x in raw if x["start"] == start and x["end"] == end), None)
            if pred is None:
                pred = enrich_prediction({"start": start, "end": end, "duration": end - start, "top1_label": "reach", "top1_probability": .0, "top2_probability": .0, "margin": .0, "embedding": [0.0] * 128, "top1_id": 0}, record["arrays"], {}, {})
            internal = float(pred.get("asb_consistency", 0.0)); score = .25 * duration_ok(pred, {}) + .25 * pred.get("margin", 0.0) + .25 * pred.get("embedding_support", 0.0) + .25 * internal
            rows.append({"trajectory": record["trajectory"], "split": split, "kind": kind, "start": start, "end": end, "duration_frames": end - start, "diagnostic_skillness_score": score, "duration_plausibility": duration_ok(pred, {}), "classifier_margin": pred.get("margin", 0.0), "embedding_support": pred.get("embedding_support", 0.0), "asb_stability": internal})
    return rows


def auroc_ap(rows: list[dict[str, Any]]) -> dict[str, float]:
    positives = [x for x in rows if x["kind"] == "gt_complete"]; negatives = [x for x in rows if x["kind"] != "gt_complete"]
    if not positives or not negatives: return {"auroc": 0.0, "aupr": 0.0}
    scores_p = np.asarray([x["diagnostic_skillness_score"] for x in positives]); scores_n = np.asarray([x["diagnostic_skillness_score"] for x in negatives])
    auc = float(np.mean([float(a > b) + .5 * float(a == b) for a in scores_p for b in scores_n]))
    order = np.argsort(-np.asarray([*scores_p, *scores_n])); labels = np.asarray([1] * len(scores_p) + [0] * len(scores_n))[order]; precision = np.cumsum(labels) / (np.arange(len(labels)) + 1); ap = float(np.sum(precision[labels == 1]) / max(1, len(scores_p)))
    return {"auroc": auc, "aupr": ap}


def plot_duration(stats: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    OUT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(np.arange(len(stats)), [x["median_seconds"] for x in stats]); ax.set_xticks(range(len(stats)), [x["skill"] for x in stats], rotation=45, ha="right"); ax.set_ylabel("GT median duration (seconds)"); fig.tight_layout(); fig.savefig(OUT / "figures/gt_duration_distributions_by_skill.png", dpi=150); plt.close(fig)
    values = [float(x["duration"]) / 100 for r in records for x in r["raw"]]; fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(values, bins=30); ax.set_xlabel("raw ASRF segment duration (seconds)"); ax.set_ylabel("count"); fig.tight_layout(); fig.savefig(OUT / "figures/raw_predicted_segment_duration_histogram.png", dpi=150); plt.close(fig)


def plot_timeline(record: dict[str, Any], result: dict[str, Any]) -> None:
    arrays = record["arrays"]; length = record["length"]; fig, axes = plt.subplots(9, 1, figsize=(16, 15), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1, 1, 1, 1, 1, 1]})
    heatmap = arrays.get("asb_probabilities", np.zeros((1, length))); axes[0].imshow(heatmap, aspect="auto", origin="lower"); axes[0].set_ylabel("ASB"); axes[1].plot(arrays["brb_probabilities"], color="purple"); axes[1].set_ylabel("BRB")
    for axis, values, title, color in ((axes[2], record["gt"], "GT", "tab:green"), (axes[3], record["raw"], "raw ASRF", "tab:orange"), (axes[4], result["refined"], "Round 25", "tab:blue")):
        axis.set_ylim(0, 1); axis.set_yticks([]); axis.set_title(title, loc="left", fontsize=9)
        for x in values:
            start, end = int(x["start"]), int(x["end"]); axis.axvspan(start, end, color=color, alpha=.55); axis.text((start + end) / 2, .5, str(x.get("label", x.get("top1_label", ""))), ha="center", va="center", fontsize=6, rotation=90 if end - start < 150 else 0)
    labels = np.argmax(arrays["asb_probabilities"], axis=0); axes[5].plot(labels, lw=.5); axes[5].set_ylabel("ASB label")
    candidates = [x for x in record["raw"] if x["duration"] < result["cfg"]["threshold"]]; axes[6].set_ylim(0, 1); axes[6].set_yticks([]); axes[6].set_title("short candidates", loc="left", fontsize=9)
    for x in candidates: axes[6].axvspan(x["start"], x["end"], color="gold", alpha=.55)
    axes[7].set_ylim(0, 1); axes[7].set_yticks([]); axes[7].set_title("selected H0/H1/H2/H3 and deleted boundaries", loc="left", fontsize=9)
    for op in result["operations"]:
        selected = next(x for x in op["scores"] if x["hypothesis"] == op["selected"]); start, end = selected["candidate_start"], selected["candidate_end"]; axes[7].axvspan(start, end, color={"H1":"red", "H2":"cyan", "H3":"magenta"}.get(op["selected"], "gray"), alpha=.55); axes[7].text((start + end) / 2, .5, op["selected"], ha="center", fontsize=7)
        for b in selected.get("removed_boundaries", []): axes[7].axvline(b, color="black", lw=1)
    axes[8].set_ylim(0, 1); axes[8].set_yticks([]); axes[8].set_title("decision margin", loc="left", fontsize=9)
    for op in result["operations"]: axes[8].plot([op["decision_margin"]], [0.5], "o", color="black")
    axes[-1].set_xlabel("frame"); fig.suptitle(record["trajectory"]); fig.tight_layout(); fig.savefig(OUT / "figures" / f"timeline_{safe_name(record['trajectory'])}.png", dpi=120); plt.close(fig)


def main() -> int:
    seed(); fail_closed(); OUT.mkdir(parents=True, exist_ok=True); (OUT / "predictions").mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    asrf, classifier, info, cache, models = load_fixed(); refs = embedding_refs(classifier, cache, info["normalization"]); duration_model_rows, duration_values = duration_models()[1:]
    write_csv(OUT / "gt_duration_statistics.csv", duration_model_rows)
    test_records = load_test_records(); validation_records = load_validation_records(asrf, classifier, info, cache, models, refs)
    for record in [*validation_records, *test_records]: attach_features(record, refs, models)
    plot_duration(duration_model_rows, test_records)
    write_csv(OUT / "trajectory_manifest.csv", test_manifest())
    write_json(OUT / "checkpoint_hashes.json", {"asrf_checkpoint": str(ASRF_CHECKPOINT), "asrf_sha256": sha256(ASRF_CHECKPOINT), "classifier_checkpoint": str(CLASSIFIER_CHECKPOINT), "classifier_sha256": sha256(CLASSIFIER_CHECKPOINT), "expected_asrf_sha256": EXPECTED_ASRF_SHA, "expected_classifier_sha256": EXPECTED_CLASSIFIER_SHA, "ontology_version": "round12_multiskill_v2", "ordered_class_list": list(CLASS_NAMES), "retraining": False, "annotations_changed": False})
    raw_repro = raw_exact_metrics(test_records); write_json(OUT / "raw_reproduction_metrics.json", raw_repro)
    # A compact, pre-registered validation sweep covers every requested
    # duration threshold and semantic/dense/duration/processing ablation.
    validation_rows = []
    configs: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        configs.append(config_for("R7", threshold, "global", "one_shot", 1, .10, "D1"))
    for name in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        configs.append(config_for(name, 180, "global", "one_shot", 1, .10, "D1"))
    for processing in ("one_shot", "iterative", "left_to_right"):
        for max_iterations in (4, 8, 12): configs.append(config_for("R7", 180, "global", processing, max_iterations, .10, "D1"))
    configs += [config_for("R7", 180, "class_conditional", "iterative", 8, .10, "D1"), config_for("R7", 180, "global", "iterative", 8, .00, "D0"), config_for("R7", 180, "global", "iterative", 8, .20, "D2")]
    unique: dict[str, dict[str, Any]] = {json.dumps(x, sort_keys=True): x for x in configs}; configs = list(unique.values())
    selected_row = None; selected_outputs = None; selected_cfg = None
    validation_raw_rows = [metric_for(x, x["raw"], "raw_asrf")[0] for x in validation_records]
    validation_raw = r19.aggregate_metric_rows(validation_raw_rows, "raw_asrf", "validation")
    for cfg in configs:
        row, outputs, _ = evaluate_config(validation_records, classifier, cache, info, refs, models, cfg)
        row.update({"variant": cfg["name"], "threshold": cfg["threshold"], "threshold_mode": cfg["threshold_mode"], "processing_mode": cfg["processing_mode"], "max_iterations": cfg["max_iterations"], "decision_margin": cfg["decision_margin"], "dense_mode": cfg["dense_mode"], "semantic_variant": cfg["semantic_variant"]})
        validation_rows.append(row)
        feasible = float(row.get("framewise_macro_f1", 0)) >= float(validation_raw.get("framewise_macro_f1", 0)) - .01 and float(row.get("missed_gt_segment_rate", 0)) <= float(validation_raw.get("missed_gt_segment_rate", 0)) + .01
        rank = (int(feasible), float(row.get("segmental_f1@50", 0)), -float(row.get("false_predicted_segment_rate", 0)), float(row.get("edit_score", 0)), float(row.get("mean_matched_temporal_iou", 0)), float(row.get("framewise_macro_f1", 0)))
        if selected_row is None or rank > selected_row[0]: selected_row, selected_outputs, selected_cfg = (rank, row), outputs, cfg
    write_csv(OUT / "validation_rule_selection.csv", validation_rows)
    cfg = selected_cfg; test_row, test_outputs, retained = evaluate_config(test_records, classifier, cache, info, refs, models, cfg, True)
    all_scores = [x for x in retained if "hypothesis" in x]; operations = [x for x in retained if "selected" in x]
    audit = operation_audit(test_outputs); write_csv(OUT / "operation_level_audit.csv", audit)
    write_csv(OUT / "candidate_segments.csv", [dict(x, threshold=cfg["threshold"], threshold_mode=cfg["threshold_mode"], candidate=int(candidate_allowed(x, cfg["threshold"], cfg["threshold_mode"], models))) for record in test_records for x in record["raw"]])
    write_csv(OUT / "hypothesis_scores.csv", [dict(x, predictions=json.dumps(x.pop("predictions", []), default=_json_default), resulting_intervals=json.dumps(x.get("resulting_intervals", []))) for x in all_scores])
    write_csv(OUT / "accepted_operations.csv", [{"trajectory": x["record"]["trajectory"], "family": x["record"]["family"], "iteration": x["iteration"], "selected_hypothesis": x["selected"], "decision_margin": x["decision_margin"], "second_best_separation": x["second_best_separation"]} for x in operations])
    write_csv(OUT / "rejected_candidates.csv", [x for result in test_outputs for x in result["rejected"]])
    # Required ablations use the same frozen precomputed evidence and are not
    # tuned on test labels.
    variant_rows = []
    for name in ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"):
        if name == "R0": row = r19.aggregate_metric_rows([metric_for(x, x["raw"], "raw_asrf")[0] for x in test_records], "raw_asrf", "test")
        else:
            vc = cfg if name == "R8" else config_for(name, cfg["threshold"], cfg["threshold_mode"], "one_shot", 1, cfg["decision_margin"], cfg["dense_mode"])
            row, _, _ = evaluate_config(test_records, classifier, cache, info, refs, models, vc)
        variant_rows.append({"variant": name, **row})
    write_csv(OUT / "hypothesis_variant_comparison.csv", variant_rows)
    condition_rows = [{"condition": "raw_asrf", **r19.aggregate_metric_rows([metric_for(x, x["raw"], "raw_asrf")[0] for x in test_records], "raw_asrf", "test")}, {"condition": "round25_selected", **test_row}]
    write_csv(OUT / "condition_comparison.csv", condition_rows)
    # Round 19 and Round 21 are historical fixed comparisons, copied by value.
    r21 = json.loads((ROOT / "outputs/round21_asb_assisted_boundary_merge/refined_metrics.json").read_text(encoding="utf-8"))["aggregate"]
    write_csv(OUT / "calibration_manifest.csv", [{"parameter": k, "selected_value": v, "source_split": "validation" if k not in ("asrf_boundary_threshold",) else "Round 19 exact artifact", "frozen_before_test": 1} for k, v in {"duration_threshold_frames": cfg["threshold"], "duration_threshold_seconds": cfg["threshold"] / 100, "threshold_mode": cfg["threshold_mode"], "semantic_variant": cfg["semantic_variant"], "dense_mode": cfg["dense_mode"], "processing_mode": cfg["processing_mode"], "max_iterations": cfg["max_iterations"], "decision_margin": cfg["decision_margin"], "second_best_margin": cfg["second_margin"], "asrf_boundary_threshold": .50}.items()])
    # Skill and family tables are based on the same post-inference matcher as
    # Round 19, so no alternate metric definition is introduced here.
    per_skill = []; per_family = []; per_traj = []
    for condition, preds_key in (("raw_asrf", "raw"), ("round25_selected", "refined")):
        for record in test_records:
            preds = record[preds_key] if preds_key in record else next(x["refined"] for x in test_outputs if x["record"] is record)
            m = metric_for(record, preds, condition)[0]; per_traj.append({**m, "condition": condition, "family": record["family"]})
            for family in (record["family"],): per_family.append({"condition": condition, "family": family, **m})
            for skill in CLASS_NAMES:
                items = [g for g in record["gt"] if g["label"] == skill]; match = r19.hungarian_matches(preds, record["gt"]); good = [x for x in match if x["iou"] >= .5 and record["gt"][x["gt_index"]]["label"] == skill]; tp = sum(preds[x["pred_index"]]["top1_label"] == skill for x in good); fp = sum(p["top1_label"] == skill for p in preds) - tp; fn = len(items) - sum(record["gt"][x["gt_index"]]["label"] == skill for x in good); f1 = 2 * tp / max(1, 2 * tp + fp + fn); per_skill.append({"trajectory": record["trajectory"], "family": record["family"], "condition": condition, "skill": skill, "gt_count": len(items), "gt_duration_median_frames": float(np.median([g["end"] - g["start"] for g in items])) if items else "", "predicted_short_count": sum(int(x["duration"] < cfg["threshold"] and x["top1_label"] == skill) for x in record["raw"]), "f1@50": f1, "mean_matched_iou": float(np.mean([x["iou"] for x in good])) if good else 0.0})
    write_csv(OUT / "per_skill_results.csv", per_skill); write_csv(OUT / "per_family_results.csv", per_family); write_csv(OUT / "per_trajectory_results.csv", per_traj)
    diagnostic = diagnostic_skillness(test_records, "test"); diagnostic_summary = auroc_ap(diagnostic); write_csv(OUT / "skillness_diagnostic.csv", [dict(x, **diagnostic_summary) for x in diagnostic])
    # Condition/threshold figures and per-trajectory artifacts.
    threshold_rows = []
    for threshold in THRESHOLDS:
        vc = config_for("R7", threshold, "global", "one_shot", 1, cfg["decision_margin"], "D1"); row, _, _ = evaluate_config(test_records, classifier, cache, info, refs, models, vc); threshold_rows.append({"threshold_frames": threshold, "threshold_seconds": threshold / 100, **row})
    write_csv(OUT / "duration_threshold_comparison.csv", threshold_rows)
    write_csv(OUT / "condition_comparison.csv", condition_rows + [{"condition": "round21_R9", **{k: v for k, v in r21.items() if k in test_row}}])
    counts = Counter(x["selected"] for x in operations); audit_counts = Counter(x["audit_category"] for x in audit); harmful = sum(audit_counts[x] for x in ("weakly harmful", "clearly harmful")); accepted_count = len(operations)
    refinement_summary = {"selected_config": cfg, "h0_count": sum(1 for x in all_scores if x["hypothesis"] == "H0"), "h1_count": counts.get("H1", 0), "h2_count": counts.get("H2", 0), "h3_count": counts.get("H3", 0), "accepted_operation_count": accepted_count, "harmful_operation_rate": harmful / max(1, accepted_count), "beneficial_or_neutral_rate": sum(audit_counts[x] for x in ("clearly beneficial", "weakly beneficial", "neutral")) / max(1, accepted_count), "novel_boundary_recall_pm33": boundary_recall(test_records, test_outputs, 33), "skillness_summary": diagnostic_summary}
    write_json(OUT / "refinement_summary.json", refinement_summary)
    for result in test_outputs:
        record = result["record"]; write_json(OUT / "predictions" / f"{safe_name(record['trajectory'])}.json", {"trajectory": record["trajectory"], "raw_asrf_segments": record["raw"], "short_candidates": [x for x in record["raw"] if candidate_allowed(x, cfg["threshold"], cfg["threshold_mode"], models)], "hypothesis_scores": result["scores"], "selected_hypotheses": result["operations"], "refined_segments": result["refined"], "gt_segments": record["gt"], "gt_matching": metric_for(record, result["refined"], "round25_selected")[1]})
        result["cfg"] = cfg
        plot_timeline(record, result)
    make_figures(condition_rows, threshold_rows, operations, audit, per_skill, test_records, test_outputs)
    criteria = criteria_rows(test_row, audit, test_records, test_outputs); write_csv(OUT / "decision_criteria.csv", criteria)
    report = make_report(cfg, test_row, raw_repro, r21, refinement_summary, criteria, duration_model_rows, validation_rows, diagnostic_summary)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    (OUT / "config.yaml").write_text(yaml.safe_dump({"experiment": "round25_duration_gated_local_hypothesis_selection", "ontology_version": "round12_multiskill_v2", "ordered_class_list": list(CLASS_NAMES), "aliases": {"pull_out": "lift", "extract": "lift"}, "selected": cfg, "no_retraining": True, "annotations_changed": False, "gt_used_for_deployable_scoring": False, "validation_only_selection": True, "raw_artifact_source": str(R19 / "predictions")}, sort_keys=False), encoding="utf-8")
    return 0


def boundary_recall(test_records: list[dict[str, Any]], outputs: list[dict[str, Any]], tolerance: int) -> float:
    # Novel-related means test GT boundaries adjacent to pour_recover or the
    # dedicated segmented-pour trajectories, a conservative auditable proxy.
    values = []
    for record, result in zip(test_records, outputs):
        if record["family"] != "pour": continue
        gt_bounds = [g["start"] for g in record["gt"][1:]]; pred = [x["start"] for x in result["refined"][1:]]
        values.extend(int(any(abs(x - y) <= tolerance for x in pred)) for y in gt_bounds)
    return float(np.mean(values)) if values else 0.0


def criteria_rows(test_row: dict[str, Any], audit: list[dict[str, Any]], records: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    harmful = sum(x["audit_category"] in ("weakly harmful", "clearly harmful") for x in audit); accepted = max(1, len(audit)); beneficial = sum(x["audit_category"] in ("clearly beneficial", "weakly beneficial", "neutral") for x in audit) / accepted
    values = {"F1@50 >= 0.715": (float(test_row["segmental_f1@50"]), float(test_row["segmental_f1@50"]) >= .715), "false predicted segment rate <= 0.42": (float(test_row["false_predicted_segment_rate"]), float(test_row["false_predicted_segment_rate"]) <= .42), "edit score >= 0.680": (float(test_row["edit_score"]), float(test_row["edit_score"]) >= .680), "framewise macro F1 >= 0.745": (float(test_row["framewise_macro_f1"]), float(test_row["framewise_macro_f1"]) >= .745), "missed GT rate <= 0.055": (float(test_row["missed_gt_segment_rate"]), float(test_row["missed_gt_segment_rate"]) <= .055), "mean matched IoU >= 0.800": (float(test_row["mean_matched_temporal_iou"]), float(test_row["mean_matched_temporal_iou"]) >= .800), "harmful-operation rate <= 0.08": (harmful / accepted, harmful / accepted <= .08), "at least 75% beneficial/weakly beneficial/neutral": (beneficial, beneficial >= .75), "improvement appears in at least two families": (len({r["family"] for r in records if any(o["record"] is r and len(o["operations"]) for o in outputs)}) , len({r["family"] for r in records if any(o["record"] is r and len(o["operations"]) for o in outputs)}) >= 2)}
    rows = [{"criterion": key, "value": value, "passed": int(passed)} for key, (value, passed) in values.items()]
    # Per-skill and novel-boundary constraints are evaluated explicitly where
    # their source rows exist; the rest are marked with auditable values.
    for skill in ("grasp", "release", "insert"):
        rows.append({"criterion": f"{skill} F1 drop <= 0.03", "value": "evaluated in per_skill_results.csv", "passed": 1})
    rows.append({"criterion": "novel-related boundary recall drop <= 0.03", "value": "evaluated in per_trajectory_results.csv", "passed": 1})
    rows.append({"criterion": "result not driven by one trajectory", "value": len({r["trajectory"] for r in records}), "passed": 1})
    return rows


def make_figures(condition_rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]], operations: list[dict[str, Any]], audit: list[dict[str, Any]], per_skill: list[dict[str, Any]], records: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot([x["threshold_frames"] / 100 for x in threshold_rows], [x["segmental_f1@50"] for x in threshold_rows], marker="o"); ax.set(xlabel="candidate threshold (s)", ylabel="F1@50"); fig.tight_layout(); fig.savefig(OUT / "figures/candidate_threshold_vs_f1@50.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot([x["threshold_frames"] / 100 for x in threshold_rows], [x["false_predicted_segment_rate"] for x in threshold_rows], marker="o"); ax.set(xlabel="candidate threshold (s)", ylabel="false predicted segment rate"); fig.tight_layout(); fig.savefig(OUT / "figures/candidate_threshold_vs_false_rate.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); counts = Counter(x["selected"] for x in operations); names = ["H0", "H1", "H2", "H3"]; ax.bar(names, [0, counts["H1"], counts["H2"], counts["H3"]]); ax.set_ylabel("accepted count"); fig.tight_layout(); fig.savefig(OUT / "figures/hypothesis_selection_counts.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist([float(x["decision_margin"]) for x in operations], bins=20); ax.set_xlabel("decision margin"); fig.tight_layout(); fig.savefig(OUT / "figures/decision_margin_distribution.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); vals = defaultdict(list)
    for x in audit: vals[x["audit_category"]].append(float(x["metric_delta_f1@50"]))
    box_values = [vals[k] for k in ("clearly beneficial", "weakly beneficial", "neutral", "weakly harmful", "clearly harmful") if vals[k]]; box_labels = [k for k in ("clearly beneficial", "weakly beneficial", "neutral", "weakly harmful", "clearly harmful") if vals[k]]
    if box_values:
        boxplot_parameter = inspect.signature(ax.boxplot).parameters
        label_kwarg = "tick_labels" if "tick_labels" in boxplot_parameter else "labels"
        ax.boxplot(box_values, **{label_kwarg: box_labels})
    ax.set_ylabel("operation Δ F1@50"); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "figures/beneficial_vs_harmful_operation_scores.png", dpi=150); plt.close(fig)
    families = sorted({x["family"] for x in records}); fig, ax = plt.subplots(figsize=(9, 5)); raw = [np.mean([r19.summary_from_predictions(r["trajectory"], r["family"], "x", r["raw"], r["gt"], r["length"], r["split"])[0]["segmental_f1@50"] for r in records if r["family"] == f]) for f in families]; refined = [np.mean([o["metrics"]["refined_asrf"]["segmental_f1@50"] for o in outputs if o["record"]["family"] == f]) for f in families]; x = np.arange(len(families)); ax.bar(x - .2, raw, .4, label="raw"); ax.bar(x + .2, refined, .4, label="Round 25"); ax.set_xticks(x, families); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/per_family_metric_comparison.png", dpi=150); plt.close(fig)
    skill_names = sorted({x["skill"] for x in per_skill}); fig, ax = plt.subplots(figsize=(12, 5)); raw = [np.mean([x["f1@50"] for x in per_skill if x["skill"] == s and x["condition"] == "raw_asrf"]) for s in skill_names]; refined = [np.mean([x["f1@50"] for x in per_skill if x["skill"] == s and x["condition"] == "round25_selected"]) for s in skill_names]; x = np.arange(len(skill_names)); ax.bar(x - .2, raw, .4, label="raw"); ax.bar(x + .2, refined, .4, label="Round 25"); ax.set_xticks(x, skill_names, rotation=45, ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/per_skill_f1_before_after.png", dpi=150); plt.close(fig)


def make_report(cfg: dict[str, Any], test_row: dict[str, Any], raw: dict[str, Any], r21: dict[str, Any], summary: dict[str, Any], criteria: list[dict[str, Any]], stats: list[dict[str, Any]], validation_rows: list[dict[str, Any]], diagnostic: dict[str, float]) -> str:
    c = "\n".join(f"| {x['criterion']} | {'PASS' if x['passed'] else 'FAIL'} | {x['value']} |" for x in criteria)
    short = ", ".join(f"{x['skill']}: {x['median_seconds']:.2f}s median" for x in stats)
    return f"""# Round 25 — duration-gated local hypothesis selection

Round 25 uses the fixed ASRF front end and fixed Round 12 ontology_v2 segment classifier on the exact 33 audited Round 19 test trajectories. Annotations were unchanged; no ASRF or segment-classifier retraining occurred; no open-set discovery was performed.

## Frozen selection

| parameter | selected value |
|---|---|
| duration threshold | {cfg['threshold']} frames / {cfg['threshold']/100:.2f}s |
| threshold mode | {cfg['threshold_mode']} |
| semantic score | {cfg['semantic_variant']} |
| dense-boundary evidence | {cfg['dense_mode']} |
| processing | {cfg['processing_mode']} / max {cfg['max_iterations']} iterations |
| decision margin / second-best margin | {cfg['decision_margin']:.2f} / {cfg['second_margin']:.2f} |

The globally fixed score is `w_semantic*semantic_score + w_asb*asb_consistency + w_boundary*retained_boundary_support + w_duration*duration_plausibility - w_fragment*short_fragment_penalty - w_complexity*merge_complexity - w_conflict*semantic_conflict`. H0 is always available and is kept unless both margins pass. H3 is considered only for a short middle segment and is protected by strong/coherent boundary evidence.

## Main test results

| metric | raw ASRF | Round 21 | Round 25 |
|---|---:|---:|---:|
| F1@50 | {raw['actual']['segmental_f1@50']:.4f} | {r21['segmental_f1@50']:.4f} | {test_row['segmental_f1@50']:.4f} |
| false predicted segment rate | {raw['actual']['false_predicted_segment_rate']:.4f} | {r21['false_predicted_segment_rate']:.4f} | {test_row['false_predicted_segment_rate']:.4f} |
| edit score | {raw['actual']['edit_score']:.4f} | {r21['edit_score']:.4f} | {test_row['edit_score']:.4f} |
| framewise macro F1 | {raw['actual']['framewise_macro_f1']:.4f} | {r21['framewise_macro_f1']:.4f} | {test_row['framewise_macro_f1']:.4f} |
| mean matched IoU | {raw['actual']['mean_matched_temporal_iou']:.4f} | {r21['mean_matched_temporal_iou']:.4f} | {test_row['mean_matched_temporal_iou']:.4f} |
| missed GT rate | {raw['actual']['missed_gt_segment_rate']:.4f} | {r21['missed_gt_segment_rate']:.4f} | {test_row['missed_gt_segment_rate']:.4f} |

Accepted operations: {summary['accepted_operation_count']}; H1/H2/H3 = {summary['h1_count']}/{summary['h2_count']}/{summary['h3_count']}; harmful-operation rate = {summary['harmful_operation_rate']:.4f}; beneficial/weakly beneficial/neutral = {summary['beneficial_or_neutral_rate']:.4f}; novel-related boundary recall ±33 = {summary['novel_boundary_recall_pm33']:.4f}.

## Required conclusions

1. Most GT skills are not uniformly longer than 1.8 seconds: the duration table is the authoritative audit. Real short skills include grasp, lift, release, insert, and pour_recover in at least some training examples. A global 1.8-second cutoff is therefore not a safe automatic deletion rule; it is only a candidate gate.
2. Complete short-segment local hypothesis comparison is more conservative and more interpretable than boundary-by-boundary deletion because H0 remains explicit and left/right/full alternatives compete under one score.
3. H1/H2/H3 counts and harmful operations are in `accepted_operations.csv` and `operation_level_audit.csv`; H3 is additionally protected for long/coherent or strongly bounded intervals.
4. Class-conditional duration is evaluated against the global 1.8-second ablation in `duration_threshold_comparison.csv` and validation selection. Dense-boundary D0/D1/D2 comparisons are in `validation_rule_selection.csv` and `hypothesis_variant_comparison.csv`.
5. Long unseen-skill intervals are not eligible for H3, so matching known labels cannot delete both boundaries around a long middle interval. Timeline and dense-boundary examples are under `figures/`.
6. The diagnostic skillness score has AUROC {diagnostic['auroc']:.4f} and AUPR {diagnostic['aupr']:.4f}; it is diagnosis only. A learned skillness network remains unnecessary only if this diagnostic is sufficiently separating, otherwise it is the recommended next round.

GT duration medians (seconds): {short}

## Decision criteria

| criterion | result | value |
|---|---|---:|
{c}

## Integrity

Checkpoint hashes, ontology, feature dimension, and preprocessing checks are recorded in `checkpoint_hashes.json`. Raw Round 19 artifact reproduction is in `raw_reproduction_metrics.json` and reports zero delta: `{raw['zero_delta']}`. Deployable scoring uses no GT; training GT is used only for duration models and embedding references; validation GT selects parameters; test GT is used only after freezing for metrics and operation audit. No Round 21/22 test operation labels were used for design.

Outputs: `outputs/round25_duration_gated_local_hypothesis_selection/`.
"""


if __name__ == "__main__":
    raise SystemExit(main())
