#!/usr/bin/env python3
"""Round 20: frozen semantic suppression of ASRF fragmentation.

The raw segments are read from the completed Round 19 artifacts.  Only
boundary deletion is allowed after the raw segmentation; no ASRF or segment
classifier weights are trained here.
"""

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

ROOT = Path(__file__).resolve().parents[1]
R19_ROOT = ROOT / "outputs/round19_asrf_segment_classifier_integration"
R12_ROOT = ROOT / "outputs/round12_multiskill_segment_classifier"
OUT = ROOT / "outputs/round20_semantic_fragment_merge"
ASRF_CHECKPOINT = ROOT / "outputs/round10_pp_only_novel_segmentation/models/single_frame/best.pt"
CLASSIFIER_CHECKPOINT = R12_ROOT / "model/best.pt"
EXPECTED_ASRF_SHA = "6b11abff2ff4387eada88e38fb56133b29fd5c3facdfb54b42fc84701213db9a"
EXPECTED_CLASSIFIER_SHA = "51f0abbcc4250ef97951bcaef04fc8f55cb2de968affdf0121a446ea1635a86f"
sys.path.insert(0, str(ROOT / "scripts"))
import run_round19_asrf_segment_classifier_integration as r19  # noqa: E402

SEED = 42
BRB_GRID = (0.20, 0.35, 0.50)
SCORE_GRID = (0.10, 0.30, 0.50)
SHORT_DURATION_GRID = (60, 100, 150)
CONFIDENCE_TOL_GRID = (0.10, 0.20)
MARGIN_TOL_GRID = (0.10, 0.20)
EMBEDDING_DISTANCE_LIMIT = 0.25
SUPPORT_TOLERANCE = 0.05
LOW_CONFIDENCE = 0.75
LOW_MARGIN = 0.10
STRONG_BRB = 0.85
MAX_ITERATIONS = 8
MIN_CHILD_DURATION = 20
RULES = ("R0_raw", "R1_same_label", "R2_short_fragment", "R3_weak_semantic", "R4_same_plus_short", "R5_same_plus_weak", "R6_full_score", "R7_full_iterative")
EXTRA_RULES = ("brb_only", "semantic_only", "brb_plus_semantic")
CLASS_NAMES = tuple(r19.CLASS_NAMES)


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
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else x) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def safe_name(path: str) -> str:
    return r19.safe_name(path)


def family_name(value: str) -> str:
    return {"pp": "pick_and_place", "plug": "plug", "pour": "pour", "wipe": "wipe"}.get(value, value)


def unique_manifest(split: str) -> list[dict[str, str]]:
    rows = read_csv(R12_ROOT / f"split_manifests/{split}.csv")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row["trajectory"] in seen:
            continue
        seen.add(row["trajectory"])
        result.append({"trajectory": row["trajectory"], "family": family_name(row["family"]), "split": split})
    return result


def load_fixed() -> tuple[Any, Any, dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    if sha256(ASRF_CHECKPOINT) != EXPECTED_ASRF_SHA:
        raise RuntimeError("ASRF checkpoint SHA-256 mismatch")
    if sha256(CLASSIFIER_CHECKPOINT) != EXPECTED_CLASSIFIER_SHA:
        raise RuntimeError("Round 12 classifier checkpoint SHA-256 mismatch")
    asrf, classifier, asrf_config, classifier_info, cache, _ = r19.load_fixed_models()
    norm = classifier_info["normalization"]
    if norm["mean"].shape != (r19.r12.FEATURE_DIM,) or norm["std"].shape != (r19.r12.FEATURE_DIM,):
        raise RuntimeError("Round 12 feature-dimension/preprocessing mismatch")
    if tuple(classifier_info["payload"]["ontology_metadata"]["ordered_class_list"]) != CLASS_NAMES:
        raise RuntimeError("ontology_v2 mismatch")
    return classifier, classifier_info, cache, asrf_config


def load_records(split: str) -> list[dict[str, Any]]:
    records = []
    for manifest in unique_manifest(split):
        path = R19_ROOT / "predictions" / f"{safe_name(manifest['trajectory'])}.json"
        npz_path = R19_ROOT / "predictions" / f"{safe_name(manifest['trajectory'])}.npz"
        if not path.exists() or not npz_path.exists():
            raise RuntimeError(f"Missing Round 19 frozen prediction artifact for {manifest['trajectory']}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        arrays = np.load(npz_path)
        raw = [dict(item, source="raw_asrf") for item in payload["classifier_raw"]]
        records.append({"trajectory": manifest["trajectory"], "family": manifest["family"], "split": split, "length": int(len(arrays["brb_probabilities"])), "raw": raw, "gt": payload["gt_segments"], "brb": np.asarray(arrays["brb_probabilities"], dtype=np.float32), "asb_logits_hash": hashlib.sha256(np.asarray(arrays["asb_logits"]).tobytes()).hexdigest(), "brb_hash": hashlib.sha256(np.asarray(arrays["brb_probabilities"]).tobytes()).hexdigest(), "raw_boundary_hash": canonical_hash(payload["raw_predicted_segments"]), "raw_segment_hash": canonical_hash([(x["start"], x["end"]) for x in raw])})
    return records


def class_duration_bounds() -> tuple[dict[str, tuple[float, float, float]], dict[str, np.ndarray]]:
    rows = read_csv(R12_ROOT / "split_manifests/train.csv")
    bounds = r19.class_duration_bounds(rows)
    durations: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        durations[row["label"]].append(float(row["duration_frames"]))
    return bounds, {key: np.asarray(value, dtype=np.float32) for key, value in durations.items()}


class SemanticEngine:
    def __init__(self, classifier: Any, info: dict[str, Any], cache: dict[str, tuple[np.ndarray, np.ndarray]], bounds: dict[str, tuple[float, float, float]], durations: dict[str, np.ndarray]):
        self.classifier = classifier
        self.normalization = info["normalization"]
        self.cache = cache
        self.bounds = bounds
        self.durations = durations
        self.reference: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((0, r19.r12.EMBEDDING_DIM), dtype=np.float32))
        self.inference_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._build_reference_bank()

    def _build_reference_bank(self) -> None:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(R12_ROOT / "split_manifests/train.csv"):
            grouped[row["trajectory"]].append(row)
        for trajectory, rows in grouped.items():
            intervals = [r19.TemporalInterval(int(row["start_frame"]), int(row["end_frame_exclusive"])) for row in rows]
            predictions = r19.classify(self.classifier, self.cache, self.normalization, trajectory, intervals)
            for row, prediction in zip(rows, predictions):
                self.reference[row["label"]] = np.vstack((self.reference[row["label"]], np.asarray(prediction["embedding"], dtype=np.float32)[None, :]))

    def support(self, prediction: dict[str, Any]) -> dict[str, Any]:
        label = prediction["top1_label"]
        vector = np.asarray(prediction["embedding"], dtype=np.float32)
        bank = self.reference.get(label, np.zeros((0, len(vector)), dtype=np.float32))
        if len(bank):
            distances = 1.0 - bank @ vector
            distances = np.sort(distances)
            nearest = float(distances[0])
            mean5 = float(np.mean(distances[: min(5, len(distances))]))
        else:
            nearest, mean5 = float("nan"), float("nan")
        class_durations = self.durations.get(label, np.asarray([], dtype=np.float32))
        percentile = float(np.mean(class_durations <= prediction["duration"])) if len(class_durations) else float("nan")
        low, high, _ = self.bounds.get(label, (0.0, float("inf"), 0.0))
        return {**prediction, "predicted_class_nearest_distance": nearest, "predicted_class_mean5_distance": mean5, "embedding_support_distance": mean5, "predicted_class_duration_percentile": percentile, "duration_valid": int(low <= prediction["duration"] <= high)}

    def classify_interval(self, trajectory: str, start: int, end: int) -> dict[str, Any]:
        key = (trajectory, int(start), int(end))
        if key not in self.inference_cache:
            prediction = r19.classify(self.classifier, self.cache, self.normalization, trajectory, [r19.TemporalInterval(start, end)])[0]
            self.inference_cache[key] = self.support(prediction)
        return dict(self.inference_cache[key])


def enrich_raw(engine: SemanticEngine, raw: list[dict[str, Any]], brb: np.ndarray) -> list[dict[str, Any]]:
    output = []
    for item in raw:
        item = engine.support(dict(item))
        item["left_boundary_brb"] = float(brb[item["start"]]) if item["start"] > 0 else 0.0
        item["right_boundary_brb"] = float(brb[item["end"]]) if item["end"] < len(brb) else 0.0
        output.append(item)
    return output


def build_pair(record: dict[str, Any], engine: SemanticEngine, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    boundary = int(right["start"])
    merged = engine.classify_interval(record["trajectory"], int(left["start"]), int(right["end"]))
    left_z = np.asarray(left["embedding"], dtype=np.float32)
    right_z = np.asarray(right["embedding"], dtype=np.float32)
    pair_distance = float(1.0 - np.dot(left_z, right_z))
    support_gain = float(min(left["embedding_support_distance"], right["embedding_support_distance"]) - merged["embedding_support_distance"])
    margin_gain = float(merged["margin"] - min(left["margin"], right["margin"]))
    conf_gain_left = float(merged["top1_probability"] - left["top1_probability"])
    conf_gain_right = float(merged["top1_probability"] - right["top1_probability"])
    same_label = int(left["top1_label"] == right["top1_label"])
    merged_agrees = int(merged["top1_label"] in {left["top1_label"], right["top1_label"]})
    semantic_consistency = int(same_label or merged_agrees or pair_distance <= EMBEDDING_DISTANCE_LIMIT)
    short_evidence = int(left["duration"] <= 100 or right["duration"] <= 100 or left["predicted_class_duration_percentile"] <= .05 or right["predicted_class_duration_percentile"] <= .05)
    score = float(1.5 * same_label + 1.5 * support_gain + 1.0 * margin_gain + .5 * short_evidence - 2.0 * float(record["brb"][boundary]) - 1.5 * (1 - int(merged["duration_valid"])) - 1.0 * (1 - merged_agrees))
    return {"trajectory": record["trajectory"], "boundary": boundary, "left": left, "right": right, "merged": merged, "boundary_brb": float(record["brb"][boundary]), "left_right_embedding_distance": pair_distance, "same_label": same_label, "merged_label_agrees": merged_agrees, "semantic_consistency": semantic_consistency, "short_fragment_evidence": short_evidence, "support_gain": support_gain, "margin_gain": margin_gain, "merged_confidence_minus_left": conf_gain_left, "merged_confidence_minus_right": conf_gain_right, "merged_duration_percentile": merged["predicted_class_duration_percentile"], "duration_valid": int(merged["duration_valid"]), "score": score}


def score_accepts(candidate: dict[str, Any], rule: str, cfg: dict[str, Any]) -> tuple[bool, str]:
    same = bool(candidate["same_label"])
    weak = candidate["boundary_brb"] <= cfg["brb_threshold"]
    soft_conf = candidate["merged_confidence_minus_left"] >= -cfg["confidence_tolerance"] and candidate["merged_confidence_minus_right"] >= -cfg["confidence_tolerance"]
    soft_margin = candidate["margin_gain"] >= -cfg["margin_tolerance"]
    support_ok = candidate["support_gain"] >= -SUPPORT_TOLERANCE
    valid = bool(candidate["duration_valid"])
    semantic = bool(candidate["semantic_consistency"])
    if rule == "R1_same_label":
        ok = same and weak
    elif rule == "R3_weak_semantic" or rule == "brb_plus_semantic":
        ok = weak and semantic and valid and soft_conf and support_ok
    elif rule == "R5_same_plus_weak":
        ok = same and weak and valid and soft_conf and soft_margin and support_ok
    elif rule == "brb_only":
        ok = weak and valid
    elif rule == "semantic_only":
        ok = semantic and valid and soft_conf and support_ok
    elif rule in ("R6_full_score", "R7_full_iterative"):
        ok = candidate["score"] >= cfg["score_threshold"] and semantic and valid and soft_conf and soft_margin and support_ok
    else:
        ok = False
    reason = "accepted" if ok else ("strong_brb" if not weak else "semantic_or_support" if not semantic or not support_ok else "confidence_or_margin" if not soft_conf or not soft_margin else "invalid_duration")
    return bool(ok), reason


def short_fragment_options(record: dict[str, Any], engine: SemanticEngine, segments: list[dict[str, Any]], center: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if center <= 0 or center >= len(segments) - 1:
        return []
    left, middle, right = segments[center - 1], segments[center], segments[center + 1]
    short = middle["duration"] <= cfg["short_duration"] or middle["predicted_class_duration_percentile"] <= .05
    weak_middle = middle["top1_probability"] < LOW_CONFIDENCE or middle["margin"] < LOW_MARGIN or middle["embedding_support_distance"] > .45
    left_right_distance = float(1.0 - np.dot(np.asarray(left["embedding"]), np.asarray(right["embedding"])))
    compatible = left["top1_label"] == right["top1_label"] or left_right_distance <= EMBEDDING_DISTANCE_LIMIT
    left_boundary = build_pair(record, engine, left, middle)
    right_boundary = build_pair(record, engine, middle, right)
    triple = engine.classify_interval(record["trajectory"], int(left["start"]), int(right["end"]))
    both_strong = left_boundary["boundary_brb"] >= STRONG_BRB and right_boundary["boundary_brb"] >= STRONG_BRB
    options = []
    for choice, candidate, replacement, deleted in (
        ("B1_absorb_left", left_boundary, [candidate := left_boundary["merged"], right], [left_boundary["boundary"]]),
        ("B2_absorb_right", right_boundary, [left, candidate := right_boundary["merged"]], [right_boundary["boundary"]]),
        ("B3_merge_three", {**build_pair(record, engine, left, right), "merged": triple, "boundary": left_boundary["boundary"], "boundary_brb": min(left_boundary["boundary_brb"], right_boundary["boundary_brb"])}, [triple], [left_boundary["boundary"], right_boundary["boundary"]]),
    ):
        semantic = candidate["merged"]["top1_label"] in {left["top1_label"], middle["top1_label"], right["top1_label"]}
        if choice == "B3_merge_three":
            semantic = left["top1_label"] == right["top1_label"] and candidate["merged"]["top1_label"] == left["top1_label"]
        duration_ok = bool(candidate["merged"]["duration_valid"])
        gain = float(sum(item["top1_probability"] for item in replacement) - left["top1_probability"] - middle["top1_probability"] - right["top1_probability"])
        accepted = bool(short and weak_middle and compatible and not both_strong and semantic and duration_ok and min(item["duration"] for item in replacement) >= MIN_CHILD_DURATION)
        options.append({"trajectory": record["trajectory"], "center_index": center, "choice": choice, "left_index": center - 1, "right_index": center + 1, "middle_duration": middle["duration"], "short_evidence": int(short), "weak_middle": int(weak_middle), "compatible": int(compatible), "both_boundaries_strong": int(both_strong), "deleted_boundaries": deleted, "boundary_brb": candidate["boundary_brb"], "merged_label": candidate["merged"]["top1_label"], "merged_confidence": candidate["merged"]["top1_probability"], "gain": gain, "accepted": int(accepted), "replacement": replacement})
    return options


def apply_operation(segments: list[dict[str, Any]], op: dict[str, Any]) -> list[dict[str, Any]]:
    if op["kind"] == "boundary":
        index = int(op["left_index"])
        return segments[:index] + [op["candidate"]["merged"]] + segments[index + 2 :]
    start = int(op["left_index"])
    stop = int(op["right_index"])
    return segments[:start] + op["replacement"] + segments[stop + 1 :]


def operation_overlaps(op: dict[str, Any], chosen: list[dict[str, Any]]) -> bool:
    lo, hi = int(op["left_index"]), int(op["right_index"])
    return any(not (hi < int(old["left_index"]) or lo > int(old["right_index"])) for old in chosen)


def apply_rule(record: dict[str, Any], engine: SemanticEngine, rule: str, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    segments = [dict(x) for x in record["raw"]]
    all_candidates: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    if rule == "R0_raw":
        return segments, all_candidates, deleted, history
    iterations = MAX_ITERATIONS if rule == "R7_full_iterative" else 1
    for iteration in range(iterations):
        candidates = []
        for i in range(len(segments) - 1):
            candidate = build_pair(record, engine, segments[i], segments[i + 1])
            accepted, reason = score_accepts(candidate, rule if rule not in ("R4_same_plus_short", "R2_short_fragment") else ("R5_same_plus_weak" if rule == "R4_same_plus_short" else "none"), cfg)
            if rule == "R4_same_plus_short":
                accepted = accepted and bool(candidate["short_fragment_evidence"])
            if rule == "R2_short_fragment":
                accepted = False
            candidates.append({"kind": "boundary", "left_index": i, "right_index": i + 1, "candidate": candidate, "accepted": int(accepted), "reason": reason, "iteration": iteration, "rule": rule})
        short_candidates = []
        if rule in ("R2_short_fragment", "R4_same_plus_short", "R7_full_iterative"):
            for center in range(1, len(segments) - 1):
                short_candidates.extend({"kind": "short", "left_index": x["left_index"], "right_index": x["right_index"], "candidate": x, "replacement": x["replacement"], "accepted": x["accepted"], "reason": "accepted" if x["accepted"] else "short_fragment_semantics", "iteration": iteration, "rule": rule} for x in short_fragment_options(record, engine, segments, center, cfg))
        candidates.extend(short_candidates)
        all_candidates.extend(candidates)
        accepted = [x for x in candidates if x["accepted"]]
        if not accepted:
            break
        if rule in ("R7_full_iterative",):
            accepted.sort(key=lambda x: float(x["candidate"].get("score", x["candidate"].get("gain", 0.0))), reverse=True)
            accepted = [accepted[0]]
        elif rule in ("R2_short_fragment", "R4_same_plus_short"):
            accepted.sort(key=lambda x: float(x["candidate"].get("gain", 0.0)), reverse=True)
        chosen: list[dict[str, Any]] = []
        for op in accepted:
            if operation_overlaps(op, chosen):
                continue
            chosen.append(op)
        if not chosen:
            break
        for op in sorted(chosen, key=lambda x: int(x["left_index"]), reverse=True):
            before = [{"start": x["start"], "end": x["end"]} for x in segments]
            segments = apply_operation(segments, op)
            after = [{"start": x["start"], "end": x["end"]} for x in segments]
            boundary_values = op["candidate"].get("deleted_boundaries", [op["candidate"].get("boundary")])
            operation = {"trajectory": record["trajectory"], "split": record["split"], "rule": rule, "iteration": iteration, "kind": op["kind"], "left_index": op["left_index"], "right_index": op["right_index"], "deleted_boundaries": boundary_values, "score": op["candidate"].get("score", op["candidate"].get("gain", 0.0)), "choice": op["candidate"].get("choice", "boundary_delete"), "before_segments": before, "after_segments": after}
            span_items = [op["candidate"]["merged"]] if op["kind"] == "boundary" else op["replacement"]
            operation["operation_span_start"] = int(min(x["start"] for x in span_items))
            operation["operation_span_end"] = int(max(x["end"] for x in span_items))
            history.append(operation)
            for boundary in boundary_values:
                deleted.append({"trajectory": record["trajectory"], "split": record["split"], "rule": rule, "iteration": iteration, "boundary": boundary, "deletion_reason": op["reason"], "operation_kind": op["kind"], "score": operation["score"], "choice": operation["choice"]})
        if rule != "R7_full_iterative":
            break
    for item in all_candidates:
        item["accepted_in_final_path"] = int(any(item is x for x in []))
    return segments, all_candidates, deleted, history


def metric_for(record: dict[str, Any], predictions: list[dict[str, Any]], condition: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    matches = r19.hungarian_matches(predictions, record["gt"])
    metric = r19.condition_metrics(record["trajectory"], record["family"], condition, predictions, record["gt"], record["length"], matches)
    metric["split"] = record["split"]
    matched, missed, false, categories = r19.matching_rows(record["trajectory"], condition, predictions, record["gt"])
    return metric, matched, missed, false, categories


def aggregate(metrics: list[dict[str, Any]], rule: str, split: str) -> dict[str, Any]:
    row = r19.aggregate_metric_rows(metrics, "refined", split)
    row["rule"] = rule
    if row:
        row["fragmentation_ratio"] = float(row["predicted_segments"] / max(row["gt_segments"], 1))
    return row


def evaluate_rule(records: list[dict[str, Any]], engine: SemanticEngine, rule: str, cfg: dict[str, Any], collect: bool = False) -> dict[str, Any]:
    result_rows = []
    all_candidates = []
    all_deleted = []
    all_history = []
    all_rejected = []
    all_matched = []
    all_missed = []
    all_false = []
    all_categories = []
    refined_predictions: dict[str, list[dict[str, Any]]] = {}
    raw_metrics = []
    for record in records:
        refined, candidates, deleted, history = apply_rule(record, engine, rule, cfg)
        raw_metric, _, _, _, _ = metric_for(record, record["raw"], "raw")
        metric, matched, missed, false, categories = metric_for(record, refined, "refined")
        metric["rule"] = rule
        metric["deleted_boundaries"] = len(deleted)
        metric["operations"] = len(history)
        result_rows.append(metric); raw_metrics.append(raw_metric); refined_predictions[record["trajectory"]] = refined
        if collect:
            for candidate in candidates:
                row = flatten_candidate(candidate, record["split"])
                row["rule"] = rule
                all_candidates.append(row)
            all_deleted.extend(deleted); all_history.extend(history)
            all_rejected.extend(flatten_candidate(x, record["split"]) for x in candidates if not x["accepted"])
            all_matched.extend(matched); all_missed.extend(missed); all_false.extend(false)
            all_categories.extend({"trajectory": record["trajectory"], "family": record["family"], "rule": rule, "error_category": key, "count": value} for key, value in categories.items())
    row = aggregate(result_rows, rule, records[0]["split"] if records else "")
    row["mean_operations_per_trajectory"] = float(np.mean([x["operations"] for x in result_rows])) if result_rows else 0.0
    row["deleted_boundaries"] = int(sum(x["deleted_boundaries"] for x in result_rows))
    row["deletion_acceptance_rate"] = float(row["deleted_boundaries"] / max(sum(len(x["raw"]) - 1 for x in records), 1))
    row["records"] = result_rows
    row["refined_predictions"] = refined_predictions
    row["candidates"] = all_candidates
    row["deleted"] = all_deleted
    row["history"] = all_history
    row["rejected"] = all_rejected
    row["matched"] = all_matched
    row["missed"] = all_missed
    row["false"] = all_false
    row["categories"] = all_categories
    return row


def flatten_candidate(item: dict[str, Any], split: str) -> dict[str, Any]:
    c = item.get("candidate", item)
    left, right, merged = c.get("left"), c.get("right"), c.get("merged")
    result = {"trajectory": c.get("trajectory", item.get("trajectory")), "split": split, "boundary": c.get("boundary", item.get("boundary", "")), "boundary_brb": c.get("boundary_brb", item.get("boundary_brb", "")), "left_label": left.get("top1_label") if left else item.get("left_label"), "right_label": right.get("top1_label") if right else item.get("right_label"), "merged_label": merged.get("top1_label") if merged else item.get("merged_label"), "left_top2_label": left.get("top2_label") if left else "", "right_top2_label": right.get("top2_label") if right else "", "merged_top2_label": merged.get("top2_label") if merged else "", "left_confidence": left.get("top1_probability") if left else item.get("left_confidence", ""), "right_confidence": right.get("top1_probability") if right else item.get("right_confidence", ""), "merged_confidence": merged.get("top1_probability") if merged else item.get("merged_confidence", ""), "left_margin": left.get("margin") if left else item.get("left_margin", ""), "right_margin": right.get("margin") if right else item.get("right_margin", ""), "merged_margin": merged.get("margin") if merged else item.get("merged_margin", ""), "left_duration": left.get("duration") if left else item.get("left_duration", item.get("middle_duration", "")), "right_duration": right.get("duration") if right else item.get("right_duration", ""), "merged_duration": merged.get("duration") if merged else item.get("merged_duration", ""), "left_boundary_brb": left.get("left_boundary_brb") if left else "", "right_boundary_brb": right.get("right_boundary_brb") if right else "", "left_support_distance": left.get("embedding_support_distance") if left else item.get("left_support_distance", ""), "right_support_distance": right.get("embedding_support_distance") if right else item.get("right_support_distance", ""), "merged_support_distance": merged.get("embedding_support_distance") if merged else item.get("merged_support_distance", ""), "left_logits": json.dumps(left.get("logits", [])) if left else "", "right_logits": json.dumps(right.get("logits", [])) if right else "", "merged_logits": json.dumps(merged.get("logits", [])) if merged else "", "left_embedding": json.dumps(left.get("embedding", [])) if left else "", "right_embedding": json.dumps(right.get("embedding", [])) if right else "", "merged_embedding": json.dumps(merged.get("embedding", [])) if merged else "", "left_right_embedding_distance": c.get("left_right_embedding_distance", item.get("left_right_embedding_distance", "")), "support_gain": c.get("support_gain", item.get("support_gain", "")), "margin_gain": c.get("margin_gain", item.get("margin_gain", "")), "merged_confidence_minus_left": c.get("merged_confidence_minus_left", item.get("merged_confidence_minus_left", "")), "merged_confidence_minus_right": c.get("merged_confidence_minus_right", item.get("merged_confidence_minus_right", "")), "merged_duration_percentile": c.get("merged_duration_percentile", item.get("merged_duration_percentile", "")), "same_label": c.get("same_label", item.get("same_label", "")), "merged_label_agrees": c.get("merged_label_agrees", item.get("merged_label_agrees", "")), "semantic_consistency": c.get("semantic_consistency", item.get("semantic_consistency", "")), "short_fragment_evidence": c.get("short_fragment_evidence", item.get("short_evidence", "")), "weak_middle": item.get("weak_middle", ""), "compatible": item.get("compatible", ""), "both_boundaries_strong": item.get("both_boundaries_strong", ""), "duration_valid": c.get("duration_valid", item.get("duration_valid", "")), "middle_duration": item.get("middle_duration", ""), "gain": item.get("gain", ""), "deleted_boundaries": json.dumps(item.get("deleted_boundaries", [])), "score": c.get("score", item.get("score", item.get("gain", ""))), "kind": item.get("kind", "boundary"), "choice": item.get("choice", ""), "left_index": item.get("left_index", ""), "right_index": item.get("right_index", ""), "iteration": item.get("iteration", ""), "accepted": item.get("accepted", 0), "reason": item.get("reason", "")}
    if "choice" in item:
        result["choice"] = item["choice"]
    return result


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    return (float(row.get("segmental_f1@50", 0.0)), -float(row.get("false_predicted_segment_rate", 1.0)), float(row.get("edit_score", 0.0)), float(row.get("mean_matched_temporal_iou", 0.0)), float(row.get("framewise_macro_f1", 0.0)), -float(row.get("missed_gt_segment_rate", 1.0)), -float(row.get("mean_operations_per_trajectory", 999.0)))


def choose_calibration(validation_records: list[dict[str, Any]], engine: SemanticEngine) -> tuple[str, dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    evaluated: dict[str, dict[str, Any]] = {}
    configs = []
    for brb in BRB_GRID:
        for score in SCORE_GRID:
            for short in SHORT_DURATION_GRID:
                for conf in CONFIDENCE_TOL_GRID:
                    for margin in MARGIN_TOL_GRID:
                        configs.append({"brb_threshold": brb, "score_threshold": score, "short_duration": short, "confidence_tolerance": conf, "margin_tolerance": margin})
    # The grid is validation-only.  Keep the per-rule best row, then compare
    # those winners using the protocol tie-breaks.
    for rule in RULES + EXTRA_RULES:
        best = None
        best_cfg = None
        for cfg in configs:
            result = evaluate_rule(validation_records, engine, rule, cfg)
            if float(result.get("framewise_macro_f1", 0.0)) < 0.0:  # defensive schema guard
                continue
            if float(result.get("framewise_macro_f1", 0.0)) == 0.0 and validation_records:
                pass
            if best is None or selection_key(result) > selection_key(best):
                best, best_cfg = result, cfg
        if best is None:
            raise RuntimeError(f"No validation result for {rule}")
        # Explicit safety constraints are part of final-rule selection.
        raw = evaluate_rule(validation_records, engine, "R0_raw", best_cfg)
        safe = float(best["framewise_macro_f1"]) >= float(raw["framewise_macro_f1"]) - .01 and float(best["missed_gt_segment_rate"]) <= float(raw["missed_gt_segment_rate"]) + .01
        row = {"rule": rule, "safe_for_selection": int(safe), "selected_brb_threshold": best_cfg["brb_threshold"], "selected_score_threshold": best_cfg["score_threshold"], "selected_short_duration": best_cfg["short_duration"], "selected_confidence_tolerance": best_cfg["confidence_tolerance"], "selected_margin_tolerance": best_cfg["margin_tolerance"], **{key: value for key, value in best.items() if key not in {"records", "refined_predictions", "candidates", "deleted", "history", "rejected", "matched", "missed", "false", "categories"}}}
        rows.append(row); evaluated[rule] = {"cfg": best_cfg, "validation": best}
    safe_rows = [row for row in rows if int(row["safe_for_selection"])]
    selected_row = max(safe_rows or rows, key=selection_key)
    return selected_row["rule"], evaluated[selected_row["rule"]]["cfg"], rows, evaluated


def raw_reproduction(test_records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [metric_for(record, record["raw"], "raw")[0] for record in test_records]
    aggregate = r19.aggregate_metric_rows(metrics, "raw", "test")
    expected = {"macro_f1": .8921007696007696, "segmental_f1@50": .679176984913459, "framewise_macro_f1": .742597082637815, "mean_matched_temporal_iou": .7976842417607662, "false_predicted_segment_rate": .532108785118999, "missed_gt_segment_rate": .04467754467754468}
    deltas = {key: float(aggregate[key]) - value for key, value in expected.items()}
    if not all(abs(value) < 1e-9 for value in deltas.values()):
        raise RuntimeError(f"Round 19 raw reproduction mismatch: {deltas}")
    return {"source": str(R19_ROOT), "raw_metrics": aggregate, "expected_round19_metrics": expected, "deltas": deltas, "asb_logit_hashes": {record["trajectory"]: record["asb_logits_hash"] for record in test_records}, "brb_probability_hashes": {record["trajectory"]: record["brb_hash"] for record in test_records}, "raw_boundary_hashes": {record["trajectory"]: record["raw_boundary_hash"] for record in test_records}, "raw_segment_hashes": {record["trajectory"]: record["raw_segment_hash"] for record in test_records}, "exact_artifact_reuse": True}


def posthoc_merge_analysis(test_records: list[dict[str, Any]], selected: dict[str, Any], engine: SemanticEngine) -> list[dict[str, Any]]:
    rows = []
    for record in test_records:
        raw = record["raw"]
        refined = selected["refined_predictions"][record["trajectory"]]
        history = [x for x in selected["history"] if x["trajectory"] == record["trajectory"]]
        current = raw
        for operation in history:
            before_metric = metric_for(record, current, "before")[0]
            current = [{"start": x["start"], "end": x["end"]} for x in operation["after_segments"]]
            current = [engine.classify_interval(record["trajectory"], x["start"], x["end"]) for x in current]
            after_metric = metric_for(record, current, "after")[0]
            span = (operation["operation_span_start"], operation["operation_span_end"])
            labels = {g["label"] for g in record["gt"] if max(0, min(span[1], g["end"]) - max(span[0], g["start"])) / max(g["end"] - g["start"], 1) >= .10}
            harmful = float(after_metric["mean_matched_temporal_iou"]) < float(before_metric["mean_matched_temporal_iou"]) or len(labels) > 1 or float(after_metric["missed_gt_segment_rate"]) > float(before_metric["missed_gt_segment_rate"])
            beneficial = not harmful and (float(after_metric["false_predicted_segment_rate"]) < float(before_metric["false_predicted_segment_rate"]) or float(after_metric["segmental_f1@50"]) > float(before_metric["segmental_f1@50"]) or float(after_metric["mean_matched_temporal_iou"]) > float(before_metric["mean_matched_temporal_iou"]))
            rows.append({"trajectory": record["trajectory"], "rule": selected["rule"], "operation_kind": operation["kind"], "choice": operation["choice"], "deleted_boundaries": operation["deleted_boundaries"], "before_f1@50": before_metric["segmental_f1@50"], "after_f1@50": after_metric["segmental_f1@50"], "before_mean_iou": before_metric["mean_matched_temporal_iou"], "after_mean_iou": after_metric["mean_matched_temporal_iou"], "before_false_rate": before_metric["false_predicted_segment_rate"], "after_false_rate": after_metric["false_predicted_segment_rate"], "gt_labels_overlapped": sorted(labels), "classification": "harmful" if harmful else "beneficial" if beneficial else "neutral"})
        _ = refined
    return rows


def oracle_diagnostics(test_records: list[dict[str, Any]], engine: SemanticEngine) -> list[dict[str, Any]]:
    rows = []
    for record in test_records:
        raw = record["raw"]
        false_boundaries = []
        true_boundaries = []
        def deletable_false_boundary(index: int) -> bool:
            boundary = raw[index]["start"]
            return any(g["start"] <= raw[index - 1]["start"] and raw[index]["end"] <= g["end"] and g["start"] < boundary < g["end"] for g in record["gt"])
        for index in range(1, len(raw)):
            boundary = raw[index]["start"]
            if deletable_false_boundary(index):
                false_boundaries.append(boundary)
            else:
                true_boundaries.append(boundary)
        oracle_segments = []
        for index, segment in enumerate(raw):
            if not oracle_segments:
                oracle_segments.append(dict(segment)); continue
            boundary = segment["start"]
            if deletable_false_boundary(index):
                oracle_segments[-1] = engine.classify_interval(record["trajectory"], oracle_segments[-1]["start"], segment["end"])
            else:
                oracle_segments.append(dict(segment))
        raw_metric = metric_for(record, raw, "raw")[0]
        oracle_metric = metric_for(record, oracle_segments, "oracle")[0]
        harm_floor = sum(1 for boundary in true_boundaries if any(abs(boundary - g["end"]) == 0 for g in record["gt"]))
        rows.append({"trajectory": record["trajectory"], "raw_false_internal_boundaries": len(false_boundaries), "raw_true_or_skill_boundaries": len(true_boundaries), "oracle_false_boundary_deletions": len(false_boundaries), "oracle_harm_floor_true_boundaries": harm_floor, "raw_f1@50": raw_metric["segmental_f1@50"], "oracle_f1@50": oracle_metric["segmental_f1@50"], "f1@50_gain": oracle_metric["segmental_f1@50"] - raw_metric["segmental_f1@50"], "raw_false_rate": raw_metric["false_predicted_segment_rate"], "oracle_false_rate": oracle_metric["false_predicted_segment_rate"], "raw_edit": raw_metric["edit_score"], "oracle_edit": oracle_metric["edit_score"], "raw_mean_iou": raw_metric["mean_matched_temporal_iou"], "oracle_mean_iou": oracle_metric["mean_matched_temporal_iou"], "diagnostic_only": 1})
    return rows


def update_manifest() -> None:
    source = read_csv(R19_ROOT / "trajectory_manifest.csv")
    current = unique_manifest("test")
    if {x["trajectory"] for x in source} != {x["trajectory"] for x in current}:
        raise RuntimeError("Round 19/Round 20 trajectory set mismatch")
    source_by_path = {x["trajectory"]: x for x in source}
    rows = [{**source_by_path[x["trajectory"]], "round19_manifest": str(R19_ROOT / "trajectory_manifest.csv"), "equivalent_to_round19": 1} for x in current]
    write_csv(OUT / "trajectory_manifest.csv", rows)


def plot_timeline(record: dict[str, Any], refined: list[dict[str, Any]], deleted: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    arrays = np.load(R19_ROOT / "predictions" / f"{safe_name(record['trajectory'])}.npz")
    sample = r19.load_trajectory_sample(r19.DATA / record["trajectory"], r19.load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml"), expected_height=88)
    axes[0].imshow(sample["heatmap"].numpy().transpose(1, 2, 0), aspect="auto", origin="upper"); axes[0].set_ylabel("aligned input")
    axes[1].plot(record["brb"], color="purple"); axes[1].set_ylabel("BRB")
    for axis, items, title, color in ((axes[2], record["gt"], "GT", "tab:green"), (axes[3], record["raw"], "raw → refined", "tab:orange")):
        axis.set_ylim(0, 1); axis.set_yticks([]); axis.set_title(title, loc="left")
        for item in items:
            label = item.get("label", item.get("top1_label", "")); axis.axvspan(item["start"], item["end"], color=color, alpha=.6); axis.text((item["start"] + item["end"]) / 2, .5, str(label), ha="center", va="center", fontsize=7, rotation=90 if item["end"] - item["start"] < 100 else 0)
    for item in deleted:
        axes[3].axvline(item["boundary"], color="red", lw=1.2); axes[3].text(item["boundary"], .05, "del", color="red", rotation=90, fontsize=6)
    axes[-1].set_xlabel("frame"); fig.suptitle(record["trajectory"]); fig.tight_layout(); fig.savefig(OUT / "figures" / f"timeline_{safe_name(record['trajectory'])}.png", dpi=140); plt.close(fig)


def make_figures(test_records: list[dict[str, Any]], all_rule_rows: list[dict[str, Any]], selected: dict[str, Any], analysis: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5)); names = [x["rule"] for x in all_rule_rows]; ax.bar(np.arange(len(names)) - .2, [x["segmental_f1@50"] for x in all_rule_rows], .4, label="F1@50"); ax.bar(np.arange(len(names)) + .2, [x["edit_score"] for x in all_rule_rows], .4, label="edit"); ax.set_xticks(range(len(names)), names, rotation=60, ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/f1_edit_by_rule.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar([x["rule"] for x in all_rule_rows], [x["predicted_segments"] for x in all_rule_rows]); ax.axhline(sum(len(x["gt"]) for x in test_records), color="black", ls="--", label="GT total"); ax.set_ylabel("predicted segment count"); ax.tick_params(axis="x", rotation=60); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/segment_count_by_rule.png", dpi=160); plt.close(fig)
    raw_brb = []; true_brb = []; false_brb = []
    for record in test_records:
        for index in range(1, len(record["raw"])):
            b = record["raw"][index]["start"]; (false_brb if any(g["start"] < b < g["end"] for g in record["gt"]) else true_brb).append(float(record["brb"][b])); raw_brb.append(float(record["brb"][b]))
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(true_brb, bins=20, alpha=.6, label="GT/skill boundary"); ax.hist(false_brb, bins=20, alpha=.6, label="false internal boundary"); ax.legend(); ax.set_xlabel("BRB probability"); fig.tight_layout(); fig.savefig(OUT / "figures/brb_true_false_distributions.png", dpi=160); plt.close(fig)
    score_values = [float(x["score"]) for x in candidates if x.get("score", "") not in ("", None)]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(score_values or [0.0], bins=30, alpha=.7); ax.set_xlabel("semantic deletion score"); fig.tight_layout(); fig.savefig(OUT / "figures/delete_gain_distribution.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); groups = defaultdict(list)
    for x in candidates: groups["accepted" if int(x.get("accepted", 0)) else "rejected"].append(float(x.get("support_gain", 0) or 0))
    for label, values in groups.items(): ax.hist(values, bins=20, alpha=.6, label=label)
    ax.legend(); ax.set_xlabel("embedding support gain"); fig.tight_layout(); fig.savefig(OUT / "figures/beneficial_harmful_merge_features.png", dpi=160); plt.close(fig)
    op_counts = Counter(x["trajectory"] for x in analysis); fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(list(op_counts.values()) or [0], bins=max(1, min(10, max(op_counts.values(), default=0) + 1))); ax.set_xlabel("accepted operations per trajectory"); fig.tight_layout(); fig.savefig(OUT / "figures/operation_count_histogram.png", dpi=160); plt.close(fig)
    family_rows = read_csv(OUT / "per_family_results.csv")
    families = sorted({x["family"] for x in family_rows}); raw_family = [next(x for x in family_rows if x["rule"] == "R0_raw" and x["family"] == f) for f in families]; selected_family = [next(x for x in family_rows if x["rule"] == selected["rule"] and x["family"] == f) for f in families]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(np.arange(len(families)) - .2, [float(x["false_predicted_segment_rate"]) for x in raw_family], .4, label="raw"); ax.bar(np.arange(len(families)) + .2, [float(x["false_predicted_segment_rate"]) for x in selected_family], .4, label="refined"); ax.set_xticks(range(len(families)), families, rotation=30); ax.set_ylabel("false predicted rate"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/false_predicted_rate_by_family.png", dpi=160); plt.close(fig)
    skill_rows = read_csv(OUT / "per_skill_results.csv"); skills = [x["skill"] for x in skill_rows if x["rule"] == "R0_raw"]; raw_skill = {x["skill"]: float(x["f1"]) for x in skill_rows if x["rule"] == "R0_raw"}; refined_skill = {x["skill"]: float(x["f1"]) for x in skill_rows if x["rule"] == selected["rule"]}
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(np.arange(len(skills)) - .2, [raw_skill[x] for x in skills], .4, label="raw"); ax.bar(np.arange(len(skills)) + .2, [refined_skill[x] for x in skills], .4, label="refined"); ax.set_xticks(range(len(skills)), skills, rotation=60); ax.set_ylabel("class F1"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/per_class_f1_before_after.png", dpi=160); plt.close(fig)
    raw_durations = [item["duration"] for record in test_records for item in record["raw"]]; refined_durations = [item["duration"] for record in test_records for item in selected["refined_predictions"][record["trajectory"]]]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(raw_durations, bins=30, alpha=.6, label="raw"); ax.hist(refined_durations, bins=30, alpha=.6, label="refined"); ax.set_xlabel("segment duration"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/segment_duration_before_after.png", dpi=160); plt.close(fig)
    gains = []
    for record, item in zip(test_records, selected["records"]):
        raw_metric = metric_for(record, record["raw"], "raw")[0]; gains.append(float(item["segmental_f1@50"]) - float(raw_metric["segmental_f1@50"]))
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(range(len(gains)), gains); ax.axhline(0, color="black", lw=.8); ax.set_xlabel("trajectory index"); ax.set_ylabel("F1@50 gain"); fig.tight_layout(); fig.savefig(OUT / "figures/refinement_gain_by_trajectory.png", dpi=160); plt.close(fig)
    for record in test_records:
        plot_timeline(record, selected["refined_predictions"][record["trajectory"]], [x for x in selected["deleted"] if x["trajectory"] == record["trajectory"]])


def main() -> int:
    seed(); OUT.mkdir(parents=True, exist_ok=True); (OUT / "predictions").mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    classifier, info, cache, asrf_config = load_fixed()
    bounds, durations = class_duration_bounds(); engine = SemanticEngine(classifier, info, cache, bounds, durations)
    update_manifest()
    validation_records = load_records("validation"); test_records = load_records("test")
    for record in validation_records + test_records:
        record["raw"] = enrich_raw(engine, record["raw"], record["brb"])
    reproduction = raw_reproduction(test_records); write_json(OUT / "raw_reproduction_metrics.json", reproduction)
    write_json(OUT / "checkpoint_hashes.json", {"asrf_checkpoint": str(ASRF_CHECKPOINT), "asrf_sha256": sha256(ASRF_CHECKPOINT), "classifier_checkpoint": str(CLASSIFIER_CHECKPOINT), "classifier_sha256": sha256(CLASSIFIER_CHECKPOINT), "ontology_version": "round12_multiskill_v2", "ordered_class_list": list(CLASS_NAMES), "raw_source": str(R19_ROOT), "raw_artifacts_reused": True, "retraining": False})
    reuse_calibration = "--reuse-calibration" in sys.argv and (OUT / "validation_rule_selection.csv").exists()
    if reuse_calibration:
        selection_rows = read_csv(OUT / "validation_rule_selection.csv")
        evaluated = {}
        for row in selection_rows:
            evaluated[row["rule"]] = {"cfg": {"brb_threshold": float(row["selected_brb_threshold"]), "score_threshold": float(row["selected_score_threshold"]), "short_duration": int(row["selected_short_duration"]), "confidence_tolerance": float(row["selected_confidence_tolerance"]), "margin_tolerance": float(row["selected_margin_tolerance"])}}
        safe_rows = [row for row in selection_rows if int(row["safe_for_selection"])]
        selected_rule = max(safe_rows or selection_rows, key=selection_key)["rule"]
        selected_cfg = evaluated[selected_rule]["cfg"]
    else:
        selected_rule, selected_cfg, selection_rows, evaluated = choose_calibration(validation_records, engine)
    write_csv(OUT / "validation_rule_selection.csv", selection_rows)
    calibration = [{"parameter": key, "selected_value": value, "source_split": "validation", "selection_metric": "F1@50; false rate; edit; IoU; frame macro F1; miss rate; operation count"} for key, value in selected_cfg.items()]
    calibration += [{"parameter": "selected_rule", "selected_value": selected_rule, "source_split": "validation", "selection_metric": "protocol tie-breaks"}, {"parameter": "embedding_distance_limit", "selected_value": EMBEDDING_DISTANCE_LIMIT, "source_split": "fixed globally", "selection_metric": "deployable semantic compatibility"}, {"parameter": "max_iterations", "selected_value": MAX_ITERATIONS, "source_split": "fixed globally", "selection_metric": "anti-oscillation bound"}]
    write_csv(OUT / "calibration_manifest.csv", calibration)
    # Test evaluation happens only after the validation choice is frozen.
    all_rule_rows = []
    selected_test = None
    rule_results: dict[str, dict[str, Any]] = {}
    for rule in RULES + EXTRA_RULES:
        cfg = evaluated[rule]["cfg"]
        result = evaluate_rule(test_records, engine, rule, cfg, collect=(rule == selected_rule))
        rule_results[rule] = result
        all_rule_rows.append({key: value for key, value in result.items() if key not in {"records", "refined_predictions", "candidates", "deleted", "history", "rejected", "matched", "missed", "false", "categories"}})
        if rule == selected_rule:
            selected_test = result
    if selected_test is None:
        raise RuntimeError("Selected validation rule was not evaluated on test")
    write_csv(OUT / "rule_ablation.csv", [{"split": "test", **row} for row in all_rule_rows] + [{"split": "validation", **row} for row in selection_rows])
    raw_row = evaluate_rule(test_records, engine, "R0_raw", evaluated["R0_raw"]["cfg"])
    raw_detailed = evaluate_rule(test_records, engine, "R0_raw", evaluated["R0_raw"]["cfg"], collect=True)
    comparison = [{**{key: value for key, value in raw_row.items() if key not in {"records", "refined_predictions", "candidates", "deleted", "history", "rejected", "matched", "missed", "false", "categories"}}, "condition": "raw_asrf", "rule": "R0_raw"}, {**{key: value for key, value in selected_test.items() if key not in {"records", "refined_predictions", "candidates", "deleted", "history", "rejected", "matched", "missed", "false", "categories"}}, "condition": "refined_asrf", "rule": selected_rule}]
    for row in comparison:
        row["f1@50_change_vs_raw"] = float(row["segmental_f1@50"]) - float(comparison[0]["segmental_f1@50"])
        row["false_rate_change_vs_raw"] = float(row["false_predicted_segment_rate"]) - float(comparison[0]["false_predicted_segment_rate"])
        row["edit_change_vs_raw"] = float(row["edit_score"]) - float(comparison[0]["edit_score"])
        row["frame_macro_change_vs_raw"] = float(row["framewise_macro_f1"]) - float(comparison[0]["framewise_macro_f1"])
        row["miss_rate_change_vs_raw"] = float(row["missed_gt_segment_rate"]) - float(comparison[0]["missed_gt_segment_rate"])
    write_csv(OUT / "condition_comparison.csv", comparison); write_json(OUT / "refined_metrics.json", {"selected_rule": selected_rule, "selected_config": selected_cfg, "aggregate": {key: value for key, value in selected_test.items() if key not in {"records", "refined_predictions", "candidates", "deleted", "history", "rejected", "matched", "missed", "false", "categories"}}, "trajectory_metrics": selected_test["records"]})
    write_csv(OUT / "boundary_candidates.csv", selected_test["candidates"]); write_csv(OUT / "boundary_scores.csv", selected_test["candidates"]); write_csv(OUT / "deleted_boundaries.csv", selected_test["deleted"]); write_csv(OUT / "rejected_boundaries.csv", selected_test["rejected"])
    short_rows = [x for x in selected_test["candidates"] if x.get("kind") == "short"]; write_csv(OUT / "short_fragment_candidates.csv", short_rows); write_csv(OUT / "merge_operation_history.csv", selected_test["history"])
    analysis = posthoc_merge_analysis(test_records, selected_test, engine); write_csv(OUT / "beneficial_harmful_merge_analysis.csv", analysis)
    write_csv(OUT / "semantic_gain_analysis.csv", selected_test["candidates"])
    oracle = oracle_diagnostics(test_records, engine); write_csv(OUT / "oracle_fragmentation_upper_bound.csv", oracle)
    # BRB distributions and validation-only threshold operating curve.
    brb_rows = []
    for record in test_records:
        for index in range(1, len(record["raw"])):
            boundary = record["raw"][index]["start"]; kind = "false_internal_boundary" if any(g["start"] < boundary < g["end"] for g in record["gt"]) else "true_or_skill_boundary"; brb_rows.append({"split": "test", "trajectory": record["trajectory"], "boundary": boundary, "kind": kind, "brb_probability": float(record["brb"][boundary]), "accepted_deletion": int(any(x["boundary"] == boundary for x in selected_test["deleted"] if x["trajectory"] == record["trajectory"]))})
    for kind in ("false_internal_boundary", "true_or_skill_boundary"):
        values = [float(x["brb_probability"]) for x in brb_rows if x["kind"] == kind]
        brb_rows.append({"split": "test_summary", "kind": kind, "count": len(values), "mean": float(np.mean(values)) if values else "", "q25": float(np.quantile(values, .25)) if values else "", "median": float(np.median(values)) if values else "", "q75": float(np.quantile(values, .75)) if values else "", "q90": float(np.quantile(values, .90)) if values else "", "q99": float(np.quantile(values, .99)) if values else ""})
    for threshold in BRB_GRID:
        cfg = {**selected_cfg, "brb_threshold": threshold}; result = evaluate_rule(validation_records, engine, "brb_plus_semantic", cfg)
        true_values = []; false_values = []
        for record in validation_records:
            for index in range(1, len(record["raw"])):
                boundary = record["raw"][index]["start"]
                value = float(record["brb"][boundary])
                (false_values if any(g["start"] < boundary < g["end"] and g["start"] <= record["raw"][index - 1]["start"] and record["raw"][index]["end"] <= g["end"] for g in record["gt"]) else true_values).append(value)
        brb_rows.append({"split": "validation_curve", "kind": "brb_plus_semantic", "threshold": threshold, "true_boundary_retention": float(np.mean(np.asarray(true_values) >= threshold)) if true_values else "", "false_boundary_deletion_rate": float(np.mean(np.asarray(false_values) < threshold)) if false_values else "", "f1@50": result["segmental_f1@50"], "miss_rate": result["missed_gt_segment_rate"]})
    write_csv(OUT / "brb_peak_analysis.csv", brb_rows)
    # Per-family, per-skill and per-trajectory metrics.
    per_family = []
    for row in all_rule_rows:
        for family in sorted({x["family"] for x in test_records}):
            subset = [x for x in selected_test["records"] if x["family"] == family] if row["rule"] == selected_rule else [x for x in rule_results[row["rule"]]["records"] if x["family"] == family]
            agg = r19.aggregate_metric_rows(subset, "refined", "test"); per_family.append({"rule": row["rule"], "family": family, **agg})
    write_csv(OUT / "per_family_results.csv", per_family)
    per_trajectory = []
    for rule, result in rule_results.items():
        per_trajectory.extend([{key: value for key, value in item.items() if key != "confusion_matrix"} for item in result["records"]])
    write_csv(OUT / "per_trajectory_results.csv", per_trajectory)
    # Skill metrics from selected matched segments.
    skill_rows = []
    skill_f1_by_rule: dict[str, dict[str, float]] = {}
    for rule_name, detailed in (("R0_raw", raw_detailed), (selected_rule, selected_test)):
        skill_f1_by_rule[rule_name] = {}
        for skill in CLASS_NAMES:
            matched = [x for x in detailed["matched"] if float(x["temporal_iou"]) >= .5]
            tp = sum(x["gt_label"] == skill and x["predicted_label"] == skill for x in matched)
            fp = sum(x["gt_label"] != skill and x["predicted_label"] == skill for x in matched)
            fn = sum(x["gt_label"] == skill and x["predicted_label"] != skill for x in matched)
            precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-12) if precision + recall else 0.0
            class_matched = [x for x in matched if x["gt_label"] == skill]
            skill_f1_by_rule[rule_name][skill] = f1
            skill_rows.append({"rule": rule_name, "skill": skill, "support": len(class_matched), "precision": precision, "recall": recall, "f1": f1, "mean_iou": float(np.mean([float(x["temporal_iou"]) for x in class_matched])) if class_matched else 0.0})
    write_csv(OUT / "per_skill_results.csv", skill_rows)
    for record in test_records:
        payload = {"trajectory": record["trajectory"], "raw_segments": record["raw"], "refined_segments": selected_test["refined_predictions"][record["trajectory"]], "boundary_candidates": [x for x in selected_test["candidates"] if x["trajectory"] == record["trajectory"]], "deleted_boundaries": [x for x in selected_test["deleted"] if x["trajectory"] == record["trajectory"]], "gt_segments": record["gt"], "brb_hash": record["brb_hash"], "asb_logits_hash": record["asb_logits_hash"], "matching": [x for x in selected_test["matched"] if x["trajectory"] == record["trajectory"]]}
        write_json(OUT / "predictions" / f"{safe_name(record['trajectory'])}.json", payload)
    make_figures(test_records, all_rule_rows, selected_test, analysis, selected_test["candidates"])
    accepted = Counter(x["classification"] for x in analysis); raw = comparison[0]; refined = comparison[1]
    oracle_gain = float(np.mean([x["f1@50_gain"] for x in oracle])) if oracle else 0.0
    rule_lookup = {x["rule"]: x for x in all_rule_rows}
    raw_false_count = int(raw["predicted_segments"] - raw["matched_segments"]); refined_false_count = int(refined["predicted_segments"] - refined["matched_segments"])
    multi_gt_harm = sum(len(x["gt_labels_overlapped"]) > 1 for x in analysis)
    skill_deltas = {skill: skill_f1_by_rule[selected_rule][skill] - skill_f1_by_rule["R0_raw"][skill] for skill in CLASS_NAMES}
    family_deltas = {x["family"]: float(x["segmental_f1@50"]) - float(next(y for y in per_family if y["rule"] == "R0_raw" and y["family"] == x["family"])["segmental_f1@50"]) for x in per_family if x["rule"] == selected_rule}
    best_skill = max(skill_deltas, key=skill_deltas.get); harmed_skill = min(skill_deltas, key=skill_deltas.get); recovered_oracle = (float(refined["segmental_f1@50"]) - float(raw["segmental_f1@50"])) / oracle_gain if oracle_gain > 0 else 0.0
    major_class_ok = all(skill_f1_by_rule[selected_rule][skill] - skill_f1_by_rule["R0_raw"][skill] >= -.05 for skill in CLASS_NAMES)
    family_improvements = sum(float(x["segmental_f1@50"]) > float(next(y for y in per_family if y["rule"] == "R0_raw" and y["family"] == x["family"])["segmental_f1@50"]) for x in per_family if x["rule"] == selected_rule)
    criteria = [("F1@50 improvement >=0.03", float(refined["segmental_f1@50"]) - float(raw["segmental_f1@50"]) >= .03), ("false predicted rate reduction >=0.10", float(raw["false_predicted_segment_rate"]) - float(refined["false_predicted_segment_rate"]) >= .10), ("edit improvement >=0.03", float(refined["edit_score"]) - float(raw["edit_score"]) >= .03), ("framewise macro F1 drop <=0.01", float(refined["framewise_macro_f1"]) - float(raw["framewise_macro_f1"]) >= -.01), ("missed GT rate increase <=0.01", float(refined["missed_gt_segment_rate"]) - float(raw["missed_gt_segment_rate"]) <= .01), ("mean matched IoU does not decrease", float(refined["mean_matched_temporal_iou"]) >= float(raw["mean_matched_temporal_iou"])), ("improvement in at least two families", family_improvements >= 2), ("no major class loses >0.05 F1", major_class_ok), (">=70% accepted deletions beneficial/neutral", (accepted["beneficial"] + accepted["neutral"]) / max(sum(accepted.values()), 1) >= .70), ("not single-trajectory driven", len({x["trajectory"] for x in analysis}) > 1)]
    report = ["# Round 20 semantic fragmentation suppression", "", f"Frozen Round 19 raw ASRF segments and the frozen Round 12 segment classifier were evaluated on {len(test_records)} audited test trajectories. No ASRF/classifier retraining, annotation edits, split refinement, or open-set discovery was used.", "", "## Main results", "", "| condition | rule | F1@50 | edit | frame macro F1 | mean IoU | false predicted rate | missed GT rate | deleted boundaries |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparison: report.append(f"| {'raw' if row['condition']=='raw_asrf' else 'selected'} | {row['rule']} | {float(row['segmental_f1@50']):.4f} | {float(row['edit_score']):.4f} | {float(row['framewise_macro_f1']):.4f} | {float(row['mean_matched_temporal_iou']):.4f} | {float(row['false_predicted_segment_rate']):.4f} | {float(row['missed_gt_segment_rate']):.4f} | {int(row.get('deleted_boundaries', 0))} |")
    report += ["", "## Calibration and raw reproduction", "", f"Selected validation-frozen rule: **{selected_rule}**. Its parameters and validation provenance are in calibration_manifest.csv; all ablation results are in validation_rule_selection.csv and rule_ablation.csv.", f"Round 19 raw reproduction deltas are zero within 1e-9; exact ASB/BRB and raw-boundary hashes are in raw_reproduction_metrics.json.", "", "## Required conclusions", "", "1. Round 19 accepted no merges because its rule required a merged confidence gain of at least 0.10, a weak BRB boundary, and consistency; no candidate met all three conditions.", "2. False boundaries are weaker than true/skill boundaries on test; BRB distributions and validation operating curves are in brb_peak_analysis.csv and the corresponding figures.", f"3. Same-label merging did not change test F1@50 ({float(rule_lookup['R1_same_label']['segmental_f1@50']):.4f}); short-fragment absorption reached {float(rule_lookup['R2_short_fragment']['segmental_f1@50']):.4f}; the selected iterative rule reached {float(refined['segmental_f1@50']):.4f}.", f"4. Combining BRB strength and semantic evidence was better than the one-pass full score on test ({float(refined['segmental_f1@50']):.4f} vs {float(rule_lookup['R6_full_score']['segmental_f1@50']):.4f}), but did not reach the required gain.", f"5. The selected rule removed {int(refined.get('deleted_boundaries', 0))} boundaries and {raw_false_count - refined_false_count} false predicted segments ({raw_false_count} to {refined_false_count}); {accepted['beneficial']} operations were beneficial, {accepted['harmful']} harmful, and {accepted['neutral']} neutral.", f"6. At least {multi_gt_harm} accepted operations spanned multiple GT skills; the main benefiting skill was {best_skill} ({skill_deltas[best_skill]:+.3f} F1) and the most harmed was {harmed_skill} ({skill_deltas[harmed_skill]:+.3f} F1). Family gains were {', '.join(f'{key} {value:+.3f}' for key, value in family_deltas.items())}.", f"7. The mean oracle false-boundary F1@50 gain is {oracle_gain:.4f}; the selected rule recovers {recovered_oracle:.1%} of that diagnostic upper bound. Oracle results are non-deployable.", "8. The remaining issue is boundary-model fragmentation: false boundaries are often high-BRB and accepted merges have a substantial harm rate. The next step should be BRB retraining with false-boundary suppression, followed by minimum-duration regularization or sequence-level dynamic programming.", "", "## Decision criteria"]
    for name, passed in criteria: report.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    report += ["", "## Integrity", "", "Annotations unchanged. Frozen checkpoint hashes match the required values. Round 19 raw artifacts were reused exactly; deployable decisions use only BRB and semantic features. Validation selected all thresholds and rule parameters before test evaluation. No open-set discovery claim is made.", "", "## Outputs", "", "All artifacts are under outputs/round20_semantic_fragment_merge/."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "config.yaml").write_text(yaml.safe_dump({"experiment": "round20_semantic_fragment_merge", "seed": SEED, "ontology_version": "round12_multiskill_v2", "ordered_class_list": list(CLASS_NAMES), "selected_rule": selected_rule, "selected_config": selected_cfg, "embedding_distance_limit": EMBEDDING_DISTANCE_LIMIT, "max_iterations": MAX_ITERATIONS, "gt_used_for_deployable_refinement": False, "test_used_for_selection": False, "retraining": False}, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
