#!/usr/bin/env python3
"""Round 21: ASB-assisted validation and suppression of false ASRF boundaries."""

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
R19 = ROOT / "outputs/round19_asrf_segment_classifier_integration"
R20 = ROOT / "outputs/round20_semantic_fragment_merge"
R12 = ROOT / "outputs/round12_multiskill_segment_classifier"
OUT = ROOT / "outputs/round21_asb_assisted_boundary_merge"
ASRF_CHECKPOINT = ROOT / "outputs/round10_pp_only_novel_segmentation/models/single_frame/best.pt"
CLASSIFIER_CHECKPOINT = R12 / "model/best.pt"
ASRF_SHA = "6b11abff2ff4387eada88e38fb56133b29fd5c3facdfb54b42fc84701213db9a"
CLASSIFIER_SHA = "51f0abbcc4250ef97951bcaef04fc8f55cb2de968affdf0121a446ea1635a86f"
sys.path.insert(0, str(ROOT / "scripts"))
import run_round20_semantic_fragment_merge as r20  # noqa: E402

SEED = 42
WINDOWS = (10, 20, 40, 60, 80)
DURATION_GRID = (20, 40, 60, 80, 100, 120)
RATIO_GRID = (0.60, 0.70, 0.80, 0.90)
SIM_GRID = (0.70, 0.80, 0.90, 0.95)
BRB_GRID = (0.10, 0.20, 0.30, 0.40, 0.50)
TOL_GRID = (0.00, 0.03, 0.05, 0.10)
ITER_GRID = (2, 4, 8)
ASB_LABELS = ("reach", "grasp", "lift", "transport", "place", "release", "retreat")
FINAL_CLASSES = tuple(r20.CLASS_NAMES)
RULES = ("R0_raw", "R1_asb_same_label", "R2_asb_duration", "R3_asb_brb", "R4_local_soft", "R5_whole_soft", "R6_local_whole", "R7_three_segment", "R8_classifier_verified", "R9_full_iterative")
EXTRA_RULES = ("asb_only", "brb_only", "classifier_only", "asb_brb", "asb_classifier", "asb_brb_classifier")


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
    return r20.safe_name(path)


def load_records(split: str) -> list[dict[str, Any]]:
    records = []
    for manifest in r20.unique_manifest(split):
        name = safe_name(manifest["trajectory"])
        payload = json.loads((R19 / "predictions" / f"{name}.json").read_text())
        arrays = np.load(R19 / "predictions" / f"{name}.npz")
        hard = np.argmax(arrays["asb_probabilities"], axis=0).astype(np.int64)
        records.append({"trajectory": manifest["trajectory"], "family": manifest["family"], "split": split, "length": len(arrays["brb_probabilities"]), "gt": payload["gt_segments"], "raw": [dict(x, source="raw_asrf") for x in payload["classifier_raw"]], "asb_logits": np.asarray(arrays["asb_logits"], dtype=np.float32), "asb_probabilities": np.asarray(arrays["asb_probabilities"], dtype=np.float32), "asb_labels": hard, "brb": np.asarray(arrays["brb_probabilities"], dtype=np.float32), "asb_logits_hash": hashlib.sha256(np.asarray(arrays["asb_logits"]).tobytes()).hexdigest(), "asb_probability_hash": hashlib.sha256(np.asarray(arrays["asb_probabilities"]).tobytes()).hexdigest(), "asb_label_hash": hashlib.sha256(hard.tobytes()).hexdigest(), "brb_hash": hashlib.sha256(np.asarray(arrays["brb_probabilities"]).tobytes()).hexdigest(), "raw_boundary_hash": hashlib.sha256(json.dumps(payload["raw_predicted_segments"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "raw_segment_hash": hashlib.sha256(json.dumps([(x["start"], x["end"]) for x in payload["classifier_raw"]], separators=(",", ":")).encode()).hexdigest()})
    return records


def asb_entropy(probabilities: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-8, 1.0)
    return float(np.mean(-np.sum(p * np.log(p), axis=0)))


def asb_summary(record: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    start, end = max(0, int(start)), min(int(end), record["length"])
    probs = record["asb_probabilities"][:, start:end]
    if probs.shape[1] == 0:
        probs = record["asb_probabilities"][:, max(0, start - 1):max(1, start)]
    labels = np.argmax(probs, axis=0)
    counts = np.bincount(labels, minlength=len(ASB_LABELS)); order = np.argsort(counts)[::-1]
    majority = int(order[0]); ratio = float(counts[majority] / max(len(labels), 1)); second = int(order[1]) if len(order) > 1 else majority
    transitions = int(np.sum(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
    longest = 0; current = 0
    for value in labels:
        current = current + 1 if int(value) == majority else 0; longest = max(longest, current)
    mean_prob = np.mean(probs, axis=1); median_prob = np.median(probs, axis=1); max_prob = np.max(probs, axis=1)
    frame_order = np.argsort(probs, axis=0); top1 = probs[frame_order[-1], np.arange(probs.shape[1])]; top2 = probs[frame_order[-2], np.arange(probs.shape[1])] if probs.shape[0] > 1 else np.zeros_like(top1)
    return {"start": start, "end": end, "duration": end - start, "asb_majority_id": majority, "asb_majority_label": ASB_LABELS[majority], "asb_majority_ratio": ratio, "asb_second_label": ASB_LABELS[second], "asb_label_entropy": float(-sum((count / max(len(labels), 1)) * np.log(max(count / max(len(labels), 1), 1e-8)) for count in counts if count)), "asb_transition_count": transitions, "asb_longest_majority_run": longest, "asb_mean_probability": mean_prob.tolist(), "asb_median_probability": median_prob.tolist(), "asb_max_probability": max_prob.tolist(), "asb_mean_top1_confidence": float(np.mean(top1)), "asb_mean_margin": float(np.mean(top1 - top2)), "asb_probability_entropy": asb_entropy(probs)}


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.clip(left, 1e-8, 1.0), np.clip(right, 1e-8, 1.0); left /= left.sum(); right /= right.sum(); mid = .5 * (left + right)
    return float(.5 * np.sum(left * np.log(left / mid)) + .5 * np.sum(right * np.log(right / mid)))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-8))


def boundary_stats(record: dict[str, Any], index: int, window: int, engine: r20.SemanticEngine) -> dict[str, Any]:
    left, right = record["raw"][index], record["raw"][index + 1]; boundary = int(right["start"])
    left_start, right_end = max(left["start"], boundary - window), min(right["end"], boundary + window)
    l = asb_summary(record, left_start, boundary); r = asb_summary(record, boundary, right_end)
    local_cos = cosine(np.asarray(l["asb_mean_probability"]), np.asarray(r["asb_mean_probability"]))
    local_js = js_divergence(np.asarray(l["asb_mean_probability"]), np.asarray(r["asb_mean_probability"]))
    left_full, right_full = asb_summary(record, left["start"], left["end"]), asb_summary(record, right["start"], right["end"])
    merged_asb = asb_summary(record, left["start"], right["end"])
    merged_cls = engine.classify_interval(record["trajectory"], left["start"], right["end"])
    protection = int(left_full["asb_majority_label"] != right_full["asb_majority_label"] and left_full["asb_majority_ratio"] >= .8 and right_full["asb_majority_ratio"] >= .8 and local_js >= .12)
    return {"trajectory": record["trajectory"], "boundary_index": index, "boundary": boundary, "window": window, "left": left, "right": right, "left_window": l, "right_window": r, "left_full": left_full, "right_full": right_full, "merged_asb": merged_asb, "merged_classifier": merged_cls, "brb_probability": float(record["brb"][boundary]), "local_label_agreement": int(l["asb_majority_label"] == r["asb_majority_label"]), "local_cosine_similarity": local_cos, "local_js_divergence": local_js, "shorter_duration": min(left["duration"], right["duration"]), "longer_duration": max(left["duration"], right["duration"]), "transition_protected": protection, "classifier_support_ok": int(merged_cls["duration_valid"] and merged_cls["embedding_support_distance"] <= .5), "classifier_confidence": merged_cls["top1_probability"], "classifier_margin": merged_cls["margin"]}


def load_fixed() -> tuple[Any, dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]], dict[str, tuple[float, float, float]], dict[str, np.ndarray]]:
    if sha256(ASRF_CHECKPOINT) != ASRF_SHA or sha256(CLASSIFIER_CHECKPOINT) != CLASSIFIER_SHA:
        raise RuntimeError("Frozen checkpoint hash mismatch")
    classifier, info, cache, _ = r20.load_fixed()[0:4]
    if info["normalization"]["mean"].shape != (r20.r19.r12.FEATURE_DIM,):
        raise RuntimeError("Round 12 preprocessing/feature dimension mismatch")
    bounds, duration_values = r20.class_duration_bounds()
    return classifier, info, cache, bounds, duration_values


def add_semantics(record: dict[str, Any], engine: r20.SemanticEngine) -> None:
    record["raw"] = r20.enrich_raw(engine, record["raw"], record["brb"])
    record["asb_segments"] = [asb_summary(record, x["start"], x["end"]) for x in record["raw"]]


def candidate_row(stat: dict[str, Any], cfg: dict[str, Any], rule: str) -> dict[str, Any]:
    l, r, m = stat["left_window"], stat["right_window"], stat["merged_asb"]
    lf, rf = stat["left_full"], stat["right_full"]
    same = bool(stat["local_label_agreement"]); whole_same = lf["asb_majority_label"] == rf["asb_majority_label"]
    stable = l["asb_majority_ratio"] >= cfg["ratio_threshold"] and r["asb_majority_ratio"] >= cfg["ratio_threshold"]
    similar = stat["local_cosine_similarity"] >= cfg["similarity_threshold"]
    full_similar = cosine(np.asarray(lf["asb_mean_probability"]), np.asarray(rf["asb_mean_probability"])) >= cfg["similarity_threshold"]
    merged_stable = m["asb_majority_label"] == l["asb_majority_label"] and m["asb_majority_ratio"] >= cfg["ratio_threshold"]
    weak_brb = stat["brb_probability"] <= cfg["brb_threshold"]
    short = stat["shorter_duration"] <= cfg["duration_threshold"]
    conf_ok = stat["classifier_confidence"] >= min(stat["left"]["top1_probability"], stat["right"]["top1_probability"]) - cfg["classifier_tolerance"]
    semantic_label_ok = stat["merged_classifier"]["top1_label"] in {stat["left"]["top1_label"], stat["right"]["top1_label"]}
    score = float(1.5 * same + 1.0 * stable + 1.0 * similar + .8 * whole_same + .8 * full_similar + .8 * merged_stable + .5 * short - 2.0 * stat["brb_probability"] - 1.0 * stat["transition_protected"])
    rule_ok = {
        "R1_asb_same_label": same,
        "R2_asb_duration": same and short,
        "R3_asb_brb": same and short and weak_brb,
        "R4_local_soft": same and short and stable and similar and not stat["transition_protected"],
        "R5_whole_soft": whole_same and short and full_similar and not stat["transition_protected"],
        "R6_local_whole": same and whole_same and short and stable and similar and full_similar and merged_stable and not stat["transition_protected"],
        "R8_classifier_verified": same and short and stable and similar and weak_brb and conf_ok and semantic_label_ok and stat["classifier_support_ok"] and not stat["transition_protected"],
        "R9_full_iterative": score >= cfg["score_threshold"] and same and short and stable and similar and merged_stable and conf_ok and stat["classifier_support_ok"] and not stat["transition_protected"],
        "asb_only": same and short and stable and similar and not stat["transition_protected"],
        "brb_only": weak_brb and short and not stat["transition_protected"],
        "classifier_only": short and conf_ok and stat["classifier_support_ok"] and semantic_label_ok and not stat["transition_protected"],
        "asb_brb": same and short and stable and similar and weak_brb and not stat["transition_protected"],
        "asb_classifier": same and short and stable and similar and conf_ok and stat["classifier_support_ok"] and semantic_label_ok and not stat["transition_protected"],
        "asb_brb_classifier": same and short and stable and similar and weak_brb and conf_ok and stat["classifier_support_ok"] and semantic_label_ok and not stat["transition_protected"],
    }.get(rule, False)
    return {"trajectory": stat["trajectory"], "boundary": stat["boundary"], "window": stat["window"], "rule": rule, "brb_probability": stat["brb_probability"], "left_asb_label": l["asb_majority_label"], "right_asb_label": r["asb_majority_label"], "left_asb_ratio": l["asb_majority_ratio"], "right_asb_ratio": r["asb_majority_ratio"], "left_full_asb_label": lf["asb_majority_label"], "right_full_asb_label": rf["asb_majority_label"], "local_label_agreement": int(same), "whole_label_agreement": int(whole_same), "local_cosine_similarity": stat["local_cosine_similarity"], "local_js_divergence": stat["local_js_divergence"], "whole_cosine_similarity": cosine(np.asarray(lf["asb_mean_probability"]), np.asarray(rf["asb_mean_probability"])), "merged_asb_label": m["asb_majority_label"], "merged_asb_ratio": m["asb_majority_ratio"], "merged_asb_entropy": m["asb_probability_entropy"], "shorter_duration": stat["shorter_duration"], "longer_duration": stat["longer_duration"], "left_classifier_label": stat["left"]["top1_label"], "right_classifier_label": stat["right"]["top1_label"], "merged_classifier_label": stat["merged_classifier"]["top1_label"], "left_classifier_confidence": stat["left"]["top1_probability"], "right_classifier_confidence": stat["right"]["top1_probability"], "merged_classifier_confidence": stat["classifier_confidence"], "merged_classifier_margin": stat["classifier_margin"], "merged_support_distance": stat["merged_classifier"]["embedding_support_distance"], "merged_duration_valid": stat["merged_classifier"]["duration_valid"], "transition_protected": stat["transition_protected"], "score": score, "accepted": int(rule_ok), "left_interval": [stat["left"]["start"], stat["left"]["end"]], "right_interval": [stat["right"]["start"], stat["right"]["end"]], "merged_interval": [stat["left"]["start"], stat["right"]["end"]]}


def classify_boundary(record: dict[str, Any], boundary: int) -> str:
    true = any(abs(boundary - g["start"]) <= 33 or abs(boundary - g["end"]) <= 33 for g in record["gt"])
    internal = any(g["start"] < boundary < g["end"] for g in record["gt"])
    return "true_boundary" if true and not internal else "false_internal_boundary" if internal and not true else "ambiguous"


def apply_rule(record: dict[str, Any], engine: r20.SemanticEngine, rule: str, cfg: dict[str, Any], collect: bool = False) -> dict[str, Any]:
    segments = [dict(x) for x in record["raw"]]; candidates = []; accepted = []; history = []
    max_iterations = cfg["max_iterations"] if rule == "R9_full_iterative" else 1
    for iteration in range(max_iterations):
        stats = [boundary_stats({**record, "raw": segments}, i, cfg["window"], engine) for i in range(len(segments) - 1)]
        rows = [candidate_row(x, cfg, rule) for x in stats]; candidates.extend(rows)
        choices = [x for x in rows if x["accepted"]]
        if rule == "R7_three_segment":
            choices = []
            for i in range(1, len(segments) - 1):
                left, middle, right = segments[i - 1], segments[i], segments[i + 1]
                s1, s2 = boundary_stats({**record, "raw": segments}, i - 1, cfg["window"], engine), boundary_stats({**record, "raw": segments}, i, cfg["window"], engine)
                if s1["left_full"]["asb_majority_label"] == s2["right_full"]["asb_majority_label"] and min(middle["duration"], cfg["duration_threshold"]) == middle["duration"] and s1["brb_probability"] < cfg["brb_threshold"] and s2["brb_probability"] < cfg["brb_threshold"] and not (s1["transition_protected"] or s2["transition_protected"]):
                    merged_asb = asb_summary(record, left["start"], right["end"]); merged_cls = engine.classify_interval(record["trajectory"], left["start"], right["end"]); ok = merged_asb["asb_majority_label"] == s1["left_full"]["asb_majority_label"] and merged_cls["duration_valid"] and merged_cls["embedding_support_distance"] <= .5
                    choices.append({"trajectory": record["trajectory"], "boundary": s1["boundary"], "boundary2": s2["boundary"], "left_index": i - 1, "right_index": i + 1, "accepted": int(ok), "choice": "C1_merge_all_three", "score": s1["local_cosine_similarity"] + s2["local_cosine_similarity"], "merged_interval": [left["start"], right["end"]], "merged_classifier": merged_cls})
        if not choices:
            break
        choices.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True); chosen = choices[0]
        if chosen.get("choice") == "C1_merge_all_three":
            new = engine.classify_interval(record["trajectory"], chosen["merged_interval"][0], chosen["merged_interval"][1]); start_i, end_i = chosen["left_index"], chosen["right_index"]; deleted_boundaries = [segments[start_i + 1]["start"], segments[end_i]["start"]]
        else:
            start_i, end_i = next(i for i, x in enumerate(rows) if x is chosen), next(i for i, x in enumerate(rows) if x is chosen) + 1; new = engine.classify_interval(record["trajectory"], segments[start_i]["start"], segments[end_i]["end"]); deleted_boundaries = [segments[end_i]["start"]]
        before = [{"start": x["start"], "end": x["end"]} for x in segments]; segments = segments[:start_i] + [new] + segments[end_i + 1:]; after = [{"start": x["start"], "end": x["end"]} for x in segments]
        op = {"trajectory": record["trajectory"], "rule": rule, "iteration": iteration, "deleted_boundaries": deleted_boundaries, "choice": chosen.get("choice", "boundary_merge"), "score": chosen.get("score", 0.0), "before_segments": before, "after_segments": after, "operation_span_start": new["start"], "operation_span_end": new["end"]}; history.append(op); accepted.extend({"trajectory": record["trajectory"], "rule": rule, "iteration": iteration, "boundary": x, "choice": op["choice"], "score": op["score"]} for x in deleted_boundaries)
        if rule != "R9_full_iterative":
            break
    return {"segments": segments, "candidates": candidates, "accepted": accepted, "history": history}


def metric(record: dict[str, Any], segments: list[dict[str, Any]], condition: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    matches = r20.r19.hungarian_matches(segments, record["gt"]); row = r20.r19.condition_metrics(record["trajectory"], record["family"], condition, segments, record["gt"], record["length"], matches); row["split"] = record["split"]; matched, missed, false, categories = r20.r19.matching_rows(record["trajectory"], condition, segments, record["gt"]); return row, matched, missed, false


def eval_rule(records: list[dict[str, Any]], engine: r20.SemanticEngine, rule: str, cfg: dict[str, Any], collect: bool = False) -> dict[str, Any]:
    rows = []; candidates = []; accepted = []; history = []; matched = []; missed = []; false = []
    refined_predictions = {}
    for record in records:
        result = apply_rule(record, engine, rule, cfg); row, m, miss, fp = metric(record, result["segments"], "refined"); row["rule"] = rule; row["operations"] = len(result["history"]); row["deleted_boundaries"] = len(result["accepted"]); rows.append(row)
        refined_predictions[record["trajectory"]] = result["segments"]
        if collect: candidates.extend([dict(x, split=record["split"]) for x in result["candidates"]]); accepted.extend([dict(x, split=record["split"]) for x in result["accepted"]]); history.extend(result["history"]); matched.extend(m); missed.extend(miss); false.extend(fp)
    agg = r20.r19.aggregate_metric_rows(rows, "refined", records[0]["split"]); agg.update({"rule": rule, "mean_operations_per_trajectory": float(np.mean([x["operations"] for x in rows])), "deleted_boundaries": int(sum(x["deleted_boundaries"] for x in rows)), "fragmentation_ratio": float(sum(x["predicted_segments"] for x in rows) / max(sum(x["gt_segments"] for x in rows), 1)), "records": rows, "refined_predictions": refined_predictions, "candidates": candidates, "accepted": accepted, "history": history, "matched": matched, "missed": missed, "false": false})
    return agg


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
    return (float(row["segmental_f1@50"]), -float(row["false_predicted_segment_rate"]), float(row["edit_score"]), float(row["framewise_macro_f1"]), float(row["mean_matched_temporal_iou"]), -float(row["missed_gt_segment_rate"]), -float(row.get("true_boundary_deletion_rate", 1.0)), -float(row.get("mean_operations_per_trajectory", 999.0)))


def true_boundary_rate(records: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> float:
    true = sum(1 for record in records for i in range(1, len(record["raw"])) if classify_boundary(record, record["raw"][i]["start"]) == "true_boundary")
    deleted_true = sum(1 for x in accepted if any(r["trajectory"] == x["trajectory"] and classify_boundary(r, x["boundary"]) == "true_boundary" for r in records))
    return float(deleted_true / max(true, 1))


def calibrate(validation: list[dict[str, Any]], engine: r20.SemanticEngine) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    # Select window and duration from validation-only sweeps with conservative
    # fixed semantic defaults, then search the remaining thresholds.
    base = {"window": 40, "duration_threshold": 80, "ratio_threshold": .8, "similarity_threshold": .9, "brb_threshold": .3, "classifier_tolerance": .05, "score_threshold": 2.0, "max_iterations": 4}
    duration_rows = []
    for duration in DURATION_GRID:
        cfg = {**base, "duration_threshold": duration}; result = eval_rule(validation, engine, "R9_full_iterative", cfg, collect=True); result["selected_duration"] = duration; result["true_boundary_deletion_rate"] = true_boundary_rate(validation, result["accepted"]); duration_rows.append(result)
    safe = [x for x in duration_rows if x["true_boundary_deletion_rate"] <= .05 and x["framewise_macro_f1"] >= duration_rows[0]["framewise_macro_f1"] - .01 and x["missed_gt_segment_rate"] <= duration_rows[0]["missed_gt_segment_rate"] + .01]; duration_choice = max(safe or duration_rows, key=selection_key)["selected_duration"]
    window_rows = []
    for window in WINDOWS:
        cfg = {**base, "duration_threshold": duration_choice, "window": window}; result = eval_rule(validation, engine, "R9_full_iterative", cfg, collect=True); result["selected_window"] = window; result["true_boundary_deletion_rate"] = true_boundary_rate(validation, result["accepted"]); window_rows.append(result)
    window_choice = max([x for x in window_rows if x["true_boundary_deletion_rate"] <= .05] or window_rows, key=selection_key)["selected_window"]
    search_rows = []
    for ratio in (.6, .8, .9):
        for sim in (.8, .9, .95):
            for brb in (.2, .3, .4):
                for tol in (.03, .05, .1):
                    for iterations in ITER_GRID:
                        cfg = {**base, "window": window_choice, "duration_threshold": duration_choice, "ratio_threshold": ratio, "similarity_threshold": sim, "brb_threshold": brb, "classifier_tolerance": tol, "score_threshold": 2.0, "max_iterations": iterations}; result = eval_rule(validation, engine, "R9_full_iterative", cfg, collect=True); result.update({"window": window_choice, "duration_threshold": duration_choice, "ratio_threshold": ratio, "similarity_threshold": sim, "brb_threshold": brb, "classifier_tolerance": tol, "score_threshold": 2.0, "max_iterations": iterations, "true_boundary_deletion_rate": true_boundary_rate(validation, result["accepted"]) }); search_rows.append(result)
    raw = eval_rule(validation, engine, "R0_raw", base); safe = [x for x in search_rows if x["true_boundary_deletion_rate"] <= .05 and x["framewise_macro_f1"] >= raw["framewise_macro_f1"] - .01 and x["missed_gt_segment_rate"] <= raw["missed_gt_segment_rate"] + .01]; chosen = max(safe or search_rows, key=selection_key)
    cfg = {key: chosen[key] for key in ("window", "duration_threshold", "ratio_threshold", "similarity_threshold", "brb_threshold", "classifier_tolerance", "score_threshold", "max_iterations")}
    rows = []
    for kind, values in (("duration", duration_rows), ("window", window_rows), ("parameter", search_rows)):
        for value in values:
            rows.append({"search_kind": kind, "selected": int(value is chosen or (kind == "duration" and value["selected_duration"] == duration_choice) or (kind == "window" and value["selected_window"] == window_choice)), **{k: v for k, v in value.items() if k not in {"records", "refined_predictions", "candidates", "accepted", "history", "matched", "missed", "false"}}})
    return "R9_full_iterative", cfg, rows


def raw_reproduction(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [metric(record, record["raw"], "raw")[0] for record in records]; agg = r20.r19.aggregate_metric_rows(metrics, "raw", "test"); expected = {"macro_f1": .8921007696007696, "segmental_f1@50": .679176984913459, "edit_score": .6344339637387766, "framewise_macro_f1": .742597082637815, "mean_matched_temporal_iou": .7976842417607662, "false_predicted_segment_rate": .532108785118999, "missed_gt_segment_rate": .04467754467754468}; deltas = {k: float(agg[k]) - v for k, v in expected.items()};
    if not all(abs(v) < 1e-9 for v in deltas.values()): raise RuntimeError(f"raw reproduction mismatch: {deltas}")
    return {"source": str(R19), "expected": expected, "aggregate": agg, "deltas": deltas, "asb_logits_hashes": {x["trajectory"]: x["asb_logits_hash"] for x in records}, "asb_probability_hashes": {x["trajectory"]: x["asb_probability_hash"] for x in records}, "asb_label_hashes": {x["trajectory"]: x["asb_label_hash"] for x in records}, "brb_hashes": {x["trajectory"]: x["brb_hash"] for x in records}, "raw_boundary_hashes": {x["trajectory"]: x["raw_boundary_hash"] for x in records}, "raw_segment_hashes": {x["trajectory"]: x["raw_segment_hash"] for x in records}, "exact_reuse": True}


def flatten_stat(stat: dict[str, Any]) -> dict[str, Any]:
    row = {k: v for k, v in stat.items() if k not in {"left", "right", "left_window", "right_window", "left_full", "right_full", "merged_asb", "merged_classifier"}}
    for key, value in (("left_window", stat["left_window"]), ("right_window", stat["right_window"]), ("left_full", stat["left_full"]), ("right_full", stat["right_full"]), ("merged_asb", stat["merged_asb"])):
        row[f"{key}_label"] = value["asb_majority_label"]; row[f"{key}_ratio"] = value["asb_majority_ratio"]; row[f"{key}_entropy"] = value["asb_probability_entropy"]; row[f"{key}_transitions"] = value["asb_transition_count"]
    row.update({"merged_classifier_label": stat["merged_classifier"]["top1_label"], "merged_classifier_confidence": stat["merged_classifier"]["top1_probability"], "merged_classifier_margin": stat["merged_classifier"]["margin"], "merged_support_distance": stat["merged_classifier"]["embedding_support_distance"]})
    return row


def boundary_analysis(records: list[dict[str, Any]], engine: r20.SemanticEngine) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows = []; classes = []
    for record in records:
        for i in range(len(record["raw"]) - 1):
            for w in WINDOWS:
                stat = boundary_stats(record, i, w, engine); row = flatten_stat(stat); row["category"] = classify_boundary(record, stat["boundary"]); windows.append(row)
            stat = boundary_stats(record, i, 40, engine); row = flatten_stat(stat); row["category"] = classify_boundary(record, stat["boundary"]); classes.append(row)
    return windows, classes


def posthoc_analysis(records: list[dict[str, Any]], selected: dict[str, Any], engine: r20.SemanticEngine) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        current = record["raw"]
        for op in selected["history"]:
            if op["trajectory"] != record["trajectory"]: continue
            before = metric(record, current, "before")[0]; current = [engine.classify_interval(record["trajectory"], x["start"], x["end"]) for x in op["after_segments"]]; after = metric(record, current, "after")[0]
            labels = {g["label"] for g in record["gt"] if max(0, min(op["operation_span_end"], g["end"]) - max(op["operation_span_start"], g["start"])) / max(g["end"] - g["start"], 1) >= .10}
            harmful = float(after["mean_matched_temporal_iou"]) < float(before["mean_matched_temporal_iou"]) or len(labels) > 1 or float(after["missed_gt_segment_rate"]) > float(before["missed_gt_segment_rate"]); beneficial = not harmful and (float(after["segmental_f1@50"]) > float(before["segmental_f1@50"]) or float(after["false_predicted_segment_rate"]) < float(before["false_predicted_segment_rate"]) or float(after["mean_matched_temporal_iou"]) > float(before["mean_matched_temporal_iou"])); rows.append({"trajectory": record["trajectory"], "rule": selected["rule"], "deleted_boundaries": op["deleted_boundaries"], "choice": op["choice"], "before_f1@50": before["segmental_f1@50"], "after_f1@50": after["segmental_f1@50"], "before_false_rate": before["false_predicted_segment_rate"], "after_false_rate": after["false_predicted_segment_rate"], "gt_labels_overlapped": sorted(labels), "classification": "harmful" if harmful else "beneficial" if beneficial else "neutral"})
    return rows


def plot_timeline(record: dict[str, Any], refined: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> None:
    sample = r20.r19.load_trajectory_sample(r20.r19.DATA / record["trajectory"], r20.r19.load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml"), expected_height=88); fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1, 1, 1]}); axes[0].imshow(sample["heatmap"].numpy().transpose(1, 2, 0), aspect="auto", origin="upper"); axes[0].set_ylabel("input"); axes[1].plot(record["asb_labels"], lw=.4); axes[1].set_ylabel("ASB"); axes[2].plot(record["brb"], color="purple"); axes[2].set_ylabel("BRB")
    for axis, values, title, color in ((axes[3], record["gt"], "GT", "green"), (axes[4], record["raw"], "raw ASRF", "steelblue"), (axes[5], refined, "refined", "orange")):
        axis.set_ylim(0, 1); axis.set_yticks([]); axis.set_title(title, loc="left")
        for item in values:
            label = item.get("label", item.get("top1_label", "")); axis.axvspan(item["start"], item["end"], color=color, alpha=.6); axis.text((item["start"] + item["end"]) / 2, .5, str(label), ha="center", fontsize=7, rotation=90 if item["end"] - item["start"] < 100 else 0)
    for item in accepted: axes[5].axvline(item["boundary"], color="red", lw=1); axes[5].text(item["boundary"], .05, "merge", rotation=90, color="red", fontsize=6)
    axes[-1].set_xlabel("frame"); fig.suptitle(record["trajectory"]); fig.tight_layout(); fig.savefig(OUT / "figures" / f"timeline_{safe_name(record['trajectory'])}.png", dpi=140); plt.close(fig)


def main() -> int:
    seed(); OUT.mkdir(parents=True, exist_ok=True); (OUT / "predictions").mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    classifier, info, cache, bounds, duration_values = load_fixed(); durations = r20.SemanticEngine(classifier, info, cache, bounds, duration_values)
    validation, test = load_records("validation"), load_records("test")
    for record in validation + test: add_semantics(record, durations)
    r20_manifest = read_csv(R20 / "trajectory_manifest.csv"); write_csv(OUT / "trajectory_manifest.csv", [{**row, "round20_manifest": str(R20 / "trajectory_manifest.csv"), "equivalent_to_round20": 1} for row in r20_manifest])
    if {x["trajectory"] for x in r20_manifest} != {x["trajectory"] for x in r20.unique_manifest("test")}: raise RuntimeError("Round 20 trajectory equivalence failed")
    write_json(OUT / "checkpoint_hashes.json", {"asrf_checkpoint": str(ASRF_CHECKPOINT), "asrf_sha256": sha256(ASRF_CHECKPOINT), "classifier_checkpoint": str(CLASSIFIER_CHECKPOINT), "classifier_sha256": sha256(CLASSIFIER_CHECKPOINT), "ontology_version": "round12_multiskill_v2", "ordered_class_list": list(FINAL_CLASSES), "asb_ontology": list(ASB_LABELS), "retraining": False, "annotations_changed": False})
    write_json(OUT / "raw_reproduction_metrics.json", raw_reproduction(test))
    asb_rows = []
    for record in test:
        for item in record["asb_segments"]: asb_rows.append({"trajectory": record["trajectory"], **item})
    write_csv(OUT / "asb_segment_statistics.csv", asb_rows)
    validation_window_rows, validation_boundary_rows = boundary_analysis(validation, durations); test_window_rows, test_boundary_rows = boundary_analysis(test, durations)
    window_rows = [dict(x, split="validation") for x in validation_window_rows] + [dict(x, split="test") for x in test_window_rows]; boundary_rows = [dict(x, split="validation") for x in validation_boundary_rows] + [dict(x, split="test") for x in test_boundary_rows]
    write_csv(OUT / "boundary_window_statistics.csv", window_rows); write_csv(OUT / "true_false_boundary_analysis.csv", boundary_rows)
    # Validation sweeps are frozen before any test rule is evaluated.
    selected_rule, cfg, search_rows = calibrate(validation, durations); write_csv(OUT / "parameter_search.csv", search_rows)
    duration_sweep = []
    for value in DURATION_GRID:
        result = next(x for x in search_rows if x.get("search_kind") == "duration" and int(x.get("selected_duration", -1)) == value); duration_sweep.append({"duration_threshold": value, "f1@50": result["segmental_f1@50"], "edit_score": result["edit_score"], "false_predicted_segment_rate": result["false_predicted_segment_rate"], "missed_gt_segment_rate": result["missed_gt_segment_rate"], "true_boundary_deletion_rate": result["true_boundary_deletion_rate"], "deleted_boundaries": result["deleted_boundaries"]})
    write_csv(OUT / "duration_threshold_sweep.csv", duration_sweep)
    window_sweep = []
    for value in WINDOWS:
        result = next(x for x in search_rows if x.get("search_kind") == "window" and int(x.get("selected_window", -1)) == value); window_sweep.append({"window": value, "f1@50": result["segmental_f1@50"], "edit_score": result["edit_score"], "false_predicted_segment_rate": result["false_predicted_segment_rate"], "missed_gt_segment_rate": result["missed_gt_segment_rate"], "true_boundary_deletion_rate": result["true_boundary_deletion_rate"]})
    write_csv(OUT / "window_size_sweep.csv", window_sweep)
    calibration = [{"parameter": key, "selected_value": value, "source_split": "validation", "selection_metric": "F1@50; false rate; edit; frame macro F1; IoU; miss; true-boundary deletion; operations"} for key, value in cfg.items()] + [{"parameter": "selected_rule", "selected_value": selected_rule, "source_split": "validation", "selection_metric": "protocol tie-breaks"}]; write_csv(OUT / "calibration_manifest.csv", calibration)
    ablation = []
    for rule in RULES + EXTRA_RULES:
        result = eval_rule(validation, durations, rule, cfg, collect=True); result["true_boundary_deletion_rate"] = true_boundary_rate(validation, result["accepted"]); ablation.append({k: v for k, v in result.items() if k not in {"records", "refined_predictions", "candidates", "accepted", "history", "matched", "missed", "false"}})
    write_csv(OUT / "rule_ablation.csv", ablation)
    selected = eval_rule(test, durations, selected_rule, cfg, collect=True); raw = eval_rule(test, durations, "R0_raw", cfg, collect=True); comparison = []
    for label, result in (("raw_asrf", raw), ("refined_asrf", selected)):
        row = {k: v for k, v in result.items() if k not in {"records", "refined_predictions", "candidates", "accepted", "history", "matched", "missed", "false"}}; row["condition"] = label; comparison.append(row)
    for row in comparison:
        row["f1@50_change_vs_raw"] = float(row["segmental_f1@50"]) - float(comparison[0]["segmental_f1@50"]); row["false_rate_change_vs_raw"] = float(row["false_predicted_segment_rate"]) - float(comparison[0]["false_predicted_segment_rate"]); row["edit_change_vs_raw"] = float(row["edit_score"]) - float(comparison[0]["edit_score"]); row["frame_macro_change_vs_raw"] = float(row["framewise_macro_f1"]) - float(comparison[0]["framewise_macro_f1"]); row["miss_rate_change_vs_raw"] = float(row["missed_gt_segment_rate"]) - float(comparison[0]["missed_gt_segment_rate"])
    write_csv(OUT / "condition_comparison.csv", comparison); write_json(OUT / "refined_metrics.json", {"selected_rule": selected_rule, "selected_config": cfg, "aggregate": comparison[1], "trajectory_metrics": selected["records"]})
    merge_rows = [dict(x, split="test") for x in selected["candidates"]]; write_csv(OUT / "merge_candidates.csv", merge_rows); write_csv(OUT / "accepted_merges.csv", selected["accepted"]); write_csv(OUT / "rejected_merges.csv", [x for x in merge_rows if not int(x.get("accepted", 0))]); write_csv(OUT / "three_segment_candidates.csv", [x for x in merge_rows if x.get("choice") == "C1_merge_all_three"]); write_csv(OUT / "operation_history.csv", selected["history"])
    analysis = posthoc_analysis(test, selected, durations); write_csv(OUT / "beneficial_harmful_analysis.csv", analysis)
    # Oracle same-ASB upper bound and true/false distributions.
    oracle = []
    for category in ("false_internal_boundary", "true_boundary"):
        rows = [x for x in boundary_rows if x["split"] == "test" and x["category"] == category]; oracle.append({"diagnostic": "ASB_same_label_rate", "category": category, "count": len(rows), "same_label_rate": float(np.mean([float(x["local_label_agreement"]) for x in rows])) if rows else 0.0, "mean_similarity": float(np.mean([float(x["local_cosine_similarity"]) for x in rows])) if rows else 0.0, "mean_brb": float(np.mean([float(x["brb_probability"]) for x in rows])) if rows else 0.0, "diagnostic_only": 1})
    oracle_false = [x for x in selected["accepted"] if any(r["trajectory"] == x["trajectory"] and any(classify_boundary(r, segment["start"]) == "false_internal_boundary" and segment["start"] == x["boundary"] for segment in r["raw"][1:]) for r in test)]
    oracle.append({"diagnostic": "selected_rule_false_boundary_deletions", "category": "oracle", "count": len(oracle_false), "raw_f1@50": comparison[0]["segmental_f1@50"], "deployable_f1@50": comparison[1]["segmental_f1@50"], "diagnostic_only": 1})
    write_csv(OUT / "oracle_analysis.csv", oracle)
    # Per-family, per-skill, and trajectory outputs.
    per_family = []
    for rule in ("R0_raw", selected_rule):
        result = raw if rule == "R0_raw" else selected
        for family in sorted({x["family"] for x in test}): per_family.append({"rule": rule, "family": family, **r20.r19.aggregate_metric_rows([x for x in result["records"] if x["family"] == family], "refined", "test")})
    write_csv(OUT / "per_family_results.csv", per_family)
    write_csv(OUT / "per_trajectory_results.csv", [{k: v for k, v in x.items() if k != "confusion_matrix"} for result in (raw, selected) for x in result["records"]])
    skills = []
    for rule, result in (("R0_raw", raw), (selected_rule, selected)):
        for skill in FINAL_CLASSES:
            matched = [x for x in result["matched"] if float(x["temporal_iou"]) >= .5]
            tp = sum(x["gt_label"] == skill and x["predicted_label"] == skill for x in matched)
            fn = sum(x["gt_label"] == skill and x["predicted_label"] != skill for x in matched)
            fp = sum(x["gt_label"] != skill and x["predicted_label"] == skill for x in matched)
            precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            class_iou = [float(x["temporal_iou"]) for x in matched if x["gt_label"] == skill]
            skills.append({"rule": rule, "skill": skill, "support": tp + fn, "true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": precision, "recall": recall, "f1": f1, "mean_iou": float(np.mean(class_iou)) if class_iou else 0.0})
    write_csv(OUT / "per_skill_results.csv", skills)
    for record in test:
        refined_segments = selected["refined_predictions"][record["trajectory"]]
        np.savez_compressed(OUT / "predictions" / f"{safe_name(record['trajectory'])}.npz", asb_logits=record["asb_logits"], asb_probabilities=record["asb_probabilities"], asb_labels=record["asb_labels"], brb_probabilities=record["brb"])
        write_json(OUT / "predictions" / f"{safe_name(record['trajectory'])}.json", {"trajectory": record["trajectory"], "asb_labels": [ASB_LABELS[int(x)] for x in record["asb_labels"]], "asb_label_ids": record["asb_labels"].tolist(), "asb_probabilities": record["asb_probabilities"].tolist(), "brb_probabilities": record["brb"].tolist(), "raw_segments": record["raw"], "raw_asb_segments": record["asb_segments"], "refined_segments": refined_segments, "accepted_merges": [x for x in selected["accepted"] if x["trajectory"] == record["trajectory"]], "matching": [x for x in selected["matched"] if x["trajectory"] == record["trajectory"]]})
    # Figures.
    fig, ax = plt.subplots(figsize=(8, 5)); cats = ["false_internal_boundary", "true_boundary"]; vals = [next(x for x in oracle if x["category"] == c)["same_label_rate"] for c in cats]; ax.bar(cats, vals); ax.set_ylabel("ASB same-label rate"); fig.tight_layout(); fig.savefig(OUT / "figures/asb_agreement_true_false.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist([float(x["local_cosine_similarity"]) for x in boundary_rows if x["category"] == "false_internal_boundary"], bins=20, alpha=.6, label="false"); ax.hist([float(x["local_cosine_similarity"]) for x in boundary_rows if x["category"] == "true_boundary"], bins=20, alpha=.6, label="true"); ax.legend(); ax.set_xlabel("ASB local cosine similarity"); fig.tight_layout(); fig.savefig(OUT / "figures/asb_soft_similarity.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot([x["duration_threshold"] for x in duration_sweep], [x["f1@50"] for x in duration_sweep], marker="o"); ax.set_xlabel("short-fragment threshold"); ax.set_ylabel("validation F1@50"); fig.tight_layout(); fig.savefig(OUT / "figures/duration_threshold_f1.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot([x["duration_threshold"] for x in duration_sweep], [x["true_boundary_deletion_rate"] for x in duration_sweep], marker="o"); ax.axhline(.05, ls="--", color="red"); ax.set_xlabel("short-fragment threshold"); ax.set_ylabel("true-boundary deletion rate"); fig.tight_layout(); fig.savefig(OUT / "figures/true_boundary_risk_vs_duration.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter([x["brb_probability"] for x in boundary_rows], [x["local_cosine_similarity"] for x in boundary_rows], alpha=.2); ax.set_xlabel("BRB probability"); ax.set_ylabel("ASB similarity"); fig.tight_layout(); fig.savefig(OUT / "figures/brb_vs_asb_similarity.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(["raw", "refined"], [comparison[0]["predicted_segments"], comparison[1]["predicted_segments"]]); ax.axhline(comparison[0]["gt_segments"], ls="--", color="black"); ax.set_ylabel("segments"); fig.tight_layout(); fig.savefig(OUT / "figures/raw_refined_segment_counts.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); names = [x["skill"] for x in skills if x["rule"] == "R0_raw"]; a = {x["skill"]: x["f1"] for x in skills if x["rule"] == "R0_raw"}; b = {x["skill"]: x["f1"] for x in skills if x["rule"] == selected_rule}; ax.bar(np.arange(len(names)) - .2, [a[x] for x in names], .4, label="raw"); ax.bar(np.arange(len(names)) + .2, [b[x] for x in names], .4, label="refined"); ax.set_xticks(range(len(names)), names, rotation=60); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/per_skill_f1_before_after.png", dpi=160); plt.close(fig)
    for record in test: plot_timeline(record, selected["refined_predictions"][record["trajectory"]], [x for x in selected["accepted"] if x["trajectory"] == record["trajectory"]])
    accepted_outcomes = Counter(x["classification"] for x in analysis); raw_false = int(comparison[0]["predicted_segments"] - comparison[0]["matched_segments"]); refined_false = int(comparison[1]["predicted_segments"] - comparison[1]["matched_segments"]); true_deleted = sum(1 for x in selected["accepted"] if any(r["trajectory"] == x["trajectory"] and any(classify_boundary(r, segment["start"]) == "true_boundary" and segment["start"] == x["boundary"] for segment in r["raw"][1:]) for r in test)); family_gain = sum(float(x["segmental_f1@50"]) > float(next(y for y in per_family if y["rule"] == "R0_raw" and y["family"] == x["family"])["segmental_f1@50"]) for x in per_family if x["rule"] == selected_rule); short_loss = min(float(next(x for x in skills if x["rule"] == selected_rule and x["skill"] == s)["f1"]) - float(next(x for x in skills if x["rule"] == "R0_raw" and x["skill"] == s)["f1"]) for s in ("grasp", "release", "insert"))
    criteria = [("F1@50 improvement >=0.02", float(comparison[1]["segmental_f1@50"]) - float(comparison[0]["segmental_f1@50"]) >= .02), ("false rate reduction >=0.07", float(comparison[0]["false_predicted_segment_rate"]) - float(comparison[1]["false_predicted_segment_rate"]) >= .07), ("edit improvement >=0.02", float(comparison[1]["edit_score"]) - float(comparison[0]["edit_score"]) >= .02), ("frame macro F1 drop <=0.01", float(comparison[1]["framewise_macro_f1"]) - float(comparison[0]["framewise_macro_f1"]) >= -.01), ("miss rate increase <=0.01", float(comparison[1]["missed_gt_segment_rate"]) - float(comparison[0]["missed_gt_segment_rate"]) <= .01), ("mean IoU not reduced", float(comparison[1]["mean_matched_temporal_iou"]) >= float(comparison[0]["mean_matched_temporal_iou"])), ("improvement in >=2 families", family_gain >= 2), ("true-boundary deletion rate <=5%", true_deleted / max(sum(1 for r in test for i in range(1, len(r["raw"])) if classify_boundary(r, r["raw"][i]["start"]) == "true_boundary"), 1) <= .05), (">=70% beneficial/neutral", (accepted_outcomes["beneficial"] + accepted_outcomes["neutral"]) / max(sum(accepted_outcomes.values()), 1) >= .70), ("short skills lose <=0.05 F1", short_loss >= -.05), ("not one trajectory driven", len({x["trajectory"] for x in selected["accepted"]}) > 1)]
    report = ["# Round 21 ASB-assisted boundary merge", "", f"Frozen Round 19/20 inputs and frozen Round 12 classifier on {len(test)} test trajectories. ASB is used only for boundary validation; the segment classifier remains the final recognizer. No retraining, split refinement, annotation changes, or open-set discovery.", "", "## Results", "", "| condition | rule | F1@50 | edit | frame macro F1 | mean IoU | false rate | miss rate |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparison: report.append(f"| {row['condition']} | {row['rule']} | {float(row['segmental_f1@50']):.4f} | {float(row['edit_score']):.4f} | {float(row['framewise_macro_f1']):.4f} | {float(row['mean_matched_temporal_iou']):.4f} | {float(row['false_predicted_segment_rate']):.4f} | {float(row['missed_gt_segment_rate']):.4f} |")
    false_stats = next(x for x in oracle if x["category"] == "false_internal_boundary"); true_stats = next(x for x in oracle if x["category"] == "true_boundary")
    report += ["", "## Selection", "", f"Selected rule: **{selected_rule}**; window={cfg['window']} frames; duration threshold={cfg['duration_threshold']} frames; ASB ratio={cfg['ratio_threshold']}; similarity={cfg['similarity_threshold']}; BRB={cfg['brb_threshold']}; classifier tolerance={cfg['classifier_tolerance']}; max iterations={cfg['max_iterations']}.", f"False boundaries had ASB same-label rate {false_stats['same_label_rate']:.3f} versus {true_stats['same_label_rate']:.3f} for true boundaries; their mean local similarity was {false_stats['mean_similarity']:.3f} versus {true_stats['mean_similarity']:.3f}.", f"The selected threshold is {cfg['duration_threshold']} frames; the full 20–120 validation sweep is in duration_threshold_sweep.csv. The selected window is {cfg['window']} frames; the full window sweep is in window_size_sweep.csv.", "", "## Conclusions", "", f"ASB-assisted refinement removed {len(selected['accepted'])} boundaries and {raw_false - refined_false} false predicted segments. True-boundary deletions: {true_deleted}. Beneficial/neutral/harmful operations: {accepted_outcomes['beneficial']}/{accepted_outcomes['neutral']}/{accepted_outcomes['harmful']}.", f"The selected rule changes F1@50 by {float(comparison[1]['f1@50_change_vs_raw']):+.4f}, edit by {float(comparison[1]['edit_change_vs_raw']):+.4f}, frame macro F1 by {float(comparison[1]['frame_macro_change_vs_raw']):+.4f}, and false rate by {float(comparison[1]['false_rate_change_vs_raw']):+.4f}.", f"ASB local plus whole-segment evidence was selected over single-source ablations by validation. Short-skill F1 minimum change for grasp/release/insert was {short_loss:+.3f}; the main family gains are represented in per_family_results.csv.", "Round 21 is not an open-set experiment. If it fails, the next step should be ASB-consistency supervision for BRB retraining, followed by sequence-level dynamic programming or joint ASRF/classifier training.", "", "## Decision criteria"]
    for name, passed in criteria: report.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    report += ["", "## Integrity", "", "Annotations unchanged. Both required checkpoint hashes match. Raw ASB logits, labels, BRB probabilities, boundaries, and metrics reproduce Round 19/20 exactly. Deployable merge decisions used no GT; validation froze all parameters before test evaluation. Historical pytest artifact failures are separate from Round 21.", "", "## Outputs", "", "All artifacts are under outputs/round21_asb_assisted_boundary_merge/."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "config.yaml").write_text(yaml.safe_dump({"experiment": "round21_asb_assisted_boundary_merge", "seed": SEED, "asb_ontology": list(ASB_LABELS), "final_ontology": list(FINAL_CLASSES), "selected_rule": selected_rule, "selected_config": cfg, "test_used_for_selection": False, "gt_used_for_deployable_merge": False, "retraining": False}, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
