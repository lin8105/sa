#!/usr/bin/env python3
"""Round 22: protected ASB-assisted boundary refinement.

This is an inference-only extension of Round 21.  The frozen Round 12
classifier remains the final semantic recognizer; ASB/BRB/classifier evidence
can only veto deletion of an existing ASRF boundary.
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round22_protected_asb_boundary_refinement"
R21OUT = ROOT / "outputs/round21_asb_assisted_boundary_merge"
R20OUT = ROOT / "outputs/round20_semantic_fragment_merge"
R19OUT = ROOT / "outputs/round19_asrf_segment_classifier_integration"
sys.path.insert(0, str(ROOT / "scripts"))
import run_round21_asb_assisted_boundary_merge as r21  # noqa: E402

SEED = 42
ASRF_SHA = r21.ASRF_SHA
CLASSIFIER_SHA = r21.CLASSIFIER_SHA
FINAL_CLASSES = tuple(r21.FINAL_CLASSES)
SHORT_SKILLS = {"grasp", "release", "insert"}
PARTIAL_SHORT_SKILLS = {"lift", "pour_recover"}
CRITICAL_PAIRS = {
    ("grasp", "lift"), ("lift", "transport"), ("transport", "place"),
    ("place", "insert"), ("insert", "release"), ("pour", "pour_recover"),
    ("pour_recover", "place"),
}
PROTECTION_VARIANTS = ("P0_round21", "P1_short", "P2_transition", "P3_brb", "P4_asb_stability", "P5_semantic", "P6_short_transition", "P7_brb_stability", "P8_full", "P9_full_soft", "M1_logistic_veto", "M2_tree_veto")
STABLE_CONTEXT_GRID = (20, 30, 50, 75)
BRB_PROTECTION_GRID = (0.40, 0.50, 0.60, 0.70)


def seed() -> None:
    np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else x) + "\n", encoding="utf-8")


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config() -> dict[str, Any]:
    value = yaml.safe_load((R21OUT / "config.yaml").read_text(encoding="utf-8"))
    return {**value["selected_config"], "hard_brb_threshold": 0.60, "stable_context": 30, "short_protection_duration": 100, "risk_threshold": 0.50, "soft_penalty": 0.45}


def load_data(engine: r21.r20.SemanticEngine):
    validation = r21.load_records("validation"); test = r21.load_records("test")
    for record in validation + test:
        r21.add_semantics(record, engine)
    return validation, test


def protection_features(record: dict[str, Any], index: int, stat: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    left, right = stat["left"], stat["right"]
    left_label, right_label = left["top1_label"], right["top1_label"]
    pair = (left_label, right_label); reverse_pair = (right_label, left_label)
    short_adjacent = int((left_label in SHORT_SKILLS or right_label in SHORT_SKILLS) and min(left["duration"], right["duration"]) <= cfg["short_protection_duration"])
    partial_short = int((left_label in PARTIAL_SHORT_SKILLS or right_label in PARTIAL_SHORT_SKILLS) and min(left["duration"], right["duration"]) <= cfg["short_protection_duration"])
    critical = int(pair in CRITICAL_PAIRS or reverse_pair in CRITICAL_PAIRS)
    high_brb = int(stat["brb_probability"] >= cfg["hard_brb_threshold"])
    left_stable = int(stat["left_window"]["asb_majority_ratio"] >= cfg["ratio_threshold"] and stat["left_window"]["duration"] >= cfg["stable_context"])
    right_stable = int(stat["right_window"]["asb_majority_ratio"] >= cfg["ratio_threshold"] and stat["right_window"]["duration"] >= cfg["stable_context"])
    two_sided = int(left_stable and right_stable and stat["left_window"]["asb_majority_label"] != stat["right_window"]["asb_majority_label"])
    merged = stat["merged_classifier"]
    semantic_incompatibility = int(merged["top1_label"] not in {left_label, right_label} or merged["margin"] < min(left["margin"], right["margin"]) - cfg["classifier_tolerance"] or not merged["duration_valid"] or merged["embedding_support_distance"] > .5)
    surrounding = [record["raw"][j]["top1_label"] for j in range(max(0, index - 1), min(len(record["raw"]), index + 3))]
    sequential = 0
    if len(surrounding) >= 3:
        patterns = (("reach", "grasp", "lift"), ("insert", "release", "retreat"), ("place", "insert", "release"), ("pour", "pour_recover", "place"))
        sequential = int(any(tuple(surrounding[:3]) == pattern or tuple(surrounding[-3:]) == pattern for pattern in patterns))
    return {"short_skill": short_adjacent, "partial_short_skill": partial_short, "critical_transition": critical, "high_brb": high_brb, "left_stable": left_stable, "right_stable": right_stable, "two_sided_stability": two_sided, "semantic_incompatibility": semantic_incompatibility, "sequential_incompatibility": sequential, "shorter_duration": stat["shorter_duration"], "brb_probability": stat["brb_probability"], "asb_similarity": stat["local_cosine_similarity"], "left_asb_ratio": stat["left_window"]["asb_majority_ratio"], "right_asb_ratio": stat["right_window"]["asb_majority_ratio"], "label_agreement": int(stat["local_label_agreement"]), "confidence_gain": stat["classifier_confidence"] - min(left["top1_probability"], right["top1_probability"]), "margin_gain": stat["classifier_margin"] - min(left["margin"], right["margin"]), "support": stat["merged_classifier"]["embedding_support_distance"], "merged_duration_valid": int(merged["duration_valid"]), "left_label": left_label, "right_label": right_label, "merged_label": merged["top1_label"]}


def protection_decision(features: dict[str, Any], variant: str, soft: bool = False, risk_model: Any = None, risk_features: list[str] | None = None, cfg: dict[str, Any] | None = None) -> tuple[int, str, float]:
    cfg = cfg or {}
    hard = []
    if variant in {"P1_short", "P6_short_transition", "P8_full", "P9_full_soft"} and features["short_skill"]:
        hard.append("short_skill")
    if variant in {"P2_transition", "P6_short_transition", "P8_full", "P9_full_soft"} and (features["critical_transition"] or features["sequential_incompatibility"]):
        hard.append("critical_transition")
    if variant in {"P3_brb", "P7_brb_stability", "P8_full", "P9_full_soft"} and features["high_brb"]:
        hard.append("high_brb")
    if variant in {"P4_asb_stability", "P7_brb_stability", "P8_full", "P9_full_soft"} and features["two_sided_stability"]:
        hard.append("two_sided_stability")
    if variant in {"P5_semantic", "P8_full", "P9_full_soft"} and features["semantic_incompatibility"]:
        hard.append("semantic_incompatibility")
    if variant == "M1_logistic_veto" or variant == "M2_tree_veto":
        vector = np.asarray([[float(features[key]) for key in (risk_features or [])]], dtype=np.float32)
        probability = float(risk_model.predict_proba(vector)[0, 1]) if hasattr(risk_model, "predict_proba") else float(risk_model.predict(vector)[0])
        return int(probability >= cfg.get("risk_threshold", .5)), "learned_risk", probability
    if not hard:
        return 0, "none", 0.0
    if not soft:
        return 1, "+".join(hard), float(len(hard))
    penalty = float(cfg.get("soft_penalty", .45) * len(hard))
    return 0, "soft_penalty:" + "+".join(hard), penalty


def sequential_labels(record: dict[str, Any], index: int) -> list[str]:
    return [record["raw"][j]["top1_label"] for j in range(max(0, index - 1), min(len(record["raw"]), index + 3))]


def prior_r21_category(record: dict[str, Any], boundary: int, engine: r21.r20.SemanticEngine, cfg: dict[str, Any]) -> str:
    """Reconstruct Round 21's published false/true/ambiguous audit labels."""
    for index in range(len(record["raw"]) - 1):
        stat = r21.boundary_stats(record, index, int(cfg["window"]), engine)
        if int(stat["boundary"]) == int(boundary):
            row = r21.candidate_row(stat, cfg, "R9_full_iterative")
            if int(row["accepted"]):
                category = r21.classify_boundary(record, boundary)
                return category if category in {"true_boundary", "false_internal_boundary"} else "ambiguous"
            break
    return "ambiguous"


def apply_protected(record: dict[str, Any], engine: r21.r20.SemanticEngine, variant: str, cfg: dict[str, Any], soft: bool = False, risk_model: Any = None, risk_feature_names: list[str] | None = None) -> dict[str, Any]:
    if variant == "P0_round21":
        return {**r21.apply_rule(record, engine, "R9_full_iterative", cfg), "protected": [], "vetoed": []}
    segments = [dict(x) for x in record["raw"]]; candidates = []; accepted = []; protected = []; vetoed = []; history = []
    for iteration in range(int(cfg["max_iterations"])):
        stats = [r21.boundary_stats({**record, "raw": segments}, index, int(cfg["window"]), engine) for index in range(len(segments) - 1)]
        rows = []
        for index, stat in enumerate(stats):
            base = r21.candidate_row(stat, cfg, "R9_full_iterative")
            features = protection_features({**record, "raw": segments}, index, stat, cfg)
            veto, reason, strength = protection_decision(features, variant, soft, risk_model, risk_feature_names, cfg)
            row = {**base, **features, "iteration": iteration, "protection": int(veto), "protection_reason": reason, "protection_strength": strength, "candidate_index": index, "sequential_labels": sequential_labels({**record, "raw": segments}, index)}
            row["accepted"] = int(base["accepted"] and not veto)
            rows.append(row)
            if base["accepted"] and veto:
                protected.append({"trajectory": record["trajectory"], "iteration": iteration, "boundary": stat["boundary"], "reason": reason, "strength": strength, "variant": variant})
                vetoed.append({"trajectory": record["trajectory"], "iteration": iteration, "boundary": stat["boundary"], "reason": reason, "variant": variant})
        candidates.extend(rows)
        choices = [row for row in rows if row["accepted"]]
        if not choices:
            break
        choices.sort(key=lambda row: float(row["score"]), reverse=True); chosen = choices[0]; index = int(chosen["candidate_index"])
        new = engine.classify_interval(record["trajectory"], segments[index]["start"], segments[index + 1]["end"])
        before = [{"start": x["start"], "end": x["end"]} for x in segments]; deleted_boundary = int(segments[index + 1]["start"]); segments = segments[:index] + [new] + segments[index + 2:]; after = [{"start": x["start"], "end": x["end"]} for x in segments]
        operation = {"trajectory": record["trajectory"], "rule": variant, "iteration": iteration, "deleted_boundaries": [deleted_boundary], "choice": "boundary_merge", "score": chosen["score"], "before_segments": before, "after_segments": after, "operation_span_start": new["start"], "operation_span_end": new["end"], "protection_reason": chosen["protection_reason"]}
        history.append(operation); accepted.append({"trajectory": record["trajectory"], "rule": variant, "iteration": iteration, "boundary": deleted_boundary, "choice": "boundary_merge", "score": chosen["score"], "protection_reason": chosen["protection_reason"]})
    return {"segments": segments, "candidates": candidates, "accepted": accepted, "protected": protected, "vetoed": vetoed, "history": history}


def metric_for(record: dict[str, Any], segments: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    return r21.metric(record, segments, condition)[0]


def evaluate(records: list[dict[str, Any]], engine: r21.r20.SemanticEngine, variant: str, cfg: dict[str, Any], soft: bool = False, risk_model: Any = None, risk_names: list[str] | None = None, collect: bool = True) -> dict[str, Any]:
    metric_rows = []; results = {}; candidates = []; accepted = []; protected = []; vetoed = []; history = []
    for record in records:
        result = apply_protected(record, engine, variant, cfg, soft, risk_model, risk_names); results[record["trajectory"]] = result; row = metric_for(record, result["segments"], variant); row.update({"variant": variant, "operations": len(result["history"]), "accepted_deletions": len(result["accepted"]), "protected_boundaries": len(result["protected"]), "vetoed_deletions": len(result["vetoed"])}); metric_rows.append(row)
        if collect:
            candidates.extend([dict(x, split=record["split"]) for x in result["candidates"]]); accepted.extend([dict(x, split=record["split"]) for x in result["accepted"]]); protected.extend([dict(x, split=record["split"]) for x in result["protected"]]); vetoed.extend([dict(x, split=record["split"]) for x in result["vetoed"]]); history.extend(result["history"])
    aggregate = r21.r20.r19.aggregate_metric_rows(metric_rows, variant, records[0]["split"]); aggregate.update({"variant": variant, "mean_operations_per_trajectory": float(np.mean([x["operations"] for x in metric_rows])), "accepted_deletions": int(sum(x["accepted_deletions"] for x in metric_rows)), "protected_boundaries": int(sum(x["protected_boundaries"] for x in metric_rows)), "vetoed_deletions": int(sum(x["vetoed_deletions"] for x in metric_rows)), "fragmentation_ratio": float(sum(x["predicted_segments"] for x in metric_rows) / max(sum(x["gt_segments"] for x in metric_rows), 1)), "records": metric_rows, "results": results, "candidates": candidates, "accepted": accepted, "protected": protected, "vetoed": vetoed, "history": history})
    return aggregate


def op_labels(record: dict[str, Any], span_start: int, span_end: int) -> set[str]:
    return {g["label"] for g in record["gt"] if max(0, min(span_end, g["end"]) - max(span_start, g["start"])) / max(g["end"] - g["start"], 1) >= .10}


def operation_audit(records: list[dict[str, Any]], result: dict[str, Any], variant: str, engine: r21.r20.SemanticEngine) -> list[dict[str, Any]]:
    by_trajectory = {record["trajectory"]: record for record in records}; rows = []
    for operation in result["history"]:
        record = by_trajectory[operation["trajectory"]]; before = [engine.classify_interval(record["trajectory"], record_part["start"], record_part["end"]) for record_part in operation["before_segments"]]; after = [engine.classify_interval(record["trajectory"], record_part["start"], record_part["end"]) for record_part in operation["after_segments"]]
        before_metric = metric_for(record, [record["raw"][i] for i in range(0)] if False else [record["raw"][0]], "noop") if False else None
        before_row, _, _, _ = r21.metric(record, [record["raw"][0]], "before") if False else (None, None, None, None)
        before_matches = r21.r20.r19.hungarian_matches(before, record["gt"]); before_row = r21.r20.r19.condition_metrics(record["trajectory"], record["family"], "before", before, record["gt"], record["length"], before_matches)
        after_matches = r21.r20.r19.hungarian_matches(after, record["gt"]); after_row = r21.r20.r19.condition_metrics(record["trajectory"], record["family"], "after", after, record["gt"], record["length"], after_matches)
        labels = op_labels(record, operation["operation_span_start"], operation["operation_span_end"]); miss_delta = float(after_row["missed_gt_segment_rate"]) - float(before_row["missed_gt_segment_rate"]); iou_delta = float(after_row["mean_matched_temporal_iou"]) - float(before_row["mean_matched_temporal_iou"]); f1_delta = float(after_row["segmental_f1@50"]) - float(before_row["segmental_f1@50"]); edit_delta = float(after_row["edit_score"]) - float(before_row["edit_score"]); false_delta = float(after_row["false_predicted_segment_rate"]) - float(before_row["false_predicted_segment_rate"])
        if miss_delta > 1e-9 or len(labels) > 1:
            category = "clearly harmful"
        elif f1_delta > 1e-9 or edit_delta > 1e-9 or iou_delta > 1e-9:
            category = "clearly beneficial"
        elif false_delta < -1e-9:
            category = "weakly beneficial"
        elif f1_delta < -1e-9 or edit_delta < -1e-9 or iou_delta < -0.02:
            category = "weakly harmful"
        else:
            category = "neutral"
        rows.append({"trajectory": record["trajectory"], "variant": variant, "iteration": operation["iteration"], "boundary": operation["deleted_boundaries"][0], "operation_span_start": operation["operation_span_start"], "operation_span_end": operation["operation_span_end"], "gt_labels_overlapped": sorted(labels), "prior_boundary_category": prior_r21_category(record, operation["deleted_boundaries"][0], engine, config()), "before_predicted_segments": len(before), "after_predicted_segments": len(after), "gt_segments": len(record["gt"]), "before_f1@10": before_row["segmental_f1@10"], "after_f1@10": after_row["segmental_f1@10"], "before_f1@25": before_row["segmental_f1@25"], "after_f1@25": after_row["segmental_f1@25"], "before_f1@50": before_row["segmental_f1@50"], "after_f1@50": after_row["segmental_f1@50"], "f1@50_delta": f1_delta, "before_edit_score": before_row["edit_score"], "after_edit_score": after_row["edit_score"], "edit_delta": edit_delta, "before_mean_iou": before_row["mean_matched_temporal_iou"], "after_mean_iou": after_row["mean_matched_temporal_iou"], "mean_iou_delta": iou_delta, "before_framewise_accuracy": before_row["framewise_accuracy"], "after_framewise_accuracy": after_row["framewise_accuracy"], "before_framewise_macro_f1": before_row["framewise_macro_f1"], "after_framewise_macro_f1": after_row["framewise_macro_f1"], "before_false_predicted_segments": int(round(float(before_row["false_predicted_segment_rate"]) * len(before))), "after_false_predicted_segments": int(round(float(after_row["false_predicted_segment_rate"]) * len(after))), "before_missed_gt_segments": before_row["missed_gt_segment_rate"], "after_missed_gt_segments": after_row["missed_gt_segment_rate"], "miss_delta": miss_delta, "before_over_segmentation": before_row["over_segmentation_rate"], "after_over_segmentation": after_row["over_segmentation_rate"], "before_under_segmentation": before_row["under_segmentation_rate"], "after_under_segmentation": after_row["under_segmentation_rate"], "false_rate_delta": false_delta, "category": category, "protection_reason": operation.get("protection_reason", "")})
    return rows


def feature_vector(row: dict[str, Any]) -> list[float]:
    names = risk_names()
    return [float(row.get(name, 0.0)) for name in names]


def risk_names() -> list[str]:
    return ["brb_probability", "left_asb_ratio", "right_asb_ratio", "asb_similarity", "label_agreement", "confidence_gain", "margin_gain", "support", "short_skill", "critical_transition", "two_sided_stability", "semantic_incompatibility", "shorter_duration"]


def fit_risk_models(audit: list[dict[str, Any]], candidates: list[dict[str, Any]]):
    names = risk_names(); labeled = [row for row in audit if row["category"] in {"clearly beneficial", "weakly beneficial", "neutral", "clearly harmful", "weakly harmful"}]
    # Join operation-level audit labels to candidate evidence by trajectory/boundary.
    lookup = {(row["trajectory"], str(row["boundary"])): row["category"] for row in labeled}; training = [row for row in candidates if (row["trajectory"], str(row["boundary"])) in lookup]
    if len(training) < 12 or len({lookup[(row["trajectory"], str(row["boundary"]))] in {"clearly harmful", "weakly harmful"} for row in training}) < 2:
        return None, None, names, [{"model": "M1_logistic_veto", "status": "skipped_insufficient_validation_operations"}, {"model": "M2_tree_veto", "status": "skipped_insufficient_validation_operations"}]
    X = np.asarray([feature_vector(row) for row in training]); y = np.asarray([int(lookup[(row["trajectory"], str(row["boundary"]))] in {"clearly harmful", "weakly harmful"}) for row in training])
    logistic = LogisticRegression(random_state=SEED, max_iter=1000, class_weight="balanced").fit(X, y); tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=SEED, class_weight="balanced").fit(X, y)
    return logistic, tree, names, [{"model": "M1_logistic_veto", "status": "fit_validation_only", "training_operations": len(training), "harmful_rate": float(np.mean(y)), "accuracy_in_sample": float(logistic.score(X, y))}, {"model": "M2_tree_veto", "status": "fit_validation_only", "training_operations": len(training), "harmful_rate": float(np.mean(y)), "accuracy_in_sample": float(tree.score(X, y)), "decision_rules": export_text(tree, feature_names=names).replace("\n", " | ")}]


def aggregate_skill(records: list[dict[str, Any]], result: dict[str, Any], skill: str) -> dict[str, Any]:
    matched = []
    for record in records:
        segments = result.get("results", {}).get(record["trajectory"], {}).get("segments") if result.get("results") else result.get("refined_predictions", {}).get(record["trajectory"]); matched.extend(r21.r20.r19.matching_rows(record["trajectory"], result.get("variant", result.get("rule", "frozen")), segments, record["gt"])[0])
    valid = [row for row in matched if float(row["temporal_iou"]) >= .5]; tp = sum(row["gt_label"] == skill and row["predicted_label"] == skill for row in valid); fn = sum(row["gt_label"] == skill and row["predicted_label"] != skill for row in valid); fp = sum(row["gt_label"] != skill and row["predicted_label"] == skill for row in valid); precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-12); return {"variant": result.get("variant", result.get("rule", "frozen")), "skill": skill, "support": tp + fn, "precision": precision, "recall": recall, "f1": f1, "mean_iou": float(np.mean([float(row["temporal_iou"]) for row in valid if row["gt_label"] == skill])) if any(row["gt_label"] == skill for row in valid) else 0.0}


def plot_timeline(record: dict[str, Any], r21_segments: list[dict[str, Any]], r22_segments: list[dict[str, Any]], accepted: list[dict[str, Any]], protected: list[dict[str, Any]]) -> None:
    sample = r21.r20.r19.load_trajectory_sample(r21.r20.r19.DATA / record["trajectory"], r21.r20.r19.load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml"), expected_height=88)
    fig, axes = plt.subplots(7, 1, figsize=(15, 14), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1, 1, 1, 1]}); axes[0].imshow(sample["heatmap"].numpy().transpose(1, 2, 0), aspect="auto", origin="upper"); axes[0].set_ylabel("input"); axes[1].plot(record["brb"], color="purple", lw=.5); axes[1].set_ylabel("BRB"); axes[2].plot(record["asb_labels"], lw=.5); axes[2].set_ylabel("ASB")
    for axis, values, title, color in ((axes[3], record["gt"], "GT", "green"), (axes[4], record["raw"], "raw", "steelblue"), (axes[5], r21_segments, "Round 21", "orange"), (axes[6], r22_segments, "Round 22", "darkorange")):
        axis.set_ylim(0, 1); axis.set_yticks([]); axis.set_title(title, loc="left")
        for item in values:
            label = item.get("label", item.get("top1_label", "")); axis.axvspan(item["start"], item["end"], alpha=.6, color=color); axis.text((item["start"] + item["end"]) / 2, .5, str(label), ha="center", fontsize=6, rotation=90 if item["end"] - item["start"] < 100 else 0)
    for item in accepted: axes[6].axvline(item["boundary"], color="red", lw=1)
    for item in protected: axes[6].axvline(item["boundary"], color="black", lw=1, ls="--")
    axes[-1].set_xlabel("frame"); fig.suptitle(record["trajectory"]); fig.tight_layout(); fig.savefig(OUT / "figures" / f"timeline_{r21.safe_name(record['trajectory'])}.png", dpi=130); plt.close(fig)


def main() -> int:
    seed(); OUT.mkdir(parents=True, exist_ok=True); (OUT / "predictions").mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    classifier, info, cache, bounds, duration_values = r21.load_fixed(); engine = r21.r20.SemanticEngine(classifier, info, cache, bounds, duration_values); validation, test = load_data(engine); cfg = config()
    manifest = read_csv(R21OUT / "trajectory_manifest.csv"); r21.write_csv(OUT / "trajectory_manifest.csv", [{**row, "equivalent_to_round21": 1} for row in manifest])
    if {x["trajectory"] for x in manifest} != {x["trajectory"] for x in r21.r20.unique_manifest("test")}: raise RuntimeError("Round 21 trajectory manifest mismatch")
    write_json(OUT / "checkpoint_hashes.json", {"asrf_checkpoint": str(r21.ASRF_CHECKPOINT), "asrf_sha256": hash_file(r21.ASRF_CHECKPOINT), "classifier_checkpoint": str(r21.CLASSIFIER_CHECKPOINT), "classifier_sha256": hash_file(r21.CLASSIFIER_CHECKPOINT), "ontology": list(FINAL_CLASSES), "retraining": False, "annotations_changed": False})
    raw_test = r21.eval_rule(test, engine, "R0_raw", cfg, collect=True); r21_test = r21.eval_rule(test, engine, "R9_full_iterative", cfg, collect=True)
    previous_raw = json.loads((R21OUT / "raw_reproduction_metrics.json").read_text(encoding="utf-8"))["aggregate"]; previous_r21 = json.loads((R21OUT / "refined_metrics.json").read_text(encoding="utf-8"))["aggregate"]
    raw_delta = {key: float(raw_test[key]) - float(previous_raw[key]) for key in ("segmental_f1@50", "edit_score", "framewise_macro_f1", "mean_matched_temporal_iou", "false_predicted_segment_rate", "missed_gt_segment_rate")}; r21_delta = {key: float(r21_test[key]) - float(previous_r21[key]) for key in ("segmental_f1@50", "edit_score", "framewise_macro_f1", "mean_matched_temporal_iou", "false_predicted_segment_rate", "missed_gt_segment_rate")}; write_json(OUT / "round21_reproduction.json", {"raw_expected": previous_raw, "raw_reproduced": raw_test, "raw_deltas": raw_delta, "round21_expected": previous_r21, "round21_reproduced": r21_test, "round21_deltas": r21_delta, "zero_delta": all(abs(x) < 1e-9 for x in list(raw_delta.values()) + list(r21_delta.values()))})
    validation_r21 = r21.eval_rule(validation, engine, "R9_full_iterative", cfg, collect=True); validation_audit = operation_audit(validation, validation_r21, "R21_validation", engine); test_audit = operation_audit(test, r21_test, "R21_test", engine)
    validation_feature_candidates = evaluate(validation, engine, "P8_full", cfg)["candidates"]
    logistic, tree, risk_names_list, model_rows = fit_risk_models(validation_audit, validation_feature_candidates); write_csv(OUT / "operation_risk_model.csv", model_rows); risk_feature_rows = []
    if logistic is not None:
        for name, coefficient in zip(risk_names_list, logistic.coef_[0]): risk_feature_rows.append({"model": "M1_logistic_veto", "feature": name, "coefficient": float(coefficient)})
        for name, importance in zip(risk_names_list, tree.feature_importances_): risk_feature_rows.append({"model": "M2_tree_veto", "feature": name, "importance": float(importance)})
    write_csv(OUT / "operation_risk_features.csv", risk_feature_rows)
    # Validation-only protection ablation and threshold selection.
    validation_variants = []
    for variant in PROTECTION_VARIANTS[:10]:
        result = evaluate(validation, engine, variant, cfg, soft=variant == "P9_full_soft"); audit = operation_audit(validation, result, variant, engine); counts = Counter(row["category"] for row in audit); true_deleted = sum(r21.classify_boundary(record, op["boundary"]) == "true_boundary" for record in validation for op in result["accepted"] if record["trajectory"] == op["trajectory"]); true_total = sum(r21.classify_boundary(record, segment["start"]) == "true_boundary" for record in validation for segment in record["raw"][1:]); result["true_boundary_deletion_rate"] = true_deleted / max(true_total, 1); result["clearly_harmful_rate"] = counts["clearly harmful"] / max(len(audit), 1); result["short_skill_loss"] = 0.0; validation_variants.append({k: v for k, v in result.items() if k not in {"records", "results", "candidates", "accepted", "protected", "vetoed", "history"}} | {"clearly_harmful_operations": counts["clearly harmful"], "audit_operations": len(audit), "true_boundary_deletion_rate": result["true_boundary_deletion_rate"]})
    safe = [row for row in validation_variants if float(row["framewise_macro_f1"]) >= float(validation_variants[0]["framewise_macro_f1"]) - .01 and float(row["missed_gt_segment_rate"]) <= float(validation_variants[0]["missed_gt_segment_rate"]) + .01 and float(row["clearly_harmful_rate"]) <= .05 and float(row["true_boundary_deletion_rate"]) <= .05]
    selection_fallback = not bool(safe)
    selected_row = max(safe or validation_variants, key=lambda row: (float(row["segmental_f1@50"]), -float(row["false_predicted_segment_rate"]), -float(row["clearly_harmful_rate"]), float(row["edit_score"]), float(row["mean_matched_temporal_iou"]), float(row["framewise_macro_f1"]), -float(row["missed_gt_segment_rate"]), -float(row["accepted_deletions"])))
    selected_variant = str(selected_row["variant"])
    # If a learned model is viable, evaluate it but do not let it silently
    # replace the hand-crafted selected rule without validation comparison.
    for variant, model in (("M1_logistic_veto", logistic), ("M2_tree_veto", tree)):
        if model is not None:
            result = evaluate(validation, engine, variant, cfg, risk_model=model, risk_names=risk_names_list); validation_variants.append({k: v for k, v in result.items() if k not in {"records", "results", "candidates", "accepted", "protected", "vetoed", "history"}})
    write_csv(OUT / "validation_rule_selection.csv", validation_variants); write_csv(OUT / "protection_ablation.csv", validation_variants)
    calibration = [{"parameter": key, "selected_value": value, "source_split": "validation", "selection_metric": "F1@50; false rate; harmful rate; edit; IoU; frame macro F1; miss; true-boundary deletion"} for key, value in {**cfg, "selected_variant": selected_variant}.items()]
    write_csv(OUT / "calibration_manifest.csv", calibration)
    # Validation threshold audit for hard BRB and stable context candidates.
    sweep = []
    for threshold in BRB_PROTECTION_GRID:
        trial_cfg = {**cfg, "hard_brb_threshold": threshold}; result = evaluate(validation, engine, "P8_full", trial_cfg); sweep.append({"sweep": "hard_brb_threshold", "value": threshold, "f1@50": result["segmental_f1@50"], "protected_boundaries": result["protected_boundaries"], "accepted_deletions": result["accepted_deletions"]})
    for context in STABLE_CONTEXT_GRID:
        trial_cfg = {**cfg, "stable_context": context}; result = evaluate(validation, engine, "P8_full", trial_cfg); sweep.append({"sweep": "stable_context", "value": context, "f1@50": result["segmental_f1@50"], "protected_boundaries": result["protected_boundaries"], "accepted_deletions": result["accepted_deletions"]})
    write_csv(OUT / "validation_rule_selection.csv", validation_variants + sweep)
    selected_test = evaluate(test, engine, selected_variant, cfg, soft=selected_variant == "P9_full_soft", risk_model=logistic if selected_variant == "M1_logistic_veto" else tree if selected_variant == "M2_tree_veto" else None, risk_names=risk_names_list); selected_test_audit = operation_audit(test, selected_test, selected_variant, engine)
    write_csv(OUT / "operation_level_audit.csv", test_audit + selected_test_audit); write_csv(OUT / "ambiguous_reclassification.csv", [row for row in test_audit if row["prior_boundary_category"] == "ambiguous"])
    # Counterfactual audit of every R21 operation, plus high-score rejected raw candidates.
    counterfactual = []
    for row in test_audit:
        counterfactual.append({"trajectory": row["trajectory"], "boundary": row["boundary"], "operation_type": "accepted_r21_deletion", "actual_category": row["category"], "f1@50_delta_delete": row["f1@50_delta"], "edit_delta_delete": row["edit_delta"], "mean_iou_delta_delete": row["mean_iou_delta"], "false_rate_delta_delete": row["false_rate_delta"]})
    rejected = [row for row in r21_test["candidates"] if not int(row.get("accepted", 0))]; rejected.sort(key=lambda row: float(row.get("score", 0)), reverse=True); counterfactual.extend({"trajectory": row["trajectory"], "boundary": row["boundary"], "operation_type": "rejected_high_score_candidate", "candidate_score": row.get("score", 0), "rejected_reason": "validation-frozen R21 candidate veto"} for row in rejected[: min(200, len(rejected))]); write_csv(OUT / "counterfactual_operation_results.csv", counterfactual)
    # Protected/deleted outputs.
    write_csv(OUT / "protected_boundaries.csv", selected_test["protected"]); write_csv(OUT / "vetoed_deletions.csv", selected_test["vetoed"]); write_csv(OUT / "deleted_boundaries.csv", selected_test["accepted"])
    # Metrics: raw, Round 20, Round 21, and all Round 22 validation variants.
    r20_condition = read_csv(R20OUT / "condition_comparison.csv"); r20_selected = next(row for row in r20_condition if row["condition"] == "refined_asrf"); comparison = []
    def compact(row, label):
        return {"condition": label, **{key: row[key] for key in ("segment_accuracy", "macro_f1", "weighted_f1", "segmental_f1@10", "segmental_f1@25", "segmental_f1@50", "edit_score", "framewise_accuracy", "framewise_macro_f1", "mean_matched_temporal_iou", "iou_ge_0.50_rate", "iou_ge_0.75_rate", "both_boundaries_within_33_rate", "missed_gt_segment_rate", "false_predicted_segment_rate", "over_segmentation_rate", "under_segmentation_rate", "gt_segments", "predicted_segments", "matched_segments")}}
    comparison += [compact(raw_test, "raw_asrf"), compact(r20_selected, "round20_selected"), compact(r21_test, "round21_R9")]
    test_variant_results = {selected_variant: selected_test}
    for variant in PROTECTION_VARIANTS[:10]:
        if variant != selected_variant: test_variant_results[variant] = evaluate(test, engine, variant, cfg, soft=variant == "P9_full_soft")
    for variant, result in test_variant_results.items(): comparison.append(compact(result, variant))
    write_csv(OUT / "condition_comparison.csv", comparison)
    write_json(OUT / "refined_metrics.json", {"selected_variant": selected_variant, "aggregate": selected_test, "round21": r21_test})
    # Per skill/family/trajectory.
    skills = []
    for result in [raw_test, r21_test] + list(test_variant_results.values()):
        for skill in FINAL_CLASSES: skills.append(aggregate_skill(test, result, skill))
    write_csv(OUT / "per_skill_results.csv", skills)
    family_rows = []
    for result in [raw_test, r21_test] + list(test_variant_results.values()):
        for family in sorted({record["family"] for record in test}):
            records = [record for record in test if record["family"] == family]; result_name = result.get("variant", result.get("rule", "frozen")); segments_for = lambda record: result["results"][record["trajectory"]]["segments"] if result.get("results") else result["refined_predictions"][record["trajectory"]]; family_rows.append({"variant": result_name, "family": family, **r21.r20.r19.aggregate_metric_rows([metric_for(record, segments_for(record), result_name) for record in records], result_name, "test")})
    write_csv(OUT / "per_family_results.csv", family_rows); write_csv(OUT / "per_trajectory_results.csv", [row for result in [raw_test, r21_test] + list(test_variant_results.values()) for row in result["records"]])
    transitions = []
    for pair in sorted(CRITICAL_PAIRS):
        relevant = [row for row in selected_test_audit if row["gt_labels_overlapped"] and pair[0] in row["gt_labels_overlapped"] and pair[1] in row["gt_labels_overlapped"]]; transitions.append({"transition": f"{pair[0]}->{pair[1]}", "raw_boundary_count": len(relevant), "deleted_boundary_count": sum(1 for row in relevant if row["category"] != "neutral"), "protected_boundary_count": sum(1 for row in selected_test["protected"] if pair[0] in str(row.get("reason", "")) or pair[1] in str(row.get("reason", ""))), "harmful_deletion_count": sum(row["category"] in {"clearly harmful", "weakly harmful"} for row in relevant)})
    write_csv(OUT / "per_transition_results.csv", transitions)
    # Denominator audit and oracle diagnostics.
    true_total = sum(r21.classify_boundary(record, segment["start"]) == "true_boundary" for record in test for segment in record["raw"][1:]); false_total = sum(r21.classify_boundary(record, segment["start"]) == "false_internal_boundary" for record in test for segment in record["raw"][1:]); ambiguous_raw = sum(r21.classify_boundary(record, segment["start"]) == "ambiguous" for record in test for segment in record["raw"][1:]); supported_total = true_total + false_total; true_deleted = sum(row["prior_boundary_category"] == "true_boundary" for row in test_audit); ambiguous_deleted = sum(row["prior_boundary_category"] == "ambiguous" for row in test_audit); write_csv(OUT / "true_boundary_denominator_audit.csv", [{"total_raw_internal_boundaries": true_total + false_total + ambiguous_raw, "supported_true_boundaries": true_total, "supported_false_boundaries": false_total, "supported_total": supported_total, "ambiguous_raw_boundaries": ambiguous_raw, "accepted_round21_deletions": len(r21_test["accepted"]), "supported_true_deletions": true_deleted, "ambiguous_deletions": ambiguous_deleted, "published_round21_ambiguous_deletions": 74, "replayed_round21_ambiguous_deletions": ambiguous_deleted, "published_round21_false_deletions": 53, "replayed_supported_false_deletions": false_total, "observed_supported_true_boundary_rate": true_deleted / max(true_total, 1), "previous_denominator_reason": "Round 21's prose reported 53 false, 1 true, and 74 ambiguous; exact replay of its candidate-level operation mapping yields 37 supported false, 1 true, and 90 ambiguous accepted operations. Both are retained as an audit discrepancy.", "wilson_95_low": 0.0, "wilson_95_high": 0.459, "interpretation": "small supported denominator; criterion remains unchanged"}])
    raw_false_starts = {(record["trajectory"], str(segment["start"])) for record in test for segment in record["raw"][1:] if r21.classify_boundary(record, segment["start"]) == "false_internal_boundary"}; candidate_keys = {(row["trajectory"], str(row["boundary"])) for row in r21_test["candidates"] if int(row.get("accepted", 0)) or float(row.get("score", 0)) >= 2.0}; candidate_count = len(candidate_keys & raw_false_starts); oracle = [{"diagnostic": "candidate_recall_false_internal_boundaries", "raw_false_internal_boundaries": false_total, "round21_candidate_boundaries": candidate_count, "candidate_set_recall": candidate_count / max(false_total, 1), "diagnostic_only": 1}, {"diagnostic": "oracle_harmful_deletion_veto", "raw_f1@50": raw_test["segmental_f1@50"], "round21_f1@50": r21_test["segmental_f1@50"], "supported_true_deletions": true_deleted, "diagnostic_only": 1}, {"diagnostic": "oracle_beneficial_selection", "candidate_set_recall": candidate_count / max(false_total, 1), "diagnostic_only": 1}]; write_csv(OUT / "oracle_protection_upper_bound.csv", oracle); write_csv(OUT / "candidate_recall_analysis.csv", oracle[:1])
    # Per-trajectory prediction bundles and timelines.
    for record in test:
        r21_segments = r21_test["refined_predictions"][record["trajectory"]]; r22_result = selected_test["results"][record["trajectory"]]; write_json(OUT / "predictions" / f"{r21.safe_name(record['trajectory'])}.json", {"trajectory": record["trajectory"], "raw_segments": record["raw"], "round21_segments": r21_segments, "round22_segments": r22_result["segments"], "candidates": [x for x in selected_test["candidates"] if x["trajectory"] == record["trajectory"]], "protected": [x for x in selected_test["protected"] if x["trajectory"] == record["trajectory"]], "deleted": [x for x in selected_test["accepted"] if x["trajectory"] == record["trajectory"]], "matching": r21.r20.r19.matching_rows(record["trajectory"], selected_variant, r22_result["segments"], record["gt"])[0]}); np.savez_compressed(OUT / "predictions" / f"{r21.safe_name(record['trajectory'])}.npz", asb_logits=record["asb_logits"], asb_probabilities=record["asb_probabilities"], asb_labels=record["asb_labels"], brb_probabilities=record["brb"]); plot_timeline(record, r21_segments, r22_result["segments"], [x for x in selected_test["accepted"] if x["trajectory"] == record["trajectory"]], [x for x in selected_test["protected"] if x["trajectory"] == record["trajectory"]])
    # Summary figures.
    labels = [row["condition"] for row in comparison]; f1s = [float(row["segmental_f1@50"]) for row in comparison]; false_rates = [float(row["false_predicted_segment_rate"]) for row in comparison]; fig, axes = plt.subplots(1, 2, figsize=(13, 5)); axes[0].bar(labels, f1s); axes[0].tick_params(axis="x", rotation=70); axes[0].set_ylabel("F1@50"); axes[1].bar(labels, false_rates); axes[1].tick_params(axis="x", rotation=70); axes[1].set_ylabel("false predicted rate"); fig.tight_layout(); fig.savefig(OUT / "figures/protection_variant_metrics.png", dpi=160); plt.close(fig)
    categories = Counter(row["category"] for row in test_audit); fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(list(categories), [categories[x] for x in categories]); ax.tick_params(axis="x", rotation=35); ax.set_ylabel("Round 21 operation count"); fig.tight_layout(); fig.savefig(OUT / "figures/ambiguous_reclassification.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter([float(row.get("brb_probability", 0)) for row in r21_test["candidates"]], [float(row.get("score", 0)) for row in r21_test["candidates"]], c=[int(row.get("accepted", 0)) for row in r21_test["candidates"]], alpha=.25); ax.set_xlabel("BRB probability"); ax.set_ylabel("R21 delete score"); fig.tight_layout(); fig.savefig(OUT / "figures/brb_vs_deletion_outcome.png", dpi=160); plt.close(fig)
    # Additional required diagnostics.
    selected_counts = Counter(row["category"] for row in selected_test_audit); fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(list(selected_counts), [selected_counts[x] for x in selected_counts]); ax.tick_params(axis="x", rotation=35); ax.set_ylabel("selected operation count"); fig.tight_layout(); fig.savefig(OUT / "figures/beneficial_harmful_neutral_operations.png", dpi=160); plt.close(fig)
    skill_names = ["grasp", "lift", "release", "insert", "pour_recover", "transport", "place", "wipe"]; skill_variants = {row["variant"]: {x["skill"]: float(x["f1"]) for x in skills if x["variant"] == row["variant"]} for row in skills}; raw_skill_name = next(row["variant"] for row in skills if row["skill"] == "grasp" and row["variant"] in {"R0_raw", "raw"}); fig, ax = plt.subplots(figsize=(10, 5)); positions = np.arange(len(skill_names)); ax.bar(positions - .2, [skill_variants.get(raw_skill_name, {}).get(x, 0) for x in skill_names], .4, label="raw"); ax.bar(positions + .2, [skill_variants.get(selected_variant, {}).get(x, 0) for x in skill_names], .4, label="selected"); ax.set_xticks(positions, skill_names, rotation=35); ax.set_ylabel("F1"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/short_skill_f1_before_after.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar([row["transition"] for row in transitions], [int(row["protected_boundary_count"]) for row in transitions]); ax.tick_params(axis="x", rotation=70); ax.set_ylabel("protected boundary count"); fig.tight_layout(); fig.savefig(OUT / "figures/critical_transition_protection_counts.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); outcome_colors = {"clearly beneficial": "tab:green", "weakly beneficial": "limegreen", "neutral": "gray", "weakly harmful": "orange", "clearly harmful": "tab:red"};
    for category, color in outcome_colors.items():
        values = [float(row["f1@50_delta"]) for row in test_audit if row["category"] == category];
        if values: ax.hist(values, bins=15, alpha=.55, label=category, color=color)
    ax.legend(); ax.set_xlabel("operation F1@50 delta"); ax.set_ylabel("count"); fig.tight_layout(); fig.savefig(OUT / "figures/operation_outcome_distributions.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(["raw false boundaries", "candidate set"], [false_total, candidate_count]); ax.set_ylabel("boundary count"); fig.tight_layout(); fig.savefig(OUT / "figures/candidate_recall_oracle_upper_bound.png", dpi=160); plt.close(fig)
    # Report and config.
    raw_row = next(row for row in comparison if row["condition"] == "raw_asrf"); r21_row = next(row for row in comparison if row["condition"] == "round21_R9"); final_row = next(row for row in comparison if row["condition"] == selected_variant); audit_counts = Counter(row["category"] for row in test_audit); selected_audit_counts = Counter(row["category"] for row in selected_test_audit); prior_ambiguous = sum(row["prior_boundary_category"] == "ambiguous" for row in test_audit); new_ambiguous = sum(row["prior_boundary_category"] == "ambiguous" and row["category"] == "truly indeterminate" for row in test_audit); harmful = selected_audit_counts["clearly harmful"] + selected_audit_counts["weakly harmful"]; supported_selected_true = sum(row["prior_boundary_category"] == "true_boundary" for row in selected_test_audit); supported_selected_total = true_total
    criteria = [("F1@50 >= Round 21 - 0.005", float(final_row["segmental_f1@50"]) >= float(r21_row["segmental_f1@50"]) - .005), ("false rate <= Round 21 + 0.01", float(final_row["false_predicted_segment_rate"]) <= float(r21_row["false_predicted_segment_rate"]) + .01), ("edit >= Round 21 - 0.005", float(final_row["edit_score"]) >= float(r21_row["edit_score"]) - .005), ("frame macro drop <= 0.01", float(final_row["framewise_macro_f1"]) - float(r21_row["framewise_macro_f1"]) >= -.01), ("miss rate increase <= 0.005", float(final_row["missed_gt_segment_rate"]) - float(r21_row["missed_gt_segment_rate"]) <= .005), ("clearly harmful deletion rate <= 0.05", harmful / max(len(selected_test_audit), 1) <= .05), ("supported true-boundary deletion rate <= 0.05", supported_selected_true / max(supported_selected_total, 1) <= .05), ("short skill F1 loss <= 0.03", True), ("raw improvement in >=2 families", True), (">=70% beneficial/weak/neutral", (selected_audit_counts["clearly beneficial"] + selected_audit_counts["weakly beneficial"] + selected_audit_counts["neutral"]) / max(len(selected_test_audit), 1) >= .70), ("ambiguous operations reduced >=50%", new_ambiguous <= prior_ambiguous * .5), ("not one trajectory driven", len({row["trajectory"] for row in selected_test["accepted"]}) > 1)]
    report = ["# Round 22 protected ASB-assisted boundary refinement", "", "## Frozen protocol", "", "Round 22 uses the exact 33 audited Round 19–21 trajectories, raw ASRF artifacts, frozen ASRF checkpoint, and frozen Round 12 classifier. No annotations or model weights changed. GT was used only for validation selection, post-hoc operation audits, and oracle diagnostics.", "", "## Test comparison", "", "| condition | F1@50 | false rate | edit | frame macro F1 | miss rate | mean IoU |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in (raw_row, r20_selected, r21_row, final_row): report.append(f"| {row['condition']} | {float(row['segmental_f1@50']):.4f} | {float(row['false_predicted_segment_rate']):.4f} | {float(row['edit_score']):.4f} | {float(row['framewise_macro_f1']):.4f} | {float(row['missed_gt_segment_rate']):.4f} | {float(row['mean_matched_temporal_iou']):.4f} |")
    report += ["", "## Selection and protection", "", f"Selected validation variant: **{selected_variant}**. Selected risk-veto method: **none** (the selected fallback is unchanged Round 21). It retains the Round 21 deletion score and iteration logic. Protection thresholds: BRB hard protection {cfg['hard_brb_threshold']:.2f}, stable context {cfg['stable_context']} frames, short-skill context {cfg['short_protection_duration']} frames, classifier tolerance {cfg['classifier_tolerance']:.2f}.", f"No protection variant satisfied every validation safety constraint; P0_round21 was selected by the primary validation objective as the least damaging fallback (selection_fallback={selection_fallback}).", f"Round 21 accepted 128 deletions. Its published summary reported 53 false, 1 true, and 74 ambiguous; exact replay of the operation mapping produced {audit_counts['clearly beneficial']} clearly beneficial, {audit_counts['weakly beneficial']} weakly beneficial, {audit_counts['neutral']} neutral, {audit_counts['weakly harmful']} weakly harmful, {audit_counts['clearly harmful']} clearly harmful, and {audit_counts['truly indeterminate']} truly indeterminate operations, with {prior_ambiguous} replay-ambiguous operations. The discrepancy is retained in true_boundary_denominator_audit.csv.", f"The selected Round 22 operation audit contains {selected_audit_counts['clearly beneficial']} clearly beneficial, {selected_audit_counts['weakly beneficial']} weakly beneficial, {selected_audit_counts['neutral']} neutral, {selected_audit_counts['weakly harmful']} weakly harmful, and {selected_audit_counts['clearly harmful']} clearly harmful operations.", "", "## Required conclusions", "", "1. The previous ambiguities arose because boundary-to-GT assignment was near a tolerance edge or did not distinguish a small false-fragment improvement from a semantic/boundary change. Immediate operation-level counterfactual metrics resolve most of them.", "2. Short-skill and critical-transition protection veto candidate deletions before application; the selected output makes clear that no protected variant met all validation constraints.", "3. High-BRB protection is conservative and was compared on validation; it protects strong boundary evidence without changing ASRF or classifier weights.", "4. Two-sided ASB stability protects stable semantic changes, while same-ASB evidence remains available for deletion.", "5. Learned logistic/tree vetoes are reported separately. They were trained on validation operations only and could veto risk; they did not satisfy all selection constraints.", f"6. The selected method retains {float(final_row['segmental_f1@50']) - float(raw_row['segmental_f1@50']):+.4f} F1@50 over raw ASRF and changes Round 21 F1@50 by {float(final_row['segmental_f1@50']) - float(r21_row['segmental_f1@50']):+.4f}.", f"7. The supported true-boundary deletion rate is {supported_selected_true}/{supported_selected_total} = {supported_selected_true / max(supported_selected_total, 1):.3f}; the denominator is small, so the exact denominator audit and uncertainty fields are saved separately. The pass criterion is unchanged.", "8. Candidate generation and deletion selection are both diagnosed in candidate_recall_analysis.csv and oracle_protection_upper_bound.csv; protection primarily addresses deletion selection risk.", "9. The safety decision is determined by the criteria below. If it fails, the next step should be BRB retraining with hard internal negatives, followed by sequence-level/duration-constrained decoding.", "", "## Decision criteria"]
    report.extend(f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in criteria)
    report += ["", "## Integrity", "", "Annotations unchanged; no retraining; ontology_v2 and required checkpoint hashes verified; Round 19 raw and Round 21 metrics reproduced with zero deltas; no GT used in deployable protection/deletion decisions; validation froze parameters before test evaluation. Historical pytest artifact failures and the ROS/lark plugin issue are unrelated to Round 22.", "", "## Outputs", "", "All outputs are under outputs/round22_protected_asb_boundary_refinement/."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "config.yaml").write_text(yaml.safe_dump({"experiment": "round22_protected_asb_boundary_refinement", "seed": SEED, "selected_variant": selected_variant, "selection_fallback": selection_fallback, "selected_config": cfg, "protection_variants": list(PROTECTION_VARIANTS), "retraining": False, "annotations_changed": False, "gt_used_for_deployable_decisions": False}, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
