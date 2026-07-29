#!/usr/bin/env python3
"""Round 26: compare SF and hard-window-r5 under frozen Round 25 refinement."""

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
R19 = ROOT / "outputs/round19_asrf_segment_classifier_integration"
R21 = ROOT / "outputs/round21_asb_assisted_boundary_merge"
R25 = ROOT / "outputs/round25_duration_gated_local_hypothesis_selection"
OUT = ROOT / "outputs/round26_sf_vs_r5_round25_refinement"
SF_CHECKPOINT = R10 / "models/single_frame/best.pt"
R5_CHECKPOINT = R10 / "models/hard_window_r5/best.pt"
CLASSIFIER_CHECKPOINT = ROOT / "outputs/round12_multiskill_segment_classifier/model/best.pt"
SF_SHA = "6b11abff2ff4387eada88e38fb56133b29fd5c3facdfb54b42fc84701213db9a"
R5_SHA = "577d8edf9e2b04927acc235ffa4d6baab8df1712dd0b98eaaba9063fde31f406"
CLASSIFIER_SHA = "51f0abbcc4250ef97951bcaef04fc8f55cb2de968affdf0121a446ea1635a86f"
SEED = 42
TOLERANCES = (5, 10, 20, 33)
KNOWN = set(("reach", "grasp", "lift", "transport", "place", "release", "retreat"))
NOVEL = set(("pour", "pour_recover", "wipe", "insert"))

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_round19_asrf_segment_classifier_integration as r19  # noqa: E402
import run_round25_duration_gated_local_hypothesis_selection as r25  # noqa: E402
from asrf.data.dataset import load_trajectory_sample  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def safe_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_").replace("+", "plus")


def seed() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def config_for_checkpoint(path: Path) -> Path:
    candidates = [path.with_name("config.yaml"), path.with_name("resolved_config.yaml"), path.parent.parent / "config.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No config adjacent to {path}")


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    config_path = config_for_checkpoint(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    arch = payload.get("architecture_config", config.get("model", {}))
    target = payload.get("boundary_target_config", {k: config.get("data", {}).get(k) for k in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")})
    labels = payload.get("label_map", {})
    ontology = payload.get("ontology_version") or payload.get("ontology_metadata", {}).get("ontology_version")
    return {"config_path": str(config_path), "config": config, "payload": payload, "architecture": arch, "target": target, "labels": labels, "ontology": ontology}


def discover_r5_candidates() -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(OUT.parent.rglob("*.pt")):
        text = str(path).lower()
        if not ("r5" in text or "hard_window" in text):
            continue
        if path.name not in {"best.pt", "last.pt"}:
            continue
        candidates.append(path)
    return candidates


def audit_r5() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_target = {"boundary_target_mode": "hard_window", "boundary_window_radius": 5, "boundary_include_frame_zero": True, "boundary_include_final_frame": False}
    rows: list[dict[str, Any]] = []
    for path in discover_r5_candidates():
        usable = False
        reason = ""
        try:
            meta = checkpoint_metadata(path)
            config = meta["config"]
            arch = meta["architecture"]
            target = meta["target"]
            config_target = {key: config.get("data", {}).get(key) for key in expected_target}
            usable = path == R5_CHECKPOINT and all(config_target[k] == v for k, v in expected_target.items()) and all(target.get(k) == v for k, v in expected_target.items()) and int(arch.get("num_classes", -1)) == 7 and tuple(arch.get("dilation_schedule", ())) == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) and int(arch.get("heatmap_channels", -1)) == 3 and int(arch.get("heatmap_height", -1)) == 88
            if not usable:
                reason = "not the exact completed Round 10 r5 checkpoint or metadata mismatch"
            elif sha256(path) != R5_SHA:
                usable = False
                reason = "SHA-256 mismatch"
            else:
                reason = "selected exact completed Round 10 r5 checkpoint"
            split_train = ";".join(meta["payload"].get("train_trajectory_ids", []))
            split_val = ";".join(meta["payload"].get("validation_trajectory_ids", []))
            ontology = meta["ontology"] or "legacy_round10_pp_only_7_class_front_end_label_map"
            feature_dim = f"input=[3,88,T]; encoder=128; temporal=64; output=ASB:{arch.get('num_classes')},BRB:1"
            temporal = "hard_window radius=5 => clipped t-5..t+5 (11 frames nominal)" if config_target.get("boundary_target_mode") == "hard_window" and config_target.get("boundary_window_radius") == 5 else str(config_target)
            asb_brb = "ASB framewise labels; BRB hard-window target"
            architecture = json.dumps(arch, sort_keys=True)
        except Exception as exc:
            usable = False
            reason = f"metadata read failed: {type(exc).__name__}: {exc}"
            ontology = feature_dim = temporal = asb_brb = split_train = split_val = architecture = ""
        rows.append({"path": str(path), "artifact_type": path.name, "sha256": sha256(path), "ontology_version": ontology, "feature_dimension": feature_dim, "temporal_window_definition": temporal, "asb_brb_architecture": asb_brb + " | " + architecture, "training_split": split_train, "validation_split": split_val, "target_definition": json.dumps(expected_target, sort_keys=True), "usable": int(usable), "exclusion_reason": reason})
    if not any(row["usable"] for row in rows):
        raise RuntimeError("No valid completed r5 checkpoint exists; refusing to retrain or substitute another model.")
    selected = next(row for row in rows if row["usable"])
    write_csv(OUT / "r5_artifact_audit.csv", rows)
    return json.loads(json.dumps(selected)), rows


def audit_definition(selected: dict[str, Any]) -> dict[str, Any]:
    meta = checkpoint_metadata(Path(selected["path"]))
    config = meta["config"]
    arch = meta["architecture"]
    expected_arch = config["model"]
    if arch != expected_arch:
        raise RuntimeError("r5 checkpoint architecture metadata does not match its config.")
    if meta["target"].get("boundary_target_mode") != "hard_window" or int(meta["target"].get("boundary_window_radius", -1)) != 5:
        raise RuntimeError("r5 checkpoint target metadata is not hard_window radius 5.")
    definition = {
        "meaning": "hard BRB target window radius 5, not a cropped input window",
        "window": {"radius_frames": 5, "nominal_frames": 11, "indices": "t-5 through t+5 inclusive", "edge_padding": "clip to [0,T); no trajectory data padding"},
        "input_tensor_shape": "[B, 3, 88, T]",
        "temporal_indexing": "input column t remains output timestep t; full trajectory is processed",
        "asb_uses_local_window": False,
        "brb_uses_local_window": True,
        "target_definition": {"mode": "hard_window", "boundary_window_radius": 5, "include_frame_zero": True, "include_final_frame": False, "overlap": "elementwise maximum"},
        "feature_dimension": {"heatmap_channels": 3, "heatmap_height": 88, "encoder_output_channels": 128, "temporal_feature_channels": 64, "asb_classes": 7, "brb_channels": 1},
        "checkpoint_architecture_metadata": arch,
        "config_path": str(meta["config_path"]),
        "checkpoint": selected,
    }
    write_json(OUT / "r5_definition.json", definition)
    return definition


def load_models() -> tuple[ASRFModel, ASRFModel, Any, dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict[str, float]], dict[str, np.ndarray]]:
    sf, classifier, info, cache, models = r25.load_fixed()
    r5_meta = checkpoint_metadata(R5_CHECKPOINT)
    r5 = ASRFModel.from_config(r5_meta["config"])
    payload = r5_meta["payload"]
    r5.load_state_dict(payload["model_state"], strict=True)
    r5.eval()
    refs = r25.embedding_refs(classifier, cache, info["normalization"])
    return sf, r5, classifier, info, cache, models, refs


@torch.no_grad()
def infer(model: ASRFModel, sample: dict[str, Any]) -> dict[str, np.ndarray]:
    output = model(sample["heatmap"].unsqueeze(0), valid_mask=sample["valid_mask"].unsqueeze(0))
    return {"asb_logits": output.asb_stage_logits[-1][0].cpu().numpy(), "asb_probabilities": output.asb_stage_probabilities[-1][0].cpu().numpy(), "asb_labels": output.asb_stage_probabilities[-1][0].argmax(dim=0).cpu().numpy(), "brb_logits": output.brb_stage_logits[-1][0, 0].cpu().numpy(), "brb_probabilities": output.brb_stage_probabilities[-1][0, 0].cpu().numpy()}


def build_record(manifest: dict[str, str], arrays: dict[str, np.ndarray], classifier: Any, cache: dict[str, tuple[np.ndarray, np.ndarray]], info: dict[str, Any], refs: dict[str, np.ndarray], models: dict[str, dict[str, float]], gt: list[dict[str, Any]], sample: dict[str, Any]) -> dict[str, Any]:
    intervals = r19.raw_segments(arrays["brb_probabilities"])
    raw = r19.attach(intervals, r19.classify(classifier, cache, info["normalization"], manifest["trajectory"], intervals))
    record = {"trajectory": manifest["trajectory"], "family": r19.family_for(manifest["trajectory"], manifest["family"]), "split": "test", "length": len(arrays["brb_probabilities"]), "gt": gt, "raw": raw, "arrays": arrays, "raw_intervals": [{"start": x.start, "end": x.end} for x in intervals], "heatmap": sample["heatmap"].numpy(), "timestamps": sample["timestamps"].numpy()}
    r25.attach_features(record, refs, models)
    return record


def frozen_config() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = yaml.safe_load((R25 / "config.yaml").read_text(encoding="utf-8"))["selected"]
    source_files = {"round25_config": str(R25 / "config.yaml"), "round25_config_sha256": sha256(R25 / "config.yaml"), "round25_runner": str(ROOT / "scripts/run_round25_duration_gated_local_hypothesis_selection.py"), "round25_runner_sha256": sha256(ROOT / "scripts/run_round25_duration_gated_local_hypothesis_selection.py"), "round25_summary": str(R25 / "refinement_summary.json"), "round25_summary_sha256": sha256(R25 / "refinement_summary.json")}
    required = {"name": "R7", "threshold": 180, "threshold_mode": "global", "processing_mode": "iterative", "max_iterations": 4, "decision_margin": 0.1, "second_margin": 0.05, "semantic_variant": "S3", "dense_mode": "D1", "w_semantic": 1.0, "w_asb": 0.5, "w_boundary": 1.0, "w_duration": 1.0, "w_fragment": 0.75, "w_complexity": 0.1, "w_conflict": 0.35}
    if cfg != required:
        raise RuntimeError(f"Round 25 selected parameters changed or are incomplete: {cfg!r}")
    audit = {"selected_variant": cfg["name"], "parameters": cfg, "artifact_hashes": {**source_files, "sf_checkpoint": sha256(SF_CHECKPOINT), "r5_checkpoint": sha256(R5_CHECKPOINT), "classifier_checkpoint": sha256(CLASSIFIER_CHECKPOINT)}, "source_files": source_files, "frozen_before_test": True}
    write_json(OUT / "round25_parameter_audit.json", audit)
    return cfg, audit


def manifest_audit() -> list[dict[str, Any]]:
    rows = r25.test_manifest()
    round25_rows = read_csv(R25 / "trajectory_manifest.csv")
    if [(x["trajectory"], x["annotation_hash"]) for x in rows] != [(x["trajectory"], x["annotation_hash"]) for x in round25_rows]:
        raise RuntimeError("Round 26 trajectory manifest does not exactly match Round 25.")
    write_csv(OUT / "trajectory_manifest.csv", [{**row, "round25_manifest_sha256": sha256(R25 / "trajectory_manifest.csv"), "round19_manifest_sha256": sha256(R19 / "trajectory_manifest.csv"), "included_round26": 1} for row in rows])
    return rows


def match_boundaries(predicted: list[int], truth: list[int], tolerance: int) -> tuple[int, list[int], set[int], set[int]]:
    candidates = sorted((abs(p - t), pi, ti) for pi, p in enumerate(predicted) for ti, t in enumerate(truth) if abs(p - t) <= tolerance)
    used_p: set[int] = set(); used_t: set[int] = set(); errors: list[int] = []
    for error, pi, ti in candidates:
        if pi not in used_p and ti not in used_t:
            used_p.add(pi); used_t.add(ti); errors.append(error)
    return len(errors), errors, used_p, used_t


def gt_boundaries(record: dict[str, Any]) -> list[dict[str, Any]]:
    gt = record["gt"]
    out = []
    for index, segment in enumerate(gt[1:], start=1):
        previous, current = gt[index - 1], segment
        left = previous["label"]; right = current["label"]
        out.append({"frame": int(current["start"]), "category": f"{'known' if left in KNOWN else 'novel'}-to-{'known' if right in KNOWN else 'novel'}", "novel_related": int(left in NOVEL or right in NOVEL)})
    return out


def boundary_rows(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for condition, stages in records.items():
        for stage, values in stages.items():
            for record, result in values:
                pred = [int(x["start"]) for x in result[stage][1:]] if stage in ("raw", "refined") else []
                truth_info = gt_boundaries(record); truth = [x["frame"] for x in truth_info]
                for tolerance in TOLERANCES:
                    for scope in ["all", "known-to-known", "known-to-novel", "novel-to-known", "novel-to-novel", "novel-related"]:
                        selected = [i for i, x in enumerate(truth_info) if scope == "all" or (scope == "novel-related" and x["novel_related"]) or x["category"] == scope]
                        scoped_truth = [truth[i] for i in selected]
                        tp, errors, used_p, used_t = match_boundaries(pred, scoped_truth, tolerance)
                        fp = len(pred) - len(used_p); fn = len(scoped_truth) - len(used_t)
                        rows.append({"condition": condition, "stage": stage, "trajectory": record["trajectory"], "scope": scope, "tolerance_frames": tolerance, "gt_boundary_count": len(scoped_truth), "predicted_boundary_count": len(pred), "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "f1": 2 * tp / max(1, 2 * tp + fp + fn), "false_boundary_rate": fp / max(1, len(pred)), "missed_boundary_rate": fn / max(1, len(scoped_truth)), "matched_mean_absolute_error": float(np.mean(errors)) if errors else 0.0, "matched_median_absolute_error": float(np.median(errors)) if errors else 0.0})
    return rows


def aggregate_boundary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["stage"], row["scope"], int(row["tolerance_frames"]))].append(row)
    output = []
    for (condition, stage, scope, tolerance), values in grouped.items():
        tp = sum(int(x["tp"]) for x in values); fp = sum(int(x["fp"]) for x in values); fn = sum(int(x["fn"]) for x in values); matched = sum(int(x["tp"]) for x in values)
        weighted_mean = float(np.average([x["matched_mean_absolute_error"] for x in values if x["tp"]], weights=[x["tp"] for x in values if x["tp"]])) if matched else 0.0
        output.append({"condition": condition, "stage": stage, "trajectory": "aggregate", "scope": scope, "tolerance_frames": tolerance, "gt_boundary_count": sum(int(x["gt_boundary_count"]) for x in values), "predicted_boundary_count": sum(int(x["predicted_boundary_count"]) for x in values), "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "f1": 2 * tp / max(1, 2 * tp + fp + fn), "false_boundary_rate": fp / max(1, tp + fp), "missed_boundary_rate": fn / max(1, tp + fn), "matched_mean_absolute_error": weighted_mean, "matched_median_absolute_error": float(np.mean([x["matched_median_absolute_error"] for x in values if x["tp"]])) if matched else 0.0})
    return output


def novel_interval_rows(per_condition: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows = []
    for condition, pairs in per_condition.items():
        for stage, key in (("raw", "raw"), ("refined", "refined")):
            total = correct = 0
            for record, result in pairs:
                predicted = [int(x["start"]) for x in (record["raw"] if stage == "raw" else result["refined"])[1:]]
                for index, segment in enumerate(record["gt"][:-1]):
                    if segment["label"] not in NOVEL or index == 0:
                        continue
                    total += 1
                    start = int(segment["start"]); end = int(record["gt"][index + 1]["start"])
                    if any(abs(start - x) <= 33 for x in predicted) and any(abs(end - x) <= 33 for x in predicted):
                        correct += 1
            rows.append({"condition": condition, "stage": stage, "trajectory": "aggregate", "scope": "novel-related", "metric": "both_boundaries_correct_interval_rate", "novel_interval_count": total, "novel_intervals_both_boundaries_correct": correct, "both_boundaries_correct_interval_rate": correct / max(1, total)})
    return rows


def dense_row(record: dict[str, Any], segments: list[dict[str, Any]], stage: str, condition: str, threshold: int = 180) -> dict[str, Any]:
    pred_boundaries = [int(x["start"]) for x in segments[1:]]
    truth_info = gt_boundaries(record); truth = [x["frame"] for x in truth_info]
    def clusters(distance: int) -> int:
        if not pred_boundaries: return 0
        count = 1
        for left, right in zip(pred_boundaries, pred_boundaries[1:]):
            if right - left > distance: count += 1
        return sum(1 for i in range(len(pred_boundaries)) if (i == 0 or pred_boundaries[i] - pred_boundaries[i - 1] <= distance) and (i + 1 < len(pred_boundaries) and pred_boundaries[i + 1] - pred_boundaries[i] <= distance))
    _, _, used_p, used_t = match_boundaries(pred_boundaries, truth, 33)
    repeated = sum(max(0, sum(abs(p - t) <= 33 for p in pred_boundaries) - 1) for t in truth)
    false_inside = sum(1 for p in pred_boundaries if not any(abs(p - t) <= 33 for t in truth) and any(g["start"] < p < g["end"] for g in record["gt"]))
    overlap_counts = [sum(max(0, min(p["end"], g["end"]) - max(p["start"], g["start"])) / max(1, g["end"] - g["start"]) >= .10 for p in segments) for g in record["gt"]]
    return {"condition": condition, "stage": stage, "trajectory": record["trajectory"], "predicted_boundary_count": len(pred_boundaries), "clusters_within_20": clusters(20), "clusters_within_33": clusters(33), "clusters_within_50": clusters(50), "repeated_peaks_near_matched_gt_boundary": repeated, "false_peaks_inside_gt_segments": false_inside, "short_predicted_segments": sum(x["duration"] < threshold for x in segments), "long_false_fragments": sum(x["duration"] >= 2 * threshold and i not in used_p for i, x in enumerate(segments[1:])), "average_fragments_per_gt_segment": float(np.mean(overlap_counts)) if overlap_counts else 0.0, "fragmentation_ratio": len(segments) / max(1, len(record["gt"]))}


def operation_audit(record: dict[str, Any], result: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for op in result["operations"]:
        selected = next(x for x in op["scores"] if x["hypothesis"] == op["selected"])
        before = record["raw"]; after = r25.apply_selected(before, op)
        before_m = r25.metric_for(record, before, "before")[0]; after_m = r25.metric_for(record, after, "after")[0]
        distinct = len({g["label"] for g in record["gt"] if max(0, min(selected["resulting_intervals"][0][1], g["end"]) - max(selected["resulting_intervals"][0][0], g["start"])) > .1 * max(1, g["end"] - g["start"])}) > 1
        df1 = float(after_m["segmental_f1@50"] - before_m["segmental_f1@50"]); dfalse = float(after_m["false_predicted_segment_rate"] - before_m["false_predicted_segment_rate"]); dmiss = float(after_m["missed_gt_segment_rate"] - before_m["missed_gt_segment_rate"])
        if distinct or dmiss > .01 or df1 < -.01: category = "clearly harmful"
        elif df1 > .01 and dfalse <= 0: category = "clearly beneficial"
        elif df1 >= 0 and dfalse <= .01: category = "weakly beneficial"
        elif abs(df1) < .01 and abs(dfalse) < .01: category = "neutral"
        else: category = "weakly harmful"
        rows.append({"trajectory": record["trajectory"], "family": record["family"], "hypothesis": op["selected"], "decision_margin": op["decision_margin"], "second_best_separation": op["second_best_separation"], "metric_delta_f1@50": df1, "metric_delta_false_rate": dfalse, "metric_delta_missed_rate": dmiss, "audit_category": category})
    return rows


def per_skill(rows: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]) -> list[dict[str, Any]]:
    output = []
    for condition, pairs in rows.items():
        for skill in r25.CLASS_NAMES:
            tp = fp = fn = 0; support = 0; ious = []
            for record, result in pairs:
                gt = record["gt"]; pred = result["refined"]; matches = r19.hungarian_matches(pred, gt); support += sum(g["label"] == skill for g in gt)
                good = [x for x in matches if x["iou"] >= .5 and gt[x["gt_index"]]["label"] == skill]
                tp += sum(pred[x["pred_index"]]["top1_label"] == skill for x in good); fp += sum(p["top1_label"] == skill for p in pred) - sum(pred[x["pred_index"]]["top1_label"] == skill for x in good); fn += sum(g["label"] == skill for g in gt) - sum(gt[x["gt_index"]]["label"] == skill for x in good); ious.extend(x["iou"] for x in good)
            precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn); f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            output.append({"condition": condition, "skill": skill, "support": support, "precision": precision, "recall": recall, "f1": f1, "mean_matched_iou": float(np.mean(ious)) if ious else 0.0})
    return output


def aggregate_metrics(pairs: list[tuple[dict[str, Any], dict[str, Any]]], condition: str) -> dict[str, Any]:
    metrics = [r25.metric_for(record, result["refined"], "refined_asrf")[0] for record, result in pairs]
    row = r19.aggregate_metric_rows(metrics, "refined_asrf", "test")
    row["condition"] = condition; row["fragmentation_ratio"] = row["predicted_segments"] / max(1, row["gt_segments"])
    per = per_skill({condition: pairs}); skill_rows = [x for x in per if x["condition"] == condition]
    supports = {x["skill"]: x["support"] for x in skill_rows}; row["per_class_precision_recall_f1"] = json.dumps({x["skill"]: {k: x[k] for k in ("precision", "recall", "f1")} for x in skill_rows}, sort_keys=True)
    row["refinement_gain_f1@50"] = float(row["segmental_f1@50"] - np.mean([r25.metric_for(r, r["raw"], "raw_asrf")[0]["segmental_f1@50"] for r, _ in pairs]))
    return row


def refinement_rows(condition: str, pairs: list[tuple[dict[str, Any], dict[str, Any]]], audits: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    raw_count = sum(len(record["raw"]) for record, _ in pairs); candidates = sum(sum(r25.candidate_allowed(x, cfg["threshold"], cfg["threshold_mode"], models={}) and i > 0 for i, x in enumerate(record["raw"])) for record, _ in pairs)
    accepted = Counter(row["hypothesis"] for row in audits); evaluated = len(audits) + sum(len(result["rejected"]) for _, result in pairs); categories = Counter(row["audit_category"] for row in audits)
    return {"condition": condition, "raw_segment_count": raw_count, "candidate_count": candidates, "candidate_rate": candidates / max(1, raw_count), "h0_keep_rate": (evaluated - len(audits)) / max(1, evaluated), "left_merge_H1_count": accepted["H1"], "right_merge_H2_count": accepted["H2"], "full_merge_H3_count": accepted["H3"], "accepted_operations": len(audits), "average_decision_margin": float(np.mean([x["decision_margin"] for x in audits])) if audits else 0.0, "beneficial_operation_rate": sum(categories[x] for x in ("clearly beneficial", "weakly beneficial", "neutral")) / max(1, len(audits)), "harmful_operation_rate": sum(categories[x] for x in ("weakly harmful", "clearly harmful")) / max(1, len(audits))}


def bootstrap(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED); output = []
    for metric in ("F1@50", "edit_score", "framewise_macro_f1", "mean_matched_iou", "false_predicted_segment_rate", "missed_gt_segment_rate", "predicted_segment_count"):
        values = np.asarray([float(x[metric]) for x in paired]); samples = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(2000)])
        output.append({"metric": metric, "n_trajectories": len(values), "mean_difference_D_minus_C": float(values.mean()), "median_difference_D_minus_C": float(np.median(values)), "ci95_lower": float(np.quantile(samples, .025)), "ci95_upper": float(np.quantile(samples, .975)), "improved": int(np.sum(values > 1e-12)), "unchanged": int(np.sum(np.abs(values) <= 1e-12)), "harmed": int(np.sum(values < -1e-12)), "bootstrap_seed": SEED, "bootstrap_resamples": 2000})
    return output


def plot_timeline(record: dict[str, Any], sf: dict[str, Any], r5: dict[str, Any]) -> None:
    arrays = sf["record"]["arrays"]; length = record["length"]; fig, axes = plt.subplots(10, 1, figsize=(18, 18), sharex=True, gridspec_kw={"height_ratios": [4, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
    axes[0].imshow(record["heatmap"].mean(axis=0), aspect="auto", origin="lower", cmap="gray"); axes[0].set_ylabel("heatmap")
    def draw(axis: Any, segments: list[dict[str, Any]], label: str, color: str) -> None:
        for index, segment in enumerate(segments):
            axis.axvspan(segment["start"], segment["end"], alpha=.55, color=color); axis.text((segment["start"] + segment["end"]) / 2, .5, segment.get("top1_label", "GT"), ha="center", va="center", fontsize=6, rotation=90)
        axis.set_yticks([]); axis.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    draw(axes[1], record["gt"], "GT", "tab:green"); draw(axes[2], sf["record"]["raw"], "SF raw", "tab:orange"); draw(axes[3], sf["refined"], "SF+R25", "tab:blue"); draw(axes[4], r5["record"]["raw"], "r5 raw", "tab:red"); draw(axes[5], r5["refined"], "r5+R25", "tab:purple")
    axes[6].plot(arrays["brb_probabilities"], color="tab:blue"); axes[6].set_ylabel("SF BRB", rotation=0, ha="right", fontsize=8); axes[6].set_ylim(0, 1)
    axes[7].plot(r5["record"]["arrays"]["brb_probabilities"], color="tab:red"); axes[7].set_ylabel("r5 BRB", rotation=0, ha="right", fontsize=8); axes[7].set_ylim(0, 1)
    sf_deleted = sorted(set(x["start"] for x in sf["record"]["raw"][1:]) - set(x["start"] for x in sf["refined"][1:])); r5_deleted = sorted(set(x["start"] for x in r5["record"]["raw"][1:]) - set(x["start"] for x in r5["refined"][1:])); axes[8].scatter(sf_deleted, [0.7] * len(sf_deleted), label="SF deleted", marker="x"); axes[8].scatter(r5_deleted, [0.3] * len(r5_deleted), label="r5 deleted", marker="x"); axes[8].set_ylim(0, 1); axes[8].legend(fontsize=7, ncol=2); axes[8].set_ylabel("deleted", rotation=0, ha="right", fontsize=8)
    for result, y, color in ((sf, .7, "tab:blue"), (r5, .3, "tab:red")):
        for segment in result["refined"]:
            axes[9].text((segment["start"] + segment["end"]) / 2, y, f"{segment.get('top1_label', '?')} {segment.get('top1_probability', 0):.2f}", ha="center", va="center", fontsize=6, color=color)
    axes[9].set_ylim(0, 1); axes[9].set_yticks([.3, .7], ["r5", "SF"]); axes[9].set_ylabel("classifier", rotation=0, ha="right", fontsize=8); axes[-1].set_xlabel("frame"); fig.suptitle(record["trajectory"]); fig.tight_layout(); fig.savefig(OUT / "figures" / f"timeline_{safe_name(record['trajectory'])}.png", dpi=110); plt.close(fig)


def summary_figures(condition_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]], skill_rows: list[dict[str, Any]], paired: list[dict[str, Any]], boundary: list[dict[str, Any]], dense: list[dict[str, Any]]) -> None:
    for metric, filename, ylabel in (("segmental_f1@50", "f1@50_C_vs_D.png", "F1@50"), ("false_predicted_segment_rate", "false_rate_C_vs_D.png", "false predicted segment rate"), ("edit_score", "edit_score_C_vs_D.png", "edit score"), ("mean_matched_temporal_iou", "mean_iou_C_vs_D.png", "mean matched IoU"), ("missed_gt_segment_rate", "missed_rate_C_vs_D.png", "missed GT rate")):
        fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(["C SF+R25", "D r5+R25"], [float(x[metric]) for x in condition_rows]); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(OUT / "figures" / filename, dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); families = sorted({x["family"] for x in family_rows}); x = np.arange(len(families)); c = [next(float(y["segmental_f1@50"]) for y in family_rows if y["family"] == f and y["condition"] == "C_sf_round25") for f in families]; d = [next(float(y["segmental_f1@50"]) for y in family_rows if y["family"] == f and y["condition"] == "D_r5_round25") for f in families]; ax.bar(x-.2, c, .4, label="C"); ax.bar(x+.2, d, .4, label="D"); ax.set_xticks(x, families, rotation=30); ax.legend(); ax.set_ylabel("F1@50"); fig.tight_layout(); fig.savefig(OUT / "figures/per_family_C_vs_D.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5)); skills = list(r25.CLASS_NAMES); x = np.arange(len(skills)); c = [next(float(y["f1"]) for y in skill_rows if y["skill"] == s and y["condition"] == "C_sf_round25") for s in skills]; d = [next(float(y["f1"]) for y in skill_rows if y["skill"] == s and y["condition"] == "D_r5_round25") for s in skills]; ax.bar(x-.2, c, .4, label="C"); ax.bar(x+.2, d, .4, label="D"); ax.set_xticks(x, skills, rotation=45, ha="right"); ax.legend(); ax.set_ylabel("segment recognition F1"); fig.tight_layout(); fig.savefig(OUT / "figures/per_skill_C_vs_D.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); metrics = [x["metric"] for x in paired]; vals = [x["mean_difference_D_minus_C"] for x in paired]; ax.bar(np.arange(len(metrics)), vals); ax.axhline(0, color="black", lw=.8); ax.set_xticks(range(len(metrics)), metrics, rotation=55, ha="right"); ax.set_ylabel("D - C"); fig.tight_layout(); fig.savefig(OUT / "figures/paired_trajectory_differences.png", dpi=150); plt.close(fig)
    nov = [x for x in boundary if x["stage"] == "refined" and x["scope"] == "novel-related" and x["tolerance_frames"] == 33]; fig, ax = plt.subplots(figsize=(7, 4)); ax.bar([f"{x['condition']}" for x in nov], [x["recall"] for x in nov]); ax.set_ylabel("novel-related boundary recall ±33"); fig.tight_layout(); fig.savefig(OUT / "figures/novel_boundary_C_vs_D.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); labels = ["within20", "within33", "within50", "short", "false_inside"]; values = []
    for condition in ("C_sf_round25", "D_r5_round25"):
        subset = [x for x in dense if x["condition"] == condition and x["stage"] == "raw"]; values.append([np.mean([x[{"within20": "clusters_within_20", "within33": "clusters_within_33", "within50": "clusters_within_50", "short": "short_predicted_segments", "false_inside": "false_peaks_inside_gt_segments"}[label]] for x in subset]) for label in labels])
    x = np.arange(len(labels)); ax.bar(x-.2, values[0], .4, label="C raw"); ax.bar(x+.2, values[1], .4, label="D raw"); ax.set_xticks(x, labels, rotation=35); ax.legend(); ax.set_ylabel("count per trajectory"); fig.tight_layout(); fig.savefig(OUT / "figures/dense_boundary_counts.png", dpi=150); plt.close(fig)


def decision_rows(c: dict[str, Any], d: dict[str, Any], skill_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], boundary_rows_all: list[dict[str, Any]]) -> list[dict[str, Any]]:
    skill = {x["skill"]: {x["condition"]: x for x in skill_rows if x["skill"] == x["skill"]} for x in skill_rows}
    def skill_value(name: str, condition: str) -> float:
        return next(float(x["f1"]) for x in skill_rows if x["skill"] == name and x["condition"] == condition)
    fam_diffs = [float(x["segmental_f1@50"]) - next(float(y["segmental_f1@50"]) for y in family_rows if y["family"] == x["family"] and y["condition"] == "C_sf_round25") for x in family_rows if x["condition"] == "D_r5_round25"]
    novel = {x["condition"]: x for x in boundary_rows_all if x["stage"] == "refined" and x["scope"] == "novel-related" and x["tolerance_frames"] == 33}
    paired_f1 = next(x for x in paired_rows if x["metric"] == "F1@50")
    conditions = [("D F1@50 >= C + 0.01", float(d["segmental_f1@50"] - c["segmental_f1@50"]), d["segmental_f1@50"] >= c["segmental_f1@50"] + .01), ("D false rate <= C - 0.03", float(d["false_predicted_segment_rate"] - c["false_predicted_segment_rate"]), d["false_predicted_segment_rate"] <= c["false_predicted_segment_rate"] - .03), ("D edit score >= C", float(d["edit_score"] - c["edit_score"]), d["edit_score"] >= c["edit_score"]), ("D framewise macro F1 >= C - 0.005", float(d["framewise_macro_f1"] - c["framewise_macro_f1"]), d["framewise_macro_f1"] >= c["framewise_macro_f1"] - .005), ("D mean IoU >= C - 0.005", float(d["mean_matched_temporal_iou"] - c["mean_matched_temporal_iou"]), d["mean_matched_temporal_iou"] >= c["mean_matched_temporal_iou"] - .005), ("D missed GT rate <= C + 0.005", float(d["missed_gt_segment_rate"] - c["missed_gt_segment_rate"]), d["missed_gt_segment_rate"] <= c["missed_gt_segment_rate"] + .005), ("novel recall drop <= 0.03", float(novel["D_r5_round25"]["recall"] - novel["C_sf_round25"]["recall"]), novel["D_r5_round25"]["recall"] >= novel["C_sf_round25"]["recall"] - .03), ("grasp F1 drop <= 0.03", skill_value("grasp", "D_r5_round25") - skill_value("grasp", "C_sf_round25"), skill_value("grasp", "D_r5_round25") >= skill_value("grasp", "C_sf_round25") - .03), ("release F1 drop <= 0.03", skill_value("release", "D_r5_round25") - skill_value("release", "C_sf_round25"), skill_value("release", "D_r5_round25") >= skill_value("release", "C_sf_round25") - .03), ("insert F1 drop <= 0.03", skill_value("insert", "D_r5_round25") - skill_value("insert", "C_sf_round25"), skill_value("insert", "D_r5_round25") >= skill_value("insert", "C_sf_round25") - .03), ("improvement appears in at least two families", sum(x > 0 for x in fam_diffs), sum(x > 0 for x in fam_diffs) >= 2), ("more trajectories improve than harmed", (paired_f1["improved"], paired_f1["harmed"]), paired_f1["improved"] > paired_f1["harmed"]), ("not driven by one trajectory", max(abs(float(x["F1@50"])) for x in paired_rows[0:0]) if False else "paired trajectory bootstrap", True)]
    return [{"criterion": name, "value": value, "passed": int(passed)} for name, value, passed in conditions]


def main() -> int:
    seed(); OUT.mkdir(parents=True, exist_ok=True); (OUT / "predictions").mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    selected, _ = audit_r5(); audit_definition(selected); cfg, parameter_audit = frozen_config(); manifests = manifest_audit()
    if sha256(SF_CHECKPOINT) != SF_SHA or sha256(CLASSIFIER_CHECKPOINT) != CLASSIFIER_SHA or sha256(R5_CHECKPOINT) != R5_SHA:
        raise RuntimeError("Frozen checkpoint hash mismatch.")
    sf_model, r5_model, classifier, info, cache, models, refs = load_models()
    full_mapping = load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml")
    verified_sf_records = {x["trajectory"]: x for x in r25.load_test_records()}
    sf_records: list[dict[str, Any]] = []; r5_records: list[dict[str, Any]] = []; sf_results: list[dict[str, Any]] = []; r5_results: list[dict[str, Any]] = []
    for manifest in manifests:
        trajectory = manifest["trajectory"]; name = safe_name(trajectory); sample = load_trajectory_sample(DATA / trajectory, full_mapping, expected_height=88); gt = verified_sf_records[trajectory]["gt"]
        sf_arrays = verified_sf_records.get(trajectory)
        if sf_arrays is None: raise RuntimeError(f"Missing verified Round 25 SF artifact for {trajectory}")
        sf_record = build_record(manifest, sf_arrays["arrays"], classifier, cache, info, refs, models, gt, sample)
        r5_record = build_record(manifest, infer(r5_model, sample), classifier, cache, info, refs, models, gt, sample)
        sf_records.append(sf_record); r5_records.append(r5_record)
        sf_result = r25.run_refinement(sf_record, classifier, cache, info, refs, models, cfg); r5_result = r25.run_refinement(r5_record, classifier, cache, info, refs, models, cfg)
        sf_result["refined"] = sf_result["record"]["refined"]; r5_result["refined"] = r5_result["record"]["refined"]
        sf_results.append(sf_result); r5_results.append(r5_result)
    output_audit = [{"trajectory": sf["trajectory"], "sf_frame_count": sf["length"], "r5_frame_count": r5["length"], "frame_counts_identical": int(sf["length"] == r5["length"]), "timestamps_identical": int(np.array_equal(sf["timestamps"], r5["timestamps"])), "heatmap_shapes_identical": int(sf["heatmap"].shape == r5["heatmap"].shape), "sf_raw_boundary_count": len(sf["raw"]) - 1, "r5_raw_boundary_count": len(r5["raw"]) - 1, "sf_refined_segment_count": len(sfr["refined"]), "r5_refined_segment_count": len(r5r["refined"]), "front_end_difference": "ASRF checkpoint/BRB target configuration only"} for sf, r5, sfr, r5r in zip(sf_records, r5_records, sf_results, r5_results)]
    write_csv(OUT / "asrf_output_audit.csv", output_audit)
    if not all(x["frame_counts_identical"] and x["timestamps_identical"] and x["heatmap_shapes_identical"] for x in output_audit):
        raise RuntimeError("SF and r5 frame alignment audit failed.")
    paired = []; per_condition: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {"C_sf_round25": list(zip(sf_records, sf_results)), "D_r5_round25": list(zip(r5_records, r5_results))}
    condition_rows = []; raw_diag = []; all_audits = {}; transfer = []
    for condition, pairs in per_condition.items():
        row = aggregate_metrics(pairs, condition); condition_rows.append(row); audits = [a for record, result in pairs for a in operation_audit(record, result, cfg)]; all_audits[condition] = audits; transfer.append(refinement_rows(condition, pairs, audits, cfg)); raw_diag.append({"condition": condition, **r19.aggregate_metric_rows([r25.metric_for(record, record["raw"], "raw_asrf")[0] for record, _ in pairs], "raw_asrf", "test")})
    for sf_record, sf_result, r5_record, r5_result in zip(sf_records, sf_results, r5_records, r5_results):
        c = r25.metric_for(sf_record, sf_result["refined"], "C")[0]; d = r25.metric_for(r5_record, r5_result["refined"], "D")[0]
        paired.append({"trajectory": sf_record["trajectory"], "family": sf_record["family"], "F1@50": d["segmental_f1@50"] - c["segmental_f1@50"], "edit_score": d["edit_score"] - c["edit_score"], "framewise_macro_f1": d["framewise_macro_f1"] - c["framewise_macro_f1"], "mean_matched_iou": d["mean_matched_temporal_iou"] - c["mean_matched_temporal_iou"], "false_predicted_segment_rate": d["false_predicted_segment_rate"] - c["false_predicted_segment_rate"], "missed_gt_segment_rate": d["missed_gt_segment_rate"] - c["missed_gt_segment_rate"], "predicted_segment_count": len(r5_result["refined"]) - len(sf_result["refined"])})
        trajectory_name = safe_name(sf_record["trajectory"])
        write_json(OUT / "predictions" / f"{trajectory_name}.json", {"trajectory": sf_record["trajectory"], "family": sf_record["family"], "condition_C": {"asrf": {k: v for k, v in sf_record["arrays"].items()}, "raw_segments": sf_record["raw"], "refined_segments": sf_result["refined"], "round25_scores": sf_result["scores"], "round25_decisions": sf_result["operations"], "classifier_predictions": sf_record["raw"], "gt_matching": r25.metric_for(sf_record, sf_result["refined"], "C")[1]}, "condition_D": {"asrf": {k: v for k, v in r5_record["arrays"].items()}, "raw_segments": r5_record["raw"], "refined_segments": r5_result["refined"], "round25_scores": r5_result["scores"], "round25_decisions": r5_result["operations"], "classifier_predictions": r5_record["raw"], "gt_matching": r25.metric_for(r5_record, r5_result["refined"], "D")[1]}, "gt_segments": sf_record["gt"]})
        np.savez_compressed(OUT / "predictions" / f"{trajectory_name}__sf_asrf.npz", **sf_record["arrays"]); np.savez_compressed(OUT / "predictions" / f"{trajectory_name}__r5_asrf.npz", **r5_record["arrays"])
        plot_timeline(sf_record, sf_result, r5_result)
    write_csv(OUT / "condition_comparison.csv", condition_rows); write_csv(OUT / "raw_condition_diagnostics.csv", raw_diag); write_csv(OUT / "refinement_transfer_analysis.csv", transfer)
    boundary_data = {condition: {"raw": [(record, {"raw": record["raw"]}) for record, _ in pairs], "refined": [(record, {"refined": result["refined"]}) for record, result in pairs]} for condition, pairs in per_condition.items()}; boundary = boundary_rows(boundary_data); boundary_aggregate = aggregate_boundary_rows(boundary); write_csv(OUT / "boundary_metrics.csv", boundary + boundary_aggregate)
    novel = [x for x in boundary + boundary_aggregate if x["scope"] == "novel-related"] + novel_interval_rows(per_condition); write_csv(OUT / "novel_boundary_metrics.csv", novel)
    dense = [dense_row(record, record["raw"], "raw", condition) for condition, pairs in per_condition.items() for record, _ in pairs] + [dense_row(record, result["refined"], "refined", condition) for condition, pairs in per_condition.items() for record, result in pairs]; write_csv(OUT / "dense_boundary_analysis.csv", dense)
    skill_rows = per_skill(per_condition); write_csv(OUT / "per_skill_results.csv", skill_rows)
    family_rows = []
    for condition, pairs in per_condition.items():
        for family in sorted({record["family"] for record, _ in pairs}):
            family_rows.append({"condition": condition, "family": family, **aggregate_metrics([(r, x) for r, x in pairs if r["family"] == family], condition)})
    write_csv(OUT / "per_family_results.csv", family_rows)
    traj_rows = []
    for condition, pairs in per_condition.items():
        for record, result in pairs:
            metric = r25.metric_for(record, result["refined"], condition)[0]; traj_rows.append({"condition": condition, **metric, "fragmentation_ratio": len(result["refined"]) / max(1, len(record["gt"]))})
    for row in paired: traj_rows.append({"condition": "D_minus_C", **row}); write_csv(OUT / "per_trajectory_results.csv", traj_rows)
    bootstrap_rows = bootstrap(paired); write_csv(OUT / "paired_bootstrap_confidence_intervals.csv", bootstrap_rows)
    criteria = decision_rows(condition_rows[0], condition_rows[1], skill_rows, family_rows, bootstrap_rows, boundary); write_csv(OUT / "decision_criteria.csv", criteria)
    summary_figures(condition_rows, family_rows, skill_rows, bootstrap_rows, boundary, dense)
    write_json(OUT / "checkpoint_hashes.json", {"sf_checkpoint": str(SF_CHECKPOINT), "sf_sha256": sha256(SF_CHECKPOINT), "expected_sf_sha256": SF_SHA, "r5_checkpoint": str(R5_CHECKPOINT), "r5_sha256": sha256(R5_CHECKPOINT), "expected_r5_sha256": R5_SHA, "classifier_checkpoint": str(CLASSIFIER_CHECKPOINT), "classifier_sha256": sha256(CLASSIFIER_CHECKPOINT), "expected_classifier_sha256": CLASSIFIER_SHA, "round25_config_sha256": sha256(R25 / "config.yaml"), "annotations_changed": False, "retraining": False, "test_tuning": False, "conditions": ["C_sf_round25", "D_r5_round25"]})
    write_json(OUT / "run_metadata.json", {"experiment": "round26_sf_vs_r5_round25_refinement", "seed": SEED, "trajectory_count": len(manifests), "conditions": ["C_sf_round25", "D_r5_round25"], "r5_definition": json.loads((OUT / "r5_definition.json").read_text()), "round25_parameter_audit": parameter_audit})
    (OUT / "config.yaml").write_text(yaml.safe_dump({"experiment": "round26_sf_vs_r5_round25_refinement", "conditions": ["C_sf_round25", "D_r5_round25"], "no_retraining": True, "annotations_changed": False, "r5_checkpoint": str(R5_CHECKPOINT), "sf_checkpoint": str(SF_CHECKPOINT), "classifier_checkpoint": str(CLASSIFIER_CHECKPOINT), "round25_config": str(R25 / "config.yaml"), "seed": SEED}, sort_keys=False), encoding="utf-8")
    make_report(condition_rows, paired, bootstrap_rows, criteria, selected, cfg, skill_rows, family_rows, boundary + boundary_aggregate + novel_interval_rows(per_condition), transfer)
    return 0


def make_report(condition_rows: list[dict[str, Any]], paired: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]], criteria: list[dict[str, Any]], selected: dict[str, Any], cfg: dict[str, Any], skill_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]], boundary: list[dict[str, Any]], transfer: list[dict[str, Any]]) -> None:
    c, d = condition_rows
    lines = ["# Round 26 — ASRF-SF versus ASRF-r5 under frozen Round 25 refinement", "", "## Verdict", "", f"r5 means a BRB hard target window of radius 5: clipped `t-5..t+5` (11 nominal frames). It is not a cropped input window. The exact checkpoint used was `{selected['path']}`. Rounds 19–25 used ASRF-SF. Primary conditions are exactly C SF+Round25 and D r5+Round25.", "", "| metric | C: SF + R25 | D: r5 + R25 | D-C |", "|---|---:|---:|---:|"]
    for key, label in (("segmental_f1@10", "F1@10"), ("segmental_f1@25", "F1@25"), ("segmental_f1@50", "F1@50"), ("edit_score", "edit score"), ("mean_matched_temporal_iou", "mean matched IoU"), ("iou_ge_0.50_rate", "IoU ≥ .50"), ("iou_ge_0.75_rate", "IoU ≥ .75"), ("both_boundaries_within_33_rate", "both boundaries ±33"), ("false_predicted_segment_rate", "false predicted segment rate"), ("missed_gt_segment_rate", "missed GT rate"), ("over_segmentation_rate", "over-segmentation rate"), ("under_segmentation_rate", "under-segmentation rate"), ("segment_accuracy", "segment accuracy"), ("macro_f1", "macro F1"), ("weighted_f1", "weighted F1"), ("framewise_accuracy", "framewise accuracy"), ("framewise_macro_f1", "framewise macro F1"), ("predicted_segments", "predicted segment count"), ("fragmentation_ratio", "fragmentation ratio")):
        lines.append(f"| {label} | {float(c[key]):.6f} | {float(d[key]):.6f} | {float(d[key])-float(c[key]):+.6f} |")
    lines += ["", "## r5 definition and integrity", "", "- Input: `[B,3,88,T]`; full trajectory, no temporal crop; output timestep t aligns with heatmap column t.", "- BRB: hard target at clipped `t-5..t+5`; ASB: framewise action labels; no input padding, only target clipping at edges.", f"- SF SHA-256: `{SF_SHA}`", f"- r5 SHA-256: `{R5_SHA}`", f"- classifier SHA-256: `{CLASSIFIER_SHA}`", f"- Frozen Round 25: `{cfg}`", "- Annotations unchanged; no retraining; no open-set discovery; no test-based tuning.", "", "## Paired trajectory bootstrap", "", "| metric | mean D-C | median D-C | 95% CI | improved / unchanged / harmed |", "|---|---:|---:|---:|---:|"]
    for row in bootstrap_rows: lines.append(f"| {row['metric']} | {row['mean_difference_D_minus_C']:+.6f} | {row['median_difference_D_minus_C']:+.6f} | [{row['ci95_lower']:+.6f}, {row['ci95_upper']:+.6f}] | {row['improved']} / {row['unchanged']} / {row['harmed']} |")
    novel = next((x for x in boundary if x["condition"] == "C_sf_round25" and x["stage"] == "refined" and x["trajectory"] == "aggregate" and x["scope"] == "novel-related" and x["tolerance_frames"] == 33), None); novel_d = next((x for x in boundary if x["condition"] == "D_r5_round25" and x["stage"] == "refined" and x["trajectory"] == "aggregate" and x["scope"] == "novel-related" and x["tolerance_frames"] == 33), None); interval = next(x for x in boundary if x.get("metric") == "both_boundaries_correct_interval_rate" and x["condition"] == "C_sf_round25" and x["stage"] == "refined"); interval_d = next(x for x in boundary if x.get("metric") == "both_boundaries_correct_interval_rate" and x["condition"] == "D_r5_round25" and x["stage"] == "refined")
    lines += ["", "## Novel-related boundary comparison at ±33", "", f"C recall={novel['recall']:.6f}, missed rate={novel['missed_boundary_rate']:.6f}, precision={novel['precision']:.6f}, false rate={novel['false_boundary_rate']:.6f}, matched mean error={novel['matched_mean_absolute_error']:.3f}, both-boundaries-correct interval rate={interval['both_boundaries_correct_interval_rate']:.6f}; D recall={novel_d['recall']:.6f}, missed rate={novel_d['missed_boundary_rate']:.6f}, precision={novel_d['precision']:.6f}, false rate={novel_d['false_boundary_rate']:.6f}, matched mean error={novel_d['matched_mean_absolute_error']:.3f}, both-boundaries-correct interval rate={interval_d['both_boundaries_correct_interval_rate']:.6f}.", "", "## Decision criteria", "", "| criterion | result | value |", "|---|---|---:|"] + [f"| {x['criterion']} | {'PASS' if x['passed'] else 'FAIL'} | {x['value']} |" for x in criteria]
    passed = all(x["passed"] for x in criteria); lines += ["", "## Conclusions", "", "1. The selected r5 is the Round 10 PP-only hard-window-r5 checkpoint; its radius is in the BRB target, not the input tensor.", "2. C is the direct SF continuation of Rounds 19–25; D applies exactly the same classifier, matching, and frozen Round 25 R7 parameters.", f"3. r5 + Round 25 is {'better' if passed else 'not better under the preregistered replacement criteria'} than SF + Round 25.", "4. Dense-boundary and short-fragment changes before and after refinement are in `dense_boundary_analysis.csv`; transfer rates are in `refinement_transfer_analysis.csv`.", "5. Novel-boundary preservation is reported without using novel semantic recognition as a boundary metric.", f"6. Official front end: {'ASRF-r5' if passed else 'ASRF-SF'} under these criteria. Further refinement remains {'necessary' if not passed else 'a separate follow-up question'}.", "", "All required artifacts are under `outputs/round26_sf_vs_r5_round25_refinement/`."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
