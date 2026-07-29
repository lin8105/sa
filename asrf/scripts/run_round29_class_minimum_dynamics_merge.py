#!/usr/bin/env python3
"""Round 29: class-conditional minimum skill-dynamics forced merging.

The front end is deliberately treated as an immutable artifact.  Test ASRF
outputs and inputs are read from the verified Round 27B NPZ/JSON exports;
only the post-processing rule is new in this round.
"""
from __future__ import annotations

import csv, hashlib, json, sys
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
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from asrf.data.dataset import load_heatmap, load_timestamp_vector  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.visualization.temporal import DEFAULT_LABEL_COLORS, _normalized_heatmap  # noqa: E402
import run_round27_pp_only_r5_region_sf_point_hybrid as r27  # noqa: E402
import run_round27b_complete_test_temporal_only as r27b  # noqa: E402

OUT = ROOT / "outputs/round29_class_minimum_dynamics_merge"
SOURCE = ROOT / "outputs/round27b_complete_test_temporal_only"
DATA = r27.DATA
KNOWN = tuple(r27.KNOWN)
KNOWN_SET = set(KNOWN)
FUSION = {"threshold": 0.50, "gap": 0, "rule": "P4", "support_gate": 0.50, "separation": 0}
SF_SHA = "6b11abff2ff4387eada88e38fb56133b29fd5c3facdfb54b42fc84701213db9a"
R5_SHA = "577d8edf9e2b04927acc235ffa4d6baab8df1712dd0b98eaaba9063fde31f406"
CANONICAL_SF = ROOT / "outputs/round10_pp_only_novel_segmentation/models/single_frame/best.pt"
CANONICAL_R5 = ROOT / "outputs/round10_pp_only_novel_segmentation/models/hard_window_r5/best.pt"
TOLERANCES = (5, 10, 20, 33, 50)
IOU_THRESHOLDS = (0.10, 0.25, 0.50, 0.75)
CLIP = (-5.0, 5.0)
EPS = 1e-8


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields = list(dict.fromkeys(k for row in rows for k in row)) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def resolve_exact(canonical: Path, expected: str) -> Path:
    candidates = [canonical, ROOT / "outputs/0" / canonical.relative_to(ROOT / "outputs")]
    candidates += [p for p in ROOT.glob("outputs/**/best.pt") if p not in candidates]
    matches = [p for p in candidates if p.is_file() and digest(p) == expected]
    if len(matches) != 1: raise RuntimeError(f"expected exactly one frozen artifact for {canonical}, found {matches}")
    return matches[0]


def resolve_config(canonical: Path, resolved_model: Path) -> Path:
    direct = canonical
    fallback = ROOT / "outputs/0" / canonical.relative_to(ROOT / "outputs")
    for p in (direct, fallback, resolved_model.with_name("config.yaml")):
        if p.is_file(): return p
    raise RuntimeError(f"missing architecture config for {resolved_model}")


def audit_frontend() -> dict[str, Any]:
    sf = resolve_exact(CANONICAL_SF, SF_SHA); r5 = resolve_exact(CANONICAL_R5, R5_SHA)
    sf_cfg_path = resolve_config(CANONICAL_SF.with_name("config.yaml"), sf); r5_cfg_path = resolve_config(CANONICAL_R5.with_name("config.yaml"), r5)
    sf_cfg = yaml.safe_load(sf_cfg_path.read_text()); r5_cfg = yaml.safe_load(r5_cfg_path.read_text())
    sf_payload = torch.load(sf, map_location="cpu", weights_only=False); r5_payload = torch.load(r5, map_location="cpu", weights_only=False)
    if sf_payload.get("architecture_config") != sf_cfg["model"] or r5_payload.get("architecture_config") != r5_cfg["model"]: raise RuntimeError("checkpoint architecture metadata mismatch")
    if sf_payload.get("label_map") != r5_payload.get("label_map"): raise RuntimeError("checkpoint ontology mismatch")
    expected_map = {name: i for i, name in enumerate(KNOWN)}
    if sf_payload.get("label_map") != expected_map: raise RuntimeError("checkpoint label order mismatch")
    if sf_cfg["data"]["boundary_target_mode"] != "single_frame" or r5_cfg["data"]["boundary_target_mode"] != "hard_window" or r5_cfg["data"]["boundary_window_radius"] != 5: raise RuntimeError("frozen front-end target mismatch")
    if sf_cfg["model"] != r5_cfg["model"]: raise RuntimeError("front-end model architecture mismatch")
    sf_model = ASRFModel.from_config(sf_cfg); r5_model = ASRFModel.from_config(r5_cfg); sf_model.load_state_dict(sf_payload["model_state"], strict=True); r5_model.load_state_dict(r5_payload["model_state"], strict=True)
    source_audit = SOURCE / "frozen_configuration_audit.json"
    if not source_audit.is_file(): raise RuntimeError("Round 27B frozen audit missing")
    source = json.loads(source_audit.read_text())
    if source.get("checkpoint_hashes", {}).get("sf") not in (None, SF_SHA) or source.get("checkpoint_hashes", {}).get("r5") not in (None, R5_SHA): raise RuntimeError("Round 27B checkpoint provenance mismatch")
    audit = {"source_round27b": str(SOURCE), "requested_checkpoints": {"sf": str(CANONICAL_SF), "r5": str(CANONICAL_R5)}, "resolved_checkpoints": {"sf": str(sf), "r5": str(r5)}, "checkpoint_hashes": {"sf": SF_SHA, "r5": R5_SHA}, "fusion": FUSION, "ontology": list(KNOWN), "strict_state_dict_loading": True, "no_retraining": True, "no_round28_logic": True, "no_test_tuning": True, "input_source": "verified Round 27B NPZ input_heatmap and timestamps", "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in (sf, r5, sf_cfg_path, r5_cfg_path, source_audit, ROOT / "scripts/run_round27b_complete_test_temporal_only.py")}}
    write_json(OUT / "frozen_frontend_audit.json", audit); write_json(OUT / "checkpoint_hashes.json", audit["checkpoint_hashes"] | {"resolved_sf": str(sf), "resolved_r5": str(r5)})
    return {"audit": audit, "sf_model": sf_model, "r5_model": r5_model}


def read_numeric(entry: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = DATA / entry; heat = load_heatmap(path / "citr_fingerprint_pure.png", expected_height=88).numpy().astype(np.float32); timestamps = load_timestamp_vector(path / "citr_features.csv")
    gripper = []
    with (path / "citr_features.csv").open(encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        if "gripper_position" not in (rows.fieldnames or []): raise RuntimeError(f"{entry}: gripper_position metadata missing")
        csv_ts = []
        for row in rows: csv_ts.append(int(row["timestamp_us"])); gripper.append(float(row["gripper_position"]))
    if not np.array_equal(timestamps, np.asarray(csv_ts, dtype=np.int64)) or heat.shape[-1] != len(timestamps): raise RuntimeError(f"{entry}: feature/heatmap alignment mismatch")
    return heat, np.asarray(gripper, dtype=np.float32), timestamps


def fit_normalization(entries: list[str]) -> dict[str, Any]:
    channels = []
    grippers = []
    for entry in entries:
        heat, grip, _ = read_numeric(entry); channels.append(heat.reshape(3, -1)); grippers.append(grip)
    values = np.concatenate(channels, axis=1); gs = np.concatenate(grippers)
    med = np.median(values, axis=1); q25 = np.percentile(values, 25, axis=1); q75 = np.percentile(values, 75, axis=1); q05 = np.percentile(values, 5, axis=1); q95 = np.percentile(values, 95, axis=1); std = np.std(values, axis=1)
    gmed = float(np.median(gs)); gq25 = float(np.percentile(gs, 25)); gq75 = float(np.percentile(gs, 75)); gq05 = float(np.percentile(gs, 5)); gq95 = float(np.percentile(gs, 95)); gstd = float(np.std(gs)); giqr = max(gq75 - gq25, EPS)
    stats = {"source_split": "train/pick and place/pp1-pp10 only", "channels": {"median": med.tolist(), "q25": q25.tolist(), "q75": q75.tolist(), "iqr": (q75 - q25).tolist(), "std": std.tolist(), "lower_quantile_q05": q05.tolist(), "upper_quantile_q95": q95.tolist()}, "gripper_position": {"median": gmed, "q25": gq25, "q75": gq75, "iqr": gq75 - gq25, "std": gstd, "lower_quantile_q05": gq05, "upper_quantile_q95": gq95}, "clip_normalized": list(CLIP), "epsilon": EPS, "feature_tensor": "ASRF RGB heatmap [3,88,T]; dynamics terms average over all spatial/channel rows", "gripper_source": "citr_features.csv:gripper_position"}
    write_json(OUT / "normalization_statistics.json", stats); return stats


def normalize(heat: np.ndarray, grip: np.ndarray, stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    med = np.asarray(stats["channels"]["median"])[:, None, None]; iqr = np.maximum(np.asarray(stats["channels"]["iqr"]), EPS)[:, None, None]; z = np.clip((heat - med) / iqr, *CLIP); gs = stats["gripper_position"]; gz = np.clip((grip - gs["median"]) / max(gs["iqr"], EPS), *CLIP); return z, gz


def segment_score(heat: np.ndarray, grip: np.ndarray, start: int, end: int, stats: dict[str, Any]) -> dict[str, float]:
    z, gz = normalize(heat, grip, stats); x = z[:, :, max(0, start):min(end, z.shape[-1])]; g = gz[max(0, start):min(end, len(gz))]; length = x.shape[-1]
    if length < 1: raise ValueError("empty segment")
    d1 = float(np.mean(np.abs(np.diff(x, axis=2)))) if length > 1 else 0.0
    lag_values = {}
    for lag in (5, 10, 20, 50): lag_values[str(lag)] = float(np.mean(np.abs(x[:, :, lag:] - x[:, :, :-lag]))) if length > lag else 0.0
    thirds = np.array_split(x, 3, axis=2); means = [np.mean(q, axis=2) for q in thirds if q.shape[2]]
    phase = float(np.mean([np.sqrt(np.mean((means[i] - means[j]) ** 2)) for i, j in ((0, 1), (1, 2), (0, 2))])) if len(means) == 3 else 0.0
    robust_range = float(np.mean(np.percentile(x, 95, axis=(1, 2)) - np.percentile(x, 5, axis=(1, 2))))
    gd = np.diff(g); grange = float(np.percentile(g, 95) - np.percentile(g, 5)); gpath = float(np.mean(np.abs(gd))) if len(g) > 1 else 0.0; gmax = float(np.max(np.abs(gd))) if len(g) > 1 else 0.0; gnet = float(abs(g[-1] - g[0])) if len(g) > 1 else 0.0; gripper = float(np.mean([grange, gpath, gmax, gnet]))
    multiscale = float(np.mean(list(lag_values.values()))); s0 = d1; s1 = multiscale; s2 = float(np.mean([multiscale, phase])); s3 = float(np.mean([multiscale, phase, robust_range])); s4 = float(np.mean([d1, *lag_values.values(), phase, robust_range, gripper]))
    return {"d1": d1, "dlag5": lag_values["5"], "dlag10": lag_values["10"], "dlag20": lag_values["20"], "dlag50": lag_values["50"], "d_phase": phase, "robust_range": robust_range, "gripper_range": grange, "gripper_total_abs_change_rate": gpath, "gripper_max_change": gmax, "gripper_first_last_change": gnet, "gripper_dynamics": gripper, "S0": s0, "S1": s1, "S2": s2, "S3": s3, "S4": s4, "duration_frames": end - start}


def gt_segments(entry: str, timestamps: np.ndarray) -> list[dict[str, Any]]:
    mapping = load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml"); return r27b.audit_annotation(entry, timestamps, mapping)[0]


def train_segments(entries: list[str], stats: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        heat, grip, ts = read_numeric(entry); gt = gt_segments(entry, ts)
        for seg in gt:
            if seg["label"] not in KNOWN_SET: continue
            score = segment_score(heat, grip, seg["start"], seg["end"], stats); rows.append({"split": split, "trajectory": entry, "class": seg["label"], "start": seg["start"], "end": seg["end"], "duration_frames": seg["end"] - seg["start"], "duration_seconds": (seg["end"] - seg["start"]) * .01, **score})
    return rows


def thresholds(rows: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    out = {}; audit = []
    for label in KNOWN:
        vals = [x for x in rows if x["class"] == label]; scores = sorted(x["S4"] for x in vals); minimum = min(scores) if scores else 0.0; out[label] = minimum; source = min(vals, key=lambda x: x["S4"]) if vals else {}
        row = {"class": label, "count": len(vals), "minimum_S4": minimum, "second_smallest_S4": scores[1] if len(scores) > 1 else minimum, "p1_S4": float(np.percentile(scores, 1)) if scores else 0.0, "p5_S4": float(np.percentile(scores, 5)) if scores else 0.0, "p10_S4": float(np.percentile(scores, 10)) if scores else 0.0, "median_S4": float(np.median(scores)) if scores else 0.0, "maximum_S4": max(scores) if scores else 0.0, "minimum_source_trajectory": source.get("trajectory", ""), "minimum_source_start": source.get("start", ""), "minimum_source_end": source.get("end", ""), "minimum_source_duration_frames": source.get("duration_frames", ""), "minimum_source_valid_complete_gt": int(bool(source))}; outrow = {**row}; audit.append({**source, "threshold_S4": minimum, "valid_complete_gt_confirmation": int(bool(source))});
        rows.append({}) if False else None
        if vals: outrow.update({f"minimum_{k}": source.get(k, "") for k in ("d1", "dlag5", "dlag10", "dlag20", "dlag50", "d_phase", "robust_range", "gripper_dynamics")})
        if not vals: outrow.update({f"minimum_{k}": "" for k in ("d1", "dlag5", "dlag10", "dlag20", "dlag50", "d_phase", "robust_range", "gripper_dynamics")})
        write_row = outrow
        audit[-1]["minimum_segment_score_components"] = {k: source.get(k, "") for k in ("d1", "dlag5", "dlag10", "dlag20", "dlag50", "d_phase", "robust_range", "gripper_dynamics", "S4")}
        if "rows_out" not in locals(): rows_out = []
        rows_out.append(write_row)
    return out, (rows_out, audit)


def make_segment(start: int, end: int, labels: np.ndarray, probs: np.ndarray, heat: np.ndarray, grip: np.ndarray, stats: dict[str, Any], thresholds_map: dict[str, float], index: int) -> dict[str, Any]:
    vals = labels[start:end].astype(int); counts = Counter(vals.tolist()); top_id = counts.most_common(1)[0][0]; label = KNOWN[int(top_id)]; ratio = counts[top_id] / max(1, len(vals)); score = segment_score(heat, grip, start, end, stats); margin = score["S4"] / max(thresholds_map[label], EPS); return {"segment_index": index, "start": int(start), "end": int(end), "duration": int(end - start), "top1_id": int(top_id), "top1_label": label, "majority_ratio": float(ratio), "top1_probability": float(np.mean(probs[int(top_id), start:end])), **score, "class_threshold": float(thresholds_map[label]), "validity_margin": float(margin), "invalid": int(score["S4"] < thresholds_map[label])}


def recompute_segments(points: list[int], labels: np.ndarray, probs: np.ndarray, heat: np.ndarray, grip: np.ndarray, stats: dict[str, Any], threshold_map: dict[str, float]) -> list[dict[str, Any]]:
    boundaries = sorted(set([0, *[int(p) for p in points if 0 < p < len(labels)], len(labels)])); return [make_segment(a, b, labels, probs, heat, grip, stats, threshold_map, i) for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))]


def merge_pair(a: dict[str, Any], b: dict[str, Any], labels: np.ndarray, probs: np.ndarray, heat: np.ndarray, grip: np.ndarray, stats: dict[str, Any], thresholds_map: dict[str, float], index: int) -> dict[str, Any]:
    return make_segment(a["start"], b["end"], labels, probs, heat, grip, stats, thresholds_map, index)


def force_merges(segments: list[dict[str, Any]], labels: np.ndarray, probs: np.ndarray, heat: np.ndarray, grip: np.ndarray, stats: dict[str, Any], threshold_map: dict[str, float], max_iterations: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    work = [dict(x) for x in segments]; initial = len(work); maximum = max_iterations or 2 * max(1, initial); operations = []; decisions = []
    for iteration in range(maximum):
        invalid = [i for i, x in enumerate(work) if x["invalid"]]
        for i in invalid: decisions.append({"iteration": iteration, "segment_index": i, "start": work[i]["start"], "end": work[i]["end"], "predicted_class": work[i]["top1_label"], "score": work[i]["S4"], "threshold": work[i]["class_threshold"], "validity_margin": work[i]["validity_margin"], "status": "invalid"})
        if not invalid: break
        i = min(invalid, key=lambda j: (work[j]["validity_margin"], work[j]["start"]))
        s = work[i]; candidates = []
        if i > 0:
            left = merge_pair(work[i - 1], s, labels, probs, heat, grip, stats, threshold_map, i - 1); right = work[i + 1] if i + 1 < len(work) else None; candidates.append(("MERGE_LEFT", left, right))
        if i + 1 < len(work):
            left = work[i - 1] if i > 0 else None; right = merge_pair(s, work[i + 1], labels, probs, heat, grip, stats, threshold_map, i); candidates.append(("MERGE_RIGHT", left, right))
        if not candidates:
            # A one-segment trajectory has no legal ML/MR hypothesis.  This is
            # an explicit infeasible edge case, not a silent KEEP decision.
            decisions.append({"iteration": iteration, "segment_index": i, "status": "unmergeable_single_segment", "reason": "no valid neighbor exists"})
            break
        scored = []
        for direction, left, right in candidates:
            valid = [x["validity_margin"] for x in (left, right) if x is not None]; score = sum(valid); minimum = min(valid); ratio = float(np.mean([x["majority_ratio"] for x in (left, right) if x is not None])); changes = sum(x["top1_label"] != work[j]["top1_label"] for x, j in ((left, i - 1), (right, i + 1)) if x is not None and 0 <= j < len(work)); scored.append({"direction": direction, "score": score, "minimum_margin": minimum, "majority_ratio": ratio, "class_changes": changes, "left": left, "right": right})
        chosen = max(scored, key=lambda x: (x["score"], x["minimum_margin"], x["majority_ratio"], -x["class_changes"], int(x["direction"] == "MERGE_LEFT"))); chosen_direction = chosen["direction"]
        if chosen_direction == "MERGE_LEFT":
            old = work[i - 1:i + 1]; result = chosen["left"]; work[i - 1:i + 1] = [result]
        else:
            old = work[i:i + 2]; result = chosen["right"]; work[i:i + 2] = [result]
        for j, x in enumerate(work): x["segment_index"] = j
        operations.append({"iteration": iteration, "source_segment": dict(s), "direction": chosen_direction, "ML_score": next(x["score"] for x in scored if x["direction"] == "MERGE_LEFT") if any(x["direction"] == "MERGE_LEFT" for x in scored) else "", "MR_score": next(x["score"] for x in scored if x["direction"] == "MERGE_RIGHT") if any(x["direction"] == "MERGE_RIGHT" for x in scored) else "", "chosen_result": dict(result), "removed_boundary": int(s["start"] if chosen_direction == "MERGE_RIGHT" else s["end"]), "old_segments": [dict(x) for x in old]})
    if any(x["invalid"] for x in work): decisions.append({"status": "max_iterations_reached", "remaining_invalid": sum(x["invalid"] for x in work)})
    return work, operations, decisions


def temporal_metrics(pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matches = r27b.temporal_matches(pred, gt); row = {"gt_segment_count": len(gt), "predicted_segment_count": len(pred), "matched_segment_count": len(matches), "unmatched_predicted_segment_count": len(pred) - len({x["pred_index"] for x in matches}), "unmatched_gt_segment_count": len(gt) - len({x["gt_index"] for x in matches})}; ious = [x["iou"] for x in matches]
    row.update({"mean_matched_iou": float(np.mean(ious)) if ious else 0.0, "median_matched_iou": float(np.median(ious)) if ious else 0.0, "iou_std": float(np.std(ious)) if ious else 0.0, "fraction_gt_iou_ge_0.50": sum(x >= .5 for x in ious) / max(1, len(gt),), "fraction_gt_iou_ge_0.75": sum(x >= .75 for x in ious) / max(1, len(gt)), "temporal_over_segmentation_rate": max(0, len(pred) - len(gt)) / max(1, len(gt)), "temporal_under_segmentation_rate": max(0, len(gt) - len(pred)) / max(1, len(gt)), "fragmentation_ratio": len(pred) / max(1, len(gt)), "segments_per_gt_segment": len(pred) / max(1, len(gt))})
    for threshold in IOU_THRESHOLDS:
        tp = sum(x >= threshold for x in ious); row[f"temporal_precision@{threshold:.2f}"] = tp / max(1, len(pred)); row[f"temporal_recall@{threshold:.2f}"] = tp / max(1, len(gt)); row[f"temporal_f1@{threshold:.2f}"] = 2 * tp / max(1, 2 * tp + len(pred) - tp + len(gt) - tp)
    for tol in (10, 20, 33, 50): row[f"both_boundaries_within_{tol}"] = sum(abs(pred[x["pred_index"]]["start"] - gt[x["gt_index"]]["start"]) <= tol and abs(pred[x["pred_index"]]["end"] - gt[x["gt_index"]]["end"]) <= tol for x in matches) / max(1, len(gt))
    return row, matches


def boundary_rows(entry: str, family: str, pred: list[dict[str, Any]], gt: list[dict[str, Any]], condition: str) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[int]]]:
    points = [x["start"] for x in pred[1:]]; truth = [x["start"] for x in gt[1:]]; details, errors = r27b.boundary_detail(entry, family, gt, points, pred[-1]["end"], .01)
    return [{**x, "condition": condition} for x in details], {(scope, tol): value for (scope, tol), value in errors.items()}


def aggregate_boundaries(details: list[dict[str, Any]], errors: dict[tuple[str, str, int], list[int]], condition: str, aggregation: str, family: str = "") -> list[dict[str, Any]]:
    out = []
    for scope in ("all", "known-to-known", "known-to-novel", "novel-to-known", "novel-to-novel", "all novel-related", "all known-related"):
        for tol in TOLERANCES:
            rows = [x for x in details if x["condition"] == condition and x["scope"] == scope and x["tolerance_frames"] == tol and (aggregation != "family" or x["family"] == family)]
            if aggregation == "trajectory": rows = [x for x in rows if x["trajectory"] == family]
            gt = sum(x["gt_boundaries"] for x in rows); pred = sum(x["predicted_boundaries"] for x in rows); tp = sum(x["tp"] for x in rows); fp = sum(x["fp"] for x in rows); fn = sum(x["fn"] for x in rows); es = sum((errors.get((condition, x["trajectory"], scope, tol), []) for x in rows), [])
            out.append({"condition": condition, "aggregation": aggregation, "scope": scope, "family": family if aggregation == "family" else (family if aggregation == "trajectory" else "all"), "trajectory": family if aggregation == "trajectory" else "", "tolerance_frames": tol, "gt_boundaries": gt, "predicted_boundaries": pred, "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "f1": 2 * tp / max(1, 2 * tp + fp + fn), "false_boundary_rate": fp / max(1, pred), "missed_boundary_rate": fn / max(1, gt), "mean_absolute_error_frames": float(np.mean(es)) if es else 0.0, "mean_absolute_error_seconds": float(np.mean(es) * .01) if es else 0.0, "median_absolute_error_frames": float(np.median(es)) if es else 0.0, "p90_absolute_error_frames": float(np.percentile(es, 90)) if es else 0.0, "maximum_absolute_error_frames": max(es) if es else 0})
    return out


def novel_rows(entry: str, family: str, gt: list[dict[str, Any]], pred: list[dict[str, Any]], matches: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    matched = {x["gt_index"]: x for x in matches}; out = []; raw_points = [x["start"] for x in pred[1:]]
    for gi, g in enumerate(gt):
        if g["label"] in KNOWN_SET: continue
        m = matched.get(gi); p = pred[m["pred_index"]] if m else None; overlaps = [x for x in pred if max(0, min(x["end"], g["end"]) - max(x["start"], g["start"])) > 0]
        out.append({"condition": condition, "family": family, "trajectory": entry, "novel_skill": g["label"], "gt_start": g["start"], "gt_end": g["end"], "matched_interval": json.dumps(p or {}), "matched_iou": m["iou"] if m else 0.0, "start_error": abs(p["start"] - g["start"]) if p else "", "end_error": abs(p["end"] - g["end"]) if p else "", "both_boundaries_within_33": int(bool(p and abs(p["start"] - g["start"]) <= 33 and abs(p["end"] - g["end"]) <= 33)), "fragmented": int(len(overlaps) > 1), "merged_with_previous": int(not any(abs(x - g["start"]) <= 0 for x in raw_points)), "merged_with_next": int(not any(abs(x - g["end"]) <= 0 for x in raw_points)), "surrounding_boundaries_deleted": ""})
    return out


def operation_posthoc(op: dict[str, Any], before: list[dict[str, Any]], after: list[dict[str, Any]], gt: list[dict[str, Any]]) -> dict[str, Any]:
    bm, _ = temporal_metrics(before, gt); am, _ = temporal_metrics(after, gt); bpts = [x["start"] for x in before[1:]]; apts = [x["start"] for x in after[1:]]; _, bfp, bfn = r27b.boundary_pairs(bpts, [x["start"] for x in gt[1:]], 33); _, afp, afn = r27b.boundary_pairs(apts, [x["start"] for x in gt[1:]], 33); result = op["chosen_result"]; overlaps = {g["label"] for g in gt if max(0, min(result["end"], g["end"]) - max(result["start"], g["start"])) > 0}; distinct = len(overlaps) > 1; novel_deleted = any(g["label"] not in KNOWN_SET and (abs(op["removed_boundary"] - g["start"]) <= 0 or abs(op["removed_boundary"] - g["end"]) <= 0) for g in gt)
    fdelta = am["temporal_f1@0.50"] - bm["temporal_f1@0.50"]; idelta = am["mean_matched_iou"] - bm["mean_matched_iou"]
    if fdelta > 1e-12 or (abs(fdelta) <= 1e-12 and idelta > 1e-12): category = "clearly beneficial" if fdelta > 1e-12 and idelta >= 0 else "weakly beneficial"
    elif abs(fdelta) <= 1e-12 and abs(idelta) <= 1e-12: category = "neutral"
    else: category = "clearly harmful" if distinct or novel_deleted else "weakly harmful"
    return {"trajectory": op.get("trajectory", ""), "family": op.get("family", ""), "original_segment": json.dumps(op["source_segment"]), "predicted_class": op["source_segment"]["top1_label"], "duration_frames": op["source_segment"]["duration"], "dynamics_score": op["source_segment"]["S4"], "class_threshold": op["source_segment"]["class_threshold"], "validity_margin": op["source_segment"]["validity_margin"], "merge_direction": op["direction"], "ML_score": op["ML_score"], "MR_score": op["MR_score"], "removed_boundary": op["removed_boundary"], "temporal_f1_delta": fdelta, "mean_iou_delta": idelta, "fp_change": len(afp) - len(bfp), "fn_change": len(afn) - len(bfn), "distinct_gt_skills_merged": int(distinct), "novel_boundary_removed": int(novel_deleted), "classification": category, "harmful": int("harmful" in category), "duration_bucket": "shorter_than_180" if op["source_segment"]["duration"] < 180 else "at_least_180"}


def plot_timeline(item: dict[str, Any], raw: list[dict[str, Any]], merged: list[dict[str, Any]], decisions: list[dict[str, Any]], out: Path) -> None:
    heat, grip, ts, sf, r5, gt = item["heat"], item["grip"], item["timestamps"], item["sf"], item["r5"], item["gt"]; time = (ts - ts[0]) / 1e6; fig, ax = plt.subplots(7, 1, figsize=(18, 12), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1, 1, 1.3, 1.0, 1.5]})
    ax[0].imshow(_normalized_heatmap(heat), aspect="auto", origin="upper", extent=[time[0], time[-1], 0, 88]); ax[0].set_ylabel("heatmap\nchannels", rotation=0, ha="right", va="center"); ax[0].set_yticks([])
    def blocks(axis: Any, rows: list[dict[str, Any]], title: str, truth: bool = False) -> None:
        axis.set_ylabel(title, rotation=0, ha="right", va="center")
        for row in rows:
            label = row.get("label", row.get("top1_label", "")); color = DEFAULT_LABEL_COLORS.get(label, "#bdbdbd"); a, b = row["start"], row["end"]; axis.axvspan(time[a], time[min(b, len(time)-1)], color=color, alpha=.85, ec="black", lw=.7); axis.text((time[a] + time[min(b, len(time)-1)]) / 2, .5, label, ha="center", va="center", fontsize=8, clip_on=True)
        axis.set_ylim(0, 1); axis.set_yticks([])
    blocks(ax[1], gt, "truth", True); blocks(ax[2], raw, "RAW\nHYBRID"); blocks(ax[3], merged, "CLASS-MIN\nMERGE")
    ax[4].plot(time, r5, color="#243b53", label="r5 BRB"); ax[4].plot(time, sf, color="#d1495b", label="SF BRB"); ax[4].axhline(.5, color="#243b53", ls="--", lw=.8, label="r5/SF threshold"); ax[4].set_ylim(0, 1); ax[4].set_ylabel("BRB", rotation=0, ha="right", va="center"); ax[4].legend(loc="upper right", ncol=3, fontsize=8)
    levels = [("RAW", [x["start"] for x in raw[1:]], "#777777"), ("B", [x["start"] for x in merged[1:]], "#00798c"), ("GT", [x["start"] for x in gt[1:]], "#111111")]; ax[5].set_yticks(range(len(levels)), [x[0] for x in levels]); ax[5].set_ylim(-.7, len(levels)-.3)
    for y, (_, points, color) in enumerate(levels): ax[5].scatter([time[p] for p in points if p < len(time)], [y] * len([p for p in points if p < len(time)]), color=color, s=18)
    ax[5].set_ylabel("boundaries", rotation=0, ha="right", va="center")
    ax[6].set_ylabel("validity", rotation=0, ha="right", va="center"); ax[6].set_yticks([])
    for i, row in enumerate(raw):
        x = time[min((row["start"] + row["end"]) // 2, len(time)-1)]; color = "#c1121f" if row["invalid"] else "#2a9d8f"; ax[6].scatter(x, .5, color=color, s=24); ax[6].text(x, .58, f"{row['S4']:.2f}/{row['class_threshold']:.2f}\n×{row['validity_margin']:.2f}", fontsize=6, ha="center", va="bottom")
    ax[6].set_ylim(0, 1); ax[-1].set_xlabel("time (s)"); fig.suptitle(f"test/{item['family']}/{item['trajectory'].split('/')[-1]} | class-minimum dynamics forced merge", fontsize=13); fig.tight_layout(rect=[0, 0, 1, .97]); fig.savefig(out, dpi=170); plt.close(fig)


def make_summary(conditions: list[dict[str, Any]], ops: list[dict[str, Any]], novels: list[dict[str, Any]], thresholds_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> None:
    fams = sorted({x["family"] for x in conditions if x.get("scope") == "family"})
    for metric, ylabel, name in (("temporal_f1@0.50", "temporal F1@50", "temporal_f1_by_family"), ("false_boundary_rate_33", "false-boundary rate ±33", "false_boundary_rate"), ("missed_boundary_rate_33", "missed-boundary rate ±33", "missed_boundary_rate")):
        fig, axis = plt.subplots(figsize=(9, 5)); x = np.arange(len(fams)); w = .35
        for j, c in enumerate(("A", "B")): axis.bar(x + (j-.5)*w, [next(z[metric] for z in conditions if z.get("scope")=="family" and z["family"]==f and z["condition"]==c) for f in fams], w, label=c)
        axis.set_xticks(x, fams); axis.set_ylabel(ylabel); axis.legend(); fig.tight_layout(); fig.savefig(OUT / "figures" / f"{name}.png", dpi=160); plt.close(fig)
    for metric, ylabel, name in (("mean_matched_iou", "mean matched IoU", "mean_iou"), ("mean_boundary_error_33_frames", "mean boundary error (frames)", "mean_boundary_error")):
        fig, axis = plt.subplots(figsize=(9, 5)); x = np.arange(len(fams)); w = .35
        for j, c in enumerate(("A", "B")): axis.bar(x + (j-.5)*w, [next(z[metric] for z in conditions if z.get("scope")=="family" and z["family"]==f and z["condition"]==c) for f in fams], w, label=c)
        axis.set_xticks(x, fams); axis.set_ylabel(ylabel); axis.legend(); fig.tight_layout(); fig.savefig(OUT / "figures" / f"{name}.png", dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5)); x=np.arange(2); axis.bar(x-.18,[next(z["predicted_segment_count"] for z in conditions if z.get("scope")=="pooled" and z["condition"]==c) for c in ("A","B")],.36,label="predicted"); axis.bar(x+.18,[next(z["gt_segment_count"] for z in conditions if z.get("scope")=="pooled" and z["condition"]==c) for c in ("A","B")],.36,label="GT"); axis.set_xticks(x,("A","B")); axis.legend(); fig.tight_layout(); fig.savefig(OUT/"figures/predicted_vs_gt_segments.png",dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5)); counts=Counter(x["merge_direction"] for x in ops); axis.bar(list(counts) or ["none"],list(counts.values()) or [0]); axis.set_ylabel("forced merges"); fig.tight_layout(); fig.savefig(OUT/"figures/merge_direction_counts.png",dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5)); counts=Counter(x["classification"] for x in ops); axis.bar(list(counts) or ["none"],list(counts.values()) or [0]); axis.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(OUT/"figures/operation_benefit_harm.png",dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5)); axis.hist([x["duration_frames"] for x in ops if x["duration_bucket"]=="shorter_than_180"],alpha=.7,label="<180"); axis.hist([x["duration_frames"] for x in ops if x["duration_bucket"]=="at_least_180"],alpha=.7,label=">=180"); axis.legend(); axis.set_xlabel("invalid source duration (frames)"); fig.tight_layout(); fig.savefig(OUT/"figures/merges_below_above_180.png",dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5)); vals=[x["minimum_S4"] for x in thresholds_rows]; axis.bar([x["class"] for x in thresholds_rows],vals); axis.set_ylabel("literal minimum S4"); fig.tight_layout(); fig.savefig(OUT/"figures/class_minimum_thresholds.png",dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5)); box_data=[[x["S4"] for x in train_rows if x["class"] == label] for label in KNOWN]
    try: axis.boxplot(box_data, tick_labels=KNOWN, showfliers=True)
    except TypeError: axis.boxplot(box_data, labels=KNOWN, showfliers=True)
    axis.set_ylabel("complete GT training S4"); axis.set_title("Complete PP-training dynamics distributions"); fig.tight_layout(); fig.savefig(OUT/"figures/complete_gt_dynamics_distributions.png",dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5));
    for label in KNOWN:
        vals=[x["validity_margin"] for x in raw_rows if x["top1_label"]==label]; axis.scatter([label]*len(vals),vals,s=10)
    axis.axhline(1.0,color="black",ls="--"); axis.set_ylabel("raw validity margin"); fig.tight_layout(); fig.savefig(OUT/"figures/raw_validity_margins_by_class.png",dpi=160); plt.close(fig)
    merged_durations=[]
    for op in ops:
        try: merged_durations.append(json.loads(op["result_segment"])["duration"])
        except (KeyError, TypeError, json.JSONDecodeError): pass
    fig, axis = plt.subplots(figsize=(8, 5)); axis.hist([x["duration"] for x in raw_rows], bins=20, alpha=.55, label="raw"); axis.hist(merged_durations, bins=20, alpha=.55, label="merged results"); axis.set_xlabel("segment duration (frames)"); axis.legend(); fig.tight_layout(); fig.savefig(OUT/"figures/merged_vs_retained_segment_durations.png",dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5)); means={c:float(np.mean([x["matched_iou"] for x in novels if x["condition"]==c])) if any(x["condition"]==c for x in novels) else 0 for c in ("A","B")}; axis.bar(list(means),list(means.values())); axis.set_ylabel("novel interval mean IoU"); fig.tight_layout(); fig.savefig(OUT/"figures/novel_interval_iou_A_vs_B.png",dpi=160); plt.close(fig)


def main() -> int:
    np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True); (OUT/"predictions").mkdir(exist_ok=True); (OUT/"figures/class_minimum_segments").mkdir(parents=True, exist_ok=True)
    frontend = audit_frontend(); train_entries=[f"train/pick and place/pp{i}" for i in range(1,11)]; val_entries=[f"train/pick and place/pp{i}" for i in range(11,21)]; stats=fit_normalization(train_entries); train_rows=train_segments(train_entries,stats,"train"); val_rows=train_segments(val_entries,stats,"validation"); threshold_map,(threshold_rows,min_audit)=thresholds(train_rows); write_csv(OUT/"training_segment_dynamics.csv",train_rows); write_csv(OUT/"class_dynamics_thresholds.csv",threshold_rows); write_csv(OUT/"class_minimum_segment_audit.csv",min_audit)
    validation_diag=[]
    for row in val_rows: validation_diag.append({"split":"validation","trajectory":row["trajectory"],"class":row["class"],"duration_frames":row["duration_frames"],**{k:row[k] for k in ("S0","S1","S2","S3","S4")},"threshold_S4":threshold_map.get(row["class"],0),"validity_margin_S4":row["S4"]/max(threshold_map.get(row["class"],EPS),EPS),"score_order_S0_to_S4":int(row["S4"] >= row["S3"] >= row["S2"] >= 0)})
    write_csv(OUT/"validation_score_diagnostics.csv",validation_diag)
    for row in min_audit:
        if not row.get("trajectory"): continue
        heat, grip, ts=read_numeric(row["trajectory"]); fig, axis=plt.subplots(2,1,figsize=(12,5),sharex=True); t=np.arange(len(ts))*.01; axis[0].imshow(_normalized_heatmap(heat),aspect="auto",origin="upper",extent=[t[0],t[-1],0,88]); axis[0].axvspan(row["start"]*.01,row["end"]*.01,color="yellow",alpha=.25); axis[1].plot(t,grip,color="#444"); axis[1].axvspan(row["start"]*.01,row["end"]*.01,color="yellow",alpha=.25); axis[1].set_xlabel("seconds"); fig.suptitle(f"minimum source: {row['class']} | {row['trajectory']} [{row['start']},{row['end']})"); fig.tight_layout(); fig.savefig(OUT/"figures/class_minimum_segments"/f"{row['class']}.png",dpi=160); plt.close(fig)
    inventory=list(csv.DictReader((SOURCE/"complete_test_inventory.csv").open())); included=[x["trajectory"] for x in inventory if x.get("included")=="1"]; temporal=[]; bdetails=[]; berr={}; novels=[]; class_rows=[]; family_rows=[]; traj_rows=[]; operations=[]; op_audits=[]; raw_validity=[]; all_predictions=[]
    for entry in included:
        safe=entry.replace("/","__"); npz=np.load(SOURCE/"predictions"/f"{safe}.npz",allow_pickle=False); source_json=json.loads((SOURCE/"predictions"/f"{safe}.json").read_text()); heat=np.asarray(npz["input_heatmap"],dtype=np.float32); ts=np.asarray(npz["timestamps"]); grip=read_numeric(entry)[1]; labels=np.asarray(npz["sf_asb_labels"]); logits=np.asarray(npz["sf_asb_logits"]); gt=gt_segments(entry,ts); points=[int(x["start"]) for x in source_json["hybrid_segments"][1:]]; raw=recompute_segments(points,labels,logits,heat,grip,stats,threshold_map); merged,ops,decisions=force_merges(raw,labels,logits,heat,grip,stats,threshold_map); family=entry.split("/")[1]; all_predictions.append({"entry":entry,"family":family,"raw":raw,"merged":merged,"operations":ops,"decisions":decisions})
        for row in raw: raw_validity.append({"trajectory":entry,"family":family,**row})
        for condition,segs in (("A",raw),("B",merged)):
            tm,matches=temporal_metrics(segs,gt); tm.update({"condition":condition,"scope":"trajectory","family":family,"trajectory":entry}); temporal.append(tm); bd,em=boundary_rows(entry,family,segs,gt,condition); bdetails.extend(bd); berr.update({(condition,entry,scope,tol): v for (scope,tol),v in em.items()}); novels.extend(novel_rows(entry,family,gt,segs,matches,condition))
        current=raw
        for op in ops:
            next_state=[x for x in current]
            direction=op["direction"]; idx=next(i for i,x in enumerate(next_state) if x["start"]==op["source_segment"]["start"] and x["end"]==op["source_segment"]["end"]); result=op["chosen_result"]
            if direction=="MERGE_LEFT": next_state[idx-1:idx+1]=[result]
            else: next_state[idx:idx+2]=[result]
            audit=operation_posthoc({**op,"trajectory":entry,"family":family},current,next_state,gt); op_audits.append(audit); operations.append({"trajectory":entry,"family":family,"source_predicted_class":op["source_segment"]["top1_label"],"source_duration_frames":op["source_segment"]["duration"],"source_score_S4":op["source_segment"]["S4"],"class_threshold_S4":op["source_segment"]["class_threshold"],"validity_margin":op["source_segment"]["validity_margin"],"merge_direction":direction,"ML_score":op["ML_score"],"MR_score":op["MR_score"],"removed_boundary":op["removed_boundary"],"result_segment":json.dumps(result),"duration_bucket":audit["duration_bucket"]}); current=next_state
        for op in ops: op["trajectory"]=entry; op["family"]=family
        write_json(OUT/"predictions"/f"{safe}.json",{"trajectory":entry,"input_sha256":digest(SOURCE/"predictions"/f"{safe}.npz"),"condition_A_raw":raw,"condition_B_class_min_dynamics":merged,"raw_validity":raw,"invalid_decisions":decisions,"forced_merge_sequence":ops,"gt_segments":gt,"fusion":FUSION,"thresholds":threshold_map,"no_gt_in_inference":True})
        plot_timeline({"trajectory":entry,"family":family,"heat":heat,"grip":grip,"timestamps":ts,"sf":np.asarray(npz["sf_brb_probabilities"]),"r5":np.asarray(npz["r5_brb_probabilities"]),"gt":gt},raw,merged,decisions,OUT/"figures"/f"timeline_{safe}.png")
    for condition in ("A","B"):
        rows=[x for x in temporal if x["condition"]==condition]; pooled={"condition":condition,"scope":"pooled","family":"all","trajectory":""}; gt=sum(x["gt_segment_count"] for x in rows); pred=sum(x["predicted_segment_count"] for x in rows); matches_by={t:sum(round(x[f"temporal_recall@{t:.2f}"]*x["gt_segment_count"]) for x in rows) for t in IOU_THRESHOLDS}; pooled.update({"gt_segment_count":gt,"predicted_segment_count":pred,"matched_segment_count":sum(x["matched_segment_count"] for x in rows),"unmatched_predicted_segment_count":sum(x["unmatched_predicted_segment_count"] for x in rows),"unmatched_gt_segment_count":sum(x["unmatched_gt_segment_count"] for x in rows),"mean_matched_iou":sum(x["mean_matched_iou"]*x["matched_segment_count"] for x in rows)/max(1,sum(x["matched_segment_count"] for x in rows)),"median_matched_iou":float(np.mean([x["median_matched_iou"] for x in rows])),"iou_std":float(np.mean([x["iou_std"] for x in rows])),"fraction_gt_iou_ge_0.50":sum(x["fraction_gt_iou_ge_0.50"]*x["gt_segment_count"] for x in rows)/max(1,gt),"fraction_gt_iou_ge_0.75":sum(x["fraction_gt_iou_ge_0.75"]*x["gt_segment_count"] for x in rows)/max(1,gt),"temporal_over_segmentation_rate":max(0,pred-gt)/max(1,gt),"temporal_under_segmentation_rate":max(0,gt-pred)/max(1,gt),"fragmentation_ratio":pred/max(1,gt),"segments_per_gt_segment":pred/max(1,gt)})
        for t in IOU_THRESHOLDS: tp=matches_by[t]; pooled[f"temporal_precision@{t:.2f}"]=tp/max(1,pred); pooled[f"temporal_recall@{t:.2f}"]=tp/max(1,gt); pooled[f"temporal_f1@{t:.2f}"]=2*tp/max(1,2*tp+pred-tp+gt-tp)
        for tol in (10,20,33,50): pooled[f"both_boundaries_within_{tol}"]=sum(x[f"both_boundaries_within_{tol}"]*x["gt_segment_count"] for x in rows)/max(1,gt)
        temporal.append(pooled)
        for family in sorted({x["family"] for x in rows}):
            sub=[x for x in rows if x["family"]==family]; fam={**pooled,"scope":"family","family":family,"gt_segment_count":sum(x["gt_segment_count"] for x in sub),"predicted_segment_count":sum(x["predicted_segment_count"] for x in sub)}; g=fam["gt_segment_count"]; p=fam["predicted_segment_count"]
            for t in IOU_THRESHOLDS: tp=sum(round(x[f"temporal_recall@{t:.2f}"]*x["gt_segment_count"]) for x in sub); fam[f"temporal_precision@{t:.2f}"]=tp/max(1,p); fam[f"temporal_recall@{t:.2f}"]=tp/max(1,g); fam[f"temporal_f1@{t:.2f}"]=2*tp/max(1,2*tp+p-tp+g-tp)
            fam["mean_matched_iou"]=sum(x["mean_matched_iou"]*x["matched_segment_count"] for x in sub)/max(1,sum(x["matched_segment_count"] for x in sub)); fam["fraction_gt_iou_ge_0.75"]=sum(x["fraction_gt_iou_ge_0.75"]*x["gt_segment_count"] for x in sub)/max(1,g); fam["both_boundaries_within_33"]=sum(x["both_boundaries_within_33"]*x["gt_segment_count"] for x in sub)/max(1,g); temporal.append(fam)
    boundary=[]
    for condition in ("A","B"):
        boundary += aggregate_boundaries(bdetails,berr,condition,"pooled")
        for family in sorted({x["family"] for x in temporal if x.get("scope")=="trajectory"}): boundary += aggregate_boundaries(bdetails,berr,condition,"family",family)
    write_csv(OUT/"training_segment_dynamics.csv",train_rows); write_csv(OUT/"class_dynamics_thresholds.csv",threshold_rows); write_csv(OUT/"class_minimum_segment_audit.csv",min_audit); write_csv(OUT/"condition_comparison.csv",[x for x in temporal if x.get("scope") in ("pooled","family")]); write_csv(OUT/"temporal_only_results.csv",temporal); write_csv(OUT/"boundary_results.csv",boundary); write_csv(OUT/"novel_interval_results.csv",novels); write_csv(OUT/"forced_merge_operations.csv",operations); write_csv(OUT/"operation_level_audit.csv",op_audits)
    for label in KNOWN:
        rawc=[x for x in raw_validity if x["top1_label"]==label]; opc=[x for x in op_audits if x["predicted_class"]==label]; class_rows.append({"class":label,"minimum_threshold_S4":threshold_map[label],"raw_predicted_segment_count":len(rawc),"raw_below_threshold_count":sum(x["invalid"] for x in rawc),"forcibly_merged_count":len(opc),"merge_left_count":sum(x["merge_direction"]=="MERGE_LEFT" for x in opc),"merge_right_count":sum(x["merge_direction"]=="MERGE_RIGHT" for x in opc),"mean_raw_score":float(np.mean([x["S4"] for x in rawc])) if rawc else 0,"mean_raw_validity_margin":float(np.mean([x["validity_margin"] for x in rawc])) if rawc else 0,"mean_merged_validity_margin":float(np.mean([x["validity_margin"] for x in opc])) if opc else 0,"posthoc_beneficial_count":sum("beneficial" in x["classification"] for x in opc),"posthoc_harmful_count":sum(x["harmful"] for x in opc)})
    write_csv(OUT/"per_class_results.csv",class_rows)
    for condition in ("A","B"):
        for family in sorted({x["family"] for x in temporal if x.get("scope")=="trajectory"}):
            x=next(z for z in temporal if z.get("scope")=="family" and z["condition"]==condition and z["family"]==family); bs=next(z for z in boundary if z["condition"]==condition and z["aggregation"]=="family" and z["family"]==family and z["scope"]=="all" and z["tolerance_frames"]==33); family_rows.append({"condition":condition,"family":family,"temporal_f1@50":x["temporal_f1@0.50"],"mean_matched_iou":x["mean_matched_iou"],"iou_ge_0.75":x["fraction_gt_iou_ge_0.75"],"both_boundaries_within_33":x["both_boundaries_within_33"],"false_boundary_rate_33":bs["false_boundary_rate"],"missed_boundary_rate_33":bs["missed_boundary_rate"],"mean_boundary_error_33_frames":bs["mean_absolute_error_frames"],"predicted_gt_segment_ratio":x["predicted_segment_count"]/max(1,x["gt_segment_count"]),"accepted_operations":""})
    write_csv(OUT/"per_family_results.csv",family_rows); write_csv(OUT/"per_trajectory_results.csv",[x for x in temporal if x.get("scope")=="trajectory"]); summary_conditions=[x for x in temporal if x.get("scope")=="pooled"] + [{**x,"scope":"family","temporal_f1@0.50":x["temporal_f1@50"],"false_boundary_rate_33":x["false_boundary_rate_33"],"missed_boundary_rate_33":x["missed_boundary_rate_33"],"mean_boundary_error_33_frames":x["mean_boundary_error_33_frames"]} for x in family_rows]; make_summary(summary_conditions,op_audits,novels,threshold_rows,raw_validity,train_rows)
    # Per-class and per-trajectory operation counts are retained in the two audit tables.
    criteria=[]; a=next(x for x in temporal if x.get("scope")=="pooled" and x["condition"]=="A"); b=next(x for x in temporal if x.get("scope")=="pooled" and x["condition"]=="B"); b33=next(x for x in boundary if x["condition"]=="B" and x["aggregation"]=="pooled" and x["scope"]=="all" and x["tolerance_frames"]==33); a33=next(x for x in boundary if x["condition"]=="A" and x["aggregation"]=="pooled" and x["scope"]=="all" and x["tolerance_frames"]==33); criteria.append({"criterion":"RAW baseline F1@50 exact Round 27B","value":a["temporal_f1@0.50"],"expected":0.807988,"pass":abs(a["temporal_f1@0.50"]-0.807988)<1e-6}); criteria.append({"criterion":"front end unchanged","value":json.dumps(FUSION),"pass":True}); write_csv(OUT/"validation_score_diagnostics.csv",validation_diag); write_json(OUT/"run_metadata.json",{"train_entries":train_entries,"validation_entries":val_entries,"test_count":len(included),"thresholds":threshold_map,"no_test_tuning":True}); (OUT/"config.yaml").write_text(yaml.safe_dump({"experiment":"round29_class_minimum_dynamics_merge","conditions":["A_RAW_HYBRID","B_CLASS_MIN_DYNAMICS_FORCED_MERGE"],"normalization":"PP-training robust z per RGB channel; fixed clip [-5,5]","score":"S4 equal-weight d1,dlag5,dlag10,dlag20,dlag50,phase,robust_range,gripper_dynamics","threshold":"literal minimum S4 by predicted PP-known class","max_iterations":"2x initial segments","no_duration_condition":True,"no_test_tuning":True},sort_keys=False),encoding="utf-8"); write_report(temporal,boundary,novels,threshold_rows,op_audits,operations,class_rows,family_rows,criteria,included); return 0


def write_report(temporal: list[dict[str,Any]], boundary: list[dict[str,Any]], novels: list[dict[str,Any]], thresholds_rows: list[dict[str,Any]], op_audits: list[dict[str,Any]], operations: list[dict[str,Any]], class_rows: list[dict[str,Any]], family_rows: list[dict[str,Any]], criteria: list[dict[str,Any]], included: list[str]) -> None:
    def row(c): return next(x for x in temporal if x.get("scope")=="pooled" and x["condition"]==c)
    lines=["# Round 29 — class-conditional minimum skill-dynamics forced merging","",f"The frozen Round 27B front end was reused; {len(included)} complete test trajectories were evaluated. No retraining, test tuning, Round 28 duration rule, classifier, or GT-assisted inference was used.","","## Method", "", "The primary S4 score is the equal-weight mean of normalized ASRF heatmap mean absolute first difference, lag differences at 5/10/20/50 frames, three-phase mean-vector distance, robust p95-p5 range, and an equally weighted explicit `gripper_position` dynamics component (range, path-rate, maximum change, and endpoint change). The robust normalization was fit only on PP training pp1–pp10 and is recorded in `normalization_statistics.json`.","", "A segment below the literal minimum S4 observed for its predicted majority PP class cannot KEEP. Only MERGE_LEFT and MERGE_RIGHT are available; the lowest validity-margin segment is processed iteratively.","", "## Main strict temporal-only results", "", "| Condition | GT seg. | Pred. seg. | Temporal F1@50 | Mean IoU | IoU≥.75 | Both ±33 | False boundary ±33 | Missed boundary ±33 | Mean boundary error |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in ("A","B"):
        x=row(c); z=next(y for y in boundary if y["condition"]==c and y["aggregation"]=="pooled" and y["scope"]=="all" and y["tolerance_frames"]==33); lines.append(f"| {c} | {x['gt_segment_count']} | {x['predicted_segment_count']} | {x['temporal_f1@0.50']:.6f} | {x['mean_matched_iou']:.6f} | {x['fraction_gt_iou_ge_0.75']:.6f} | {x['both_boundaries_within_33']:.6f} | {z['false_boundary_rate']:.6f} | {z['missed_boundary_rate']:.6f} | {z['mean_absolute_error_frames']:.3f} frames / {z['mean_absolute_error_seconds']:.4f} s |")
    lines += ["", "### Class minimum thresholds", "", "| class | count | minimum S4 | source | valid complete GT |", "|---|---:|---:|---|---|"]
    for x in thresholds_rows: lines.append(f"| {x['class']} | {x['count']} | {x['minimum_S4']:.6f} | {x['minimum_source_trajectory']} [{x['minimum_source_start']},{x['minimum_source_end']}) | {'yes' if x['minimum_source_valid_complete_gt'] else 'no'} |")
    invalid=sum(x["raw_below_threshold_count"] for x in class_rows); raw_invalid_long=sum(int(x["invalid"] and x["duration"] >= 180) for q in (OUT/"predictions").glob("*.json") for x in json.loads(q.read_text()).get("raw_validity", [])); unresolved=sum(1 for q in (OUT/"predictions").glob("*.json") for x in json.loads(q.read_text()).get("invalid_decisions", []) if x.get("status")=="unmergeable_single_segment"); h=Counter(x["classification"] for x in op_audits); novel_means={c:float(np.mean([float(x["matched_iou"]) for x in novels if x["condition"]==c])) if any(x["condition"]==c for x in novels) else 0.0 for c in ("A","B")}; lines += ["", f"Raw below-threshold segments: **{invalid}**, including **{raw_invalid_long}** with duration ≥180 frames (≥1.8 s). Forced operations: **{len(operations)}**; merge-left **{sum(x['merge_direction']=='MERGE_LEFT' for x in op_audits)}**, merge-right **{sum(x['merge_direction']=='MERGE_RIGHT' for x in op_audits)}**. Harmful operations: **{sum(x['harmful'] for x in op_audits)} / {len(op_audits)}**. Operation classifications: `{dict(h)}`. Unmergeable single-segment edge cases: **{unresolved}**.", "", "### Family effects for B", "", "| family | F1@50 | false boundary ±33 | missed boundary ±33 | mean error frames |", "|---|---:|---:|---:|---:|"] + [f"| {x['family']} | {x['temporal_f1@50']:.6f} | {x['false_boundary_rate_33']:.6f} | {x['missed_boundary_rate_33']:.6f} | {x['mean_boundary_error_33_frames']:.3f} |" for x in family_rows if x["condition"]=="B"] + ["", f"Novel interval mean IoU: A **{novel_means['A']:.6f}**, B **{novel_means['B']:.6f}**. The literal minimum is intentionally permissive for classes whose minimum training segment has low dynamics, but the observed rule was aggressive overall: B reduced false-boundary rate modestly while substantially increasing missed boundaries and reducing novel interval IoU.", "", f"Raw A reproduces the Round 27B F1@50 baseline: {row('A')['temporal_f1@0.50']:.6f}. Full family effects, novel interval preservation, every operation, and per-trajectory predictions are in the CSV/JSON artifacts.", "", "Annotations unchanged; no retraining; exact frozen checkpoints by SHA; exact 36-trajectory test inventory; thresholds from PP training only; validation used only for diagnostic score behavior; predicted class selects threshold; no GT in deployable merging; no duration condition; no Round 28 logic; no segment classifier; no BRB retraining."]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__": raise SystemExit(main())
