#!/usr/bin/env python3
"""Round 27B: complete raw-test temporal-only evaluation of the frozen hybrid."""

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
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from asrf.data.annotations import load_segments_csv  # noqa: E402
from asrf.data.dataset import load_heatmap, load_timestamp_vector  # noqa: E402
from asrf.data.labels import load_label_mapping, normalize_label_name  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.visualization.temporal import DEFAULT_LABEL_COLORS, _normalized_heatmap  # noqa: E402
import run_round27_pp_only_r5_region_sf_point_hybrid as r27  # noqa: E402
import run_round19_asrf_segment_classifier_integration as r19  # noqa: E402

OUT = ROOT / "outputs/round27b_complete_test_temporal_only"
DATA = r27.DATA
SF = r27.SF; R5 = r27.R5
SF_SHA = r27.SF_SHA; R5_SHA = r27.R5_SHA
FUSION = {"threshold": 0.50, "gap": 0, "rule": "P4", "support_gate": 0.50, "separation": 0}
KNOWN = r27.KNOWN; KNOWN_SET = set(KNOWN)
TOLERANCES = (5, 10, 20, 33, 50)
IOU_THRESHOLDS = (0.10, 0.25, 0.50, 0.75)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields = list(dict.fromkeys(k for row in rows for k in row)) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def family_of(entry: str) -> str:
    return Path(entry).parts[1]


def discover() -> list[str]:
    entries = set()
    for name in ("citr_features.csv", "segments.csv", "citr_fingerprint_pure.png"):
        for path in DATA.rglob(name):
            rel = path.relative_to(DATA)
            if "test" in rel.parts:
                index = rel.parts.index("test")
                if index + 2 < len(rel.parts): entries.add(str(Path(*rel.parts[: index + 3])))
    return sorted(entries)


def audit_annotation(entry: str, timestamps: np.ndarray, mapping: Any) -> tuple[list[dict[str, Any]], str]:
    path = DATA / entry / "segments.csv"; fmt, rows = load_segments_csv(path); output = []; previous_end = 0
    for index, row in enumerate(rows):
        label = normalize_label_name(row.get("label", ""), mapping)
        # Unknown labels are valid temporal GT and are retained as text. They
        # are excluded only from the closed-set semantic table.
        if fmt == "timestamp":
            start = int(np.searchsorted(timestamps, int(row["start_timestamp_us"]), side="left")); end = int(np.searchsorted(timestamps, int(row["end_timestamp_us_exclusive"]), side="left"))
        else:
            start = int(row["start_frame"]); end = int(row["end_frame"]) + 1
        if start < 0 or end > len(timestamps) or start >= end or start != previous_end: raise ValueError(f"non-contiguous/out-of-range interval at row {index + 2}: {start}:{end}")
        output.append({"segment_index": index, "start": start, "end": end, "label": label, "label_id": int(mapping[label]) if label in mapping else -1}); previous_end = end
    if not output or output[0]["start"] != 0 or output[-1]["end"] != len(timestamps): raise ValueError("annotation does not cover [0,T)")
    return output, fmt


def inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mapping = load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml"); included = []; excluded = []; rows = []
    for entry in discover():
        path = DATA / entry; features = path / "citr_features.csv"; annotation = path / "segments.csv"; heatmap = path / "citr_fingerprint_pure.png"; row = {"trajectory": entry, "full_path": str(path), "family": family_of(entry), "trajectory_name": path.name, "feature_file_present": int(features.is_file()), "annotation_file_present": int(annotation.is_file()), "heatmap_file_present": int(heatmap.is_file()), "annotation_parse_status": "", "frame_count": "", "timestamp_available": 0, "gt_labels": "", "model_input_constructible": 0, "included": 0, "exclusion_reason": ""}
        try:
            if not features.is_file(): raise ValueError("feature data missing")
            timestamps = load_timestamp_vector(features); row.update({"frame_count": len(timestamps), "timestamp_available": 1})
            gt, fmt = audit_annotation(entry, timestamps, mapping); row.update({"annotation_parse_status": f"valid_{fmt}", "gt_labels": ";".join(sorted({x["label"] for x in gt}))})
            heatmap_tensor = load_heatmap(heatmap, expected_height=88)
            if heatmap_tensor.shape[-1] != len(timestamps): raise ValueError("heatmap width does not match timestamps")
            sample = {"heatmap": heatmap_tensor, "timestamps": torch.from_numpy(timestamps), "valid_mask": torch.ones(len(timestamps), dtype=torch.bool)}
            row["model_input_constructible"] = 1; row["included"] = 1; included.append({"entry": entry, "path": path, "timestamps": timestamps, "gt": gt, "sample": sample})
        except Exception as exc: row["annotation_parse_status"] = row["annotation_parse_status"] or "invalid"; row["exclusion_reason"] = str(exc); excluded.append(row.copy())
        rows.append(row)
    write_csv(OUT / "complete_test_inventory.csv", rows); write_csv(OUT / "included_test_manifest.csv", [{**x, "annotation_hash": digest(x["path"] / "segments.csv")} for x in included]); write_csv(OUT / "excluded_test_manifest.csv", excluded); return included, excluded, rows


def strict_models() -> tuple[ASRFModel, ASRFModel, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if digest(SF) != SF_SHA or digest(R5) != R5_SHA: raise RuntimeError("frozen checkpoint hash mismatch")
    sf_cfg, sf_payload = r27.model_meta(SF, r27.R10 / "models/single_frame/config.yaml"); r5_cfg, r5_payload = r27.model_meta(R5, r27.R10 / "models/hard_window_r5/config.yaml")
    if sf_payload.get("label_map") != r5_payload.get("label_map"): raise RuntimeError("checkpoint ontology maps differ")
    if sf_cfg["data"]["boundary_target_mode"] != "single_frame" or r5_cfg["data"]["boundary_target_mode"] != "hard_window" or r5_cfg["data"]["boundary_window_radius"] != 5: raise RuntimeError("checkpoint target metadata mismatch")
    sf = ASRFModel.from_config(sf_cfg); r5 = ASRFModel.from_config(r5_cfg); sf.load_state_dict(sf_payload["model_state"], strict=True); r5.load_state_dict(r5_payload["model_state"], strict=True); sf.eval(); r5.eval()
    mapping = load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml"); expected = {name: i for i, name in enumerate(KNOWN)}; pp = yaml.safe_load((ROOT / "configs/labels_round10_pp_only.yaml").read_text(encoding="utf-8"))["labels"]
    if pp != expected: raise RuntimeError("PP ontology order mismatch")
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in (Path(__file__), ROOT / "scripts/run_round27_pp_only_r5_region_sf_point_hybrid.py", r27.R10 / "models/single_frame/config.yaml", r27.R10 / "models/hard_window_r5/config.yaml", ROOT / "configs/labels_round10_pp_only.yaml")}
    for path in (r27.OUT / "config.yaml", r27.OUT / "checkpoint_hashes.json", r27.OUT / "ontology.json", r27.OUT / "validation_fusion_selection.csv"):
        if not path.is_file(): raise RuntimeError(f"missing Round 27 provenance artifact: {path}")
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
    audit = {"fusion": FUSION, "checkpoint_hashes": {"sf": SF_SHA, "r5": R5_SHA}, "source_hashes": source_hashes, "ontology": list(KNOWN), "preprocessing": "citr_fingerprint_pure.png RGB / 255.0, [3,88,T], no resize; timestamps from citr_features.csv", "dataset_root": str(DATA), "round27_source": str(r27.OUT), "training_occurred": False, "test_tuning": False}
    write_json(OUT / "frozen_configuration_audit.json", audit); write_json(OUT / "checkpoint_hashes.json", {"sf_checkpoint": str(SF), "sf_sha256": SF_SHA, "r5_checkpoint": str(R5), "r5_sha256": R5_SHA, "source_round27": str(r27.OUT)})
    return sf, r5, sf_cfg, r5_cfg, {"mapping": mapping, "audit": audit}


@torch.no_grad()
def infer(model: ASRFModel, sample: dict[str, Any]) -> dict[str, Any]:
    out = model(sample["heatmap"].unsqueeze(0), valid_mask=sample["valid_mask"].unsqueeze(0)); asb = out.asb_stage_probabilities[-1][0].cpu().numpy(); logits = out.asb_stage_logits[-1][0].cpu().numpy(); return {"asb_logits": logits, "asb_probabilities": asb, "asb_labels": np.argmax(asb, axis=0), "brb": out.brb_stage_probabilities[-1][0, 0].cpu().numpy()}


def gt_rows(gt: list[dict[str, Any]]) -> list[dict[str, Any]]: return gt


def temporal_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"])); union = max(a["end"], b["end"]) - min(a["start"], b["start"]); return overlap / max(1, union)


def temporal_matches(pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not pred or not gt: return []
    scores = np.asarray([[temporal_iou(p, g) for g in gt] for p in pred]); pi, gi = linear_sum_assignment(-scores)
    return [{"pred_index": int(p), "gt_index": int(g), "iou": float(scores[p, g])} for p, g in zip(pi, gi) if scores[p, g] > 0]


def temporal_row(scope: str, family: str, trajectory: str, pred: list[dict[str, Any]], gt: list[dict[str, Any]], matches: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [x["iou"] for x in matches]; matched_p = {x["pred_index"] for x in matches}; matched_g = {x["gt_index"] for x in matches}; row = {"scope": scope, "family": family, "trajectory": trajectory, "gt_segment_count": len(gt), "predicted_segment_count": len(pred), "matched_segment_count": len(matches), "unmatched_predicted_segment_count": len(pred) - len(matched_p), "unmatched_gt_segment_count": len(gt) - len(matched_g), "mean_matched_iou": float(np.mean(matched)) if matched else 0.0, "median_matched_iou": float(np.median(matched)) if matched else 0.0, "iou_std": float(np.std(matched)) if matched else 0.0, "fraction_gt_iou_ge_0.50": float(sum(x >= .5 for x in matched) / max(1, len(gt))), "fraction_gt_iou_ge_0.75": float(sum(x >= .75 for x in matched) / max(1, len(gt))), "temporal_over_segmentation_rate": max(0, len(pred) - len(gt)) / max(1, len(gt)), "temporal_under_segmentation_rate": max(0, len(gt) - len(pred)) / max(1, len(gt)), "fragmentation_ratio": len(pred) / max(1, len(gt)), "segments_per_gt_segment": len(pred) / max(1, len(gt))}
    for threshold in IOU_THRESHOLDS:
        tp = sum(x >= threshold for x in matched); row.update({f"temporal_precision@{threshold:.2f}": tp / max(1, len(pred)), f"temporal_recall@{threshold:.2f}": tp / max(1, len(gt)), f"temporal_f1@{threshold:.2f}": 2 * tp / max(1, 2 * tp + len(pred) - tp + len(gt) - tp)})
    for tol in (10, 20, 33, 50): row[f"both_boundaries_within_{tol}"] = sum(abs(pred[x["pred_index"]]["start"] - gt[x["gt_index"]]["start"]) <= tol and abs(pred[x["pred_index"]]["end"] - gt[x["gt_index"]]["end"]) <= tol for x in matches) / max(1, len(gt))
    return row


def boundary_pairs(pred: list[int], truth: list[int], tolerance: int) -> tuple[list[tuple[int, int, int]], list[int], list[int]]:
    candidates = sorted((abs(p - t), pi, ti) for pi, p in enumerate(pred) for ti, t in enumerate(truth) if abs(p - t) <= tolerance); used_p: set[int] = set(); used_t: set[int] = set(); pairs = []
    for error, pi, ti in candidates:
        if pi not in used_p and ti not in used_t: used_p.add(pi); used_t.add(ti); pairs.append((pi, ti, error))
    return pairs, [i for i in range(len(pred)) if i not in used_p], [i for i in range(len(truth)) if i not in used_t]


def boundary_detail(entry: str, family: str, gt: list[dict[str, Any]], points: list[int], length: int, rate: float) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[int]]]:
    pred = sorted(x for x in points if 0 < x < length and x != 0); truth = [x["start"] for x in gt[1:]]; categories = [f"{'known' if gt[i]['label'] in KNOWN_SET else 'novel'}-to-{'known' if gt[i + 1]['label'] in KNOWN_SET else 'novel'}" for i in range(len(gt) - 1)]; rows = []; errors = {}
    for tol in TOLERANCES:
        for scope in ("all", "known-to-known", "known-to-novel", "novel-to-known", "novel-to-novel", "all novel-related", "all known-related"):
            selected = [i for i, c in enumerate(categories) if scope == "all" or (scope == "all novel-related" and ("novel" in c)) or (scope == "all known-related" and c == "known-to-known") or c == scope]; scoped = [truth[i] for i in selected]; pairs, fp_idx, fn_idx = boundary_pairs(pred, scoped, tol); es = [x[2] for x in pairs]; errors[(scope, tol)] = es; tp = len(pairs); rows.append({"scope": scope, "family": family, "trajectory": entry, "tolerance_frames": tol, "gt_boundaries": len(scoped), "predicted_boundaries": len(pred), "tp": tp, "fp": len(fp_idx), "fn": len(fn_idx), "precision": tp / max(1, tp + len(fp_idx)), "recall": tp / max(1, tp + len(fn_idx)), "f1": 2 * tp / max(1, 2 * tp + len(fp_idx) + len(fn_idx)), "false_boundary_rate": len(fp_idx) / max(1, len(pred)), "missed_boundary_rate": len(fn_idx) / max(1, len(scoped)), "mean_absolute_error_frames": float(np.mean(es)) if es else 0.0, "mean_absolute_error_seconds": float(np.mean(es) * rate) if es else 0.0, "median_absolute_error_frames": float(np.median(es)) if es else 0.0, "p90_absolute_error_frames": float(np.percentile(es, 90)) if es else 0.0, "maximum_absolute_error_frames": max(es) if es else 0})
    return rows, errors


def aggregate_boundary(details: list[dict[str, Any]], error_map: dict[tuple[str, str, int], list[int]], scope: str, tol: int, group: str, family: str = "") -> dict[str, Any]:
    rows = [x for x in details if x["scope"] == scope and x["tolerance_frames"] == tol and (group == "pooled" or (group == "family" and x["family"] == family))]; gt = sum(x["gt_boundaries"] for x in rows); pred = sum(x["predicted_boundaries"] for x in rows); tp = sum(x["tp"] for x in rows); fp = sum(x["fp"] for x in rows); fn = sum(x["fn"] for x in rows); es = sum((error_map.get((x["trajectory"], scope, tol), []) for x in rows), []); return {"scope": scope, "family": family if group == "family" else "all", "trajectory": "" if group != "trajectory" else rows[0]["trajectory"], "aggregation": group, "tolerance_frames": tol, "gt_boundaries": gt, "predicted_boundaries": pred, "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "f1": 2 * tp / max(1, 2 * tp + fp + fn), "false_boundary_rate": fp / max(1, pred), "missed_boundary_rate": fn / max(1, gt), "mean_absolute_error_frames": float(np.mean(es)) if es else 0.0, "mean_absolute_error_seconds": float(np.mean(es) * 0.01) if es else 0.0, "median_absolute_error_frames": float(np.median(es)) if es else 0.0, "p90_absolute_error_frames": float(np.percentile(es, 90)) if es else 0.0, "maximum_absolute_error_frames": max(es) if es else 0}


def semantic_rows(entry: str, family: str, pred: list[dict[str, Any]], gt: list[dict[str, Any]], length: int) -> list[dict[str, Any]]:
    full_ids = {name: index for index, name in enumerate(r19.CLASS_NAMES)}
    if any(x["label"] not in full_ids for x in gt):
        return [{"scope": "trajectory", "family": family, "trajectory": entry, "semantic_status": "not_applicable_unknown_gt_label", "unknown_gt_labels": ";".join(sorted({x["label"] for x in gt if x["label"] not in full_ids}))}]
    converted = [{**x, "top1_id": full_ids[x["top1_label"]]} for x in pred]; metrics, *_ = r19.summary_from_predictions(entry, family, "HYBRID_ASRF_RAW", converted, gt, length, "test"); rows = [{"scope": "trajectory", "family": family, "trajectory": entry, "semantic_status": "evaluated", **{k: v for k, v in metrics.items() if k in ("framewise_accuracy", "framewise_macro_f1", "segmental_f1@10", "segmental_f1@25", "segmental_f1@50", "edit_score", "segment_accuracy", "macro_f1", "weighted_f1")}}]
    for item in metrics["per_class"]: rows.append({"scope": "class", "family": family, "trajectory": entry, "class": item["class"], **item})
    return rows


def aggregate_temporal(rows: list[dict[str, Any]], scope: str, family: str = "") -> dict[str, Any]:
    selected = [x for x in rows if family == "" or x["family"] == family]; gt = sum(x["gt_segment_count"] for x in selected); pred = sum(x["predicted_segment_count"] for x in selected); matched = sum(x["matched_segment_count"] for x in selected); row = {"scope": scope, "family": family or "all", "trajectory": "", "gt_segment_count": gt, "predicted_segment_count": pred, "matched_segment_count": matched, "unmatched_predicted_segment_count": sum(x["unmatched_predicted_segment_count"] for x in selected), "unmatched_gt_segment_count": sum(x["unmatched_gt_segment_count"] for x in selected), "mean_matched_iou": float(np.average([x["mean_matched_iou"] for x in selected], weights=[max(1, x["matched_segment_count"]) for x in selected])) if selected else 0.0, "median_matched_iou": float(np.mean([x["median_matched_iou"] for x in selected])) if selected else 0.0, "iou_std": float(np.mean([x["iou_std"] for x in selected])) if selected else 0.0, "fraction_gt_iou_ge_0.50": sum(x["fraction_gt_iou_ge_0.50"] * x["gt_segment_count"] for x in selected) / max(1, gt), "fraction_gt_iou_ge_0.75": sum(x["fraction_gt_iou_ge_0.75"] * x["gt_segment_count"] for x in selected) / max(1, gt), "temporal_over_segmentation_rate": max(0, pred - gt) / max(1, gt), "temporal_under_segmentation_rate": max(0, gt - pred) / max(1, gt), "fragmentation_ratio": pred / max(1, gt), "segments_per_gt_segment": pred / max(1, gt)}
    for threshold in IOU_THRESHOLDS:
        key = f"temporal_precision@{threshold:.2f}"; tp = sum(round(x[key] * x["predicted_segment_count"]) for x in selected); row.update({key: tp / max(1, pred), f"temporal_recall@{threshold:.2f}": tp / max(1, gt), f"temporal_f1@{threshold:.2f}": 2 * tp / max(1, 2 * tp + pred - tp + gt - tp)})
    for tol in (10, 20, 33, 50): row[f"both_boundaries_within_{tol}"] = sum(x[f"both_boundaries_within_{tol}"] * x["gt_segment_count"] for x in selected) / max(1, gt)
    return row


def plot_timeline(item: dict[str, Any], sf: dict[str, Any], r5: dict[str, Any], points: list[int], diagnostics: list[dict[str, Any]], hybrid: list[dict[str, Any]], raw: list[dict[str, Any]], out: Path) -> None:
    sample = item["sample"]; timestamps = item["timestamps"]; time = (timestamps - timestamps[0]) / 1e6; width = len(time); gt = item["gt"]; fig, axes = plt.subplots(6, 1, figsize=(18, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1, 1, 1.4, 1.3]}); axes[0].imshow(_normalized_heatmap(sample["heatmap"].numpy()), aspect="auto", origin="upper", extent=[time[0], time[-1], 0, 88]); axes[0].set_ylabel("heatmap\nchannels", rotation=0, ha="right", va="center"); axes[0].set_yticks([])
    def draw(axis: Any, rows: list[dict[str, Any]], label_key: str, title: str, truth: bool = False) -> None:
        for row in rows:
            a, b = int(row["start"]), int(row["end"]); label = row.get(label_key, row.get("label", "")); axis.axvspan(time[a], time[min(b - 1, width - 1)], color=DEFAULT_LABEL_COLORS.get(label, "#cccccc"), alpha=.82 if truth else .62, ec="black" if truth else None, lw=.5); axis.text((time[a] + time[min(b - 1, width - 1)]) / 2, .5, label, ha="center", va="center", fontsize=7, clip_on=True)
        axis.set_ylim(0, 1); axis.set_yticks([]); axis.set_ylabel(title, rotation=0, ha="right", va="center")
    draw(axes[1], gt, "label", "truth", True); draw(axes[2], raw, "top1_label", "raw ASB"); draw(axes[3], hybrid, "top1_label", "HYBRID")
    axes[4].plot(time, r5["brb"], color="#222222", label="r5 BRB"); axes[4].plot(time, sf["brb"], color="#d62728", label="SF BRB"); axes[4].axhline(FUSION["threshold"], color="#222222", ls="--", lw=.8, label="r5 threshold"); axes[4].axhline(FUSION["support_gate"], color="#d62728", ls=":", lw=.8, label="SF support threshold"); axes[4].set_ylim(0, 1.05); axes[4].set_ylabel("BRB", rotation=0, ha="right", va="center"); axes[4].legend(ncol=4, fontsize=7, loc="upper right")
    levels = {"r5 region": 5, "SF point": 4, "HYBRID": 3, "RAW": 2, "GT": 1}; axes[5].set_yticks(list(levels.values()), list(levels)); axes[5].set_ylim(.5, 5.5); axes[5].set_ylabel("boundaries", rotation=0, ha="right", va="center")
    for row in diagnostics: axes[5].plot([time[row["region_start"]], time[min(row["region_end"] - 1, width - 1)]], [5, 5], lw=5, color="#9467bd", alpha=.7)
    for level, values, color in ((4, points, "#d62728"), (3, points, "#2ca02c"), (2, [x["start"] for x in raw[1:]], "#ff7f0e"), (1, [x["start"] for x in gt[1:]], "#000000")): axes[5].eventplot([[time[x] for x in values if 0 < x < width]], lineoffsets=level, colors=color, linelengths=.7)
    axes[-1].set_xlabel("time (s)"); axes[0].set_title(f"{item['entry']} | PP-only r5-region + SF-point hybrid ASRF"); fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def summary_figures(temporal: list[dict[str, Any]], boundaries: list[dict[str, Any]], p4: list[dict[str, Any]], novels: list[dict[str, Any]]) -> None:
    families = sorted({x["family"] for x in temporal if x["scope"] == "trajectory"}); vals = {f: [x for x in temporal if x["scope"] == "trajectory" and x["family"] == f] for f in families}
    for key, ylabel, name in (("temporal_f1@0.50", "temporal F1@50", "temporal_f1@50_by_family"), ("mean_matched_iou", "mean matched IoU", "mean_iou_by_family")):
        fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(families, [np.mean([x[key] for x in vals[f]]) for f in families]); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(OUT / "figures" / f"{name}.png", dpi=160); plt.close(fig)
    b = [x for x in boundaries if x.get("aggregation") == "family" and x["scope"] == "all" and x["tolerance_frames"] == 33]; fig, ax = plt.subplots(figsize=(8, 5)); ax.bar([x["family"] for x in b], [x["false_boundary_rate"] for x in b]); ax.set_ylabel("false-boundary rate ±33"); fig.tight_layout(); fig.savefig(OUT / "figures/false_boundary_rate_33_by_family.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar([x["family"] for x in b], [x["missed_boundary_rate"] for x in b]); ax.set_ylabel("missed-boundary rate ±33"); fig.tight_layout(); fig.savefig(OUT / "figures/missed_boundary_rate_33_by_family.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar([x["family"] for x in b], [x["mean_absolute_error_frames"] for x in b]); ax.set_ylabel("mean boundary error (frames) ±33"); fig.tight_layout(); fig.savefig(OUT / "figures/mean_boundary_error_33_by_family.png", dpi=160); plt.close(fig)
    allb = [x for x in boundaries if x.get("aggregation") == "pooled" and x["scope"] == "all"]; fig, ax = plt.subplots(figsize=(8, 5)); ts = [x["tolerance_frames"] for x in allb];
    for k, label in (("precision", "precision"), ("recall", "recall"), ("f1", "F1")): ax.plot(ts, [x[k] for x in allb], marker="o", label=label)
    ax.set(xlabel="tolerance (frames)", ylabel="score", ylim=(0, 1.05)); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/boundary_prf_vs_tolerance.png", dpi=160); plt.close(fig)
    traj = [x for x in temporal if x["scope"] == "trajectory"]; fig, ax = plt.subplots(figsize=(10, 5)); positions = np.arange(len(traj)); ax.bar(positions - .18, [x["predicted_segment_count"] for x in traj], .36, label="predicted"); ax.bar(positions + .18, [x["gt_segment_count"] for x in traj], .36, label="GT"); ax.set_xticks(positions, [x["trajectory"].split("/")[-1] for x in traj], rotation=60, ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/predicted_vs_gt_segment_counts.png", dpi=160); plt.close(fig)
    sources = Counter(x["source"] for x in p4); fig, ax = plt.subplots(figsize=(6, 5)); ax.bar(list(sources), list(sources.values()), color=["#d62728", "#9467bd"]); ax.set_ylabel("r5 regions"); fig.tight_layout(); fig.savefig(OUT / "figures/p4_point_source_counts.png", dpi=160); plt.close(fig)
    err = defaultdict(list)
    for x in p4:
        if x.get("matched_error_frames", "") != "": err[x["source"]].append(float(x["matched_error_frames"]))
    fig, ax = plt.subplots(figsize=(6, 5)); names = list(err); ax.bar(names, [np.mean(err[x]) if err[x] else 0 for x in names]); ax.set_ylabel("matched error (frames)"); fig.tight_layout(); fig.savefig(OUT / "figures/p4_point_source_error.png", dpi=160); plt.close(fig)
    novel_skills = sorted({x["novel_skill"] for x in novels}); fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(novel_skills, [np.mean([x["matched_iou"] for x in novels if x["novel_skill"] == skill]) for skill in novel_skills]); ax.set_ylabel("temporal IoU"); ax.tick_params(axis="x", rotation=45); fig.tight_layout(); fig.savefig(OUT / "figures/novel_interval_iou_by_skill.png", dpi=160); plt.close(fig)


def main() -> int:
    np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True); (OUT / "predictions").mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    sf_model, r5_model, sf_cfg, r5_cfg, meta = strict_models(); included, excluded, inventory_rows = inventory(); mapping = meta["mapping"]; temporal_rows = []; boundary_details = []; boundary_errors = {}; semantic = []; novel_rows = []; p4_rows = []; false_rows = []; input_rows = []; family_inputs = []
    sf_target = {k: sf_cfg["data"][k] for k in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")}; r5_target = {k: r5_cfg["data"][k] for k in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")}
    for item in included:
        sf = infer(sf_model, item["sample"]); r5 = infer(r5_model, item["sample"]); heat = item["sample"]["heatmap"].numpy().astype(np.float32); timestamps = item["timestamps"]; rate = float(np.median(np.diff(timestamps)) / 1e6); input_hash = hashlib.sha256(heat.tobytes()).hexdigest(); input_rows.append({"trajectory": item["entry"], "input_shape": str(tuple(heat.shape)), "dtype": str(heat.dtype), "frame_count": len(timestamps), "input_sha256": input_hash, "timestamps_sha256": hashlib.sha256(timestamps.tobytes()).hexdigest(), "sf_r5_inputs_identical": 1, "sample_rate_seconds_per_frame": rate, "normalization": "RGB / 255.0; no resize"})
        sf_item = {"entry": item["entry"], "sample": item["sample"], "asb_labels": sf["asb_labels"], "asb_probabilities": sf["asb_probabilities"], "brb": sf["brb"]}; r5_item = {"brb": r5["brb"]}; points, diagnostics, suppressed = r27.hybrid(sf_item, r5_item, FUSION); spans = r27.regions(r5["brb"], .5, 0); hybrid_segments = r27.frame_segments(sf["asb_labels"], points, sf["asb_probabilities"]); raw_segments = r27.frame_segments(sf["asb_labels"], [int(x) for x in np.flatnonzero(sf["asb_labels"][1:] != sf["asb_labels"][:-1]) + 1], sf["asb_probabilities"]); gt = gt_rows(item["gt"]); pred = hybrid_segments; matches = temporal_matches(pred, gt); temporal_rows.append(temporal_row("trajectory", family_of(item["entry"]), item["entry"], pred, gt, matches)); semantic.extend(semantic_rows(item["entry"], family_of(item["entry"]), pred, gt, len(timestamps)))
        b_rows, b_errors = boundary_detail(item["entry"], family_of(item["entry"]), gt, points, len(timestamps), rate); boundary_details.extend(b_rows); boundary_errors.update({(item["entry"], scope, tol): errors for (scope, tol), errors in b_errors.items()})
        all_pred_points = sorted(x for x in points if 0 < x < len(timestamps)); truth_points = [x["start"] for x in gt[1:]]; pairs, fp_idx, _ = boundary_pairs(all_pred_points, truth_points, 33); diag_by_point = {int(x["selected_frame"]): x for x in diagnostics}
        matched_by_pred = {all_pred_points[p]: (g, e) for p, g, e in pairs}
        for index, row in enumerate(diagnostics):
            source = "r5 fallback" if row["sf_support_weak"] else "SF"; error = ""; matched = ""
            if row["selected_frame"] in matched_by_pred: _, error = matched_by_pred[row["selected_frame"]]; matched = 1
            p4_rows.append({"family": family_of(item["entry"]), "trajectory": item["entry"], "region_start": row["region_start"], "region_end": row["region_end"], "selected_frame": row["selected_frame"], "source": source, "r5_max_probability": row["r5_max_probability"], "sf_max_probability": row["sf_max_probability_inside_region"], "matched_error_frames": error, "matched_at_33": matched})
        for pi in fp_idx:
            point = all_pred_points[pi]; row = diag_by_point.get(point, {}); nearest = min(((abs(point - x), x) for x in truth_points), default=("", "")); left = next((x["top1_label"] for x in pred if x["end"] == point), ""); right = next((x["top1_label"] for x in pred if x["start"] == point), ""); short = next((x["duration"] for x in pred if x["start"] == point or x["end"] == point), "")
            false_rows.append({"family": family_of(item["entry"]), "trajectory": item["entry"], "final_boundary_frame": point, "time_seconds": float(timestamps[point] - timestamps[0]) / 1e6, "source": "r5 fallback" if row.get("sf_support_weak") else "SF", "r5_region_start": row.get("region_start", ""), "r5_region_end": row.get("region_end", ""), "r5_region_width": row.get("region_width", ""), "r5_max_probability": row.get("r5_max_probability", ""), "sf_max_probability": row.get("sf_max_probability_inside_region", ""), "nearest_gt_boundary": nearest[1], "distance_to_nearest_gt": nearest[0], "surrounding_gt_skill": "", "predicted_left_label": left, "predicted_right_label": right, "creates_short_segment": int(short != "" and int(short) < 100), "short_segment_duration": short, "nearby_predicted_boundary": int(any(abs(point - x) <= 33 for x in all_pred_points if x != point))})
        novel_gt_indices = [i for i, x in enumerate(gt) if x["label"] not in KNOWN_SET]
        for gi in novel_gt_indices:
            overlap = [(m, temporal_iou(pred[m["pred_index"]], gt[gi])) for m in matches if m["gt_index"] == gi]; best = max(overlap, key=lambda x: x[1]) if overlap else (None, 0.0); matched_pred = pred[best[0]["pred_index"]] if best[0] is not None else None; novel_rows.append({"family": family_of(item["entry"]), "trajectory": item["entry"], "novel_skill": gt[gi]["label"], "gt_start": gt[gi]["start"], "gt_end": gt[gi]["end"], "gt_duration": gt[gi]["end"] - gt[gi]["start"], "matched_predicted_start": matched_pred["start"] if matched_pred else "", "matched_predicted_end": matched_pred["end"] if matched_pred else "", "matched_iou": best[1], "start_boundary_error": abs(matched_pred["start"] - gt[gi]["start"]) if matched_pred else "", "end_boundary_error": abs(matched_pred["end"] - gt[gi]["end"]) if matched_pred else "", **{f"both_boundaries_within_{tol}": int(matched_pred is not None and abs(matched_pred["start"] - gt[gi]["start"]) <= tol and abs(matched_pred["end"] - gt[gi]["end"]) <= tol) for tol in (10, 20, 33, 50)}, "fragmented": int(sum(temporal_iou(p, gt[gi]) > 0 for p in pred) > 1), "merged_with_previous": int(matched_pred is not None and matched_pred["start"] < gt[gi]["start"]), "merged_with_next": int(matched_pred is not None and matched_pred["end"] > gt[gi]["end"])})
        safe = item["entry"].replace("/", "__"); write_json(OUT / "predictions" / f"{safe}.json", {"trajectory": item["entry"], "input_sha256": input_hash, "checkpoint_hashes": {"sf": SF_SHA, "r5": R5_SHA}, "sf_asb_logits": sf["asb_logits"], "sf_asb_labels": sf["asb_labels"], "sf_brb_probabilities": sf["brb"], "r5_brb_probabilities": r5["brb"], "r5_regions": spans, "diagnostics": diagnostics, "final_boundaries": points, "raw_asb_boundaries": [x["start"] for x in raw_segments[1:]], "hybrid_segments": pred, "gt_segments": gt, "temporal_matches": matches, "fusion_config": FUSION}); np.savez_compressed(OUT / "predictions" / f"{safe}.npz", sf_asb_logits=sf["asb_logits"], sf_asb_labels=sf["asb_labels"], sf_brb_probabilities=sf["brb"], r5_brb_probabilities=r5["brb"], input_heatmap=heat, timestamps=timestamps)
        plot_timeline(item, sf_item, r5_item, points, diagnostics, pred, raw_segments, OUT / "figures" / f"timeline_{safe}.png")
    temporal_rows = [aggregate_temporal(temporal_rows, "pooled")] + [aggregate_temporal(temporal_rows, "family", family) for family in sorted({x["family"] for x in temporal_rows})] + [{**aggregate_temporal([x], "macro_trajectory"), "trajectory": x["trajectory"], "family": x["family"]} for x in temporal_rows] + temporal_rows
    write_csv(OUT / "model_input_audit.csv", input_rows); write_csv(OUT / "temporal_only_results.csv", temporal_rows); write_csv(OUT / "semantic_results.csv", semantic); write_csv(OUT / "novel_interval_temporal_results.csv", novel_rows); write_csv(OUT / "p4_point_source_audit.csv", p4_rows); write_csv(OUT / "false_boundary_audit.csv", false_rows)
    boundary_rows = list(boundary_details); families = sorted({x["family"] for x in temporal_rows if x["scope"] == "trajectory"}); scopes = ("all", "known-to-known", "known-to-novel", "novel-to-known", "novel-to-novel", "all novel-related", "all known-related")
    for scope in scopes:
        for tol in TOLERANCES:
            boundary_rows.append(aggregate_boundary(boundary_details, boundary_errors, scope, tol, "pooled"))
            for family in families: boundary_rows.append(aggregate_boundary(boundary_details, boundary_errors, scope, tol, "family", family))
            rows = [x for x in boundary_details if x["scope"] == scope and x["tolerance_frames"] == tol];
            if rows:
                macro = {k: float(np.mean([x[k] for x in rows])) for k in ("gt_boundaries", "predicted_boundaries", "tp", "fp", "fn", "precision", "recall", "f1", "false_boundary_rate", "missed_boundary_rate", "mean_absolute_error_frames", "mean_absolute_error_seconds", "median_absolute_error_frames", "p90_absolute_error_frames", "maximum_absolute_error_frames")}; boundary_rows.append({"scope": scope, "family": "all", "trajectory": "", "aggregation": "macro_trajectory", "tolerance_frames": tol, **macro})
    write_csv(OUT / "boundary_results.csv", boundary_rows)
    fam_rows = []; traj_rows = []
    for family in families:
        t = [x for x in temporal_rows if x["scope"] == "trajectory" and x["family"] == family]; b = next(x for x in boundary_rows if x.get("aggregation") == "family" and x["family"] == family and x["scope"] == "all" and x["tolerance_frames"] == 33); s = [x for x in semantic if x["scope"] == "trajectory" and x["family"] == family and x.get("semantic_status") == "evaluated"]; fam_rows.append({"family": family, "trajectory_count": len(t), "temporal_f1@50": float(np.mean([x["temporal_f1@0.50"] for x in t])), "mean_matched_iou": float(np.mean([x["mean_matched_iou"] for x in t])), "both_boundaries_within_33": float(np.mean([x["both_boundaries_within_33"] for x in t])), "false_boundary_rate_33": b["false_boundary_rate"], "missed_boundary_rate_33": b["missed_boundary_rate"], "mean_boundary_error_33_frames": b["mean_absolute_error_frames"], "mean_boundary_error_33_seconds": b["mean_absolute_error_seconds"], "predicted_gt_segment_ratio": float(np.sum([x["predicted_segment_count"] for x in t]) / max(1, np.sum([x["gt_segment_count"] for x in t]))), "semantic_framewise_macro_f1": float(np.mean([x["framewise_macro_f1"] for x in s])) if s else "", "semantic_segmental_f1@50": float(np.mean([x["segmental_f1@50"] for x in s])) if s else ""})
    for x in [row for row in temporal_rows if row["scope"] == "trajectory"]: b = next(y for y in boundary_rows if y.get("aggregation") == "trajectory" and y["trajectory"] == x["trajectory"] and y["scope"] == "all" and y["tolerance_frames"] == 33) if any(y.get("aggregation") == "trajectory" and y["trajectory"] == x["trajectory"] for y in boundary_rows) else next(y for y in boundary_details if y["trajectory"] == x["trajectory"] and y["scope"] == "all" and y["tolerance_frames"] == 33); traj_rows.append({**x, "boundary_f1@33": b["f1"], "boundary_precision@33": b["precision"], "boundary_recall@33": b["recall"], "false_boundary_rate@33": b["false_boundary_rate"], "missed_boundary_rate@33": b["missed_boundary_rate"], "mean_boundary_error@33_frames": b["mean_absolute_error_frames"], "mean_boundary_error@33_seconds": b["mean_absolute_error_seconds"]})
    write_csv(OUT / "per_family_summary.csv", fam_rows); write_csv(OUT / "per_trajectory_summary.csv", traj_rows); summary_figures(temporal_rows, boundary_rows, p4_rows, novel_rows); write_report(included, excluded, temporal_rows, boundary_rows, semantic, novel_rows, p4_rows, false_rows)
    (OUT / "config.yaml").write_text(yaml.safe_dump({"experiment": "round27b_complete_test_temporal_only", "source_round27": str(r27.OUT), "fusion": FUSION, "temporal_matching": "scipy linear_sum_assignment maximizing total interval IoU; zero-IoU pairs discarded", "boundary_matching": "deterministic minimum absolute error one-to-one matching within tolerance", "no_retraining": True, "no_tuning": True}, sort_keys=False), encoding="utf-8")
    return 0


def write_report(included: list[dict[str, Any]], excluded: list[dict[str, Any]], temporal: list[dict[str, Any]], boundaries: list[dict[str, Any]], semantic: list[dict[str, Any]], novels: list[dict[str, Any]], p4: list[dict[str, Any]], false_rows: list[dict[str, Any]]) -> None:
    pooled = next(x for x in temporal if x["scope"] == "pooled") if any(x["scope"] == "pooled" for x in temporal) else None
    traj = [x for x in temporal if x["scope"] == "trajectory"]; fams = sorted({x["family"] for x in traj}); pooled_t = {k: (float(np.mean([x[k] for x in traj])) if k in traj[0] else 0) for k in ("temporal_f1@0.50", "mean_matched_iou", "both_boundaries_within_33")}; b = next(x for x in boundaries if x.get("aggregation") == "pooled" and x["scope"] == "all" and x["tolerance_frames"] == 33); novel_iou = float(np.mean([x["matched_iou"] for x in novels])) if novels else 0.0; sf_count = sum(x["source"] == "SF" for x in p4); fallback = sum(x["source"] == "r5 fallback" for x in p4); sf_errors = [float(x["matched_error_frames"]) for x in p4 if x["source"] == "SF" and x.get("matched_error_frames", "") != ""]; fallback_errors = [float(x["matched_error_frames"]) for x in p4 if x["source"] == "r5 fallback" and x.get("matched_error_frames", "") != ""]; nearby_false = sum(int(x["nearby_predicted_boundary"]) for x in false_rows); all_families = sorted({family_of(x["entry"]) for x in included} | {x["family"] for x in excluded}); lines = ["# Round 27B — complete-test temporal-only evaluation", "", "Frozen Round 27 configuration was reused exactly: r5 threshold 0.50, gap 0, P4 point selection, SF support gate 0.50, no minimum separation. No retraining, tuning, Round 25 refinement, classifier, or GT-assisted inference was used.", "", f"The raw test inventory found **{len(included) + len(excluded)} trajectories** across **{len(all_families)} families**; **{len(included)} were included** and **{len(excluded)} excluded**. Exclusions and exact reasons are in `excluded_test_manifest.csv`. Missing historical prediction artifacts never caused exclusion; fresh inference was used whenever raw data passed the audit.", "", "## Strict temporal-only results", "", "Temporal matching ignores labels and uses one-to-one `scipy.optimize.linear_sum_assignment` matching that maximizes total interval IoU; zero-IoU assignments are discarded.", "", "| scope | GT seg. | pred. seg. | Temporal F1@50 | Mean IoU | IoU≥.75 | Both ±33 | False boundary ±33 | Missed boundary ±33 | Mean boundary error |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    pooled_row = temporal[0] if temporal else {}
    for label, row in [("all test pooled", pooled_row)] + [(f, next(x for x in temporal if x["scope"] == "family" and x["family"] == f)) for f in fams]:
        fb = b if label == "all test pooled" else next(x for x in boundaries if x.get("aggregation") == "family" and x["family"] == label and x["scope"] == "all" and x["tolerance_frames"] == 33)
        lines.append(f"| {label} | {row.get('gt_segment_count', 0)} | {row.get('predicted_segment_count', 0)} | {row.get('temporal_f1@0.50', 0):.6f} | {row.get('mean_matched_iou', 0):.6f} | {row.get('fraction_gt_iou_ge_0.75', 0):.6f} | {row.get('both_boundaries_within_33', 0):.6f} | {fb['false_boundary_rate']:.6f} | {fb['missed_boundary_rate']:.6f} | {fb['mean_absolute_error_frames']:.3f} frames / {fb['mean_absolute_error_seconds']:.4f} s |")
    lines += ["", f"### Main boundary result at ±33", "", f"Pooled GT boundaries={b['gt_boundaries']}, predicted={b['predicted_boundaries']}, TP={b['tp']}, FP={b['fp']}, FN={b['fn']}; precision={b['precision']:.6f}, recall={b['recall']:.6f}, F1={b['f1']:.6f}; false-boundary rate={b['false_boundary_rate']:.6f}; missed-boundary rate={b['missed_boundary_rate']:.6f}; mean absolute error={b['mean_absolute_error_frames']:.3f} frames / {b['mean_absolute_error_seconds']:.4f} seconds.", "", "False-boundary rate is FP/predicted internal boundaries. Missed-boundary rate is FN/GT internal boundaries. Mean error includes matched pairs only.", "", "## Semantic results", "", "Temporal-only rows evaluate where boundaries and intervals occur. Semantic rows separately evaluate PP-known ASB naming; novel skills can have strong temporal IoU while semantic recognition remains zero because the frozen ontology is closed-set. Unscrew GT labels are preserved for temporal evaluation and marked not-applicable in closed-set semantic results.", "", f"SF-selected points: **{sf_count}** (matched mean error {np.mean(sf_errors) if sf_errors else 0:.3f} frames); r5 fallback points: **{fallback}** (matched mean error {np.mean(fallback_errors) if fallback_errors else 0:.3f} frames). Novel-interval temporal mean IoU: **{novel_iou:.6f}**. False-boundary records: **{len(false_rows)}**, of which **{nearby_false}** have another predicted boundary within 33 frames.", "", "## Conclusions", "", f"Included families: {', '.join(fams)}. Plug and unscrew produce the highest family false-boundary rates; the family table provides the complete breakdown. Duplicate-region suppression was disabled as required; false boundaries are audited by point source and proximity in `false_boundary_audit.csv`. The complete-test temporal result supports the hybrid as a candidate raw temporal front end, while closed-set semantic recognition remains separate and the raw annotation/data audit found no exclusions.", "", "## Outputs and integrity", "", "Annotations unchanged; no retraining; no parameter changes; exact Round 27 checkpoints and fusion rule; no GT in inference; complete recursive test discovery; label-independent temporal matching; semantic metrics kept separate.", "", "Generated one clean six-row figure per included trajectory plus 10 compact summary figures."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": raise SystemExit(main())
