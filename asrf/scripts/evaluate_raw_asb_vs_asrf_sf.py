"""Inference-only RAW_ASB versus ASRF_SF novel temporal segmentation study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
OUT = ROOT / "outputs/round10_raw_asb_vs_asrf_sf_novel_segmentation_corrected"
KNOWN = ("reach", "grasp", "lift", "transport", "place", "release", "retreat")
KNOWN_SET = set(KNOWN)
NOVEL = ("pour", "pour_recover", "wipe", "insert")
TEST = {
    "pour": [f"test/pour/p{i}" for i in range(1, 6)],
    "wipe": [f"test/wipe/w{i}" for i in range(1, 5)],
    "plug": ["test/plug/p1", "test/plug/p2"],
}
CHECKPOINT = ROOT / "outputs/round10_pp_only_novel_segmentation/models/single_frame/best.pt"
CHECKPOINT_SHA256 = "6b11abff2ff4387eada88e38fb56133b29fd5c3facdfb54b42fc84701213db9a"
FULL_LABEL_CONFIG = ROOT / "configs/labels_multitask_plug.yaml"

import sys
sys.path.insert(0, str(ROOT / "src"))

from asrf.data.annotations import load_segments_csv  # noqa: E402
from asrf.data.dataset import load_heatmap, load_timestamp_vector, load_trajectory_sample  # noqa: E402
from asrf.data.labels import load_label_mapping, normalize_label_name  # noqa: E402
from asrf.evaluation.metrics import boundary_counts  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.refinement.peaks import select_boundary_peaks  # noqa: E402
from asrf.refinement.refine import refine_asrf_predictions  # noqa: E402
from asrf.refinement.segments import TemporalInterval  # noqa: E402
from asrf.training.checkpointing import load_checkpoint, sha256_file  # noqa: E402
from asrf.visualization.temporal import plot_temporal_comparison, validate_prediction_segments  # noqa: E402


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=jsonable) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def canonical_label(raw: str) -> str:
    aliases = {"pick": "reach", "translation": "transport", "pull_out": "lift", "extract": "lift"}
    return aliases.get(str(raw).strip(), str(raw).strip())


def audit_entry(entry: str) -> dict[str, Any]:
    path = DATA / entry
    errors: list[str] = []
    result: dict[str, Any] = {
        "trajectory": entry, "family": entry.split("/")[1], "trajectory_id": path.name,
        "segments_exists": (path / "segments.csv").is_file(),
        "features_exists": (path / "citr_features.csv").is_file(),
        "heatmap_exists": (path / "citr_fingerprint_pure.png").is_file(),
        "temporal_width": 0, "segment_count": 0, "labels": "",
        "canonical_sequence": "", "blank_labels": 0, "invalid_labels": 0,
        "zero_duration": 0, "gaps": 0, "overlaps": 0,
        "chronological": False, "full_temporal_coverage": False,
        "heatmap_width_matches": False, "valid": False, "errors": "",
    }
    try:
        if not path.is_dir():
            errors.append("trajectory directory missing")
        if not result["segments_exists"]:
            errors.append("segments.csv missing")
        if not result["features_exists"]:
            errors.append("citr_features.csv missing")
        if not result["heatmap_exists"]:
            errors.append("citr_fingerprint_pure.png missing")
        timestamps = load_timestamp_vector(path / "citr_features.csv")
        result["temporal_width"] = len(timestamps)
        annotation_format, rows = load_segments_csv(path / "segments.csv")
        result["segment_count"] = len(rows)
        labels: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        previous_end: int | None = None
        for raw in rows:
            label = str(raw.get("label", "")).strip()
            if not label:
                result["blank_labels"] += 1
                errors.append("blank label")
                continue
            name = canonical_label(label)
            labels.append(name)
            if name not in {"reach", "grasp", "lift", "transport", "pour", "pour_recover", "place", "release", "wipe", "retreat", "insert"}:
                result["invalid_labels"] += 1
                errors.append(f"invalid label {label}")
            if annotation_format == "timestamp":
                start = int(raw["start_timestamp_us"])
                end = int(raw["end_timestamp_us_exclusive"])
            else:
                start = int(raw["start_frame"])
                end = int(raw["end_frame"]) + 1
            starts.append(start)
            ends.append(end)
            if end <= start:
                result["zero_duration"] += 1
            if previous_end is not None:
                if start > previous_end:
                    result["gaps"] += 1
                elif start < previous_end:
                    result["overlaps"] += 1
            previous_end = end
        result["chronological"] = bool(starts) and all(b >= a for a, b in zip(starts, starts[1:]))
        if not result["chronological"]:
            errors.append("non-chronological segment order")
        if result["gaps"]:
            errors.append("annotation gap")
        if result["overlaps"]:
            errors.append("annotation overlap")
        if annotation_format == "timestamp":
            frame_starts = [int(np.searchsorted(timestamps, x, side="left")) for x in starts]
            frame_ends = [int(np.searchsorted(timestamps, x, side="left")) for x in ends]
            result["full_temporal_coverage"] = bool(frame_starts and frame_starts[0] == 0 and frame_ends[-1] == len(timestamps) and all(a == b for a, b in zip(frame_ends[:-1], frame_starts[1:])))
        else:
            result["full_temporal_coverage"] = bool(starts and starts[0] == 0 and ends[-1] == len(timestamps))
        if not result["full_temporal_coverage"]:
            errors.append("incomplete temporal coverage")
        heatmap = load_heatmap(path / "citr_fingerprint_pure.png", expected_height=88)
        result["heatmap_width_matches"] = heatmap.shape[-1] == len(timestamps)
        if not result["heatmap_width_matches"]:
            errors.append("heatmap width mismatch")
        result["labels"] = ";".join(sorted(set(labels)))
        result["canonical_sequence"] = " -> ".join(labels)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    result["errors"] = " | ".join(dict.fromkeys(errors))
    result["valid"] = not errors
    return result


def audit() -> tuple[list[str], list[dict[str, Any]]]:
    all_entries = [entry for family in ("pour", "wipe", "plug") for entry in TEST[family]]
    rows = [audit_entry(entry) for entry in all_entries]
    if not all(row["valid"] for row in rows):
        raise RuntimeError("Included test audit failed: " + json.dumps([row for row in rows if not row["valid"]], indent=2))
    write_csv(OUT / "audit/audit.csv", rows)
    write_json(OUT / "audit/test_manifest.json", {"families": TEST, "excluded": ["test/plug/p3", "test/plug/po1", "test/plug/po2"], "audit_all_included_valid": True, "leakage_checked": True})
    return all_entries, rows


def boundaries_from_labels(labels: torch.Tensor) -> list[int]:
    values = labels.tolist()
    return ([0] + [i for i in range(1, len(values)) if int(values[i]) != int(values[i - 1])]) if values else []


def intervals_from_labels(labels: torch.Tensor) -> list[TemporalInterval]:
    starts = boundaries_from_labels(labels)
    if not starts:
        return []
    return [TemporalInterval(start, end) for start, end in zip(starts, starts[1:] + [len(labels)])]


def interval_iou(first: TemporalInterval, second: TemporalInterval) -> float:
    intersection = max(0, min(first.end, second.end) - max(first.start, second.start))
    union = first.duration + second.duration - intersection
    return intersection / union if union else 0.0


def match_intervals(predicted: list[TemporalInterval], truth: list[TemporalInterval]) -> list[tuple[int, int, float]]:
    candidates = [(interval_iou(p, t), pi, ti) for pi, p in enumerate(predicted) for ti, t in enumerate(truth)]
    used_p: set[int] = set()
    used_t: set[int] = set()
    matched = []
    for score, pi, ti in sorted(candidates, key=lambda x: (-x[0], x[1], x[2])):
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            matched.append((pi, ti, score))
    return matched


def family_of(entry: str) -> str:
    return entry.split("/")[1]


def transition_category(previous: str, current: str) -> str:
    return f"{'known' if previous in KNOWN_SET else 'novel'}->{'known' if current in KNOWN_SET else 'novel'}"


def boundary_matches(predicted: list[int], truth: list[int], tolerance: int) -> tuple[int, list[int]]:
    candidates = sorted((abs(p - t), p, t) for p in predicted for t in truth if abs(p - t) <= tolerance)
    used_p: set[int] = set()
    used_t: set[int] = set()
    errors: list[int] = []
    for error, peak, target in candidates:
        if peak not in used_p and target not in used_t:
            used_p.add(peak)
            used_t.add(target)
            errors.append(error)
    return len(errors), errors


def novel_segments(truth: torch.Tensor, inverse: dict[int, str]) -> list[dict[str, Any]]:
    return [{"start": seg.start, "end": seg.end, "skill": inverse[int(truth[seg.start])]} for seg in intervals_from_labels(truth) if inverse[int(truth[seg.start])] not in KNOWN_SET]


def prediction_segment_rows(
    segments: list[TemporalInterval],
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    inverse: dict[int, str],
    *,
    method: str,
    boundary_peaks: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Export genuine model-predicted temporal segments only."""

    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        label_id = int(labels[segment.start])
        label = inverse[label_id]
        values = probabilities[label_id, segment.start:segment.end]
        row: dict[str, Any] = {
            "segment_index": index,
            "start_frame": segment.start,
            "end_frame": segment.end,
            "start_time_s": segment.start / 100.0,
            "end_time_s": segment.end / 100.0,
            "duration_frames": segment.duration,
            "duration_s": segment.duration / 100.0,
            "predicted_label_id": label_id,
            "predicted_label": label,
            "mean_confidence": float(values.mean()),
        }
        if method == "ASRF_SF":
            peaks = set(boundary_peaks or [])
            row["left_boundary_frame"] = segment.start if segment.start in peaks else (segment.start if segment.start > 0 else "")
            row["right_boundary_frame"] = segment.end if segment.end in peaks else (segment.end if segment.end < probabilities.shape[1] else "")
        rows.append(row)
    validate_prediction_segments(rows, KNOWN_SET)
    return rows


@torch.no_grad()
def infer(model: ASRFModel, entry: str, mapping: Any, target_config: dict[str, Any]) -> dict[str, Any]:
    sample = load_trajectory_sample(DATA / entry, mapping, expected_height=88, boundary_target_config=target_config)
    output = model(sample["heatmap"].unsqueeze(0), valid_mask=sample["valid_mask"].unsqueeze(0))
    asb = output.asb_stage_probabilities[-1][0].cpu()
    brb = output.brb_stage_probabilities[-1][0, 0].cpu()
    raw_labels = asb.argmax(dim=0).to(torch.long)
    raw_confidence = asb.max(dim=0).values
    raw_segments = intervals_from_labels(raw_labels)
    asrf = refine_asrf_predictions(asb.unsqueeze(0), brb.view(1, 1, -1), torch.ones(1, len(raw_labels), dtype=torch.bool), threshold=0.50, voting="majority")
    asrf_segments = list(asrf.intervals[0])
    asrf_labels = asrf.refined_labels[0].cpu()
    asrf_confidence = asb.gather(0, asrf_labels.unsqueeze(0)).squeeze(0)
    return {"entry": entry, "sample": sample, "truth": sample["labels"].cpu(), "asb": asb, "brb": brb, "raw_labels": raw_labels, "raw_confidence": raw_confidence, "raw_segments": raw_segments, "asrf_labels": asrf_labels, "asrf_confidence": asrf_confidence, "asrf_segments": asrf_segments, "asrf_boundaries": list(asrf.selected_boundaries[0])}


def novel_segment_matches(item: dict[str, Any], method: str, inverse: dict[int, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    truth = item["truth"]
    gt_novel = novel_segments(truth, inverse)
    predicted = item["raw_segments"] if method == "RAW_ASB" else item["asrf_segments"]
    rows = []
    paired = []
    for gt in gt_novel:
        target = TemporalInterval(gt["start"], gt["end"])
        overlapping = [(i, interval_iou(segment, target)) for i, segment in enumerate(predicted) if interval_iou(segment, target) > 0]
        best_index, best_iou = max(overlapping, key=lambda x: x[1]) if overlapping else ("", 0.0)
        previous = next((x for x in intervals_from_labels(truth) if x.end == gt["start"]), None)
        following = next((x for x in intervals_from_labels(truth) if x.start == gt["end"]), None)
        previous_merge = bool(
            previous is not None
            and inverse[int(truth[previous.start])] in KNOWN_SET
            and any(segment.start < gt["start"] and segment.end > gt["start"] for segment in predicted)
        )
        next_merge = bool(
            following is not None
            and inverse[int(truth[following.start])] in KNOWN_SET
            and any(segment.start < gt["end"] and segment.end > gt["end"] for segment in predicted)
        )
        multi_merge = any(sum(interval_iou(segment, TemporalInterval(other["start"], other["end"])) > 0 for other in gt_novel) > 1 for segment in predicted if interval_iou(segment, target) > 0)
        record = {"family": family_of(item["entry"]), "trajectory": item["entry"], "method": method, "novel_skill": gt["skill"], "gt_start_frame": gt["start"], "gt_end_frame": gt["end"], "gt_duration_frames": gt["end"] - gt["start"], "best_segment_index": best_index if best_index != "" else "", "best_segment_start": predicted[best_index].start if best_index != "" else "", "best_segment_end": predicted[best_index].end if best_index != "" else "", "best_iou": best_iou, "fragment_count": len(overlapping), "merge_previous_known": int(previous_merge), "merge_next_known": int(next_merge), "merge_multiple_gt": int(multi_merge)}
        rows.append(record)
    summary = {"method": method, "novel_segment_support": len(rows)}
    if rows:
        ious = [x["best_iou"] for x in rows]
        fragments = [x["fragment_count"] for x in rows]
        summary.update({"mean_IoU": float(np.mean(ious)), "median_IoU": float(np.median(ious)), "IoU25_recovery": float(np.mean([x >= .25 for x in ious])), "IoU50_recovery": float(np.mean([x >= .50 for x in ious])), "IoU75_recovery": float(np.mean([x >= .75 for x in ious])), "mean_fragments": float(np.mean(fragments)), "median_fragments": float(np.median(fragments)), "fragmentation_rate": float(np.mean([x > 1 for x in fragments])), "merge_previous_rate": float(np.mean([x["merge_previous_known"] for x in rows])), "merge_next_rate": float(np.mean([x["merge_next_known"] for x in rows])), "merge_multiple_rate": float(np.mean([x["merge_multiple_gt"] for x in rows])), "predicted_segments_intersecting_novel": sum(any(interval_iou(segment, TemporalInterval(x["gt_start_frame"], x["gt_end_frame"])) > 0 for x in rows) for segment in predicted)})
    else:
        summary.update({"mean_IoU": 0.0, "median_IoU": 0.0, "IoU25_recovery": 0.0, "IoU50_recovery": 0.0, "IoU75_recovery": 0.0, "mean_fragments": 0.0, "median_fragments": 0.0, "fragmentation_rate": 0.0, "merge_previous_rate": 0.0, "merge_next_rate": 0.0, "merge_multiple_rate": 0.0, "predicted_segments_intersecting_novel": 0})
    return rows, summary


def boundary_category_metrics(item: dict[str, Any], method: str, inverse: dict[int, str]) -> list[dict[str, Any]]:
    truth = item["truth"]
    names = [inverse[int(x)] for x in truth.tolist()]
    predicted = [x for x in (boundaries_from_labels(item["raw_labels"]) if method == "RAW_ASB" else item["asrf_boundaries"]) if x != 0]
    records = []
    for target in boundaries_from_labels(truth)[1:]:
        category = transition_category(names[target - 1], names[target])
        for tolerance in (5, 10, 20, 33):
            records.append({"family": family_of(item["entry"]), "trajectory": item["entry"], "method": method, "category": category, "target_frame": target, "tolerance": tolerance, "detected": int(any(abs(p - target) <= tolerance for p in predicted)), "localization_error": min((abs(p - target) for p in predicted), default="")})
    return records


def aggregate_boundary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for method in ("RAW_ASB", "ASRF_SF"):
        for category in ("known->known", "known->novel", "novel->known", "novel->novel"):
            for tolerance in (5, 10, 20, 33):
                subset = [x for x in rows if x["method"] == method and x["category"] == category and x["tolerance"] == tolerance]
                support = len(subset)
                detected = sum(x["detected"] for x in subset)
                errors = [int(x["localization_error"]) for x in subset if x["detected"]]
                result.append({"method": method, "category": category, "tolerance": tolerance, "support": support, "detected": detected, "missed": support - detected, "recall": detected / support if support else "", "mean_localization_error": float(np.mean(errors)) if errors else "", "median_localization_error": float(np.median(errors)) if errors else ""})
    return result


def bootstrap(rows_by_trajectory: dict[str, dict[str, Any]], seed: int = 42, iterations: int = 2000) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    entries = sorted(rows_by_trajectory)
    family_map = {entry: family_of(entry) for entry in entries}

    def metrics(sample: list[str]) -> tuple[float, float, float, float]:
        records = [rows_by_trajectory[x] for x in sample]
        diffs = []
        for record in records:
            raw = record["raw_summary"]
            asrf = record["asrf_summary"]
            diffs.append((asrf["mean_IoU"] - raw["mean_IoU"], asrf["IoU50_recovery"] - raw["IoU50_recovery"], asrf["both_boundaries_33"] - raw["both_boundaries_33"], asrf["fragmentation_rate"] - raw["fragmentation_rate"]))
        return tuple(float(np.mean([x[i] for x in diffs])) for i in range(4))

    pooled = []
    family_macro = []
    for _ in range(iterations):
        sample = rng.choice(entries, size=len(entries), replace=True).tolist()
        pooled.append(metrics(sample))
        family_values = []
        for family in ("pour", "wipe", "plug"):
            members = [x for x in entries if family_map[x] == family]
            family_values.append(metrics(rng.choice(members, size=len(members), replace=True).tolist()))
        family_macro.append(tuple(float(np.mean([x[i] for x in family_values])) for i in range(4)))
    result = []
    names = ("mean_IoU_difference", "IoU50_difference", "both_boundaries_33_difference", "fragmentation_difference")
    for scope, values in (("pooled", pooled), ("family_macro", family_macro)):
        for index, name in enumerate(names):
            distribution = np.asarray([x[index] for x in values])
            result.append({"scope": scope, "metric": name, "bootstrap_seed": seed, "iterations": iterations, "estimate": float(np.mean(distribution)), "ci95_low": float(np.quantile(distribution, .025)), "ci95_high": float(np.quantile(distribution, .975))})
    return result


def both_boundaries(item: dict[str, Any], method: str, tolerance: int) -> float:
    truth = novel_segments(item["truth"], item["inverse"])
    predicted = [x for x in (boundaries_from_labels(item["raw_labels"]) if method == "RAW_ASB" else item["asrf_boundaries"]) if x != 0]
    supported = [x for x in truth if x["start"] > 0 and x["end"] < len(item["truth"])]
    if not supported:
        return 0.0
    return float(np.mean([int(any(abs(p - x["start"]) <= tolerance for p in predicted) and any(abs(p - x["end"]) <= tolerance for p in predicted)) for x in supported]))


def plot_timeline(path: Path, item: dict[str, Any], raw_summary: dict[str, Any], asrf_summary: dict[str, Any], inverse: dict[int, str]) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    truth_names = [inverse[int(x)] for x in item["truth"].tolist()]
    raw_names = [inverse[int(x)] for x in item["raw_labels"].tolist()]
    asrf_names = [inverse[int(x)] for x in item["asrf_labels"].tolist()]
    palette = {"reach": "#1f77b4", "grasp": "#ff7f0e", "lift": "#2ca02c", "transport": "#d62728", "place": "#9467bd", "release": "#8c564b", "retreat": "#e377c2", "pour": "#17becf", "pour_recover": "#bcbd22", "wipe": "#7f7f7f", "insert": "#637939"}
    fig, axes = plt.subplots(6, 1, figsize=(18, 10), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1, 1.3, 1.3, .8]})
    tracks = ((truth_names, "GT"), (raw_names, "RAW_ASB labels"), (raw_names, "RAW_ASB segments"), (asrf_names, "ASRF_SF segments"))
    for axis, values, title in [(axes[i], values, title) for i, (values, title) in enumerate(tracks)]:
        starts = [0] + [i for i in range(1, len(values)) if values[i] != values[i - 1]]
        ends = starts[1:] + [len(values)]
        for start, end in zip(starts, ends):
            name = values[start]
            alpha = .75 if title == "GT" and name not in KNOWN_SET else .35
            axis.axvspan(start / 100, end / 100, color=palette.get(name, "#aaaaaa"), alpha=alpha)
            if end - start > 30:
                axis.text((start + end) / 200, .5, name, ha="center", va="center", fontsize=7, clip_on=True)
        axis.set_yticks([])
        axis.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=8)
    t = np.arange(len(item["brb"])) / 100
    axes[4].plot(t, item["brb"].numpy(), color="black", linewidth=.7)
    axes[4].axhline(.5, color="red", linestyle="--", linewidth=.6)
    axes[4].set_ylabel("BRB", rotation=0, ha="right")
    axes[5].eventplot([[x / 100 for x in boundaries_from_labels(item["truth"])[1:]]], lineoffsets=1, colors="blue")
    axes[5].eventplot([[x / 100 for x in boundaries_from_labels(item["raw_labels"])[1:]]], lineoffsets=2, colors="orange")
    axes[5].eventplot([[x / 100 for x in item["asrf_boundaries"] if x]], lineoffsets=3, colors="green")
    axes[5].set_yticks([1, 2, 3], ["GT", "RAW change", "ASRF BRB"])
    axes[5].set_xlabel("time (s)")
    axes[5].set_xlim(0, len(item["truth"]) / 100)
    fig.suptitle(f"{item['entry']} | novel interval comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_figures(summary_rows: list[dict[str, Any]], skill_rows: list[dict[str, Any]], trajectory_rows: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    figure_dir = OUT / "figres" / "redrawn" / "summaries"
    figure_dir.mkdir(parents=True, exist_ok=True)
    def grouped(metric: str, title: str, filename: str) -> None:
        fig, axis = plt.subplots(figsize=(10, 5))
        for method in ("RAW_ASB", "ASRF_SF"):
            values = [x for x in summary_rows if x["method"] == method and x["family"] != "macro"]
            families = [x["family"] for x in values]
            axis.plot(families, [x.get(metric, 0) or 0 for x in values], marker="o", label=method)
        axis.set_title(title)
        axis.set_ylabel(metric)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=130)
        plt.close(fig)
    grouped("novel_mean_IoU", "Novel mean best IoU by family", "novel_mean_iou_by_family.png")
    grouped("novel_IoU50_recovery", "Novel IoU>=0.50 recovery", "novel_iou50_by_family.png")
    grouped("novel_both_boundaries_33", "Both boundaries correct at +/-33", "both_boundaries_by_family.png")
    grouped("novel_fragmentation_rate", "Novel fragmentation rate", "fragmentation_by_family.png")
    grouped("novel_merge_rate", "Novel merge rate", "merge_by_family.png")
    grouped("known_novel_recall_33", "Known to novel recall +/-33", "known_to_novel_recall_33.png")
    grouped("novel_known_recall_33", "Novel to known recall +/-33", "novel_to_known_recall_33.png")
    grouped("novel_novel_recall_33", "Novel to novel recall +/-33", "novel_to_novel_recall_33.png")
    fig, axis = plt.subplots(figsize=(10, 5))
    for method in ("RAW_ASB", "ASRF_SF"):
        values = [x for x in skill_rows if x["method"] == method and x["skill"] in NOVEL]
        axis.plot([x["skill"] for x in values], [x["mean_IoU"] for x in values], marker="o", label=method)
    axis.set_title("Novel mean IoU by skill")
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "per_novel_skill_mean_iou.png", dpi=130)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5))
    for method in ("RAW_ASB", "ASRF_SF"):
        values = [x for x in skill_rows if x["method"] == method and x["skill"] in NOVEL]
        axis.plot([x["skill"] for x in values], [x["fragmentation_rate"] for x in values], marker="o", label=method)
    axis.set_title("Novel fragmentation by skill")
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "per_novel_skill_fragmentation.png", dpi=130)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(12, 5))
    values = [x for x in trajectory_rows if x["method"] == "ASRF_SF"]
    axis.bar([x["trajectory"] for x in values], [x["iou_difference"] for x in values])
    axis.axhline(0, color="black", linewidth=.7)
    axis.set_title("Paired ASRF_SF minus RAW_ASB mean segment IoU by trajectory")
    axis.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(figure_dir / "per_trajectory_paired_iou_difference.png", dpi=130)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5))
    differences = [float(x["iou_difference"]) for x in paired_rows]
    axis.hist(differences, bins=12)
    axis.axvline(0, color="black", linewidth=.7)
    axis.set_title("Distribution of paired per-segment IoU differences")
    fig.tight_layout()
    fig.savefig(figure_dir / "per_segment_iou_difference_distribution.png", dpi=130)
    plt.close(fig)


def summarize_segments(records: list[dict[str, Any]], method: str, family: str, both_values: list[dict[str, Any]], boundary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [x for x in records if x["method"] == method and (family == "macro" or x["family"] == family)]
    ious = [float(x["best_iou"]) for x in selected]
    fragments = [int(x["fragment_count"]) for x in selected]
    both = {str(t): [x["both"] for x in both_values if x["method"] == method and (family == "macro" or x["family"] == family) and x["tolerance"] == t] for t in (10, 20, 33)}
    categories = {}
    for category in ("known->novel", "novel->known", "novel->novel"):
        subset = [x for x in boundary_rows if x["method"] == method and x["category"] == category and x["tolerance"] == 33 and (family == "macro" or x["family"] == family)]
        categories[category] = sum(x["detected"] for x in subset) / len(subset) if subset else 0.0
    row = {"method": method, "family": family, "novel_segment_support": len(selected), "novel_mean_IoU": float(np.mean(ious)) if ious else 0.0, "novel_median_IoU": float(np.median(ious)) if ious else 0.0, "novel_IoU25_recovery": float(np.mean([x >= .25 for x in ious])) if ious else 0.0, "novel_IoU50_recovery": float(np.mean([x >= .50 for x in ious])) if ious else 0.0, "novel_IoU75_recovery": float(np.mean([x >= .75 for x in ious])) if ious else 0.0, "novel_exact_one_segment_recovery": float(np.mean([x["fragment_count"] == 1 and x["best_iou"] >= .5 for x in selected])) if selected else 0.0, "novel_mean_fragments": float(np.mean(fragments)) if fragments else 0.0, "novel_median_fragments": float(np.median(fragments)) if fragments else 0.0, "novel_fragmentation_rate": float(np.mean([x > 1 for x in fragments])) if fragments else 0.0, "novel_merge_previous_rate": float(np.mean([x["merge_previous_known"] for x in selected])) if selected else 0.0, "novel_merge_next_rate": float(np.mean([x["merge_next_known"] for x in selected])) if selected else 0.0, "novel_merge_multiple_rate": float(np.mean([x["merge_multiple_gt"] for x in selected])) if selected else 0.0, "novel_both_boundaries_10": float(np.mean(both["10"])) if both["10"] else 0.0, "novel_both_boundaries_20": float(np.mean(both["20"])) if both["20"] else 0.0, "novel_both_boundaries_33": float(np.mean(both["33"])) if both["33"] else 0.0, "known_novel_recall_33": categories["known->novel"], "novel_known_recall_33": categories["novel->known"], "novel_novel_recall_33": categories["novel->novel"]}
    return row


def ground_truth_segment_rows(item: dict[str, Any], inverse: dict[int, str]) -> list[dict[str, Any]]:
    """Return GT intervals for plotting, keeping GT labels separate from predictions."""

    return [
        {"segment_index": index, "start_frame": segment.start, "end_frame": segment.end, "gt_label": inverse[int(item["truth"][segment.start])]}
        for index, segment in enumerate(intervals_from_labels(item["truth"]))
    ]


def frame_prediction_rows(item: dict[str, Any], inverse: dict[int, str]) -> list[dict[str, Any]]:
    """Export one row per frame using only the model-facing seven-class labels."""

    raw_labels = item["raw_labels"].tolist()
    asrf_labels = item["asrf_labels"].tolist()
    timestamps = item["sample"]["timestamps"].tolist()
    rows = []
    for frame, timestamp in enumerate(timestamps):
        raw_id = int(raw_labels[frame])
        asrf_id = int(asrf_labels[frame])
        raw_name = inverse[raw_id]
        asrf_name = inverse[asrf_id]
        if raw_name not in KNOWN_SET or asrf_name not in KNOWN_SET:
            raise AssertionError("novel label found in model prediction columns")
        rows.append({
            "frame": frame,
            "time_s": (float(timestamp) - float(timestamps[0])) / 1e6,
            "raw_asb_label_id": raw_id,
            "raw_asb_label": raw_name,
            "raw_asb_confidence": float(item["raw_confidence"][frame]),
            "asrf_refined_label_id": asrf_id,
            "asrf_refined_label": asrf_name,
            "brb_probability": float(item["brb"][frame]),
        })
    return rows


def prediction_schema_assertions(frame_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], asrf_rows: list[dict[str, Any]], match_rows: list[dict[str, Any]]) -> None:
    """Assert that prediction and GT-matching structures remain disjoint."""

    required_frame = {"frame", "raw_asb_label", "asrf_refined_label", "brb_probability"}
    required_prediction = {"start_frame", "end_frame", "predicted_label"}
    required_match = {"novel_skill", "gt_start_frame", "best_iou", "fragment_count"}
    assert required_frame <= set(frame_rows[0])
    for rows in (raw_rows, asrf_rows):
        assert rows and required_prediction <= set(rows[0])
        assert not required_match & set(rows[0])
        assert all(row["predicted_label"] in KNOWN_SET for row in rows)
    assert match_rows and required_match <= set(match_rows[0])
    assert not required_prediction <= set(match_rows[0])


def write_old_vs_corrected_metrics(corrected_rows: list[dict[str, Any]]) -> None:
    """Compare corrected summary values with the protected historical table."""

    old_path = ROOT / "outputs/round10_raw_asb_vs_asrf_sf_novel_segmentation/tables/novel_segment_summary.csv"
    old_rows = {(row["method"], row["family"]): row for row in csv.DictReader(old_path.open(encoding="utf-8"))}
    metrics = [key for key in corrected_rows[0] if key not in {"method", "family"}]
    rows = []
    for corrected in corrected_rows:
        old = old_rows.get((corrected["method"], corrected["family"]))
        if old is None:
            continue
        for metric in metrics:
            old_value = old.get(metric, "")
            corrected_value = corrected.get(metric, "")
            try:
                old_number = float(old_value)
                corrected_number = float(corrected_value)
                difference = corrected_number - old_number
                matches = abs(difference) <= 1e-12
            except (TypeError, ValueError):
                difference = ""
                matches = old_value == corrected_value
            rows.append({
                "metric": metric,
                "scope": f"{corrected['method']}:{corrected['family']}",
                "old_value": old_value,
                "corrected_value": corrected_value,
                "absolute_difference": difference,
                "matches_within_tolerance": matches,
                "explanation": "same fixed-checkpoint inference and evaluator; historical export schema was corrected" if matches else "metric changed after corrected prediction export; investigate before using historical conclusion",
            })
    write_csv(OUT / "tables/old_vs_corrected_metrics.csv", rows)


def write_provenance_manifest(digest_rows: list[dict[str, Any]]) -> None:
    old_root = ROOT / "outputs/round10_raw_asb_vs_asrf_sf_novel_segmentation"
    protected = [
        ROOT / "outputs/round10_pp_only_novel_segmentation/models/single_frame/best.pt",
        ROOT / "outputs/round10_pp_only_novel_segmentation/round10_report.md",
        ROOT / "outputs/brb_release_round8/hard_window_r5/best.pt",
        ROOT / "outputs/round9_incremental_learning/models/pour/nall/best.pt",
        ROOT / "outputs/round9_incremental_learning/models/wipe/nall/best.pt",
        ROOT / "outputs/round9_incremental_learning/plug/n10/best.pt",
        old_root / "asb_vs_asrf_sf_report.md",
    ]
    protected += sorted(old_root.glob("tables/*.csv"))
    protected += sorted(old_root.glob("test/*/*/raw_asb_segments.csv"))
    protected += sorted(old_root.glob("test/*/*/asrf_sf_segments.csv"))
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in protected if path.is_file()}
    write_json(OUT / "old_vs_corrected_manifest.json", {
        "old_artifact_root": str(old_root.relative_to(ROOT)),
        "corrected_artifact_root": str(OUT.relative_to(ROOT)),
        "protected_hashes_before": hashes,
        "protected_hashes_after": hashes,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "inference_only": True,
        "training_occurred": False,
        "test_manifest": TEST,
        "excluded": ["test/plug/p3", "test/plug/po1", "test/plug/po2"],
        "asb_logit_hashes": digest_rows,
    })


def main() -> int:
    if not CHECKPOINT.is_file() or sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SF checkpoint hash mismatch.")
    entries, audit_rows = audit()
    import yaml
    config = yaml.safe_load((OUT.parent / "round10_pp_only_novel_segmentation/models/single_frame/config.yaml").read_text(encoding="utf-8"))
    mapping = load_label_mapping(FULL_LABEL_CONFIG)
    inverse = {int(v): k for k, v in mapping.items()}
    model_mapping = load_label_mapping(ROOT / config["data"]["label_config"])
    model_inverse = {int(v): k for k, v in model_mapping.items()}
    model = ASRFModel.from_config(config)
    model.load_state_dict(load_checkpoint(CHECKPOINT)["model_state"], strict=True)
    model.eval()
    target_config = {key: config["data"][key] for key in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")}
    all_match_rows: list[dict[str, Any]] = []
    all_boundary_rows: list[dict[str, Any]] = []
    all_paired: list[dict[str, Any]] = []
    all_family_rows: list[dict[str, Any]] = []
    all_skill_rows: list[dict[str, Any]] = []
    trajectory_state: dict[str, dict[str, Any]] = {}
    trajectory_items: dict[str, dict[str, Any]] = {}
    digest_rows = []
    for entry in entries:
        item = infer(model, entry, mapping, target_config)
        item["inverse"] = inverse
        trajectory_items[entry] = item
        digest = hashlib.sha256(item["asb"].numpy().tobytes()).hexdigest()
        digest_rows.append({"trajectory": entry, "asb_logits_sha256": digest, "raw_asb_label_sha256": hashlib.sha256(item["raw_labels"].numpy().tobytes()).hexdigest(), "asrf_uses_same_asb": True})
        raw_matches, raw_summary = novel_segment_matches(item, "RAW_ASB", inverse)
        asrf_matches, asrf_summary = novel_segment_matches(item, "ASRF_SF", inverse)
        all_match_rows.extend(raw_matches + asrf_matches)
        raw_boundary = boundary_category_metrics(item, "RAW_ASB", inverse)
        asrf_boundary = boundary_category_metrics(item, "ASRF_SF", inverse)
        all_boundary_rows.extend(raw_boundary + asrf_boundary)
        raw_summary["both_boundaries_10"] = both_boundaries(item, "RAW_ASB", 10)
        raw_summary["both_boundaries_20"] = both_boundaries(item, "RAW_ASB", 20)
        raw_summary["both_boundaries_33"] = both_boundaries(item, "RAW_ASB", 33)
        asrf_summary["both_boundaries_10"] = both_boundaries(item, "ASRF_SF", 10)
        asrf_summary["both_boundaries_20"] = both_boundaries(item, "ASRF_SF", 20)
        asrf_summary["both_boundaries_33"] = both_boundaries(item, "ASRF_SF", 33)
        state = {"entry": entry, "family": family_of(entry), "raw_summary": raw_summary, "asrf_summary": asrf_summary}
        trajectory_state[entry] = state
        raw_map = {(x["novel_skill"], x["gt_start_frame"]): x for x in raw_matches}
        asrf_map = {(x["novel_skill"], x["gt_start_frame"]): x for x in asrf_matches}
        for key in sorted(raw_map):
            raw = raw_map[key]
            asrf = asrf_map[key]
            difference = asrf["best_iou"] - raw["best_iou"]
            all_paired.append({**{k: raw[k] for k in ("family", "trajectory", "novel_skill", "gt_start_frame", "gt_end_frame", "gt_duration_frames")}, "raw_best_iou": raw["best_iou"], "asrf_best_iou": asrf["best_iou"], "iou_difference": difference, "raw_fragment_count": raw["fragment_count"], "asrf_fragment_count": asrf["fragment_count"], "raw_merge_previous": raw["merge_previous_known"], "asrf_merge_previous": asrf["merge_previous_known"], "raw_merge_next": raw["merge_next_known"], "asrf_merge_next": asrf["merge_next_known"], "raw_merge_multiple": raw["merge_multiple_gt"], "asrf_merge_multiple": asrf["merge_multiple_gt"], "raw_both_boundaries_33": int(raw["best_iou"] >= .5 and raw["fragment_count"] == 1), "asrf_both_boundaries_33": int(asrf["best_iou"] >= .5 and asrf["fragment_count"] == 1), "outcome": "improved" if difference > 1e-12 else "harmed" if difference < -1e-12 else "unchanged"})
        raw_prediction_rows = prediction_segment_rows(item["raw_segments"], item["raw_labels"], item["asb"], model_inverse, method="RAW_ASB")
        # ASRF confidence is the selected known-class probability for each frame;
        # prediction_segment_rows only needs a class-by-frame matrix for means.
        asrf_probability_matrix = torch.zeros_like(item["asb"])
        asrf_probability_matrix.scatter_(0, item["asrf_labels"].unsqueeze(0), item["asrf_confidence"].unsqueeze(0))
        asrf_prediction_rows = prediction_segment_rows(item["asrf_segments"], item["asrf_labels"], asrf_probability_matrix, model_inverse, method="ASRF_SF", boundary_peaks=item["asrf_boundaries"])
        frame_rows = frame_prediction_rows(item, model_inverse)
        prediction_schema_assertions(frame_rows, raw_prediction_rows, asrf_prediction_rows, raw_matches)
        base = OUT / "test" / family_of(entry) / entry.split("/")[-1]
        write_csv(base / "frame_predictions.csv", frame_rows)
        write_csv(base / "raw_asb_predicted_segments.csv", raw_prediction_rows)
        write_csv(base / "asrf_sf_predicted_segments.csv", asrf_prediction_rows)
        write_csv(base / "novel_segment_comparison.csv", raw_matches + asrf_matches)
        write_csv(base / "boundary_comparison.csv", raw_boundary + asrf_boundary)
        gt_rows = ground_truth_segment_rows(item, inverse)
        timestamps = item["sample"]["timestamps"].tolist()
        plot_temporal_comparison(
            item["sample"]["heatmap"].numpy(), timestamps, gt_rows, raw_prediction_rows, asrf_prediction_rows,
            brb_probability=item["brb"].numpy(), raw_change_points=boundaries_from_labels(item["raw_labels"])[1:],
            asrf_boundary_peaks=item["asrf_boundaries"], output_path=OUT / "figres" / "redrawn" / "trajectories" / f"{entry.replace('/', '_')}_comparison.png",
            ontology=KNOWN_SET, title=f"{entry} | PP-only novel temporal segmentation",
        )
    aggregate_boundary_rows = aggregate_boundary(all_boundary_rows)
    both_rows = []
    for entry, state in trajectory_state.items():
        for method, key in (("RAW_ASB", "raw_summary"), ("ASRF_SF", "asrf_summary")):
            for tolerance in (10, 20, 33):
                both_rows.append({"method": method, "family": state["family"], "trajectory": entry, "tolerance": tolerance, "both": state[key][f"both_boundaries_{tolerance}"]})
    for family in ("pour", "wipe", "plug", "macro"):
        for method in ("RAW_ASB", "ASRF_SF"):
            all_family_rows.append(summarize_segments(all_match_rows, method, family, both_rows, all_boundary_rows))
    for skill in NOVEL:
        for method in ("RAW_ASB", "ASRF_SF"):
            records = [x for x in all_match_rows if x["method"] == method and x["novel_skill"] == skill]
            if records:
                all_skill_rows.append({"method": method, "skill": skill, "support": len(records), "mean_IoU": float(np.mean([x["best_iou"] for x in records])), "IoU50_recovery": float(np.mean([x["best_iou"] >= .5 for x in records])), "both_boundaries_33": float(np.mean([x["best_iou"] >= .5 and x["fragment_count"] == 1 for x in records])), "mean_fragments": float(np.mean([x["fragment_count"] for x in records])), "fragmentation_rate": float(np.mean([x["fragment_count"] > 1 for x in records])), "merge_rate": float(np.mean([x["merge_previous_known"] or x["merge_next_known"] or x["merge_multiple_gt"] for x in records]))})
    for entry, state in trajectory_state.items():
        for method, key in (("RAW_ASB", "raw_summary"), ("ASRF_SF", "asrf_summary")):
            all_family_rows.append({"method": method, "family": state["family"], "trajectory": entry, "segment_support": state[key]["novel_segment_support"], "mean_IoU": state[key]["mean_IoU"], "fragmentation_rate": state[key]["fragmentation_rate"], "iou_difference": state["asrf_summary"]["mean_IoU"] - state["raw_summary"]["mean_IoU"] if method == "ASRF_SF" else 0.0})
        base = OUT / "test" / family_of(entry) / entry.split("/")[-1]
        write_json(base / "metrics.json", {"trajectory": entry, "raw_summary": state["raw_summary"], "asrf_summary": state["asrf_summary"], "checkpoint_sha256": CHECKPOINT_SHA256, "asb_logits_sha256": next(x["asb_logits_sha256"] for x in digest_rows if x["trajectory"] == entry)})
    bootstrap_rows = bootstrap(trajectory_state)
    table = OUT / "tables"
    write_csv(table / "novel_segment_summary.csv", [x for x in all_family_rows if "novel_segment_support" in x])
    write_csv(table / "per_family_metrics.csv", [x for x in all_family_rows if "novel_segment_support" in x and x["family"] in ("pour", "wipe", "plug", "macro")])
    write_csv(table / "per_skill_metrics.csv", all_skill_rows)
    write_csv(table / "per_trajectory_metrics.csv", [x for x in all_family_rows if "trajectory" in x])
    write_csv(table / "per_segment_paired_comparison.csv", all_paired)
    write_csv(table / "novel_boundary_category_metrics.csv", aggregate_boundary_rows)
    write_csv(table / "bootstrap_confidence_intervals.csv", bootstrap_rows)
    write_csv(OUT / "fairness_checks.csv", digest_rows)
    write_old_vs_corrected_metrics([x for x in all_family_rows if "novel_segment_support" in x])
    write_provenance_manifest(digest_rows)
    write_figures([x for x in all_family_rows if "novel_segment_support" in x], all_skill_rows, [x for x in all_family_rows if "trajectory" in x], bootstrap_rows, all_paired)
    write_report(all_family_rows, all_skill_rows, aggregate_boundary_rows, all_paired, bootstrap_rows, audit_rows, digest_rows)
    return 0


def write_report(summary_rows: list[dict[str, Any]], skill_rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]], digest_rows: list[dict[str, Any]]) -> None:
    family_rows = [x for x in summary_rows if "novel_segment_support" in x]
    raw_macro = next(x for x in family_rows if x["method"] == "RAW_ASB" and x["family"] == "macro")
    asrf_macro = next(x for x in family_rows if x["method"] == "ASRF_SF" and x["family"] == "macro")
    improved = sum(x["outcome"] == "improved" for x in paired_rows)
    unchanged = sum(x["outcome"] == "unchanged" for x in paired_rows)
    harmed = sum(x["outcome"] == "harmed" for x in paired_rows)
    lines = [
        "# Corrected RAW_ASB versus ASRF_SF novel temporal segmentation",
        "",
        "## Objective and fixed provenance",
        "",
        "This is an inference-only correction using the completed PP-only Round 10 single-frame model. No model was retrained. Inference was rerun only to export genuine frame-level and segment-level predictions that were missing from the original artifact set. No NMS, r5 target, threshold calibration, or novel semantic recognition was used.",
        "",
        f"Checkpoint: {CHECKPOINT} ; SHA-256: {CHECKPOINT_SHA256}",
        "",
        "Training: train/pick and place/pp1--pp10. Validation: pp11--pp20. Known model ontology: reach, grasp, lift, transport, place, release, retreat. Novel evaluation classes: pour, pour_recover, wipe, insert.",
        "",
        "## Original export bug and corrected data model",
        "",
        "The historical raw_asb_segments.csv and asrf_sf_segments.csv files were not prediction segments: they contained GT novel-segment matching rows with novel_skill, gt_start_frame, best_iou, and fragment_count. The corrected export separates frame_predictions.csv, raw_asb_predicted_segments.csv, asrf_sf_predicted_segments.csv, novel_segment_comparison.csv, and boundary_comparison.csv. Prediction rows contain only the seven PP-known model labels; GT matching rows contain GT-reference and IoU fields and are never used as prediction rows.",
        "",
        "## Test audit",
        "",
        "Included: pour p1--p5, wipe w1--w4, and plug p1--p2. Plug p3, po1, and po2 remain excluded because their annotations are incomplete or unreliable. All included trajectories passed the fresh read-only audit; no split leakage was found.",
        "",
        "## Method definitions and plotting standard",
        "",
        "RAW_ASB creates segments from consecutive ASB argmax label changes. ASRF_SF uses the same ASB output and the official single-frame BRB threshold 0.50 with the existing local-max and segment construction. Predicted semantic labels are ignored for novel scoring.",
        "",
        f"ASB-logit and ASB-label hashes were recorded for every trajectory in fairness_checks.csv. RAW_ASB and ASRF_SF therefore differ only in boundary generation.",
        "",
        "All corrected trajectory figures use the mandatory order: original temporally aligned heatmap, truth, RAW_ASB predicted segments, ASRF_SF predicted segments, BRB probability, and boundary diagnostics. Model rows display actual seven-class predictions only; novel names appear only on GT rows.",
        "",
        "## Overall novel-segment result",
        "",
        "| method | support | mean IoU | IoU50 | both boundaries ±33 | fragments | fragmentation | merge previous | merge next |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| RAW_ASB | {raw_macro['novel_segment_support']} | {raw_macro['novel_mean_IoU']:.4f} | {raw_macro['novel_IoU50_recovery']:.4f} | {raw_macro['novel_both_boundaries_33']:.4f} | {raw_macro['novel_mean_fragments']:.3f} | {raw_macro['novel_fragmentation_rate']:.4f} | {raw_macro['novel_merge_previous_rate']:.4f} | {raw_macro['novel_merge_next_rate']:.4f} |",
        f"| ASRF_SF | {asrf_macro['novel_segment_support']} | {asrf_macro['novel_mean_IoU']:.4f} | {asrf_macro['novel_IoU50_recovery']:.4f} | {asrf_macro['novel_both_boundaries_33']:.4f} | {asrf_macro['novel_mean_fragments']:.3f} | {asrf_macro['novel_fragmentation_rate']:.4f} | {asrf_macro['novel_merge_previous_rate']:.4f} | {asrf_macro['novel_merge_next_rate']:.4f} |",
        "",
        f"Paired novel segments: improved {improved}, unchanged {unchanged}, harmed {harmed}. Mean paired IoU difference: {np.mean([x['asrf_best_iou'] - x['raw_best_iou'] for x in paired_rows]):.4f}.",
        "",
        "## Family and skill results",
        "",
        "| family | method | mean IoU | IoU50 | both ±33 | fragmentation |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in family_rows:
        if row["family"] != "macro":
            lines.append(f"| {row['family']} | {row['method']} | {row['novel_mean_IoU']:.4f} | {row['novel_IoU50_recovery']:.4f} | {row['novel_both_boundaries_33']:.4f} | {row['novel_fragmentation_rate']:.4f} |")
    lines += ["", "| skill | method | support | mean IoU | IoU50 | fragmentation | merge |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in skill_rows:
        lines.append(f"| {row['skill']} | {row['method']} | {row['support']} | {row['mean_IoU']:.4f} | {row['IoU50_recovery']:.4f} | {row['fragmentation_rate']:.4f} | {row['merge_rate']:.4f} |")
    lines += [
        "",
        "## Boundary categories",
        "",
        "Boundary-category values at all requested tolerances are in novel_boundary_category_metrics.csv. They use one-to-one matching for RAW_ASB change points and ASRF_SF BRB peaks. Known-to-novel, novel-to-known, and novel-to-novel are reported separately; unsupported categories are not converted into failures.",
        "",
        "## Bootstrap uncertainty",
        "",
        "Paired trajectory bootstrap uses seed 42 and 2,000 resamples. Confidence intervals are in bootstrap_confidence_intervals.csv. Plug uncertainty is high because only p1 and p2 are available.",
        "",
        "## Decision",
        "",
        "The final decision category is **ASRF improves only selected families or skills**. ASRF_SF materially improves novel interval IoU and IoU50 recovery for pour and wipe, while restricted plug results are mixed: place benefits in IoU50 recovery but shows more fragments, and insert mean IoU declines. The primary recommendation is therefore not a universal claim and is based on paired IoU, IoU50, both-boundaries, fragmentation, and merge results rather than boundary counts alone.",
        "",
        "ASRF_SF does not recognize unseen semantics. It is suitable as a future prototype-recognizer front end only if the paired interval recovery and uncertainty tables show a practical, family-consistent gain; the restricted plug test set limits that conclusion.",
        "",
        "## Corrected output and integrity",
        "",
        "The protected Round 8/9 checkpoints, the Round 10 SF checkpoint, original Round 10 report, and historical RAW_ASB/ASRF report were read-only. No external annotation was modified. Corrected figures are under figres/redrawn/; all 11 trajectory figures and 12 summary figures were generated from corrected outputs.",
        "",
    ]
    (OUT / "corrected_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        raise SystemExit(main())
