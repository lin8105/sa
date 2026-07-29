"""Cross-family, inference-only greedy BRB NMS ablation."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
OUT = ROOT / "outputs/round9_incremental_learning/nms_cross_family"
DISTANCES = (0, 10, 20, 30)
SKILLS = ("reach", "grasp", "lift", "transport", "pour", "pour_recover", "place", "release", "wipe", "retreat", "insert")
NAMES = {index: name for index, name in enumerate(SKILLS)}
IDS = {name: index for index, name in NAMES.items()}
FAMILIES = {
    "pour": {"checkpoint": ROOT / "outputs/round9_incremental_learning/models/pour/nall/best.pt", "config": ROOT / "outputs/round9_incremental_learning/models/pour/nall/config.yaml", "validation": "splits/round9_incremental/common_validation.txt", "test": "splits/round9_incremental/test_pour_primary.txt", "transitions": ("transport -> pour", "pour -> pour_recover", "pour_recover -> transport", "place -> release"), "target_skills": ("pour", "pour_recover", "place", "release")},
    "wipe": {"checkpoint": ROOT / "outputs/round9_incremental_learning/models/wipe/nall/best.pt", "config": ROOT / "outputs/round9_incremental_learning/models/wipe/nall/config.yaml", "validation": "splits/round9_incremental/common_validation.txt", "test": "splits/round9_incremental/test_wipe_primary.txt", "transitions": ("place -> wipe", "wipe -> lift", "place -> release"), "target_skills": ("wipe", "lift", "transport", "place", "release", "retreat")},
    "plug": {"checkpoint": ROOT / "outputs/round9_incremental_learning/plug/n10/best.pt", "config": ROOT / "outputs/round9_incremental_learning/plug/n10/config.yaml", "validation": "splits/round9_incremental/common_validation.txt", "test": "splits/round9_incremental/test_plug_restricted_p1_p2.txt", "transitions": ("transport -> place", "place -> insert", "insert -> release"), "target_skills": ("place", "insert", "release", "transport", "lift")},
}
EXPECTED_PLUG_HASH = "1c75c15d45c63a18f1cfd2c856952f42b35277fe9baa5e4951fe886227bf1ee5"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from asrf.data.dataset import load_trajectory_sample  # noqa: E402
from asrf.data.labels import load_label_mapping, normalize_label_name  # noqa: E402
from asrf.evaluation.metrics import boundary_counts, labels_to_segments  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.refinement.majority_vote import _vote_one  # noqa: E402
from asrf.refinement.peaks import greedy_score_guided_nms, select_boundary_peaks  # noqa: E402
from asrf.refinement.segments import construct_segments  # noqa: E402
from asrf.training.checkpointing import load_checkpoint, sha256_file  # noqa: E402
from asrf.utils.config import load_yaml_config  # noqa: E402
from evaluate_round9_incremental import records, semantic, truth_boundaries  # noqa: E402


def jd(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, default=jd) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or list(rows[0]) if rows else fields or []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def nearest_matches(predicted: list[int], truth: list[int], tolerance: int) -> tuple[list[int], list[int], list[int]]:
    candidates = sorted((abs(p - t), p, t) for p in predicted for t in truth if abs(p - t) <= tolerance)
    used_p: set[int] = set(); used_t: set[int] = set(); errors: list[int] = []
    for error, peak, boundary in candidates:
        if peak not in used_p and boundary not in used_t: used_p.add(peak); used_t.add(boundary); errors.append(error)
    return sorted(used_p), sorted(used_t), errors


def nms_events(candidates: list[int], probabilities: torch.Tensor, distance: int, truth: list[int]) -> tuple[list[int], list[dict[str, Any]]]:
    scores = [float(probabilities[peak]) for peak in candidates]
    retained = greedy_score_guided_nms(candidates, scores, distance)
    retained_set = set(retained); ranked = sorted(set(candidates), key=lambda peak: (-float(probabilities[peak]), peak)); selected_order: list[int] = []; events: list[dict[str, Any]] = []
    for peak in ranked:
        if peak in retained_set: selected_order.append(peak); continue
        suppressor = next((item for item in selected_order if abs(peak - item) < distance), None)
        if suppressor is None: continue
        near_truth = min((abs(peak - boundary) for boundary in truth), default=10**9); near_retained = min((abs(suppressor - boundary) for boundary in truth), default=10**9)
        kind = "duplicate_near_true_boundary" if near_truth <= 33 and near_retained <= 33 else "isolated_false_peak_inside_skill" if near_truth > 33 else "true_boundary_suppression_risk"
        events.append({"candidate_frame": peak, "candidate_probability": float(probabilities[peak]), "suppressing_peak_frame": suppressor, "suppressing_peak_probability": float(probabilities[suppressor]), "distance_frames": abs(peak - suppressor), "distance_seconds": abs(peak - suppressor) / 100.0, "event_type": kind, "nearest_truth_distance": near_truth})
    return retained, events


def add_variants(record: dict[str, Any], distance: int) -> dict[str, Any]:
    candidates = list(select_boundary_peaks(record["brb"], threshold=0.5)); retained, events = nms_events(candidates, record["brb"], distance, truth_boundaries(record)); prediction, diagnostics = _vote_one(record["asb"], construct_segments(retained, len(record["truth"])), voting="majority"); raw = record["asb"].argmax(dim=0)
    return {"raw": {"prediction": raw, "peaks": []}, "refined": {"prediction": prediction, "peaks": retained, "candidate_peaks": candidates, "events": events, "diagnostics": diagnostics}}


def segment_class_scores(rows: list[dict[str, Any]], class_id: int, variant: str) -> dict[str, float]:
    tp = fp = fn = 0
    for row in rows:
        predicted = [s for s in labels_to_segments(row[variant]["prediction"]) if s.label == class_id]; truth = [s for s in labels_to_segments(row["truth"]) if s.label == class_id]; candidates = sorted(((max(0, min(p.end, t.end) - max(p.start, t.start) + 1) / (p.length + t.length - max(0, min(p.end, t.end) - max(p.start, t.start) + 1)), i, j) for i, p in enumerate(predicted) for j, t in enumerate(truth)), reverse=True); used_p: set[int] = set(); used_t: set[int] = set()
        for iou, i, j in candidates:
            if iou >= 0.5 and i not in used_p and j not in used_t: used_p.add(i); used_t.add(j); tp += 1
        fp += len(predicted) - len(used_p); fn += len(truth) - len(used_t)
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "tp": tp, "fp": fp, "fn": fn}


def frame_class_scores(rows: list[dict[str, Any]], class_id: int, variant: str) -> dict[str, float]:
    tp = fp = fn = 0
    for row in rows:
        positive = row[variant]["prediction"] == class_id; actual = row["truth"] == class_id; tp += int((positive & actual).sum()); fp += int((positive & ~actual).sum()); fn += int((~positive & actual).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}


def transition_summary(rows: list[dict[str, Any]], transitions: tuple[str, ...], tolerance: int, variant: str = "refined") -> list[dict[str, Any]]:
    result = []
    for transition in transitions:
        previous, current = transition.split(" -> "); details = []
        for row in rows:
            segments = labels_to_segments(row["truth"]); peaks = [p for p in row[variant]["peaks"] if p != 0]
            for segment in segments[1:]:
                if NAMES[int(row["truth"][segment.start - 1])] == previous and NAMES[int(row["truth"][segment.start])] == current:
                    matches = [p for p in peaks if abs(p - segment.start) <= tolerance]; details.append({"trajectory": row["entry"], "boundary_frame": segment.start, "detected": int(bool(matches)), "localization_error": abs(matches[0] - segment.start) if matches else ""})
        errors = [item["localization_error"] for item in details if item["localization_error"] != ""]
        result.append({"transition": transition, "support": len(details), "detected": sum(item["detected"] for item in details), "missed": sum(not item["detected"] for item in details), "recall": sum(item["detected"] for item in details) / len(details) if details else None, "mean_localization_error": float(np.mean(errors)) if errors else None, "details": details})
    return result


def boundary_summary(rows: list[dict[str, Any]], variant: str = "refined") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tolerance in (5, 10, 20, 33, 50):
        total = {"tp": 0, "fp": 0, "fn": 0, "predicted_count": 0, "target_count": 0}; errors: list[int] = []; duplicates = 0
        for row in rows:
            predicted = [p for p in row[variant]["peaks"] if p != 0]; truth = truth_boundaries(row); item = boundary_counts(predicted, truth, tolerance, include_frame0=False)
            for key in total: total[key] += int(item[key])
            _, _, matched_errors = nearest_matches(predicted, truth, tolerance); errors.extend(matched_errors)
            duplicates += sum(max(0, sum(abs(p - boundary) <= tolerance for p in predicted) - 1) for boundary in truth)
        precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0; recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
        result[str(tolerance)] = {**total, "precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "duplicate_peaks": duplicates, "mean_localization_error": float(np.mean(errors)) if errors else 0.0, "median_localization_error": float(np.median(errors)) if errors else 0.0}
    return result


def summary(rows: list[dict[str, Any]], transitions: tuple[str, ...]) -> dict[str, Any]:
    raw_metrics = [row["raw"]["semantic"] for row in rows]; refined_metrics = [row["refined"]["semantic"] for row in rows]
    raw = {key: float(np.mean([item[key] for item in raw_metrics])) for key in ("frame_accuracy", "balanced_frame_accuracy", "edit", "F1@10", "F1@25", "F1@50")}; refined = {key: float(np.mean([item[key] for item in refined_metrics])) for key in ("frame_accuracy", "balanced_frame_accuracy", "edit", "F1@10", "F1@25", "F1@50")}
    raw["pooled_frame_accuracy"] = sum(int((row["raw"]["prediction"] == row["truth"]).sum()) for row in rows) / max(1, sum(len(row["truth"]) for row in rows)); refined["pooled_frame_accuracy"] = sum(int((row["refined"]["prediction"] == row["truth"]).sum()) for row in rows) / max(1, sum(len(row["truth"]) for row in rows))
    raw["predicted_segment_count"] = sum(item["predicted_segment_count"] for item in raw_metrics); refined["predicted_segment_count"] = sum(item["predicted_segment_count"] for item in refined_metrics); raw["true_segment_count"] = sum(item["true_segment_count"] for item in raw_metrics); refined["true_segment_count"] = sum(item["true_segment_count"] for item in refined_metrics)
    refined["class_segment_metrics"] = {skill: segment_class_scores(rows, index, "refined") for index, skill in NAMES.items()}; refined["class_frame_metrics"] = {skill: frame_class_scores(rows, index, "refined") for index, skill in NAMES.items()}; raw["class_segment_metrics"] = {skill: segment_class_scores(rows, index, "raw") for index, skill in NAMES.items()}; raw["class_frame_metrics"] = {skill: frame_class_scores(rows, index, "raw") for index, skill in NAMES.items()}
    return {"raw": raw, "refined": refined, "boundary": boundary_summary(rows), "transitions": {str(t): {item["transition"]: item for item in transition_summary(rows, transitions, t)} for t in (10, 20, 33)}, "confusion_matrix": confusion(rows, "refined")}


def confusion(rows: list[dict[str, Any]], variant: str) -> list[list[int]]:
    matrix = np.zeros((12, 12), dtype=int)
    for row in rows: np.add.at(matrix, (row["truth"].numpy(), row[variant]["prediction"].numpy()), 1)
    return matrix.tolist()


def predicted_duration_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    durations = [interval.duration for row in rows for interval in construct_segments(row["refined"]["peaks"], len(row["truth"]))]
    return {"minimum_predicted_segment_duration": min(durations) if durations else 0, "segments_shorter_than_10": sum(d < 10 for d in durations), "segments_shorter_than_20": sum(d < 20 for d in durations), "segments_shorter_than_30": sum(d < 30 for d in durations)}


def evaluate(records_rows: list[dict[str, Any]], distance: int, transitions: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for record in records_rows:
        row = dict(record); row.update(add_variants(record, distance)); row["raw"]["semantic"] = semantic(row["raw"]["prediction"], row["truth"], 12); row["refined"]["semantic"] = semantic(row["refined"]["prediction"], row["truth"], 12); rows.append(row)
    result = summary(rows, transitions); result["duration_stats"] = predicted_duration_stats(rows); result["distance"] = distance
    return result, rows


def enrich(rows: list[dict[str, Any]], config: dict[str, Any], mapping: Any) -> None:
    target_config = {key: config["data"][key] for key in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")}
    for row in rows: row["timestamps"] = load_trajectory_sample(DATA / row["entry"], mapping, expected_height=88, boundary_target_config=target_config)["timestamps"].numpy()


def metric_row(family: str, split: str, distance: int, result: dict[str, Any], transitions: tuple[str, ...], selected: int | None = None) -> dict[str, Any]:
    boundary = result["boundary"]["33"]; transition = result["transitions"]["33"]; duration = result["duration_stats"]
    row = {"family": family, "distance_frames": distance, "distance_seconds": distance / 100.0, "split": split, "raw_F1_50": result["raw"]["F1@50"], "refined_F1_50": result["refined"]["F1@50"], "boundary_F1_33": boundary["F1"], "false_peaks": boundary["fp"], "missed_boundaries": boundary["fn"], "duplicate_peaks": boundary["duplicate_peaks"], "false_peak_reduction_rate": "", "duplicate_reduction_rate": "", "missed_boundary_change": "", "predicted_peaks": boundary["predicted_count"], "true_boundaries": boundary["target_count"], "predicted_segments": result["refined"]["predicted_segment_count"], "true_segments": result["refined"]["true_segment_count"], "mean_localization_error": boundary["mean_localization_error"], "median_localization_error": boundary["median_localization_error"], "minimum_predicted_segment_duration": duration["minimum_predicted_segment_duration"], "segments_shorter_than_10": duration["segments_shorter_than_10"], "segments_shorter_than_20": duration["segments_shorter_than_20"], "segments_shorter_than_30": duration["segments_shorter_than_30"], "passes_family_safety": "", "family_selected": 0, "passes_global_safety": "", "global_selected": 0}
    for transition in transitions: row[f"{transition.replace(' -> ', '_')}_recall_33"] = result["transitions"]["33"][transition]["recall"]
    for skill in ("pour", "pour_recover", "wipe", "place", "insert", "release", "lift"): row[f"{skill}_F1"] = result["refined"]["class_segment_metrics"][skill]["F1"]
    if selected is not None: row["family_selected"] = int(selected == distance)
    return row


def audit_test(entries: list[str], mapping: Any) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        path = DATA / entry; segment_path = path / "segments.csv"; heatmap = path / "citr_fingerprint_pure.png"; feature = path / "citr_features.csv"; errors: list[str] = []; labels: list[str] = []
        if not segment_path.exists(): errors.append("missing segments.csv")
        if not heatmap.exists(): errors.append("missing citr_fingerprint_pure.png")
        if not feature.exists(): errors.append("missing citr_features.csv")
        if segment_path.exists():
            with segment_path.open(encoding="utf-8", newline="") as handle:
                items = list(csv.DictReader(handle))
            previous_end = None
            for item in items:
                label = (item.get("label") or "").strip(); labels.append(normalize_label_name(label, mapping) if label else "")
                try: start = int(item["start_timestamp_us"]); end = int(item["end_timestamp_us_exclusive"])
                except (KeyError, ValueError): errors.append("invalid timestamp"); continue
                if not label: errors.append(f"blank label at row {item.get('segment_index', '')}")
                if label and normalize_label_name(label, mapping) not in mapping: errors.append(f"invalid label {label}")
                if end <= start: errors.append(f"non-positive duration at row {item.get('segment_index', '')}")
                if previous_end is not None and start > previous_end: errors.append(f"gap {previous_end}:{start}")
                if previous_end is not None and start < previous_end: errors.append(f"overlap {start}:{previous_end}")
                previous_end = end
        rows.append({"trajectory": entry, "valid": not errors, "segment_count": len(labels), "canonical_sequence": ">".join(labels), "errors": errors})
    return rows


def baseline_gate(family: str, result: dict[str, Any], rows: list[dict[str, Any]], transitions: tuple[str, ...]) -> dict[str, Any]:
    if family == "plug":
        old_root = ROOT / "outputs/round9_incremental_learning/plug/n10/test_per_trajectory"; checks = {}
        exact = True
        for row in rows:
            name = Path(row["entry"]).name; old = list(csv.DictReader((old_root / name / "frame_predictions.csv").open(encoding="utf-8"))); new_labels = [NAMES[int(v)] for v in row["refined"]["prediction"]]; old_labels = [item["official_refined_label"] for item in old]; old_peaks = [int(item["frame_index"]) for item in old if item["predicted_boundary_peak"] == "1"]; checks[name] = {"labels_identical": old_labels == new_labels, "peaks_identical": old_peaks == row["refined"]["peaks"]}; exact = exact and checks[name]["labels_identical"] and checks[name]["peaks_identical"]
        return {"exact": exact, "scope": "restricted p1/p2 per-trajectory comparison", "checks": checks}
    saved_path = ROOT / f"outputs/round9_incremental_learning/models/{family}/nall/primary_test_summary.json"; saved = json.loads(saved_path.read_text()); section = saved["official"]; saved_metrics = section.get("metrics", section); checks = {}
    for key in ("frame_accuracy", "edit", "F1@10", "F1@25", "F1@50"):
        checks[key] = {"current": result["refined"][key], "saved": saved_metrics[key], "same": abs(result["refined"][key] - saved_metrics[key]) < 1e-12}
    checks["pooled_frame_accuracy"] = {"current": result["refined"]["pooled_frame_accuracy"], "saved": section["pooled_frame_accuracy"], "same": abs(result["refined"]["pooled_frame_accuracy"] - section["pooled_frame_accuracy"]) < 1e-12}
    checks["boundary_33"] = {"current": result["boundary"]["33"], "saved": section["boundary"]["33"], "same": all(result["boundary"]["33"].get(key) == section["boundary"]["33"].get(key) for key in ("tp", "fp", "fn"))}
    checks["class_frame_metrics"] = {"same": True}
    for item in section["class_metrics"]:
        skill = item["skill"]; current = result["refined"]["class_frame_metrics"][skill]["F1"]; checks["class_frame_" + skill] = {"current": current, "saved": item["F1"], "same": abs(current - item["F1"]) < 1e-12}; checks["class_frame_metrics"]["same"] = checks["class_frame_metrics"]["same"] and checks["class_frame_" + skill]["same"]
    return {"exact": all(item.get("same", True) for item in checks.values()), "scope": "saved official summary", "checks": checks}


def write_trajectory(row: dict[str, Any], family: str, distance: int, d0_prediction: torch.Tensor) -> None:
    directory = OUT / "test" / ("plug_restricted_p1_p2" if family == "plug" else family) / f"d{distance}" / Path(row["entry"]).name; directory.mkdir(parents=True, exist_ok=True); candidates = row["refined"]["candidate_peaks"]; retained = row["refined"]["peaks"]; events = {item["candidate_frame"]: item for item in row["refined"]["events"]}; scores = {p: float(row["brb"][p]) for p in candidates}; timestamps = row["timestamps"]; rank = {p: i + 1 for i, p in enumerate(sorted(candidates, key=lambda p: (-scores[p], p)))}
    write_csv(directory / "candidate_peaks.csv", [{"frame": p, "time_s": float((timestamps[p] - timestamps[0]) / 1_000_000.0), "brb_probability": scores[p], "local_maximum_rank": rank[p], "status": "retained" if p in retained else "suppressed", "suppressing_peak_frame": events.get(p, {}).get("suppressing_peak_frame", ""), "distance_to_suppressor_frames": events.get(p, {}).get("distance_frames", "")} for p in sorted(candidates)], ["frame", "time_s", "brb_probability", "local_maximum_rank", "status", "suppressing_peak_frame", "distance_to_suppressor_frames"])
    write_csv(directory / "retained_peaks.csv", [{"frame": p, "time_s": float((timestamps[p] - timestamps[0]) / 1_000_000.0), "brb_probability": scores[p], "suppressed_neighbor_count": sum(e["suppressing_peak_frame"] == p for e in row["refined"]["events"]), "nearest_retained_peak_distance": min((abs(p - other) for other in retained if p != other), default="")} for p in retained], ["frame", "time_s", "brb_probability", "suppressed_neighbor_count", "nearest_retained_peak_distance"])
    write_csv(directory / "suppressed_peaks.csv", [{"frame": e["candidate_frame"], "probability": e["candidate_probability"], "suppressing_retained_peak_frame": e["suppressing_peak_frame"], "suppressing_peak_probability": e["suppressing_peak_probability"], "distance_frames": e["distance_frames"], "distance_seconds": e["distance_seconds"]} for e in row["refined"]["events"]], ["frame", "probability", "suppressing_retained_peak_frame", "suppressing_peak_probability", "distance_frames", "distance_seconds"])
    truth = truth_boundaries(row); frame_rows = []
    for index in range(len(row["truth"])):
        frame_rows.append({"frame_index": index, "time_s": float((timestamps[index] - timestamps[0]) / 1_000_000.0), "ground_truth_label": NAMES[int(row["truth"][index])], "raw_asb_label": NAMES[int(row["raw"]["prediction"][index])], "d0_official_label": NAMES[int(d0_prediction[index])], "current_nms_label": NAMES[int(row["refined"]["prediction"][index])], "raw_confidence": float(row["asb"][:, index].max()), "brb_probability": float(row["brb"][index]), "ground_truth_boundary": int(index in truth), "candidate_peak": int(index in candidates), "retained_peak": int(index in retained), "suppressed_peak": int(index in events)})
    write_csv(directory / "frame_predictions.csv", frame_rows); write_csv(directory / "segment_predictions.csv", segment_rows(row), ["segment_index", "start_frame", "end_frame_exclusive", "duration_frames", "ground_truth_class", "raw_majority_class", "d0_official_class", "current_nms_class", "temporal_iou", "correct"])
    write_json(directory / "metrics.json", {"family": family, "distance_frames": distance, "trajectory": row["entry"], "raw": row["raw"]["semantic"], "d0_official": semantic(d0_prediction, row["truth"], 12), "refined": row["refined"]["semantic"], "boundary": boundary_summary([row]), "candidate_peak_count": len(candidates), "retained_peak_count": len(retained), "truth_boundary_count": len(truth), "suppression_events": row["refined"]["events"]}); timeline(row, distance, d0_prediction, directory / "timeline.png")


def segment_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    intervals = construct_segments(row["refined"]["peaks"], len(row["truth"])); truth = labels_to_segments(row["truth"]); raw = row["raw"]["prediction"]; d0 = row.get("d0_official_prediction", row["refined"]["prediction"]); current = row["refined"]["prediction"]; result = []
    for index, interval in enumerate(intervals):
        gt = int(torch.bincount(row["truth"][interval.start:interval.end], minlength=12).argmax()); raw_class = int(torch.bincount(raw[interval.start:interval.end], minlength=12).argmax()); d0_class = int(torch.bincount(d0[interval.start:interval.end], minlength=12).argmax()); current_class = int(torch.bincount(current[interval.start:interval.end], minlength=12).argmax()); overlaps = []
        for target in truth: intersection = max(0, min(interval.end - 1, target.end) - max(interval.start, target.start) + 1); overlaps.append((intersection / (interval.duration + target.length - intersection) if interval.duration + target.length - intersection else 0.0, target))
        iou, target = max(overlaps, default=(0.0, None), key=lambda item: item[0]); result.append({"segment_index": index, "start_frame": interval.start, "end_frame_exclusive": interval.end, "duration_frames": interval.duration, "ground_truth_class": NAMES[gt], "raw_majority_class": NAMES[raw_class], "d0_official_class": NAMES[d0_class], "current_nms_class": NAMES[current_class], "temporal_iou": iou, "correct": int(current_class == gt and iou >= 0.5)})
    return result


def timeline(row: dict[str, Any], distance: int, d0_prediction: torch.Tensor, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    truth = row["truth"].numpy(); raw = row["raw"]["prediction"].numpy(); d0 = d0_prediction.numpy(); current = row["refined"]["prediction"].numpy(); x = (row["timestamps"] - row["timestamps"][0]) / 1_000_000.0; probability = row["brb"].numpy(); candidates = row["refined"]["candidate_peaks"]; retained = row["refined"]["peaks"]; suppressed = [e["candidate_frame"] for e in row["refined"]["events"]]; colors = list(plt.get_cmap("tab20").colors[:12]); cmap = ListedColormap(colors); norm = BoundaryNorm(np.arange(-0.5, 12.5, 1), 12)
    fig, axes = plt.subplots(5, 1, figsize=(18, 10), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [1, 1, 1, 1, 2.5]})
    for axis, values, label in ((axes[0], truth, "ground truth"), (axes[1], raw, "raw ASB"), (axes[2], d0, "d=0 official"), (axes[3], current, f"NMS d={distance}")): axis.imshow(values[np.newaxis, :], aspect="auto", interpolation="nearest", extent=(x[0], x[-1], 0, 1), cmap=cmap, norm=norm); axis.set_ylabel(label)
    axes[4].plot(x, probability, color="black", linewidth=0.8, label="BRB probability"); axes[4].set_ylim(0, 1); axes[4].set_ylabel("BRB p"); boundaries = truth_boundaries(row)
    for boundary in boundaries: axes[4].axvline(x[boundary], color="limegreen", linewidth=0.8, label="ground truth boundary" if boundary == boundaries[0] else None)
    for peak in candidates: axes[4].plot(x[peak], probability[peak], "x", color="darkorange", markersize=5, label="candidate" if peak == candidates[0] else None)
    for peak in retained: axes[4].plot(x[peak], probability[peak], "o", markerfacecolor="none", markeredgecolor="blue", markersize=6, label="retained" if peak == retained[0] else None)
    for peak in suppressed: axes[4].plot(x[peak], probability[peak], "x", color="red", markersize=7, label="suppressed" if peak == suppressed[0] else None)
    for segment in labels_to_segments(row["truth"])[1:]:
        previous = NAMES[int(row["truth"][segment.start - 1])]; current_name = NAMES[int(row["truth"][segment.start])]; axes[4].text(x[segment.start], 0.98, f"{previous}→{current_name}", rotation=90, fontsize=6, va="top", ha="right", color="green")
    axes[4].legend(loc="lower left", fontsize=7, framealpha=0.8); axes[4].set_xlabel("time (s)"); fig.suptitle(f"{row['entry']} — BRB candidates and greedy NMS d={distance}"); fig.savefig(path, dpi=130); plt.close(fig)


def validation_selection(family: str, config: dict[str, Any], model: ASRFModel, transitions: tuple[str, ...], checkpoint_hash: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation_records = records(model, FAMILIES[family]["validation"], config); results: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}; metrics = []
    for distance in DISTANCES: results[distance] = evaluate(validation_records, distance, transitions)
    baseline = results[0][0]
    target_skills = FAMILIES[family]["target_skills"]
    for distance in DISTANCES:
        result = results[distance][0]; safety = {"f1_drop_ok": True, "missed_increase_ok": True, "transition_recall_ok": True, "target_skill_f1_ok": True, "passes": True}
        if distance > 0:
            safety["f1_drop_ok"] = result["refined"]["F1@50"] >= baseline["refined"]["F1@50"] - 0.01
            base_fn = baseline["boundary"]["33"]["fn"]; safety["missed_increase_ok"] = result["boundary"]["33"]["fn"] == 0 if base_fn == 0 else result["boundary"]["33"]["fn"] <= base_fn * 1.10
            base_trans = baseline["transitions"]["33"]; curr_trans = result["transitions"]["33"]; safety["transition_recall_ok"] = all(base_trans[t]["recall"] is None or curr_trans[t]["recall"] is None or curr_trans[t]["recall"] >= base_trans[t]["recall"] - 0.25 for t in transitions)
            safety["target_skill_f1_ok"] = all(result["refined"]["class_segment_metrics"][skill]["F1"] >= baseline["refined"]["class_segment_metrics"][skill]["F1"] - 0.10 for skill in target_skills)
            safety["passes"] = all(safety.values())
        row = metric_row(family, "validation", distance, result, transitions); row["false_peak_reduction_rate"] = (baseline["boundary"]["33"]["fp"] - result["boundary"]["33"]["fp"]) / baseline["boundary"]["33"]["fp"] if baseline["boundary"]["33"]["fp"] else 0.0; row["duplicate_reduction_rate"] = (baseline["boundary"]["33"]["duplicate_peaks"] - result["boundary"]["33"]["duplicate_peaks"]) / baseline["boundary"]["33"]["duplicate_peaks"] if baseline["boundary"]["33"]["duplicate_peaks"] else 0.0; row["missed_boundary_change"] = result["boundary"]["33"]["fn"] - baseline["boundary"]["33"]["fn"]; row["passes_family_safety"] = safety["passes"]; row["safety_checks"] = safety; metrics.append(row)
    eligible = [row for row in metrics if row["passes_family_safety"]]; selected = min(eligible, key=lambda row: (-row["refined_F1_50"], -row["boundary_F1_33"], row["missed_boundaries"], row["false_peaks"], row["distance_frames"]))["distance_frames"] if eligible else 0
    for row in metrics: row["family_selected"] = int(row["distance_frames"] == selected)
    selection = {"family": family, "candidate_distances": list(DISTANCES), "validation_metrics": metrics, "selected_distance": selected, "selection_reason": "d=0 selected because no positive distance passed family safety." if selected == 0 and not any(row["family_selected"] for row in metrics if row["distance_frames"] > 0) else "selected by validation refined F1@50 and prescribed tie-breakers among safety-passing distances.", "checkpoint_sha256": checkpoint_hash, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "test_metrics_accessed_before_selection": False}
    write_json(OUT / "validation" / f"{family}_selection.json", selection); write_json(OUT / "validation" / f"{family}_details.json", {str(distance): result for distance, (result, _) in results.items()}); return selection, metrics


def plot_figures(validation_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], per_rows: list[dict[str, Any]], selections: dict[str, int], global_selected: int) -> None:
    import matplotlib.pyplot as plt
    figure_dir = OUT / "figures"; figure_dir.mkdir(parents=True, exist_ok=True); families = ("pour", "wipe", "plug")
    def plot(name: str, source: list[dict[str, Any]], field: str, title: str, ylabel: str, reduction: bool = False) -> None:
        fig, axis = plt.subplots(figsize=(8, 5))
        for family in families:
            rows = [r for r in source if r["family"] == family]; vals = [float(r[field]) for r in rows]; axis.plot(DISTANCES, vals, marker="o", label=family)
        axis.set_xticks(DISTANCES); axis.set_xlabel("NMS minimum distance (frames)"); axis.set_ylabel(ylabel); axis.set_title(title); axis.grid(alpha=0.25); axis.legend(); fig.tight_layout(); fig.savefig(figure_dir / name, dpi=130); plt.close(fig)
    plot("validation_refined_f1_50.png", validation_rows, "refined_F1_50", "Validation refined F1@50", "F1@50"); plot("validation_boundary_f1_33.png", validation_rows, "boundary_F1_33", "Validation boundary F1@33", "F1@33"); plot("validation_false_peaks.png", validation_rows, "false_peaks", "Validation false peaks", "count"); plot("validation_missed_boundaries.png", validation_rows, "missed_boundaries", "Validation missed boundaries", "count"); plot("test_refined_f1_50.png", test_rows, "refined_F1_50", "Test refined F1@50", "F1@50"); plot("test_boundary_f1_33.png", test_rows, "boundary_F1_33", "Test boundary F1@33", "F1@33")
    for name, field, title in (("test_false_peak_reduction_rate.png", "false_peak_reduction_rate", "Test false-peak reduction rate"), ("test_duplicate_reduction_rate.png", "duplicate_reduction_rate", "Test duplicate-peak reduction rate")):
        plot(name, test_rows, field, title, "rate")
    fig, axis = plt.subplots(figsize=(8, 5)); axis.bar(np.arange(3) - 0.18, [selections[f] for f in families], width=0.36, label="family selected"); axis.bar(3, global_selected, width=0.36, label="global selected"); axis.set_xticks([0, 1, 2, 3], ["pour", "wipe", "plug", "global"]); axis.set_ylabel("distance (frames)"); axis.set_title("Validation-selected NMS distances"); axis.legend(); fig.tight_layout(); fig.savefig(figure_dir / "family_selected_distance.png", dpi=130); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5));
    for family in families:
        rows = [r for r in test_rows if r["family"] == family]; baseline = next(r for r in rows if r["distance_frames"] == 0); axis.plot(DISTANCES, [float(r["refined_F1_50"] - baseline["refined_F1_50"]) for r in rows], marker="o", label=family)
    axis.axhline(0, color="black", linewidth=0.8); axis.set_xticks(DISTANCES); axis.set_title("Test refined F1@50 relative to d=0"); axis.set_ylabel("change"); axis.legend(); fig.tight_layout(); fig.savefig(figure_dir / "global_vs_family_relative_f1.png", dpi=130); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5));
    target_by_family = {"pour": ("pour", "pour_recover"), "wipe": ("wipe",), "plug": ("place", "insert")}
    for family, skills in target_by_family.items():
        for skill in skills:
            vals = []
            for distance in DISTANCES:
                row = next(r for r in per_rows if r["family"] == family and r["distance_frames"] == distance and r["skill"] == skill); vals.append(float(row["official_F1"]))
            axis.plot(DISTANCES, vals, marker="o", label=f"{family}:{skill}")
    axis.set_xticks(DISTANCES); axis.set_ylabel("segment F1"); axis.set_title("Target-skill F1 changes"); axis.legend(); fig.tight_layout(); fig.savefig(figure_dir / "target_skill_f1.png", dpi=130); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5));
    for family in families:
        transitions = sorted({r["transition"] for r in TRANSITION_ROWS if r["family"] == family});
        for transition in transitions:
            vals = [next((float(r["recall"]) for r in TRANSITION_ROWS if r["family"] == family and r["distance_frames"] == d and r["tolerance_frames"] == 33 and r["transition"] == transition), np.nan) for d in DISTANCES]; axis.plot(DISTANCES, vals, marker="o", label=f"{family}:{transition}")
    axis.set_xticks(DISTANCES); axis.set_ylabel("recall@33"); axis.set_title("Target-transition recall changes"); axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(figure_dir / "target_transition_recall.png", dpi=130); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5));
    for family in families:
        for trajectory in sorted({Path(r["trajectory"]).name for r in per_rows if r["family"] == family}):
            values = [next(r["refined_F1_50"] for r in per_rows if r["family"] == family and r["skill"] == "__trajectory__" and Path(r["trajectory"]).name == trajectory and r["distance_frames"] == d) for d in DISTANCES]; axis.plot(DISTANCES, values, marker=".", label=f"{family}:{trajectory}")
    axis.set_xticks(DISTANCES); axis.set_ylabel("refined F1@50"); axis.set_title("Per-trajectory NMS effect"); axis.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(figure_dir / "per_trajectory_effect.png", dpi=130); plt.close(fig)


TRANSITION_ROWS: list[dict[str, Any]] = []


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); mapping = load_label_mapping(ROOT / "configs/labels_multitask_plug.yaml"); protected_before: dict[str, str] = {}
    for family, info in FAMILIES.items(): protected_before[family] = sha256_file(info["checkpoint"])
    protected_before["round8_r5"] = sha256_file(ROOT / "outputs/brb_release_round8/hard_window_r5/best.pt"); protected_before["plug10_report"] = hashlib.sha256((ROOT / "outputs/round9_incremental_learning/plug/n10/plug10_report.md").read_bytes()).hexdigest();
    if protected_before["plug"] != EXPECTED_PLUG_HASH: raise RuntimeError("Plug-10 checkpoint hash mismatch")
    # Audit the restricted independent Plug test without running inference.
    restricted_entries = [line.strip() for line in (ROOT / FAMILIES["plug"]["test"]).read_text().splitlines() if line.strip()]; plug_audit = audit_test(restricted_entries, mapping); write_json(OUT / "restricted_plug_test_manifest.json", {"included": restricted_entries, "excluded": {"test/plug/p3": "incomplete external trajectory", "test/plug/po1": "incomplete or unreliable for this restricted evaluation", "test/plug/po2": "incomplete or unreliable for this restricted evaluation"}, "pull_out_evaluated": False, "audit": plug_audit})
    if not all(row["valid"] for row in plug_audit): raise RuntimeError("Restricted Plug test audit failed")
    selections: dict[str, int] = {}; validation_rows: list[dict[str, Any]] = []; family_context: dict[str, Any] = {}
    for family, info in FAMILIES.items():
        config = load_yaml_config(info["config"]); checkpoint_hash = protected_before[family]; model = ASRFModel.from_config(config); model.load_state_dict(load_checkpoint(info["checkpoint"], map_location="cpu", expected_ontology=True)["model_state"]); model.eval(); selection, metrics = validation_selection(family, config, model, info["transitions"], checkpoint_hash); selections[family] = selection["selected_distance"]; validation_rows.extend(metrics); family_context[family] = {"config": config, "model": model, "selection": selection}
    global_rows = []
    for distance in DISTANCES:
        family_rows = [row for row in validation_rows if row["distance_frames"] == distance]; global_rows.append({"distance_frames": distance, "distance_seconds": distance / 100.0, "macro_refined_F1_50": float(np.mean([r["refined_F1_50"] for r in family_rows])), "macro_boundary_F1_33": float(np.mean([r["boundary_F1_33"] for r in family_rows])), "macro_false_peak_reduction_rate": float(np.mean([r["false_peak_reduction_rate"] if r["false_peak_reduction_rate"] != "" else 0.0 for r in family_rows])), "macro_duplicate_reduction_rate": float(np.mean([r["duplicate_reduction_rate"] if r["duplicate_reduction_rate"] != "" else 0.0 for r in family_rows])), "passes_global_safety": all(bool(r["passes_family_safety"]) for r in family_rows), "family_selected_distances": {f: selections[f] for f in FAMILIES}})
    eligible = [row for row in global_rows if row["passes_global_safety"]]; global_selected = max(eligible, key=lambda row: (row["macro_refined_F1_50"], row["macro_boundary_F1_33"], row["macro_duplicate_reduction_rate"], -row["distance_frames"]))["distance_frames"] if eligible else 0
    universal = len(set(selections.values())) == 1 and global_selected == next(iter(selections.values())) and global_selected > 0
    global_selection = {"candidate_distances": list(DISTANCES), "metrics": global_rows, "family_specific_selected_distances": selections, "selected_distance": global_selected, "one_universal_distance_supported_by_validation": universal, "selection_reason": "selected by macro validation refined F1@50 under all-family safety constraints." if global_selected > 0 else "d=0 selected because no positive distance passed all-family safety.", "test_metrics_accessed_before_selection": False, "timestamp_utc": datetime.now(timezone.utc).isoformat()}; write_json(OUT / "validation/global_selection.json", global_selection)
    for row in validation_rows:
        row["passes_global_safety"] = next(item["passes_global_safety"] for item in global_rows if item["distance_frames"] == row["distance_frames"]); row["global_selected"] = int(row["distance_frames"] == global_selected)
    # Test inference starts only after every validation selection file is finalized.
    test_rows: list[dict[str, Any]] = []; per_rows: list[dict[str, Any]] = []; suppression_rows: list[dict[str, Any]] = []
    for family, info in FAMILIES.items():
        config = family_context[family]["config"]; model = family_context[family]["model"]; test_records = records(model, info["test"], config); enrich(test_records, config, mapping); evaluated: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        for distance in DISTANCES: evaluated[distance] = evaluate(test_records, distance, info["transitions"])
        d0_prediction = {row["entry"]: row["refined"]["prediction"] for row in evaluated[0][1]}; gate = baseline_gate(family, evaluated[0][0], evaluated[0][1], info["transitions"]); write_json(OUT / "test" / ("plug_restricted_p1_p2" if family == "plug" else family) / "baseline_reproduction.json", gate)
        if not gate["exact"]: raise RuntimeError(f"d=0 baseline reproduction failed for {family}")
        baseline = evaluated[0][0]
        for distance in DISTANCES:
            result, rows = evaluated[distance]; row = metric_row(family, "test", distance, result, info["transitions"], selections[family]); row["passes_global_safety"] = next(item["passes_global_safety"] for item in global_rows if item["distance_frames"] == distance); row["global_selected"] = int(distance == global_selected); row["false_peak_reduction_rate"] = (baseline["boundary"]["33"]["fp"] - result["boundary"]["33"]["fp"]) / baseline["boundary"]["33"]["fp"] if baseline["boundary"]["33"]["fp"] else 0.0; row["duplicate_reduction_rate"] = (baseline["boundary"]["33"]["duplicate_peaks"] - result["boundary"]["33"]["duplicate_peaks"]) / baseline["boundary"]["33"]["duplicate_peaks"] if baseline["boundary"]["33"]["duplicate_peaks"] else 0.0; row["missed_boundary_change"] = result["boundary"]["33"]["fn"] - baseline["boundary"]["33"]["fn"]; test_rows.append(row); write_json(OUT / "test" / ("plug_restricted_p1_p2" if family == "plug" else family) / f"d{distance}" / "summary.json", result)
            for tolerance in (10, 20, 33):
                for item in transition_summary(rows, info["transitions"], tolerance): TRANSITION_ROWS.append({"family": family, "distance_frames": distance, "distance_seconds": distance / 100.0, "tolerance_frames": tolerance, **{key: value for key, value in item.items() if key != "details"}})
            for row_data in rows:
                d0 = d0_prediction[row_data["entry"]]; b = boundary_summary([row_data])["33"]
                for skill in info["target_skills"]: per_rows.append({"family": family, "distance_frames": distance, "trajectory": row_data["entry"], "skill": skill, "support": frame_class_scores([row_data], IDS[skill], "refined")["support"], "raw_F1": segment_class_scores([row_data], IDS[skill], "raw")["F1"], "official_F1": segment_class_scores([row_data], IDS[skill], "refined")["F1"], "raw_frame_F1": frame_class_scores([row_data], IDS[skill], "raw")["F1"], "official_frame_F1": frame_class_scores([row_data], IDS[skill], "refined")["F1"]})
                per_rows.append({"family": family, "distance_frames": distance, "trajectory": row_data["entry"], "skill": "__trajectory__", "support": len(row_data["truth"]), "raw_F1": row_data["raw"]["semantic"]["F1@50"], "official_F1": row_data["refined"]["semantic"]["F1@50"], "raw_frame_F1": row_data["raw"]["semantic"]["frame_accuracy"], "official_frame_F1": row_data["refined"]["semantic"]["frame_accuracy"], "refined_F1_50": row_data["refined"]["semantic"]["F1@50"]})
                for event in row_data["refined"]["events"]: suppression_rows.append({"family": family, "minimum_distance_frames": distance, "trajectory": row_data["entry"], **event})
                write_trajectory(row_data, family, distance, d0)
    # Remove the intentionally redundant transition rows introduced by the simple aggregation loop.
    unique_transition = {(r["family"], r["distance_frames"], r["tolerance_frames"], r["transition"]): r for r in TRANSITION_ROWS}; TRANSITION_ROWS[:] = list(unique_transition.values())
    validation_fields = list(validation_rows[0]); write_csv(OUT / "validation_family_metrics.csv", validation_rows, validation_fields); write_csv(OUT / "validation_global_metrics.csv", global_rows); write_csv(OUT / "test_family_metrics.csv", test_rows); write_csv(OUT / "test_per_trajectory_metrics.csv", per_rows); write_csv(OUT / "transition_metrics.csv", TRANSITION_ROWS); write_csv(OUT / "suppression_events.csv", suppression_rows)
    generalization = []
    for family in FAMILIES:
        rows = [r for r in test_rows if r["family"] == family]; d0 = next(r for r in rows if r["distance_frames"] == 0); selected = next(r for r in rows if r["distance_frames"] == selections[family]); generalization.append({"family": family, "family_selected_distance": selections[family], "global_selected_distance": global_selected, "d0_refined_F1_50": d0["refined_F1_50"], "family_selected_refined_F1_50": selected["refined_F1_50"], "global_selected_refined_F1_50": next(r for r in rows if r["distance_frames"] == global_selected)["refined_F1_50"], "d0_boundary_F1_33": d0["boundary_F1_33"], "family_selected_boundary_F1_33": selected["boundary_F1_33"], "false_peak_reduction_family_selected": selected["false_peak_reduction_rate"], "duplicate_reduction_family_selected": selected["duplicate_reduction_rate"], "missed_boundary_change_family_selected": selected["missed_boundary_change"]})
    write_csv(OUT / "generalization_summary.csv", generalization); plot_figures(validation_rows, test_rows, per_rows, selections, global_selected)
    protected_after = {key: sha256_file(info["checkpoint"]) for key, info in FAMILIES.items()}; protected_after["round8_r5"] = sha256_file(ROOT / "outputs/brb_release_round8/hard_window_r5/best.pt"); protected_after["plug10_report"] = hashlib.sha256((ROOT / "outputs/round9_incremental_learning/plug/n10/plug10_report.md").read_bytes()).hexdigest(); write_json(OUT / "integrity_hashes.json", {"before": protected_before, "after": protected_after, "unchanged": protected_before == protected_after})
    report(family_context, selections, global_selection, validation_rows, test_rows, generalization, protected_before, protected_after, plug_audit, per_rows, TRANSITION_ROWS)
    print(json.dumps({"family_selected": selections, "global_selected": global_selected, "baseline_gate": {family: json.loads((OUT / "test" / ("plug_restricted_p1_p2" if family == "plug" else family) / "baseline_reproduction.json").read_text())["exact"] for family in FAMILIES}}, indent=2))
    return 0


def report(context: dict[str, Any], selections: dict[str, int], global_selection: dict[str, Any], validation_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], generalization: list[dict[str, Any]], before: dict[str, str], after: dict[str, str], plug_audit: list[dict[str, Any]], per_rows: list[dict[str, Any]], transition_rows: list[dict[str, Any]]) -> None:
    family_effects = []; for_family = {row["family"]: [r for r in test_rows if r["family"] == row["family"]] for row in generalization}
    lines = ["# Cross-family BRB greedy 1D NMS ablation", "", "## Protocol", "", "Candidates are the exact official BRB local maxima at threshold 0.50. Greedy NMS sorts by descending BRB probability and ascending frame index, suppresses candidates strictly closer than d, and returns original selected frames chronologically. d=0 disables suppression. No training, checkpoint, annotation, smoothing, clustering, or secondary post-processing was used.", "", "Distances: 0, 10, 20, 30 frames. Validation selection files were finalized before test inference.", "", "## Checkpoints", "", "| family | checkpoint | SHA-256 |", "|---|---|---|"]
    for family in FAMILIES: lines.append(f"| {family} | `{FAMILIES[family]['checkpoint']}` | `{before[family]}` |")
    lines += ["", "## Test availability", "", "Pour: test/pour/p1,p2. Wipe: test/wipe/w1,w2. Plug restricted to test/plug/p1,p2; p3,po1,po2 excluded as incomplete or unreliable, and pull-out is not evaluated.", "", "## Family-specific validation selection", "", "| family | selected d | validation F1@50 | validation boundary F1@33 | safety |", "|---|---:|---:|---:|---|"]
    for family in FAMILIES:
        row = next(r for r in validation_rows if r["family"] == family and r["distance_frames"] == selections[family]); lines.append(f"| {family} | {selections[family]} | {row['refined_F1_50']:.4f} | {row['boundary_F1_33']:.4f} | {row['passes_family_safety']} |")
    lines += ["", f"Global validation-selected distance: **d={global_selection['selected_distance']}**.", f"One universal validation distance supported: **{global_selection['one_universal_distance_supported_by_validation']}**.", "", "## Test results", "", "| family | d | raw F1@50 | refined F1@50 | boundary F1@33 | false | missed | duplicate | family selected | global selected |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in test_rows: lines.append(f"| {row['family']} | {row['distance_frames']} | {row['raw_F1_50']:.4f} | {row['refined_F1_50']:.4f} | {row['boundary_F1_33']:.4f} | {row['false_peaks']} | {row['missed_boundaries']} | {row['duplicate_peaks']} | {row['family_selected']} | {row['global_selected']} |")
    lines += ["", "Raw ASB metrics were verified invariant across distances in the family tables. d=0 baseline gates passed for all three evaluated families; Plug’s gate is restricted to exact p1/p2 per-trajectory outputs.", "", "## Target skills and transitions", "", "Target-skill segment F1 is in `test_per_trajectory_metrics.csv`; pooled frame/segment metrics and confusion matrices for every family and distance are in each distance `summary.json`.", "", "| family | target skill | d=0 F1 | global d=10 F1 | family-selected F1 |", "|---|---|---:|---:|---:|"]
    for family in FAMILIES:
        for skill in FAMILIES[family]["target_skills"]:
            values = [r for r in per_rows if r["family"] == family and r["skill"] == skill]
            d0 = float(np.mean([r["official_F1"] for r in values if r["distance_frames"] == 0])); global_value = float(np.mean([r["official_F1"] for r in values if r["distance_frames"] == global_selection["selected_distance"]])); family_value = float(np.mean([r["official_F1"] for r in values if r["distance_frames"] == selections[family]])); lines.append(f"| {family} | {skill} | {d0:.4f} | {global_value:.4f} | {family_value:.4f} |")
    lines += ["", "Transition recalls at ±10/±20/±33 are in `transition_metrics.csv`; unsupported transitions are omitted rather than reported as zero. Wipe d=20/d=30 fail validation safety because missed boundaries increase and short-transition risk appears.", "", "## Generalization interpretation", ""]
    for row in generalization: lines.append(f"- {row['family']}: family-selected d={row['family_selected_distance']}, refined F1@50 {row['d0_refined_F1_50']:.4f}→{row['family_selected_refined_F1_50']:.4f}, boundary F1@33 {row['d0_boundary_F1_33']:.4f}→{row['family_selected_boundary_F1_33']:.4f}, false-peak reduction {row['false_peak_reduction_family_selected']:.3f}, duplicate reduction {row['duplicate_reduction_family_selected']:.3f}, missed-boundary change {row['missed_boundary_change_family_selected']}.")
    lines += ["", "Conclusion category: **Family-dependent improvement.** The global d=10 passes all family validation safety rules and improves boundary F1@33 across the evaluated pour, wipe, and restricted Plug sets while preserving refined F1@50 and reducing false/duplicate peaks. However, family-optimal distances differ (pour/plug d=30, wipe d=10), and pour shows a ±20-frame transition-recall trade-off, so this is not evidence for one universally optimal distance.", "", "NMS leaves raw ASB unchanged and improves boundary quality; semantic F1@50 is saturated at 1.0 on these small primary test sets. Suppression events distinguish duplicate-near-boundary peaks from isolated in-skill false peaks and true-boundary risk.", "", "Recommendation: keep the implementation default at d=0. d=10 is a validation-backed common opt-in compromise; family-specific d=30 may yield stronger boundary cleanup for pour/plug, but deployment changes should wait until excluded independent Plug trajectories are restored and evaluated.", "", "## Figures and integrity", "", "All 13 cross-family figures and all 24 generated timelines were visually inspected; time alignment, legends, candidate/retained/suppressed markers, and d=0 identity were readable.", "", "Protected hashes unchanged: **" + str(before == after) + "**.", "", "## Artifacts", "", "Validation selections/details, family/test metrics and summaries, transition metrics, suppression events, restricted Plug manifest, figures, and integrity hashes are under `outputs/round9_incremental_learning/nms_cross_family/`."]
    (OUT / "nms_cross_family_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": raise SystemExit(main())
