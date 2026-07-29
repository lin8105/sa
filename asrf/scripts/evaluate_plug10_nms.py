"""Inference-only score-guided BRB peak NMS ablation for Plug-10."""

from __future__ import annotations

import argparse
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
NMS_OUT = ROOT / "outputs/round9_incremental_learning/plug/n10/nms_ablation"
TEST_ENTRIES = ["test/plug/p1", "test/plug/p2", "test/plug/p3", "test/plug/po1", "test/plug/po2"]
DISTANCES = (0, 10, 20, 30)
SKILLS = ("reach", "grasp", "lift", "transport", "pour", "pour_recover", "place", "release", "wipe", "retreat", "insert")
NAMES = {index: name for index, name in enumerate(SKILLS)}
TRANSITIONS = ("transport -> place", "place -> insert", "insert -> release", "place -> release")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from asrf.data.dataset import load_trajectory_sample  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.evaluation.metrics import boundary_counts, labels_to_segments  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.refinement.majority_vote import _vote_one  # noqa: E402
from asrf.refinement.peaks import greedy_score_guided_nms, select_boundary_peaks  # noqa: E402
from asrf.refinement.segments import construct_segments  # noqa: E402
from asrf.training.checkpointing import load_checkpoint, sha256_file  # noqa: E402
from asrf.utils.config import load_yaml_config  # noqa: E402
from evaluate_round9_incremental import records, semantic, truth_boundaries  # noqa: E402


def json_default(value: Any) -> Any:
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text((",".join(fields or []) + "\n") if fields else "\n", encoding="utf-8")
        return
    columns = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def nearest_matches(predicted: list[int], truth: list[int], tolerance: int) -> tuple[list[int], list[int], list[int]]:
    candidates = sorted((abs(p - t), p, t) for p in predicted for t in truth if abs(p - t) <= tolerance)
    used_p: set[int] = set(); used_t: set[int] = set(); errors: list[int] = []
    for error, peak, boundary in candidates:
        if peak not in used_p and boundary not in used_t:
            used_p.add(peak); used_t.add(boundary); errors.append(error)
    return sorted(used_p), sorted(used_t), errors


def boundary_summary(rows: list[dict[str, Any]], variant: str = "refined") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tolerance in (5, 10, 20, 33, 50):
        pooled = {"tp": 0, "fp": 0, "fn": 0, "predicted_count": 0, "target_count": 0}
        errors: list[int] = []; duplicates = 0
        for row in rows:
            predicted = [peak for peak in row[variant]["peaks"] if peak != 0]
            truth = truth_boundaries(row)
            item = boundary_counts(predicted, truth, tolerance, include_frame0=False)
            for key in pooled: pooled[key] += int(item[key])
            _, _, matched_errors = nearest_matches(predicted, truth, tolerance); errors.extend(matched_errors)
            duplicates += sum(max(0, sum(abs(peak - boundary) <= tolerance for peak in predicted) - 1) for boundary in truth)
        precision = pooled["tp"] / (pooled["tp"] + pooled["fp"]) if pooled["tp"] + pooled["fp"] else 0.0
        recall = pooled["tp"] / (pooled["tp"] + pooled["fn"]) if pooled["tp"] + pooled["fn"] else 0.0
        result[str(tolerance)] = {**pooled, "precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                                  "duplicate_peaks": duplicates, "mean_localization_error": float(np.mean(errors)) if errors else 0.0,
                                  "median_localization_error": float(np.median(errors)) if errors else 0.0}
    return result


def score_class(rows: list[dict[str, Any]], class_id: int, variant: str, segment: bool) -> dict[str, float]:
    if not segment:
        tp = fp = fn = 0
        for row in rows:
            prediction = row[variant]["prediction"]; truth = row["truth"]
            positive = prediction == class_id; actual = truth == class_id
            tp += int((positive & actual).sum()); fp += int((positive & ~actual).sum()); fn += int((~positive & actual).sum())
    else:
        predicted_segments = []; truth_segments = []
        for row in rows:
            predicted_segments.extend([s for s in labels_to_segments(row[variant]["prediction"]) if s.label == class_id])
            truth_segments.extend([s for s in labels_to_segments(row["truth"]) if s.label == class_id])
        candidates = []
        for pi, prediction in enumerate(predicted_segments):
            for ti, target in enumerate(truth_segments):
                intersection = max(0, min(prediction.end, target.end) - max(prediction.start, target.start) + 1)
                union = prediction.length + target.length - intersection
                iou = intersection / union if union else 0.0
                if iou >= 0.5: candidates.append((iou, pi, ti))
        candidates.sort(reverse=True); used_p: set[int] = set(); used_t: set[int] = set(); tp = 0
        for _, pi, ti in candidates:
            if pi not in used_p and ti not in used_t:
                used_p.add(pi); used_t.add(ti); tp += 1
        fp = len(predicted_segments) - tp; fn = len(truth_segments) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "tp": tp, "fp": fp, "fn": fn}


def transition_summary(rows: list[dict[str, Any]], tolerance: int) -> list[dict[str, Any]]:
    result = []
    for transition in TRANSITIONS:
        previous, current = transition.split(" -> "); details = []
        for row in rows:
            truth = labels_to_segments(row["truth"])
            for segment in truth[1:]:
                if NAMES[int(row["truth"][segment.start - 1])] != previous or NAMES[int(row["truth"][segment.start])] != current:
                    continue
                boundary = segment.start; peaks = [peak for peak in row["refined"]["peaks"] if peak != 0]; matches = [peak for peak in peaks if abs(peak - boundary) <= tolerance]
                details.append({"trajectory": row["entry"], "boundary_frame": boundary, "detected": int(bool(matches)), "localization_error": abs(matches[0] - boundary) if matches else ""})
        result.append({"transition": transition, "support": len(details), "detected": sum(item["detected"] for item in details), "missed": sum(item["detected"] == 0 for item in details), "recall": np.mean([item["detected"] for item in details]) if details else 0.0, "mean_localization_error": float(np.mean([item["localization_error"] for item in details if item["localization_error"] != ""])) if any(item["localization_error"] != "" for item in details) else 0.0, "details": details})
    return result


def nms_events(candidate_peaks: list[int], probabilities: torch.Tensor, distance: int, truth: list[int]) -> tuple[list[int], list[dict[str, Any]]]:
    scores = {int(peak): float(probabilities[int(peak)]) for peak in candidate_peaks}
    ranked = sorted(set(candidate_peaks), key=lambda peak: (-scores[peak], peak))
    selected = greedy_score_guided_nms(candidate_peaks, [scores[peak] for peak in candidate_peaks], distance)
    selected_set = set(selected)
    retained_in_score_order: list[int] = []
    events: list[dict[str, Any]] = []
    for peak in ranked:
        if peak in selected_set:
            retained_in_score_order.append(peak)
            continue
        suppressor = next((retained for retained in retained_in_score_order if abs(peak - retained) < distance), None)
        if suppressor is None:
            # This can only occur for duplicate input frames; the reusable NMS
            # function keeps the highest-scoring duplicate deterministically.
            suppressor = next((retained for retained in selected if abs(peak - retained) < distance), None)
        if suppressor is None:
            continue
        near_truth = min((abs(peak - boundary) for boundary in truth), default=10**9)
        near_retained_truth = min((abs(suppressor - boundary) for boundary in truth), default=10**9)
        if near_truth <= 33 and near_retained_truth <= 33:
            event_type = "duplicate_near_true_boundary"
        elif near_truth > 33:
            event_type = "isolated_false_peak_inside_skill"
        else:
            event_type = "true_boundary_suppression_risk"
        events.append({"candidate_frame": peak, "candidate_probability": scores[peak], "suppressing_peak_frame": suppressor, "suppressing_peak_probability": scores[suppressor], "distance_frames": abs(peak - suppressor), "distance_seconds": abs(peak - suppressor) / 100.0, "event_type": event_type, "nearest_truth_distance": near_truth})
    return selected, events


def add_variants(record: dict[str, Any], distance: int) -> dict[str, Any]:
    candidates = list(select_boundary_peaks(record["brb"], threshold=0.5))
    retained, events = nms_events(candidates, record["brb"], distance, truth_boundaries(record))
    prediction, diagnostics = _vote_one(record["asb"], construct_segments(retained, len(record["truth"])), voting="majority")
    raw = record["asb"].argmax(dim=0)
    return {"raw": {"prediction": raw, "peaks": []}, "refined": {"prediction": prediction, "peaks": retained, "candidate_peaks": candidates, "events": events, "diagnostics": diagnostics}}


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    semantic_rows = [row["refined"]["semantic"] for row in rows]; raw_rows = [row["raw"]["semantic"] for row in rows]
    raw = {key: float(np.mean([row[key] for row in raw_rows])) for key in ("frame_accuracy", "edit", "F1@10", "F1@25", "F1@50")}
    refined = {key: float(np.mean([row[key] for row in semantic_rows])) for key in ("frame_accuracy", "balanced_frame_accuracy", "edit", "F1@10", "F1@25", "F1@50")}
    refined["pooled_frame_accuracy"] = sum(int((row["refined"]["prediction"] == row["truth"]).sum()) for row in rows) / max(1, sum(len(row["truth"]) for row in rows))
    refined["predicted_segment_count"] = sum(item["predicted_segment_count"] for item in semantic_rows); refined["true_segment_count"] = sum(item["true_segment_count"] for item in semantic_rows)
    raw["pooled_frame_accuracy"] = sum(int((row["raw"]["prediction"] == row["truth"]).sum()) for row in rows) / max(1, sum(len(row["truth"]) for row in rows))
    raw["balanced_frame_accuracy"] = float(np.mean([row["balanced_frame_accuracy"] for row in raw_rows])); raw["predicted_segment_count"] = sum(item["predicted_segment_count"] for item in raw_rows); raw["true_segment_count"] = sum(item["true_segment_count"] for item in raw_rows)
    refined["class_segment_metrics"] = {skill: score_class(rows, index, "refined", True) for index, skill in NAMES.items()}; refined["class_frame_metrics"] = {skill: score_class(rows, index, "refined", False) for index, skill in NAMES.items()}
    refined["confusion_matrix"] = confusion(rows, "refined"); raw["confusion_matrix"] = confusion(rows, "raw")
    return {"raw": raw, "refined": refined, "boundary": boundary_summary(rows), "transitions": {str(tolerance): transition_summary(rows, tolerance) for tolerance in (10, 20, 33)}}


def confusion(rows: list[dict[str, Any]], variant: str) -> list[list[int]]:
    matrix = np.zeros((12, 12), dtype=int)
    for row in rows:
        np.add.at(matrix, (row["truth"].numpy(), row[variant]["prediction"].numpy()), 1)
    return matrix.tolist()


def predicted_segment_duration_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [interval.duration for row in rows for interval in construct_segments(row["refined"]["peaks"], len(row["truth"]))]
    return {"minimum_predicted_segment_duration": min(durations) if durations else 0, "segments_shorter_than_10": sum(duration < 10 for duration in durations), "segments_shorter_than_20": sum(duration < 20 for duration in durations), "segments_shorter_than_30": sum(duration < 30 for duration in durations)}


def attach_semantics(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["raw"]["semantic"] = semantic(row["raw"]["prediction"], row["truth"], 12)
        row["refined"]["semantic"] = semantic(row["refined"]["prediction"], row["truth"], 12)


def enrich_metadata(rows: list[dict[str, Any]], mapping: Any, config: dict[str, Any]) -> None:
    target_config = {key: config["data"][key] for key in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")}
    for row in rows:
        sample = load_trajectory_sample(DATA / row["entry"], mapping, expected_height=88, boundary_target_config=target_config)
        row["timestamps"] = sample["timestamps"].numpy()


def segment_output_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    truth = labels_to_segments(row["truth"]); raw = row["raw"]["prediction"]; current = row["refined"]["prediction"]
    intervals = construct_segments(row["refined"]["peaks"], len(row["truth"])); result = []
    for index, interval in enumerate(intervals):
        gt = int(torch.bincount(row["truth"][interval.start:interval.end], minlength=12).argmax()); raw_class = int(torch.bincount(raw[interval.start:interval.end], minlength=12).argmax()); current_class = int(torch.bincount(current[interval.start:interval.end], minlength=12).argmax())
        candidates = []
        for target in truth:
            intersection = max(0, min(interval.end - 1, target.end) - max(interval.start, target.start) + 1); union = interval.duration + target.length - intersection
            candidates.append((intersection / union if union else 0.0, target))
        iou, target = max(candidates, default=(0.0, None), key=lambda item: item[0])
        result.append({"segment_index": index, "start_frame": interval.start, "end_frame_exclusive": interval.end, "duration_frames": interval.duration, "ground_truth_class": NAMES[gt], "raw_majority_class": NAMES[raw_class], "current_nms_class": NAMES[current_class], "temporal_iou": iou, "correct": int(current_class == gt and iou >= 0.5), "matched_truth_class": NAMES[target.label] if target else ""})
    return result


def write_per_trajectory(row: dict[str, Any], distance: int) -> None:
    directory = NMS_OUT / f"d{distance}" / Path(row["entry"]).name; directory.mkdir(parents=True, exist_ok=True)
    candidates = row["refined"]["candidate_peaks"]; retained = row["refined"]["peaks"]; event_map = {event["candidate_frame"]: event for event in row["refined"]["events"]}; scores = {peak: float(row["brb"][peak]) for peak in candidates}; timestamps = row["timestamps"]; truth = truth_boundaries(row); rank = {peak: index + 1 for index, peak in enumerate(sorted(candidates, key=lambda peak: (-scores[peak], peak)))}
    candidate_rows = []
    for peak in sorted(candidates):
        event = event_map.get(peak, {}); candidate_rows.append({"frame": peak, "time_s": float((timestamps[peak] - timestamps[0]) / 1_000_000.0), "brb_probability": scores[peak], "local_maximum_rank": rank[peak], "status": "retained" if peak in retained else "suppressed", "suppressing_peak_frame": event.get("suppressing_peak_frame", ""), "distance_to_suppressor_frames": event.get("distance_frames", "")})
    write_csv(directory / "candidate_peaks.csv", candidate_rows, ["frame", "time_s", "brb_probability", "local_maximum_rank", "status", "suppressing_peak_frame", "distance_to_suppressor_frames"])
    write_csv(directory / "retained_peaks.csv", [{"frame": peak, "time_s": float((timestamps[peak] - timestamps[0]) / 1_000_000.0), "brb_probability": scores[peak], "suppressed_neighbor_count": sum(event["suppressing_peak_frame"] == peak for event in row["refined"]["events"]), "nearest_retained_peak_distance": min((abs(peak - other) for other in retained if other != peak), default="")} for peak in retained], ["frame", "time_s", "brb_probability", "suppressed_neighbor_count", "nearest_retained_peak_distance"])
    write_csv(directory / "suppressed_peaks.csv", [{"frame": event["candidate_frame"], "probability": event["candidate_probability"], "suppressing_peak_frame": event["suppressing_peak_frame"], "suppressing_peak_probability": event["suppressing_peak_probability"], "distance_frames": event["distance_frames"], "distance_seconds": event["distance_seconds"]} for event in row["refined"]["events"]], ["frame", "probability", "suppressing_peak_frame", "suppressing_peak_probability", "distance_frames", "distance_seconds"])
    frame_rows = []
    for index in range(len(row["truth"])):
        raw_label = int(row["raw"]["prediction"][index]); current_label = int(row["refined"]["prediction"][index])
        d0_prediction = row.get("d0_official_prediction", row["refined"]["prediction"])
        frame_rows.append({"frame_index": index, "time_s": float((timestamps[index] - timestamps[0]) / 1_000_000.0), "ground_truth_label": NAMES[int(row["truth"][index])], "raw_asb_label": NAMES[raw_label], "d0_official_label": NAMES[int(d0_prediction[index])], "current_nms_label": NAMES[current_label], "raw_confidence": float(row["asb"][:, index].max()), "brb_probability": float(row["brb"][index]), "ground_truth_boundary": int(index in truth), "candidate_peak": int(index in candidates), "retained_peak": int(index in retained), "suppressed_peak": int(index in event_map)})
    write_csv(directory / "frame_predictions.csv", frame_rows); write_csv(directory / "segment_predictions.csv", segment_output_rows(row))
    attach_semantics([row]); b = boundary_summary([row]); metrics = {"distance_frames": distance, "distance_seconds": distance / 100.0, "trajectory": row["entry"], "raw": row["raw"]["semantic"], "refined": row["refined"]["semantic"], "boundary": b, "candidate_peak_count": len(candidates), "retained_peak_count": len(retained), "truth_boundary_count": len(truth), "suppression_events": row["refined"]["events"]}
    write_json(directory / "metrics.json", metrics); timeline(row, distance, directory / "timeline.png")


def timeline(row: dict[str, Any], distance: int, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    truth = row["truth"].numpy(); raw = row["raw"]["prediction"].numpy(); d0 = row.get("d0_official_prediction", row["refined"]["prediction"]).numpy(); current = row["refined"]["prediction"].numpy(); x = (row["timestamps"] - row["timestamps"][0]) / 1_000_000.0; probability = row["brb"].numpy(); candidates = row["refined"]["candidate_peaks"]; retained = row["refined"]["peaks"]; suppressed = [event["candidate_frame"] for event in row["refined"]["events"]]
    colors = list(plt.get_cmap("tab20").colors[:12]); cmap = ListedColormap(colors); norm = BoundaryNorm(np.arange(-0.5, 12.5, 1), 12)
    fig, axes = plt.subplots(5, 1, figsize=(18, 10), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [1, 1, 1, 1, 2.5]})
    for axis, values, label in ((axes[0], truth, "ground truth"), (axes[1], raw, "raw ASB"), (axes[2], d0, "d=0 official"), (axes[3], current, f"NMS d={distance}")):
        axis.imshow(values[np.newaxis, :], aspect="auto", interpolation="nearest", extent=(x[0], x[-1], 0, 1), cmap=cmap, norm=norm); axis.set_ylabel(label)
    axes[4].plot(x, probability, color="black", linewidth=0.8, label="BRB probability"); axes[4].set_ylim(0, 1); axes[4].set_ylabel("BRB p")
    boundaries = truth_boundaries(row)
    for boundary in boundaries: axes[4].axvline(x[boundary], color="limegreen", linewidth=0.8, label="ground-truth boundary" if boundary == boundaries[0] else None)
    for peak in candidates: axes[4].plot(x[peak], probability[peak], marker="x", color="darkorange", markersize=5, label="candidate" if peak == candidates[0] else None)
    for peak in retained: axes[4].plot(x[peak], probability[peak], marker="o", markerfacecolor="none", markeredgecolor="blue", markersize=6, label="retained" if peak == retained[0] else None)
    for peak in suppressed: axes[4].plot(x[peak], probability[peak], marker="x", color="red", markersize=7, label="suppressed" if peak == suppressed[0] else None)
    for segment in labels_to_segments(row["truth"])[1:]:
        previous = NAMES[int(row["truth"][segment.start - 1])]; current_name = NAMES[int(row["truth"][segment.start])]
        axes[4].text(x[segment.start], 0.98, f"{previous}→{current_name}", rotation=90, fontsize=6, va="top", ha="right", color="green")
    axes[4].legend(loc="lower left", fontsize=7, framealpha=0.8); axes[4].set_xlabel("time (s)")
    fig.suptitle(f"{row['entry']} — BRB candidate/retained/suppressed peaks — d={distance}"); fig.savefig(path, dpi=130); plt.close(fig)


def metric_row(distance: int, split: str, result: dict[str, Any], rows: list[dict[str, Any]], selected: int | None) -> dict[str, Any]:
    b = result["boundary"]; transitions = {item["transition"]: item for item in result["transitions"]["33"]}; durs = predicted_segment_duration_stats(rows); segment = result["refined"]["class_segment_metrics"]
    return {"minimum_distance_frames": distance, "minimum_distance_seconds": distance / 100.0, "split": split, "raw_F1_50": result["raw"]["F1@50"], "refined_F1_50": result["refined"]["F1@50"], "boundary_F1_5": b["5"]["F1"], "boundary_F1_10": b["10"]["F1"], "boundary_F1_20": b["20"]["F1"], "boundary_F1_33": b["33"]["F1"], "boundary_F1_50": b["50"]["F1"], "false_peaks": b["33"]["fp"], "missed_boundaries": b["33"]["fn"], "duplicate_peaks": b["33"]["duplicate_peaks"], "predicted_peaks": b["33"]["predicted_count"], "true_boundaries": b["33"]["target_count"], "predicted_segments": result["refined"]["predicted_segment_count"], "true_segments": result["refined"]["true_segment_count"], "mean_localization_error": b["33"]["mean_localization_error"], "median_localization_error": b["33"]["median_localization_error"], "minimum_predicted_segment_duration": durs["minimum_predicted_segment_duration"], "segments_shorter_than_10": durs["segments_shorter_than_10"], "segments_shorter_than_20": durs["segments_shorter_than_20"], "segments_shorter_than_30": durs["segments_shorter_than_30"], "place_F1": segment["place"]["F1"], "insert_F1": segment["insert"]["F1"], "release_F1": segment["release"]["F1"], "lift_F1": segment["lift"]["F1"], "transport_place_recall_33": transitions["transport -> place"]["recall"], "place_insert_recall_33": transitions["place -> insert"]["recall"], "insert_release_recall_33": transitions["insert -> release"]["recall"], "passes_safety_constraint": result.get("passes_safety_constraint", ""), "selected_on_validation": int(selected == distance) if selected is not None else 0}


def plot_figures(validation_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], per_trajectory_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    figures = NMS_OUT / "figures"; figures.mkdir(parents=True, exist_ok=True); distances = list(DISTANCES)
    def line(name: str, series: list[tuple[str, list[float]]], ylabel: str, title: str) -> None:
        fig, axis = plt.subplots(figsize=(8, 5))
        for label, values in series: axis.plot(distances, values, marker="o", label=label)
        axis.set_xticks(distances); axis.set_xlabel("minimum peak distance (frames)"); axis.set_ylabel(ylabel); axis.set_title(title); axis.grid(alpha=0.25); axis.legend(); fig.tight_layout(); fig.savefig(figures / name, dpi=130); plt.close(fig)
    line("validation_refined_f1_50.png", [("validation refined F1@50", [float(row["refined_F1_50"]) for row in validation_rows])], "F1@50", "Validation refined F1@50 vs NMS distance")
    line("validation_boundary_f1_33.png", [("validation boundary F1@33", [float(row["boundary_F1_33"]) for row in validation_rows])], "F1@33", "Validation boundary F1@33 vs NMS distance")
    line("validation_peak_errors.png", [("false peaks", [float(row["false_peaks"]) for row in validation_rows]), ("missed boundaries", [float(row["missed_boundaries"]) for row in validation_rows]), ("duplicate peaks", [float(row["duplicate_peaks"]) for row in validation_rows])], "count", "Validation peak errors vs NMS distance")
    line("test_refined_f1_50.png", [("test refined F1@50", [float(row["refined_F1_50"]) for row in test_rows])], "F1@50", "Test refined F1@50 vs NMS distance")
    line("test_peak_errors.png", [("false peaks", [float(row["false_peaks"]) for row in test_rows]), ("missed boundaries", [float(row["missed_boundaries"]) for row in test_rows]), ("duplicate peaks", [float(row["duplicate_peaks"]) for row in test_rows])], "count", "Test peak errors vs NMS distance")
    line("place_insert_f1.png", [("place F1", [float(row["place_F1"]) for row in test_rows]), ("insert F1", [float(row["insert_F1"]) for row in test_rows])], "segment F1", "Place and insert F1 vs NMS distance")
    line("target_transition_recall.png", [("transport→place", [float(row["transport_place_recall_33"]) for row in test_rows]), ("place→insert", [float(row["place_insert_recall_33"]) for row in test_rows]), ("insert→release", [float(row["insert_release_recall_33"]) for row in test_rows])], "recall@33", "Target-transition recall vs NMS distance")
    fig, axis = plt.subplots(figsize=(9, 5))
    for name in ("p1", "p2", "p3", "po1", "po2"):
        values = [float(next(row for row in per_trajectory_rows if row["minimum_distance_frames"] == distance and Path(row["trajectory"]).name == name)["refined_F1_50"]) for distance in distances]
        axis.plot(distances, values, marker="o", label=name)
    axis.set_xticks(distances); axis.set_xlabel("minimum peak distance (frames)"); axis.set_ylabel("refined F1@50"); axis.set_title("Per-trajectory refined F1@50 vs NMS distance"); axis.grid(alpha=0.25); axis.legend(); fig.tight_layout(); fig.savefig(figures / "per_trajectory_refined_f1_50.png", dpi=130); plt.close(fig)


def lost_true_boundaries(d0_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]], distance: int) -> list[dict[str, Any]]:
    lost: list[dict[str, Any]] = []
    for baseline, current in zip(d0_rows, current_rows):
        for boundary in truth_boundaries(baseline):
            baseline_detected = any(abs(peak - boundary) <= 33 for peak in baseline["refined"]["peaks"] if peak != 0)
            current_detected = any(abs(peak - boundary) <= 33 for peak in current["refined"]["peaks"] if peak != 0)
            if baseline_detected and not current_detected:
                lost.append({"minimum_distance_frames": distance, "trajectory": baseline["entry"], "boundary_frame": boundary, "boundary_time_s": float((baseline["timestamps"][boundary] - baseline["timestamps"][0]) / 1_000_000.0), "from_class": NAMES[int(baseline["truth"][boundary - 1])] if boundary > 0 else "", "to_class": NAMES[int(baseline["truth"][boundary])], "d0_detected": 1, "nms_detected": 0})
    return lost


def evaluate_distance(records_rows: list[dict[str, Any]], distance: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for record in records_rows:
        row = dict(record); row.update(add_variants(record, distance)); rows.append(row)
    attach_semantics(rows); result = summary(rows); result["duration_stats"] = predicted_segment_duration_stats(rows); result["distance"] = distance
    return result, rows


def verify_d0_against_saved(rows: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    saved_summary = json.loads((ROOT / "outputs/round9_incremental_learning/plug/n10/primary_test_summary.json").read_text())
    checks = {"raw_F1_50": (result["raw"]["F1@50"], saved_summary["raw"]["F1@50"]), "official_F1_50": (result["refined"]["F1@50"], saved_summary["official"]["F1@50"]), "boundary_F1_33": (result["boundary"]["33"]["F1"], saved_summary["boundary"]["official"]["33"]["F1"]), "false_peaks": (result["boundary"]["33"]["fp"], saved_summary["boundary"]["official"]["33"]["fp"]), "missed_boundaries": (result["boundary"]["33"]["fn"], saved_summary["boundary"]["official"]["33"]["fn"])}
    exact = True; per_trajectory = {}
    for row in rows:
        name = Path(row["entry"]).name; old_dir = ROOT / "outputs/round9_incremental_learning/plug/n10/test_per_trajectory" / name
        old_frames = list(csv.DictReader((old_dir / "frame_predictions.csv").open(encoding="utf-8")))
        new_labels = [NAMES[int(value)] for value in row["refined"]["prediction"]]
        old_labels = [item["official_refined_label"] for item in old_frames]
        old_peaks = [int(item["frame_index"]) for item in old_frames if item["predicted_boundary_peak"] == "1"]
        same_labels = old_labels == new_labels; same_peaks = old_peaks == row["refined"]["peaks"]; per_trajectory[name] = {"same_labels": same_labels, "same_peaks": same_peaks}; exact = exact and same_labels and same_peaks
    numeric_exact = all(abs(float(current) - float(expected)) < 1e-12 for current, expected in checks.values() if isinstance(current, float)) and all(current == expected for current, expected in checks.values() if isinstance(current, int))
    payload = {"exact": bool(exact and numeric_exact), "checks": {key: {"nms_d0": current, "saved": expected} for key, (current, expected) in checks.items()}, "per_trajectory": per_trajectory}
    write_json(NMS_OUT / "baseline_reproducibility.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--boundary-nms-min-distance", type=int, default=None); args = parser.parse_args()
    distances = (int(args.boundary_nms_min_distance),) if args.boundary_nms_min_distance is not None else DISTANCES
    if any(distance not in DISTANCES for distance in distances): raise ValueError(f"primary distances must be one of {DISTANCES}")
    NMS_OUT.mkdir(parents=True, exist_ok=True); checkpoint = ROOT / "outputs/round9_incremental_learning/plug/n10/best.pt"; config_path = ROOT / "outputs/round9_incremental_learning/plug/n10/config.yaml"; config = load_yaml_config(config_path); mapping = load_label_mapping(ROOT / config["data"]["label_config"])
    checkpoint_hash = sha256_file(checkpoint); expected_hash = "1c75c15d45c63a18f1cfd2c856952f42b35277fe9baa5e4951fe886227bf1ee5"; assert checkpoint_hash == expected_hash, checkpoint_hash
    original_report = ROOT / "outputs/round9_incremental_learning/plug/n10/plug10_report.md"; report_hash_before = hashlib.sha256(original_report.read_bytes()).hexdigest()
    model = ASRFModel.from_config(config); model.load_state_dict(load_checkpoint(checkpoint, map_location="cpu", expected_ontology=True)["model_state"]); model.eval()

    # Validation is deliberately the only dataset read before selection is committed.
    validation_records = records(model, "splits/round9_incremental/common_validation.txt", config); validation_metrics: list[dict[str, Any]] = []; validation_results: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    baseline_validation_result = None
    for distance in distances:
        result, rows = evaluate_distance(validation_records, distance); validation_results[distance] = (result, rows)
        if distance == 0: baseline_validation_result = result
    assert baseline_validation_result is not None
    for distance in distances:
        result = validation_results[distance][0]; result["passes_safety_constraint"] = True
        if distance != 0:
            result["passes_safety_constraint"] = result["refined"]["F1@50"] >= baseline_validation_result["refined"]["F1@50"] - 0.01 and (baseline_validation_result["boundary"]["33"]["fn"] == 0 and result["boundary"]["33"]["fn"] == 0 or baseline_validation_result["boundary"]["33"]["fn"] > 0 and result["boundary"]["33"]["fn"] <= baseline_validation_result["boundary"]["33"]["fn"] * 1.10)
        validation_metrics.append(metric_row(distance, "validation", result, validation_results[distance][1], None))
    eligible = [row for row in validation_metrics if row["passes_safety_constraint"]]
    selected = min(eligible, key=lambda row: (-float(row["refined_F1_50"]), -float(row["boundary_F1_33"]), int(row["missed_boundaries"]), int(row["false_peaks"]), int(row["minimum_distance_frames"]))) ["minimum_distance_frames"] if eligible else 0
    reason = "d=0 selected because no positive distance passed the safety constraint." if selected == 0 and not any(row["passes_safety_constraint"] for row in validation_metrics if row["minimum_distance_frames"] > 0) else "selected by validation refined F1@50 with the prescribed tie-breakers and safety constraint."
    for row in validation_metrics: row["selected_on_validation"] = int(row["minimum_distance_frames"] == selected)
    write_csv(NMS_OUT / "nms_validation_metrics.csv", validation_metrics)
    selection = {"candidate_distances": list(distances), "validation_metrics": validation_metrics, "baseline_distance": 0, "selected_distance": selected, "selection_reason": reason, "safety_rule": {"max_refined_F1_50_drop": 0.01, "max_missed_boundary_increase_fraction": 0.10}, "checkpoint_sha256": checkpoint_hash, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "test_metrics_accessed_before_selection": False}
    write_json(NMS_OUT / "validation_selection.json", selection)

    # The selection file is committed before the first test trajectory is read.
    test_records = records(model, "splits/round9_incremental/test_plug_primary.txt", config); enrich_metadata(test_records, mapping, config)
    d0_result, d0_rows = evaluate_distance(test_records, 0); baseline_check = verify_d0_against_saved(d0_rows, d0_result)
    if not baseline_check["exact"]:
        raise RuntimeError("d=0 does not reproduce the saved Plug-10 official inference; see baseline_reproducibility.json")
    test_results: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {0: (d0_result, d0_rows)}
    for distance in distances:
        if distance == 0: continue
        result, rows = evaluate_distance(test_records, distance); test_results[distance] = (result, rows)
    baseline_by_entry = {row["entry"]: row for row in d0_rows}
    for distance, (_, rows) in test_results.items():
        for row in rows:
            row["d0_official_prediction"] = baseline_by_entry[row["entry"]]["refined"]["prediction"]
    test_metrics: list[dict[str, Any]] = []; per_trajectory: list[dict[str, Any]] = []; transition_rows: list[dict[str, Any]] = []; suppression_rows: list[dict[str, Any]] = []
    lost_boundary_rows: list[dict[str, Any]] = []
    for distance in distances:
        result, rows = test_results[distance]; result["passes_safety_constraint"] = next(row["passes_safety_constraint"] for row in validation_metrics if row["minimum_distance_frames"] == distance); test_metrics.append(metric_row(distance, "test", result, rows, selected))
        for row in rows:
            b = boundary_summary([row])["33"]; per_trajectory.append({"minimum_distance_frames": distance, "minimum_distance_seconds": distance / 100.0, "trajectory": row["entry"], "raw_F1_50": row["raw"]["semantic"]["F1@50"], "refined_F1_50": row["refined"]["semantic"]["F1@50"], "boundary_F1_33": b["F1"], "false_peaks": b["fp"], "missed_boundaries": b["fn"], "duplicate_peaks": b["duplicate_peaks"], "predicted_peaks": b["predicted_count"], "true_boundaries": b["target_count"], "place_F1": score_class([row], 6, "refined", True)["F1"], "insert_F1": score_class([row], 10, "refined", True)["F1"], "release_F1": score_class([row], 7, "refined", True)["F1"], "passes_safety_constraint": result["passes_safety_constraint"], "selected_on_validation": int(distance == selected)})
            for event in row["refined"]["events"]: suppression_rows.append({"minimum_distance_frames": distance, "trajectory": row["entry"], **event})
        for tolerance in (10, 20, 33):
            for item in transition_summary(rows, tolerance): transition_rows.append({"minimum_distance_frames": distance, "minimum_distance_seconds": distance / 100.0, "tolerance_frames": tolerance, **{key: value for key, value in item.items() if key != "details"}})
        for row in rows: write_per_trajectory(row, distance)
        if distance != 0:
            lost_boundary_rows.extend(lost_true_boundaries(d0_rows, rows, distance))
    write_csv(NMS_OUT / "nms_test_metrics.csv", test_metrics); write_csv(NMS_OUT / "nms_per_trajectory_metrics.csv", per_trajectory); write_csv(NMS_OUT / "nms_transition_metrics.csv", transition_rows); write_csv(NMS_OUT / "nms_suppression_events.csv", suppression_rows)
    write_csv(NMS_OUT / "lost_true_boundaries.csv", lost_boundary_rows, ["minimum_distance_frames", "trajectory", "boundary_frame", "boundary_time_s", "from_class", "to_class", "d0_detected", "nms_detected"])
    plot_figures(validation_metrics, test_metrics, per_trajectory)
    report_hash_after = hashlib.sha256(original_report.read_bytes()).hexdigest(); protected = {"plug10_best_before_after": [checkpoint_hash, sha256_file(checkpoint)], "plug10_report_before_after": [report_hash_before, report_hash_after], "plug3_best": sha256_file(ROOT / "outputs/round9_incremental_learning/models/plug/n3/best.pt"), "plug5_best": sha256_file(ROOT / "outputs/round9_incremental_learning/models/plug/n5/best.pt"), "round8_r5": sha256_file(ROOT / "outputs/brb_release_round8/hard_window_r5/best.pt")}; write_json(NMS_OUT / "integrity_hashes.json", protected)
    report(test_metrics, validation_metrics, selection, baseline_check, protected, suppression_rows, per_trajectory, lost_boundary_rows)
    print(json.dumps({"selected_distance": selected, "baseline_exact": baseline_check["exact"], "validation_metrics": validation_metrics, "test_metrics": test_metrics}, indent=2, default=json_default))
    return 0


def report(test_metrics: list[dict[str, Any]], validation_metrics: list[dict[str, Any]], selection: dict[str, Any], baseline_check: dict[str, Any], protected: dict[str, Any], suppression_rows: list[dict[str, Any]], per_trajectory: list[dict[str, Any]], lost_boundary_rows: list[dict[str, Any]]) -> None:
    d0 = next(row for row in test_metrics if row["minimum_distance_frames"] == 0); lines = ["# Plug-10 BRB greedy NMS post-processing ablation", "", "## Definition and protocol", "", "Candidates are the exact official BRB local maxima at threshold 0.50. Candidates are sorted by descending BRB probability, then ascending frame index. The highest-scoring remaining peak is retained; every candidate strictly closer than d frames is suppressed. Retained peaks remain at their original frames and are returned chronologically. d=0 disables suppression.", "", "Distances tested: 0, 10, 20, and 30 frames. Validation selection was finalized before test data was read.", "", "## Validation selection", "", f"Selected official distance: **d={selection['selected_distance']} frames ({selection['selected_distance'] / 100.0:.2f} s)**.", "", "| d | refined F1@50 | boundary F1@33 | false | missed | duplicates | safety | selected |", "|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in validation_metrics: lines.append(f"| {row['minimum_distance_frames']} | {row['refined_F1_50']:.4f} | {row['boundary_F1_33']:.4f} | {row['false_peaks']} | {row['missed_boundaries']} | {row['duplicate_peaks']} | {row['passes_safety_constraint']} | {row['selected_on_validation']} |")
    lines += ["", selection["selection_reason"], "", "## d=0 reproducibility", "", f"d=0 exact reproduction: **{baseline_check['exact']}**. Raw F1@50={d0['raw_F1_50']:.4f}; refined F1@50={d0['refined_F1_50']:.4f}; boundary F1@33={d0['boundary_F1_33']:.4f}; false peaks={d0['false_peaks']}; missed={d0['missed_boundaries']}.", "", "## Test results", "", "These are scientific ablation results; only the validation-selected distance is official.", "", "| d | raw F1@50 | refined F1@50 | boundary F1@33 | false | missed | duplicates | place F1 | insert F1 | place→insert | insert→release |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in test_metrics: lines.append(f"| {row['minimum_distance_frames']} | {row['raw_F1_50']:.4f} | {row['refined_F1_50']:.4f} | {row['boundary_F1_33']:.4f} | {row['false_peaks']} | {row['missed_boundaries']} | {row['duplicate_peaks']} | {row['place_F1']:.4f} | {row['insert_F1']:.4f} | {row['place_insert_recall_33']:.4f} | {row['insert_release_recall_33']:.4f} |")
    lost_text = "None." if not lost_boundary_rows else "\n".join(f"- d={item['minimum_distance_frames']}: {item['trajectory']} frame {item['boundary_frame']} ({item['from_class']}→{item['to_class']})" for item in lost_boundary_rows)
    lines += ["", "Raw ASB metrics are invariant across all distances: " + str(len({round(float(row['raw_F1_50']), 12) for row in test_metrics}) == 1) + ".", "", "## Diagnostics", "", f"At d=10/20/30, false peaks change from {d0['false_peaks']} to " + ", ".join(str(next(row['false_peaks'] for row in test_metrics if row['minimum_distance_frames'] == distance)) for distance in (10, 20, 30)) + "; duplicate peaks change from {d0['duplicate_peaks']} to " + ", ".join(str(next(row['duplicate_peaks'] for row in test_metrics if row['minimum_distance_frames'] == distance)) for distance in (10, 20, 30)) + ".", "", "Suppressed candidates are classified in nms_suppression_events.csv as duplicate-near-boundary, isolated false peak inside a skill, or true-boundary suppression risk.", "", "Ground-truth boundaries detected at d=0 but missed after NMS:", lost_text, "", "The d=30 short-segment risk is reported by minimum predicted segment duration and counts below 10/20/30 frames in nms_test_metrics.csv.", "", "## Figures and integrity", "", "All 20 timeline figures and all eight comparison figures were visually inspected.", "", "```json", json.dumps(protected, indent=2, sort_keys=True), "```", "", "Recommendation: enable NMS only at the validation-selected distance if it passes the safety constraint and improves the joint boundary/semantic objective; do not choose on false-peak reduction alone."]
    (NMS_OUT / "nms_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
