#!/usr/bin/env python3
"""Make a direct, readable visualization of greedy NMS effects.

This script is deliberately inference-free.  It consumes the already exported
Round 9 NMS artifacts and the read-only trajectory heatmap/annotations, then
creates a full comparison figure, local suppression zooms, and machine-readable
summaries.  Predicted labels always come from the prediction segment tables;
GT overlap is used only for diagnostic annotations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asrf.data.dataset import load_trajectory_sample
from asrf.data.labels import load_label_mapping
from asrf.visualization.temporal import DEFAULT_LABEL_COLORS, _normalized_heatmap


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
SOURCE_ROOT = ROOT / "outputs/round9_incremental_learning/nms_cross_family/test"
OUT_ROOT = ROOT / "outputs/round9_incremental_learning/nms_cross_family/figres/nms_visible_difference"
ONTOLOGY = {
    "reach", "grasp", "lift", "transport", "pour", "pour_recover", "place",
    "release", "wipe", "retreat", "insert",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_int(value: Any) -> int:
    return int(float(value))


def _as_float(value: Any) -> float:
    return float(value)


def _time_edges(timestamps_us: np.ndarray) -> np.ndarray:
    time_s = (timestamps_us.astype(float) - float(timestamps_us[0])) / 1_000_000.0
    if len(time_s) == 1:
        step = 0.01
    else:
        step = float(np.median(np.diff(time_s)))
    return np.concatenate([time_s, [time_s[-1] + step]])


def _segments_from_labels(labels: np.ndarray, names: dict[int, str]) -> list[dict[str, Any]]:
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("ground-truth labels must be a non-empty vector")
    starts = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    ends = starts[1:] + [len(labels)]
    return [
        {"start_frame": start, "end_frame": end, "predicted_label": names[int(labels[start])], "gt_label": names[int(labels[start])]}
        for start, end in zip(starts, ends)
    ]


def _prediction_segments(rows: list[dict[str, str]], label_field: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous_end = 0
    for row in rows:
        start = _as_int(row["start_frame"])
        end = _as_int(row["end_frame_exclusive"])
        label = str(row[label_field]).strip()
        if label not in ONTOLOGY:
            raise ValueError(f"prediction label outside ontology: {label!r}")
        if end <= start or start != previous_end:
            raise ValueError(f"invalid/non-covering prediction segment [{start}, {end})")
        result.append({
            "start_frame": start,
            "end_frame": end,
            "predicted_label": label,
        })
        previous_end = end
    if not result:
        raise ValueError("prediction segment table is empty")
    return result


def _segment_schema(rows: list[dict[str, Any]], width: int) -> None:
    if rows[0]["start_frame"] != 0 or rows[-1]["end_frame"] != width:
        raise ValueError("prediction segments do not cover the trajectory")
    for left, right in zip(rows, rows[1:]):
        if left["end_frame"] != right["start_frame"]:
            raise ValueError("prediction segments have a gap or overlap")


def _event_frames(rows: list[dict[str, str]], status: str | None = None) -> list[int]:
    return sorted(_as_int(row["frame"]) for row in rows if status is None or row["status"] == status)


def _gt_boundaries(gt_segments: list[dict[str, Any]]) -> list[int]:
    return [int(row["start_frame"]) for row in gt_segments[1:]]


def _nearest_distance(frame: int, boundaries: Iterable[int]) -> int | None:
    values = list(boundaries)
    return min((abs(int(frame) - int(boundary)) for boundary in values), default=None)


def _group_suppression_events(events: list[dict[str, Any]], gap_frames: int = 100) -> list[list[dict[str, Any]]]:
    """Group nearby suppression events into local visual windows."""
    if not events:
        return []
    ordered = sorted(events, key=lambda row: int(row["frame"]))
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    for event in ordered[1:]:
        if int(event["frame"]) - int(groups[-1][-1]["frame"]) <= gap_frames:
            groups[-1].append(event)
        else:
            groups.append([event])
    return groups


def _overlapping(rows: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["end_frame"]) > start and int(row["start_frame"]) < end]


def _collapsed_labels(rows: list[dict[str, Any]], start: int, end: int) -> list[str]:
    labels: list[str] = []
    selected = _overlapping(rows, start, end)
    # Include the segment immediately to the left when the window starts on a
    # boundary.  This makes a local transition such as reach -> grasp visible
    # in the region summary rather than dropping the left-hand label.
    if selected and int(selected[0]["start_frame"]) == start:
        left = next((row for row in rows if int(row["end_frame"]) == start), None)
        if left is not None:
            selected = [left] + selected
    for row in selected:
        label = str(row["predicted_label"])
        if not labels or labels[-1] != label:
            labels.append(label)
    return labels


def _removed_boundary_semantics(frame: int, before: list[dict[str, Any]]) -> tuple[bool, bool]:
    for left, right in zip(before, before[1:]):
        if int(left["end_frame"]) == frame and int(right["start_frame"]) == frame:
            return left["predicted_label"] != right["predicted_label"], left["predicted_label"] == right["predicted_label"]
    return False, False


def _draw_segment_row(axis: Any, segments: list[dict[str, Any]], edges: np.ndarray, colors: dict[str, str], *, label_key: str = "predicted_label", gt: bool = False, start: int = 0, end: int | None = None) -> None:
    end = len(edges) - 1 if end is None else end
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xlim(float(edges[start]), float(edges[end]))
    for index, row in enumerate(segments):
        left = max(start, int(row["start_frame"]))
        right = min(end, int(row["end_frame"]))
        if right <= left:
            continue
        label = str(row[label_key])
        axis.add_patch(Rectangle((edges[left], 0.08), edges[right] - edges[left], 0.84, facecolor=colors.get(label, "#cccccc"), edgecolor="white", linewidth=0.45, alpha=0.88 if gt else 0.72))
        duration = right - left
        width_s = float(edges[right] - edges[left])
        if duration >= 35 or width_s >= 0.55:
            axis.text((edges[left] + edges[right]) / 2, 0.5, label, ha="center", va="center", fontsize=8, fontweight="bold", clip_on=True)
        elif duration >= 10:
            axis.text(edges[left] + 0.01 * (edges[-1] - edges[0]), 0.5, label, ha="left", va="center", fontsize=6, clip_on=True)


def _draw_heatmap(axis: Any, heatmap: np.ndarray, edges: np.ndarray) -> None:
    axis.imshow(_normalized_heatmap(heatmap), aspect="auto", origin="upper", extent=(edges[0], edges[-1], heatmap.shape[1], 0), interpolation="nearest")
    axis.set_ylabel("heatmap\nchannels", rotation=0, ha="right", va="center", fontsize=9)
    axis.set_yticks([])
    axis.set_xlim(edges[0], edges[-1])


def _event_legend(axis: Any) -> None:
    axis.scatter([], [], marker="*", s=100, color="#111111", label="GT boundary")
    axis.scatter([], [], marker="o", s=55, color="#1b9e77", label="retained in both")
    axis.scatter([], [], marker="X", s=75, color="#d62728", label="suppressed by NMS")
    axis.scatter([], [], marker="^", s=65, color="#7b2cbf", label="NMS-only retained")
    axis.scatter([], [], marker=".", s=28, color="#777777", label="candidate peak")
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=3, fontsize=8, frameon=True)


def _plot_full(
    *,
    heatmap: np.ndarray,
    edges: np.ndarray,
    gt: list[dict[str, Any]],
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    retained_before: list[int],
    retained_after: list[int],
    suppressed: list[dict[str, Any]],
    brb: np.ndarray,
    output: Path,
    title: str,
    gt_boundaries: list[int],
    groups: list[list[dict[str, Any]]],
) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(24, 16), sharex=True, gridspec_kw={"height_ratios": [2.8, 1, 1, 1, 2.3, 1.7, 1.2]})
    colors = DEFAULT_LABEL_COLORS
    _draw_heatmap(axes[0], heatmap, edges)
    axes[0].set_title(title, loc="left", fontsize=15, fontweight="bold")
    _draw_segment_row(axes[1], gt, edges, colors, gt=True)
    axes[1].set_ylabel("GT", rotation=0, ha="right", va="center")
    _draw_segment_row(axes[2], before, edges, colors)
    axes[2].set_ylabel("r5\nNMS=0", rotation=0, ha="right", va="center")
    _draw_segment_row(axes[3], after, edges, colors)
    axes[3].set_ylabel("r5\nNMS=10", rotation=0, ha="right", va="center")

    # Difference row: large symbols and event annotations make suppression visible.
    ax = axes[4]
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("difference", rotation=0, ha="right", va="center")
    for group_index, group in enumerate(groups, 1):
        left = min(int(e["frame"]) for e in group + [{"frame": e["suppressing_retained_peak_frame"]} for e in group])
        right = max(int(e["frame"]) for e in group + [{"frame": e["suppressing_retained_peak_frame"]} for e in group])
        ax.axvspan(edges[max(0, left - 5)], edges[min(len(edges) - 1, right + 5)], color="#d62728", alpha=0.10, zorder=0)
        ax.text((edges[left] + edges[right]) / 2, 0.94, f"region {group_index}", ha="center", va="top", fontsize=9, color="#9c1c1c", fontweight="bold")
    candidate_frames = [_as_int(row["frame"]) for row in candidates]
    both = sorted(set(retained_before) & set(retained_after))
    only_before = sorted(set(retained_before) - set(retained_after))
    only_after = sorted(set(retained_after) - set(retained_before))
    ax.scatter(edges[candidate_frames], np.full(len(candidate_frames), 0.46), marker=".", s=35, color="#777777", alpha=0.75, zorder=2)
    ax.scatter(edges[gt_boundaries], np.full(len(gt_boundaries), 0.73), marker="*", s=110, color="#111111", zorder=4)
    ax.scatter(edges[both], np.full(len(both), 0.30), marker="o", s=48, facecolors="white", edgecolors="#1b9e77", linewidths=1.8, zorder=4)
    if only_after:
        ax.scatter(edges[only_after], np.full(len(only_after), 0.30), marker="^", s=65, color="#7b2cbf", zorder=5)
    if only_before:
        ax.scatter(edges[only_before], np.full(len(only_before), 0.18), marker="X", s=90, color="#d62728", edgecolors="white", linewidths=0.7, zorder=6)
    for index, event in enumerate(sorted(suppressed, key=lambda row: int(row["frame"]))):
        frame = int(event["frame"])
        nearest_gt = _nearest_distance(frame, gt_boundaries)
        text = f"{frame}\n{float(event['time_s']):.2f}s\np={float(event['probability']):.3f}\nΔR={int(event['distance_frames'])}f\nΔGT={nearest_gt if nearest_gt is not None else '—'}f"
        y = 0.06 if index % 2 == 0 else 0.88
        ax.annotate(text, xy=(edges[frame], 0.18), xytext=(edges[frame], y), ha="center", va="bottom" if y < 0.2 else "top", fontsize=7, color="#8b0000", bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fff3f3", "edgecolor": "#d62728", "alpha": 0.96}, arrowprops={"arrowstyle": "-", "color": "#d62728", "lw": 0.7})
    _event_legend(ax)

    axes[5].plot(edges[:-1], brb, color="#202020", linewidth=0.9, label="BRB probability")
    axes[5].axhline(0.5, color="#d62728", linestyle="--", linewidth=0.9, label="threshold 0.50")
    axes[5].scatter(edges[candidate_frames], brb[candidate_frames], s=24, color="#777777", zorder=3)
    if only_before:
        ax5 = axes[5]
        ax5.scatter(edges[only_before], brb[only_before], marker="X", s=80, color="#d62728", edgecolors="white", zorder=5, label="suppressed")
    axes[5].set_ylim(0, 1.05)
    axes[5].set_ylabel("BRB", rotation=0, ha="right", va="center")
    axes[5].legend(loc="upper left", ncol=2, fontsize=8)

    axes[6].set_ylim(0.5, 4.5)
    axes[6].set_yticks([1, 2, 3, 4], ["GT", "both", "suppressed", "NMS-only"])
    axes[6].eventplot([[edges[x] for x in gt_boundaries]], lineoffsets=1, colors="#111111", linewidths=2.0)
    axes[6].eventplot([[edges[x] for x in both]], lineoffsets=2, colors="#1b9e77", linewidths=2.0)
    axes[6].eventplot([[edges[x] for x in only_before]], lineoffsets=3, colors="#d62728", linewidths=3.0)
    if only_after:
        axes[6].eventplot([[edges[x] for x in only_after]], lineoffsets=4, colors="#7b2cbf", linewidths=2.0)
    axes[6].set_ylabel("events", rotation=0, ha="right", va="center")
    axes[6].set_xlabel("time (s)")

    labels_before = _collapsed_labels(before, 0, len(edges) - 1)
    labels_after = _collapsed_labels(after, 0, len(edges) - 1)
    semantic_changed = labels_before != labels_after
    same_label_merges = sum(1 for event in suppressed if _removed_boundary_semantics(int(event["frame"]), before)[1])
    semantic_boundary_changes = sum(1 for event in suppressed if _removed_boundary_semantics(int(event["frame"]), before)[0])
    false_removed = sum(1 for event in suppressed if int(event["nearest_truth_distance"]) > 33)
    summary = (
        f"candidate peaks: {len(candidates)}\n"
        f"retained without NMS: {len(retained_before)}\n"
        f"retained with NMS: {len(retained_after)}\n"
        f"suppressed by NMS: {len(suppressed)}\n"
        f"semantic segment count without NMS: {len(before)}\n"
        f"semantic segment count with NMS: {len(after)}\n"
        f"semantic boundaries changed: {semantic_boundary_changes}\n"
        f"same-label merges: {same_label_merges}\n"
        f"isolated false peaks removed: {false_removed}\n"
        f"collapsed label sequence changed: {semantic_changed}"
    )
    fig.text(0.995, 0.985, summary, ha="right", va="top", fontsize=9, family="monospace", bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#444444", "alpha": 0.95})
    event_lines = ["suppressed-boundary annotations:"] + [
        f"{int(event['frame']):4d}  t={float(event['time_s']):5.2f}s  p={float(event['probability']):.3f}  "
        f"ΔR={int(event['distance_frames']):2d}f  ΔGT={int(event['nearest_truth_distance']):3d}f"
        for event in sorted(suppressed, key=lambda row: int(row["frame"]))
    ]
    fig.text(0.995, 0.57, "\n".join(event_lines), ha="right", va="top", fontsize=7.0, family="monospace", bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff8f8", "edgecolor": "#d62728", "alpha": 0.96})
    axes[-1].set_xlim(edges[0], edges[-1])
    fig.subplots_adjust(left=0.08, right=0.78, top=0.96, bottom=0.08, hspace=0.42)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_zoom(*, heatmap: np.ndarray, edges: np.ndarray, gt: list[dict[str, Any]], before: list[dict[str, Any]], after: list[dict[str, Any]], candidates: list[dict[str, Any]], retained_before: list[int], retained_after: list[int], suppressed: list[dict[str, Any]], brb: np.ndarray, gt_boundaries: list[int], start: int, end: int, output: Path, title: str) -> None:
    fig, axes = plt.subplots(8, 1, figsize=(18, 12), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.5, 0.8, 0.8, 0.8, 0.8]})
    colors = DEFAULT_LABEL_COLORS
    for axis, segments, label, gt_flag in [(axes[0], gt, "GT", True), (axes[1], before, "r5 NMS=0", False), (axes[2], after, "r5 NMS=10", False)]:
        _draw_segment_row(axis, segments, edges, colors, gt=gt_flag, start=start, end=end)
        axis.set_ylabel(label, rotation=0, ha="right", va="center")
    axes[3].plot(edges[start:end], brb[start:end], color="#222222", linewidth=1.0)
    axes[3].axhline(0.5, color="#d62728", linestyle="--", linewidth=0.8)
    axes[3].set_ylim(0, 1.05)
    axes[3].set_ylabel("BRB", rotation=0, ha="right", va="center")
    cframes = [_as_int(row["frame"]) for row in candidates if start <= _as_int(row["frame"]) < end]
    sframes = [_as_int(row["frame"]) for row in suppressed if start <= _as_int(row["frame"]) < end]
    bframes = [frame for frame in retained_before if start <= frame < end]
    aframes = [frame for frame in retained_after if start <= frame < end]
    for axis, y, marker, color, frames, label in [
        (axes[4], 0.5, ".", "#777777", cframes, "candidate"),
        (axes[5], 0.5, "o", "#1b9e77", bframes, "retained no-NMS"),
        (axes[6], 0.5, "o", "#2ca02c", aframes, "retained NMS"),
        (axes[7], 0.5, "X", "#d62728", sframes, "suppressed"),
    ]:
        axis.scatter([edges[x] for x in frames], [y] * len(frames), marker=marker, s=80 if marker == "X" else 42, color=color, edgecolors="white" if marker == "X" else None, linewidths=0.6)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    for frame in gt_boundaries:
        if start <= frame < end:
            axes[0].axvline(edges[frame], color="#111111", linestyle=":", linewidth=1.1)
            axes[3].axvline(edges[frame], color="#111111", linestyle=":", linewidth=1.1)
    for event in suppressed:
        frame = _as_int(event["frame"])
        if not start <= frame < end:
            continue
        text = f"{frame}\n{float(event['probability']):.3f}\nΔR {int(event['distance_frames'])}f\nΔGT {int(event['nearest_truth_distance'])}f"
        axes[3].annotate(text, xy=(edges[frame], brb[frame]), xytext=(edges[frame], min(1.02, brb[frame] + 0.22)), ha="center", fontsize=7, color="#8b0000", bbox={"boxstyle": "round,pad=0.2", "facecolor": "#fff3f3", "edgecolor": "#d62728"}, arrowprops={"arrowstyle": "-", "color": "#d62728", "lw": 0.7})
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(edges[start], edges[end])
    fig.suptitle(title, x=0.06, ha="left", fontsize=13, fontweight="bold")
    local_events = [event for event in suppressed if start <= int(event["frame"]) < end]
    event_lines = ["suppressed peaks"] + [
        f"frame {int(event['frame'])}: t={float(event['time_s']):.2f}s, p={float(event['probability']):.3f}, "
        f"ΔR={int(event['distance_frames'])}f, ΔGT={int(event['nearest_truth_distance'])}f"
        for event in sorted(local_events, key=lambda row: int(row["frame"]))
    ]
    fig.text(0.995, 0.94, "\n".join(event_lines), ha="right", va="top", fontsize=8, family="monospace", bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff8f8", "edgecolor": "#d62728", "alpha": 0.96})
    fig.tight_layout(rect=(0, 0, 0.72, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _load(trajectory: str, source_family: str) -> dict[str, Any]:
    mapping = load_label_mapping(ROOT / "configs/labels_multitask_plug.yaml")
    sample = load_trajectory_sample(DATA / trajectory, mapping, expected_height=88)
    heatmap = sample["heatmap"].numpy()
    timestamps = sample["timestamps"].numpy()
    width = int(heatmap.shape[-1])
    if width != len(timestamps):
        raise ValueError("heatmap/timestamp width mismatch")
    inverse = {int(index): name for name, index in mapping.items()}
    gt = _segments_from_labels(sample["labels"].numpy(), inverse)
    trajectory_name = Path(trajectory).name
    before_dir = SOURCE_ROOT / source_family / "d0" / trajectory_name
    after_dir = SOURCE_ROOT / source_family / "d10" / trajectory_name
    candidates0 = _read_csv(before_dir / "candidate_peaks.csv")
    candidates10 = _read_csv(after_dir / "candidate_peaks.csv")
    candidate_frames0 = [_as_int(row["frame"]) for row in candidates0]
    candidate_frames10 = [_as_int(row["frame"]) for row in candidates10]
    if candidate_frames0 != candidate_frames10:
        raise ValueError("d0 and d10 candidate peak lists differ")
    frame0 = _read_csv(before_dir / "frame_predictions.csv")
    frame10 = _read_csv(after_dir / "frame_predictions.csv")
    if len(frame0) != width or len(frame10) != width:
        raise ValueError("frame prediction width mismatch")
    for left, right in zip(frame0, frame10):
        if left["frame_index"] != right["frame_index"] or left["raw_asb_label"] != right["raw_asb_label"] or left["brb_probability"] != right["brb_probability"]:
            raise ValueError("d0/d10 frame predictions are not aligned")
    segments0_rows = _read_csv(before_dir / "segment_predictions.csv")
    segments10_rows = _read_csv(after_dir / "segment_predictions.csv")
    before = _prediction_segments(segments0_rows, "d0_official_class")
    after = _prediction_segments(segments10_rows, "current_nms_class")
    _segment_schema(before, width)
    _segment_schema(after, width)
    candidates = [{"frame": _as_int(row["frame"]), "time_s": _as_float(row["time_s"]), "probability": _as_float(row["brb_probability"])} for row in candidates0]
    retained_before = _event_frames(candidates0, "retained")
    retained_after = _event_frames(candidates10, "retained")
    suppression_rows = _read_csv(after_dir / "suppressed_peaks.csv")
    gt_boundaries = _gt_boundaries(gt)
    candidate_time = {int(row["frame"]): _as_float(row["time_s"]) for row in candidates0}
    suppressed: list[dict[str, Any]] = []
    for row in suppression_rows:
        frame = _as_int(row["frame"])
        suppressed.append({
            "frame": frame,
            "time_s": candidate_time[frame],
            "probability": _as_float(row["probability"]),
            "suppressing_retained_peak_frame": _as_int(row["suppressing_retained_peak_frame"]),
            "distance_frames": _as_int(row["distance_frames"]),
            "nearest_truth_distance": int(_nearest_distance(frame, gt_boundaries) if _nearest_distance(frame, gt_boundaries) is not None else -1),
        })
    brb = np.asarray([_as_float(row["brb_probability"]) for row in frame0], dtype=float)
    return {"trajectory": trajectory, "mapping": mapping, "sample": sample, "heatmap": heatmap, "timestamps": timestamps, "edges": _time_edges(timestamps), "gt": gt, "before": before, "after": after, "candidates": candidates, "retained_before": retained_before, "retained_after": retained_after, "suppressed": suppressed, "brb": brb, "gt_boundaries": gt_boundaries, "source_files": [before_dir / name for name in ["candidate_peaks.csv", "frame_predictions.csv", "segment_predictions.csv", "retained_peaks.csv", "suppressed_peaks.csv"]] + [after_dir / name for name in ["candidate_peaks.csv", "frame_predictions.csv", "segment_predictions.csv", "retained_peaks.csv", "suppressed_peaks.csv"]]}


def _summaries(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suppressed = data["suppressed"]
    groups = _group_suppression_events(suppressed)
    before = data["before"]
    after = data["after"]
    gt_boundaries = data["gt_boundaries"]
    region_rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups, 1):
        suppressor_frames = [int(e["suppressing_retained_peak_frame"]) for e in group]
        event_start = min(min(int(e["frame"]) for e in group), min(suppressor_frames))
        event_end = max(max(int(e["frame"]) for e in group), max(suppressor_frames)) + 1
        before_labels = _collapsed_labels(before, event_start, event_end)
        after_labels = _collapsed_labels(after, event_start, event_end)
        same_label = sum(1 for event in group if _removed_boundary_semantics(int(event["frame"]), before)[1]) > 0
        semantic_changed = sum(1 for event in group if _removed_boundary_semantics(int(event["frame"]), before)[0]) > 0
        nearest_gt = min(int(e["nearest_truth_distance"]) for e in group)
        region_rows.append({
            "trajectory": data["trajectory"],
            "region_index": index,
            "region_start_s": float(data["edges"][event_start]),
            "region_end_s": float(data["edges"][event_end]),
            "candidate_count": sum(event_start <= int(row["frame"]) < event_end for row in data["candidates"]),
            "suppressed_count": len(group),
            "retained_count": sum(event_start <= frame < event_end for frame in data["retained_after"]),
            "nearest_gt_boundary_error_frames": nearest_gt,
            "semantic_segment_count_before": len(_overlapping(before, event_start, event_end)),
            "semantic_segment_count_after": len(_overlapping(after, event_start, event_end)),
            "semantic_labels_before": "|".join(before_labels),
            "semantic_labels_after": "|".join(after_labels),
            "semantic_output_changed": semantic_changed,
            "same_label_merge": same_label,
            "false_peak_removed": all(int(e["nearest_truth_distance"]) > 33 for e in group),
            "possible_true_boundary_removed": any(int(e["nearest_truth_distance"]) <= 33 for e in group),
        })
    semantic_boundary_changes = sum(1 for event in suppressed if _removed_boundary_semantics(int(event["frame"]), before)[0])
    same_label_merges = sum(1 for event in suppressed if _removed_boundary_semantics(int(event["frame"]), before)[1])
    false_removed = sum(1 for event in suppressed if int(event["nearest_truth_distance"]) > 33)
    trajectory_row = {
        "trajectory": data["trajectory"],
        "candidate_peaks": len(data["candidates"]),
        "retained_no_nms": len(data["retained_before"]),
        "retained_nms": len(data["retained_after"]),
        "suppressed_peaks": len(suppressed),
        "predicted_segments_no_nms": len(before),
        "predicted_segments_nms": len(after),
        "semantic_boundaries_removed": semantic_boundary_changes,
        "semantic_labels_changed": _collapsed_labels(before, 0, len(data["edges"]) - 1) != _collapsed_labels(after, 0, len(data["edges"]) - 1),
        "same_label_merges": same_label_merges,
        "false_peaks_removed": false_removed,
        "missed_boundaries_change": 0,
    }
    return region_rows, trajectory_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root
    entries = [
        ("test/pour/p1", "pour"), ("test/pour/p2", "pour"),
        ("test/wipe/w1", "wipe"), ("test/wipe/w2", "wipe"),
        ("test/plug/p1", "plug_restricted_p1_p2"), ("test/plug/p2", "plug_restricted_p1_p2"),
    ]
    all_region_rows: list[dict[str, Any]] = []
    all_trajectory_rows: list[dict[str, Any]] = []
    all_manifests: list[dict[str, Any]] = []
    for trajectory, source_family in entries:
        data = _load(trajectory, source_family)
        region_rows, trajectory_row = _summaries(data)
        all_region_rows.extend(region_rows)
        all_trajectory_rows.append(trajectory_row)
        slug = trajectory.replace("/", "_")
        full_path = output_root / f"full/{slug}_nms_direct_comparison.png"
        _plot_full(heatmap=data["heatmap"], edges=data["edges"], gt=data["gt"], before=data["before"], after=data["after"], candidates=data["candidates"], retained_before=data["retained_before"], retained_after=data["retained_after"], suppressed=data["suppressed"], brb=data["brb"], output=full_path, title=f"{trajectory} — r5 NMS direct comparison", gt_boundaries=data["gt_boundaries"], groups=_group_suppression_events(data["suppressed"]))
        zoom_root = output_root / f"zoom/{slug}"
        zoom_paths: list[str] = []
        for index, group in enumerate(_group_suppression_events(data["suppressed"]), 1):
            frames = [int(event["frame"]) for event in group] + [int(event["suppressing_retained_peak_frame"]) for event in group]
            start = max(0, min(frames) - 75)
            end = min(len(data["edges"]) - 1, max(frames) + 76)
            start_s = float(data["edges"][start])
            end_s = float(data["edges"][end])
            path = zoom_root / f"{slug}_zoom_{index:02d}_{start_s:.1f}_{end_s:.1f}s.png"
            _plot_zoom(heatmap=data["heatmap"], edges=data["edges"], gt=data["gt"], before=data["before"], after=data["after"], candidates=data["candidates"], retained_before=data["retained_before"], retained_after=data["retained_after"], suppressed=data["suppressed"], brb=data["brb"], gt_boundaries=data["gt_boundaries"], start=start, end=end, output=path, title=f"{trajectory} — NMS suppression region {index} ({start_s:.1f}–{end_s:.1f}s)")
            zoom_paths.append(str(path.relative_to(ROOT)))
        all_manifests.append({"trajectory": trajectory, "source_prediction_files": {str(path.relative_to(ROOT)): _sha256(path) for path in data["source_files"]}, "candidate_peaks": len(data["candidates"]), "retained_no_nms": len(data["retained_before"]), "retained_nms": len(data["retained_after"]), "suppressed_peaks": len(data["suppressed"]), "full_figure": str(full_path.relative_to(ROOT)), "zoom_figures": zoom_paths, "region_count": len(region_rows)})
    table_root = output_root / "tables"
    region_fields = list(all_region_rows[0]) if all_region_rows else ["trajectory"]
    _write_csv(table_root / "nms_visual_difference_summary.csv", all_region_rows, region_fields)
    _write_csv(table_root / "nms_visual_difference_trajectory_summary.csv", all_trajectory_rows, list(all_trajectory_rows[0]))
    manifest = {
        "trajectories": all_manifests,
        "inference_rerun": False,
        "evaluated_trajectory_count": len(all_manifests),
        "region_count": len(all_region_rows),
    }
    (output_root / "nms_visual_difference_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "nms_visual_difference_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"evaluated_trajectories": [row["trajectory"] for row in all_trajectory_rows], "figure_count": len(all_manifests), "zoom_count": sum(len(row["zoom_figures"]) for row in all_manifests), "region_count": len(all_region_rows)}, indent=2))


if __name__ == "__main__":
    main()
