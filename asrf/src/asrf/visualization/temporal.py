"""Canonical temporal segmentation comparison plots.

The plotting contract is intentionally strict: a temporally aligned heatmap is
followed by GT, RAW_ASB, and ASRF segment rows, with optional BRB diagnostics.
Prediction labels are validated against the model ontology and are never
derived from GT overlap records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_LABEL_COLORS = {
    "reach": "#8ecae6",
    "grasp": "#ffb703",
    "lift": "#90be6d",
    "transport": "#f28482",
    "place": "#cdb4db",
    "release": "#b7a99a",
    "retreat": "#577590",
    "pour": "#48cae4",
    "pour_recover": "#b5bd00",
    "wipe": "#999999",
    "insert": "#718355",
}


def validate_prediction_segments(segments: Sequence[dict], ontology: set[str]) -> None:
    """Validate non-overlapping, complete prediction segment records.

    The function deliberately rejects GT matching fields in prediction rows so
    that evaluation records cannot accidentally be plotted as predictions.
    """

    forbidden = {"novel_skill", "best_iou", "fragment_count", "gt_start_frame"}
    previous_end = None
    for row in segments:
        if forbidden & set(row):
            raise AssertionError("GT matching fields found in prediction segment row")
        if "start_frame" not in row or "end_frame" not in row:
            raise AssertionError("prediction segment requires start_frame and end_frame")
        label = str(row.get("predicted_label", ""))
        if label not in ontology:
            raise AssertionError(f"prediction label outside ontology: {label}")
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        if end <= start:
            raise AssertionError("prediction segment must have positive duration")
        if previous_end is not None and start < previous_end:
            raise AssertionError("prediction segments overlap")
        previous_end = end


def _normalized_heatmap(heatmap: np.ndarray) -> np.ndarray:
    array = np.asarray(heatmap, dtype=float)
    if array.ndim != 3 or array.shape[0] not in (1, 3):
        raise ValueError("heatmap must have shape [1|3, H, T]")
    array = array[:3]
    low = np.nanpercentile(array, 1)
    high = np.nanpercentile(array, 99)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(array))
        high = float(np.nanmax(array))
    scaled = np.clip((array - low) / max(high - low, 1e-8), 0.0, 1.0)
    if scaled.shape[0] == 1:
        scaled = np.repeat(scaled, 3, axis=0)
    return np.moveaxis(scaled, 0, -1)


def _segments_from_frame_labels(labels: Sequence[str]) -> list[dict]:
    if not labels:
        return []
    starts = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    ends = starts[1:] + [len(labels)]
    return [{"start_frame": start, "end_frame": end, "predicted_label": labels[start]} for start, end in zip(starts, ends)]


def _draw_segments(axis, segments: Sequence[dict], time_axis: np.ndarray, colors: dict[str, str], label_key: str, gt: bool = False) -> None:
    duration = len(time_axis)
    for index, row in enumerate(segments):
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        label = str(row.get(label_key, row.get("predicted_label", "")))
        start = max(0, min(start, duration))
        end = max(start, min(end, duration))
        if end <= start:
            continue
        alpha = 0.82 if gt else 0.62
        axis.axvspan(time_axis[start], time_axis[end - 1] if end > start else time_axis[start], color=colors.get(label, "#cccccc"), alpha=alpha)
        if end - start >= max(25, duration // 120):
            axis.text(float((time_axis[start] + time_axis[end - 1]) / 2), 0.5, label, ha="center", va="center", fontsize=7, clip_on=True)
        elif end - start >= 8:
            axis.text(float(time_axis[start]), 0.5, str(index), ha="left", va="center", fontsize=5, clip_on=True)
    axis.set_ylim(0, 1)
    axis.set_yticks([])


def plot_temporal_comparison(
    heatmap: np.ndarray,
    timestamps: Sequence[float],
    ground_truth_segments: Sequence[dict],
    raw_asb_segments: Sequence[dict],
    asrf_segments: Sequence[dict],
    *,
    brb_probability: Sequence[float] | None = None,
    raw_change_points: Sequence[int] | None = None,
    asrf_boundary_peaks: Sequence[int] | None = None,
    output_path: str | Path,
    ontology: set[str],
    title: str,
    candidate_boundaries: Sequence[int] | None = None,
    label_colors: dict[str, str] | None = None,
) -> None:
    """Draw the canonical heatmap/GT/RAW_ASB/ASRF temporal comparison."""

    import matplotlib.pyplot as plt

    width = len(timestamps)
    if np.asarray(heatmap).shape[-1] != width:
        raise ValueError("heatmap width and timestamp width do not match")
    if width == 0:
        raise ValueError("empty temporal axis")
    validate_prediction_segments(raw_asb_segments, ontology)
    validate_prediction_segments(asrf_segments, ontology)
    if not ground_truth_segments:
        raise ValueError("ground truth segments are required")
    gt_segments = [{**row, "gt_label": str(row.get("gt_label", row.get("label", "")))} for row in ground_truth_segments]
    colors = {**DEFAULT_LABEL_COLORS, **(label_colors or {})}
    time_axis = np.asarray(timestamps, dtype=float)
    if np.nanmax(time_axis) > 1e6:
        time_axis = (time_axis - time_axis[0]) / 1e6
    else:
        time_axis = time_axis - time_axis[0]
    raw_labels = [str(row["predicted_label"]) for row in raw_asb_segments]
    asrf_labels = [str(row["predicted_label"]) for row in asrf_segments]
    rows = 4 + (2 if brb_probability is not None or raw_change_points is not None or asrf_boundary_peaks is not None else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(18, max(9, rows * 1.35)), sharex=True, gridspec_kw={"height_ratios": [2.3, 1, 1, 1] + ([1.3, 0.8] if rows > 4 else [])})
    axes = np.atleast_1d(axes)
    axes[0].imshow(_normalized_heatmap(np.asarray(heatmap)), aspect="auto", origin="upper", extent=[time_axis[0], time_axis[-1], 0, np.asarray(heatmap).shape[1]])
    axes[0].set_ylabel("heatmap\nchannels", rotation=0, ha="right", va="center")
    axes[0].set_yticks([])
    _draw_segments(axes[1], gt_segments, time_axis, colors, "gt_label", gt=True)
    axes[1].set_ylabel("truth", rotation=0, ha="right", va="center")
    _draw_segments(axes[2], raw_asb_segments, time_axis, colors, "predicted_label")
    axes[2].set_ylabel("raw ASB", rotation=0, ha="right", va="center")
    _draw_segments(axes[3], asrf_segments, time_axis, colors, "predicted_label")
    axes[3].set_ylabel("ASRF", rotation=0, ha="right", va="center")
    if rows > 4:
        probability = np.asarray(brb_probability if brb_probability is not None else np.zeros(width), dtype=float)
        axes[4].plot(time_axis, probability, color="#222222", linewidth=0.8)
        axes[4].axhline(0.5, color="#d62728", linestyle="--", linewidth=0.7)
        axes[4].set_ylim(0, 1.05)
        axes[4].set_ylabel("BRB", rotation=0, ha="right", va="center")
        axes[5].set_yticks([1, 2, 3, 4], ["GT", "RAW", "ASRF", "candidate"])
        gt_points = [int(x["start_frame"]) for x in gt_segments[1:]]
        axes[5].eventplot([[time_axis[x] for x in gt_points]], lineoffsets=1, colors="#1f77b4")
        axes[5].eventplot([[time_axis[x] for x in (raw_change_points or []) if 0 < x < width]], lineoffsets=2, colors="#ff7f0e")
        axes[5].eventplot([[time_axis[x] for x in (asrf_boundary_peaks or []) if 0 < x < width]], lineoffsets=3, colors="#2ca02c")
        axes[5].eventplot([[time_axis[x] for x in (candidate_boundaries or []) if 0 < x < width]], lineoffsets=4, colors="#9467bd")
        axes[5].set_ylim(0.5, 4.5)
        axes[5].set_ylabel("boundaries", rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(time_axis[0], time_axis[-1])
    fig.suptitle(title)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
