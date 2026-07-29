"""Round 8 multi-target comparison plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .temporal import DEFAULT_LABEL_COLORS, _draw_segments, _normalized_heatmap, validate_prediction_segments


def plot_round8_comparison_figure(
    heatmap: np.ndarray,
    timestamps: Sequence[float],
    ground_truth_segments: Sequence[dict],
    method_segments_dict: Mapping[str, Sequence[dict]],
    output_path: str | Path,
    *,
    brb_probabilities: Mapping[str, Sequence[float]] | None = None,
    boundary_peaks: Mapping[str, Sequence[int]] | None = None,
    title: str | None = None,
    ontology: set[str],
    label_colors: Mapping[str, str] | None = None,
) -> None:
    """Draw heatmap, GT, and exactly the supplied ordered method rows."""

    import matplotlib.pyplot as plt

    names = list(method_segments_dict)
    expected = ["ASRF-SF", "r5", "r10", "r20", "s5", "s10", "s20"]
    if names != expected:
        raise ValueError(f"method row order must be {expected}, got {names}")
    width = len(timestamps)
    array = np.asarray(heatmap)
    if array.ndim != 3 or array.shape[-1] != width:
        raise ValueError("heatmap width must match timestamp width")
    if not ground_truth_segments:
        raise ValueError("ground-truth segments are required")
    for rows in method_segments_dict.values():
        validate_prediction_segments(rows, ontology)
    colors = {**DEFAULT_LABEL_COLORS, **(label_colors or {})}
    time_axis = np.asarray(timestamps, dtype=float)
    if np.nanmax(time_axis) > 1e6:
        time_axis = (time_axis - time_axis[0]) / 1e6
    else:
        time_axis = time_axis - time_axis[0]
    diagnostic = brb_probabilities is not None or boundary_peaks is not None
    nrows = 2 + len(names) + (2 if diagnostic else 0)
    ratios = [2.5, 1.0] + [1.0] * len(names) + ([1.5, 0.9] if diagnostic else [])
    fig, axes = plt.subplots(nrows, 1, figsize=(19, max(13, nrows * 1.25)), sharex=True, gridspec_kw={"height_ratios": ratios})
    axes = np.atleast_1d(axes)
    axes[0].imshow(_normalized_heatmap(array), aspect="auto", origin="upper", extent=[time_axis[0], time_axis[-1], 0, array.shape[1]])
    axes[0].set_ylabel("heatmap\nchannels", rotation=0, ha="right", va="center")
    axes[0].set_yticks([])
    _draw_segments(axes[1], ground_truth_segments, time_axis, colors, "gt_label", gt=True)
    axes[1].set_ylabel("truth", rotation=0, ha="right", va="center")
    for index, name in enumerate(names, start=2):
        _draw_segments(axes[index], method_segments_dict[name], time_axis, colors, "predicted_label")
        axes[index].set_ylabel(name, rotation=0, ha="right", va="center")
    if diagnostic:
        probability_axis = axes[2 + len(names)]
        if brb_probabilities:
            for name in names:
                probability_axis.plot(time_axis, np.asarray(brb_probabilities[name], dtype=float), linewidth=0.75, label=name)
        probability_axis.axhline(0.5, color="#d62728", linestyle="--", linewidth=0.7, label="threshold 0.50")
        probability_axis.set_ylim(0, 1.05)
        probability_axis.set_ylabel("BRB", rotation=0, ha="right", va="center")
        probability_axis.legend(ncol=4, fontsize=7, loc="upper right")
        boundary_axis = axes[3 + len(names)]
        boundary_axis.set_yticks(range(1, len(names) + 2), ["GT"] + names)
        gt_points = [int(row["start_frame"]) for row in ground_truth_segments[1:]]
        boundary_axis.eventplot([[time_axis[x] for x in gt_points]], lineoffsets=1, colors="#111111")
        for row_index, name in enumerate(names, start=2):
            points = (boundary_peaks or {}).get(name, [])
            boundary_axis.eventplot([[time_axis[x] for x in points if 0 < x < width]], lineoffsets=row_index, colors=colors.get(name, "#555555"))
        boundary_axis.set_ylim(0.5, len(names) + 1.5)
        boundary_axis.set_ylabel("boundaries", rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(time_axis[0], time_axis[-1])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
