"""Read-only segments.csv parsing and frame-label conversion."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np

from .labels import LabelMapping, normalize_label_name

AnnotationFormat = Literal["timestamp", "frame"]


def load_segments_csv(path: str | Path) -> tuple[AnnotationFormat, list[dict[str, str]]]:
    """Read annotation rows and detect timestamp- or frame-based endpoints."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {field.strip() for field in (reader.fieldnames or []) if field}
        timestamp = {"start_timestamp_us", "end_timestamp_us_exclusive", "label"}
        frame = {"start_frame", "end_frame", "label"}
        if timestamp.issubset(fields) == frame.issubset(fields):
            raise ValueError("segments.csv must contain exactly one supported endpoint format.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: segments.csv contains no rows.")
    return ("timestamp" if timestamp.issubset(fields) else "frame"), rows


def convert_segments_to_frame_labels(
    path: str | Path,
    timestamps_us: np.ndarray,
    label_mapping: LabelMapping,
    *,
    background_enabled: bool = False,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Convert exclusive timestamp or inclusive frame segments to labels [T]."""
    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    if timestamps.ndim != 1 or not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps_us must be a non-empty strictly increasing vector.")

    annotation_format, rows = load_segments_csv(path)
    labels = np.full(len(timestamps), -1, dtype=np.int64)
    occupied = np.zeros(len(timestamps), dtype=bool)
    for row_number, row in enumerate(rows, start=2):
        name = normalize_label_name(row.get("label", ""), label_mapping)
        if name not in label_mapping:
            raise ValueError(f"{path}: unknown label {name!r} at row {row_number}.")
        if annotation_format == "timestamp":
            start = int(row["start_timestamp_us"])
            end = int(row["end_timestamp_us_exclusive"])
            start_frame = int(np.searchsorted(timestamps, start, side="left"))
            end_frame = int(np.searchsorted(timestamps, end, side="left"))
        else:
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"]) + 1
        start_frame = max(0, start_frame)
        end_frame = min(len(timestamps), end_frame)
        if start_frame >= end_frame or occupied[start_frame:end_frame].any():
            raise ValueError(f"{path}: invalid or overlapping segment at row {row_number}.")
        labels[start_frame:end_frame] = label_mapping[name]
        occupied[start_frame:end_frame] = True

    if not occupied.all():
        if background_enabled and "background" in label_mapping:
            labels[~occupied] = label_mapping["background"]
        else:
            raise ValueError(f"{path}: annotation coverage does not span all heatmap columns.")
    return labels, rows

