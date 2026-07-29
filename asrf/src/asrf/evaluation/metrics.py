"""Independent frame, segment, and boundary metrics for ASRF training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class Segment:
    label: int
    start: int
    end: int  # inclusive

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def labels_to_segments(labels: Any, valid_mask: Any | None = None) -> list[Segment]:
    values = np.asarray(labels.detach().cpu() if isinstance(labels, torch.Tensor) else labels)
    if values.ndim != 1:
        raise ValueError("labels must have shape [T].")
    if valid_mask is None:
        mask = np.ones(len(values), dtype=bool)
    else:
        mask = np.asarray(valid_mask.detach().cpu() if isinstance(valid_mask, torch.Tensor) else valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must match labels.")
    segments: list[Segment] = []
    start: int | None = None
    current: int | None = None
    for index, value in enumerate(values):
        if not mask[index] or int(value) < 0:
            if start is not None:
                segments.append(Segment(int(current), start, index - 1))
                start, current = None, None
            continue
        label = int(value)
        if start is None:
            start, current = index, label
        elif label != current:
            segments.append(Segment(int(current), start, index - 1))
            start, current = index, label
    if start is not None:
        segments.append(Segment(int(current), start, len(values) - 1))
    return segments


def frame_accuracy(prediction: Any, target: Any, valid_mask: Any | None = None) -> float:
    pred = np.asarray(prediction.detach().cpu() if isinstance(prediction, torch.Tensor) else prediction)
    truth = np.asarray(target.detach().cpu() if isinstance(target, torch.Tensor) else target)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have equal shape.")
    if valid_mask is None:
        mask = np.ones(pred.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask.detach().cpu() if isinstance(valid_mask, torch.Tensor) else valid_mask, dtype=bool)
    if mask.shape != pred.shape:
        raise ValueError("valid_mask must match prediction.")
    return float(np.mean(pred[mask] == truth[mask])) if mask.any() else 0.0


def edit_score(prediction: Any, target: Any, valid_mask: Any | None = None) -> float:
    predicted = [segment.label for segment in labels_to_segments(prediction, valid_mask)]
    truth = [segment.label for segment in labels_to_segments(target, valid_mask)]
    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0
    previous = list(range(len(truth) + 1))
    for value in predicted:
        current = [previous[0] + 1]
        for index, expected in enumerate(truth, start=1):
            current.append(min(current[-1] + 1, previous[index] + 1, previous[index - 1] + (value != expected)))
        previous = current
    return float(1.0 - previous[-1] / max(len(predicted), len(truth)))


def _iou(first: Segment, second: Segment) -> float:
    start = max(first.start, second.start)
    end = min(first.end, second.end)
    intersection = max(0, end - start + 1)
    return float(intersection / (first.length + second.length - intersection)) if intersection else 0.0


def segmental_f1(prediction: Any, target: Any, overlap: float, valid_mask: Any | None = None) -> float:
    predicted = labels_to_segments(prediction, valid_mask)
    truth = labels_to_segments(target, valid_mask)
    candidates: list[tuple[float, int, int]] = []
    for pi, pseg in enumerate(predicted):
        for ti, tseg in enumerate(truth):
            if pseg.label == tseg.label:
                score = _iou(pseg, tseg)
                if score >= overlap:
                    candidates.append((score, pi, ti))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_p: set[int] = set()
    used_t: set[int] = set()
    tp = 0
    for _, pi, ti in candidates:
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            tp += 1
    fp = len(predicted) - tp
    fn = len(truth) - tp
    if not predicted and not truth:
        return 1.0
    return float(2 * tp / (2 * tp + fp + fn)) if tp or fp or fn else 0.0


def boundary_indices_from_labels(labels: Any, valid_mask: Any | None = None, *, include_frame0: bool = True) -> list[int]:
    segments = labels_to_segments(labels, valid_mask)
    if not segments:
        return []
    indices = [segment.start for segment in segments[1:]]
    if include_frame0:
        return [segments[0].start] + indices
    return indices


def boundary_counts(predicted: Iterable[int], target: Iterable[int], tolerance: int, *, include_frame0: bool) -> dict[str, int | float]:
    pred = sorted({int(v) for v in predicted if include_frame0 or int(v) != 0})
    truth = sorted({int(v) for v in target if include_frame0 or int(v) != 0})
    candidates = sorted((abs(p - t), pi, ti) for pi, p in enumerate(pred) for ti, t in enumerate(truth) if abs(p - t) <= tolerance)
    used_p: set[int] = set()
    used_t: set[int] = set()
    tp = 0
    for _, pi, ti in candidates:
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            tp += 1
    fp = len(pred) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "predicted_count": len(pred), "target_count": len(truth)}


def trajectory_metrics(prediction: Any, target: Any, valid_mask: Any | None = None) -> dict[str, float | int]:
    return {
        "frame_accuracy": frame_accuracy(prediction, target, valid_mask),
        "edit_score": edit_score(prediction, target, valid_mask),
        "f1@10": segmental_f1(prediction, target, 0.10, valid_mask),
        "f1@25": segmental_f1(prediction, target, 0.25, valid_mask),
        "f1@50": segmental_f1(prediction, target, 0.50, valid_mask),
        "segment_count": len(labels_to_segments(prediction, valid_mask)),
        "fragmentation": max(0, len(labels_to_segments(prediction, valid_mask)) - len(labels_to_segments(target, valid_mask))),
    }


def aggregate_trajectory_metrics(rows: list[dict[str, float | int]]) -> dict[str, float]:
    if not rows:
        return {"frame_accuracy": 0.0, "edit_score": 0.0, "f1@10": 0.0, "f1@25": 0.0, "f1@50": 0.0}
    keys = ("frame_accuracy", "edit_score", "f1@10", "f1@25", "f1@50")
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


__all__ = [
    "Segment", "aggregate_trajectory_metrics", "boundary_counts", "boundary_indices_from_labels",
    "edit_score", "frame_accuracy", "labels_to_segments", "segmental_f1", "trajectory_metrics",
]
