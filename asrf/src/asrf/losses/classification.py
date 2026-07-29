"""Training-only statistics and masked class-weighted ASB losses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from asrf.data.annotations import convert_segments_to_frame_labels, load_segments_csv
from asrf.data.boundary_targets import generate_boundary_targets
from asrf.data.dataset import load_timestamp_vector, read_split_file
from asrf.data.labels import LabelMapping, normalize_label_name


@dataclass(frozen=True)
class TrainingStatistics:
    class_counts: Tensor
    class_frequencies: Tensor
    median_frequency: Tensor
    class_weights: Tensor
    segment_counts: Tensor
    total_valid_frames: int
    boundary_positive_count: int
    boundary_negative_count: int
    boundary_positive_mass: float = 0.0
    boundary_negative_mass: float = 0.0

    @property
    def boundary_positive_ratio(self) -> float:
        total = self.boundary_positive_count + self.boundary_negative_count
        return self.boundary_positive_count / total if total else 0.0

    @property
    def boundary_positive_weight(self) -> float:
        if self.boundary_positive_count <= 0:
            raise ValueError("Cannot compute a positive weight without boundary positives.")
        total = self.boundary_positive_count + self.boundary_negative_count
        return float(total / self.boundary_positive_count)


def median_frequency_class_weights(class_counts: Tensor | Sequence[int], *, allow_zero: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    """Match official ``median / frequency`` class weighting.

    Returns ``(frequencies, median_frequency, weights)``.  A zero-count class
    is an error because the official formula would otherwise produce infinity
    and silently hide a training-split configuration error.
    """

    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    if counts.ndim != 1 or not counts.numel():
        raise ValueError("class_counts must be a non-empty vector.")
    if (counts < 0).any() or ((counts == 0).any() and not allow_zero):
        raise ValueError("Every configured class must have a positive training count.")
    total = counts.sum()
    frequencies = counts / total
    positive = frequencies > 0
    median_frequency = torch.median(frequencies[positive])
    weights = torch.zeros_like(frequencies)
    weights[positive] = median_frequency / frequencies[positive]
    return frequencies, median_frequency, weights


def collect_training_statistics(
    dataset_root: str | Path,
    split_path: str | Path,
    label_mapping: LabelMapping,
    boundary_target_config: dict[str, object] | None = None,
    allow_zero_class_weights: bool = False,
) -> TrainingStatistics:
    """Read labels/annotations from the training split only, read-only."""

    num_classes = len(label_mapping)
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    segment_counts = torch.zeros(num_classes, dtype=torch.long)
    boundary_positive_count = 0
    boundary_negative_count = 0
    boundary_positive_mass = 0.0
    boundary_negative_mass = 0.0
    for trajectory_id in read_split_file(split_path):
        demo = Path(dataset_root) / trajectory_id
        timestamps = load_timestamp_vector(demo / "citr_features.csv")
        annotation_format, rows = load_segments_csv(demo / "segments.csv")
        labels_np, _ = convert_segments_to_frame_labels(
            demo / "segments.csv", timestamps, label_mapping
        )
        labels = torch.from_numpy(labels_np)
        class_counts += torch.bincount(labels, minlength=num_classes)
        for row in rows:
            name = normalize_label_name(row["label"], label_mapping)
            segment_counts[label_mapping[name]] += 1
        targets = generate_boundary_targets(labels, **dict(boundary_target_config or {}))
        positive = targets > 0.5
        boundary_positive_count += int(positive.sum())
        boundary_negative_count += int((~positive).sum())
        boundary_positive_mass += float(targets.sum())
        boundary_negative_mass += float((1.0 - targets).sum())
        if annotation_format not in {"timestamp", "frame"}:  # defensive invariant
            raise ValueError(f"Unsupported annotation format: {annotation_format}")
    frequencies, median_frequency, weights = median_frequency_class_weights(class_counts, allow_zero=allow_zero_class_weights)
    return TrainingStatistics(
        class_counts=class_counts,
        class_frequencies=frequencies,
        median_frequency=median_frequency,
        class_weights=weights,
        segment_counts=segment_counts,
        total_valid_frames=int(class_counts.sum()),
        boundary_positive_count=boundary_positive_count,
        boundary_negative_count=boundary_negative_count,
        boundary_positive_mass=boundary_positive_mass,
        boundary_negative_mass=boundary_negative_mass,
    )


def collect_statistics_for_entries(
    dataset_root: str | Path,
    split_path: str | Path,
    label_mapping: LabelMapping,
    boundary_target_config: dict[str, object] | None = None,
    allow_zero_class_weights: bool = False,
) -> TrainingStatistics:
    """Collect the same statistics for global-root relative split entries."""

    root = Path(dataset_root)
    num_classes = len(label_mapping)
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    segment_counts = torch.zeros(num_classes, dtype=torch.long)
    boundary_positive_count = 0
    boundary_negative_count = 0
    boundary_positive_mass = 0.0
    boundary_negative_mass = 0.0
    for entry in read_split_file(split_path):
        demo = (root / entry).resolve()
        timestamps = load_timestamp_vector(demo / "citr_features.csv")
        annotation_format, rows = load_segments_csv(demo / "segments.csv")
        labels_np, _ = convert_segments_to_frame_labels(demo / "segments.csv", timestamps, label_mapping)
        labels = torch.from_numpy(labels_np)
        class_counts += torch.bincount(labels, minlength=num_classes)
        for row in rows:
            name = normalize_label_name(row["label"], label_mapping)
            segment_counts[label_mapping[name]] += 1
        targets = generate_boundary_targets(labels, **dict(boundary_target_config or {}))
        positive = targets > 0.5
        boundary_positive_count += int(positive.sum())
        boundary_negative_count += int((~positive).sum())
        boundary_positive_mass += float(targets.sum())
        boundary_negative_mass += float((1.0 - targets).sum())
        if annotation_format not in {"timestamp", "frame"}:
            raise ValueError(f"Unsupported annotation format: {annotation_format}")
    frequencies, median_frequency, weights = median_frequency_class_weights(class_counts, allow_zero=allow_zero_class_weights)
    return TrainingStatistics(
        class_counts=class_counts,
        class_frequencies=frequencies,
        median_frequency=median_frequency,
        class_weights=weights,
        segment_counts=segment_counts,
        total_valid_frames=int(class_counts.sum()),
        boundary_positive_count=boundary_positive_count,
        boundary_negative_count=boundary_negative_count,
        boundary_positive_mass=boundary_positive_mass,
        boundary_negative_mass=boundary_negative_mass,
    )


@dataclass(frozen=True)
class CrossEntropyOutput:
    loss: Tensor
    valid_frame_count: int


def masked_class_weighted_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    *,
    class_weights: Tensor | None = None,
) -> CrossEntropyOutput:
    """Official CE with explicit right-padding masking and safe empty batches."""

    if logits.ndim != 3 or targets.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("Expected logits [B,C,T] and targets [B,T].")
    mask = torch.as_tensor(valid_mask, device=logits.device, dtype=torch.bool)
    if mask.shape != targets.shape:
        raise ValueError("valid_mask must have shape [B,T].")
    count = int(mask.sum())
    if count == 0:
        return CrossEntropyOutput(logits.sum() * 0.0, 0)
    safe_targets = targets.masked_fill(~mask, 0).to(dtype=torch.long)
    weights = None if class_weights is None else class_weights.to(device=logits.device, dtype=logits.dtype)
    losses = F.cross_entropy(logits, safe_targets, weight=weights, reduction="none")
    selected = losses[mask]
    if weights is None:
        loss = selected.mean()
    else:
        denominator = weights[safe_targets[mask]].sum()
        loss = selected.sum() / denominator.clamp_min(torch.finfo(losses.dtype).eps)
    return CrossEntropyOutput(loss, count)


class MaskedClassWeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights: Tensor | None = None) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None  # type: ignore[assignment]

    def forward(self, logits: Tensor, targets: Tensor, valid_mask: Tensor) -> CrossEntropyOutput:
        return masked_class_weighted_cross_entropy(
            logits, targets, valid_mask, class_weights=self.class_weights
        )


__all__ = [
    "CrossEntropyOutput",
    "MaskedClassWeightedCrossEntropy",
    "TrainingStatistics",
    "collect_training_statistics",
    "collect_statistics_for_entries",
    "masked_class_weighted_cross_entropy",
    "median_frequency_class_weights",
]
