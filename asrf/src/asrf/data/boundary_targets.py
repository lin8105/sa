"""In-memory boundary targets aligned to CITR heatmap columns.

The official ASRF target is a binary array with a positive at frame zero and
at every frame whose action label differs from the preceding frame.  This
module keeps that convention in memory and never writes dataset artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .labels import LabelMapping, normalize_label_name


def _as_label_tensor(labels: Tensor | Sequence[int]) -> Tensor:
    result = labels if isinstance(labels, Tensor) else torch.as_tensor(labels)
    if result.ndim != 1:
        raise ValueError(f"labels must have shape [T], got {tuple(result.shape)}")
    return result.to(dtype=torch.long)


def generate_boundary_targets(
    labels: Tensor | Sequence[int],
    *,
    valid_mask: Tensor | Sequence[bool] | None = None,
    valid_length: int | None = None,
    boundary_target_mode: str = "single_frame",
    boundary_window_radius: int = 0,
    boundary_gaussian_sigma: float = 1.0,
    boundary_include_frame_zero: bool = True,
    boundary_include_final_frame: bool = False,
) -> Tensor:
    """Generate configurable BRB targets without changing the default target.

    ``single_frame`` marks each boundary at one frame, ``hard_window`` marks
    the clipped radius around it, and ``gaussian`` uses a clipped Gaussian.
    Overlapping targets combine by element-wise maximum.  By default this is
    the historical ASRF target: frame zero plus label transitions, without a
    synthetic final-frame target.
    """

    label_tensor = _as_label_tensor(labels)
    temporal_width = int(label_tensor.numel())
    if valid_length is None:
        valid_length = temporal_width
    if not 0 <= valid_length <= temporal_width:
        raise ValueError("valid_length must be within [0, len(labels)].")
    if valid_mask is None:
        mask = torch.arange(temporal_width, device=label_tensor.device) < valid_length
    else:
        mask = torch.as_tensor(valid_mask, device=label_tensor.device, dtype=torch.bool)
        if mask.shape != label_tensor.shape:
            raise ValueError("valid_mask must have the same shape as labels.")
        mask = mask & (torch.arange(temporal_width, device=label_tensor.device) < valid_length)

    mode = str(boundary_target_mode).strip().lower()
    if mode not in {"single_frame", "hard_window", "gaussian"}:
        raise ValueError(f"Unsupported boundary_target_mode: {boundary_target_mode!r}")
    radius = int(boundary_window_radius)
    if radius < 0:
        raise ValueError("boundary_window_radius must be non-negative.")
    sigma = float(boundary_gaussian_sigma)
    if mode == "gaussian" and sigma <= 0.0:
        raise ValueError("boundary_gaussian_sigma must be positive.")

    targets = torch.zeros(temporal_width, dtype=torch.float32, device=label_tensor.device)
    valid_indices = torch.where(mask)[0]
    if not valid_indices.numel():
        return targets

    first = int(valid_indices[0])
    boundary_indices: list[int] = []
    if boundary_include_frame_zero:
        boundary_indices.append(first)
    if boundary_include_final_frame:
        boundary_indices.append(int(valid_indices[-1]))
    for previous, current in zip(valid_indices[:-1], valid_indices[1:]):
        if int(current) == int(previous) + 1 and label_tensor[current] != label_tensor[previous]:
            boundary_indices.append(int(current))
    for boundary in sorted(set(boundary_indices)):
        if mode == "single_frame":
            targets[boundary] = 1.0
        elif mode == "hard_window":
            start = max(0, boundary - radius)
            end = min(temporal_width, boundary + radius + 1)
            targets[start:end] = torch.maximum(targets[start:end], mask[start:end].to(torch.float32))
        else:
            positions = torch.arange(temporal_width, device=label_tensor.device, dtype=torch.float32)
            gaussian = torch.exp(-0.5 * ((positions - float(boundary)) / sigma) ** 2)
            targets = torch.maximum(targets, gaussian * mask.to(torch.float32))
    return targets


def boundary_targets_from_segments(
    segments: Sequence[Mapping[str, Any]],
    valid_length: int,
    *,
    timestamps_us: Sequence[int] | Tensor | None = None,
    label_mapping: LabelMapping | Mapping[str, int] | None = None,
) -> Tensor:
    """Generate targets from parsed segments without modifying their source.

    Frame-format rows use inclusive ``end_frame``.  Timestamp-format rows use
    exclusive ``end_timestamp_us_exclusive`` and require ``timestamps_us``.
    The segment labels are converted to a framewise sequence first, so an
    adjacent segment with the same canonical label does not create a target
    that the official framewise implementation would not create.
    """

    if valid_length < 0:
        raise ValueError("valid_length must be non-negative.")
    if not segments and valid_length:
        raise ValueError("At least one segment is required for a non-empty trajectory.")
    timestamps = None if timestamps_us is None else torch.as_tensor(timestamps_us, dtype=torch.long)
    if timestamps is not None and timestamps.ndim != 1:
        raise ValueError("timestamps_us must have shape [T].")
    labels = torch.full((valid_length,), -1, dtype=torch.long)
    for row in segments:
        raw_label = row.get("label")
        if label_mapping is None:
            if isinstance(raw_label, str):
                raise ValueError("String segment labels require label_mapping.")
            label_id = int(raw_label)
        else:
            name = normalize_label_name(str(raw_label), label_mapping)  # type: ignore[arg-type]
            if name not in label_mapping:
                raise ValueError(f"Unknown segment label {name!r}.")
            label_id = int(label_mapping[name])

        if "start_frame" in row and "end_frame" in row:
            start = int(row["start_frame"])
            end = int(row["end_frame"]) + 1
        elif "start_timestamp_us" in row and "end_timestamp_us_exclusive" in row:
            if timestamps is None:
                raise ValueError("Timestamp segments require timestamps_us.")
            start = int(torch.searchsorted(timestamps, torch.tensor(int(row["start_timestamp_us"])), right=False))
            end = int(torch.searchsorted(timestamps, torch.tensor(int(row["end_timestamp_us_exclusive"])), right=False))
        else:
            raise ValueError("Unsupported segment endpoint fields.")
        start = max(0, start)
        end = min(valid_length, end)
        if start >= end or (labels[start:end] >= 0).any():
            raise ValueError("Segments must be non-empty, in-range, and non-overlapping.")
        labels[start:end] = label_id

    if (labels < 0).any():
        raise ValueError("Segments must cover every valid frame.")
    return generate_boundary_targets(labels)


def boundary_indices(targets: Tensor | Sequence[float]) -> list[int]:
    """Return positive target indices for diagnostics."""

    tensor = torch.as_tensor(targets)
    return torch.where(tensor > 0)[0].tolist()


__all__ = [
    "boundary_indices",
    "boundary_targets_from_segments",
    "generate_boundary_targets",
]
