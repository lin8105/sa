"""Independent, read-only CITR heatmap trajectory loader."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .annotations import convert_segments_to_frame_labels
from .boundary_targets import generate_boundary_targets
from .labels import LabelMapping, load_label_mapping


def read_split_file(path: str | Path) -> list[str]:
    """Read non-empty trajectory IDs without modifying the split file."""
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_timestamp_vector(path: str | Path) -> np.ndarray:
    """Read the per-column timestamp vector from ``citr_features.csv``."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "timestamp_us" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: missing timestamp_us column.")
        values = [int((row.get("timestamp_us") or "").strip()) for row in reader]
    timestamps = np.asarray(values, dtype=np.int64)
    if not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{path}: timestamps must be non-empty and strictly increasing.")
    return timestamps


def load_heatmap(path: str | Path, *, expected_height: int = 88) -> torch.Tensor:
    """Load RGB heatmap pixels as [3, H, T], preserving source width exactly."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32).copy() / 255.0
    height, width, channels = array.shape
    if channels != 3 or height != expected_height:
        raise ValueError(f"{path}: expected RGB height {expected_height}, got {array.shape}.")
    return torch.from_numpy(np.transpose(array, (2, 0, 1)))


def load_trajectory_sample(
    demonstration_path: str | Path,
    label_mapping: LabelMapping,
    *,
    expected_height: int = 88,
    boundary_target_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one labeled demonstration without resizing or writing external data."""
    demo = Path(demonstration_path)
    heatmap_path = demo / "citr_fingerprint_pure.png"
    timestamp_path = demo / "citr_features.csv"
    segments_path = demo / "segments.csv"
    heatmap = load_heatmap(heatmap_path, expected_height=expected_height)
    timestamps = load_timestamp_vector(timestamp_path)
    if heatmap.shape[-1] != len(timestamps):
        raise ValueError(f"{demo}: heatmap width does not match timestamp rows; resizing is forbidden.")
    labels, segments = convert_segments_to_frame_labels(segments_path, timestamps, label_mapping)
    labels_tensor = torch.from_numpy(labels).to(dtype=torch.long)
    target_config = dict(boundary_target_config or {})
    hard_boundary_targets = generate_boundary_targets(labels_tensor)
    boundary_targets = generate_boundary_targets(labels_tensor, **target_config)
    return {
        "heatmap": heatmap,
        "labels": labels_tensor,
        "boundary_targets": boundary_targets,
        "hard_boundary_targets": hard_boundary_targets,
        "valid_mask": torch.ones(len(labels_tensor), dtype=torch.bool),
        "trajectory_id": demo.name,
        "timestamps": torch.from_numpy(timestamps),
        "segments": segments,
        "demonstration_path": demo,
    }


class TrajectoryDataset(Dataset[dict[str, Any]]):
    """Dataset indexed by copied split IDs and an external data root."""

    def __init__(self, dataset_root: str | Path, split_path: str | Path, label_path: str | Path, *, expected_height: int = 88, boundary_target_config: dict[str, Any] | None = None) -> None:
        self.dataset_root = Path(dataset_root)
        self.split_path = Path(split_path)
        self.label_mapping = load_label_mapping(label_path)
        self.trajectory_ids = read_split_file(split_path)
        self.expected_height = expected_height
        self.boundary_target_config = dict(boundary_target_config or {})

    def __len__(self) -> int:
        return len(self.trajectory_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory_id = self.trajectory_ids[index]
        return load_trajectory_sample(
            self.dataset_root / trajectory_id,
            self.label_mapping,
            expected_height=self.expected_height, boundary_target_config=self.boundary_target_config,
        )


class MultiTaskTrajectoryDataset(Dataset[dict[str, Any]]):
    """Resolve split entries relative to the global dataset root.

    Entries are paths such as ``train/pour/p1`` or ``test/wipe/w1``.  The
    basename is never used as the resolver key, preventing collisions between
    task roots and keeping train/validation/test provenance explicit.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split_path: str | Path,
        label_path: str | Path,
        *,
        expected_height: int = 88,
        allow_test: bool = False,
        boundary_target_config: dict[str, Any] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.split_path = Path(split_path).resolve()
        self.label_mapping = load_label_mapping(label_path)
        self.entries = read_split_file(self.split_path)
        # Keep the existing trainer/checkpoint interface while preserving the
        # full relative path as the unambiguous trajectory identifier.
        self.trajectory_ids = list(self.entries)
        self.expected_height = expected_height
        self.allow_test = bool(allow_test)
        self.boundary_target_config = dict(boundary_target_config or {})
        resolved: list[Path] = []
        for entry in self.entries:
            relative = Path(entry)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Split entry must be a relative dataset path: {entry!r}")
            if not self.allow_test and relative.parts and relative.parts[0] == "test":
                raise ValueError(f"Test path is not allowed in this dataset: {entry}")
            path = (self.dataset_root / relative).resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"Split entry does not resolve to a directory: {entry} -> {path}")
            if path in resolved:
                raise ValueError(f"Duplicate physical recording in split: {path}")
            resolved.append(path)
        self.resolved_paths = resolved

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        sample = load_trajectory_sample(
            self.resolved_paths[index], self.label_mapping, expected_height=self.expected_height,
            boundary_target_config=self.boundary_target_config,
        )
        sample["trajectory_id"] = entry
        sample["relative_path"] = entry
        sample["task_name"] = self._task_name(entry)
        sample["recording_family"] = "unavailable"
        return sample

    @staticmethod
    def _task_name(entry: str) -> str:
        parts = Path(entry).parts
        if len(parts) < 2:
            return "unknown"
        task = parts[1]
        return "pick_and_place" if task == "pick and place" else task


__all__ = ["MultiTaskTrajectoryDataset", "TrajectoryDataset", "load_heatmap", "load_trajectory_sample", "read_split_file"]
