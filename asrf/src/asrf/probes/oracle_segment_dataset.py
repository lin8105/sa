"""Ground-truth skill segments cropped before any neural encoding.

The dataset deliberately returns one complete annotation segment at a time.
No neighboring frame, sequence position, task identifier, or trajectory
identifier is part of the feature input; metadata is retained for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from asrf.data.dataset import MultiTaskTrajectoryDataset
from asrf.evaluation.metrics import labels_to_segments


@dataclass(frozen=True)
class OracleSegmentRecord:
    split: str
    task: str
    trajectory_id: str
    path: str
    segment_index: int
    start_frame: int
    end_frame: int
    label_id: int
    label_name: str
    duration_frames: int


def crop_segment_heatmap(heatmap: torch.Tensor, record: OracleSegmentRecord) -> torch.Tensor:
    """Crop ``[3,88,T]`` before encoding, using the exclusive end boundary."""
    if heatmap.ndim != 3:
        raise ValueError(f"Expected [3,88,T], got {tuple(heatmap.shape)}")
    cropped = heatmap[:, :, record.start_frame:record.end_frame]
    if cropped.shape[-1] != record.duration_frames or cropped.shape[-1] <= 0:
        raise ValueError("Segment crop has an invalid temporal width.")
    return cropped.contiguous()


class OracleSegmentDataset(Dataset[tuple[OracleSegmentRecord, torch.Tensor]]):
    """Lazy segment dataset; trajectories are loaded, then immediately cropped."""

    def __init__(self, trajectory_dataset: MultiTaskTrajectoryDataset, split_name: str) -> None:
        self.trajectory_dataset = trajectory_dataset
        self.split_name = split_name
        self.records: list[OracleSegmentRecord] = []
        names = {value: name for name, value in trajectory_dataset.label_mapping.items()}
        for trajectory_index, entry in enumerate(trajectory_dataset.entries):
            sample = trajectory_dataset[trajectory_index]
            segments = labels_to_segments(sample["labels"], sample["valid_mask"])
            path = str(trajectory_dataset.resolved_paths[trajectory_index])
            task = str(sample.get("task_name", "unknown"))
            for segment_index, segment in enumerate(segments):
                start = int(segment.start)
                end = int(segment.end) + 1
                label_id = int(segment.label)
                self.records.append(OracleSegmentRecord(split_name, task, entry, path, segment_index, start, end, label_id, names[label_id], end - start))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[OracleSegmentRecord, torch.Tensor]:
        record = self.records[index]
        trajectory_index = self.trajectory_dataset.entries.index(record.trajectory_id)
        sample = self.trajectory_dataset[trajectory_index]
        return record, crop_segment_heatmap(sample["heatmap"], record)

    def __iter__(self) -> Iterator[tuple[OracleSegmentRecord, torch.Tensor]]:
        for index in range(len(self)):
            yield self[index]


def build_oracle_segment_dataset(
    dataset_root: str | Path,
    split_path: str | Path,
    label_path: str | Path,
    *,
    split_name: str,
    allow_test: bool = False,
) -> OracleSegmentDataset:
    trajectories = MultiTaskTrajectoryDataset(dataset_root, split_path, label_path, expected_height=88, allow_test=allow_test)
    return OracleSegmentDataset(trajectories, split_name)


__all__ = ["OracleSegmentDataset", "OracleSegmentRecord", "build_oracle_segment_dataset", "crop_segment_heatmap"]
