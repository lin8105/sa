"""ASRF variable-length batch interface with right-only temporal padding."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def collate_fn(samples: Sequence[dict[str, Any]], *, ignore_index: int = -100) -> dict[str, Any]:
    """Return [B,3,H,Tmax] tensors and a boolean valid mask."""
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    lengths = [int(sample["heatmap"].shape[-1]) for sample in samples]
    channels, height = samples[0]["heatmap"].shape[:2]
    max_length = max(lengths)
    batch_size = len(samples)
    heatmap = samples[0]["heatmap"].new_zeros((batch_size, channels, height, max_length))
    labels = torch.full((batch_size, max_length), ignore_index, dtype=torch.long)
    boundary_targets = torch.zeros((batch_size, max_length), dtype=torch.float32)
    hard_boundary_targets = torch.zeros((batch_size, max_length), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for index, (sample, length) in enumerate(zip(samples, lengths)):
        heatmap[index, :, :, :length] = sample["heatmap"]
        labels[index, :length] = sample["labels"]
        boundary_targets[index, :length] = sample["boundary_targets"]
        hard_boundary_targets[index, :length] = sample.get("hard_boundary_targets", sample["boundary_targets"])
        valid_mask[index, :length] = sample["valid_mask"]
    return {
        "heatmap": heatmap,
        "labels": labels,
        "boundary_targets": boundary_targets,
        "hard_boundary_targets": hard_boundary_targets,
        "valid_mask": valid_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "trajectory_ids": [sample["trajectory_id"] for sample in samples],
        "relative_paths": [sample.get("relative_path", sample["trajectory_id"]) for sample in samples],
        "task_names": [sample.get("task_name", "unknown") for sample in samples],
        "demonstration_paths": [str(sample.get("demonstration_path", "")) for sample in samples],
        "segments": [sample["segments"] for sample in samples],
    }
