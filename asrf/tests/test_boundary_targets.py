from __future__ import annotations

from pathlib import Path

import torch

from asrf.data.annotations import convert_segments_to_frame_labels
from asrf.data.boundary_targets import boundary_targets_from_segments, generate_boundary_targets
from asrf.data.dataset import load_timestamp_vector
from asrf.data.labels import load_label_mapping


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")


def test_two_segments_mark_first_frame_and_one_transition() -> None:
    targets = generate_boundary_targets(torch.tensor([0, 0, 1, 1]))
    assert targets.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_one_segment_does_not_mark_final_frame() -> None:
    targets = generate_boundary_targets([2, 2, 2])
    assert targets.tolist() == [1.0, 0.0, 0.0]


def test_padding_is_zero_and_transitions_are_aligned() -> None:
    targets = generate_boundary_targets(torch.tensor([0, 1, 1, -100, -100]), valid_length=3)
    assert targets.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_segment_rows_can_generate_targets() -> None:
    targets = boundary_targets_from_segments(
        [
            {"start_frame": 0, "end_frame": 1, "label": 0},
            {"start_frame": 2, "end_frame": 3, "label": 1},
        ],
        4,
    )
    assert targets.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_real_p1_segments_match_targets() -> None:
    mapping = load_label_mapping(ROOT / "configs/labels_multitask_release.yaml")
    demo = DATA / "train/pour/p1"
    timestamps = load_timestamp_vector(demo / "citr_features.csv")
    labels, rows = convert_segments_to_frame_labels(demo / "segments.csv", timestamps, mapping)
    targets = generate_boundary_targets(torch.from_numpy(labels))
    expected = [0] + (torch.where(torch.from_numpy(labels[1:]) != torch.from_numpy(labels[:-1]))[0] + 1).tolist()
    assert torch.where(targets > 0)[0].tolist() == expected
    assert len(rows) == 8
