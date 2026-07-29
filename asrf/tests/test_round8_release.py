from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml

from asrf.data.boundary_targets import generate_boundary_targets
from asrf.data.labels import load_label_mapping
from asrf.evaluation.metrics import boundary_counts


ROOT = Path(__file__).resolve().parents[1]


def test_round12_canonical_classes_and_insert_index() -> None:
    mapping = load_label_mapping(ROOT / "configs/labels_multitask_release.yaml")
    assert dict(mapping) == {"reach": 0, "grasp": 1, "lift": 2, "transport": 3, "pour": 4, "pour_recover": 5, "place": 6, "release": 7, "wipe": 8, "retreat": 9, "insert": 10}
    assert mapping.aliases == {"pull_out": "lift", "extract": "lift"}


def test_nine_class_config_and_protected_checkpoints_are_unchanged() -> None:
    old = yaml.safe_load((ROOT / "configs/labels_multitask.yaml").read_text())
    assert len(old["labels"]) == 9 and "release" not in old["labels"]
    expected = {
        "outputs/multitask_baseline/best.pt": "ad557bc5b10bc00d1582c3a1d82897e81173f6abc83dfc2220a2fb96ee2c0241",
        "outputs/pour_baseline/best.pt": "586fc50c91c735f7212c16baa052f43655b3140408aa3c0d534d11daa1fbc358",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_single_frame_mode_reproduces_historical_target_exactly() -> None:
    labels = [0, 0, 1, 1, 2]
    expected = [1.0, 0.0, 1.0, 0.0, 1.0]
    assert generate_boundary_targets(labels).tolist() == expected
    assert generate_boundary_targets(labels, boundary_target_mode="single_frame", boundary_window_radius=99, boundary_gaussian_sigma=9.0).tolist() == expected


def test_hard_window_radius_and_clipping() -> None:
    targets = generate_boundary_targets([0, 0, 1, 1, 1, 2], boundary_target_mode="hard_window", boundary_window_radius=1)
    assert targets.tolist() == [1.0] * 6
    clipped = generate_boundary_targets([0, 0, 1], boundary_target_mode="hard_window", boundary_window_radius=20, boundary_include_frame_zero=False)
    assert clipped.tolist() == [1.0, 1.0, 1.0]


def test_overlapping_hard_windows_are_deterministic() -> None:
    first = generate_boundary_targets([0, 1, 2], boundary_target_mode="hard_window", boundary_window_radius=1)
    second = generate_boundary_targets([0, 1, 2], boundary_target_mode="hard_window", boundary_window_radius=1)
    assert torch.equal(first, second) and torch.all(first == 1)


def test_gaussian_center_range_overlap_and_sigma() -> None:
    narrow = generate_boundary_targets([0, 0, 1, 1, 1], boundary_target_mode="gaussian", boundary_gaussian_sigma=1.0)
    wide = generate_boundary_targets([0, 0, 1, 1, 1], boundary_target_mode="gaussian", boundary_gaussian_sigma=2.0)
    assert narrow[0].item() == 1.0 and narrow[2].item() == 1.0
    assert torch.all((narrow >= 0) & (narrow <= 1)) and torch.all((wide >= 0) & (wide <= 1))
    assert wide[1] > narrow[1]
    assert torch.all(generate_boundary_targets([0, 1, 2], boundary_target_mode="gaussian", boundary_gaussian_sigma=1.0) <= 1)


def test_frame_zero_and_final_frame_are_configurable() -> None:
    default = generate_boundary_targets([0, 0, 0])
    final = generate_boundary_targets([0, 0, 0], boundary_include_frame_zero=False, boundary_include_final_frame=True)
    assert default.tolist() == [1.0, 0.0, 0.0]
    assert final.tolist() == [0.0, 0.0, 1.0]


def test_round8_audit_and_split_outputs_pass() -> None:
    audit = json.loads((ROOT / "outputs/brb_release_round8/data_audit_summary.json").read_text())
    split = json.loads((ROOT / "outputs/brb_release_round8/split_integrity.json").read_text())
    assert audit["pass"] and audit["scan_rows_identical"]
    assert split["pass"] and split["w4_occurrences"] == 1


def test_round8_configs_are_random_init_and_not_test_selected() -> None:
    for path in sorted((ROOT / "configs/brb_release_round8").glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        assert config["experiment"]["seed"] == 42
        assert config["training"]["initialize_from_checkpoint"] is None
        assert config["training"]["best_metric"] == "val_total_loss"
        assert "test" not in config["data"]["train_split"] and "test" not in config["data"]["val_split"]
        assert config["data"]["num_classes"] == 10


def test_one_to_one_boundary_matching_and_raw_oracle_contract() -> None:
    result = boundary_counts([10, 11, 12], [11], 2, include_frame0=False)
    assert result["tp"] == 1 and result["fp"] == 2 and result["fn"] == 0
    doc = (ROOT / "docs/current_asb_brb_pipeline.md").read_text()
    assert "Raw ASB" in doc and "does not use BRB" in doc and "ground-truth" in doc
