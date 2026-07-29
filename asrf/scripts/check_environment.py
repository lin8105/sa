#!/usr/bin/env python3
"""Read-only ASRF environment and external-data smoke check."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import asrf  # noqa: E402
from asrf.data.dataset import TrajectoryDataset  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.utils.config import load_yaml_config, resolve_repo_path  # noqa: E402


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, int, int]]:
    snapshot: dict[str, tuple[int, int, int, int]] = {}
    for path in root.rglob("*"):
        stat = path.stat()
        snapshot[str(path.relative_to(root))] = (
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_mode,
        )
    return snapshot


def main() -> int:
    config_path = REPO_ROOT / "configs/pour.yaml"
    config = load_yaml_config(config_path)
    paths = config["paths"]
    train_root = Path(paths["train_dataset_root"])
    test_root = Path(paths["test_dataset_root"])
    labels_path = resolve_repo_path(paths["label_mapping"])
    train_split = resolve_repo_path(paths["train_split"])
    test_split = resolve_repo_path(paths["test_split"])

    for required in (train_root, test_root, labels_path, train_split, test_split):
        if not required.exists():
            raise FileNotFoundError(required)

    label_mapping = load_label_mapping(labels_path)
    train_before = _tree_snapshot(train_root / "p1")
    test_before = _tree_snapshot(test_root / "p1")
    train = TrajectoryDataset(train_root, train_split, labels_path, expected_height=88)
    test = TrajectoryDataset(test_root, test_split, labels_path, expected_height=88)
    train_sample = train[0]
    test_sample = test[0]
    train_after = _tree_snapshot(train_root / "p1")
    test_after = _tree_snapshot(test_root / "p1")

    for sample, name in ((train_sample, "train"), (test_sample, "test")):
        heatmap = sample["heatmap"]
        if heatmap.ndim != 3 or tuple(heatmap.shape[:2]) != (3, 88):
            raise AssertionError(f"{name}: unexpected heatmap shape {tuple(heatmap.shape)}")
        with Image.open(sample["demonstration_path"] / "citr_fingerprint_pure.png") as image:
            source_width = image.width
        if heatmap.shape[-1] != source_width:
            raise AssertionError(f"{name}: temporal width changed from {source_width}")
        if len(sample["segments"]) == 0:
            raise AssertionError(f"{name}: segments.csv was not parsed")

    if train_before != train_after or test_before != test_after:
        raise AssertionError("External dataset metadata changed during the read-only check.")

    print(f"asrf={asrf.__version__}")
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"train_sample={train_sample['trajectory_id']} shape={tuple(train_sample['heatmap'].shape)}")
    print(f"test_sample={test_sample['trajectory_id']} shape={tuple(test_sample['heatmap'].shape)}")
    print(f"labels={dict(label_mapping)} aliases={label_mapping.aliases}")
    print("external_dataset_write_check=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

