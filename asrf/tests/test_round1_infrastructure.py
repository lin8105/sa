from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path

import numpy as np
from PIL import Image

from asrf.data.annotations import load_segments_csv
from asrf.data.dataset import TrajectoryDataset, load_heatmap
from asrf.data.labels import load_label_mapping, normalize_label_name
from asrf.utils.config import PROJECT_ROOT, load_yaml_config, resolve_repo_path


ASRF_ROOT = Path(__file__).resolve().parents[1]
MSTCN_ROOT = ASRF_ROOT.parent / "mstcn"
DATA_ROOT = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
TRAIN_ROOT = DATA_ROOT / "train/pour"
TEST_ROOT = DATA_ROOT / "test/pour"
EXPECTED_CHECKPOINT_SHA256 = "0c70426fd58bc164494e39e61c1ffc3ca011b574db02e3898938568a090c4f56"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root: Path) -> dict[str, tuple[int, int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino)
        for path in root.rglob("*")
    }


def test_import_asrf_succeeds() -> None:
    module = importlib.import_module("asrf")
    assert module.__name__ == "asrf"


def test_config_and_paths_load() -> None:
    config = load_yaml_config("configs/pour.yaml")
    assert PROJECT_ROOT == ASRF_ROOT
    assert config["project_name"] == "ASRF"
    assert Path(config["paths"]["train_dataset_root"]) == TRAIN_ROOT
    assert Path(config["paths"]["test_dataset_root"]) == TEST_ROOT
    for relative in (
        "configs/labels_pour.yaml",
        "splits/pour_train.txt",
        "splits/pour_val.txt",
        "splits/pour_test.txt",
        "splits/pour_test_p3_p5.txt",
    ):
        assert resolve_repo_path(relative).is_file()


def test_copied_split_hashes_match_mstcn() -> None:
    for name in ("pour_train.txt", "pour_val.txt", "pour_test.txt", "pour_test_p3_p5.txt"):
        assert sha256(ASRF_ROOT / "splits" / name) == sha256(MSTCN_ROOT / "splits" / name)


def test_copied_label_hash_matches_mstcn() -> None:
    assert sha256(ASRF_ROOT / "configs/labels_pour.yaml") == sha256(
        MSTCN_ROOT / "configs/labels_pour.yaml"
    )


def test_external_roots_and_samples_resolve() -> None:
    assert TRAIN_ROOT.is_dir()
    assert TEST_ROOT.is_dir()
    labels_path = ASRF_ROOT / "configs/labels_multitask_release.yaml"
    train = TrajectoryDataset(TRAIN_ROOT, ASRF_ROOT / "splits/pour_train.txt", labels_path)
    test = TrajectoryDataset(TEST_ROOT, ASRF_ROOT / "splits/pour_test.txt", labels_path)
    assert len(train) > 0 and len(test) > 0
    assert (train[0]["demonstration_path"] / "segments.csv").is_file()
    assert (test[0]["demonstration_path"] / "segments.csv").is_file()


def test_heatmap_shape_and_temporal_width_are_preserved() -> None:
    image_path = TRAIN_ROOT / "p1/citr_fingerprint_pure.png"
    with Image.open(image_path) as image:
        source_width, source_height = image.size
    heatmap = load_heatmap(image_path)
    assert tuple(heatmap.shape[:2]) == (3, 88)
    assert heatmap.shape[-1] == source_width
    assert source_height == 88


def test_segments_csv_is_read_and_aliases_resolve() -> None:
    annotation_format, rows = load_segments_csv(TRAIN_ROOT / "p1/segments.csv")
    assert annotation_format == "timestamp"
    assert rows[0]["label"] == "reach"
    labels = load_label_mapping(ASRF_ROOT / "configs/labels_pour.yaml")
    assert normalize_label_name("pick", labels) == "reach"
    assert normalize_label_name("translation", labels) == "transport"


def test_canonical_class_ids_are_correct() -> None:
    labels = load_label_mapping(ASRF_ROOT / "configs/labels_pour.yaml")
    assert dict(labels) == {
        "reach": 0,
        "grasp": 1,
        "lift": 2,
        "transport": 3,
        "pour": 4,
        "pour_recover": 5,
        "place": 6,
    }


def test_reading_samples_does_not_write_external_dataset() -> None:
    labels_path = ASRF_ROOT / "configs/labels_multitask_release.yaml"
    train_demo = TRAIN_ROOT / "p1"
    test_demo = TEST_ROOT / "p1"
    before = snapshot(train_demo) | {f"test/{key}": value for key, value in snapshot(test_demo).items()}
    TrajectoryDataset(TRAIN_ROOT, ASRF_ROOT / "splits/pour_train.txt", labels_path)[0]
    TrajectoryDataset(TEST_ROOT, ASRF_ROOT / "splits/pour_test.txt", labels_path)[0]
    after = snapshot(train_demo) | {f"test/{key}": value for key, value in snapshot(test_demo).items()}
    assert after == before


def test_mstcn_checkpoint_hash_remains_expected() -> None:
    checkpoint = MSTCN_ROOT / "outputs/pour_generalization/best.pt"
    assert sha256(checkpoint) == EXPECTED_CHECKPOINT_SHA256


def test_asrf_has_no_dynamic_mstcn_runtime_import() -> None:
    source_files = list((ASRF_ROOT / "src/asrf").rglob("*.py")) + list((ASRF_ROOT / "scripts").rglob("*.py"))
    for path in source_files:
        text = path.read_text(encoding="utf-8")
        assert "importlib" not in text
        assert "mstcn" not in text.lower()
