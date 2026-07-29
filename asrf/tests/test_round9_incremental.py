from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from asrf.data.labels import load_label_mapping
from asrf.models import ASRFModel
from asrf.training.checkpointing import load_checkpoint
from asrf.training.transfer import expand_asrf_state_dict
from asrf.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round9_incremental_learning"
SPLITS = ROOT / "splits/round9_incremental"


def split(name: str) -> list[str]:
    return [line.strip() for line in (SPLITS / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_base_pp10_is_fixed_and_shared() -> None:
    expected = [f"train/pick and place/pp{i}" for i in range(1, 11)]
    assert split("base_pp10.txt") == expected
    for family in ("pour", "wipe", "plug"):
        for size in ("3", "5", "all"):
            assert split(f"{family}_train_{size}_with_base_pp10.txt")[:10] == expected


def test_nested_target_subsets() -> None:
    for family in ("pour", "wipe", "plug"):
        n3 = set(split(f"{family}_train_3.txt")); n5 = set(split(f"{family}_train_5.txt")); all_data = set(split(f"{family}_train_all.txt"))
        assert n3 <= n5 <= all_data
        assert len(n3) == 3
        assert len(n5) >= 3


def test_fixed_primary_tests() -> None:
    assert split("test_pour_primary.txt") == ["test/pour/p1", "test/pour/p2"]
    assert split("test_wipe_primary.txt") == ["test/wipe/w1", "test/wipe/w2"]
    assert split("test_plug_primary.txt") == ["test/plug/p1", "test/plug/p2", "test/plug/p3", "test/plug/po1", "test/plug/po2"]


def test_no_train_validation_test_leakage() -> None:
    base = set(split("base_pp10.txt")); validation = set(split("common_validation.txt"))
    assert not base & validation
    for family in ("pour", "wipe", "plug"):
        training = set(split(f"{family}_train_all_with_base_pp10.txt")); test = set(split(f"test_{family}_primary.txt"))
        assert not training & validation
        assert not training & test
        assert not validation & test


def test_round12_ontology_and_aliases() -> None:
    mapping = load_label_mapping(ROOT / "configs/labels_multitask_plug.yaml")
    assert len(mapping) == 11
    assert mapping["place"] == 6 and mapping["insert"] == 10
    assert mapping.aliases["pull_out"] == "lift"
    assert mapping.aliases["extract"] == "lift"


def test_all_configs_use_hard_window_r5_and_common_initialization() -> None:
    for family in ("pour", "wipe", "plug"):
        for size in ("3", "5", "all"):
            config = load_yaml_config(OUT / "models" / family / f"n{size}" / "config.yaml")
            assert config["data"]["boundary_target_mode"] == "hard_window"
            assert config["data"]["boundary_window_radius"] == 5
            assert config["refinement"]["official_boundary_threshold"] == 0.5
            assert config["training"]["initialize_from_checkpoint"] == "outputs/brb_release_round8/hard_window_r5/best.pt"


def test_transfer_copies_old_rows_and_randomizes_new_rows() -> None:
    config = load_yaml_config(OUT / "models/pour/n3/config.yaml")
    model = ASRFModel.from_config(config)
    old = load_checkpoint(ROOT / "outputs/brb_release_round8/hard_window_r5/best.pt", map_location="cpu")["model_state"]
    expanded, metadata = expand_asrf_state_dict(old, model.state_dict())
    assert metadata["copied_class_rows"] == list(range(10))
    assert metadata["new_class_rows_randomly_initialized"] == [10, 11]
    assert expanded["asb.initial_projection.weight"].shape[0] == 12
    assert expanded["asb.initial_projection.weight"][:10].equal(old["asb.initial_projection.weight"])


def test_same_initialization_hash_recorded_for_all_runs() -> None:
    expected = "61f32711d6de9e8c3809a0c1447459cb754adb31d3a0be8c9a0ba06f9b9c35af"
    for family in ("pour", "wipe", "plug"):
        for size in ("3", "5", "all"):
            metadata = json.loads((OUT / "models" / family / f"n{size}" / "round9_run_metadata.json").read_text())
            assert metadata["initialization_checkpoint_sha256"] == expected


def test_support_and_learning_curve_include_primary_skills() -> None:
    support = list(csv.DictReader((OUT / "training_support.csv").open(encoding="utf-8")))
    curve = list(csv.DictReader((OUT / "per_skill_learning_curve.csv").open(encoding="utf-8")))
    for family, skills in {"pour": {"pour", "pour_recover"}, "wipe": {"wipe"}, "plug": {"place", "insert"}}.items():
        assert skills <= {row["skill"] for row in support if row["target_family"] == family}
        assert skills <= {row["skill"] for row in curve if row["target_family"] == family}


def test_macro_target_skill_f1_calculation() -> None:
    tasks = list(csv.DictReader((OUT / "task_learning_curve.csv").open(encoding="utf-8")))
    skills = list(csv.DictReader((OUT / "per_skill_learning_curve.csv").open(encoding="utf-8")))
    for task in tasks:
        target = {"pour": {"pour", "pour_recover"}, "wipe": {"wipe"}, "plug": {"place", "insert"}}[task["target_family"]]
        values = [float(row["official_F1"]) for row in skills if row["target_family"] == task["target_family"] and row["target_trajectory_count"] == task["target_trajectory_count"] and row["skill"] in target]
        assert abs(float(task["macro_target_skill_F1"]) - sum(values) / len(values)) < 1e-9


def test_prior_round8_checkpoint_hash_is_preserved() -> None:
    digest = hashlib.sha256((ROOT / "outputs/brb_release_round8/hard_window_r5/best.pt").read_bytes()).hexdigest()
    assert digest == "61f32711d6de9e8c3809a0c1447459cb754adb31d3a0be8c9a0ba06f9b9c35af"
