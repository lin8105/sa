from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from asrf.data.boundary_targets import generate_boundary_targets
from asrf.evaluation.metrics import boundary_counts
from asrf.refinement.peaks import select_boundary_peaks


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BASELINE = "ad557bc5b10bc00d1582c3a1d82897e81173f6abc83dfc2220a2fb96ee2c0241"
EXPECTED_POUR = "586fc50c91c735f7212c16baa052f43655b3140408aa3c0d534d11daa1fbc358"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_round7a_single_frame_target_is_unchanged() -> None:
    assert generate_boundary_targets([0, 0, 1, 1, 2]).tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert generate_boundary_targets([0, 0, 1], valid_length=2).tolist() == [1.0, 0.0, 0.0]


def test_round7a_positive_weight_is_configurable() -> None:
    for value in (200, 100, 50, 25):
        config = yaml.safe_load((ROOT / f"configs/brb_ablation_round7a/brb_pw_{value}.yaml").read_text())
        assert config["loss"]["boundary_positive_weighting"] == "fixed"
        assert config["loss"]["boundary_positive_weight"] == float(value)


def test_round7a_configs_change_only_weight_and_output_identity() -> None:
    base = yaml.safe_load((ROOT / "configs/brb_ablation_round7a/brb_pw_200.yaml").read_text())
    for value in (100, 50, 25):
        other = yaml.safe_load((ROOT / f"configs/brb_ablation_round7a/brb_pw_{value}.yaml").read_text())
        assert other["model"] == base["model"]
        assert other["data"] == base["data"]
        assert {k: v for k, v in other["loss"].items() if k != "boundary_positive_weight"} == {k: v for k, v in base["loss"].items() if k != "boundary_positive_weight"}
        assert other["training"] == base["training"]
        assert other["refinement"] == base["refinement"]
        assert other["paths"]["output_dir"] != base["paths"]["output_dir"]


def test_round7a_split_files_are_unchanged_and_w4_once() -> None:
    train = (ROOT / "splits/multitask_train.txt").read_text()
    val = (ROOT / "splits/multitask_val.txt").read_text()
    wipe = [line for line in (ROOT / "splits/multitask_test_wipe.txt").read_text().splitlines() if line]
    assert "test/wipe/w4" in wipe and wipe.count("test/wipe/w4") == 1
    assert not set(train.splitlines()) & set(val.splitlines())
    assert all("pick" not in line or "pick and place" in line for line in train.splitlines() + val.splitlines())


def test_round7a_frozen_checkpoints_are_unchanged() -> None:
    assert _sha(ROOT / "outputs/multitask_baseline/best.pt") == EXPECTED_BASELINE
    assert _sha(ROOT / "outputs/pour_baseline/best.pt") == EXPECTED_POUR


def test_round7a_independent_initialization_and_no_test_selection() -> None:
    for value in (200, 100, 50, 25):
        config = yaml.safe_load((ROOT / f"configs/brb_ablation_round7a/brb_pw_{value}.yaml").read_text())
        assert config["experiment"]["seed"] == 42
        assert config["training"]["initialize_from_checkpoint"] is None
        assert config["training"]["best_metric"] == "val_total_loss"
        assert "test" not in config["data"]["train_split"]
        assert "test" not in config["data"]["val_split"]


def test_round7a_official_threshold_is_present_for_every_model() -> None:
    for path in sorted((ROOT / "configs/brb_ablation_round7a").glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        assert config["refinement"]["official_boundary_threshold"] == 0.5


def test_round7a_one_to_one_matching_and_duplicate_peaks() -> None:
    result = boundary_counts([10, 11, 12], [11], 2, include_frame0=False)
    assert result["tp"] == 1 and result["fp"] == 2 and result["fn"] == 0
    assert select_boundary_peaks([0.0, 0.8, 0.7, 0.0], threshold=0.5) == [0, 1]


def test_round7a_raw_and_oracle_contracts_are_documented() -> None:
    doc = (ROOT / "docs/current_asb_brb_pipeline.md").read_text()
    assert "Raw ASB" in doc and "does not use BRB" in doc
    assert "Oracle refinement" in doc and "ground-truth" in doc


def test_round7a_baseline_manifest_records_hashes() -> None:
    manifest_path = ROOT / "outputs/brb_ablation_round7a/baseline_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["baseline_hashes_match"] is True
        assert manifest["frozen_baseline"]["sha256"] == EXPECTED_BASELINE
        assert manifest["frozen_pour_baseline"]["sha256"] == EXPECTED_POUR
