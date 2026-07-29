from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch

from asrf.probes.oracle_segment_dataset import OracleSegmentRecord, crop_segment_heatmap


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_manual_correction_verification_passes() -> None:
    rows = _rows(ROOT / "outputs/round6_diagnostics/manual_correction_verification.csv")
    assert len(rows) == 16
    assert all(row["pick_count"] == "0" and row["translation_count"] == "0" for row in rows)
    assert all(row["status"] == "valid" and row["coverage_valid"] == "True" for row in rows)
    assert all("tranport" not in row.values() for row in rows)


def test_w4_is_stable_and_w1_w3_are_preserved() -> None:
    first = json.loads((ROOT / "outputs/round6_diagnostics/w4_scan_1.json").read_text())
    second = json.loads((ROOT / "outputs/round6_diagnostics/w4_scan_2.json").read_text())
    stability = json.loads((ROOT / "outputs/round6_diagnostics/w4_stability.json").read_text())
    assert first == second
    assert stability["identical"] is True
    assert stability["valid"] is True
    assert json.loads((ROOT / "outputs/multitask_baseline/test/wipe/w4/evaluation_round6.json").read_text())["status"] == "evaluated"
    assert json.loads((ROOT / "outputs/multitask_baseline/test/wipe_summary.json").read_text())["trajectory_count"] == 3
    assert (ROOT / "outputs/multitask_baseline/test/wipe_summary_w1_w4.json").is_file()


def test_current_prediction_exports_use_canonical_names_only() -> None:
    aliases = {"pick", "translation", "tranport"}
    for path in (ROOT / "outputs/multitask_baseline/test").rglob("*.csv"):
        if path.name in {"ground_truth.csv", "raw_asb_predictions.csv", "official_asrf_predictions.csv", "calibrated_asrf_predictions.csv", "oracle_boundary_predictions.csv"}:
            text = path.read_text(encoding="utf-8").lower()
            assert not any(f",{alias}\n" in text for alias in aliases), path


def test_oracle_crop_is_independent_and_does_not_cross_boundaries() -> None:
    heatmap = torch.arange(3 * 88 * 20, dtype=torch.float32).reshape(3, 88, 20)
    record = OracleSegmentRecord("test", "pour", "p1", "/external/read-only", 1, 5, 10, 3, "transport", 5)
    crop = crop_segment_heatmap(heatmap, record)
    assert tuple(crop.shape) == (3, 88, 5)
    assert torch.equal(crop, heatmap[:, :, 5:10])


def test_per_skill_primary_rows_have_required_support_and_rate() -> None:
    rows = [row for row in _rows(ROOT / "outputs/skill_segment_probe/per_skill_segment_recognition.csv") if row["selection_role"] == "primary"]
    assert len(rows) == 27
    assert {row["probe_type"] for row in rows} == {"raw_citr", "heatmap_encoder", "shared_features"}
    assert {row["skill"] for row in rows} == {"reach", "grasp", "lift", "transport", "pour", "pour_recover", "place", "wipe", "retreat"}
    for row in rows:
        assert int(row["test_segments"]) == int(row["support"])
        assert int(row["correct_test_segments"]) == int(row["correct"])
        assert abs(float(row["segment_recognition_rate"]) - int(row["correct_test_segments"]) / int(row["test_segments"])) < 1e-12


def test_every_test_segment_is_exported_once_per_probe_and_duration_is_separate() -> None:
    rows = _rows(ROOT / "outputs/skill_segment_probe/test_segment_predictions.csv")
    assert len(rows) == 96 * 6
    keys = [(row["probe_type"], row["uses_duration"], row["task"], row["trajectory_id"], row["segment_index"]) for row in rows]
    assert len(keys) == len(set(keys))
    assert {row["uses_duration"] for row in rows} == {"False", "True"}
    assert all(row["ground_truth_skill"] not in {"pick", "translation"} for row in rows)


def test_frozen_checkpoint_hashes_remain_unchanged() -> None:
    expected = {
        ROOT / "outputs/multitask_baseline/best.pt": "ad557bc5b10bc00d1582c3a1d82897e81173f6abc83dfc2220a2fb96ee2c0241",
        ROOT / "outputs/pour_baseline/best.pt": "586fc50c91c735f7212c16baa052f43655b3140408aa3c0d534d11daa1fbc358",
    }
    for path, digest in expected.items():
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        assert hasher.hexdigest() == digest
