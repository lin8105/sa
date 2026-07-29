from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts import run_round27_pp_only_r5_region_sf_point_hybrid as round27


def test_frozen_checkpoint_hashes_and_pp_split() -> None:
    assert round27.sha256(round27.SF) == round27.SF_SHA
    assert round27.sha256(round27.R5) == round27.R5_SHA
    assert round27.read_entries(round27.TRAIN_MANIFEST) == [f"train/pick and place/pp{i}" for i in range(1, 11)]
    assert round27.read_entries(round27.VAL_MANIFEST) == [f"train/pick and place/pp{i}" for i in range(11, 21)]


def test_regions_bridge_only_selected_gap() -> None:
    probability = np.array([0.8, 0.8, 0.1, 0.8, 0.8, 0.1, 0.1, 0.8])
    assert round27.regions(probability, 0.5, 0) == [(0, 2), (3, 5), (7, 8)]
    assert round27.regions(probability, 0.5, 1) == [(0, 5), (7, 8)]


def test_hybrid_has_at_most_one_point_per_region() -> None:
    sf = {"brb": np.array([0.1, 0.4, 0.9, 0.3, 0.8, 0.1])}
    r5 = {"brb": np.array([0.1, 0.8, 0.7, 0.8, 0.1, 0.8])}
    points, diagnostics, suppressed = round27.hybrid(sf, r5, {"threshold": 0.5, "gap": 0, "rule": "P0", "support_gate": None, "separation": 0})
    assert points == [2, 5]
    assert len(diagnostics) == 2
    assert suppressed == 0


def test_required_round27_outputs_exist() -> None:
    output = round27.OUT
    assert (output / "report.md").is_file()
    assert (output / "config.yaml").is_file()
    assert (output / "validation_fusion_selection.csv").is_file()
    assert len(list((output / "predictions").glob("*.json"))) == 10
    assert len(list((output / "figures").glob("timeline_*.png"))) == 10
    assert len(list((output / "figures").glob("summary_*.png"))) == 5
    manifest = list(csv.DictReader((output / "test_manifest.csv").open()))
    assert sum(row["included"] == "1" for row in manifest) == 10
    assert any(row["trajectory"] == "test/wipe/w4" and row["included"] == "0" for row in manifest)
    config = (output / "config.yaml").read_text()
    assert "no_round25_refinement: true" in config
    assert "no_segment_classifier: true" in config
    metadata = json.loads((output / "checkpoint_hashes.json").read_text())
    assert metadata["sf_sha256"] == round27.SF_SHA
    assert metadata["r5_sha256"] == round27.R5_SHA
