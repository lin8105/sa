import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
import run_round26_sf_vs_r5_round25_refinement as round26  # noqa: E402


ROOT = Path("outputs/round26_sf_vs_r5_round25_refinement")


def test_r5_definition_is_hard_window_radius_five_and_checkpoint_is_verified():
    definition = json.loads((ROOT / "r5_definition.json").read_text())
    assert definition["window"]["radius_frames"] == 5
    assert definition["window"]["nominal_frames"] == 11
    assert definition["asb_uses_local_window"] is False
    assert definition["brb_uses_local_window"] is True
    assert definition["checkpoint"]["usable"] == 1
    assert definition["checkpoint"]["sha256"] == round26.R5_SHA


def test_round25_parameters_and_primary_conditions_are_frozen():
    audit = json.loads((ROOT / "round25_parameter_audit.json").read_text())
    assert audit["selected_variant"] == "R7"
    assert audit["parameters"]["threshold"] == 180
    assert audit["parameters"]["processing_mode"] == "iterative"
    rows = list(csv.DictReader((ROOT / "condition_comparison.csv").open(newline="")))
    assert [row["condition"] for row in rows] == ["C_sf_round25", "D_r5_round25"]
    assert len(rows) == 2


def test_all_audited_trajectories_have_both_front_end_exports():
    manifest = list(csv.DictReader((ROOT / "trajectory_manifest.csv").open(newline="")))
    predictions = ROOT / "predictions"
    assert len(manifest) == 33
    assert len(list(predictions.glob("*.json"))) == 33
    assert len(list(predictions.glob("*__sf_asrf.npz"))) == 33
    assert len(list(predictions.glob("*__r5_asrf.npz"))) == 33


def test_novel_boundary_aggregate_is_present_for_both_conditions():
    rows = list(csv.DictReader((ROOT / "novel_boundary_metrics.csv").open(newline="")))
    selected = {(row["condition"], row["stage"], row["trajectory"], row["tolerance_frames"]) for row in rows}
    assert ("C_sf_round25", "refined", "aggregate", "33") in selected
    assert ("D_r5_round25", "refined", "aggregate", "33") in selected
