from __future__ import annotations

import csv
from pathlib import Path

from scripts import preflight_round30b_auxiliary_skill_independence as r30b


def test_round27b_inventory_has_exact_final_test_set() -> None:
    assert len(r30b.read_round27_test_set()) == 36


def test_discovered_inventory_has_no_eligible_auxiliary_family() -> None:
    rows, _ = r30b.discover()
    assert rows
    assert not [row for row in rows if row["independence_training_eligible"] == 1]


def test_final_test_families_are_excluded_from_development() -> None:
    rows, final_test = r30b.discover()
    final_families = {r30b.family_of(entry) for entry in final_test}
    assert final_families == {"plug", "pour", "pp", "unscrew", "wipe"}
    for row in rows:
        if row["split"] == "train" and row["family"] in {"plug", "pour", "unscrew", "wipe"}:
            assert row["independence_training_eligible"] == 0
            assert "final-test family" in row["exclusion_reason"]


def test_preflight_outputs_are_explicitly_blocked() -> None:
    report = Path("outputs/round30b_auxiliary_skill_independence/report.md").read_text(encoding="utf-8")
    assert "BLOCKED at preflight" in report
    assert "no training or final-test evaluation was run" in report
    with Path("outputs/round30b_auxiliary_skill_independence/leakage_audit.csv").open(encoding="utf-8", newline="") as handle:
        checks = list(csv.DictReader(handle))
    assert any(row["check"] == "eligible_auxiliary_family_count" and row["status"] == "FAIL" for row in checks)
