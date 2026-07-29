from __future__ import annotations

import csv

from scripts import run_round28_short_fragment_consolidation as r28


def seg(start: int, end: int, label: str) -> dict:
    return {"start": start, "end": end, "duration": end - start, "top1_label": label, "top1_id": 0, "top1_probability": 0.9}


def test_same_label_merge_requires_a_short_side() -> None:
    merged, _ = r28.same_label_merge([seg(0, 200, "reach"), seg(200, 400, "reach")], 180, "B")
    assert len(merged) == 2
    merged, _ = r28.same_label_merge([seg(0, 100, "reach"), seg(100, 400, "reach")], 180, "B")
    assert len(merged) == 1 and merged[0]["duration"] == 400


def test_iterative_same_label_merge_stabilizes() -> None:
    merged, ops = r28.same_label_merge([seg(0, 50, "reach"), seg(50, 100, "reach"), seg(100, 300, "reach")], 180, "B")
    assert len(merged) == 1 and merged[0]["duration"] == 300 and len(ops) == 2


def test_h3_cannot_absorb_long_middle_interval() -> None:
    a, s, b = seg(0, 200, "reach"), seg(200, 350, "grasp"), seg(350, 550, "reach")
    cfg = {"weights": {"asb": 1, "duration": 1, "pattern": 1, "boundary": 1, "long": 2, "complexity": .5}, "max_merge": 100, "margin": 0, "second_margin": 0, "order": "left_to_right", "iterations": 1}
    out, accepted, _ = r28.local_hypotheses([a, s, b], 180, cfg)
    assert len(out) == 3 and not accepted


def test_round28_outputs_and_frozen_baseline() -> None:
    output = r28.OUT
    rows = list(csv.DictReader((output / "condition_comparison.csv").open()))
    pooled = {x["condition"]: x for x in rows if x["scope"] == "all"}
    assert abs(float(pooled["A"]["temporal_f1@0.50"]) - 0.807988) < 1e-5
    assert float(pooled["A"]["mean_matched_iou"]) > 0.8008
    assert (output / "decision_criteria.csv").is_file()
    assert len(list((output / "figures").glob("timeline_*.png"))) == 36
