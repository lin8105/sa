from __future__ import annotations

import csv
from pathlib import Path

from scripts import run_round27b_complete_test_temporal_only as r27b


def test_temporal_matching_ignores_semantic_labels() -> None:
    pred = [{"start": 0, "end": 10, "top1_label": "reach"}]
    gt = [{"start": 0, "end": 10, "label": "unscrew"}]
    matches = r27b.temporal_matches(pred, gt)
    assert len(matches) == 1 and matches[0]["iou"] == 1.0


def test_temporal_matching_is_one_to_one() -> None:
    pred = [{"start": 0, "end": 10}, {"start": 0, "end": 10}]
    gt = [{"start": 0, "end": 10}]
    assert len(r27b.temporal_matches(pred, gt)) == 1


def test_boundary_matching_is_one_to_one_and_uses_matched_error_only() -> None:
    pairs, fp, fn = r27b.boundary_pairs([10, 11], [10], 2)
    assert pairs == [(0, 0, 0)]
    assert fp == [1] and fn == []


def test_temporal_rates_and_pooled_count_aggregation() -> None:
    row = r27b.temporal_row("trajectory", "x", "x/1", [{"start": 0, "end": 10}], [{"start": 0, "end": 10}], [{"pred_index": 0, "gt_index": 0, "iou": 1.0}])
    assert row["temporal_precision@0.50"] == 1.0
    assert row["temporal_f1@0.50"] == 1.0
    aggregate = r27b.aggregate_temporal([row], "pooled")
    assert aggregate["temporal_f1@0.50"] == 1.0


def test_complete_round27b_outputs() -> None:
    output = r27b.OUT
    inventory = list(csv.DictReader((output / "complete_test_inventory.csv").open()))
    assert len(inventory) == 36
    assert sum(row["included"] == "1" for row in inventory) == 36
    assert len(list((output / "figures").glob("timeline_*.png"))) == 36
    assert len(list((output / "predictions").glob("*.json"))) == 36
    assert (output / "temporal_only_results.csv").is_file()
    assert (output / "boundary_results.csv").is_file()
