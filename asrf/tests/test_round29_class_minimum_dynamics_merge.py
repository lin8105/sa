from __future__ import annotations

import inspect

import numpy as np

from scripts import run_round29_class_minimum_dynamics_merge as r29


def stats() -> dict:
    return {
        "channels": {"median": [0.0, 0.0, 0.0], "iqr": [1.0, 1.0, 1.0]},
        "gripper_position": {"median": 0.0, "iqr": 1.0},
    }


def segment_fixture(threshold: float = 1.0, length: int = 20):
    heat = np.zeros((3, 88, length), dtype=np.float32)
    grip = np.zeros(length, dtype=np.float32)
    labels = np.zeros(length, dtype=np.int64)
    probs = np.zeros((7, length), dtype=np.float32); probs[0] = 1.0
    segments = [r29.make_segment(0, length, labels, probs, heat, grip, stats(), {"reach": threshold, **{x: 0.0 for x in r29.KNOWN if x != "reach"}}, 0)]
    return segments, labels, probs, heat, grip


def test_below_threshold_cannot_remain_unchanged_when_a_neighbor_exists() -> None:
    heat = np.zeros((3, 88, 40), dtype=np.float32); grip = np.zeros(40); labels = np.zeros(40, dtype=np.int64); probs = np.zeros((7, 40)); probs[0] = 1.0
    thresholds = {x: 0.0 for x in r29.KNOWN}; thresholds["reach"] = 1.0
    raw = [r29.make_segment(0, 20, labels, probs, heat, grip, stats(), thresholds, 0), r29.make_segment(20, 40, labels, probs, heat, grip, stats(), thresholds, 1)]
    final, operations, _ = r29.force_merges(raw, labels, probs, heat, grip, stats(), thresholds)
    assert len(operations) >= 1 and len(final) < len(raw)


def test_segments_longer_than_180_frames_can_be_forced_to_merge() -> None:
    heat = np.zeros((3, 88, 420), dtype=np.float32); grip = np.zeros(420); labels = np.zeros(420, dtype=np.int64); probs = np.zeros((7, 420)); probs[0] = 1.0
    thresholds = {x: 0.0 for x in r29.KNOWN}; thresholds["reach"] = 1.0
    raw = [r29.make_segment(0, 210, labels, probs, heat, grip, stats(), thresholds, 0), r29.make_segment(210, 420, labels, probs, heat, grip, stats(), thresholds, 1)]
    _, operations, _ = r29.force_merges(raw, labels, probs, heat, grip, stats(), thresholds)
    assert operations and operations[0]["source_segment"]["duration"] > 180


def test_predicted_class_selects_threshold_not_gt_class() -> None:
    heat = np.zeros((3, 88, 20), dtype=np.float32); grip = np.zeros(20); labels = np.zeros(20, dtype=np.int64); probs = np.zeros((7, 20)); probs[0] = 1.0
    thresholds = {x: 0.0 for x in r29.KNOWN}; thresholds["reach"] = 1.0; thresholds["grasp"] = 0.0
    row = r29.make_segment(0, 20, labels, probs, heat, grip, stats(), thresholds, 0)
    assert row["top1_label"] == "reach" and row["class_threshold"] == 1.0 and row["invalid"] == 1


def test_merge_recomputes_dynamics_and_predicted_label() -> None:
    heat = np.zeros((3, 88, 20), dtype=np.float32); grip = np.zeros(20); labels = np.r_[np.zeros(10, dtype=np.int64), np.ones(10, dtype=np.int64)]; probs = np.zeros((7, 20)); probs[0, :10] = 1.0; probs[1, 10:] = 1.0
    thresholds = {x: 0.0 for x in r29.KNOWN}; thresholds["reach"] = 1.0
    merged = r29.merge_pair(r29.make_segment(0, 10, labels, probs, heat, grip, stats(), thresholds, 0), r29.make_segment(10, 20, labels, probs, heat, grip, stats(), thresholds, 1), labels, probs, heat, grip, stats(), thresholds, 0)
    assert merged["duration"] == 20 and merged["top1_label"] == "reach" and "S4" in merged


def test_literal_minimum_threshold_is_exact() -> None:
    rows = [{"class": label, "S4": float(i + 1), "trajectory": "t", "start": 0, "end": 1, "duration_frames": 1, "d1": 0, "dlag5": 0, "dlag10": 0, "dlag20": 0, "dlag50": 0, "d_phase": 0, "robust_range": 0, "gripper_dynamics": 0} for i, label in enumerate(r29.KNOWN)]
    rows += [{**rows[0], "S4": 0.125, "trajectory": "minimum"}]
    mapping, (summary, _) = r29.thresholds(rows)
    assert mapping["reach"] == 0.125 and next(x for x in summary if x["class"] == "reach")["minimum_S4"] == 0.125


def test_gripper_dynamics_contributes_to_grasp_release_score() -> None:
    heat = np.zeros((3, 88, 30), dtype=np.float32); flat = np.zeros(30); changing = np.linspace(0, 1, 30)
    flat_score = r29.segment_score(heat, flat, 0, 30, stats()); changing_score = r29.segment_score(heat, changing, 0, 30, stats())
    assert changing_score["gripper_dynamics"] > flat_score["gripper_dynamics"] and changing_score["S4"] > flat_score["S4"]


def test_deployable_merge_function_has_no_gt_argument() -> None:
    assert "gt" not in inspect.signature(r29.force_merges).parameters


def test_every_merge_reduces_count_by_one() -> None:
    raw, labels, probs, heat, grip = segment_fixture()
    final, operations, _ = r29.force_merges(raw, labels, probs, heat, grip, stats(), {x: 0.0 for x in r29.KNOWN} | {"reach": 1.0})
    assert len(raw) - len(final) == len(operations)


def test_iterative_processing_terminates() -> None:
    raw, labels, probs, heat, grip = segment_fixture()
    final, _, decisions = r29.force_merges(raw, labels, probs, heat, grip, stats(), {x: 0.0 for x in r29.KNOWN} | {"reach": 1.0}, max_iterations=2)
    assert len(final) >= 1 and isinstance(decisions, list)


def test_temporal_matching_remains_label_independent() -> None:
    pred = [{"start": 0, "end": 10, "top1_label": "reach"}]
    gt = [{"start": 0, "end": 10, "label": "insert"}]
    row, matches = r29.temporal_metrics(pred, gt)
    assert matches and row["temporal_f1@0.50"] == 1.0
