import sys

sys.path.insert(0, "scripts")
import run_round20_semantic_fragment_merge as round20  # noqa: E402


def _prediction(start, end, label="transport", confidence=0.8, margin=0.2, support=0.1):
    return {
        "start": start,
        "end": end,
        "duration": end - start,
        "top1_label": label,
        "top2_label": "place",
        "top1_probability": confidence,
        "top2_probability": confidence - margin,
        "margin": margin,
        "embedding": [1.0, 0.0],
        "embedding_support_distance": support,
        "predicted_class_duration_percentile": 0.5,
        "duration_valid": 1,
    }


def _candidate(brb=0.2, same=1, score=0.6):
    left = _prediction(0, 10)
    right = _prediction(10, 20)
    merged = _prediction(0, 20, confidence=0.79, margin=0.18)
    return {
        "boundary_brb": brb,
        "same_label": same,
        "merged_label_agrees": 1,
        "semantic_consistency": 1,
        "duration_valid": 1,
        "support_gain": 0.05,
        "margin_gain": -0.02,
        "merged_confidence_minus_left": -0.01,
        "merged_confidence_minus_right": -0.01,
        "score": score,
        "left": left,
        "right": right,
        "merged": merged,
    }


def test_same_label_boundary_deletion_requires_weak_brb():
    config = {"brb_threshold": 0.35, "score_threshold": 0.1, "short_duration": 60, "confidence_tolerance": 0.1, "margin_tolerance": 0.1}
    accepted, _ = round20.score_accepts(_candidate(brb=0.2), "R1_same_label", config)
    rejected, _ = round20.score_accepts(_candidate(brb=0.8), "R1_same_label", config)
    assert accepted is True
    assert rejected is False


def test_boundary_operation_only_deletes_and_merges_adjacent_segments():
    segments = [_prediction(0, 10), _prediction(10, 20), _prediction(20, 30)]
    merged = _prediction(0, 20)
    operation = {"kind": "boundary", "left_index": 0, "right_index": 1, "candidate": {"merged": merged}}
    result = round20.apply_operation(segments, operation)
    assert [(x["start"], x["end"]) for x in result] == [(0, 20), (20, 30)]


def test_round20_output_reuses_round19_raw_metrics_exactly():
    import json
    from pathlib import Path

    payload = json.loads(Path("outputs/round20_semantic_fragment_merge/raw_reproduction_metrics.json").read_text())
    assert payload["exact_artifact_reuse"] is True
    assert all(abs(float(value)) < 1e-9 for value in payload["deltas"].values())

