import sys

import numpy as np

sys.path.insert(0, "scripts")
import run_round21_asb_assisted_boundary_merge as round21  # noqa: E402


def _record(labels):
    probabilities = np.zeros((len(round21.ASB_LABELS), len(labels)), dtype=np.float32)
    for index, label in enumerate(labels):
        probabilities[int(label), index] = 1.0
    return {"length": len(labels), "asb_probabilities": probabilities}


def test_js_divergence_is_symmetric_and_nonnegative():
    left = np.array([0.8, 0.2], dtype=np.float32)
    right = np.array([0.3, 0.7], dtype=np.float32)
    assert round21.js_divergence(left, right) >= 0.0
    assert np.isclose(round21.js_divergence(left, right), round21.js_divergence(right, left))


def test_asb_summary_reports_majority_and_transitions():
    record = _record([0, 0, 1, 0])
    summary = round21.asb_summary(record, 0, 4)
    assert summary["asb_majority_label"] == round21.ASB_LABELS[0]
    assert summary["asb_majority_ratio"] == 0.75
    assert summary["asb_transition_count"] == 2
    assert summary["asb_longest_majority_run"] == 2


def test_boundary_classification_distinguishes_true_and_internal_boundaries():
    record = {"gt": [{"start": 0, "end": 100, "label": "reach"}, {"start": 100, "end": 200, "label": "transport"}]}
    assert round21.classify_boundary(record, 100) == "true_boundary"
    assert round21.classify_boundary(record, 60) == "false_internal_boundary"


def test_asb_summary_handles_empty_window_deterministically():
    record = _record([2, 2, 2])
    summary = round21.asb_summary(record, 3, 3)
    assert summary["duration"] == 0
    assert summary["asb_majority_label"] == round21.ASB_LABELS[2]
