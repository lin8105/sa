import sys

import numpy as np

sys.path.insert(0, "scripts")
import run_round19_asrf_segment_classifier_integration as round19  # noqa: E402


def test_round19_hungarian_matching_is_one_to_one():
    predicted = [
        {"start": 0, "end": 10, "top1_label": "reach", "top1_id": 0},
        {"start": 10, "end": 20, "top1_label": "grasp", "top1_id": 1},
    ]
    gt = [
        {"start": 0, "end": 10, "label": "reach", "label_id": 0},
        {"start": 10, "end": 20, "label": "grasp", "label_id": 1},
    ]
    matches = round19.hungarian_matches(predicted, gt)
    assert len(matches) == 2
    assert {(row["pred_index"], row["gt_index"]) for row in matches} == {(0, 0), (1, 1)}


def test_round19_official_peaks_and_segments_cover_sequence():
    brb = np.asarray([0.9, 0.1, 0.8, 0.1, 0.7], dtype=np.float32)
    intervals = round19.raw_segments(brb)
    assert [(x.start, x.end) for x in intervals] == [(0, 2), (2, 5)]


def test_round19_error_categories_include_unmatched_segments():
    predicted = [{"start": 0, "end": 5, "top1_label": "reach", "top1_id": 0, "top1_probability": 0.9, "embedding_norm": 1.0, "duration": 5}]
    gt = [{"start": 0, "end": 5, "label": "reach", "label_id": 0}, {"start": 5, "end": 10, "label": "grasp", "label_id": 1}]
    matched, missed, false, categories = round19.matching_rows("t", "raw_asrf", predicted, gt)
    assert len(matched) == 1
    assert len(missed) == 1
    assert len(false) == 0
    assert categories["F_missed_gt_segment"] == 1
