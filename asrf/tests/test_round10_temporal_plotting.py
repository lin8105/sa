from pathlib import Path

import numpy as np
import pytest

from asrf.visualization.temporal import plot_temporal_comparison, validate_prediction_segments


ONTOLOGY = {"reach", "grasp", "lift", "transport", "place", "release", "retreat"}


def _segment(label: str, start: int, end: int) -> dict:
    return {"segment_index": 0, "start_frame": start, "end_frame": end, "predicted_label": label}


def test_prediction_rows_reject_novel_labels() -> None:
    with pytest.raises(AssertionError, match="outside ontology"):
        validate_prediction_segments([_segment("pour", 0, 10)], ONTOLOGY)


def test_prediction_rows_reject_gt_matching_schema() -> None:
    row = _segment("reach", 0, 10)
    row["novel_skill"] = "pour"
    with pytest.raises(AssertionError, match="GT matching"):
        validate_prediction_segments([row], ONTOLOGY)


def test_heatmap_width_must_match_timeline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="width"):
        plot_temporal_comparison(
            np.zeros((3, 4, 9)), list(range(10)),
            [{"start_frame": 0, "end_frame": 10, "gt_label": "pour"}],
            [_segment("reach", 0, 10)], [_segment("reach", 0, 10)],
            output_path=tmp_path / "bad.png", ontology=ONTOLOGY, title="bad",
        )


def test_prediction_segment_source_is_distinct_from_match_source() -> None:
    prediction = _segment("reach", 0, 10)
    match = {"novel_skill": "pour", "gt_start_frame": 0, "best_iou": 0.5, "fragment_count": 1}
    assert "novel_skill" not in prediction
    assert "start_frame" not in match


def test_canonical_plot_has_heatmap_truth_raw_asrf_order(tmp_path: Path) -> None:
    output = tmp_path / "comparison.png"
    plot_temporal_comparison(
        np.random.default_rng(42).random((3, 4, 20)), np.arange(20),
        [{"start_frame": 0, "end_frame": 10, "gt_label": "pour"}, {"start_frame": 10, "end_frame": 20, "gt_label": "place"}],
        [_segment("reach", 0, 10), {"segment_index": 1, "start_frame": 10, "end_frame": 20, "predicted_label": "transport"}],
        [_segment("grasp", 0, 12), {"segment_index": 1, "start_frame": 12, "end_frame": 20, "predicted_label": "lift"}],
        brb_probability=np.zeros(20), raw_change_points=[10], asrf_boundary_peaks=[12],
        output_path=output, ontology=ONTOLOGY, title="canonical",
    )
    assert output.is_file()
