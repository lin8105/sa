from pathlib import Path

import numpy as np
import pytest

from asrf.visualization.round8 import plot_round8_comparison_figure


def _rows(label: str) -> list[dict]:
    return [{"segment_index": 0, "start_frame": 0, "end_frame": 10, "predicted_label": label}]


def test_round8_plot_requires_exact_method_order(tmp_path: Path) -> None:
    methods = {name: _rows("reach") for name in ("ASRF-SF", "r5", "r10", "r20", "s5", "s10", "s20")}
    plot_round8_comparison_figure(
        np.zeros((3, 4, 10)), np.arange(10), [{"segment_index": 0, "start_frame": 0, "end_frame": 10, "gt_label": "reach"}], methods,
        tmp_path / "round8.png", ontology={"reach"}, title="test",
    )
    assert (tmp_path / "round8.png").is_file()


def test_round8_plot_rejects_wrong_method_order(tmp_path: Path) -> None:
    methods = {name: _rows("reach") for name in ("r5", "ASRF-SF", "r10", "r20", "s5", "s10", "s20")}
    with pytest.raises(ValueError, match="row order"):
        plot_round8_comparison_figure(
            np.zeros((3, 4, 10)), np.arange(10), [{"segment_index": 0, "start_frame": 0, "end_frame": 10, "gt_label": "reach"}], methods,
            tmp_path / "wrong.png", ontology={"reach"}, title="test",
        )
