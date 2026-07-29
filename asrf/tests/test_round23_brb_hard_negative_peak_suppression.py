from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_round23_brb_hard_negative_peak_suppression as r23


def test_narrow_gaussian_is_symmetric_and_peaks_at_boundary() -> None:
    labels = torch.zeros(21, dtype=torch.long)
    labels[10:] = 1
    target = r23.narrow_gaussian_targets(labels, 2.0)
    assert float(target[10]) == 1.0
    assert torch.allclose(target[8], target[12])
    assert float(target[1]) < float(target[10])


def test_interior_mask_excludes_both_boundary_shoulders() -> None:
    labels = np.zeros(100, dtype=np.int64)
    labels[50:] = 1
    mask = r23.interior_mask(labels, 10)
    assert not mask[:10].any()
    assert not mask[40:60].any()
    assert mask[20:40].all()
    assert mask[60:80].all()


def test_sensitive_boundary_weight_only_changes_transition_frames() -> None:
    labels = np.zeros(80, dtype=np.int64)
    labels[40:] = 1
    weights = r23.boundary_frame_weights(labels, short_cutoff=100, sensitive_weight=2.0)
    assert weights[40] == 2.0
    assert np.all(weights[np.arange(80) != 40] == 1.0)


def test_boundary_matching_is_one_to_one() -> None:
    result = r23.boundary_counts_local([0, 10, 11], [0, 10], 2)
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0
