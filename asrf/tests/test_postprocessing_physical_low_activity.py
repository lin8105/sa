from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from postprocessing.compress_timeline import (
    PHYSICAL_SIGNAL_NAMES,
    physical_low_activity_mask,
)


def test_every_synchronized_signal_genuinely_vetoes_low_activity():
    thresholds = {name: 1.0 for name in PHYSICAL_SIGNAL_NAMES}
    signals = {name: np.zeros(3, dtype=float) for name in PHYSICAL_SIGNAL_NAMES}
    assert physical_low_activity_mask(signals, thresholds).tolist() == [True] * 3

    # Each of the seven added channels must independently change the operative
    # joint mask; this guards against diagnostic-only or disconnected gates.
    for name in PHYSICAL_SIGNAL_NAMES[2:]:
        trial = {key: value.copy() for key, value in signals.items()}
        trial[name][1] = 1.0  # extra physical limits are strict upper bounds
        result = physical_low_activity_mask(trial, thresholds)
        assert result.tolist() == [True, False, True], name

    # Keep the historical inclusive velocity threshold semantics unchanged.
    for name in PHYSICAL_SIGNAL_NAMES[:2]:
        trial = {key: value.copy() for key, value in signals.items()}
        trial[name][1] = thresholds[name]
        assert physical_low_activity_mask(trial, thresholds).tolist() == [True] * 3


def test_physical_mask_rejects_nonfinite_limits_and_misaligned_signals():
    thresholds = {name: 1.0 for name in PHYSICAL_SIGNAL_NAMES}
    signals = {name: np.zeros(2, dtype=float) for name in PHYSICAL_SIGNAL_NAMES}
    thresholds["force_change"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        physical_low_activity_mask(signals, thresholds)

    thresholds["force_change"] = 1.0
    signals["force_change"] = np.zeros(3, dtype=float)
    with pytest.raises(ValueError, match="aligned"):
        physical_low_activity_mask(signals, thresholds)
