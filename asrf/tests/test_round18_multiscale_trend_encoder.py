import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
import run_round18_multiscale_trend_encoder_loso as round18  # noqa: E402


def test_round18_finite_differences_are_segment_local_and_deterministic():
    values = np.asarray([[1.0], [4.0], [10.0]], dtype=np.float32)
    first, second = round18.finite_differences(values)
    np.testing.assert_allclose(first[:, 0], [0.0, 3.0, 6.0])
    np.testing.assert_allclose(second[:, 0], [0.0, 0.0, 3.0])


def test_round18_short_segments_and_phase_assignment():
    assert round18.relative_time(0).shape == (0,)
    np.testing.assert_allclose(round18.relative_time(1), [0.0])
    assert round18.phase_bin_bounds(1) == [(0, 0)] * 7 + [(0, 1)]
    stats = round18.phase_statistics_from_features(np.ones((1, 2), dtype=np.float32), 1)
    assert stats.shape == (8, 6)
    np.testing.assert_allclose(stats[:-1], 0.0)
    np.testing.assert_allclose(stats[-1, :2], 1.0)


def test_round18_phase_statistics_ignore_padding():
    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    padded = np.concatenate((values, np.full((5, 2), 999.0, dtype=np.float32)))
    np.testing.assert_allclose(
        round18.phase_statistics_from_features(values, 6),
        round18.phase_statistics_from_features(padded, 6),
    )


def test_round18_encoder_padding_mask_and_phase_gradients():
    torch.manual_seed(42)
    model = round18.TrendEncoder(round18.input_dim("phase"), 3, ordered_phase=True, multiscale=True)
    sequence = torch.randn(2, 7, round18.input_dim("phase"), requires_grad=True)
    mask = torch.tensor([[True, True, True, True, True, False, False], [True] * 7])
    embedding, logits = model(sequence, mask, mask.sum(1), torch.zeros(2))
    (embedding.sum() + logits.sum()).backward()
    assert embedding.shape == (2, 128)
    assert logits.shape == (2, 3)
    assert sequence.grad is not None
    assert float(sequence.grad.abs().sum()) > 0.0
