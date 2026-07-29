from __future__ import annotations

from pathlib import Path

import torch

from asrf.models import ASRFModel
from asrf.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]


def test_state_dict_save_and_reload_matches_eval_outputs(tmp_path: Path) -> None:
    config = load_yaml_config(ROOT / "configs/pour_asrf_architecture.yaml")
    torch.manual_seed(11)
    model = ASRFModel.from_config(config).eval()
    heatmap = torch.randn(1, 3, 88, 11)
    valid_mask = torch.tensor([[True] * 8 + [False] * 3])
    with torch.no_grad():
        expected = model(heatmap, valid_mask)

    checkpoint_path = tmp_path / "architecture_state.pt"
    torch.save(model.state_dict(), checkpoint_path)
    restored = ASRFModel.from_config(config).eval()
    restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    with torch.no_grad():
        actual = restored(heatmap, valid_mask)

    assert torch.allclose(expected.encoder_features, actual.encoder_features)
    assert torch.allclose(expected.shared_features, actual.shared_features)
    for expected_values, actual_values in zip(expected.asb_stage_logits, actual.asb_stage_logits):
        assert torch.allclose(expected_values, actual_values)
    for expected_values, actual_values in zip(expected.brb_stage_probabilities, actual.brb_stage_probabilities):
        assert torch.allclose(expected_values, actual_values)

