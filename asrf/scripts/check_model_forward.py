#!/usr/bin/env python3
"""Run a tiny ASRF forward/backward architecture smoke test without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asrf.models import ASRFModel  # noqa: E402
from asrf.utils.config import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="ASRF architecture YAML path.")
    return parser.parse_args()


def main() -> int:
    config = load_yaml_config(parse_args().config)
    model = ASRFModel.from_config(config)
    num_classes = int(config.get("model", {}).get("num_classes", config.get("data", {}).get("num_classes", 7)))
    torch.manual_seed(int(config.get("experiment", {}).get("seed", 42)))
    heatmap = torch.randn(2, 3, 88, 17, requires_grad=True)
    valid_mask = torch.tensor(
        [[True] * 17, [True] * 11 + [False] * 6], dtype=torch.bool
    )

    model.train()
    output = model(heatmap, valid_mask=valid_mask)
    if len(output.asb_stage_logits) != 4 or len(output.brb_stage_logits) != 4:
        raise AssertionError("Expected four ASB and BRB stage outputs.")
    for logits, probabilities in zip(output.asb_stage_logits, output.asb_stage_probabilities):
        if logits.shape != (2, num_classes, 17) or probabilities.shape != (2, num_classes, 17):
            raise AssertionError("Unexpected ASB stage shape.")
        valid_probabilities = probabilities.permute(0, 2, 1)[valid_mask]
        if not torch.allclose(valid_probabilities.sum(dim=1), torch.ones(28)):
            raise AssertionError("ASB probabilities do not sum to one on valid frames.")
    for logits, probabilities in zip(output.brb_stage_logits, output.brb_stage_probabilities):
        if logits.shape != (2, 1, 17) or probabilities.shape != (2, 1, 17):
            raise AssertionError("Unexpected BRB stage shape.")
        if not torch.all((probabilities >= 0) & (probabilities <= 1)):
            raise AssertionError("BRB probabilities are outside [0,1].")

    print(f"encoder_features={tuple(output.encoder_features.shape)}")
    print(f"shared_features={tuple(output.shared_features.shape)}")
    print(f"asb_logits={[tuple(value.shape) for value in output.asb_stage_logits]}")
    print(f"asb_probabilities={[tuple(value.shape) for value in output.asb_stage_probabilities]}")
    print(f"brb_logits={[tuple(value.shape) for value in output.brb_stage_logits]}")
    print(f"brb_probabilities={[tuple(value.shape) for value in output.brb_stage_probabilities]}")

    loss = sum(value[valid_mask.unsqueeze(1).expand_as(value)].square().mean() for value in output.asb_stage_logits)
    loss = loss + sum(value[valid_mask.unsqueeze(1).expand_as(value)].square().mean() for value in output.brb_stage_logits)
    loss.backward()
    components = {
        "encoder": model.encoder,
        "shared_extractor": model.feature_extractor,
        "asb": model.asb,
        "brb": model.brb,
    }
    for name, component in components.items():
        gradient_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in component.parameters()
            if parameter.grad is not None
        )
        if gradient_norm <= 0:
            raise AssertionError(f"No gradient reached {name}.")
        print(f"{name}_gradient_sum={gradient_norm:.6f}")
    print("forward_backward_smoke=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
