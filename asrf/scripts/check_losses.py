"""Round-3 synthetic loss and gradient smoke test; never trains a model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asrf.losses import ASRFLoss
from asrf.models import ASRFModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    torch.manual_seed(42)
    model = ASRFModel(dropout=0.0)
    model.train()
    heatmap = torch.randn(2, 3, 88, 11)
    valid_mask = torch.tensor(
        [[True] * 11, [True] * 7 + [False] * 4], dtype=torch.bool
    )
    labels = torch.randint(0, 7, (2, 11))
    labels[1, 7:] = -100
    boundaries = torch.zeros(2, 11)
    boundaries[:, 0] = 1.0
    boundaries[0, 5] = 1.0
    boundaries[1, 3] = 1.0
    output = model(heatmap, valid_mask)
    criterion = ASRFLoss(
        class_weights=torch.ones(7),
        boundary_positive_weight=4.0,
        tau=4.0,
        sigma=1.0,
        smoothing_weight=1.0,
        boundary_loss_weight=0.1,
    )
    result = criterion(output, labels, boundaries, valid_mask)
    if not torch.isfinite(result.total_loss):
        raise RuntimeError("combined loss is not finite")
    result.total_loss.backward()
    parameter_groups = {
        "encoder": model.encoder,
        "shared": model.feature_extractor,
        "asb": model.asb,
        "brb": model.brb,
    }
    gradient_sums = {}
    for name, module in parameter_groups.items():
        value = sum(float(parameter.grad.abs().sum()) for parameter in module.parameters() if parameter.grad is not None)
        if value <= 0.0:
            raise RuntimeError(f"no gradient reached {name}")
        gradient_sums[name] = value
    print(f"total_loss={float(result.total_loss.detach()):.9f}")
    print(f"asb_ce={float(result.asb_ce.detach()):.9f}")
    print(f"asb_smoothing={float(result.asb_smoothing.detach()):.9f}")
    print(f"brb_loss={float(result.brb_loss.detach()):.9f}")
    print(f"valid_frame_count={result.valid_frame_count}")
    print(f"valid_transition_count={result.valid_transition_count}")
    print(f"boundary_positive_count={result.boundary_positive_count}")
    for name, value in gradient_sums.items():
        print(f"{name}_gradient_sum={value:.6f}")
    print("loss_smoke=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
