#!/usr/bin/env python
"""Train the independent nine-class multi-task ASRF baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asrf.training.trainer import ASRFTrainer  # noqa: E402
from asrf.utils.config import load_yaml_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multitask_asrf_train.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    if config["training"].get("initialize_from_checkpoint"):
        raise ValueError("Multi-task round 5M must initialize from random weights.")
    trainer = ASRFTrainer(config, device=args.device, resume=args.resume)
    summary = trainer.train()
    print(f"best_epoch={summary['best_epoch']}")
    print(f"stopping_epoch={summary['stopping_epoch']}")
    print(f"best_validation_total_loss={summary['best_validation_total_loss']:.9f}")
    print(f"elapsed_seconds={summary['elapsed_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
