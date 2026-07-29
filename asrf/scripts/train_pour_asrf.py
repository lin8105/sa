"""Train the strict seven-class pour-only ASRF baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asrf.training.trainer import ASRFTrainer
from asrf.utils.config import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pour_asrf_train.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    trainer = ASRFTrainer(config, device=args.device, resume=args.resume)
    summary = trainer.train()
    print(f"best_epoch={summary['best_epoch']}")
    print(f"stopping_epoch={summary['stopping_epoch']}")
    print(f"best_validation_total_loss={summary['best_validation_total_loss']:.9f}")
    print(f"elapsed_seconds={summary['elapsed_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
