"""Finish non-training artifacts for a run whose optimization already completed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asrf.training.trainer import ASRFTrainer  # noqa: E402
from asrf.utils.config import load_yaml_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    config = load_yaml_config(output / "config.yaml")
    trainer = ASRFTrainer(config)
    trainer.export_validation()
    summary_path = output / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    metadata = {
        "target_family": config["experiment"]["target_family"],
        "target_trajectory_count": config["experiment"]["target_trajectory_count"],
        "initialization_checkpoint_sha256": trainer.initialization_metadata.get("checkpoint_sha256"),
        "best_epoch": summary.get("best_epoch"),
        "stopping_epoch": summary.get("stopping_epoch"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "status": "training_completed_postprocessing_repaired",
    }
    (output / "round9_run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
