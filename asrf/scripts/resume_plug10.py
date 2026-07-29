"""Resume the recoverable focused Round 9 Plug-10 run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asrf.training.checkpointing import sha256_file  # noqa: E402
from asrf.training.trainer import ASRFTrainer  # noqa: E402
from asrf.utils.config import load_yaml_config  # noqa: E402


def main() -> int:
    output = ROOT / "outputs/round9_incremental_learning/plug/n10"
    config = load_yaml_config(output / "config.yaml")
    last = output / "last.pt"
    if not last.is_file():
        raise FileNotFoundError(last)
    trainer = ASRFTrainer(config, resume=last)
    summary = trainer.train()
    summary["target_family"] = "plug"
    summary["target_trajectory_count"] = 10
    summary["last_resume_checkpoint_sha256"] = sha256_file(last)
    (output / "round9_run_metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": summary["best_epoch"], "stopping_epoch": summary["stopping_epoch"], "elapsed_seconds": summary["elapsed_seconds"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
