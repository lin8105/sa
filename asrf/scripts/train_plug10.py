"""Train the focused Round 9 Plug-10 continuation without touching prior runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from asrf.training.checkpointing import sha256_file  # noqa: E402
from asrf.training.trainer import ASRFTrainer  # noqa: E402
from train_round9_incremental import REFERENCE, REFERENCE_SHA256, make_config  # noqa: E402


def main() -> int:
    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)
    reference_sha256 = sha256_file(REFERENCE)
    if reference_sha256 != REFERENCE_SHA256:
        raise ValueError(f"Round 8 initialization hash mismatch: {reference_sha256}")

    config = make_config("plug", 10)
    config["experiment"]["name"] = "round9_incremental_plug_10"
    config["data"]["train_split"] = "splits/round9_incremental/plug_train_10_with_base_pp10.txt"
    config["paths"]["output_dir"] = "outputs/round9_incremental_learning/plug/n10"
    output = ROOT / config["paths"]["output_dir"]
    if (output / "best.pt").exists() or (output / "last.pt").exists():
        raise FileExistsError(f"Refusing to overwrite existing Plug-10 checkpoint: {output}")
    output.mkdir(parents=True, exist_ok=True)

    import yaml

    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    trainer = ASRFTrainer(config)
    summary = trainer.train()
    summary["target_family"] = "plug"
    summary["target_trajectory_count"] = 10
    summary["initialization_checkpoint_sha256"] = reference_sha256
    (output / "round9_run_metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": summary["best_epoch"], "elapsed_seconds": summary["elapsed_seconds"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
