"""Train the nine revised Round 9 incremental target-family models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asrf.training.checkpointing import sha256_file  # noqa: E402
from asrf.training.trainer import ASRFTrainer  # noqa: E402
from asrf.utils.config import resolve_repo_path  # noqa: E402


ORDER = (("pour", 3), ("pour", 5), ("wipe", 3), ("wipe", 5), ("plug", 3), ("plug", 5), ("pour", "all"), ("wipe", "all"), ("plug", "all"))
REFERENCE = ROOT / "outputs/brb_release_round8/hard_window_r5/best.pt"
REFERENCE_SHA256 = "61f32711d6de9e8c3809a0c1447459cb754adb31d3a0be8c9a0ba06f9b9c35af"


def make_config(family: str, size: int | str) -> dict:
    name = f"{family}_{size}"
    split = f"splits/round9_incremental/{family}_train_{size}_with_base_pp10.txt"
    return {
        "experiment": {"name": f"round9_incremental_{name}", "seed": 42, "target_family": family, "target_trajectory_count": size},
        "data": {
            "dataset_root": "/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data",
            "train_split": split,
            "val_split": "splits/round9_incremental/common_validation.txt",
            "label_config": "configs/labels_multitask_plug.yaml",
            "heatmap_height": 88,
            "num_classes": 12,
            "batch_size": 1,
            "num_workers": 0,
            "allow_zero_class_weights": True,
            "boundary_target_mode": "hard_window",
            "boundary_window_radius": 5,
            "boundary_gaussian_sigma": 1.0,
            "boundary_include_frame_zero": True,
            "boundary_include_final_frame": False,
        },
        "model": {"heatmap_channels": 3, "heatmap_height": 88, "encoder_output_channels": 128, "temporal_feature_channels": 64, "num_classes": 12, "num_temporal_layers": 10, "dilation_schedule": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], "kernel_size": 3, "dropout": 0.5, "causal": False, "asb_refinement_stages": 3, "brb_refinement_stages": 3},
        "loss": {"class_weighting": "median_frequency", "smoothing": "gs_tmse", "tau": 4.0, "sigma": 1.0, "smoothing_weight": 1.0, "boundary_loss_weight": 0.1, "boundary_positive_weighting": "reciprocal_frequency"},
        "refinement": {"official_boundary_threshold": 0.5, "boundary_threshold": 0.5, "local_maximum": "strict", "voting": "majority"},
        "training": {"optimizer": "adam", "learning_rate": 0.0005, "stage2_learning_rate": 0.00005, "stage1_epochs": 5, "weight_decay": 0.0, "max_epochs": 35, "early_stopping_patience": 15, "minimum_epochs": 20, "batch_size": 1, "num_workers": 0, "device": "cpu", "best_metric": "val_total_loss", "save_last": True, "deterministic": True, "initialize_from_checkpoint": "outputs/brb_release_round8/hard_window_r5/best.pt", "gradient_clip_norm": None},
        "paths": {"output_dir": f"outputs/round9_incremental_learning/models/{family}/n{size}"},
    }


def run_one(family: str, size: int | str) -> dict:
    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)
    actual_sha = sha256_file(REFERENCE)
    if actual_sha != REFERENCE_SHA256:
        raise ValueError(f"Round 8 initialization hash mismatch: {actual_sha}")
    config = make_config(family, size)
    output = ROOT / config["paths"]["output_dir"]
    if (output / "best.pt").exists() or (output / "last.pt").exists():
        raise FileExistsError(f"Refusing to overwrite existing Round 9 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    import yaml

    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    trainer = ASRFTrainer(config)
    summary = trainer.train()
    summary["target_family"] = family
    summary["target_trajectory_count"] = size
    summary["initialization_checkpoint_sha256"] = actual_sha
    (output / "round9_run_metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"family": family, "size": size, "best_epoch": summary["best_epoch"], "elapsed_seconds": summary["elapsed_seconds"], "output": str(output)}, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("pour", "wipe", "plug"))
    parser.add_argument("--size", choices=("3", "5", "all"))
    parser.add_argument("--through", action="store_true", help="Run the prescribed order through the selected family/size.")
    args = parser.parse_args()
    if args.family is None:
        selected = list(ORDER)
    else:
        selected = [(args.family, "all" if args.size == "all" else int(args.size))]
        if args.through:
            selected = []
            target = (args.family, "all" if args.size == "all" else int(args.size))
            for item in ORDER:
                selected.append(item)
                if item == target:
                    break
    for family, size in selected:
        run_one(family, size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
