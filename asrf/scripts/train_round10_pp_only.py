"""Train the two isolated PP-only Round 10 BRB-target models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asrf.training.checkpointing import sha256_file  # noqa: E402
from asrf.training.trainer import ASRFTrainer  # noqa: E402

REFERENCE = ROOT / "outputs/brb_release_round8/hard_window_r5/best.pt"
REFERENCE_SHA256 = "61f32711d6de9e8c3809a0c1447459cb754adb31d3a0be8c9a0ba06f9b9c35af"
OUT = ROOT / "outputs/round10_pp_only_novel_segmentation/models"


def make_config(mode: str) -> dict:
    if mode not in {"single_frame", "hard_window_r5"}:
        raise ValueError(mode)
    target_mode = "single_frame" if mode == "single_frame" else "hard_window"
    return {
        "experiment": {"name": f"round10_pp_only_{mode}", "seed": 42, "training_scope": "pp_only", "novel_data_forbidden": True},
        "data": {
            "dataset_root": "/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data",
            "train_split": "outputs/round10_pp_only_novel_segmentation/audit/pp_train_manifest.txt",
            "val_split": "outputs/round10_pp_only_novel_segmentation/audit/pp_validation_manifest.txt",
            "label_config": "configs/labels_round10_pp_only.yaml",
            "heatmap_height": 88,
            "num_classes": 7,
            "batch_size": 1,
            "num_workers": 0,
            # Retreat is part of the PP-only ontology in later PP recordings,
            # but pp1--pp10 are the preregistered training split and contain
            # no retreat interval.  Keep the output row explicit with a zero
            # training weight rather than silently removing the ontology row.
            "allow_zero_class_weights": True,
            "boundary_target_mode": target_mode,
            "boundary_window_radius": 5,
            "boundary_gaussian_sigma": 1.0,
            "boundary_include_frame_zero": True,
            "boundary_include_final_frame": False,
        },
        "model": {
            "heatmap_channels": 3, "heatmap_height": 88, "encoder_output_channels": 128,
            "temporal_feature_channels": 64, "num_classes": 7, "num_temporal_layers": 10,
            "dilation_schedule": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], "kernel_size": 3,
            "dropout": 0.5, "causal": False, "asb_refinement_stages": 3, "brb_refinement_stages": 3,
        },
        "loss": {
            "class_weighting": "median_frequency", "smoothing": "gs_tmse", "tau": 4.0,
            "sigma": 1.0, "smoothing_weight": 1.0, "boundary_loss_weight": 0.1,
            "boundary_positive_weighting": "reciprocal_frequency",
        },
        "refinement": {"official_boundary_threshold": 0.5, "boundary_threshold": 0.5, "local_maximum": "strict", "voting": "majority"},
        "training": {
            "optimizer": "adam", "learning_rate": 0.0005, "stage2_learning_rate": 0.00005,
            "stage1_epochs": 5, "weight_decay": 0.0, "max_epochs": 35,
            "early_stopping_patience": 15, "minimum_epochs": 20, "batch_size": 1,
            "num_workers": 0, "device": "cpu", "best_metric": "val_total_loss", "save_last": True,
            "deterministic": True, "initialize_from_checkpoint": "outputs/brb_release_round8/hard_window_r5/best.pt",
            "initialize_class_row_mapping": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 6, "5": 7, "6": 9},
            "gradient_clip_norm": None,
        },
        "paths": {"output_dir": f"outputs/round10_pp_only_novel_segmentation/models/{mode}"},
    }


def run(mode: str, *, resume: bool = False) -> dict:
    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)
    actual = sha256_file(REFERENCE)
    if actual != REFERENCE_SHA256:
        raise ValueError(f"Round 8 initialization hash mismatch: {actual}")
    config = make_config(mode)
    output = ROOT / config["paths"]["output_dir"]
    if not resume and ((output / "best.pt").exists() or (output / "last.pt").exists()):
        raise FileExistsError(f"Refusing to overwrite existing Round 10 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    import yaml
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    resume_path = output / "last.pt" if resume else None
    if resume and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    trainer = ASRFTrainer(config, resume=resume_path)
    summary = trainer.train()
    summary.update({"condition": mode, "initialization_checkpoint_sha256": actual, "model_output_classes": 7, "novel_semantic_recognition": False})
    (output / "round10_run_metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"condition": mode, "best_epoch": summary.get("best_epoch"), "stopping_epoch": summary.get("stopping_epoch"), "elapsed_seconds": summary.get("elapsed_seconds"), "best_sha256": sha256_file(output / "best.pt"), "last_sha256": sha256_file(output / "last.pt")}, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("single_frame", "hard_window_r5"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.condition, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
