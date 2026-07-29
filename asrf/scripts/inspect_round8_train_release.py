"""Create one Round 8 train/release visual for manual inspection."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_round7a as evaluation  # noqa: E402
from asrf.data.dataset import MultiTaskTrajectoryDataset  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.training.checkpointing import load_checkpoint  # noqa: E402
from asrf.utils.config import load_yaml_config, resolve_repo_path  # noqa: E402


def main() -> int:
    config = load_yaml_config("configs/brb_release_round8/baseline_single_frame.yaml")
    mapping = load_label_mapping(ROOT / config["data"]["label_config"])
    dataset = MultiTaskTrajectoryDataset(config["data"]["dataset_root"], ROOT / "splits/multitask_train.txt", ROOT / config["data"]["label_config"], allow_test=False, boundary_target_config={"boundary_target_mode": "single_frame", "boundary_include_frame_zero": True, "boundary_include_final_frame": False})
    index = next(i for i, entry in enumerate(dataset.entries) if entry == "train/pour/p1")
    sample = dataset[index]
    model = ASRFModel.from_config(config)
    checkpoint = resolve_repo_path(config["paths"]["output_dir"]) / "best.pt"
    model.load_state_dict(load_checkpoint(checkpoint, map_location="cpu")["model_state"]); model.eval()
    with torch.no_grad():
        output = model(sample["heatmap"].unsqueeze(0), valid_mask=sample["valid_mask"].unsqueeze(0))
    record = {"truth": sample["labels"], "targets": sample["hard_boundary_targets"], "heatmap": sample["heatmap"], "asb": output.asb_stage_probabilities[-1][0], "brb": output.brb_stage_probabilities[-1][0, 0]}
    record["entry"] = "train/pour/p1"; record["task"] = "pour"
    evaluation._attach_variant_metrics([record], 0.9, len(mapping))
    variants = record["variants"]
    figure, axes = plt.subplots(5, 1, figsize=(18, 9), sharex=True)
    axes[0].imshow(np.moveaxis(record["heatmap"].numpy(), 0, -1), aspect="auto", interpolation="nearest"); axes[0].set_ylabel("heatmap")
    for axis, key in zip(axes[1:], ("truth", "raw", "official", "calibrated")):
        values = record["truth"].numpy() if key == "truth" else variants[key]["prediction"].numpy()
        axis.imshow(values[np.newaxis, :], aspect="auto", interpolation="nearest", cmap="tab10", vmin=0, vmax=9); axis.set_ylabel(key)
    for peak in variants["official"]["boundaries"]: axes[3].axvline(peak, color="red", linewidth=0.7)
    for peak in evaluation._truth_boundaries(record): axes[1].axvline(peak, color="lime", linewidth=0.7)
    axes[-1].set_xlabel("frame"); figure.suptitle("baseline_single_frame — train/pour/p1 — contains release"); figure.tight_layout()
    path = ROOT / "outputs/brb_release_round8/figures/representative/baseline_single_frame_train_pour_p1.png"; path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=120); plt.close(figure)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
