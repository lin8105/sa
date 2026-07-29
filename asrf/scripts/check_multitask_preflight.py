#!/usr/bin/env python
"""Read-only multi-task data/model preflight plus one synthetic optimizer step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asrf.data.collate import collate_fn  # noqa: E402
from asrf.data.dataset import MultiTaskTrajectoryDataset  # noqa: E402
from asrf.losses.combined import ASRFLoss  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.training.checkpointing import load_checkpoint, save_checkpoint  # noqa: E402
from asrf.training.trainer import seed_everything  # noqa: E402
from asrf.utils.config import load_yaml_config, resolve_repo_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multitask_asrf_train.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    data = config["data"]
    label_path = resolve_repo_path(data["label_config"])
    root = Path(data["dataset_root"])
    train = MultiTaskTrajectoryDataset(root, resolve_repo_path(data["train_split"]), label_path, expected_height=88)
    val = MultiTaskTrajectoryDataset(root, resolve_repo_path(data["val_split"]), label_path, expected_height=88)
    if not train.entries or not val.entries:
        raise AssertionError("Train and validation splits must be non-empty.")
    seen_tasks: set[str] = set()
    for dataset in (train, val):
        for index in range(len(dataset)):
            sample = dataset[index]
            seen_tasks.add(str(sample["task_name"]))
            t = sample["heatmap"].shape[-1]
            if sample["heatmap"].shape != (3, 88, t):
                raise AssertionError(f"Unexpected heatmap shape for {sample['trajectory_id']}")
            if not (len(sample["labels"]) == len(sample["boundary_targets"]) == len(sample["valid_mask"]) == t):
                raise AssertionError(f"Temporal mismatch for {sample['trajectory_id']}")
            if not torch.all((sample["labels"] >= 0) & (sample["labels"] < 9)):
                raise AssertionError(f"Label range error for {sample['trajectory_id']}")
    expected_tasks = {"pour", "pick_and_place", "wipe"}
    if not expected_tasks.issubset(seen_tasks):
        raise AssertionError(f"Missing task in preflight: {expected_tasks - seen_tasks}")

    model = ASRFModel.from_config(config).cpu()
    model.eval()
    with torch.no_grad():
        for entry in ("train/pour/p1", "train/pick and place/pp1", "train/wipe/w1"):
            ds = MultiTaskTrajectoryDataset(root, _temporary_split(entry), label_path, expected_height=88)
            sample = collate_fn([ds[0]])
            output = model(sample["heatmap"], valid_mask=sample["valid_mask"])
            if output.asb_stage_logits[-1].shape[1] != 9:
                raise AssertionError("Multi-task forward did not produce nine ASB classes.")
            print(f"forward_ok task={sample['task_names'][0]} entry={entry} T={sample['lengths'][0].item()}")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    heatmap = torch.randn(2, 3, 88, 17)
    mask = torch.tensor([[True] * 17, [True] * 11 + [False] * 6])
    labels = torch.randint(0, 9, (2, 17))
    labels[1, 11:] = -100
    boundaries = torch.zeros(2, 17)
    boundaries[:, 0] = 1.0
    output = model(heatmap, valid_mask=mask)
    criterion = ASRFLoss(class_weights=torch.ones(9), boundary_positive_weight=2.0)
    loss = criterion(output, labels, boundaries, mask).total_loss
    if not torch.isfinite(loss):
        raise AssertionError("Synthetic preflight loss is not finite.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.requires_grad):
        raise AssertionError("Synthetic preflight gradient check failed.")
    optimizer.step()

    out_dir = REPO_ROOT / "outputs" / "multitask_baseline" / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "preflight.pt"
    save_checkpoint(checkpoint, {"model_state": model.state_dict(), "config": config})
    restored = ASRFModel.from_config(config).cpu()
    restored.load_state_dict(load_checkpoint(checkpoint)["model_state"])
    print(json.dumps({"tasks": sorted(seen_tasks), "train_count": len(train), "val_count": len(val), "loss": float(loss.detach()), "checkpoint": str(checkpoint)}, indent=2))
    return 0


def _temporary_split(entry: str) -> Path:
    path = REPO_ROOT / "outputs" / "multitask_baseline" / "preflight" / (entry.replace("/", "_") + ".txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(entry + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
