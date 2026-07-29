"""Resolve the strict pour train/validation split and write read-only statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from asrf.data.dataset import load_trajectory_sample, read_split_file
from asrf.data.labels import load_label_mapping
from asrf.losses.classification import collect_training_statistics
from asrf.utils.config import load_yaml_config, resolve_repo_path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _describe(root: Path, split: Path, label_path: Path, mapping: dict[str, int]) -> dict[str, object]:
    ids = read_split_file(split)
    class_names = sorted(mapping, key=mapping.get)
    class_frames = Counter()
    class_segments = Counter()
    boundary_positive = 0
    frames = 0
    segment_hashes: dict[str, str] = {}
    heatmap_hashes: dict[str, str] = {}
    trajectories: list[dict[str, object]] = []
    for trajectory_id in ids:
        path = root / trajectory_id
        sample = load_trajectory_sample(path, mapping)
        labels = sample["labels"]
        frames += int(labels.numel())
        boundary_positive += int(sample["boundary_targets"].sum())
        for value in labels.tolist():
            class_frames[class_names[int(value)]] += 1
        for row in sample["segments"]:
            raw = row["label"]
            normalized = raw
            if raw == "pick":
                normalized = "reach"
            if raw == "translation":
                normalized = "transport"
            class_segments[normalized] += 1
        segment_hashes[trajectory_id] = _sha(path / "segments.csv")
        heatmap_hashes[trajectory_id] = _sha(path / "citr_fingerprint_pure.png")
        trajectories.append({"trajectory_id": trajectory_id, "path": str(path.resolve()), "frames": int(labels.numel()), "segments": len(sample["segments"]), "temporal_width": int(sample["heatmap"].shape[-1])})
    return {"split": str(split), "trajectory_ids": ids, "trajectories": trajectories, "total_frames": frames, "total_segments": sum(class_segments.values()), "class_frames": dict(class_frames), "class_segments": dict(class_segments), "boundary_positive_count": boundary_positive, "boundary_negative_count": frames - boundary_positive, "boundary_positive_weight": frames / boundary_positive if boundary_positive else None, "segments_sha256": segment_hashes, "heatmap_sha256": heatmap_hashes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pour_asrf_train.yaml")
    parser.add_argument("--output", default="outputs/pour_baseline/split_statistics.json")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    data = config["data"]
    root = Path(data["train_root"])
    label_path = resolve_repo_path(data["label_config"])
    mapping = load_label_mapping(label_path)
    train_split = resolve_repo_path(data["train_split"])
    val_split = resolve_repo_path(data["val_split"])
    train_ids, val_ids = read_split_file(train_split), read_split_file(val_split)
    if set(train_ids) & set(val_ids):
        raise SystemExit("train/validation overlap")
    result = {"label_map": dict(mapping), "train": _describe(root, train_split, label_path, mapping), "validation": _describe(root, val_split, label_path, mapping), "overlap": sorted(set(train_ids) & set(val_ids))}
    result["exact_duplicate_hashes_across_splits"] = {"segments": [], "heatmaps": []}
    for kind, field in (("segments", "segments_sha256"), ("heatmaps", "heatmap_sha256")):
        train_hashes = result["train"][field]
        val_hashes = result["validation"][field]
        result["exact_duplicate_hashes_across_splits"][kind] = sorted(set(train_hashes.values()) & set(val_hashes.values()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
