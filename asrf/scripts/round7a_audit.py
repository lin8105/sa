#!/usr/bin/env python
"""Read-only Round 7A integrity, target, and frozen-baseline audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asrf.data.annotations import convert_segments_to_frame_labels, load_segments_csv  # noqa: E402
from asrf.data.boundary_targets import generate_boundary_targets  # noqa: E402
from asrf.data.dataset import MultiTaskTrajectoryDataset, load_heatmap, load_timestamp_vector, read_split_file  # noqa: E402
from asrf.data.labels import load_label_mapping, normalize_label_name  # noqa: E402
from asrf.losses.classification import collect_statistics_for_entries  # noqa: E402
from asrf.training.checkpointing import checkpoint_manifest, sha256_file  # noqa: E402
from asrf.utils.config import load_yaml_config, resolve_repo_path  # noqa: E402


DATA_ROOT = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
EXPECTED_BASELINE = "ad557bc5b10bc00d1582c3a1d82897e81173f6abc83dfc2220a2fb96ee2c0241"
EXPECTED_POUR = "586fc50c91c735f7212c16baa052f43655b3140408aa3c0d534d11daa1fbc358"
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def audit_splits(output: Path) -> dict[str, Any]:
    mapping = load_label_mapping(REPO_ROOT / "configs/labels_multitask.yaml")
    split_paths = {
        "train": REPO_ROOT / "splits/multitask_train.txt",
        "validation": REPO_ROOT / "splits/multitask_val.txt",
        "test_pour": REPO_ROOT / "splits/multitask_test_pour.txt",
        "test_pp": REPO_ROOT / "splits/multitask_test_pp.txt",
        "test_wipe": REPO_ROOT / "splits/multitask_test_wipe.txt",
    }
    entries = {name: read_split_file(path) for name, path in split_paths.items()}
    all_entries = [entry for values in entries.values() for entry in values]
    resolved: dict[str, Path] = {}
    file_fingerprints: dict[str, dict[str, Any]] = {}
    canonical_violations: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    for entry in all_entries:
        path = (DATA_ROOT / entry).resolve()
        resolved[entry] = path
        try:
            timestamps = load_timestamp_vector(path / "citr_features.csv")
            heatmap = load_heatmap(path / "citr_fingerprint_pure.png")
            _, rows = load_segments_csv(path / "segments.csv")
            labels, _ = convert_segments_to_frame_labels(path / "segments.csv", timestamps, mapping)
            for row in rows:
                canonical = normalize_label_name(row["label"], mapping)
                if canonical not in mapping:
                    canonical_violations.append({"entry": entry, "raw_label": row["label"], "canonical": canonical})
            file_fingerprints[entry] = {
                "resolved_path": str(path),
                "T": int(len(timestamps)),
                "heatmap_shape": list(heatmap.shape),
                "heatmap_sha256": _sha(path / "citr_fingerprint_pure.png"),
                "timestamps_sha256": _sha(path / "citr_features.csv"),
                "segments_sha256": _sha(path / "segments.csv"),
                "label_count": int(len(labels)),
            }
        except Exception as exc:  # record all integrity failures in one artifact
            invalid.append({"entry": entry, "error": f"{type(exc).__name__}: {exc}"})

    duplicate_entries = sorted(entry for entry, count in Counter(all_entries).items() if count > 1)
    path_to_entries: dict[str, list[str]] = {}
    for entry, path in resolved.items():
        path_to_entries.setdefault(str(path), []).append(entry)
    duplicate_physical = {path: sorted(values) for path, values in path_to_entries.items() if len(values) > 1}
    train_val_overlap = sorted(set(entries["train"]) & set(entries["validation"]))
    train_val_test_leakage = sorted((set(entries["train"]) | set(entries["validation"])) & set(sum((entries[k] for k in ("test_pour", "test_pp", "test_wipe")), [])))
    wipe = entries["test_wipe"]
    expected_wipe = ["test/wipe/w1", "test/wipe/w2", "test/wipe/w3", "test/wipe/w4"]
    w4 = file_fingerprints.get("test/wipe/w4", {})
    result = {
        "dataset_root": str(DATA_ROOT),
        "split_files": {name: str(path) for name, path in split_paths.items()},
        "counts": {name: len(values) for name, values in entries.items()},
        "entries": entries,
        "train_validation_overlap": train_val_overlap,
        "train_validation_test_leakage": train_val_test_leakage,
        "duplicate_split_entries": duplicate_entries,
        "duplicate_physical_trajectories": duplicate_physical,
        "canonical_label_violations": canonical_violations,
        "invalid_trajectories": invalid,
        "wipe_test_exact": wipe == expected_wipe,
        "wipe_test_entries": wipe,
        "w4_valid": bool("test/wipe/w4" in file_fingerprints and not invalid and file_fingerprints["test/wipe/w4"]["label_count"] == file_fingerprints["test/wipe/w4"]["T"]),
        "scan_1": file_fingerprints,
    }
    # Repeat all file reads and compare metadata/hashes, without modifying data.
    scan_2: dict[str, dict[str, Any]] = {}
    for entry, path in resolved.items():
        timestamps = load_timestamp_vector(path / "citr_features.csv")
        labels, _ = convert_segments_to_frame_labels(path / "segments.csv", timestamps, mapping)
        scan_2[entry] = {
            "resolved_path": str(path),
            "T": int(len(timestamps)),
            "heatmap_shape": list(load_heatmap(path / "citr_fingerprint_pure.png").shape),
            "heatmap_sha256": _sha(path / "citr_fingerprint_pure.png"),
            "timestamps_sha256": _sha(path / "citr_features.csv"),
            "segments_sha256": _sha(path / "segments.csv"),
            "label_count": int(len(labels)),
        }
    result["scan_2"] = scan_2
    result["stable_across_two_scans"] = file_fingerprints == scan_2
    result["pass"] = bool(
        not train_val_overlap and not train_val_test_leakage and not duplicate_entries
        and not duplicate_physical and not canonical_violations and not invalid
        and result["wipe_test_exact"] and result["w4_valid"] and result["stable_across_two_scans"]
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json) + "\n", encoding="utf-8")
    return result


def _target_stats(root: Path, split: Path, mapping) -> dict[str, Any]:
    positive = negative = 0
    frame0_positive = final_positive = 0
    rows: list[dict[str, Any]] = []
    for entry in read_split_file(split):
        path = root / entry
        timestamps = load_timestamp_vector(path / "citr_features.csv")
        labels, _ = convert_segments_to_frame_labels(path / "segments.csv", timestamps, mapping)
        tensor = torch.from_numpy(labels)
        targets = generate_boundary_targets(tensor)
        indices = torch.where(targets > 0.5)[0].tolist()
        positive += len(indices)
        negative += len(targets) - len(indices)
        frame0_positive += int(bool(len(targets) and targets[0] == 1))
        final_positive += int(bool(len(targets) and targets[-1] == 1))
        rows.append({"trajectory_id": entry, "T": len(targets), "positive_count": len(indices), "negative_count": len(targets) - len(indices), "positive_indices": indices})
    total = positive + negative
    return {"trajectory_count": len(rows), "positive_count": positive, "negative_count": negative, "total_frames": total, "positive_ratio": positive / total if total else 0.0, "frame0_positive_trajectories": frame0_positive, "final_frame_positive_trajectories": final_positive, "rows": rows}


def audit_targets(output: Path) -> dict[str, Any]:
    mapping = load_label_mapping(REPO_ROOT / "configs/labels_multitask.yaml")
    stats = {name: _target_stats(DATA_ROOT, path, mapping) for name, path in {
        "train": REPO_ROOT / "splits/multitask_train.txt", "validation": REPO_ROOT / "splits/multitask_val.txt",
        "test_pour": REPO_ROOT / "splits/multitask_test_pour.txt", "test_pp": REPO_ROOT / "splits/multitask_test_pp.txt", "test_wipe": REPO_ROOT / "splits/multitask_test_wipe.txt",
    }.items()}
    train = stats["train"]
    output.write_text(json.dumps({
        "definition": "target[0]=1 for the first valid frame; target[t]=1 iff the canonical frame label changes from t-1 to t; no final-frame target is added merely because it is final; no widening or smoothing",
        "frame_0_is_positive": train["frame0_positive_trajectories"] == train["trajectory_count"],
        "final_frame_is_target": train["final_frame_positive_trajectories"] > 0,
        "final_frame_positive_trajectory_count": train["final_frame_positive_trajectories"],
        "positive_target_frames": train["positive_count"],
        "negative_target_frames": train["negative_count"],
        "original_positive_ratio": train["positive_ratio"],
        "positive_weight_formula": "(positive + negative) / positive = 1 / positive_ratio, computed from multitask_train only",
        "original_positive_weight": (train["positive_count"] + train["negative_count"]) / train["positive_count"],
        "split_statistics": stats,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def baseline_manifest(output: Path) -> dict[str, Any]:
    baseline = REPO_ROOT / "outputs/multitask_baseline/best.pt"
    pour = REPO_ROOT / "outputs/pour_baseline/best.pt"
    config = load_yaml_config("configs/multitask_asrf_train.yaml")
    stats = json.loads((REPO_ROOT / "outputs/multitask_baseline/split_statistics.json").read_text())
    boundary = json.loads((REPO_ROOT / "outputs/multitask_baseline/boundary_statistics.json").read_text())
    summary = json.loads((REPO_ROOT / "outputs/multitask_baseline/training_summary.json").read_text())
    with (REPO_ROOT / "outputs/multitask_baseline/metrics.csv").open(encoding="utf-8", newline="") as handle:
        metric_rows = [row for row in csv.DictReader(handle) if row.get("split") == "val"]
    best_validation_row = min(metric_rows, key=lambda row: float(row["total_loss"])) if metric_rows else {}
    test_summaries = {}
    for task in ("pour", "pp", "wipe", "all_tasks"):
        path = REPO_ROOT / "outputs/multitask_baseline/test" / f"{task}_summary.json"
        if path.is_file():
            test_summaries[task] = json.loads(path.read_text())
    manifest = {
        "round": "7A",
        "experiment": "brb positive-weight ablation",
        "frozen_baseline": checkpoint_manifest(baseline),
        "expected_baseline_sha256": EXPECTED_BASELINE,
        "frozen_pour_baseline": checkpoint_manifest(pour),
        "expected_pour_sha256": EXPECTED_POUR,
        "baseline_hashes_match": sha256_file(baseline) == EXPECTED_BASELINE and sha256_file(pour) == EXPECTED_POUR,
        "baseline_config": config,
        "baseline_config_path": "configs/multitask_asrf_train.yaml",
        "baseline_positive_weight": boundary["train_split"]["boundary_positive_weight"],
        "baseline_boundary_statistics": boundary["train_split"],
        "baseline_split_statistics": stats,
        "training_selection": {"primary": "validation total loss", "best_epoch": summary["best_epoch"], "stopping_epoch": summary["stopping_epoch"], "best_validation_total_loss": summary["best_validation_total_loss"]},
        "baseline_validation_metrics_source": "outputs/multitask_baseline/metrics.csv and training_summary.json",
        "baseline_validation_metrics_at_primary_epoch": best_validation_row,
        "baseline_test_metrics": test_summaries,
        "baseline_outputs_preserved": True,
        "new_experiments_root": "outputs/brb_ablation_round7a",
        "fixed_thresholds": list(THRESHOLDS),
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/brb_ablation_round7a")
    args = parser.parse_args()
    out = REPO_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    baseline_manifest(out / "baseline_manifest.json")
    split = audit_splits(out / "split_integrity.json")
    audit_targets(out / "target_audit.json")
    print(json.dumps({"split_integrity_pass": split["pass"], "baseline_sha256": sha256_file(REPO_ROOT / "outputs/multitask_baseline/best.pt")}, indent=2))
    return 0 if split["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
