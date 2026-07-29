"""Audit round-3 boundary targets against external annotations read-only."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asrf.data.annotations import convert_segments_to_frame_labels, load_segments_csv
from asrf.data.boundary_targets import boundary_indices, generate_boundary_targets
from asrf.data.dataset import load_timestamp_vector, read_split_file
from asrf.data.labels import load_label_mapping
from asrf.utils.config import load_yaml_config, resolve_repo_path


def audit_root(root: Path, split_path: Path, labels_path: Path) -> list[dict[str, str | int]]:
    mapping = load_label_mapping(labels_path)
    rows: list[dict[str, str | int]] = []
    for trajectory_id in read_split_file(split_path):
        demo = root / trajectory_id
        timestamps = load_timestamp_vector(demo / "citr_features.csv")
        _, annotation_rows = load_segments_csv(demo / "segments.csv")
        labels_np, _ = convert_segments_to_frame_labels(demo / "segments.csv", timestamps, mapping)
        labels = torch.from_numpy(labels_np)
        targets = generate_boundary_targets(labels)
        generated = boundary_indices(targets)
        transition_indices = torch.where(labels[1:] != labels[:-1])[0].add(1).tolist()
        expected = [0] + transition_indices if len(labels) else []
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "T": len(labels),
                "number_of_segments": len(annotation_rows),
                "expected_internal_boundaries": len(transition_indices),
                "generated_positive_frames": len(generated),
                "positive_indices": " ".join(str(index) for index in generated),
                "annotation_transition_indices": " ".join(str(index) for index in transition_indices),
                "match_status": "match" if generated == expected else "mismatch",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pour.yaml")
    parser.add_argument("--output", default="outputs/round3_diagnostics/boundary_target_audit.csv")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    paths = config["paths"]
    labels_path = resolve_repo_path(paths["label_mapping"])
    all_rows = audit_root(Path(paths["train_dataset_root"]), resolve_repo_path(paths["train_split"]), labels_path)
    all_rows.extend(audit_root(Path(paths["train_dataset_root"]), resolve_repo_path(paths["val_split"]), labels_path))
    all_rows.extend(audit_root(Path(paths["test_dataset_root"]), resolve_repo_path(paths["test_split"]), labels_path))
    all_rows.extend(audit_root(Path(paths["test_dataset_root"]), resolve_repo_path(paths["test_p3_p5_split"]), labels_path))
    output = resolve_repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trajectory_id", "T", "number_of_segments", "expected_internal_boundaries",
        "generated_positive_frames", "positive_indices", "annotation_transition_indices", "match_status",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    mismatches = [row for row in all_rows if row["match_status"] != "match"]
    print(f"audited_trajectories={len(all_rows)}")
    print(f"mismatches={len(mismatches)}")
    print(f"output={output}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
