"""Report training-only class and boundary statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asrf.data.labels import load_label_mapping
from asrf.losses.classification import collect_training_statistics
from asrf.utils.config import load_yaml_config, resolve_repo_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pour.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    paths = config["paths"]
    mapping = load_label_mapping(resolve_repo_path(paths["label_mapping"]))
    stats = collect_training_statistics(
        paths["train_dataset_root"], resolve_repo_path(paths["train_split"]), mapping
    )
    names = sorted(mapping, key=mapping.get)
    print("class_statistics")
    for name, count, segments, frequency, weight in zip(names, stats.class_counts, stats.segment_counts, stats.class_frequencies, stats.class_weights):
        print(f"{name}: frames={int(count)} segments={int(segments)} frequency={float(frequency):.9f} weight={float(weight):.9f}")
    print(f"total_valid_frames={stats.total_valid_frames}")
    print(f"boundary_positive_count={stats.boundary_positive_count}")
    print(f"boundary_negative_count={stats.boundary_negative_count}")
    print(f"boundary_positive_ratio={stats.boundary_positive_ratio:.9f}")
    print(f"boundary_positive_weight={stats.boundary_positive_weight:.9f}")
    print(f"median_frequency={float(stats.median_frequency):.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
