#!/usr/bin/env python
"""Verify the user-applied train/pour annotation correction read-only."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asrf.data.annotations import convert_segments_to_frame_labels, load_segments_csv  # noqa: E402
from asrf.data.dataset import load_timestamp_vector  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.utils.config import resolve_repo_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
    parser.add_argument("--output", default="outputs/round6_diagnostics/manual_correction_verification.csv")
    args = parser.parse_args()
    root = Path(args.data_root) / "train" / "pour"
    mapping = load_label_mapping(resolve_repo_path("configs/labels_multitask.yaml"))
    rows: list[dict[str, object]] = []
    remaining: list[str] = []
    for demo in sorted(path for path in root.iterdir() if path.is_dir()):
        segments_path = demo / "segments.csv"
        timestamps_path = demo / "citr_features.csv"
        heatmap_path = demo / "citr_fingerprint_pure.png"
        if not (segments_path.is_file() and timestamps_path.is_file() and heatmap_path.is_file()):
            continue
        status = "valid"
        coverage_valid = False
        pick_count = translation_count = reach_count = transport_count = 0
        row_count = 0
        try:
            _, annotation_rows = load_segments_csv(segments_path)
            row_count = len(annotation_rows)
            counts = Counter(str(row.get("label", "")).strip() for row in annotation_rows)
            pick_count = counts["pick"]
            translation_count = counts["translation"]
            reach_count = counts["reach"]
            transport_count = counts["transport"]
            timestamps = load_timestamp_vector(timestamps_path)
            labels, _ = convert_segments_to_frame_labels(segments_path, timestamps, mapping)
            coverage_valid = len(labels) > 0 and (labels >= 0).all()
            if pick_count or translation_count:
                status = "manual_correction_incomplete"
                remaining.append(demo.name)
            elif not coverage_valid:
                status = "invalid_coverage"
        except Exception as exc:  # pragma: no cover - diagnostic path
            status = f"invalid: {exc}"
        rows.append({
            "trajectory_id": demo.name, "segments_path": str(segments_path),
            "pick_count": pick_count, "translation_count": translation_count,
            "reach_count": reach_count, "transport_count": transport_count,
            "row_count": row_count, "coverage_valid": coverage_valid, "status": status,
        })
    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["trajectory_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"valid_train_pour_trajectories={len(rows)}")
    print(f"remaining_pick_or_translation_trajectories={remaining}")
    if remaining:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
