#!/usr/bin/env python
"""Read-only validation of test/wipe/w4, repeated twice."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asrf.data.annotations import load_segments_csv  # noqa: E402
from asrf.data.dataset import load_heatmap, load_timestamp_vector  # noqa: E402
from asrf.data.labels import load_label_mapping, normalize_label_name  # noqa: E402
from asrf.utils.config import resolve_repo_path  # noqa: E402


DATA_ROOT = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
W4 = DATA_ROOT / "test/wipe/w4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(mapping) -> dict:
    heatmap_path = W4 / "citr_fingerprint_pure.png"
    segments_path = W4 / "segments.csv"
    timestamp_path = W4 / "citr_features.csv"
    with Image.open(heatmap_path) as image:
        rgb = image.mode in {"RGB", "RGBA"}
        dimensions = {"width": image.width, "height": image.height, "mode": image.mode}
    timestamps = load_timestamp_vector(timestamp_path)
    annotation_format, rows = load_segments_csv(segments_path)
    if dimensions["width"] != len(timestamps):
        raise ValueError("w4 temporal alignment mismatch")
    canonical = [normalize_label_name(row["label"], mapping) for row in rows]
    raw_counts = Counter(row["label"] for row in rows)
    canonical_counts = Counter(canonical)
    canonical_frame_counts: Counter[str] = Counter()
    segment_metadata = []
    occupied = [False] * len(timestamps)
    coverage_valid = True
    for index, row in enumerate(rows):
        start = int(np.searchsorted(timestamps, int(row["start_timestamp_us"]), side="left"))
        end = int(np.searchsorted(timestamps, int(row["end_timestamp_us_exclusive"]), side="left"))
        if start >= end or any(occupied[start:end]):
            coverage_valid = False
        for frame in range(max(0, start), min(len(occupied), end)):
            occupied[frame] = True
        segment_metadata.append({
            "index": index, "raw_label": row["label"], "canonical_label": normalize_label_name(row["label"], mapping),
            "start": start, "end_exclusive": end,
            "duration_frames": end - start,
        })
        canonical_frame_counts[normalize_label_name(row["label"], mapping)] += max(0, end - start)
    coverage_valid = coverage_valid and all(occupied)
    return {
        "trajectory_id": "w4", "path": str(W4), "heatmap_sha256": sha256(heatmap_path), "segments_sha256": sha256(segments_path),
        "heatmap_exists": heatmap_path.is_file(), "segments_exists": segments_path.is_file(), "timestamp_exists": timestamp_path.is_file(),
        "rgb": rgb, "dimensions": dimensions, "T": int(len(timestamps)), "timestamp_rows": int(len(timestamps)),
        "duration_seconds": float((int(timestamps[-1]) - int(timestamps[0])) / 1_000_000), "annotation_format": annotation_format,
        "number_of_segments": len(rows), "raw_sequence": [row["label"] for row in rows],
        "canonical_sequence": canonical, "raw_label_counts": dict(sorted(raw_counts.items())),
        "canonical_frame_counts": dict(sorted(canonical_frame_counts.items())),
        "canonical_segment_counts": dict(sorted(canonical_counts.items())), "segments": segment_metadata,
        "all_labels_mapped": all(bool(name) and name in mapping for name in canonical),
        "coverage_valid": bool(coverage_valid), "all_durations_positive": all(item["duration_frames"] > 0 for item in segment_metadata),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/round6_diagnostics")
    args = parser.parse_args()
    mapping = load_label_mapping(resolve_repo_path("configs/labels_multitask.yaml"))
    first = audit(mapping)
    second = audit(mapping)
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "w4_scan_1.json").write_text(json.dumps(first, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "w4_scan_2.json").write_text(json.dumps(second, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stability = {"identical": first == second, "scan_1_sha256": first["segments_sha256"], "scan_2_sha256": second["segments_sha256"], "path": str(W4), "valid": bool(first["coverage_valid"] and first["all_durations_positive"] and first["all_labels_mapped"] and first["rgb"])}
    (output_dir / "w4_stability.json").write_text(json.dumps(stability, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stability": stability, "T": first["T"], "duration_seconds": first["duration_seconds"], "segments": first["number_of_segments"], "canonical_sequence": first["canonical_sequence"], "frame_counts": first["canonical_frame_counts"]}, indent=2, sort_keys=True))
    return 0 if stability["identical"] and stability["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
