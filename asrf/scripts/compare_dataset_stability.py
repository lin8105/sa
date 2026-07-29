"""Compare two read-only dataset inventory scans and publish the stability record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _file_sha(path: Path) -> str:
    if not path.is_file():
        return ""
    return _sha(path)


def _canonical(rows: list[dict[str, str]]) -> list[tuple[str, ...]]:
    fields = (
        "split_root", "task_name", "trajectory_id", "absolute_path", "heatmap_exists",
        "heatmap_channels", "heatmap_height", "temporal_width", "segments_exists",
        "segments_filename", "timestamp_exists", "valid_for_training", "exclusion_reason",
        "number_of_segments", "segments_sha256", "heatmap_sha256", "segments_size", "heatmap_size",
    )
    available = set(rows[0]) if rows else set()
    fields = tuple(field for field in fields if field in available)
    canonical: list[tuple[str, ...]] = []
    for row in rows:
        path = Path(row.get("absolute_path", ""))
        values = [row.get(field, "") for field in fields]
        if row.get("valid_for_training") == "True":
            values.extend([
                _file_sha(path / "segments.csv"),
                _file_sha(path / "citr_fingerprint_pure.png"),
                str((path / "segments.csv").stat().st_size if (path / "segments.csv").is_file() else ""),
                str((path / "citr_fingerprint_pure.png").stat().st_size if (path / "citr_fingerprint_pure.png").is_file() else ""),
            ])
        else:
            values.extend(["", "", "", ""])
        canonical.append(tuple(values))
    return sorted(canonical)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-1", required=True, type=Path)
    parser.add_argument("--scan-2", required=True, type=Path)
    parser.add_argument("--output", default="outputs/pour_baseline/data_stability", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory_1 = _rows(args.scan_1 / "dataset_inventory.csv")
    inventory_2 = _rows(args.scan_2 / "dataset_inventory.csv")
    canonical_1 = _canonical(inventory_1)
    canonical_2 = _canonical(inventory_2)
    valid_1 = [row for row in inventory_1 if row.get("valid_for_training") == "True"]
    valid_2 = [row for row in inventory_2 if row.get("valid_for_training") == "True"]
    referenced = {row.get("path", "") for row in _rows(args.scan_1 / "split_expansion.csv") if row.get("included") == "True"}
    referenced_rows_1 = [row for row in valid_1 if row.get("absolute_path") in referenced]
    referenced_rows_2 = [row for row in valid_2 if row.get("absolute_path") in referenced]
    stable = canonical_1 == canonical_2
    referenced_stable = _canonical(referenced_rows_1) == _canonical(referenced_rows_2)
    summary = {
        "scan_1": str(args.scan_1.resolve()),
        "scan_2": str(args.scan_2.resolve()),
        "inventory_sha256": {"scan_1": _sha(args.scan_1 / "dataset_inventory.csv"), "scan_2": _sha(args.scan_2 / "dataset_inventory.csv")},
        "inventory_match": stable,
        "valid_trajectory_count": {"scan_1": len(valid_1), "scan_2": len(valid_2)},
        "referenced_pour_paths_match": referenced_stable,
        "referenced_pour_valid_count": {"scan_1": len(referenced_rows_1), "scan_2": len(referenced_rows_2)},
        "train_wipe_w9_stable": all(
            len([row for row in records if row.get("absolute_path", "").endswith("/train/wipe/w9")]) == 1
            for records in (inventory_1, inventory_2)
        ) and _canonical([row for row in inventory_1 if row.get("absolute_path", "").endswith("/train/wipe/w9")]) == _canonical([row for row in inventory_2 if row.get("absolute_path", "").endswith("/train/wipe/w9")]),
        "file_size_dimension_and_annotation_hash_fields_match": stable,
        "proceed_pour_training": bool(stable and referenced_stable and len(referenced_rows_1) == len(referenced_rows_2)),
    }
    for name in ("inventory_scan_1.csv", "split_expansion_scan_1.csv", "raw_label_inventory_scan_1.csv", "trajectory_sequences_scan_1.csv"):
        source_name = name.replace("_scan_1", "")
        source = args.scan_1 / source_name
        if source.is_file():
            shutil.copyfile(source, args.output / name)
    shutil.copyfile(args.scan_1 / "dataset_inventory.csv", args.output / "inventory_scan_1.csv")
    shutil.copyfile(args.scan_2 / "dataset_inventory.csv", args.output / "inventory_scan_2.csv")
    (args.output / "stability_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["proceed_pour_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
