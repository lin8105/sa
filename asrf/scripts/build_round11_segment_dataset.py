#!/usr/bin/env python3
"""Build the round-11 GT segment dataset without fitting or training anything.

The source data are read-only.  Each output sample stores the source CITR
frame-feature rows for one timestamp-exclusive GT segment in a compressed NPZ
file, and each split manifest points to that NPZ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_DATA_ROOT = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
DEFAULT_OUTPUT = Path("outputs/round11_segment_embedding/data")
FEATURE_COLUMNS = (
    "citr_ff", "citr_ftau", "citr_tautau", "citr_fv", "citr_tauv",
    "citr_vv", "citr_fw", "citr_tauw", "citr_vw", "citr_ww",
    "gripper_position", "gripper_norm",
)
REQUIRED_FILES = ("segments.csv", "citr_features.csv", "citr_fingerprint_pure.png")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_features(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    fields, rows = read_csv(path / "citr_features.csv")
    expected = ["timestamp_us", *FEATURE_COLUMNS]
    if fields != expected:
        raise ValueError(f"{path}: unexpected citr_features.csv columns: {fields}")
    timestamps = np.asarray([int(row["timestamp_us"]) for row in rows], dtype=np.int64)
    values = np.asarray([[float(row[name]) for name in FEATURE_COLUMNS] for row in rows], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_COLUMNS) or not len(values):
        raise ValueError(f"{path}: empty or malformed frame features {values.shape}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{path}: feature timestamps are not strictly increasing")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: frame features contain non-finite values")
    with Image.open(path / "citr_fingerprint_pure.png") as image:
        if image.size[1] != 88 or image.size[0] != len(timestamps):
            raise ValueError(f"{path}: feature/heatmap shape mismatch: {image.size} vs {len(timestamps)}")
    return fields, timestamps, values


def validate_trajectory(path: Path, data_root: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        return {"valid": False, "path": path, "reason": f"missing_files:{','.join(missing)}"}
    fields, rows = read_csv(path / "segments.csv")
    required_segment_fields = {"segment_index", "start_timestamp_us", "end_timestamp_us_exclusive", "label"}
    if not required_segment_fields.issubset(fields):
        return {"valid": False, "path": path, "reason": "missing_timestamp_annotation_fields"}
    if not rows:
        return {"valid": False, "path": path, "reason": "empty_segments"}
    try:
        _, timestamps, values = load_features(path)
        intervals: list[tuple[int, int, int, str, int, int]] = []
        for row_number, row in enumerate(rows, start=2):
            label = (row.get("label") or "").strip()
            if not label:
                return {"valid": False, "path": path, "reason": f"blank_label_row_{row_number}"}
            start_us = int(row["start_timestamp_us"])
            end_us = int(row["end_timestamp_us_exclusive"])
            if end_us <= start_us:
                return {"valid": False, "path": path, "reason": f"nonpositive_timestamp_duration_row_{row_number}"}
            start = int(np.searchsorted(timestamps, start_us, side="left"))
            end = int(np.searchsorted(timestamps, end_us, side="left"))
            if not (0 <= start < end <= len(timestamps)):
                return {"valid": False, "path": path, "reason": f"invalid_frame_interval_row_{row_number}:{start}:{end}"}
            intervals.append((start, end, int(row["segment_index"]), label, start_us, end_us))
        ordered = sorted(intervals)
        gaps: list[str] = []
        overlaps: list[str] = []
        if ordered[0][0] != 0:
            gaps.append(f"0:{ordered[0][0]}")
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                overlaps.append(f"{previous[2]}:{current[2]}")
            elif current[0] > previous[1]:
                gaps.append(f"{previous[1]}:{current[0]}")
        if ordered[-1][1] != len(timestamps):
            gaps.append(f"{ordered[-1][1]}:{len(timestamps)}")
        if gaps or overlaps:
            return {"valid": False, "path": path, "reason": f"frame_gaps={gaps};frame_overlaps={overlaps}"}
        return {
            "valid": True, "path": path, "relative": path.relative_to(data_root).as_posix(),
            "fields": fields, "rows": rows, "timestamps": timestamps, "values": values,
            "intervals": intervals, "feature_count": len(values),
        }
    except Exception as exc:
        return {"valid": False, "path": path, "reason": f"{type(exc).__name__}:{exc}"}


def direct_children(root: Path, prefix: str) -> list[Path]:
    return sorted((root / prefix).iterdir(), key=lambda item: item.name) if (root / prefix).is_dir() else []


def first_occurrence_labels(records: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for record in records:
        for interval in record["intervals"]:
            label = interval[3]
            if label not in result:
                result.append(label)
    return result


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id", "trajectory", "source_path", "task", "segment_index", "label", "label_id",
        "start_frame", "end_frame", "end_frame_exclusive", "num_frames", "duration", "duration_s",
        "duration_frames", "duration_us", "start_timestamp_us", "end_timestamp_us_exclusive",
        "frame_feature_path", "frame_feature_shape", "frame_feature_columns", "split", "known_or_novel",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = (Path(__file__).resolve().parents[1] / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sequence_dir = output / "frame_feature_sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)

    train_paths = [data_root / "train" / "pick and place" / f"pp{i}" for i in range(1, 11)]
    validation_paths = [data_root / "train" / "pick and place" / f"pp{i}" for i in range(11, 21)]
    test_pp_candidates = direct_children(data_root, "test/pp")
    test_wipe_expected = [data_root / "test" / "wipe" / f"w{i}" for i in range(1, 5)]

    specs = [
        ("train", "train_manifest.csv", train_paths, "pick_and_place", "known"),
        ("validation", "validation_manifest.csv", validation_paths, "pick_and_place", "known"),
        ("test_pp", "test_pp_manifest.csv", test_pp_candidates, "pick_and_place", "known"),
        ("test_wipe", "test_wipe_manifest.csv", test_wipe_expected, "wipe", "novel"),
    ]
    inventories: dict[str, list[dict[str, Any]]] = {}
    missing_expected: dict[str, str] = {}
    for split, _, paths, _, _ in specs:
        records: list[dict[str, Any]] = []
        for path in paths:
            record = validate_trajectory(path, data_root)
            if record["valid"]:
                records.append(record)
            else:
                missing_expected[record["path"].relative_to(data_root).as_posix()] = record["reason"]
        inventories[split] = records
    for path in test_wipe_expected:
        if not path.is_dir():
            missing_expected[path.relative_to(data_root).as_posix()] = "trajectory_directory_missing"
    expected_test_pp_c1 = data_root / "test" / "pp" / "pp_c1"
    if not expected_test_pp_c1.is_dir():
        missing_expected[expected_test_pp_c1.relative_to(data_root).as_posix()] = "expected_reference_trajectory_directory_missing"

    ontology_labels = first_occurrence_labels(inventories["train"])
    label_to_id = {label: index for index, label in enumerate(ontology_labels)}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    label_counts: dict[str, Counter[str]] = {}
    source_hashes: dict[str, str] = {}
    for split, manifest_name, records, task, known_or_novel in specs:
        rows: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for record in inventories[split]:
            trajectory = record["relative"]
            source_hashes[trajectory] = sha256(record["path"] / "citr_features.csv")
            for local_index, (start, end, segment_index, label, start_us, end_us) in enumerate(record["intervals"]):
                duration_us = end_us - start_us
                duration_s = duration_us / 1_000_000.0
                if label not in label_to_id:
                    label_id: int | str = ""
                else:
                    label_id = label_to_id[label]
                sample_id = f"{split}__{trajectory.replace('/', '__').replace(' ', '_')}__segment_{segment_index:03d}"
                sequence_name = f"{sample_id}.npz"
                sequence_path = sequence_dir / sequence_name
                np.savez_compressed(
                    sequence_path,
                    features=record["values"][start:end].astype(np.float32, copy=False),
                    timestamp_us=record["timestamps"][start:end],
                    feature_columns=np.asarray(FEATURE_COLUMNS),
                    trajectory=np.asarray(trajectory),
                    label=np.asarray(label),
                    start_frame=np.asarray(start, dtype=np.int64),
                    end_frame_exclusive=np.asarray(end, dtype=np.int64),
                )
                row = {
                    "sample_id": sample_id, "trajectory": trajectory, "source_path": str(record["path"]),
                    "task": task, "segment_index": segment_index, "label": label, "label_id": label_id,
                    "start_frame": start, "end_frame": end - 1, "end_frame_exclusive": end, "num_frames": end - start,
                    "duration": f"{duration_s:.9f}", "duration_s": f"{duration_s:.9f}",
                    "duration_frames": end - start, "duration_us": duration_us,
                    "start_timestamp_us": start_us, "end_timestamp_us_exclusive": end_us,
                    "frame_feature_path": sequence_path.relative_to(output).as_posix(),
                    "frame_feature_shape": f"{end - start}x{len(FEATURE_COLUMNS)}",
                    "frame_feature_columns": "|".join(FEATURE_COLUMNS), "split": split,
                    "known_or_novel": known_or_novel,
                }
                rows.append(row)
                counts[label] += 1
        all_rows[split] = rows
        label_counts[split] = counts
        write_manifest(output / manifest_name, rows)

    # The ontology is deliberately based on train only.  Test-only PP labels
    # are retained in the test manifest, but do not receive known IDs.
    ontology = {
        "dataset": "round11_segment_embedding",
        "protocol": {
            "train": "train/pick and place/pp1-pp10",
            "validation": "train/pick and place/pp11-pp20",
            "test_known": "all valid direct-child test/pp trajectories",
            "test_novel": "test/wipe/w1-w4 when present and valid",
        },
        "ontology_source": [record["relative"] for record in inventories["train"]],
        "known_labels": ontology_labels,
        "label_to_id": label_to_id,
        "classes": [{"id": index, "label": label} for label, index in label_to_id.items()],
        "known_or_novel_definition": "known/novel marks the requested trajectory-family split; PP test is known and wipe test is novel.",
        "test_only_labels": sorted(set(label_counts["test_pp"]) - set(ontology_labels)),
        "feature_source": "citr_features.csv; timestamp_us is stored separately and is not a feature column",
        "encoder_fitted": False,
        "threshold_selected": False,
        "model_trained": False,
    }
    (output / "ontology.json").write_text(json.dumps(ontology, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    # Audits are generated from the written manifests and NPZ files so that
    # the report checks the delivered artifact rather than only in-memory data.
    manifest_paths = {split: output / name for split, name, _, _, _ in specs}
    audit: dict[str, Any] = {
        "status": "PASS",
        "output_dir": str(output),
        "data_root": str(data_root),
        "split_counts": {split: {"trajectories": len(inventories[split]), "segments": len(all_rows[split]), "labels": dict(sorted(label_counts[split].items()))} for split in all_rows},
        "ontology_labels": ontology_labels,
        "test_only_pp_labels": ontology["test_only_labels"],
        "expected_missing_or_invalid": missing_expected,
        "checks": {},
        "source_feature_hashes": source_hashes,
    }
    checks = audit["checks"]
    checks["no_blank_labels"] = all(bool(str(row["label"]).strip()) for rows in all_rows.values() for row in rows)
    checks["all_feature_sequences_present"] = all((output / row["frame_feature_path"]).is_file() for rows in all_rows.values() for row in rows)
    checks["all_feature_sequences_nonempty_finite"] = True
    checks["segment_frame_spans_valid"] = all(int(row["start_frame"]) >= 0 and int(row["end_frame_exclusive"]) > int(row["start_frame"]) and int(row["num_frames"]) == int(row["end_frame_exclusive"]) - int(row["start_frame"]) for rows in all_rows.values() for row in rows)
    checks["no_gaps_or_overlaps_within_trajectory"] = all(record["valid"] for records in inventories.values() for record in records)
    checks["no_trajectory_overlap_between_splits"] = not (set(record["relative"] for record in inventories["train"]) & set(record["relative"] for record in inventories["validation"]) or set(record["relative"] for record in inventories["train"]) & set(record["relative"] for record in inventories["test_pp"] + inventories["test_wipe"]) or set(record["relative"] for record in inventories["validation"]) & set(record["relative"] for record in inventories["test_pp"] + inventories["test_wipe"]))
    checks["no_duplicate_source_feature_hashes_across_splits"] = len(source_hashes) == len(set(source_hashes.values()))
    checks["wipe_absent_from_train_validation"] = all(not row["trajectory"].startswith("train/wipe/") for rows in all_rows.values() for row in rows if row["split"] in {"train", "validation"})
    checks["wipe_absent_from_ontology_source"] = all("wipe" not in record["relative"] for record in inventories["train"])
    checks["no_model_training_or_encoder_fitting"] = not ontology["model_trained"] and not ontology["encoder_fitted"] and not ontology["threshold_selected"]
    checks["manifests_written"] = all(path.is_file() for path in manifest_paths.values())
    # Check every written sequence against its manifest metadata.
    for rows in all_rows.values():
        for row in rows:
            with np.load(output / row["frame_feature_path"], allow_pickle=False) as archive:
                features = archive["features"]
                if features.shape != (int(row["num_frames"]), len(FEATURE_COLUMNS)) or not np.isfinite(features).all():
                    checks["all_feature_sequences_nonempty_finite"] = False
    checks["all_checks_pass"] = all(checks.values())
    if not checks["all_checks_pass"]:
        audit["status"] = "FAIL"
    (output / "audit_report.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_lines = [
        "# Round 11 segment-level dataset audit", "", f"Status: **{audit['status']}**", "",
        "## Protocol", "",
        "- Training ontology source: `train/pick and place/pp1`–`pp10` only.",
        "- Validation: `train/pick and place/pp11`–`pp20`.",
        "- Known test: every structurally valid direct child found under `test/pp`.",
        "- Novel test: requested `test/wipe/w1`–`w4`; only present and valid requested trajectories are emitted.",
        "- No model, encoder, threshold, or feature normalizer was fitted.", "",
        "## Delivered counts", "", "| split | trajectories | segments | labels |", "|---|---:|---:|---|",
    ]
    for split in ("train", "validation", "test_pp", "test_wipe"):
        item = audit["split_counts"][split]
        labels = ", ".join(f"{key}={value}" for key, value in item["labels"].items())
        report_lines.append(f"| {split} | {item['trajectories']} | {item['segments']} | {labels} |")
    report_lines += ["", "## Ontology", "", f"Known labels (in ID order): `{', '.join(f'{i}:{x}' for i, x in enumerate(ontology_labels))}`.", f"Test-only PP labels retained without known IDs: `{', '.join(ontology['test_only_labels']) or 'none'}`.", "", "## Integrity checks", "", "| check | result |", "|---|---|"]
    for name, result in checks.items():
        if name != "all_checks_pass":
            report_lines.append(f"| `{name}` | {'PASS' if result else 'FAIL'} |")
    report_lines += ["", "## Source inventory caveats", ""]
    if missing_expected:
        report_lines.append("The following expected candidates were absent or invalid on disk and were not fabricated:")
        for path, reason in sorted(missing_expected.items()):
            report_lines.append(f"- `{path}` — `{reason}`")
    else:
        report_lines.append("No expected candidate was absent or invalid.")
    report_lines += ["", "## Artifact layout", "", "Each manifest row points to one compressed NPZ under `frame_feature_sequences/`. Each NPZ contains `features` with shape `[num_frames, 12]`, the aligned `timestamp_us` vector, feature-column names, trajectory, label, and exclusive frame bounds.", ""]
    (output / "audit_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "output": str(output), "split_counts": audit["split_counts"], "missing_expected": missing_expected}, indent=2, sort_keys=True))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
