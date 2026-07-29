"""Read-only audit for the Round 12 multiskill segment annotations.

The audit deliberately fails if the legacy Plug label is still present. It
does not rewrite annotations or any other file under the dataset root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
OUT = ROOT / "outputs/round12_multiskill_segment_embedding/data_audit"
sys.path.insert(0, str(ROOT / "src"))

from asrf.data.dataset import load_heatmap, load_timestamp_vector  # noqa: E402
from asrf.data.ontology import (  # noqa: E402
    ALIASES,
    CANONICAL_LABELS,
    LABEL_TO_ID,
    LEGACY_MIGRATION,
    ONTOLOGY_VERSION,
)


def task_name(split: str, family: str) -> str:
    return "pp" if family in {"pp", "pick and place"} else family


def trajectory_dirs() -> list[Path]:
    result: list[Path] = []
    for split in ("train", "test"):
        for family in ("pick and place", "pp", "wipe", "pour", "plug"):
            root = DATA_ROOT / split / family
            if root.is_dir():
                result.extend(sorted(path for path in root.iterdir() if path.is_dir()))
    return result


def frame_interval(row: dict[str, str], timestamps: np.ndarray) -> tuple[int, int]:
    if "start_timestamp_us" in row:
        return (
            int(np.searchsorted(timestamps, int(row["start_timestamp_us"]), side="left")),
            int(np.searchsorted(timestamps, int(row["end_timestamp_us_exclusive"]), side="left")),
        )
    return int(row["start_frame"]), int(row["end_frame"]) + 1


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_trajectory(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    relative = path.relative_to(DATA_ROOT).as_posix()
    split, family = relative.split("/", 2)[:2]
    if family == "pick and place":
        family = "pp"
    row: dict[str, Any] = {
        "trajectory": relative, "split": split, "task": family, "trajectory_id": path.name,
        "frame_count": 0, "segment_count": 0, "feature_file_present": int((path / "citr_features.csv").is_file()),
        "heatmap_present": int((path / "citr_fingerprint_pure.png").is_file()),
        "missing_files": "", "blank_labels": 0, "unknown_labels": "", "remaining_align_annotations": 0,
        "gaps": 0, "overlaps": 0, "zero_length_segments": 0, "invalid_intervals": 0,
        "split_leakage": 0, "schema_valid": 0, "annotation_schema": "", "class_sequence": "", "error": "",
    }
    distribution: list[dict[str, Any]] = []
    missing = [name for name in ("segments.csv", "citr_features.csv", "citr_fingerprint_pure.png") if not (path / name).is_file()]
    row["missing_files"] = ";".join(missing)
    row["annotation_sha256"] = file_hash(path / "segments.csv") if (path / "segments.csv").is_file() else ""
    row["feature_sha256"] = file_hash(path / "citr_features.csv") if (path / "citr_features.csv").is_file() else ""
    if missing:
        row["error"] = "missing_required_files"
        return row, distribution
    try:
        timestamps = load_timestamp_vector(path / "citr_features.csv")
        heatmap = load_heatmap(path / "citr_fingerprint_pure.png")
        row["frame_count"] = int(len(timestamps))
        if int(heatmap.shape[-1]) != len(timestamps):
            row["error"] = "heatmap_timestamp_width_mismatch"
        fields, raw_rows = read_rows(path / "segments.csv")
        row["annotation_schema"] = ",".join(fields)
        has_timestamps = {"start_timestamp_us", "end_timestamp_us_exclusive"}.issubset(fields)
        has_frames = {"start_frame", "end_frame"}.issubset(fields)
        if "label" not in fields or has_timestamps == has_frames:
            row["error"] = "invalid_annotation_schema"
            return row, distribution
        row["schema_valid"] = 1
        row["segment_count"] = len(raw_rows)
        occupied = np.zeros(len(timestamps), dtype=bool)
        sequence: list[str] = []
        counts: Counter[str] = Counter()
        frames: Counter[str] = Counter()
        unknown: set[str] = set()
        for item in raw_rows:
            raw_label = (item.get("label") or "").strip()
            if not raw_label:
                row["blank_labels"] += 1
                continue
            if raw_label == "align":
                row["remaining_align_annotations"] += 1
            canonical = ALIASES.get(raw_label, raw_label)
            if canonical not in LABEL_TO_ID:
                unknown.add(raw_label)
            else:
                sequence.append(canonical)
                counts[canonical] += 1
            try:
                start, end = frame_interval(item, timestamps)
            except (KeyError, TypeError, ValueError):
                row["invalid_intervals"] += 1
                continue
            if end <= start:
                row["zero_length_segments"] += 1
                continue
            if start < 0 or end > len(timestamps):
                row["invalid_intervals"] += 1
            left, right = max(0, start), min(len(timestamps), end)
            if left >= right:
                row["invalid_intervals"] += 1
                continue
            if occupied[left:right].any():
                row["overlaps"] += 1
            occupied[left:right] = True
            if canonical in LABEL_TO_ID:
                frames[canonical] += right - left
        row["unknown_labels"] = ";".join(sorted(unknown))
        row["gaps"] = int((~occupied).sum())
        row["class_sequence"] = " > ".join(sequence)
        for label in CANONICAL_LABELS:
            distribution.append({"split": split, "task": family, "trajectory": relative, "label": label, "segment_count": counts[label], "frame_count": frames[label]})
    except Exception as exc:  # keep all trajectory diagnostics in the report
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row, distribution


def expected_trajectory_report(observed: set[str]) -> list[dict[str, str]]:
    expected: dict[str, list[str]] = {
        "train/pp": [f"train/pick and place/pp{i}" for i in range(1, 33)],
        "test/pp": ["test/pp/pp_c1"],
        "train/wipe": [f"train/wipe/w{i}" for i in range(1, 26)],
        "test/wipe": [f"test/wipe/w{i}" for i in range(1, 5)],
        "train/pour": [f"train/pour/p{i}" for i in range(1, 26)],
        "test/pour": [*(f"test/pour/p{i}" for i in range(1, 6)), *(f"test/pour/pr_seg{i}" for i in range(1, 5))],
        "train/plug": [f"train/plug/p{i}" for i in range(1, 13)],
        "test/plug": ["test/plug/p1", "test/plug/p2", "test/plug/p3", "test/plug/po1", "test/plug/po2", "test/plug/pl+pp"],
    }
    return [{"split_task": key, "trajectory": item, "status": "missing"} for key, values in expected.items() for item in values if item not in observed]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ontology = {
        "ontology_version": ONTOLOGY_VERSION, "labels": LABEL_TO_ID, "aliases": ALIASES,
        "num_classes": len(CANONICAL_LABELS), "legacy_migration": LEGACY_MIGRATION,
    }
    (OUT / "ontology_v2.json").write_text(json.dumps(ontology, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    for path in trajectory_dirs():
        item, rows = audit_trajectory(path)
        manifest.append(item)
        distributions.extend(rows)
    observed = {row["trajectory"] for row in manifest}
    missing = expected_trajectory_report(observed)
    split_hashes = defaultdict(set)
    for row in manifest:
        key = (row.get("annotation_sha256", ""), row.get("feature_sha256", ""))
        if all(key):
            split_hashes[key].add(row["split"])
    for row in manifest:
        key = (row.get("annotation_sha256", ""), row.get("feature_sha256", ""))
        row["split_leakage"] = int(all(key) and len(split_hashes[key]) > 1)
    manifest_fields = list(manifest[0]) if manifest else ["trajectory"]
    write_csv(OUT / "trajectory_manifest.csv", manifest, manifest_fields)
    write_csv(OUT / "class_distribution.csv", distributions, ["split", "task", "trajectory", "label", "segment_count", "frame_count"])
    available = [r for r in manifest if not r["missing_files"] and int(r["schema_valid"])]

    checks: list[dict[str, Any]] = []
    def add(name: str, status: str, count: int, details: str = "") -> None:
        checks.append({"check": name, "status": status, "count": count, "details": details})
    align_rows = [r for r in manifest if int(r["remaining_align_annotations"])]
    unknown_rows = [r for r in manifest if r["unknown_labels"]]
    blank_rows = [r for r in manifest if int(r["blank_labels"])]
    add("trajectory_count", "pass", len(manifest))
    add("available_trajectory_count", "pass", len(available))
    incomplete = [r for r in manifest if r["missing_files"]]
    add("incomplete_trajectory_directories", "info", len(incomplete), ";".join(r["trajectory"] for r in incomplete))
    allowed_missing = {"test/pp/pp_c1", "test/wipe/w4"}
    disallowed_missing = [item for item in missing if item["trajectory"] not in allowed_missing]
    missing_status = "fail" if disallowed_missing else ("info" if missing else "pass")
    missing_details = ";".join(item["trajectory"] for item in missing)
    if missing and not disallowed_missing:
        missing_details += " (documented optional absence)"
    add("missing_trajectories", missing_status, len(missing), missing_details)
    add("unknown_labels", "fail" if unknown_rows else "pass", len(unknown_rows), ";".join(f"{r['trajectory']}:{r['unknown_labels']}" for r in unknown_rows))
    add("blank_labels", "fail" if blank_rows else "pass", sum(int(r["blank_labels"]) for r in manifest), ";".join(f"{r['trajectory']}:{r['blank_labels']}" for r in blank_rows))
    add("gaps", "fail" if any(r["gaps"] for r in manifest) else "pass", sum(int(r["gaps"]) for r in manifest))
    add("overlaps", "fail" if any(r["overlaps"] for r in manifest) else "pass", sum(int(r["overlaps"]) for r in manifest))
    add("zero_length_segments", "fail" if any(r["zero_length_segments"] for r in manifest) else "pass", sum(int(r["zero_length_segments"]) for r in manifest))
    add("invalid_start_end_intervals", "fail" if any(r["invalid_intervals"] for r in manifest) else "pass", sum(int(r["invalid_intervals"]) for r in manifest), ";".join(f"{r['trajectory']}:{r['invalid_intervals']}" for r in manifest if int(r["invalid_intervals"])))
    schema_rows = [r for r in manifest if not r["missing_files"] and not int(r["schema_valid"])]
    add("canonical_annotation_schema", "fail" if schema_rows else "pass", len(schema_rows), ";".join(r["trajectory"] for r in schema_rows))
    ontology_valid = tuple(LABEL_TO_ID) == CANONICAL_LABELS and sorted(LABEL_TO_ID.values()) == list(range(len(CANONICAL_LABELS)))
    add("ontology_class_ids_and_names", "pass" if ontology_valid else "fail", len(LABEL_TO_ID), str(LABEL_TO_ID))
    add("remaining_align_annotations", "fail" if align_rows else "pass", sum(int(r["remaining_align_annotations"]) for r in manifest), ";".join(r["trajectory"] for r in align_rows))
    leakage = [r for r in manifest if r["split_leakage"]]
    add("split_leakage", "fail" if leakage else "pass", len(leakage), ";".join(r["trajectory"] for r in leakage))
    by_split = defaultdict(set)
    for row in distributions:
        if int(row["segment_count"]):
            by_split[row["split"]].add(row["label"])
    train_only = sorted(by_split["train"] - by_split["test"])
    test_only = sorted(by_split["test"] - by_split["train"])
    add("classes_train_absent_test", "info", len(train_only), ",".join(train_only))
    add("classes_test_absent_train", "info", len(test_only), ",".join(test_only))
    write_csv(OUT / "dataset_audit.csv", checks, ["check", "status", "count", "details"])

    lines = ["# Round 12 multiskill dataset audit", "", f"Ontology: `{ONTOLOGY_VERSION}` ({len(CANONICAL_LABELS)} classes).", f"Dataset root: `{DATA_ROOT}`.", "", "## Result", ""]
    failed = [item for item in checks if item["status"] == "fail"]
    lines.append("**FAIL** — one or more annotation-integrity gates failed." if failed else "**PASS**")
    lines.append("")
    lines.append("The audit is read-only. No dataset annotation was modified.")
    lines.extend(["", "## Checks", "", "| check | status | count | details |", "|---|---|---:|---|"])
    lines.extend(f"| {item['check']} | {item['status']} | {item['count']} | {item['details']} |" for item in checks)
    lines.extend(["", "## Trajectories", "", "Observed trajectories:", ""])
    grouped = defaultdict(list)
    for item in available:
        grouped[f"{item['split']}/{item['task']}"].append(item["trajectory"])
    for key in sorted(grouped):
        lines.append(f"- `{key}`: " + ", ".join(sorted(grouped[key])))
    lines.extend(["", "Expected-but-missing paths are included in the `missing_trajectories` row.", "Incomplete directories lacking required files are reported separately and are not counted as available trajectories.", "", "## Available split trajectories", "", f"- Train: {sum(item['split'] == 'train' for item in available)} trajectories.", f"- Validation: {sum(item['split'] == 'validation' for item in available)} trajectories discovered (no validation `segments.csv` is currently present).", f"- Test: {sum(item['split'] == 'test' for item in available)} trajectories.", "", "## Segment counts by family and class", "", "| split | family | class | segments | frames |", "|---|---|---|---:|---:|"])
    totals = defaultdict(lambda: [0, 0])
    for item in distributions:
        key = (item["split"], item["task"], item["label"])
        totals[key][0] += int(item["segment_count"])
        totals[key][1] += int(item["frame_count"])
    for key in sorted(totals):
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {totals[key][0]} | {totals[key][1]} |")
    lines.extend(["", "Full per-trajectory and per-class records are in `trajectory_manifest.csv` and `class_distribution.csv`.", "", "## Migration gate", "", f"Remaining align count: {sum(int(item['remaining_align_annotations']) for item in manifest)}. Any remaining legacy align annotation is an error; runtime code does not map it automatically."])
    (OUT / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
