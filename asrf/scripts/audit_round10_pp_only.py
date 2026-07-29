"""Read-only audit and manifest generation for the Round 10 PP-only study."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
OUT = ROOT / "outputs/round10_pp_only_novel_segmentation/audit"
KNOWN = {"reach", "grasp", "lift", "transport", "place", "release", "retreat"}
CANONICAL = {
    "reach", "grasp", "lift", "transport", "pour", "pour_recover", "place",
    "release", "wipe", "retreat", "insert",
}
ALIASES = {"pick": "reach", "translation": "transport", "pull_out": "lift", "extract": "lift"}


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def canonical(raw: str) -> str:
    name = str(raw).strip()
    return ALIASES.get(name, name)


def parse_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_one(entry: str, *, expected_known: bool) -> dict[str, Any]:
    path = DATA / entry
    row: dict[str, Any] = {
        "trajectory": entry,
        "split": entry.split("/", 1)[0],
        "family": entry.split("/")[1] if len(entry.split("/")) > 1 else "",
        "trajectory_id": path.name,
        "path_exists": path.is_dir(),
        "segments_exists": False,
        "features_exists": (path / "citr_features.csv").is_file(),
        "heatmap_exists": (path / "citr_fingerprint_pure.png").is_file(),
        "annotation_format": "",
        "temporal_width": 0,
        "temporal_coverage_s": 0.0,
        "segment_count": 0,
        "canonical_sequence": "",
        "labels": "",
        "blank_label_count": 0,
        "invalid_label_count": 0,
        "invalid_labels": "",
        "zero_duration_count": 0,
        "gap_count": 0,
        "overlap_count": 0,
        "chronological": False,
        "full_temporal_coverage": False,
        "heatmap_width_matches": False,
        "timestamp_valid": False,
        "trajectory_type": "invalid",
        "ontology_compatible": False,
        "expected_known_only": bool(expected_known),
        "error": "",
    }
    errors: list[str] = []
    try:
        if not path.is_dir():
            raise ValueError("trajectory directory missing")
        segments_path = path / "segments.csv"
        if not segments_path.is_file():
            raise ValueError("segments.csv missing")
        row["segments_exists"] = True
        feature_path = path / "citr_features.csv"
        timestamps: list[int] = []
        if not feature_path.is_file():
            errors.append("citr_features.csv missing")
        else:
            with feature_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if "timestamp_us" not in (reader.fieldnames or []):
                    errors.append("timestamp_us column missing")
                else:
                    timestamps = [int((item.get("timestamp_us") or "").strip()) for item in reader]
                    row["timestamp_valid"] = bool(timestamps) and all(b > a for a, b in zip(timestamps, timestamps[1:]))
        if not timestamps or not row["timestamp_valid"]:
            errors.append("timestamps missing or not strictly increasing")
        rows = parse_rows(segments_path)
        row["segment_count"] = len(rows)
        if not rows:
            errors.append("segments.csv empty")
        labels: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        previous_end: int | None = None
        for raw in rows:
            raw_label = raw.get("label", "")
            if not raw_label or not raw_label.strip():
                row["blank_label_count"] += 1
                errors.append("blank label")
                labels.append("")
                continue
            name = canonical(raw_label)
            labels.append(name)
            if name not in CANONICAL or (expected_known and name not in KNOWN):
                row["invalid_label_count"] += 1
                errors.append(f"invalid label {raw_label!r} -> {name!r}")
            try:
                if "start_timestamp_us" in raw:
                    start = int(raw["start_timestamp_us"])
                    end = int(raw["end_timestamp_us_exclusive"])
                    row["annotation_format"] = "timestamp"
                else:
                    start = int(raw["start_frame"])
                    end = int(raw["end_frame"]) + 1
                    row["annotation_format"] = "frame"
            except (TypeError, ValueError, KeyError):
                errors.append("invalid segment endpoint")
                continue
            starts.append(start); ends.append(end)
            if end <= start:
                row["zero_duration_count"] += 1
                errors.append(f"non-positive duration [{start},{end})")
            if previous_end is not None:
                if start > previous_end:
                    row["gap_count"] += 1
                elif start < previous_end:
                    row["overlap_count"] += 1
            previous_end = end
        row["chronological"] = bool(starts) and all(b >= a for a, b in zip(starts, starts[1:]))
        if not row["chronological"]:
            errors.append("segment order is not chronological")
        if row["gap_count"]:
            errors.append("annotation gap")
        if row["overlap_count"]:
            errors.append("annotation overlap")
        if timestamps and starts and row["annotation_format"] == "timestamp":
            frame_starts = [int(np.searchsorted(timestamps, value, side="left")) for value in starts]
            frame_ends = [int(np.searchsorted(timestamps, value, side="left")) for value in ends]
            row["full_temporal_coverage"] = frame_starts[0] == 0 and frame_ends[-1] == len(timestamps) and all(
                a == b for a, b in zip(frame_ends[:-1], frame_starts[1:])
            )
            if not row["full_temporal_coverage"]:
                errors.append("converted annotation coverage does not span all heatmap columns")
            row["temporal_coverage_s"] = (timestamps[-1] - timestamps[0]) / 1e6
        elif starts and row["annotation_format"] == "frame":
            row["full_temporal_coverage"] = starts[0] == 0 and ends[-1] == len(timestamps)
            if not row["full_temporal_coverage"]:
                errors.append("frame annotation coverage does not span heatmap width")
        row["temporal_width"] = len(timestamps)
        if row["heatmap_exists"]:
            from PIL import Image
            with Image.open(path / "citr_fingerprint_pure.png") as image:
                row["heatmap_width_matches"] = image.size[0] == len(timestamps) and image.size[1] == 88
            if not row["heatmap_width_matches"]:
                errors.append("heatmap dimensions do not match [88,T]")
        else:
            errors.append("citr_fingerprint_pure.png missing")
        row["canonical_sequence"] = " -> ".join(labels)
        row["labels"] = ";".join(sorted(set(labels)))
        row["invalid_labels"] = ";".join(sorted({label for label in labels if label not in CANONICAL or (expected_known and label not in KNOWN)}))
        if "insert" in labels:
            row["trajectory_type"] = "plug-in"
        elif "lift" in labels and "insert" not in labels:
            row["trajectory_type"] = "pull-out-or-standard"
        else:
            row["trajectory_type"] = "standard-or-other"
        row["ontology_compatible"] = not any(
            name not in CANONICAL or (expected_known and name not in KNOWN) for name in labels
        )
    except Exception as exc:  # audit must record exact failures without writing external data
        errors.append(f"{type(exc).__name__}: {exc}")
    row["error"] = " | ".join(dict.fromkeys(errors))
    row["valid"] = not errors and bool(row["ontology_compatible"])
    return row


def entries_for(split: str, family: str) -> list[str]:
    parent = DATA / split / family
    return [f"{split}/{family}/{path.name}" for path in sorted(parent.iterdir(), key=natural_key) if path.is_dir()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["trajectory", "valid", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = entries_for("train", "pick and place") + entries_for("test", "pour") + entries_for("test", "wipe") + entries_for("test", "plug")
    first = [audit_one(entry, expected_known=entry.startswith("train/pick and place/")) for entry in entries]
    second = [audit_one(entry, expected_known=entry.startswith("train/pick and place/")) for entry in entries]
    fields = list(first[0])
    write_csv(OUT / "scan1.csv", first)
    write_csv(OUT / "scan2.csv", second)
    if first != second:
        raise SystemExit("two audit scans disagree")
    by_entry = {row["trajectory"]: row for row in first}
    train = [f"train/pick and place/pp{i}" for i in range(1, 11)]
    validation = [f"train/pick and place/pp{i}" for i in range(11, 21)]
    test_families: dict[str, list[str]] = {}
    for family in ("pour", "wipe", "plug"):
        candidates = [entry for entry in entries_for("test", family) if by_entry[entry]["valid"]]
        test_families[family] = candidates
    summary = {
        "scan_count": 2,
        "scans_identical": True,
        "known_ontology": ["reach", "grasp", "lift", "transport", "place", "release", "retreat"],
        "canonical_test_ontology": sorted(CANONICAL),
        "train_pp_valid": all(by_entry[item]["valid"] for item in train),
        "validation_pp_valid": all(by_entry[item]["valid"] for item in validation),
        "test_valid_entries": test_families,
        "test_invalid_entries": [entry for entry in entries if entry.startswith("test/") and not by_entry[entry]["valid"]],
        "plug_restriction": "p1 and p2 only" if test_families["plug"] == ["test/plug/p1", "test/plug/p2"] else "audit-derived",
        "all_rows_valid": all(row["valid"] for row in first),
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "pp_train_manifest.json").write_text(json.dumps({"split": "train", "entries": train, "ontology": sorted(KNOWN)}, indent=2) + "\n", encoding="utf-8")
    (OUT / "pp_validation_manifest.json").write_text(json.dumps({"split": "validation", "entries": validation, "ontology": sorted(KNOWN)}, indent=2) + "\n", encoding="utf-8")
    (OUT / "pp_train_manifest.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (OUT / "pp_validation_manifest.txt").write_text("\n".join(validation) + "\n", encoding="utf-8")
    (OUT / "test_manifest.json").write_text(json.dumps({"families": test_families, "excluded": summary["test_invalid_entries"], "selection_frozen_after_audit": True}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_pp": train, "validation_pp": validation, "test": test_families, "all_rows_valid": summary["all_rows_valid"]}, sort_keys=True))
    for row in first:
        if row["trajectory"].startswith("test/plug/"):
            print(json.dumps({"trajectory": row["trajectory"], "valid": row["valid"], "error": row["error"], "sequence": row["canonical_sequence"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
