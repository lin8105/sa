"""Audit Round 9 annotations and prepare deterministic, leakage-safe manifests.

This stage is intentionally read-only with respect to the external dataset and does not train.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
OUT = ROOT / "outputs/round9_plug_learning_curve"
SPLIT_OUT = ROOT / "splits/round9"
LABEL_PATH = ROOT / "configs/labels_multitask_plug.yaml"

import sys

sys.path.insert(0, str(ROOT / "src"))

from asrf.data.dataset import load_heatmap, load_timestamp_vector  # noqa: E402
from asrf.data.labels import load_label_mapping, normalize_label_name  # noqa: E402
from asrf.data.ontology import CANONICAL_LABELS  # noqa: E402


TASKS = ("plug", "pour", "pick and place", "wipe")
CANONICAL = CANONICAL_LABELS


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _frame_range(row: dict[str, str], timestamps: np.ndarray) -> tuple[int, int]:
    if "start_timestamp_us" in row:
        start = int(row["start_timestamp_us"])
        end = int(row["end_timestamp_us_exclusive"])
        return int(np.searchsorted(timestamps, start, side="left")), int(np.searchsorted(timestamps, end, side="left"))
    return int(row["start_frame"]), int(row["end_frame"]) + 1


def _classify_plug(sequence: list[str]) -> str:
    has_insert = "insert" in sequence
    first_release = sequence.index("release") if "release" in sequence else len(sequence)
    has_later_reach = "reach" in sequence[first_release + 1 :]
    if has_insert and has_later_reach:
        return "combined plug-in/pull-out"
    if has_insert:
        return "plug-in"
    if "place" in sequence and "release" in sequence:
        return "pull-out"
    return "invalid or ambiguous"


def audit_recording(relative: str, mapping: dict[str, int]) -> dict[str, Any]:
    path = DATA_ROOT / relative
    segments_path = path / "segments.csv"
    parts = Path(relative).parts
    task_family = "pick_and_place" if len(parts) > 1 and parts[1] in {"pick and place", "pp"} else (parts[1] if len(parts) > 1 else "unknown")
    row: dict[str, Any] = {
        "trajectory": relative,
        "task_family": task_family,
        "trajectory_id": Path(relative).name,
        "split": Path(relative).parts[0],
        "temporal_width": 0,
        "segment_count": 0,
        "canonical_sequence": "",
        "plug_category": "not_plug",
        "temporal_coverage_s": 0.0,
        "annotation_gaps": "",
        "annotation_overlaps": "",
        "invalid_durations": "",
        "blank_labels": "",
        "invalid_labels": "",
        "duplicate_intervals": "",
        "compatible_with_round12_ontology": False,
    }
    for skill in CANONICAL:
        row[f"{skill}_frames"] = 0
        row[f"{skill}_segments"] = 0
    errors: list[str] = []
    try:
        heatmap = load_heatmap(path / "citr_fingerprint_pure.png")
        timestamps = load_timestamp_vector(path / "citr_features.csv")
        row["temporal_width"] = int(heatmap.shape[-1])
        row["temporal_coverage_s"] = float((timestamps[-1] - timestamps[0]) / 1_000_000.0)
        if heatmap.shape[-1] != len(timestamps):
            errors.append("heatmap_timestamp_width_mismatch")
        fields, raw_rows = _read_rows(segments_path)
        if not {"label"}.issubset(fields):
            errors.append("missing_label_column")
        if not (({"start_timestamp_us", "end_timestamp_us_exclusive"} <= set(fields)) or ({"start_frame", "end_frame"} <= set(fields))):
            errors.append("unsupported_endpoint_columns")
        occupied = np.zeros(len(timestamps), dtype=bool)
        sequence: list[str] = []
        seen_intervals: set[tuple[int, int]] = set()
        for index, raw in enumerate(raw_rows, start=2):
            raw_label = (raw.get("label") or "").strip()
            if not raw_label:
                row["blank_labels"] = f"{row['blank_labels']};row{index}".strip(";")
                errors.append(f"blank_label_row_{index}")
                continue
            canonical = normalize_label_name(raw_label, mapping)
            if canonical not in mapping:
                row["invalid_labels"] = f"{row['invalid_labels']};{raw_label}".strip(";")
                errors.append(f"invalid_label_{raw_label}")
                continue
            try:
                start, end = _frame_range(raw, timestamps)
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid_endpoint_row_{index}")
                continue
            if end <= start:
                row["invalid_durations"] = f"{row['invalid_durations']};row{index}".strip(";")
                errors.append(f"invalid_duration_row_{index}")
                continue
            if (start, end) in seen_intervals:
                row["duplicate_intervals"] = f"{row['duplicate_intervals']};row{index}".strip(";")
                errors.append(f"duplicate_interval_row_{index}")
            seen_intervals.add((start, end))
            start_clip, end_clip = max(0, start), min(len(timestamps), end)
            if start_clip >= end_clip or occupied[start_clip:end_clip].any():
                row["annotation_overlaps"] = f"{row['annotation_overlaps']};row{index}".strip(";")
                errors.append(f"overlap_row_{index}")
            if start < 0 or end > len(timestamps):
                errors.append(f"endpoint_out_of_range_row_{index}")
            if start_clip < end_clip:
                occupied[start_clip:end_clip] = True
                row[f"{canonical}_frames"] += int(end_clip - start_clip)
            row[f"{canonical}_segments"] += 1
            sequence.append(canonical)
        gaps = np.flatnonzero(~occupied)
        if len(gaps):
            row["annotation_gaps"] = f"{int(gaps[0])}-{int(gaps[-1])}" if len(gaps) else ""
            errors.append("annotation_gap")
        row["segment_count"] = len(raw_rows)
        row["canonical_sequence"] = ">".join(sequence)
        row["plug_category"] = _classify_plug(sequence) if row["task_family"] == "plug" else "not_plug"
        if row["task_family"] == "plug" and not set(sequence) <= {"reach", "grasp", "lift", "transport", "insert", "place", "release", "retreat"}:
            errors.append("unexpected_plug_skill")
        row["compatible_with_round12_ontology"] = not errors
    except Exception as exc:  # report the scan failure rather than hiding it
        errors.append(f"load_error:{type(exc).__name__}:{exc}")
    row["errors"] = ";".join(errors)
    return row


def scan() -> list[dict[str, Any]]:
    mapping = load_label_mapping(LABEL_PATH)
    rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        for path in sorted((DATA_ROOT / split).glob("**/segments.csv")):
            relative = str(path.parent.relative_to(DATA_ROOT)).replace("\\", "/")
            if Path(relative).parts[1] in TASKS or Path(relative).parts[1] in {"pp"}:
                rows.append(audit_recording(relative, mapping))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["trajectory"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _write_split(name: str, entries: list[str]) -> None:
    path = SPLIT_OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _numeric_key(value: str) -> tuple[str, int, str]:
    stem = Path(value).name
    prefix = "".join(character for character in stem if not character.isdigit())
    digits = "".join(character for character in stem if character.isdigit())
    return prefix, int(digits or 0), value


def _family_entries(rows: list[dict[str, Any]], split: str, family: str) -> list[str]:
    return sorted([row["trajectory"] for row in rows if row["split"] == split and row["task_family"] == family and row["compatible_with_round12_ontology"]], key=_numeric_key)


def make_manifests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    plug_train = _family_entries(rows, "train", "plug")
    plug_test = _family_entries(rows, "test", "plug")
    plug_val_pool = list(plug_train)
    if len(plug_train) >= 3:
        plug_orders = {
            42: ["train/plug/p1", "train/plug/p3", "train/plug/p5", "train/plug/p2", "train/plug/p4"],
            43: ["train/plug/p2", "train/plug/p4", "train/plug/p1", "train/plug/p5", "train/plug/p3"],
            44: ["train/plug/p3", "train/plug/p5", "train/plug/p2", "train/plug/p4", "train/plug/p1"],
        }
        plug_orders = {seed: [entry for entry in order if entry in plug_train] for seed, order in plug_orders.items()}
        for seed, order in plug_orders.items():
            _write_split(f"plug_train_3_seed{seed}.txt", order[:3])
            _write_split(f"plug_train_5_seed{seed}.txt", order[:5])
        _write_split("plug_train_3.txt", plug_orders[42][:3])
        _write_split("plug_train_5.txt", plug_orders[42][:5])
        _write_split("plug_train_all.txt", plug_train)
    _write_split("plug_val.txt", plug_val_pool)
    _write_split("plug_test.txt", plug_test)

    family_specs = {
        "pour": ((3, 5, 8, "all"), "pour_train", "pour_val", "pour_test"),
        "pick_and_place": ((3, 5, 10, 20, "all"), "pp_train", "pp_val", "pp_test"),
        "wipe": ((3, 5, "all"), "wipe_train", "wipe_val", "wipe_test"),
    }
    source_family = {"pick_and_place": "pick and place"}
    official_train_entries = [line.strip() for line in (ROOT / "splits/multitask_train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    family_manifests: dict[str, dict[str, Any]] = {}
    for family, (sizes, _, val_name, test_name) in family_specs.items():
        raw_family = source_family.get(family, family)
        entries = [entry for entry in official_train_entries if any(row["trajectory"] == entry and row["task_family"] == family and row["compatible_with_round12_ontology"] for row in rows)]
        val = _family_entries(rows, "test", raw_family) if False else []
        # Preserve the existing official train/validation policy.
        existing_val = {"pour": "splits/multitask_val.txt", "pick_and_place": "splits/multitask_val.txt", "wipe": "splits/multitask_val.txt"}[family]
        all_existing_val = [line.strip() for line in (ROOT / existing_val).read_text(encoding="utf-8").splitlines() if line.strip()]
        val = [entry for entry in all_existing_val if any(row["trajectory"] == entry and row["task_family"] == family and row["compatible_with_round12_ontology"] for row in rows)]
        test = _family_entries(rows, "test", family)
        _write_split(f"{family}_val.txt", val)
        _write_split(f"{family}_test.txt", test)
        family_manifests[family] = {"train_pool": entries, "validation": val, "test": test, "requested_sizes": list(sizes)}
        if family == "pour":
            order = entries
        elif family == "pick_and_place":
            order = entries
        else:
            order = entries
        for seed in (42, 43, 44):
            shuffled = list(order)
            random.Random(seed).shuffle(shuffled)
            previous = 0
            for size in sizes:
                if size == "all":
                    chosen = order
                else:
                    chosen = shuffled[: int(size)]
                if size in (3, 5) or size == "all" or (family == "pour" and size == 8) or (family == "pick_and_place" and size in (10, 20)):
                    _write_split(f"{family}_train_{size}_seed{seed}.txt", chosen)
                previous = int(size) if size != "all" else len(order)
        # Canonical seed-42 names are convenient aliases for the required primary manifests.
        for size in sizes:
            source = SPLIT_OUT / f"{family}_train_{size}_seed42.txt"
            if source.is_file():
                (SPLIT_OUT / f"{family}_train_{size}.txt").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "plug_train": plug_train,
        "plug_val_pool": plug_val_pool,
        "plug_test": plug_test,
        "family_train_counts": {family: len(details["train_pool"]) for family, details in family_manifests.items()} | {"plug": len(plug_train)},
        "family_validation_counts": {family: len(details["validation"]) for family, details in family_manifests.items()},
        "family_test_counts": {family: len(_family_entries(rows, "test", family)) for family in ("pour", "pick_and_place", "wipe", "plug")},
        "family_manifests": family_manifests,
        "split_policy": "Existing official task-family train/validation manifests are preserved. Plug has no independent validation pool; plug_val.txt is the complete train/plug leave-one-trajectory-out validation pool. Test manifests are fixed external test trajectories.",
    }


def estimate_runs(manifest: dict[str, Any]) -> dict[str, Any]:
    # Repetitions are three where distinct subsets are available; a full subset is one run.
    plan = {
        "plug": {"n3": 3, "n5": 1, "n_all": 1, "random_init_all_secondary": 1},
        "pour": {"n3": 3, "n5": 3, "n8": 1, "n_all": 1},
        "pick_and_place": {"n3": 3, "n5": 3, "n10": 1, "n20": 1, "n_all": 1},
        "wipe": {"n3": 3, "n5": 1, "n_all": 1},
    }
    primary = sum(sum(values.values()) for values in plan.values())
    round8_r5_summary = ROOT / "outputs/brb_release_round8/hard_window_r5/training_summary.json"
    reference_seconds = float(json.loads(round8_r5_summary.read_text()).get("elapsed_seconds", 988.119)) if round8_r5_summary.is_file() else 988.119
    return {
        "plan": plan,
        "primary_model_count": primary - 1,
        "secondary_random_initialization_count": 1,
        "total_model_count_including_secondary": primary,
        "reference_round8_r5_duration_s": reference_seconds,
        "estimated_training_duration_s": primary * reference_seconds,
        "estimated_training_duration_h": primary * reference_seconds / 3600.0,
        "exceeds_confirmation_threshold_20": primary > 20,
        "note": "Estimate uses the completed Round 8 hard_window_r5 CPU duration as a conservative per-model reference and excludes evaluation/plot overhead.",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows1 = scan()
    rows2 = scan()
    fields = list(rows1[0])
    digest1 = _write_csv(OUT / "data_audit_scan1.csv", rows1)
    digest2 = _write_csv(OUT / "data_audit_scan2.csv", rows2)
    stable = rows1 == rows2
    errors = [row for row in rows1 if row.get("errors")]
    summary = {
        "scan_rows_identical": stable,
        "scan1_sha256": digest1,
        "scan2_sha256": digest2,
        "row_count": len(rows1),
        "invalid_row_count": len(errors),
        "invalid_trajectories": [row["trajectory"] for row in errors],
        "all_annotations_compatible": not errors,
        "plug_categories": Counter(row["plug_category"] for row in rows1 if row["task_family"] == "plug"),
        "plug_sequences": {row["trajectory"]: row["canonical_sequence"] for row in rows1 if row["task_family"] == "plug"},
        "ontology": list(CANONICAL),
        "aliases": {"pick": "reach", "translation": "transport", "pull_out": "lift", "extract": "lift"},
        "external_dataset_modified": False,
        "pass": bool(stable and not errors),
    }
    (OUT / "data_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = make_manifests(rows1)
    manifest["audit_pass"] = summary["pass"]
    manifest["test_trajectories_fixed"] = True
    manifest["no_train_test_overlap"] = not (set(manifest["plug_train"]) & set(manifest["plug_test"]))
    manifest["no_validation_test_overlap"] = not (set(manifest["plug_val_pool"]) & set(manifest["plug_test"]))
    (OUT / "test_split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    estimate = estimate_runs(manifest)
    (OUT / "stage1_run_plan.json").write_text(json.dumps(estimate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit_pass": summary["pass"], "rows": len(rows1), "plug_train": manifest["plug_train"], "plug_test": manifest["plug_test"], **estimate}, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
