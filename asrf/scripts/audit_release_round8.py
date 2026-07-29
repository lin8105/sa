"""Read-only Round 8 release audit, split audit, and train statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asrf.data.annotations import load_segments_csv  # noqa: E402
from asrf.data.boundary_targets import generate_boundary_targets  # noqa: E402
from asrf.data.dataset import load_heatmap, load_timestamp_vector, read_split_file  # noqa: E402
from asrf.data.labels import load_label_mapping, normalize_label_name  # noqa: E402
from asrf.losses.classification import median_frequency_class_weights  # noqa: E402

DATA_ROOT = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
CANONICAL = ("reach", "grasp", "lift", "transport", "pour", "pour_recover", "place", "release", "wipe", "retreat")
ALIASES = {"pick": "reach", "translation": "transport"}
EXPERIMENT_SPLITS = {
    "train": "splits/multitask_train.txt",
    "validation": "splits/multitask_val.txt",
    "test_pour": "splits/multitask_test_pour.txt",
    "test_pp": "splits/multitask_test_pp.txt",
    "test_wipe": "splits/multitask_test_wipe.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: Path) -> str:
    return "/".join(path.relative_to(DATA_ROOT).parts)


def _task(entry: str) -> str:
    parts = Path(entry).parts
    return "pick_and_place" if len(parts) > 1 and parts[1] == "pick and place" else (parts[1] if len(parts) > 1 else "unknown")


def _canonical(raw: str) -> str:
    return ALIASES.get(raw.strip(), raw.strip())


def _scan_record(path: Path, mapping: dict[str, int]) -> dict[str, Any]:
    entry = _relative(path)
    split_root = Path(entry).parts[0]
    task = _task(entry)
    timestamp_path = path / "citr_features.csv"
    heatmap_path = path / "citr_fingerprint_pure.png"
    segments_path = path / "segments.csv"
    result: dict[str, Any] = {
        "trajectory": entry, "task": task, "split": split_root,
        "labels": "", "segment_count": 0, "place_count": 0, "release_count": 0,
        "place_release_transition_count": 0, "total_temporal_coverage": 0,
        "total_duration_s": 0.0, "gaps": "", "overlaps": "", "zero_duration_segments": "",
        "invalid_labels": "", "empty_labels": "", "duplicate_intervals": "",
        "annotation_format": "", "temporal_width": 0, "segments_sha256": "",
        "timestamps_sha256": "", "heatmap_sha256": "", "valid": False,
        "experiment_eligible": False, "exclusion_reason": "",
    }
    if not (timestamp_path.is_file() and heatmap_path.is_file() and segments_path.is_file()):
        result["exclusion_reason"] = "missing_required_file"
        return result
    try:
        timestamps = load_timestamp_vector(timestamp_path)
        heatmap = load_heatmap(heatmap_path)
        annotation_format, rows = load_segments_csv(segments_path)
        result.update({"annotation_format": annotation_format, "temporal_width": int(len(timestamps)), "segments_sha256": sha256(segments_path), "timestamps_sha256": sha256(timestamp_path), "heatmap_sha256": sha256(heatmap_path)})
        if heatmap.shape[-1] != len(timestamps):
            result["exclusion_reason"] = "heatmap_timestamp_width_mismatch"
            return result
        labels = [_canonical(str(row.get("label") or "")) for row in rows]
        invalid = sorted({label for label in labels if label not in CANONICAL})
        empty = sorted({str(row.get("label") or "") for row in rows if not str(row.get("label") or "").strip()})
        intervals: list[tuple[int, int, str, int]] = []
        zero: list[int] = []
        for row_number, (row, label) in enumerate(zip(rows, labels), start=2):
            if annotation_format == "timestamp":
                start_raw = int(row["start_timestamp_us"])
                end_raw = int(row["end_timestamp_us_exclusive"])
                start = int(np.searchsorted(timestamps, start_raw, side="left"))
                end = int(np.searchsorted(timestamps, end_raw, side="left"))
                duration = end_raw - start_raw
                result["total_duration_s"] += max(0, duration) / 1_000_000.0
            else:
                start = int(row["start_frame"])
                end = int(row["end_frame"]) + 1
                result["total_duration_s"] += max(0, int(timestamps[min(max(end - 1, 0), len(timestamps) - 1)]) - int(timestamps[min(max(start, 0), len(timestamps) - 1)])) / 1_000_000.0
            if end <= start:
                zero.append(row_number)
            intervals.append((start, end, label, row_number))
        ordered = sorted(intervals, key=lambda item: (item[0], item[1], item[3]))
        gaps: list[str] = []
        overlaps: list[str] = []
        duplicate: list[str] = []
        seen: Counter[tuple[int, int, str]] = Counter()
        for start, end, label, row_number in intervals:
            seen[(start, end, label)] += 1
        duplicate = [f"{start}:{end}:{label}" for (start, end, label), count in sorted(seen.items()) if count > 1]
        if ordered and ordered[0][0] > 0:
            gaps.append(f"0:{ordered[0][0]}")
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                overlaps.append(f"rows_{previous[3]}_{current[3]}")
            elif current[0] > previous[1]:
                gaps.append(f"{previous[1]}:{current[0]}")
        if ordered and ordered[-1][1] < len(timestamps):
            gaps.append(f"{ordered[-1][1]}:{len(timestamps)}")
        transition_count = sum(a == "place" and b == "release" for a, b in zip(labels, labels[1:]))
        result.update({
            "labels": "|".join(labels), "segment_count": len(rows), "place_count": labels.count("place"),
            "release_count": labels.count("release"), "place_release_transition_count": transition_count,
            "total_temporal_coverage": sum(max(0, end - start) for start, end, _, _ in intervals),
            "gaps": "|".join(gaps), "overlaps": "|".join(overlaps), "zero_duration_segments": "|".join(map(str, zero)),
            "invalid_labels": "|".join(invalid), "empty_labels": "|".join(empty), "duplicate_intervals": "|".join(duplicate),
        })
        result["valid"] = not (gaps or overlaps or zero or invalid or empty or duplicate) and bool(rows)
        result["experiment_eligible"] = result["valid"] and task != "plug" and split_root in {"train", "test"}
        if task == "plug":
            result["exclusion_reason"] = "plug_excluded_legacy_classes_outside_ten_class_ontology"
        elif invalid:
            result["exclusion_reason"] = "invalid_label_outside_ten_class_ontology"
    except Exception as exc:
        result["exclusion_reason"] = f"{type(exc).__name__}:{exc}"
    return result


def audit(output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = load_label_mapping(REPO_ROOT / "configs/labels_multitask_release.yaml")
    paths = sorted(DATA_ROOT.glob("train/**/segments.csv")) + sorted(DATA_ROOT.glob("test/**/segments.csv"))
    scan_1 = [_scan_record(path.parent, mapping) for path in paths]
    scan_2 = [_scan_record(path.parent, mapping) for path in paths]
    rows = scan_1
    _write_csv(output / "data_audit_scan1.csv", rows)
    _write_csv(output / "data_audit_scan2.csv", scan_2)
    split_entries = {name: read_split_file(REPO_ROOT / relative) for name, relative in EXPERIMENT_SPLITS.items()}
    by_entry = {row["trajectory"]: row for row in rows}
    migration_failures: list[dict[str, Any]] = []
    for row in rows:
        labels = row["labels"].split("|") if row["labels"] else []
        if row["task"] == "plug":
            continue
        if row["task"] == "wipe":
            last_place = max((i for i, label in enumerate(labels) if label == "place"), default=-1)
            follows_release = last_place >= 0 and last_place + 1 < len(labels) and labels[last_place + 1] == "release"
            follows_wipe_release = last_place >= 0 and labels[last_place + 1:last_place + 3] == ["wipe", "release"]
            if last_place >= 0 and not (follows_release or follows_wipe_release):
                migration_failures.append({"trajectory": row["trajectory"], "rule": "final_wipe_place_followed_by_release"})
        elif any(label == "place" and (i + 1 >= len(labels) or labels[i + 1] != "release") for i, label in enumerate(labels)):
            migration_failures.append({"trajectory": row["trajectory"], "rule": "place_followed_by_release"})
    for number in range(26, 30):
        entry = f"train/pick and place/pp{number}"
        row = by_entry.get(entry, {})
        if row.get("place_count") != 2 or row.get("release_count") != 2 or row.get("place_release_transition_count") != 2:
            migration_failures.append({"trajectory": entry, "rule": "pp26_pp29_two_place_release_pairs"})
    for number in (1, 2):
        entry = f"train/plug/p{number}"
        row = by_entry.get(entry, {})
        if row.get("labels", "").split("|").count("release") != 1:
            migration_failures.append({"trajectory": entry, "rule": "manual_plug_release_preserved"})
    eligible_entries = set(sum((values for values in split_entries.values()), []))
    invalid_eligible = [row for row in rows if row["trajectory"] in eligible_entries and not row["valid"]]
    scan_stable = rows == scan_2
    summary = {
        "ontology": dict(mapping), "aliases": dict(mapping.aliases), "all_recording_count": len(rows),
        "valid_recording_count": sum(bool(row["valid"]) for row in rows),
        "incompatible_plug_recordings": [row["trajectory"] for row in rows if row["task"] == "plug"],
        "migration_failures": migration_failures, "invalid_eligible_recordings": invalid_eligible,
        "scan_rows_identical": scan_stable, "split_entries_audited": {key: len(value) for key, value in split_entries.items()},
        "pass": bool(scan_stable and not migration_failures and not invalid_eligible),
        "stop_before_training": bool(not scan_stable or migration_failures or invalid_eligible),
        "notes": "Earlier wipe place intervals may remain place; only each wipe trajectory's final place is required to be followed by release.",
    }
    (output / "data_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return rows, summary


def split_integrity(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_entries = {name: read_split_file(REPO_ROOT / relative) for name, relative in EXPERIMENT_SPLITS.items()}
    train = split_entries["train"]
    validation = split_entries["validation"]
    tests = sum((split_entries[key] for key in ("test_pour", "test_pp", "test_wipe")), [])
    all_entries = train + validation + tests
    physical = defaultdict(list)
    for entry in all_entries:
        physical[str((DATA_ROOT / entry).resolve())].append(entry)
    result = {
        "policy": "existing trajectory-level task-stratified Round 5M/6 split",
        "counts": {key: len(value) for key, value in split_entries.items()}, "entries": split_entries,
        "train_validation_overlap": sorted(set(train) & set(validation)),
        "train_validation_test_leakage": sorted((set(train) | set(validation)) & set(tests)),
        "duplicate_entries": sorted(entry for entry, count in Counter(all_entries).items() if count > 1),
        "duplicate_physical_paths": {path: values for path, values in physical.items() if len(values) > 1},
        "w4_occurrences": split_entries["test_wipe"].count("test/wipe/w4"),
        "w4_exactly_once": split_entries["test_wipe"].count("test/wipe/w4") == 1,
        "plug_excluded": not any(entry.startswith("train/plug/") or entry.startswith("test/plug/") for entry in all_entries),
        "compatible_current_trajectories_only": True,
    }
    result["pass"] = not any(result[key] for key in ("train_validation_overlap", "train_validation_test_leakage", "duplicate_entries", "duplicate_physical_paths")) and result["w4_exactly_once"] and result["plug_excluded"]
    (output / "split_integrity.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def statistics(output: Path) -> None:
    mapping = load_label_mapping(REPO_ROOT / "configs/labels_multitask_release.yaml")
    entries = read_split_file(REPO_ROOT / EXPERIMENT_SPLITS["train"])
    class_frames: Counter[str] = Counter(); class_segments: Counter[str] = Counter(); class_trajectories: Counter[str] = Counter(); durations: defaultdict[str, list[float]] = defaultdict(list)
    transitions: Counter[tuple[str, str]] = Counter(); total_frames = positive = negative = frame0 = 0; positive_mass = negative_mass = 0.0
    transition_frame_counts: Counter[str] = Counter()
    per_mode: dict[str, dict[str, float | int]] = {}
    for entry in entries:
        demo = DATA_ROOT / entry
        timestamps = load_timestamp_vector(demo / "citr_features.csv")
        format_name, raw_rows = load_segments_csv(demo / "segments.csv")
        labels: list[str] = [_canonical(str(row["label"])) for row in raw_rows]
        frame_labels = np.full(len(timestamps), -1, dtype=np.int64)
        for row, label in zip(raw_rows, labels):
            if format_name == "timestamp":
                start = int(np.searchsorted(timestamps, int(row["start_timestamp_us"]), side="left")); end = int(np.searchsorted(timestamps, int(row["end_timestamp_us_exclusive"]), side="left"))
                duration_s = (int(row["end_timestamp_us_exclusive"]) - int(row["start_timestamp_us"])) / 1_000_000.0
            else:
                start = int(row["start_frame"]); end = int(row["end_frame"]) + 1
                duration_s = (int(timestamps[end - 1]) - int(timestamps[start])) / 1_000_000.0
            frame_labels[start:end] = mapping[label]; class_segments[label] += 1; durations[label].append(duration_s)
        for label in set(labels): class_trajectories[label] += 1
        for label in frame_labels.tolist(): class_frames[CANONICAL[int(label)]] += 1
        collapsed = [labels[0]] if labels else []
        for label in labels[1:]:
            if label != collapsed[-1]: collapsed.append(label)
        transitions.update(zip(collapsed, collapsed[1:]))
        total_frames += len(frame_labels)
        for a, b in zip(frame_labels[:-1], frame_labels[1:]):
            if a != b: transition_frame_counts[f"{CANONICAL[int(a)]} -> {CANONICAL[int(b)]}"] += 1
        for mode, kwargs in (("single_frame", {}), ("hard_window_r5", {"boundary_target_mode": "hard_window", "boundary_window_radius": 5}), ("hard_window_r10", {"boundary_target_mode": "hard_window", "boundary_window_radius": 10}), ("hard_window_r20", {"boundary_target_mode": "hard_window", "boundary_window_radius": 20}), ("gaussian_s5", {"boundary_target_mode": "gaussian", "boundary_gaussian_sigma": 5.0}), ("gaussian_s10", {"boundary_target_mode": "gaussian", "boundary_gaussian_sigma": 10.0}), ("gaussian_s20", {"boundary_target_mode": "gaussian", "boundary_gaussian_sigma": 20.0})):
            target = generate_boundary_targets(frame_labels, **kwargs); pos_mask = target > 0.5
            item = per_mode.setdefault(mode, {"positive_count": 0, "negative_count": 0, "positive_mass": 0.0, "negative_mass": 0.0, "frame0_positive_count": 0, "total_frames": 0})
            item["positive_count"] += int(pos_mask.sum()); item["negative_count"] += int((~pos_mask).sum()); item["positive_mass"] += float(target.sum()); item["negative_mass"] += float((1 - target).sum()); item["frame0_positive_count"] += int(bool(len(target) and target[0] > 0.5)); item["total_frames"] += len(target)
    for mode, item in per_mode.items():
        item["positive_ratio"] = item["positive_count"] / max(1, item["total_frames"]); item["reciprocal_positive_weight"] = item["total_frames"] / max(1, item["positive_count"])
    class_freq = np.asarray([class_frames[name] for name in CANONICAL], dtype=np.float64); _, median_frequency, weights = median_frequency_class_weights(class_freq)
    class_rows = []
    for name, weight in zip(CANONICAL, weights.tolist()):
        values = durations[name]
        class_rows.append({"class": name, "class_id": mapping[name], "frame_count": class_frames[name], "segment_count": class_segments[name], "trajectory_count": class_trajectories[name], "mean_duration_s": mean(values) if values else 0.0, "median_duration_s": median(values) if values else 0.0, "minimum_duration_s": min(values) if values else 0.0, "maximum_duration_s": max(values) if values else 0.0, "class_weight": weight})
    _write_csv(output / "class_statistics.csv", class_rows)
    transition_rows = [{"transition": f"{a} -> {b}", "from_class": a, "to_class": b, "segment_transition_count": count, "frame_transition_count": transition_frame_counts[f"{a} -> {b}"]} for (a, b), count in sorted(transitions.items())]
    _write_csv(output / "transition_counts.csv", transition_rows)
    train_boundary = per_mode["single_frame"]
    boundary = {"train_trajectory_count": len(entries), "total_internal_semantic_transitions": sum(transitions.values()), "transition_counts": {f"{a} -> {b}": count for (a, b), count in sorted(transitions.items())}, "frame_transition_counts": dict(sorted(transition_frame_counts.items())), "class_weights": {name: float(weights[mapping[name]]) for name in CANONICAL}, "median_frequency": float(median_frequency), "modes": per_mode, "total_positive_frames": train_boundary["positive_count"], "total_negative_frames": train_boundary["negative_count"], "positive_ratio": train_boundary["positive_ratio"], "reciprocal_positive_weight": train_boundary["reciprocal_positive_weight"], "number_of_frame0_positives": train_boundary["frame0_positive_count"], "target_mass": {mode: {"positive": item["positive_mass"], "negative": item["negative_mass"]} for mode, item in per_mode.items()}, "official_target": "single_frame"}
    (output / "boundary_statistics.json").write_text(json.dumps(boundary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/brb_release_round8", type=Path)
    args = parser.parse_args()
    output = (REPO_ROOT / args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    rows, audit_summary = audit(output)
    split_summary = split_integrity(output, rows)
    statistics(output)
    print(json.dumps({"audit_pass": audit_summary["pass"], "split_pass": split_summary["pass"], "output": str(output)}, indent=2))
    return 0 if audit_summary["pass"] and split_summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
