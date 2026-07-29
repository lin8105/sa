"""Read-only full-data audit for ASRF round 3.5.

All writes are confined to this repository's ``outputs/round3_5_data_audit``
directory. No file under the external dataset is opened for writing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asrf.data.dataset import load_timestamp_vector


HEATMAP_PREFERRED = "citr_fingerprint_pure.png"
HEATMAP_ALTERNATIVES = ("citr_fingerprint.png", "heatmap.png", "heatmap.jpg")
ANNOTATION_PREFERRED = "segments.csv"
ANNOTATION_ALTERNATIVES = ("segments.csv.bak", "annotations.csv", "ground_truth.csv")
TIMESTAMP_PREFERRED = "citr_features.csv"
TIMESTAMP_ALTERNATIVES = ("video_timestamps.csv", "timestamps.csv")
KNOWN_ALIASES = {"pick": "reach", "translation": "transport"}
CANDIDATES = (
    "reach", "pick", "approach", "grasp", "close_gripper", "lift", "transport",
    "translation", "place", "release", "pour", "pour_recover", "recover", "plug",
    "unplug", "push", "wipe", "retreat", "return",
)


@dataclass
class SegmentRecord:
    raw_label: str
    normalized_label: str
    start_raw: float | None
    end_raw: float | None
    start_frame: int | None
    end_frame_exclusive: int | None
    duration_frames: int | None
    duration_seconds: float | None


@dataclass
class Recording:
    split_root: str
    task_name: str
    group_id: str
    trajectory_id: str
    absolute_path: str
    heatmap_exists: bool
    heatmap_filename: str
    heatmap_channels: int | None
    heatmap_height: int | None
    temporal_width: int | None
    segments_exists: bool
    segments_filename: str
    timestamp_exists: bool
    number_of_segments: int
    first_label: str
    last_label: str
    valid_for_training: bool
    exclusion_reason: str
    possible_duplicate_group: str
    alternative_heatmap_filenames: str
    alternative_annotation_filenames: str
    alternative_timestamp_filenames: str
    timestamp_filename: str
    segment_records: list[SegmentRecord]
    raw_sequence: list[str]
    normalized_sequence: list[str]
    duration_seconds: float | None
    segments_sha256: str
    heatmap_sha256: str
    annotation_issues: list[str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [field.strip() for field in (reader.fieldnames or []) if field]
        return fields, list(reader)


def first_existing(directory: Path, names: tuple[str, ...]) -> str | None:
    for name in names:
        if (directory / name).is_file():
            return name
    return None


def annotation_fields(fields: list[str]) -> tuple[str | None, str | None, str | None]:
    label = next((name for name in ("label", "action", "skill", "class") if name in fields), None)
    if "start_timestamp_us" in fields and "end_timestamp_us_exclusive" in fields:
        return label, "start_timestamp_us", "end_timestamp_us_exclusive"
    if "start_frame" in fields and "end_frame" in fields:
        return label, "start_frame", "end_frame"
    if "start_time" in fields and "end_time" in fields:
        return label, "start_time", "end_time"
    return label, None, None


def to_float(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def inspect_segments(path: Path, timestamps: np.ndarray | None, temporal_width: int | None) -> tuple[list[SegmentRecord], list[str]]:
    fields, rows = csv_rows(path)
    label_field, start_field, end_field = annotation_fields(fields)
    issues: list[str] = []
    if label_field is None:
        issues.append("missing_label_column")
    if start_field is None or end_field is None:
        issues.append("unsupported_endpoint_columns")
    records: list[SegmentRecord] = []
    for row in rows:
        raw_label = (row.get(label_field or "") or "").strip()
        if not raw_label:
            issues.append("missing_or_empty_label")
        normalized = KNOWN_ALIASES.get(raw_label, raw_label)
        start_raw = to_float(row.get(start_field or ""))
        end_raw = to_float(row.get(end_field or ""))
        start_frame: int | None = None
        end_frame: int | None = None
        duration_frames: int | None = None
        duration_seconds: float | None = None
        if start_raw is not None and end_raw is not None:
            if end_raw <= start_raw:
                issues.append("zero_or_negative_duration")
            if start_field == "start_timestamp_us":
                duration_seconds = (end_raw - start_raw) / 1_000_000.0
                if timestamps is not None:
                    start_frame = int(np.searchsorted(timestamps, int(start_raw), side="left"))
                    end_frame = int(np.searchsorted(timestamps, int(end_raw), side="left"))
            elif start_field == "start_frame":
                start_frame = int(start_raw)
                end_frame = int(end_raw) + 1
                if timestamps is not None and 0 <= start_frame < len(timestamps) and 0 <= end_frame - 1 < len(timestamps):
                    duration_seconds = float((timestamps[min(end_frame - 1, len(timestamps) - 1)] - timestamps[start_frame]) / 1_000_000.0)
            else:
                duration_seconds = end_raw - start_raw
            if start_frame is not None and end_frame is not None:
                duration_frames = end_frame - start_frame
                if duration_frames <= 0:
                    issues.append("zero_or_negative_frame_duration")
                if temporal_width is not None and (start_frame < 0 or end_frame > temporal_width):
                    issues.append("annotation_out_of_range")
        records.append(SegmentRecord(raw_label, normalized, start_raw, end_raw, start_frame, end_frame, duration_frames, duration_seconds))
    ordered = [record for record in records if record.start_frame is not None and record.end_frame_exclusive is not None]
    ordered.sort(key=lambda record: int(record.start_frame or 0))
    for previous, current in zip(ordered, ordered[1:]):
        if int(current.start_frame) < int(previous.end_frame_exclusive):
            issues.append("overlapping_annotations")
        elif int(current.start_frame) > int(previous.end_frame_exclusive):
            issues.append("temporal_gap")
    if temporal_width is not None and ordered:
        if int(ordered[0].start_frame) > 0:
            issues.append("leading_temporal_gap")
        if int(ordered[-1].end_frame_exclusive) < temporal_width:
            issues.append("trailing_temporal_gap")
    return records, sorted(set(issues))


def discover_recording_dirs(root: Path) -> list[Path]:
    signatures = set((HEATMAP_PREFERRED, *HEATMAP_ALTERNATIVES, ANNOTATION_PREFERRED, *ANNOTATION_ALTERNATIVES, TIMESTAMP_PREFERRED, *TIMESTAMP_ALTERNATIVES))
    candidates: list[Path] = []
    for task_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        def visit(directory: Path) -> None:
            child_directories = sorted(path for path in directory.iterdir() if path.is_dir())
            filenames = {path.name for path in directory.iterdir() if path.is_file()}
            if signatures.intersection(filenames) or not child_directories:
                candidates.append(directory)
                return
            for child in child_directories:
                visit(child)
        for child in sorted(path for path in task_directory.iterdir() if path.is_dir()):
            visit(child)
    return sorted(candidates)


def inspect_recording(root: Path, directory: Path, split_root: str) -> Recording:
    relative = directory.relative_to(root)
    task_name = relative.parts[0] if relative.parts else ""
    trajectory_id = "/".join(relative.parts[1:]) if len(relative.parts) > 1 else relative.name
    group_id = relative.parts[1] if len(relative.parts) > 1 else relative.name
    preferred_heatmap = directory / HEATMAP_PREFERRED
    alternative_heatmaps = [name for name in HEATMAP_ALTERNATIVES if (directory / name).is_file()]
    annotation_path = directory / ANNOTATION_PREFERRED
    alternative_annotations = [name for name in ANNOTATION_ALTERNATIVES if (directory / name).is_file()]
    timestamp_name = first_existing(directory, (TIMESTAMP_PREFERRED, *TIMESTAMP_ALTERNATIVES))
    timestamp_path = directory / timestamp_name if timestamp_name else None
    timestamps: np.ndarray | None = None
    timestamp_error = ""
    if timestamp_path and timestamp_name == TIMESTAMP_PREFERRED:
        try:
            timestamps = load_timestamp_vector(timestamp_path)
        except Exception as exc:
            timestamp_error = f"invalid_timestamp_file:{type(exc).__name__}"
    elif timestamp_path:
        try:
            fields, rows = csv_rows(timestamp_path)
            timestamp_field = "timestamp_us" if "timestamp_us" in fields else next((field for field in ("timestamp", "time_us", "time") if field in fields), None)
            if timestamp_field:
                timestamps = np.asarray([int(float(row[timestamp_field])) for row in rows], dtype=np.int64)
        except Exception as exc:
            timestamp_error = f"invalid_alternative_timestamp:{type(exc).__name__}"
    heatmap_channels: int | None = None
    heatmap_height: int | None = None
    temporal_width: int | None = None
    heatmap_error = ""
    if preferred_heatmap.is_file():
        try:
            with Image.open(preferred_heatmap) as image:
                heatmap_channels = len(image.convert("RGB").getbands())
                heatmap_height = image.height
                temporal_width = image.width
        except Exception as exc:
            heatmap_error = f"invalid_heatmap:{type(exc).__name__}"
    records: list[SegmentRecord] = []
    annotation_issues: list[str] = []
    if annotation_path.is_file():
        try:
            records, annotation_issues = inspect_segments(annotation_path, timestamps, temporal_width)
        except Exception as exc:
            annotation_issues = [f"annotation_read_error:{type(exc).__name__}"]
    reasons: list[str] = []
    if not preferred_heatmap.is_file():
        reasons.append("missing_preferred_heatmap" + (f";alternatives={','.join(alternative_heatmaps)}" if alternative_heatmaps else ""))
    if heatmap_error:
        reasons.append(heatmap_error)
    if not annotation_path.is_file():
        reasons.append("missing_preferred_segments" + (f";alternatives={','.join(alternative_annotations)}" if alternative_annotations else ""))
    if not timestamp_path:
        reasons.append("missing_timestamp_file")
    if timestamp_error:
        reasons.append(timestamp_error)
    if temporal_width is not None and timestamps is not None and temporal_width != len(timestamps):
        reasons.append("heatmap_timestamp_width_mismatch")
    reasons.extend(annotation_issues)
    valid = bool(preferred_heatmap.is_file() and annotation_path.is_file() and timestamps is not None and not reasons)
    raw_sequence = [record.raw_label for record in records]
    normalized_sequence = [record.normalized_label for record in records]
    duration_values = [record.duration_seconds for record in records if record.duration_seconds is not None]
    duration_seconds = float(sum(duration_values)) if duration_values else None
    return Recording(
        split_root, task_name, group_id, trajectory_id, str(directory.resolve()),
        preferred_heatmap.is_file(), HEATMAP_PREFERRED if preferred_heatmap.is_file() else "",
        heatmap_channels, heatmap_height, temporal_width, annotation_path.is_file(),
        ANNOTATION_PREFERRED if annotation_path.is_file() else "", timestamps is not None,
        len(records), raw_sequence[0] if raw_sequence else "", raw_sequence[-1] if raw_sequence else "",
        valid, ";".join(sorted(set(reasons))), "", ",".join(alternative_heatmaps),
        ",".join(alternative_annotations), ",".join(name for name in TIMESTAMP_ALTERNATIVES if (directory / name).is_file()),
        timestamp_name or "", records, raw_sequence, normalized_sequence, duration_seconds,
        sha256(annotation_path) if annotation_path.is_file() else "", sha256(preferred_heatmap) if preferred_heatmap.is_file() else "",
        sorted(set(annotation_issues)),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def recording_inventory_row(recording: Recording) -> dict[str, Any]:
    fields = ("split_root", "task_name", "group_id", "trajectory_id", "absolute_path", "heatmap_exists", "heatmap_filename", "heatmap_channels", "heatmap_height", "temporal_width", "segments_exists", "segments_filename", "timestamp_exists", "number_of_segments", "first_label", "last_label", "valid_for_training", "exclusion_reason", "possible_duplicate_group")
    return {field: getattr(recording, field) for field in fields}


def build_split_expansion(repo_root: Path, recordings: list[Recording]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_specs = [("pour_train", "splits/pour_train.txt", "train"), ("pour_val", "splits/pour_val.txt", "train"), ("pour_test", "splits/pour_test.txt", "test"), ("pour_test_p3_p5", "splits/pour_test_p3_p5.txt", "test")]
    for split_name, relative_path, split_root in split_specs:
        entries = [line.strip() for line in (repo_root / relative_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        for entry in entries:
            matches = [r for r in recordings if r.split_root == split_root and r.task_name == "pour" and (r.trajectory_id == entry or r.group_id == entry)]
            if not matches:
                rows.append({"split_name": split_name, "split_entry": entry, "expanded_trajectory_id": "", "task": "pour", "path": "", "included": False, "reason": "no_matching_recording"})
            for recording in matches:
                rows.append({"split_name": split_name, "split_entry": entry, "expanded_trajectory_id": recording.trajectory_id, "task": "pour", "path": recording.absolute_path, "included": recording.valid_for_training, "reason": "included" if recording.valid_for_training else recording.exclusion_reason})
    return rows


def build_label_inventory(recordings: list[Recording]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    tasks_raw: dict[str, set[str]] = defaultdict(set)
    for recording in recordings:
        for segment in recording.segment_records:
            key = (segment.raw_label, segment.normalized_label)
            row = aggregate.setdefault(key, {"tasks": set(), "trajectories": set(), "occurrence_count": 0, "total_frame_count": 0, "durations_frames": [], "durations_seconds": []})
            row["tasks"].add(recording.task_name)
            row["trajectories"].add(f"{recording.split_root}:{recording.task_name}/{recording.trajectory_id}")
            row["occurrence_count"] += 1
            row["total_frame_count"] += segment.duration_frames or 0
            if segment.duration_frames is not None:
                row["durations_frames"].append(segment.duration_frames)
            if segment.duration_seconds is not None:
                row["durations_seconds"].append(segment.duration_seconds)
            tasks_raw[segment.raw_label].add(recording.task_name)
    rows: list[dict[str, Any]] = []
    for (raw_label, normalized_label), row in sorted(aggregate.items()):
        trajectories = sorted(row["trajectories"])
        rows.append({
            "raw_label": raw_label, "normalized_label": normalized_label, "tasks": ",".join(sorted(row["tasks"])),
            "trajectory_count": len(trajectories), "occurrence_count": row["occurrence_count"], "total_frame_count": row["total_frame_count"],
            "median_duration_frames": median(row["durations_frames"]) if row["durations_frames"] else "", "min_duration_frames": min(row["durations_frames"]) if row["durations_frames"] else "", "max_duration_frames": max(row["durations_frames"]) if row["durations_frames"] else "", "median_duration_seconds": median(row["durations_seconds"]) if row["durations_seconds"] else "",
            "split_presence": "both" if any(item.startswith("train:") for item in trajectories) and any(item.startswith("test:") for item in trajectories) else ("train" if any(item.startswith("train:") for item in trajectories) else "test"),
            "label_notes": "known_alias" if raw_label in KNOWN_ALIASES else ("candidate_requires_review" if raw_label in CANDIDATES else "unlisted_label"),
        })
    return rows, {raw: sorted(tasks) for raw, tasks in sorted(tasks_raw.items())}


def build_sequences(recordings: list[Recording]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequence_rows: list[dict[str, Any]] = []
    transition_counts: Counter[tuple[str, str, str]] = Counter()
    transition_trajectories: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for recording in recordings:
        if not recording.valid_for_training:
            continue
        collapsed: list[str] = []
        for label in recording.raw_sequence:
            if not collapsed or label != collapsed[-1]:
                collapsed.append(label)
        sequence_rows.append({"split_root": recording.split_root, "task": recording.task_name, "trajectory_id": recording.trajectory_id, "collapsed_sequence": " -> ".join(collapsed), "normalized_collapsed_sequence": " -> ".join(recording.normalized_sequence), "number_of_segments": recording.number_of_segments, "T": recording.temporal_width or "", "duration_seconds": recording.duration_seconds if recording.duration_seconds is not None else ""})
        for first, second in zip(collapsed, collapsed[1:]):
            key = (recording.task_name, first, second)
            transition_counts[key] += 1
            transition_trajectories[key].add(recording.trajectory_id)
    transition_rows = [{"task": task, "from_label": first, "to_label": second, "count": count, "trajectory_count": len(transition_trajectories[(task, first, second)])} for (task, first, second), count in sorted(transition_counts.items())]
    return sequence_rows, transition_rows


def candidate_stats(recordings: list[Recording]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        matching = [(recording, segment) for recording in recordings if recording.valid_for_training for segment in recording.segment_records if segment.raw_label == candidate]
        durations = [segment.duration_frames for _, segment in matching if segment.duration_frames is not None]
        tasks = sorted({recording.task_name for recording, _ in matching})
        trajectories = {f"{recording.split_root}:{recording.task_name}/{recording.trajectory_id}" for recording, _ in matching}
        review = candidate in {"pick", "reach", "place", "release", "recover", "pour_recover", "retreat", "return"} or len(tasks) > 1
        rows.append({"candidate": candidate, "raw_labels_mapped": candidate, "tasks": ",".join(tasks), "trajectory_count": len(trajectories), "segment_count": len(matching), "total_frames": sum(durations), "median_duration_frames": median(durations) if durations else "", "min_duration_frames": min(durations) if durations else "", "max_duration_frames": max(durations) if durations else "", "physical_definition_consistency": "manual_review_required" if review else ("not_observed" if not matching else "task_local_only"), "automatic_merging_safe": "no" if review else "not_applicable", "review_required": "yes" if matching else "no"})
    return rows


def duplicate_rows(recordings: list[Recording]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Recording]] = defaultdict(list)
    for recording in recordings:
        if recording.valid_for_training:
            groups[("segments_sha256", recording.segments_sha256)].append(recording)
            groups[("heatmap_sha256", recording.heatmap_sha256)].append(recording)
            groups[("shape_sequence", f"{recording.temporal_width}|{' -> '.join(recording.raw_sequence)}")].append(recording)
    rows: list[dict[str, Any]] = []
    group_number = 0
    for (evidence, key), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        group_number += 1
        for recording in members:
            rows.append({"possible_duplicate_group": f"dup{group_number:03d}", "evidence": evidence, "evidence_key": key, "split_root": recording.split_root, "task": recording.task_name, "trajectory_id": recording.trajectory_id, "path": recording.absolute_path, "temporal_width": recording.temporal_width, "segments_sha256": recording.segments_sha256, "heatmap_sha256": recording.heatmap_sha256, "collapsed_sequence": " -> ".join(recording.raw_sequence)})
    return rows


def boundary_statistics(recordings: list[Recording]) -> list[dict[str, Any]]:
    valid = [recording for recording in recordings if recording.valid_for_training]
    rows: list[dict[str, Any]] = []
    for task in sorted({recording.task_name for recording in valid}):
        task_recordings = [recording for recording in valid if recording.task_name == task]
        internal = 0
        total_frames = 0
        transitions: Counter[tuple[str, str]] = Counter()
        transition_trajectories: dict[tuple[str, str], set[str]] = defaultdict(set)
        for recording in task_recordings:
            total_frames += recording.temporal_width or 0
            collapsed: list[str] = []
            for label in recording.raw_sequence:
                if not collapsed or label != collapsed[-1]:
                    collapsed.append(label)
            internal += max(0, len(collapsed) - 1)
            for first, second in zip(collapsed, collapsed[1:]):
                transitions[(first, second)] += 1
                transition_trajectories[(first, second)].add(recording.trajectory_id)
        rows.append({"scope": "task", "task": task, "from_label": "", "to_label": "", "trajectory_count": len(task_recordings), "total_frames": total_frames, "internal_boundaries": internal, "frame0_boundaries": len(task_recordings), "positives_including_frame0": internal + len(task_recordings), "positives_excluding_frame0": internal, "positive_ratio_including_frame0": (internal + len(task_recordings)) / total_frames if total_frames else 0.0, "positive_ratio_excluding_frame0": internal / total_frames if total_frames else 0.0, "reciprocal_weight_including_frame0": total_frames / (internal + len(task_recordings)) if internal + len(task_recordings) else "", "reciprocal_weight_excluding_frame0": total_frames / internal if internal else ""})
        for (first, second), count in sorted(transitions.items()):
            rows.append({"scope": "transition", "task": task, "from_label": first, "to_label": second, "trajectory_count": len(transition_trajectories[(first, second)]), "total_frames": "", "internal_boundaries": count, "frame0_boundaries": 0, "positives_including_frame0": "", "positives_excluding_frame0": count, "positive_ratio_including_frame0": "", "positive_ratio_excluding_frame0": "", "reciprocal_weight_including_frame0": "", "reciprocal_weight_excluding_frame0": ""})
    all_recordings = valid
    total_frames = sum(recording.temporal_width or 0 for recording in all_recordings)
    internal = sum(int(row["internal_boundaries"]) for row in rows if row["scope"] == "task")
    rows.append({"scope": "all", "task": "", "from_label": "", "to_label": "", "trajectory_count": len(all_recordings), "total_frames": total_frames, "internal_boundaries": internal, "frame0_boundaries": len(all_recordings), "positives_including_frame0": internal + len(all_recordings), "positives_excluding_frame0": internal, "positive_ratio_including_frame0": (internal + len(all_recordings)) / total_frames if total_frames else 0.0, "positive_ratio_excluding_frame0": internal / total_frames if total_frames else 0.0, "reciprocal_weight_including_frame0": total_frames / (internal + len(all_recordings)) if internal + len(all_recordings) else "", "reciprocal_weight_excluding_frame0": total_frames / internal if internal else ""})
    return rows


def make_figures(recordings: list[Recording], output_dir: Path) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt
    selected: list[Recording] = []
    for split_root, task in (("train", "pour"), ("train", "pick and place"), ("train", "wipe"), ("test", "pour"), ("test", "pp")):
        candidates = sorted([r for r in recordings if r.split_root == split_root and r.task_name == task and r.valid_for_training], key=lambda r: r.trajectory_id)
        if candidates:
            selected.append(candidates[0])
    paths: list[str] = []
    for recording in selected:
        path = Path(recording.absolute_path)
        with Image.open(path / HEATMAP_PREFERRED) as image:
            array = np.asarray(image.convert("RGB"))
        timestamps = load_timestamp_vector(path / TIMESTAMP_PREFERRED)
        _, rows = csv_rows(path / ANNOTATION_PREFERRED)
        starts: list[tuple[int, str]] = []
        ends: list[int] = []
        for row in rows:
            starts.append((int(np.searchsorted(timestamps, int(row["start_timestamp_us"]), side="left")), row.get("label", "")))
            ends.append(int(np.searchsorted(timestamps, int(row["end_timestamp_us_exclusive"]), side="left")))
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [4, 1]}, constrained_layout=True)
        axes[0].imshow(array, aspect="auto", extent=(0, array.shape[1], array.shape[0], 0))
        axes[0].set_ylabel("heatmap height")
        axes[0].set_title(f"{recording.split_root}/{recording.task_name}/{recording.trajectory_id} | shape={array.shape} | display-only scaling")
        axes[1].set_ylim(-0.5, 0.5)
        axes[1].set_yticks([])
        for (start, label), end in zip(starts, ends):
            axes[1].axvspan(start, end, alpha=0.35)
            axes[1].text((start + end) / 2.0, 0.0, label, rotation=45, ha="center", va="center", fontsize=8)
            axes[0].axvline(start, color="white", linewidth=0.8)
        axes[1].set_xlabel("heatmap column (T preserved; plotting only)")
        output = output_dir / f"{recording.split_root}_{recording.task_name.replace(' ', '_')}_{recording.trajectory_id.replace('/', '_')}.png"
        fig.savefig(output, dpi=120)
        plt.close(fig)
        paths.append(str(output))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
    parser.add_argument("--output", default="outputs/round3_5_data_audit")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(__file__).resolve().parents[1] / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    recordings = [inspect_recording(data_root / split_root, directory, split_root) for split_root in ("train", "test") for directory in discover_recording_dirs(data_root / split_root)]
    duplicates = duplicate_rows(recordings)
    duplicate_groups: dict[str, set[str]] = defaultdict(set)
    for row in duplicates:
        duplicate_groups[row["path"]].add(row["possible_duplicate_group"])
    for recording in recordings:
        recording.possible_duplicate_group = ",".join(sorted(duplicate_groups.get(recording.absolute_path, set())))
    inventory_fields = ["split_root", "task_name", "group_id", "trajectory_id", "absolute_path", "heatmap_exists", "heatmap_filename", "heatmap_channels", "heatmap_height", "temporal_width", "segments_exists", "segments_filename", "timestamp_exists", "number_of_segments", "first_label", "last_label", "valid_for_training", "exclusion_reason", "possible_duplicate_group", "alternative_heatmap_filenames", "alternative_annotation_filenames", "alternative_timestamp_filenames", "timestamp_filename"]
    for recording in recordings:
        inventory_row = recording_inventory_row(recording)
        inventory_row["alternative_heatmap_filenames"] = recording.alternative_heatmap_filenames
        inventory_row["alternative_annotation_filenames"] = recording.alternative_annotation_filenames
        inventory_row["alternative_timestamp_filenames"] = recording.alternative_timestamp_filenames
        inventory_row["timestamp_filename"] = recording.timestamp_filename
    inventory_rows = []
    for recording in recordings:
        row = recording_inventory_row(recording)
        row.update({"alternative_heatmap_filenames": recording.alternative_heatmap_filenames, "alternative_annotation_filenames": recording.alternative_annotation_filenames, "alternative_timestamp_filenames": recording.alternative_timestamp_filenames, "timestamp_filename": recording.timestamp_filename})
        inventory_rows.append(row)
    write_csv(output_dir / "dataset_inventory.csv", inventory_rows, inventory_fields)
    split_rows = build_split_expansion(Path(__file__).resolve().parents[1], recordings)
    write_csv(output_dir / "split_expansion.csv", split_rows, ["split_name", "split_entry", "expanded_trajectory_id", "task", "path", "included", "reason"])
    label_rows, raw_tasks = build_label_inventory(recordings)
    write_csv(output_dir / "raw_label_inventory.csv", label_rows, list(label_rows[0]) if label_rows else ["raw_label"])
    sequence_rows, transition_rows = build_sequences(recordings)
    write_csv(output_dir / "trajectory_sequences.csv", sequence_rows, list(sequence_rows[0]) if sequence_rows else ["task"])
    write_csv(output_dir / "transition_counts.csv", transition_rows, ["task", "from_label", "to_label", "count", "trajectory_count"])
    candidate_rows = candidate_stats(recordings)
    write_csv(output_dir / "candidate_skill_statistics.csv", candidate_rows, list(candidate_rows[0]))
    boundary_rows = boundary_statistics(recordings)
    write_csv(output_dir / "boundary_statistics.csv", boundary_rows, list(boundary_rows[0]))
    write_csv(output_dir / "possible_duplicates.csv", duplicates, ["possible_duplicate_group", "evidence", "evidence_key", "split_root", "task", "trajectory_id", "path", "temporal_width", "segments_sha256", "heatmap_sha256", "collapsed_sequence"])
    figure_paths = make_figures(recordings, figures_dir)
    valid = [recording for recording in recordings if recording.valid_for_training]
    split_entry_counts = {name: sum(row["split_name"] == name for row in split_rows) for name in sorted({row["split_name"] for row in split_rows})}
    summary = {
        "data_root": str(data_root), "total_directories_inspected": len(recordings), "valid_train_trajectories": sum(recording.split_root == "train" for recording in valid), "valid_test_trajectories": sum(recording.split_root == "test" for recording in valid),
        "valid_by_split_and_task": {f"{split}:{task}": sum(recording.split_root == split and recording.task_name == task for recording in valid) for split in ("train", "test") for task in sorted({recording.task_name for recording in recordings})},
        "incomplete_by_task": dict(sorted(Counter(recording.task_name for recording in recordings if not recording.valid_for_training).items())), "task_directories_train": sorted({recording.task_name for recording in recordings if recording.split_root == "train"}), "task_directories_test": sorted({recording.task_name for recording in recordings if recording.split_root == "test"}), "raw_labels": sorted(raw_tasks),
        "current_split_rows": len(split_rows), "current_split_included_rows": sum(bool(row["included"]) for row in split_rows), "current_split_entries": split_entry_counts, "valid_unreferenced_recordings": [recording.absolute_path for recording in valid if not any(row["path"] == recording.absolute_path and row["included"] for row in split_rows)], "possible_duplicate_row_count": len(duplicates), "annotation_issue_counts": dict(sorted(Counter(issue for recording in recordings for issue in recording.annotation_issues).items())), "boundary_statistics_all": next(row for row in boundary_rows if row["scope"] == "all"), "figure_paths": figure_paths, "known_aliases_used": KNOWN_ALIASES,
        "notes": ["Preferred files are required for validity; alternative filenames are detected but never silently substituted.", "The current pour split union is 15 physical paths: train p1-p8, validation p9-p10, and test p1-p5.", "p1-p8 alone expands to eight train recordings; the 15 count is the complete round-3 pour split union."],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
