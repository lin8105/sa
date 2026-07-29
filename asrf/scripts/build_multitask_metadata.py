"""Build canonical multi-task inventory, manifest, and auditable alias evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asrf.data.dataset import load_timestamp_vector


ALIASES = {"pick": "reach", "translation": "transport"}
CANONICAL = {"reach", "grasp", "lift", "transport", "pour", "pour_recover", "place", "wipe", "retreat"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def normalize(label: str) -> str:
    return ALIASES.get(label.strip(), label.strip())


def segment_records(path: Path) -> list[dict[str, object]]:
    fields, rows = read_csv(path / "segments.csv")
    timestamp_path = path / "citr_features.csv"
    timestamps = load_timestamp_vector(timestamp_path)
    records: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        raw = str(row.get("label", "")).strip()
        start_us = int(row["start_timestamp_us"])
        end_us = int(row["end_timestamp_us_exclusive"])
        start = int(np.searchsorted(timestamps, start_us, side="left"))
        end = int(np.searchsorted(timestamps, end_us, side="left"))
        records.append({"index": index, "raw_label": raw, "canonical_label": normalize(raw), "start_frame": start, "end_frame_exclusive": end, "duration_frames": end - start, "duration_seconds": (end_us - start_us) / 1_000_000.0, "start_us": start_us, "end_us": end_us})
    return records


def gripper_evidence(path: Path, start_us: int, end_us: int) -> dict[str, object]:
    result: dict[str, object] = {
        "gripper_position_start": "unavailable", "gripper_position_end": "unavailable", "gripper_position_delta": "unavailable",
        "gripper_position_behavior": "unavailable", "held_fraction": "unavailable", "held_evidence": "unavailable",
        "robot_state_available": False,
    }
    gripper_path = path / "gripper_10hz.csv"
    if gripper_path.is_file():
        _, rows = read_csv(gripper_path)
        selected = []
        for row in rows:
            timestamp = int(row.get("timestamp_us", "0"))
            if start_us <= timestamp < end_us and row.get("position", "") != "":
                selected.append(float(row["position"]))
        if selected:
            window = max(1, len(selected) // 10)
            start_value = float(np.median(selected[:window]))
            end_value = float(np.median(selected[-window:]))
            delta = end_value - start_value
        result.update({"gripper_position_start": start_value, "gripper_position_end": end_value, "gripper_position_delta": delta, "gripper_position_behavior": "closing" if delta < -0.005 else "opening" if delta > 0.005 else "stable", "position_held_evidence": "closed_position" if start_value < 0.023 else "open_position"})
    else:
        result["position_held_evidence"] = "unavailable"
    robot_path = path / "robot_states.csv"
    if robot_path.is_file():
        fields, rows = read_csv(robot_path)
        if "is_grasped" in fields:
            selected = []
            for row in rows:
                timestamp = int(row.get("timestamp_us", "0"))
                if start_us <= timestamp < end_us and row.get("is_grasped", "") != "":
                    selected.append(float(row["is_grasped"]))
            if selected:
                fraction = float(np.mean(np.asarray(selected) > 0.5))
                result.update({"robot_state_available": True, "held_fraction": fraction, "robot_state_evidence": "held" if fraction >= 0.5 else "not_held"})
    if result.get("robot_state_available") and float(result.get("held_fraction", 0.0)) >= 0.5:
        result["held_evidence"] = "held_robot_state"
    elif result.get("position_held_evidence") == "closed_position":
        result["held_evidence"] = "held_closed_gripper_position"
    elif result.get("position_held_evidence") == "open_position":
        result["held_evidence"] = "not_held_open_gripper_position"
    else:
        result["held_evidence"] = "unavailable"
    return result


def representative_figure(path: Path, record: dict[str, object], all_records: list[dict[str, object]], output_dir: Path) -> str:
    import matplotlib.pyplot as plt

    image_path = path / "citr_fingerprint_pure.png"
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = str(record["raw_label"])
    filename = f"{raw}_{str(record['task']).replace(' ', '_')}_{record['recording_id']}.png"
    destination = output_dir / filename
    figure, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True, constrained_layout=True)
    axes[0].imshow(rgb, origin="upper", aspect="auto", interpolation="nearest", extent=(0, rgb.shape[1], rgb.shape[0], 0))
    axes[0].axvspan(float(record["start_frame"]), float(record["end_frame_exclusive"]), color="red", alpha=0.25)
    axes[0].set_ylabel("CITR heatmap")
    sequence = np.zeros(rgb.shape[1], dtype=int)
    for item in all_records:
        sequence[int(item["start_frame"]):int(item["end_frame_exclusive"])] = hash(str(item["canonical_label"])) % 10
    axes[1].plot(np.arange(len(sequence)) + 0.5, sequence, linewidth=1)
    axes[1].axvspan(float(record["start_frame"]), float(record["end_frame_exclusive"]), color="red", alpha=0.25)
    axes[1].set_ylabel("annotated span")
    axes[1].set_xlabel("frame index; display aspect only")
    figure.suptitle(f"{record['task']} / {record['recording_id']} — raw={raw}, canonical={record['canonical_label']}")
    figure.savefig(destination, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return str(destination.relative_to(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data", type=Path)
    parser.add_argument("--scan", default="outputs/multitask_baseline/data_stability/scan1/dataset_inventory.csv", type=Path)
    parser.add_argument("--output-dir", default="outputs/multitask_baseline", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    inventory_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    alias_rows: list[dict[str, object]] = []
    first_alias_record: dict[str, dict[str, object]] = {}
    with args.scan.open("r", encoding="utf-8", newline="") as handle:
        scan_rows = list(csv.DictReader(handle))
    for scan_row in scan_rows:
        absolute = Path(scan_row["absolute_path"])
        relative = absolute.relative_to(args.data_root.resolve()).as_posix()
        parts = Path(relative).parts
        split_root = parts[0] if parts else ""
        task = parts[1] if len(parts) > 1 else ""
        recording_id = "/".join(parts[2:]) if len(parts) > 2 else absolute.name
        valid = scan_row.get("valid_for_training") == "True"
        raw_records: list[dict[str, object]] = []
        if valid:
            raw_records = segment_records(absolute)
        raw_labels = [str(item["raw_label"]) for item in raw_records]
        canonical_labels = [normalize(value) for value in raw_labels]
        canonical_valid = valid and all(value in CANONICAL for value in canonical_labels) and bool(raw_records)
        exclusion = scan_row.get("exclusion_reason", "")
        if valid and not canonical_valid:
            exclusion = "label_outside_nine_class_ontology"
        heatmap = absolute / "citr_fingerprint_pure.png"
        timestamps = absolute / "citr_features.csv"
        inventory_record = {
            "split_root": split_root, "task": task, "recording_id": recording_id, "absolute_path": str(absolute), "relative_path": relative,
            "T": int(scan_row["temporal_width"]) if scan_row.get("temporal_width") else "", "segment_count": len(raw_records),
            "raw_labels": "|".join(raw_labels), "normalized_labels": "|".join(canonical_labels), "valid": canonical_valid,
            "exclusion_reason": "" if canonical_valid else exclusion, "heatmap_sha256": sha256(heatmap) if heatmap.is_file() else "", "segments_sha256": sha256(absolute / "segments.csv") if (absolute / "segments.csv").is_file() else "", "timestamp_sha256": sha256(timestamps) if timestamps.is_file() else "", "heatmap_channels": scan_row.get("heatmap_channels", ""), "heatmap_height": scan_row.get("heatmap_height", ""), "suspected_recording_family": "unavailable",
        }
        inventory_rows.append(inventory_record)
        if canonical_valid:
            collapsed: list[str] = []
            for label in canonical_labels:
                if not collapsed or collapsed[-1] != label:
                    collapsed.append(label)
            manifest_rows.append({
                "trajectory_id": relative, "task": task, "path": str(absolute), "recording_family": "unavailable", "session": "unavailable", "operator": "unavailable", "object": "unavailable", "sequence": " -> ".join(collapsed), "T": inventory_record["T"], "num_segments": len(raw_records), "raw_labels": "|".join(raw_labels), "canonical_labels": "|".join(canonical_labels), "eligible_for_train": split_root == "train", "eligible_for_test": split_root == "test", "notes": "No reliable session/operator/object metadata found; split is task-stratified and trajectory-level."})
            for index, item in enumerate(raw_records):
                raw = str(item["raw_label"])
                if raw not in {"pick", "reach", "translation", "transport"}:
                    continue
                previous = str(raw_records[index - 1]["raw_label"]) if index else ""
                following = str(raw_records[index + 1]["raw_label"]) if index + 1 < len(raw_records) else ""
                evidence = gripper_evidence(absolute, int(item["start_us"]), int(item["end_us"]))
                held = evidence["held_evidence"]
                if raw in {"pick", "reach"}:
                    supported = following == "grasp" and held in {"not_held_open_gripper_position", "not_held", "unavailable"}
                    confidence = "high" if supported and held != "unavailable" else "medium" if following == "grasp" else "low"
                    recommended = "reach"
                else:
                    supported = following in {"place", "pour", "translation", "wipe"} and held in {"held_robot_state", "held_closed_gripper_position", "unavailable"}
                    confidence = "high" if supported and held != "unavailable" else "medium" if following in {"place", "pour", "translation", "wipe"} else "low"
                    recommended = "transport"
                row = {"task": task, "recording": relative, "recording_id": recording_id, "raw_label": raw, "canonical_label": normalize(raw), "recommended_canonical_class": recommended, "start_frame": item["start_frame"], "end_frame_exclusive": item["end_frame_exclusive"], "duration_frames": item["duration_frames"], "preceding_label": previous, "following_label": following, **evidence, "confidence": confidence, "manual_review_flag": not supported, "semantic_assessment": "supported_by_boundary_and_gripper_evidence" if supported else "ambiguous_requires_manual_review", "representative_figure": ""}
                alias_rows.append(row)
                first_alias_record.setdefault(raw, {"path": absolute, "record": row, "all_records": raw_records, "task": task, "recording_id": recording_id})
    # Create one auditable heatmap figure per raw alias/reference label.
    figure_dir = args.output_dir / "alias_figures"
    for raw, payload in first_alias_record.items():
        figure_path = representative_figure(payload["path"], payload["record"], payload["all_records"], figure_dir)
        for row in alias_rows:
            if row["raw_label"] == raw and not row["representative_figure"]:
                row["representative_figure"] = figure_path

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0]) if rows else []
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "dataset_inventory.csv", inventory_rows)
    write_csv(args.output_dir / "alias_review.csv", alias_rows)
    manifest_fields = ["trajectory_id", "task", "path", "recording_family", "session", "operator", "object", "sequence", "T", "num_segments", "raw_labels", "canonical_labels", "eligible_for_train", "eligible_for_test", "notes"]
    with (repo_root / "splits/multitask_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "valid_train": sum(bool(row["valid"]) and row["split_root"] == "train" for row in inventory_rows),
        "valid_test": sum(bool(row["valid"]) and row["split_root"] == "test" for row in inventory_rows),
        "valid_by_task": dict(sorted({f"{row['split_root']}:{row['task']}": sum(bool(other["valid"]) and other["split_root"] == row["split_root"] and other["task"] == row["task"] for other in inventory_rows) for row in inventory_rows}.items())),
        "alias_rows": len(alias_rows), "alias_manual_review_rows": sum(bool(row["manual_review_flag"]) for row in alias_rows),
        "raw_labels": sorted({label for row in inventory_rows for label in str(row["raw_labels"]).split("|") if label}),
    }
    (args.output_dir / "metadata_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["alias_manual_review_rows"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
