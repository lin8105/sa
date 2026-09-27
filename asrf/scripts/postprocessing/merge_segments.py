#!/usr/bin/env python3
"""Apply the frozen one-sided merge with physical event segment protection."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from postprocessing.common import Segment, TrajectoryBundle, load_bundle, plot_bundle, save_bundle  # type: ignore
else:
    from .common import Segment, TrajectoryBundle, load_bundle, plot_bundle, save_bundle

ROOT = Path(__file__).resolve().parents[2]
# The completed Round40 artifacts live under the repository's historical
# ``outputs/0`` namespace in this checkout.  Keep the canonical defaults tied to
# those verified artifacts rather than silently falling back to new thresholds.
DEFAULT_SCALER = ROOT / "outputs/0/round40_round27_post_segmentation_compression_merge/similarity_feature_scaler.yaml"
DEFAULT_THRESHOLDS = ROOT / "outputs/0/round40_round27_post_segmentation_compression_merge/motion_thresholds.yaml"
LINEAR_THRESHOLD = 0.02600749068
ANGULAR_THRESHOLD = 0.0477127692
LINEAR_ACCEL_THRESHOLD = 2.4995922160215494
ANGULAR_ACCEL_THRESHOLD = 4.541420560023916
EVENT_THRESHOLD_POSITION_MPS = 3.0000000111022306e-08
GRIPPER_MAX_POSITION = 0.0247059
GRIPPER_MIN_POSITION = 0.000490196
EVENT_THRESHOLD_NORM_PER_S = EVENT_THRESHOLD_POSITION_MPS / (GRIPPER_MAX_POSITION - GRIPPER_MIN_POSITION)
EVENT_BRIDGE_S = 0.10
EVENT_CONTEXT_S = 0.50


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(changes[::2], changes[1::2])]


def detect_physical_events(bundle: TrajectoryBundle) -> list[dict[str, Any]]:
    ts = bundle.timestamps_s if bundle.timestamps_s is not None else np.arange(bundle.current_T, dtype=float) / 100.0
    if len(ts) < 2 or np.any(np.diff(ts) <= 0):
        raise ValueError("bundle timestamps must be strictly increasing for gripper event detection")
    speed = np.abs(np.gradient(bundle.gripper_norm, ts))
    valid = np.isfinite(speed)
    raw = valid & (speed > EVENT_THRESHOLD_NORM_PER_S)
    rate = 1.0 / float(np.median(np.diff(ts)))
    max_gap = max(1, int(round(EVENT_BRIDGE_S * rate)))
    for start, end in runs(~raw):
        if start == 0 or end == len(raw) or end - start > max_gap:
            continue
        if raw[start - 1] and raw[end] and np.max(speed[start:end]) <= EVENT_THRESHOLD_NORM_PER_S * 2 + 1e-9:
            raw[start:end] = True
    events: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(runs(raw)):
        slope = float(np.median(np.gradient(bundle.gripper_norm[start:end], ts[start:end]))) if end - start > 1 else float(bundle.gripper_norm[start] - bundle.gripper_norm[max(0, start - 1)])
        direction = "CLOSING_EVENT" if slope > 0 else "OPENING_EVENT" if slope < 0 else "UNKNOWN_EVENT"
        context_start = max(0, int(np.searchsorted(ts, ts[start] - EVENT_CONTEXT_S, side="left")))
        context_end = min(len(ts), int(np.searchsorted(ts, ts[end - 1] + EVENT_CONTEXT_S, side="right")))
        events.append({"event_index": index, "start": start, "end_exclusive": end, "direction": direction, "start_time_s": float(ts[start]), "end_time_s": float(ts[end - 1]), "magnitude_norm_per_s": float(np.max(speed[start:end])), "magnitude_position_m_per_s": float(np.max(speed[start:end]) * (GRIPPER_MAX_POSITION - GRIPPER_MIN_POSITION)), "context_start": context_start, "context_end_exclusive": context_end, "threshold_position_m_per_s": EVENT_THRESHOLD_POSITION_MPS, "threshold_norm_per_s": EVENT_THRESHOLD_NORM_PER_S, "bridge_s": EVENT_BRIDGE_S, "context_s": EVENT_CONTEXT_S})
    return events


def event_overlaps(segment: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if max(segment["start"], event["start"]) < min(segment["end"], event["end_exclusive"])]


def load_scaler(path: Path) -> dict[str, np.ndarray]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {"median": np.asarray(raw["median"], dtype=float), "scale": np.asarray(raw["robust_scale"], dtype=float)}


def load_motion_thresholds(path: Path) -> dict[str, float]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {"linear": float(raw.get("linear_velocity", LINEAR_THRESHOLD)), "angular": float(raw.get("angular_velocity", ANGULAR_THRESHOLD)), "linear_acceleration": float(raw.get("linear_acceleration", LINEAR_ACCEL_THRESHOLD)), "angular_acceleration": float(raw.get("angular_acceleration", ANGULAR_ACCEL_THRESHOLD))}


def motion_rows(segment: dict[str, Any], features: np.ndarray, thresholds: dict[str, float], fraction: float) -> dict[str, Any]:
    start, end = int(segment["start"]), int(segment["end"])
    if end <= start:
        return {"low_motion": 0, "fraction_all_thresholds": 0.0, "p90_linear_velocity": 0.0, "p90_angular_velocity": 0.0, "p90_linear_acceleration": 0.0, "p90_angular_acceleration": 0.0}
    lin, ang, lacc, aacc = features[start:end, -4:].T
    low = (lin <= thresholds["linear"]) & (ang <= thresholds["angular"]) & (lacc <= thresholds["linear_acceleration"]) & (aacc <= thresholds["angular_acceleration"])
    p90 = [float(np.percentile(x, 90)) for x in (lin, ang, lacc, aacc)]
    limits = [thresholds["linear"], thresholds["angular"], thresholds["linear_acceleration"], thresholds["angular_acceleration"]]
    return {"low_motion": int(all(p90[i] <= limits[i] for i in range(4)) and float(low.mean()) >= fraction), "fraction_all_thresholds": float(low.mean()), "p90_linear_velocity": p90[0], "p90_angular_velocity": p90[1], "p90_linear_acceleration": p90[2], "p90_angular_acceleration": p90[3]}


def distance(a: np.ndarray, b: np.ndarray, scaler: dict[str, np.ndarray]) -> float:
    n = min(len(a), len(b))
    if not n:
        return float("inf")
    return float(np.mean(np.abs((a[-n:] - scaler["median"]) / scaler["scale"] - (b[:n] - scaler["median"]) / scaler["scale"])))


def local_distances(segments: list[dict[str, Any]], index: int, features: np.ndarray, scaler: dict[str, np.ndarray], window: int) -> tuple[float, float]:
    current = segments[index]
    left = distance(features[segments[index - 1]["end"] - window:segments[index - 1]["end"]], features[current["start"]:current["start"] + window], scaler) if index else float("inf")
    right = distance(features[current["end"] - window:current["end"]], features[segments[index + 1]["start"]:segments[index + 1]["start"] + window], scaler) if index + 1 < len(segments) else float("inf")
    return left, right


def merge_prediction(bundle: TrajectoryBundle, *, fraction: float = 0.80, window: int = 5, mode: str = "one_pass", protection: str = "gripper-event-segment") -> tuple[list[Segment], dict[str, Any]]:
    if protection != "gripper-event-segment":
        raise ValueError("canonical protection is gripper-event-segment")
    if bundle.motion_features is None:
        raise ValueError("bundle lacks canonical motion features required by the Round40 similarity rule")
    if mode not in {"one_pass", "iterative"}:
        raise ValueError("mode must be one_pass or iterative")
    events = detect_physical_events(bundle)
    thresholds = load_motion_thresholds(DEFAULT_THRESHOLDS)
    scaler = load_scaler(DEFAULT_SCALER)
    segments = [{"id": f"m{i}", "start": s.start, "end": s.end + 1, "skill_id": s.skill_id, "skill_name": s.skill_name, "members": [f"m{i}"]} for i, s in enumerate(bundle.prediction_segments)]
    initial_ids = [s["id"] for s in segments]
    locked: set[frozenset[str]] = set()
    decisions: list[dict[str, Any]] = []
    iteration = 0
    while True:
        changed = False
        candidate_ids = initial_ids if iteration == 0 and mode == "one_pass" else [s["id"] for s in segments]
        for candidate_id in candidate_ids:
            index = next((i for i, s in enumerate(segments) if candidate_id in s["members"]), None)
            if index is None or (mode == "one_pass" and len(segments[index]["members"]) > 1):
                continue
            current = segments[index]
            left = segments[index - 1] if index else None
            right = segments[index + 1] if index + 1 < len(segments) else None
            current_events = event_overlaps(current, events)
            left_events = event_overlaps(left, events) if left else []
            right_events = event_overlaps(right, events) if right else []
            motion = motion_rows(current, bundle.motion_features, thresholds, fraction)
            left_lock = bool(left and frozenset((left["id"], current["id"])) in locked)
            right_lock = bool(right and frozenset((current["id"], right["id"])) in locked)
            left_valid = bool(left and not current_events and not left_events and not left_lock)
            right_valid = bool(right and not current_events and not right_events and not right_lock)
            left_distance, right_distance = local_distances(segments, index, bundle.motion_features, scaler, window)
            selected = ""
            reason = ""
            if not motion["low_motion"]:
                reason = "not_low_motion"
            elif current_events:
                reason = "GRIPPER_EVENT_SEGMENT_PROTECTED"
            elif not left_valid and not right_valid:
                reason = "GRIPPER_EVENT_SEGMENT_PROTECTION" if (left_events or right_events) else "no_valid_target"
            elif left_valid and right_valid:
                selected = "left" if left_distance <= right_distance else "right"
            elif left_valid:
                selected = "left"
            else:
                selected = "right"
            before = len(segments); deleted = ""; preserved = ""; result_id = ""; executed = 0
            if selected == "left":
                parent, next_segment = segments[index - 1], segments[index + 1] if index + 1 < len(segments) else None
                deleted = f"{parent['id']}|{current['id']}"; parent["end"] = current["end"]; parent["members"] += current["members"]; result_id = parent["id"]; segments.pop(index); executed = 1
                if next_segment is not None:
                    preserved = f"{parent['id']}|{next_segment['id']}"; locked.add(frozenset((parent["id"], next_segment["id"])))
            elif selected == "right":
                previous, parent = segments[index - 1] if index else None, segments[index + 1]
                deleted = f"{current['id']}|{parent['id']}"; parent["start"] = current["start"]; parent["members"] = current["members"] + parent["members"]; result_id = parent["id"]; segments.pop(index); executed = 1
                if previous is not None:
                    preserved = f"{previous['id']}|{parent['id']}"; locked.add(frozenset((previous["id"], parent["id"])))
            blocked = int(motion["low_motion"] and not executed and bool(current_events or left_events or right_events))
            decisions.append({"candidate_id": candidate_id, "start": current["start"], "end_exclusive": current["end"], "skill_id": current["skill_id"], "skill_name": current["skill_name"], **motion, "current_event_indices": [e["event_index"] for e in current_events], "left_event_indices": [e["event_index"] for e in left_events], "right_event_indices": [e["event_index"] for e in right_events], "candidate_protected": int(bool(current_events)), "left_protected": int(bool(left_events)), "right_protected": int(bool(right_events)), "left_lock": int(left_lock), "right_lock": int(right_lock), "left_distance": left_distance, "right_distance": right_distance, "selected_side": selected, "merge_executed": executed, "blocked_by_gripper_event_protection": blocked, "reason": reason, "deleted_boundary": deleted, "preserved_opposite_boundary": preserved, "resulting_segment_id": result_id, "segment_count_before": before, "segment_count_after": len(segments)})
            if executed:
                if len(segments) != before - 1:
                    raise RuntimeError("accepted merge did not remove exactly one boundary")
                changed = True
        if mode == "one_pass" or not changed:
            break
        iteration += 1
        if iteration > len(initial_ids) + 2:
            raise RuntimeError("iterative merge exceeded finite-pass guard")
    final = [Segment(int(s["start"]), int(s["end"] - 1), int(s["skill_id"]), str(s["skill_name"])) for s in segments]
    metadata = {"protection_mode": "GRIPPER_EVENT_SEGMENT_PROTECTION", "fraction": fraction, "window": window, "mode": mode, "events": events, "decisions": decisions, "protected_segment_count": sum(int(x["candidate_protected"]) for x in decisions), "blocked_merge_count": sum(int(x["blocked_by_gripper_event_protection"]) for x in decisions)}
    return final, metadata


def merge_bundle(bundle: TrajectoryBundle, *, fraction: float = 0.80, window: int = 5, mode: str = "one_pass", protection: str = "gripper-event-segment") -> TrajectoryBundle:
    merged_segments, merge_metadata = merge_prediction(bundle, fraction=fraction, window=window, mode=mode, protection=protection)
    out = bundle.clone()
    out.metadata = json.loads(json.dumps(bundle.metadata))
    out.metadata["pre_merge_segments"] = [segment.to_dict() for segment in bundle.prediction_segments]
    out.metadata["merge"] = merge_metadata
    out.prediction_segments = merged_segments
    out.prediction_frame_labels = np.full(bundle.current_T, -1, dtype=np.int64)
    for segment in merged_segments:
        out.prediction_frame_labels[segment.start:segment.end + 1] = segment.skill_id
    out.metadata["current_T"] = bundle.current_T
    out.validate()
    return out


def bundle_dirs(root: Path) -> list[Path]:
    if (root / "trajectory_bundle.json").is_file():
        return [root]
    return sorted(p.parent for p in root.rglob("trajectory_bundle.json"))


def process_one(path: Path, output_root: Path, fraction: float, window: int, mode: str, protection: str, figure_mode: str, force: bool) -> Path:
    bundle = load_bundle(path)
    merged = merge_bundle(bundle, fraction=fraction, window=window, mode=mode, protection=protection)
    output = output_root / bundle.metadata["trajectory_id"].replace("/", "__").replace(" ", "_")
    if output.exists() and not force:
        existing = load_bundle(output)
        actual = existing.metadata.get("merge", {})
        if (float(actual.get("fraction", -1)) != float(fraction) or int(actual.get("window", -1)) != int(window) or actual.get("mode") != mode or actual.get("protection_mode") != "GRIPPER_EVENT_SEGMENT_PROTECTION"):
            raise ValueError(f"existing merged bundle provenance does not match requested merge configuration: {output}")
        return output
    save_bundle(merged, output, force=force)
    if force or not (output / "trajectory_merged.png").exists():
        plot_bundle(merged, output / "trajectory_merged.png", mode=figure_mode, heatmap_path=merged.metadata.get("source_paths", {}).get("canonical_heatmap"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge compressed Hybrid segments using frozen Round40 one-sided logic and GRIPPER_EVENT_SEGMENT_PROTECTION.")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--input-bundle-dir", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.80)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--mode", choices=("one_pass", "iterative"), default="one_pass")
    parser.add_argument("--protection", choices=("gripper-event-segment",), default="gripper-event-segment")
    parser.add_argument("--figure-mode", choices=("merged", "merged-simple"), default="merged")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.bundle:
        inputs = [args.bundle]
    elif args.input_bundle_dir and args.all:
        inputs = bundle_dirs(args.input_bundle_dir)
    else:
        parser.error("use --bundle or --input-bundle-dir with --all")
    outputs = [process_one(x, args.output_dir, args.fraction, args.window, args.mode, args.protection, args.figure_mode, args.force) for x in inputs]
    print(json.dumps({"trajectory_count": len(outputs), "outputs": [str(x.resolve()) for x in outputs]}, indent=2))


if __name__ == "__main__":
    main()
