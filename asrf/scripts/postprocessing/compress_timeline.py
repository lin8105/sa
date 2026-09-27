#!/usr/bin/env python3
"""Apply one verified retained-frame mapping to every bundle timeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PHYSICAL_SIGNAL_NAMES = (
    "linear_velocity", "angular_velocity", "linear_acceleration",
    "angular_acceleration", "force_magnitude", "torque_magnitude",
    "force_change", "torque_change", "gripper_velocity",
)


def physical_low_activity_mask(
    signals: dict[str, np.ndarray], thresholds: dict[str, float],
) -> np.ndarray:
    """Return the actual joint nine-signal low-activity candidate mask.

    Round35 used inclusive velocity comparisons; preserving ``<=`` here is
    required for exact replay at the two frozen velocity limits. All seven
    additional signals participate directly in the same conjunction.
    Temporal run filtering, gripper-event protection, edge context, and skill
    survival remain the caller's unchanged Round35 policy.
    """
    missing = set(PHYSICAL_SIGNAL_NAMES) - set(signals)
    if missing:
        raise ValueError(f"missing physical signals: {sorted(missing)}")
    required_thresholds = set(PHYSICAL_SIGNAL_NAMES)
    if set(thresholds) != required_thresholds:
        raise ValueError(
            f"threshold names must be exactly {sorted(required_thresholds)}"
        )
    arrays = {name: np.asarray(signals[name], dtype=np.float64) for name in PHYSICAL_SIGNAL_NAMES}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1 or any(array.ndim != 1 for array in arrays.values()):
        raise ValueError("all physical signals must be aligned one-dimensional arrays")
    limits = {name: float(thresholds[name]) for name in PHYSICAL_SIGNAL_NAMES}
    if any(not np.isfinite(value) for value in limits.values()):
        raise ValueError("all physical thresholds must be finite")
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("physical signals must contain only finite values")

    # The <= velocity comparisons are the historical Round35 boundary rule;
    # every added physical signal uses a strict finite upper limit.
    return (
        (arrays["linear_velocity"] <= limits["linear_velocity"])
        & (arrays["angular_velocity"] <= limits["angular_velocity"])
        & (arrays["linear_acceleration"] < limits["linear_acceleration"])
        & (arrays["angular_acceleration"] < limits["angular_acceleration"])
        & (arrays["force_magnitude"] < limits["force_magnitude"])
        & (arrays["torque_magnitude"] < limits["torque_magnitude"])
        & (arrays["force_change"] < limits["force_change"])
        & (arrays["torque_change"] < limits["torque_change"])
        & (arrays["gripper_velocity"] < limits["gripper_velocity"])
    )

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from postprocessing.common import (  # type: ignore
        Segment, TrajectoryBundle, frame_labels_to_segments, hash_if_exists, load_bundle,
        plot_bundle, provenance_path, save_bundle, validate_retained_indices,
    )
else:
    from .common import Segment, TrajectoryBundle, frame_labels_to_segments, hash_if_exists, load_bundle, plot_bundle, provenance_path, save_bundle, validate_retained_indices


def bundle_dirs(root: Path) -> list[Path]:
    if (root / "trajectory_bundle.json").is_file():
        return [root]
    return sorted(p.parent for p in root.rglob("trajectory_bundle.json"))


def mapping_path(source: Path, bundle: TrajectoryBundle) -> Path:
    if source.is_file():
        if source.name not in {"frame_mapping.csv", "compressed_to_original_mapping.csv"}:
            raise ValueError("--compression-source must be a mapping CSV or directory")
        return source
    direct = source / "frame_mapping.csv"
    if direct.is_file():
        return direct
    trajectory = str(bundle.metadata.get("trajectory_id", ""))
    parts = trajectory.split("/")
    candidates = [source / parts[-2] / parts[-1] / "frame_mapping.csv", source / parts[-1] / "frame_mapping.csv"] if len(parts) >= 2 else []
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no canonical frame_mapping.csv found under {source} for {trajectory}")


def load_retained(mapping: Path, original_T: int) -> np.ndarray:
    table = pd.read_csv(mapping)
    if "kept" in table.columns and "original_frame" in table.columns:
        if len(table) != original_T:
            raise ValueError(f"mapping length {len(table)} does not equal original_T {original_T}")
        keep = table["kept"].to_numpy(bool)
        retained = table.loc[keep, "original_frame"].to_numpy(np.int64)
    elif "original_frame" in table.columns:
        retained = table["original_frame"].to_numpy(np.int64)
    else:
        raise ValueError(f"mapping lacks original_frame/kept columns: {mapping}")
    return validate_retained_indices(retained, original_T)


def remap_segments(segments: list[Segment], retained_indices: np.ndarray) -> list[Segment]:
    """Map boundaries without collapsing adjacent equal-skill source segments."""
    output: list[Segment] = []
    for segment in segments:
        inside = np.flatnonzero((retained_indices >= segment.start) & (retained_indices <= segment.end))
        if len(inside) == 0:
            continue
        start = int(inside[0])
        end = int(inside[-1])
        output.append(Segment(start, end, segment.skill_id, segment.skill_name))
    if output:
        output[0] = Segment(0, output[0].end, output[0].skill_id, output[0].skill_name)
        output[-1] = Segment(output[-1].start, len(retained_indices) - 1, output[-1].skill_id, output[-1].skill_name)
    for left, right in zip(output, output[1:]):
        if left.end + 1 != right.start:
            raise ValueError("retained mapping produced a gap between remapped segments")
    return output


def compress_bundle(bundle: TrajectoryBundle, retained_indices: np.ndarray, mapping: Path, *, heatmap_path: Path | None = None) -> TrajectoryBundle:
    retained_indices = validate_retained_indices(retained_indices, bundle.current_T)
    out = bundle.clone()
    out.citr = bundle.citr[:, retained_indices]
    out.gt_frame_labels = bundle.gt_frame_labels[retained_indices]
    out.prediction_frame_labels = bundle.prediction_frame_labels[retained_indices]
    out.sf_brb = bundle.sf_brb[retained_indices]
    out.r5_brb = bundle.r5_brb[retained_indices]
    out.gripper_norm = bundle.gripper_norm[retained_indices]
    out.timestamps_s = None if bundle.timestamps_s is None else bundle.timestamps_s[retained_indices]
    out.timestamps_us = None if bundle.timestamps_us is None else bundle.timestamps_us[retained_indices]
    out.motion_features = None if bundle.motion_features is None else bundle.motion_features[retained_indices]
    out.gt_segments = remap_segments(bundle.gt_segments, retained_indices)
    out.prediction_segments = remap_segments(bundle.prediction_segments, retained_indices)
    metadata = json.loads(json.dumps(bundle.metadata))
    metadata["original_T"] = bundle.original_T
    metadata["current_T"] = len(retained_indices)
    metadata["retained_indices"] = retained_indices.tolist()
    metadata["compression"] = {"method": "verified_round35_retained_frame_mapping", "source": provenance_path(mapping), "source_sha256": hash_if_exists(mapping), "original_T": bundle.current_T, "compressed_T": len(retained_indices), "temporal_resizing": False, "same_mapping_for": ["citr", "gt", "prediction", "gripper_norm", "sf_brb", "r5_brb", "timestamps", "motion_features"], "segment_boundaries_preserved": True}
    metadata.setdefault("source_paths", {})["retained_mapping"] = provenance_path(mapping)
    if heatmap_path and heatmap_path.is_file():
        metadata["source_paths"]["canonical_heatmap"] = provenance_path(heatmap_path)
    out.metadata = metadata
    out.validate()
    return out


def process_one(bundle_path: Path, compression_source: Path, output_root: Path, force: bool) -> Path:
    bundle = load_bundle(bundle_path)
    mapping = mapping_path(compression_source, bundle)
    retained = load_retained(mapping, bundle.current_T)
    heatmap = mapping.parent / "citr_fingerprint_pure.png"
    compressed = compress_bundle(bundle, retained, mapping, heatmap_path=heatmap)
    output = output_root / bundle.metadata["trajectory_id"].replace("/", "__").replace(" ", "_")
    if output.exists() and not force:
        existing = load_bundle(output)
        expected = compressed.metadata["compression"]
        actual = existing.metadata.get("compression", {})
        if actual.get("source_sha256") != expected.get("source_sha256") or int(actual.get("compressed_T", -1)) != compressed.current_T:
            raise ValueError(f"existing compressed bundle provenance does not match requested mapping: {output}")
        return output
    save_bundle(compressed, output, force=force)
    if force or not (output / "trajectory.png").exists():
        plot_bundle(compressed, output / "trajectory.png", mode="standard", heatmap_path=compressed.metadata.get("source_paths", {}).get("canonical_heatmap"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress a standard TrajectoryBundle with one verified retained-frame mapping.")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--input-bundle-dir", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--compression-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.bundle:
        inputs = [args.bundle]
    elif args.input_bundle_dir and args.all:
        inputs = bundle_dirs(args.input_bundle_dir)
    else:
        parser.error("use --bundle or --input-bundle-dir with --all")
    outputs = [process_one(x, args.compression_source, args.output_dir, args.force) for x in inputs]
    print(json.dumps({"trajectory_count": len(outputs), "outputs": [str(x.resolve()) for x in outputs]}, indent=2))


if __name__ == "__main__":
    main()
