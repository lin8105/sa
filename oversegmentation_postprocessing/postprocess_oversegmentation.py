#!/usr/bin/env python3
"""Apply frozen joint physical compression, then the canonical one-sided merge."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ASRF = REPO / "asrf"
sys.path.insert(0, str(ASRF / "scripts"))
sys.path.insert(0, str(ASRF / "src"))
from postprocessing.compress_timeline import (  # noqa: E402
    bundle_dirs, compress_bundle, physical_low_activity_mask,
)
from postprocessing.common import (  # noqa: E402
    load_bundle, plot_bundle, save_bundle,
)
import postprocessing.merge_segments as canonical_merge  # noqa: E402
import run_round35_multisignal_compression_v1 as round35  # noqa: E402

PACKAGE = Path(__file__).resolve().parent
canonical_merge.DEFAULT_SCALER = PACKAGE / "merge_scaler.yaml"
canonical_merge.DEFAULT_THRESHOLDS = PACKAGE / "merge_motion_thresholds.yaml"
merge_bundle = canonical_merge.merge_bundle


def _write_mapping(path: Path, keep: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("original_frame", "kept", "compressed_frame"))
        compressed = -1
        for index, retained in enumerate(keep):
            if retained:
                compressed += 1
            writer.writerow((index, int(retained), compressed if retained else -1))


def _physical_signals(context: dict[str, object]) -> dict[str, np.ndarray]:
    """Use Round35 synchronized derivatives plus canonical raw Bota magnitudes."""
    signals = {name: np.asarray(value, dtype=np.float64)
               for name, value in context["signals"].items()}
    bota = pd.read_csv(Path(context["source"]) / "bota_100hz.csv")
    source_t = bota["timestamp_us"].to_numpy(dtype=np.float64) / 1e6
    target_t = np.asarray(context["ts"], dtype=np.float64)
    if len(source_t) != len(bota) or np.any(np.diff(source_t) <= 0):
        raise ValueError(f"{context['trajectory_id']}: invalid Bota timeline")
    if target_t[0] < source_t[0] or target_t[-1] > source_t[-1]:
        raise ValueError(f"{context['trajectory_id']}: CITR timeline exceeds Bota support")
    force = bota.loc[:, ["F_x", "F_y", "F_z"]].to_numpy(dtype=np.float64)
    torque = bota.loc[:, ["tau_x", "tau_y", "tau_z"]].to_numpy(dtype=np.float64)
    force = np.column_stack([np.interp(target_t, source_t, force[:, j]) for j in range(3)])
    torque = np.column_stack([np.interp(target_t, source_t, torque[:, j]) for j in range(3)])
    signals["force_magnitude"] = np.linalg.norm(force, axis=1)
    signals["torque_magnitude"] = np.linalg.norm(torque, axis=1)
    signals["linear_velocity"] = np.asarray(context["lin"], dtype=np.float64)
    signals["angular_velocity"] = np.asarray(context["ang"], dtype=np.float64)
    return signals


def _write_report(path: Path, report: dict[str, object]) -> None:
    lines = ["# Frozen joint physical compression evaluation", "",
             "No training or ASRF inference was run by this postprocessing pipeline. Physical thresholds were fitted and frozen using TRAIN only; TEST was evaluated once, with no threshold or configuration changes after TEST access.", "",
             f"- Frozen threshold SHA256: `{report['threshold_config_sha256']}`",
             f"- Locally available Round35 mapping comparison: `{report['mapping_identity']}`.",
             f"- Mean compression ratio: {report['mean_compression_ratio']:.4%}.",
             "- Merge: unchanged canonical Round40 settings (fraction 0.80, window 5, one pass, whole-segment physical gripper-event protection).", ""]
    lines += ["", "## Interpretation and limits", "",
              "This report records the frozen mask application and does not claim that a non-identical mapping reproduces historical Round45 metrics. No thresholds were changed after seeing evaluation results.",
              "", "Per-trajectory mappings and processing provenance are in `mappings/` and `run_summary.json`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(bundle_root: Path, output_root: Path, data_root: Path, split: str,
        config_path: Path, make_figures: bool) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_train_only":
        raise ValueError("threshold config is not marked frozen_train_only")
    thresholds = {name: float(item["value"]) for name, item in payload["thresholds"].items()}
    round35.DATA = data_root
    rows = []
    for source_bundle in bundle_dirs(bundle_root):
        bundle = load_bundle(source_bundle)
        tid = str(bundle.metadata["trajectory_id"])
        parts = tid.split("/")
        if len(parts) != 3:
            raise ValueError(f"unexpected trajectory id in bundle: {tid}")
        if parts[0] == split:
            rows.append({"trajectory_id": tid, "split": split, "family": parts[1],
                         "original_frame_count": str(bundle.current_T),
                         "bundle_path": source_bundle})
    if not rows:
        raise ValueError(f"no {split} standard bundles found under {bundle_root}")
    outputs: list[dict[str, object]] = []
    for row in rows:
        context = round35.load_context(row)
        # Reuse the validated Round35 derivatives and the v002 canonical
        # timestamp synchronization for absolute Bota force/torque magnitudes.
        signals = _physical_signals(context)
        low = physical_low_activity_mask(signals, thresholds)
        event_protection, _ = round35.event_mask(context, context_s=0.50)
        keep, _, _ = round35.apply_policy(context, low, event_protection)
        if len(keep) != int(row["original_frame_count"]):
            raise RuntimeError(f"{row['trajectory_id']}: policy output length mismatch")

        tid = row["trajectory_id"]
        mapping = output_root / "mappings" / (tid.replace("/", "__") + ".csv")
        _write_mapping(mapping, keep)
        reference_map = round35.OUT / "data" / tid / "retained_indices.npy"
        retained_indices = np.flatnonzero(keep)
        differing_frames = None
        if reference_map.is_file():
            reference_indices = np.load(reference_map, allow_pickle=False)
            differing_frames = int(len(np.setxor1d(retained_indices, reference_indices)))
        encoded = tid.replace("/", "__").replace(" ", "_")
        source_bundle = row["bundle_path"]
        item: dict[str, object] = {
            "trajectory_id": tid,
            "family": row["family"],
            "split": split,
            "frames": int(len(keep)),
            "retained_frames": int(keep.sum()),
            "removed_frames": int((~keep).sum()),
            "compression_ratio": float((~keep).mean()),
            "historical_retained_index_identity": None if differing_frames is None else differing_frames == 0,
            "differing_retained_indices": differing_frames,
            "mapping": str(mapping),
            "bundle_available": (source_bundle / "trajectory_bundle.json").is_file(),
        }
        if item["bundle_available"]:
            bundle = load_bundle(source_bundle)
            if bundle.current_T != len(keep):
                raise RuntimeError(f"{tid}: bundle and physical timeline lengths differ")
            compressed = compress_bundle(bundle, np.flatnonzero(keep), mapping)
            compressed.metadata.setdefault("compression", {})["method"] = "joint_nine_signal_physical_low_activity"
            compressed.metadata["compression"]["threshold_config"] = str(config_path.resolve())
            compressed.metadata["compression"]["threshold_config_sha256"] = round35.sha256(config_path)
            compressed.metadata["compression"]["thresholds"] = payload["thresholds"]
            compressed_dir = output_root / "compressed" / encoded
            save_bundle(compressed, compressed_dir, force=True)
            merged = merge_bundle(compressed, fraction=0.80, window=5, mode="one_pass",
                                  protection="gripper-event-segment")
            merged_dir = output_root / "merged" / encoded
            save_bundle(merged, merged_dir, force=True)
            if make_figures:
                heatmap = bundle.metadata.get("source_paths", {}).get("canonical_heatmap")
                plot_bundle(compressed, compressed_dir / "trajectory.png", mode="standard", heatmap_path=heatmap)
                plot_bundle(merged, merged_dir / "trajectory_merged.png", mode="merged-simple", heatmap_path=heatmap)
            item["compressed_bundle"] = str(compressed_dir)
            item["merged_bundle"] = str(merged_dir)
        outputs.append(item)
    report = {
        "split": split,
        "threshold_config": str(config_path.resolve()),
        "threshold_config_sha256": round35.sha256(config_path),
        "trajectory_count": len(outputs),
        "mean_compression_ratio": float(np.mean([float(x["compression_ratio"]) for x in outputs])),
        "mapping_identity": None if all(x["differing_retained_indices"] is None for x in outputs) else {
            "audited_trajectories": sum(x["differing_retained_indices"] is not None for x in outputs),
            "exact_trajectories": sum(x["differing_retained_indices"] == 0 for x in outputs),
            "trajectories_with_differences": sum(x["differing_retained_indices"] > 0 for x in outputs),
            "differing_retained_indices": sum(int(x["differing_retained_indices"] or 0) for x in outputs),
        },
        "trajectories": outputs,
        "merge": {"fraction": 0.80, "window": 5, "mode": "one_pass",
                  "protection": "GRIPPER_EVENT_SEGMENT_PROTECTION"},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(output_root / "report.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True,
                        help="Round91 standard bundle directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Dataset containing train0/test0 robot recordings")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--thresholds", type=Path,
                        default=Path(__file__).with_name("physical_thresholds.json"))
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    result = run(args.bundle_dir, args.output_dir, args.data_root, args.split,
                 args.thresholds, args.figures)
    summary = {key: value for key, value in result.items() if key != "trajectories"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
