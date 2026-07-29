"""Minimal Round 8 inference/export and selected comparison redraw."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
OUT = ROOT / "outputs/brb_release_round8/figres/redrawn_selected"
LABEL_CONFIG = ROOT / "configs/labels_multitask_release.yaml"
METHODS = {
    "ASRF-SF": "baseline_single_frame",
    "r5": "hard_window_r5",
    "r10": "hard_window_r10",
    "r20": "hard_window_r20",
    "s5": "gaussian_s5",
    "s10": "gaussian_s10",
    "s20": "gaussian_s20",
}
ENTRIES = ["test/pour/p1", "test/pp/pp_c1", "test/wipe/w1", "test/wipe/w4"]
ONTOLOGY = {"reach", "grasp", "lift", "transport", "pour", "pour_recover", "place", "release", "wipe", "retreat"}
CHECKPOINTS = {name: ROOT / "outputs/brb_release_round8" / directory / "best.pt" for name, directory in METHODS.items()}

import sys
sys.path.insert(0, str(ROOT / "src"))

from asrf.data.annotations import load_segments_csv  # noqa: E402
from asrf.data.dataset import load_heatmap, load_timestamp_vector, load_trajectory_sample  # noqa: E402
from asrf.data.labels import load_label_mapping, normalize_label_name  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.refinement.refine import refine_asrf_predictions  # noqa: E402
from asrf.refinement.segments import TemporalInterval  # noqa: E402
from asrf.training.checkpointing import load_checkpoint, sha256_file  # noqa: E402
from asrf.visualization.round8 import plot_round8_comparison_figure  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_entry(entry: str, mapping: Any) -> dict[str, Any]:
    path = DATA / entry
    errors: list[str] = []
    result: dict[str, Any] = {"trajectory": entry, "segments_exists": (path / "segments.csv").is_file(), "features_exists": (path / "citr_features.csv").is_file(), "heatmap_exists": (path / "citr_fingerprint_pure.png").is_file(), "temporal_width": 0, "heatmap_width": 0, "segment_count": 0, "labels": "", "gaps": 0, "overlaps": 0, "zero_duration": 0, "chronological": False, "coverage": False, "valid": False, "errors": ""}
    try:
        timestamps = load_timestamp_vector(path / "citr_features.csv")
        heatmap = load_heatmap(path / "citr_fingerprint_pure.png", expected_height=88)
        result["temporal_width"] = len(timestamps)
        result["heatmap_width"] = int(heatmap.shape[-1])
        if result["temporal_width"] != result["heatmap_width"]:
            errors.append("heatmap width mismatch")
        fmt, rows = load_segments_csv(path / "segments.csv")
        result["segment_count"] = len(rows)
        starts: list[int] = []
        ends: list[int] = []
        labels: list[str] = []
        previous_end: int | None = None
        for row in rows:
            name = normalize_label_name(str(row.get("label", "")), mapping)
            labels.append(name)
            if name not in ONTOLOGY:
                errors.append(f"invalid label {name}")
            if fmt == "timestamp":
                start = int(np.searchsorted(timestamps, int(row["start_timestamp_us"]), side="left"))
                end = int(np.searchsorted(timestamps, int(row["end_timestamp_us_exclusive"]), side="left"))
            else:
                start = int(row["start_frame"])
                end = int(row["end_frame"]) + 1
            starts.append(start); ends.append(end)
            if end <= start: result["zero_duration"] += 1
            if previous_end is not None:
                if start > previous_end: result["gaps"] += 1
                if start < previous_end: result["overlaps"] += 1
            previous_end = end
        result["labels"] = ";".join(sorted(set(labels)))
        result["chronological"] = all(a <= b for a, b in zip(starts, starts[1:]))
        result["coverage"] = bool(starts and starts[0] == 0 and ends[-1] == len(timestamps) and all(a == b for a, b in zip(ends[:-1], starts[1:])))
        if not result["chronological"]: errors.append("non-chronological")
        if result["gaps"]: errors.append("gaps")
        if result["overlaps"]: errors.append("overlaps")
        if result["zero_duration"]: errors.append("zero-duration")
        if not result["coverage"]: errors.append("incomplete coverage")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    result["errors"] = " | ".join(errors)
    result["valid"] = not errors and result["segments_exists"] and result["features_exists"] and result["heatmap_exists"]
    return result


def intervals(labels: torch.Tensor) -> list[TemporalInterval]:
    values = labels.tolist()
    starts = [0] + [index for index in range(1, len(values)) if values[index] != values[index - 1]]
    return [TemporalInterval(start, end) for start, end in zip(starts, starts[1:] + [len(values)])]


def gt_rows(labels: torch.Tensor, inverse: dict[int, str]) -> list[dict[str, Any]]:
    return [{"segment_index": i, "start_frame": seg.start, "end_frame": seg.end, "gt_label": inverse[int(labels[seg.start])]} for i, seg in enumerate(intervals(labels))]


def predicted_rows(segments: list[TemporalInterval], labels: torch.Tensor, probabilities: torch.Tensor, inverse: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    for index, seg in enumerate(segments):
        label_id = int(labels[seg.start])
        rows.append({"segment_index": index, "start_frame": seg.start, "end_frame": seg.end, "predicted_label": inverse[label_id], "predicted_label_id": label_id, "mean_confidence": float(probabilities[label_id, seg.start:seg.end].mean())})
    previous_end = 0
    for row in rows:
        if row["predicted_label"] not in ONTOLOGY or row["start_frame"] < previous_end or row["end_frame"] <= row["start_frame"]:
            raise AssertionError("invalid prediction segment export")
        previous_end = row["end_frame"]
    return rows


@torch.no_grad()
def infer_one(model: ASRFModel, entry: str, mapping: Any, target_config: dict[str, Any], inverse: dict[int, str]) -> dict[str, Any]:
    sample = load_trajectory_sample(DATA / entry, mapping, expected_height=88, boundary_target_config=target_config)
    output = model(sample["heatmap"].unsqueeze(0), valid_mask=sample["valid_mask"].unsqueeze(0))
    asb = output.asb_stage_probabilities[-1][0].cpu()
    brb = output.brb_stage_probabilities[-1][0, 0].cpu()
    raw_labels = asb.argmax(dim=0).to(torch.long)
    refined = refine_asrf_predictions(asb.unsqueeze(0), brb.view(1, 1, -1), torch.ones(1, len(raw_labels), dtype=torch.bool), threshold=0.50, voting="majority")
    asrf_labels = refined.refined_labels[0].cpu()
    asrf_segments = list(refined.intervals[0])
    return {"entry": entry, "sample": sample, "asb": asb, "brb": brb, "raw_labels": raw_labels, "raw_segments": intervals(raw_labels), "asrf_labels": asrf_labels, "asrf_segments": asrf_segments, "peaks": list(refined.selected_boundaries[0]), "inverse": inverse}


def save_diagnostic_prediction(root: Path, method: str, entry: str, item: dict[str, Any], inverse: dict[int, str]) -> dict[str, str]:
    slug = entry.replace("/", "_")
    base = root / "diagnostics" / "predictions" / method / slug
    raw_rows = predicted_rows(item["raw_segments"], item["raw_labels"], item["asb"], inverse)
    asrf_rows = predicted_rows(item["asrf_segments"], item["asrf_labels"], item["asb"], inverse)
    frame_rows = []
    timestamps = item["sample"]["timestamps"].tolist()
    for frame, timestamp in enumerate(timestamps):
        raw_id = int(item["raw_labels"][frame]); asrf_id = int(item["asrf_labels"][frame])
        frame_rows.append({"frame": frame, "time_s": (float(timestamp) - float(timestamps[0])) / 1e6, "raw_label": inverse[raw_id], "asrf_label": inverse[asrf_id], "brb_probability": float(item["brb"][frame])})
    write_csv(base / "frame_predictions.csv", frame_rows)
    write_csv(base / "predicted_segments_asrf.csv", asrf_rows)
    write_csv(base / "predicted_segments_raw_asb.csv", raw_rows)
    write_csv(base / "boundary_probabilities.csv", [{"frame": i, "time_s": (float(ts) - float(timestamps[0])) / 1e6, "brb_probability": float(item["brb"][i])} for i, ts in enumerate(timestamps)])
    write_csv(base / "selected_boundaries.csv", [{"frame": int(frame), "time_s": (float(timestamps[int(frame)]) - float(timestamps[0])) / 1e6} for frame in item["peaks"]])
    return {"frame_predictions": str((base / "frame_predictions.csv").relative_to(ROOT)), "predicted_segments": str((base / "predicted_segments_asrf.csv").relative_to(ROOT)), "boundary_probabilities": str((base / "boundary_probabilities.csv").relative_to(ROOT))}


def main() -> int:
    import yaml

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostics").mkdir(parents=True, exist_ok=True)
    mapping = load_label_mapping(LABEL_CONFIG)
    inverse = {int(value): key for key, value in mapping.items()}
    audit_rows = [audit_entry(entry, mapping) for entry in ENTRIES]
    if not all(row["valid"] for row in audit_rows):
        raise RuntimeError(json.dumps([row for row in audit_rows if not row["valid"]], indent=2))
    write_csv(OUT / "diagnostics/audit.csv", audit_rows)
    missing: dict[str, list[str]] = {name: [] for name in METHODS}
    for name, directory in METHODS.items():
        for entry in ENTRIES:
            if not any((ROOT / "outputs/brb_release_round8" / directory / candidate).exists() for candidate in [f"test/{entry.split('/')[-2]}/{entry.split('/')[-1]}/predicted_segments_asrf.csv", f"test/{entry.split('/')[-1]}/predicted_segments_asrf.csv"]):
                missing[name].append(entry)
    hashes_before = {name: sha256_file(path) for name, path in CHECKPOINTS.items()}
    sources: dict[str, dict[str, str]] = {}
    method_items: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in METHODS}
    for name, directory in METHODS.items():
        config_path = ROOT / "outputs/brb_release_round8" / directory / "resolved_config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = ASRFModel.from_config(config)
        model.load_state_dict(load_checkpoint(CHECKPOINTS[name])["model_state"], strict=True)
        model.eval()
        target_config = {key: config["data"][key] for key in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")}
        for entry in ENTRIES:
            item = infer_one(model, entry, mapping, target_config, inverse)
            method_items[name][entry] = item
            sources[f"{name}:{entry}"] = save_diagnostic_prediction(OUT, name, entry, item, inverse)
    for entry in ENTRIES:
        first = method_items["ASRF-SF"][entry]
        method_rows = {}
        brb = {}
        peaks = {}
        for name in METHODS:
            item = method_items[name][entry]
            method_rows[name] = predicted_rows(item["asrf_segments"], item["asrf_labels"], item["asb"], inverse)
            brb[name] = item["brb"].numpy()
            peaks[name] = item["peaks"]
        plot_round8_comparison_figure(first["sample"]["heatmap"].numpy(), first["sample"]["timestamps"].tolist(), gt_rows(first["sample"]["labels"], inverse), method_rows, OUT / "trajectories" / f"{entry.replace('/', '_')}_round8_comparison.png", brb_probabilities=brb, boundary_peaks=peaks, title=f"{entry} — Round 8 comparison", ontology=ONTOLOGY)
    hashes_after = {name: sha256_file(path) for name, path in CHECKPOINTS.items()}
    manifest = {"methods_in_row_order": list(METHODS), "trajectories": ENTRIES, "missing_existing_prediction_artifacts": missing, "minimal_inference_rerun": True, "full_experiment_rerun": False, "training_rerun": False, "source_prediction_files": sources, "checkpoint_hashes_before": hashes_before, "checkpoint_hashes_after": hashes_after, "checkpoint_hashes_unchanged": hashes_before == hashes_after, "audit_rows": audit_rows, "label_ontology": sorted(ONTOLOGY)}
    write_json(OUT / "figure_manifest.json", manifest)
    print(json.dumps({"figures": len(list((OUT / "trajectories").glob("*.png"))), "output": str(OUT), "missing_sources": missing, "inference_only": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
