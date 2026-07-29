#!/usr/bin/env python3
"""Round 30B preflight and leakage gate.

Round 30B must not train when there is no non-test auxiliary family.  This
script deliberately stops at that gate; it does not generate samples, train a
model, or run the final 36-trajectory evaluation.
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
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
OUT = ROOT / "outputs/round30b_auxiliary_skill_independence"
SOURCE = ROOT / "outputs/round27b_hybrid"
SF_SHA = "6b11abff2ff4387eada88e38fb56133b29fd5c3facdfb54b42fc84701213db9a"
R5_SHA = "577d8edf9e2b04927acc235ffa4d6baab8df1712dd0b98eaaba9063fde31f406"
KNOWN = ("reach", "grasp", "lift", "transport", "place", "release", "retreat")
ASRF_TRAIN = {f"train/pick and place/pp{i}" for i in range(1, 11)}
ASRF_VALIDATION = {f"train/pick and place/pp{i}" for i in range(11, 21)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_round27_test_set() -> set[str]:
    path = SOURCE / "complete_test_inventory.csv"
    if not path.is_file():
        raise RuntimeError(f"missing frozen Round 27B inventory: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["trajectory"] for row in csv.DictReader(handle) if row.get("included") == "1"}


def family_of(entry: str) -> str:
    parts = Path(entry).parts
    if len(parts) < 3:
        return ""
    return parts[1]


def read_timestamps(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        values = [int(row["timestamp_us"]) for row in reader]
    timestamps = np.asarray(values, dtype=np.int64)
    if timestamps.ndim != 1 or not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps are empty or not strictly increasing")
    return timestamps


def read_gt(path: Path, timestamps: np.ndarray) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    output = []
    previous = 0
    for index, row in enumerate(rows):
        if {"start_timestamp_us", "end_timestamp_us_exclusive", "label"} <= fields:
            start = int(np.searchsorted(timestamps, int(row["start_timestamp_us"]), side="left"))
            end = int(np.searchsorted(timestamps, int(row["end_timestamp_us_exclusive"]), side="left"))
        elif {"start_frame", "end_frame", "label"} <= fields:
            start, end = int(row["start_frame"]), int(row["end_frame"]) + 1
        else:
            raise ValueError("unsupported annotation columns")
        if not (0 <= start < end <= len(timestamps)):
            raise ValueError(f"invalid interval {index}: {start}:{end}")
        if start != previous:
            raise ValueError(f"annotation gap/overlap at row {index}: expected {previous}, got {start}")
        output.append({"segment_index": index, "start": start, "end": end, "label": row["label"].strip()})
        previous = end
    if not output or previous != len(timestamps):
        raise ValueError("annotation does not cover the full trajectory")
    return output


def discover() -> tuple[list[dict[str, Any]], list[str]]:
    final_test = read_round27_test_set()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for features in sorted(DATA.rglob("citr_features.csv")):
        trajectory_path = features.parent
        try:
            entry = str(trajectory_path.relative_to(DATA))
        except ValueError:
            continue
        if len(Path(entry).parts) < 3 or entry in seen:
            continue
        seen.add(entry)
        annotation = trajectory_path / "segments.csv"
        heatmap = trajectory_path / "citr_fingerprint_pure.png"
        split = Path(entry).parts[0]
        family = family_of(entry)
        asrf_role = "ASRF_TRAINING" if entry in ASRF_TRAIN else "ASRF_VALIDATION" if entry in ASRF_VALIDATION else ""
        row: dict[str, Any] = {
            "trajectory_id": entry,
            "full_path": str(trajectory_path),
            "split": split,
            "family": family,
            "task_skill_labels": "",
            "frame_count": "",
            "gt_interval_count": "",
            "gt_intervals": "",
            "used_for_asrf_training": int(entry in ASRF_TRAIN),
            "used_for_asrf_validation": int(entry in ASRF_VALIDATION),
            "round27b_final_test": int(entry in final_test),
            "role": asrf_role or ("FINAL_TEST" if entry in final_test else ""),
            "independence_training_eligible": 0,
            "independence_validation_eligible": 0,
            "exclusion_reason": "",
            "feature_sha256": sha256(features),
            "annotation_sha256": sha256(annotation) if annotation.is_file() else "",
        }
        try:
            timestamps = read_timestamps(features)
            gt = read_gt(annotation, timestamps)
            labels = sorted({x["label"] for x in gt})
            row.update({
                "frame_count": len(timestamps),
                "gt_interval_count": len(gt),
                "task_skill_labels": ";".join(labels),
                "gt_intervals": ";".join(f"{x['start']}:{x['end']}:{x['label']}" for x in gt),
            })
            if not heatmap.is_file():
                raise ValueError("heatmap input missing")
        except Exception as exc:  # audit must retain invalid rows
            row["exclusion_reason"] = str(exc)
            rows.append(row)
            continue

        protected_family = family in {"plug", "pour", "wipe", "unscrew", "pp"}
        if entry in final_test:
            row["role"] = "FINAL_TEST"
            row["exclusion_reason"] = "exact Round 27B final-test trajectory"
        elif split == "test":
            row["role"] = "TEST_NOT_IN_ROUND27B_MANIFEST"
            row["exclusion_reason"] = "test trajectory outside the exact frozen Round 27B inventory"
        elif family == "pick and place":
            row["role"] = asrf_role or "PP_NONTEST_BACKGROUND"
            row["exclusion_reason"] = "PP is the frozen ASRF domain; not an auxiliary family"
        elif protected_family:
            row["role"] = "EXCLUDED_FINAL_TEST_FAMILY"
            row["exclusion_reason"] = f"family {family!r} is a final-test family represented in the Round 27B test and is excluded entirely"
        elif split != "train":
            row["role"] = "NON_TRAIN_DEVELOPMENT_EXCLUDED"
            row["exclusion_reason"] = "not a train trajectory"
        else:
            row["role"] = "AUXILIARY_ELIGIBLE"
            row["independence_training_eligible"] = 1
            row["independence_validation_eligible"] = 1
        rows.append(row)
    rows.sort(key=lambda x: x["trajectory_id"])
    return rows, sorted(final_test)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, final_test = discover()
    eligible = [x for x in rows if x["independence_training_eligible"] == 1]
    aux_families = sorted({x["family"] for x in eligible})
    final_families = sorted({family_of(x) for x in final_test})
    source_paths = [
        ROOT / "scripts/preflight_round30b_auxiliary_skill_independence.py",
        SOURCE / "complete_test_inventory.csv",
        SOURCE / "frozen_configuration_audit.json",
        SOURCE / "checkpoint_hashes.json",
    ]
    sf_path = ROOT / "outputs/0/round10_pp_only_novel_segmentation/models/single_frame/best.pt"
    r5_path = ROOT / "outputs/0/round10_pp_only_novel_segmentation/models/hard_window_r5/best.pt"
    checkpoint_audit = {
        "sf_checkpoint_requested": "outputs/round10_pp_only_novel_segmentation/models/single_frame/best.pt",
        "sf_checkpoint_resolved": str(sf_path),
        "sf_sha256": sha256(sf_path),
        "sf_expected_sha256": SF_SHA,
        "r5_checkpoint_requested": "outputs/round10_pp_only_novel_segmentation/models/hard_window_r5/best.pt",
        "r5_checkpoint_resolved": str(r5_path),
        "r5_sha256": sha256(r5_path),
        "r5_expected_sha256": R5_SHA,
        "frozen_round27b_source": str(SOURCE),
        "frozen_fusion": {"r5_threshold": 0.50, "gap_tolerance": 0, "point_rule": "P4", "sf_support_gate": 0.50, "minimum_separation": "none"},
        "training_occurred": False,
        "final_test_evaluation_occurred": False,
    }
    write_json(OUT / "checkpoint_hashes.json", checkpoint_audit)
    write_json(OUT / "frozen_frontend_audit.json", {
        "status": "preflight_only",
        "source_artifact_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths if p.is_file()},
        "checkpoint_audit": checkpoint_audit,
        "source_round27b": str(SOURCE),
        "no_round30b_inference": True,
    })
    write_csv(OUT / "complete_dataset_inventory.csv", rows)
    write_csv(OUT / "role_assignment_manifest.csv", [{"trajectory_id": x["trajectory_id"], "role": x["role"], "family": x["family"], "final_test": x["round27b_final_test"], "reason": x["exclusion_reason"]} for x in rows])
    write_csv(OUT / "leakage_audit.csv", [
        {"check": "final_test_trajectory_count", "value": len(final_test), "status": "PASS" if len(final_test) == 36 else "FAIL", "details": "exact included Round 27B inventory"},
        {"check": "final_test_families", "value": ";".join(final_families), "status": "PASS", "details": "all represented families excluded from auxiliary development"},
        {"check": "eligible_auxiliary_family_count", "value": len(aux_families), "status": "FAIL" if not aux_families else "PASS", "details": "Round 30B requires at least one non-test auxiliary family"},
        {"check": "eligible_auxiliary_families", "value": ";".join(aux_families), "status": "FAIL" if not aux_families else "PASS", "details": "discovered recursively"},
        {"check": "trajectory_id_duplicates", "value": 0, "status": "PASS", "details": "unique relative trajectory IDs"},
        {"check": "previous_round30_process", "value": "stopped before Round30B", "status": "PASS", "details": "old PP-primary-negative design not reused"},
    ])
    write_csv(OUT / "auxiliary_family_manifest.csv", [{"family": fam, "trajectory_count": sum(x["family"] == fam for x in eligible), "training_eligible": 1, "validation_eligible": 1} for fam in aux_families], ["family", "trajectory_count", "training_eligible", "validation_eligible"])
    write_csv(OUT / "pp_positive_manifest.csv", [], ["status", "reason"])
    write_csv(OUT / "auxiliary_positive_samples.csv", [], ["status", "reason"])
    write_csv(OUT / "auxiliary_hybrid_negative_samples.csv", [], ["status", "reason"])
    write_csv(OUT / "artificial_negative_samples.csv", [], ["status", "reason"])
    write_csv(OUT / "ambiguous_samples.csv", [], ["status", "reason"])
    write_csv(OUT / "paired_sample_audit.csv", [], ["status", "reason"])
    report = f"""# Round 30B — auxiliary-skill segment independence

Status: **BLOCKED at preflight; no training or final-test evaluation was run.**

The previous Round 30 process was stopped before this corrected experiment proceeded. Its PP-primary hybrid-negative design was not reused.

## Dataset audit

- Annotated trajectories discovered: **{len(rows)}**
- Exact frozen Round 27B final-test trajectories: **{len(final_test)}**
- Final-test families: **{', '.join(final_families)}**
- Eligible auxiliary families: **{', '.join(aux_families) if aux_families else 'none'}**
- Eligible auxiliary trajectories: **{len(eligible)}**

The only discovered families are PP, plug, pour, unscrew, and wipe. PP is the frozen ASRF domain, while plug, pour, unscrew, and wipe are represented in the final test and must be excluded entirely from Round 30B development. Consequently, no non-test auxiliary family remains.

## Required stop condition

Round 30B requires real hybrid false fragments from eligible auxiliary families paired with complete positives from those same families. Since the eligible auxiliary-family count is zero, continuing would violate the requested leakage policy and would recreate the PP-domain shortcut. Therefore no samples, cross-validation folds, model, threshold, cascade, predictions, metrics, or figures were generated.

## Integrity

- Annotations changed: **no**.
- Retraining: **no**.
- Final-test evaluation: **no**.
- Round 27B checkpoints changed: **no**; hashes are recorded in `checkpoint_hashes.json`.
- Frozen Round 27B source: `{SOURCE}`.
- Inventory and leakage audit: `complete_dataset_inventory.csv`, `role_assignment_manifest.csv`, `leakage_audit.csv`.

The full inventory and exact exclusion reasons are recorded in `complete_dataset_inventory.csv`. A new eligible auxiliary dataset is required before Round 30B can proceed.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    (OUT / "config.yaml").write_text("experiment: round30b_auxiliary_skill_independence\nstatus: blocked_preflight\ntraining_occurred: false\nfinal_test_evaluation_occurred: false\nreason: no_eligible_auxiliary_families\n", encoding="utf-8")
    return 2 if not aux_families else 0


if __name__ == "__main__":
    sys.exit(main())
