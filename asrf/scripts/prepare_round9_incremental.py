"""Stage the revised Round 9 incremental target-family protocol without training."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round9_incremental_learning"
SPLITS = ROOT / "splits/round9_incremental"
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")

import sys

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from prepare_round9_stage1 import CANONICAL, audit_recording, scan  # noqa: E402


def natural_key(value: str) -> tuple[str, int, str]:
    stem = Path(value).name
    prefix = "".join(character for character in stem if not character.isdigit())
    digits = "".join(character for character in stem if character.isdigit())
    return prefix, int(digits or 0), value


def write_split(name: str, entries: list[str]) -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    (SPLITS / name).write_text("\n".join(entries) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["trajectory"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family(rows: list[dict[str, object]], split: str, name: str) -> list[str]:
    return sorted(
        [str(row["trajectory"]) for row in rows if row["split"] == split and row["task_family"] == name and row["compatible_with_round12_ontology"]],
        key=natural_key,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows1 = scan()
    rows2 = scan()
    digest1 = write_csv(OUT / "data_audit_scan1.csv", rows1)
    digest2 = write_csv(OUT / "data_audit_scan2.csv", rows2)
    errors = [row for row in rows1 if row.get("errors")]
    audit_summary = {
        "scan_rows_identical": rows1 == rows2,
        "scan1_sha256": digest1,
        "scan2_sha256": digest2,
        "row_count": len(rows1),
        "invalid_trajectories": [row["trajectory"] for row in errors],
        "invalid_row_count": len(errors),
        "pass": rows1 == rows2 and not errors,
        "ontology": list(CANONICAL),
        "aliases": {"pull_out": "lift", "extract": "lift"},
        "external_dataset_modified": False,
    }
    (OUT / "data_audit_summary.json").write_text(json.dumps(audit_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not audit_summary["pass"]:
        raise SystemExit("Round 9 incremental audit failed; training is prohibited.")

    valid_train = {str(row["trajectory"]): row for row in rows1 if row["split"] == "train" and row["compatible_with_round12_ontology"]}
    base = [f"train/pick and place/pp{i}" for i in range(1, 11)]
    if any(entry not in valid_train for entry in base):
        raise SystemExit("base_pp10 contains an invalid or missing trajectory")
    common_val = [f"train/pick and place/pp{i}" for i in range(11, 21)]
    if any(entry not in valid_train for entry in common_val):
        raise SystemExit("common validation set contains an invalid or missing trajectory")
    if set(base) & set(common_val):
        raise SystemExit("base and common validation overlap")
    write_split("base_pp10.txt", base)
    write_split("common_validation.txt", common_val)

    target_pools = {
        "pour": family(rows1, "train", "pour"),
        "wipe": family(rows1, "train", "wipe"),
        "plug": family(rows1, "train", "plug"),
    }
    target_tests = {
        "pour": family(rows1, "test", "pour")[:2],
        "wipe": family(rows1, "test", "wipe")[:2],
        "plug": family(rows1, "test", "plug"),
    }
    if len(target_tests["pour"]) != 2 or len(target_tests["wipe"]) != 2 or not target_tests["plug"]:
        raise SystemExit("required independent primary test set is unavailable")
    for name, entries in target_tests.items():
        write_split(f"test_{name}_primary.txt", entries)
    for name, entries in target_pools.items():
        write_split(f"{name}_train_3.txt", entries[:3])
        write_split(f"{name}_train_5.txt", entries[:5])
        write_split(f"{name}_train_all.txt", entries)
        for size, subset in ((3, entries[:3]), (5, entries[:5]), ("all", entries)):
            write_split(f"{name}_train_{size}_with_base_pp10.txt", base + subset)

    base_support = {skill: {"frames": 0, "segments": 0} for skill in CANONICAL}
    by_trajectory = {row["trajectory"]: row for row in rows1}
    for entry in base:
        row = by_trajectory[entry]
        for skill in CANONICAL:
            base_support[skill]["frames"] += int(row[f"{skill}_frames"])
            base_support[skill]["segments"] += int(row[f"{skill}_segments"])

    manifest = {
        "base_pp10": base,
        "common_validation": common_val,
        "target_training_pools": target_pools,
        "primary_test": target_tests,
        "test_policy": "pour p1/p2, wipe w1/w2, and every valid independent test/plug trajectory; no train trajectory is used as test",
        "base_support": base_support,
        "ontology": list(CANONICAL),
        "nested_subsets": True,
        "initialization_policy": "same verified Round 8 hard_window_r5 ten-class checkpoint for every run; no sequential subset initialization",
        "validation_policy": "fixed pp11-pp20 trajectory-level validation for every family and size",
    }
    (OUT / "test_split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan = {
        "order": ["pour-3", "pour-5", "wipe-3", "wipe-5", "plug-3", "plug-5", "pour-all", "wipe-all", "plug-all"],
        "primary_model_count": 9,
        "model_count_by_family": {"pour": 3, "wipe": 3, "plug": 3},
        "round8_reference_checkpoint": "outputs/brb_release_round8/hard_window_r5/best.pt",
        "round8_reference_sha256": "61f32711d6de9e8c3809a0c1447459cb754adb31d3a0be8c9a0ba06f9b9c35af",
        "reference_duration_s": 988.1193370819092,
        "estimated_training_duration_s": 9 * 988.1193370819092,
        "estimated_training_duration_h": 9 * 988.1193370819092 / 3600.0,
        "exceeds_nine_model_limit": False,
        "no_task_specific_training": True,
    }
    (OUT / "stage1_run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit_summary, "manifest": manifest, "plan": plan}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
