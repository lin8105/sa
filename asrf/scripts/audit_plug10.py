"""Read-only Plug-10 continuation audit; exits nonzero on invalid annotations."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round9_incremental_learning/plug/n10"
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")

import sys

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from prepare_round9_stage1 import audit_recording
from asrf.data.labels import load_label_mapping


def natural_key(value: str) -> tuple[str, int, str]:
    stem = Path(value).name
    prefix = "".join(character for character in stem if not character.isdigit())
    digits = "".join(character for character in stem if character.isdigit())
    return prefix, int(digits or 0), value


def write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    mapping = load_label_mapping(ROOT / "configs/labels_multitask_plug.yaml")
    requested = [f"train/plug/p{i}" for i in range(1, 11)]
    requested += [f"test/plug/{name}" for name in ("p1", "p2", "p3", "po1", "po2")]
    requested += [f"train/pick and place/pp{i}" for i in range(1, 21)]
    rows1 = [audit_recording(entry, mapping) for entry in requested]
    rows2 = [audit_recording(entry, mapping) for entry in requested]
    digest1 = write_csv(OUT / "data_audit_scan1.csv", rows1)
    digest2 = write_csv(OUT / "data_audit_scan2.csv", rows2)
    invalid = [row for row in rows1 if row.get("errors")]
    selected = [row for row in rows1 if row["trajectory"] in requested[:10]]
    summary = {
        "scan_rows_identical": rows1 == rows2,
        "scan1_sha256": digest1,
        "scan2_sha256": digest2,
        "requested_train_plug_count": 10,
        "valid_train_plug_count": sum(bool(row["compatible_with_round12_ontology"]) for row in selected),
        "selected_train_plug_paths": requested[:10],
        "invalid_trajectories": [row["trajectory"] for row in invalid],
        "invalid_row_count": len(invalid),
        "pass": rows1 == rows2 and not invalid and len(selected) == 10,
        "stop_before_training": bool(rows1 != rows2 or invalid or len(selected) != 10),
        "failure_reasons": sorted({error for row in invalid for error in str(row.get("errors", "")).split(";") if error}),
        "ontology_version": "round12_multiskill_v2",
        "ontology": {"reach": 0, "grasp": 1, "lift": 2, "transport": 3, "pour": 4, "pour_recover": 5, "place": 6, "release": 7, "wipe": 8, "retreat": 9, "insert": 10},
        "aliases": {"pull_out": "lift", "extract": "lift"},
        "external_dataset_modified": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
