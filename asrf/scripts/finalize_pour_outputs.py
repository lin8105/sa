"""Finalize reproducible metadata and curves after a pour-only run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asrf.training.checkpointing import checkpoint_manifest


def _float(value: str) -> float:
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/pour_baseline", type=Path)
    args = parser.parse_args()
    output = args.output
    with (output / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validation = [row for row in rows if row["split"] == "val"]
    if not validation:
        raise SystemExit("No validation rows found.")
    selectors = {
        "minimum_validation_total_loss": ("total_loss", min),
        "maximum_refined_f1@50": ("refined_f1@50", max),
        "maximum_refined_edit_score": ("refined_edit_score", max),
        "maximum_internal_boundary_f1@33": ("boundary_33_internal_only_f1", max),
        "maximum_raw_frame_accuracy": ("raw_frame_accuracy", max),
    }
    selection = {}
    for name, (field, function) in selectors.items():
        row = function(validation, key=lambda item: _float(item[field]))
        selection[name] = {"epoch": int(row["epoch"]), "value": _float(row[field]), "field": field}

    starts = []
    for line in (output / "training.log").read_text(encoding="utf-8").splitlines():
        if line.startswith("start_time_epoch="):
            starts.append(datetime.strptime(line.split("=", 1)[1], "%Y-%m-%dT%H:%M:%S%z"))
    end = None
    for line in reversed((output / "training.log").read_text(encoding="utf-8").splitlines()):
        if line.startswith("end_time_epoch="):
            stamp = line.split("=", 1)[1].split(" elapsed_seconds=", 1)[0]
            end = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z")
            break
    summary_path = output / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    epochs = [int(row["epoch"]) for row in rows]
    summary.update({
        "start_epoch": min(epochs),
        "stopping_epoch": max(epochs),
        "metric_selection": selection,
        "training_start_times": [value.isoformat() for value in starts],
        "training_end_time": end.isoformat() if end else None,
        "wall_clock_seconds_including_resume": (end - starts[0]).total_seconds() if starts and end else summary.get("elapsed_seconds"),
        "metrics_row_count": len(rows),
    })
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifests = {name: checkpoint_manifest(output / name) for name in ("best.pt", "last.pt")}
    (output / "checkpoint_manifest.json").write_text(json.dumps(manifests, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    epochs_sorted = sorted({int(row["epoch"]) for row in rows})
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for split, color in (("train", "tab:blue"), ("val", "tab:orange")):
        subset = [row for row in rows if row["split"] == split]
        x = [int(row["epoch"]) for row in subset]
        axes[0].plot(x, [_float(row["total_loss"]) for row in subset], marker=".", label=split, color=color)
        axes[1].plot(x, [_float(row["raw_frame_accuracy"]) for row in subset], linestyle="--", label=f"{split} raw", color=color)
        axes[1].plot(x, [_float(row["refined_frame_accuracy"]) for row in subset], label=f"{split} refined", color=color)
    axes[0].axvline(selection["minimum_validation_total_loss"]["epoch"], color="black", linestyle=":", label="selected epoch")
    axes[0].set_ylabel("total loss")
    axes[1].set_ylabel("frame accuracy")
    axes[1].set_xlabel("epoch")
    axes[0].legend()
    axes[1].legend(ncol=2)
    axes[0].set_title("Pour-only ASRF training curves; validation-only model selection")
    figure.savefig(output / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"selection": selection, "wall_clock_seconds_including_resume": summary["wall_clock_seconds_including_resume"], "checkpoint_manifest": manifests}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
