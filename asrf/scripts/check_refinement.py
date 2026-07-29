"""Synthetic refinement examples and diagnostic figure for round 3."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/.matplotlib"))
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asrf.refinement import refine_asrf_predictions


def probabilities(labels: list[int], *, classes: int = 3) -> torch.Tensor:
    result = torch.full((1, classes, len(labels)), 0.05)
    for index, label in enumerate(labels):
        result[0, label, index] = 0.9
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/round3_diagnostics/refinement_synthetic.png")
    args = parser.parse_args()
    raw = [0, 0, 1, 0, 0, 0, 2, 2, 2]
    ground_truth = [0, 0, 0, 0, 0, 0, 2, 2, 2]
    boundary = torch.tensor([[[0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.85, 0.1, 0.1]]])
    mask = torch.ones(1, len(raw), dtype=torch.bool)
    asb = probabilities(raw)
    result = refine_asrf_predictions(asb, boundary, mask, threshold=0.5, voting="majority")
    mean_result = refine_asrf_predictions(asb, boundary, mask, threshold=0.5, voting="mean_probability")
    print(f"raw_labels={result.raw_labels[0].tolist()}")
    print(f"brb_probabilities={boundary[0, 0].tolist()}")
    print(f"selected_boundaries={list(result.selected_boundaries[0])}")
    print(f"intervals={[(interval.start, interval.end) for interval in result.intervals[0]]}")
    print(f"majority_result={result.refined_labels[0].tolist()}")
    print(f"mean_probability_result={mean_result.refined_labels[0].tolist()}")

    output = Path(__file__).resolve().parents[1] / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    x = list(range(len(raw)))
    figure, axes = plt.subplots(5, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes[0].step(x, ground_truth, where="mid")
    axes[0].set_ylabel("ground truth")
    axes[1].step(x, result.raw_labels[0].tolist(), where="mid")
    axes[1].set_ylabel("raw ASB")
    axes[2].plot(x, boundary[0, 0].tolist(), marker="o")
    axes[2].axhline(0.5, color="tab:red", linestyle="--", linewidth=1)
    for index in result.selected_boundaries[0]:
        axes[2].axvline(index, color="tab:green", linestyle=":")
    axes[2].set_ylabel("BRB p")
    axes[3].step(x, result.refined_labels[0].tolist(), where="mid")
    axes[3].set_ylabel("majority")
    axes[4].step(x, mean_result.refined_labels[0].tolist(), where="mid")
    axes[4].set_ylabel("mean prob.")
    axes[4].set_xlabel("heatmap frame")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(f"figure={output}")
    print("refinement_smoke=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
