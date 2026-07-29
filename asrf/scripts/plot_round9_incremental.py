"""Generate the revised Round 9 incremental learning-curve figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round9_incremental_learning"
FIG = OUT / "figures"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def num(value: str) -> float:
    return float(value)


def x(value: str) -> float:
    return 99.0 if value == "all" else float(value)


def curve_position(value: str) -> int:
    return {"3": 0, "5": 1, "all": 2}[str(value)]


def set_curve_ticks(axis: plt.Axes) -> None:
    axis.set_xticks([0, 1, 2], ["3 + pp10", "5 + pp10", "all + pp10"])


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)


def main() -> int:
    task = rows(OUT / "task_learning_curve.csv")
    skills = rows(OUT / "per_skill_learning_curve.csv")
    support = rows(OUT / "training_support.csv")
    transitions = rows(OUT / "target_transition_boundary_metrics.csv")

    for family, names, title, filename in (("pour", ("pour", "pour_recover"), "Pour target-skill learning", "pour_target_skill_f1_vs_trajectories.png"), ("wipe", ("wipe",), "Wipe target-skill learning", "wipe_target_skill_f1_vs_trajectories.png"), ("plug", ("place", "insert"), "Plug target-skill learning", "plug_place_insert_f1_vs_trajectories.png")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for skill in names:
            values = sorted([row for row in skills if row["target_family"] == family and row["skill"] == skill], key=lambda row: x(row["target_trajectory_count"]))
            ax.plot([curve_position(row["target_trajectory_count"]) for row in values], [num(row["official_F1"]) for row in values], marker="o", label=skill)
        set_curve_ticks(ax); ax.set_ylim(0, 1.05); ax.set_ylabel("official segment F1"); ax.set_title(title); ax.legend(); save(fig, filename)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for skill in ("pour", "pour_recover", "wipe", "place", "insert"):
        values = [row for row in skills if row["skill"] == skill and row["primary_target_skill"] == "True"]
        ax.scatter([num(row["train_segments"]) for row in values], [num(row["official_F1"]) for row in values], label=skill)
    ax.set_xlabel("target-skill training segments"); ax.set_ylabel("official segment F1"); ax.set_title("Target-skill F1 versus training segments"); ax.legend(); save(fig, "target_skill_f1_vs_segments.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for skill in ("pour", "pour_recover", "wipe", "place", "insert"):
        values = [row for row in skills if row["skill"] == skill and row["primary_target_skill"] == "True"]
        ax.scatter([num(row["train_frames"]) for row in values], [num(row["official_frame_F1"]) for row in values], label=skill)
    ax.set_xlabel("target-skill training frames"); ax.set_ylabel("official frame F1"); ax.set_title("Target-skill frame F1 versus training frames"); ax.legend(); save(fig, "target_skill_f1_vs_frames.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for family in ("pour", "wipe", "plug"):
        values = [row for row in transitions if row["target_family"] == family]
        for transition in sorted({row["transition"] for row in values}):
            selected = sorted([row for row in values if row["transition"] == transition], key=lambda row: x(row["target_trajectory_count"]))
            ax.plot([curve_position(row["target_trajectory_count"]) for row in selected], [num(row["boundary_recall_33"]) for row in selected], marker="o", label=transition)
    set_curve_ticks(ax); ax.set_ylim(0, 1.05); ax.set_ylabel("boundary recall ±33"); ax.set_title("Target-transition boundary recall"); ax.legend(fontsize=7); save(fig, "target_transition_boundary_recall.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for family in ("pour", "wipe", "plug"):
        selected = sorted([row for row in task if row["target_family"] == family], key=lambda row: x(row["target_trajectory_count"]))
        ax.plot([curve_position(row["target_trajectory_count"]) for row in selected], [num(row["official_F1_50"]) for row in selected], marker="o", label=family)
    set_curve_ticks(ax); ax.set_ylim(0, 1.05); ax.set_ylabel("official F1@50"); ax.set_title("Overall refined F1@50"); ax.legend(); save(fig, "overall_f1_vs_trajectories.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for family in ("pour", "wipe", "plug"):
        selected = [row for row in skills if row["target_family"] == family and row["skill"] in {"reach", "grasp", "lift", "transport", "place", "release"}]
        grouped = {}
        for row in selected: grouped.setdefault(row["target_trajectory_count"], []).append(num(row["official_frame_F1"]))
        ordered = sorted(grouped, key=x)
        ax.plot([curve_position(value) for value in ordered], [np.mean(grouped[value]) for value in ordered], marker="o", label=family)
    set_curve_ticks(ax); ax.set_ylim(0, 1.05); ax.set_ylabel("shared-skill official frame F1"); ax.set_title("Shared-skill retention on primary test sets"); ax.legend(); save(fig, "shared_skill_retention.png")
    print(f"wrote {len(list(FIG.glob('*.png')))} figures to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
