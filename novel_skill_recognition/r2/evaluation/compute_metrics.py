#!/usr/bin/env python
"""Recompute Round94 AUROC, AUPR-OOD, FPR95, OSCR and trajectory bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, roc_auc_score,
                             roc_curve)

OLD_SKILLS = {"reach", "grasp", "lift", "transport", "place", "release", "retreat"}


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["matched_gt_known_or_novel"].eq("novel").astype(int).to_numpy()
    score = frame["r2_novelty_score"].to_numpy(float)
    if np.unique(y).size != 2:
        raise ValueError("a metric sample must contain both OLD and novel segments")
    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
    fpr95 = float(fpr[np.flatnonzero(tpr >= 0.95)[0]])
    correct_old = (frame["matched_gt_known_or_novel"].eq("known") &
                   frame["asrf_predicted_skill"].eq(frame["matched_gt_skill"])).to_numpy()
    # OSCR accepts low-novelty scores as OLD; ties enter together.
    cutoffs = np.r_[-np.inf, np.unique(score), np.inf]
    novel = y == 1
    old = ~novel
    ccr = np.asarray([np.sum(correct_old & old & (score < c)) / old.sum() for c in cutoffs])
    false_accept = np.asarray([np.sum(novel & (score < c)) / novel.sum() for c in cutoffs])
    order = np.argsort(false_accept, kind="stable")
    return {"auroc": float(roc_auc_score(y, score)),
            "aupr_ood": float(average_precision_score(y, score)),
            "fpr95": fpr95,
            "oscr": float(auc(false_accept[order], ccr[order]))}


def trajectory_bootstrap(frame: pd.DataFrame, replicates: int = 1000,
                         seed: int = 94094) -> dict[str, dict[str, float]]:
    # The released Round94 bootstrap used lexicographically sorted trajectory IDs.
    ids = np.asarray(sorted(frame["trajectory_id"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in ("auroc", "aupr_ood", "fpr95", "oscr")}
    for _ in range(replicates):
        selected = rng.choice(ids, size=len(ids), replace=True)
        sample = pd.concat([frame.loc[frame.trajectory_id.eq(tid)] for tid in selected],
                           ignore_index=True)
        if sample["matched_gt_known_or_novel"].nunique() != 2:
            continue
        for key, value in metric_values(sample).items():
            draws[key].append(value)
    result = {}
    for key, values in draws.items():
        arr = np.asarray(values, dtype=float)
        result[key] = {"median": float(np.median(arr)),
                       "ci95_low": float(np.quantile(arr, 0.025)),
                       "ci95_high": float(np.quantile(arr, 0.975)),
                       "valid_replicates": int(len(arr))}
    return result


def evaluate(frame: pd.DataFrame, replicates: int = 1000,
             seed: int = 94094) -> dict[str, object]:
    labels = frame["matched_gt_known_or_novel"].astype(str)
    if not labels.isin({"known", "novel"}).all():
        raise ValueError("the final benchmark must not contain unmatched predictions")
    point = metric_values(frame)
    return {"trajectory_count": int(frame.trajectory_id.nunique()),
            "segment_count": int(len(frame)),
            "old_count": int(labels.eq("known").sum()),
            "novel_count": int(labels.eq("novel").sum()),
            "unmatched_count": 0, "point": point,
            "bootstrap": trajectory_bootstrap(frame, replicates, seed)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path, help="final per-ASRF-segment score CSV")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=94094)
    args = parser.parse_args()
    result = evaluate(pd.read_csv(args.scores), args.replicates, args.seed)
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
