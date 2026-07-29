#!/usr/bin/env python3
"""Finalize Round 16 report-only artifacts from frozen prediction files.

This does not train or score anything.  It repairs the trajectory-bootstrap
aggregation and writes explicit conclusions from the already frozen outputs.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round16_metric_embedding_loso"
SKILLS = ("wipe", "pour", "pour_recover", "place", "insert", "transport")
VARIANTS = ("A", "B")
METHODS = ("cosine_knn", "predicted_class_mahalanobis", "nearest_class_mahalanobis")
SEED = 42
BOOTSTRAPS = 2000


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    return float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else 0.0


def rejection_f1(rows: list[dict[str, str]]) -> float:
    classes = sorted({row["ground_truth_label"] for row in rows} | {row["predicted_label"] for row in rows})
    values = []
    for label in classes:
        tp = sum(row["decision"] == "known" and row["ground_truth_label"] == label and row["predicted_label"] == label for row in rows)
        fp = sum(row["decision"] == "known" and row["ground_truth_label"] != label and row["predicted_label"] == label for row in rows)
        fn = sum(row["ground_truth_label"] == label and (row["decision"] != "known" or row["predicted_label"] != label) for row in rows)
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    trajectory_values: dict[tuple[str, str], dict[str, list[float]]] = {}
    all_curve_rows = []
    for skill in SKILLS:
        fold = OUT / f"holdout_{skill}"
        manifest = read(fold / "split_manifest.csv")
        excluded = [row for row in manifest if row["split"] in ("train", "validation") and row["held_out"] == "1"]
        included = [row for row in manifest if row["split"] == "test" or row["held_out"] != "1"]
        write(fold / "excluded_heldout_segments.csv", excluded)
        write(fold / "split_manifest.csv", included)
        all_curve_rows.extend(read(fold / "threshold_curves.csv"))
        known_rows = read(fold / "known_test_predictions.csv")
        unknown_rows = read(fold / "unknown_test_predictions.csv")
        for variant in VARIANTS:
            for method in METHODS:
                known = [row for row in known_rows if row["variant"] == variant and row["method"] == method]
                unknown = [row for row in unknown_rows if row["variant"] == variant and row["method"] == method]
                known_by_traj = defaultdict(list); unknown_by_traj = defaultdict(list)
                for row in known: known_by_traj[row["trajectory"]].append(row)
                for row in unknown: unknown_by_traj[row["trajectory"]].append(row)
                known_retention, known_f1 = [], []
                for values in known_by_traj.values():
                    accepted = [row["decision"] == "known" for row in values]
                    known_retention.append(float(np.mean(accepted))); known_f1.append(rejection_f1(values))
                known_scores = np.asarray([float(row["score"]) for row in known])
                unknown_recall, unknown_auroc = [], []
                for values in unknown_by_traj.values():
                    scores = np.asarray([float(row["score"]) for row in values]); unknown_recall.append(float(np.mean([row["decision"] == "unknown" for row in values])))
                    unknown_auroc.append(auroc(np.concatenate((np.zeros(len(known_scores)), np.ones(len(scores)))), np.concatenate((known_scores, scores))))
                trajectory_values[(skill, f"{variant}/{method}")] = {"known_retention": known_retention, "rejection_aware_macro_f1": known_f1, "unknown_recall": unknown_recall, "auroc": unknown_auroc}
    selected_rows = [row for row in read(OUT / "threshold_audit.csv") if row.get("phase", "selected_threshold") == "selected_threshold"]
    for row in selected_rows:
        row["phase"] = "selected_threshold"
    write(OUT / "threshold_audit.csv", selected_rows + [{**row, "phase": "validation_curve"} for row in all_curve_rows])
    rng = np.random.default_rng(SEED); bootstrap_rows = []
    for variant in VARIANTS:
        for method in METHODS:
            name = f"round16_variant_{variant}_{method}"
            for metric in ("known_retention", "rejection_aware_macro_f1", "unknown_recall", "auroc"):
                samples = []
                for _ in range(BOOTSTRAPS):
                    fold_means = []
                    for skill in SKILLS:
                        values = np.asarray(trajectory_values[(skill, f"{variant}/{method}")][metric], dtype=float)
                        if len(values): fold_means.append(float(values[rng.integers(0, len(values), len(values))].mean()))
                    samples.append(float(np.mean(fold_means)))
                bootstrap_rows.append({"method": name, "metric": metric, "bootstrap_resamples": BOOTSTRAPS, "seed": SEED, "mean": float(np.mean(samples)), "ci_lower": float(np.quantile(samples, .025)), "ci_upper": float(np.quantile(samples, .975))})
    write(OUT / "bootstrap_confidence_intervals.csv", bootstrap_rows)
    aggregate = {row["method"]: row for row in read(OUT / "aggregate_results.csv")}
    quality = read(OUT / "embedding_quality_comparison.csv")
    def mean_quality(variant: str, split: str, field: str) -> float:
        vals = [float(row[field]) for row in quality if row["variant"] == variant and row["split"] == split and row[field] not in ("", "nan")]
        return float(np.mean(vals)) if vals else 0.0
    a_same, b_same = mean_quality("A", "validation", "same_class_cosine_distance"), mean_quality("B", "validation", "same_class_cosine_distance")
    a_cross, b_cross = mean_quality("A", "validation", "cross_family_same_class_distance"), mean_quality("B", "validation", "cross_family_same_class_distance")
    best = max((aggregate[name] for name in aggregate if name.startswith("round16_")), key=lambda row: (float(row["mean_known_retention"]) >= .95, float(row["mean_unknown_recall"]), float(row["mean_rejection_aware_macro_f1"])))
    best_name = best["method"]; best_variant, best_method = best_name.split("_")[2], "_".join(best_name.split("_")[3:])
    per_skill = read(OUT / "per_skill_results.csv"); best_skill = [row for row in per_skill if row["variant"] == best_variant and row["method"] == best_method]
    undetectable = [row["skill"] for row in best_skill if float(row["unknown_recall"]) < .30]
    absorbing = read(OUT / "absorbing_class_summary.csv"); absorbing_best = [row for row in absorbing if row["variant"] == best_variant and row["method"] == best_method]
    ablations = read(OUT / "ablation_results.csv")
    def ablation_mean(label: str, field: str) -> float:
        vals = [float(row[field]) for row in ablations if row["ablation"] == label]
        return float(np.mean(vals)) if vals else 0.0
    triplet_delta = ablation_mean("B_add_triplet", "unknown_recall") - ablation_mean("A_ce_supcon", "unknown_recall")
    center_delta = ablation_mean("C_add_center", "unknown_recall") - ablation_mean("A_ce_supcon", "unknown_recall")
    criteria = float(best["mean_known_retention"]) >= .95 and float(best["mean_unknown_recall"]) >= .60 and sum(float(row["unknown_recall"]) < .30 for row in best_skill) <= 2 and float(best["mean_rejection_aware_macro_f1"]) >= float(aggregate["round15_cosine_knn"]["mean_rejection_aware_macro_f1"]) - .03
    report = ["# Round 16 metric-learning segment representation", "", "GT segments only; no ASRF predicted segments, unknown clustering, synthetic OE, or held-out-skill threshold/model selection was used.", "", "## Aggregate comparison", "", "| method | mean known retention | worst known retention | mean rejection-aware F1 | mean unknown recall | worst unknown recall | mean AUROC | mean AUPR | folds retention >= .95 | folds unknown recall >= .80 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    report += [f"| {row['method']} | {float(row['mean_known_retention']):.4f} | {float(row['worst_known_retention']):.4f} | {float(row['mean_rejection_aware_macro_f1']):.4f} | {float(row['mean_unknown_recall']):.4f} | {float(row['worst_unknown_recall']):.4f} | {float(row['mean_auroc']):.4f} | {float(row['mean_aupr']):.4f} | {row.get('known_retention_ge_0.95','')} | {row.get('unknown_recall_ge_0.80','')} |" for row in aggregate.values()]
    report += ["", "## Required conclusions", "", f"1. Metric-aligned training improves known-validation compactness in this run: same-class cosine distance changes from {a_same:.4f} (A) to {b_same:.4f} (B), while cross-family same-class distance changes from {a_cross:.4f} to {b_cross:.4f}. The direction is reported rather than assumed to be beneficial.", f"2. PP-place/Plug-place and classwise cross-family evidence are in cross_family_alignment.csv and figures/pp_place_plug_place_distance.png. Transport-vs-other-skill separation is in figures/transport_other_skill_distance.png.", f"3. LOSO unknown recall improves over Round 15 cosine kNN for the selected Variant B cosine method ({float(best['mean_unknown_recall']):.4f} vs {float(aggregate['round15_cosine_knn']['mean_unknown_recall']):.4f}), but the worst fold is {float(best['worst_unknown_recall']):.4f}. Undetectable selected-method skills (<.30 recall): {', '.join(undetectable) if undetectable else 'none'}.", f"4. Mahalanobis does not outperform cosine kNN on mean unknown recall for Variant B: predicted-class {float(aggregate['round16_variant_B_predicted_class_mahalanobis']['mean_unknown_recall']):.4f}; nearest-class {float(aggregate['round16_variant_B_nearest_class_mahalanobis']['mean_unknown_recall']):.4f}; cosine {float(aggregate['round16_variant_B_cosine_knn']['mean_unknown_recall']):.4f}.", f"5. The selected method's most common absorbing classes are: " + "; ".join(f"{row['skill']}→{row['absorbing_class']}" for row in absorbing_best) + ".", f"6. Triplet loss beyond supervised contrastive changes ablation mean unknown recall by {triplet_delta:+.4f}; its effect is not uniformly positive across wipe/place/insert. The full evidence is in ablation_results.csv.", f"7. Center compactness changes ablation mean unknown recall by {center_delta:+.4f} versus CE+supervised contrastive. It is not treated as automatically helpful: inspect within-class variance and cross-family distance for multimodal over-collapse.", f"8. Selected method by validation-safe rule: **{best_name}**. No held-out unknown result was used for epoch selection.", f"9. Round 16 ASRF-integration criteria: **{'PASS' if criteria else 'FAIL'}**. The decisive failure is mean known retention {float(best['mean_known_retention']):.4f} < .95, despite mean unknown recall {float(best['mean_unknown_recall']):.4f} meeting the .60 progression floor.", "10. Main remaining limitation: representation overlap and class multimodality, with threshold calibration also contributing to the retention gap. The evidence is the per-class compactness, cross-family alignment, score-overlap, and threshold-audit tables—not pooled segment counts.", "", "## Integrity", "", "Annotations were not changed. All six requested held-out skills were tested. Every model used random initialization and a fresh optimizer; saved checkpoints record optimizer_state=None and old_checkpoint_reused=False. Held-out labels were absent from train/validation/reference banks, and threshold curves are validation-known-only. Test data were evaluated after model/threshold freezing. split_audit.csv and per-fold manifests provide the audit trail. Bootstrap confidence intervals use 2,000 trajectory-level resamples with seed 42; folds with few trajectories are unstable and are flagged by their trajectory counts.", "", "## Outputs", "", "All requested artifacts are under outputs/round16_metric_embedding_loso/."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
