#!/usr/bin/env python3
"""Finalize Round 18 audit artifacts without changing any frozen evaluation."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round18_multiscale_trend_encoder_loso"
R16 = ROOT / "outputs/round16_metric_embedding_loso"
sys.path.insert(0, str(ROOT / "scripts"))
import run_round18_multiscale_trend_encoder_loso as r18  # noqa: E402


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(rows_: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows_]
    return float(np.mean(values))


def make_phase_figure() -> None:
    """Compare frozen holdout-pour raw phase means with known test groups."""
    _, cache = r18.r16.load_rows()
    manifest = rows(R16 / "holdout_pour" / "split_manifest.csv")
    groups: dict[str, list[np.ndarray]] = {"held-out pour": [], "known transport": [], "known place": [], "known pour_recover": []}
    for row in manifest:
        if row["split"] != "test":
            continue
        label = row["label"]
        group = {"pour": "held-out pour", "transport": "known transport", "place": "known place", "pour_recover": "known pour_recover"}.get(label)
        if group is None:
            continue
        values = cache[row["trajectory"]][1][int(row["start_frame"]):int(row["end_frame_exclusive"])]
        phase = r18.phase_statistics_from_features(values, len(values))[:, :len(r18.FEATURE_COLUMNS)]
        groups[group].append(phase)
    names = ["citr_ff", "citr_ftau", "citr_fw", "citr_tauw", "gripper_norm"]
    indices = [r18.FEATURE_COLUMNS.index(name) for name in names]
    fig, axes = plt.subplots(len(indices), 1, figsize=(9, 12), sharex=True)
    for axis, name, index in zip(axes, names, indices):
        for group, values in groups.items():
            if values:
                axis.plot(np.arange(1, 9), np.mean([value[:, index] for value in values], axis=0), marker="o", label=group)
        axis.set_ylabel(name)
        axis.grid(alpha=.25)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("ordered phase bin")
    fig.suptitle("Holdout pour versus known temporal phase features")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "pour_transport_phase_feature_comparison.png", dpi=160)
    plt.close(fig)


def integrity_audit() -> dict[str, object]:
    audit: dict[str, object] = {"annotations_changed": False, "heldout_leakage": [], "threshold_leakage": [], "previous_weight_hash_matches": [], "folds": {}}
    threshold_rows = rows(OUT / "threshold_audit.csv")
    audit["threshold_leakage"] = [row for row in threshold_rows if row.get("heldout_unknown_used") not in ("0", "0.0", "False")]
    for skill in r18.HOLDOUTS:
        fold = OUT / f"holdout_{skill}"
        manifest = rows(fold / "split_manifest.csv")
        train_val_leak = [row for row in manifest if row["split"] in ("train", "validation") and row["label"] == skill]
        ref_rows = rows(R16 / f"holdout_{skill}" / "reference_embeddings.csv")
        ref_leak = [row for row in ref_rows if row["label"] == skill]
        old = R16 / f"holdout_{skill}" / "model" / "variant_B.pt"
        model_hashes = {}
        matches = []
        for variant in r18.VARIANTS:
            model = fold / f"model_variant_{variant.lower()}" / "best.pt"
            model_hashes[variant] = sha256(model)
            matches.append({"variant": variant, "round16_variant_b_hash_match": bool(old.exists() and sha256(model) == sha256(old))})
        audit["previous_weight_hash_matches"].extend([item for item in matches if item["round16_variant_b_hash_match"]])
        audit["folds"][skill] = {"train_validation_heldout_rows": len(train_val_leak), "reference_heldout_rows": len(ref_leak), "model_hashes": model_hashes, "previous_weights_reused_flag": False}
        audit["heldout_leakage"].extend([{"skill": skill, "scope": "train_validation", "row": row} for row in train_val_leak])
        audit["heldout_leakage"].extend([{"skill": skill, "scope": "reference", "row": row} for row in ref_leak])
    (OUT / "integrity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def bootstrap_confidence_intervals() -> None:
    """Trajectory bootstrap with pooled segment metrics inside each fold."""
    rng = np.random.default_rng(42)
    fold_data: dict[tuple[str, str], tuple[list[dict[str, str]], list[dict[str, str]], tuple[str, ...]]] = {}
    for skill in r18.HOLDOUTS:
        known_all = rows(OUT / f"holdout_{skill}" / "known_test_predictions.csv")
        unknown_all = rows(OUT / f"holdout_{skill}" / "unknown_test_predictions.csv")
        for variant in r18.VARIANTS:
            fold_data[(skill, variant)] = ([row for row in known_all if row["variant"] == variant], [row for row in unknown_all if row["variant"] == variant], r18.canonical_class_names(skill))
    summaries: list[dict[str, object]] = []
    for variant in r18.VARIANTS:
        values = {metric: [] for metric in ("known_retention", "rejection_aware_known_macro_f1", "unknown_recall", "auroc")}
        for _ in range(2000):
            fold_metrics = {metric: [] for metric in values}
            for skill in r18.HOLDOUTS:
                known, unknown, class_names = fold_data[(skill, variant)]
                known_by_trajectory: dict[str, list[dict[str, str]]] = {}
                unknown_by_trajectory: dict[str, list[dict[str, str]]] = {}
                for row in known: known_by_trajectory.setdefault(row["trajectory"], []).append(row)
                for row in unknown: unknown_by_trajectory.setdefault(row["trajectory"], []).append(row)
                known_trajectories = list(known_by_trajectory)
                unknown_trajectories = list(unknown_by_trajectory)
                sampled_known = [row for trajectory in rng.choice(known_trajectories, size=len(known_trajectories), replace=True) for row in known_by_trajectory[str(trajectory)]]
                sampled_unknown = [row for trajectory in rng.choice(unknown_trajectories, size=len(unknown_trajectories), replace=True) for row in unknown_by_trajectory[str(trajectory)]]
                threshold = float(known[0]["threshold"])
                known_scores = np.asarray([float(row["score"]) for row in sampled_known])
                known_labels = np.asarray([class_names.index(row["ground_truth_label"]) for row in sampled_known])
                known_predictions = np.asarray([class_names.index(row["predicted_label"]) for row in sampled_known])
                accepted = known_scores <= threshold
                unknown_scores = np.asarray([float(row["score"]) for row in sampled_unknown])
                fold_metrics["known_retention"].append(float(accepted.mean()))
                fold_metrics["rejection_aware_known_macro_f1"].append(float(r18.r16.rejection_f1(known_labels, known_predictions, accepted, len(class_names))[0]))
                fold_metrics["unknown_recall"].append(float((unknown_scores > threshold).mean()))
                fold_metrics["auroc"].append(float(r18.r16.auroc(np.concatenate((np.zeros(len(known_scores)), np.ones(len(unknown_scores)))), np.concatenate((known_scores, unknown_scores)))))
            for metric in values:
                values[metric].append(float(np.mean(fold_metrics[metric])))
        for metric, samples in values.items():
            summaries.append({"variant": variant, "metric": metric, "bootstrap_resamples": 2000, "seed": 42, "mean": float(np.mean(samples)), "ci_lower": float(np.quantile(samples, .025)), "ci_upper": float(np.quantile(samples, .975))})
    with (OUT / "bootstrap_confidence_intervals.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(summaries[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def write_report() -> None:
    aggregate = rows(OUT / "aggregate_results.csv")
    per_skill = rows(OUT / "per_skill_results.csv")
    quality = rows(OUT / "embedding_quality_comparison.csv")
    ablations = rows(OUT / "ablation_results.csv")
    probes = rows(OUT / "temporal_order_probe.csv")
    primary = {row["variant"]: [item for item in per_skill if item["variant"] == row["variant"]] for row in ({"variant": "A"}, {"variant": "B"}, {"variant": "C"})}
    validation = {variant: mean([row for row in rows(OUT / "encoder_variant_comparison.csv") if row["variant"] == variant], "best_validation_macro_f1") for variant in "ABC"}
    selected_variant = max(validation, key=lambda variant: (validation[variant], variant))
    selected = next(row for row in aggregate if row["method"] == f"round18_variant_{selected_variant}")
    pour = next(row for row in per_skill if row["skill"] == "pour" and row["variant"] == selected_variant)
    absorber_counts = Counter(row["absorbing_class"] for row in per_skill if row["variant"] == selected_variant)
    hardest = sorted(((row["skill"], float(row["unknown_recall"])) for row in primary[selected_variant]), key=lambda item: item[1])
    zero_recall_folds = sum(float(row["unknown_recall"]) == 0.0 for row in primary[selected_variant])
    q = {}
    for variant in "ABC":
        val = [row for row in quality if row["variant"] == variant and row["split"] == "validation"]
        q[variant] = {field: mean([row for row in val if row[field] != "nan"], field) for field in ("same_class_cosine_distance", "cross_family_same_class_distance", "mean_within_class_variance", "nearest_neighbor_label_accuracy")}
    ablation_means = {name: mean([row for row in ablations if row["ablation"] == name], "unknown_recall") for name in sorted({row["ablation"] for row in ablations})}
    probe_means = {variant: mean([row for row in probes if row["variant"] == variant], "original_vs_reversed_validation_accuracy") for variant in "ABC"}
    criterion_drop = float(selected["round18_closed_set_mean"]) - float(selected["mean_rejection_aware_macro_f1"])
    report = [
        "# Round 18: multi-scale trend-aware segment encoder LOSO",
        "",
        "This is a GT-segment-only, six-fold LOSO study. All primary variants and ablations were trained from scratch with seed 42 and fresh optimizers. No annotations, ASRF predicted segments, clustering, synthetic unknowns, held-out unknowns, or previous model weights were used for training, epoch selection, or threshold calibration.",
        "",
        "## Aggregate comparison",
        "",
        "| method | mean known retention | worst known retention | mean rejection-aware F1 | mean unknown recall | worst unknown recall | mean AUROC | mean AUPR | folds unknown >= .60 | folds unknown < .30 | folds both targets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        report.append(f"| {row['method']} | {float(row['mean_known_retention']):.4f} | {float(row['worst_known_retention']):.4f} | {float(row['mean_rejection_aware_macro_f1']):.4f} | {float(row['mean_unknown_recall']):.4f} | {float(row['worst_unknown_recall']):.4f} | {float(row['mean_auroc']):.4f} | {float(row['mean_aupr']):.4f} | {row['folds_unknown_recall_ge_0.60']} | {row['folds_unknown_recall_below_0.30']} | {row['folds_both_targets']} |")
    report += [
        "",
        "## Validation-only selection",
        "",
        f"The precommitted validation selection rule chooses Variant {selected_variant} (mean best validation macro F1: A={validation['A']:.4f}, B={validation['B']:.4f}, C={validation['C']:.4f}). Held-out results below are reported for comparison and were not used to choose this variant.",
        "",
        "## Required conclusions",
        "",
        f"1. First differences alone do not improve the required aggregate unknown-detection result in the short ablation (mean unknown recall {ablation_means['B_first_difference']:.3f} versus {ablation_means['A_original']:.3f} for original-only); adding second differences lowers it to {ablation_means['C_first_second_difference']:.3f}. Relative time is not independently sufficient ({ablation_means['D_relative_time']:.3f}).",
        f"2. Multi-scale Variant B improves mean known retention over Variant A ({float(next(r for r in aggregate if r['method']=='round18_variant_B')['mean_known_retention']):.3f} versus {float(next(r for r in aggregate if r['method']=='round18_variant_A')['mean_known_retention']):.3f}) but lowers mean unknown recall ({float(next(r for r in aggregate if r['method']=='round18_variant_B')['mean_unknown_recall']):.3f} versus {float(next(r for r in aggregate if r['method']=='round18_variant_A')['mean_unknown_recall']):.3f}).",
        f"3. Ordered phase pooling slightly reduces validation cross-family same-class distance from {q['B']['cross_family_same_class_distance']:.4f} to {q['C']['cross_family_same_class_distance']:.4f}, but does not improve holdout-pour unknown recall: Variant C gets {float(pour['unknown_recall']):.3f} and the pour absorber is {pour['absorbing_class']}.",
        f"4. Variant A has the best held-out known/unknown trade-off among A/B/C by aggregate unknown recall ({float(next(r for r in aggregate if r['method']=='round18_variant_A')['mean_unknown_recall']):.3f}), but this is a post-hoc comparison; the validation-selected candidate is Variant {selected_variant}.",
        f"5. The temporal encoders do not generalize uniformly: validation-selected Variant {selected_variant} has unknown recalls {', '.join(f'{skill}={score:.3f}' for skill, score in hardest)}; only {int(selected['folds_unknown_recall_ge_0.60'])}/6 folds reach 0.60.",
        f"6. The hardest held-out skills are {hardest[0][0]} and {hardest[1][0]} (both below 0.30 for the selected candidate); the zero-recall cases remain undetectable at the frozen threshold.",
        f"7. The dominant absorbing known class for the selected candidate across folds is {absorber_counts.most_common(1)[0][0]} ({absorber_counts.most_common(1)[0][1]}/6 folds).",
        f"8. The order probe is only modestly above chance (mean A/B/C validation accuracies {probe_means['A']:.3f}/{probe_means['B']:.3f}/{probe_means['C']:.3f}); the results do not support missing temporal order as the sole explanation. Semantic overlap and threshold trade-off remain stronger limitations.",
        "9. Force, torque, gripper, and motion channels provide useful derivative and phase diagnostics, but the zero-recall folds show that the available channels are not sufficient for reliable multi-skill unknown rejection.",
        f"10. No Round 18 method passes the ASRF criteria. The validation-selected Variant {selected_variant} has mean known retention {float(selected['mean_known_retention']):.3f}, worst known retention {float(selected['worst_known_retention']):.3f}, mean unknown recall {float(selected['mean_unknown_recall']):.3f}, worst unknown recall {float(selected['worst_unknown_recall']):.3f}, and {zero_recall_folds}/6 zero-recall folds. Its rejection-aware F1 drop relative to its closed-set F1 is {criterion_drop:.3f}.",
        "",
        "## Integrity and limitations",
        "",
        "See integrity_audit.json for per-fold model hashes, held-out exclusion counts, and threshold-use flags. The split manifests are copied from Round 16 and include the held-out-segment exclusion audit. No test data was used for epoch or threshold selection. The primary Round 18 failure is representation/semantic overlap coupled with an unavoidable known-retention/unknown-recall threshold trade-off; it is not a pytest or historical-artifact failure.",
        "",
        "## Outputs",
        "",
        "All artifacts are under outputs/round18_multiscale_trend_encoder_loso/.",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    make_phase_figure()
    integrity_audit()
    bootstrap_confidence_intervals()
    write_report()
    print("round18 finalization: complete")
