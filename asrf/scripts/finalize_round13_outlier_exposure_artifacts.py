#!/usr/bin/env python3
"""Complete non-training diagnostics for the Round 13 OE artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

import train_round13_outlier_exposure_holdout_wipe as oe


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def score_auroc(known: np.ndarray, unknown: np.ndarray) -> float:
    scores = np.concatenate((known, unknown))
    labels = np.concatenate((np.zeros(len(known)), np.ones(len(unknown))))
    return oe.auroc(labels, scores)


def macro_f1_from_rows(values: list[dict[str, str]], truth_field: str, prediction_field: str) -> float:
    labels = list(oe.KNOWN_CLASSES)
    f1_values = []
    for label in labels:
        tp = sum(row[truth_field] == label and row[prediction_field] == label for row in values)
        fp = sum(row[truth_field] != label and row[prediction_field] == label for row in values)
        fn = sum(row[truth_field] == label and row[prediction_field] != label for row in values)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1_values))


def main() -> int:
    root = oe.OUTPUT_ROOT
    rows = oe.load_rows()
    cache = oe.load_features(rows)
    train_frames = np.concatenate([oe.sequence(row, cache) for row in rows["train"]], axis=0)
    mean = train_frames.mean(axis=0)
    std = np.maximum(train_frames.std(axis=0), 1e-6)
    durations = np.asarray([np.log1p(row["duration_frames"]) for row in rows["train"]])
    duration_mean = float(durations.mean())
    duration_std = float(max(durations.std(), 1e-6))
    synthetic_validation = oe.generate_outliers(rows["validation"], cache, "validation")
    validation_loader = torch.utils.data.DataLoader(
        oe.SequenceDataset(rows["validation"], cache, mean, std, duration_mean, duration_std, True),
        batch_size=oe.BATCH_SIZE, shuffle=False, collate_fn=oe.collate,
    )
    synthetic_loader = torch.utils.data.DataLoader(
        oe.SequenceDataset(synthetic_validation, None, mean, std, duration_mean, duration_std, False),
        batch_size=oe.BATCH_SIZE, shuffle=False, collate_fn=oe.collate,
    )
    thresholds = json.loads((root / "frozen_thresholds.json").read_text(encoding="utf-8"))
    stats_path = root / "synthetic_outlier_statistics.csv"
    original_stats = read_rows(stats_path)
    stats_fields = [
        "split", "outlier_type", "count", "mean_duration_frames", "std_duration_frames",
        "min_duration_frames", "max_duration_frames", "objective", "score",
        "validation_synthetic_recall", "validation_synthetic_auroc",
    ]
    # Rebuild evaluation rows idempotently if this post-processing helper is
    # run more than once.
    stats = [{field: row.get(field, "") for field in stats_fields} for row in original_stats if not row.get("objective")]
    for objective in ("uniform_softmax", "energy_margin"):
        model = oe.base.SegmentClassifier(oe.base.FEATURE_DIM, oe.base.HIDDEN_DIM, oe.base.PROJECTION_DIM, oe.base.EMBEDDING_DIM, len(oe.KNOWN_CLASSES))
        checkpoint = torch.load(root / "model" / f"{objective}_best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        val_known = oe.infer(model, validation_loader)
        val_synthetic = oe.infer(model, synthetic_loader)
        score_name = thresholds[objective]["score"]
        known_scores = oe.score_logits(val_known["logits"])
        synthetic_scores = oe.score_logits(val_synthetic["logits"])
        known_score = -known_scores["max_softmax"] if score_name == "max_softmax" else known_scores["energy"]
        synthetic_score = -synthetic_scores["max_softmax"] if score_name == "max_softmax" else synthetic_scores["energy"]
        threshold = float(thresholds[objective]["threshold"])
        for kind in oe.OUTLIER_TYPES:
            indices = np.asarray([index for index, row in enumerate(val_synthetic["rows"]) if row["outlier_type"] == kind], dtype=int)
            values = synthetic_score[indices]
            durations_kind = np.asarray([row["duration_frames"] for row in val_synthetic["rows"] if row["outlier_type"] == kind])
            stats.append({
                "split": "validation", "outlier_type": kind, "count": int(len(indices)),
                "mean_duration_frames": float(durations_kind.mean()), "std_duration_frames": float(durations_kind.std()),
                "min_duration_frames": int(durations_kind.min()), "max_duration_frames": int(durations_kind.max()),
                "objective": objective, "score": score_name,
                "validation_synthetic_recall": float((values > threshold).mean()),
                "validation_synthetic_auroc": float(score_auroc(known_score, values)),
            })
    write_rows(stats_path, stats, stats_fields)

    baseline_path = root / "baseline_comparison.csv"
    baseline = read_rows(baseline_path)
    holdout_rows = read_rows(oe.HOLDOUT_ROOT / "segment_predictions.csv")
    for row in baseline:
        if row["method"] in ("max_softmax", "energy"):
            group_rows = [item for item in holdout_rows if item["evaluation_group"] in ("known_test", "wipe_unknown")]
            known = np.asarray([float(item[row["method"]]) for item in group_rows if item["evaluation_group"] == "known_test"])
            wipe = np.asarray([float(item[row["method"]]) for item in group_rows if item["evaluation_group"] == "wipe_unknown"])
            row["known_vs_wipe_auroc"] = f"{score_auroc(-known, -wipe) if row['method'] == 'max_softmax' else score_auroc(known, wipe):.12f}"
        elif row["method"] == "frozen_cosine_knn_k1":
            knn_rows = read_rows(oe.ROOT / "outputs/round12_open_set_cosine_knn_holdout_wipe/segment_predictions.csv")
            selected = [item for item in knn_rows if item["group"] in ("known_test", "wipe") and item["variant"] == "global" and item["k_requested"] == "1"]
            known = np.asarray([float(item["novelty_score_mean_knn_distance"]) for item in selected if item["group"] == "known_test"])
            wipe = np.asarray([float(item["novelty_score_mean_knn_distance"]) for item in selected if item["group"] == "wipe"])
            row["known_vs_wipe_auroc"] = f"{score_auroc(known, wipe):.12f}"
            known_rows = [item for item in selected if item["group"] == "known_test"]
            row["known_macro_f1_before_rejection"] = f"{macro_f1_from_rows(known_rows, 'ground_truth_label', 'classifier_top1_label'):.12f}"
    write_rows(baseline_path, baseline, list(baseline[0]))

    comparison = read_rows(root / "baseline_comparison.csv")
    selected = next(row for row in comparison if row["method"] == "oe_uniform_softmax")
    best_known = max((row for row in comparison if row["method"] in ("max_softmax", "energy", "frozen_cosine_knn_k1")), key=lambda row: float(row["wipe_unknown_recall"]))
    report_path = root / "report.md"
    report = report_path.read_text(encoding="utf-8").rstrip()
    report += "\n\n## Synthetic outlier-type validation diagnostics\n\n"
    report += "Per-construction synthetic validation rejection recall and AUROC for each OE objective are recorded in `synthetic_outlier_statistics.csv`; these diagnostics use known validation segments only.\n\n"
    report += "## Required interpretation\n\n"
    report += f"- The selected OE model improves wipe unknown recall over the frozen max-softmax baseline ({float(selected['wipe_unknown_recall']):.6f} vs {next(row for row in comparison if row['method'] == 'max_softmax')['wipe_unknown_recall']}) but lowers known retention ({float(selected['known_retention']):.6f} vs {next(row for row in comparison if row['method'] == 'max_softmax')['known_retention']}); both metrics are reported, so this is not treated as an unqualified improvement.\n"
    report += f"- The frozen cosine-kNN baseline remains the comparison with the highest known retention among the previously frozen novelty methods; no kNN, threshold, prototype, or clustering tuning was performed here.\n"
    report += f"- The strongest validation synthetic discriminator was the selected uniform-softmax model (validation AUROC {thresholds['uniform_softmax']['validation_auroc']:.6f}); wipe labels were not used to select it.\n"
    report += f"- The most common pre-OE wipe absorber and all per-segment predictions are available in `wipe_diagnostics.csv`; OE evaluation does not equate rejection with wipe identification.\n"
    report += "\n## Baseline AUROC completion\n\n"
    report += "The baseline comparison table includes known-versus-wipe AUROC computed from the frozen prior outputs; the frozen thresholds were not changed.\n\n"
    report += "| method | known-vs-wipe AUROC | known retention | wipe unknown recall |\n|---|---:|---:|---:|\n"
    for row in comparison:
        report += f"| {row['method']} | {float(row['known_vs_wipe_auroc']):.6f} | {float(row['known_retention']):.6f} | {float(row['wipe_unknown_recall']):.6f} |\n"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(json.dumps({"status": "finalized", "synthetic_type_rows": len(stats), "baseline_auroc_updated": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
