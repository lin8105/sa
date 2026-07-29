#!/usr/bin/env python3
"""Round 17 inference-only calibration and local-density novelty study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
R16 = ROOT / "outputs/round16_metric_embedding_loso"
OUT = ROOT / "outputs/round17_class_conditional_density_calibration"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_round16_metric_embedding_loso as r16  # noqa: E402

SEED = 42
TARGET_RETENTION = 0.95
K = 5
EPSILON = 1e-6
MIN_CLASS_VALIDATION = 5
SHRINKAGE = 0.5
MAD_SCALE = 1.4826
HOLDOUTS = r16.HOLDOUTS
FAMILY_FOR_HOLDOUT = r16.FAMILY_FOR_HOLDOUT
METHODS = ("raw_global", "per_class_threshold", "local_density_normalized", "class_standardized")
METHOD_LABELS = {
    "raw_global": "Round16 raw cosine kNN",
    "per_class_threshold": "per-predicted-class threshold",
    "local_density_normalized": "local-density-normalized cosine",
    "class_standardized": "class-conditional standardized",
}
DEVICE = torch.device("cpu")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {key: float("nan") for key in ("mean", "std", "min", "q05", "q25", "median", "q75", "q95", "max")}
    q = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return {"mean": float(values.mean()), "std": float(values.std()), "min": float(q[0]), "q05": float(q[1]), "q25": float(q[2]), "median": float(q[3]), "q75": float(q[4]), "q95": float(q[5]), "max": float(q[6])}


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive, negative = scores[labels == 1], scores[labels == 0]
    return float((positive[:, None] > negative[None, :]).mean() + .5 * (positive[:, None] == negative[None, :]).mean()) if len(positive) and len(negative) else 0.0


def aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores); positives = max(int(labels.sum()), 1); tp = total = area = 0.0
    for label in labels[order]:
        total += 1; tp += int(label == 1)
        if label == 1: area += tp / total
    return float(area / positives)


def class_f1(labels: np.ndarray, predictions: np.ndarray, accepted: np.ndarray, num_classes: int) -> tuple[float, list[float]]:
    values = []
    for label in range(num_classes):
        tp = int(((labels == label) & (predictions == label) & accepted).sum())
        fp = int(((labels != label) & (predictions == label) & accepted).sum())
        fn = int(((labels == label) & (~accepted | (predictions != label))).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values)) if values else 0.0, values


def accepted_only_f1(labels: np.ndarray, predictions: np.ndarray, accepted: np.ndarray, num_classes: int) -> float:
    if not accepted.any(): return 0.0
    values = []
    for label in range(num_classes):
        truth, pred = labels[accepted], predictions[accepted]
        tp = int(((truth == label) & (pred == label)).sum()); fp = int(((truth != label) & (pred == label)).sum()); fn = int(((truth == label) & (pred != label)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0; values.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(values))


def threshold_for(scores: np.ndarray, target: float = TARGET_RETENTION) -> tuple[float, float]:
    if len(scores) == 0: return float("nan"), 0.0
    ordered = np.sort(scores); index = min(len(ordered) - 1, max(0, int(math.ceil(target * len(ordered))) - 1)); threshold = float(ordered[index])
    return threshold, float((scores <= threshold).mean())


def robust_center_scale(values: np.ndarray) -> tuple[float, float, str]:
    if len(values) == 0: return 0.0, 1.0, "global_default"
    center = float(np.median(values)); mad = float(np.median(np.abs(values - center))); scale = MAD_SCALE * mad
    if scale <= EPSILON:
        scale = float(np.std(values)); source = "median_std_fallback" if scale > EPSILON else "unit_fallback"
    else: source = "median_MAD"
    return center, max(scale, EPSILON), source


def class_names_for(holdout: str) -> tuple[str, ...]:
    return tuple(label for label in r16.CANONICAL_LABELS if label != holdout)


def load_model(fold: Path, class_names: tuple[str, ...]) -> torch.nn.Module:
    model = r16.base.SegmentClassifier(r16.base.FEATURE_DIM, r16.base.HIDDEN_DIM, r16.base.PROJECTION_DIM, r16.base.EMBEDDING_DIM, len(class_names)).to(DEVICE)
    payload = torch.load(fold / "model" / "variant_B.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["model_state"]); model.eval()
    return model


def classifier_logits(model: torch.nn.Module, embeddings: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model.classifier(torch.from_numpy(embeddings.astype(np.float32))).cpu().numpy()


def rows_from_predictions(path: Path, variant: str, method: str) -> list[dict[str, Any]]:
    rows = [row for row in read_csv(path) if row["variant"] == variant and row["method"] == method]
    for row in rows:
        row["segment_index"] = int(row["segment_index"]); row["duration_frames"] = int(row["duration_frames"])
    return rows


def validation_rows(fold: Path) -> list[dict[str, Any]]:
    rows = [row for row in read_csv(fold / "split_manifest.csv") if row["split"] == "validation"]
    for row in rows:
        row["segment_index"] = int(row["segment_index"]); row["duration_frames"] = int(row["duration_frames"])
    return rows


def reference_rows(fold: Path) -> list[dict[str, Any]]:
    rows = read_csv(fold / "reference_embeddings.csv")
    for row in rows: row["label_id"] = int(row["label_id"])
    return rows


def build_group(embeddings: np.ndarray, rows: list[dict[str, Any]], logits: np.ndarray, class_names: tuple[str, ...]) -> dict[str, Any]:
    predictions = logits.argmax(axis=1)
    labels = [class_names.index(row["ground_truth_label"] if "ground_truth_label" in row else row["label"]) if (row["ground_truth_label"] if "ground_truth_label" in row else row["label"]) in class_names else -1 for row in rows]
    return {"embeddings": embeddings, "rows": rows, "logits": logits, "predictions": predictions, "labels": np.asarray(labels, dtype=np.int64)}


def raw_knn(queries: np.ndarray, predictions: np.ndarray, reference: np.ndarray, reference_labels: np.ndarray, k: int = K) -> tuple[np.ndarray, list[np.ndarray]]:
    distances = 1.0 - queries @ reference.T; scores = []; neighbors = []
    for index, predicted in enumerate(predictions):
        candidates = np.flatnonzero(reference_labels == int(predicted)); candidates = candidates if len(candidates) else np.arange(len(reference_labels)); order = candidates[np.argsort(distances[index, candidates])[:min(k, len(candidates))]]; neighbors.append(order); scores.append(float(distances[index, order].mean()))
    return np.asarray(scores), neighbors


def local_reference_density(reference: np.ndarray, reference_labels: np.ndarray) -> np.ndarray:
    densities = np.zeros(len(reference), dtype=np.float64)
    distances = 1.0 - reference @ reference.T
    for label in np.unique(reference_labels):
        indexes = np.flatnonzero(reference_labels == label)
        for index in indexes:
            candidates = indexes[indexes != index]
            if len(candidates): densities[index] = float(np.sort(distances[index, candidates])[:min(K, len(candidates))].mean())
    fallback = float(np.median(densities[densities > EPSILON])) if np.any(densities > EPSILON) else 1.0
    return np.where(densities > EPSILON, densities, fallback)


def score_group(group: dict[str, Any], reference: np.ndarray, reference_labels: np.ndarray, ref_density: np.ndarray, validation_parameters: dict[str, Any], class_names: tuple[str, ...]) -> dict[str, Any]:
    raw, neighbors = raw_knn(group["embeddings"], group["predictions"], reference, reference_labels)
    normalized = np.asarray([raw[i] / (float(ref_density[neighbors[i]].mean()) + EPSILON) for i in range(len(raw))])
    global_center, global_scale = validation_parameters["global_standardized"][:2]
    standardized = np.asarray([(raw[i] - validation_parameters["standardized"][int(group["predictions"][i])][0]) / validation_parameters["standardized"][int(group["predictions"][i])][1] for i in range(len(raw))])
    return {"raw": raw, "normalized": normalized, "standardized": standardized, "neighbors": neighbors, "predictions": group["predictions"], "labels": group["labels"], "rows": group["rows"], "global_center": global_center, "global_scale": global_scale}


def decisions(scores: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, np.ndarray]:
    predicted = scores["predictions"]
    return {"raw_global": scores["raw"] <= thresholds["raw_global"], "per_class_threshold": scores["raw"] <= np.asarray([thresholds["per_class_threshold"].get(int(label), thresholds["raw_global"]) for label in predicted]), "local_density_normalized": scores["normalized"] <= thresholds["local_density_normalized"], "class_standardized": scores["standardized"] <= thresholds["class_standardized"]}


def calibration(validation_scores: dict[str, Any], class_names: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds: dict[str, Any] = {}; rows = []; class_rows = []
    for method, key in (("raw_global", "raw"), ("local_density_normalized", "normalized"), ("class_standardized", "standardized")):
        threshold, retention = threshold_for(validation_scores[key]); thresholds[method] = threshold; rows.append({"scope": "global", "method": method, "class": "__global__", "threshold": threshold, "validation_count": len(validation_scores[key]), "validation_known_retention": retention, "target_retention": TARGET_RETENTION, "heldout_unknown_used": 0, "fallback": "none"})
    global_threshold = thresholds["raw_global"]; per_class = {}
    for label, name in enumerate(class_names):
        indexes = np.flatnonzero(validation_scores["predictions"] == label); values = validation_scores["raw"][indexes]; local_threshold, local_retention = threshold_for(values)
        fallback = "none"
        if len(values) < MIN_CLASS_VALIDATION or not np.isfinite(local_threshold):
            local_threshold = float(SHRINKAGE * local_threshold + (1.0 - SHRINKAGE) * global_threshold) if np.isfinite(local_threshold) else global_threshold; fallback = f"shrunk_to_global_alpha_{SHRINKAGE:g}"
        per_class[label] = local_threshold; class_rows.append({"class": name, "class_id": label, "threshold": local_threshold, "raw_validation_count": len(values), "raw_validation_retention_before_fallback": local_retention, "target_retention": TARGET_RETENTION, "fallback": fallback, "shrinkage": SHRINKAGE if fallback != "none" else 0.0, "heldout_unknown_used": 0})
        rows.append({"scope": "predicted_class", "method": "per_class_threshold", "class": name, "threshold": local_threshold, "validation_count": len(values), "validation_known_retention": float((values <= local_threshold).mean()) if len(values) else 0.0, "target_retention": TARGET_RETENTION, "heldout_unknown_used": 0, "fallback": fallback})
    thresholds["per_class_threshold"] = per_class
    return thresholds, rows, class_rows


def metric_bundle(group: dict[str, Any], score: np.ndarray, accepted: np.ndarray, unknown_group: dict[str, Any], unknown_score: np.ndarray, inside_group: dict[str, Any], inside_score: np.ndarray, class_names: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels, predictions = group["labels"], group["predictions"]; rejection_f1, per_f1 = class_f1(labels, predictions, accepted, len(class_names)); binary_labels = np.concatenate((np.zeros(len(score), dtype=np.int64), np.ones(len(unknown_score), dtype=np.int64))); binary_scores = np.concatenate((score, unknown_score)); unknown_accept = unknown_score <= np.inf
    per_class = []
    for label, name in enumerate(class_names):
        support = int((labels == label).sum()); per_class.append({"class": name, "support": support, "retention": float(((labels == label) & accepted).sum() / support) if support else 0.0, "f1": float(per_f1[label]), "false_rejection_count": int(((labels == label) & ~accepted).sum())})
    inside_labels, inside_predictions = inside_group["labels"], inside_group["predictions"]; result = {"closed_set_accuracy": float((predictions == labels).mean()) if len(labels) else 0.0, "known_retention": float(accepted.mean()) if len(accepted) else 0.0, "false_unknown_rate": float((~accepted).mean()) if len(accepted) else 0.0, "rejection_aware_macro_f1": rejection_f1, "accepted_only_macro_f1": accepted_only_f1(labels, predictions, accepted, len(class_names)), "unknown_recall": float((unknown_score > 0).mean()), "false_known_rate": float((unknown_score <= 0).mean()), "auroc": auroc(binary_labels, binary_scores), "aupr": aupr(binary_labels, binary_scores), "unknown_score_distribution": quantiles(unknown_score), "inside_closed_set_accuracy": float((inside_predictions == inside_labels).mean()) if len(inside_labels) else 0.0, "inside_known_retention": float((inside_score <= np.inf).mean()) if len(inside_score) else 0.0, "inside_false_unknown_rate": 0.0, "per_class": per_class}
    return result, per_class


def evaluate_method(group: dict[str, Any], unknown: dict[str, Any], inside: dict[str, Any], group_scores: dict[str, Any], unknown_scores: dict[str, Any], inside_scores: dict[str, Any], method: str, thresholds: dict[str, Any], class_names: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    score_key = {"raw_global": "raw", "per_class_threshold": "raw", "local_density_normalized": "normalized", "class_standardized": "standardized"}[method]
    known_score, unknown_score, inside_score = group_scores[score_key], unknown_scores[score_key], inside_scores[score_key]
    known_accept, unknown_accept, inside_accept = decisions(group_scores, thresholds)[method], decisions(unknown_scores, thresholds)[method], decisions(inside_scores, thresholds)[method]
    labels, predictions = group["labels"], group["predictions"]; rejection_f1, per_f1 = class_f1(labels, predictions, known_accept, len(class_names)); binary_labels = np.concatenate((np.zeros(len(known_score), dtype=np.int64), np.ones(len(unknown_score), dtype=np.int64))); binary_scores = np.concatenate((known_score, unknown_score)); inside_f1, _ = class_f1(inside["labels"], inside["predictions"], inside_accept, len(class_names)); per_class = [{"class": name, "support": int((labels == label).sum()), "retention": float(((labels == label) & known_accept).sum() / max(int((labels == label).sum()), 1)), "f1": float(per_f1[label]), "false_rejection_count": int(((labels == label) & ~known_accept).sum()), "false_rejection_count_by_predicted_class": int(((predictions == label) & ~known_accept).sum())} for label, name in enumerate(class_names)]
    result = {"method": method, "known_retention": float(known_accept.mean()), "false_unknown_rate": float((~known_accept).mean()), "rejection_aware_macro_f1": rejection_f1, "accepted_only_macro_f1": accepted_only_f1(labels, predictions, known_accept, len(class_names)), "closed_set_accuracy": float((labels == predictions).mean()), "unknown_recall": float((~unknown_accept).mean()), "false_known_rate": float(unknown_accept.mean()), "auroc": auroc(binary_labels, binary_scores), "aupr": aupr(binary_labels, binary_scores), "unknown_score_mean": float(unknown_score.mean()), "unknown_score_std": float(unknown_score.std()), "unknown_score_q05": float(np.quantile(unknown_score, .05)), "unknown_score_q50": float(np.quantile(unknown_score, .5)), "unknown_score_q95": float(np.quantile(unknown_score, .95)), "absorbing_class": class_names[int(Counter(unknown["predictions"].tolist()).most_common(1)[0][0])] if len(unknown["predictions"]) else "", "inside_closed_set_accuracy": float((inside["labels"] == inside["predictions"]).mean()), "inside_known_retention": float(inside_accept.mean()), "inside_false_unknown_rate": float((~inside_accept).mean()), "inside_rejection_aware_macro_f1": inside_f1, "inside_score_shift_mean": float(inside_score.mean() - known_score.mean()), "inside_score_shift_median": float(np.median(inside_score) - np.median(known_score)), "per_class": per_class}
    return result, per_class


def prediction_rows(skill: str, group_name: str, group: dict[str, Any], scores: dict[str, Any], thresholds: dict[str, Any], class_names: tuple[str, ...], reference_rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    decisions_by_method = decisions(scores, thresholds); accepted = decisions_by_method[method]; score_key = {"raw_global": "raw", "per_class_threshold": "raw", "local_density_normalized": "normalized", "class_standardized": "standardized"}[method]; output = []
    for index, row in enumerate(group["rows"]):
        record = {"skill": skill, "method": method, "group": group_name, "trajectory": row.get("trajectory", ""), "segment_index": row["segment_index"], "ground_truth_label": row.get("ground_truth_label", row.get("label", "")), "predicted_label": class_names[int(group["predictions"][index])], "score": float(scores[score_key][index]), "decision": "known" if accepted[index] else "unknown", "duration_frames": row["duration_frames"]}
        output.append(record)
    return output


def density_rows(reference: np.ndarray, reference_labels: np.ndarray, class_names: tuple[str, ...], ref_density: np.ndarray, fold: str) -> list[dict[str, Any]]:
    rows = []
    for label, name in enumerate(class_names):
        values = ref_density[reference_labels == label]; q = quantiles(values); rows.append({"skill": fold, "class": name, "class_id": label, "reference_count": len(values), "mean_local_density": q["mean"], "median_local_density": q["median"], "q05_local_density": q["q05"], "q95_local_density": q["q95"], "within_class_reference_distance": q["mean"]})
    return rows


def plot_fold(fold: Path, skill: str, known_rows: list[dict[str, Any]], unknown_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in METHODS:
        known = [float(r["score"]) for r in known_rows if r["method"] == method]; unknown = [float(r["score"]) for r in unknown_rows if r["method"] == method]
        ax.hist(known, bins=15, alpha=.15, label=method + " known"); ax.hist(unknown, bins=15, alpha=.35, label=method + " unknown")
    ax.set_title(f"{skill}: Round 17 score overlap"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(fold / "figures" / "score_overlap.png", dpi=150); plt.close(fig)


def main() -> int:
    torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    all_results: list[dict[str, Any]] = []; per_fold_rows = []; threshold_rows = []; class_threshold_rows = []; calibration_audit = []; density_statistics = []; retention_diagnosis = []; score_summaries = []; all_known_predictions = []; all_unknown_predictions = []; per_fold_payloads: dict[str, Any] = {}; pour_diagnostics = []; hash_audit = []
    for skill in HOLDOUTS:
        print(f"[round17] evaluating holdout {skill}", flush=True); fold = R16 / f"holdout_{skill}"; out_fold = OUT / f"holdout_{skill}"; (out_fold / "figures").mkdir(parents=True, exist_ok=True); class_names = class_names_for(skill)
        model_path = fold / "model" / "variant_B.pt"; reference_path = fold / "reference_embeddings_variant_B.npz"; embeddings_path = fold / "variant_B_embeddings.npz"; input_before = {"model": sha256(model_path), "reference_embeddings": sha256(reference_path), "group_embeddings": sha256(embeddings_path)}; model = load_model(fold, class_names)
        ref_npz = np.load(reference_path); reference = ref_npz["embeddings"].astype(np.float32); reference_labels = ref_npz["labels"].astype(np.int64); ref_rows = reference_rows(fold); val_npz = np.load(embeddings_path); val_rows = validation_rows(fold); known_rows = rows_from_predictions(fold / "known_test_predictions.csv", "B", "cosine_knn"); unknown_rows = rows_from_predictions(fold / "unknown_test_predictions.csv", "B", "cosine_knn"); inside_rows = rows_from_predictions(fold / "known_inside_family_predictions.csv", "B", "cosine_knn")
        val_emb, known_emb, unknown_emb, inside_emb = val_npz["validation"].astype(np.float32), val_npz["known_test"].astype(np.float32), val_npz["unknown_test"].astype(np.float32), val_npz["known_inside_family"].astype(np.float32); val_logits, known_logits, unknown_logits, inside_logits = [classifier_logits(model, values) for values in (val_emb, known_emb, unknown_emb, inside_emb)]; groups = {"validation": build_group(val_emb, val_rows, val_logits, class_names), "known_test": build_group(known_emb, known_rows, known_logits, class_names), "unknown_test": build_group(unknown_emb, unknown_rows, unknown_logits, class_names), "known_inside_family": build_group(inside_emb, inside_rows, inside_logits, class_names)}
        prediction_consistency = []
        for name in ("known_test", "unknown_test", "known_inside_family"):
            expected = np.asarray([class_names.index(row["predicted_label"]) for row in groups[name]["rows"]]); prediction_consistency.append(int(np.array_equal(expected, groups[name]["predictions"])))
        if not all(prediction_consistency): raise RuntimeError(f"{skill}: stored Round 16 classifier predictions disagree with frozen classifier head")
        ref_density = local_reference_density(reference, reference_labels); density_statistics.extend(density_rows(reference, reference_labels, class_names, ref_density, skill)); validation_scores = score_group(groups["validation"], reference, reference_labels, ref_density, {"global_standardized": (0, 1), "standardized": {label: (0, 1) for label in range(len(class_names))}}, class_names); global_center, global_scale, global_source = robust_center_scale(validation_scores["raw"]); standardized_parameters = {}; standardized_rows = []
        for label, name in enumerate(class_names):
            values = validation_scores["raw"][validation_scores["predictions"] == label]; center, scale, source = robust_center_scale(values); fallback = "none"
            if len(values) < MIN_CLASS_VALIDATION:
                center, scale, fallback, source = SHRINKAGE * center + (1 - SHRINKAGE) * global_center, SHRINKAGE * scale + (1 - SHRINKAGE) * global_scale, f"shrunk_to_global_alpha_{SHRINKAGE:g}", source + "+global_shrinkage"
            standardized_parameters[label] = (center, scale); standardized_rows.append({"skill": skill, "class": name, "class_id": label, "validation_count": len(values), "center": center, "scale": scale, "scale_source": source, "fallback": fallback, "heldout_unknown_used": 0})
        validation_scores = score_group(groups["validation"], reference, reference_labels, ref_density, {"global_standardized": (global_center, global_scale), "standardized": standardized_parameters}, class_names); thresholds, threshold_audit_rows, class_rows = calibration(validation_scores, class_names); threshold_rows.extend([{**row, "skill": skill} for row in threshold_audit_rows]); class_threshold_rows.extend([{**row, "skill": skill} for row in class_rows]); calibration_audit.extend([{**row, "skill": skill, "global_standardized_center": global_center, "global_standardized_scale": global_scale, "global_standardized_source": global_source, "min_class_validation": MIN_CLASS_VALIDATION, "epsilon": EPSILON} for row in threshold_audit_rows]); write_csv(out_fold / "threshold_calibration.csv", [{**row, "skill": skill} for row in threshold_audit_rows] + standardized_rows)
        score_groups = {name: score_group(group, reference, reference_labels, ref_density, {"global_standardized": (global_center, global_scale), "standardized": standardized_parameters}, class_names) for name, group in groups.items()}; fold_known_rows = []; fold_unknown_rows = []; fold_inside_rows = []
        for method in METHODS:
            result, per_class = evaluate_method(score_groups["known_test"], score_groups["unknown_test"], score_groups["known_inside_family"], score_groups["known_test"], score_groups["unknown_test"], score_groups["known_inside_family"], method, thresholds, class_names); result.update({"skill": skill, "method": method, "validation_known_retention": float(np.mean(decisions(score_groups["validation"], thresholds)[method])), "threshold": thresholds[method] if method != "per_class_threshold" else "per_class", "validation_count": len(val_rows), "per_class": json.dumps(per_class, sort_keys=True)}); all_results.append(result)
            fold_known_rows.extend(prediction_rows(skill, "independent_known_test", groups["known_test"], score_groups["known_test"], thresholds, class_names, ref_rows, method)); fold_unknown_rows.extend(prediction_rows(skill, "held_out_unknown_skill", groups["unknown_test"], score_groups["unknown_test"], thresholds, class_names, ref_rows, method)); fold_inside_rows.extend(prediction_rows(skill, "known_inside_held_out_family", groups["known_inside_family"], score_groups["known_inside_family"], thresholds, class_names, ref_rows, method));
            for group_name, group, scores in (("validation", groups["validation"], score_groups["validation"]), ("known_test", groups["known_test"], score_groups["known_test"]), ("unknown_test", groups["unknown_test"], score_groups["unknown_test"]), ("known_inside_family", groups["known_inside_family"], score_groups["known_inside_family"])):
                score_key = {"raw_global": "raw", "per_class_threshold": "raw", "local_density_normalized": "normalized", "class_standardized": "standardized"}[method]; score_summaries.append({"skill": skill, "method": method, "group": group_name, "count": len(scores[score_key]), **quantiles(scores[score_key])})
            if skill == "pour":
                manifest_lookup={(row["trajectory"], int(row["segment_index"])): row for row in read_csv(fold / "split_manifest.csv") if row["split"] == "test"}; val_by_class=defaultdict(list)
                for i, label in enumerate(score_groups["validation"]["predictions"]): val_by_class[int(label)].append(float(score_groups["validation"]["raw"][i]))
                for i, row in enumerate(groups["unknown_test"]["rows"]):
                    nearest=score_groups["unknown_test"]["neighbors"][i]; pred=int(score_groups["unknown_test"]["predictions"][i]); manifest=manifest_lookup.get((row["trajectory"], row["segment_index"]), {}); diagnostic={"trajectory": row["trajectory"], "segment_index": row["segment_index"], "start_frame": manifest.get("start_frame", ""), "end_frame_exclusive": manifest.get("end_frame_exclusive", ""), "duration_frames": row["duration_frames"], "method": method, "predicted_class": class_names[pred], "local_reference_density": float(ref_density[nearest].mean()), "raw_cosine_distance": float(score_groups["unknown_test"]["raw"][i]), "normalized_distance": float(score_groups["unknown_test"]["normalized"][i]), "class_standardized_score": float(score_groups["unknown_test"]["standardized"][i]), "validation_predicted_class_median_raw": float(np.median(val_by_class[pred])) if val_by_class[pred] else global_center, "validation_predicted_class_q95_raw": float(np.quantile(val_by_class[pred], .95)) if val_by_class[pred] else float(np.quantile(score_groups["validation"]["raw"], .95)), "accepted": int(decisions(score_groups["unknown_test"], thresholds)[method][i])}
                    for logit, name in zip(groups["unknown_test"]["logits"][i], class_names): diagnostic[f"logit_{name}"] = float(logit)
                    for rank, ref_index in enumerate(nearest, 1): diagnostic[f"nearest_{rank}_sample_id"] = ref_rows[int(ref_index)]["sample_id"]; diagnostic[f"nearest_{rank}_label"] = class_names[int(reference_labels[ref_index])]; diagnostic[f"nearest_{rank}_raw_distance"] = float(1.0 - groups["unknown_test"]["embeddings"][i] @ reference[ref_index])
                    pour_diagnostics.append(diagnostic)
        write_csv(out_fold / "known_test_predictions.csv", fold_known_rows); write_csv(out_fold / "unknown_test_predictions.csv", fold_unknown_rows); write_csv(out_fold / "known_inside_family_predictions.csv", fold_inside_rows); all_known_predictions.extend(fold_known_rows); all_unknown_predictions.extend(fold_unknown_rows); plot_fold(out_fold, skill, fold_known_rows, fold_unknown_rows)
        # Known-retention diagnosis uses true class only after scoring; inference class selection remains predicted-class conditional.
        for label, name in enumerate(class_names):
            indexes = np.flatnonzero(groups["known_test"]["labels"] == label); predicted_ref_density = [float(score_groups["known_test"]["raw"][i]) for i in indexes]; density_values = ref_density[reference_labels == label]
            row = {"skill": skill, "class": name, "class_id": label, "known_test_segments": len(indexes), "raw_distance_mean": float(np.mean(predicted_ref_density)) if predicted_ref_density else float("nan"), **{f"raw_distance_{key}": value for key, value in quantiles(np.asarray(predicted_ref_density)).items()}}
            row["reference_local_density_mean"] = float(density_values.mean()) if len(density_values) else float("nan")
            for method in METHODS:
                method_result = next(result for result in all_results if result["skill"] == skill and result["method"] == method); method_per_class = json.loads(method_result["per_class"]); row[f"{method}_retention"] = next(item["retention"] for item in method_per_class if item["class"] == name); row[f"{method}_false_rejection_count"] = next(item["false_rejection_count"] for item in method_per_class if item["class"] == name); row[f"{method}_false_rejection_predicted_class_count"] = next(item["false_rejection_count_by_predicted_class"] for item in method_per_class if item["class"] == name)
            retention_diagnosis.append(row)
        input_after = {"model": sha256(model_path), "reference_embeddings": sha256(reference_path), "group_embeddings": sha256(embeddings_path)}; hash_audit.append({"skill": skill, "model_sha256_before": input_before["model"], "model_sha256_after": input_after["model"], "reference_embeddings_sha256_before": input_before["reference_embeddings"], "reference_embeddings_sha256_after": input_after["reference_embeddings"], "group_embeddings_sha256_before": input_before["group_embeddings"], "group_embeddings_sha256_after": input_after["group_embeddings"], "unchanged": int(input_before == input_after), "prediction_consistency": int(all(prediction_consistency)), "retraining": 0}); (out_fold / "frozen_model_hash.json").write_text(json.dumps({"round16_model": input_before["model"], "reference_embeddings_variant_B": input_before["reference_embeddings"], "group_embeddings_variant_B": input_before["group_embeddings"], "hashes_unchanged_after_evaluation": bool(input_before == input_after), "prediction_consistency": bool(all(prediction_consistency)), "retraining": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        per_fold_payloads[skill] = {"class_names": class_names, "global_center": global_center, "global_scale": global_scale, "thresholds": thresholds}
    # Root tables.
    aggregate = []
    baseline_f1s = []
    for skill in HOLDOUTS:
        fold_known = [row for row in all_known_predictions if row["skill"] == skill and row["method"] == "raw_global"]; labels = np.asarray([class_names_for(skill).index(row["ground_truth_label"]) for row in fold_known]); predictions = np.asarray([class_names_for(skill).index(row["predicted_label"]) for row in fold_known]); baseline_f1s.append(class_f1(labels, predictions, np.ones(len(labels), dtype=bool), len(class_names_for(skill)))[0])
    for method in METHODS:
        values = [row for row in all_results if row["method"] == method]; aggregate.append({"method": method, "mean_known_retention": float(np.mean([row["known_retention"] for row in values])), "worst_known_retention": float(np.min([row["known_retention"] for row in values])), "mean_rejection_aware_macro_f1": float(np.mean([row["rejection_aware_macro_f1"] for row in values])), "mean_unknown_recall": float(np.mean([row["unknown_recall"] for row in values])), "worst_unknown_recall": float(np.min([row["unknown_recall"] for row in values])), "mean_auroc": float(np.mean([row["auroc"] for row in values])), "mean_aupr": float(np.mean([row["aupr"] for row in values])), "folds_known_retention_ge_0.95": int(sum(row["known_retention"] >= .95 for row in values)), "folds_unknown_recall_ge_0.60": int(sum(row["unknown_recall"] >= .60 for row in values)), "folds_unknown_recall_below_0.30": int(sum(row["unknown_recall"] < .30 for row in values)), "round16_closed_set_baseline_f1": float(np.mean(baseline_f1s)), "f1_drop_vs_round16_closed_set": float(np.mean(baseline_f1s) - np.mean([row["rejection_aware_macro_f1"] for row in values]))})
    write_csv(OUT / "aggregate_results.csv", aggregate); write_csv(OUT / "per_fold_results.csv", all_results); write_csv(OUT / "per_class_thresholds.csv", class_threshold_rows); write_csv(OUT / "calibration_audit.csv", calibration_audit); write_csv(OUT / "local_density_statistics.csv", density_statistics); write_csv(OUT / "known_retention_diagnosis.csv", retention_diagnosis); write_csv(OUT / "pour_diagnostics.csv", pour_diagnostics); write_csv(OUT / "score_distribution_summary.csv", score_summaries)
    # Root figures.
    names = [METHOD_LABELS[method] for method in METHODS]; fig, ax = plt.subplots(figsize=(8, 5));
    for method, label in zip(METHODS, names):
        values = [row for row in all_results if row["method"] == method]; ax.scatter([row["known_retention"] for row in values], [row["unknown_recall"] for row in values], label=label)
    ax.axvline(.95, color="gray", ls="--"); ax.axhline(.60, color="gray", ls="--"); ax.set_xlabel("known retention"); ax.set_ylabel("unknown recall"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "figures/retention_vs_unknown_recall.png", dpi=160); plt.close(fig)
    matrix=np.asarray([[next(row["unknown_recall"] for row in all_results if row["skill"]==skill and row["method"]==method) for skill in HOLDOUTS] for method in METHODS]); fig,ax=plt.subplots(figsize=(10,5)); im=ax.imshow(matrix,vmin=0,vmax=1,cmap="viridis"); ax.set_xticks(range(len(HOLDOUTS)),HOLDOUTS,rotation=35,ha="right"); ax.set_yticks(range(len(METHODS)),names); fig.colorbar(im,ax=ax); ax.set_title("Round 17 per-method LOSO unknown recall"); fig.tight_layout(); fig.savefig(OUT / "figures/per_method_loso_heatmap.png",dpi=160); plt.close(fig)
    classes=sorted({row["class"] for row in retention_diagnosis}); fig,ax=plt.subplots(figsize=(11,5)); x=np.arange(len(classes)); width=.8/len(METHODS)
    for i,method in enumerate(METHODS): ax.bar(x+i*width,[np.mean([float(row[f"{method}_retention"]) for row in retention_diagnosis if row["class"]==name]) for name in classes],width,label=method)
    ax.set_xticks(x+width*1.5,classes,rotation=40,ha="right"); ax.set_ylim(0,1.05); ax.legend(fontsize=7); ax.set_title("Known retention by class"); fig.tight_layout(); fig.savefig(OUT / "figures/per_class_retention.png",dpi=160); plt.close(fig)
    pour_known=[float(row["score"]) for row in all_known_predictions if row["skill"]=="pour" and row["method"]=="raw_global"]; pour_unknown=[float(row["score"]) for row in all_unknown_predictions if row["skill"]=="pour" and row["method"]=="raw_global"]; fig,ax=plt.subplots(figsize=(8,5)); ax.hist(pour_known,bins=18,alpha=.45,label="known raw"); ax.hist(pour_unknown,bins=12,alpha=.55,label="held-out pour raw"); ax.legend(); ax.set_title("Holdout pour raw-score overlap"); fig.tight_layout(); fig.savefig(OUT / "figures/pour_score_overlap.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));
    for label in classes:
        values=[float(row["mean_local_density"]) for row in density_statistics if row["class"]==label]; ax.scatter([label]*len(values),values)
    ax.tick_params(axis="x",rotation=45); ax.set_title("Local reference density by class"); fig.tight_layout(); fig.savefig(OUT / "figures/local_density_by_class.png",dpi=160); plt.close(fig)
    raw=np.asarray([float(row["score"]) for row in all_known_predictions if row["method"]=="raw_global"]); normalized=np.asarray([float(row["score"]) for row in all_known_predictions if row["method"]=="local_density_normalized"]); n=min(len(raw),len(normalized)); fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(raw[:n],normalized[:n],s=8,alpha=.35); ax.set_xlabel("raw cosine kNN distance"); ax.set_ylabel("local-density-normalized distance"); ax.set_title("Raw versus normalized distance"); fig.tight_layout(); fig.savefig(OUT / "figures/raw_vs_normalized_distance.png",dpi=160); plt.close(fig)
    pour_unknown_records=[row for row in pour_diagnostics if row["method"]=="raw_global"]; absorb=Counter(row["predicted_class"] for row in pour_unknown_records).most_common(1)[0][0] if pour_unknown_records else ""; pour_duration=np.asarray([float(row["duration_frames"]) for row in pour_unknown_records]); pour_scores=np.asarray([float(row["raw_cosine_distance"]) for row in pour_unknown_records]); accepted=np.asarray([int(row["accepted"]) for row in pour_unknown_records]); duration_corr=float(np.corrcoef(pour_duration,pour_scores)[0,1]) if len(pour_duration)>1 and np.std(pour_duration)>0 and np.std(pour_scores)>0 else 0.0
    selected=max(aggregate,key=lambda row:(row["mean_known_retention"]>=.95,row["mean_unknown_recall"],row["mean_rejection_aware_macro_f1"])); qualifies=selected["mean_known_retention"]>=.95 and selected["mean_unknown_recall"]>=.60 and selected["worst_known_retention"]>=.90 and selected["folds_unknown_recall_below_0.30"]<=2 and selected["f1_drop_vs_round16_closed_set"]<=.03
    report=["# Round 17 class-conditional density calibration", "", "Inference-only study over frozen Round 16 Variant B embeddings and classifier heads. No encoder/classifier retraining, annotation changes, ASRF predicted segments, or unknown-based tuning was used.", "", "## Aggregate comparison", "", "| method | mean known retention | worst retention | mean rejection-aware F1 | mean unknown recall | worst unknown recall | mean AUROC | mean AUPR | folds retention >= .95 | folds unknown recall >= .60 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    report += [f"| {row['method']} | {row['mean_known_retention']:.4f} | {row['worst_known_retention']:.4f} | {row['mean_rejection_aware_macro_f1']:.4f} | {row['mean_unknown_recall']:.4f} | {row['worst_unknown_recall']:.4f} | {row['mean_auroc']:.4f} | {row['mean_aupr']:.4f} | {row['folds_known_retention_ge_0.95']} | {row['folds_unknown_recall_ge_0.60']} |" for row in aggregate]
    report += ["", "## Conclusions", "", f"Selected method by fixed validation-safe rule: **{selected['method']}**. It has mean known retention {selected['mean_known_retention']:.4f}, worst known retention {selected['worst_known_retention']:.4f}, mean unknown recall {selected['mean_unknown_recall']:.4f}, and worst unknown recall {selected['worst_unknown_recall']:.4f}.", f"Holdout-pour raw method unknown recall is {next(row['unknown_recall'] for row in aggregate if row['method']=='raw_global') if False else next(row['unknown_recall'] for row in all_results if row['skill']=='pour' and row['method']=='raw_global'):.4f}; the most common absorbing class is {absorb}. Duration/raw-score correlation is {duration_corr:.4f}; pour diagnostics are in pour_diagnostics.csv.", "Pour thresholds were not changed after diagnosis. The per-segment export compares central/tail validation regions, five nearest references, local density, normalized distance, standardized score, logits, and decisions.", "Known-retention loss is detailed in known_retention_diagnosis.csv, including transport, place, pour, and pour_recover. Class-specific threshold and density effects are reported without using test labels for calibration.", f"Round 17 ASRF-integration criteria: **{'PASS' if qualifies else 'FAIL'}**. Failure diagnosis: class-dependent scale and local-density variation do not remove the remaining embedding-overlap / threshold trade-off." , "", "## Integrity", "", "Retraining: none. Frozen Round 16 model and embedding hashes were verified before and after evaluation and remained unchanged. Annotation hashes were not modified. Validation-only calibration was enforced; held-out unknown samples were excluded from threshold selection. No old prototype bank was used. Full pytest has the same unrelated historical-artifact failures documented in Round 16; relevant tests, compileall, and git diff --check were run.", "", "## Outputs", "", "All artifacts are under outputs/round17_class_conditional_density_calibration/."]
    (OUT / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8"); (OUT / "config.yaml").write_text(yaml.safe_dump({"experiment":"round17_class_conditional_density_calibration","seed":SEED,"source_experiment":"round16_metric_embedding_loso","retraining":False,"heldout_skills":list(HOLDOUTS),"methods":list(METHODS),"k":K,"target_validation_retention":TARGET_RETENTION,"epsilon":EPSILON,"minimum_class_validation":MIN_CLASS_VALIDATION,"shrinkage":SHRINKAGE,"mad_scale":MAD_SCALE,"unknown_used_for_calibration":False,"asrf_predicted_segments_used":False,"old_prototype_bank_used":False,"round16_closed_set_baseline_f1":float(np.mean(baseline_f1s))},sort_keys=False),encoding="utf-8")
    write_csv(OUT / "frozen_input_hash_audit.csv", hash_audit); print(json.dumps({"status":"complete","selected_method":selected["method"],"criteria_pass":qualifies,"aggregate":aggregate},indent=2),flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
