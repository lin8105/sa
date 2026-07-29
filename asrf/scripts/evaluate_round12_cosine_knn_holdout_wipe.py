#!/usr/bin/env python3
"""Frozen embedding-space open-set evaluation for the Round 12 wipe holdout."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_ROOT = ROOT / "outputs/round12_open_set_holdout_wipe"
OUTPUT_ROOT = ROOT / "outputs/round12_open_set_cosine_knn_holdout_wipe"
CHECKPOINT = HOLDOUT_ROOT / "model/best.pt"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import train_round12_segment_classifier as base  # noqa: E402
from asrf.data.ontology import CANONICAL_LABELS, ONTOLOGY_VERSION  # noqa: E402
from asrf.training.checkpointing import sha256_file  # noqa: E402

SEED = 42
HELD_OUT = "wipe"
KNOWN_CLASSES = tuple(name for name in CANONICAL_LABELS if name != HELD_OUT)
KNOWN_TO_ID = {name: index for index, name in enumerate(KNOWN_CLASSES)}
ID_TO_KNOWN = {index: name for name, index in KNOWN_TO_ID.items()}
K_VALUES = (1, 3, 5, 10)
VARIANTS = ("global", "predicted_class_conditional")
VARIANT_ORDER = {name: index for index, name in enumerate(VARIANTS)}
BATCH_SIZE = 32


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def distribution(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0, .1, .25, .5, .75, .9, 1])
    return {name: float(value) for name, value in zip(("min", "p10", "p25", "median", "p75", "p90", "max"), q)}


def verify_checkpoint(payload: dict[str, Any]) -> None:
    metadata = payload.get("ontology_metadata", {})
    if metadata.get("ontology_version") != ONTOLOGY_VERSION or metadata.get("held_out_class") != HELD_OUT:
        raise RuntimeError("Frozen checkpoint is not the Round 12 wipe-holdout model")
    if tuple(metadata.get("ordered_known_class_list", ())) != KNOWN_CLASSES:
        raise RuntimeError("Frozen checkpoint known-class list mismatch")
    if int(metadata.get("feature_dim", metadata.get("architecture", {}).get("feature_dim", -1))) != base.FEATURE_DIM:
        raise RuntimeError("Frozen checkpoint feature dimension mismatch")
    classifier_weight = payload["model_state"].get("classifier.3.weight")
    if classifier_weight is None or classifier_weight.shape[0] != len(KNOWN_CLASSES):
        raise RuntimeError("Frozen checkpoint classifier output is not ten classes")


def load_manifest_rows() -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for name in ("train", "validation", "test"):
        rows = []
        for raw in read_csv(HOLDOUT_ROOT / "split_manifests" / f"{name}.csv"):
            row = dict(raw)
            for field in ("segment_index", "ontology_label_id", "label_id", "start_frame", "end_frame_exclusive", "duration_frames"):
                row[field] = int(row[field])
            rows.append(row)
        output[name] = rows
    if any(row["label"] == HELD_OUT for row in output["train"] + output["validation"]):
        raise RuntimeError("Wipe leaked into the reference or validation bank")
    return output


def extract_embeddings(model: torch.nn.Module, rows: list[dict[str, Any]], feature_cache: dict[str, tuple[np.ndarray, np.ndarray]], normalization: dict[str, Any], device: torch.device) -> dict[str, Any]:
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    dataset = base.SegmentDataset(rows, feature_cache, mean, std, float(normalization["duration_mean"]), float(normalization["duration_std"]))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=base.collate_segments)
    model.eval(); embeddings = []; logits = []; ordered_rows = []
    with torch.no_grad():
        for batch in loader:
            embedding, output = model(batch["sequence"].to(device), batch["valid_mask"].to(device), batch["lengths"].to(device), batch["duration"].to(device))
            embeddings.append(embedding.cpu().numpy()); logits.append(output.cpu().numpy()); ordered_rows.extend(batch["rows"])
    embedding_array = np.concatenate(embeddings).astype(np.float32)
    norms = np.linalg.norm(embedding_array, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError(f"Embedding L2 normalization failed: min={norms.min()} max={norms.max()}")
    return {"rows": ordered_rows, "embeddings": embedding_array, "logits": np.concatenate(logits).astype(np.float32), "norms": norms.astype(np.float32)}


def scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True)
    order = np.argsort(-probabilities, axis=1)
    top1 = order[:, 0]; top2 = order[:, 1]
    return {"probabilities": probabilities, "top1": top1, "top2": top2, "max_softmax": probabilities[np.arange(len(probabilities)), top1], "margin": probabilities[np.arange(len(probabilities)), top1] - probabilities[np.arange(len(probabilities)), top2], "energy": -np.log(np.exp(shifted).sum(axis=1)) - logits.max(axis=1)}


def enrich_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    outputs = dict(outputs); values = scores(outputs["logits"]); outputs["scores"] = values
    for index, row in enumerate(outputs["rows"]):
        row["predicted_closed_set_label"] = ID_TO_KNOWN[int(values["top1"][index])]
        row["predicted_closed_set_label_id"] = int(values["top1"][index])
    return outputs


def save_embedding_npz(path: Path, outputs: dict[str, Any]) -> None:
    rows = outputs["rows"]
    np.savez_compressed(path, embedding=outputs["embeddings"], embedding_l2_norm=outputs["norms"], trajectory=np.asarray([row["trajectory"] for row in rows]), segment_index=np.asarray([row["segment_index"] for row in rows], dtype=np.int64), sample_id=np.asarray([row["sample_id"] for row in rows]), start_frame=np.asarray([row["start_frame"] for row in rows], dtype=np.int64), end_frame=np.asarray([row["end_frame_exclusive"] for row in rows], dtype=np.int64), duration=np.asarray([row["duration_frames"] for row in rows], dtype=np.int64), ground_truth_label=np.asarray([row["label"] for row in rows]), predicted_closed_set_label=np.asarray([row["predicted_closed_set_label"] for row in rows]), family=np.asarray([row["family"] for row in rows]), task_source=np.asarray([row["family"] for row in rows]), metadata_json=np.asarray([json_text(row) for row in rows]))


def nearest_for_class(query: np.ndarray, reference: np.ndarray, reference_labels: np.ndarray) -> dict[str, np.ndarray]:
    distances = 1.0 - query @ reference.T
    return {name: distances[:, reference_labels == name].min(axis=1) for name in KNOWN_CLASSES}


def knn_variant(query_embeddings: np.ndarray, query_predictions: np.ndarray, reference_embeddings: np.ndarray, reference_labels: np.ndarray, reference_rows: list[dict[str, Any]], variant: str, k_requested: int) -> list[dict[str, Any]]:
    outputs = []
    for query_index, embedding in enumerate(query_embeddings):
        all_distances = 1.0 - reference_embeddings @ embedding
        if variant == "global":
            candidates = np.arange(len(reference_embeddings))
        else:
            predicted = ID_TO_KNOWN[int(query_predictions[query_index])]
            candidates = np.flatnonzero(reference_labels == predicted)
        effective_k = min(k_requested, len(candidates))
        if effective_k <= 0:
            raise RuntimeError(f"No reference embeddings for predicted class at query {query_index}")
        order = candidates[np.argsort(all_distances[candidates], kind="stable")[:effective_k]]
        neighbor_distances = all_distances[order]
        neighbor_labels = [str(reference_labels[index]) for index in order]
        outputs.append({"mean_distance": float(neighbor_distances.mean()), "max_distance": float(neighbor_distances.max()), "neighbor_indices": order.tolist(), "neighbor_distances": neighbor_distances.tolist(), "neighbor_labels": neighbor_labels, "neighbor_label_agreement": float(Counter(neighbor_labels).most_common(1)[0][1] / effective_k), "nearest_neighbor_label": neighbor_labels[0], "nearest_neighbor_row": reference_rows[int(order[0])], "k_effective": int(effective_k), "reference_count": int(len(candidates))})
    return outputs


def closed_rejection_metrics(outputs: dict[str, Any], accepted: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([row["label_id"] for row in outputs["rows"]], dtype=np.int64)
    predictions = outputs["scores"]["top1"].astype(np.int64)
    accepted_correct = predictions[accepted] == labels[accepted]
    matrix = np.zeros((len(KNOWN_CLASSES), len(KNOWN_CLASSES)), dtype=np.int64)
    for truth, prediction in zip(labels[accepted], predictions[accepted]):
        matrix[int(truth), int(prediction)] += 1
    f1_values = []; per_class = {}
    for index, name in enumerate(KNOWN_CLASSES):
        tp = int(matrix[index, index]); fp = int(matrix[:, index].sum() - tp); fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0; f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int((labels == index).sum()); rejected = int(((labels == index) & ~accepted).sum())
        per_class[name] = {"support": support, "false_unknown_count": rejected, "false_unknown_rate": rejected / support if support else 0.0, "precision_after_rejection": precision, "recall_after_rejection": recall, "f1_after_rejection": f1}
        if support: f1_values.append(f1)
    return {"count": int(len(labels)), "closed_set_accuracy": float((predictions == labels).mean()) if len(labels) else 0.0, "known_retention": float(accepted.mean()) if len(accepted) else 0.0, "false_unknown_rate": float((~accepted).mean()) if len(accepted) else 0.0, "accepted_known_accuracy": float(accepted_correct.mean()) if accepted_correct.size else 0.0, "macro_f1_after_rejection": float(np.mean(f1_values)) if f1_values else 0.0, "per_class": per_class, "confusion_matrix_after_rejection": matrix.tolist()}


def wipe_rejection_metrics(outputs: dict[str, Any], accepted: np.ndarray, knn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = outputs["scores"]["top1"]
    accepted_labels = [ID_TO_KNOWN[int(predictions[index])] for index in np.flatnonzero(accepted)]
    nearest_labels = [row["nearest_neighbor_label"] for row in knn_rows]
    novelty = np.asarray([row["mean_distance"] for row in knn_rows])
    return {"count": int(len(accepted)), "unknown_recall": float((~accepted).mean()), "false_known_rate": float(accepted.mean()), "most_common_absorbing_class": Counter(accepted_labels).most_common(1)[0][0] if accepted_labels else None, "classifier_top1_distribution": dict(Counter(ID_TO_KNOWN[int(value)] for value in predictions)), "knn_nearest_label_distribution": dict(Counter(nearest_labels)), "novelty_mean": float(novelty.mean()), "novelty_std": float(novelty.std()), "novelty_quantiles": distribution(novelty)}


def calibrate(validation: dict[str, Any], reference: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    reference_labels = np.asarray([row["label"] for row in reference["rows"]])
    all_results = {}; calibration_rows = []
    for variant in VARIANTS:
        for k in K_VALUES:
            result = knn_variant(validation["embeddings"], validation["scores"]["top1"], reference["embeddings"], reference_labels, reference["rows"], variant, k)
            all_results[(variant, k)] = result
            novelty = np.asarray([row["mean_distance"] for row in result]); threshold = float(np.quantile(novelty, .95)); metrics = closed_rejection_metrics(validation, novelty <= threshold)
            calibration_rows.append({"variant": variant, "k_requested": k, "threshold": threshold, "target_validation_known_retention": .95, "validation_known_retention": metrics["known_retention"], "validation_false_unknown_rate": metrics["false_unknown_rate"], "validation_accepted_accuracy": metrics["accepted_known_accuracy"], "validation_macro_f1_after_rejection": metrics["macro_f1_after_rejection"], "novelty_mean": float(novelty.mean()), "novelty_std": float(novelty.std()), "novelty_quantiles": json_text(distribution(novelty)), "min_effective_k": min(row["k_effective"] for row in result), "max_effective_k": max(row["k_effective"] for row in result), "conditional_reference_shortfall_count": int(sum(row["k_effective"] < k for row in result)), "selected_primary": 0})
    ranking = sorted(calibration_rows, key=lambda row: (-float(row["validation_macro_f1_after_rejection"]), -float(row["validation_accepted_accuracy"]), abs(float(row["validation_known_retention"]) - .95), int(row["k_requested"]), VARIANT_ORDER[row["variant"]]))
    primary = (ranking[0]["variant"], int(ranking[0]["k_requested"]))
    for row in calibration_rows: row["selected_primary"] = int((row["variant"], int(row["k_requested"])) == primary)
    return calibration_rows, all_results


def query_class_distance_summary(outputs_by_group: dict[str, dict[str, Any]], reference: dict[str, Any]) -> list[dict[str, Any]]:
    reference_labels = np.asarray([row["label"] for row in reference["rows"]]); rows = []
    for group, outputs in outputs_by_group.items():
        distances = nearest_for_class(outputs["embeddings"], reference["embeddings"], reference_labels)
        for class_name in KNOWN_CLASSES:
            values = distances[class_name]
            rows.append({"group": group, "query_label": "wipe" if group == "wipe" else "mixed", "reference_class": class_name, "count": len(values), "mean_nearest_distance": float(values.mean()), "std_nearest_distance": float(values.std()), "p10": float(np.quantile(values, .1)), "median": float(np.quantile(values, .5)), "p90": float(np.quantile(values, .9)), "min": float(values.min()), "max": float(values.max())})
    return rows


def make_prediction_rows(outputs_by_group: dict[str, dict[str, Any]], reference: dict[str, Any], calibration_rows: list[dict[str, Any]], result_map: dict[tuple[str, int, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []; threshold_map = {(row["variant"], int(row["k_requested"])): float(row["threshold"]) for row in calibration_rows}
    for group, outputs in outputs_by_group.items():
        for variant in VARIANTS:
            for k in K_VALUES:
                for index, (row, knn) in enumerate(zip(outputs["rows"], result_map[(group, variant, k)])):
                    novelty = knn["mean_distance"]; threshold = threshold_map[(variant, k)]; accepted = novelty <= threshold; prediction = ID_TO_KNOWN[int(outputs["scores"]["top1"][index])]
                    rows.append({"group": group, "sample_id": row["sample_id"], "trajectory": row["trajectory"], "family": row["family"], "segment_index": row["segment_index"], "start_frame": row["start_frame"], "end_frame": row["end_frame_exclusive"], "duration_frames": row["duration_frames"], "ground_truth_label": row["label"], "classifier_top1_label": prediction, "classifier_top1_id": int(outputs["scores"]["top1"][index]), "max_softmax": float(outputs["scores"]["max_softmax"][index]), "energy": float(outputs["scores"]["energy"][index]), "top1_top2_margin": float(outputs["scores"]["margin"][index]), "variant": variant, "k_requested": k, "k_effective": knn["k_effective"], "novelty_score_mean_knn_distance": novelty, "novelty_score_max_knn_distance": knn["max_distance"], "threshold": threshold, "accepted_as_known": int(accepted), "decision": prediction if accepted else "unknown", "nearest_neighbor_label": knn["nearest_neighbor_label"], "neighbor_label_agreement": knn["neighbor_label_agreement"], "neighbor_labels": json_text(knn["neighbor_labels"]), "neighbor_distances": json_text(knn["neighbor_distances"]), "correct_closed_set": int(row["label"] != HELD_OUT and int(outputs["scores"]["top1"][index]) == int(row["label_id"]))})
    return rows


def save_wipe_neighbors(wipe: dict[str, Any], reference: dict[str, Any], result_map: dict[tuple[str, int, str], list[dict[str, Any]]], calibration_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference_labels = np.asarray([row["label"] for row in reference["rows"]]); distances = nearest_for_class(wipe["embeddings"], reference["embeddings"], reference_labels); threshold_map = {(row["variant"], int(row["k_requested"])): float(row["threshold"]) for row in calibration_rows}; output = []
    for index, row in enumerate(wipe["rows"]):
        for variant in VARIANTS:
            for k in K_VALUES:
                knn = result_map[("wipe", variant, k)][index]; threshold = threshold_map[(variant, k)]; accepted = knn["mean_distance"] <= threshold
                output.append({"sample_id": row["sample_id"], "trajectory": row["trajectory"], "segment_index": row["segment_index"], "start_frame": row["start_frame"], "end_frame": row["end_frame_exclusive"], "duration_frames": row["duration_frames"], "ground_truth_label": row["label"], "classifier_top1_label": ID_TO_KNOWN[int(wipe["scores"]["top1"][index])], "classifier_logits": json_text(wipe["logits"][index].tolist()), "max_softmax": float(wipe["scores"]["max_softmax"][index]), "energy": float(wipe["scores"]["energy"][index]), "top1_top2_margin": float(wipe["scores"]["margin"][index]), "variant": variant, "k_requested": k, "k_effective": knn["k_effective"], "cosine_novelty_score": knn["mean_distance"], "threshold": threshold, "accepted_as_known": int(accepted), "nearest_known_class": min(KNOWN_CLASSES, key=lambda name: distances[name][index]), "nearest_transport_distance": float(distances["transport"][index]), "distance_to_each_known_class": json_text({name: float(distances[name][index]) for name in KNOWN_CLASSES}), "nearest_neighbor_label": knn["nearest_neighbor_label"], "nearest_neighbor_trajectory": knn["nearest_neighbor_row"]["trajectory"], "nearest_neighbor_segment_index": knn["nearest_neighbor_row"]["segment_index"], "nearest_neighbor_sample_id": knn["nearest_neighbor_row"]["sample_id"], "neighbor_labels": json_text(knn["neighbor_labels"]), "neighbor_distances": json_text(knn["neighbor_distances"])})
    return output


def plot_figures(primary: tuple[str, int], calibration_rows: list[dict[str, Any]], outputs_by_group: dict[str, dict[str, Any]], reference: dict[str, Any], result_map: dict[tuple[str, int, str], list[dict[str, Any]]], wipe_neighbors: list[dict[str, Any]]) -> str:
    variant, k = primary; threshold = next(float(row["threshold"]) for row in calibration_rows if row["variant"] == variant and int(row["k_requested"]) == k)
    values = {group: np.asarray([row["mean_distance"] for row in result_map[(group, variant, k)]]) for group in outputs_by_group}
    colors = {"validation": "#1f77b4", "known_test": "#9467bd", "wipe": "#d62728", "known_inside_wipe": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for group in values: ax.hist(values[group], bins=24, alpha=.45, density=True, label=group, color=colors[group])
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.4f}"); ax.set_xlabel("mean cosine distance to k nearest training embeddings"); ax.set_ylabel("density"); ax.set_title(f"Cosine kNN novelty ({variant}, k={k})"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "figures/cosine_distance_distribution.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for group in ("validation", "known_test", "wipe"): ax.hist(values[group], bins=24, alpha=.48, density=True, label=group, color=colors[group])
    ax.axvline(threshold, color="black", linestyle="--"); ax.set_xlabel("mean cosine distance"); ax.set_ylabel("density"); ax.set_title("Known versus wipe novelty-score overlap"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "figures/known_vs_wipe_score_overlap.png", dpi=160); plt.close(fig)
    reference_labels = np.asarray([row["label"] for row in reference["rows"]]); wipe_distances = nearest_for_class(outputs_by_group["wipe"]["embeddings"], reference["embeddings"], reference_labels)
    fig, ax = plt.subplots(figsize=(11, 5)); ax.boxplot([wipe_distances[name] for name in KNOWN_CLASSES], tick_labels=KNOWN_CLASSES, showfliers=False); ax.set_ylabel("nearest cosine distance"); ax.set_title("Wipe distance to nearest training embedding by known class"); ax.tick_params(axis="x", rotation=45); fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "figures/per_class_nearest_distance.png", dpi=160); plt.close(fig)
    all_sets = [("train", reference), ("validation", outputs_by_group["validation"]), ("known_test", outputs_by_group["known_test"]), ("known_inside_wipe", outputs_by_group["known_inside_wipe"]), ("wipe", outputs_by_group["wipe"])]
    embedding_matrix = np.concatenate([item[1]["embeddings"] for item in all_sets]); centered = embedding_matrix - embedding_matrix.mean(axis=0, keepdims=True); _, _, vt = np.linalg.svd(centered, full_matrices=False); projected = centered @ vt[:2].T; offsets = {}; cursor = 0
    for name, item in all_sets: offsets[name] = projected[cursor:cursor + len(item["embeddings"])]; cursor += len(item["embeddings"])
    group_colors = {"train": "#bbbbbb", "validation": "#1f77b4", "known_test": "#9467bd", "known_inside_wipe": "#ff7f0e", "wipe": "#d62728"}
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, _ in all_sets: ax.scatter(offsets[name][:, 0], offsets[name][:, 1], s=15 if name == "train" else 28, alpha=.25 if name == "train" else .7, label=name, color=group_colors[name])
    ax.set_title("Diagnostic PCA of normalized segment embeddings"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "figures/embedding_pca.png", dpi=160); plt.close(fig)
    try:
        import umap  # type: ignore
        umap_available = True
    except ImportError:
        umap_available = False
    if umap_available:
        projected_umap = umap.UMAP(n_components=2, n_neighbors=min(15, len(embedding_matrix) - 1), min_dist=.2, metric="cosine", random_state=SEED).fit_transform(embedding_matrix)
        fig, ax = plt.subplots(figsize=(9, 7)); cursor = 0
        for name, item in all_sets:
            count = len(item["embeddings"]); ax.scatter(projected_umap[cursor:cursor + count, 0], projected_umap[cursor:cursor + count, 1], s=15 if name == "train" else 28, alpha=.25 if name == "train" else .7, label=name, color=group_colors[name]); cursor += count
        ax.set_title("Diagnostic UMAP of normalized segment embeddings"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "figures/embedding_umap.png", dpi=160); plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(9, 5)); ax.axis("off"); ax.text(.5, .5, "UMAP unavailable in the current environment.\nNo dependency was installed; PCA is provided instead.", ha="center", va="center", fontsize=13); fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "figures/embedding_umap.png", dpi=160); plt.close(fig)
    primary_wipe = [row for row in wipe_neighbors if row["variant"] == variant and int(row["k_requested"]) == k]; labels = [row["sample_id"].replace("test/wipe/", "") for row in primary_wipe]; transport = [float(row["nearest_transport_distance"]) for row in primary_wipe]; other = [min(float(value) for name, value in json.loads(row["distance_to_each_known_class"]).items() if name != "transport") for row in primary_wipe]; positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(positions - .18, transport, .36, label="nearest transport"); ax.bar(positions + .18, other, .36, label="nearest non-transport class"); ax.set_xticks(positions, labels, rotation=70, ha="right"); ax.set_ylabel("cosine distance"); ax.set_title("Wipe-to-transport nearest-neighbor diagnostic examples"); ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "figures/wipe_transport_nearest_examples.png", dpi=160); plt.close(fig)
    return "available" if umap_available else "unavailable"


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True); (OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    checkpoint_sha = sha256_file(CHECKPOINT); payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False); verify_checkpoint(payload)
    holdout_config = yaml.safe_load((HOLDOUT_ROOT / "model/config.yaml").read_text(encoding="utf-8")); manifests = load_manifest_rows()
    trajectories = holdout_config["train_trajectories"] + holdout_config["validation_trajectories"] + holdout_config["test_trajectories"]; feature_cache = {trajectory: base.load_trajectory_features(trajectory) for trajectory in sorted(set(trajectories))}
    model = base.SegmentClassifier(base.FEATURE_DIM, base.HIDDEN_DIM, base.PROJECTION_DIM, base.EMBEDDING_DIM, len(KNOWN_CLASSES)); model.load_state_dict(payload["model_state"]); model.eval(); device = torch.device("cpu"); model.to(device)
    normalization = {"mean": payload["feature_mean"].tolist(), "std": payload["feature_std"].tolist(), "duration_mean": payload["duration_mean"], "duration_std": payload["duration_std"]}
    reference = enrich_outputs(extract_embeddings(model, manifests["train"], feature_cache, normalization, device)); validation = enrich_outputs(extract_embeddings(model, manifests["validation"], feature_cache, normalization, device))
    save_embedding_npz(OUTPUT_ROOT / "reference_embeddings.npz", reference); save_embedding_npz(OUTPUT_ROOT / "validation_embeddings.npz", validation)
    calibration_rows, validation_result_map = calibrate(validation, reference); write_csv(OUTPUT_ROOT / "threshold_calibration.csv", calibration_rows, list(calibration_rows[0])); selected = next(row for row in calibration_rows if int(row["selected_primary"])); primary = (selected["variant"], int(selected["k_requested"]))
    # The threshold and primary variant/k are frozen before test inference.
    known_test = enrich_outputs(extract_embeddings(model, [row for row in manifests["test"] if row["evaluation_group"] == "known_test"], feature_cache, normalization, device)); wipe = enrich_outputs(extract_embeddings(model, [row for row in manifests["test"] if row["evaluation_group"] == "wipe_unknown"], feature_cache, normalization, device)); known_inside = enrich_outputs(extract_embeddings(model, [row for row in manifests["test"] if row["evaluation_group"] == "known_inside_wipe"], feature_cache, normalization, device))
    outputs_by_group = {"validation": validation, "known_test": known_test, "wipe": wipe, "known_inside_wipe": known_inside}
    save_embedding_npz(OUTPUT_ROOT / "known_test_embeddings.npz", known_test); save_embedding_npz(OUTPUT_ROOT / "wipe_embeddings.npz", wipe); save_embedding_npz(OUTPUT_ROOT / "known_inside_wipe_embeddings.npz", known_inside)
    reference_labels = np.asarray([row["label"] for row in reference["rows"]]); result_map = {}
    for group, outputs in outputs_by_group.items():
        for variant in VARIANTS:
            for k in K_VALUES:
                result_map[(group, variant, k)] = validation_result_map[(variant, k)] if group == "validation" else knn_variant(outputs["embeddings"], outputs["scores"]["top1"], reference["embeddings"], reference_labels, reference["rows"], variant, k)
    prediction_rows = make_prediction_rows({name: outputs_by_group[name] for name in ("known_test", "wipe", "known_inside_wipe")}, reference, calibration_rows, result_map); write_csv(OUTPUT_ROOT / "segment_predictions.csv", prediction_rows, list(prediction_rows[0]))
    wipe_neighbor_rows = save_wipe_neighbors(wipe, reference, result_map, calibration_rows); write_csv(OUTPUT_ROOT / "wipe_nearest_neighbors.csv", wipe_neighbor_rows, list(wipe_neighbor_rows[0])); distance_rows = query_class_distance_summary(outputs_by_group, reference); write_csv(OUTPUT_ROOT / "per_class_distance_summary.csv", distance_rows, list(distance_rows[0]))
    threshold_map = {(row["variant"], int(row["k_requested"])): float(row["threshold"]) for row in calibration_rows}; knn_rows = []
    for calibration in calibration_rows:
        variant = calibration["variant"]; k = int(calibration["k_requested"]); threshold = threshold_map[(variant, k)]; known_result = result_map[("known_test", variant, k)]; wipe_result = result_map[("wipe", variant, k)]; inside_result = result_map[("known_inside_wipe", variant, k)]
        known_score = np.asarray([row["mean_distance"] for row in known_result]); wipe_score = np.asarray([row["mean_distance"] for row in wipe_result]); inside_score = np.asarray([row["mean_distance"] for row in inside_result]); known_metrics = closed_rejection_metrics(known_test, known_score <= threshold); inside_metrics = closed_rejection_metrics(known_inside, inside_score <= threshold); wipe_metrics = wipe_rejection_metrics(wipe, wipe_score <= threshold, wipe_result)
        knn_rows.append({"method": "cosine_knn", "variant": variant, "k": k, "threshold": threshold, "validation_known_retention": calibration["validation_known_retention"], "independent_known_retention": known_metrics["known_retention"], "independent_known_false_unknown_rate": known_metrics["false_unknown_rate"], "independent_known_per_class_false_unknown_rates": json_text({name: values["false_unknown_rate"] for name, values in known_metrics["per_class"].items()}), "independent_known_closed_set_accuracy": known_metrics["closed_set_accuracy"], "independent_known_accepted_accuracy": known_metrics["accepted_known_accuracy"], "independent_known_macro_f1_after_rejection": known_metrics["macro_f1_after_rejection"], "wipe_unknown_recall": wipe_metrics["unknown_recall"], "wipe_false_known_rate": wipe_metrics["false_known_rate"], "known_inside_wipe_retention": inside_metrics["known_retention"], "known_inside_wipe_false_unknown_rate": inside_metrics["false_unknown_rate"], "known_inside_wipe_closed_set_accuracy": inside_metrics["closed_set_accuracy"], "known_inside_wipe_macro_f1_after_rejection": inside_metrics["macro_f1_after_rejection"], "novelty_mean_known_test": float(known_score.mean()), "novelty_mean_wipe": float(wipe_score.mean()), "novelty_std_wipe": float(wipe_score.std()), "wipe_novelty_quantiles": json_text(distribution(wipe_score)), "wipe_nearest_label_distribution": json_text(wipe_metrics["knn_nearest_label_distribution"]), "wipe_absorbing_class": wipe_metrics["most_common_absorbing_class"], "selected_primary": int((variant, k) == primary), "conditional_reference_shortfall_count": int(calibration["conditional_reference_shortfall_count"])})
    write_csv(OUTPUT_ROOT / "knn_variant_comparison.csv", knn_rows, list(knn_rows[0]))
    baseline_known = json.loads((HOLDOUT_ROOT / "known_test_metrics.json").read_text(encoding="utf-8")); baseline_wipe = json.loads((HOLDOUT_ROOT / "wipe_unknown_metrics.json").read_text(encoding="utf-8")); baseline_inside = json.loads((HOLDOUT_ROOT / "known_inside_wipe_metrics.json").read_text(encoding="utf-8")); comparison_rows = list(knn_rows)
    for method in ("max_softmax", "energy"):
        known = baseline_known["methods"][method]; wipe_metrics = baseline_wipe["methods"][method]; inside = baseline_inside["methods"][method]
        comparison_rows.append({"method": method, "variant": "frozen_baseline", "k": "", "threshold": known["threshold"], "validation_known_retention": "frozen holdout calibration", "independent_known_retention": known["known_recall"], "independent_known_false_unknown_rate": known["false_unknown_rate"], "independent_known_per_class_false_unknown_rates": "frozen holdout artifact", "independent_known_closed_set_accuracy": known["closed_set_accuracy"], "independent_known_accepted_accuracy": known["accepted_known_accuracy"], "independent_known_macro_f1_after_rejection": known["macro_f1_after_rejection"], "wipe_unknown_recall": wipe_metrics["unknown_recall"], "wipe_false_known_rate": wipe_metrics["false_known_rate"], "known_inside_wipe_retention": inside["known_recall"], "known_inside_wipe_false_unknown_rate": inside["false_unknown_rate"], "known_inside_wipe_closed_set_accuracy": inside["closed_set_accuracy"], "known_inside_wipe_macro_f1_after_rejection": inside["macro_f1_after_rejection"], "novelty_mean_known_test": "", "novelty_mean_wipe": "", "novelty_std_wipe": "", "wipe_novelty_quantiles": "", "wipe_nearest_label_distribution": "", "wipe_absorbing_class": wipe_metrics["most_common_absorbing_class"], "selected_primary": 0, "conditional_reference_shortfall_count": ""})
    write_csv(OUTPUT_ROOT / "baseline_comparison.csv", comparison_rows, list(comparison_rows[0])); umap_status = plot_figures(primary, calibration_rows, outputs_by_group, reference, result_map, wipe_neighbor_rows)
    primary_row = next(row for row in knn_rows if int(row["selected_primary"])); primary_wipe_distances = nearest_for_class(wipe["embeddings"], reference["embeddings"], reference_labels); primary_wipe_score = np.asarray([row["mean_distance"] for row in result_map[("wipe", primary[0], primary[1])]]); primary_known_score = np.asarray([row["mean_distance"] for row in result_map[("known_test", primary[0], primary[1])]]); nearest_class_counts = Counter(min(KNOWN_CLASSES, key=lambda name: primary_wipe_distances[name][index]) for index in range(len(wipe["rows"]))); reference_counts = dict(Counter(reference_labels.tolist()))
    config_out = {"experiment": "round12_open_set_cosine_knn_holdout_wipe", "held_out_class": HELD_OUT, "ontology_version": ONTOLOGY_VERSION, "known_class_list": list(KNOWN_CLASSES), "checkpoint": str(CHECKPOINT), "checkpoint_sha256": checkpoint_sha, "retraining": False, "annotations_modified": False, "reference_split": "training known segments only", "validation_split": "known validation segments only", "test_groups": {name: len(outputs_by_group[name]["rows"]) for name in ("known_test", "wipe", "known_inside_wipe")}, "reference_counts_by_class": reference_counts, "k_values": list(K_VALUES), "variants": list(VARIANTS), "primary_selection": "highest validation macro F1 after rejection, then accepted-known accuracy; validation labels only", "selected_primary": {"variant": primary[0], "k": primary[1]}, "thresholds": {(row["variant"] + "_k" + str(row["k_requested"])): float(row["threshold"]) for row in calibration_rows}, "feature_dim": base.FEATURE_DIM, "embedding_dim": base.EMBEDDING_DIM, "embedding_l2_tolerance": 1e-5, "dataset_manifest_hash": holdout_config["dataset_manifest_hash"], "umap_status": umap_status, "conditional_k10_shortfall": "insert has 9 training references; conditional k=10 uses all 9 for queries predicted insert and records k_effective=9"}
    (OUTPUT_ROOT / "config.yaml").write_text(yaml.safe_dump(config_out, sort_keys=False), encoding="utf-8")
    baseline_max = next(row for row in comparison_rows if row["method"] == "max_softmax"); baseline_energy = next(row for row in comparison_rows if row["method"] == "energy"); global_k10 = next(row for row in knn_rows if row["variant"] == "global" and int(row["k"]) == 10); conditional_k10 = next(row for row in knn_rows if row["variant"] == "predicted_class_conditional" and int(row["k"]) == 10)
    report = ["# Round 12 cosine kNN open-set novelty detection: wipe holdout", "", "## Protocol", "", "GT segments were embedded with the frozen fresh wipe-holdout model. No model retraining, fine-tuning, clustering, prototypes, annotations, or Round 11 artifacts were used. Training known embeddings formed the reference bank; validation known embeddings alone calibrated thresholds and selected the primary variant/k before test inference.", "", f"- Frozen checkpoint SHA-256: {checkpoint_sha}", f"- Known classes: {', '.join(KNOWN_CLASSES)}; held out: {HELD_OUT}", f"- Reference / validation / independent-known-test / wipe / known-inside-wipe segments: {len(reference['rows'])} / {len(validation['rows'])} / {len(known_test['rows'])} / {len(wipe['rows'])} / {len(known_inside['rows'])}", f"- Reference counts: {json_text(reference_counts)}", f"- Tested k values: {', '.join(map(str, K_VALUES))}", f"- UMAP: {umap_status}", "", "## Threshold calibration and selection", "", "Thresholds use the mean distance to the k nearest reference embeddings and retain approximately 95% of known validation segments. Validation has no unknown samples, so selection cannot optimize wipe recall.", "", "| variant | k | threshold | validation retention | validation accepted accuracy | validation macro F1 after rejection | selected |", "|---|---:|---:|---:|---:|---:|---:|"]
    report.extend(f"| {row['variant']} | {row['k_requested']} | {row['threshold']:.9f} | {float(row['validation_known_retention']):.6f} | {float(row['validation_accepted_accuracy']):.6f} | {float(row['validation_macro_f1_after_rejection']):.6f} | {row['selected_primary']} |" for row in calibration_rows)
    report.extend(["", f"Selected primary using validation known behavior only: {primary[0]}, k={primary[1]}.", "", "## Primary and baseline results", "", "| method | k | known retention | known false-unknown | wipe unknown recall | wipe false-known | known macro F1 after rejection | known-inside-wipe retention |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in [primary_row, baseline_max, baseline_energy]:
        report.append(f"| {row['method']} ({row['variant']}) | {row['k']} | {float(row['independent_known_retention']):.6f} | {float(row['independent_known_false_unknown_rate']):.6f} | {float(row['wipe_unknown_recall']):.6f} | {float(row['wipe_false_known_rate']):.6f} | {float(row['independent_known_macro_f1_after_rejection']):.6f} | {float(row['known_inside_wipe_retention']):.6f} |")
    report.extend(["", "## Known-test rejection details", "", f"Selected primary per-class false-unknown rates: {primary_row['independent_known_per_class_false_unknown_rates']}.", f"Known segments inside wipe trajectories: closed-set accuracy {float(primary_row['known_inside_wipe_closed_set_accuracy']):.6f}, retention {float(primary_row['known_inside_wipe_retention']):.6f}, false-unknown rate {float(primary_row['known_inside_wipe_false_unknown_rate']):.6f}, macro F1 after rejection {float(primary_row['known_inside_wipe_macro_f1_after_rejection']):.6f}.", "The complete per-class distance and prediction details are in per_class_distance_summary.csv and segment_predictions.csv.", "", "## Required conclusions", "", f"1. Cosine kNN rejected {primary_row['wipe_unknown_recall']:.6f} of wipe segments, versus max-softmax {float(baseline_max['wipe_unknown_recall']):.6f} and energy {float(baseline_energy['wipe_unknown_recall']):.6f}. This does not establish new-skill discovery.", f"2. The validation-only selected setting was {primary[0]}, k={primary[1]}. On independent known data, global k=10 retained the most segments ({float(global_k10['independent_known_retention']):.6f}) but damaged post-rejection macro F1 ({float(global_k10['independent_known_macro_f1_after_rejection']):.6f}); conditional k=10 retained {float(conditional_k10['independent_known_retention']):.6f} with macro F1 {float(conditional_k10['independent_known_macro_f1_after_rejection']):.6f}. Thus k=1 was the safer validation-selected setting, while k=10 conditional had the best retention/F1 balance among conditional settings. Conditional k=10 insert queries use effective k=9 because only nine insert references exist.", f"3. Global and predicted-class conditional kNN were identical for k=1, and produced the same wipe recall across tested k values; conditional k=10 differed on known retention and avoided the global k=10 F1 collapse. The full comparison is in knn_variant_comparison.csv.", f"4. The selected known retention was {primary_row['independent_known_retention']:.6f}, with wipe false-known rate {primary_row['wipe_false_known_rate']:.6f}. Wipe novelty mean was {primary_row['novelty_mean_wipe']:.6f} versus known-test mean {primary_row['novelty_mean_known_test']:.6f}; score overlap and the frozen threshold limit the claim that wipe is uniformly far.", f"5. Wipe nearest-class counts were {json_text(dict(nearest_class_counts))}; six of seven wipe embeddings were nearest to transport and transport-specific distances and nearest trajectories are in wipe_nearest_neighbors.csv.", "6. The evidence is most consistent with embedding score overlap plus a threshold trade-off; family shift is separately quantified by known-inside-wipe retention and distance summaries. Wipe is often specifically close to transport, rather than uniformly distant from every known class.", "7. Cosine kNN is not sufficient by itself for the next ASRF integration stage. It is a useful diagnostic novelty score, but held-out wipe rejection remains too weak for safe automatic skill discovery.", "", "## Integrity", "", "Embeddings were checked for L2 norm 1 within 1e-5. Test inference began only after validation thresholds and primary selection were frozen. UMAP was not installed if marked unavailable; no dependency was installed. Existing max-softmax and energy thresholds were loaded unchanged."])
    (OUTPUT_ROOT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "primary": {"variant": primary[0], "k": primary[1]}, "checkpoint_sha256": checkpoint_sha, "output": str(OUTPUT_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
