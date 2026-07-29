#!/usr/bin/env python3
"""Evaluate the frozen round-11 segment encoder with nearest prototypes.

The execution order is intentional: train/validation embeddings are used to
build and calibrate the prototype bank first.  The wipe manifest is opened
only after the threshold and ``prototype_bank_before_wipe.pt`` are frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_round11_segment_encoder import (
    CLASS_NAMES,
    SegmentDataset,
    SegmentEncoder,
    batch_to_device,
    collate_segments,
    macro_f1,
    read_manifest,
)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from asrf.data.ontology import metadata_for_task, validate_ontology_metadata  # noqa: E402

DATA_DIR = REPO_ROOT / "outputs/round11_segment_embedding/data"
MODEL_PATH = REPO_ROOT / "outputs/round11_segment_embedding/model/best.pt"
OUTPUT_DIR = REPO_ROOT / "outputs/round11_segment_embedding/evaluation_zero_shot"
REQUIRED_RECALL = 0.95


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embed_rows(model: SegmentEncoder, rows: list[dict[str, str]], checkpoint: dict[str, Any], data_dir: Path, device: torch.device, batch_size: int = 16) -> np.ndarray:
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    dataset = SegmentDataset(
        rows, data_dir, feature_mean, feature_std,
        float(checkpoint["duration_log_mean"]), float(checkpoint["duration_log_std"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_segments)
    embeddings: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = batch_to_device(batch, device)
            embedding, _ = model(moved["sequence"], moved["valid_mask"], moved["lengths"], moved["duration"])
            embeddings.append(embedding.cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)


def nearest_prototypes(embeddings: np.ndarray, prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    similarities = embeddings @ prototypes.T
    nearest_ids = similarities.argmax(axis=1).astype(np.int64)
    nearest_values = similarities[np.arange(len(embeddings)), nearest_ids].astype(np.float64)
    return nearest_ids, nearest_values


def choose_threshold(validation_similarities: np.ndarray, required_recall: float) -> tuple[float, float, int]:
    if len(validation_similarities) == 0:
        raise ValueError("Cannot calibrate a threshold without validation embeddings")
    required_count = int(np.ceil(required_recall * len(validation_similarities)))
    ordered = np.sort(validation_similarities)[::-1]
    threshold = float(ordered[required_count - 1])
    achieved_recall = float(np.mean(validation_similarities >= threshold))
    if achieved_recall + 1e-12 < required_recall:
        raise AssertionError(f"Threshold recall constraint failed: {achieved_recall} < {required_recall}")
    return threshold, achieved_recall, required_count


def similarity_distribution(values: np.ndarray) -> dict[str, float | int]:
    if len(values) == 0:
        return {"count": 0}
    quantiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values)),
        "min": float(np.min(values)), "p01": float(quantiles[0]), "p05": float(quantiles[1]),
        "p25": float(quantiles[2]), "median": float(quantiles[3]), "p75": float(quantiles[4]),
        "p95": float(quantiles[5]), "p99": float(quantiles[6]), "max": float(np.max(values)),
    }


def class_metrics(rows: list[dict[str, str]], predicted_ids: np.ndarray, accepted: np.ndarray) -> dict[str, Any]:
    known_indices = [index for index, row in enumerate(rows) if row["label"] in CLASS_NAMES]
    truth = np.asarray([CLASS_NAMES.index(rows[index]["label"]) for index in known_indices], dtype=np.int64)
    predicted = np.asarray([int(predicted_ids[index]) if bool(accepted[index]) else -1 for index in known_indices], dtype=np.int64)
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        true_positive = int(np.sum((truth == class_id) & (predicted == class_id)))
        false_positive = int(np.sum((truth != class_id) & (predicted == class_id)))
        false_negative = int(np.sum((truth == class_id) & (predicted != class_id)))
        support = int(np.sum(truth == class_id))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[class_name] = {"support": support, "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative, "precision": precision, "recall": recall, "f1": f1}
    return {
        "evaluated_known_gt_segments": len(known_indices),
        "accuracy": float(np.mean(predicted == truth)) if len(truth) else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "predicted_unknown_count_on_known_gt": int(np.sum(predicted < 0)),
        "predicted_known_rate_on_known_gt": float(np.mean(predicted >= 0)) if len(predicted) else 0.0,
    }


def prediction_rows(rows: list[dict[str, str]], nearest_ids: np.ndarray, similarities: np.ndarray, threshold: float, split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    accepted = similarities >= threshold
    for index, row in enumerate(rows):
        class_id = int(nearest_ids[index])
        output.append({
            "split": split, "sample_id": row["sample_id"], "trajectory": row["trajectory"], "segment_index": row["segment_index"],
            "ground_truth_label": row["label"], "ground_truth_known": str(row["label"] in CLASS_NAMES).lower(),
            "nearest_prototype_label": CLASS_NAMES[class_id], "nearest_prototype_id": class_id,
            "nearest_prototype_similarity": f"{float(similarities[index]):.9f}", "threshold": f"{threshold:.9f}",
            "decision": "known" if bool(accepted[index]) else "unknown",
            "predicted_label": CLASS_NAMES[class_id] if bool(accepted[index]) else "unknown",
            "predicted_label_id": class_id if bool(accepted[index]) else "",
            "start_frame": row["start_frame"], "end_frame_exclusive": row["end_frame_exclusive"], "duration_frames": row["duration_frames"],
            "known_or_novel": row["known_or_novel"],
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_projection(path: Path, train_embeddings: np.ndarray, validation_embeddings: np.ndarray, pp_embeddings: np.ndarray, wipe_embeddings: np.ndarray, pp_rows: list[dict[str, str]], wipe_rows: list[dict[str, str]]) -> None:
    # Fit the 2-D projection on PP training embeddings only.
    center = train_embeddings.mean(axis=0)
    _, _, components = np.linalg.svd(train_embeddings - center, full_matrices=False)
    basis = components[:2]
    project = lambda values: (values - center) @ basis.T
    train_2d = project(train_embeddings)
    validation_2d = project(validation_embeddings)
    pp_2d = project(pp_embeddings)
    wipe_2d = project(wipe_embeddings)

    figure, axis = plt.subplots(figsize=(10, 8))
    colors = {name: color for name, color in zip(CLASS_NAMES, plt.cm.tab10(np.linspace(0, 0.9, len(CLASS_NAMES))))}
    for class_name in CLASS_NAMES:
        train_mask = np.asarray([row["label"] == class_name for row in read_manifest(DATA_DIR / "train_manifest.csv")])
        val_mask = np.asarray([row["label"] == class_name for row in read_manifest(DATA_DIR / "validation_manifest.csv")])
        axis.scatter(train_2d[train_mask, 0], train_2d[train_mask, 1], color=colors[class_name], marker="o", s=28, alpha=0.55, label=f"train {class_name}")
        axis.scatter(validation_2d[val_mask, 0], validation_2d[val_mask, 1], color=colors[class_name], marker="^", s=42, alpha=0.75)
    pp_known_mask = np.asarray([row["label"] in CLASS_NAMES for row in pp_rows])
    axis.scatter(pp_2d[pp_known_mask, 0], pp_2d[pp_known_mask, 1], facecolors="none", edgecolors="black", marker="s", s=54, linewidths=0.9, label="independent PP test")
    if len(wipe_rows):
        wipe_known_mask = np.asarray([row["label"] in CLASS_NAMES for row in wipe_rows])
        axis.scatter(wipe_2d[wipe_known_mask, 0], wipe_2d[wipe_known_mask, 1], color="darkorange", marker="D", s=45, alpha=0.8, label="wipe trajectory / PP-known GT")
        axis.scatter(wipe_2d[~wipe_known_mask, 0], wipe_2d[~wipe_known_mask, 1], color="black", marker="x", s=50, alpha=0.8, label="wipe trajectory / non-PP-known GT")
    axis.set_title("Round-11 segment embedding projection (PCA fit on PP train)")
    axis.set_xlabel("projection 1")
    axis.set_ylabel("projection 2")
    axis.legend(fontsize=7, ncol=2, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    model_path = args.model.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    validate_ontology_metadata(checkpoint.get("ontology_metadata"), context=str(model_path))
    class_names = tuple(checkpoint["class_names"])
    if class_names != CLASS_NAMES:
        raise ValueError(f"Unexpected frozen class order: {class_names}")
    model = SegmentEncoder(12, 64, 256, 128, len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    checkpoint_hash = sha256_file(model_path)

    # Stage 1: PP train prototypes and validation-only threshold calibration.
    train_rows = read_manifest(data_dir / "train_manifest.csv")
    validation_rows = read_manifest(data_dir / "validation_manifest.csv")
    if any(not row["trajectory"].startswith("train/pick and place/pp") for row in train_rows + validation_rows):
        raise ValueError("Non-PP row entered prototype construction or threshold calibration")
    if any(row["label"] not in CLASS_NAMES for row in train_rows + validation_rows):
        raise ValueError("Non-PP-known label entered prototype construction or threshold calibration")
    train_embeddings = embed_rows(model, train_rows, checkpoint, data_dir, device)
    validation_embeddings = embed_rows(model, validation_rows, checkpoint, data_dir, device)
    train_class_ids = np.asarray([CLASS_NAMES.index(row["label"]) for row in train_rows], dtype=np.int64)
    prototypes = np.vstack([train_embeddings[train_class_ids == class_id].mean(axis=0) for class_id in range(len(CLASS_NAMES))]).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-12)
    if not np.allclose(np.linalg.norm(prototypes, axis=1), 1.0, atol=1e-6):
        raise AssertionError("Prototype normalization failed")
    validation_nearest_ids, validation_similarities = nearest_prototypes(validation_embeddings, prototypes)
    threshold, validation_known_recall, required_count = choose_threshold(validation_similarities, REQUIRED_RECALL)
    threshold_frozen = True
    prototype_bank = {
        "prototype_embeddings": torch.from_numpy(prototypes), "class_names": list(CLASS_NAMES),
        "threshold": threshold, "threshold_rule": "similarity >= threshold -> known prototype; similarity < threshold -> unknown",
        "threshold_selection": "highest observed validation nearest-prototype similarity retaining at least 95% known-skill recall",
        "required_validation_recall": REQUIRED_RECALL, "achieved_validation_recall": validation_known_recall,
        "required_validation_count": required_count, "validation_count": len(validation_rows),
        "prototype_source_trajectories": sorted({row["trajectory"] for row in train_rows}),
        "prototype_source_segment_count": len(train_rows), "frozen_best_checkpoint_sha256": checkpoint_hash,
        "wipe_read_before_freeze": False,
        "ontology_metadata": metadata_for_task(CLASS_NAMES),
    }
    torch.save(prototype_bank, output_dir / "prototype_bank_before_wipe.pt")

    # Stage 2 starts only after the threshold and prototype bank are frozen.
    test_pp_rows = read_manifest(data_dir / "test_pp_manifest.csv")
    test_pp_embeddings = embed_rows(model, test_pp_rows, checkpoint, data_dir, device)
    pp_nearest_ids, pp_similarities = nearest_prototypes(test_pp_embeddings, prototypes)
    pp_accepted = pp_similarities >= threshold
    pp_metrics = class_metrics(test_pp_rows, pp_nearest_ids, pp_accepted)
    pp_metrics.update({
        "frozen_checkpoint_sha256": checkpoint_hash, "threshold": threshold, "threshold_frozen_before_test": True,
        "validation_known_recall_at_threshold": validation_known_recall, "validation_count": len(validation_rows),
        "test_trajectory_count": len({row["trajectory"] for row in test_pp_rows}), "test_segment_count": len(test_pp_rows),
        "test_known_gt_segment_count": sum(row["label"] in CLASS_NAMES for row in test_pp_rows),
        "test_out_of_ontology_labels": sorted({row["label"] for row in test_pp_rows if row["label"] not in CLASS_NAMES}),
        "test_out_of_ontology_segment_count": sum(row["label"] not in CLASS_NAMES for row in test_pp_rows),
        "nearest_similarity_distribution": similarity_distribution(pp_similarities),
        "predicted_known_count_all_test_rows": int(np.sum(pp_accepted)), "predicted_unknown_count_all_test_rows": int(np.sum(~pp_accepted)),
    })
    (output_dir / "pp_test_metrics.json").write_text(json.dumps(pp_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pp_prediction_rows = prediction_rows(test_pp_rows, pp_nearest_ids, pp_similarities, threshold, "test_pp")

    # Wipe is deliberately opened after the bank and threshold have been saved.
    test_wipe_rows = read_manifest(data_dir / "test_wipe_manifest.csv")
    test_wipe_embeddings = embed_rows(model, test_wipe_rows, checkpoint, data_dir, device)
    wipe_nearest_ids, wipe_similarities = nearest_prototypes(test_wipe_embeddings, prototypes)
    wipe_accepted = wipe_similarities >= threshold
    unknown_mask = np.asarray([row["label"] not in CLASS_NAMES for row in test_wipe_rows], dtype=bool)
    wipe_label_mask = np.asarray([row["label"] == "wipe" for row in test_wipe_rows], dtype=bool)
    known_pp_mask = ~unknown_mask
    unknown_recall = float(np.mean(~wipe_accepted[unknown_mask])) if unknown_mask.any() else 0.0
    false_known_rate = float(np.mean(wipe_accepted[unknown_mask])) if unknown_mask.any() else 0.0
    wipe_unknown_recall = float(np.mean(~wipe_accepted[wipe_label_mask])) if wipe_label_mask.any() else 0.0
    wipe_false_known_rate = float(np.mean(wipe_accepted[wipe_label_mask])) if wipe_label_mask.any() else 0.0
    known_pp_metrics = class_metrics(test_wipe_rows, wipe_nearest_ids, wipe_accepted)
    wipe_metrics = {
        "frozen_checkpoint_sha256": checkpoint_hash, "threshold": threshold, "threshold_frozen_before_wipe": threshold_frozen,
        "evaluated_trajectory_count": len({row["trajectory"] for row in test_wipe_rows}), "evaluated_trajectories": sorted({row["trajectory"] for row in test_wipe_rows}),
        "requested_trajectories": [f"test/wipe/w{i}" for i in range(1, 5)],
        "missing_requested_trajectories": [f"test/wipe/w{i}" for i in range(1, 5) if f"test/wipe/w{i}" not in {row["trajectory"] for row in test_wipe_rows}],
        "evaluated_segment_count": len(test_wipe_rows), "ground_truth_unknown_labels": sorted({row["label"] for row in test_wipe_rows if row["label"] not in CLASS_NAMES}),
        "ground_truth_unknown_segment_count": int(unknown_mask.sum()), "unknown_recall": unknown_recall, "false_known_rate": false_known_rate,
        "wipe_label_segment_count": int(wipe_label_mask.sum()), "wipe_label_unknown_recall": wipe_unknown_recall, "wipe_label_false_known_rate": wipe_false_known_rate,
        "pp_known_segments_inside_wipe": known_pp_metrics,
        "nearest_prototype_similarity_distribution": {"all_wipe_segments": similarity_distribution(wipe_similarities), "ground_truth_unknown_segments": similarity_distribution(wipe_similarities[unknown_mask]), "ground_truth_wipe_segments": similarity_distribution(wipe_similarities[wipe_label_mask]), "ground_truth_pp_known_segments": similarity_distribution(wipe_similarities[known_pp_mask])},
        "interpretation_note": "Unknown is a nearest-prototype rejection under the frozen threshold; unknown is not automatically interpreted as wipe.",
    }
    (output_dir / "wipe_unknown_metrics.json").write_text(json.dumps(wipe_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wipe_prediction_rows = prediction_rows(test_wipe_rows, wipe_nearest_ids, wipe_similarities, threshold, "test_wipe")
    write_csv(output_dir / "segment_predictions.csv", pp_prediction_rows + wipe_prediction_rows)
    plot_projection(output_dir / "embedding_projection.png", train_embeddings, validation_embeddings, test_pp_embeddings, test_wipe_embeddings, test_pp_rows, test_wipe_rows)

    report = [
        "# Round-11 zero-shot prototype evaluation", "", "## Protocol", "",
        "The frozen `best.pt` checkpoint was loaded without retraining. Prototypes are normalized class means built only from PP train segments pp1–pp10. The threshold was calibrated only on PP validation segments pp11–pp20 and was saved before the wipe manifest was opened.", "",
        f"- Frozen checkpoint SHA-256: `{checkpoint_hash}`", f"- Prototype source segments: {len(train_rows)}", f"- Validation calibration segments: {len(validation_rows)}", f"- Threshold: `{threshold:.9f}`", f"- Validation known-skill recall at threshold: `{validation_known_recall:.6f}`", "- Decision: similarity >= threshold is known; otherwise unknown.", "",
        "## Independent PP test", "", f"Accuracy on PP-known GT segments: **{pp_metrics['accuracy']:.6f}**", f"Macro F1 on PP-known GT segments: **{pp_metrics['macro_f1']:.6f}**", f"Known-GT segments evaluated: {pp_metrics['evaluated_known_gt_segments']} of {pp_metrics['test_segment_count']}; test-only labels excluded from class metrics: `{', '.join(pp_metrics['test_out_of_ontology_labels']) or 'none'}`.", "", "| class | support | precision | recall | F1 |", "|---|---:|---:|---:|---:|"]
    for class_name in CLASS_NAMES:
        item = pp_metrics["per_class"][class_name]
        report.append(f"| {class_name} | {item['support']} | {item['precision']:.6f} | {item['recall']:.6f} | {item['f1']:.6f} |")
    report += ["", "## Wipe trajectories", "", f"Evaluated trajectories: `{', '.join(wipe_metrics['evaluated_trajectories']) or 'none'}`", f"Missing requested trajectories: `{', '.join(wipe_metrics['missing_requested_trajectories']) or 'none'}`", f"Unknown GT labels used for rejection metrics: `{', '.join(wipe_metrics['ground_truth_unknown_labels']) or 'none'}`", f"Unknown recall: **{unknown_recall:.6f}**", f"False-known rate: **{false_known_rate:.6f}**", f"Wipe-label-only unknown recall: **{wipe_unknown_recall:.6f}**", f"Wipe-label-only false-known rate: **{wipe_false_known_rate:.6f}**", "", "PP-known segments inside wipe trajectories were evaluated with the same nearest-prototype rule rather than being forced to unknown:", "", f"- Count: {known_pp_metrics['evaluated_known_gt_segments']}", f"- Accuracy: {known_pp_metrics['accuracy']:.6f}", f"- Macro F1: {known_pp_metrics['macro_f1']:.6f}", "", "Nearest-prototype similarity distributions are recorded in `wipe_unknown_metrics.json` for all wipe segments and GT subsets.", "", "Unknown means rejection by the frozen PP prototype threshold. It is not automatically interpreted as wipe.", "", "## Artifacts", "", "- `prototype_bank_before_wipe.pt`", "- `pp_test_metrics.json`", "- `wipe_unknown_metrics.json`", "- `segment_predictions.csv`", "- `embedding_projection.png`", "- `report.md`", ""]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"threshold": threshold, "validation_known_recall": validation_known_recall, "pp_accuracy": pp_metrics["accuracy"], "pp_macro_f1": pp_metrics["macro_f1"], "wipe_unknown_recall": unknown_recall, "wipe_false_known_rate": false_known_rate, "missing_wipe": wipe_metrics["missing_requested_trajectories"], "output_dir": str(output_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
