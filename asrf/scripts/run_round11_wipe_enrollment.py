#!/usr/bin/env python3
"""Run the frozen-encoder few-shot wipe prototype-enrollment experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_round11_zero_shot import embed_rows, read_manifest, similarity_distribution
from train_round11_segment_encoder import CLASS_NAMES, SegmentEncoder
from asrf.data.ontology import metadata_for_task, validate_ontology_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "outputs/round11_segment_embedding/data"
MODEL_PATH = REPO_ROOT / "outputs/round11_segment_embedding/model/best.pt"
PP_BANK_PATH = REPO_ROOT / "outputs/round11_segment_embedding/evaluation_zero_shot/prototype_bank_before_wipe.pt"
OUTPUT_DIR = REPO_ROOT / "outputs/round11_segment_embedding/wipe_enrollment"
THRESHOLD = 0.689683914
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest_fields() -> list[str]:
    return [
        "sample_id", "trajectory", "source_path", "task", "segment_index", "label", "label_id",
        "start_frame", "end_frame", "end_frame_exclusive", "num_frames", "duration", "duration_s",
        "duration_frames", "duration_us", "start_timestamp_us", "end_timestamp_us_exclusive",
        "frame_feature_path", "frame_feature_shape", "frame_feature_columns", "split", "known_or_novel",
        "manifest_role",
    ]


def write_support_query(run_id: str, support_rows: list[dict[str, str]], query_rows: list[dict[str, str]], directory: Path) -> None:
    fields = manifest_fields()
    for role, rows in (("support", support_rows), ("query", query_rows)):
        exported = [{**row, "manifest_role": role} for row in rows]
        write_csv(directory / f"{run_id}_{role}.csv", exported, fields)


def choose_support_segments(wipe_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    rng = random.Random(SEED)
    selected: dict[str, dict[str, str]] = {}
    trajectories = sorted({row["trajectory"] for row in wipe_rows})
    for trajectory in trajectories:
        candidates = sorted((row for row in wipe_rows if row["trajectory"] == trajectory and row["label"] == "wipe"), key=lambda row: (int(row["segment_index"]), row["sample_id"]))
        if not candidates:
            raise ValueError(f"No GT wipe segment available in {trajectory}")
        selected[trajectory] = rng.choice(candidates)
    return selected


def normalize_prototype(embeddings: np.ndarray) -> np.ndarray:
    prototype = embeddings.mean(axis=0).astype(np.float32)
    norm = float(np.linalg.norm(prototype))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize zero wipe prototype")
    return prototype / norm


def build_bank(original: dict[str, Any], wipe_prototype: np.ndarray | None, support_rows: list[dict[str, str]], shot: int, bank_path: Path) -> dict[str, Any]:
    pp_prototypes = original["prototype_embeddings"].detach().cpu().float()
    if wipe_prototype is None:
        bank = dict(original)
        bank["enrollment_shot"] = 0
        bank["enrollment_support_sample_ids"] = []
        bank["original_pp_bank_sha256"] = sha256_file(PP_BANK_PATH)
    else:
        bank = {
            "prototype_embeddings": torch.cat((pp_prototypes, torch.from_numpy(wipe_prototype).float().unsqueeze(0)), dim=0),
            "class_names": list(CLASS_NAMES) + ["wipe"],
            "threshold": float(original["threshold"]),
            "threshold_rule": original["threshold_rule"],
            "original_pp_prototype_embeddings": pp_prototypes,
            "wipe_prototype_embedding": torch.from_numpy(wipe_prototype).float(),
            "enrollment_shot": shot,
            "enrollment_support_sample_ids": [row["sample_id"] for row in support_rows],
            "enrollment_support_trajectories": sorted({row["trajectory"] for row in support_rows}),
            "frozen_best_checkpoint_sha256": original.get("frozen_best_checkpoint_sha256", sha256_file(MODEL_PATH)),
            "original_pp_bank_sha256": sha256_file(PP_BANK_PATH),
            "threshold_recalibrated": False,
            "ontology_metadata": metadata_for_task(tuple(CLASS_NAMES) + ("wipe",)),
        }
    torch.save(bank, bank_path)
    return bank


def predict(embeddings: np.ndarray, bank: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    prototypes = bank["prototype_embeddings"].detach().cpu().numpy().astype(np.float32)
    class_names = list(bank["class_names"])
    similarities = embeddings @ prototypes.T
    order = np.argsort(-similarities, axis=1)
    pp_similarities = embeddings @ prototypes[: len(CLASS_NAMES)].T
    pp_ids = pp_similarities.argmax(axis=1)
    pp_nearest = pp_similarities[np.arange(len(embeddings)), pp_ids]
    results: list[dict[str, Any]] = []
    for index in range(len(embeddings)):
        top1_id = int(order[index, 0])
        top2_id = int(order[index, 1]) if similarities.shape[1] > 1 else top1_id
        top1 = float(similarities[index, top1_id])
        top2 = float(similarities[index, top2_id])
        if top1 < threshold:
            status = "unknown"
            predicted_label = "unknown"
        elif top1_id == len(CLASS_NAMES):
            status = "wipe"
            predicted_label = "wipe"
        else:
            status = "PP-known"
            predicted_label = class_names[top1_id]
        results.append({
            "top1_id": top1_id, "top1_label": class_names[top1_id], "top1_similarity": top1,
            "top2_id": top2_id, "top2_label": class_names[top2_id], "top2_similarity": top2,
            "margin": top1 - top2, "predicted_status": status, "predicted_label": predicted_label,
            "nearest_pp_prototype": class_names[int(pp_ids[index])], "nearest_pp_similarity": float(pp_nearest[index]),
            "similarity_to_wipe_prototype": float(similarities[index, len(CLASS_NAMES)]) if similarities.shape[1] > len(CLASS_NAMES) else "",
        })
    return results


def diagnostic_rows(rows: list[dict[str, str]], predictions: list[dict[str, Any]], run_id: str, role: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        gt_label = row["label"]
        output.append({
            "run_id": run_id, "evaluation_role": role, "sample_id": row["sample_id"], "trajectory": row["trajectory"],
            "segment_index": row["segment_index"], "ground_truth_label": gt_label,
            "predicted_label": prediction["predicted_label"], "predicted_status": prediction["predicted_status"],
            "similarity_to_wipe_prototype": prediction["similarity_to_wipe_prototype"],
            "nearest_pp_prototype": prediction["nearest_pp_prototype"], "nearest_pp_similarity": f"{prediction['nearest_pp_similarity']:.9f}",
            "top1_label": prediction["top1_label"], "top1_similarity": f"{prediction['top1_similarity']:.9f}",
            "top2_label": prediction["top2_label"], "top2_similarity": f"{prediction['top2_similarity']:.9f}",
            "margin": f"{prediction['margin']:.9f}", "threshold": f"{THRESHOLD:.9f}",
            "correct_or_incorrect": "correct" if prediction["predicted_label"] == gt_label else "incorrect",
            "status_correct": "correct" if ((gt_label == "wipe" and prediction["predicted_status"] == "wipe") or (gt_label in CLASS_NAMES and prediction["predicted_status"] == "PP-known" and prediction["predicted_label"] == gt_label) or (gt_label not in CLASS_NAMES and prediction["predicted_status"] == "unknown")) else "incorrect",
        })
    return output


def class_metrics(rows: list[dict[str, str]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    indices = [index for index, row in enumerate(rows) if row["label"] in CLASS_NAMES]
    truth = np.asarray([CLASS_NAMES.index(rows[index]["label"]) for index in indices], dtype=np.int64)
    predicted = np.asarray([CLASS_NAMES.index(predictions[index]["predicted_label"]) if predictions[index]["predicted_label"] in CLASS_NAMES else -1 for index in indices], dtype=np.int64)
    per_class: dict[str, Any] = {}
    f1s: list[float] = []
    for class_id, name in enumerate(CLASS_NAMES):
        tp = int(np.sum((truth == class_id) & (predicted == class_id)))
        fp = int(np.sum((truth != class_id) & (predicted == class_id)))
        fn = int(np.sum((truth == class_id) & (predicted != class_id)))
        support = int(np.sum(truth == class_id))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[name] = {"support": support, "precision": precision, "recall": recall, "f1": f1, "true_positive": tp, "false_positive": fp, "false_negative": fn}
    return {
        "count": len(indices), "accuracy": float(np.mean(predicted == truth)) if len(truth) else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"), "per_class": per_class,
        "predicted_unknown_count": int(np.sum(predicted < 0)),
    }


def wipe_metrics(rows: list[dict[str, str]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    wipe_indices = [index for index, row in enumerate(rows) if row["label"] == "wipe"]
    true_wipe = np.asarray([row["label"] == "wipe" for row in rows], dtype=bool)
    predicted_wipe = np.asarray([item["predicted_status"] == "wipe" for item in predictions], dtype=bool)
    tp = int(np.sum(true_wipe & predicted_wipe)); fp = int(np.sum(~true_wipe & predicted_wipe)); fn = int(np.sum(true_wipe & ~predicted_wipe))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    wipe_predicted_unknown = float(np.mean([predictions[index]["predicted_status"] == "unknown" for index in wipe_indices])) if wipe_indices else float("nan")
    wipe_false_known = float(np.mean([predictions[index]["predicted_status"] == "PP-known" for index in wipe_indices])) if wipe_indices else float("nan")
    confusion: dict[str, int] = {}
    for index in wipe_indices:
        label = predictions[index]["predicted_label"]
        confusion[label] = confusion.get(label, 0) + 1
    return {
        "query_segment_count": len(rows), "wipe_query_count": len(wipe_indices), "wipe_precision": precision, "wipe_recall": recall,
        "wipe_F1": f1, "wipe_accuracy": float(np.mean(predicted_wipe == true_wipe)) if len(rows) else float("nan"),
        "wipe_unknown_rate": wipe_predicted_unknown, "wipe_false_known_rate": wipe_false_known,
        "wipe_confusion": dict(sorted(confusion.items())),
        "nearest_pp_absorption_on_wipe": dict(sorted({label: int(sum(1 for index in wipe_indices if predictions[index]["nearest_pp_prototype"] == label)) for label in CLASS_NAMES}.items())),
        "wipe_similarity_to_enrolled_prototype": similarity_distribution(np.asarray([float(predictions[index]["similarity_to_wipe_prototype"]) for index in wipe_indices if predictions[index]["similarity_to_wipe_prototype"] != ""], dtype=np.float64)),
        "nearest_pp_similarity_on_wipe": similarity_distribution(np.asarray([predictions[index]["nearest_pp_similarity"] for index in wipe_indices], dtype=np.float64)),
        "top1_minus_top2_margin_on_wipe": similarity_distribution(np.asarray([predictions[index]["margin"] for index in wipe_indices], dtype=np.float64)),
    }


def save_prediction_file(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_id", "evaluation_role", "sample_id", "trajectory", "segment_index", "ground_truth_label", "predicted_label", "predicted_status",
        "similarity_to_wipe_prototype", "nearest_pp_prototype", "nearest_pp_similarity", "top1_label", "top1_similarity", "top2_label", "top2_similarity", "margin", "threshold", "correct_or_incorrect", "status_correct",
    ]
    write_csv(path, rows, fields)


def summary_value(values: list[float], mode: str) -> float:
    if mode == "mean":
        return float(np.mean(values)) if values else float("nan")
    return float(np.std(values, ddof=0)) if values else float("nan")


def result_row(shot: str, run_id: str, support_rows: list[dict[str, str]], query_rows: list[dict[str, str]], independent: bool, wipe: dict[str, Any], pp: dict[str, Any], wipe_pp: dict[str, Any], note: str = "") -> dict[str, Any]:
    fit = wipe.get("fit_diagnostic", {})
    return {
        "shot": shot, "run_id": run_id, "support_trajectories": "|".join(sorted({row["trajectory"] for row in support_rows})), "query_trajectories": "|".join(sorted({row["trajectory"] for row in query_rows})), "independent_query": str(independent).lower(), "wipe_support_count": len(support_rows), "wipe_query_count": wipe.get("wipe_query_count", ""), "query_segment_count": wipe.get("query_segment_count", ""), "wipe_precision": wipe.get("wipe_precision", ""), "wipe_recall": wipe.get("wipe_recall", ""), "wipe_F1": wipe.get("wipe_F1", ""), "wipe_accuracy": wipe.get("wipe_accuracy", ""), "wipe_unknown_rate": wipe.get("wipe_unknown_rate", ""), "wipe_false_known_rate": wipe.get("wipe_false_known_rate", ""), "wipe_fit_precision": fit.get("wipe_precision", ""), "wipe_fit_recall": fit.get("wipe_recall", ""), "wipe_fit_F1": fit.get("wipe_F1", ""), "wipe_fit_accuracy": fit.get("wipe_accuracy", ""), "PP_test_accuracy": pp.get("accuracy", ""), "PP_test_macro_F1": pp.get("macro_f1", ""), "PP_accuracy_change": pp.get("accuracy_change", ""), "PP_macro_F1_change": pp.get("macro_f1_change", ""), "PP_known_inside_wipe_accuracy": wipe_pp.get("accuracy", ""), "PP_known_inside_wipe_macro_F1": wipe_pp.get("macro_f1", ""), "note": note,
    }


def plot_similarity(path: Path, zero_wipe: np.ndarray, one_wipe: dict[str, np.ndarray], three_wipe_fit: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    labels = ["0-shot\nnearest PP"]
    values = [zero_wipe]
    for run_id, array in sorted(one_wipe.items()):
        labels.append(f"1-shot\n{run_id.replace('1_shot_support_', '')}")
        values.append(array)
    labels.append("3-shot\nfit diagnostic")
    values.append(three_wipe_fit)
    axis.boxplot(values, tick_labels=labels, showmeans=True)
    axis.axhline(THRESHOLD, color="red", linestyle="--", linewidth=1.2, label=f"frozen threshold={THRESHOLD:.3f}")
    axis.set_ylabel("cosine similarity")
    axis.set_title("Wipe-segment similarity before and after prototype enrollment")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_embedding(path: Path, train_embeddings: np.ndarray, pp_prototypes: np.ndarray, pp_test_embeddings: np.ndarray, wipe_embeddings: dict[str, tuple[np.ndarray, list[str]]], support_ids: set[str]) -> None:
    center = train_embeddings.mean(axis=0)
    _, _, components = np.linalg.svd(train_embeddings - center, full_matrices=False)
    basis = components[:2]
    project = lambda array: (array - center) @ basis.T
    figure, axis = plt.subplots(figsize=(10, 8))
    train_2d = project(train_embeddings)
    pp_2d = project(pp_prototypes)
    test_2d = project(pp_test_embeddings)
    axis.scatter(train_2d[:, 0], train_2d[:, 1], s=18, alpha=0.25, color="steelblue", label="PP train segments")
    axis.scatter(test_2d[:, 0], test_2d[:, 1], s=22, alpha=0.35, color="gray", label="PP test segments")
    axis.scatter(pp_2d[:, 0], pp_2d[:, 1], marker="*", s=200, color="red", edgecolors="black", label="PP prototypes")
    for index, name in enumerate(CLASS_NAMES):
        axis.annotate(name, (pp_2d[index, 0], pp_2d[index, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    for trajectory, (embeddings, sample_ids) in sorted(wipe_embeddings.items()):
        points = project(embeddings)
        support_mask = np.asarray([sample_id in support_ids for sample_id in sample_ids])
        if support_mask.any():
            axis.scatter(points[support_mask, 0], points[support_mask, 1], s=100, marker="*", alpha=0.9, label=f"{trajectory} support")
        if (~support_mask).any():
            axis.scatter(points[~support_mask, 0], points[~support_mask, 1], s=45, marker="D", alpha=0.65, label=f"{trajectory} query-capable")
    axis.set_title("Diagnostic embedding projection (PCA fit on PP train embeddings)")
    axis.set_xlabel("projection 1")
    axis.set_ylabel("projection 2")
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    data_dir = DATA_DIR.resolve()
    output = OUTPUT_DIR.resolve()
    for directory in (output, output / "support_query_manifests", output / "prototype_banks", output / "predictions", output / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    validate_ontology_metadata(checkpoint.get("ontology_metadata"), context=str(MODEL_PATH))
    model = SegmentEncoder(12, 64, 256, 128, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    original_bank = torch.load(PP_BANK_PATH, map_location="cpu", weights_only=False)
    validate_ontology_metadata(original_bank.get("ontology_metadata"), context=str(PP_BANK_PATH))
    original_threshold = float(original_bank["threshold"])
    if abs(original_threshold - THRESHOLD) > 1e-9:
        raise ValueError(f"Frozen threshold mismatch: bank={original_threshold}, requested={THRESHOLD}")
    encoder_hash = sha256_file(MODEL_PATH)
    pp_bank_hash = sha256_file(PP_BANK_PATH)
    pp_prototypes = original_bank["prototype_embeddings"].detach().cpu().numpy().astype(np.float32)

    train_rows = read_manifest(data_dir / "train_manifest.csv")
    pp_test_rows = read_manifest(data_dir / "test_pp_manifest.csv")
    # Wipe is deliberately not read until after all prototypes and the 0.689... threshold are frozen.
    train_embeddings = embed_rows(model, train_rows, checkpoint, data_dir, torch.device("cpu"))
    pp_test_embeddings = embed_rows(model, pp_test_rows, checkpoint, data_dir, torch.device("cpu"))
    wipe_rows = read_manifest(data_dir / "test_wipe_manifest.csv")
    wipe_embeddings_matrix = embed_rows(model, wipe_rows, checkpoint, data_dir, torch.device("cpu"))
    wipe_embedding_by_id = {row["sample_id"]: embedding for row, embedding in zip(wipe_rows, wipe_embeddings_matrix)}
    support_by_trajectory = choose_support_segments(wipe_rows)
    support_by_id = {row["sample_id"]: row for row in support_by_trajectory.values()}
    wipe_trajectories = sorted({row["trajectory"] for row in wipe_rows})

    # Exact requested support/query manifests.
    runs: list[dict[str, Any]] = []
    zero_query_sets = {
        "0_shot_pooled_all": wipe_rows,
        **{f"0_shot_rotation_{trajectory.split('/')[-1]}": [row for row in wipe_rows if row["trajectory"] != trajectory] for trajectory in wipe_trajectories},
    }
    one_query_sets = {f"1_shot_support_{trajectory.split('/')[-1]}": [row for row in wipe_rows if row["trajectory"] != trajectory] for trajectory in wipe_trajectories}
    for run_id, query_rows in zero_query_sets.items():
        runs.append({"shot": "0-shot", "run_id": run_id, "support_rows": [], "query_rows": query_rows, "independent": True})
    for run_id, query_rows in one_query_sets.items():
        support_trajectory = run_id.removeprefix("1_shot_support_")
        support_rows = [support_by_trajectory[f"test/wipe/{support_trajectory}"]]
        runs.append({"shot": "1-shot", "run_id": run_id, "support_rows": support_rows, "query_rows": query_rows, "independent": True})
    three_support = [support_by_trajectory[trajectory] for trajectory in wipe_trajectories]
    runs.append({"shot": "3-shot", "run_id": "3_shot_support_w1_w2_w3", "support_rows": three_support, "query_rows": [], "independent": False})

    original_bank_path = output / "prototype_banks/0_shot_original_pp.pt"
    build_bank(original_bank, None, [], 0, original_bank_path)
    banks: dict[str, dict[str, Any]] = {"0-shot": original_bank}
    bank_paths: dict[str, Path] = {"0-shot": original_bank_path}
    for run in runs:
        if run["shot"] == "1-shot":
            support_embeddings = np.vstack([wipe_embedding_by_id[row["sample_id"]] for row in run["support_rows"]])
            wipe_prototype = normalize_prototype(support_embeddings)
            bank_path = output / "prototype_banks" / f"{run['run_id']}.pt"
            banks[run["run_id"]] = build_bank(original_bank, wipe_prototype, run["support_rows"], 1, bank_path)
            bank_paths[run["run_id"]] = bank_path
        elif run["shot"] == "3-shot":
            support_embeddings = np.vstack([wipe_embedding_by_id[row["sample_id"]] for row in run["support_rows"]])
            wipe_prototype = normalize_prototype(support_embeddings)
            bank_path = output / "prototype_banks" / f"{run['run_id']}.pt"
            banks[run["run_id"]] = build_bank(original_bank, wipe_prototype, run["support_rows"], 3, bank_path)
            bank_paths[run["run_id"]] = bank_path

    # Record support/query manifests before evaluation files are written.
    for run in runs:
        write_support_query(run["run_id"], run["support_rows"], run["query_rows"], output / "support_query_manifests")
    write_csv(output / "support_query_manifests/3_shot_support_fit_eval.csv", [{**row, "manifest_role": "non_independent_enrollment_fit_diagnostic"} for row in three_support], manifest_fields())

    baseline_pp = {"accuracy": 0.933333, "macro_f1": 0.958333}
    results: list[dict[str, Any]] = []
    pp_retention: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    diagnostic_by_run: dict[str, list[dict[str, Any]]] = {}
    wipe_metrics_by_run: dict[str, dict[str, Any]] = {}
    pp_metrics_by_run: dict[str, dict[str, Any]] = {}
    wipe_pp_metrics_by_run: dict[str, dict[str, Any]] = {}
    one_shot_wipe_similarity: dict[str, np.ndarray] = {}
    zero_wipe_similarity: np.ndarray | None = None
    three_fit_similarity: np.ndarray | None = None

    for run in runs:
        run_id = run["run_id"]
        bank = banks["0-shot"] if run["shot"] == "0-shot" else banks[run_id]
        query_rows = run["query_rows"]
        query_embeddings = np.vstack([wipe_embedding_by_id[row["sample_id"]] for row in query_rows]) if query_rows else np.empty((0, 128), dtype=np.float32)
        query_predictions = predict(query_embeddings, bank, THRESHOLD) if len(query_rows) else []
        if query_rows:
            diagnostic = diagnostic_rows(query_rows, query_predictions, run_id, "independent_query")
            diagnostic_by_run[run_id] = diagnostic
            save_prediction_file(output / "predictions" / f"{run_id}.csv", diagnostic)
            wipe = wipe_metrics(query_rows, query_predictions)
            wipe_pp = class_metrics([row for row in query_rows if row["label"] in CLASS_NAMES], [item for row, item in zip(query_rows, query_predictions) if row["label"] in CLASS_NAMES])
            if run["shot"] == "0-shot" and run_id == "0_shot_pooled_all":
                zero_wipe_similarity = np.asarray([float(item["nearest_pp_similarity"]) for row, item in zip(query_rows, query_predictions) if row["label"] == "wipe"], dtype=np.float64)
            if run["shot"] == "1-shot":
                one_shot_wipe_similarity[run_id] = np.asarray([float(item["similarity_to_wipe_prototype"]) for row, item in zip(query_rows, query_predictions) if row["label"] == "wipe"], dtype=np.float64)
        else:
            wipe = {"wipe_query_count": 0, "query_segment_count": 0}
            wipe_pp = {"count": 0, "accuracy": float("nan"), "macro_f1": float("nan"), "per_class": {}}
            fit_predictions = predict(np.vstack([wipe_embedding_by_id[row["sample_id"]] for row in run["support_rows"]]), bank, THRESHOLD)
            fit_diagnostic = diagnostic_rows(run["support_rows"], fit_predictions, run_id, "non_independent_enrollment_fit_diagnostic")
            diagnostic_by_run[run_id] = fit_diagnostic
            save_prediction_file(output / "predictions" / f"{run_id}_enrollment_fit.csv", fit_diagnostic)
            fit_wipe = wipe_metrics(run["support_rows"], fit_predictions)
            wipe["fit_diagnostic"] = fit_wipe
            three_fit_similarity = np.asarray([float(item["similarity_to_wipe_prototype"]) for item in fit_predictions], dtype=np.float64)
        test_pp_predictions = predict(pp_test_embeddings, bank, THRESHOLD)
        pp = class_metrics(pp_test_rows, test_pp_predictions)
        pp["accuracy_change"] = pp["accuracy"] - baseline_pp["accuracy"]
        pp["macro_f1_change"] = pp["macro_f1"] - baseline_pp["macro_f1"]
        write_csv(output / "predictions" / f"pp_test_{run_id}.csv", diagnostic_rows(pp_test_rows, test_pp_predictions, run_id, "independent_pp_test"), [
            "run_id", "evaluation_role", "sample_id", "trajectory", "segment_index", "ground_truth_label", "predicted_label", "predicted_status", "similarity_to_wipe_prototype", "nearest_pp_prototype", "nearest_pp_similarity", "top1_label", "top1_similarity", "top2_label", "top2_similarity", "margin", "threshold", "correct_or_incorrect", "status_correct",
        ])
        pp_retention.append({"shot": run["shot"], "run_id": run_id, "prototype_bank": str(bank_paths["0-shot"] if run["shot"] == "0-shot" else bank_paths[run_id]), "PP_test_accuracy": pp["accuracy"], "PP_test_macro_F1": pp["macro_f1"], "PP_accuracy_change": pp["accuracy_change"], "PP_macro_F1_change": pp["macro_f1_change"], "PP_macro_F1_damage_gt_0.02": str(pp["macro_f1_change"] < -0.02).lower(), "PP_known_inside_wipe_accuracy": wipe_pp.get("accuracy", ""), "PP_known_inside_wipe_macro_F1": wipe_pp.get("macro_f1", ""), "PP_known_inside_wipe_count": wipe_pp.get("count", "")})
        pp_metrics_by_run[run_id] = pp
        wipe_metrics_by_run[run_id] = wipe
        wipe_pp_metrics_by_run[run_id] = wipe_pp
        results.append(result_row(run["shot"], run_id, run["support_rows"], run["query_rows"], run["independent"], wipe, pp, wipe_pp, "" if run["independent"] else "non-independent enrollment-fit diagnostic; no independent wipe query trajectory"))
        if query_rows:
            for label in (*CLASS_NAMES, "wipe", "unknown"):
                confusion_rows.append({"shot": run["shot"], "run_id": run_id, "ground_truth_label": "wipe", "predicted_label": label, "count": int(wipe["wipe_confusion"].get(label, 0))})

    # Summary over independent 1-shot rotations, with matched 0-shot rotation baselines.
    one_rows = [row for row in results if row["shot"] == "1-shot"]
    zero_rotation_rows = [row for row in results if row["shot"] == "0-shot" and row["run_id"].startswith("0_shot_rotation_")]
    for shot_name, source_rows, note in (("0-shot-rotation-summary", zero_rotation_rows, "mean/std across matched rotation query sets"), ("1-shot-rotation-summary", one_rows, "mean/std across support-trajectory rotations")):
        for mode in ("mean", "std"):
            numeric_fields = ["wipe_precision", "wipe_recall", "wipe_F1", "wipe_accuracy", "wipe_unknown_rate", "wipe_false_known_rate", "PP_test_accuracy", "PP_test_macro_F1", "PP_accuracy_change", "PP_macro_F1_change", "PP_known_inside_wipe_accuracy", "PP_known_inside_wipe_macro_F1"]
            values: dict[str, Any] = {field: summary_value([float(row[field]) for row in source_rows if row[field] not in ("", "nan")], mode) for field in numeric_fields}
            values.update({"shot": shot_name, "run_id": f"{mode}_across_rotations", "support_trajectories": "", "query_trajectories": "other two available wipe trajectories per rotation", "independent_query": "true", "wipe_support_count": 0 if shot_name.startswith("0") else 1, "wipe_query_count": int(round(np.mean([int(row["wipe_query_count"]) for row in source_rows]))), "query_segment_count": int(round(np.mean([int(row["query_segment_count"]) for row in source_rows]))), "note": note})
            results.append(values)

    results_fields = ["shot", "run_id", "support_trajectories", "query_trajectories", "independent_query", "wipe_support_count", "wipe_query_count", "query_segment_count", "wipe_precision", "wipe_recall", "wipe_F1", "wipe_accuracy", "wipe_unknown_rate", "wipe_false_known_rate", "wipe_fit_precision", "wipe_fit_recall", "wipe_fit_F1", "wipe_fit_accuracy", "PP_test_accuracy", "PP_test_macro_F1", "PP_accuracy_change", "PP_macro_F1_change", "PP_known_inside_wipe_accuracy", "PP_known_inside_wipe_macro_F1", "note"]
    write_csv(output / "results_0_1_3_shot.csv", results, results_fields)
    write_csv(output / "pp_retention_metrics.csv", pp_retention, ["shot", "run_id", "prototype_bank", "PP_test_accuracy", "PP_test_macro_F1", "PP_accuracy_change", "PP_macro_F1_change", "PP_macro_F1_damage_gt_0.02", "PP_known_inside_wipe_accuracy", "PP_known_inside_wipe_macro_F1", "PP_known_inside_wipe_count"])
    write_csv(output / "wipe_confusion.csv", confusion_rows, ["shot", "run_id", "ground_truth_label", "predicted_label", "count"])

    # Similarity comparison uses 0-shot matched pooled wipe labels, 1-shot
    # independent query labels, and 3-shot support fit only.
    if zero_wipe_similarity is None:
        zero_wipe_similarity = np.empty(0)
    if three_fit_similarity is None:
        three_fit_similarity = np.empty(0)
    plot_similarity(output / "figures/similarity_comparison.png", zero_wipe_similarity, one_shot_wipe_similarity, three_fit_similarity)
    wipe_embedding_groups = {trajectory: (np.vstack([wipe_embedding_by_id[row["sample_id"]] for row in wipe_rows if row["trajectory"] == trajectory]), [row["sample_id"] for row in wipe_rows if row["trajectory"] == trajectory]) for trajectory in wipe_trajectories}
    plot_embedding(output / "figures/embedding_projection.png", train_embeddings, pp_prototypes, pp_test_embeddings, wipe_embedding_groups, set(support_by_id))

    enrolled_hashes = {shot: {run_id: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for run_id, path in bank_paths.items() if (shot == "0-shot" and run_id == "0-shot") or (shot == "1-shot" and run_id.startswith("1_shot")) or (shot == "3-shot" and run_id.startswith("3_shot"))} for shot in ("0-shot", "1-shot", "3-shot")}
    hashes = {"frozen_encoder": {"path": str(MODEL_PATH), "sha256": encoder_hash, "bytes": MODEL_PATH.stat().st_size}, "original_pp_prototype_bank": {"path": str(PP_BANK_PATH), "sha256": pp_bank_hash, "bytes": PP_BANK_PATH.stat().st_size}, "enrolled_prototype_banks": enrolled_hashes}
    (output / "prototype_banks/hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    baseline = next(row for row in results if row["run_id"] == "0_shot_pooled_all")
    one_independent_f1 = [float(row["wipe_F1"]) for row in one_rows]
    one_mean_f1 = float(np.mean(one_independent_f1)) if one_independent_f1 else float("nan")
    one_std_f1 = float(np.std(one_independent_f1)) if one_independent_f1 else float("nan")
    one_after_absorption: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    for run_id in [row["run_id"] for row in one_rows]:
        for label, count in wipe_metrics_by_run[run_id]["nearest_pp_absorption_on_wipe"].items():
            one_after_absorption[label] += count
    before_absorption = wipe_metrics_by_run["0_shot_pooled_all"]["nearest_pp_absorption_on_wipe"]
    before_class = max(before_absorption, key=before_absorption.get) if before_absorption else "none"
    after_class = max(one_after_absorption, key=one_after_absorption.get) if one_after_absorption else "none"
    min_enrolled_pp_f1 = min(float(row["PP_test_macro_F1"]) for row in pp_retention if row["shot"] != "0-shot")
    support_lines = [f"- `{trajectory}`: `{support_by_trajectory[trajectory]['sample_id']}` (segment `{support_by_trajectory[trajectory]['segment_index']}`)" for trajectory in wipe_trajectories]
    three_fit_metrics = wipe_metrics_by_run["3_shot_support_w1_w2_w3"].get("fit_diagnostic", {})
    report = [
        "# Few-shot wipe prototype enrollment", "", "## Frozen protocol", "",
        "The Round 11 encoder and PP prototype bank were loaded without retraining, fine-tuning, annotation changes, or ASRF-predicted segments. The PP prototypes were left unchanged; enrolled banks append one normalized wipe prototype.", "", f"- Encoder SHA-256: `{encoder_hash}`", f"- Original PP prototype bank SHA-256: `{pp_bank_hash}`", f"- Frozen threshold: `{THRESHOLD:.9f}`", "- Threshold was not recalibrated after enrollment.", f"- Available wipe trajectories: `{', '.join(wipe_trajectories)}`; missing requested trajectory: `{'test/wipe/w4' if 'test/wipe/w4' not in wipe_trajectories else 'none'}`.", "- Support selection seed: 42; one GT `wipe` segment per support trajectory.", "", "## Results", "", f"0-shot pooled wipe F1: **{float(baseline['wipe_F1']):.6f}**", f"1-shot rotation wipe F1 mean ± SD: **{one_mean_f1:.6f} ± {one_std_f1:.6f}**", f"3-shot enrollment-fit wipe precision/recall/F1: **{three_fit_metrics.get('wipe_precision', float('nan')):.6f} / {three_fit_metrics.get('wipe_recall', float('nan')):.6f} / {three_fit_metrics.get('wipe_F1', float('nan')):.6f}**; this is not independent.", "3-shot: **non-independent enrollment-fit diagnostic only**; all three trajectories are used for support, so no independent wipe query trajectory remains.", "", "## Exact support selections", "", *support_lines, "", "For each 1-shot run, the query contains every GT segment from the other two trajectories. The 3-shot query manifest is intentionally empty because trajectory-disjoint independent wipe query data do not remain.", "", "### Required conclusions", "", f"1. **1-shot improvement:** {'yes' if one_mean_f1 > float(baseline['wipe_F1']) else 'no'} versus the matched 0-shot pooled baseline; the rotation-level results are in `results_0_1_3_shot.csv`.", f"2. **Rotation consistency:** {'consistent' if one_std_f1 <= 0.10 else 'not consistent'} by the observed F1 SD (`{one_std_f1:.6f}`); this is only three support rotations.", "3. **3-shot evidence:** no independent generalization evidence is available; support-fit results are diagnostic only.", f"4. **Pre-enrollment PP absorber:** `{before_class}` by nearest-PP counts on wipe GT segments (`{before_absorption}`).", f"5. **Post-enrollment PP confusion:** `{after_class}` remains the largest nearest-PP absorber across 1-shot rotations (`{one_after_absorption}`).", f"6. **PP retention:** the lowest enrolled PP-test macro F1 is `{min_enrolled_pp_f1:.6f}` versus the pre-enrollment baseline `0.958333`; damage greater than 0.02 is {'observed' if 0.958333 - min_enrolled_pp_f1 > 0.02 else 'not observed'}.", "7. **Threshold usability:** the frozen threshold remains the decision rule after adding wipe; no query labels were used to recalibrate it.", "8. **Encoder suitability:** the encoder is usable for a preliminary prototype-registration test, but the limited cross-trajectory wipe evidence is not sufficient for a strong generalization claim.", "9. **More data:** yes. The missing w4 and only three available trajectories make additional wipe data necessary before a strong claim.", "", "## Interpretation", "", "`wipe_accuracy` in the results table is binary wipe-vs-not-wipe accuracy over all query segments; wipe precision/recall/F1 are the primary wipe recognition metrics. Unknown is a threshold rejection, not an automatic wipe label. Wipe GT segments rejected by the threshold and wipe GT segments assigned to a PP class are reported separately. PP-known segments inside wipe trajectories are evaluated under the same nearest-prototype rule and are not forced to wipe or unknown.", "", "## Outputs", "", "- `support_query_manifests/`: exact trajectory-disjoint support/query manifests.", "- `prototype_banks/`: original PP bank, 1-shot banks, 3-shot bank, and hashes.", "- `predictions/`: wipe-query diagnostics and PP retention predictions.", "- `figures/similarity_comparison.png`: similarity distributions; 3-shot is marked fit-only.", "- `figures/embedding_projection.png`: diagnostic PCA projection fit on PP train embeddings.", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "encoder_sha256": encoder_hash, "pp_bank_sha256": pp_bank_hash, "zero_shot_wipe_f1": baseline["wipe_F1"], "one_shot_wipe_f1_mean": one_mean_f1, "one_shot_wipe_f1_std": one_std_f1, "three_shot_independent_query": False, "missing_w4": "test/wipe/w4" not in wipe_trajectories}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
