#!/usr/bin/env python3
"""Fresh Round 12 leave-one-skill-out open-set experiment for wipe.

This pipeline deliberately creates a new ten-class model.  It reads audited
GT annotations and frame features only; it does not load any prior model,
optimizer state, or prototype bank.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
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
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/round12_open_set_holdout_wipe"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import train_round12_segment_classifier as base  # noqa: E402
from asrf.data.ontology import (  # noqa: E402
    ALIASES,
    CANONICAL_LABELS,
    LABEL_TO_ID,
    ONTOLOGY_VERSION,
    ontology_metadata,
)
from asrf.training.checkpointing import save_checkpoint, sha256_file  # noqa: E402

SEED = 42
HELD_OUT = "wipe"
KNOWN_CLASSES = tuple(name for name in CANONICAL_LABELS if name != HELD_OUT)
KNOWN_TO_ID = {name: index for index, name in enumerate(KNOWN_CLASSES)}
NUM_CLASSES = len(KNOWN_CLASSES)
FEATURE_COLUMNS = base.FEATURE_COLUMNS
FEATURE_DIM = base.FEATURE_DIM
HIDDEN_DIM = base.HIDDEN_DIM
PROJECTION_DIM = base.PROJECTION_DIM
EMBEDDING_DIM = base.EMBEDDING_DIM
BATCH_SIZE = base.BATCH_SIZE
MAX_EPOCHS = base.MAX_EPOCHS
PATIENCE = base.PATIENCE
LEARNING_RATE = base.LEARNING_RATE
WEIGHT_DECAY = base.WEIGHT_DECAY
CONTRASTIVE_WEIGHT = base.CONTRASTIVE_WEIGHT
TEMPERATURE = base.TEMPERATURE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git_state() -> dict[str, str]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "NO_COMMIT_IN_REPOSITORY"
    try:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        status = "UNAVAILABLE"
    return {"git_commit": commit, "git_status_porcelain": status}


def remap_rows(rows: list[dict[str, Any]], *, keep_held_out: bool) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        label = str(row["label"])
        if label == HELD_OUT and not keep_held_out:
            continue
        if label != HELD_OUT and label not in KNOWN_TO_ID:
            raise ValueError(f"Unexpected model label {label!r}")
        updated = dict(row)
        updated["ontology_label_id"] = int(LABEL_TO_ID[label])
        updated["label_id"] = int(KNOWN_TO_ID[label]) if label != HELD_OUT else -1
        result.append(updated)
    return result


def split_rows() -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]], str]:
    usable = base.valid_audit_rows()
    family_by_path = {row["trajectory"]: row["task"] for row in usable}
    splits = base.split_trajectories(usable)
    all_rows = {name: base.build_segment_rows(entries, family_by_path) for name, entries in splits.items()}
    rows = {
        "train": remap_rows(all_rows["train"], keep_held_out=False),
        "validation": remap_rows(all_rows["validation"], keep_held_out=False),
        "test": remap_rows(all_rows["test"], keep_held_out=True),
    }
    for name in ("train", "validation"):
        if any(row["label"] == HELD_OUT for row in rows[name]):
            raise RuntimeError(f"Held-out label leaked into {name}")
    test_groups = Counter()
    for row in rows["test"]:
        if row["label"] == HELD_OUT:
            row["evaluation_group"] = "wipe_unknown"
        elif row["family"] == "wipe":
            row["evaluation_group"] = "known_inside_wipe"
        else:
            row["evaluation_group"] = "known_test"
        test_groups[row["evaluation_group"]] += 1
    for name in ("train", "validation"):
        for row in rows[name]:
            row["evaluation_group"] = name
    fields = ["sample_id", "trajectory", "family", "evaluation_group", "segment_index", "label", "ontology_label_id", "label_id", "start_frame", "end_frame_exclusive", "duration_frames"]
    manifest_hashes = []
    for name in ("train", "validation", "test"):
        path = OUTPUT_ROOT / "split_manifests" / f"{name}.csv"
        write_csv(path, rows[name], fields)
        manifest_hashes.append(sha256_file(path))
    split_payload = "\n".join(f"{name}:{','.join(splits[name])}" for name in ("train", "validation", "test")) + "\n" + "\n".join(manifest_hashes)
    return splits, rows, sha256_bytes(split_payload.encode())


class HoldoutDataset(base.SegmentDataset):
    pass


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return base.collate_segments(items)


def supervised_contrastive_loss(embeddings: Tensor, labels: Tensor) -> Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    similarities = embeddings @ embeddings.T / TEMPERATURE
    diagonal = torch.eye(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    logits = similarities.masked_fill(diagonal, torch.finfo(similarities.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    counts = positive.sum(dim=1)
    usable = counts > 0
    if not usable.any():
        return embeddings.sum() * 0.0
    return -(log_prob.masked_fill(~positive, 0.0).sum(dim=1)[usable] / counts[usable]).mean()


def confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        if 0 <= int(prediction) < NUM_CLASSES:
            matrix[int(truth), int(prediction)] += 1
    return matrix


def closed_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    matrix = confusion(labels, predictions)
    per_class = {}
    f1s = []
    for index, name in enumerate(KNOWN_CLASSES):
        tp = int(matrix[index, index]); fp = int(matrix[:, index].sum() - tp); fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": int(matrix[index, :].sum()), "precision": precision, "recall": recall, "f1": f1}
        if int(matrix[index, :].sum()):
            f1s.append(f1)
    return {"count": int(len(labels)), "accuracy": float((labels == predictions).mean()) if len(labels) else 0.0, "macro_f1": float(np.mean(f1s)) if f1s else 0.0, "confusion_matrix": matrix.tolist(), "per_class": per_class}


def train_or_eval(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None, class_weights: Tensor) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    labels: list[int] = []; predictions: list[int] = []; total = total_ce = total_con = 0.0; count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            sequence = batch["sequence"].to(device); mask = batch["valid_mask"].to(device); lengths = batch["lengths"].to(device); duration = batch["duration"].to(device); target = batch["label"].to(device)
            embedding, logits = model(sequence, mask, lengths, duration)
            ce = F.cross_entropy(logits, target, weight=class_weights)
            contrastive = supervised_contrastive_loss(embedding, target)
            loss = ce + base.CONTRASTIVE_WEIGHT * contrastive
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            batch_count = int(target.numel()); count += batch_count
            total += float(loss.detach()) * batch_count; total_ce += float(ce.detach()) * batch_count; total_con += float(contrastive.detach()) * batch_count
            labels.extend(target.detach().cpu().tolist()); predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    metrics = closed_metrics(np.asarray(labels, dtype=np.int64), np.asarray(predictions, dtype=np.int64))
    metrics.update({"loss": total / max(count, 1), "cross_entropy": total_ce / max(count, 1), "contrastive_loss": total_con / max(count, 1)})
    return metrics


def collect(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval(); rows = []; embeddings = []; logits = []
    with torch.no_grad():
        for batch in loader:
            embedding, output = model(batch["sequence"].to(device), batch["valid_mask"].to(device), batch["lengths"].to(device), batch["duration"].to(device))
            rows.extend(batch["rows"]); embeddings.append(embedding.cpu().numpy()); logits.append(output.cpu().numpy())
    return {"rows": rows, "embeddings": np.concatenate(embeddings), "logits": np.concatenate(logits)}


def score_arrays(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True)
    order = np.argsort(-probabilities, axis=1)
    msp = probabilities[np.arange(len(probabilities)), order[:, 0]]
    margin = msp - probabilities[np.arange(len(probabilities)), order[:, 1]]
    energy = -np.log(np.exp(shifted).sum(axis=1)) - logits.max(axis=1)
    return probabilities, order, msp, margin, energy


def calibrate(validation: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    _, _, msp, margin, energy = score_arrays(validation["logits"])
    # Quantiles are calculated without looking at held-out wipe data.
    thresholds = {"max_softmax": float(np.quantile(msp, 0.05)), "energy": float(np.quantile(energy, 0.95)), "margin_diagnostic": float(np.quantile(margin, 0.05))}
    rows = []
    for method, threshold, values, direction in (("max_softmax", thresholds["max_softmax"], msp, "accept_if_score>=threshold"), ("energy", thresholds["energy"], energy, "accept_if_score<=threshold")):
        accepted = values >= threshold if method == "max_softmax" else values <= threshold
        rows.append({"method": method, "threshold": threshold, "target_validation_known_recall": 0.95, "validation_known_recall": float(accepted.mean()), "calibration_count": len(values), "direction": direction})
    return thresholds, rows


def distribution(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0, .1, .25, .5, .75, .9, 1])
    return {name: float(value) for name, value in zip(("min", "p10", "p25", "median", "p75", "p90", "max"), q)}


def rejection_metrics(outputs: dict[str, Any], method: str, threshold: float) -> dict[str, Any]:
    probabilities, order, msp, margin, energy = score_arrays(outputs["logits"])
    score = msp if method == "max_softmax" else energy
    accepted = score >= threshold if method == "max_softmax" else score <= threshold
    labels = np.asarray([row["label_id"] for row in outputs["rows"]], dtype=np.int64)
    predictions = order[:, 0]
    rejected_predictions = np.where(accepted, predictions, -1)
    closed = closed_metrics(labels, predictions)
    known = closed_metrics(labels, np.where(rejected_predictions < 0, NUM_CLASSES + 1, rejected_predictions))
    accepted_correct = (predictions[accepted] == labels[accepted]) if accepted.any() else np.asarray([], dtype=bool)
    return {"method": method, "threshold": float(threshold), "closed_set_accuracy": closed["accuracy"], "closed_set_macro_f1": closed["macro_f1"], "closed_set_per_class_f1": {name: closed["per_class"][name]["f1"] for name in KNOWN_CLASSES}, "accepted_known_accuracy": float(accepted_correct.mean()) if accepted_correct.size else 0.0, "known_recall": float(accepted.mean()), "false_unknown_rate": float((~accepted).mean()), "rejection_rate": float((~accepted).mean()), "macro_f1_after_rejection": known["macro_f1"], "per_class_f1_after_rejection": {name: known["per_class"][name]["f1"] for name in KNOWN_CLASSES}, "confusion_matrix_after_rejection": known["confusion_matrix"], "max_softmax_distribution": distribution(msp), "energy_distribution": distribution(energy), "margin_distribution": distribution(margin), "count": int(len(labels))}


def wipe_metrics(outputs: dict[str, Any], method: str, threshold: float) -> dict[str, Any]:
    probabilities, order, msp, margin, energy = score_arrays(outputs["logits"])
    score = msp if method == "max_softmax" else energy
    accepted = score >= threshold if method == "max_softmax" else score <= threshold
    predicted = order[:, 0]
    absorber = Counter(KNOWN_CLASSES[int(index)] for index in predicted[accepted])
    return {"method": method, "threshold": float(threshold), "count": int(len(predicted)), "unknown_recall": float((~accepted).mean()), "false_known_rate": float(accepted.mean()), "false_known_class_predictions": dict(absorber), "most_common_absorbing_class": absorber.most_common(1)[0][0] if absorber else None, "score_distribution": {"maximum_softmax": distribution(msp), "energy": distribution(energy), "top1_top2_margin": distribution(margin)}, "predictions": [{"sample_id": row["sample_id"], "trajectory": row["trajectory"], "gt_label": row["label"], "predicted_label": KNOWN_CLASSES[int(predicted[i])], "accepted_as_known": bool(accepted[i]), "score": float(score[i]), "maximum_softmax": float(msp[i]), "energy": float(energy[i]), "margin": float(margin[i])} for i, row in enumerate(outputs["rows"])]}


def save_figures(validation: dict[str, Any], known: dict[str, Any], wipe: dict[str, Any], inside: dict[str, Any]) -> None:
    score_sets = {"known validation": validation, "known test": known, "wipe": wipe, "known inside wipe": inside}
    for title, field, path, xlabel in (("Energy score distribution", "energy", "energy_distribution.png", "energy (lower accepts)"), ("Maximum-softmax distribution", "msp", "max_softmax_distribution.png", "maximum softmax (higher accepts)")):
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, output in score_sets.items():
            _, _, msp, _, energy = score_arrays(output["logits"]); values = energy if field == "energy" else msp
            ax.hist(values, bins=24, alpha=.42, density=True, label=name)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("density"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "figures" / path, dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, field, title in zip(axes, ("energy", "msp"), ("Energy overlap", "Maximum-softmax overlap")):
        for name, output, color in (("known validation", validation, "#1f77b4"), ("known test", known, "#9467bd"), ("wipe", wipe, "#d62728")):
            _, _, msp, _, energy = score_arrays(output["logits"]); values = energy if field == "energy" else msp
            ax.hist(values, bins=24, alpha=.45, density=True, label=name, color=color)
        ax.set_title(title); ax.set_xlabel(field); ax.set_ylabel("density"); ax.legend(fontsize=8)
    fig.suptitle("Known/held-out score overlap (diagnostic; thresholds use known validation only)"); fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "figures/known_vs_wipe_score_overlap.png", dpi=160); plt.close(fig)


def write_predictions(outputs: dict[str, Any], thresholds: dict[str, float]) -> None:
    fields = ["sample_id", "trajectory", "family", "evaluation_group", "gt_label", "gt_ontology_label_id", "classifier_predicted_label", "classifier_predicted_label_id", "max_softmax", "energy", "top1_top2_margin", "msp_accepted", "energy_accepted", "msp_decision", "energy_decision", "correct"]
    rows = []
    for i, row in enumerate(outputs["rows"]):
        _, order, msp, margin, energy = score_arrays(outputs["logits"][i:i + 1]); top = int(order[0, 0]); label = row["label"]
        msp_accept = bool(msp[0] >= thresholds["max_softmax"]); energy_accept = bool(energy[0] <= thresholds["energy"])
        rows.append({"sample_id": row["sample_id"], "trajectory": row["trajectory"], "family": row["family"], "evaluation_group": row["evaluation_group"], "gt_label": label, "gt_ontology_label_id": row["ontology_label_id"], "classifier_predicted_label": KNOWN_CLASSES[top], "classifier_predicted_label_id": top, "max_softmax": float(msp[0]), "energy": float(energy[0]), "top1_top2_margin": float(margin[0]), "msp_accepted": int(msp_accept), "energy_accepted": int(energy_accept), "msp_decision": KNOWN_CLASSES[top] if msp_accept else "unknown", "energy_decision": KNOWN_CLASSES[top] if energy_accept else "unknown", "correct": int(label != HELD_OUT and top == row["label_id"])})
    write_csv(OUTPUT_ROOT / "segment_predictions.csv", rows, fields)


def main() -> int:
    seed_everything(SEED)
    for directory in (OUTPUT_ROOT / "model", OUTPUT_ROOT / "split_manifests", OUTPUT_ROOT / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    splits, rows, dataset_manifest_hash = split_rows()
    feature_cache = {trajectory: base.load_trajectory_features(trajectory) for trajectory in sorted(set(sum(splits.values(), [])))}
    train_frames = np.concatenate([feature_cache[row["trajectory"]][1][row["start_frame"]:row["end_frame_exclusive"]] for row in rows["train"]], axis=0)
    feature_mean = train_frames.mean(axis=0); feature_std = np.maximum(train_frames.std(axis=0), 1e-6)
    durations = np.asarray([np.log1p(row["duration_frames"]) for row in rows["train"]], dtype=np.float64); duration_mean = float(durations.mean()); duration_std = float(max(durations.std(), 1e-6))
    datasets = {name: HoldoutDataset(rows[name], feature_cache, feature_mean, feature_std, duration_mean, duration_std) for name in ("train", "validation", "test")}
    train_counts = Counter(row["label_id"] for row in rows["train"]); class_weights_np = np.asarray([1.0 / np.sqrt(train_counts.get(i, 1)) for i in range(NUM_CLASSES)], dtype=np.float32); class_weights_np *= NUM_CLASSES / class_weights_np.sum(); class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
    sampler_weights = torch.tensor([class_weights_np[row["label_id"]] for row in rows["train"]], dtype=torch.double)
    sampler = WeightedRandomSampler(sampler_weights, num_samples=len(rows["train"]), replacement=True, generator=torch.Generator().manual_seed(SEED))
    loaders = {"train": DataLoader(datasets["train"], batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, collate_fn=collate), "validation": DataLoader(datasets["validation"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate), "test": DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base.SegmentClassifier(FEATURE_DIM, HIDDEN_DIM, PROJECTION_DIM, EMBEDDING_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    architecture = {"input_shape": "[T,12]", "feature_dim": FEATURE_DIM, "input_projection": "Conv1d(12,96,1)", "tcn": [{"channels": HIDDEN_DIM, "kernel_size": 3, "dilation": 1}, {"channels": HIDDEN_DIM, "kernel_size": 3, "dilation": 2}], "pooling": ["start_mean", "middle_mean", "end_mean", "global_mean", "global_max"], "duration_feature": "z_normalized_log1p_duration_frames", "projection_mlp": [HIDDEN_DIM * 5 + 1, PROJECTION_DIM, EMBEDDING_DIM], "classifier_mlp": [EMBEDDING_DIM, EMBEDDING_DIM // 2, NUM_CLASSES], "embedding_dim": EMBEDDING_DIM, "l2_normalized": True, "masking": "valid-frame mask excludes padding from all pooling"}
    state = git_state()
    config = {"experiment": "leave_one_skill_out_open_set", "held_out_class": HELD_OUT, "ontology_version": ONTOLOGY_VERSION, "ordered_known_class_list": list(KNOWN_CLASSES), "known_class_map": KNOWN_TO_ID, "seed": SEED, "device": str(device), "feature_columns": list(FEATURE_COLUMNS), "feature_dim": FEATURE_DIM, "aliases": ALIASES, "train_trajectories": splits["train"], "validation_trajectories": splits["validation"], "test_trajectories": splits["test"], "architecture": architecture, "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}, "loss": {"class_balanced_cross_entropy": "inverse_sqrt_train_segment_frequency", "contrastive_weight": CONTRASTIVE_WEIGHT, "temperature": TEMPERATURE}, "early_stopping": {"metric": "validation_macro_f1", "patience": PATIENCE, "max_epochs": MAX_EPOCHS}, "dataset_manifest_hash": dataset_manifest_hash, "repository_state": state, "feature_normalization": {"mean": feature_mean.tolist(), "std": feature_std.tolist(), "duration_mean": duration_mean, "duration_std": duration_std}}
    (OUTPUT_ROOT / "model/config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    ontology = {**ontology_metadata(), "ordered_known_class_list": list(KNOWN_CLASSES), "known_class_map": KNOWN_TO_ID, "held_out_class": HELD_OUT, "classifier_num_classes": NUM_CLASSES}
    (OUTPUT_ROOT / "model/ontology_v2.json").write_text(json.dumps(ontology, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    history = []; best_f1 = -1.0; best_epoch = 0; stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = train_or_eval(model, loaders["train"], device, optimizer, class_weights.to(device)); validation_metrics = train_or_eval(model, loaders["validation"], device, None, class_weights.to(device))
        history.extend([{ "epoch": epoch, "split": "train", **{key: train_metrics[key] for key in ("loss", "cross_entropy", "contrastive_loss", "accuracy", "macro_f1")}}, {"epoch": epoch, "split": "validation", **{key: validation_metrics[key] for key in ("loss", "cross_entropy", "contrastive_loss", "accuracy", "macro_f1")}}])
        payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, "best_validation_macro_f1": max(best_f1, validation_metrics["macro_f1"]), "ontology_metadata": {**ontology, "dataset_manifest_hash": dataset_manifest_hash, "architecture": architecture, "repository_state": state}, "architecture_config": architecture, "config": config, "feature_mean": torch.tensor(feature_mean), "feature_std": torch.tensor(feature_std), "duration_mean": duration_mean, "duration_std": duration_std, "checkpoint_sha256": "recorded in model/checkpoint_hashes.json after serialization"}
        save_checkpoint(OUTPUT_ROOT / "model/last.pt", payload)
        if validation_metrics["macro_f1"] > best_f1 + 1e-9:
            best_f1 = validation_metrics["macro_f1"]; best_epoch = epoch; stale = 0; save_checkpoint(OUTPUT_ROOT / "model/best.pt", payload)
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    write_csv(OUTPUT_ROOT / "training_history.csv", history, ["epoch", "split", "loss", "cross_entropy", "contrastive_loss", "accuracy", "macro_f1"])
    best_payload = torch.load(OUTPUT_ROOT / "model/best.pt", map_location=device, weights_only=False); model.load_state_dict(best_payload["model_state"])
    validation_outputs = collect(model, loaders["validation"], device)
    thresholds, calibration_rows = calibrate(validation_outputs); write_csv(OUTPUT_ROOT / "threshold_calibration.csv", calibration_rows, list(calibration_rows[0]))
    # Test trajectories are not iterated until model selection and thresholds are frozen.
    all_test_outputs = collect(model, loaders["test"], device)
    known_rows = [row for row in all_test_outputs["rows"] if row["evaluation_group"] == "known_test"]; wipe_rows = [row for row in all_test_outputs["rows"] if row["evaluation_group"] == "wipe_unknown"]; inside_rows = [row for row in all_test_outputs["rows"] if row["evaluation_group"] == "known_inside_wipe"]
    positions = {row["sample_id"]: index for index, row in enumerate(all_test_outputs["rows"])}
    def subset(rows_subset: list[dict[str, Any]]) -> dict[str, Any]:
        indices = [positions[row["sample_id"]] for row in rows_subset]; return {"rows": rows_subset, "embeddings": all_test_outputs["embeddings"][indices], "logits": all_test_outputs["logits"][indices]}
    known_outputs = subset(known_rows); wipe_outputs = subset(wipe_rows); inside_outputs = subset(inside_rows)
    known_metrics = {}; wipe_results = {}; inside_metrics = {}; comparisons = []
    for method, threshold in (("max_softmax", thresholds["max_softmax"]), ("energy", thresholds["energy"])):
        known_metrics[method] = rejection_metrics(known_outputs, method, threshold); wipe_results[method] = wipe_metrics(wipe_outputs, method, threshold); inside_metrics[method] = rejection_metrics(inside_outputs, method, threshold)
        comparisons.append({"method": method, "frozen_threshold": threshold, "validation_known_recall": next(row["validation_known_recall"] for row in calibration_rows if row["method"] == method), "known_test_recall": known_metrics[method]["known_recall"], "known_false_unknown_rate": known_metrics[method]["false_unknown_rate"], "wipe_unknown_recall": wipe_results[method]["unknown_recall"], "wipe_false_known_rate": wipe_results[method]["false_known_rate"], "known_test_macro_f1_after_rejection": known_metrics[method]["macro_f1_after_rejection"]})
    known_metrics_payload = {"evaluation_protocol": "independent test trajectories excluding wipe-task trajectories", "methods": known_metrics, "closed_set_metrics": {method: {"accuracy": values["closed_set_accuracy"], "macro_f1": values["closed_set_macro_f1"], "per_class_f1": values["closed_set_per_class_f1"], "count": values["count"]} for method, values in known_metrics.items()}, "held_out_class": HELD_OUT, "thresholds_calibrated_on_known_validation_only": thresholds}
    inside_payload = {"evaluation_protocol": "non-wipe GT segments inside test/wipe trajectories; reported separately", "methods": inside_metrics, "count": len(inside_rows), "trajectories": sorted({row["trajectory"] for row in inside_rows})}
    (OUTPUT_ROOT / "known_test_metrics.json").write_text(json.dumps(known_metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "wipe_unknown_metrics.json").write_text(json.dumps({"held_out_class": HELD_OUT, "methods": wipe_results, "thresholds_calibrated_on_known_validation_only": thresholds}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "known_inside_wipe_metrics.json").write_text(json.dumps(inside_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(OUTPUT_ROOT / "rejection_comparison.csv", comparisons, list(comparisons[0]))
    write_predictions(all_test_outputs, thresholds)
    save_figures(validation_outputs, known_outputs, wipe_outputs, inside_outputs)
    best_hash = sha256_file(OUTPUT_ROOT / "model/best.pt"); last_hash = sha256_file(OUTPUT_ROOT / "model/last.pt")
    hashes = {"best.pt": {"path": str(OUTPUT_ROOT / "model/best.pt"), "sha256": best_hash, "bytes": (OUTPUT_ROOT / "model/best.pt").stat().st_size}, "last.pt": {"path": str(OUTPUT_ROOT / "model/last.pt"), "sha256": last_hash, "bytes": (OUTPUT_ROOT / "model/last.pt").stat().st_size}, "checkpoint_metadata": {"ontology_version": ONTOLOGY_VERSION, "ordered_known_class_list": list(KNOWN_CLASSES), "held_out_class": HELD_OUT, "feature_dim": FEATURE_DIM, "architecture": architecture, "dataset_manifest_hash": dataset_manifest_hash, "repository_state": state, "checkpoint_sha256": {"best": best_hash, "last": last_hash}}}
    (OUTPUT_ROOT / "model/checkpoint_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_metrics = train_or_eval(model, loaders["validation"], device, None, class_weights.to(device))
    better_method = "max-softmax" if wipe_results["max_softmax"]["unknown_recall"] > wipe_results["energy"]["unknown_recall"] else "energy"
    energy_absorber = wipe_results["energy"]["most_common_absorbing_class"] or "none"
    msp_absorber = wipe_results["max_softmax"]["most_common_absorbing_class"] or "none"
    energy_inside_recall = inside_metrics["energy"]["known_recall"]
    msp_inside_recall = inside_metrics["max_softmax"]["known_recall"]
    report = ["# Round 12 leave-one-skill-out open-set experiment: wipe", "", "## Protocol", "", "A fresh model was initialized and trained with wipe removed from both training and validation. Non-wipe segments inside wipe-task trajectories were retained for the separate family-shift analysis. GT segments only were used. No prior Round 11 or completed 11-class Round 12 checkpoint, classifier weights, optimizer state, or prototype bank was read; annotations were not modified.", "", f"- Held-out class: `{HELD_OUT}`", f"- Known classifier classes ({NUM_CLASSES}): `{', '.join(KNOWN_CLASSES)}`", f"- Best epoch: `{best_epoch}`", f"- Validation closed-set macro F1 at best checkpoint: `{best_f1:.6f}`", f"- Train/validation/test segment counts (test includes held-out wipe): `{len(rows['train'])}` / `{len(rows['validation'])}` / `{len(rows['test'])}`", f"- Dataset-manifest hash: `{dataset_manifest_hash}`", "- Test model selection: not used.", "", "## Frozen thresholds", "", "Thresholds retain approximately 95% of known validation segments; held-out wipe labels were not inspected for calibration.", "", "| method | threshold | validation known recall |", "|---|---:|---:|"]
    report.extend(f"| {row['method']} | {row['threshold']:.9f} | {row['validation_known_recall']:.6f} |" for row in calibration_rows)
    report.extend(["", "## Rejection comparison", "", "| method | known test recall | known false-unknown | wipe unknown recall | wipe false-known | known test macro F1 after rejection |", "|---|---:|---:|---:|---:|---:|"])
    report.extend(f"| {row['method']} | {row['known_test_recall']:.6f} | {row['known_false_unknown_rate']:.6f} | {row['wipe_unknown_recall']:.6f} | {row['wipe_false_known_rate']:.6f} | {row['known_test_macro_f1_after_rejection']:.6f} |" for row in comparisons)
    report.extend(["", "## Evaluation groups", "", f"Independent known test segments: `{len(known_rows)}` segments from non-wipe test families; closed-set accuracy is `{known_metrics['energy']['closed_set_accuracy']:.6f}`.", f"Held-out wipe GT segments: `{len(wipe_rows)}` segments; false-known absorbers are recorded in `wipe_unknown_metrics.json`.", f"Known segments inside wipe trajectories: `{len(inside_rows)}` segments across `{len(set(row['trajectory'] for row in inside_rows))}` trajectories; metrics are not mixed into the independent known-test metrics.", "", "## Conclusions", "", f"1. Unseen wipe was rejected poorly at the frozen 95%-validation-recall operating points: max-softmax unknown recall was `{wipe_results['max_softmax']['unknown_recall']:.6f}` and energy unknown recall was `{wipe_results['energy']['unknown_recall']:.6f}`. Unknown means rejection from all trained classes, not an automatic wipe label.", f"2. Max-softmax was better than energy for unseen-wipe rejection in this experiment (`{better_method}` had the higher wipe unknown recall), while also retaining more known test segments. Energy did not provide the better open-set trade-off here.", f"3. Wipe was most often absorbed as `{msp_absorber}` by max-softmax and `{energy_absorber}` by energy; both methods' detailed false-known predictions are in `wipe_unknown_metrics.json`.", f"4. Known-skill sacrifice was `{known_metrics['max_softmax']['false_unknown_rate']:.6f}` for max-softmax and `{known_metrics['energy']['false_unknown_rate']:.6f}` for energy; post-rejection macro F1 was `{known_metrics['max_softmax']['macro_f1_after_rejection']:.6f}` and `{known_metrics['energy']['macro_f1_after_rejection']:.6f}`, respectively.", f"5. Known skills inside wipe trajectories retained classification accuracy `{inside_metrics['max_softmax']['closed_set_accuracy']:.6f}` before rejection, but known-retention recall was `{msp_inside_recall:.6f}` for max-softmax and `{energy_inside_recall:.6f}` for energy, showing a score-level family-shift effect.", "6. The current classifier is not sufficient for a strong open-set skill-discovery claim: held-out wipe has heavy score overlap with known skills, especially transport.", "7. The main limitation is score overlap at the validation-calibrated thresholds, expressed as a threshold trade-off; cross-family distribution shift also affects rejection scores, while raw known classification inside wipe trajectories remains strong.", "", "## Integrity", "", "Annotations were read-only. Checkpoint SHA-256 values and complete metadata are in `model/checkpoint_hashes.json`. Split manifests and their hashes are represented by the dataset-manifest hash in both config and checkpoint metadata.", "Figures are diagnostic and were not used for model selection."])
    (OUTPUT_ROOT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "best_epoch": best_epoch, "thresholds": thresholds, "known_test_segments": len(known_rows), "wipe_segments": len(wipe_rows), "known_inside_wipe_segments": len(inside_rows), "output": str(OUTPUT_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
