#!/usr/bin/env python3
"""Train and evaluate a fresh Round 12 GT-segment classifier.

This script intentionally has no dependency on Round 11 checkpoints,
prototypes, or optimizer state. It uses only the audited external annotations
and the existing per-frame feature CSVs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
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
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
AUDIT_ROOT = ROOT / "outputs/round12_multiskill_segment_embedding/data_audit"
OUTPUT_ROOT = ROOT / "outputs/round12_multiskill_segment_classifier"
LABEL_CONFIG = ROOT / "configs/labels_multiskill_v2.yaml"
sys.path.insert(0, str(ROOT / "src"))

from asrf.data.ontology import (  # noqa: E402
    ALIASES,
    CANONICAL_LABELS,
    LABEL_TO_ID,
    ONTOLOGY_VERSION,
    metadata_for_task,
    ontology_metadata,
)
from asrf.training.checkpointing import save_checkpoint, sha256_file  # noqa: E402

SEED = 42
FEATURE_COLUMNS = (
    "citr_ff", "citr_ftau", "citr_tautau", "citr_fv", "citr_tauv", "citr_vv",
    "citr_fw", "citr_tauw", "citr_vw", "citr_ww", "gripper_position", "gripper_norm",
)
FEATURE_DIM = len(FEATURE_COLUMNS)
HIDDEN_DIM = 96
PROJECTION_DIM = 256
EMBEDDING_DIM = 128
NUM_CLASSES = len(CANONICAL_LABELS)
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
CONTRASTIVE_WEIGHT = 0.1
TEMPERATURE = 0.07


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


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "NO_COMMIT_IN_REPOSITORY"


def canonical_label(raw: str) -> str:
    name = str(raw).strip()
    name = ALIASES.get(name, name)
    if name not in LABEL_TO_ID:
        raise ValueError(f"Unknown label in training data: {raw!r}")
    return name


def valid_audit_rows() -> list[dict[str, str]]:
    rows = list(csv.DictReader((AUDIT_ROOT / "trajectory_manifest.csv").open(encoding="utf-8")))
    usable = [row for row in rows if not row["missing_files"] and int(row["schema_valid"]) and not int(row["blank_labels"])]
    if any(int(row["remaining_align_annotations"] or 0) for row in rows):
        raise RuntimeError("Audit contains active align annotations; training is prohibited.")
    if any(row["unknown_labels"] or int(row["gaps"]) or int(row["overlaps"]) or int(row["zero_length_segments"]) or int(row["invalid_intervals"]) for row in rows):
        raise RuntimeError("Audit integrity gates are not all passing; training is prohibited.")
    if len(usable) != 106:
        raise RuntimeError(f"Expected 106 usable trajectories after audit, found {len(usable)}.")
    return usable


def trajectory_path(relative: str) -> Path:
    return DATA_ROOT / relative


def split_trajectories(usable: list[dict[str, str]]) -> dict[str, list[str]]:
    observed = {row["trajectory"] for row in usable}
    train = [*(f"train/pick and place/pp{i}" for i in range(1, 11)), *(f"train/wipe/w{i}" for i in range(1, 11)), *(f"train/pour/p{i}" for i in range(1, 13)), *(f"train/plug/p{i}" for i in range(1, 10))]
    validation = [*(f"train/pick and place/pp{i}" for i in range(11, 21)), *(f"train/wipe/w{i}" for i in range(11, 14)), *(f"train/pour/p{i}" for i in range(13, 17)), *(f"train/plug/p{i}" for i in range(10, 13))]
    test = sorted(row["trajectory"] for row in usable if row["split"] == "test")
    for name, entries in (("train", train), ("validation", validation)):
        missing = sorted(set(entries) - observed)
        if missing:
            raise RuntimeError(f"{name} split has missing usable trajectories: {missing}")
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("Trajectory-level split leakage detected.")
    return {"train": train, "validation": validation, "test": test}


def load_trajectory_features(relative: str) -> tuple[np.ndarray, np.ndarray]:
    path = trajectory_path(relative)
    fields, rows = read_csv_rows(path / "citr_features.csv")
    missing = [name for name in ("timestamp_us", *FEATURE_COLUMNS) if name not in fields]
    if missing:
        raise ValueError(f"{relative}: missing feature columns {missing}")
    timestamps = np.asarray([int(row["timestamp_us"]) for row in rows], dtype=np.int64)
    features = np.asarray([[float(row[name]) for name in FEATURE_COLUMNS] for row in rows], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM or len(timestamps) != len(features):
        raise ValueError(f"{relative}: invalid feature shape {features.shape}")
    return timestamps, features


def build_segment_rows(trajectories: list[str], family_by_path: dict[str, str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for relative in trajectories:
        path = trajectory_path(relative)
        fields, annotation_rows = read_csv_rows(path / "segments.csv")
        if "label" not in fields:
            raise ValueError(f"{relative}/segments.csv: canonical label column is missing")
        timestamps, _ = load_trajectory_features(relative)
        for row_number, annotation in enumerate(annotation_rows):
            label = canonical_label(annotation.get("label", ""))
            start = int(np.searchsorted(timestamps, int(annotation["start_timestamp_us"]), side="left"))
            end = int(np.searchsorted(timestamps, int(annotation["end_timestamp_us_exclusive"]), side="left"))
            if end <= start or start < 0 or end > len(timestamps):
                raise ValueError(f"{relative}/segments.csv row {row_number + 2}: invalid interval")
            output.append({
                "sample_id": f"{relative}#segment{annotation.get('segment_index', row_number)}",
                "trajectory": relative,
                "family": family_by_path[relative],
                "segment_index": int(annotation.get("segment_index", row_number)),
                "label": label,
                "label_id": LABEL_TO_ID[label],
                "start_frame": start,
                "end_frame_exclusive": end,
                "duration_frames": end - start,
            })
    return output


class SegmentDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], feature_cache: dict[str, tuple[np.ndarray, np.ndarray]], feature_mean: np.ndarray, feature_std: np.ndarray, duration_mean: float, duration_std: float) -> None:
        self.rows = rows
        self.feature_cache = feature_cache
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.duration_mean = float(duration_mean)
        self.duration_std = float(duration_std)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        _, features = self.feature_cache[row["trajectory"]]
        sequence = (features[row["start_frame"]:row["end_frame_exclusive"]] - self.feature_mean) / self.feature_std
        duration = (np.log1p(row["duration_frames"]) - self.duration_mean) / self.duration_std
        return {"sequence": torch.from_numpy(sequence.astype(np.float32)), "duration": torch.tensor(duration, dtype=torch.float32), "label": torch.tensor(row["label_id"], dtype=torch.long), "row": row}


def collate_segments(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["sequence"].shape[0] for item in items], dtype=torch.long)
    max_length = int(lengths.max())
    sequence = torch.zeros((len(items), max_length, FEATURE_DIM), dtype=torch.float32)
    valid_mask = torch.zeros((len(items), max_length), dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(item["sequence"].shape[0])
        sequence[index, :length] = item["sequence"]
        valid_mask[index, :length] = True
    return {"sequence": sequence, "valid_mask": valid_mask, "lengths": lengths, "duration": torch.stack([item["duration"] for item in items]), "label": torch.stack([item["label"] for item in items]), "rows": [item["row"] for item in items]}


class SegmentClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, projection_dim: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.tcn1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.tcn2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.dropout = nn.Dropout(0.15)
        pooled_dim = hidden_dim * 5 + 1
        self.projection = nn.Sequential(nn.Linear(pooled_dim, projection_dim), nn.GELU(), nn.Dropout(0.15), nn.Linear(projection_dim, embedding_dim))
        self.classifier = nn.Sequential(nn.Linear(embedding_dim, embedding_dim // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(embedding_dim // 2, num_classes))

    @staticmethod
    def masked_mean(values: Tensor, mask: Tensor, fallback: Tensor) -> Tensor:
        weights = mask.unsqueeze(1).to(values.dtype)
        count = weights.sum(dim=2)
        result = (values * weights).sum(dim=2) / count.clamp_min(1.0)
        return torch.where(count > 0, result, fallback)

    def forward(self, sequence: Tensor, valid_mask: Tensor, lengths: Tensor, duration: Tensor) -> tuple[Tensor, Tensor]:
        mask = valid_mask.unsqueeze(1)
        values = sequence.masked_fill(~valid_mask.unsqueeze(2), 0.0).transpose(1, 2)
        values = F.gelu(self.input_projection(values)).masked_fill(~mask, 0.0)
        values = F.gelu(self.tcn1(values)).masked_fill(~mask, 0.0)
        values = self.dropout(F.gelu(self.tcn2(values))).masked_fill(~mask, 0.0)
        global_mean = self.masked_mean(values, valid_mask, torch.zeros_like(values[:, :, 0]))
        global_max = values.masked_fill(~mask, torch.finfo(values.dtype).min).amax(dim=2)
        time = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
        first_end = torch.div(lengths + 2, 3, rounding_mode="floor").unsqueeze(1)
        second_end = torch.div(2 * lengths + 2, 3, rounding_mode="floor").unsqueeze(1)
        start = valid_mask & (time < first_end)
        middle = valid_mask & (time >= first_end) & (time < second_end)
        end = valid_mask & (time >= second_end)
        pooled = torch.cat((self.masked_mean(values, start, global_mean), self.masked_mean(values, middle, global_mean), self.masked_mean(values, end, global_mean), global_mean, global_max, duration.unsqueeze(1)), dim=1)
        embedding = F.normalize(self.projection(pooled), p=2, dim=1, eps=1e-8)
        return embedding, self.classifier(embedding)


def supervised_contrastive_loss(embeddings: Tensor, labels: Tensor, temperature: float) -> Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    similarities = embeddings @ embeddings.T / temperature
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
        matrix[int(truth), int(prediction)] += 1
    return matrix


def metric_bundle(labels: np.ndarray, predictions: np.ndarray, *, macro_over_supported: bool = False) -> dict[str, Any]:
    matrix = confusion(labels, predictions)
    per_class: dict[str, dict[str, float | int]] = {}
    scores = []
    for index, name in enumerate(CANONICAL_LABELS):
        tp = int(matrix[index, index]); fp = int(matrix[:, index].sum() - tp); fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": int(matrix[index, :].sum()), "precision": precision, "recall": recall, "f1": f1}
        if not macro_over_supported or int(matrix[index, :].sum()):
            scores.append(f1)
    return {"count": int(len(labels)), "accuracy": float((labels == predictions).mean()) if len(labels) else 0.0, "macro_f1": float(np.mean(scores)) if scores else 0.0, "confusion_matrix": matrix.tolist(), "per_class": per_class}


def run_model(model: SegmentClassifier, loader: DataLoader[dict[str, Any]], device: torch.device, optimizer: torch.optim.Optimizer | None, class_weights: Tensor | None) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_ce = total_con = 0.0; count = 0; labels: list[int] = []; predictions: list[int] = []
    with torch.set_grad_enabled(training):
        for batch in loader:
            sequence = batch["sequence"].to(device); valid_mask = batch["valid_mask"].to(device); lengths = batch["lengths"].to(device); duration = batch["duration"].to(device); target = batch["label"].to(device)
            embedding, logits = model(sequence, valid_mask, lengths, duration)
            ce = F.cross_entropy(logits, target, weight=class_weights)
            contrastive = supervised_contrastive_loss(embedding, target, TEMPERATURE)
            loss = ce + CONTRASTIVE_WEIGHT * contrastive
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            batch_count = int(target.numel()); count += batch_count; total_loss += float(loss.detach()) * batch_count; total_ce += float(ce.detach()) * batch_count; total_con += float(contrastive.detach()) * batch_count
            labels.extend(target.detach().cpu().tolist()); predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    metrics = metric_bundle(np.asarray(labels, dtype=np.int64), np.asarray(predictions, dtype=np.int64))
    metrics.update({"loss": total_loss / max(count, 1), "cross_entropy": total_ce / max(count, 1), "contrastive_loss": total_con / max(count, 1)})
    return metrics


def collect_outputs(model: SegmentClassifier, loader: DataLoader[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    model.eval(); rows: list[dict[str, Any]] = []; embeddings: list[np.ndarray] = []; logits_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            embedding, logits = model(batch["sequence"].to(device), batch["valid_mask"].to(device), batch["lengths"].to(device), batch["duration"].to(device))
            embeddings.append(embedding.cpu().numpy()); logits_all.append(logits.cpu().numpy())
            rows.extend(batch["rows"])
    return {"rows": rows, "embeddings": np.concatenate(embeddings), "logits": np.concatenate(logits_all)}


def scores(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True)
    order = np.argsort(-probabilities, axis=1)
    max_probability = probabilities[np.arange(len(probabilities)), order[:, 0]]
    margin = max_probability - probabilities[np.arange(len(probabilities)), order[:, 1]]
    energy = -np.log(np.exp(shifted).sum(axis=1)) - logits.max(axis=1)
    return probabilities, order, max_probability, margin, energy


def calibrate_thresholds(validation: dict[str, Any]) -> dict[str, float | int]:
    probabilities, _, max_probability, margin, energy = scores(validation["logits"])
    del probabilities
    msp_threshold = float(np.quantile(max_probability, 0.05))
    margin_threshold = float(np.quantile(margin, 0.05))
    energy_threshold = float(np.quantile(energy, 0.95))
    return {"max_softmax_min_known": msp_threshold, "margin_min_known": margin_threshold, "energy_max_known": energy_threshold, "validation_count": int(len(energy)), "validation_known_recall_at_threshold": 0.95}


def write_split_manifests(splits: dict[str, list[str]], rows_by_split: dict[str, list[dict[str, Any]]]) -> str:
    fields = ["sample_id", "trajectory", "family", "segment_index", "label", "label_id", "start_frame", "end_frame_exclusive", "duration_frames"]
    hashes: list[str] = []
    for name in ("train", "validation", "test"):
        path = OUTPUT_ROOT / "split_manifests" / f"{name}.csv"
        write_csv(path, rows_by_split[name], fields)
        hashes.append(sha256_file(path))
    payload = "\n".join(f"{name}:{','.join(splits[name])}" for name in ("train", "validation", "test")) + "\n" + "\n".join(hashes)
    return sha256_bytes(payload.encode())


def plot_confusion(matrix: np.ndarray) -> None:
    fig, axis = plt.subplots(figsize=(9, 8)); image = axis.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=axis, fraction=0.046)
    axis.set_xticks(range(NUM_CLASSES), CANONICAL_LABELS, rotation=45, ha="right"); axis.set_yticks(range(NUM_CLASSES), CANONICAL_LABELS); axis.set_xlabel("predicted"); axis.set_ylabel("ground truth"); axis.set_title("Round 12 test confusion matrix")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if matrix[i, j]: axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "confusion_matrix.png", dpi=160); plt.close(fig)


def plot_projection(outputs: dict[str, Any]) -> None:
    arrays = [outputs[name]["embeddings"] for name in ("train", "validation", "test")]
    all_embeddings = np.concatenate(arrays); centered = all_embeddings - all_embeddings.mean(axis=0, keepdims=True); _, _, vt = np.linalg.svd(centered, full_matrices=False); projected = centered @ vt[:2].T
    offsets = {}; cursor = 0
    for name, array in zip(("train", "validation", "test"), arrays): offsets[name] = projected[cursor:cursor + len(array)]; cursor += len(array)
    fig, axis = plt.subplots(figsize=(10, 7)); colors = {"pp": "#1f77b4", "wipe": "#2ca02c", "pour": "#d62728", "plug": "#9467bd"}
    for split in ("train", "validation", "test"):
        rows = outputs[split]["rows"]
        for family in ("pp", "wipe", "pour", "plug"):
            indices = [i for i, row in enumerate(rows) if row["family"] == family]
            if indices: axis.scatter(offsets[split][indices, 0], offsets[split][indices, 1], s=18, alpha=0.45 if split != "test" else 0.8, color=colors[family], label=f"{split}/{family}")
    axis.set_title("Diagnostic 2D projection of segment embeddings"); axis.set_xlabel("PC1"); axis.set_ylabel("PC2"); axis.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "figures/embedding_projection.png", dpi=160); plt.close(fig)


def plot_place_analysis(test: dict[str, Any]) -> None:
    rows = test["rows"]; predictions = test["logits"].argmax(axis=1); families = ("pp", "plug"); fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, family in zip(axes, families):
        indices = [i for i, row in enumerate(rows) if row["family"] == family and row["label"] == "place"]
        counts = Counter(CANONICAL_LABELS[int(predictions[i])] for i in indices)
        axis.bar(CANONICAL_LABELS, [counts[name] for name in CANONICAL_LABELS], color="#9467bd"); axis.set_title(f"{family.upper()} ground-truth place"); axis.tick_params(axis="x", rotation=60); axis.set_ylabel("segments")
    fig.suptitle("PP-place vs Plug-place classifier predictions"); fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "figures/place_family_analysis.png", dpi=160); plt.close(fig)


def main() -> int:
    seed_everything(SEED)
    for directory in (OUTPUT_ROOT / "model", OUTPUT_ROOT / "split_manifests", OUTPUT_ROOT / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    usable = valid_audit_rows()
    family_by_path = {row["trajectory"]: row["task"] for row in usable}
    splits = split_trajectories(usable)
    rows_by_split = {name: build_segment_rows(entries, family_by_path) for name, entries in splits.items()}
    dataset_manifest_hash = write_split_manifests(splits, rows_by_split)
    feature_cache = {trajectory: load_trajectory_features(trajectory) for trajectory in sorted(set(sum(splits.values(), [])))}
    train_frames = np.concatenate([feature_cache[row["trajectory"]][1][row["start_frame"]:row["end_frame_exclusive"]] for row in rows_by_split["train"]], axis=0)
    feature_mean = train_frames.mean(axis=0); feature_std = train_frames.std(axis=0); feature_std = np.maximum(feature_std, 1e-6)
    train_durations = np.asarray([np.log1p(row["duration_frames"]) for row in rows_by_split["train"]], dtype=np.float64)
    duration_mean = float(train_durations.mean()); duration_std = float(max(train_durations.std(), 1e-6))
    datasets = {name: SegmentDataset(rows_by_split[name], feature_cache, feature_mean, feature_std, duration_mean, duration_std) for name in rows_by_split}
    train_counts = Counter(row["label_id"] for row in rows_by_split["train"]); weights = np.asarray([1.0 / np.sqrt(train_counts.get(i, 1)) for i in range(NUM_CLASSES)], dtype=np.float32); weights *= NUM_CLASSES / weights.sum(); class_weights = torch.tensor(weights, dtype=torch.float32)
    sampler = WeightedRandomSampler(torch.tensor([weights[row["label_id"]] for row in rows_by_split["train"]], dtype=torch.double), num_samples=len(rows_by_split["train"]), replacement=True, generator=torch.Generator().manual_seed(SEED))
    loaders = {"train": DataLoader(datasets["train"], batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, collate_fn=collate_segments), "validation": DataLoader(datasets["validation"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_segments), "test": DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_segments)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegmentClassifier(FEATURE_DIM, HIDDEN_DIM, PROJECTION_DIM, EMBEDDING_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    architecture = {"input_shape": "[T,12]", "feature_dim": FEATURE_DIM, "input_projection": "Conv1d(12,96,1)", "tcn": [{"channels": HIDDEN_DIM, "kernel_size": 3, "dilation": 1}, {"channels": HIDDEN_DIM, "kernel_size": 3, "dilation": 2}], "pooling": ["start_mean", "middle_mean", "end_mean", "global_mean", "global_max"], "duration_feature": "z_normalized_log1p_duration_frames", "projection_mlp": [HIDDEN_DIM * 5 + 1, PROJECTION_DIM, EMBEDDING_DIM], "classifier_mlp": [EMBEDDING_DIM, EMBEDDING_DIM // 2, NUM_CLASSES], "embedding_dim": EMBEDDING_DIM, "l2_normalized": True, "masking": "valid-frame mask excludes padding from all pooling"}
    config = {"ontology_version": ONTOLOGY_VERSION, "seed": SEED, "device": str(device), "feature_columns": list(FEATURE_COLUMNS), "feature_dim": FEATURE_DIM, "class_names": list(CANONICAL_LABELS), "aliases": ALIASES, "train_trajectories": splits["train"], "validation_trajectories": splits["validation"], "test_trajectories": splits["test"], "architecture": architecture, "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}, "loss": {"class_balanced_cross_entropy": "inverse_sqrt_train_segment_frequency", "contrastive_weight": CONTRASTIVE_WEIGHT, "temperature": TEMPERATURE}, "early_stopping": {"metric": "validation_macro_f1", "patience": PATIENCE, "max_epochs": MAX_EPOCHS}, "dataset_manifest_hash": dataset_manifest_hash, "git_commit": git_commit(), "feature_normalization": {"mean": feature_mean.tolist(), "std": feature_std.tolist(), "duration_mean": duration_mean, "duration_std": duration_std}}
    (OUTPUT_ROOT / "model/config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (OUTPUT_ROOT / "model/ontology_v2.json").write_text(json.dumps({**ontology_metadata(), "ordered_class_list": list(CANONICAL_LABELS), "feature_dim": FEATURE_DIM}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    history: list[dict[str, Any]] = []; best_f1 = -1.0; best_epoch = 0; stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_model(model, loaders["train"], device, optimizer, class_weights.to(device)); validation_metrics = run_model(model, loaders["validation"], device, None, class_weights.to(device))
        history.extend([{ "epoch": epoch, "split": "train", **{key: train_metrics[key] for key in ("loss", "cross_entropy", "contrastive_loss", "accuracy", "macro_f1")}}, {"epoch": epoch, "split": "validation", **{key: validation_metrics[key] for key in ("loss", "cross_entropy", "contrastive_loss", "accuracy", "macro_f1")}}])
        payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, "best_validation_macro_f1": max(best_f1, validation_metrics["macro_f1"]), "ontology_metadata": {**ontology_metadata(), "ordered_class_list": list(CANONICAL_LABELS), "feature_dim": FEATURE_DIM, "dataset_manifest_hash": dataset_manifest_hash, "git_commit": config["git_commit"], "architecture": architecture}, "architecture_config": architecture, "config": config, "feature_mean": torch.tensor(feature_mean), "feature_std": torch.tensor(feature_std), "duration_mean": duration_mean, "duration_std": duration_std}
        save_checkpoint(OUTPUT_ROOT / "model/last.pt", payload)
        if validation_metrics["macro_f1"] > best_f1 + 1e-9:
            best_f1 = validation_metrics["macro_f1"]; best_epoch = epoch; stale = 0; save_checkpoint(OUTPUT_ROOT / "model/best.pt", payload)
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    write_csv(OUTPUT_ROOT / "training_history.csv", history, ["epoch", "split", "loss", "cross_entropy", "contrastive_loss", "accuracy", "macro_f1"])
    best_payload = torch.load(OUTPUT_ROOT / "model/best.pt", map_location=device, weights_only=False); model.load_state_dict(best_payload["model_state"])
    validation_outputs = collect_outputs(model, loaders["validation"], device); test_outputs = collect_outputs(model, loaders["test"], device)
    thresholds = calibrate_thresholds(validation_outputs)
    validation_labels = np.asarray([row["label_id"] for row in validation_outputs["rows"]]); validation_predictions = validation_outputs["logits"].argmax(axis=1); validation_metrics = metric_bundle(validation_labels, validation_predictions); validation_metrics["open_set_thresholds"] = thresholds; validation_metrics["best_epoch"] = best_epoch
    test_labels = np.asarray([row["label_id"] for row in test_outputs["rows"]]); test_predictions = test_outputs["logits"].argmax(axis=1); test_metrics = metric_bundle(test_labels, test_predictions); probabilities, order, max_probability, margin, energy = scores(test_outputs["logits"])
    def distribution(values: np.ndarray) -> dict[str, float]:
        quantiles = np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        return {key: float(value) for key, value in zip(("min", "p10", "p25", "median", "p75", "p90", "max"), quantiles)}
    test_metrics["open_set"] = {"primary_score": "energy", "thresholds_calibrated_on_validation_only": thresholds, "known_test_energy_rejection_rate": float((energy > thresholds["energy_max_known"]).mean()), "unknown_detection_claim": False, "max_softmax_distribution": distribution(max_probability), "energy_distribution": distribution(energy), "margin_distribution": distribution(margin)}
    test_metrics["per_family"] = {}
    for family in ("pp", "wipe", "pour", "plug"):
        indices = [i for i, row in enumerate(test_outputs["rows"]) if row["family"] == family]; test_metrics["per_family"][family] = metric_bundle(test_labels[indices], test_predictions[indices], macro_over_supported=True)
    test_metrics["place_family"] = {}
    for family in ("pp", "plug"):
        indices = [i for i, row in enumerate(test_outputs["rows"]) if row["family"] == family and row["label"] == "place"]
        test_metrics["place_family"][family] = {"support": len(indices), "accuracy": float((test_predictions[indices] == LABEL_TO_ID["place"]).mean()) if indices else 0.0, "predicted_class_counts": dict(Counter(CANONICAL_LABELS[int(test_predictions[i])] for i in indices))}
    (OUTPUT_ROOT / "validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"); (OUTPUT_ROOT / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    per_class_rows = [{"class": name, **test_metrics["per_class"][name]} for name in CANONICAL_LABELS]; write_csv(OUTPUT_ROOT / "per_class_metrics.csv", per_class_rows, ["class", "support", "precision", "recall", "f1"])
    per_family_rows = [{"family": family, **{key: value for key, value in test_metrics["per_family"][family].items() if key in ("count", "accuracy", "macro_f1")}} for family in ("pp", "wipe", "pour", "plug")]; write_csv(OUTPUT_ROOT / "per_family_metrics.csv", per_family_rows, ["family", "count", "accuracy", "macro_f1"])
    prediction_rows: list[dict[str, Any]] = []
    for i, row in enumerate(test_outputs["rows"]):
        top1, top2 = int(order[i, 0]), int(order[i, 1]); prediction_rows.append({**row, "predicted_label": CANONICAL_LABELS[top1], "predicted_label_id": top1, "max_softmax": float(max_probability[i]), "top1_probability": float(probabilities[i, top1]), "top2_probability": float(probabilities[i, top2]), "top1_top2_margin": float(margin[i]), "energy": float(energy[i]), "energy_known_by_validation_threshold": int(energy[i] <= thresholds["energy_max_known"]), "correct": int(top1 == row["label_id"])})
    write_csv(OUTPUT_ROOT / "segment_predictions.csv", prediction_rows, list(prediction_rows[0]))
    plot_confusion(np.asarray(test_metrics["confusion_matrix"])); plot_projection({"train": collect_outputs(model, DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_segments), device), "validation": validation_outputs, "test": test_outputs}); plot_place_analysis(test_outputs)
    hashes = {name: {"path": str(OUTPUT_ROOT / "model" / name), "sha256": sha256_file(OUTPUT_ROOT / "model" / name), "bytes": (OUTPUT_ROOT / "model" / name).stat().st_size} for name in ("best.pt", "last.pt", "config.yaml", "ontology_v2.json")}; (OUTPUT_ROOT / "model/checkpoint_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = ["# Round 12 segment-level temporal classifier", "", "## Protocol", "", "A fresh classifier was trained from random initialization using GT segments only. No Round 11 checkpoint, classifier weight, optimizer state, or prototype bank was read. Test trajectories were evaluated only after validation-macro-F1 early stopping and validation-only open-set threshold calibration.", "", f"- Ontology: `{ONTOLOGY_VERSION}`; classes: {NUM_CLASSES}.", f"- Best epoch: `{best_epoch}`; validation macro F1: `{best_f1:.6f}`.", f"- Dataset-manifest hash: `{dataset_manifest_hash}`.", f"- Train/validation/test segment counts: `{len(rows_by_split['train'])}` / `{len(rows_by_split['validation'])}` / `{len(rows_by_split['test'])}`.", "", "## Results", "", f"Overall test accuracy: **{test_metrics['accuracy']:.6f}**", f"Overall test macro F1: **{test_metrics['macro_f1']:.6f}**", "", "| family | segments | accuracy | macro F1 |", "|---|---:|---:|---:|"]
    report.extend(f"| {family} | {test_metrics['per_family'][family]['count']} | {test_metrics['per_family'][family]['accuracy']:.6f} | {test_metrics['per_family'][family]['macro_f1']:.6f} |" for family in ("pp", "wipe", "pour", "plug"))
    report.extend(["", "## Required analyses", "", f"- Transport vs wipe: transport F1={test_metrics['per_class']['transport']['f1']:.6f}; wipe F1={test_metrics['per_class']['wipe']['f1']:.6f}. The full confusion matrix is in `confusion_matrix.png`.", f"- Pour vs pour_recover: pour F1={test_metrics['per_class']['pour']['f1']:.6f}; pour_recover F1={test_metrics['per_class']['pour_recover']['f1']:.6f}.", f"- Place vs insert: place F1={test_metrics['per_class']['place']['f1']:.6f}; insert F1={test_metrics['per_class']['insert']['f1']:.6f}.", f"- PP-place: support={test_metrics['place_family']['pp']['support']}, classifier place accuracy={test_metrics['place_family']['pp']['accuracy']:.6f}; Plug-place: support={test_metrics['place_family']['plug']['support']}, classifier place accuracy={test_metrics['place_family']['plug']['accuracy']:.6f}. Both use canonical class ID 6; prediction breakdowns are in `test_metrics.json` and `figures/place_family_analysis.png`.", "", "## Open-set score", "", f"Energy is the primary rejection score with validation-only known threshold `{thresholds['energy_max_known']:.6f}`. Maximum-softmax and top-1/top-2 margin thresholds are diagnostics. Validation contains known labels only, so no unknown-detection performance claim is made.", f"Known test segments rejected by the frozen energy threshold: `{test_metrics['open_set']['known_test_energy_rejection_rate']:.6f}`; this is not unknown recall.", "", "## Conclusions", "", "1. The classifier recognizes the trained ontology to the extent shown by the overall and per-family metrics above.", "2. Transport/wipe confusion is quantified by the class metrics and confusion matrix; it is not inferred from open-set rejection.", "3. Pour/pour_recover and place/insert separability are reported independently above.", "4. PP-place and Plug-place share one class ID; their family-specific behavior is shown separately.", "5. The fresh classifier and validation-only energy calibration provide a suitable basis for later open-set evaluation, but unknown performance requires held-out unknown data.", "", "Diagnostic embedding projection and place-family figures are not used for model selection."])
    (OUTPUT_ROOT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "best_epoch": best_epoch, "test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"], "output": str(OUTPUT_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
