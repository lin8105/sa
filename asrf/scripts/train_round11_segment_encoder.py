#!/usr/bin/env python3
"""Train the round-11 PP-only GT-segment encoder.

This script consumes only the round-11 train and validation PP manifests.  It
does not read test PP or wipe rows, and it does not implement prototypes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from asrf.data.ontology import metadata_for_task  # noqa: E402
DATA_DIR = REPO_ROOT / "outputs/round11_segment_embedding/data"
MODEL_DIR = REPO_ROOT / "outputs/round11_segment_embedding/model"
FEATURE_DIM = 12
HIDDEN_DIM = 64
PROJECTION_DIM = 256
EMBEDDING_DIM = 128
CONTRASTIVE_WEIGHT = 0.1
TEMPERATURE = 0.07
CLASS_NAMES = ("reach", "grasp", "lift", "transport", "place", "release")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def assert_pp_only(rows: list[dict[str, str]], split: str) -> None:
    expected_prefix = "train/pick and place/pp"
    for row in rows:
        trajectory = row["trajectory"]
        if not trajectory.startswith(expected_prefix):
            raise ValueError(f"{split}: non-PP trajectory found: {trajectory}")
        number = trajectory.removeprefix(expected_prefix)
        if not number.isdigit():
            raise ValueError(f"{split}: unexpected trajectory ID: {trajectory}")
        value = int(number)
        if split == "train" and not 1 <= value <= 10:
            raise ValueError(f"train: trajectory outside pp1-pp10: {trajectory}")
        if split == "validation" and not 11 <= value <= 20:
            raise ValueError(f"validation: trajectory outside pp11-pp20: {trajectory}")
        if row.get("known_or_novel") != "known":
            raise ValueError(f"{split}: unexpected known_or_novel value: {row}")
        if row.get("label") not in CLASS_NAMES:
            raise ValueError(f"{split}: label outside PP ontology: {row.get('label')!r}")
        if not row.get("label_id", "").isdigit():
            raise ValueError(f"{split}: missing label_id: {row['sample_id']}")


def load_sequence(row: dict[str, str], feature_mean: np.ndarray, feature_std: np.ndarray, data_dir: Path) -> np.ndarray:
    path = data_dir / row["frame_feature_path"]
    with np.load(path, allow_pickle=False) as archive:
        sequence = np.asarray(archive["features"], dtype=np.float32)
    expected_length = int(row["num_frames"])
    if sequence.shape != (expected_length, FEATURE_DIM):
        raise ValueError(f"{row['sample_id']}: expected {(expected_length, FEATURE_DIM)}, got {sequence.shape}")
    if not np.isfinite(sequence).all():
        raise ValueError(f"{row['sample_id']}: non-finite sequence")
    return ((sequence - feature_mean) / feature_std).astype(np.float32, copy=False)


class SegmentDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, str]], data_dir: Path, feature_mean: np.ndarray, feature_std: np.ndarray, duration_mean: float, duration_std: float) -> None:
        self.rows = rows
        self.data_dir = data_dir
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.duration_mean = duration_mean
        self.duration_std = duration_std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sequence = load_sequence(row, self.feature_mean, self.feature_std, self.data_dir)
        raw_duration = float(row["duration_frames"])
        log_duration = float(np.log1p(raw_duration))
        normalized_duration = (log_duration - self.duration_mean) / self.duration_std
        return {
            "sequence": torch.from_numpy(sequence),
            "duration": torch.tensor([normalized_duration], dtype=torch.float32),
            # Test-only labels (for example retreat) have no known ontology
            # ID.  Inference does not use this field; -1 keeps those rows
            # loadable without assigning them a known class.
            "label": torch.tensor(int(row["label_id"]) if row.get("label_id", "").isdigit() else -1, dtype=torch.long),
            "row": row,
        }


def collate_segments(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([item["sequence"].shape[0] for item in items], dtype=torch.long)
    max_length = int(lengths.max())
    batch = torch.zeros((len(items), max_length, FEATURE_DIM), dtype=torch.float32)
    valid_mask = torch.zeros((len(items), max_length), dtype=torch.bool)
    for index, item in enumerate(items):
        length = item["sequence"].shape[0]
        batch[index, :length] = item["sequence"]
        valid_mask[index, :length] = True
    return {
        "sequence": batch,
        "valid_mask": valid_mask,
        "lengths": lengths,
        "duration": torch.stack([item["duration"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
        "rows": [item["row"] for item in items],
    }


class SegmentEncoder(nn.Module):
    """Masked variable-length segment encoder with a six-class head."""

    def __init__(self, feature_dim: int, hidden_dim: int, projection_dim: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.temporal_conv1 = nn.Conv1d(feature_dim, hidden_dim, kernel_size=5, padding=2)
        self.temporal_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        pooled_dim = hidden_dim * 5 + 1
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor, fallback: Tensor) -> Tensor:
        weights = mask.unsqueeze(1).to(values.dtype)
        count = weights.sum(dim=2)
        result = (values * weights).sum(dim=2) / count.clamp_min(1.0)
        return torch.where(count > 0, result, fallback)

    def forward(self, sequence: Tensor, valid_mask: Tensor, lengths: Tensor, duration: Tensor) -> tuple[Tensor, Tensor]:
        mask = valid_mask.unsqueeze(1)
        sequence = sequence.masked_fill(~valid_mask.unsqueeze(2), 0.0)
        values = F.gelu(self.temporal_conv1(sequence.transpose(1, 2)))
        values = values.masked_fill(~mask, 0.0)
        values = F.gelu(self.temporal_conv2(values))
        values = values.masked_fill(~mask, 0.0)
        global_mean = self._masked_mean(values, valid_mask, torch.zeros_like(values[:, :, 0]))
        global_max = values.masked_fill(~mask, torch.finfo(values.dtype).min).amax(dim=2)

        time = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
        first_end = torch.div(lengths + 2, 3, rounding_mode="floor").unsqueeze(1)
        second_end = torch.div(2 * lengths + 2, 3, rounding_mode="floor").unsqueeze(1)
        start_mask = valid_mask & (time < first_end)
        middle_mask = valid_mask & (time >= first_end) & (time < second_end)
        end_mask = valid_mask & (time >= second_end)
        start_mean = self._masked_mean(values, start_mask, global_mean)
        middle_mean = self._masked_mean(values, middle_mask, global_mean)
        end_mean = self._masked_mean(values, end_mask, global_mean)

        pooled = torch.cat((start_mean, middle_mean, end_mean, global_mean, global_max, duration), dim=1)
        embedding = F.normalize(self.projection(pooled), p=2, dim=1, eps=1e-8)
        logits = self.classifier(embedding)
        return embedding, logits


def supervised_contrastive_loss(embeddings: Tensor, labels: Tensor, temperature: float) -> Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    similarities = torch.matmul(embeddings, embeddings.transpose(0, 1)) / temperature
    diagonal = torch.eye(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~diagonal
    logits = similarities.masked_fill(diagonal, torch.finfo(similarities.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive_mask.sum(dim=1)
    per_anchor = -(log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_count.clamp_min(1))
    usable = positive_count > 0
    if not usable.any():
        return embeddings.sum() * 0.0
    return per_anchor[usable].mean()


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def macro_f1(matrix: np.ndarray) -> float:
    scores: list[float] = []
    for index in range(matrix.shape[0]):
        true_positive = float(matrix[index, index])
        precision_denominator = float(matrix[:, index].sum())
        recall_denominator = float(matrix[index, :].sum())
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def run_epoch(model: SegmentEncoder, loader: DataLoader[dict[str, Any]], optimizer: torch.optim.Optimizer | None, device: torch.device, contrastive_weight: float, temperature: float) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_ce = total_contrastive = 0.0
    total_items = 0
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        embeddings, logits = model(batch["sequence"], batch["valid_mask"], batch["lengths"], batch["duration"])
        ce = F.cross_entropy(logits, batch["label"])
        contrastive = supervised_contrastive_loss(embeddings, batch["label"], temperature)
        loss = ce + contrastive_weight * contrastive
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        count = int(batch["label"].shape[0])
        total_items += count
        total_loss += float(loss.detach().cpu()) * count
        total_ce += float(ce.detach().cpu()) * count
        total_contrastive += float(contrastive.detach().cpu()) * count
        truths.append(batch["label"].detach().cpu().numpy())
        predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    matrix = confusion_matrix(truth, prediction, len(CLASS_NAMES))
    return {
        "loss": total_loss / total_items,
        "cross_entropy": total_ce / total_items,
        "supervised_contrastive": total_contrastive / total_items,
        "accuracy": float(np.mean(truth == prediction)),
        "macro_f1": macro_f1(matrix),
        "truth": truth,
        "prediction": prediction,
        "confusion_matrix": matrix,
    }


def collect_embeddings(model: SegmentEncoder, loader: DataLoader[dict[str, Any]], device: torch.device) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output_rows: list[dict[str, str]] = []
    embeddings: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            moved = batch_to_device(batch, device)
            embedding, logits = model(moved["sequence"], moved["valid_mask"], moved["lengths"], moved["duration"])
            probabilities = logits.softmax(dim=1)
            prediction = logits.argmax(dim=1).cpu().numpy()
            confidence = probabilities.max(dim=1).values.cpu().numpy()
            embedding_np = embedding.cpu().numpy()
            label_np = moved["label"].cpu().numpy()
            for index, row in enumerate(batch["rows"]):
                item = {
                    "sample_id": row["sample_id"], "trajectory": row["trajectory"], "segment_index": row["segment_index"],
                    "label": row["label"], "label_id": row["label_id"], "predicted_label": CLASS_NAMES[int(prediction[index])],
                    "predicted_label_id": str(int(prediction[index])), "confidence": f"{float(confidence[index]):.9f}",
                    "start_frame": row["start_frame"], "end_frame_exclusive": row["end_frame_exclusive"],
                    "duration_frames": row["duration_frames"], "split": row["split"], "known_or_novel": row["known_or_novel"],
                }
                for dimension, value in enumerate(embedding_np[index]):
                    item[f"embedding_{dimension:03d}"] = f"{float(value):.9f}"
                output_rows.append(item)
            embeddings.append(embedding_np)
            truths.append(label_np)
            predictions.append(prediction)
    return output_rows, np.concatenate(embeddings), np.concatenate(truths), np.concatenate(predictions)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_embeddings(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *CLASS_NAMES])
        for index, label in enumerate(CLASS_NAMES):
            writer.writerow([label, *matrix[index].tolist()])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    train_rows = read_manifest(data_dir / "train_manifest.csv")
    validation_rows = read_manifest(data_dir / "validation_manifest.csv")
    assert_pp_only(train_rows, "train")
    assert_pp_only(validation_rows, "validation")
    train_trajectories = sorted({row["trajectory"] for row in train_rows})
    validation_trajectories = sorted({row["trajectory"] for row in validation_rows})
    if set(train_trajectories) & set(validation_trajectories):
        raise ValueError("Train/validation trajectory leakage")
    if len(train_rows) != 60 or len(validation_rows) != 60:
        raise ValueError(f"Unexpected audited PP segment counts: train={len(train_rows)}, validation={len(validation_rows)}")

    # Fit preprocessing statistics on train frames and train segment durations only.
    all_train_features = []
    for row in train_rows:
        with np.load(data_dir / row["frame_feature_path"], allow_pickle=False) as archive:
            all_train_features.append(np.asarray(archive["features"], dtype=np.float64))
    feature_matrix = np.concatenate(all_train_features, axis=0)
    feature_mean = feature_matrix.mean(axis=0).astype(np.float32)
    feature_std = feature_matrix.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    duration_values = np.log1p(np.asarray([float(row["duration_frames"]) for row in train_rows], dtype=np.float64))
    duration_mean = float(duration_values.mean())
    duration_std = float(duration_values.std())
    if duration_std < 1e-6:
        duration_std = 1.0

    train_dataset = SegmentDataset(train_rows, data_dir, feature_mean, feature_std, duration_mean, duration_std)
    validation_dataset = SegmentDataset(validation_rows, data_dir, feature_mean, feature_std, duration_mean, duration_std)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0, collate_fn=collate_segments)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_segments)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    model = SegmentEncoder(FEATURE_DIM, HIDDEN_DIM, PROJECTION_DIM, EMBEDDING_DIM, len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    config = {
        "dataset": {"data_dir": str(data_dir), "train_manifest": "train_manifest.csv", "validation_manifest": "validation_manifest.csv", "train_trajectories": train_trajectories, "validation_trajectories": validation_trajectories, "wipe_used": False},
        "architecture": {"input_shape": "[T,D]", "feature_dim": FEATURE_DIM, "temporal_conv_layers": 2, "conv_kernel_size": 5, "hidden_dim": HIDDEN_DIM, "pooling": ["start_mean", "middle_mean", "end_mean", "global_mean", "global_max"], "duration_feature": "z_normalized_log1p_duration_frames", "projection_mlp": [HIDDEN_DIM * 5 + 1, PROJECTION_DIM, EMBEDDING_DIM], "embedding_dim": EMBEDDING_DIM, "l2_normalized": True, "classifier_classes": list(CLASS_NAMES)},
        "loss": {"classification": "cross_entropy", "supervised_contrastive": True, "supervised_contrastive_weight": CONTRASTIVE_WEIGHT, "temperature": TEMPERATURE},
        "training": {"seed": args.seed, "epochs_max": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "early_stopping_metric": "validation_macro_f1", "patience": args.patience, "min_delta": args.min_delta, "device": str(device), "prototypes_trained": False},
        "preprocessing": {"feature_mean_train_only": feature_mean.tolist(), "feature_std_train_only": feature_std.tolist(), "duration_log_mean_train_only": duration_mean, "duration_log_std_train_only": duration_std},
    }
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, CONTRASTIVE_WEIGHT, TEMPERATURE)
        with torch.no_grad():
            validation_metrics = run_epoch(model, validation_loader, None, device, CONTRASTIVE_WEIGHT, TEMPERATURE)
        current_score = validation_metrics["macro_f1"]
        improved = current_score > best_score + args.min_delta
        if improved:
            best_score = current_score
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                "best_validation_macro_f1": best_score, "class_names": list(CLASS_NAMES), "config": config, "ontology_metadata": metadata_for_task(CLASS_NAMES),
                "feature_mean": feature_mean, "feature_std": feature_std,
                "duration_log_mean": duration_mean, "duration_log_std": duration_std,
            }
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            stale_epochs += 1
        history.append({
            "epoch": epoch, "train_loss": train_metrics["loss"], "train_cross_entropy": train_metrics["cross_entropy"], "train_supervised_contrastive": train_metrics["supervised_contrastive"],
            "train_accuracy": train_metrics["accuracy"], "train_macro_f1": train_metrics["macro_f1"], "validation_loss": validation_metrics["loss"], "validation_cross_entropy": validation_metrics["cross_entropy"], "validation_supervised_contrastive": validation_metrics["supervised_contrastive"], "validation_accuracy": validation_metrics["accuracy"], "validation_macro_f1": current_score, "learning_rate": optimizer.param_groups[0]["lr"], "improved": int(improved), "stale_epochs": stale_epochs,
        })
        print(f"epoch={epoch:03d} train_loss={train_metrics['loss']:.5f} val_macro_f1={current_score:.5f} val_acc={validation_metrics['accuracy']:.5f}{' *' if improved else ''}")
        if stale_epochs >= args.patience:
            break
    write_history(output_dir / "training_history.csv", history)

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_validation = run_epoch(model, validation_loader, None, device, CONTRASTIVE_WEIGHT, TEMPERATURE)
    matrix = final_validation["confusion_matrix"]
    write_confusion(output_dir / "validation_confusion_matrix.csv", matrix)
    np.save(output_dir / "validation_confusion_matrix.npy", matrix)
    embedding_rows, embeddings, truths, predictions = collect_embeddings(model, validation_loader, device)
    write_embeddings(output_dir / "validation_embedding_table.csv", embedding_rows)
    np.save(output_dir / "validation_embeddings.npy", embeddings)
    np.save(output_dir / "validation_embedding_labels.npy", truths)
    np.save(output_dir / "validation_embedding_predictions.npy", predictions)
    hashes = {
        "best.pt": {"sha256": sha256_file(output_dir / "best.pt"), "bytes": (output_dir / "best.pt").stat().st_size},
        "config.yaml": {"sha256": sha256_file(output_dir / "config.yaml"), "bytes": (output_dir / "config.yaml").stat().st_size},
        "training_history.csv": {"sha256": sha256_file(output_dir / "training_history.csv"), "bytes": (output_dir / "training_history.csv").stat().st_size},
        "validation_confusion_matrix.csv": {"sha256": sha256_file(output_dir / "validation_confusion_matrix.csv"), "bytes": (output_dir / "validation_confusion_matrix.csv").stat().st_size},
        "validation_embedding_table.csv": {"sha256": sha256_file(output_dir / "validation_embedding_table.csv"), "bytes": (output_dir / "validation_embedding_table.csv").stat().st_size},
        "best_epoch": best_epoch, "best_validation_macro_f1": float(checkpoint["best_validation_macro_f1"]),
    }
    (output_dir / "checkpoint_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_validation_macro_f1": float(checkpoint["best_validation_macro_f1"]), "validation_accuracy": final_validation["accuracy"], "validation_confusion_matrix": matrix.tolist(), "output_dir": str(output_dir), "device": str(device)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
