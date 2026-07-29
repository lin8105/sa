#!/usr/bin/env python3
"""Round 14 hard synthetic outlier exposure with wipe strictly held out."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_ROOT = ROOT / "outputs/round12_open_set_holdout_wipe"
ROUND13_ROOT = ROOT / "outputs/round13_open_set_outlier_exposure_holdout_wipe"
OUTPUT_ROOT = ROOT / "outputs/round14_open_set_hard_oe_holdout_wipe"
INIT_CHECKPOINT = ROUND13_ROOT / "model/energy_margin_best.pt"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import train_round12_segment_classifier as base  # noqa: E402
from asrf.data.ontology import CANONICAL_LABELS, ONTOLOGY_VERSION  # noqa: E402
from asrf.training.checkpointing import sha256_file  # noqa: E402

SEED = 42
HELD_OUT = "wipe"
KNOWN_CLASSES = tuple(label for label in CANONICAL_LABELS if label != HELD_OUT)
ID_TO_LABEL = {index: label for index, label in enumerate(KNOWN_CLASSES)}
LABEL_TO_ID = {label: index for index, label in ID_TO_LABEL.items()}
FEATURE_COLUMNS = tuple(base.FEATURE_COLUMNS)
FEATURE_DIM = base.FEATURE_DIM
ORIGINAL_TYPES = ("cross_boundary", "mixed_skill_concatenation", "temporal_shuffle", "invalid_duration_crop")
HARD_TYPES = ("same_class_temporal_corruption", "boundary_extension", "feature_channel_mismatch", "local_temporal_splice", "duration_preserving_mixed", "embedding_interpolation")
ALL_TYPES = ORIGINAL_TYPES + HARD_TYPES
SOURCE_CLASS_COUNT = len(KNOWN_CLASSES)
TRAIN_PER_TYPE = 30
VALIDATION_PER_TYPE_CLASS = 10
BATCH_SIZE = 32
MAX_EPOCHS = 18
PATIENCE = 4
LAMBDA_ENERGY = 0.1
ENERGY_MARGIN = 5.0
EMBEDDING_LAMBDAS = (0.02, 0.05, 0.1)
EMBEDDING_MARGINS = (0.02, 0.04)
UNIFORM_ALPHAS = (0.02, 0.05, 0.1)
STABILITY_LAMBDAS = (0.01, 0.05, 0.1)
COMBINED_ENERGY_WEIGHTS = (0.25, 0.5, 0.75)
DEVICE = torch.device("cpu")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return 0.0
    return float((positive[:, None] > negative[None, :]).mean() + 0.5 * (positive[:, None] == negative[None, :]).mean())


def load_rows() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        values = []
        for raw in read_csv(HOLDOUT_ROOT / "split_manifests" / f"{split}.csv"):
            row = dict(raw)
            for field in ("segment_index", "label_id", "start_frame", "end_frame_exclusive", "duration_frames"):
                row[field] = int(row[field])
            values.append(row)
        result[split] = values
    if any(row["label"] == HELD_OUT for row in result["train"] + result["validation"]):
        raise RuntimeError("Wipe is present in train or validation.")
    return result


def load_features(rows: dict[str, list[dict[str, Any]]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    paths = sorted({row["trajectory"] for values in rows.values() for row in values})
    return {path: base.load_trajectory_features(path) for path in paths}


def sequence(row: dict[str, Any], cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return cache[row["trajectory"]][1][row["start_frame"]:row["end_frame_exclusive"]].astype(np.float32)


def runs(rows: list[dict[str, Any]]) -> dict[str, list[list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["trajectory"]].append(row)
    result = {}
    for trajectory, values in grouped.items():
        current: list[dict[str, Any]] = []
        result[trajectory] = []
        for row in sorted(values, key=lambda item: item["start_frame"]):
            if current and row["start_frame"] != current[-1]["end_frame_exclusive"]:
                result[trajectory].append(current)
                current = []
            current.append(row)
        if current:
            result[trajectory].append(current)
    return result


def item(values: np.ndarray, kind: str, sources: list[dict[str, Any]], sample_id: str, source_label: str | None = None, embedding: np.ndarray | None = None) -> dict[str, Any]:
    return {"sequence": values.astype(np.float32), "duration_frames": int(len(values)), "label_id": -1, "label": "synthetic_outlier", "outlier_type": kind, "sample_id": sample_id, "source_rows": sources, "source_label": source_label or sources[0]["label"], "embedding_override": embedding}


def grouped_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[row["label"]].append(row)
    return result


def choose_rows(rows: list[dict[str, Any]], count: int, offset: int = 0) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError("Cannot sample synthetic source from an empty class.")
    return [rows[(offset + index) % len(rows)] for index in range(count)]


def make_original(rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], split: str, count: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + (0 if split == "train" else 1000))
    by_label = grouped_by_label(rows)
    output: list[dict[str, Any]] = []
    boundary_rows = []
    for trajectory, run_list in runs(rows).items():
        for run in run_list:
            for left, right in zip(run[:-1], run[1:]):
                if left["label"] != right["label"]:
                    boundary_rows.append((left, right))
    for index in range(count):
        kind = ORIGINAL_TYPES[index % len(ORIGINAL_TYPES)]
        source = choose_rows(list(rows), 1, index)[0]
        if kind == "cross_boundary" and boundary_rows:
            left, right = boundary_rows[index % len(boundary_rows)]
            fraction = (0.35, 0.55)[index % 2]
            boundary = left["end_frame_exclusive"]
            left_take = max(1, int(left["duration_frames"] * fraction))
            right_take = max(1, int(right["duration_frames"] * fraction))
            values = cache[left["trajectory"]][1][boundary - left_take:boundary + right_take]
            output.append(item(values, kind, [left, right], f"{split}/{kind}/{index}", left["label"]))
        elif kind == "mixed_skill_concatenation":
            second = rows[int(rng.integers(0, len(rows)))]
            while second["label"] == source["label"]:
                second = rows[int(rng.integers(0, len(rows)))]
            first_values = sequence(source, cache); second_values = sequence(second, cache)
            cut = max(1, int(len(first_values) * rng.uniform(.25, .75)))
            start = min(len(second_values) - 1, int(len(second_values) * rng.uniform(.1, .55)))
            output.append(item(np.concatenate((first_values[:cut], second_values[start:])), kind, [source, second], f"{split}/{kind}/{index}", source["label"]))
        elif kind == "temporal_shuffle":
            values = sequence(source, cache)
            chunks = int(rng.integers(3, 6))
            bounds = np.linspace(0, len(values), chunks + 1, dtype=int)
            parts = [values[bounds[i]:bounds[i + 1]] for i in range(chunks)]
            order = rng.permutation(chunks)
            if np.array_equal(order, np.arange(chunks)):
                order = np.roll(order, 1)
            output.append(item(np.concatenate([parts[int(i)] for i in order]), kind, [source], f"{split}/{kind}/{index}", source["label"]))
        else:
            values = sequence(source, cache)
            end = max(1, int(len(values) * rng.uniform(.2, .45)))
            output.append(item(values[:end], kind, [source], f"{split}/{kind}/{index}", source["label"]))
    return output


def make_hard(rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], split: str, per_type: int, embedding_pairs: list[tuple[np.ndarray, dict[str, Any], dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 200 + (0 if split == "train" else 1000))
    by_label = grouped_by_label(rows)
    output: list[dict[str, Any]] = []
    for kind_index, kind in enumerate(HARD_TYPES):
        selected = choose_rows(list(rows), per_type, kind_index * 7)
        for index, source in enumerate(selected):
            values = sequence(source, cache)
            source_label = source["label"]
            if kind == "same_class_temporal_corruption":
                chunks = int(rng.integers(4, 9)); bounds = np.linspace(0, len(values), chunks + 1, dtype=int)
                parts = [values[bounds[i]:bounds[i + 1]] for i in range(chunks)]
                touched = rng.choice(chunks, size=2 if chunks > 5 else 1, replace=False)
                for chunk in touched:
                    operation = int(rng.integers(0, 4))
                    if operation == 0 and len(parts[chunk]): parts[chunk] = np.concatenate((parts[chunk], parts[chunk][-1:]))
                    elif operation == 1 and len(parts[chunk]) > 2: parts[chunk] = np.delete(parts[chunk], len(parts[chunk]) // 2, axis=0)
                    elif operation == 2: parts[chunk] = parts[chunk][::-1]
                    else: parts[chunk] = parts[chunk][::-1]
                corrupted = np.concatenate(parts)
                output.append(item(corrupted, kind, [source], f"{split}/{kind}/{index}", source_label))
            elif kind == "boundary_extension":
                adjacent = next((right for run in runs(rows).get(source["trajectory"], []) for left, right in zip(run[:-1], run[1:]) if left["sample_id"] == source["sample_id"]), None)
                if adjacent is None:
                    adjacent = next((left for run in runs(rows).get(source["trajectory"], []) for left, right in zip(run[:-1], run[1:]) if right["sample_id"] == source["sample_id"]), source)
                contamination = (0.1, 0.2, 0.3, 0.4, 0.5)[index % 5]
                take = max(1, int(len(values) * contamination))
                adjacent_values = sequence(adjacent, cache)
                if adjacent["start_frame"] > source["start_frame"]:
                    corrupted = np.concatenate((values[:-take], adjacent_values[:take]))
                else:
                    corrupted = np.concatenate((adjacent_values[-take:], values[take:]))
                output.append(item(corrupted, kind, [source, adjacent], f"{split}/{kind}/{index}", source_label))
            elif kind == "feature_channel_mismatch":
                other = next(row for row in rows if row["label"] != source_label)
                other_values = sequence(other, cache)
                aligned = np.stack([np.interp(np.linspace(0, len(other_values) - 1, len(values)), np.arange(len(other_values)), other_values[:, channel]) for channel in range(FEATURE_DIM)], axis=1)
                corrupted = values.copy(); corrupted[:, -2:] = aligned[:, -2:]
                output.append(item(corrupted, kind, [source, other], f"{split}/{kind}/{index}", source_label))
            elif kind == "local_temporal_splice":
                other = next(row for row in rows if row["label"] != source_label)
                other_values = sequence(other, cache)
                ratio = (0.1, 0.2, 0.3)[index % 3]; length = max(1, int(len(values) * ratio)); center = len(values) // 2
                other_aligned = np.stack([np.interp(np.linspace(0, len(other_values) - 1, length), np.arange(len(other_values)), other_values[:, channel]) for channel in range(FEATURE_DIM)], axis=1)
                corrupted = values.copy(); start = max(0, center - length // 2); corrupted[start:start + length] = other_aligned[:len(corrupted[start:start + length])]
                output.append(item(corrupted, kind, [source, other], f"{split}/{kind}/{index}", source_label))
            elif kind == "duration_preserving_mixed":
                other = next(row for row in rows if row["label"] != source_label)
                other_values = sequence(other, cache); ratio = (0.25, 0.35, 0.45)[index % 3]; cut = max(1, int(len(values) * ratio)); tail = len(values) - cut
                other_aligned = np.stack([np.interp(np.linspace(0, len(other_values) - 1, tail), np.arange(len(other_values)), other_values[:, channel]) for channel in range(FEATURE_DIM)], axis=1)
                output.append(item(np.concatenate((values[:cut], other_aligned)), kind, [source, other], f"{split}/{kind}/{index}", source_label))
            else:
                if embedding_pairs:
                    interpolated, first, second = embedding_pairs[index % len(embedding_pairs)]
                    coefficient = (0.25, 0.5, 0.75)[index % 3]
                    # The pair array stores the two normalized embeddings.
                    first_embedding, second_embedding = interpolated
                    embedding = ((1.0 - coefficient) * first_embedding + coefficient * second_embedding).astype(np.float32)
                    embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
                    output.append(item(sequence(first, cache), kind, [first, second], f"{split}/{kind}/{index}", first["label"], embedding))
                else:
                    output.append(item(values, kind, [source], f"{split}/{kind}/{index}", source_label))
    return output


class SegmentDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]] | None, mean: np.ndarray, std: np.ndarray, duration_mean: float, duration_std: float, known: bool):
        self.items = items; self.cache = cache; self.mean = mean; self.std = std; self.duration_mean = duration_mean; self.duration_std = duration_std; self.known = known

    def __len__(self) -> int: return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.items[index]
        values = sequence(row, self.cache) if self.known else row["sequence"]
        normalized = (values - self.mean) / self.std
        duration = (np.log1p(row["duration_frames"]) - self.duration_mean) / self.duration_std
        override = row.get("embedding_override")
        return {"sequence": torch.from_numpy(normalized.astype(np.float32)), "duration": torch.tensor(duration, dtype=torch.float32), "label": torch.tensor(row["label_id"], dtype=torch.long), "override": torch.from_numpy(override.astype(np.float32)) if override is not None else torch.zeros(base.EMBEDDING_DIM), "override_valid": torch.tensor(override is not None), "row": row}


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["sequence"].shape[0] for item in items], dtype=torch.long); max_length = int(lengths.max())
    values = torch.zeros((len(items), max_length, FEATURE_DIM)); mask = torch.zeros((len(items), max_length), dtype=torch.bool)
    for index, entry in enumerate(items):
        length = int(entry["sequence"].shape[0]); values[index, :length] = entry["sequence"]; mask[index, :length] = True
    return {"sequence": values, "valid_mask": mask, "lengths": lengths, "duration": torch.stack([item["duration"] for item in items]), "label": torch.stack([item["label"] for item in items]), "override": torch.stack([item["override"] for item in items]), "override_valid": torch.stack([item["override_valid"] for item in items]), "rows": [item["row"] for item in items]}


def model_infer(model: nn.Module, loader: DataLoader) -> dict[str, Any]:
    model.eval(); embeddings = []; logits = []; rows = []
    with torch.no_grad():
        for batch in loader:
            embedding, output = model(batch["sequence"], batch["valid_mask"], batch["lengths"], batch["duration"])
            valid = batch["override_valid"]
            if valid.any():
                embedding = embedding.clone(); embedding[valid] = F.normalize(batch["override"][valid], p=2, dim=1)
                output = output.clone(); output[valid] = model.classifier(embedding[valid])
            embeddings.append(embedding.numpy()); logits.append(output.numpy()); rows.extend(batch["rows"])
    return {"embeddings": np.concatenate(embeddings), "logits": np.concatenate(logits), "rows": rows}


def logits_scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True); probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True); order = np.argsort(-probabilities, axis=1)
    return {"probability": probabilities, "top1": order[:, 0], "max_softmax": probabilities[np.arange(len(logits)), order[:, 0]], "energy": -np.log(np.exp(shifted).sum(axis=1)) - logits.max(axis=1), "margin": probabilities[np.arange(len(logits)), order[:, 0]] - probabilities[np.arange(len(logits)), order[:, 1]]}


def closed_f1(outputs: dict[str, Any]) -> float:
    labels = np.asarray([row["label_id"] for row in outputs["rows"]]); predictions = logits_scores(outputs["logits"])["top1"]; values = []
    for label_id in range(len(KNOWN_CLASSES)):
        tp = int(((labels == label_id) & (predictions == label_id)).sum()); fp = int(((labels != label_id) & (predictions == label_id)).sum()); fn = int(((labels == label_id) & (predictions != label_id)).sum()); precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0; values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))


def reference_embeddings(model: nn.Module, rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], mean: np.ndarray, std: np.ndarray, duration_mean: float, duration_std: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    loader = DataLoader(SegmentDataset(rows, cache, mean, std, duration_mean, duration_std, True), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    output = model_infer(model, loader); return output["embeddings"], output["rows"]


def build_embedding_pairs(embeddings: np.ndarray, rows: list[dict[str, Any]]) -> list[tuple[tuple[np.ndarray, np.ndarray], dict[str, Any], dict[str, Any]]]:
    similarities = embeddings @ embeddings.T
    pairs = []
    for first_index in range(len(embeddings)):
        candidates = [second for second in range(len(embeddings)) if rows[second]["label"] != rows[first_index]["label"]]
        second_index = max(candidates, key=lambda index: similarities[first_index, index])
        pairs.append(((embeddings[first_index], embeddings[second_index]), rows[first_index], rows[second_index]))
    return pairs


def cosine_distance(embeddings: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return 1.0 - embeddings @ reference.T


def hardness(outputs: dict[str, Any], reference: np.ndarray) -> np.ndarray:
    scores = logits_scores(outputs["logits"]); distances = cosine_distance(outputs["embeddings"], reference).min(axis=1)
    # Larger hardness means more known-like: low energy, high MSP, low distance.
    def percentile(values: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(values)); return order / max(len(values) - 1, 1)
    return (1.0 - percentile(scores["energy"]) + percentile(scores["max_softmax"]) + (1.0 - percentile(distances))) / 3.0


def mine_pool(model: nn.Module, pool: list[dict[str, Any]], loader_factory: Any, reference: np.ndarray, epoch: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outputs = model_infer(model, loader_factory(pool)); hardness_values = hardness(outputs, reference); rng = np.random.default_rng(SEED + epoch)
    count = len(pool); hard_count = count // 2; hard_indices = np.argsort(-hardness_values)[:hard_count]; remaining = np.asarray([index for index in range(count) if index not in set(hard_indices)], dtype=int); random_indices = rng.choice(remaining, size=count - hard_count, replace=False) if len(remaining) >= count - hard_count else rng.choice(np.arange(count), size=count - hard_count, replace=False)
    indices = np.concatenate((hard_indices, random_indices)); selected = [pool[int(index)] for index in indices]
    records = []
    for kind in ALL_TYPES:
        type_indices = [int(index) for index in indices if pool[int(index)]["outlier_type"] == kind]
        all_indices = [index for index, row in enumerate(pool) if row["outlier_type"] == kind]
        values = hardness_values[all_indices]
        type_scores = logits_scores(outputs["logits"]); type_distances = cosine_distance(outputs["embeddings"], reference).min(axis=1)[all_indices]
        records.append({"epoch": epoch, "outlier_type": kind, "pool_count": len(all_indices), "selected_count": len(type_indices), "selection_rate": len(type_indices) / max(len(all_indices), 1), "hardness_mean": float(values.mean()) if len(values) else 0.0, "hardness_std": float(values.std()) if len(values) else 0.0, "hardness_min": float(values.min()) if len(values) else 0.0, "hardness_max": float(values.max()) if len(values) else 0.0, "energy_mean": float(type_scores["energy"][all_indices].mean()) if len(all_indices) else 0.0, "energy_std": float(type_scores["energy"][all_indices].std()) if len(all_indices) else 0.0, "max_softmax_mean": float(type_scores["max_softmax"][all_indices].mean()) if len(all_indices) else 0.0, "cosine_distance_mean": float(type_distances.mean()) if len(type_distances) else 0.0, "cosine_distance_std": float(type_distances.std()) if len(type_distances) else 0.0})
    return selected, records


def train_one(model: nn.Module, teacher: nn.Module, known_rows: list[dict[str, Any]], outlier_pool: list[dict[str, Any]], validation_loader: DataLoader, outlier_validation_loader: DataLoader, cache: dict[str, tuple[np.ndarray, np.ndarray]], mean: np.ndarray, std: np.ndarray, duration_mean: float, duration_std: float, class_weights: Tensor, reference: np.ndarray, variant: dict[str, Any]) -> tuple[dict[str, Tensor], list[dict[str, Any]], int, list[dict[str, Any]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)
    loader_factory = lambda values: DataLoader(SegmentDataset(values, None if values and values[0].get("sequence") is not None else cache, mean, std, duration_mean, duration_std, False), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    known_loader_factory = lambda: DataLoader(SegmentDataset(known_rows, cache, mean, std, duration_mean, duration_std, True), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED), collate_fn=collate)
    best_state = None; best_f1 = -1.0; best_epoch = 0; stale = 0; history = []; mining_records = []
    teacher.eval()
    for epoch in range(1, MAX_EPOCHS + 1):
        selected, mining = mine_pool(model, outlier_pool, loader_factory, reference, epoch); mining_records.extend({"variant": variant["name"], **record} for record in mining)
        known_loader = known_loader_factory(); outlier_loader = DataLoader(SegmentDataset(selected, None, mean, std, duration_mean, duration_std, False), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED + epoch), collate_fn=collate)
        known_iter = iter(known_loader); outlier_iter = iter(outlier_loader); steps = max(len(known_loader), len(outlier_loader)); totals = Counter()
        model.train()
        for _ in range(steps):
            try: known_batch = next(known_iter)
            except StopIteration: known_iter = iter(known_loader); known_batch = next(known_iter)
            try: outlier_batch = next(outlier_iter)
            except StopIteration: outlier_iter = iter(outlier_loader); outlier_batch = next(outlier_iter)
            known_embedding, known_logits = model(known_batch["sequence"], known_batch["valid_mask"], known_batch["lengths"], known_batch["duration"])
            outlier_embedding, outlier_logits = model(outlier_batch["sequence"], outlier_batch["valid_mask"], outlier_batch["lengths"], outlier_batch["duration"])
            valid = outlier_batch["override_valid"]
            if valid.any():
                outlier_embedding = outlier_embedding.clone(); outlier_embedding[valid] = F.normalize(outlier_batch["override"][valid], p=2, dim=1); outlier_logits = outlier_logits.clone(); outlier_logits[valid] = model.classifier(outlier_embedding[valid])
            ce = F.cross_entropy(known_logits, known_batch["label"], weight=class_weights)
            known_energy = -torch.logsumexp(known_logits, dim=1); outlier_energy = -torch.logsumexp(outlier_logits, dim=1)
            energy_loss = F.relu(ENERGY_MARGIN + known_energy.mean() - outlier_energy.mean())
            uniform_loss = -F.log_softmax(outlier_logits, dim=1).mean()
            nearest_distance = (1.0 - outlier_embedding @ torch.from_numpy(reference).float().T).min(dim=1).values
            embedding_loss = F.relu(float(variant.get("embedding_margin", 0.0)) - nearest_distance).mean()
            with torch.no_grad():
                teacher_embedding, teacher_logits = teacher(known_batch["sequence"], known_batch["valid_mask"], known_batch["lengths"], known_batch["duration"])
            stability = F.kl_div(F.log_softmax(known_logits, dim=1), F.softmax(teacher_logits, dim=1), reduction="batchmean")
            loss = ce + LAMBDA_ENERGY * energy_loss + float(variant.get("uniform_alpha", 0.0)) * uniform_loss + float(variant.get("embedding_lambda", 0.0)) * embedding_loss + float(variant["stability_lambda"]) * stability
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            totals.update({"loss": float(loss.detach()), "ce": float(ce.detach()), "energy": float(energy_loss.detach()), "uniform": float(uniform_loss.detach()), "embedding": float(embedding_loss.detach()), "stability": float(stability.detach())})
        validation = model_infer(model, validation_loader); f1 = closed_f1(validation)
        history.append({"variant": variant["name"], "epoch": epoch, "validation_known_macro_f1": f1, **{f"train_{key}": value / steps for key, value in totals.items()}})
        if f1 > best_f1 + 1e-9:
            best_f1 = f1; best_epoch = epoch; stale = 0; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else: stale += 1
        if stale >= PATIENCE: break
    if best_state is None: raise RuntimeError("No checkpoint state produced.")
    model.load_state_dict(best_state)
    return best_state, history, best_epoch, mining_records


def score_bundle(outputs: dict[str, Any], reference: np.ndarray, normalization: dict[str, tuple[float, float]] | None = None) -> dict[str, np.ndarray]:
    scores = logits_scores(outputs["logits"]); cosine = cosine_distance(outputs["embeddings"], reference).min(axis=1); bundle = {"max_softmax": -scores["max_softmax"], "energy": scores["energy"], "cosine": cosine}
    if normalization is not None:
        for key in ("energy", "cosine"):
            low, high = normalization[key]; bundle[key] = (bundle[key] - low) / max(high - low, 1e-8)
        for weight in COMBINED_ENERGY_WEIGHTS:
            bundle[f"combined_{weight:.2f}"] = weight * bundle["energy"] + (1.0 - weight) * bundle["cosine"]
    return bundle


def threshold_for(known_scores: np.ndarray, unknown_scores: np.ndarray, target: float) -> dict[str, float]:
    # Use an observed score cutoff so the discrete validation set satisfies
    # the retention constraint instead of falling just below it due to linear
    # quantile interpolation.
    ordered = np.sort(known_scores); required = max(1, int(np.ceil(target * len(ordered)))); threshold = float(ordered[min(required - 1, len(ordered) - 1)]); accepted = known_scores <= threshold; rejected_unknown = unknown_scores > threshold; labels = np.concatenate((np.zeros(len(known_scores)), np.ones(len(unknown_scores)))); return {"threshold": threshold, "known_retention": float(accepted.mean()), "synthetic_recall": float(rejected_unknown.mean()), "synthetic_auroc": auroc(labels, np.concatenate((known_scores, unknown_scores)))}


def metric_rejection(outputs: dict[str, Any], accepted: np.ndarray) -> dict[str, Any]:
    scores = logits_scores(outputs["logits"]); labels = np.asarray([row["label_id"] for row in outputs["rows"]]); predictions = scores["top1"]; f1s = []
    per_class = {}
    for label_id, label in enumerate(KNOWN_CLASSES):
        tp = int(((labels == label_id) & (predictions == label_id) & accepted).sum()); fp = int(((labels != label_id) & (predictions == label_id) & accepted).sum()); fn = int(((labels == label_id) & ((predictions != label_id) | ~accepted)).sum()); precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0; f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0; support = int((labels == label_id).sum()); rejected = int(((labels == label_id) & ~accepted).sum()); per_class[label] = {"support": support, "false_rejection_rate": rejected / support if support else 0.0, "f1": f1}; f1s.append(f1)
    return {"known_retention": float(accepted.mean()), "false_rejection_rate": float((~accepted).mean()), "accepted_accuracy": float((predictions[accepted] == labels[accepted]).mean()) if accepted.any() else 0.0, "macro_f1_after_rejection": float(np.mean(f1s)), "per_class": per_class}


def balanced_validation_types(rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], split: str, embedding_pairs: list[tuple[np.ndarray, dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    # Generate from validation trajectories only, then balance each type by
    # the source class. This prevents source-class composition from becoming
    # an accidental validation signal.
    original_pool = make_original(rows, cache, split, VALIDATION_PER_TYPE_CLASS * SOURCE_CLASS_COUNT * len(ORIGINAL_TYPES))
    hard_pool = make_hard(rows, cache, split, VALIDATION_PER_TYPE_CLASS * SOURCE_CLASS_COUNT, embedding_pairs)
    output = []
    for kind in ALL_TYPES:
        pool = original_pool if kind in ORIGINAL_TYPES else hard_pool
        for label in KNOWN_CLASSES:
            candidates = [entry for entry in pool if entry["outlier_type"] == kind and entry["source_label"] == label]
            if len(candidates) < VALIDATION_PER_TYPE_CLASS:
                if candidates:
                    candidates = candidates + choose_rows(candidates, VALIDATION_PER_TYPE_CLASS - len(candidates))
                elif kind in ORIGINAL_TYPES:
                    source_rows = grouped_by_label(rows)[label]
                    other = next(row for row in rows if row["label"] != label)
                    for index in range(VALIDATION_PER_TYPE_CLASS):
                        source = source_rows[index % len(source_rows)]; values = sequence(source, cache); other_values = sequence(other, cache)
                        if kind in ("cross_boundary", "mixed_skill_concatenation"):
                            take = max(1, len(values) // 2); values = np.concatenate((values[-take:], other_values[:take]))
                        elif kind == "temporal_shuffle":
                            values = values[::-1]
                        else:
                            values = values[:max(1, len(values) // 3)]
                        candidates.append(item(values, kind, [source, other], f"{split}/{kind}/fallback/{label}/{index}", label))
                else:
                    class_rows = grouped_by_label(rows)[label]
                    if kind == "embedding_interpolation":
                        pairs = [pair for pair in embedding_pairs if pair[1]["label"] == label]
                        for index in range(VALIDATION_PER_TYPE_CLASS):
                            pair = pairs[index % len(pairs)]; coefficient = (0.25, 0.5, 0.75)[index % 3]; first_embedding, second_embedding = pair[0]; embedding = ((1.0 - coefficient) * first_embedding + coefficient * second_embedding).astype(np.float32); embedding /= max(float(np.linalg.norm(embedding)), 1e-8); candidates.append(item(sequence(pair[1], cache), kind, [pair[1], pair[2]], f"{split}/{kind}/fallback/{label}/{index}", label, embedding))
                    else:
                        other_global = next(row for row in rows if row["label"] != label)
                        candidates = [entry for entry in make_hard(class_rows + [other_global], cache, split, VALIDATION_PER_TYPE_CLASS, embedding_pairs) if entry["outlier_type"] == kind and entry["source_label"] == label]
                        if len(candidates) < VALIDATION_PER_TYPE_CLASS and candidates:
                            candidates = candidates + choose_rows(candidates, VALIDATION_PER_TYPE_CLASS - len(candidates))
            if len(candidates) < VALIDATION_PER_TYPE_CLASS:
                raise RuntimeError(f"Validation type {kind} lacks source class {label} samples.")
            output.extend(candidates[:VALIDATION_PER_TYPE_CLASS])
    return output


def model_from_state(state: dict[str, Tensor]) -> nn.Module:
    model = base.SegmentClassifier(FEATURE_DIM, base.HIDDEN_DIM, base.PROJECTION_DIM, base.EMBEDDING_DIM, len(KNOWN_CLASSES)); model.load_state_dict(state); return model


def manifest_hash() -> str:
    digest = hashlib.sha256()
    for split in ("train", "validation", "test"):
        digest.update((HOLDOUT_ROOT / "split_manifests" / f"{split}.csv").read_bytes())
    return digest.hexdigest()


def main() -> int:
    seed_everything(); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = torch.load(INIT_CHECKPOINT, map_location="cpu", weights_only=False); init_hash = sha256_file(INIT_CHECKPOINT)
    if payload.get("held_out_class") != HELD_OUT or payload.get("known_class_list") != list(KNOWN_CLASSES): raise RuntimeError("Initialization checkpoint metadata mismatch.")
    rows = load_rows(); cache = load_features(rows)
    if any(row["label"] == HELD_OUT for row in rows["train"] + rows["validation"]): raise RuntimeError("Wipe leakage in train/validation.")
    train_frames = np.concatenate([sequence(row, cache) for row in rows["train"]]); mean = train_frames.mean(axis=0); std = np.maximum(train_frames.std(axis=0), 1e-6); log_durations = np.asarray([np.log1p(row["duration_frames"]) for row in rows["train"]]); duration_mean = float(log_durations.mean()); duration_std = float(max(log_durations.std(), 1e-6))
    teacher = model_from_state(payload["model_state"]); teacher.eval(); reference, reference_rows = reference_embeddings(teacher, rows["train"], cache, mean, std, duration_mean, duration_std); validation_reference, validation_reference_rows = reference_embeddings(teacher, rows["validation"], cache, mean, std, duration_mean, duration_std)
    # Training interpolation uses known training embeddings. Validation
    # interpolation uses validation embeddings so no train trajectory enters
    # the synthetic validation set.
    embedding_pairs = build_embedding_pairs(reference, rows["train"]); validation_embedding_pairs = build_embedding_pairs(validation_reference, rows["validation"])
    synthetic_train = make_original(rows["train"], cache, "train", TRAIN_PER_TYPE * len(ORIGINAL_TYPES)) + make_hard(rows["train"], cache, "train", TRAIN_PER_TYPE, embedding_pairs)
    synthetic_validation = balanced_validation_types(rows["validation"], cache, "validation", validation_embedding_pairs)
    if any(row["label"] == HELD_OUT for row in rows["train"] + rows["validation"]): raise RuntimeError("Wipe appeared during synthetic setup.")
    known_validation_loader = DataLoader(SegmentDataset(rows["validation"], cache, mean, std, duration_mean, duration_std, True), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    synthetic_validation_loader = DataLoader(SegmentDataset(synthetic_validation, None, mean, std, duration_mean, duration_std, False), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    class_counts = Counter(row["label_id"] for row in rows["train"]); class_weights = torch.tensor([1.0 / np.sqrt(class_counts[index]) for index in range(len(KNOWN_CLASSES))], dtype=torch.float32); class_weights *= len(KNOWN_CLASSES) / class_weights.sum()
    variants = [{"name": f"energy_margin_stability_{stability}", "stability_lambda": stability} for stability in STABILITY_LAMBDAS]
    variants += [{"name": f"energy_uniform_{alpha}_stability_0.05", "uniform_alpha": alpha, "stability_lambda": 0.05} for alpha in UNIFORM_ALPHAS]
    variants += [{"name": f"energy_embedding_{lam}_margin_{margin}_stability_0.05", "embedding_lambda": lam, "embedding_margin": margin, "stability_lambda": 0.05} for lam in EMBEDDING_LAMBDAS for margin in EMBEDDING_MARGINS]
    model_selection = []; histories = []; mining_records = []; saved_states = {}; validation_outputs = {}; validation_normalizations = {}
    for run_index, variant in enumerate(variants):
        print(f"[round14] run {run_index + 1}/{len(variants)} {variant['name']}", flush=True)
        model = model_from_state(payload["model_state"]); state, history, epoch, mining = train_one(model, teacher, rows["train"], synthetic_train, known_validation_loader, synthetic_validation_loader, cache, mean, std, duration_mean, duration_std, class_weights, reference, variant)
        model = model_from_state(state); known_output = model_infer(model, known_validation_loader); synthetic_output = model_infer(model, synthetic_validation_loader); raw_known = score_bundle(known_output, reference); raw_synthetic = score_bundle(synthetic_output, reference); combined_values = {key: np.concatenate((raw_known[key], raw_synthetic[key])) for key in ("energy", "cosine")}; normalization = {key: (float(values.min()), float(values.max())) for key, values in combined_values.items()}; scores = score_bundle(known_output, reference, normalization); synthetic_scores = score_bundle(synthetic_output, reference, normalization); selection_row = {"run_index": run_index, **variant, "best_epoch": epoch, "validation_known_macro_f1_closed": closed_f1(known_output)}
        for score_name in scores:
            if score_name.startswith("combined") or score_name in ("max_softmax", "energy", "cosine"):
                metric = threshold_for(scores[score_name], synthetic_scores[score_name], 0.95); selection_row[f"{score_name}_auroc"] = metric["synthetic_auroc"]; selection_row[f"{score_name}_retention"] = metric["known_retention"]
        model_selection.append(selection_row); histories.extend({"run_index": run_index, **row} for row in history); mining_records.extend(mining); saved_states[variant["name"]] = state; validation_outputs[variant["name"]] = (known_output, synthetic_output); validation_normalizations[variant["name"]] = normalization
        print(f"[round14] completed {variant['name']} epoch={epoch} f1={selection_row['validation_known_macro_f1_closed']:.4f}", flush=True)
    # Validation-only score/model selection with the requested retention rule.
    candidates = []
    for row in model_selection:
        name = row["name"]
        known_output, synthetic_output = validation_outputs[name]; score_values = score_bundle(known_output, reference, validation_normalizations[name]); synthetic_values = score_bundle(synthetic_output, reference, validation_normalizations[name])
        for score_name in score_values:
            metric = threshold_for(score_values[score_name], synthetic_values[score_name], 0.95); diagnostic = threshold_for(score_values[score_name], synthetic_values[score_name], 0.97); known_rejection = metric_rejection(known_output, score_values[score_name] <= metric["threshold"])
            candidates.append({"model": name, "score": score_name, "threshold": metric["threshold"], "known_validation_retention": metric["known_retention"], "known_validation_macro_f1_after_rejection": known_rejection["macro_f1_after_rejection"], "synthetic_validation_recall": metric["synthetic_recall"], "synthetic_validation_auroc": metric["synthetic_auroc"], "threshold_at_0.97": diagnostic["threshold"], "retention_at_0.97": diagnostic["known_retention"], "synthetic_recall_at_0.97": diagnostic["synthetic_recall"], "eligible_0.95": int(metric["known_retention"] >= 0.95), "normalization": json.dumps(validation_normalizations[name], sort_keys=True)})
    eligible = [row for row in candidates if row["eligible_0.95"]]
    if eligible:
        best_auroc = max(row["synthetic_validation_auroc"] for row in eligible); close = [row for row in eligible if best_auroc - row["synthetic_validation_auroc"] <= 0.005]; selected = max(close, key=lambda row: (row["known_validation_macro_f1_after_rejection"], row["known_validation_retention"], row["model"], row["score"]))
        selection_rule = "eligible_retention_then_auroc_within_0.005_f1_tiebreak"
    else:
        highest_retention = max(row["known_validation_retention"] for row in candidates); close = [row for row in candidates if abs(row["known_validation_retention"] - highest_retention) < 1e-9]; selected = max(close, key=lambda row: (row["synthetic_validation_auroc"], row["known_validation_macro_f1_after_rejection"])); selection_rule = "highest_attainable_retention_then_auroc"
    selected_model_name = selected["model"]; selected_state = saved_states[selected_model_name]; selected_model = model_from_state(selected_state); selected_known, selected_synthetic = validation_outputs[selected_model_name]; selected_score = selected["score"]; selected_normalization = validation_normalizations[selected_model_name]; frozen = {"primary": selected, "selection_rule": selection_rule, "score_candidates": candidates, "normalization": selected_normalization}
    (OUTPUT_ROOT / "frozen_thresholds.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save({"model_state": selected_state, "ontology_version": ONTOLOGY_VERSION, "held_out_class": HELD_OUT, "known_class_list": list(KNOWN_CLASSES), "metadata": {"initialization_checkpoint": str(INIT_CHECKPOINT), "initialization_checkpoint_sha256": init_hash, "optimizer_state_reused": False, "selected_model": selected, "selection_rule": selection_rule}, "optimizer_state": None}, OUTPUT_ROOT / "selected_model.pt")
    (OUTPUT_ROOT / "initialization_checkpoint_sha256.txt").write_text(init_hash + "\n", encoding="utf-8")
    write_csv(OUTPUT_ROOT / "model_selection.csv", model_selection); write_csv(OUTPUT_ROOT / "validation_selection_details.csv", candidates); write_csv(OUTPUT_ROOT / "training_history.csv", histories); write_csv(OUTPUT_ROOT / "synthetic_outlier_statistics.csv", mining_records)
    # Validation diagnostics for the selected model, by original/hard type and source class.
    selected_synthetic_scores = score_bundle(selected_synthetic, reference, selected_normalization); selected_known_scores = score_bundle(selected_known, reference, selected_normalization); selected_threshold = float(selected["threshold"]); type_metrics = []
    for category, types in (("original", ORIGINAL_TYPES), ("hard", HARD_TYPES), ("all", ALL_TYPES)):
        for kind in types if category != "all" else ("all",):
            indices = np.arange(len(synthetic_validation)) if kind == "all" else np.asarray([i for i, row in enumerate(synthetic_validation) if row["outlier_type"] == kind])
            known_values = selected_known_scores[selected_score]; synthetic_values = selected_synthetic_scores[selected_score][indices]; type_metrics.append({"model": selected_model_name, "score": selected_score, "category": category, "outlier_type": kind, "source_class": "all", "count": len(indices), "synthetic_validation_recall": float((synthetic_values > selected_threshold).mean()), "synthetic_validation_auroc": auroc(np.concatenate((np.zeros(len(selected_known_scores[selected_score])), np.ones(len(synthetic_values)))), np.concatenate((known_values, synthetic_values)))})
    for label in KNOWN_CLASSES:
        indices = np.asarray([i for i, row in enumerate(synthetic_validation) if row["source_label"] == label]); values = selected_synthetic_scores[selected_score][indices]; type_metrics.append({"model": selected_model_name, "score": selected_score, "category": "all", "outlier_type": "all", "source_class": label, "count": len(indices), "synthetic_validation_recall": float((values > selected_threshold).mean()), "synthetic_validation_auroc": auroc(np.concatenate((np.zeros(len(selected_known_scores[selected_score])), np.ones(len(values)))), np.concatenate((selected_known_scores[selected_score], values)))})
    write_csv(OUTPUT_ROOT / "synthetic_outlier_type_metrics.csv", type_metrics)
    # Final evaluation starts only after the primary model, score and threshold are frozen.
    known_test_rows = [row for row in rows["test"] if row["evaluation_group"] == "known_test"]; wipe_rows = [row for row in rows["test"] if row["evaluation_group"] == "wipe_unknown"]; inside_rows = [row for row in rows["test"] if row["evaluation_group"] == "known_inside_wipe"]
    test_outputs = {}
    for group, group_rows in (("known_test", known_test_rows), ("wipe", wipe_rows), ("known_inside_wipe", inside_rows)):
        loader = DataLoader(SegmentDataset(group_rows, cache, mean, std, duration_mean, duration_std, True), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate); test_outputs[group] = model_infer(selected_model, loader)
    test_scores = {group: score_bundle(output, reference, selected_normalization) for group, output in test_outputs.items()}; threshold = selected_threshold
    prediction_rows = []; wipe_diagnostics = []
    for group, output in test_outputs.items():
        scores = test_scores[group]; accepted = scores[selected_score] <= threshold; logits = logits_scores(output["logits"]); distances = cosine_distance(output["embeddings"], reference); nearest = distances.argmin(axis=1)
        for index, row in enumerate(output["rows"]):
            record = {"group": group, "sample_id": row["sample_id"], "trajectory": row["trajectory"], "segment_index": row["segment_index"], "ground_truth_label": row["label"], "duration_frames": row["duration_frames"], "predicted_known_class": ID_TO_LABEL[int(logits["top1"][index])], "max_softmax": float(logits["max_softmax"][index]), "energy": float(logits["energy"][index]), "cosine_novelty": float(scores["cosine"][index]), "combined_novelty": float(scores[selected_score][index]) if selected_score.startswith("combined") else "", "score_name": selected_score, "threshold": threshold, "accepted_as_known": int(accepted[index]), "decision": ID_TO_LABEL[int(logits["top1"][index])] if accepted[index] else "unknown", "nearest_known_training_segment": reference_rows[int(nearest[index])]["sample_id"], "nearest_known_class": reference_rows[int(nearest[index])]["label"], "nearest_known_cosine_distance": float(distances[index, nearest[index]])}
            prediction_rows.append(record)
            if group == "wipe": wipe_diagnostics.append(record)
    write_csv(OUTPUT_ROOT / "segment_predictions.csv", prediction_rows); write_csv(OUTPUT_ROOT / "wipe_diagnostics.csv", wipe_diagnostics)
    known_metric = metric_rejection(test_outputs["known_test"], np.asarray([row["accepted_as_known"] for row in prediction_rows if row["group"] == "known_test"], dtype=bool)); inside_metric = metric_rejection(test_outputs["known_inside_wipe"], np.asarray([row["accepted_as_known"] for row in prediction_rows if row["group"] == "known_inside_wipe"], dtype=bool)); wipe_accept = np.asarray([row["accepted_as_known"] for row in wipe_diagnostics], dtype=bool)
    known_score_values = test_scores["known_test"][selected_score]; wipe_score_values = test_scores["wipe"][selected_score]; known_wipe_auroc = auroc(np.concatenate((np.zeros(len(known_score_values)), np.ones(len(wipe_score_values)))), np.concatenate((known_score_values, wipe_score_values)))
    variant_results = [{"method": "round14_selected_hard_oe", "known_retention": known_metric["known_retention"], "known_false_rejection_rate": known_metric["false_rejection_rate"], "known_macro_f1_before_rejection": closed_f1(test_outputs["known_test"]), "known_macro_f1_after_rejection": known_metric["macro_f1_after_rejection"], "known_vs_wipe_auroc": known_wipe_auroc, "wipe_unknown_recall": float((~wipe_accept).mean()), "wipe_false_known_rate": float(wipe_accept.mean()), "known_inside_wipe_accuracy": float((logits_scores(test_outputs["known_inside_wipe"]["logits"])["top1"] == np.asarray([row["label_id"] for row in test_outputs["known_inside_wipe"]["rows"]])).mean()), "known_inside_wipe_retention": inside_metric["known_retention"]}]
    # Preserve prior frozen comparisons and add them only after the new threshold is frozen.
    prior = []
    for path in (ROUND13_ROOT / "baseline_comparison.csv",):
        prior.extend(read_csv(path))
    baseline_rows = variant_results + [{"method": row["method"], "known_retention": row["known_retention"], "known_false_rejection_rate": row["known_false_unknown_rate"], "known_macro_f1_before_rejection": row["known_macro_f1_before_rejection"], "known_macro_f1_after_rejection": row["known_macro_f1_after_rejection"], "known_vs_wipe_auroc": row["known_vs_wipe_auroc"], "wipe_unknown_recall": row["wipe_unknown_recall"], "wipe_false_known_rate": row["wipe_false_known_rate"], "known_inside_wipe_accuracy": row["known_inside_wipe_closed_set_accuracy"], "known_inside_wipe_retention": row["known_inside_wipe_retention"]} for row in prior if row["method"] in ("max_softmax", "energy", "frozen_cosine_knn_k1", "oe_uniform_softmax", "oe_energy_margin")]
    write_csv(OUTPUT_ROOT / "hard_oe_variant_comparison.csv", variant_results); write_csv(OUTPUT_ROOT / "baseline_comparison.csv", baseline_rows)
    false_rejection_rows = []
    for label in KNOWN_CLASSES:
        label_rows = [row for row in prediction_rows if row["group"] == "known_test" and row["ground_truth_label"] == label]; false_rejection_rows.append({"class": label, "support": len(label_rows), "false_rejection_count": sum(not bool(row["accepted_as_known"]) for row in label_rows), "false_rejection_rate": sum(not bool(row["accepted_as_known"]) for row in label_rows) / max(len(label_rows), 1)})
    write_csv(OUTPUT_ROOT / "known_class_false_rejection.csv", false_rejection_rows)
    # Failure analysis is diagnostic and happens after the one frozen wipe evaluation.
    failure_rows = []
    train_feature_stats = train_frames.std(axis=0)
    for record in wipe_diagnostics:
        if record["decision"] == "unknown": continue
        wipe_seq = cache[record["trajectory"]][1][next(row for row in wipe_rows if row["sample_id"] == record["sample_id"])["start_frame"]:next(row for row in wipe_rows if row["sample_id"] == record["sample_id"])["end_frame_exclusive"]]
        nearest_row = next(row for row in rows["train"] if row["sample_id"] == record["nearest_known_training_segment"]); train_seq = sequence(nearest_row, cache)
        grid = np.linspace(0.0, 1.0, 32); wipe_resampled = np.stack([np.interp(grid, np.linspace(0.0, 1.0, len(wipe_seq)), wipe_seq[:, channel]) for channel in range(FEATURE_DIM)], axis=1); train_resampled = np.stack([np.interp(grid, np.linspace(0.0, 1.0, len(train_seq)), train_seq[:, channel]) for channel in range(FEATURE_DIM)], axis=1); differences = np.abs(wipe_resampled - train_resampled).mean(axis=0) / train_feature_stats; similar = np.argsort(differences)[:5]; contributors = np.argsort(-differences)[:5]; force_wipe = wipe_seq[:, :10]; force_train = train_seq[:, :10]; grip_wipe = wipe_seq[:, -2:]; grip_train = train_seq[:, -2:]
        failure_rows.append({"sample_id": record["sample_id"], "trajectory": record["trajectory"], "segment_index": record["segment_index"], "predicted_class": record["predicted_known_class"], "nearest_training_segment": record["nearest_known_training_segment"], "nearest_class": record["nearest_known_class"], "wipe_duration": next(row["duration_frames"] for row in wipe_rows if row["sample_id"] == record["sample_id"]), "nearest_duration": nearest_row["duration_frames"], "duration_ratio": next(row["duration_frames"] for row in wipe_rows if row["sample_id"] == record["sample_id"]) / max(nearest_row["duration_frames"], 1), "most_similar_channels": json.dumps([FEATURE_COLUMNS[int(index)] for index in similar]), "largest_difference_channels": json.dumps([FEATURE_COLUMNS[int(index)] for index in contributors]), "force_mean_abs_difference": float(np.abs(force_wipe.mean(axis=0) - force_train.mean(axis=0)).mean()), "force_max_abs_difference": float(np.abs(force_wipe.max(axis=0) - force_train.max(axis=0)).mean()), "torque_mean_abs_difference": float(np.abs(force_wipe.mean(axis=0)[1:3] - force_train.mean(axis=0)[1:3]).mean()), "gripper_mean_abs_difference": float(np.abs(grip_wipe.mean(axis=0) - grip_train.mean(axis=0)).mean()), "gripper_max_abs_difference": float(np.abs(grip_wipe.max(axis=0) - grip_train.max(axis=0)).mean()), "diagnostic_hypotheses": "representation_overlap;gripper_similarity;threshold_calibration"})
    write_csv(OUTPUT_ROOT / "remaining_failure_analysis.csv", failure_rows)
    config = {"experiment": "round14_open_set_hard_oe_holdout_wipe", "seed": SEED, "ontology_version": ONTOLOGY_VERSION, "held_out_class": HELD_OUT, "known_classes": list(KNOWN_CLASSES), "initialization_checkpoint": str(INIT_CHECKPOINT), "initialization_checkpoint_sha256": init_hash, "optimizer_state_reused": False, "wipe_train_segments": 0, "wipe_validation_segments": 0, "manifest_hash": manifest_hash(), "outlier_types_original": list(ORIGINAL_TYPES), "outlier_types_hard": list(HARD_TYPES), "variants": variants, "score_grid": ["max_softmax", "energy", "cosine", *[f"combined_{weight:.2f}" for weight in COMBINED_ENERGY_WEIGHTS]], "selected_model": selected_model_name, "selected_score": selected_score, "selection_rule": selection_rule, "wipe_used_during_selection": False, "threshold_target": 0.95}
    (OUTPUT_ROOT / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = ["# Round 14 hard outlier exposure: wipe holdout", "", "## Protocol", "", f"Wipe was absent from train/validation, synthetic generation, online mining, model selection, score selection, and threshold selection. Initialization used `{INIT_CHECKPOINT}` with SHA-256 `{init_hash}`. A fresh optimizer was used; no optimizer state was reused. Test evaluation began only after the model, score, and threshold were frozen.", "", "## Round 13 selection-rule audit", "", "Round 13 selected by `validation_synthetic_auroc_best`, then `validation_known_macro_f1`, then `validation_known_accuracy`. It did not enforce a validation-retention constraint during model selection. Uniform-softmax won because its best validation synthetic AUROC was 0.928694 versus 0.922836 for energy-margin. Under the requested >=0.95 retention constraint, both frozen Round 13 objective winners retained 0.955224 known validation segments, so uniform-softmax would still win on AUROC; energy-margin would not be selected.", "", "## Round 14 selection", "", f"- Rule: {selection_rule}", f"- Selected model: {selected_model_name}", f"- Selected score: {selected_score}", f"- Frozen threshold: {selected_threshold:.9f}", f"- Validation known retention: {selected['known_validation_retention']:.6f}", f"- Validation synthetic recall: {selected['synthetic_validation_recall']:.6f}", f"- Validation synthetic AUROC: {selected['synthetic_validation_auroc']:.6f}", "", "## Final comparison", "", "| method | known retention | known F1 before | known F1 after | known-vs-wipe AUROC | wipe unknown recall | wipe false-known | inside-wipe accuracy | inside-wipe retention |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in baseline_rows: report.append(f"| {row['method']} | {float(row['known_retention']):.6f} | {float(row['known_macro_f1_before_rejection']):.6f} | {float(row['known_macro_f1_after_rejection']):.6f} | {float(row['known_vs_wipe_auroc']):.6f} | {float(row['wipe_unknown_recall']):.6f} | {float(row['wipe_false_known_rate']):.6f} | {float(row['known_inside_wipe_accuracy']):.6f} | {float(row['known_inside_wipe_retention']):.6f} |")
    report += ["", "## Hard-outlier diagnostics", "", "`synthetic_outlier_type_metrics.csv` reports original and hard types separately, combined, and by source class. `synthetic_outlier_statistics.csv` records per-type online hard-mining acceptance and score distributions. `remaining_failure_analysis.csv` is diagnostic only and was generated after threshold freezing.", "", "## Result", "", f"Round 14 primary wipe unknown recall was {variant_results[0]['wipe_unknown_recall']:.6f} with known retention {variant_results[0]['known_retention']:.6f}. The Round 13 energy-margin model remains preferred unless Round 14 improves both rejection and retention targets; no success claim is made from wipe recall alone.", "", "## Integrity", "", "Annotations were not modified. Wipe train/validation counts are zero. No wipe-derived statistic entered selection. Selected checkpoint optimizer state is empty. Relevant tests, full pytest, compileall, and git diff --check are recorded in the final handoff."]
    (OUTPUT_ROOT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "selected_model": selected_model_name, "selected_score": selected_score, "threshold": selected_threshold, "primary": variant_results[0], "output": str(OUTPUT_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
