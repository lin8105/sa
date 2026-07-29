#!/usr/bin/env python3
"""Round 13 synthetic outlier-exposure experiment with wipe held out."""

from __future__ import annotations

import csv
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
OUTPUT_ROOT = ROOT / "outputs/round13_open_set_outlier_exposure_holdout_wipe"
CHECKPOINT = HOLDOUT_ROOT / "model/best.pt"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import train_round12_segment_classifier as base  # noqa: E402
from asrf.data.ontology import CANONICAL_LABELS, ONTOLOGY_VERSION  # noqa: E402
from asrf.training.checkpointing import sha256_file  # noqa: E402

SEED = 42
HELD_OUT = "wipe"
KNOWN_CLASSES = tuple(label for label in CANONICAL_LABELS if label != HELD_OUT)
ID_TO_LABEL = {index: label for index, label in enumerate(KNOWN_CLASSES)}
OUTLIER_TYPES = ("cross_boundary", "mixed_skill_concatenation", "temporal_shuffle", "invalid_duration_crop")
LAMBDAS = (0.05, 0.1, 0.2, 0.5)
MARGINS = (1.0, 2.0, 5.0)
BATCH_SIZE = 32
MAX_EPOCHS = 25
PATIENCE = 5
DEVICE = torch.device("cpu")
MAX_SYNTHETIC_PER_TYPE = {"train": 100, "validation": 50}


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # The experiment is CPU-bound on short segment batches; a single thread
    # avoids oversubscription across the fixed hyperparameter grid.
    torch.set_num_threads(1)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return 0.0
    return float((positive[:, None] > negative[None, :]).mean() + 0.5 * (positive[:, None] == negative[None, :]).mean())


def score_logits(logits: np.ndarray) -> dict[str, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    order = np.argsort(-probability, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    return {
        "probability": probability,
        "top1": top1,
        "max_softmax": probability[np.arange(len(logits)), top1],
        "energy": -np.log(np.exp(shifted).sum(axis=1)) - logits.max(axis=1),
        "margin": probability[np.arange(len(logits)), top1] - probability[np.arange(len(logits)), top2],
    }


def load_rows() -> dict[str, list[dict[str, Any]]]:
    output = {}
    for split in ("train", "validation", "test"):
        values = []
        for raw in read_csv(HOLDOUT_ROOT / "split_manifests" / f"{split}.csv"):
            row = dict(raw)
            for field in ("segment_index", "label_id", "start_frame", "end_frame_exclusive", "duration_frames"):
                row[field] = int(row[field])
            values.append(row)
        output[split] = values
    if any(row["label"] == HELD_OUT for row in output["train"] + output["validation"]):
        raise RuntimeError("Wipe is present in the training or validation manifests.")
    return output


def load_features(rows: dict[str, list[dict[str, Any]]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    trajectories = sorted({row["trajectory"] for values in rows.values() for row in values})
    return {trajectory: base.load_trajectory_features(trajectory) for trajectory in trajectories}


def sequence(row: dict[str, Any], cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return cache[row["trajectory"]][1][row["start_frame"]:row["end_frame_exclusive"]].astype(np.float32)


def known_runs(rows: list[dict[str, Any]]) -> dict[str, list[list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["trajectory"]].append(row)
    result = {}
    for trajectory, values in grouped.items():
        runs = []
        current = []
        for row in sorted(values, key=lambda value: value["start_frame"]):
            if current and row["start_frame"] != current[-1]["end_frame_exclusive"]:
                runs.append(current)
                current = []
            current.append(row)
        if current:
            runs.append(current)
        result[trajectory] = runs
    return result


def synthetic_item(values: np.ndarray, kind: str, sources: list[dict[str, Any]], sample_id: str) -> dict[str, Any]:
    return {
        "sequence": values.astype(np.float32),
        "duration_frames": int(len(values)),
        "label_id": -1,
        "label": "synthetic_outlier",
        "outlier_type": kind,
        "sample_id": sample_id,
        "source_rows": sources,
    }


def make_cross_boundary(rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], split: str) -> list[dict[str, Any]]:
    output = []
    for trajectory, runs in known_runs(rows).items():
        for run_index, run in enumerate(runs):
            for left, right in zip(run[:-1], run[1:]):
                if left["label"] == right["label"]:
                    continue
                boundary = left["end_frame_exclusive"]
                for variant, fraction in enumerate((0.35, 0.55)):
                    left_take = max(1, int(left["duration_frames"] * fraction))
                    right_take = max(1, int(right["duration_frames"] * fraction))
                    values = cache[trajectory][1][boundary - left_take:boundary + right_take]
                    output.append(synthetic_item(values, "cross_boundary", [left, right], f"{split}/cross_boundary/{trajectory}/{run_index}/{variant}"))
    return output


def make_mixed(rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], split: str) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + (0 if split == "train" else 1000))
    output = []
    for index, first in enumerate(rows):
        second = rows[int(rng.integers(0, len(rows)))]
        while second["label"] == first["label"]:
            second = rows[int(rng.integers(0, len(rows)))]
        first_values = sequence(first, cache)
        second_values = sequence(second, cache)
        first_end = max(1, int(len(first_values) * rng.uniform(.25, .75)))
        second_start = min(len(second_values) - 1, int(len(second_values) * rng.uniform(.1, .55)))
        output.append(synthetic_item(np.concatenate((first_values[:first_end], second_values[second_start:])), "mixed_skill_concatenation", [first, second], f"{split}/mixed/{index}"))
    return output


def make_shuffle(rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], split: str) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 17 + (0 if split == "train" else 1000))
    output = []
    for index, row in enumerate(rows):
        values = sequence(row, cache)
        if len(values) < 6:
            continue
        chunks = int(rng.integers(3, 6))
        boundaries = np.linspace(0, len(values), chunks + 1, dtype=int)
        parts = [values[boundaries[i]:boundaries[i + 1]] for i in range(chunks)]
        order = rng.permutation(chunks)
        if np.array_equal(order, np.arange(chunks)):
            order = np.roll(order, 1)
        output.append(synthetic_item(np.concatenate([parts[int(index)] for index in order]), "temporal_shuffle", [row], f"{split}/shuffle/{index}"))
    return output


def make_invalid_duration(rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], split: str) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 31 + (0 if split == "train" else 1000))
    medians = {}
    for label in set(row["label"] for row in rows):
        medians[label] = float(np.median([row["duration_frames"] for row in rows if row["label"] == label]))
    output = []
    for index, row in enumerate(rows):
        values = sequence(row, cache)
        short_end = max(1, int(len(values) * rng.uniform(.2, .45)))
        output.append(synthetic_item(values[:short_end], "invalid_duration_crop", [row], f"{split}/invalid_short/{index}"))
    runs = known_runs(rows)
    for index, row in enumerate(rows):
        matching = next((run for run_list in runs.values() for run in run_list if any(item["sample_id"] == row["sample_id"] for item in run)), None)
        if matching is None:
            continue
        target = int(medians[row["label"]] * rng.uniform(1.4, 2.0))
        start_run = matching[0]["start_frame"]
        end_run = matching[-1]["end_frame_exclusive"]
        if end_run - start_run < target:
            continue
        center = (row["start_frame"] + row["end_frame_exclusive"]) // 2
        start = max(start_run, min(center - target // 2, end_run - target))
        values = cache[row["trajectory"]][1][start:start + target]
        if len(values) > row["duration_frames"]:
            output.append(synthetic_item(values, "invalid_duration_crop", matching, f"{split}/invalid_long/{index}"))
    return output


def generate_outliers(rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], split: str) -> list[dict[str, Any]]:
    output = []
    output.extend(make_cross_boundary(rows, cache, split))
    output.extend(make_mixed(rows, cache, split))
    output.extend(make_shuffle(rows, cache, split))
    output.extend(make_invalid_duration(rows, cache, split))
    # Keep the fixed grid tractable while preserving every construction. The
    # cap is applied independently and deterministically per type; it does
    # not affect known data or any held-out evaluation data.
    cap = MAX_SYNTHETIC_PER_TYPE[split]
    selected = []
    for type_index, kind in enumerate(OUTLIER_TYPES):
        values = [item for item in output if item["outlier_type"] == kind]
        rng = np.random.default_rng(SEED + (0 if split == "train" else 1000) + type_index * 101)
        if len(values) > cap:
            values = [values[index] for index in sorted(rng.choice(len(values), size=cap, replace=False))]
        selected.extend(values)
    output = selected
    if set(item["outlier_type"] for item in output) != set(OUTLIER_TYPES):
        raise RuntimeError(f"{split}: missing one or more synthetic outlier types.")
    return output


class SequenceDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]] | None, mean: np.ndarray, std: np.ndarray, duration_mean: float, duration_std: float, known: bool):
        self.items = items
        self.cache = cache
        self.mean = mean
        self.std = std
        self.duration_mean = duration_mean
        self.duration_std = duration_std
        self.known = known

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        values = sequence(item, self.cache) if self.known else item["sequence"]
        values = (values - self.mean) / self.std
        duration = (np.log1p(item["duration_frames"]) - self.duration_mean) / self.duration_std
        return {"sequence": torch.from_numpy(values.astype(np.float32)), "duration": torch.tensor(duration, dtype=torch.float32), "label": torch.tensor(item["label_id"], dtype=torch.long), "row": item}


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["sequence"].shape[0] for item in items], dtype=torch.long)
    max_length = int(lengths.max())
    values = torch.zeros((len(items), max_length, base.FEATURE_DIM), dtype=torch.float32)
    mask = torch.zeros((len(items), max_length), dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(item["sequence"].shape[0])
        values[index, :length] = item["sequence"]
        mask[index, :length] = True
    return {"sequence": values, "valid_mask": mask, "lengths": lengths, "duration": torch.stack([item["duration"] for item in items]), "label": torch.stack([item["label"] for item in items]), "rows": [item["row"] for item in items]}


def infer(model: nn.Module, loader: DataLoader) -> dict[str, Any]:
    model.eval()
    embeddings = []
    logits = []
    rows = []
    with torch.no_grad():
        for batch in loader:
            embedding, output = model(batch["sequence"], batch["valid_mask"], batch["lengths"], batch["duration"])
            embeddings.append(embedding.numpy())
            logits.append(output.numpy())
            rows.extend(batch["rows"])
    return {"embeddings": np.concatenate(embeddings), "logits": np.concatenate(logits), "rows": rows}


def closed_metrics(outputs: dict[str, Any]) -> dict[str, Any]:
    scores = score_logits(outputs["logits"])
    labels = np.asarray([row["label_id"] for row in outputs["rows"]], dtype=np.int64)
    predictions = scores["top1"]
    matrix = np.zeros((len(KNOWN_CLASSES), len(KNOWN_CLASSES)), dtype=np.int64)
    for truth, prediction in zip(labels, predictions):
        matrix[int(truth), int(prediction)] += 1
    f1s = []
    for index in range(len(KNOWN_CLASSES)):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": float((predictions == labels).mean()), "macro_f1": float(np.mean(f1s))}


def rejection_metrics(outputs: dict[str, Any], accepted: np.ndarray) -> dict[str, Any]:
    scores = score_logits(outputs["logits"])
    labels = np.asarray([row["label_id"] for row in outputs["rows"]], dtype=np.int64)
    predictions = scores["top1"]
    matrix = np.zeros((len(KNOWN_CLASSES), len(KNOWN_CLASSES)), dtype=np.int64)
    for truth, prediction in zip(labels[accepted], predictions[accepted]):
        matrix[int(truth), int(prediction)] += 1
    f1s = []
    per_class = {}
    for index, label in enumerate(KNOWN_CLASSES):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int((labels == index).sum())
        rejected = int(((labels == index) & ~accepted).sum())
        per_class[label] = {"support": support, "false_unknown_rate": rejected / support if support else 0.0, "f1": f1}
        if support:
            f1s.append(f1)
    return {"closed_set_accuracy": float((predictions == labels).mean()), "known_retention": float(accepted.mean()), "false_unknown_rate": float((~accepted).mean()), "accepted_known_accuracy": float((predictions[accepted] == labels[accepted]).mean()) if accepted.any() else 0.0, "macro_f1_after_rejection": float(np.mean(f1s)), "per_class": per_class}


def train_model(model: nn.Module, known_loader: DataLoader, outlier_loader: DataLoader, validation_loader: DataLoader, weights: Tensor, objective: str, lambda_oe: float, margin: float | None) -> tuple[dict[str, Tensor], list[dict[str, Any]], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)
    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        known_iter = iter(known_loader)
        outlier_iter = iter(outlier_loader)
        steps = max(len(known_loader), len(outlier_loader))
        total_loss = total_ce = total_oe = 0.0
        for _ in range(steps):
            try:
                known_batch = next(known_iter)
            except StopIteration:
                known_iter = iter(known_loader)
                known_batch = next(known_iter)
            try:
                outlier_batch = next(outlier_iter)
            except StopIteration:
                outlier_iter = iter(outlier_loader)
                outlier_batch = next(outlier_iter)
            _, known_logits = model(known_batch["sequence"], known_batch["valid_mask"], known_batch["lengths"], known_batch["duration"])
            _, outlier_logits = model(outlier_batch["sequence"], outlier_batch["valid_mask"], outlier_batch["lengths"], outlier_batch["duration"])
            ce = F.cross_entropy(known_logits, known_batch["label"], weight=weights)
            if objective == "uniform_softmax":
                oe_loss = -F.log_softmax(outlier_logits, dim=1).mean()
            else:
                known_energy = -torch.logsumexp(known_logits, dim=1)
                outlier_energy = -torch.logsumexp(outlier_logits, dim=1)
                oe_loss = F.relu(float(margin) + known_energy.mean() - outlier_energy.mean())
            loss = ce + float(lambda_oe) * oe_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            total_ce += float(ce.detach())
            total_oe += float(oe_loss.detach())
        validation_f1 = closed_metrics(infer(model, validation_loader))["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total_loss / steps, "train_cross_entropy": total_ce / steps, "train_oe_loss": total_oe / steps, "validation_macro_f1": validation_f1})
        if validation_f1 > best_f1 + 1e-9:
            best_f1 = validation_f1
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("No best model state was produced.")
    model.load_state_dict(best_state)
    return best_state, history, best_epoch


def validation_auroc(known: dict[str, Any], synthetic: dict[str, Any]) -> dict[str, float]:
    known_scores = score_logits(known["logits"])
    synthetic_scores = score_logits(synthetic["logits"])
    labels = np.concatenate((np.zeros(len(known["rows"])), np.ones(len(synthetic["rows"]))))
    msp = auroc(labels, np.concatenate((-known_scores["max_softmax"], -synthetic_scores["max_softmax"])))
    energy = auroc(labels, np.concatenate((known_scores["energy"], synthetic_scores["energy"])))
    return {"max_softmax_auroc": msp, "energy_auroc": energy, "best_auroc": max(msp, energy)}


def choose_threshold(known: dict[str, Any], synthetic: dict[str, Any]) -> dict[str, Any]:
    known_scores = score_logits(known["logits"])
    synthetic_scores = score_logits(synthetic["logits"])
    options = []
    for score_name, known_score, synthetic_score in (
        ("max_softmax", -known_scores["max_softmax"], -synthetic_scores["max_softmax"]),
        ("energy", known_scores["energy"], synthetic_scores["energy"]),
    ):
        ordered = np.sort(known_score)
        threshold = float(ordered[max(0, int(np.ceil(.95 * len(ordered))) - 1)])
        known_accept = known_score <= threshold
        synthetic_accept = synthetic_score <= threshold
        labels = np.concatenate((np.zeros(len(known_score)), np.ones(len(synthetic_score))))
        options.append({"score": score_name, "threshold": threshold, "known_validation_retention": float(known_accept.mean()), "synthetic_validation_recall": float((~synthetic_accept).mean()), "validation_auroc": auroc(labels, np.concatenate((known_score, synthetic_score)))})
    return max(options, key=lambda row: (row["validation_auroc"], row["synthetic_validation_recall"], -abs(row["known_validation_retention"] - .95)))


def save_model(path: Path, state: dict[str, Tensor], metadata: dict[str, Any]) -> None:
    torch.save({"model_state": state, "ontology_version": ONTOLOGY_VERSION, "held_out_class": HELD_OUT, "known_class_list": list(KNOWN_CLASSES), "metadata": metadata, "optimizer_state": None}, path)


def main() -> int:
    seed_everything()
    (OUTPUT_ROOT / "model").mkdir(parents=True, exist_ok=True)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    checkpoint_hash = sha256_file(CHECKPOINT)
    if payload.get("ontology_metadata", {}).get("held_out_class") != HELD_OUT:
        raise RuntimeError("The frozen checkpoint is not the wipe-holdout model.")
    rows = load_rows()
    cache = load_features(rows)
    train_frames = np.concatenate([sequence(row, cache) for row in rows["train"]], axis=0)
    mean = train_frames.mean(axis=0)
    std = np.maximum(train_frames.std(axis=0), 1e-6)
    durations = np.asarray([np.log1p(row["duration_frames"]) for row in rows["train"]])
    duration_mean = float(durations.mean())
    duration_std = float(max(durations.std(), 1e-6))
    synthetic_train = generate_outliers(rows["train"], cache, "train")
    synthetic_validation = generate_outliers(rows["validation"], cache, "validation")
    known_train_loader = DataLoader(SequenceDataset(rows["train"], cache, mean, std, duration_mean, duration_std, True), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED), collate_fn=collate)
    validation_loader = DataLoader(SequenceDataset(rows["validation"], cache, mean, std, duration_mean, duration_std, True), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    outlier_train_loader = DataLoader(SequenceDataset(synthetic_train, None, mean, std, duration_mean, duration_std, False), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED + 1), collate_fn=collate)
    outlier_validation_loader = DataLoader(SequenceDataset(synthetic_validation, None, mean, std, duration_mean, duration_std, False), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    counts = Counter(row["label_id"] for row in rows["train"])
    class_weights = torch.tensor([1.0 / np.sqrt(counts[index]) for index in range(len(KNOWN_CLASSES))], dtype=torch.float32)
    class_weights *= len(KNOWN_CLASSES) / class_weights.sum()
    stats = []
    for split, values in (("train", synthetic_train), ("validation", synthetic_validation)):
        for kind in OUTLIER_TYPES:
            subset = [item for item in values if item["outlier_type"] == kind]
            lengths = np.asarray([len(item["sequence"]) for item in subset])
            stats.append({"split": split, "outlier_type": kind, "count": len(subset), "mean_duration_frames": float(lengths.mean()), "std_duration_frames": float(lengths.std()), "min_duration_frames": int(lengths.min()), "max_duration_frames": int(lengths.max())})
    write_csv(OUTPUT_ROOT / "synthetic_outlier_statistics.csv", stats)
    grid = [("uniform_softmax", value, None) for value in LAMBDAS] + [("energy_margin", value, margin) for value in LAMBDAS for margin in MARGINS]
    selection = []
    best_models: dict[str, dict[str, Any]] = {}
    history_rows = []
    for run_index, (objective, lambda_oe, margin) in enumerate(grid):
        print(f"[round13] run {run_index + 1}/{len(grid)}: {objective}, lambda={lambda_oe}, margin={margin}", flush=True)
        model = base.SegmentClassifier(base.FEATURE_DIM, base.HIDDEN_DIM, base.PROJECTION_DIM, base.EMBEDDING_DIM, len(KNOWN_CLASSES))
        model.load_state_dict(payload["model_state"])
        run_known_loader = DataLoader(SequenceDataset(rows["train"], cache, mean, std, duration_mean, duration_std, True), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED + run_index), collate_fn=collate)
        run_outlier_loader = DataLoader(SequenceDataset(synthetic_train, None, mean, std, duration_mean, duration_std, False), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED + 100 + run_index), collate_fn=collate)
        state, history, epoch = train_model(model, run_known_loader, run_outlier_loader, validation_loader, class_weights, objective, lambda_oe, margin)
        history_rows.extend({"run_index": run_index, "objective": objective, "lambda_oe": lambda_oe, "energy_margin": margin or "", **row} for row in history)
        val_known = infer(model, validation_loader)
        val_synthetic = infer(model, outlier_validation_loader)
        metrics = validation_auroc(val_known, val_synthetic)
        val_f1 = closed_metrics(val_known)
        candidate = {"run_index": run_index, "objective": objective, "lambda_oe": lambda_oe, "energy_margin": margin or "", "best_epoch": epoch, "validation_known_macro_f1": val_f1["macro_f1"], "validation_known_accuracy": val_f1["accuracy"], "validation_synthetic_auroc_max_softmax": metrics["max_softmax_auroc"], "validation_synthetic_auroc_energy": metrics["energy_auroc"], "validation_synthetic_auroc_best": metrics["best_auroc"], "selected_within_objective": 0, "selected_overall": 0}
        selection.append(candidate)
        previous = best_models.get(objective)
        key = (candidate["validation_synthetic_auroc_best"], candidate["validation_known_macro_f1"], candidate["validation_known_accuracy"])
        if previous is None or key > previous["key"]:
            best_models[objective] = {"key": key, "candidate": candidate, "state": state}
        print(f"[round13] completed run {run_index + 1}: epoch={epoch}, val_f1={val_f1['macro_f1']:.4f}, synthetic_auroc={metrics['best_auroc']:.4f}", flush=True)
    for objective, value in best_models.items():
        value["candidate"]["selected_within_objective"] = 1
    overall = max(best_models.values(), key=lambda value: value["key"])
    overall["candidate"]["selected_overall"] = 1
    selected_objective = overall["candidate"]["objective"]
    for objective, value in best_models.items():
        save_model(OUTPUT_ROOT / "model" / f"{objective}_best.pt", value["state"], {"objective": objective, "selection": value["candidate"], "base_checkpoint_sha256": checkpoint_hash, "optimizer_state_reused": False})
    save_model(OUTPUT_ROOT / "model/selected_model.pt", overall["state"], {"objective": selected_objective, "selection": overall["candidate"], "base_checkpoint_sha256": checkpoint_hash, "optimizer_state_reused": False})
    write_csv(OUTPUT_ROOT / "model_selection.csv", selection)
    write_csv(OUTPUT_ROOT / "training_history.csv", history_rows)
    frozen = {}
    selected_outputs = {}
    for objective, value in best_models.items():
        model = base.SegmentClassifier(base.FEATURE_DIM, base.HIDDEN_DIM, base.PROJECTION_DIM, base.EMBEDDING_DIM, len(KNOWN_CLASSES))
        model.load_state_dict(value["state"])
        val_known = infer(model, validation_loader)
        val_synthetic = infer(model, outlier_validation_loader)
        frozen[objective] = choose_threshold(val_known, val_synthetic)
        selected_outputs[objective] = {"model": model, "validation": val_known, "validation_synthetic": val_synthetic}
    (OUTPUT_ROOT / "frozen_thresholds.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    known_test_rows = [row for row in rows["test"] if row["evaluation_group"] == "known_test"]
    wipe_rows = [row for row in rows["test"] if row["evaluation_group"] == "wipe_unknown"]
    inside_rows = [row for row in rows["test"] if row["evaluation_group"] == "known_inside_wipe"]
    test_loaders = {name: DataLoader(SequenceDataset(values, cache, mean, std, duration_mean, duration_std, True), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate) for name, values in (("known_test", known_test_rows), ("wipe", wipe_rows), ("known_inside_wipe", inside_rows))}
    reference_model = base.SegmentClassifier(base.FEATURE_DIM, base.HIDDEN_DIM, base.PROJECTION_DIM, base.EMBEDDING_DIM, len(KNOWN_CLASSES))
    reference_model.load_state_dict(payload["model_state"])
    reference_loader = DataLoader(SequenceDataset(rows["train"], cache, mean, std, duration_mean, duration_std, True), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    reference = infer(reference_model, reference_loader)
    reference_embeddings = reference["embeddings"]
    reference_rows = reference["rows"]
    test_results = {}
    for objective, result in selected_outputs.items():
        result["known_test"] = infer(result["model"], test_loaders["known_test"])
        result["wipe"] = infer(result["model"], test_loaders["wipe"])
        result["known_inside_wipe"] = infer(result["model"], test_loaders["known_inside_wipe"])
        test_results[objective] = result
    prediction_rows = []
    wipe_rows_output = []
    oe_comparison = []
    for objective, result in test_results.items():
        threshold_info = frozen[objective]
        score_name = threshold_info["score"]
        threshold = float(threshold_info["threshold"])
        group_metrics = {}
        for group in ("known_test", "wipe", "known_inside_wipe"):
            outputs = result[group]
            values = score_logits(outputs["logits"])
            rejection_score = -values["max_softmax"] if score_name == "max_softmax" else values["energy"]
            accepted = rejection_score <= threshold
            group_metrics[group] = rejection_metrics(outputs, accepted) if group != "wipe" else {"unknown_recall": float((~accepted).mean()), "false_known_rate": float(accepted.mean())}
            distances = 1.0 - outputs["embeddings"] @ reference_embeddings.T
            nearest = distances.argmin(axis=1)
            for index, row in enumerate(outputs["rows"]):
                record = {"method": f"oe_{objective}", "group": group, "sample_id": row["sample_id"], "trajectory": row["trajectory"], "segment_index": row["segment_index"], "ground_truth_label": row["label"], "predicted_known_class": ID_TO_LABEL[int(values["top1"][index])], "max_softmax": float(values["max_softmax"][index]), "energy": float(values["energy"][index]), "rejection_score": float(rejection_score[index]), "threshold": threshold, "accepted_as_known": int(accepted[index]), "decision": ID_TO_LABEL[int(values["top1"][index])] if accepted[index] else "unknown", "nearest_known_training_sample": reference_rows[int(nearest[index])]["sample_id"], "nearest_neighbor_cosine_distance": float(distances[index, nearest[index]]), "duration_frames": row["duration_frames"]}
                prediction_rows.append(record)
                if group == "wipe":
                    wipe_rows_output.append(record)
        known_scores = score_logits(result["known_test"]["logits"])
        wipe_scores = score_logits(result["wipe"]["logits"])
        unknown_known = -known_scores["max_softmax"] if score_name == "max_softmax" else known_scores["energy"]
        unknown_wipe = -wipe_scores["max_softmax"] if score_name == "max_softmax" else wipe_scores["energy"]
        auroc_value = auroc(np.concatenate((np.zeros(len(unknown_known)), np.ones(len(unknown_wipe)))), np.concatenate((unknown_known, unknown_wipe)))
        known_metric = group_metrics["known_test"]
        inside_metric = group_metrics["known_inside_wipe"]
        oe_comparison.append({"method": f"oe_{objective}", "threshold_score": score_name, "threshold": threshold, "known_retention": known_metric["known_retention"], "known_false_unknown_rate": known_metric["false_unknown_rate"], "known_macro_f1_before_rejection": closed_metrics(result["known_test"])["macro_f1"], "known_macro_f1_after_rejection": known_metric["macro_f1_after_rejection"], "known_accepted_accuracy": known_metric["accepted_known_accuracy"], "wipe_unknown_recall": group_metrics["wipe"]["unknown_recall"], "wipe_false_known_rate": group_metrics["wipe"]["false_known_rate"], "known_vs_wipe_auroc": auroc_value, "known_inside_wipe_closed_set_accuracy": inside_metric["closed_set_accuracy"], "known_inside_wipe_retention": inside_metric["known_retention"], "known_inside_wipe_false_unknown_rate": inside_metric["false_unknown_rate"], "known_inside_wipe_macro_f1_after_rejection": inside_metric["macro_f1_after_rejection"], "selected_overall": int(objective == selected_objective)})
    write_csv(OUTPUT_ROOT / "segment_predictions.csv", prediction_rows)
    write_csv(OUTPUT_ROOT / "wipe_diagnostics.csv", wipe_rows_output)
    write_csv(OUTPUT_ROOT / "oe_variant_comparison.csv", oe_comparison)
    baseline_known = json.loads((HOLDOUT_ROOT / "known_test_metrics.json").read_text(encoding="utf-8"))
    baseline_wipe = json.loads((HOLDOUT_ROOT / "wipe_unknown_metrics.json").read_text(encoding="utf-8"))
    baseline_inside = json.loads((HOLDOUT_ROOT / "known_inside_wipe_metrics.json").read_text(encoding="utf-8"))
    comparison = list(oe_comparison)
    for method in ("max_softmax", "energy"):
        known = baseline_known["methods"][method]
        wipe = baseline_wipe["methods"][method]
        inside = baseline_inside["methods"][method]
        comparison.append({"method": method, "threshold_score": method, "threshold": known["threshold"], "known_retention": known["known_recall"], "known_false_unknown_rate": known["false_unknown_rate"], "known_macro_f1_before_rejection": known["closed_set_macro_f1"], "known_macro_f1_after_rejection": known["macro_f1_after_rejection"], "known_accepted_accuracy": known["accepted_known_accuracy"], "wipe_unknown_recall": wipe["unknown_recall"], "wipe_false_known_rate": wipe["false_known_rate"], "known_vs_wipe_auroc": "", "known_inside_wipe_closed_set_accuracy": inside["closed_set_accuracy"], "known_inside_wipe_retention": inside["known_recall"], "known_inside_wipe_false_unknown_rate": inside["false_unknown_rate"], "known_inside_wipe_macro_f1_after_rejection": inside["macro_f1_after_rejection"], "selected_overall": 0})
    knn_path = ROOT / "outputs/round12_open_set_cosine_knn_holdout_wipe/baseline_comparison.csv"
    knn = next(row for row in read_csv(knn_path) if row["method"] == "cosine_knn" and row["variant"] == "global" and row["k"] == "1")
    comparison.append({"method": "frozen_cosine_knn_k1", "threshold_score": "mean_cosine_distance", "threshold": knn["threshold"], "known_retention": knn["independent_known_retention"], "known_false_unknown_rate": knn["independent_known_false_unknown_rate"], "known_macro_f1_before_rejection": "", "known_macro_f1_after_rejection": knn["independent_known_macro_f1_after_rejection"], "known_accepted_accuracy": knn["independent_known_accepted_accuracy"], "wipe_unknown_recall": knn["wipe_unknown_recall"], "wipe_false_known_rate": knn["wipe_false_known_rate"], "known_vs_wipe_auroc": "", "known_inside_wipe_closed_set_accuracy": knn["known_inside_wipe_closed_set_accuracy"], "known_inside_wipe_retention": knn["known_inside_wipe_retention"], "known_inside_wipe_false_unknown_rate": "", "known_inside_wipe_macro_f1_after_rejection": knn["known_inside_wipe_macro_f1_after_rejection"], "selected_overall": 0})
    write_csv(OUTPUT_ROOT / "baseline_comparison.csv", comparison)
    checkpoint_names = ("uniform_softmax_best.pt", "energy_margin_best.pt", "selected_model.pt")
    training_config = {"experiment": "round12_open_set_outlier_exposure_holdout_wipe", "ontology_version": ONTOLOGY_VERSION, "held_out_class": HELD_OUT, "known_classes": list(KNOWN_CLASSES), "base_checkpoint": str(CHECKPOINT), "base_checkpoint_sha256": checkpoint_hash, "base_weights_reused": True, "optimizer_state_reused": False, "synthetic_outlier_types": list(OUTLIER_TYPES), "lambda_oe_grid": list(LAMBDAS), "energy_margin_grid": list(MARGINS), "train_segments": len(rows["train"]), "validation_segments": len(rows["validation"]), "test_groups": {"known_test": len(known_test_rows), "wipe": len(wipe_rows), "known_inside_wipe": len(inside_rows)}, "wipe_segments_in_train": 0, "wipe_segments_in_validation": 0, "selected_objective": selected_objective, "checkpoint_hashes": {name: sha256_file(OUTPUT_ROOT / "model" / name) for name in checkpoint_names}, "wipe_used_during_selection": False}
    (OUTPUT_ROOT / "training_config.yaml").write_text(yaml.safe_dump(training_config, sort_keys=False), encoding="utf-8")
    primary = next(row for row in oe_comparison if row["selected_overall"])
    report = [
        "# Round 13 synthetic outlier exposure: wipe holdout",
        "",
        "## Protocol",
        "",
        "Fresh OE models were trained with new optimizer state. Their known-skill backbone was initialized from the frozen Round 12 wipe-holdout checkpoint. Wipe was absent from training, validation, synthetic generation, threshold selection, and model selection. Test inference occurred only after all model and threshold choices were frozen. Annotations were not modified. No prototypes, clustering, or kNN tuning was used.",
        "",
        f"- Selected objective: {selected_objective}",
        f"- Base checkpoint SHA-256: {checkpoint_hash}",
        f"- Segments train/validation/known-test/wipe/inside-wipe: {len(rows['train'])}/{len(rows['validation'])}/{len(known_test_rows)}/{len(wipe_rows)}/{len(inside_rows)}",
        f"- Synthetic outliers train/validation: {len(synthetic_train)}/{len(synthetic_validation)}",
        "",
        "## Frozen thresholds",
        "",
        "| objective | score | threshold | known validation retention | synthetic validation recall | validation AUROC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for objective, threshold in frozen.items():
        report.append(f"| {objective} | {threshold['score']} | {threshold['threshold']:.9f} | {threshold['known_validation_retention']:.6f} | {threshold['synthetic_validation_recall']:.6f} | {threshold['validation_auroc']:.6f} |")
    report.extend(["", "## Final comparison", "", "| method | known retention | known F1 before | known F1 after | wipe unknown recall | wipe false-known | known-vs-wipe AUROC | inside-wipe accuracy | inside-wipe retention |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in comparison:
        def fmt(value: Any) -> str:
            try:
                return f"{float(value):.6f}"
            except (TypeError, ValueError):
                return ""
        report.append(f"| {row['method']} | {fmt(row['known_retention'])} | {fmt(row['known_macro_f1_before_rejection'])} | {fmt(row['known_macro_f1_after_rejection'])} | {fmt(row['wipe_unknown_recall'])} | {fmt(row['wipe_false_known_rate'])} | {fmt(row['known_vs_wipe_auroc'])} | {fmt(row['known_inside_wipe_closed_set_accuracy'])} | {fmt(row['known_inside_wipe_retention'])} |")
    report.extend([
        "",
        "## Conclusions",
        "",
        f"1. The selected OE model achieved wipe unknown recall {primary['wipe_unknown_recall']:.6f} with known retention {primary['known_retention']:.6f}; both values are required for an improvement claim.",
        "2. Uniform-softmax and energy-margin objective results, including validation AUROC and synthetic recall, are in oe_variant_comparison.csv and model_selection.csv.",
        "3. synthetic_outlier_statistics.csv reports all four constructions. No wipe-derived statistic was used for selection.",
        "4. Wipe remains a held-out unknown skill. Rejection does not constitute new-skill discovery.",
        "5. Results must be interpreted against the frozen max-softmax, energy, and cosine-kNN baselines, especially the known-retention cost.",
        "",
        "## Integrity",
        "",
        "Training and validation manifests contain zero wipe segments. The selected and objective checkpoints contain no optimizer state. Wipe diagnostics include classifier scores, decisions, nearest known training segments, cosine distances, and durations.",
    ])
    (OUTPUT_ROOT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "selected_objective": selected_objective, "primary": primary, "output": str(OUTPUT_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
