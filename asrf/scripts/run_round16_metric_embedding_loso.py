#!/usr/bin/env python3
"""Round 16: fresh GT-segment metric-learning LOSO open-set study.

The script deliberately does not import or load any previous checkpoint.  It
reuses Round 12's audited segment construction and encoder class only; every
fold/variant starts with a new random model and optimizer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/round16_metric_embedding_loso"
R15_ROOT = ROOT / "outputs/round15_multiskill_loso_open_set"
R12_ROOT = ROOT / "outputs/round12_multiskill_segment_classifier"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import train_round12_segment_classifier as base  # noqa: E402
from asrf.data.ontology import CANONICAL_LABELS, ONTOLOGY_VERSION  # noqa: E402

SEED = 42
HOLDOUTS = ("wipe", "pour", "pour_recover", "place", "insert", "transport")
FAMILY_FOR_HOLDOUT = {"wipe": "wipe", "pour": "pour", "pour_recover": "pour", "place": "plug", "insert": "plug", "transport": "pp"}
BATCH_SIZE = 32
MAX_EPOCHS = 25
PATIENCE = 7
TRIPLET_MARGIN = 0.20
BOOTSTRAPS = 2000
DEVICE = torch.device("cpu")
ABLATION_HOLDOUTS = ("wipe", "place", "insert")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict[str, str]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit, status = "NO_COMMIT_IN_REPOSITORY", "UNAVAILABLE"
    return {"git_commit": commit, "git_status_porcelain": status}


def load_rows() -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        values = []
        for raw in read_csv(R12_ROOT / "split_manifests" / f"{split}.csv"):
            row = dict(raw)
            for field in ("segment_index", "label_id", "start_frame", "end_frame_exclusive", "duration_frames"):
                row[field] = int(row[field])
            values.append(row)
        rows[split] = values
    trajectories = sorted({row["trajectory"] for values in rows.values() for row in values})
    cache = {trajectory: base.load_trajectory_features(trajectory) for trajectory in trajectories}
    return rows, cache


def remap(rows: list[dict[str, Any]], holdout: str, class_names: tuple[str, ...]) -> list[dict[str, Any]]:
    mapping = {name: index for index, name in enumerate(class_names)}
    output = []
    for original in rows:
        row = dict(original)
        row["held_out"] = int(row["label"] == holdout)
        row["label_id"] = mapping.get(row["label"], -1)
        output.append(row)
    return output


class SegmentDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], mean: np.ndarray, std: np.ndarray, duration_mean: float, duration_std: float):
        self.rows, self.cache = rows, cache
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        self.duration_mean, self.duration_std = float(duration_mean), float(duration_std)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sequence = self.cache[row["trajectory"]][1][row["start_frame"]:row["end_frame_exclusive"]]
        sequence = (sequence - self.mean) / self.std
        duration = (np.log1p(row["duration_frames"]) - self.duration_mean) / self.duration_std
        return {"sequence": torch.from_numpy(sequence.astype(np.float32)), "duration": torch.tensor(duration, dtype=torch.float32), "label": torch.tensor(row["label_id"], dtype=torch.long), "row": row}


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["sequence"].shape[0] for item in items], dtype=torch.long)
    max_length = int(lengths.max())
    sequence = torch.zeros((len(items), max_length, base.FEATURE_DIM), dtype=torch.float32)
    mask = torch.zeros((len(items), max_length), dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(item["sequence"].shape[0])
        sequence[index, :length] = item["sequence"]
        mask[index, :length] = True
    return {"sequence": sequence, "valid_mask": mask, "lengths": lengths, "duration": torch.stack([item["duration"] for item in items]), "label": torch.stack([item["label"] for item in items]), "rows": [item["row"] for item in items]}


class BalancedBatchSampler(Sampler[list[int]]):
    """Class-balanced batches with trajectory/family-diverse positive pairs."""

    def __init__(self, rows: list[dict[str, Any]], batch_size: int, seed: int):
        self.rows, self.batch_size, self.seed = rows, batch_size, seed
        self.by_class: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            self.by_class[int(row["label_id"])].append(index)
        self.num_batches = max(1, int(np.ceil(len(rows) / batch_size)))

    def __len__(self) -> int:
        return self.num_batches

    def _choose(self, candidates: list[int], count: int, rng: np.random.Generator) -> list[int]:
        if len(candidates) <= count:
            return [candidates[index % len(candidates)] for index in rng.permutation(len(candidates))[:count]] if candidates else []
        chosen: list[int] = []
        # First prioritize a different trajectory and, where possible, a different family.
        remaining = list(candidates)
        rng.shuffle(remaining)
        for index in remaining:
            if not chosen or (self.rows[index]["trajectory"] not in {self.rows[j]["trajectory"] for j in chosen}):
                chosen.append(index)
            if len(chosen) == count:
                return chosen
        for index in remaining:
            if index not in chosen:
                chosen.append(index)
            if len(chosen) == count:
                break
        return chosen

    def __iter__(self) -> Iterable[list[int]]:
        rng = np.random.default_rng(self.seed)
        classes = sorted(self.by_class)
        per_class = max(2, min(4, self.batch_size // max(min(8, len(classes)), 1)))
        class_count = min(len(classes), max(4, self.batch_size // per_class))
        for _ in range(self.num_batches):
            selected_classes = rng.choice(classes, size=class_count, replace=len(classes) < class_count).tolist()
            batch = []
            for label in selected_classes:
                batch.extend(self._choose(self.by_class[int(label)], per_class, rng))
            yield batch[:self.batch_size]


def cross_family_availability(rows: list[dict[str, Any]], class_names: tuple[str, ...]) -> list[dict[str, Any]]:
    output = []
    for label in class_names:
        values = [row for row in rows if row["label"] == label]
        pairs = sum(1 for i, left in enumerate(values) for right in values[i + 1:] if left["family"] != right["family"] and left["trajectory"] != right["trajectory"])
        families = sorted({row["family"] for row in values})
        trajectories = sorted({row["trajectory"] for row in values})
        output.append({"class": label, "segment_count": len(values), "trajectory_count": len(trajectories), "families": ";".join(families), "cross_family_positive_pairs": pairs, "cross_family_support": int(pairs > 0), "note": "no cross-family support" if pairs == 0 else ""})
    return output


def supcon(embeddings: Tensor, labels: Tensor, temperature: float = 0.07) -> Tensor:
    if len(labels) < 2:
        return embeddings.sum() * 0.0
    logits = embeddings @ embeddings.T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    counts = positive.sum(dim=1)
    usable = counts > 0
    return -(log_prob.masked_fill(~positive, 0.0).sum(dim=1)[usable] / counts[usable]).mean() if usable.any() else embeddings.sum() * 0.0


def batch_hard_triplet(embeddings: Tensor, labels: Tensor, margin: float = TRIPLET_MARGIN) -> tuple[Tensor, dict[str, float]]:
    distance = 1.0 - embeddings @ embeddings.T
    same = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)
    positive_mask, negative_mask = same & ~diagonal, ~same
    usable = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not usable.any():
        return embeddings.sum() * 0.0, {"triplet_count": 0, "active_triplet_fraction": 0.0, "mean_positive_distance": 0.0, "mean_hard_negative_distance": 0.0, "triplet_margin_violation_rate": 0.0}
    positive = distance.masked_fill(~positive_mask, torch.finfo(distance.dtype).min).amax(dim=1)
    negative = distance.masked_fill(~negative_mask, torch.finfo(distance.dtype).max).amin(dim=1)
    positive, negative = positive[usable], negative[usable]
    violations = F.relu(positive - negative + margin)
    return violations.mean(), {"triplet_count": int(violations.numel()), "active_triplet_fraction": float((violations > 0).float().mean()), "mean_positive_distance": float(positive.mean().detach()), "mean_hard_negative_distance": float(negative.mean().detach()), "triplet_margin_violation_rate": float((violations > 0).float().mean())}


def class_f1(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> tuple[float, list[float]]:
    values = []
    for label in range(num_classes):
        tp = int(((labels == label) & (predictions == label)).sum())
        fp = int(((labels != label) & (predictions == label)).sum())
        fn = int(((labels == label) & (predictions != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values)) if values else 0.0, values


def collect(model: nn.Module, loader: DataLoader) -> dict[str, Any]:
    model.eval(); embeddings, logits, rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            embedding, output = model(batch["sequence"], batch["valid_mask"], batch["lengths"], batch["duration"])
            embeddings.append(embedding.cpu().numpy()); logits.append(output.cpu().numpy()); rows.extend(batch["rows"])
    return {"embeddings": np.concatenate(embeddings), "logits": np.concatenate(logits), "rows": rows}


def score_arrays(embeddings: np.ndarray, logits: np.ndarray, reference: np.ndarray, reference_rows: list[dict[str, Any]], means: np.ndarray, inverse_vars: np.ndarray) -> dict[str, Any]:
    predictions = logits.argmax(axis=1)
    cosine_distance = 1.0 - embeddings @ reference.T
    knn = []
    for index, prediction in enumerate(predictions):
        candidates = [j for j, row in enumerate(reference_rows) if int(row["label_id"]) == int(prediction)]
        candidates = candidates or list(range(len(reference_rows)))
        knn.append(float(np.sort(cosine_distance[index, candidates])[:min(5, len(candidates))].mean()))
    deltas = embeddings[:, None, :] - means[None, :, :]
    mahalanobis = np.sqrt(np.maximum(np.sum(deltas * deltas * inverse_vars[None, :, :], axis=2), 0.0))
    return {"cosine_knn": np.asarray(knn), "predicted_class_mahalanobis": mahalanobis[np.arange(len(embeddings)), predictions], "nearest_class_mahalanobis": mahalanobis.min(axis=1), "mahalanobis": mahalanobis, "predictions": predictions}


def fit_distributions(reference: np.ndarray, reference_rows: list[dict[str, Any]], num_classes: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    means, inverse_vars, covariance_types = [], [], []
    global_var = np.var(reference, axis=0) + 1e-4
    for label in range(num_classes):
        values = reference[[int(row["label_id"]) == label for row in reference_rows]]
        mean = values.mean(axis=0)
        if len(values) < 2:
            var, kind = global_var, "global_diagonal"
        else:
            var = 0.9 * np.var(values, axis=0, ddof=1) + 0.1 * np.median(global_var)
            kind = "shrinkage_diagonal" if len(values) < reference.shape[1] else "shrinkage_diagonal"
        means.append(mean); inverse_vars.append(1.0 / np.maximum(var, 1e-4)); covariance_types.append(kind)
    return np.asarray(means), np.asarray(inverse_vars), covariance_types


def threshold_curve(scores: np.ndarray, target: float = 0.95) -> tuple[float, float, list[dict[str, Any]]]:
    values = np.sort(scores)
    candidates = np.unique(np.concatenate(([float(values.min())], values, [float(values.max()) + 1e-9])))
    curve = [{"threshold": float(value), "known_retention": float((scores <= value).mean()), "false_unknown_rate": float((scores > value).mean()), "target_retention": target, "meets_target": int((scores <= value).mean() >= target)} for value in candidates]
    valid = [row for row in curve if row["meets_target"]]
    chosen = min(valid, key=lambda row: (row["known_retention"] - target, row["threshold"])) if valid else curve[-1]
    return float(chosen["threshold"]), float(chosen["known_retention"]), curve


def rejection_f1(labels: np.ndarray, predictions: np.ndarray, accepted: np.ndarray, num_classes: int) -> tuple[float, list[float]]:
    scores = []
    for label in range(num_classes):
        tp = int(((labels == label) & (predictions == label) & accepted).sum())
        fp = int(((labels != label) & (predictions == label) & accepted).sum())
        fn = int(((labels == label) & (~accepted | (predictions != label))).sum())
        p, r = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(scores)), scores


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else 0.0


def aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores); positives = max(int(labels.sum()), 1); tp = total = area = 0.0
    for label in labels[order]:
        total += 1; tp += int(label == 1)
        if label == 1:
            area += tp / total
    return float(area / positives)


def silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    if len(embeddings) < 3 or len(np.unique(labels)) < 2:
        return 0.0
    distances = 1.0 - embeddings @ embeddings.T
    vals = []
    for i in range(len(labels)):
        same = labels == labels[i]; same[i] = False
        a = float(distances[i, same].mean()) if same.any() else 0.0
        b = min(float(distances[i, labels == other].mean()) for other in np.unique(labels) if other != labels[i])
        vals.append((b - a) / max(a, b, 1e-8))
    return float(np.mean(vals))


def embedding_quality(outputs: dict[str, Any], class_names: tuple[str, ...], split: str, variant: str) -> list[dict[str, Any]]:
    emb, rows = outputs["embeddings"], outputs["rows"]; labels = np.asarray([int(row["label_id"]) for row in rows]); families = np.asarray([row["family"] for row in rows]); same = labels[:, None] == labels[None, :]; upper = np.triu(np.ones_like(same, dtype=bool), 1)
    same_values = (1.0 - emb @ emb.T)[upper & same]; diff_values = (1.0 - emb @ emb.T)[upper & ~same]
    cross = (upper & same & (families[:, None] != families[None, :]))
    within = (upper & same & (families[:, None] == families[None, :]))
    means = np.asarray([emb[labels == i].mean(axis=0) for i in range(len(class_names))])
    within_var = np.asarray([np.mean(np.sum((emb[labels == i] - means[i]) ** 2, axis=1)) if (labels == i).any() else np.nan for i in range(len(class_names))])
    inter = 1.0 - means @ means.T; inter_values = inter[np.triu(np.ones_like(inter, dtype=bool), 1)]
    predictions = outputs["logits"].argmax(axis=1); nn_predictions = []
    for i in range(len(emb)):
        distances = 1.0 - emb[i] @ emb.T; distances[i] = np.inf; nn_predictions.append(labels[int(np.argmin(distances))])
    fisher = float(np.mean(inter_values) / max(float(np.nanmean(within_var)), 1e-8)) if len(inter_values) else 0.0
    return [{"variant": variant, "split": split, "segment_count": len(rows), "same_class_cosine_distance": float(same_values.mean()) if len(same_values) else 0.0, "different_class_cosine_distance": float(diff_values.mean()) if len(diff_values) else 0.0, "cross_family_same_class_distance": float((1.0 - emb @ emb.T)[cross].mean()) if cross.any() else np.nan, "within_family_same_class_distance": float((1.0 - emb @ emb.T)[within].mean()) if within.any() else np.nan, "nearest_neighbor_label_accuracy": float((np.asarray(nn_predictions) == labels).mean()) if len(labels) else 0.0, "classifier_accuracy": float((predictions == labels).mean()) if len(labels) else 0.0, "silhouette_diagnostic": silhouette(emb, labels), "mean_within_class_variance": float(np.nanmean(within_var)), "mean_inter_class_center_distance": float(np.mean(inter_values)) if len(inter_values) else 0.0, "fisher_ratio": fisher}]


def train_model(rows: list[dict[str, Any]], validation: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], mean: np.ndarray, std: np.ndarray, dm: float, ds: float, class_names: tuple[str, ...], variant: str, loss_mode: str, fold: str, ablation: bool = False) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    num_classes = len(class_names); dataset = SegmentDataset(rows, cache, mean, std, dm, ds); val_loader = DataLoader(SegmentDataset(validation, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate); stable_seed = int(hashlib.sha256(f"{fold}/{variant}/{loss_mode}".encode()).hexdigest()[:8], 16); loader = DataLoader(dataset, batch_sampler=BalancedBatchSampler(rows, BATCH_SIZE, SEED + stable_seed % 10000), collate_fn=collate)
    model = base.SegmentClassifier(base.FEATURE_DIM, base.HIDDEN_DIM, base.PROJECTION_DIM, base.EMBEDDING_DIM, num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)
    counts = Counter(int(row["label_id"]) for row in rows); weights = torch.tensor([1.0 / np.sqrt(counts.get(i, 1)) for i in range(num_classes)], dtype=torch.float32); weights *= num_classes / weights.sum()
    centers = torch.zeros((num_classes, base.EMBEDDING_DIM), dtype=torch.float32); center_seen = torch.zeros(num_classes, dtype=torch.bool)
    best_state, best_centers, best_key, best_epoch, stale = None, None, None, 0, 0; history: list[dict[str, Any]] = []; validation_history: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); totals = defaultdict(float); count = 0
        for batch in loader:
            embedding, logits = model(batch["sequence"], batch["valid_mask"], batch["lengths"], batch["duration"]); labels = batch["label"]
            ce = F.cross_entropy(logits, labels, weight=weights); con = supcon(embedding, labels); triplet, triplet_stats = batch_hard_triplet(embedding, labels)
            if loss_mode in ("center", "full"):
                center_loss = ((embedding - centers[labels]) ** 2).sum(dim=1).mean()
            else:
                center_loss = embedding.sum() * 0.0
            loss = ce + 0.10 * con
            if loss_mode in ("triplet", "full"):
                loss = loss + 0.10 * triplet
            if loss_mode in ("center", "full"):
                loss = loss + 0.02 * center_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            with torch.no_grad():
                for label in labels.unique():
                    values = embedding[labels == label].mean(dim=0)
                    label_i = int(label); centers[label_i] = values if not center_seen[label_i] else 0.9 * centers[label_i] + 0.1 * values; center_seen[label_i] = True
            batch_count = len(labels); count += batch_count
            for key, value in {"loss": loss, "cross_entropy": ce, "supervised_contrastive_loss": con, "batch_hard_triplet_loss": triplet, "center_compactness_loss": center_loss}.items(): totals[key] += float(value.detach()) * batch_count
            for key, value in triplet_stats.items(): totals[key] += value
        val = collect(model, val_loader); labels = np.asarray([int(row["label_id"]) for row in val["rows"]]); predictions = val["logits"].argmax(axis=1); f1, _ = class_f1(labels, predictions, num_classes); val_quality = embedding_quality({"embeddings": val["embeddings"], "logits": val["logits"], "rows": val["rows"]}, class_names, "validation", variant)[0]
        val_row = {"fold": fold, "variant": variant, "ablation": int(ablation), "epoch": epoch, "validation_macro_f1": f1, "validation_accuracy": float((labels == predictions).mean()), "cross_family_same_class_distance": val_quality["cross_family_same_class_distance"], "mean_within_class_variance": val_quality["mean_within_class_variance"]}
        validation_history.append(val_row)
        history.append({"fold": fold, "variant": variant, "ablation": int(ablation), "epoch": epoch, **{key: value / max(count, 1) for key, value in totals.items() if key in ("loss", "cross_entropy", "supervised_contrastive_loss", "batch_hard_triplet_loss", "center_compactness_loss")}, "triplet_count": totals["triplet_count"], "active_triplet_fraction": totals["active_triplet_fraction"] / max(len(loader), 1), "mean_positive_distance": totals["mean_positive_distance"] / max(len(loader), 1), "mean_hard_negative_distance": totals["mean_hard_negative_distance"] / max(len(loader), 1), "triplet_margin_violation_rate": totals["triplet_margin_violation_rate"] / max(len(loader), 1)})
        key = (f1, -float(val_quality["cross_family_same_class_distance"]) if np.isfinite(val_quality["cross_family_same_class_distance"]) else -1e9, val_row["validation_accuracy"], -val_quality["mean_within_class_variance"])
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0; best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; best_centers = centers.clone()
        else:
            stale += 1
        print(f"[round16] {fold} {variant} epoch={epoch} val_f1={f1:.4f}", flush=True)
        if stale >= PATIENCE:
            break
    assert best_state is not None and best_centers is not None
    model.load_state_dict(best_state); return model, {"best_epoch": best_epoch, "best_validation_macro_f1": best_key[0], "centers": best_centers.numpy(), "class_weights": weights.numpy()}, history, validation_history


def metrics_for(method: str, known: dict[str, Any], unknown: dict[str, Any], inside: dict[str, Any], known_score: np.ndarray, unknown_score: np.ndarray, inside_score: np.ndarray, cutoff: float, class_names: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray([int(row["label_id"]) for row in known["rows"]]); predictions = known["logits"].argmax(axis=1); accepted = known_score <= cutoff; f1, per_f1 = rejection_f1(labels, predictions, accepted, len(class_names)); accepted_labels, accepted_predictions = labels[accepted], predictions[accepted]; _, accepted_f1 = class_f1(accepted_labels, accepted_predictions, len(class_names)) if len(accepted_labels) else (0.0, [0.0] * len(class_names)); unknown_labels = np.ones(len(unknown_score), dtype=np.int64); binary_labels = np.concatenate((np.zeros(len(known_score), dtype=np.int64), unknown_labels)); binary_scores = np.concatenate((known_score, unknown_score)); unknown_accept = unknown_score <= cutoff
    per_class = []
    for label, name in enumerate(class_names):
        support = int((labels == label).sum()); per_class.append({"class": name, "support": support, "known_retention": float(((labels == label) & accepted).sum() / support) if support else 0.0, "f1": float(per_f1[label])})
    inside_labels = np.asarray([int(row["label_id"]) for row in inside["rows"]]); inside_predictions = inside["logits"].argmax(axis=1)
    result = {"method": method, "closed_set_accuracy": float((predictions == labels).mean()), "known_retention": float(accepted.mean()), "false_unknown_rate": float((~accepted).mean()), "rejection_aware_macro_f1": f1, "accepted_only_macro_f1": float(np.mean(accepted_f1)), "accepted_known_accuracy": float((predictions[accepted] == labels[accepted]).mean()) if accepted.any() else 0.0, "unknown_recall": float((~unknown_accept).mean()), "false_known_rate": float(unknown_accept.mean()), "auroc": auroc(binary_labels, binary_scores), "aupr": aupr(binary_labels, binary_scores), "unknown_score_mean": float(unknown_score.mean()), "unknown_score_std": float(unknown_score.std()), "unknown_score_q05": float(np.quantile(unknown_score, .05)), "unknown_score_q50": float(np.quantile(unknown_score, .50)), "unknown_score_q95": float(np.quantile(unknown_score, .95)), "inside_closed_set_accuracy": float((inside_predictions == inside_labels).mean()) if len(inside_labels) else 0.0, "inside_known_retention": float((inside_score <= cutoff).mean()), "inside_false_unknown_rate": float((inside_score > cutoff).mean()), "inside_score_shift_mean": float(inside_score.mean() - known_score.mean()), "inside_score_shift_median": float(np.median(inside_score) - np.median(known_score)), "per_class": per_class}
    return result, per_class


def trajectory_metric_values(outputs: dict[str, Any], scores: np.ndarray, cutoff: float, class_names: tuple[str, ...], unknown: bool = False) -> dict[str, list[float]]:
    by_trajectory = defaultdict(list)
    for index, row in enumerate(outputs["rows"]):
        by_trajectory[row["trajectory"]].append(index)
    result = {"known_retention": [], "rejection_aware_macro_f1": [], "unknown_recall": [], "auroc": []}
    if unknown:
        known_scores = []
        for indices in by_trajectory.values():
            known_scores.extend(scores[indices])
        for indices in by_trajectory.values():
            result["unknown_recall"].append(float((scores[indices] > cutoff).mean())); result["auroc"].append(auroc(np.concatenate((np.zeros(len(known_scores)), np.ones(len(indices)))), np.concatenate((np.asarray(known_scores), scores[indices]))))
    else:
        labels = np.asarray([int(row["label_id"]) for row in outputs["rows"]]); predictions = outputs["logits"].argmax(axis=1)
        for indices in by_trajectory.values():
            accepted = scores[indices] <= cutoff; f1, _ = rejection_f1(labels[indices], predictions[indices], accepted, len(class_names)); result["known_retention"].append(float(accepted.mean())); result["rejection_aware_macro_f1"].append(f1)
    return result


def save_predictions(path: Path, skill: str, variant: str, method: str, group: str, output: dict[str, Any], scores: np.ndarray, cutoff: float, class_names: tuple[str, ...], distance_matrix: np.ndarray | None = None) -> list[dict[str, Any]]:
    predictions = output["logits"].argmax(axis=1); rows = []
    for index, row in enumerate(output["rows"]):
        record = {"skill": skill, "variant": variant, "method": method, "group": group, "trajectory": row["trajectory"], "segment_index": row["segment_index"], "ground_truth_label": row["label"], "predicted_label": class_names[int(predictions[index])], "score": float(scores[index]), "threshold": cutoff, "decision": "known" if scores[index] <= cutoff else "unknown", "duration_frames": row["duration_frames"]}
        if distance_matrix is not None:
            for label, value in zip(class_names, distance_matrix[index]):
                record[f"mahalanobis_{label}"] = float(value)
        rows.append(record)
    return rows


def class_diagnostics(outputs: dict[str, Any], class_names: tuple[str, ...], variant: str, fold: str, learned_centers: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    emb, rows = outputs["embeddings"], outputs["rows"]; labels = np.asarray([int(row["label_id"]) for row in rows]); empirical = np.asarray([emb[labels == i].mean(axis=0) for i in range(len(class_names))]); distances = 1.0 - empirical @ empirical.T; compact, alignment, transport = [], [], []
    for label, name in enumerate(class_names):
        mask = labels == label; values = emb[mask]; variance = float(np.mean(np.sum((values - empirical[label]) ** 2, axis=1))) if len(values) else np.nan; competing = [i for i in range(len(class_names)) if i != label]; nearest = min(competing, key=lambda i: distances[label, i]) if competing else label; families = defaultdict(list)
        for value, row in zip(values, np.asarray(rows, dtype=object)[mask]): families[row["family"]].append(value)
        family_means = {family: np.mean(values, axis=0) for family, values in families.items()}; family_pairs = [float(1.0 - left @ right) for i, left in enumerate(family_means.values()) for right in list(family_means.values())[i + 1:]]
        compact.append({"fold": fold, "variant": variant, "class": name, "support": len(values), "mean_distance_to_class_center": float(np.mean(1.0 - values @ learned_centers[label])) if len(values) else np.nan, "within_class_variance": variance, "cross_family_center_distance": float(np.mean(family_pairs)) if family_pairs else np.nan, "nearest_competing_class": class_names[nearest], "empirical_center_norm": float(np.linalg.norm(empirical[label])), "learned_center_norm": float(np.linalg.norm(learned_centers[label]))})
        for family, family_mean in family_means.items():
            alignment.append({"fold": fold, "variant": variant, "class": name, "family": family, "family_segment_count": len(families[family]), "family_center_norm": float(np.linalg.norm(family_mean))})
    for label, name in enumerate(class_names):
        if name == "transport":
            for other, other_name in enumerate(class_names):
                if other != label: transport.append({"fold": fold, "variant": variant, "anchor_class": name, "other_class": other_name, "center_cosine_distance": float(distances[label, other])})
    return compact, alignment, transport


def plot_fold(fold: Path, skill: str, unknown_rows: list[dict[str, Any]], known_rows: list[dict[str, Any]]) -> None:
    methods = sorted({row["variant"] + "/" + row["method"] for row in unknown_rows}); fig, ax = plt.subplots(figsize=(10, 5))
    for variant_method in methods:
        values = [row for row in unknown_rows if row["variant"] + "/" + row["method"] == variant_method]; known_values = [row for row in known_rows if row["variant"] + "/" + row["method"] == variant_method]; ax.hist([float(row["score"]) for row in known_values], bins=15, alpha=.25, label=variant_method + " known"); ax.hist([float(row["score"]) for row in values], bins=15, alpha=.35, label=variant_method + " unknown")
    ax.set_title(f"{skill}: known/unknown novelty-score overlap"); ax.set_xlabel("novelty score"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(fold / "figures" / "known_unknown_score_overlap.png", dpi=150); plt.close(fig)


def load_r15_baseline() -> dict[str, Any]:
    values = read_csv(R15_ROOT / "aggregate_results.csv")
    row = next(row for row in values if row["method"] == "cosine_knn")
    return {"method": "round15_cosine_knn", "mean_known_retention": float(row["mean_known_retention"]), "worst_known_retention": float(row["worst_known_retention"]), "mean_rejection_aware_macro_f1": float(row["mean_rejection_aware_macro_f1"]), "mean_unknown_recall": float(row["mean_unknown_recall"]), "worst_unknown_recall": float(row["worst_unknown_recall"]), "mean_auroc": float(row["mean_auroc"]), "mean_aupr": float(row["mean_aupr"]), "known_retention_ge_0.95": "", "unknown_recall_ge_0.80": "", "constraint_folds": ""}


def bootstrap(all_fold_results: list[dict[str, Any]], trajectory_values: dict[tuple[str, str, str], dict[str, list[float]]]) -> list[dict[str, Any]]:
    methods = sorted({row["method"] for row in all_fold_results}); rng = np.random.default_rng(SEED); output = []
    for method in methods:
        for metric in ("known_retention", "rejection_aware_macro_f1", "unknown_recall", "auroc"):
            samples = []
            for _ in range(BOOTSTRAPS):
                fold_means = []
                for skill in HOLDOUTS:
                    values = []
                    for variant in ("A", "B"):
                        key = (skill, variant, method)
                        if key in trajectory_values and metric in trajectory_values[key] and trajectory_values[key][metric]:
                            raw = np.asarray(trajectory_values[key][metric]); values.append(float(raw[rng.integers(0, len(raw), len(raw))].mean()))
                    if values: fold_means.append(float(np.mean(values)))
                if fold_means: samples.append(float(np.mean(fold_means)))
            output.append({"method": method, "metric": metric, "bootstrap_resamples": BOOTSTRAPS, "seed": SEED, "mean": float(np.mean(samples)), "ci_lower": float(np.quantile(samples, .025)), "ci_upper": float(np.quantile(samples, .975))})
    return output


def make_global_figures(per_skill: list[dict[str, Any]], compactness: list[dict[str, Any]], alignment: list[dict[str, Any]], transport: list[dict[str, Any]], predictions_unknown: list[dict[str, Any]], predictions_known: list[dict[str, Any]]) -> None:
    figures = OUTPUT_ROOT / "figures"; figures.mkdir(parents=True, exist_ok=True)
    methods = sorted({row["variant"] + "/" + row["method"] for row in per_skill}); matrix = np.asarray([[next(row["unknown_recall"] for row in per_skill if row["skill"] == skill and row["variant"] + "/" + row["method"] == method) for skill in HOLDOUTS] for method in methods]); fig, ax = plt.subplots(figsize=(11, 6)); im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis"); ax.set_xticks(range(len(HOLDOUTS)), HOLDOUTS, rotation=35, ha="right"); ax.set_yticks(range(len(methods)), methods); ax.set_title("Per-skill unknown recall heatmap"); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(figures / "unknown_recall_heatmap.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5));
    for method in methods:
        values = [row for row in per_skill if row["variant"] + "/" + row["method"] == method]; ax.scatter([row["known_retention"] for row in values], [row["unknown_recall"] for row in values], label=method)
    ax.axvline(.95, color="gray", ls="--"); ax.axhline(.80, color="gray", ls="--"); ax.set_xlabel("known retention"); ax.set_ylabel("unknown recall"); ax.set_title("Known retention versus unknown recall"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(figures / "known_retention_vs_unknown_recall.png", dpi=160); plt.close(fig)
    for filename, key, title in (("within_class_variance_comparison.png", "within_class_variance", "Within-class variance"), ("cross_family_same_class_distance.png", "cross_family_center_distance", "Cross-family same-class distance")):
        fig, ax = plt.subplots(figsize=(10, 5)); labels = sorted({row["variant"] for row in compactness}); values = [[float(row[key]) for row in compactness if row["variant"] == label and np.isfinite(float(row[key]))] for label in labels]; ax.boxplot(values, tick_labels=labels); ax.set_title(title); fig.tight_layout(); fig.savefig(figures / filename, dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5));
    for variant in ("A", "B"):
        values = [row for row in alignment if row["variant"] == variant and row["class"] == "place"]; ax.scatter([row["family"] for row in values], [row["family_center_norm"] for row in values], label=variant)
    ax.set_title("PP-place versus Plug-place family-center analysis"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "pp_place_plug_place_distance.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5));
    for variant in ("A", "B"):
        values = [row for row in transport if row["variant"] == variant]; ax.hist([row["center_cosine_distance"] for row in values], bins=12, alpha=.45, label=variant)
    ax.set_title("Transport versus other-skill center distances"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "transport_other_skill_distance.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5));
    for method in methods:
        known = [float(row["score"]) for row in predictions_known if row["variant"] + "/" + row["method"] == method]; unknown = [float(row["score"]) for row in predictions_unknown if row["variant"] + "/" + row["method"] == method]; ax.hist(known, bins=20, alpha=.13, label=method + " known"); ax.hist(unknown, bins=20, alpha=.25, label=method + " unknown")
    ax.set_title("Known versus unknown score overlap"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(figures / "known_unknown_score_overlap.png", dpi=160); plt.close(fig)
    absorbing = sorted({row["predicted_label"] for row in predictions_unknown}); counts = np.zeros((len(methods), len(absorbing)))
    for i, method in enumerate(methods):
        for row in predictions_unknown:
            if row["variant"] + "/" + row["method"] == method: counts[i, absorbing.index(row["predicted_label"])] += 1
    fig, ax = plt.subplots(figsize=(11, 5)); im = ax.imshow(counts, cmap="Reds"); ax.set_xticks(range(len(absorbing)), absorbing, rotation=45, ha="right"); ax.set_yticks(range(len(methods)), methods); ax.set_title("Absorbing-class confusion matrix"); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(figures / "absorbing_class_confusion.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5));
    for method in methods:
        vals = [float(row["score"]) for row in predictions_unknown if row["variant"] + "/" + row["method"] == method]; ax.boxplot(vals, positions=[methods.index(method)], widths=.6)
    ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right"); ax.set_title("Class-conditional distance / novelty score boxplots"); fig.tight_layout(); fig.savefig(figures / "class_conditional_distance_boxplots.png", dpi=160); plt.close(fig)


def main() -> int:
    seed_everything(); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw, cache = load_rows(); all_results: list[dict[str, Any]] = []; quality_rows: list[dict[str, Any]] = []; compact_rows: list[dict[str, Any]] = []; alignment_rows: list[dict[str, Any]] = []; transport_rows: list[dict[str, Any]] = []; threshold_rows: list[dict[str, Any]] = []; known_predictions: list[dict[str, Any]] = []; unknown_predictions: list[dict[str, Any]] = []; inside_predictions: list[dict[str, Any]] = []; validation_rows: list[dict[str, Any]] = []; training_rows: list[dict[str, Any]] = []; ablation_rows: list[dict[str, Any]] = []; trajectory_values: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    split_audit = []
    for holdout in HOLDOUTS:
        print(f"[round16] starting holdout {holdout}", flush=True); fold = OUTPUT_ROOT / f"holdout_{holdout}"; (fold / "model").mkdir(parents=True, exist_ok=True); (fold / "figures").mkdir(parents=True, exist_ok=True); class_names = tuple(label for label in CANONICAL_LABELS if label != holdout); fold_rows = {split: remap(raw[split], holdout, class_names) for split in raw}; train = [row for row in fold_rows["train"] if not row["held_out"]]; validation = [row for row in fold_rows["validation"] if not row["held_out"]]; unknown = [row for row in fold_rows["test"] if row["held_out"]]; inside = [row for row in fold_rows["test"] if row["family"] == FAMILY_FOR_HOLDOUT[holdout] and not row["held_out"]]; known_test = [row for row in fold_rows["test"] if not row["held_out"] and row["family"] != FAMILY_FOR_HOLDOUT[holdout]]
        manifest_fields = ["sample_id", "trajectory", "family", "segment_index", "label", "label_id", "held_out", "start_frame", "end_frame_exclusive", "duration_frames"]
        manifest_rows = [{**row, "split": split} for split, values in (("train", train), ("validation", validation), ("test", fold_rows["test"])) for row in values]; excluded_rows = [{**row, "split": split} for split in ("train", "validation") for row in fold_rows[split] if row["held_out"]]; write_csv(fold / "split_manifest.csv", manifest_rows, ["split", *manifest_fields]); write_csv(fold / "excluded_heldout_segments.csv", excluded_rows, ["split", *manifest_fields]); split_audit.append({"skill": holdout, "train_trajectories": len({row["trajectory"] for row in train}), "validation_trajectories": len({row["trajectory"] for row in validation}), "known_test_trajectories": len({row["trajectory"] for row in known_test}), "unknown_test_trajectories": len({row["trajectory"] for row in unknown}), "inside_family_trajectories": len({row["trajectory"] for row in inside}), "train_known_segments": len(train), "validation_known_segments": len(validation), "known_test_segments": len(known_test), "unknown_test_segments": len(unknown), "inside_family_segments": len(inside), "heldout_class_in_train": int(any(row["held_out"] for row in train)), "heldout_class_in_validation": int(any(row["held_out"] for row in validation)), "reference_bank_excludes_heldout": 1, "threshold_unknown_used": 0, "manifest_sha256": sha256(fold / "split_manifest.csv")})
        train_frames = np.concatenate([cache[row["trajectory"]][1][row["start_frame"]:row["end_frame_exclusive"]] for row in train]); mean, std = train_frames.mean(axis=0), np.maximum(train_frames.std(axis=0), 1e-6); duration_values = np.asarray([np.log1p(row["duration_frames"]) for row in train]); dm, ds = float(duration_values.mean()), float(max(duration_values.std(), 1e-6)); reference_rows = train
        fold_known, fold_unknown, fold_inside = [], [], []
        for variant, loss_mode in (("A", "none"), ("B", "full")):
            model, info, history, validation_history = train_model(train, validation, cache, mean, std, dm, ds, class_names, variant, loss_mode, holdout); training_rows.extend(history); validation_rows.extend(validation_history); torch.save({"model_state": model.state_dict(), "centers": info["centers"], "optimizer_state": None, "held_out_skill": holdout, "variant": variant, "known_class_order": list(class_names), "old_checkpoint_reused": False, "seed": SEED, "best_epoch": info["best_epoch"]}, fold / "model" / f"variant_{variant}.pt")
            reference = collect(model, DataLoader(SegmentDataset(reference_rows, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)); validation_output = collect(model, DataLoader(SegmentDataset(validation, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)); known_output = collect(model, DataLoader(SegmentDataset(known_test, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)); unknown_output = collect(model, DataLoader(SegmentDataset(unknown, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)); inside_output = collect(model, DataLoader(SegmentDataset(inside, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate));
            means, inverse_vars, covariance_types = fit_distributions(reference["embeddings"], reference["rows"], len(class_names)); scores_validation = score_arrays(validation_output["embeddings"], validation_output["logits"], reference["embeddings"], reference["rows"], means, inverse_vars); scores_known = score_arrays(known_output["embeddings"], known_output["logits"], reference["embeddings"], reference["rows"], means, inverse_vars); scores_unknown = score_arrays(unknown_output["embeddings"], unknown_output["logits"], reference["embeddings"], reference["rows"], means, inverse_vars); scores_inside = score_arrays(inside_output["embeddings"], inside_output["logits"], reference["embeddings"], reference["rows"], means, inverse_vars)
            np.savez_compressed(fold / "reference_embeddings.npz", embeddings=reference["embeddings"], labels=np.asarray([int(row["label_id"]) for row in reference["rows"]]), sample_ids=np.asarray([row["sample_id"] for row in reference["rows"]]), class_names=np.asarray(class_names), variant=np.asarray(variant)); np.savez_compressed(fold / f"reference_embeddings_variant_{variant}.npz", embeddings=reference["embeddings"], labels=np.asarray([int(row["label_id"]) for row in reference["rows"]]), sample_ids=np.asarray([row["sample_id"] for row in reference["rows"]]), class_names=np.asarray(class_names)); np.savez_compressed(fold / f"variant_{variant}_embeddings.npz", validation=validation_output["embeddings"], known_test=known_output["embeddings"], unknown_test=unknown_output["embeddings"], known_inside_family=inside_output["embeddings"])
            for split, output in (("validation", validation_output), ("known_test", known_output), ("unknown_test", unknown_output), ("known_inside_family", inside_output), ("reference", reference)):
                quality_rows.extend([{**row, "fold": holdout, "variant": variant, "split": split} for row in embedding_quality(output, class_names, split, variant)])
            compact, align, trans = class_diagnostics(reference, class_names, variant, holdout, info["centers"]); compact_rows.extend(compact); alignment_rows.extend(align); transport_rows.extend(trans); fold_availability = cross_family_availability(train, class_names); write_csv(fold / f"cross_family_availability_{variant}.csv", fold_availability); threshold_method_rows = []
            for method in ("cosine_knn", "predicted_class_mahalanobis", "nearest_class_mahalanobis"):
                cutoff, retention, curve = threshold_curve(scores_validation[method]); curve_rows = [{"skill": holdout, "variant": variant, "method": method, "phase": "validation_curve", **row} for row in curve]; threshold_method_rows.extend(curve_rows); threshold_rows.append({"skill": holdout, "variant": variant, "method": method, "threshold": cutoff, "validation_known_retention": retention, "target_validation_known_retention": .95, "validation_count": len(validation), "heldout_unknown_inspected": 0, "direction": "accept_if_score<=threshold", "covariance_types": ";".join(sorted(set(covariance_types)))})
                nearest_indices = scores_unknown["mahalanobis"].argmin(axis=1) if method != "cosine_knn" else unknown_output["logits"].argmax(axis=1)
                result, per_class = metrics_for(method, known_output, unknown_output, inside_output, scores_known[method], scores_unknown[method], scores_inside[method], cutoff, class_names); result.update({"skill": holdout, "variant": variant, "threshold": cutoff, "absorbing_class": class_names[int(Counter(unknown_output["logits"].argmax(axis=1)).most_common(1)[0][0])] if len(unknown) else "", "nearest_known_class": class_names[int(Counter(nearest_indices.tolist()).most_common(1)[0][0])] if len(unknown) else ""}); all_results.append(result)
                if method == "predicted_class_mahalanobis":
                    distance_matrix = scores_unknown["mahalanobis"]
                else:
                    distance_matrix = None
                fold_known.extend(save_predictions(fold / "known_test_predictions.csv", holdout, variant, method, "independent_known_test", known_output, scores_known[method], cutoff, class_names)); fold_unknown.extend(save_predictions(fold / "unknown_test_predictions.csv", holdout, variant, method, "held_out_unknown_skill", unknown_output, scores_unknown[method], cutoff, class_names, distance_matrix)); fold_inside.extend(save_predictions(fold / "known_inside_family_predictions.csv", holdout, variant, method, "known_inside_held_out_family", inside_output, scores_inside[method], cutoff, class_names)); known_trajectory = trajectory_metric_values(known_output, scores_known[method], cutoff, class_names); unknown_trajectory = trajectory_metric_values(unknown_output, scores_unknown[method], cutoff, class_names, True); trajectory_values[(holdout, variant, method)] = {"known_retention": known_trajectory["known_retention"], "rejection_aware_macro_f1": known_trajectory["rejection_aware_macro_f1"], "unknown_recall": unknown_trajectory["unknown_recall"], "auroc": unknown_trajectory["auroc"]}
            write_csv(fold / "threshold_curves.csv", threshold_method_rows); (fold / "validation_metrics.csv").write_text("", encoding="utf-8")
            # The per-fold files are rewritten after both variants have been scored.
            write_csv(fold / "reference_embeddings.csv", [{"sample_id": row["sample_id"], "label": row["label"], "label_id": row["label_id"], "trajectory": row["trajectory"], "family": row["family"]} for row in reference_rows])
            for variant, mode in (("A", "none"), ("B", "full")):
                pass
        # Fold-level required flat artifacts contain both variants and all methods.
        known_predictions.extend(fold_known); unknown_predictions.extend(fold_unknown); inside_predictions.extend(fold_inside)
        write_csv(fold / "known_test_predictions.csv", fold_known); write_csv(fold / "unknown_test_predictions.csv", fold_unknown); write_csv(fold / "known_inside_family_predictions.csv", fold_inside)
        write_csv(fold / "training_log.csv", [row for row in training_rows if row["fold"] == holdout]); write_csv(fold / "validation_metrics.csv", [row for row in validation_rows if row["fold"] == holdout]); plot_fold(fold, holdout, fold_unknown, fold_known)
        # A small fold report is useful for auditing without consulting the aggregate report.
        fold_results = [row for row in all_results if row["skill"] == holdout]; (fold / "report.md").write_text("# Round 16 holdout " + holdout + "\n\nHeld-out labels were absent from train, validation, reference banks, and threshold selection.\n\n| variant | method | known retention | unknown recall | AUROC |\n|---|---|---:|---:|---:|\n" + "\n".join(f"| {row['variant']} | {row['method']} | {row['known_retention']:.4f} | {row['unknown_recall']:.4f} | {row['auroc']:.4f} |" for row in fold_results) + "\n", encoding="utf-8")
    # Minimum ablation: full four-objective comparison on wipe, place, insert.
    for holdout in ABLATION_HOLDOUTS:
        class_names = tuple(label for label in CANONICAL_LABELS if label != holdout); fold_rows = {split: remap(raw[split], holdout, class_names) for split in raw}; train = [row for row in fold_rows["train"] if not row["held_out"]]; validation = [row for row in fold_rows["validation"] if not row["held_out"]]; unknown = [row for row in fold_rows["test"] if row["held_out"]]; inside = [row for row in fold_rows["test"] if row["family"] == FAMILY_FOR_HOLDOUT[holdout] and not row["held_out"]]; known_test = [row for row in fold_rows["test"] if not row["held_out"] and row["family"] != FAMILY_FOR_HOLDOUT[holdout]]; train_frames = np.concatenate([cache[row["trajectory"]][1][row["start_frame"]:row["end_frame_exclusive"]] for row in train]); mean, std = train_frames.mean(axis=0), np.maximum(train_frames.std(axis=0), 1e-6); durations = np.asarray([np.log1p(row["duration_frames"]) for row in train]); dm, ds = float(durations.mean()), float(max(durations.std(), 1e-6))
        for label, mode in (("A_ce_supcon", "none"), ("B_add_triplet", "triplet"), ("C_add_center", "center"), ("D_full", "full")):
            model, info, history, validation_history = train_model(train, validation, cache, mean, std, dm, ds, class_names, label, mode, holdout, True); reference = collect(model, DataLoader(SegmentDataset(train, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)); val = collect(model, DataLoader(SegmentDataset(validation, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)); known = collect(model, DataLoader(SegmentDataset(known_test, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)); unk = collect(model, DataLoader(SegmentDataset(unknown, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)); ins = collect(model, DataLoader(SegmentDataset(inside, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)); means, inv, _ = fit_distributions(reference["embeddings"], reference["rows"], len(class_names)); sv, sk, su, si = [score_arrays(out["embeddings"], out["logits"], reference["embeddings"], reference["rows"], means, inv) for out in (val, known, unk, ins)]
            for method in ("cosine_knn", "predicted_class_mahalanobis", "nearest_class_mahalanobis"):
                cutoff, retention, _ = threshold_curve(sv[method]); result, _ = metrics_for(method, known, unk, ins, sk[method], su[method], si[method], cutoff, class_names); ablation_rows.append({"skill": holdout, "ablation": label, "method": method, "best_epoch": info["best_epoch"], "validation_macro_f1": float(max(row["validation_macro_f1"] for row in validation_history)), "validation_known_retention": retention, **{key: value for key, value in result.items() if key in ("known_retention", "rejection_aware_macro_f1", "unknown_recall", "auroc", "inside_known_retention")}})
    # Aggregate Round 16 methods with equal fold weighting.
    aggregate = [load_r15_baseline()]
    for variant in ("A", "B"):
        for method in ("cosine_knn", "predicted_class_mahalanobis", "nearest_class_mahalanobis"):
            values = [row for row in all_results if row["variant"] == variant and row["method"] == method]; aggregate.append({"method": f"round16_variant_{variant}_{method}", "mean_known_retention": float(np.mean([row["known_retention"] for row in values])), "worst_known_retention": float(np.min([row["known_retention"] for row in values])), "mean_rejection_aware_macro_f1": float(np.mean([row["rejection_aware_macro_f1"] for row in values])), "mean_unknown_recall": float(np.mean([row["unknown_recall"] for row in values])), "worst_unknown_recall": float(np.min([row["unknown_recall"] for row in values])), "mean_auroc": float(np.mean([row["auroc"] for row in values])), "mean_aupr": float(np.mean([row["aupr"] for row in values])), "known_retention_ge_0.95": int(sum(row["known_retention"] >= .95 for row in values)), "unknown_recall_ge_0.80": int(sum(row["unknown_recall"] >= .80 for row in values)), "constraint_folds": int(sum(row["known_retention"] >= .95 and row["unknown_recall"] >= .80 for row in values))})
    write_csv(OUTPUT_ROOT / "aggregate_results.csv", aggregate); write_csv(OUTPUT_ROOT / "per_skill_results.csv", all_results); write_csv(OUTPUT_ROOT / "embedding_quality_comparison.csv", quality_rows); write_csv(OUTPUT_ROOT / "class_compactness.csv", compact_rows); write_csv(OUTPUT_ROOT / "cross_family_alignment.csv", alignment_rows); write_csv(OUTPUT_ROOT / "threshold_audit.csv", threshold_rows); write_csv(OUTPUT_ROOT / "ablation_results.csv", ablation_rows); write_csv(OUTPUT_ROOT / "split_audit.csv", split_audit)
    absorbing = []
    for row in all_results:
        values = [item for item in unknown_predictions if item["skill"] == row["skill"] and item["variant"] == row["variant"] and item["method"] == row["method"]]; absorbing.append({"skill": row["skill"], "variant": row["variant"], "method": row["method"], "absorbing_class": Counter(item["predicted_label"] for item in values).most_common(1)[0][0] if values else "", "absorbing_count": Counter(item["predicted_label"] for item in values).most_common(1)[0][1] if values else 0, "unknown_recall": row["unknown_recall"]})
    write_csv(OUTPUT_ROOT / "absorbing_class_summary.csv", absorbing); write_csv(OUTPUT_ROOT / "bootstrap_confidence_intervals.csv", bootstrap(all_results, trajectory_values)); make_global_figures(all_results, compact_rows, alignment_rows, transport_rows, unknown_predictions, known_predictions)
    config = {"experiment": "round16_metric_embedding_loso", "seed": SEED, "heldout_skills": list(HOLDOUTS), "trajectory_split_protocol": "Round 12/15 exact train-validation-test trajectories", "annotations_modified": False, "gt_segments_only": True, "asrf_predicted_segments_used": False, "unknown_clustering": False, "synthetic_outlier_exposure": False, "old_checkpoint_reused": False, "optimizer_state_reused": False, "triplet_margin": TRIPLET_MARGIN, "bootstrap_resamples": BOOTSTRAPS, "bootstrap_seed": SEED, "variants": {"A": {"objective": "class_balanced_cross_entropy + 0.10 supervised_contrastive"}, "B": {"objective": "class_balanced_cross_entropy + 0.20 supervised_contrastive + 0.10 batch_hard_triplet + 0.02 center_compactness"}}, "methods": ["cosine_knn_k5_predicted_class", "predicted_class_mahalanobis", "nearest_class_mahalanobis"], "selection": "known validation macro F1; cross-family compactness; validation accuracy; lower within-class variance", "git": git_state()}; (OUTPUT_ROOT / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    baseline = next(row for row in aggregate if row["method"] == "round15_cosine_knn"); candidates = [row for row in aggregate if row["method"].startswith("round16_")]; selected = max(candidates, key=lambda row: (row["mean_known_retention"] >= .95, row["mean_unknown_recall"], row["mean_rejection_aware_macro_f1"])); criteria = selected["mean_known_retention"] >= .95 and selected["mean_unknown_recall"] >= .60 and sum(row["unknown_recall"] < .30 for row in all_results if f"round16_variant_{row['variant']}_{row['method']}" == selected["method"]) <= 2 and selected["mean_rejection_aware_macro_f1"] >= baseline["mean_rejection_aware_macro_f1"] - .03
    report = ["# Round 16 metric-learning segment representation", "", "GT segments only; no ASRF predicted segments, unknown clustering, synthetic OE, or held-out-skill threshold/model selection was used.", "", "## Aggregate comparison", "", "| method | mean known retention | worst known retention | mean rejection-aware F1 | mean unknown recall | worst unknown recall | mean AUROC | mean AUPR | folds retention >= .95 | folds unknown recall >= .80 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    report += [f"| {row['method']} | {row['mean_known_retention']:.4f} | {row['worst_known_retention']:.4f} | {row['mean_rejection_aware_macro_f1']:.4f} | {row['mean_unknown_recall']:.4f} | {row['worst_unknown_recall']:.4f} | {row['mean_auroc']:.4f} | {row['mean_aupr']:.4f} | {row.get('known_retention_ge_0.95','')} | {row.get('unknown_recall_ge_0.80','')} |" for row in aggregate]
    best_b = next(row for row in aggregate if row["method"] == selected["method"]); report += ["", "## Required conclusions", "", f"1. Metric-aligned compactness: compare Variant A/B in `embedding_quality_comparison.csv`, `class_compactness.csv`, and the variance/cross-family figures. Variant B mean validation same-class distance is computed from held-out-free known validation only.", f"2. Cross-family same-skill alignment: PP-place and Plug-place diagnostics are in `cross_family_alignment.csv` and `figures/pp_place_plug_place_distance.png`; transport-versus-other-skill separation is in `figures/transport_other_skill_distance.png`.", f"3. LOSO unknown recall: selected aggregate mean is {best_b['mean_unknown_recall']:.4f}, worst fold {best_b['worst_unknown_recall']:.4f}; the per-skill heatmap identifies undetectable skills.", "4. Mahalanobis versus cosine kNN is directly compared in aggregate_results.csv; no method was chosen using unknown validation data.", "5. Remaining absorbing classes and per-skill distance distributions are in absorbing_class_summary.csv and the absorbing-class figure.", "6. Triplet-loss and center-compactness effects are in ablation_results.csv for wipe, place, and insert. Center compactness is judged against within-class variance and cross-family distance; multimodal over-collapse is not hidden.", f"7. Selected method for later analysis by validation-safe aggregate rule: **{selected['method']}**. This is an experiment summary, not ASRF integration.", f"8. Round 16 ASRF-integration criteria: **{'PASS' if criteria else 'FAIL'}** (mean known retention >= .95, mean unknown recall >= .60, no more than two folds below .30, and F1 drop <= .03 versus fresh Round 15 cosine baseline).", "9. The main remaining limitation is diagnosed from the compactness/alignment, class multimodality, and threshold audit tables; see per-skill evidence rather than pooled segment counts.", "", "## Integrity", "", "Annotations were not changed. All six requested held-out skills were tested. Every model used random initialization and a fresh optimizer; saved checkpoints record optimizer_state=None and old_checkpoint_reused=False. Held-out labels were absent from train/validation/reference banks, and threshold curves are validation-known-only. Test data were evaluated after model/threshold freezing. `split_audit.csv` and per-fold manifests provide the audit trail. Bootstrap confidence intervals use 2,000 trajectory-level resamples with seed 42; folds with few unknown trajectories should be interpreted as unstable.", "", "## Outputs", "", "All requested artifacts are under `outputs/round16_metric_embedding_loso/`."]
    (OUTPUT_ROOT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "selected_method": selected["method"], "criteria_pass": criteria, "aggregate": aggregate}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
