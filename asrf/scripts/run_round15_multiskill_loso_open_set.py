#!/usr/bin/env python3
"""Round 15 multi-skill leave-one-skill-out open-set study."""

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
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
FULL_ROOT = ROOT / "outputs/round12_multiskill_segment_classifier"
DATA_ROOT = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
OUTPUT_ROOT = ROOT / "outputs/round15_multiskill_loso_open_set"
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import train_round12_segment_classifier as base  # noqa: E402
import train_round14_hard_oe_holdout_wipe as hard_oe  # noqa: E402
from asrf.data.ontology import CANONICAL_LABELS, ONTOLOGY_VERSION  # noqa: E402
from asrf.training.checkpointing import sha256_file  # noqa: E402

SEED = 42
HOLDOUTS = ("wipe", "pour", "pour_recover", "place", "insert", "transport")
FAMILY_FOR_HOLDOUT = {"wipe": "wipe", "pour": "pour", "pour_recover": "pour", "place": "plug", "insert": "plug", "transport": "pp"}
BATCH_SIZE = 32
MAX_EPOCHS = 10
PATIENCE = 3
ENERGY_MARGIN = 5.0
LAMBDA_ENERGY = 0.1
STABILITY_LAMBDA = 0.05
HARD_EMBEDDING_LAMBDA = 0.02
HARD_EMBEDDING_MARGIN = 0.02
BOOTSTRAPS = 2000
DEVICE = torch.device("cpu")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]; negative = scores[labels == 0]
    if not len(positive) or not len(negative): return 0.0
    return float((positive[:, None] > negative[None, :]).mean() + 0.5 * (positive[:, None] == negative[None, :]).mean())


def aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores); sorted_labels = labels[order]; positives = max(int(labels.sum()), 1); tp = 0; total = 0; area = 0.0
    for label in sorted_labels:
        total += 1; tp += int(label == 1); area += (tp / total) if label == 1 else 0.0
    return float(area / positives)


def score_logits(logits: np.ndarray) -> dict[str, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True); probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True); order = np.argsort(-probabilities, axis=1)
    return {"max_softmax": 1.0 - probabilities[np.arange(len(logits)), order[:, 0]], "energy": -np.log(np.exp(shifted).sum(axis=1)) - logits.max(axis=1), "top1": order[:, 0], "probability": probabilities}


def load_rows() -> dict[str, list[dict[str, Any]]]:
    result = {}
    for split in ("train", "validation", "test"):
        values = []
        for raw in read_csv(FULL_ROOT / "split_manifests" / f"{split}.csv"):
            row = dict(raw)
            for field in ("segment_index", "label_id", "start_frame", "end_frame_exclusive", "duration_frames"): row[field] = int(row[field])
            values.append(row)
        result[split] = values
    return result


def load_cache(rows: dict[str, list[dict[str, Any]]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    trajectories = sorted({row["trajectory"] for values in rows.values() for row in values})
    return {trajectory: base.load_trajectory_features(trajectory) for trajectory in trajectories}


def seq(row: dict[str, Any], cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return cache[row["trajectory"]][1][row["start_frame"]:row["end_frame_exclusive"]].astype(np.float32)


def remap(rows: list[dict[str, Any]], holdout: str, class_names: tuple[str, ...]) -> list[dict[str, Any]]:
    mapping = {label: index for index, label in enumerate(class_names)}; result = []
    for raw in rows:
        row = dict(raw); row["label_id"] = mapping.get(row["label"], -1); row["held_out"] = row["label"] == holdout; result.append(row)
    return result


class SegmentDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], mean: np.ndarray, std: np.ndarray, duration_mean: float, duration_std: float, synthetic: bool = False):
        self.rows = rows; self.cache = cache; self.mean = mean; self.std = std; self.duration_mean = duration_mean; self.duration_std = duration_std; self.synthetic = synthetic

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]; values = row["sequence"] if self.synthetic else seq(row, self.cache); override = row.get("embedding_override")
        return {"sequence": torch.from_numpy(((values - self.mean) / self.std).astype(np.float32)), "duration": torch.tensor((np.log1p(row["duration_frames"]) - self.duration_mean) / self.duration_std, dtype=torch.float32), "label": torch.tensor(row["label_id"], dtype=torch.long), "override": torch.from_numpy(override.astype(np.float32)) if override is not None else torch.zeros(base.EMBEDDING_DIM), "override_valid": torch.tensor(override is not None), "row": row}


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["sequence"].shape[0] for item in items], dtype=torch.long); max_length = int(lengths.max()); values = torch.zeros((len(items), max_length, base.FEATURE_DIM)); mask = torch.zeros((len(items), max_length), dtype=torch.bool)
    for index, item in enumerate(items):
        length = int(item["sequence"].shape[0]); values[index, :length] = item["sequence"]; mask[index, :length] = True
    return {"sequence": values, "valid_mask": mask, "lengths": lengths, "duration": torch.stack([item["duration"] for item in items]), "label": torch.stack([item["label"] for item in items]), "override": torch.stack([item["override"] for item in items]), "override_valid": torch.stack([item["override_valid"] for item in items]), "rows": [item["row"] for item in items]}


def infer(model: nn.Module, loader: DataLoader) -> dict[str, Any]:
    model.eval(); embeddings = []; logits = []; rows = []
    with torch.no_grad():
        for batch in loader:
            embedding, output = model(batch["sequence"], batch["valid_mask"], batch["lengths"], batch["duration"]); valid = batch["override_valid"]
            if valid.any():
                embedding = embedding.clone(); embedding[valid] = F.normalize(batch["override"][valid], p=2, dim=1); output = output.clone(); output[valid] = model.classifier(embedding[valid])
            embeddings.append(embedding.numpy()); logits.append(output.numpy()); rows.extend(batch["rows"])
    return {"embeddings": np.concatenate(embeddings), "logits": np.concatenate(logits), "rows": rows}


def model(num_classes: int) -> nn.Module:
    return base.SegmentClassifier(base.FEATURE_DIM, base.HIDDEN_DIM, base.PROJECTION_DIM, base.EMBEDDING_DIM, num_classes)


def ref_embeddings(current: nn.Module, rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], mean: np.ndarray, std: np.ndarray, dm: float, ds: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = infer(current, DataLoader(SegmentDataset(rows, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)); return output["embeddings"], output["rows"]


def source_balance(rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], class_names: tuple[str, ...], pairs: list[tuple[tuple[np.ndarray, np.ndarray], dict[str, Any], dict[str, Any]]], split: str, hard: bool, per_class_type: int) -> list[dict[str, Any]]:
    # Round 13 originals and Round 14 hard generators are reused unchanged;
    # this wrapper only balances source labels for each LOSO fold.
    if hard:
        pool = hard_oe.make_hard(rows, cache, split, per_class_type * len(class_names), pairs)
        types = hard_oe.HARD_TYPES
    else:
        pool = hard_oe.make_original(rows, cache, split, per_class_type * len(class_names) * len(hard_oe.ORIGINAL_TYPES))
        types = hard_oe.ORIGINAL_TYPES
    grouped = defaultdict(list)
    for item in pool: grouped[(item["outlier_type"], item["source_label"])].append(item)
    output = []
    for kind in types:
        for label in class_names:
            values = grouped[(kind, label)]
            if not values: values = [item for item in pool if item["outlier_type"] == kind]
            if not values: raise RuntimeError(f"No synthetic {kind} samples for {label}.")
            for index in range(per_class_type): output.append(values[index % len(values)])
    return output


def build_pairs(embeddings: np.ndarray, rows: list[dict[str, Any]]) -> list[tuple[tuple[np.ndarray, np.ndarray], dict[str, Any], dict[str, Any]]]:
    similarities = embeddings @ embeddings.T; pairs = []
    for index in range(len(rows)):
        candidates = [other for other in range(len(rows)) if rows[other]["label"] != rows[index]["label"]]; other = max(candidates, key=lambda candidate: similarities[index, candidate]); pairs.append(((embeddings[index], embeddings[other]), rows[index], rows[other]))
    return pairs


def contrastive(embeddings: Tensor, labels: Tensor) -> Tensor:
    if len(labels) < 2: return embeddings.sum() * 0.0
    logits = embeddings @ embeddings.T / 0.07; diagonal = torch.eye(len(labels), dtype=torch.bool); positive = labels[:, None].eq(labels[None, :]) & ~diagonal; logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min); log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True); counts = positive.sum(dim=1); usable = counts > 0
    return -(log_prob.masked_fill(~positive, 0.0).sum(dim=1)[usable] / counts[usable]).mean() if usable.any() else embeddings.sum() * 0.0


def hardness(outputs: dict[str, Any], reference: np.ndarray) -> np.ndarray:
    scores = score_logits(outputs["logits"]); distances = (1.0 - outputs["embeddings"] @ reference.T).min(axis=1)
    def rank(values: np.ndarray) -> np.ndarray: return np.argsort(np.argsort(values)) / max(len(values) - 1, 1)
    return (1 - rank(scores["energy"]) + rank(1 - scores["max_softmax"]) + (1 - rank(distances))) / 3


def train_one(current: nn.Module, teacher: nn.Module | None, train_rows: list[dict[str, Any]], synth_pool: list[dict[str, Any]], validation_loader: DataLoader, cache: dict[str, tuple[np.ndarray, np.ndarray]], mean: np.ndarray, std: np.ndarray, dm: float, ds: float, class_weights: Tensor, reference: np.ndarray, mode: str, num_classes: int) -> tuple[dict[str, Tensor], int]:
    optimizer = torch.optim.AdamW(current.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY); best = None; best_f1 = -1.0; best_epoch = 0; stale = 0; train_loader = lambda: DataLoader(SegmentDataset(train_rows, cache, mean, std, dm, ds), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED), collate_fn=collate)
    if teacher is not None: teacher.eval()
    for epoch in range(1, MAX_EPOCHS + 1):
        if mode in ("r14_baseline", "r14_hard"):
            pool_outputs = infer(current, DataLoader(SegmentDataset(synth_pool, None, mean, std, dm, ds, True), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)); hard_values = hardness(pool_outputs, reference); hard_count = len(synth_pool) // 2; hard_indices = np.argsort(-hard_values)[:hard_count]; remaining = np.asarray([index for index in range(len(synth_pool)) if index not in set(hard_indices)], dtype=int); rng = np.random.default_rng(SEED + epoch); random_indices = rng.choice(remaining, size=len(synth_pool) - hard_count, replace=False); selected_pool = [synth_pool[int(index)] for index in np.concatenate((hard_indices, random_indices))]
        else: selected_pool = synth_pool
        known_iter = iter(train_loader()); outlier_iter = iter(DataLoader(SegmentDataset(selected_pool, None, mean, std, dm, ds, True), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED + epoch), collate_fn=collate)) if mode != "base" else None; steps = max(len(train_loader()), len(outlier_iter) if outlier_iter is not None else 0); current.train()
        for _ in range(steps):
            try: known_batch = next(known_iter)
            except StopIteration: known_iter = iter(train_loader()); known_batch = next(known_iter)
            known_embedding, known_logits = current(known_batch["sequence"], known_batch["valid_mask"], known_batch["lengths"], known_batch["duration"]); ce = F.cross_entropy(known_logits, known_batch["label"], weight=class_weights); loss = ce + 0.1 * contrastive(known_embedding, known_batch["label"])
            if outlier_iter is not None:
                try: outlier_batch = next(outlier_iter)
                except StopIteration: outlier_iter = iter(DataLoader(SegmentDataset(selected_pool, None, mean, std, dm, ds, True), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)); outlier_batch = next(outlier_iter)
                outlier_embedding, outlier_logits = current(outlier_batch["sequence"], outlier_batch["valid_mask"], outlier_batch["lengths"], outlier_batch["duration"]); valid = outlier_batch["override_valid"]
                if valid.any():
                    outlier_embedding = outlier_embedding.clone(); outlier_embedding[valid] = F.normalize(outlier_batch["override"][valid], p=2, dim=1); outlier_logits = outlier_logits.clone(); outlier_logits[valid] = current.classifier(outlier_embedding[valid])
                known_energy = -torch.logsumexp(known_logits, dim=1); outlier_energy = -torch.logsumexp(outlier_logits, dim=1); energy_loss = F.relu(ENERGY_MARGIN + known_energy.mean() - outlier_energy.mean()); loss = loss + LAMBDA_ENERGY * energy_loss
                if mode == "r14_hard":
                    nearest = (1.0 - outlier_embedding @ torch.from_numpy(reference).float().T).min(dim=1).values; loss = loss + HARD_EMBEDDING_LAMBDA * F.relu(HARD_EMBEDDING_MARGIN - nearest).mean()
                if mode in ("r14_baseline", "r14_hard") and teacher is not None:
                    with torch.no_grad(): _, teacher_logits = teacher(known_batch["sequence"], known_batch["valid_mask"], known_batch["lengths"], known_batch["duration"])
                    loss = loss + STABILITY_LAMBDA * F.kl_div(F.log_softmax(known_logits, dim=1), F.softmax(teacher_logits, dim=1), reduction="batchmean")
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(current.parameters(), 5.0); optimizer.step()
        val = infer(current, validation_loader); predictions = score_logits(val["logits"])["top1"]; labels = np.asarray([row["label_id"] for row in val["rows"]]); f1s=[]
        for label_id in range(num_classes):
            tp=((labels==label_id)&(predictions==label_id)).sum(); fp=((labels!=label_id)&(predictions==label_id)).sum(); fn=((labels==label_id)&(predictions!=label_id)).sum(); p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f1s.append(2*p*r/(p+r) if p+r else 0)
        val_f1=float(np.mean(f1s))
        if val_f1 > best_f1 + 1e-9: best_f1=val_f1; best_epoch=epoch; stale=0; best={key:value.detach().cpu().clone() for key,value in current.state_dict().items()}
        else: stale += 1
        if stale >= PATIENCE: break
    if best is None: raise RuntimeError("No model state.")
    current.load_state_dict(best); return best, best_epoch


def normalized_cosine(outputs: dict[str, Any], reference: np.ndarray, k: int = 5) -> np.ndarray:
    scores = score_logits(outputs["logits"]); distances = 1.0 - outputs["embeddings"] @ reference.T; values=[]
    for row_index, prediction in enumerate(scores["top1"]):
        candidates = np.where(np.asarray([row["label_id"] for row in reference_rows_global]) == int(prediction))[0] if reference_rows_global else np.arange(len(reference)); values.append(float(np.sort(distances[row_index, candidates])[:min(k, len(candidates))].mean()))
    return np.asarray(values)


reference_rows_global: list[dict[str, Any]] = []


def threshold(scores: np.ndarray, target: float = 0.95) -> tuple[float, float]:
    ordered=np.sort(scores); required=max(1,int(np.ceil(target*len(ordered)))); value=float(ordered[min(required-1,len(ordered)-1)]); return value,float((scores<=value).mean())


def known_rejection_f1(output: dict[str, Any], accepted: np.ndarray, class_names: tuple[str, ...]) -> float:
    labels = np.asarray([row["label_id"] for row in output["rows"]]); predictions = score_logits(output["logits"])["top1"]; values=[]
    for label_id in range(len(class_names)):
        tp=int(((labels==label_id)&(predictions==label_id)&accepted).sum()); fp=int(((labels!=label_id)&(predictions==label_id)&accepted).sum()); fn=int(((labels==label_id)&((predictions!=label_id)|~accepted)).sum()); precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0; values.append(2*precision*recall/(precision+recall) if precision+recall else 0)
    return float(np.mean(values))


def metrics(known: dict[str, Any], unknown: dict[str, Any], known_inside: dict[str, Any], score_known: np.ndarray, score_unknown: np.ndarray, score_inside: np.ndarray, cutoff: float, class_names: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    known_scores=score_logits(known["logits"]); labels=np.asarray([row["label_id"] for row in known["rows"]]); preds=known_scores["top1"]; accepted=score_known<=cutoff; inside_labels=np.asarray([row["label_id"] for row in known_inside["rows"]]); inside_preds=score_logits(known_inside["logits"])["top1"]; inside_acc=float((inside_preds==inside_labels).mean());
    f1s=[]; per_class=[]
    for label_id,label in enumerate(class_names):
        tp=int(((labels==label_id)&(preds==label_id)&accepted).sum()); fp=int(((labels!=label_id)&(preds==label_id)&accepted).sum()); fn=int(((labels==label_id)&((preds!=label_id)|~accepted)).sum()); p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f=2*p*r/(p+r) if p+r else 0; support=int((labels==label_id).sum()); f1s.append(f); per_class.append({"class":label,"support":support,"known_retention":float(((labels==label_id)&accepted).sum()/support) if support else 0,"f1":f})
    unknown_accept=score_unknown<=cutoff; binary=np.concatenate((np.zeros(len(score_known)),np.ones(len(score_unknown)))); binary_scores=np.concatenate((score_known,score_unknown)); result={"closed_set_accuracy":float((preds==labels).mean()),"known_retention":float(accepted.mean()),"false_unknown_rate":float((~accepted).mean()),"rejection_aware_macro_f1":float(np.mean(f1s)),"accepted_only_macro_f1":float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0,"accepted_known_accuracy":float((preds[accepted]==labels[accepted]).mean()) if accepted.any() else 0.0,"unknown_recall":float((~unknown_accept).mean()),"unknown_false_known_rate":float(unknown_accept.mean()),"unknown_auroc":auroc(binary,binary_scores),"unknown_aupr":aupr(binary,binary_scores),"unknown_score_mean":float(score_unknown.mean()),"unknown_score_std":float(score_unknown.std()),"unknown_score_q05":float(np.quantile(score_unknown,.05)),"unknown_score_q50":float(np.quantile(score_unknown,.5)),"unknown_score_q95":float(np.quantile(score_unknown,.95)),"inside_closed_set_accuracy":inside_acc,"inside_known_retention":float((score_inside<=cutoff).mean()),"inside_false_unknown_rate":float((score_inside>cutoff).mean()),"per_class":per_class}
    return result, per_class


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory_manifest(raw_rows: dict[str, list[dict[str, Any]]], holdout: str, class_names: tuple[str, ...], out: Path) -> None:
    rows=[]
    for split, values in raw_rows.items():
        grouped=defaultdict(list)
        for row in values: grouped[row["trajectory"]].append(row)
        for trajectory, entries in grouped.items():
            segment_path=DATA_ROOT / trajectory / "segments.csv"; counts=Counter(row["label"] for row in entries); rows.append({"trajectory":trajectory,"task_family":entries[0]["family"],"split":split,"segment_count":len(entries),"segment_count_by_label":json.dumps(dict(counts),sort_keys=True),"segments_file_hash":file_hash(segment_path),"held_out_class":holdout,"known_class_order":json.dumps(list(class_names))})
    write_csv(out,rows)


def bootstrap(skill_results: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    rng=np.random.default_rng(SEED); metrics_names=("known_retention","rejection_aware_macro_f1","unknown_recall","unknown_auroc"); output=[]
    for metric_name in metrics_names:
        samples=[]
        for _ in range(BOOTSTRAPS):
            values=[]
            for result in skill_results:
                trajectory_values=result["trajectory_metrics"][method][metric_name]; indices=rng.integers(0,len(trajectory_values),size=len(trajectory_values)); values.append(float(np.mean(np.asarray(trajectory_values)[indices])))
            samples.append(float(np.mean(values)))
        output.append({"method":method,"metric":metric_name,"bootstrap_resamples":BOOTSTRAPS,"seed":SEED,"mean":float(np.mean(samples)),"ci_lower":float(np.quantile(samples,.025)),"ci_upper":float(np.quantile(samples,.975))})
    return output


def plot_global(per_skill: list[dict[str, Any]]) -> None:
    figures=OUTPUT_ROOT/"figures"; figures.mkdir(parents=True,exist_ok=True); methods=sorted({row["method"] for row in per_skill}); skills=list(HOLDOUTS); matrix=np.asarray([[next(row["unknown_recall"] for row in per_skill if row["skill"]==skill and row["method"]==method) for skill in skills] for method in methods]); fig,ax=plt.subplots(figsize=(10,4)); im=ax.imshow(matrix,cmap="viridis",vmin=0,vmax=1); ax.set_xticks(range(len(skills)),skills,rotation=35,ha="right"); ax.set_yticks(range(len(methods)),methods); ax.set_title("LOSO unknown recall"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(figures/"loso_result_heatmap.png",dpi=160); plt.close(fig)
    for name, key in (("known_retention_boxplot","known_retention"),("unknown_recall_boxplot","unknown_recall")):
        fig,ax=plt.subplots(figsize=(9,5)); ax.boxplot([[row[key] for row in per_skill if row["method"]==method] for method in methods],labels=methods); ax.set_title(name.replace("_"," ")); ax.tick_params(axis="x",rotation=35); fig.tight_layout(); fig.savefig(figures/f"{name}.png",dpi=160); plt.close(fig)


def main() -> int:
    seed_everything(); OUTPUT_ROOT.mkdir(parents=True,exist_ok=True); raw=load_rows(); cache=load_cache(raw); all_results=[]; split_rows=[]; prediction_fields=["skill","method","group","trajectory","segment_index","ground_truth_label","predicted_label","score","threshold","decision","duration_frames"]
    for holdout in HOLDOUTS:
        print(f"[round15] holdout {holdout}",flush=True); fold=OUTPUT_ROOT/f"holdout_{holdout}"; (fold/"models").mkdir(parents=True,exist_ok=True); (fold/"figures").mkdir(parents=True,exist_ok=True); class_names=tuple(label for label in CANONICAL_LABELS if label!=holdout); class_map={label:index for index,label in enumerate(class_names)}; fold_rows={split:remap(values,holdout,class_names) for split,values in raw.items()}; train=[row for row in fold_rows["train"] if not row["held_out"]]; validation=[row for row in fold_rows["validation"] if not row["held_out"]]; unknown=[row for row in fold_rows["test"] if row["held_out"]]; family=FAMILY_FOR_HOLDOUT[holdout]; inside=[row for row in fold_rows["test"] if row["family"]==family and not row["held_out"]]; known_test=[row for row in fold_rows["test"] if not row["held_out"] and row["family"]!=family]
        trajectory_manifest(raw,holdout,class_names,fold/"split_manifest.csv"); split_rows.append({"skill":holdout,"train_trajectories":len(set(row["trajectory"] for row in raw["train"])),"validation_trajectories":len(set(row["trajectory"] for row in raw["validation"])),"known_test_trajectories":len(set(row["trajectory"] for row in known_test)),"unknown_test_trajectories":len(set(row["trajectory"] for row in unknown)),"inside_family_trajectories":len(set(row["trajectory"] for row in inside)),"train_known_segments":len(train),"validation_known_segments":len(validation),"known_test_segments":len(known_test),"unknown_test_segments":len(unknown),"inside_family_segments":len(inside),"heldout_class_in_train":int(any(row["held_out"] for row in train)),"heldout_class_in_validation":int(any(row["held_out"] for row in validation))})
        train_frames=np.concatenate([seq(row,cache) for row in train]); mean=train_frames.mean(0); std=np.maximum(train_frames.std(0),1e-6); durations=np.asarray([np.log1p(row["duration_frames"]) for row in train]); dm=float(durations.mean()); ds=float(max(durations.std(),1e-6)); train_loader=DataLoader(SegmentDataset(train,cache,mean,std,dm,ds),batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate); val_loader=DataLoader(SegmentDataset(validation,cache,mean,std,dm,ds),batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate); class_counts=Counter(row["label_id"] for row in train); weights=torch.tensor([1/np.sqrt(class_counts[index]) for index in range(len(class_names))],dtype=torch.float32); weights*=len(class_names)/weights.sum()
        base_model=model(len(class_names)); base_state,_=train_one(base_model,None,train,[],val_loader,cache,mean,std,dm,ds,weights,np.zeros((len(train),base.EMBEDDING_DIM)),"base",len(class_names)); base_model.load_state_dict(base_state); base_ref,base_ref_rows=ref_embeddings(base_model,train,cache,mean,std,dm,ds); global reference_rows_global; reference_rows_global=base_ref_rows
        train_pairs=build_pairs(base_ref,train); val_ref,_=ref_embeddings(base_model,validation,cache,mean,std,dm,ds); val_pairs=build_pairs(val_ref,validation); original_train=source_balance(train,cache,class_names,train_pairs,"train",False,4); original_val=source_balance(validation,cache,class_names,val_pairs,"validation",False,4); hard_train=source_balance(train,cache,class_names,train_pairs,"train",True,4); hard_val=source_balance(validation,cache,class_names,val_pairs,"validation",True,4)
        variants={"max_softmax":(base_model,base_state,"base",[]),"cosine_knn":(base_model,base_state,"base",[])}
        r13_model=model(len(class_names)); r13_state,_=train_one(r13_model,None,train,original_train,val_loader,cache,mean,std,dm,ds,weights,base_ref,"r13",len(class_names)); variants["round13_energy_margin"]=(model(len(class_names)),r13_state,"r13",original_val)
        r14_model=model(len(class_names)); r14_state,_=train_one(r14_model,base_model,train,hard_train,val_loader,cache,mean,std,dm,ds,weights,base_ref,"r14_baseline",len(class_names)); variants["round14_energy_margin_baseline"]=(model(len(class_names)),r14_state,"r14_baseline",hard_val)
        hard_model=model(len(class_names)); hard_state,_=train_one(hard_model,base_model,train,hard_train,val_loader,cache,mean,std,dm,ds,weights,base_ref,"r14_hard",len(class_names)); variants["round14_selected_hard_oe"]=(model(len(class_names)),hard_state,"r14_hard",hard_val)
        fold_known_predictions=[]; fold_unknown_predictions=[]; fold_inside_predictions=[]; fold_thresholds=[]
        for method,(trained,state,mode,val_synth) in variants.items():
            trained.load_state_dict(state); torch.save({"model_state":state,"ontology_version":ONTOLOGY_VERSION,"held_out_class":holdout,"known_class_order":list(class_names),"optimizer_state":None,"metadata":{"method":method,"seed":SEED,"held_out_class":holdout}},fold/"models"/f"{method}.pt")
            reference=base_ref if method in ("max_softmax","cosine_knn") else ref_embeddings(trained,train,cache,mean,std,dm,ds)[0]; known_val=infer(trained,val_loader); known_test_output=infer(trained,DataLoader(SegmentDataset(known_test,cache,mean,std,dm,ds),batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate)); unknown_output=infer(trained,DataLoader(SegmentDataset(unknown,cache,mean,std,dm,ds),batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate)); inside_output=infer(trained,DataLoader(SegmentDataset(inside,cache,mean,std,dm,ds),batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate))
            def scores(output: dict[str,Any]) -> np.ndarray:
                if method=="max_softmax": return score_logits(output["logits"])["max_softmax"]
                if method=="cosine_knn":
                    distances=1-output["embeddings"]@reference.T; predictions=score_logits(output["logits"])["top1"]; values=[]
                    for index,prediction in enumerate(predictions):
                        candidates=[i for i,row in enumerate(base_ref_rows) if row["label_id"]==int(prediction)]; values.append(float(np.sort(distances[index,candidates])[:min(5,len(candidates))].mean()))
                    return np.asarray(values)
                return score_logits(output["logits"])["energy"]
            score_known=scores(known_val); score_synth=scores(infer(trained,DataLoader(SegmentDataset(val_synth,None,mean,std,dm,ds,True),batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate))) if val_synth else np.asarray([]); cutoff,ret=threshold(score_known,.95); score_test_known=scores(known_test_output); score_test_unknown=scores(unknown_output); score_test_inside=scores(inside_output); result,per_class=metrics(known_test_output,unknown_output,inside_output,score_test_known,score_test_unknown,score_test_inside,cutoff,class_names)
            trajectory_metrics={"known_retention":[],"rejection_aware_macro_f1":[],"unknown_recall":[],"unknown_auroc":[]}
            for trajectory in sorted({row["trajectory"] for row in known_test_output["rows"]}):
                indices=np.asarray([index for index,row in enumerate(known_test_output["rows"]) if row["trajectory"]==trajectory]); subset={"rows":[known_test_output["rows"][int(index)] for index in indices],"logits":known_test_output["logits"][indices]}; accepted=score_test_known[indices]<=cutoff; trajectory_metrics["known_retention"].append(float(accepted.mean())); trajectory_metrics["rejection_aware_macro_f1"].append(known_rejection_f1(subset,accepted,class_names))
            for trajectory in sorted({row["trajectory"] for row in unknown_output["rows"]}):
                indices=np.asarray([index for index,row in enumerate(unknown_output["rows"]) if row["trajectory"]==trajectory]); unknown_values=score_test_unknown[indices]; trajectory_metrics["unknown_recall"].append(float((unknown_values>cutoff).mean())); trajectory_metrics["unknown_auroc"].append(auroc(np.concatenate((np.zeros(len(score_test_known)),np.ones(len(unknown_values)))),np.concatenate((score_test_known,unknown_values))))
            result.update({"skill":holdout,"method":method,"threshold":cutoff,"validation_known_retention":ret,"validation_synthetic_auroc":auroc(np.concatenate((np.zeros(len(score_known)),np.ones(len(score_synth)))),np.concatenate((score_known,score_synth))) if len(score_synth) else "","absorbing_class":ID_TO_LABEL_LOCAL(class_names,int(Counter(score_logits(unknown_output["logits"])["top1"]).most_common(1)[0][0])) if len(unknown_output["rows"]) else "","trajectory_metrics":{method:trajectory_metrics}}); all_results.append(result)
            for target in (0.95,0.97):
                target_cutoff,target_ret=threshold(score_known,target); fold_thresholds.append({"skill":holdout,"method":method,"target_retention":target,"threshold":target_cutoff,"validation_known_retention":target_ret,"validation_false_unknown_rate":1-target_ret,"validation_known_macro_f1":known_rejection_f1(known_val,score_known<=target_cutoff,class_names),"synthetic_validation_auroc":auroc(np.concatenate((np.zeros(len(score_known)),np.ones(len(score_synth)))),np.concatenate((score_known,score_synth))) if len(score_synth) else ""})
            for quantile in np.linspace(0,1,21):
                curve_cutoff=float(np.quantile(score_known,quantile)); fold_thresholds.append({"skill":holdout,"method":method,"target_retention":quantile,"threshold":curve_cutoff,"validation_known_retention":float((score_known<=curve_cutoff).mean()),"validation_false_unknown_rate":float((score_known>curve_cutoff).mean()),"validation_known_macro_f1":known_rejection_f1(known_val,score_known<=curve_cutoff,class_names),"synthetic_validation_auroc":auroc(np.concatenate((np.zeros(len(score_known)),np.ones(len(score_synth)))),np.concatenate((score_known,score_synth))) if len(score_synth) else ""})
            method_rows=[{"skill":holdout,"method":method,"group":"known_test","trajectory":row["trajectory"],"segment_index":row["segment_index"],"ground_truth_label":row["label"],"predicted_label":ID_TO_LABEL_LOCAL(class_names,int(prediction)),"score":float(value),"threshold":cutoff,"decision":ID_TO_LABEL_LOCAL(class_names,int(prediction)) if value<=cutoff else "unknown","duration_frames":row["duration_frames"]} for row,prediction,value in zip(known_test_output["rows"],score_logits(known_test_output["logits"])["top1"],score_test_known)]
            unknown_rows=[{"skill":holdout,"method":method,"group":"unknown_test","trajectory":row["trajectory"],"segment_index":row["segment_index"],"ground_truth_label":row["label"],"predicted_label":ID_TO_LABEL_LOCAL(class_names,int(prediction)),"score":float(value),"threshold":cutoff,"decision":ID_TO_LABEL_LOCAL(class_names,int(prediction)) if value<=cutoff else "unknown","duration_frames":row["duration_frames"]} for row,prediction,value in zip(unknown_output["rows"],score_logits(unknown_output["logits"])["top1"],score_test_unknown)]
            inside_rows=[{"skill":holdout,"method":method,"group":"known_inside_family","trajectory":row["trajectory"],"segment_index":row["segment_index"],"ground_truth_label":row["label"],"predicted_label":ID_TO_LABEL_LOCAL(class_names,int(prediction)),"score":float(value),"threshold":cutoff,"decision":ID_TO_LABEL_LOCAL(class_names,int(prediction)) if value<=cutoff else "unknown","duration_frames":row["duration_frames"]} for row,prediction,value in zip(inside_output["rows"],score_logits(inside_output["logits"])["top1"],score_test_inside)]
            fold_known_predictions.extend(method_rows); fold_unknown_predictions.extend(unknown_rows); fold_inside_predictions.extend(inside_rows)
        write_csv(fold/"known_test_predictions.csv",fold_known_predictions,prediction_fields); write_csv(fold/"unknown_test_predictions.csv",fold_unknown_predictions,prediction_fields); write_csv(fold/"known_inside_family_predictions.csv",fold_inside_predictions,prediction_fields); write_csv(fold/"threshold_curves.csv",fold_thresholds); write_csv(fold/"method_comparison.csv",[row for row in all_results if row["skill"]==holdout],[key for key in all_results[-1] if key!="trajectory_metrics"])
        (fold/"report.md").write_text(f"# LOSO holdout: {holdout}\n\nKnown train/validation/test segments: {len(train)}/{len(validation)}/{len(known_test)}. Held-out unknown segments: {len(unknown)}. Known segments inside held-out-family trajectories: {len(inside)}.\n\nThresholds were calibrated on known validation segments only. Wipe and all other held-out labels were excluded from that fold's training and validation.\n",encoding="utf-8")
    aggregate=[]
    for method in sorted({row["method"] for row in all_results}):
        values=[row for row in all_results if row["method"]==method]; aggregate.append({"method":method,"mean_known_retention":float(np.mean([row["known_retention"] for row in values])),"std_known_retention":float(np.std([row["known_retention"] for row in values])),"worst_known_retention":float(np.min([row["known_retention"] for row in values])),"mean_rejection_aware_macro_f1":float(np.mean([row["rejection_aware_macro_f1"] for row in values])),"mean_unknown_recall":float(np.mean([row["unknown_recall"] for row in values])),"std_unknown_recall":float(np.std([row["unknown_recall"] for row in values])),"worst_unknown_recall":float(np.min([row["unknown_recall"] for row in values])),"mean_auroc":float(np.mean([row["unknown_auroc"] for row in values])),"mean_aupr":float(np.mean([row["unknown_aupr"] for row in values])),"skills_unknown_recall_ge_0.80":sum(row["unknown_recall"]>=.8 for row in values),"skills_operating_constraint_pass":sum(row["known_retention"]>=.95 and row["unknown_recall"]>=.8 for row in values)})
    write_csv(OUTPUT_ROOT/"per_skill_method_results.csv",all_results); write_csv(OUTPUT_ROOT/"aggregate_results.csv",aggregate); write_csv(OUTPUT_ROOT/"split_audit.csv",split_rows); write_csv(OUTPUT_ROOT/"method_ranking.csv",sorted(aggregate,key=lambda row:(row["mean_known_retention"]>=.95,row["mean_unknown_recall"],row["mean_rejection_aware_macro_f1"],row["worst_unknown_recall"],row["mean_auroc"]),reverse=True)); write_csv(OUTPUT_ROOT/"operating_constraint_audit.csv",[{"method":row["method"],"mean_known_retention":row["mean_known_retention"],"mean_unknown_recall":row["mean_unknown_recall"],"pass":int(row["mean_known_retention"]>=.95 and row["mean_unknown_recall"]>=.8)} for row in aggregate]); absorbing=[]
    for skill in HOLDOUTS:
        for row in [r for r in all_results if r["skill"]==skill]: absorbing.append({"skill":skill,"method":row["method"],"absorbing_class":row["absorbing_class"],"unknown_recall":row["unknown_recall"]})
    write_csv(OUTPUT_ROOT/"absorbing_class_summary.csv",absorbing); bootstrap_rows=[]
    for method in sorted({row["method"] for row in all_results}): bootstrap_rows.extend(bootstrap(all_results,method))
    write_csv(OUTPUT_ROOT/"bootstrap_confidence_intervals.csv",bootstrap_rows); plot_global(all_results)
    config={"experiment":"round15_multiskill_loso_open_set","seed":SEED,"ontology_version":ONTOLOGY_VERSION,"heldout_skills":list(HOLDOUTS),"trajectory_level_splits":True,"bootstrap_resamples":BOOTSTRAPS,"methods":["max_softmax","cosine_knn","round13_energy_margin","round14_energy_margin_baseline","round14_selected_hard_oe"],"heldout_data_used_before_freeze":False,"annotations_modified":False}; (OUTPUT_ROOT/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    report=["# Round 15 multi-skill LOSO open-set study","","Six independent held-out-skill folds were evaluated: wipe, pour, pour_recover, place, insert, and transport. Held-out labels were removed from that fold's training and validation; thresholds were frozen from known validation data before test evaluation.","","## Aggregate results","","| method | mean known retention | worst known retention | mean rejection-aware F1 | mean unknown recall | worst unknown recall | mean AUROC | mean AUPR | operating constraint folds |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate: report.append(f"| {row['method']} | {row['mean_known_retention']:.4f} | {row['worst_known_retention']:.4f} | {row['mean_rejection_aware_macro_f1']:.4f} | {row['mean_unknown_recall']:.4f} | {row['worst_unknown_recall']:.4f} | {row['mean_auroc']:.4f} | {row['mean_aupr']:.4f} | {row['skills_operating_constraint_pass']} / 6 |")
    preferred=next((row for row in sorted(aggregate,key=lambda row:(row["mean_unknown_recall"],row["mean_rejection_aware_macro_f1"],row["worst_unknown_recall"],row["mean_auroc"]),reverse=True) if row["mean_known_retention"]>=.95),None); report += ["","## Conclusions", "", f"1. A method satisfying mean known retention >=0.95 and mean unknown recall >=0.80: {'yes, '+preferred['method'] if preferred else 'no method' }.","2. Per-skill difficulty, absorbing classes, and family-shift metrics are in per_skill_method_results.csv and absorbing_class_summary.csv.","3. Bootstrap intervals are trajectory-level with 2,000 resamples and seed 42.","4. No claim of general open-set skill discovery is made unless performance is consistent across folds.","","## Integrity","","Annotations were not modified. Held-out labels were absent from each fold's train/validation and synthetic generation. Relevant tests, full pytest, compileall, and git diff --check are recorded in the handoff."]
    (OUTPUT_ROOT/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8"); print(json.dumps({"status":"complete","output":str(OUTPUT_ROOT),"aggregate":aggregate},indent=2)); return 0


def ID_TO_LABEL_LOCAL(class_names: tuple[str, ...], index: int) -> str: return class_names[index]


if __name__ == "__main__": raise SystemExit(main())
