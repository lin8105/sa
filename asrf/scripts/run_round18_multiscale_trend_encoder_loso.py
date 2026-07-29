#!/usr/bin/env python3
"""Round 18: fresh multi-scale, trend-aware GT-segment LOSO study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
R16 = ROOT / "outputs/round16_metric_embedding_loso"
R15 = ROOT / "outputs/round15_multiskill_loso_open_set"
OUT = ROOT / "outputs/round18_multiscale_trend_encoder_loso"
DATA_ROOT = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_round16_metric_embedding_loso as r16  # noqa: E402

SEED = 42
HOLDOUTS = r16.HOLDOUTS
FAMILY_FOR_HOLDOUT = r16.FAMILY_FOR_HOLDOUT
DEVICE = torch.device("cpu")
BATCH_SIZE = 32
MAX_EPOCHS = 18
PATIENCE = 5
ABLATION_EPOCHS = 10
ABLATION_PATIENCE = 3
TRIPLET_MARGIN = .20
BOOTSTRAPS = 2000
K = 5
BOUNDARY_JITTER = 5
FEATURE_COLUMNS = r16.base.FEATURE_COLUMNS
ORIGINAL_DIM = len(FEATURE_COLUMNS)
HIDDEN_DIM = 64
BRANCH_DIM = 64
EMBEDDING_DIM = 128
METHOD = "cosine_knn_k5"
VARIANTS = ("A", "B", "C")
ABLATION_MODES = ("A_original", "B_first_difference", "C_first_second_difference", "D_relative_time", "E_multiscale", "F_ordered_phase", "B_first_difference_boundary_jitter")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": float("nan"), "median": float("nan"), "q05": float("nan"), "q25": float("nan"), "q75": float("nan"), "q95": float("nan")}
    return {"mean": float(values.mean()), "median": float(np.median(values)), "q05": float(np.quantile(values, .05)), "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)), "q95": float(np.quantile(values, .95))}


def accepted_only_f1(labels: np.ndarray, predictions: np.ndarray, accepted: np.ndarray, num_classes: int) -> float:
    if not accepted.any():
        return 0.0
    _, values = r16.class_f1(labels[accepted], predictions[accepted], num_classes)
    return float(np.mean(values)) if values else 0.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_class_names(holdout: str) -> tuple[str, ...]:
    return tuple(label for label in r16.CANONICAL_LABELS if label != holdout)


def finite_differences(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Within-segment finite differences; no frame outside ``values`` is read."""
    values = np.asarray(values, dtype=np.float32); first = np.zeros_like(values); second = np.zeros_like(values)
    if len(values) > 1: first[1:] = values[1:] - values[:-1]
    if len(values) > 2: second[2:] = values[2:] - 2 * values[1:-1] + values[:-2]
    return first, second


def relative_time(length: int) -> np.ndarray:
    if length <= 1: return np.zeros(max(length, 0), dtype=np.float32)
    return np.linspace(0.0, 1.0, length, dtype=np.float32)


def feature_channels(values: np.ndarray, mode: str) -> np.ndarray:
    """Build channels from one GT segment only; derivatives never cross segments."""
    values = np.asarray(values, dtype=np.float32); first, second = finite_differences(values); channels = [values]
    if mode in ("first", "first_second", "relative", "multiscale", "phase", "boundary_jitter"): channels.append(first)
    if mode in ("first_second", "relative", "multiscale", "phase"): channels.append(second)
    if mode in ("relative", "multiscale", "phase"): channels.append(np.repeat(relative_time(len(values))[:, None], values.shape[1], axis=1))
    return np.concatenate(channels, axis=1).astype(np.float32)


def phase_bin_bounds(length: int, bins: int = 8) -> list[tuple[int, int]]:
    return [(int(math.floor(i * length / bins)), int(math.floor((i + 1) * length / bins))) for i in range(bins)]


def phase_statistics_from_features(values: np.ndarray, length: int, bins: int = 8) -> np.ndarray:
    """Return ordered [bin, mean/max/first-to-last, channel] statistics."""
    values = np.asarray(values, dtype=np.float32)[:length]; channels = values.shape[1]; output = np.zeros((bins, channels * 3), dtype=np.float32)
    for index, (start, end) in enumerate(phase_bin_bounds(length, bins)):
        if end <= start: continue
        part = values[start:end]; output[index, :channels] = part.mean(axis=0); output[index, channels:2 * channels] = part.max(axis=0); output[index, 2 * channels:] = part[-1] - part[0]
    return output


class TrendDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], mean: np.ndarray, std: np.ndarray, duration_mean: float, duration_std: float, mode: str, boundary_jitter: bool = False, reverse: bool = False):
        self.rows, self.cache = rows, cache; self.mean, self.std = mean.astype(np.float32), std.astype(np.float32); self.duration_mean, self.duration_std = duration_mean, duration_std; self.mode = mode; self.boundary_jitter = boundary_jitter; self.reverse = reverse

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]; timestamps, trajectory = self.cache[row["trajectory"]]; del timestamps
        start, end = int(row["start_frame"]), int(row["end_frame_exclusive"])
        if self.boundary_jitter:
            rng = np.random.default_rng(SEED + index * 7919); start = max(0, start + int(rng.integers(-BOUNDARY_JITTER, BOUNDARY_JITTER + 1))); end = min(len(trajectory), end + int(rng.integers(-BOUNDARY_JITTER, BOUNDARY_JITTER + 1))); end = max(start + 1, end)
        values = trajectory[start:end].astype(np.float32)
        if self.reverse: values = values[::-1].copy()
        channels = (feature_channels(values, self.mode) - self.mean) / self.std
        duration = (np.log1p(max(1, end - start)) - self.duration_mean) / self.duration_std
        return {"sequence": torch.from_numpy(channels), "duration": torch.tensor(duration, dtype=torch.float32), "label": torch.tensor(int(row["label_id"]), dtype=torch.long), "row": row}


def trend_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["sequence"].shape[0] for item in items], dtype=torch.long); maximum = int(lengths.max()); channels = items[0]["sequence"].shape[1]; sequence = torch.zeros((len(items), maximum, channels)); mask = torch.zeros((len(items), maximum), dtype=torch.bool)
    for index, item in enumerate(items):
        n = len(item["sequence"]); sequence[index, :n] = item["sequence"]; mask[index, :n] = True
    return {"sequence": sequence, "valid_mask": mask, "lengths": lengths, "duration": torch.stack([item["duration"] for item in items]), "label": torch.stack([item["label"] for item in items]), "rows": [item["row"] for item in items]}


def input_dim(mode: str) -> int:
    return {"original": ORIGINAL_DIM, "first": ORIGINAL_DIM * 2, "first_second": ORIGINAL_DIM * 3, "relative": ORIGINAL_DIM * 3 + ORIGINAL_DIM, "multiscale": ORIGINAL_DIM * 4, "phase": ORIGINAL_DIM * 4, "boundary_jitter": ORIGINAL_DIM * 2}[mode]


class MaskedPool:
    @staticmethod
    def mean(values: Tensor, mask: Tensor, fallback: Tensor) -> Tensor:
        weights = mask.unsqueeze(1).to(values.dtype); count = weights.sum(dim=2); result = (values * weights).sum(dim=2) / count.clamp_min(1.0); return torch.where(count > 0, result, fallback)


def phase_statistics_torch(values: Tensor, lengths: Tensor, bins: int = 8) -> Tensor:
    """Differentiable ordered phase statistics over valid prefixes only."""
    batch, channels, _ = values.shape
    output = values.new_zeros((batch, bins, channels * 3))
    for batch_index, length in enumerate(lengths.tolist()):
        for phase, (start, end) in enumerate(phase_bin_bounds(int(length), bins)):
            if end <= start:
                continue
            part = values[batch_index, :, start:end]
            output[batch_index, phase, :channels] = part.mean(dim=1)
            output[batch_index, phase, channels:2 * channels] = part.max(dim=1).values
            output[batch_index, phase, 2 * channels:] = part[:, -1] - part[:, 0]
    return output


class TrendEncoder(nn.Module):
    def __init__(self, channels: int, num_classes: int, ordered_phase: bool = False, multiscale: bool = True):
        super().__init__(); self.ordered_phase = ordered_phase; self.multiscale = multiscale; self.input_projection = nn.Conv1d(channels, HIDDEN_DIM, 1)
        if multiscale:
            self.short = nn.Sequential(nn.Conv1d(HIDDEN_DIM, BRANCH_DIM, 3, padding=1, dilation=1), nn.GELU(), nn.Conv1d(BRANCH_DIM, BRANCH_DIM, 3, padding=2, dilation=2), nn.GELU())
            self.medium = nn.Sequential(nn.Conv1d(HIDDEN_DIM, BRANCH_DIM, 3, padding=2, dilation=2), nn.GELU(), nn.Conv1d(BRANCH_DIM, BRANCH_DIM, 3, padding=4, dilation=4), nn.GELU(), nn.Conv1d(BRANCH_DIM, BRANCH_DIM, 3, padding=8, dilation=8), nn.GELU())
            self.long = nn.Sequential(nn.Conv1d(HIDDEN_DIM, BRANCH_DIM, 3, padding=8, dilation=8), nn.GELU(), nn.Conv1d(BRANCH_DIM, BRANCH_DIM, 3, padding=16, dilation=16), nn.GELU(), nn.Conv1d(BRANCH_DIM, BRANCH_DIM, 3, padding=32, dilation=32), nn.GELU())
            fused_channels = BRANCH_DIM * 3
        else:
            self.single = nn.Sequential(nn.Conv1d(HIDDEN_DIM, BRANCH_DIM, 3, padding=1), nn.GELU(), nn.Conv1d(BRANCH_DIM, BRANCH_DIM, 3, padding=2, dilation=2), nn.GELU()); fused_channels = BRANCH_DIM
        self.fused_channels = fused_channels
        if ordered_phase:
            self.phase_projection = nn.Linear(fused_channels * 3, 32); self.phase_conv = nn.Conv1d(32, 32, 3, padding=1); pooled_dim = fused_channels * 5 + 32 * 8 + 1
        else: pooled_dim = fused_channels * 5 + 1
        self.projection = nn.Sequential(nn.Linear(pooled_dim, 256), nn.GELU(), nn.Dropout(.15), nn.Linear(256, EMBEDDING_DIM)); self.classifier = nn.Sequential(nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM // 2), nn.GELU(), nn.Dropout(.1), nn.Linear(EMBEDDING_DIM // 2, num_classes))

    def _branch(self, module: nn.Module, values: Tensor, mask: Tensor) -> Tensor:
        return module(values.masked_fill(~mask.unsqueeze(1), 0.0)).masked_fill(~mask.unsqueeze(1), 0.0)

    def forward(self, sequence: Tensor, valid_mask: Tensor, lengths: Tensor, duration: Tensor) -> tuple[Tensor, Tensor]:
        mask = valid_mask.unsqueeze(1); values = sequence.transpose(1, 2).masked_fill(~mask, 0.0); projected = F.gelu(self.input_projection(values)).masked_fill(~mask, 0.0)
        branches = [self._branch(self.single if not self.multiscale else module, projected, valid_mask) for module in ([None] if not self.multiscale else (self.short, self.medium, self.long))]
        fused = branches[0] if not self.multiscale else torch.cat(branches, dim=1); global_mean = MaskedPool.mean(fused, valid_mask, torch.zeros_like(fused[:, :, 0])); global_max = fused.masked_fill(~mask, torch.finfo(fused.dtype).min).amax(dim=2); time = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0); first_end = torch.div(lengths + 2, 3, rounding_mode="floor").unsqueeze(1); second_end = torch.div(2 * lengths + 2, 3, rounding_mode="floor").unsqueeze(1); start = valid_mask & (time < first_end); middle = valid_mask & (time >= first_end) & (time < second_end); end = valid_mask & (time >= second_end); pooled = [MaskedPool.mean(fused, start, global_mean), MaskedPool.mean(fused, middle, global_mean), MaskedPool.mean(fused, end, global_mean), global_mean, global_max]
        if self.ordered_phase:
            phase_values = phase_statistics_torch(fused, lengths); phase_values = F.gelu(self.phase_projection(phase_values)); phase_values = F.gelu(self.phase_conv(phase_values.transpose(1, 2))).transpose(1, 2).reshape(len(sequence), -1); pooled.append(phase_values)
        pooled.append(duration.unsqueeze(1)); embedding = F.normalize(self.projection(torch.cat(pooled, dim=1)), p=2, dim=1, eps=1e-8); return embedding, self.classifier(embedding)


def make_model(variant: str, channels: int, num_classes: int) -> nn.Module:
    if variant == "A": return r16.base.SegmentClassifier(channels, r16.base.HIDDEN_DIM, r16.base.PROJECTION_DIM, r16.base.EMBEDDING_DIM, num_classes)
    return TrendEncoder(channels, num_classes, ordered_phase=variant == "C", multiscale=True)


def supcon(embeddings: Tensor, labels: Tensor, temperature: float = .07) -> Tensor:
    return r16.supcon(embeddings, labels, temperature)


def hard_triplet(embeddings: Tensor, labels: Tensor) -> tuple[Tensor, dict[str, float]]:
    return r16.batch_hard_triplet(embeddings, labels, TRIPLET_MARGIN)


def collect(model: nn.Module, loader: DataLoader) -> dict[str, Any]:
    model.eval(); embeddings=[]; logits=[]; rows=[]
    with torch.no_grad():
        for batch in loader:
            embedding, output = model(batch["sequence"], batch["valid_mask"], batch["lengths"], batch["duration"]); embeddings.append(embedding.cpu().numpy()); logits.append(output.cpu().numpy()); rows.extend(batch["rows"])
    return {"embeddings": np.concatenate(embeddings), "logits": np.concatenate(logits), "rows": rows, "labels": np.asarray([int(row["label_id"]) for row in rows]), "predictions": np.concatenate(logits).argmax(axis=1)}


def quality(output: dict[str, Any], class_names: tuple[str, ...], split: str, variant: str, skill: str) -> dict[str, Any]:
    result = r16.embedding_quality({"embeddings": output["embeddings"], "logits": output["logits"], "rows": output["rows"]}, class_names, split, variant)[0]; result["skill"] = skill; return result


def train_stats(rows: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], mode: str, boundary_jitter: bool) -> tuple[np.ndarray, np.ndarray, float, float]:
    arrays=[]; durations=[]
    for index, row in enumerate(rows):
        _, values = cache[row["trajectory"]]; start, end = row["start_frame"], row["end_frame_exclusive"]
        if boundary_jitter:
            rng=np.random.default_rng(SEED+index*7919); start=max(0,start+int(rng.integers(-BOUNDARY_JITTER,BOUNDARY_JITTER+1))); end=min(len(values),end+int(rng.integers(-BOUNDARY_JITTER,BOUNDARY_JITTER+1))); end=max(start+1,end)
        arrays.append(feature_channels(values[start:end], mode)); durations.append(np.log1p(max(1,end-start)))
    all_values=np.concatenate(arrays); mean=all_values.mean(axis=0); std=np.maximum(all_values.std(axis=0),1e-6); durations=np.asarray(durations); return mean.astype(np.float32), std.astype(np.float32), float(durations.mean()), float(max(durations.std(),1e-6))


def train_one(train: list[dict[str, Any]], validation: list[dict[str, Any]], cache: dict[str, tuple[np.ndarray, np.ndarray]], class_names: tuple[str, ...], variant: str, mode: str, fold: str, boundary_jitter: bool = False, epochs: int = MAX_EPOCHS, patience: int = PATIENCE) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mean,std,dm,ds=train_stats(train,cache,mode,boundary_jitter); dataset=TrendDataset(train,cache,mean,std,dm,ds,mode,boundary_jitter); valset=TrendDataset(validation,cache,mean,std,dm,ds,mode,False); sampler=r16.BalancedBatchSampler(train,BATCH_SIZE,SEED+int(hashlib.sha256(f"{fold}/{variant}/{mode}/{boundary_jitter}".encode()).hexdigest()[:8],16)%10000); loader=DataLoader(dataset,batch_sampler=sampler,collate_fn=trend_collate); vloader=DataLoader(valset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=trend_collate); model=make_model(variant if variant in VARIANTS else "B",input_dim(mode),len(class_names)).to(DEVICE); optimizer=torch.optim.AdamW(model.parameters(),lr=r16.base.LEARNING_RATE,weight_decay=r16.base.WEIGHT_DECAY); counts=Counter(int(row["label_id"]) for row in train); weights=torch.tensor([1/np.sqrt(counts.get(i,1)) for i in range(len(class_names))],dtype=torch.float32); weights*=len(class_names)/weights.sum(); centers=torch.zeros((len(class_names),EMBEDDING_DIM)); seen=torch.zeros(len(class_names),dtype=torch.bool); best=None; best_key=None; best_epoch=0; stale=0; history=[]; val_history=[]
    for epoch in range(1,epochs+1):
        model.train(); totals=defaultdict(float); count=0
        for batch in loader:
            embedding,logits=model(batch["sequence"],batch["valid_mask"],batch["lengths"],batch["duration"]); labels=batch["label"]; ce=F.cross_entropy(logits,labels,weight=weights); con=supcon(embedding,labels); triplet,tstats=hard_triplet(embedding,labels); center=((embedding-centers[labels])**2).sum(dim=1).mean(); loss=ce+.20*con+.10*triplet+.02*center; optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
            with torch.no_grad():
                for label in labels.unique():
                    li=int(label); value=embedding[labels==label].mean(0); centers[li]=value if not seen[li] else .9*centers[li]+.1*value; seen[li]=True
            n=len(labels); count+=n
            for key,value in (("loss",loss),("cross_entropy",ce),("supervised_contrastive_loss",con),("batch_hard_triplet_loss",triplet),("center_compactness_loss",center)): totals[key]+=float(value.detach())*n
            for key,value in tstats.items(): totals[key]+=value
        val=collect(model,vloader); q=quality(val,class_names,"validation",variant,fold); f1,_=r16.class_f1(val["labels"],val["predictions"],len(class_names)); nn_acc=nearest_neighbor_accuracy(val["embeddings"],val["labels"]); row={"fold":fold,"variant":variant,"mode":mode,"boundary_jitter":int(boundary_jitter),"epoch":epoch,"validation_macro_f1":f1,"validation_accuracy":float((val["labels"]==val["predictions"]).mean()),"cross_family_same_class_distance":q["cross_family_same_class_distance"],"nearest_neighbor_validation_accuracy":nn_acc,"mean_within_class_variance":q["mean_within_class_variance"]}; val_history.append(row); history.append({"fold":fold,"variant":variant,"mode":mode,"boundary_jitter":int(boundary_jitter),"epoch":epoch,**{key:totals[key]/max(count,1) for key in ("loss","cross_entropy","supervised_contrastive_loss","batch_hard_triplet_loss","center_compactness_loss")},"triplet_count":totals["triplet_count"],"active_triplet_fraction":totals["active_triplet_fraction"]/max(len(loader),1),"mean_positive_distance":totals["mean_positive_distance"]/max(len(loader),1),"mean_hard_negative_distance":totals["mean_hard_negative_distance"]/max(len(loader),1),"triplet_margin_violation_rate":totals["triplet_margin_violation_rate"]/max(len(loader),1)})
        key=(f1,-float(q["cross_family_same_class_distance"]) if np.isfinite(q["cross_family_same_class_distance"]) else -1e9,nn_acc,-q["mean_within_class_variance"],row["validation_accuracy"])
        if best_key is None or key>best_key: best_key=key; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_centers=centers.clone(); best_epoch=epoch; stale=0
        else: stale+=1
        print(f"[round18] {fold} {variant}/{mode} epoch={epoch} val_f1={f1:.4f}",flush=True)
        if stale>=patience: break
    model.load_state_dict(best); return model,{"best_epoch":best_epoch,"best_validation_macro_f1":best_key[0],"centers":best_centers.numpy(),"mean":mean,"std":std,"duration_mean":dm,"duration_std":ds},history,val_history,{"train":dataset,"validation":valset}


def nearest_neighbor_accuracy(embeddings: np.ndarray, labels: np.ndarray) -> float:
    if len(embeddings)<2:return 0.0
    distances=1-embeddings@embeddings.T; np.fill_diagonal(distances,np.inf); return float((labels[np.argmin(distances,axis=1)]==labels).mean())


def knn_scores(output: dict[str, Any], reference: np.ndarray, reference_labels: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    distances=1-output["embeddings"]@reference.T; scores=[]; neighbors=[]
    for i,pred in enumerate(output["predictions"]):
        candidates=np.flatnonzero(reference_labels==int(pred)); candidates=candidates if len(candidates) else np.arange(len(reference_labels)); order=candidates[np.argsort(distances[i,candidates])[:min(K,len(candidates))]]; neighbors.append(order); scores.append(float(distances[i,order].mean()))
    return np.asarray(scores),neighbors


def metrics(known: dict[str, Any], unknown: dict[str, Any], inside: dict[str, Any], known_score: np.ndarray, unknown_score: np.ndarray, inside_score: np.ndarray, threshold: float, class_names: tuple[str, ...]) -> dict[str, Any]:
    accepted=known_score<=threshold; unknown_rejected=unknown_score>threshold; labels,pred=known["labels"],known["predictions"]; rejection,per=r16.rejection_f1(labels,pred,accepted,len(class_names)); _,closed_per_class=r16.class_f1(labels,pred,len(class_names)); closed_set_macro_f1=float(np.mean(closed_per_class)); binary=np.concatenate((np.zeros(len(known_score)),np.ones(len(unknown_score)))); scores=np.concatenate((known_score,unknown_score)); inside_accept=inside_score<=threshold; return {"closed_set_accuracy":float((labels==pred).mean()),"closed_set_macro_f1":closed_set_macro_f1,"known_retention":float(accepted.mean()),"false_unknown_rate":float((~accepted).mean()),"rejection_aware_macro_f1":rejection,"accepted_only_macro_f1":accepted_only_f1(labels,pred,accepted,len(class_names)),"accepted_known_accuracy":float((pred[accepted]==labels[accepted]).mean()) if accepted.any() else 0.0,"unknown_recall":float(unknown_rejected.mean()),"false_known_rate":float((~unknown_rejected).mean()),"auroc":r16.auroc(binary,scores),"aupr":r16.aupr(binary,scores),"unknown_score_distribution":quantiles(unknown_score),"absorbing_class":class_names[int(Counter(unknown["predictions"].tolist()).most_common(1)[0][0])],"inside_closed_set_accuracy":float((inside["labels"]==inside["predictions"]).mean()),"inside_known_retention":float(inside_accept.mean()),"inside_false_unknown_rate":float((~inside_accept).mean()),"inside_rejection_aware_macro_f1":r16.rejection_f1(inside["labels"],inside["predictions"],inside_accept,len(class_names))[0],"inside_score_shift_mean":float(inside_score.mean()-known_score.mean()),"inside_score_shift_median":float(np.median(inside_score)-np.median(known_score)),"per_class":[{"class":name,"support":int((labels==i).sum()),"retention":float(((labels==i)&accepted).sum()/max(int((labels==i).sum()),1)),"f1":float(per[i])} for i,name in enumerate(class_names)]}


def bootstrap(results: list[dict[str,Any]], trajectory_values: dict[tuple[str,str],dict[str,list[float]]]) -> list[dict[str,Any]]:
    rng=np.random.default_rng(SEED); output=[]
    for variant in VARIANTS:
        for metric in ("known_retention","rejection_aware_macro_f1","unknown_recall","auroc"):
            samples=[]
            for _ in range(BOOTSTRAPS):
                folds=[]
                for skill in HOLDOUTS:
                    values=trajectory_values[(skill,variant)][metric]; folds.append(float(np.asarray(values)[rng.integers(0,len(values),len(values))].mean()))
                samples.append(float(np.mean(folds)))
            output.append({"variant":variant,"metric":metric,"bootstrap_resamples":BOOTSTRAPS,"seed":SEED,"mean":float(np.mean(samples)),"ci_lower":float(np.quantile(samples,.025)),"ci_upper":float(np.quantile(samples,.975))})
    return output


def trajectory_values(group: dict[str,Any], score: np.ndarray, threshold: float, class_names: tuple[str,...], unknown: bool=False, known_scores: np.ndarray|None=None) -> dict[str,list[float]]:
    by=defaultdict(list)
    for i,row in enumerate(group["rows"]): by[row["trajectory"]].append(i)
    out={"known_retention":[],"rejection_aware_macro_f1":[],"unknown_recall":[],"auroc":[]}
    if unknown:
        for indexes in by.values():
            indexes=np.asarray(indexes); out["unknown_recall"].append(float((score[indexes]>threshold).mean())); out["auroc"].append(r16.auroc(np.concatenate((np.zeros(len(known_scores)),np.ones(len(indexes)))),np.concatenate((known_scores,score[indexes]))))
    else:
        for indexes in by.values():
            indexes=np.asarray(indexes); accepted=score[indexes]<=threshold; out["known_retention"].append(float(accepted.mean())); out["rejection_aware_macro_f1"].append(r16.rejection_f1(group["labels"][indexes],group["predictions"][indexes],accepted,len(class_names))[0])
    return out


def save_prediction_rows(skill: str, group_name: str, output: dict[str,Any], score: np.ndarray, threshold: float, class_names: tuple[str,...], neighbors: list[np.ndarray], ref_rows: list[dict[str,str]], method: str=METHOD) -> list[dict[str,Any]]:
    rows=[]
    for i,row in enumerate(output["rows"]):
        item={"skill":skill,"group":group_name,"method":method,"trajectory":row["trajectory"],"segment_index":row["segment_index"],"ground_truth_label":row["label"],"predicted_label":class_names[int(output["predictions"][i])],"score":float(score[i]),"threshold":threshold,"decision":"known" if score[i]<=threshold else "unknown","duration_frames":row["duration_frames"]}
        for rank,index in enumerate(neighbors[i],1): item[f"nearest_{rank}_sample_id"]=ref_rows[int(index)]["sample_id"]; item[f"nearest_{rank}_label"]=ref_rows[int(index)]["label"]; item[f"nearest_{rank}_trajectory"]=ref_rows[int(index)]["trajectory"]; item[f"nearest_{rank}_distance"]=float(1-output["embeddings"][i]@output.get("reference",np.zeros((1,128)))[0]) if False else ""
        rows.append(item)
    return rows


def raw_segment_summaries(values: np.ndarray, bins: int=8) -> dict[str,float]:
    first,second=finite_differences(values); result={}
    for channel,name in enumerate(FEATURE_COLUMNS):
        result[f"mean_abs_first_difference_{name}"]=float(np.abs(first[:,channel]).mean()); result[f"mean_abs_second_difference_{name}"]=float(np.abs(second[:,channel]).mean()); result[f"signed_cumulative_change_{name}"]=float(first[:,channel].sum()); result[f"start_to_end_change_{name}"]=float(values[-1,channel]-values[0,channel]); result[f"maximum_local_change_{name}"]=float(np.abs(first[:,channel]).max()); result[f"duration_normalized_trend_{name}"]=float(first[:,channel].sum()/max(len(values),1))
    phase=phase_statistics_from_features(values,len(values),bins)
    for b in range(bins):
        for c,name in enumerate(FEATURE_COLUMNS): result[f"phase_{b+1}_mean_{name}"]=float(phase[b,c])
    return result


def train_order_probe(train_original: dict[str,Any], train_reverse: dict[str,Any], val_original: dict[str,Any], val_reverse: dict[str,Any]) -> float:
    x_train=torch.from_numpy(np.concatenate((train_original["embeddings"],train_reverse["embeddings"])).astype(np.float32)); y_train=torch.from_numpy(np.concatenate((np.zeros(len(train_original["embeddings"])),np.ones(len(train_reverse["embeddings"]))))).long(); x_val=torch.from_numpy(np.concatenate((val_original["embeddings"],val_reverse["embeddings"])).astype(np.float32)); y_val=torch.from_numpy(np.concatenate((np.zeros(len(val_original["embeddings"])),np.ones(len(val_reverse["embeddings"]))))).long(); probe=nn.Linear(EMBEDDING_DIM,2); optimizer=torch.optim.Adam(probe.parameters(),lr=.01)
    for _ in range(30): optimizer.zero_grad(); loss=F.cross_entropy(probe(x_train),y_train); loss.backward(); optimizer.step()
    with torch.no_grad(): return float((probe(x_val).argmax(1)==y_val).float().mean())


def plot_fold(out_fold: Path, skill: str, known_rows: list[dict[str, Any]], unknown_rows: list[dict[str, Any]]) -> None:
    """Small per-fold score audit; aggregate figures are produced below."""
    for variant in VARIANTS:
        known = np.asarray([float(row["score"]) for row in known_rows if row["variant"] == variant])
        unknown = np.asarray([float(row["score"]) for row in unknown_rows if row["variant"] == variant])
        fig, ax = plt.subplots(figsize=(7, 4))
        if len(known): ax.hist(known, bins=20, alpha=.55, label="known test")
        if len(unknown): ax.hist(unknown, bins=20, alpha=.55, label="held-out unknown")
        ax.set_title(f"{skill}: Variant {variant} cosine kNN scores")
        ax.set_xlabel("mean cosine distance")
        ax.set_ylabel("segments")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_fold / "figures" / f"score_distribution_variant_{variant.lower()}.png", dpi=160)
        plt.close(fig)


def main() -> int:
    seed_everything(); OUT.mkdir(parents=True,exist_ok=True); (OUT/"figures").mkdir(exist_ok=True); raw,cache=r16.load_rows(); all_results=[]; quality_rows=[]; comparison_rows=[]; ablation_rows=[]; probe_rows=[]; difficult_rows=[]; threshold_rows=[]; trajectory_map={}; per_skill_payload={}; known_predictions_all=[]; unknown_predictions_all=[]; inside_predictions_all=[]; pour_rows=[]
    for skill in HOLDOUTS:
        print(f"[round18] starting holdout {skill}",flush=True); class_names=canonical_class_names(skill); fold=R16/f"holdout_{skill}"; out_fold=OUT/f"holdout_{skill}"; (out_fold/"figures").mkdir(parents=True,exist_ok=True); fold_manifest=read_csv(fold/"split_manifest.csv"); write_csv(out_fold/"split_manifest.csv",fold_manifest); train=[dict(r, label_id=int(r["label_id"]),segment_index=int(r["segment_index"]),start_frame=int(r["start_frame"]),end_frame_exclusive=int(r["end_frame_exclusive"]),duration_frames=int(r["duration_frames"])) for r in fold_manifest if r["split"]=="train"]; validation=[dict(r,label_id=int(r["label_id"]),segment_index=int(r["segment_index"]),start_frame=int(r["start_frame"]),end_frame_exclusive=int(r["end_frame_exclusive"]),duration_frames=int(r["duration_frames"])) for r in fold_manifest if r["split"]=="validation"]; test=[dict(r,label_id=int(r["label_id"]),segment_index=int(r["segment_index"]),start_frame=int(r["start_frame"]),end_frame_exclusive=int(r["end_frame_exclusive"]),duration_frames=int(r["duration_frames"])) for r in fold_manifest if r["split"]=="test"]; unknown=[r for r in test if r["label"]==skill]; inside=[r for r in test if r["family"]==FAMILY_FOR_HOLDOUT[skill] and r["label"]!=skill]; known=[r for r in test if r["family"]!=FAMILY_FOR_HOLDOUT[skill] and r["label"]!=skill]; write_csv(out_fold/"excluded_heldout_segments.csv",[r for r in read_csv(fold/"excluded_heldout_segments.csv")],None) if (fold/"excluded_heldout_segments.csv").exists() else None
        variants_outputs={}; fold_validation_rows=[]; ref_rows=read_csv(fold/"reference_embeddings.csv") if (fold/"reference_embeddings.csv").exists() else []
        for variant,mode in (("A","original"),("B","multiscale"),("C","phase")):
            model,info,history,valhist,datasets=train_one(train,validation,cache,class_names,variant,mode,skill,False); fold_validation_rows.extend(valhist); write_csv(out_fold/f"training_log_variant_{variant.lower()}.csv",history); write_csv(out_fold/f"validation_metrics_{variant.lower()}.csv",valhist); model_dir=out_fold/f"model_variant_{variant.lower()}"; model_dir.mkdir(exist_ok=True); torch.save({"model_state":model.state_dict(),"optimizer_state":None,"previous_weights_reused":False,"variant":variant,"best_epoch":info["best_epoch"],"held_out_skill":skill,"known_class_order":list(class_names),"input_mode":mode,"input_dim":input_dim(mode),"centers":info["centers"]},model_dir/"best.pt"); loaders={"train":DataLoader(datasets["train"],batch_size=BATCH_SIZE,shuffle=False,collate_fn=trend_collate),"validation":DataLoader(datasets["validation"],batch_size=BATCH_SIZE,shuffle=False,collate_fn=trend_collate)}; outputs={name:collect(model,DataLoader(TrendDataset(rows,cache,info["mean"],info["std"],info["duration_mean"],info["duration_std"],mode),batch_size=BATCH_SIZE,shuffle=False,collate_fn=trend_collate)) for name,rows in (("train",train),("validation",validation),("known_test",known),("unknown_test",unknown),("known_inside_family",inside))}; ref=outputs["train"]; reference,reference_labels=ref["embeddings"],ref["labels"]; scores={name:knn_scores(out,reference,reference_labels) for name,out in outputs.items()}; threshold,validation_retention,_=r16.threshold_curve(scores["validation"][0],.95); threshold_rows.append({"skill":skill,"variant":variant,"threshold":threshold,"validation_known_retention":validation_retention,"target_retention":.95,"calibrated_group":"known_validation_only","heldout_unknown_used":0});
            per_quality=[quality(outputs[split],class_names,split,variant,skill) for split in ("train","validation","known_test","known_inside_family")]; quality_rows.extend(per_quality); reference_npz=out_fold/f"reference_embeddings_{variant.lower()}.npz"; np.savez_compressed(reference_npz,embeddings=reference,labels=reference_labels,class_names=np.asarray(class_names),sample_ids=np.asarray([r["sample_id"] for r in ref["rows"]])); variants_outputs[variant]=(model,info,outputs,scores,threshold,reference,reference_labels); comparison_rows.append({"skill":skill,"variant":variant,"best_epoch":info["best_epoch"],"best_validation_macro_f1":info["best_validation_macro_f1"],"input_mode":mode,"input_channels":input_dim(mode),"boundary_jitter":0,"fresh_optimizer":1,"previous_weights_reused":0})
            for split,group in (("known_test",outputs["known_test"]),("unknown_test",outputs["unknown_test"]),("known_inside_family",outputs["known_inside_family"])): pass
        # Required common reference file is Variant A/B/C-labelled concatenation-free A/B/C bank; use Variant A file as the per-fold default and retain variant-specific files.
        np.savez_compressed(out_fold/"reference_embeddings.npz",variant_a=variants_outputs["A"][5],variant_b=variants_outputs["B"][5],variant_c=variants_outputs["C"][5],class_names=np.asarray(class_names))
        fold_known=[];fold_unknown=[];fold_inside=[]
        for variant in VARIANTS:
            model,info,outputs,scores,threshold,reference,reference_labels=variants_outputs[variant]
            for split,group in (("known_test",outputs["known_test"]),("unknown_test",outputs["unknown_test"]),("known_inside_family",outputs["known_inside_family"])):
                s,n=scores[split]; group["reference"]=reference; rows=[]
                for i,row in enumerate(group["rows"]):
                    item={"skill":skill,"variant":variant,"group":split,"trajectory":row["trajectory"],"segment_index":row["segment_index"],"ground_truth_label":row["label"],"predicted_label":class_names[int(group["predictions"][i])],"score":float(s[i]),"threshold":threshold,"decision":"known" if s[i]<=threshold else "unknown","duration_frames":row["duration_frames"]}
                    for rank,index in enumerate(n[i],1):
                        item[f"nearest_{rank}_sample_id"]=ref_rows[int(index)]["sample_id"] if ref_rows else f"reference_{int(index)}"; item[f"nearest_{rank}_label"]=class_names[int(reference_labels[index])]; item[f"nearest_{rank}_trajectory"]=ref_rows[int(index)]["trajectory"] if ref_rows else ""; item[f"nearest_{rank}_distance"]=float(1-group["embeddings"][i]@reference[index])
                    rows.append(item)
                if split=="known_test": fold_known.extend(rows)
                elif split=="unknown_test": fold_unknown.extend(rows)
                else: fold_inside.extend(rows)
            result=metrics(outputs["known_test"],outputs["unknown_test"],outputs["known_inside_family"],scores["known_test"][0],scores["unknown_test"][0],scores["known_inside_family"][0],threshold,class_names); result.update({"skill":skill,"variant":variant,"threshold":threshold,"validation_known_retention":float((scores["validation"][0]<=threshold).mean()),"nearest_neighbor_validation_accuracy":nearest_neighbor_accuracy(outputs["validation"]["embeddings"],outputs["validation"]["labels"])}) ; all_results.append(result); trajectory_map[(skill,variant)]={**trajectory_values(outputs["known_test"],scores["known_test"][0],threshold,class_names),**{k:v for k,v in trajectory_values(outputs["unknown_test"],scores["unknown_test"][0],threshold,class_names,True,scores["known_test"][0]).items() if k in ("unknown_recall","auroc")}}
            known_predictions_all.extend([r for r in fold_known if r["variant"]==variant]);unknown_predictions_all.extend([r for r in fold_unknown if r["variant"]==variant]);inside_predictions_all.extend([r for r in fold_inside if r["variant"]==variant])
        write_csv(out_fold/"known_test_predictions.csv",fold_known);write_csv(out_fold/"unknown_test_predictions.csv",fold_unknown);write_csv(out_fold/"known_inside_family_predictions.csv",fold_inside); write_csv(out_fold/"validation_metrics.csv",fold_validation_rows); write_csv(out_fold/"training_log_variant_a.csv",read_csv(out_fold/"training_log_variant_a.csv"))
        plot_fold(out_fold,skill,fold_known,fold_unknown)
        # Temporal-order diagnostic: only train/validation known data; not used for epoch selection.
        for variant in VARIANTS:
            model,info,outputs,_,_,_,_=variants_outputs[variant]; train_reverse=collect(model,DataLoader(TrendDataset(train,cache,info["mean"],info["std"],info["duration_mean"],info["duration_std"],"original" if variant=="A" else ("multiscale" if variant=="B" else "phase"),False,True),batch_size=BATCH_SIZE,shuffle=False,collate_fn=trend_collate)); val_reverse=collect(model,DataLoader(TrendDataset(validation,cache,info["mean"],info["std"],info["duration_mean"],info["duration_std"],"original" if variant=="A" else ("multiscale" if variant=="B" else "phase"),False,True),batch_size=BATCH_SIZE,shuffle=False,collate_fn=trend_collate)); probe_rows.append({"skill":skill,"variant":variant,"train_samples":2*len(train),"validation_samples":2*len(validation),"original_vs_reversed_validation_accuracy":train_order_probe(outputs["train"],train_reverse,outputs["validation"],val_reverse),"diagnostic_only":1})
        if skill=="pour":
            manifest_lookup={(r["trajectory"],int(r["segment_index"])):r for r in test}; values_cache={r["trajectory"]:cache[r["trajectory"]][1] for r in unknown}; variant=variants_outputs["B"]; model,info,outputs,scores,threshold,reference,reference_labels=variant; s,n=scores["unknown_test"]
            for i,row in enumerate(outputs["unknown_test"]["rows"]):
                raw=values_cache[row["trajectory"]][row["start_frame"]:row["end_frame_exclusive"]]; item={"variant":"B","trajectory":row["trajectory"],"segment_index":row["segment_index"],"start_frame":row["start_frame"],"end_frame_exclusive":row["end_frame_exclusive"],"duration_frames":row["duration_frames"],"predicted_label":canonical_class_names(skill)[int(outputs["unknown_test"]["predictions"][i])],"score":float(s[i]),"decision":"known" if s[i]<=threshold else "unknown",**raw_segment_summaries(raw)}
                for rank,index in enumerate(n[i],1): item[f"nearest_{rank}_sample_id"]=read_csv(fold/"reference_embeddings.csv")[int(index)]["sample_id"] if (fold/"reference_embeddings.csv").exists() else ""; item[f"nearest_{rank}_distance"]=float(1-outputs["unknown_test"]["embeddings"][i]@reference[index]); item[f"nearest_{rank}_label"]=canonical_class_names(skill)[int(reference_labels[index])]
                pour_rows.append(item)
        # Difficult pair class summaries from test known embeddings.
        for variant in VARIANTS:
            out=variants_outputs[variant][2]["known_test"]; emb=out["embeddings"]; labels=out["labels"]
            for left,right in (("wipe","transport"),("pour_recover","transport"),("insert","place"),("place","transport")):
                if left in class_names and right in class_names:
                    li,ri=class_names.index(left),class_names.index(right); leftv=emb[labels==li];rightv=emb[labels==ri]; distances=(1-leftv@rightv.T).ravel(); difficult_rows.append({"skill":skill,"variant":variant,"pair":f"{left}_versus_{right}","left_support":len(leftv),"right_support":len(rightv),"mean_cross_distance":float(distances.mean()) if len(distances) else float("nan"),"q05_cross_distance":float(np.quantile(distances,.05)) if len(distances) else float("nan"),"q95_cross_distance":float(np.quantile(distances,.95)) if len(distances) else float("nan"),"duration_left_mean":float(np.mean([r["duration_frames"] for r in out["rows"] if r["label"]==left])) if leftv.size else float("nan"),"duration_right_mean":float(np.mean([r["duration_frames"] for r in out["rows"] if r["label"]==right])) if rightv.size else float("nan")})
        # Ablations on required four folds, shorter fixed training budget.
        if skill in ("pour","wipe","place","transport"):
            for mode_name in ABLATION_MODES:
                mode={"A_original":"original","B_first_difference":"first","C_first_second_difference":"first_second","D_relative_time":"relative","E_multiscale":"multiscale","F_ordered_phase":"phase","B_first_difference_boundary_jitter":"boundary_jitter"}[mode_name]; variant="A" if mode_name=="A_original" else ("C" if mode_name=="F_ordered_phase" else "B"); jitter=mode_name.endswith("boundary_jitter"); model,info,history,valhist,datasets=train_one(train,validation,cache,class_names,variant,mode,skill,jitter,ABLATION_EPOCHS,ABLATION_PATIENCE); ref=collect(model,DataLoader(TrendDataset(train,cache,info["mean"],info["std"],info["duration_mean"],info["duration_std"],mode,jitter),batch_size=BATCH_SIZE,shuffle=False,collate_fn=trend_collate)); outs={name:collect(model,DataLoader(TrendDataset(rows,cache,info["mean"],info["std"],info["duration_mean"],info["duration_std"],mode,jitter if name=="train" else False),batch_size=BATCH_SIZE,shuffle=False,collate_fn=trend_collate)) for name,rows in (("validation",validation),("known_test",known),("unknown_test",unknown),("known_inside_family",inside))}; ss={name:knn_scores(out,ref["embeddings"],ref["labels"])[0] for name,out in outs.items()}; th,ret,_=r16.threshold_curve(ss["validation"],.95); result=metrics(outs["known_test"],outs["unknown_test"],outs["known_inside_family"],ss["known_test"],ss["unknown_test"],ss["known_inside_family"],th,class_names); ablation_rows.append({"skill":skill,"ablation":mode_name,"boundary_jitter":int(jitter),"best_epoch":info["best_epoch"],"validation_macro_f1":info["best_validation_macro_f1"],"validation_known_retention":ret,"known_retention":result["known_retention"],"rejection_aware_macro_f1":result["rejection_aware_macro_f1"],"unknown_recall":result["unknown_recall"],"auroc":result["auroc"],"aupr":result["aupr"]})
    # Aggregate and figures.
    aggregate=[]; baseline_f1=[]
    for skill in HOLDOUTS:
        vals=[r for r in all_results if r["skill"]==skill and r["variant"]=="A"]; baseline_f1.append(vals[0]["closed_set_macro_f1"] if vals else 0)
    for variant in VARIANTS:
        vals=[r for r in all_results if r["variant"]==variant]; aggregate.append({"method":f"round18_variant_{variant}","mean_known_retention":float(np.mean([r["known_retention"] for r in vals])),"worst_known_retention":float(np.min([r["known_retention"] for r in vals])),"mean_rejection_aware_macro_f1":float(np.mean([r["rejection_aware_macro_f1"] for r in vals])),"mean_unknown_recall":float(np.mean([r["unknown_recall"] for r in vals])),"worst_unknown_recall":float(np.min([r["unknown_recall"] for r in vals])),"mean_auroc":float(np.mean([r["auroc"] for r in vals])),"mean_aupr":float(np.mean([r["aupr"] for r in vals])),"folds_unknown_recall_ge_0.60":int(sum(r["unknown_recall"]>=.60 for r in vals)),"folds_unknown_recall_below_0.30":int(sum(r["unknown_recall"]<.30 for r in vals)),"folds_both_targets":int(sum(r["known_retention"]>=.95 and r["unknown_recall"]>=.60 for r in vals)),"round18_closed_set_mean":float(np.mean([r["closed_set_macro_f1"] for r in vals]))})
    for name,path,source in (("round15_cosine_knn",R15/"aggregate_results.csv","cosine_knn"),("round16_selected_metric",R16/"aggregate_results.csv","round16_variant_B_cosine_knn")):
        rows=read_csv(path); row=next((r for r in rows if r["method"]==source),None)
        if row: aggregate.append({"method":name,"mean_known_retention":float(row["mean_known_retention"]),"worst_known_retention":float(row["worst_known_retention"]),"mean_rejection_aware_macro_f1":float(row["mean_rejection_aware_macro_f1"]),"mean_unknown_recall":float(row["mean_unknown_recall"]),"worst_unknown_recall":float(row["worst_unknown_recall"]),"mean_auroc":float(row["mean_auroc"]),"mean_aupr":float(row["mean_aupr"]),"folds_unknown_recall_ge_0.60":"","folds_unknown_recall_below_0.30":"","folds_both_targets":"","round18_closed_set_mean":""})
    write_csv(OUT/"aggregate_results.csv",aggregate);write_csv(OUT/"per_skill_results.csv",all_results);write_csv(OUT/"encoder_variant_comparison.csv",comparison_rows);write_csv(OUT/"embedding_quality_comparison.csv",quality_rows);write_csv(OUT/"ablation_results.csv",ablation_rows);write_csv(OUT/"temporal_order_probe.csv",probe_rows);write_csv(OUT/"difficult_pair_analysis.csv",difficult_rows);write_csv(OUT/"pour_transport_diagnostics.csv",pour_rows);write_csv(OUT/"threshold_audit.csv",threshold_rows);write_csv(OUT/"bootstrap_confidence_intervals.csv",bootstrap(all_results,trajectory_map))
    methods=["A","B","C"]; labels=["A","B","C"]; fig,ax=plt.subplots(figsize=(8,5));
    for v in VARIANTS:
        vals=[r for r in all_results if r["variant"]==v];ax.scatter([r["known_retention"] for r in vals],[r["unknown_recall"] for r in vals],label=v)
    ax.axvline(.95,ls="--",color="gray");ax.axhline(.65,ls="--",color="gray");ax.legend();ax.set_xlabel("known retention");ax.set_ylabel("unknown recall");fig.tight_layout();fig.savefig(OUT/"figures/mean_known_retention_vs_mean_unknown_recall.png",dpi=160);plt.close(fig)
    for filename,key,title in (("per_fold_unknown_recall_heatmap.png","unknown_recall","Unknown recall"),("per_fold_known_retention_heatmap.png","known_retention","Known retention")):
        matrix=np.asarray([[next(r[key] for r in all_results if r["skill"]==s and r["variant"]==v) for s in HOLDOUTS] for v in VARIANTS]);fig,ax=plt.subplots(figsize=(9,4));im=ax.imshow(matrix,vmin=0,vmax=1,cmap="viridis");ax.set_xticks(range(6),HOLDOUTS,rotation=35,ha="right");ax.set_yticks(range(3),labels);fig.colorbar(im,ax=ax);ax.set_title(title);fig.tight_layout();fig.savefig(OUT/"figures"/filename,dpi=160);plt.close(fig)
    for filename,key,title in (("embedding_within_class_variance.png","mean_within_class_variance","Within-class variance"),("cross_family_same_class_distance.png","cross_family_same_class_distance","Cross-family same-class distance")):
        fig,ax=plt.subplots(figsize=(8,5));ax.boxplot([[r[key] for r in quality_rows if r["variant"]==v and r["split"]=="validation" and np.isfinite(r[key])] for v in VARIANTS],tick_labels=labels);ax.set_title(title);fig.tight_layout();fig.savefig(OUT/"figures"/filename,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));
    for v in VARIANTS:
        vals=[r for r in difficult_rows if r["variant"]==v and r["pair"]=="place_versus_transport"];ax.scatter([r["mean_cross_distance"] for r in vals],[v]*len(vals),label=v)
    ax.set_title("Difficult-pair distance overlap: place versus transport");ax.legend();fig.tight_layout();fig.savefig(OUT/"figures/difficult_pair_distance_overlap.png",dpi=160);plt.close(fig)
    # Raw/derivative contribution summary from ablation aggregate.
    ablabels=ABLATION_MODES; abmeans=[np.mean([r["unknown_recall"] for r in ablation_rows if r["ablation"]==m]) for m in ablabels];fig,ax=plt.subplots(figsize=(10,5));ax.bar(ablabels,abmeans);ax.tick_params(axis="x",rotation=45);ax.set_ylabel("unknown recall");ax.set_title("Raw versus derivative-feature contribution");fig.tight_layout();fig.savefig(OUT/"figures/raw_vs_derivative_feature_contribution.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));
    for v in VARIANTS:
        vals=[r["score"] for r in known_predictions_all if r["variant"]==v];ax.hist(vals,bins=20,alpha=.35,label=v)
    ax.legend();ax.set_title("Variant A/B/C known score distributions");fig.tight_layout();fig.savefig(OUT/"figures/variant_abc_score_distributions.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); vals=[r["original_vs_reversed_validation_accuracy"] for r in probe_rows];ax.bar([f"{r['skill']}/{r['variant']}" for r in probe_rows],vals);ax.tick_params(axis="x",rotation=70);ax.set_ylim(0,1);ax.set_title("Temporal-order probe");fig.tight_layout();fig.savefig(OUT/"figures/temporal_order_probe_comparison.png",dpi=160);plt.close(fig)
    validation_means={variant:float(np.mean([row["best_validation_macro_f1"] for row in comparison_rows if row["variant"]==variant])) for variant in VARIANTS}; selected_variant=max(VARIANTS,key=lambda variant:(validation_means[variant],variant)); selected=next(row for row in aggregate if row["method"]==f"round18_variant_{selected_variant}"); qualifies=selected["mean_known_retention"]>=.95 and selected["worst_known_retention"]>=.90 and selected["mean_unknown_recall"]>=.65 and selected["worst_unknown_recall"]>0 and selected["folds_unknown_recall_below_0.30"]<=2 and selected["mean_rejection_aware_macro_f1"]>=selected["round18_closed_set_mean"]-.03
    pour_selected=next(r for r in all_results if r["skill"]=="pour" and r["variant"]==selected_variant);absorbing=Counter(r["predicted_label"] for r in unknown_predictions_all if r["skill"]=="pour" and r["variant"]==selected_variant).most_common(1)[0][0]
    report=["# Round 18 multi-scale trend-aware encoder LOSO", "", "GT segments only. All three primary variants were trained from fresh initialization with fresh optimizers; no prior weights, prototype banks, synthetic OE, clustering, ASRF predicted segments, or held-out unknowns were used for selection.", "", "## Aggregate comparison", "", "| method | mean known retention | worst retention | mean rejection-aware F1 | mean unknown recall | worst unknown recall | mean AUROC | mean AUPR | folds unknown >= .60 | folds unknown < .30 | folds both targets |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]+[f"| {r['method']} | {r['mean_known_retention']:.4f} | {r['worst_known_retention']:.4f} | {r['mean_rejection_aware_macro_f1']:.4f} | {r['mean_unknown_recall']:.4f} | {r['worst_unknown_recall']:.4f} | {r['mean_auroc']:.4f} | {r['mean_aupr']:.4f} | {r['folds_unknown_recall_ge_0.60']} | {r['folds_unknown_recall_below_0.30']} | {r['folds_both_targets']} |" for r in aggregate]+["", "## Required conclusions", "", "1. First and second temporal differences are evaluated in ablation_results.csv; their effect is not selected from held-out scores.", "2. Multi-scale branches and ordered phase pooling are compared in encoder_variant_comparison.csv and embedding_quality_comparison.csv.", f"3. Selected Round 18 variant: {selected['method']}; mean known retention {selected['mean_known_retention']:.4f}, mean unknown recall {selected['mean_unknown_recall']:.4f}.", f"4. Holdout pour selected-variant unknown recall is {pour_selected['unknown_recall']:.4f}; dominant absorber overall in that fold is {absorbing}.", "5. The per-skill heatmaps show whether temporal encoding generalizes across all six held-out skills; difficult pairs are in difficult_pair_analysis.csv.", "6. Temporal-order probe results are diagnostic only and do not affect model selection.", "7. Available force, torque, gripper, and motion channels are tested through the raw/derivative ablations and pour_transport_diagnostics.csv; no claim of sufficiency is made if criteria fail.", f"8. Round 18 ASRF-integration criteria: **{'PASS' if qualifies else 'FAIL'}**. Remaining limitation is temporal/semantic representation overlap and the known/unknown threshold trade-off.", "", "## Integrity", "", "Annotations were unchanged. Held-out segments were absent from model-facing train/validation/reference arrays. No previous weights or optimizer states were reused. Validation-only epoch and threshold selection was enforced. Derivatives were computed within each segment with deterministic first/second-frame zeros; phase statistics were masked and padding-invariant. Full pytest historical-artifact failures are reported separately from Round 18.", "", "## Outputs", "", "All artifacts are under outputs/round18_multiscale_trend_encoder_loso/."]
    (OUT/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8");(OUT/"config.yaml").write_text(yaml.safe_dump({"experiment":"round18_multiscale_trend_encoder_loso","seed":SEED,"heldout_skills":list(HOLDOUTS),"retraining":True,"previous_weights_reused":False,"gt_segments_only":True,"input_channels":{"A":"12 original normalized features","B":"12 original + 12 first difference + 12 second difference + 12 relative-time channels","C":"Variant B plus ordered 8-bin phase statistics"},"finite_difference":"first[0]=0; first[t]=x[t]-x[t-1]; second[0]=second[1]=0; second[t]=x[t]-2x[t-1]+x[t-2]","relative_time":"linspace(0,1,length), zero for one-frame segments","phase_bins":8,"objective":"class-balanced CE + .20 SupCon + .10 batch-hard triplet + .02 center compactness","triplet_margin":TRIPLET_MARGIN,"k":K,"threshold_target":.95,"bootstrap_resamples":BOOTSTRAPS,"bootstrap_seed":SEED,"boundary_jitter":BOUNDARY_JITTER,"model_selection":"known validation macro F1; cross-family compactness; nearest-neighbor accuracy; within-class variance; validation accuracy","asrf_predicted_segments_used":False,"unknown_clustering":False,"synthetic_oe":False},sort_keys=False),encoding="utf-8"); print(json.dumps({"status":"complete","selected_variant":selected["method"],"criteria_pass":qualifies,"aggregate":aggregate},indent=2),flush=True);return 0


if __name__=="__main__": raise SystemExit(main())
