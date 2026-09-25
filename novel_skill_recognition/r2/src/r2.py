"""Round94-compatible R2 representation and OLD-memory scorer.

The functions operate on already aligned canonical CITR channels. Source-data
loading and canonical CITR extraction remain the responsibility of the dataset
pipeline; see README.md for the frozen Round94 preprocessing contract.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEEDS = (84, 184, 284)
EMBEDDING_DIM = 128
KNN_K = 20
THRESHOLD = 0.3693014085292816
EPS = 1e-12


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class R2Encoder(nn.Module):
    """Exact Round84-style [B,11,128] to normalized [B,128] encoder."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(11, 64, 5, padding=2), nn.GroupNorm(8, 64), nn.GELU()
        )
        self.blocks = nn.Sequential(*(ResidualBlock(64, d) for d in (1, 2, 4, 8)))
        self.fc = nn.Linear(64, EMBEDDING_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(self.blocks(self.stem(x)).mean(-1)), dim=1)


def resample_segment(value: np.ndarray, length: int = 128) -> np.ndarray:
    """Linear endpoint-aligned resampling, matching Round93's np.interp path."""
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] != 11 or value.shape[1] < 2:
        raise ValueError(f"expected [11,T>=2], got {value.shape}")
    old_x = np.linspace(0.0, 1.0, value.shape[1])
    new_x = np.linspace(0.0, 1.0, length)
    return np.vstack([np.interp(new_x, old_x, row) for row in value]).astype(np.float32)


def old_only_trajectory_scale(raw_citr: np.ndarray, old_mask: np.ndarray) -> np.ndarray:
    """Max-absolute per channel over OLD frames; novel frames never set scale.

    Args: raw_citr [10,T] canonical numeric CITR scalar features; old_mask [T].
    """
    raw = np.asarray(raw_citr, dtype=np.float64)
    mask = np.asarray(old_mask, dtype=bool)
    if raw.ndim != 2 or raw.shape[0] != 10 or raw.shape[1] != len(mask):
        raise ValueError("raw_citr must be [10,T] and align with old_mask")
    if not mask.any():
        raise ValueError("trajectory has no OLD frames for Round94 normalization")
    scale = np.max(np.abs(raw[:, mask]), axis=1)
    scale[scale == 0] = 1.0
    return scale


def prepare_asrf_segment(
    raw_citr: np.ndarray,
    gripper_norm: np.ndarray,
    scale: np.ndarray,
    start_frame: int,
    end_frame: int,
    *,
    end_inclusive: bool = True,
) -> np.ndarray:
    """Apply trajectory scale and slice one frozen ASRF proposal.

    The frozen Round94 source manifests store inclusive end_frame values.
    Set end_inclusive=False only when adapting a half-open source manifest.
    """
    raw = np.asarray(raw_citr, dtype=np.float64)
    grip = np.asarray(gripper_norm, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != 10 or raw.shape[1] != len(grip):
        raise ValueError("expected aligned raw CITR [10,T] and gripper [T]")
    if scale.shape != (10,):
        raise ValueError("scale must contain one value for each of ten CITR rows")
    a, b = int(start_frame), int(end_frame) + int(end_inclusive)
    if a < 0 or b > raw.shape[1] or b - a < 2:
        raise ValueError(f"invalid segment [{a},{b}) for T={raw.shape[1]}")
    interaction = raw[:, a:b] / np.maximum(scale[:, None], EPS)
    return resample_segment(np.vstack((interaction, grip[a:b][None, :])))


def augment(x: torch.Tensor, channel_std: torch.Tensor, seed: int) -> torch.Tensor:
    """Frozen paired-view augmentation: crop 122/128, shift ±2, scale ±5%, noise 2%."""
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    batch = x.shape[0]
    starts = torch.randint(0, 7, (batch,), generator=generator, device=x.device)
    outputs = []
    for i in range(batch):
        start = int(starts[i])
        view = F.interpolate(
            x[i : i + 1, :, start : start + 122],
            size=128,
            mode="linear",
            align_corners=True,
        )
        shift = int(torch.randint(-2, 3, (1,), generator=generator, device=x.device))
        view = torch.roll(view, shift, dims=2)
        amplitude = 1.0 + 0.05 * (
            2 * torch.rand((1, 11, 1), generator=generator, device=x.device) - 1
        )
        noise = 0.02 * channel_std.view(1, 11, 1) * torch.randn(
            view.shape, generator=generator, device=x.device
        )
        outputs.append(view * amplitude + noise)
    return torch.cat(outputs, dim=0)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric cross-view NT-Xent used by frozen R2 (in-batch paired positives)."""
    logits = z1 @ z2.T / temperature
    target = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_seed(
    x_train: np.ndarray,
    x_dev: np.ndarray,
    seed: int,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    epochs: int = 60,
    batch_size: int = 64,
    patience: int = 12,
) -> list[dict[str, float]]:
    """Round84/94 OLD-only trainer; caller must pass only frozen OLD TRAIN/DEV."""
    device = torch.device(device)
    seed_all(seed)
    train = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    dev = torch.as_tensor(x_dev, dtype=torch.float32, device=device)
    if train.ndim != 3 or train.shape[1:] != (11, 128) or dev.ndim != 3:
        raise ValueError("training tensors must have shape [N,11,128]")
    std = torch.from_numpy(np.maximum(np.asarray(x_train).std((0, 2)), 1e-5).astype(np.float32)).to(device)
    model = R2Encoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_similarity, best_state, stale, history = -np.inf, None, 0, []
    checkpoint_path = Path(checkpoint_path)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train), device=device)
        losses, similarities = [], []
        for start in range(0, len(permutation), batch_size):
            batch = train[permutation[start : start + batch_size]]
            v1 = augment(batch, std, seed * 100000 + epoch * 100 + start)
            v2 = augment(batch, std, seed * 100000 + epoch * 100 + start + 1)
            z1, z2 = model(v1), model(v2)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            similarities.append(float((z1 * z2).sum(1).mean().detach().cpu()))
        model.eval()
        with torch.no_grad():
            d1 = model(augment(dev, std, seed * 100000 + epoch * 11))
            d2 = model(augment(dev, std, seed * 100000 + epoch * 11 + 1))
            similarity = float((d1 * d2).sum(1).mean().cpu())
            embedding = d1.cpu().numpy()
            variance = float(np.mean(np.var(embedding, axis=0)))
            singular = np.linalg.svd(embedding - embedding.mean(0), compute_uv=False)
            power = singular * singular
            effective_rank = float(power.sum() ** 2 / (np.sum(power * power) + EPS))
        collapsed = variance < 1e-5 or effective_rank < 2.0
        selection_score = similarity - float(collapsed)
        history.append({"epoch": epoch, "ssl_loss": float(np.mean(losses)),
                        "positive_similarity": float(np.mean(similarities)),
                        "dev_similarity": similarity, "embedding_variance": variance,
                        "effective_rank": effective_rank, "collapse_flag": float(collapsed)})
        if selection_score > best_similarity:
            best_similarity = selection_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("no finite checkpoint selected")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "seed": seed,
                "architecture": "Round84-style OLD-only SSL encoder", "temperature": 0.1,
                "augmentation": {"crop_frames": 122, "jitter_frames": 2,
                                 "amplitude_fraction": 0.05, "noise_fraction": 0.02}}, checkpoint_path)
    return history


def load_memory(path: str | Path, key: str = "aggregate") -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive[key], dtype=np.float32)


@torch.no_grad()
def build_old_memory(
    checkpoints: Mapping[int | str, str | Path],
    old_train_segments: np.ndarray,
    output_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Encode only caller-selected OLD TRAIN segments and persist all seed banks."""
    x = torch.as_tensor(old_train_segments, dtype=torch.float32, device=device)
    if x.ndim != 3 or x.shape[1:] != (11, 128):
        raise ValueError("OLD TRAIN segments must have shape [N,11,128]")
    by_seed = {}
    for seed in SEEDS:
        model = R2Encoder().to(device)
        path = checkpoints[seed] if seed in checkpoints else checkpoints[str(seed)]
        payload = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        by_seed[f"seed_{seed}"] = model(x).cpu().numpy()
    median = np.median(np.stack([by_seed[f"seed_{seed}"] for seed in SEEDS]), axis=0)
    aggregate = median / np.maximum(np.linalg.norm(median, axis=1, keepdims=True), EPS)
    by_seed["aggregate"] = aggregate.astype(np.float32)
    np.savez_compressed(output_path, **by_seed)
    return aggregate


@torch.no_grad()
def aggregate_embeddings(
    checkpoints: Mapping[int | str, str | Path],
    segments: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Embed with three frozen seeds, take coordinate-wise median, then L2-normalize."""
    if set(map(str, checkpoints)) != set(map(str, SEEDS)):
        raise ValueError(f"expected checkpoints for seeds {SEEDS}")
    x = torch.as_tensor(segments, dtype=torch.float32, device=device)
    if x.ndim != 3 or x.shape[1:] != (11, 128):
        raise ValueError("segments must have shape [N,11,128]")
    values = []
    for seed in SEEDS:
        model = R2Encoder().to(device)
        payload = torch.load(checkpoints[seed] if seed in checkpoints else checkpoints[str(seed)],
                             map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        values.append(model(x).cpu().numpy())
    median = np.median(np.stack(values), axis=0)
    return median / np.maximum(np.linalg.norm(median, axis=1, keepdims=True), EPS)


def cosine_knn_scores(memory: np.ndarray, query: np.ndarray, k: int = KNN_K) -> np.ndarray:
    memory = np.asarray(memory, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    memory = memory / np.maximum(np.linalg.norm(memory, axis=1, keepdims=True), EPS)
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), EPS)
    if len(memory) < k:
        raise ValueError(f"OLD memory has {len(memory)} rows; k={k} requires at least {k}")
    distances = 1.0 - np.clip(query @ memory.T, -1.0, 1.0)
    indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
    return distances[np.arange(len(query))[:, None], indices].mean(axis=1)


def threshold_from_old_dev(scores: np.ndarray, target_fur: float = 0.10) -> tuple[float, float]:
    """Round94 threshold rule: smallest observed OLD-DEV cutoff meeting target FUR."""
    scores = np.asarray(scores, dtype=np.float64)
    values = np.sort(np.unique(scores))
    feasible = [float(v) for v in values if np.mean(scores >= v) <= target_fur + 1e-12]
    threshold = min(feasible) if feasible else float(np.nextafter(values[-1], np.inf))
    return threshold, float(np.mean(scores >= threshold))
