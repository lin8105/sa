#!/usr/bin/env python
"""Score all 313 frozen ASRF intervals from an aligned [N,11,128] tensor NPZ.

NPZ keys: segments, trajectory_id, pred_segment_id, start_frame, end_frame.
The latter three fields are checked against the bundled frozen score manifest
before inference. Provide the checkpoints and OLD memory separately as release assets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.r2 import KNN_K, SEEDS, THRESHOLD, R2Encoder, cosine_knn_scores
import torch


@torch.no_grad()
def infer(segments: np.ndarray, checkpoints: dict[int, Path], memory: Mapping[str, np.ndarray],
          device: str = "cpu") -> dict[str, np.ndarray]:
    x = torch.as_tensor(segments, dtype=torch.float32, device=device)
    seed_embeddings = {}
    scores = {}
    for seed in SEEDS:
        payload = torch.load(checkpoints[seed], map_location=device, weights_only=False)
        model = R2Encoder().to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        embedding = model(x).cpu().numpy()
        seed_embeddings[seed] = embedding
        scores[f"r2_seed{seed}_score"] = cosine_knn_scores(
            memory[f"seed_{seed}"], embedding, KNN_K
        )
    aggregate = np.median(np.stack([seed_embeddings[seed] for seed in SEEDS]), axis=0)
    aggregate /= np.maximum(np.linalg.norm(aggregate, axis=1, keepdims=True), 1e-12)
    scores["r2_novelty_score"] = cosine_knn_scores(memory["aggregate"], aggregate, KNN_K)
    return scores


def validate_and_order_segments(
    segments: np.ndarray,
    trajectory_ids: np.ndarray,
    segment_ids: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    manifest: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Require exact frozen trajectory/segment/frame identity and order rows."""
    if len(manifest) != 313 or manifest.trajectory_id.nunique() != 30:
        raise ValueError("bundled frozen score manifest is not the expected 30/313 population")
    if manifest.matched_gt_known_or_novel.value_counts().to_dict() != {"known": 238, "novel": 75}:
        raise ValueError("bundled frozen score manifest class counts changed")
    incoming = pd.DataFrame({
        "trajectory_id": np.asarray(trajectory_ids, dtype=str),
        "pred_segment_id": np.asarray(segment_ids, dtype=str),
        "start_frame": np.asarray(starts, dtype=int),
        "end_frame": np.asarray(ends, dtype=int),
        "input_row": np.arange(len(segments)),
    })
    if incoming.pred_segment_id.duplicated().any() or manifest.pred_segment_id.duplicated().any():
        raise ValueError("pred_segment_id must uniquely identify every frozen interval")
    expected = manifest[["trajectory_id", "pred_segment_id", "start_frame", "end_frame"]]
    joined = expected.merge(incoming, how="outer", on="pred_segment_id", suffixes=("_frozen", "_input"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("input segment IDs do not exactly match the frozen ASRF manifest")
    for field in ("trajectory_id", "start_frame", "end_frame"):
        if not joined[f"{field}_frozen"].astype(str).equals(joined[f"{field}_input"].astype(str)):
            raise ValueError(f"input {field} values do not match frozen ASRF intervals")
    ordered = joined.set_index("pred_segment_id").loc[manifest.pred_segment_id]
    indices = ordered.input_row.to_numpy(dtype=int)
    return np.asarray(segments)[indices], manifest.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segments_npz", type=Path)
    parser.add_argument("memory_npz", type=Path)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with np.load(args.segments_npz, allow_pickle=False) as archive:
        segments = np.asarray(archive["segments"], dtype=np.float32)
        trajectory_id = archive["trajectory_id"].astype(str)
        segment_id = archive["pred_segment_id"].astype(str)
        starts = np.asarray(archive["start_frame"], dtype=int)
        ends = np.asarray(archive["end_frame"], dtype=int)
    manifest_path = Path(__file__).resolve().parents[1] / "results" / "held_out_test_asrf_segment_r2_scores.csv"
    manifest = pd.read_csv(manifest_path)
    segments, manifest = validate_and_order_segments(segments, trajectory_id, segment_id,
                                                     starts, ends, manifest)
    if segments.shape != (313, 11, 128):
        raise ValueError(f"expected frozen final tensors [313,11,128], got {segments.shape}")
    checkpoints = {seed: args.checkpoint_dir / f"seed_{seed}_best.pt" for seed in SEEDS}
    if not all(path.is_file() for path in checkpoints.values()):
        raise FileNotFoundError("release assets must provide seed_84/184/284 best checkpoints")
    with np.load(args.memory_npz, allow_pickle=False) as archive:
        memory = {key: np.asarray(archive[key], dtype=np.float32)
                  for key in ("seed_84", "seed_184", "seed_284", "aggregate")}
    scores = infer(segments, checkpoints, memory, args.device)
    result = pd.DataFrame({"trajectory_id": manifest.trajectory_id,
                           "pred_segment_id": manifest.pred_segment_id,
                           "start_frame": manifest.start_frame,
                           "end_frame": manifest.end_frame,
                           **scores,
                           "r2_frozen_threshold": THRESHOLD,
                           "frozen_old_unknown_prediction": np.where(
                               scores["r2_novelty_score"] >= THRESHOLD, "UNKNOWN", "OLD")})
    result.to_csv(args.output_csv, index=False)
    print(f"wrote {len(result)} segment scores across {result.trajectory_id.nunique()} trajectories")


if __name__ == "__main__":
    main()
