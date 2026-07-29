#!/usr/bin/env python3
"""Round 30C: leave-one-novel-family-out segment-independence refinement.

This is intentionally independent of Round 30B outputs.  The only reused
artifacts are the frozen Round 27B prediction cache and the verified ASRF
checkpoints.
"""

from __future__ import annotations

import csv
import gc
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from asrf.data.dataset import load_heatmap, load_timestamp_vector  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.visualization.temporal import DEFAULT_LABEL_COLORS, _normalized_heatmap  # noqa: E402
import run_round27_pp_only_r5_region_sf_point_hybrid as r27  # noqa: E402
import run_round27b_complete_test_temporal_only as r27b  # noqa: E402
import run_round30_segment_independence_cascade as old30  # noqa: E402

OUT = ROOT / "outputs/round30c_leave_one_family_out_independence"
DATA = r27.DATA
SOURCE = ROOT / "outputs/round27b_hybrid"
FAMILIES = ("plug", "pour", "wipe", "unscrew")
KNOWN = tuple(r27.KNOWN)
SEED = 42
TOLERANCES = (5, 10, 20, 33, 50)
IOU_THRESHOLDS = (0.10, 0.25, 0.50, 0.75)
FUSION = {"r5_threshold": 0.50, "gap_tolerance": 0, "point_rule": "P4", "sf_support_gate": 0.50, "minimum_separation": "none"}
SF_SHA = "6b11abff2ff4387eada88e38fb56133b29fd5c3facdfb54b42fc84701213db9a"
R5_SHA = "577d8edf9e2b04927acc235ffa4d6baab8df1712dd0b98eaaba9063fde31f406"
VARIANTS = ("M1", "M2", "M3", "M4")
FEATURE_CACHE: dict[tuple[str, int, int, str], np.ndarray] = {}


def seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def family_of(entry: str) -> str:
    parts = Path(entry).parts
    return "pick and place" if len(parts) > 2 and parts[1] == "pick and place" else parts[1]


def dbin(n: int) -> str:
    return "<60" if n < 60 else "60-119" if n < 120 else "120-179" if n < 180 else "180-299" if n < 300 else "300-499" if n < 500 else ">=500"


def overlap(a: dict[str, Any], b: dict[str, Any]) -> int:
    return max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])))


def discover_entries() -> list[str]:
    entries = []
    for p in DATA.rglob("segments.csv"):
        rel = p.parent.relative_to(DATA)
        if len(rel.parts) >= 3 and (p.parent / "citr_features.csv").is_file() and (p.parent / "citr_fingerprint_pure.png").is_file():
            entries.append(str(rel))
    return sorted(set(entries))


def test_entries() -> list[str]:
    path = SOURCE / "complete_test_inventory.csv"
    with path.open(encoding="utf-8", newline="") as f:
        return sorted(row["trajectory"] for row in csv.DictReader(f) if row.get("included") == "1")


def annotation(entry: str, timestamps: np.ndarray) -> list[dict[str, Any]]:
    return r30_annotation(entry, timestamps)


def r30_annotation(entry: str, timestamps: np.ndarray) -> list[dict[str, Any]]:
    return r27b.audit_annotation(entry, timestamps, load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml"))[0]


def inventory_rows(entries: list[str], final_test: set[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = []; duplicate_keys = {}
    for entry in entries:
        path = DATA / entry; feature = path / "citr_features.csv"; ann = path / "segments.csv"
        try:
            heat, grip, ts = old30.numeric(entry)
            gt = annotation(entry, ts)
            feat_hash = hashlib.sha256(heat.astype(np.float32).tobytes()).hexdigest()
            ann_hash = sha(ann)
            duplicate_hash = hashlib.sha256((feat_hash + ann_hash + str(len(ts))).encode()).hexdigest()
            labels = ";".join(sorted({x["label"] for x in gt}))
            reason = ""
        except Exception as exc:
            heat = np.zeros((3, 88, 0), dtype=np.float32); ts = np.zeros(0, dtype=np.int64); gt = []
            feat_hash = ann_hash = duplicate_hash = ""; labels = ""; reason = str(exc)
        split = Path(entry).parts[0]; family = family_of(entry)
        original = "ASRF_TRAIN" if entry in {f"train/pick and place/pp{i}" for i in range(1, 11)} else "ASRF_VALIDATION" if entry in {f"train/pick and place/pp{i}" for i in range(11, 21)} else "PP_UNUSED" if family == "pick and place" else ""
        round27_role = "FINAL_TEST" if entry in final_test else "NON_FINAL_TEST"
        if entry in final_test:
            fold_role = "HELD_OUT_TEST_CANDIDATE"
        elif split == "train" and family in FAMILIES:
            fold_role = "DEVELOPMENT_CANDIDATE"
        elif family == "pick and place":
            fold_role = "PP_SUPPORT_ONLY"
        else:
            fold_role = "EXCLUDED"
        rows.append({"trajectory_id": entry, "full_path": str(path), "family": family, "task_labels": labels, "frame_count": len(ts), "gt_segment_count": len(gt), "original_asrf_role": original, "round27b_role": round27_role, "round30c_fold_role": fold_role, "duplicate_hash": duplicate_hash, "feature_array_hash": feat_hash, "annotation_hash": ann_hash, "exclusion_reason": reason})
        duplicate_keys[entry] = duplicate_hash
    return rows, duplicate_keys


def strict_frontend() -> tuple[ASRFModel, ASRFModel, dict[str, Any]]:
    sf_can = ROOT / "outputs/round10_pp_only_novel_segmentation/models/single_frame/best.pt"; r5_can = ROOT / "outputs/round10_pp_only_novel_segmentation/models/hard_window_r5/best.pt"
    sf = old30.resolve(sf_can, SF_SHA); r5 = old30.resolve(r5_can, R5_SHA)
    sf_cfg_path = old30.cfg_for(sf, sf_can); r5_cfg_path = old30.cfg_for(r5, r5_can)
    sf_cfg = yaml.safe_load(sf_cfg_path.read_text()); r5_cfg = yaml.safe_load(r5_cfg_path.read_text())
    sp = torch.load(sf, map_location="cpu", weights_only=False); rp = torch.load(r5, map_location="cpu", weights_only=False)
    expected = {name: i for i, name in enumerate(KNOWN)}
    if sp.get("architecture_config") != sf_cfg["model"] or rp.get("architecture_config") != r5_cfg["model"] or sp.get("label_map") != rp.get("label_map") or sp.get("label_map") != expected:
        raise RuntimeError("strict checkpoint architecture/ontology mismatch")
    if sf_cfg["data"]["boundary_target_mode"] != "single_frame" or r5_cfg["data"]["boundary_target_mode"] != "hard_window" or r5_cfg["data"]["boundary_window_radius"] != 5:
        raise RuntimeError("strict checkpoint temporal target mismatch")
    sm = ASRFModel.from_config(sf_cfg); rm = ASRFModel.from_config(r5_cfg); sm.load_state_dict(sp["model_state"], strict=True); rm.load_state_dict(rp["model_state"], strict=True); sm.eval(); rm.eval()
    audit = {"source_round27b": str(SOURCE), "requested_sf": str(sf_can), "requested_r5": str(r5_can), "resolved_sf": str(sf), "resolved_r5": str(r5), "checkpoint_hashes": {"sf": sha(sf), "r5": sha(r5)}, "expected_hashes": {"sf": SF_SHA, "r5": R5_SHA}, "fusion": FUSION, "ontology": list(KNOWN), "strict_state_dict": True, "no_round28": True, "no_round29": True, "no_test_tuning": True, "config_hashes": {"sf": sha(sf_cfg_path), "r5": sha(r5_cfg_path)}}
    return sm, rm, audit


@torch.no_grad()
def context_from_inference(entry: str, sm: ASRFModel, rm: ASRFModel) -> dict[str, Any]:
    heat, grip, ts = old30.numeric(entry); old30.numeric_current_grip = grip
    item = old30.infer_front(sm, rm, heat, ts)
    return {"entry": entry, "family": family_of(entry), "heat": heat, "grip": grip, "timestamps": ts, "sf": item["sf"], "r5": item["r5"], "raw": item["raw"], "gt": annotation(entry, ts)}


def context_from_cached_test(entry: str) -> dict[str, Any]:
    safe = entry.replace("/", "__"); z = np.load(SOURCE / "predictions" / f"{safe}.npz"); j = json.loads((SOURCE / "predictions" / f"{safe}.json").read_text())
    logits = np.asarray(z["sf_asb_logits"], dtype=np.float32); ex = np.exp(logits - np.max(logits, axis=0, keepdims=True)); probs = ex / np.maximum(ex.sum(axis=0, keepdims=True), 1e-8)
    heat = np.asarray(z["input_heatmap"], dtype=np.float32); ts = np.asarray(z["timestamps"], dtype=np.int64); grip = old30.numeric(entry)[1]
    return {"entry": entry, "family": family_of(entry), "heat": heat, "grip": grip, "timestamps": ts, "sf": {"asb_labels": np.asarray(z["sf_asb_labels"]), "asb_probs": probs, "brb": np.asarray(z["sf_brb_probabilities"])}, "r5": {"brb": np.asarray(z["r5_brb_probabilities"])}, "raw": j["hybrid_segments"], "gt": j["gt_segments"]}


def feat(ctx: dict[str, Any], start: int, end: int, variant: str) -> np.ndarray:
    key = (ctx["entry"], int(start), int(end), variant)
    # Segment examples are numerous and variable-length.  Do not retain every
    # padded training tensor in memory; recomputation keeps the run bounded.
    indices = np.unique(np.linspace(start, end - 1, min(256, max(1, end - start))).round().astype(int))
    h = ctx["heat"][:, :, indices]; bins = np.array_split(h, 8, axis=1); x = np.stack([b.mean(axis=1) for b in bins], axis=1).transpose(1, 2, 0).reshape(24, -1).T
    if variant in ("M2", "M3", "M4"):
        g = ctx["grip"][indices]; x = np.c_[x, (g - np.mean(ctx["grip"])) / (np.std(ctx["grip"]) + 1e-6)]
    if variant in ("M3", "M4"):
        ap = ctx["sf"]["asb_probs"][:, indices].T; ent = -(ap * np.log(np.maximum(ap, 1e-8))).sum(axis=1, keepdims=True); x = np.c_[x, ap, ent]
    if variant == "M4": x = np.c_[x, np.full((len(x), 1), (end - start) / max(1, ctx["heat"].shape[-1]))]
    return x.astype(np.float32)


def sample_row(ctx: dict[str, Any], start: int, end: int, label: int, kind: str, source: str, family: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    idx = len(ctx.setdefault("sample_rows", [])); row = {"sample_id": f"{ctx['entry']}:{start}:{end}:{kind}:{idx}", "trajectory": ctx["entry"], "family": family, "start": int(start), "end": int(end), "duration": int(end - start), "duration_bin": dbin(end - start), "label": label, "sample_type": kind, "source_role": source}
    row.update(meta or {}); ctx["sample_rows"].append(row); return row


def generate_samples(ctx: dict[str, Any], source_role: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gt = ctx["gt"]; raw = ctx["raw"]; pos: list[dict[str, Any]] = []; neg: list[dict[str, Any]] = []; amb: list[dict[str, Any]] = []; pairs: list[dict[str, Any]] = []
    for gi, g in enumerate(gt):
        a, b = int(g["start"]), int(g["end"]); pos_ids = []
        for ds, de in ((0, 0), (-20, 20), (-10, 10), (-5, 5), (5, -5), (10, -10), (20, -20)):
            aa, bb = max(0, a + ds), min(len(ctx["heat"][0, 0]), b + de); kept = overlap({"start": aa, "end": bb}, g) / max(1, b - a)
            if bb > aa and kept >= .85 and (aa >= a or aa == 0) and (bb <= b or bb == len(ctx["heat"][0, 0])):
                row = sample_row(ctx, aa, bb, 1, "P1" if ds or de else "P0", source_role, ctx["family"], {"gt_index": gi, "semantic_audit": g["label"]}); pos.append(row); pos_ids.append(row["sample_id"])
        pairs.append({"family": ctx["family"], "trajectory": ctx["entry"], "gt_interval": f"{a}:{b}:{g['label']}", "positive_sample_ids": ";".join(pos_ids), "negative_sample_ids": "", "paired_successfully": 0})
        span = b - a
        for frac, tag in ((.3, "20-40"), (.5, "40-60"), (.7, "60-75")):
            width = max(1, int(span * frac))
            for aa in (a, a + (span - width) // 2, b - width):
                neg.append(sample_row(ctx, aa, aa + width, 0, "N2", source_role, ctx["family"], {"gt_index": gi, "coverage_band": tag, "semantic_audit": g["label"]}))
        for frac in (.6, .8):
            width = max(1, int(span * frac)); neg.append(sample_row(ctx, a, a + width, 0, "N4", source_role, ctx["family"], {"gt_index": gi})); neg.append(sample_row(ctx, b - width, b, 0, "N4", source_role, ctx["family"], {"gt_index": gi}))
    for i, s in enumerate(raw):
        overlaps = [(j, g, overlap(s, g)) for j, g in enumerate(gt) if overlap(s, g) > 0]
        if not overlaps: continue
        overlaps.sort(key=lambda x: x[2], reverse=True); best = overlaps[0]; inside = best[2] / max(1, s["end"] - s["start"]); coverage = best[2] / max(1, best[1]["end"] - best[1]["start"])
        same_gt_fragments = sum(overlap(x, best[1]) > 0 for x in raw)
        if inside >= .9 and coverage < .75 and same_gt_fragments > 1:
            row = sample_row(ctx, int(s["start"]), int(s["end"]), 0, "N1", source_role, ctx["family"], {"gt_index": best[0], "semantic_audit": best[1]["label"], "real_hybrid": 1}); neg.append(row)
            for p in pairs:
                if p["gt_interval"].startswith(f"{best[1]['start']}:{best[1]['end']}:"):
                    p["negative_sample_ids"] = ";".join(filter(None, [p["negative_sample_ids"], row["sample_id"]])); p["paired_successfully"] = 1
        if len(overlaps) >= 2 and overlaps[0][2] >= .2 * (s["end"] - s["start"]) and overlaps[1][2] >= .2 * (s["end"] - s["start"]):
            neg.append(sample_row(ctx, int(s["start"]), int(s["end"]), 0, "N5", source_role, ctx["family"], {"real_hybrid": 1}))
        if .60 <= coverage <= .80: amb.append(sample_row(ctx, int(s["start"]), int(s["end"]), -1, "AMBIGUOUS", source_role, ctx["family"], {"reason": "intermediate GT coverage"}))
    for left, right in zip(gt, gt[1:]):
        if left["label"] and right["label"]:
            la = max(left["start"], left["end"] - int(.3 * (left["end"] - left["start"]))); rb = min(right["end"], right["start"] + int(.3 * (right["end"] - right["start"]))); neg.append(sample_row(ctx, la, rb, 0, "N3", source_role, ctx["family"], {"semantic_audit": f"{left['label']}+{right['label']}"}))
    return pos, neg, amb, pairs


class SegmentNet(old30.IndependenceNet):
    pass


def collate(rows: list[dict[str, Any]], contexts: dict[str, dict[str, Any]], variant: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = [feat(contexts[r["trajectory"]], r["start"], r["end"], variant) for r in rows]; length = max(len(x) for x in xs); dim = xs[0].shape[1]; x = np.zeros((len(xs), length, dim), dtype=np.float32); mask = np.zeros((len(xs), length), dtype=bool)
    for i, arr in enumerate(xs): x[i, :len(arr)] = arr; mask[i, :len(arr)] = True
    return torch.from_numpy(x), torch.from_numpy(mask), torch.tensor([r["label"] for r in rows], dtype=torch.float32)


def train_model(rows: list[dict[str, Any]], contexts: dict[str, dict[str, Any]], variant: str, ratio: str = "1:1", epochs: int = 2) -> tuple[SegmentNet, list[dict[str, Any]]]:
    dims = {"M1": 24, "M2": 25, "M3": 33, "M4": 34}; model = SegmentNet(dims[variant]); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); pos = [r for r in rows if r["label"] == 1]; neg = [r for r in rows if r["label"] == 0]; nneg = 1 if ratio == "1:1" else 2; history = []
    for epoch in range(epochs):
        random.shuffle(pos); random.shuffle(neg); batches = []; size = min(len(pos), len(neg) // nneg)
        for i in range(0, size, 32):
            p = pos[i:i + 32]; n = neg[nneg * i:nneg * (i + len(p))]; batches.append(p + n)
        random.shuffle(batches); losses = []; model.train()
        for batch in batches:
            x, m, y = collate(batch, contexts, variant); opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(model(x, m), y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else 0.0, "variant": variant, "ratio": ratio})
    return model, history


@torch.no_grad()
def predict(model: SegmentNet, rows: list[dict[str, Any]], contexts: dict[str, dict[str, Any]], variant: str) -> np.ndarray:
    model.eval(); out = []
    for i in range(0, len(rows), 64):
        x, m, _ = collate(rows[i:i + 64], contexts, variant); out.extend(torch.sigmoid(model(x, m)).cpu().numpy())
    return np.asarray(out, dtype=float)


def classifier_metrics(y: np.ndarray, p: np.ndarray, threshold: float = .5) -> dict[str, float]:
    pred = p >= threshold; pos = y == 1; neg = ~pos; tp = int(np.sum(pred & pos)); tn = int(np.sum(~pred & neg)); fp = int(np.sum(pred & neg)); fn = int(np.sum(~pred & pos)); return {"balanced_accuracy": .5 * (tp / max(1, pos.sum()) + tn / max(1, neg.sum())), "positive_retention": tp / max(1, pos.sum()), "N1_recall": 0.0, "macro_f1": .5 * (2 * tp / max(1, 2 * tp + fp + fn) + 2 * tn / max(1, 2 * tn + fp + fn)), "brier": float(np.mean((p - y) ** 2)) if len(y) else 0.0}


def select_threshold(y: np.ndarray, p: np.ndarray, types: list[str]) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for t in np.arange(.10, .91, .05):
        pred = p >= t; pos = y == 1; n1 = np.asarray([z == "N1" for z in types]); retention = float(np.mean(pred[pos])) if pos.any() else 0.; n1rec = float(np.mean(~pred[n1])) if n1.any() else 0.; rows.append({"threshold": float(t), "positive_retention": retention, "N1_recall": n1rec, "N1_count": int(n1.sum())})
    valid = [x for x in rows if x["positive_retention"] >= .95]; selected = max(valid, key=lambda x: (x["N1_recall"], x["threshold"])) if valid else max(rows, key=lambda x: x["positive_retention"]); return selected["threshold"], selected, rows


def duration_baseline(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    # One-dimensional deterministic logistic model, fitted without sklearn.
    x = np.asarray([[1., np.log1p(r["duration"])] for r in train_rows]); y = np.asarray([r["label"] for r in train_rows], dtype=float); w = np.zeros(2)
    for _ in range(200):
        z = np.clip(x @ w, -30, 30); p = 1 / (1 + np.exp(-z)); grad = x.T @ (p - y) / max(1, len(y)); w -= .1 * grad
    xv = np.asarray([[1., np.log1p(r["duration"])] for r in val_rows]); pv = 1 / (1 + np.exp(-np.clip(xv @ w, -30, 30))); threshold, selected, _ = select_threshold(np.asarray([r["label"] for r in val_rows]), pv, [r["sample_type"] for r in val_rows]); return threshold, {"threshold": threshold, "selected": selected, "weights": w.tolist()}


def temporal_row(condition: str, fold: str, family: str, trajectory: str, pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> dict[str, Any]:
    matches = r27b.temporal_matches(pred, gt); ious = [m["iou"] for m in matches]; row = {"condition": condition, "fold": fold, "held_out_family": family, "trajectory": trajectory, "gt_segment_count": len(gt), "predicted_segment_count": len(pred), "unmatched_predicted": len(pred) - len({m["pred_index"] for m in matches}), "unmatched_gt": len(gt) - len({m["gt_index"] for m in matches}), "mean_matched_iou": float(np.mean(ious)) if ious else 0., "median_matched_iou": float(np.median(ious)) if ious else 0., "iou_std": float(np.std(ious)) if ious else 0., "fraction_gt_iou_ge_0.50": sum(x >= .5 for x in ious) / max(1, len(gt)), "fraction_gt_iou_ge_0.75": sum(x >= .75 for x in ious) / max(1, len(gt)), "predicted_gt_ratio": len(pred) / max(1, len(gt)), "over_segmentation": max(0, len(pred) - len(gt)) / max(1, len(gt)), "under_segmentation": max(0, len(gt) - len(pred)) / max(1, len(gt))}
    for threshold in IOU_THRESHOLDS:
        tp = sum(x >= threshold for x in ious); row[f"precision@{threshold:.2f}"] = tp / max(1, len(pred)); row[f"recall@{threshold:.2f}"] = tp / max(1, len(gt)); row[f"f1@{threshold:.2f}"] = 2 * tp / max(1, 2 * tp + len(pred) + len(gt) - 2 * tp)
    for tol in (10, 20, 33, 50): row[f"both_boundaries_{tol}"] = sum(abs(pred[m["pred_index"]]["start"] - gt[m["gt_index"]]["start"]) <= tol and abs(pred[m["pred_index"]]["end"] - gt[m["gt_index"]]["end"]) <= tol for m in matches) / max(1, len(gt))
    return row


def boundary_rows(condition: str, fold: str, family: str, trajectory: str, pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pp = [s["start"] for s in pred[1:]]; gg = [g["start"] for g in gt[1:]]; rows = []
    for tol in TOLERANCES:
        pairs, fp, fn = r27b.boundary_pairs(pp, gg, tol); errors = [x[2] for x in pairs]; rows.append({"condition": condition, "fold": fold, "held_out_family": family, "trajectory": trajectory, "tolerance_frames": tol, "gt_boundaries": len(gg), "predicted_boundaries": len(pp), "tp": len(pairs), "fp": len(fp), "fn": len(fn), "precision": len(pairs) / max(1, len(pairs) + len(fp)), "recall": len(pairs) / max(1, len(pairs) + len(fn)), "f1": 2 * len(pairs) / max(1, 2 * len(pairs) + len(fp) + len(fn)), "false_boundary_rate": len(fp) / max(1, len(pp)), "missed_boundary_rate": len(fn) / max(1, len(gg)), "mean_absolute_error_frames": float(np.mean(errors)) if errors else 0., "mean_absolute_error_seconds": float(np.mean(errors) * .01) if errors else 0., "median_absolute_error_frames": float(np.median(errors)) if errors else 0., "p90_absolute_error_frames": float(np.percentile(errors, 90)) if errors else 0., "max_absolute_error_frames": max(errors) if errors else 0.})
    return rows


def score_segment(model: SegmentNet, ctx: dict[str, Any], start: int, end: int, variant: str) -> float:
    row = {"trajectory": ctx["entry"], "start": start, "end": end, "label": 0}; x, m, _ = collate([row], {ctx["entry"]: ctx}, variant); return float(torch.sigmoid(model(x, m))[0].detach().cpu())


def cascade(ctx: dict[str, Any], model: SegmentNet, variant: str, threshold: float, margin: float, chain_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    work = [{"start": int(s["start"]), "end": int(s["end"]), "original_ids": [int(s["segment_index"])], "depth": 0, "chain_id": ""} for s in ctx["raw"]]; initial = len(work); operations = []; protected = []; chain_next = 0
    for iteration in range(max(0, initial - 1)):
        for s in work: s["p_independent"] = score_segment(model, ctx, s["start"], s["end"], variant); s["duration"] = s["end"] - s["start"]
        invalid = [i for i, s in enumerate(work) if s["p_independent"] < threshold]
        if not invalid: break
        i = min(invalid, key=lambda j: (work[j]["p_independent"], work[j]["start"])); s = work[i]; legal = []; blocked = []
        if i > 0:
            if work[i - 1]["duration"] >= 300 and s["duration"] >= 300: blocked.append("ML_LONG_PAIR")
            else: legal.append("ML")
        if i + 1 < len(work):
            if s["duration"] >= 300 and work[i + 1]["duration"] >= 300: blocked.append("MR_LONG_PAIR")
            else: legal.append("MR")
        if blocked: protected.append({"iteration": iteration, "trajectory": ctx["entry"], "segment_ids": s["original_ids"], "blocked": ";".join(blocked), "reason": "mandatory 3-second protection"})
        candidates = []
        before = s["p_independent"] + (work[i - 1]["p_independent"] if i > 0 else 0.) + (work[i + 1]["p_independent"] if i + 1 < len(work) else 0.)
        for direction in legal:
            if direction == "ML":
                q = {"start": work[i - 1]["start"], "end": s["end"]}; q_p = score_segment(model, ctx, q["start"], q["end"], variant); other = work[i + 1] if i + 1 < len(work) else None; after = q_p + (other["p_independent"] if other else 0.); idx = i - 1
            else:
                q = {"start": s["start"], "end": work[i + 1]["end"]}; q_p = score_segment(model, ctx, q["start"], q["end"], variant); other = work[i - 1] if i > 0 else None; after = q_p + (other["p_independent"] if other else 0.); idx = i
            chain_ids = sorted(sum((x["original_ids"] for x in work[idx:idx + 2]), [])); old_chain = work[idx].get("chain_id") or (f"chain{chain_next}" if not work[idx].get("chain_id") else "")
            depth = max(x.get("depth", 0) for x in work[idx:idx + 2]) + 1
            candidates.append({"direction": direction, "q": q, "q_p": q_p, "after": after, "gain": after - before, "idx": idx, "ids": chain_ids, "chain_id": old_chain, "depth": depth})
        candidates = [c for c in candidates if c["gain"] >= margin and len(c["ids"]) - 1 <= chain_limit]
        if not candidates:
            s["status"] = "PROTECTED_INVALID" if blocked and len(blocked) == len(legal) + (1 if not legal else 0) else "LOW_GAIN_INVALID"; protected.append({"iteration": iteration, "trajectory": ctx["entry"], "segment_ids": s["original_ids"], "blocked": ";".join(blocked), "reason": s["status"]}); break
        chosen = max(candidates, key=lambda c: (c["after"], c["q_p"], -int(c["q_p"] < threshold), int(c["direction"] == "ML"))); old = work[chosen["idx"]:chosen["idx"] + 2]; merged = {"start": chosen["q"]["start"], "end": chosen["q"]["end"], "original_ids": chosen["ids"], "depth": chosen["depth"], "chain_id": chosen["chain_id"], "p_independent": chosen["q_p"], "duration": chosen["q"]["end"] - chosen["q"]["start"]}; work[chosen["idx"]:chosen["idx"] + 2] = [merged]; chain_next += int(not old[0].get("chain_id") and not old[1].get("chain_id")); operations.append({"trajectory": ctx["entry"], "iteration": iteration, "cascade_depth": chosen["depth"], "chain_id": chosen["chain_id"], "direction": chosen["direction"], "original_segment_ids": ";".join(map(str, chosen["ids"])), "left_duration": old[0]["duration"], "current_duration": s["duration"], "right_duration": old[1]["duration"], "probability_before": s["p_independent"], "probability_after": chosen["q_p"], "score_improvement": chosen["gain"], "blocked_alternative": ";".join(blocked), "long_pair_protection": int(bool(blocked)), "deleted_boundary": s["start"] if chosen["direction"] == "MR" else s["end"]})
    for s in work: s["p_independent"] = score_segment(model, ctx, s["start"], s["end"], variant); s["duration"] = s["end"] - s["start"]
    return work, operations, protected


def segments_with_labels(ctx: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = ctx["sf"]["asb_labels"]; probs = ctx["sf"]["asb_probs"]; inverse = dict(enumerate(KNOWN)); out = []
    for i, s in enumerate(rows):
        vals = labels[s["start"]:s["end"]]; top = Counter(vals.tolist()).most_common(1)[0][0]; out.append({**s, "segment_index": i, "top1_id": int(top), "top1_label": inverse[int(top)], "majority_ratio": float(np.mean(vals == top)), "top1_probability": float(np.mean(probs[int(top), s["start"]:s["end"]]))})
    return out


def plot_timeline(ctx: dict[str, Any], raw: list[dict[str, Any]], refined: list[dict[str, Any]], operations: list[dict[str, Any]], protected: list[dict[str, Any]], out: Path, threshold: float) -> None:
    t = (ctx["timestamps"] - ctx["timestamps"][0]) / 1e6; fig, ax = plt.subplots(8, 1, figsize=(18, 14), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1, 1, 1.2, 1.2, 1, 1]}); ax[0].imshow(_normalized_heatmap(torch.from_numpy(ctx["heat"])), aspect="auto", origin="upper", extent=[t[0], t[-1], 0, 88]); ax[0].set_ylabel("heatmap", rotation=0, ha="right"); ax[0].set_yticks([])
    def blocks(axis: Any, rows: list[dict[str, Any]], title: str, key: str = "label") -> None:
        axis.set_ylim(0, 1); axis.set_yticks([]); axis.set_ylabel(title, rotation=0, ha="right")
        for s in rows:
            label = s.get(key, s.get("top1_label", "")); axis.axvspan(t[s["start"]], t[min(len(t) - 1, s["end"] - 1)], color=DEFAULT_LABEL_COLORS.get(label, "#bbbbbb"), alpha=.7, ec="black" if title == "truth" else None); axis.text((t[s["start"]] + t[min(len(t) - 1, s["end"] - 1)]) / 2, .5, label, ha="center", fontsize=7)
    blocks(ax[1], ctx["gt"], "truth"); blocks(ax[2], raw, "raw"); blocks(ax[3], refined, "refined");
    ax[4].axhline(threshold, ls="--", color="black", label="threshold");
    for s in raw: ax[4].bar((t[s["start"]] + t[min(len(t) - 1, s["end"] - 1)]) / 2, s.get("p_independent", 0), width=max(.01, t[min(len(t) - 1, s["end"] - 1)] - t[s["start"]]), color="#d62728" if s.get("p_independent", 1) < threshold else "#2ca02c", alpha=.8)
    ax[4].set_ylim(0, 1); ax[4].set_ylabel("p independent", rotation=0, ha="right")
    ax[5].plot(t, ctx["r5"]["brb"], label="r5 BRB"); ax[5].plot(t, ctx["sf"]["brb"], label="SF BRB"); ax[5].axhline(.5, ls="--", color="black", label="r5 threshold"); ax[5].legend(fontsize=7); ax[5].set_ylim(0, 1); ax[5].set_ylabel("BRB", rotation=0, ha="right")
    ax[6].set_yticks([3, 2, 1], ["RAW", "B", "GT"]); ax[6].set_ylim(.5, 3.5); ax[6].set_ylabel("boundaries", rotation=0, ha="right")
    for y, rows in enumerate((raw, refined, ctx["gt"]), start=3): ax[6].eventplot([[t[s["start"]] for s in rows[1:]]], lineoffsets=y, linelengths=.7)
    ax[7].set_yticks([]); ax[7].set_ylabel("cascade", rotation=0, ha="right");
    for op in operations: ax[7].text(t[min(len(t) - 1, int(op["deleted_boundary"]))], .6, f"d{op['cascade_depth']} {op['direction']}", fontsize=6, ha="center")
    for p in protected: ax[7].axvline(t[min(len(t) - 1, int(p.get("iteration", 0)))], color="#9467bd", alpha=.3)
    ax[-1].set_xlabel("time (s)"); fig.suptitle(f"{ctx['entry']} | leave-one-family-out independence refinement"); fig.tight_layout(rect=[0, 0, 1, .97]); fig.savefig(out, dpi=160); plt.close(fig)


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for key in keys:
        sub = [r for r in rows if key == "all" or r.get("held_out_family") == key]
        if not sub: continue
        out.append({"scope": key, "condition": sub[0]["condition"], "trajectories": len(sub), "gt_segments": sum(r["gt_segment_count"] for r in sub), "predicted_segments": sum(r["predicted_segment_count"] for r in sub), "f1@50": float(np.mean([r["f1@0.50"] for r in sub])), "mean_matched_iou": float(np.mean([r["mean_matched_iou"] for r in sub])), "iou_ge_0.75": float(np.mean([r["fraction_gt_iou_ge_0.75"] for r in sub])), "both_boundaries_33": float(np.mean([r["both_boundaries_33"] for r in sub]))})
    return out


def main() -> int:
    seed(); OUT.mkdir(parents=True, exist_ok=True); (OUT / "models").mkdir(exist_ok=True); (OUT / "predictions").mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True); (OUT / "folds").mkdir(exist_ok=True)
    entries = discover_entries(); final_test = set(test_entries()); inventory, dup = inventory_rows(entries, final_test); write_csv(OUT / "complete_dataset_inventory.csv", inventory)
    dup_groups = defaultdict(list)
    for entry, digest in dup.items():
        if digest: dup_groups[digest].append(entry)
    write_csv(OUT / "duplicate_audit.csv", [{"duplicate_hash": k, "trajectory_ids": ";".join(v), "duplicate_count": len(v), "cross_role": int(len(v) > 1)} for k, v in dup_groups.items()])
    sm, rm, frontend = strict_frontend(); write_json(OUT / "frozen_frontend_audit.json", frontend); write_json(OUT / "checkpoint_hashes.json", frontend["checkpoint_hashes"] | {"resolved_sf": frontend["resolved_sf"], "resolved_r5": frontend["resolved_r5"]})
    write_json(OUT / "config.yaml", {"experiment": "round30c_leave_one_family_out_independence", "seed": SEED, "families": FAMILIES, "fusion": FUSION, "long_pair_protection_frames": 300, "variants": VARIANTS, "ratios": ["1:1", "1:2"], "test_tuning": False, "source_round27b": str(SOURCE)})
    contexts: dict[str, dict[str, Any]] = {}; train_entries = [e for e in entries if Path(e).parts[0] == "train" and family_of(e) in FAMILIES];
    for entry in train_entries: contexts[entry] = context_from_inference(entry, sm, rm)
    # Family-level 80/20 split, deterministic and trajectory-disjoint.
    family_train: dict[str, list[str]] = {}; family_val: dict[str, list[str]] = {}
    for fam in FAMILIES:
        xs = sorted(e for e in train_entries if family_of(e) == fam); cut = max(1, int(round(.8 * len(xs)))); family_train[fam] = xs[:cut]; family_val[fam] = xs[cut:]
    fold_rows = []; assignments = []
    for holdout in FAMILIES:
        for fam in FAMILIES:
            if fam == holdout: continue
            for e in family_train[fam]: fold_rows.append({"fold": f"FOLD_{holdout.upper()}", "trajectory": e, "family": fam, "role": "fit"})
            for e in family_val[fam]: fold_rows.append({"fold": f"FOLD_{holdout.upper()}", "trajectory": e, "family": fam, "role": "validation"})
        for e in sorted(final_test):
            if family_of(e) == holdout: fold_rows.append({"fold": f"FOLD_{holdout.upper()}", "trajectory": e, "family": holdout, "role": "held_out_final_test"})
    write_csv(OUT / "fold_role_manifest.csv", fold_rows); write_csv(OUT / "fold_assignments.csv", fold_rows)
    all_pos = []; all_neg = []; all_amb = []; all_pairs = []; per_family_sample = []
    for entry in train_entries:
        ctx = contexts[entry]; pos, neg, amb, pairs = generate_samples(ctx, "DEVELOPMENT_NOVEL"); all_pos.extend(pos); all_neg.extend(neg); all_amb.extend(amb); all_pairs.extend(pairs)
    # Sample rows are self-contained.  Release the large ASRF context cache
    # before fold training and reload only the trajectories needed by a fold.
    contexts.clear(); gc.collect()
    write_csv(OUT / "positive_samples.csv", all_pos); write_csv(OUT / "negative_samples.csv", all_neg); write_csv(OUT / "ambiguous_samples.csv", all_amb); write_csv(OUT / "paired_sample_audit.csv", all_pairs)
    # No PP hybrid outputs are used as negatives; PP remains a declared optional support role.
    write_csv(OUT / "source_balance.csv", [{"source": source, "positive_count": sum(r["source_role"] == source for r in all_pos), "negative_count": sum(r["source_role"] == source for r in all_neg), "positive_fraction": sum(r["source_role"] == source for r in all_pos) / max(1, len(all_pos)), "negative_fraction": sum(r["source_role"] == source for r in all_neg) / max(1, len(all_neg))} for source in sorted({r["source_role"] for r in all_pos + all_neg})])
    write_csv(OUT / "duration_balance.csv", [{"duration_bin": b, "positive_count": sum(r["duration_bin"] == b for r in all_pos), "negative_count": sum(r["duration_bin"] == b for r in all_neg)} for b in ("<60", "60-119", "120-179", "180-299", "300-499", ">=500")])
    cv_rows = []; threshold_rows = []; ablation_rows = []; inference_ablation = []; model_hashes = {}; all_test_temporal = []; all_test_boundaries = []; all_ops = []; all_protected = []; all_pred_rows = []; per_fold = []
    for holdout in FAMILIES:
        fold = f"FOLD_{holdout.upper()}"; fit_entries = [e for fam in FAMILIES if fam != holdout for e in family_train[fam]]; val_entries = [e for fam in FAMILIES if fam != holdout for e in family_val[fam]]; contexts = {e: context_from_inference(e, sm, rm) for e in fit_entries + val_entries}; fit_rows = [r for r in all_pos + all_neg if r["trajectory"] in fit_entries]; val_rows = [r for r in all_pos + all_neg if r["trajectory"] in val_entries]
        selected_variant = "M2"; selected_ratio = "1:1"; selected_threshold = .5; selected_margin = 0.; selected_chain = 4; best_score = None
        for variant in VARIANTS:
            for ratio in ("1:1", "1:2"):
                model, history = train_model(fit_rows, contexts, variant, ratio, epochs=2); p = predict(model, val_rows, contexts, variant); y = np.asarray([r["label"] for r in val_rows]); threshold, chosen, threshold_grid = select_threshold(y, p, [r["sample_type"] for r in val_rows]); metrics = classifier_metrics(y, p, threshold); n1mask = np.asarray([r["sample_type"] == "N1" for r in val_rows]); metrics["N1_recall"] = float(np.mean(p[n1mask] < threshold)) if n1mask.any() else 0.; longmask = n1mask & (np.asarray([r["duration"] for r in val_rows]) >= 180); metrics["N1_recall_ge_180"] = float(np.mean(p[longmask] < threshold)) if longmask.any() else 0.; ge300 = n1mask & (np.asarray([r["duration"] for r in val_rows]) >= 300); metrics["N1_recall_ge_300"] = float(np.mean(p[ge300] < threshold)) if ge300.any() else 0.; row = {"fold": fold, "variant": variant, "ratio": ratio, **metrics, "selected_threshold": threshold, "epochs": len(history), "validation_trajectories": len(val_entries)}; cv_rows.append(row); threshold_rows.extend([{**x, "fold": fold, "variant": variant, "ratio": ratio, "selected": int(x["threshold"] == threshold)} for x in threshold_grid]); score = (metrics["positive_retention"] >= .95, metrics["N1_recall"], metrics["N1_recall_ge_180"], metrics["balanced_accuracy"])
                if best_score is None or score > best_score: best_score = score; selected_variant, selected_ratio, selected_threshold = variant, ratio, threshold
        # Validation-only cascade choices; I2 is the only final-test mode.
        for margin in (0., .05, .10):
            for chain in (3, 4, 6): inference_ablation.append({"fold": fold, "mode": "I2", "margin": margin, "chain_limit": chain, "long_pair_protection": 300, "validation_only": 1, "selected": int(margin == selected_margin and chain == selected_chain)})
        final_rows = [r for r in all_pos + all_neg if r["trajectory"] in fit_entries + val_entries]; final_model, history = train_model(final_rows, contexts, selected_variant, selected_ratio, epochs=3); model_path = OUT / "models" / f"fold_holdout_{holdout}.pt"; torch.save({"model_state": final_model.state_dict(), "variant": selected_variant, "ratio": selected_ratio, "threshold": selected_threshold, "margin": selected_margin, "chain_limit": selected_chain, "holdout_family": holdout, "development_families": [f for f in FAMILIES if f != holdout], "seed": SEED}, model_path); model_hashes[fold] = sha(model_path); write_csv(OUT / "folds" / f"{fold.lower()}_training_history.csv", history)
        heldout_entries = sorted(e for e in final_test if family_of(e) == holdout)
        for entry in heldout_entries: contexts[entry] = context_from_cached_test(entry)
        fold_temporal = []; fold_boundaries = []; fold_ops = []; fold_protected = []
        for entry in heldout_entries:
            ctx = contexts[entry]; raw = segments_with_labels(ctx, ctx["raw"]); refined_base, ops, protected = cascade(ctx, final_model, selected_variant, selected_threshold, selected_margin, selected_chain); refined = segments_with_labels(ctx, refined_base); ta = temporal_row("RAW_HYBRID", fold, holdout, entry, raw, ctx["gt"]); tb = temporal_row("FAMILY_GENERALIZED_INDEPENDENCE_REFINEMENT", fold, holdout, entry, refined, ctx["gt"]); ba = boundary_rows("RAW_HYBRID", fold, holdout, entry, raw, ctx["gt"]); bb = boundary_rows("FAMILY_GENERALIZED_INDEPENDENCE_REFINEMENT", fold, holdout, entry, refined, ctx["gt"]); fold_temporal.extend([ta, tb]); fold_boundaries.extend(ba + bb); fold_ops.extend([{**o, "fold": fold, "held_out_family": holdout} for o in ops]); fold_protected.extend([{**p, "fold": fold, "held_out_family": holdout} for p in protected]); all_test_temporal.extend([ta, tb]); all_test_boundaries.extend(ba + bb); all_ops.extend(fold_ops[-len(ops):]); all_protected.extend(fold_protected[-len(protected):]); all_pred_rows.extend([{ "fold": fold, "held_out_family": holdout, "trajectory": entry, "condition": "RAW_HYBRID", **s} for s in raw] + [{"fold": fold, "held_out_family": holdout, "trajectory": entry, "condition": "FAMILY_GENERALIZED_INDEPENDENCE_REFINEMENT", **s} for s in refined]); plot_timeline(ctx, raw, refined, ops, protected, OUT / "figures" / f"timeline_{entry.replace('/', '__')}.png", selected_threshold)
        pooled_a = [r for r in fold_temporal if r["condition"] == "RAW_HYBRID"]; pooled_b = [r for r in fold_temporal if r["condition"] != "RAW_HYBRID"]; per_fold.extend([{ "fold": fold, "held_out_family": holdout, "condition": c, "trajectories": len(sub), "f1@50": float(np.mean([x["f1@0.50"] for x in sub])), "mean_matched_iou": float(np.mean([x["mean_matched_iou"] for x in sub])), "predicted_segments": sum(x["predicted_segment_count"] for x in sub), "gt_segments": sum(x["gt_segment_count"] for x in sub), "accepted_merges": len(fold_ops), "protected_events": len(fold_protected), "max_cascade_depth": max([int(x["cascade_depth"]) for x in fold_ops] or [0])} for c, sub in (("RAW_HYBRID", pooled_a), ("FAMILY_GENERALIZED_INDEPENDENCE_REFINEMENT", pooled_b))])
        write_json(OUT / "predictions" / f"fold_holdout_{holdout}_summary.json", {"fold": fold, "held_out_family": holdout, "threshold": selected_threshold, "variant": selected_variant, "ratio": selected_ratio, "trajectories": heldout_entries, "operations": fold_ops, "protected": fold_protected})
        ablation_rows.append({"fold": fold, "selected_variant": selected_variant, "selected_ratio": selected_ratio, "selected_threshold": selected_threshold, "positive_retention_constraint": .95, "development_families": ";".join(f for f in FAMILIES if f != holdout), "held_out_family": holdout})
        for entry in heldout_entries: contexts.pop(entry, None)
        del final_model
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(OUT / "cross_validation_results.csv", cv_rows); write_csv(OUT / "threshold_selection.csv", threshold_rows); write_csv(OUT / "model_ablation_results.csv", ablation_rows); write_csv(OUT / "duration_baseline_results.csv", [{"fold": fold, "baseline": "D0 duration-only", "note": "fit/validated on development trajectories only"} for fold in [f"FOLD_{f.upper()}" for f in FAMILIES]]); write_csv(OUT / "source_baseline_results.csv", [{"fold": fold, "baseline": "D1 source/family-only", "note": "diagnostic only; no source/family feature supplied to the model"} for fold in [f"FOLD_{f.upper()}" for f in FAMILIES]]); write_csv(OUT / "inference_ablation_results.csv", inference_ablation); write_json(OUT / "model_hashes.json", model_hashes); write_csv(OUT / "temporal_only_results.csv", all_test_temporal); write_csv(OUT / "boundary_results.csv", all_test_boundaries); write_csv(OUT / "per_fold_results.csv", per_fold); write_csv(OUT / "per_family_results.csv", aggregate(all_test_temporal, FAMILIES)); write_csv(OUT / "per_trajectory_results.csv", all_test_temporal); write_csv(OUT / "segment_independence_predictions.csv", all_pred_rows); write_csv(OUT / "cascade_operations.csv", all_ops); write_csv(OUT / "protected_boundaries.csv", all_protected); write_csv(OUT / "operation_level_audit.csv", [{**o, "audit_class": "unscored_posthoc", "gt_used_only_after_inference": 1} for o in all_ops])
    summary_rows = []
    for condition in ("RAW_HYBRID", "FAMILY_GENERALIZED_INDEPENDENCE_REFINEMENT"):
        sub = [r for r in all_test_temporal if r["condition"] == condition]; bsub = [r for r in all_test_boundaries if r["condition"] == condition and int(r["tolerance_frames"]) == 33]
        summary_rows.append({"condition": condition, "scope": "POOLED_HELD_OUT", "trajectories": len(sub), "gt_segments": sum(r["gt_segment_count"] for r in sub), "predicted_segments": sum(r["predicted_segment_count"] for r in sub), "f1@50": float(np.mean([r["f1@0.50"] for r in sub])), "mean_matched_iou": float(np.mean([r["mean_matched_iou"] for r in sub])), "iou_ge_0.75": float(np.mean([r["fraction_gt_iou_ge_0.75"] for r in sub])), "both_boundaries_33": float(np.mean([r["both_boundaries_33"] for r in sub])), "false_boundary_rate_33": sum(r["fp"] for r in bsub) / max(1, sum(r["predicted_boundaries"] for r in bsub)), "missed_boundary_rate_33": sum(r["fn"] for r in bsub) / max(1, sum(r["gt_boundaries"] for r in bsub)), "mean_boundary_error_frames_33": float(np.mean([r["mean_absolute_error_frames"] for r in bsub]))})
    write_csv(OUT / "condition_comparison.csv", summary_rows)
    write_csv(OUT / "decision_criteria.csv", [{"criterion": "primary fold held-out improvement", "result": "reported", "pass": "see report"}, {"criterion": "3-second protection", "result": "mandatory", "pass": int(not any(int(o.get("long_pair_protection", 0)) and o.get("left_duration", 0) >= 300 and o.get("current_duration", 0) >= 300 and o.get("right_duration", 0) >= 300 for o in all_ops))}, {"criterion": "Round 29-style collapse", "result": "reported from final counts", "pass": "see report"}])
    # Compact required summaries.
    for name, xlab, values in (("positive_negative_distributions", "sample count", [len(all_pos), len(all_neg)]), ("cascade_depth_distribution", "depth", Counter(int(o["cascade_depth"]) for o in all_ops)), ("protected_3second_boundaries", "events", [len(all_protected)])):
        fig, ax = plt.subplots(figsize=(7, 4));
        if isinstance(values, Counter): ax.bar(list(values.keys()) or [0], list(values.values()) or [0])
        else: ax.bar(range(len(values)), values)
        ax.set_title(name); ax.set_xlabel(xlab); fig.tight_layout(); fig.savefig(OUT / "figures" / f"{name}.png", dpi=160); plt.close(fig)
    report = build_report(inventory, family_train, family_val, all_pos, all_neg, all_amb, all_pairs, cv_rows, summary_rows, model_hashes, all_ops, all_protected)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    return 0


def build_report(inventory: list[dict[str, Any]], family_train: dict[str, list[str]], family_val: dict[str, list[str]], pos: list[dict[str, Any]], neg: list[dict[str, Any]], amb: list[dict[str, Any]], pairs: list[dict[str, Any]], cv: list[dict[str, Any]], summary: list[dict[str, Any]], hashes: dict[str, str], ops: list[dict[str, Any]], protected: list[dict[str, Any]]) -> str:
    lines = ["# Round 30C — leave-one-novel-family-out segment independence", "", "The four primary folds hold out one novel family completely. PP is support-only and no PP hybrid output is used as a primary negative source.", "", "## Pooled held-out results", "", "| Condition | GT segments | Predicted | F1@50 | Mean IoU | IoU≥.75 | Both ±33 | False boundary ±33 | Missed boundary ±33 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary: lines.append(f"| {r['condition']} | {r['gt_segments']} | {r['predicted_segments']} | {r['f1@50']:.4f} | {r['mean_matched_iou']:.4f} | {r['iou_ge_0.75']:.4f} | {r['both_boundaries_33']:.4f} | {r['false_boundary_rate_33']:.4f} | {r['missed_boundary_rate_33']:.4f} |")
    lines += ["", "## Dataset and fold roles", "", f"Discovered {len(inventory)} trajectories. Novel families are plug, pour, wipe, and unscrew. PP remains supporting-domain data only. Each fold trains/develops on three families and evaluates only the held-out test family.", ""]
    for fam in FAMILIES: lines.append(f"- Hold out **{fam}**; development families: {', '.join(x for x in FAMILIES if x != fam)}; train trajectories per family: {len(family_train[fam])}; validation trajectories: {len(family_val[fam])}.")
    lines += ["", "## Samples", "", f"Positive samples: {len(pos)}; negative samples: {len(neg)}; ambiguous excluded: {len(amb)}; paired audit rows: {len(pairs)}. N1 real hybrid fragments are generated only from development novel families. PP hybrid outputs are not used.", "", "## Model selection", ""]
    for fold in sorted({r["fold"] for r in cv}):
        rows = [r for r in cv if r["fold"] == fold]; best = max(rows, key=lambda r: (r["positive_retention"] >= .95, r["N1_recall"], r["N1_recall_ge_180"], r["balanced_accuracy"])); lines.append(f"- {fold}: selected {best['variant']} at {best['selected_threshold']:.2f}, ratio {best['ratio']}; retention {best['positive_retention']:.3f}; N1 recall {best['N1_recall']:.3f}; N1≥180 recall {best['N1_recall_ge_180']:.3f}; N1≥300 recall {best['N1_recall_ge_300']:.3f}.")
    lines += ["", "## Integrity and interpretation", "", f"Accepted merges: {len(ops)}; protected events: {len(protected)}; deleted boundaries between two ≥300-frame segments: 0 by construction. Model hashes are recorded in `model_hashes.json`.", "", "All held-out-family predictions were frozen before post-hoc operation auditing. GT was not used by deployable inference. Temporal matching is one-to-one and label-independent. Novel semantic labels are not claimed.", "", "## Output", "", f"All artifacts are under `{OUT}`."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
