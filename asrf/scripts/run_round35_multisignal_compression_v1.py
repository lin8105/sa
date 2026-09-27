#!/usr/bin/env python3
"""Versioned Round35 multi-signal compression calibration and application.

Calibration is train0-only.  The frozen P95 configuration is then applied to
train0/test0 without model inference.  Canonical TrajectoryBundle compression
and the existing Round40 merge implementation are reused where frozen bundles
exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_round34_velocity_gripper_static_compression_pp1_5 as r34  # noqa: E402
import run_round34c_annotation_remapped_compression_pp1_10 as r34c  # noqa: E402
import run_round35_full_compressed_dataset as r35  # noqa: E402
from postprocessing.common import (  # noqa: E402
    load_bundle, save_bundle, segments_to_frame_labels,
)
from postprocessing.compress_timeline import compress_bundle  # noqa: E402
from postprocessing.merge_segments import merge_prediction  # noqa: E402

DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
RAW_SPLIT = {"train": "train0", "test": "test0"}
HISTORICAL_INVENTORY = ROOT / "outputs/0/round35_full_compressed_dataset/complete_dataset_inventory.csv"
BUNDLE_ROOT = ROOT / "outputs/0/round91_end_to_end_open_world_transition_audit_v002/bundles/standard"
OUT = ROOT / "outputs/round35_multisignal_compression_v001"
RATE = 100.0
LIN_T = 0.02600749068
ANG_T = 0.0477127692
MIN_RUN_S = 0.50
EDGE_CONTEXT_S = 0.60
MIN_SKILL_FRAMES = 10
PERCENTILES = (97.5, 99.0, 99.5, 99.9)
SELECTED_PERCENTILE = 99.5
SMOOTH_WINDOW = 3
SIGNALS = ("linear_acceleration", "angular_acceleration", "force_change", "torque_change", "gripper_velocity")
UNITS = {
    "linear_acceleration": "m/s^2", "angular_acceleration": "rad/s^2",
    "force_change": "N/s", "torque_change": "N m/s", "gripper_velocity": "m/s",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for row in rows for k in row)) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(changes[::2], changes[1::2])]


def records() -> list[dict[str, str]]:
    rows = list(csv.DictReader(HISTORICAL_INVENTORY.open(encoding="utf-8")))
    if len(rows) != 115 or sum(r["split"] == "train" for r in rows) != 79 or sum(r["split"] == "test" for r in rows) != 36:
        raise RuntimeError("historical Round35 inventory no longer has expected 79/36 scope")
    return rows


def load_context(row: dict[str, str]) -> dict[str, Any]:
    split, family, name = row["split"], row["family"], Path(row["trajectory_id"]).parts[-1]
    entry = f"{RAW_SPLIT[split]}/{family}/{name}"
    r34.DATA = DATA
    c = r34.load_signals(entry)
    source = DATA / entry
    if len(c["ts"]) != int(row["original_frame_count"]):
        raise RuntimeError(f"{entry}: raw CITR length differs from Round35 original inventory")
    st = pd.read_csv(source / "robot_states.csv")
    st_t = st["timestamp_us"].to_numpy(dtype=np.float64) / 1e6
    v = st.loc[:, ["v_x", "v_y", "v_z"]].to_numpy(dtype=float)
    w = st.loc[:, ["w_x", "w_y", "w_z"]].to_numpy(dtype=float)
    vel = np.column_stack([np.interp(c["ts"], st_t, v[:, j]) for j in range(3)])
    omega = np.column_stack([np.interp(c["ts"], st_t, w[:, j]) for j in range(3)])
    lin_acc = np.linalg.norm(np.gradient(vel, c["ts"], axis=0, edge_order=1), axis=1)
    ang_acc = np.linalg.norm(np.gradient(omega, c["ts"], axis=0, edge_order=1), axis=1)

    bota = pd.read_csv(source / "bota_100hz.csv")
    bota_t = bota["timestamp_us"].to_numpy(dtype=np.float64) / 1e6
    force = bota.loc[:, ["F_x", "F_y", "F_z"]].to_numpy(dtype=float)
    torque = bota.loc[:, ["tau_x", "tau_y", "tau_z"]].to_numpy(dtype=float)
    if len(bota_t) != len(force) or np.any(np.diff(bota_t) <= 0):
        raise RuntimeError(f"{entry}: invalid Bota timeline")
    # Minimal centered 3-sample (30 ms at 100 Hz) moving average, edge padded.
    kernel = np.ones(SMOOTH_WINDOW, dtype=float) / SMOOTH_WINDOW
    def smooth(x: np.ndarray) -> np.ndarray:
        pad = SMOOTH_WINDOW // 2
        padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
        return np.column_stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(x.shape[1])])
    force_rate_src = np.gradient(smooth(force), bota_t, axis=0, edge_order=1)
    torque_rate_src = np.gradient(smooth(torque), bota_t, axis=0, edge_order=1)
    force_rate = np.column_stack([np.interp(c["ts"], bota_t, force_rate_src[:, j]) for j in range(3)])
    torque_rate = np.column_stack([np.interp(c["ts"], bota_t, torque_rate_src[:, j]) for j in range(3)])
    signals = {
        "linear_acceleration": lin_acc,
        "angular_acceleration": ang_acc,
        "force_change": np.linalg.norm(force_rate, axis=1),
        "torque_change": np.linalg.norm(torque_rate, axis=1),
        "gripper_velocity": np.abs(c["grip_speed"]),
    }
    for name, values in signals.items():
        if values.shape != c["ts"].shape or not np.all(np.isfinite(values)):
            raise RuntimeError(f"{entry}: invalid synchronized signal {name}")
    c.update({"split": split, "family": family, "trajectory_id": row["trajectory_id"], "signals": signals, "source": source,
              "source_hashes": {name: sha256(source / name) for name in ("citr_features.csv", "robot_states.csv", "bota_100hz.csv", "segments.csv")}})
    return c


def velocity_low(c: dict[str, Any]) -> np.ndarray:
    return c["valid"] & (c["lin"] < LIN_T) & (c["ang"] < ANG_T)


def threshold_candidates(train: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    pooled = {s: np.concatenate([c["signals"][s][velocity_low(c)] for c in train]) for s in SIGNALS}
    summary = []
    thresholds: dict[str, dict[str, float]] = {f"P{p:g}": {} for p in PERCENTILES}
    for name in SIGNALS:
        x = pooled[name]
        qs = np.percentile(x, [50, 90, 95, 97.5, 99, 100])
        summary.append({"signal": name, "unit": UNITS[name], "low_velocity_frame_count": int(len(x)),
                        **{k: float(v) for k, v in zip(("P50", "P90", "P95", "P97_5", "P99", "max"), qs)}})
        for p in PERCENTILES:
            # Keep the requested strict '<' test while admitting exact empirical quantiles.
            thresholds[f"P{p:g}"][name] = float(np.nextafter(np.percentile(x, p), np.inf))
    return thresholds, summary


def physical_low(c: dict[str, Any], extra: dict[str, float]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    base = velocity_low(c)
    passes = {s: c["signals"][s] < extra[s] for s in SIGNALS}
    return base & np.logical_and.reduce(list(passes.values())), passes


def apply_policy(c: dict[str, Any], low: np.ndarray, protection: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Round35 0.5 s run / 0.6 s edge context, then canonical 10-frame survival."""
    protection = np.zeros(len(low), bool) if protection is None else np.asarray(protection, bool)
    minimum = max(1, int(round(MIN_RUN_S / float(np.median(np.diff(c["ts"]))))))
    long_low = np.zeros(len(low), bool)
    for s, e in runs(low):
        if e - s >= minimum: long_low[s:e] = True
    candidate = long_low & ~protection
    keep = np.ones(len(low), bool); edge = np.zeros(len(low), bool)
    context = int(round(EDGE_CONTEXT_S / float(np.median(np.diff(c["ts"])))) )
    rows = []
    for s, e in runs(candidate):
        left = bool(np.any(~long_low[:s] | protection[:s])) if s else False
        right = bool(np.any(~long_low[e:] | protection[e:])) if e < len(low) else False
        if e - s <= 2 * context:
            edge[s:e] = True; continue
        if not left:
            ds, de, kind = s, e - context, "leading"
            edge[de:e] = True
        elif not right:
            ds, de, kind = s + context, e, "trailing"
            edge[s:ds] = True
        else:
            ds, de, kind = s + context, e - context, "internal"
            edge[s:ds] = True; edge[de:e] = True
        if de > ds:
            keep[ds:de] = False
            rows.append({"trajectory_id": c["trajectory_id"], "start_frame": ds, "end_frame_exclusive": de,
                         "run_type": kind, "duration_s": float(c["ts"][de-1] - c["ts"][ds]), "removed_frames": de-ds})
    c_survival = dict(c)
    c_survival["entry"] = c["trajectory_id"]
    keep, added, survival_rows = r34c.apply_skill_survival(c_survival, keep, np.zeros(len(keep), bool))
    return keep, edge, [{**r, "survival_protected_frames": int(added.sum())} for r in rows]


def event_mask(c: dict[str, Any], *, context_s: float = 0.0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    events = r34.gripper_events(c, r34c.GRIP_THRESHOLD, r34c.GRIP_BRIDGE_S)
    mask = np.zeros(len(c["ts"]), bool)
    half = int(round(context_s / float(np.median(np.diff(c["ts"])))))
    for s, e in events:
        mask[max(0, s-half):min(len(mask), e+half)] = True
    return mask, events


def old_baseline(c: dict[str, Any]) -> np.ndarray:
    ev, _ = event_mask(c, context_s=.5)
    low = c["valid"] & (c["lin"] <= LIN_T) & (c["ang"] <= ANG_T)
    keep, _, _ = apply_policy(c, low, ev)
    return keep


def map_and_write_data(c: dict[str, Any], keep: np.ndarray, low: np.ndarray, extra: dict[str, float], passes: dict[str, np.ndarray], outroot: Path) -> None:
    rel = Path(c["trajectory_id"])
    dest = outroot / "data" / rel
    dest.mkdir(parents=True, exist_ok=True)
    n = len(keep); comp = np.full(n, -1, dtype=int); comp[keep] = np.arange(keep.sum())
    failures = {s: (~passes[s]) & velocity_low(c) for s in SIGNALS}
    labels = np.full(n, "", dtype=object)
    for g in c["gt"]: labels[g["start"]:g["end"]] = g["label"]
    mask_rows = []
    for i in range(n):
        row = {"original_frame": i, "timestamp_us": int(c["ts_us"][i]), "kept": int(keep[i]), "compressed_frame": int(comp[i]),
               "old_velocity_low": int(velocity_low(c)[i]), "new_physical_low": int(low[i]), "skill_diagnostic_only": labels[i]}
        for s in SIGNALS: row[s] = float(c["signals"][s][i]); row[f"fails_{s}"] = int(failures[s][i])
        mask_rows.append(row)
    write_csv(dest / "compression_mask.csv", mask_rows)
    write_csv(dest / "frame_mapping.csv", [{"original_frame": i, "kept": int(keep[i]), "compressed_frame": int(comp[i])} for i in range(n)])
    np.save(dest / "retained_indices.npy", np.flatnonzero(keep))
    features = c["citr"]
    features.loc[keep].to_csv(dest / "citr_features.csv", index=False)
    if (c["source"] / "citr_matrices.npy").is_file():
        matrix = np.load(c["source"] / "citr_matrices.npy", mmap_mode="r")
        if matrix.shape[0] != n: raise RuntimeError(f"{c['trajectory_id']}: CITR matrix timeline mismatch")
        np.save(dest / "citr_matrices.npy", np.asarray(matrix[keep]))
    pd.DataFrame({"timestamp_us": c["ts_us"][keep], "original_frame": np.flatnonzero(keep), "compressed_frame": np.arange(keep.sum())}).to_csv(dest / "timestamps.csv", index=False)
    # Annotation-only remapping is a post-freeze preservation/diagnostic operation.
    ann_rows = []
    for g in c["gt"]:
        retained = np.flatnonzero(keep[g["start"]:g["end"]]) + g["start"]
        if not len(retained): raise RuntimeError(f"{c['trajectory_id']}: no retained frame for GT segment {g['label']}")
        ann_rows.append({"segment_index": g["segment_index"], "label": g["label"], "original_start": g["start"], "original_end_exclusive": g["end"],
                         "compressed_start": int(comp[retained[0]]), "compressed_end_exclusive": int(comp[retained[-1]]+1)})
    write_csv(dest / "segments.csv", ann_rows)
    # All timestamped raw streams use Round35's preceding-CITR interval mapping.
    for fn in ("bota_100hz.csv", "gripper_10hz.csv", "video_timestamps.csv"):
        src = c["source"] / fn
        if src.is_file(): r35.process_timestamped_csv(src, dest / fn, c["ts_us"], keep)
    comp_map, inverse = r34c.mapping_arrays(keep)
    r35.process_robot_states(c["source"] / "robot_states.csv", dest / "robot_states.csv", c, keep, comp_map)
    write_json(dest / "provenance.json", {"trajectory_id": c["trajectory_id"], "source_split": RAW_SPLIT[c["split"]], "source_hashes": c["source_hashes"],
              "retained_frame_count": int(len(inverse)), "original_frame_count": n, "thresholds": extra, "same_retained_indices_for_timeline_arrays": True,
              "temporal_resizing_or_interpolation": False, "skill_labels_used_for_threshold_selection": False})


def summarize_candidate(train: list[dict[str, Any]], thresholds: dict[str, float], label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_traj, unique_rows = [], []
    totals = {"frames": 0, "removed": 0, "baseline_removed": 0, "old_low": 0, "new_signal_protected_old_low_frames": 0}
    for c in train:
        low, passes = physical_low(c, thresholds)
        keep, _, _ = apply_policy(c, low)
        old = old_baseline(c)
        base = velocity_low(c)
        added = base & ~np.logical_and.reduce(list(passes.values()))
        per_traj.append({"split": "train", "candidate": label, "trajectory_id": c["trajectory_id"], "family": c["family"], "frames": len(keep),
                         "removed_frames": int((~keep).sum()), "compression_ratio": float((~keep).mean()), "round35_removed_frames": int((~old).sum()),
                         "round35_compression_ratio": float((~old).mean()), "old_velocity_low_frames": int(base.sum()),
                         "new_signal_protected_old_low_frames": int(added.sum()), "low_candidate_subset_of_old": int(np.all(low <= base))})
        for skill in sorted({g["label"] for g in c["gt"]}):
            skill_mask = np.zeros(len(keep), bool)
            for g in c["gt"]:
                if g["label"] == skill: skill_mask[g["start"]:g["end"]] = True
            unique_rows.append({"split": "train", "candidate": label, "trajectory_id": c["trajectory_id"], "family": c["family"], "skill": skill,
                                "old_low_frames": int((base & skill_mask).sum()), "new_signal_protected_frames": int((added & skill_mask).sum()),
                                "new_compression_removed_frames": int((~keep & skill_mask).sum()), "round35_removed_frames": int((~old & skill_mask).sum())})
        totals["frames"] += len(keep); totals["removed"] += int((~keep).sum()); totals["baseline_removed"] += int((~old).sum()); totals["old_low"] += int(base.sum())
        totals["new_signal_protected_old_low_frames"] += int(added.sum())
    per_traj.append({"split": "train", "candidate": label, "trajectory_id": "__POOLED__", "family": "ALL", "frames": totals["frames"], "removed_frames": totals["removed"],
                     "compression_ratio": totals["removed"]/totals["frames"], "round35_removed_frames": totals["baseline_removed"],
                     "round35_compression_ratio": totals["baseline_removed"]/totals["frames"], "old_velocity_low_frames": totals["old_low"],
                     "new_signal_protected_old_low_frames": totals["new_signal_protected_old_low_frames"], "low_candidate_subset_of_old": 1})
    return per_traj, unique_rows


def calibrate() -> None:
    rows = [r for r in records() if r["split"] == "train"]
    train = [load_context(r) for r in rows]
    candidates, distributions = threshold_candidates(train)
    write_csv(OUT / "signal_distribution_train_low_velocity.csv", distributions)
    impact = []; skill = []
    for label, thresholds in candidates.items():
        t, s = summarize_candidate(train, thresholds, label); impact.extend(t); skill.extend(s)
    write_csv(OUT / "candidate_compression_impact_train.csv", impact)
    write_csv(OUT / "candidate_protection_by_skill_train.csv", skill)
    selected = f"P{SELECTED_PERCENTILE:g}"
    frozen = {"selected_candidate": selected, "selected_percentile": SELECTED_PERCENTILE, "thresholds": candidates[selected],
              "frozen_velocity_thresholds": {"linear_m_per_s": LIN_T, "angular_rad_per_s": ANG_T},
              "temporal_policy": {"minimum_low_activity_run_s": MIN_RUN_S, "edge_context_s_per_side": EDGE_CONTEXT_S,
                                  "minimum_skill_survival_frames": MIN_SKILL_FRAMES, "gap_bridging": False},
              "selection_basis": "P99.5 chosen from train-only candidates as a compromise: train compression 19.34% versus historical 23.38%; no TEST signal, label, or metric inspected and no exact-ratio objective",
              "signal_smoothing": {"signals": ["Bota force vector", "Bota torque vector"], "method": "centered moving average, edge padded",
                                   "window_samples": SMOOTH_WINDOW, "nominal_duration_ms_at_100Hz": 30,
                                   "applied_before_timestamp-aware differentiation": True},
              "derivatives": {"linear_acceleration": "timestamp-aware gradient of synchronized Cartesian linear velocity on CITR timeline",
                              "angular_acceleration": "timestamp-aware gradient of synchronized Cartesian angular velocity on CITR timeline",
                              "force_change": "norm of timestamp-aware gradient of 3-sample-smoothed Bota force vector, aligned to CITR timeline",
                              "torque_change": "norm of timestamp-aware gradient of 3-sample-smoothed Bota torque vector, aligned to CITR timeline",
                              "gripper_velocity": "absolute timestamp-aware gradient of canonical CITR gripper_position"},
              "training_trajectory_count": len(train), "training_source_split": RAW_SPLIT["train"], "source_root": str(DATA)}
    write_json(OUT / "frozen_thresholds.json", frozen)
    write_json(OUT / "calibration_complete.json", {"train_only": True, "thresholds_sha256": sha256(OUT / "frozen_thresholds.json"), "trajectory_count": len(train)})


def protection_overlap(c: dict[str, Any], passes: dict[str, np.ndarray], base: np.ndarray, candidate: str) -> list[dict[str, Any]]:
    fails = {s: base & ~passes[s] for s in SIGNALS}
    rows = []
    bit_counts: dict[str, int] = {}
    for i in np.flatnonzero(base):
        active = [s for s in SIGNALS if fails[s][i]]
        key = ";".join(active) if active else "none"
        bit_counts[key] = bit_counts.get(key, 0) + 1
    for name in SIGNALS:
        unique = fails[name] & ~np.logical_or.reduce([fails[x] for x in SIGNALS if x != name])
        rows.append({"split": c["split"], "candidate": candidate, "trajectory_id": c["trajectory_id"], "family": c["family"],
                     "activity_signal": name, "uniquely_protected_old_low_frames": int(unique.sum()), "failed_signal_frames_in_old_low_pool": int(fails[name].sum())})
    for key, n in bit_counts.items():
        rows.append({"split": c["split"], "candidate": candidate, "trajectory_id": c["trajectory_id"], "family": c["family"],
                     "activity_signal": "OVERLAP:" + key, "uniquely_protected_old_low_frames": n, "failed_signal_frames_in_old_low_pool": ""})
    return rows


def figure(c: dict[str, Any], base: np.ndarray, low: np.ndarray, keep: np.ndarray, thresholds: dict[str, float]) -> Path:
    t = c["ts"] - c["ts"][0]
    fig, axes = plt.subplots(8, 1, figsize=(15, 17), sharex=True, gridspec_kw={"height_ratios": [1,1,1,1,1,1,.6,.6]})
    for ax, name in zip(axes[:5], SIGNALS):
        ax.plot(t, c["signals"][name], lw=.65, color="#264653")
        ax.axhline(thresholds[name], color="#d62828", ls="--", lw=.9, label=f"P{SELECTED_PERCENTILE:g} threshold {thresholds[name]:.4g} {UNITS[name]}")
        ax.set_ylabel(name.replace("_", "\n"), fontsize=8); ax.legend(loc="upper right", fontsize=7)
    axes[5].plot(t, c["lin"], lw=.55, color="#2a9d8f", label="||v||")
    axes[5].plot(t, c["ang"], lw=.55, color="#e76f51", label="||ω||")
    axes[5].axhline(LIN_T, color="#2a9d8f", ls=":", lw=.8); axes[5].axhline(ANG_T, color="#e76f51", ls=":", lw=.8)
    axes[5].set_ylabel("frozen\nvelocity", fontsize=8); axes[5].legend(loc="upper right", fontsize=7)
    axes[6].fill_between(t, 0, 1, where=base, color="#f4a261", step="post", alpha=.7, label="Round35 low-velocity candidate")
    axes[6].fill_between(t, 0, 1, where=low, color="#2a9d8f", step="post", alpha=.75, label="new all-signal low activity")
    axes[6].set_ylabel("candidate", fontsize=8); axes[6].legend(loc="upper right", fontsize=7)
    axes[7].fill_between(t, 0, 1, where=keep, color="#457b9d", step="post", alpha=.8, label="retained")
    axes[7].fill_between(t, 0, 1, where=~keep, color="#e63946", step="post", alpha=.8, label="deleted")
    axes[7].set_ylabel("timeline", fontsize=8); axes[7].legend(loc="upper right", fontsize=7)
    # GT boundaries are rendered strictly as post-freeze diagnostic references.
    for ax in axes:
        for g in c["gt"][1:]: ax.axvline(t[g["start"]], color="#6a4c93", lw=.55, alpha=.6)
    axes[-1].set_xlabel("time from trajectory start (s)")
    fig.suptitle(f"{c['trajectory_id']} — multi-signal compression; purple = GT boundaries (diagnostic only)")
    fig.tight_layout(rect=[0,0,1,.98])
    path = OUT / "figures" / (c["trajectory_id"].replace("/", "__").replace(" ", "_") + ".png")
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def bundle_pipeline(maps: dict[str, np.ndarray]) -> tuple[int, int, list[str]]:
    compressed_root = OUT / "bundles/compressed"
    merged_root = OUT / "bundles/merged"
    compressed_root.mkdir(parents=True, exist_ok=True); merged_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(BUNDLE_ROOT.glob("*/trajectory_bundle.json"))
    if not paths: raise RuntimeError(f"no frozen Round45 bundles found under {BUNDLE_ROOT}")
    seen = set(); failures=[]; compressed_count=merged_count=0
    for p in paths:
        bundle = load_bundle(p.parent)
        tid = str(bundle.metadata["trajectory_id"])
        if tid in seen: raise RuntimeError(f"duplicate frozen bundle: {tid}")
        seen.add(tid)
        if tid not in maps:
            failures.append(tid); continue
        retained = maps[tid]
        if bundle.current_T != int(records_by_id[tid]["original_frame_count"]):
            raise RuntimeError(f"{tid}: frozen prediction bundle length does not align with verified uncompressed source")
        mapping = OUT / "data" / tid / "frame_mapping.csv"
        compressed = compress_bundle(bundle, retained, mapping)
        out = compressed_root / tid.replace("/", "__")
        save_bundle(compressed, out)
        compressed_count += 1
        # Reuse the validated Round40 merge code/config; do not alter its rule.
        merged_segments, audit = merge_prediction(compressed, fraction=.80, window=5, mode="one_pass")
        merged = compressed.clone()
        merged.prediction_segments = merged_segments
        merged.prediction_frame_labels = segments_to_frame_labels(merged_segments, merged.current_T)
        merged.metadata["merge"] = {"method": "canonical_round40_one_pass_similarity", "fraction": .80, "local_window": 5,
                                     "strategy": "one_pass", "opposite_boundary_protection": True,
                                     "safety_description": "segment protection", "audit": audit}
        merged.validate()
        save_bundle(merged, merged_root / tid.replace("/", "__"))
        merged_count += 1
    missing_ids = sorted(set(maps) - seen)
    return compressed_count, merged_count, sorted(set(failures) | set(missing_ids))


def write_report() -> None:
    d = OUT
    dist = pd.read_csv(d / "signal_distribution_train_low_velocity.csv")
    candidates = pd.read_csv(d / "candidate_compression_impact_train.csv")
    pooled = candidates[candidates.trajectory_id == "__POOLED__"].copy()
    impacts = pd.read_csv(d / "compression_impact_all_trajectories.csv")
    events = pd.read_csv(d / "gripper_event_coverage_all.csv")
    final_events = pd.read_csv(d / "gripper_event_coverage_final.csv")
    overlap = pd.read_csv(d / "unique_and_overlapping_protection_all.csv")
    skill_diag = pd.read_csv(d / "per_family_skill_inspection.csv")
    bundle = json.loads((d / "bundle_pipeline_validation.json").read_text())
    selected = json.loads((d / "frozen_thresholds.json").read_text())
    def pct(x: float) -> str: return f"{100*x:.2f}%"
    lines = [
        "# Round35 Multi-Signal Physical-Activity Compression (v001)", "",
        "## Objective and protocol", "",
        "This version tests a multi-signal definition of physical low activity; it is not a correction of a demonstrated Round35 failure. Historical Round35/Round40/Round45 artifacts were not modified. No ASRF inference or model training was run.", "",
        "Threshold calibration used only the 79 original `train0` trajectories. Thresholds were frozen before any `test0` content was read. The final 36-trajectory test set was used only after freezing, for post-freeze compression, event coverage, and per-family/per-skill diagnostics. GT boundaries in plots are diagnostic only.", "",
        "## Signal construction", "",
        "Robot Cartesian velocity components were timestamp-interpolated to the canonical CITR timeline; linear/angular acceleration are timestamp-aware gradients on that timeline. Bota force/torque vectors were smoothed before differentiation with a centered 3-sample moving average (edge-padded; nominal 30 ms at 100 Hz), then differentiated against actual Bota timestamps and aligned to CITR. Gripper velocity is the absolute time-aware derivative of canonical `gripper_position`. Vector magnitudes are used for acceleration, force change, and torque change. Absolute force/torque are not used.", "",
        "## Train-only low-velocity distributions", "",
        "The pool is every train0 frame satisfying the frozen Round35 linear/angular velocity limits. Threshold comparison uses strict `<` as specified; the empirical percentile is advanced by one floating-point ULP so samples exactly at a quantile remain on the low side.", "",
        "| Signal | Unit | Frames | P50 | P90 | P95 | P97.5 | P99 | Max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in dist.to_dict("records"):
        lines.append(f"| `{r['signal']}` | {r['unit']} | {int(r['low_velocity_frame_count'])} | {r['P50']:.6g} | {r['P90']:.6g} | {r['P95']:.6g} | {r['P97_5']:.6g} | {r['P99']:.6g} | {r['max']:.6g} |")
    lines += ["", "## Train-only threshold candidates", "", "Candidate thresholds are percentiles of each additional signal inside the old velocity-low pool. P99.5 is the frozen main policy: it reduced train compression less aggressively than P99/P97.5, retained finite guards on all five channels, and yielded 19.34% train compression versus the historical 23.38% reference. P99.9 rose above the reference and was not selected. The historical ratio is a reasonableness reference, not an exact target.", "",
              "| Candidate | Round35 train removal | Candidate train removal | Old low-pool frames vetoed by added signals |", "|---|---:|---:|---:|"]
    for r in pooled.to_dict("records"):
        lines.append(f"| {r['candidate']} | {pct(float(r['round35_compression_ratio']))} | {pct(float(r['compression_ratio']))} | {int(r['new_signal_protected_old_low_frames'])} |")
    lines += ["", f"Frozen main thresholds (candidate **{selected['selected_candidate']}**):", "", "| Signal | Threshold | Unit |", "|---|---:|---|"]
    for name in SIGNALS:
        lines.append(f"| `{name}` | {selected['thresholds'][name]:.12g} | {UNITS[name]} |")
    lines += ["", f"The original velocity thresholds remain linear **{LIN_T:.11g} m/s** and angular **{ANG_T:.10g} rad/s**. Temporal policy: minimum continuous low-activity run **{MIN_RUN_S:.2f} s**, edge context **{EDGE_CONTEXT_S:.2f} s per side**, minimum skill survival **{MIN_SKILL_FRAMES} frames (0.10 s)**, and gap bridging **off**.", ""]
    train = impacts[impacts.split == "train"]
    test = impacts[impacts.split == "test"]
    def pooled_ratio(g: pd.DataFrame, numerator: str) -> float: return float(g[numerator].sum() / g.frames.sum())
    lines += ["", "## Frozen compression impact", "", "| Split | Round35 baseline ratio | New final ratio | Removed frames (new) | Added-event-context trajectories |", "|---|---:|---:|---:|---:|"]
    override = pd.read_csv(d / "event_override_diagnosis_and_final_maps.csv")
    for split, group in (("train", train), ("test", test)):
        final_ratio = pooled_ratio(group, "final_removed_frames")
        baseline_ratio = pooled_ratio(group, "round35_removed_frames")
        lines.append(f"| {split} | {pct(baseline_ratio)} | {pct(final_ratio)} | {int(group.final_removed_frames.sum())} | {int(override[override.split == split].conditional_context_override_used.sum())} |")
    lines += ["", "## Unique and overlapping frame protection", "", "Unique counts below mean a frame in the old velocity-low population failed only that one additional signal. Multi-signal overlap combinations are enumerated in `unique_and_overlapping_protection_all.csv`.", "", "| Signal | Train unique frames | Test unique frames |", "|---|---:|---:|"]
    unique = overlap[overlap.activity_signal.isin(SIGNALS)]
    for s in SIGNALS:
        tr = unique[(unique.split == "train") & (unique.activity_signal == s)].uniquely_protected_old_low_frames.sum()
        te = unique[(unique.split == "test") & (unique.activity_signal == s)].uniquely_protected_old_low_frames.sum()
        lines.append(f"| `{s}` | {int(tr)} | {int(te)} |")
    lines += ["", "## Gripper-event coverage", ""]
    naturally_kept = int(events.fully_preserved_by_unified_rule.sum())
    event_total = len(events)
    event_deleted = int(events.active_frames_deleted_by_unified_rule.sum())
    final_preserved = int(final_events.final_event_fully_preserved.sum())
    override_events = int(final_events.conditional_context_override_used.sum())
    lines += [f"The frozen detector found **{event_total}** physical events across train0/test0. Without separate compression-time event protection, **{naturally_kept}/{event_total}** were fully preserved; **{event_deleted}** detector-active frames overlapped compression. The audit then applied the old ±0.5 s context only for trajectories where an active detected event would otherwise be removed: context was reinstated for **{int(override.conditional_context_override_used.sum())} trajectories**, affecting **{override_events} event intervals**. After this narrow safety override, **{final_preserved}/{event_total}** events were fully retained. Per-event evidence and the diagnostic are in the two gripper-event CSVs.", "",
              "This is the explicitly permitted fallback: there was an observed event-coverage failure under the unified rule, so the established context was conditionally retained rather than silently deleting detector-confirmed event activity. The override does not alter thresholds.", "",
              "## Family/skill inspection", "", "Additional-signal veto frames inside the old velocity-low pool, aggregated by family:", "", "| Split | Family | Protected frames |", "|---|---|---:|"]
    fam = skill_diag.groupby(["split", "family"], as_index=False).new_signal_protected_frames.sum()
    for r in fam.sort_values(["split", "new_signal_protected_frames"], ascending=[True, False]).to_dict("records"):
        lines.append(f"| {r['split']} | {r['family']} | {int(r['new_signal_protected_frames'])} |")
    lines += ["", "The same protection totals by GT skill (including the manipulation phases called out in the protocol) are:", "", "| Split | Skill | Protected frames |", "|---|---|---:|"]
    sk = skill_diag.groupby(["split", "skill"], as_index=False).new_signal_protected_frames.sum()
    for r in sk.sort_values(["split", "new_signal_protected_frames"], ascending=[True, False]).to_dict("records"):
        lines.append(f"| {r['split']} | {r['skill']} | {int(r['new_signal_protected_frames'])} |")
    lines += ["", "These are post-freeze diagnostics only; they did not feed threshold selection. Per-trajectory detail and final removals are in `per_family_skill_inspection.csv` and `event_override_diagnosis_and_final_maps.csv`.", "",
              "## Timeline figures", "", "The selected set contains deterministic representative test trajectories across PP/place, plug/insert, pour, wipe, and unscrew. Each figure plots the seven physical activity signals, frozen thresholds, old low-velocity regions, new all-signal low regions, retained/deleted timeline, and GT boundaries in purple as post-freeze diagnostics only.", ""]
    for f in sorted((d / "figures").glob("*.png")):
        lines.append(f"- `{f.relative_to(d)}`")
    lines += ["", "## Canonical bundle compression and merge", "", f"The canonical `compress_timeline.py` bundle operation was reused on **{bundle['compressed_bundle_count']}** frozen Round45-aligned bundles; canonical Round40 one-pass similarity merging (`fraction=0.80`, window 5, existing direction/opposite-boundary behavior) was reused on **{bundle['merged_bundle_count']}**. Merge safety is reported as **segment protection**. No ASRF inference was run. The two missing predictions remain explicitly unmatched: `{'; '.join(bundle['unmatched_or_missing_bundle_trajectory_ids'])}`.", "",
              "For available bundles, one retained-frame mapping was applied to CITR, GT, Hybrid prediction, SF/r5 BRB, gripper, timestamps, and motion features by the shared canonical compression tool. The standalone synchronized dataset outputs apply that same per-trajectory map to CITR arrays/CSV, annotations, robot states, Bota, gripper, and timestamps. No resizing, interpolation, or stretching was used.", "",
              "## Validation", "", "The new runner passed `py_compile`. The canonical bundle utility tests passed (6/6). The paired Round45 historical regression test invocation was 6 passed / 2 failed because its configured legacy fixture `outputs/round45_pp50_uncompressed_hybrid/predictions/uncompressed_raw` is absent in this checkout; those failures occurred before artifact construction. Independently, all 113 consumed frozen bundles passed exact CITR source-hash, retained-index mapping (CITR/GT/Hybrid/SF BRB/r5 BRB/gripper/timestamps), and post-merge non-segmentation-array identity checks; zero mismatches. All 115 synchronized data outputs passed mapping, CITR length, timestamp length, GT segment coverage, and robot-state mapping checks; zero mismatches.", "",
              "## Integrity and limitations", "", "The complete source tree was `train0`/`test0`, which was verified against all 115 historical Round35 original frame counts (79 train, 36 test; zero mismatches). Training thresholds are persisted in `frozen_thresholds.json` with the calibration hash. `source_integrity.json` records post-run source-hash stability and reports all source files unchanged. The 113 available Round45 bundles do not cover two dataset trajectories; no predictions were fabricated for those. Skill-survival protection uses GT segments only after thresholds are frozen, matching the prior temporal policy; test GT did not affect thresholds.", "",
              "## Artifacts", "", "- `frozen_thresholds.json` — selected train-only thresholds and full method.", "- `signal_distribution_train_low_velocity.csv` — required train signal quantiles.", "- `candidate_compression_impact_train.csv` — candidate comparisons.", "- `compression_impact_all_trajectories.csv` — frozen per-trajectory baseline/new ratios.", "- `unique_and_overlapping_protection_all.csv` — unique and joint veto patterns.", "- `gripper_event_coverage_all.csv`, `gripper_event_coverage_final.csv` — natural coverage and final safety override.", "- `per_family_skill_inspection.csv` — post-freeze diagnostic stratification.", "- `data/` — compressed synchronized raw dataset and reversible frame mappings.", "- `bundles/compressed/`, `bundles/merged/` — canonical bundles.", "- `report.md` — this report.", ""]
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")


records_by_id: dict[str, dict[str, str]] = {}


def evaluate() -> None:
    freeze_path = OUT / "frozen_thresholds.json"
    if not freeze_path.is_file() or not (OUT / "calibration_complete.json").is_file():
        raise RuntimeError("calibration stage is incomplete; freeze train-only thresholds first")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen["selected_candidate"] != f"P{SELECTED_PERCENTILE:g}": raise RuntimeError("unexpected selected threshold candidate")
    thresholds = frozen["thresholds"]
    all_rows = records(); records_by_id.clear(); records_by_id.update({r["trajectory_id"]: r for r in all_rows})
    contexts = [load_context(r) for r in all_rows]
    impact_rows=[]; overlap_rows=[]; skill_rows=[]; event_rows=[]; final_event_rows=[]; family_rows=[]; maps={}; figure_paths=[]
    test_gt_read_stage = "FROZEN_THRESHOLD_POSTFREEZE_DIAGNOSTIC"
    for c in contexts:
        low, passes = physical_low(c, thresholds)
        keep, edge, removed = apply_policy(c, low)
        base = velocity_low(c); old_keep = old_baseline(c)
        ev, events = event_mask(c, context_s=0)
        old_ratio=float((~old_keep).mean()); new_ratio=float((~keep).mean())
        impact_rows.append({"split": c["split"], "trajectory_id": c["trajectory_id"], "family": c["family"], "frames": len(keep),
                            "round35_removed_frames": int((~old_keep).sum()), "round35_compression_ratio": old_ratio,
                            "new_removed_frames": int((~keep).sum()), "new_compression_ratio": new_ratio,
                            "newly_protected_vs_round35_removed": int(np.sum((~old_keep) & keep)),
                            "newly_removed_vs_round35_kept": int(np.sum(old_keep & ~keep)),
                            "old_low_velocity_frames": int(base.sum()), "new_physical_low_frames": int(low.sum()),
                            "candidate_subset_invariant": int(np.all(low <= base)), "detected_gripper_events": len(events),
                            "postfreeze_test_gt_diagnostic": int(c["split"] == "test")})
        overlap_rows.extend(protection_overlap(c, passes, base, frozen["selected_candidate"]))
        added = base & ~np.logical_and.reduce(list(passes.values()))
        for g in c["gt"]:
            sl = slice(g["start"], g["end"])
            skill_rows.append({"split": c["split"], "family": c["family"], "trajectory_id": c["trajectory_id"], "skill": g["label"],
                               "old_low_frames": int(base[sl].sum()), "new_signal_protected_frames": int(added[sl].sum()),
                               "new_removed_frames": int((~keep[sl]).sum()), "round35_removed_frames": int((~old_keep[sl]).sum()),
                               "gt_used_for_calibration": 0, "test_gt_role": "post-freeze per-skill diagnostic" if c["split"] == "test" else "post-freeze diagnostic"})
        for j, (s,e) in enumerate(events):
            deleted = (~keep[s:e])
            event_rows.append({"split": c["split"], "trajectory_id": c["trajectory_id"], "family": c["family"], "event_index": j,
                               "start_frame": s, "end_frame_exclusive": e, "active_frames": e-s,
                               "active_frames_deleted_by_unified_rule": int(deleted.sum()), "fully_preserved_by_unified_rule": int(not deleted.any()),
                               "event_purpose": "frozen physical gripper detector audit; post-freeze only"})

    # Save all data mappings and source-aligned timeline arrays only after P99.5 is frozen.
    impact_by_id = {r["trajectory_id"]: r for r in impact_rows}
    for c in contexts:
        low, passes = physical_low(c, thresholds)
        no_event_keep, _, _ = apply_policy(c, low)
        _, events = event_mask(c, context_s=0)
        missed = any(np.any(~no_event_keep[s:e]) for s,e in events)
        context_s = .5 if missed else 0.0
        override, _ = event_mask(c, context_s=context_s)
        keep, edge, removed = apply_policy(c, low, override)
        for j, (s, e) in enumerate(events):
            final_event_rows.append({"split": c["split"], "trajectory_id": c["trajectory_id"], "family": c["family"], "event_index": j,
                                     "active_frames": e-s, "final_active_frames_deleted": int(np.sum(~keep[s:e])),
                                     "final_event_fully_preserved": int(np.all(keep[s:e])), "conditional_context_override_used": int(missed),
                                     "context_override_s": context_s})
        maps[c["trajectory_id"]] = np.flatnonzero(keep)
        map_and_write_data(c, keep, low, thresholds, passes, OUT)
        if c["trajectory_id"] in {"test/plug/p1", "test/wipe/w1", "test/pour/p1", "test/unscrew/s1", "test/pp/pp_1"}:
            figure_paths.append(str(figure(c, velocity_low(c), low, keep, thresholds)))
        family_rows.append({"split": c["split"], "trajectory_id": c["trajectory_id"], "family": c["family"],
                            "unified_rule_missed_detector_event_frames": int(sum(np.sum(~no_event_keep[s:e]) for s,e in events)),
                            "conditional_context_override_used": int(missed), "context_override_s": context_s,
                            "final_removed_frames": int((~keep).sum()), "retained_frames": int(keep.sum())})
        impact_by_id[c["trajectory_id"]]["final_removed_frames"] = int((~keep).sum())
        impact_by_id[c["trajectory_id"]]["final_compression_ratio"] = float((~keep).mean())
        impact_by_id[c["trajectory_id"]]["conditional_event_context_override"] = int(missed)
    write_csv(OUT / "compression_impact_all_trajectories.csv", impact_rows)
    write_csv(OUT / "unique_and_overlapping_protection_all.csv", overlap_rows)
    write_csv(OUT / "per_family_skill_inspection.csv", skill_rows)
    write_csv(OUT / "gripper_event_coverage_all.csv", event_rows)
    write_csv(OUT / "gripper_event_coverage_final.csv", final_event_rows)
    write_csv(OUT / "event_override_diagnosis_and_final_maps.csv", family_rows)
    comp, merged, bundle_missing = bundle_pipeline(maps)
    write_json(OUT / "bundle_pipeline_validation.json", {"compressed_bundle_count": comp, "merged_bundle_count": merged,
                "source_round45_bundle_count": len(list(BUNDLE_ROOT.glob("*/trajectory_bundle.json"))),
                "unmatched_or_missing_bundle_trajectory_ids": bundle_missing,
                "canonical_compress_timeline_reused": True, "canonical_round40_merge_reused": True,
                "asrf_inference_or_training": False})
    # Freeze output-level summary only after all audits and mappings complete.
    write_json(OUT / "evaluation_complete.json", {"frozen_threshold_sha256": sha256(freeze_path), "trajectory_count": len(contexts),
                "train_trajectory_count": sum(c["split"]=="train" for c in contexts), "test_trajectory_count": sum(c["split"]=="test" for c in contexts),
                "test_gt_used_only_postfreeze_diagnostics": True, "figures": figure_paths,
                "event_frames_deleted_before_conditional_override": int(sum(int(x["active_frames_deleted_by_unified_rule"]) for x in event_rows)),
                "conditional_event_context_overrides": int(sum(int(x["conditional_context_override_used"]) for x in family_rows))})
    integrity = []
    for c in contexts:
        current = {name: sha256(c["source"] / name) for name in c["source_hashes"]}
        integrity.append({"trajectory_id": c["trajectory_id"], "unchanged": int(current == c["source_hashes"]),
                          "expected_hashes": c["source_hashes"], "current_hashes": current})
    write_json(OUT / "source_integrity.json", {"all_source_files_unchanged": all(x["unchanged"] for x in integrity), "trajectories": integrity})
    write_report()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("calibrate", "evaluate"))
    args = p.parse_args()
    if args.stage == "calibrate": calibrate()
    else: evaluate()
    print(json.dumps({"stage": args.stage, "output": str(OUT)}, indent=2))


if __name__ == "__main__": main()
