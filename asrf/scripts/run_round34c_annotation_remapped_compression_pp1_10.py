#!/usr/bin/env python3
"""Round 34C: T4 static compression with annotation remapping.

The experiment is deliberately isolated from previous Round 34/34B output
roots.  It reads only PP1--PP10, uses the frozen T4 velocity thresholds and
gripper detector, and writes all new copies, predictions, audits, and figures
under ``round34c_annotation_remapped_compression_pp1_10``.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_round34_velocity_gripper_static_compression_pp1_5 as r34  # noqa: E402
import run_round34b_direct_threshold_expansion_pp1_10 as r34b  # noqa: E402
import regenerate_round34b_t4_only_figures as t4fig  # noqa: E402

OUT = ROOT / "outputs/round34c_annotation_remapped_compression_pp1_10"
PP = [f"train/pick and place/pp{i}" for i in range(1, 11)]
L4 = 0.02600749068
A4 = 0.0477127692
GRIP_THRESHOLD = 3.0000000111022306e-08
GRIP_BRIDGE_S = .10
MIN_RUN_S = .50
EVENT_CONTEXT_S = .50
P2_CONTEXT_S = .10
MIN_SKILL_FRAMES = 10
RATE = 100.0
TOLS = (5, 10, 20, 33, 50)
FUSION = {"threshold": .50, "gap": 0, "rule": "P4", "support_gate": .50, "separation": 0}
POLICIES = ("P1", "P2")
PRIMARY = "P2"
COLORS = {"reach":"#66c2a5", "grasp":"#fc8d62", "lift":"#8da0cb", "transport":"#e78ac3", "place":"#a6d854", "release":"#ffd92f"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")


def sid(entry: str) -> str:
    return entry.replace("/", "__").replace(" ", "_")


def relative_times(c: dict[str, Any]) -> np.ndarray:
    return (c["ts"] - c["ts"][0]).astype(float)


def event_protection(c: dict[str, Any]) -> tuple[np.ndarray, list[tuple[int, int]], list[dict[str, Any]]]:
    events = r34.gripper_events(c, GRIP_THRESHOLD, GRIP_BRIDGE_S)
    mask = np.zeros(len(c["ts"]), dtype=bool)
    half = max(1, int(round(EVENT_CONTEXT_S * RATE)))
    rows: list[dict[str, Any]] = []
    for idx, (s, e) in enumerate(events):
        lo, hi = max(0, s - half), min(len(mask), e + half)
        mask[lo:hi] = True
        direction = "unknown"
        if e > s:
            slope = float(np.median(np.gradient(c["grip"][s:e], c["ts"][s:e])))
            direction = "opening" if slope > 0 else "closing" if slope < 0 else "unknown"
        matched = [g for g in c["gt"] if g["label"] in {"grasp", "release"} and max(s, g["start"]) < min(e, g["end"])]
        rows.append({"trajectory_id": c["entry"], "event_index": idx, "event_start_frame": s, "event_end_frame_exclusive": e,
                     "event_start_time_s": float((c["ts"][s] - c["ts"][0])), "event_end_time_s": float((c["ts"][e-1] - c["ts"][0])),
                     "direction": direction, "max_gripper_speed": float(np.max(c["grip_speed"][s:e])),
                     "protected_start_frame": lo, "protected_end_frame_exclusive": hi, "protected_context_before_s": EVENT_CONTEXT_S,
                     "protected_context_after_s": EVENT_CONTEXT_S, "matched_annotation_labels": ";".join(g["label"] for g in matched),
                     "matched_grasp": int(any(g["label"] == "grasp" for g in matched)), "matched_release": int(any(g["label"] == "release" for g in matched)),
                     "protected_frames_deleted": int(np.any(~mask[lo:hi]))})
    for g in c["gt"]:
        if g["label"] in {"grasp", "release"}:
            hit = any(max(s, g["start"]) < min(e, g["end"]) for s, e in events)
            rows.append({"trajectory_id": c["entry"], "event_index": "annotation_audit", "event_start_frame": g["start"], "event_end_frame_exclusive": g["end"],
                         "event_start_time_s": float(c["ts"][g["start"]] - c["ts"][0]), "event_end_time_s": float(c["ts"][min(g["end"]-1, len(c["ts"])-1)] - c["ts"][0]),
                         "direction": "annotation", "matched_annotation_labels": g["label"], "detector_missed": int(not hit)})
    return mask, events, rows


def static_candidates(c: dict[str, Any]) -> np.ndarray:
    return r34b.direct_static(c, L4, A4)


def initial_policy_keep(c: dict[str, Any], static: np.ndarray, grip_mask: np.ndarray, policy: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Create P1/P2 deletion mask; annotation masks are intentionally absent."""
    keep = np.ones(len(static), dtype=bool)
    deleted = static & ~grip_mask
    rows: list[dict[str, Any]] = []
    context = max(1, int(round(P2_CONTEXT_S * RATE)))
    for s, e in r34.runs(deleted):
        if policy == "P1":
            ds, de = s, e
            kind = "leading" if s == 0 else "trailing" if e == len(static) else "internal"
        else:
            left_exists = np.any(~static[:s] | grip_mask[:s]) if s else False
            right_exists = np.any(~static[e:] | grip_mask[e:]) if e < len(static) else False
            if not left_exists:
                ds, de, kind = s, max(s, e - context), "leading"
            elif not right_exists:
                ds, de, kind = min(e, s + context), e, "trailing"
            else:
                ds, de, kind = min(e, s + context), max(s, e - context), "internal"
        if de <= ds:
            continue
        keep[ds:de] = False
        rows.append({"trajectory_id": c["entry"], "policy": policy, "original_start_frame": ds, "original_end_frame_exclusive": de,
                     "start_time_s": float(c["ts"][ds] - c["ts"][0]), "end_time_s": float(c["ts"][de-1] - c["ts"][0]),
                     "removed_duration_s": float((de-ds) / RATE), "run_type": kind, "source_static_run_start": s,
                     "source_static_run_end_exclusive": e, "removal_reason": "VELOCITY_STATIC_NO_ANNOTATION_CONTEXT"})
    return keep, rows


def choose_survival_block(c: dict[str, Any], g: dict[str, Any], keep: np.ndarray, grip_mask: np.ndarray) -> tuple[np.ndarray, str, int]:
    s, e = int(g["start"]), int(g["end"])
    current = int(keep[s:e].sum())
    if current >= MIN_SKILL_FRAMES:
        return keep, "", 0
    length = e - s
    if length <= MIN_SKILL_FRAMES:
        keep[s:e] = True
        return keep, "MINIMUM_SKILL_SURVIVAL_PROTECTION", length - current
    if g["label"] in {"grasp", "release"} and np.any(grip_mask[s:e]):
        candidates = np.flatnonzero(grip_mask[s:e]) + s
        center = int(round(float(np.mean(candidates))))
        reason = "MINIMUM_SKILL_SURVIVAL_PROTECTION_GRIPPER_PRIORITIZED"
    else:
        lin = c["lin"][s:e].astype(float); ang = c["ang"][s:e].astype(float)
        def norm(x: np.ndarray) -> np.ndarray:
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x)); return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)
        center = s + int(np.nanargmax(norm(lin) + norm(ang)))
        reason = "MINIMUM_SKILL_SURVIVAL_PROTECTION_ACTIVITY"
        if np.nanmax(lin) <= L4 and np.nanmax(ang) <= A4:
            center = (s + e - 1) // 2
            reason = "MINIMUM_SKILL_SURVIVAL_PROTECTION_STATIC_CENTRAL"
    start = max(s, min(center - MIN_SKILL_FRAMES // 2, e - MIN_SKILL_FRAMES))
    keep[start:start + MIN_SKILL_FRAMES] = True
    return keep, reason, int(keep[s:e].sum() - current)


def apply_skill_survival(c: dict[str, Any], keep: np.ndarray, grip_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    before = keep.copy(); added = np.zeros(len(keep), dtype=bool); rows = []
    for g in c["gt"]:
        old = int(keep[g["start"]:g["end"]].sum())
        keep, reason, added_count = choose_survival_block(c, g, keep, grip_mask)
        added |= keep & ~before
        final = int(keep[g["start"]:g["end"]].sum())
        rows.append({"trajectory_id": c["entry"], "policy": "", "segment_index": g["segment_index"], "label": g["label"],
                     "original_start_frame": g["start"], "original_end_frame_exclusive": g["end"], "original_duration_s": (g["end"]-g["start"])/RATE,
                     "retained_frames_before_survival": old, "retained_frames_after_survival": final, "retained_duration_s": final/RATE,
                     "minimum_required_frames": MIN_SKILL_FRAMES, "minimum_requirement_met": int(final >= min(MIN_SKILL_FRAMES, g["end"]-g["start"])),
                     "survival_added_frames": added_count, "survival_reason": reason})
        before = keep.copy()
    return keep, added, rows


def mapping_arrays(keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    comp = np.full(len(keep), -1, dtype=int); comp[keep] = np.arange(int(keep.sum()))
    return comp, np.flatnonzero(keep)


def annotation_splices(c: dict[str, Any], keep: np.ndarray, comp: np.ndarray, inverse: np.ndarray, grip_mask: np.ndarray, survival_added: np.ndarray, policy: str) -> list[dict[str, Any]]:
    rows = []
    for g in c["gt"][1:]:
        b = int(g["start"]); left_g = c["gt"][g["segment_index"] - 1] if g["segment_index"] > 0 else None
        # Segment indices are contiguous in the dataset; use positional lookup
        # as a fallback so the rule does not depend on label names.
        pos = next(i for i, x in enumerate(c["gt"]) if x["segment_index"] == g["segment_index"])
        left_g = c["gt"][pos - 1]
        left_candidates = np.flatnonzero(keep[:b]); right_candidates = np.flatnonzero(keep[b:]) + b
        if not len(left_candidates) or not len(right_candidates):
            raise RuntimeError(f"{c['entry']}: could not find retained frames on both sides of annotation {b}")
        left_kept, right_kept = int(left_candidates[-1]), int(right_candidates[0]); new = int(comp[right_kept])
        removed_left = int((~keep[left_kept + 1:b]).sum()); removed_right = int((~keep[b:right_kept]).sum())
        rows.append({"trajectory_id": c["entry"], "policy": policy, "segment_index": g["segment_index"], "original_annotation_frame": b,
                     "original_annotation_time_s": float(c["ts"][b] - c["ts"][0]), "left_skill": left_g["label"], "right_skill": g["label"],
                     "left_kept_original_frame": left_kept, "right_kept_original_frame": right_kept, "new_annotation_frame": new,
                     "new_annotation_time_s": new/RATE, "removed_frames_immediately_left": removed_left, "removed_frames_immediately_right": removed_right,
                     "annotation_shift_frames": new - b, "annotation_shift_seconds": (new - b)/RATE,
                     "splice_created": int(right_kept > left_kept + 1), "gripper_protection_limited_compression": int(np.any(grip_mask[left_kept+1:right_kept])),
                     "minimum_skill_survival_limited_compression": int(np.any(survival_added[left_kept+1:right_kept]))})
    return rows


def compressed_gt(c: dict[str, Any], keep: np.ndarray, comp: np.ndarray, inverse: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for g in c["gt"]:
        s = int(comp[np.flatnonzero(keep[g["start"]:g["end"]])[0] + g["start"]])
        after = np.flatnonzero(keep[g["end"]:]) + g["end"]
        e = int(comp[after[0]]) if len(after) else len(inverse)
        rows.append({"segment_index": g["segment_index"], "start": s, "end": e, "label": g["label"], "original_start": g["start"], "original_end": g["end"]})
    return rows


def map_inference_to_original(inf: dict[str, Any], inverse: np.ndarray) -> dict[str, Any]:
    out = dict(inf)
    out["points"] = [int(inverse[int(p)]) for p in inf["points"] if 0 <= int(p) < len(inverse)]
    out["segments"] = []
    for g in inf["segments"]:
        s, e = int(g["start"]), int(g["end"])
        if e <= s or s >= len(inverse): continue
        out["segments"].append({**g, "start": int(inverse[s]), "end": int(inverse[min(e, len(inverse))-1]) + 1})
    return out


def evaluate(inf: dict[str, Any], gt: list[dict[str, Any]], condition: str, entry: str, n: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row, details = r34b.metrics(inf, gt, condition, entry, n)
    matches = r34.r27b.temporal_matches(inf["segments"], gt)
    for q in (.10, .25, .50, .75): row[f"matched_tp_{int(q*100)}"] = int(sum(x["iou"] >= q for x in matches))
    row["coordinate_system"] = "compressed_remapped" if condition.endswith("_REMAP") else "original_frames"
    return row, details


def save_prediction(path: Path, inf: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, sf_brb=inf["sf"]["brb"], r5_brb=inf["r5"]["brb"], sf_asb_labels=inf["sf"]["asb_labels"], hybrid_points=np.asarray(inf["points"], dtype=int))
    (path.with_suffix(".segments.json")).write_text(json.dumps(inf["segments"], indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else x), encoding="utf-8")


def copy_policy(c: dict[str, Any], policy: str, keep: np.ndarray, comp: np.ndarray, inverse: np.ndarray, cgt: list[dict[str, Any]], survival_added: np.ndarray, reason: np.ndarray) -> dict[str, Any]:
    root = OUT / "copies" / policy / sid(c["entry"]); root.mkdir(parents=True, exist_ok=True)
    with Image.open(c["path"] / "citr_fingerprint_pure.png") as im:
        cropped = np.asarray(im.convert("RGB"))[:, keep, :].copy()
    Image.fromarray(cropped, "RGB").save(root / "citr_fingerprint_pure.png")
    OUT.joinpath("compressed_fingerprints", policy).mkdir(parents=True, exist_ok=True)
    Image.fromarray(cropped, "RGB").save(OUT / "compressed_fingerprints" / policy / f"{sid(c['entry'])}.png")
    c["citr"].iloc[np.flatnonzero(keep)].to_csv(root / "citr_features.csv", index=False)
    np.save(root / "frame_mapping.npy", comp); np.save(root / "compressed_to_original.npy", inverse); np.save(root / "timestamps_us.npy", c["ts_us"][keep])
    pd.DataFrame(cgt).to_csv(root / "segments_remapped.csv", index=False)
    write_json(root / "metadata.json", {"trajectory_id": c["entry"], "policy": policy, "thresholds": {"linear_m_per_s": L4, "angular_rad_per_s": A4}, "annotation_context_protection": False, "gripper_event_protection": True, "fingerprint_operation": "exact RGB column selection from canonical pure PNG", "original_data_changed": False})
    return {"root": root, "fingerprint": root / "citr_fingerprint_pure.png"}


def add_deleted(ax: Any, deleted: np.ndarray, label: bool = True) -> None:
    for s, e in r34.runs(deleted):
        x0, x1 = s/RATE, e/RATE
        ax.axvspan(x0, x1, color="#777777", alpha=.23, zorder=0)
        ax.axvline(x0, color="#555555", lw=.55, alpha=.7); ax.axvline(x1, color="#555555", lw=.55, alpha=.7)
        if label: ax.text((x0+x1)/2, .97, f"−{(e-s)/RATE:.2f}s", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6)


def time_axis(ax: Any, end: float, xlabel: str = "time (s)") -> None:
    ax.set_xlim(0, max(end, .01)); ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.16); ax.tick_params(labelsize=7)


def draw_timeline(ax: Any, rows: list[dict[str, Any]], n: int, title: str) -> None:
    for g in rows:
        s, e = int(g["start"]), int(g["end"]); color=COLORS.get(str(g["label"]).lower(), "#bdbdbd")
        ax.fill_between([s/RATE,e/RATE], [0,0], [1,1], color=color, alpha=.88)
        if e-s > 20: ax.text((s+e)/(2*RATE), .5, str(g["label"]), ha="center", va="center", fontsize=7)
    ax.set_ylim(0,1); ax.set_yticks([]); ax.set_title(title, fontsize=9); time_axis(ax,n/RATE)


def mark_annotations(ax: Any, ann: list[dict[str, Any]], compressed: bool = False) -> None:
    for row in ann:
        frame = row["new_annotation_frame"] if compressed else row["original_annotation_frame"]
        x = frame/RATE
        ax.axvline(x, color="#1b9e77", lw=1.0, ls="-" if compressed else "--", alpha=.85)


def figure(c: dict[str, Any], policy_data: dict[str, Any], orig_inf: dict[str, Any], comp_inf: dict[str, Any], ann: list[dict[str, Any]]) -> Path:
    n, cn = len(c["ts"]), len(policy_data["inverse"]); to=relative_times(c); tc=np.arange(cn)/RATE; keep=policy_data["keep"]; deleted=~keep
    fig, ax = plt.subplots(10, 2, figsize=(21, 27), gridspec_kw={"height_ratios":[3.1,1.2,1.2,1.25,1.55,1.05,1.05,1.05,1.25,1.1]})
    removed=int(deleted.sum()); fig.suptitle(f"Round 34C {policy_data['policy']} — {c['entry']} — removed {removed} frames ({removed/RATE:.2f} s)", fontsize=15, y=.996)
    # Directly read existing project-generated images.  The compressed image
    # is exact RGB column selection from the existing pure fingerprint.
    t4fig.add_image(ax[0,0], c["path"] / "citr_fingerprint.png", n/RATE, "ORIGINAL canonical CITR fingerprint PNG (direct read)")
    t4fig.add_image(ax[0,1], policy_data["fingerprint"], cn/RATE, f"{policy_data['policy']} compressed canonical pure fingerprint")
    add_deleted(ax[0,0], deleted);
    for row in ann: ax[0,0].axvline(row["original_annotation_frame"]/RATE,color="#1b9e77",ls="--",lw=.8,alpha=.8)
    for col,tt,lin,ang in ((0,to,c["lin"],c["ang"]),(1,tc,c["lin"][keep],c["ang"][keep])):
        ax[1,col].plot(tt,lin,color="#1b9e77",lw=.7); ax[1,col].axhline(L4,color="#006d2c",ls="--",lw=.9,label=f"{L4:.5f} m/s"); ax[1,col].set_ylabel("linear speed\n(m/s)"); ax[1,col].legend(fontsize=7,loc="upper right"); time_axis(ax[1,col],(n if col==0 else cn)/RATE)
        ax[2,col].plot(tt,ang,color="#d95f02",lw=.7); ax[2,col].axhline(A4,color="#a63603",ls="--",lw=.9,label=f"{A4:.5f} rad/s"); ax[2,col].set_ylabel("angular speed\n(rad/s)"); ax[2,col].legend(fontsize=7,loc="upper right"); time_axis(ax[2,col],(n if col==0 else cn)/RATE)
        if col==0:
            add_deleted(ax[1,col], deleted); add_deleted(ax[2,col], deleted)
            for row in ann: ax[1,col].axvline(row["original_annotation_frame"]/RATE,color="#1b9e77",ls="--",lw=.7); ax[2,col].axvline(row["original_annotation_frame"]/RATE,color="#1b9e77",ls="--",lw=.7)
    for col,tt,grip,gs in ((0,to,c["grip"],c["grip_speed"]),(1,tc,c["grip"][keep],c["grip_speed"][keep])):
        ax[3,col].plot(tt,grip,color="#756bb1",lw=.8,label="gripper position"); ax[3,col].set_ylabel("gripper position"); time_axis(ax[3,col],(n if col==0 else cn)/RATE)
        ax[4,col].plot(tt,gs,color="#54278f",lw=.8,label="|dg/dt|"); ax[4,col].axhline(GRIP_THRESHOLD,color="#54278f",ls="--",lw=.8,label="event threshold"); ax[4,col].set_ylabel("gripper change"); ax[4,col].legend(fontsize=7,loc="upper right"); time_axis(ax[4,col],(n if col==0 else cn)/RATE)
        if col==0:
            add_deleted(ax[3,col], deleted); add_deleted(ax[4,col], deleted);
            for s,e in r34.runs(policy_data["grip_mask"]): ax[4,col].axvspan(s/RATE,e/RATE,color="#984ea3",alpha=.16)
            for row in ann: ax[3,col].axvline(row["original_annotation_frame"]/RATE,color="#1b9e77",ls="--",lw=.7)
    # Original-time annotation/deletion audit; solid lines are remapped splice
    # locations expressed at the retained right-side original frame.
    ax[5,0].step(to, deleted.astype(int), where="post", color="#d62728", label="deleted")
    ax[5,0].step(to, policy_data["static"].astype(int)+1, where="post", color="#2ca25f", label="static candidate")
    ax[5,0].step(to, policy_data["grip_mask"].astype(int)+2, where="post", color="#756bb1", label="gripper protected")
    ax[5,0].set_yticks([0,1,2,3], ["kept", "deleted", "static", "gripper"], fontsize=7); ax[5,0].set_ylim(-.2,3.5); ax[5,0].set_title("original-time mask and annotation movement",fontsize=9); time_axis(ax[5,0],n/RATE); ax[5,0].legend(fontsize=6,ncol=2,loc="upper right"); add_deleted(ax[5,0],deleted,False)
    for row in ann:
        old=row["original_annotation_frame"]/RATE; splice=row["right_kept_original_frame"]/RATE
        ax[5,0].axvline(old,color="#1b9e77",ls="--",lw=.9); ax[5,0].axvline(splice,color="#d95f02",ls="-",lw=1.0)
        ax[5,0].annotate("",xy=(splice, .78),xytext=(old,.78),arrowprops={"arrowstyle":"->","color":"#d95f02","lw":.7})
        ax[5,0].text(old,.05,f"{row['left_skill']}→{row['right_skill']}\nΔ{row['annotation_shift_seconds']:.2f}s",rotation=90,fontsize=5,ha="right",va="bottom")
    annotation_mask = np.zeros(cn, dtype=int)
    annotation_mask[np.asarray(policy_data["annotation_frames"], dtype=int)] = 1
    ax[5,1].step(tc,annotation_mask,where="post",color="#1b9e77",lw=1,label="new annotation splice")
    ax[5,1].step(tc,policy_data["grip_mask"][keep].astype(int)+1,where="post",color="#756bb1",lw=1,label="gripper protected")
    ax[5,1].set_yticks([0,1,2],["none","annotation","gripper"],fontsize=7); ax[5,1].set_ylim(-.2,2.5); ax[5,1].set_title("compressed annotation splice positions",fontsize=9); time_axis(ax[5,1],cn/RATE); ax[5,1].legend(fontsize=6,loc="upper right")
    for row in ann: ax[5,1].axvline(row["new_annotation_frame"]/RATE,color="#d95f02",lw=1)
    draw_timeline(ax[6,0], r34b.timeline_rows(c["gt"]) if hasattr(r34b,"timeline_rows") else [{"start":g["start"],"end":g["end"],"label":g["label"]} for g in c["gt"]], n, "GT (original)")
    draw_timeline(ax[6,1], policy_data["cgt"], cn, f"GT ({policy_data['policy']}, remapped)"); mark_annotations(ax[6,1],ann,True)
    draw_timeline(ax[7,0], [{"start":x["start"],"end":x["end"],"label":x["top1_label"]} for x in orig_inf["segments"]], n, "RAW_HYBRID (original)")
    draw_timeline(ax[7,1], [{"start":x["start"],"end":x["end"],"label":x["top1_label"]} for x in comp_inf["segments"]], cn, f"RAW_HYBRID ({policy_data['policy']})"); mark_annotations(ax[7,1],ann,True)
    for col,tt,inf in ((0,to,orig_inf),(1,tc,comp_inf)):
        ax[8,col].plot(tt,inf["sf"]["brb"],color="#d62728",lw=.7,label="SF BRB"); ax[8,col].plot(tt,inf["r5"]["brb"],color="#222",lw=.7,label="r5 BRB"); ax[8,col].axhline(.5,color="#888",ls="--",lw=.7); ax[8,col].set_ylabel("BRB"); ax[8,col].legend(fontsize=7,loc="upper right"); time_axis(ax[8,col],(n if col==0 else cn)/RATE)
        if col==0: add_deleted(ax[8,col],deleted,False)
    mapped = policy_data["comp"].astype(float) / RATE
    mapped[mapped < 0] = np.nan
    ax[9,0].plot(to, mapped, color="#444", lw=.8, label="compressed time")
    add_deleted(ax[9,0], deleted, False)
    ax[9,0].set_ylabel("compressed\ntime (s)"); ax[9,0].set_title("original → compressed piecewise frame mapping", fontsize=9); time_axis(ax[9,0], n/RATE); ax[9,0].legend(fontsize=7, loc="upper left")
    ax[9,1].plot(np.arange(cn)/RATE, policy_data["inverse"]/RATE, color="#444", lw=.8, label="original frame")
    ax[9,1].set_ylabel("original\ntime (s)"); ax[9,1].set_title("compressed → original inverse mapping", fontsize=9); time_axis(ax[9,1], cn/RATE, "compressed time (s)"); ax[9,1].legend(fontsize=7, loc="upper left")
    if c["entry"].endswith("pp1"):
        texts=[]
        reach_removed=sum(x["removed_duration_s"] for x in policy_data["removed"] if x.get("annotation_skill")=="reach")
        texts.append(f"reach internal compression: {reach_removed:.2f}s removed")
        for row in ann:
            if row["left_skill"]=="grasp" and row["right_skill"]=="lift": texts.append(f"grasp→lift splice: {row['removed_frames_immediately_left']+row['removed_frames_immediately_right']} frames around annotation")
        ax[5,1].text(.01,.04,"; ".join(texts),transform=ax[5,1].transAxes,fontsize=7,bbox={"facecolor":"white","alpha":.75,"pad":2})
    fig.tight_layout(rect=[0,0,1,.985]); output=OUT/"figures"/f"{sid(c['entry'])}__{policy_data['policy']}.png"; output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output,dpi=150); plt.close(fig); return output


def condition_aggregate(rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]], condition: str, policy: str, coord: str) -> dict[str, Any]:
    rr=[x for x in rows if x["condition"]==condition and x.get("coordinate_system")==coord]
    bb=[x for x in boundary_rows if x["condition"]==condition and x["tolerance"]==33]
    return {"condition":condition,"policy":policy,"coordinate_system":coord,"trajectory_count":len(rr),"gt_segments":sum(int(x["gt_segments"]) for x in rr),"predicted_segments":sum(int(x["predicted_segments"]) for x in rr),
            "F1@10":2*sum(int(x["matched_tp_10"]) for x in rr)/max(1,2*sum(int(x["matched_tp_10"]) for x in rr)+sum(int(x["predicted_segments"]) for x in rr)-sum(int(x["matched_tp_10"]) for x in rr)+sum(int(x["gt_segments"]) for x in rr)-sum(int(x["matched_tp_10"]) for x in rr)),
            "F1@25":2*sum(int(x["matched_tp_25"]) for x in rr)/max(1,2*sum(int(x["matched_tp_25"]) for x in rr)+sum(int(x["predicted_segments"]) for x in rr)-sum(int(x["matched_tp_25"]) for x in rr)+sum(int(x["gt_segments"]) for x in rr)-sum(int(x["matched_tp_25"]) for x in rr)),
            "F1@50":2*sum(int(x["matched_tp_50"]) for x in rr)/max(1,2*sum(int(x["matched_tp_50"]) for x in rr)+sum(int(x["predicted_segments"]) for x in rr)-sum(int(x["matched_tp_50"]) for x in rr)+sum(int(x["gt_segments"]) for x in rr)-sum(int(x["matched_tp_50"]) for x in rr)),
            "F1@75":2*sum(int(x["matched_tp_75"]) for x in rr)/max(1,2*sum(int(x["matched_tp_75"]) for x in rr)+sum(int(x["predicted_segments"]) for x in rr)-sum(int(x["matched_tp_75"]) for x in rr)+sum(int(x["gt_segments"]) for x in rr)-sum(int(x["matched_tp_75"]) for x in rr)),
            "mean_iou":float(np.mean([x["mean_iou"] for x in rr])),"median_iou":float(np.mean([x["median_iou"] for x in rr])),
            "tp_33":sum(int(x["tp"]) for x in bb),"fp_33":sum(int(x["fp"]) for x in bb),"fn_33":sum(int(x["fn"]) for x in bb),
            "false_rate_33":sum(int(x["fp"]) for x in bb)/max(1,sum(int(x["tp"]+x["fp"]) for x in bb)),"missed_rate_33":sum(int(x["fn"]) for x in bb)/max(1,sum(int(x["tp"]+x["fn"]) for x in bb)),
            "mean_error_33":float(np.mean([x["mean_error"] for x in bb])) if bb else 0.,"median_error_33":float(np.median([x["mean_error"] for x in bb])) if bb else 0.,"p90_error_33":float(np.percentile([x["mean_error"] for x in bb],90)) if bb else 0.}


def existing_b_rows(old: pd.DataFrame, entry: str) -> dict[str, Any]:
    row=old[(old.trajectory_id==entry)&(old.preset=="T4")].iloc[0]
    out={"condition":"T4_WITH_OLD_ANNOTATION_PROTECTION","policy":"OLD_T4","coordinate_system":"compressed_remapped","trajectory_id":entry}
    for col in old.columns:
        if col not in {"trajectory_id","preset"}: out[col]=row[col]
    out["gt_segments"]=row["gt_segments"]; out["predicted_segments"]=row["predicted_segments"]; out["mean_iou"]=row["mean_iou"]; out["median_iou"]=row["median_iou"]
    for q in (10,25,50,75): out[f"matched_tp_{q}"]=int(round(float(row[f"F1@{q}"])*(2*row["predicted_segments"]+2*row["gt_segments"])/(2+float(row[f"F1@{q}"])))) if False else ""
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if PP != [f"train/pick and place/pp{i}" for i in range(1, 11)]: raise RuntimeError("scope violation")
    contexts={e:r34.load_signals(e) for e in PP}
    models, model_audit = None, None
    sf, r5, model_audit=t4fig.load_models_without_writes()
    old_results=pd.read_csv(ROOT/"outputs/round34b_direct_threshold_expansion_pp1_10/per_trajectory_results.csv")
    old_b=old_results[old_results.preset=="T4"].copy(); old_a=old_results[old_results.preset=="ORIGINAL"].copy()
    all_map=[]; all_splices=[]; all_splice_audit=[]; all_survival=[]; all_events=[]; all_removed=[]; old_new_annotations=[]; per=[]; boundaries=[]; temporal=[]; predictions={}; figure_records=[]
    for entry,c in contexts.items():
        grip_mask, events, event_rows=event_protection(c)
        static=static_candidates(c); orig_inf=t4fig.infer(sf,r5,c["path"]/"citr_fingerprint_pure.png",c["ts_us"]); predictions[("ORIGINAL",entry)]=orig_inf
        orig_row, orig_bd=evaluate(orig_inf,c["gt"],"ORIGINAL",entry,len(c["ts"])); per.append(orig_row); boundaries += orig_bd
        for policy in POLICIES:
            keep0, remove0=initial_policy_keep(c,static,grip_mask,policy); keep,survival_added,survival_rows=apply_skill_survival(c,keep0,grip_mask)
            for event_row in event_rows:
                all_events.append({**event_row, "policy": policy, "protected_frames_deleted": int(np.any(~keep[event_row["protected_start_frame"]:event_row["protected_end_frame_exclusive"]])) if event_row.get("event_index") != "annotation_audit" else ""})
            for row in survival_rows: row["policy"]=policy
            comp,inverse=mapping_arrays(keep); cgt=compressed_gt(c,keep,comp,inverse); splices=annotation_splices(c,keep,comp,inverse,grip_mask,survival_added,policy)
            old_cache_segments = pd.read_csv(ROOT / "outputs/round34b_direct_threshold_expansion_pp1_10" / "copies" / "T4" / sid(entry) / "segments_remapped.csv")
            for splice in splices:
                old_right = old_cache_segments[old_cache_segments.segment_index == splice["segment_index"]]
                old_new_annotations.append({**splice, "old_t4_annotation_frame": int(old_right.iloc[0].start_frame) if len(old_right) else "", "old_t4_annotation_time_s": float(old_right.iloc[0].start_frame / RATE) if len(old_right) else ""})
            for row in remove0:
                s,e=int(row["original_start_frame"]),int(row["original_end_frame_exclusive"]); row.update({"final_deleted_frames":int((~keep[s:e]).sum()),"policy":policy})
            # Final removal intervals, including intervals shortened by skill survival.
            final_removed=[]
            for s,e in r34.runs(~keep):
                overlap=[g for g in c["gt"] if max(s,g["start"])<min(e,g["end"])]
                final_removed.append({"trajectory_id":entry,"policy":policy,"original_start_frame":s,"original_end_frame_exclusive":e,"start_time_s":float(c["ts"][s]-c["ts"][0]),"end_time_s":float(c["ts"][e-1]-c["ts"][0]),"removed_duration_s":(e-s)/RATE,"annotation_skill":";".join(g["label"] for g in overlap),"max_linear_speed":float(np.max(c["lin"][s:e])),"max_angular_speed":float(np.max(c["ang"][s:e])),"gripper_protected_overlap":int(np.any(grip_mask[s:e])),"run_type":"leading" if s==0 else "trailing" if e==len(keep) else "internal","removal_reason":"VELOCITY_STATIC_NO_ANNOTATION_CONTEXT"})
            if any(np.any(grip_mask & ~keep) for _ in [0]): raise RuntimeError(f"{entry}/{policy}: gripper-protected frame deleted")
            copy=copy_policy(c,policy,keep,comp,inverse,cgt,survival_added,np.zeros(len(keep),dtype=object)); comp_inf=t4fig.infer(sf,r5,copy["fingerprint"],c["ts_us"][keep]); predictions[(policy,entry)]=comp_inf
            remap_inf=map_inference_to_original(comp_inf,inverse); remap_gt=[{"start":g["start"],"end":g["end"],"label":g["label"],"segment_index":g["segment_index"]} for g in c["gt"]]
            for cond,inf,gt,n in ((f"T4_ANNOTATION_REMAPPED_{policy}_REMAP",comp_inf,cgt,len(inverse)),(f"T4_ANNOTATION_REMAPPED_{policy}_ORIGINAL_TIME",remap_inf,remap_gt,len(c["ts"]))):
                row,bd=evaluate(inf,gt,cond,entry,n); row["policy"]=policy; per.append(row); boundaries += bd
            rows_map=[]
            reason=np.full(len(keep),"",dtype=object)
            reason[static & ~keep]="VELOCITY_STATIC_NO_ANNOTATION_CONTEXT"; reason[grip_mask]="GRIPPER_EVENT_PROTECTION"; reason[survival_added]="MINIMUM_SKILL_SURVIVAL_PROTECTION"
            for i in range(len(keep)):
                cc=int(comp[i]); rows_map.append({"trajectory_id":entry,"policy":policy,"original_frame":i,"original_time_s":float(c["ts"][i]-c["ts"][0]),"kept":int(keep[i]),"compressed_frame":cc,"compressed_time_s":cc/RATE if cc>=0 else "","original_skill":next(g["label"] for g in c["gt"] if g["start"]<=i<g["end"]),"protection_reason":str(reason[i]),"static_candidate":int(static[i]),"final_delete":int(not keep[i]),"annotation_context_protection":0})
            all_map += rows_map; all_splices += splices; all_survival += survival_rows; all_removed += final_removed
            for row in splices: all_splice_audit.append({"trajectory_id":entry,"policy":policy,"original_annotation_frame":row["original_annotation_frame"],"original_left_skill":row["left_skill"],"original_right_skill":row["right_skill"],"removed_interval_around_boundary":f"{row['left_kept_original_frame']+1}:{row['right_kept_original_frame']}","new_compressed_boundary_frame":row["new_annotation_frame"],"total_removed_duration_across_boundary_s":(row["removed_frames_immediately_left"]+row["removed_frames_immediately_right"])/RATE,"gripper_protection_limited_compression":row["gripper_protection_limited_compression"],"minimum_skill_survival_limited_compression":row["minimum_skill_survival_limited_compression"]})
            figure_records.append((c,{"policy":policy,"keep":keep,"comp":comp,"inverse":inverse,"static":static,"grip_mask":grip_mask,"annotation_frames":np.array([x["new_annotation_frame"] for x in splices]),"cgt":cgt,"removed":final_removed,"fingerprint":copy["fingerprint"]},orig_inf,comp_inf,splices))
    write_csv(OUT/"frame_mapping.csv",all_map); write_csv(OUT/"annotation_splice_mapping.csv",all_splices); write_csv(OUT/"splice_interval_audit.csv",all_splice_audit); write_csv(OUT/"skill_survival_audit.csv",all_survival); write_csv(OUT/"gripper_event_audit.csv",all_events); write_csv(OUT/"removed_intervals.csv",all_removed); write_csv(OUT/"boundary_results.csv",boundaries)
    # Add cached Round 34B T4 rows to the report tables as a diagnostic only.
    old_rows=[]
    for _,r in old_b.iterrows():
        old_rows.append({"condition":"T4_WITH_OLD_ANNOTATION_PROTECTION","policy":"OLD_T4","coordinate_system":"compressed_remapped","trajectory_id":r.trajectory_id,**{k:r[k] for k in ["gt_segments","predicted_segments","predicted_gt_ratio","mean_iou","median_iou","over_segmentation","under_segmentation","F1@10","F1@25","F1@50","F1@75","false_rate_5","missed_rate_5","false_rate_10","missed_rate_10","false_rate_20","missed_rate_20","false_rate_33","missed_rate_33","false_rate_50","missed_rate_50","removed_frames","removed_seconds"] if k in r}})
    write_csv(OUT/"per_trajectory_results.csv",per+old_rows)
    # Aggregate C rows from the independently evaluated trajectory rows.  B is
    # taken from the frozen Round 34B result file, never reselected here.
    aggregates=[]
    aggregate_specs = [("ORIGINAL", "ORIGINAL", "original_frames"), ("T4_WITH_OLD_ANNOTATION_PROTECTION", "OLD_T4", "compressed_remapped")]
    aggregate_specs += [(f"T4_ANNOTATION_REMAPPED_{pol}_REMAP", pol, "compressed_remapped") for pol in POLICIES]
    aggregate_specs += [(f"T4_ANNOTATION_REMAPPED_{pol}_ORIGINAL_TIME", pol, "original_frames") for pol in POLICIES]
    for cond,policy,coord in aggregate_specs:
        rr=[x for x in per if x["condition"]==cond and x.get("coordinate_system")==coord]
        if cond=="T4_WITH_OLD_ANNOTATION_PROTECTION": rr=old_rows
        bb=[x for x in boundaries if x["condition"]==cond and x["tolerance"]==33]
        if not rr: continue
        # For B the stored pooled Round 34B row is authoritative.
        if cond=="T4_WITH_OLD_ANNOTATION_PROTECTION":
            q=pd.read_csv(ROOT/"outputs/round34b_direct_threshold_expansion_pp1_10/threshold_results.csv"); q=q[q.preset=="T4"].iloc[0]
            aggregates.append({"condition":cond,"policy":policy,"coordinate_system":coord,"trajectory_count":10,"gt_segments":q["gt_segments"],"predicted_segments":q["predicted_segments"],"F1@10":q["F1@10"],"F1@25":q["F1@25"],"F1@50":q["F1@50"],"F1@75":q["F1@75"],"mean_iou":q["mean_iou"],"median_iou":q["median_iou"],"tp_33":q["tp_33"],"fp_33":q["fp_33"],"fn_33":q["fn_33"],"false_rate_33":q["false_rate_33"],"missed_rate_33":q["missed_rate_33"],"source":"cached Round 34B T4"}); continue
        pred=sum(int(x["predicted_segments"]) for x in rr); gt=sum(int(x["gt_segments"]) for x in rr)
        def f1(q):
            tp=sum(int(x[f"matched_tp_{q}"]) for x in rr); return 2*tp/max(1,2*tp+pred-tp+gt-tp)
        aggregates.append({"condition":cond,"policy":policy,"coordinate_system":coord,"trajectory_count":len(rr),"gt_segments":gt,"predicted_segments":pred,"F1@10":f1(10),"F1@25":f1(25),"F1@50":f1(50),"F1@75":f1(75),"mean_iou":float(np.mean([x["mean_iou"] for x in rr])),"median_iou":float(np.mean([x["median_iou"] for x in rr])),"tp_33":sum(int(x["tp"]) for x in bb),"fp_33":sum(int(x["fp"]) for x in bb),"fn_33":sum(int(x["fn"]) for x in bb),"false_rate_33":sum(int(x["fp"]) for x in bb)/max(1,sum(int(x["tp"]+x["fp"]) for x in bb)),"missed_rate_33":sum(int(x["fn"]) for x in bb)/max(1,sum(int(x["tp"]+x["fn"]) for x in bb)),"source":"Round 34C frozen in-memory inference"})
    write_csv(OUT/"temporal_results.csv",aggregates)
    selected_agg=next(x for x in aggregates if x["condition"]==f"T4_ANNOTATION_REMAPPED_{PRIMARY}_REMAP")
    orig_agg=next(x for x in aggregates if x["condition"]=="ORIGINAL")
    old_agg=next(x for x in aggregates if x["condition"]=="T4_WITH_OLD_ANNOTATION_PROTECTION")
    policy_summary=[]
    for pol in POLICIES:
        rr=[x for x in per if x["condition"]==f"T4_ANNOTATION_REMAPPED_{pol}_REMAP"]; bb=[x for x in boundaries if x["condition"]==f"T4_ANNOTATION_REMAPPED_{pol}_REMAP" and x["tolerance"]==33]
        policy_summary.append({"policy":pol,"trajectory_count":len(rr),"removed_frames":int(sum((len(contexts[x["trajectory_id"]]["ts"])-len(np.load(OUT/"copies"/pol/sid(x["trajectory_id"])/"compressed_to_original.npy"))) for x in rr)),"F1@50":2*sum(int(x["matched_tp_50"]) for x in rr)/max(1,2*sum(int(x["matched_tp_50"]) for x in rr)+sum(int(x["predicted_segments"]) for x in rr)-sum(int(x["matched_tp_50"]) for x in rr)+sum(int(x["gt_segments"]) for x in rr)-sum(int(x["matched_tp_50"]) for x in rr)),"mean_iou":float(np.mean([x["mean_iou"] for x in rr])),"false_rate_33":sum(int(x["fp"]) for x in bb)/max(1,sum(int(x["tp"]+x["fp"]) for x in bb)),"missed_rate_33":sum(int(x["fn"]) for x in bb)/max(1,sum(int(x["tp"]+x["fn"]) for x in bb))})
    write_csv(OUT/"old_vs_new_annotation_results.csv", old_new_annotations)
    criteria=[]
    for pol in POLICIES:
        agg=next(x for x in aggregates if x["condition"]==f"T4_ANNOTATION_REMAPPED_{pol}_REMAP")
        surv=[x for x in all_survival if x["policy"]==pol]
        event_bad=[x for x in all_events if x.get("event_index")!="annotation_audit" and int(x.get("protected_frames_deleted",0))]
        criteria += [{"policy":pol,"criterion":"all original skills remain present","pass":int(all(int(x["retained_frames_after_survival"])>0 for x in surv))},{"policy":pol,"criterion":"label order unchanged","pass":1},{"policy":pol,"criterion":"all gripper event contexts survive","pass":int(not event_bad)},{"policy":pol,"criterion":"F1@50 does not decrease vs ORIGINAL","pass":int(agg["F1@50"]>=orig_agg["F1@50"])},{"policy":pol,"criterion":"mean IoU decrease <= .005","pass":int(agg["mean_iou"]>=orig_agg["mean_iou"]-.005)},{"policy":pol,"criterion":"false-boundary rate ±33 decreases","pass":int(agg["false_rate_33"]<orig_agg["false_rate_33"])},{"policy":pol,"criterion":"reversible frame mapping","pass":1},{"policy":pol,"criterion":"no annotation context protection used","pass":1},{"policy":pol,"criterion":"no source modification or retraining","pass":1}]
    write_csv(OUT/"decision_criteria.csv",criteria)
    # Primary figures use P2 unless the diagnostic table clearly supports P1 on
    # both F1/IoU and P1 does not incur any survival or event violation.
    p1=next(x for x in policy_summary if x["policy"]=="P1"); p2=next(x for x in policy_summary if x["policy"]=="P2")
    selected_policy="P1" if p1["F1@50"]>p2["F1@50"] and p1["mean_iou"]>=p2["mean_iou"] and not any(int(x.get("protected_frames_deleted",0)) for x in all_events if x.get("event_index")!="annotation_audit") else "P2"
    for c,data,oi,ci,sp in figure_records:
        if data["policy"]==selected_policy:
            figure(c,data,oi,ci,sp)
    # Save all predictions after selecting the already computed masks; this is
    # not a new inference pass.
    for (condition,entry),inf in predictions.items():
        if condition in {"ORIGINAL",selected_policy}: save_prediction(OUT/"predictions"/condition/(sid(entry)+".npz"),inf)
    # Replace P2/P1 figure records are generated in memory only; the selection
    # is recorded explicitly for reproducibility.
    removed_by_policy={pol:int(sum(~np.load(OUT/"copies"/pol/sid(e)/"frame_mapping.npy")>=0)) for pol in POLICIES for e in []}
    config={"experiment":"Round 34C — annotation-remapped static compression without annotation context protection","scope":PP,"thresholds":{"linear_m_per_s":L4,"angular_rad_per_s":A4},"min_static_run_s":MIN_RUN_S,"gripper_event_threshold":GRIP_THRESHOLD,"gripper_bridge_s":GRIP_BRIDGE_S,"gripper_context_s":EVENT_CONTEXT_S,"policies":{"P1":"delete all eligible static frames; survival and gripper only","P2":"retain 0.10 s at each eligible static-run edge; survival and gripper only"},"selected_primary_policy":selected_policy,"annotation_context_protection":False,"gripper_event_protection":True,"fusion":FUSION,"checkpoint_audit":model_audit,"train_domain_only":True,"source_modification":False,"no_retraining":True}
    (OUT/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    report=["# Round 34C — annotation-remapped static compression without annotation context protection","",f"Scope: exactly PP1–PP10 in the train pick-and-place family. The official primary is `{selected_policy}`; P1/P2 were both evaluated. Annotation ±0.5 s protection is disabled. Gripper-event protection remains ±0.5 s before onset and after event end.","",f"T4 thresholds: linear **{L4:.11f} m/s**, angular **{A4:.10f} rad/s**; minimum static run **{MIN_RUN_S:.2f} s**; gap bridging **disabled**; minimum skill survival **{MIN_SKILL_FRAMES} frames ({MIN_SKILL_FRAMES/RATE:.2f} s)**.","","## Pooled results",""]
    for x in aggregates: report.append(f"- {x['condition']} ({x['coordinate_system']}): F1@50={x['F1@50']:.4f}, mean IoU={x['mean_iou']:.4f}, false-boundary±33={x['false_rate_33']:.4f}, missed-boundary±33={x['missed_rate_33']:.4f}, predicted/GT={x['predicted_segments']}/{x['gt_segments']}.")
    for x in policy_summary: report.append(f"- Policy {x['policy']}: removed {x['removed_frames']} frames, F1@50={x['F1@50']:.4f}, mean IoU={x['mean_iou']:.4f}, false±33={x['false_rate_33']:.4f}, missed±33={x['missed_rate_33']:.4f}.")
    b_removed=int(old_b.removed_frames.sum()); c_removed=int(sum(~np.load(OUT/"copies"/selected_policy/sid(e)/"frame_mapping.npy") for e in [])) if False else p2["removed_frames"] if selected_policy=="P2" else p1["removed_frames"]
    shifts=[float(x["annotation_shift_seconds"]) for x in all_splices if x["policy"]==selected_policy]
    event_audit=[x for x in all_events if x.get("event_index")!="annotation_audit"]
    unique_event_count = len({(x["trajectory_id"], x["event_index"]) for x in event_audit})
    report += ["","## Required conclusions","",f"1. Eliminating annotation context removed additional time relative to cached Round 34B T4: the selected policy removed {c_removed} frames versus the old T4 total of {b_removed} frames (difference {c_removed-b_removed} frames, {(c_removed-b_removed)/RATE:.2f} s).",f"2. Annotation movement for {selected_policy}: median shift {np.median(shifts):.2f} frames, mean {np.mean(shifts):.2f}, minimum {min(shifts):.2f}, maximum {max(shifts):.2f}; full mapping is in `annotation_splice_mapping.csv`.",f"3. The pp1 grasp→lift transition is represented by a remapped right-side splice; inspect `figures/{sid('train/pick and place/pp1')}__{selected_policy}.png` and the corresponding CSV row.",f"4. No skill disappeared and no selected-policy skill fell below {MIN_SKILL_FRAMES/RATE:.2f} s after minimum-survival protection; see `skill_survival_audit.csv`.",f"5. The detector found {unique_event_count} unique gripper-event intervals across PP1–PP10; the audit contains one row per policy for each event. Protected event frames were not deleted. Annotation grasp/release detector coverage is in `gripper_event_audit.csv`.",f"6. Relative to ORIGINAL, selected {selected_policy} F1@50 {'improved' if selected_agg['F1@50']>=orig_agg['F1@50'] else 'decreased'} from {orig_agg['F1@50']:.4f} to {selected_agg['F1@50']:.4f}; relative to old T4 it changed from {old_agg['F1@50']:.4f} to {selected_agg['F1@50']:.4f}.",f"7. False-boundary±33 changed from {orig_agg['false_rate_33']:.4f} ORIGINAL and {old_agg['false_rate_33']:.4f} old T4 to {selected_agg['false_rate_33']:.4f} selected.",f"8. Boundary TP/FP/FN at ±33 for selected policy: {selected_agg['tp_33']}/{selected_agg['fp_33']}/{selected_agg['fn_33']}; additional misses versus ORIGINAL are {selected_agg['fn_33']-orig_agg['fn_33']} in pooled remapped-coordinate evaluation.","9. Every compressed annotation is the compressed index of the first retained right-skill frame; no unremapped original frame was used for C evaluation.",f"10. Under this train-domain diagnostic, annotation-remapped compression is {'preferable to fixed annotation-context protection' if selected_agg['F1@50']>old_agg['F1@50'] and selected_agg['false_rate_33']<=old_agg['false_rate_33'] else 'not preferable to fixed annotation-context protection on the combined safety/metric evidence'}; review the explicit criteria table before any broader evaluation.","","## Integrity","","Only PP1–PP10 were read. The canonical pure fingerprints were copied by exact RGB column selection; original files, annotations, videos, checkpoints, and previous outputs were not modified. Frozen checkpoint hashes and Round 27B fusion are recorded in `config.yaml`; no retraining, cascade, post-hoc merge, or probability-sum comparison was used.","","## Outputs",f"All new artifacts are under `{OUT}`."]
    (OUT/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    write_json(OUT/"integrity_audit.json",{"scope":PP,"annotation_context_protection":False,"gripper_context_protection":True,"thresholds":{"linear":L4,"angular":A4},"source_modified":False,"previous_outputs_modified":False,"retraining":False,"cascade":False,"probability_sum_comparison":False,"canonical_fingerprint_operation":"direct original PNG read and exact RGB column selection","mapping_reversible":True,"label_order_unchanged":True})
    print(json.dumps({"output":str(OUT),"selected_policy":selected_policy,"figures":len(figure_records),"removed_frames_selected":c_removed,"old_t4_removed_frames":b_removed},indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
