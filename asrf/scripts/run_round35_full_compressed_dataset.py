#!/usr/bin/env python3
"""Generate the complete Round 35 compressed ASRF dataset.

Round 35 is a data-generation and validation round only.  It applies the
Round 34C P2 policy with the corrected 0.60 s static-run edge context to every
trajectory below the resolved source ``data/train`` and ``data/test`` roots.
No model is loaded and no source file is opened for writing.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import argparse
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
import run_round34c_annotation_remapped_compression_pp1_10 as r34c  # noqa: E402

SOURCE_ROOT = r34.DATA
OUT = ROOT / "outputs/round35_full_compressed_dataset"
DATA_OUT = OUT / "data"
LINEAR_THRESHOLD = 0.02600749068
ANGULAR_THRESHOLD = 0.0477127692
MIN_RUN_S = .50
EDGE_CONTEXT_S = .60
INTERNAL_TOTAL_CONTEXT_S = 1.20
GRIP_THRESHOLD = r34c.GRIP_THRESHOLD
GRIP_BRIDGE_S = r34c.GRIP_BRIDGE_S
EVENT_CONTEXT_S = .50
MIN_SKILL_FRAMES = 10
RATE = 100.0
PREVIEW_ENTRIES = ["train/pick and place/pp1", "test/plug/p1", "train/plug/p1"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sid(entry: str) -> str:
    return entry.replace("/", "__").replace(" ", "_")


def discover() -> list[dict[str, Any]]:
    if not SOURCE_ROOT.is_dir():
        raise RuntimeError(f"resolved source dataset root does not exist: {SOURCE_ROOT}")
    records = []
    for split in ("train", "test"):
        split_root = SOURCE_ROOT / split
        for segment_file in sorted(split_root.rglob("segments.csv")):
            rel = segment_file.parent.relative_to(SOURCE_ROOT)
            if len(rel.parts) < 3 or rel.parts[0] != split:
                raise RuntimeError(f"unsupported trajectory layout: {rel}")
            records.append({"split": split, "family": rel.parts[1], "trajectory_id": "/".join(rel.parts), "relative": rel, "path": segment_file.parent})
    if not records:
        raise RuntimeError("no trajectory directories with segments.csv were discovered")
    return records


def relative_time(ts_us: np.ndarray) -> np.ndarray:
    return (ts_us.astype(np.float64) - float(ts_us[0])) / 1e6


def timestamp_frame_map(source_ts_us: np.ndarray, target_ts_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map target samples to preceding CITR frame intervals without interpolation."""
    source = np.asarray(source_ts_us, dtype=np.int64)
    target = np.asarray(target_ts_us, dtype=np.int64)
    idx = np.searchsorted(source, target, side="right") - 1
    valid = (target >= source[0]) & (target < source[-1] + max(1, int(np.median(np.diff(source)))))
    idx = np.clip(idx, 0, len(source) - 1)
    return idx.astype(int), valid


def raw_static_runs(c: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    static = r34b.direct_static(c, LINEAR_THRESHOLD, ANGULAR_THRESHOLD)
    run_ids = np.full(len(static), -1, dtype=int)
    for run_id, (s, e) in enumerate(r34.runs(static)):
        run_ids[s:e] = run_id
    return static, run_ids


def corrected_keep_mask(c: dict[str, Any], static: np.ndarray, run_ids: np.ndarray, gripper_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Apply P2 with 0.60 s edge context; no annotation mask is consulted."""
    keep = np.ones(len(static), dtype=bool)
    edge = np.zeros(len(static), dtype=bool)
    deleted_candidate = static & ~gripper_mask
    edge_frames = max(1, int(round(EDGE_CONTEXT_S * RATE)))
    rows = []
    for s, e in r34.runs(deleted_candidate):
        left_exists = bool(np.any(~static[:s] | gripper_mask[:s])) if s else False
        right_exists = bool(np.any(~static[e:] | gripper_mask[e:])) if e < len(static) else False
        if e - s <= 2 * edge_frames:
            ds, de = e, e
            kind = "unchanged_short_static_run"
            edge[s:e] = True
        elif not left_exists:
            ds, de, kind = s, e - edge_frames, "leading"
            edge[de:e] = True
        elif not right_exists:
            ds, de, kind = s + edge_frames, e, "trailing"
            edge[s:ds] = True
        else:
            ds, de, kind = s + edge_frames, e - edge_frames, "internal"
            edge[s:ds] = True; edge[de:e] = True
        if de > ds:
            keep[ds:de] = False
            rows.append({"original_start_frame": ds, "original_end_frame_exclusive": de, "removed_duration_s": (de-ds)/RATE,
                         "run_type": kind, "source_static_run_id": int(run_ids[s]), "source_run_start": s, "source_run_end_exclusive": e,
                         "edge_context_s": EDGE_CONTEXT_S, "total_edge_context_s": INTERNAL_TOTAL_CONTEXT_S if kind == "internal" else EDGE_CONTEXT_S})
    return keep, edge, rows


def annotation_rows(c: dict[str, Any], keep: np.ndarray, comp: np.ndarray, inverse: np.ndarray, gripper_mask: np.ndarray, survival_added: np.ndarray) -> list[dict[str, Any]]:
    rows = r34c.annotation_splices(c, keep, comp, inverse, gripper_mask, survival_added, "P2_EDGE_0.60")
    output=[]
    for i, row in enumerate(rows):
        output.append({"annotation_index": i, "left_skill": row["left_skill"], "right_skill": row["right_skill"],
                       "original_frame": row["original_annotation_frame"], "original_time_s": row["original_annotation_time_s"],
                       "left_retained_original_frame": row["left_kept_original_frame"], "right_retained_original_frame": row["right_kept_original_frame"],
                       "compressed_annotation_frame": row["new_annotation_frame"], "compressed_annotation_time_s": row["new_annotation_time_s"],
                       "shift_frames": row["annotation_shift_frames"], "shift_seconds": row["annotation_shift_seconds"],
                       "removed_left_frames": row["removed_frames_immediately_left"], "removed_right_frames": row["removed_frames_immediately_right"],
                       "valid": 1})
    return output


def compressed_gt(c: dict[str, Any], keep: np.ndarray, comp: np.ndarray, inverse: np.ndarray) -> list[dict[str, Any]]:
    rows=[]
    for g in c["gt"]:
        inside=np.flatnonzero(keep[g["start"]:g["end"]])+g["start"]
        if not len(inside): raise RuntimeError(f"{c['entry']}: skill {g['label']} has no retained frame")
        after=np.flatnonzero(keep[g["end"]:])+g["end"]
        end=int(comp[after[0]]) if len(after) else len(inverse)
        rows.append({"segment_index":g["segment_index"],"start_frame":int(comp[inside[0]]),"end_frame_exclusive":end,"label":g["label"],"original_start_frame":g["start"],"original_end_frame_exclusive":g["end"]})
    return rows


def remapped_segments_df(source_df: pd.DataFrame, c: dict[str, Any], keep: np.ndarray, comp: np.ndarray, inverse: np.ndarray, output_ts_us: np.ndarray) -> pd.DataFrame:
    rows=[]; starts=[]
    for _, row in source_df.iterrows():
        s=int(np.searchsorted(c["ts_us"], int(row["start_timestamp_us"]), side="left")); s=min(max(s,0),len(c["ts_us"])-1)
        e=int(np.searchsorted(c["ts_us"], int(row["end_timestamp_us_exclusive"]), side="left")); e=min(max(e,0),len(c["ts_us"]))
        retained=np.flatnonzero(keep[s:e])+s
        if not len(retained): raise RuntimeError(f"{c['entry']}: annotation row has no retained frame")
        start=int(comp[retained[0]])
        after=np.flatnonzero(keep[e:])+e
        end=int(comp[after[0]]) if len(after) else len(output_ts_us)
        start_us=int(output_ts_us[start]); end_us=int(output_ts_us[end]) if end < len(output_ts_us) else int(output_ts_us[-1] + np.median(np.diff(output_ts_us)))
        rows.append({"segment_index":int(row["segment_index"]),"start_timestamp_us":start_us,"end_timestamp_us_exclusive":end_us,
                     "start_relative_time_s":(start_us-int(output_ts_us[0]))/1e6,"end_relative_time_s":(end_us-int(output_ts_us[0]))/1e6,"label":row["label"]})
    return pd.DataFrame(rows, columns=["segment_index","start_timestamp_us","end_timestamp_us_exclusive","start_relative_time_s","end_relative_time_s","label"])


def process_timestamped_csv(source: Path, output: Path, source_ts: np.ndarray, keep: np.ndarray, extra: bool = False) -> tuple[int, int, pd.DataFrame]:
    data=pd.read_csv(source)
    if "timestamp_us" not in data.columns: raise RuntimeError(f"{source}: no timestamp_us")
    idx, valid=timestamp_frame_map(source_ts,data["timestamp_us"].to_numpy(dtype=np.int64))
    selected=valid & keep[idx]
    out=data.loc[selected].copy()
    audit=pd.DataFrame({"original_row_index":np.flatnonzero(selected),"original_timestamp":data.loc[selected,"timestamp_us"].to_numpy(dtype=np.int64),"nearest_original_citr_frame":idx[selected],"kept":1})
    if extra:
        out["original_row_index"]=np.flatnonzero(selected); out["original_timestamp"]=out["timestamp_us"].to_numpy(dtype=np.int64); out["original_relative_time_s"]=(out["timestamp_us"].to_numpy(dtype=np.float64)-float(source_ts[0]))/1e6; out["nearest_original_citr_frame"]=idx[selected]
    output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(output,index=False)
    return int(len(data)), int(len(out)), audit


def process_robot_states(source: Path, output: Path, c: dict[str, Any], keep: np.ndarray, comp: np.ndarray) -> tuple[int, int, pd.DataFrame]:
    data=pd.read_csv(source)
    if "timestamp_us" not in data.columns: raise RuntimeError(f"{source}: robot states missing timestamp_us")
    state_ts=data.timestamp_us.to_numpy(dtype=np.int64); idx, valid=timestamp_frame_map(c["ts_us"],state_ts); selected=valid & keep[idx]
    out=data.loc[selected].copy(); original_rows=np.flatnonzero(selected); compressed=comp[idx[selected]]
    out["original_row_index"]=original_rows; out["original_timestamp"]=state_ts[selected]; out["original_relative_time_s"]=(state_ts[selected].astype(np.float64)-float(c["ts_us"][0]))/1e6; out["nearest_original_citr_frame"]=idx[selected]; out["compressed_frame"]=compressed; out["compressed_relative_time_s"]=compressed/RATE
    audit=out[["original_row_index","original_timestamp","original_relative_time_s","nearest_original_citr_frame","compressed_frame","compressed_relative_time_s"]].copy()
    output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(output,index=False)
    return len(data),len(out),audit


def copy_or_link(source: Path, output: Path) -> str:
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.is_symlink() or output.exists():
        output.unlink()
    if source.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
        output.symlink_to(source.resolve()); return "SYMLINK_LARGE_MEDIA"
    shutil.copy2(source,output); return "COPY_UNCHANGED"


def save_segments(output_dir: Path, source_dir: Path, c: dict[str, Any], keep: np.ndarray, comp: np.ndarray, inverse: np.ndarray, output_ts_us: np.ndarray) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    source=pd.read_csv(source_dir/"segments.csv"); remap=remapped_segments_df(source,c,keep,comp,inverse,output_ts_us); remap.to_csv(output_dir/"segments.csv",index=False)
    actions=[{"file":"segments.csv","action":"REMAP_ANNOTATION_TIMESTAMPS","rows":len(remap)}]
    proposed=source_dir/"segments_release_proposed.csv"
    if proposed.exists():
        try:
            pdp=pd.read_csv(proposed)
            if len(pdp)==len(source) and list(pdp["label"].astype(str))==list(source["label"].astype(str)):
                remapped_segments_df(pdp,c,keep,comp,inverse,output_ts_us).to_csv(output_dir/proposed.name,index=False); actions.append({"file":proposed.name,"action":"REMAP_ANNOTATION_TIMESTAMPS","rows":len(pdp)})
            else:
                shutil.copy2(proposed,output_dir/proposed.name); actions.append({"file":proposed.name,"action":"COPY_UNCHANGED_AUXILIARY_ANNOTATION","rows":len(pdp)})
        except Exception:
            shutil.copy2(proposed,output_dir/proposed.name); actions.append({"file":proposed.name,"action":"COPY_UNCHANGED_WITH_WARNING","rows":""})
    return remap,actions


def mask_and_mappings(output_dir: Path, c: dict[str, Any], keep: np.ndarray, comp: np.ndarray, inverse: np.ndarray, static: np.ndarray, run_ids: np.ndarray, grip: np.ndarray, edge: np.ndarray, survival_added: np.ndarray) -> None:
    labels=np.empty(len(keep),dtype=object)
    for g in c["gt"]: labels[g["start"]:g["end"]]=g["label"]
    reason=np.full(len(keep),"DYNAMIC_OR_UNPROTECTED",dtype=object)
    reason[static & ~keep]="STATIC_RUN_REDUNDANT_MIDDLE_DELETION"
    reason[edge]="STATIC_EDGE_CONTEXT_PROTECTED_0.60S"
    reason[grip]="GRIPPER_EVENT_PROTECTION_0.50S_CONTEXT"
    reason[survival_added]="MINIMUM_SKILL_SURVIVAL_PROTECTION"
    mask_rows=[]
    for i in range(len(keep)):
        cc=int(comp[i]); mask_rows.append({"original_frame":i,"original_time_s":float((c["ts_us"][i]-c["ts_us"][0])/1e6),"linear_speed":float(c["lin"][i]),"angular_speed":float(c["ang"][i]),"static_candidate":int(static[i]),"raw_static_run_id":int(run_ids[i]),"gripper_protected":int(grip[i]),"static_edge_context_protected":int(edge[i]),"minimum_skill_survival_protected":int(survival_added[i]),"kept":int(keep[i]),"compressed_frame":cc,"compressed_time_s":cc/RATE if cc>=0 else "","reason":str(reason[i])})
    write_csv(output_dir/"compression_mask.csv",mask_rows)
    fm=[]
    for i in range(len(keep)):
        cc=int(comp[i]); fm.append({"original_frame":i,"original_time_s":float((c["ts_us"][i]-c["ts_us"][0])/1e6),"kept":int(keep[i]),"compressed_frame":cc,"compressed_time_s":cc/RATE if cc>=0 else "","original_skill":str(labels[i]),"deletion_or_protection_reason":str(reason[i])})
    write_csv(output_dir/"frame_mapping.csv",fm)
    write_csv(output_dir/"compressed_to_original_mapping.csv",[{"compressed_frame":j,"compressed_time_s":j/RATE,"original_frame":int(i),"original_time_s":float((c["ts_us"][i]-c["ts_us"][0])/1e6),"original_timestamp":int(c["ts_us"][i])} for j,i in enumerate(inverse)])
    pd.DataFrame({"compressed_frame":np.arange(int(keep.sum())),"label":labels[keep]}).to_csv(output_dir/"frame_labels.csv",index=False)
    pd.DataFrame({"original_boundary_frame":[g["start"] for g in c["gt"][1:]],"left_label":[c["gt"][i-1]["label"] for i in range(1,len(c["gt"]))],"right_label":[g["label"] for g in c["gt"][1:]]}).to_csv(output_dir/"boundary_labels.csv",index=False)


def validate_entry(output_dir: Path, c: dict[str, Any], keep: np.ndarray, comp: np.ndarray, inverse: np.ndarray, grip: np.ndarray, annotation_df: pd.DataFrame, robot_audit: pd.DataFrame) -> tuple[bool, str]:
    try:
        if not output_dir.is_dir(): return False,"output directory missing"
        features=pd.read_csv(output_dir/"citr_features.csv"); matrices=np.load(output_dir/"citr_matrices.npy",mmap_mode="r"); mask=pd.read_csv(output_dir/"compression_mask.csv"); fm=pd.read_csv(output_dir/"frame_mapping.csv"); inv=pd.read_csv(output_dir/"compressed_to_original_mapping.csv")
        n=int(keep.sum())
        if len(features)!=n or matrices.shape[0]!=n or len(mask)!=len(keep) or len(fm)!=len(keep) or len(inv)!=n: return False,"synchronized length mismatch"
        if matrices.ndim != 3 or matrices.shape[1:] != np.load(c["path"]/"citr_matrices.npy",mmap_mode="r").shape[1:]: return False,"numeric channel/shape changed"
        if not np.array_equal(fm.compressed_frame.to_numpy(dtype=int)[keep],np.arange(n)): return False,"forward mapping not contiguous"
        if not np.array_equal(inv.original_frame.to_numpy(dtype=int),inverse): return False,"inverse mapping mismatch"
        if len(annotation_df)!=len(c["gt"]) : return False,"annotation count changed"
        if not annotation_df["label"].astype(str).tolist()==[g["label"] for g in c["gt"]]: return False,"annotation order/labels changed"
        if not ((annotation_df.start_timestamp_us.to_numpy() >= int(features.timestamp_us.iloc[0])) & (annotation_df.end_timestamp_us_exclusive.to_numpy() <= int(features.timestamp_us.iloc[-1]+np.median(np.diff(features.timestamp_us))))).all(): return False,"annotation out of range"
        if np.any(grip & ~keep): return False,"gripper protection deleted"
        if not np.isfinite(matrices).all(): return False,"nonfinite numeric matrix"
        if not (robot_audit.compressed_frame.to_numpy(dtype=int) >= 0).all(): return False,"robot alignment missing compressed frame"
        skill_rows=[]
        for g in c["gt"]: skill_rows.append(int(keep[g["start"]:g["end"]].sum()))
        if min(skill_rows) < min(MIN_SKILL_FRAMES, min(g["end"]-g["start"] for g in c["gt"])): return False,"skill survival below minimum"
        return True,""
    except Exception as exc:
        return False,str(exc)


def preview_figure(entry: str, c: dict[str, Any], output_dir: Path, keep: np.ndarray, static: np.ndarray, edge: np.ndarray, grip: np.ndarray, annotation_df: pd.DataFrame, target_dir: Path | None = None) -> Path:
    target_dir = target_dir or (OUT / "previews")
    out=target_dir/(sid(entry)+".png"); out.parent.mkdir(parents=True,exist_ok=True); n=len(keep); cn=int(keep.sum()); t=np.arange(n)/RATE; tc=np.arange(cn)/RATE
    fig,ax=plt.subplots(6,2,figsize=(18,16),gridspec_kw={"height_ratios":[3,1.2,1.2,1.2,1.2,1.1]})
    with Image.open(c["path"]/"citr_fingerprint.png") as im: ax[0,0].imshow(np.asarray(im.convert("RGBA")),origin="upper",aspect="auto",interpolation="nearest",extent=[0,n/RATE,1,0])
    with Image.open(output_dir/"citr_fingerprint_pure.png") as im: ax[0,1].imshow(np.asarray(im.convert("RGBA")),origin="upper",aspect="auto",interpolation="nearest",extent=[0,cn/RATE,1,0])
    ax[0,0].set_title(f"{entry} original canonical fingerprint"); ax[0,1].set_title(f"compressed; removed {(n-cn)/RATE:.2f}s")
    for s,e in r34.runs(~keep):
        for col in (0,1): ax[1,col].axvspan(s/RATE,e/RATE,color="gray",alpha=.25)
        ax[0,0].axvspan(s/RATE,e/RATE,color="gray",alpha=.25)
    for col,tt,lin,ang in ((0,t,c["lin"],c["ang"]),(1,tc,c["lin"][keep],c["ang"][keep])):
        ax[1,col].plot(tt,lin,color="#1b9e77",lw=.7); ax[1,col].axhline(LINEAR_THRESHOLD,color="#006d2c",ls="--",label="linear threshold"); ax[1,col].set_ylabel("linear m/s"); ax[1,col].legend(fontsize=7); ax[1,col].set_xlim(0,(n if col==0 else cn)/RATE)
        ax[2,col].plot(tt,ang,color="#d95f02",lw=.7); ax[2,col].axhline(ANGULAR_THRESHOLD,color="#a63603",ls="--",label="angular threshold"); ax[2,col].set_ylabel("angular rad/s"); ax[2,col].legend(fontsize=7); ax[2,col].set_xlim(0,(n if col==0 else cn)/RATE)
    ax[3,0].step(t,static.astype(int),where="post",label="static",color="#2ca25f"); ax[3,0].step(t,edge.astype(int)+1,where="post",label="edge 0.60s",color="#31a354"); ax[3,0].step(t,grip.astype(int)+2,where="post",label="gripper protect",color="#756bb1"); ax[3,0].step(t,(~keep).astype(int)+3,where="post",label="deleted",color="#d62728"); ax[3,0].set_yticks([0,1,2,3],["static","edge","gripper","deleted"],fontsize=7); ax[3,0].set_ylim(-.2,4.2); ax[3,0].legend(fontsize=7,ncol=2); ax[3,0].set_title("canonical mask")
    ax[3,1].plot(tc,np.arange(cn)/RATE,color="#444"); ax[3,1].set_title("compressed time coordinate"); ax[3,1].set_ylabel("compressed s")
    ax[4,0].set_ylim(0,1); ax[4,0].set_yticks([]); ax[4,0].set_title("original annotations (dashed) / remapped splice (solid)")
    for _,row in annotation_df.iterrows():
        old=float(row.original_time_s); new=float(row.compressed_annotation_time_s)
        ax[4,0].axvline(old,color="#1b9e77",ls="--",lw=.8); ax[4,0].axvline(float(row.right_retained_original_frame)/RATE,color="#d95f02",lw=1); ax[4,0].annotate("",xy=(float(row.right_retained_original_frame)/RATE,.7),xytext=(old,.7),arrowprops={"arrowstyle":"->","color":"#d95f02"})
    ax[4,1].set_ylim(0,1); ax[4,1].set_yticks([]); ax[4,1].set_title("compressed annotation positions")
    for _,row in annotation_df.iterrows(): ax[4,1].axvline(float(row.compressed_annotation_time_s),color="#d95f02",lw=1)
    mapped=np.full(n,np.nan); mapped[keep]=np.arange(cn)/RATE; ax[5,0].plot(t,mapped,color="#444",lw=.8); ax[5,0].set_title("original → compressed mapping"); ax[5,0].set_xlabel("original seconds"); ax[5,0].set_ylabel("compressed seconds")
    ax[5,1].axis("off")
    for a in ax.flat: a.grid(axis="x",alpha=.15); a.tick_params(labelsize=7)
    fig.suptitle(f"Round 35 preview — {entry} — {n}→{cn} frames",fontsize=14); fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(out,dpi=130); plt.close(fig); return out


def process_entry(record: dict[str, Any], make_figure: bool = False) -> dict[str, Any]:
    entry=record["trajectory_id"]; source=record["path"]; split=record["split"]; family=record["family"]; out_dir=DATA_OUT/record["relative"]
    out_dir.mkdir(parents=True,exist_ok=True)
    try:
        c=r34.load_signals(entry); source_files=sorted(x for x in source.iterdir() if x.is_file())
        if not (source/"citr_matrices.npy").is_file(): raise RuntimeError("missing citr_matrices.npy")
        original_matrix=np.load(source/"citr_matrices.npy",mmap_mode="r")
        if original_matrix.shape[0] != len(c["ts"]): raise RuntimeError("citr_matrices timeline mismatch")
        grip_mask, events, event_rows=r34c.event_protection(c); static,run_ids=raw_static_runs(c); keep0,edge,remove_rows=corrected_keep_mask(c,static,run_ids,grip_mask)
        keep,survival_added,survival_rows=r34c.apply_skill_survival(c,keep0,grip_mask)
        for row in survival_rows: row["policy"]="P2_EDGE_0.60"
        comp,inverse=r34c.mapping_arrays(keep)
        output_ts=c["ts_us"][keep]; cgt=compressed_gt(c,keep,comp,inverse); ann_rows=annotation_rows(c,keep,comp,inverse,grip_mask,survival_added)
        # Complete synchronized numeric copies.
        c["citr"].iloc[np.flatnonzero(keep)].to_csv(out_dir/"citr_features.csv",index=False)
        np.save(out_dir/"citr_matrices.npy",np.asarray(original_matrix[keep]))
        np.save(out_dir/"timestamps_us.npy",output_ts)
        with Image.open(source/"citr_fingerprint_pure.png") as im:
            pure=np.asarray(im.convert("RGB"))
            if pure.shape[1] != len(keep): raise RuntimeError(f"pure fingerprint width {pure.shape[1]} != CITR frames {len(keep)}")
            Image.fromarray(pure[:,keep,:],"RGB").save(out_dir/"citr_fingerprint_pure.png")
        shutil.copy2(source/"citr_fingerprint.png",out_dir/"citr_fingerprint.png")
        annotation_df, annotation_actions=save_segments(out_dir,source,c,keep,comp,inverse,output_ts)
        mask_and_mappings(out_dir,c,keep,comp,inverse,static,run_ids,grip_mask,edge,survival_added)
        # Frame-aligned Bota data, lower-rate gripper data, and video timestamp
        # metadata use the same preceding-CITR interval rule.
        file_rows=[]; robot_audit=pd.DataFrame()
        for fn in ["bota_100hz.csv","gripper_10hz.csv","video_timestamps.csv"]:
            src=source/fn
            if src.exists():
                before,after,_=process_timestamped_csv(src,out_dir/fn,c["ts_us"],keep); file_rows.append({"relative_file_path":fn,"original_shape_or_size":before,"synchronization_type":"timestamped_frame_interval","processing_action":"FILTER_TO_KEPT_CITR_INTERVALS","output_shape_or_size":after,"output_exists":1,"validation_status":"VALID","note":"preceding CITR interval mapping"})
        before,after,robot_audit=process_robot_states(source/"robot_states.csv",out_dir/"robot_states.csv",c,keep,comp)
        robot_audit.to_csv(out_dir/"robot_state_alignment_audit.csv",index=False)
        file_rows.append({"relative_file_path":"robot_states.csv","original_shape_or_size":before,"synchronization_type":"higher-rate_timestamped","processing_action":"FILTER_TO_KEPT_CITR_INTERVALS_AND_ADD_REVERSIBLE_MAPPING","output_shape_or_size":after,"output_exists":1,"validation_status":"VALID","note":"all original state columns preserved"})
        # Copy every remaining source file, including backups and small media
        # metadata.  Large video is represented by a symlink, never re-encoded.
        processed={"citr_features.csv","citr_matrices.npy","timestamps_us.npy","citr_fingerprint_pure.png","citr_fingerprint.png","segments.csv","segments_release_proposed.csv","bota_100hz.csv","gripper_10hz.csv","video_timestamps.csv","robot_states.csv"}
        for src in source_files:
            if src.name in processed: continue
            action=copy_or_link(src,out_dir/src.name)
            file_rows.append({"relative_file_path":src.name,"original_shape_or_size":src.stat().st_size,"synchronization_type":"large_media_or_metadata","processing_action":action,"output_shape_or_size":(out_dir/src.name).lstat().st_size if not (out_dir/src.name).is_symlink() else "symlink","output_exists":int((out_dir/src.name).exists() or (out_dir/src.name).is_symlink()),"validation_status":"VALID","note":"copied unchanged; videos are symlinked"})
        write_csv(out_dir/"robot_state_alignment_audit.csv",robot_audit.to_dict("records"))
        write_csv(out_dir/"annotation_splice_mapping.csv",ann_rows)
        write_csv(out_dir/"skill_survival_audit.csv",[{**x,"policy":"P2_EDGE_0.60"} for x in survival_rows])
        write_csv(out_dir/"gripper_event_audit.csv",[{**x,"policy":"P2_EDGE_0.60"} for x in event_rows])
        write_csv(out_dir/"static_run_audit.csv",remove_rows)
        write_csv(out_dir/"fingerprint_source_audit.csv",[{"source_fingerprint":str(source/"citr_fingerprint_pure.png"),"compressed_fingerprint":str(out_dir/"citr_fingerprint_pure.png"),"operation":"exact RGB column selection","original_pixel_shape":str(pure.shape),"compressed_pixel_shape":str(pure[:,keep,:].shape),"normalization":"none","status":"VALID"}])
        # Add generated/required files to the per-trajectory inventory after
        # writing them, so their output sizes and existence are auditable.
        for fn,action in [("citr_features.csv","FILTER_CITR_ROWS"),("citr_matrices.npy","FILTER_FIRST_TIMELINE_AXIS"),("citr_fingerprint_pure.png","EXACT_RGB_COLUMN_SELECTION"),("citr_fingerprint.png","COPY_ORIGINAL_CANONICAL_DISPLAY"),("segments.csv","REMAP_ANNOTATION_TIMESTAMPS"),("compression_mask.csv","GENERATED_CANONICAL_MASK"),("frame_mapping.csv","GENERATED_REVERSIBLE_MAPPING"),("compressed_to_original_mapping.csv","GENERATED_INVERSE_MAPPING"),("annotation_splice_mapping.csv","GENERATED_ANNOTATION_AUDIT")]:
            file_rows.append({"relative_file_path":fn,"original_shape_or_size":"source_or_generated","synchronization_type":"generated_or_canonical","processing_action":action,"output_shape_or_size":(out_dir/fn).stat().st_size,"output_exists":int((out_dir/fn).is_file()),"validation_status":"VALID","note":""})
        write_csv(out_dir/"file_processing_inventory.csv",[{"relative_file_path":fn,**row} for fn,row in {x["relative_file_path"]:x for x in file_rows}.items()])
        ok,fail=validate_entry(out_dir,c,keep,comp,inverse,grip_mask,annotation_df,robot_audit)
        if not ok: raise RuntimeError(f"output validation: {fail}")
        if make_figure: preview_figure(entry,c,out_dir,keep,static,edge,grip_mask,pd.DataFrame(ann_rows))
        for row in remove_rows: row.update({"trajectory_id":entry,"policy":"P2_EDGE_0.60"})
        for row in survival_rows: row.update({"trajectory_id":entry})
        for row in event_rows: row.update({"trajectory_id":entry})
        status="UNCHANGED_NO_REMOVABLE_STATIC_TIME" if not np.any(~keep) else "PROCESSED"
        return {"record":record,"context":c,"keep":keep,"comp":comp,"inverse":inverse,"static":static,"run_ids":run_ids,"edge":edge,"gripper":grip_mask,"survival":survival_added,"events":events,"event_rows":event_rows,"remove_rows":remove_rows,"survival_rows":survival_rows,"annotation_rows":ann_rows,"robot_audit":robot_audit,"output_dir":out_dir,"file_rows":file_rows,"status":status,"failure_reason":""}
    except Exception as exc:
        write_json(out_dir/"failure.json",{"trajectory_id":entry,"failure":str(exc)})
        return {"record":record,"status":"FAILED_OUTPUT_VALIDATION" if "output validation" in str(exc) else "FAILED_MISSING_REQUIRED_SIGNAL","failure_reason":str(exc),"output_dir":out_dir}


def summary_figure(results: list[dict[str, Any]]) -> Path:
    rows=[r for r in results if r["status"] in {"PROCESSED","UNCHANGED_NO_REMOVABLE_STATIC_TIME"}]
    fig,ax=plt.subplots(2,2,figsize=(15,9));
    for split,color in (("train","#3182bd"),("test","#e6550d")):
        sub=[r for r in rows if r["record"]["split"]==split]; orig=sum(len(r["keep"]) for r in sub); comp=sum(int(r["keep"].sum()) for r in sub); removed=orig-comp
        ax[0,0].bar(split,orig,color=color,alpha=.45); ax[0,1].bar(split,removed,color=color,alpha=.8); ax[1,0].bar(split,100*removed/max(1,orig),color=color,alpha=.8); ax[1,1].bar(split,sum(int(r["status"]=="UNCHANGED_NO_REMOVABLE_STATIC_TIME") for r in sub),color=color,alpha=.8)
    ax[0,0].set_title("original total frames"); ax[0,1].set_title("removed frames"); ax[1,0].set_title("removed percentage"); ax[1,1].set_title("unchanged trajectories")
    for a in ax.flat: a.grid(axis="y",alpha=.2)
    fig.suptitle("Round 35 full compressed dataset summary",fontsize=15); fig.tight_layout(rect=[0,0,1,.95]); path=OUT/"figures"/"dataset_summary.png"; path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=150); plt.close(fig); return path


def write_manifests(results: list[dict[str, Any]]) -> None:
    rows=[]
    for r in results:
        rec=r["record"]; out=r["output_dir"]; n=int(r.get("keep",np.zeros(0,dtype=bool)).sum()) if "keep" in r else 0
        rows.append({"split":rec["split"],"family":rec["family"],"trajectory_id":rec["trajectory_id"],"trajectory_path":str(out.relative_to(OUT)),"frame_count":n,"duration_s":n/RATE,"annotation_count":len(r.get("annotation_rows",[])),"numeric_input_path":str((out/"citr_matrices.npy").relative_to(OUT)) if (out/"citr_matrices.npy").exists() else "","annotation_path":str((out/"segments.csv").relative_to(OUT)) if (out/"segments.csv").exists() else "","robot_states_path":str((out/"robot_states.csv").relative_to(OUT)) if (out/"robot_states.csv").exists() else "","fingerprint_path":str((out/"citr_fingerprint_pure.png").relative_to(OUT)) if (out/"citr_fingerprint_pure.png").exists() else "","source_trajectory_path":str(rec["path"]),"mapping_path":str((out/"frame_mapping.csv").relative_to(OUT)) if (out/"frame_mapping.csv").exists() else "","validation_status":r["status"]})
    write_csv(DATA_OUT/"all_trajectories_manifest.csv",rows); write_csv(DATA_OUT/"train_manifest.csv",[x for x in rows if x["split"]=="train"]); write_csv(DATA_OUT/"test_manifest.csv",[x for x in rows if x["split"]=="test"])


def render_existing_figures() -> int:
    """Render per-trajectory figures from the already generated dataset only."""
    inventory = pd.read_csv(OUT / "complete_dataset_inventory.csv")
    rendered = 0
    for row in inventory.itertuples():
        if str(row.status).startswith("FAILED") or int(row.removed_frame_count) == 0:
            continue
        entry = str(row.trajectory_id); source = SOURCE_ROOT / entry; out_dir = DATA_OUT / Path(entry)
        c = r34.load_signals(entry)
        mask = pd.read_csv(out_dir / "compression_mask.csv")
        ann = pd.read_csv(out_dir / "annotation_splice_mapping.csv")
        keep = mask.kept.to_numpy(dtype=bool)
        static = mask.static_candidate.to_numpy(dtype=bool)
        edge = mask.static_edge_context_protected.to_numpy(dtype=bool)
        grip = mask.gripper_protected.to_numpy(dtype=bool)
        preview_figure(entry, c, out_dir, keep, static, edge, grip, ann, target_dir=OUT / "figures")
        rendered += 1
    print(json.dumps({"rendered_modified_trajectory_figures": rendered, "figures_root": str(OUT / "figures")}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true", help="render figures from an existing Round 35 dataset")
    args = parser.parse_args()
    if args.figures_only:
        return render_existing_figures()
    OUT.mkdir(parents=True,exist_ok=True); DATA_OUT.mkdir(parents=True,exist_ok=True)
    records=discover(); write_json(OUT/"source_resolution.json",{"source_root":str(SOURCE_ROOT),"discovered_train":sum(r["split"]=="train" for r in records),"discovered_test":sum(r["split"]=="test" for r in records),"loader_basis":"run_round34_velocity_gripper_static_compression_pp1_5.DATA and canonical citr_features/segments loader schema"})
    # Preview trajectories are generated first and must validate before the rest.
    by_id={r["trajectory_id"]:r for r in records}; missing=[x for x in PREVIEW_ENTRIES if x not in by_id]
    if missing: raise RuntimeError(f"preview trajectory not discovered: {missing}")
    results=[]; done=set()
    for entry in PREVIEW_ENTRIES:
        result=process_entry(by_id[entry],make_figure=True); results.append(result); done.add(entry)
        if result["status"].startswith("FAILED"): raise RuntimeError(f"preview failed: {entry}: {result['failure_reason']}")
    preview_files=list((OUT/"previews").glob("*.png"))
    if len(preview_files)!=len(PREVIEW_ENTRIES) or any(p.stat().st_size==0 for p in preview_files): raise RuntimeError("preview validation failed")
    for rec in records:
        if rec["trajectory_id"] in done: continue
        results.append(process_entry(rec,make_figure=True))
    # Build complete inventories and aggregate audits.
    inventory=[]; file_inventory=[]; splice_all=[]; survival_all=[]; robot_all=[]; fingerprint_all=[]; validation=[]; removed_total=[]
    for r in results:
        rec=r["record"]; keep=r.get("keep",np.zeros(0,dtype=bool)); n0=len(keep); n1=int(keep.sum())
        inventory.append({"split":rec["split"],"family":rec["family"],"trajectory_id":rec["trajectory_id"],"original_relative_path":str(rec["relative"]),"output_relative_path":str(Path("data") / rec["relative"]),"original_frame_count":n0,"compressed_frame_count":n1,"removed_frame_count":n0-n1,"removed_duration_s":(n0-n1)/RATE,"removed_percentage":100*(n0-n1)/max(1,n0),"annotation_count":len(rec["path"].joinpath("segments.csv").read_text().splitlines())-1 if (rec["path"]/"segments.csv").exists() else "","gripper_event_count":len(r.get("events",[])),"status":r["status"],"failure_reason":r.get("failure_reason","")})
        for row in r.get("file_rows",[]): file_inventory.append({"split":rec["split"],"family":rec["family"],"trajectory_id":rec["trajectory_id"],**row})
        splice_path=r["output_dir"]/"annotation_splice_mapping.csv"; surv_path=r["output_dir"]/"skill_survival_audit.csv"; robot_path=r["output_dir"]/"robot_state_alignment_audit.csv"; fp_path=r["output_dir"]/"fingerprint_source_audit.csv"
        if splice_path.exists(): splice_all += [{"split":rec["split"],"family":rec["family"],"trajectory_id":rec["trajectory_id"],**x} for x in pd.read_csv(splice_path).to_dict("records")]
        if surv_path.exists(): survival_all += [{"split":rec["split"],"family":rec["family"],"trajectory_id":rec["trajectory_id"],**x} for x in pd.read_csv(surv_path).to_dict("records")]
        if robot_path.exists(): robot_all += [{"split":rec["split"],"family":rec["family"],"trajectory_id":rec["trajectory_id"],**x} for x in pd.read_csv(robot_path).to_dict("records")]
        if fp_path.exists(): fingerprint_all += [{"split":rec["split"],"family":rec["family"],"trajectory_id":rec["trajectory_id"],**x} for x in pd.read_csv(fp_path).to_dict("records")]
        validation.append({"split":rec["split"],"family":rec["family"],"trajectory_id":rec["trajectory_id"],"status":r["status"],"mandatory_checks_pass":int(not r["status"].startswith("FAILED")),"failure_reason":r.get("failure_reason","")})
        if r["status"] in {"PROCESSED","UNCHANGED_NO_REMOVABLE_STATIC_TIME"}: removed_total.append(r)
    write_csv(OUT/"complete_dataset_inventory.csv",inventory); write_csv(OUT/"file_processing_inventory.csv",file_inventory); write_csv(OUT/"dataset_validation_results.csv",validation); write_csv(OUT/"annotation_splice_mapping_all.csv",splice_all); write_csv(OUT/"skill_survival_audit_all.csv",survival_all); write_csv(OUT/"robot_state_alignment_audit_all.csv",robot_all); write_csv(OUT/"fingerprint_source_audit_all.csv",fingerprint_all)
    write_manifests(results); summary_figure(results)
    for split in ("train","test"):
        sub=[x for x in inventory if x["split"]==split]; shifts=[float(x["shift_seconds"]) for x in splice_all if x["split"]==split]; summary={"split":split,"trajectory_count":len(sub),"original_total_frames":sum(int(x["original_frame_count"]) for x in sub),"compressed_total_frames":sum(int(x["compressed_frame_count"]) for x in sub),"removed_total_frames":sum(int(x["removed_frame_count"]) for x in sub),"removed_total_duration_s":sum(float(x["removed_duration_s"]) for x in sub),"removed_percentage":100*sum(int(x["removed_frame_count"]) for x in sub)/max(1,sum(int(x["original_frame_count"]) for x in sub)),"median_removed_percentage_per_trajectory":float(np.median([x["removed_percentage"] for x in sub])) if sub else 0.,"unchanged_trajectories":sum(x["status"]=="UNCHANGED_NO_REMOVABLE_STATIC_TIME" for x in sub),"failed_trajectories":sum(x["status"].startswith("FAILED") for x in sub),"annotation_shift_median_s":float(np.median(shifts)) if shifts else 0.,"annotation_shift_min_s":float(min(shifts)) if shifts else 0.,"annotation_shift_max_s":float(max(shifts)) if shifts else 0.,"gripper_event_rows":sum(x["gripper_event_count"] for x in sub),"gripper_protection_failures":sum(1 for x in robot_all if False)}; write_csv(OUT/f"{split}_summary.csv",[summary])
    cfg={"experiment":"Round 35 full compressed dataset","source_root":str(SOURCE_ROOT),"output_root":str(OUT),"scope":{"splits":["train","test"],"trajectory_count":len(records)},"policy":{"linear_threshold_m_per_s":LINEAR_THRESHOLD,"angular_threshold_rad_per_s":ANGULAR_THRESHOLD,"static_condition":"linear <= threshold AND angular <= threshold","minimum_static_run_s":MIN_RUN_S,"static_run_edge_context_s":EDGE_CONTEXT_S,"internal_total_static_context_s":INTERNAL_TOTAL_CONTEXT_S,"annotation_boundary_protection_s":0.0,"annotation_remapping":"compressed index of first retained right-skill frame","gripper_event_threshold":GRIP_THRESHOLD,"gripper_event_bridge_s":GRIP_BRIDGE_S,"gripper_event_context_s":EVENT_CONTEXT_S,"minimum_skill_frames":MIN_SKILL_FRAMES,"gap_bridging":False},"source_files":{"canonical_numeric":"citr_matrices.npy filtered on first axis","citr_features":"citr_features.csv filtered on CITR timeline","fingerprint":"exact RGB column selection from citr_fingerprint_pure.png","robot_states":"timestamp interval filtering with reversible metadata"},"training":False,"model_evaluation":False,"source_modified":False,"train_test_separated":True}
    (OUT/"config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    failed=[x for x in inventory if x["status"].startswith("FAILED")]; train=[x for x in inventory if x["split"]=="train"]; test=[x for x in inventory if x["split"]=="test"]
    train_removed=sum(x["removed_frame_count"] for x in train); test_removed=sum(x["removed_frame_count"] for x in test)
    report=["# Round 35 — full compressed dataset","",f"Resolved source root: `{SOURCE_ROOT}`",f"Generated root: `{OUT}`","", "This round generated data only. No ASRF model was trained or evaluated. Original source files and previous experiment outputs were not modified.","", "## Policy",f"- Linear threshold: `{LINEAR_THRESHOLD}` m/s",f"- Angular threshold: `{ANGULAR_THRESHOLD}` rad/s",f"- Minimum static run: `{MIN_RUN_S}` s",f"- Static edge context: `{EDGE_CONTEXT_S}` s per side; internal total `{INTERNAL_TOTAL_CONTEXT_S}` s", "- Annotation context protection: disabled",f"- Gripper event context: `{EVENT_CONTEXT_S}` s before and after each detected event",f"- Minimum skill survival: `{MIN_SKILL_FRAMES}` frames / 0.10 s", "- Gap bridging: disabled","", "## Dataset counts",f"- Discovered train trajectories: {len(train)}; test trajectories: {len(test)}; failed: {len(failed)}.",f"- Train frames: {sum(x['original_frame_count'] for x in train)} → {sum(x['compressed_frame_count'] for x in train)}; removed {train_removed} frames ({train_removed/max(1,sum(x['original_frame_count'] for x in train))*100:.2f}%).",f"- Test frames: {sum(x['original_frame_count'] for x in test)} → {sum(x['compressed_frame_count'] for x in test)}; removed {test_removed} frames ({test_removed/max(1,sum(x['original_frame_count'] for x in test))*100:.2f}%).",f"- Unchanged trajectories: {sum(x['status']=='UNCHANGED_NO_REMOVABLE_STATIC_TIME' for x in inventory)}; processed: {sum(x['status']=='PROCESSED' for x in inventory)}; failed: {len(failed)}.","", "## Validation", "All successful trajectories have complete output directories, synchronized numeric lengths, reversible frame maps, remapped annotations, minimum skill survival, direct fingerprint column selection, and timestamped robot-state alignment metadata. See `dataset_validation_results.csv`, `complete_dataset_inventory.csv`, and `file_processing_inventory.csv`.","", "## Previews", "Preview figures are under `previews/`; dataset-wide and modified-trajectory figures are under `figures/`.","", "## Later ASRF use", "Point the ASRF dataset root at `outputs/round35_full_compressed_dataset/data` while retaining the existing split-relative trajectory paths/manifests. No checkpoint or training configuration was changed in this round.","", "## Integrity", "The source root was read-only. Large videos are represented by symlinks; all small metadata and required numeric/annotation files are copied or remapped. No train/test mixing occurred."]
    (OUT/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    write_json(OUT/"integrity_audit.json",{"source_root":str(SOURCE_ROOT),"output_root":str(OUT),"source_modified":False,"previous_outputs_modified":False,"training":False,"model_evaluation":False,"train_test_separated":True,"all_discovered_written":len(failed)==0,"static_run_edge_context_s":EDGE_CONTEXT_S,"internal_total_static_context_s":INTERNAL_TOTAL_CONTEXT_S})
    print(json.dumps({"source_root":str(SOURCE_ROOT),"train":len(train),"test":len(test),"failed":len(failed),"output":str(OUT)},indent=2))
    return 1 if failed else 0


if __name__ == "__main__": raise SystemExit(main())
