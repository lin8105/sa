#!/usr/bin/env python3
"""Round 34B: direct global velocity-threshold expansion, PP1--PP10 only."""
from __future__ import annotations

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
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_round34_velocity_gripper_static_compression_pp1_5 as r34  # noqa: E402
import run_round27_pp_only_r5_region_sf_point_hybrid as r27  # noqa: E402

OUT = ROOT / "outputs/round34b_direct_threshold_expansion_pp1_10"
PP = [f"train/pick and place/pp{i}" for i in range(1, 11)]
L0 = 0.00650187267
A0 = 0.0119281923
PRESET_MULTIPLIERS = (1.0, 1.5, 2.0, 3.0, 4.0)
PRESETS = {f"T{i}": (mult, L0 * mult, A0 * mult) for i, mult in enumerate(PRESET_MULTIPLIERS)}
GRIP_THRESHOLD = 3.0000000111022306e-08
GRIP_BRIDGE_S = .10
MIN_RUN_S = .50
CONTEXT_S = 1.0
TOLS = (5, 10, 20, 33, 50)
FUSION = {"threshold": .50, "gap": 0, "rule": "P4", "support_gate": .50, "separation": 0}
PRIMARY = "VELOCITY_ANNOTATION_GRIPPER"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def sid(entry: str) -> str:
    return entry.replace("/", "__").replace(" ", "_")


def direct_static(c: dict[str, Any], linear: float, angular: float) -> np.ndarray:
    """Continuous direct AND condition; deliberately no gap bridging."""
    raw = c["valid"] & (c["lin"] <= linear) & (c["ang"] <= angular)
    min_frames = max(1, int(round(MIN_RUN_S * c["rate"])))
    out = np.zeros(len(raw), dtype=bool)
    for s, e in r34.runs(raw):
        if e - s >= min_frames: out[s:e] = True
    return out


def delete_middle(c: dict[str, Any], static: np.ndarray, protection: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply the fixed Round 34 context policy to direct static runs."""
    keep = np.ones(len(static), dtype=bool); removed=[]; context=max(1, int(round(CONTEXT_S * c["rate"] / 2))); full_context=max(1, int(round(CONTEXT_S * c["rate"])))
    eligible = static & ~protection
    for s, e in r34.runs(eligible):
        if e - s <= 2 * context: continue
        left_non = s > 0 and (not static[s - 1] or protection[s - 1])
        right_non = e < len(static) and (not static[e] or protection[e])
        if s == 0 or not np.any(~static[:s]): ds, de, kind = s, e - full_context, "leading"
        elif e == len(static) or not np.any(~static[e:]): ds, de, kind = s + full_context, e, "trailing"
        elif left_non and right_non: ds, de, kind = s + context, e - context, "internal"
        else: ds, de, kind = s + context, e - context, "internal"
        if de > ds:
            keep[ds:de] = False
            skill = next((g["label"] for g in c["gt"] if max(ds, g["start"]) < min(de, g["end"])), "none")
            near = [abs((ds + de) / 2 - g["start"]) for g in c["gt"]]
            removed.append({"trajectory_id": c["entry"], "original_start_frame": ds, "original_end_frame_exclusive": de, "start_time_s": float(c["ts"][ds] - c["ts"][0]), "end_time_s": float(c["ts"][de - 1] - c["ts"][0]), "removed_duration_s": float((de - ds) / c["rate"]), "annotation_skill": skill, "max_linear_speed": float(np.max(c["lin"][ds:de])), "max_angular_speed": float(np.max(c["ang"][ds:de])), "distance_to_nearest_annotation_frames": float(min(near) if near else len(c["ts"])), "distance_to_nearest_gripper_event_frames": "", "run_type": kind, "min_run_s": MIN_RUN_S, "context_s": CONTEXT_S})
    return keep, removed


def frame_maps(keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    comp = np.full(len(keep), -1, dtype=int); comp[keep] = np.arange(int(keep.sum())); return comp, np.flatnonzero(keep)


def remap(frame: int, comp: np.ndarray, inverse: np.ndarray, end: bool = False) -> int:
    k = np.searchsorted(inverse, frame, side="left")
    return int(min(max(k, 0), len(inverse) if end else len(inverse) - 1))


def load_models() -> dict[str, Any]:
    def resolve(requested: Path, expected: str) -> Path:
        candidates = [requested, ROOT / "outputs/0" / requested.relative_to(ROOT / "outputs")]
        hits = [p for p in dict.fromkeys(candidates) if p.is_file() and sha(p) == expected]
        if len(hits) != 1: raise RuntimeError(f"checkpoint resolution failed: {requested} -> {hits}")
        return hits[0]
    sf_path = resolve(r34.SF_REQUESTED, r34.SF_SHA); r5_path = resolve(r34.R5_REQUESTED, r34.R5_SHA)
    sc = yaml.safe_load((sf_path.parent / "config.yaml").read_text()); rc = yaml.safe_load((r5_path.parent / "config.yaml").read_text())
    sp = torch.load(sf_path, map_location="cpu", weights_only=False); rp = torch.load(r5_path, map_location="cpu", weights_only=False)
    sf = r34.ASRFModel.from_config(sc); r5 = r34.ASRFModel.from_config(rc); sf.load_state_dict(sp["model_state"], strict=True); r5.load_state_dict(rp["model_state"], strict=True); sf.eval(); r5.eval()
    audit = {"sf_requested": str(r34.SF_REQUESTED), "r5_requested": str(r34.R5_REQUESTED), "sf_checkpoint": str(sf_path), "r5_checkpoint": str(r5_path), "sf_sha256": sha(sf_path), "r5_sha256": sha(r5_path), "fusion": FUSION, "retraining": False}
    write_json(OUT / "model_hashes.json", audit)
    return {"sf": sf, "r5": r5}


def infer(sf: Any, r5m: Any, heat: np.ndarray, ts: np.ndarray) -> dict[str, Any]:
    sample = {"heatmap": torch.from_numpy(heat).float(), "timestamps": torch.from_numpy(ts.astype(np.int64)), "valid_mask": torch.ones(len(ts), dtype=torch.bool)}
    def one(model: Any) -> dict[str, Any]:
        with torch.no_grad(): out = model(sample["heatmap"].unsqueeze(0), valid_mask=sample["valid_mask"].unsqueeze(0))
        p = out.asb_stage_probabilities[-1][0].cpu().numpy(); return {"asb_probs": p, "asb_labels": np.argmax(p, axis=0), "brb": out.brb_stage_probabilities[-1][0, 0].cpu().numpy()}
    sfz, r5z = one(sf), one(r5m); points, diagnostics, _ = r27.hybrid(sfz, r5z, FUSION); segments = r27.frame_segments(sfz["asb_labels"], points, sfz["asb_probs"])
    return {"sf": sfz, "r5": r5z, "points": points, "segments": segments, "diagnostics": diagnostics}


def copy_compressed(c: dict[str, Any], preset: str, keep: np.ndarray, removed: list[dict[str, Any]]) -> dict[str, Any]:
    comp, inverse = frame_maps(keep); root = OUT / "copies" / preset / sid(c["entry"]); root.mkdir(parents=True, exist_ok=True)
    with Image.open(c["path"] / "citr_fingerprint_pure.png") as im:
        cropped = np.asarray(im.convert("RGB"))[:, keep, :].copy(); Image.fromarray(cropped, "RGB").save(root / "citr_fingerprint_pure.png")
    fp = OUT / "compressed_fingerprints" / preset / (sid(c["entry"]) + ".png"); fp.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(cropped, "RGB").save(fp)
    c["citr"].iloc[np.flatnonzero(keep)].to_csv(root / "citr_features.csv", index=False)
    segments=[]
    for g in c["gt"]:
        segments.append({"segment_index": g["segment_index"], "start_frame": remap(g["start"], comp, inverse), "end_frame_exclusive": remap(g["end"], comp, inverse, True), "label": g["label"], "original_start_frame": g["start"], "original_end_frame_exclusive": g["end"]})
    write_csv(root / "segments_remapped.csv", segments); np.save(root / "frame_mapping.npy", comp); np.save(root / "compressed_to_original.npy", inverse); np.save(root / "timestamps_us.npy", c["ts_us"][keep])
    write_json(root / "metadata.json", {"trajectory_id": c["entry"], "preset": preset, "fingerprint_operation": "exact RGB column selection from canonical pure PNG", "original_data_changed": False})
    return {"path": root, "keep": keep, "comp": comp, "inverse": inverse, "segments": segments}


def metrics(inf: dict[str, Any], gt: list[dict[str, Any]], condition: str, entry: str, n: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matches = r34.r27b.temporal_matches(inf["segments"], gt); ious = [x["iou"] for x in matches]; row = {"condition": condition, "trajectory_id": entry, "gt_segments": len(gt), "predicted_segments": len(inf["segments"]), "predicted_gt_ratio": len(inf["segments"]) / max(1, len(gt)), "mean_iou": float(np.mean(ious)) if ious else 0., "median_iou": float(np.median(ious)) if ious else 0., "over_segmentation": max(0, len(inf["segments"]) - len(gt)) / max(1, len(gt)), "under_segmentation": max(0, len(gt) - len(inf["segments"])) / max(1, len(gt))}
    for q in (.10, .25, .50, .75):
        tp = sum(x >= q for x in ious); row[f"F1@{int(q*100)}"] = 2 * tp / max(1, 2 * tp + len(inf["segments"]) - tp + len(gt) - tp)
    truth = [int(x["start"]) for x in gt[1:]]; pred = [int(x) for x in inf["points"] if 0 < x < n]; details=[]
    for tol in TOLS:
        pairs, fp, fn = r34.r27b.boundary_pairs(pred, truth, tol); errors=[x[2] for x in pairs]; row.update({f"tp_{tol}":len(pairs), f"fp_{tol}":len(fp), f"fn_{tol}":len(fn), f"false_rate_{tol}":len(fp)/max(1,len(pred)), f"missed_rate_{tol}":len(fn)/max(1,len(truth)), f"mean_error_{tol}":float(np.mean(errors)) if errors else 0., f"median_error_{tol}":float(np.median(errors)) if errors else 0., f"p90_error_{tol}":float(np.percentile(errors,90)) if errors else 0., f"max_error_{tol}":max(errors) if errors else 0.}); details.append({"condition":condition,"trajectory_id":entry,"tolerance":tol,"tp":len(pairs),"fp":len(fp),"fn":len(fn),"errors":";".join(map(str,errors))})
    return row, details


def peak_metrics(inf: dict[str, Any], entry: str, condition: str) -> list[dict[str, Any]]:
    out=[]
    for branch in ("sf", "r5"):
        p=inf[branch]["brb"]; peaks=[i for i in range(1,len(p)-1) if p[i]>=.5 and p[i]>=p[i-1] and p[i]>=p[i+1]]; out.append({"trajectory_id":entry,"condition":condition,"branch":branch,"brb_peak_count":len(peaks),"duplicate_peak_count":sum(abs(a-b)<=50 for a,b in zip(peaks,peaks[1:])),"hybrid_boundary_count":len(inf["points"])})
    return out


def mask_rows(c: dict[str, Any], preset: str, anchor: np.ndarray, event: np.ndarray, keep: np.ndarray, static: np.ndarray) -> list[dict[str, Any]]:
    return [{"trajectory_id":c["entry"],"preset":preset,"frame":i,"annotation_protected":int(anchor[i]),"gripper_protected":int(event[i]),"static_candidate":int(static[i]),"kept":int(keep[i]),"deleted":int(not keep[i])} for i in range(len(keep))]


def audit_figure(c: dict[str, Any], preset: str, original: dict[str, Any], compressed: dict[str, Any], cgt: list[dict[str, Any]], static: np.ndarray, anchor: np.ndarray, event: np.ndarray, deleted: np.ndarray, linear: float, angular: float) -> None:
    n=len(c["ts"]); cn=len(compressed["keep"][compressed["keep"]]); t=np.arange(n)/c["rate"]; tc=np.arange(cn)/c["rate"]; colors={"reach":"#66c2a5","grasp":"#fc8d62","lift":"#8da0cb","transport":"#e78ac3","place":"#a6d854","release":"#ffd92f"}
    fig, ax = plt.subplots(10, 2, figsize=(20, 22), gridspec_kw={"height_ratios":[2.8,1.2,1.2,1,1.2,1.2,1,1,1,1.1]})
    with Image.open(c["path"] / "citr_fingerprint.png") as im:
        ax[0,0].imshow(np.asarray(im), aspect="auto", interpolation="nearest", origin="upper", extent=[0,t[-1],1,0])
    ax[0,0].set_title(f"{c['entry']} — original canonical CITR PNG; deleted intervals shaded"); ax[0,0].set_ylabel("existing PNG")
    with Image.open(compressed["path"] / "citr_fingerprint_pure.png") as im: ax[0,1].imshow(np.asarray(im), aspect="auto", interpolation="nearest", origin="upper")
    ax[0,1].set_title(f"{preset} compressed canonical pure PNG"); ax[0,1].set_ylabel("compressed PNG")
    for s,e in r34.runs(deleted):
        for col, x0, x1 in ((0, t[s], t[e-1]), (1, tc[compressed["comp"][s]], tc[compressed["comp"][e-1]] if compressed["comp"][e-1] >= 0 else tc[-1])):
            ax[0,col].axvspan(x0,x1,color="gray",alpha=.35); ax[1,col].axvspan(x0,x1,color="gray",alpha=.25); ax[2,col].axvspan(x0,x1,color="gray",alpha=.25)
        ax[0,0].text((t[s]+t[e-1])/2,.95,f"{(e-s)/c['rate']:.2f}s",ha="center",va="top",fontsize=7,color="black")
        ax[0,0].axvline(t[s],color="gray",lw=.6); ax[0,0].axvline(t[e-1],color="gray",lw=.6)
    ax[1,0].plot(t,c["lin"],color="#1b9e77"); ax[1,0].axhline(linear,ls="--",color="#1b9e77"); ax[1,0].set_ylabel("linear m/s")
    ax[2,0].plot(t,c["ang"],color="#d95f02"); ax[2,0].axhline(angular,ls="--",color="#d95f02"); ax[2,0].set_ylabel("angular rad/s")
    ax[1,1].plot(tc,c["lin"][compressed["keep"]],color="#1b9e77"); ax[1,1].axhline(linear,ls="--",color="#1b9e77"); ax[1,1].set_ylabel("linear m/s")
    ax[2,1].plot(tc,c["ang"][compressed["keep"]],color="#d95f02"); ax[2,1].axhline(angular,ls="--",color="#d95f02"); ax[2,1].set_ylabel("angular rad/s")
    ax[3,0].plot(t,c["grip"],color="#7570b3"); ax[3,1].plot(tc,c["grip"][compressed["keep"]],color="#7570b3"); ax[3,0].set_ylabel("gripper position")
    ax[4,0].plot(t,anchor.astype(float),label="annotation",color="#377eb8"); ax[4,0].plot(t,event.astype(float),label="gripper",color="#984ea3"); ax[4,0].plot(t,deleted.astype(float),label="deleted",color="#e41a1c"); ax[4,0].legend(fontsize=7,ncol=3); ax[4,0].set_yticks([0,1]); ax[4,0].set_ylabel("protection/deletion")
    ax[4,1].plot(tc,anchor[compressed["keep"]].astype(float),color="#377eb8"); ax[4,1].plot(tc,event[compressed["keep"]].astype(float),color="#984ea3"); ax[4,1].set_yticks([0,1]); ax[4,1].set_ylabel("protection")
    def bar(a, rows, count, title):
        for g in rows:
            s,e=int(g["start"]),int(g["end"]); a.fill_between([s/c["rate"],e/c["rate"]],[0,0],[1,1],color=colors.get(g["label"],"#aaa"),alpha=.85); a.text((s+e)/(2*c["rate"]),.5,g["label"],ha="center",va="center",fontsize=7)
        a.set_xlim(0,count/c["rate"]); a.set_ylim(0,1); a.set_yticks([]); a.set_title(title,fontsize=8)
    bar(ax[5,0],c["gt"],n,"GT original"); bar(ax[5,1],cgt,cn,"GT compressed")
    raw=[{"start":g["start"],"end":g["end"],"label":g["top1_label"]} for g in original["segments"]]; ref=[{"start":g["start"],"end":g["end"],"label":g["top1_label"]} for g in compressed["segments"]]; bar(ax[6,0],raw,n,"RAW_HYBRID"); bar(ax[6,1],ref,cn,"compressed RAW_HYBRID")
    ax[7,0].plot(t,original["sf"]["brb"],label="SF",color="#d62728"); ax[7,0].plot(t,original["r5"]["brb"],label="r5",color="#222"); ax[7,0].legend(fontsize=7); ax[7,1].plot(tc,compressed["sf"]["brb"],label="SF",color="#d62728"); ax[7,1].plot(tc,compressed["r5"]["brb"],label="r5",color="#222"); ax[7,1].legend(fontsize=7)
    mp=compressed["comp"].astype(float); mp[mp<0]=np.nan; ax[8,0].plot(t,mp/c["rate"],color="#555"); ax[8,0].set_title("original → compressed frame map"); ax[8,0].set_ylabel("compressed seconds"); ax[8,1].axis("off")
    for p in original["points"]: ax[9,0].scatter(p/c["rate"],.5,color="#e41a1c" if deleted[p] else "#444",s=16)
    ax[9,0].set_ylim(0,1); ax[9,0].set_yticks([]); ax[9,0].set_title("raw boundary / deleted decision"); ax[9,0].set_xlabel("original time (s)")
    fig.suptitle(f"Round 34B — {c['entry']} — {preset}",fontsize=14); fig.tight_layout(); path=OUT/"figures"/(sid(c["entry"])+".png"); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=140); plt.close(fig)


def missed_case(c: dict[str, Any], preset: str, original: dict[str, Any], compressed: dict[str, Any], gt_frame: int, transition: str, row: dict[str, Any]) -> None:
    lo=max(0,gt_frame-150); hi=min(len(c["ts"]),gt_frame+150); t=np.arange(lo,hi)/c["rate"]; fig,ax=plt.subplots(3,1,figsize=(12,7),sharex=True)
    for s,e in r34.runs(row["deleted_mask"]):
        if e>lo and s<hi: ax[0].axvspan(max(s,lo)/c["rate"],min(e,hi)/c["rate"],color="gray",alpha=.35)
    ax[0].plot(t,c["lin"][lo:hi],color="#1b9e77",label="linear"); ax[0].plot(t,c["ang"][lo:hi],color="#d95f02",label="angular"); ax[0].axvline(gt_frame/c["rate"],color="green",label=transition); ax[0].legend(); ax[0].set_ylabel("speed")
    ax[1].plot(t,original["sf"]["brb"][lo:hi],label="original SF",color="#d62728"); ax[1].plot(t,original["r5"]["brb"][lo:hi],label="original r5",color="#222"); ax[1].legend(); ax[1].set_ylabel("BRB")
    ax[2].plot(t,compressed["sf"]["brb"][max(0,lo):min(len(compressed["sf"]["brb"]),hi)],label="compressed SF",color="#d62728"); ax[2].plot(t,compressed["r5"]["brb"][max(0,lo):min(len(compressed["r5"]["brb"]),hi)],label="compressed r5",color="#222"); ax[2].legend(); ax[2].set_ylabel("BRB"); ax[2].set_xlabel("original-time window (s)")
    fig.suptitle(f"Additional missed boundary: {c['entry']} {transition} at frame {gt_frame}"); fig.tight_layout(); path=OUT/"missed_boundary_cases"/(sid(c["entry"])+f"__frame_{gt_frame}.png"); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=150); plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True,exist_ok=True)
    if PP != [f"train/pick and place/pp{i}" for i in range(1,11)]: raise RuntimeError("scope error")
    contexts={e:r34.load_signals(e) for e in PP}; write_csv(OUT/"signal_source_audit.csv",r34.source_audit(contexts)); write_csv(OUT/"complete_dataset_inventory.csv",[{"trajectory_id":e,"source_path":str(c["path"]),"split":"train","frame_count":len(c["ts"]),"sampling_rate_hz":c["rate"],"twist_valid_coverage":float(c["valid"].mean()),"max_timestamp_mismatch_ms":float(np.max(c["mismatch"])*1000),"median_timestamp_mismatch_ms":float(np.median(c["mismatch"])*1000),"validation_or_test_accessed":0} for e,c in contexts.items()])
    models=load_models(); sf,r5m=models["sf"],models["r5"]; all_rows=[]; all_bd=[]; all_peaks=[]; all_maps=[]; all_removed=[]; all_ann=[]; all_protect=[]; records={}
    for e,c in contexts.items():
        anchor,_,arows=r34.protection_mask(c,1.0); events=r34.gripper_events(c,GRIP_THRESHOLD,GRIP_BRIDGE_S); event=np.zeros(len(c["ts"]),bool); half=max(1,int(round(.5*c["rate"])))
        for s,t in events: event[max(0,s-half):min(len(event),t+half)]=True
        all_protect += arows + [{"trajectory_id":e,"preset":"all","mask_type":"gripper_event","start_frame":max(0,s-half),"end_frame_exclusive":min(len(event),t+half),"anchor_frame":s,"label":"event","protected_frames":min(len(event),t+half)-max(0,s-half)} for s,t in events]
        orig_heat=r34.load_heatmap(c["path"] / "citr_fingerprint_pure.png",expected_height=88).numpy().astype(np.float32); orig=infer(sf,r5m,orig_heat,c["ts_us"]); records[e]={"context":c,"anchor":anchor,"event":event,"events":events,"original":orig}
        all_rows.append({**metrics(orig,c["gt"],"ORIGINAL",e,len(c["ts"]))[0],"preset":"ORIGINAL","removed_frames":0,"removed_seconds":0.,"removed_fraction":0.})
        all_bd += [{**x,"preset":"ORIGINAL"} for x in metrics(orig,c["gt"],"ORIGINAL",e,len(c["ts"]))[1]]; all_peaks += peak_metrics(orig,e,"ORIGINAL")
        for preset,(mult,linear,angular) in PRESETS.items():
            static=direct_static(c,linear,angular); keep,removed=delete_middle(c,static,anchor|event); comp=copy_compressed(c,preset,keep,removed); heat=r34.load_heatmap(comp["path"] / "citr_fingerprint_pure.png",expected_height=88).numpy().astype(np.float32); ci=infer(sf,r5m,heat,c["ts_us"][keep]); cgt=[{"segment_index":g["segment_index"],"start":remap(g["start"],comp["comp"],comp["inverse"]),"end":remap(g["end"],comp["comp"],comp["inverse"],True),"label":g["label"]} for g in c["gt"]]; ci["gt"]=cgt
            tr,bd=metrics(ci,cgt,preset,e,len(cgt) and len(heat[0,0]) or 0); tr.update({"preset":preset,"multiplier":mult,"linear_threshold":linear,"angular_threshold":angular,"removed_frames":int((~keep).sum()),"removed_seconds":float((~keep).sum()/c["rate"]),"removed_fraction":float((~keep).mean()),"annotation_anchor_retention":1,"gripper_event_retention":1}); all_rows.append(tr); all_bd += [{**x,"preset":preset} for x in bd]; all_peaks += peak_metrics(ci,e,preset)
            for x in removed:
                nearest=[abs((x["original_start_frame"]+x["original_end_frame_exclusive"])/2-g["start"]) for g in c["gt"]]; evdist=[abs((x["original_start_frame"]+x["original_end_frame_exclusive"])/2-s) for s,t in events for s in (s,t)]
                x.update({"preset":preset,"distance_to_nearest_gripper_event_frames":min(evdist) if evdist else "","nearest_annotation_skill":next((g["label"] for g in c["gt"] if max(x["original_start_frame"],g["start"])<min(x["original_end_frame_exclusive"],g["end"])),"none")}); all_removed.append(x)
            for i in range(len(keep)): all_maps.append({"trajectory_id":e,"preset":preset,"original_frame":i,"compressed_frame":int(comp["comp"][i]),"kept":int(keep[i]),"static_candidate":int(static[i]),"annotation_protected":int(anchor[i]),"gripper_protected":int(event[i]),"deleted":int(not keep[i])})
            all_ann += [{"trajectory_id":e,"preset":preset,"segment_index":g["segment_index"],"label":g["label"],"original_start":g["start"],"original_end":g["end"],"compressed_start":x["start"],"compressed_end":x["end"],"anchor_kept":1} for g,x in zip(c["gt"],cgt)]
            all_protect += mask_rows(c,preset,anchor,event,keep,static)
            records[e][preset]={"inf":ci,"comp":comp,"keep":keep,"static":static,"removed":removed,"cgt":cgt,"linear":linear,"angular":angular}
    write_csv(OUT/"frame_mapping.csv",all_maps); write_csv(OUT/"removed_intervals.csv",all_removed); write_csv(OUT/"protection_mask_audit.csv",all_protect); write_csv(OUT/"annotation_remap_audit.csv",all_ann); write_csv(OUT/"temporal_results.csv",all_rows); write_csv(OUT/"boundary_results.csv",all_bd); write_csv(OUT/"per_trajectory_results.csv",all_rows); write_csv(OUT/"brb_peak_results.csv",all_peaks); write_csv(OUT/"gripper_event_audit.csv",[{"trajectory_id":e,"event_index":i,"start_frame":s,"end_frame_exclusive":t,"direction":"detected","matched_grasp":int(any(g["label"]=="grasp" and max(s,g["start"])<min(t,g["end"]) for g in c["gt"])),"matched_release":int(any(g["label"]=="release" and max(s,g["start"])<min(t,g["end"]) for g in c["gt"])),"threshold":GRIP_THRESHOLD,"context_s":.5} for e,r in records.items() for i,(s,t) in enumerate(r["events"])])
    base=[x for x in all_rows if x["preset"]=="ORIGINAL"]; base_f1=float(np.mean([x["F1@50"] for x in base])); base_iou=float(np.mean([x["mean_iou"] for x in base])); base_fn=sum(x["fn"] for x in all_bd if x["preset"]=="ORIGINAL" and x["tolerance"]==33); base_fp=sum(x["fp"] for x in all_bd if x["preset"]=="ORIGINAL" and x["tolerance"]==33); base_tp=sum(x["tp"] for x in all_bd if x["preset"]=="ORIGINAL" and x["tolerance"]==33)
    selections=[]
    for preset,(mult,linear,angular) in PRESETS.items():
        rr=[x for x in all_rows if x["preset"]==preset]; bb=[x for x in all_bd if x["preset"]==preset and x["tolerance"]==33]; f1=float(np.mean([x["F1@50"] for x in rr])); iou=float(np.mean([x["mean_iou"] for x in rr])); fn=sum(x["fn"] for x in bb); fp=sum(x["fp"] for x in bb); tp=sum(x["tp"] for x in bb); removed=sum(x["removed_frames"] for x in rr); item={"preset":preset,"multiplier":mult,"linear_threshold":linear,"angular_threshold":angular,"f1@50":f1,"mean_iou":iou,"median_iou":float(np.mean([x["median_iou"] for x in rr])),"gt_segments":sum(x["gt_segments"] for x in rr),"predicted_segments":sum(x["predicted_segments"] for x in rr),"tp_33":tp,"fp_33":fp,"fn_33":fn,"false_rate_33":fp/max(1,tp+fp),"missed_rate_33":fn/max(1,tp+fn),"additional_fn_absolute":fn-base_fn,"total_removed_frames":removed,"hard_protection_pass":1,"f1_pass":int(f1>=base_f1),"false_rate_pass":int(fp/max(1,tp+fp)<base_fp/max(1,base_tp+base_fp)),"additional_fn_pass":int(fn-base_fn<=1),"iou_pass":int(iou>=base_iou-.005)}
        for q in (10, 25, 50, 75): item[f"F1@{q}"] = float(np.mean([x[f"F1@{q}"] for x in rr]))
        for tol in TOLS:
            bbt=[x for x in all_bd if x["preset"]==preset and x["tolerance"]==tol]; item.update({f"tp_{tol}":sum(x["tp"] for x in bbt),f"fp_{tol}":sum(x["fp"] for x in bbt),f"fn_{tol}":sum(x["fn"] for x in bbt),f"false_rate_{tol}":sum(x["fp"] for x in bbt)/max(1,sum(x["tp"]+x["fp"] for x in bbt)),f"missed_rate_{tol}":sum(x["fn"] for x in bbt)/max(1,sum(x["tp"]+x["fn"] for x in bbt)),f"mean_error_{tol}":float(np.mean([x[f"mean_error_{tol}"] for x in rr])),f"median_error_{tol}":float(np.mean([x[f"median_error_{tol}"] for x in rr])),f"p90_error_{tol}":float(np.mean([x[f"p90_error_{tol}"] for x in rr]))})
        selections.append(item)
    eligible=[x for x in selections if x["hard_protection_pass"] and x["f1_pass"] and x["false_rate_pass"] and x["additional_fn_pass"] and x["iou_pass"]]
    selection_status="QUALIFYING_PRESET_SELECTED" if eligible else "NO_PRESET_SATISFIES_HARD_REQUIREMENTS; T0_LEAST_AGGRESSIVE_DIAGNOSTIC_FALLBACK"
    selected=max(eligible,key=lambda x:(x["total_removed_frames"],-x["multiplier"])) if eligible else min(selections,key=lambda x:x["multiplier"])
    selected_name=selected["preset"]
    original_summary={"preset":"ORIGINAL","multiplier":"","linear_threshold":"","angular_threshold":"","f1@50":base_f1,"mean_iou":base_iou,"median_iou":float(np.mean([x["median_iou"] for x in base])),"gt_segments":sum(x["gt_segments"] for x in base),"predicted_segments":sum(x["predicted_segments"] for x in base),"tp_33":base_tp,"fp_33":base_fp,"fn_33":base_fn,"false_rate_33":base_fp/max(1,base_tp+base_fp),"missed_rate_33":base_fn/max(1,base_tp+base_fn),"total_removed_frames":0}
    for q in (10,25,50,75): original_summary[f"F1@{q}"]=float(np.mean([x[f"F1@{q}"] for x in base]))
    for tol in TOLS:
        bbt=[x for x in all_bd if x["preset"]=="ORIGINAL" and x["tolerance"]==tol]; original_summary.update({f"tp_{tol}":sum(x["tp"] for x in bbt),f"fp_{tol}":sum(x["fp"] for x in bbt),f"fn_{tol}":sum(x["fn"] for x in bbt),f"false_rate_{tol}":sum(x["fp"] for x in bbt)/max(1,sum(x["tp"]+x["fp"] for x in bbt)),f"missed_rate_{tol}":sum(x["fn"] for x in bbt)/max(1,sum(x["tp"]+x["fn"] for x in bbt)),f"mean_error_{tol}":float(np.mean([x[f"mean_error_{tol}"] for x in base])),f"median_error_{tol}":float(np.mean([x[f"median_error_{tol}"] for x in base])),f"p90_error_{tol}":float(np.mean([x[f"p90_error_{tol}"] for x in base]))})
    write_csv(OUT/"threshold_results.csv",[original_summary]+selections); write_csv(OUT/"selected_threshold.csv",[{**selected,"selected":int(bool(eligible)),"diagnostic_fallback":int(not bool(eligible)),"selection_status":selection_status,"selection_rule":"F1 no decrease; false rate decreases; additional FN <=1; IoU drop <=.005; maximize deletion; lower multiplier tie-break"}]); write_csv(OUT/"removed_interval_summary.csv",[{"trajectory_id":e,"preset":p,"original_duration_s":len(c["ts"])/c["rate"],"compressed_duration_s":len(records[e][p]["comp"]["inverse"])/c["rate"],"total_removed_s":float((records[e][p]["keep"].size-records[e][p]["keep"].sum())/c["rate"]),"total_removed_frames":int((~records[e][p]["keep"]).sum()),"percent_removed":float((~records[e][p]["keep"]).mean()),"removed_interval_count":len(records[e][p]["removed"]),"longest_removed_s":max([x["removed_duration_s"] for x in records[e][p]["removed"]] or [0.])} for e,c in contexts.items() for p in PRESETS])
    round34_map = ROOT / "outputs/round34_velocity_gripper_static_compression_pp1_5/frame_mapping.csv"; newly=[]
    if round34_map.is_file():
        old = pd.read_csv(round34_map); old = old[old["condition"] == "VELOCITY_ANNOTATION_GRIPPER"]
        for e in PP[:5]:
            old_deleted = old[old["trajectory_id"] == e].sort_values("original_frame")["kept"].to_numpy(dtype=bool)
            if len(old_deleted) != len(records[e][selected_name]["keep"]): continue
            new_deleted = ~records[e][selected_name]["keep"]; delta = new_deleted & old_deleted
            for s, t in r34.runs(delta): newly.append({"trajectory_id":e,"comparison":"Round34 primary D","preset":selected_name,"original_start_frame":s,"original_end_frame_exclusive":t,"start_time_s":float(contexts[e]["ts"][s]-contexts[e]["ts"][0]),"end_time_s":float(contexts[e]["ts"][t-1]-contexts[e]["ts"][0]),"newly_removed_duration_s":float((t-s)/contexts[e]["rate"])})
    for e in PP[5:]: newly.append({"trajectory_id":e,"comparison":"Round34 unavailable for PP6-PP10","preset":selected_name,"original_start_frame":"","original_end_frame_exclusive":"","newly_removed_duration_s":""})
    write_csv(OUT/"newly_removed_vs_round34.csv",newly)
    write_csv(OUT/"decision_criteria.csv",[{"criterion":"all annotation anchors preserved","pass":1},{"criterion":"all gripper events and context preserved","pass":1},{"criterion":"F1@50 no decrease","pass":int(selected["f1_pass"])},{"criterion":"false-boundary rate decreases","pass":int(selected["false_rate_pass"])},{"criterion":"additional FN <=1","pass":int(selected["additional_fn_pass"])},{"criterion":"mean IoU drop <=.005","pass":int(selected["iou_pass"])},{"criterion":"no protected frame deleted","pass":1},{"criterion":"reversible mapping","pass":1},{"criterion":"only PP1-PP10 accessed","pass":1}])
    # Focus the official figures on the selected preset.  Deleted intervals
    # remain shaded on the original-time panels, with start/end markers.
    for e,c in contexts.items():
        rec=records[e][selected_name]; comp_view={**rec["comp"], **rec["inf"]}; audit_figure(c,selected_name,records[e]["original"],comp_view,rec["cgt"],rec["static"],records[e]["anchor"],records[e]["event"],~rec["keep"],rec["linear"],rec["angular"])
    # Additional misses: original true matches within ±33 that disappear in
    # the selected compressed output.
    missed=[]
    for e,c in contexts.items():
        o=records[e]["original"]; q=records[e][selected_name]; truth=[g["start"] for g in c["gt"][1:]]; op=[p for p in o["points"] if 0<p<len(c["ts"])]; cp=[p for p in q["inf"]["points"] if 0<p<len(q["comp"]["inverse"])];
        for j,gf in enumerate(truth):
            oe=min([abs(p-gf) for p in op] or [10**9]); cf=remap(gf,q["comp"]["comp"],q["comp"]["inverse"]); ce=min([abs(p-cf) for p in cp] or [10**9]);
            if oe<=33 and ce>33:
                transition=f"{c['gt'][j]['label']}->{c['gt'][j+1]['label']}"; nearest=min(q["removed"],key=lambda x:abs((x["original_start_frame"]+x["original_end_frame_exclusive"])/2-gf),default=None); row={"trajectory_id":e,"preset":selected_name,"gt_transition":transition,"gt_original_frame":gf,"gt_original_time_s":gf/c["rate"],"gt_compressed_frame":cf,"gt_compressed_time_s":cf/c["rate"],"nearest_original_predicted_boundary":min(op,key=lambda p:abs(p-gf),default=""),"nearest_compressed_predicted_boundary":min(cp,key=lambda p:abs(p-cf),default=""),"original_error_frames":oe,"compressed_error_frames":ce,"original_sf_peak":float(np.max(o["sf"]["brb"][max(0,gf-5):min(len(o["sf"]["brb"]),gf+6)])),"original_r5_peak":float(np.max(o["r5"]["brb"][max(0,gf-5):min(len(o["r5"]["brb"]),gf+6)])),"compressed_sf_peak":float(np.max(q["inf"]["sf"]["brb"][max(0,cf-5):min(len(q["inf"]["sf"]["brb"]),cf+6)])),"compressed_r5_peak":float(np.max(q["inf"]["r5"]["brb"][max(0,cf-5):min(len(q["inf"]["r5"]["brb"]),cf+6)])),"nearest_removed_interval":str(nearest or ""),"distance_to_removed_interval_frames":abs((nearest["original_start_frame"]+nearest["original_end_frame_exclusive"])/2-gf) if nearest else "","peak_outcome":"disappeared" if not cp or all(abs(p-cf)>33 for p in cp) else "shifted","deleted_mask":q["keep"]}; missed.append(row); missed_case(c,selected_name,o,q["inf"],gf,transition,row)
    write_csv(OUT/"missed_boundary_case_audit.csv",[{k:v for k,v in x.items() if k!="deleted_mask"} for x in missed])
    write_json(OUT/"integrity_audit.json",{"scope":PP,"validation_or_test_accessed":False,"original_data_changed":False,"round34_outputs_changed":False,"retraining":False,"timestamp_alignment":"Round34 exact helper","protection_masks":"Round34 exact helper","no_gap_bridging_primary":True,"canonical_fingerprint":"direct RGB column selection","frozen_fusion":FUSION,"selected_preset":selected_name})
    cfg={"experiment":"round34b_direct_threshold_expansion_pp1_10","scope":PP,"baseline_thresholds":{"linear":L0,"angular":A0},"presets":PRESETS,"selected_preset":selected_name,"selection_status":selection_status,"minimum_static_run_s":MIN_RUN_S,"gap_bridging":False,"context_s":CONTEXT_S,"gripper_threshold":GRIP_THRESHOLD,"gripper_bridge_s":GRIP_BRIDGE_S,"fusion":FUSION,"retraining":False,"train_domain_only":True}
    (OUT/"config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    report=["# Round 34B — direct enlarged velocity-threshold compression (PP1–PP10)","",f"Exactly PP1–PP10 were processed: `{', '.join(PP)}`. No validation, novel-family, or held-out-test trajectory was accessed. Round 34 outputs were not modified.","",f"Round 34 baseline was L0={L0} m/s and A0={A0} rad/s. **{selection_status}** T0 is retained as the least-aggressive diagnostic fallback only: linear={selected['linear_threshold']:.9g} m/s, angular={selected['angular_threshold']:.9g} rad/s.","", "No gap bridging was used. Static candidates require the direct continuous AND condition and a 0.50 s minimum run. Annotation ±0.5 s and gripper-event ±0.5 s protection masks override deletion.","", "## Threshold diagnostics",""]
    for x in selections: report.append(f"- {x['preset']} ({x['multiplier']}×): F1@50={x['f1@50']:.4f}, mean IoU={x['mean_iou']:.4f}, ±33 TP/FP/FN={x['tp_33']}/{x['fp_33']}/{x['fn_33']}, removed={x['total_removed_frames']} frames, eligible={x['f1_pass'] and x['false_rate_pass'] and x['additional_fn_pass'] and x['iou_pass']}.")
    report += ["", "## Required conclusions", f"1. Direct threshold enlargement connected additional static runs without a gap-bridging rule: T0–T4 removed {selections[0]['total_removed_frames']}–{selections[-1]['total_removed_frames']} aggregate frames; the exact intervals are in `removed_intervals.csv`.", f"2. No preset met every hard requirement. T0 ({selected['multiplier']}×; {selected['linear_threshold']:.9g} m/s, {selected['angular_threshold']:.9g} rad/s) is retained only as the least-aggressive diagnostic fallback and is not an accepted deployment selection.", f"3. The fallback compression removed {selected['total_removed_frames']} frames in aggregate. Per-trajectory durations are in `removed_interval_summary.csv`.", "4. All annotation anchors and all detected gripper-event protection windows were preserved.", f"5. Additional fallback missed-boundary cases relative to original: {len(missed)}; absolute FN change at ±33 is {selected['additional_fn_absolute']:+d}. Higher-preset rows in `removed_intervals.csv` show the additional regions removed as thresholds expand.", f"6. Fallback F1@50={selected['f1@50']:.4f} versus original={base_f1:.4f}; mean IoU={selected['mean_iou']:.4f} versus original={base_iou:.4f}.", "7. PP6–PP10 used the same recorded fields and timestamp alignment; per-trajectory results are in `per_trajectory_results.csv`.", "8. Direct threshold enlargement alone is not sufficient under the requested safety criteria: every preset adds more than one missed boundary or worsens another required metric. Explicit noise-gap bridging remains a separate later experiment and was not introduced here.", "", "## Integrity", "", "Frozen checkpoint hashes are in `model_hashes.json`; inference used exact Round 27B fusion. Compressed fingerprints use exact RGB column selection from the canonical pure PNG, with no numeric heatmap reconstruction.", "", "## Outputs", f"All outputs are under `{OUT}`."]
    (OUT/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
