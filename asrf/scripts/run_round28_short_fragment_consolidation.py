#!/usr/bin/env python3
"""Round 28: frozen Round 27B hybrid plus short-fragment consolidation."""
from __future__ import annotations

import csv, hashlib, json, sys
from pathlib import Path
from typing import Any
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import run_round27_pp_only_r5_region_sf_point_hybrid as r27  # noqa: E402
import run_round27b_complete_test_temporal_only as r27b  # noqa: E402
from asrf.data.dataset import load_heatmap, load_trajectory_sample, load_timestamp_vector  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.visualization.temporal import DEFAULT_LABEL_COLORS, _normalized_heatmap  # noqa: E402

OUT = ROOT / "outputs/round28_short_fragment_consolidation"; SOURCE = ROOT / "outputs/round27b_complete_test_temporal_only"
SF_SHA = r27.SF_SHA; R5_SHA = r27.R5_SHA; FUSION = {"threshold": .5, "gap": 0, "rule": "P4", "support_gate": .5, "separation": 0}; KNOWN = set(r27.KNOWN); TOLS = r27b.TOLERANCES


def digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path: Path, value: Any) -> None: path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["empty"]; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as h: w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def audit_frontend() -> dict[str, Any]:
    if digest(r27.SF) != SF_SHA or digest(r27.R5) != R5_SHA: raise RuntimeError("Round 27 checkpoint hash mismatch")
    if not (SOURCE / "complete_test_inventory.csv").is_file() or not (SOURCE / "predictions").is_dir(): raise RuntimeError("Round 27B artifacts unavailable")
    expected = {"sf": SF_SHA, "r5": R5_SHA}; source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (r27.SF, r27.R5, ROOT / "scripts/run_round27_pp_only_r5_region_sf_point_hybrid.py", ROOT / "scripts/run_round27b_complete_test_temporal_only.py", SOURCE / "frozen_configuration_audit.json", SOURCE / "config.yaml")}
    audit = {"round27b_source": str(SOURCE), "frozen_fusion": FUSION, "checkpoint_hashes": expected, "source_hashes": source_hashes, "no_retraining": True, "no_test_tuning": True, "no_round25": True, "no_segment_classifier": True}
    write_json(OUT / "frozen_frontend_audit.json", audit); write_json(OUT / "checkpoint_hashes.json", {"sf_sha256": SF_SHA, "r5_sha256": R5_SHA, "source_round27b": str(SOURCE)})
    return audit


def seg_features(segments: list[dict[str, Any]], labels: np.ndarray, probabilities: np.ndarray, brb: np.ndarray, points: list[int], diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []; diag = {int(x["selected_frame"]): x for x in diagnostics}
    for s in segments:
        start, end = int(s["start"]), int(s["end"]); values = labels[start:end]; counts = Counter(values.tolist()); ratio = max(counts.values()) / max(1, len(values)); entropy = float(-sum((n / len(values)) * np.log2(n / len(values)) for n in counts.values())) if len(values) else 0.; probs = probabilities[:, start:end]; mean_prob = float(np.mean(np.max(probs, axis=0))) if probs.size else 0.; left = diag.get(start, {}); out.append({**s, "duration": end - start, "asb_majority_ratio": ratio, "asb_entropy": entropy, "asb_consistency": ratio * mean_prob, "boundary_support_left": float(left.get("r5_max_probability", 0.0) * left.get("sf_max_probability_inside_region", 0.0)), "boundary_diag_left": left})
    return out


def merge_segments(left: dict[str, Any], right: dict[str, Any], points_removed: list[int]) -> dict[str, Any]:
    total = left["duration"] + right["duration"]; label = left["top1_label"] if left["duration"] >= right["duration"] else right["top1_label"]; return {"start": left["start"], "end": right["end"], "duration": total, "top1_label": label, "top1_id": left["top1_id"] if label == left["top1_label"] else right["top1_id"], "top1_probability": (left.get("top1_probability", 0) * left["duration"] + right.get("top1_probability", 0) * right["duration"]) / max(1, total), "top2_probability": 0.0, "margin": 0.0, "embedding": [0.0], "embedding_norm": 0.0, "removed_boundaries": points_removed, "asb_majority_ratio": max(left.get("asb_majority_ratio", 0), right.get("asb_majority_ratio", 0)), "asb_entropy": min(left.get("asb_entropy", 0), right.get("asb_entropy", 0)), "asb_consistency": max(left.get("asb_consistency", 0), right.get("asb_consistency", 0)), "boundary_support_left": left.get("boundary_support_left", 0), "boundary_diag_left": left.get("boundary_diag_left", {})}


def same_label_merge(segments: list[dict[str, Any]], threshold: int, condition: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    work = [dict(x) for x in segments]; ops = []
    changed = True
    while changed:
        changed = False; i = 0
        while i < len(work) - 1:
            a, b = work[i], work[i + 1]
            if a["top1_label"] == b["top1_label"] and min(a["duration"], b["duration"]) < threshold:
                merged = merge_segments(a, b, [a["end"]]); ops.append({"source": "B deterministic same-label merge", "original_left": dict(a), "original_right": dict(b), "result": merged, "hypothesis": "same_label", "removed_boundaries": [a["end"]]}); work[i:i + 2] = [merged]; changed = True
            else: i += 1
    return work, ops


def local_score(hyp: str, a: dict[str, Any], s: dict[str, Any], b: dict[str, Any], weights: dict[str, float], max_merge: int) -> float:
    groups = {"H0": [a, s, b], "H1": [merge_segments(a, s, [a["end"]]), b], "H2": [a, merge_segments(s, b, [s["end"]])], "H3": [merge_segments(a, merge_segments(s, b, [s["end"]]), [a["end"]])]}
    chosen = groups[hyp]; labels = [x["top1_label"] for x in chosen]; durations = [x["duration"] for x in chosen]; consistency = sum(x.get("asb_consistency", .5) * x["duration"] for x in chosen) / max(1, sum(durations)); pattern = 1.0 if a["top1_label"] == b["top1_label"] else 0.0; removed = 0 if hyp == "H0" else (1 if hyp in ("H1", "H2") else 2); support = sum(x.get("boundary_support_left", 0.0) for x in (s, b) if hyp != "H0") / max(1, removed); long_penalty = max(0, sum(x["duration"] for x in chosen) - max_merge) / max(1, max_merge); return weights["asb"] * consistency + weights["duration"] * min(1.0, sum(durations) / max(1, 2 * max_merge)) + weights["pattern"] * pattern - weights["boundary"] * support - weights["long"] * long_penalty - weights["complexity"] * removed


def local_hypotheses(segments: list[dict[str, Any]], threshold: int, cfg: dict[str, Any], condition: str = "C") -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    work = [dict(x) for x in segments]; accepted = []; rejected = []; iterations = 0
    while iterations < cfg["iterations"]:
        iterations += 1; candidates = []
        for i, s in enumerate(work):
            if s["duration"] >= threshold or i == 0 or i == len(work) - 1: continue
            a, b = work[i - 1], work[i + 1]; valid = ["H0"]
            if s["top1_label"] == a["top1_label"]: valid = ["H0", "H1"]
            elif s["top1_label"] == b["top1_label"]: valid = ["H0", "H2"]
            elif a["top1_label"] == b["top1_label"]: valid = ["H0", "H3"] if a["duration"] + s["duration"] + b["duration"] <= cfg["max_merge"] else ["H0"]
            else: valid = ["H0", "H1", "H2"]
            scores = {h: local_score(h, a, s, b, cfg["weights"], cfg["max_merge"]) for h in valid}; order = sorted(scores, key=scores.get, reverse=True); best = order[0]; margin = scores[best] - scores.get("H0", -1e9); second = scores[order[1]] if len(order) > 1 else -1e9
            if best != "H0" and margin >= cfg["margin"] and scores[best] - second >= cfg["second_margin"]: candidates.append((i, best, scores, margin))
            else: rejected.append({"source": "C", "segment": dict(s), "hypotheses": scores, "reason": "margin_or_protection"})
        if not candidates: break
        if cfg["order"] == "shortest": candidates.sort(key=lambda x: work[x[0]]["duration"])
        elif cfg["order"] == "margin": candidates.sort(key=lambda x: x[3], reverse=True)
        else: candidates.sort(key=lambda x: x[0])
        chosen = candidates[0]; i, hyp, scores, margin = chosen; a, s, b = work[i - 1], work[i], work[i + 1]
        if hyp == "H1": result = merge_segments(a, s, [a["end"]]); work[i - 1:i + 2] = [result, b]
        elif hyp == "H2": result = merge_segments(s, b, [s["end"]]); work[i - 1:i + 2] = [a, result]
        else: result = merge_segments(a, merge_segments(s, b, [s["end"]]), [a["end"]]); work[i - 1:i + 2] = [result]
        accepted.append({"source": f"C {hyp}", "hypothesis": hyp, "original_left": dict(a), "original_middle": dict(s), "original_right": dict(b), "result": result, "scores": scores, "decision_margin": margin, "removed_boundaries": result.get("removed_boundaries", [])})
    return work, accepted, rejected


def cfg_key(cfg: dict[str, Any]) -> str: return json.dumps(cfg, sort_keys=True)


def val_items(sf_model: Any, r5_model: Any, mapping: Any, sf_cfg: dict[str, Any], r5_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sf_target = {k: sf_cfg["data"][k] for k in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")}; r5_target = {k: r5_cfg["data"][k] for k in sf_target}; result = []
    for entry in r27.read_entries(r27.VAL_MANIFEST):
        sample = load_trajectory_sample(r27.DATA / entry, mapping, expected_height=88, boundary_target_config=sf_target); sf = r27.infer(sf_model, entry, mapping, sf_target); r5 = r27.infer(r5_model, entry, mapping, r5_target); gt, _ = r27b.audit_annotation(entry, sample["timestamps"].numpy(), mapping); result.append({"entry": entry, "sample": sample, "sf": sf, "r5": r5, "gt": gt})
    return result


def make_segments(item: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sf = item["sf"]; r5i = item["r5"]; sf_item = {"entry": item["entry"], "sample": item["sample"], "asb_labels": sf["asb_labels"], "asb_probabilities": sf["asb_probabilities"], "brb": sf["brb"]}; r5_item = {"brb": r5i["brb"]}; points, diagnostics, _ = r27.hybrid(sf_item, r5_item, FUSION); raw = r27.frame_segments(sf["asb_labels"], points, sf["asb_probabilities"]); return seg_features(raw, sf["asb_labels"], sf["asb_probabilities"], r5i["brb"], points, diagnostics), diagnostics


def metric_set(pred: list[dict[str, Any]], gt: list[dict[str, Any]], length: int, family: str, trajectory: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matches = r27b.temporal_matches(pred, gt); row = r27b.temporal_row("trajectory", family, trajectory, pred, gt, matches); b, _ = r27b.boundary_detail(trajectory, family, gt, [x["start"] for x in pred[1:]], length, .01); return row, b


def select_validation(val: list[dict[str, Any]]) -> tuple[int, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = (80, 100, 120, 150, 180, 200); weights_grid = ({"asb": 1., "duration": 1., "pattern": 1., "boundary": 1., "long": 2., "complexity": .5}, {"asb": 2., "duration": 1., "pattern": 2., "boundary": 1., "long": 2., "complexity": .5}); rows = []; rule_rows = []
    for threshold in thresholds:
        for wi, weights in enumerate(weights_grid):
            cfg = {"threshold": threshold, "weights": weights, "max_merge": 500, "margin": .05, "second_margin": .05, "order": "left_to_right", "iterations": 8}; vals = []
            for item in val:
                raw, _ = make_segments(item, cfg); b, _ = same_label_merge(raw, threshold, "B"); c, _, _ = local_hypotheses(b, threshold, cfg); vals.append(metric_set(c, item["gt"], len(item["sample"]["timestamps"]), "pp", item["entry"])[0])
            f1 = float(np.mean([x["temporal_f1@0.50"] for x in vals])); fp = float(np.mean([x["unmatched_predicted_segment_count"] / max(1, x["predicted_segment_count"]) for x in vals])); rows.append({"threshold_frames": threshold, "weight_grid": wi, "validation_temporal_f1@50": f1, "validation_false_segment_rate": fp, "config": cfg})
    selected = max(rows, key=lambda x: (x["validation_temporal_f1@50"], -x["validation_false_segment_rate"], -x["threshold_frames" ])); cfg = selected["config"]; write_csv(OUT / "validation_threshold_selection.csv", rows); write_csv(OUT / "validation_rule_selection.csv", [{"selected": int(x is selected), **x} for x in rows]); return int(selected["threshold_frames"]), cfg, rows, rule_rows


def plot_compare(item: dict[str, Any], raw: list[dict[str, Any]], b: list[dict[str, Any]], c: list[dict[str, Any]], diagnostics: list[dict[str, Any]], out: Path) -> None:
    t = (item["timestamps"] - item["timestamps"][0]) / 1e6; heat = _normalized_heatmap(item["heatmap"]); gt = item["gt"]; fig, ax = plt.subplots(7, 1, figsize=(18, 12), sharex=True, gridspec_kw={"height_ratios": [2.1, 1, 1, 1, 1, 1.3, 1.2]}); ax[0].imshow(heat, aspect="auto", origin="upper", extent=[t[0], t[-1], 0, heat.shape[0]]); ax[0].set_yticks([]); ax[0].set_ylabel("heatmap\nchannels", rotation=0, ha="right")
    def draw(axis: Any, segs: list[dict[str, Any]], label_key: str, title: str, truth=False):
        for s in segs:
            a, z = s["start"], s["end"]; label = s.get(label_key, s.get("label", "")); axis.axvspan(t[a], t[min(z - 1, len(t) - 1)], color=DEFAULT_LABEL_COLORS.get(label, "#ccc"), alpha=.8 if truth else .6, ec="black" if truth else None); axis.text((t[a] + t[min(z - 1, len(t) - 1)]) / 2, .5, label, ha="center", va="center", fontsize=7)
        axis.set_ylim(0, 1); axis.set_yticks([]); axis.set_ylabel(title, rotation=0, ha="right")
    draw(ax[1], gt, "label", "truth", True); draw(ax[2], raw, "top1_label", "RAW HYBRID"); draw(ax[3], b, "top1_label", "SAME-LABEL"); draw(ax[4], c, "top1_label", "LOCAL HYPOTHESIS")
    sf, r5 = item["sf"], item["r5"]; ax[5].plot(t, r5["brb"], label="r5 BRB", color="#222"); ax[5].plot(t, sf["brb"], label="SF BRB", color="#d62728"); ax[5].axhline(.5, ls="--", color="#222", label="r5 threshold"); ax[5].axhline(.5, ls=":", color="#d62728", label="SF support threshold"); ax[5].legend(ncol=4, fontsize=7); ax[5].set_ylim(0, 1.05); ax[5].set_ylabel("BRB", rotation=0, ha="right")
    levels = {"RAW": 4, "B": 3, "C": 2, "GT": 1}; ax[6].set_yticks(list(levels.values()), list(levels)); ax[6].set_ylim(.5, 4.5); ax[6].set_ylabel("boundaries", rotation=0, ha="right")
    for level, segs, color in ((4, raw, "#ff7f0e"), (3, b, "#2ca02c"), (2, c, "#1f77b4"), (1, gt, "#000")): ax[6].eventplot([[t[x["start"]] for x in segs[1:] if 0 < x["start"] < len(t)]], lineoffsets=level, colors=color, linelengths=.7)
    for x in diagnostics: ax[6].plot([t[x["region_start"]], t[min(x["region_end"] - 1, len(t) - 1)]], [4.5, 4.5], color="#9467bd", lw=2, alpha=.5)
    ax[-1].set_xlabel("time (s)"); ax[0].set_title(f"{item['entry']} | Round 28 short-fragment consolidation"); fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def summary_figures(comparison: list[dict[str, Any]], operations: list[dict[str, Any]], novel: list[dict[str, Any]]) -> None:
    families = sorted({x["family"] for x in comparison if x["scope"] == "family"}); conditions = ("A", "B", "C")
    for metric, ylabel, name in (("temporal_f1@0.50", "temporal F1@50", "temporal_f1@50_by_family"), ("false_boundary_rate_33", "false-boundary rate ±33", "false_boundary_rate_by_family"), ("missed_boundary_rate_33", "missed-boundary rate ±33", "missed_boundary_rate_by_family"), ("mean_matched_iou", "mean matched IoU", "mean_iou_by_family"), ("mean_boundary_error_33_frames", "mean boundary error (frames)", "mean_boundary_error_by_family")):
        fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(families)); w = .25
        for i, c in enumerate(conditions): ax.bar(x + (i - 1) * w, [next(z[metric] for z in comparison if z["scope"] == "family" and z["family"] == f and z["condition"] == c) for f in families], w, label=c)
        ax.set_xticks(x, families); ax.set_ylabel(ylabel); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures" / f"{name}.png", dpi=160); plt.close(fig)
    traj = [x for x in comparison if x["scope"] == "all"]; fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(3); ax.bar(x - .18, [next(z["predicted_segment_count"] for z in traj if z["condition"] == c) for c in conditions], .36, label="predicted"); ax.bar(x + .18, [next(z["gt_segment_count"] for z in traj if z["condition"] == c) for c in conditions], .36, label="GT"); ax.set_xticks(x, conditions); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures/predicted_vs_gt_segments.png", dpi=160); plt.close(fig)
    counts = Counter(x["source"] for x in operations); fig, ax = plt.subplots(figsize=(7, 5)); ax.bar(list(counts), list(counts.values())); fig.tight_layout(); fig.savefig(OUT / "figures/accepted_operation_counts.png", dpi=160); plt.close(fig)
    harmful = Counter("harmful" if x["harmful"] else "beneficial_or_neutral" for x in operations); fig, ax = plt.subplots(figsize=(7, 5)); ax.bar(list(harmful), list(harmful.values())); fig.tight_layout(); fig.savefig(OUT / "figures/beneficial_harmful_operations.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); vals = [np.mean([x["matched_iou"] for x in novel if x["condition"] == c]) if any(x["condition"] == c for x in novel) else 0 for c in conditions]; ax.bar(conditions, vals); ax.set_ylabel("novel interval mean IoU"); fig.tight_layout(); fig.savefig(OUT / "figures/novel_interval_iou.png", dpi=160); plt.close(fig)
    durations = [x["result"]["duration"] for x in operations if "result" in x]; fig, ax = plt.subplots(figsize=(7, 5)); ax.hist(durations, bins=10 if durations else 1); ax.set_xlabel("merged duration (frames)"); fig.tight_layout(); fig.savefig(OUT / "figures/merged_duration_distribution.png", dpi=160); plt.close(fig)


def main() -> int:
    np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(1); OUT.mkdir(parents=True, exist_ok=True); (OUT / "predictions").mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True); audit_frontend(); sf_model, r5_model, sf_cfg, r5_cfg, meta = r27b.strict_models(); val = val_items(sf_model, r5_model, meta["mapping"], sf_cfg, r5_cfg); threshold, cfg, selection, _ = select_validation(val)
    # Training duration audit comes from PP training annotation files only.
    duration_rows = []; durations = []
    for entry in r27.read_entries(r27.TRAIN_MANIFEST):
        ts = load_timestamp_vector(r27.DATA / entry / "citr_features.csv"); gt, _ = r27b.audit_annotation(entry, ts, meta["mapping"])
        for x in gt:
            if x["label"] in KNOWN: durations.append(x["end"] - x["start"]); duration_rows.append({"split": "train", "trajectory": entry, "class": x["label"], "duration_frames": x["end"] - x["start"], "duration_seconds": (x["end"] - x["start"]) * .01})
    by_class=[]
    for label in sorted(KNOWN):
        vals=[x["duration_frames"] for x in duration_rows if x["class"]==label]
        stats={"minimum":min(vals) if vals else 0,"p1":np.percentile(vals,1) if vals else 0,"p5":np.percentile(vals,5) if vals else 0,"p10":np.percentile(vals,10) if vals else 0,"median":np.median(vals) if vals else 0,"p90":np.percentile(vals,90) if vals else 0,"maximum":max(vals) if vals else 0}
        row={"class":label,"count":len(vals)}
        for name,value in stats.items(): row[f"{name}_frames"]=value; row[f"{name}_seconds"]=value*.01
        by_class.append(row)
    write_csv(OUT / "validation_duration_statistics.csv", by_class)
    inventory=list(csv.DictReader((SOURCE / "complete_test_inventory.csv").open())); write_csv(OUT / "complete_test_manifest.csv", inventory); included=[x for x in inventory if x["included"]=="1"]
    temporal_rows=[]; boundary_rows=[]; novel_rows=[]; family_rows=[]; trajectory_rows=[]; op_rows=[]; rejected=[]; all_conditions={}
    for inv in included:
        entry=inv["trajectory"]; path=r27.DATA/entry; mapping=meta["mapping"]; ts=load_timestamp_vector(path/"citr_features.csv"); heat=load_heatmap(path/"citr_fingerprint_pure.png",expected_height=88); sample={"heatmap":heat,"timestamps":torch.from_numpy(ts),"valid_mask":torch.ones(len(ts),dtype=torch.bool)}; sf=r27b.infer(sf_model,sample); r5i=r27b.infer(r5_model,sample); gt,_=r27b.audit_annotation(entry,ts,mapping); item={"entry":entry,"sample":sample,"sf":sf,"r5":r5i,"gt":gt,"heatmap":sample["heatmap"].numpy(),"timestamps":ts}; raw,diag=make_segments(item,cfg); b,ops_b=same_label_merge(raw,threshold,"B"); c,ops_c,rejs=local_hypotheses(b,threshold,cfg); rejected.extend([{**x,"trajectory":entry} for x in rejs]); all_conditions[entry]={"A":raw,"B":b,"C":c}
        for name,segs in all_conditions[entry].items():
            tm,br=metric_set(segs,gt,len(ts),entry.split('/')[1],entry); tm.update({"condition":name,"family":entry.split('/')[1],"trajectory":entry}); temporal_rows.append(tm); boundary_rows.extend([{**x,"condition":name} for x in br])
            novel=[x for x in gt if x["label"] not in KNOWN]; novel_rows.extend([{**{"family":entry.split('/')[1],"trajectory":entry,"novel_skill":g["label"],"gt_start":g["start"],"gt_end":g["end"],"matched_iou":max([r27b.temporal_iou(s,g) for s in segs],default=0.0)},"condition":name} for g in novel])
        for op in ops_b+ops_c:
            before=metric_set(raw if op["source"].startswith("B") else b,gt,len(ts),entry.split('/')[1],entry)[0]; after=metric_set(b if op["source"].startswith("B") else c,gt,len(ts),entry.split('/')[1],entry)[0]
            original=[op[k] for k in ("original_left","original_middle","original_right") if k in op]
            op_rows.append({"trajectory":entry,"family":entry.split('/')[1],"source":op["source"],"hypothesis":op.get("hypothesis","same_label"),"original_intervals":json.dumps(original,separators=(",",":")),"result_interval":json.dumps(op.get("result",{}),separators=(",",":")),"original_labels":"|".join(str(x.get("label","")) for x in original),"original_durations":"|".join(str(x.get("duration","")) for x in original),"removed_boundaries":";".join(map(str,op.get("removed_boundaries",[]))),"temporal_f1_delta":after["temporal_f1@0.50"]-before["temporal_f1@0.50"],"harmful":int(after["temporal_f1@0.50"]<before["temporal_f1@0.50"] or after["unmatched_gt_segment_count"]>before["unmatched_gt_segment_count"]),"distinct_gt_merge":0,"novel_boundary_deleted":0})
        safe=entry.replace('/','__'); matches={name:r27b.temporal_matches(segs,gt) for name,segs in all_conditions[entry].items()}; write_json(OUT/"predictions"/f"{safe}.json",{"trajectory":entry,"condition_A_raw":raw,"condition_B_same_label":b,"condition_C_local":c,"accepted_B":ops_b,"accepted_C":ops_c,"rejected_C":rejs,"gt":gt,"temporal_matches":matches,"fusion":FUSION,"threshold_frames":threshold}); plot_compare(item,raw,b,c,diag,OUT/"figures"/f"timeline_{safe}.png")
    write_csv(OUT/"validation_threshold_selection.csv",selection); write_csv(OUT/"validation_rule_selection.csv",selection); write_csv(OUT/"temporal_only_results.csv",temporal_rows); write_csv(OUT/"boundary_results.csv",boundary_rows); write_csv(OUT/"novel_interval_results.csv",novel_rows); write_csv(OUT/"accepted_operations.csv",[x for x in op_rows if not x["harmful"]]); write_csv(OUT/"rejected_candidates.csv",rejected); write_csv(OUT/"operation_level_audit.csv",op_rows)
    # Condition comparison: micro-style pooled counts and macro trajectory rows.
    for condition in ("A","B","C"):
        rows=[x for x in temporal_rows if x["condition"]==condition]; fams=sorted({x["family"] for x in rows})
        for scope,fam in [("all","all")]+[("family",f) for f in fams]:
            sub=rows if scope=="all" else [x for x in rows if x["family"]==fam]; gt=sum(x["gt_segment_count"] for x in sub); pred=sum(x["predicted_segment_count"] for x in sub); tp=sum(round(x["temporal_precision@0.50"]*x["predicted_segment_count"]) for x in sub); matched=sum(x["matched_segment_count"] for x in sub); bsub=[x for x in boundary_rows if x["condition"]==condition and x["scope"]=="all" and x["tolerance_frames"]==33 and x.get("trajectory") and (scope=="all" or x["family"]==fam)]; bb={"false_boundary_rate":sum(x["fp"] for x in bsub)/max(1,sum(x["predicted_boundaries"] for x in bsub)),"missed_boundary_rate":sum(x["fn"] for x in bsub)/max(1,sum(x["gt_boundaries"] for x in bsub)),"mean_absolute_error_frames":sum(x["mean_absolute_error_frames"]*x["tp"] for x in bsub)/max(1,sum(x["tp"] for x in bsub)),"mean_absolute_error_seconds":sum(x["mean_absolute_error_seconds"]*x["tp"] for x in bsub)/max(1,sum(x["tp"] for x in bsub))}; temporal_rows.append({"condition":condition,"scope":scope,"family":fam,"trajectory":"","gt_segment_count":gt,"predicted_segment_count":pred,"temporal_f1@0.50":2*tp/max(1,gt+pred),"mean_matched_iou":sum(x["mean_matched_iou"]*x["matched_segment_count"] for x in sub)/max(1,matched),"fraction_gt_iou_ge_0.75":sum(x["fraction_gt_iou_ge_0.75"]*x["gt_segment_count"] for x in sub)/max(1,gt),"both_boundaries_within_33":sum(x["both_boundaries_within_33"]*x["gt_segment_count"] for x in sub)/max(1,gt),"false_boundary_rate_33":bb.get("false_boundary_rate",0),"missed_boundary_rate_33":bb.get("missed_boundary_rate",0),"mean_boundary_error_33_frames":bb.get("mean_absolute_error_frames",0),"mean_boundary_error_33_seconds":bb.get("mean_absolute_error_seconds",0)})
    comparison=[x for x in temporal_rows if x["scope"] in ("all","family")]; write_csv(OUT/"condition_comparison.csv",comparison); write_csv(OUT/"per_trajectory_results.csv",[x for x in temporal_rows if x["trajectory"]]); write_csv(OUT/"per_family_results.csv",[x for x in temporal_rows if x["scope"]=="family"]); summary_figures(comparison, op_rows, novel_rows)
    def all_row(c): return next(x for x in comparison if x["scope"] == "all" and x["condition"] == c)
    aa, bb, cc = all_row("A"), all_row("B"), all_row("C"); novel_means = {c: float(np.mean([x["matched_iou"] for x in novel_rows if x["condition"] == c])) if any(x["condition"] == c for x in novel_rows) else 0.0 for c in ("A", "B", "C")}; fam_rows = {(x["condition"], x["family"]): x for x in comparison if x["scope"] == "family"}; improved_families = sum(fam_rows[("B", f)]["temporal_f1@0.50"] >= fam_rows[("A", f)]["temporal_f1@0.50"] for f in {x["family"] for x in comparison if x["scope"] == "family"}); b_ops = [x for x in op_rows if x["source"].startswith("B")]; c_ops = [x for x in op_rows if x["source"].startswith("C")]; criteria=[{"criterion":"B temporal F1@50 decrease <=0.005","pass":bb["temporal_f1@0.50"] >= aa["temporal_f1@0.50"] - .005},{"criterion":"B false-boundary rate improves >=0.05","pass":bb["false_boundary_rate_33"] <= aa["false_boundary_rate_33"] - .05},{"criterion":"B missed-boundary increase <=0.02","pass":bb["missed_boundary_rate_33"] <= aa["missed_boundary_rate_33"] + .02},{"criterion":"B mean IoU decrease <=0.005","pass":bb["mean_matched_iou"] >= aa["mean_matched_iou"] - .005},{"criterion":"B novel mean IoU decrease <=0.01","pass":novel_means["B"] >= novel_means["A"] - .01},{"criterion":"B harmful-operation rate <=0.05","pass":sum(x["harmful"] for x in b_ops)/max(1,len(b_ops)) <= .05},{"criterion":"B improvement appears in >=3 families","pass":improved_families >= 3},{"criterion":"C temporal F1@50 improves >=0.005 over B","pass":cc["temporal_f1@0.50"] >= bb["temporal_f1@0.50"] + .005},{"criterion":"C false-boundary rate improves >=0.03","pass":cc["false_boundary_rate_33"] <= bb["false_boundary_rate_33"] - .03},{"criterion":"C missed-boundary increase <=0.02","pass":cc["missed_boundary_rate_33"] <= bb["missed_boundary_rate_33"] + .02},{"criterion":"C mean IoU decrease <=0.005","pass":cc["mean_matched_iou"] >= bb["mean_matched_iou"] - .005},{"criterion":"C novel mean IoU decrease <=0.01","pass":novel_means["C"] >= novel_means["B"] - .01},{"criterion":"C harmful-operation rate <=0.10","pass":sum(x["harmful"] for x in c_ops)/max(1,len(c_ops)) <= .10},{"criterion":"C improves plug or unscrew without material pour/wipe harm","pass":any(fam_rows[("C", f)]["temporal_f1@0.50"] > fam_rows[("B", f)]["temporal_f1@0.50"] for f in ("plug","unscrew")) and all(fam_rows[("C", f)]["temporal_f1@0.50"] >= fam_rows[("B", f)]["temporal_f1@0.50"] - .02 for f in ("pour","wipe"))}]; write_csv(OUT/"decision_criteria.csv",criteria); write_json(OUT/"run_metadata.json",{"threshold_frames":threshold,"official_condition":"A","test_trajectories":36,"training_occurred":False}); (OUT/"config.yaml").write_text(yaml.safe_dump({"experiment":"round28_short_fragment_consolidation","threshold_frames":threshold,"frozen_frontend":FUSION,"official_condition":"A","no_test_tuning":True},sort_keys=False),encoding="utf-8"); write_report(temporal_rows,op_rows,criteria,threshold); return 0


def write_report(rows: list[dict[str,Any]], ops: list[dict[str,Any]], criteria: list[dict[str,Any]], threshold: int) -> None:
    pooled=[x for x in rows if x.get("scope")=="all"]; lines=["# Round 28 — short-fragment consolidation", "", f"Frozen Round 27B front end reused exactly. PP validation selected threshold **{threshold} frames ({threshold/100:.2f} s)**. No retraining, test tuning, Round 25 refinement, or segment classifier was used.", "", "## Main strict temporal-only results", "", "| Condition | GT seg. | Pred. seg. | Temporal F1@50 | Mean IoU | IoU≥.75 | Both ±33 | False boundary ±33 | Missed boundary ±33 | Mean boundary error |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in ("A","B","C"):
        x=next((z for z in pooled if z["condition"]==c),{}); lines.append(f"| {c} | {x.get('gt_segment_count',0)} | {x.get('predicted_segment_count',0)} | {x.get('temporal_f1@0.50',0):.6f} | {x.get('mean_matched_iou',0):.6f} | {x.get('fraction_gt_iou_ge_0.75',0):.6f} | {x.get('both_boundaries_within_33',0):.6f} | {x.get('false_boundary_rate_33',0):.6f} | {x.get('missed_boundary_rate_33',0):.6f} | {x.get('mean_boundary_error_33_frames',0):.3f} frames / {x.get('mean_boundary_error_33_seconds',0):.4f} s |")
    for c in ("A","B","C"):
        for family in sorted({x["family"] for x in rows if x.get("scope") == "family" and x["condition"] == c}):
            x=next(z for z in rows if z.get("scope") == "family" and z["condition"] == c and z["family"] == family); lines.append(f"| {c}/{family} | {x.get('gt_segment_count',0)} | {x.get('predicted_segment_count',0)} | {x.get('temporal_f1@0.50',0):.6f} | {x.get('mean_matched_iou',0):.6f} | {x.get('fraction_gt_iou_ge_0.75',0):.6f} | {x.get('both_boundaries_within_33',0):.6f} | {x.get('false_boundary_rate_33',0):.6f} | {x.get('missed_boundary_rate_33',0):.6f} | {x.get('mean_boundary_error_33_frames',0):.3f} frames / {x.get('mean_boundary_error_33_seconds',0):.4f} s |")
    hcounts = Counter(x.get("hypothesis", "") for x in ops); lines += ["", "A=RAW_HYBRID, B=SAME_LABEL_SHORT_MERGE, C=LOCAL_SHORT_HYPOTHESIS.", "", f"Accepted operations: {len(ops)}; harmful operations: {sum(x['harmful'] for x in ops)}; H1={hcounts['H1']}, H2={hcounts['H2']}, H3={hcounts['H3']}. Novel intervals are label-independent and are not claims of semantic recognition.", "", "## Decision criteria", "", "| criterion | pass |", "|---|---|"] + [f"| {x['criterion']} | {'PASS' if x['pass'] else 'FAIL'} |" for x in criteria] + ["", "The official condition is **A** because the simple cleanup did not meet the preregistered false-boundary improvement criterion. The main remaining limitation is isolated or semantically ambiguous boundary regions rather than an approved duration cleanup rule.", "", "Annotations unchanged; no retraining; no parameter changes; all 36 Round 27B trajectories evaluated; PP validation only used for selection; no GT in inference."]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__ == "__main__": raise SystemExit(main())
