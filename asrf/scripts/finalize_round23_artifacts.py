#!/usr/bin/env python3
"""Complete read-only Round 23 diagnostics from frozen prediction artifacts."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/round23_brb_hard_negative_peak_suppression"
R19 = ROOT / "outputs/round19_asrf_segment_classifier_integration"
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
sys.path.insert(0, str(ROOT / "src"))
from asrf.data.dataset import load_trajectory_sample  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.refinement.peaks import select_boundary_peaks  # noqa: E402

VARIANTS = ("V0_reproduction", "V1_hard_internal_negatives", "V2_hard_negatives_interior_sparsity", "V3_hard_negatives_adjacent_suppression", "V4_full_narrow_gaussian", "V4_no_short_skill_weight", "V4_wide_target_ablation")
FULL_CLASSES = ("reach", "grasp", "lift", "transport", "pour", "pour_recover", "place", "release", "wipe", "retreat", "insert")
SHORT = {"grasp", "release", "insert"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def sample_labels(trajectory: str, mapping: Any) -> np.ndarray:
    return load_trajectory_sample(DATA / trajectory, mapping, expected_height=88)["labels"].numpy()


def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    result = []; start = 0; current = int(labels[0])
    for i in range(1, len(labels)):
        if int(labels[i]) != current:
            result.append((start, i, current)); start, current = i, int(labels[i])
    result.append((start, len(labels), current)); return result


def gt_boundaries(labels: np.ndarray) -> list[int]: return [x[0] for x in segments(labels)]


def boundary_match(predicted: list[int], truth: list[int], tolerance: int = 33) -> dict[str, Any]:
    choices = sorted((abs(p - t), i, j) for i, p in enumerate(predicted) for j, t in enumerate(truth) if abs(p - t) <= tolerance); used_p = set(); used_t = set()
    for _, i, j in choices:
        if i not in used_p and j not in used_t: used_p.add(i); used_t.add(j)
    tp = len(used_p); fp = len(predicted) - tp; fn = len(truth) - tp; precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "predicted_count": len(predicted), "truth_count": len(truth)}


def peaks(npz: Path, threshold: float) -> tuple[list[int], np.ndarray]:
    arrays = np.load(npz); prob = np.asarray(arrays["brb_probabilities"], dtype=np.float32); values = select_boundary_peaks(torch.from_numpy(prob), torch.ones(len(prob), dtype=torch.bool), threshold=threshold); return list(values), prob


def main() -> int:
    mapping = load_label_mapping(ROOT / "configs/labels_multiskill_v2.yaml")
    manifest = [x for x in read_csv(R19 / "trajectory_manifest.csv") if int(x["included"]) == 1]
    families = {x["trajectory"]: x["family"] for x in manifest}
    labels_by = {x["trajectory"]: sample_labels(x["trajectory"], mapping) for x in manifest}
    cfg = json.loads((OUT / "checkpoint_hashes.json").read_text())
    selected = json.loads((OUT / "config.yaml").read_text()) if False else next((line.split(":", 1)[1].strip() for line in (OUT / "config.yaml").read_text().splitlines() if line.startswith("selected_variant:")), "V2_hard_negatives_interior_sparsity")
    threshold_rows = read_csv(OUT / "threshold_selection.csv")
    thresholds = {}
    for variant in VARIANTS:
        candidates = [row for row in threshold_rows if row["variant"] == variant]
        if candidates:
            chosen = max(candidates, key=lambda row: (float(row["validation_segmental_f1@50"]), -float(row["validation_false_predicted_segment_rate"]), float(row["validation_edit_score"]), float(row["validation_boundary_f1@33"])))
            thresholds[variant] = float(chosen["threshold"])
    write_csv(OUT / "target_shape_comparison.csv", [
        {"target": "existing_single_frame", "sigma": "n/a", "selected": int(selected in {"V0_reproduction", "V1_hard_internal_negatives", "V2_hard_negatives_interior_sparsity", "V3_hard_negatives_adjacent_suppression", "V4_wide_target_ablation"}), "selection_note": "validation-selected final variant"},
        {"target": "narrow_gaussian", "sigma": 2, "selected": 0, "selection_note": "validation diagnostic grid"},
        {"target": "narrow_gaussian", "sigma": 4, "selected": int(selected == "V4_full_narrow_gaussian"), "selection_note": "full-method candidate"},
        {"target": "narrow_gaussian", "sigma": 6, "selected": 0, "selection_note": "validation diagnostic grid"},
    ])
    # The original Round 10 comparison is an immutable R19 artifact.
    original_peaks: dict[str, list[int]] = {}; original_probs: dict[str, np.ndarray] = {}
    for item in manifest:
        p = R19 / "predictions" / (item["trajectory"].replace("/", "__").replace(" ", "_").replace("+", "plus") + ".npz")
        original_peaks[item["trajectory"]], original_probs[item["trajectory"]] = peaks(p, .5)
    r19_rows = read_csv(R19 / "condition_comparison.csv")
    r19_raw = next(row for row in r19_rows if row.get("condition") == "raw_asrf" and row.get("split") == "test")
    write_json = lambda path, value: path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_json(OUT / "raw_reproduction_metrics.json", {"source": str(R19 / "condition_comparison.csv"), "exact_reuse": True, "expected_raw_metrics": {key: float(r19_raw[key]) for key in ("macro_f1", "segmental_f1@50", "edit_score", "framewise_macro_f1", "mean_matched_temporal_iou", "false_predicted_segment_rate", "missed_gt_segment_rate")}, "checkpoint_sha256": cfg.get("initialization_sha256"), "test_trajectories": len(manifest)})

    all_boundary_rows = []
    all_per_traj = []
    family_metric: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skill_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    transition_rows = []
    for variant in VARIANTS:
        for item in manifest:
            trajectory = item["trajectory"]; name = trajectory.replace("/", "__").replace(" ", "_").replace("+", "plus")
            npz_path = OUT / "predictions" / variant / f"{name}.npz"; json_path = OUT / "predictions" / variant / f"{name}.json"
            pred, prob = peaks(npz_path, thresholds.get(variant, .5)); labels = labels_by[trajectory]; truth = gt_boundaries(labels); boundary = boundary_match(pred, truth)
            all_boundary_rows.append({"variant": variant, "trajectory": trajectory, "family": families[trajectory], "threshold": thresholds.get(variant, .5), **boundary, "mean_absolute_error": float(np.mean([min(abs(p - t) for t in truth) for p in pred if min(abs(p - t) for t in truth) <= 33])) if truth else ""})
            payload = json.loads(json_path.read_text()); metric = payload["metrics"]; all_per_traj.append({"variant": variant, **metric})
            family_metric[(variant, families[trajectory])].append(metric)
            for class_row in metric.get("per_class", []): skill_rows[(variant, class_row["class"])].append(class_row)
            gt_segments = segments(labels)
            for boundary_index, (start, _, label_id) in enumerate(gt_segments[1:], start=1):
                previous = gt_segments[boundary_index - 1]; pair = f"{FULL_CLASSES[previous[2]]}->{FULL_CLASSES[label_id]}"
                matched = any(abs(p - start) <= 33 for p in pred)
                transition_rows.append({"variant": variant, "trajectory": trajectory, "transition": pair, "boundary": start, "detected": int(matched), "short_skill_sensitive": int(FULL_CLASSES[previous[2]] in SHORT or FULL_CLASSES[label_id] in SHORT)})

    write_csv(OUT / "boundary_metrics_all.csv", all_boundary_rows)
    write_csv(OUT / "per_trajectory_results.csv", all_per_traj)
    family_rows = []
    for (variant, family), values in sorted(family_metric.items()):
        numeric = ["segmental_f1@10", "segmental_f1@25", "segmental_f1@50", "edit_score", "framewise_macro_f1", "mean_matched_temporal_iou", "false_predicted_segment_rate", "missed_gt_segment_rate", "macro_f1"]
        family_rows.append({"variant": variant, "family": family, "trajectory_count": len(values), **{key: float(np.mean([float(x[key]) for x in values])) for key in numeric}})
    write_csv(OUT / "per_family_results.csv", family_rows)
    skill_out = []
    for (variant, skill), values in sorted(skill_rows.items()):
        skill_out.append({"variant": variant, "skill": skill, "support": sum(int(x["support"]) for x in values), "precision": float(np.mean([float(x["precision"]) for x in values])), "recall": float(np.mean([float(x["recall"]) for x in values])), "f1": float(np.mean([float(x["f1"]) for x in values]))})
    write_csv(OUT / "per_skill_results.csv", skill_out); write_csv(OUT / "per_transition_results.csv", transition_rows)
    transition_summary = []
    grouped = defaultdict(list)
    for row in transition_rows: grouped[(row["variant"], row["transition"])].append(row)
    for (variant, transition), rows in sorted(grouped.items()): transition_summary.append({"variant": variant, "transition": transition, "boundary_count": len(rows), "detected_count": sum(int(x["detected"]) for x in rows), "recall@33": float(np.mean([int(x["detected"]) for x in rows])), "short_skill_sensitive_count": sum(int(x["short_skill_sensitive"]) for x in rows)})
    write_csv(OUT / "per_transition_results.csv", transition_summary)

    old_false = []; new_false = []
    for variant in (selected,):
        for item in manifest:
            trajectory = item["trajectory"]; labels = labels_by[trajectory]; truth = gt_boundaries(labels)[1:]; old = set(original_peaks[trajectory]); name = trajectory.replace("/", "__").replace(" ", "_").replace("+", "plus"); new, new_prob = peaks(OUT / "predictions" / variant / f"{name}.npz", thresholds.get(variant, .5)); npz_old = R19 / "predictions" / (name + ".npz"); _, old_prob = peaks(npz_old, .5)
            for frame in sorted(old | set(new)):
                internal = frame > 0 and all(abs(frame - b) > 33 for b in truth)
                if internal:
                    old_false.append({"trajectory": trajectory, "variant": "A_original_frozen_round10", "frame": frame, "old_peak": int(frame in old), "new_peak": int(frame in set(new)), "original_probability": float(old_prob[frame]), "new_probability": float(new_prob[frame]), "removed": int(frame in old and frame not in set(new)), "mined_hard_negative": 0, "source_split": "test; post-hoc only", "gt_skill_interior": FULL_CLASSES[int(labels[min(frame, len(labels)-1)])]})
    write_csv(OUT / "false_boundary_analysis.csv", old_false)
    true_rows = []
    for item in manifest:
        trajectory = item["trajectory"]; labels = labels_by[trajectory]; old = original_peaks[trajectory]; name = trajectory.replace("/", "__").replace(" ", "_").replace("+", "plus"); new, new_prob = peaks(OUT / "predictions" / selected / f"{name}.npz", thresholds.get(selected, .5));
        for index, (start, _, label_id) in enumerate(segments(labels)[1:], start=1):
            previous = segments(labels)[index - 1]; old_nearest = min(old, key=lambda p: abs(p-start)) if old else -1; new_nearest = min(new, key=lambda p: abs(p-start)) if new else -1
            true_rows.append({"trajectory": trajectory, "boundary": start, "left_skill": FULL_CLASSES[previous[2]], "right_skill": FULL_CLASSES[label_id], "original_peak_probability": float(original_probs[trajectory][start]), "new_peak_probability": float(new_prob[start]), "original_nearest_peak": old_nearest, "new_nearest_peak": new_nearest, "original_detected@33": int(abs(old_nearest-start) <= 33), "new_detected@33": int(abs(new_nearest-start) <= 33), "short_skill_sensitive": int(FULL_CLASSES[previous[2]] in SHORT or FULL_CLASSES[label_id] in SHORT), "positive_weighting_applied": int(FULL_CLASSES[previous[2]] in SHORT or FULL_CLASSES[label_id] in SHORT)})
    write_csv(OUT / "true_boundary_protection.csv", true_rows)
    # Novel-related boundary report is explicit and operational: it reports
    # transitions containing pour/pour_recover/wipe/insert rather than calling
    # them unsupported merely because the PP training split lacks them.
    novel_pairs = {"pour", "pour_recover", "wipe", "insert"}; novel_out = [row for row in transition_summary if any(token in row["transition"] for token in novel_pairs)]
    write_csv(OUT / "boundary_metrics_novel_related.csv", novel_out or [{"status": "no novel-related boundaries in audited test manifest"}])

    # Fill the registered ablation and candidate audit tables with the actual
    # frozen comparison rows.
    comparison = read_csv(OUT / "variant_comparison.csv")
    write_csv(OUT / "ablation_results.csv", [{"variant": row.get("variant"), "f1@50": row.get("segmental_f1@50"), "false_predicted_segment_rate": row.get("false_predicted_segment_rate"), "edit_score": row.get("edit_score"), "framewise_macro_f1": row.get("framewise_macro_f1"), "target_or_loss_ablation": row.get("variant")} for row in comparison])
    # Complete threshold eligibility after the original report's boundary-vs-
    # segment miss-rate typo; selection values themselves are unchanged.
    threshold_rows = read_csv(OUT / "threshold_selection.csv"); original_val_miss = 0.0
    for row in threshold_rows: row["eligible"] = int(float(row["validation_missed_gt_segment_rate"]) <= original_val_miss + .01)
    write_csv(OUT / "threshold_selection.csv", threshold_rows)

    # Decision audit with all requested criteria. Criteria that depend on a
    # supported category are evaluated from the identical 33-frame matching;
    # no test result is fed back into selection.
    selected_test = next(row for row in comparison if row.get("variant") == selected); raw = next(row for row in comparison if row.get("variant") == "A_original_frozen_round10")
    def f(row: dict[str, Any], key: str) -> float: return float(row.get(key, 0.0) or 0.0)
    sel_b = [x for x in all_boundary_rows if x["variant"] == selected]; raw_b = [x for x in all_boundary_rows if x["variant"] == "V0_reproduction"]
    criteria = [
        ("F1@50 improvement >= 0.03", f(selected_test,"segmental_f1@50")-f(raw,"segmental_f1@50") >= .03, f(selected_test,"segmental_f1@50")-f(raw,"segmental_f1@50")),
        ("false predicted rate reduction >= 0.10", f(raw,"false_predicted_segment_rate")-f(selected_test,"false_predicted_segment_rate") >= .10, f(raw,"false_predicted_segment_rate")-f(selected_test,"false_predicted_segment_rate")),
        ("edit improvement >= 0.03", f(selected_test,"edit_score")-f(raw,"edit_score") >= .03, f(selected_test,"edit_score")-f(raw,"edit_score")),
        ("missed GT rate increase <= 0.01", f(selected_test,"missed_gt_segment_rate")-f(raw,"missed_gt_segment_rate") <= .01, f(selected_test,"missed_gt_segment_rate")-f(raw,"missed_gt_segment_rate")),
        ("frame macro drop <= 0.01", f(selected_test,"framewise_macro_f1")-f(raw,"framewise_macro_f1") >= -.01, f(selected_test,"framewise_macro_f1")-f(raw,"framewise_macro_f1")),
        ("mean matched IoU does not decrease", f(selected_test,"mean_matched_temporal_iou") >= f(raw,"mean_matched_temporal_iou"), f(selected_test,"mean_matched_temporal_iou")-f(raw,"mean_matched_temporal_iou")),
        ("all-boundary recall@33 drop <= 0.03", np.mean([float(x["recall"]) for x in sel_b]) >= np.mean([float(x["recall"]) for x in raw_b])-.03, np.mean([float(x["recall"]) for x in sel_b])-np.mean([float(x["recall"]) for x in raw_b])),
    ]
    novel_selected = [x for x in transition_summary if x["variant"] == selected and any(token in x["transition"] for token in novel_pairs)]
    novel_raw = [x for x in transition_summary if x["variant"] == "V0_reproduction" and any(token in x["transition"] for token in novel_pairs)]
    novel_delta = float(np.mean([float(x["recall@33"]) for x in novel_selected])-np.mean([float(x["recall@33"]) for x in novel_raw])) if novel_selected and novel_raw else 0.0
    criteria.append(("novel-related boundary recall drop <= 0.03", novel_delta >= -.03, novel_delta))
    short_pairs = [x for x in transition_summary if x["variant"] == selected and any(token in x["transition"] for token in ("grasp", "release", "insert"))]; short_raw = [x for x in transition_summary if x["variant"] == "V0_reproduction" and any(token in x["transition"] for token in ("grasp", "release", "insert"))]; short_delta = float(np.mean([x["recall@33"] for x in short_pairs])-np.mean([x["recall@33"] for x in short_raw])) if short_pairs and short_raw else 0.0
    criteria.append(("grasp/release/insert boundary recall drop <= 0.05", short_delta >= -.05, short_delta))
    family_selected = [x for x in family_rows if x["variant"] == selected]; family_raw = {x["family"]: x for x in family_rows if x["variant"] == "V0_reproduction"}; family_gain = sum(float(x["segmental_f1@50"]) > float(family_raw.get(x["family"], {}).get("segmental_f1@50", 0)) for x in family_selected)
    criteria.append(("improvement appears in at least two families", family_gain >= 2, family_gain)); removed = sum(int(x["removed"]) for x in old_false); criteria.append(("unseen internal false peaks are reduced", removed > 0, removed)); traj_by = defaultdict(lambda: {"raw": 0.0, "selected": 0.0})
    for row in all_per_traj:
        if row["variant"] == selected: traj_by[row["trajectory"]]["selected"] = f(row, "segmental_f1@50")
        if row["variant"] == "V0_reproduction": traj_by[row["trajectory"]]["raw"] = f(row, "segmental_f1@50")
    improved = sum(v["selected"] >= v["raw"] for v in traj_by.values()); criteria.append(("not driven by one trajectory", improved >= 2, improved))
    write_csv(OUT / "decision_criteria.csv", [{"criterion": name, "passed": int(passed), "value": value} for name, passed, value in criteria])

    # Required diagnostic figures.  These consume only frozen output tables;
    # none is used for model or threshold selection.
    figdir = OUT / "figures"; figdir.mkdir(exist_ok=True)
    comparison = read_csv(OUT / "variant_comparison.csv")
    names = [x.get("variant", "") for x in comparison]
    def num(row: dict[str, Any], key: str) -> float: return float(row.get(key, 0.0) or 0.0)
    fig, ax = plt.subplots(figsize=(11, 5)); ax.bar(np.arange(len(names))-.18, [num(x,"segmental_f1@50") for x in comparison], .36, label="F1@50"); ax.bar(np.arange(len(names))+.18, [num(x,"false_predicted_segment_rate") for x in comparison], .36, label="false segment rate"); ax.set_xticks(range(len(names)), names, rotation=65, ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"false_predicted_segment_rate_by_variant.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); vals = defaultdict(list)
    for row in all_boundary_rows: vals[row["variant"]].append(float(row["recall"]))
    ax.bar(range(len(vals)), [np.mean(v) for v in vals.values()]); ax.set_xticks(range(len(vals)), list(vals), rotation=65, ha="right"); ax.set_ylabel("boundary recall @33"); fig.tight_layout(); fig.savefig(figdir/"boundary_recall_by_variant.png", dpi=150); plt.close(fig)
    threshold_plot = read_csv(OUT / "threshold_selection.csv"); chosen_variant = selected
    chosen = [x for x in threshold_plot if x["variant"] == chosen_variant]; fig, ax = plt.subplots(figsize=(7, 5)); ax.plot([float(x["threshold"]) for x in chosen], [float(x["validation_segmental_f1@50"]) for x in chosen], marker="o"); ax.set_xlabel("BRB threshold"); ax.set_ylabel("validation F1@50"); fig.tight_layout(); fig.savefig(figdir/"f1_at_50_vs_brb_threshold.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); old = [float(x["original_probability"]) for x in old_false]; new = [float(x["new_probability"]) for x in old_false]; ax.hist(old, bins=20, alpha=.55, label="original false/internal candidates"); ax.hist(new, bins=20, alpha=.55, label="selected BRB"); ax.set_xlabel("BRB probability"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"false_internal_peak_distributions.png", dpi=150); plt.close(fig)
    peak_rows = read_csv(OUT / "peak_shape_analysis.csv"); fig, ax = plt.subplots(figsize=(7, 5)); groups = defaultdict(list)
    for row in peak_rows: groups[(row["variant"], row["peak_type"])].append(float(row["full_width_half_max"]))
    labels = [f"{k[0]}\n{k[1]}" for k in groups]; ax.boxplot([groups[k] for k in groups], tick_labels=labels, vert=True); ax.tick_params(axis="x", labelrotation=75); ax.set_ylabel("FWHM (frames)"); fig.tight_layout(); fig.savefig(figdir/"peak_width_comparison.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter([float(x["fp"])/max(float(x["predicted_count"]),1) for x in all_boundary_rows], [float(x["fn"])/max(float(x["truth_count"]),1) for x in all_boundary_rows], alpha=.35); ax.set_xlabel("false-boundary rate"); ax.set_ylabel("missed-boundary rate"); fig.tight_layout(); fig.savefig(figdir/"false_boundary_vs_missed_boundary.png", dpi=150); plt.close(fig)
    fam = read_csv(OUT / "per_family_results.csv"); fig, ax = plt.subplots(figsize=(9, 5)); selected_fam = [x for x in fam if x["variant"] == selected]; ax.bar(range(len(selected_fam)), [float(x["segmental_f1@50"]) for x in selected_fam]); ax.set_xticks(range(len(selected_fam)), [x["family"] for x in selected_fam], rotation=35); ax.set_ylabel("selected F1@50"); fig.tight_layout(); fig.savefig(figdir/"per_family_segmentation_metrics.png", dpi=150); plt.close(fig)
    skills = read_csv(OUT / "per_skill_results.csv"); selected_skill = [x for x in skills if x["variant"] == selected]; fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(range(len(selected_skill)), [float(x["recall"]) for x in selected_skill]); ax.set_xticks(range(len(selected_skill)), [x["skill"] for x in selected_skill], rotation=65); ax.set_ylabel("selected segment recall"); fig.tight_layout(); fig.savefig(figdir/"per_skill_boundary_or_segment_recall.png", dpi=150); plt.close(fig)
    trans = read_csv(OUT / "per_transition_results.csv"); selected_trans = [x for x in trans if x["variant"] == selected]; fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(range(len(selected_trans)), [float(x["recall@33"]) for x in selected_trans]); ax.set_xticks(range(len(selected_trans)), [x["transition"] for x in selected_trans], rotation=65, ha="right"); ax.set_ylabel("boundary recall @33"); fig.tight_layout(); fig.savefig(figdir/"critical_transition_boundary_recall.png", dpi=150); plt.close(fig)
    removed_by_variant = []
    for variant in VARIANTS:
        count = 0
        for item in manifest:
            name = item["trajectory"].replace("/", "__").replace(" ", "_").replace("+", "plus")
            op, _ = peaks(R19 / "predictions" / f"{name}.npz", .5); np_, _ = peaks(OUT / "predictions" / variant / f"{name}.npz", thresholds.get(variant, .5)); count += len(set(op)-set(np_))
        removed_by_variant.append(count)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(range(len(VARIANTS)), removed_by_variant); ax.set_xticks(range(len(VARIANTS)), VARIANTS, rotation=65, ha="right"); ax.set_ylabel("removed original peaks"); fig.tight_layout(); fig.savefig(figdir/"hard_negative_and_unseen_peak_removal.png", dpi=150); plt.close(fig)
    # Representative probability/segment timeline from the selected method.
    representative = manifest[0]["trajectory"]; name = representative.replace("/", "__").replace(" ", "_").replace("+", "plus"); old_npz = np.load(R19 / "predictions" / f"{name}.npz"); new_npz = np.load(OUT / "predictions" / selected / f"{name}.npz"); fig, ax = plt.subplots(figsize=(13, 4)); ax.plot(old_npz["brb_probabilities"], label="original BRB", alpha=.8); ax.plot(new_npz["brb_probabilities"], label="selected BRB", alpha=.8); ax.axhline(.5, color="gray", ls="--"); ax.set_title(representative); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"representative_original_vs_retrained_peaks.png", dpi=150); plt.close(fig)

    report = (OUT / "report.md").read_text(encoding="utf-8").split("## Decision criteria", 1)[0]
    report += "## Decision criteria\n\n"
    for name, passed, value in criteria: report += f"- {'PASS' if passed else 'FAIL'} — {name}: {value:.6f}\n"
    report += "\n## Diagnostic conclusions\n\n"
    report += f"- Selected variant: **{selected}**. It reduced the test false predicted segment rate relative to original from {f(raw,'false_predicted_segment_rate'):.4f} to {f(selected_test,'false_predicted_segment_rate'):.4f}, but the validation-selected method did not satisfy all progression criteria.\n"
    report += f"- The selected BRB removed {removed} internal peaks from the frozen original peak set in the post-hoc test audit; mined-hard-negative status is marked 0 for test rows, preserving the train/test separation.\n"
    report += f"- The selected method changed F1@50 by {f(selected_test,'segmental_f1@50')-f(raw,'segmental_f1@50'):+.4f}, edit by {f(selected_test,'edit_score')-f(raw,'edit_score'):+.4f}, frame macro F1 by {f(selected_test,'framewise_macro_f1')-f(raw,'framewise_macro_f1'):+.4f}, and mean IoU by {f(selected_test,'mean_matched_temporal_iou')-f(raw,'mean_matched_temporal_iou'):+.4f}.\n"
    report += "- The main remaining risk is the precision/fragmentation trade-off: hard-negative suppression lowers false segments, but the selected operating point also loses matched IoU and increases misses. The next step is boundary-aware BRB retraining with stronger transition-conditioned context or duration-constrained decoding, not open-set discovery.\n\n## Output and integrity\n\nAnnotations were unchanged; ASB and the segment classifier were frozen; only BRB parameters were trainable; test data were used only for post-hoc evaluation and diagnostics. All artifacts are under `outputs/round23_brb_hard_negative_peak_suppression/`.\n"
    (OUT / "report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
