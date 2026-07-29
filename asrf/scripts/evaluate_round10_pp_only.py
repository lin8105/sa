"""Round 10 PP-only validation-gated factorial evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
OUT = ROOT / "outputs/round10_pp_only_novel_segmentation"
AUDIT = OUT / "audit"
VAL = OUT / "validation"
KNOWN = ("reach", "grasp", "lift", "transport", "place", "release", "retreat")
KNOWN_SET = set(KNOWN)
DISTANCES = (0, 10, 20, 30)
FAMILY_NOVEL = {"pour": ("pour", "pour_recover"), "wipe": ("wipe",), "plug": ("place", "insert")}

import sys
sys.path.insert(0, str(ROOT / "src"))

from asrf.data.dataset import load_trajectory_sample
from asrf.data.labels import load_label_mapping
from asrf.evaluation.metrics import boundary_counts, edit_score, labels_to_segments, segmental_f1
from asrf.models import ASRFModel
from asrf.refinement.majority_vote import _vote_one
from asrf.refinement.peaks import greedy_score_guided_nms, select_boundary_peaks
from asrf.refinement.segments import TemporalInterval, construct_segments
from asrf.training.checkpointing import load_checkpoint, sha256_file


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def entries(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def boundaries(labels: torch.Tensor) -> list[int]:
    values = labels.tolist()
    return ([0] + [i for i in range(1, len(values)) if int(values[i]) != int(values[i - 1])]) if values else []


def intervals(labels: torch.Tensor) -> list[TemporalInterval]:
    return construct_segments(boundaries(labels), len(labels))


def iou(a: TemporalInterval, b: TemporalInterval) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = a.duration + b.duration - inter
    return inter / union if union else 0.0


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return precision, recall, f1


def duplicate_count(peaks_list: list[int], truth_list: list[int], tolerance: int = 33) -> int:
    return sum(max(0, sum(abs(p - t) <= tolerance for p in peaks_list) - 1) for t in truth_list)


def match(predicted: list[TemporalInterval], truth: list[TemporalInterval], same_label: bool = False, pred_labels: list[str] | None = None, truth_labels: list[str] | None = None) -> list[tuple[int, int, float]]:
    candidates = []
    for pi, p in enumerate(predicted):
        for ti, t in enumerate(truth):
            if same_label and pred_labels is not None and truth_labels is not None and pred_labels[pi] != truth_labels[ti]:
                continue
            candidates.append((iou(p, t), pi, ti))
    used_p: set[int] = set()
    used_t: set[int] = set()
    result = []
    for score, pi, ti in sorted(candidates, key=lambda x: (-x[0], x[1], x[2])):
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            result.append((pi, ti, score))
    return result


def load_model(short: str) -> tuple[ASRFModel, dict[str, Any], dict[int, str], str]:
    import yaml
    mode = "single_frame" if short == "sf" else "hard_window_r5"
    config_path = OUT / "models" / mode / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mapping = load_label_mapping(ROOT / config["data"]["label_config"])
    model = ASRFModel.from_config(config)
    checkpoint = OUT / "models" / mode / "best.pt"
    model.load_state_dict(load_checkpoint(checkpoint)["model_state"], strict=True)
    return model.eval(), config, {int(v): k for k, v in mapping.items()}, sha256_file(checkpoint)


def target_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config["data"][key] for key in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")}


@torch.no_grad()
def infer(model: ASRFModel, paths: list[str], label_path: Path, target: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = load_label_mapping(label_path)
    result = []
    for entry in paths:
        sample = load_trajectory_sample(DATA / entry, mapping, expected_height=88, boundary_target_config=target)
        output = model(sample["heatmap"].unsqueeze(0), valid_mask=sample["valid_mask"].unsqueeze(0))
        result.append({"entry": entry, "sample": sample, "truth": sample["labels"].cpu(), "asb": output.asb_stage_probabilities[-1][0].cpu(), "brb": output.brb_stage_probabilities[-1][0, 0].cpu()})
    return result


def peaks(row: dict[str, Any], distance: int) -> dict[str, Any]:
    brb = row["brb"]
    candidate = [int(x) for x in select_boundary_peaks(brb, torch.ones(len(brb), dtype=torch.bool), threshold=0.5)]
    scores = [float(brb[x]) for x in candidate]
    retained = greedy_score_guided_nms(candidate, scores, distance)
    retained_set = set(retained)
    suppressor: dict[int, int] = {}
    if distance:
        for p, score in sorted(zip(candidate, scores), key=lambda x: (-x[1], x[0])):
            if p in retained_set:
                continue
            nearby = [(float(brb[q]), q) for q in retained if abs(q - p) < distance]
            if nearby:
                suppressor[p] = max(nearby, key=lambda x: (x[0], -x[1]))[1]
    return {"candidate": candidate, "retained": retained, "suppressed": sorted(set(candidate) - retained_set), "suppressor": suppressor}


def variant(row: dict[str, Any], distance: int) -> dict[str, Any]:
    p = peaks(row, distance)
    refined, _ = _vote_one(row["asb"], construct_segments(p["retained"], len(row["truth"])), voting="majority")
    raw = row["asb"].argmax(dim=0)
    return {"raw": raw, "refined": refined, **p}


def semantic(pred: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    return {"accuracy": float((pred == truth).float().mean()), "edit": edit_score(pred, truth), "F1@10": segmental_f1(pred, truth, .10), "F1@25": segmental_f1(pred, truth, .25), "F1@50": segmental_f1(pred, truth, .50)}


def validation_metrics(rows: list[dict[str, Any]], distance: int) -> dict[str, Any]:
    sem = []
    pooled = {str(t): {"tp": 0, "fp": 0, "fn": 0} for t in (5, 10, 20, 33, 50)}
    known_f1 = []
    for row in rows:
        v = variant(row, distance)
        sem.append(semantic(v["refined"], row["truth"]))
        known_f1.append(semantic(v["refined"], row["truth"])["F1@50"])
        truth_b = [x for x in boundaries(row["truth"]) if x]
        pred_b = [x for x in v["retained"] if x]
        for tolerance, counts in pooled.items():
            item = boundary_counts(pred_b, truth_b, int(tolerance), include_frame0=False)
            for key in ("tp", "fp", "fn"):
                counts[key] += int(item[key])
    for counts in pooled.values():
        counts["precision"], counts["recall"], counts["F1"] = prf(counts["tp"], counts["fp"], counts["fn"])
    return {"refined_accuracy": float(np.mean([x["accuracy"] for x in sem])), "refined_edit": float(np.mean([x["edit"] for x in sem])), "refined_F1@10": float(np.mean([x["F1@10"] for x in sem])), "refined_F1@25": float(np.mean([x["F1@25"] for x in sem])), "refined_F1@50": float(np.mean([x["F1@50"] for x in sem])), "boundary": pooled, "false_peaks": pooled["33"]["fp"], "missed_boundaries": pooled["33"]["fn"], "known_skill_segment_F1@50": float(np.mean(known_f1)), "predicted_peaks": sum(len(variant(row, distance)["retained"]) for row in rows), "true_boundaries": sum(len(boundaries(row["truth"])) - 1 for row in rows)}


def safety(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, bool]:
    return {"refined_F1@50_drop_ok": candidate["refined_F1@50"] >= base["refined_F1@50"] - .01, "missed_boundary_increase_ok": candidate["missed_boundaries"] <= base["missed_boundaries"] * 1.10, "known_skill_F1_drop_ok": candidate["known_skill_segment_F1@50"] >= base["known_skill_segment_F1@50"] - .01}


def run_validation() -> None:
    VAL.mkdir(parents=True, exist_ok=True)
    val_paths = entries(AUDIT / "pp_validation_manifest.txt")
    saved = {}
    rows_csv = []
    for short in ("sf", "r5"):
        model, config, names, checkpoint_hash = load_model(short)
        rows = infer(model, val_paths, ROOT / config["data"]["label_config"], target_config(config))
        metrics = {str(d): validation_metrics(rows, d) for d in DISTANCES}
        base = metrics["0"]
        for d in DISTANCES:
            metrics[str(d)]["safety"] = safety(base, metrics[str(d)])
            metrics[str(d)]["passes_safety"] = d == 0 or all(metrics[str(d)]["safety"].values())
        candidates = [d for d in DISTANCES if metrics[str(d)]["passes_safety"]]
        selected = min(candidates, key=lambda d: (-metrics[str(d)]["refined_F1@50"], -metrics[str(d)]["boundary"]["33"]["F1"], metrics[str(d)]["missed_boundaries"], metrics[str(d)]["false_peaks"], d))
        payload = {"model": short, "candidate_distances": list(DISTANCES), "metrics": metrics, "baseline_metrics": base, "selected_distance": selected, "selection_reason": "highest validation refined F1@50 after safety gates and prescribed tie-breakers", "checkpoint_sha256": checkpoint_hash, "test_metrics_accessed_before_selection": False, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json(VAL / f"{short}_nms_selection.json", payload)
        saved[short] = payload
        for d in DISTANCES:
            m = metrics[str(d)]
            rows_csv.append({"model": short, "distance_frames": d, "split": "validation", "refined_F1@50": m["refined_F1@50"], "boundary_F1@33": m["boundary"]["33"]["F1"], "false_peaks": m["false_peaks"], "missed_boundaries": m["missed_boundaries"], "known_skill_segment_F1@50": m["known_skill_segment_F1@50"], "passes_safety": m["passes_safety"], "selected": int(d == selected)})
    global_metrics = []
    for d in DISTANCES:
        sf = saved["sf"]["metrics"][str(d)]
        r5 = saved["r5"]["metrics"][str(d)]
        global_metrics.append({"distance_frames": d, "macro_refined_F1@50": (sf["refined_F1@50"] + r5["refined_F1@50"]) / 2, "macro_boundary_F1@33": (sf["boundary"]["33"]["F1"] + r5["boundary"]["33"]["F1"]) / 2, "macro_false_peaks": (sf["false_peaks"] + r5["false_peaks"]) / 2, "macro_missed_boundaries": (sf["missed_boundaries"] + r5["missed_boundaries"]) / 2, "passes_global_safety": sf["passes_safety"] and r5["passes_safety"]})
    candidates = [x for x in global_metrics if x["passes_global_safety"]]
    selected = min(candidates, key=lambda x: (-x["macro_refined_F1@50"], -x["macro_boundary_F1@33"], x["macro_missed_boundaries"], x["macro_false_peaks"], x["distance_frames"]))["distance_frames"]
    write_json(VAL / "global_nms_selection.json", {"candidate_distances": list(DISTANCES), "metrics": global_metrics, "family_model_selected_distances": {"sf": saved["sf"]["selected_distance"], "r5": saved["r5"]["selected_distance"]}, "selected_distance": selected, "one_common_global_distance_supported": selected > 0 and all(x["passes_global_safety"] for x in global_metrics if x["distance_frames"] == selected), "test_metrics_accessed_before_selection": False, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    write_csv(VAL / "validation_metrics.csv", rows_csv)
    write_csv(VAL / "global_validation_metrics.csv", global_metrics)


def named_intervals(labels: torch.Tensor, inverse: dict[int, str]) -> list[dict[str, Any]]:
    return [{"start": s.start, "end": s.end, "label": inverse[int(labels[s.start])]} for s in intervals(labels)]


def recover(truth: list[dict[str, Any]], predicted: list[TemporalInterval], novel: bool, peaks_list: list[int], length: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    targets = [x for x in truth if (x["label"] not in KNOWN_SET) == novel]
    target_intervals = [TemporalInterval(x["start"], x["end"]) for x in targets]
    matches = match(predicted, target_intervals)
    best = {ti: (pi, score) for pi, ti, score in matches}
    scores = [score for _, _, score in matches]
    rows = []
    fragments = merges_prev = merges_next = 0
    for ti, target in enumerate(target_intervals):
        overlaps = [pi for pi, p in enumerate(predicted) if iou(p, target) > 0]
        start_ok = target.start == 0 or any(abs(p - target.start) <= 33 for p in peaks_list)
        end_ok = target.end == length or any(abs(p - target.end) <= 33 for p in peaks_list)
        if len(overlaps) > 1:
            fragments += 1
        previous = any(p.start < target.start and p.end > target.start for p in predicted)
        following = any(p.start < target.end and p.end > target.end for p in predicted)
        merges_prev += int(previous)
        merges_next += int(following)
        pi, score = best.get(ti, ("", 0.0))
        rows.append({"truth_start": target.start, "truth_end_exclusive": target.end, "iou": score, "matched_prediction": pi, "overlapping_prediction_count": len(overlaps), "start_boundary_correct_33": int(start_ok), "end_boundary_correct_33": int(end_ok), "merge_previous": int(previous), "merge_next": int(following)})
    support = len(targets)
    both = [x for x in targets if x["start"] > 0 and x["end"] < length]
    summary = {
        "support": support,
        "mean_IoU": float(np.mean(scores)) if scores else 0.0,
        "median_IoU": float(np.median(scores)) if scores else 0.0,
        "IoU10_recovery": sum(score >= .10 for score in scores) / support if support else 0.0,
        "IoU25_recovery": sum(score >= .25 for score in scores) / support if support else 0.0,
        "IoU50_recovery": sum(score >= .50 for score in scores) / support if support else 0.0,
        "IoU75_recovery": sum(score >= .75 for score in scores) / support if support else 0.0,
        "start_boundary_recall_33": np.mean([x["start_boundary_correct_33"] for x in rows if x["truth_start"] > 0]) if any(x["truth_start"] > 0 for x in rows) else 0.0,
        "end_boundary_recall_33": np.mean([x["end_boundary_correct_33"] for x in rows if x["truth_end_exclusive"] < length]) if any(x["truth_end_exclusive"] < length for x in rows) else 0.0,
        "both_boundaries_correct": np.mean([int(x["start_boundary_correct_33"] and x["end_boundary_correct_33"]) for x in rows if x["truth_start"] > 0 and x["truth_end_exclusive"] < length]) if both else 0.0,
        "exact_one_segment_recovery": sum(x["overlapping_prediction_count"] == 1 for x in rows) / support if support else 0.0,
        "fragmentation_rate": fragments / support if support else 0.0,
        "merge_previous_rate": merges_prev / support if support else 0.0,
        "merge_next_rate": merges_next / support if support else 0.0,
        "mean_predicted_fragments": float(np.mean([x["overlapping_prediction_count"] for x in rows])) if rows else 0.0,
    }
    return summary, rows


def oracle_known(asb: torch.Tensor, truth: list[dict[str, Any]], model_inverse: dict[int, str]) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for segment in truth:
        if segment["label"] in KNOWN_SET:
            y_true.append(segment["label"])
            y_pred.append(model_inverse[int(asb[:, segment["start"]:segment["end"]].mean(dim=1).argmax())])
    per = {}
    for skill in KNOWN:
        tp = sum(a == skill and b == skill for a, b in zip(y_true, y_pred))
        fp = sum(a != skill and b == skill for a, b in zip(y_true, y_pred))
        fn = sum(a == skill and b != skill for a, b in zip(y_true, y_pred))
        if tp + fp + fn:
            p, r, f = prf(tp, fp, fn)
            per[skill] = {"precision": p, "recall": r, "F1": f, "support": sum(a == skill for a in y_true)}
    return {"segment_accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0, "macro_F1": float(np.mean([x["F1"] for x in per.values()])) if per else 0.0, "per_class": per, "confusion": {f"{a}->{b}": sum(x == a and y == b for x, y in zip(y_true, y_pred)) for a in KNOWN for b in KNOWN if any(x == a and y == b for x, y in zip(y_true, y_pred))}}


def analyze(row: dict[str, Any], v: dict[str, Any], model_inverse: dict[int, str], test_inverse_map: dict[int, str]) -> dict[str, Any]:
    truth = row["truth"]
    truth_named = named_intervals(truth, test_inverse_map)
    predicted = intervals(v["refined"])
    predicted_names = [model_inverse[int(v["refined"][segment.start])] for segment in predicted]
    all_boundaries = {str(t): boundary_counts([x for x in v["retained"] if x], [x for x in boundaries(truth) if x], t, include_frame0=False) for t in (5, 10, 20, 33, 50)}
    for item in all_boundaries.values():
        item["F1"] = item["f1"]
    known, known_rows = recover(truth_named, predicted, False, v["retained"], len(truth))
    novel, novel_rows = recover(truth_named, predicted, True, v["retained"], len(truth))
    known_truth = [x for x in truth_named if x["label"] in KNOWN_SET]
    tp = fp = fn = 0
    pred_known = [i for i, name in enumerate(predicted_names) if name in KNOWN_SET]
    candidates = []
    for pi in pred_known:
        for ti, target in enumerate(known_truth):
            if predicted_names[pi] == target["label"]:
                score = iou(predicted[pi], TemporalInterval(target["start"], target["end"]))
                if score >= .5:
                    candidates.append((score, pi, ti))
    used_p: set[int] = set()
    used_t: set[int] = set()
    for _, pi, ti in sorted(candidates, reverse=True):
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            tp += 1
    fp = len(pred_known) - tp
    fn = len(known_truth) - tp
    _, _, known_pred_f1 = prf(tp, fp, fn)
    known_frame_mask = torch.zeros(len(truth), dtype=torch.bool)
    for segment in known_truth:
        known_frame_mask[segment["start"]:segment["end"]] = True
    known_frame_accuracy = float((v["refined"][known_frame_mask] == truth[known_frame_mask]).float().mean()) if known_frame_mask.any() else 0.0
    category_counts = defaultdict(lambda: {"support": 0, "detected_5": 0, "detected_10": 0, "detected_20": 0, "detected_33": 0})
    name_frames = [test_inverse_map[int(x)] for x in truth.tolist()]
    for boundary in boundaries(truth)[1:]:
        category = f"{'known' if name_frames[boundary - 1] in KNOWN_SET else 'novel'}->{'known' if name_frames[boundary] in KNOWN_SET else 'novel'}"
        counts = category_counts[category]
        counts["support"] += 1
        for tolerance in (5, 10, 20, 33):
            counts[f"detected_{tolerance}"] += int(any(abs(p - boundary) <= tolerance for p in v["retained"] if p))
    category_summary = {}
    for category, counts in category_counts.items():
        category_summary[category] = {"support": counts["support"], **{f"recall_{t}": counts[f"detected_{t}"] / counts["support"] for t in (5, 10, 20, 33)}}
    return {"boundary": all_boundaries, "known": known, "known_rows": known_rows, "novel": novel, "novel_rows": novel_rows, "known_oracle": oracle_known(row["asb"], truth_named, model_inverse), "known_predicted_F1@50": known_pred_f1, "known_frame_accuracy": known_frame_accuracy, "category_summary": category_summary, "predicted": predicted, "predicted_names": predicted_names, "truth_named": truth_named}


def aggregate_test(rows: list[dict[str, Any]], model_inverse: dict[int, str], test_inverse_map: dict[int, str], distance: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    analyzed = []
    boundary = {str(t): {"tp": 0, "fp": 0, "fn": 0} for t in (5, 10, 20, 33, 50)}
    categories = defaultdict(lambda: {"support": 0, "detected_5": 0, "detected_10": 0, "detected_20": 0, "detected_33": 0})
    for row in rows:
        v = variant(row, distance)
        a = analyze(row, v, model_inverse, test_inverse_map)
        analyzed.append((row, v, a))
        for tolerance, counts in a["boundary"].items():
            for key in ("tp", "fp", "fn"):
                boundary[tolerance][key] += int(counts[key])
        for category, counts in a["category_summary"].items():
            for key, value in counts.items():
                if key == "support":
                    categories[category][key] += value
                else:
                    categories[category][f"detected_{key.split('_')[-1]}"] += value * counts["support"]
    for counts in boundary.values():
        counts["precision"], counts["recall"], counts["F1"] = prf(counts["tp"], counts["fp"], counts["fn"])
    category_summary = {}
    for category, counts in categories.items():
        support = counts["support"]
        category_summary[category] = {"support": support, **{f"recall_{t}": counts[f"detected_{t}"] / support if support else 0.0 for t in (5, 10, 20, 33)}}
    def mean(path: str) -> float:
        values = [float(item[0][path]) for item in analyzed]
        return float(np.mean(values)) if values else 0.0
    known = [item[2]["known"] for item in analyzed]
    novel = [item[2]["novel"] for item in analyzed]
    oracle = [item[2]["known_oracle"]["macro_F1"] for item in analyzed]
    aggregate = {"raw_F1@50": None, "refined_F1@50": float(np.mean([segmental_f1(x[1]["refined"], x[0]["truth"], .5) for x in analyzed])) if analyzed and all((x[0]["truth"] < len(model_inverse)).all() for x in analyzed) else None, "raw_accuracy": None, "refined_accuracy": None, "boundary": boundary, "false_peaks": boundary["33"]["fp"], "missed_boundaries": boundary["33"]["fn"], "duplicate_peaks": sum(duplicate_count([p for p in x[1]["retained"] if p], [p for p in boundaries(x[0]["truth"]) if p]) for x in analyzed), "predicted_peaks": sum(len(x[1]["retained"]) for x in analyzed), "true_boundaries": sum(len(boundaries(x[0]["truth"])) - 1 for x in analyzed), "known_segmentation": {key: float(np.mean([x[key] for x in known])) for key in known[0]} if known else {}, "novel_segmentation": {key: float(np.mean([x[key] for x in novel])) for key in novel[0]} if novel else {}, "known_oracle_macro_F1": float(np.mean(oracle)) if oracle else 0.0, "known_predicted_segment_F1@50": float(np.mean([x[2]["known_predicted_F1@50"] for x in analyzed])) if analyzed else 0.0, "categories": category_summary}
    return aggregate, analyzed


def plot_timeline(path: Path, row: dict[str, Any], v: dict[str, Any], a: dict[str, Any], model_inverse: dict[int, str], test_inverse_map: dict[int, str], condition: str, distance: int) -> None:
    import matplotlib.pyplot as plt
    path.mkdir(parents=True, exist_ok=True)
    truth = row["truth"]
    names = [test_inverse_map[int(x)] for x in truth.tolist()]
    raw_names = [model_inverse[int(x)] for x in v["raw"].tolist()]
    refined_names = [model_inverse[int(x)] for x in v["refined"].tolist()]
    palette = {"reach": "#1f77b4", "grasp": "#ff7f0e", "lift": "#2ca02c", "transport": "#d62728", "place": "#9467bd", "release": "#8c564b", "retreat": "#e377c2", "pour": "#17becf", "pour_recover": "#bcbd22", "wipe": "#7f7f7f", "insert": "#637939"}
    fig, axes = plt.subplots(5, 1, figsize=(18, 9), sharex=True, gridspec_kw={"height_ratios": [1.2, 1, 1, 1.5, .8]})
    for axis, values, title in ((axes[0], names, "GT (known/novel)"), (axes[1], raw_names, "raw ASB"), (axes[2], refined_names, f"{condition}, NMS d={distance}")):
        starts = [0] + [i for i in range(1, len(values)) if values[i] != values[i - 1]]
        ends = starts[1:] + [len(values)]
        for start, end in zip(starts, ends):
            name = values[start]
            axis.axvspan(start / 100, end / 100, color=palette.get(name, "#aaaaaa"), alpha=.7)
            axis.text((start + end) / 200, .5, name, ha="center", va="center", fontsize=7, clip_on=True)
        axis.set_yticks([])
        axis.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=8)
    t = np.arange(len(row["brb"])) / 100
    axes[3].plot(t, row["brb"].numpy(), color="black", linewidth=.7)
    axes[3].axhline(.5, color="red", linestyle="--", linewidth=.6)
    axes[3].set_ylabel("BRB", rotation=0, ha="right")
    axes[4].eventplot([[p / 100 for p in v["candidate"]]], lineoffsets=1, colors="orange")
    axes[4].eventplot([[p / 100 for p in v["retained"]]], lineoffsets=2, colors="green")
    axes[4].eventplot([[p / 100 for p in v["suppressed"]]], lineoffsets=3, colors="red")
    axes[4].eventplot([[p / 100 for p in boundaries(truth)[1:]]], lineoffsets=4, colors="blue")
    axes[4].set_yticks([1, 2, 3, 4], ["candidate", "retained", "suppressed", "GT boundary"])
    axes[4].set_xlabel("time (s)")
    axes[4].set_xlim(0, len(truth) / 100)
    fig.suptitle(row["entry"])
    fig.tight_layout()
    fig.savefig(path / "timeline.png", dpi=120)
    plt.close(fig)


def write_per_trajectory(base: Path, row: dict[str, Any], v: dict[str, Any], a: dict[str, Any], model_inverse: dict[int, str], test_inverse_map: dict[int, str], condition: str, distance: int) -> None:
    base.mkdir(parents=True, exist_ok=True)
    truth = row["truth"]
    truth_names = [test_inverse_map[int(x)] for x in truth.tolist()]
    raw_names = [model_inverse[int(x)] for x in v["raw"].tolist()]
    refined_names = [model_inverse[int(x)] for x in v["refined"].tolist()]
    timestamps = row["sample"]["timestamps"].numpy()
    candidate = set(v["candidate"])
    retained = set(v["retained"])
    suppressed = set(v["suppressed"])
    write_csv(base / "frame_predictions.csv", [{"frame": i, "time_s": float((timestamps[i] - timestamps[0]) / 1e6), "gt_label": truth_names[i], "gt_known_or_novel": "known" if truth_names[i] in KNOWN_SET else "novel", "raw_asb_label": raw_names[i], "raw_asb_confidence": float(row["asb"][:, i].max()), "refined_label": refined_names[i], "brb_probability": float(row["brb"][i]), "gt_boundary": int(i in boundaries(truth)), "candidate_peak": int(i in candidate), "retained_peak": int(i in retained), "suppressed_peak": int(i in suppressed)} for i in range(len(truth))])
    predicted = intervals(v["refined"])
    pred_segments = []
    truth_named = a["truth_named"]
    for index, segment in enumerate(predicted):
        overlaps = [(x["label"], iou(segment, TemporalInterval(x["start"], x["end"]))) for x in truth_named if iou(segment, TemporalInterval(x["start"], x["end"])) > 0]
        best = max(overlaps, key=lambda x: x[1]) if overlaps else ("", 0.0)
        pred_segments.append({"segment_index": index, "start_frame": segment.start, "end_frame_exclusive": segment.end, "duration_frames": segment.duration, "duration_s": segment.duration / 100, "predicted_label": refined_names[segment.start], "best_gt_label": best[0], "temporal_iou": best[1], "correct_at_iou50": int(best[1] >= .5 and best[0] == refined_names[segment.start])})
    write_csv(base / "segment_predictions.csv", pred_segments)
    write_csv(base / "boundary_predictions.csv", [{"frame": p, "time_s": float((timestamps[p] - timestamps[0]) / 1e6), "brb_probability": float(row["brb"][p]), "candidate": 1, "retained": int(p in retained), "suppressed": int(p in suppressed), "suppressing_peak": v["suppressor"].get(p, "")} for p in sorted(candidate)])
    write_csv(base / "known_segment_recovery.csv", a["known_rows"])
    write_csv(base / "novel_segment_recovery.csv", a["novel_rows"])
    write_json(base / "metrics.json", {"entry": row["entry"], "condition": condition, "distance_frames": distance, "boundary": a["boundary"], "known_skill_segmentation": a["known"], "known_skill_oracle_recognition": a["known_oracle"], "known_skill_predicted_segment_F1@50": a["known_predicted_F1@50"], "known_frame_accuracy": a["known_frame_accuracy"], "novel_skill_segmentation": a["novel"], "boundary_categories": a["category_summary"], "candidate_peaks": v["candidate"], "retained_peaks": v["retained"], "suppressed_peaks": v["suppressed"]})
    plot_timeline(base, row, v, a, model_inverse, test_inverse_map, condition, distance)


def run_test() -> None:
    required = [VAL / "sf_nms_selection.json", VAL / "r5_nms_selection.json", VAL / "global_nms_selection.json"]
    if any(not path.is_file() for path in required):
        raise RuntimeError("Validation selection files are incomplete.")
    selections = {key: json.loads((VAL / f"{key}_nms_selection.json").read_text(encoding="utf-8")) for key in ("sf", "r5")}
    global_selection = json.loads((VAL / "global_nms_selection.json").read_text(encoding="utf-8"))
    global_distance = int(global_selection["selected_distance"])
    manifest = json.loads((AUDIT / "test_manifest.json").read_text(encoding="utf-8"))
    test_inverse_map = {int(v): k for k, v in load_label_mapping(ROOT / "configs/labels_multitask_plug.yaml").items()}
    family_rows = []
    per_rows = []
    skill_rows = []
    category_rows = []
    suppression_rows = []
    for family, test_paths in manifest["families"].items():
        test_paths = list(test_paths)
        if not test_paths:
            continue
        for short in ("sf", "r5"):
            model, config, model_inverse, checkpoint_hash = load_model(short)
            rows = infer(model, test_paths, ROOT / "configs/labels_multitask_plug.yaml", target_config(config))
            for distance in (0, global_distance):
                aggregate, analyzed = aggregate_test(rows, model_inverse, test_inverse_map, distance)
                condition = short if distance == 0 else f"{short}_nms"
                family_rows.append({"family": family, "condition": condition, "distance_frames": distance, "distance_seconds": distance / 100, "raw_F1@50": aggregate["raw_F1@50"], "refined_F1@50": aggregate["refined_F1@50"], "boundary_F1@33": aggregate["boundary"]["33"]["F1"], "false_peaks": aggregate["false_peaks"], "missed_boundaries": aggregate["missed_boundaries"], "duplicate_peaks": aggregate["duplicate_peaks"], "predicted_peaks": aggregate["predicted_peaks"], "true_boundaries": aggregate["true_boundaries"], "known_seg_mean_IoU": aggregate["known_segmentation"].get("mean_IoU", 0), "known_seg_IoU50_recovery": aggregate["known_segmentation"].get("IoU50_recovery", 0), "known_seg_both_boundaries_correct": aggregate["known_segmentation"].get("both_boundaries_correct", 0), "known_seg_fragmentation_rate": aggregate["known_segmentation"].get("fragmentation_rate", 0), "known_recognition_oracle_macro_F1": aggregate["known_oracle_macro_F1"], "known_recognition_predicted_segment_F1_50": aggregate["known_predicted_segment_F1@50"], "novel_seg_mean_IoU": aggregate["novel_segmentation"].get("mean_IoU", 0), "novel_seg_IoU50_recovery": aggregate["novel_segmentation"].get("IoU50_recovery", 0), "novel_seg_both_boundaries_correct": aggregate["novel_segmentation"].get("both_boundaries_correct", 0), "novel_seg_fragmentation_rate": aggregate["novel_segmentation"].get("fragmentation_rate", 0), "novel_seg_merge_previous_rate": aggregate["novel_segmentation"].get("merge_previous_rate", 0), "novel_seg_merge_next_rate": aggregate["novel_segmentation"].get("merge_next_rate", 0), "family_selected": int(distance == selections[short]["selected_distance"]), "global_selected": int(distance == global_distance), "checkpoint_sha256": checkpoint_hash})
                for tolerance in (5, 10, 20, 33, 50):
                    family_rows[-1].update({f"boundary_precision_{tolerance}": aggregate["boundary"][str(tolerance)]["precision"], f"boundary_recall_{tolerance}": aggregate["boundary"][str(tolerance)]["recall"], f"boundary_F1_{tolerance}": aggregate["boundary"][str(tolerance)]["F1"]})
                for row, v, analysis in analyzed:
                    trajectory = row["entry"].split("/")[-1]
                    write_per_trajectory(OUT / "test" / family / condition / trajectory, row, v, analysis, model_inverse, test_inverse_map, condition, distance)
                    per_rows.append({"family": family, "condition": condition, "distance_frames": distance, "trajectory": row["entry"], "known_mean_IoU": analysis["known"]["mean_IoU"], "novel_mean_IoU": analysis["novel"]["mean_IoU"], "novel_IoU50_recovery": analysis["novel"]["IoU50_recovery"], "known_recognition_F1@50": analysis["known_predicted_F1@50"], "boundary_F1@33": analysis["boundary"]["33"]["F1"], "false_peaks": analysis["boundary"]["33"]["fp"], "missed_boundaries": analysis["boundary"]["33"]["fn"]})
                    for skill in KNOWN + tuple(x for x in FAMILY_NOVEL[family] if x not in KNOWN):
                        truth_segments = [x for x in analysis["truth_named"] if x["label"] == skill]
                        pred_segments = [i for i, name in enumerate(analysis["predicted_names"]) if name == skill]
                        candidates = []
                        for pi in pred_segments:
                            for ti, target in enumerate(truth_segments):
                                score = iou(analysis["predicted"][pi], TemporalInterval(target["start"], target["end"]))
                                if score >= .5:
                                    candidates.append((score, pi, ti))
                        used_p: set[int] = set(); used_t: set[int] = set(); tp = 0
                        for _, pi, ti in sorted(candidates, reverse=True):
                            if pi not in used_p and ti not in used_t:
                                used_p.add(pi); used_t.add(ti); tp += 1
                        p, r, f = prf(tp, len(pred_segments) - tp, len(truth_segments) - tp)
                        skill_rows.append({"family": family, "condition": condition, "distance_frames": distance, "trajectory": row["entry"], "skill": skill, "novel_skill": skill not in KNOWN_SET, "support_segments": len(truth_segments), "segment_precision": p, "segment_recall": r, "segment_F1": f})
                    for category, values in analysis["category_summary"].items():
                        category_rows.append({"family": family, "condition": condition, "distance_frames": distance, "trajectory": row["entry"], "category": category, **values})
                    for suppressed in v["suppressed"]:
                        suppressor = v["suppressor"].get(suppressed, "")
                        suppression_rows.append({"family": family, "condition": condition, "distance_frames": distance, "trajectory": row["entry"], "suppressed_frame": suppressed, "suppressing_frame": suppressor, "distance_to_suppressor": abs(suppressed - suppressor) if suppressor != "" else "", "near_gt_boundary": int(any(abs(suppressed - x) <= 33 for x in boundaries(row["truth"]))), "inside_skill": int(not any(abs(suppressed - x) <= 33 for x in boundaries(row["truth"])))})
    for item in family_rows:
        grouped = [x for x in category_rows if x["family"] == item["family"] and x["condition"] == item["condition"] and x["distance_frames"] == item["distance_frames"]]
        for category, target in (("known->known", "known_known_boundary_recall_33"), ("known->novel", "known_novel_boundary_recall_33"), ("novel->known", "novel_known_boundary_recall_33"), ("novel->novel", "novel_novel_boundary_recall_33")):
            matching = [x for x in grouped if x["category"] == category]
            support = sum(int(x["support"]) for x in matching)
            detected = sum(float(x.get("recall_33", 0.0)) * int(x["support"]) for x in matching)
            item[target] = detected / support if support else 0.0
    table = OUT / "tables"
    write_csv(table / "boundary_metrics.csv", family_rows)
    write_csv(table / "family_macro_summary.csv", family_rows)
    write_csv(table / "per_trajectory_metrics.csv", per_rows)
    write_csv(table / "per_skill_metrics.csv", skill_rows)
    write_csv(table / "boundary_category_metrics.csv", category_rows)
    write_csv(table / "suppression_events.csv", suppression_rows)
    write_csv(table / "known_skill_segmentation.csv", family_rows)
    write_csv(table / "known_skill_oracle_interval_recognition.csv", family_rows)
    write_csv(table / "known_skill_predicted_segment_recognition.csv", family_rows)
    write_csv(table / "novel_skill_segmentation.csv", family_rows)
    with (VAL / "validation_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        write_csv(table / "validation_metrics.csv", list(csv.DictReader(handle)))
    with (VAL / "global_validation_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        write_csv(table / "nms_selection_metrics.csv", list(csv.DictReader(handle)))
    effects = []
    for family in manifest["families"]:
        base = {x["condition"]: x for x in family_rows if x["family"] == family and x["distance_frames"] in (0, global_distance)}
        if not all(x in base for x in ("sf", "r5", "sf_nms", "r5_nms")):
            continue
        for metric in ("known_seg_mean_IoU", "known_seg_IoU50_recovery", "known_recognition_predicted_segment_F1_50", "novel_seg_mean_IoU", "novel_seg_IoU50_recovery", "known_novel_boundary_recall_33", "novel_known_boundary_recall_33", "novel_novel_boundary_recall_33", "false_peaks", "missed_boundaries", "duplicate_peaks"):
            effects.append({"family": family, "metric": metric, "r5_without_nms": base["r5"].get(metric, 0) - base["sf"].get(metric, 0), "nms_on_sf": base["sf_nms"].get(metric, 0) - base["sf"].get(metric, 0), "nms_on_r5": base["r5_nms"].get(metric, 0) - base["r5"].get(metric, 0), "combined": base["r5_nms"].get(metric, 0) - base["sf"].get(metric, 0)})
    write_csv(table / "factorial_effects.csv", effects)
    write_json(OUT / "test_summary.json", {"global_distance": global_distance, "families": manifest["families"], "family_specific_distances": {key: selections[key]["selected_distance"] for key in selections}, "test_metrics_accessed_after_selection": True})
    write_report(family_rows, manifest, selections, global_selection)


def write_report(rows: list[dict[str, Any]], manifest: dict[str, Any], selections: dict[str, Any], global_selection: dict[str, Any]) -> None:
    OUT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    for metric, title in (("novel_seg_mean_IoU", "Novel class-agnostic mean IoU"), ("novel_seg_IoU50_recovery", "Novel IoU>=0.50 recovery"), ("known_seg_mean_IoU", "Known class-agnostic mean IoU"), ("boundary_F1@33", "Boundary F1@33"), ("false_peaks", "False peaks"), ("missed_boundaries", "Missed boundaries"), ("duplicate_peaks", "Duplicate peaks")):
        fig, axis = plt.subplots(figsize=(10, 5))
        for family in sorted({x["family"] for x in rows}):
            values = [x for x in rows if x["family"] == family]
            axis.plot([f"{x['condition']}" for x in values], [x.get(metric, 0) or 0 for x in values], marker="o", label=family)
        axis.set_title(title)
        axis.set_ylabel(metric)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"{metric}.png", dpi=130)
        plt.close(fig)
    extra = [
        ("known_recognition_oracle_macro_F1", "Known oracle recognition macro F1", "known_oracle_recognition_macro_F1.png"),
        ("known_recognition_predicted_segment_F1_50", "Known predicted-segment F1@50", "known_predicted_segment_F1_50.png"),
        ("novel_seg_both_boundaries_correct", "Novel both-boundaries-correct", "novel_both_boundaries_correct.png"),
        ("novel_seg_fragmentation_rate", "Novel fragmentation and merging", "novel_fragmentation_merge_rates.png"),
        ("known_novel_boundary_recall_33", "Known to novel boundary recall +/-33", "known_to_novel_recall_33.png"),
        ("novel_known_boundary_recall_33", "Novel to known boundary recall +/-33", "novel_to_known_recall_33.png"),
        ("novel_novel_boundary_recall_33", "Novel to novel boundary recall +/-33", "novel_to_novel_recall_33.png"),
    ]
    for metric, title, filename in extra:
        fig, axis = plt.subplots(figsize=(10, 5))
        for family in sorted({x["family"] for x in rows}):
            values = [x for x in rows if x["family"] == family]
            axis.plot([x["condition"] for x in values], [x.get(metric, 0) or 0 for x in values], marker="o", label=family)
        axis.set_title(title)
        axis.set_ylabel(metric)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUT / "figures" / filename, dpi=130)
        plt.close(fig)
    for filename, title, metric_list in (
        ("false_missed_duplicate_peaks.png", "False, missed, and duplicate peaks", ("false_peaks", "missed_boundaries", "duplicate_peaks")),
        ("raw_asb_vs_brb_segmentation.png", "Raw ASB diagnostic versus BRB conditions", ("known_seg_mean_IoU", "novel_seg_mean_IoU")),
    ):
        fig, axis = plt.subplots(figsize=(10, 5))
        for metric in metric_list:
            values = [float(np.mean([x.get(metric, 0) or 0 for x in rows if x["condition"] == condition])) for condition in ("sf", "r5", "sf_nms", "r5_nms")]
            axis.plot(("sf", "r5", "sf_nms", "r5_nms"), values, marker="o", label=metric)
        axis.set_title(title)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUT / "figures" / filename, dpi=130)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5))
    for family in sorted({x["family"] for x in rows}):
        values = [x for x in rows if x["family"] == family]
        axis.plot([x["condition"] for x in values], [x["novel_seg_mean_IoU"] for x in values], marker="o", label=family)
    axis.set_title("Factorial effects of r5 target and NMS")
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures/factorial_effects.png", dpi=130)
    plt.close(fig)
    summary = []
    for family in sorted({x["family"] for x in rows}):
        for condition in ("sf", "r5", "sf_nms", "r5_nms"):
            values = [x for x in rows if x["family"] == family and x["condition"] == condition]
            if values:
                summary.append((family, condition, values[0]["novel_seg_mean_IoU"], values[0]["known_seg_mean_IoU"], values[0]["known_recognition_predicted_segment_F1_50"]))
    lines = [
        "# Round 10 PP-only zero-shot novel segmentation",
        "",
        "## Objective and protocol",
        "",
        "The model was trained only on PP trajectories pp1-pp10. Validation was pp11-pp20. The PP-only model-facing ontology is reach, grasp, lift, transport, place, release, retreat; retreat has zero frame support in pp1-pp10 but is retained because it occurs in the PP training corpus. Novel semantic recognition was not evaluated.",
        "",
        "The four primary conditions are single-frame without NMS, hard-window-r5 without NMS, and those same two models with the common validation-selected greedy NMS distance. NMS is inference-only.",
        "",
        f"Family-specific validation selections: SF d={selections['sf']['selected_distance']}, R5 d={selections['r5']['selected_distance']}. Global selected distance: d={global_selection['selected_distance']} frames. Test metrics were accessed only after these files were finalized.",
        "",
        "## Test coverage",
        "",
        json.dumps(manifest["families"], indent=2),
        "",
        "Plug is restricted to independent p1 and p2. p3, po1, and po2 were excluded because their annotation files are incomplete; pull-out generalization is therefore not assessed.",
        "",
        "## Results",
        "",
        "| family | condition | novel mean IoU | novel IoU50 | known mean IoU | known recognition segment F1@50 | boundary F1@33 | false peaks | missed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, condition, novel_iou, known_iou, recognition in summary:
        row = next(x for x in rows if x["family"] == family and x["condition"] == condition)
        lines.append(f"| {family} | {condition} | {novel_iou:.4f} | {row['novel_seg_IoU50_recovery']:.4f} | {known_iou:.4f} | {recognition:.4f} | {row['boundary_F1@33']:.4f} | {row['false_peaks']} | {row['missed_boundaries']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Novel intervals are scored only by temporal boundaries and overlap. Predicted old-class assignments inside novel intervals are not interpreted as novel semantic recognition. The factorial effects and per-skill tables separate boundary-target and NMS effects.",
        "",
        "Raw ASB change-point segmentation is retained as a diagnostic in the per-trajectory artifacts; BRB conditions remain the four primary comparisons.",
        "",
        "## Integrity and limitations",
        "",
        "No pour, wipe, or plug trajectory was used for training, validation, model selection, or NMS selection. Existing Round 8/9 outputs and checkpoints were preserved. Plug conclusions are restricted to p1/p2 and should not be generalized to pull-out trajectories.",
        "",
        "All timelines are under test/ and all comparison figures are under figures/.",
        "",
    ]
    (OUT / "round10_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    if args.phase == "validation":
        run_validation()
    else:
        run_test()
