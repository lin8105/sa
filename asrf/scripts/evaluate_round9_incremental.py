"""Evaluate revised Round 9 incremental models and write learning-curve artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")
OUT = ROOT / "outputs/round9_incremental_learning"
SPLITS = ROOT / "splits/round9_incremental"
sys_path = str(ROOT / "src")
import sys

sys.path.insert(0, sys_path)

from asrf.data.dataset import MultiTaskTrajectoryDataset  # noqa: E402
from asrf.data.labels import LabelMapping, load_label_mapping  # noqa: E402
from asrf.evaluation.metrics import boundary_counts, edit_score, labels_to_segments, segmental_f1  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.refinement.refine import refine_asrf_predictions  # noqa: E402
from asrf.training.checkpointing import load_checkpoint, sha256_file  # noqa: E402
from asrf.utils.config import load_yaml_config, resolve_repo_path  # noqa: E402
from asrf.data.ontology import CANONICAL_LABELS  # noqa: E402


SKILLS = CANONICAL_LABELS
TARGET_SKILLS = {"pour": ("pour", "pour_recover"), "wipe": ("wipe",), "plug": ("place", "insert")}
SHARED_SKILLS = ("reach", "grasp", "lift", "transport", "place", "release")
ORDER = (("pour", 3), ("pour", 5), ("wipe", 3), ("wipe", 5), ("plug", 3), ("plug", 5), ("pour", "all"), ("wipe", "all"), ("plug", "all"))


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=jsonable) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def semantic(pred: torch.Tensor, truth: torch.Tensor, n_classes: int) -> dict[str, Any]:
    segments = labels_to_segments(pred)
    truth_segments = labels_to_segments(truth)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (truth.numpy(), pred.numpy()), 1)
    recalls = [float(((pred == c) & (truth == c)).sum()) / int((truth == c).sum()) for c in range(n_classes) if int((truth == c).sum())]
    return {
        "frame_accuracy": float((pred == truth).float().mean()),
        "balanced_frame_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "edit": float(edit_score(pred, truth)),
        "F1@10": float(segmental_f1(pred, truth, 0.10)),
        "F1@25": float(segmental_f1(pred, truth, 0.25)),
        "F1@50": float(segmental_f1(pred, truth, 0.50)),
        "predicted_segment_count": len(segments),
        "true_segment_count": len(truth_segments),
        "confusion_matrix": confusion.tolist(),
    }


def truth_boundaries(record: dict[str, Any], internal: bool = True) -> list[int]:
    values = [int(v) for v in torch.where(record["targets"] > 0.5)[0].tolist()]
    return [v for v in values if not internal or v != 0]


def matched_segment_counts(pred: list[Any], truth: list[Any], class_id: int, overlap: float = 0.5) -> tuple[int, int, int]:
    candidates: list[tuple[float, int, int]] = []
    for pi, p in enumerate(pred):
        if p.label != class_id:
            continue
        for ti, t in enumerate(truth):
            if t.label != class_id:
                continue
            intersection = max(0, min(p.end, t.end) - max(p.start, t.start) + 1)
            union = p.length + t.length - intersection
            iou = intersection / union if union else 0.0
            if iou >= overlap:
                candidates.append((iou, pi, ti))
    candidates.sort(reverse=True)
    used_p: set[int] = set(); used_t: set[int] = set(); tp = 0
    for _, pi, ti in candidates:
        if pi not in used_p and ti not in used_t:
            used_p.add(pi); used_t.add(ti); tp += 1
    predicted = sum(seg.label == class_id for seg in pred)
    actual = sum(seg.label == class_id for seg in truth)
    return tp, predicted - tp, actual - tp


def class_metrics(rows: list[dict[str, Any]], mapping: LabelMapping, variant: str) -> list[dict[str, Any]]:
    result = []
    for class_id, name in sorted(((int(v), k) for k, v in mapping.items())):
        tp = fp = fn = support = 0
        for row in rows:
            prediction = row["variants"][variant]["prediction"]
            truth = row["truth"]
            positive = prediction == class_id; actual = truth == class_id
            tp += int((positive & actual).sum()); fp += int((positive & ~actual).sum()); fn += int((~positive & actual).sum()); support += int(actual.sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        result.append({"variant": variant, "class_id": class_id, "skill": name, "tp": tp, "fp": fp, "fn": fn, "support": support, "precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    return result


def variants(record: dict[str, Any], threshold: float) -> dict[str, Any]:
    asb = record["asb"]; length = len(record["truth"]); mask = torch.ones(1, length, dtype=torch.bool)
    official = refine_asrf_predictions(asb.unsqueeze(0), record["brb"].view(1, 1, -1), mask, threshold=0.5, voting="majority")
    calibrated = refine_asrf_predictions(asb.unsqueeze(0), record["brb"].view(1, 1, -1), mask, threshold=threshold, voting="majority")
    oracle = refine_asrf_predictions(asb.unsqueeze(0), record["brb"].view(1, 1, -1), mask, threshold=0.0, voting="majority")
    oracle_boundaries = truth_boundaries(record, internal=False)
    from asrf.refinement.segments import construct_segments
    from asrf.refinement.majority_vote import _vote_one
    oracle_prediction, _ = _vote_one(asb, construct_segments(oracle_boundaries, length), voting="majority")
    return {
        "raw": {"prediction": asb.argmax(dim=0), "boundaries": []},
        "official": {"prediction": official.refined_labels[0], "boundaries": list(official.selected_boundaries[0])},
        "calibrated": {"prediction": calibrated.refined_labels[0], "boundaries": list(calibrated.selected_boundaries[0])},
        "oracle": {"prediction": oracle_prediction, "boundaries": oracle_boundaries},
    }


@torch.no_grad()
def records(model: ASRFModel, split: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    data = config["data"]
    target_config = {key: data[key] for key in ("boundary_target_mode", "boundary_window_radius", "boundary_gaussian_sigma", "boundary_include_frame_zero", "boundary_include_final_frame")}
    dataset = MultiTaskTrajectoryDataset(DATA, ROOT / split, ROOT / data["label_config"], expected_height=88, allow_test=split.startswith("splits/round9_incremental/test_"), boundary_target_config=target_config)
    result = []
    for index in range(len(dataset)):
        sample = dataset[index]
        output = model(sample["heatmap"].unsqueeze(0), valid_mask=sample["valid_mask"].unsqueeze(0))
        result.append({"entry": sample["trajectory_id"], "truth": sample["labels"].cpu(), "targets": sample["hard_boundary_targets"].cpu(), "asb": output.asb_stage_probabilities[-1][0].cpu(), "brb": output.brb_stage_probabilities[-1][0, 0].cpu()})
    return result


def attach(rows: list[dict[str, Any]], threshold: float, n_classes: int) -> None:
    for row in rows:
        row["variants"] = variants(row, threshold)
        for name, item in row["variants"].items():
            item["semantic"] = semantic(item["prediction"], row["truth"], n_classes)


def calibrate(rows: list[dict[str, Any]]) -> float:
    best = (0.5, -1.0)
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        pooled = {"tp": 0, "fp": 0, "fn": 0}
        f1_values = []
        for row in rows:
            ref = refine_asrf_predictions(row["asb"].unsqueeze(0), row["brb"].view(1, 1, -1), torch.ones(1, len(row["truth"]), dtype=torch.bool), threshold=threshold, voting="majority")
            item = boundary_counts([v for v in ref.selected_boundaries[0] if v != 0], truth_boundaries(row), 33, include_frame0=False)
            for key in pooled: pooled[key] += int(item[key])
            f1_values.append(float(semantic(ref.refined_labels[0], row["truth"], 12)["F1@50"]))
        p = pooled["tp"] / (pooled["tp"] + pooled["fp"]) if pooled["tp"] + pooled["fp"] else 0.0
        r = pooled["tp"] / (pooled["tp"] + pooled["fn"]) if pooled["tp"] + pooled["fn"] else 0.0
        score = 2 * p * r / (p + r) if p + r else 0.0
        candidate = (threshold, score, float(np.mean(f1_values)) if f1_values else 0.0)
        if (candidate[1], candidate[2], candidate[0]) > (best[1], -1.0, best[0]): best = (threshold, candidate[1])
    return float(best[0])


def summarize(rows: list[dict[str, Any]], mapping: LabelMapping, variant: str) -> dict[str, Any]:
    metrics = [row["variants"][variant]["semantic"] for row in rows]
    pooled = {key: float(np.mean([item[key] for item in metrics])) if metrics else 0.0 for key in ("frame_accuracy", "balanced_frame_accuracy", "edit", "F1@10", "F1@25", "F1@50")}
    boundary = {str(tol): {"tp": 0, "fp": 0, "fn": 0} for tol in (10, 20, 33)}
    for row in rows:
        predicted = [v for v in row["variants"][variant]["boundaries"] if v != 0]
        truth = truth_boundaries(row)
        for tol in boundary:
            item = boundary_counts(predicted, truth, int(tol), include_frame0=False)
            for key in ("tp", "fp", "fn"): boundary[tol][key] += int(item[key])
    for item in boundary.values():
        p = item["tp"] / (item["tp"] + item["fp"]) if item["tp"] + item["fp"] else 0.0; r = item["tp"] / (item["tp"] + item["fn"]) if item["tp"] + item["fn"] else 0.0
        item.update({"precision": p, "recall": r, "F1": 2 * p * r / (p + r) if p + r else 0.0})
    correct = sum(int((row["variants"][variant]["prediction"] == row["truth"]).sum()) for row in rows); total = sum(len(row["truth"]) for row in rows)
    confusion = np.sum([np.asarray(item["confusion_matrix"]) for item in metrics], axis=0).tolist() if metrics else []
    return {"trajectory_count": len(rows), "metrics": pooled, "pooled_frame_accuracy": correct / total if total else 0.0, "boundary": boundary, "class_metrics": class_metrics(rows, mapping, variant), "confusion_matrix": confusion}


def support_rows(audit_rows: list[dict[str, Any]], entries: list[str], validation: list[str], test: list[str], family_name: str, size: int | str, target_skills: tuple[str, ...]) -> list[dict[str, Any]]:
    by_id = {str(row["trajectory"]): row for row in audit_rows}; result = []
    for skill in SKILLS:
        train_rows = [by_id[entry] for entry in entries]
        val_rows = [by_id[entry] for entry in validation]
        test_rows = [by_id[entry] for entry in test]
        result.append({"target_family": family_name, "target_trajectory_count": size, "total_training_trajectories": len(entries), "skill": skill, "train_trajectories_with_skill": sum(int(row[f"{skill}_segments"]) > 0 for row in train_rows), "train_segments": sum(int(row[f"{skill}_segments"]) for row in train_rows), "train_frames": sum(int(row[f"{skill}_frames"]) for row in train_rows), "train_duration_s": sum(int(row[f"{skill}_frames"]) for row in train_rows) / 100.0, "validation_segments": sum(int(by_id[entry][f"{skill}_segments"]) for entry in validation), "test_segments": sum(int(by_id[entry][f"{skill}_segments"]) for entry in test), "test_frames": sum(int(by_id[entry][f"{skill}_frames"]) for entry in test), "primary_target_skill": skill in target_skills})
    return result


def per_skill_rows(rows: list[dict[str, Any]], mapping: LabelMapping, support: list[dict[str, Any]], family_name: str, size: int | str) -> list[dict[str, Any]]:
    support_map = {item["skill"]: item for item in support}; result = []
    for skill in SKILLS:
        class_id = mapping[skill]; raw_tp = raw_fp = raw_fn = off_tp = off_fp = off_fn = 0; entry_targets = exit_targets = entry_hits = exit_hits = 0
        raw_frame_tp = raw_frame_fp = raw_frame_fn = off_frame_tp = off_frame_fp = off_frame_fn = 0
        for row in rows:
            truth = labels_to_segments(row["truth"]); raw = labels_to_segments(row["variants"]["raw"]["prediction"]); off = labels_to_segments(row["variants"]["official"]["prediction"])
            a, b, c = matched_segment_counts(raw, truth, class_id); raw_tp += a; raw_fp += b; raw_fn += c
            a, b, c = matched_segment_counts(off, truth, class_id); off_tp += a; off_fp += b; off_fn += c
            raw_positive = row["variants"]["raw"]["prediction"] == class_id; off_positive = row["variants"]["official"]["prediction"] == class_id; truth_positive = row["truth"] == class_id
            raw_frame_tp += int((raw_positive & truth_positive).sum()); raw_frame_fp += int((raw_positive & ~truth_positive).sum()); raw_frame_fn += int((~raw_positive & truth_positive).sum())
            off_frame_tp += int((off_positive & truth_positive).sum()); off_frame_fp += int((off_positive & ~truth_positive).sum()); off_frame_fn += int((~off_positive & truth_positive).sum())
            for index, segment in enumerate(truth):
                if segment.label != class_id: continue
                if index > 0:
                    entry_targets += 1; entry_hits += int(any(abs(p - segment.start) <= 33 for p in row["variants"]["official"]["boundaries"]))
                if index + 1 < len(truth):
                    exit_targets += 1; exit_hits += int(any(abs(p - truth[index + 1].start) <= 33 for p in row["variants"]["official"]["boundaries"]))
        def score(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
            p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
            return p, r, 2 * p * r / (p + r) if p + r else 0.0
        rp, rr, rf = score(raw_tp, raw_fp, raw_fn); op, or_, of = score(off_tp, off_fp, off_fn)
        rfp, rfr, rff = score(raw_frame_tp, raw_frame_fp, raw_frame_fn); ofp, offr, off = score(off_frame_tp, off_frame_fp, off_frame_fn)
        oracle_hits = 0; oracle_support = 0
        for row in rows:
            truth = labels_to_segments(row["truth"]); oracle = labels_to_segments(row["variants"]["oracle"]["prediction"])
            for segment in truth:
                if segment.label == class_id:
                    oracle_support += 1
                    oracle_hits += int(any(item.label == class_id and min(item.end, segment.end) - max(item.start, segment.start) + 1 >= 0.5 * max(item.length, segment.length) for item in oracle))
        result.append({"target_family": family_name, "target_trajectory_count": size, "total_training_trajectories": support_map[skill]["total_training_trajectories"], "skill": skill, "train_segments": support_map[skill]["train_segments"], "train_frames": support_map[skill]["train_frames"], "train_duration_s": support_map[skill]["train_duration_s"], "test_support_segments": support_map[skill]["test_segments"], "raw_precision": rp, "raw_recall": rr, "raw_F1": rf, "official_precision": op, "official_recall": or_, "official_F1": of, "raw_frame_precision": rfp, "raw_frame_recall": rfr, "raw_frame_F1": rff, "official_frame_precision": ofp, "official_frame_recall": offr, "official_frame_F1": off, "oracle_segment_recognition_rate": oracle_hits / oracle_support if oracle_support else 0.0, "entry_boundary_recall_33": entry_hits / entry_targets if entry_targets else 0.0, "exit_boundary_recall_33": exit_hits / exit_targets if exit_targets else 0.0, "primary_target_skill": bool(support_map[skill]["primary_target_skill"])})
    return result


def transition_rows(rows: list[dict[str, Any]], family_name: str, size: int | str) -> list[dict[str, Any]]:
    requested = {"pour": ("transport -> pour", "pour -> pour_recover", "pour_recover -> transport"), "wipe": ("place -> wipe", "wipe -> lift"), "plug": ("transport -> place", "place -> insert", "insert -> release")}[family_name]
    names = {index: name for index, name in enumerate(SKILLS)}
    grouped = {transition: {"support": 0, "detected": 0} for transition in requested}
    for row in rows:
        truth = labels_to_segments(row["truth"])
        peaks = [peak for peak in row["variants"]["official"]["boundaries"] if peak != 0]
        for index in range(len(truth) - 1):
            transition = f"{names[truth[index].label]} -> {names[truth[index + 1].label]}"
            if transition in grouped:
                grouped[transition]["support"] += 1
                grouped[transition]["detected"] += int(any(abs(peak - truth[index + 1].start) <= 33 for peak in peaks))
    return [{"target_family": family_name, "target_trajectory_count": size, "transition": transition, "support": values["support"], "detected": values["detected"], "missed": values["support"] - values["detected"], "boundary_recall_33": values["detected"] / values["support"] if values["support"] else 0.0} for transition, values in grouped.items()]


def evaluate_one(family_name: str, size: int | str, audit_rows: list[dict[str, Any]], mapping: LabelMapping) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    output = OUT / "models" / family_name / f"n{size}"; config = load_yaml_config(output / "config.yaml"); checkpoint = output / "best.pt"
    model = ASRFModel.from_config(config); model.load_state_dict(load_checkpoint(checkpoint, map_location="cpu", expected_ontology=True)["model_state"]); model.eval()
    validation = records(model, "splits/round9_incremental/common_validation.txt", config); threshold = calibrate(validation); attach(validation, threshold, len(mapping))
    test = records(model, f"splits/round9_incremental/test_{family_name}_primary.txt", config); attach(test, threshold, len(mapping))
    val_summary = {variant: summarize(validation, mapping, variant) for variant in ("raw", "official", "calibrated", "oracle")}; test_summary = {variant: summarize(test, mapping, variant) for variant in ("raw", "official", "calibrated", "oracle")}
    write_json(output / "validation_summary.json", {"calibrated_threshold": threshold, **val_summary}); write_json(output / "primary_test_summary.json", {"calibrated_threshold": threshold, **test_summary})
    rows_by_id = {str(row["trajectory"]): row for row in audit_rows}; train_entries = [line.strip() for line in (SPLITS / f"{family_name}_train_{size}_with_base_pp10.txt").read_text().splitlines() if line.strip()]; validation_entries = [line.strip() for line in (SPLITS / "common_validation.txt").read_text().splitlines() if line.strip()]; test_entries = [line.strip() for line in (SPLITS / f"test_{family_name}_primary.txt").read_text().splitlines() if line.strip()]
    support = support_rows(audit_rows, train_entries, validation_entries, test_entries, family_name, size, TARGET_SKILLS[family_name]); per_skill = per_skill_rows(test, mapping, support, family_name, size)
    write_csv(output / "training_support.csv", support); write_csv(output / "per_skill_metrics.csv", per_skill)
    target_f1 = [item["official_F1"] for item in per_skill if item["skill"] in TARGET_SKILLS[family_name]]; official = test_summary["official"]; training_summary = json.loads((output / "training_summary.json").read_text())
    boundary = official["boundary"]["33"]
    task_row = {"target_family": family_name, "target_trajectory_count": size, "total_training_trajectories": len(train_entries), "best_epoch": training_summary.get("best_epoch", 0), "training_duration_s": training_summary.get("elapsed_seconds", 0.0), "raw_accuracy": test_summary["raw"]["metrics"]["frame_accuracy"], "raw_F1_50": test_summary["raw"]["metrics"]["F1@50"], "official_accuracy": official["metrics"]["frame_accuracy"], "official_F1_50": official["metrics"]["F1@50"], "boundary_F1_33": boundary["F1"], "false_peaks": boundary["fp"], "missed_boundaries": boundary["fn"], "macro_target_skill_F1": float(np.mean(target_f1)) if target_f1 else 0.0, "checkpoint_sha256": sha256_file(checkpoint), "calibrated_threshold": threshold}
    return task_row, support, per_skill


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--interim", action="store_true"); args = parser.parse_args()
    mapping = load_label_mapping(ROOT / "configs/labels_multitask_plug.yaml"); audit_rows = list(csv.DictReader((OUT / "data_audit_scan1.csv").open(encoding="utf-8")))
    available = []
    for family_name, size in ORDER:
        if (OUT / "models" / family_name / f"n{size}" / "best.pt").is_file(): available.append((family_name, size))
    if args.interim: available = [item for item in available if item in ORDER[:6]]
    task_rows = []; support_rows_all = []; per_skill_all = []; transition_all = []
    for family_name, size in available:
        task, support, per_skill = evaluate_one(family_name, size, audit_rows, mapping); task_rows.append(task); support_rows_all.extend(support); per_skill_all.extend(per_skill)
        output = OUT / "models" / family_name / f"n{size}"; config = load_yaml_config(output / "config.yaml"); model = ASRFModel.from_config(config); model.load_state_dict(load_checkpoint(output / "best.pt", map_location="cpu", expected_ontology=True)["model_state"]); model.eval(); validation = records(model, "splits/round9_incremental/common_validation.txt", config); threshold = calibrate(validation); test = records(model, f"splits/round9_incremental/test_{family_name}_primary.txt", config); attach(test, threshold, len(mapping)); transition_all.extend(transition_rows(test, family_name, size))
    write_csv(OUT / "task_learning_curve.csv", task_rows); write_csv(OUT / "training_support.csv", support_rows_all); write_csv(OUT / "per_skill_learning_curve.csv", per_skill_all)
    write_csv(OUT / "all_task_learning_curves.csv", task_rows); write_csv(OUT / "target_transition_boundary_metrics.csv", transition_all)
    write_json(OUT / ("interim_evaluation_manifest.json" if args.interim else "evaluation_manifest.json"), {"models": [f"{family}/n{size}" for family, size in available], "official_threshold": 0.5, "calibration_selection": "common validation only", "target_skills": {key: list(value) for key, value in TARGET_SKILLS.items()}, "primary_tests": json.loads((OUT / "test_split_manifest.json").read_text())["primary_test"]})
    print(json.dumps({"evaluated": [f"{family}/n{size}" for family, size in available], "interim": args.interim}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
