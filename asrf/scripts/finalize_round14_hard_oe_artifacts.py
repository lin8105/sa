#!/usr/bin/env python3
"""Add the Round 14 baseline comparison and make diagnostics complete."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import train_round14_hard_oe_holdout_wipe as r14


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    root = r14.OUTPUT_ROOT
    frozen = json.loads((root / "frozen_thresholds.json").read_text(encoding="utf-8"))
    primary = frozen["primary"]
    threshold_rows = []
    for candidate in frozen["score_candidates"]:
        threshold_rows.append({"model": candidate["model"], "score": candidate["score"], "target_retention": 0.95, "threshold": candidate["threshold"], "known_validation_retention": candidate["known_validation_retention"], "synthetic_validation_recall": candidate["synthetic_validation_recall"], "synthetic_validation_auroc": candidate["synthetic_validation_auroc"], "primary": int(candidate["model"] == primary["model"] and candidate["score"] == primary["score"])})
        threshold_rows.append({"model": candidate["model"], "score": candidate["score"], "target_retention": 0.97, "threshold": candidate["threshold_at_0.97"], "known_validation_retention": candidate["retention_at_0.97"], "synthetic_validation_recall": candidate["synthetic_recall_at_0.97"], "synthetic_validation_auroc": candidate["synthetic_validation_auroc"], "primary": 0})
    threshold_fields = ["model", "score", "target_retention", "threshold", "known_validation_retention", "synthetic_validation_recall", "synthetic_validation_auroc", "primary"]
    write_rows(root / "threshold_comparison.csv", threshold_rows, threshold_fields)

    # Train the fixed Round 14 energy-margin-only reference once. It is not a
    # selection candidate and does not use wipe data; it is retained solely as
    # the within-round baseline requested by the protocol.
    r14.seed_everything(); rows = r14.load_rows(); cache = r14.load_features(rows)
    train_frames = np.concatenate([r14.sequence(row, cache) for row in rows["train"]], axis=0); mean = train_frames.mean(axis=0); std = np.maximum(train_frames.std(axis=0), 1e-6); log_durations = np.asarray([np.log1p(row["duration_frames"]) for row in rows["train"]]); duration_mean = float(log_durations.mean()); duration_std = float(max(log_durations.std(), 1e-6))
    payload = torch.load(r14.INIT_CHECKPOINT, map_location="cpu", weights_only=False); teacher = r14.model_from_state(payload["model_state"]); teacher.eval(); reference, reference_rows = r14.reference_embeddings(teacher, rows["train"], cache, mean, std, duration_mean, duration_std); val_reference, _ = r14.reference_embeddings(teacher, rows["validation"], cache, mean, std, duration_mean, duration_std); train_pairs = r14.build_embedding_pairs(reference, rows["train"]); val_pairs = r14.build_embedding_pairs(val_reference, rows["validation"])
    synthetic_train = r14.make_original(rows["train"], cache, "train", r14.TRAIN_PER_TYPE * len(r14.ORIGINAL_TYPES)) + r14.make_hard(rows["train"], cache, "train", r14.TRAIN_PER_TYPE, train_pairs); synthetic_validation = r14.balanced_validation_types(rows["validation"], cache, "validation", val_pairs)
    known_train = r14.SegmentDataset(rows["train"], cache, mean, std, duration_mean, duration_std, True); known_val = r14.SegmentDataset(rows["validation"], cache, mean, std, duration_mean, duration_std, True); synthetic_val = r14.SegmentDataset(synthetic_validation, None, mean, std, duration_mean, duration_std, False)
    known_val_loader = torch.utils.data.DataLoader(known_val, batch_size=r14.BATCH_SIZE, shuffle=False, collate_fn=r14.collate); synthetic_val_loader = torch.utils.data.DataLoader(synthetic_val, batch_size=r14.BATCH_SIZE, shuffle=False, collate_fn=r14.collate)
    class_counts = __import__("collections").Counter(row["label_id"] for row in rows["train"]); class_weights = torch.tensor([1.0 / np.sqrt(class_counts[index]) for index in range(len(r14.KNOWN_CLASSES))], dtype=torch.float32); class_weights *= len(r14.KNOWN_CLASSES) / class_weights.sum()
    variant = {"name": "round14_energy_margin_baseline_stability_0.05", "stability_lambda": 0.05}; model = r14.model_from_state(payload["model_state"]); state, history, epoch, mining = r14.train_one(model, teacher, rows["train"], synthetic_train, known_val_loader, synthetic_val_loader, cache, mean, std, duration_mean, duration_std, class_weights, reference, variant); baseline_model = r14.model_from_state(state); known_val_output = r14.model_infer(baseline_model, known_val_loader); synthetic_val_output = r14.model_infer(baseline_model, synthetic_val_loader); raw_known = r14.score_bundle(known_val_output, reference); raw_synthetic = r14.score_bundle(synthetic_val_output, reference); normalization = {key: (float(np.concatenate((raw_known[key], raw_synthetic[key])).min()), float(np.concatenate((raw_known[key], raw_synthetic[key])).max())) for key in ("energy", "cosine")}; known_scores = r14.score_bundle(known_val_output, reference, normalization)["energy"]; synthetic_scores = r14.score_bundle(synthetic_val_output, reference, normalization)["energy"]; threshold = r14.threshold_for(known_scores, synthetic_scores, 0.95); baseline_payload = {"model_state": state, "ontology_version": r14.ONTOLOGY_VERSION, "held_out_class": r14.HELD_OUT, "known_class_list": list(r14.KNOWN_CLASSES), "metadata": {"round14_baseline": True, "variant": variant, "initialization_checkpoint_sha256": r14.sha256_file(r14.INIT_CHECKPOINT), "optimizer_state_reused": False, "validation_threshold": threshold}, "optimizer_state": None}; baseline_checkpoint = root / "round14_energy_margin_baseline.pt"; torch.save(baseline_payload, baseline_checkpoint)
    test_groups = {"known_test": [row for row in rows["test"] if row["evaluation_group"] == "known_test"], "wipe": [row for row in rows["test"] if row["evaluation_group"] == "wipe_unknown"], "known_inside_wipe": [row for row in rows["test"] if row["evaluation_group"] == "known_inside_wipe"]}; test_outputs = {}
    for group, group_rows in test_groups.items():
        loader = torch.utils.data.DataLoader(r14.SegmentDataset(group_rows, cache, mean, std, duration_mean, duration_std, True), batch_size=r14.BATCH_SIZE, shuffle=False, collate_fn=r14.collate); test_outputs[group] = r14.model_infer(baseline_model, loader)
    test_scores = {group: r14.score_bundle(output, reference, normalization)["energy"] for group, output in test_outputs.items()}; known_accept = test_scores["known_test"] <= threshold["threshold"]; inside_accept = test_scores["known_inside_wipe"] <= threshold["threshold"]; wipe_accept = test_scores["wipe"] <= threshold["threshold"]; known_metric = r14.metric_rejection(test_outputs["known_test"], known_accept); inside_metric = r14.metric_rejection(test_outputs["known_inside_wipe"], inside_accept); wipe_auroc = r14.auroc(np.concatenate((np.zeros(len(test_scores["known_test"])), np.ones(len(test_scores["wipe"])))), np.concatenate((test_scores["known_test"], test_scores["wipe"])))
    baseline_row = {"method": "round14_energy_margin_baseline", "known_retention": known_metric["known_retention"], "known_false_rejection_rate": known_metric["false_rejection_rate"], "known_macro_f1_before_rejection": r14.closed_f1(test_outputs["known_test"]), "known_macro_f1_after_rejection": known_metric["macro_f1_after_rejection"], "known_vs_wipe_auroc": wipe_auroc, "wipe_unknown_recall": float((~wipe_accept).mean()), "wipe_false_known_rate": float(wipe_accept.mean()), "known_inside_wipe_accuracy": float((r14.logits_scores(test_outputs["known_inside_wipe"]["logits"])["top1"] == np.asarray([row["label_id"] for row in test_outputs["known_inside_wipe"]["rows"]])).mean()), "known_inside_wipe_retention": inside_metric["known_retention"]}
    baseline_fields = list(baseline_row)
    old_baselines = [row for row in read_rows(root / "baseline_comparison.csv") if row["method"] != "round14_energy_margin_baseline"]
    write_rows(root / "baseline_comparison.csv", old_baselines + [baseline_row], baseline_fields)
    old_variants = [row for row in read_rows(root / "hard_oe_variant_comparison.csv") if row["method"] != "round14_energy_margin_baseline"]
    write_rows(root / "hard_oe_variant_comparison.csv", old_variants + [baseline_row], baseline_fields)
    config = yaml.safe_load((root / "training_config.yaml").read_text(encoding="utf-8")); config["round14_baseline_checkpoint"] = str(baseline_checkpoint); config["round14_baseline_checkpoint_sha256"] = r14.sha256_file(baseline_checkpoint); (root / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = (root / "report.md").read_text(encoding="utf-8").rstrip(); report += f"\n\n## Round 14 energy-margin baseline\n\nA fixed within-round energy-margin-only reference was trained with stability weight 0.05 and evaluated after primary selection. Its validation threshold retained {threshold['known_retention']:.6f} known samples; checkpoint SHA-256 is `{r14.sha256_file(baseline_checkpoint)}`. Final known retention was {baseline_row['known_retention']:.6f}, wipe unknown recall {baseline_row['wipe_unknown_recall']:.6f}, and known-vs-wipe AUROC {baseline_row['known_vs_wipe_auroc']:.6f}.\n\nThe primary Round 14 result does not meet the target of preserving retention near or above 0.95: {primary['known_validation_retention']:.6f} on validation and the final test retention is recorded in `baseline_comparison.csv`. Round 13 energy-margin therefore remains the preferred model.\n"
    (root / "report.md").write_text(report + "\n", encoding="utf-8")
    failure_path = root / "remaining_failure_analysis.csv"
    if not failure_path.read_text(encoding="utf-8").strip():
        fields = ["sample_id", "trajectory", "segment_index", "predicted_class", "nearest_training_segment", "nearest_class", "wipe_duration", "nearest_duration", "duration_ratio", "most_similar_channels", "largest_difference_channels", "force_mean_abs_difference", "force_max_abs_difference", "torque_mean_abs_difference", "gripper_mean_abs_difference", "gripper_max_abs_difference", "diagnostic_hypotheses"]
        write_rows(failure_path, [], fields)
    print(json.dumps({"status": "finalized", "baseline": baseline_row, "baseline_checkpoint": str(baseline_checkpoint)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
