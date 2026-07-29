#!/usr/bin/env python
"""Evaluate the newly complete wipe/w4 and build a non-destructive w1-w4 summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from asrf.data.dataset import MultiTaskTrajectoryDataset  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.refinement.refine import refine_asrf_predictions  # noqa: E402
from asrf.training.checkpointing import load_checkpoint  # noqa: E402
from asrf.utils.config import load_yaml_config, resolve_repo_path  # noqa: E402
from evaluate_multitask import _oracle_refinement, _record, _summarize, _variant_metrics, _write_trajectory_outputs  # noqa: E402


def _balanced_accuracy(prediction: torch.Tensor, truth: torch.Tensor, number_of_classes: int) -> float:
    recalls = []
    for class_id in range(number_of_classes):
        target = truth == class_id
        if bool(target.any()):
            recalls.append(float(((prediction == class_id) & target).sum()) / int(target.sum()))
    return float(np.mean(recalls)) if recalls else 0.0


def main() -> int:
    config = load_yaml_config("configs/multitask_asrf_train.yaml")
    mapping = load_label_mapping(resolve_repo_path(config["data"]["label_config"]))
    model = ASRFModel.from_config(config).to("cpu")
    model.load_state_dict(load_checkpoint(REPO_ROOT / "outputs/multitask_baseline/best.pt", map_location="cpu")["model_state"])
    model.eval()
    dataset = MultiTaskTrajectoryDataset(config["data"]["dataset_root"], REPO_ROOT / "splits/multitask_test_wipe.txt", resolve_repo_path(config["data"]["label_config"]), expected_height=88, allow_test=True)
    records = []
    for index in range(len(dataset)):
        record = _record(model, dataset, index, torch.device("cpu"))
        official = refine_asrf_predictions(record["asb"].unsqueeze(0), record["brb"].view(1, 1, -1), torch.ones(1, len(record["truth"]), dtype=torch.bool), threshold=0.5)
        calibrated = refine_asrf_predictions(record["asb"].unsqueeze(0), record["brb"].view(1, 1, -1), torch.ones(1, len(record["truth"]), dtype=torch.bool), threshold=0.9)
        oracle_prediction, oracle_boundaries, oracle_intervals, oracle_diagnostics = _oracle_refinement(record["asb"], record["truth"])
        raw_prediction = record["asb"].argmax(dim=0)
        variants = {
            "raw": {"prediction": raw_prediction, **_variant_metrics(raw_prediction, record["truth"], [], record["boundary_targets"])},
            "official": {"prediction": official.refined_labels[0], "refinement": official, **_variant_metrics(official.refined_labels[0], record["truth"], list(official.selected_boundaries[0]), record["boundary_targets"])},
            "calibrated": {"prediction": calibrated.refined_labels[0], "refinement": calibrated, **_variant_metrics(calibrated.refined_labels[0], record["truth"], list(calibrated.selected_boundaries[0]), record["boundary_targets"])},
            "oracle": {"prediction": oracle_prediction, "boundaries": oracle_boundaries, "intervals": oracle_intervals, "diagnostics": oracle_diagnostics, **_variant_metrics(oracle_prediction, record["truth"], oracle_boundaries, record["boundary_targets"])},
        }
        for item in variants.values():
            item["metrics"]["balanced_frame_accuracy"] = _balanced_accuracy(item["prediction"], record["truth"], len(mapping))
        record["variants"] = variants
        records.append(record)
        if Path(str(record["entry"])).name == "w4":
            _write_trajectory_outputs(REPO_ROOT / "outputs/multitask_baseline/test", record, variants, mapping, 0.9)
            w4_record = record

    summary = _summarize(records, mapping, task="wipe")
    summary["thresholds"] = {"official": 0.5, "calibrated": 0.9}
    summary["trajectory_ids"] = [record["entry"] for record in records]
    summary["historical_w1_w3_summary_preserved_at"] = "outputs/multitask_baseline/test/wipe_summary.json"
    output_path = REPO_ROOT / "outputs/multitask_baseline/test/wipe_summary_w1_w4.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REPO_ROOT / "outputs/multitask_baseline/test/wipe/w4/evaluation_round6.json").write_text(json.dumps({"status": "evaluated", "checkpoint": "outputs/multitask_baseline/best.pt", "official_threshold": 0.5, "calibrated_threshold": 0.9, "metrics_path": "outputs/multitask_baseline/test/wipe/w4/metrics.json", "historical_unavailable_record_preserved": "evaluation_unavailable.json"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"trajectory_count": len(records), "w4": {"entry": w4_record["entry"], "T": len(w4_record["truth"]), "variants": {name: item["metrics"] for name, item in w4_record["variants"].items()}}, "summary": str(output_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
