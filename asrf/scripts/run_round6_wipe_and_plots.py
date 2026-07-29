#!/usr/bin/env python
"""Regenerate selected canonical-label plots without changing frozen models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


def main() -> int:
    config = load_yaml_config("configs/multitask_asrf_train.yaml")
    mapping = load_label_mapping(resolve_repo_path(config["data"]["label_config"]))
    model = ASRFModel.from_config(config).to("cpu")
    model.load_state_dict(load_checkpoint(REPO_ROOT / "outputs/multitask_baseline/best.pt", map_location="cpu")["model_state"])
    model.eval()
    output_root = REPO_ROOT / "outputs/multitask_baseline"

    selected = {"wipe": "test/wipe/w1", "pour": "test/pour/p1", "pp": "test/pp/pp_c1"}
    records_by_task: dict[str, list[dict[str, object]]] = {}
    for task, entry in selected.items():
        split = REPO_ROOT / "outputs/round6_diagnostics" / f"round6_{task}.txt"
        split.write_text(entry + "\n", encoding="utf-8")
        dataset = MultiTaskTrajectoryDataset(config["data"]["dataset_root"], split, resolve_repo_path(config["data"]["label_config"]), expected_height=88, allow_test=True)
        record = _record(model, dataset, 0, torch.device("cpu"))
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
        record["variants"] = variants
        records_by_task[task] = [record]
        _write_trajectory_outputs(output_root / "test", record, variants, mapping, 0.9)

    previous_path = output_root / "test/wipe_summary.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    w4 = json.loads((REPO_ROOT / "outputs/round6_diagnostics/w4_stability.json").read_text(encoding="utf-8"))
    aggregate = {
        "status": "w1_w3_validated_w4_excluded",
        "historical_w1_w3_preserved": True,
        "historical_w1_w3_summary": previous,
        "w4": {"status": "not_evaluated", "reason": "all 18 labels are blank; canonical ground truth unavailable", "stability": w4},
        "valid_trajectory_count": int(previous.get("trajectory_count", 0)),
        "note": "No w4 metric is fabricated or merged into the numerical w1-w3 aggregate.",
    }
    (output_root / "test/wipe_summary_w1_w4.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"regenerated": list(selected), "w4_status": "excluded_invalid_blank_labels"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
