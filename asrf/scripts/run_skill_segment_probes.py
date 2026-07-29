#!/usr/bin/env python
"""Train validation-selected, context-free skill-segment classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.probes.oracle_segment_dataset import OracleSegmentDataset, OracleSegmentRecord, build_oracle_segment_dataset  # noqa: E402
from asrf.training.checkpointing import load_checkpoint  # noqa: E402
from asrf.utils.config import load_yaml_config, resolve_repo_path  # noqa: E402


SKILLS = ["reach", "grasp", "lift", "transport", "pour", "pour_recover", "place", "wipe", "retreat"]
CLASS_IDS = list(range(len(SKILLS)))
POOLINGS = ("mean", "mean_max", "mean_max_std", "mean_max_std_first_last_delta")
CS = (0.01, 0.1, 1.0, 10.0)


def _pool(sequence: np.ndarray, pooling: str) -> np.ndarray:
    if sequence.ndim != 2 or sequence.shape[1] == 0:
        raise ValueError(f"Expected [C,L] non-empty sequence, got {sequence.shape}")
    mean = sequence.mean(axis=1)
    if pooling == "mean":
        return mean
    maximum = sequence.max(axis=1)
    if pooling == "mean_max":
        return np.concatenate([mean, maximum])
    std = sequence.std(axis=1)
    if pooling == "mean_max_std":
        return np.concatenate([mean, maximum, std])
    if pooling == "mean_max_std_first_last_delta":
        first = sequence[:, 0]
        last = sequence[:, -1]
        return np.concatenate([mean, maximum, std, first, last, last - first])
    raise ValueError(f"Unknown pooling {pooling!r}")


def raw_citr_features(heatmap: torch.Tensor) -> np.ndarray:
    """Per-row temporal statistics from a cropped [3,88,L] raw heatmap."""
    values = heatmap.detach().cpu().numpy().astype(np.float64, copy=False).reshape(3 * 88, -1)
    derivative = np.diff(values, axis=1) if values.shape[1] > 1 else np.zeros((values.shape[0], 0), dtype=values.dtype)
    blocks = [values.mean(axis=1), values.std(axis=1), values.min(axis=1), values.max(axis=1), np.median(values, axis=1), values[:, -1] - values[:, 0], np.abs(derivative).mean(axis=1) if derivative.shape[1] else np.zeros(values.shape[0]), np.abs(derivative).max(axis=1) if derivative.shape[1] else np.zeros(values.shape[0])]
    return np.concatenate(blocks).astype(np.float32)


def _duration_feature(record: OracleSegmentRecord, enabled: bool) -> np.ndarray:
    return np.asarray([np.log1p(record.duration_frames)], dtype=np.float32) if enabled else np.empty(0, dtype=np.float32)


def _load_splits(config: dict[str, Any]) -> dict[str, OracleSegmentDataset]:
    data = config["data"]
    label_path = resolve_repo_path(data["label_config"])
    root = Path(data["dataset_root"])
    datasets = {
        "train": build_oracle_segment_dataset(root, resolve_repo_path(data["train_split"]), label_path, split_name="train", allow_test=False),
        "validation": build_oracle_segment_dataset(root, resolve_repo_path(data["val_split"]), label_path, split_name="validation", allow_test=False),
    }
    test_datasets: list[OracleSegmentDataset] = []
    for task, split in (("pour", "splits/multitask_test_pour.txt"), ("pp", "splits/multitask_test_pp.txt"), ("wipe", "splits/multitask_test_wipe.txt")):
        path = resolve_repo_path(split)
        if path.is_file() and path.read_text(encoding="utf-8").strip():
            test_datasets.append(build_oracle_segment_dataset(root, path, label_path, split_name="test", allow_test=True))
    records: list[tuple[OracleSegmentRecord, torch.Tensor]] = []
    for dataset in test_datasets:
        records.extend(list(dataset))
    # A combined in-memory dataset is unnecessary; retain a lightweight wrapper.
    datasets["test"] = _InMemorySegments(records)
    return datasets


class _InMemorySegments:
    def __init__(self, items: list[tuple[OracleSegmentRecord, torch.Tensor]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def _extract_features(datasets: dict[str, Any], checkpoint: Path, config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    model = ASRFModel.from_config(config).to("cpu")
    model.load_state_dict(load_checkpoint(checkpoint, map_location="cpu")["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    feature_rows: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    with torch.no_grad():
        for split, dataset in datasets.items():
            rows: list[dict[str, Any]] = []
            for record, crop in dataset:
                if crop.shape[0:2] != (3, 88):
                    raise ValueError(f"Invalid crop shape for {record.trajectory_id}: {tuple(crop.shape)}")
                encoder = model.encoder(crop.unsqueeze(0))[0].cpu().numpy()
                shared = model.feature_extractor(torch.from_numpy(encoder).unsqueeze(0))[0].cpu().numpy()
                rows.append({"record": record, "raw": raw_citr_features(crop), "encoder": {pooling: _pool(encoder, pooling) for pooling in POOLINGS}, "shared": {pooling: _pool(shared, pooling) for pooling in POOLINGS}})
            feature_rows[split] = rows
            counts[split] = len(rows)
    return feature_rows, counts


def _matrix(rows: list[dict[str, Any]], source: str, pooling: str, use_duration: bool) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for row in rows:
        values = row[source] if source == "raw" else row[source][pooling]
        features.append(np.concatenate([values, _duration_feature(row["record"], use_duration)]))
        labels.append(int(row["record"].label_id))
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def _fit(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], source: str, pooling: str, use_duration: bool) -> tuple[StandardScaler, LogisticRegression, dict[str, float]]:
    train_x, train_y = _matrix(train_rows, source, pooling, use_duration)
    val_x, val_y = _matrix(val_rows, source, pooling, use_duration)
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    val_scaled = scaler.transform(val_x)
    best_score: tuple[float, float, float] | None = None
    best_classifier: LogisticRegression | None = None
    best_c = CS[0]
    for c_value in CS:
        # scikit-learn 1.9 selects multinomial handling automatically for
        # multiclass lbfgs; older versions also use this default.
        classifier = LogisticRegression(C=c_value, max_iter=3000, solver="lbfgs", random_state=42)
        classifier.fit(train_scaled, train_y)
        prediction = classifier.predict(val_scaled)
        macro_f1 = float(f1_score(val_y, prediction, labels=CLASS_IDS, average="macro", zero_division=0))
        accuracy = float(np.mean(prediction == val_y)) if len(val_y) else 0.0
        score = (macro_f1, accuracy, -c_value)
        if best_score is None or score > best_score:
            best_score = score
            best_classifier = classifier
            best_c = c_value
    assert best_classifier is not None and best_score is not None
    return scaler, best_classifier, {"C": float(best_c), "validation_macro_f1": best_score[0], "validation_accuracy": best_score[1]}


def _predict(model: tuple[StandardScaler, LogisticRegression], rows: list[dict[str, Any]], source: str, pooling: str, use_duration: bool) -> tuple[np.ndarray, np.ndarray]:
    scaler, classifier = model
    x, _ = _matrix(rows, source, pooling, use_duration)
    probabilities = classifier.predict_proba(scaler.transform(x))
    return classifier.predict(scaler.transform(x)).astype(np.int64), probabilities.max(axis=1)


def _metrics(rows: list[dict[str, Any]], prediction: np.ndarray, confidence: np.ndarray, probe_type: str, use_duration: bool, *, split: str, names: list[str]) -> tuple[list[dict[str, Any]], np.ndarray]:
    truth = np.asarray([row["record"].label_id for row in rows], dtype=np.int64)
    matrix = confusion_matrix(truth, prediction, labels=CLASS_IDS)
    precision, recall, f1, support = precision_recall_fscore_support(truth, prediction, labels=CLASS_IDS, zero_division=0)
    class_rows: list[dict[str, Any]] = []
    for class_id, name in enumerate(names):
        wrong = Counter(int(prediction[index]) for index, value in enumerate(truth) if value == class_id and prediction[index] != class_id)
        durations = [row["record"].duration_frames for row in rows if row["record"].label_id == class_id]
        class_rows.append({"probe_type": probe_type, "uses_duration": use_duration, "split": split, "skill": name, "support": int(support[class_id]), "correct": int(matrix[class_id, class_id]), "segment_recognition_rate": float(recall[class_id]), "precision": float(precision[class_id]), "recall": float(recall[class_id]), "F1": float(f1[class_id]), "most_common_wrong_prediction": names[wrong.most_common(1)[0][0]] if wrong else "", "wrong_prediction_count": int(sum(wrong.values())), "mean_duration_frames": float(np.mean(durations)) if durations else 0.0, "median_duration_frames": float(np.median(durations)) if durations else 0.0, "minimum_duration_frames": int(min(durations)) if durations else 0, "maximum_duration_frames": int(max(durations)) if durations else 0})
    class_rows.insert(0, {"probe_type": probe_type, "uses_duration": use_duration, "split": split, "overall_segment_accuracy": float(np.mean(truth == prediction)) if len(truth) else 0.0, "macro_segment_accuracy": float(np.mean(recall[support > 0])) if np.any(support > 0) else 0.0, "macro_f1": float(np.mean(f1[support > 0])) if np.any(support > 0) else 0.0, "segment_count": len(rows), "confusion_matrix": matrix.tolist()})
    return class_rows, matrix


def _with_support_counts(row: dict[str, Any], train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    skill_id = SKILLS.index(str(row["skill"]))
    train_count = sum(int(item["record"].label_id == skill_id) for item in train_rows)
    validation_count = sum(int(item["record"].label_id == skill_id) for item in val_rows)
    return row | {"train_segments": train_count, "validation_segments": validation_count, "test_segments": int(row["support"]), "correct_test_segments": int(row["correct"])}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion(path: Path, matrix: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(CLASS_IDS, SKILLS, rotation=45, ha="right")
    axis.set_yticks(CLASS_IDS, SKILLS)
    axis.set_xlabel("predicted canonical skill")
    axis.set_ylabel("ground-truth canonical skill")
    axis.set_title(title)
    for i in CLASS_IDS:
        for j in CLASS_IDS:
            if matrix[i, j]:
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_results(output: Path, primary: dict[str, dict[str, Any]], duration_rows: list[dict[str, Any]], per_task: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(SKILLS))
    width = 0.25
    figure, axis = plt.subplots(figsize=(14, 6))
    for offset, source in enumerate(("raw_citr", "heatmap_encoder", "shared_features")):
        rows = primary[source]["class_rows"]
        values = [next(row["segment_recognition_rate"] for row in rows if row.get("skill") == skill) for skill in SKILLS]
        axis.bar(x + (offset - 1) * width, values, width, label=source)
    axis.set_xticks(x, SKILLS, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("test segment recognition rate (recall)")
    axis.set_title("Context-free full skill-segment recognition")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "per_skill_recognition_rates.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for axis, source in zip(axes, ("raw_citr", "heatmap_encoder", "shared_features")):
        values_by_duration = {}
        for enabled in (False, True):
            item = next(row for row in duration_rows if row["probe_type"] == source and bool(row["uses_duration"]) == enabled and row["selection_role"] == "best_for_duration")
            values_by_duration[enabled] = [next(r["segment_recognition_rate"] for r in item["class_rows"] if r.get("skill") == skill) for skill in SKILLS]
        axis.bar(x - width / 2, values_by_duration[False], width, label="without log duration")
        axis.bar(x + width / 2, values_by_duration[True], width, label="with log duration")
        axis.set_title(source)
        axis.set_xticks(x, SKILLS, rotation=65, ha="right")
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("segment recognition rate")
    axes[-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "duration_effect.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 6))
    task_rows = [row for row in per_task if row.get("selection_role") == "primary" and row.get("skill") in {"reach", "grasp", "lift", "transport", "place"}]
    task_names = ["pour", "pp", "wipe"]
    positions = np.arange(len(task_names) * 5)
    for source_index, source in enumerate(("raw_citr", "heatmap_encoder", "shared_features")):
        values = []
        for task in task_names:
            for skill in ("reach", "grasp", "lift", "transport", "place"):
                matching = [r for r in task_rows if r["probe_type"] == source and r["task"] == task and r["skill"] == skill]
                values.append(matching[0]["segment_recognition_rate"] if matching else 0.0)
        axis.plot(positions, values, marker="o", label=source)
    axis.set_xticks(positions, [f"{task}\n{skill}" for task in task_names for skill in ("reach", "grasp", "lift", "transport", "place")], rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("segment recognition rate")
    axis.set_title("Shared-skill rates by task")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "per_task_skill_rates.png", dpi=140)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multitask_asrf_train.yaml")
    parser.add_argument("--checkpoint", default="outputs/multitask_baseline/best.pt")
    parser.add_argument("--output", default="outputs/skill_segment_probe")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    mapping = load_label_mapping(resolve_repo_path(config["data"]["label_config"]))
    if list(sorted(mapping, key=mapping.get)) != SKILLS:
        raise ValueError("The segment probes require the canonical nine-class order.")
    datasets = _load_splits(config)
    feature_rows, split_counts = _extract_features(datasets, resolve_repo_path(args.checkpoint), config)
    output = resolve_repo_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    w4_included = any(row["record"].trajectory_id == "test/wipe/w4" for row in feature_rows["test"])
    (output / "dataset_summary.json").write_text(json.dumps({"segment_counts": split_counts, "test_source_splits": ["multitask_test_pour", "multitask_test_pp", "multitask_test_wipe"], "w4": "included with 18 validated canonical segments" if w4_included else "excluded because its annotation is invalid", "features": "each sample was cropped before encoder invocation; no task or sequence metadata enters features"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    train_rows, val_rows, test_rows = feature_rows["train"], feature_rows["validation"], feature_rows["test"]
    all_result_rows: list[dict[str, Any]] = []
    per_skill_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    primary: dict[str, dict[str, Any]] = {}
    duration_results: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}

    for source, probe_name in (("raw", "raw_citr"), ("encoder", "heatmap_encoder"), ("shared", "shared_features")):
        variants: list[dict[str, Any]] = []
        for use_duration in (False, True):
            poolings = ("raw_statistics",) if source == "raw" else POOLINGS
            for pooling in poolings:
                model, classifier, val_metrics = _fit(train_rows, val_rows, source, "mean" if source == "raw" else pooling, use_duration)
                val_pred, _ = _predict((model, classifier), val_rows, source, "mean" if source == "raw" else pooling, use_duration)
                test_pred, test_conf = _predict((model, classifier), test_rows, source, "mean" if source == "raw" else pooling, use_duration)
                val_truth = np.asarray([row["record"].label_id for row in val_rows])
                test_truth = np.asarray([row["record"].label_id for row in test_rows])
                test_macro_f1 = float(f1_score(test_truth, test_pred, labels=CLASS_IDS, average="macro", zero_division=0))
                variants.append({"source": source, "probe_name": probe_name, "uses_duration": use_duration, "pooling": pooling, "model": (model, classifier), "val_macro_f1": val_metrics["validation_macro_f1"], "val_accuracy": val_metrics["validation_accuracy"], "test_macro_f1": test_macro_f1, "test_accuracy": float(np.mean(test_truth == test_pred)), "test_pred": test_pred, "test_conf": test_conf})
            best_for_duration = max([v for v in variants if v["uses_duration"] == use_duration], key=lambda v: (v["val_macro_f1"], v["val_accuracy"], -CS.index(v["model"][1].C)))
            class_rows, matrix = _metrics(test_rows, best_for_duration["test_pred"], best_for_duration["test_conf"], probe_name, use_duration, split="test", names=SKILLS)
            duration_result = {"probe_type": probe_name, "uses_duration": use_duration, "selection_role": "best_for_duration", "pooling": best_for_duration["pooling"], "C": best_for_duration["model"][1].C, "validation_macro_f1": best_for_duration["val_macro_f1"], "validation_accuracy": best_for_duration["val_accuracy"], "test_macro_f1": best_for_duration["test_macro_f1"], "test_accuracy": best_for_duration["test_accuracy"], "class_rows": class_rows}
            duration_results.append(duration_result)
            all_result_rows.append({key: value for key, value in duration_result.items() if key != "class_rows"})
            for row in class_rows:
                if "skill" in row:
                    per_skill_rows.append(_with_support_counts(row, train_rows, val_rows) | {"pooling": best_for_duration["pooling"], "selection_role": "best_for_duration"})
            for index, (test_row, prediction, confidence) in enumerate(zip(test_rows, best_for_duration["test_pred"], best_for_duration["test_conf"])):
                record = test_row["record"]
                prediction_rows.append({"task": record.task, "trajectory_id": record.trajectory_id, "segment_index": record.segment_index, "start_frame": record.start_frame, "end_frame": record.end_frame, "duration_frames": record.duration_frames, "ground_truth_skill": record.label_name, "predicted_skill": SKILLS[int(prediction)], "correct": bool(int(prediction) == record.label_id), "confidence": float(confidence), "probe_type": probe_name, "uses_duration": use_duration, "pooling": best_for_duration["pooling"]})
            for task in ("pour", "pp", "wipe"):
                task_subset = [(row, pred, conf) for row, pred, conf in zip(test_rows, best_for_duration["test_pred"], best_for_duration["test_conf"]) if row["record"].task == task]
                if not task_subset:
                    continue
                task_rows_data = [item[0] for item in task_subset]
                task_pred = np.asarray([item[1] for item in task_subset])
                task_conf = np.asarray([item[2] for item in task_subset])
                task_class_rows, _ = _metrics(task_rows_data, task_pred, task_conf, probe_name, use_duration, split="test", names=SKILLS)
                for row in task_class_rows:
                    if "skill" in row:
                        per_task_rows.append(row | {"task": task, "pooling": best_for_duration["pooling"], "selection_role": "best_for_duration"})

        primary_variant = max(duration_results[-2:], key=lambda v: (v["validation_macro_f1"], v["validation_accuracy"], -int(v["uses_duration"])))
        primary_variant = dict(primary_variant)
        primary_variant["selection_role"] = "primary"
        primary[probe_name] = primary_variant
        for row in primary_variant["class_rows"]:
            if "skill" in row:
                per_skill_rows.append(_with_support_counts(row, train_rows, val_rows) | {"pooling": primary_variant["pooling"], "selection_role": "primary"})
        chosen_variant = next(v for v in variants if v["uses_duration"] == primary_variant["uses_duration"] and v["pooling"] == primary_variant["pooling"])
        for task in ("pour", "pp", "wipe"):
            task_subset = [(row, pred, conf) for row, pred, conf in zip(test_rows, chosen_variant["test_pred"], chosen_variant["test_conf"]) if row["record"].task == task]
            if task_subset:
                task_rows_data = [item[0] for item in task_subset]
                task_pred = np.asarray([item[1] for item in task_subset])
                task_conf = np.asarray([item[2] for item in task_subset])
                task_class_rows, _ = _metrics(task_rows_data, task_pred, task_conf, probe_name, bool(primary_variant["uses_duration"]), split="test", names=SKILLS)
                for row in task_class_rows:
                    if "skill" in row:
                        per_task_rows.append(row | {"task": task, "pooling": primary_variant["pooling"], "selection_role": "primary"})
        _, primary_matrix = _metrics(test_rows, next(v["test_pred"] for v in variants if v["uses_duration"] == primary_variant["uses_duration"] and v["pooling"] == primary_variant["pooling"]), np.zeros(len(test_rows)), probe_name, bool(primary_variant["uses_duration"]), split="test", names=SKILLS)
        matrices[probe_name] = primary_matrix

    # Ensure exactly one canonical primary row per skill and probe in the main table.
    _write_csv(output / "per_skill_segment_recognition.csv", per_skill_rows)
    _write_csv(output / "all_probe_results.csv", all_result_rows)
    _write_csv(output / "test_segment_predictions.csv", prediction_rows)
    _write_csv(output / "per_skill_per_task.csv", per_task_rows)
    for probe_name, matrix in matrices.items():
        _plot_confusion(output / f"confusion_{probe_name}.png", matrix, f"{probe_name}: canonical test-segment confusion")
    _plot_results(output, primary, duration_results, per_task_rows)
    summary = {"primary": {name: {key: value for key, value in item.items() if key != "class_rows"} for name, item in primary.items()}, "segment_counts": split_counts, "test_predictions": len(test_rows) * 6, "test_selection_used": False, "w4_included": w4_included}
    (output / "probe_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
