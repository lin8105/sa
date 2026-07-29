"""Create deterministic train/validation/test split files and statistics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asrf.data.dataset import MultiTaskTrajectoryDataset, load_trajectory_sample, read_split_file
from asrf.data.labels import load_label_mapping
from asrf.losses.classification import collect_statistics_for_entries
from asrf.utils.config import resolve_repo_path


TRAIN_ENTRIES = (
    [f"train/pour/p{i}" for i in range(1, 13)]
    + [f"train/pick and place/pp{i}" for i in range(1, 21)]
    + ["train/pick and place/pp26", "train/pick and place/pp27"]
    + [f"train/wipe/w{i}" for i in range(1, 7)]
)
VAL_ENTRIES = (
    [f"train/pour/p{i}" for i in range(13, 17)]
    + [f"train/pick and place/pp{i}" for i in range(21, 26)]
    + ["train/pick and place/pp28"]
    + [f"train/wipe/w{i}" for i in range(7, 10)]
)
TEST_ENTRIES = {
    "pour": [f"test/pour/p{i}" for i in range(1, 6)],
    "pp": [f"test/pp/pp_c{i}" for i in range(1, 4)],
    "wipe": [f"test/wipe/w{i}" for i in range(1, 4)],
}


def _task(entry: str) -> str:
    value = Path(entry).parts[1]
    return "pick_and_place" if value == "pick and place" else value


def _stats(dataset_root: Path, entries: list[str], label_path: Path, mapping: dict[str, int]) -> dict[str, object]:
    class_names = sorted(mapping, key=mapping.get)
    class_frames = Counter()
    class_segments = Counter()
    trajectory_coverage: dict[str, int] = Counter()
    task_counts: Counter[str] = Counter()
    transition_counts: Counter[tuple[str, str]] = Counter()
    frames = 0
    segments = 0
    for entry in entries:
        sample = load_trajectory_sample(dataset_root / entry, mapping)
        task_counts[_task(entry)] += 1
        labels = sample["labels"].tolist()
        frames += len(labels)
        for value in labels:
            class_frames[class_names[int(value)]] += 1
        raw = [str(row["label"]) for row in sample["segments"]]
        canonical = [mapping.aliases.get(label, label) for label in raw]
        collapsed: list[str] = []
        for label in canonical:
            if not collapsed or collapsed[-1] != label:
                collapsed.append(label)
        segments += len(canonical)
        for label in canonical:
            class_segments[label] += 1
        for label in set(canonical):
            trajectory_coverage[label] += 1
        transition_counts.update(zip(collapsed, collapsed[1:]))
    stats = collect_statistics_for_entries(dataset_root, _write_temp_split(entries), mapping)
    return {
        "entries": entries, "trajectory_count": len(entries), "total_frames": frames, "total_segments": segments,
        "class_frames": dict(class_frames), "class_segments": dict(class_segments), "trajectories_containing_class": dict(trajectory_coverage),
        "task_counts": dict(task_counts), "transition_counts": {f"{a} -> {b}": count for (a, b), count in sorted(transition_counts.items())},
        "boundary_positive_count": stats.boundary_positive_count, "boundary_negative_count": stats.boundary_negative_count,
        "boundary_positive_ratio": stats.boundary_positive_ratio, "boundary_positive_weight": stats.boundary_positive_weight,
        "class_weights": {name: float(stats.class_weights[mapping[name]]) for name in class_names},
        "class_frequencies": {name: float(stats.class_frequencies[mapping[name]]) for name in class_names},
        "median_frequency": float(stats.median_frequency),
    }


_TEMP_SPLIT: Path | None = None


def _write_temp_split(entries: list[str]) -> Path:
    global _TEMP_SPLIT
    if _TEMP_SPLIT is None:
        _TEMP_SPLIT = Path("outputs/multitask_baseline/.statistics_split.txt").resolve()
        _TEMP_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    _TEMP_SPLIT.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return _TEMP_SPLIT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data", type=Path)
    parser.add_argument("--label-config", default="configs/labels_multitask.yaml", type=Path)
    parser.add_argument("--output-dir", default="outputs/multitask_baseline", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root.resolve()
    label_path = resolve_repo_path(args.label_config)
    mapping = load_label_mapping(label_path)
    if set(mapping) != {"reach", "grasp", "lift", "transport", "pour", "pour_recover", "place", "wipe", "retreat"}:
        raise SystemExit("Unexpected multi-task ontology")
    all_train = set(TRAIN_ENTRIES) | set(VAL_ENTRIES)
    if len(all_train) != len(TRAIN_ENTRIES) + len(VAL_ENTRIES):
        raise SystemExit("Train/validation overlap")
    train_dataset = MultiTaskTrajectoryDataset(dataset_root, _write_temp_split(TRAIN_ENTRIES), label_path, allow_test=False)
    val_dataset = MultiTaskTrajectoryDataset(dataset_root, _write_temp_split(VAL_ENTRIES), label_path, allow_test=False)
    if set(train_dataset.entries) & set(val_dataset.entries):
        raise SystemExit("Train/validation overlap")
    for entry in TRAIN_ENTRIES + VAL_ENTRIES:
        if entry.startswith("test/"):
            raise SystemExit("Test entry found in train/validation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / "splits/multitask_train.txt").write_text("\n".join(TRAIN_ENTRIES) + "\n", encoding="utf-8")
    (repo_root / "splits/multitask_val.txt").write_text("\n".join(VAL_ENTRIES) + "\n", encoding="utf-8")
    test_paths: dict[str, list[str]] = {}
    for task, entries in TEST_ENTRIES.items():
        valid_entries = []
        for entry in entries:
            path = dataset_root / entry
            if not path.is_dir():
                continue
            sample = load_trajectory_sample(path, mapping)
            if len(sample["labels"]) and all(int(value) in range(len(mapping)) for value in sample["labels"].tolist()):
                valid_entries.append(entry)
        test_paths[task] = valid_entries
        (repo_root / f"splits/multitask_test_{task}.txt").write_text("\n".join(valid_entries) + ("\n" if valid_entries else ""), encoding="utf-8")
    all_test = [entry for task in ("pour", "pp", "wipe") for entry in test_paths[task]]
    (repo_root / "splits/multitask_test_all.txt").write_text("\n".join(all_test) + ("\n" if all_test else ""), encoding="utf-8")
    train_stats = _stats(dataset_root, TRAIN_ENTRIES, label_path, mapping)
    val_stats = _stats(dataset_root, VAL_ENTRIES, label_path, mapping)
    class_names = sorted(mapping, key=mapping.get)
    missing_train = [name for name in class_names if train_stats["class_frames"].get(name, 0) == 0]
    missing_val = [name for name in class_names if val_stats["class_frames"].get(name, 0) == 0]
    if missing_train or missing_val:
        raise SystemExit(f"Missing class coverage train={missing_train} val={missing_val}")
    duplicate_paths = len(set(TRAIN_ENTRIES + VAL_ENTRIES)) != len(TRAIN_ENTRIES + VAL_ENTRIES)
    summary = {"ontology": dict(mapping), "aliases": dict(mapping.aliases), "train": train_stats, "validation": val_stats, "test": {task: {"entries": entries, "trajectory_count": len(entries), "task": task} for task, entries in test_paths.items()}, "train_validation_duplicate_entry": duplicate_paths, "metadata_family_grouping": "unavailable; exact path/hash leakage checks used"}
    (args.output_dir / "split_statistics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
