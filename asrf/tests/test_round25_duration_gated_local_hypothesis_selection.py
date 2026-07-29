import sys

sys.path.insert(0, "scripts")
import run_round25_duration_gated_local_hypothesis_selection as round25  # noqa: E402


def _prediction(start, end, label="grasp"):
    return {
        "start": start,
        "end": end,
        "duration": end - start,
        "top1_id": round25.CLASS_NAMES.index(label),
        "top1_label": label,
        "top1_probability": 0.9,
        "top2_probability": 0.05,
        "margin": 0.85,
        "embedding": [1.0] + [0.0] * 127,
        "embedding_norm": 1.0,
    }


def test_validation_record_without_metric_wrapper_is_summarized_with_validation_split():
    record = {
        "trajectory": "train/pick and place/pp11",
        "family": "pick_and_place",
        "split": "validation",
        "length": 100,
        "gt": [{"start": 0, "end": 100, "label": "grasp", "label_id": 1}],
    }
    metrics, *_ = round25.metric_for(record, [_prediction(0, 100)], "raw_asrf")
    assert metrics["split"] == "validation"
    assert metrics["segmental_f1@50"] == 1.0
