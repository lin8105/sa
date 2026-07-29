from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from asrf.data.collate import collate_fn
from asrf.data.dataset import MultiTaskTrajectoryDataset
from asrf.data.labels import load_label_mapping, normalize_label_name
from asrf.losses.combined import ASRFLoss
from asrf.models import ASRFModel
from asrf.training.checkpointing import load_checkpoint, save_checkpoint
from asrf.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data")


def test_nine_class_ontology_and_aliases() -> None:
    mapping = load_label_mapping(ROOT / "configs/labels_multitask.yaml")
    assert len(mapping) == 9
    assert mapping["wipe"] == 7
    assert mapping["retreat"] == 8
    assert mapping.aliases == {"pick": "reach", "translation": "transport"}
    assert normalize_label_name("pick", mapping) == "reach"
    assert normalize_label_name("translation", mapping) == "transport"


def test_multi_root_resolution_and_basename_collision_protection(tmp_path: Path) -> None:
    label_path = ROOT / "configs/labels_multitask.yaml"
    train_split = tmp_path / "train.txt"
    train_split.write_text("train/pour/p1\ntrain/pick and place/pp1\n", encoding="utf-8")
    dataset = MultiTaskTrajectoryDataset(DATA_ROOT, train_split, label_path, allow_test=False)
    assert dataset.resolved_paths[0].name == "p1"
    assert dataset.resolved_paths[1].name == "pp1"
    assert len(set(dataset.resolved_paths)) == 2
    test_split = tmp_path / "test.txt"
    test_split.write_text("test/pour/p1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Test path"):
        MultiTaskTrajectoryDataset(DATA_ROOT, test_split, label_path, allow_test=False)


def test_all_training_labels_map_to_current_ten_classes() -> None:
    config = load_yaml_config(ROOT / "configs/brb_release_round8/baseline_single_frame.yaml")
    dataset = MultiTaskTrajectoryDataset(DATA_ROOT, ROOT / config["data"]["train_split"], ROOT / config["data"]["label_config"], allow_test=False)
    for index in range(len(dataset)):
        sample = dataset[index]
        assert int(sample["labels"].min()) >= 0
        assert int(sample["labels"].max()) < 10


def test_one_nine_class_training_step_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = load_yaml_config(ROOT / "configs/multitask_asrf_train.yaml")
    torch.manual_seed(42)
    model = ASRFModel.from_config(config)
    model.train()
    heatmap = torch.randn(1, 3, 88, 19)
    mask = torch.ones(1, 19, dtype=torch.bool)
    labels = torch.randint(0, 9, (1, 19))
    targets = torch.zeros(1, 19)
    targets[:, 0] = 1
    output = model(heatmap, valid_mask=mask)
    assert len(output.asb_stage_logits) == 4
    assert output.asb_stage_logits[-1].shape == (1, 9, 19)
    loss = ASRFLoss(class_weights=torch.ones(9), boundary_positive_weight=2.0)(output, labels, targets, mask).total_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    checkpoint = tmp_path / "round5.pt"
    save_checkpoint(checkpoint, {"model_state": model.state_dict(), "label_map": {"wipe": 7, "retreat": 8}})
    restored = ASRFModel.from_config(config)
    restored.load_state_dict(load_checkpoint(checkpoint)["model_state"])
    restored.eval()
    with torch.no_grad():
        first = model.eval()(heatmap, valid_mask=mask).asb_stage_logits[-1]
        second = restored(heatmap, valid_mask=mask).asb_stage_logits[-1]
    assert torch.allclose(first, second)


def test_old_seven_class_checkpoint_is_frozen_and_separate() -> None:
    checkpoint = ROOT / "outputs/pour_baseline/best.pt"
    expected = "586fc50c91c735f7212c16baa052f43655b3140408aa3c0d534d11daa1fbc358"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == expected
    payload = load_checkpoint(checkpoint)
    assert len(payload["label_map"]) == 7
    assert payload["architecture_config"]["num_classes"] == 7
    multitask = ROOT / "outputs/multitask_baseline/best.pt"
    assert multitask.is_file()
    multitask_payload = load_checkpoint(multitask)
    assert len(multitask_payload["label_map"]) == 9
    assert multitask != checkpoint
    assert hashlib.sha256(multitask.read_bytes()).hexdigest() != expected
