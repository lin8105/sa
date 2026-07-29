from __future__ import annotations

from pathlib import Path

import torch

from asrf.models import ASRFModel
from asrf.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]


def make_model() -> ASRFModel:
    return ASRFModel.from_config(load_yaml_config(ROOT / "configs/pour_asrf_architecture.yaml"))


def make_input() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    return (
        torch.randn(2, 3, 88, 17),
        torch.tensor([[True] * 17, [True] * 11 + [False] * 6]),
    )


def test_top_level_output_shapes_and_stage_counts() -> None:
    model = make_model().eval()
    heatmap, valid_mask = make_input()
    with torch.no_grad():
        output = model(heatmap, valid_mask)
    assert output.encoder_features.shape == (2, 128, 17)
    assert output.shared_features.shape == (2, 64, 17)
    assert len(output.asb_stage_logits) == len(output.asb_stage_probabilities) == 4
    assert len(output.brb_stage_logits) == len(output.brb_stage_probabilities) == 4
    assert all(value.shape == (2, 7, 17) for value in output.asb_stage_logits)
    assert all(value.shape == (2, 7, 17) for value in output.asb_stage_probabilities)
    assert all(value.shape == (2, 1, 17) for value in output.brb_stage_logits)
    assert all(value.shape == (2, 1, 17) for value in output.brb_stage_probabilities)


def test_asb_probabilities_sum_to_one_on_valid_frames_and_brb_is_bounded() -> None:
    model = make_model().eval()
    heatmap, valid_mask = make_input()
    with torch.no_grad():
        output = model(heatmap, valid_mask)
    for probabilities in output.asb_stage_probabilities:
        valid_values = probabilities.permute(0, 2, 1)[valid_mask]
        assert torch.allclose(valid_values.sum(dim=1), torch.ones(valid_values.shape[0]))
    for probabilities in output.brb_stage_probabilities:
        assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_mask_is_preserved_and_invalid_positions_are_zero() -> None:
    model = make_model().eval()
    heatmap, valid_mask = make_input()
    with torch.no_grad():
        output = model(heatmap, valid_mask)
    assert torch.equal(output.valid_mask, valid_mask)
    invalid = ~valid_mask
    for values in (
        output.encoder_features,
        output.shared_features,
        *output.asb_stage_logits,
        *output.asb_stage_probabilities,
        *output.brb_stage_logits,
        *output.brb_stage_probabilities,
    ):
        assert torch.count_nonzero(values.permute(0, 2, 1)[invalid]) == 0


def test_omitted_mask_equals_all_valid_mask_in_eval_mode() -> None:
    model = make_model().eval()
    heatmap, _ = make_input()
    all_valid = torch.ones(heatmap.shape[0], heatmap.shape[-1], dtype=torch.bool)
    with torch.no_grad():
        omitted = model(heatmap)
        explicit = model(heatmap, all_valid)
    assert torch.equal(omitted.valid_mask, explicit.valid_mask)
    for left, right in zip(omitted.asb_stage_logits, explicit.asb_stage_logits):
        assert torch.allclose(left, right)
    for left, right in zip(omitted.brb_stage_probabilities, explicit.brb_stage_probabilities):
        assert torch.allclose(left, right)


def test_train_mode_reaches_encoder_extractor_asb_and_brb() -> None:
    model = make_model().train()
    heatmap, valid_mask = make_input()
    output = model(heatmap, valid_mask)
    loss = sum(value.square().mean() for value in output.asb_stage_logits)
    loss = loss + sum(value.square().mean() for value in output.brb_stage_logits)
    loss.backward()
    for component in (model.encoder, model.feature_extractor, model.asb, model.brb):
        gradients = [parameter.grad for parameter in component.parameters() if parameter.grad is not None]
        assert gradients
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_train_and_eval_modes_are_supported() -> None:
    model = make_model()
    heatmap, valid_mask = make_input()
    model.train()
    train_output = model(heatmap, valid_mask)
    model.eval()
    with torch.no_grad():
        eval_output = model(heatmap, valid_mask)
    assert len(train_output.asb_stage_logits) == 4
    assert len(eval_output.brb_stage_logits) == 4


def test_real_p1_sample_runs_through_encoder_without_resize() -> None:
    from asrf.data.dataset import TrajectoryDataset

    dataset = TrajectoryDataset(
        Path("/media/yue/cdb9583f-c583-4b69-965e-b0d778e3bf71/seg_learning/data/train/pour"),
        ROOT / "splits/pour_train.txt",
        ROOT / "configs/labels_multitask_release.yaml",
    )
    sample = dataset[0]
    model = make_model().encoder.eval()
    with torch.no_grad():
        features = model(sample["heatmap"].unsqueeze(0))
    assert features.shape == (1, 128, 4195)


def test_forward_does_not_modify_input() -> None:
    model = make_model().eval()
    heatmap, valid_mask = make_input()
    original = heatmap.clone()
    with torch.no_grad():
        model(heatmap, valid_mask)
    assert torch.equal(heatmap, original)


def test_architecture_mapping_documents_official_commit_and_components() -> None:
    mapping = (ROOT / "docs/asrf_architecture_mapping.md").read_text(encoding="utf-8")
    for required in (
        "9623f1e8d9a1171333a4eeb65d190997b6c44a95",
        "DilatedResidualLayer",
        "SingleStageTCN",
        "ActionSegmentRefinementFramework",
        "HeatmapEncoder",
        "ASB",
        "BRB",
    ):
        assert required in mapping


def test_model_source_has_no_training_ids_or_temporal_resize() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src/asrf/models").glob("*.py"))
    assert "p1" not in source and "p2" not in source
    assert "interpolate(" not in source
    assert ".resize(" not in source


def test_asrf_does_not_import_runtime_code_from_mstcn() -> None:
    for path in (ROOT / "src/asrf").rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        assert "/home/yue/documents/zsc_franka/mstcn" not in source
        assert "seg_learning" not in source
