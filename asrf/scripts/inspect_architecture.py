#!/usr/bin/env python3
"""Print ASRF architecture counts, receptive fields, and shape diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asrf.models import ASRFModel  # noqa: E402
from asrf.models.layers import temporal_receptive_field  # noqa: E402
from asrf.utils.config import load_yaml_config, resolve_repo_path  # noqa: E402


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="ASRF architecture YAML path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    model = ASRFModel.from_config(config)
    dilations = tuple(config["model"]["dilation_schedule"])
    kernel_size = int(config["model"]["kernel_size"])
    stack_rf = temporal_receptive_field(dilations, kernel_size)
    refinement_increment = stack_rf - 1
    refinement_stages = int(config["model"]["asb_refinement_stages"])
    shared_rf = stack_rf
    branch_rf_from_encoder = shared_rf + refinement_stages * refinement_increment
    # Three k=5 temporal convolutions in the RGB encoder add 4 frames each.
    encoder_temporal_rf = 1 + 3 * (5 - 1)
    end_to_end_rf = encoder_temporal_rf + branch_rf_from_encoder - 1

    print(f"config={resolve_repo_path(args.config)}")
    print(f"heatmap_encoder_parameters={parameter_count(model.encoder)}")
    print(f"shared_extractor_parameters={parameter_count(model.feature_extractor)}")
    print(f"asb_parameters={parameter_count(model.asb)}")
    print(f"brb_parameters={parameter_count(model.brb)}")
    print(f"total_parameters={parameter_count(model)}")
    print(f"stages_total=4 (initial + {refinement_stages} refinement stages)")
    print(f"temporal_layers_per_stack={len(dilations)}")
    print(f"dilation_schedule={dilations}")
    print(f"stage_receptive_field_frames={stack_rf}")
    print(f"stage_receptive_field_seconds_at_100hz={stack_rf / 100.0:.2f}")
    print(f"shared_receptive_field_frames={shared_rf}")
    print(f"final_branch_receptive_field_from_encoder_frames={branch_rf_from_encoder}")
    print(f"encoder_temporal_receptive_field_frames={encoder_temporal_rf}")
    print(f"end_to_end_receptive_field_heatmap_columns={end_to_end_rf}")
    print(f"end_to_end_receptive_field_seconds_at_100hz={end_to_end_rf / 100.0:.2f}")
    print("receptive_field_composition=stage-wise additive composition; no cross-branch composition")

    model.eval()
    with torch.no_grad():
        sample = torch.randn(1, 3, 88, 17)
        output = model(sample)
    print(f"synthetic_input_shape={tuple(sample.shape)}")
    print(f"encoder_output_shape={tuple(output.encoder_features.shape)}")
    print(f"shared_output_shape={tuple(output.shared_features.shape)}")
    print(f"asb_stage_shapes={[tuple(value.shape) for value in output.asb_stage_logits]}")
    print(f"brb_stage_shapes={[tuple(value.shape) for value in output.brb_stage_logits]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

