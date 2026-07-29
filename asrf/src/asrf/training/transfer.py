"""Explicit class-head transfer helpers for isolated ASRF experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor


OLD_CLASS_COUNT = 10
NEW_CLASS_COUNT = 11
COPIED_CLASS_ROWS = tuple(range(10))
NEW_CLASS_ROWS = (10,)


def project_asrf_state_dict(
    source_state: Mapping[str, Tensor],
    destination_state: Mapping[str, Tensor],
    *,
    source_class_rows: Mapping[int, int],
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Project a compatible ASRF checkpoint onto a smaller class ontology.

    The temporal extractor and BRB are copied exactly.  ASB tensors whose
    class dimension changes are copied only through the explicit destination
    to source row mapping; no mismatched classifier tensor is loaded
    implicitly.  Destination-only tensors are left at the model's seeded
    initialization and are reported.
    """
    result = {key: value.detach().clone() for key, value in destination_state.items()}
    copied: list[str] = []
    partial: list[str] = []
    destination_only: list[str] = []
    source_rows = {int(destination): int(source) for destination, source in source_class_rows.items()}
    for key, destination in destination_state.items():
        source = source_state.get(key)
        if source is None:
            destination_only.append(key)
            continue
        if tuple(source.shape) == tuple(destination.shape):
            result[key] = source.detach().clone()
            copied.append(key)
            continue
        if key == "asb.initial_projection.weight" and source.ndim == destination.ndim == 3 and source.shape[1:] == destination.shape[1:]:
            for dest, src in source_rows.items():
                result[key][dest] = source[src]
            partial.append(key)
        elif key == "asb.initial_projection.bias" and source.ndim == destination.ndim == 1:
            for dest, src in source_rows.items():
                result[key][dest] = source[src]
            partial.append(key)
        elif key.startswith("asb.refinement_stages.") and key.endswith("conv_in.weight") and source.ndim == destination.ndim == 3 and source.shape[0] == destination.shape[0] and source.shape[2] == destination.shape[2]:
            for dest, src in source_rows.items():
                result[key][:, dest] = source[:, src]
            partial.append(key)
        elif key.startswith("asb.refinement_stages.") and key.endswith("conv_out.weight") and source.ndim == destination.ndim == 3 and source.shape[1:] == destination.shape[1:]:
            for dest, src in source_rows.items():
                result[key][dest] = source[src]
            partial.append(key)
        elif key.startswith("asb.refinement_stages.") and key.endswith("conv_out.bias") and source.ndim == destination.ndim == 1:
            for dest, src in source_rows.items():
                result[key][dest] = source[src]
            partial.append(key)
        else:
            raise ValueError(f"Unexpected incompatible checkpoint tensor: {key}: {tuple(source.shape)} -> {tuple(destination.shape)}")
    metadata = {
        "source_class_count": max(source_rows.values()) + 1 if source_rows else 0,
        "destination_class_count": len(destination_state["asb.initial_projection.bias"]),
        "destination_to_source_class_rows": {str(dest): src for dest, src in sorted(source_rows.items())},
        "copied_exact_keys": copied,
        "copied_partial_keys": partial,
        "destination_only_keys": destination_only,
    }
    return result, metadata


def expand_asrf_state_dict(old_state: Mapping[str, Tensor], new_state: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Copy compatible parameters and explicit class rows into an 11-class model.

    ASB refinement ``conv_in`` and ``conv_out`` tensors change shape when the class count grows;
    compatible temporal channels and the ten historical class rows are copied explicitly. The new
    destination row remains at independent random initialization. This helper is only reachable
    after an explicit ontology-compatible checkpoint validation.
    """
    result = {key: value.detach().clone() for key, value in new_state.items()}
    copied: list[str] = []
    partial: list[str] = []
    skipped: list[str] = []
    for key, destination in new_state.items():
        source = old_state.get(key)
        if source is None:
            skipped.append(key)
            continue
        if tuple(source.shape) == tuple(destination.shape):
            result[key] = source.detach().clone()
            copied.append(key)
            continue
        if key == "asb.initial_projection.weight" and source.shape[0] == OLD_CLASS_COUNT and destination.shape[0] == NEW_CLASS_COUNT:
            result[key][:OLD_CLASS_COUNT] = source
            partial.append(key)
        elif key == "asb.initial_projection.bias" and source.shape[0] == OLD_CLASS_COUNT and destination.shape[0] == NEW_CLASS_COUNT:
            result[key][:OLD_CLASS_COUNT] = source
            partial.append(key)
        elif key.startswith("asb.refinement_stages.") and key.endswith("conv_in.weight") and source.shape[1] == OLD_CLASS_COUNT and destination.shape[1] == NEW_CLASS_COUNT:
            result[key][:, :OLD_CLASS_COUNT] = source
            partial.append(key)
        elif key.startswith("asb.refinement_stages.") and key.endswith("conv_out.weight") and source.shape[0] == OLD_CLASS_COUNT and destination.shape[0] == NEW_CLASS_COUNT:
            result[key][:OLD_CLASS_COUNT] = source
            partial.append(key)
        elif key.startswith("asb.refinement_stages.") and key.endswith("conv_out.bias") and source.shape[0] == OLD_CLASS_COUNT and destination.shape[0] == NEW_CLASS_COUNT:
            result[key][:OLD_CLASS_COUNT] = source
            partial.append(key)
        else:
            raise ValueError(f"Unexpected incompatible checkpoint tensor: {key}: {tuple(source.shape)} -> {tuple(destination.shape)}")
    metadata = {
        "source_class_count": OLD_CLASS_COUNT,
        "destination_class_count": NEW_CLASS_COUNT,
        "copied_class_rows": list(COPIED_CLASS_ROWS),
        "new_class_rows_randomly_initialized": list(NEW_CLASS_ROWS),
        "copied_exact_keys": copied,
        "copied_partial_keys": partial,
        "destination_only_keys": skipped,
    }
    return result, metadata


__all__ = ["COPIED_CLASS_ROWS", "NEW_CLASS_ROWS", "expand_asrf_state_dict", "project_asrf_state_dict"]
