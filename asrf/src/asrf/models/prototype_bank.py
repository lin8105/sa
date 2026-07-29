"""Versioned persistence for model-facing prototype banks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from asrf.data.ontology import ontology_metadata, validate_ontology_metadata


def save_prototype_bank(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save a new prototype bank stamped with the active ontology."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stamped = dict(payload)
    stamped["ontology_metadata"] = ontology_metadata()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(stamped, temporary)
    os.replace(temporary, destination)
    return destination


def load_prototype_bank(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load only a prototype bank created under the active ontology."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    validate_ontology_metadata(payload.get("ontology_metadata"), context=f"prototype bank {path}")
    return payload


__all__ = ["load_prototype_bank", "save_prototype_bank"]
