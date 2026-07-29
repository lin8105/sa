"""Atomic checkpoint persistence and integrity helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

from asrf.data.ontology import (
    OntologyVersionMismatchError,
    ontology_metadata,
    validate_ontology_metadata,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path: str | Path, payload: dict[str, Any], *, include_ontology: bool = False) -> Path:
    """Save a checkpoint, optionally stamping the current model ontology."""
    if include_ontology:
        payload = dict(payload)
        payload["ontology_metadata"] = ontology_metadata()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_ontology: bool = False,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if expected_ontology:
        validate_ontology_metadata(payload.get("ontology_metadata"), context=str(path))
    return payload


def checkpoint_manifest(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    return {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path), "bytes": checkpoint_path.stat().st_size}


__all__ = ["OntologyVersionMismatchError", "checkpoint_manifest", "load_checkpoint", "save_checkpoint", "sha256_file"]
