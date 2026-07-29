"""Versioned model-facing ontology for the Round 12 multiskill models.

The migration from the previous Plug ontology is intentionally explicit.  Old
classifier rows are not reinterpret-able because class 10 changed meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ONTOLOGY_VERSION = "round12_multiskill_v2"
CANONICAL_LABELS = (
    "reach", "grasp", "lift", "transport", "pour", "pour_recover",
    "place", "release", "wipe", "retreat", "insert",
)
LABEL_TO_ID = {name: index for index, name in enumerate(CANONICAL_LABELS)}
ALIASES = {"pull_out": "lift", "extract": "lift"}
LEGACY_MIGRATION = {
    "previous_ontology": "round9_multiskill_v1",
    "old_align_id": 10,
    "old_insert_id": 11,
    "new_insert_id": 10,
    "note": "Old align ID 10 was removed; old insert ID 11 becomes new insert ID 10.",
    "automatic_weight_reinterpretation": False,
}


class OntologyVersionMismatchError(ValueError):
    """Raised when an artifact was produced under a different ontology."""


def ontology_metadata() -> dict[str, Any]:
    """Return immutable metadata to embed in new model artifacts."""
    return {
        "ontology_version": ONTOLOGY_VERSION,
        "labels": dict(LABEL_TO_ID),
        "aliases": dict(ALIASES),
        "num_classes": len(CANONICAL_LABELS),
    }


def metadata_for_mapping(mapping: Mapping[str, Any], aliases: Mapping[str, Any] | None = None, *, version: str | None = None) -> dict[str, Any]:
    """Stamp an artifact with its actual mapping, including task-specific maps."""
    mapping_dict = {str(key): int(value) for key, value in mapping.items()}
    alias_dict = {str(key): str(value) for key, value in (aliases or {}).items()}
    if mapping_dict == LABEL_TO_ID and alias_dict == ALIASES:
        return ontology_metadata()
    return {
        "ontology_version": version or "legacy_task_specific_unversioned",
        "labels": mapping_dict,
        "aliases": alias_dict,
        "num_classes": len(mapping_dict),
    }


def metadata_for_task(class_names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Stamp a subset model while retaining the full canonical ontology identity."""
    names = [str(name) for name in class_names]
    if any(name not in LABEL_TO_ID for name in names):
        raise ValueError(f"Task class names are outside the canonical ontology: {names!r}")
    metadata = ontology_metadata()
    metadata["task_class_names"] = names
    metadata["task_class_ids"] = [LABEL_TO_ID[name] for name in names]
    return metadata


def validate_ontology_metadata(metadata: Mapping[str, Any] | None, *, context: str = "artifact") -> None:
    """Require an exact Round 12 ontology identity for model-facing artifacts."""
    if not isinstance(metadata, Mapping):
        raise OntologyVersionMismatchError(
            f"{context} has no ontology metadata; expected {ONTOLOGY_VERSION}. "
            "Legacy checkpoints and prototype banks cannot be reinterpreted automatically."
        )
    version = metadata.get("ontology_version")
    labels = metadata.get("labels", metadata.get("label_map"))
    aliases = metadata.get("aliases", metadata.get("label_aliases", {}))
    if version != ONTOLOGY_VERSION or dict(labels or {}) != LABEL_TO_ID or dict(aliases or {}) != ALIASES:
        raise OntologyVersionMismatchError(
            f"{context} ontology mismatch: found version={version!r}, labels={dict(labels or {})!r}; "
            f"expected version={ONTOLOGY_VERSION!r}, labels={LABEL_TO_ID!r}. "
            "Do not reinterpret old classifier weights or prototype rows."
        )
    if int(metadata.get("num_classes", len(LABEL_TO_ID))) != len(LABEL_TO_ID):
        raise OntologyVersionMismatchError(f"{context} class count does not match {ONTOLOGY_VERSION}.")


def normalize_model_label(label_name: str) -> str:
    """Canonicalize the two permitted annotation aliases only."""
    normalized = str(label_name).strip()
    if normalized in ALIASES:
        normalized = ALIASES[normalized]
    if normalized not in LABEL_TO_ID:
        raise ValueError(f"Unknown Round 12 model label {normalized!r}; no legacy labels are mapped at runtime.")
    return normalized


__all__ = [
    "ALIASES", "CANONICAL_LABELS", "LABEL_TO_ID", "LEGACY_MIGRATION",
    "ONTOLOGY_VERSION", "OntologyVersionMismatchError", "normalize_model_label",
    "metadata_for_mapping", "metadata_for_task", "ontology_metadata", "validate_ontology_metadata",
]
