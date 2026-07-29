"""Canonical ASRF labels and annotation aliases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ontology import ALIASES as ROUND12_ALIASES
from .ontology import LABEL_TO_ID as ROUND12_LABEL_TO_ID
from .ontology import ONTOLOGY_VERSION


class LabelMapping(dict[str, int]):
    """Canonical label IDs with non-class aliases retained as metadata."""

    def __init__(self, mapping: dict[str, int], aliases: dict[str, str] | None = None) -> None:
        super().__init__(mapping)
        self.aliases = dict(aliases or {})


def load_label_mapping(path: str | Path) -> LabelMapping:
    """Load and validate canonical labels plus aliases from YAML."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("PyYAML is required to load label mappings.") from exc

    mapping_path = Path(path)
    with mapping_path.open("r", encoding="utf-8") as handle:
        data: Any = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{mapping_path}: label mapping must be a mapping.")

    raw_labels = data.get("labels", data)
    raw_aliases = data.get("aliases", {}) if "labels" in data else {}
    if not isinstance(raw_labels, dict) or not isinstance(raw_aliases, dict):
        raise ValueError(f"{mapping_path}: labels and aliases must be mappings.")

    labels = {str(name).strip(): int(index) for name, index in raw_labels.items()}
    expected_ids = list(range(len(labels)))
    if sorted(labels.values()) != expected_ids:
        raise ValueError(f"{mapping_path}: canonical IDs must be contiguous from zero.")

    aliases = {str(alias).strip(): str(target).strip() for alias, target in raw_aliases.items()}
    for alias, target in aliases.items():
        if not alias or target not in labels:
            raise ValueError(f"{mapping_path}: invalid alias {alias!r} -> {target!r}.")
        if alias in labels and alias != target:
            raise ValueError(f"{mapping_path}: alias conflicts with canonical label {alias!r}.")
    declared_version = data.get("ontology_version")
    if declared_version == ONTOLOGY_VERSION:
        if labels != ROUND12_LABEL_TO_ID or aliases != ROUND12_ALIASES:
            raise ValueError(f"{mapping_path}: {ONTOLOGY_VERSION} must use the canonical Round 12 labels and aliases.")
    if "align" in labels:
        raise ValueError(
            f"{mapping_path}: legacy label 'align' is not a model-facing class. "
            "Relabel annotations manually as place before using the Round 12 ontology."
        )
    return LabelMapping(labels, aliases)


def normalize_label_name(label_name: str, label_mapping: LabelMapping) -> str:
    """Resolve aliases such as ``pick`` and ``translation`` to canonical names."""
    normalized = str(label_name).strip()
    visited: set[str] = set()
    while normalized in label_mapping.aliases:
        if normalized in visited:
            raise ValueError(f"Label alias cycle detected at {normalized!r}.")
        visited.add(normalized)
        normalized = label_mapping.aliases[normalized]
    return normalized
