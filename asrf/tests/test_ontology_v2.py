from pathlib import Path

import pytest

from asrf.data.labels import load_label_mapping
from asrf.data.ontology import (
    ALIASES,
    CANONICAL_LABELS,
    LABEL_TO_ID,
    ONTOLOGY_VERSION,
    OntologyVersionMismatchError,
    normalize_model_label,
    ontology_metadata,
    validate_ontology_metadata,
)
from asrf.models.prototype_bank import load_prototype_bank


def test_round12_ids_are_contiguous_and_align_is_not_active() -> None:
    assert len(CANONICAL_LABELS) == 11
    assert LABEL_TO_ID["insert"] == 10
    assert "align" not in LABEL_TO_ID
    assert ALIASES == {"pull_out": "lift", "extract": "lift"}


def test_round12_config_matches_canonical_ontology() -> None:
    mapping = load_label_mapping(Path(__file__).parents[1] / "configs/labels_multiskill_v2.yaml")
    assert dict(mapping) == LABEL_TO_ID
    assert dict(mapping.aliases) == ALIASES


def test_old_artifacts_fail_closed() -> None:
    with pytest.raises(OntologyVersionMismatchError, match="ontology mismatch"):
        validate_ontology_metadata({"ontology_version": "round9_multiskill_v1", "labels": {"align": 10, "insert": 11}}, context="old checkpoint")
    with pytest.raises(ValueError, match="no legacy labels"):
        normalize_model_label("align")


def test_metadata_round_trip_shape() -> None:
    metadata = ontology_metadata()
    assert metadata["ontology_version"] == ONTOLOGY_VERSION
    validate_ontology_metadata(metadata)


def test_old_prototype_bank_fails_closed(tmp_path: Path) -> None:
    import torch

    path = tmp_path / "old_prototypes.pt"
    torch.save({"prototype_embeddings": torch.zeros(2, 4), "class_names": ["legacy"]}, path)
    with pytest.raises(OntologyVersionMismatchError, match="no ontology metadata"):
        load_prototype_bank(path)
