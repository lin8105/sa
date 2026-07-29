"""Read-only CITR trajectory data interfaces for ASRF."""

from .dataset import MultiTaskTrajectoryDataset, TrajectoryDataset, load_trajectory_sample
from .boundary_targets import boundary_targets_from_segments, generate_boundary_targets
from .labels import LabelMapping, load_label_mapping, normalize_label_name
from .ontology import CANONICAL_LABELS, LABEL_TO_ID, ONTOLOGY_VERSION

__all__ = [
    "LabelMapping",
    "MultiTaskTrajectoryDataset",
    "TrajectoryDataset",
    "boundary_targets_from_segments",
    "generate_boundary_targets",
    "load_label_mapping",
    "load_trajectory_sample",
    "normalize_label_name",
    "CANONICAL_LABELS",
    "LABEL_TO_ID",
    "ONTOLOGY_VERSION",
]
