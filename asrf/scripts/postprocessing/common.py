"""Shared schema, validation, persistence, and plotting helpers.

Frame segments use inclusive ``end`` values in this project-level schema.  Source
annotation/prediction formats with timestamp or half-open ends are converted at
their boundaries by the loaders.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SCHEMA_VERSION = "trajectory-bundle-v1"
TOOL_VERSION = "postprocessing-v1"
CITR_FEATURE_NAMES = ("citr_ff", "citr_ftau", "citr_tautau", "citr_fv", "citr_tauv", "citr_vv", "citr_fw", "citr_tauw", "citr_vw", "citr_ww")
SKILL_COLORS = {"reach": "#4c78a8", "grasp": "#f58518", "lift": "#eeca3b", "transport": "#54a24b", "place": "#e45756", "release": "#b279a2", "retreat": "#72b7b2"}


@dataclass(frozen=True)
class Segment:
    """A contiguous frame segment with an inclusive frame endpoint."""

    start: int
    end: int
    skill_id: int
    skill_name: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid inclusive segment [{self.start}, {self.end}]")

    @property
    def end_exclusive(self) -> int:
        return self.end + 1

    def to_dict(self) -> dict[str, Any]:
        return {"start": int(self.start), "end": int(self.end), "skill_id": int(self.skill_id), "skill_name": self.skill_name}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Segment":
        return cls(int(value["start"]), int(value["end"]), int(value["skill_id"]), str(value["skill_name"]))


def _array(value: Any, dtype: Any = float) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


@dataclass
class TrajectoryBundle:
    """Canonical trajectory artifact shared by builder, compression, and merge."""

    citr: np.ndarray
    gt_frame_labels: np.ndarray
    gt_segments: list[Segment]
    prediction_frame_labels: np.ndarray
    prediction_segments: list[Segment]
    sf_brb: np.ndarray
    r5_brb: np.ndarray
    gripper_norm: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamps_s: np.ndarray | None = None
    timestamps_us: np.ndarray | None = None
    motion_features: np.ndarray | None = None
    motion_feature_names: list[str] = field(default_factory=list)

    @property
    def current_T(self) -> int:
        return int(self.citr.shape[1])

    @property
    def original_T(self) -> int:
        return int(self.metadata.get("original_T", self.current_T))

    def clone(self) -> "TrajectoryBundle":
        return TrajectoryBundle(
            citr=self.citr.copy(), gt_frame_labels=self.gt_frame_labels.copy(), gt_segments=copy.deepcopy(self.gt_segments),
            prediction_frame_labels=self.prediction_frame_labels.copy(), prediction_segments=copy.deepcopy(self.prediction_segments),
            sf_brb=self.sf_brb.copy(), r5_brb=self.r5_brb.copy(), gripper_norm=self.gripper_norm.copy(),
            metadata=copy.deepcopy(self.metadata), timestamps_s=None if self.timestamps_s is None else self.timestamps_s.copy(),
            timestamps_us=None if self.timestamps_us is None else self.timestamps_us.copy(),
            motion_features=None if self.motion_features is None else self.motion_features.copy(), motion_feature_names=list(self.motion_feature_names),
        )

    def validate(self) -> None:
        if self.citr.ndim != 2 or self.citr.shape[0] != 11:
            raise ValueError(f"CITR must have shape [11,T], got {self.citr.shape}")
        n = self.current_T
        arrays = {"gt_frame_labels": self.gt_frame_labels, "prediction_frame_labels": self.prediction_frame_labels, "sf_brb": self.sf_brb, "r5_brb": self.r5_brb, "gripper_norm": self.gripper_norm}
        for name, value in arrays.items():
            if np.asarray(value).shape != (n,):
                raise ValueError(f"{name} must have length {n}, got {np.asarray(value).shape}")
        if self.timestamps_s is not None and self.timestamps_s.shape != (n,):
            raise ValueError("timestamps_s length does not match current_T")
        if self.timestamps_us is not None and self.timestamps_us.shape != (n,):
            raise ValueError("timestamps_us length does not match current_T")
        if self.motion_features is not None and self.motion_features.shape[0] != n:
            raise ValueError("motion_features length does not match current_T")
        validate_segments(self.gt_segments, n, "GT")
        validate_segments(self.prediction_segments, n, "prediction")
        skill_mapping = self.metadata.get("skill_mapping", {})
        for seg in [*self.gt_segments, *self.prediction_segments]:
            if seg.skill_name != str(skill_mapping.get(str(seg.skill_id), skill_mapping.get(seg.skill_id, seg.skill_name))):
                raise ValueError(f"skill mapping disagrees with segment {seg}")
        validate_frame_labels(self.gt_frame_labels, self.gt_segments, "GT")
        validate_frame_labels(self.prediction_frame_labels, self.prediction_segments, "prediction")
        if self.metadata.get("current_T", n) != n:
            raise ValueError("metadata current_T disagrees with arrays")


def validate_segments(segments: list[Segment], n: int, name: str) -> None:
    if not segments:
        raise ValueError(f"{name} segmentation is empty")
    if segments[0].start != 0 or segments[-1].end != n - 1:
        raise ValueError(f"{name} segments do not cover [0,{n - 1}]")
    for a, b in zip(segments, segments[1:]):
        if a.end + 1 != b.start:
            raise ValueError(f"{name} segments are not contiguous: {a}, {b}")
    for segment in segments:
        if segment.end >= n:
            raise ValueError(f"{name} segment exceeds timeline: {segment}")


def validate_frame_labels(labels: np.ndarray, segments: list[Segment], name: str) -> None:
    expected = segments_to_frame_labels(segments, len(labels), dtype=labels.dtype)
    if not np.array_equal(labels, expected):
        raise ValueError(f"{name} frame labels do not match its segments")


def segments_to_frame_labels(segments: Iterable[Segment], n: int, *, dtype: Any = np.int64) -> np.ndarray:
    labels = np.full(n, -1, dtype=dtype)
    for segment in segments:
        if segment.start < 0 or segment.end >= n:
            raise ValueError(f"segment outside timeline: {segment}")
        labels[segment.start:segment.end + 1] = segment.skill_id
    if np.any(labels < 0):
        raise ValueError("segments do not cover the complete timeline")
    return labels


def frame_labels_to_segments(labels: np.ndarray, skill_names: dict[int, str] | dict[str, str]) -> list[Segment]:
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("frame labels must be a non-empty vector")
    out: list[Segment] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            skill_id = int(labels[start])
            name = skill_names.get(skill_id, skill_names.get(str(skill_id), str(skill_id)))
            out.append(Segment(start, i - 1, skill_id, str(name)))
            start = i
    return out


def validate_skill_mapping(mapping: dict[Any, Any]) -> dict[str, str]:
    out = {str(k): str(v) for k, v in mapping.items()}
    if not out:
        raise ValueError("skill mapping is empty")
    if len(out) != len(set(out.values())):
        raise ValueError("skill mapping contains duplicate names")
    return out


def trajectory_id_from_path(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    if text.endswith("/trajectory_bundle.json"):
        text = text[:-len("/trajectory_bundle.json")]
    return text.rsplit("/", 1)[-1]


def artifact_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_if_exists(path: str | Path) -> str:
    p = Path(path)
    return artifact_hash(p) if p.is_file() else ""


def provenance_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_bundle(bundle: TrajectoryBundle, output_dir: str | Path, *, force: bool = False) -> Path:
    bundle.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "trajectory_bundle.json"
    npz_path = output / "trajectory_bundle.npz"
    if json_path.exists() and npz_path.exists() and not force:
        existing = load_bundle(output)
        existing.validate()
        return output
    metadata = copy.deepcopy(bundle.metadata)
    metadata.update({"schema_version": SCHEMA_VERSION, "tool_version": TOOL_VERSION, "original_T": bundle.original_T, "current_T": bundle.current_T, "frame_end_convention": "inclusive", "timestamp_end_convention": "exclusive"})
    payload = {"metadata": metadata, "gt_segments": [x.to_dict() for x in bundle.gt_segments], "prediction_segments": [x.to_dict() for x in bundle.prediction_segments], "motion_feature_names": bundle.motion_feature_names}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    arrays: dict[str, Any] = {"citr": bundle.citr, "gt_frame_labels": bundle.gt_frame_labels, "prediction_frame_labels": bundle.prediction_frame_labels, "sf_brb": bundle.sf_brb, "r5_brb": bundle.r5_brb, "gripper_norm": bundle.gripper_norm}
    if bundle.timestamps_s is not None: arrays["timestamps_s"] = bundle.timestamps_s
    if bundle.timestamps_us is not None: arrays["timestamps_us"] = bundle.timestamps_us
    if bundle.motion_features is not None: arrays["motion_features"] = bundle.motion_features
    np.savez_compressed(npz_path, **arrays)
    return output


def load_bundle(bundle_dir_or_file: str | Path) -> TrajectoryBundle:
    path = Path(bundle_dir_or_file)
    if path.is_file():
        path = path.parent
    payload = json.loads((path / "trajectory_bundle.json").read_text(encoding="utf-8"))
    if payload.get("metadata", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported or missing TrajectoryBundle schema in {path}")
    z = np.load(path / "trajectory_bundle.npz", allow_pickle=False)
    bundle = TrajectoryBundle(
        citr=z["citr"], gt_frame_labels=z["gt_frame_labels"], gt_segments=[Segment.from_dict(x) for x in payload["gt_segments"]],
        prediction_frame_labels=z["prediction_frame_labels"], prediction_segments=[Segment.from_dict(x) for x in payload["prediction_segments"]],
        sf_brb=z["sf_brb"], r5_brb=z["r5_brb"], gripper_norm=z["gripper_norm"], metadata=payload["metadata"],
        timestamps_s=z["timestamps_s"] if "timestamps_s" in z else None, timestamps_us=z["timestamps_us"] if "timestamps_us" in z else None,
        motion_features=z["motion_features"] if "motion_features" in z else None, motion_feature_names=list(payload.get("motion_feature_names", [])),
    )
    bundle.validate()
    validate_provenance(bundle)
    return bundle


def validate_provenance(bundle: TrajectoryBundle) -> None:
    """Validate stored schema/source hashes when provenance is available."""
    metadata = bundle.metadata
    if metadata.get("current_T") != bundle.current_T or metadata.get("original_T") != bundle.original_T:
        raise ValueError("bundle provenance timeline lengths are inconsistent")
    for key, expected in metadata.get("source_hashes", {}).items():
        path = metadata.get("source_paths", {}).get(key)
        if expected and path and Path(path).is_file() and artifact_hash(path) != expected:
            raise ValueError(f"provenance hash changed for {key}: {path}")
    compression = metadata.get("compression", {})
    mapping_path = compression.get("source")
    mapping_hash = compression.get("source_sha256")
    if mapping_hash and mapping_path and Path(mapping_path).is_file() and artifact_hash(mapping_path) != mapping_hash:
        raise ValueError(f"provenance hash changed for retained mapping: {mapping_path}")


def validate_retained_indices(indices: np.ndarray, original_T: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("retained_indices must be a non-empty vector")
    if np.any(indices < 0) or np.any(indices >= original_T) or np.any(np.diff(indices) <= 0):
        raise ValueError("retained_indices must be strictly increasing and within original_T")
    return indices


def _heatmap_from_bundle(bundle: TrajectoryBundle) -> np.ndarray:
    return np.asarray(bundle.citr, dtype=float)


def _plot_segments(axis: Any, segments: list[Segment], n: int, label: str, *, label_prefix: str = "") -> None:
    for segment in segments:
        axis.axvspan(segment.start, segment.end + 1, color=SKILL_COLORS.get(segment.skill_name, "#888888"), alpha=0.92)
        if segment.end - segment.start + 1 >= max(12, n * 0.012):
            axis.text((segment.start + segment.end + 1) / 2, 0.5, segment.skill_name, ha="center", va="center", fontsize=7, color="white", clip_on=True)
    axis.set_xlim(0, n); axis.set_ylim(0, 1); axis.set_yticks([]); axis.set_ylabel(label_prefix + label, rotation=0, ha="right", va="center", labelpad=30, fontsize=9)


def plot_bundle(bundle: TrajectoryBundle, output_path: str | Path, *, mode: str = "standard", heatmap_path: str | Path | None = None, show_thresholds: bool = True) -> Path:
    """Plot the canonical standard or merged comparison figure."""
    bundle.validate()
    n = bundle.current_T
    if mode == "standard":
        fig, axes = plt.subplots(4, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [3.0, 0.75, 0.75, 1.0]})
        axes[0].set_ylabel("CITR", rotation=0, ha="right", va="center", labelpad=30)
        _plot_segments(axes[1], bundle.gt_segments, n, "GT")
        _plot_segments(axes[2], bundle.prediction_segments, n, "HYBRID")
        brb_axis = axes[3]
    elif mode in {"merged", "merged-simple"}:
        merged = bundle.metadata.get("pre_merge_segments")
        if mode == "merged-simple" or not merged:
            fig, axes = plt.subplots(4, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [3.0, 0.75, 0.75, 1.0]})
            _plot_segments(axes[1], bundle.gt_segments, n, "GT")
            _plot_segments(axes[2], bundle.prediction_segments, n, "HYBRID")
            brb_axis = axes[3]
        else:
            pre_segments = [Segment.from_dict(x) for x in merged]
            fig, axes = plt.subplots(5, 1, figsize=(13, 7.7), sharex=True, gridspec_kw={"height_ratios": [3.0, 0.7, 0.7, 0.7, 1.0]})
            _plot_segments(axes[1], bundle.gt_segments, n, "GT")
            _plot_segments(axes[2], pre_segments, n, "HYBRID\npre")
            _plot_segments(axes[3], bundle.prediction_segments, n, "HYBRID\nmerged")
            brb_axis = axes[4]
    else:
        raise ValueError(f"unsupported figure mode: {mode}")
    if heatmap_path and Path(heatmap_path).is_file():
        image = np.asarray(Image.open(heatmap_path).convert("RGB"))
        axes[0].imshow(image, aspect="auto", origin="upper", interpolation="nearest", extent=(0, n, 1, 0))
    else:
        axes[0].imshow(_heatmap_from_bundle(bundle), cmap="jet", vmin=-1, vmax=1, aspect="auto", origin="upper", interpolation="nearest", extent=(0, n, 11, 0))
    axes[0].set_xlim(0, n)
    brb_axis.plot(np.arange(n), bundle.sf_brb, color="tab:blue", lw=1.2, label="SF BRB")
    brb_axis.plot(np.arange(n), bundle.r5_brb, color="tab:orange", lw=1.2, ls="--", label="r5 BRB")
    config = bundle.metadata.get("hybrid_config", {})
    if show_thresholds and config.get("threshold") is not None:
        brb_axis.axhline(float(config["threshold"]), color="black", ls=":", lw=0.8, label="Hybrid threshold")
    if show_thresholds and config.get("support_gate") is not None:
        brb_axis.axhline(float(config["support_gate"]), color="gray", ls="-.", lw=0.8, label="SF support gate")
    brb_axis.set_ylabel("BRB", rotation=0, ha="right", va="center", labelpad=30); brb_axis.legend(loc="upper right", fontsize=7, ncol=3)
    axes[-1].set_xlabel("time frame")
    trajectory = bundle.metadata.get("trajectory_id", "trajectory")
    axes[0].set_title(f"{bundle.metadata.get('experiment', 'Trajectory')} | {trajectory}", fontsize=11)
    for axis in axes:
        axis.grid(False)
        axis.set_xlim(0, n)
    fig.tight_layout(h_pad=0.35)
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=160, bbox_inches="tight"); plt.close(fig)
    return output
