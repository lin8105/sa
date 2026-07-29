"""Repository-local YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a relative path against the independent ASRF project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a mapping-style YAML file without changing any external path."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("PyYAML is required to load ASRF configuration files.") from exc

    config_path = resolve_repo_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at {config_path}.")
    return data

