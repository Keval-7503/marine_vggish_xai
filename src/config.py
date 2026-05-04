"""Configuration loading and path handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and resolve project-relative path entries."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(config_path.parent)
    return cfg


def project_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["_root"]).resolve()


def resolve_path(cfg: dict[str, Any], value: str | Path | None) -> Path | None:
    """Resolve a path relative to the project root unless already absolute."""
    if value is None:
        return None
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return project_root(cfg) / p


def path_from_config(cfg: dict[str, Any], section: str, key: str) -> Path:
    return resolve_path(cfg, cfg[section][key])  # type: ignore[return-value]


def ensure_project_dirs(cfg: dict[str, Any]) -> None:
    """Create all standard output directories used by the pipeline."""
    dirs = [
        "raw_data_dir",
        "metadata_dir",
        "processed_dir",
        "pretrained_dir",
        "checkpoint_dir",
        "outputs_dir",
    ]
    for key in dirs:
        path_from_config(cfg, "paths", key).mkdir(parents=True, exist_ok=True)

    root = project_root(cfg)
    for rel in [
        "data/processed/audio_clips",
        "data/processed/logmel",
        "data/processed/splits",
        "data/sample_debug",
        "outputs/logs",
        "outputs/metrics",
        "outputs/plots",
        "outputs/confusion_matrices",
        "outputs/test_predictions",
        "outputs/xai/gradcam",
        "outputs/xai/integrated_gradients",
        "outputs/xai/occlusion",
        "outputs/xai/lime",
        "outputs/xai_evaluation",
        "outputs/report_assets",
        "models/pretrained",
        "models/checkpoints",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
