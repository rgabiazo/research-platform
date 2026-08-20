"""Run-manifest persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._yaml import load_yaml, write_yaml


def write_run_manifest(run_root: str | Path, manifest: dict[str, Any]) -> Path:
    return write_yaml(Path(run_root) / "run-manifest.yaml", manifest)


def read_run_manifest(run_root: str | Path) -> dict[str, Any]:
    return load_yaml(Path(run_root) / "run-manifest.yaml")


def write_status(run_root: str | Path, status: dict[str, Any]) -> Path:
    return write_yaml(Path(run_root) / "status.yaml", status)


def read_status(run_root: str | Path) -> dict[str, Any]:
    return load_yaml(Path(run_root) / "status.yaml")
