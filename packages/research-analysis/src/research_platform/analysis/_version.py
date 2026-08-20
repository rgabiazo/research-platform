"""Resolve the installed ``research-analysis`` distribution version."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
import tomllib


DISTRIBUTION_NAME = "research-analysis"
UNKNOWN_VERSION = "unknown"


def package_version() -> str:
    """Return installed metadata, or the coordinated source-checkout version."""

    try:
        return metadata.version(DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return _source_tree_version()


def _source_tree_version() -> str:
    for parent in Path(__file__).resolve().parents:
        pyproject_path = parent / "pyproject.toml"
        if not pyproject_path.is_file():
            continue
        try:
            with pyproject_path.open("rb") as handle:
                project = tomllib.load(handle).get("project", {})
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if not isinstance(project, Mapping) or project.get("name") != DISTRIBUTION_NAME:
            continue
        version = project.get("version")
        if isinstance(version, str) and version:
            return version
    return UNKNOWN_VERSION


__all__ = ["DISTRIBUTION_NAME", "UNKNOWN_VERSION", "package_version"]
