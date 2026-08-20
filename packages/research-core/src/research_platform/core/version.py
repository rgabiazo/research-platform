"""Resolve the public software identity owned by ``research-core``."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import tomllib


CORE_DISTRIBUTION = "research-core"
PUBLIC_PRODUCT = "research-platform"
_SOURCE_PROJECT_FILE = Path(__file__).resolve().parents[3] / "pyproject.toml"


def research_core_version() -> str:
    """Return the installed version, falling back to source-checkout metadata."""

    try:
        return metadata.version(CORE_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return _source_checkout_version()


def version_report() -> str:
    """Return the stable user-facing platform version line."""

    return f"{PUBLIC_PRODUCT} {research_core_version()}"


def _source_checkout_version(project_file: Path | None = None) -> str:
    source_file = _SOURCE_PROJECT_FILE if project_file is None else project_file
    try:
        document = tomllib.loads(source_file.read_text(encoding="utf-8"))
        version = document["project"]["version"]
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(
            f"Could not resolve {CORE_DISTRIBUTION} version from installed or source-checkout metadata."
        ) from exc
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"Source-checkout metadata has no valid {CORE_DISTRIBUTION} version.")
    return version.strip()
