"""Output-path checks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def resolved_output_root(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser()
    if has_parent_traversal(root):
        raise ValueError("output_root must not contain parent traversal.")
    return root.resolve(strict=False)


def has_parent_traversal(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def reject_duplicate_targets(targets: Mapping[str, Any]) -> None:
    seen: dict[Path, str] = {}
    for name, target in targets.items():
        previous = seen.get(target.path)
        if previous is not None:
            raise ValueError(f"Output paths for {previous!r} and {name!r} must be distinct.")
        seen[target.path] = name


def reject_existing_targets(targets: Mapping[str, Any], *, overwrite: bool) -> None:
    for name, target in targets.items():
        if target.path.is_dir():
            raise IsADirectoryError(f"Output path for {name!r} is an existing directory: {target.path}")
        if target.path.exists() and not overwrite:
            raise FileExistsError(f"Output path for {name!r} already exists: {target.path}")
