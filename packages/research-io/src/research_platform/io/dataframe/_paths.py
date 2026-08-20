from __future__ import annotations

from pathlib import Path
from glob import glob
from typing import Iterable

from ._types import PathInput, PathLike

_GLOB_CHARS = "*?[]"
_FORMAT_BY_SUFFIX = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "txt",
    ".parquet": "parquet",
    ".feather": "feather",
}
_SUPPORTED_FORMATS = {"csv", "tsv", "txt", "parquet", "feather"}


def _has_glob(value: str) -> bool:
    return any(ch in value for ch in _GLOB_CHARS)


def infer_format(path: PathLike) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in _FORMAT_BY_SUFFIX:
        raise ValueError(f"Unsupported format from extension '{suffix}'. Supported: {sorted(_SUPPORTED_FORMATS)}")
    return _FORMAT_BY_SUFFIX[suffix]


def expand_paths(paths: PathInput) -> list[Path]:
    if isinstance(paths, (str, Path)):
        path_values: Iterable[PathLike] = [paths]
    else:
        path_values = list(paths)

    expanded: list[Path] = []
    for value in path_values:
        text = str(value)
        if _has_glob(text):
            matched = sorted(Path(p) for p in glob(text))
            if not matched:
                raise FileNotFoundError(f"No paths matched glob pattern '{text}'.")
            expanded.extend(matched)
            continue

        candidate = Path(text)
        if not candidate.exists():
            raise FileNotFoundError(f"Path '{text}' does not exist.")
        expanded.append(candidate)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        if path not in seen:
            deduped.append(path)
            seen.add(path)

    return deduped


def validate_format(format_name: str) -> str:
    normalized = format_name.lower()
    if normalized not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{format_name}'. Supported formats: {sorted(_SUPPORTED_FORMATS)}")
    return normalized


def ensure_same_format(paths: list[Path], explicit_format: str | None = None) -> str:
    if explicit_format is not None:
        return validate_format(explicit_format)

    formats = {infer_format(path) for path in paths}
    if len(formats) != 1:
        raise ValueError("Multiple file formats detected. Provide format explicitly.")
    (format_name,) = formats
    return format_name
