from __future__ import annotations

from research_platform.io.dataframe._paths import ensure_same_format, expand_paths
from research_platform.io.dataframe.adapters import resolve_backend


def read_tabular(
    paths,
    *,
    format: str | None = None,
    backend: str = "polars",
    lazy: bool = False,
    read_kwargs: dict | None = None,
):
    """Read one table from one or many files.

    Multiple paths are concatenated into a single table.
    """

    expanded = expand_paths(paths)
    format_name = ensure_same_format(expanded, format)
    backend_api = resolve_backend(backend)
    return backend_api.read(expanded, format=format_name, lazy=lazy, read_kwargs=read_kwargs or {})


def read_tabulars(
    paths,
    *,
    format: str | None = None,
    backend: str = "polars",
    lazy: bool = False,
    read_kwargs: dict | None = None,
) -> list:
    """Read one or more tables and return one table per input path."""

    expanded = expand_paths(paths)
    format_name = ensure_same_format(expanded, format)
    backend_api = resolve_backend(backend)
    return [
        backend_api.read([path], format=format_name, lazy=lazy, read_kwargs=read_kwargs or {})
        for path in expanded
    ]
