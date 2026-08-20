"""Tabular write helpers."""

from pathlib import Path
from typing import Any, Mapping

from research_platform.io.dataframe.adapters import resolve_backend
from research_platform.io.dataframe._paths import infer_format, validate_format


def write_tabular(
    table: Any,
    path,
    *,
    format: str | None = None,
    backend: str = "polars",
    write_kwargs: Mapping[str, Any] | None = None,
) -> None:
    output_path = Path(path)
    backend_api = resolve_backend(backend)
    if format is not None:
        output_format = validate_format(format)
    else:
        output_format = infer_format(output_path)
    if not backend_api.format_supported(output_format):
        raise ValueError(f"Backend {backend!r} does not support format {output_format!r}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    backend_api.write(table=table, path=output_path, format=output_format, write_kwargs=write_kwargs or {})
