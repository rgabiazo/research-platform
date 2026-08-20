from __future__ import annotations

from typing import Any, Sequence

from research_platform.io.dataframe._backend_protocol import MergeHow
from research_platform.io.dataframe._preview import format_preview
from research_platform.io.dataframe.adapters import resolve_backend


def _table_row_count(table: Any) -> int:
    if hasattr(table, "height"):
        return int(getattr(table, "height"))
    if hasattr(table, "shape"):
        shape = getattr(table, "shape")
        return int(shape[0])
    if hasattr(table, "__len__"):
        return len(table)
    return 0


def _table_column_count(table: Any) -> int:
    return len(getattr(table, "columns", []))


def _select_first_columns(table: Any, cols: int | None) -> Any:
    if cols is None:
        return table
    if cols <= 0:
        return table.head(0) if hasattr(table, "head") else table

    column_count = _table_column_count(table)
    if cols >= column_count:
        return table

    selected = list(getattr(table, "columns", []))[:cols]
    if hasattr(table, "select"):
        return table.select(selected)
    if hasattr(table, "iloc"):
        return table.iloc[:, :cols]
    return table


def _table_columns(table: Any) -> list[str]:
    columns = getattr(table, "columns", None)
    if columns is None:
        raise ValueError("Merge inputs must expose a 'columns' attribute.")
    return [str(column) for column in list(columns)]


def _rename_columns(table: Any, rename_map: dict[str, str]) -> Any:
    if not rename_map:
        return table
    if hasattr(table, "rename"):
        return table.rename(rename_map)
    raise ValueError("Merge input does not implement column renaming.")


def _dedupe_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    index = 1
    candidate = f"{name}_{index}"
    while candidate in used:
        index += 1
        candidate = f"{name}_{index}"
    return candidate


def _rename_collisions(
    table: Any,
    *,
    existing_columns: set[str],
    key: str,
    source_suffix: str,
) -> Any:
    current_columns = _table_columns(table)
    rename_map: dict[str, str] = {}
    reserved = set(existing_columns)
    for column in current_columns:
        if column == key or column not in reserved:
            continue
        renamed = f"{column}_{source_suffix}"
        renamed = _dedupe_name(renamed, reserved)
        rename_map[column] = renamed
        reserved.add(renamed)
    return _rename_columns(table, rename_map)


def collect_tabular(table, *, backend: str = "polars"):
    """Materialize a lazy table for the selected backend."""
    backend_api = resolve_backend(backend)
    return backend_api.collect(table)


def preview_tabular(
    table,
    *,
    n: int = 5,
    cols: int | None = None,
    display_all: bool = False,
    backend: str = "polars",
    width: int | None = None,
) -> str:
    """Return terminal-friendly preview text for a table or lazy table."""
    if n <= 0:
        raise ValueError("--head must be a positive integer.")
    if cols is not None and cols <= 0:
        raise ValueError("--cols must be a positive integer.")

    backend_api = resolve_backend(backend)
    preview = collect_tabular(table, backend=backend)
    row_count = _table_row_count(preview)
    column_count = _table_column_count(preview)

    if display_all:
        preview_rows = row_count
        preview_cols = column_count
    else:
        preview_rows = row_count if n > row_count else n
        if cols is not None:
            preview_cols = min(cols, column_count)
        else:
            preview_cols = column_count
        preview = backend_api.head(preview, preview_rows)

    preview = _select_first_columns(preview, preview_cols)
    preview = collect_tabular(preview, backend=backend)
    return format_preview(preview, width=width, max_rows=preview_rows, max_cols=preview_cols)


def concat_tabular(
    tables: Sequence[Any],
    *,
    backend: str = "polars",
) -> Any:
    """Concatenate one or more tables with backend-specific behavior."""
    if not tables:
        raise ValueError("No tables provided for concat.")
    backend_api = resolve_backend(backend)
    return backend_api.concat(tables)


def merge_tabular(
    left,
    right,
    *,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    how: MergeHow = "inner",
    backend: str = "polars",
):
    """Merge two tables with backend-specific behavior."""
    if how == "cross":
        if on is not None or left_on is not None or right_on is not None:
            raise ValueError("Cross joins do not use join keys. Omit --on and --left-on/--right-on.")
    else:
        if on is None and (left_on is None or right_on is None):
            raise ValueError("Merge requires either --on or both --left-on and --right-on.")
        if on is not None and (left_on is not None or right_on is not None):
            raise ValueError("Cannot provide --on and --left-on/--right-on together.")

    backend_api = resolve_backend(backend)
    return backend_api.merge(
        left,
        right,
        on=on,
        left_on=left_on,
        right_on=right_on,
        how=how,
    )


def merge_tabulars(
    tables: Sequence[Any],
    *,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    how: MergeHow = "inner",
    backend: str = "polars",
    source_labels: Sequence[str] | None = None,
) -> Any:
    if len(tables) < 2:
        raise ValueError("At least two tables are required to merge.")
    if source_labels is not None and len(source_labels) != len(tables):
        raise ValueError("The number of source_labels must match the number of tables.")

    if len(tables) == 2 and (left_on is not None or right_on is not None or on is not None):
        return merge_tabular(
            tables[0],
            tables[1],
            on=on,
            left_on=left_on,
            right_on=right_on,
            how=how,
            backend=backend,
        )

    if len(tables) == 2:
        return merge_tabular(
            tables[0],
            tables[1],
            on=on,
            how=how,
            backend=backend,
        )

    if how == "cross":
        raise ValueError("N-way merges do not support --how cross.")

    if left_on is not None or right_on is not None:
        raise ValueError("Use --on when merging more than two tables.")

    if on is None:
        raise ValueError("N-way merge requires --on to be provided.")

    if not all(isinstance(table, object) for table in tables):
        raise ValueError("All merge inputs must be table objects.")

    backend_api = resolve_backend(backend)
    merged = tables[0]
    used_columns = set(_table_columns(merged))
    if on not in used_columns:
        raise ValueError(f"Cannot merge: first input is missing merge key {on!r}.")

    for index, table in enumerate(tables[1:], start=2):
        table_columns = _table_columns(table)
        if on not in table_columns:
            raise ValueError(f"Cannot merge table at position {index}; merge key {on!r} is missing.")

        suffix_source = source_labels[index - 2] if source_labels is not None else f"src{index}"
        table = _rename_collisions(
            table,
            existing_columns=used_columns,
            key=on,
            source_suffix=suffix_source,
        )
        merged = backend_api.merge(
            merged,
            table,
            on=on,
            left_on=None,
            right_on=None,
            how=how,
        )
        used_columns = set(_table_columns(merged))

    return merged


def inspect_dtypes(table, *, backend: str = "polars") -> dict[str, str]:
    backend_api = resolve_backend(backend)
    return backend_api.dtypes(table)


def inspect_describe(table, *, backend: str = "polars") -> dict[str, dict[str, Any]]:
    backend_api = resolve_backend(backend)
    return backend_api.describe(table)


def inspect_nulls(table, *, backend: str = "polars") -> dict[str, int]:
    backend_api = resolve_backend(backend)
    return backend_api.null_summary(table)


def inspect_columns_with_nulls(table, *, backend: str = "polars") -> list[str]:
    return [name for name, count in inspect_nulls(table, backend=backend).items() if count > 0]


def replace_invalid_values(
    table,
    *,
    columns: Sequence[str],
    invalid_values: Sequence[Any],
    replacement: Any = None,
    backend: str = "polars",
) -> Any:
    if not columns:
        raise ValueError("replace_invalid_values requires at least one column.")
    if not invalid_values:
        return table
    backend_api = resolve_backend(backend)
    return backend_api.replace_invalid_values(
        table,
        columns=list(columns),
        invalid_values=list(invalid_values),
        replacement=replacement,
    )


def fill_missing_with_median(
    table,
    *,
    columns: Sequence[str] | None = None,
    backend: str = "polars",
) -> Any:
    backend_api = resolve_backend(backend)
    return backend_api.fill_missing_median(table, columns=list(columns) if columns is not None else None)


def fill_missing_with_mode(
    table,
    *,
    columns: Sequence[str] | None = None,
    backend: str = "polars",
) -> Any:
    backend_api = resolve_backend(backend)
    return backend_api.fill_missing_mode(table, columns=list(columns) if columns is not None else None)


def drop_rows_by_row_numbers(table, row_numbers: Sequence[int], *, backend: str = "polars") -> Any:
    backend_api = resolve_backend(backend)
    return backend_api.drop_rows_by_numbers(table, row_numbers=list(row_numbers))


def drop_rows_by_id_values(
    table,
    *,
    id_column: str,
    id_values: Sequence[Any],
    backend: str = "polars",
) -> Any:
    if not id_values:
        raise ValueError("id_values must not be empty.")
    backend_api = resolve_backend(backend)
    return backend_api.drop_rows_by_id_values(table, id_column=id_column, id_values=list(id_values))


__all__ = [
    "collect_tabular",
    "preview_tabular",
    "concat_tabular",
    "merge_tabular",
    "merge_tabulars",
    "inspect_dtypes",
    "inspect_describe",
    "inspect_nulls",
    "inspect_columns_with_nulls",
    "replace_invalid_values",
    "fill_missing_with_median",
    "fill_missing_with_mode",
    "drop_rows_by_row_numbers",
    "drop_rows_by_id_values",
]
