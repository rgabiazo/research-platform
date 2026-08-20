from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def _configured_polars_display(max_rows: int | None, max_cols: int | None, width: int | None) -> Iterator[None]:
    try:
        import polars as pl  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - backend optional
        yield
        return

    config = getattr(pl, "Config", None)
    if config is None:
        yield
        return

    with config() as cfg:
        if max_rows is not None:
            try:
                cfg.set_tbl_rows(max_rows)
            except AttributeError:
                pass
        if max_cols is not None:
            try:
                cfg.set_tbl_cols(max_cols)
            except AttributeError:
                pass
        if width is not None:
            try:
                cfg.set_tbl_width_chars(width)
            except AttributeError:
                pass
        yield


def _is_polars_table(table: Any) -> bool:
    return "polars" in table.__class__.__module__


def _is_pandas_table(table: Any) -> bool:
    return "pandas" in table.__class__.__module__


def _format_polars(
    table: Any,
    *,
    width: int | None = None,
    max_rows: int | None = None,
    max_cols: int | None = None,
) -> str:
    try:
        import polars as pl  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - backend optional
        return str(table.to_string())

    if width is None:
        width = 10_000

    to_string_kwargs = {"width": width}
    to_string = getattr(table, "to_string", None)

    with _configured_polars_display(max_rows=max_rows, max_cols=max_cols, width=width):
        if callable(to_string):
            try:
                return str(to_string(**to_string_kwargs, truncate=False))
            except TypeError:
                try:
                    return str(to_string(**to_string_kwargs))
                except TypeError:
                    pass
        return str(table)


def _format_pandas(table: Any, *, width: int | None = None, max_rows: int | None = None, max_cols: int | None = None) -> str:
    to_string = table.to_string
    kwargs = {"index": False}
    if max_rows is not None:
        kwargs["max_rows"] = max_rows
    if max_cols is not None:
        kwargs["max_cols"] = max_cols
    # Use full column width for preview windows unless caller explicitly requests truncation.
    kwargs["max_colwidth"] = None

    try:
        return str(to_string(**kwargs))
    except TypeError:
        kwargs_without_row_col_limits: dict[str, Any] = {k: v for k, v in kwargs.items() if k not in {"max_rows", "max_cols", "max_colwidth"}}
        try:
            return str(to_string(**kwargs_without_row_col_limits))
        except TypeError:
            if width is not None:
                kwargs_without_width = dict(kwargs_without_row_col_limits)
                kwargs_without_width.pop("max_colwidth", None)
                try:
                    return str(to_string(**kwargs_without_width))
                except TypeError:
                    pass
            return str(to_string())


def format_preview(
    table: Any,
    *,
    width: int | None = None,
    max_rows: int | None = None,
    max_cols: int | None = None,
) -> str:
    if _is_polars_table(table):
        return _format_polars(table, width=width, max_rows=max_rows, max_cols=max_cols)

    if _is_pandas_table(table):
        return _format_pandas(table, width=width, max_rows=max_rows, max_cols=max_cols)

    if hasattr(table, "to_string"):
        to_string = getattr(table, "to_string")

        to_string = getattr(table, "to_string")
        if width is not None:
            try:
                return str(to_string(width=width))
            except TypeError:
                return str(to_string())
        return str(to_string())

    return str(table)
