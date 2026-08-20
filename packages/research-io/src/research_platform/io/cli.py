from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from research_platform.io.dataframe._backend_protocol import LazyBackendUnsupportedError, UnsupportedBackendError
from research_platform.io.dataframe.ops import (
    collect_tabular,
    concat_tabular,
    drop_rows_by_id_values,
    drop_rows_by_row_numbers,
    fill_missing_with_mode,
    fill_missing_with_median,
    inspect_columns_with_nulls,
    inspect_describe,
    inspect_dtypes,
    inspect_nulls,
    merge_tabular,
    merge_tabulars,
    preview_tabular,
    replace_invalid_values,
)
from research_platform.io.readers.tabular import read_tabular, read_tabulars
from research_platform.io.writers import write_tabular

_SUPPORTED_HOW = ("inner", "left", "right", "outer", "full", "cross")
_SUPPORTED_FORMATS = ("csv", "tsv", "txt", "parquet", "feather")
_SUPPORTED_BACKENDS = ("polars", "pandas")


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value!r}.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"invalid value: {value!r}. Must be > 0.")
    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"invalid value: {value!r}. Must be >= 0.")
    return parsed


def _coerce_cli_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"null", "none", "na", "nan"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=_SUPPORTED_BACKENDS, default="polars")
    parser.add_argument("--head", type=_positive_integer, default=10)
    parser.add_argument("--cols", type=_positive_integer)
    parser.add_argument("--display-all", action="store_true")
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--format", choices=_SUPPORTED_FORMATS)
    parser.add_argument(
        "--string-columns",
        nargs="+",
        metavar="COLUMN",
        help="Columns to force-read as strings for CSV/TSV/TXT inputs",
    )


def _add_clean_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("paths", nargs="+", help="Input paths")
    _add_common(parser)
    parser.add_argument("--output", help="Write transformed table to this path")


def _add_inspect_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("paths", nargs="+", help="Input paths")
    parser.add_argument("--backend", choices=_SUPPORTED_BACKENDS, default="polars")
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--format", choices=_SUPPORTED_FORMATS)
    parser.add_argument(
        "--string-columns",
        nargs="+",
        metavar="COLUMN",
        help="Columns to force-read as strings for CSV/TSV/TXT inputs",
    )


def _read_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    string_columns = getattr(args, "string_columns", None)
    if not string_columns:
        return None
    unique_columns = list(dict.fromkeys(str(column) for column in string_columns))
    return {"string_columns": unique_columns}


def _read_tabular_from_args(
    args: argparse.Namespace,
    paths,
    *,
    format_name: str | None = None,
):
    return read_tabular(
        paths,
        backend=args.backend,
        format=args.format if format_name is None else format_name,
        lazy=args.lazy,
        read_kwargs=_read_kwargs_from_args(args),
    )


def _read_tabulars_from_args(args: argparse.Namespace):
    return read_tabulars(
        args.paths,
        backend=args.backend,
        format=args.format,
        lazy=args.lazy,
        read_kwargs=_read_kwargs_from_args(args),
    )


def _table_row_count(table) -> int:
    if hasattr(table, "height"):
        return int(getattr(table, "height"))
    if hasattr(table, "shape"):
        return int(getattr(table, "shape")[0])
    if hasattr(table, "__len__"):
        return int(len(table))
    return 0


def _table_col_count(table) -> int:
    return len(getattr(table, "columns", []))


def _inferred_merge_key(left, right) -> str:
    left_columns = list(getattr(left, "columns", []))
    right_columns = list(getattr(right, "columns", []))

    if len(left_columns) == 0 or len(right_columns) == 0:
        raise ValueError(
            "Cannot infer merge key: both tables must expose at least one column."
        )

    left_first = left_columns[0]
    right_first = right_columns[0]
    if left_first != right_first:
        raise ValueError(
            f"Cannot infer merge key because first column differs: left={left_first!r}, right={right_first!r}. "
            "Use --on or --left-on/--right-on."
        )

    return left_first


def _preview_config(args: argparse.Namespace, frame) -> tuple[int, int | None, list[str], bool]:
    row_count = _table_row_count(frame)
    col_count = _table_col_count(frame)
    notes: list[str] = []

    if args.display_all:
        notes.append(f"note: --display-all is set; showing all {row_count} rows and {col_count} columns.")
        if args.head != row_count:
            notes.append("note: --display-all overrides --head.")
        if args.cols is not None:
            notes.append("note: --display-all overrides --cols.")
        return row_count, col_count, notes, True

    preview_rows = args.head if args.head <= row_count else row_count
    if args.head > row_count:
        notes.append(f"note: --head={args.head} exceeds available rows ({row_count}); showing all {row_count} rows.")

    if args.cols is None:
        preview_cols = col_count
    else:
        preview_cols = args.cols
        if args.cols > col_count:
            notes.append(
                f"note: --cols={args.cols} exceeds available columns ({col_count}); showing all {col_count} columns."
            )
            preview_cols = col_count
    return preview_rows, preview_cols, notes, False


def _print_output(message: str, frame, args: argparse.Namespace, backend: str) -> None:
    print(message)
    print("-" * len(message))
    loaded = collect_tabular(frame, backend=backend)
    preview_rows, preview_cols, notes, display_all = _preview_config(args, loaded)
    for note in notes:
        print(note)

    preview = preview_tabular(
        loaded,
        n=preview_rows,
        cols=preview_cols,
        display_all=display_all,
        backend=backend,
    )
    print(preview)


def _write_or_print_output(
    table,
    args: argparse.Namespace,
    backend: str,
    operation: str,
) -> int:
    if getattr(args, "output", None):
        write_tabular(table, args.output, backend=backend)
        print(f"written: {args.output} ({backend})")
        return 0
    _print_output(operation, table, args, backend)
    return 0


def _run_preview(args: argparse.Namespace) -> int:
    try:
        frame = _read_tabular_from_args(args, args.paths)
        _print_output(f"preview ({args.backend})", frame, args, args.backend)
    except (UnsupportedBackendError, LazyBackendUnsupportedError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_concat(args: argparse.Namespace) -> int:
    try:
        frames = _read_tabulars_from_args(args)
        result = concat_tabular(frames, backend=args.backend)
        _print_output(f"concat ({args.backend})", result, args, args.backend)
    except (UnsupportedBackendError, LazyBackendUnsupportedError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_merge(args: argparse.Namespace) -> int:
    try:
        how = args.how
        on = args.on
        left_on = args.left_on
        right_on = args.right_on

        if len(args.paths) < 2:
            raise ValueError("Merge requires at least two input paths.")

        if len(args.paths) == 2:
            left_format = args.left_format if args.left_format is not None else args.format
            right_format = args.right_format if args.right_format is not None else args.format

            left = _read_tabular_from_args(args, args.paths[0], format_name=left_format)
            right = _read_tabular_from_args(args, args.paths[1], format_name=right_format)

            if how == "cross":
                if on is not None or left_on is not None or right_on is not None:
                    raise ValueError("Cross joins do not use join keys. Omit --on and --left-on/--right-on.")
            elif on is None and left_on is None and right_on is None:
                on = _inferred_merge_key(left, right)
                print(f"note: inferred merge key from first column: {on!r}")

            merged = merge_tabular(
                left,
                right,
                on=on,
                left_on=left_on,
                right_on=right_on,
                how=how,
                backend=args.backend,
            )
        else:
            if how == "cross":
                raise ValueError("Cross joins are only supported when merging exactly two inputs.")
            if left_on is not None or right_on is not None:
                raise ValueError("--left-on/--right-on are only supported when merging exactly two inputs.")
            if on is None:
                raise ValueError("--on is required when merging more than two inputs.")
            if args.left_format is not None or args.right_format is not None:
                raise ValueError("--left-format/--right-format are only supported when merging exactly two inputs.")

            tables = [
                _read_tabular_from_args(args, path)
                for path in args.paths
            ]
            merged = merge_tabulars(
                tables,
                on=on,
                how=how,
                backend=args.backend,
            )

        return _write_or_print_output(merged, args, args.backend, f"merge ({args.backend})")
    except (UnsupportedBackendError, LazyBackendUnsupportedError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_inspect_dtypes(args: argparse.Namespace) -> int:
    try:
        frame = _read_tabular_from_args(args, args.paths)
        result = inspect_dtypes(frame, backend=args.backend)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (UnsupportedBackendError, LazyBackendUnsupportedError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_inspect_describe(args: argparse.Namespace) -> int:
    try:
        frame = _read_tabular_from_args(args, args.paths)
        result = inspect_describe(frame, backend=args.backend)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (UnsupportedBackendError, LazyBackendUnsupportedError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_inspect_nulls(args: argparse.Namespace) -> int:
    try:
        frame = _read_tabular_from_args(args, args.paths)
        null_summary = inspect_nulls(frame, backend=args.backend)
        null_columns = inspect_columns_with_nulls(frame, backend=args.backend)
        print("null_count:")
        print(json.dumps(null_summary, indent=2, sort_keys=True))
        print("columns_with_nulls:")
        print(json.dumps(null_columns, indent=2, sort_keys=True))
    except (UnsupportedBackendError, LazyBackendUnsupportedError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_clean_replace_values(args: argparse.Namespace) -> int:
    try:
        frame = _read_tabular_from_args(args, args.paths)
        result = replace_invalid_values(
            frame,
            columns=args.columns,
            invalid_values=[_coerce_cli_value(value) for value in args.invalid_values],
            backend=args.backend,
        )
        return _write_or_print_output(result, args, args.backend, f"replace-values ({args.backend})")
    except (UnsupportedBackendError, LazyBackendUnsupportedError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_clean_fill_missing(args: argparse.Namespace) -> int:
    try:
        frame = _read_tabular_from_args(args, args.paths)
        columns = list(args.columns) if args.columns else None
        if args.strategy == "median":
            result = fill_missing_with_median(
                frame,
                columns=columns,
                backend=args.backend,
            )
            operation = "fill-missing (median)"
        else:
            result = fill_missing_with_mode(
                frame,
                columns=columns,
                backend=args.backend,
            )
            operation = "fill-missing (mode)"
        return _write_or_print_output(result, args, args.backend, operation)
    except (UnsupportedBackendError, LazyBackendUnsupportedError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_clean_drop_rows(args: argparse.Namespace) -> int:
    try:
        frame = _read_tabular_from_args(args, args.paths)
        row_numbers = list(args.row_numbers) if args.row_numbers else []
        if args.id_values:
            id_values = [_coerce_cli_value(value) for value in args.id_values]
            if not args.id_column:
                raise ValueError("--id-values requires --id-column.")
            if row_numbers:
                raise ValueError("Use either --row-numbers or --id-values, not both.")
            result = drop_rows_by_id_values(
                frame,
                id_column=args.id_column,
                id_values=id_values,
                backend=args.backend,
            )
            operation = "drop-rows-by-id"
        elif row_numbers:
            result = drop_rows_by_row_numbers(frame, row_numbers=row_numbers, backend=args.backend)
            operation = "drop-rows-by-number"
        else:
            raise ValueError("Provide --row-numbers or --id-values/--id-column.")
        return _write_or_print_output(result, args, args.backend, operation)
    except (UnsupportedBackendError, LazyBackendUnsupportedError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic tabular I/O CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="Preview a table")
    preview.add_argument("paths", nargs="+", help="One or more input paths")
    _add_common(preview)
    preview.set_defaults(handler=_run_preview)

    concat = subparsers.add_parser("concat", help="Concatenate input tables")
    concat.add_argument("paths", nargs="+", help="Input paths")
    _add_common(concat)
    concat.set_defaults(handler=_run_concat)

    merge = subparsers.add_parser("merge", help="Merge tabular inputs")
    merge.add_argument("paths", nargs="+", help="Input paths (two or more)")
    _add_common(merge)
    merge.add_argument("--on", help="Join key shared by both tables")
    merge.add_argument("--left-on", dest="left_on", help="Left join key when keys are different")
    merge.add_argument("--right-on", dest="right_on", help="Right join key when keys are different")
    merge.add_argument(
        "--left-format",
        dest="left_format",
        choices=_SUPPORTED_FORMATS,
        help="Left input format (overrides --format)",
    )
    merge.add_argument(
        "--right-format",
        dest="right_format",
        choices=_SUPPORTED_FORMATS,
        help="Right input format (overrides --format)",
    )
    merge.add_argument(
        "--how",
        choices=_SUPPORTED_HOW,
        default="inner",
        help="Join type: inner, left, right, outer, full, or cross (cross ignores all key arguments).",
    )
    merge.add_argument("--output", help="Write merged table to this path")
    merge.set_defaults(handler=_run_merge)

    inspect = subparsers.add_parser("inspect", help="Inspect table metadata")
    inspect_subparsers = inspect.add_subparsers(dest="inspect_command", required=True)

    inspect_dtypes_parser = inspect_subparsers.add_parser("dtypes", help="Show inferred dtypes")
    _add_inspect_common(inspect_dtypes_parser)
    inspect_dtypes_parser.set_defaults(handler=_run_inspect_dtypes)

    inspect_describe_parser = inspect_subparsers.add_parser("describe", help="Show describe-like column stats")
    _add_inspect_common(inspect_describe_parser)
    inspect_describe_parser.set_defaults(handler=_run_inspect_describe)

    inspect_nulls_parser = inspect_subparsers.add_parser("nulls", help="Show null summary")
    _add_inspect_common(inspect_nulls_parser)
    inspect_nulls_parser.set_defaults(handler=_run_inspect_nulls)

    clean = subparsers.add_parser("clean", help="Transform tables")
    clean_subparsers = clean.add_subparsers(dest="clean_command", required=True)

    replace_parser = clean_subparsers.add_parser("replace-values", help="Replace explicit sentinel/invalid values")
    _add_clean_common(replace_parser)
    replace_parser.add_argument("--columns", nargs="+", required=True, help="Columns to clean")
    replace_parser.add_argument(
        "--invalid-values",
        nargs="+",
        required=True,
        help="Values to replace (space separated)",
    )
    replace_parser.set_defaults(handler=_run_clean_replace_values)

    fill_missing_parser = clean_subparsers.add_parser("fill-missing", help="Fill missing values")
    _add_clean_common(fill_missing_parser)
    fill_missing_parser.add_argument(
        "--strategy",
        choices=("median", "mode"),
        default="median",
        help="Missing fill strategy",
    )
    fill_missing_parser.add_argument("--columns", nargs="+", help="Optional explicit columns")
    fill_missing_parser.set_defaults(handler=_run_clean_fill_missing)

    drop_rows_parser = clean_subparsers.add_parser("drop-rows", help="Drop rows by row numbers or ID values")
    _add_clean_common(drop_rows_parser)
    drop_rows_parser.add_argument(
        "--row-numbers",
        nargs="+",
        type=_non_negative_integer,
        help="0-based row positions to drop",
    )
    drop_rows_parser.add_argument("--id-column", help="ID column for row filtering")
    drop_rows_parser.add_argument(
        "--id-values",
        nargs="+",
        help="Values to drop from id-column",
    )
    drop_rows_parser.set_defaults(handler=_run_clean_drop_rows)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
