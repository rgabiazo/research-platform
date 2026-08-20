from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ._backend_protocol import BackendProtocol, LazyBackendUnsupportedError, MergeHow
from ._types import FrameLike, SupportedFormat
from .records import (
    DEFAULT_INPUT_ROW_INDEX_FIELD,
    DataframeRecordsAdapterSpec,
    DataframeRecordsResult,
    _add_input_row_index,
    _error_result,
    _json_safe_key,
    _json_safe_value,
    _merge_column_names,
    _success_result,
)

POLARS_RECORDS_ADAPTER_VERSION = "research_platform.io.dataframe.polars_records.v1"
POLARS_RECORDS_ADAPTER_KIND = "polars_records_adapter"


@lru_cache(maxsize=1)
def _load_polars() -> Any:
    try:
        import polars as pl
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(f"polars backend is unavailable: {exc}") from exc
    return pl


def _normalize_string_columns(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(column) for column in value]


def _merge_string_schema_overrides(
    schema_overrides: Any,
    string_columns: Sequence[str],
) -> Any:
    if not string_columns:
        return schema_overrides
    pl = _load_polars()
    overrides = {column: pl.Utf8 for column in string_columns}
    if schema_overrides is None:
        return overrides
    if isinstance(schema_overrides, Mapping):
        return {**dict(schema_overrides), **overrides}
    return schema_overrides


class PolarsBackend(BackendProtocol):
    name = "polars"
    supports_lazy = True
    supported_formats = {"csv", "tsv", "txt", "parquet", "feather"}

    def __init__(self) -> None:
        _load_polars()

    def read(
        self,
        paths: Sequence[Path],
        *,
        format: SupportedFormat,
        lazy: bool = False,
        read_kwargs: Mapping[str, Any] | None = None,
    ) -> FrameLike:
        pl = _load_polars()
        read_kwargs = dict(read_kwargs or {})
        string_columns = _normalize_string_columns(read_kwargs.pop("string_columns", None))
        if string_columns and format in {"csv", "tsv", "txt"}:
            read_kwargs["schema_overrides"] = _merge_string_schema_overrides(
                read_kwargs.get("schema_overrides"),
                string_columns,
            )

        if format == "csv":
            separator = read_kwargs.pop("separator", ",")
            read_fn = pl.scan_csv if lazy else pl.read_csv
            frames = [read_fn(str(path), separator=separator, **read_kwargs) for path in paths]
            return _concat(frames)

        if format == "tsv":
            read_fn = pl.scan_csv if lazy else pl.read_csv
            frames = [read_fn(str(path), separator="\t", **read_kwargs) for path in paths]
            return _concat(frames)

        if format == "txt":
            separator = read_kwargs.pop("separator", ",")
            read_fn = pl.scan_csv if lazy else pl.read_csv
            frames = [read_fn(str(path), separator=separator, **read_kwargs) for path in paths]
            return _concat(frames)

        if format == "parquet":
            if lazy:
                frames = [pl.scan_parquet(str(path), **read_kwargs) for path in paths]
                return _concat(frames)
            return _concat([pl.read_parquet(str(path), **read_kwargs) for path in paths])

        if format == "feather":
            if lazy:
                frames = [pl.scan_ipc(str(path), **read_kwargs) for path in paths]
                return _concat(frames)
            return _concat([pl.read_ipc(str(path), **read_kwargs) for path in paths])

        raise ValueError(f"Unsupported format '{format}'.")

    def collect(self, table: FrameLike) -> FrameLike:
        if not self.is_lazy(table):
            return table
        return table.collect()

    def head(self, table: FrameLike, n: int) -> FrameLike:
        if self.is_lazy(table):
            return table.head(n)
        return table.head(n)

    def is_lazy(self, table: FrameLike) -> bool:
        pl = _load_polars()
        return isinstance(table, pl.LazyFrame)

    def format_supported(self, format: str) -> bool:
        return format in self.supported_formats

    def concat(
        self,
        tables: Sequence[FrameLike],
    ) -> FrameLike:
        return _concat(tables)

    def merge(
        self,
        left: FrameLike,
        right: FrameLike,
        *,
        on: str | None,
        left_on: str | None,
        right_on: str | None,
        how: MergeHow,
    ) -> FrameLike:
        if how == "cross":
            if on is not None or left_on is not None or right_on is not None:
                raise ValueError("Cross joins do not use join keys. Omit --on and --left-on/--right-on.")
            return left.join(right, how=how)
        if on is not None:
            return left.join(right, on=on, how=how)
        if left_on is None or right_on is None:
            raise ValueError("Both --left-on and --right-on are required when --on is not provided.")
        return left.join(right, left_on=left_on, right_on=right_on, how=how)

    def dtypes(self, table: FrameLike) -> dict[str, str]:
        if self.is_lazy(table):
            schema = table.collect_schema()
        else:
            schema = table.schema
        return {str(name): str(dtype) for name, dtype in schema.items()}

    def describe(self, table: FrameLike) -> dict[str, dict[str, Any]]:
        frame = self._ensure_eager(table)
        summary = frame.describe()
        data = summary.to_dict(as_series=False)
        statistics = list(data.pop("statistic"))
        result: dict[str, dict[str, Any]] = {}
        for name in summary.columns:
            if name == "statistic":
                continue
            result[name] = {str(stat): value for stat, value in zip(statistics, data[name])}
        return result

    def null_summary(self, table: FrameLike) -> dict[str, int]:
        frame = self._ensure_eager(table)
        null_counts = frame.null_count()
        return {column: int(value) for column, value in zip(frame.columns, null_counts.row(0))}

    def replace_invalid_values(
        self,
        table: FrameLike,
        columns: Sequence[str],
        invalid_values: Sequence[Any],
        replacement: Any = None,
    ) -> FrameLike:
        if not columns:
            return table
        pl = _load_polars()
        self._require_columns(table, columns)
        frame = self._ensure_eager(table)
        if not invalid_values:
            return frame
        if replacement is None:
            replacement_expr = None
        else:
            replacement_expr = pl.lit(replacement)
        return frame.with_columns(
            [
                pl.when(~pl.col(column).is_in(invalid_values))
                .then(pl.col(column))
                .otherwise(replacement_expr)
                .alias(column)
                for column in columns
            ]
        )

    def fill_missing_median(
        self,
        table: FrameLike,
        columns: Sequence[str] | None = None,
    ) -> FrameLike:
        pl = _load_polars()
        frame = self._ensure_eager(table)
        if columns is None:
            columns = [name for name, dtype in frame.schema.items() if self._is_numeric_dtype(dtype)]
        for column in columns:
            if column not in frame.columns:
                raise ValueError(f"Unknown column {column!r}.")
            median = frame.select(pl.col(column).median()).item(0, 0)
            if median is None:
                continue
            frame = frame.with_columns(pl.col(column).fill_null(median).alias(column))
        return frame

    def fill_missing_mode(
        self,
        table: FrameLike,
        columns: Sequence[str] | None = None,
    ) -> FrameLike:
        pl = _load_polars()
        frame = self._ensure_eager(table)
        if columns is None:
            columns = [name for name, dtype in frame.schema.items() if not self._is_numeric_dtype(dtype)]
        for column in columns:
            if column not in frame.columns:
                raise ValueError(f"Unknown column {column!r}.")
            series = frame.get_column(column).drop_nulls()
            if series.len() == 0:
                continue
            value_counts = series.value_counts()
            if value_counts.height == 0:
                continue
            count_column = "counts" if "counts" in value_counts.columns else "count"
            value_column = next((col for col in value_counts.columns if col != count_column), None)
            if value_column is None:
                continue
            mode_row = value_counts.sort(count_column, descending=True).row(0, named=True)
            mode_value = mode_row[value_column]
            frame = frame.with_columns(pl.col(column).fill_null(mode_value).alias(column))
        return frame

    def drop_rows_by_numbers(self, table: FrameLike, row_numbers: Sequence[int]) -> FrameLike:
        pl = _load_polars()
        frame = self._ensure_eager(table)
        if not row_numbers:
            return frame
        row_set = set(int(row) for row in row_numbers)
        if any(row < 0 for row in row_set):
            raise ValueError("Row numbers must be non-negative integers.")
        keep = frame.height - len(row_set)
        if keep <= 0:
            return frame.head(0)
        filtered = frame.with_row_index("_row_num").filter(~pl.col("_row_num").is_in(row_set)).drop("_row_num")
        return filtered

    def drop_rows_by_id_values(
        self,
        table: FrameLike,
        id_column: str,
        id_values: Sequence[Any],
    ) -> FrameLike:
        pl = _load_polars()
        frame = self._ensure_eager(table)
        if id_column not in frame.columns:
            raise ValueError(f"Column {id_column!r} does not exist.")
        return frame.filter(~pl.col(id_column).is_in(id_values))

    def write(
        self,
        table: FrameLike,
        path: str | Path,
        format: str,
        write_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        table_to_write = self._ensure_eager(table)
        write_kwargs = dict(write_kwargs or {})

        if format in {"csv", "txt"}:
            separator = write_kwargs.pop("separator", ",")
            table_to_write.write_csv(path, separator=separator, **write_kwargs)
            return
        if format == "tsv":
            table_to_write.write_csv(path, separator="\t", **write_kwargs)
            return
        if format == "parquet":
            table_to_write.write_parquet(path, **write_kwargs)
            return
        if format == "feather":
            table_to_write.write_ipc(path, **write_kwargs)
            return
        raise ValueError(f"Unsupported format '{format}'.")

    @staticmethod
    def _is_numeric_dtype(dtype: Any) -> bool:
        if hasattr(dtype, "is_numeric"):
            return bool(dtype.is_numeric())
        return str(dtype).lower() in {"int64", "int32", "int16", "int8", "uint8", "uint16", "uint32", "uint64", "float64", "float32"}

    @staticmethod
    def _require_columns(table: FrameLike, columns: Sequence[str]) -> None:
        frame_columns = set(getattr(table, "columns", []))
        for column in columns:
            if column not in frame_columns:
                raise ValueError(f"Unknown column {column!r}.")

    @staticmethod
    def _ensure_eager(table: FrameLike) -> Any:
        pl = _load_polars()
        if isinstance(table, pl.LazyFrame):
            return table.collect()
        return table


def _concat(frames: Sequence[FrameLike]) -> FrameLike:
    if len(frames) == 1:
        return frames[0]
    pl = _load_polars()
    return pl.concat(frames)


def collect_if_lazy(table: FrameLike) -> FrameLike:
    pl = _load_polars()
    if not isinstance(table, (pl.DataFrame, pl.LazyFrame)):
        raise LazyBackendUnsupportedError("Polars backend can only collect Polars data objects.")
    if isinstance(table, pl.LazyFrame):
        return table.collect()
    return table


_MISSING = object()


@dataclass(frozen=True)
class PolarsRecordsAdapter:
    spec: DataframeRecordsAdapterSpec = field(default_factory=lambda: plan_polars_records_adapter())
    adapter_kind: str = POLARS_RECORDS_ADAPTER_KIND

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_kind", str(self.adapter_kind))

    def inspect(self, source: Any) -> DataframeRecordsResult:
        return inspect_polars_dataframe_source(
            source,
            include_input_row_index=self.spec.include_input_row_index,
            input_row_index_field=self.spec.input_row_index_field,
            metadata=self.spec.metadata,
            adapter_id=self.spec.adapter_id,
        )

    def to_records(self, source: Any) -> DataframeRecordsResult:
        return polars_dataframe_to_records(
            source,
            include_input_row_index=self.spec.include_input_row_index,
            input_row_index_field=self.spec.input_row_index_field,
            metadata=self.spec.metadata,
            adapter_id=self.spec.adapter_id,
        )

    def iter_records(self, source: Any) -> Iterator[dict[str, Any]]:
        return iter(self.to_records(source).records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_kind": self.adapter_kind,
            "spec": self.spec.to_dict(),
        }


def plan_polars_records_adapter(
    source: Any = _MISSING,
    *,
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    metadata: Mapping[str, Any] | None = None,
    adapter_id: str = POLARS_RECORDS_ADAPTER_VERSION,
) -> DataframeRecordsAdapterSpec:
    row_source_kind = "unknown" if source is _MISSING else detect_polars_dataframe_protocol(source)
    return DataframeRecordsAdapterSpec(
        adapter_id=adapter_id,
        requested_backend="polars",
        row_source_kind=row_source_kind,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata={} if metadata is None else metadata,
    )


def detect_polars_dataframe_protocol(source: Any) -> str:
    if source is None:
        return "none"

    try:
        pl = _load_polars()
    except ModuleNotFoundError:
        return "polars_unavailable"

    if isinstance(source, pl.DataFrame):
        return "polars_dataframe"
    if isinstance(source, pl.Series):
        return "polars_series"
    if isinstance(source, pl.LazyFrame):
        return "polars_lazyframe"
    return "unsupported"


def inspect_polars_dataframe_source(
    source: Any,
    *,
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    metadata: Mapping[str, Any] | None = None,
    adapter_id: str = POLARS_RECORDS_ADAPTER_VERSION,
) -> DataframeRecordsResult:
    return polars_dataframe_to_records(
        source,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=metadata,
        adapter_id=adapter_id,
    )


def polars_dataframe_to_records(
    source: Any,
    *,
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    metadata: Mapping[str, Any] | None = None,
    adapter_id: str = POLARS_RECORDS_ADAPTER_VERSION,
) -> DataframeRecordsResult:
    adapter_metadata = {} if metadata is None else dict(metadata)
    spec = plan_polars_records_adapter(
        source,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=adapter_metadata,
        adapter_id=adapter_id,
    )

    if include_input_row_index and not input_row_index_field:
        return _error_result(
            spec,
            code="invalid_input_row_index_field",
            message="include_input_row_index=True requires a non-empty input_row_index_field.",
            metadata=adapter_metadata,
        )

    try:
        pl = _load_polars()
    except ModuleNotFoundError as exc:
        return _error_result(
            spec,
            code="polars_unavailable",
            message=str(exc),
            metadata=adapter_metadata,
        )

    if not isinstance(source, pl.DataFrame):
        if isinstance(source, pl.Series):
            code = "polars_series_source"
        elif isinstance(source, pl.LazyFrame):
            code = "polars_lazyframe_source"
        else:
            code = "unsupported_polars_source"
        return _error_result(
            spec,
            code=code,
            message="Polars records adapter accepts polars.DataFrame objects only.",
            metadata=adapter_metadata,
        )

    safe_columns = _safe_polars_columns(source.columns)
    if len(set(safe_columns)) != len(safe_columns):
        return _error_result(
            spec,
            code="duplicate_column_name",
            message="Polars DataFrame contains duplicate column names after key normalization.",
            metadata=adapter_metadata,
        )

    if include_input_row_index and input_row_index_field in safe_columns:
        return _error_result(
            spec,
            code="input_row_index_collision",
            message=f"Cannot add input row index field {input_row_index_field!r}; DataFrame already contains that column.",
            metadata=adapter_metadata,
        )

    records = _polars_dataframe_records(source, safe_columns=safe_columns)
    if include_input_row_index:
        records = _add_input_row_index(records, input_row_index_field)

    observed_columns = _merge_column_names(safe_columns)
    if include_input_row_index:
        observed_columns = _merge_column_names(observed_columns, (input_row_index_field,))

    spec = DataframeRecordsAdapterSpec(
        adapter_id=adapter_id,
        requested_backend="polars",
        row_source_kind="polars_dataframe",
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=adapter_metadata,
    )
    return _success_result(spec, records, observed_columns, metadata=adapter_metadata)


def iter_polars_dataframe_records(
    source: Any,
    *,
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    metadata: Mapping[str, Any] | None = None,
    adapter_id: str = POLARS_RECORDS_ADAPTER_VERSION,
) -> Iterator[dict[str, Any]]:
    result = polars_dataframe_to_records(
        source,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=metadata,
        adapter_id=adapter_id,
    )
    return iter(result.records)


def _safe_polars_columns(columns: Any) -> tuple[str, ...]:
    return tuple(_json_safe_key(column) for column in list(columns))


def _polars_dataframe_records(
    source: Any,
    *,
    safe_columns: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for values in source.iter_rows(named=False):
        record = {
            column: _polars_json_safe_value(value)
            for column, value in zip(safe_columns, values)
        }
        records.append(record)
    return tuple(records)


def _polars_json_safe_value(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            item_value = item()
        except (TypeError, ValueError):
            pass
        else:
            if item_value is not value:
                return _polars_json_safe_value(item_value)
    return _json_safe_value(value)


__all__ = [
    "POLARS_RECORDS_ADAPTER_KIND",
    "POLARS_RECORDS_ADAPTER_VERSION",
    "PolarsBackend",
    "PolarsRecordsAdapter",
    "collect_if_lazy",
    "detect_polars_dataframe_protocol",
    "inspect_polars_dataframe_source",
    "iter_polars_dataframe_records",
    "plan_polars_records_adapter",
    "polars_dataframe_to_records",
]
