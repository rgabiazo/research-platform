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

PANDAS_RECORDS_ADAPTER_VERSION = "research_platform.io.dataframe.pandas_records.v1"
PANDAS_RECORDS_ADAPTER_KIND = "pandas_records_adapter"
DEFAULT_PANDAS_INDEX_FIELD = "pandas_index"


@lru_cache(maxsize=1)
def _load_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(f"pandas backend is unavailable: {exc}") from exc
    return pd


@lru_cache(maxsize=1)
def _load_pandas_api_types() -> Any:
    try:
        from pandas.api import types as pandas_api_types
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(f"pandas api types are unavailable: {exc}") from exc
    return pandas_api_types


def _normalize_string_columns(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(column) for column in value]


def _merge_string_dtypes(dtype: Any, string_columns: Sequence[str]) -> Any:
    if not string_columns:
        return dtype
    overrides = {column: str for column in string_columns}
    if dtype is None:
        return overrides
    if isinstance(dtype, Mapping):
        return {**dict(dtype), **overrides}
    return dtype


class PandasBackend(BackendProtocol):
    name = "pandas"
    supports_lazy = False
    supported_formats = {"csv", "tsv", "txt", "parquet", "feather"}

    def __init__(self) -> None:
        _load_pandas()

    def read(
        self,
        paths: Sequence[Path],
        *,
        format: SupportedFormat,
        lazy: bool = False,
        read_kwargs: Mapping[str, Any] | None = None,
    ) -> FrameLike:
        if lazy:
            raise LazyBackendUnsupportedError("pandas backend does not support lazy=True.")

        pd = _load_pandas()
        read_kwargs = dict(read_kwargs or {})
        string_columns = _normalize_string_columns(read_kwargs.pop("string_columns", None))
        if string_columns and format in {"csv", "tsv", "txt"}:
            read_kwargs["dtype"] = _merge_string_dtypes(read_kwargs.get("dtype"), string_columns)
        if format in {"csv", "txt"}:
            separator = read_kwargs.pop("separator", ",")
            frames = [pd.read_csv(path, sep=separator, **read_kwargs) for path in paths]
            return pd.concat(frames) if len(frames) > 1 else frames[0]

        if format == "tsv":
            frames = [pd.read_csv(path, sep="\t", **read_kwargs) for path in paths]
            return pd.concat(frames) if len(frames) > 1 else frames[0]

        if format == "parquet":
            frames = [pd.read_parquet(path, **read_kwargs) for path in paths]
            return pd.concat(frames) if len(frames) > 1 else frames[0]

        if format == "feather":
            frames = [pd.read_feather(path, **read_kwargs) for path in paths]
            return pd.concat(frames) if len(frames) > 1 else frames[0]

        raise ValueError(f"Unsupported format '{format}'.")

    def collect(self, table: FrameLike) -> FrameLike:
        return table

    def head(self, table: FrameLike, n: int) -> FrameLike:
        return table.head(n)

    def is_lazy(self, table: FrameLike) -> bool:
        return False

    def format_supported(self, format: str) -> bool:
        return format in self.supported_formats

    def concat(
        self,
        tables: Sequence[FrameLike],
    ) -> FrameLike:
        if not tables:
            raise ValueError("No tables provided for concat.")
        pd = _load_pandas()
        return pd.concat(list(tables))

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
        if how not in {"inner", "left", "right", "outer", "full", "cross"}:
            raise ValueError(f"Unsupported how '{how}' for pandas backend.")

        pandas_how = "outer" if how == "full" else how
        return left.merge(
            right,
            how=pandas_how,
            on=on,
            left_on=left_on,
            right_on=right_on,
        )

    def dtypes(self, table: FrameLike) -> dict[str, str]:
        return {str(column): str(dtype) for column, dtype in table.dtypes.items()}

    def describe(self, table: FrameLike) -> dict[str, dict[str, Any]]:
        stats = table.describe(include="all").T
        return {
            str(column): {
                str(statistic): value.item() if hasattr(value, "item") else value
                for statistic, value in row.items()
            }
            for column, row in stats.to_dict(orient="index").items()
        }

    def null_summary(self, table: FrameLike) -> dict[str, int]:
        return {str(column): int(count) for column, count in table.isna().sum().items()}

    def replace_invalid_values(
        self,
        table: FrameLike,
        columns: Sequence[str],
        invalid_values: Sequence[Any],
        replacement: Any = None,
    ) -> FrameLike:
        if not columns:
            return table
        self._require_columns(table, columns)
        return table.replace({col: list(invalid_values) for col in columns}, replacement)

    def fill_missing_median(self, table: FrameLike, columns: Sequence[str] | None = None) -> FrameLike:
        pd = _load_pandas()
        pandas_api_types = _load_pandas_api_types()
        if not columns:
            columns = [str(column) for column in table.columns if pandas_api_types.is_numeric_dtype(table[column])]
        table_filled = table.copy()
        for column in columns:
            if column not in table_filled.columns:
                raise ValueError(f"Unknown column {column!r}.")
            if table_filled[column].isna().sum() == 0:
                continue
            numeric_series = pd.to_numeric(table_filled[column], errors="coerce")
            median = numeric_series.median()
            if pd.isna(median):
                continue
            table_filled[column] = numeric_series.fillna(median)
        return table_filled

    def fill_missing_mode(self, table: FrameLike, columns: Sequence[str] | None = None) -> FrameLike:
        pandas_api_types = _load_pandas_api_types()
        if not columns:
            columns = [str(column) for column in table.columns if not pandas_api_types.is_numeric_dtype(table[column])]
        table_filled = table.copy()
        for column in columns:
            if column not in table_filled.columns:
                raise ValueError(f"Unknown column {column!r}.")
            if table_filled[column].isna().sum() == 0:
                continue
            modes = table_filled[column].mode(dropna=True)
            if modes.empty:
                continue
            table_filled[column] = table_filled[column].fillna(modes.iloc[0])
        return table_filled

    def drop_rows_by_numbers(self, table: FrameLike, row_numbers: Sequence[int]) -> FrameLike:
        if not row_numbers:
            return table
        row_set = set(row_numbers)
        if any(row < 0 for row in row_set):
            raise ValueError("Row numbers must be non-negative integers.")
        max_row = len(table) - 1
        keep = [row for row in range(len(table)) if row not in row_set and row <= max_row]
        return table.iloc[keep]

    def drop_rows_by_id_values(self, table: FrameLike, id_column: str, id_values: Sequence[Any]) -> FrameLike:
        if id_column not in table.columns:
            raise ValueError(f"Column {id_column!r} does not exist.")
        return table[~table[id_column].isin(list(id_values))]

    def write(
        self,
        table: FrameLike,
        path: str | Path,
        format: str,
        write_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        write_kwargs = dict(write_kwargs or {})
        if format in {"csv", "txt"}:
            separator = write_kwargs.pop("separator", ",")
            table.to_csv(path, sep=separator, index=False, **write_kwargs)
            return
        if format == "tsv":
            table.to_csv(path, sep="\t", index=False, **write_kwargs)
            return
        if format == "parquet":
            table.to_parquet(path, **write_kwargs)
            return
        if format == "feather":
            table.to_feather(path, **write_kwargs)
            return
        raise ValueError(f"Unsupported format '{format}'.")

    @staticmethod
    def _require_columns(table: FrameLike, columns: Sequence[str]) -> None:
        for column in columns:
            if column not in table.columns:
                raise ValueError(f"Unknown column {column!r}.")


_MISSING = object()


@dataclass(frozen=True)
class PandasRecordsAdapter:
    spec: DataframeRecordsAdapterSpec = field(default_factory=lambda: plan_pandas_records_adapter())
    include_index: bool = False
    index_field: str = DEFAULT_PANDAS_INDEX_FIELD
    adapter_kind: str = PANDAS_RECORDS_ADAPTER_KIND

    def __post_init__(self) -> None:
        options = _pandas_records_options_from_metadata(self.spec.metadata)
        include_index = bool(self.include_index) or bool(options.get("include_index", False))
        index_field = self.index_field
        if index_field == DEFAULT_PANDAS_INDEX_FIELD and "index_field" in options:
            index_field = options["index_field"]
        object.__setattr__(self, "include_index", include_index)
        object.__setattr__(self, "index_field", _normalize_pandas_index_field(index_field))
        object.__setattr__(self, "adapter_kind", str(self.adapter_kind))

    def inspect(self, source: Any) -> DataframeRecordsResult:
        return inspect_pandas_dataframe_source(
            source,
            include_index=self.include_index,
            index_field=self.index_field,
            include_input_row_index=self.spec.include_input_row_index,
            input_row_index_field=self.spec.input_row_index_field,
            metadata=self.spec.metadata,
            adapter_id=self.spec.adapter_id,
        )

    def to_records(self, source: Any) -> DataframeRecordsResult:
        return pandas_dataframe_to_records(
            source,
            include_index=self.include_index,
            index_field=self.index_field,
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
            "include_index": self.include_index,
            "index_field": self.index_field,
            "spec": self.spec.to_dict(),
        }


def plan_pandas_records_adapter(
    source: Any = _MISSING,
    *,
    include_index: bool = False,
    index_field: str | None = DEFAULT_PANDAS_INDEX_FIELD,
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    metadata: Mapping[str, Any] | None = None,
    adapter_id: str = PANDAS_RECORDS_ADAPTER_VERSION,
) -> DataframeRecordsAdapterSpec:
    normalized_index_field = _normalize_pandas_index_field(index_field)
    adapter_metadata = _pandas_records_metadata(
        metadata,
        include_index=include_index,
        index_field=normalized_index_field,
    )
    row_source_kind = "unknown" if source is _MISSING else detect_pandas_dataframe_protocol(source)
    return DataframeRecordsAdapterSpec(
        adapter_id=adapter_id,
        requested_backend="pandas",
        row_source_kind=row_source_kind,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=adapter_metadata,
    )


def detect_pandas_dataframe_protocol(source: Any) -> str:
    if source is None:
        return "none"

    try:
        pd = _load_pandas()
    except ModuleNotFoundError:
        return "pandas_unavailable"

    if isinstance(source, pd.DataFrame):
        return "pandas_dataframe"
    if isinstance(source, pd.Series):
        return "pandas_series"
    return "unsupported"


def inspect_pandas_dataframe_source(
    source: Any,
    *,
    include_index: bool = False,
    index_field: str | None = DEFAULT_PANDAS_INDEX_FIELD,
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    metadata: Mapping[str, Any] | None = None,
    adapter_id: str = PANDAS_RECORDS_ADAPTER_VERSION,
) -> DataframeRecordsResult:
    return pandas_dataframe_to_records(
        source,
        include_index=include_index,
        index_field=index_field,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=metadata,
        adapter_id=adapter_id,
    )


def pandas_dataframe_to_records(
    source: Any,
    *,
    include_index: bool = False,
    index_field: str | None = DEFAULT_PANDAS_INDEX_FIELD,
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    metadata: Mapping[str, Any] | None = None,
    adapter_id: str = PANDAS_RECORDS_ADAPTER_VERSION,
) -> DataframeRecordsResult:
    normalized_index_field = _normalize_pandas_index_field(index_field)
    adapter_metadata = _pandas_records_metadata(
        metadata,
        include_index=include_index,
        index_field=normalized_index_field,
    )
    spec = plan_pandas_records_adapter(
        source,
        include_index=include_index,
        index_field=normalized_index_field,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=metadata,
        adapter_id=adapter_id,
    )

    if include_index and not normalized_index_field:
        return _error_result(
            spec,
            code="invalid_pandas_index_field",
            message="include_index=True requires a non-empty index_field.",
            metadata=adapter_metadata,
        )

    if include_input_row_index and not input_row_index_field:
        return _error_result(
            spec,
            code="invalid_input_row_index_field",
            message="include_input_row_index=True requires a non-empty input_row_index_field.",
            metadata=adapter_metadata,
        )

    try:
        pd = _load_pandas()
    except ModuleNotFoundError as exc:
        return _error_result(
            spec,
            code="pandas_unavailable",
            message=str(exc),
            metadata=adapter_metadata,
        )

    if not isinstance(source, pd.DataFrame):
        code = "pandas_series_source" if isinstance(source, pd.Series) else "unsupported_pandas_source"
        return _error_result(
            spec,
            code=code,
            message="Pandas records adapter accepts pandas.DataFrame objects only.",
            metadata=adapter_metadata,
        )

    safe_columns = _safe_pandas_columns(source.columns)
    if len(set(safe_columns)) != len(safe_columns):
        return _error_result(
            spec,
            code="duplicate_column_name",
            message="Pandas DataFrame contains duplicate column names after key normalization.",
            metadata=adapter_metadata,
        )

    if include_index and normalized_index_field in safe_columns:
        return _error_result(
            spec,
            code="pandas_index_field_collision",
            message=f"Cannot add pandas index field {normalized_index_field!r}; DataFrame already contains that column.",
            metadata=adapter_metadata,
        )

    if include_input_row_index and input_row_index_field in safe_columns:
        return _error_result(
            spec,
            code="input_row_index_collision",
            message=f"Cannot add input row index field {input_row_index_field!r}; DataFrame already contains that column.",
            metadata=adapter_metadata,
        )

    records = _pandas_dataframe_records(source, safe_columns=safe_columns, include_index=include_index, index_field=normalized_index_field)
    if include_input_row_index:
        records = _add_input_row_index(records, input_row_index_field)

    observed_columns = _merge_column_names(safe_columns)
    if include_index:
        observed_columns = _merge_column_names(observed_columns, (normalized_index_field,))
    if include_input_row_index:
        observed_columns = _merge_column_names(observed_columns, (input_row_index_field,))

    spec = DataframeRecordsAdapterSpec(
        adapter_id=adapter_id,
        requested_backend="pandas",
        row_source_kind="pandas_dataframe",
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=adapter_metadata,
    )
    return _success_result(spec, records, observed_columns, metadata=adapter_metadata)


def iter_pandas_dataframe_records(
    source: Any,
    *,
    include_index: bool = False,
    index_field: str | None = DEFAULT_PANDAS_INDEX_FIELD,
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    metadata: Mapping[str, Any] | None = None,
    adapter_id: str = PANDAS_RECORDS_ADAPTER_VERSION,
) -> Iterator[dict[str, Any]]:
    result = pandas_dataframe_to_records(
        source,
        include_index=include_index,
        index_field=index_field,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=metadata,
        adapter_id=adapter_id,
    )
    return iter(result.records)


def _normalize_pandas_index_field(index_field: str | None) -> str:
    if index_field is None:
        return DEFAULT_PANDAS_INDEX_FIELD
    return str(index_field)


def _pandas_records_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    include_index: bool,
    index_field: str,
) -> dict[str, Any]:
    adapter_metadata = {} if metadata is None else dict(metadata)
    adapter_metadata["pandas_records_adapter"] = {
        "include_index": bool(include_index),
        "index_field": index_field,
    }
    return adapter_metadata


def _pandas_records_options_from_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    options = metadata.get("pandas_records_adapter", {})
    if isinstance(options, Mapping):
        return options
    return {}


def _safe_pandas_columns(columns: Any) -> tuple[str, ...]:
    return tuple(_json_safe_key(column) for column in list(columns))


def _pandas_dataframe_records(
    source: Any,
    *,
    safe_columns: Sequence[str],
    include_index: bool,
    index_field: str,
) -> tuple[dict[str, Any], ...]:
    pd = _load_pandas()
    index_values = list(source.index) if include_index else []
    records: list[dict[str, Any]] = []
    for row_number, values in enumerate(source.itertuples(index=False, name=None)):
        record = {
            column: _pandas_json_safe_value(value, pd)
            for column, value in zip(safe_columns, values)
        }
        if include_index:
            record[index_field] = _pandas_json_safe_value(index_values[row_number], pd)
        records.append(record)
    return tuple(records)


def _pandas_json_safe_value(value: Any, pd: Any) -> Any:
    if isinstance(value, Mapping):
        return {_json_safe_key(key): _pandas_json_safe_value(item, pd) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_pandas_json_safe_value(item, pd) for item in value]
    if isinstance(value, set | frozenset):
        return [_pandas_json_safe_value(item, pd) for item in sorted(value, key=repr)]
    if _is_pandas_missing_scalar(value, pd):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            item_value = item()
        except (TypeError, ValueError):
            pass
        else:
            if item_value is not value:
                return _pandas_json_safe_value(item_value, pd)
    return _json_safe_value(value)


def _is_pandas_missing_scalar(value: Any, pd: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool):
        return missing
    if getattr(missing, "shape", None) == ():
        return bool(missing)
    return False


__all__ = [
    "DEFAULT_PANDAS_INDEX_FIELD",
    "PANDAS_RECORDS_ADAPTER_KIND",
    "PANDAS_RECORDS_ADAPTER_VERSION",
    "PandasBackend",
    "PandasRecordsAdapter",
    "detect_pandas_dataframe_protocol",
    "inspect_pandas_dataframe_source",
    "iter_pandas_dataframe_records",
    "pandas_dataframe_to_records",
    "plan_pandas_records_adapter",
]
