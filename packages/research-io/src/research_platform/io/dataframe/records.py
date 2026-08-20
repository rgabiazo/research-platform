from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from typing import Any, Iterator

DATAFRAME_RECORDS_ADAPTER_VERSION = "research_platform.io.dataframe.records.v1"
DATAFRAME_RECORDS_RUNTIME_BACKEND = "records"
DATAFRAME_RECORDS_ADAPTER_KIND = "dataframe_records_adapter"
DEFAULT_INPUT_ROW_INDEX_FIELD = "input_row_index"


class DataframeRecordsSourceError(ValueError):
    """Raised when a dataframe-like source cannot be coerced to mapping rows."""

    def __init__(self, message: str, *, code: str = "unsupported_row_source") -> None:
        super().__init__(message)
        self.code = code


def _json_safe_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    return str(_json_safe_value(key))


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, bytes):
        return repr(value)
    if isinstance(value, Mapping):
        return {_json_safe_key(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_json_safe_value(item) for item in sorted(value, key=repr)]

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except TypeError:
            pass

    return str(value)


def _tsv_safe_value(value: Any) -> str:
    safe_value = _json_safe_value(value)
    if safe_value is None:
        text = ""
    elif isinstance(safe_value, dict | list):
        text = json.dumps(safe_value, allow_nan=False, sort_keys=True)
    else:
        text = str(safe_value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _json_safe_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_value(value) for key, value in mapping.items()}


def _tuple_of_strings(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _merge_column_names(*column_groups: Iterable[Any]) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for group in column_groups:
        for column in group:
            name = _json_safe_key(column)
            if name in seen:
                continue
            seen.add(name)
            columns.append(name)
    return tuple(columns)


def _output_paths_tuple(values: Iterable[Any] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return _tuple_of_strings(values)


@dataclass(frozen=True)
class DataframeRecordsAdapterSpec:
    adapter_id: str = DATAFRAME_RECORDS_ADAPTER_VERSION
    requested_backend: str | None = None
    runtime_backend: str = DATAFRAME_RECORDS_RUNTIME_BACKEND
    row_source_kind: str = "unknown"
    include_input_row_index: bool = False
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD
    metadata: Mapping[str, Any] = field(default_factory=dict)
    will_write: bool = False
    output_written: bool = False
    output_paths_written: tuple[str, ...] = field(default_factory=tuple)
    no_output_written: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_id", str(self.adapter_id))
        requested_backend = None if self.requested_backend is None else str(self.requested_backend)
        object.__setattr__(self, "requested_backend", requested_backend)
        object.__setattr__(self, "runtime_backend", DATAFRAME_RECORDS_RUNTIME_BACKEND)
        object.__setattr__(self, "row_source_kind", str(self.row_source_kind))
        object.__setattr__(self, "include_input_row_index", bool(self.include_input_row_index))
        object.__setattr__(self, "input_row_index_field", str(self.input_row_index_field))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "output_paths_written", _output_paths_tuple(self.output_paths_written))
        object.__setattr__(self, "no_output_written", True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "requested_backend": self.requested_backend,
            "runtime_backend": self.runtime_backend,
            "row_source_kind": self.row_source_kind,
            "include_input_row_index": self.include_input_row_index,
            "input_row_index_field": self.input_row_index_field,
            "metadata": _json_safe_mapping(self.metadata),
            "will_write": self.will_write,
            "output_written": self.output_written,
            "output_paths_written": list(self.output_paths_written),
            "no_output_written": self.no_output_written,
        }


@dataclass(frozen=True)
class DataframeRecordsQcRow:
    status: str
    code: str
    message: str
    adapter_id: str = DATAFRAME_RECORDS_ADAPTER_VERSION
    schema_version: str = DATAFRAME_RECORDS_ADAPTER_VERSION
    requested_backend: str | None = None
    runtime_backend: str = DATAFRAME_RECORDS_RUNTIME_BACKEND
    row_source_kind: str = "unknown"
    row_count: int = 0
    observed_column_count: int = 0
    observed_columns: tuple[str, ...] = field(default_factory=tuple)
    include_input_row_index: bool = False
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD
    will_write: bool = False
    output_written: bool = False
    output_paths_written: tuple[str, ...] = field(default_factory=tuple)
    no_output_written: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "code", str(self.code))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "adapter_id", str(self.adapter_id))
        object.__setattr__(self, "schema_version", DATAFRAME_RECORDS_ADAPTER_VERSION)
        requested_backend = None if self.requested_backend is None else str(self.requested_backend)
        object.__setattr__(self, "requested_backend", requested_backend)
        object.__setattr__(self, "runtime_backend", DATAFRAME_RECORDS_RUNTIME_BACKEND)
        object.__setattr__(self, "row_source_kind", str(self.row_source_kind))
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(self, "observed_column_count", int(self.observed_column_count))
        object.__setattr__(self, "observed_columns", _tuple_of_strings(self.observed_columns))
        object.__setattr__(self, "include_input_row_index", bool(self.include_input_row_index))
        object.__setattr__(self, "input_row_index_field", str(self.input_row_index_field))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "output_paths_written", _output_paths_tuple(self.output_paths_written))
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "status": self.status,
            "code": self.code,
            "message": self.message,
            "requested_backend": self.requested_backend,
            "runtime_backend": self.runtime_backend,
            "row_source_kind": self.row_source_kind,
            "row_count": self.row_count,
            "observed_column_count": self.observed_column_count,
            "observed_columns": list(self.observed_columns),
            "include_input_row_index": self.include_input_row_index,
            "input_row_index_field": self.input_row_index_field,
            "will_write": self.will_write,
            "output_written": self.output_written,
            "output_paths_written": list(self.output_paths_written),
            "no_output_written": self.no_output_written,
            "metadata": _json_safe_mapping(self.metadata),
        }

    def to_tsv_row(self) -> dict[str, str]:
        return {key: _tsv_safe_value(value) for key, value in self.to_dict().items()}


@dataclass(frozen=True)
class DataframeRecordsProvenanceRow:
    adapter_id: str = DATAFRAME_RECORDS_ADAPTER_VERSION
    schema_version: str = DATAFRAME_RECORDS_ADAPTER_VERSION
    adapter_kind: str = DATAFRAME_RECORDS_ADAPTER_KIND
    requested_backend: str | None = None
    runtime_backend: str = DATAFRAME_RECORDS_RUNTIME_BACKEND
    row_source_kind: str = "unknown"
    row_count: int = 0
    observed_column_count: int = 0
    include_input_row_index: bool = False
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD
    will_write: bool = False
    output_written: bool = False
    output_paths_written: tuple[str, ...] = field(default_factory=tuple)
    no_output_written: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_id", str(self.adapter_id))
        object.__setattr__(self, "schema_version", DATAFRAME_RECORDS_ADAPTER_VERSION)
        object.__setattr__(self, "adapter_kind", str(self.adapter_kind))
        requested_backend = None if self.requested_backend is None else str(self.requested_backend)
        object.__setattr__(self, "requested_backend", requested_backend)
        object.__setattr__(self, "runtime_backend", DATAFRAME_RECORDS_RUNTIME_BACKEND)
        object.__setattr__(self, "row_source_kind", str(self.row_source_kind))
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(self, "observed_column_count", int(self.observed_column_count))
        object.__setattr__(self, "include_input_row_index", bool(self.include_input_row_index))
        object.__setattr__(self, "input_row_index_field", str(self.input_row_index_field))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "output_paths_written", _output_paths_tuple(self.output_paths_written))
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "adapter_kind": self.adapter_kind,
            "requested_backend": self.requested_backend,
            "runtime_backend": self.runtime_backend,
            "row_source_kind": self.row_source_kind,
            "row_count": self.row_count,
            "observed_column_count": self.observed_column_count,
            "include_input_row_index": self.include_input_row_index,
            "input_row_index_field": self.input_row_index_field,
            "will_write": self.will_write,
            "output_written": self.output_written,
            "output_paths_written": list(self.output_paths_written),
            "no_output_written": self.no_output_written,
            "metadata": _json_safe_mapping(self.metadata),
        }

    def to_tsv_row(self) -> dict[str, str]:
        return {key: _tsv_safe_value(value) for key, value in self.to_dict().items()}


@dataclass(frozen=True)
class DataframeRecordsResult:
    adapter_spec: DataframeRecordsAdapterSpec
    records: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    valid: bool = False
    status: str = "error"
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    row_source_kind: str = "unknown"
    row_count: int = 0
    observed_column_count: int = 0
    observed_columns: tuple[str, ...] = field(default_factory=tuple)
    qc_rows: tuple[DataframeRecordsQcRow, ...] = field(default_factory=tuple)
    provenance_rows: tuple[DataframeRecordsProvenanceRow, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    will_write: bool = False
    output_written: bool = False
    output_paths_written: tuple[str, ...] = field(default_factory=tuple)
    no_output_written: bool = True

    def __post_init__(self) -> None:
        copied_records = tuple(_copy_record(record) for record in self.records)
        object.__setattr__(self, "records", copied_records)
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "errors", _tuple_of_strings(self.errors))
        object.__setattr__(self, "warnings", _tuple_of_strings(self.warnings))
        object.__setattr__(self, "row_source_kind", str(self.row_source_kind))
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(self, "observed_column_count", int(self.observed_column_count))
        object.__setattr__(self, "observed_columns", _tuple_of_strings(self.observed_columns))
        object.__setattr__(self, "qc_rows", tuple(self.qc_rows))
        object.__setattr__(self, "provenance_rows", tuple(self.provenance_rows))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "output_paths_written", _output_paths_tuple(self.output_paths_written))
        object.__setattr__(self, "no_output_written", True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DATAFRAME_RECORDS_ADAPTER_VERSION,
            "adapter_spec": self.adapter_spec.to_dict(),
            "valid": self.valid,
            "status": self.status,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "row_source_kind": self.row_source_kind,
            "row_count": self.row_count,
            "observed_column_count": self.observed_column_count,
            "observed_columns": list(self.observed_columns),
            "records": [_copy_record(record) for record in self.records],
            "qc_rows": [row.to_dict() for row in self.qc_rows],
            "provenance_rows": [row.to_dict() for row in self.provenance_rows],
            "metadata": _json_safe_mapping(self.metadata),
            "will_write": self.will_write,
            "output_written": self.output_written,
            "output_paths_written": list(self.output_paths_written),
            "no_output_written": self.no_output_written,
        }


@dataclass(frozen=True)
class DataframeRecordsAdapter:
    spec: DataframeRecordsAdapterSpec = field(default_factory=DataframeRecordsAdapterSpec)
    adapter_kind: str = DATAFRAME_RECORDS_ADAPTER_KIND

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_kind", str(self.adapter_kind))

    def inspect(self, source: Any) -> DataframeRecordsResult:
        return inspect_dataframe_like_source(
            source,
            requested_backend=self.spec.requested_backend,
            include_input_row_index=self.spec.include_input_row_index,
            input_row_index_field=self.spec.input_row_index_field,
            metadata=self.spec.metadata,
            adapter_id=self.spec.adapter_id,
        )

    def to_records(self, source: Any) -> DataframeRecordsResult:
        return dataframe_to_records(
            source,
            requested_backend=self.spec.requested_backend,
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


_MISSING = object()


def plan_dataframe_records_adapter(
    source: Any = _MISSING,
    *,
    requested_backend: str | None = None,
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    metadata: Mapping[str, Any] | None = None,
    adapter_id: str = DATAFRAME_RECORDS_ADAPTER_VERSION,
) -> DataframeRecordsAdapterSpec:
    row_source_kind = "unknown" if source is _MISSING else detect_dataframe_like_protocol(source)
    return DataframeRecordsAdapterSpec(
        adapter_id=adapter_id,
        requested_backend=requested_backend,
        row_source_kind=row_source_kind,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata={} if metadata is None else metadata,
    )


def detect_dataframe_like_protocol(source: Any) -> str:
    if source is None:
        return "none"
    if isinstance(source, str):
        return "string"
    if isinstance(source, bytes | bytearray):
        return "bytes"
    if isinstance(source, Mapping):
        return "single_mapping"
    if callable(getattr(source, "to_dicts", None)):
        return "to_dicts"
    if callable(getattr(source, "to_records", None)):
        return "to_records"
    if callable(getattr(source, "iter_rows", None)):
        return "iter_rows"

    rows = getattr(source, "rows", _MISSING)
    if rows is not _MISSING:
        return "rows_method" if callable(rows) else "rows_attribute"

    records = getattr(source, "records", _MISSING)
    if records is not _MISSING:
        return "records_method" if callable(records) else "records_attribute"

    if isinstance(source, Sequence) and not isinstance(source, str | bytes | bytearray):
        return "mapping_sequence"

    return "unsupported"


def inspect_dataframe_like_source(
    source: Any,
    *,
    requested_backend: str | None = None,
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    metadata: Mapping[str, Any] | None = None,
    adapter_id: str = DATAFRAME_RECORDS_ADAPTER_VERSION,
) -> DataframeRecordsResult:
    return dataframe_to_records(
        source,
        requested_backend=requested_backend,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=metadata,
        adapter_id=adapter_id,
    )


def dataframe_to_records(
    source: Any,
    *,
    requested_backend: str | None = None,
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    metadata: Mapping[str, Any] | None = None,
    adapter_id: str = DATAFRAME_RECORDS_ADAPTER_VERSION,
) -> DataframeRecordsResult:
    metadata = {} if metadata is None else dict(metadata)
    spec = plan_dataframe_records_adapter(
        source,
        requested_backend=requested_backend,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=metadata,
        adapter_id=adapter_id,
    )

    if include_input_row_index and not input_row_index_field:
        return _error_result(
            spec,
            code="invalid_input_row_index_field",
            message="include_input_row_index=True requires a non-empty input_row_index_field.",
            metadata=metadata,
        )

    try:
        raw_rows, row_source_kind = _extract_row_source(source)
        records = _coerce_mapping_rows(raw_rows)
        observed_columns = _observed_columns(source, records)
        if include_input_row_index:
            records = _add_input_row_index(records, input_row_index_field)
            observed_columns = _merge_column_names(observed_columns, (input_row_index_field,))
    except DataframeRecordsSourceError as exc:
        return _error_result(
            spec,
            code=exc.code,
            message=str(exc),
            metadata=metadata,
        )

    spec = DataframeRecordsAdapterSpec(
        adapter_id=adapter_id,
        requested_backend=requested_backend,
        row_source_kind=row_source_kind,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=metadata,
    )
    return _success_result(spec, records, observed_columns, metadata=metadata)


def iter_dataframe_records(
    source: Any,
    *,
    requested_backend: str | None = None,
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    metadata: Mapping[str, Any] | None = None,
    adapter_id: str = DATAFRAME_RECORDS_ADAPTER_VERSION,
) -> Iterator[dict[str, Any]]:
    result = dataframe_to_records(
        source,
        requested_backend=requested_backend,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=metadata,
        adapter_id=adapter_id,
    )
    return iter(result.records)


def _extract_row_source(source: Any) -> tuple[Any, str]:
    if source is None:
        raise DataframeRecordsSourceError("None is not a supported dataframe-like row source.", code="none_source")
    if isinstance(source, str):
        raise DataframeRecordsSourceError("String values are not supported as dataframe-like row sources.", code="string_source")
    if isinstance(source, bytes | bytearray):
        raise DataframeRecordsSourceError("Bytes values are not supported as dataframe-like row sources.", code="bytes_source")
    if isinstance(source, Mapping):
        raise DataframeRecordsSourceError(
            "A single mapping is not a supported row source; provide a sequence of mapping rows.",
            code="single_mapping_source",
        )

    to_dicts = getattr(source, "to_dicts", None)
    if callable(to_dicts):
        return _call_no_argument_row_method(to_dicts, "to_dicts"), "to_dicts"

    to_records = getattr(source, "to_records", None)
    if callable(to_records):
        return _call_no_argument_row_method(to_records, "to_records"), "to_records"

    iter_rows = getattr(source, "iter_rows", None)
    if callable(iter_rows):
        try:
            return iter_rows(named=True), "iter_rows_named"
        except TypeError:
            try:
                return iter_rows(), "iter_rows"
            except TypeError as exc:
                raise DataframeRecordsSourceError(
                    "iter_rows() must support named=True or a zero-argument mapping-row iterator.",
                    code="unsupported_iter_rows",
                ) from exc

    rows = getattr(source, "rows", _MISSING)
    if rows is not _MISSING:
        if callable(rows):
            return _call_no_argument_row_method(rows, "rows"), "rows_method"
        return rows, "rows_attribute"

    records = getattr(source, "records", _MISSING)
    if records is not _MISSING:
        if callable(records):
            return _call_no_argument_row_method(records, "records"), "records_method"
        return records, "records_attribute"

    if isinstance(source, Sequence) and not isinstance(source, str | bytes | bytearray):
        return source, "mapping_sequence"

    raise DataframeRecordsSourceError(
        "Unsupported dataframe-like row source. Expected mapping-row sequences or a generic record-producing protocol.",
        code="unsupported_row_source",
    )


def _call_no_argument_row_method(method: Any, method_name: str) -> Any:
    try:
        return method()
    except TypeError as exc:
        raise DataframeRecordsSourceError(
            f"{method_name} must be callable with no arguments.",
            code=f"unsupported_{method_name}",
        ) from exc


def _coerce_mapping_rows(raw_rows: Any) -> tuple[dict[str, Any], ...]:
    if raw_rows is None:
        raise DataframeRecordsSourceError("The row-producing protocol returned None.", code="empty_protocol_result")
    if isinstance(raw_rows, str):
        raise DataframeRecordsSourceError("The row-producing protocol returned a string.", code="string_rows")
    if isinstance(raw_rows, bytes | bytearray):
        raise DataframeRecordsSourceError("The row-producing protocol returned bytes.", code="bytes_rows")
    if isinstance(raw_rows, Mapping):
        raise DataframeRecordsSourceError(
            "The row-producing protocol returned a single mapping instead of an iterable of mapping rows.",
            code="single_mapping_rows",
        )

    try:
        iterator = iter(raw_rows)
    except TypeError as exc:
        raise DataframeRecordsSourceError(
            "The row-producing protocol did not return an iterable of mapping rows.",
            code="non_iterable_rows",
        ) from exc

    records: list[dict[str, Any]] = []
    for index, row in enumerate(iterator):
        records.append(_copy_mapping_row(row, row_number=index))
    return tuple(records)


def _copy_mapping_row(row: Any, *, row_number: int) -> dict[str, Any]:
    items = _mapping_items(row)
    if items is None:
        raise DataframeRecordsSourceError(
            f"Row {row_number} is not mapping-like; positional tuple/list rows are not supported.",
            code="non_mapping_row",
        )

    record: dict[str, Any] = {}
    for key, value in items:
        safe_key = _json_safe_key(key)
        if safe_key in record:
            raise DataframeRecordsSourceError(
                f"Row {row_number} contains duplicate column name {safe_key!r} after key normalization.",
                code="duplicate_column_name",
            )
        record[safe_key] = _json_safe_value(value)
    return record


def _mapping_items(row: Any) -> tuple[tuple[Any, Any], ...] | None:
    if isinstance(row, Mapping):
        return tuple(row.items())
    items = getattr(row, "items", None)
    if not callable(items):
        return None
    try:
        raw_items = tuple(items())
    except TypeError:
        return None

    normalized_items: list[tuple[Any, Any]] = []
    for item in raw_items:
        if not isinstance(item, tuple | list) or len(item) != 2:
            return None
        normalized_items.append((item[0], item[1]))
    return tuple(normalized_items)


def _copy_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_value(value) for key, value in record.items()}


def _observed_columns(source: Any, records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return _merge_column_names(_observed_columns_from_attribute(source), _observed_columns_from_records(records))


def _observed_columns_from_attribute(source: Any) -> tuple[str, ...]:
    columns = getattr(source, "columns", None)
    if columns is None or callable(columns):
        return ()
    if isinstance(columns, str | bytes | bytearray | Mapping):
        return ()
    if isinstance(columns, set | frozenset):
        return _merge_column_names(sorted(columns, key=repr))
    try:
        return _merge_column_names(columns)
    except TypeError:
        return ()


def _observed_columns_from_records(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return _merge_column_names(*(record.keys() for record in records))


def _add_input_row_index(
    records: Sequence[Mapping[str, Any]],
    input_row_index_field: str,
) -> tuple[dict[str, Any], ...]:
    for record in records:
        if input_row_index_field in record:
            raise DataframeRecordsSourceError(
                f"Cannot add input row index field {input_row_index_field!r}; at least one input record already contains it.",
                code="input_row_index_collision",
            )

    indexed_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        copied = _copy_record(record)
        copied[input_row_index_field] = index
        indexed_records.append(copied)
    return tuple(indexed_records)


def _success_result(
    spec: DataframeRecordsAdapterSpec,
    records: tuple[dict[str, Any], ...],
    observed_columns: tuple[str, ...],
    *,
    metadata: Mapping[str, Any],
) -> DataframeRecordsResult:
    qc_row = DataframeRecordsQcRow(
        adapter_id=spec.adapter_id,
        status="ok",
        code="records_coerced",
        message="Dataframe-like source was coerced to copied ordered records.",
        requested_backend=spec.requested_backend,
        row_source_kind=spec.row_source_kind,
        row_count=len(records),
        observed_column_count=len(observed_columns),
        observed_columns=observed_columns,
        include_input_row_index=spec.include_input_row_index,
        input_row_index_field=spec.input_row_index_field,
        metadata=metadata,
    )
    provenance_row = _provenance_row(spec, row_count=len(records), observed_column_count=len(observed_columns), metadata=metadata)
    return DataframeRecordsResult(
        adapter_spec=spec,
        records=records,
        valid=True,
        status="ok",
        row_source_kind=spec.row_source_kind,
        row_count=len(records),
        observed_column_count=len(observed_columns),
        observed_columns=observed_columns,
        qc_rows=(qc_row,),
        provenance_rows=(provenance_row,),
        metadata=metadata,
    )


def _error_result(
    spec: DataframeRecordsAdapterSpec,
    *,
    code: str,
    message: str,
    metadata: Mapping[str, Any],
) -> DataframeRecordsResult:
    safe_message = str(_json_safe_value(message))
    qc_row = DataframeRecordsQcRow(
        adapter_id=spec.adapter_id,
        status="error",
        code=code,
        message=safe_message,
        requested_backend=spec.requested_backend,
        row_source_kind=spec.row_source_kind,
        row_count=0,
        observed_column_count=0,
        observed_columns=(),
        include_input_row_index=spec.include_input_row_index,
        input_row_index_field=spec.input_row_index_field,
        metadata=metadata,
    )
    provenance_row = _provenance_row(spec, row_count=0, observed_column_count=0, metadata=metadata)
    return DataframeRecordsResult(
        adapter_spec=spec,
        records=(),
        valid=False,
        status="error",
        errors=(safe_message,),
        row_source_kind=spec.row_source_kind,
        row_count=0,
        observed_column_count=0,
        observed_columns=(),
        qc_rows=(qc_row,),
        provenance_rows=(provenance_row,),
        metadata=metadata,
    )


def _provenance_row(
    spec: DataframeRecordsAdapterSpec,
    *,
    row_count: int,
    observed_column_count: int,
    metadata: Mapping[str, Any],
) -> DataframeRecordsProvenanceRow:
    return DataframeRecordsProvenanceRow(
        adapter_id=spec.adapter_id,
        requested_backend=spec.requested_backend,
        row_source_kind=spec.row_source_kind,
        row_count=row_count,
        observed_column_count=observed_column_count,
        include_input_row_index=spec.include_input_row_index,
        input_row_index_field=spec.input_row_index_field,
        metadata=metadata,
    )


__all__ = [
    "DATAFRAME_RECORDS_ADAPTER_VERSION",
    "DATAFRAME_RECORDS_RUNTIME_BACKEND",
    "DATAFRAME_RECORDS_ADAPTER_KIND",
    "DEFAULT_INPUT_ROW_INDEX_FIELD",
    "DataframeRecordsAdapter",
    "DataframeRecordsAdapterSpec",
    "DataframeRecordsProvenanceRow",
    "DataframeRecordsQcRow",
    "DataframeRecordsResult",
    "DataframeRecordsSourceError",
    "dataframe_to_records",
    "detect_dataframe_like_protocol",
    "inspect_dataframe_like_source",
    "iter_dataframe_records",
    "plan_dataframe_records_adapter",
]
