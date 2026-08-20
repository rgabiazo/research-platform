from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .pandas_ops import pandas_dataframe_to_records
from .polars_ops import polars_dataframe_to_records
from .records import (
    DEFAULT_INPUT_ROW_INDEX_FIELD,
    DataframeRecordsProvenanceRow,
    DataframeRecordsQcRow,
    DataframeRecordsResult,
    _copy_record,
    _error_result,
    _json_safe_value,
    _tuple_of_strings,
    dataframe_to_records,
    plan_dataframe_records_adapter,
)

TABULAR_ASSOCIATION_SOURCE_ROWS_VERSION = "research_platform.io.dataframe.association_records.v1"
TABULAR_ASSOCIATION_SOURCE_ROWS_RUNTIME_BACKEND = "records"
TABULAR_ASSOCIATION_SOURCE_ROWS_BACKENDS = frozenset({"records", "dataframe_like", "pandas", "polars"})


@dataclass(frozen=True)
class TabularAssociationSourceRowsResult:
    valid: bool
    status: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    source_rows_by_id: Mapping[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    conversion_results_by_id: Mapping[str, DataframeRecordsResult] = field(default_factory=dict)
    qc_rows: tuple[DataframeRecordsQcRow, ...] = field(default_factory=tuple)
    provenance_rows: tuple[DataframeRecordsProvenanceRow, ...] = field(default_factory=tuple)
    source_count: int = 0
    converted_source_count: int = 0
    failed_source_count: int = 0
    row_count_by_source_id: Mapping[str, int] = field(default_factory=dict)
    backend_by_source_id: Mapping[str, str] = field(default_factory=dict)
    runtime_backend: str = TABULAR_ASSOCIATION_SOURCE_ROWS_RUNTIME_BACKEND
    will_write: bool = False
    output_written: bool = False
    no_output_written: bool = True
    output_paths_written: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        source_rows_by_id = {
            str(source_id): tuple(_copy_record(record) for record in records)
            for source_id, records in self.source_rows_by_id.items()
        }
        conversion_results_by_id = {str(source_id): result for source_id, result in self.conversion_results_by_id.items()}
        row_count_by_source_id = {str(source_id): int(row_count) for source_id, row_count in self.row_count_by_source_id.items()}
        backend_by_source_id = {str(source_id): str(backend) for source_id, backend in self.backend_by_source_id.items()}

        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "warnings", _tuple_of_strings(self.warnings))
        object.__setattr__(self, "errors", _tuple_of_strings(self.errors))
        object.__setattr__(self, "source_rows_by_id", source_rows_by_id)
        object.__setattr__(self, "conversion_results_by_id", conversion_results_by_id)
        object.__setattr__(self, "qc_rows", tuple(self.qc_rows))
        object.__setattr__(self, "provenance_rows", tuple(self.provenance_rows))
        object.__setattr__(self, "source_count", int(self.source_count))
        object.__setattr__(self, "converted_source_count", int(self.converted_source_count))
        object.__setattr__(self, "failed_source_count", int(self.failed_source_count))
        object.__setattr__(self, "row_count_by_source_id", row_count_by_source_id)
        object.__setattr__(self, "backend_by_source_id", backend_by_source_id)
        object.__setattr__(self, "runtime_backend", TABULAR_ASSOCIATION_SOURCE_ROWS_RUNTIME_BACKEND)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(self, "output_paths_written", ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TABULAR_ASSOCIATION_SOURCE_ROWS_VERSION,
            "valid": self.valid,
            "status": self.status,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "source_rows_by_id": {
                source_id: [_copy_record(record) for record in records]
                for source_id, records in self.source_rows_by_id.items()
            },
            "conversion_results_by_id": {
                source_id: result.to_dict()
                for source_id, result in self.conversion_results_by_id.items()
            },
            "qc_rows": [row.to_dict() for row in self.qc_rows],
            "provenance_rows": [row.to_dict() for row in self.provenance_rows],
            "source_count": self.source_count,
            "converted_source_count": self.converted_source_count,
            "failed_source_count": self.failed_source_count,
            "row_count_by_source_id": dict(self.row_count_by_source_id),
            "backend_by_source_id": dict(self.backend_by_source_id),
            "runtime_backend": self.runtime_backend,
            "will_write": self.will_write,
            "output_written": self.output_written,
            "no_output_written": self.no_output_written,
            "output_paths_written": list(self.output_paths_written),
        }


def prepare_tabular_association_source_rows(
    sources_by_id: Mapping[str, Any],
    *,
    backend_by_source_id: Mapping[str, str] | None = None,
    default_backend: str = "records",
    include_input_row_index: bool = False,
    input_row_index_field: str = DEFAULT_INPUT_ROW_INDEX_FIELD,
    pandas_include_index: bool = False,
    pandas_index_field: str | None = "pandas_index",
    metadata_by_source_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> TabularAssociationSourceRowsResult:
    if not isinstance(sources_by_id, Mapping):
        return _aggregate_input_error("sources_by_id must be a mapping from source_id to row source.")
    if backend_by_source_id is not None and not isinstance(backend_by_source_id, Mapping):
        return _aggregate_input_error("backend_by_source_id must be a mapping when supplied.")
    if metadata_by_source_id is not None and not isinstance(metadata_by_source_id, Mapping):
        return _aggregate_input_error("metadata_by_source_id must be a mapping when supplied.")

    normalized_sources = {str(source_id): source for source_id, source in sources_by_id.items()}
    backend_overrides = {} if backend_by_source_id is None else {str(source_id): str(backend) for source_id, backend in backend_by_source_id.items()}

    source_rows_by_id: dict[str, tuple[dict[str, Any], ...]] = {}
    conversion_results_by_id: dict[str, DataframeRecordsResult] = {}
    qc_rows: list[DataframeRecordsQcRow] = []
    provenance_rows: list[DataframeRecordsProvenanceRow] = []
    row_count_by_source_id: dict[str, int] = {}
    resolved_backend_by_source_id: dict[str, str] = {}
    warnings: list[str] = []
    errors: list[str] = []

    for source_id, source in normalized_sources.items():
        backend = _normalize_backend(backend_overrides.get(source_id, default_backend))
        resolved_backend_by_source_id[source_id] = backend
        metadata = _metadata_for_source(source_id, metadata_by_source_id)

        if backend not in TABULAR_ASSOCIATION_SOURCE_ROWS_BACKENDS:
            result = _unknown_backend_result(
                source,
                backend=backend,
                include_input_row_index=include_input_row_index,
                input_row_index_field=input_row_index_field,
                metadata=metadata,
            )
        else:
            result = _convert_source(
                source,
                backend=backend,
                include_input_row_index=include_input_row_index,
                input_row_index_field=input_row_index_field,
                pandas_include_index=pandas_include_index,
                pandas_index_field=pandas_index_field,
                metadata=metadata,
            )

        conversion_results_by_id[source_id] = result
        qc_rows.extend(result.qc_rows)
        provenance_rows.extend(result.provenance_rows)
        row_count_by_source_id[source_id] = result.row_count
        warnings.extend(f"{source_id}: {warning}" for warning in result.warnings)

        if result.valid:
            source_rows_by_id[source_id] = tuple(_copy_record(record) for record in result.records)
        else:
            errors.extend(f"{source_id}: {error}" for error in result.errors)
            if not result.errors:
                errors.append(f"{source_id}: source conversion failed with status {result.status!r}.")

    failed_source_count = sum(1 for result in conversion_results_by_id.values() if not result.valid)
    converted_source_count = len(conversion_results_by_id) - failed_source_count
    valid = failed_source_count == 0 and not errors

    return TabularAssociationSourceRowsResult(
        valid=valid,
        status="ok" if valid else "error",
        warnings=tuple(warnings),
        errors=tuple(errors),
        source_rows_by_id=source_rows_by_id,
        conversion_results_by_id=conversion_results_by_id,
        qc_rows=tuple(qc_rows),
        provenance_rows=tuple(provenance_rows),
        source_count=len(normalized_sources),
        converted_source_count=converted_source_count,
        failed_source_count=failed_source_count,
        row_count_by_source_id=row_count_by_source_id,
        backend_by_source_id=resolved_backend_by_source_id,
    )


def _convert_source(
    source: Any,
    *,
    backend: str,
    include_input_row_index: bool,
    input_row_index_field: str,
    pandas_include_index: bool,
    pandas_index_field: str | None,
    metadata: Mapping[str, Any],
) -> DataframeRecordsResult:
    if backend in {"records", "dataframe_like"}:
        return dataframe_to_records(
            source,
            requested_backend=backend,
            include_input_row_index=include_input_row_index,
            input_row_index_field=input_row_index_field,
            metadata=metadata,
        )
    if backend == "pandas":
        return pandas_dataframe_to_records(
            source,
            include_index=pandas_include_index,
            index_field=pandas_index_field,
            include_input_row_index=include_input_row_index,
            input_row_index_field=input_row_index_field,
            metadata=metadata,
        )
    return polars_dataframe_to_records(
        source,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=metadata,
    )


def _unknown_backend_result(
    source: Any,
    *,
    backend: str,
    include_input_row_index: bool,
    input_row_index_field: str,
    metadata: Mapping[str, Any],
) -> DataframeRecordsResult:
    spec = plan_dataframe_records_adapter(
        source,
        requested_backend=backend,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=metadata,
        adapter_id=TABULAR_ASSOCIATION_SOURCE_ROWS_VERSION,
    )
    return _error_result(
        spec,
        code="unknown_tabular_association_source_rows_backend",
        message=(
            f"Unknown tabular association source rows backend {backend!r}. "
            f"Supported backends are {sorted(TABULAR_ASSOCIATION_SOURCE_ROWS_BACKENDS)!r}."
        ),
        metadata=metadata,
    )


def _aggregate_input_error(message: str) -> TabularAssociationSourceRowsResult:
    safe_message = str(_json_safe_value(message))
    return TabularAssociationSourceRowsResult(
        valid=False,
        status="error",
        errors=(safe_message,),
        source_count=0,
        converted_source_count=0,
        failed_source_count=0,
    )


def _normalize_backend(value: Any) -> str:
    return str(value).strip().lower()


def _metadata_for_source(
    source_id: str,
    metadata_by_source_id: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source_id": source_id}
    if metadata_by_source_id is None:
        return metadata
    source_metadata = metadata_by_source_id.get(source_id)
    if source_metadata is None:
        return metadata
    metadata.update({str(key): _json_safe_value(value) for key, value in source_metadata.items()})
    metadata["source_id"] = source_id
    return metadata


__all__ = [
    "TABULAR_ASSOCIATION_SOURCE_ROWS_BACKENDS",
    "TABULAR_ASSOCIATION_SOURCE_ROWS_RUNTIME_BACKEND",
    "TABULAR_ASSOCIATION_SOURCE_ROWS_VERSION",
    "TabularAssociationSourceRowsResult",
    "prepare_tabular_association_source_rows",
]
