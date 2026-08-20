"""Generic publication table shaping and manifest writers.

The helpers in this module consume already-computed rectangular rows. They do
not perform inference, distance computation, extraction, visualization, or
domain-specific analysis work.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = "research_platform.analysis.publication_tables.v1"

_DISPLAY_TSV = "display_tsv"
_DISPLAY_CSV = "display_csv"
_DISPLAY_MARKDOWN = "display_markdown"
_MACHINE_TSV = "machine_tsv"
_MACHINE_CSV = "machine_csv"
_MACHINE_JSON = "machine_json"
_MANIFEST = "manifest"
_OUTPUT_NAMES = (
    _DISPLAY_TSV,
    _DISPLAY_CSV,
    _DISPLAY_MARKDOWN,
    _MACHINE_TSV,
    _MACHINE_CSV,
    _MACHINE_JSON,
    _MANIFEST,
)


@dataclass(frozen=True)
class NumericFormatSpec:
    """Display formatting for numeric values."""

    precision: int = 3
    trim_trailing_zeros: bool = False
    missing_value: str | None = None

    def __post_init__(self) -> None:
        if self.precision < 0:
            raise ValueError("precision must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PValueFormatSpec:
    """Display formatting for p-like values."""

    precision: int = 3
    threshold: float | None = 0.001
    include_leading_zero: bool = True
    missing_value: str | None = None

    def __post_init__(self) -> None:
        if self.precision < 0:
            raise ValueError("precision must be non-negative.")
        if self.threshold is not None and self.threshold <= 0:
            raise ValueError("threshold must be positive when provided.")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ConfidenceIntervalFormatSpec:
    """Display formatting for estimate plus confidence interval columns."""

    estimate: NumericFormatSpec = field(default_factory=NumericFormatSpec)
    bounds: NumericFormatSpec = field(default_factory=NumericFormatSpec)
    template: str = "{estimate} [{low}, {high}]"
    missing_value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PublicationFormatSpec:
    """Default display formatting for a publication table."""

    missing_value: str = ""
    numeric: NumericFormatSpec = field(default_factory=NumericFormatSpec)
    p_value: PValueFormatSpec = field(default_factory=PValueFormatSpec)
    q_value: PValueFormatSpec = field(default_factory=PValueFormatSpec)
    confidence_interval: ConfidenceIntervalFormatSpec = field(default_factory=ConfidenceIntervalFormatSpec)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PublicationSourceSpec:
    """Source rows for a publication table.

    In-memory rows can be supplied to the public functions directly. File-backed
    sources use ``path`` and optionally ``json_rows_key`` for JSON objects that
    contain row lists under a named key.
    """

    source_id: str = "source"
    path: str | Path | None = None
    format: str | None = None
    json_rows_key: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        if self.format is not None:
            normalized = str(self.format).lower()
            if normalized not in {"tsv", "csv", "json"}:
                raise ValueError("PublicationSourceSpec.format must be one of: tsv, csv, json.")
            object.__setattr__(self, "format", normalized)
        if self.json_rows_key is not None:
            object.__setattr__(self, "json_rows_key", _non_empty_text(self.json_rows_key, field_name="json_rows_key"))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PublicationColumnSpec:
    """A display and machine-readable output column mapping."""

    output_name: str
    source: str | None = None
    sources: Sequence[str] = ()
    column_type: str = "field"
    value_filter: Mapping[str, Any] = field(default_factory=dict)
    ci_low_source: str | None = None
    ci_high_source: str | None = None
    template: str | None = None
    constant: Any = None
    required: bool = False
    missing_value: str | None = None
    numeric_format: NumericFormatSpec | None = None
    p_value_format: PValueFormatSpec | None = None
    q_value_format: PValueFormatSpec | None = None
    confidence_interval_format: ConfidenceIntervalFormatSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_name", _non_empty_text(self.output_name, field_name="output_name"))
        column_type = str(self.column_type or "field")
        if column_type not in {"field", "numeric", "p_value", "q_value", "confidence_interval", "template", "constant"}:
            raise ValueError(
                "PublicationColumnSpec.column_type must be one of: "
                "field, numeric, p_value, q_value, confidence_interval, template, constant."
            )
        object.__setattr__(self, "column_type", column_type)
        object.__setattr__(self, "sources", tuple(str(source) for source in self.sources))
        object.__setattr__(self, "value_filter", _json_safe_mapping(self.value_filter))
        if self.source is not None:
            object.__setattr__(self, "source", str(self.source))
        if self.ci_low_source is not None:
            object.__setattr__(self, "ci_low_source", str(self.ci_low_source))
        if self.ci_high_source is not None:
            object.__setattr__(self, "ci_high_source", str(self.ci_high_source))
        if self.template is not None:
            object.__setattr__(self, "template", str(self.template))

    def source_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        if self.source is not None:
            columns.append(self.source)
        columns.extend(self.sources)
        if self.ci_low_source is not None:
            columns.append(self.ci_low_source)
        if self.ci_high_source is not None:
            columns.append(self.ci_high_source)
        columns.extend(str(column) for column in self.value_filter)
        return tuple(dict.fromkeys(columns))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PublicationTableSpec:
    """Specification for shaping source rows into publication table rows."""

    table_id: str
    columns: Sequence[PublicationColumnSpec]
    format: PublicationFormatSpec = field(default_factory=PublicationFormatSpec)
    filters: Mapping[str, Any] = field(default_factory=dict)
    status_values: Sequence[str] = ()
    sort_by: Sequence[str] = ()
    metadata_columns: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "table_id", _non_empty_text(self.table_id, field_name="table_id"))
        object.__setattr__(self, "columns", tuple(self.columns))
        if not self.columns:
            raise ValueError("PublicationTableSpec.columns must contain at least one column.")
        output_names = [column.output_name for column in self.columns]
        duplicates = sorted({name for name in output_names if output_names.count(name) > 1})
        if duplicates:
            raise ValueError(f"PublicationTableSpec.columns contains duplicate output names: {', '.join(duplicates)}")
        object.__setattr__(self, "filters", _json_safe_mapping(self.filters))
        object.__setattr__(self, "status_values", tuple(str(value) for value in self.status_values))
        object.__setattr__(self, "sort_by", tuple(str(column) for column in self.sort_by))
        object.__setattr__(self, "metadata_columns", tuple(str(column) for column in self.metadata_columns))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PublicationOutputSpec:
    """Configured publication table and manifest output paths."""

    output_root: str | Path
    display_tsv_path: str | Path | None = None
    display_csv_path: str | Path | None = None
    display_markdown_path: str | Path | None = None
    machine_tsv_path: str | Path | None = None
    machine_csv_path: str | Path | None = None
    machine_json_path: str | Path | None = None
    manifest_path: str | Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PublicationManifestRow:
    """One source or output artifact record in a publication manifest."""

    role: str
    name: str
    path: str | None = None
    relative_path: str | None = None
    file_format: str | None = None
    row_count: int | None = None
    columns: Sequence[str] = ()
    sha256: str | None = None
    exists: bool | None = None
    written: bool = False
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _non_empty_text(self.role, field_name="role"))
        object.__setattr__(self, "name", _non_empty_text(self.name, field_name="name"))
        object.__setattr__(self, "columns", tuple(str(column) for column in self.columns))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PublicationManifest:
    """JSON-safe manifest for publication table outputs."""

    schema_version: str
    table_id: str
    source_paths: Sequence[str] = ()
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    source_row_counts: Mapping[str, int] = field(default_factory=dict)
    output_paths: Mapping[str, str] = field(default_factory=dict)
    output_hashes: Mapping[str, str] = field(default_factory=dict)
    output_row_counts: Mapping[str, int] = field(default_factory=dict)
    table_spec_summary: Mapping[str, Any] = field(default_factory=dict)
    column_mappings: Sequence[Mapping[str, Any]] = ()
    filters: Mapping[str, Any] = field(default_factory=dict)
    sort_settings: Sequence[str] = ()
    format_settings: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    rows: Sequence[PublicationManifestRow] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_paths", tuple(str(path) for path in self.source_paths))
        object.__setattr__(self, "source_hashes", _json_safe_mapping(self.source_hashes))
        object.__setattr__(self, "source_row_counts", _json_safe_mapping(self.source_row_counts))
        object.__setattr__(self, "output_paths", _json_safe_mapping(self.output_paths))
        object.__setattr__(self, "output_hashes", _json_safe_mapping(self.output_hashes))
        object.__setattr__(self, "output_row_counts", _json_safe_mapping(self.output_row_counts))
        object.__setattr__(self, "table_spec_summary", _json_safe_mapping(self.table_spec_summary))
        object.__setattr__(self, "column_mappings", tuple(_json_safe_mapping(mapping) for mapping in self.column_mappings))
        object.__setattr__(self, "filters", _json_safe_mapping(self.filters))
        object.__setattr__(self, "sort_settings", tuple(str(setting) for setting in self.sort_settings))
        object.__setattr__(self, "format_settings", _json_safe_mapping(self.format_settings))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "provenance", _json_safe_mapping(self.provenance))
        object.__setattr__(self, "rows", tuple(self.rows))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PublicationTablePlan:
    """Dry-run preview for publication table outputs."""

    schema_version: str
    table_id: str
    status: str
    will_write: bool
    output_written: bool
    display_rows: Sequence[Mapping[str, Any]]
    machine_rows: Sequence[Mapping[str, Any]]
    output_paths: Mapping[str, str]
    column_mappings: Sequence[Mapping[str, Any]]
    manifest: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "display_rows", tuple(_json_safe_mapping(row) for row in self.display_rows))
        object.__setattr__(self, "machine_rows", tuple(_json_safe_mapping(row) for row in self.machine_rows))
        object.__setattr__(self, "output_paths", _json_safe_mapping(self.output_paths))
        object.__setattr__(self, "column_mappings", tuple(_json_safe_mapping(mapping) for mapping in self.column_mappings))
        object.__setattr__(self, "manifest", _json_safe_mapping(self.manifest))
        object.__setattr__(self, "provenance", _json_safe_mapping(self.provenance))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PublicationTableWriteResult:
    """Write result for configured publication table outputs."""

    schema_version: str
    table_id: str
    status: str
    will_write: bool
    output_written: bool
    output_paths: Mapping[str, str]
    output_hashes: Mapping[str, str]
    output_row_counts: Mapping[str, int]
    manifest: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_paths", _json_safe_mapping(self.output_paths))
        object.__setattr__(self, "output_hashes", _json_safe_mapping(self.output_hashes))
        object.__setattr__(self, "output_row_counts", _json_safe_mapping(self.output_row_counts))
        object.__setattr__(self, "manifest", _json_safe_mapping(self.manifest))
        object.__setattr__(self, "provenance", _json_safe_mapping(self.provenance))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class _LoadedSource:
    spec: PublicationSourceSpec
    rows: tuple[dict[str, Any], ...]
    columns: tuple[str, ...]
    path: Path | None = None
    file_format: str | None = None
    sha256: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ResolvedOutputPath:
    name: str
    path: Path
    relative_path: str
    file_format: str


def build_publication_table_rows(
    rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    source_spec: PublicationSourceSpec | None = None,
    source_specs: Sequence[PublicationSourceSpec] = (),
    table_spec: PublicationTableSpec,
) -> dict[str, Any]:
    """Build display and machine-readable rows from already-computed inputs."""

    loaded_sources = _load_sources(rows=rows, source_spec=source_spec, source_specs=source_specs, strict=True)
    source_rows = tuple(row for source in loaded_sources for row in source.rows)
    display_rows, machine_rows, warnings, errors = _shape_rows(source_rows, table_spec)
    source_columns = _columns_for_rows(source_rows)
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "table_id": table_spec.table_id,
            "status": _status(errors),
            "display_rows": display_rows,
            "machine_rows": machine_rows,
            "source_row_count": len(source_rows),
            "source_columns": source_columns,
            "column_mappings": _column_mappings(table_spec),
            "warnings": warnings,
            "errors": errors,
        }
    )


def plan_publication_table_outputs(
    rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    source_spec: PublicationSourceSpec | None = None,
    source_specs: Sequence[PublicationSourceSpec] = (),
    table_spec: PublicationTableSpec,
    output_spec: PublicationOutputSpec,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> PublicationTablePlan:
    """Validate and preview publication table outputs without writing files."""

    plan = _build_plan(
        rows=rows,
        source_spec=source_spec,
        source_specs=source_specs,
        table_spec=table_spec,
        output_spec=output_spec,
        provenance=provenance or {},
        overwrite=overwrite,
        output_written=False,
    )
    return plan


def write_publication_table_outputs(
    rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    source_spec: PublicationSourceSpec | None = None,
    source_specs: Sequence[PublicationSourceSpec] = (),
    table_spec: PublicationTableSpec,
    output_spec: PublicationOutputSpec,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> PublicationTableWriteResult:
    """Write only configured publication table outputs and manifest files."""

    plan = _build_plan(
        rows=rows,
        source_spec=source_spec,
        source_specs=source_specs,
        table_spec=table_spec,
        output_spec=output_spec,
        provenance=provenance or {},
        overwrite=overwrite,
        output_written=False,
    )
    if plan.errors:
        raise ValueError("; ".join(plan.errors))

    targets = _resolved_output_targets(output_spec)
    _reject_existing_targets(targets, overwrite=overwrite)

    display_rows = tuple(dict(row) for row in plan.display_rows)
    machine_rows = tuple(dict(row) for row in plan.machine_rows)
    row_counts = _output_row_counts(display_rows, machine_rows, targets)
    texts = _output_texts(display_rows, machine_rows, targets, manifest=None)
    data_targets = {name: target for name, target in targets.items() if name != _MANIFEST}
    for name, target in data_targets.items():
        _write_text_atomic(target.path, texts[name])

    output_hashes = {name: _file_sha256(target.path) for name, target in data_targets.items()}
    source_records = tuple(row for row in plan.manifest["rows"] if row.get("role") == "source")
    written_manifest = build_publication_manifest(
        source_records=source_records,
        table_spec=table_spec,
        output_paths=plan.output_paths,
        output_row_counts=row_counts,
        output_hashes=output_hashes,
        warnings=plan.warnings,
        errors=(),
        provenance=plan.provenance,
        output_written=True,
    )
    manifest_dict = written_manifest.to_dict()
    if _MANIFEST in targets:
        manifest_text = _json_text(manifest_dict)
        _write_text_atomic(targets[_MANIFEST].path, manifest_text)
        output_hashes[_MANIFEST] = _file_sha256(targets[_MANIFEST].path)

    output_hashes = {name: _file_sha256(target.path) for name, target in targets.items()}
    return PublicationTableWriteResult(
        schema_version=SCHEMA_VERSION,
        table_id=table_spec.table_id,
        status="ok",
        will_write=True,
        output_written=True,
        output_paths=plan.output_paths,
        output_hashes=output_hashes,
        output_row_counts=row_counts,
        manifest=manifest_dict,
        provenance=plan.provenance,
        warnings=plan.warnings,
        errors=(),
    )


def build_publication_manifest(
    *,
    source_records: Sequence[Mapping[str, Any] | PublicationManifestRow],
    table_spec: PublicationTableSpec,
    output_paths: Mapping[str, str],
    output_row_counts: Mapping[str, int],
    output_hashes: Mapping[str, str] | None = None,
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
    output_written: bool = False,
) -> PublicationManifest:
    """Build a JSON-safe publication manifest."""

    source_rows = tuple(_manifest_row_from_record(record) for record in source_records)
    output_rows = tuple(
        PublicationManifestRow(
            role="output",
            name=name,
            path=path,
            relative_path=path,
            file_format=_output_format_from_name(name),
            row_count=output_row_counts.get(name),
            columns=_output_columns_from_name(name, table_spec),
            sha256=(output_hashes or {}).get(name),
            exists=Path(path).exists() if path else None,
            written=output_written,
        )
        for name, path in output_paths.items()
    )
    source_paths = tuple(row.path for row in source_rows if row.role == "source" and row.path is not None)
    source_hashes = {row.name: row.sha256 for row in source_rows if row.role == "source" and row.sha256 is not None}
    source_row_counts = {
        row.name: row.row_count for row in source_rows if row.role == "source" and row.row_count is not None
    }
    return PublicationManifest(
        schema_version=SCHEMA_VERSION,
        table_id=table_spec.table_id,
        source_paths=source_paths,
        source_hashes=source_hashes,
        source_row_counts=source_row_counts,
        output_paths=output_paths,
        output_hashes=output_hashes or {},
        output_row_counts=output_row_counts,
        table_spec_summary=_table_spec_summary(table_spec),
        column_mappings=_column_mappings(table_spec),
        filters=_filter_settings(table_spec),
        sort_settings=table_spec.sort_by,
        format_settings=table_spec.format.to_dict(),
        warnings=warnings,
        errors=errors,
        provenance=_manifest_provenance(provenance or {}),
        rows=(*source_rows, *output_rows),
    )


def _build_plan(
    *,
    rows: Iterable[Mapping[str, Any]] | None,
    source_spec: PublicationSourceSpec | None,
    source_specs: Sequence[PublicationSourceSpec],
    table_spec: PublicationTableSpec,
    output_spec: PublicationOutputSpec,
    provenance: Mapping[str, Any],
    overwrite: bool,
    output_written: bool,
) -> PublicationTablePlan:
    warnings: list[str] = []
    errors: list[str] = []
    loaded_sources = _load_sources(rows=rows, source_spec=source_spec, source_specs=source_specs, strict=False)
    for source in loaded_sources:
        warnings.extend(source.warnings)
        errors.extend(source.errors)

    targets: dict[str, _ResolvedOutputPath] = {}
    try:
        targets = _resolved_output_targets(output_spec)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))

    source_rows = tuple(row for source in loaded_sources for row in source.rows)
    display_rows: tuple[dict[str, Any], ...] = ()
    machine_rows: tuple[dict[str, Any], ...] = ()
    if not errors:
        display_rows, machine_rows, shape_warnings, shape_errors = _shape_rows(source_rows, table_spec)
        warnings.extend(shape_warnings)
        errors.extend(shape_errors)

    output_paths = {name: str(target.path) for name, target in targets.items()}
    output_row_counts = _output_row_counts(display_rows, machine_rows, targets)
    source_records = tuple(_source_manifest_row(source).to_dict() for source in loaded_sources)
    manifest = build_publication_manifest(
        source_records=source_records,
        table_spec=table_spec,
        output_paths=output_paths,
        output_row_counts=output_row_counts,
        output_hashes={},
        warnings=warnings,
        errors=errors,
        provenance=provenance,
        output_written=output_written,
    )
    return PublicationTablePlan(
        schema_version=SCHEMA_VERSION,
        table_id=table_spec.table_id,
        status=_status(errors),
        will_write=False,
        output_written=False,
        display_rows=display_rows,
        machine_rows=machine_rows,
        output_paths=output_paths,
        column_mappings=_column_mappings(table_spec),
        manifest=manifest.to_dict(),
        provenance=_manifest_provenance(provenance),
        warnings=warnings,
        errors=errors,
    )


def _load_sources(
    *,
    rows: Iterable[Mapping[str, Any]] | None,
    source_spec: PublicationSourceSpec | None,
    source_specs: Sequence[PublicationSourceSpec],
    strict: bool,
) -> tuple[_LoadedSource, ...]:
    specs = tuple(source_specs)
    if source_spec is not None:
        specs = (*specs, source_spec)
    loaded: list[_LoadedSource] = []
    if rows is not None:
        memory_spec = PublicationSourceSpec(source_id="in_memory", required=True)
        memory_rows = _normalize_rows(rows, field_name="rows")
        loaded.append(
            _LoadedSource(
                spec=memory_spec,
                rows=memory_rows,
                columns=_columns_for_rows(memory_rows),
                file_format="memory",
            )
        )
    for spec in specs:
        try:
            loaded.append(_load_source_spec(spec))
        except (OSError, TypeError, ValueError) as exc:
            if strict:
                raise
            errors = (str(exc),) if spec.required else ()
            warnings = () if spec.required else (str(exc),)
            loaded.append(_LoadedSource(spec=spec, rows=(), columns=(), warnings=warnings, errors=errors))
    if not loaded:
        message = "At least one in-memory row iterable or source spec is required."
        if strict:
            raise ValueError(message)
        loaded.append(
            _LoadedSource(
                spec=PublicationSourceSpec(source_id="missing_input"),
                rows=(),
                columns=(),
                errors=(message,),
            )
        )
    return tuple(loaded)


def _load_source_spec(spec: PublicationSourceSpec) -> _LoadedSource:
    if spec.path is None:
        if spec.required:
            raise ValueError(f"Source {spec.source_id!r} is missing a path.")
        return _LoadedSource(spec=spec, rows=(), columns=(), warnings=(f"Optional source {spec.source_id!r} is missing a path.",))
    path = Path(spec.path).expanduser()
    if not path.exists():
        if spec.required:
            raise FileNotFoundError(f"Source path does not exist for {spec.source_id!r}: {path}")
        return _LoadedSource(
            spec=spec,
            path=path,
            rows=(),
            columns=(),
            warnings=(f"Optional source path does not exist for {spec.source_id!r}: {path}",),
        )
    if path.is_dir():
        raise IsADirectoryError(f"Source path for {spec.source_id!r} is a directory: {path}")
    file_format = spec.format or _infer_source_format(path)
    if file_format in {"tsv", "csv"}:
        rows = _read_delimited_rows(path, delimiter="\t" if file_format == "tsv" else ",")
    elif file_format == "json":
        rows = _read_json_rows(path, rows_key=spec.json_rows_key)
    else:
        raise ValueError(f"Unsupported source format for {path!s}: {file_format}")
    return _LoadedSource(
        spec=spec,
        rows=rows,
        columns=_columns_for_rows(rows),
        path=path.resolve(strict=False),
        file_format=file_format,
        sha256=_file_sha256(path),
    )


def _read_delimited_rows(path: Path, *, delimiter: str) -> tuple[dict[str, Any], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Source table {path!s} is missing a header row.")
        return tuple({str(key): value for key, value in row.items()} for row in reader)


def _read_json_rows(path: Path, *, rows_key: str | None) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if rows_key is not None:
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSON source {path!s} must be an object when json_rows_key is configured.")
        if rows_key not in payload:
            raise ValueError(f"JSON source {path!s} does not contain row key {rows_key!r}.")
        payload = payload[rows_key]
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"JSON source {path!s} must contain a list of row mappings.")
    return _normalize_rows(payload, field_name=f"JSON source {path!s}")


def _normalize_rows(rows: Iterable[Mapping[str, Any]], *, field_name: str) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        mapping = _as_mapping(row, field_name=f"{field_name}[{index}]")
        normalized.append({str(key): _json_safe(value) for key, value in mapping.items()})
    return tuple(normalized)


def _shape_rows(
    rows: Sequence[Mapping[str, Any]],
    table_spec: PublicationTableSpec,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    errors: list[str] = []
    filtered_rows = [row for row in rows if _row_matches_table(row, table_spec)]
    sorted_rows = _sort_rows(filtered_rows, table_spec.sort_by)
    display_rows: list[dict[str, Any]] = []
    machine_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(sorted_rows):
        display_row: dict[str, Any] = {}
        machine_row: dict[str, Any] = {}
        for column in table_spec.columns:
            try:
                display_value, machine_value = _column_values(row, column, table_spec.format)
            except ValueError as exc:
                message = f"Row {row_index} column {column.output_name!r}: {exc}"
                if column.required:
                    errors.append(message)
                else:
                    warnings.append(message)
                display_value = _missing_display_value(column, table_spec.format)
                machine_value = None
            display_row[column.output_name] = display_value
            machine_row[column.output_name] = machine_value
        for metadata_column in table_spec.metadata_columns:
            display_row[metadata_column] = _display_scalar(row.get(metadata_column), table_spec.format.missing_value)
            machine_row[metadata_column] = _json_safe(row.get(metadata_column))
        display_rows.append(display_row)
        machine_rows.append(machine_row)
    return tuple(display_rows), tuple(machine_rows), tuple(warnings), tuple(errors)


def _row_matches_table(row: Mapping[str, Any], table_spec: PublicationTableSpec) -> bool:
    if table_spec.status_values and str(row.get("status")) not in set(table_spec.status_values):
        return False
    return _matches_filters(row, table_spec.filters)


def _matches_filters(row: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = row.get(str(key))
        if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
            if not any(_value_matches(actual, item) for item in expected):
                return False
        elif not _value_matches(actual, expected):
            return False
    return True


def _value_matches(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    actual_number = _maybe_number(actual)
    expected_number = _maybe_number(expected)
    if actual_number is not None and expected_number is not None:
        return actual_number == expected_number
    if actual is None or expected is None:
        return False
    return str(actual) == str(expected)


def _sort_rows(rows: Sequence[Mapping[str, Any]], sort_by: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    sorted_rows = list(rows)
    for sort_key in reversed(sort_by):
        descending = sort_key.startswith("-")
        column = sort_key[1:] if descending else sort_key
        sorted_rows.sort(key=lambda row: _sort_value(row.get(column)), reverse=descending)
    return tuple(sorted_rows)


def _sort_value(value: Any) -> tuple[int, int, Any]:
    if value is None or value == "":
        return (1, 0, "")
    number = _maybe_number(value)
    if number is not None:
        return (0, 0, number)
    return (0, 1, str(value))


def _column_values(
    row: Mapping[str, Any],
    column: PublicationColumnSpec,
    table_format: PublicationFormatSpec,
) -> tuple[str, Any]:
    if column.value_filter and not _matches_filters(row, column.value_filter):
        return _missing_display_value(column, table_format), None
    if column.column_type == "constant":
        value = _json_safe(column.constant)
        return _display_scalar(value, _missing_display_value(column, table_format)), value
    if column.column_type == "confidence_interval":
        estimate_source = _required_source(column.source, column.output_name)
        low_source = _required_source(column.ci_low_source, column.output_name)
        high_source = _required_source(column.ci_high_source, column.output_name)
        estimate = row.get(estimate_source)
        low = row.get(low_source)
        high = row.get(high_source)
        if column.required and (_is_missing(estimate) or _is_missing(low) or _is_missing(high)):
            raise ValueError(
                f"Missing required confidence interval source value among "
                f"{estimate_source!r}, {low_source!r}, and {high_source!r}."
            )
        ci_format = column.confidence_interval_format or table_format.confidence_interval
        display = _format_confidence_interval(
            estimate,
            low,
            high,
            ci_format,
            missing_value=_missing_display_value(column, table_format),
        )
        return display, {
            estimate_source: _numeric_or_none(estimate),
            low_source: _numeric_or_none(low),
            high_source: _numeric_or_none(high),
        }
    if column.column_type == "template":
        source_values = {source: row.get(source) for source in column.sources}
        if column.required and any(_is_missing(value) for value in source_values.values()):
            raise ValueError(f"Missing required template source value for {column.output_name!r}.")
        template = column.template or " ".join("{" + source + "}" for source in column.sources)
        display = _format_template(
            template,
            source_values,
            table_format,
            missing_value=_missing_display_value(column, table_format),
        )
        return display, {source: _json_safe(value) for source, value in source_values.items()}

    source = _required_source(column.source, column.output_name)
    value = row.get(source)
    if column.required and _is_missing(value):
        raise ValueError(f"Missing required source value {source!r}.")
    if column.column_type == "numeric":
        numeric_format = column.numeric_format or table_format.numeric
        return (
            _format_number(value, numeric_format, missing_value=_missing_display_value(column, table_format)),
            _numeric_or_none(value),
        )
    if column.column_type == "p_value":
        p_format = column.p_value_format or table_format.p_value
        return (
            _format_p_like(value, p_format, missing_value=_missing_display_value(column, table_format)),
            _numeric_or_none(value),
        )
    if column.column_type == "q_value":
        q_format = column.q_value_format or table_format.q_value
        return (
            _format_p_like(value, q_format, missing_value=_missing_display_value(column, table_format)),
            _numeric_or_none(value),
        )
    return _display_scalar(value, _missing_display_value(column, table_format)), _json_safe(value)


def _format_confidence_interval(
    estimate: Any,
    low: Any,
    high: Any,
    ci_format: ConfidenceIntervalFormatSpec,
    *,
    missing_value: str,
) -> str:
    ci_missing = ci_format.missing_value if ci_format.missing_value is not None else missing_value
    if _is_missing(estimate) or _is_missing(low) or _is_missing(high):
        return ci_missing
    values = {
        "estimate": _format_number(estimate, ci_format.estimate, missing_value=ci_missing),
        "low": _format_number(low, ci_format.bounds, missing_value=ci_missing),
        "high": _format_number(high, ci_format.bounds, missing_value=ci_missing),
    }
    return ci_format.template.format(**values)


def _format_template(
    template: str,
    values: Mapping[str, Any],
    table_format: PublicationFormatSpec,
    *,
    missing_value: str,
) -> str:
    formatted = {name: _display_scalar(value, missing_value or table_format.missing_value) for name, value in values.items()}
    return template.format(**formatted)


def _format_number(value: Any, spec: NumericFormatSpec, *, missing_value: str) -> str:
    if _is_missing(value):
        return spec.missing_value if spec.missing_value is not None else missing_value
    number = _require_number(value)
    rendered = f"{number:.{spec.precision}f}"
    if spec.trim_trailing_zeros and "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if rendered in {"", "-0"} or set(rendered) <= {"-", "0", "."}:
        return "0"
    return rendered


def _format_p_like(value: Any, spec: PValueFormatSpec, *, missing_value: str) -> str:
    if _is_missing(value):
        return spec.missing_value if spec.missing_value is not None else missing_value
    number = _require_number(value)
    if spec.threshold is not None and 0 <= number < spec.threshold:
        threshold = f"{spec.threshold:.{spec.precision}f}"
        return f"<{_without_leading_zero(threshold, enabled=not spec.include_leading_zero)}"
    rendered = f"{number:.{spec.precision}f}"
    return _without_leading_zero(rendered, enabled=not spec.include_leading_zero)


def _display_scalar(value: Any, missing_value: str) -> str:
    safe_value = _json_safe(value)
    if safe_value is None or safe_value == "":
        return missing_value
    if isinstance(safe_value, bool):
        return "true" if safe_value else "false"
    if isinstance(safe_value, (int, float)):
        return repr(safe_value)
    if isinstance(safe_value, str):
        return _without_table_control_chars(safe_value)
    return _without_table_control_chars(json.dumps(safe_value, allow_nan=False, separators=(",", ":"), sort_keys=True))


def _missing_display_value(column: PublicationColumnSpec, table_format: PublicationFormatSpec) -> str:
    return table_format.missing_value if column.missing_value is None else column.missing_value


def _without_leading_zero(value: str, *, enabled: bool) -> str:
    if not enabled:
        return value
    if value.startswith("0."):
        return value[1:]
    if value.startswith("-0."):
        return "-" + value[2:]
    return value


def _required_source(source: str | None, output_name: str) -> str:
    if source is None:
        raise ValueError(f"Column {output_name!r} requires a source field.")
    return source


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _require_number(value: Any) -> float:
    number = _maybe_number(value)
    if number is None:
        raise ValueError(f"Value {value!r} is not numeric.")
    if not math.isfinite(number):
        raise ValueError("Publication table values cannot contain non-finite floats.")
    return number


def _numeric_or_none(value: Any) -> int | float | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"Value {value!r} is not numeric.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Publication table values cannot contain non-finite floats.")
        return value
    number = _require_number(value)
    if isinstance(value, str) and _looks_like_integer(value):
        return int(number)
    return number


def _maybe_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _looks_like_integer(value: str) -> bool:
    stripped = value.strip()
    if stripped.startswith(("-", "+")):
        stripped = stripped[1:]
    return stripped.isdecimal()


def _source_manifest_row(source: _LoadedSource) -> PublicationManifestRow:
    path = str(source.path) if source.path is not None else None
    return PublicationManifestRow(
        role="source",
        name=source.spec.source_id,
        path=path,
        file_format=source.file_format,
        row_count=len(source.rows),
        columns=source.columns,
        sha256=source.sha256,
        exists=source.path.exists() if source.path is not None else None,
        written=False,
        warnings=source.warnings,
        errors=source.errors,
    )


def _manifest_row_from_record(record: Mapping[str, Any] | PublicationManifestRow) -> PublicationManifestRow:
    if isinstance(record, PublicationManifestRow):
        return record
    return PublicationManifestRow(
        role=str(record.get("role", "source")),
        name=str(record.get("name", "source")),
        path=record.get("path"),
        relative_path=record.get("relative_path"),
        file_format=record.get("file_format"),
        row_count=record.get("row_count"),
        columns=tuple(record.get("columns", ())),
        sha256=record.get("sha256"),
        exists=record.get("exists"),
        written=bool(record.get("written", False)),
        warnings=tuple(record.get("warnings", ())),
        errors=tuple(record.get("errors", ())),
    )


def _resolved_output_targets(output_spec: PublicationOutputSpec) -> dict[str, _ResolvedOutputPath]:
    root = _resolved_output_root(output_spec.output_root)
    configured = _configured_output_paths(output_spec)
    targets = {name: _resolved_output_path(root, name, path) for name, path in configured.items()}
    _reject_duplicate_targets(targets)
    return targets


def _configured_output_paths(output_spec: PublicationOutputSpec) -> dict[str, str | Path]:
    configured: dict[str, str | Path] = {}
    for name in _OUTPUT_NAMES:
        value = getattr(output_spec, f"{name}_path", None) if name != _MANIFEST else output_spec.manifest_path
        if value is not None:
            configured[name] = value
    if not configured:
        raise ValueError("PublicationOutputSpec must configure at least one output path.")
    return configured


def _resolved_output_root(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser()
    if _has_parent_traversal(root):
        raise ValueError("output_root must not contain parent traversal.")
    return root.resolve(strict=False)


def _resolved_output_path(root: Path, name: str, output_path: str | Path) -> _ResolvedOutputPath:
    raw_path = Path(output_path).expanduser()
    if _has_parent_traversal(raw_path):
        raise ValueError(f"Output path {output_path!s} contains parent traversal.")
    target = raw_path if raw_path.is_absolute() else root / raw_path
    resolved = target.resolve(strict=False)
    if resolved == root:
        raise ValueError("Output path must name a file inside output_root.")
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Output path {output_path!s} resolves outside output_root.") from exc
    return _ResolvedOutputPath(
        name=name,
        path=resolved,
        relative_path=relative.as_posix(),
        file_format=_output_format_from_name(name),
    )


def _has_parent_traversal(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def _reject_duplicate_targets(targets: Mapping[str, _ResolvedOutputPath]) -> None:
    seen: dict[Path, str] = {}
    for name, target in targets.items():
        previous = seen.get(target.path)
        if previous is not None:
            raise ValueError(f"Output paths for {previous!r} and {name!r} must be distinct.")
        seen[target.path] = name


def _reject_existing_targets(targets: Mapping[str, _ResolvedOutputPath], *, overwrite: bool) -> None:
    for name, target in targets.items():
        if target.path.is_dir():
            raise IsADirectoryError(f"Output path for {name!r} is an existing directory: {target.path}")
        if target.path.exists() and not overwrite:
            raise FileExistsError(f"Output path for {name!r} already exists: {target.path}")


def _output_texts(
    display_rows: Sequence[Mapping[str, Any]],
    machine_rows: Sequence[Mapping[str, Any]],
    targets: Mapping[str, _ResolvedOutputPath],
    *,
    manifest: Mapping[str, Any] | None,
) -> dict[str, str]:
    texts: dict[str, str] = {}
    for name in targets:
        if name == _DISPLAY_TSV:
            texts[name] = _delimited_text(display_rows, delimiter="\t")
        elif name == _DISPLAY_CSV:
            texts[name] = _delimited_text(display_rows, delimiter=",")
        elif name == _DISPLAY_MARKDOWN:
            texts[name] = _markdown_text(display_rows)
        elif name == _MACHINE_TSV:
            texts[name] = _delimited_text(machine_rows, delimiter="\t")
        elif name == _MACHINE_CSV:
            texts[name] = _delimited_text(machine_rows, delimiter=",")
        elif name == _MACHINE_JSON:
            texts[name] = _json_text({"rows": list(machine_rows)})
        elif name == _MANIFEST and manifest is not None:
            texts[name] = _json_text(manifest)
    return texts


def _output_row_counts(
    display_rows: Sequence[Mapping[str, Any]],
    machine_rows: Sequence[Mapping[str, Any]],
    targets: Mapping[str, _ResolvedOutputPath],
) -> dict[str, int]:
    row_counts: dict[str, int] = {}
    for name in targets:
        if name.startswith("display_"):
            row_counts[name] = len(display_rows)
        elif name.startswith("machine_"):
            row_counts[name] = len(machine_rows)
        elif name == _MANIFEST:
            row_counts[name] = 1
    return row_counts


def _delimited_text(rows: Sequence[Mapping[str, Any]], *, delimiter: str) -> str:
    columns = _columns_for_rows(rows)
    output = io.StringIO()
    writer = csv.writer(output, delimiter=delimiter, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_serialize_cell(row.get(column)) for column in columns])
    return output.getvalue()


def _markdown_text(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = _columns_for_rows(rows)
    if not columns:
        return "\n"
    lines = [
        "| " + " | ".join(_markdown_cell(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(_serialize_cell(row.get(column))) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True) + "\n"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _serialize_cell(value: Any) -> str:
    safe_value = _json_safe(value)
    if safe_value is None:
        return ""
    if isinstance(safe_value, bool):
        return "true" if safe_value else "false"
    if isinstance(safe_value, (int, float)):
        return repr(safe_value)
    if isinstance(safe_value, str):
        return _without_table_control_chars(safe_value)
    return _without_table_control_chars(json.dumps(safe_value, allow_nan=False, separators=(",", ":"), sort_keys=True))


def _markdown_cell(value: Any) -> str:
    return _serialize_cell(value).replace("|", "\\|")


def _without_table_control_chars(value: str) -> str:
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _columns_for_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    columns: list[str] = []
    for row in rows:
        for column in row:
            text_column = str(column)
            if text_column not in columns:
                columns.append(text_column)
    return tuple(columns)


def _column_mappings(table_spec: PublicationTableSpec) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "output_name": column.output_name,
            "source": column.source,
            "sources": list(column.sources),
            "column_type": column.column_type,
            "value_filter": dict(column.value_filter),
            "ci_low_source": column.ci_low_source,
            "ci_high_source": column.ci_high_source,
            "required": column.required,
        }
        for column in table_spec.columns
    )


def _table_spec_summary(table_spec: PublicationTableSpec) -> dict[str, Any]:
    return {
        "table_id": table_spec.table_id,
        "display_columns": [column.output_name for column in table_spec.columns],
        "metadata_columns": list(table_spec.metadata_columns),
        "column_count": len(table_spec.columns) + len(table_spec.metadata_columns),
    }


def _filter_settings(table_spec: PublicationTableSpec) -> dict[str, Any]:
    return {
        "filters": dict(table_spec.filters),
        "status_values": list(table_spec.status_values),
    }


def _manifest_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    combined = {
        "writer_module": __name__,
        "schema_version": SCHEMA_VERSION,
    }
    combined.update(_json_safe_mapping(provenance))
    return combined


def _output_columns_from_name(name: str, table_spec: PublicationTableSpec) -> tuple[str, ...]:
    if name.startswith("display_") or name.startswith("machine_"):
        return tuple(column.output_name for column in table_spec.columns) + tuple(table_spec.metadata_columns)
    return ()


def _output_format_from_name(name: str) -> str:
    if name.endswith("_tsv"):
        return "tsv"
    if name.endswith("_csv"):
        return "csv"
    if name.endswith("_json") or name == _MANIFEST:
        return "json"
    if name.endswith("_markdown"):
        return "markdown"
    return name


def _infer_source_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".tsv" or suffix == ".txt":
        return "tsv"
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    raise ValueError(f"Unsupported source file suffix for {path!s}. Use .tsv, .csv, or .json.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status(errors: Sequence[str]) -> str:
    return "error" if errors else "ok"


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif is_dataclass(value):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or expose to_dict().")
    return value


def _json_safe_dataclass(instance: object) -> dict[str, Any]:
    if not is_dataclass(instance):
        raise TypeError("_json_safe_dataclass requires a dataclass instance.")
    return {field_.name: _json_safe(getattr(instance, field_.name)) for field_ in fields(instance)}


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(item) for key, item in value.items()}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe_dataclass(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Publication table outputs cannot contain non-finite floats.")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _non_empty_text(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


__all__ = [
    "ConfidenceIntervalFormatSpec",
    "NumericFormatSpec",
    "PValueFormatSpec",
    "PublicationColumnSpec",
    "PublicationFormatSpec",
    "PublicationManifest",
    "PublicationManifestRow",
    "PublicationOutputSpec",
    "PublicationSourceSpec",
    "PublicationTablePlan",
    "PublicationTableSpec",
    "PublicationTableWriteResult",
    "SCHEMA_VERSION",
    "build_publication_manifest",
    "build_publication_table_rows",
    "plan_publication_table_outputs",
    "write_publication_table_outputs",
]
