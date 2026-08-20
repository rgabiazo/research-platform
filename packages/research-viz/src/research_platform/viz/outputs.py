"""Plan-first visualization and report output writers.

This module is intentionally dependency-free. It consumes already-computed
rows from memory or simple source files, previews report/figure artifacts, and
writes only explicitly configured outputs when ``render_visualization_outputs``
is called.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from string import Formatter
import tempfile
from typing import Any

from research_platform.viz.plots import (
    PointIntervalPlotSpec,
    VisualQcRow,
    build_visual_layout_qc_rows,
    render_point_interval_svg,
)
from research_platform.viz.reports import (
    ReportSpec,
    TableSectionSpec,
    build_report_document,
)


SCHEMA_VERSION = "research_platform.viz.visualization_outputs.v1"

_REPORT_MARKDOWN = "report_markdown"
_REPORT_HTML = "report_html"
_REPORT_TEXT = "report_text"
_FIGURE_SVG = "figure_svg"
_FIGURE_PNG = "figure_png"
_FIGURE_PDF = "figure_pdf"
_MANIFEST_JSON = "manifest_json"

_OUTPUT_NAMES = (
    _REPORT_MARKDOWN,
    _REPORT_HTML,
    _REPORT_TEXT,
    _FIGURE_SVG,
    _FIGURE_PNG,
    _FIGURE_PDF,
    _MANIFEST_JSON,
)

_REPORT_FORMATS = {"markdown", "html", "text"}
_FIGURE_FORMATS = {"svg", "png", "pdf"}
_BUILTIN_FIGURE_FORMATS = {"svg"}
_OPTIONAL_UNAVAILABLE_FIGURE_FORMATS = {"png", "pdf"}
_MANIFEST_FORMATS = {"json"}


@dataclass(frozen=True)
class VisualizationSourceSpec:
    """Rows for visualization/report rendering.

    File-backed sources support TSV, CSV, and JSON. JSON sources may be a list
    of row mappings or an object with rows under ``json_rows_key``.
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
                raise ValueError("VisualizationSourceSpec.format must be one of: tsv, csv, json.")
            object.__setattr__(self, "format", normalized)
        if self.json_rows_key is not None:
            object.__setattr__(self, "json_rows_key", _non_empty_text(self.json_rows_key, field_name="json_rows_key"))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class VisualizationFormatSpec:
    """Requested report, figure, and manifest formats."""

    report_formats: Sequence[str] = ()
    figure_formats: Sequence[str] = ()
    manifest_formats: Sequence[str] = ("json",)
    optional_renderer_policy: str = "error"

    def __post_init__(self) -> None:
        object.__setattr__(self, "report_formats", tuple(str(value).lower() for value in self.report_formats))
        object.__setattr__(self, "figure_formats", tuple(str(value).lower() for value in self.figure_formats))
        object.__setattr__(self, "manifest_formats", tuple(str(value).lower() for value in self.manifest_formats))
        policy = str(self.optional_renderer_policy or "error").lower()
        if policy not in {"warning", "error"}:
            raise ValueError("optional_renderer_policy must be one of: warning, error.")
        object.__setattr__(self, "optional_renderer_policy", policy)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class VisualizationOutputSpec:
    """Configured report, figure, and manifest output paths."""

    output_root: str | Path
    report_markdown_path: str | Path | None = None
    report_html_path: str | Path | None = None
    report_text_path: str | Path | None = None
    figure_svg_path: str | Path | None = None
    figure_png_path: str | Path | None = None
    figure_pdf_path: str | Path | None = None
    manifest_path: str | Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class VisualizationWarningRow:
    """One structured visualization warning."""

    code: str
    message: str
    scope: str = "plan"
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        object.__setattr__(self, "scope", str(self.scope or "plan"))
        if self.artifact_id is not None:
            object.__setattr__(self, "artifact_id", str(self.artifact_id))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class VisualizationErrorRow:
    """One structured visualization error."""

    code: str
    message: str
    scope: str = "plan"
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        object.__setattr__(self, "scope", str(self.scope or "plan"))
        if self.artifact_id is not None:
            object.__setattr__(self, "artifact_id", str(self.artifact_id))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class VisualizationProvenanceRow:
    """One JSON-safe provenance key/value row."""

    key: str
    value: Any
    source: str = "input"

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty_text(self.key, field_name="key"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "source", str(self.source or "input"))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RenderedArtifactRow:
    """One source or planned/rendered output artifact record."""

    role: str
    name: str
    path: str | None = None
    relative_path: str | None = None
    file_format: str | None = None
    renderer: str | None = None
    artifact_id: str | None = None
    row_count: int | None = None
    columns: Sequence[str] = ()
    sha256: str | None = None
    exists: bool | None = None
    planned: bool = True
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
class VisualizationManifest:
    """JSON-safe manifest for visualization and report outputs."""

    schema_version: str
    source_paths: Sequence[str] = ()
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    source_row_counts: Mapping[str, int] = field(default_factory=dict)
    output_paths: Mapping[str, str] = field(default_factory=dict)
    output_hashes: Mapping[str, str] = field(default_factory=dict)
    output_row_counts: Mapping[str, int] = field(default_factory=dict)
    renderer_settings: Mapping[str, Any] = field(default_factory=dict)
    text_settings: Mapping[str, Any] = field(default_factory=dict)
    layout_settings: Mapping[str, Any] = field(default_factory=dict)
    visual_qc_rows: Sequence[Mapping[str, Any]] = ()
    warning_rows: Sequence[Mapping[str, Any]] = ()
    error_rows: Sequence[Mapping[str, Any]] = ()
    provenance_rows: Sequence[Mapping[str, Any]] = ()
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    rows: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_paths", tuple(str(path) for path in self.source_paths))
        object.__setattr__(self, "source_hashes", _json_safe_mapping(self.source_hashes))
        object.__setattr__(self, "source_row_counts", _json_safe_mapping(self.source_row_counts))
        object.__setattr__(self, "output_paths", _json_safe_mapping(self.output_paths))
        object.__setattr__(self, "output_hashes", _json_safe_mapping(self.output_hashes))
        object.__setattr__(self, "output_row_counts", _json_safe_mapping(self.output_row_counts))
        object.__setattr__(self, "renderer_settings", _json_safe_mapping(self.renderer_settings))
        object.__setattr__(self, "text_settings", _json_safe_mapping(self.text_settings))
        object.__setattr__(self, "layout_settings", _json_safe_mapping(self.layout_settings))
        object.__setattr__(self, "visual_qc_rows", tuple(_json_safe_mapping(row) for row in self.visual_qc_rows))
        object.__setattr__(self, "warning_rows", tuple(_json_safe_mapping(row) for row in self.warning_rows))
        object.__setattr__(self, "error_rows", tuple(_json_safe_mapping(row) for row in self.error_rows))
        object.__setattr__(self, "provenance_rows", tuple(_json_safe_mapping(row) for row in self.provenance_rows))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "provenance", _json_safe_mapping(self.provenance))
        object.__setattr__(self, "rows", tuple(_json_safe_mapping(row) for row in self.rows))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class VisualizationPlan:
    """Dry-run preview for visualization and report outputs."""

    schema_version: str
    status: str
    will_write: bool
    output_written: bool
    source_row_count: int
    source_columns: Sequence[str]
    source_rows_preview: Sequence[Mapping[str, Any]]
    output_paths: Mapping[str, str]
    planned_artifacts: Sequence[RenderedArtifactRow]
    report_previews: Mapping[str, str] = field(default_factory=dict)
    figure_previews: Mapping[str, str] = field(default_factory=dict)
    visual_qc_rows: Sequence[VisualQcRow] = ()
    warning_rows: Sequence[VisualizationWarningRow] = ()
    error_rows: Sequence[VisualizationErrorRow] = ()
    provenance_rows: Sequence[VisualizationProvenanceRow] = ()
    manifest: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_columns", tuple(str(column) for column in self.source_columns))
        object.__setattr__(self, "source_rows_preview", tuple(_json_safe_mapping(row) for row in self.source_rows_preview))
        object.__setattr__(self, "output_paths", _json_safe_mapping(self.output_paths))
        object.__setattr__(self, "planned_artifacts", tuple(self.planned_artifacts))
        object.__setattr__(self, "report_previews", _json_safe_mapping(self.report_previews))
        object.__setattr__(self, "figure_previews", _json_safe_mapping(self.figure_previews))
        object.__setattr__(self, "visual_qc_rows", tuple(self.visual_qc_rows))
        object.__setattr__(self, "warning_rows", tuple(self.warning_rows))
        object.__setattr__(self, "error_rows", tuple(self.error_rows))
        object.__setattr__(self, "provenance_rows", tuple(self.provenance_rows))
        object.__setattr__(self, "manifest", _json_safe_mapping(self.manifest))
        object.__setattr__(self, "provenance", _json_safe_mapping(self.provenance))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class VisualizationWriteResult:
    """Write result for configured visualization/report outputs."""

    schema_version: str
    status: str
    output_written: bool
    output_paths: Mapping[str, str]
    output_hashes: Mapping[str, str]
    output_row_counts: Mapping[str, int]
    artifacts: Sequence[RenderedArtifactRow]
    manifest: Mapping[str, Any]
    visual_qc_rows: Sequence[VisualQcRow] = ()
    warning_rows: Sequence[VisualizationWarningRow] = ()
    error_rows: Sequence[VisualizationErrorRow] = ()
    provenance_rows: Sequence[VisualizationProvenanceRow] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_paths", _json_safe_mapping(self.output_paths))
        object.__setattr__(self, "output_hashes", _json_safe_mapping(self.output_hashes))
        object.__setattr__(self, "output_row_counts", _json_safe_mapping(self.output_row_counts))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "manifest", _json_safe_mapping(self.manifest))
        object.__setattr__(self, "visual_qc_rows", tuple(self.visual_qc_rows))
        object.__setattr__(self, "warning_rows", tuple(self.warning_rows))
        object.__setattr__(self, "error_rows", tuple(self.error_rows))
        object.__setattr__(self, "provenance_rows", tuple(self.provenance_rows))
        object.__setattr__(self, "provenance", _json_safe_mapping(self.provenance))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class _LoadedSource:
    spec: VisualizationSourceSpec
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
    role: str
    renderer: str


def plan_visualization_outputs(
    rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    source_spec: VisualizationSourceSpec | None = None,
    source_specs: Sequence[VisualizationSourceSpec] = (),
    output_spec: VisualizationOutputSpec,
    format_spec: VisualizationFormatSpec | None = None,
    plot_spec: PointIntervalPlotSpec | None = None,
    plot_specs: Sequence[PointIntervalPlotSpec] = (),
    report_spec: ReportSpec | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> VisualizationPlan:
    """Validate and preview visualization outputs without writing files."""

    format_settings = format_spec or VisualizationFormatSpec()
    provenance_mapping = _manifest_provenance(provenance or {})
    metadata_mapping = _json_safe_mapping(metadata or {})
    warnings: list[str] = []
    errors: list[str] = []
    warning_rows: list[VisualizationWarningRow] = []
    error_rows: list[VisualizationErrorRow] = []
    provenance_rows = [
        VisualizationProvenanceRow(key="schema_version", value=SCHEMA_VERSION, source="research-viz"),
        VisualizationProvenanceRow(key="writer_module", value=__name__, source="research-viz"),
    ]
    provenance_rows.extend(_coerced_provenance_rows(provenance or {}))

    loaded_sources = _load_sources(rows=rows, source_spec=source_spec, source_specs=source_specs, strict=False)
    for source in loaded_sources:
        _extend_messages(
            warnings,
            errors,
            warning_rows,
            error_rows,
            source.warnings,
            source.errors,
            scope="source",
            artifact_id=source.spec.source_id,
        )

    targets: dict[str, _ResolvedOutputPath] = {}
    try:
        targets = _resolved_output_targets(output_spec)
    except (TypeError, ValueError) as exc:
        _append_error(errors, error_rows, "invalid_output_path", str(exc), scope="output")

    _validate_requested_formats(format_settings, targets, warnings, errors, warning_rows, error_rows)

    all_plot_specs = _combined_plot_specs(plot_spec=plot_spec, plot_specs=plot_specs)
    source_rows = tuple(row for source in loaded_sources for row in source.rows)
    source_columns = _columns_for_rows(source_rows)
    if any(target.role == "figure" for target in targets.values()) and not all_plot_specs:
        _append_error(errors, error_rows, "missing_plot_spec", "Figure output paths require a plot_spec.", scope="figure")

    active_report_spec = report_spec or (ReportSpec() if any(target.role == "report" for target in targets.values()) else None)
    _validate_required_columns(source_columns, all_plot_specs, active_report_spec, errors, error_rows)
    _validate_report_templates(active_report_spec, metadata_mapping, errors, error_rows)

    visual_qc_rows: list[VisualQcRow] = []
    for spec in all_plot_specs:
        try:
            visual_qc_rows.extend(build_visual_layout_qc_rows(source_rows, spec, metadata=metadata_mapping))
        except (TypeError, ValueError) as exc:
            _append_error(errors, error_rows, "visual_qc_failed", str(exc), scope="visual_qc", artifact_id=spec.plot_id)
    for qc_row in visual_qc_rows:
        if qc_row.severity == "error":
            _append_error(errors, error_rows, qc_row.check_id, qc_row.message, scope=qc_row.scope, artifact_id=qc_row.artifact_id)
        elif qc_row.severity == "warning":
            _append_warning(
                warnings,
                warning_rows,
                qc_row.check_id,
                qc_row.message,
                scope=qc_row.scope,
                artifact_id=qc_row.artifact_id,
            )

    source_records = tuple(_source_artifact_row(source).to_dict() for source in loaded_sources)
    artifact_rows = list(
        _artifact_row_for_target(target, source_rows, all_plot_specs, overwrite=overwrite) for target in targets.values()
    )
    for artifact in artifact_rows:
        _extend_messages(
            warnings,
            errors,
            warning_rows,
            error_rows,
            artifact.warnings,
            artifact.errors,
            scope=artifact.role,
            artifact_id=artifact.artifact_id or artifact.name,
        )

    report_previews: dict[str, str] = {}
    figure_previews: dict[str, str] = {}
    if not errors:
        report_previews, figure_previews = _build_previews(
            source_rows=source_rows,
            report_spec=active_report_spec,
            plot_specs=all_plot_specs,
            targets=targets,
            metadata=metadata_mapping,
            visual_qc_rows=visual_qc_rows,
            artifact_rows=artifact_rows,
            warnings=warnings,
            provenance=provenance_mapping,
            errors=errors,
            error_rows=error_rows,
        )

    output_paths = {name: str(target.path) for name, target in targets.items()}
    output_row_counts = _output_row_counts(source_rows, targets)
    manifest = build_manifest_from_visualization_outputs(
        source_records=source_records,
        artifact_rows=tuple(row.to_dict() for row in artifact_rows),
        output_hashes={},
        output_row_counts=output_row_counts,
        format_spec=format_settings,
        report_spec=active_report_spec,
        plot_specs=all_plot_specs,
        visual_qc_rows=tuple(row.to_dict() for row in visual_qc_rows),
        warning_rows=tuple(row.to_dict() for row in warning_rows),
        error_rows=tuple(row.to_dict() for row in error_rows),
        provenance_rows=tuple(row.to_dict() for row in provenance_rows),
        warnings=warnings,
        errors=errors,
        provenance=provenance_mapping,
        output_written=False,
    )
    return VisualizationPlan(
        schema_version=SCHEMA_VERSION,
        status=_status(errors),
        will_write=False,
        output_written=False,
        source_row_count=len(source_rows),
        source_columns=source_columns,
        source_rows_preview=source_rows[:20],
        output_paths=output_paths,
        planned_artifacts=tuple(artifact_rows),
        report_previews=report_previews,
        figure_previews=figure_previews,
        visual_qc_rows=tuple(visual_qc_rows),
        warning_rows=tuple(warning_rows),
        error_rows=tuple(error_rows),
        provenance_rows=tuple(provenance_rows),
        manifest=manifest.to_dict(),
        provenance=provenance_mapping,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def render_visualization_outputs(
    rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    source_spec: VisualizationSourceSpec | None = None,
    source_specs: Sequence[VisualizationSourceSpec] = (),
    output_spec: VisualizationOutputSpec,
    format_spec: VisualizationFormatSpec | None = None,
    plot_spec: PointIntervalPlotSpec | None = None,
    plot_specs: Sequence[PointIntervalPlotSpec] = (),
    report_spec: ReportSpec | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> VisualizationWriteResult:
    """Write only planned report, figure, and manifest artifacts."""

    plan = plan_visualization_outputs(
        rows=rows,
        source_spec=source_spec,
        source_specs=source_specs,
        output_spec=output_spec,
        format_spec=format_spec,
        plot_spec=plot_spec,
        plot_specs=plot_specs,
        report_spec=report_spec,
        metadata=metadata,
        provenance=provenance,
        overwrite=overwrite,
    )
    if plan.errors:
        raise ValueError("; ".join(plan.errors))

    targets = _resolved_output_targets(output_spec)
    _reject_existing_targets(targets, overwrite=overwrite)

    texts: dict[str, str] = {}
    for name, target in targets.items():
        if target.role == "report":
            texts[name] = plan.report_previews[name]
        elif target.role == "figure":
            texts[name] = plan.figure_previews[name]

    for name, text in texts.items():
        _write_text_atomic(targets[name].path, text)

    output_hashes = {name: _file_sha256(target.path) for name, target in targets.items() if name in texts}
    output_row_counts = _output_row_counts(tuple(plan.source_rows_preview), targets, source_row_count=plan.source_row_count)
    written_artifacts = tuple(
        replace(
            artifact,
            written=True,
            exists=True,
            sha256=output_hashes.get(artifact.name),
        )
        for artifact in plan.planned_artifacts
    )
    written_manifest = build_manifest_from_visualization_outputs(
        source_records=tuple(row for row in plan.manifest.get("rows", ()) if row.get("role") == "source"),
        artifact_rows=tuple(artifact.to_dict() for artifact in written_artifacts),
        output_hashes=output_hashes,
        output_row_counts=output_row_counts,
        format_spec=format_spec or VisualizationFormatSpec(),
        report_spec=report_spec or (ReportSpec() if any(target.role == "report" for target in targets.values()) else None),
        plot_specs=_combined_plot_specs(plot_spec=plot_spec, plot_specs=plot_specs),
        visual_qc_rows=tuple(row.to_dict() for row in plan.visual_qc_rows),
        warning_rows=tuple(row.to_dict() for row in plan.warning_rows),
        error_rows=(),
        provenance_rows=tuple(row.to_dict() for row in plan.provenance_rows),
        warnings=plan.warnings,
        errors=(),
        provenance=plan.provenance,
        output_written=True,
    )
    manifest_dict = written_manifest.to_dict()
    if _MANIFEST_JSON in targets:
        _write_text_atomic(targets[_MANIFEST_JSON].path, _json_text(manifest_dict))
        output_hashes[_MANIFEST_JSON] = _file_sha256(targets[_MANIFEST_JSON].path)
        written_artifacts = tuple(
            replace(artifact, sha256=output_hashes.get(artifact.name), written=True, exists=True)
            for artifact in written_artifacts
        )

    return VisualizationWriteResult(
        schema_version=SCHEMA_VERSION,
        status="ok",
        output_written=True,
        output_paths=plan.output_paths,
        output_hashes=output_hashes,
        output_row_counts=output_row_counts,
        artifacts=written_artifacts,
        manifest=manifest_dict,
        visual_qc_rows=plan.visual_qc_rows,
        warning_rows=plan.warning_rows,
        error_rows=(),
        provenance_rows=plan.provenance_rows,
        provenance=plan.provenance,
        warnings=plan.warnings,
        errors=(),
    )


def build_manifest_from_visualization_outputs(
    *,
    source_records: Sequence[Mapping[str, Any] | RenderedArtifactRow],
    artifact_rows: Sequence[Mapping[str, Any] | RenderedArtifactRow],
    output_hashes: Mapping[str, str] | None = None,
    output_row_counts: Mapping[str, int] | None = None,
    format_spec: VisualizationFormatSpec | None = None,
    report_spec: ReportSpec | None = None,
    plot_specs: Sequence[PointIntervalPlotSpec] = (),
    visual_qc_rows: Sequence[Mapping[str, Any]] = (),
    warning_rows: Sequence[Mapping[str, Any]] = (),
    error_rows: Sequence[Mapping[str, Any]] = (),
    provenance_rows: Sequence[Mapping[str, Any]] = (),
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
    output_written: bool = False,
) -> VisualizationManifest:
    """Build a JSON-safe visualization manifest from planned/rendered rows."""

    source_rows = tuple(_artifact_from_record(row) for row in source_records)
    output_rows = tuple(_artifact_from_record(row) for row in artifact_rows)
    if output_written:
        output_rows = tuple(replace(row, written=True) for row in output_rows)
    output_hash_mapping = _json_safe_mapping(output_hashes or {})
    output_count_mapping = _json_safe_mapping(output_row_counts or {})
    output_paths = {row.name: row.path for row in output_rows if row.path is not None}
    source_paths = tuple(row.path for row in source_rows if row.path is not None)
    return VisualizationManifest(
        schema_version=SCHEMA_VERSION,
        source_paths=source_paths,
        source_hashes={row.name: row.sha256 for row in source_rows if row.sha256 is not None},
        source_row_counts={row.name: row.row_count for row in source_rows if row.row_count is not None},
        output_paths=output_paths,
        output_hashes=output_hash_mapping,
        output_row_counts=output_count_mapping,
        renderer_settings=_renderer_settings(format_spec or VisualizationFormatSpec(), output_rows),
        text_settings=_text_settings(report_spec, plot_specs),
        layout_settings=_layout_settings(plot_specs),
        visual_qc_rows=tuple(_json_safe_mapping(row) for row in visual_qc_rows),
        warning_rows=tuple(_json_safe_mapping(row) for row in warning_rows),
        error_rows=tuple(_json_safe_mapping(row) for row in error_rows),
        provenance_rows=tuple(_json_safe_mapping(row) for row in provenance_rows),
        warnings=warnings,
        errors=errors,
        provenance=_manifest_provenance(provenance or {}),
        rows=tuple(row.to_dict() for row in (*source_rows, *output_rows)),
    )


def _build_previews(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    report_spec: ReportSpec | None,
    plot_specs: Sequence[PointIntervalPlotSpec],
    targets: Mapping[str, _ResolvedOutputPath],
    metadata: Mapping[str, Any],
    visual_qc_rows: Sequence[VisualQcRow],
    artifact_rows: Sequence[RenderedArtifactRow],
    warnings: Sequence[str],
    provenance: Mapping[str, Any],
    errors: list[str],
    error_rows: list[VisualizationErrorRow],
) -> tuple[dict[str, str], dict[str, str]]:
    report_previews: dict[str, str] = {}
    figure_previews: dict[str, str] = {}
    figure_paths = _figure_paths_by_plot_id(targets, plot_specs)
    for name, target in targets.items():
        if target.role == "figure" and target.file_format == "svg":
            spec = plot_specs[0] if plot_specs else None
            if spec is None:
                continue
            try:
                figure_previews[name] = render_point_interval_svg(source_rows, spec, metadata=metadata)
            except (TypeError, ValueError) as exc:
                _append_error(errors, error_rows, "figure_preview_failed", str(exc), scope="figure", artifact_id=spec.plot_id)
        elif target.role == "report" and report_spec is not None:
            try:
                report_previews[name] = build_report_document(
                    source_rows,
                    report_spec=report_spec,
                    format=target.file_format,
                    metadata=metadata,
                    figure_paths=figure_paths,
                    visual_qc_rows=visual_qc_rows,
                    artifact_rows=artifact_rows,
                    warnings=warnings,
                    provenance=provenance,
                )
            except (TypeError, ValueError) as exc:
                _append_error(errors, error_rows, "report_preview_failed", str(exc), scope="report", artifact_id=report_spec.report_id)
    return report_previews, figure_previews


def _load_sources(
    *,
    rows: Iterable[Mapping[str, Any]] | None,
    source_spec: VisualizationSourceSpec | None,
    source_specs: Sequence[VisualizationSourceSpec],
    strict: bool,
) -> tuple[_LoadedSource, ...]:
    specs = tuple(source_specs)
    if source_spec is not None:
        specs = (*specs, source_spec)
    loaded: list[_LoadedSource] = []
    if rows is not None:
        memory_rows = _normalize_rows(rows, field_name="rows")
        loaded.append(
            _LoadedSource(
                spec=VisualizationSourceSpec(source_id="in_memory"),
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
                spec=VisualizationSourceSpec(source_id="missing_input"),
                rows=(),
                columns=(),
                errors=(message,),
            )
        )
    return tuple(loaded)


def _load_source_spec(spec: VisualizationSourceSpec) -> _LoadedSource:
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
    resolved = path.resolve(strict=False)
    return _LoadedSource(
        spec=spec,
        rows=rows,
        columns=_columns_for_rows(rows),
        path=resolved,
        file_format=file_format,
        sha256=_file_sha256(path),
    )


def _read_delimited_rows(path: Path, *, delimiter: str) -> tuple[dict[str, Any], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Source table {path!s} is missing a header row.")
        return tuple({str(key): _json_safe(value) for key, value in row.items()} for row in reader)


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


def _resolved_output_targets(output_spec: VisualizationOutputSpec) -> dict[str, _ResolvedOutputPath]:
    root = _resolved_output_root(output_spec.output_root)
    configured = _configured_output_paths(output_spec)
    targets = {name: _resolved_output_path(root, name, path) for name, path in configured.items()}
    _reject_duplicate_targets(targets)
    return targets


def _configured_output_paths(output_spec: VisualizationOutputSpec) -> dict[str, str | Path]:
    configured: dict[str, str | Path] = {}
    for name, field_name in (
        (_REPORT_MARKDOWN, "report_markdown_path"),
        (_REPORT_HTML, "report_html_path"),
        (_REPORT_TEXT, "report_text_path"),
        (_FIGURE_SVG, "figure_svg_path"),
        (_FIGURE_PNG, "figure_png_path"),
        (_FIGURE_PDF, "figure_pdf_path"),
        (_MANIFEST_JSON, "manifest_path"),
    ):
        value = getattr(output_spec, field_name)
        if value is not None:
            configured[name] = value
    if not configured:
        raise ValueError("VisualizationOutputSpec must configure at least one output path.")
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
    file_format = _format_from_output_name(name)
    role = _role_from_output_name(name)
    return _ResolvedOutputPath(
        name=name,
        path=resolved,
        relative_path=relative.as_posix(),
        file_format=file_format,
        role=role,
        renderer=_renderer_name(role, file_format),
    )


def _validate_requested_formats(
    spec: VisualizationFormatSpec,
    targets: Mapping[str, _ResolvedOutputPath],
    warnings: list[str],
    errors: list[str],
    warning_rows: list[VisualizationWarningRow],
    error_rows: list[VisualizationErrorRow],
) -> None:
    target_report_formats = {target.file_format for target in targets.values() if target.role == "report"}
    target_figure_formats = {target.file_format for target in targets.values() if target.role == "figure"}
    target_manifest_formats = {target.file_format for target in targets.values() if target.role == "manifest"}
    for report_format in (*spec.report_formats, *target_report_formats):
        if report_format not in _REPORT_FORMATS:
            _append_error(errors, error_rows, "unsupported_report_format", f"Unsupported report format: {report_format}", scope="format")
        elif report_format in spec.report_formats and report_format not in target_report_formats:
            _append_warning(warnings, warning_rows, "requested_report_format_without_path", f"Requested report format {report_format!r} has no configured output path.", scope="format")
    for figure_format in (*spec.figure_formats, *target_figure_formats):
        if figure_format not in _FIGURE_FORMATS:
            _append_error(errors, error_rows, "unsupported_figure_format", f"Unsupported figure format: {figure_format}", scope="format")
        elif figure_format in _OPTIONAL_UNAVAILABLE_FIGURE_FORMATS:
            message = f"Figure renderer for {figure_format!r} is optional and unavailable in dependency-free research-viz."
            _append_warning(warnings, warning_rows, "renderer_unavailable", message, scope="renderer")
            if spec.optional_renderer_policy == "error" or figure_format in target_figure_formats:
                _append_error(errors, error_rows, "renderer_unavailable", message, scope="renderer")
        elif figure_format in spec.figure_formats and figure_format not in target_figure_formats:
            _append_warning(warnings, warning_rows, "requested_figure_format_without_path", f"Requested figure format {figure_format!r} has no configured output path.", scope="format")
    for manifest_format in (*spec.manifest_formats, *target_manifest_formats):
        if manifest_format not in _MANIFEST_FORMATS:
            _append_error(errors, error_rows, "unsupported_manifest_format", f"Unsupported manifest format: {manifest_format}", scope="format")
        elif manifest_format in spec.manifest_formats and manifest_format not in target_manifest_formats:
            _append_warning(warnings, warning_rows, "requested_manifest_format_without_path", f"Requested manifest format {manifest_format!r} has no configured output path.", scope="format")


def _validate_required_columns(
    source_columns: Sequence[str],
    plot_specs: Sequence[PointIntervalPlotSpec],
    report_spec: ReportSpec | None,
    errors: list[str],
    error_rows: list[VisualizationErrorRow],
) -> None:
    available = set(source_columns)
    for spec in plot_specs:
        missing = [column for column in spec.source_columns() if column not in available]
        if missing:
            _append_error(
                errors,
                error_rows,
                "missing_required_columns",
                f"Plot {spec.plot_id!r} is missing required source columns: {', '.join(missing)}",
                scope="columns",
                artifact_id=spec.plot_id,
            )
    if report_spec is None:
        return
    for section in report_spec.sections:
        if isinstance(section, TableSectionSpec):
            required = tuple(dict.fromkeys((*section.columns, *section.required_columns)))
            missing = [column for column in required if column not in available and not section.rows]
            if missing:
                _append_error(
                    errors,
                    error_rows,
                    "missing_required_columns",
                    f"Table section {section.section_id!r} is missing required source columns: {', '.join(missing)}",
                    scope="columns",
                    artifact_id=section.section_id,
                )


def _validate_report_templates(
    report_spec: ReportSpec | None,
    metadata: Mapping[str, Any],
    errors: list[str],
    error_rows: list[VisualizationErrorRow],
) -> None:
    if report_spec is None:
        return
    metadata = _json_safe_mapping({**metadata, **report_spec.metadata})
    for field_path, template in _iter_template_strings(report_spec.to_dict()):
        for message in _template_errors(template, metadata, field_path=field_path):
            _append_error(errors, error_rows, "missing_template_field", message, scope="text", artifact_id=report_spec.report_id)


def _iter_template_strings(payload: Any, *, prefix: str = "") -> tuple[tuple[str, str], ...]:
    text_keys = {
        "title",
        "subtitle",
        "caption",
        "footnote",
        "alt_text",
        "methods_note",
        "heading",
        "text",
        "x_label",
        "y_label",
    }
    found: list[tuple[str, str]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in text_keys and isinstance(value, str):
                found.append((path, value))
            elif key_text in {"label_mapping", "panel_labels"} and isinstance(value, Mapping):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, str):
                        found.append((f"{path}.{nested_key}", nested_value))
            else:
                found.extend(_iter_template_strings(value, prefix=path))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for index, value in enumerate(payload):
            found.extend(_iter_template_strings(value, prefix=f"{prefix}[{index}]"))
    return tuple(found)


def _template_errors(template: str, metadata: Mapping[str, Any], *, field_path: str) -> tuple[str, ...]:
    formatter = Formatter()
    errors: list[str] = []
    for _, raw_name, format_spec, _ in formatter.parse(template):
        if raw_name is None:
            continue
        name = raw_name
        if "." in name or "[" in name or "]" in name:
            errors.append(f"Template field {name!r} in {field_path!r} is not a simple metadata key.")
        elif name not in metadata:
            errors.append(f"Template field {name!r} in {field_path!r} is missing from metadata.")
        elif "{" in format_spec or "}" in format_spec:
            errors.append(f"Template format spec for {name!r} in {field_path!r} must not contain nested fields.")
    return tuple(errors)


def _artifact_row_for_target(
    target: _ResolvedOutputPath,
    rows: Sequence[Mapping[str, Any]],
    plot_specs: Sequence[PointIntervalPlotSpec],
    *,
    overwrite: bool,
) -> RenderedArtifactRow:
    warnings: list[str] = []
    errors: list[str] = []
    if target.path.exists() and not overwrite:
        warnings.append(f"Output path already exists and render will refuse overwrite by default: {target.path}")
    if target.file_format in _OPTIONAL_UNAVAILABLE_FIGURE_FORMATS:
        errors.append(f"Renderer for {target.file_format!r} is unavailable.")
    artifact_id = None
    if target.role == "figure" and plot_specs:
        artifact_id = plot_specs[0].plot_id
    elif target.role == "report":
        artifact_id = "report"
    return RenderedArtifactRow(
        role=target.role,
        name=target.name,
        path=str(target.path),
        relative_path=target.relative_path,
        file_format=target.file_format,
        renderer=target.renderer,
        artifact_id=artifact_id,
        row_count=1 if target.role == "manifest" else len(rows),
        columns=_columns_for_rows(rows) if target.role in {"report", "figure"} else (),
        exists=target.path.exists(),
        planned=True,
        written=False,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _source_artifact_row(source: _LoadedSource) -> RenderedArtifactRow:
    path = str(source.path) if source.path is not None else None
    return RenderedArtifactRow(
        role="source",
        name=source.spec.source_id,
        path=path,
        relative_path=None,
        file_format=source.file_format,
        renderer=None,
        artifact_id=source.spec.source_id,
        row_count=len(source.rows),
        columns=source.columns,
        sha256=source.sha256,
        exists=source.path.exists() if source.path is not None else None,
        planned=False,
        written=False,
        warnings=source.warnings,
        errors=source.errors,
    )


def _artifact_from_record(record: Mapping[str, Any] | RenderedArtifactRow) -> RenderedArtifactRow:
    if isinstance(record, RenderedArtifactRow):
        return record
    return RenderedArtifactRow(
        role=str(record.get("role", "output")),
        name=str(record.get("name", "artifact")),
        path=record.get("path"),
        relative_path=record.get("relative_path"),
        file_format=record.get("file_format"),
        renderer=record.get("renderer"),
        artifact_id=record.get("artifact_id"),
        row_count=record.get("row_count"),
        columns=tuple(record.get("columns", ())),
        sha256=record.get("sha256"),
        exists=record.get("exists"),
        planned=bool(record.get("planned", True)),
        written=bool(record.get("written", False)),
        warnings=tuple(record.get("warnings", ())),
        errors=tuple(record.get("errors", ())),
    )


def _combined_plot_specs(
    *,
    plot_spec: PointIntervalPlotSpec | None,
    plot_specs: Sequence[PointIntervalPlotSpec],
) -> tuple[PointIntervalPlotSpec, ...]:
    specs = tuple(plot_specs)
    if plot_spec is not None:
        specs = (*specs, plot_spec)
    return specs


def _figure_paths_by_plot_id(
    targets: Mapping[str, _ResolvedOutputPath],
    plot_specs: Sequence[PointIntervalPlotSpec],
) -> dict[str, str]:
    if not plot_specs:
        return {}
    figure_targets = [target for target in targets.values() if target.role == "figure"]
    if not figure_targets:
        return {}
    return {plot_specs[0].plot_id: figure_targets[0].relative_path}


def _output_row_counts(
    rows: Sequence[Mapping[str, Any]],
    targets: Mapping[str, _ResolvedOutputPath],
    *,
    source_row_count: int | None = None,
) -> dict[str, int]:
    row_count = len(rows) if source_row_count is None else source_row_count
    counts: dict[str, int] = {}
    for name, target in targets.items():
        counts[name] = 1 if target.role == "manifest" else row_count
    return counts


def _renderer_settings(format_spec: VisualizationFormatSpec, artifact_rows: Sequence[RenderedArtifactRow]) -> dict[str, Any]:
    return {
        "format_spec": format_spec.to_dict(),
        "renderers": {
            row.name: {
                "role": row.role,
                "file_format": row.file_format,
                "renderer": row.renderer,
                "available": row.file_format not in _OPTIONAL_UNAVAILABLE_FIGURE_FORMATS,
            }
            for row in artifact_rows
        },
        "required_dependencies": [],
    }


def _text_settings(report_spec: ReportSpec | None, plot_specs: Sequence[PointIntervalPlotSpec]) -> dict[str, Any]:
    return {
        "report": report_spec.to_dict() if report_spec is not None else None,
        "plots": [
            {
                "plot_id": spec.plot_id,
                "text": spec.text.to_dict(),
                "axes": spec.axes.to_dict(),
                "legend": spec.legend.to_dict(),
                "caption": spec.caption.to_dict(),
            }
            for spec in plot_specs
        ],
    }


def _layout_settings(plot_specs: Sequence[PointIntervalPlotSpec]) -> dict[str, Any]:
    return {spec.plot_id: spec.layout.to_dict() for spec in plot_specs}


def _manifest_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    combined = {
        "writer_module": __name__,
        "schema_version": SCHEMA_VERSION,
    }
    combined.update(_json_safe_mapping(provenance))
    return combined


def _coerced_provenance_rows(provenance: Mapping[str, Any]) -> tuple[VisualizationProvenanceRow, ...]:
    return tuple(VisualizationProvenanceRow(key=str(key), value=value, source="input") for key, value in provenance.items())


def _extend_messages(
    warnings: list[str],
    errors: list[str],
    warning_rows: list[VisualizationWarningRow],
    error_rows: list[VisualizationErrorRow],
    new_warnings: Sequence[str],
    new_errors: Sequence[str],
    *,
    scope: str,
    artifact_id: str | None = None,
) -> None:
    for warning in new_warnings:
        _append_warning(warnings, warning_rows, "warning", warning, scope=scope, artifact_id=artifact_id)
    for error in new_errors:
        _append_error(errors, error_rows, "error", error, scope=scope, artifact_id=artifact_id)


def _append_warning(
    warnings: list[str],
    warning_rows: list[VisualizationWarningRow],
    code: str,
    message: str,
    *,
    scope: str,
    artifact_id: str | None = None,
) -> None:
    if message not in warnings:
        warnings.append(message)
    warning_rows.append(VisualizationWarningRow(code=code, message=message, scope=scope, artifact_id=artifact_id))


def _append_error(
    errors: list[str],
    error_rows: list[VisualizationErrorRow],
    code: str,
    message: str,
    *,
    scope: str,
    artifact_id: str | None = None,
) -> None:
    if message not in errors:
        errors.append(message)
    error_rows.append(VisualizationErrorRow(code=code, message=message, scope=scope, artifact_id=artifact_id))


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


def _role_from_output_name(name: str) -> str:
    if name.startswith("report_"):
        return "report"
    if name.startswith("figure_"):
        return "figure"
    if name == _MANIFEST_JSON:
        return "manifest"
    return "artifact"


def _format_from_output_name(name: str) -> str:
    if name == _REPORT_MARKDOWN:
        return "markdown"
    if name == _REPORT_HTML:
        return "html"
    if name == _REPORT_TEXT:
        return "text"
    if name == _FIGURE_SVG:
        return "svg"
    if name == _FIGURE_PNG:
        return "png"
    if name == _FIGURE_PDF:
        return "pdf"
    if name == _MANIFEST_JSON:
        return "json"
    return name


def _renderer_name(role: str, file_format: str) -> str:
    if role == "report":
        return f"{file_format}_report"
    if role == "figure":
        return f"{file_format}_figure"
    if role == "manifest":
        return "json_manifest"
    return file_format


def _infer_source_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".tsv" or suffix == ".txt":
        return "tsv"
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    raise ValueError(f"Unsupported source file suffix for {path!s}. Use .tsv, .csv, or .json.")


def _columns_for_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    columns: list[str] = []
    for row in rows:
        for column in row:
            text_column = str(column)
            if text_column not in columns:
                columns.append(text_column)
    return tuple(columns)


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


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True) + "\n"


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
        value = {field_.name: getattr(value, field_.name) for field_ in fields(value)}
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
        return value if math.isfinite(value) else str(value)
    if isinstance(value, int | str | bool) or value is None:
        return value
    return str(value)


def _non_empty_text(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


__all__ = [
    "RenderedArtifactRow",
    "SCHEMA_VERSION",
    "VisualizationErrorRow",
    "VisualizationFormatSpec",
    "VisualizationManifest",
    "VisualizationOutputSpec",
    "VisualizationPlan",
    "VisualizationProvenanceRow",
    "VisualizationSourceSpec",
    "VisualizationWarningRow",
    "VisualizationWriteResult",
    "build_manifest_from_visualization_outputs",
    "plan_visualization_outputs",
    "render_visualization_outputs",
]
