"""Pathless visualization/report handoff for tabular association rows.

This module consumes already-computed generic association rows and produces
in-memory rowsets/specs that the existing ``research-viz`` primitives can use
later. It does not compute statistics, render figures or reports, write files,
or implement dataframe backends.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
import json
import math
from pathlib import Path
from string import Formatter
from typing import Any

from research_platform.viz.outputs import SCHEMA_VERSION
from research_platform.viz.plots import (
    FigureTextSpec,
    PointIntervalPlotSpec,
    VisualQcRow,
    VisualQcSpec,
    build_point_interval_plot_spec,
    build_visual_layout_qc_rows,
)
from research_platform.viz.reports import FigureSectionSpec, ReportSpec, TableSectionSpec


TABULAR_ASSOCIATION_SCHEMA_VERSION = ".".join(("research_platform", "analysis", "tabular_associations", "v1"))
TABULAR_ASSOCIATION_VISUALIZATION_HANDOFF_VERSION = (
    "research_platform.viz.tabular_associations.visualization_handoff.v1"
)
RUNTIME_BACKEND_RECORDS = "records"

_PRIMARY_ASSOCIATION_ROWSET = "association_rows"
_PRIMARY_PUBLICATION_ROWSET = "publication_table_rows"
_ROWSET_NAMES = (
    _PRIMARY_ASSOCIATION_ROWSET,
    _PRIMARY_PUBLICATION_ROWSET,
    "publication_display_rows",
    "publication_machine_rows",
    "multiplicity_rows",
    "qc_rows",
    "missingness_rows",
    "provenance_rows",
)
_DEFAULT_ROWSET_NAMES = {name: name for name in _ROWSET_NAMES}
_OUTPUT_ROWSET_NAMES = (
    "input_summary_rows",
    "association_dataset_rows",
    "figure_specs",
    "report_handoff_specs",
    "spec_rows",
    "visual_qc_rows",
    "qc_summary_rows",
    "missingness_summary_rows",
    "multiplicity_summary_rows",
    "provenance_table_rows",
    "manifest_rows",
)
_DATASET_COLUMNS = (
    "label",
    "estimate",
    "lower",
    "upper",
    "has_interval",
    "p_value",
    "q_value",
    "workflow_id",
    "method_id",
    "method_kind",
    "method_name",
    "family_id",
    "source_id",
    "outcome_id",
    "predictor_id",
    "statistic_name",
    "statistic_value",
    "result_row_id",
    "pair_id",
    "input_row_index",
    "status",
)
_IDENTIFIER_FIELDS = (
    "workflow_id",
    "method_id",
    "method_kind",
    "method_name",
    "family_id",
    "source_id",
    "outcome_id",
    "predictor_id",
    "covariate_ids",
    "statistic_name",
    "statistic_value",
    "p_value",
    "q_value",
    "status",
    "warnings",
    "errors",
    "result_row_id",
    "pair_id",
    "input_row_index",
)
_COMMON_INTERVAL_FIELD_PAIRS = (
    ("ci_low", "ci_high"),
    ("ci_lower", "ci_upper"),
    ("lower", "upper"),
    ("lower_bound", "upper_bound"),
    ("interval_lower", "interval_upper"),
    ("confidence_interval_low", "confidence_interval_high"),
)


class _AssociationVisualizationRowMixin:
    """Shared JSON/TSV conversion for public handoff rows."""

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)

    def to_tsv_row(self) -> dict[str, str]:
        return _tsv_safe_mapping(self.to_dict())


@dataclass(frozen=True)
class AssociationVisualizationTextSpec:
    """Configurable text controls for association visualization handoff."""

    title: str | None = None
    subtitle: str | None = None
    caption: str | None = None
    footnote: str | None = None
    x_axis_label: str | None = None
    y_axis_label: str | None = None
    legend_title: str | None = None
    legend_labels: Mapping[str, str] = field(default_factory=dict)
    panel_labels: Mapping[str, str] = field(default_factory=dict)
    alt_text: str | None = None
    methods_note: str | None = None
    report_title: str | None = None
    report_subtitle: str | None = None
    report_caption: str | None = None
    report_footnote: str | None = None
    report_section_headings: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "legend_labels", {str(key): str(value) for key, value in self.legend_labels.items()})
        object.__setattr__(self, "panel_labels", {str(key): str(value) for key, value in self.panel_labels.items()})
        object.__setattr__(
            self,
            "report_section_headings",
            {str(key): str(value) for key, value in self.report_section_headings.items()},
        )
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AssociationVisualizationFigureSpec(_AssociationVisualizationRowMixin):
    """Pathless figure handoff around an existing point/interval spec."""

    workflow_id: str
    figure_id: str
    figure_name: str
    dataset_rowset_name: str
    plot_type: str
    plot_spec: PointIntervalPlotSpec | Mapping[str, Any]
    row_count: int
    has_interval: bool
    executed: bool = True
    plan_only: bool = False
    will_render: bool = False
    rendered: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "figure_id", _non_empty_text(self.figure_id, field_name="figure_id"))
        object.__setattr__(self, "figure_name", _non_empty_text(self.figure_name, field_name="figure_name"))
        object.__setattr__(
            self,
            "dataset_rowset_name",
            _non_empty_text(self.dataset_rowset_name, field_name="dataset_rowset_name"),
        )
        object.__setattr__(self, "plot_type", _non_empty_text(self.plot_type, field_name="plot_type"))
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(self, "has_interval", bool(self.has_interval))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_render", False)
        object.__setattr__(self, "rendered", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationReportHandoffSpec(_AssociationVisualizationRowMixin):
    """Pathless report handoff around an existing report spec."""

    workflow_id: str
    report_id: str
    report_name: str
    report_spec: ReportSpec | Mapping[str, Any]
    section_ids: Sequence[str]
    section_count: int
    executed: bool = True
    plan_only: bool = False
    will_render: bool = False
    rendered: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "report_id", _non_empty_text(self.report_id, field_name="report_id"))
        object.__setattr__(self, "report_name", _non_empty_text(self.report_name, field_name="report_name"))
        object.__setattr__(self, "section_ids", tuple(str(section_id) for section_id in self.section_ids))
        object.__setattr__(self, "section_count", int(self.section_count))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_render", False)
        object.__setattr__(self, "rendered", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationVisualizationInputSummaryRow(_AssociationVisualizationRowMixin):
    """One input rowset summary for no-write visualization handoff."""

    workflow_id: str
    rowset_name: str
    source_rowset_name: str
    row_count: int
    normalized_row_count: int
    selected_for_visualization: bool = False
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "rowset_name", _non_empty_text(self.rowset_name, field_name="rowset_name"))
        object.__setattr__(
            self,
            "source_rowset_name",
            _non_empty_text(self.source_rowset_name, field_name="source_rowset_name"),
        )
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(self, "normalized_row_count", int(self.normalized_row_count))
        object.__setattr__(self, "selected_for_visualization", bool(self.selected_for_visualization))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationVisualizationDatasetRow(_AssociationVisualizationRowMixin):
    """Point/interval-ready association summary row."""

    workflow_id: str
    dataset_row_id: str
    source_rowset_name: str
    source_row_index: int
    input_row_index: int
    label: str
    estimate: Any
    lower: Any = None
    upper: Any = None
    has_interval: bool = False
    group_id: str | None = None
    data_label: str | None = None
    method_id: str | None = None
    method_kind: str | None = None
    method_name: str | None = None
    family_id: str | None = None
    source_id: str | None = None
    outcome_id: str | None = None
    predictor_id: str | None = None
    covariate_ids: Sequence[str] = ()
    statistic_name: str | None = None
    statistic_value: Any = None
    p_value: Any = None
    q_value: Any = None
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    result_row_id: str | None = None
    pair_id: str | None = None
    extra_fields: Mapping[str, Any] = field(default_factory=dict)
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "dataset_row_id", _non_empty_text(self.dataset_row_id, field_name="dataset_row_id"))
        object.__setattr__(
            self,
            "source_rowset_name",
            _non_empty_text(self.source_rowset_name, field_name="source_rowset_name"),
        )
        object.__setattr__(self, "source_row_index", int(self.source_row_index))
        input_row_index = int(self.input_row_index)
        if input_row_index < 0:
            raise ValueError("input_row_index must be non-negative.")
        object.__setattr__(self, "input_row_index", input_row_index)
        object.__setattr__(self, "label", _non_empty_text(self.label, field_name="label"))
        for field_name in (
            "group_id",
            "data_label",
            "method_id",
            "method_kind",
            "method_name",
            "family_id",
            "source_id",
            "outcome_id",
            "predictor_id",
            "statistic_name",
            "result_row_id",
            "pair_id",
        ):
            object.__setattr__(self, field_name, _optional_text(getattr(self, field_name)))
        object.__setattr__(self, "estimate", _json_safe(self.estimate))
        object.__setattr__(self, "lower", _json_safe(self.lower))
        object.__setattr__(self, "upper", _json_safe(self.upper))
        object.__setattr__(self, "has_interval", bool(self.has_interval))
        object.__setattr__(self, "covariate_ids", _text_tuple(self.covariate_ids))
        object.__setattr__(self, "statistic_value", _json_safe(self.statistic_value))
        object.__setattr__(self, "p_value", _json_safe(self.p_value))
        object.__setattr__(self, "q_value", _json_safe(self.q_value))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "extra_fields", _json_safe_mapping(self.extra_fields))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationVisualizationSpecRow(_AssociationVisualizationRowMixin):
    """JSON/TSV-safe spec inventory row for the handoff bundle."""

    workflow_id: str
    spec_id: str
    spec_type: str
    rowset_name: str
    artifact_id: str | None
    row_count: int
    columns: Sequence[str] = ()
    spec_payload: Mapping[str, Any] = field(default_factory=dict)
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "spec_id", _non_empty_text(self.spec_id, field_name="spec_id"))
        object.__setattr__(self, "spec_type", _non_empty_text(self.spec_type, field_name="spec_type"))
        object.__setattr__(self, "rowset_name", _non_empty_text(self.rowset_name, field_name="rowset_name"))
        object.__setattr__(self, "artifact_id", _optional_text(self.artifact_id))
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(self, "columns", tuple(str(column) for column in self.columns))
        object.__setattr__(self, "spec_payload", _json_safe_mapping(self.spec_payload))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationVisualizationManifestRow(_AssociationVisualizationRowMixin):
    """Manifest row for pathless visualization/report handoff."""

    workflow_id: str
    rowset_name: str
    row_count: int
    tabular_association_schema_version: str
    visualization_handoff_schema_version: str
    research_viz_schema_version: str
    input_row_counts: Mapping[str, int]
    output_row_counts: Mapping[str, int]
    planned_figure_names: Sequence[str]
    planned_report_names: Sequence[str]
    planned_output_names: Sequence[str]
    source_rowset_names: Mapping[str, str]
    runtime_backend: str
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    no_output_written: bool
    output_paths_written: Sequence[str]
    will_render: bool
    rendered: bool
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "rowset_name", _non_empty_text(self.rowset_name, field_name="rowset_name"))
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(
            self,
            "tabular_association_schema_version",
            _non_empty_text(
                self.tabular_association_schema_version,
                field_name="tabular_association_schema_version",
            ),
        )
        object.__setattr__(
            self,
            "visualization_handoff_schema_version",
            _non_empty_text(
                self.visualization_handoff_schema_version,
                field_name="visualization_handoff_schema_version",
            ),
        )
        object.__setattr__(
            self,
            "research_viz_schema_version",
            _non_empty_text(self.research_viz_schema_version, field_name="research_viz_schema_version"),
        )
        object.__setattr__(self, "input_row_counts", _json_safe_mapping(self.input_row_counts))
        object.__setattr__(self, "output_row_counts", _json_safe_mapping(self.output_row_counts))
        object.__setattr__(self, "planned_figure_names", _text_tuple(self.planned_figure_names))
        object.__setattr__(self, "planned_report_names", _text_tuple(self.planned_report_names))
        object.__setattr__(self, "planned_output_names", _text_tuple(self.planned_output_names))
        object.__setattr__(self, "source_rowset_names", _json_safe_mapping(self.source_rowset_names))
        object.__setattr__(self, "runtime_backend", _non_empty_text(self.runtime_backend, field_name="runtime_backend"))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(str(path) for path in self.output_paths_written if str(path)),
        )
        object.__setattr__(self, "will_render", False)
        object.__setattr__(self, "rendered", False)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))


@dataclass(frozen=True)
class AssociationVisualizationProvenanceRow(_AssociationVisualizationRowMixin):
    """Provenance/methods-note row for visualization handoff."""

    workflow_id: str
    key: str
    value: Any
    source: str = "tabular_association_visualization_handoff"
    input_row_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "key", _non_empty_text(self.key, field_name="key"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "source", _non_empty_text(self.source, field_name="source"))
        if self.input_row_index is not None:
            object.__setattr__(self, "input_row_index", int(self.input_row_index))


@dataclass(frozen=True)
class TabularAssociationVisualizationPlan:
    """No-write plan for generic tabular association visualization handoff."""

    schema_version: str
    tabular_association_schema_version: str
    visualization_handoff_schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    no_output_written: bool
    output_paths_written: Sequence[str]
    will_render: bool
    rendered: bool
    figure_rendered: bool
    report_rendered: bool
    runtime_backend: str
    primary_rowset_name: str
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    input_summary_rows: Sequence[AssociationVisualizationInputSummaryRow | Mapping[str, Any]]
    association_dataset_rows: Sequence[AssociationVisualizationDatasetRow | Mapping[str, Any]]
    publication_display_rows: Sequence[Mapping[str, Any]]
    publication_machine_rows: Sequence[Mapping[str, Any]]
    qc_summary_rows: Sequence[Mapping[str, Any]]
    missingness_summary_rows: Sequence[Mapping[str, Any]]
    multiplicity_summary_rows: Sequence[Mapping[str, Any]]
    provenance_table_rows: Sequence[AssociationVisualizationProvenanceRow | Mapping[str, Any]]
    figure_specs: Sequence[AssociationVisualizationFigureSpec | Mapping[str, Any]]
    report_handoff_specs: Sequence[AssociationReportHandoffSpec | Mapping[str, Any]]
    spec_rows: Sequence[AssociationVisualizationSpecRow | Mapping[str, Any]]
    visual_qc_rows: Sequence[VisualQcRow | Mapping[str, Any]]
    manifest_rows: Sequence[AssociationVisualizationManifestRow | Mapping[str, Any]]
    planned_row_counts: Mapping[str, int] = field(default_factory=dict)
    planned_figure_names: Sequence[str] = ()
    planned_report_names: Sequence[str] = ()
    planned_output_names: Sequence[str] = ()

    def __post_init__(self) -> None:
        _normalize_plan_result(self, executed=False, plan_only=True)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularAssociationVisualizationResult:
    """In-memory no-write visualization/report handoff for supplied rows."""

    schema_version: str
    tabular_association_schema_version: str
    visualization_handoff_schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    no_output_written: bool
    output_paths_written: Sequence[str]
    will_render: bool
    rendered: bool
    figure_rendered: bool
    report_rendered: bool
    runtime_backend: str
    primary_rowset_name: str
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    input_summary_rows: Sequence[AssociationVisualizationInputSummaryRow | Mapping[str, Any]]
    association_dataset_rows: Sequence[AssociationVisualizationDatasetRow | Mapping[str, Any]]
    publication_display_rows: Sequence[Mapping[str, Any]]
    publication_machine_rows: Sequence[Mapping[str, Any]]
    qc_summary_rows: Sequence[Mapping[str, Any]]
    missingness_summary_rows: Sequence[Mapping[str, Any]]
    multiplicity_summary_rows: Sequence[Mapping[str, Any]]
    provenance_table_rows: Sequence[AssociationVisualizationProvenanceRow | Mapping[str, Any]]
    figure_specs: Sequence[AssociationVisualizationFigureSpec | Mapping[str, Any]]
    report_handoff_specs: Sequence[AssociationReportHandoffSpec | Mapping[str, Any]]
    spec_rows: Sequence[AssociationVisualizationSpecRow | Mapping[str, Any]]
    visual_qc_rows: Sequence[VisualQcRow | Mapping[str, Any]]
    manifest_rows: Sequence[AssociationVisualizationManifestRow | Mapping[str, Any]]
    planned_row_counts: Mapping[str, int] = field(default_factory=dict)
    planned_figure_names: Sequence[str] = ()
    planned_report_names: Sequence[str] = ()
    planned_output_names: Sequence[str] = ()

    def __post_init__(self) -> None:
        _normalize_plan_result(self, executed=True, plan_only=False)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


def plan_tabular_association_visualization_handoff(
    association_rows: Iterable[Any] | None = None,
    *,
    publication_table_rows: Iterable[Any] | None = None,
    publication_display_rows: Iterable[Any] | None = None,
    publication_machine_rows: Iterable[Any] | None = None,
    multiplicity_rows: Iterable[Any] | None = None,
    qc_rows: Iterable[Any] | None = None,
    missingness_rows: Iterable[Any] | None = None,
    provenance_rows: Iterable[Any] | None = None,
    workflow_id: str | None = None,
    source_rowset_names: Mapping[str, str] | None = None,
    estimate_field: str | None = None,
    lower_field: str | None = None,
    upper_field: str | None = None,
    label_field: str | None = None,
    group_field: str | None = None,
    data_label_field: str | None = None,
    figure_id: str = "association-summary-figure",
    figure_name: str | None = None,
    report_id: str = "association-summary-report",
    report_name: str | None = None,
    text_spec: AssociationVisualizationTextSpec | Mapping[str, Any] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    footnote: str | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    legend_title: str | None = None,
    legend_labels: Mapping[str, str] | None = None,
    panel_labels: Mapping[str, str] | None = None,
    alt_text: str | None = None,
    methods_note: str | None = None,
    report_section_headings: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    visual_qc: VisualQcSpec | None = None,
    require_methods_note: bool = False,
) -> TabularAssociationVisualizationPlan:
    """Plan a pathless no-write visualization/report handoff."""

    payload = _tabular_association_visualization_payload(
        association_rows=association_rows,
        publication_table_rows=publication_table_rows,
        publication_display_rows=publication_display_rows,
        publication_machine_rows=publication_machine_rows,
        multiplicity_rows=multiplicity_rows,
        qc_rows=qc_rows,
        missingness_rows=missingness_rows,
        provenance_rows=provenance_rows,
        workflow_id=workflow_id,
        source_rowset_names=source_rowset_names,
        estimate_field=estimate_field,
        lower_field=lower_field,
        upper_field=upper_field,
        label_field=label_field,
        group_field=group_field,
        data_label_field=data_label_field,
        figure_id=figure_id,
        figure_name=figure_name,
        report_id=report_id,
        report_name=report_name,
        text_spec=text_spec,
        title=title,
        subtitle=subtitle,
        caption=caption,
        footnote=footnote,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        legend_title=legend_title,
        legend_labels=legend_labels,
        panel_labels=panel_labels,
        alt_text=alt_text,
        methods_note=methods_note,
        report_section_headings=report_section_headings,
        metadata=metadata,
        visual_qc=visual_qc,
        require_methods_note=require_methods_note,
        executed=False,
        plan_only=True,
    )
    return TabularAssociationVisualizationPlan(**payload)


def build_tabular_association_visualization_handoff(
    association_rows: Iterable[Any] | None = None,
    *,
    publication_table_rows: Iterable[Any] | None = None,
    publication_display_rows: Iterable[Any] | None = None,
    publication_machine_rows: Iterable[Any] | None = None,
    multiplicity_rows: Iterable[Any] | None = None,
    qc_rows: Iterable[Any] | None = None,
    missingness_rows: Iterable[Any] | None = None,
    provenance_rows: Iterable[Any] | None = None,
    workflow_id: str | None = None,
    source_rowset_names: Mapping[str, str] | None = None,
    estimate_field: str | None = None,
    lower_field: str | None = None,
    upper_field: str | None = None,
    label_field: str | None = None,
    group_field: str | None = None,
    data_label_field: str | None = None,
    figure_id: str = "association-summary-figure",
    figure_name: str | None = None,
    report_id: str = "association-summary-report",
    report_name: str | None = None,
    text_spec: AssociationVisualizationTextSpec | Mapping[str, Any] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    footnote: str | None = None,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    legend_title: str | None = None,
    legend_labels: Mapping[str, str] | None = None,
    panel_labels: Mapping[str, str] | None = None,
    alt_text: str | None = None,
    methods_note: str | None = None,
    report_section_headings: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    visual_qc: VisualQcSpec | None = None,
    require_methods_note: bool = False,
) -> TabularAssociationVisualizationResult:
    """Build in-memory pathless visualization/report handoff rows and specs."""

    payload = _tabular_association_visualization_payload(
        association_rows=association_rows,
        publication_table_rows=publication_table_rows,
        publication_display_rows=publication_display_rows,
        publication_machine_rows=publication_machine_rows,
        multiplicity_rows=multiplicity_rows,
        qc_rows=qc_rows,
        missingness_rows=missingness_rows,
        provenance_rows=provenance_rows,
        workflow_id=workflow_id,
        source_rowset_names=source_rowset_names,
        estimate_field=estimate_field,
        lower_field=lower_field,
        upper_field=upper_field,
        label_field=label_field,
        group_field=group_field,
        data_label_field=data_label_field,
        figure_id=figure_id,
        figure_name=figure_name,
        report_id=report_id,
        report_name=report_name,
        text_spec=text_spec,
        title=title,
        subtitle=subtitle,
        caption=caption,
        footnote=footnote,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        legend_title=legend_title,
        legend_labels=legend_labels,
        panel_labels=panel_labels,
        alt_text=alt_text,
        methods_note=methods_note,
        report_section_headings=report_section_headings,
        metadata=metadata,
        visual_qc=visual_qc,
        require_methods_note=require_methods_note,
        executed=True,
        plan_only=False,
    )
    return TabularAssociationVisualizationResult(**payload)


def _tabular_association_visualization_payload(
    *,
    association_rows: Iterable[Any] | None,
    publication_table_rows: Iterable[Any] | None,
    publication_display_rows: Iterable[Any] | None,
    publication_machine_rows: Iterable[Any] | None,
    multiplicity_rows: Iterable[Any] | None,
    qc_rows: Iterable[Any] | None,
    missingness_rows: Iterable[Any] | None,
    provenance_rows: Iterable[Any] | None,
    workflow_id: str | None,
    source_rowset_names: Mapping[str, str] | None,
    estimate_field: str | None,
    lower_field: str | None,
    upper_field: str | None,
    label_field: str | None,
    group_field: str | None,
    data_label_field: str | None,
    figure_id: str,
    figure_name: str | None,
    report_id: str,
    report_name: str | None,
    text_spec: AssociationVisualizationTextSpec | Mapping[str, Any] | None,
    title: str | None,
    subtitle: str | None,
    caption: str | None,
    footnote: str | None,
    x_axis_label: str | None,
    y_axis_label: str | None,
    legend_title: str | None,
    legend_labels: Mapping[str, str] | None,
    panel_labels: Mapping[str, str] | None,
    alt_text: str | None,
    methods_note: str | None,
    report_section_headings: Mapping[str, str] | None,
    metadata: Mapping[str, Any] | None,
    visual_qc: VisualQcSpec | None,
    require_methods_note: bool,
    executed: bool,
    plan_only: bool,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    source_names = _source_rowset_names(source_rowset_names)
    normalized_rowsets: dict[str, tuple[dict[str, Any], ...]] = {name: () for name in _ROWSET_NAMES}
    resolved_workflow_id = _optional_text(workflow_id) or "unresolved-workflow"
    clean_figure_id = _non_empty_text(figure_id, field_name="figure_id")
    clean_report_id = _non_empty_text(report_id, field_name="report_id")

    try:
        normalized_rowsets = {
            _PRIMARY_ASSOCIATION_ROWSET: _coerce_rows(association_rows, field_name=_PRIMARY_ASSOCIATION_ROWSET),
            _PRIMARY_PUBLICATION_ROWSET: _coerce_rows(publication_table_rows, field_name=_PRIMARY_PUBLICATION_ROWSET),
            "publication_display_rows": _coerce_rows(publication_display_rows, field_name="publication_display_rows"),
            "publication_machine_rows": _coerce_rows(publication_machine_rows, field_name="publication_machine_rows"),
            "multiplicity_rows": _coerce_rows(multiplicity_rows, field_name="multiplicity_rows"),
            "qc_rows": _coerce_rows(qc_rows, field_name="qc_rows"),
            "missingness_rows": _coerce_rows(missingness_rows, field_name="missingness_rows"),
            "provenance_rows": _coerce_rows(provenance_rows, field_name="provenance_rows"),
        }
        resolved_workflow_id, workflow_warnings, workflow_errors = _resolve_workflow_id(
            explicit_workflow_id=workflow_id,
            rowsets=normalized_rowsets,
        )
        warnings.extend(workflow_warnings)
        errors.extend(workflow_errors)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))

    primary_rowset_name = (
        _PRIMARY_PUBLICATION_ROWSET
        if normalized_rowsets[_PRIMARY_PUBLICATION_ROWSET]
        else _PRIMARY_ASSOCIATION_ROWSET
    )
    if not normalized_rowsets[primary_rowset_name]:
        warnings.append("No association_rows or publication_table_rows were supplied for visualization handoff.")

    input_counts = {name: len(rows) for name, rows in normalized_rowsets.items()}
    text = _coerce_text_spec(
        text_spec,
        title=title,
        subtitle=subtitle,
        caption=caption,
        footnote=footnote,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        legend_title=legend_title,
        legend_labels=legend_labels,
        panel_labels=panel_labels,
        alt_text=alt_text,
        methods_note=methods_note,
        report_section_headings=report_section_headings,
        metadata=metadata,
    )
    metadata_mapping = _json_safe_mapping({**text.metadata, **(metadata or {})})
    primary_rows = normalized_rowsets[primary_rowset_name]
    dataset_rows, dataset_warnings = _association_dataset_rows(
        workflow_id=resolved_workflow_id,
        rows=primary_rows,
        source_rowset_name=source_names[primary_rowset_name],
        estimate_field=estimate_field,
        lower_field=lower_field,
        upper_field=upper_field,
        label_field=label_field,
        group_field=group_field,
        data_label_field=data_label_field,
        executed=executed,
        plan_only=plan_only,
    )
    warnings.extend(dataset_warnings)

    has_interval = any(row.has_interval for row in dataset_rows)
    has_groups = any(_optional_text(row.group_id) is not None for row in dataset_rows)
    has_data_labels = any(_optional_text(row.data_label) is not None for row in dataset_rows)
    plot_spec = _point_interval_spec(
        text=text,
        figure_id=clean_figure_id,
        has_interval=has_interval,
        has_groups=has_groups,
        has_data_labels=has_data_labels,
        metadata=metadata_mapping,
        visual_qc=visual_qc,
    )
    visual_qc_rows = _visual_qc_rows(
        dataset_rows,
        plot_spec=plot_spec,
        metadata=metadata_mapping,
        report_id=clean_report_id,
        report_text=text,
        require_methods_note=require_methods_note,
    )
    for qc_row in visual_qc_rows:
        if qc_row.severity == "error":
            errors.append(qc_row.message)
        elif qc_row.severity == "warning":
            warnings.append(qc_row.message)

    publication_display_summary_rows = normalized_rowsets["publication_display_rows"]
    publication_machine_summary_rows = normalized_rowsets["publication_machine_rows"]
    qc_summary_rows = normalized_rowsets["qc_rows"]
    missingness_summary_rows = normalized_rowsets["missingness_rows"]
    multiplicity_summary_rows = normalized_rowsets["multiplicity_rows"]

    report_spec = _report_spec(
        workflow_id=resolved_workflow_id,
        report_id=clean_report_id,
        plot_id=clean_figure_id,
        text=text,
        dataset_rows=dataset_rows,
        publication_display_rows=publication_display_summary_rows,
        publication_machine_rows=publication_machine_summary_rows,
        qc_rows=qc_summary_rows,
        missingness_rows=missingness_summary_rows,
        multiplicity_rows=multiplicity_summary_rows,
        provenance_rows=normalized_rowsets["provenance_rows"],
        metadata=metadata_mapping,
    )
    figure_specs = (
        AssociationVisualizationFigureSpec(
            workflow_id=resolved_workflow_id,
            figure_id=clean_figure_id,
            figure_name=figure_name or clean_figure_id,
            dataset_rowset_name="association_dataset_rows",
            plot_type="point_interval" if has_interval else "point",
            plot_spec=plot_spec,
            row_count=len(dataset_rows),
            has_interval=has_interval,
            executed=executed,
            plan_only=plan_only,
        ),
    )
    report_handoff_specs = (
        AssociationReportHandoffSpec(
            workflow_id=resolved_workflow_id,
            report_id=clean_report_id,
            report_name=report_name or clean_report_id,
            report_spec=report_spec,
            section_ids=tuple(getattr(section, "section_id", "") for section in report_spec.sections),
            section_count=len(report_spec.sections),
            executed=executed,
            plan_only=plan_only,
        ),
    )
    provenance_table_rows = _provenance_rows(
        workflow_id=resolved_workflow_id,
        supplied_rows=normalized_rowsets["provenance_rows"],
        input_counts=input_counts,
        output_counts={},
        planned_figure_names=(figure_specs[0].figure_name,),
        planned_report_names=(report_handoff_specs[0].report_name,),
        planned_output_names=_OUTPUT_ROWSET_NAMES,
        source_rowset_names=source_names,
        executed=executed,
        plan_only=plan_only,
        text=text,
    )
    status = "error" if errors else ("warning" if warnings else "ok")
    spec_rows = _spec_rows(
        workflow_id=resolved_workflow_id,
        dataset_rows=dataset_rows,
        figure_specs=figure_specs,
        report_handoff_specs=report_handoff_specs,
        status=status,
        warnings=warnings,
        errors=errors,
        executed=executed,
        plan_only=plan_only,
    )
    planned_row_counts = {
        "input_summary_rows": len(_ROWSET_NAMES),
        "association_dataset_rows": len(dataset_rows),
        "figure_specs": len(figure_specs),
        "report_handoff_specs": len(report_handoff_specs),
        "spec_rows": len(spec_rows),
        "visual_qc_rows": len(visual_qc_rows),
        "qc_summary_rows": len(qc_summary_rows),
        "missingness_summary_rows": len(missingness_summary_rows),
        "multiplicity_summary_rows": len(multiplicity_summary_rows),
        "provenance_table_rows": len(provenance_table_rows),
        "manifest_rows": len(_OUTPUT_ROWSET_NAMES),
    }
    provenance_table_rows = _provenance_rows(
        workflow_id=resolved_workflow_id,
        supplied_rows=normalized_rowsets["provenance_rows"],
        input_counts=input_counts,
        output_counts=planned_row_counts,
        planned_figure_names=(figure_specs[0].figure_name,),
        planned_report_names=(report_handoff_specs[0].report_name,),
        planned_output_names=_OUTPUT_ROWSET_NAMES,
        source_rowset_names=source_names,
        executed=executed,
        plan_only=plan_only,
        text=text,
    )
    planned_row_counts["provenance_table_rows"] = len(provenance_table_rows)
    report_spec = _report_spec(
        workflow_id=resolved_workflow_id,
        report_id=clean_report_id,
        plot_id=clean_figure_id,
        text=text,
        dataset_rows=dataset_rows,
        publication_display_rows=publication_display_summary_rows,
        publication_machine_rows=publication_machine_summary_rows,
        qc_rows=qc_summary_rows,
        missingness_rows=missingness_summary_rows,
        multiplicity_rows=multiplicity_summary_rows,
        provenance_rows=tuple(row.to_dict() for row in provenance_table_rows),
        metadata=metadata_mapping,
    )
    report_handoff_specs = (
        AssociationReportHandoffSpec(
            workflow_id=resolved_workflow_id,
            report_id=clean_report_id,
            report_name=report_name or clean_report_id,
            report_spec=report_spec,
            section_ids=tuple(getattr(section, "section_id", "") for section in report_spec.sections),
            section_count=len(report_spec.sections),
            executed=executed,
            plan_only=plan_only,
        ),
    )
    spec_rows = _spec_rows(
        workflow_id=resolved_workflow_id,
        dataset_rows=dataset_rows,
        figure_specs=figure_specs,
        report_handoff_specs=report_handoff_specs,
        status=status,
        warnings=warnings,
        errors=errors,
        executed=executed,
        plan_only=plan_only,
    )
    planned_row_counts["report_handoff_specs"] = len(report_handoff_specs)
    planned_row_counts["spec_rows"] = len(spec_rows)
    input_summary_rows = _input_summary_rows(
        workflow_id=resolved_workflow_id,
        rowsets=normalized_rowsets,
        source_rowset_names=source_names,
        primary_rowset_name=primary_rowset_name,
        status=status,
        warnings=warnings,
        errors=errors,
        executed=executed,
        plan_only=plan_only,
    )
    manifest_rows = _manifest_rows(
        workflow_id=resolved_workflow_id,
        row_counts=planned_row_counts,
        input_counts=input_counts,
        source_rowset_names=source_names,
        planned_figure_names=(figure_specs[0].figure_name,),
        planned_report_names=(report_handoff_specs[0].report_name,),
        planned_output_names=_OUTPUT_ROWSET_NAMES,
        executed=executed,
        plan_only=plan_only,
        status=status,
        warnings=warnings,
        errors=errors,
    )
    planned_row_counts["manifest_rows"] = len(manifest_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "tabular_association_schema_version": TABULAR_ASSOCIATION_SCHEMA_VERSION,
        "visualization_handoff_schema_version": TABULAR_ASSOCIATION_VISUALIZATION_HANDOFF_VERSION,
        "workflow_id": resolved_workflow_id,
        "valid": not errors,
        "executed": executed,
        "plan_only": plan_only,
        "will_write": False,
        "output_written": False,
        "no_output_written": True,
        "output_paths_written": (),
        "will_render": False,
        "rendered": False,
        "figure_rendered": False,
        "report_rendered": False,
        "runtime_backend": RUNTIME_BACKEND_RECORDS,
        "primary_rowset_name": primary_rowset_name,
        "status": status,
        "warnings": tuple(_unique_text(warnings)),
        "errors": tuple(_unique_text(errors)),
        "input_summary_rows": input_summary_rows,
        "association_dataset_rows": dataset_rows,
        "publication_display_rows": publication_display_summary_rows,
        "publication_machine_rows": publication_machine_summary_rows,
        "qc_summary_rows": qc_summary_rows,
        "missingness_summary_rows": missingness_summary_rows,
        "multiplicity_summary_rows": multiplicity_summary_rows,
        "provenance_table_rows": provenance_table_rows,
        "figure_specs": figure_specs,
        "report_handoff_specs": report_handoff_specs,
        "spec_rows": spec_rows,
        "visual_qc_rows": visual_qc_rows,
        "manifest_rows": manifest_rows,
        "planned_row_counts": planned_row_counts,
        "planned_figure_names": (figure_specs[0].figure_name,),
        "planned_report_names": (report_handoff_specs[0].report_name,),
        "planned_output_names": _OUTPUT_ROWSET_NAMES,
    }


def _association_dataset_rows(
    *,
    workflow_id: str,
    rows: Sequence[Mapping[str, Any]],
    source_rowset_name: str,
    estimate_field: str | None,
    lower_field: str | None,
    upper_field: str | None,
    label_field: str | None,
    group_field: str | None,
    data_label_field: str | None,
    executed: bool,
    plan_only: bool,
) -> tuple[tuple[AssociationVisualizationDatasetRow, ...], tuple[str, ...]]:
    dataset_rows: list[AssociationVisualizationDatasetRow] = []
    warnings: list[str] = []
    estimate_key = _optional_text(estimate_field) or "statistic_value"
    label_key = _optional_text(label_field)
    group_key = _optional_text(group_field)
    data_label_key = _optional_text(data_label_field)
    for position, row in enumerate(rows):
        input_row_index = _input_row_index(row, position=position)
        if _int_or_none(row.get("input_row_index")) is not None and _int_or_none(row.get("input_row_index")) < 0:
            warnings.append(
                f"{source_rowset_name}[{position}] input_row_index is negative; using positional index."
            )
        lower_value, upper_value = _interval_values(row, lower_field=lower_field, upper_field=upper_field)
        has_interval = _has_value(lower_value) and _has_value(upper_value)
        result_row_id = _optional_text(row.get("result_row_id"))
        pair_id = _optional_text(row.get("pair_id"))
        dataset_row_id = result_row_id or pair_id or f"association-visualization-row-{input_row_index}"
        used_fields = {
            *_IDENTIFIER_FIELDS,
            estimate_key,
            *(field for field in (lower_field, upper_field, label_key, group_key, data_label_key) if field),
            "label",
            "display_label",
            "outcome_label",
            "predictor_label",
            *[field for pair in _COMMON_INTERVAL_FIELD_PAIRS for field in pair],
            "data_label",
        }
        dataset_rows.append(
            AssociationVisualizationDatasetRow(
                workflow_id=_optional_text(row.get("workflow_id")) or workflow_id,
                dataset_row_id=dataset_row_id,
                source_rowset_name=source_rowset_name,
                source_row_index=position,
                input_row_index=input_row_index,
                label=_row_label(row, position=position, label_field=label_key),
                estimate=row.get(estimate_key),
                lower=lower_value if has_interval else None,
                upper=upper_value if has_interval else None,
                has_interval=has_interval,
                group_id=_optional_text(row.get(group_key)) if group_key is not None else None,
                data_label=_optional_text(row.get(data_label_key or "data_label")),
                method_id=_optional_text(row.get("method_id")),
                method_kind=_optional_text(row.get("method_kind")),
                method_name=_optional_text(row.get("method_name")),
                family_id=_optional_text(row.get("family_id")),
                source_id=_optional_text(row.get("source_id")),
                outcome_id=_optional_text(row.get("outcome_id")),
                predictor_id=_optional_text(row.get("predictor_id")),
                covariate_ids=_text_tuple(row.get("covariate_ids", ())),
                statistic_name=_optional_text(row.get("statistic_name")),
                statistic_value=row.get("statistic_value"),
                p_value=row.get("p_value"),
                q_value=row.get("q_value"),
                status=_optional_text(row.get("status")) or "ok",
                warnings=_messages(row.get("warnings")),
                errors=_messages(row.get("errors")),
                result_row_id=result_row_id,
                pair_id=pair_id,
                extra_fields=_extra_fields(row, used_fields=used_fields),
                executed=executed,
                plan_only=plan_only,
            )
        )
    return tuple(dataset_rows), tuple(_unique_text(warnings))


def _point_interval_spec(
    *,
    text: AssociationVisualizationTextSpec,
    figure_id: str,
    has_interval: bool,
    has_groups: bool,
    has_data_labels: bool,
    metadata: Mapping[str, Any],
    visual_qc: VisualQcSpec | None,
) -> PointIntervalPlotSpec:
    spec = build_point_interval_plot_spec(
        label_column="label",
        estimate_column="estimate",
        lower_column="lower" if has_interval else None,
        upper_column="upper" if has_interval else None,
        group_column="group_id" if has_groups else None,
        data_label_column="data_label" if has_data_labels else None,
        plot_id=figure_id,
        title=text.title,
        subtitle=text.subtitle,
        caption=text.caption,
        footnote=text.footnote,
        alt_text=text.alt_text,
        x_label=text.x_axis_label,
        y_label=text.y_axis_label,
        legend_title=text.legend_title,
        legend_label_mapping=text.legend_labels,
        visual_qc=visual_qc,
        metadata={
            "tabular_association_schema_version": TABULAR_ASSOCIATION_SCHEMA_VERSION,
            "visualization_handoff_schema_version": TABULAR_ASSOCIATION_VISUALIZATION_HANDOFF_VERSION,
            **metadata,
        },
    )
    if text.methods_note is None and not text.panel_labels:
        return spec
    return replace(
        spec,
        text=FigureTextSpec(
            title=spec.text.title,
            subtitle=spec.text.subtitle,
            footnote=spec.text.footnote,
            alt_text=spec.text.alt_text,
            methods_note=text.methods_note,
            panel_labels=text.panel_labels,
            metadata=spec.text.metadata,
        ),
    )


def _report_spec(
    *,
    workflow_id: str,
    report_id: str,
    plot_id: str,
    text: AssociationVisualizationTextSpec,
    dataset_rows: Sequence[AssociationVisualizationDatasetRow],
    publication_display_rows: Sequence[Mapping[str, Any]],
    publication_machine_rows: Sequence[Mapping[str, Any]],
    qc_rows: Sequence[Mapping[str, Any]],
    missingness_rows: Sequence[Mapping[str, Any]],
    multiplicity_rows: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> ReportSpec:
    headings = _report_section_headings(text)
    dataset_dicts = tuple(row.to_dict() for row in dataset_rows)
    sections: list[TableSectionSpec | FigureSectionSpec] = [
        FigureSectionSpec(
            section_id="association_figure",
            heading=headings["association_figure"],
            plot_id=plot_id,
            figure_path=None,
            caption=text.caption,
            alt_text=text.alt_text,
            metadata={"workflow_id": workflow_id, "rowset_name": "association_dataset_rows"},
        ),
        TableSectionSpec(
            section_id="association_summary",
            heading=headings["association_summary"],
            rows=dataset_dicts,
            columns=tuple(column for column in _DATASET_COLUMNS if column in _columns_for_rows(dataset_dicts)),
            caption=text.caption,
            metadata={"workflow_id": workflow_id, "rowset_name": "association_dataset_rows"},
        ),
        TableSectionSpec(
            section_id="qc_summary",
            heading=headings["qc_summary"],
            rows=qc_rows,
            metadata={"workflow_id": workflow_id, "rowset_name": "qc_summary_rows"},
        ),
        TableSectionSpec(
            section_id="missingness_summary",
            heading=headings["missingness_summary"],
            rows=missingness_rows,
            metadata={"workflow_id": workflow_id, "rowset_name": "missingness_summary_rows"},
        ),
        TableSectionSpec(
            section_id="multiplicity_summary",
            heading=headings["multiplicity_summary"],
            rows=multiplicity_rows,
            metadata={"workflow_id": workflow_id, "rowset_name": "multiplicity_summary_rows"},
        ),
        TableSectionSpec(
            section_id="provenance",
            heading=headings["provenance"],
            rows=provenance_rows,
            metadata={"workflow_id": workflow_id, "rowset_name": "provenance_rows"},
        ),
    ]
    if publication_display_rows:
        sections.append(
            TableSectionSpec(
                section_id="publication_display",
                heading=headings["publication_display"],
                rows=publication_display_rows,
                metadata={"workflow_id": workflow_id, "rowset_name": "publication_display_rows"},
            )
        )
    if publication_machine_rows:
        sections.append(
            TableSectionSpec(
                section_id="publication_machine",
                heading=headings["publication_machine"],
                rows=publication_machine_rows,
                metadata={"workflow_id": workflow_id, "rowset_name": "publication_machine_rows"},
            )
        )
    return ReportSpec(
        report_id=report_id,
        title=text.report_title or text.title,
        subtitle=text.report_subtitle or text.subtitle,
        caption=text.report_caption,
        footnote=text.report_footnote or text.footnote,
        alt_text=text.alt_text,
        methods_note=text.methods_note,
        sections=tuple(sections),
        metadata={
            "workflow_id": workflow_id,
            "tabular_association_schema_version": TABULAR_ASSOCIATION_SCHEMA_VERSION,
            "visualization_handoff_schema_version": TABULAR_ASSOCIATION_VISUALIZATION_HANDOFF_VERSION,
            **metadata,
        },
        include_artifact_list=False,
        include_visual_qc=True,
    )


def _visual_qc_rows(
    rows: Sequence[AssociationVisualizationDatasetRow],
    *,
    plot_spec: PointIntervalPlotSpec,
    metadata: Mapping[str, Any],
    report_id: str,
    report_text: AssociationVisualizationTextSpec,
    require_methods_note: bool,
) -> tuple[VisualQcRow, ...]:
    qc_rows: list[VisualQcRow] = []
    row_dicts = tuple(row.to_dict() for row in rows)
    qc_rows.extend(
        _invalid_template_qc_rows(
            artifact_id=plot_spec.plot_id,
            payload={
                "text": plot_spec.text.to_dict(),
                "axes": plot_spec.axes.to_dict(),
                "legend": plot_spec.legend.to_dict(),
                "caption": plot_spec.caption.to_dict(),
            },
        )
    )
    try:
        qc_rows.extend(build_visual_layout_qc_rows(row_dicts, plot_spec, metadata=metadata))
    except (TypeError, ValueError) as exc:
        qc_rows.append(
            VisualQcRow(
                artifact_id=plot_spec.plot_id,
                check_id="visual_qc_failed",
                severity="error",
                status="error",
                scope="visual_qc",
                message=str(exc),
            )
        )
    qc_rows.extend(
        _report_template_qc_rows(
            report_id=report_id,
            report_text=report_text,
            metadata=metadata,
        )
    )
    if require_methods_note and not _has_text(report_text.methods_note):
        qc_rows.append(
            VisualQcRow(
                artifact_id=report_id,
                check_id="missing_methods_note",
                severity="error",
                status="error",
                scope="text",
                message="Methods/provenance note is required by visual handoff policy.",
            )
    )
    return tuple(qc_rows)


def _invalid_template_qc_rows(*, artifact_id: str, payload: Mapping[str, Any]) -> tuple[VisualQcRow, ...]:
    qc_rows: list[VisualQcRow] = []
    for field_path, template in _iter_template_strings(payload):
        try:
            tuple(Formatter().parse(template))
        except ValueError as exc:
            qc_rows.append(
                VisualQcRow(
                    artifact_id=artifact_id,
                    check_id="invalid_template_field",
                    severity="error",
                    status="error",
                    scope="text",
                    message=f"Template in {field_path!r} is invalid: {exc}",
                    metric=field_path,
                )
            )
    return tuple(qc_rows)


def _report_template_qc_rows(
    *,
    report_id: str,
    report_text: AssociationVisualizationTextSpec,
    metadata: Mapping[str, Any],
) -> tuple[VisualQcRow, ...]:
    qc_rows: list[VisualQcRow] = []
    report_payload = {
        "report_title": report_text.report_title,
        "report_subtitle": report_text.report_subtitle,
        "report_caption": report_text.report_caption,
        "report_footnote": report_text.report_footnote,
        "methods_note": report_text.methods_note,
        "report_section_headings": report_text.report_section_headings,
    }
    for field_path, template in _iter_template_strings(report_payload):
        for check_id, message in _template_errors(template, metadata, field_path=field_path):
            qc_rows.append(
                VisualQcRow(
                    artifact_id=report_id,
                    check_id=check_id,
                    severity="error",
                    status="error",
                    scope="text",
                    message=message,
                    metric=field_path,
                )
            )
    return tuple(qc_rows)


def _input_summary_rows(
    *,
    workflow_id: str,
    rowsets: Mapping[str, Sequence[Mapping[str, Any]]],
    source_rowset_names: Mapping[str, str],
    primary_rowset_name: str,
    status: str,
    warnings: Sequence[str],
    errors: Sequence[str],
    executed: bool,
    plan_only: bool,
) -> tuple[AssociationVisualizationInputSummaryRow, ...]:
    return tuple(
        AssociationVisualizationInputSummaryRow(
            workflow_id=workflow_id,
            rowset_name=rowset_name,
            source_rowset_name=source_rowset_names[rowset_name],
            row_count=len(rowsets[rowset_name]),
            normalized_row_count=len(rowsets[rowset_name]),
            selected_for_visualization=rowset_name == primary_rowset_name,
            status=status,
            warnings=warnings if status == "warning" else (),
            errors=errors if status == "error" else (),
            executed=executed,
            plan_only=plan_only,
        )
        for rowset_name in _ROWSET_NAMES
    )


def _spec_rows(
    *,
    workflow_id: str,
    dataset_rows: Sequence[AssociationVisualizationDatasetRow],
    figure_specs: Sequence[AssociationVisualizationFigureSpec],
    report_handoff_specs: Sequence[AssociationReportHandoffSpec],
    status: str,
    warnings: Sequence[str],
    errors: Sequence[str],
    executed: bool,
    plan_only: bool,
) -> tuple[AssociationVisualizationSpecRow, ...]:
    dataset_dicts = tuple(row.to_dict() for row in dataset_rows)
    rows: list[AssociationVisualizationSpecRow] = [
        AssociationVisualizationSpecRow(
            workflow_id=workflow_id,
            spec_id="association_dataset",
            spec_type="dataset",
            rowset_name="association_dataset_rows",
            artifact_id=None,
            row_count=len(dataset_rows),
            columns=_columns_for_rows(dataset_dicts),
            spec_payload={
                "runtime_backend": RUNTIME_BACKEND_RECORDS,
                "label_column": "label",
                "estimate_column": "estimate",
                "lower_column": "lower" if any(row.has_interval for row in dataset_rows) else None,
                "upper_column": "upper" if any(row.has_interval for row in dataset_rows) else None,
            },
            status=status,
            warnings=warnings,
            errors=errors,
            executed=executed,
            plan_only=plan_only,
        )
    ]
    rows.extend(
        AssociationVisualizationSpecRow(
            workflow_id=workflow_id,
            spec_id=figure_spec.figure_id,
            spec_type="figure",
            rowset_name="figure_specs",
            artifact_id=figure_spec.figure_id,
            row_count=figure_spec.row_count,
            columns=figure_spec.plot_spec.source_columns()
            if isinstance(figure_spec.plot_spec, PointIntervalPlotSpec)
            else (),
            spec_payload=figure_spec.to_dict(),
            status=status,
            warnings=warnings,
            errors=errors,
            executed=executed,
            plan_only=plan_only,
        )
        for figure_spec in figure_specs
    )
    rows.extend(
        AssociationVisualizationSpecRow(
            workflow_id=workflow_id,
            spec_id=report_spec.report_id,
            spec_type="report",
            rowset_name="report_handoff_specs",
            artifact_id=report_spec.report_id,
            row_count=report_spec.section_count,
            columns=report_spec.section_ids,
            spec_payload=report_spec.to_dict(),
            status=status,
            warnings=warnings,
            errors=errors,
            executed=executed,
            plan_only=plan_only,
        )
        for report_spec in report_handoff_specs
    )
    return tuple(rows)


def _provenance_rows(
    *,
    workflow_id: str,
    supplied_rows: Sequence[Mapping[str, Any]],
    input_counts: Mapping[str, int],
    output_counts: Mapping[str, int],
    planned_figure_names: Sequence[str],
    planned_report_names: Sequence[str],
    planned_output_names: Sequence[str],
    source_rowset_names: Mapping[str, str],
    executed: bool,
    plan_only: bool,
    text: AssociationVisualizationTextSpec,
) -> tuple[AssociationVisualizationProvenanceRow, ...]:
    rows: list[AssociationVisualizationProvenanceRow] = []
    for index, row in enumerate(supplied_rows):
        key = _optional_text(row.get("key")) or f"supplied_provenance_row_{index}"
        value = row.get("value") if "value" in row else row
        rows.append(
            AssociationVisualizationProvenanceRow(
                workflow_id=_optional_text(row.get("workflow_id")) or workflow_id,
                key=key,
                value=value,
                source=_optional_text(row.get("source")) or "supplied_visualization_handoff_provenance",
                input_row_index=index,
            )
        )
    generated_values: tuple[tuple[str, Any], ...] = (
        ("tabular_association_schema_version", TABULAR_ASSOCIATION_SCHEMA_VERSION),
        ("visualization_handoff_schema_version", TABULAR_ASSOCIATION_VISUALIZATION_HANDOFF_VERSION),
        ("research_viz_schema_version", SCHEMA_VERSION),
        ("workflow_id", workflow_id),
        ("input_row_counts", dict(input_counts)),
        ("output_row_counts", dict(output_counts)),
        ("planned_figure_names", tuple(planned_figure_names)),
        ("planned_report_names", tuple(planned_report_names)),
        ("planned_output_names", tuple(planned_output_names)),
        ("source_rowset_names", dict(source_rowset_names)),
        ("runtime_backend", RUNTIME_BACKEND_RECORDS),
        ("methods_note", text.methods_note),
        ("executed", executed),
        ("plan_only", plan_only),
        ("will_render", False),
        ("rendered", False),
        ("figure_rendered", False),
        ("report_rendered", False),
        ("will_write", False),
        ("output_written", False),
        ("no_output_written", True),
        ("output_paths_written", ()),
    )
    rows.extend(
        AssociationVisualizationProvenanceRow(workflow_id=workflow_id, key=key, value=value)
        for key, value in generated_values
    )
    return tuple(rows)


def _manifest_rows(
    *,
    workflow_id: str,
    row_counts: Mapping[str, int],
    input_counts: Mapping[str, int],
    source_rowset_names: Mapping[str, str],
    planned_figure_names: Sequence[str],
    planned_report_names: Sequence[str],
    planned_output_names: Sequence[str],
    executed: bool,
    plan_only: bool,
    status: str,
    warnings: Sequence[str],
    errors: Sequence[str],
) -> tuple[AssociationVisualizationManifestRow, ...]:
    return tuple(
        AssociationVisualizationManifestRow(
            workflow_id=workflow_id,
            rowset_name=rowset_name,
            row_count=row_counts.get(rowset_name, 0),
            tabular_association_schema_version=TABULAR_ASSOCIATION_SCHEMA_VERSION,
            visualization_handoff_schema_version=TABULAR_ASSOCIATION_VISUALIZATION_HANDOFF_VERSION,
            research_viz_schema_version=SCHEMA_VERSION,
            input_row_counts=input_counts,
            output_row_counts=row_counts,
            planned_figure_names=planned_figure_names,
            planned_report_names=planned_report_names,
            planned_output_names=planned_output_names,
            source_rowset_names=source_rowset_names,
            runtime_backend=RUNTIME_BACKEND_RECORDS,
            executed=executed,
            plan_only=plan_only,
            will_write=False,
            output_written=False,
            no_output_written=True,
            output_paths_written=(),
            will_render=False,
            rendered=False,
            status=status,
            warnings=warnings,
            errors=errors,
        )
        for rowset_name in _OUTPUT_ROWSET_NAMES
    )


def _report_section_headings(text: AssociationVisualizationTextSpec) -> dict[str, str]:
    headings = {
        "association_figure": "Association Figure",
        "association_summary": "Association Summary",
        "qc_summary": "QC Summary",
        "missingness_summary": "Missingness Summary",
        "multiplicity_summary": "Multiplicity Summary",
        "provenance": "Provenance",
        "publication_display": "Publication Display Rows",
        "publication_machine": "Publication Machine Rows",
    }
    headings.update(text.report_section_headings)
    return headings


def _interval_values(
    row: Mapping[str, Any],
    *,
    lower_field: str | None,
    upper_field: str | None,
) -> tuple[Any, Any]:
    if lower_field is not None or upper_field is not None:
        lower_key = _optional_text(lower_field)
        upper_key = _optional_text(upper_field)
        return (row.get(lower_key) if lower_key else None, row.get(upper_key) if upper_key else None)
    for lower_key, upper_key in _COMMON_INTERVAL_FIELD_PAIRS:
        if lower_key in row and upper_key in row:
            return row.get(lower_key), row.get(upper_key)
    return None, None


def _row_label(row: Mapping[str, Any], *, position: int, label_field: str | None) -> str:
    if label_field is not None and _has_text(row.get(label_field)):
        return _display_scalar(row.get(label_field))
    for key in ("label", "display_label"):
        if _has_text(row.get(key)):
            return _display_scalar(row.get(key))
    outcome = _optional_text(row.get("outcome_label")) or _optional_text(row.get("outcome_id"))
    predictor = _optional_text(row.get("predictor_label")) or _optional_text(row.get("predictor_id"))
    if outcome and predictor:
        return f"{outcome} / {predictor}"
    if outcome:
        return outcome
    if predictor:
        return predictor
    for key in ("result_row_id", "pair_id"):
        if _has_text(row.get(key)):
            return _display_scalar(row.get(key))
    return f"row-{position}"


def _coerce_text_spec(
    text_spec: AssociationVisualizationTextSpec | Mapping[str, Any] | None,
    *,
    title: str | None,
    subtitle: str | None,
    caption: str | None,
    footnote: str | None,
    x_axis_label: str | None,
    y_axis_label: str | None,
    legend_title: str | None,
    legend_labels: Mapping[str, str] | None,
    panel_labels: Mapping[str, str] | None,
    alt_text: str | None,
    methods_note: str | None,
    report_section_headings: Mapping[str, str] | None,
    metadata: Mapping[str, Any] | None,
) -> AssociationVisualizationTextSpec:
    if text_spec is None:
        spec = AssociationVisualizationTextSpec()
    elif isinstance(text_spec, AssociationVisualizationTextSpec):
        spec = text_spec
    else:
        mapping = _as_mapping(text_spec, field_name="text_spec")
        spec = AssociationVisualizationTextSpec(**mapping)
    updates: dict[str, Any] = {}
    for key, value in (
        ("title", title),
        ("subtitle", subtitle),
        ("caption", caption),
        ("footnote", footnote),
        ("x_axis_label", x_axis_label),
        ("y_axis_label", y_axis_label),
        ("legend_title", legend_title),
        ("alt_text", alt_text),
        ("methods_note", methods_note),
    ):
        if value is not None:
            updates[key] = value
    if legend_labels is not None:
        updates["legend_labels"] = legend_labels
    if panel_labels is not None:
        updates["panel_labels"] = panel_labels
    if report_section_headings is not None:
        updates["report_section_headings"] = report_section_headings
    if metadata is not None:
        updates["metadata"] = {**spec.metadata, **metadata}
    return replace(spec, **updates) if updates else spec


def _source_rowset_names(source_rowset_names: Mapping[str, str] | None) -> dict[str, str]:
    names = dict(_DEFAULT_ROWSET_NAMES)
    if source_rowset_names is not None:
        for key, value in source_rowset_names.items():
            normalized_key = _non_empty_text(key, field_name="source_rowset_names key")
            if normalized_key in names:
                names[normalized_key] = _non_empty_text(value, field_name=f"source_rowset_names[{normalized_key}]")
    return names


def _coerce_rows(value: Iterable[Any] | None, *, field_name: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{field_name} must be a sequence of mapping/dataclass rows, not a single row.")
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable sequence of mapping/dataclass rows.") from exc
    return tuple(_row_mapping(row, field_name=f"{field_name}[{index}]") for index, row in enumerate(rows))


def _row_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    mapping = _as_mapping(value, field_name=field_name)
    return {str(key): _json_safe(item) for key, item in mapping.items()}


def _resolve_workflow_id(
    *,
    explicit_workflow_id: str | None,
    rowsets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    explicit = _optional_text(explicit_workflow_id)
    found_ids = _workflow_ids(rowsets)
    if explicit is not None:
        warnings = ()
        if found_ids and found_ids != {explicit}:
            warnings = ("Explicit workflow_id overrides workflow_id values found in supplied visualization rows.",)
        return explicit, warnings, ()
    if len(found_ids) == 1:
        return next(iter(found_ids)), (), ()
    if len(found_ids) > 1:
        return (
            "unresolved-workflow",
            (),
            ("Multiple workflow_id values were found in supplied visualization rows: " + ", ".join(sorted(found_ids)),),
        )
    return "unresolved-workflow", ("No workflow_id was supplied or found in visualization rows.",), ()


def _workflow_ids(rowsets: Mapping[str, Sequence[Mapping[str, Any]]]) -> set[str]:
    workflow_ids: set[str] = set()
    for rows in rowsets.values():
        for row in rows:
            row_workflow_id = _optional_text(row.get("workflow_id"))
            if row_workflow_id is not None:
                workflow_ids.add(row_workflow_id)
            if _optional_text(row.get("key")) == "workflow_id":
                provenance_workflow_id = _optional_text(row.get("value"))
                if provenance_workflow_id is not None:
                    workflow_ids.add(provenance_workflow_id)
    return workflow_ids


def _input_row_index(row: Mapping[str, Any], *, position: int) -> int:
    supplied = _int_or_none(row.get("input_row_index"))
    if supplied is not None and supplied >= 0:
        return supplied
    return position


def _extra_fields(row: Mapping[str, Any], *, used_fields: set[str]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in row.items() if str(key) not in used_fields}


def _iter_template_strings(payload: Any, *, prefix: str = "") -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if isinstance(value, str):
                found.append((path, value))
            else:
                found.extend(_iter_template_strings(value, prefix=path))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for index, value in enumerate(payload):
            found.extend(_iter_template_strings(value, prefix=f"{prefix}[{index}]"))
    return tuple(found)


def _template_errors(template: str, metadata: Mapping[str, Any], *, field_path: str) -> tuple[tuple[str, str], ...]:
    formatter = Formatter()
    errors: list[tuple[str, str]] = []
    try:
        parsed = tuple(formatter.parse(template))
    except ValueError as exc:
        return (("invalid_template_field", f"Template in {field_path!r} is invalid: {exc}"),)
    for _, raw_name, format_spec, _ in parsed:
        if raw_name is None:
            continue
        name = raw_name
        if "." in name or "[" in name or "]" in name:
            errors.append(
                (
                    "missing_template_field",
                    f"Template field {name!r} in {field_path!r} is not a simple metadata key.",
                )
            )
        elif name not in metadata:
            errors.append(
                (
                    "missing_template_field",
                    f"Template field {name!r} in {field_path!r} is missing from metadata.",
                )
            )
        elif "{" in format_spec or "}" in format_spec:
            errors.append(
                (
                    "missing_template_field",
                    f"Template format spec for {name!r} in {field_path!r} must not contain nested fields.",
                )
            )
    return tuple(errors)


def _normalize_plan_result(instance: object, *, executed: bool, plan_only: bool) -> None:
    object.__setattr__(instance, "schema_version", _non_empty_text(instance.schema_version, field_name="schema_version"))
    object.__setattr__(
        instance,
        "tabular_association_schema_version",
        _non_empty_text(instance.tabular_association_schema_version, field_name="tabular_association_schema_version"),
    )
    object.__setattr__(
        instance,
        "visualization_handoff_schema_version",
        _non_empty_text(
            instance.visualization_handoff_schema_version,
            field_name="visualization_handoff_schema_version",
        ),
    )
    object.__setattr__(instance, "workflow_id", _non_empty_text(instance.workflow_id, field_name="workflow_id"))
    object.__setattr__(instance, "valid", bool(instance.valid))
    object.__setattr__(instance, "executed", executed)
    object.__setattr__(instance, "plan_only", plan_only)
    object.__setattr__(instance, "will_write", False)
    object.__setattr__(instance, "output_written", False)
    object.__setattr__(instance, "no_output_written", True)
    object.__setattr__(
        instance,
        "output_paths_written",
        tuple(str(path) for path in instance.output_paths_written if str(path)),
    )
    object.__setattr__(instance, "will_render", False)
    object.__setattr__(instance, "rendered", False)
    object.__setattr__(instance, "figure_rendered", False)
    object.__setattr__(instance, "report_rendered", False)
    object.__setattr__(instance, "runtime_backend", _non_empty_text(instance.runtime_backend, field_name="runtime_backend"))
    object.__setattr__(
        instance,
        "primary_rowset_name",
        _non_empty_text(instance.primary_rowset_name, field_name="primary_rowset_name"),
    )
    object.__setattr__(instance, "status", _non_empty_text(instance.status, field_name="status"))
    object.__setattr__(instance, "warnings", tuple(_unique_text(instance.warnings)))
    object.__setattr__(instance, "errors", tuple(_unique_text(instance.errors)))
    for field_name in (
        "input_summary_rows",
        "association_dataset_rows",
        "publication_display_rows",
        "publication_machine_rows",
        "qc_summary_rows",
        "missingness_summary_rows",
        "multiplicity_summary_rows",
        "provenance_table_rows",
        "figure_specs",
        "report_handoff_specs",
        "spec_rows",
        "visual_qc_rows",
        "manifest_rows",
    ):
        object.__setattr__(instance, field_name, tuple(getattr(instance, field_name)))
    object.__setattr__(instance, "planned_row_counts", _json_safe_mapping(instance.planned_row_counts))
    object.__setattr__(instance, "planned_figure_names", _text_tuple(instance.planned_figure_names))
    object.__setattr__(instance, "planned_report_names", _text_tuple(instance.planned_report_names))
    object.__setattr__(instance, "planned_output_names", _text_tuple(instance.planned_output_names))


def _columns_for_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    columns: list[str] = []
    for row in rows:
        for column in row:
            text_column = str(column)
            if text_column not in columns:
                columns.append(text_column)
    return tuple(columns)


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif is_dataclass(value):
        value = {field_.name: getattr(value, field_.name) for field_ in fields(value)}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or expose to_dict().")
    return value


def _messages(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            number = float(stripped)
        except ValueError:
            return None
        return int(number) if math.isfinite(number) and number.is_integer() else None
    return None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _has_text(value: Any) -> bool:
    return _optional_text(value) is not None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _non_empty_text(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _display_scalar(value: Any) -> str:
    safe_value = _json_safe(value)
    if safe_value is None:
        return ""
    if isinstance(safe_value, bool):
        return "true" if safe_value else "false"
    if isinstance(safe_value, int | float):
        return repr(safe_value)
    if isinstance(safe_value, str):
        return safe_value.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return json.dumps(safe_value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _unique_text(values: Sequence[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            unique.append(text)
            seen.add(text)
    return tuple(unique)


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


def _tsv_safe_mapping(value: Mapping[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, item in value.items():
        if item is None:
            row[str(key)] = ""
        elif isinstance(item, bool):
            row[str(key)] = "true" if item else "false"
        elif isinstance(item, (str, int, float)):
            row[str(key)] = str(_json_safe(item))
        else:
            row[str(key)] = json.dumps(_json_safe(item), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return row


__all__ = [
    "AssociationReportHandoffSpec",
    "AssociationVisualizationDatasetRow",
    "AssociationVisualizationFigureSpec",
    "AssociationVisualizationInputSummaryRow",
    "AssociationVisualizationManifestRow",
    "AssociationVisualizationProvenanceRow",
    "AssociationVisualizationSpecRow",
    "AssociationVisualizationTextSpec",
    "RUNTIME_BACKEND_RECORDS",
    "TABULAR_ASSOCIATION_SCHEMA_VERSION",
    "TABULAR_ASSOCIATION_VISUALIZATION_HANDOFF_VERSION",
    "TabularAssociationVisualizationPlan",
    "TabularAssociationVisualizationResult",
    "build_tabular_association_visualization_handoff",
    "plan_tabular_association_visualization_handoff",
]
