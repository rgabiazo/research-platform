from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from research_platform.viz import (
    AssociationVisualizationDatasetRow,
    AssociationVisualizationTextSpec,
    VisualQcSpec,
    build_tabular_association_visualization_handoff,
    plan_tabular_association_visualization_handoff,
)


@dataclass(frozen=True)
class SyntheticAssociationRow:
    workflow_id: str = "workflow-alpha"
    result_row_id: str = "result-alpha"
    pair_id: str = "pair-alpha"
    method_id: str = "method-alpha"
    method_kind: str = "correlation"
    method_name: str = "method-alpha"
    family_id: str = "family-alpha"
    source_id: str = "source-alpha"
    outcome_id: str = "outcome-alpha"
    predictor_id: str = "predictor-alpha"
    covariate_ids: tuple[str, ...] = ("covariate-alpha",)
    statistic_name: str = "statistic-alpha"
    statistic_value: float = 0.1
    p_value: float = 0.01
    status: str = "ok"


class ToDictRow:
    def __init__(self, **values: object) -> None:
        self.values = values

    def to_dict(self) -> dict[str, object]:
        return dict(self.values)


def _association_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "workflow_id": "workflow-alpha",
        "result_row_id": "result-alpha",
        "pair_id": "pair-alpha",
        "method_id": "method-alpha",
        "method_kind": "correlation",
        "method_name": "method-alpha",
        "family_id": "family-alpha",
        "source_id": "source-alpha",
        "outcome_id": "outcome-alpha",
        "predictor_id": "predictor-alpha",
        "covariate_ids": ["covariate-alpha"],
        "statistic_name": "statistic-alpha",
        "statistic_value": 0.1,
        "p_value": 0.01,
        "status": "ok",
        "warnings": [],
        "errors": [],
    }
    row.update(updates)
    return row


def test_publication_table_rows_are_preferred_and_p_q_pass_through_without_mutation() -> None:
    association_row = SyntheticAssociationRow(statistic_value=9.0, p_value=0.9)
    publication_row = {
        "workflow_id": "workflow-alpha",
        "result_row_id": "result-alpha",
        "pair_id": "pair-alpha",
        "method_id": "method-alpha",
        "method_kind": "correlation",
        "method_name": "method-alpha",
        "family_id": "family-alpha",
        "source_id": "source-alpha",
        "outcome_id": "outcome-alpha",
        "predictor_id": "predictor-alpha",
        "covariate_ids": ["covariate-alpha"],
        "statistic_name": "statistic-alpha",
        "statistic_value": 0.2,
        "ci_low": 0.1,
        "ci_high": 0.3,
        "p_value": 0.02,
        "q_value": 0.04,
        "input_row_index": 5,
        "status": "ok",
    }
    original_publication_row = dict(publication_row)

    plan = plan_tabular_association_visualization_handoff(
        [association_row],
        publication_table_rows=[publication_row],
        publication_display_rows=[{"display_label": "result-alpha", "q_value": "0.040"}],
        publication_machine_rows=[ToDictRow(result_row_id="result-alpha", q_value=0.04)],
        source_rowset_names={"publication_table_rows": "publication-table-alpha"},
    )
    result = build_tabular_association_visualization_handoff(
        [association_row],
        publication_table_rows=[publication_row],
        publication_display_rows=[{"display_label": "result-alpha", "q_value": "0.040"}],
        publication_machine_rows=[ToDictRow(result_row_id="result-alpha", q_value=0.04)],
        source_rowset_names={"publication_table_rows": "publication-table-alpha"},
    )

    assert plan.executed is False
    assert plan.plan_only is True
    assert result.executed is True
    assert result.plan_only is False
    assert result.primary_rowset_name == "publication_table_rows"
    assert result.association_dataset_rows[0].source_rowset_name == "publication-table-alpha"
    assert result.association_dataset_rows[0].estimate == 0.2
    assert result.association_dataset_rows[0].p_value == 0.02
    assert result.association_dataset_rows[0].q_value == 0.04
    assert result.association_dataset_rows[0].input_row_index == 5
    assert result.association_dataset_rows[0].has_interval is True
    assert result.publication_display_rows[0]["q_value"] == "0.040"
    assert result.publication_machine_rows[0]["q_value"] == 0.04
    assert publication_row == original_publication_row


def test_raw_association_rows_create_point_interval_and_point_only_specs() -> None:
    interval_result = build_tabular_association_visualization_handoff(
        [_association_row(ci_low=0.0, ci_high=0.2, input_row_index=8)],
        title="Figure Alpha",
        x_axis_label="Estimate Alpha",
        y_axis_label="Result Alpha",
    )
    point_result = build_tabular_association_visualization_handoff(
        [_association_row(result_row_id="result-beta", statistic_value=0.4, p_value=None)],
        title="Figure Alpha",
    )

    interval_spec = interval_result.figure_specs[0].plot_spec
    point_spec = point_result.figure_specs[0].plot_spec

    assert interval_result.primary_rowset_name == "association_rows"
    assert interval_result.association_dataset_rows[0].input_row_index == 8
    assert interval_result.association_dataset_rows[0].lower == 0.0
    assert interval_result.association_dataset_rows[0].upper == 0.2
    assert interval_spec.lower_column == "lower"
    assert interval_spec.upper_column == "upper"
    assert point_result.association_dataset_rows[0].has_interval is False
    assert point_result.association_dataset_rows[0].q_value is None
    assert point_spec.lower_column is None
    assert point_spec.upper_column is None


def test_configurable_fields_text_controls_and_report_sections_are_pathless() -> None:
    text = AssociationVisualizationTextSpec(
        title="Figure {analysis_label}",
        subtitle="Subtitle {analysis_label}",
        caption="Caption {analysis_label}",
        footnote="Footnote {analysis_label}",
        x_axis_label="Configured x",
        y_axis_label="Configured y",
        legend_title="Configured legend",
        legend_labels={"family-alpha": "Family Alpha"},
        panel_labels={"panel-alpha": "Panel Alpha"},
        alt_text="Alt {analysis_label}",
        methods_note="Methods {analysis_label}",
        report_title="Report {analysis_label}",
        report_section_headings={"association_summary": "Configured Association Summary"},
    )
    result = build_tabular_association_visualization_handoff(
        [
            _association_row(
                estimate_alpha=0.5,
                low_alpha=0.2,
                high_alpha=0.8,
                label_alpha="label-alpha",
                family_id="family-alpha",
                data_label="label-alpha-data",
            )
        ],
        estimate_field="estimate_alpha",
        lower_field="low_alpha",
        upper_field="high_alpha",
        label_field="label_alpha",
        group_field="family_id",
        data_label_field="data_label",
        figure_id="figure-alpha",
        report_id="report-alpha",
        text_spec=text,
        metadata={"analysis_label": "Alpha"},
    )

    row = result.association_dataset_rows[0]
    plot_spec = result.figure_specs[0].plot_spec
    report_spec = result.report_handoff_specs[0].report_spec

    assert row.label == "label-alpha"
    assert row.estimate == 0.5
    assert row.lower == 0.2
    assert row.upper == 0.8
    assert row.group_id == "family-alpha"
    assert row.data_label == "label-alpha-data"
    assert plot_spec.plot_id == "figure-alpha"
    assert plot_spec.group_column == "group_id"
    assert plot_spec.data_label_column == "data_label"
    assert plot_spec.text.methods_note == "Methods {analysis_label}"
    assert plot_spec.text.panel_labels["panel-alpha"] == "Panel Alpha"
    assert report_spec.report_id == "report-alpha"
    assert any(section.section_id == "association_figure" and section.figure_path is None for section in report_spec.sections)
    assert any(section.heading == "Configured Association Summary" for section in report_spec.sections)
    assert result.status == "ok"


def test_safe_metadata_templates_and_invalid_templates_return_visual_qc_rows() -> None:
    ok_result = build_tabular_association_visualization_handoff(
        [_association_row()],
        title="Figure {analysis_label}",
        report_section_headings={"association_summary": "Summary {analysis_label}"},
        metadata={"analysis_label": "Alpha"},
    )
    bad_result = build_tabular_association_visualization_handoff(
        [_association_row()],
        title="Figure {missing_label}",
        report_section_headings={
            "association_summary": "Summary {analysis.label}",
            "provenance": "Provenance {analysis_label:{nested}}",
        },
        metadata={"analysis_label": "Alpha"},
    )

    assert not any(row.check_id == "missing_template_field" for row in ok_result.visual_qc_rows)
    check_ids = {row.check_id for row in bad_result.visual_qc_rows}
    messages = " ".join(row.message for row in bad_result.visual_qc_rows)
    assert "missing_template_field" in check_ids
    assert "missing_label" in messages
    assert "analysis.label" in messages
    assert "nested fields" in messages
    assert bad_result.status == "error"


def test_visual_qc_covers_required_text_dense_long_labels_values_and_intervals() -> None:
    rows = [
        _association_row(
            result_row_id=f"result-alpha-{index}",
            label="long-label-alpha-" + ("segment-" * 5) + str(index),
            statistic_value=0.1 + index,
            ci_low=0.3 if index == 0 else 0.0,
            ci_high=0.2 if index == 0 else 0.5,
        )
        for index in range(5)
    ]

    result = build_tabular_association_visualization_handoff(
        rows,
        label_field="label",
        visual_qc=VisualQcSpec(
            require_title=True,
            require_axis_labels=True,
            require_caption=True,
            require_alt_text=True,
            max_tick_count=2,
            max_tick_label_chars=12,
        ),
    )
    check_ids = {row.check_id for row in result.visual_qc_rows}

    assert "missing_title" in check_ids
    assert "missing_x_label" in check_ids
    assert "missing_y_label" in check_ids
    assert "missing_caption" in check_ids
    assert "missing_alt_text" in check_ids
    assert "dense_tick_labels" in check_ids
    assert "long_tick_labels" in check_ids
    assert "interval_bounds_invalid" in check_ids
    assert result.status == "error"


def test_qc_missingness_multiplicity_and_provenance_are_report_handoff_rowsets() -> None:
    result = build_tabular_association_visualization_handoff(
        [_association_row(p_value=0.01, q_value=None)],
        multiplicity_rows=[
            {
                "workflow_id": "workflow-alpha",
                "family_id": "family-alpha",
                "result_row_id": "result-alpha",
                "p_value": 0.01,
                "q_value": 0.05,
                "status": "ok",
            }
        ],
        qc_rows=[{"workflow_id": "workflow-alpha", "code": "qc-alpha", "message": "QC alpha.", "status": "ok"}],
        missingness_rows=[
            {
                "workflow_id": "workflow-alpha",
                "source_id": "source-alpha",
                "column_name": "outcome-alpha",
                "role": "outcome",
                "missing_count": 0,
                "nonmissing_count": 3,
                "total_count": 3,
                "status": "ok",
            }
        ],
        provenance_rows=[{"workflow_id": "workflow-alpha", "key": "source_rowset", "value": "source-alpha"}],
        methods_note="Methods alpha.",
    )

    section_ids = set(result.report_handoff_specs[0].section_ids)
    provenance = {row.key: row.value for row in result.provenance_table_rows}

    assert result.association_dataset_rows[0].q_value is None
    assert result.multiplicity_summary_rows[0]["q_value"] == 0.05
    assert result.qc_summary_rows[0]["code"] == "qc-alpha"
    assert result.missingness_summary_rows[0]["missing_count"] == 0
    assert {"association_summary", "qc_summary", "missingness_summary", "multiplicity_summary", "provenance"}.issubset(section_ids)
    assert provenance["source_rowset"] == "source-alpha"
    assert provenance["methods_note"] == "Methods alpha."
    assert provenance["runtime_backend"] == "records"
    assert provenance["output_paths_written"] == []
    assert result.manifest_rows[0].input_row_counts["multiplicity_rows"] == 1
    assert result.manifest_rows[0].output_paths_written == ()


def test_json_safe_tsv_safe_and_no_write_or_render_flags(tmp_path: Path) -> None:
    result = build_tabular_association_visualization_handoff(
        [_association_row(statistic_value=math.nan, ci_low=0.1, ci_high=0.2, input_row_index=-1)],
        figure_id="figure-alpha",
        report_id="report-alpha",
    )
    payload = result.to_dict()
    tsv_row = result.association_dataset_rows[0].to_tsv_row()
    manifest_tsv = result.manifest_rows[0].to_tsv_row()
    direct_row_tsv = AssociationVisualizationDatasetRow(
        workflow_id="workflow-alpha",
        dataset_row_id="result-alpha",
        source_rowset_name="association_rows",
        source_row_index=0,
        input_row_index=0,
        label="result-alpha",
        estimate=0.1,
    ).to_tsv_row()

    json.dumps(payload, sort_keys=True, allow_nan=False)
    assert result.will_write is False
    assert result.output_written is False
    assert result.no_output_written is True
    assert result.output_paths_written == ()
    assert result.will_render is False
    assert result.figure_rendered is False
    assert result.report_rendered is False
    assert tsv_row["estimate"] == "nan"
    assert tsv_row["input_row_index"] == "0"
    assert manifest_tsv["output_paths_written"] == "[]"
    assert direct_row_tsv["output_written"] == "false"
    assert list(tmp_path.iterdir()) == []


def test_missing_methods_note_can_be_required_by_adapter_qc() -> None:
    missing_result = build_tabular_association_visualization_handoff(
        [_association_row()],
        require_methods_note=True,
    )
    supplied_result = build_tabular_association_visualization_handoff(
        [_association_row()],
        require_methods_note=True,
        methods_note="Methods alpha.",
    )

    assert any(row.check_id == "missing_methods_note" for row in missing_result.visual_qc_rows)
    assert not any(row.check_id == "missing_methods_note" for row in supplied_result.visual_qc_rows)


def test_new_module_has_no_forbidden_imports_or_study_specific_constants() -> None:
    source_files = [
        Path("packages/research-viz/src/research_platform/viz/tabular_associations.py"),
        Path("packages/research-viz/src/research_platform/viz/__init__.py"),
    ]
    forbidden_modules = (
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "sklearn",
        "statsmodels",
        "research_platform.analysis",
        "research_platform.core",
        "research_platform.io",
        "research_platform.neuro",
        "research_platform.bids",
        "pipelines",
        "ops",
    )
    forbidden_calls = {
        "plan_visualization_outputs",
        "render_visualization_outputs",
        "render_point_interval_svg",
        "build_report_document",
    }
    forbidden_text = (
        "confidential-study-marker",
        "private-task-marker",
        "private-cohort-marker",
        "participant-alpha",
        "participant-beta",
    )

    combined_text = ""
    for source_file in source_files:
        source_text = source_file.read_text(encoding="utf-8")
        combined_text += source_text
        tree = ast.parse(source_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules = [node.module]
            else:
                imported_modules = []
            for imported_module in imported_modules:
                assert not any(
                    imported_module == forbidden or imported_module.startswith(f"{forbidden}.")
                    for forbidden in forbidden_modules
                )
            if source_file.name == "tabular_associations.py" and isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in forbidden_calls
                elif isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in forbidden_calls

    for text in forbidden_text:
        assert text not in combined_text
    assert re.search("sub-" + r"\d{3}", combined_text) is None
