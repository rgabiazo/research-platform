from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from research_platform.viz import (
    FigureSectionSpec,
    LayoutSpec,
    ReportSectionSpec,
    ReportSpec,
    TableSectionSpec,
    VisualizationFormatSpec,
    VisualizationOutputSpec,
    VisualizationSourceSpec,
    VisualQcSpec,
    build_point_interval_plot_spec,
    build_report_document,
    build_table_section,
    build_visual_layout_qc_rows,
    plan_visualization_outputs,
    render_visualization_outputs,
)


def _rows(count: int = 3) -> list[dict[str, object]]:
    labels = ("effect-alpha", "effect-beta", "effect-gamma", "effect-delta", "effect-epsilon")
    return [
        {
            "participant": f"participant-{chr(97 + index)}",
            "session": "session-a",
            "task": "task-alpha",
            "label": labels[index % len(labels)],
            "estimate": round(0.2 + index * 0.1, 3),
            "low": round(0.1 + index * 0.1, 3),
            "high": round(0.3 + index * 0.1, 3),
            "group": "group-alpha" if index % 2 == 0 else "group-beta",
            "measure": "measure-alpha",
            "data_label": f"measure-alpha-{index}",
        }
        for index in range(count)
    ]


def _plot_spec(**overrides: object):
    values = {
        "label_column": "label",
        "estimate_column": "estimate",
        "lower_column": "low",
        "upper_column": "high",
        "group_column": "group",
        "data_label_column": "data_label",
        "title": "Analysis {analysis_title}",
        "subtitle": "Synthetic summary",
        "caption": "Intervals were supplied by the caller.",
        "footnote": "No statistics are recomputed.",
        "alt_text": "Point interval plot for synthetic effects.",
        "x_label": "Already-computed estimate",
        "y_label": "Effect label",
        "legend_title": "Group",
        "legend_label_mapping": {"group-alpha": "Group Alpha", "group-beta": "Group Beta"},
    }
    values.update(overrides)
    return build_point_interval_plot_spec(**values)


def _report_spec() -> ReportSpec:
    return ReportSpec(
        title="Report {analysis_title}",
        subtitle="Reusable visualization output",
        methods_note="Input rows were already computed before rendering.",
        sections=(
            FigureSectionSpec(
                section_id="figure",
                heading="Figure",
                plot_id="point_interval",
                caption="Caller-provided estimates and intervals.",
                alt_text="Synthetic point interval figure.",
            ),
            build_table_section(
                section_id="rows",
                heading="Rows",
                columns=("label", "estimate", "low", "high", "group"),
                caption="Synthetic source rows.",
            ),
        ),
    )


def _output(tmp_path: Path, **paths: str) -> VisualizationOutputSpec:
    return VisualizationOutputSpec(output_root=tmp_path / "out", **paths)


def test_plan_mode_writes_nothing(tmp_path: Path) -> None:
    output = _output(
        tmp_path,
        report_markdown_path="report.md",
        figure_svg_path="figure.svg",
        manifest_path="manifest.json",
    )

    plan = plan_visualization_outputs(
        _rows(),
        output_spec=output,
        plot_spec=_plot_spec(),
        report_spec=_report_spec(),
        metadata={"analysis_title": "Alpha"},
    )

    assert plan.status == "ok"
    assert plan.output_written is False
    assert not (tmp_path / "out").exists()
    assert plan.report_previews["report_markdown"].startswith("# Report Alpha")
    assert plan.figure_previews["figure_svg"].startswith("<svg")


def test_render_mode_writes_only_planned_files(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    unplanned = output_root / "unplanned.txt"
    unplanned.write_text("keep me\n", encoding="utf-8")
    output = VisualizationOutputSpec(
        output_root=output_root,
        report_markdown_path="report.md",
        manifest_path="manifest.json",
    )

    result = render_visualization_outputs(
        _rows(),
        output_spec=output,
        report_spec=_report_spec(),
        metadata={"analysis_title": "Alpha"},
    )

    assert result.output_written is True
    assert (output_root / "report.md").exists()
    assert (output_root / "manifest.json").exists()
    assert unplanned.read_text(encoding="utf-8") == "keep me\n"
    assert sorted(path.name for path in output_root.iterdir()) == ["manifest.json", "report.md", "unplanned.txt"]


def test_overwrite_refusal_by_default(tmp_path: Path) -> None:
    output = _output(tmp_path, report_markdown_path="report.md")
    render_visualization_outputs(
        _rows(),
        output_spec=output,
        report_spec=_report_spec(),
        metadata={"analysis_title": "Alpha"},
    )

    with pytest.raises(FileExistsError):
        render_visualization_outputs(
            _rows(),
            output_spec=output,
            report_spec=_report_spec(),
            metadata={"analysis_title": "Alpha"},
        )


def test_overwrite_true_replaces_only_planned_files(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    sidecar = output_root / "sidecar.txt"
    sidecar.write_text("do not touch\n", encoding="utf-8")
    output = VisualizationOutputSpec(output_root=output_root, report_markdown_path="report.md")
    render_visualization_outputs(
        _rows(),
        output_spec=output,
        report_spec=ReportSpec(title="Report {analysis_title}", sections=(build_table_section(columns=("label",)),)),
        metadata={"analysis_title": "Alpha"},
    )

    render_visualization_outputs(
        _rows(),
        output_spec=output,
        report_spec=ReportSpec(title="Report {analysis_title}", sections=(build_table_section(columns=("label",)),)),
        metadata={"analysis_title": "Beta"},
        overwrite=True,
    )

    assert "# Report Beta" in (output_root / "report.md").read_text(encoding="utf-8")
    assert sidecar.read_text(encoding="utf-8") == "do not touch\n"


def test_markdown_report_from_in_memory_rows(tmp_path: Path) -> None:
    output = _output(tmp_path, report_markdown_path="report.md")

    render_visualization_outputs(
        _rows(),
        output_spec=output,
        report_spec=_report_spec(),
        metadata={"analysis_title": "Alpha"},
    )

    text = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "# Report Alpha" in text
    assert "| label | estimate | low | high | group |" in text
    assert "effect-alpha" in text


def test_markdown_report_from_tsv_source_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.tsv"
    source.write_text(
        "label\testimate\tlow\thigh\tgroup\n"
        "effect-alpha\t0.2\t0.1\t0.3\tgroup-alpha\n",
        encoding="utf-8",
    )
    output = _output(tmp_path, report_markdown_path="report.md")

    render_visualization_outputs(
        source_spec=VisualizationSourceSpec(source_id="table-alpha", path=source),
        output_spec=output,
        report_spec=ReportSpec(title="Report {analysis_title}", sections=(build_table_section(columns=("label", "estimate")),)),
        metadata={"analysis_title": "Alpha"},
    )

    assert "effect-alpha" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_markdown_report_from_json_source_rows_with_row_key(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"rows": _rows(2)}), encoding="utf-8")
    output = _output(tmp_path, report_markdown_path="report.md")

    render_visualization_outputs(
        source_spec=VisualizationSourceSpec(source_id="json-alpha", path=source, json_rows_key="rows"),
        output_spec=output,
        report_spec=ReportSpec(title="Report {analysis_title}", sections=(build_table_section(columns=("participant", "label")),)),
        metadata={"analysis_title": "Alpha"},
    )

    text = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "participant-a" in text
    assert "effect-beta" in text


def test_lightweight_html_report(tmp_path: Path) -> None:
    output = _output(tmp_path, report_html_path="report.html")

    render_visualization_outputs(
        _rows(),
        output_spec=output,
        report_spec=_report_spec(),
        metadata={"analysis_title": "Alpha"},
    )

    html = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "<table>" in html
    assert "Report Alpha" in html


def test_plain_text_report(tmp_path: Path) -> None:
    output = _output(tmp_path, report_text_path="report.txt")

    render_visualization_outputs(
        _rows(),
        output_spec=output,
        report_spec=_report_spec(),
        metadata={"analysis_title": "Alpha"},
    )

    text = (tmp_path / "out" / "report.txt").read_text(encoding="utf-8")
    assert "Report Alpha" in text
    assert "label\testimate\tlow\thigh\tgroup" in text


def test_dependency_free_svg_point_interval_figure(tmp_path: Path) -> None:
    output = _output(tmp_path, figure_svg_path="figure.svg")

    render_visualization_outputs(
        _rows(),
        output_spec=output,
        plot_spec=_plot_spec(),
        metadata={"analysis_title": "Alpha"},
    )

    svg = (tmp_path / "out" / "figure.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "Analysis Alpha" in svg
    assert "effect-alpha" in svg
    assert "Group Alpha" in svg


def test_configurable_title_subtitle_caption_footnote(tmp_path: Path) -> None:
    output = _output(tmp_path, report_markdown_path="report.md", figure_svg_path="figure.svg")

    render_visualization_outputs(
        _rows(),
        output_spec=output,
        plot_spec=_plot_spec(
            title="Custom {analysis_title}",
            subtitle="Subtitle {analysis_title}",
            caption="Caption {analysis_title}",
            footnote="Footnote {analysis_title}",
        ),
        report_spec=ReportSpec(
            title="Report {analysis_title}",
            subtitle="Subtitle {analysis_title}",
            caption="Caption {analysis_title}",
            footnote="Footnote {analysis_title}",
            sections=(FigureSectionSpec(plot_id="point_interval"),),
        ),
        metadata={"analysis_title": "Alpha"},
    )

    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    figure = (tmp_path / "out" / "figure.svg").read_text(encoding="utf-8")
    assert "Subtitle Alpha" in report
    assert "Caption Alpha" in report
    assert "Footnote Alpha" in report
    assert "Custom Alpha" in figure
    assert "Caption Alpha" in figure
    assert "Footnote Alpha" in figure


def test_configurable_axis_and_legend_labels(tmp_path: Path) -> None:
    output = _output(tmp_path, figure_svg_path="figure.svg")

    render_visualization_outputs(
        _rows(),
        output_spec=output,
        plot_spec=_plot_spec(
            x_label="Configured x label",
            y_label="Configured y label",
            legend_title="Configured legend",
            legend_label_mapping={"group-alpha": "Mapped Alpha", "group-beta": "Mapped Beta"},
        ),
        metadata={"analysis_title": "Alpha"},
    )

    svg = (tmp_path / "out" / "figure.svg").read_text(encoding="utf-8")
    assert "Configured x label" in svg
    assert "Configured y label" in svg
    assert "Configured legend" in svg
    assert "Mapped Alpha" in svg


def test_metadata_template_expansion(tmp_path: Path) -> None:
    output = _output(tmp_path, report_markdown_path="report.md", figure_svg_path="figure.svg")

    plan = plan_visualization_outputs(
        _rows(),
        output_spec=output,
        plot_spec=_plot_spec(title="Plot {analysis_title}"),
        report_spec=ReportSpec(title="Report {analysis_title}", sections=(FigureSectionSpec(plot_id="point_interval"),)),
        metadata={"analysis_title": "Alpha"},
    )

    assert "# Report Alpha" in plan.report_previews["report_markdown"]
    assert "Plot Alpha" in plan.figure_previews["figure_svg"]
    assert plan.errors == ()


def test_missing_template_fields_are_reported_as_errors_and_qc(tmp_path: Path) -> None:
    plan = plan_visualization_outputs(
        _rows(),
        output_spec=_output(tmp_path, report_markdown_path="report.md", figure_svg_path="figure.svg"),
        plot_spec=_plot_spec(title="Plot {missing_field}"),
        report_spec=ReportSpec(title="Report {missing_field}"),
    )

    assert plan.status == "error"
    assert any("missing_field" in error for error in plan.errors)
    assert any(row.check_id == "missing_template_field" for row in plan.visual_qc_rows)


def test_visual_qc_for_long_titles_axis_labels_dense_ticks_missing_labels_and_overlaps() -> None:
    long_text = " ".join(["long-title"] * 18)
    qc = VisualQcSpec(
        require_title=True,
        require_axis_labels=True,
        require_caption=True,
        require_alt_text=True,
        max_tick_count=2,
        max_data_labels=2,
        max_axis_label_chars=20,
        max_title_chars=30,
    )
    spec = _plot_spec(
        title=long_text,
        x_label=" ".join(["very-long-axis-label"] * 5),
        y_label=None,
        caption=None,
        alt_text=None,
        visual_qc=qc,
        layout=LayoutSpec(width=500, height=330),
    )

    rows = build_visual_layout_qc_rows(_rows(5), spec, metadata={"analysis_title": "Alpha"})
    check_ids = {row.check_id for row in rows}

    assert "title_clip_risk" in check_ids
    assert "x_axis_label_clip_risk" in check_ids
    assert "dense_tick_labels" in check_ids
    assert "missing_y_label" in check_ids
    assert "missing_caption" in check_ids
    assert "missing_alt_text" in check_ids
    assert "data_label_overlap_risk" in check_ids


def test_manifest_records_sources_outputs_settings_qc_warnings_and_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text(
        "label,estimate,low,high,group,data_label\n"
        "effect-alpha,0.2,0.1,0.3,group-alpha,label-alpha\n"
        "effect-beta,0.3,0.2,0.4,group-beta,label-beta\n"
        "effect-gamma,0.4,0.3,0.5,group-alpha,label-gamma\n",
        encoding="utf-8",
    )
    output = _output(
        tmp_path,
        report_markdown_path="report.md",
        figure_svg_path="figure.svg",
        manifest_path="manifest.json",
    )
    spec = _plot_spec(
        title=" ".join(["long-title"] * 12),
        visual_qc=VisualQcSpec(max_title_chars=20),
    )

    render_visualization_outputs(
        source_spec=VisualizationSourceSpec(source_id="source-alpha", path=source),
        output_spec=output,
        plot_spec=spec,
        report_spec=_report_spec(),
        metadata={"analysis_title": "Alpha"},
        provenance={"analysis": "analysis-alpha"},
    )

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert str(source.resolve()) in manifest["source_paths"]
    assert manifest["source_row_counts"]["source-alpha"] == 3
    assert manifest["output_paths"]["report_markdown"].endswith("report.md")
    assert manifest["renderer_settings"]["renderers"]["figure_svg"]["renderer"] == "svg_figure"
    assert manifest["text_settings"]["plots"][0]["axes"]["x_label"] == "Already-computed estimate"
    assert manifest["layout_settings"]["point_interval"]["width"] == 720
    assert any(row["check_id"] == "title_clip_risk" for row in manifest["visual_qc_rows"])
    assert manifest["warnings"]
    assert manifest["provenance"]["analysis"] == "analysis-alpha"


def test_missing_source_path_errors(tmp_path: Path) -> None:
    plan = plan_visualization_outputs(
        source_spec=VisualizationSourceSpec(source_id="missing-alpha", path=tmp_path / "missing.tsv"),
        output_spec=_output(tmp_path, report_markdown_path="report.md"),
        report_spec=ReportSpec(title="Report"),
    )

    assert plan.status == "error"
    assert any("does not exist" in error for error in plan.errors)


def test_unsupported_format_errors(tmp_path: Path) -> None:
    plan = plan_visualization_outputs(
        _rows(),
        output_spec=_output(tmp_path, report_markdown_path="report.md"),
        report_spec=ReportSpec(title="Report"),
        format_spec=VisualizationFormatSpec(report_formats=("docx",)),
    )

    assert plan.status == "error"
    assert any("Unsupported report format" in error for error in plan.errors)


def test_renderer_unavailable_warnings_and_errors(tmp_path: Path) -> None:
    plan = plan_visualization_outputs(
        _rows(),
        output_spec=_output(tmp_path, figure_png_path="figure.png"),
        plot_spec=_plot_spec(),
        metadata={"analysis_title": "Alpha"},
    )

    assert plan.status == "error"
    assert any("unavailable" in warning for warning in plan.warnings)
    assert any("unavailable" in error for error in plan.errors)


def test_generic_mvpa_style_report_from_neutral_synthetic_rows() -> None:
    publication_rows = (
        {
            "roi": "roi-alpha",
            "contrast": "contrast-alpha",
            "effect": "effect-alpha",
            "estimate": "0.200",
            "interval": "[0.100, 0.300]",
            "status": "ok",
        },
    )
    inference_rows = (
        {
            "effect": "effect-alpha",
            "measure": "measure-alpha",
            "participant_count": 3,
            "source_table": "step-nine-table-alpha",
        },
    )
    report = build_report_document(
        report_spec=ReportSpec(
            title="{analysis_title}",
            methods_note="This report references already-computed rows and does not rerun inference.",
            sections=(
                TableSectionSpec(section_id="publication", heading="Publication Rows", rows=publication_rows),
                TableSectionSpec(section_id="inference", heading="Inference Rows", rows=inference_rows),
                ReportSectionSpec(section_id="links", heading="References", text="See {source_reference}."),
            ),
        ),
        metadata={"analysis_title": "Analysis Alpha", "source_reference": "step-nine-table-alpha"},
    )

    assert "Analysis Alpha" in report
    assert "roi-alpha" in report
    assert "contrast-alpha" in report
    assert "step-nine-table-alpha" in report
    assert "does not rerun inference" in report


def test_public_results_are_json_safe_with_allow_nan_false(tmp_path: Path) -> None:
    rows = [{"label": "effect-alpha", "estimate": float("nan")}]
    plan = plan_visualization_outputs(
        rows,
        output_spec=_output(tmp_path, report_markdown_path="report.md"),
        report_spec=ReportSpec(title="Report"),
    )
    rendered = render_visualization_outputs(
        _rows(),
        output_spec=_output(tmp_path / "rendered", report_markdown_path="report.md"),
        report_spec=ReportSpec(title="Report"),
    )

    json.dumps(plan.to_dict(), allow_nan=False)
    json.dumps(rendered.to_dict(), allow_nan=False)


def test_no_prohibited_imports_or_runtime_dependencies() -> None:
    source_files = [
        Path("packages/research-viz/src/research_platform/viz/plots.py"),
        Path("packages/research-viz/src/research_platform/viz/reports.py"),
        Path("packages/research-viz/src/research_platform/viz/outputs.py"),
        Path("packages/research-viz/src/research_platform/viz/__init__.py"),
    ]
    banned_top_level = {
        "matplotlib",
        "nilearn",
        "numpy",
        "pandas",
        "polars",
        "rsatoolbox",
        "scipy",
        "seaborn",
        "sklearn",
    }
    banned_prefixes = (
        "research_platform.neuro",
        "research_platform.core",
        "research_platform.bids",
        "pipelines",
        "ops",
    )

    for source_file in source_files:
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    assert name.split(".", 1)[0] not in banned_top_level
                    assert not name.startswith(banned_prefixes)
            elif isinstance(node, ast.ImportFrom) and node.module:
                name = node.module
                assert name.split(".", 1)[0] not in banned_top_level
                assert not name.startswith(banned_prefixes)


def test_no_hard_coded_study_specific_constants_in_new_files() -> None:
    files = [
        Path("packages/research-viz/src/research_platform/viz/plots.py"),
        Path("packages/research-viz/src/research_platform/viz/reports.py"),
        Path("packages/research-viz/src/research_platform/viz/outputs.py"),
    ]
    forbidden_literals = (
        "confidential-study-marker",
        "private-task-marker",
        "private-cohort-marker",
    )

    for path in files:
        text = path.read_text(encoding="utf-8")
        assert str(Path.home()) not in text
        assert re.search(r"\bsub-\d{3}\b", text) is None
        for value in forbidden_literals:
            assert value not in text
