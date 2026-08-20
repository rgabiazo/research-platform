from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from research_platform.analysis.mvpa.figure_export import (
    plan_or_execute_mvpa_figure_export,
    validate_mvpa_figure_export_document,
)


def _config(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "baseline_crossnobis",
        "input": {
            "table_set": "baseline_crossnobis",
            "root_ref": "artifact_root",
            "path": ".research-platform/mvpa/reports/{table_set}/subject.tsv",
        },
        "outputs": {
            "root_ref": "artifact_root",
            "path": ".research-platform/mvpa/reports/{figure_set}/figures",
        },
        "figures": [
            _strip_figure(),
        ],
    }
    payload.update(overrides)
    return {"mvpa_figure_export": payload}


def _strip_figure(**overrides: object) -> dict[str, object]:
    figure: dict[str, object] = {
        "figure_id": "roiwise_encoding_main",
        "kind": "strip_mean_ci",
        "filters": {
            "analysis_variant": "main",
            "phase_id": "encoding",
            "contrast_id": "pair_minus_item",
        },
        "x": "roi_label",
        "y": "crossnobis",
        "order": ["RoiA", "RoiB"],
        "display_labels": {"RoiA": "ROI A", "RoiB": "ROI B"},
        "title": "Encoding crossnobis",
        "ylabel": "Crossnobis",
        "xlabel": "ROI",
        "zero_line": True,
        "ci_level": 0.95,
        "jitter_width": 0.0,
        "random_seed": 7,
        "output_basename": "ses-01_task-memory_desc-RoiwiseEncodingCrossnobis_mvpa",
        "output_formats": ["svg", "pdf", "png"],
        "dpi": 120,
    }
    figure.update(overrides)
    return figure


def _category_figure(**overrides: object) -> dict[str, object]:
    figure: dict[str, object] = {
        "figure_id": "phase_summary_main",
        "kind": "category_distribution_mean_ci",
        "filters": {"analysis_variant": "main", "phase_id": "encoding", "contrast_id": "pair_minus_item"},
        "aggregate": {
            "group_by": ["participant_id", "analysis_variant", "phase_id", "contrast_id"],
            "value": "crossnobis",
            "method": "mean",
            "across": "roi_label",
        },
        "x": "phase_id",
        "y": "crossnobis",
        "order": ["encoding"],
        "display_labels": {"encoding": "Encoding"},
        "title": "Phase summary",
        "ylabel": "Mean crossnobis",
        "xlabel": "Phase",
        "zero_line": True,
        "violin": True,
        "jitter_width": 0.0,
        "output_basename": "ses-01_task-memory_desc-EncodingRecognitionCrossnobisSummary_mvpa",
        "output_formats": ["svg", "pdf", "png"],
        "dpi": 120,
    }
    figure.update(overrides)
    return figure


def _write_table(artifact_root: Path, rows: list[dict[str, object]], *, columns: list[str] | None = None) -> Path:
    path = artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "subject.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = columns or [
        "participant_id",
        "analysis_variant",
        "phase_id",
        "roi_label",
        "contrast_id",
        "crossnobis",
        "feature_count",
        "cv_unit_count",
        "observation_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            base = {
                "participant_id": "sub-001",
                "analysis_variant": "main",
                "phase_id": "encoding",
                "roi_label": "RoiA",
                "contrast_id": "pair_minus_item",
                "crossnobis": 1.0,
                "feature_count": 10,
                "cv_unit_count": 3,
                "observation_count": 6,
            }
            base.update(row)
            writer.writerow({column: base.get(column, "") for column in fieldnames})
    return path


def _rows_for_strip() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for participant_id, value in (("sub-001", 1.0), ("sub-002", 2.0), ("sub-003", 3.0)):
        rows.append({"participant_id": participant_id, "roi_label": "RoiA", "crossnobis": value})
    for participant_id, value in (("sub-001", 2.0), ("sub-002", 4.0), ("sub-003", 6.0)):
        rows.append({"participant_id": participant_id, "roi_label": "RoiB", "crossnobis": value})
    rows.append(
        {
            "participant_id": "sub-001",
            "analysis_variant": "sensitivity",
            "roi_label": "RoiA",
            "crossnobis": 100.0,
        }
    )
    return rows


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_config_validation_accepts_figure_export_document() -> None:
    assert validate_mvpa_figure_export_document(_config()) == []


def test_plan_mode_writes_nothing(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_strip())

    result = plan_or_execute_mvpa_figure_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    assert result["executed"] is False
    assert result["figures"][0]["row_counts"] == {"plot_data": 6, "summary": 2}
    assert not (artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "figures").exists()


def test_strip_mean_ci_filters_rows_and_computes_mean_and_ci(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_strip())

    result = plan_or_execute_mvpa_figure_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    roi_a_summary = result["figures"][0]["summary_rows"][0]
    assert roi_a_summary["category"] == "RoiA"
    assert roi_a_summary["n"] == 3
    assert roi_a_summary["mean"] == pytest.approx(2.0)
    assert roi_a_summary["sd"] == pytest.approx(1.0)
    assert roi_a_summary["sem"] == pytest.approx(1.0 / math.sqrt(3))
    assert roi_a_summary["ci_low"] == pytest.approx(2.0 - 4.3026527299 / math.sqrt(3))
    assert roi_a_summary["ci_high"] == pytest.approx(2.0 + 4.3026527299 / math.sqrt(3))
    assert all(row["analysis_variant"] == "main" for row in result["figures"][0]["plot_data_rows"])


def test_category_distribution_aggregates_within_participant_only(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(
        artifact_root,
        [
            {"participant_id": "sub-001", "roi_label": "RoiA", "crossnobis": 1.0},
            {"participant_id": "sub-001", "roi_label": "RoiB", "crossnobis": 3.0},
            {"participant_id": "sub-002", "roi_label": "RoiA", "crossnobis": 5.0},
            {"participant_id": "sub-002", "roi_label": "RoiB", "crossnobis": 7.0},
        ],
    )

    result = plan_or_execute_mvpa_figure_export(
        _config(figures=[_category_figure()]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    plot_values = [row["crossnobis"] for row in result["figures"][0]["plot_data_rows"]]
    assert plot_values == [2.0, 6.0]
    assert result["figures"][0]["plot_data_rows"][0]["pooled_roi_count"] == 2
    assert result["figures"][0]["summary_rows"][0]["n"] == 2
    assert result["figures"][0]["summary_rows"][0]["mean"] == pytest.approx(4.0)


def test_missing_required_columns_fail_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(
        artifact_root,
        [{"participant_id": "sub-001"}],
        columns=["participant_id", "analysis_variant", "phase_id", "roi_label", "contrast_id"],
    )

    result = plan_or_execute_mvpa_figure_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert any("missing required column" in error for error in result["errors"])


def test_nonnumeric_crossnobis_fails_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, [{"participant_id": "sub-001", "crossnobis": "bad"}, {"participant_id": "sub-002"}])

    result = plan_or_execute_mvpa_figure_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert any("crossnobis must be finite numeric" in error for error in result["errors"])


def test_too_few_participants_fails_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, [{"participant_id": "sub-001", "roi_label": "RoiA", "crossnobis": 1.0}])

    result = plan_or_execute_mvpa_figure_export(
        _config(figures=[_strip_figure(order=["RoiA"])]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert any("at least two participants" in error for error in result["errors"])


def test_embedded_windows_drive_path_in_plot_cell_fails_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_strip())
    figure = _strip_figure(
        display_labels={"RoiA": r"result=C:\Data\example.tsv", "RoiB": "ROI B"}
    )

    result = plan_or_execute_mvpa_figure_export(
        _config(figures=[figure]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert any("plot-data rows contain an absolute local path" in error for error in result["errors"])


def test_execute_writes_vector_and_preview_outputs_with_editable_svg_text(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_strip())

    result = plan_or_execute_mvpa_figure_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    assert result["valid"] is True
    assert result["executed"] is True
    outputs = result["figures"][0]["outputs"]
    for key in ("figure_svg", "figure_pdf", "figure_png", "plot_data_tsv", "summary_tsv", "manifest_json"):
        assert Path(outputs[key]["path"]).is_file()
    svg = Path(outputs["figure_svg"]["path"]).read_text(encoding="utf-8")
    assert "<text" in svg
    assert "Encoding crossnobis" in svg
    plot_rows = _read_tsv(Path(outputs["plot_data_tsv"]["path"]))
    summary_rows = _read_tsv(Path(outputs["summary_tsv"]["path"]))
    manifest = json.loads(Path(outputs["manifest_json"]["path"]).read_text(encoding="utf-8"))
    assert "source_distances_relpath" not in plot_rows[0]
    assert summary_rows[0]["mean"] == "2.0"
    assert manifest["editable_text_settings"]["svg.fonttype"] == "none"
    assert manifest["layout_qc"]["status"] == "ok"
    assert manifest["layout_qc"]["warnings"] == []


def test_recognition_like_short_roi_plot_exports_with_zero_layout_warnings(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    artifact_root = tmp_path / "artifacts"
    _write_table(
        artifact_root,
        [
            {
                "participant_id": "sub-001",
                "phase_id": "recognition",
                "roi_label": "RecognitionPrecuneus",
                "contrast_id": "recog_minus_item",
                "crossnobis": 0.18,
            },
            {
                "participant_id": "sub-002",
                "phase_id": "recognition",
                "roi_label": "RecognitionPrecuneus",
                "contrast_id": "recog_minus_item",
                "crossnobis": 0.22,
            },
            {
                "participant_id": "sub-001",
                "phase_id": "recognition",
                "roi_label": "RecognitionFusiform",
                "contrast_id": "recog_minus_item",
                "crossnobis": 0.16,
            },
            {
                "participant_id": "sub-002",
                "phase_id": "recognition",
                "roi_label": "RecognitionFusiform",
                "contrast_id": "recog_minus_item",
                "crossnobis": 0.24,
            },
        ],
    )
    figure = _strip_figure(
        figure_id="roiwise_recognition_main",
        filters={"analysis_variant": "main", "phase_id": "recognition", "contrast_id": "recog_minus_item"},
        order=["RecognitionPrecuneus", "RecognitionFusiform"],
        display_labels={"RecognitionPrecuneus": "Precuneus", "RecognitionFusiform": "Temporal-occipital / fusiform / lingual"},
        title="Recognition crossnobis by LOSO ROI",
        output_basename="ses-01_task-memory_desc-RoiwiseRecognitionCrossnobis_mvpa",
        layout={"width_inches": 9.5, "height_inches": 6.0},
        x_tick_rotation=25,
    )

    result = plan_or_execute_mvpa_figure_export(
        _config(figures=[figure]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    manifest = json.loads(Path(result["figures"][0]["outputs"]["manifest_json"]["path"]).read_text(encoding="utf-8"))
    assert result["valid"] is True
    assert result["figures"][0]["row_counts"] == {"plot_data": 4, "summary": 2}
    assert manifest["layout_qc"]["status"] == "ok"
    assert manifest["layout_qc"]["warnings"] == []


def test_phase_summary_plot_exports_with_zero_layout_warnings_and_same_row_counts(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    artifact_root = tmp_path / "artifacts"
    _write_table(
        artifact_root,
        [
            {"participant_id": "sub-001", "phase_id": "encoding", "roi_label": "EncA", "contrast_id": "enc_minus_item", "crossnobis": 1.0},
            {"participant_id": "sub-001", "phase_id": "encoding", "roi_label": "EncB", "contrast_id": "enc_minus_item", "crossnobis": 3.0},
            {"participant_id": "sub-002", "phase_id": "encoding", "roi_label": "EncA", "contrast_id": "enc_minus_item", "crossnobis": 5.0},
            {"participant_id": "sub-002", "phase_id": "encoding", "roi_label": "EncB", "contrast_id": "enc_minus_item", "crossnobis": 7.0},
            {"participant_id": "sub-001", "phase_id": "recognition", "roi_label": "RecA", "contrast_id": "recog_minus_item", "crossnobis": 2.0},
            {"participant_id": "sub-001", "phase_id": "recognition", "roi_label": "RecB", "contrast_id": "recog_minus_item", "crossnobis": 4.0},
            {"participant_id": "sub-002", "phase_id": "recognition", "roi_label": "RecA", "contrast_id": "recog_minus_item", "crossnobis": 6.0},
            {"participant_id": "sub-002", "phase_id": "recognition", "roi_label": "RecB", "contrast_id": "recog_minus_item", "crossnobis": 8.0},
        ],
    )
    figure = _category_figure(
        filters={
            "analysis_variant": "main",
            "phase_id": ["encoding", "recognition"],
            "contrast_id": ["enc_minus_item", "recog_minus_item"],
        },
        order=["encoding", "recognition"],
        display_labels={"encoding": "Encoding", "recognition": "Recognition"},
        annotations={"show": True},
        layout={"width_inches": 9.5, "height_inches": 6.2},
    )

    result = plan_or_execute_mvpa_figure_export(
        _config(figures=[figure]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    manifest = json.loads(Path(result["figures"][0]["outputs"]["manifest_json"]["path"]).read_text(encoding="utf-8"))
    assert result["valid"] is True
    assert result["figures"][0]["row_counts"] == {"plot_data": 4, "summary": 2}
    assert [row["crossnobis"] for row in result["figures"][0]["plot_data_rows"]] == [2.0, 6.0, 3.0, 7.0]
    assert manifest["layout_qc"]["status"] == "ok"
    assert manifest["layout_qc"]["warnings"] == []


def test_execute_refuses_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_strip())
    config = _config()
    first = plan_or_execute_mvpa_figure_export(
        config,
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )
    second = plan_or_execute_mvpa_figure_export(
        config,
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    assert first["valid"] is True
    assert second["valid"] is False
    assert any("refuses to overwrite" in error for error in second["errors"])


def test_manifest_records_layout_warnings(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_strip())

    result = plan_or_execute_mvpa_figure_export(
        _config(figures=[_strip_figure(title="Very long title for layout warning", layout_qc={"max_title_chars": 5})]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    manifest_path = Path(result["figures"][0]["outputs"]["manifest_json"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["layout_qc"]["status"] == "warning"
    assert manifest["layout_qc"]["warnings"]
