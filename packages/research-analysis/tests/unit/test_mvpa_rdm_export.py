from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from research_platform.analysis.mvpa.rdm_export import (
    LONG_COLUMNS,
    MATRIX_FIRST_COLUMN,
    SUBJECT_PAIR_COLUMNS,
    _annotation_text_color_for_rgba,
    plan_or_execute_mvpa_rdm_export,
    validate_mvpa_rdm_export_document,
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
            "path": ".research-platform/mvpa/reports/{rdm_set}/rdms",
        },
        "rdms": [_two_condition_rdm()],
    }
    payload.update(overrides)
    return {"mvpa_rdm_export": payload}


def _two_condition_rdm(**overrides: object) -> dict[str, object]:
    rdm: dict[str, object] = {
        "rdm_id": "encoding_pair_item_main",
        "enabled": True,
        "kind": "rdm_heatmap",
        "value_column": "crossnobis",
        "filters": {
            "analysis_variant": "main",
            "phase_id": "encoding",
            "contrast_id": "pair_minus_item",
        },
        "conditions": [
            {"condition_id": "item", "label": "Item"},
            {"condition_id": "pair", "label": "Pair"},
        ],
        "pair_mappings": [
            {"contrast_id": "pair_minus_item", "condition_a": "item", "condition_b": "pair"},
        ],
        "aggregate_within_participant": {"enabled": True, "across": "roi_label", "method": "mean"},
        "group_summary": {"method": "mean", "ci_level": 0.95},
        "strict_all_pairs": True,
        "diagonal_value": 0.0,
        "symmetric": True,
        "title": "Encoding pair/item RDM",
        "colorbar_label": "Crossnobis",
        "annotate_cells": True,
        "annotation_decimals": 3,
        "output_basename": "ses-01_task-memory_desc-EncodingPairItemCrossnobisRDM_mvpa",
        "output_formats": ["svg", "pdf", "png"],
        "dpi": 120,
        "figure_size": {"width_inches": 4.8, "height_inches": 4.4},
    }
    rdm.update(overrides)
    return rdm


def _n_condition_rdm(condition_ids: list[str], *, value_column: str = "crossnobis") -> dict[str, object]:
    pairs: list[dict[str, str]] = []
    for left_index, condition_a in enumerate(condition_ids):
        for condition_b in condition_ids[left_index + 1 :]:
            pairs.append(
                {
                    "contrast_id": f"{condition_a}_minus_{condition_b}",
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                }
            )
    return _two_condition_rdm(
        rdm_id=f"{len(condition_ids)}condition",
        value_column=value_column,
        filters={"analysis_variant": "main"},
        conditions=[{"condition_id": condition_id, "label": condition_id.upper()} for condition_id in condition_ids],
        pair_mappings=pairs,
        aggregate_within_participant={"enabled": False, "method": "mean"},
        title=f"{len(condition_ids)} condition RDM",
        output_basename=f"ses-01_task-memory_desc-{len(condition_ids)}ConditionRDM_mvpa",
    )


def _write_table(
    artifact_root: Path,
    rows: list[dict[str, object]],
    *,
    table_set: str = "baseline_crossnobis",
    filename: str = "subject.tsv",
    columns: list[str] | None = None,
) -> Path:
    path = artifact_root / ".research-platform" / "mvpa" / "reports" / table_set / filename
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
        "distance_value",
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
                "distance_value": 1.0,
            }
            base.update(row)
            writer.writerow({column: base.get(column, "") for column in fieldnames})
    return path


def _rows_for_two_condition() -> list[dict[str, object]]:
    return [
        {"participant_id": "sub-001", "roi_label": "RoiA", "crossnobis": 1.0},
        {"participant_id": "sub-001", "roi_label": "RoiB", "crossnobis": 3.0},
        {"participant_id": "sub-002", "roi_label": "RoiA", "crossnobis": 5.0},
        {"participant_id": "sub-002", "roi_label": "RoiB", "crossnobis": 7.0},
        {"participant_id": "sub-001", "analysis_variant": "sensitivity", "roi_label": "RoiA", "crossnobis": 100.0},
    ]


def _rows_for_all_pairs(condition_ids: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    value = 1.0
    for participant_id, offset in (("sub-001", 0.0), ("sub-002", 0.5)):
        for left_index, condition_a in enumerate(condition_ids):
            for condition_b in condition_ids[left_index + 1 :]:
                rows.append(
                    {
                        "participant_id": participant_id,
                        "contrast_id": f"{condition_a}_minus_{condition_b}",
                        "crossnobis": value + offset,
                    }
                )
                value += 1.0
    return rows


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_config_validation_accepts_rdm_export_document() -> None:
    assert validate_mvpa_rdm_export_document(_config()) == []


def test_plan_mode_writes_nothing(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_two_condition())

    result = plan_or_execute_mvpa_rdm_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    assert result["executed"] is False
    assert result["rdms"][0]["row_counts"] == {"matrix": 2, "long": 1, "subject_pairs": 2, "summary": 1}
    assert result["rdms"][0]["subject_pair_rows"][0]["crossnobis"] == pytest.approx(2.0)
    assert not (artifact_root / ".research-platform" / "mvpa" / "reports" / "baseline_crossnobis" / "rdms").exists()


def test_auto_annotation_text_contrast_uses_light_on_dark_and_dark_on_bright_cells() -> None:
    assert _annotation_text_color_for_rgba((0.267004, 0.004874, 0.329415, 1.0)) == "#ffffff"
    assert _annotation_text_color_for_rgba((0.993248, 0.906157, 0.143936, 1.0)) == "#111827"


def test_execute_writes_rdm_outputs_with_editable_svg_text_and_symmetric_matrix(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_two_condition())

    result = plan_or_execute_mvpa_rdm_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    assert result["valid"] is True
    assert result["executed"] is True
    outputs = result["rdms"][0]["outputs"]
    for key in ("figure_svg", "figure_pdf", "figure_png", "matrix_tsv", "long_tsv", "subject_pairs_tsv", "summary_tsv", "manifest_json"):
        assert Path(outputs[key]["path"]).is_file()
    svg = Path(outputs["figure_svg"]["path"]).read_text(encoding="utf-8")
    assert "<text" in svg
    assert "#ffffff" in svg
    assert "#111827" in svg
    matrix = _read_tsv(Path(outputs["matrix_tsv"]["path"]))
    assert list(matrix[0])[0] == MATRIX_FIRST_COLUMN
    assert matrix[0]["Item"] == "0.0"
    assert matrix[0]["Pair"] == matrix[1]["Item"]
    long_rows = _read_tsv(Path(outputs["long_tsv"]["path"]))
    subject_rows = _read_tsv(Path(outputs["subject_pairs_tsv"]["path"]))
    manifest = json.loads(Path(outputs["manifest_json"]["path"]).read_text(encoding="utf-8"))
    assert list(long_rows[0]) == list(LONG_COLUMNS)
    assert list(subject_rows[0]) == list(SUBJECT_PAIR_COLUMNS)
    assert manifest["editable_text_settings"]["svg.fonttype"] == "none"
    assert manifest["layout_qc"]["status"] == "ok"
    assert manifest["absolute_source_paths_excluded"] is True


def test_recognition_like_two_condition_heatmap_has_no_layout_warnings(tmp_path: Path) -> None:
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
                "crossnobis": 0.1,
            },
            {
                "participant_id": "sub-002",
                "phase_id": "recognition",
                "roi_label": "RecognitionTemporalOccipitalFusiformLingual",
                "contrast_id": "recog_minus_item",
                "crossnobis": 0.2,
            },
        ],
    )
    rdm = _two_condition_rdm(
        rdm_id="recognition_pair_item_main",
        filters={"analysis_variant": "main", "phase_id": "recognition", "contrast_id": "recog_minus_item"},
        conditions=[
            {"condition_id": "item_recog_correct", "label": "Item recognition correct"},
            {"condition_id": "pair_recog_correct", "label": "Pair recognition correct"},
        ],
        pair_mappings=[
            {
                "contrast_id": "recog_minus_item",
                "condition_a": "item_recog_correct",
                "condition_b": "pair_recog_correct",
            }
        ],
        title="Recognition pair/item crossnobis RDM",
        colorbar_label="Crossnobis distance",
        annotation_text_color="auto",
        title_fontsize=11,
        title_pad=14,
        colorbar_shrink=0.8,
        colorbar_pad=0.08,
        figure_size={"width_inches": 6.2, "height_inches": 5.4},
        output_basename="ses-01_task-memory_desc-RecognitionPairItemCrossnobisRDM_mvpa",
    )

    result = plan_or_execute_mvpa_rdm_export(
        _config(rdms=[rdm]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    manifest = json.loads(Path(result["rdms"][0]["outputs"]["manifest_json"]["path"]).read_text(encoding="utf-8"))
    assert result["valid"] is True
    assert result["rdms"][0]["row_counts"] == {"matrix": 2, "long": 1, "subject_pairs": 2, "summary": 1}
    assert manifest["layout_qc"]["status"] == "ok"
    assert manifest["layout_qc"]["warnings"] == []


def test_four_condition_heatmap_with_long_labels_has_no_layout_warnings(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_all_pairs(["item_enc_hit", "pair_enc_hit", "item_recog_correct", "pair_recog_correct"]))
    rdm = _n_condition_rdm(["item_enc_hit", "pair_enc_hit", "item_recog_correct", "pair_recog_correct"])
    rdm.update(
        {
            "conditions": [
                {"condition_id": "item_enc_hit", "label": "Item encoding hit"},
                {"condition_id": "pair_enc_hit", "label": "Pair encoding hit"},
                {"condition_id": "item_recog_correct", "label": "Item recognition correct"},
                {"condition_id": "pair_recog_correct", "label": "Pair recognition correct"},
            ],
            "title": "Encoding/recognition four-condition crossnobis RDM",
            "colorbar_label": "Crossnobis distance",
            "annotation_text_color": "auto",
            "title_fontsize": 11,
            "title_pad": 16,
            "colorbar_shrink": 0.78,
            "colorbar_pad": 0.10,
            "x_tick_rotation": 35,
            "fail_on_layout_warning": True,
            "figure_size": {"width_inches": 8.0, "height_inches": 6.8},
            "output_basename": "ses-01_task-memory_desc-EncodingRecognitionFourConditionCrossnobisRDM_mvpa",
        }
    )

    result = plan_or_execute_mvpa_rdm_export(
        _config(rdms=[rdm]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    manifest = json.loads(Path(result["rdms"][0]["outputs"]["manifest_json"]["path"]).read_text(encoding="utf-8"))
    assert result["valid"] is True
    assert result["rdms"][0]["row_counts"] == {"matrix": 4, "long": 6, "subject_pairs": 12, "summary": 6}
    assert manifest["layout_qc"]["status"] == "ok"
    assert manifest["layout_qc"]["warnings"] == []


def test_generic_three_condition_complete_rdm_works(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_all_pairs(["alpha", "beta", "gamma"]))

    result = plan_or_execute_mvpa_rdm_export(
        _config(rdms=[_n_condition_rdm(["alpha", "beta", "gamma"])]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    assert result["rdms"][0]["row_counts"]["matrix"] == 3
    assert result["rdms"][0]["row_counts"]["long"] == 3
    assert result["rdms"][0]["row_counts"]["subject_pairs"] == 6


def test_generic_four_condition_complete_rdm_works(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_all_pairs(["a", "b", "c", "d"]))

    result = plan_or_execute_mvpa_rdm_export(
        _config(rdms=[_n_condition_rdm(["a", "b", "c", "d"])]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    assert result["rdms"][0]["row_counts"]["matrix"] == 4
    assert result["rdms"][0]["row_counts"]["long"] == 6
    assert result["rdms"][0]["row_counts"]["subject_pairs"] == 12
    matrix_rows = result["rdms"][0]["matrix_rows"]
    assert [row[MATRIX_FIRST_COLUMN] for row in matrix_rows] == ["a", "b", "c", "d"]
    assert matrix_rows[0]["A"] == 0.0
    assert matrix_rows[0]["B"] == matrix_rows[1]["A"]
    assert matrix_rows[2]["D"] == matrix_rows[3]["C"]


def test_four_condition_rdm_can_use_per_rdm_input_table(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_two_condition())
    _write_table(
        artifact_root,
        _rows_for_all_pairs(["a", "b", "c", "d"]),
        table_set="baseline_crossnobis_allpairs",
        filename="allpairs.tsv",
    )
    four_condition = _n_condition_rdm(["a", "b", "c", "d"])
    four_condition["input"] = {
        "table_set": "baseline_crossnobis_allpairs",
        "root_ref": "artifact_root",
        "path": ".research-platform/mvpa/reports/{table_set}/allpairs.tsv",
    }

    result = plan_or_execute_mvpa_rdm_export(
        _config(rdms=[_two_condition_rdm(), four_condition]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    assert result["rdms"][0]["input_table"]["relative_path"] == ".research-platform/mvpa/reports/baseline_crossnobis/subject.tsv"
    assert result["rdms"][1]["input_table"]["relative_path"] == ".research-platform/mvpa/reports/baseline_crossnobis_allpairs/allpairs.tsv"
    assert result["rdms"][0]["row_counts"] == {"matrix": 2, "long": 1, "subject_pairs": 2, "summary": 1}
    assert result["rdms"][1]["row_counts"] == {"matrix": 4, "long": 6, "subject_pairs": 12, "summary": 6}


def test_strict_all_pairs_fails_when_required_pair_mapping_is_missing(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_all_pairs(["alpha", "beta", "gamma"]))
    rdm = _n_condition_rdm(["alpha", "beta", "gamma"])
    rdm["pair_mappings"] = [rdm["pair_mappings"][0]]  # type: ignore[index]

    result = plan_or_execute_mvpa_rdm_export(
        _config(rdms=[rdm]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert any("strict_all_pairs" in error for error in result["errors"])


def test_within_participant_aggregation_averages_roi_only_within_participant(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_two_condition())

    result = plan_or_execute_mvpa_rdm_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    subject_values = [row["crossnobis"] for row in result["rdms"][0]["subject_pair_rows"]]
    assert subject_values == [2.0, 6.0]
    assert result["rdms"][0]["subject_pair_rows"][0]["pooled_roi_count"] == 2
    assert result["rdms"][0]["long_rows"][0]["group_mean_crossnobis"] == pytest.approx(4.0)
    assert result["rdms"][0]["long_rows"][0]["n"] == 2


def test_filters_select_rows_and_custom_value_column_is_supported(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(
        artifact_root,
        [
            {"participant_id": "sub-001", "crossnobis": "bad", "distance_value": 1.0},
            {"participant_id": "sub-002", "crossnobis": "bad", "distance_value": 3.0},
            {"participant_id": "sub-003", "analysis_variant": "sensitivity", "crossnobis": "bad", "distance_value": 99.0},
        ],
    )

    result = plan_or_execute_mvpa_rdm_export(
        _config(rdms=[_two_condition_rdm(value_column="distance_value")]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    assert [row["participant_id"] for row in result["rdms"][0]["subject_pair_rows"]] == ["sub-001", "sub-002"]
    assert result["rdms"][0]["long_rows"][0]["group_mean_crossnobis"] == pytest.approx(2.0)


def test_missing_required_columns_fail_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, [{"participant_id": "sub-001"}], columns=["participant_id", "contrast_id", "crossnobis"])

    result = plan_or_execute_mvpa_rdm_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert any("missing required column" in error for error in result["errors"])


def test_nonnumeric_value_column_fails_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, [{"participant_id": "sub-001", "crossnobis": "bad"}, {"participant_id": "sub-002"}])

    result = plan_or_execute_mvpa_rdm_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert any("crossnobis must be finite numeric" in error for error in result["errors"])


def test_too_few_participants_fail_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, [{"participant_id": "sub-001", "crossnobis": 1.0}])

    result = plan_or_execute_mvpa_rdm_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert any("at least two participants" in error for error in result["errors"])


def test_embedded_unc_path_in_export_cell_fails_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_two_condition())
    rdm = _two_condition_rdm(
        conditions=[
            {"condition_id": "item", "label": "Item"},
            {
                "condition_id": "pair",
                "label": r"source \\cluster.example\example-share\data.tsv",
            },
        ]
    )

    result = plan_or_execute_mvpa_rdm_export(
        _config(rdms=[rdm]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert any("export rows contain an absolute local path" in error for error in result["errors"])


def test_execute_refuses_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_two_condition())
    config = _config()

    first = plan_or_execute_mvpa_rdm_export(config, workspace_root=tmp_path, root_refs={"artifact_root": artifact_root}, execute=True)
    second = plan_or_execute_mvpa_rdm_export(config, workspace_root=tmp_path, root_refs={"artifact_root": artifact_root}, execute=True)

    assert first["valid"] is True
    assert second["valid"] is False
    assert any("refuses to overwrite" in error for error in second["errors"])


def test_manifest_records_layout_warnings(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    artifact_root = tmp_path / "artifacts"
    _write_table(artifact_root, _rows_for_two_condition())

    result = plan_or_execute_mvpa_rdm_export(
        _config(rdms=[_two_condition_rdm(title="Very long title for layout warning", layout_qc={"max_title_chars": 5})]),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    manifest = json.loads(Path(result["rdms"][0]["outputs"]["manifest_json"]["path"]).read_text(encoding="utf-8"))
    assert manifest["layout_qc"]["status"] == "warning"
    assert manifest["layout_qc"]["warnings"]
