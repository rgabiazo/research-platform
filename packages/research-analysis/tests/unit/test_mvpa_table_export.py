from __future__ import annotations

import csv
import json
from pathlib import Path

from research_platform.analysis.mvpa.table_export import (
    AUDIT_COLUMNS,
    TABLE_A_BASE_COLUMNS,
    plan_or_execute_mvpa_table_export,
    validate_mvpa_table_export_document,
)


def _config(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "baseline_crossnobis",
        "entities": {"session_id": "ses-01", "task_id": "memory"},
        "sources": [
            {
                "mvpa_set": "encoding_main",
                "phase_id": "encoding",
                "analysis_variant": "main",
                "family_id": "primary_main",
                "expected_rows": 2,
            },
            {
                "mvpa_set": "recognition_main",
                "phase_id": "recognition",
                "analysis_variant": "main",
                "family_id": "primary_main",
                "expected_rows": 2,
            },
        ],
        "expected": {"total_rows": 4, "participant_count": 2},
        "outputs": {
            "root_ref": "artifact_root",
            "path": ".research-platform/mvpa/reports/{table_set}",
            "filename_prefix": "ses-01_task-memory",
        },
    }
    payload.update(overrides)
    return {"mvpa_table_export": payload}


def _write_source(
    artifact_root: Path,
    mvpa_set: str,
    rows: list[dict[str, object]],
) -> Path:
    path = artifact_root / ".research-platform" / "mvpa" / mvpa_set / "analysis" / "prepared-distances" / "distances.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "group_id",
        "group_key",
        "condition_id_a",
        "condition_id_b",
        "condition_pair_id",
        "distance",
        "metric",
        "engine_name",
        "normalization_method",
        "cv_unit_count",
        "feature_count",
        "observation_count",
        "context",
        "subject_id",
        "session_id",
        "task_id",
        "roi_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            base = {
                "group_id": "group",
                "group_key": "{}",
                "condition_id_a": "pair",
                "condition_id_b": "item",
                "condition_pair_id": "pair_minus_item",
                "distance": 0.25,
                "metric": "crossnobis",
                "engine_name": "manual_diagonal_crossnobis_v1",
                "normalization_method": "diagonal",
                "cv_unit_count": 3,
                "feature_count": 12,
                "observation_count": 6,
                "context": "{}",
                "session_id": "ses-01",
                "task_id": "memory",
                "roi_label": "RoiA",
            }
            base.update(row)
            writer.writerow(base)
    return path


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_config_validation_accepts_table_export_document() -> None:
    assert validate_mvpa_table_export_document(_config()) == []


def test_config_validation_accepts_output_filename_overrides() -> None:
    config = _config(
        outputs={
            "root_ref": "artifact_root",
            "path": ".research-platform/mvpa/reports/{table_set}",
            "filename_prefix": "ses-01_task-memory",
            "filenames": {
                "subject_level_distances": "ses-01_task-memory_desc-SubjectLevelAllPairsCrossnobisDistances_mvpa.tsv",
                "subject_level_audit": "ses-01_task-memory_desc-SubjectLevelAllPairsCrossnobisAudit_mvpa.tsv",
                "manifest": "ses-01_task-memory_desc-AllPairsCrossnobisTables_manifest.json",
            },
        }
    )

    assert validate_mvpa_table_export_document(config) == []


def test_config_validation_rejects_output_filename_paths() -> None:
    config = _config(
        outputs={
            "root_ref": "artifact_root",
            "path": ".research-platform/mvpa/reports/{table_set}",
            "filenames": {"subject_level_distances": "nested/table.tsv"},
        }
    )

    assert any("must be a filename" in error for error in validate_mvpa_table_export_document(config))


def test_plan_builds_lean_and_audit_tables_without_writing(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_source(
        artifact_root,
        "encoding_main",
        [{"subject_id": "sub-001", "distance": 0.1}, {"subject_id": "sub-002", "distance": 0.2}],
    )
    _write_source(
        artifact_root,
        "recognition_main",
        [{"subject_id": "sub-001", "distance": 0.3}, {"subject_id": "sub-002", "distance": 0.4}],
    )

    result = plan_or_execute_mvpa_table_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    assert result["executed"] is False
    assert result["table_a_columns"] == list(TABLE_A_BASE_COLUMNS)
    assert result["audit_table_columns"] == list(AUDIT_COLUMNS)
    assert result["row_counts"] == {"table_a": 4, "audit": 4, "participants": 2}
    assert "session_id" not in result["table_a_columns"]
    assert "task_id" not in result["table_a_columns"]
    assert result["manifest"]["invariant_entities_moved_out_of_table_a"] == {
        "session_id": "ses-01",
        "task_id": "memory",
    }
    assert not any(Path(output["path"]).exists() for output in result["outputs"].values())


def test_execute_writes_table_a_audit_and_manifest_with_relative_source_paths(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_source(
        artifact_root,
        "encoding_main",
        [{"subject_id": "sub-001", "distance": 0.1}, {"subject_id": "sub-002", "distance": 0.2}],
    )
    _write_source(
        artifact_root,
        "recognition_main",
        [{"subject_id": "sub-001", "distance": 0.3}, {"subject_id": "sub-002", "distance": 0.4}],
    )

    result = plan_or_execute_mvpa_table_export(
        _config(),
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    assert result["valid"] is True
    table_a_path = Path(result["outputs"]["subject_level_distances"]["path"])
    audit_path = Path(result["outputs"]["subject_level_audit"]["path"])
    manifest_path = Path(result["outputs"]["manifest"]["path"])
    table_rows = _read_tsv(table_a_path)
    audit_rows = _read_tsv(audit_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert list(table_rows[0]) == list(TABLE_A_BASE_COLUMNS)
    assert "source_distances_relpath" not in table_rows[0]
    assert list(audit_rows[0]) == list(AUDIT_COLUMNS)
    assert not audit_rows[0]["source_distances_relpath"].startswith("/")
    assert manifest["row_counts"]["subject_level"] == 4
    assert manifest["absolute_source_paths_excluded"] is True


def test_execute_uses_configured_output_filenames(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_source(
        artifact_root,
        "encoding_main",
        [{"subject_id": "sub-001", "distance": 0.1}, {"subject_id": "sub-002", "distance": 0.2}],
    )
    config = _config(
        sources=[
            {
                "mvpa_set": "encoding_main",
                "phase_id": "encoding_recognition",
                "analysis_variant": "main",
                "family_id": "allpairs_main",
                "expected_rows": 2,
            }
        ],
        expected={"total_rows": 2, "participant_count": 2},
        outputs={
            "root_ref": "artifact_root",
            "path": ".research-platform/mvpa/reports/{table_set}",
            "filename_prefix": "ses-01_task-memory",
            "filenames": {
                "subject_level_distances": "ses-01_task-memory_desc-SubjectLevelAllPairsCrossnobisDistances_mvpa.tsv",
                "subject_level_audit": "ses-01_task-memory_desc-SubjectLevelAllPairsCrossnobisAudit_mvpa.tsv",
                "manifest": "ses-01_task-memory_desc-AllPairsCrossnobisTables_manifest.json",
            },
        },
    )

    result = plan_or_execute_mvpa_table_export(
        config,
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    assert result["valid"] is True
    assert result["outputs"]["subject_level_distances"]["path"].endswith(
        "ses-01_task-memory_desc-SubjectLevelAllPairsCrossnobisDistances_mvpa.tsv"
    )
    assert Path(result["outputs"]["subject_level_distances"]["path"]).is_file()
    assert (
        result["manifest"]["outputs"]["manifest"]["filename"]
        == "ses-01_task-memory_desc-AllPairsCrossnobisTables_manifest.json"
    )


def test_participant_id_is_bids_formatted_without_changing_audit_subject_id(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_source(
        artifact_root,
        "encoding_main",
        [
            {"subject_id": "002", "roi_label": "RoiA", "distance": 0.1},
            {"subject_id": "2", "roi_label": "RoiB", "distance": 0.2},
            {"subject_id": "sub-004", "roi_label": "RoiC", "distance": 0.3},
        ],
    )
    config = _config(
        sources=[
            {
                "mvpa_set": "encoding_main",
                "phase_id": "encoding",
                "analysis_variant": "main",
                "family_id": "primary_main",
                "expected_rows": 3,
            }
        ],
        expected={"total_rows": 3, "participant_count": 2},
    )

    result = plan_or_execute_mvpa_table_export(
        config,
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
        execute=True,
    )

    assert result["valid"] is True
    table_rows = _read_tsv(Path(result["outputs"]["subject_level_distances"]["path"]))
    audit_rows = _read_tsv(Path(result["outputs"]["subject_level_audit"]["path"]))
    assert list(table_rows[0]) == list(TABLE_A_BASE_COLUMNS)
    assert "session_id" not in table_rows[0]
    assert "task_id" not in table_rows[0]
    assert [row["participant_id"] for row in table_rows] == ["sub-002", "sub-002", "sub-004"]
    assert [row["participant_id"] for row in audit_rows] == ["sub-002", "sub-002", "sub-004"]
    assert [row["subject_id"] for row in audit_rows] == ["002", "2", "sub-004"]
    assert all(row["participant_id"].startswith("sub-") for row in table_rows)
    assert all(row["participant_id"].removeprefix("sub-").isdigit() for row in table_rows)
    assert all(len(row["participant_id"].removeprefix("sub-")) == 3 for row in table_rows)


def test_two_session_export_includes_session_without_flagging_cross_session_duplicates(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_source(
        artifact_root,
        "encoding_main",
        [
            {"subject_id": "sub-001", "session_id": "ses-01", "distance": 0.1},
            {"subject_id": "sub-001", "session_id": "ses-02", "distance": 0.2},
        ],
    )
    config = _config(
        sources=[
            {
                "mvpa_set": "encoding_main",
                "phase_id": "encoding",
                "analysis_variant": "main",
                "family_id": "primary_main",
                "expected_rows": 2,
            }
        ],
        expected={"total_rows": 2, "participant_count": 1},
    )

    result = plan_or_execute_mvpa_table_export(
        config,
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    assert result["table_a_columns"] == [
        "participant_id",
        "session_id",
        "analysis_variant",
        "phase_id",
        "roi_label",
        "contrast_id",
        "crossnobis",
        "feature_count",
        "cv_unit_count",
        "observation_count",
    ]
    assert result["manifest"]["invariant_entities_moved_out_of_table_a"] == {"task_id": "memory"}
    assert result["row_counts"] == {"table_a": 2, "audit": 2, "participants": 1}


def test_two_session_export_fails_duplicates_within_same_session_key(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_source(
        artifact_root,
        "encoding_main",
        [
            {"subject_id": "sub-001", "session_id": "ses-01", "distance": 0.1},
            {"subject_id": "sub-001", "session_id": "ses-01", "distance": 0.2},
            {"subject_id": "sub-001", "session_id": "ses-02", "distance": 0.3},
        ],
    )
    config = _config(
        sources=[
            {
                "mvpa_set": "encoding_main",
                "phase_id": "encoding",
                "analysis_variant": "main",
                "family_id": "primary_main",
            }
        ],
        expected={"total_rows": 3, "participant_count": 1},
    )

    result = plan_or_execute_mvpa_table_export(
        config,
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert "session_id" in result["table_a_columns"]
    assert any("duplicate" in error for error in result["errors"])


def test_varying_task_is_retained_in_table_a_when_session_is_invariant(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_source(
        artifact_root,
        "encoding_main",
        [
            {"subject_id": "sub-001", "task_id": "memory", "distance": 0.1},
            {"subject_id": "sub-001", "task_id": "localizer", "distance": 0.2},
        ],
    )
    config = _config(
        sources=[
            {
                "mvpa_set": "encoding_main",
                "phase_id": "encoding",
                "analysis_variant": "main",
                "family_id": "primary_main",
                "expected_rows": 2,
            }
        ],
        expected={"total_rows": 2, "participant_count": 1},
    )

    result = plan_or_execute_mvpa_table_export(
        config,
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is True
    assert result["table_a_columns"] == [
        "participant_id",
        "task_id",
        "analysis_variant",
        "phase_id",
        "roi_label",
        "contrast_id",
        "crossnobis",
        "feature_count",
        "cv_unit_count",
        "observation_count",
    ]
    assert result["manifest"]["invariant_entities_moved_out_of_table_a"] == {"session_id": "ses-01"}


def test_duplicate_subject_variant_phase_roi_contrast_fails_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_source(
        artifact_root,
        "encoding_main",
        [{"subject_id": "sub-001", "distance": 0.1}, {"subject_id": "sub-001", "distance": 0.2}],
    )
    config = _config(
        sources=[
            {
                "mvpa_set": "encoding_main",
                "phase_id": "encoding",
                "analysis_variant": "main",
                "family_id": "primary_main",
            }
        ],
        expected={"total_rows": 2, "participant_count": 1},
    )

    result = plan_or_execute_mvpa_table_export(
        config,
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert any("duplicate" in error for error in result["errors"])


def test_nonnumeric_distance_fails_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_source(artifact_root, "encoding_main", [{"subject_id": "sub-001", "distance": "bad"}])
    config = _config(
        sources=[
            {
                "mvpa_set": "encoding_main",
                "phase_id": "encoding",
                "analysis_variant": "main",
                "family_id": "primary_main",
            }
        ],
        expected={},
    )

    result = plan_or_execute_mvpa_table_export(
        config,
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert any("distance must be finite numeric" in error for error in result["errors"])


def test_json_escaped_windows_path_in_table_cell_fails_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_source(
        artifact_root,
        "encoding_main",
        [
            {
                "subject_id": "sub-001",
                "engine_name": r'{"path":"C:\\Data\\example.tsv"}',
                "distance": 0.1,
            }
        ],
    )
    config = _config(
        sources=[
            {
                "mvpa_set": "encoding_main",
                "phase_id": "encoding",
                "analysis_variant": "main",
                "family_id": "primary_main",
            }
        ],
        expected={},
    )

    result = plan_or_execute_mvpa_table_export(
        config,
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert any("Table A contains an absolute local path" in error for error in result["errors"])


def test_expected_row_mismatch_fails_cleanly(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _write_source(artifact_root, "encoding_main", [{"subject_id": "sub-001", "distance": 0.1}])
    config = _config(
        sources=[
            {
                "mvpa_set": "encoding_main",
                "phase_id": "encoding",
                "analysis_variant": "main",
                "family_id": "primary_main",
                "expected_rows": 2,
            }
        ],
        expected={"total_rows": 2, "participant_count": 1},
    )

    result = plan_or_execute_mvpa_table_export(
        config,
        workspace_root=tmp_path,
        root_refs={"artifact_root": artifact_root},
    )

    assert result["valid"] is False
    assert "Expected 2 subject-level row(s), found 1." in result["errors"]
    assert "Source encoding_main expected 2 row(s), found 1." in result["errors"]
