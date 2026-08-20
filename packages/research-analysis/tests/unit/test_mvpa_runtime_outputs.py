from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path

import pytest

import research_platform.analysis.mvpa as mvpa
import research_platform.analysis.mvpa.runtime_outputs as runtime_outputs
from research_platform.analysis.mvpa import (
    MvpaPatternRowPreparationProvenanceRow,
    MvpaPatternRowPreparationQcRow,
    MvpaPatternRowPreparationResult,
    PreparedMvpaDistanceProvenanceRow,
    PreparedMvpaDistanceQcRow,
    PreparedMvpaDistanceResult,
    PreparedMvpaDistanceRow,
    PreparedMvpaDistanceSummaryProvenanceRow,
    PreparedMvpaDistanceSummaryQcRow,
    PreparedMvpaDistanceSummaryResult,
    PreparedMvpaPatternGroup,
    PreparedMvpaPatternRow,
    plan_prepared_mvpa_distance_outputs,
    plan_prepared_mvpa_pattern_outputs,
    plan_prepared_mvpa_summary_outputs,
    write_prepared_mvpa_distance_outputs,
    write_prepared_mvpa_pattern_outputs,
    write_prepared_mvpa_summary_outputs,
)


_PATTERN_ROW_HEADER = [
    "group_id",
    "group_key",
    "group_by",
    "group_cv_unit",
    "group_cv_labels",
    "group_condition_ids",
    "group_feature_count",
    "group_voxel_order",
    "group_voxel_index_hash",
    "pattern_id",
    "condition_id",
    "cv_unit",
    "cv_label",
    "feature_values",
    "feature_count",
    "source_row_index",
    "voxel_order",
    "voxel_index_hash",
    "mean_centering_applied",
    "mean_centering_scope",
    "event_count",
    "noise_values",
    "noise_usable",
    "noise_feature_count",
    "noise_voxel_index_hash",
]
_PATTERN_QC_HEADER = [
    "level",
    "status",
    "code",
    "message",
    "source_row_index",
    "group_id",
    "group_key",
    "pattern_id",
    "condition_id",
    "cv_unit",
    "cv_label",
    "usable",
    "context",
]
_DISTANCE_HEADER = [
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
]
_DISTANCE_QC_HEADER = [
    "level",
    "status",
    "code",
    "message",
    "source",
    "source_row_index",
    "group_id",
    "group_key",
    "pattern_id",
    "condition_id",
    "condition_id_a",
    "condition_id_b",
    "condition_pair_id",
    "cv_unit",
    "cv_label",
    "usable",
    "context",
]
_SUMMARY_HEADER = [
    "group_id",
    "condition_id_a",
    "condition_id_b",
    "metric",
    "engine_name",
    "normalization_method",
    "n",
    "mean_distance",
    "std_distance",
    "sem_distance",
    "min_distance",
    "max_distance",
]
_SUMMARY_QC_HEADER = [
    "level",
    "status",
    "code",
    "message",
    "source",
    "source_row_index",
    "group_id",
    "group_key",
    "condition_id_a",
    "condition_id_b",
    "field_name",
    "context",
]


def _pattern_result() -> MvpaPatternRowPreparationResult:
    group_key = {"task_id": "localizer", "subject_id": "sub-01"}
    row = PreparedMvpaPatternRow(
        pattern_id="p1",
        condition_id="face",
        cv_unit="run",
        cv_label="run-1",
        feature_values=[1.0, 2.5],
        feature_count=2,
        group_key=group_key,
        source_row_index=7,
        voxel_order="c_flat_index",
        voxel_index_hash="hash-a",
        noise_values=[0.25, 0.5],
        noise_usable=True,
        noise_feature_count=2,
        noise_voxel_index_hash="hash-a",
    )
    group = PreparedMvpaPatternGroup(
        group_id="g1",
        group_key=group_key,
        group_by=("subject_id", "task_id"),
        rows=(row,),
        cv_unit="run",
        cv_labels=("run-1",),
        condition_ids=("face",),
        feature_count=2,
        voxel_order="c_flat_index",
        voxel_index_hash="hash-a",
    )
    qc = MvpaPatternRowPreparationQcRow(
        level="noise",
        status="warning",
        code="invalid_noise_values",
        message="Synthetic warning.",
        source_row_index=7,
        group_id="g1",
        group_key=group_key,
        pattern_id="p1",
        condition_id="face",
        cv_unit="run",
        cv_label="run-1",
        usable=True,
    )
    return MvpaPatternRowPreparationResult(
        groups=(group,),
        qc_rows=(qc,),
        provenance=(
            MvpaPatternRowPreparationProvenanceRow("phase", "3B.1"),
            MvpaPatternRowPreparationProvenanceRow("output_written", False),
        ),
        warnings=("Synthetic warning.",),
        errors=(),
    )


def _distance_result() -> PreparedMvpaDistanceResult:
    group_key = {"task_id": "localizer", "subject_id": "sub-01"}
    distance = PreparedMvpaDistanceRow(
        group_id="g1",
        group_key=group_key,
        condition_id_a="face",
        condition_id_b="house",
        distance=1.25,
        metric="crossnobis",
        engine_name="native_reference",
        normalization_method="identity",
        cv_unit_count=2,
        feature_count=2,
        observation_count=4,
        context={"labels": ["face", "house"], "condition_index_a": 0},
    )
    qc = PreparedMvpaDistanceQcRow(
        level="group",
        status="warning",
        code="synthetic_distance_warning",
        message="Synthetic distance warning.",
        group_id="g1",
        group_key=group_key,
        condition_id_a="face",
        condition_id_b="house",
        cv_unit="run",
        usable=True,
        context={"thresholds": [1, 2], "phase": "3B.4"},
    )
    return PreparedMvpaDistanceResult(
        distances=(distance,),
        qc_rows=(qc,),
        provenance=(
            PreparedMvpaDistanceProvenanceRow("phase", "3B.4"),
            PreparedMvpaDistanceProvenanceRow("distance_computation", True),
            PreparedMvpaDistanceProvenanceRow("output_written", False),
        ),
        warnings=("Synthetic distance warning.",),
        errors=(),
    )


def _summary_result() -> PreparedMvpaDistanceSummaryResult:
    qc = PreparedMvpaDistanceSummaryQcRow(
        level="summary",
        status="warning",
        code="synthetic_summary_warning",
        message="Synthetic summary warning.",
        group_id="group-alpha",
        group_key={"analysis_unit": "unit-a"},
        condition_id_a="condition-a",
        condition_id_b="condition-b",
        field_name="metric",
        context={"distinct_value_count": 1},
    )
    return PreparedMvpaDistanceSummaryResult(
        summary_rows=(
            {
                "group_id": "group-alpha",
                "condition_id_a": "condition-a",
                "condition_id_b": "condition-b",
                "metric": "crossnobis",
                "engine_name": "native_reference",
                "normalization_method": "identity",
                "n": 2,
                "mean_distance": 1.5,
                "std_distance": 0.5,
                "sem_distance": 0.35355339059327373,
                "min_distance": 1.0,
                "max_distance": 2.0,
            },
        ),
        qc_rows=(qc,),
        provenance=(
            PreparedMvpaDistanceSummaryProvenanceRow("phase", "3D.1"),
            PreparedMvpaDistanceSummaryProvenanceRow("output_written", False),
        ),
        warnings=("Synthetic summary warning.",),
        errors=(),
    )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _header(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()[0].split("\t")


def test_prepared_pattern_outputs_write_rows_qc_and_deterministic_columns(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    rows_path = "mvpa/pattern_rows.tsv"
    qc_path = "mvpa/pattern_qc.tsv"
    provenance_path = "mvpa/pattern_provenance.json"

    assert not (output_root / "mvpa").exists()

    record = write_prepared_mvpa_pattern_outputs(
        _pattern_result(),
        output_root=output_root,
        rows_path=rows_path,
        qc_path=qc_path,
        provenance_path=provenance_path,
    )

    assert (output_root / "mvpa").is_dir()
    assert record["will_write"] is True
    assert _header(output_root / rows_path) == [*_PATTERN_ROW_HEADER, "subject_id", "task_id"]
    assert _header(output_root / qc_path) == _PATTERN_QC_HEADER

    rows = _read_tsv(output_root / rows_path)
    assert len(rows) == 1
    assert rows[0]["group_key"] == '{"subject_id":"sub-01","task_id":"localizer"}'
    assert rows[0]["subject_id"] == "sub-01"
    assert rows[0]["task_id"] == "localizer"
    assert rows[0]["group_by"] == '["subject_id","task_id"]'
    assert rows[0]["group_cv_labels"] == '["run-1"]'
    assert rows[0]["feature_values"] == "[1.0,2.5]"
    assert rows[0]["noise_values"] == "[0.25,0.5]"
    assert rows[0]["noise_usable"] == "true"

    qc_rows = _read_tsv(output_root / qc_path)
    assert len(qc_rows) == 1
    assert qc_rows[0]["code"] == "invalid_noise_values"
    assert qc_rows[0]["group_key"] == '{"subject_id":"sub-01","task_id":"localizer"}'


def test_prepared_distance_outputs_write_distances_qc_and_compact_context(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    distances_path = "mvpa/distances.tsv"
    qc_path = "mvpa/distance_qc.tsv"
    provenance_path = "mvpa/distance_provenance.json"

    record = write_prepared_mvpa_distance_outputs(
        _distance_result(),
        output_root=output_root,
        distances_path=distances_path,
        qc_path=qc_path,
        provenance_path=provenance_path,
    )

    assert record["will_write"] is True
    assert _header(output_root / distances_path) == [*_DISTANCE_HEADER, "subject_id", "task_id"]
    assert _header(output_root / qc_path) == _DISTANCE_QC_HEADER

    distances = _read_tsv(output_root / distances_path)
    assert len(distances) == 1
    assert distances[0]["distance"] == "1.25"
    assert distances[0]["group_key"] == '{"subject_id":"sub-01","task_id":"localizer"}'
    assert distances[0]["subject_id"] == "sub-01"
    assert distances[0]["task_id"] == "localizer"
    assert distances[0]["context"] == '{"condition_index_a":0,"labels":["face","house"]}'

    qc_rows = _read_tsv(output_root / qc_path)
    assert len(qc_rows) == 1
    assert qc_rows[0]["code"] == "synthetic_distance_warning"
    assert qc_rows[0]["context"] == '{"phase":"3B.4","thresholds":[1,2]}'


def test_distance_provenance_json_uses_relative_paths_and_input_provenance(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"

    write_prepared_mvpa_distance_outputs(
        _distance_result(),
        output_root=output_root,
        distances_path="mvpa/distances.tsv",
        qc_path="mvpa/distance_qc.tsv",
        provenance_path="mvpa/distance_provenance.json",
    )

    provenance_path = output_root / "mvpa/distance_provenance.json"
    text = provenance_path.read_text(encoding="utf-8")
    provenance = json.loads(text)

    assert str(tmp_path) not in text
    assert provenance["schema_version"] == runtime_outputs.SCHEMA_VERSION
    assert provenance["artifact_kind"] == "prepared_mvpa_distance_outputs"
    assert provenance["writer_module"] == "research_platform.analysis.mvpa.runtime_outputs"
    assert provenance["output_written"] is True
    assert provenance["output_paths"] == {
        "distances": "mvpa/distances.tsv",
        "qc": "mvpa/distance_qc.tsv",
        "provenance": "mvpa/distance_provenance.json",
    }
    assert provenance["row_counts"] == {"distances": 1, "qc": 1}
    assert provenance["distance_row_count"] == 1
    assert provenance["qc_row_count"] == 1
    assert provenance["columns"]["distances"] == [*_DISTANCE_HEADER, "subject_id", "task_id"]
    assert provenance["columns"]["qc"] == _DISTANCE_QC_HEADER
    assert provenance["input_provenance"]["phase"] == "3B.4"
    assert provenance["input_provenance"]["output_written"] is False


def test_prepared_summary_outputs_write_summaries_qc_and_provenance(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"

    record = write_prepared_mvpa_summary_outputs(
        _summary_result(),
        output_root=output_root,
        summaries_path="mvpa/summaries.tsv",
        qc_path="mvpa/summary_qc.tsv",
        provenance_path="mvpa/summary_provenance.json",
    )

    assert record["will_write"] is True
    assert _header(output_root / "mvpa/summaries.tsv") == _SUMMARY_HEADER
    assert _header(output_root / "mvpa/summary_qc.tsv") == _SUMMARY_QC_HEADER

    summaries = _read_tsv(output_root / "mvpa/summaries.tsv")
    assert summaries == [
        {
            "group_id": "group-alpha",
            "condition_id_a": "condition-a",
            "condition_id_b": "condition-b",
            "metric": "crossnobis",
            "engine_name": "native_reference",
            "normalization_method": "identity",
            "n": "2",
            "mean_distance": "1.5",
            "std_distance": "0.5",
            "sem_distance": "0.35355339059327373",
            "min_distance": "1.0",
            "max_distance": "2.0",
        }
    ]

    qc_rows = _read_tsv(output_root / "mvpa/summary_qc.tsv")
    assert qc_rows[0]["code"] == "synthetic_summary_warning"
    assert qc_rows[0]["group_key"] == '{"analysis_unit":"unit-a"}'
    assert qc_rows[0]["context"] == '{"distinct_value_count":1}'

    provenance = json.loads((output_root / "mvpa/summary_provenance.json").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == runtime_outputs.SCHEMA_VERSION
    assert provenance["artifact_kind"] == "prepared_mvpa_distance_summary_outputs"
    assert provenance["output_paths"] == {
        "summaries": "mvpa/summaries.tsv",
        "qc": "mvpa/summary_qc.tsv",
        "provenance": "mvpa/summary_provenance.json",
    }
    assert provenance["row_counts"] == {"summaries": 1, "qc": 1}
    assert provenance["summary_row_count"] == 1
    assert provenance["qc_row_count"] == 1
    assert provenance["input_provenance"]["phase"] == "3D.1"
    assert provenance["input_provenance"]["output_written"] is False


def test_runtime_outputs_preserve_grouping_columns_and_condition_pair_ids(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    group_key = {"participant_id": "participant-a", "task_id": "task-alpha"}
    pattern_result = {
        "groups": [
            {
                "group_id": "group-alpha",
                "group_key": group_key,
                "group_by": ["participant_id", "task_id"],
                "cv_unit": "run",
                "cv_labels": ["run-a"],
                "condition_ids": ["condition-a"],
                "feature_count": 1,
                "rows": [
                    {
                        "pattern_id": "pattern-alpha",
                        "condition_id": "condition-a",
                        "cv_unit": "run",
                        "cv_label": "run-a",
                        "feature_values": [1.0],
                        "feature_count": 1,
                        "source_row_index": 0,
                        "mean_centering_applied": True,
                        "mean_centering_scope": "roi",
                        "event_count": 2,
                    }
                ],
            }
        ],
        "qc_rows": [],
        "provenance": [],
        "warnings": [],
        "errors": [],
        "executed": True,
    }
    distance_result = {
        "distances": [
            {
                "group_id": "group-alpha",
                "group_key": group_key,
                "condition_id_a": "condition-a",
                "condition_id_b": "condition-b",
                "condition_pair_id": "pair-alpha",
                "distance": 1.0,
                "metric": "crossnobis",
                "engine_name": "native_reference",
                "normalization_method": "identity",
            }
        ],
        "qc_rows": [],
        "provenance": [],
        "warnings": [],
        "errors": [],
        "executed": True,
    }
    summary_result = {
        "summary_rows": [
            {
                "group_id": "group-alpha",
                "participant_id": "participant-a",
                "task_id": "task-alpha",
                "condition_id_a": "condition-a",
                "condition_id_b": "condition-b",
                "condition_pair_id": "pair-alpha",
                "metric": "crossnobis",
                "engine_name": "native_reference",
                "normalization_method": "identity",
                "n": 1,
                "mean_distance": 1.0,
                "std_distance": 0.0,
                "sem_distance": 0.0,
                "min_distance": 1.0,
                "max_distance": 1.0,
            }
        ],
        "qc_rows": [],
        "provenance": [],
        "warnings": [],
        "errors": [],
        "executed": True,
    }

    write_prepared_mvpa_pattern_outputs(
        pattern_result,
        output_root=output_root,
        rows_path="patterns/rows.tsv",
        qc_path="patterns/qc.tsv",
        provenance_path="patterns/provenance.json",
    )
    write_prepared_mvpa_distance_outputs(
        distance_result,
        output_root=output_root,
        distances_path="distances/rows.tsv",
        qc_path="distances/qc.tsv",
        provenance_path="distances/provenance.json",
    )
    write_prepared_mvpa_summary_outputs(
        summary_result,
        output_root=output_root,
        summaries_path="summaries/rows.tsv",
        qc_path="summaries/qc.tsv",
        provenance_path="summaries/provenance.json",
    )

    assert _header(output_root / "patterns/rows.tsv") == [*_PATTERN_ROW_HEADER, "participant_id", "task_id"]
    pattern_rows = _read_tsv(output_root / "patterns/rows.tsv")
    assert pattern_rows[0]["participant_id"] == "participant-a"
    assert pattern_rows[0]["task_id"] == "task-alpha"
    assert pattern_rows[0]["mean_centering_applied"] == "true"
    assert pattern_rows[0]["event_count"] == "2"

    assert _header(output_root / "distances/rows.tsv") == [*_DISTANCE_HEADER, "participant_id", "task_id"]
    distance_rows = _read_tsv(output_root / "distances/rows.tsv")
    assert distance_rows[0]["participant_id"] == "participant-a"
    assert distance_rows[0]["task_id"] == "task-alpha"
    assert distance_rows[0]["condition_pair_id"] == "pair-alpha"

    assert _header(output_root / "summaries/rows.tsv") == [
        *_SUMMARY_HEADER,
        "condition_pair_id",
        "participant_id",
        "task_id",
    ]
    summary_rows = _read_tsv(output_root / "summaries/rows.tsv")
    assert summary_rows[0]["participant_id"] == "participant-a"
    assert summary_rows[0]["task_id"] == "task-alpha"
    assert summary_rows[0]["condition_pair_id"] == "pair-alpha"


def test_dry_run_preview_creates_no_files_or_directories(tmp_path: Path) -> None:
    output_root = tmp_path / "dry-run-cache"

    plan = plan_prepared_mvpa_pattern_outputs(
        _pattern_result(),
        output_root=output_root,
        rows_path="mvpa/pattern_rows.tsv",
        qc_path="mvpa/pattern_qc.tsv",
        provenance_path="mvpa/pattern_provenance.json",
    )

    assert plan["will_write"] is False
    assert plan["output_written"] is False
    assert not output_root.exists()
    assert [artifact["will_write"] for artifact in plan["artifacts"]] == [False, False, False]
    assert [artifact["exists"] for artifact in plan["artifacts"]] == [False, False, False]
    assert plan["artifacts"][0]["row_count"] == 1
    assert plan["artifacts"][0]["columns"] == [*_PATTERN_ROW_HEADER, "subject_id", "task_id"]

    summary_plan = plan_prepared_mvpa_summary_outputs(
        _summary_result(),
        output_root=output_root,
        summaries_path="mvpa/summaries.tsv",
        qc_path="mvpa/summary_qc.tsv",
        provenance_path="mvpa/summary_provenance.json",
    )

    assert summary_plan["will_write"] is False
    assert summary_plan["output_written"] is False
    assert not output_root.exists()
    assert [artifact["will_write"] for artifact in summary_plan["artifacts"]] == [False, False, False]
    assert summary_plan["artifacts"][0]["row_count"] == 1
    assert summary_plan["artifacts"][0]["columns"] == _SUMMARY_HEADER


def test_overwrite_false_rejects_existing_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    kwargs = {
        "output_root": output_root,
        "rows_path": "mvpa/pattern_rows.tsv",
        "qc_path": "mvpa/pattern_qc.tsv",
        "provenance_path": "mvpa/pattern_provenance.json",
    }

    write_prepared_mvpa_pattern_outputs(_pattern_result(), **kwargs)

    with pytest.raises(FileExistsError):
        write_prepared_mvpa_pattern_outputs(_pattern_result(), **kwargs)

    summary_kwargs = {
        "output_root": output_root,
        "summaries_path": "mvpa/summaries.tsv",
        "qc_path": "mvpa/summary_qc.tsv",
        "provenance_path": "mvpa/summary_provenance.json",
    }
    write_prepared_mvpa_summary_outputs(_summary_result(), **summary_kwargs)

    with pytest.raises(FileExistsError):
        write_prepared_mvpa_summary_outputs(_summary_result(), **summary_kwargs)


def test_overwrite_true_replaces_existing_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    mvpa_dir = output_root / "mvpa"
    mvpa_dir.mkdir(parents=True)
    (mvpa_dir / "pattern_rows.tsv").write_text("old rows\n", encoding="utf-8")
    (mvpa_dir / "pattern_qc.tsv").write_text("old qc\n", encoding="utf-8")
    (mvpa_dir / "pattern_provenance.json").write_text('{"old":true}\n', encoding="utf-8")

    record = write_prepared_mvpa_pattern_outputs(
        _pattern_result(),
        output_root=output_root,
        rows_path="mvpa/pattern_rows.tsv",
        qc_path="mvpa/pattern_qc.tsv",
        provenance_path="mvpa/pattern_provenance.json",
        overwrite=True,
    )

    assert record["overwrite"] is True
    assert "old rows" not in (mvpa_dir / "pattern_rows.tsv").read_text(encoding="utf-8")
    assert _read_tsv(mvpa_dir / "pattern_rows.tsv")[0]["pattern_id"] == "p1"
    assert json.loads((mvpa_dir / "pattern_provenance.json").read_text(encoding="utf-8"))["overwrite"] is True


def test_unsafe_parent_traversal_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parent traversal"):
        plan_prepared_mvpa_pattern_outputs(
            _pattern_result(),
            output_root=tmp_path / "cache",
            rows_path="../pattern_rows.tsv",
            qc_path="mvpa/pattern_qc.tsv",
            provenance_path="mvpa/pattern_provenance.json",
        )

    with pytest.raises(ValueError, match="parent traversal"):
        plan_prepared_mvpa_summary_outputs(
            _summary_result(),
            output_root=tmp_path / "cache",
            summaries_path="../summaries.tsv",
            qc_path="mvpa/summary_qc.tsv",
            provenance_path="mvpa/summary_provenance.json",
        )


def test_absolute_path_outside_output_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside output_root"):
        plan_prepared_mvpa_distance_outputs(
            _distance_result(),
            output_root=tmp_path / "cache",
            distances_path=tmp_path / "outside.tsv",
            qc_path="mvpa/distance_qc.tsv",
            provenance_path="mvpa/distance_provenance.json",
        )


def test_json_safe_serialization_rejects_nonfinite_floats_before_writing(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    bad_result = {
        "groups": [
            {
                "group_id": "g1",
                "group_key": {"subject_id": "sub-01"},
                "group_by": ["subject_id"],
                "rows": [
                    {
                        "pattern_id": "p1",
                        "condition_id": "face",
                        "cv_unit": "run",
                        "cv_label": "run-1",
                        "feature_values": [math.nan],
                        "feature_count": 1,
                        "group_key": {"subject_id": "sub-01"},
                        "source_row_index": 0,
                    }
                ],
                "cv_unit": "run",
                "cv_labels": ["run-1"],
                "condition_ids": ["face"],
                "feature_count": 1,
            }
        ],
        "qc_rows": [],
        "provenance": [],
        "warnings": [],
        "errors": [],
        "executed": True,
    }

    with pytest.raises(ValueError, match="non-finite"):
        write_prepared_mvpa_pattern_outputs(
            bad_result,
            output_root=output_root,
            rows_path="mvpa/pattern_rows.tsv",
            qc_path="mvpa/pattern_qc.tsv",
            provenance_path="mvpa/pattern_provenance.json",
        )

    assert not output_root.exists()

    bad_summary = {
        "summary_rows": [{"group_id": "group-alpha", "distance": math.inf}],
        "qc_rows": [],
        "provenance": [],
        "warnings": [],
        "errors": [],
        "executed": True,
    }

    with pytest.raises(ValueError, match="non-finite"):
        write_prepared_mvpa_summary_outputs(
            bad_summary,
            output_root=output_root,
            summaries_path="mvpa/summaries.tsv",
            qc_path="mvpa/summary_qc.tsv",
            provenance_path="mvpa/summary_provenance.json",
        )

    assert not output_root.exists()


def test_prepared_summary_outputs_use_deterministic_headers_and_exported_api(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    result = {
        "summary_rows": [
            {
                "group_id": "group-alpha",
                "condition_id_a": "condition-a",
                "condition_id_b": "condition-b",
                "metric": "crossnobis",
                "engine_name": "native_reference",
                "normalization_method": "identity",
                "n": 1,
                "mean_distance": 2.0,
                "std_distance": 0.0,
                "sem_distance": 0.0,
                "min_distance": 2.0,
                "max_distance": 2.0,
                "zeta_extra": "last",
                "alpha_extra": "first",
            }
        ],
        "qc_rows": [],
        "provenance": [],
        "warnings": [],
        "errors": [],
        "executed": True,
    }

    write_prepared_mvpa_summary_outputs(
        result,
        output_root=output_root,
        summaries_path="summary/summaries.tsv",
        qc_path="summary/qc.tsv",
        provenance_path="summary/provenance.json",
    )

    assert _header(output_root / "summary/summaries.tsv") == [*_SUMMARY_HEADER, "alpha_extra", "zeta_extra"]
    assert "plan_prepared_mvpa_summary_outputs" in mvpa.__all__
    assert "write_prepared_mvpa_summary_outputs" in mvpa.__all__
    assert callable(plan_prepared_mvpa_summary_outputs)
    assert callable(write_prepared_mvpa_summary_outputs)


def test_empty_row_collections_write_header_only_tsvs(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    empty_pattern = MvpaPatternRowPreparationResult(
        groups=(),
        qc_rows=(),
        provenance=(),
        warnings=(),
        errors=(),
    )
    empty_distance = PreparedMvpaDistanceResult(
        distances=(),
        qc_rows=(),
        provenance=(),
        warnings=(),
        errors=(),
    )

    write_prepared_mvpa_pattern_outputs(
        empty_pattern,
        output_root=output_root,
        rows_path="pattern/rows.tsv",
        qc_path="pattern/qc.tsv",
        provenance_path="pattern/provenance.json",
    )
    write_prepared_mvpa_distance_outputs(
        empty_distance,
        output_root=output_root,
        distances_path="distance/distances.tsv",
        qc_path="distance/qc.tsv",
        provenance_path="distance/provenance.json",
    )

    assert (output_root / "pattern/rows.tsv").read_text(encoding="utf-8").splitlines() == [
        "\t".join(_PATTERN_ROW_HEADER)
    ]
    assert (output_root / "pattern/qc.tsv").read_text(encoding="utf-8").splitlines() == [
        "\t".join(_PATTERN_QC_HEADER)
    ]
    assert (output_root / "distance/distances.tsv").read_text(encoding="utf-8").splitlines() == [
        "\t".join(_DISTANCE_HEADER)
    ]
    assert (output_root / "distance/qc.tsv").read_text(encoding="utf-8").splitlines() == [
        "\t".join(_DISTANCE_QC_HEADER)
    ]


def test_forbidden_import_guard_for_runtime_outputs_module_and_tests() -> None:
    forbidden_modules = (
        "research_platform.neuro",
        "research_platform.bids",
        "research_platform.core",
        "research_platform.viz",
        "research_platform.ml",
        "research_platform.io",
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "rsatoolbox",
        "nilearn",
        "sklearn",
        "nibabel",
        "pipelines",
        "ops",
    )

    imported_modules: list[str] = []
    for path in (Path(runtime_outputs.__file__), Path(__file__)):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)

    for imported_module in imported_modules:
        assert not any(
            imported_module == forbidden or imported_module.startswith(f"{forbidden}.")
            for forbidden in forbidden_modules
        )
