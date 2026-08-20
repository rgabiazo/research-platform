from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json
import os
import subprocess
import sys

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "research-analysis" / "src"))

import research_platform.neuro.mvpa.materialized_pattern_table as materialized
from research_platform.analysis.mvpa import (
    compute_mvpa_distances_from_prepared_groups,
    prepare_mvpa_pattern_row_groups,
)
from research_platform.neuro.mvpa.materialized_pattern_table import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    load_materialized_pattern_table,
    plan_materialized_pattern_table,
    validate_materialized_pattern_source_fields,
)
from research_platform.neuro.mvpa.pattern_sources import ResolvedAnalysisUnit


def _config(
    *,
    conditions: tuple[str, ...] = ("condition-a",),
    roi_labels: tuple[str, ...] = ("SeedA",),
    cv_unit: str = "run",
    noise_method: str = "identity",
    centered: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        conditions=tuple(SimpleNamespace(id=value) for value in conditions),
        roi_sources=(
            SimpleNamespace(
                name="toy-rois",
                fields={"roi_labels": list(roi_labels)},
                masks=(),
            ),
        ),
        distance=SimpleNamespace(
            cv_unit=cv_unit,
            grouping_columns=("fold_id",) if cv_unit == "custom" else (),
            noise_normalization=SimpleNamespace(method=noise_method),
        ),
        mean_centering=SimpleNamespace(enabled=centered, scope="roi" if centered else "none"),
    )


def _unit(
    subject: str,
    run: str,
    *,
    source_row: int = 1,
    session: str | None = "ses-01",
    task: str | None = "exampletask",
    fold: str = "fold-a",
) -> ResolvedAnalysisUnit:
    values = {"subject_id": subject, "run_id": run, "fold_id": fold}
    keys = ("subject_id", "run_id")
    if session is not None:
        values["session_id"] = session
    if task is not None:
        values["task_id"] = task
    return ResolvedAnalysisUnit(
        unit_id=f"unit-{source_row}",
        source_row=source_row,
        key_columns=keys,
        subject_id=subject,
        session_id=session,
        task_id=task,
        run_id=run,
        values=values,
    )


def _source(path: str = "tables/patterns.tsv") -> SimpleNamespace:
    fields = {
        "name": "prepared-patterns",
        "backend": "materialized_pattern_table",
        "root_ref": "artifact_root",
        "path": path,
        "schema_version": SCHEMA_VERSION,
    }
    return SimpleNamespace(
        name="prepared-patterns",
        root_ref="artifact_root",
        path=path,
        fields=fields,
    )


def _row(
    pattern_id: str,
    *,
    subject_id: str = "sub-toy01",
    session_id: str = "ses-01",
    task_id: str = "exampletask",
    run_id: str = "run-01",
    condition_id: str = "condition-a",
    roi_label: str = "SeedA",
    feature_values: str = "[1.0,2.5,-0.5]",
    feature_count: str = "3",
    noise_method: str = "identity",
    **overrides: str,
) -> dict[str, str]:
    row = {
        "schema_version": SCHEMA_VERSION,
        "pattern_id": pattern_id,
        "subject_id": subject_id,
        "session_id": session_id,
        "task_id": task_id,
        "run_id": run_id,
        "condition_id": condition_id,
        "pattern_source_name": "prepared-patterns",
        "roi_source_name": "toy-rois",
        "roi_label": roi_label,
        "feature_count": feature_count,
        "voxel_order": "c-order",
        "voxel_index_hash": "index-hash-a",
        "feature_space_id": "space-a",
        "roi_definition_id": f"toy-rois:{roi_label}",
        "feature_values": feature_values,
        "usable": "true",
        "status": "ok",
        "mean_centering_applied": "false",
        "mean_centering_scope": "none",
        "noise_status": "unused" if noise_method == "identity" else "ok",
        "noise_usable": "false" if noise_method == "identity" else "true",
        "cross_validation_label": run_id,
        "event_count": "4",
        "qc_status": "pass",
        "qc_reason": "",
        "grouping_values": "{}",
        "warnings": "[]",
        "errors": "[]",
        "roi_reference": f"root_ref:artifact_root/rois/{roi_label}.nii",
        "generator_version": "toy-generator-v1",
        "software_version": "research-neuro-alpha",
        "derivation_id": "derivation-a",
        "holdout_id": "holdout-a",
        "noise_values": "" if noise_method == "identity" else "[1.0,2.0,3.0]",
        "noise_feature_count": "" if noise_method == "identity" else feature_count,
        "noise_voxel_order": "" if noise_method == "identity" else "c-order",
        "noise_voxel_index_hash": "" if noise_method == "identity" else "index-hash-a",
        "noise_feature_space_id": "" if noise_method == "identity" else "space-a",
        "noise_roi_definition_id": "" if noise_method == "identity" else f"toy-rois:{roi_label}",
        "noise_value_kind": "" if noise_method == "identity" else "variance",
        "noise_estimation_scope": "" if noise_method == "identity" else "run",
        "noise_source": "" if noise_method == "identity" else "residual-model",
    }
    row.update(overrides)
    return row


def _write_table(path: Path, rows: list[dict[str, str]], *, columns: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or tuple(dict.fromkeys((*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plan(tmp_path: Path, rows: list[dict[str, str]], *, config=None, units=None):
    table = tmp_path / "tables" / "patterns.tsv"
    _write_table(table, rows)
    config = config or _config()
    units = units or (_unit("sub-toy01", "run-01"),)
    return plan_materialized_pattern_table(
        config,
        _source(),
        units,
        roots={"artifact_root": tmp_path},
    )


def test_source_contract_requires_safe_root_path_and_exact_schema() -> None:
    assert validate_materialized_pattern_source_fields(_source().fields, "source") == ()
    errors = validate_materialized_pattern_source_fields(
        {"root_ref": "artifact_root", "path": "../patterns.tsv", "schema_version": "v0"},
        "source",
    )
    assert any("safe relative" in error for error in errors)
    assert any(SCHEMA_VERSION in error for error in errors)
    embedded = validate_materialized_pattern_source_fields(
        {
            "root_ref": "artifact_root",
            "path": "tables/--input=/home/alice/private.tsv",
            "schema_version": SCHEMA_VERSION,
        },
        "source",
    )
    assert any("non-portable local path" in error for error in embedded)


def test_source_contract_requires_fixed_name_backend_and_named_root() -> None:
    errors = validate_materialized_pattern_source_fields(
        {
            "path": "patterns.tsv",
            "schema_version": SCHEMA_VERSION,
        },
        "source",
    )

    assert any("source.name must be defined" in error for error in errors)
    assert any("source.backend must be" in error for error in errors)
    assert any("source.root_ref must be defined" in error for error in errors)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/home/alice/private.tsv",
        "~/private.tsv",
        "~\\private.tsv",
        "C:\\Data\\private.tsv",
        "\\\\cluster.example\\share\\private.tsv",
        "file:///home/alice/private.tsv",
        "file://cluster.example/share/private.tsv",
        "https://example.org/patterns.tsv",
        "s3://bucket/patterns.tsv",
        "tables\\patterns.tsv",
        "tables/../private.tsv",
        "tables/--input=/home/alice/private.tsv",
    ],
)
def test_source_contract_rejects_every_local_path_form(unsafe_path: str) -> None:
    errors = validate_materialized_pattern_source_fields(
        {
            **_source().fields,
            "path": unsafe_path,
        },
        "source",
    )
    assert any(
        "safe relative path" in error or "non-portable local path" in error
        for error in errors
    )


def test_invalid_source_declaration_does_not_echo_local_paths_in_plan(tmp_path: Path) -> None:
    source = _source("/home/alice/private.tsv")
    source.fields["name"] = "/home/alice/source-name"
    source.name = "/home/alice/source-name"
    source.fields["root_ref"] = "/home/alice/root"
    source.root_ref = "/home/alice/root"

    plan = plan_materialized_pattern_table(
        _config(),
        source,
        (_unit("sub-toy01", "run-01"),),
        roots={},
    )
    payload = json.dumps(plan.public_summary(), sort_keys=True)

    assert not plan.valid
    assert "/home/alice" not in payload
    assert plan.portable_reference is None


def test_source_contract_rejects_deferred_format_options() -> None:
    errors = validate_materialized_pattern_source_fields(
        {
            **_source().fields,
            "delimiter": ",",
            "column_mapping": {"condition": "trial_type"},
            "arbitrary_option": "not-part-of-v1",
        },
        "source",
    )

    assert any("unsupported" in error for error in errors)
    assert any("arbitrary_option" in error for error in errors)


def test_planner_requires_at_least_one_resolved_exact_unit(tmp_path: Path) -> None:
    table = tmp_path / "tables" / "patterns.tsv"
    _write_table(table, [_row("pattern-a")])

    plan = plan_materialized_pattern_table(
        _config(),
        _source(),
        (),
        roots={"artifact_root": tmp_path},
    )

    assert not plan.valid
    assert any("resolved exact analysis unit" in error for error in plan.errors)


def test_direct_planner_rejects_duplicate_exact_unit_keys(tmp_path: Path) -> None:
    table = tmp_path / "tables" / "patterns.tsv"
    _write_table(table, [_row("pattern-a")])
    units = (
        _unit("sub-toy01", "run-01", source_row=1),
        _unit("sub-toy01", "run-01", source_row=2),
    )

    plan = plan_materialized_pattern_table(
        _config(),
        _source(),
        units,
        roots={"artifact_root": tmp_path},
    )

    assert not plan.valid
    assert any("repeat a configured unit key" in error for error in plan.errors)


def test_source_symlink_must_resolve_beneath_named_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside" / "patterns.tsv"
    _write_table(outside, [_row("pattern-a")])
    link = root / "tables" / "patterns.tsv"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)

    plan = plan_materialized_pattern_table(
        _config(),
        _source(),
        (_unit("sub-toy01", "run-01"),),
        roots={"artifact_root": root},
    )

    assert not plan.valid
    assert any("does not remain beneath root_ref" in error for error in plan.errors)


def test_planner_hashes_streams_and_does_not_decode_vectors(tmp_path: Path) -> None:
    plan = _plan(tmp_path, [_row("pattern-a", feature_values="not-json")])

    assert plan.valid
    assert plan.ready_for_materialization
    assert plan.source_sha256 == hashlib.sha256((tmp_path / "tables/patterns.tsv").read_bytes()).hexdigest()
    assert plan.scalar_rows[0].pattern_id == "pattern-a"
    assert not hasattr(plan.scalar_rows[0], "feature_values")
    assert "not-json" not in json.dumps(plan.public_summary())
    assert plan.pattern_rows[0].pattern_reference == "root_ref:artifact_root/tables/patterns.tsv#row=1"

    loaded = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")
    assert not loaded.valid
    assert loaded.rows == ()
    assert any("JSON numeric array" in error for error in loaded.errors)


def test_required_header_columns_are_enforced_and_optional_columns_may_be_absent(
    tmp_path: Path,
) -> None:
    cross_sectional = ResolvedAnalysisUnit(
        unit_id="unit-cross-sectional",
        source_row=1,
        key_columns=("subject_id",),
        subject_id="sub-toy01",
        values={"subject_id": "sub-toy01"},
    )
    config = _config(cv_unit="subject")
    row = _row(
        "pattern-a",
        session_id="",
        task_id="",
        run_id="",
        cross_validation_label="",
    )
    required_only = {column: row[column] for column in REQUIRED_COLUMNS}
    table = tmp_path / "tables" / "patterns.tsv"
    _write_table(table, [required_only], columns=REQUIRED_COLUMNS)

    plan = plan_materialized_pattern_table(
        config,
        _source(),
        (cross_sectional,),
        roots={"artifact_root": tmp_path},
    )
    loaded = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")

    assert plan.valid
    assert loaded.valid
    assert loaded.rows[0]["cross_validation_label"] == "sub-toy01"
    assert loaded.rows[0]["session_id"] is None

    missing_root = tmp_path / "missing-required"
    missing_columns = tuple(column for column in REQUIRED_COLUMNS if column != "feature_space_id")
    missing_row = {column: row[column] for column in missing_columns}
    _write_table(
        missing_root / "tables" / "patterns.tsv",
        [missing_row],
        columns=missing_columns,
    )
    invalid = plan_materialized_pattern_table(
        config,
        _source(),
        (cross_sectional,),
        roots={"artifact_root": missing_root},
    )
    assert not invalid.valid
    assert any("feature_space_id" in error and "missing required" in error for error in invalid.errors)


def test_planner_never_calls_json_decoder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("planner decoded vector JSON")

    monkeypatch.setattr(materialized.json, "loads", forbidden)
    plan = _plan(tmp_path, [_row("pattern-a")])
    assert plan.valid


def test_loader_returns_analysis_ready_mappings_and_portable_provenance(tmp_path: Path) -> None:
    plan = _plan(tmp_path, [_row("pattern-a")])
    result = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")

    assert result.valid and result.materialized and not result.executed
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["feature_values"] == (1.0, 2.5, -0.5)
    assert row["feature_count"] == 3
    assert row["cross_validation_label"] == "run-01"
    assert row["voxel_order"] == "c-order"
    assert row["noise_status"] == "unused"
    assert row["noise_usable"] is False
    assert result.provenance["source_reference"] == "root_ref:artifact_root/tables/patterns.tsv"
    assert str(tmp_path) not in json.dumps(result.provenance)
    assert row["roi_reference"] == "root_ref:artifact_root/rois/SeedA.nii"
    assert row["generator_version"] == "toy-generator-v1"
    assert row["software_version"] == "research-neuro-alpha"
    assert row["derivation_id"] == "derivation-a"
    assert row["holdout_id"] == "holdout-a"
    assert result.provenance["generator_versions"] == ("toy-generator-v1",)


def test_loader_rows_roundtrip_through_existing_preparation_and_distance_math(tmp_path: Path) -> None:
    units = (
        _unit("sub-toy01", "run-01", source_row=1),
        _unit("sub-toy01", "run-02", source_row=2),
    )
    config = _config(conditions=("condition-a", "condition-b"))
    plan = _plan(
        tmp_path,
        [
            _row("run1-a", run_id="run-01", condition_id="condition-a", feature_values="[1.0,0.0]", feature_count="2"),
            _row("run1-b", run_id="run-01", condition_id="condition-b", feature_values="[0.0,0.0]", feature_count="2"),
            _row("run2-a", run_id="run-02", condition_id="condition-a", feature_values="[3.0,0.0]", feature_count="2"),
            _row("run2-b", run_id="run-02", condition_id="condition-b", feature_values="[0.0,0.0]", feature_count="2"),
        ],
        config=config,
        units=units,
    )
    loaded = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")

    prepared = prepare_mvpa_pattern_row_groups(
        loaded.rows,
        cv_unit="run",
        cv_label_column="cross_validation_label",
    )
    distances = compute_mvpa_distances_from_prepared_groups(prepared.groups)

    assert loaded.valid
    assert prepared.errors == ()
    assert prepared.groups[0].cv_labels == ("run-01", "run-02")
    assert distances.errors == ()
    assert len(distances.distances) == 1
    assert distances.distances[0].distance == pytest.approx(3.0)


def test_exact_join_and_output_order_are_unit_condition_then_roi(tmp_path: Path) -> None:
    units = (_unit("sub-toy02", "run-02", source_row=2), _unit("sub-toy01", "run-01"))
    config = _config(conditions=("condition-a", "condition-b"), roi_labels=("SeedA", "SeedB"))
    rows = []
    for unit in reversed(units):
        for condition in reversed(("condition-a", "condition-b")):
            for roi in reversed(("SeedA", "SeedB")):
                rows.append(
                    _row(
                        f"{unit.subject_id}-{condition}-{roi}",
                        subject_id=unit.subject_id,
                        run_id=unit.run_id or "",
                        condition_id=condition,
                        roi_label=roi,
                    )
                )
    plan = _plan(tmp_path, rows, config=config, units=units)

    assert plan.valid
    assert [
        (row.subject_id, row.condition_id, row.roi_label) for row in plan.scalar_rows
    ] == [
        (subject, condition, roi)
        for subject in ("sub-toy02", "sub-toy01")
        for condition in ("condition-a", "condition-b")
        for roi in ("SeedA", "SeedB")
    ]


def test_irregular_longitudinal_multiple_runs_are_not_cartesian_expanded(tmp_path: Path) -> None:
    unit_values = (
        ("sub-toy01", "ses-early", "run-01"),
        ("sub-toy01", "ses-early", "run-02"),
        ("sub-toy01", "ses-late", "run-03"),
        ("sub-toy02", "ses-only", "run-01"),
    )
    units = tuple(
        ResolvedAnalysisUnit(
            unit_id=f"unit-{index}",
            source_row=index,
            key_columns=("subject_id", "session_id", "run_id"),
            subject_id=subject,
            session_id=session,
            task_id="exampletask",
            run_id=run,
            values={
                "subject_id": subject,
                "session_id": session,
                "task_id": "exampletask",
                "run_id": run,
                "visit_index": "2" if session == "ses-late" else "1",
            },
        )
        for index, (subject, session, run) in enumerate(unit_values, start=1)
    )
    rows = [
        _row(
            f"pattern-{index}",
            subject_id=unit.subject_id,
            session_id=unit.session_id or "",
            run_id=unit.run_id or "",
        )
        for index, unit in reversed(tuple(enumerate(units, start=1)))
    ]
    plan = _plan(tmp_path, rows, units=units)
    assert plan.valid
    assert len(plan.scalar_rows) == len(units)
    assert [
        (row.subject_id, row.session_id, row.run_id) for row in plan.scalar_rows
    ] == list(unit_values)


def test_unselected_rows_are_audited_without_becoming_selected(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        [
            _row("selected"),
            _row("not-selected", subject_id="sub-toy99", run_id="run-99"),
        ],
    )
    assert plan.valid
    assert plan.selected_row_count == 1
    assert plan.unselected_row_count == 1
    assert plan.unselected_pattern_ids == ("not-selected",)
    assert plan.public_summary()["counts"]["unselected_rows"] == 1


def test_unselected_rows_still_require_complete_schema_and_unit_keys(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        [
            _row("selected"),
            _row("not-selected", subject_id="sub-toy99", run_id="", feature_space_id=""),
        ],
    )
    assert not plan.valid
    assert any("run_id" in error and "requires" in error for error in plan.errors)
    assert any("feature_space_id" in error and "requires" in error for error in plan.errors)


@pytest.mark.parametrize(
    "rows,error_fragment",
    [
        ([_row("same"), _row("same")], "pattern_id is duplicated"),
        ([_row("a"), _row("b")], "repeats one unit-condition-ROI"),
        ([], "does not fully cover"),
        ([_row("a", condition_id="unknown")], "unknown condition_id"),
        ([_row("a", pattern_source_name="other")], "does not match its source config"),
        ([_row("a", cross_validation_label="run-other")], "cross_validation_label"),
    ],
)
def test_planner_rejects_identity_coverage_and_cv_contract_failures(
    tmp_path: Path,
    rows: list[dict[str, str]],
    error_fragment: str,
) -> None:
    plan = _plan(tmp_path, rows)
    assert not plan.valid
    assert any(error_fragment in error for error in plan.errors)
    assert plan.pattern_rows == ()


@pytest.mark.parametrize(
    "overrides,error_fragment",
    [
        ({"feature_values": "[1.0,true,2.0]"}, "non-numeric"),
        ({"feature_values": '[1.0,"not-a-number",2.0]'}, "non-numeric"),
        ({"feature_values": "[]"}, "non-empty"),
        ({"feature_values": "[1.0,NaN,2.0]"}, "invalid numeric"),
        ({"feature_values": "[1.0,Infinity,2.0]"}, "invalid numeric"),
        ({"feature_values": "[1.0,2.0]"}, "feature_count"),
        ({"errors": '["declared failure"]'}, "usable but declares"),
    ],
)
def test_loader_is_strict_and_all_or_nothing(
    tmp_path: Path,
    overrides: dict[str, str],
    error_fragment: str,
) -> None:
    plan = _plan(tmp_path, [_row("pattern-a", **overrides)])
    assert plan.valid
    result = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")
    assert not result.valid
    assert result.rows == ()
    assert any(error_fragment in error for error in result.errors)


def test_one_invalid_selected_row_prevents_returning_an_otherwise_valid_row(
    tmp_path: Path,
) -> None:
    config = _config(conditions=("condition-a", "condition-b"))
    plan = _plan(
        tmp_path,
        [
            _row("valid-row", condition_id="condition-a"),
            _row(
                "invalid-row",
                condition_id="condition-b",
                feature_values='[1.0,"not-a-number",2.0]',
            ),
        ],
        config=config,
    )

    result = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")

    assert plan.valid
    assert not result.valid
    assert result.rows == ()
    assert any("non-numeric" in error for error in result.errors)


def test_planner_rejects_invalid_scalar_feature_count(tmp_path: Path) -> None:
    plan = _plan(tmp_path, [_row("pattern-a", feature_count="0")])
    assert not plan.valid
    assert any("positive integer" in error for error in plan.errors)


def test_diagonal_noise_is_positive_finite_and_geometry_matched(tmp_path: Path) -> None:
    config = _config(noise_method="diagonal")
    plan = _plan(tmp_path, [_row("pattern-a", noise_method="diagonal")], config=config)
    assert plan.valid
    loaded = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")
    assert loaded.valid
    assert loaded.rows[0]["noise_values"] == (1.0, 2.0, 3.0)
    assert loaded.rows[0]["noise_feature_count"] == 3

    bad_values_plan = _plan(
        tmp_path / "bad-values",
        [_row("pattern-a", noise_method="diagonal", noise_values="[1.0,0.0,3.0]")],
        config=config,
    )
    bad_values = load_materialized_pattern_table(
        bad_values_plan,
        expected_sha256=bad_values_plan.source_sha256 or "",
    )
    assert not bad_values.valid
    assert any("invalid numeric" in error for error in bad_values.errors)

    bad_identity_plan = _plan(
        tmp_path / "bad-identity",
        [_row("pattern-a", noise_method="diagonal", noise_voxel_index_hash="other")],
        config=config,
    )
    assert not bad_identity_plan.valid
    assert any("does not match" in error for error in bad_identity_plan.errors)


@pytest.mark.parametrize(
    "overrides,planning_error,error_fragment",
    [
        ({"noise_values": ""}, True, "noise_values"),
        ({"noise_values": "[1.0,NaN,3.0]"}, False, "invalid numeric"),
        ({"noise_values": "[1.0,0.0,3.0]"}, False, "invalid numeric"),
        (
            {"noise_values": "[1.0,2.0]", "noise_feature_count": "2"},
            True,
            "feature and noise counts differ",
        ),
        ({"noise_voxel_order": "different-order"}, True, "does not match"),
        ({"noise_voxel_index_hash": "different-index"}, True, "does not match"),
        ({"noise_feature_space_id": "different-space"}, True, "does not match"),
        ({"noise_roi_definition_id": "different-roi"}, True, "does not match"),
        ({"noise_value_kind": "standard_deviation"}, True, "variances"),
    ],
)
def test_diagonal_noise_rejects_missing_invalid_width_or_identity_mismatches(
    tmp_path: Path,
    overrides: dict[str, str],
    planning_error: bool,
    error_fragment: str,
) -> None:
    config = _config(noise_method="diagonal")
    plan = _plan(
        tmp_path,
        [_row("pattern-a", noise_method="diagonal", **overrides)],
        config=config,
    )

    if planning_error:
        assert not plan.valid
        errors = plan.errors
    else:
        assert plan.valid
        loaded = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")
        assert not loaded.valid
        errors = loaded.errors
    assert any(error_fragment in error for error in errors)


@pytest.mark.parametrize(
    "overrides,error_fragment",
    [
        ({"noise_estimation_scope": ""}, "noise_estimation_scope"),
        ({"noise_source": ""}, "noise_source"),
        ({"noise_status": "unused"}, "declared usable"),
        ({"noise_usable": "false"}, "declared usable"),
    ],
)
def test_diagonal_noise_requires_scope_source_and_usable_state(
    tmp_path: Path,
    overrides: dict[str, str],
    error_fragment: str,
) -> None:
    config = _config(noise_method="diagonal")
    plan = _plan(
        tmp_path,
        [_row("pattern-a", noise_method="diagonal", **overrides)],
        config=config,
    )

    assert not plan.valid
    assert any(error_fragment in error for error in plan.errors)


def test_identity_noise_must_explicitly_be_unused(tmp_path: Path) -> None:
    plan = _plan(tmp_path, [_row("pattern-a", noise_status="ok", noise_usable="true")])
    assert not plan.valid
    assert any("identity noise" in error for error in plan.errors)


def test_identity_noise_payload_is_ignored_with_an_auditable_warning(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        [
            _row(
                "pattern-a",
                noise_values="[9.0,8.0,7.0]",
                noise_feature_count="3",
                noise_voxel_order="c-order",
            )
        ],
    )
    result = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")

    assert result.valid
    assert result.rows[0]["noise_values"] == ()
    assert result.rows[0]["noise_usable"] is False
    assert any("unused under identity normalization" in warning for warning in result.warnings)


def test_diagonal_noise_must_match_across_conditions_for_one_unit_roi_cv(tmp_path: Path) -> None:
    config = _config(conditions=("condition-a", "condition-b"), noise_method="diagonal")
    plan = _plan(
        tmp_path,
        [
            _row("a", condition_id="condition-a", noise_method="diagonal"),
            _row("b", condition_id="condition-b", noise_method="diagonal", noise_values="[1.0,2.0,4.0]"),
        ],
        config=config,
    )
    assert plan.valid
    result = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")
    assert not result.valid
    assert any("differ across conditions" in error for error in result.errors)


def test_diagonal_noise_identity_mismatch_fails_during_scalar_planning(tmp_path: Path) -> None:
    config = _config(conditions=("condition-a", "condition-b"), noise_method="diagonal")
    plan = _plan(
        tmp_path,
        [
            _row("a", condition_id="condition-a", noise_method="diagonal"),
            _row(
                "b",
                condition_id="condition-b",
                noise_method="diagonal",
                noise_estimation_scope="different-scope",
            ),
        ],
        config=config,
    )

    assert not plan.valid
    assert not plan.ready_for_materialization
    assert any("noise identity metadata is inconsistent" in error for error in plan.errors)


@pytest.mark.parametrize(
    "overrides",
    [
        {"feature_values": "[1.0,2.0]", "feature_count": "2"},
        {"voxel_order": "different-order"},
        {"voxel_index_hash": "different-index"},
        {"feature_space_id": "different-space"},
        {"roi_definition_id": "different-roi"},
    ],
)
def test_feature_metadata_must_match_within_unit_roi_group(
    tmp_path: Path,
    overrides: dict[str, str],
) -> None:
    config = _config(conditions=("condition-a", "condition-b"))
    plan = _plan(
        tmp_path,
        [
            _row("a", condition_id="condition-a"),
            _row("b", condition_id="condition-b", **overrides),
        ],
        config=config,
    )
    assert not plan.valid
    assert not plan.ready_for_materialization
    assert any("feature metadata is inconsistent" in error for error in plan.errors)


def test_feature_metadata_must_match_across_runs_in_one_analysis_group(tmp_path: Path) -> None:
    units = (
        _unit("sub-toy01", "run-01", source_row=1),
        _unit("sub-toy01", "run-02", source_row=2),
    )
    plan = _plan(
        tmp_path,
        [
            _row("run-a", run_id="run-01"),
            _row("run-b", run_id="run-02", feature_space_id="space-b"),
        ],
        units=units,
    )

    assert not plan.valid
    assert not plan.ready_for_materialization
    assert any("feature metadata is inconsistent" in error for error in plan.errors)


def test_mean_centering_state_and_scope_must_match_configuration(tmp_path: Path) -> None:
    centered = _config(centered=True)
    valid_plan = _plan(
        tmp_path,
        [
            _row(
                "pattern-a",
                mean_centering_applied="true",
                mean_centering_scope="roi",
            )
        ],
        config=centered,
    )
    assert valid_plan.valid

    wrong_state = _plan(tmp_path / "state", [_row("pattern-a")], config=centered)
    wrong_scope = _plan(
        tmp_path / "scope",
        [_row("pattern-a", mean_centering_applied="true", mean_centering_scope="none")],
        config=centered,
    )
    assert any("mean-centering state" in error for error in wrong_state.errors)
    assert any("mean-centering scope" in error for error in wrong_scope.errors)


def test_digest_mismatch_and_post_plan_change_fail_before_decode(tmp_path: Path) -> None:
    plan = _plan(tmp_path, [_row("pattern-a")])
    wrong = load_materialized_pattern_table(plan, expected_sha256="0" * 64)
    assert not wrong.valid and wrong.rows == ()
    assert any("planned digest" in error for error in wrong.errors)

    table = tmp_path / "tables/patterns.tsv"
    table.write_text(table.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="")
    changed = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")
    assert not changed.valid and changed.rows == ()
    assert any("changed after planning" in error for error in changed.errors)


def test_event_threshold_requires_materialized_event_count(tmp_path: Path) -> None:
    config = _config()
    config.event_thresholds = SimpleNamespace(min_events_per_condition_per_run=1)
    plan = _plan(tmp_path, [_row("pattern-a", event_count="")], config=config)
    assert not plan.valid
    assert any("event_count is required" in error for error in plan.errors)


def test_materialized_event_thresholds_report_explicit_success(tmp_path: Path) -> None:
    config = _config()
    config.event_thresholds = SimpleNamespace(
        min_events_per_condition_per_run=3,
        min_runs_per_condition=2,
    )
    units = (
        _unit("sub-toy01", "run-01", source_row=1),
        _unit("sub-toy01", "run-02", source_row=2),
    )

    plan = _plan(
        tmp_path,
        [
            _row("pattern-run-01", run_id="run-01", event_count="3"),
            _row("pattern-run-02", run_id="run-02", event_count="5"),
        ],
        config=config,
        units=units,
    )

    assert plan.valid
    assert plan.ready_for_materialization
    assert plan.ready_for_execution
    assert plan.event_threshold_rows == (
        {
            "threshold": "min_events_per_condition_per_run",
            "value": 3,
            "status": "passed",
            "reason": "all_usable_rows_passed",
            "evaluated_row_count": 2,
            "missing_row_count": 0,
            "failing_row_count": 0,
        },
        {
            "threshold": "min_runs_per_condition",
            "value": 2,
            "status": "passed",
            "reason": "all_condition_roi_groups_passed",
            "evaluated_group_count": 1,
            "failing_group_count": 0,
            "missing_run_row_count": 0,
        },
    )
    assert (
        plan.public_summary()["execution_reason"]
        == "scalar_plan_usable_coverage_and_event_thresholds_ready"
    )


def test_min_runs_threshold_failure_is_audited_and_not_execution_ready(tmp_path: Path) -> None:
    config = _config()
    config.event_thresholds = SimpleNamespace(min_runs_per_condition=3)
    units = (
        _unit("sub-toy01", "run-01", source_row=1),
        _unit("sub-toy01", "run-02", source_row=2),
    )

    plan = _plan(
        tmp_path,
        [
            _row("pattern-run-01", run_id="run-01"),
            _row("pattern-run-02", run_id="run-02"),
        ],
        config=config,
        units=units,
    )

    assert not plan.valid
    assert not plan.ready_for_materialization
    assert not plan.ready_for_execution
    assert plan.event_threshold_rows == (
        {
            "threshold": "min_runs_per_condition",
            "value": 3,
            "status": "failed",
            "reason": "run_count_below_threshold",
            "evaluated_group_count": 1,
            "failing_group_count": 1,
            "missing_run_row_count": 0,
        },
    )
    assert plan.errors == (
        "Materialized pattern table does not satisfy configured min_runs_per_condition=3.",
    )
    assert (
        plan.public_summary()["execution_reason"]
        == "scalar_plan_usable_coverage_or_event_thresholds_not_ready"
    )


def test_min_events_threshold_failure_is_audited_and_not_execution_ready(tmp_path: Path) -> None:
    config = _config()
    config.event_thresholds = SimpleNamespace(min_events_per_condition_per_run=5)
    units = (
        _unit("sub-toy01", "run-01", source_row=1),
        _unit("sub-toy01", "run-02", source_row=2),
    )

    plan = _plan(
        tmp_path,
        [
            _row("pattern-run-01", run_id="run-01", event_count="4"),
            _row("pattern-run-02", run_id="run-02", event_count="6"),
        ],
        config=config,
        units=units,
    )

    assert not plan.valid
    assert not plan.ready_for_materialization
    assert not plan.ready_for_execution
    assert plan.event_threshold_rows == (
        {
            "threshold": "min_events_per_condition_per_run",
            "value": 5,
            "status": "failed",
            "reason": "event_count_below_threshold",
            "evaluated_row_count": 2,
            "missing_row_count": 0,
            "failing_row_count": 1,
        },
    )
    assert plan.errors == (
        "Materialized table data row 1 is usable despite failing its event-count threshold.",
        "Materialized pattern table does not satisfy configured "
        "min_events_per_condition_per_run=5.",
    )
    assert (
        plan.public_summary()["execution_reason"]
        == "scalar_plan_usable_coverage_or_event_thresholds_not_ready"
    )


def test_qc_and_exclusion_state_must_match_usability(tmp_path: Path) -> None:
    conflicting = _plan(
        tmp_path,
        [_row("pattern-a", exclusion_id="rule-a", exclusion_reason="Synthetic exclusion")],
    )
    assert not conflicting.valid
    assert any("conflict with usable=true" in error for error in conflicting.errors)

    unaudited = _plan(
        tmp_path / "unaudited",
        [_row("pattern-a", usable="false", status="excluded", qc_status="excluded", qc_reason="")],
    )
    assert not unaudited.valid
    assert any("auditable reason" in error for error in unaudited.errors)

    contradictory = _plan(
        tmp_path / "contradictory",
        [
            _row(
                "pattern-a",
                usable="false",
                status="excluded",
                qc_status="pass",
                qc_reason="Synthetic deterministic exclusion",
            )
        ],
    )
    assert not contradictory.valid
    assert any("QC status conflicts" in error for error in contradictory.errors)


def test_unusable_rows_remain_auditable_and_provenance_is_strict_json_safe(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        [
            _row(
                "pattern-excluded",
                usable="false",
                status="excluded",
                qc_status="excluded",
                qc_reason="Synthetic deterministic exclusion",
                exclusion_id="rule-toy",
                exclusion_reason="Synthetic deterministic exclusion",
            )
        ],
    )

    summary = plan.public_summary()
    assert plan.valid
    assert plan.ready_for_materialization
    assert not plan.ready_for_execution
    assert not plan.usable_coverage_complete
    assert summary["counts"] == {
        "total_rows": 1,
        "selected_rows": 1,
        "usable_selected_rows": 0,
        "usable_units": 0,
        "usable_conditions": 0,
        "usable_rois": 0,
        "unselected_rows": 0,
    }

    result = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")

    assert result.valid
    assert result.rows[0]["usable"] is False
    assert result.qc_rows[0]["status"] == "excluded"
    assert result.qc_rows[0]["exclusion_id"] == "rule-toy"
    json.dumps(result.provenance, sort_keys=True, allow_nan=False)


def test_partial_usable_coverage_is_not_execution_ready(tmp_path: Path) -> None:
    config = _config(conditions=("condition-a", "condition-b"))
    plan = _plan(
        tmp_path,
        [
            _row("usable", condition_id="condition-a"),
            _row(
                "excluded",
                condition_id="condition-b",
                usable="false",
                status="excluded",
                qc_status="excluded",
                qc_reason="Synthetic deterministic exclusion",
                exclusion_id="rule-toy",
                exclusion_reason="Synthetic deterministic exclusion",
            ),
        ],
        config=config,
    )

    assert plan.valid
    assert plan.ready_for_materialization
    assert not plan.ready_for_execution
    assert plan.usable_selected_row_count == 1
    assert not plan.usable_coverage_complete


def test_grouping_metadata_rejects_nonfinite_json_scalars(tmp_path: Path) -> None:
    plan = _plan(tmp_path, [_row("pattern-a", grouping_values='{ "score": NaN }')])

    result = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")

    assert not result.valid
    assert any("non-finite" in error for error in result.errors)


def test_cross_sectional_exact_unit_preserves_absent_optional_entities(tmp_path: Path) -> None:
    unit = ResolvedAnalysisUnit(
        unit_id="unit-cross-sectional",
        source_row=1,
        key_columns=("subject_id",),
        subject_id="sub-toy01",
        values={"subject_id": "sub-toy01"},
    )
    config = _config(cv_unit="subject")
    plan = _plan(
        tmp_path,
        [_row("pattern-a", session_id="", task_id="", run_id="", cross_validation_label="sub-toy01")],
        config=config,
        units=(unit,),
    )
    assert plan.valid
    assert plan.scalar_rows[0].session_id is None
    assert plan.scalar_rows[0].task_id is None
    assert plan.scalar_rows[0].run_id is None


def test_missing_selected_exact_unit_is_rejected(tmp_path: Path) -> None:
    units = (
        _unit("sub-toy01", "run-01", source_row=1),
        _unit("sub-toy02", "run-02", source_row=2),
    )
    plan = _plan(tmp_path, [_row("pattern-a")], units=units)

    assert not plan.valid
    assert any("does not fully cover" in error for error in plan.errors)


def test_custom_cv_label_is_derived_from_exact_unit_metadata(tmp_path: Path) -> None:
    config = _config(cv_unit="custom")
    plan = _plan(
        tmp_path,
        [_row("pattern-a", cross_validation_label="fold_id=fold-a")],
        config=config,
    )
    assert plan.valid
    assert plan.scalar_rows[0].cross_validation_label == "fold_id=fold-a"


def test_header_case_collisions_are_rejected(tmp_path: Path) -> None:
    table = tmp_path / "tables" / "patterns.tsv"
    row = _row("pattern-a")
    row["Subject_ID"] = "duplicate-case"
    columns = tuple(dict.fromkeys((*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, "Subject_ID")))
    _write_table(table, [row], columns=columns)
    plan = plan_materialized_pattern_table(
        _config(), _source(), (_unit("sub-toy01", "run-01"),), roots={"artifact_root": tmp_path}
    )
    assert not plan.valid
    assert any("case-insensitive" in error for error in plan.errors)


def test_nonscalar_extra_columns_are_rejected(tmp_path: Path) -> None:
    table = tmp_path / "tables" / "patterns.tsv"
    row = _row("pattern-a")
    row["adapter_metadata"] = "[1,2]"
    columns = tuple(dict.fromkeys((*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, "adapter_metadata")))
    _write_table(table, [row], columns=columns)
    plan = plan_materialized_pattern_table(
        _config(), _source(), (_unit("sub-toy01", "run-01"),), roots={"artifact_root": tmp_path}
    )
    assert not plan.valid
    assert any("scalar value" in error for error in plan.errors)


def test_extra_columns_must_not_collide_with_derived_row_fields(tmp_path: Path) -> None:
    table = tmp_path / "tables" / "patterns.tsv"
    row = _row("pattern-a")
    row["source_reference"] = "relative/reference.tsv"
    columns = tuple(dict.fromkeys((*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, "source_reference")))
    _write_table(table, [row], columns=columns)

    plan = plan_materialized_pattern_table(
        _config(), _source(), (_unit("sub-toy01", "run-01"),), roots={"artifact_root": tmp_path}
    )

    assert not plan.valid
    assert any("collides with a derived row field" in error for error in plan.errors)


def test_safe_table_extras_and_authoritative_unit_metadata_survive_loading(
    tmp_path: Path,
) -> None:
    table = tmp_path / "tables" / "patterns.tsv"
    row = _row("pattern-a")
    row["visit_index"] = "2"
    row["adapter_note"] = "synthetic-note"
    columns = tuple(
        dict.fromkeys((*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, "visit_index", "adapter_note"))
    )
    _write_table(table, [row], columns=columns)
    unit = _unit("sub-toy01", "run-01")
    unit = ResolvedAnalysisUnit(
        **{
            **unit.__dict__,
            "values": {**dict(unit.values), "visit_index": "2"},
        }
    )

    plan = plan_materialized_pattern_table(
        _config(), _source(), (unit,), roots={"artifact_root": tmp_path}
    )
    loaded = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")

    assert plan.valid
    assert plan.public_source_rows()[0]["visit_index"] == "2"
    assert plan.public_source_rows()[0]["adapter_note"] == "synthetic-note"
    assert loaded.valid
    assert loaded.rows[0]["visit_index"] == "2"
    assert loaded.rows[0]["adapter_note"] == "synthetic-note"
    assert loaded.rows[0]["unit_metadata"]["visit_index"] == "2"


def test_table_metadata_cannot_disagree_with_authoritative_exact_unit(tmp_path: Path) -> None:
    table = tmp_path / "tables" / "patterns.tsv"
    row = _row("pattern-a")
    row["visit_index"] = "99"
    columns = tuple(dict.fromkeys((*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, "visit_index")))
    _write_table(table, [row], columns=columns)
    unit = _unit("sub-toy01", "run-01")
    unit = ResolvedAnalysisUnit(
        **{
            **unit.__dict__,
            "values": {**dict(unit.values), "visit_index": "2"},
        }
    )

    plan = plan_materialized_pattern_table(
        _config(), _source(), (unit,), roots={"artifact_root": tmp_path}
    )

    assert not plan.valid
    assert any("authoritative exact unit" in error for error in plan.errors)


@pytest.mark.parametrize(
    "metadata",
    [
        {"input_note": "--input=/home/alice/private.tsv"},
        {"input_note": "file:///home/alice/private.tsv"},
        {"score": float("nan")},
        {"nested": {"value": "not-scalar"}},
    ],
)
def test_exact_unit_metadata_must_be_portable_finite_and_scalar(
    tmp_path: Path,
    metadata: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "subject_id": "sub-toy01",
        "run_id": "run-01",
        **metadata,
    }
    unit = ResolvedAnalysisUnit(
        unit_id="unit-1",
        source_row=1,
        key_columns=("subject_id", "run_id"),
        subject_id="sub-toy01",
        run_id="run-01",
        values=values,
    )
    table = tmp_path / "tables" / "patterns.tsv"
    _write_table(table, [_row("pattern-a", session_id="", task_id="")])

    plan = plan_materialized_pattern_table(
        _config(), _source(), (unit,), roots={"artifact_root": tmp_path}
    )

    assert not plan.valid
    assert any("Exact unit source row" in error for error in plan.errors)


@pytest.mark.parametrize(
    "column,value",
    [
        ("roi_reference", "/home/alice/private/roi.nii"),
        ("qc_reason", "file:///home/alice/private.tsv"),
        ("generator_version", "../private/tool"),
        ("warnings", '["C:\\\\Data\\\\private.tsv"]'),
    ],
)
def test_planner_rejects_embedded_local_path_references(tmp_path: Path, column: str, value: str) -> None:
    plan = _plan(tmp_path, [_row("pattern-a", **{column: value})])
    assert not plan.valid
    assert any("non-portable local path" in error for error in plan.errors)


def test_planning_and_loading_do_not_write_or_invoke_subprocesses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    table = tmp_path / "tables/patterns.tsv"
    _write_table(table, [_row("pattern-a")])
    before = (table.read_bytes(), table.stat().st_mtime_ns, set(tmp_path.rglob("*")))

    real_path_open = Path.open

    def read_only_open(path: Path, mode: str = "r", *args, **kwargs):
        if any(marker in mode for marker in ("w", "a", "x", "+")):
            raise AssertionError("materialized planning attempted a write")
        return real_path_open(path, mode, *args, **kwargs)

    def forbidden_subprocess(*_args, **_kwargs):
        raise AssertionError("materialized planning invoked a subprocess")

    monkeypatch.setattr(Path, "open", read_only_open)
    monkeypatch.setattr(subprocess, "run", forbidden_subprocess)
    plan = plan_materialized_pattern_table(
        _config(), _source(), (_unit("sub-toy01", "run-01"),), roots={"artifact_root": tmp_path}
    )
    loaded = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256 or "")
    after = (table.read_bytes(), table.stat().st_mtime_ns, set(tmp_path.rglob("*")))
    assert loaded.valid
    assert after == before
