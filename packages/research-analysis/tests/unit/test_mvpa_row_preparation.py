from __future__ import annotations

import ast
import builtins
import csv
import io
import json
import math
from pathlib import Path

import pytest

from research_platform.analysis.mvpa import (
    DEFAULT_MVPA_PATTERN_GROUP_BY,
    compute_mvpa_distances_from_prepared_groups,
    prepare_mvpa_pattern_row_groups,
)
from research_platform.analysis.mvpa import row_preparation


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "condition_id": "face",
        "feature_values": [1.0, -2.0],
        "feature_count": 2,
        "subject_id": "sub-01",
        "session_id": "ses-01",
        "run_id": "run-1",
        "task_id": "localizer",
        "direction": "ap",
        "model": "first_level",
        "pattern_source_name": "cope",
        "roi_source_name": "atlas",
        "roi_label": "ffa",
        "voxel_order": "c_flat_index",
        "voxel_index_hash": "hash-a",
        "usable": True,
    }
    row.update(overrides)
    return row


def _codes(result) -> set[str]:
    return {qc.code for qc in result.qc_rows}


def _provenance(result) -> dict[str, object]:
    return {row.key: row.value for row in result.provenance}


def test_valid_extraction_like_rows_are_grouped_deterministically() -> None:
    result = prepare_mvpa_pattern_row_groups(
        [
            _row(subject_id="sub-02", condition_id="face", run_id="run-2"),
            _row(subject_id="sub-01", condition_id="house", run_id="run-1"),
            _row(subject_id="sub-02", condition_id="house", run_id="run-1"),
        ]
    )

    assert [group.group_key["subject_id"] for group in result.groups] == ["sub-01", "sub-02"]
    assert result.groups[0].group_by == DEFAULT_MVPA_PATTERN_GROUP_BY
    assert result.groups[1].condition_ids == ("face", "house")
    assert result.groups[1].cv_labels == ("run-2", "run-1")
    assert result.errors == ()


def test_cv_unit_run_derives_fold_labels_from_run_id() -> None:
    result = prepare_mvpa_pattern_row_groups([_row(run_id="run-actual")], cv_unit="run")

    prepared = result.groups[0].rows[0]
    assert prepared.cv_unit == "run"
    assert prepared.cv_label == "run-actual"


def test_default_cv_label_derivation_ignores_unrequested_canonical_column() -> None:
    result = prepare_mvpa_pattern_row_groups(
        [_row(run_id="run-existing", cross_validation_label="fold-canonical")],
        cv_unit="run",
    )

    assert result.groups[0].rows[0].cv_label == "run-existing"
    assert _provenance(result)["cv_label_column"] == "run_id"


@pytest.mark.parametrize("cv_unit", ["run", "session", "subject", "custom"])
def test_canonical_cv_label_column_is_preserved_for_any_cv_unit(cv_unit: str) -> None:
    result = prepare_mvpa_pattern_row_groups(
        [
            _row(
                run_id="run-existing",
                session_id="ses-existing",
                subject_id="sub-existing",
                cv_unit="fold-existing",
                cross_validation_label="fold-visit-02",
            )
        ],
        cv_unit=cv_unit,
        cv_label_column="cross_validation_label",
    )

    prepared = result.groups[0].rows[0]
    assert prepared.cv_unit == cv_unit
    assert prepared.cv_label == "fold-visit-02"
    assert result.groups[0].cv_labels == ("fold-visit-02",)
    assert _provenance(result)["cv_label_column"] == "cross_validation_label"


def test_explicit_identity_noise_contract_does_not_warn_about_an_unused_vector() -> None:
    rows = [
        _row(condition_id="face", run_id="run-1"),
        _row(condition_id="house", run_id="run-2"),
    ]
    for row in rows:
        row.update(
            {
                "noise_status": "unused",
                "noise_usable": False,
                "noise_values": (),
                "noise_feature_count": 0,
                "noise_voxel_index_hash": None,
            }
        )

    result = prepare_mvpa_pattern_row_groups(rows)

    assert result.errors == ()
    assert not any(row.code in {"invalid_noise_values", "missing_noise_values"} for row in result.qc_rows)


def test_loader_shaped_tsv_mapping_preserves_canonical_cv_label() -> None:
    stream = io.StringIO()
    fieldnames = (
        "pattern_id",
        "condition_id",
        "cross_validation_label",
        "subject_id",
        "feature_values",
        "feature_count",
        "usable",
    )
    writer = csv.DictWriter(stream, delimiter="\t", lineterminator="\n", fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(
        {
            "pattern_id": "pattern-toy-a",
            "condition_id": "condition-a",
            "cross_validation_label": "fold-irregular-03",
            "subject_id": "sub-toy01",
            "feature_values": json.dumps([1.25, -0.5], separators=(",", ":")),
            "feature_count": "2",
            "usable": "true",
        }
    )
    stream.seek(0)
    loaded = dict(next(csv.DictReader(stream, delimiter="\t")))
    loaded["feature_values"] = json.loads(loaded["feature_values"])

    result = prepare_mvpa_pattern_row_groups(
        [loaded],
        group_by=("subject_id",),
        cv_unit="custom",
        cv_label_column="cross_validation_label",
    )

    prepared = result.groups[0].rows[0]
    assert result.errors == ()
    assert prepared.cv_label == "fold-irregular-03"
    assert prepared.feature_values == (1.25, -0.5)
    assert prepared.feature_count == 2


def test_canonical_cv_labels_remain_compatible_with_existing_distance_computation() -> None:
    rows = [
        _row(
            pattern_id=f"pattern-{fold}-{condition}",
            condition_id=condition,
            cross_validation_label=fold,
            run_id=f"legacy-{index}",
            feature_values=features,
        )
        for index, (fold, condition, features) in enumerate(
            (
                ("fold-a", "condition-a", [1.0, 0.0]),
                ("fold-a", "condition-b", [0.0, 0.0]),
                ("fold-b", "condition-a", [3.0, 0.0]),
                ("fold-b", "condition-b", [0.0, 0.0]),
            ),
            start=1,
        )
    ]
    prepared = prepare_mvpa_pattern_row_groups(
        rows,
        cv_unit="run",
        cv_label_column="cross_validation_label",
    )

    result = compute_mvpa_distances_from_prepared_groups(prepared.groups)

    assert prepared.errors == ()
    assert prepared.groups[0].cv_labels == ("fold-a", "fold-b")
    assert result.errors == ()
    assert len(result.distances) == 1
    assert result.distances[0].cv_unit_count == 2
    assert result.distances[0].observation_count == 4


@pytest.mark.parametrize(
    "cv_unit,column,value",
    [
        ("session", "session_id", "ses-actual"),
        ("subject", "subject_id", "sub-actual"),
        ("custom", "cv_unit", "fold-actual"),
    ],
)
def test_cv_unit_modes_derive_actual_fold_labels(cv_unit: str, column: str, value: str) -> None:
    result = prepare_mvpa_pattern_row_groups([_row(**{column: value})], cv_unit=cv_unit)

    assert result.groups[0].rows[0].cv_label == value


def test_unusable_rows_are_excluded_from_groups_and_emit_qc() -> None:
    result = prepare_mvpa_pattern_row_groups([_row(usable=False)])

    assert result.groups == ()
    assert _codes(result) == {"row_unusable"}
    assert result.qc_rows[0].status == "excluded"


def test_missing_condition_id_produces_qc_failure() -> None:
    result = prepare_mvpa_pattern_row_groups([_row(condition_id=" ")])

    assert result.groups == ()
    assert "missing_condition_id" in _codes(result)
    assert result.errors


def test_missing_cv_label_produces_qc_failure() -> None:
    result = prepare_mvpa_pattern_row_groups([_row(run_id=None)])

    assert result.groups == ()
    assert "missing_cv_label" in _codes(result)


@pytest.mark.parametrize("bad_value", ["bad", True, math.nan, math.inf, -math.inf])
def test_invalid_feature_values_produce_qc_failure(bad_value: object) -> None:
    result = prepare_mvpa_pattern_row_groups([_row(feature_values=[1.0, bad_value])])

    assert result.groups == ()
    assert "invalid_feature_values" in _codes(result)


def test_feature_count_mismatch_produces_qc_failure() -> None:
    result = prepare_mvpa_pattern_row_groups([_row(feature_count=3)])

    assert result.groups == ()
    assert "feature_count_mismatch" in _codes(result)


def test_feature_width_mismatch_within_group_produces_group_qc_failure() -> None:
    result = prepare_mvpa_pattern_row_groups(
        [
            _row(pattern_id="p1", feature_values=[1.0, 2.0], feature_count=2),
            _row(pattern_id="p2", condition_id="house", feature_values=[1.0, 2.0, 3.0], feature_count=3),
        ]
    )

    assert result.groups == ()
    assert "feature_width_mismatch" in _codes(result)
    assert result.qc_rows[-1].level == "group"


def test_voxel_index_hash_mismatch_within_group_produces_group_qc_failure() -> None:
    result = prepare_mvpa_pattern_row_groups(
        [
            _row(pattern_id="p1", voxel_index_hash="hash-a"),
            _row(pattern_id="p2", condition_id="house", voxel_index_hash="hash-b"),
        ]
    )

    assert result.groups == ()
    assert "voxel_index_hash_mismatch" in _codes(result)


def test_valid_noise_values_are_carried_but_not_used_for_computation() -> None:
    result = prepare_mvpa_pattern_row_groups(
        [
            _row(
                noise_values=[0.5, 1.5],
                noise_usable=True,
                noise_feature_count=2,
                noise_voxel_index_hash="noise-hash",
            )
        ]
    )

    prepared = result.groups[0].rows[0]
    assert prepared.noise_values == (0.5, 1.5)
    assert prepared.noise_usable is True
    assert prepared.noise_feature_count == 2
    assert prepared.noise_voxel_index_hash == "noise-hash"
    assert _provenance(result)["distance_computation"] is False


def test_grouping_columns_mean_centering_and_event_count_are_preserved_in_prepared_rows() -> None:
    row = {
        "pattern_id": "pattern-alpha",
        "condition_id": "condition-a",
        "run_id": "run-a",
        "participant_id": "participant-a",
        "session_id": "session-a",
        "task_id": "task-alpha",
        "roi_label": "roi-alpha",
        "feature_values": [1.0, -1.0],
        "feature_count": 2,
        "usable": True,
        "mean_centering_applied": True,
        "mean_centering_scope": "roi",
        "event_count": 4,
    }

    result = prepare_mvpa_pattern_row_groups([row], group_by=("participant_id", "task_id"))

    assert result.errors == ()
    prepared = result.groups[0].rows[0]
    assert prepared.group_key == {"participant_id": "participant-a", "task_id": "task-alpha"}
    assert prepared.mean_centering_applied is True
    assert prepared.mean_centering_scope == "roi"
    assert prepared.event_count == 4
    assert result.groups[0].group_by == ("participant_id", "task_id")


def test_min_events_threshold_failure_records_json_safe_context() -> None:
    result = prepare_mvpa_pattern_row_groups(
        [
            {
                "pattern_id": "pattern-alpha",
                "condition_id": "condition-a",
                "run_id": "run-a",
                "subject_id": "participant-a",
                "session_id": "session-a",
                "task_id": "task-alpha",
                "roi_label": "roi-alpha",
                "feature_values": [1.0, -1.0],
                "feature_count": 2,
                "usable": True,
                "event_count": 1,
            }
        ],
        threshold_sweeps=[{"id": "threshold-events-high", "min_events": 2}],
    )

    assert result.groups == ()
    assert result.errors == ()
    assert result.warnings
    assert _codes(result) == {"threshold_failure"}
    qc = result.qc_rows[0]
    assert qc.context == {
        "threshold_id": "threshold-events-high",
        "threshold_type": "min_events",
        "required_value": 2,
        "observed_value": 1,
        "failure_reason": "observed_events_below_required_minimum",
        "condition_id": "condition-a",
        "subject_id": "participant-a",
        "session_id": "session-a",
        "run_id": "run-a",
        "roi_label": "roi-alpha",
    }
    assert _provenance(result)["threshold_failure_count"] == 1
    json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)


def test_invalid_or_missing_noise_is_recorded_without_blocking_preparation() -> None:
    result = prepare_mvpa_pattern_row_groups(
        [
            _row(pattern_id="p1", noise_values=[1.0, "bad"]),
            _row(pattern_id="p2", condition_id="house", noise_usable=False),
        ]
    )

    assert len(result.groups) == 1
    assert {"invalid_noise_values", "missing_noise_values"} <= _codes(result)
    assert result.errors == ()
    assert result.warnings


def test_result_to_dict_is_json_safe_with_strict_nan_policy() -> None:
    result = prepare_mvpa_pattern_row_groups([_row()])

    json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)


def test_no_output_writing_occurs(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = builtins.open

    def guarded_open(file, mode: str = "r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"unexpected write open for {file!r}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    result = prepare_mvpa_pattern_row_groups([_row()])

    assert len(result.groups) == 1
    assert _provenance(result)["output_written"] is False


def test_forbidden_import_guard_for_row_preparation_module_and_tests() -> None:
    forbidden_modules = (
        "research_platform.neuro",
        "research_platform.bids",
        "research_platform.core",
        "research_platform.viz",
        "research_platform.ml",
        "nibabel",
        "nilearn",
        "rsatoolbox",
        "pymvpa",
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "sklearn",
        "pipelines",
        "ops",
    )

    imported_modules: list[str] = []
    for path in (Path(row_preparation.__file__), Path(__file__)):
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
