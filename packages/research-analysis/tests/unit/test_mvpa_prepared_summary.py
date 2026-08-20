from __future__ import annotations

import ast
import builtins
import json
import math
from pathlib import Path

import pytest

import research_platform.analysis.mvpa.prepared_summary as prepared_summary
from research_platform.analysis.mvpa import (
    DEFAULT_PREPARED_MVPA_DISTANCE_SUMMARY_GROUP_BY,
    PreparedMvpaDistanceResult,
    PreparedMvpaDistanceRow,
    prepared_mvpa_distance_summary_rows,
    summarize_prepared_mvpa_distances,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "group_id": "g1",
        "group_key": {"subject_id": "sub-01", "group_id": "shadowed"},
        "condition_id_a": "face",
        "condition_id_b": "house",
        "distance": 1.0,
        "metric": "crossnobis",
        "engine_name": "native_reference",
        "normalization_method": "identity",
    }
    row.update(overrides)
    return row


def _distance(**overrides: object) -> PreparedMvpaDistanceRow:
    row = _row(**overrides)
    return PreparedMvpaDistanceRow(
        group_id=str(row["group_id"]),
        group_key=row["group_key"],
        condition_id_a=str(row["condition_id_a"]),
        condition_id_b=str(row["condition_id_b"]),
        distance=row["distance"],
        metric=str(row["metric"]),
        engine_name=str(row["engine_name"]),
        normalization_method=str(row["normalization_method"]),
    )


def _codes(result) -> set[str]:
    return {qc.code for qc in result.qc_rows}


def _failed_codes(result) -> set[str]:
    return {qc.code for qc in result.qc_rows if qc.status == "failed"}


def _provenance(result) -> dict[str, object]:
    return {row.key: row.value for row in result.provenance}


def test_grouped_summaries_from_synthetic_prepared_distance_row_mappings() -> None:
    rows = [
        _row(group_key={"subject_id": "sub-02"}, distance=4.0),
        _row(group_key={"subject_id": "sub-01"}, distance=1.0),
        _row(group_key={"subject_id": "sub-01"}, distance=3.0),
    ]

    result = summarize_prepared_mvpa_distances(
        rows,
        group_by=(
            "subject_id",
            "condition_id_a",
            "condition_id_b",
            "metric",
            "engine_name",
            "normalization_method",
        ),
    )

    assert result.errors == ()
    assert result.summary_rows == (
        {
            "subject_id": "sub-01",
            "condition_id_a": "face",
            "condition_id_b": "house",
            "metric": "crossnobis",
            "engine_name": "native_reference",
            "normalization_method": "identity",
            "n": 2,
            "mean_distance": 2.0,
            "std_distance": pytest.approx(math.sqrt(2.0)),
            "sem_distance": pytest.approx(1.0),
            "min_distance": 1.0,
            "max_distance": 3.0,
        },
        {
            "subject_id": "sub-02",
            "condition_id_a": "face",
            "condition_id_b": "house",
            "metric": "crossnobis",
            "engine_name": "native_reference",
            "normalization_method": "identity",
            "n": 1,
            "mean_distance": 4.0,
            "std_distance": 0.0,
            "sem_distance": 0.0,
            "min_distance": 4.0,
            "max_distance": 4.0,
        },
    )
    assert list(result.summary_rows[0]) == [
        "subject_id",
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


def test_grouping_columns_and_condition_pair_id_are_preserved_in_summary_rows() -> None:
    result = summarize_prepared_mvpa_distances(
        [
            {
                "group_id": "group-alpha",
                "group_key": {"participant_id": "participant-a", "task_id": "task-alpha"},
                "condition_id_a": "condition-a",
                "condition_id_b": "condition-b",
                "condition_pair_id": "pair-alpha",
                "distance": 1.0,
                "metric": "crossnobis",
                "engine_name": "native_reference",
                "normalization_method": "identity",
            },
            {
                "group_id": "group-alpha",
                "group_key": {"participant_id": "participant-a", "task_id": "task-alpha"},
                "condition_id_a": "condition-a",
                "condition_id_b": "condition-b",
                "condition_pair_id": "pair-alpha",
                "distance": 3.0,
                "metric": "crossnobis",
                "engine_name": "native_reference",
                "normalization_method": "identity",
            },
        ],
        group_by=(
            "participant_id",
            "task_id",
            "condition_pair_id",
            "metric",
            "engine_name",
            "normalization_method",
        ),
    )

    assert result.errors == ()
    assert result.summary_rows == (
        {
            "participant_id": "participant-a",
            "task_id": "task-alpha",
            "condition_pair_id": "pair-alpha",
            "metric": "crossnobis",
            "engine_name": "native_reference",
            "normalization_method": "identity",
            "n": 2,
            "mean_distance": 2.0,
            "std_distance": pytest.approx(math.sqrt(2.0)),
            "sem_distance": pytest.approx(1.0),
            "min_distance": 1.0,
            "max_distance": 3.0,
        },
    )


def test_negative_distances_contribute_normally_to_global_summary() -> None:
    result = summarize_prepared_mvpa_distances(
        [_row(distance=-2.0), _row(distance=4.0)],
        group_by=(),
    )

    assert result.summary_rows == (
        {
            "n": 2,
            "mean_distance": 1.0,
            "std_distance": pytest.approx(math.sqrt(18.0)),
            "sem_distance": pytest.approx(3.0),
            "min_distance": -2.0,
            "max_distance": 4.0,
        },
    )


def test_summary_row_ordering_is_deterministic_for_unsorted_inputs() -> None:
    rows = [
        _row(group_id="g2", distance=2.0),
        _row(group_id="g1", distance=1.0),
        _row(group_id="g3", distance=3.0),
    ]

    summary_rows = prepared_mvpa_distance_summary_rows(rows)

    assert [row["group_id"] for row in summary_rows] == ["g1", "g2", "g3"]


def test_empty_input_default_grouping_returns_no_summary_rows_with_warning_qc() -> None:
    result = summarize_prepared_mvpa_distances([])

    assert result.summary_rows == ()
    assert _codes(result) == {"empty_input"}
    assert result.qc_rows[0].status == "warning"
    assert result.warnings
    provenance = _provenance(result)
    assert provenance["input_row_count"] == 0
    assert provenance["valid_distance_row_count"] == 0
    assert provenance["invalid_distance_row_count"] == 0
    assert provenance["summary_row_count"] == 0
    assert provenance["group_by"] == DEFAULT_PREPARED_MVPA_DISTANCE_SUMMARY_GROUP_BY
    assert provenance["grouping_policy"] == "default_non_collapsing"


def test_empty_input_global_grouping_returns_empty_summary_row_with_warning_qc() -> None:
    result = summarize_prepared_mvpa_distances([], group_by=())

    assert result.summary_rows == (
        {
            "n": 0,
            "mean_distance": None,
            "std_distance": None,
            "sem_distance": None,
            "min_distance": None,
            "max_distance": None,
        },
    )
    assert _codes(result) == {"empty_input"}
    assert _provenance(result)["grouping_policy"] == "global"


def test_missing_distance_creates_failed_qc_and_strict_wrapper_raises() -> None:
    row = _row()
    row.pop("distance")

    result = summarize_prepared_mvpa_distances([row])

    assert _failed_codes(result) == {"missing_distance"}
    assert result.errors
    with pytest.raises(ValueError, match="missing_distance"):
        prepared_mvpa_distance_summary_rows([row])


def test_nonnumeric_distance_creates_failed_qc() -> None:
    result = summarize_prepared_mvpa_distances([_row(distance="not-numeric")])

    assert _failed_codes(result) == {"invalid_distance"}


def test_bool_distance_creates_failed_qc() -> None:
    result = summarize_prepared_mvpa_distances([_row(distance=True)])

    assert _failed_codes(result) == {"invalid_distance"}


@pytest.mark.parametrize("distance", [math.nan, math.inf, -math.inf])
def test_nan_and_inf_distances_create_failed_qc(distance: float) -> None:
    result = summarize_prepared_mvpa_distances([_row(distance=distance)])

    assert _failed_codes(result) == {"invalid_distance"}


def test_missing_requested_group_field_creates_failed_qc() -> None:
    result = summarize_prepared_mvpa_distances([_row(group_key={})], group_by=("subject_id",))

    assert _failed_codes(result) == {"missing_group_field"}
    assert result.qc_rows[0].field_name == "subject_id"


def test_invalid_group_key_creates_failed_qc() -> None:
    result = summarize_prepared_mvpa_distances([_row(group_key="not-a-mapping")], group_by=())

    assert _failed_codes(result) == {"invalid_group_key"}


def test_summary_from_prepared_mvpa_distance_result_object() -> None:
    source = PreparedMvpaDistanceResult(
        distances=(
            _distance(distance=1.0),
            _distance(distance=3.0),
        ),
        qc_rows=(),
        provenance=(),
        warnings=(),
        errors=(),
    )

    result = summarize_prepared_mvpa_distances(source, group_by=())

    assert result.errors == ()
    assert result.summary_rows[0]["n"] == 2
    assert result.summary_rows[0]["mean_distance"] == 2.0


def test_summary_from_prepared_mvpa_distance_result_to_dict_mapping() -> None:
    source = PreparedMvpaDistanceResult(
        distances=(_distance(distance=7.5),),
        qc_rows=(),
        provenance=(),
        warnings=(),
        errors=(),
    )

    result = summarize_prepared_mvpa_distances(source.to_dict(), group_by=())

    assert result.errors == ()
    assert result.summary_rows == (
        {
            "n": 1,
            "mean_distance": 7.5,
            "std_distance": 0.0,
            "sem_distance": 0.0,
            "min_distance": 7.5,
            "max_distance": 7.5,
        },
    )


def test_result_to_dict_is_json_safe_with_strict_nan_policy() -> None:
    result = summarize_prepared_mvpa_distances([_row(distance=1.5)], group_by=())

    json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)


def test_no_output_writing_occurs(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = builtins.open

    def guarded_open(file, mode: str = "r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"unexpected write open for {file!r}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    result = summarize_prepared_mvpa_distances([_row()], group_by=())

    assert result.summary_rows[0]["n"] == 1
    assert _provenance(result)["output_written"] is False


def test_mixed_metadata_not_grouped_adds_warning_qc() -> None:
    result = summarize_prepared_mvpa_distances(
        [
            _row(metric="crossnobis", distance=1.0),
            _row(metric="euclidean", distance=2.0),
        ],
        group_by=("group_id",),
    )

    assert _failed_codes(result) == set()
    assert "mixed_metadata_not_grouped" in _codes(result)
    assert result.warnings


def test_unsupported_row_shape_creates_failed_qc() -> None:
    result = summarize_prepared_mvpa_distances([object()], group_by=())

    assert _failed_codes(result) == {"unsupported_row_shape"}


def test_forbidden_import_guard_for_prepared_summary_module_and_tests() -> None:
    forbidden_modules = (
        "research_platform.neuro",
        "research_platform.bids",
        "research_platform.core",
        "research_platform.viz",
        "research_platform.ml",
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "nilearn",
        "rsatoolbox",
        "sklearn",
        "nibabel",
        "pipelines",
        "ops",
    )

    imported_modules: list[str] = []
    for path in (Path(prepared_summary.__file__), Path(__file__)):
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
