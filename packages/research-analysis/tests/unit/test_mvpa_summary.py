from __future__ import annotations

import math

import pytest

from research_platform.analysis.mvpa import (
    METRIC_CROSSNOBIS,
    DistanceEstimate,
    distance_summary_rows_from_estimates,
    distance_summary_rows_from_long_rows,
)


def test_grouped_long_rows_are_summarized_deterministically() -> None:
    rows = [
        {"contrast": "b_vs_c", "metric": "crossnobis", "distance": 6.0},
        {"contrast": "a_vs_b", "metric": "crossnobis", "distance": 1.0},
        {"contrast": "b_vs_c", "metric": "crossnobis", "distance": 4.0},
        {"contrast": "a_vs_b", "metric": "crossnobis", "distance": 3.0},
    ]

    summary_rows = distance_summary_rows_from_long_rows(rows, group_by=["contrast", "metric"])

    assert [list(row) for row in summary_rows] == [
        [
            "contrast",
            "metric",
            "n",
            "mean_distance",
            "std_distance",
            "sem_distance",
            "min_distance",
            "max_distance",
        ],
        [
            "contrast",
            "metric",
            "n",
            "mean_distance",
            "std_distance",
            "sem_distance",
            "min_distance",
            "max_distance",
        ],
    ]
    assert summary_rows == [
        {
            "contrast": "a_vs_b",
            "metric": "crossnobis",
            "n": 2,
            "mean_distance": 2.0,
            "std_distance": pytest.approx(math.sqrt(2.0)),
            "sem_distance": pytest.approx(1.0),
            "min_distance": 1.0,
            "max_distance": 3.0,
        },
        {
            "contrast": "b_vs_c",
            "metric": "crossnobis",
            "n": 2,
            "mean_distance": 5.0,
            "std_distance": pytest.approx(math.sqrt(2.0)),
            "sem_distance": pytest.approx(1.0),
            "min_distance": 4.0,
            "max_distance": 6.0,
        },
    ]


def test_distance_estimates_can_group_by_context_and_preserve_negative_distances() -> None:
    estimates = [
        _estimate("a", "b", -2.0, context={"analysis_unit": "unit-a"}),
        _estimate("a", "c", 4.0, context={"analysis_unit": "unit-a"}),
        _estimate("b", "c", 7.5, context={"analysis_unit": "unit-b"}),
    ]

    summary_rows = distance_summary_rows_from_estimates(estimates, group_by=["analysis_unit"])

    assert summary_rows == [
        {
            "analysis_unit": "unit-a",
            "n": 2,
            "mean_distance": 1.0,
            "std_distance": pytest.approx(math.sqrt(18.0)),
            "sem_distance": pytest.approx(3.0),
            "min_distance": -2.0,
            "max_distance": 4.0,
        },
        {
            "analysis_unit": "unit-b",
            "n": 1,
            "mean_distance": 7.5,
            "std_distance": 0.0,
            "sem_distance": 0.0,
            "min_distance": 7.5,
            "max_distance": 7.5,
        },
    ]


def test_empty_long_rows_with_no_groups_return_empty_summary_row() -> None:
    assert distance_summary_rows_from_long_rows([], group_by=()) == [
        {
            "n": 0,
            "mean_distance": None,
            "std_distance": None,
            "sem_distance": None,
            "min_distance": None,
            "max_distance": None,
        }
    ]


def test_empty_long_rows_with_groups_return_no_rows() -> None:
    assert distance_summary_rows_from_long_rows([], group_by=["contrast"]) == []


def test_singleton_long_row_summary_uses_zero_std_and_sem() -> None:
    assert distance_summary_rows_from_long_rows([{"distance": 7.5}], group_by=()) == [
        {
            "n": 1,
            "mean_distance": 7.5,
            "std_distance": 0.0,
            "sem_distance": 0.0,
            "min_distance": 7.5,
            "max_distance": 7.5,
        }
    ]


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"distance": math.nan},
        {"distance": math.inf},
        {"distance": -math.inf},
        {"distance": True},
        {"distance": "not-numeric"},
    ],
)
def test_invalid_or_missing_long_row_distances_raise_value_error(row: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        distance_summary_rows_from_long_rows([row], group_by=())


def test_missing_requested_long_row_group_field_raises_value_error() -> None:
    with pytest.raises(ValueError, match="missing requested group field"):
        distance_summary_rows_from_long_rows([{"distance": 1.0}], group_by=["contrast"])


def test_empty_group_by_field_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty field"):
        distance_summary_rows_from_long_rows([{"distance": 1.0}], group_by=[""])


def test_duplicate_group_by_field_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unique"):
        distance_summary_rows_from_long_rows(
            [{"distance": 1.0, "contrast": "a_vs_b"}],
            group_by=["contrast", "contrast"],
        )


@pytest.mark.parametrize("context_key", ["metric", "n"])
def test_distance_estimate_context_key_collisions_raise_value_error(context_key: str) -> None:
    with pytest.raises(ValueError, match="collides"):
        distance_summary_rows_from_estimates(
            [_estimate("a", "b", 1.0, context={context_key: "collision"})],
            group_by=(),
        )


def _estimate(
    condition_id_a: str,
    condition_id_b: str,
    distance: float,
    *,
    context: dict[str, object] | None = None,
) -> DistanceEstimate:
    return DistanceEstimate(
        condition_id_a=condition_id_a,
        condition_id_b=condition_id_b,
        distance=distance,
        metric=METRIC_CROSSNOBIS,
        engine_name="native_reference",
        normalization_method="identity",
        cv_unit_count=2,
        context=context or {},
    )
