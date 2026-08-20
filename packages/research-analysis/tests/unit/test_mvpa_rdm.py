from __future__ import annotations

import pytest

from research_platform.analysis.mvpa import METRIC_CROSSNOBIS, DistanceEstimate
from research_platform.analysis.mvpa import rdm_long_rows_from_estimates, rdm_wide_rows_from_estimates


def _estimate(condition_id_a: str, condition_id_b: str, distance: float) -> DistanceEstimate:
    return DistanceEstimate(
        condition_id_a=condition_id_a,
        condition_id_b=condition_id_b,
        distance=distance,
        metric=METRIC_CROSSNOBIS,
        engine_name="fake",
        normalization_method="identity",
        cv_unit_count=2,
    )


def test_rdm_long_rows_are_deterministic() -> None:
    rows = rdm_long_rows_from_estimates(
        [
            _estimate("c", "b", 1.0),
            _estimate("b", "a", 2.0),
            _estimate("c", "a", 3.0),
        ],
        condition_order=["a", "b", "c"],
    )

    assert [(row["condition_id_a"], row["condition_id_b"], row["distance"]) for row in rows] == [
        ("a", "b", 2.0),
        ("a", "c", 3.0),
        ("b", "c", 1.0),
    ]
    assert rows[0]["metric"] == METRIC_CROSSNOBIS
    assert rows[0]["engine_name"] == "fake"


def test_rdm_wide_rows_are_symmetric_and_diagonal_zero() -> None:
    rows = rdm_wide_rows_from_estimates(
        [
            _estimate("c", "b", 1.0),
            _estimate("b", "a", 2.0),
            _estimate("c", "a", 3.0),
        ],
        condition_order=["a", "b", "c"],
    )

    assert rows == [
        {"condition_id": "a", "a": 0.0, "b": 2.0, "c": 3.0},
        {"condition_id": "b", "a": 2.0, "b": 0.0, "c": 1.0},
        {"condition_id": "c", "a": 3.0, "b": 1.0, "c": 0.0},
    ]


def test_rdm_wide_rejects_duplicate_empty_and_index_collision_condition_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        rdm_wide_rows_from_estimates([], condition_order=["a", "a"])
    with pytest.raises(ValueError, match="empty"):
        rdm_wide_rows_from_estimates([], condition_order=["a", ""])
    with pytest.raises(ValueError, match="index column"):
        rdm_wide_rows_from_estimates([], condition_order=["a", "condition_id"])
