from __future__ import annotations

import pytest

from research_platform.analysis.mvpa import CrossValidationSpec, condition_order, cv_units_by_condition
from research_platform.analysis.mvpa import pattern_dataset_from_rows, validate_cross_validation


def _dataset(rows: list[dict[str, object]]):
    return pattern_dataset_from_rows(rows, feature_columns=["f1"])


def test_condition_order_is_first_appearance_or_explicit() -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "b", "cv_unit": "run-1", "f1": 1},
            {"pattern_id": "p2", "condition_id": "a", "cv_unit": "run-1", "f1": 2},
            {"pattern_id": "p3", "condition_id": "b", "cv_unit": "run-2", "f1": 3},
            {"pattern_id": "p4", "condition_id": "a", "cv_unit": "run-2", "f1": 4},
        ]
    )

    assert condition_order(dataset) == ("b", "a")
    assert condition_order(dataset, explicit_order=["a", "b"]) == ("a", "b")


def test_cv_unit_grouping_by_condition_is_deterministic() -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "b", "cv_unit": "run-1", "f1": 1},
            {"pattern_id": "p2", "condition_id": "a", "cv_unit": "run-1", "f1": 2},
            {"pattern_id": "p3", "condition_id": "b", "cv_unit": "run-2", "f1": 3},
            {"pattern_id": "p4", "condition_id": "a", "cv_unit": "run-2", "f1": 4},
        ]
    )

    assert cv_units_by_condition(dataset) == {
        "b": ("run-1", "run-2"),
        "a": ("run-1", "run-2"),
    }


def test_validate_cross_validation_accepts_balanced_two_condition_two_unit_data() -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 2},
            {"pattern_id": "p3", "condition_id": "a", "cv_unit": "run-2", "f1": 3},
            {"pattern_id": "p4", "condition_id": "b", "cv_unit": "run-2", "f1": 4},
        ]
    )

    assert validate_cross_validation(dataset, CrossValidationSpec()) == ("a", "b")


def test_validate_cross_validation_rejects_fewer_than_minimum_cv_units() -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 2},
        ]
    )

    with pytest.raises(ValueError, match="at least 2 CV units"):
        validate_cross_validation(dataset, CrossValidationSpec(minimum_units=2))


def test_validate_cross_validation_rejects_missing_condition_when_balanced_required() -> None:
    dataset = _dataset(
        [
            {"pattern_id": "p1", "condition_id": "a", "cv_unit": "run-1", "f1": 1},
            {"pattern_id": "p2", "condition_id": "b", "cv_unit": "run-1", "f1": 2},
            {"pattern_id": "p3", "condition_id": "a", "cv_unit": "run-2", "f1": 3},
        ]
    )

    with pytest.raises(ValueError, match="missing observations"):
        validate_cross_validation(dataset, CrossValidationSpec(require_balanced_units=True))
