from __future__ import annotations

import pytest

from research_platform.analysis.mvpa import (
    CONDITION_ID_COLUMN,
    CV_UNIT_COLUMN,
    PATTERN_ID_COLUMN,
    NoiseNormalization,
    pattern_dataset_from_rows,
)


def test_pattern_dataset_from_rows_builds_dataset_and_requested_context() -> None:
    rows = [
        {
            PATTERN_ID_COLUMN: "p1",
            CONDITION_ID_COLUMN: "face",
            CV_UNIT_COLUMN: "run-1",
            "f1": "1.5",
            "f2": 2,
            "subject_id": "sub-01",
            "ignored": "not-carried",
        },
        {
            PATTERN_ID_COLUMN: "p2",
            CONDITION_ID_COLUMN: "house",
            CV_UNIT_COLUMN: "run-1",
            "f1": "3.25",
            "f2": 4,
            "subject_id": "sub-01",
            "ignored": "not-carried",
        },
    ]

    dataset = pattern_dataset_from_rows(rows, feature_columns=["f1", "f2"], context_columns=["subject_id"])

    assert dataset.feature_names == ("f1", "f2")
    assert len(dataset.observations) == 2
    assert dataset.observations[0].features == (1.5, 2.0)
    assert dataset.observations[0].context == {"subject_id": "sub-01"}
    assert dataset.observations[1].condition_id == "house"


def test_pattern_dataset_from_rows_omits_context_when_not_requested() -> None:
    dataset = pattern_dataset_from_rows(
        [
            {
                PATTERN_ID_COLUMN: "p1",
                CONDITION_ID_COLUMN: "face",
                CV_UNIT_COLUMN: "run-1",
                "f1": 1,
                "subject_id": "sub-01",
            }
        ],
        feature_columns=["f1"],
    )

    assert dataset.observations[0].context == {}


def test_missing_required_columns_fail() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        pattern_dataset_from_rows(
            [
                {
                    PATTERN_ID_COLUMN: "p1",
                    CONDITION_ID_COLUMN: "face",
                    "f1": 1,
                }
            ],
            feature_columns=["f1"],
        )


def test_missing_feature_columns_fail() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        pattern_dataset_from_rows(
            [
                {
                    PATTERN_ID_COLUMN: "p1",
                    CONDITION_ID_COLUMN: "face",
                    CV_UNIT_COLUMN: "run-1",
                    "f1": 1,
                }
            ],
            feature_columns=["f1", "f2"],
        )


@pytest.mark.parametrize("bad_value", ["not-numeric", "nan", "inf", "-inf", float("nan"), float("inf")])
def test_nonnumeric_nan_inf_and_nonfinite_feature_values_fail(bad_value: object) -> None:
    with pytest.raises(ValueError, match="numeric and finite"):
        pattern_dataset_from_rows(
            [
                {
                    PATTERN_ID_COLUMN: "p1",
                    CONDITION_ID_COLUMN: "face",
                    CV_UNIT_COLUMN: "run-1",
                    "f1": bad_value,
                }
            ],
            feature_columns=["f1"],
        )


@pytest.mark.parametrize("column", [PATTERN_ID_COLUMN, CONDITION_ID_COLUMN, CV_UNIT_COLUMN])
def test_empty_pattern_condition_or_cv_unit_ids_fail(column: str) -> None:
    row = {
        PATTERN_ID_COLUMN: "p1",
        CONDITION_ID_COLUMN: "face",
        CV_UNIT_COLUMN: "run-1",
        "f1": 1,
    }
    row[column] = " "

    with pytest.raises(ValueError, match="non-empty"):
        pattern_dataset_from_rows([row], feature_columns=["f1"])


def test_noise_normalization_phase_1a_method_labels() -> None:
    assert NoiseNormalization(method="identity").method == "identity"
    assert NoiseNormalization(method="diagonal").method == "diagonal"
    with pytest.raises(ValueError, match="Unsupported noise normalization"):
        NoiseNormalization(method="full")
