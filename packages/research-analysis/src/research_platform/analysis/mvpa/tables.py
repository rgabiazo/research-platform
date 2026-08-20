from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any, Sequence

from .contracts import PatternDataset, PatternObservation


PATTERN_ID_COLUMN = "pattern_id"
CONDITION_ID_COLUMN = "condition_id"
CV_UNIT_COLUMN = "cv_unit"
REQUIRED_PATTERN_COLUMNS = (PATTERN_ID_COLUMN, CONDITION_ID_COLUMN, CV_UNIT_COLUMN)

RDM_INDEX_COLUMN = CONDITION_ID_COLUMN
DISTANCE_CONDITION_A_COLUMN = "condition_id_a"
DISTANCE_CONDITION_B_COLUMN = "condition_id_b"
DISTANCE_VALUE_COLUMN = "distance"
DISTANCE_METRIC_COLUMN = "metric"
DISTANCE_ENGINE_COLUMN = "engine_name"
DISTANCE_NORMALIZATION_COLUMN = "normalization_method"
DISTANCE_CV_UNIT_COUNT_COLUMN = "cv_unit_count"
LONG_DISTANCE_COLUMNS = (
    DISTANCE_CONDITION_A_COLUMN,
    DISTANCE_CONDITION_B_COLUMN,
    DISTANCE_VALUE_COLUMN,
    DISTANCE_METRIC_COLUMN,
    DISTANCE_ENGINE_COLUMN,
    DISTANCE_NORMALIZATION_COLUMN,
    DISTANCE_CV_UNIT_COUNT_COLUMN,
)


def pattern_dataset_from_rows(
    rows: Iterable[Mapping[str, Any]],
    feature_columns: Sequence[str],
    context_columns: Sequence[str] | None = None,
) -> PatternDataset:
    features = _validated_columns(feature_columns, label="feature_columns")
    context = _validated_columns(context_columns or (), label="context_columns")

    observations: list[PatternObservation] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Pattern row {row_number} must be a mapping.")
        _require_columns(row, (*REQUIRED_PATTERN_COLUMNS, *features, *context), row_number=row_number)
        observations.append(
            PatternObservation(
                pattern_id=_non_empty_cell(row[PATTERN_ID_COLUMN], column=PATTERN_ID_COLUMN, row_number=row_number),
                condition_id=_non_empty_cell(
                    row[CONDITION_ID_COLUMN], column=CONDITION_ID_COLUMN, row_number=row_number
                ),
                cv_unit=_non_empty_cell(row[CV_UNIT_COLUMN], column=CV_UNIT_COLUMN, row_number=row_number),
                features=tuple(
                    _finite_feature_value(row[column], column=column, row_number=row_number) for column in features
                ),
                context={column: row[column] for column in context},
            )
        )

    if not observations:
        raise ValueError("At least one pattern row is required.")
    return PatternDataset(observations=observations, feature_names=features)


def _validated_columns(columns: Sequence[str], *, label: str) -> tuple[str, ...]:
    validated = tuple(str(column).strip() for column in columns)
    if label == "feature_columns" and not validated:
        raise ValueError("feature_columns must contain at least one column.")
    empty = [column for column in validated if not column]
    if empty:
        raise ValueError(f"{label} must not contain empty column names.")
    seen: set[str] = set()
    duplicates: list[str] = []
    for column in validated:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    if duplicates:
        raise ValueError(f"{label} must be unique: {', '.join(duplicates)}.")
    return validated


def _require_columns(row: Mapping[str, Any], columns: Sequence[str], *, row_number: int) -> None:
    missing = [column for column in columns if column not in row]
    if missing:
        missing_columns = ", ".join(missing)
        raise ValueError(f"Pattern row {row_number} is missing required columns: {missing_columns}.")


def _non_empty_cell(value: object, *, column: str, row_number: int) -> str:
    label = str(value).strip()
    if not label:
        raise ValueError(f"Column {column!r} must be non-empty at row {row_number}.")
    return label


def _finite_feature_value(value: object, *, column: str, row_number: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Feature column {column!r} must be numeric and finite at row {row_number}.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature column {column!r} must be numeric and finite at row {row_number}.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"Feature column {column!r} must be numeric and finite at row {row_number}.")
    return numeric


__all__ = [
    "CONDITION_ID_COLUMN",
    "CV_UNIT_COLUMN",
    "DISTANCE_CONDITION_A_COLUMN",
    "DISTANCE_CONDITION_B_COLUMN",
    "DISTANCE_CV_UNIT_COUNT_COLUMN",
    "DISTANCE_ENGINE_COLUMN",
    "DISTANCE_METRIC_COLUMN",
    "DISTANCE_NORMALIZATION_COLUMN",
    "DISTANCE_VALUE_COLUMN",
    "LONG_DISTANCE_COLUMNS",
    "PATTERN_ID_COLUMN",
    "RDM_INDEX_COLUMN",
    "REQUIRED_PATTERN_COLUMNS",
    "pattern_dataset_from_rows",
]
