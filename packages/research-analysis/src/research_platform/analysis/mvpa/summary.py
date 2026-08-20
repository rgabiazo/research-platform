from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields
import math
from numbers import Real
from typing import Any

from .contracts import DistanceEstimate
from .tables import DISTANCE_VALUE_COLUMN


SUMMARY_COUNT_COLUMN = "n"
SUMMARY_MEAN_DISTANCE_COLUMN = "mean_distance"
SUMMARY_STD_DISTANCE_COLUMN = "std_distance"
SUMMARY_SEM_DISTANCE_COLUMN = "sem_distance"
SUMMARY_MIN_DISTANCE_COLUMN = "min_distance"
SUMMARY_MAX_DISTANCE_COLUMN = "max_distance"
SUMMARY_DISTANCE_COLUMNS = (
    SUMMARY_COUNT_COLUMN,
    SUMMARY_MEAN_DISTANCE_COLUMN,
    SUMMARY_STD_DISTANCE_COLUMN,
    SUMMARY_SEM_DISTANCE_COLUMN,
    SUMMARY_MIN_DISTANCE_COLUMN,
    SUMMARY_MAX_DISTANCE_COLUMN,
)

_RESERVED_ESTIMATE_FIELDS = frozenset(field.name for field in fields(DistanceEstimate))
_STANDARD_ESTIMATE_FIELDS = _RESERVED_ESTIMATE_FIELDS - {"context"}
_RESERVED_CONTEXT_KEYS = _RESERVED_ESTIMATE_FIELDS | frozenset(SUMMARY_DISTANCE_COLUMNS)


def distance_summary_rows_from_estimates(
    estimates: Iterable[DistanceEstimate],
    group_by: Sequence[str] = (),
) -> list[dict[str, Any]]:
    groups = _empty_groups(group_by)

    for row_number, estimate in enumerate(estimates, start=1):
        _validate_estimate_context(estimate, row_number=row_number)
        group_key = _estimate_group_key(estimate, groups.group_by, row_number=row_number)
        groups.add(group_key, _validated_distance(estimate.distance, row_number=row_number))

    return groups.summary_rows()


def distance_summary_rows_from_long_rows(
    rows: Iterable[Mapping[str, Any]],
    group_by: Sequence[str] = (),
) -> list[dict[str, Any]]:
    groups = _empty_groups(group_by)

    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Distance row {row_number} must be a mapping.")
        if DISTANCE_VALUE_COLUMN not in row:
            raise ValueError(f"Distance row {row_number} is missing required field {DISTANCE_VALUE_COLUMN!r}.")

        group_key = _row_group_key(row, groups.group_by, row_number=row_number)
        groups.add(group_key, _validated_distance(row[DISTANCE_VALUE_COLUMN], row_number=row_number))

    return groups.summary_rows()


class _SummaryGroups:
    def __init__(self, group_by: Sequence[str]) -> None:
        self.group_by = _validated_group_by(group_by)
        self._distances_by_group: dict[tuple[Any, ...], list[float]] = {}

    def add(self, group_key: tuple[Any, ...], distance: float) -> None:
        try:
            distances = self._distances_by_group.setdefault(group_key, [])
        except TypeError as exc:
            raise ValueError("Group field values must be hashable.") from exc
        distances.append(distance)

    def summary_rows(self) -> list[dict[str, Any]]:
        if not self._distances_by_group:
            if self.group_by:
                return []
            return [_summary_row((), self.group_by, ())]

        return [
            _summary_row(group_key, self.group_by, distances)
            for group_key, distances in sorted(
                self._distances_by_group.items(),
                key=lambda item: tuple(_sort_value(value) for value in item[0]),
            )
        ]


def _empty_groups(group_by: Sequence[str]) -> _SummaryGroups:
    return _SummaryGroups(group_by)


def _validated_group_by(group_by: Sequence[str]) -> tuple[str, ...]:
    if isinstance(group_by, (str, bytes)):
        raise ValueError("group_by must be a sequence of field names.")
    try:
        raw_fields = tuple(group_by)
    except TypeError as exc:
        raise ValueError("group_by must be a sequence of field names.") from exc

    fields_: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in raw_fields:
        field_name = str(value).strip()
        if not field_name:
            raise ValueError("group_by must not contain empty field names.")
        if field_name in seen and field_name not in duplicates:
            duplicates.append(field_name)
        seen.add(field_name)
        fields_.append(field_name)

    if duplicates:
        raise ValueError(f"group_by field names must be unique: {', '.join(duplicates)}.")
    return tuple(fields_)


def _estimate_group_key(
    estimate: DistanceEstimate,
    group_by: Sequence[str],
    *,
    row_number: int,
) -> tuple[Any, ...]:
    values: list[Any] = []
    for field_name in group_by:
        if field_name in _STANDARD_ESTIMATE_FIELDS:
            values.append(getattr(estimate, field_name))
        elif field_name in estimate.context:
            values.append(estimate.context[field_name])
        else:
            raise ValueError(f"Distance estimate {row_number} is missing requested group field {field_name!r}.")
    return tuple(values)


def _row_group_key(
    row: Mapping[str, Any],
    group_by: Sequence[str],
    *,
    row_number: int,
) -> tuple[Any, ...]:
    values: list[Any] = []
    for field_name in group_by:
        if field_name not in row:
            raise ValueError(f"Distance row {row_number} is missing requested group field {field_name!r}.")
        values.append(row[field_name])
    return tuple(values)


def _validate_estimate_context(estimate: DistanceEstimate, *, row_number: int) -> None:
    for key in estimate.context:
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Distance estimate {row_number} context keys must be non-empty strings.")
        if key in _RESERVED_CONTEXT_KEYS:
            raise ValueError(
                f"Distance estimate {row_number} context key {key!r} collides with a reserved field name."
            )


def _validated_distance(value: object, *, row_number: int) -> float:
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        raise ValueError(f"Distance value at row {row_number} must be numeric and finite.")
    try:
        distance = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Distance value at row {row_number} must be numeric and finite.") from exc
    if not math.isfinite(distance):
        raise ValueError(f"Distance value at row {row_number} must be numeric and finite.")
    return distance


def _summary_row(
    group_key: Sequence[Any],
    group_by: Sequence[str],
    distances: Sequence[float],
) -> dict[str, Any]:
    row = {field_name: value for field_name, value in zip(group_by, group_key)}
    row.update(_distance_summary(distances))
    return row


def _distance_summary(distances: Sequence[float]) -> dict[str, float | int | None]:
    n = len(distances)
    if n == 0:
        return {
            SUMMARY_COUNT_COLUMN: 0,
            SUMMARY_MEAN_DISTANCE_COLUMN: None,
            SUMMARY_STD_DISTANCE_COLUMN: None,
            SUMMARY_SEM_DISTANCE_COLUMN: None,
            SUMMARY_MIN_DISTANCE_COLUMN: None,
            SUMMARY_MAX_DISTANCE_COLUMN: None,
        }

    mean = sum(distances) / n
    if n == 1:
        std = 0.0
    else:
        std = math.sqrt(sum((distance - mean) ** 2 for distance in distances) / (n - 1))
    return {
        SUMMARY_COUNT_COLUMN: n,
        SUMMARY_MEAN_DISTANCE_COLUMN: mean,
        SUMMARY_STD_DISTANCE_COLUMN: std,
        SUMMARY_SEM_DISTANCE_COLUMN: std / math.sqrt(n),
        SUMMARY_MIN_DISTANCE_COLUMN: min(distances),
        SUMMARY_MAX_DISTANCE_COLUMN: max(distances),
    }


def _sort_value(value: Any) -> tuple[int, str, Any]:
    if value is None:
        return (0, "", "")
    if isinstance(value, bool):
        return (1, "bool", value)
    if isinstance(value, Real):
        return (2, "number", float(value))
    if isinstance(value, str):
        return (3, "str", value)
    return (4, type(value).__name__, repr(value))


__all__ = [
    "distance_summary_rows_from_estimates",
    "distance_summary_rows_from_long_rows",
]
