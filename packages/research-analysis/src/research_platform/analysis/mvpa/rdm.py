from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .contracts import DistanceEstimate
from .tables import (
    DISTANCE_CONDITION_A_COLUMN,
    DISTANCE_CONDITION_B_COLUMN,
    DISTANCE_CV_UNIT_COUNT_COLUMN,
    DISTANCE_ENGINE_COLUMN,
    DISTANCE_METRIC_COLUMN,
    DISTANCE_NORMALIZATION_COLUMN,
    DISTANCE_VALUE_COLUMN,
    RDM_INDEX_COLUMN,
)


def rdm_long_rows_from_estimates(
    estimates: Iterable[DistanceEstimate],
    condition_order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    estimate_list = list(estimates)
    resolved_order = _resolve_condition_order(estimate_list, condition_order=condition_order)
    order_index = {condition_id: index for index, condition_id in enumerate(resolved_order)}

    return [
        _estimate_to_row(estimate, order_index=order_index)
        for estimate in sorted(estimate_list, key=lambda item: _pair_sort_key(item, order_index))
    ]


def rdm_wide_rows_from_estimates(
    estimates: Iterable[DistanceEstimate],
    condition_order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    estimate_list = list(estimates)
    resolved_order = _resolve_condition_order(estimate_list, condition_order=condition_order)
    _validate_wide_condition_ids(resolved_order)

    matrix: dict[str, dict[str, float | None]] = {
        row_condition: {column_condition: None for column_condition in resolved_order}
        for row_condition in resolved_order
    }
    for condition_id in resolved_order:
        matrix[condition_id][condition_id] = 0.0

    seen_pairs: set[frozenset[str]] = set()
    allowed_conditions = set(resolved_order)
    for estimate in estimate_list:
        if estimate.condition_id_a not in allowed_conditions or estimate.condition_id_b not in allowed_conditions:
            raise ValueError("Distance estimates contain conditions not present in condition_order.")
        pair = frozenset((estimate.condition_id_a, estimate.condition_id_b))
        if len(pair) == 1:
            continue
        if pair in seen_pairs:
            raise ValueError(
                "Wide RDM output requires at most one distance estimate per unordered condition pair."
            )
        seen_pairs.add(pair)
        matrix[estimate.condition_id_a][estimate.condition_id_b] = estimate.distance
        matrix[estimate.condition_id_b][estimate.condition_id_a] = estimate.distance

    rows: list[dict[str, Any]] = []
    for condition_id in resolved_order:
        row: dict[str, Any] = {RDM_INDEX_COLUMN: condition_id}
        for column_condition in resolved_order:
            row[column_condition] = matrix[condition_id][column_condition]
        rows.append(row)
    return rows


def _estimate_to_row(estimate: DistanceEstimate, *, order_index: dict[str, int]) -> dict[str, Any]:
    condition_id_a = estimate.condition_id_a
    condition_id_b = estimate.condition_id_b
    if order_index[condition_id_a] > order_index[condition_id_b]:
        condition_id_a, condition_id_b = condition_id_b, condition_id_a
    return {
        DISTANCE_CONDITION_A_COLUMN: condition_id_a,
        DISTANCE_CONDITION_B_COLUMN: condition_id_b,
        DISTANCE_VALUE_COLUMN: estimate.distance,
        DISTANCE_METRIC_COLUMN: estimate.metric,
        DISTANCE_ENGINE_COLUMN: estimate.engine_name,
        DISTANCE_NORMALIZATION_COLUMN: estimate.normalization_method,
        DISTANCE_CV_UNIT_COUNT_COLUMN: estimate.cv_unit_count,
    }


def _pair_sort_key(estimate: DistanceEstimate, order_index: dict[str, int]) -> tuple[int, int, int, int]:
    index_a = order_index[estimate.condition_id_a]
    index_b = order_index[estimate.condition_id_b]
    return (min(index_a, index_b), max(index_a, index_b), index_a, index_b)


def _resolve_condition_order(
    estimates: Sequence[DistanceEstimate],
    *,
    condition_order: Sequence[str] | None,
) -> tuple[str, ...]:
    observed_order: list[str] = []
    seen: set[str] = set()
    for estimate in estimates:
        for condition_id in (estimate.condition_id_a, estimate.condition_id_b):
            if condition_id not in seen:
                observed_order.append(condition_id)
                seen.add(condition_id)

    if condition_order is None:
        if not observed_order:
            raise ValueError("At least one distance estimate or explicit condition_order is required.")
        return tuple(observed_order)

    requested_order = _validated_condition_ids(condition_order)
    requested = set(requested_order)
    missing = [condition_id for condition_id in observed_order if condition_id not in requested]
    if missing:
        raise ValueError(f"condition_order is missing estimated conditions: {', '.join(missing)}.")
    return requested_order


def _validate_wide_condition_ids(condition_ids: Sequence[str]) -> None:
    _validated_condition_ids(condition_ids)
    collisions = [condition_id for condition_id in condition_ids if condition_id == RDM_INDEX_COLUMN]
    if collisions:
        raise ValueError(f"Condition ids must not collide with the wide RDM index column {RDM_INDEX_COLUMN!r}.")


def _validated_condition_ids(condition_ids: Sequence[str]) -> tuple[str, ...]:
    resolved: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in condition_ids:
        condition_id = str(value).strip()
        if not condition_id:
            raise ValueError("condition_order must not contain empty condition ids.")
        if condition_id in seen and condition_id not in duplicates:
            duplicates.append(condition_id)
        seen.add(condition_id)
        resolved.append(condition_id)
    if not resolved:
        raise ValueError("condition_order must contain at least one condition id.")
    if duplicates:
        raise ValueError(f"condition_order must be unique: {', '.join(duplicates)}.")
    return tuple(resolved)


__all__ = [
    "rdm_long_rows_from_estimates",
    "rdm_wide_rows_from_estimates",
]
