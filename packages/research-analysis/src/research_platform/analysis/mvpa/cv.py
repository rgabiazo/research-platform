from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence

from .contracts import CrossValidationSpec, PatternDataset


def condition_order(dataset: PatternDataset, explicit_order: Sequence[str] | None = None) -> tuple[str, ...]:
    return _condition_order(dataset, explicit_order=explicit_order)


def _condition_order(dataset: PatternDataset, explicit_order: Sequence[str] | None = None) -> tuple[str, ...]:
    observed_order = _first_seen(observation.condition_id for observation in dataset.observations)
    if explicit_order is None:
        return observed_order

    requested_order = _validated_order(explicit_order, label="condition_order")
    observed = set(observed_order)
    requested = set(requested_order)
    missing = [condition_id for condition_id in observed_order if condition_id not in requested]
    unknown = [condition_id for condition_id in requested_order if condition_id not in observed]
    if missing:
        raise ValueError(f"condition_order is missing observed conditions: {', '.join(missing)}.")
    if unknown:
        raise ValueError(f"condition_order contains unknown conditions: {', '.join(unknown)}.")
    return requested_order


def cv_units_by_condition(dataset: PatternDataset) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, OrderedDict[str, None]] = {}
    for observation in dataset.observations:
        grouped.setdefault(observation.condition_id, OrderedDict())[observation.cv_unit] = None
    return {condition_id: tuple(units) for condition_id, units in grouped.items()}


def validate_cross_validation(
    dataset: PatternDataset,
    spec: CrossValidationSpec,
    condition_order: Sequence[str] | None = None,
) -> tuple[str, ...]:
    resolved_condition_order = _condition_order(dataset, explicit_order=condition_order)
    if len(resolved_condition_order) < 2:
        raise ValueError("Cross-validation requires at least two conditions.")

    units = _first_seen(observation.cv_unit for observation in dataset.observations)
    if len(units) < spec.minimum_units:
        raise ValueError(
            f"Cross-validation requires at least {spec.minimum_units} CV units; found {len(units)}."
        )

    if spec.require_balanced_units:
        conditions_by_unit: dict[str, set[str]] = {unit: set() for unit in units}
        for observation in dataset.observations:
            conditions_by_unit[observation.cv_unit].add(observation.condition_id)

        for unit in units:
            missing = [
                condition_id for condition_id in resolved_condition_order if condition_id not in conditions_by_unit[unit]
            ]
            if missing:
                raise ValueError(
                    f"CV unit {unit!r} is missing observations for conditions: {', '.join(missing)}."
                )

    return resolved_condition_order


def _first_seen(values: Iterable[object]) -> tuple[str, ...]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for value in values:
        ordered[str(value)] = None
    return tuple(ordered)


def _validated_order(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        condition_id = str(value).strip()
        if not condition_id:
            raise ValueError(f"{label} must not contain empty condition ids.")
        if condition_id in seen and condition_id not in duplicates:
            duplicates.append(condition_id)
        seen.add(condition_id)
        ordered.append(condition_id)
    if not ordered:
        raise ValueError(f"{label} must contain at least one condition id.")
    if duplicates:
        raise ValueError(f"{label} must be unique: {', '.join(duplicates)}.")
    return tuple(ordered)


__all__ = [
    "condition_order",
    "cv_units_by_condition",
    "validate_cross_validation",
]
