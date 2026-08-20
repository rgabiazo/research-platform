from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import ClassVar

from .contracts import (
    DistanceEstimate,
    DistanceRequest,
    METRIC_CORRELATION,
    METRIC_CROSSNOBIS,
    METRIC_EUCLIDEAN,
    NOISE_NORMALIZATION_DIAGONAL,
    NOISE_NORMALIZATION_IDENTITY,
    PatternDataset,
)
from .cv import condition_order as resolve_condition_order
from .cv import validate_cross_validation


@dataclass(frozen=True)
class _CellMean:
    vector: tuple[float, ...]
    observation_count: int


@dataclass(frozen=True)
class _Pair:
    index_a: int
    index_b: int
    condition_id_a: str
    condition_id_b: str


@dataclass(frozen=True)
class NativeReferenceDistanceEngine:
    """Pure standard-library reference distance engine for MVPA contracts."""

    diagonal_variances: Sequence[float] | None = None

    name: ClassVar[str] = "native_reference"
    supported_metrics: ClassVar[frozenset[str]] = frozenset(
        {
            METRIC_CROSSNOBIS,
            METRIC_EUCLIDEAN,
            METRIC_CORRELATION,
        }
    )

    def compute_distances(
        self,
        dataset: PatternDataset,
        request: DistanceRequest,
    ) -> tuple[DistanceEstimate, ...]:
        if request.metric not in self.supported_metrics:
            supported = ", ".join(sorted(self.supported_metrics))
            raise ValueError(f"Unsupported distance metric {request.metric!r}. Use one of: {supported}.")

        if request.metric == METRIC_CROSSNOBIS:
            return self._compute_crossnobis(dataset, request)
        if request.noise_normalization.method != NOISE_NORMALIZATION_IDENTITY:
            raise ValueError(
                "Diagonal variance normalization is only supported for crossnobis distances in Phase 1B."
            )
        if request.metric == METRIC_EUCLIDEAN:
            return self._compute_euclidean(dataset, request)
        return self._compute_correlation(dataset, request)

    def _compute_crossnobis(
        self,
        dataset: PatternDataset,
        request: DistanceRequest,
    ) -> tuple[DistanceEstimate, ...]:
        resolved_condition_order = validate_cross_validation(
            dataset,
            request.cross_validation,
            condition_order=request.condition_order,
        )
        cells = _cell_means(dataset)
        cv_units = _ordered_cv_units(dataset)
        diagonal_variances = self._resolve_diagonal_variances(
            request.noise_normalization.method,
            feature_count=len(dataset.feature_names),
        )

        estimates: list[DistanceEstimate] = []
        for pair in _condition_pairs(resolved_condition_order):
            pair_units = tuple(
                cv_unit
                for cv_unit in cv_units
                if (pair.condition_id_a, cv_unit) in cells and (pair.condition_id_b, cv_unit) in cells
            )
            if len(pair_units) < 2:
                raise ValueError(
                    "Crossnobis requires at least two CV units with observations for conditions "
                    f"{pair.condition_id_a!r} and {pair.condition_id_b!r}; found {len(pair_units)}."
                )
            if len(pair_units) < request.cross_validation.minimum_units:
                raise ValueError(
                    "Crossnobis requires at least "
                    f"{request.cross_validation.minimum_units} CV units with observations for conditions "
                    f"{pair.condition_id_a!r} and {pair.condition_id_b!r}; found {len(pair_units)}."
                )

            deltas = tuple(
                _subtract(cells[(pair.condition_id_a, cv_unit)].vector, cells[(pair.condition_id_b, cv_unit)].vector)
                for cv_unit in pair_units
            )
            distance = _crossnobis_distance(deltas, diagonal_variances=diagonal_variances)
            estimates.append(
                _distance_estimate(
                    pair=pair,
                    distance=distance,
                    metric=request.metric,
                    normalization_method=request.noise_normalization.method,
                    cv_unit_count=len(pair_units),
                    feature_count=len(dataset.feature_names),
                    observation_count=_pair_observation_count(cells, pair, cv_units=pair_units),
                )
            )
        return tuple(estimates)

    def _compute_euclidean(
        self,
        dataset: PatternDataset,
        request: DistanceRequest,
    ) -> tuple[DistanceEstimate, ...]:
        resolved_condition_order = resolve_condition_order(dataset, explicit_order=request.condition_order)
        _require_at_least_two_conditions(resolved_condition_order, metric=request.metric)
        cells = _cell_means(dataset)
        condition_means = _condition_mean_vectors(cells, resolved_condition_order)

        estimates: list[DistanceEstimate] = []
        for pair in _condition_pairs(resolved_condition_order):
            vector_a = condition_means[pair.condition_id_a]
            vector_b = condition_means[pair.condition_id_b]
            distance = _euclidean_distance(vector_a, vector_b)
            estimates.append(
                _distance_estimate(
                    pair=pair,
                    distance=distance,
                    metric=request.metric,
                    normalization_method=request.noise_normalization.method,
                    cv_unit_count=_pair_cv_unit_count(cells, pair),
                    feature_count=len(dataset.feature_names),
                    observation_count=_pair_observation_count(cells, pair),
                )
            )
        return tuple(estimates)

    def _compute_correlation(
        self,
        dataset: PatternDataset,
        request: DistanceRequest,
    ) -> tuple[DistanceEstimate, ...]:
        resolved_condition_order = resolve_condition_order(dataset, explicit_order=request.condition_order)
        _require_at_least_two_conditions(resolved_condition_order, metric=request.metric)
        cells = _cell_means(dataset)
        condition_means = _condition_mean_vectors(cells, resolved_condition_order)

        estimates: list[DistanceEstimate] = []
        for pair in _condition_pairs(resolved_condition_order):
            vector_a = condition_means[pair.condition_id_a]
            vector_b = condition_means[pair.condition_id_b]
            distance = _correlation_distance(vector_a, vector_b)
            estimates.append(
                _distance_estimate(
                    pair=pair,
                    distance=distance,
                    metric=request.metric,
                    normalization_method=request.noise_normalization.method,
                    cv_unit_count=_pair_cv_unit_count(cells, pair),
                    feature_count=len(dataset.feature_names),
                    observation_count=_pair_observation_count(cells, pair),
                )
            )
        return tuple(estimates)

    def _resolve_diagonal_variances(
        self,
        normalization_method: str,
        *,
        feature_count: int,
    ) -> tuple[float, ...] | None:
        if normalization_method == NOISE_NORMALIZATION_IDENTITY:
            return None
        if normalization_method != NOISE_NORMALIZATION_DIAGONAL:
            raise ValueError(f"Unsupported crossnobis normalization method {normalization_method!r}.")
        if self.diagonal_variances is None:
            raise ValueError("diagonal_variances are required for diagonal crossnobis normalization.")
        return _validated_diagonal_variances(self.diagonal_variances, feature_count=feature_count)


def _cell_means(dataset: PatternDataset) -> dict[tuple[str, str], _CellMean]:
    sums_by_cell: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
    counts_by_cell: dict[tuple[str, str], int] = {}

    for observation in dataset.observations:
        key = (observation.condition_id, observation.cv_unit)
        if key not in sums_by_cell:
            sums_by_cell[key] = [0.0] * len(observation.features)
            counts_by_cell[key] = 0
        for index, value in enumerate(observation.features):
            sums_by_cell[key][index] += value
        counts_by_cell[key] += 1

    return {
        key: _CellMean(
            vector=tuple(value / counts_by_cell[key] for value in feature_sums),
            observation_count=counts_by_cell[key],
        )
        for key, feature_sums in sums_by_cell.items()
    }


def _ordered_cv_units(dataset: PatternDataset) -> tuple[str, ...]:
    units: OrderedDict[str, None] = OrderedDict()
    for observation in dataset.observations:
        units[observation.cv_unit] = None
    return tuple(units)


def _condition_pairs(condition_ids: Sequence[str]) -> tuple[_Pair, ...]:
    return tuple(
        _Pair(
            index_a=index_a,
            index_b=index_b,
            condition_id_a=condition_ids[index_a],
            condition_id_b=condition_ids[index_b],
        )
        for index_a in range(len(condition_ids))
        for index_b in range(index_a + 1, len(condition_ids))
    )


def _condition_mean_vectors(
    cells: dict[tuple[str, str], _CellMean],
    condition_ids: Sequence[str],
) -> dict[str, tuple[float, ...]]:
    return {
        condition_id: _mean_vector(
            tuple(cell.vector for (cell_condition_id, _), cell in cells.items() if cell_condition_id == condition_id)
        )
        for condition_id in condition_ids
    }


def _mean_vector(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("A condition mean requires at least one CV-unit mean vector.")

    feature_count = len(vectors[0])
    sums = [0.0] * feature_count
    for vector in vectors:
        for index, value in enumerate(vector):
            sums[index] += value
    count = len(vectors)
    return tuple(value / count for value in sums)


def _crossnobis_distance(
    deltas: Sequence[Sequence[float]],
    *,
    diagonal_variances: Sequence[float] | None,
) -> float:
    fold_distances: list[float] = []
    for index, delta in enumerate(deltas):
        other_delta_mean = _mean_vector(
            tuple(other for other_index, other in enumerate(deltas) if other_index != index)
        )
        fold_distances.append(_dot(delta, other_delta_mean, diagonal_variances=diagonal_variances))
    return _finite_distance(sum(fold_distances) / len(fold_distances), label="crossnobis distance")


def _dot(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
    *,
    diagonal_variances: Sequence[float] | None,
) -> float:
    if diagonal_variances is None:
        value = sum(value_a * value_b for value_a, value_b in zip(vector_a, vector_b))
    else:
        value = sum(
            value_a * value_b / variance
            for value_a, value_b, variance in zip(vector_a, vector_b, diagonal_variances)
        )
    return _finite_distance(value, label="dot product")


def _euclidean_distance(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    return _finite_distance(
        math.sqrt(sum((value_a - value_b) ** 2 for value_a, value_b in zip(vector_a, vector_b))),
        label="euclidean distance",
    )


def _correlation_distance(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    mean_a = sum(vector_a) / len(vector_a)
    mean_b = sum(vector_b) / len(vector_b)
    centered_a = tuple(value - mean_a for value in vector_a)
    centered_b = tuple(value - mean_b for value in vector_b)
    sum_squares_a = sum(value * value for value in centered_a)
    sum_squares_b = sum(value * value for value in centered_b)
    if sum_squares_a == 0.0 or sum_squares_b == 0.0:
        raise ValueError("Correlation distance is undefined for constant condition mean vectors.")

    denominator = math.sqrt(sum_squares_a * sum_squares_b)
    correlation = sum(value_a * value_b for value_a, value_b in zip(centered_a, centered_b)) / denominator
    correlation = max(-1.0, min(1.0, correlation))
    return _finite_distance(1.0 - correlation, label="correlation distance")


def _subtract(vector_a: Sequence[float], vector_b: Sequence[float]) -> tuple[float, ...]:
    return tuple(value_a - value_b for value_a, value_b in zip(vector_a, vector_b))


def _pair_observation_count(
    cells: dict[tuple[str, str], _CellMean],
    pair: _Pair,
    *,
    cv_units: Sequence[str] | None = None,
) -> int:
    condition_ids = {pair.condition_id_a, pair.condition_id_b}
    allowed_units = set(cv_units) if cv_units is not None else None
    return sum(
        cell.observation_count
        for (condition_id, cv_unit), cell in cells.items()
        if condition_id in condition_ids and (allowed_units is None or cv_unit in allowed_units)
    )


def _pair_cv_unit_count(cells: dict[tuple[str, str], _CellMean], pair: _Pair) -> int:
    condition_ids = {pair.condition_id_a, pair.condition_id_b}
    return len({cv_unit for (condition_id, cv_unit) in cells if condition_id in condition_ids})


def _distance_estimate(
    *,
    pair: _Pair,
    distance: float,
    metric: str,
    normalization_method: str,
    cv_unit_count: int,
    feature_count: int,
    observation_count: int,
) -> DistanceEstimate:
    return DistanceEstimate(
        condition_id_a=pair.condition_id_a,
        condition_id_b=pair.condition_id_b,
        distance=distance,
        metric=metric,
        engine_name=NativeReferenceDistanceEngine.name,
        normalization_method=normalization_method,
        cv_unit_count=cv_unit_count,
        context={
            "condition_index_a": pair.index_a,
            "condition_index_b": pair.index_b,
            "feature_count": feature_count,
            "observation_count": observation_count,
        },
    )


def _validated_diagonal_variances(
    diagonal_variances: Sequence[float],
    *,
    feature_count: int,
) -> tuple[float, ...]:
    if isinstance(diagonal_variances, (str, bytes)):
        raise ValueError("diagonal_variances must contain positive finite numeric values.")
    try:
        raw_variances = tuple(diagonal_variances)
    except TypeError as exc:
        raise ValueError("diagonal_variances must contain positive finite numeric values.") from exc

    if len(raw_variances) != feature_count:
        raise ValueError(
            f"diagonal_variances length must match feature count ({len(raw_variances)} != {feature_count})."
        )

    variances: list[float] = []
    for variance in raw_variances:
        if isinstance(variance, bool):
            raise ValueError("diagonal_variances must contain positive finite numeric values.")
        try:
            numeric = float(variance)
        except (TypeError, ValueError) as exc:
            raise ValueError("diagonal_variances must contain positive finite numeric values.") from exc
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("diagonal_variances must contain positive finite numeric values.")
        variances.append(numeric)
    return tuple(variances)


def _require_at_least_two_conditions(condition_ids: Sequence[str], *, metric: str) -> None:
    if len(condition_ids) < 2:
        raise ValueError(f"{metric} distance requires at least two conditions.")


def _finite_distance(value: float, *, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return value


__all__ = ["NativeReferenceDistanceEngine"]
