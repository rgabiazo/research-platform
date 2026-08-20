from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


METRIC_CROSSNOBIS = "crossnobis"
METRIC_EUCLIDEAN = "euclidean"
METRIC_CORRELATION = "correlation"

NOISE_NORMALIZATION_IDENTITY = "identity"
NOISE_NORMALIZATION_DIAGONAL = "diagonal"
SUPPORTED_NOISE_NORMALIZATION_METHODS = frozenset(
    {
        NOISE_NORMALIZATION_IDENTITY,
        NOISE_NORMALIZATION_DIAGONAL,
    }
)


@dataclass(frozen=True)
class PatternObservation:
    pattern_id: str
    condition_id: str
    cv_unit: str
    features: Sequence[float]
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern_id", _non_empty_label(self.pattern_id, field_name="pattern_id"))
        object.__setattr__(self, "condition_id", _non_empty_label(self.condition_id, field_name="condition_id"))
        object.__setattr__(self, "cv_unit", _non_empty_label(self.cv_unit, field_name="cv_unit"))
        object.__setattr__(self, "features", _finite_float_tuple(self.features, field_name="features"))
        object.__setattr__(self, "context", dict(self.context))


@dataclass(frozen=True)
class PatternDataset:
    observations: Sequence[PatternObservation]
    feature_names: Sequence[str]

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        feature_names = tuple(_non_empty_label(name, field_name="feature_names") for name in self.feature_names)
        if not observations:
            raise ValueError("PatternDataset requires at least one observation.")
        if not feature_names:
            raise ValueError("PatternDataset requires at least one feature name.")
        duplicate_names = _duplicates(feature_names)
        if duplicate_names:
            raise ValueError(f"Feature names must be unique: {', '.join(duplicate_names)}.")

        expected_width = len(feature_names)
        for observation in observations:
            if len(observation.features) != expected_width:
                raise ValueError(
                    "Observation feature width must match feature_names "
                    f"({len(observation.features)} != {expected_width}) for pattern_id {observation.pattern_id!r}."
                )

        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "feature_names", feature_names)


@dataclass(frozen=True)
class CrossValidationSpec:
    minimum_units: int = 2
    require_balanced_units: bool = True

    def __post_init__(self) -> None:
        if self.minimum_units < 1:
            raise ValueError("minimum_units must be at least 1.")


@dataclass(frozen=True)
class NoiseNormalization:
    method: str = NOISE_NORMALIZATION_IDENTITY

    def __post_init__(self) -> None:
        method = _non_empty_label(self.method, field_name="method")
        if method not in SUPPORTED_NOISE_NORMALIZATION_METHODS:
            supported = ", ".join(sorted(SUPPORTED_NOISE_NORMALIZATION_METHODS))
            raise ValueError(f"Unsupported noise normalization method {method!r}. Use one of: {supported}.")
        object.__setattr__(self, "method", method)


@dataclass(frozen=True)
class DistanceRequest:
    metric: str
    condition_order: Sequence[str] | None = None
    cross_validation: CrossValidationSpec = field(default_factory=CrossValidationSpec)
    noise_normalization: NoiseNormalization = field(default_factory=NoiseNormalization)
    engine_name: str = "adapter"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", _non_empty_label(self.metric, field_name="metric"))
        object.__setattr__(self, "engine_name", _non_empty_label(self.engine_name, field_name="engine_name"))
        if self.condition_order is not None:
            condition_order = tuple(
                _non_empty_label(condition_id, field_name="condition_order") for condition_id in self.condition_order
            )
            duplicate_conditions = _duplicates(condition_order)
            if duplicate_conditions:
                raise ValueError(f"condition_order must be unique: {', '.join(duplicate_conditions)}.")
            object.__setattr__(self, "condition_order", condition_order)


@dataclass(frozen=True)
class DistanceEstimate:
    condition_id_a: str
    condition_id_b: str
    distance: float
    metric: str
    engine_name: str = "adapter"
    normalization_method: str = NOISE_NORMALIZATION_IDENTITY
    cv_unit_count: int | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "condition_id_a", _non_empty_label(self.condition_id_a, field_name="condition_id_a")
        )
        object.__setattr__(
            self, "condition_id_b", _non_empty_label(self.condition_id_b, field_name="condition_id_b")
        )
        object.__setattr__(self, "distance", _finite_float(self.distance, field_name="distance"))
        object.__setattr__(self, "metric", _non_empty_label(self.metric, field_name="metric"))
        object.__setattr__(self, "engine_name", _non_empty_label(self.engine_name, field_name="engine_name"))
        object.__setattr__(
            self,
            "normalization_method",
            _non_empty_label(self.normalization_method, field_name="normalization_method"),
        )
        if self.cv_unit_count is not None and self.cv_unit_count < 0:
            raise ValueError("cv_unit_count must be non-negative when provided.")
        object.__setattr__(self, "context", dict(self.context))


def _non_empty_label(value: object, *, field_name: str) -> str:
    label = str(value).strip()
    if not label:
        raise ValueError(f"{field_name} must be a non-empty value.")
    return label


def _finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite numeric value.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite numeric value.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be a finite numeric value.")
    return numeric


def _finite_float_tuple(values: Sequence[float], *, field_name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must contain finite numeric values.")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field_name} must contain finite numeric values.") from exc
    features = tuple(_finite_float(value, field_name=field_name) for value in raw_values)
    if not features:
        raise ValueError(f"{field_name} must contain at least one value.")
    return features


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


__all__ = [
    "CrossValidationSpec",
    "DistanceEstimate",
    "DistanceRequest",
    "METRIC_CORRELATION",
    "METRIC_CROSSNOBIS",
    "METRIC_EUCLIDEAN",
    "NOISE_NORMALIZATION_DIAGONAL",
    "NOISE_NORMALIZATION_IDENTITY",
    "NoiseNormalization",
    "PatternDataset",
    "PatternObservation",
    "SUPPORTED_NOISE_NORMALIZATION_METHODS",
]
