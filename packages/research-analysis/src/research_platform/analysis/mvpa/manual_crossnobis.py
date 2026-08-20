from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from itertools import combinations
import math
from pathlib import Path
from typing import Any


ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1 = "manual_diagonal_crossnobis_v1"
NOISE_NONPOSITIVE_POLICY_STRICT = "strict"
NOISE_NONPOSITIVE_POLICY_DROP_PATTERN = "drop_pattern"
NOISE_NONPOSITIVE_POLICY_DROP_FEATURES = "drop_features"
NOISE_NONPOSITIVE_POLICY_FILTER_NONPOSITIVE_FEATURES = "filter_nonpositive_features"
SUPPORTED_NOISE_NONPOSITIVE_POLICIES = frozenset(
    {
        NOISE_NONPOSITIVE_POLICY_STRICT,
        "fail",
        NOISE_NONPOSITIVE_POLICY_DROP_PATTERN,
        NOISE_NONPOSITIVE_POLICY_DROP_FEATURES,
        NOISE_NONPOSITIVE_POLICY_FILTER_NONPOSITIVE_FEATURES,
    }
)
DEFAULT_MANUAL_MIN_RETAINED_FEATURES = 5
DEFAULT_MANUAL_WARN_DROPPED_FEATURE_FRACTION = 0.10


@dataclass(frozen=True)
class ManualCrossnobisRun:
    """One run-level input for the manual-compatible diagonal crossnobis estimator."""

    run_id: str
    condition_a: Sequence[float]
    condition_b: Sequence[float]
    sigma_squared: Sequence[float]
    event_count_a: int | None = None
    event_count_b: int | None = None
    excluded: bool = False
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _non_empty_label(self.run_id, field_name="run_id"))
        object.__setattr__(self, "condition_a", _float_tuple(self.condition_a, field_name="condition_a"))
        object.__setattr__(self, "condition_b", _float_tuple(self.condition_b, field_name="condition_b"))
        object.__setattr__(self, "sigma_squared", _float_tuple(self.sigma_squared, field_name="sigma_squared"))
        _validate_optional_count(self.event_count_a, field_name="event_count_a")
        _validate_optional_count(self.event_count_b, field_name="event_count_b")


@dataclass(frozen=True)
class ManualCrossnobisRunQc:
    run_id: str
    status: str
    reason: str | None = None
    event_count_a: int | None = None
    event_count_b: int | None = None
    excluded: bool = False
    original_feature_count: int | None = None
    retained_feature_count: int | None = None
    dropped_noise_feature_count: int = 0
    dropped_nonfinite_feature_count: int = 0
    dropped_feature_count: int = 0
    dropped_feature_fraction: float = 0.0
    noise_nonpositive_policy: str = NOISE_NONPOSITIVE_POLICY_STRICT

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ManualCrossnobisRunPairDetail:
    run_id_i: str
    run_id_j: str
    dot_product: float
    n_voxels_used: int
    distance: float

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ManualCrossnobisResult:
    status: str
    crossnobis: float | None
    n_valid_runs: int
    valid_runs: Sequence[str]
    excluded_runs: Sequence[str]
    invalid_runs: Sequence[str]
    n_voxels_raw: int
    n_voxels_used: int
    min_events: int | None
    min_valid_voxels: int
    sigma_pooling_source_runs: Sequence[str]
    noise_nonpositive_policy: str = NOISE_NONPOSITIVE_POLICY_STRICT
    retained_feature_count: int | None = None
    dropped_noise_feature_count: int = 0
    dropped_nonfinite_feature_count: int = 0
    dropped_feature_count: int = 0
    dropped_feature_fraction: float = 0.0
    warn_dropped_feature_fraction: float | None = None
    run_qc: Sequence[ManualCrossnobisRunQc] = ()
    run_pair_details: Sequence[ManualCrossnobisRunPairDetail] = ()
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid_runs", tuple(self.valid_runs))
        object.__setattr__(self, "excluded_runs", tuple(self.excluded_runs))
        object.__setattr__(self, "invalid_runs", tuple(self.invalid_runs))
        object.__setattr__(self, "sigma_pooling_source_runs", tuple(self.sigma_pooling_source_runs))
        object.__setattr__(self, "run_qc", tuple(self.run_qc))
        object.__setattr__(self, "run_pair_details", tuple(self.run_pair_details))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class _ManualFeatureFilter:
    indexes: tuple[int, ...]
    dropped_noise_feature_count: int
    dropped_nonfinite_feature_count: int
    dropped_feature_count: int
    dropped_feature_fraction: float


def compute_manual_diagonal_crossnobis_v1(
    runs: Sequence[ManualCrossnobisRun],
    *,
    min_events: int | None = 1,
    min_valid_voxels: int = 1,
    noise_nonpositive_policy: str = NOISE_NONPOSITIVE_POLICY_STRICT,
    warn_dropped_feature_fraction: float = DEFAULT_MANUAL_WARN_DROPPED_FEATURE_FRACTION,
) -> ManualCrossnobisResult:
    """Compute the audited manual diagonal crossnobis convention.

    The estimator uses only non-excluded runs whose two target-condition event
    counts meet ``min_events``. It pools ``sigmasquareds`` across those valid
    runs, normalizes each run's PE difference by the square root of the pooled
    variance, mean-centers after normalization, and averages unordered run-pair
    dot products divided by the number of retained voxels.
    """

    run_rows = tuple(runs)
    if min_events is not None:
        _validate_optional_count(min_events, field_name="min_events")
    if min_valid_voxels < 1:
        raise ValueError("min_valid_voxels must be at least 1.")
    resolved_noise_policy = _normalize_noise_nonpositive_policy(noise_nonpositive_policy)
    warn_fraction = _validated_warn_dropped_feature_fraction(warn_dropped_feature_fraction)

    n_voxels_raw = _raw_feature_count(run_rows)
    length_errors = _feature_length_errors(run_rows, expected=n_voxels_raw)
    if length_errors:
        return _result(
            status="feature_width_mismatch",
            crossnobis=None,
            runs=run_rows,
            valid_runs=(),
            n_voxels_raw=n_voxels_raw,
            n_voxels_used=0,
            min_events=min_events,
            min_valid_voxels=min_valid_voxels,
            noise_nonpositive_policy=resolved_noise_policy,
            warn_dropped_feature_fraction=warn_fraction,
            run_qc=(),
            errors=length_errors,
        )

    run_qc: list[ManualCrossnobisRunQc] = []
    valid_runs: list[ManualCrossnobisRun] = []
    for run in run_rows:
        if run.excluded:
            run_qc.append(
                ManualCrossnobisRunQc(
                    run_id=run.run_id,
                    status="excluded",
                    reason=run.exclusion_reason or "configured_run_exclusion",
                    event_count_a=run.event_count_a,
                    event_count_b=run.event_count_b,
                    excluded=True,
                    noise_nonpositive_policy=resolved_noise_policy,
                )
            )
            continue
        event_reason = _event_threshold_failure_reason(run, min_events=min_events)
        if event_reason is not None:
            run_qc.append(
                ManualCrossnobisRunQc(
                    run_id=run.run_id,
                    status="invalid_events",
                    reason=event_reason,
                    event_count_a=run.event_count_a,
                    event_count_b=run.event_count_b,
                    noise_nonpositive_policy=resolved_noise_policy,
                )
            )
            continue
        valid_runs.append(run)
        run_qc.append(
            ManualCrossnobisRunQc(
                run_id=run.run_id,
                status="valid",
                event_count_a=run.event_count_a,
                event_count_b=run.event_count_b,
                noise_nonpositive_policy=resolved_noise_policy,
            )
        )

    if len(valid_runs) < 2:
        return _result(
            status="insufficient_valid_runs",
            crossnobis=None,
            runs=run_rows,
            valid_runs=valid_runs,
            n_voxels_raw=n_voxels_raw,
            n_voxels_used=0,
            min_events=min_events,
            min_valid_voxels=min_valid_voxels,
            noise_nonpositive_policy=resolved_noise_policy,
            warn_dropped_feature_fraction=warn_fraction,
            run_qc=run_qc,
            errors=("Manual diagonal crossnobis requires at least two valid runs.",),
        )

    sigma_pool = _mean_sigma_squared(valid_runs, feature_count=n_voxels_raw)
    feature_filter = _manual_feature_filter(
        valid_runs,
        sigma_pool=sigma_pool,
        feature_count=n_voxels_raw,
        noise_nonpositive_policy=resolved_noise_policy,
    )
    run_qc = _run_qc_with_feature_filter(
        run_qc,
        runs=valid_runs,
        feature_filter=feature_filter,
        noise_nonpositive_policy=resolved_noise_policy,
    )
    warnings = _feature_filter_warnings(
        feature_filter,
        noise_nonpositive_policy=resolved_noise_policy,
        warn_dropped_feature_fraction=warn_fraction,
    )
    if len(feature_filter.indexes) < min_valid_voxels:
        return _result(
            status="insufficient_valid_voxels",
            crossnobis=None,
            runs=run_rows,
            valid_runs=valid_runs,
            n_voxels_raw=n_voxels_raw,
            n_voxels_used=len(feature_filter.indexes),
            min_events=min_events,
            min_valid_voxels=min_valid_voxels,
            noise_nonpositive_policy=resolved_noise_policy,
            retained_feature_count=len(feature_filter.indexes),
            dropped_noise_feature_count=feature_filter.dropped_noise_feature_count,
            dropped_nonfinite_feature_count=feature_filter.dropped_nonfinite_feature_count,
            dropped_feature_count=feature_filter.dropped_feature_count,
            dropped_feature_fraction=feature_filter.dropped_feature_fraction,
            warn_dropped_feature_fraction=warn_fraction,
            run_qc=run_qc,
            warnings=warnings,
            errors=(
                f"Manual diagonal crossnobis requires at least {min_valid_voxels} valid voxels; "
                f"found {len(feature_filter.indexes)}.",
            ),
        )

    centered_by_run = {
        run.run_id: _centered_normalized_difference(run, sigma_pool=sigma_pool, indexes=feature_filter.indexes)
        for run in valid_runs
    }
    pair_details: list[ManualCrossnobisRunPairDetail] = []
    for left, right in combinations(valid_runs, 2):
        dot_product = _dot(centered_by_run[left.run_id], centered_by_run[right.run_id])
        distance = dot_product / len(feature_filter.indexes)
        pair_details.append(
            ManualCrossnobisRunPairDetail(
                run_id_i=left.run_id,
                run_id_j=right.run_id,
                dot_product=dot_product,
                n_voxels_used=len(feature_filter.indexes),
                distance=distance,
            )
        )

    crossnobis = sum(pair.distance for pair in pair_details) / len(pair_details)
    return _result(
        status="ok",
        crossnobis=crossnobis,
        runs=run_rows,
        valid_runs=valid_runs,
        n_voxels_raw=n_voxels_raw,
        n_voxels_used=len(feature_filter.indexes),
        min_events=min_events,
        min_valid_voxels=min_valid_voxels,
        noise_nonpositive_policy=resolved_noise_policy,
        retained_feature_count=len(feature_filter.indexes),
        dropped_noise_feature_count=feature_filter.dropped_noise_feature_count,
        dropped_nonfinite_feature_count=feature_filter.dropped_nonfinite_feature_count,
        dropped_feature_count=feature_filter.dropped_feature_count,
        dropped_feature_fraction=feature_filter.dropped_feature_fraction,
        warn_dropped_feature_fraction=warn_fraction,
        run_qc=run_qc,
        run_pair_details=pair_details,
        warnings=warnings,
    )


def _event_threshold_failure_reason(run: ManualCrossnobisRun, *, min_events: int | None) -> str | None:
    if min_events is None:
        return None
    if run.event_count_a is None or run.event_count_b is None:
        return "event_count_metadata_missing"
    if run.event_count_a < min_events or run.event_count_b < min_events:
        return "observed_events_below_required_minimum"
    return None


def _mean_sigma_squared(runs: Sequence[ManualCrossnobisRun], *, feature_count: int) -> tuple[float, ...]:
    sums = [0.0] * feature_count
    for run in runs:
        for index, value in enumerate(run.sigma_squared):
            sums[index] += value
    return tuple(value / len(runs) for value in sums)


def _centered_normalized_difference(
    run: ManualCrossnobisRun,
    *,
    sigma_pool: Sequence[float],
    indexes: Sequence[int],
) -> tuple[float, ...]:
    normalized = tuple(
        (run.condition_a[index] - run.condition_b[index]) / math.sqrt(sigma_pool[index])
        for index in indexes
    )
    mean_value = sum(normalized) / len(normalized)
    return tuple(value - mean_value for value in normalized)


def _finite_difference(run: ManualCrossnobisRun, index: int) -> bool:
    return math.isfinite(run.condition_a[index]) and math.isfinite(run.condition_b[index])


def _valid_pooled_sigma(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _valid_single_sigma(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _manual_feature_filter(
    runs: Sequence[ManualCrossnobisRun],
    *,
    sigma_pool: Sequence[float],
    feature_count: int,
    noise_nonpositive_policy: str,
) -> _ManualFeatureFilter:
    valid_indexes: list[int] = []
    dropped_noise = 0
    dropped_nonfinite_features = 0
    dropped_any = 0
    for index in range(feature_count):
        if noise_nonpositive_policy == NOISE_NONPOSITIVE_POLICY_DROP_FEATURES:
            noise_valid = all(_valid_single_sigma(run.sigma_squared[index]) for run in runs)
        else:
            noise_valid = _valid_pooled_sigma(sigma_pool[index])
        feature_valid = all(_finite_difference(run, index) for run in runs)
        if noise_valid and feature_valid:
            valid_indexes.append(index)
            continue
        dropped_any += 1
        if not noise_valid:
            dropped_noise += 1
        if not feature_valid:
            dropped_nonfinite_features += 1
    dropped_fraction = 0.0 if feature_count == 0 else dropped_any / feature_count
    return _ManualFeatureFilter(
        indexes=tuple(valid_indexes),
        dropped_noise_feature_count=dropped_noise,
        dropped_nonfinite_feature_count=dropped_nonfinite_features,
        dropped_feature_count=dropped_any,
        dropped_feature_fraction=dropped_fraction,
    )


def _run_qc_with_feature_filter(
    run_qc: Sequence[ManualCrossnobisRunQc],
    *,
    runs: Sequence[ManualCrossnobisRun],
    feature_filter: _ManualFeatureFilter,
    noise_nonpositive_policy: str,
) -> list[ManualCrossnobisRunQc]:
    valid_by_id = {run.run_id: run for run in runs}
    resolved: list[ManualCrossnobisRunQc] = []
    for row in run_qc:
        run = valid_by_id.get(row.run_id)
        if run is None or row.status != "valid":
            resolved.append(row)
            continue
        nonpositive_noise_count = sum(1 for value in run.sigma_squared if not _valid_single_sigma(value))
        nonfinite_feature_count = sum(
            1 for index in range(len(run.condition_a)) if not _finite_difference(run, index)
        )
        resolved.append(
            ManualCrossnobisRunQc(
                run_id=row.run_id,
                status=row.status,
                reason=row.reason,
                event_count_a=row.event_count_a,
                event_count_b=row.event_count_b,
                excluded=row.excluded,
                original_feature_count=len(run.condition_a),
                retained_feature_count=len(feature_filter.indexes),
                dropped_noise_feature_count=nonpositive_noise_count,
                dropped_nonfinite_feature_count=nonfinite_feature_count,
                dropped_feature_count=feature_filter.dropped_feature_count,
                dropped_feature_fraction=feature_filter.dropped_feature_fraction,
                noise_nonpositive_policy=noise_nonpositive_policy,
            )
        )
    return resolved


def _feature_filter_warnings(
    feature_filter: _ManualFeatureFilter,
    *,
    noise_nonpositive_policy: str,
    warn_dropped_feature_fraction: float,
) -> tuple[str, ...]:
    if noise_nonpositive_policy != NOISE_NONPOSITIVE_POLICY_DROP_FEATURES:
        return ()
    if feature_filter.dropped_feature_count < 1:
        return ()
    if feature_filter.dropped_feature_fraction <= warn_dropped_feature_fraction:
        return ()
    return (
        "Manual diagonal crossnobis dropped "
        f"{feature_filter.dropped_feature_count} feature(s) "
        f"({feature_filter.dropped_feature_fraction:.3f}) because noise or feature values were invalid.",
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    value = sum(left_value * right_value for left_value, right_value in zip(left, right))
    if not math.isfinite(value):
        raise ValueError("Manual diagonal crossnobis dot product must be finite.")
    return value


def _raw_feature_count(runs: Sequence[ManualCrossnobisRun]) -> int:
    if not runs:
        return 0
    return len(runs[0].condition_a)


def _feature_length_errors(runs: Sequence[ManualCrossnobisRun], *, expected: int) -> tuple[str, ...]:
    errors: list[str] = []
    for run in runs:
        lengths = {
            "condition_a": len(run.condition_a),
            "condition_b": len(run.condition_b),
            "sigma_squared": len(run.sigma_squared),
        }
        mismatches = {name: length for name, length in lengths.items() if length != expected}
        if mismatches:
            errors.append(f"Run {run.run_id} feature lengths do not match {expected}: {mismatches}.")
    return tuple(errors)


def _result(
    *,
    status: str,
    crossnobis: float | None,
    runs: Sequence[ManualCrossnobisRun],
    valid_runs: Sequence[ManualCrossnobisRun],
    n_voxels_raw: int,
    n_voxels_used: int,
    min_events: int | None,
    min_valid_voxels: int,
    noise_nonpositive_policy: str,
    run_qc: Sequence[ManualCrossnobisRunQc],
    retained_feature_count: int | None = None,
    dropped_noise_feature_count: int = 0,
    dropped_nonfinite_feature_count: int = 0,
    dropped_feature_count: int = 0,
    dropped_feature_fraction: float = 0.0,
    warn_dropped_feature_fraction: float | None = None,
    run_pair_details: Sequence[ManualCrossnobisRunPairDetail] = (),
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
) -> ManualCrossnobisResult:
    valid_run_ids = tuple(run.run_id for run in valid_runs)
    excluded_run_ids = tuple(run.run_id for run in runs if run.excluded)
    invalid_run_ids = tuple(
        row.run_id for row in run_qc if row.status not in {"valid", "excluded"}
    )
    return ManualCrossnobisResult(
        status=status,
        crossnobis=crossnobis,
        n_valid_runs=len(valid_runs),
        valid_runs=valid_run_ids,
        excluded_runs=excluded_run_ids,
        invalid_runs=invalid_run_ids,
        n_voxels_raw=n_voxels_raw,
        n_voxels_used=n_voxels_used,
        min_events=min_events,
        min_valid_voxels=min_valid_voxels,
        sigma_pooling_source_runs=valid_run_ids,
        noise_nonpositive_policy=noise_nonpositive_policy,
        retained_feature_count=n_voxels_used if retained_feature_count is None else retained_feature_count,
        dropped_noise_feature_count=dropped_noise_feature_count,
        dropped_nonfinite_feature_count=dropped_nonfinite_feature_count,
        dropped_feature_count=dropped_feature_count,
        dropped_feature_fraction=dropped_feature_fraction,
        warn_dropped_feature_fraction=warn_dropped_feature_fraction,
        run_qc=tuple(run_qc),
        run_pair_details=tuple(run_pair_details),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _normalize_noise_nonpositive_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized in {"", NOISE_NONPOSITIVE_POLICY_STRICT, "fail", NOISE_NONPOSITIVE_POLICY_DROP_PATTERN}:
        return NOISE_NONPOSITIVE_POLICY_STRICT
    if normalized in {
        NOISE_NONPOSITIVE_POLICY_DROP_FEATURES,
        NOISE_NONPOSITIVE_POLICY_FILTER_NONPOSITIVE_FEATURES,
    }:
        return NOISE_NONPOSITIVE_POLICY_DROP_FEATURES
    raise ValueError(
        "noise_nonpositive_policy must be one of: "
        f"{', '.join(sorted(SUPPORTED_NOISE_NONPOSITIVE_POLICIES))}."
    )


def _validated_warn_dropped_feature_fraction(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("warn_dropped_feature_fraction must be between 0 and 1.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("warn_dropped_feature_fraction must be between 0 and 1.") from exc
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError("warn_dropped_feature_fraction must be between 0 and 1.")
    return numeric


def _non_empty_label(value: object, *, field_name: str) -> str:
    label = str(value).strip()
    if not label:
        raise ValueError(f"{field_name} must be a non-empty value.")
    return label


def _float_tuple(values: Sequence[float], *, field_name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must contain numeric values.")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field_name} must contain numeric values.") from exc
    result: list[float] = []
    for value in raw_values:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must contain numeric values.")
        try:
            result.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain numeric values.") from exc
    if not result:
        raise ValueError(f"{field_name} must contain at least one value.")
    return tuple(result)


def _validate_optional_count(value: int | None, *, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer when provided.")


def _json_safe_dataclass(value: Any) -> dict[str, Any]:
    return {field_.name: _json_safe(getattr(value, field_.name)) for field_ in fields(value)}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child) for child in value]
    return str(value)


__all__ = [
    "DEFAULT_MANUAL_MIN_RETAINED_FEATURES",
    "DEFAULT_MANUAL_WARN_DROPPED_FEATURE_FRACTION",
    "ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1",
    "ManualCrossnobisResult",
    "ManualCrossnobisRun",
    "ManualCrossnobisRunPairDetail",
    "ManualCrossnobisRunQc",
    "NOISE_NONPOSITIVE_POLICY_DROP_FEATURES",
    "NOISE_NONPOSITIVE_POLICY_DROP_PATTERN",
    "NOISE_NONPOSITIVE_POLICY_FILTER_NONPOSITIVE_FEATURES",
    "NOISE_NONPOSITIVE_POLICY_STRICT",
    "SUPPORTED_NOISE_NONPOSITIVE_POLICIES",
    "compute_manual_diagonal_crossnobis_v1",
]
