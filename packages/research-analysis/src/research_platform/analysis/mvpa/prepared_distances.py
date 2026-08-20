from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
import math
import re
from typing import Any

from .contracts import (
    CrossValidationSpec,
    DistanceEstimate,
    DistanceRequest,
    METRIC_CROSSNOBIS,
    NOISE_NORMALIZATION_DIAGONAL,
    NOISE_NORMALIZATION_IDENTITY,
    NoiseNormalization,
    PatternDataset,
    PatternObservation,
)
from .native_reference import NativeReferenceDistanceEngine
from .manual_crossnobis import (
    DEFAULT_MANUAL_MIN_RETAINED_FEATURES,
    DEFAULT_MANUAL_WARN_DROPPED_FEATURE_FRACTION,
    ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
    ManualCrossnobisRun,
    NOISE_NONPOSITIVE_POLICY_DROP_FEATURES,
    NOISE_NONPOSITIVE_POLICY_DROP_PATTERN,
    NOISE_NONPOSITIVE_POLICY_FILTER_NONPOSITIVE_FEATURES,
    NOISE_NONPOSITIVE_POLICY_STRICT,
    SUPPORTED_NOISE_NONPOSITIVE_POLICIES,
    compute_manual_diagonal_crossnobis_v1,
)
from .row_preparation import (
    MvpaPatternRowPreparationQcRow,
    PreparedMvpaPatternGroup,
    PreparedMvpaPatternRow,
)


_RSATOOLBOX_ENGINE_NAME = "rsatoolbox"
_DISTANCE_ENGINE_FAILURE_CODES = frozenset(
    {
        "distance_computation_failed",
        "optional_distance_engine_failed",
        "optional_distance_engine_unavailable",
    }
)
_PROPAGATED_EXPECTED_EXCLUSION_QC_CODES = frozenset({"threshold_failure"})


@dataclass(frozen=True)
class PreparedMvpaDistanceRow:
    """One JSON-safe distance row computed from a prepared MVPA pattern group."""

    group_id: str
    group_key: Mapping[str, str]
    condition_id_a: str
    condition_id_b: str
    distance: float
    metric: str
    engine_name: str
    normalization_method: str
    cv_unit_count: int | None = None
    feature_count: int | None = None
    observation_count: int | None = None
    condition_pair_id: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _non_empty_label(self.group_id, field_name="group_id"))
        object.__setattr__(self, "group_key", dict(self.group_key))
        object.__setattr__(
            self,
            "condition_id_a",
            _non_empty_label(self.condition_id_a, field_name="condition_id_a"),
        )
        object.__setattr__(
            self,
            "condition_id_b",
            _non_empty_label(self.condition_id_b, field_name="condition_id_b"),
        )
        object.__setattr__(self, "distance", _finite_float(self.distance, field_name="distance"))
        object.__setattr__(self, "metric", _non_empty_label(self.metric, field_name="metric"))
        object.__setattr__(self, "engine_name", _non_empty_label(self.engine_name, field_name="engine_name"))
        object.__setattr__(
            self,
            "normalization_method",
            _non_empty_label(self.normalization_method, field_name="normalization_method"),
        )
        _validate_optional_non_negative(self.cv_unit_count, field_name="cv_unit_count")
        _validate_optional_non_negative(self.feature_count, field_name="feature_count")
        _validate_optional_non_negative(self.observation_count, field_name="observation_count")
        if self.condition_pair_id is not None:
            object.__setattr__(
                self,
                "condition_pair_id",
                _non_empty_label(self.condition_pair_id, field_name="condition_pair_id"),
            )
        object.__setattr__(self, "context", dict(self.context))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PreparedMvpaDistanceQcRow:
    """One JSON-safe QC row from prepared-group MVPA distance computation."""

    level: str
    status: str
    code: str
    message: str
    source: str = "prepared_distances"
    source_row_index: int | None = None
    group_id: str | None = None
    group_key: Mapping[str, str] = field(default_factory=dict)
    pattern_id: str | None = None
    condition_id: str | None = None
    condition_id_a: str | None = None
    condition_id_b: str | None = None
    condition_pair_id: str | None = None
    cv_unit: str | None = None
    cv_label: str | None = None
    usable: bool | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", _non_empty_label(self.level, field_name="level"))
        object.__setattr__(self, "status", _non_empty_label(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_label(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_label(self.message, field_name="message"))
        object.__setattr__(self, "source", _non_empty_label(self.source, field_name="source"))
        object.__setattr__(self, "group_key", dict(self.group_key))
        object.__setattr__(self, "context", dict(self.context))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PreparedMvpaDistanceProvenanceRow:
    """One JSON-safe provenance key/value for prepared-group distance computation."""

    key: str
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty_label(self.key, field_name="key"))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PreparedMvpaDistanceResult:
    """JSON-safe in-memory result for prepared-group MVPA distance computation."""

    distances: Sequence[PreparedMvpaDistanceRow]
    qc_rows: Sequence[PreparedMvpaDistanceQcRow]
    provenance: Sequence[PreparedMvpaDistanceProvenanceRow]
    warnings: Sequence[str]
    errors: Sequence[str]
    executed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "distances", tuple(self.distances))
        object.__setattr__(self, "qc_rows", tuple(self.qc_rows))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class _ConditionPairSelection:
    id: str | None
    left: str
    right: str

    @property
    def key(self) -> frozenset[str]:
        return frozenset((self.left, self.right))


@dataclass(frozen=True)
class _ThresholdSweep:
    id: str
    min_events: int | None = None
    min_observations: int | None = None


def compute_mvpa_distances_from_prepared_groups(
    groups: Iterable[PreparedMvpaPatternGroup],
    *,
    metric: str = METRIC_CROSSNOBIS,
    engine_name: str = NativeReferenceDistanceEngine.name,
    noise_normalization_method: str = NOISE_NORMALIZATION_IDENTITY,
    noise_aggregation: str = "mean",
    minimum_cv_units: int = 2,
    require_balanced_condition_cv: bool = True,
    condition_pairs: Sequence[Any] | None = None,
    threshold_sweeps: Sequence[Mapping[str, Any]] | None = None,
    preparation_qc_rows: Iterable[MvpaPatternRowPreparationQcRow] | None = None,
    noise_nonpositive_policy: str = NOISE_NONPOSITIVE_POLICY_STRICT,
    min_retained_features: int = DEFAULT_MANUAL_MIN_RETAINED_FEATURES,
    warn_dropped_feature_fraction: float = DEFAULT_MANUAL_WARN_DROPPED_FEATURE_FRACTION,
) -> PreparedMvpaDistanceResult:
    """Compute MVPA distances from Phase 3B.1 prepared groups.

    Phase 3B.4 supports crossnobis distances with the default
    ``native_reference`` engine, or the optional ``rsatoolbox`` adapter when
    explicitly requested. Identity normalization remains the default; diagonal
    normalization uses Phase 3B.3 prepared-row ``noise_values`` aggregation.
    """

    engine_name = _validated_engine_name(engine_name)
    metric = _validated_metric(metric)
    noise_normalization_method = _validated_noise_normalization_method(noise_normalization_method)
    noise_aggregation = _validated_noise_aggregation(noise_aggregation)
    minimum_cv_units = _validated_minimum_cv_units(minimum_cv_units)
    selected_pairs = _validated_condition_pairs(condition_pairs)
    thresholds = _validated_threshold_sweeps(threshold_sweeps)
    resolved_noise_policy = _validated_noise_nonpositive_policy(noise_nonpositive_policy)
    resolved_min_retained_features = _validated_min_retained_features(min_retained_features)
    resolved_warn_dropped_fraction = _validated_warn_dropped_feature_fraction(warn_dropped_feature_fraction)
    group_rows = tuple(groups)

    qc_rows = list(_propagated_preparation_qc_rows(preparation_qc_rows or ()))
    distances: list[PreparedMvpaDistanceRow] = []
    diagonal_noise_groups_attempted = 0
    diagonal_noise_groups_succeeded = 0
    diagonal_noise_groups_failed = 0

    for group in sorted(group_rows, key=_group_sort_key):
        diagonal_variances: tuple[float, ...] | None = None
        if (
            noise_normalization_method == NOISE_NORMALIZATION_DIAGONAL
            and engine_name != ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1
        ):
            diagonal_noise_groups_attempted += 1
            diagonal_variances, diagonal_qc_rows = _diagonal_variances_from_prepared_group(
                group,
                metric=metric,
                engine_name=engine_name,
                normalization_method=noise_normalization_method,
                noise_aggregation=noise_aggregation,
            )
            if diagonal_qc_rows:
                diagonal_noise_groups_failed += 1
                qc_rows.extend(diagonal_qc_rows)
                continue
            diagonal_noise_groups_succeeded += 1

        group_distances, group_qc_rows = _compute_group_distances(
            group,
            engine_name=engine_name,
            diagonal_variances=diagonal_variances,
            metric=metric,
            noise_normalization_method=noise_normalization_method,
            minimum_cv_units=minimum_cv_units,
            require_balanced_condition_cv=require_balanced_condition_cv,
            selected_pairs=selected_pairs,
            threshold_sweeps=thresholds,
            noise_nonpositive_policy=resolved_noise_policy,
            min_retained_features=resolved_min_retained_features,
            warn_dropped_feature_fraction=resolved_warn_dropped_fraction,
        )
        distances.extend(group_distances)
        qc_rows.extend(group_qc_rows)

    warnings, errors = _messages_from_qc(qc_rows)
    distance_engine_failure_count = _distance_engine_failure_count(qc_rows)
    provenance = _provenance_rows(
        input_group_count=len(group_rows),
        distance_count=len(distances),
        qc_row_count=len(qc_rows),
        metric=metric,
        engine_name=engine_name,
        noise_normalization_method=noise_normalization_method,
        noise_aggregation=noise_aggregation,
        minimum_cv_units=minimum_cv_units,
        require_balanced_condition_cv=require_balanced_condition_cv,
        condition_pair_count=len(selected_pairs) if selected_pairs is not None else None,
        threshold_sweep_count=len(thresholds),
        threshold_failure_count=sum(1 for row in qc_rows if row.code == "threshold_failure"),
        diagonal_noise_groups_attempted=diagonal_noise_groups_attempted,
        diagonal_noise_groups_succeeded=diagonal_noise_groups_succeeded,
        diagonal_noise_groups_failed=diagonal_noise_groups_failed,
        distance_engine_failure_count=distance_engine_failure_count,
        noise_nonpositive_policy=resolved_noise_policy,
        min_retained_features=resolved_min_retained_features,
        warn_dropped_feature_fraction=resolved_warn_dropped_fraction,
    )
    return PreparedMvpaDistanceResult(
        distances=tuple(distances),
        qc_rows=tuple(qc_rows),
        provenance=provenance,
        warnings=warnings,
        errors=errors,
        executed=True,
    )


def _compute_group_distances(
    group: PreparedMvpaPatternGroup,
    *,
    engine_name: str,
    diagonal_variances: Sequence[float] | None,
    metric: str,
    noise_normalization_method: str,
    minimum_cv_units: int,
    require_balanced_condition_cv: bool,
    selected_pairs: tuple[_ConditionPairSelection, ...] | None,
    threshold_sweeps: Sequence[_ThresholdSweep],
    noise_nonpositive_policy: str,
    min_retained_features: int,
    warn_dropped_feature_fraction: float,
) -> tuple[tuple[PreparedMvpaDistanceRow, ...], tuple[PreparedMvpaDistanceQcRow, ...]]:
    rows = tuple(group.rows)
    try:
        group_condition_order = _group_condition_order(group, rows)
    except ValueError as exc:
        return (), (
            _qc_row(
                level="group",
                status="failed",
                code="invalid_condition_order",
                message=f"Group {group.group_id} has invalid condition_ids: {exc}",
                group_id=group.group_id,
                group_key=group.group_key,
                cv_unit=group.cv_unit,
            ),
        )
    selected_condition_order, allowed_pair_keys, condition_pair_ids, pair_qc_rows = _selected_condition_order(
        group,
        condition_order=group_condition_order,
        selected_pairs=selected_pairs,
    )
    if pair_qc_rows:
        return (), pair_qc_rows

    if engine_name == ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1:
        return _compute_manual_diagonal_group_distances(
            group,
            rows=rows,
            selected_condition_order=selected_condition_order,
            allowed_pair_keys=allowed_pair_keys,
            condition_pair_ids=condition_pair_ids,
            metric=metric,
            noise_normalization_method=noise_normalization_method,
            minimum_cv_units=minimum_cv_units,
            threshold_sweeps=threshold_sweeps,
            noise_nonpositive_policy=noise_nonpositive_policy,
            min_retained_features=min_retained_features,
            warn_dropped_feature_fraction=warn_dropped_feature_fraction,
        )

    threshold_qc_rows: tuple[PreparedMvpaDistanceQcRow, ...] = ()
    if threshold_sweeps:
        allowed_pair_keys, condition_pair_ids, threshold_qc_rows = _threshold_filtered_pairs(
            group,
            rows=rows,
            condition_order=selected_condition_order,
            allowed_pair_keys=allowed_pair_keys,
            condition_pair_ids=condition_pair_ids,
            threshold_sweeps=threshold_sweeps,
        )
        if allowed_pair_keys is not None and not allowed_pair_keys:
            return (), threshold_qc_rows

    selected_rows = tuple(row for row in rows if row.condition_id in set(selected_condition_order))
    validation_qc_rows = _group_validation_qc_rows(
        group,
        rows=selected_rows,
        condition_order=selected_condition_order,
        minimum_cv_units=minimum_cv_units,
        require_balanced_condition_cv=require_balanced_condition_cv,
    )
    if validation_qc_rows:
        return (), (*threshold_qc_rows, *validation_qc_rows)

    try:
        dataset = _pattern_dataset_from_prepared_rows(group, selected_rows)
        request = DistanceRequest(
            metric=metric,
            condition_order=selected_condition_order,
            cross_validation=CrossValidationSpec(
                minimum_units=minimum_cv_units,
                require_balanced_units=require_balanced_condition_cv,
            ),
            noise_normalization=NoiseNormalization(method=noise_normalization_method),
            engine_name=engine_name,
        )
        engine = _distance_engine(
            engine_name=engine_name,
            diagonal_variances=diagonal_variances,
        )
        estimates = engine.compute_distances(dataset, request)
    except (ImportError, RuntimeError, ValueError) as exc:
        return (), (
            _distance_failure_qc_row(
                group,
                engine_name=engine_name,
                metric=metric,
                normalization_method=noise_normalization_method,
                error=exc,
            ),
        )

    filtered_estimates = _filtered_estimates(estimates, allowed_pair_keys=allowed_pair_keys)
    ordered_estimates = _ordered_estimates(filtered_estimates, condition_order=selected_condition_order)
    return (
        tuple(
            _distance_row_from_estimate(
                estimate,
                group=group,
                condition_order=selected_condition_order,
                condition_pair_ids=condition_pair_ids,
            )
            for estimate in ordered_estimates
        ),
        threshold_qc_rows,
    )


def _pattern_dataset_from_prepared_rows(
    group: PreparedMvpaPatternGroup,
    rows: Sequence[PreparedMvpaPatternRow],
) -> PatternDataset:
    return PatternDataset(
        observations=tuple(
            PatternObservation(
                pattern_id=row.pattern_id,
                condition_id=row.condition_id,
                cv_unit=row.cv_label,
                features=row.feature_values,
                context={
                    "group_id": group.group_id,
                    "prepared_cv_unit": row.cv_unit,
                    "source_row_index": row.source_row_index,
                },
            )
            for row in rows
        ),
        feature_names=_feature_names(group.feature_count),
    )


def _compute_manual_diagonal_group_distances(
    group: PreparedMvpaPatternGroup,
    *,
    rows: Sequence[PreparedMvpaPatternRow],
    selected_condition_order: Sequence[str],
    allowed_pair_keys: frozenset[frozenset[str]] | None,
    condition_pair_ids: Mapping[frozenset[str], str | None],
    metric: str,
    noise_normalization_method: str,
    minimum_cv_units: int,
    threshold_sweeps: Sequence[_ThresholdSweep],
    noise_nonpositive_policy: str,
    min_retained_features: int,
    warn_dropped_feature_fraction: float,
) -> tuple[tuple[PreparedMvpaDistanceRow, ...], tuple[PreparedMvpaDistanceQcRow, ...]]:
    if noise_normalization_method != NOISE_NORMALIZATION_DIAGONAL:
        return (), (
            _qc_row(
                level="group",
                status="failed",
                code="manual_estimator_requires_diagonal_noise",
                message=(
                    f"Group {group.group_id} requested {ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1!r}, "
                    "which requires diagonal noise normalization."
                ),
                group_id=group.group_id,
                group_key=group.group_key,
                cv_unit=group.cv_unit,
            ),
        )

    candidate_pair_keys = allowed_pair_keys or _all_pair_keys(selected_condition_order)
    distance_rows: list[PreparedMvpaDistanceRow] = []
    qc_rows: list[PreparedMvpaDistanceQcRow] = []
    min_events = _manual_min_events(threshold_sweeps)

    for pair_key in sorted(candidate_pair_keys, key=lambda key: _pair_sort_key(key, selected_condition_order)):
        left, right = _ordered_pair_values(pair_key, selected_condition_order)
        manual_runs, run_qc_rows = _manual_runs_from_prepared_rows(
            group,
            rows=rows,
            condition_id_a=left,
            condition_id_b=right,
            noise_nonpositive_policy=noise_nonpositive_policy,
        )
        qc_rows.extend(run_qc_rows)
        result = compute_manual_diagonal_crossnobis_v1(
            manual_runs,
            min_events=min_events,
            min_valid_voxels=_manual_min_valid_voxels(
                noise_nonpositive_policy,
                min_retained_features=min_retained_features,
            ),
            noise_nonpositive_policy=noise_nonpositive_policy,
            warn_dropped_feature_fraction=warn_dropped_feature_fraction,
        )
        if result.n_valid_runs < minimum_cv_units and result.status == "ok":
            result_status = "insufficient_valid_runs"
        else:
            result_status = result.status
        if result_status != "ok" or result.crossnobis is None:
            pair_id = condition_pair_ids.get(pair_key)
            qc_rows.append(
                _qc_row(
                    level="group",
                    status="failed",
                    code=result_status,
                    message=(
                        f"Group {group.group_id} condition pair {left}/{right} failed "
                        f"{ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1}: "
                        f"{'; '.join(result.errors) if result.errors else result_status}."
                    ),
                    group_id=group.group_id,
                    group_key=group.group_key,
                    condition_id_a=left,
                    condition_id_b=right,
                    condition_pair_id=pair_id,
                    cv_unit=group.cv_unit,
                    context=result.to_dict(),
                )
            )
            continue
        for warning in result.warnings:
            qc_rows.append(
                _qc_row(
                    level="group",
                    status="warning",
                    code="manual_diagonal_noise_feature_filter_warning",
                    message=f"Group {group.group_id} condition pair {left}/{right}: {warning}",
                    group_id=group.group_id,
                    group_key=group.group_key,
                    condition_id_a=left,
                    condition_id_b=right,
                    condition_pair_id=condition_pair_ids.get(pair_key),
                    cv_unit=group.cv_unit,
                    context=result.to_dict(),
                )
            )

        distance_rows.append(
            PreparedMvpaDistanceRow(
                group_id=group.group_id,
                group_key=group.group_key,
                condition_id_a=left,
                condition_id_b=right,
                condition_pair_id=condition_pair_ids.get(pair_key),
                distance=result.crossnobis,
                metric=metric,
                engine_name=ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
                normalization_method=noise_normalization_method,
                cv_unit_count=result.n_valid_runs,
                feature_count=result.n_voxels_used,
                observation_count=2 * result.n_valid_runs,
                context={
                    "estimator": ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1,
                    "feature_count": result.n_voxels_used,
                    "n_voxels_raw": result.n_voxels_raw,
                    "n_voxels_used": result.n_voxels_used,
                    "original_feature_count": result.n_voxels_raw,
                    "retained_feature_count": result.retained_feature_count,
                    "dropped_noise_feature_count": result.dropped_noise_feature_count,
                    "dropped_nonfinite_feature_count": result.dropped_nonfinite_feature_count,
                    "dropped_feature_count": result.dropped_feature_count,
                    "dropped_feature_fraction": result.dropped_feature_fraction,
                    "noise_nonpositive_policy": result.noise_nonpositive_policy,
                    "min_retained_features": min_retained_features,
                    "warn_dropped_feature_fraction": warn_dropped_feature_fraction,
                    "min_events": result.min_events,
                    "valid_runs": list(result.valid_runs),
                    "excluded_runs": list(result.excluded_runs),
                    "invalid_runs": list(result.invalid_runs),
                    "sigma_pooling_source_runs": list(result.sigma_pooling_source_runs),
                    "run_pair_details": [row.to_dict() for row in result.run_pair_details],
                    "observation_count": 2 * result.n_valid_runs,
                },
            )
        )

    return tuple(distance_rows), tuple(qc_rows)


def _manual_runs_from_prepared_rows(
    group: PreparedMvpaPatternGroup,
    *,
    rows: Sequence[PreparedMvpaPatternRow],
    condition_id_a: str,
    condition_id_b: str,
    noise_nonpositive_policy: str,
) -> tuple[tuple[ManualCrossnobisRun, ...], tuple[PreparedMvpaDistanceQcRow, ...]]:
    rows_by_cv_condition: OrderedDict[tuple[str, str], list[PreparedMvpaPatternRow]] = OrderedDict()
    for row in rows:
        if row.condition_id in {condition_id_a, condition_id_b}:
            rows_by_cv_condition.setdefault((row.cv_label, row.condition_id), []).append(row)

    cv_labels = _first_seen(row.cv_label for row in rows if row.condition_id in {condition_id_a, condition_id_b})
    manual_runs: list[ManualCrossnobisRun] = []
    qc_rows: list[PreparedMvpaDistanceQcRow] = []
    for cv_label in cv_labels:
        rows_a = tuple(rows_by_cv_condition.get((cv_label, condition_id_a), ()))
        rows_b = tuple(rows_by_cv_condition.get((cv_label, condition_id_b), ()))
        if not rows_a or not rows_b:
            qc_rows.append(
                _qc_row(
                    level="cv_unit",
                    status="warning",
                    code="manual_pair_missing_condition_in_run",
                    message=(
                        f"Group {group.group_id} CV unit {cv_label!r} does not have both "
                        f"{condition_id_a!r} and {condition_id_b!r}; it is not a valid manual run."
                    ),
                    group_id=group.group_id,
                    group_key=group.group_key,
                    condition_id_a=condition_id_a,
                    condition_id_b=condition_id_b,
                    cv_unit=group.cv_unit,
                    cv_label=cv_label,
                )
            )
            continue

        noise_values, noise_qc_row = _manual_run_noise_values(
            group,
            cv_label=cv_label,
            rows=(*rows_a, *rows_b),
            condition_id_a=condition_id_a,
            condition_id_b=condition_id_b,
            noise_nonpositive_policy=noise_nonpositive_policy,
        )
        if noise_qc_row is not None:
            qc_rows.append(noise_qc_row)
            continue

        manual_runs.append(
            ManualCrossnobisRun(
                run_id=cv_label,
                condition_a=_mean_prepared_vectors(tuple(row.feature_values for row in rows_a)),
                condition_b=_mean_prepared_vectors(tuple(row.feature_values for row in rows_b)),
                sigma_squared=noise_values,
                event_count_a=_minimum_present_event_count(rows_a),
                event_count_b=_minimum_present_event_count(rows_b),
            )
        )
    return tuple(manual_runs), tuple(qc_rows)


def _manual_run_noise_values(
    group: PreparedMvpaPatternGroup,
    *,
    cv_label: str,
    rows: Sequence[PreparedMvpaPatternRow],
    condition_id_a: str,
    condition_id_b: str,
    noise_nonpositive_policy: str,
) -> tuple[tuple[float, ...], PreparedMvpaDistanceQcRow | None]:
    if noise_nonpositive_policy == NOISE_NONPOSITIVE_POLICY_DROP_FEATURES:
        usable_noise_rows = tuple(row for row in rows if row.noise_values)
    else:
        usable_noise_rows = tuple(row for row in rows if row.noise_usable is True and row.noise_values)
    if not usable_noise_rows:
        return (), _qc_row(
            level="cv_unit",
            status="failed",
            code="manual_diagonal_noise_missing",
            message=(
                f"Group {group.group_id} CV unit {cv_label!r} has no usable diagonal noise values "
                f"for pair {condition_id_a}/{condition_id_b}."
            ),
            group_id=group.group_id,
            group_key=group.group_key,
            condition_id_a=condition_id_a,
            condition_id_b=condition_id_b,
            cv_unit=group.cv_unit,
            cv_label=cv_label,
        )

    hash_qc_row = _manual_run_noise_hash_qc_row(
        group,
        cv_label=cv_label,
        rows=usable_noise_rows,
        condition_id_a=condition_id_a,
        condition_id_b=condition_id_b,
    )
    if hash_qc_row is not None:
        return (), hash_qc_row

    first = tuple(usable_noise_rows[0].noise_values)
    if len(first) != group.feature_count:
        return (), _qc_row(
            level="cv_unit",
            status="failed",
            code="manual_diagonal_noise_feature_count_mismatch",
            message=(
                f"Group {group.group_id} CV unit {cv_label!r} noise feature count does not "
                f"match group.feature_count={group.feature_count}."
            ),
            group_id=group.group_id,
            group_key=group.group_key,
            condition_id_a=condition_id_a,
            condition_id_b=condition_id_b,
            cv_unit=group.cv_unit,
            cv_label=cv_label,
        )
    if noise_nonpositive_policy == NOISE_NONPOSITIVE_POLICY_STRICT:
        for feature_index, value in enumerate(first):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = math.nan
            if not math.isfinite(numeric_value):
                return (), _qc_row(
                    level="cv_unit",
                    status="failed",
                    code="manual_diagonal_noise_nonfinite_value",
                    message=(
                        f"Group {group.group_id} CV unit {cv_label!r} has non-finite "
                        f"diagonal noise at feature {feature_index} for pair {condition_id_a}/{condition_id_b}."
                    ),
                    group_id=group.group_id,
                    group_key=group.group_key,
                    condition_id_a=condition_id_a,
                    condition_id_b=condition_id_b,
                    cv_unit=group.cv_unit,
                    cv_label=cv_label,
                    context={"feature_index": feature_index, "noise_nonpositive_policy": noise_nonpositive_policy},
                )
            if numeric_value <= 0.0:
                return (), _qc_row(
                    level="cv_unit",
                    status="failed",
                    code="manual_diagonal_noise_nonpositive_value",
                    message=(
                        f"Group {group.group_id} CV unit {cv_label!r} has zero or negative "
                        f"diagonal noise at feature {feature_index} for pair {condition_id_a}/{condition_id_b}."
                    ),
                    group_id=group.group_id,
                    group_key=group.group_key,
                    condition_id_a=condition_id_a,
                    condition_id_b=condition_id_b,
                    cv_unit=group.cv_unit,
                    cv_label=cv_label,
                    context={"feature_index": feature_index, "noise_nonpositive_policy": noise_nonpositive_policy},
                )
    for row in usable_noise_rows[1:]:
        if tuple(row.noise_values) != first:
            return (), _qc_row(
                level="cv_unit",
                status="failed",
                code="manual_diagonal_noise_mismatch_within_run",
                message=(
                    f"Group {group.group_id} CV unit {cv_label!r} has inconsistent diagonal "
                    "noise values across condition rows."
                ),
                group_id=group.group_id,
                group_key=group.group_key,
                condition_id_a=condition_id_a,
                condition_id_b=condition_id_b,
                cv_unit=group.cv_unit,
                cv_label=cv_label,
            )
    return first, None


def _manual_run_noise_hash_qc_row(
    group: PreparedMvpaPatternGroup,
    *,
    cv_label: str,
    rows: Sequence[PreparedMvpaPatternRow],
    condition_id_a: str,
    condition_id_b: str,
) -> PreparedMvpaDistanceQcRow | None:
    group_voxel_hash = _optional_present_label(group.voxel_index_hash)
    for row in rows:
        row_voxel_hash = _optional_present_label(row.voxel_index_hash)
        noise_voxel_hash = _optional_present_label(row.noise_voxel_index_hash)
        if (
            group_voxel_hash is None
            or row_voxel_hash is None
            or noise_voxel_hash is None
            or row_voxel_hash != group_voxel_hash
            or noise_voxel_hash != group_voxel_hash
        ):
            return _qc_row(
                level="cv_unit",
                status="failed",
                code="manual_diagonal_noise_voxel_hash_mismatch",
                message=(
                    f"Group {group.group_id} CV unit {cv_label!r} does not have aligned "
                    "row, noise, and group voxel_index_hash values for manual diagonal noise."
                ),
                group_id=group.group_id,
                group_key=group.group_key,
                pattern_id=row.pattern_id,
                condition_id=row.condition_id,
                condition_id_a=condition_id_a,
                condition_id_b=condition_id_b,
                cv_unit=group.cv_unit,
                cv_label=cv_label,
                context={
                    "source_row_index": row.source_row_index,
                    "group_voxel_index_hash": group_voxel_hash,
                    "row_voxel_index_hash": row_voxel_hash,
                    "noise_voxel_index_hash": noise_voxel_hash,
                },
            )
    return None


def _manual_min_events(threshold_sweeps: Sequence[_ThresholdSweep]) -> int | None:
    values = tuple(sweep.min_events for sweep in threshold_sweeps if sweep.min_events is not None)
    return max(values) if values else None


def _minimum_present_event_count(rows: Sequence[PreparedMvpaPatternRow]) -> int | None:
    counts = tuple(row.event_count for row in rows if row.event_count is not None)
    return min(counts) if counts else None


def _diagonal_variances_from_prepared_group(
    group: PreparedMvpaPatternGroup,
    *,
    metric: str,
    engine_name: str,
    normalization_method: str,
    noise_aggregation: str,
) -> tuple[tuple[float, ...] | None, tuple[PreparedMvpaDistanceQcRow, ...]]:
    rows = tuple(group.rows)
    base_context = _diagonal_noise_context(
        metric=metric,
        engine_name=engine_name,
        normalization_method=normalization_method,
        noise_aggregation=noise_aggregation,
    )
    if not rows:
        return None, (
            _qc_row(
                level="group",
                status="failed",
                code="diagonal_noise_missing",
                message=(
                    f"Group {group.group_id} has no prepared rows from which to aggregate "
                    "diagonal noise variances."
                ),
                group_id=group.group_id,
                group_key=group.group_key,
                cv_unit=group.cv_unit,
                context=base_context,
            ),
        )

    noise_available = tuple(_row_has_diagonal_noise_metadata(row) for row in rows)
    if not any(noise_available):
        return None, (
            _qc_row(
                level="group",
                status="failed",
                code="diagonal_noise_missing",
                message=f"Group {group.group_id} has no prepared rows with diagonal noise metadata.",
                group_id=group.group_id,
                group_key=group.group_key,
                cv_unit=group.cv_unit,
                context=base_context,
            ),
        )

    qc_rows: list[PreparedMvpaDistanceQcRow] = []
    if not all(noise_available):
        qc_rows.append(
            _qc_row(
                level="group",
                status="failed",
                code="diagonal_noise_partial_availability",
                message=(
                    f"Group {group.group_id} has diagonal noise metadata for only "
                    f"{sum(1 for available in noise_available if available)} of {len(rows)} prepared rows."
                ),
                group_id=group.group_id,
                group_key=group.group_key,
                cv_unit=group.cv_unit,
                context=base_context,
            )
        )

    row_variances: list[tuple[float, ...]] = []
    for row, available in zip(rows, noise_available):
        row_context = _diagonal_noise_context(
            metric=metric,
            engine_name=engine_name,
            normalization_method=normalization_method,
            noise_aggregation=noise_aggregation,
            expected_feature_count=group.feature_count,
        )
        if not available:
            qc_rows.append(
                _diagonal_noise_row_qc(
                    group,
                    row,
                    code="diagonal_noise_missing",
                    message=(
                        f"Prepared row {row.source_row_index} in group {group.group_id} "
                        "has no diagonal noise metadata."
                    ),
                    context=row_context,
                )
            )
            continue

        values, row_qc_rows = _validated_diagonal_noise_values(
            group,
            row,
            metric=metric,
            engine_name=engine_name,
            normalization_method=normalization_method,
            noise_aggregation=noise_aggregation,
        )
        qc_rows.extend(row_qc_rows)
        if not row_qc_rows:
            row_variances.append(values)

    if qc_rows:
        return None, tuple(qc_rows)

    return _mean_diagonal_variances(row_variances, feature_count=group.feature_count), ()


def _validated_diagonal_noise_values(
    group: PreparedMvpaPatternGroup,
    row: PreparedMvpaPatternRow,
    *,
    metric: str,
    engine_name: str,
    normalization_method: str,
    noise_aggregation: str,
) -> tuple[tuple[float, ...], tuple[PreparedMvpaDistanceQcRow, ...]]:
    qc_rows: list[PreparedMvpaDistanceQcRow] = []
    base_context = _diagonal_noise_context(
        metric=metric,
        engine_name=engine_name,
        normalization_method=normalization_method,
        noise_aggregation=noise_aggregation,
        expected_feature_count=group.feature_count,
        noise_feature_count=row.noise_feature_count,
        noise_value_count=len(row.noise_values),
    )

    if row.noise_usable is not True:
        qc_rows.append(
            _diagonal_noise_row_qc(
                group,
                row,
                code="diagonal_noise_unusable",
                message=(
                    f"Prepared row {row.source_row_index} in group {group.group_id} "
                    "does not have noise_usable=True."
                ),
                context=base_context,
            )
        )

    if not row.noise_values:
        qc_rows.append(
            _diagonal_noise_row_qc(
                group,
                row,
                code="diagonal_noise_values_missing",
                message=(
                    f"Prepared row {row.source_row_index} in group {group.group_id} "
                    "has no noise_values for diagonal normalization."
                ),
                context=base_context,
            )
        )
    elif len(row.noise_values) != group.feature_count:
        qc_rows.append(
            _diagonal_noise_row_qc(
                group,
                row,
                code="diagonal_noise_feature_count_mismatch",
                message=(
                    f"Prepared row {row.source_row_index} in group {group.group_id} has "
                    f"{len(row.noise_values)} noise_values but group.feature_count is {group.feature_count}."
                ),
                context=base_context,
            )
        )

    if row.noise_feature_count != group.feature_count:
        qc_rows.append(
            _diagonal_noise_row_qc(
                group,
                row,
                code="diagonal_noise_feature_count_mismatch",
                message=(
                    f"Prepared row {row.source_row_index} in group {group.group_id} declares "
                    f"noise_feature_count={row.noise_feature_count!r}, expected {group.feature_count}."
                ),
                context=base_context,
            )
        )

    group_voxel_hash = _optional_present_label(group.voxel_index_hash)
    row_voxel_hash = _optional_present_label(row.voxel_index_hash)
    noise_voxel_hash = _optional_present_label(row.noise_voxel_index_hash)
    if (
        group_voxel_hash is None
        or row_voxel_hash is None
        or noise_voxel_hash is None
        or row_voxel_hash != group_voxel_hash
        or noise_voxel_hash != group_voxel_hash
    ):
        qc_rows.append(
            _diagonal_noise_row_qc(
                group,
                row,
                code="diagonal_noise_voxel_hash_mismatch",
                message=(
                    f"Prepared row {row.source_row_index} in group {group.group_id} does not have "
                    "aligned row, noise, and group voxel_index_hash values."
                ),
                context=base_context,
            )
        )

    values: list[float] = []
    for feature_index, raw_value in enumerate(row.noise_values):
        numeric_value, value_qc_row = _validated_diagonal_noise_value(
            raw_value,
            group=group,
            row=row,
            feature_index=feature_index,
            metric=metric,
            engine_name=engine_name,
            normalization_method=normalization_method,
            noise_aggregation=noise_aggregation,
        )
        if value_qc_row is not None:
            qc_rows.append(value_qc_row)
            continue
        values.append(numeric_value)

    return tuple(values), tuple(qc_rows)


def _validated_diagonal_noise_value(
    value: object,
    *,
    group: PreparedMvpaPatternGroup,
    row: PreparedMvpaPatternRow,
    feature_index: int,
    metric: str,
    engine_name: str,
    normalization_method: str,
    noise_aggregation: str,
) -> tuple[float, PreparedMvpaDistanceQcRow | None]:
    context = _diagonal_noise_context(
        metric=metric,
        engine_name=engine_name,
        normalization_method=normalization_method,
        noise_aggregation=noise_aggregation,
        expected_feature_count=group.feature_count,
        noise_feature_count=row.noise_feature_count,
        noise_value_count=len(row.noise_values),
        feature_index=feature_index,
    )
    if isinstance(value, bool):
        return 0.0, _diagonal_noise_row_qc(
            group,
            row,
            code="diagonal_noise_nonfinite_value",
            message=(
                f"Prepared row {row.source_row_index} in group {group.group_id} has a non-numeric "
                f"diagonal noise value at feature index {feature_index}."
            ),
            context=context,
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0, _diagonal_noise_row_qc(
            group,
            row,
            code="diagonal_noise_nonfinite_value",
            message=(
                f"Prepared row {row.source_row_index} in group {group.group_id} has a non-numeric "
                f"diagonal noise value at feature index {feature_index}."
            ),
            context=context,
        )
    if not math.isfinite(numeric):
        return 0.0, _diagonal_noise_row_qc(
            group,
            row,
            code="diagonal_noise_nonfinite_value",
            message=(
                f"Prepared row {row.source_row_index} in group {group.group_id} has a non-finite "
                f"diagonal noise value at feature index {feature_index}."
            ),
            context=context,
        )
    if numeric <= 0.0:
        return 0.0, _diagonal_noise_row_qc(
            group,
            row,
            code="diagonal_noise_nonpositive_value",
            message=(
                f"Prepared row {row.source_row_index} in group {group.group_id} has a non-positive "
                f"diagonal noise value at feature index {feature_index}."
            ),
            context=context,
        )
    return numeric, None


def _row_has_diagonal_noise_metadata(row: PreparedMvpaPatternRow) -> bool:
    return (
        row.noise_usable is not None
        or bool(row.noise_values)
        or row.noise_feature_count is not None
        or row.noise_voxel_index_hash is not None
    )


def _mean_diagonal_variances(
    row_variances: Sequence[Sequence[float]],
    *,
    feature_count: int,
) -> tuple[float, ...]:
    if not row_variances:
        raise ValueError("At least one prepared row is required to aggregate diagonal noise variances.")
    sums = [0.0] * feature_count
    for values in row_variances:
        for index, value in enumerate(values):
            sums[index] += value
    return tuple(value / len(row_variances) for value in sums)


def _mean_prepared_vectors(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("At least one prepared vector is required.")
    feature_count = len(vectors[0])
    sums = [0.0] * feature_count
    for vector in vectors:
        for index, value in enumerate(vector):
            sums[index] += value
    return tuple(value / len(vectors) for value in sums)


def _diagonal_noise_row_qc(
    group: PreparedMvpaPatternGroup,
    row: PreparedMvpaPatternRow,
    *,
    code: str,
    message: str,
    context: Mapping[str, Any],
) -> PreparedMvpaDistanceQcRow:
    return _qc_row(
        level="row",
        status="failed",
        code=code,
        message=message,
        source_row_index=row.source_row_index,
        group_id=group.group_id,
        group_key=group.group_key,
        pattern_id=row.pattern_id,
        condition_id=row.condition_id,
        cv_unit=row.cv_unit,
        cv_label=row.cv_label,
        usable=True,
        context=context,
    )


def _diagonal_noise_context(
    *,
    metric: str,
    engine_name: str,
    normalization_method: str,
    noise_aggregation: str,
    **extra: Any,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "phase": "3B.4",
        "metric": metric,
        "engine_name": engine_name,
        "normalization_method": normalization_method,
        "noise_aggregation": noise_aggregation,
    }
    context.update(extra)
    return context


def _distance_row_from_estimate(
    estimate: DistanceEstimate,
    *,
    group: PreparedMvpaPatternGroup,
    condition_order: Sequence[str],
    condition_pair_ids: Mapping[frozenset[str], str | None],
) -> PreparedMvpaDistanceRow:
    order_index = {condition_id: index for index, condition_id in enumerate(condition_order)}
    condition_id_a = estimate.condition_id_a
    condition_id_b = estimate.condition_id_b
    if order_index[condition_id_a] > order_index[condition_id_b]:
        condition_id_a, condition_id_b = condition_id_b, condition_id_a

    return PreparedMvpaDistanceRow(
        group_id=group.group_id,
        group_key=group.group_key,
        condition_id_a=condition_id_a,
        condition_id_b=condition_id_b,
        condition_pair_id=condition_pair_ids.get(frozenset((condition_id_a, condition_id_b))),
        distance=estimate.distance,
        metric=estimate.metric,
        engine_name=estimate.engine_name,
        normalization_method=estimate.normalization_method,
        cv_unit_count=estimate.cv_unit_count,
        feature_count=_optional_int_context(estimate.context.get("feature_count")),
        observation_count=_optional_int_context(estimate.context.get("observation_count")),
        context=estimate.context,
    )


def _group_validation_qc_rows(
    group: PreparedMvpaPatternGroup,
    *,
    rows: Sequence[PreparedMvpaPatternRow],
    condition_order: Sequence[str],
    minimum_cv_units: int,
    require_balanced_condition_cv: bool,
) -> tuple[PreparedMvpaDistanceQcRow, ...]:
    qc_rows: list[PreparedMvpaDistanceQcRow] = []
    if not rows:
        qc_rows.append(
            _qc_row(
                level="group",
                status="failed",
                code="empty_prepared_group",
                message=f"Group {group.group_id} has no prepared rows for distance computation.",
                group_id=group.group_id,
                group_key=group.group_key,
                cv_unit=group.cv_unit,
            )
        )
        return tuple(qc_rows)

    if len(condition_order) < 2:
        qc_rows.append(
            _qc_row(
                level="group",
                status="failed",
                code="too_few_conditions",
                message=f"Group {group.group_id} requires at least two selected conditions.",
                group_id=group.group_id,
                group_key=group.group_key,
                cv_unit=group.cv_unit,
            )
        )

    observed_conditions = set(row.condition_id for row in rows)
    missing_conditions = tuple(
        condition_id for condition_id in condition_order if condition_id not in observed_conditions
    )
    if missing_conditions:
        qc_rows.append(
            _qc_row(
                level="group",
                status="failed",
                code="missing_selected_condition",
                message=(
                    f"Group {group.group_id} is missing prepared rows for selected conditions: "
                    f"{_joined(missing_conditions)}."
                ),
                group_id=group.group_id,
                group_key=group.group_key,
                cv_unit=group.cv_unit,
            )
        )

    cv_labels = _first_seen(row.cv_label for row in rows)
    if len(cv_labels) < minimum_cv_units:
        qc_rows.append(
            _qc_row(
                level="group",
                status="failed",
                code="insufficient_cv_units",
                message=(
                    f"Group {group.group_id} requires at least {minimum_cv_units} CV units; "
                    f"found {len(cv_labels)}."
                ),
                group_id=group.group_id,
                group_key=group.group_key,
                cv_unit=group.cv_unit,
                context={"cv_unit_count": len(cv_labels), "minimum_cv_units": minimum_cv_units},
            )
        )

    if require_balanced_condition_cv:
        conditions_by_cv: dict[str, set[str]] = {cv_label: set() for cv_label in cv_labels}
        for row in rows:
            conditions_by_cv[row.cv_label].add(row.condition_id)
        for cv_label in cv_labels:
            missing = tuple(
                condition_id
                for condition_id in condition_order
                if condition_id not in conditions_by_cv[cv_label]
            )
            if missing:
                qc_rows.append(
                    _qc_row(
                        level="group",
                        status="failed",
                        code="unbalanced_condition_cv_presence",
                        message=(
                            f"Group {group.group_id} CV unit {cv_label!r} is missing selected conditions: "
                            f"{_joined(missing)}."
                        ),
                        group_id=group.group_id,
                        group_key=group.group_key,
                        cv_unit=group.cv_unit,
                        cv_label=cv_label,
                    )
                )

    width_mismatches = tuple(row.pattern_id for row in rows if len(row.feature_values) != group.feature_count)
    if width_mismatches:
        qc_rows.append(
            _qc_row(
                level="group",
                status="failed",
                code="prepared_feature_width_mismatch",
                message=(
                    f"Group {group.group_id} has prepared rows whose feature width does not match "
                    f"group.feature_count: {_joined(width_mismatches)}."
                ),
                group_id=group.group_id,
                group_key=group.group_key,
                cv_unit=group.cv_unit,
            )
        )

    return tuple(qc_rows)


def _selected_condition_order(
    group: PreparedMvpaPatternGroup,
    *,
    condition_order: Sequence[str],
    selected_pairs: tuple[_ConditionPairSelection, ...] | None,
) -> tuple[
    tuple[str, ...],
    frozenset[frozenset[str]] | None,
    Mapping[frozenset[str], str | None],
    tuple[PreparedMvpaDistanceQcRow, ...],
]:
    if selected_pairs is None:
        return tuple(condition_order), None, {}, ()

    known_conditions = set(condition_order)
    requested_conditions = set(condition_id for pair in selected_pairs for condition_id in (pair.left, pair.right))
    unknown_conditions = tuple(
        condition_id for condition_id in sorted(requested_conditions) if condition_id not in known_conditions
    )
    if unknown_conditions:
        return (
            (),
            None,
            {},
            (
                _qc_row(
                    level="group",
                    status="failed",
                    code="unknown_condition_pair",
                    message=(
                        f"Group {group.group_id} condition_pairs reference unknown conditions: "
                        f"{_joined(unknown_conditions)}."
                    ),
                    group_id=group.group_id,
                    group_key=group.group_key,
                    cv_unit=group.cv_unit,
                ),
            ),
        )

    selected_condition_order = tuple(
        condition_id for condition_id in condition_order if condition_id in requested_conditions
    )
    condition_pair_ids = {pair.key: pair.id for pair in selected_pairs}
    allowed_pair_keys = frozenset(condition_pair_ids)
    return selected_condition_order, allowed_pair_keys, condition_pair_ids, ()


def _filtered_estimates(
    estimates: Sequence[DistanceEstimate],
    *,
    allowed_pair_keys: frozenset[frozenset[str]] | None,
) -> tuple[DistanceEstimate, ...]:
    if allowed_pair_keys is None:
        return tuple(estimates)
    return tuple(
        estimate
        for estimate in estimates
        if frozenset((estimate.condition_id_a, estimate.condition_id_b)) in allowed_pair_keys
    )


def _threshold_filtered_pairs(
    group: PreparedMvpaPatternGroup,
    *,
    rows: Sequence[PreparedMvpaPatternRow],
    condition_order: Sequence[str],
    allowed_pair_keys: frozenset[frozenset[str]] | None,
    condition_pair_ids: Mapping[frozenset[str], str | None],
    threshold_sweeps: Sequence[_ThresholdSweep],
) -> tuple[frozenset[frozenset[str]] | None, Mapping[frozenset[str], str | None], tuple[PreparedMvpaDistanceQcRow, ...]]:
    candidate_pair_keys = allowed_pair_keys or _all_pair_keys(condition_order)
    counts_by_condition: dict[str, int] = {condition_id: 0 for condition_id in condition_order}
    for row in rows:
        if row.condition_id in counts_by_condition:
            counts_by_condition[row.condition_id] += 1

    failed_pair_keys: set[frozenset[str]] = set()
    qc_rows: list[PreparedMvpaDistanceQcRow] = []
    for sweep in threshold_sweeps:
        if sweep.min_observations is None:
            continue
        for pair_key in sorted(candidate_pair_keys, key=lambda key: _pair_sort_key(key, condition_order)):
            left, right = _ordered_pair_values(pair_key, condition_order)
            observed = min(counts_by_condition.get(left, 0), counts_by_condition.get(right, 0))
            if observed >= sweep.min_observations:
                continue
            failed_pair_keys.add(pair_key)
            pair_id = condition_pair_ids.get(pair_key)
            qc_rows.append(
                _qc_row(
                    level="threshold",
                    status="failed",
                    code="threshold_failure",
                    message=(
                        f"Group {group.group_id} condition pair {left}/{right} failed threshold "
                        f"{sweep.id!r} for min_observations: observed {observed}, "
                        f"required {sweep.min_observations}."
                    ),
                    group_id=group.group_id,
                    group_key=group.group_key,
                    condition_id_a=left,
                    condition_id_b=right,
                    condition_pair_id=pair_id,
                    cv_unit=group.cv_unit,
                    context={
                        "threshold_id": sweep.id,
                        "threshold_type": "min_observations",
                        "required_value": sweep.min_observations,
                        "observed_value": observed,
                        "failure_reason": "observed_observations_below_required_minimum",
                        "condition_pair_id": pair_id,
                    },
                )
            )

    if not failed_pair_keys:
        return allowed_pair_keys, condition_pair_ids, tuple(qc_rows)

    remaining_pair_keys = frozenset(pair_key for pair_key in candidate_pair_keys if pair_key not in failed_pair_keys)
    remaining_ids = {pair_key: condition_pair_ids.get(pair_key) for pair_key in remaining_pair_keys}
    return remaining_pair_keys, remaining_ids, tuple(qc_rows)


def _all_pair_keys(condition_order: Sequence[str]) -> frozenset[frozenset[str]]:
    return frozenset(
        frozenset((left, right))
        for left_index, left in enumerate(condition_order)
        for right in condition_order[left_index + 1 :]
    )


def _pair_sort_key(pair_key: frozenset[str], condition_order: Sequence[str]) -> tuple[int, int]:
    order_index = {condition_id: index for index, condition_id in enumerate(condition_order)}
    values = sorted(order_index[condition_id] for condition_id in pair_key)
    return values[0], values[1]


def _ordered_pair_values(pair_key: frozenset[str], condition_order: Sequence[str]) -> tuple[str, str]:
    order_index = {condition_id: index for index, condition_id in enumerate(condition_order)}
    left, right = sorted(tuple(pair_key), key=lambda condition_id: order_index[condition_id])
    return left, right


def _ordered_estimates(
    estimates: Sequence[DistanceEstimate],
    *,
    condition_order: Sequence[str],
) -> tuple[DistanceEstimate, ...]:
    order_index = {condition_id: index for index, condition_id in enumerate(condition_order)}
    return tuple(
        sorted(
            estimates,
            key=lambda estimate: (
                min(order_index[estimate.condition_id_a], order_index[estimate.condition_id_b]),
                max(order_index[estimate.condition_id_a], order_index[estimate.condition_id_b]),
            ),
        )
    )


def _group_condition_order(
    group: PreparedMvpaPatternGroup,
    rows: Sequence[PreparedMvpaPatternRow],
) -> tuple[str, ...]:
    raw_order = tuple(group.condition_ids) or _first_seen(row.condition_id for row in rows)
    condition_order = _validated_condition_order(raw_order, label=f"group {group.group_id} condition_ids")
    known_conditions = set(condition_order)
    missing_observed = tuple(
        condition_id
        for condition_id in _first_seen(row.condition_id for row in rows)
        if condition_id not in known_conditions
    )
    if missing_observed:
        raise ValueError(f"condition_ids is missing observed conditions: {_joined(missing_observed)}.")
    return condition_order


def _validated_condition_pairs(
    condition_pairs: Sequence[Any] | None,
) -> tuple[_ConditionPairSelection, ...] | None:
    if condition_pairs is None:
        return None
    if isinstance(condition_pairs, (str, bytes)):
        raise ValueError("condition_pairs must be a sequence of two-condition sequences, not a string.")

    pairs: list[_ConditionPairSelection] = []
    seen: set[frozenset[str]] = set()
    duplicate_pairs: list[str] = []
    for index, raw_pair in enumerate(condition_pairs, start=1):
        pair_id: str | None = None
        if isinstance(raw_pair, Mapping):
            pair_id = _optional_present_label(raw_pair.get("id") or raw_pair.get("name"))
            raw_values = (
                raw_pair.get("left", raw_pair.get("condition_id_a", raw_pair.get("condition_a"))),
                raw_pair.get("right", raw_pair.get("condition_id_b", raw_pair.get("condition_b"))),
            )
        elif isinstance(raw_pair, (str, bytes)):
            raise ValueError(f"condition_pairs item {index} must contain exactly two condition ids.")
        else:
            try:
                raw_values = tuple(raw_pair)
            except TypeError as exc:
                raise ValueError(f"condition_pairs item {index} must contain exactly two condition ids.") from exc
            if len(raw_values) != 2:
                raise ValueError(f"condition_pairs item {index} must contain exactly two condition ids.")
        if raw_values[0] is None or raw_values[1] is None:
            raise ValueError(f"condition_pairs item {index} must contain exactly two condition ids.")
        condition_id_a = _non_empty_label(raw_values[0], field_name="condition_pairs")
        condition_id_b = _non_empty_label(raw_values[1], field_name="condition_pairs")
        if condition_id_a == condition_id_b:
            raise ValueError("condition_pairs must contain two distinct condition ids per pair.")
        pair_key = frozenset((condition_id_a, condition_id_b))
        if pair_key in seen:
            duplicate_pairs.append(f"{condition_id_a}/{condition_id_b}")
        seen.add(pair_key)
        pairs.append(_ConditionPairSelection(pair_id, condition_id_a, condition_id_b))
    if duplicate_pairs:
        raise ValueError(f"condition_pairs must be unique as unordered pairs: {_joined(duplicate_pairs)}.")
    if not pairs:
        raise ValueError("condition_pairs must contain at least one pair when provided.")
    return tuple(pairs)


def _validated_threshold_sweeps(value: Sequence[Mapping[str, Any]] | None) -> tuple[_ThresholdSweep, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError("threshold_sweeps must be a sequence of mappings.")
    sweeps: list[_ThresholdSweep] = []
    for index, raw_sweep in enumerate(value, start=1):
        if not isinstance(raw_sweep, Mapping):
            raise ValueError(f"threshold_sweeps item {index} must be a mapping.")
        threshold_id = _optional_present_label(raw_sweep.get("id") or raw_sweep.get("name")) or f"threshold-{index}"
        min_events = _optional_int_context(raw_sweep.get("min_events", raw_sweep.get("min_events_per_condition")))
        min_observations = _optional_int_context(
            raw_sweep.get("min_observations", raw_sweep.get("min_observations_per_condition"))
        )
        if min_events is None and min_observations is None:
            raise ValueError(f"threshold_sweeps item {index} must define min_events or min_observations.")
        if min_events is not None and min_events < 0:
            raise ValueError(f"threshold_sweeps item {index} min_events must be non-negative.")
        if min_observations is not None and min_observations < 0:
            raise ValueError(f"threshold_sweeps item {index} min_observations must be non-negative.")
        sweeps.append(_ThresholdSweep(threshold_id, min_events=min_events, min_observations=min_observations))
    return tuple(sweeps)


def _validated_condition_order(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of condition ids, not a string.")
    resolved: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        condition_id = _non_empty_label(value, field_name=label)
        if condition_id in seen and condition_id not in duplicates:
            duplicates.append(condition_id)
        seen.add(condition_id)
        resolved.append(condition_id)
    if duplicates:
        raise ValueError(f"{label} must be unique: {_joined(duplicates)}.")
    return tuple(resolved)


def _validated_engine_name(engine_name: str) -> str:
    resolved = _non_empty_label(engine_name, field_name="engine_name")
    supported = {NativeReferenceDistanceEngine.name, _RSATOOLBOX_ENGINE_NAME, ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1}
    if resolved not in supported:
        raise ValueError(
            "Phase 3B.4 prepared-group MVPA distances support only "
            f"engine_name values: {_joined(sorted(supported))}."
        )
    return resolved


def _validated_metric(metric: str) -> str:
    resolved = _non_empty_label(metric, field_name="metric")
    if resolved != METRIC_CROSSNOBIS:
        raise ValueError("Phase 3B.4 prepared-group MVPA distances support crossnobis only.")
    return resolved


def _validated_noise_normalization_method(noise_normalization_method: str) -> str:
    resolved = _non_empty_label(noise_normalization_method, field_name="noise_normalization_method")
    if resolved not in {NOISE_NORMALIZATION_IDENTITY, NOISE_NORMALIZATION_DIAGONAL}:
        raise ValueError(
            "Phase 3B.4 prepared-group MVPA distances support only identity or diagonal "
            "noise normalization."
        )
    return resolved


def _validated_noise_aggregation(noise_aggregation: str) -> str:
    resolved = _non_empty_label(noise_aggregation, field_name="noise_aggregation")
    if resolved != "mean":
        raise ValueError("Phase 3B.4 prepared-row diagonal noise aggregation supports only 'mean'.")
    return resolved


def _distance_engine(
    *,
    engine_name: str,
    diagonal_variances: Sequence[float] | None,
) -> Any:
    if engine_name == NativeReferenceDistanceEngine.name:
        return NativeReferenceDistanceEngine(diagonal_variances=diagonal_variances)
    if engine_name == _RSATOOLBOX_ENGINE_NAME:
        from .rsatoolbox_adapter import RsatoolboxDistanceEngine

        return RsatoolboxDistanceEngine(diagonal_variances=diagonal_variances)
    raise ValueError(f"Unsupported prepared-distance engine_name {engine_name!r}.")


def _distance_failure_qc_row(
    group: PreparedMvpaPatternGroup,
    *,
    engine_name: str,
    metric: str,
    normalization_method: str,
    error: BaseException,
) -> PreparedMvpaDistanceQcRow:
    message = str(error)
    context: dict[str, Any] = {
        "phase": "3B.4",
        "engine_name": engine_name,
        "metric": metric,
        "normalization_method": normalization_method,
        "error_type": type(error).__name__,
    }
    if engine_name == _RSATOOLBOX_ENGINE_NAME:
        context["native_reference_fallback_used"] = False
        optional_dependency = _optional_dependency_from_message(message)
        if optional_dependency is not None:
            context["optional_dependency"] = optional_dependency

    return _qc_row(
        level="group",
        status="failed",
        code=_distance_failure_code(engine_name=engine_name, error=error, message=message),
        message=f"Group {group.group_id} distance computation failed with engine {engine_name!r}: {message}",
        group_id=group.group_id,
        group_key=group.group_key,
        cv_unit=group.cv_unit,
        context=context,
    )


def _distance_failure_code(
    *,
    engine_name: str,
    error: BaseException,
    message: str,
) -> str:
    if engine_name == _RSATOOLBOX_ENGINE_NAME:
        if isinstance(error, ImportError) or message.startswith("Optional dependency"):
            return "optional_distance_engine_unavailable"
        return "optional_distance_engine_failed"
    return "distance_computation_failed"


def _optional_dependency_from_message(message: str) -> str | None:
    for dependency in ("numpy", "rsatoolbox"):
        if f"'{dependency}'" in message:
            return dependency
    return None


def _validated_minimum_cv_units(minimum_cv_units: int) -> int:
    if isinstance(minimum_cv_units, bool):
        raise ValueError("minimum_cv_units must be an integer of at least 2.")
    try:
        resolved = int(minimum_cv_units)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_cv_units must be an integer of at least 2.") from exc
    if resolved != minimum_cv_units:
        raise ValueError("minimum_cv_units must be an integer of at least 2.")
    if resolved < 2:
        raise ValueError("minimum_cv_units must be at least 2 for crossnobis distances.")
    return resolved


def _validated_noise_nonpositive_policy(policy: str) -> str:
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


def _validated_min_retained_features(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("min_retained_features must be an integer of at least 1.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_retained_features must be an integer of at least 1.") from exc
    if resolved != value:
        raise ValueError("min_retained_features must be an integer of at least 1.")
    if resolved < 1:
        raise ValueError("min_retained_features must be at least 1.")
    return resolved


def _validated_warn_dropped_feature_fraction(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("warn_dropped_feature_fraction must be between 0 and 1.")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("warn_dropped_feature_fraction must be between 0 and 1.") from exc
    if not math.isfinite(resolved) or resolved < 0.0 or resolved > 1.0:
        raise ValueError("warn_dropped_feature_fraction must be between 0 and 1.")
    return resolved


def _manual_min_valid_voxels(noise_nonpositive_policy: str, *, min_retained_features: int) -> int:
    if noise_nonpositive_policy == NOISE_NONPOSITIVE_POLICY_DROP_FEATURES:
        return min_retained_features
    return 1


def _propagated_preparation_qc_rows(
    preparation_qc_rows: Iterable[MvpaPatternRowPreparationQcRow],
) -> tuple[PreparedMvpaDistanceQcRow, ...]:
    rows: list[PreparedMvpaDistanceQcRow] = []
    for qc_row in preparation_qc_rows:
        context = dict(qc_row.context)
        context.setdefault("phase", "3B.1")
        rows.append(
            PreparedMvpaDistanceQcRow(
                level=qc_row.level,
                status=_propagated_preparation_qc_status(qc_row),
                code=qc_row.code,
                message=qc_row.message,
                source="row_preparation",
                source_row_index=qc_row.source_row_index,
                group_id=qc_row.group_id,
                group_key=qc_row.group_key,
                pattern_id=qc_row.pattern_id,
                condition_id=qc_row.condition_id,
                cv_unit=qc_row.cv_unit,
                cv_label=qc_row.cv_label,
                usable=qc_row.usable,
                context=context,
            )
        )
    return tuple(rows)


def _propagated_preparation_qc_status(qc_row: MvpaPatternRowPreparationQcRow) -> str:
    if qc_row.status == "failed" and qc_row.code in _PROPAGATED_EXPECTED_EXCLUSION_QC_CODES:
        return "warning"
    return qc_row.status


def _provenance_rows(
    *,
    input_group_count: int,
    distance_count: int,
    qc_row_count: int,
    metric: str,
    engine_name: str,
    noise_normalization_method: str,
    noise_aggregation: str,
    minimum_cv_units: int,
    require_balanced_condition_cv: bool,
    condition_pair_count: int | None,
    threshold_sweep_count: int,
    threshold_failure_count: int,
    diagonal_noise_groups_attempted: int,
    diagonal_noise_groups_succeeded: int,
    diagonal_noise_groups_failed: int,
    distance_engine_failure_count: int,
    noise_nonpositive_policy: str,
    min_retained_features: int,
    warn_dropped_feature_fraction: float,
) -> tuple[PreparedMvpaDistanceProvenanceRow, ...]:
    return (
        PreparedMvpaDistanceProvenanceRow("source", "research_platform.analysis.mvpa.prepared_distances"),
        PreparedMvpaDistanceProvenanceRow("phase", "3B.4"),
        PreparedMvpaDistanceProvenanceRow("input_group_count", input_group_count),
        PreparedMvpaDistanceProvenanceRow("distance_count", distance_count),
        PreparedMvpaDistanceProvenanceRow("qc_row_count", qc_row_count),
        PreparedMvpaDistanceProvenanceRow("metric", metric),
        PreparedMvpaDistanceProvenanceRow("engine_name", engine_name),
        PreparedMvpaDistanceProvenanceRow("noise_normalization_method", noise_normalization_method),
        PreparedMvpaDistanceProvenanceRow("noise_aggregation", noise_aggregation),
        PreparedMvpaDistanceProvenanceRow("noise_nonpositive_policy", noise_nonpositive_policy),
        PreparedMvpaDistanceProvenanceRow("min_retained_features", min_retained_features),
        PreparedMvpaDistanceProvenanceRow("warn_dropped_feature_fraction", warn_dropped_feature_fraction),
        PreparedMvpaDistanceProvenanceRow("minimum_cv_units", minimum_cv_units),
        PreparedMvpaDistanceProvenanceRow("require_balanced_condition_cv", require_balanced_condition_cv),
        PreparedMvpaDistanceProvenanceRow("condition_pair_count", condition_pair_count),
        PreparedMvpaDistanceProvenanceRow("threshold_sweep_count", threshold_sweep_count),
        PreparedMvpaDistanceProvenanceRow("threshold_failure_count", threshold_failure_count),
        PreparedMvpaDistanceProvenanceRow("diagonal_noise_groups_attempted", diagonal_noise_groups_attempted),
        PreparedMvpaDistanceProvenanceRow("diagonal_noise_groups_succeeded", diagonal_noise_groups_succeeded),
        PreparedMvpaDistanceProvenanceRow("diagonal_noise_groups_failed", diagonal_noise_groups_failed),
        PreparedMvpaDistanceProvenanceRow("distance_engine_failure_count", distance_engine_failure_count),
        PreparedMvpaDistanceProvenanceRow("distance_computation", True),
        PreparedMvpaDistanceProvenanceRow("output_written", False),
        PreparedMvpaDistanceProvenanceRow("native_reference_fallback_used", False),
        PreparedMvpaDistanceProvenanceRow("rsatoolbox_adapter_used", engine_name == _RSATOOLBOX_ENGINE_NAME),
    )


def _qc_row(
    *,
    level: str,
    status: str,
    code: str,
    message: str,
    source_row_index: int | None = None,
    group_id: str | None = None,
    group_key: Mapping[str, str] | None = None,
    pattern_id: str | None = None,
    condition_id: str | None = None,
    condition_id_a: str | None = None,
    condition_id_b: str | None = None,
    condition_pair_id: str | None = None,
    cv_unit: str | None = None,
    cv_label: str | None = None,
    usable: bool | None = None,
    context: Mapping[str, Any] | None = None,
) -> PreparedMvpaDistanceQcRow:
    return PreparedMvpaDistanceQcRow(
        level=level,
        status=status,
        code=code,
        message=message,
        source="prepared_distances",
        source_row_index=source_row_index,
        group_id=group_id,
        group_key=group_key or {},
        pattern_id=pattern_id,
        condition_id=condition_id,
        condition_id_a=condition_id_a,
        condition_id_b=condition_id_b,
        condition_pair_id=condition_pair_id,
        cv_unit=cv_unit,
        cv_label=cv_label,
        usable=usable,
        context=context or {},
    )


def _messages_from_qc(qc_rows: Sequence[PreparedMvpaDistanceQcRow]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    errors: list[str] = []
    for qc_row in qc_rows:
        if qc_row.status == "failed":
            errors.append(qc_row.message)
        else:
            warnings.append(qc_row.message)
    return _unique(warnings), _unique(errors)


def _distance_engine_failure_count(qc_rows: Sequence[PreparedMvpaDistanceQcRow]) -> int:
    return sum(1 for qc_row in qc_rows if qc_row.code in _DISTANCE_ENGINE_FAILURE_CODES)


def _feature_names(feature_count: int) -> tuple[str, ...]:
    if feature_count < 1:
        raise ValueError("Prepared MVPA groups must contain at least one feature.")
    return tuple(f"feature_{index:06d}" for index in range(feature_count))


def _group_sort_key(group: PreparedMvpaPatternGroup) -> tuple[tuple[tuple[str, str], ...], str]:
    group_key = dict(group.group_key)
    return (
        tuple((str(column), str(group_key.get(column, ""))) for column in group.group_by),
        str(group.group_id),
    )


def _first_seen(values: Iterable[object]) -> tuple[str, ...]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for value in values:
        ordered[str(value)] = None
    return tuple(ordered)


def _joined(values: Iterable[object]) -> str:
    return ", ".join(str(value) for value in values)


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for value in values:
        ordered[value] = None
    return tuple(ordered)


def _optional_int_context(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"-?\d+", text):
            return int(text)
    return None


def _validate_optional_non_negative(value: int | None, *, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided.")


def _non_empty_label(value: object, *, field_name: str) -> str:
    label = str(value).strip()
    if not label:
        raise ValueError(f"{field_name} must be a non-empty value.")
    return label


def _optional_present_label(value: object) -> str | None:
    if value is None:
        return None
    label = str(value).strip()
    return label or None


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


def _json_safe_dataclass(instance: object) -> dict[str, Any]:
    if not is_dataclass(instance):
        raise TypeError("_json_safe_dataclass requires a dataclass instance.")
    return {field.name: _json_safe(getattr(instance, field.name)) for field in fields(instance)}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON-safe prepared MVPA distance results cannot contain non-finite floats.")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "PreparedMvpaDistanceProvenanceRow",
    "PreparedMvpaDistanceQcRow",
    "PreparedMvpaDistanceResult",
    "PreparedMvpaDistanceRow",
    "compute_mvpa_distances_from_prepared_groups",
]
