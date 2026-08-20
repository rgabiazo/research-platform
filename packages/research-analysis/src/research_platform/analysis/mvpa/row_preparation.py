from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from typing import Any


DEFAULT_MVPA_PATTERN_GROUP_BY = (
    "subject_id",
    "session_id",
    "task_id",
    "direction",
    "model",
    "pattern_source_name",
    "roi_source_name",
    "roi_label",
)
SUPPORTED_MVPA_ROW_PREPARATION_CV_UNITS = frozenset({"run", "session", "subject", "custom"})

_CV_LABEL_COLUMNS = {
    "run": "run_id",
    "session": "session_id",
    "subject": "subject_id",
    "custom": "cv_unit",
}
_NOISE_COLUMNS = (
    "noise_values",
    "noise_usable",
    "noise_feature_count",
    "noise_voxel_index_hash",
)
_EXPECTED_EXCLUSION_QC_CODES = frozenset({"threshold_failure"})


@dataclass(frozen=True)
class PreparedMvpaPatternRow:
    """One validated MVPA pattern row ready for later distance computation."""

    pattern_id: str
    condition_id: str
    cv_unit: str
    cv_label: str
    feature_values: Sequence[float]
    feature_count: int
    group_key: Mapping[str, str]
    source_row_index: int
    voxel_order: str | None = None
    voxel_index_hash: str | None = None
    mean_centering_applied: bool | None = None
    mean_centering_scope: str | None = None
    event_count: int | None = None
    noise_values: Sequence[float] = ()
    noise_usable: bool | None = None
    noise_feature_count: int | None = None
    noise_voxel_index_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_values", tuple(self.feature_values))
        object.__setattr__(self, "group_key", dict(self.group_key))
        object.__setattr__(self, "noise_values", tuple(self.noise_values))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PreparedMvpaPatternGroup:
    """A deterministic group of prepared pattern rows sharing an analysis key."""

    group_id: str
    group_key: Mapping[str, str]
    group_by: Sequence[str]
    rows: Sequence[PreparedMvpaPatternRow]
    cv_unit: str
    cv_labels: Sequence[str]
    condition_ids: Sequence[str]
    feature_count: int
    voxel_order: str | None = None
    voxel_index_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_key", dict(self.group_key))
        object.__setattr__(self, "group_by", tuple(self.group_by))
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "cv_labels", tuple(self.cv_labels))
        object.__setattr__(self, "condition_ids", tuple(self.condition_ids))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MvpaPatternRowPreparationQcRow:
    """One row- or group-level QC finding from MVPA pattern row preparation."""

    level: str
    status: str
    code: str
    message: str
    source_row_index: int | None = None
    group_id: str | None = None
    group_key: Mapping[str, str] = field(default_factory=dict)
    pattern_id: str | None = None
    condition_id: str | None = None
    cv_unit: str | None = None
    cv_label: str | None = None
    usable: bool | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_key", dict(self.group_key))
        object.__setattr__(self, "context", dict(self.context))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MvpaPatternRowPreparationProvenanceRow:
    """One JSON-safe provenance key/value emitted by row preparation."""

    key: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MvpaPatternRowPreparationResult:
    """JSON-safe in-memory result for generic MVPA pattern row preparation."""

    groups: Sequence[PreparedMvpaPatternGroup]
    qc_rows: Sequence[MvpaPatternRowPreparationQcRow]
    provenance: Sequence[MvpaPatternRowPreparationProvenanceRow]
    warnings: Sequence[str]
    errors: Sequence[str]
    executed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "qc_rows", tuple(self.qc_rows))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class _ValidatedNoise:
    values: tuple[float, ...]
    usable: bool | None
    feature_count: int | None
    voxel_index_hash: str | None


@dataclass(frozen=True)
class _ThresholdSweep:
    id: str
    min_events: int | None = None
    min_observations: int | None = None


def prepare_mvpa_pattern_row_groups(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_by: Sequence[str] | None = None,
    cv_unit: str = "run",
    cv_label_column: str | None = None,
    condition_column: str = "condition_id",
    feature_values_column: str = "feature_values",
    threshold_sweeps: Sequence[Mapping[str, Any]] | None = None,
) -> MvpaPatternRowPreparationResult:
    """Validate extraction-like pattern rows and group them without computing distances.

    The input is deliberately mapping-based rather than dataframe-based. Prepared
    rows carry actual CV labels derived from ``run_id``, ``session_id``,
    ``subject_id``, or row ``cv_unit`` values, depending on the requested
    ``cv_unit`` mode. Callers with a backend-neutral canonical label may opt
    into another input column with ``cv_label_column``; omitting it preserves
    the historical column selected by ``cv_unit``.
    """

    resolved_cv_unit = _validated_cv_unit(cv_unit)
    resolved_cv_label_column = (
        _CV_LABEL_COLUMNS[resolved_cv_unit]
        if cv_label_column is None
        else _validated_column_name(cv_label_column, label="cv_label_column")
    )
    condition_column = _validated_column_name(condition_column, label="condition_column")
    feature_values_column = _validated_column_name(feature_values_column, label="feature_values_column")
    thresholds = _validated_threshold_sweeps(threshold_sweeps)
    raw_rows = tuple(rows)
    present_columns = _present_columns(raw_rows)
    group_columns = _resolved_group_by(group_by, present_columns=present_columns)

    prepared_rows: list[PreparedMvpaPatternRow] = []
    qc_rows: list[MvpaPatternRowPreparationQcRow] = []

    for source_row_index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            qc_rows.append(
                _qc_row(
                    level="row",
                    status="failed",
                    code="row_not_mapping",
                    message=f"Source row {source_row_index} is not a mapping.",
                    source_row_index=source_row_index,
                    cv_unit=resolved_cv_unit,
                    usable=False,
                )
            )
            continue

        prepared_row, row_qc = _prepare_one_row(
            row,
            source_row_index=source_row_index,
            group_by=group_columns,
            cv_unit=resolved_cv_unit,
            cv_label_column=resolved_cv_label_column,
            condition_column=condition_column,
            feature_values_column=feature_values_column,
            threshold_sweeps=thresholds,
        )
        qc_rows.extend(row_qc)
        if prepared_row is not None:
            prepared_rows.append(prepared_row)

    groups, group_qc_rows = _prepared_groups(prepared_rows, group_by=group_columns, cv_unit=resolved_cv_unit)
    qc_rows.extend(group_qc_rows)

    warnings, errors = _messages_from_qc(qc_rows)
    provenance = _provenance_rows(
        input_row_count=len(raw_rows),
        prepared_row_count=sum(len(group.rows) for group in groups),
        candidate_prepared_row_count=len(prepared_rows),
        group_count=len(groups),
        qc_row_count=len(qc_rows),
        cv_unit=resolved_cv_unit,
        cv_label_column=resolved_cv_label_column,
        condition_column=condition_column,
        feature_values_column=feature_values_column,
        group_by=group_columns,
        threshold_sweeps=thresholds,
        threshold_failure_count=sum(1 for row in qc_rows if row.code == "threshold_failure"),
    )
    return MvpaPatternRowPreparationResult(
        groups=tuple(groups),
        qc_rows=tuple(qc_rows),
        provenance=provenance,
        warnings=warnings,
        errors=errors,
        executed=True,
    )


def _prepare_one_row(
    row: Mapping[str, Any],
    *,
    source_row_index: int,
    group_by: Sequence[str],
    cv_unit: str,
    cv_label_column: str,
    condition_column: str,
    feature_values_column: str,
    threshold_sweeps: Sequence[_ThresholdSweep],
) -> tuple[PreparedMvpaPatternRow | None, tuple[MvpaPatternRowPreparationQcRow, ...]]:
    pattern_id = _pattern_id(row, source_row_index=source_row_index)
    condition_id = _optional_label(row.get(condition_column))
    cv_label = _optional_label(row.get(cv_label_column))
    group_key = _group_key(row, group_by)
    qc_rows: list[MvpaPatternRowPreparationQcRow] = []
    event_count, event_count_error = _optional_non_negative_int(row.get("event_count", row.get("n_events")), label="event_count")
    if event_count_error is not None:
        qc_rows.append(
            _qc_row(
                level="row",
                status="failed",
                code="invalid_event_count",
                message=f"Source row {source_row_index} {event_count_error}",
                source_row_index=source_row_index,
                group_key=group_key,
                pattern_id=pattern_id,
                condition_id=condition_id,
                cv_unit=cv_unit,
                cv_label=cv_label,
                usable=True,
            )
        )
    qc_rows.extend(
        _event_threshold_qc_rows(
            row,
            threshold_sweeps=threshold_sweeps,
            observed_event_count=event_count,
            source_row_index=source_row_index,
            group_key=group_key,
            pattern_id=pattern_id,
            condition_id=condition_id,
            cv_unit=cv_unit,
            cv_label=cv_label,
        )
    )

    if row.get("usable") is False:
        return None, (
            _qc_row(
                level="row",
                status="excluded",
                code="row_unusable",
                message=f"Source row {source_row_index} is marked usable=False.",
                source_row_index=source_row_index,
                group_key=group_key,
                pattern_id=pattern_id,
                condition_id=condition_id,
                cv_unit=cv_unit,
                cv_label=cv_label,
                usable=False,
            ),
        )

    if condition_id is None:
        qc_rows.append(
            _qc_row(
                level="row",
                status="failed",
                code="missing_condition_id",
                message=f"Source row {source_row_index} is missing non-empty {condition_column!r}.",
                source_row_index=source_row_index,
                group_key=group_key,
                pattern_id=pattern_id,
                cv_unit=cv_unit,
                cv_label=cv_label,
                usable=True,
            )
        )

    if cv_label is None:
        qc_rows.append(
            _qc_row(
                level="row",
                status="failed",
                code="missing_cv_label",
                message=(
                    f"Source row {source_row_index} is missing non-empty CV label column "
                    f"{cv_label_column!r} for cv_unit={cv_unit!r}."
                ),
                source_row_index=source_row_index,
                group_key=group_key,
                pattern_id=pattern_id,
                condition_id=condition_id,
                cv_unit=cv_unit,
                usable=True,
            )
        )

    missing_group_columns = _missing_group_columns(row, group_by)
    for column in missing_group_columns:
        qc_rows.append(
            _qc_row(
                level="row",
                status="failed",
                code="missing_group_value",
                message=f"Source row {source_row_index} is missing non-empty grouping value {column!r}.",
                source_row_index=source_row_index,
                group_key=group_key,
                pattern_id=pattern_id,
                condition_id=condition_id,
                cv_unit=cv_unit,
                cv_label=cv_label,
                usable=True,
            )
        )

    feature_values, feature_error = _finite_number_tuple(row.get(feature_values_column), label=feature_values_column)
    if feature_error is not None:
        qc_rows.append(
            _qc_row(
                level="row",
                status="failed",
                code="invalid_feature_values",
                message=f"Source row {source_row_index} {feature_error}",
                source_row_index=source_row_index,
                group_key=group_key,
                pattern_id=pattern_id,
                condition_id=condition_id,
                cv_unit=cv_unit,
                cv_label=cv_label,
                usable=True,
            )
        )

    feature_count = len(feature_values) if feature_values is not None else None
    declared_feature_count, count_error = _optional_non_negative_int(row.get("feature_count"), label="feature_count")
    if count_error is not None:
        qc_rows.append(
            _qc_row(
                level="row",
                status="failed",
                code="invalid_feature_count",
                message=f"Source row {source_row_index} {count_error}",
                source_row_index=source_row_index,
                group_key=group_key,
                pattern_id=pattern_id,
                condition_id=condition_id,
                cv_unit=cv_unit,
                cv_label=cv_label,
                usable=True,
            )
        )
    elif declared_feature_count is not None and feature_count is not None and declared_feature_count != feature_count:
        qc_rows.append(
            _qc_row(
                level="row",
                status="failed",
                code="feature_count_mismatch",
                message=(
                    f"Source row {source_row_index} declares feature_count={declared_feature_count}, "
                    f"but {feature_values_column!r} has length {feature_count}."
                ),
                source_row_index=source_row_index,
                group_key=group_key,
                pattern_id=pattern_id,
                condition_id=condition_id,
                cv_unit=cv_unit,
                cv_label=cv_label,
                usable=True,
            )
        )

    noise, noise_qc_rows = _validated_noise(
        row,
        source_row_index=source_row_index,
        group_key=group_key,
        pattern_id=pattern_id,
        condition_id=condition_id,
        cv_unit=cv_unit,
        cv_label=cv_label,
    )
    qc_rows.extend(noise_qc_rows)

    if any(qc.status == "failed" for qc in qc_rows):
        return None, tuple(qc_rows)

    if condition_id is None or cv_label is None or feature_values is None or feature_count is None:
        return None, tuple(qc_rows)

    return (
        PreparedMvpaPatternRow(
            pattern_id=pattern_id,
            condition_id=condition_id,
            cv_unit=cv_unit,
            cv_label=cv_label,
            feature_values=feature_values,
            feature_count=feature_count,
            group_key=group_key,
            source_row_index=source_row_index,
            voxel_order=_optional_label(row.get("voxel_order")),
            voxel_index_hash=_optional_label(row.get("voxel_index_hash")),
            mean_centering_applied=_optional_bool(row.get("mean_centering_applied")),
            mean_centering_scope=_optional_label(row.get("mean_centering_scope")),
            event_count=event_count,
            noise_values=noise.values,
            noise_usable=noise.usable,
            noise_feature_count=noise.feature_count,
            noise_voxel_index_hash=noise.voxel_index_hash,
        ),
        tuple(qc_rows),
    )


def _prepared_groups(
    rows: Sequence[PreparedMvpaPatternRow],
    *,
    group_by: Sequence[str],
    cv_unit: str,
) -> tuple[list[PreparedMvpaPatternGroup], list[MvpaPatternRowPreparationQcRow]]:
    grouped: OrderedDict[tuple[tuple[str, str], ...], list[PreparedMvpaPatternRow]] = OrderedDict()
    for row in sorted(rows, key=lambda candidate: candidate.source_row_index):
        grouped.setdefault(_group_tuple(row.group_key, group_by), []).append(row)

    groups: list[PreparedMvpaPatternGroup] = []
    qc_rows: list[MvpaPatternRowPreparationQcRow] = []
    for group_tuple in sorted(grouped):
        group_rows = tuple(grouped[group_tuple])
        group_key = dict(group_tuple)
        group_id = _group_id(group_key, group_by)
        group_failures = _group_consistency_qc(group_rows, group_id=group_id, group_key=group_key, cv_unit=cv_unit)
        qc_rows.extend(group_failures)
        if group_failures:
            continue

        groups.append(
            PreparedMvpaPatternGroup(
                group_id=group_id,
                group_key=group_key,
                group_by=tuple(group_by),
                rows=group_rows,
                cv_unit=cv_unit,
                cv_labels=_first_seen(row.cv_label for row in group_rows),
                condition_ids=_first_seen(row.condition_id for row in group_rows),
                feature_count=group_rows[0].feature_count,
                voxel_order=_only_present_value(row.voxel_order for row in group_rows),
                voxel_index_hash=_only_present_value(row.voxel_index_hash for row in group_rows),
            )
        )
    return groups, qc_rows


def _group_consistency_qc(
    rows: Sequence[PreparedMvpaPatternRow],
    *,
    group_id: str,
    group_key: Mapping[str, str],
    cv_unit: str,
) -> tuple[MvpaPatternRowPreparationQcRow, ...]:
    qc_rows: list[MvpaPatternRowPreparationQcRow] = []
    feature_counts = tuple(sorted({row.feature_count for row in rows}))
    if len(feature_counts) > 1:
        qc_rows.append(
            _qc_row(
                level="group",
                status="failed",
                code="feature_width_mismatch",
                message=f"Group {group_id} has inconsistent feature widths: {_joined(feature_counts)}.",
                group_id=group_id,
                group_key=group_key,
                cv_unit=cv_unit,
            )
        )

    voxel_orders = tuple(sorted({row.voxel_order for row in rows if row.voxel_order is not None}))
    if len(voxel_orders) > 1:
        qc_rows.append(
            _qc_row(
                level="group",
                status="failed",
                code="voxel_order_mismatch",
                message=f"Group {group_id} has inconsistent voxel_order values: {_joined(voxel_orders)}.",
                group_id=group_id,
                group_key=group_key,
                cv_unit=cv_unit,
            )
        )

    voxel_hashes = tuple(sorted({row.voxel_index_hash for row in rows if row.voxel_index_hash is not None}))
    if len(voxel_hashes) > 1:
        qc_rows.append(
            _qc_row(
                level="group",
                status="failed",
                code="voxel_index_hash_mismatch",
                message=f"Group {group_id} has inconsistent voxel_index_hash values: {_joined(voxel_hashes)}.",
                group_id=group_id,
                group_key=group_key,
                cv_unit=cv_unit,
            )
        )
    return tuple(qc_rows)


def _validated_noise(
    row: Mapping[str, Any],
    *,
    source_row_index: int,
    group_key: Mapping[str, str],
    pattern_id: str,
    condition_id: str | None,
    cv_unit: str,
    cv_label: str | None,
) -> tuple[_ValidatedNoise, tuple[MvpaPatternRowPreparationQcRow, ...]]:
    qc_rows: list[MvpaPatternRowPreparationQcRow] = []
    noise_fields_present = any(column in row for column in _NOISE_COLUMNS)
    raw_noise_values = row.get("noise_values")
    noise_values: tuple[float, ...] = ()
    noise_feature_count: int | None = None
    identity_noise_unused = (
        str(row.get("noise_status") or "").strip().lower() == "unused"
        and row.get("noise_usable") is False
        and raw_noise_values in (None, (), [])
    )

    if identity_noise_unused:
        pass
    elif "noise_values" in row and raw_noise_values is not None:
        values, error = _finite_number_tuple(raw_noise_values, label="noise_values")
        if error is None and values is not None:
            noise_values = values
        else:
            qc_rows.append(
                _qc_row(
                    level="noise",
                    status="warning",
                    code="invalid_noise_values",
                    message=f"Source row {source_row_index} {error}",
                    source_row_index=source_row_index,
                    group_key=group_key,
                    pattern_id=pattern_id,
                    condition_id=condition_id,
                    cv_unit=cv_unit,
                    cv_label=cv_label,
                    usable=True,
                )
            )
    elif noise_fields_present:
        qc_rows.append(
            _qc_row(
                level="noise",
                status="warning",
                code="missing_noise_values",
                message=f"Source row {source_row_index} has noise metadata but no usable noise_values.",
                source_row_index=source_row_index,
                group_key=group_key,
                pattern_id=pattern_id,
                condition_id=condition_id,
                cv_unit=cv_unit,
                cv_label=cv_label,
                usable=True,
            )
        )

    declared_noise_feature_count, count_error = _optional_non_negative_int(
        row.get("noise_feature_count"),
        label="noise_feature_count",
    )
    if count_error is not None:
        qc_rows.append(
            _qc_row(
                level="noise",
                status="warning",
                code="invalid_noise_feature_count",
                message=f"Source row {source_row_index} {count_error}",
                source_row_index=source_row_index,
                group_key=group_key,
                pattern_id=pattern_id,
                condition_id=condition_id,
                cv_unit=cv_unit,
                cv_label=cv_label,
                usable=True,
            )
        )
    elif declared_noise_feature_count is not None:
        noise_feature_count = declared_noise_feature_count
        if noise_values and declared_noise_feature_count != len(noise_values):
            qc_rows.append(
                _qc_row(
                    level="noise",
                    status="warning",
                    code="noise_feature_count_mismatch",
                    message=(
                        f"Source row {source_row_index} declares noise_feature_count="
                        f"{declared_noise_feature_count}, but noise_values has length {len(noise_values)}."
                    ),
                    source_row_index=source_row_index,
                    group_key=group_key,
                    pattern_id=pattern_id,
                    condition_id=condition_id,
                    cv_unit=cv_unit,
                    cv_label=cv_label,
                    usable=True,
                )
            )
    elif noise_values:
        noise_feature_count = len(noise_values)

    noise_usable = row.get("noise_usable")
    if noise_usable is not None and not isinstance(noise_usable, bool):
        qc_rows.append(
            _qc_row(
                level="noise",
                status="warning",
                code="invalid_noise_usable",
                message=f"Source row {source_row_index} noise_usable must be a boolean when provided.",
                source_row_index=source_row_index,
                group_key=group_key,
                pattern_id=pattern_id,
                condition_id=condition_id,
                cv_unit=cv_unit,
                cv_label=cv_label,
                usable=True,
            )
        )
        noise_usable = None

    return (
        _ValidatedNoise(
            values=noise_values,
            usable=noise_usable,
            feature_count=noise_feature_count,
            voxel_index_hash=_optional_label(row.get("noise_voxel_index_hash")),
        ),
        tuple(qc_rows),
    )


def _event_threshold_qc_rows(
    row: Mapping[str, Any],
    *,
    threshold_sweeps: Sequence[_ThresholdSweep],
    observed_event_count: int | None,
    source_row_index: int,
    group_key: Mapping[str, str],
    pattern_id: str,
    condition_id: str | None,
    cv_unit: str,
    cv_label: str | None,
) -> tuple[MvpaPatternRowPreparationQcRow, ...]:
    qc_rows: list[MvpaPatternRowPreparationQcRow] = []
    for sweep in threshold_sweeps:
        if sweep.min_events is None:
            continue
        failure_reason = None
        if observed_event_count is None:
            failure_reason = "event_count_metadata_missing"
        elif observed_event_count < sweep.min_events:
            failure_reason = "observed_events_below_required_minimum"
        if failure_reason is None:
            continue
        context = _threshold_context(
            row,
            threshold_id=sweep.id,
            threshold_type="min_events",
            required_value=sweep.min_events,
            observed_value=observed_event_count,
            failure_reason=failure_reason,
            condition_id=condition_id,
        )
        qc_rows.append(
            _qc_row(
                level="threshold",
                status="failed",
                code="threshold_failure",
                message=(
                    f"Source row {source_row_index} failed threshold {sweep.id!r} "
                    f"for min_events: observed {observed_event_count!r}, required {sweep.min_events}."
                ),
                source_row_index=source_row_index,
                group_key=group_key,
                pattern_id=pattern_id,
                condition_id=condition_id,
                cv_unit=cv_unit,
                cv_label=cv_label,
                usable=True,
                context=context,
            )
        )
    return tuple(qc_rows)


def _threshold_context(
    row: Mapping[str, Any],
    *,
    threshold_id: str,
    threshold_type: str,
    required_value: int,
    observed_value: int | None,
    failure_reason: str,
    condition_id: str | None,
) -> dict[str, Any]:
    context = {
        "threshold_id": threshold_id,
        "threshold_type": threshold_type,
        "required_value": required_value,
        "observed_value": observed_value,
        "failure_reason": failure_reason,
        "condition_id": condition_id,
    }
    for key in (
        "subject_id",
        "session_id",
        "run_id",
        "roi_label",
        "contrast_id",
        "group_id",
        "pattern_source_name",
        "roi_source_name",
    ):
        value = row.get(key)
        if value is not None:
            context[key] = _json_safe(value)
    return context


def _validated_threshold_sweeps(value: Sequence[Mapping[str, Any]] | None) -> tuple[_ThresholdSweep, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError("threshold_sweeps must be a sequence of mappings.")
    sweeps: list[_ThresholdSweep] = []
    for index, raw_sweep in enumerate(value, start=1):
        if not isinstance(raw_sweep, Mapping):
            raise ValueError(f"threshold_sweeps item {index} must be a mapping.")
        threshold_id = _optional_label(raw_sweep.get("id") or raw_sweep.get("name")) or f"threshold-{index}"
        min_events, min_events_error = _optional_non_negative_int(
            raw_sweep.get("min_events", raw_sweep.get("min_events_per_condition")),
            label="min_events",
        )
        min_observations, min_observations_error = _optional_non_negative_int(
            raw_sweep.get("min_observations", raw_sweep.get("min_observations_per_condition")),
            label="min_observations",
        )
        if min_events_error is not None:
            raise ValueError(f"threshold_sweeps item {index} {min_events_error}")
        if min_observations_error is not None:
            raise ValueError(f"threshold_sweeps item {index} {min_observations_error}")
        if min_events is None and min_observations is None:
            raise ValueError(f"threshold_sweeps item {index} must define min_events or min_observations.")
        sweeps.append(_ThresholdSweep(threshold_id, min_events=min_events, min_observations=min_observations))
    return tuple(sweeps)


def _validated_cv_unit(value: object) -> str:
    cv_unit = str(value).strip()
    if cv_unit not in SUPPORTED_MVPA_ROW_PREPARATION_CV_UNITS:
        supported = ", ".join(sorted(SUPPORTED_MVPA_ROW_PREPARATION_CV_UNITS))
        raise ValueError(f"Unsupported cv_unit {cv_unit!r}. Use one of: {supported}.")
    return cv_unit


def _validated_column_name(value: object, *, label: str) -> str:
    column = str(value).strip()
    if not column:
        raise ValueError(f"{label} must be a non-empty column name.")
    return column


def _resolved_group_by(group_by: Sequence[str] | None, *, present_columns: set[str]) -> tuple[str, ...]:
    if group_by is None:
        return tuple(column for column in DEFAULT_MVPA_PATTERN_GROUP_BY if column in present_columns)
    if isinstance(group_by, (str, bytes)):
        raise ValueError("group_by must be a sequence of column names, not a string.")

    columns = tuple(_validated_column_name(column, label="group_by") for column in group_by)
    duplicates = _duplicates(columns)
    if duplicates:
        raise ValueError(f"group_by must be unique: {', '.join(duplicates)}.")
    return columns


def _present_columns(rows: Sequence[object]) -> set[str]:
    columns: set[str] = set()
    for row in rows:
        if isinstance(row, Mapping):
            columns.update(str(column) for column in row)
    return columns


def _missing_group_columns(row: Mapping[str, Any], group_by: Sequence[str]) -> tuple[str, ...]:
    return tuple(column for column in group_by if _optional_label(row.get(column)) is None)


def _group_key(row: Mapping[str, Any], group_by: Sequence[str]) -> dict[str, str]:
    key: dict[str, str] = {}
    for column in group_by:
        value = _optional_label(row.get(column))
        if value is not None:
            key[column] = value
    return key


def _group_tuple(group_key: Mapping[str, str], group_by: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple((column, group_key[column]) for column in group_by)


def _group_id(group_key: Mapping[str, str], group_by: Sequence[str]) -> str:
    if not group_by:
        return "mvpa-pattern-group-all"
    payload = json.dumps(
        [(column, group_key[column]) for column in group_by],
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"mvpa-pattern-group-{digest}"


def _pattern_id(row: Mapping[str, Any], *, source_row_index: int) -> str:
    pattern_id = _optional_label(row.get("pattern_id"))
    if pattern_id is not None:
        return pattern_id
    return f"mvpa-pattern-row-{source_row_index:06d}"


def _optional_label(value: object) -> str | None:
    if value is None:
        return None
    label = str(value).strip()
    return label or None


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    return None


def _finite_number_tuple(value: object, *, label: str) -> tuple[tuple[float, ...] | None, str | None]:
    if value is None:
        return None, f"{label!r} must contain at least one finite numeric value."
    if isinstance(value, (str, bytes)):
        return None, f"{label!r} must contain finite numeric values, not a string."
    try:
        raw_values = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None, f"{label!r} must contain finite numeric values."
    if not raw_values:
        return None, f"{label!r} must contain at least one finite numeric value."

    values: list[float] = []
    for raw_value in raw_values:
        if isinstance(raw_value, bool):
            return None, f"{label!r} must contain finite non-bool numeric values."
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            return None, f"{label!r} must contain finite numeric values."
        if not math.isfinite(numeric):
            return None, f"{label!r} must contain finite numeric values."
        values.append(numeric)
    return tuple(values), None


def _optional_non_negative_int(value: object, *, label: str) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, f"{label} must be a non-negative integer when provided."
    if isinstance(value, int):
        count = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None, f"{label} must be a non-negative integer when provided."
        count = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None, None
        try:
            count = int(stripped)
        except ValueError:
            return None, f"{label} must be a non-negative integer when provided."
    else:
        return None, f"{label} must be a non-negative integer when provided."

    if count < 0:
        return None, f"{label} must be a non-negative integer when provided."
    return count, None


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
    cv_unit: str | None = None,
    cv_label: str | None = None,
    usable: bool | None = None,
    context: Mapping[str, Any] | None = None,
) -> MvpaPatternRowPreparationQcRow:
    return MvpaPatternRowPreparationQcRow(
        level=level,
        status=status,
        code=code,
        message=message,
        source_row_index=source_row_index,
        group_id=group_id,
        group_key=group_key or {},
        pattern_id=pattern_id,
        condition_id=condition_id,
        cv_unit=cv_unit,
        cv_label=cv_label,
        usable=usable,
        context=context or {},
    )


def _messages_from_qc(qc_rows: Sequence[MvpaPatternRowPreparationQcRow]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    errors: list[str] = []
    for qc_row in qc_rows:
        if qc_row.status == "failed" and qc_row.code not in _EXPECTED_EXCLUSION_QC_CODES:
            errors.append(qc_row.message)
        else:
            warnings.append(qc_row.message)
    return _unique(warnings), _unique(errors)


def _provenance_rows(
    *,
    input_row_count: int,
    prepared_row_count: int,
    candidate_prepared_row_count: int,
    group_count: int,
    qc_row_count: int,
    cv_unit: str,
    cv_label_column: str,
    condition_column: str,
    feature_values_column: str,
    group_by: Sequence[str],
    threshold_sweeps: Sequence[_ThresholdSweep],
    threshold_failure_count: int,
) -> tuple[MvpaPatternRowPreparationProvenanceRow, ...]:
    return (
        MvpaPatternRowPreparationProvenanceRow("source", "research_platform.analysis.mvpa.row_preparation"),
        MvpaPatternRowPreparationProvenanceRow("phase", "3B.1"),
        MvpaPatternRowPreparationProvenanceRow("input_row_count", input_row_count),
        MvpaPatternRowPreparationProvenanceRow("candidate_prepared_row_count", candidate_prepared_row_count),
        MvpaPatternRowPreparationProvenanceRow("prepared_row_count", prepared_row_count),
        MvpaPatternRowPreparationProvenanceRow("group_count", group_count),
        MvpaPatternRowPreparationProvenanceRow("qc_row_count", qc_row_count),
        MvpaPatternRowPreparationProvenanceRow("cv_unit", cv_unit),
        MvpaPatternRowPreparationProvenanceRow("cv_label_column", cv_label_column),
        MvpaPatternRowPreparationProvenanceRow("condition_column", condition_column),
        MvpaPatternRowPreparationProvenanceRow("feature_values_column", feature_values_column),
        MvpaPatternRowPreparationProvenanceRow("group_by", tuple(group_by)),
        MvpaPatternRowPreparationProvenanceRow(
            "threshold_sweeps",
            tuple(
                {
                    "id": sweep.id,
                    "min_events": sweep.min_events,
                    "min_observations": sweep.min_observations,
                }
                for sweep in threshold_sweeps
            ),
        ),
        MvpaPatternRowPreparationProvenanceRow("threshold_failure_count", threshold_failure_count),
        MvpaPatternRowPreparationProvenanceRow("distance_computation", False),
        MvpaPatternRowPreparationProvenanceRow("output_written", False),
    )


def _first_seen(values: Iterable[str]) -> tuple[str, ...]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for value in values:
        ordered[value] = None
    return tuple(ordered)


def _only_present_value(values: Iterable[str | None]) -> str | None:
    present = tuple(_first_seen(value for value in values if value is not None))
    if len(present) == 1:
        return present[0]
    return None


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _joined(values: Iterable[object]) -> str:
    return ", ".join(str(value) for value in values)


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for value in values:
        ordered[value] = None
    return tuple(ordered)


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
            raise ValueError("JSON-safe MVPA row preparation results cannot contain non-finite floats.")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "DEFAULT_MVPA_PATTERN_GROUP_BY",
    "MvpaPatternRowPreparationProvenanceRow",
    "MvpaPatternRowPreparationQcRow",
    "MvpaPatternRowPreparationResult",
    "PreparedMvpaPatternGroup",
    "PreparedMvpaPatternRow",
    "SUPPORTED_MVPA_ROW_PREPARATION_CV_UNITS",
    "prepare_mvpa_pattern_row_groups",
]
