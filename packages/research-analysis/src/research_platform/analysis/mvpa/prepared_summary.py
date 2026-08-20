from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
import math
from typing import Any

from .prepared_distances import PreparedMvpaDistanceResult, PreparedMvpaDistanceRow
from .summary import distance_summary_rows_from_long_rows


DEFAULT_PREPARED_MVPA_DISTANCE_SUMMARY_GROUP_BY = (
    "group_id",
    "condition_id_a",
    "condition_id_b",
    "metric",
    "engine_name",
    "normalization_method",
)
_PREPARED_DISTANCE_TOP_LEVEL_FIELDS = frozenset(
    {
        "group_id",
        "group_key",
        "condition_id_a",
        "condition_id_b",
        "condition_pair_id",
        "distance",
        "metric",
        "engine_name",
        "normalization_method",
        "cv_unit_count",
        "feature_count",
        "observation_count",
        "context",
    }
)
_MIXED_METADATA_FIELDS = ("metric", "engine_name", "normalization_method")


@dataclass(frozen=True)
class PreparedMvpaDistanceSummaryQcRow:
    """One JSON-safe QC row from prepared MVPA distance summarization."""

    level: str
    status: str
    code: str
    message: str
    source: str = "prepared_summary"
    source_row_index: int | None = None
    group_id: str | None = None
    group_key: Mapping[str, Any] = field(default_factory=dict)
    condition_id_a: str | None = None
    condition_id_b: str | None = None
    field_name: str | None = None
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
class PreparedMvpaDistanceSummaryProvenanceRow:
    """One JSON-safe provenance key/value for prepared MVPA distance summaries."""

    key: str
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty_label(self.key, field_name="key"))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PreparedMvpaDistanceSummaryResult:
    """JSON-safe in-memory result for prepared MVPA distance summaries."""

    summary_rows: Sequence[Mapping[str, Any]]
    qc_rows: Sequence[PreparedMvpaDistanceSummaryQcRow]
    provenance: Sequence[PreparedMvpaDistanceSummaryProvenanceRow]
    warnings: Sequence[str]
    errors: Sequence[str]
    executed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary_rows", tuple(dict(row) for row in self.summary_rows))
        object.__setattr__(self, "qc_rows", tuple(self.qc_rows))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


def summarize_prepared_mvpa_distances(
    source: Any,
    *,
    group_by: Sequence[str] | None = None,
) -> PreparedMvpaDistanceSummaryResult:
    """Summarize in-memory prepared MVPA distance rows without computing distances."""

    resolved_group_by = _resolved_group_by(group_by)
    grouping_policy = _grouping_policy(group_by, resolved_group_by)
    raw_rows, source_qc_rows = _source_rows(source)
    qc_rows = list(source_qc_rows)
    normalized_rows: list[dict[str, Any]] = []
    invalid_distance_row_count = 0

    if not raw_rows and not any(qc_row.status == "failed" for qc_row in qc_rows):
        qc_rows.append(
            _qc_row(
                level="input",
                status="warning",
                code="empty_input",
                message="Prepared MVPA distance summary input contains no distance rows.",
            )
        )

    for source_row_index, raw_row in enumerate(raw_rows):
        normalized_row, row_qc_rows = _normalized_prepared_distance_row(
            raw_row,
            source_row_index=source_row_index,
            group_by=resolved_group_by,
        )
        qc_rows.extend(row_qc_rows)
        if normalized_row is None:
            invalid_distance_row_count += 1
            continue
        normalized_rows.append(normalized_row)

    qc_rows.extend(_mixed_metadata_qc_rows(normalized_rows, group_by=resolved_group_by))
    summary_rows = distance_summary_rows_from_long_rows(normalized_rows, group_by=resolved_group_by)
    warnings, errors = _messages_from_qc(qc_rows)
    provenance = _provenance_rows(
        input_row_count=len(raw_rows),
        valid_distance_row_count=len(normalized_rows),
        invalid_distance_row_count=invalid_distance_row_count,
        summary_row_count=len(summary_rows),
        qc_row_count=len(qc_rows),
        group_by=resolved_group_by,
        grouping_policy=grouping_policy,
    )
    return PreparedMvpaDistanceSummaryResult(
        summary_rows=tuple(summary_rows),
        qc_rows=tuple(qc_rows),
        provenance=provenance,
        warnings=warnings,
        errors=errors,
        executed=True,
    )


def prepared_mvpa_distance_summary_rows(
    source: Any,
    *,
    group_by: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return prepared MVPA distance summary rows, raising if QC failed."""

    result = summarize_prepared_mvpa_distances(source, group_by=group_by)
    failed_qc_rows = tuple(qc_row for qc_row in result.qc_rows if qc_row.status == "failed")
    if failed_qc_rows:
        codes = ", ".join(_unique(qc_row.code for qc_row in failed_qc_rows))
        raise ValueError(f"Prepared MVPA distance summary failed QC: {codes}.")
    return [dict(row) for row in result.summary_rows]


def _resolved_group_by(group_by: Sequence[str] | None) -> tuple[str, ...]:
    if group_by is None:
        return DEFAULT_PREPARED_MVPA_DISTANCE_SUMMARY_GROUP_BY
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


def _grouping_policy(group_by: Sequence[str] | None, resolved_group_by: Sequence[str]) -> str:
    if group_by is None:
        return "default_non_collapsing"
    if not resolved_group_by:
        return "global"
    return "explicit"


def _source_rows(source: Any) -> tuple[tuple[Any, ...], tuple[PreparedMvpaDistanceSummaryQcRow, ...]]:
    if isinstance(source, PreparedMvpaDistanceResult):
        return tuple(source.distances), ()
    if isinstance(source, PreparedMvpaDistanceRow):
        return (source,), ()
    if isinstance(source, Mapping):
        if "distances" in source:
            return _rows_from_result_mapping(source)
        return (source,), ()
    if isinstance(source, (str, bytes)):
        return (), (
            _qc_row(
                level="input",
                status="failed",
                code="unsupported_row_shape",
                message="Prepared MVPA distance summary source must be a result, row, mapping, or iterable rows.",
            ),
        )
    if isinstance(source, Iterable):
        return tuple(source), ()
    return (), (
        _qc_row(
            level="input",
            status="failed",
            code="unsupported_row_shape",
            message="Prepared MVPA distance summary source must be a result, row, mapping, or iterable rows.",
        ),
    )


def _rows_from_result_mapping(
    source: Mapping[str, Any],
) -> tuple[tuple[Any, ...], tuple[PreparedMvpaDistanceSummaryQcRow, ...]]:
    distances = source.get("distances")
    if isinstance(distances, Mapping) or isinstance(distances, PreparedMvpaDistanceRow):
        return (distances,), ()
    if isinstance(distances, (str, bytes)) or not isinstance(distances, Iterable):
        return (), (
            _qc_row(
                level="input",
                status="failed",
                code="unsupported_row_shape",
                message="Prepared MVPA distance result mapping field 'distances' must be iterable rows.",
                field_name="distances",
            ),
        )
    return tuple(distances), ()


def _normalized_prepared_distance_row(
    row: Any,
    *,
    source_row_index: int,
    group_by: Sequence[str],
) -> tuple[dict[str, Any] | None, tuple[PreparedMvpaDistanceSummaryQcRow, ...]]:
    if isinstance(row, PreparedMvpaDistanceRow):
        row_mapping = row.to_dict()
    elif isinstance(row, Mapping):
        row_mapping = dict(row)
    else:
        return None, (
            _qc_row(
                level="row",
                status="failed",
                code="unsupported_row_shape",
                message=f"Prepared MVPA distance row {source_row_index} must be a mapping or PreparedMvpaDistanceRow.",
                source_row_index=source_row_index,
            ),
        )

    group_key = row_mapping.get("group_key", {})
    if "group_key" in row_mapping and not isinstance(group_key, Mapping):
        return None, (
            _qc_row(
                level="row",
                status="failed",
                code="invalid_group_key",
                message=f"Prepared MVPA distance row {source_row_index} has a non-mapping group_key.",
                source_row_index=source_row_index,
                group_id=_optional_label(row_mapping.get("group_id")),
                condition_id_a=_optional_label(row_mapping.get("condition_id_a")),
                condition_id_b=_optional_label(row_mapping.get("condition_id_b")),
            ),
        )

    normalized_row = _flattened_row(row_mapping, group_key=group_key)
    if "distance" not in normalized_row or normalized_row["distance"] is None:
        return None, (
            _qc_row(
                level="row",
                status="failed",
                code="missing_distance",
                message=f"Prepared MVPA distance row {source_row_index} is missing required field 'distance'.",
                source_row_index=source_row_index,
                group_id=_optional_label(normalized_row.get("group_id")),
                group_key=_qc_group_key(group_key),
                condition_id_a=_optional_label(normalized_row.get("condition_id_a")),
                condition_id_b=_optional_label(normalized_row.get("condition_id_b")),
                field_name="distance",
            ),
        )

    distance, distance_qc_row = _validated_distance(
        normalized_row["distance"],
        source_row_index=source_row_index,
        normalized_row=normalized_row,
        group_key=group_key,
    )
    if distance_qc_row is not None:
        return None, (distance_qc_row,)
    normalized_row["distance"] = distance

    missing_group_fields = tuple(field_name for field_name in group_by if field_name not in normalized_row)
    if missing_group_fields:
        return None, tuple(
            _qc_row(
                level="row",
                status="failed",
                code="missing_group_field",
                message=(
                    f"Prepared MVPA distance row {source_row_index} is missing requested "
                    f"group field {field_name!r}."
                ),
                source_row_index=source_row_index,
                group_id=_optional_label(normalized_row.get("group_id")),
                group_key=_qc_group_key(group_key),
                condition_id_a=_optional_label(normalized_row.get("condition_id_a")),
                condition_id_b=_optional_label(normalized_row.get("condition_id_b")),
                field_name=field_name,
            )
            for field_name in missing_group_fields
        )

    return normalized_row, ()


def _flattened_row(row: Mapping[str, Any], *, group_key: Mapping[Any, Any]) -> dict[str, Any]:
    normalized_row: dict[str, Any] = {}
    for key, value in group_key.items():
        if isinstance(key, str) and key not in _PREPARED_DISTANCE_TOP_LEVEL_FIELDS:
            normalized_row[key] = value
    for key, value in row.items():
        if key == "group_key":
            continue
        normalized_row[str(key)] = value
    return normalized_row


def _validated_distance(
    value: Any,
    *,
    source_row_index: int,
    normalized_row: Mapping[str, Any],
    group_key: Mapping[Any, Any],
) -> tuple[float, PreparedMvpaDistanceSummaryQcRow | None]:
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        return 0.0, _invalid_distance_qc_row(
            source_row_index=source_row_index,
            normalized_row=normalized_row,
            group_key=group_key,
        )
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return 0.0, _invalid_distance_qc_row(
            source_row_index=source_row_index,
            normalized_row=normalized_row,
            group_key=group_key,
        )
    if not math.isfinite(distance):
        return 0.0, _invalid_distance_qc_row(
            source_row_index=source_row_index,
            normalized_row=normalized_row,
            group_key=group_key,
        )
    return distance, None


def _invalid_distance_qc_row(
    *,
    source_row_index: int,
    normalized_row: Mapping[str, Any],
    group_key: Mapping[Any, Any],
) -> PreparedMvpaDistanceSummaryQcRow:
    return _qc_row(
        level="row",
        status="failed",
        code="invalid_distance",
        message=f"Prepared MVPA distance row {source_row_index} has a non-numeric or non-finite distance.",
        source_row_index=source_row_index,
        group_id=_optional_label(normalized_row.get("group_id")),
        group_key=_qc_group_key(group_key),
        condition_id_a=_optional_label(normalized_row.get("condition_id_a")),
        condition_id_b=_optional_label(normalized_row.get("condition_id_b")),
        field_name="distance",
    )


def _mixed_metadata_qc_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str],
) -> tuple[PreparedMvpaDistanceSummaryQcRow, ...]:
    qc_rows: list[PreparedMvpaDistanceSummaryQcRow] = []
    grouped_fields = set(group_by)
    for field_name in _MIXED_METADATA_FIELDS:
        if field_name in grouped_fields:
            continue
        values = _distinct_present_values(rows, field_name=field_name)
        if len(values) <= 1:
            continue
        qc_rows.append(
            _qc_row(
                level="summary",
                status="warning",
                code="mixed_metadata_not_grouped",
                message=(
                    f"Prepared MVPA distance rows mix {field_name!r} values, but "
                    "that field is not included in group_by."
                ),
                field_name=field_name,
                context={"distinct_value_count": len(values)},
            )
        )
    return tuple(qc_rows)


def _distinct_present_values(rows: Sequence[Mapping[str, Any]], *, field_name: str) -> tuple[Any, ...]:
    values: list[Any] = []
    for row in rows:
        if field_name not in row:
            continue
        value = row[field_name]
        if not any(value == existing for existing in values):
            values.append(value)
    return tuple(values)


def _provenance_rows(
    *,
    input_row_count: int,
    valid_distance_row_count: int,
    invalid_distance_row_count: int,
    summary_row_count: int,
    qc_row_count: int,
    group_by: Sequence[str],
    grouping_policy: str,
) -> tuple[PreparedMvpaDistanceSummaryProvenanceRow, ...]:
    return (
        PreparedMvpaDistanceSummaryProvenanceRow("source", "research_platform.analysis.mvpa.prepared_summary"),
        PreparedMvpaDistanceSummaryProvenanceRow("phase", "3D.1"),
        PreparedMvpaDistanceSummaryProvenanceRow("input_row_count", input_row_count),
        PreparedMvpaDistanceSummaryProvenanceRow("valid_distance_row_count", valid_distance_row_count),
        PreparedMvpaDistanceSummaryProvenanceRow("invalid_distance_row_count", invalid_distance_row_count),
        PreparedMvpaDistanceSummaryProvenanceRow("summary_row_count", summary_row_count),
        PreparedMvpaDistanceSummaryProvenanceRow("qc_row_count", qc_row_count),
        PreparedMvpaDistanceSummaryProvenanceRow("group_by", tuple(group_by)),
        PreparedMvpaDistanceSummaryProvenanceRow("grouping_policy", grouping_policy),
        PreparedMvpaDistanceSummaryProvenanceRow("sensitivity_policy", "not_run"),
        PreparedMvpaDistanceSummaryProvenanceRow("output_written", False),
    )


def _qc_row(
    *,
    level: str,
    status: str,
    code: str,
    message: str,
    source_row_index: int | None = None,
    group_id: str | None = None,
    group_key: Mapping[str, Any] | None = None,
    condition_id_a: str | None = None,
    condition_id_b: str | None = None,
    field_name: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> PreparedMvpaDistanceSummaryQcRow:
    return PreparedMvpaDistanceSummaryQcRow(
        level=level,
        status=status,
        code=code,
        message=message,
        source="prepared_summary",
        source_row_index=source_row_index,
        group_id=group_id,
        group_key=group_key or {},
        condition_id_a=condition_id_a,
        condition_id_b=condition_id_b,
        field_name=field_name,
        context=context or {},
    )


def _qc_group_key(group_key: Mapping[Any, Any]) -> dict[str, Any]:
    return {key: value for key, value in group_key.items() if isinstance(key, str)}


def _messages_from_qc(
    qc_rows: Sequence[PreparedMvpaDistanceSummaryQcRow],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    errors: list[str] = []
    for qc_row in qc_rows:
        if qc_row.status == "failed":
            errors.append(qc_row.message)
        else:
            warnings.append(qc_row.message)
    return _unique(warnings), _unique(errors)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for value in values:
        ordered[value] = None
    return tuple(ordered)


def _non_empty_label(value: object, *, field_name: str) -> str:
    label = str(value).strip()
    if not label:
        raise ValueError(f"{field_name} must be a non-empty value.")
    return label


def _optional_label(value: object) -> str | None:
    if value is None:
        return None
    label = str(value).strip()
    return label or None


def _json_safe_dataclass(instance: object) -> dict[str, Any]:
    if not is_dataclass(instance):
        raise TypeError("_json_safe_dataclass requires a dataclass instance.")
    return {field_.name: _json_safe(getattr(instance, field_.name)) for field_ in fields(instance)}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON-safe prepared MVPA distance summaries cannot contain non-finite floats.")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "DEFAULT_PREPARED_MVPA_DISTANCE_SUMMARY_GROUP_BY",
    "PreparedMvpaDistanceSummaryProvenanceRow",
    "PreparedMvpaDistanceSummaryQcRow",
    "PreparedMvpaDistanceSummaryResult",
    "prepared_mvpa_distance_summary_rows",
    "summarize_prepared_mvpa_distances",
]
