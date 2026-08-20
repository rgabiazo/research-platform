"""Generic subject-level inference summaries for rectangular table rows."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
import json
import math
import random
from statistics import NormalDist, median
from typing import Any


DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_EXACT_SIGN_FLIP_MAX_N = 20
DEFAULT_MONTE_CARLO_SIGN_FLIP_ITERATIONS = 10000
DEFAULT_BOOTSTRAP_ITERATIONS = 2000

_ALTERNATIVES = frozenset({"two_sided", "greater", "less"})
_CI_METHODS = frozenset({"normal_approximation", "bootstrap"})
_DUPLICATE_SUBJECT_POLICIES = frozenset({"fail", "mean"})


@dataclass(frozen=True)
class SubjectInferenceResultRow:
    """One JSON-safe generic subject-level inference summary row."""

    group_label: str | None = None
    effect_label: str | None = None
    measure: str | None = None
    group_key: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    n: int = 0
    mean: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_method: str | None = None
    confidence_level: float | None = None
    sd: float | None = None
    se: float | None = None
    median: float | None = None
    null_value: float | None = None
    t: float | None = None
    df: int | None = None
    p_method: str | None = None
    p_method_detail: str | None = None
    p_value: float | None = None
    p_alternative: str | None = None
    q_method: str | None = None
    q_value: float | None = None
    effect_size: float | None = None
    effect_size_type: str | None = None
    percent_positive: float | None = None
    loo_min: float | None = None
    loo_max: float | None = None
    status: str = "ok"
    warnings: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_label", _optional_label(self.group_label))
        object.__setattr__(self, "effect_label", _optional_label(self.effect_label))
        object.__setattr__(self, "measure", _optional_label(self.measure))
        object.__setattr__(self, "group_key", _json_safe_mapping(self.group_key))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))
        object.__setattr__(self, "status", _non_empty_label(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)

    def to_tsv_row(self) -> dict[str, str]:
        return _tsv_safe_dataclass(self)


@dataclass(frozen=True)
class SubjectInferenceMultiplicityRow:
    """One JSON-safe multiple-comparison correction row."""

    family_id: str
    family_key: Mapping[str, Any] = field(default_factory=dict)
    group_label: str | None = None
    effect_label: str | None = None
    measure: str | None = None
    group_key: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    p_method: str | None = None
    p_value: float | None = None
    q_method: str = "benjamini_hochberg"
    q_value: float | None = None
    rank: int | None = None
    family_size: int | None = None
    status: str = "ok"
    warnings: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _non_empty_label(self.family_id, field_name="family_id"))
        object.__setattr__(self, "family_key", _json_safe_mapping(self.family_key))
        object.__setattr__(self, "group_label", _optional_label(self.group_label))
        object.__setattr__(self, "effect_label", _optional_label(self.effect_label))
        object.__setattr__(self, "measure", _optional_label(self.measure))
        object.__setattr__(self, "group_key", _json_safe_mapping(self.group_key))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))
        object.__setattr__(self, "q_method", _non_empty_label(self.q_method, field_name="q_method"))
        object.__setattr__(self, "status", _non_empty_label(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)

    def to_tsv_row(self) -> dict[str, str]:
        return _tsv_safe_dataclass(self)


@dataclass(frozen=True)
class SubjectInferenceLosoRow:
    """One JSON-safe leave-one-subject-out sensitivity row."""

    left_out_subject: str
    group_label: str | None = None
    effect_label: str | None = None
    measure: str | None = None
    group_key: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    n: int = 0
    mean: float | None = None
    sd: float | None = None
    se: float | None = None
    median: float | None = None
    null_value: float | None = None
    status: str = "ok"
    warnings: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_out_subject", _non_empty_label(self.left_out_subject, field_name="left_out_subject"))
        object.__setattr__(self, "group_label", _optional_label(self.group_label))
        object.__setattr__(self, "effect_label", _optional_label(self.effect_label))
        object.__setattr__(self, "measure", _optional_label(self.measure))
        object.__setattr__(self, "group_key", _json_safe_mapping(self.group_key))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))
        object.__setattr__(self, "status", _non_empty_label(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)

    def to_tsv_row(self) -> dict[str, str]:
        return _tsv_safe_dataclass(self)


@dataclass(frozen=True)
class SubjectInferenceMissingnessRow:
    """One JSON-safe expected-subject missingness row."""

    subject_id: str
    group_label: str | None = None
    effect_label: str | None = None
    measure: str | None = None
    group_key: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "missing"
    warnings: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _non_empty_label(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "group_label", _optional_label(self.group_label))
        object.__setattr__(self, "effect_label", _optional_label(self.effect_label))
        object.__setattr__(self, "measure", _optional_label(self.measure))
        object.__setattr__(self, "group_key", _json_safe_mapping(self.group_key))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))
        object.__setattr__(self, "status", _non_empty_label(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)

    def to_tsv_row(self) -> dict[str, str]:
        return _tsv_safe_dataclass(self)


@dataclass(frozen=True)
class SubjectInferenceQcRow:
    """One JSON-safe subject-level inference QC row."""

    level: str
    status: str
    code: str
    message: str
    source: str = "subject_inference"
    source_row_index: int | None = None
    subject_id: str | None = None
    field_name: str | None = None
    group_label: str | None = None
    effect_label: str | None = None
    measure: str | None = None
    group_key: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", _non_empty_label(self.level, field_name="level"))
        object.__setattr__(self, "status", _non_empty_label(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_label(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_label(self.message, field_name="message"))
        object.__setattr__(self, "source", _non_empty_label(self.source, field_name="source"))
        object.__setattr__(self, "subject_id", _optional_label(self.subject_id))
        object.__setattr__(self, "field_name", _optional_label(self.field_name))
        object.__setattr__(self, "group_label", _optional_label(self.group_label))
        object.__setattr__(self, "effect_label", _optional_label(self.effect_label))
        object.__setattr__(self, "measure", _optional_label(self.measure))
        object.__setattr__(self, "group_key", _json_safe_mapping(self.group_key))
        object.__setattr__(self, "context", _json_safe_mapping(self.context))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)

    def to_tsv_row(self) -> dict[str, str]:
        return _tsv_safe_dataclass(self)


@dataclass(frozen=True)
class SubjectInferenceProvenanceRow:
    """One JSON-safe subject-level inference provenance key/value row."""

    key: str
    value: Any
    source: str = "subject_inference"

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty_label(self.key, field_name="key"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "source", _non_empty_label(self.source, field_name="source"))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)

    def to_tsv_row(self) -> dict[str, str]:
        return _tsv_safe_dataclass(self)


@dataclass(frozen=True)
class SubjectInferenceSummaryResult:
    """JSON-safe in-memory result for generic subject-level inference."""

    summary_rows: Sequence[SubjectInferenceResultRow]
    multiplicity_rows: Sequence[SubjectInferenceMultiplicityRow]
    loso_rows: Sequence[SubjectInferenceLosoRow]
    missingness_rows: Sequence[SubjectInferenceMissingnessRow]
    qc_rows: Sequence[SubjectInferenceQcRow]
    provenance_rows: Sequence[SubjectInferenceProvenanceRow]
    warnings: Sequence[str]
    errors: Sequence[str]
    executed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary_rows", tuple(self.summary_rows))
        object.__setattr__(self, "multiplicity_rows", tuple(self.multiplicity_rows))
        object.__setattr__(self, "loso_rows", tuple(self.loso_rows))
        object.__setattr__(self, "missingness_rows", tuple(self.missingness_rows))
        object.__setattr__(self, "qc_rows", tuple(self.qc_rows))
        object.__setattr__(self, "provenance_rows", tuple(self.provenance_rows))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class _InputRow:
    source_row_index: int
    subject_id: str
    value: float
    null_value: float
    source_row: Mapping[str, Any]


@dataclass
class _GroupBucket:
    group_key: dict[str, Any]
    rows: list[_InputRow] = field(default_factory=list)


@dataclass(frozen=True)
class _SubjectValue:
    subject_id: str
    value: float


def summarize_subject_level_inference(
    rows: Iterable[Mapping[str, Any]],
    *,
    subject_column: str,
    value_column: str,
    group_columns: Sequence[str] = (),
    metadata_columns: Sequence[str] = (),
    group_label_column: str | None = None,
    effect_label_column: str | None = None,
    measure_column: str | None = None,
    measure: str | None = None,
    null_value: float = 0.0,
    null_value_column: str | None = None,
    expected_subjects: Iterable[Any] | None = None,
    duplicate_subject_policy: str = "fail",
    min_n: int = 2,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ci_method: str = "normal_approximation",
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    alternative: str = "two_sided",
    exact_max_n: int = DEFAULT_EXACT_SIGN_FLIP_MAX_N,
    sign_flip_iterations: int = DEFAULT_MONTE_CARLO_SIGN_FLIP_ITERATIONS,
    seed: int = 0,
    fdr_family_columns: Sequence[str] | None = None,
    provenance_rows: Iterable[Mapping[str, Any] | SubjectInferenceProvenanceRow] | None = None,
) -> SubjectInferenceSummaryResult:
    """Summarize one subject-level value per group against a finite null value.

    The implementation is intentionally generic: grouping, labels, metadata,
    measures, and multiplicity families are caller-configured table columns.
    """

    resolved_subject_column = _required_column_name(subject_column, field_name="subject_column")
    resolved_value_column = _required_column_name(value_column, field_name="value_column")
    resolved_group_columns = _resolved_columns(group_columns, field_name="group_columns")
    resolved_metadata_columns = _resolved_columns(metadata_columns, field_name="metadata_columns")
    resolved_group_label_column = _optional_column_name(group_label_column, field_name="group_label_column")
    resolved_effect_label_column = _optional_column_name(effect_label_column, field_name="effect_label_column")
    resolved_measure_column = _optional_column_name(measure_column, field_name="measure_column")
    resolved_null_value_column = _optional_column_name(null_value_column, field_name="null_value_column")
    resolved_fdr_family_columns = (
        None if fdr_family_columns is None else _resolved_columns(fdr_family_columns, field_name="fdr_family_columns")
    )
    resolved_expected_subjects = _expected_subject_labels(expected_subjects)
    resolved_measure = _optional_label(measure) or (resolved_measure_column if resolved_measure_column else resolved_value_column)

    if duplicate_subject_policy not in _DUPLICATE_SUBJECT_POLICIES:
        raise ValueError("duplicate_subject_policy must be 'fail' or 'mean'.")
    if ci_method not in _CI_METHODS:
        raise ValueError("ci_method must be 'normal_approximation' or 'bootstrap'.")
    if alternative not in _ALTERNATIVES:
        raise ValueError("alternative must be 'two_sided', 'greater', or 'less'.")
    if min_n < 1:
        raise ValueError("min_n must be at least 1.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1.")
    if exact_max_n < 0:
        raise ValueError("exact_max_n must be non-negative.")
    if sign_flip_iterations < 1:
        raise ValueError("sign_flip_iterations must be at least 1.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    default_null_value = _finite_float(null_value, field_name="null_value")
    if isinstance(rows, (str, bytes)) or isinstance(rows, Mapping):
        raise ValueError("rows must be an iterable of mapping rows, not a single row or string.")

    qc_rows: list[SubjectInferenceQcRow] = []
    buckets: OrderedDict[tuple[str, ...], _GroupBucket] = OrderedDict()
    input_row_count = 0
    valid_row_count = 0

    for source_row_index, row in enumerate(rows):
        input_row_count += 1
        input_row, row_group_key, row_qc_rows = _validated_input_row(
            row,
            source_row_index=source_row_index,
            subject_column=resolved_subject_column,
            value_column=resolved_value_column,
            group_columns=resolved_group_columns,
            null_value=default_null_value,
            null_value_column=resolved_null_value_column,
        )
        qc_rows.extend(row_qc_rows)
        if input_row is None or row_group_key is None:
            continue
        valid_row_count += 1
        group_token = _group_token(row_group_key, resolved_group_columns)
        if group_token not in buckets:
            buckets[group_token] = _GroupBucket(group_key=row_group_key)
        buckets[group_token].rows.append(input_row)

    summary_rows: list[SubjectInferenceResultRow] = []
    loso_rows: list[SubjectInferenceLosoRow] = []
    missingness_rows: list[SubjectInferenceMissingnessRow] = []

    for _, bucket in sorted(buckets.items(), key=lambda item: item[0]):
        group_qc_rows, group_summary_rows, group_loso_rows, group_missingness_rows = _summarize_group(
            bucket,
            subject_column=resolved_subject_column,
            metadata_columns=resolved_metadata_columns,
            group_label_column=resolved_group_label_column,
            effect_label_column=resolved_effect_label_column,
            measure_column=resolved_measure_column,
            default_measure=resolved_measure,
            expected_subjects=resolved_expected_subjects,
            duplicate_subject_policy=duplicate_subject_policy,
            min_n=min_n,
            confidence_level=confidence_level,
            ci_method=ci_method,
            bootstrap_iterations=bootstrap_iterations,
            alternative=alternative,
            exact_max_n=exact_max_n,
            sign_flip_iterations=sign_flip_iterations,
            seed=seed,
        )
        qc_rows.extend(group_qc_rows)
        summary_rows.extend(group_summary_rows)
        loso_rows.extend(group_loso_rows)
        missingness_rows.extend(group_missingness_rows)

    if input_row_count == 0:
        qc_rows.append(
            _qc_row(
                level="input",
                status="warning",
                code="empty_input",
                message="Subject-level inference input contains no rows.",
            )
        )

    summary_rows, multiplicity_rows = _apply_bh_fdr(summary_rows, family_columns=resolved_fdr_family_columns)
    warnings, errors = _messages_from_qc(qc_rows)
    result_provenance_rows = _provenance_rows(
        input_row_count=input_row_count,
        valid_row_count=valid_row_count,
        invalid_row_count=input_row_count - valid_row_count,
        summary_row_count=len(summary_rows),
        multiplicity_row_count=len(multiplicity_rows),
        loso_row_count=len(loso_rows),
        missingness_row_count=len(missingness_rows),
        qc_row_count=len(qc_rows),
        subject_column=resolved_subject_column,
        value_column=resolved_value_column,
        group_columns=resolved_group_columns,
        metadata_columns=resolved_metadata_columns,
        null_value=default_null_value,
        null_value_column=resolved_null_value_column,
        expected_subject_count=len(resolved_expected_subjects) if resolved_expected_subjects is not None else None,
        duplicate_subject_policy=duplicate_subject_policy,
        min_n=min_n,
        confidence_level=confidence_level,
        ci_method=ci_method,
        bootstrap_iterations=bootstrap_iterations,
        alternative=alternative,
        exact_max_n=exact_max_n,
        sign_flip_iterations=sign_flip_iterations,
        seed=seed,
        fdr_family_columns=resolved_fdr_family_columns,
    )
    result_provenance_rows.extend(_coerced_provenance_rows(provenance_rows))

    return SubjectInferenceSummaryResult(
        summary_rows=tuple(summary_rows),
        multiplicity_rows=tuple(multiplicity_rows),
        loso_rows=tuple(loso_rows),
        missingness_rows=tuple(missingness_rows),
        qc_rows=tuple(qc_rows),
        provenance_rows=tuple(result_provenance_rows),
        warnings=warnings,
        errors=errors,
        executed=True,
    )


def _validated_input_row(
    row: Any,
    *,
    source_row_index: int,
    subject_column: str,
    value_column: str,
    group_columns: Sequence[str],
    null_value: float,
    null_value_column: str | None,
) -> tuple[_InputRow | None, dict[str, Any] | None, tuple[SubjectInferenceQcRow, ...]]:
    if not isinstance(row, Mapping):
        return None, None, (
            _qc_row(
                level="row",
                status="failed",
                code="unsupported_row_shape",
                message=f"Subject-level inference row {source_row_index} must be a mapping.",
                source_row_index=source_row_index,
            ),
        )

    row_qc_rows: list[SubjectInferenceQcRow] = []
    group_key: dict[str, Any] = {}
    for column in group_columns:
        if column not in row:
            row_qc_rows.append(
                _qc_row(
                    level="row",
                    status="failed",
                    code="missing_group_field",
                    message=f"Subject-level inference row {source_row_index} is missing requested group field {column!r}.",
                    source_row_index=source_row_index,
                    field_name=column,
                )
            )
            continue
        try:
            group_key[column] = _json_safe(row[column])
        except ValueError:
            row_qc_rows.append(
                _qc_row(
                    level="row",
                    status="failed",
                    code="invalid_group_field",
                    message=f"Subject-level inference row {source_row_index} has a non-finite group field {column!r}.",
                    source_row_index=source_row_index,
                    field_name=column,
                )
            )

    subject_id = _subject_label(row.get(subject_column))
    if subject_id is None:
        row_qc_rows.append(
            _qc_row(
                level="row",
                status="failed",
                code="missing_subject",
                message=f"Subject-level inference row {source_row_index} is missing subject field {subject_column!r}.",
                source_row_index=source_row_index,
                field_name=subject_column,
                group_key=group_key,
            )
        )

    if value_column not in row:
        value = None
        row_qc_rows.append(
            _qc_row(
                level="row",
                status="failed",
                code="missing_value",
                message=f"Subject-level inference row {source_row_index} is missing value field {value_column!r}.",
                source_row_index=source_row_index,
                subject_id=subject_id,
                field_name=value_column,
                group_key=group_key,
            )
        )
    else:
        value = _coerce_finite_float(row[value_column])
        if value is None:
            row_qc_rows.append(
                _qc_row(
                    level="row",
                    status="failed",
                    code="invalid_value",
                    message=(
                        f"Subject-level inference row {source_row_index} has a non-numeric, bool, "
                        f"or non-finite value in {value_column!r}."
                    ),
                    source_row_index=source_row_index,
                    subject_id=subject_id,
                    field_name=value_column,
                    group_key=group_key,
                )
            )

    row_null_value = null_value
    if null_value_column is not None:
        if null_value_column not in row:
            row_qc_rows.append(
                _qc_row(
                    level="row",
                    status="failed",
                    code="missing_null_value",
                    message=f"Subject-level inference row {source_row_index} is missing null field {null_value_column!r}.",
                    source_row_index=source_row_index,
                    subject_id=subject_id,
                    field_name=null_value_column,
                    group_key=group_key,
                )
            )
        else:
            parsed_null_value = _coerce_finite_float(row[null_value_column])
            if parsed_null_value is None:
                row_qc_rows.append(
                    _qc_row(
                        level="row",
                        status="failed",
                        code="invalid_null_value",
                        message=(
                            f"Subject-level inference row {source_row_index} has a non-numeric, bool, "
                            f"or non-finite null value in {null_value_column!r}."
                        ),
                        source_row_index=source_row_index,
                        subject_id=subject_id,
                        field_name=null_value_column,
                        group_key=group_key,
                    )
                )
            else:
                row_null_value = parsed_null_value

    if row_qc_rows:
        return None, None, tuple(row_qc_rows)
    assert subject_id is not None
    assert value is not None
    return (
        _InputRow(
            source_row_index=source_row_index,
            subject_id=subject_id,
            value=value,
            null_value=row_null_value,
            source_row=dict(row),
        ),
        group_key,
        (),
    )


def _summarize_group(
    bucket: _GroupBucket,
    *,
    subject_column: str,
    metadata_columns: Sequence[str],
    group_label_column: str | None,
    effect_label_column: str | None,
    measure_column: str | None,
    default_measure: str | None,
    expected_subjects: tuple[str, ...] | None,
    duplicate_subject_policy: str,
    min_n: int,
    confidence_level: float,
    ci_method: str,
    bootstrap_iterations: int,
    alternative: str,
    exact_max_n: int,
    sign_flip_iterations: int,
    seed: int,
) -> tuple[
    tuple[SubjectInferenceQcRow, ...],
    tuple[SubjectInferenceResultRow, ...],
    tuple[SubjectInferenceLosoRow, ...],
    tuple[SubjectInferenceMissingnessRow, ...],
]:
    qc_rows: list[SubjectInferenceQcRow] = []
    metadata, metadata_qc_rows = _constant_metadata(bucket, metadata_columns=metadata_columns)
    qc_rows.extend(metadata_qc_rows)
    group_label = _label_from_column(
        bucket,
        metadata=metadata,
        column=group_label_column,
        fallback=_first_group_value(bucket.group_key),
    )
    effect_label = _label_from_column(
        bucket,
        metadata=metadata,
        column=effect_label_column,
        fallback=_nth_group_value(bucket.group_key, 1),
    )
    measure = _label_from_column(bucket, metadata=metadata, column=measure_column, fallback=default_measure)

    missingness_rows = _missingness_rows(
        bucket,
        expected_subjects=expected_subjects,
        group_label=group_label,
        effect_label=effect_label,
        measure=measure,
        metadata=metadata,
    )

    null_values = _unique_floats(row.null_value for row in bucket.rows)
    if len(null_values) != 1:
        qc_rows.append(
            _qc_row(
                level="group",
                status="failed",
                code="nonconstant_null_value",
                message="Subject-level inference null value column must be finite and constant within each group.",
                group_label=group_label,
                effect_label=effect_label,
                measure=measure,
                group_key=bucket.group_key,
                context={"null_values": null_values},
            )
        )
        return (
            tuple(qc_rows),
            (
                _failed_summary_row(
                    bucket,
                    group_label=group_label,
                    effect_label=effect_label,
                    measure=measure,
                    metadata=metadata,
                    n=0,
                    null_value=None,
                    warnings=("nonconstant_null_value",),
                ),
            ),
            (),
            missingness_rows,
        )
    group_null_value = null_values[0]

    rows_by_subject: dict[str, list[_InputRow]] = defaultdict(list)
    for row in bucket.rows:
        rows_by_subject[row.subject_id].append(row)
    duplicate_subjects = tuple(subject_id for subject_id, subject_rows in rows_by_subject.items() if len(subject_rows) > 1)

    group_warning_codes: list[str] = []
    if duplicate_subjects and duplicate_subject_policy == "fail":
        for subject_id in duplicate_subjects:
            qc_rows.append(
                _qc_row(
                    level="group",
                    status="failed",
                    code="duplicate_subject",
                    message=f"Subject {subject_id!r} has duplicate rows within a subject-level inference group.",
                    subject_id=subject_id,
                    field_name=subject_column,
                    group_label=group_label,
                    effect_label=effect_label,
                    measure=measure,
                    group_key=bucket.group_key,
                    context={"row_count": len(rows_by_subject[subject_id])},
                )
            )
        return (
            tuple(qc_rows),
            (
                _failed_summary_row(
                    bucket,
                    group_label=group_label,
                    effect_label=effect_label,
                    measure=measure,
                    metadata=metadata,
                    n=len(rows_by_subject),
                    null_value=group_null_value,
                    warnings=("duplicate_subject",),
                ),
            ),
            (),
            missingness_rows,
        )

    subject_values: list[_SubjectValue] = []
    for subject_id, subject_rows in sorted(rows_by_subject.items()):
        if len(subject_rows) > 1:
            group_warning_codes.append("duplicate_subject_aggregated")
            qc_rows.append(
                _qc_row(
                    level="group",
                    status="warning",
                    code="duplicate_subject_aggregated",
                    message=f"Subject {subject_id!r} duplicate rows were aggregated by mean.",
                    subject_id=subject_id,
                    field_name=subject_column,
                    group_label=group_label,
                    effect_label=effect_label,
                    measure=measure,
                    group_key=bucket.group_key,
                    context={"row_count": len(subject_rows)},
                )
            )
        subject_values.append(_SubjectValue(subject_id=subject_id, value=_mean([row.value for row in subject_rows])))

    if len(subject_values) < min_n:
        qc_rows.append(
            _qc_row(
                level="group",
                status="failed",
                code="minimum_n_not_met",
                message=(
                    f"Subject-level inference group has N={len(subject_values)}, "
                    f"below the configured minimum N={min_n}."
                ),
                group_label=group_label,
                effect_label=effect_label,
                measure=measure,
                group_key=bucket.group_key,
                context={"n": len(subject_values), "min_n": min_n},
            )
        )
        return (
            tuple(qc_rows),
            (
                _failed_summary_row(
                    bucket,
                    group_label=group_label,
                    effect_label=effect_label,
                    measure=measure,
                    metadata=metadata,
                    n=len(subject_values),
                    null_value=group_null_value,
                    warnings=(*_unique(group_warning_codes), "minimum_n_not_met"),
                ),
            ),
            (),
            missingness_rows,
        )

    summary_row, loso_rows, stats_warning_codes = _valid_summary_rows(
        bucket,
        subject_values=subject_values,
        null_value=group_null_value,
        group_label=group_label,
        effect_label=effect_label,
        measure=measure,
        metadata=metadata,
        confidence_level=confidence_level,
        ci_method=ci_method,
        bootstrap_iterations=bootstrap_iterations,
        alternative=alternative,
        exact_max_n=exact_max_n,
        sign_flip_iterations=sign_flip_iterations,
        seed=seed,
    )
    group_warning_codes.extend(stats_warning_codes)
    if group_warning_codes:
        summary_row = replace(summary_row, status="warning", warnings=_unique(group_warning_codes))

    return tuple(qc_rows), (summary_row,), tuple(loso_rows), missingness_rows


def _valid_summary_rows(
    bucket: _GroupBucket,
    *,
    subject_values: Sequence[_SubjectValue],
    null_value: float,
    group_label: str | None,
    effect_label: str | None,
    measure: str | None,
    metadata: Mapping[str, Any],
    confidence_level: float,
    ci_method: str,
    bootstrap_iterations: int,
    alternative: str,
    exact_max_n: int,
    sign_flip_iterations: int,
    seed: int,
) -> tuple[SubjectInferenceResultRow, tuple[SubjectInferenceLosoRow, ...], tuple[str, ...]]:
    values = [subject_value.value for subject_value in subject_values]
    differences = [value - null_value for value in values]
    n = len(values)
    mean_value = _mean(values)
    sd = _sample_sd(values)
    se = sd / math.sqrt(n)
    median_value = float(median(values))
    if ci_method == "normal_approximation":
        ci_low, ci_high = _normal_mean_ci(mean_value, se=se, confidence_level=confidence_level)
    else:
        ci_low, ci_high = _bootstrap_mean_ci(
            values,
            confidence_level=confidence_level,
            iterations=bootstrap_iterations,
            seed=seed,
        )

    mean_difference = _mean(differences)
    warning_codes: list[str] = []
    t_value: float | None
    effect_size: float | None
    if sd == 0.0:
        t_value = None
        effect_size = None
        warning_codes.append("zero_variance")
    else:
        t_value = mean_difference / se
        effect_size = mean_difference / sd

    p_value, p_method_detail = _sign_flip_p_value(
        differences,
        alternative=alternative,
        exact_max_n=exact_max_n,
        iterations=sign_flip_iterations,
        seed=seed,
    )
    loso_rows = _loso_rows(
        subject_values,
        null_value=null_value,
        group_label=group_label,
        effect_label=effect_label,
        measure=measure,
        group_key=bucket.group_key,
        metadata=metadata,
    )
    loso_means = [row.mean for row in loso_rows if row.mean is not None]

    return (
        SubjectInferenceResultRow(
            group_label=group_label,
            effect_label=effect_label,
            measure=measure,
            group_key=bucket.group_key,
            metadata=metadata,
            n=n,
            mean=mean_value,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_method=ci_method,
            confidence_level=confidence_level,
            sd=sd,
            se=se,
            median=median_value,
            null_value=null_value,
            t=t_value,
            df=n - 1,
            p_method="sign_flip",
            p_method_detail=p_method_detail,
            p_value=p_value,
            p_alternative=alternative,
            effect_size=effect_size,
            effect_size_type="dz" if effect_size is not None else None,
            percent_positive=100.0 * sum(1 for difference in differences if difference > 0.0) / n,
            loo_min=min(loso_means) if loso_means else None,
            loo_max=max(loso_means) if loso_means else None,
            status="ok",
            warnings=(),
        ),
        tuple(loso_rows),
        tuple(warning_codes),
    )


def _failed_summary_row(
    bucket: _GroupBucket,
    *,
    group_label: str | None,
    effect_label: str | None,
    measure: str | None,
    metadata: Mapping[str, Any],
    n: int,
    null_value: float | None,
    warnings: Sequence[str],
) -> SubjectInferenceResultRow:
    return SubjectInferenceResultRow(
        group_label=group_label,
        effect_label=effect_label,
        measure=measure,
        group_key=bucket.group_key,
        metadata=metadata,
        n=n,
        null_value=null_value,
        status="failed",
        warnings=warnings,
    )


def _missingness_rows(
    bucket: _GroupBucket,
    *,
    expected_subjects: tuple[str, ...] | None,
    group_label: str | None,
    effect_label: str | None,
    measure: str | None,
    metadata: Mapping[str, Any],
) -> tuple[SubjectInferenceMissingnessRow, ...]:
    if expected_subjects is None:
        return ()
    observed_subjects = {row.subject_id for row in bucket.rows}
    return tuple(
        SubjectInferenceMissingnessRow(
            subject_id=subject_id,
            group_label=group_label,
            effect_label=effect_label,
            measure=measure,
            group_key=bucket.group_key,
            metadata=metadata,
        )
        for subject_id in expected_subjects
        if subject_id not in observed_subjects
    )


def _loso_rows(
    subject_values: Sequence[_SubjectValue],
    *,
    null_value: float,
    group_label: str | None,
    effect_label: str | None,
    measure: str | None,
    group_key: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[SubjectInferenceLosoRow, ...]:
    rows: list[SubjectInferenceLosoRow] = []
    for left_out in subject_values:
        retained = [subject_value.value for subject_value in subject_values if subject_value.subject_id != left_out.subject_id]
        if retained:
            retained_sd = _sample_sd(retained)
            retained_se = retained_sd / math.sqrt(len(retained)) if len(retained) > 1 else 0.0
            rows.append(
                SubjectInferenceLosoRow(
                    left_out_subject=left_out.subject_id,
                    group_label=group_label,
                    effect_label=effect_label,
                    measure=measure,
                    group_key=group_key,
                    metadata=metadata,
                    n=len(retained),
                    mean=_mean(retained),
                    sd=retained_sd,
                    se=retained_se,
                    median=float(median(retained)),
                    null_value=null_value,
                    status="ok",
                )
            )
        else:
            rows.append(
                SubjectInferenceLosoRow(
                    left_out_subject=left_out.subject_id,
                    group_label=group_label,
                    effect_label=effect_label,
                    measure=measure,
                    group_key=group_key,
                    metadata=metadata,
                    n=0,
                    null_value=null_value,
                    status="failed",
                    warnings=("empty_loso_sample",),
                )
            )
    return tuple(rows)


def _constant_metadata(
    bucket: _GroupBucket,
    *,
    metadata_columns: Sequence[str],
) -> tuple[dict[str, Any], tuple[SubjectInferenceQcRow, ...]]:
    metadata: dict[str, Any] = {}
    qc_rows: list[SubjectInferenceQcRow] = []
    for column in metadata_columns:
        values: OrderedDict[str, Any] = OrderedDict()
        invalid_count = 0
        for row in bucket.rows:
            if column not in row.source_row:
                continue
            try:
                value = _json_safe(row.source_row[column])
            except ValueError:
                invalid_count += 1
                continue
            values[_stable_token(value)] = value
        if invalid_count:
            qc_rows.append(
                _qc_row(
                    level="metadata",
                    status="warning",
                    code="invalid_metadata",
                    message=f"Metadata column {column!r} contains non-finite values that were not propagated.",
                    field_name=column,
                    group_key=bucket.group_key,
                    context={"invalid_value_count": invalid_count},
                )
            )
        if len(values) == 1:
            metadata[column] = next(iter(values.values()))
        elif len(values) > 1:
            metadata[column] = None
            qc_rows.append(
                _qc_row(
                    level="metadata",
                    status="warning",
                    code="mixed_metadata",
                    message=f"Metadata column {column!r} is not constant within a subject-level inference group.",
                    field_name=column,
                    group_key=bucket.group_key,
                    context={"distinct_value_count": len(values)},
                )
            )
        else:
            metadata[column] = None
    return metadata, tuple(qc_rows)


def _apply_bh_fdr(
    summary_rows: Sequence[SubjectInferenceResultRow],
    *,
    family_columns: Sequence[str] | None,
) -> tuple[list[SubjectInferenceResultRow], list[SubjectInferenceMultiplicityRow]]:
    if family_columns is None:
        return list(summary_rows), []

    families: OrderedDict[tuple[str, ...], list[tuple[int, SubjectInferenceResultRow]]] = OrderedDict()
    family_keys: dict[tuple[str, ...], dict[str, Any]] = {}
    for index, row in enumerate(summary_rows):
        if row.p_value is None or row.status == "failed":
            continue
        family_key = {column: _row_lookup(row, column) for column in family_columns}
        family_token = tuple(_stable_token(family_key[column]) for column in family_columns)
        families.setdefault(family_token, []).append((index, row))
        family_keys[family_token] = family_key

    adjusted_rows = list(summary_rows)
    multiplicity_rows: list[SubjectInferenceMultiplicityRow] = []
    for family_token, entries in families.items():
        sorted_entries = sorted(
            entries,
            key=lambda item: (item[1].p_value if item[1].p_value is not None else math.inf, _summary_sort_token(item[1]), item[0]),
        )
        q_values_by_index = _benjamini_hochberg([row.p_value for _, row in sorted_entries if row.p_value is not None])
        family_size = len(sorted_entries)
        family_key = family_keys[family_token]
        family_id = "all" if not family_key else _stable_token(family_key)
        for rank, ((index, row), q_value) in enumerate(zip(sorted_entries, q_values_by_index, strict=True), start=1):
            adjusted_rows[index] = replace(row, q_method="benjamini_hochberg", q_value=q_value)
            multiplicity_rows.append(
                SubjectInferenceMultiplicityRow(
                    family_id=family_id,
                    family_key=family_key,
                    group_label=row.group_label,
                    effect_label=row.effect_label,
                    measure=row.measure,
                    group_key=row.group_key,
                    metadata=row.metadata,
                    p_method=row.p_method,
                    p_value=row.p_value,
                    q_method="benjamini_hochberg",
                    q_value=q_value,
                    rank=rank,
                    family_size=family_size,
                )
            )
    return adjusted_rows, multiplicity_rows


def _benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    raw = [min(max(p_value * m / rank, 0.0), 1.0) for rank, p_value in enumerate(p_values, start=1)]
    adjusted = raw[:]
    running_min = 1.0
    for index in range(m - 1, -1, -1):
        running_min = min(running_min, adjusted[index])
        adjusted[index] = running_min
    return adjusted


def _sign_flip_p_value(
    differences: Sequence[float],
    *,
    alternative: str,
    exact_max_n: int,
    iterations: int,
    seed: int,
) -> tuple[float, str]:
    if len(differences) <= exact_max_n:
        return _exact_sign_flip_p_value(differences, alternative=alternative), "exact"
    return (
        _monte_carlo_sign_flip_p_value(
            differences,
            alternative=alternative,
            iterations=iterations,
            seed=seed,
        ),
        "monte_carlo",
    )


def _exact_sign_flip_p_value(differences: Sequence[float], *, alternative: str) -> float:
    observed = _mean(differences)
    total = 1 << len(differences)
    count = 0
    for mask in range(total):
        signed_sum = 0.0
        for index, difference in enumerate(differences):
            signed_sum += difference if (mask >> index) & 1 else -difference
        statistic = signed_sum / len(differences)
        if _sign_flip_extreme(statistic, observed=observed, alternative=alternative):
            count += 1
    return count / total


def _monte_carlo_sign_flip_p_value(
    differences: Sequence[float],
    *,
    alternative: str,
    iterations: int,
    seed: int,
) -> float:
    observed = _mean(differences)
    rng = random.Random(seed)
    count = 0
    for _ in range(iterations):
        statistic = sum(difference if rng.random() < 0.5 else -difference for difference in differences) / len(differences)
        if _sign_flip_extreme(statistic, observed=observed, alternative=alternative):
            count += 1
    return (count + 1) / (iterations + 1)


def _sign_flip_extreme(statistic: float, *, observed: float, alternative: str) -> bool:
    tolerance = 1e-12
    if alternative == "greater":
        return statistic >= observed - tolerance
    if alternative == "less":
        return statistic <= observed + tolerance
    return abs(statistic) >= abs(observed) - tolerance


def _normal_mean_ci(mean_value: float, *, se: float, confidence_level: float) -> tuple[float, float]:
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return mean_value - z_value * se, mean_value + z_value * se


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence_level: float,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    sample_size = len(values)
    means = sorted(_mean([values[rng.randrange(sample_size)] for _ in range(sample_size)]) for _ in range(iterations))
    alpha = 1.0 - confidence_level
    lower_index = max(0, min(iterations - 1, math.floor((alpha / 2.0) * (iterations - 1))))
    upper_index = max(0, min(iterations - 1, math.ceil((1.0 - alpha / 2.0) * (iterations - 1))))
    return means[lower_index], means[upper_index]


def _provenance_rows(
    *,
    input_row_count: int,
    valid_row_count: int,
    invalid_row_count: int,
    summary_row_count: int,
    multiplicity_row_count: int,
    loso_row_count: int,
    missingness_row_count: int,
    qc_row_count: int,
    subject_column: str,
    value_column: str,
    group_columns: Sequence[str],
    metadata_columns: Sequence[str],
    null_value: float,
    null_value_column: str | None,
    expected_subject_count: int | None,
    duplicate_subject_policy: str,
    min_n: int,
    confidence_level: float,
    ci_method: str,
    bootstrap_iterations: int,
    alternative: str,
    exact_max_n: int,
    sign_flip_iterations: int,
    seed: int,
    fdr_family_columns: Sequence[str] | None,
) -> list[SubjectInferenceProvenanceRow]:
    return [
        SubjectInferenceProvenanceRow("source", "research_platform.analysis.subject_inference"),
        SubjectInferenceProvenanceRow("input_row_count", input_row_count),
        SubjectInferenceProvenanceRow("valid_row_count", valid_row_count),
        SubjectInferenceProvenanceRow("invalid_row_count", invalid_row_count),
        SubjectInferenceProvenanceRow("summary_row_count", summary_row_count),
        SubjectInferenceProvenanceRow("multiplicity_row_count", multiplicity_row_count),
        SubjectInferenceProvenanceRow("loso_row_count", loso_row_count),
        SubjectInferenceProvenanceRow("missingness_row_count", missingness_row_count),
        SubjectInferenceProvenanceRow("qc_row_count", qc_row_count),
        SubjectInferenceProvenanceRow("subject_column", subject_column),
        SubjectInferenceProvenanceRow("value_column", value_column),
        SubjectInferenceProvenanceRow("group_columns", tuple(group_columns)),
        SubjectInferenceProvenanceRow("metadata_columns", tuple(metadata_columns)),
        SubjectInferenceProvenanceRow("null_value", null_value),
        SubjectInferenceProvenanceRow("null_value_column", null_value_column),
        SubjectInferenceProvenanceRow("expected_subject_count", expected_subject_count),
        SubjectInferenceProvenanceRow("duplicate_subject_policy", duplicate_subject_policy),
        SubjectInferenceProvenanceRow("min_n", min_n),
        SubjectInferenceProvenanceRow("confidence_level", confidence_level),
        SubjectInferenceProvenanceRow("ci_method", ci_method),
        SubjectInferenceProvenanceRow("bootstrap_iterations", bootstrap_iterations),
        SubjectInferenceProvenanceRow("p_method", "sign_flip"),
        SubjectInferenceProvenanceRow("alternative", alternative),
        SubjectInferenceProvenanceRow("exact_max_n", exact_max_n),
        SubjectInferenceProvenanceRow("sign_flip_iterations", sign_flip_iterations),
        SubjectInferenceProvenanceRow("seed", seed),
        SubjectInferenceProvenanceRow("fdr_family_columns", None if fdr_family_columns is None else tuple(fdr_family_columns)),
        SubjectInferenceProvenanceRow("output_written", False),
    ]


def _coerced_provenance_rows(
    rows: Iterable[Mapping[str, Any] | SubjectInferenceProvenanceRow] | None,
) -> list[SubjectInferenceProvenanceRow]:
    if rows is None:
        return []
    coerced_rows: list[SubjectInferenceProvenanceRow] = []
    for index, row in enumerate(rows):
        if isinstance(row, SubjectInferenceProvenanceRow):
            coerced_rows.append(row)
        elif isinstance(row, Mapping) and "key" in row and "value" in row:
            coerced_rows.append(
                SubjectInferenceProvenanceRow(
                    key=str(row["key"]),
                    value=row["value"],
                    source=str(row.get("source", "input")),
                )
            )
        elif isinstance(row, Mapping):
            coerced_rows.append(SubjectInferenceProvenanceRow(key=f"input_provenance_{index}", value=dict(row), source="input"))
        else:
            coerced_rows.append(SubjectInferenceProvenanceRow(key=f"input_provenance_{index}", value=str(row), source="input"))
    return coerced_rows


def _qc_row(
    *,
    level: str,
    status: str,
    code: str,
    message: str,
    source_row_index: int | None = None,
    subject_id: str | None = None,
    field_name: str | None = None,
    group_label: str | None = None,
    effect_label: str | None = None,
    measure: str | None = None,
    group_key: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> SubjectInferenceQcRow:
    return SubjectInferenceQcRow(
        level=level,
        status=status,
        code=code,
        message=message,
        source="subject_inference",
        source_row_index=source_row_index,
        subject_id=subject_id,
        field_name=field_name,
        group_label=group_label,
        effect_label=effect_label,
        measure=measure,
        group_key=group_key or {},
        context=context or {},
    )


def _messages_from_qc(qc_rows: Sequence[SubjectInferenceQcRow]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    errors: list[str] = []
    for qc_row in qc_rows:
        if qc_row.status == "failed":
            errors.append(qc_row.message)
        else:
            warnings.append(qc_row.message)
    return _unique(warnings), _unique(errors)


def _required_column_name(value: object, *, field_name: str) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError(f"{field_name} must be a non-empty field name.")
    return name


def _optional_column_name(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_column_name(value, field_name=field_name)


def _resolved_columns(columns: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of field names.")
    try:
        raw_columns = tuple(columns)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of field names.") from exc
    resolved: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for column in raw_columns:
        name = _required_column_name(column, field_name=field_name)
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
        resolved.append(name)
    if duplicates:
        raise ValueError(f"{field_name} field names must be unique: {', '.join(duplicates)}.")
    return tuple(resolved)


def _expected_subject_labels(expected_subjects: Iterable[Any] | None) -> tuple[str, ...] | None:
    if expected_subjects is None:
        return None
    if isinstance(expected_subjects, (str, bytes)):
        raise ValueError("expected_subjects must be an iterable of subject labels, not a single string.")
    labels: OrderedDict[str, None] = OrderedDict()
    for value in expected_subjects:
        label = _subject_label(value)
        if label is None:
            raise ValueError("expected_subjects must not contain empty subject labels.")
        labels[label] = None
    return tuple(labels)


def _subject_label(value: object) -> str | None:
    if value is None:
        return None
    label = str(value).strip()
    return label or None


def _label_from_column(
    bucket: _GroupBucket,
    *,
    metadata: Mapping[str, Any],
    column: str | None,
    fallback: object,
) -> str | None:
    if column is None:
        return _optional_label(fallback)
    if column in bucket.group_key:
        return _optional_label(bucket.group_key[column])
    if column in metadata:
        return _optional_label(metadata[column])
    values: OrderedDict[str, Any] = OrderedDict()
    for row in bucket.rows:
        if column in row.source_row:
            try:
                value = _json_safe(row.source_row[column])
            except ValueError:
                continue
            values[_stable_token(value)] = value
    if len(values) == 1:
        return _optional_label(next(iter(values.values())))
    return _optional_label(fallback)


def _first_group_value(group_key: Mapping[str, Any]) -> Any:
    if not group_key:
        return None
    return next(iter(group_key.values()))


def _nth_group_value(group_key: Mapping[str, Any], index: int) -> Any:
    values = tuple(group_key.values())
    if index >= len(values):
        return None
    return values[index]


def _group_token(group_key: Mapping[str, Any], group_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(_stable_token(group_key[column]) for column in group_columns)


def _row_lookup(row: SubjectInferenceResultRow, column: str) -> Any:
    if column in row.group_key:
        return row.group_key[column]
    if column in row.metadata:
        return row.metadata[column]
    if column == "group_label":
        return row.group_label
    if column == "effect_label":
        return row.effect_label
    if column == "measure":
        return row.measure
    if hasattr(row, column):
        return getattr(row, column)
    return None


def _summary_sort_token(row: SubjectInferenceResultRow) -> str:
    return _stable_token(
        {
            "group_label": row.group_label,
            "effect_label": row.effect_label,
            "measure": row.measure,
            "group_key": row.group_key,
        }
    )


def _finite_float(value: object, *, field_name: str) -> float:
    parsed = _coerce_finite_float(value)
    if parsed is None:
        raise ValueError(f"{field_name} must be a finite numeric value and must not be bool-valued.")
    return parsed


def _coerce_finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _unique_floats(values: Iterable[float]) -> tuple[float, ...]:
    unique_values: OrderedDict[str, float] = OrderedDict()
    for value in values:
        unique_values[repr(value)] = value
    return tuple(unique_values.values())


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


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


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(item) for key, item in value.items()}


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
            raise ValueError("JSON-safe subject-level inference outputs cannot contain non-finite floats.")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _tsv_safe_dataclass(instance: object) -> dict[str, str]:
    if not is_dataclass(instance):
        raise TypeError("_tsv_safe_dataclass requires a dataclass instance.")
    return {field_.name: _tsv_safe(getattr(instance, field_.name)) for field_ in fields(instance)}


def _tsv_safe(value: Any) -> str:
    safe_value = _json_safe(value)
    if safe_value is None:
        return ""
    if isinstance(safe_value, bool):
        return "true" if safe_value else "false"
    if isinstance(safe_value, (int, float)):
        return repr(safe_value)
    if isinstance(safe_value, str):
        return _without_tsv_control_chars(safe_value)
    return _without_tsv_control_chars(json.dumps(safe_value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _without_tsv_control_chars(value: str) -> str:
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _stable_token(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_EXACT_SIGN_FLIP_MAX_N",
    "DEFAULT_MONTE_CARLO_SIGN_FLIP_ITERATIONS",
    "SubjectInferenceLosoRow",
    "SubjectInferenceMissingnessRow",
    "SubjectInferenceMultiplicityRow",
    "SubjectInferenceProvenanceRow",
    "SubjectInferenceQcRow",
    "SubjectInferenceResultRow",
    "SubjectInferenceSummaryResult",
    "summarize_subject_level_inference",
]
