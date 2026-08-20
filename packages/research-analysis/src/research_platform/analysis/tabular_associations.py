"""Contracts for generic tabular association workflow planning and QC.

This module validates declarative association workflow configuration, returns
JSON-safe preview rows, can inspect records or small stdlib-readable source
files for QC-only inventory, can coerce generic standard-library row sources
into copied records, and computes bounded Pearson/Spearman correlation
association rows plus bounded same-source residualized partial and OLS
primary-predictor regression-style rows. It does not render reports, write
outputs, fit mixed or generalized models, or implement concrete dataframe
backends.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
import math
from pathlib import Path
from typing import Any, ClassVar

from .publication_tables import (
    NumericFormatSpec,
    PValueFormatSpec,
    PublicationColumnSpec,
    PublicationFormatSpec,
    PublicationTableSpec,
    build_publication_table_rows,
)


SCHEMA_VERSION = "research_platform.analysis.tabular_associations.v1"
TABULAR_ASSOCIATION_PUBLICATION_HANDOFF_VERSION = (
    "research_platform.analysis.tabular_associations.publication_handoff.v1"
)
TABULAR_ASSOCIATION_ROW_SOURCE_ADAPTER_VERSION = (
    "research_platform.analysis.tabular_associations.row_source_adapter.v1"
)
TABULAR_ASSOCIATION_REPEATED_MEASURES_PLAN_VERSION = (
    "research_platform.analysis.tabular_associations.repeated_measures_plan.v1"
)
TABULAR_ASSOCIATION_REPEATED_MEASURES_METADATA_VERSION = (
    "research_platform.analysis.tabular_associations.repeated_measures_metadata.v1"
)
TABULAR_ASSOCIATION_MODEL_RESULTS_CONTRACT_VERSION = (
    "research_platform.analysis.tabular_associations.model_results_contract.v1"
)

BACKEND_RECORDS = "records"
BACKEND_POLARS = "polars"
BACKEND_PANDAS = "pandas"
SUPPORTED_TABULAR_ASSOCIATION_BACKENDS = frozenset({BACKEND_RECORDS, BACKEND_POLARS, BACKEND_PANDAS})

METHOD_PEARSON = "pearson"
METHOD_SPEARMAN = "spearman"
METHOD_PARTIAL_CORRELATION = "partial_correlation"
METHOD_REGRESSION = "regression"
METHOD_REPEATED_MEASURES = "repeated_measures"
METHOD_MIXED_MODEL = "mixed_model"
MODEL_RESULT_KIND_MODEL_FIT_SUMMARY = "model_fit_summary"
MODEL_RESULT_KIND_FIXED_EFFECT = "fixed_effect"
MODEL_RESULT_KIND_RANDOM_EFFECT = "random_effect"
MODEL_RESULT_KIND_VARIANCE_COMPONENT = "variance_component"
MODEL_RESULT_KIND_PLANNED_COMPARISON = "planned_comparison"
MODEL_RESULT_KIND_CONTRAST = "contrast"
SUPPORTED_MODEL_RESULT_KINDS = frozenset(
    {
        MODEL_RESULT_KIND_MODEL_FIT_SUMMARY,
        MODEL_RESULT_KIND_FIXED_EFFECT,
        MODEL_RESULT_KIND_RANDOM_EFFECT,
        MODEL_RESULT_KIND_VARIANCE_COMPONENT,
        MODEL_RESULT_KIND_PLANNED_COMPARISON,
        MODEL_RESULT_KIND_CONTRAST,
    }
)
SUPPORTED_ASSOCIATION_METHODS = frozenset(
    {
        METHOD_PEARSON,
        METHOD_SPEARMAN,
        METHOD_PARTIAL_CORRELATION,
        METHOD_REGRESSION,
        METHOD_REPEATED_MEASURES,
        METHOD_MIXED_MODEL,
    }
)
DEFERRED_ASSOCIATION_METHODS = frozenset({METHOD_REPEATED_MEASURES, METHOD_MIXED_MODEL})

SUPPORTED_MISSING_DATA_POLICIES = frozenset({"error", "listwise", "pairwise", "drop_rows", "allow"})
SUPPORTED_DUPLICATE_SUBJECT_POLICIES = frozenset({"error", "allow", "first", "last", "aggregate_deferred"})
SUPPORTED_NONFINITE_POLICIES = frozenset({"error", "allow", "coerce_missing", "drop_rows"})
SUPPORTED_CATEGORICAL_VALIDATION_POLICIES = frozenset({"none", "declare", "strict"})
SUPPORTED_NUMERIC_VALIDATION_POLICIES = frozenset({"none", "declare", "strict"})
SUPPORTED_STANDARDIZATION_METHODS = frozenset({"none", "center", "z_score", "scale_unit", "rank", "defer"})
SUPPORTED_TRANSFORMATION_METHODS = frozenset({"none", "log", "log1p", "sqrt", "rank", "winsorize", "custom_deferred"})
SUPPORTED_MULTIPLE_TESTING_METHODS = frozenset(
    {"none", "benjamini_hochberg", "benjamini_yekutieli", "bonferroni", "holm", "fdr_bh"}
)
SUPPORTED_P_VALUE_POLICIES = frozenset({"warn", "error"})
BENJAMINI_HOCHBERG_METHODS = frozenset({"benjamini_hochberg", "fdr_bh"})
SUPPORTED_OUTPUT_TYPES = frozenset(
    {
        "association_results",
        "missingness",
        "qc",
        "provenance",
        "publication_handoff",
        "visualization_handoff",
        "report_handoff",
    }
)
SUPPORTED_HANDOFF_TYPES = frozenset({"publication", "visualization", "report"})
SUPPORTED_SOURCE_INVENTORY_FORMATS = frozenset({"tsv", "csv", "json"})
SOURCE_KIND_IN_MEMORY = "in_memory"
SOURCE_KIND_TSV = "tsv"
SOURCE_KIND_CSV = "csv"
SOURCE_KIND_JSON = "json"
SOURCE_KIND_MISSING = "missing"
SOURCE_KIND_UNSUPPORTED = "unsupported"
RUNTIME_BACKEND_RECORDS = BACKEND_RECORDS

_REPEATED_MEASURES_METADATA_KEYS = (
    "model_design",
    "fixed_effect_terms",
    "random_effect_terms",
    "random_intercepts",
    "random_slopes",
    "repeated_factors",
    "within_subject_factors",
    "between_subject_factors",
    "grouping_factors",
    "cluster_terms",
    "timepoint_roles",
    "categorical_coding",
    "formula_metadata",
    "formula_like",
    "planned_comparisons",
    "contrast_metadata",
    "model_family",
    "link_function",
)


@dataclass(frozen=True)
class ColumnSpec:
    """Declared column metadata for a source table."""

    column_name: str
    value_type: str = "unspecified"
    role: str | None = None
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "column_name", _non_empty_text(self.column_name, field_name="column_name"))
        object.__setattr__(self, "value_type", _non_empty_text(self.value_type, field_name="value_type"))
        object.__setattr__(self, "role", _optional_text(self.role))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class CategoricalValidationPolicy:
    """Declared categorical validation policy; no row values are inspected."""

    policy: str = "declare"
    allowed_values: Mapping[str, Sequence[str]] = field(default_factory=dict)
    allow_unlisted: bool = True
    case_sensitive: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policy = _normalized_choice(
            self.policy,
            field_name="categorical validation policy",
            supported=SUPPORTED_CATEGORICAL_VALIDATION_POLICIES,
        )
        object.__setattr__(self, "policy", policy)
        normalized_values = {
            _non_empty_text(column, field_name="categorical validation column"): tuple(
                _non_empty_text(value, field_name="categorical allowed value") for value in values
            )
            for column, values in self.allowed_values.items()
        }
        object.__setattr__(self, "allowed_values", normalized_values)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class NumericValidationPolicy:
    """Declared numeric validation policy; no row values are inspected."""

    policy: str = "declare"
    min_value: float | int | None = None
    max_value: float | int | None = None
    integer_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policy = _normalized_choice(
            self.policy,
            field_name="numeric validation policy",
            supported=SUPPORTED_NUMERIC_VALIDATION_POLICIES,
        )
        object.__setattr__(self, "policy", policy)
        if self.min_value is not None:
            object.__setattr__(self, "min_value", _finite_number(self.min_value, field_name="min_value"))
        if self.max_value is not None:
            object.__setattr__(self, "max_value", _finite_number(self.max_value, field_name="max_value"))
        if self.min_value is not None and self.max_value is not None and self.min_value > self.max_value:
            raise ValueError("NumericValidationPolicy.min_value must be less than or equal to max_value.")
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularSchemaSpec:
    """Declared schema metadata for one source table."""

    subject_id_column: str
    columns: Sequence[ColumnSpec] = ()
    session_column: str | None = None
    timepoint_column: str | None = None
    categorical_validation: CategoricalValidationPolicy = field(default_factory=CategoricalValidationPolicy)
    numeric_validation: NumericValidationPolicy = field(default_factory=NumericValidationPolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subject_id_column",
            _non_empty_text(self.subject_id_column, field_name="subject_id_column"),
        )
        object.__setattr__(self, "session_column", _optional_text(self.session_column))
        object.__setattr__(self, "timepoint_column", _optional_text(self.timepoint_column))
        columns = tuple(_coerce_column_spec(column) for column in self.columns)
        duplicate_columns = _duplicates([column.column_name for column in columns])
        if duplicate_columns:
            raise ValueError(f"TabularSchemaSpec.columns contains duplicate columns: {', '.join(duplicate_columns)}.")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(
            self,
            "categorical_validation",
            _policy_from_mapping(self.categorical_validation, CategoricalValidationPolicy),
        )
        object.__setattr__(
            self,
            "numeric_validation",
            _policy_from_mapping(self.numeric_validation, NumericValidationPolicy),
        )
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def column_names(self) -> tuple[str, ...]:
        return tuple(column.column_name for column in self.columns)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularSourceSpec:
    """Declared source table metadata.

    ``path`` and ``root_ref`` are metadata only. They are not resolved, opened,
    checked for existence, or written by this module.
    """

    source_id: str
    schema: TabularSchemaSpec
    format: str | None = None
    path: str | None = None
    root_ref: str | None = None
    backend: str = BACKEND_RECORDS
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "schema", _coerce_schema_spec(self.schema))
        object.__setattr__(self, "format", _optional_text(self.format))
        object.__setattr__(self, "path", _optional_text(self.path))
        object.__setattr__(self, "root_ref", _optional_text(self.root_ref))
        backend = _normalized_choice(self.backend, field_name="backend", supported=SUPPORTED_TABULAR_ASSOCIATION_BACKENDS)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AssociationVariableSpec:
    """Declared association variable metadata."""

    variable_id: str
    source_id: str
    column_name: str
    role: str = "variable"
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable_id", _non_empty_text(self.variable_id, field_name="variable_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "column_name", _non_empty_text(self.column_name, field_name="column_name"))
        object.__setattr__(self, "role", _non_empty_text(self.role, field_name="role"))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class OutcomeSpec(AssociationVariableSpec):
    """Declared outcome or measure variable."""

    role: str = "outcome"

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", "outcome")
        super().__post_init__()


@dataclass(frozen=True)
class PredictorSpec(AssociationVariableSpec):
    """Declared predictor or exposure variable."""

    role: str = "predictor"

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", "predictor")
        super().__post_init__()


@dataclass(frozen=True)
class CovariateSpec(AssociationVariableSpec):
    """Declared covariate variable."""

    role: str = "covariate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", "covariate")
        super().__post_init__()


@dataclass(frozen=True)
class GroupingSpec(AssociationVariableSpec):
    """Declared grouping or stratification variable."""

    role: str = "grouping"

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", "grouping")
        super().__post_init__()


@dataclass(frozen=True)
class RepeatedMeasuresSpec:
    """Declared repeated-measures identifiers for planning only."""

    source_id: str
    subject_id_column: str
    session_column: str | None = None
    timepoint_column: str | None = None
    unit_columns: Sequence[str] = ()
    planned_only: bool = True
    executable: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(
            self,
            "subject_id_column",
            _non_empty_text(self.subject_id_column, field_name="subject_id_column"),
        )
        object.__setattr__(self, "session_column", _optional_text(self.session_column))
        object.__setattr__(self, "timepoint_column", _optional_text(self.timepoint_column))
        object.__setattr__(
            self,
            "unit_columns",
            tuple(_non_empty_text(column, field_name="unit_columns") for column in self.unit_columns),
        )
        if not self.planned_only or self.executable:
            raise ValueError("RepeatedMeasuresSpec is schema-only and must remain planned-only/non-executable.")
        object.__setattr__(self, "planned_only", True)
        object.__setattr__(self, "executable", False)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MissingDataPolicy:
    """Declared missing-data policy for future adapters."""

    strategy: str = "listwise"
    required_roles: Sequence[str] = ("outcome", "predictor")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        strategy = _normalized_choice(
            self.strategy,
            field_name="missing-data policy",
            supported=SUPPORTED_MISSING_DATA_POLICIES,
        )
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(
            self,
            "required_roles",
            tuple(_non_empty_text(role, field_name="required_roles") for role in self.required_roles),
        )
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class DuplicateSubjectPolicy:
    """Declared duplicate-subject policy for future adapters."""

    strategy: str = "error"
    key_columns: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        strategy = _normalized_choice(
            self.strategy,
            field_name="duplicate-subject policy",
            supported=SUPPORTED_DUPLICATE_SUBJECT_POLICIES,
        )
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(
            self,
            "key_columns",
            tuple(_non_empty_text(column, field_name="key_columns") for column in self.key_columns),
        )
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class NonFinitePolicy:
    """Declared non-finite value policy for future adapters."""

    strategy: str = "error"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        strategy = _normalized_choice(
            self.strategy,
            field_name="non-finite policy",
            supported=SUPPORTED_NONFINITE_POLICIES,
        )
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class StandardizationPolicy:
    """Declared standardization/scaling policy for future adapters."""

    method: str = "none"
    variable_ids: Sequence[str] = ()
    group_by: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        method = _normalized_choice(
            self.method,
            field_name="standardization method",
            supported=SUPPORTED_STANDARDIZATION_METHODS,
        )
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self,
            "variable_ids",
            tuple(_non_empty_text(variable_id, field_name="variable_ids") for variable_id in self.variable_ids),
        )
        object.__setattr__(
            self,
            "group_by",
            tuple(_non_empty_text(grouping_id, field_name="group_by") for grouping_id in self.group_by),
        )
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TransformationPolicy:
    """Declared transformation policy for future adapters."""

    method: str = "none"
    variable_ids: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        method = _normalized_choice(
            self.method,
            field_name="transformation method",
            supported=SUPPORTED_TRANSFORMATION_METHODS,
        )
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self,
            "variable_ids",
            tuple(_non_empty_text(variable_id, field_name="variable_ids") for variable_id in self.variable_ids),
        )
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AssociationMethodSpec:
    """Declared association method metadata.

    Method specs are always plan-only and non-executable in Step 11B.
    """

    method_id: str
    method_name: str
    outcome_ids: Sequence[str] = ()
    predictor_ids: Sequence[str] = ()
    covariate_ids: Sequence[str] = ()
    grouping_ids: Sequence[str] = ()
    family_id: str | None = None
    output_id: str | None = None
    planned_only: bool = True
    executable: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        method_name = _normalized_choice(
            self.method_name,
            field_name="association method",
            supported=SUPPORTED_ASSOCIATION_METHODS,
        )
        object.__setattr__(self, "method_name", method_name)
        object.__setattr__(self, "outcome_ids", _text_tuple(self.outcome_ids, field_name="outcome_ids"))
        object.__setattr__(self, "predictor_ids", _text_tuple(self.predictor_ids, field_name="predictor_ids"))
        object.__setattr__(self, "covariate_ids", _text_tuple(self.covariate_ids, field_name="covariate_ids"))
        object.__setattr__(self, "grouping_ids", _text_tuple(self.grouping_ids, field_name="grouping_ids"))
        object.__setattr__(self, "family_id", _optional_text(self.family_id))
        object.__setattr__(self, "output_id", _optional_text(self.output_id))
        if not self.planned_only or self.executable:
            raise ValueError("AssociationMethodSpec is schema-only and must remain planned-only/non-executable.")
        object.__setattr__(self, "planned_only", True)
        object.__setattr__(self, "executable", False)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def is_deferred(self) -> bool:
        return self.method_name in DEFERRED_ASSOCIATION_METHODS

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class CorrelationSpec(AssociationMethodSpec):
    """Declared Pearson or Spearman correlation plan."""

    method_name: str = METHOD_PEARSON

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.method_name not in {METHOD_PEARSON, METHOD_SPEARMAN}:
            raise ValueError("CorrelationSpec.method_name must be pearson or spearman.")


@dataclass(frozen=True)
class PartialCorrelationSpec(AssociationMethodSpec):
    """Declared covariate-adjusted association plan."""

    method_name: str = METHOD_PARTIAL_CORRELATION

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.method_name != METHOD_PARTIAL_CORRELATION:
            raise ValueError("PartialCorrelationSpec.method_name must be partial_correlation.")


@dataclass(frozen=True)
class RegressionAssociationSpec(AssociationMethodSpec):
    """Declared regression-style association plan."""

    method_name: str = METHOD_REGRESSION

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.method_name != METHOD_REGRESSION:
            raise ValueError("RegressionAssociationSpec.method_name must be regression.")


@dataclass(frozen=True)
class RepeatedMeasuresAssociationSpec(AssociationMethodSpec):
    """Declared repeated-measures or mixed-model-style plan, deferred only."""

    method_name: str = METHOD_REPEATED_MEASURES
    deferred_reason: str = "repeated-measures execution is deferred"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.method_name not in DEFERRED_ASSOCIATION_METHODS:
            raise ValueError("RepeatedMeasuresAssociationSpec.method_name must be repeated_measures or mixed_model.")
        object.__setattr__(
            self,
            "deferred_reason",
            _non_empty_text(self.deferred_reason, field_name="deferred_reason"),
        )


@dataclass(frozen=True)
class FixedEffectTermSpec:
    """Explicit fixed-effect term metadata; no model coefficients are computed."""

    term_id: str
    variable_ids: Sequence[str] = ()
    column_names: Sequence[str] = ()
    factor_ids: Sequence[str] = ()
    coding_ids: Sequence[str] = ()
    source_id: str | None = None
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "term_id", _non_empty_text(self.term_id, field_name="term_id"))
        object.__setattr__(self, "variable_ids", _text_tuple(self.variable_ids, field_name="variable_ids"))
        object.__setattr__(self, "column_names", _text_tuple(self.column_names, field_name="column_names"))
        object.__setattr__(self, "factor_ids", _text_tuple(self.factor_ids, field_name="factor_ids"))
        object.__setattr__(self, "coding_ids", _text_tuple(self.coding_ids, field_name="coding_ids"))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RandomInterceptSpec:
    """Explicit random-intercept metadata; model fitting is deferred."""

    intercept_id: str
    grouping_ids: Sequence[str] = ()
    grouping_columns: Sequence[str] = ()
    cluster_ids: Sequence[str] = ()
    source_id: str | None = None
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "intercept_id", _non_empty_text(self.intercept_id, field_name="intercept_id"))
        object.__setattr__(self, "grouping_ids", _text_tuple(self.grouping_ids, field_name="grouping_ids"))
        object.__setattr__(self, "grouping_columns", _text_tuple(self.grouping_columns, field_name="grouping_columns"))
        object.__setattr__(self, "cluster_ids", _text_tuple(self.cluster_ids, field_name="cluster_ids"))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RandomSlopeSpec:
    """Explicit random-slope metadata; no random-slope model is fitted."""

    slope_id: str
    variable_ids: Sequence[str] = ()
    column_names: Sequence[str] = ()
    factor_ids: Sequence[str] = ()
    grouping_ids: Sequence[str] = ()
    grouping_columns: Sequence[str] = ()
    cluster_ids: Sequence[str] = ()
    source_id: str | None = None
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slope_id", _non_empty_text(self.slope_id, field_name="slope_id"))
        object.__setattr__(self, "variable_ids", _text_tuple(self.variable_ids, field_name="variable_ids"))
        object.__setattr__(self, "column_names", _text_tuple(self.column_names, field_name="column_names"))
        object.__setattr__(self, "factor_ids", _text_tuple(self.factor_ids, field_name="factor_ids"))
        object.__setattr__(self, "grouping_ids", _text_tuple(self.grouping_ids, field_name="grouping_ids"))
        object.__setattr__(self, "grouping_columns", _text_tuple(self.grouping_columns, field_name="grouping_columns"))
        object.__setattr__(self, "cluster_ids", _text_tuple(self.cluster_ids, field_name="cluster_ids"))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RandomEffectTermSpec:
    """Explicit random-effect term metadata; no mixed model is executed."""

    term_id: str
    random_intercept_ids: Sequence[str] = ()
    random_slope_ids: Sequence[str] = ()
    variable_ids: Sequence[str] = ()
    column_names: Sequence[str] = ()
    factor_ids: Sequence[str] = ()
    grouping_ids: Sequence[str] = ()
    cluster_ids: Sequence[str] = ()
    source_id: str | None = None
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "term_id", _non_empty_text(self.term_id, field_name="term_id"))
        object.__setattr__(
            self,
            "random_intercept_ids",
            _text_tuple(self.random_intercept_ids, field_name="random_intercept_ids"),
        )
        object.__setattr__(self, "random_slope_ids", _text_tuple(self.random_slope_ids, field_name="random_slope_ids"))
        object.__setattr__(self, "variable_ids", _text_tuple(self.variable_ids, field_name="variable_ids"))
        object.__setattr__(self, "column_names", _text_tuple(self.column_names, field_name="column_names"))
        object.__setattr__(self, "factor_ids", _text_tuple(self.factor_ids, field_name="factor_ids"))
        object.__setattr__(self, "grouping_ids", _text_tuple(self.grouping_ids, field_name="grouping_ids"))
        object.__setattr__(self, "cluster_ids", _text_tuple(self.cluster_ids, field_name="cluster_ids"))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RepeatedFactorSpec:
    """Explicit repeated-factor metadata for design planning only."""

    factor_id: str
    column_name: str | None = None
    source_id: str | None = None
    levels: Sequence[str] = ()
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor_id", _non_empty_text(self.factor_id, field_name="factor_id"))
        object.__setattr__(self, "column_name", _optional_text(self.column_name))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "levels", _text_tuple(self.levels, field_name="levels"))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class WithinSubjectFactorSpec:
    """Explicit within-subject factor metadata for planning only."""

    factor_id: str
    column_name: str | None = None
    source_id: str | None = None
    repeated_factor_id: str | None = None
    levels: Sequence[str] = ()
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor_id", _non_empty_text(self.factor_id, field_name="factor_id"))
        object.__setattr__(self, "column_name", _optional_text(self.column_name))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "repeated_factor_id", _optional_text(self.repeated_factor_id))
        object.__setattr__(self, "levels", _text_tuple(self.levels, field_name="levels"))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class BetweenSubjectFactorSpec:
    """Explicit between-subject factor metadata for planning only."""

    factor_id: str
    column_name: str | None = None
    source_id: str | None = None
    variable_id: str | None = None
    levels: Sequence[str] = ()
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor_id", _non_empty_text(self.factor_id, field_name="factor_id"))
        object.__setattr__(self, "column_name", _optional_text(self.column_name))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "variable_id", _optional_text(self.variable_id))
        object.__setattr__(self, "levels", _text_tuple(self.levels, field_name="levels"))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class GroupingFactorSpec:
    """Explicit grouping-factor metadata for mixed-model declarations."""

    grouping_id: str
    variable_id: str | None = None
    column_name: str | None = None
    source_id: str | None = None
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "grouping_id", _non_empty_text(self.grouping_id, field_name="grouping_id"))
        object.__setattr__(self, "variable_id", _optional_text(self.variable_id))
        object.__setattr__(self, "column_name", _optional_text(self.column_name))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ClusterTermSpec:
    """Explicit cluster metadata for design planning only."""

    cluster_id: str
    column_name: str | None = None
    source_id: str | None = None
    grouping_id: str | None = None
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cluster_id", _non_empty_text(self.cluster_id, field_name="cluster_id"))
        object.__setattr__(self, "column_name", _optional_text(self.column_name))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "grouping_id", _optional_text(self.grouping_id))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TimepointRoleSpec:
    """Explicit timepoint-role metadata for design planning only."""

    role_id: str
    column_name: str | None = None
    source_id: str | None = None
    factor_id: str | None = None
    role: str | None = None
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_id", _non_empty_text(self.role_id, field_name="role_id"))
        object.__setattr__(self, "column_name", _optional_text(self.column_name))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "factor_id", _optional_text(self.factor_id))
        object.__setattr__(self, "role", _optional_text(self.role))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class CategoricalCodingSpec:
    """Explicit categorical-coding metadata; coding is not computed."""

    coding_id: str
    target_id: str | None = None
    variable_id: str | None = None
    factor_id: str | None = None
    column_name: str | None = None
    source_id: str | None = None
    scheme: str | None = None
    reference_level: str | None = None
    levels: Sequence[str] = ()
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coding_id", _non_empty_text(self.coding_id, field_name="coding_id"))
        object.__setattr__(self, "target_id", _optional_text(self.target_id))
        object.__setattr__(self, "variable_id", _optional_text(self.variable_id))
        object.__setattr__(self, "factor_id", _optional_text(self.factor_id))
        object.__setattr__(self, "column_name", _optional_text(self.column_name))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "scheme", _optional_text(self.scheme))
        object.__setattr__(self, "reference_level", _optional_text(self.reference_level))
        object.__setattr__(self, "levels", _text_tuple(self.levels, field_name="levels"))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ModelFormulaMetadataSpec:
    """Formula/design-intent metadata; formula strings are not parsed."""

    formula_id: str | None = None
    formula_like: str | None = None
    design_intent: str | None = None
    fixed_formula: str | None = None
    random_formula: str | None = None
    variable_ids: Sequence[str] = ()
    factor_ids: Sequence[str] = ()
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "formula_id", _optional_text(self.formula_id))
        object.__setattr__(self, "formula_like", _optional_text(self.formula_like))
        object.__setattr__(self, "design_intent", _optional_text(self.design_intent))
        object.__setattr__(self, "fixed_formula", _optional_text(self.fixed_formula))
        object.__setattr__(self, "random_formula", _optional_text(self.random_formula))
        object.__setattr__(self, "variable_ids", _text_tuple(self.variable_ids, field_name="variable_ids"))
        object.__setattr__(self, "factor_ids", _text_tuple(self.factor_ids, field_name="factor_ids"))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PlannedComparisonSpec:
    """Explicit planned-comparison metadata; no contrast is computed."""

    comparison_id: str
    factor_ids: Sequence[str] = ()
    variable_ids: Sequence[str] = ()
    grouping_ids: Sequence[str] = ()
    cluster_ids: Sequence[str] = ()
    coding_ids: Sequence[str] = ()
    contrast_metadata_ids: Sequence[str] = ()
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparison_id", _non_empty_text(self.comparison_id, field_name="comparison_id"))
        object.__setattr__(self, "factor_ids", _text_tuple(self.factor_ids, field_name="factor_ids"))
        object.__setattr__(self, "variable_ids", _text_tuple(self.variable_ids, field_name="variable_ids"))
        object.__setattr__(self, "grouping_ids", _text_tuple(self.grouping_ids, field_name="grouping_ids"))
        object.__setattr__(self, "cluster_ids", _text_tuple(self.cluster_ids, field_name="cluster_ids"))
        object.__setattr__(self, "coding_ids", _text_tuple(self.coding_ids, field_name="coding_ids"))
        object.__setattr__(
            self,
            "contrast_metadata_ids",
            _text_tuple(self.contrast_metadata_ids, field_name="contrast_metadata_ids"),
        )
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ContrastMetadataSpec:
    """Explicit contrast metadata; contrast estimates are not computed."""

    contrast_id: str
    comparison_ids: Sequence[str] = ()
    factor_ids: Sequence[str] = ()
    variable_ids: Sequence[str] = ()
    coding_ids: Sequence[str] = ()
    label: str | None = None
    metadata_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "contrast_id", _non_empty_text(self.contrast_id, field_name="contrast_id"))
        object.__setattr__(self, "comparison_ids", _text_tuple(self.comparison_ids, field_name="comparison_ids"))
        object.__setattr__(self, "factor_ids", _text_tuple(self.factor_ids, field_name="factor_ids"))
        object.__setattr__(self, "variable_ids", _text_tuple(self.variable_ids, field_name="variable_ids"))
        object.__setattr__(self, "coding_ids", _text_tuple(self.coding_ids, field_name="coding_ids"))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ModelDesignMetadataSpec:
    """Normalized repeated-measures/mixed-model design metadata declaration."""

    model_design_id: str | None = None
    fixed_effect_terms: Sequence[FixedEffectTermSpec | Mapping[str, Any] | str] = ()
    random_effect_terms: Sequence[RandomEffectTermSpec | Mapping[str, Any] | str] = ()
    random_intercepts: Sequence[RandomInterceptSpec | Mapping[str, Any] | str] = ()
    random_slopes: Sequence[RandomSlopeSpec | Mapping[str, Any] | str] = ()
    repeated_factors: Sequence[RepeatedFactorSpec | Mapping[str, Any] | str] = ()
    within_subject_factors: Sequence[WithinSubjectFactorSpec | Mapping[str, Any] | str] = ()
    between_subject_factors: Sequence[BetweenSubjectFactorSpec | Mapping[str, Any] | str] = ()
    grouping_factors: Sequence[GroupingFactorSpec | Mapping[str, Any] | str] = ()
    cluster_terms: Sequence[ClusterTermSpec | Mapping[str, Any] | str] = ()
    timepoint_roles: Sequence[TimepointRoleSpec | Mapping[str, Any] | str] = ()
    categorical_coding: Sequence[CategoricalCodingSpec | Mapping[str, Any] | str] = ()
    formula_metadata: ModelFormulaMetadataSpec | Mapping[str, Any] | str | None = None
    formula_like: str | None = None
    planned_comparisons: Sequence[PlannedComparisonSpec | Mapping[str, Any] | str] = ()
    contrast_metadata: Sequence[ContrastMetadataSpec | Mapping[str, Any] | str] = ()
    model_family: str | None = None
    link_function: str | None = None
    variable_ids: Sequence[str] = ()
    factor_ids: Sequence[str] = ()
    metadata_only: bool = True
    model_fitting_deferred: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_design_id", _optional_text(self.model_design_id))
        object.__setattr__(
            self,
            "fixed_effect_terms",
            tuple(_coerce_fixed_effect_term_spec(item) for item in self.fixed_effect_terms),
        )
        object.__setattr__(
            self,
            "random_effect_terms",
            tuple(_coerce_random_effect_term_spec(item) for item in self.random_effect_terms),
        )
        object.__setattr__(
            self,
            "random_intercepts",
            tuple(_coerce_random_intercept_spec(item) for item in self.random_intercepts),
        )
        object.__setattr__(self, "random_slopes", tuple(_coerce_random_slope_spec(item) for item in self.random_slopes))
        object.__setattr__(
            self,
            "repeated_factors",
            tuple(_coerce_repeated_factor_spec(item) for item in self.repeated_factors),
        )
        object.__setattr__(
            self,
            "within_subject_factors",
            tuple(_coerce_within_subject_factor_spec(item) for item in self.within_subject_factors),
        )
        object.__setattr__(
            self,
            "between_subject_factors",
            tuple(_coerce_between_subject_factor_spec(item) for item in self.between_subject_factors),
        )
        object.__setattr__(
            self,
            "grouping_factors",
            tuple(_coerce_grouping_factor_spec(item) for item in self.grouping_factors),
        )
        object.__setattr__(self, "cluster_terms", tuple(_coerce_cluster_term_spec(item) for item in self.cluster_terms))
        object.__setattr__(
            self,
            "timepoint_roles",
            tuple(_coerce_timepoint_role_spec(item) for item in self.timepoint_roles),
        )
        object.__setattr__(
            self,
            "categorical_coding",
            tuple(_coerce_categorical_coding_spec(item) for item in self.categorical_coding),
        )
        object.__setattr__(
            self,
            "formula_metadata",
            _coerce_model_formula_metadata_spec(self.formula_metadata),
        )
        object.__setattr__(self, "formula_like", _optional_text(self.formula_like))
        object.__setattr__(
            self,
            "planned_comparisons",
            tuple(_coerce_planned_comparison_spec(item) for item in self.planned_comparisons),
        )
        object.__setattr__(
            self,
            "contrast_metadata",
            tuple(_coerce_contrast_metadata_spec(item) for item in self.contrast_metadata),
        )
        object.__setattr__(self, "model_family", _optional_text(self.model_family))
        object.__setattr__(self, "link_function", _optional_text(self.link_function))
        object.__setattr__(self, "variable_ids", _text_tuple(self.variable_ids, field_name="variable_ids"))
        object.__setattr__(self, "factor_ids", _text_tuple(self.factor_ids, field_name="factor_ids"))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "model_fitting_deferred", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def has_declarations(self) -> bool:
        return any(
            (
                self.model_design_id,
                self.fixed_effect_terms,
                self.random_effect_terms,
                self.random_intercepts,
                self.random_slopes,
                self.repeated_factors,
                self.within_subject_factors,
                self.between_subject_factors,
                self.grouping_factors,
                self.cluster_terms,
                self.timepoint_roles,
                self.categorical_coding,
                self.formula_metadata,
                self.formula_like,
                self.planned_comparisons,
                self.contrast_metadata,
                self.model_family,
                self.link_function,
                self.variable_ids,
                self.factor_ids,
                self.metadata,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AssociationFamilySpec:
    """Declared family of association methods."""

    family_id: str
    method_ids: Sequence[str] = ()
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _non_empty_text(self.family_id, field_name="family_id"))
        object.__setattr__(self, "method_ids", _text_tuple(self.method_ids, field_name="method_ids"))
        object.__setattr__(self, "description", _optional_text(self.description))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MultipleTestingSpec:
    """Declared multiple-testing/FDR family metadata; no correction is run."""

    family_id: str
    method: str = "benjamini_hochberg"
    planned_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_id", _non_empty_text(self.family_id, field_name="family_id"))
        method = _normalized_choice(
            self.method,
            field_name="multiple-testing method",
            supported=SUPPORTED_MULTIPLE_TESTING_METHODS,
        )
        object.__setattr__(self, "method", method)
        if not self.planned_only:
            raise ValueError("MultipleTestingSpec is schema-only and must remain planned-only.")
        object.__setattr__(self, "planned_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AssociationOutputSpec:
    """Declared planned output rows/fields; no output is written."""

    output_id: str
    output_type: str = "association_results"
    planned_fields: Sequence[str] = ()
    source_method_ids: Sequence[str] = ()
    family_ids: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_id", _non_empty_text(self.output_id, field_name="output_id"))
        output_type = _normalized_choice(self.output_type, field_name="output_type", supported=SUPPORTED_OUTPUT_TYPES)
        object.__setattr__(self, "output_type", output_type)
        object.__setattr__(
            self,
            "planned_fields",
            tuple(_non_empty_text(field_name, field_name="planned_fields") for field_name in self.planned_fields),
        )
        object.__setattr__(
            self,
            "source_method_ids",
            _text_tuple(self.source_method_ids, field_name="source_method_ids"),
        )
        object.__setattr__(self, "family_ids", _text_tuple(self.family_ids, field_name="family_ids"))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AssociationHandoffSpec:
    """Declared metadata-only publication, visualization, or report handoff."""

    handoff_id: str
    handoff_type: str
    output_ids: Sequence[str] = ()
    target: str | None = None
    planned_fields: Sequence[str] = ()
    plan_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "handoff_id", _non_empty_text(self.handoff_id, field_name="handoff_id"))
        handoff_type = _normalized_choice(
            self.handoff_type,
            field_name="handoff_type",
            supported=SUPPORTED_HANDOFF_TYPES,
        )
        object.__setattr__(self, "handoff_type", handoff_type)
        object.__setattr__(self, "output_ids", _text_tuple(self.output_ids, field_name="output_ids"))
        object.__setattr__(
            self,
            "planned_fields",
            tuple(_non_empty_text(field_name, field_name="planned_fields") for field_name in self.planned_fields),
        )
        object.__setattr__(self, "target", _optional_text(self.target))
        if not self.plan_only:
            raise ValueError("AssociationHandoffSpec declarations are metadata-only and must remain plan-only.")
        object.__setattr__(self, "plan_only", True)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AssociationValidationRow:
    """One JSON-safe validation row."""

    level: str
    status: str
    code: str
    message: str
    location: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", _non_empty_text(self.level, field_name="level"))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        object.__setattr__(self, "location", _optional_text(self.location))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AssociationProvenanceRow:
    """One JSON-safe provenance row for the plan preview."""

    key: str
    value: Any
    source: str = "tabular_associations"

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _non_empty_text(self.key, field_name="key"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "source", _non_empty_text(self.source, field_name="source"))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularAssociationWorkflowSpec:
    """Top-level schema-only association workflow specification."""

    workflow_id: str
    sources: Sequence[TabularSourceSpec]
    outcomes: Sequence[OutcomeSpec]
    predictors: Sequence[PredictorSpec] = ()
    name: str | None = None
    description: str | None = None
    covariates: Sequence[CovariateSpec] = ()
    groupings: Sequence[GroupingSpec] = ()
    repeated_measures: RepeatedMeasuresSpec | None = None
    missing_data_policy: MissingDataPolicy = field(default_factory=MissingDataPolicy)
    duplicate_subject_policy: DuplicateSubjectPolicy = field(default_factory=DuplicateSubjectPolicy)
    nonfinite_policy: NonFinitePolicy = field(default_factory=NonFinitePolicy)
    standardization_policy: StandardizationPolicy = field(default_factory=StandardizationPolicy)
    transformation_policy: TransformationPolicy = field(default_factory=TransformationPolicy)
    methods: Sequence[AssociationMethodSpec] = ()
    families: Sequence[AssociationFamilySpec] = ()
    multiple_testing: Sequence[MultipleTestingSpec] = ()
    outputs: Sequence[AssociationOutputSpec] = ()
    handoffs: Sequence[AssociationHandoffSpec] = ()
    backend: str = BACKEND_RECORDS
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "name", _optional_text(self.name) or self.workflow_id)
        object.__setattr__(self, "description", _optional_text(self.description))
        object.__setattr__(self, "sources", tuple(_coerce_source_spec(source) for source in self.sources))
        object.__setattr__(self, "outcomes", tuple(_coerce_variable_spec(outcome, OutcomeSpec) for outcome in self.outcomes))
        object.__setattr__(
            self,
            "predictors",
            tuple(_coerce_variable_spec(predictor, PredictorSpec) for predictor in self.predictors),
        )
        object.__setattr__(
            self,
            "covariates",
            tuple(_coerce_variable_spec(covariate, CovariateSpec) for covariate in self.covariates),
        )
        object.__setattr__(
            self,
            "groupings",
            tuple(_coerce_variable_spec(grouping, GroupingSpec) for grouping in self.groupings),
        )
        if self.repeated_measures is not None and not isinstance(self.repeated_measures, RepeatedMeasuresSpec):
            object.__setattr__(self, "repeated_measures", _repeated_measures_from_mapping(self.repeated_measures))
        object.__setattr__(
            self,
            "missing_data_policy",
            _policy_from_mapping(self.missing_data_policy, MissingDataPolicy),
        )
        object.__setattr__(
            self,
            "duplicate_subject_policy",
            _policy_from_mapping(self.duplicate_subject_policy, DuplicateSubjectPolicy),
        )
        object.__setattr__(self, "nonfinite_policy", _policy_from_mapping(self.nonfinite_policy, NonFinitePolicy))
        object.__setattr__(
            self,
            "standardization_policy",
            _policy_from_mapping(self.standardization_policy, StandardizationPolicy),
        )
        object.__setattr__(
            self,
            "transformation_policy",
            _policy_from_mapping(self.transformation_policy, TransformationPolicy),
        )
        object.__setattr__(self, "methods", tuple(_coerce_method_spec(method) for method in self.methods))
        object.__setattr__(self, "families", tuple(_coerce_family_spec(family) for family in self.families))
        object.__setattr__(
            self,
            "multiple_testing",
            tuple(_coerce_multiple_testing_spec(spec) for spec in self.multiple_testing),
        )
        object.__setattr__(self, "outputs", tuple(_coerce_output_spec(output) for output in self.outputs))
        object.__setattr__(self, "handoffs", tuple(_coerce_handoff_spec(handoff) for handoff in self.handoffs))
        backend = _normalized_choice(self.backend, field_name="backend", supported=SUPPORTED_TABULAR_ASSOCIATION_BACKENDS)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AssociationPlanPreview:
    """Dry-run preview for a tabular association workflow."""

    schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    validation_rows: Sequence[Mapping[str, Any] | AssociationValidationRow]
    source_rows: Sequence[Mapping[str, Any]]
    column_rows: Sequence[Mapping[str, Any]]
    variable_rows: Sequence[Mapping[str, Any]]
    method_rows: Sequence[Mapping[str, Any]]
    family_rows: Sequence[Mapping[str, Any]]
    output_rows: Sequence[Mapping[str, Any]]
    publication_handoff_rows: Sequence[Mapping[str, Any]]
    visualization_handoff_rows: Sequence[Mapping[str, Any]]
    provenance_rows: Sequence[Mapping[str, Any] | AssociationProvenanceRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "executed", False)
        object.__setattr__(self, "plan_only", True)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "validation_rows", tuple(_json_safe(row) for row in self.validation_rows))
        object.__setattr__(self, "source_rows", tuple(_json_safe_mapping(row) for row in self.source_rows))
        object.__setattr__(self, "column_rows", tuple(_json_safe_mapping(row) for row in self.column_rows))
        object.__setattr__(self, "variable_rows", tuple(_json_safe_mapping(row) for row in self.variable_rows))
        object.__setattr__(self, "method_rows", tuple(_json_safe_mapping(row) for row in self.method_rows))
        object.__setattr__(self, "family_rows", tuple(_json_safe_mapping(row) for row in self.family_rows))
        object.__setattr__(self, "output_rows", tuple(_json_safe_mapping(row) for row in self.output_rows))
        object.__setattr__(
            self,
            "publication_handoff_rows",
            tuple(_json_safe_mapping(row) for row in self.publication_handoff_rows),
        )
        object.__setattr__(
            self,
            "visualization_handoff_rows",
            tuple(_json_safe_mapping(row) for row in self.visualization_handoff_rows),
        )
        object.__setattr__(self, "provenance_rows", tuple(_json_safe(row) for row in self.provenance_rows))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


class _TabularQcRowMixin:
    """Shared JSON/TSV conversion for QC rows."""

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)

    def to_tsv_row(self) -> dict[str, str]:
        return _tsv_safe_mapping(self.to_dict())


@dataclass(frozen=True)
class TabularAssociationRowSourceAdapterSpec:
    """Dependency-free row-source adapter plan.

    ``requested_backend`` records caller intent only. Runtime coercion in this
    module is always the standard-library ``records`` backend.
    """

    adapter_id: str = TABULAR_ASSOCIATION_ROW_SOURCE_ADAPTER_VERSION
    requested_backend: str = BACKEND_RECORDS
    runtime_backend: str = RUNTIME_BACKEND_RECORDS
    row_source_kind: str = "uninspected"
    include_input_row_index: bool = False
    input_row_index_field: str = "input_row_index"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    executed: bool = False
    plan_only: bool = True
    will_write: bool = False
    output_written: bool = False
    no_output_written: bool = True
    output_paths_written: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_id", _non_empty_text(self.adapter_id, field_name="adapter_id"))
        requested_backend = _normalized_choice(
            self.requested_backend or BACKEND_RECORDS,
            field_name="requested_backend",
            supported=SUPPORTED_TABULAR_ASSOCIATION_BACKENDS,
        )
        object.__setattr__(self, "requested_backend", requested_backend)
        object.__setattr__(self, "runtime_backend", RUNTIME_BACKEND_RECORDS)
        object.__setattr__(
            self,
            "row_source_kind",
            _non_empty_text(self.row_source_kind or "uninspected", field_name="row_source_kind"),
        )
        object.__setattr__(self, "include_input_row_index", bool(self.include_input_row_index))
        object.__setattr__(
            self,
            "input_row_index_field",
            _non_empty_text(self.input_row_index_field, field_name="input_row_index_field"),
        )
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularAssociationRowSourceQcRow(_TabularQcRowMixin):
    """QC row for standard-library row-source coercion."""

    adapter_id: str
    requested_backend: str
    runtime_backend: str
    row_source_kind: str
    status: str
    code: str
    message: str
    row_count: int
    observed_column_count: int
    include_input_row_index: bool = False
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_id", _non_empty_text(self.adapter_id, field_name="adapter_id"))
        object.__setattr__(self, "requested_backend", _non_empty_text(self.requested_backend, field_name="requested_backend"))
        object.__setattr__(self, "runtime_backend", _non_empty_text(self.runtime_backend, field_name="runtime_backend"))
        object.__setattr__(self, "row_source_kind", _non_empty_text(self.row_source_kind, field_name="row_source_kind"))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(self, "observed_column_count", int(self.observed_column_count))
        object.__setattr__(self, "include_input_row_index", bool(self.include_input_row_index))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))


@dataclass(frozen=True)
class TabularAssociationRowSourceProvenanceRow(_TabularQcRowMixin):
    """Provenance row for standard-library row-source coercion."""

    adapter_id: str
    key: str
    value: Any
    source: str = "tabular_association_row_source_adapter"

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_id", _non_empty_text(self.adapter_id, field_name="adapter_id"))
        object.__setattr__(self, "key", _non_empty_text(self.key, field_name="key"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "source", _non_empty_text(self.source, field_name="source"))


@dataclass(frozen=True)
class TabularAssociationRowSourceResult:
    """Result for no-write row-source-to-records coercion."""

    adapter_version: str
    spec: TabularAssociationRowSourceAdapterSpec
    valid: bool
    status: str
    records: Sequence[Mapping[str, Any]]
    observed_columns: Sequence[str]
    warnings: Sequence[str]
    errors: Sequence[str]
    qc_rows: Sequence[Mapping[str, Any] | TabularAssociationRowSourceQcRow]
    provenance_rows: Sequence[Mapping[str, Any] | TabularAssociationRowSourceProvenanceRow]
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False
    no_output_written: bool = True
    output_paths_written: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_version", _non_empty_text(self.adapter_version, field_name="adapter_version"))
        object.__setattr__(self, "spec", _coerce_row_source_adapter_spec(self.spec))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "records", tuple(_json_safe_mapping(row) for row in self.records))
        object.__setattr__(self, "observed_columns", _text_tuple(self.observed_columns, field_name="observed_columns"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "qc_rows", tuple(_json_safe(row) for row in self.qc_rows))
        object.__setattr__(self, "provenance_rows", tuple(_json_safe(row) for row in self.provenance_rows))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularAssociationRecordsAdapter:
    """Small standard-library records adapter for generic row sources."""

    spec: TabularAssociationRowSourceAdapterSpec = field(default_factory=TabularAssociationRowSourceAdapterSpec)

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec", _coerce_row_source_adapter_spec(self.spec))

    def to_dict(self) -> dict[str, Any]:
        return {"adapter_version": TABULAR_ASSOCIATION_ROW_SOURCE_ADAPTER_VERSION, "spec": self.spec.to_dict()}

    def plan(self) -> TabularAssociationRowSourceAdapterSpec:
        return self.spec

    def inspect(self, row_source: Any) -> TabularAssociationRowSourceResult:
        return coerce_tabular_association_records(
            row_source,
            requested_backend=self.spec.requested_backend,
            include_input_row_index=self.spec.include_input_row_index,
            input_row_index_field=self.spec.input_row_index_field,
            metadata=self.spec.metadata,
        )

    def coerce(self, row_source: Any) -> TabularAssociationRowSourceResult:
        return self.inspect(row_source)

    def iter_records(self, row_source: Any) -> Iterable[Mapping[str, Any]]:
        return iter(self.inspect(row_source).records)


@dataclass(frozen=True)
class TabularSourceInventorySpec:
    """Optional runtime source inventory configuration for QC-only loading."""

    source_id: str
    source_format: str | None = None
    path: str | None = None
    required: bool = True
    row_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        source_format = _optional_text(self.source_format)
        object.__setattr__(self, "source_format", source_format.lower() if source_format else None)
        object.__setattr__(self, "path", _optional_text(self.path))
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "row_key", _optional_text(self.row_key))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularSourceInventoryRow(_TabularQcRowMixin):
    """Inventory for one declared source after QC-only inspection."""

    workflow_id: str
    source_id: str
    source_kind: str
    source_format: str | None
    path: str | None
    requested_backend: str
    runtime_backend: str
    row_count: int | None
    observed_column_count: int
    observed_columns: Sequence[str]
    declared_columns: Sequence[str]
    declared_only_columns: Sequence[str]
    observed_only_columns: Sequence[str]
    declared_and_observed_columns: Sequence[str]
    load_status: str
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "source_kind", _non_empty_text(self.source_kind, field_name="source_kind"))
        object.__setattr__(self, "source_format", _optional_text(self.source_format))
        object.__setattr__(self, "path", _optional_text(self.path))
        object.__setattr__(self, "requested_backend", _non_empty_text(self.requested_backend, field_name="requested_backend"))
        object.__setattr__(self, "runtime_backend", _non_empty_text(self.runtime_backend, field_name="runtime_backend"))
        object.__setattr__(self, "observed_column_count", int(self.observed_column_count))
        object.__setattr__(self, "observed_columns", _text_tuple(self.observed_columns, field_name="observed_columns"))
        object.__setattr__(self, "declared_columns", _text_tuple(self.declared_columns, field_name="declared_columns"))
        object.__setattr__(
            self,
            "declared_only_columns",
            _text_tuple(self.declared_only_columns, field_name="declared_only_columns"),
        )
        object.__setattr__(
            self,
            "observed_only_columns",
            _text_tuple(self.observed_only_columns, field_name="observed_only_columns"),
        )
        object.__setattr__(
            self,
            "declared_and_observed_columns",
            _text_tuple(self.declared_and_observed_columns, field_name="declared_and_observed_columns"),
        )
        object.__setattr__(self, "load_status", _non_empty_text(self.load_status, field_name="load_status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "provenance", _json_safe_mapping(self.provenance))


@dataclass(frozen=True)
class TabularSourceLoadRow(_TabularQcRowMixin):
    """Load status for one source in QC-only mode."""

    workflow_id: str
    source_id: str
    source_kind: str
    source_format: str | None
    path: str | None
    required: bool
    load_status: str
    row_count: int | None
    observed_column_count: int
    warning_count: int
    error_count: int
    message: str
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "source_kind", _non_empty_text(self.source_kind, field_name="source_kind"))
        object.__setattr__(self, "source_format", _optional_text(self.source_format))
        object.__setattr__(self, "path", _optional_text(self.path))
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "load_status", _non_empty_text(self.load_status, field_name="load_status"))
        object.__setattr__(self, "observed_column_count", int(self.observed_column_count))
        object.__setattr__(self, "warning_count", int(self.warning_count))
        object.__setattr__(self, "error_count", int(self.error_count))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "provenance", _json_safe_mapping(self.provenance))


@dataclass(frozen=True)
class TabularColumnInventoryRow(_TabularQcRowMixin):
    """Declared/observed column inventory row."""

    workflow_id: str
    source_id: str
    column_name: str
    declared: bool
    observed: bool
    value_type: str | None = None
    role: str | None = None
    required: bool | None = None
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "column_name", _non_empty_text(self.column_name, field_name="column_name"))
        object.__setattr__(self, "declared", bool(self.declared))
        object.__setattr__(self, "observed", bool(self.observed))
        object.__setattr__(self, "value_type", _optional_text(self.value_type))
        object.__setattr__(self, "role", _optional_text(self.role))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))


@dataclass(frozen=True)
class TabularSchemaValidationRow(_TabularQcRowMixin):
    """Schema-vs-observed validation row."""

    workflow_id: str
    source_id: str
    check_name: str
    status: str
    code: str
    message: str
    column_name: str | None = None
    role: str | None = None
    required: bool = True
    observed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "check_name", _non_empty_text(self.check_name, field_name="check_name"))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        object.__setattr__(self, "column_name", _optional_text(self.column_name))
        object.__setattr__(self, "role", _optional_text(self.role))
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "observed", bool(self.observed))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))


@dataclass(frozen=True)
class TabularMissingnessRow(_TabularQcRowMixin):
    """Missing-value count for one declared column."""

    workflow_id: str
    source_id: str
    column_name: str
    role: str | None
    required: bool
    missing_count: int
    nonmissing_count: int
    total_count: int
    policy_strategy: str
    status: str
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "column_name", _non_empty_text(self.column_name, field_name="column_name"))
        object.__setattr__(self, "role", _optional_text(self.role))
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "missing_count", int(self.missing_count))
        object.__setattr__(self, "nonmissing_count", int(self.nonmissing_count))
        object.__setattr__(self, "total_count", int(self.total_count))
        object.__setattr__(self, "policy_strategy", _non_empty_text(self.policy_strategy, field_name="policy_strategy"))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))


@dataclass(frozen=True)
class TabularDuplicateRow(_TabularQcRowMixin):
    """Duplicate-key QC row."""

    workflow_id: str
    source_id: str
    key_type: str
    key_columns: Sequence[str]
    duplicate_key: str
    duplicate_count: int
    row_numbers: Sequence[int]
    policy_strategy: str
    status: str
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "key_type", _non_empty_text(self.key_type, field_name="key_type"))
        object.__setattr__(self, "key_columns", _text_tuple(self.key_columns, field_name="key_columns"))
        object.__setattr__(self, "duplicate_key", _non_empty_text(self.duplicate_key, field_name="duplicate_key"))
        object.__setattr__(self, "duplicate_count", int(self.duplicate_count))
        object.__setattr__(self, "row_numbers", tuple(int(row_number) for row_number in self.row_numbers))
        object.__setattr__(self, "policy_strategy", _non_empty_text(self.policy_strategy, field_name="policy_strategy"))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))


@dataclass(frozen=True)
class TabularNonFiniteRow(_TabularQcRowMixin):
    """Non-finite value QC row."""

    workflow_id: str
    source_id: str
    column_name: str
    role: str | None
    nonfinite_count: int
    tokens: Sequence[str]
    policy_strategy: str
    status: str
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "column_name", _non_empty_text(self.column_name, field_name="column_name"))
        object.__setattr__(self, "role", _optional_text(self.role))
        object.__setattr__(self, "nonfinite_count", int(self.nonfinite_count))
        object.__setattr__(self, "tokens", _text_tuple(self.tokens, field_name="tokens"))
        object.__setattr__(self, "policy_strategy", _non_empty_text(self.policy_strategy, field_name="policy_strategy"))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))


@dataclass(frozen=True)
class TabularCategoricalQcRow(_TabularQcRowMixin):
    """Allowed-level QC row for one categorical column."""

    workflow_id: str
    source_id: str
    column_name: str
    role: str | None
    policy: str
    allowed_values: Sequence[str]
    allow_unlisted: bool
    case_sensitive: bool
    observed_level_count: int
    unknown_level_count: int
    unknown_levels: Sequence[str]
    status: str
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "column_name", _non_empty_text(self.column_name, field_name="column_name"))
        object.__setattr__(self, "role", _optional_text(self.role))
        object.__setattr__(self, "policy", _non_empty_text(self.policy, field_name="policy"))
        object.__setattr__(self, "allowed_values", _text_tuple(self.allowed_values, field_name="allowed_values"))
        object.__setattr__(self, "allow_unlisted", bool(self.allow_unlisted))
        object.__setattr__(self, "case_sensitive", bool(self.case_sensitive))
        object.__setattr__(self, "observed_level_count", int(self.observed_level_count))
        object.__setattr__(self, "unknown_level_count", int(self.unknown_level_count))
        object.__setattr__(self, "unknown_levels", _text_tuple(self.unknown_levels, field_name="unknown_levels"))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))


@dataclass(frozen=True)
class TabularNumericQcRow(_TabularQcRowMixin):
    """Numeric-interpretable value QC row for one declared numeric column."""

    workflow_id: str
    source_id: str
    column_name: str
    role: str | None
    policy: str
    min_value: float | int | None
    max_value: float | int | None
    integer_only: bool
    total_count: int
    missing_count: int
    valid_numeric_count: int
    invalid_numeric_count: int
    bool_count: int
    nonfinite_count: int
    below_min_count: int
    above_max_count: int
    noninteger_count: int
    status: str
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "column_name", _non_empty_text(self.column_name, field_name="column_name"))
        object.__setattr__(self, "role", _optional_text(self.role))
        object.__setattr__(self, "policy", _non_empty_text(self.policy, field_name="policy"))
        object.__setattr__(self, "integer_only", bool(self.integer_only))
        for field_name in (
            "total_count",
            "missing_count",
            "valid_numeric_count",
            "invalid_numeric_count",
            "bool_count",
            "nonfinite_count",
            "below_min_count",
            "above_max_count",
            "noninteger_count",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))


@dataclass(frozen=True)
class TabularVariableQcRow(_TabularQcRowMixin):
    """Observed-column QC row for one declared association variable."""

    workflow_id: str
    variable_id: str
    variable_role: str
    source_id: str
    column_name: str
    declared_in_schema: bool
    observed: bool
    value_type: str | None
    status: str
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "variable_id", _non_empty_text(self.variable_id, field_name="variable_id"))
        object.__setattr__(self, "variable_role", _non_empty_text(self.variable_role, field_name="variable_role"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "column_name", _non_empty_text(self.column_name, field_name="column_name"))
        object.__setattr__(self, "declared_in_schema", bool(self.declared_in_schema))
        object.__setattr__(self, "observed", bool(self.observed))
        object.__setattr__(self, "value_type", _optional_text(self.value_type))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))


@dataclass(frozen=True)
class TabularAssociationQcProvenanceRow(_TabularQcRowMixin):
    """QC provenance row."""

    workflow_id: str
    key: str
    value: Any
    source: str = "tabular_association_qc"

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "key", _non_empty_text(self.key, field_name="key"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "source", _non_empty_text(self.source, field_name="source"))


@dataclass(frozen=True)
class TabularAssociationQcPlan:
    """No-write QC plan for a tabular association workflow."""

    schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    workflow_validation_rows: Sequence[Mapping[str, Any] | AssociationValidationRow]
    source_inventory_specs: Sequence[Mapping[str, Any] | TabularSourceInventorySpec]
    source_inventory_rows: Sequence[Mapping[str, Any] | TabularSourceInventoryRow]
    provenance_rows: Sequence[Mapping[str, Any] | TabularAssociationQcProvenanceRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", False)
        object.__setattr__(self, "plan_only", True)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "workflow_validation_rows", tuple(_json_safe(row) for row in self.workflow_validation_rows))
        object.__setattr__(self, "source_inventory_specs", tuple(_json_safe(row) for row in self.source_inventory_specs))
        object.__setattr__(self, "source_inventory_rows", tuple(_json_safe(row) for row in self.source_inventory_rows))
        object.__setattr__(self, "provenance_rows", tuple(_json_safe(row) for row in self.provenance_rows))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularAssociationQcResult:
    """QC-only source inventory and validation result."""

    schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    workflow_validation_rows: Sequence[Mapping[str, Any] | AssociationValidationRow]
    source_inventory_rows: Sequence[Mapping[str, Any] | TabularSourceInventoryRow]
    source_load_rows: Sequence[Mapping[str, Any] | TabularSourceLoadRow]
    column_inventory_rows: Sequence[Mapping[str, Any] | TabularColumnInventoryRow]
    schema_validation_rows: Sequence[Mapping[str, Any] | TabularSchemaValidationRow]
    variable_qc_rows: Sequence[Mapping[str, Any] | TabularVariableQcRow]
    missingness_rows: Sequence[Mapping[str, Any] | TabularMissingnessRow]
    duplicate_rows: Sequence[Mapping[str, Any] | TabularDuplicateRow]
    nonfinite_rows: Sequence[Mapping[str, Any] | TabularNonFiniteRow]
    categorical_qc_rows: Sequence[Mapping[str, Any] | TabularCategoricalQcRow]
    numeric_qc_rows: Sequence[Mapping[str, Any] | TabularNumericQcRow]
    provenance_rows: Sequence[Mapping[str, Any] | TabularAssociationQcProvenanceRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", False)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        for field_name in (
            "workflow_validation_rows",
            "source_inventory_rows",
            "source_load_rows",
            "column_inventory_rows",
            "schema_validation_rows",
            "variable_qc_rows",
            "missingness_rows",
            "duplicate_rows",
            "nonfinite_rows",
            "categorical_qc_rows",
            "numeric_qc_rows",
            "provenance_rows",
        ):
            object.__setattr__(self, field_name, tuple(_json_safe(row) for row in getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AssociationPairPlanRow(_TabularQcRowMixin):
    """Planned same-source or deferred correlation pair."""

    workflow_id: str
    pair_id: str
    method_id: str
    method_kind: str
    method_name: str
    family_id: str | None
    source_id: str | None
    outcome_id: str
    outcome_source_id: str
    outcome_column: str
    predictor_id: str
    predictor_source_id: str
    predictor_column: str
    executable: bool
    deferred: bool
    status: str
    code: str
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = False
    plan_only: bool = True
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "pair_id", _non_empty_text(self.pair_id, field_name="pair_id"))
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_kind", _non_empty_text(self.method_kind, field_name="method_kind"))
        object.__setattr__(self, "method_name", _non_empty_text(self.method_name, field_name="method_name"))
        object.__setattr__(self, "family_id", _optional_text(self.family_id))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "outcome_id", _non_empty_text(self.outcome_id, field_name="outcome_id"))
        object.__setattr__(self, "outcome_source_id", _non_empty_text(self.outcome_source_id, field_name="outcome_source_id"))
        object.__setattr__(self, "outcome_column", _non_empty_text(self.outcome_column, field_name="outcome_column"))
        object.__setattr__(self, "predictor_id", _non_empty_text(self.predictor_id, field_name="predictor_id"))
        object.__setattr__(
            self,
            "predictor_source_id",
            _non_empty_text(self.predictor_source_id, field_name="predictor_source_id"),
        )
        object.__setattr__(self, "predictor_column", _non_empty_text(self.predictor_column, field_name="predictor_column"))
        object.__setattr__(self, "executable", bool(self.executable))
        object.__setattr__(self, "deferred", bool(self.deferred))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", False)
        object.__setattr__(self, "plan_only", True)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class CorrelationAssociationResultRow(_TabularQcRowMixin):
    """Computed or deferred Pearson/Spearman association result row."""

    workflow_id: str
    pair_id: str
    method_id: str
    method_kind: str
    method_name: str
    correlation_method: str
    family_id: str | None
    source_id: str | None
    outcome_id: str
    outcome_source_id: str
    outcome_column: str
    predictor_id: str
    predictor_source_id: str
    predictor_column: str
    n_total: int
    n_used: int
    n_missing_outcome: int
    n_missing_predictor: int
    n_missing_pairwise: int
    n_nonfinite: int
    n_invalid_numeric: int
    n_bool_numeric: int
    statistic_name: str
    statistic_value: float | None
    tie_count_outcome: int = 0
    tie_count_predictor: int = 0
    tie_group_count_outcome: int = 0
    tie_group_count_predictor: int = 0
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "pair_id", _non_empty_text(self.pair_id, field_name="pair_id"))
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_kind", _non_empty_text(self.method_kind, field_name="method_kind"))
        object.__setattr__(self, "method_name", _non_empty_text(self.method_name, field_name="method_name"))
        object.__setattr__(self, "correlation_method", _non_empty_text(self.correlation_method, field_name="correlation_method"))
        object.__setattr__(self, "family_id", _optional_text(self.family_id))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "outcome_id", _non_empty_text(self.outcome_id, field_name="outcome_id"))
        object.__setattr__(self, "outcome_source_id", _non_empty_text(self.outcome_source_id, field_name="outcome_source_id"))
        object.__setattr__(self, "outcome_column", _non_empty_text(self.outcome_column, field_name="outcome_column"))
        object.__setattr__(self, "predictor_id", _non_empty_text(self.predictor_id, field_name="predictor_id"))
        object.__setattr__(
            self,
            "predictor_source_id",
            _non_empty_text(self.predictor_source_id, field_name="predictor_source_id"),
        )
        object.__setattr__(self, "predictor_column", _non_empty_text(self.predictor_column, field_name="predictor_column"))
        for field_name in (
            "n_total",
            "n_used",
            "n_missing_outcome",
            "n_missing_predictor",
            "n_missing_pairwise",
            "n_nonfinite",
            "n_invalid_numeric",
            "n_bool_numeric",
            "tie_count_outcome",
            "tie_count_predictor",
            "tie_group_count_outcome",
            "tie_group_count_predictor",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "statistic_name", _non_empty_text(self.statistic_name, field_name="statistic_name"))
        if self.statistic_value is not None:
            object.__setattr__(self, "statistic_value", _finite_number(self.statistic_value, field_name="statistic_value"))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class CorrelationComputationQcRow(_TabularQcRowMixin):
    """Computation QC for one Pearson/Spearman pair."""

    workflow_id: str
    pair_id: str
    method_id: str
    method_name: str
    source_id: str | None
    outcome_id: str
    outcome_column: str
    predictor_id: str
    predictor_column: str
    n_total: int
    n_used: int
    n_missing_outcome: int
    n_missing_predictor: int
    n_missing_pairwise: int
    n_nonfinite: int
    n_invalid_numeric: int
    n_bool_numeric: int
    tie_count_outcome: int = 0
    tie_count_predictor: int = 0
    tie_group_count_outcome: int = 0
    tie_group_count_predictor: int = 0
    status: str = "ok"
    code: str = "correlation_computed"
    message: str = "Correlation was computed."
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "pair_id", _non_empty_text(self.pair_id, field_name="pair_id"))
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_name", _non_empty_text(self.method_name, field_name="method_name"))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "outcome_id", _non_empty_text(self.outcome_id, field_name="outcome_id"))
        object.__setattr__(self, "outcome_column", _non_empty_text(self.outcome_column, field_name="outcome_column"))
        object.__setattr__(self, "predictor_id", _non_empty_text(self.predictor_id, field_name="predictor_id"))
        object.__setattr__(self, "predictor_column", _non_empty_text(self.predictor_column, field_name="predictor_column"))
        for field_name in (
            "n_total",
            "n_used",
            "n_missing_outcome",
            "n_missing_predictor",
            "n_missing_pairwise",
            "n_nonfinite",
            "n_invalid_numeric",
            "n_bool_numeric",
            "tie_count_outcome",
            "tie_count_predictor",
            "tie_group_count_outcome",
            "tie_group_count_predictor",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))


@dataclass(frozen=True)
class AssociationInputQcSummaryRow(_TabularQcRowMixin):
    """Source-level QC summary used by correlation execution."""

    workflow_id: str
    source_id: str
    requested_backend: str
    runtime_backend: str
    source_kind: str
    load_status: str
    row_count: int
    observed_column_count: int
    missingness_error_count: int = 0
    missingness_warning_count: int = 0
    numeric_error_count: int = 0
    numeric_warning_count: int = 0
    nonfinite_error_count: int = 0
    nonfinite_warning_count: int = 0
    duplicate_error_count: int = 0
    duplicate_warning_count: int = 0
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "requested_backend", _non_empty_text(self.requested_backend, field_name="requested_backend"))
        object.__setattr__(self, "runtime_backend", _non_empty_text(self.runtime_backend, field_name="runtime_backend"))
        object.__setattr__(self, "source_kind", _non_empty_text(self.source_kind, field_name="source_kind"))
        object.__setattr__(self, "load_status", _non_empty_text(self.load_status, field_name="load_status"))
        for field_name in (
            "row_count",
            "observed_column_count",
            "missingness_error_count",
            "missingness_warning_count",
            "numeric_error_count",
            "numeric_warning_count",
            "nonfinite_error_count",
            "nonfinite_warning_count",
            "duplicate_error_count",
            "duplicate_warning_count",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))


@dataclass(frozen=True)
class CorrelationMethodSummaryRow(_TabularQcRowMixin):
    """Method-level summary for correlation planning and execution."""

    workflow_id: str
    method_id: str
    method_kind: str
    method_name: str
    family_id: str | None
    outcome_count: int
    predictor_count: int
    pair_count: int
    executable_pair_count: int
    deferred_pair_count: int
    result_row_count: int = 0
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = False
    plan_only: bool = True
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_kind", _non_empty_text(self.method_kind, field_name="method_kind"))
        object.__setattr__(self, "method_name", _non_empty_text(self.method_name, field_name="method_name"))
        object.__setattr__(self, "family_id", _optional_text(self.family_id))
        for field_name in (
            "outcome_count",
            "predictor_count",
            "pair_count",
            "executable_pair_count",
            "deferred_pair_count",
            "result_row_count",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class TabularAssociationCorrelationProvenanceRow(_TabularQcRowMixin):
    """Correlation provenance row."""

    workflow_id: str
    key: str
    value: Any
    source: str = "tabular_association_correlations"

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "key", _non_empty_text(self.key, field_name="key"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "source", _non_empty_text(self.source, field_name="source"))


@dataclass(frozen=True)
class TabularAssociationCorrelationPlan:
    """No-write Pearson/Spearman association-row plan."""

    schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    workflow_validation_rows: Sequence[Mapping[str, Any] | AssociationValidationRow]
    pair_plan_rows: Sequence[Mapping[str, Any] | AssociationPairPlanRow]
    method_summary_rows: Sequence[Mapping[str, Any] | CorrelationMethodSummaryRow]
    provenance_rows: Sequence[Mapping[str, Any] | TabularAssociationCorrelationProvenanceRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", False)
        object.__setattr__(self, "plan_only", True)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "workflow_validation_rows", tuple(_json_safe(row) for row in self.workflow_validation_rows))
        object.__setattr__(self, "pair_plan_rows", tuple(_json_safe(row) for row in self.pair_plan_rows))
        object.__setattr__(self, "method_summary_rows", tuple(_json_safe(row) for row in self.method_summary_rows))
        object.__setattr__(self, "provenance_rows", tuple(_json_safe(row) for row in self.provenance_rows))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularAssociationCorrelationResult:
    """No-write Pearson/Spearman association-row execution result."""

    schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    workflow_validation_rows: Sequence[Mapping[str, Any] | AssociationValidationRow]
    pair_plan_rows: Sequence[Mapping[str, Any] | AssociationPairPlanRow]
    source_inventory_rows: Sequence[Mapping[str, Any] | TabularSourceInventoryRow]
    source_load_rows: Sequence[Mapping[str, Any] | TabularSourceLoadRow]
    input_qc_summary_rows: Sequence[Mapping[str, Any] | AssociationInputQcSummaryRow]
    computation_qc_rows: Sequence[Mapping[str, Any] | CorrelationComputationQcRow]
    result_rows: Sequence[Mapping[str, Any] | CorrelationAssociationResultRow]
    method_summary_rows: Sequence[Mapping[str, Any] | CorrelationMethodSummaryRow]
    provenance_rows: Sequence[Mapping[str, Any] | TabularAssociationCorrelationProvenanceRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        for field_name in (
            "workflow_validation_rows",
            "pair_plan_rows",
            "source_inventory_rows",
            "source_load_rows",
            "input_qc_summary_rows",
            "computation_qc_rows",
            "result_rows",
            "method_summary_rows",
            "provenance_rows",
        ):
            object.__setattr__(self, field_name, tuple(_json_safe(row) for row in getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AdjustedAssociationPairPlanRow(_TabularQcRowMixin):
    """Planned same-source or deferred adjusted/regression association pair."""

    workflow_id: str
    pair_id: str
    method_id: str
    method_kind: str
    method_name: str
    family_id: str | None
    source_id: str | None
    outcome_id: str
    outcome_source_id: str
    outcome_column: str
    predictor_id: str
    predictor_source_id: str
    predictor_column: str
    covariate_ids: Sequence[str]
    covariate_source_ids: Sequence[str]
    covariate_columns: Sequence[str]
    covariate_count: int
    executable: bool
    deferred: bool
    status: str
    code: str
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = False
    plan_only: bool = True
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "pair_id", _non_empty_text(self.pair_id, field_name="pair_id"))
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_kind", _non_empty_text(self.method_kind, field_name="method_kind"))
        object.__setattr__(self, "method_name", _non_empty_text(self.method_name, field_name="method_name"))
        object.__setattr__(self, "family_id", _optional_text(self.family_id))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "outcome_id", _non_empty_text(self.outcome_id, field_name="outcome_id"))
        object.__setattr__(self, "outcome_source_id", _non_empty_text(self.outcome_source_id, field_name="outcome_source_id"))
        object.__setattr__(self, "outcome_column", _non_empty_text(self.outcome_column, field_name="outcome_column"))
        object.__setattr__(self, "predictor_id", _non_empty_text(self.predictor_id, field_name="predictor_id"))
        object.__setattr__(
            self,
            "predictor_source_id",
            _non_empty_text(self.predictor_source_id, field_name="predictor_source_id"),
        )
        object.__setattr__(self, "predictor_column", _non_empty_text(self.predictor_column, field_name="predictor_column"))
        object.__setattr__(self, "covariate_ids", _text_tuple(self.covariate_ids, field_name="covariate_ids"))
        object.__setattr__(
            self,
            "covariate_source_ids",
            _text_tuple(self.covariate_source_ids, field_name="covariate_source_ids"),
        )
        object.__setattr__(self, "covariate_columns", _text_tuple(self.covariate_columns, field_name="covariate_columns"))
        object.__setattr__(self, "covariate_count", int(self.covariate_count))
        object.__setattr__(self, "executable", bool(self.executable))
        object.__setattr__(self, "deferred", bool(self.deferred))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", False)
        object.__setattr__(self, "plan_only", True)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AdjustedAssociationResultRow(_TabularQcRowMixin):
    """Computed or deferred residualized Pearson-style partial association row."""

    workflow_id: str
    pair_id: str
    method_id: str
    method_kind: str
    method_name: str
    family_id: str | None
    source_id: str | None
    outcome_id: str
    outcome_source_id: str
    outcome_column: str
    predictor_id: str
    predictor_source_id: str
    predictor_column: str
    covariate_ids: Sequence[str]
    covariate_columns: Sequence[str]
    covariate_count: int
    n_total: int
    n_used: int
    n_missing_outcome: int
    n_missing_predictor: int
    n_missing_covariates: int
    n_missing_listwise: int
    n_nonfinite: int
    n_invalid_numeric: int
    n_bool_numeric: int
    statistic_name: str
    statistic_value: float | None
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "pair_id", _non_empty_text(self.pair_id, field_name="pair_id"))
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_kind", _non_empty_text(self.method_kind, field_name="method_kind"))
        object.__setattr__(self, "method_name", _non_empty_text(self.method_name, field_name="method_name"))
        object.__setattr__(self, "family_id", _optional_text(self.family_id))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "outcome_id", _non_empty_text(self.outcome_id, field_name="outcome_id"))
        object.__setattr__(self, "outcome_source_id", _non_empty_text(self.outcome_source_id, field_name="outcome_source_id"))
        object.__setattr__(self, "outcome_column", _non_empty_text(self.outcome_column, field_name="outcome_column"))
        object.__setattr__(self, "predictor_id", _non_empty_text(self.predictor_id, field_name="predictor_id"))
        object.__setattr__(
            self,
            "predictor_source_id",
            _non_empty_text(self.predictor_source_id, field_name="predictor_source_id"),
        )
        object.__setattr__(self, "predictor_column", _non_empty_text(self.predictor_column, field_name="predictor_column"))
        object.__setattr__(self, "covariate_ids", _text_tuple(self.covariate_ids, field_name="covariate_ids"))
        object.__setattr__(self, "covariate_columns", _text_tuple(self.covariate_columns, field_name="covariate_columns"))
        for field_name in (
            "covariate_count",
            "n_total",
            "n_used",
            "n_missing_outcome",
            "n_missing_predictor",
            "n_missing_covariates",
            "n_missing_listwise",
            "n_nonfinite",
            "n_invalid_numeric",
            "n_bool_numeric",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "statistic_name", _non_empty_text(self.statistic_name, field_name="statistic_name"))
        if self.statistic_value is not None:
            object.__setattr__(self, "statistic_value", _finite_number(self.statistic_value, field_name="statistic_value"))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class RegressionAssociationResultRow(_TabularQcRowMixin):
    """Computed or deferred OLS primary-predictor association row."""

    workflow_id: str
    pair_id: str
    method_id: str
    method_kind: str
    method_name: str
    family_id: str | None
    source_id: str | None
    outcome_id: str
    outcome_source_id: str
    outcome_column: str
    predictor_id: str
    predictor_source_id: str
    predictor_column: str
    covariate_ids: Sequence[str]
    covariate_columns: Sequence[str]
    covariate_count: int
    model_parameter_count: int
    residual_degrees_of_freedom: int
    n_total: int
    n_used: int
    n_missing_outcome: int
    n_missing_predictor: int
    n_missing_covariates: int
    n_missing_listwise: int
    n_nonfinite: int
    n_invalid_numeric: int
    n_bool_numeric: int
    statistic_name: str
    statistic_value: float | None
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "pair_id", _non_empty_text(self.pair_id, field_name="pair_id"))
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_kind", _non_empty_text(self.method_kind, field_name="method_kind"))
        object.__setattr__(self, "method_name", _non_empty_text(self.method_name, field_name="method_name"))
        object.__setattr__(self, "family_id", _optional_text(self.family_id))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "outcome_id", _non_empty_text(self.outcome_id, field_name="outcome_id"))
        object.__setattr__(self, "outcome_source_id", _non_empty_text(self.outcome_source_id, field_name="outcome_source_id"))
        object.__setattr__(self, "outcome_column", _non_empty_text(self.outcome_column, field_name="outcome_column"))
        object.__setattr__(self, "predictor_id", _non_empty_text(self.predictor_id, field_name="predictor_id"))
        object.__setattr__(
            self,
            "predictor_source_id",
            _non_empty_text(self.predictor_source_id, field_name="predictor_source_id"),
        )
        object.__setattr__(self, "predictor_column", _non_empty_text(self.predictor_column, field_name="predictor_column"))
        object.__setattr__(self, "covariate_ids", _text_tuple(self.covariate_ids, field_name="covariate_ids"))
        object.__setattr__(self, "covariate_columns", _text_tuple(self.covariate_columns, field_name="covariate_columns"))
        for field_name in (
            "covariate_count",
            "model_parameter_count",
            "residual_degrees_of_freedom",
            "n_total",
            "n_used",
            "n_missing_outcome",
            "n_missing_predictor",
            "n_missing_covariates",
            "n_missing_listwise",
            "n_nonfinite",
            "n_invalid_numeric",
            "n_bool_numeric",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "statistic_name", _non_empty_text(self.statistic_name, field_name="statistic_name"))
        if self.statistic_value is not None:
            object.__setattr__(self, "statistic_value", _finite_number(self.statistic_value, field_name="statistic_value"))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AdjustedAssociationComputationQcRow(_TabularQcRowMixin):
    """Computation QC for one adjusted/regression association pair."""

    workflow_id: str
    pair_id: str
    method_id: str
    method_kind: str
    method_name: str
    source_id: str | None
    outcome_id: str
    outcome_column: str
    predictor_id: str
    predictor_column: str
    covariate_ids: Sequence[str]
    covariate_columns: Sequence[str]
    covariate_count: int
    n_total: int
    n_used: int
    n_missing_outcome: int
    n_missing_predictor: int
    n_missing_covariates: int
    n_missing_listwise: int
    n_nonfinite: int
    n_invalid_numeric: int
    n_bool_numeric: int
    model_parameter_count: int | None = None
    residual_degrees_of_freedom: int | None = None
    status: str = "ok"
    code: str = "adjusted_association_computed"
    message: str = "Adjusted association was computed."
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "pair_id", _non_empty_text(self.pair_id, field_name="pair_id"))
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_kind", _non_empty_text(self.method_kind, field_name="method_kind"))
        object.__setattr__(self, "method_name", _non_empty_text(self.method_name, field_name="method_name"))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "outcome_id", _non_empty_text(self.outcome_id, field_name="outcome_id"))
        object.__setattr__(self, "outcome_column", _non_empty_text(self.outcome_column, field_name="outcome_column"))
        object.__setattr__(self, "predictor_id", _non_empty_text(self.predictor_id, field_name="predictor_id"))
        object.__setattr__(self, "predictor_column", _non_empty_text(self.predictor_column, field_name="predictor_column"))
        object.__setattr__(self, "covariate_ids", _text_tuple(self.covariate_ids, field_name="covariate_ids"))
        object.__setattr__(self, "covariate_columns", _text_tuple(self.covariate_columns, field_name="covariate_columns"))
        for field_name in (
            "covariate_count",
            "n_total",
            "n_used",
            "n_missing_outcome",
            "n_missing_predictor",
            "n_missing_covariates",
            "n_missing_listwise",
            "n_nonfinite",
            "n_invalid_numeric",
            "n_bool_numeric",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        if self.model_parameter_count is not None:
            object.__setattr__(self, "model_parameter_count", int(self.model_parameter_count))
        if self.residual_degrees_of_freedom is not None:
            object.__setattr__(self, "residual_degrees_of_freedom", int(self.residual_degrees_of_freedom))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))


@dataclass(frozen=True)
class AdjustedAssociationMethodSummaryRow(_TabularQcRowMixin):
    """Method-level summary for adjusted/regression planning and execution."""

    workflow_id: str
    method_id: str
    method_kind: str
    method_name: str
    family_id: str | None
    outcome_count: int
    predictor_count: int
    covariate_count: int
    pair_count: int
    executable_pair_count: int
    deferred_pair_count: int
    result_row_count: int = 0
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = False
    plan_only: bool = True
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_kind", _non_empty_text(self.method_kind, field_name="method_kind"))
        object.__setattr__(self, "method_name", _non_empty_text(self.method_name, field_name="method_name"))
        object.__setattr__(self, "family_id", _optional_text(self.family_id))
        for field_name in (
            "outcome_count",
            "predictor_count",
            "covariate_count",
            "pair_count",
            "executable_pair_count",
            "deferred_pair_count",
            "result_row_count",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class TabularAssociationAdjustedProvenanceRow(_TabularQcRowMixin):
    """Adjusted/regression association provenance row."""

    workflow_id: str
    key: str
    value: Any
    source: str = "tabular_association_adjusted"

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "key", _non_empty_text(self.key, field_name="key"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "source", _non_empty_text(self.source, field_name="source"))


@dataclass(frozen=True)
class TabularAssociationAdjustedPlan:
    """No-write partial/regression association-row plan."""

    schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    workflow_validation_rows: Sequence[Mapping[str, Any] | AssociationValidationRow]
    pair_plan_rows: Sequence[Mapping[str, Any] | AdjustedAssociationPairPlanRow]
    method_summary_rows: Sequence[Mapping[str, Any] | AdjustedAssociationMethodSummaryRow]
    provenance_rows: Sequence[Mapping[str, Any] | TabularAssociationAdjustedProvenanceRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", False)
        object.__setattr__(self, "plan_only", True)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "workflow_validation_rows", tuple(_json_safe(row) for row in self.workflow_validation_rows))
        object.__setattr__(self, "pair_plan_rows", tuple(_json_safe(row) for row in self.pair_plan_rows))
        object.__setattr__(self, "method_summary_rows", tuple(_json_safe(row) for row in self.method_summary_rows))
        object.__setattr__(self, "provenance_rows", tuple(_json_safe(row) for row in self.provenance_rows))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularAssociationAdjustedResult:
    """No-write partial/regression association-row execution result."""

    schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    workflow_validation_rows: Sequence[Mapping[str, Any] | AssociationValidationRow]
    pair_plan_rows: Sequence[Mapping[str, Any] | AdjustedAssociationPairPlanRow]
    source_inventory_rows: Sequence[Mapping[str, Any] | TabularSourceInventoryRow]
    source_load_rows: Sequence[Mapping[str, Any] | TabularSourceLoadRow]
    input_qc_summary_rows: Sequence[Mapping[str, Any] | AssociationInputQcSummaryRow]
    computation_qc_rows: Sequence[Mapping[str, Any] | AdjustedAssociationComputationQcRow]
    result_rows: Sequence[Mapping[str, Any] | AdjustedAssociationResultRow | RegressionAssociationResultRow]
    adjusted_result_rows: Sequence[Mapping[str, Any] | AdjustedAssociationResultRow]
    regression_result_rows: Sequence[Mapping[str, Any] | RegressionAssociationResultRow]
    method_summary_rows: Sequence[Mapping[str, Any] | AdjustedAssociationMethodSummaryRow]
    provenance_rows: Sequence[Mapping[str, Any] | TabularAssociationAdjustedProvenanceRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        for field_name in (
            "workflow_validation_rows",
            "pair_plan_rows",
            "source_inventory_rows",
            "source_load_rows",
            "input_qc_summary_rows",
            "computation_qc_rows",
            "result_rows",
            "adjusted_result_rows",
            "regression_result_rows",
            "method_summary_rows",
            "provenance_rows",
        ):
            object.__setattr__(self, field_name, tuple(_json_safe(row) for row in getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RepeatedMeasuresModelPlanRow(_TabularQcRowMixin):
    """Metadata-only repeated-measures or mixed-model plan row."""

    workflow_id: str
    model_plan_id: str
    source_id: str | None
    method_id: str
    method_name: str
    method_kind: str
    family_id: str | None
    outcome_id: str
    outcome_source_id: str
    outcome_column: str
    predictor_id: str
    predictor_source_id: str
    predictor_column: str
    covariate_ids: Sequence[str]
    covariate_source_ids: Sequence[str]
    covariate_columns: Sequence[str]
    group_id: str | None
    group_column: str | None
    group_ids: Sequence[str]
    group_columns: Sequence[str]
    subject_id_column: str | None
    session_column: str | None
    timepoint_column: str | None
    repeated_unit_columns: Sequence[str]
    repeated_factor_columns: Sequence[str]
    cluster_columns: Sequence[str]
    fixed_effect_term_ids: Sequence[str]
    fixed_effect_metadata: Mapping[str, Any]
    random_effect_metadata: Mapping[str, Any]
    formula_metadata: Mapping[str, Any]
    method_metadata: Mapping[str, Any]
    repeated_measures_metadata: Mapping[str, Any]
    model_design_id: str | None = None
    random_effect_term_ids: Sequence[str] = ()
    random_intercept_ids: Sequence[str] = ()
    random_slope_ids: Sequence[str] = ()
    repeated_factor_ids: Sequence[str] = ()
    within_subject_factor_ids: Sequence[str] = ()
    within_subject_factor_columns: Sequence[str] = ()
    between_subject_factor_ids: Sequence[str] = ()
    between_subject_factor_columns: Sequence[str] = ()
    grouping_factor_ids: Sequence[str] = ()
    grouping_factor_columns: Sequence[str] = ()
    cluster_term_ids: Sequence[str] = ()
    timepoint_role_ids: Sequence[str] = ()
    timepoint_columns: Sequence[str] = ()
    categorical_coding_ids: Sequence[str] = ()
    formula_like: str | None = None
    planned_comparison_ids: Sequence[str] = ()
    planned_comparison_metadata: Mapping[str, Any] = field(default_factory=dict)
    contrast_metadata_ids: Sequence[str] = ()
    contrast_metadata: Mapping[str, Any] = field(default_factory=dict)
    model_family: str | None = None
    link_function: str | None = None
    metadata_only: bool = True
    model_fitting_deferred: bool = True
    runtime_backend: str = RUNTIME_BACKEND_RECORDS
    executable: bool = False
    deferred: bool = True
    status: str = "deferred"
    code: str = "model_fitting_deferred"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = False
    plan_only: bool = True
    will_write: bool = False
    output_written: bool = False
    no_output_written: bool = True
    output_paths_written: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "model_plan_id", _non_empty_text(self.model_plan_id, field_name="model_plan_id"))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_name", _non_empty_text(self.method_name, field_name="method_name"))
        object.__setattr__(self, "method_kind", _non_empty_text(self.method_kind, field_name="method_kind"))
        object.__setattr__(self, "family_id", _optional_text(self.family_id))
        object.__setattr__(self, "outcome_id", _non_empty_text(self.outcome_id, field_name="outcome_id"))
        object.__setattr__(self, "outcome_source_id", _non_empty_text(self.outcome_source_id, field_name="outcome_source_id"))
        object.__setattr__(self, "outcome_column", _non_empty_text(self.outcome_column, field_name="outcome_column"))
        object.__setattr__(self, "predictor_id", _non_empty_text(self.predictor_id, field_name="predictor_id"))
        object.__setattr__(
            self,
            "predictor_source_id",
            _non_empty_text(self.predictor_source_id, field_name="predictor_source_id"),
        )
        object.__setattr__(self, "predictor_column", _non_empty_text(self.predictor_column, field_name="predictor_column"))
        object.__setattr__(self, "covariate_ids", _text_tuple(self.covariate_ids, field_name="covariate_ids"))
        object.__setattr__(
            self,
            "covariate_source_ids",
            _text_tuple(self.covariate_source_ids, field_name="covariate_source_ids"),
        )
        object.__setattr__(self, "covariate_columns", _text_tuple(self.covariate_columns, field_name="covariate_columns"))
        object.__setattr__(self, "group_id", _optional_text(self.group_id))
        object.__setattr__(self, "group_column", _optional_text(self.group_column))
        object.__setattr__(self, "group_ids", _text_tuple(self.group_ids, field_name="group_ids"))
        object.__setattr__(self, "group_columns", _text_tuple(self.group_columns, field_name="group_columns"))
        object.__setattr__(self, "subject_id_column", _optional_text(self.subject_id_column))
        object.__setattr__(self, "session_column", _optional_text(self.session_column))
        object.__setattr__(self, "timepoint_column", _optional_text(self.timepoint_column))
        object.__setattr__(
            self,
            "repeated_unit_columns",
            _text_tuple(self.repeated_unit_columns, field_name="repeated_unit_columns"),
        )
        object.__setattr__(
            self,
            "repeated_factor_columns",
            _text_tuple(self.repeated_factor_columns, field_name="repeated_factor_columns"),
        )
        object.__setattr__(self, "cluster_columns", _text_tuple(self.cluster_columns, field_name="cluster_columns"))
        object.__setattr__(
            self,
            "fixed_effect_term_ids",
            _text_tuple(self.fixed_effect_term_ids, field_name="fixed_effect_term_ids"),
        )
        object.__setattr__(self, "fixed_effect_metadata", _json_safe_mapping(self.fixed_effect_metadata))
        object.__setattr__(self, "random_effect_metadata", _json_safe_mapping(self.random_effect_metadata))
        object.__setattr__(self, "formula_metadata", _json_safe_mapping(self.formula_metadata))
        object.__setattr__(self, "method_metadata", _json_safe_mapping(self.method_metadata))
        object.__setattr__(self, "repeated_measures_metadata", _json_safe_mapping(self.repeated_measures_metadata))
        object.__setattr__(self, "model_design_id", _optional_text(self.model_design_id))
        object.__setattr__(
            self,
            "random_effect_term_ids",
            _text_tuple(self.random_effect_term_ids, field_name="random_effect_term_ids"),
        )
        object.__setattr__(
            self,
            "random_intercept_ids",
            _text_tuple(self.random_intercept_ids, field_name="random_intercept_ids"),
        )
        object.__setattr__(self, "random_slope_ids", _text_tuple(self.random_slope_ids, field_name="random_slope_ids"))
        object.__setattr__(self, "repeated_factor_ids", _text_tuple(self.repeated_factor_ids, field_name="repeated_factor_ids"))
        object.__setattr__(
            self,
            "within_subject_factor_ids",
            _text_tuple(self.within_subject_factor_ids, field_name="within_subject_factor_ids"),
        )
        object.__setattr__(
            self,
            "within_subject_factor_columns",
            _text_tuple(self.within_subject_factor_columns, field_name="within_subject_factor_columns"),
        )
        object.__setattr__(
            self,
            "between_subject_factor_ids",
            _text_tuple(self.between_subject_factor_ids, field_name="between_subject_factor_ids"),
        )
        object.__setattr__(
            self,
            "between_subject_factor_columns",
            _text_tuple(self.between_subject_factor_columns, field_name="between_subject_factor_columns"),
        )
        object.__setattr__(
            self,
            "grouping_factor_ids",
            _text_tuple(self.grouping_factor_ids, field_name="grouping_factor_ids"),
        )
        object.__setattr__(
            self,
            "grouping_factor_columns",
            _text_tuple(self.grouping_factor_columns, field_name="grouping_factor_columns"),
        )
        object.__setattr__(self, "cluster_term_ids", _text_tuple(self.cluster_term_ids, field_name="cluster_term_ids"))
        object.__setattr__(self, "timepoint_role_ids", _text_tuple(self.timepoint_role_ids, field_name="timepoint_role_ids"))
        object.__setattr__(self, "timepoint_columns", _text_tuple(self.timepoint_columns, field_name="timepoint_columns"))
        object.__setattr__(
            self,
            "categorical_coding_ids",
            _text_tuple(self.categorical_coding_ids, field_name="categorical_coding_ids"),
        )
        object.__setattr__(self, "formula_like", _optional_text(self.formula_like))
        object.__setattr__(
            self,
            "planned_comparison_ids",
            _text_tuple(self.planned_comparison_ids, field_name="planned_comparison_ids"),
        )
        object.__setattr__(
            self,
            "planned_comparison_metadata",
            _json_safe_mapping(self.planned_comparison_metadata),
        )
        object.__setattr__(
            self,
            "contrast_metadata_ids",
            _text_tuple(self.contrast_metadata_ids, field_name="contrast_metadata_ids"),
        )
        object.__setattr__(self, "contrast_metadata", _json_safe_mapping(self.contrast_metadata))
        object.__setattr__(self, "model_family", _optional_text(self.model_family))
        object.__setattr__(self, "link_function", _optional_text(self.link_function))
        object.__setattr__(self, "metadata_only", True)
        object.__setattr__(self, "model_fitting_deferred", True)
        object.__setattr__(self, "runtime_backend", RUNTIME_BACKEND_RECORDS)
        object.__setattr__(self, "executable", False)
        object.__setattr__(self, "deferred", True)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", False)
        object.__setattr__(self, "plan_only", True)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )


@dataclass(frozen=True)
class RepeatedMeasuresDesignSummaryRow(_TabularQcRowMixin):
    """Long-format repeated-measures design summary for one source/method."""

    workflow_id: str
    source_id: str
    method_id: str
    method_name: str
    row_count: int
    observation_count: int
    participant_count: int
    cluster_count: int
    min_observations_per_participant: int
    max_observations_per_participant: int
    singleton_participant_count: int
    insufficient_repeat_participant_count: int
    duplicate_repeated_unit_count: int
    missing_subject_id_count: int
    missing_repeated_key_count: int
    balanced_design: bool
    imbalance_indicator: str | None
    subject_id_column: str | None
    repeated_unit_columns: Sequence[str]
    repeated_factor_columns: Sequence[str]
    cluster_columns: Sequence[str]
    runtime_backend: str = RUNTIME_BACKEND_RECORDS
    status: str = "ok"
    code: str = "repeated_measures_design_summarized"
    message: str = "Repeated-measures design was summarized without fitting a model."
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False
    no_output_written: bool = True
    output_paths_written: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_name", _non_empty_text(self.method_name, field_name="method_name"))
        for field_name in (
            "row_count",
            "observation_count",
            "participant_count",
            "cluster_count",
            "min_observations_per_participant",
            "max_observations_per_participant",
            "singleton_participant_count",
            "insufficient_repeat_participant_count",
            "duplicate_repeated_unit_count",
            "missing_subject_id_count",
            "missing_repeated_key_count",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "balanced_design", bool(self.balanced_design))
        object.__setattr__(self, "imbalance_indicator", _optional_text(self.imbalance_indicator))
        object.__setattr__(self, "subject_id_column", _optional_text(self.subject_id_column))
        object.__setattr__(
            self,
            "repeated_unit_columns",
            _text_tuple(self.repeated_unit_columns, field_name="repeated_unit_columns"),
        )
        object.__setattr__(
            self,
            "repeated_factor_columns",
            _text_tuple(self.repeated_factor_columns, field_name="repeated_factor_columns"),
        )
        object.__setattr__(self, "cluster_columns", _text_tuple(self.cluster_columns, field_name="cluster_columns"))
        object.__setattr__(self, "runtime_backend", RUNTIME_BACKEND_RECORDS)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )


@dataclass(frozen=True)
class RepeatedMeasuresFactorSummaryRow(_TabularQcRowMixin):
    """Level summary for a timepoint or repeated-factor column."""

    workflow_id: str
    source_id: str
    method_id: str
    method_name: str
    factor_column: str
    level_count: int
    levels: Sequence[str]
    observations_by_level: Mapping[str, int]
    participants_by_level: Mapping[str, int]
    missing_count: int
    runtime_backend: str = RUNTIME_BACKEND_RECORDS
    status: str = "ok"
    code: str = "repeated_factor_summarized"
    message: str = "Repeated-factor levels were summarized without fitting a model."
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False
    no_output_written: bool = True
    output_paths_written: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _non_empty_text(self.source_id, field_name="source_id"))
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, field_name="method_id"))
        object.__setattr__(self, "method_name", _non_empty_text(self.method_name, field_name="method_name"))
        object.__setattr__(self, "factor_column", _non_empty_text(self.factor_column, field_name="factor_column"))
        object.__setattr__(self, "level_count", int(self.level_count))
        object.__setattr__(self, "levels", _text_tuple(self.levels, field_name="levels"))
        object.__setattr__(
            self,
            "observations_by_level",
            {str(key): int(value) for key, value in self.observations_by_level.items()},
        )
        object.__setattr__(
            self,
            "participants_by_level",
            {str(key): int(value) for key, value in self.participants_by_level.items()},
        )
        object.__setattr__(self, "missing_count", int(self.missing_count))
        object.__setattr__(self, "runtime_backend", RUNTIME_BACKEND_RECORDS)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )


@dataclass(frozen=True)
class RepeatedMeasuresDesignQcRow(_TabularQcRowMixin):
    """QC row for repeated-measures design planning."""

    workflow_id: str
    source_id: str | None
    method_id: str | None
    method_name: str | None
    model_plan_id: str | None
    runtime_backend: str
    status: str
    code: str
    message: str
    row_count: int = 0
    participant_count: int = 0
    duplicate_repeated_unit_count: int = 0
    missing_subject_id_count: int = 0
    missing_repeated_key_count: int = 0
    singleton_participant_count: int = 0
    insufficient_repeat_participant_count: int = 0
    model_fitting_deferred: bool = True
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False
    no_output_written: bool = True
    output_paths_written: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "method_id", _optional_text(self.method_id))
        object.__setattr__(self, "method_name", _optional_text(self.method_name))
        object.__setattr__(self, "model_plan_id", _optional_text(self.model_plan_id))
        object.__setattr__(self, "runtime_backend", RUNTIME_BACKEND_RECORDS)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        for field_name in (
            "row_count",
            "participant_count",
            "duplicate_repeated_unit_count",
            "missing_subject_id_count",
            "missing_repeated_key_count",
            "singleton_participant_count",
            "insufficient_repeat_participant_count",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "model_fitting_deferred", True)
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )


@dataclass(frozen=True)
class TabularAssociationRepeatedMeasuresProvenanceRow(_TabularQcRowMixin):
    """Provenance row for repeated-measures design planning."""

    workflow_id: str
    source_id: str | None
    method_id: str | None
    runtime_backend: str
    step_version: str
    model_fitting_deferred: bool
    will_write: bool
    output_written: bool
    no_output_written: bool
    output_paths_written: Sequence[str]
    key: str
    value: Any
    source: str = "tabular_association_repeated_measures_design_qc"

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "method_id", _optional_text(self.method_id))
        object.__setattr__(self, "runtime_backend", RUNTIME_BACKEND_RECORDS)
        object.__setattr__(
            self,
            "step_version",
            _non_empty_text(self.step_version, field_name="step_version"),
        )
        object.__setattr__(self, "model_fitting_deferred", True)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )
        object.__setattr__(self, "key", _non_empty_text(self.key, field_name="key"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "source", _non_empty_text(self.source, field_name="source"))


@dataclass(frozen=True)
class TabularAssociationRepeatedMeasuresPlan:
    """No-write repeated-measures/mixed-model design plan."""

    schema_version: str
    repeated_measures_plan_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    no_output_written: bool
    output_paths_written: Sequence[str]
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    workflow_validation_rows: Sequence[Mapping[str, Any] | AssociationValidationRow]
    model_plan_rows: Sequence[Mapping[str, Any] | RepeatedMeasuresModelPlanRow]
    provenance_rows: Sequence[Mapping[str, Any] | TabularAssociationRepeatedMeasuresProvenanceRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(
            self,
            "repeated_measures_plan_version",
            _non_empty_text(self.repeated_measures_plan_version, field_name="repeated_measures_plan_version"),
        )
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", False)
        object.__setattr__(self, "plan_only", True)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "workflow_validation_rows", tuple(_json_safe(row) for row in self.workflow_validation_rows))
        object.__setattr__(self, "model_plan_rows", tuple(_json_safe(row) for row in self.model_plan_rows))
        object.__setattr__(self, "provenance_rows", tuple(_json_safe(row) for row in self.provenance_rows))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularAssociationRepeatedMeasuresDesignQcResult:
    """No-write repeated-measures/mixed-model design QC result."""

    schema_version: str
    repeated_measures_plan_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    no_output_written: bool
    output_paths_written: Sequence[str]
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    workflow_validation_rows: Sequence[Mapping[str, Any] | AssociationValidationRow]
    model_plan_rows: Sequence[Mapping[str, Any] | RepeatedMeasuresModelPlanRow]
    source_inventory_rows: Sequence[Mapping[str, Any] | TabularSourceInventoryRow]
    source_load_rows: Sequence[Mapping[str, Any] | TabularSourceLoadRow]
    design_summary_rows: Sequence[Mapping[str, Any] | RepeatedMeasuresDesignSummaryRow]
    factor_summary_rows: Sequence[Mapping[str, Any] | RepeatedMeasuresFactorSummaryRow]
    qc_rows: Sequence[Mapping[str, Any] | RepeatedMeasuresDesignQcRow]
    provenance_rows: Sequence[Mapping[str, Any] | TabularAssociationRepeatedMeasuresProvenanceRow]
    model_fitting_deferred: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(
            self,
            "repeated_measures_plan_version",
            _non_empty_text(self.repeated_measures_plan_version, field_name="repeated_measures_plan_version"),
        )
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        for field_name in (
            "workflow_validation_rows",
            "model_plan_rows",
            "source_inventory_rows",
            "source_load_rows",
            "design_summary_rows",
            "factor_summary_rows",
            "qc_rows",
            "provenance_rows",
        ):
            object.__setattr__(self, field_name, tuple(_json_safe(row) for row in getattr(self, field_name)))
        object.__setattr__(self, "model_fitting_deferred", True)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class _ModelResultRowBase(_TabularQcRowMixin):
    """Supplied-only model result row shared by Step 11J-C row contracts."""

    _ROW_KIND: ClassVar[str | None] = None

    workflow_id: str | None = None
    result_row_id: str | None = None
    result_id: str | None = None
    result_kind: str = "model_result"
    model_id: str | None = None
    model_plan_id: str | None = None
    method_id: str | None = None
    method_name: str | None = None
    method_kind: str | None = None
    family_id: str | None = None
    source_id: str | None = None
    outcome_id: str | None = None
    outcome_column: str | None = None
    predictor_id: str | None = None
    predictor_column: str | None = None
    covariate_ids: Sequence[str] = ()
    covariate_columns: Sequence[str] = ()
    term_id: str | None = None
    term_label: str | None = None
    comparison_id: str | None = None
    contrast_id: str | None = None
    grouping_id: str | None = None
    cluster_id: str | None = None
    statistic_name: str | None = None
    statistic_value: float | int | None = None
    coefficient: float | int | None = None
    standard_error: float | int | None = None
    p_value: float | int | None = None
    q_value: float | int | None = None
    ci_low: float | int | None = None
    ci_high: float | int | None = None
    confidence_level: float | int | None = None
    effect_size: float | int | None = None
    effect_size_name: str | None = None
    degrees_of_freedom: float | int | None = None
    model_fit_metric_name: str | None = None
    model_fit_metric_value: float | int | None = None
    observation_count: int | None = None
    participant_count: int | None = None
    cluster_count: int | None = None
    status: str = "supplied"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime_backend: str = RUNTIME_BACKEND_RECORDS
    supplied_only: bool = True
    computed_by_research_analysis: bool = False
    model_fitting_performed: bool = False
    will_write: bool = False
    output_written: bool = False
    no_output_written: bool = True
    output_paths_written: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_kind",
            _non_empty_text(self._ROW_KIND or self.result_kind, field_name="result_kind"),
        )
        for field_name in (
            "workflow_id",
            "result_row_id",
            "result_id",
            "model_id",
            "model_plan_id",
            "method_id",
            "method_name",
            "method_kind",
            "family_id",
            "source_id",
            "outcome_id",
            "outcome_column",
            "predictor_id",
            "predictor_column",
            "term_id",
            "term_label",
            "comparison_id",
            "contrast_id",
            "grouping_id",
            "cluster_id",
            "statistic_name",
            "effect_size_name",
            "model_fit_metric_name",
        ):
            object.__setattr__(self, field_name, _optional_text(getattr(self, field_name)))
        object.__setattr__(self, "covariate_ids", _text_tuple(self.covariate_ids, field_name="covariate_ids"))
        object.__setattr__(self, "covariate_columns", _text_tuple(self.covariate_columns, field_name="covariate_columns"))
        for field_name in _MODEL_RESULT_NUMERIC_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _model_result_numeric_value(value, field_name=field_name))
        for field_name in _MODEL_RESULT_COUNT_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _model_result_count_value(value, field_name=field_name))
        for field_name in ("p_value", "q_value"):
            value = getattr(self, field_name)
            if value is not None and (float(value) < 0.0 or float(value) > 1.0):
                raise ValueError(f"{field_name} must be in [0, 1].")
        if self.confidence_level is not None and (float(self.confidence_level) <= 0.0 or float(self.confidence_level) > 1.0):
            raise ValueError("confidence_level must be in (0, 1].")
        if self.ci_low is not None and self.ci_high is not None and float(self.ci_low) > float(self.ci_high):
            raise ValueError("ci_low must be less than or equal to ci_high.")
        object.__setattr__(self, "status", _optional_text(self.status) or "supplied")
        object.__setattr__(self, "warnings", _model_result_message_tuple(self.warnings, field_name="warnings"))
        object.__setattr__(self, "errors", _model_result_message_tuple(self.errors, field_name="errors"))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))
        object.__setattr__(self, "runtime_backend", RUNTIME_BACKEND_RECORDS)
        object.__setattr__(self, "supplied_only", True)
        object.__setattr__(self, "computed_by_research_analysis", False)
        object.__setattr__(self, "model_fitting_performed", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )


@dataclass(frozen=True)
class ModelFitSummaryRow(_ModelResultRowBase):
    """Supplied-only model-fit summary result row."""

    _ROW_KIND: ClassVar[str] = MODEL_RESULT_KIND_MODEL_FIT_SUMMARY


@dataclass(frozen=True)
class ModelFixedEffectResultRow(_ModelResultRowBase):
    """Supplied-only fixed-effect model result row."""

    _ROW_KIND: ClassVar[str] = MODEL_RESULT_KIND_FIXED_EFFECT


@dataclass(frozen=True)
class ModelRandomEffectResultRow(_ModelResultRowBase):
    """Supplied-only random-effect model result row."""

    _ROW_KIND: ClassVar[str] = MODEL_RESULT_KIND_RANDOM_EFFECT


@dataclass(frozen=True)
class ModelVarianceComponentResultRow(_ModelResultRowBase):
    """Supplied-only variance-component model result row."""

    _ROW_KIND: ClassVar[str] = MODEL_RESULT_KIND_VARIANCE_COMPONENT


@dataclass(frozen=True)
class ModelPlannedComparisonResultRow(_ModelResultRowBase):
    """Supplied-only planned-comparison model result row."""

    _ROW_KIND: ClassVar[str] = MODEL_RESULT_KIND_PLANNED_COMPARISON


@dataclass(frozen=True)
class ModelContrastResultRow(_ModelResultRowBase):
    """Supplied-only contrast model result row."""

    _ROW_KIND: ClassVar[str] = MODEL_RESULT_KIND_CONTRAST


@dataclass(frozen=True)
class ModelResultQcRow(_TabularQcRowMixin):
    """JSON/TSV-safe QC row for supplied model-result contracts."""

    workflow_id: str = "unresolved-workflow"
    input_row_index: int | None = None
    result_row_id: str | None = None
    result_id: str | None = None
    result_kind: str | None = None
    model_id: str | None = None
    model_plan_id: str | None = None
    method_id: str | None = None
    family_id: str | None = None
    term_id: str | None = None
    comparison_id: str | None = None
    contrast_id: str | None = None
    field_name: str | None = None
    status: str = "info"
    code: str = "supplied_only_model_result"
    message: str = "Model-result row was supplied by the caller."
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime_backend: str = RUNTIME_BACKEND_RECORDS
    supplied_only: bool = True
    computed_by_research_analysis: bool = False
    model_fitting_performed: bool = False
    will_write: bool = False
    output_written: bool = False
    no_output_written: bool = True
    output_paths_written: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _optional_text(self.workflow_id) or "unresolved-workflow")
        if self.input_row_index is not None:
            object.__setattr__(self, "input_row_index", int(self.input_row_index))
        for field_name in (
            "result_row_id",
            "result_id",
            "result_kind",
            "model_id",
            "model_plan_id",
            "method_id",
            "family_id",
            "term_id",
            "comparison_id",
            "contrast_id",
            "field_name",
        ):
            object.__setattr__(self, field_name, _optional_text(getattr(self, field_name)))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        object.__setattr__(self, "warnings", _model_result_message_tuple(self.warnings, field_name="warnings"))
        object.__setattr__(self, "errors", _model_result_message_tuple(self.errors, field_name="errors"))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))
        object.__setattr__(self, "runtime_backend", RUNTIME_BACKEND_RECORDS)
        object.__setattr__(self, "supplied_only", True)
        object.__setattr__(self, "computed_by_research_analysis", False)
        object.__setattr__(self, "model_fitting_performed", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )


@dataclass(frozen=True)
class ModelResultProvenanceRow(_TabularQcRowMixin):
    """Provenance row for supplied-only model-result contracts."""

    workflow_id: str = "unresolved-workflow"
    model_id: str | None = None
    model_plan_id: str | None = None
    method_id: str | None = None
    runtime_backend: str = RUNTIME_BACKEND_RECORDS
    model_results_contract_version: str = TABULAR_ASSOCIATION_MODEL_RESULTS_CONTRACT_VERSION
    supplied_only: bool = True
    computed_by_research_analysis: bool = False
    model_fitting_performed: bool = False
    will_write: bool = False
    output_written: bool = False
    no_output_written: bool = True
    output_paths_written: Sequence[str] = ()
    key: str = "model_results_contract_version"
    value: Any = TABULAR_ASSOCIATION_MODEL_RESULTS_CONTRACT_VERSION
    source: str = "tabular_association_model_results"

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _optional_text(self.workflow_id) or "unresolved-workflow")
        object.__setattr__(self, "model_id", _optional_text(self.model_id))
        object.__setattr__(self, "model_plan_id", _optional_text(self.model_plan_id))
        object.__setattr__(self, "method_id", _optional_text(self.method_id))
        object.__setattr__(self, "runtime_backend", RUNTIME_BACKEND_RECORDS)
        object.__setattr__(
            self,
            "model_results_contract_version",
            _non_empty_text(self.model_results_contract_version, field_name="model_results_contract_version"),
        )
        object.__setattr__(self, "supplied_only", True)
        object.__setattr__(self, "computed_by_research_analysis", False)
        object.__setattr__(self, "model_fitting_performed", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )
        object.__setattr__(self, "key", _non_empty_text(self.key, field_name="key"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "source", _non_empty_text(self.source, field_name="source"))


@dataclass(frozen=True)
class _TabularAssociationModelResultContainerBase:
    """Shared no-write container for supplied model-result row contracts."""

    schema_version: str
    model_results_contract_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    no_output_written: bool
    output_paths_written: Sequence[str]
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    model_result_rows: Sequence[Mapping[str, Any] | _ModelResultRowBase]
    qc_rows: Sequence[Mapping[str, Any] | ModelResultQcRow]
    provenance_rows: Sequence[Mapping[str, Any] | ModelResultProvenanceRow]
    row_count: int
    valid_row_count: int
    invalid_row_count: int
    result_kind_counts: Mapping[str, int]
    runtime_backend: str = RUNTIME_BACKEND_RECORDS
    supplied_only: bool = True
    computed_by_research_analysis: bool = False
    model_fitting_performed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(
            self,
            "model_results_contract_version",
            _non_empty_text(self.model_results_contract_version, field_name="model_results_contract_version"),
        )
        object.__setattr__(self, "workflow_id", _optional_text(self.workflow_id) or "unresolved-workflow")
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(_non_empty_text(path, field_name="output_paths_written") for path in self.output_paths_written),
        )
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", _model_result_message_tuple(self.warnings, field_name="warnings"))
        object.__setattr__(self, "errors", _model_result_message_tuple(self.errors, field_name="errors"))
        object.__setattr__(self, "model_result_rows", tuple(_json_safe(row) for row in self.model_result_rows))
        object.__setattr__(self, "qc_rows", tuple(_json_safe(row) for row in self.qc_rows))
        object.__setattr__(self, "provenance_rows", tuple(_json_safe(row) for row in self.provenance_rows))
        for field_name in ("row_count", "valid_row_count", "invalid_row_count"):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(
            self,
            "result_kind_counts",
            {str(key): int(value) for key, value in self.result_kind_counts.items()},
        )
        object.__setattr__(self, "runtime_backend", RUNTIME_BACKEND_RECORDS)
        object.__setattr__(self, "supplied_only", True)
        object.__setattr__(self, "computed_by_research_analysis", False)
        object.__setattr__(self, "model_fitting_performed", False)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularAssociationModelResultContract(_TabularAssociationModelResultContainerBase):
    """Normalized supplied-only model-result rows plus QC/provenance."""


@dataclass(frozen=True)
class TabularAssociationModelResultPlan(_TabularAssociationModelResultContainerBase):
    """No-write plan/preview for supplied model-result row contracts."""


@dataclass(frozen=True)
class TabularAssociationModelResultValidationResult(_TabularAssociationModelResultContainerBase):
    """Validation result for supplied model-result row contracts."""


@dataclass(frozen=True)
class AssociationMultiplicityFamilyPlanRow(_TabularQcRowMixin):
    """Planned no-write multiple-testing family correction."""

    workflow_id: str
    family_id: str
    multiple_testing_method: str | None
    correction_method: str | None
    method_ids: Sequence[str]
    declared_in_families: bool
    declared_in_multiple_testing: bool
    executable: bool
    deferred: bool
    status: str
    code: str
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = False
    plan_only: bool = True
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "family_id", _non_empty_text(self.family_id, field_name="family_id"))
        object.__setattr__(self, "multiple_testing_method", _optional_text(self.multiple_testing_method))
        object.__setattr__(self, "correction_method", _optional_text(self.correction_method))
        object.__setattr__(self, "method_ids", _text_tuple(self.method_ids, field_name="method_ids"))
        object.__setattr__(self, "declared_in_families", bool(self.declared_in_families))
        object.__setattr__(self, "declared_in_multiple_testing", bool(self.declared_in_multiple_testing))
        object.__setattr__(self, "executable", bool(self.executable))
        object.__setattr__(self, "deferred", bool(self.deferred))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationMultiplicityInputRow(_TabularQcRowMixin):
    """Sanitized supplied association/result row metadata for multiplicity."""

    workflow_id: str
    family_id: str | None
    multiple_testing_method: str | None
    correction_method: str | None
    result_row_id: str | None
    input_row_index: int
    method_id: str | None
    method_kind: str | None
    method_name: str | None
    source_id: str | None
    outcome_id: str | None
    predictor_id: str | None
    covariate_ids: Sequence[str]
    statistic_name: str | None
    statistic_value: float | int | None
    p_value_field: str
    p_value: float | None
    p_value_status: str
    status: str
    code: str
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "family_id", _optional_text(self.family_id))
        object.__setattr__(self, "multiple_testing_method", _optional_text(self.multiple_testing_method))
        object.__setattr__(self, "correction_method", _optional_text(self.correction_method))
        object.__setattr__(self, "result_row_id", _optional_text(self.result_row_id))
        object.__setattr__(self, "input_row_index", int(self.input_row_index))
        object.__setattr__(self, "method_id", _optional_text(self.method_id))
        object.__setattr__(self, "method_kind", _optional_text(self.method_kind))
        object.__setattr__(self, "method_name", _optional_text(self.method_name))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "outcome_id", _optional_text(self.outcome_id))
        object.__setattr__(self, "predictor_id", _optional_text(self.predictor_id))
        object.__setattr__(self, "covariate_ids", _text_tuple(self.covariate_ids, field_name="covariate_ids"))
        object.__setattr__(self, "statistic_name", _optional_text(self.statistic_name))
        if self.statistic_value is not None:
            object.__setattr__(self, "statistic_value", _finite_number(self.statistic_value, field_name="statistic_value"))
        object.__setattr__(self, "p_value_field", _non_empty_text(self.p_value_field, field_name="p_value_field"))
        if self.p_value is not None:
            p_value = float(_finite_number(self.p_value, field_name="p_value"))
            if p_value < 0.0 or p_value > 1.0:
                raise ValueError("p_value must be in [0, 1].")
            object.__setattr__(self, "p_value", p_value)
        object.__setattr__(self, "p_value_status", _non_empty_text(self.p_value_status, field_name="p_value_status"))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationMultiplicityResultRow(_TabularQcRowMixin):
    """Step 11F multiplicity result row with existing p-values and optional q-values."""

    workflow_id: str
    family_id: str | None
    multiple_testing_method: str | None
    correction_method: str | None
    result_row_id: str | None
    input_row_index: int
    method_id: str | None
    method_kind: str | None
    method_name: str | None
    source_id: str | None
    outcome_id: str | None
    predictor_id: str | None
    covariate_ids: Sequence[str]
    statistic_name: str | None
    statistic_value: float | int | None
    p_value: float | None
    q_value: float | None
    n_family_total: int
    n_valid_p: int
    n_missing_p: int
    n_invalid_p: int
    n_adjusted: int
    status: str
    code: str
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "family_id", _optional_text(self.family_id))
        object.__setattr__(self, "multiple_testing_method", _optional_text(self.multiple_testing_method))
        object.__setattr__(self, "correction_method", _optional_text(self.correction_method))
        object.__setattr__(self, "result_row_id", _optional_text(self.result_row_id))
        object.__setattr__(self, "input_row_index", int(self.input_row_index))
        object.__setattr__(self, "method_id", _optional_text(self.method_id))
        object.__setattr__(self, "method_kind", _optional_text(self.method_kind))
        object.__setattr__(self, "method_name", _optional_text(self.method_name))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "outcome_id", _optional_text(self.outcome_id))
        object.__setattr__(self, "predictor_id", _optional_text(self.predictor_id))
        object.__setattr__(self, "covariate_ids", _text_tuple(self.covariate_ids, field_name="covariate_ids"))
        object.__setattr__(self, "statistic_name", _optional_text(self.statistic_name))
        if self.statistic_value is not None:
            object.__setattr__(self, "statistic_value", _finite_number(self.statistic_value, field_name="statistic_value"))
        for field_name in ("p_value", "q_value"):
            value = getattr(self, field_name)
            if value is None:
                continue
            number = float(_finite_number(value, field_name=field_name))
            if number < 0.0 or number > 1.0:
                raise ValueError(f"{field_name} must be in [0, 1].")
            object.__setattr__(self, field_name, number)
        for field_name in ("n_family_total", "n_valid_p", "n_missing_p", "n_invalid_p", "n_adjusted"):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationMultiplicityQcRow(_TabularQcRowMixin):
    """Multiplicity/FDR QC row."""

    workflow_id: str
    family_id: str | None
    multiple_testing_method: str | None
    correction_method: str | None
    result_row_id: str | None
    input_row_index: int | None
    status: str
    code: str
    message: str
    n_family_total: int = 0
    n_valid_p: int = 0
    n_missing_p: int = 0
    n_invalid_p: int = 0
    n_adjusted: int = 0
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "family_id", _optional_text(self.family_id))
        object.__setattr__(self, "multiple_testing_method", _optional_text(self.multiple_testing_method))
        object.__setattr__(self, "correction_method", _optional_text(self.correction_method))
        object.__setattr__(self, "result_row_id", _optional_text(self.result_row_id))
        if self.input_row_index is not None:
            object.__setattr__(self, "input_row_index", int(self.input_row_index))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        for field_name in ("n_family_total", "n_valid_p", "n_missing_p", "n_invalid_p", "n_adjusted"):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationMultiplicityMethodSummaryRow(_TabularQcRowMixin):
    """Family-level correction-method summary for multiplicity planning/execution."""

    workflow_id: str
    family_id: str
    multiple_testing_method: str | None
    correction_method: str | None
    executable: bool
    deferred: bool
    n_family_total: int
    n_valid_p: int
    n_missing_p: int
    n_invalid_p: int
    n_adjusted: int
    status: str
    code: str
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = False
    plan_only: bool = True
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "family_id", _non_empty_text(self.family_id, field_name="family_id"))
        object.__setattr__(self, "multiple_testing_method", _optional_text(self.multiple_testing_method))
        object.__setattr__(self, "correction_method", _optional_text(self.correction_method))
        object.__setattr__(self, "executable", bool(self.executable))
        object.__setattr__(self, "deferred", bool(self.deferred))
        for field_name in ("n_family_total", "n_valid_p", "n_missing_p", "n_invalid_p", "n_adjusted"):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _non_empty_text(self.code, field_name="code"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class TabularAssociationMultiplicityProvenanceRow(_TabularQcRowMixin):
    """Multiplicity/FDR provenance row."""

    workflow_id: str
    key: str
    value: Any
    source: str = "tabular_association_multiplicity"

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "key", _non_empty_text(self.key, field_name="key"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "source", _non_empty_text(self.source, field_name="source"))


@dataclass(frozen=True)
class TabularAssociationMultiplicityPlan:
    """No-write multiplicity/FDR family plan for supplied association rows."""

    schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    status: str
    p_value_field: str
    p_value_policy: str
    warnings: Sequence[str]
    errors: Sequence[str]
    workflow_validation_rows: Sequence[Mapping[str, Any] | AssociationValidationRow]
    family_plan_rows: Sequence[Mapping[str, Any] | AssociationMultiplicityFamilyPlanRow]
    qc_rows: Sequence[Mapping[str, Any] | AssociationMultiplicityQcRow]
    method_summary_rows: Sequence[Mapping[str, Any] | AssociationMultiplicityMethodSummaryRow]
    provenance_rows: Sequence[Mapping[str, Any] | TabularAssociationMultiplicityProvenanceRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", False)
        object.__setattr__(self, "plan_only", True)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "p_value_field", _non_empty_text(self.p_value_field, field_name="p_value_field"))
        p_value_policy = _normalized_choice(
            self.p_value_policy,
            field_name="p-value policy",
            supported=SUPPORTED_P_VALUE_POLICIES,
        )
        object.__setattr__(self, "p_value_policy", p_value_policy)
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "workflow_validation_rows", tuple(_json_safe(row) for row in self.workflow_validation_rows))
        object.__setattr__(self, "family_plan_rows", tuple(_json_safe(row) for row in self.family_plan_rows))
        object.__setattr__(self, "qc_rows", tuple(_json_safe(row) for row in self.qc_rows))
        object.__setattr__(self, "method_summary_rows", tuple(_json_safe(row) for row in self.method_summary_rows))
        object.__setattr__(self, "provenance_rows", tuple(_json_safe(row) for row in self.provenance_rows))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularAssociationMultiplicityResult:
    """No-write multiplicity/FDR execution result for supplied association rows."""

    schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    status: str
    p_value_field: str
    p_value_policy: str
    warnings: Sequence[str]
    errors: Sequence[str]
    workflow_validation_rows: Sequence[Mapping[str, Any] | AssociationValidationRow]
    family_plan_rows: Sequence[Mapping[str, Any] | AssociationMultiplicityFamilyPlanRow]
    input_rows: Sequence[Mapping[str, Any] | AssociationMultiplicityInputRow]
    result_rows: Sequence[Mapping[str, Any] | AssociationMultiplicityResultRow]
    qc_rows: Sequence[Mapping[str, Any] | AssociationMultiplicityQcRow]
    method_summary_rows: Sequence[Mapping[str, Any] | AssociationMultiplicityMethodSummaryRow]
    provenance_rows: Sequence[Mapping[str, Any] | TabularAssociationMultiplicityProvenanceRow]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "p_value_field", _non_empty_text(self.p_value_field, field_name="p_value_field"))
        p_value_policy = _normalized_choice(
            self.p_value_policy,
            field_name="p-value policy",
            supported=SUPPORTED_P_VALUE_POLICIES,
        )
        object.__setattr__(self, "p_value_policy", p_value_policy)
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        for field_name in (
            "workflow_validation_rows",
            "family_plan_rows",
            "input_rows",
            "result_rows",
            "qc_rows",
            "method_summary_rows",
            "provenance_rows",
        ):
            object.__setattr__(self, field_name, tuple(_json_safe(row) for row in getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AssociationPublicationInputSummaryRow(_TabularQcRowMixin):
    """One Step 11G input rowset summary for no-write publication handoff."""

    workflow_id: str
    rowset_name: str
    source_rowset_name: str
    row_count: int
    normalized_row_count: int
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "rowset_name", _non_empty_text(self.rowset_name, field_name="rowset_name"))
        object.__setattr__(
            self,
            "source_rowset_name",
            _non_empty_text(self.source_rowset_name, field_name="source_rowset_name"),
        )
        object.__setattr__(self, "row_count", int(self.row_count))
        object.__setattr__(self, "normalized_row_count", int(self.normalized_row_count))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationPublicationTableRow(_TabularQcRowMixin):
    """Publication-table-compatible association handoff row."""

    workflow_id: str
    method_id: str | None
    method_kind: str | None
    method_name: str | None
    family_id: str | None
    source_id: str | None
    outcome_id: str | None
    predictor_id: str | None
    covariate_ids: Sequence[str]
    statistic_name: str | None
    statistic_value: float | int | None
    p_value: float | None
    q_value: float | None
    n: int | None
    n_used: int | None
    n_total: int | None
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    result_row_id: str | None
    pair_id: str | None
    input_row_index: int
    multiplicity_match_field: str | None = None
    multiplicity_match_value: str | int | None = None
    multiplicity_input_row_index: int | None = None
    extra_fields: Mapping[str, Any] = field(default_factory=dict)
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        for field_name in (
            "method_id",
            "method_kind",
            "method_name",
            "family_id",
            "source_id",
            "outcome_id",
            "predictor_id",
            "statistic_name",
            "result_row_id",
            "pair_id",
            "multiplicity_match_field",
        ):
            object.__setattr__(self, field_name, _optional_text(getattr(self, field_name)))
        object.__setattr__(self, "covariate_ids", _publication_text_tuple(self.covariate_ids))
        if self.statistic_value is not None:
            object.__setattr__(self, "statistic_value", _finite_number(self.statistic_value, field_name="statistic_value"))
        for field_name in ("p_value", "q_value"):
            value = getattr(self, field_name)
            if value is not None:
                number = float(_finite_number(value, field_name=field_name))
                if number < 0.0 or number > 1.0:
                    raise ValueError(f"{field_name} must be in [0, 1].")
                object.__setattr__(self, field_name, number)
        for field_name in ("n", "n_used", "n_total", "multiplicity_input_row_index"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, int(value))
        object.__setattr__(self, "input_row_index", int(self.input_row_index))
        object.__setattr__(
            self,
            "multiplicity_match_value",
            _publication_json_safe_scalar(self.multiplicity_match_value),
        )
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "extra_fields", _publication_json_safe_mapping(self.extra_fields))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationPublicationQcTableRow(_TabularQcRowMixin):
    """JSON/TSV-safe QC row for publication-table handoff."""

    workflow_id: str
    rowset_name: str
    input_row_index: int
    status: str
    code: str | None = None
    message: str | None = None
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    row_payload: Mapping[str, Any] = field(default_factory=dict)
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "rowset_name", _non_empty_text(self.rowset_name, field_name="rowset_name"))
        object.__setattr__(self, "input_row_index", int(self.input_row_index))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "code", _optional_text(self.code))
        object.__setattr__(self, "message", _optional_text(self.message))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "row_payload", _publication_json_safe_mapping(self.row_payload))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationPublicationMissingnessTableRow(_TabularQcRowMixin):
    """JSON/TSV-safe missingness row for publication-table handoff."""

    workflow_id: str
    source_id: str | None
    column_name: str | None
    role: str | None
    missing_count: int | None
    nonmissing_count: int | None
    total_count: int | None
    status: str
    code: str | None = None
    message: str | None = None
    input_row_index: int = 0
    row_payload: Mapping[str, Any] = field(default_factory=dict)
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        for field_name in ("source_id", "column_name", "role", "code", "message"):
            object.__setattr__(self, field_name, _optional_text(getattr(self, field_name)))
        for field_name in ("missing_count", "nonmissing_count", "total_count", "input_row_index"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, int(value))
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "row_payload", _publication_json_safe_mapping(self.row_payload))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationPublicationMultiplicityTableRow(_TabularQcRowMixin):
    """JSON/TSV-safe multiplicity row for publication-table handoff."""

    workflow_id: str
    family_id: str | None
    result_row_id: str | None
    pair_id: str | None
    input_row_index: int | None
    p_value: float | None
    q_value: float | None
    status: str
    code: str | None = None
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()
    row_payload: Mapping[str, Any] = field(default_factory=dict)
    executed: bool = True
    plan_only: bool = False
    will_write: bool = False
    output_written: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        for field_name in ("family_id", "result_row_id", "pair_id", "code"):
            object.__setattr__(self, field_name, _optional_text(getattr(self, field_name)))
        if self.input_row_index is not None:
            object.__setattr__(self, "input_row_index", int(self.input_row_index))
        for field_name in ("p_value", "q_value"):
            value = getattr(self, field_name)
            if value is not None:
                number = float(_finite_number(value, field_name=field_name))
                if number < 0.0 or number > 1.0:
                    raise ValueError(f"{field_name} must be in [0, 1].")
                object.__setattr__(self, field_name, number)
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        object.__setattr__(self, "row_payload", _publication_json_safe_mapping(self.row_payload))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)


@dataclass(frozen=True)
class AssociationPublicationProvenanceRow(_TabularQcRowMixin):
    """Provenance row for Step 11G publication-table handoff."""

    workflow_id: str
    key: str
    value: Any
    source: str = "tabular_association_publication_handoff"
    input_row_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "key", _non_empty_text(self.key, field_name="key"))
        object.__setattr__(self, "value", _publication_json_safe(self.value))
        object.__setattr__(self, "source", _non_empty_text(self.source, field_name="source"))
        if self.input_row_index is not None:
            object.__setattr__(self, "input_row_index", int(self.input_row_index))


@dataclass(frozen=True)
class AssociationPublicationManifestRow(_TabularQcRowMixin):
    """Manifest row for Step 11G no-write publication-table handoff."""

    workflow_id: str
    table_name: str
    row_count: int
    tabular_association_schema_version: str
    publication_handoff_schema_version: str
    input_row_counts: Mapping[str, int]
    association_result_row_count: int
    qc_row_count: int
    missingness_row_count: int
    multiplicity_row_count: int
    provenance_row_count: int
    display_row_count: int
    machine_row_count: int
    manifest_row_count: int
    output_table_names: Sequence[str]
    source_rowset_names: Mapping[str, str]
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    no_output_written: bool
    output_paths_written: Sequence[str]
    status: str = "ok"
    warnings: Sequence[str] = ()
    errors: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "table_name", _non_empty_text(self.table_name, field_name="table_name"))
        for field_name in (
            "row_count",
            "association_result_row_count",
            "qc_row_count",
            "missingness_row_count",
            "multiplicity_row_count",
            "provenance_row_count",
            "display_row_count",
            "machine_row_count",
            "manifest_row_count",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(
            self,
            "tabular_association_schema_version",
            _non_empty_text(
                self.tabular_association_schema_version,
                field_name="tabular_association_schema_version",
            ),
        )
        object.__setattr__(
            self,
            "publication_handoff_schema_version",
            _non_empty_text(self.publication_handoff_schema_version, field_name="publication_handoff_schema_version"),
        )
        object.__setattr__(self, "input_row_counts", _json_safe_mapping(self.input_row_counts))
        object.__setattr__(self, "output_table_names", _text_tuple(self.output_table_names, field_name="output_table_names"))
        object.__setattr__(self, "source_rowset_names", _json_safe_mapping(self.source_rowset_names))
        object.__setattr__(self, "executed", bool(self.executed))
        object.__setattr__(self, "plan_only", bool(self.plan_only))
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(str(path) for path in self.output_paths_written if str(path)),
        )
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))


@dataclass(frozen=True)
class TabularAssociationPublicationPlan:
    """No-write plan for generic tabular association publication-table handoff."""

    schema_version: str
    publication_handoff_schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    no_output_written: bool
    output_paths_written: Sequence[str]
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    input_summary_rows: Sequence[Mapping[str, Any] | AssociationPublicationInputSummaryRow]
    association_table_rows: Sequence[Mapping[str, Any] | AssociationPublicationTableRow]
    association_display_rows: Sequence[Mapping[str, Any]]
    association_machine_rows: Sequence[Mapping[str, Any]]
    qc_table_rows: Sequence[Mapping[str, Any] | AssociationPublicationQcTableRow]
    missingness_table_rows: Sequence[Mapping[str, Any] | AssociationPublicationMissingnessTableRow]
    multiplicity_table_rows: Sequence[Mapping[str, Any] | AssociationPublicationMultiplicityTableRow]
    provenance_table_rows: Sequence[Mapping[str, Any] | AssociationPublicationProvenanceRow]
    manifest_rows: Sequence[Mapping[str, Any] | AssociationPublicationManifestRow]
    column_mappings: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(
            self,
            "publication_handoff_schema_version",
            _non_empty_text(self.publication_handoff_schema_version, field_name="publication_handoff_schema_version"),
        )
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", False)
        object.__setattr__(self, "plan_only", True)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(str(path) for path in self.output_paths_written if str(path)),
        )
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        for field_name in (
            "input_summary_rows",
            "association_table_rows",
            "association_display_rows",
            "association_machine_rows",
            "qc_table_rows",
            "missingness_table_rows",
            "multiplicity_table_rows",
            "provenance_table_rows",
            "manifest_rows",
            "column_mappings",
        ):
            object.__setattr__(self, field_name, tuple(_json_safe(row) for row in getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TabularAssociationPublicationResult:
    """In-memory no-write publication-table handoff for supplied association rows."""

    schema_version: str
    publication_handoff_schema_version: str
    workflow_id: str
    valid: bool
    executed: bool
    plan_only: bool
    will_write: bool
    output_written: bool
    no_output_written: bool
    output_paths_written: Sequence[str]
    status: str
    warnings: Sequence[str]
    errors: Sequence[str]
    input_summary_rows: Sequence[Mapping[str, Any] | AssociationPublicationInputSummaryRow]
    association_table_rows: Sequence[Mapping[str, Any] | AssociationPublicationTableRow]
    association_display_rows: Sequence[Mapping[str, Any]]
    association_machine_rows: Sequence[Mapping[str, Any]]
    qc_table_rows: Sequence[Mapping[str, Any] | AssociationPublicationQcTableRow]
    missingness_table_rows: Sequence[Mapping[str, Any] | AssociationPublicationMissingnessTableRow]
    multiplicity_table_rows: Sequence[Mapping[str, Any] | AssociationPublicationMultiplicityTableRow]
    provenance_table_rows: Sequence[Mapping[str, Any] | AssociationPublicationProvenanceRow]
    manifest_rows: Sequence[Mapping[str, Any] | AssociationPublicationManifestRow]
    column_mappings: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _non_empty_text(self.schema_version, field_name="schema_version"))
        object.__setattr__(
            self,
            "publication_handoff_schema_version",
            _non_empty_text(self.publication_handoff_schema_version, field_name="publication_handoff_schema_version"),
        )
        object.__setattr__(self, "workflow_id", _non_empty_text(self.workflow_id, field_name="workflow_id"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "executed", True)
        object.__setattr__(self, "plan_only", False)
        object.__setattr__(self, "will_write", False)
        object.__setattr__(self, "output_written", False)
        object.__setattr__(self, "no_output_written", True)
        object.__setattr__(
            self,
            "output_paths_written",
            tuple(str(path) for path in self.output_paths_written if str(path)),
        )
        object.__setattr__(self, "status", _non_empty_text(self.status, field_name="status"))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings if str(warning)))
        object.__setattr__(self, "errors", tuple(str(error) for error in self.errors if str(error)))
        for field_name in (
            "input_summary_rows",
            "association_table_rows",
            "association_display_rows",
            "association_machine_rows",
            "qc_table_rows",
            "missingness_table_rows",
            "multiplicity_table_rows",
            "provenance_table_rows",
            "manifest_rows",
            "column_mappings",
        ):
            object.__setattr__(self, field_name, tuple(_json_safe(row) for row in getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


def parse_tabular_association_workflow_document(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
) -> TabularAssociationWorkflowSpec:
    """Parse and validate a mapping-style workflow document."""

    workflow = _coerce_workflow_spec(document)
    rows = _validate_workflow(workflow)
    errors = [row.message for row in rows if row.status == "error"]
    if errors:
        raise ValueError("; ".join(errors))
    return workflow


def validate_tabular_association_workflow_document(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
) -> AssociationPlanPreview:
    """Validate a workflow document and return a JSON-safe plan preview."""

    return plan_tabular_association_workflow(document)


def plan_tabular_association_workflow(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
) -> AssociationPlanPreview:
    """Return a plan-only preview without loading data, computing, or writing."""

    try:
        workflow = _coerce_workflow_spec(document)
    except (TypeError, ValueError) as exc:
        workflow_id = _best_effort_workflow_id(document)
        error_row = AssociationValidationRow(
            level="error",
            status="error",
            code="workflow_parse_error",
            message=str(exc),
            location="workflow",
        )
        return AssociationPlanPreview(
            schema_version=SCHEMA_VERSION,
            workflow_id=workflow_id,
            valid=False,
            executed=False,
            plan_only=True,
            will_write=False,
            output_written=False,
            status="error",
            warnings=(),
            errors=(str(exc),),
            validation_rows=(error_row,),
            source_rows=(),
            column_rows=(),
            variable_rows=(),
            method_rows=(),
            family_rows=(),
            output_rows=(),
            publication_handoff_rows=(),
            visualization_handoff_rows=(),
            provenance_rows=_provenance_rows_for_error(workflow_id),
        )

    validation_rows = _validate_workflow(workflow)
    warnings = tuple(row.message for row in validation_rows if row.status == "warning")
    errors = tuple(row.message for row in validation_rows if row.status == "error")
    return AssociationPlanPreview(
        schema_version=SCHEMA_VERSION,
        workflow_id=workflow.workflow_id,
        valid=not errors,
        executed=False,
        plan_only=True,
        will_write=False,
        output_written=False,
        status="ok" if not errors else "error",
        warnings=warnings,
        errors=errors,
        validation_rows=validation_rows,
        source_rows=_source_rows(workflow),
        column_rows=_column_rows(workflow),
        variable_rows=_variable_rows(workflow),
        method_rows=_method_rows(workflow),
        family_rows=_family_rows(workflow),
        output_rows=_output_rows(workflow),
        publication_handoff_rows=_handoff_rows(workflow, handoff_types={"publication"}),
        visualization_handoff_rows=_handoff_rows(workflow, handoff_types={"visualization", "report"}),
        provenance_rows=_provenance_rows(workflow),
    )


def plan_tabular_association_row_source_adapter(
    *,
    requested_backend: str = BACKEND_RECORDS,
    row_source_kind: str = "uninspected",
    include_input_row_index: bool = False,
    input_row_index_field: str = "input_row_index",
    metadata: Mapping[str, Any] | None = None,
) -> TabularAssociationRowSourceAdapterSpec:
    """Return a no-write plan for standard-library row-source coercion."""

    return TabularAssociationRowSourceAdapterSpec(
        requested_backend=requested_backend,
        runtime_backend=RUNTIME_BACKEND_RECORDS,
        row_source_kind=row_source_kind,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata={} if metadata is None else metadata,
        executed=False,
        plan_only=True,
        will_write=False,
        output_written=False,
        no_output_written=True,
        output_paths_written=(),
    )


def inspect_tabular_association_row_source(
    row_source: Any,
    *,
    requested_backend: str = BACKEND_RECORDS,
    include_input_row_index: bool = False,
    input_row_index_field: str = "input_row_index",
    metadata: Mapping[str, Any] | None = None,
) -> TabularAssociationRowSourceResult:
    """Inspect and coerce a supported row source into copied record dictionaries."""

    return coerce_tabular_association_records(
        row_source,
        requested_backend=requested_backend,
        include_input_row_index=include_input_row_index,
        input_row_index_field=input_row_index_field,
        metadata=metadata,
    )


def coerce_tabular_association_records(
    row_source: Any,
    *,
    requested_backend: str = BACKEND_RECORDS,
    include_input_row_index: bool = False,
    input_row_index_field: str = "input_row_index",
    metadata: Mapping[str, Any] | None = None,
) -> TabularAssociationRowSourceResult:
    """Coerce supported standard-library row sources into copied records.

    This helper implements only generic duck-typed row-source protocols. It
    does not import or execute pandas, Polars, or any concrete dataframe
    backend adapter.
    """

    try:
        spec = plan_tabular_association_row_source_adapter(
            requested_backend=requested_backend,
            include_input_row_index=include_input_row_index,
            input_row_index_field=input_row_index_field,
            metadata={} if metadata is None else metadata,
        )
    except (TypeError, ValueError) as exc:
        return _row_source_error_result(
            adapter_id=TABULAR_ASSOCIATION_ROW_SOURCE_ADAPTER_VERSION,
            requested_backend=str(requested_backend or BACKEND_RECORDS),
            row_source_kind=SOURCE_KIND_UNSUPPORTED,
            include_input_row_index=include_input_row_index,
            input_row_index_field=input_row_index_field,
            metadata={},
            errors=(str(exc),),
        )
    return _coerce_row_source_records(row_source, spec=spec)


def iter_tabular_association_records(
    row_source: Any,
    *,
    requested_backend: str = BACKEND_RECORDS,
    include_input_row_index: bool = False,
    input_row_index_field: str = "input_row_index",
    metadata: Mapping[str, Any] | None = None,
) -> Iterable[Mapping[str, Any]]:
    """Yield copied records from a supported standard-library row source."""

    return iter(
        coerce_tabular_association_records(
            row_source,
            requested_backend=requested_backend,
            include_input_row_index=include_input_row_index,
            input_row_index_field=input_row_index_field,
            metadata=metadata,
        ).records
    )


def plan_tabular_association_qc(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
    *,
    source_inventory_specs: Mapping[str, Any] | Sequence[Mapping[str, Any] | TabularSourceInventorySpec] = (),
) -> TabularAssociationQcPlan:
    """Return a no-write QC plan without loading source rows."""

    try:
        workflow = _coerce_workflow_spec(_unwrap_tabular_association_workflow_document(document))
        inventory_specs = _coerce_source_inventory_specs(source_inventory_specs)
    except (TypeError, ValueError) as exc:
        workflow_id = _best_effort_qc_workflow_id(document)
        error_row = AssociationValidationRow(
            level="error",
            status="error",
            code="workflow_parse_error",
            message=str(exc),
            location="workflow",
        )
        return TabularAssociationQcPlan(
            schema_version=SCHEMA_VERSION,
            workflow_id=workflow_id,
            valid=False,
            executed=False,
            plan_only=True,
            will_write=False,
            output_written=False,
            status="error",
            warnings=(),
            errors=(str(exc),),
            workflow_validation_rows=(error_row,),
            source_inventory_specs=(),
            source_inventory_rows=(),
            provenance_rows=_qc_provenance_rows_for_error(workflow_id),
        )

    validation_rows = _validate_workflow(workflow)
    source_inventory_rows = _planned_source_inventory_rows(
        workflow,
        spec_by_id=_source_inventory_specs_by_id(inventory_specs),
    )
    warnings = tuple(row.message for row in validation_rows if row.status == "warning")
    errors = tuple(row.message for row in validation_rows if row.status == "error")
    return TabularAssociationQcPlan(
        schema_version=SCHEMA_VERSION,
        workflow_id=workflow.workflow_id,
        valid=not errors,
        executed=False,
        plan_only=True,
        will_write=False,
        output_written=False,
        status="ok" if not errors else "error",
        warnings=warnings,
        errors=errors,
        workflow_validation_rows=validation_rows,
        source_inventory_specs=inventory_specs,
        source_inventory_rows=source_inventory_rows,
        provenance_rows=_qc_provenance_rows(
            workflow,
            plan_only=True,
            source_count=len(workflow.sources),
            loaded_source_count=0,
        ),
    )


def run_tabular_association_qc(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
    *,
    source_rows_by_id: Mapping[str, Any] | None = None,
    source_inventory_specs: Mapping[str, Any] | Sequence[Mapping[str, Any] | TabularSourceInventorySpec] = (),
) -> TabularAssociationQcResult:
    """Inspect supplied records or small stdlib-readable sources for QC only.

    The result is inventory and validation metadata. It writes nothing, does
    not mutate source rows, and does not compute association statistics.
    """

    try:
        workflow = _coerce_workflow_spec(_unwrap_tabular_association_workflow_document(document))
        inventory_specs = _coerce_source_inventory_specs(source_inventory_specs)
        rows_by_id = _coerce_source_rows_by_id(source_rows_by_id)
    except (TypeError, ValueError) as exc:
        workflow_id = _best_effort_qc_workflow_id(document)
        error_row = AssociationValidationRow(
            level="error",
            status="error",
            code="workflow_parse_error",
            message=str(exc),
            location="workflow",
        )
        return TabularAssociationQcResult(
            schema_version=SCHEMA_VERSION,
            workflow_id=workflow_id,
            valid=False,
            executed=False,
            plan_only=False,
            will_write=False,
            output_written=False,
            status="error",
            warnings=(),
            errors=(str(exc),),
            workflow_validation_rows=(error_row,),
            source_inventory_rows=(),
            source_load_rows=(),
            column_inventory_rows=(),
            schema_validation_rows=(),
            variable_qc_rows=(),
            missingness_rows=(),
            duplicate_rows=(),
            nonfinite_rows=(),
            categorical_qc_rows=(),
            numeric_qc_rows=(),
            provenance_rows=_qc_provenance_rows_for_error(workflow_id),
        )

    workflow_validation_rows = _validate_workflow(workflow)
    spec_by_id = _source_inventory_specs_by_id(inventory_specs)

    source_inventory_rows: list[TabularSourceInventoryRow] = []
    source_load_rows: list[TabularSourceLoadRow] = []
    column_inventory_rows: list[TabularColumnInventoryRow] = []
    schema_validation_rows: list[TabularSchemaValidationRow] = []
    variable_qc_rows: list[TabularVariableQcRow] = []
    missingness_rows: list[TabularMissingnessRow] = []
    duplicate_rows: list[TabularDuplicateRow] = []
    nonfinite_rows: list[TabularNonFiniteRow] = []
    categorical_qc_rows: list[TabularCategoricalQcRow] = []
    numeric_qc_rows: list[TabularNumericQcRow] = []

    for source in workflow.sources:
        loaded_source = _load_qc_source(
            workflow=workflow,
            source=source,
            source_rows_by_id=rows_by_id,
            inventory_spec=spec_by_id.get(source.source_id),
        )
        source_inventory_rows.append(_source_inventory_row(workflow, source, loaded_source))
        source_load_rows.append(_source_load_row(workflow, source, loaded_source))
        column_inventory_rows.extend(_column_inventory_rows_for_source(workflow, source, loaded_source))
        schema_validation_rows.extend(_schema_validation_rows_for_source(workflow, source, loaded_source))
        variable_qc_rows.extend(_variable_qc_rows_for_source(workflow, source, loaded_source))
        if loaded_source["load_status"] in {"loaded", "empty"}:
            missingness_rows.extend(_missingness_rows_for_source(workflow, source, loaded_source))
            duplicate_rows.extend(_duplicate_rows_for_source(workflow, source, loaded_source))
            nonfinite_rows.extend(_nonfinite_rows_for_source(workflow, source, loaded_source))
            categorical_qc_rows.extend(_categorical_qc_rows_for_source(workflow, source, loaded_source))
            numeric_qc_rows.extend(_numeric_qc_rows_for_source(workflow, source, loaded_source))

    warnings, errors = _qc_result_messages(
        workflow_validation_rows=workflow_validation_rows,
        source_inventory_rows=source_inventory_rows,
        source_load_rows=source_load_rows,
        column_inventory_rows=column_inventory_rows,
        schema_validation_rows=schema_validation_rows,
        variable_qc_rows=variable_qc_rows,
        missingness_rows=missingness_rows,
        duplicate_rows=duplicate_rows,
        nonfinite_rows=nonfinite_rows,
        categorical_qc_rows=categorical_qc_rows,
        numeric_qc_rows=numeric_qc_rows,
    )
    status = "error" if errors else ("warning" if warnings else "ok")
    loaded_source_count = sum(1 for row in source_load_rows if row.load_status in {"loaded", "empty"})
    return TabularAssociationQcResult(
        schema_version=SCHEMA_VERSION,
        workflow_id=workflow.workflow_id,
        valid=not errors,
        executed=False,
        plan_only=False,
        will_write=False,
        output_written=False,
        status=status,
        warnings=warnings,
        errors=errors,
        workflow_validation_rows=workflow_validation_rows,
        source_inventory_rows=source_inventory_rows,
        source_load_rows=source_load_rows,
        column_inventory_rows=column_inventory_rows,
        schema_validation_rows=schema_validation_rows,
        variable_qc_rows=variable_qc_rows,
        missingness_rows=missingness_rows,
        duplicate_rows=duplicate_rows,
        nonfinite_rows=nonfinite_rows,
        categorical_qc_rows=categorical_qc_rows,
        numeric_qc_rows=numeric_qc_rows,
        provenance_rows=_qc_provenance_rows(
            workflow,
            plan_only=False,
            source_count=len(workflow.sources),
            loaded_source_count=loaded_source_count,
        ),
    )


def plan_tabular_association_correlations(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
) -> TabularAssociationCorrelationPlan:
    """Return a no-write Pearson/Spearman correlation association-row plan."""

    try:
        workflow = _coerce_workflow_spec(_unwrap_tabular_association_workflow_document(document))
    except (TypeError, ValueError) as exc:
        workflow_id = _best_effort_qc_workflow_id(document)
        error_row = AssociationValidationRow(
            level="error",
            status="error",
            code="workflow_parse_error",
            message=str(exc),
            location="workflow",
        )
        return TabularAssociationCorrelationPlan(
            schema_version=SCHEMA_VERSION,
            workflow_id=workflow_id,
            valid=False,
            executed=False,
            plan_only=True,
            will_write=False,
            output_written=False,
            status="error",
            warnings=(),
            errors=(str(exc),),
            workflow_validation_rows=(error_row,),
            pair_plan_rows=(),
            method_summary_rows=(),
            provenance_rows=_correlation_provenance_rows_for_error(
                workflow_id,
                executed=False,
                plan_only=True,
                qc_mode="planned",
            ),
        )

    workflow_validation_rows = _validate_workflow(workflow)
    pair_plan_rows = _correlation_pair_plan_rows(workflow)
    method_summary_rows = _correlation_method_summary_rows(
        workflow,
        pair_plan_rows=pair_plan_rows,
        result_rows=(),
        executed=False,
        plan_only=True,
    )
    warnings, errors = _correlation_messages(
        workflow_validation_rows=workflow_validation_rows,
        pair_plan_rows=pair_plan_rows,
        source_load_rows=(),
        input_qc_summary_rows=(),
        computation_qc_rows=(),
        result_rows=(),
    )
    status = "error" if errors else ("warning" if warnings else "ok")
    return TabularAssociationCorrelationPlan(
        schema_version=SCHEMA_VERSION,
        workflow_id=workflow.workflow_id,
        valid=not errors,
        executed=False,
        plan_only=True,
        will_write=False,
        output_written=False,
        status=status,
        warnings=warnings,
        errors=errors,
        workflow_validation_rows=workflow_validation_rows,
        pair_plan_rows=pair_plan_rows,
        method_summary_rows=method_summary_rows,
        provenance_rows=_correlation_provenance_rows(
            workflow,
            executed=False,
            plan_only=True,
            source_count=len(workflow.sources),
            loaded_source_count=0,
            method_count=len(workflow.methods),
            pair_count=len(pair_plan_rows),
            result_row_count=0,
            qc_mode="planned",
        ),
    )


def run_tabular_association_correlations(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
    *,
    source_rows_by_id: Mapping[str, Any] | None = None,
    source_inventory_specs: Mapping[str, Any] | Sequence[Mapping[str, Any] | TabularSourceInventorySpec] = (),
    qc_result: Mapping[str, Any] | TabularAssociationQcResult | None = None,
) -> TabularAssociationCorrelationResult:
    """Compute bounded Pearson/Spearman rows from supplied records or small sources.

    The function writes nothing, does not mutate source rows, and keeps runtime
    execution on the standard-library records backend even when pandas or Polars
    are declared as future schema backends.
    """

    try:
        workflow = _coerce_workflow_spec(_unwrap_tabular_association_workflow_document(document))
        inventory_specs = _coerce_source_inventory_specs(source_inventory_specs)
        rows_by_id = _coerce_source_rows_by_id(source_rows_by_id)
    except (TypeError, ValueError) as exc:
        workflow_id = _best_effort_qc_workflow_id(document)
        error_row = AssociationValidationRow(
            level="error",
            status="error",
            code="workflow_parse_error",
            message=str(exc),
            location="workflow",
        )
        return TabularAssociationCorrelationResult(
            schema_version=SCHEMA_VERSION,
            workflow_id=workflow_id,
            valid=False,
            executed=True,
            plan_only=False,
            will_write=False,
            output_written=False,
            status="error",
            warnings=(),
            errors=(str(exc),),
            workflow_validation_rows=(error_row,),
            pair_plan_rows=(),
            source_inventory_rows=(),
            source_load_rows=(),
            input_qc_summary_rows=(),
            computation_qc_rows=(),
            result_rows=(),
            method_summary_rows=(),
            provenance_rows=_correlation_provenance_rows_for_error(
                workflow_id,
                executed=True,
                plan_only=False,
                qc_mode="run_inline",
            ),
        )

    workflow_validation_rows = _validate_workflow(workflow)
    pair_plan_rows = _correlation_pair_plan_rows(workflow)
    spec_by_id = _source_inventory_specs_by_id(inventory_specs)
    qc_mode = "supplied" if qc_result is not None else "run_inline"

    loaded_by_id: dict[str, dict[str, Any]] = {}
    source_inventory_rows: list[TabularSourceInventoryRow] = []
    source_load_rows: list[TabularSourceLoadRow] = []
    input_qc_summary_rows: list[AssociationInputQcSummaryRow] = []

    for source in workflow.sources:
        loaded_source = _load_qc_source(
            workflow=workflow,
            source=source,
            source_rows_by_id=rows_by_id,
            inventory_spec=spec_by_id.get(source.source_id),
        )
        loaded_by_id[source.source_id] = loaded_source
        source_inventory_rows.append(_source_inventory_row(workflow, source, loaded_source))
        source_load_rows.append(_source_load_row(workflow, source, loaded_source))
        input_qc_summary_rows.append(_association_input_qc_summary_row(workflow, source, loaded_source))

    computation_qc_rows: list[CorrelationComputationQcRow] = []
    result_rows: list[CorrelationAssociationResultRow] = []
    for pair in pair_plan_rows:
        if pair.deferred:
            result_row, qc_row = _deferred_correlation_result_row(pair)
        else:
            loaded_source = loaded_by_id.get(pair.source_id or "")
            if loaded_source is None or loaded_source["load_status"] not in {"loaded", "empty"}:
                result_row, qc_row = _unloaded_source_correlation_result_row(pair, loaded_source)
            else:
                result_row, qc_row = _computed_correlation_result_row(
                    workflow=workflow,
                    pair=pair,
                    loaded_source=loaded_source,
                )
        result_rows.append(result_row)
        computation_qc_rows.append(qc_row)

    method_summary_rows = _correlation_method_summary_rows(
        workflow,
        pair_plan_rows=pair_plan_rows,
        result_rows=result_rows,
        executed=True,
        plan_only=False,
    )
    warnings, errors = _correlation_messages(
        workflow_validation_rows=workflow_validation_rows,
        pair_plan_rows=pair_plan_rows,
        source_load_rows=source_load_rows,
        input_qc_summary_rows=input_qc_summary_rows,
        computation_qc_rows=computation_qc_rows,
        result_rows=result_rows,
    )
    status = "error" if errors else ("warning" if warnings else "ok")
    loaded_source_count = sum(1 for row in source_load_rows if row.load_status in {"loaded", "empty"})
    return TabularAssociationCorrelationResult(
        schema_version=SCHEMA_VERSION,
        workflow_id=workflow.workflow_id,
        valid=not errors,
        executed=True,
        plan_only=False,
        will_write=False,
        output_written=False,
        status=status,
        warnings=warnings,
        errors=errors,
        workflow_validation_rows=workflow_validation_rows,
        pair_plan_rows=pair_plan_rows,
        source_inventory_rows=source_inventory_rows,
        source_load_rows=source_load_rows,
        input_qc_summary_rows=input_qc_summary_rows,
        computation_qc_rows=computation_qc_rows,
        result_rows=result_rows,
        method_summary_rows=method_summary_rows,
        provenance_rows=_correlation_provenance_rows(
            workflow,
            executed=True,
            plan_only=False,
            source_count=len(workflow.sources),
            loaded_source_count=loaded_source_count,
            method_count=len(workflow.methods),
            pair_count=len(pair_plan_rows),
            result_row_count=len(result_rows),
            qc_mode=qc_mode,
        ),
    )


def plan_tabular_association_adjusted(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
) -> TabularAssociationAdjustedPlan:
    """Return a no-write partial/regression association-row plan."""

    try:
        workflow = _coerce_workflow_spec(_unwrap_tabular_association_workflow_document(document))
    except (TypeError, ValueError) as exc:
        workflow_id = _best_effort_qc_workflow_id(document)
        error_row = AssociationValidationRow(
            level="error",
            status="error",
            code="workflow_parse_error",
            message=str(exc),
            location="workflow",
        )
        return TabularAssociationAdjustedPlan(
            schema_version=SCHEMA_VERSION,
            workflow_id=workflow_id,
            valid=False,
            executed=False,
            plan_only=True,
            will_write=False,
            output_written=False,
            status="error",
            warnings=(),
            errors=(str(exc),),
            workflow_validation_rows=(error_row,),
            pair_plan_rows=(),
            method_summary_rows=(),
            provenance_rows=_adjusted_provenance_rows_for_error(
                workflow_id,
                executed=False,
                plan_only=True,
                qc_mode="planned",
            ),
        )

    workflow_validation_rows = _validate_workflow(workflow)
    pair_plan_rows = _adjusted_pair_plan_rows(workflow)
    method_summary_rows = _adjusted_method_summary_rows(
        workflow,
        pair_plan_rows=pair_plan_rows,
        result_rows=(),
        executed=False,
        plan_only=True,
    )
    warnings, errors = _adjusted_messages(
        workflow_validation_rows=workflow_validation_rows,
        pair_plan_rows=pair_plan_rows,
        source_load_rows=(),
        input_qc_summary_rows=(),
        computation_qc_rows=(),
        result_rows=(),
    )
    status = "error" if errors else ("warning" if warnings else "ok")
    return TabularAssociationAdjustedPlan(
        schema_version=SCHEMA_VERSION,
        workflow_id=workflow.workflow_id,
        valid=not errors,
        executed=False,
        plan_only=True,
        will_write=False,
        output_written=False,
        status=status,
        warnings=warnings,
        errors=errors,
        workflow_validation_rows=workflow_validation_rows,
        pair_plan_rows=pair_plan_rows,
        method_summary_rows=method_summary_rows,
        provenance_rows=_adjusted_provenance_rows(
            workflow,
            executed=False,
            plan_only=True,
            source_count=len(workflow.sources),
            loaded_source_count=0,
            method_count=len(workflow.methods),
            plan_row_count=len(pair_plan_rows),
            result_row_count=0,
            qc_mode="planned",
        ),
    )


def run_tabular_association_adjusted(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
    *,
    source_rows_by_id: Mapping[str, Any] | None = None,
    source_inventory_specs: Mapping[str, Any] | Sequence[Mapping[str, Any] | TabularSourceInventorySpec] = (),
    qc_result: Mapping[str, Any] | TabularAssociationQcResult | None = None,
) -> TabularAssociationAdjustedResult:
    """Compute bounded partial/regression rows from supplied records or small sources.

    The function writes nothing, does not mutate source rows, and keeps runtime
    execution on the standard-library records backend even when pandas or Polars
    are declared as future schema backends.
    """

    try:
        workflow = _coerce_workflow_spec(_unwrap_tabular_association_workflow_document(document))
        inventory_specs = _coerce_source_inventory_specs(source_inventory_specs)
        rows_by_id = _coerce_source_rows_by_id(source_rows_by_id)
    except (TypeError, ValueError) as exc:
        workflow_id = _best_effort_qc_workflow_id(document)
        error_row = AssociationValidationRow(
            level="error",
            status="error",
            code="workflow_parse_error",
            message=str(exc),
            location="workflow",
        )
        return TabularAssociationAdjustedResult(
            schema_version=SCHEMA_VERSION,
            workflow_id=workflow_id,
            valid=False,
            executed=True,
            plan_only=False,
            will_write=False,
            output_written=False,
            status="error",
            warnings=(),
            errors=(str(exc),),
            workflow_validation_rows=(error_row,),
            pair_plan_rows=(),
            source_inventory_rows=(),
            source_load_rows=(),
            input_qc_summary_rows=(),
            computation_qc_rows=(),
            result_rows=(),
            adjusted_result_rows=(),
            regression_result_rows=(),
            method_summary_rows=(),
            provenance_rows=_adjusted_provenance_rows_for_error(
                workflow_id,
                executed=True,
                plan_only=False,
                qc_mode="run_inline",
            ),
        )

    workflow_validation_rows = _validate_workflow(workflow)
    pair_plan_rows = _adjusted_pair_plan_rows(workflow)
    spec_by_id = _source_inventory_specs_by_id(inventory_specs)
    qc_mode = "supplied" if qc_result is not None else "run_inline"

    loaded_by_id: dict[str, dict[str, Any]] = {}
    source_inventory_rows: list[TabularSourceInventoryRow] = []
    source_load_rows: list[TabularSourceLoadRow] = []
    input_qc_summary_rows: list[AssociationInputQcSummaryRow] = []

    for source in workflow.sources:
        loaded_source = _load_qc_source(
            workflow=workflow,
            source=source,
            source_rows_by_id=rows_by_id,
            inventory_spec=spec_by_id.get(source.source_id),
        )
        loaded_by_id[source.source_id] = loaded_source
        source_inventory_rows.append(_source_inventory_row(workflow, source, loaded_source))
        source_load_rows.append(_source_load_row(workflow, source, loaded_source))
        input_qc_summary_rows.append(_association_input_qc_summary_row(workflow, source, loaded_source))

    computation_qc_rows: list[AdjustedAssociationComputationQcRow] = []
    result_rows: list[AdjustedAssociationResultRow | RegressionAssociationResultRow] = []
    adjusted_result_rows: list[AdjustedAssociationResultRow] = []
    regression_result_rows: list[RegressionAssociationResultRow] = []
    for pair in pair_plan_rows:
        if pair.deferred:
            result_row, qc_row = _deferred_adjusted_result_row(pair)
        else:
            loaded_source = loaded_by_id.get(pair.source_id or "")
            if loaded_source is None or loaded_source["load_status"] not in {"loaded", "empty"}:
                result_row, qc_row = _unloaded_source_adjusted_result_row(pair, loaded_source)
            else:
                result_row, qc_row = _computed_adjusted_result_row(
                    workflow=workflow,
                    pair=pair,
                    loaded_source=loaded_source,
                )
        result_rows.append(result_row)
        if isinstance(result_row, RegressionAssociationResultRow):
            regression_result_rows.append(result_row)
        else:
            adjusted_result_rows.append(result_row)
        computation_qc_rows.append(qc_row)

    method_summary_rows = _adjusted_method_summary_rows(
        workflow,
        pair_plan_rows=pair_plan_rows,
        result_rows=result_rows,
        executed=True,
        plan_only=False,
    )
    warnings, errors = _adjusted_messages(
        workflow_validation_rows=workflow_validation_rows,
        pair_plan_rows=pair_plan_rows,
        source_load_rows=source_load_rows,
        input_qc_summary_rows=input_qc_summary_rows,
        computation_qc_rows=computation_qc_rows,
        result_rows=result_rows,
    )
    status = "error" if errors else ("warning" if warnings else "ok")
    loaded_source_count = sum(1 for row in source_load_rows if row.load_status in {"loaded", "empty"})
    return TabularAssociationAdjustedResult(
        schema_version=SCHEMA_VERSION,
        workflow_id=workflow.workflow_id,
        valid=not errors,
        executed=True,
        plan_only=False,
        will_write=False,
        output_written=False,
        status=status,
        warnings=warnings,
        errors=errors,
        workflow_validation_rows=workflow_validation_rows,
        pair_plan_rows=pair_plan_rows,
        source_inventory_rows=source_inventory_rows,
        source_load_rows=source_load_rows,
        input_qc_summary_rows=input_qc_summary_rows,
        computation_qc_rows=computation_qc_rows,
        result_rows=result_rows,
        adjusted_result_rows=adjusted_result_rows,
        regression_result_rows=regression_result_rows,
        method_summary_rows=method_summary_rows,
        provenance_rows=_adjusted_provenance_rows(
            workflow,
            executed=True,
            plan_only=False,
            source_count=len(workflow.sources),
            loaded_source_count=loaded_source_count,
            method_count=len(workflow.methods),
            plan_row_count=len(pair_plan_rows),
            result_row_count=len(result_rows),
            qc_mode=qc_mode,
        ),
    )


def plan_tabular_association_repeated_measures(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
) -> TabularAssociationRepeatedMeasuresPlan:
    """Return a no-write repeated-measures/mixed-model design plan."""

    try:
        workflow = _coerce_workflow_spec(_unwrap_tabular_association_workflow_document(document))
    except (TypeError, ValueError) as exc:
        workflow_id = _best_effort_qc_workflow_id(document)
        error_row = AssociationValidationRow(
            level="error",
            status="error",
            code="workflow_parse_error",
            message=str(exc),
            location="workflow",
        )
        return TabularAssociationRepeatedMeasuresPlan(
            schema_version=SCHEMA_VERSION,
            repeated_measures_plan_version=TABULAR_ASSOCIATION_REPEATED_MEASURES_PLAN_VERSION,
            workflow_id=workflow_id,
            valid=False,
            executed=False,
            plan_only=True,
            will_write=False,
            output_written=False,
            no_output_written=True,
            output_paths_written=(),
            status="error",
            warnings=(),
            errors=(str(exc),),
            workflow_validation_rows=(error_row,),
            model_plan_rows=(),
            provenance_rows=_repeated_measures_provenance_rows_for_error(
                workflow_id,
                executed=False,
                plan_only=True,
                qc_mode="planned",
            ),
        )

    workflow_validation_rows = _validate_workflow(workflow)
    model_plan_rows = _repeated_measures_model_plan_rows(workflow)
    warnings, errors = _repeated_measures_messages(
        workflow_validation_rows=workflow_validation_rows,
        model_plan_rows=model_plan_rows,
        source_load_rows=(),
        design_summary_rows=(),
        factor_summary_rows=(),
        qc_rows=(),
    )
    status = "error" if errors else ("warning" if warnings else "ok")
    return TabularAssociationRepeatedMeasuresPlan(
        schema_version=SCHEMA_VERSION,
        repeated_measures_plan_version=TABULAR_ASSOCIATION_REPEATED_MEASURES_PLAN_VERSION,
        workflow_id=workflow.workflow_id,
        valid=not errors,
        executed=False,
        plan_only=True,
        will_write=False,
        output_written=False,
        no_output_written=True,
        output_paths_written=(),
        status=status,
        warnings=warnings,
        errors=errors,
        workflow_validation_rows=workflow_validation_rows,
        model_plan_rows=model_plan_rows,
        provenance_rows=_repeated_measures_provenance_rows(
            workflow,
            executed=False,
            plan_only=True,
            source_count=len(workflow.sources),
            loaded_source_count=0,
            method_count=len(workflow.methods),
            repeated_method_count=_repeated_measures_method_count(workflow),
            model_plan_row_count=len(model_plan_rows),
            design_summary_row_count=0,
            factor_summary_row_count=0,
            qc_row_count=0,
            qc_mode="planned",
            source_methods=_repeated_measures_source_methods(model_plan_rows),
            metadata_summary=_repeated_measures_metadata_provenance_summary(model_plan_rows),
        ),
    )


def run_tabular_association_repeated_measures_design_qc(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
    *,
    source_rows_by_id: Mapping[str, Any] | None = None,
    source_inventory_specs: Mapping[str, Any] | Sequence[Mapping[str, Any] | TabularSourceInventorySpec] = (),
    qc_result: Mapping[str, Any] | TabularAssociationQcResult | None = None,
) -> TabularAssociationRepeatedMeasuresDesignQcResult:
    """Inspect repeated-measures/mixed-model design structure without model fitting."""

    try:
        workflow = _coerce_workflow_spec(_unwrap_tabular_association_workflow_document(document))
        inventory_specs = _coerce_source_inventory_specs(source_inventory_specs)
        rows_by_id = _coerce_source_rows_by_id(source_rows_by_id)
    except (TypeError, ValueError) as exc:
        workflow_id = _best_effort_qc_workflow_id(document)
        error_row = AssociationValidationRow(
            level="error",
            status="error",
            code="workflow_parse_error",
            message=str(exc),
            location="workflow",
        )
        return TabularAssociationRepeatedMeasuresDesignQcResult(
            schema_version=SCHEMA_VERSION,
            repeated_measures_plan_version=TABULAR_ASSOCIATION_REPEATED_MEASURES_PLAN_VERSION,
            workflow_id=workflow_id,
            valid=False,
            executed=True,
            plan_only=False,
            will_write=False,
            output_written=False,
            no_output_written=True,
            output_paths_written=(),
            status="error",
            warnings=(),
            errors=(str(exc),),
            workflow_validation_rows=(error_row,),
            model_plan_rows=(),
            source_inventory_rows=(),
            source_load_rows=(),
            design_summary_rows=(),
            factor_summary_rows=(),
            qc_rows=(),
            provenance_rows=_repeated_measures_provenance_rows_for_error(
                workflow_id,
                executed=True,
                plan_only=False,
                qc_mode="run_inline",
            ),
        )

    workflow_validation_rows = _validate_workflow(workflow)
    model_plan_rows = _repeated_measures_model_plan_rows(workflow)
    spec_by_id = _source_inventory_specs_by_id(inventory_specs)
    qc_mode = "supplied" if qc_result is not None else "run_inline"

    source_inventory_rows: list[TabularSourceInventoryRow] = []
    source_load_rows: list[TabularSourceLoadRow] = []
    loaded_by_id: dict[str, dict[str, Any]] = {}
    source_by_id = {source.source_id: source for source in workflow.sources}
    source_ids_to_load = _repeated_measures_source_ids_to_load(workflow, model_plan_rows)

    for source in workflow.sources:
        if source.source_id not in source_ids_to_load:
            continue
        loaded_source = _load_qc_source(
            workflow=workflow,
            source=source,
            source_rows_by_id=rows_by_id,
            inventory_spec=spec_by_id.get(source.source_id),
        )
        loaded_by_id[source.source_id] = loaded_source
        source_inventory_rows.append(_source_inventory_row(workflow, source, loaded_source))
        source_load_rows.append(_source_load_row(workflow, source, loaded_source))

    qc_rows: list[RepeatedMeasuresDesignQcRow] = []
    design_summary_rows: list[RepeatedMeasuresDesignSummaryRow] = []
    factor_summary_rows: list[RepeatedMeasuresFactorSummaryRow] = []
    method_by_id = {method.method_id: method for method in workflow.methods}

    if not _repeated_measures_methods(workflow):
        message = "No repeated-measures or mixed-model methods are declared for design QC."
        qc_rows.append(
            RepeatedMeasuresDesignQcRow(
                workflow_id=workflow.workflow_id,
                source_id=None,
                method_id=None,
                method_name=None,
                model_plan_id=None,
                runtime_backend=RUNTIME_BACKEND_RECORDS,
                status="warning",
                code="no_repeated_mixed_model_methods_declared",
                message=message,
                warnings=(message,),
                metadata={"repeated_measures_plan_version": TABULAR_ASSOCIATION_REPEATED_MEASURES_PLAN_VERSION},
            )
        )

    for model_plan_row in model_plan_rows:
        message = "Model fitting is deferred; this row records design/QC planning metadata only."
        qc_rows.append(
            RepeatedMeasuresDesignQcRow(
                workflow_id=workflow.workflow_id,
                source_id=model_plan_row.source_id,
                method_id=model_plan_row.method_id,
                method_name=model_plan_row.method_name,
                model_plan_id=model_plan_row.model_plan_id,
                runtime_backend=RUNTIME_BACKEND_RECORDS,
                status="deferred",
                code="model_fitting_deferred",
                message=message,
                warnings=(message,),
                errors=model_plan_row.errors,
                metadata={
                    "outcome_id": model_plan_row.outcome_id,
                    "predictor_id": model_plan_row.predictor_id,
                    "method_kind": model_plan_row.method_kind,
                },
            )
        )
        method = method_by_id.get(model_plan_row.method_id)
        if method is not None:
            repeated_metadata = workflow.repeated_measures.metadata if workflow.repeated_measures is not None else {}
            qc_rows.extend(
                _repeated_measures_metadata_qc_rows(
                    workflow=workflow,
                    method=method,
                    model_plan_row=model_plan_row,
                    repeated_metadata=repeated_metadata,
                    loaded_sources_by_id=loaded_by_id,
                )
            )

    for method in _repeated_measures_methods(workflow):
        method_plan_rows = tuple(row for row in model_plan_rows if row.method_id == method.method_id)
        method_source_ids = _repeated_measures_method_source_ids(workflow, method_plan_rows)
        for source_id in method_source_ids:
            source = source_by_id.get(source_id)
            loaded_source = loaded_by_id.get(source_id)
            if source is None:
                message = f"Source {source_id!r} is not declared for repeated-measures design QC."
                qc_rows.append(
                    RepeatedMeasuresDesignQcRow(
                        workflow_id=workflow.workflow_id,
                        source_id=source_id,
                        method_id=method.method_id,
                        method_name=method.method_name,
                        model_plan_id=None,
                        runtime_backend=RUNTIME_BACKEND_RECORDS,
                        status="error",
                        code="repeated_measures_source_not_declared",
                        message=message,
                        errors=(message,),
                    )
                )
                continue
            if loaded_source is None or loaded_source["load_status"] not in {"loaded", "empty"}:
                message = "Source rows are not available for repeated-measures design QC."
                if loaded_source is not None:
                    message = f"repeated_measures_source_not_loaded: {loaded_source['message']}"
                qc_rows.append(
                    RepeatedMeasuresDesignQcRow(
                        workflow_id=workflow.workflow_id,
                        source_id=source_id,
                        method_id=method.method_id,
                        method_name=method.method_name,
                        model_plan_id=None,
                        runtime_backend=RUNTIME_BACKEND_RECORDS,
                        status="error",
                        code="repeated_measures_source_not_loaded",
                        message=message,
                        errors=(message,),
                    )
                )
                continue

            summary_row, factor_rows, method_qc_rows = _repeated_measures_design_rows_for_method_source(
                workflow=workflow,
                method=method,
                source=source,
                loaded_source=loaded_source,
                model_plan_rows=method_plan_rows,
            )
            design_summary_rows.append(summary_row)
            factor_summary_rows.extend(factor_rows)
            qc_rows.extend(method_qc_rows)

    warnings, errors = _repeated_measures_messages(
        workflow_validation_rows=workflow_validation_rows,
        model_plan_rows=model_plan_rows,
        source_load_rows=source_load_rows,
        design_summary_rows=design_summary_rows,
        factor_summary_rows=factor_summary_rows,
        qc_rows=qc_rows,
    )
    status = "error" if errors else ("warning" if warnings else "ok")
    loaded_source_count = sum(1 for row in source_load_rows if row.load_status in {"loaded", "empty"})
    return TabularAssociationRepeatedMeasuresDesignQcResult(
        schema_version=SCHEMA_VERSION,
        repeated_measures_plan_version=TABULAR_ASSOCIATION_REPEATED_MEASURES_PLAN_VERSION,
        workflow_id=workflow.workflow_id,
        valid=not errors,
        executed=True,
        plan_only=False,
        will_write=False,
        output_written=False,
        no_output_written=True,
        output_paths_written=(),
        status=status,
        warnings=warnings,
        errors=errors,
        workflow_validation_rows=workflow_validation_rows,
        model_plan_rows=model_plan_rows,
        source_inventory_rows=source_inventory_rows,
        source_load_rows=source_load_rows,
        design_summary_rows=design_summary_rows,
        factor_summary_rows=factor_summary_rows,
        qc_rows=qc_rows,
        provenance_rows=_repeated_measures_provenance_rows(
            workflow,
            executed=True,
            plan_only=False,
            source_count=len(source_ids_to_load),
            loaded_source_count=loaded_source_count,
            method_count=len(workflow.methods),
            repeated_method_count=_repeated_measures_method_count(workflow),
            model_plan_row_count=len(model_plan_rows),
            design_summary_row_count=len(design_summary_rows),
            factor_summary_row_count=len(factor_summary_rows),
            qc_row_count=len(qc_rows),
            qc_mode=qc_mode,
            source_methods=_repeated_measures_source_methods(model_plan_rows),
            metadata_summary=_repeated_measures_metadata_provenance_summary(model_plan_rows),
        ),
    )


def plan_tabular_association_model_results(
    model_result_rows: Sequence[Any] = (),
    *,
    model_plan_rows: Sequence[Any] = (),
    model_design_metadata: ModelDesignMetadataSpec | Mapping[str, Any] | Sequence[Any] | None = None,
    workflow_id: str | None = None,
) -> TabularAssociationModelResultPlan:
    """Return a no-write Step 11J-C plan for supplied model-result rows only."""

    return TabularAssociationModelResultPlan(
        **_tabular_association_model_result_payload(
            model_result_rows=model_result_rows,
            model_plan_rows=model_plan_rows,
            model_design_metadata=model_design_metadata,
            workflow_id=workflow_id,
            executed=False,
            plan_only=True,
            qc_mode="planned",
        )
    )


def validate_tabular_association_model_result_rows(
    model_result_rows: Sequence[Any] = (),
    *,
    model_plan_rows: Sequence[Any] = (),
    model_design_metadata: ModelDesignMetadataSpec | Mapping[str, Any] | Sequence[Any] | None = None,
    workflow_id: str | None = None,
) -> TabularAssociationModelResultValidationResult:
    """Validate supplied in-memory model-result rows without fitting or writing."""

    return TabularAssociationModelResultValidationResult(
        **_tabular_association_model_result_payload(
            model_result_rows=model_result_rows,
            model_plan_rows=model_plan_rows,
            model_design_metadata=model_design_metadata,
            workflow_id=workflow_id,
            executed=True,
            plan_only=False,
            qc_mode="validate_supplied_rows",
        )
    )


def normalize_tabular_association_model_result_rows(
    model_result_rows: Sequence[Any] = (),
    *,
    model_plan_rows: Sequence[Any] = (),
    model_design_metadata: ModelDesignMetadataSpec | Mapping[str, Any] | Sequence[Any] | None = None,
    workflow_id: str | None = None,
) -> TabularAssociationModelResultContract:
    """Normalize supplied in-memory model-result rows into JSON/TSV-safe contracts."""

    return TabularAssociationModelResultContract(
        **_tabular_association_model_result_payload(
            model_result_rows=model_result_rows,
            model_plan_rows=model_plan_rows,
            model_design_metadata=model_design_metadata,
            workflow_id=workflow_id,
            executed=True,
            plan_only=False,
            qc_mode="normalize_supplied_rows",
        )
    )


def plan_tabular_association_multiplicity(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
    *,
    result_rows: Sequence[Mapping[str, Any]] = (),
    p_value_field: str = "p_value",
    p_value_policy: str = "warn",
) -> TabularAssociationMultiplicityPlan:
    """Return a no-write multiple-testing/FDR plan without computing q-values."""

    del result_rows
    try:
        workflow = _coerce_workflow_spec(_unwrap_tabular_association_workflow_document(document))
        p_value_field = _non_empty_text(p_value_field, field_name="p_value_field")
        p_value_policy = _normalized_choice(
            p_value_policy,
            field_name="p-value policy",
            supported=SUPPORTED_P_VALUE_POLICIES,
        )
    except (TypeError, ValueError) as exc:
        workflow_id = _best_effort_qc_workflow_id(document)
        error_row = AssociationValidationRow(
            level="error",
            status="error",
            code="workflow_parse_error",
            message=str(exc),
            location="workflow",
        )
        return TabularAssociationMultiplicityPlan(
            schema_version=SCHEMA_VERSION,
            workflow_id=workflow_id,
            valid=False,
            executed=False,
            plan_only=True,
            will_write=False,
            output_written=False,
            status="error",
            p_value_field="p_value",
            p_value_policy="warn",
            warnings=(),
            errors=(str(exc),),
            workflow_validation_rows=(error_row,),
            family_plan_rows=(),
            qc_rows=(),
            method_summary_rows=(),
            provenance_rows=_multiplicity_provenance_rows_for_error(
                workflow_id,
                executed=False,
                plan_only=True,
                input_row_count=0,
                qc_mode="planned",
            ),
        )

    workflow_validation_rows = _validate_workflow(workflow)
    family_plan_rows = _multiplicity_family_plan_rows(workflow)
    qc_rows = _multiplicity_plan_qc_rows(workflow, family_plan_rows)
    method_summary_rows = _multiplicity_method_summary_rows(
        workflow,
        family_plan_rows=family_plan_rows,
        family_counts={},
        adjusted_by_family={},
        p_value_policy=p_value_policy,
        executed=False,
        plan_only=True,
    )
    warnings, errors = _multiplicity_messages(
        workflow_validation_rows=workflow_validation_rows,
        family_plan_rows=family_plan_rows,
        qc_rows=qc_rows,
        result_rows=(),
        method_summary_rows=method_summary_rows,
    )
    status = "error" if errors else ("warning" if warnings else "ok")
    return TabularAssociationMultiplicityPlan(
        schema_version=SCHEMA_VERSION,
        workflow_id=workflow.workflow_id,
        valid=not errors,
        executed=False,
        plan_only=True,
        will_write=False,
        output_written=False,
        status=status,
        p_value_field=p_value_field,
        p_value_policy=p_value_policy,
        warnings=warnings,
        errors=errors,
        workflow_validation_rows=workflow_validation_rows,
        family_plan_rows=family_plan_rows,
        qc_rows=qc_rows,
        method_summary_rows=method_summary_rows,
        provenance_rows=_multiplicity_provenance_rows(
            workflow,
            executed=False,
            plan_only=True,
            input_row_count=0,
            family_count=len(family_plan_rows),
            adjusted_row_count=0,
            missing_p_value_count=0,
            invalid_p_value_count=0,
            correction_method_count=_multiplicity_correction_method_count(family_plan_rows),
            qc_mode="planned",
            p_value_field=p_value_field,
            p_value_policy=p_value_policy,
        ),
    )


def run_tabular_association_multiplicity(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
    *,
    result_rows: Sequence[Mapping[str, Any]],
    p_value_field: str = "p_value",
    p_value_policy: str = "warn",
) -> TabularAssociationMultiplicityResult:
    """Compute Benjamini-Hochberg q-values from supplied in-memory result rows.

    The function writes nothing, does not mutate supplied rows, does not compute
    p-values, and keeps runtime execution on standard-library record mappings.
    """

    try:
        workflow = _coerce_workflow_spec(_unwrap_tabular_association_workflow_document(document))
        p_value_field = _non_empty_text(p_value_field, field_name="p_value_field")
        p_value_policy = _normalized_choice(
            p_value_policy,
            field_name="p-value policy",
            supported=SUPPORTED_P_VALUE_POLICIES,
        )
        supplied_rows = _coerce_association_result_rows(result_rows)
    except (TypeError, ValueError) as exc:
        workflow_id = _best_effort_qc_workflow_id(document)
        error_row = AssociationValidationRow(
            level="error",
            status="error",
            code="workflow_parse_error",
            message=str(exc),
            location="workflow",
        )
        return TabularAssociationMultiplicityResult(
            schema_version=SCHEMA_VERSION,
            workflow_id=workflow_id,
            valid=False,
            executed=True,
            plan_only=False,
            will_write=False,
            output_written=False,
            status="error",
            p_value_field="p_value",
            p_value_policy="warn",
            warnings=(),
            errors=(str(exc),),
            workflow_validation_rows=(error_row,),
            family_plan_rows=(),
            input_rows=(),
            result_rows=(),
            qc_rows=(),
            method_summary_rows=(),
            provenance_rows=_multiplicity_provenance_rows_for_error(
                workflow_id,
                executed=True,
                plan_only=False,
                input_row_count=0,
                qc_mode="run_supplied_rows",
            ),
        )

    workflow_validation_rows = _validate_workflow(workflow)
    family_plan_rows = _multiplicity_family_plan_rows(workflow)
    context = _multiplicity_family_context(workflow)
    row_infos = tuple(
        _multiplicity_row_info(
            workflow=workflow,
            row=row,
            input_row_index=input_row_index,
            p_value_field=p_value_field,
            p_value_policy=p_value_policy,
            context=context,
        )
        for input_row_index, row in enumerate(supplied_rows)
    )
    family_counts = _multiplicity_family_counts(row_infos)
    q_values_by_index = _benjamini_hochberg_q_values_by_index(row_infos, family_plan_rows)
    adjusted_by_family: dict[str, int] = {}
    for input_row_index in q_values_by_index:
        family_id = row_infos[input_row_index]["family_id"]
        if family_id is not None:
            adjusted_by_family[family_id] = adjusted_by_family.get(family_id, 0) + 1

    input_rows = tuple(_multiplicity_input_row(info) for info in row_infos)
    result_rows = tuple(
        _multiplicity_result_row(
            info,
            family_counts=family_counts,
            q_values_by_index=q_values_by_index,
            adjusted_by_family=adjusted_by_family,
            p_value_policy=p_value_policy,
        )
        for info in row_infos
    )
    qc_rows = _multiplicity_run_qc_rows(
        workflow=workflow,
        row_infos=row_infos,
        family_plan_rows=family_plan_rows,
        family_counts=family_counts,
        adjusted_by_family=adjusted_by_family,
        p_value_policy=p_value_policy,
    )
    method_summary_rows = _multiplicity_method_summary_rows(
        workflow,
        family_plan_rows=family_plan_rows,
        family_counts=family_counts,
        adjusted_by_family=adjusted_by_family,
        p_value_policy=p_value_policy,
        executed=True,
        plan_only=False,
    )
    warnings, errors = _multiplicity_messages(
        workflow_validation_rows=workflow_validation_rows,
        family_plan_rows=family_plan_rows,
        qc_rows=qc_rows,
        result_rows=result_rows,
        method_summary_rows=method_summary_rows,
    )
    status = "error" if errors else ("warning" if warnings else "ok")
    missing_p_value_count = sum(1 for info in row_infos if info["p_value_status"] == "missing")
    invalid_p_value_count = sum(1 for info in row_infos if info["p_value_status"] == "invalid")
    return TabularAssociationMultiplicityResult(
        schema_version=SCHEMA_VERSION,
        workflow_id=workflow.workflow_id,
        valid=not errors,
        executed=True,
        plan_only=False,
        will_write=False,
        output_written=False,
        status=status,
        p_value_field=p_value_field,
        p_value_policy=p_value_policy,
        warnings=warnings,
        errors=errors,
        workflow_validation_rows=workflow_validation_rows,
        family_plan_rows=family_plan_rows,
        input_rows=input_rows,
        result_rows=result_rows,
        qc_rows=qc_rows,
        method_summary_rows=method_summary_rows,
        provenance_rows=_multiplicity_provenance_rows(
            workflow,
            executed=True,
            plan_only=False,
            input_row_count=len(supplied_rows),
            family_count=len(family_plan_rows),
            adjusted_row_count=len(q_values_by_index),
            missing_p_value_count=missing_p_value_count,
            invalid_p_value_count=invalid_p_value_count,
            correction_method_count=_multiplicity_correction_method_count(family_plan_rows),
            qc_mode="run_supplied_rows",
            p_value_field=p_value_field,
            p_value_policy=p_value_policy,
        ),
    )


_MODEL_RESULT_NUMERIC_FIELDS = (
    "statistic_value",
    "coefficient",
    "standard_error",
    "p_value",
    "q_value",
    "ci_low",
    "ci_high",
    "confidence_level",
    "effect_size",
    "degrees_of_freedom",
    "model_fit_metric_value",
)

_MODEL_RESULT_COUNT_FIELDS = ("observation_count", "participant_count", "cluster_count")

_MODEL_RESULT_TEXT_FIELDS = (
    "workflow_id",
    "result_row_id",
    "result_id",
    "model_id",
    "model_plan_id",
    "method_id",
    "method_name",
    "method_kind",
    "family_id",
    "source_id",
    "outcome_id",
    "outcome_column",
    "predictor_id",
    "predictor_column",
    "term_id",
    "term_label",
    "comparison_id",
    "contrast_id",
    "grouping_id",
    "cluster_id",
    "statistic_name",
    "effect_size_name",
    "model_fit_metric_name",
)

_MODEL_RESULT_ROW_CLASSES: Mapping[str, type[_ModelResultRowBase]] = {
    MODEL_RESULT_KIND_MODEL_FIT_SUMMARY: ModelFitSummaryRow,
    MODEL_RESULT_KIND_FIXED_EFFECT: ModelFixedEffectResultRow,
    MODEL_RESULT_KIND_RANDOM_EFFECT: ModelRandomEffectResultRow,
    MODEL_RESULT_KIND_VARIANCE_COMPONENT: ModelVarianceComponentResultRow,
    MODEL_RESULT_KIND_PLANNED_COMPARISON: ModelPlannedComparisonResultRow,
    MODEL_RESULT_KIND_CONTRAST: ModelContrastResultRow,
}


def _tabular_association_model_result_payload(
    *,
    model_result_rows: Sequence[Any],
    model_plan_rows: Sequence[Any],
    model_design_metadata: ModelDesignMetadataSpec | Mapping[str, Any] | Sequence[Any] | None,
    workflow_id: str | None,
    executed: bool,
    plan_only: bool,
    qc_mode: str,
) -> dict[str, Any]:
    resolved_workflow_id = _optional_text(workflow_id) or "unresolved-workflow"
    raw_rows, row_count, malformed_count, qc_rows = _coerce_model_result_rows(
        model_result_rows,
        workflow_id=resolved_workflow_id,
    )
    resolved_workflow_id = _resolve_model_result_workflow_id(workflow_id, raw_rows)

    model_plan_ids, model_plan_qc_rows = _model_result_plan_reference_ids(
        model_plan_rows,
        workflow_id=resolved_workflow_id,
    )
    design_reference_ids, design_qc_rows = _model_result_design_reference_ids(
        model_design_metadata,
        workflow_id=resolved_workflow_id,
    )
    qc_rows.extend(model_plan_qc_rows)
    qc_rows.extend(design_qc_rows)

    normalized_rows: list[_ModelResultRowBase] = []
    invalid_row_indexes: set[int] = set()
    for input_row_index, row in enumerate(raw_rows):
        normalized_row, row_qc_rows = _normalize_model_result_row(
            row,
            input_row_index=input_row_index,
            workflow_id=resolved_workflow_id,
            model_plan_ids=model_plan_ids,
            model_design_reference_ids=design_reference_ids,
            has_model_plan_rows=bool(model_plan_rows),
            has_model_design_metadata=model_design_metadata is not None,
        )
        qc_rows.extend(row_qc_rows)
        if any(qc_row.status == "error" for qc_row in row_qc_rows):
            invalid_row_indexes.add(input_row_index)
            continue
        if normalized_row is not None:
            normalized_rows.append(normalized_row)
            qc_rows.extend(_model_result_supplied_only_qc_rows(normalized_row, input_row_index=input_row_index))

    kind_counts: dict[str, int] = {}
    for row in normalized_rows:
        kind_counts[row.result_kind] = kind_counts.get(row.result_kind, 0) + 1

    invalid_row_count = malformed_count + len(invalid_row_indexes)
    warnings, errors = _model_result_container_messages(qc_rows)
    status = "error" if errors else ("warning" if warnings else "ok")
    return {
        "schema_version": SCHEMA_VERSION,
        "model_results_contract_version": TABULAR_ASSOCIATION_MODEL_RESULTS_CONTRACT_VERSION,
        "workflow_id": resolved_workflow_id,
        "valid": not errors,
        "executed": executed,
        "plan_only": plan_only,
        "will_write": False,
        "output_written": False,
        "no_output_written": True,
        "output_paths_written": (),
        "status": status,
        "warnings": warnings,
        "errors": errors,
        "model_result_rows": tuple(normalized_rows),
        "qc_rows": tuple(qc_rows),
        "provenance_rows": _model_result_provenance_rows(
            workflow_id=resolved_workflow_id,
            model_result_rows=normalized_rows,
            row_count=row_count,
            valid_row_count=len(normalized_rows),
            invalid_row_count=invalid_row_count,
            result_kind_counts=kind_counts,
            executed=executed,
            plan_only=plan_only,
            qc_mode=qc_mode,
        ),
        "row_count": row_count,
        "valid_row_count": len(normalized_rows),
        "invalid_row_count": invalid_row_count,
        "result_kind_counts": kind_counts,
        "runtime_backend": RUNTIME_BACKEND_RECORDS,
        "supplied_only": True,
        "computed_by_research_analysis": False,
        "model_fitting_performed": False,
    }


def _coerce_model_result_rows(
    value: Sequence[Any],
    *,
    workflow_id: str,
) -> tuple[tuple[dict[str, Any], ...], int, int, list[ModelResultQcRow]]:
    qc_rows: list[ModelResultQcRow] = []
    if value is None:
        return (), 0, 0, qc_rows
    if isinstance(value, (str, bytes, Mapping)):
        message = "model_result_rows must be a sequence of mapping/result rows, not a single mapping or string."
        qc_rows.append(
            _model_result_qc_row(
                workflow_id=workflow_id,
                input_row_index=0,
                status="error",
                code="malformed_model_result_row",
                message=message,
                errors=(message,),
            )
        )
        return (), 1, 1, qc_rows
    try:
        items = tuple(value)
    except TypeError:
        message = "model_result_rows must be an iterable sequence of mapping/result rows."
        qc_rows.append(
            _model_result_qc_row(
                workflow_id=workflow_id,
                input_row_index=0,
                status="error",
                code="malformed_model_result_row",
                message=message,
                errors=(message,),
            )
        )
        return (), 1, 1, qc_rows

    rows: list[dict[str, Any]] = []
    malformed_count = 0
    for input_row_index, item in enumerate(items):
        try:
            row = _model_result_mapping_from_object(item)
        except (TypeError, ValueError) as exc:
            malformed_count += 1
            message = f"malformed_model_result_row: row {input_row_index} is not a supported mapping/result row ({exc})."
            qc_rows.append(
                _model_result_qc_row(
                    workflow_id=workflow_id,
                    input_row_index=input_row_index,
                    status="error",
                    code="malformed_model_result_row",
                    message=message,
                    errors=(message,),
                )
            )
            continue
        rows.append(row)
    return tuple(rows), len(items), malformed_count, qc_rows


def _model_result_mapping_from_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if hasattr(value, "to_dict"):
        row_dict = value.to_dict()
        if not isinstance(row_dict, Mapping):
            raise TypeError("to_dict() did not return a mapping")
        return {str(key): item for key, item in row_dict.items()}
    if is_dataclass(value):
        return {field_.name: getattr(value, field_.name) for field_ in fields(value)}
    raise TypeError("row is not a mapping, dataclass, or object exposing to_dict()")


def _normalize_model_result_row(
    row: Mapping[str, Any],
    *,
    input_row_index: int,
    workflow_id: str,
    model_plan_ids: set[str],
    model_design_reference_ids: Mapping[str, set[str]],
    has_model_plan_rows: bool,
    has_model_design_metadata: bool,
) -> tuple[_ModelResultRowBase | None, tuple[ModelResultQcRow, ...]]:
    qc_rows: list[ModelResultQcRow] = []
    text_values = {
        field_name: _model_result_optional_text_from_row(row, field_name)
        for field_name in _MODEL_RESULT_TEXT_FIELDS
    }
    row_workflow_id = text_values["workflow_id"] or workflow_id
    result_kind = _model_result_optional_text_from_row(row, "result_kind")
    result_row_id = text_values["result_row_id"] or _model_result_optional_text_from_row(row, "row_id")
    result_id = text_values["result_id"]
    model_id = text_values["model_id"]
    model_plan_id = text_values["model_plan_id"]

    def add_qc(
        *,
        status: str,
        code: str,
        message: str,
        field_name: str | None = None,
        warnings: Sequence[str] = (),
        errors: Sequence[str] = (),
    ) -> None:
        qc_rows.append(
            _model_result_qc_row(
                workflow_id=row_workflow_id,
                input_row_index=input_row_index,
                result_row_id=result_row_id,
                result_id=result_id,
                result_kind=result_kind,
                model_id=model_id,
                model_plan_id=model_plan_id,
                method_id=text_values["method_id"],
                family_id=text_values["family_id"],
                term_id=text_values["term_id"],
                comparison_id=text_values["comparison_id"],
                contrast_id=text_values["contrast_id"],
                field_name=field_name,
                status=status,
                code=code,
                message=message,
                warnings=warnings,
                errors=errors,
            )
        )

    for identifier_name, identifier_value in (
        ("workflow_id", row_workflow_id),
        ("result_row_id_or_result_id", result_row_id or result_id),
        ("model_id_or_model_plan_id", model_id or model_plan_id),
    ):
        if not identifier_value:
            message = f"missing_required_identifier: {identifier_name} is required for supplied model-result rows."
            add_qc(
                status="error",
                code="missing_required_identifier",
                message=message,
                field_name=identifier_name,
                errors=(message,),
            )

    if not result_kind:
        message = "missing_required_identifier: result_kind is required for supplied model-result rows."
        add_qc(
            status="error",
            code="missing_required_identifier",
            message=message,
            field_name="result_kind",
            errors=(message,),
        )
    elif result_kind not in SUPPORTED_MODEL_RESULT_KINDS:
        message = f"unsupported_model_result_kind: result_kind {result_kind!r} is not supported."
        add_qc(
            status="error",
            code="unsupported_model_result_kind",
            message=message,
            field_name="result_kind",
            errors=(message,),
        )

    if has_model_plan_rows:
        if not model_plan_id:
            message = "missing_model_plan_reference: model_plan_id is required when model_plan_rows are supplied."
            add_qc(
                status="error",
                code="missing_model_plan_reference",
                message=message,
                field_name="model_plan_id",
                errors=(message,),
            )
        elif model_plan_id not in model_plan_ids:
            message = f"missing_model_plan_reference: model_plan_id {model_plan_id!r} is not present in supplied model_plan_rows."
            add_qc(
                status="error",
                code="missing_model_plan_reference",
                message=message,
                field_name="model_plan_id",
                errors=(message,),
            )

    if has_model_design_metadata:
        term_id = text_values["term_id"]
        comparison_id = text_values["comparison_id"]
        contrast_id = text_values["contrast_id"]
        if term_id and term_id not in model_design_reference_ids["term_ids"]:
            message = f"unknown_model_term_reference: term_id {term_id!r} is not declared in supplied model design metadata."
            add_qc(
                status="error",
                code="unknown_model_term_reference",
                message=message,
                field_name="term_id",
                errors=(message,),
            )
        if comparison_id and comparison_id not in model_design_reference_ids["comparison_ids"]:
            message = (
                f"unknown_model_comparison_reference: comparison_id {comparison_id!r} "
                "is not declared in supplied model design metadata."
            )
            add_qc(
                status="error",
                code="unknown_model_comparison_reference",
                message=message,
                field_name="comparison_id",
                errors=(message,),
            )
        if contrast_id and contrast_id not in model_design_reference_ids["contrast_ids"]:
            message = (
                f"unknown_model_contrast_reference: contrast_id {contrast_id!r} "
                "is not declared in supplied model design metadata."
            )
            add_qc(
                status="error",
                code="unknown_model_contrast_reference",
                message=message,
                field_name="contrast_id",
                errors=(message,),
            )

    numeric_values: dict[str, float | int | None] = {field_name: None for field_name in _MODEL_RESULT_NUMERIC_FIELDS}
    for field_name in _MODEL_RESULT_NUMERIC_FIELDS:
        if field_name not in row or _is_missing_value(row.get(field_name)):
            continue
        status, number = _finite_float(row.get(field_name))
        if status != "valid" or number is None:
            code = "malformed_supplied_numeric_field"
            if field_name == "p_value":
                code = "invalid_supplied_p_value"
            elif field_name == "q_value":
                code = "invalid_supplied_q_value"
            message = f"{code}: {field_name} must be a finite numeric value."
            add_qc(status="error", code=code, message=message, field_name=field_name, errors=(message,))
            continue
        if field_name == "p_value" and (number < 0.0 or number > 1.0):
            message = "invalid_supplied_p_value: p_value must be in [0, 1]."
            add_qc(status="error", code="invalid_supplied_p_value", message=message, field_name=field_name, errors=(message,))
            continue
        if field_name == "q_value" and (number < 0.0 or number > 1.0):
            message = "invalid_supplied_q_value: q_value must be in [0, 1]."
            add_qc(status="error", code="invalid_supplied_q_value", message=message, field_name=field_name, errors=(message,))
            continue
        if field_name == "confidence_level" and (number <= 0.0 or number > 1.0):
            message = "invalid_supplied_confidence_interval: confidence_level must be in (0, 1]."
            add_qc(
                status="error",
                code="invalid_supplied_confidence_interval",
                message=message,
                field_name=field_name,
                errors=(message,),
            )
            continue
        numeric_values[field_name] = _model_result_numeric_value(row.get(field_name), field_name=field_name)

    if numeric_values["ci_low"] is not None and numeric_values["ci_high"] is not None:
        if float(numeric_values["ci_low"]) > float(numeric_values["ci_high"]):
            message = "invalid_supplied_confidence_interval: ci_low must be less than or equal to ci_high."
            add_qc(
                status="error",
                code="invalid_supplied_confidence_interval",
                message=message,
                field_name="ci_low",
                errors=(message,),
            )

    count_values: dict[str, int | None] = {field_name: None for field_name in _MODEL_RESULT_COUNT_FIELDS}
    for field_name in _MODEL_RESULT_COUNT_FIELDS:
        if field_name not in row or _is_missing_value(row.get(field_name)):
            continue
        try:
            count_values[field_name] = _model_result_count_value(row.get(field_name), field_name=field_name)
        except ValueError:
            message = f"malformed_supplied_numeric_field: {field_name} must be a finite non-negative integer."
            add_qc(
                status="error",
                code="malformed_supplied_numeric_field",
                message=message,
                field_name=field_name,
                errors=(message,),
            )

    status_value, field_qc_rows = _model_result_status_value(row, workflow_id=row_workflow_id, input_row_index=input_row_index)
    qc_rows.extend(field_qc_rows)
    warning_values, warning_qc_rows = _model_result_messages_from_row(
        row,
        "warnings",
        workflow_id=row_workflow_id,
        input_row_index=input_row_index,
    )
    error_values, error_qc_rows = _model_result_messages_from_row(
        row,
        "errors",
        workflow_id=row_workflow_id,
        input_row_index=input_row_index,
    )
    qc_rows.extend(warning_qc_rows)
    qc_rows.extend(error_qc_rows)
    metadata, metadata_qc_rows = _model_result_metadata_from_row(
        row,
        workflow_id=row_workflow_id,
        input_row_index=input_row_index,
    )
    qc_rows.extend(metadata_qc_rows)

    if ("p_value" in row or "q_value" in row) and not text_values["family_id"]:
        message = "missing_multiplicity_family_id: family_id is needed for downstream multiplicity compatibility."
        add_qc(status="warning", code="missing_multiplicity_family_id", message=message, field_name="family_id", warnings=(message,))

    if any(qc_row.status == "error" for qc_row in qc_rows):
        return None, tuple(qc_rows)

    row_class = _MODEL_RESULT_ROW_CLASSES[str(result_kind)]
    normalized_row = row_class(
        workflow_id=row_workflow_id,
        result_row_id=result_row_id,
        result_id=result_id,
        model_id=model_id,
        model_plan_id=model_plan_id,
        method_id=text_values["method_id"],
        method_name=text_values["method_name"],
        method_kind=text_values["method_kind"],
        family_id=text_values["family_id"],
        source_id=text_values["source_id"],
        outcome_id=text_values["outcome_id"],
        outcome_column=text_values["outcome_column"],
        predictor_id=text_values["predictor_id"],
        predictor_column=text_values["predictor_column"],
        covariate_ids=_model_result_text_tuple_from_row(row, "covariate_ids"),
        covariate_columns=_model_result_text_tuple_from_row(row, "covariate_columns"),
        term_id=text_values["term_id"],
        term_label=text_values["term_label"],
        comparison_id=text_values["comparison_id"],
        contrast_id=text_values["contrast_id"],
        grouping_id=text_values["grouping_id"],
        cluster_id=text_values["cluster_id"],
        statistic_name=text_values["statistic_name"],
        statistic_value=numeric_values["statistic_value"],
        coefficient=numeric_values["coefficient"],
        standard_error=numeric_values["standard_error"],
        p_value=numeric_values["p_value"],
        q_value=numeric_values["q_value"],
        ci_low=numeric_values["ci_low"],
        ci_high=numeric_values["ci_high"],
        confidence_level=numeric_values["confidence_level"],
        effect_size=numeric_values["effect_size"],
        effect_size_name=text_values["effect_size_name"],
        degrees_of_freedom=numeric_values["degrees_of_freedom"],
        model_fit_metric_name=text_values["model_fit_metric_name"],
        model_fit_metric_value=numeric_values["model_fit_metric_value"],
        observation_count=count_values["observation_count"],
        participant_count=count_values["participant_count"],
        cluster_count=count_values["cluster_count"],
        status=status_value,
        warnings=warning_values,
        errors=error_values,
        metadata=metadata,
    )
    return normalized_row, tuple(qc_rows)


def _model_result_qc_row(
    *,
    workflow_id: str,
    input_row_index: int | None,
    status: str,
    code: str,
    message: str,
    result_row_id: str | None = None,
    result_id: str | None = None,
    result_kind: str | None = None,
    model_id: str | None = None,
    model_plan_id: str | None = None,
    method_id: str | None = None,
    family_id: str | None = None,
    term_id: str | None = None,
    comparison_id: str | None = None,
    contrast_id: str | None = None,
    field_name: str | None = None,
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ModelResultQcRow:
    return ModelResultQcRow(
        workflow_id=workflow_id,
        input_row_index=input_row_index,
        result_row_id=result_row_id,
        result_id=result_id,
        result_kind=result_kind,
        model_id=model_id,
        model_plan_id=model_plan_id,
        method_id=method_id,
        family_id=family_id,
        term_id=term_id,
        comparison_id=comparison_id,
        contrast_id=contrast_id,
        field_name=field_name,
        status=status,
        code=code,
        message=message,
        warnings=warnings,
        errors=errors,
        metadata={} if metadata is None else metadata,
    )


def _model_result_optional_text_from_row(row: Mapping[str, Any], field_name: str) -> str | None:
    if field_name not in row or _is_missing_value(row.get(field_name)):
        return None
    return str(row[field_name]).strip() or None


def _model_result_text_tuple_from_row(row: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    if field_name not in row or _is_missing_value(row.get(field_name)):
        return ()
    try:
        return _text_tuple(row.get(field_name), field_name=field_name)
    except (TypeError, ValueError):
        return ()


def _model_result_numeric_value(value: Any, *, field_name: str) -> float | int:
    if isinstance(value, bool) or _nonfinite_token(value) is not None:
        raise ValueError(f"{field_name} must be a finite numeric value.")
    if isinstance(value, (float, int)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{field_name} must be a finite numeric value.")
        return value
    status, number = _finite_float(value)
    if status != "valid" or number is None:
        raise ValueError(f"{field_name} must be a finite numeric value.")
    return float(number)


def _model_result_count_value(value: Any, *, field_name: str) -> int:
    status, number = _finite_float(value)
    if status != "valid" or number is None or number < 0 or not float(number).is_integer():
        raise ValueError(f"{field_name} must be a finite non-negative integer.")
    return int(number)


def _model_result_message_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a string or sequence of strings, not a mapping.")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a string or sequence of strings.") from exc
    return tuple(str(item).strip() for item in items if str(item).strip())


def _model_result_status_value(
    row: Mapping[str, Any],
    *,
    workflow_id: str,
    input_row_index: int,
) -> tuple[str, tuple[ModelResultQcRow, ...]]:
    if "status" not in row or row.get("status") is None:
        return "supplied", ()
    value = row.get("status")
    if not isinstance(value, str) or not value.strip():
        message = "malformed_model_result_row: status must be a non-empty string when supplied."
        return "supplied", (
            _model_result_qc_row(
                workflow_id=workflow_id,
                input_row_index=input_row_index,
                status="error",
                code="malformed_model_result_row",
                message=message,
                field_name="status",
                errors=(message,),
            ),
        )
    return value.strip(), ()


def _model_result_messages_from_row(
    row: Mapping[str, Any],
    field_name: str,
    *,
    workflow_id: str,
    input_row_index: int,
) -> tuple[tuple[str, ...], tuple[ModelResultQcRow, ...]]:
    if field_name not in row or row.get(field_name) is None:
        return (), ()
    try:
        return _model_result_message_tuple(row.get(field_name), field_name=field_name), ()
    except TypeError as exc:
        message = f"malformed_model_result_row: {exc}"
        return (), (
            _model_result_qc_row(
                workflow_id=workflow_id,
                input_row_index=input_row_index,
                status="error",
                code="malformed_model_result_row",
                message=message,
                field_name=field_name,
                errors=(message,),
            ),
        )


def _model_result_metadata_from_row(
    row: Mapping[str, Any],
    *,
    workflow_id: str,
    input_row_index: int,
) -> tuple[dict[str, Any], tuple[ModelResultQcRow, ...]]:
    if "metadata" not in row or row.get("metadata") is None:
        return {}, ()
    value = row.get("metadata")
    if not isinstance(value, Mapping):
        message = "malformed_model_result_row: metadata must be a mapping when supplied."
        return {}, (
            _model_result_qc_row(
                workflow_id=workflow_id,
                input_row_index=input_row_index,
                status="error",
                code="malformed_model_result_row",
                message=message,
                field_name="metadata",
                errors=(message,),
            ),
        )
    try:
        return _json_safe_mapping(value), ()
    except ValueError as exc:
        message = f"malformed_model_result_row: metadata is not JSON-safe ({exc})."
        return {}, (
            _model_result_qc_row(
                workflow_id=workflow_id,
                input_row_index=input_row_index,
                status="error",
                code="malformed_model_result_row",
                message=message,
                field_name="metadata",
                errors=(message,),
            ),
        )


def _resolve_model_result_workflow_id(workflow_id: str | None, rows: Sequence[Mapping[str, Any]]) -> str:
    explicit = _optional_text(workflow_id)
    if explicit:
        return explicit
    for row in rows:
        candidate = _model_result_optional_text_from_row(row, "workflow_id")
        if candidate:
            return candidate
    return "unresolved-workflow"


def _model_result_plan_reference_ids(
    model_plan_rows: Sequence[Any],
    *,
    workflow_id: str,
) -> tuple[set[str], tuple[ModelResultQcRow, ...]]:
    rows, _, _, qc_rows = _coerce_model_result_rows(model_plan_rows, workflow_id=workflow_id)
    plan_ids = {
        str(model_plan_id)
        for row in rows
        for model_plan_id in (_model_result_optional_text_from_row(row, "model_plan_id"),)
        if model_plan_id
    }
    return plan_ids, tuple(qc_rows)


def _model_result_design_reference_ids(
    model_design_metadata: ModelDesignMetadataSpec | Mapping[str, Any] | Sequence[Any] | None,
    *,
    workflow_id: str,
) -> tuple[dict[str, set[str]], tuple[ModelResultQcRow, ...]]:
    references = {"term_ids": set[str](), "comparison_ids": set[str](), "contrast_ids": set[str]()}
    if model_design_metadata is None:
        return references, ()
    items: tuple[Any, ...]
    if isinstance(model_design_metadata, (ModelDesignMetadataSpec, Mapping)):
        items = (model_design_metadata,)
    elif isinstance(model_design_metadata, (str, bytes)):
        message = "malformed_model_result_row: model_design_metadata must be metadata objects or mappings."
        return references, (
            _model_result_qc_row(
                workflow_id=workflow_id,
                input_row_index=None,
                status="error",
                code="malformed_model_result_row",
                message=message,
                field_name="model_design_metadata",
                errors=(message,),
            ),
        )
    else:
        try:
            items = tuple(model_design_metadata)
        except TypeError:
            items = (model_design_metadata,)

    qc_rows: list[ModelResultQcRow] = []
    for item_index, item in enumerate(items):
        try:
            design = item if isinstance(item, ModelDesignMetadataSpec) else _coerce_model_design_metadata_spec(item)
        except (TypeError, ValueError) as exc:
            message = f"malformed_model_result_row: model_design_metadata item {item_index} is invalid ({exc})."
            qc_rows.append(
                _model_result_qc_row(
                    workflow_id=workflow_id,
                    input_row_index=None,
                    status="error",
                    code="malformed_model_result_row",
                    message=message,
                    field_name="model_design_metadata",
                    errors=(message,),
                )
            )
            continue
        references["term_ids"].update(term.term_id for term in design.fixed_effect_terms)
        references["term_ids"].update(term.term_id for term in design.random_effect_terms)
        references["comparison_ids"].update(comparison.comparison_id for comparison in design.planned_comparisons)
        references["contrast_ids"].update(contrast.contrast_id for contrast in design.contrast_metadata)
    return references, tuple(qc_rows)


def _model_result_supplied_only_qc_rows(
    row: _ModelResultRowBase,
    *,
    input_row_index: int,
) -> tuple[ModelResultQcRow, ...]:
    supplied_message = "supplied_only_model_result: model-result row was supplied by the caller."
    fitting_message = "model_fitting_not_performed: research-analysis did not fit a model for this row."
    common = {
        "workflow_id": row.workflow_id or "unresolved-workflow",
        "input_row_index": input_row_index,
        "result_row_id": row.result_row_id,
        "result_id": row.result_id,
        "result_kind": row.result_kind,
        "model_id": row.model_id,
        "model_plan_id": row.model_plan_id,
        "method_id": row.method_id,
        "family_id": row.family_id,
        "term_id": row.term_id,
        "comparison_id": row.comparison_id,
        "contrast_id": row.contrast_id,
    }
    return (
        _model_result_qc_row(
            **common,
            status="info",
            code="supplied_only_model_result",
            message=supplied_message,
        ),
        _model_result_qc_row(
            **common,
            status="info",
            code="model_fitting_not_performed",
            message=fitting_message,
        ),
    )


def _model_result_container_messages(qc_rows: Sequence[ModelResultQcRow]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    errors: list[str] = []
    for row in qc_rows:
        if row.status == "error":
            errors.append(row.message)
            errors.extend(row.errors)
        elif row.status in {"warning", "deferred"}:
            warnings.append(row.message)
            warnings.extend(row.warnings)
    return _unique_texts(warnings), _unique_texts(errors)


def _model_result_provenance_rows(
    *,
    workflow_id: str,
    model_result_rows: Sequence[_ModelResultRowBase],
    row_count: int,
    valid_row_count: int,
    invalid_row_count: int,
    result_kind_counts: Mapping[str, int],
    executed: bool,
    plan_only: bool,
    qc_mode: str,
) -> tuple[ModelResultProvenanceRow, ...]:
    values: list[tuple[str | None, str | None, str | None, str, Any]] = [
        (None, None, None, "schema_version", SCHEMA_VERSION),
        (None, None, None, "model_results_contract_version", TABULAR_ASSOCIATION_MODEL_RESULTS_CONTRACT_VERSION),
        (None, None, None, "workflow_id", workflow_id),
        (None, None, None, "runtime_backend", RUNTIME_BACKEND_RECORDS),
        (None, None, None, "row_count", row_count),
        (None, None, None, "valid_row_count", valid_row_count),
        (None, None, None, "invalid_row_count", invalid_row_count),
        (None, None, None, "result_kind_counts", dict(result_kind_counts)),
        (None, None, None, "qc_mode", qc_mode),
        (None, None, None, "supplied_only", True),
        (None, None, None, "computed_by_research_analysis", False),
        (None, None, None, "model_fitting_performed", False),
        (None, None, None, "will_write", False),
        (None, None, None, "output_written", False),
        (None, None, None, "no_output_written", True),
        (None, None, None, "output_paths_written", ()),
        (None, None, None, "executed", executed),
        (None, None, None, "plan_only", plan_only),
    ]
    seen_references: set[tuple[str | None, str | None, str | None]] = set()
    for row in model_result_rows:
        reference = (row.model_id, row.model_plan_id, row.method_id)
        if reference in seen_references:
            continue
        seen_references.add(reference)
        if any(reference):
            values.append((*reference, "model_result_reference", {"model_id": row.model_id, "model_plan_id": row.model_plan_id, "method_id": row.method_id}))
    return tuple(
        ModelResultProvenanceRow(
            workflow_id=workflow_id,
            model_id=model_id,
            model_plan_id=model_plan_id,
            method_id=method_id,
            runtime_backend=RUNTIME_BACKEND_RECORDS,
            model_results_contract_version=TABULAR_ASSOCIATION_MODEL_RESULTS_CONTRACT_VERSION,
            supplied_only=True,
            computed_by_research_analysis=False,
            model_fitting_performed=False,
            will_write=False,
            output_written=False,
            no_output_written=True,
            output_paths_written=(),
            key=key,
            value=value,
        )
        for model_id, model_plan_id, method_id, key, value in values
    )


def plan_tabular_association_publication_tables(
    association_rows: Sequence[Any] = (),
    *,
    multiplicity_rows: Sequence[Any] = (),
    qc_rows: Sequence[Any] = (),
    missingness_rows: Sequence[Any] = (),
    provenance_rows: Sequence[Any] = (),
    workflow_id: str | None = None,
    source_rowset_names: Mapping[str, str] | None = None,
    table_spec: PublicationTableSpec | None = None,
    format_spec: PublicationFormatSpec | None = None,
) -> TabularAssociationPublicationPlan:
    """Plan Step 11G publication-table handoff without writing outputs."""

    payload = _tabular_association_publication_payload(
        association_rows=association_rows,
        multiplicity_rows=multiplicity_rows,
        qc_rows=qc_rows,
        missingness_rows=missingness_rows,
        provenance_rows=provenance_rows,
        workflow_id=workflow_id,
        source_rowset_names=source_rowset_names,
        table_spec=table_spec,
        format_spec=format_spec,
        executed=False,
        plan_only=True,
    )
    return TabularAssociationPublicationPlan(**payload)


def build_tabular_association_publication_tables(
    association_rows: Sequence[Any] = (),
    *,
    multiplicity_rows: Sequence[Any] = (),
    qc_rows: Sequence[Any] = (),
    missingness_rows: Sequence[Any] = (),
    provenance_rows: Sequence[Any] = (),
    workflow_id: str | None = None,
    source_rowset_names: Mapping[str, str] | None = None,
    table_spec: PublicationTableSpec | None = None,
    format_spec: PublicationFormatSpec | None = None,
) -> TabularAssociationPublicationResult:
    """Build in-memory Step 11G publication-table handoff rows only."""

    payload = _tabular_association_publication_payload(
        association_rows=association_rows,
        multiplicity_rows=multiplicity_rows,
        qc_rows=qc_rows,
        missingness_rows=missingness_rows,
        provenance_rows=provenance_rows,
        workflow_id=workflow_id,
        source_rowset_names=source_rowset_names,
        table_spec=table_spec,
        format_spec=format_spec,
        executed=True,
        plan_only=False,
    )
    return TabularAssociationPublicationResult(**payload)


def _coerce_association_result_rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        raise TypeError("result_rows must be a sequence of mapping/result rows.")
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError("result_rows must be a sequence of mapping/result rows, not a single mapping or string.")
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise TypeError("result_rows must be an iterable sequence of mapping/result rows.") from exc
    result: list[Mapping[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        if isinstance(row, Mapping):
            result.append(row)
        elif hasattr(row, "to_dict"):
            row_dict = row.to_dict()
            if not isinstance(row_dict, Mapping):
                raise TypeError(f"result_rows item {row_index} to_dict() did not return a mapping.")
            result.append(row_dict)
        elif is_dataclass(row):
            result.append(_json_safe_dataclass(row))
        else:
            raise TypeError(f"result_rows item {row_index} is not a mapping/result row.")
    return tuple(result)


_PUBLICATION_ROWSET_NAMES = {
    "association_rows": "association_rows",
    "multiplicity_rows": "multiplicity_rows",
    "qc_rows": "qc_rows",
    "missingness_rows": "missingness_rows",
    "provenance_rows": "provenance_rows",
}

_PUBLICATION_OUTPUT_TABLE_NAMES = (
    "input_summary_rows",
    "association_table_rows",
    "association_display_rows",
    "association_machine_rows",
    "qc_table_rows",
    "missingness_table_rows",
    "multiplicity_table_rows",
    "provenance_table_rows",
    "manifest_rows",
)

_ASSOCIATION_PUBLICATION_STANDARD_FIELDS = frozenset(
    {
        "workflow_id",
        "method_id",
        "method_kind",
        "method_name",
        "correlation_method",
        "family_id",
        "multiple_testing_family_id",
        "comparison_family_id",
        "source_id",
        "outcome_id",
        "outcome_source_id",
        "outcome_column",
        "predictor_id",
        "predictor_source_id",
        "predictor_column",
        "covariate_ids",
        "covariate_source_ids",
        "covariate_columns",
        "statistic_name",
        "statistic_value",
        "p_value",
        "q_value",
        "n",
        "n_rows",
        "n_used",
        "n_total",
        "status",
        "warnings",
        "errors",
        "result_row_id",
        "pair_id",
        "input_row_index",
        "executed",
        "plan_only",
        "will_write",
        "output_written",
    }
)


def _tabular_association_publication_payload(
    *,
    association_rows: Sequence[Any],
    multiplicity_rows: Sequence[Any],
    qc_rows: Sequence[Any],
    missingness_rows: Sequence[Any],
    provenance_rows: Sequence[Any],
    workflow_id: str | None,
    source_rowset_names: Mapping[str, str] | None,
    table_spec: PublicationTableSpec | None,
    format_spec: PublicationFormatSpec | None,
    executed: bool,
    plan_only: bool,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    source_names = _publication_source_rowset_names(source_rowset_names)
    normalized_rowsets: dict[str, tuple[dict[str, Any], ...]] = {
        "association_rows": (),
        "multiplicity_rows": (),
        "qc_rows": (),
        "missingness_rows": (),
        "provenance_rows": (),
    }
    resolved_workflow_id = _optional_text(workflow_id) or "unresolved-workflow"

    try:
        normalized_rowsets = {
            "association_rows": _coerce_publication_rows(association_rows, field_name="association_rows"),
            "multiplicity_rows": _coerce_publication_rows(multiplicity_rows, field_name="multiplicity_rows"),
            "qc_rows": _coerce_publication_rows(qc_rows, field_name="qc_rows"),
            "missingness_rows": _coerce_publication_rows(missingness_rows, field_name="missingness_rows"),
            "provenance_rows": _coerce_publication_rows(provenance_rows, field_name="provenance_rows"),
        }
        resolved_workflow_id, workflow_warnings, workflow_errors = _resolve_publication_workflow_id(
            explicit_workflow_id=workflow_id,
            rowsets=normalized_rowsets,
        )
        warnings.extend(workflow_warnings)
        errors.extend(workflow_errors)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))

    input_counts = {name: len(rows) for name, rows in normalized_rowsets.items()}
    input_summary_rows = _publication_input_summary_rows(
        workflow_id=resolved_workflow_id,
        source_rowset_names=source_names,
        rowsets=normalized_rowsets,
        executed=executed,
        plan_only=plan_only,
        status="error" if errors else "ok",
        errors=errors,
    )

    association_table_rows: tuple[AssociationPublicationTableRow, ...] = ()
    qc_table_rows: tuple[AssociationPublicationQcTableRow, ...] = ()
    missingness_table_rows: tuple[AssociationPublicationMissingnessTableRow, ...] = ()
    multiplicity_table_rows: tuple[AssociationPublicationMultiplicityTableRow, ...] = ()
    association_display_rows: tuple[Mapping[str, Any], ...] = ()
    association_machine_rows: tuple[Mapping[str, Any], ...] = ()
    column_mappings: tuple[Mapping[str, Any], ...] = ()

    if not errors:
        association_source_rows = _publication_association_source_rows(
            normalized_rowsets["association_rows"],
            warnings=warnings,
        )
        multiplicity_source_rows = normalized_rowsets["multiplicity_rows"]
        matches = _publication_multiplicity_matches(
            association_rows=association_source_rows,
            multiplicity_rows=multiplicity_source_rows,
        )
        warnings.extend(matches["warnings"])
        association_table_rows = tuple(
            _association_publication_table_row(
                workflow_id=resolved_workflow_id,
                row=row,
                row_position=row_position,
                match=matches["matches"].get(row_position),
                executed=executed,
                plan_only=plan_only,
                warnings=warnings,
            )
            for row_position, row in enumerate(association_source_rows)
        )
        qc_table_rows = _publication_qc_table_rows(
            workflow_id=resolved_workflow_id,
            rows=normalized_rowsets["qc_rows"],
            executed=executed,
            plan_only=plan_only,
        )
        missingness_table_rows = _publication_missingness_table_rows(
            workflow_id=resolved_workflow_id,
            rows=normalized_rowsets["missingness_rows"],
            executed=executed,
            plan_only=plan_only,
        )
        multiplicity_table_rows = _publication_multiplicity_table_rows(
            workflow_id=resolved_workflow_id,
            rows=multiplicity_source_rows,
            executed=executed,
            plan_only=plan_only,
            warnings=warnings,
        )
        publication_spec = table_spec or _default_association_publication_table_spec(format_spec=format_spec)
        try:
            publication_rows = build_publication_table_rows(
                (row.to_dict() for row in association_table_rows),
                table_spec=publication_spec,
            )
            association_display_rows = tuple(publication_rows["display_rows"])
            association_machine_rows = tuple(publication_rows["machine_rows"])
            column_mappings = tuple(publication_rows["column_mappings"])
            warnings.extend(str(warning) for warning in publication_rows["warnings"])
            errors.extend(str(error) for error in publication_rows["errors"])
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

    status = "error" if errors else ("warning" if warnings else "ok")
    valid = not errors
    manifest_row_count = len(_PUBLICATION_OUTPUT_TABLE_NAMES)
    supplied_provenance_rows = _publication_supplied_provenance_rows(
        workflow_id=resolved_workflow_id,
        rows=normalized_rowsets["provenance_rows"],
    )
    generated_provenance_rows = _publication_generated_provenance_rows(
        workflow_id=resolved_workflow_id,
        input_counts=input_counts,
        association_result_row_count=len(association_table_rows),
        qc_row_count=len(qc_table_rows),
        missingness_row_count=len(missingness_table_rows),
        multiplicity_row_count=len(multiplicity_table_rows),
        supplied_provenance_row_count=len(supplied_provenance_rows),
        display_row_count=len(association_display_rows),
        machine_row_count=len(association_machine_rows),
        manifest_row_count=manifest_row_count,
        source_rowset_names=source_names,
        executed=executed,
        plan_only=plan_only,
    )
    provenance_table_rows = (*supplied_provenance_rows, *generated_provenance_rows)
    manifest_rows = _publication_manifest_rows(
        workflow_id=resolved_workflow_id,
        input_counts=input_counts,
        association_result_row_count=len(association_table_rows),
        qc_row_count=len(qc_table_rows),
        missingness_row_count=len(missingness_table_rows),
        multiplicity_row_count=len(multiplicity_table_rows),
        provenance_row_count=len(provenance_table_rows),
        display_row_count=len(association_display_rows),
        machine_row_count=len(association_machine_rows),
        source_rowset_names=source_names,
        executed=executed,
        plan_only=plan_only,
        status=status,
        warnings=warnings,
        errors=errors,
        table_row_counts={
            "input_summary_rows": len(input_summary_rows),
            "association_table_rows": len(association_table_rows),
            "association_display_rows": len(association_display_rows),
            "association_machine_rows": len(association_machine_rows),
            "qc_table_rows": len(qc_table_rows),
            "missingness_table_rows": len(missingness_table_rows),
            "multiplicity_table_rows": len(multiplicity_table_rows),
            "provenance_table_rows": len(provenance_table_rows),
            "manifest_rows": manifest_row_count,
        },
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "publication_handoff_schema_version": TABULAR_ASSOCIATION_PUBLICATION_HANDOFF_VERSION,
        "workflow_id": resolved_workflow_id,
        "valid": valid,
        "executed": executed,
        "plan_only": plan_only,
        "will_write": False,
        "output_written": False,
        "no_output_written": True,
        "output_paths_written": (),
        "status": status,
        "warnings": tuple(_unique_text(warnings)),
        "errors": tuple(_unique_text(errors)),
        "input_summary_rows": input_summary_rows,
        "association_table_rows": association_table_rows,
        "association_display_rows": association_display_rows,
        "association_machine_rows": association_machine_rows,
        "qc_table_rows": qc_table_rows,
        "missingness_table_rows": missingness_table_rows,
        "multiplicity_table_rows": multiplicity_table_rows,
        "provenance_table_rows": provenance_table_rows,
        "manifest_rows": manifest_rows,
        "column_mappings": column_mappings,
    }


def _publication_source_rowset_names(source_rowset_names: Mapping[str, str] | None) -> dict[str, str]:
    names = dict(_PUBLICATION_ROWSET_NAMES)
    if source_rowset_names is not None:
        for key, value in source_rowset_names.items():
            normalized_key = _non_empty_text(key, field_name="source_rowset_names key")
            if normalized_key in names:
                names[normalized_key] = _non_empty_text(value, field_name=f"source_rowset_names[{normalized_key}]")
    return names


def _coerce_publication_rows(value: Any, *, field_name: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{field_name} must be a sequence of mapping/dataclass rows, not a single row.")
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable sequence of mapping/dataclass rows.") from exc
    return tuple(_publication_row_mapping(row, field_name=f"{field_name}[{index}]") for index, row in enumerate(rows))


def _publication_row_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): _publication_json_safe(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        row_dict = value.to_dict()
        if not isinstance(row_dict, Mapping):
            raise TypeError(f"{field_name}.to_dict() did not return a mapping.")
        return {str(key): _publication_json_safe(item) for key, item in row_dict.items()}
    if is_dataclass(value):
        return {field_.name: _publication_json_safe(getattr(value, field_.name)) for field_ in fields(value)}
    raise TypeError(f"{field_name} must be a mapping or dataclass-style row.")


def _resolve_publication_workflow_id(
    *,
    explicit_workflow_id: str | None,
    rowsets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    explicit = _optional_text(explicit_workflow_id)
    found_ids = _publication_workflow_ids(rowsets)
    if explicit is not None:
        warnings = ()
        if found_ids and found_ids != {explicit}:
            warnings = (
                "Explicit workflow_id overrides workflow_id values found in supplied publication handoff rows.",
            )
        return explicit, warnings, ()
    if len(found_ids) == 1:
        return next(iter(found_ids)), (), ()
    if len(found_ids) > 1:
        return (
            "unresolved-workflow",
            (),
            (
                "Multiple workflow_id values were found in supplied publication handoff rows: "
                + ", ".join(sorted(found_ids)),
            ),
        )
    return "unresolved-workflow", ("No workflow_id was supplied or found in publication handoff rows.",), ()


def _publication_workflow_ids(rowsets: Mapping[str, Sequence[Mapping[str, Any]]]) -> set[str]:
    workflow_ids: set[str] = set()
    for rows in rowsets.values():
        for row in rows:
            row_workflow_id = _optional_text(row.get("workflow_id"))
            if row_workflow_id is not None:
                workflow_ids.add(row_workflow_id)
            if _optional_text(row.get("key")) == "workflow_id":
                provenance_workflow_id = _optional_text(row.get("value"))
                if provenance_workflow_id is not None:
                    workflow_ids.add(provenance_workflow_id)
    return workflow_ids


def _publication_input_summary_rows(
    *,
    workflow_id: str,
    source_rowset_names: Mapping[str, str],
    rowsets: Mapping[str, Sequence[Mapping[str, Any]]],
    executed: bool,
    plan_only: bool,
    status: str,
    errors: Sequence[str],
) -> tuple[AssociationPublicationInputSummaryRow, ...]:
    return tuple(
        AssociationPublicationInputSummaryRow(
            workflow_id=workflow_id,
            rowset_name=rowset_name,
            source_rowset_name=source_rowset_names[rowset_name],
            row_count=len(rowsets[rowset_name]),
            normalized_row_count=len(rowsets[rowset_name]),
            status=status,
            errors=errors if status == "error" else (),
            executed=executed,
            plan_only=plan_only,
        )
        for rowset_name in _PUBLICATION_ROWSET_NAMES
    )


def _publication_association_source_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    warnings: list[str],
) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for row_position, row in enumerate(rows):
        row_dict = dict(row)
        input_row_index = _publication_int_or_none(row.get("input_row_index"), field_name="input_row_index")
        if input_row_index is None:
            input_row_index = row_position
        elif input_row_index < 0:
            warnings.append(f"association_rows[{row_position}] input_row_index is negative; using positional index.")
            input_row_index = row_position
        row_dict["input_row_index"] = input_row_index
        normalized.append(row_dict)
    return tuple(normalized)


def _publication_multiplicity_matches(
    *,
    association_rows: Sequence[Mapping[str, Any]],
    multiplicity_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    warnings: list[str] = []
    match_by_position: dict[int, Mapping[str, Any] | None] = {}
    association_indexes = {
        key: _publication_key_index(association_rows, key, use_default_input_index=False)
        for key in ("result_row_id", "pair_id", "input_row_index")
    }
    multiplicity_indexes = {
        key: _publication_key_index(multiplicity_rows, key, use_default_input_index=False)
        for key in ("result_row_id", "pair_id", "input_row_index")
    }
    for row_position, row in enumerate(association_rows):
        match_by_position[row_position] = _publication_match_for_row(
            row=row,
            row_position=row_position,
            association_indexes=association_indexes,
            multiplicity_indexes=multiplicity_indexes,
            multiplicity_rows=multiplicity_rows,
            warnings=warnings,
        )
    return {"matches": match_by_position, "warnings": tuple(_unique_text(warnings))}


def _publication_key_index(
    rows: Sequence[Mapping[str, Any]],
    key_name: str,
    *,
    use_default_input_index: bool,
) -> dict[Any, tuple[int, ...]]:
    positions_by_key: dict[Any, list[int]] = {}
    for position, row in enumerate(rows):
        key = _publication_match_key(row, key_name, default_input_index=position if use_default_input_index else None)
        if key is None:
            continue
        positions_by_key.setdefault(key, []).append(position)
    return {key: tuple(positions) for key, positions in positions_by_key.items()}


def _publication_match_for_row(
    *,
    row: Mapping[str, Any],
    row_position: int,
    association_indexes: Mapping[str, Mapping[Any, Sequence[int]]],
    multiplicity_indexes: Mapping[str, Mapping[Any, Sequence[int]]],
    multiplicity_rows: Sequence[Mapping[str, Any]],
    warnings: list[str],
) -> Mapping[str, Any] | None:
    blocked_by_ambiguity = False
    for key_name in ("result_row_id", "pair_id", "input_row_index"):
        key = _publication_match_key(row, key_name, default_input_index=None)
        if key is None:
            continue
        association_positions = tuple(association_indexes[key_name].get(key, ()))
        if len(association_positions) > 1:
            warnings.append(
                f"Association publication q_value match for {key_name}={key!r} is ambiguous across association rows."
            )
            blocked_by_ambiguity = True
            continue
        multiplicity_positions = tuple(multiplicity_indexes[key_name].get(key, ()))
        if len(multiplicity_positions) > 1:
            warnings.append(
                f"Association publication q_value match for {key_name}={key!r} is ambiguous across multiplicity rows."
            )
            blocked_by_ambiguity = True
            continue
        if len(multiplicity_positions) == 1:
            multiplicity_position = multiplicity_positions[0]
            return {
                "row": multiplicity_rows[multiplicity_position],
                "match_field": key_name,
                "match_value": key,
                "multiplicity_position": multiplicity_position,
            }
    if blocked_by_ambiguity:
        return None
    if row_position < len(multiplicity_rows):
        return {
            "row": multiplicity_rows[row_position],
            "match_field": "position",
            "match_value": row_position,
            "multiplicity_position": row_position,
        }
    return None


def _publication_match_key(
    row: Mapping[str, Any],
    key_name: str,
    *,
    default_input_index: int | None,
) -> str | int | None:
    if key_name == "input_row_index":
        return _publication_int_or_none(
            row.get("input_row_index", default_input_index),
            field_name="input_row_index",
        )
    return _optional_text(row.get(key_name))


def _association_publication_table_row(
    *,
    workflow_id: str,
    row: Mapping[str, Any],
    row_position: int,
    match: Mapping[str, Any] | None,
    executed: bool,
    plan_only: bool,
    warnings: list[str],
) -> AssociationPublicationTableRow:
    row_warnings = list(_publication_messages(row.get("warnings")))
    row_errors = list(_publication_messages(row.get("errors")))
    matched_row: Mapping[str, Any] | None = None
    match_field: str | None = None
    match_value: str | int | None = None
    multiplicity_position: int | None = None
    if match is not None:
        matched_row = _as_mapping(match["row"], field_name="multiplicity match row")
        match_field = _optional_text(match.get("match_field"))
        match_value = _publication_json_safe_scalar(match.get("match_value"))
        multiplicity_position = _publication_int_or_none(
            match.get("multiplicity_position"),
            field_name="multiplicity_position",
        )

    p_value = _publication_p_like_or_none(row.get("p_value"), field_name="p_value", warnings=row_warnings)
    if p_value is None and matched_row is not None:
        p_value = _publication_p_like_or_none(matched_row.get("p_value"), field_name="p_value", warnings=row_warnings)
    q_value = None
    if matched_row is not None:
        q_value = _publication_p_like_or_none(matched_row.get("q_value"), field_name="q_value", warnings=row_warnings)
    statistic_value = _publication_number_or_none(
        row.get("statistic_value"),
        field_name="statistic_value",
        warnings=row_warnings,
    )
    input_row_index = _publication_int_or_none(row.get("input_row_index"), field_name="input_row_index")
    if input_row_index is None:
        input_row_index = row_position
    warnings.extend(row_warnings)
    return AssociationPublicationTableRow(
        workflow_id=_optional_text(row.get("workflow_id")) or workflow_id,
        method_id=_optional_text(row.get("method_id")),
        method_kind=_optional_text(row.get("method_kind")),
        method_name=_optional_text(row.get("method_name")) or _optional_text(row.get("correlation_method")),
        family_id=_optional_text(_first_present(row, "family_id", "multiple_testing_family_id", "comparison_family_id")),
        source_id=_optional_text(row.get("source_id")),
        outcome_id=_optional_text(row.get("outcome_id")),
        predictor_id=_optional_text(row.get("predictor_id")),
        covariate_ids=_publication_text_tuple(row.get("covariate_ids")),
        statistic_name=_optional_text(row.get("statistic_name")),
        statistic_value=statistic_value,
        p_value=p_value,
        q_value=q_value,
        n=_publication_int_or_none(_first_present(row, "n", "n_rows"), field_name="n"),
        n_used=_publication_int_or_none(row.get("n_used"), field_name="n_used"),
        n_total=_publication_int_or_none(row.get("n_total"), field_name="n_total"),
        status=_optional_text(row.get("status")) or "ok",
        warnings=row_warnings,
        errors=row_errors,
        result_row_id=_optional_text(row.get("result_row_id")),
        pair_id=_optional_text(row.get("pair_id")),
        input_row_index=input_row_index,
        multiplicity_match_field=match_field,
        multiplicity_match_value=match_value,
        multiplicity_input_row_index=multiplicity_position,
        extra_fields=_publication_extra_fields(row),
        executed=executed,
        plan_only=plan_only,
    )


def _publication_qc_table_rows(
    *,
    workflow_id: str,
    rows: Sequence[Mapping[str, Any]],
    executed: bool,
    plan_only: bool,
) -> tuple[AssociationPublicationQcTableRow, ...]:
    table_rows: list[AssociationPublicationQcTableRow] = []
    for index, row in enumerate(rows):
        input_row_index = _publication_int_or_none(row.get("input_row_index"), field_name="input_row_index")
        if input_row_index is None:
            input_row_index = index
        table_rows.append(
            AssociationPublicationQcTableRow(
                workflow_id=_optional_text(row.get("workflow_id")) or workflow_id,
                rowset_name="qc_rows",
                input_row_index=input_row_index,
                status=_optional_text(row.get("status")) or "ok",
                code=_optional_text(row.get("code")),
                message=_optional_text(row.get("message")),
                warnings=_publication_messages(row.get("warnings")),
                errors=_publication_messages(row.get("errors")),
                row_payload=row,
                executed=executed,
                plan_only=plan_only,
            )
        )
    return tuple(table_rows)


def _publication_missingness_table_rows(
    *,
    workflow_id: str,
    rows: Sequence[Mapping[str, Any]],
    executed: bool,
    plan_only: bool,
) -> tuple[AssociationPublicationMissingnessTableRow, ...]:
    table_rows: list[AssociationPublicationMissingnessTableRow] = []
    for index, row in enumerate(rows):
        input_row_index = _publication_int_or_none(row.get("input_row_index"), field_name="input_row_index")
        if input_row_index is None:
            input_row_index = index
        table_rows.append(
            AssociationPublicationMissingnessTableRow(
                workflow_id=_optional_text(row.get("workflow_id")) or workflow_id,
                source_id=_optional_text(row.get("source_id")),
                column_name=_optional_text(row.get("column_name")),
                role=_optional_text(row.get("role")),
                missing_count=_publication_int_or_none(row.get("missing_count"), field_name="missing_count"),
                nonmissing_count=_publication_int_or_none(row.get("nonmissing_count"), field_name="nonmissing_count"),
                total_count=_publication_int_or_none(row.get("total_count"), field_name="total_count"),
                status=_optional_text(row.get("status")) or "ok",
                code=_optional_text(row.get("code")),
                message=_optional_text(row.get("message")),
                input_row_index=input_row_index,
                row_payload=row,
                executed=executed,
                plan_only=plan_only,
            )
        )
    return tuple(table_rows)


def _publication_multiplicity_table_rows(
    *,
    workflow_id: str,
    rows: Sequence[Mapping[str, Any]],
    executed: bool,
    plan_only: bool,
    warnings: list[str],
) -> tuple[AssociationPublicationMultiplicityTableRow, ...]:
    table_rows: list[AssociationPublicationMultiplicityTableRow] = []
    for index, row in enumerate(rows):
        row_warnings = list(_publication_messages(row.get("warnings")))
        p_value = _publication_p_like_or_none(row.get("p_value"), field_name="p_value", warnings=row_warnings)
        q_value = _publication_p_like_or_none(row.get("q_value"), field_name="q_value", warnings=row_warnings)
        warnings.extend(row_warnings)
        table_rows.append(
            AssociationPublicationMultiplicityTableRow(
                workflow_id=_optional_text(row.get("workflow_id")) or workflow_id,
                family_id=_optional_text(row.get("family_id")),
                result_row_id=_optional_text(row.get("result_row_id")),
                pair_id=_optional_text(row.get("pair_id")),
                input_row_index=_publication_int_or_none(row.get("input_row_index"), field_name="input_row_index"),
                p_value=p_value,
                q_value=q_value,
                status=_optional_text(row.get("status")) or "ok",
                code=_optional_text(row.get("code")),
                warnings=row_warnings,
                errors=_publication_messages(row.get("errors")),
                row_payload=row,
                executed=executed,
                plan_only=plan_only,
            )
        )
    return tuple(table_rows)


def _publication_supplied_provenance_rows(
    *,
    workflow_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[AssociationPublicationProvenanceRow, ...]:
    provenance_rows: list[AssociationPublicationProvenanceRow] = []
    for index, row in enumerate(rows):
        key = _optional_text(row.get("key")) or f"supplied_provenance_row_{index}"
        value = row.get("value") if "value" in row else row
        provenance_rows.append(
            AssociationPublicationProvenanceRow(
                workflow_id=_optional_text(row.get("workflow_id")) or workflow_id,
                key=key,
                value=value,
                source=_optional_text(row.get("source")) or "supplied_publication_handoff_provenance",
                input_row_index=index,
            )
        )
    return tuple(provenance_rows)


def _publication_generated_provenance_rows(
    *,
    workflow_id: str,
    input_counts: Mapping[str, int],
    association_result_row_count: int,
    qc_row_count: int,
    missingness_row_count: int,
    multiplicity_row_count: int,
    supplied_provenance_row_count: int,
    display_row_count: int,
    machine_row_count: int,
    manifest_row_count: int,
    source_rowset_names: Mapping[str, str],
    executed: bool,
    plan_only: bool,
) -> tuple[AssociationPublicationProvenanceRow, ...]:
    generated_row_count = 21
    generated_values: tuple[tuple[str, Any], ...] = (
        ("tabular_association_schema_version", SCHEMA_VERSION),
        ("publication_handoff_schema_version", TABULAR_ASSOCIATION_PUBLICATION_HANDOFF_VERSION),
        ("workflow_id", workflow_id),
        ("input_row_counts", dict(input_counts)),
        ("association_result_row_count", association_result_row_count),
        ("qc_row_count", qc_row_count),
        ("missingness_row_count", missingness_row_count),
        ("multiplicity_row_count", multiplicity_row_count),
        ("provenance_row_count", supplied_provenance_row_count + generated_row_count),
        ("display_row_count", display_row_count),
        ("machine_row_count", machine_row_count),
        ("manifest_row_count", manifest_row_count),
        ("output_table_names", _PUBLICATION_OUTPUT_TABLE_NAMES),
        ("source_rowset_names", dict(source_rowset_names)),
        ("runtime_backend", RUNTIME_BACKEND_RECORDS),
        ("executed", executed),
        ("plan_only", plan_only),
        ("will_write", False),
        ("output_written", False),
        ("no_output_written", True),
        ("output_paths_written", ()),
    )
    return tuple(
        AssociationPublicationProvenanceRow(workflow_id=workflow_id, key=key, value=value)
        for key, value in generated_values
    )


def _publication_manifest_rows(
    *,
    workflow_id: str,
    input_counts: Mapping[str, int],
    association_result_row_count: int,
    qc_row_count: int,
    missingness_row_count: int,
    multiplicity_row_count: int,
    provenance_row_count: int,
    display_row_count: int,
    machine_row_count: int,
    source_rowset_names: Mapping[str, str],
    executed: bool,
    plan_only: bool,
    status: str,
    warnings: Sequence[str],
    errors: Sequence[str],
    table_row_counts: Mapping[str, int],
) -> tuple[AssociationPublicationManifestRow, ...]:
    manifest_row_count = len(_PUBLICATION_OUTPUT_TABLE_NAMES)
    return tuple(
        AssociationPublicationManifestRow(
            workflow_id=workflow_id,
            table_name=table_name,
            row_count=table_row_counts.get(table_name, 0),
            tabular_association_schema_version=SCHEMA_VERSION,
            publication_handoff_schema_version=TABULAR_ASSOCIATION_PUBLICATION_HANDOFF_VERSION,
            input_row_counts=input_counts,
            association_result_row_count=association_result_row_count,
            qc_row_count=qc_row_count,
            missingness_row_count=missingness_row_count,
            multiplicity_row_count=multiplicity_row_count,
            provenance_row_count=provenance_row_count,
            display_row_count=display_row_count,
            machine_row_count=machine_row_count,
            manifest_row_count=manifest_row_count,
            output_table_names=_PUBLICATION_OUTPUT_TABLE_NAMES,
            source_rowset_names=source_rowset_names,
            executed=executed,
            plan_only=plan_only,
            will_write=False,
            output_written=False,
            no_output_written=True,
            output_paths_written=(),
            status=status,
            warnings=warnings,
            errors=errors,
        )
        for table_name in _PUBLICATION_OUTPUT_TABLE_NAMES
    )


def _default_association_publication_table_spec(
    *,
    format_spec: PublicationFormatSpec | None,
) -> PublicationTableSpec:
    table_format = format_spec or PublicationFormatSpec(missing_value="")
    return PublicationTableSpec(
        table_id="tabular-association-publication-handoff",
        columns=(
            PublicationColumnSpec(output_name="workflow_id", source="workflow_id"),
            PublicationColumnSpec(output_name="method_id", source="method_id"),
            PublicationColumnSpec(output_name="method_kind", source="method_kind"),
            PublicationColumnSpec(output_name="method_name", source="method_name"),
            PublicationColumnSpec(output_name="family_id", source="family_id"),
            PublicationColumnSpec(output_name="source_id", source="source_id"),
            PublicationColumnSpec(output_name="outcome_id", source="outcome_id"),
            PublicationColumnSpec(output_name="predictor_id", source="predictor_id"),
            PublicationColumnSpec(output_name="covariate_ids", source="covariate_ids"),
            PublicationColumnSpec(output_name="statistic_name", source="statistic_name"),
            PublicationColumnSpec(
                output_name="statistic_value",
                source="statistic_value",
                column_type="numeric",
                numeric_format=NumericFormatSpec(precision=3),
            ),
            PublicationColumnSpec(
                output_name="p_value",
                source="p_value",
                column_type="p_value",
                p_value_format=PValueFormatSpec(precision=3, threshold=0.001),
            ),
            PublicationColumnSpec(
                output_name="q_value",
                source="q_value",
                column_type="q_value",
                q_value_format=PValueFormatSpec(precision=3, threshold=0.001),
            ),
            PublicationColumnSpec(
                output_name="n",
                source="n",
                column_type="numeric",
                numeric_format=NumericFormatSpec(precision=0),
            ),
            PublicationColumnSpec(
                output_name="n_used",
                source="n_used",
                column_type="numeric",
                numeric_format=NumericFormatSpec(precision=0),
            ),
            PublicationColumnSpec(
                output_name="n_total",
                source="n_total",
                column_type="numeric",
                numeric_format=NumericFormatSpec(precision=0),
            ),
            PublicationColumnSpec(output_name="status", source="status"),
            PublicationColumnSpec(output_name="warnings", source="warnings"),
            PublicationColumnSpec(output_name="errors", source="errors"),
            PublicationColumnSpec(output_name="result_row_id", source="result_row_id"),
            PublicationColumnSpec(output_name="pair_id", source="pair_id"),
            PublicationColumnSpec(
                output_name="input_row_index",
                source="input_row_index",
                column_type="numeric",
                numeric_format=NumericFormatSpec(precision=0),
            ),
        ),
        format=table_format,
    )


def _publication_extra_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _publication_json_safe(value)
        for key, value in row.items()
        if str(key) not in _ASSOCIATION_PUBLICATION_STANDARD_FIELDS
    }


def _publication_text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        text = _optional_text(value)
        return (text,) if text is not None else ()
    try:
        values = tuple(value)
    except TypeError:
        text = _optional_text(value)
        return (text,) if text is not None else ()
    return tuple(text for item in values if (text := _optional_text(item)) is not None)


def _publication_messages(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, (str, bytes)):
        text = _optional_text(value)
        return (text,) if text is not None else ()
    try:
        values = tuple(value)
    except TypeError:
        text = _optional_text(value)
        return (text,) if text is not None else ()
    return tuple(text for item in values if (text := _optional_text(item)) is not None)


def _publication_p_like_or_none(value: Any, *, field_name: str, warnings: list[str]) -> float | None:
    number = _publication_number_or_none(value, field_name=field_name, warnings=warnings)
    if number is None:
        return None
    number_float = float(number)
    if number_float < 0.0 or number_float > 1.0:
        warnings.append(f"{field_name} value {number!r} is outside [0, 1] and was omitted.")
        return None
    return number_float


def _publication_int_or_none(value: Any, *, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            number = float(stripped)
        except ValueError:
            return None
        if not math.isfinite(number) or not number.is_integer():
            return None
        return int(number)
    return None


def _publication_number_or_none(value: Any, *, field_name: str, warnings: list[str]) -> float | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        warnings.append(f"{field_name} value {value!r} is not numeric and was omitted.")
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            warnings.append(f"{field_name} value is non-finite and was omitted.")
            return None
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            number = float(stripped)
        except ValueError:
            warnings.append(f"{field_name} value {value!r} is not numeric and was omitted.")
            return None
        if not math.isfinite(number):
            warnings.append(f"{field_name} value is non-finite and was omitted.")
            return None
        if stripped.startswith(("-", "+")):
            unsigned = stripped[1:]
        else:
            unsigned = stripped
        if unsigned.isdecimal():
            return int(number)
        return number
    warnings.append(f"{field_name} value {value!r} is not numeric and was omitted.")
    return None


def _publication_json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _publication_json_safe(item) for key, item in value.items()}


def _publication_json_safe_scalar(value: Any) -> Any:
    safe = _publication_json_safe(value)
    if isinstance(safe, (str, int, float, bool)) or safe is None:
        return safe
    return json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _publication_json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {field_.name: _publication_json_safe(getattr(value, field_.name)) for field_ in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _publication_json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_publication_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _unique_text(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return tuple(unique)


def _multiplicity_family_plan_rows(
    workflow: TabularAssociationWorkflowSpec,
) -> tuple[AssociationMultiplicityFamilyPlanRow, ...]:
    family_by_id = {family.family_id: family for family in workflow.families}
    multiple_testing_by_id = {spec.family_id: spec for spec in workflow.multiple_testing}
    ordered_family_ids: list[str] = []
    for family in workflow.families:
        if family.family_id not in ordered_family_ids:
            ordered_family_ids.append(family.family_id)
    for spec in workflow.multiple_testing:
        if spec.family_id not in ordered_family_ids:
            ordered_family_ids.append(spec.family_id)

    rows: list[AssociationMultiplicityFamilyPlanRow] = []
    for family_id in ordered_family_ids:
        family = family_by_id.get(family_id)
        spec = multiple_testing_by_id.get(family_id)
        method_ids = family.method_ids if family is not None else ()
        multiple_testing_method = spec.method if spec is not None else None
        correction_method = _multiplicity_correction_method(multiple_testing_method)
        warnings: tuple[str, ...] = ()
        errors: tuple[str, ...] = ()
        executable = spec is not None and multiple_testing_method in BENJAMINI_HOCHBERG_METHODS
        deferred = not executable
        status = "planned" if executable else "deferred"
        code = "multiple_testing_family_planned" if executable else "multiple_testing_method_deferred"
        if spec is None:
            code = "missing_multiple_testing_spec"
            warnings = (f"{code}: family {family_id!r} has no MultipleTestingSpec declaration.",)
        elif not executable:
            warnings = (
                f"multiple_testing_method_deferred: method {multiple_testing_method!r} for family {family_id!r} "
                "is declared but not implemented in Step 11F.",
            )
        rows.append(
            AssociationMultiplicityFamilyPlanRow(
                workflow_id=workflow.workflow_id,
                family_id=family_id,
                multiple_testing_method=multiple_testing_method,
                correction_method=correction_method,
                method_ids=method_ids,
                declared_in_families=family is not None,
                declared_in_multiple_testing=spec is not None,
                executable=executable,
                deferred=deferred,
                status=status,
                code=code,
                warnings=warnings,
                errors=errors,
                executed=False,
                plan_only=True,
            )
        )
    return tuple(rows)


def _multiplicity_plan_qc_rows(
    workflow: TabularAssociationWorkflowSpec,
    family_plan_rows: Sequence[AssociationMultiplicityFamilyPlanRow],
) -> tuple[AssociationMultiplicityQcRow, ...]:
    rows: list[AssociationMultiplicityQcRow] = []
    for row in family_plan_rows:
        if row.code not in {"missing_multiple_testing_spec", "multiple_testing_method_deferred"}:
            continue
        message = row.warnings[0] if row.warnings else _multiplicity_qc_message(row.code)
        rows.append(
            AssociationMultiplicityQcRow(
                workflow_id=workflow.workflow_id,
                family_id=row.family_id,
                multiple_testing_method=row.multiple_testing_method,
                correction_method=row.correction_method,
                result_row_id=None,
                input_row_index=None,
                status="warning",
                code=row.code,
                message=message,
                warnings=(message,),
                executed=False,
                plan_only=True,
            )
        )
    return tuple(rows)


def _multiplicity_correction_method(method: str | None) -> str | None:
    if method is None:
        return None
    return "benjamini_hochberg" if method in BENJAMINI_HOCHBERG_METHODS else method


def _multiplicity_family_context(workflow: TabularAssociationWorkflowSpec) -> dict[str, Any]:
    method_to_family_ids: dict[str, list[str]] = {}
    for family in workflow.families:
        for method_id in family.method_ids:
            method_to_family_ids.setdefault(method_id, []).append(family.family_id)
    return {
        "family_by_id": {family.family_id: family for family in workflow.families},
        "multiple_testing_by_id": {spec.family_id: spec for spec in workflow.multiple_testing},
        "method_by_id": {method.method_id: method for method in workflow.methods},
        "method_to_family_ids": method_to_family_ids,
        "declared_family_ids": {family.family_id for family in workflow.families}
        | {spec.family_id for spec in workflow.multiple_testing},
    }


def _multiplicity_row_info(
    *,
    workflow: TabularAssociationWorkflowSpec,
    row: Mapping[str, Any],
    input_row_index: int,
    p_value_field: str,
    p_value_policy: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    method_by_id: Mapping[str, AssociationMethodSpec] = context["method_by_id"]
    multiple_testing_by_id: Mapping[str, MultipleTestingSpec] = context["multiple_testing_by_id"]
    family_by_id: Mapping[str, AssociationFamilySpec] = context["family_by_id"]
    declared_family_ids: set[str] = context["declared_family_ids"]
    method_to_family_ids: Mapping[str, list[str]] = context["method_to_family_ids"]

    method_id = _association_row_optional_text(row, "method_id")
    method = method_by_id.get(method_id or "")
    family_id, family_code, family_message = _resolve_multiplicity_family_id(
        row=row,
        method_id=method_id,
        method_to_family_ids=method_to_family_ids,
        declared_family_ids=declared_family_ids,
        family_by_id=family_by_id,
        multiple_testing_by_id=multiple_testing_by_id,
    )
    multiple_testing_spec = multiple_testing_by_id.get(family_id or "")
    multiple_testing_method = multiple_testing_spec.method if multiple_testing_spec is not None else None
    correction_method = _multiplicity_correction_method(multiple_testing_method)
    p_value_status, p_value, p_value_message = _p_value_from_row(row, p_value_field=p_value_field)
    method_name = _association_row_optional_text(row, "method_name", "association_method", "method")
    method_kind = _association_row_optional_text(row, "method_kind")
    if method is not None:
        method_name = method_name or method.method_name
        method_kind = method_kind or _multiplicity_method_kind(method.method_name)
    method_deferred = multiple_testing_method is not None and multiple_testing_method not in BENJAMINI_HOCHBERG_METHODS
    method_deferred_message = ""
    if method_deferred:
        method_deferred_message = (
            f"multiple_testing_method_deferred: method {multiple_testing_method!r} for family {family_id!r} "
            "is declared but not implemented in Step 11F."
        )
    return {
        "workflow_id": _association_row_optional_text(row, "workflow_id") or workflow.workflow_id,
        "family_id": family_id,
        "family_code": family_code,
        "family_message": family_message,
        "multiple_testing_method": multiple_testing_method,
        "correction_method": correction_method,
        "method_deferred": method_deferred,
        "method_deferred_message": method_deferred_message,
        "result_row_id": _association_row_optional_text(row, "result_row_id", "row_id", "pair_id", "association_id"),
        "input_row_index": input_row_index,
        "method_id": method_id,
        "method_kind": method_kind,
        "method_name": method_name,
        "source_id": _association_row_optional_text(row, "source_id"),
        "outcome_id": _association_row_optional_text(row, "outcome_id"),
        "predictor_id": _association_row_optional_text(row, "predictor_id"),
        "covariate_ids": _association_row_text_tuple(row, "covariate_ids", "covariate_id"),
        "statistic_name": _association_row_optional_text(row, "statistic_name"),
        "statistic_value": _association_row_finite_number(row, "statistic_value"),
        "p_value_field": p_value_field,
        "p_value_status": p_value_status,
        "p_value": p_value,
        "p_value_message": p_value_message,
        "p_value_policy": p_value_policy,
    }


def _resolve_multiplicity_family_id(
    *,
    row: Mapping[str, Any],
    method_id: str | None,
    method_to_family_ids: Mapping[str, list[str]],
    declared_family_ids: set[str],
    family_by_id: Mapping[str, AssociationFamilySpec],
    multiple_testing_by_id: Mapping[str, MultipleTestingSpec],
) -> tuple[str | None, str | None, str | None]:
    for key in ("multiple_testing_family_id", "comparison_family_id", "family_id"):
        family_id = _association_row_optional_text(row, key)
        if family_id:
            break
    else:
        family_matches = tuple(method_to_family_ids.get(method_id or "", ()))
        family_id = family_matches[0] if len(family_matches) == 1 else None
        if family_matches and family_id is None:
            return (
                None,
                "missing_family_id",
                f"missing_family_id: method {method_id!r} did not uniquely resolve one association family.",
            )
    if family_id is None:
        return None, "missing_family_id", "missing_family_id: association result row has no resolvable family id."
    if family_id not in declared_family_ids:
        return family_id, "undeclared_family_id", f"undeclared_family_id: family {family_id!r} is not declared."
    if family_id in family_by_id and family_id not in multiple_testing_by_id:
        return (
            family_id,
            "missing_multiple_testing_spec",
            f"missing_multiple_testing_spec: family {family_id!r} has no MultipleTestingSpec declaration.",
        )
    return family_id, None, None


def _multiplicity_method_kind(method_name: str) -> str:
    if method_name in {METHOD_PEARSON, METHOD_SPEARMAN}:
        return "correlation"
    return method_name


def _association_row_optional_text(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if _is_missing_value(value):
            continue
        return str(value).strip() or None
    return None


def _association_row_text_tuple(row: Mapping[str, Any], *keys: str) -> tuple[str, ...]:
    for key in keys:
        if key in row and not _is_missing_value(row[key]):
            try:
                return _text_tuple(row[key], field_name=key)
            except (TypeError, ValueError):
                return ()
    return ()


def _association_row_finite_number(row: Mapping[str, Any], key: str) -> float | None:
    if key not in row:
        return None
    status, number = _finite_float_or_missing(row.get(key))
    return number if status == "valid" else None


def _p_value_from_row(row: Mapping[str, Any], *, p_value_field: str) -> tuple[str, float | None, str]:
    if p_value_field not in row or _is_missing_value(row.get(p_value_field)):
        return "missing", None, f"missing_p_value: row has no valid {p_value_field!r} value."
    value = row.get(p_value_field)
    if isinstance(value, bool):
        return "invalid", None, f"invalid_p_value: {p_value_field!r} cannot be a boolean value."
    status, number = _finite_float(value)
    if status != "valid" or number is None:
        return "invalid", None, f"invalid_p_value: {p_value_field!r} must be a finite numeric value in [0, 1]."
    if number < 0.0 or number > 1.0:
        return "invalid", None, f"invalid_p_value: {p_value_field!r} must be in [0, 1]."
    return "valid", float(number), "valid_p_value: p-value was supplied by the input row."


def _multiplicity_input_row(info: Mapping[str, Any]) -> AssociationMultiplicityInputRow:
    status = "ok" if info["p_value_status"] == "valid" else "warning"
    code = "valid_p_value"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    if info["p_value_status"] == "missing":
        code = "missing_p_value"
        if info["p_value_policy"] == "error":
            status = "error"
            errors = (str(info["p_value_message"]),)
        else:
            warnings = (str(info["p_value_message"]),)
    elif info["p_value_status"] == "invalid":
        code = "invalid_p_value"
        if info["p_value_policy"] == "error":
            status = "error"
            errors = (str(info["p_value_message"]),)
        else:
            warnings = (str(info["p_value_message"]),)
    return AssociationMultiplicityInputRow(
        workflow_id=str(info["workflow_id"]),
        family_id=info["family_id"],
        multiple_testing_method=info["multiple_testing_method"],
        correction_method=info["correction_method"],
        result_row_id=info["result_row_id"],
        input_row_index=int(info["input_row_index"]),
        method_id=info["method_id"],
        method_kind=info["method_kind"],
        method_name=info["method_name"],
        source_id=info["source_id"],
        outcome_id=info["outcome_id"],
        predictor_id=info["predictor_id"],
        covariate_ids=info["covariate_ids"],
        statistic_name=info["statistic_name"],
        statistic_value=info["statistic_value"],
        p_value_field=str(info["p_value_field"]),
        p_value=info["p_value"],
        p_value_status=str(info["p_value_status"]),
        status=status,
        code=code,
        warnings=warnings,
        errors=errors,
    )


def _multiplicity_family_counts(row_infos: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for info in row_infos:
        family_id = info["family_id"]
        if family_id is None:
            continue
        family_counts = counts.setdefault(
            str(family_id),
            {"n_family_total": 0, "n_valid_p": 0, "n_missing_p": 0, "n_invalid_p": 0},
        )
        family_counts["n_family_total"] += 1
        if info["p_value_status"] == "valid":
            family_counts["n_valid_p"] += 1
        elif info["p_value_status"] == "missing":
            family_counts["n_missing_p"] += 1
        else:
            family_counts["n_invalid_p"] += 1
    return counts


def _benjamini_hochberg_q_values_by_index(
    row_infos: Sequence[Mapping[str, Any]],
    family_plan_rows: Sequence[AssociationMultiplicityFamilyPlanRow],
) -> dict[int, float]:
    q_values: dict[int, float] = {}
    executable_family_ids = {row.family_id for row in family_plan_rows if row.executable and row.correction_method == "benjamini_hochberg"}
    for family_id in executable_family_ids:
        family_p_values: list[tuple[int, float]] = []
        for info in row_infos:
            if info["family_id"] != family_id or info["family_code"] is not None or info["method_deferred"]:
                continue
            if info["p_value_status"] == "valid" and info["p_value"] is not None:
                family_p_values.append((int(info["input_row_index"]), float(info["p_value"])))
        q_values.update(_benjamini_hochberg_q_values(family_p_values))
    return q_values


def _benjamini_hochberg_q_values(indexed_p_values: Sequence[tuple[int, float]]) -> dict[int, float]:
    if not indexed_p_values:
        return {}
    sorted_values = sorted(indexed_p_values, key=lambda item: (item[1], item[0]))
    total = len(sorted_values)
    q_values: dict[int, float] = {}
    running_min = 1.0
    for rank_from_one, (input_row_index, p_value) in reversed(tuple(enumerate(sorted_values, start=1))):
        adjusted = p_value * total / rank_from_one
        if adjusted < running_min:
            running_min = adjusted
        q_values[input_row_index] = min(1.0, max(0.0, running_min))
    return q_values


def _multiplicity_result_row(
    info: Mapping[str, Any],
    *,
    family_counts: Mapping[str, Mapping[str, int]],
    q_values_by_index: Mapping[int, float],
    adjusted_by_family: Mapping[str, int],
    p_value_policy: str,
) -> AssociationMultiplicityResultRow:
    family_id = info["family_id"]
    counts = family_counts.get(str(family_id), {}) if family_id is not None else {}
    n_family_total = int(counts.get("n_family_total", 0))
    n_valid_p = int(counts.get("n_valid_p", 0))
    n_missing_p = int(counts.get("n_missing_p", 0))
    n_invalid_p = int(counts.get("n_invalid_p", 0))
    n_adjusted = int(adjusted_by_family.get(str(family_id), 0)) if family_id is not None else 0
    input_row_index = int(info["input_row_index"])
    q_value = q_values_by_index.get(input_row_index)
    warnings: list[str] = []
    errors: list[str] = []
    status = "ok"
    code = "benjamini_hochberg_adjusted" if q_value is not None else "not_adjusted"
    if info["family_code"] is not None:
        status = "deferred"
        code = str(info["family_code"])
        warnings.append(str(info["family_message"]))
    elif info["method_deferred"]:
        status = "deferred"
        code = "multiple_testing_method_deferred"
        warnings.append(str(info["method_deferred_message"]))
    elif info["p_value_status"] == "missing":
        code = "missing_p_value"
        message = str(info["p_value_message"])
        if p_value_policy == "error":
            status = "error"
            errors.append(message)
        else:
            status = "warning"
            warnings.append(message)
    elif info["p_value_status"] == "invalid":
        code = "invalid_p_value"
        message = str(info["p_value_message"])
        if p_value_policy == "error":
            status = "error"
            errors.append(message)
        else:
            status = "warning"
            warnings.append(message)
    elif q_value is None:
        status = "deferred"
        code = "no_valid_p_values" if n_valid_p == 0 else "multiple_testing_not_adjusted"
        warnings.append(_multiplicity_qc_message(code))
    return AssociationMultiplicityResultRow(
        workflow_id=str(info["workflow_id"]),
        family_id=family_id,
        multiple_testing_method=info["multiple_testing_method"],
        correction_method=info["correction_method"],
        result_row_id=info["result_row_id"],
        input_row_index=input_row_index,
        method_id=info["method_id"],
        method_kind=info["method_kind"],
        method_name=info["method_name"],
        source_id=info["source_id"],
        outcome_id=info["outcome_id"],
        predictor_id=info["predictor_id"],
        covariate_ids=info["covariate_ids"],
        statistic_name=info["statistic_name"],
        statistic_value=info["statistic_value"],
        p_value=info["p_value"] if info["p_value_status"] == "valid" else None,
        q_value=q_value,
        n_family_total=n_family_total,
        n_valid_p=n_valid_p,
        n_missing_p=n_missing_p,
        n_invalid_p=n_invalid_p,
        n_adjusted=n_adjusted,
        status=status,
        code=code,
        warnings=_unique_texts(warnings),
        errors=_unique_texts(errors),
    )


def _multiplicity_run_qc_rows(
    *,
    workflow: TabularAssociationWorkflowSpec,
    row_infos: Sequence[Mapping[str, Any]],
    family_plan_rows: Sequence[AssociationMultiplicityFamilyPlanRow],
    family_counts: Mapping[str, Mapping[str, int]],
    adjusted_by_family: Mapping[str, int],
    p_value_policy: str,
) -> tuple[AssociationMultiplicityQcRow, ...]:
    rows: list[AssociationMultiplicityQcRow] = []
    for info in row_infos:
        row_code: str | None = None
        message = ""
        status = "warning"
        if info["family_code"] is not None:
            row_code = str(info["family_code"])
            message = str(info["family_message"])
        elif info["method_deferred"]:
            row_code = "multiple_testing_method_deferred"
            message = str(info["method_deferred_message"])
        elif info["p_value_status"] in {"missing", "invalid"}:
            row_code = "missing_p_value" if info["p_value_status"] == "missing" else "invalid_p_value"
            message = str(info["p_value_message"])
            if p_value_policy == "error":
                status = "error"
        if row_code is None:
            continue
        family_id = info["family_id"]
        counts = family_counts.get(str(family_id), {}) if family_id is not None else {}
        rows.append(
            AssociationMultiplicityQcRow(
                workflow_id=str(info["workflow_id"]),
                family_id=family_id,
                multiple_testing_method=info["multiple_testing_method"],
                correction_method=info["correction_method"],
                result_row_id=info["result_row_id"],
                input_row_index=int(info["input_row_index"]),
                status=status,
                code=row_code,
                message=message,
                n_family_total=int(counts.get("n_family_total", 0)),
                n_valid_p=int(counts.get("n_valid_p", 0)),
                n_missing_p=int(counts.get("n_missing_p", 0)),
                n_invalid_p=int(counts.get("n_invalid_p", 0)),
                n_adjusted=int(adjusted_by_family.get(str(family_id), 0)) if family_id is not None else 0,
                warnings=(message,) if status == "warning" else (),
                errors=(message,) if status == "error" else (),
            )
        )

    for family_plan_row in family_plan_rows:
        counts = family_counts.get(family_plan_row.family_id, {})
        n_family_total = int(counts.get("n_family_total", 0))
        n_valid_p = int(counts.get("n_valid_p", 0))
        n_missing_p = int(counts.get("n_missing_p", 0))
        n_invalid_p = int(counts.get("n_invalid_p", 0))
        n_adjusted = int(adjusted_by_family.get(family_plan_row.family_id, 0))
        if family_plan_row.executable and n_adjusted:
            rows.append(
                AssociationMultiplicityQcRow(
                    workflow_id=workflow.workflow_id,
                    family_id=family_plan_row.family_id,
                    multiple_testing_method=family_plan_row.multiple_testing_method,
                    correction_method=family_plan_row.correction_method,
                    result_row_id=None,
                    input_row_index=None,
                    status="ok",
                    code="benjamini_hochberg_adjusted",
                    message="Benjamini-Hochberg q-values were computed for valid p-values in this family.",
                    n_family_total=n_family_total,
                    n_valid_p=n_valid_p,
                    n_missing_p=n_missing_p,
                    n_invalid_p=n_invalid_p,
                    n_adjusted=n_adjusted,
                )
            )
        elif family_plan_row.executable and n_family_total and n_valid_p == 0:
            message = "no_valid_p_values: no valid p-values were available for this family."
            rows.append(
                AssociationMultiplicityQcRow(
                    workflow_id=workflow.workflow_id,
                    family_id=family_plan_row.family_id,
                    multiple_testing_method=family_plan_row.multiple_testing_method,
                    correction_method=family_plan_row.correction_method,
                    result_row_id=None,
                    input_row_index=None,
                    status="warning",
                    code="no_valid_p_values",
                    message=message,
                    n_family_total=n_family_total,
                    n_valid_p=n_valid_p,
                    n_missing_p=n_missing_p,
                    n_invalid_p=n_invalid_p,
                    n_adjusted=0,
                    warnings=(message,),
                )
            )
    return tuple(rows)


def _multiplicity_method_summary_rows(
    workflow: TabularAssociationWorkflowSpec,
    *,
    family_plan_rows: Sequence[AssociationMultiplicityFamilyPlanRow],
    family_counts: Mapping[str, Mapping[str, int]],
    adjusted_by_family: Mapping[str, int],
    p_value_policy: str,
    executed: bool,
    plan_only: bool,
) -> tuple[AssociationMultiplicityMethodSummaryRow, ...]:
    del workflow
    rows: list[AssociationMultiplicityMethodSummaryRow] = []
    for plan_row in family_plan_rows:
        counts = family_counts.get(plan_row.family_id, {})
        n_family_total = int(counts.get("n_family_total", 0))
        n_valid_p = int(counts.get("n_valid_p", 0))
        n_missing_p = int(counts.get("n_missing_p", 0))
        n_invalid_p = int(counts.get("n_invalid_p", 0))
        n_adjusted = int(adjusted_by_family.get(plan_row.family_id, 0))
        warnings = list(plan_row.warnings)
        errors = list(plan_row.errors)
        code = plan_row.code
        status = "ok" if plan_row.executable else plan_row.status
        if executed and plan_row.executable:
            if n_adjusted:
                code = "benjamini_hochberg_adjusted"
                status = "ok"
            elif n_family_total and n_valid_p == 0:
                code = "no_valid_p_values"
                status = "warning"
                warnings.append("no_valid_p_values: no valid p-values were available for this family.")
            if n_missing_p:
                message = f"missing_p_value: observed {n_missing_p} rows without valid p-values in this family."
                if p_value_policy == "error":
                    status = "error"
                    errors.append(message)
                else:
                    status = "warning" if status == "ok" else status
                    warnings.append(message)
            if n_invalid_p:
                message = f"invalid_p_value: observed {n_invalid_p} rows with invalid p-values in this family."
                if p_value_policy == "error":
                    status = "error"
                    errors.append(message)
                else:
                    status = "warning" if status == "ok" else status
                    warnings.append(message)
        rows.append(
            AssociationMultiplicityMethodSummaryRow(
                workflow_id=plan_row.workflow_id,
                family_id=plan_row.family_id,
                multiple_testing_method=plan_row.multiple_testing_method,
                correction_method=plan_row.correction_method,
                executable=plan_row.executable,
                deferred=plan_row.deferred,
                n_family_total=n_family_total,
                n_valid_p=n_valid_p,
                n_missing_p=n_missing_p,
                n_invalid_p=n_invalid_p,
                n_adjusted=n_adjusted,
                status=status,
                code=code,
                warnings=_unique_texts(warnings),
                errors=_unique_texts(errors),
                executed=executed,
                plan_only=plan_only,
            )
        )
    return tuple(rows)


def _multiplicity_qc_message(code: str) -> str:
    if code == "missing_p_value":
        return "missing_p_value: no valid input p-value was supplied."
    if code == "invalid_p_value":
        return "invalid_p_value: supplied p-value is not a finite numeric value in [0, 1]."
    if code == "missing_family_id":
        return "missing_family_id: association result row has no resolvable family id."
    if code == "undeclared_family_id":
        return "undeclared_family_id: resolved family id is not declared by families or multiple_testing."
    if code == "missing_multiple_testing_spec":
        return "missing_multiple_testing_spec: resolved association family has no MultipleTestingSpec declaration."
    if code == "multiple_testing_method_deferred":
        return "multiple_testing_method_deferred: declared multiple-testing method is not implemented in Step 11F."
    if code == "no_valid_p_values":
        return "no_valid_p_values: no valid p-values were available for this family."
    if code == "benjamini_hochberg_adjusted":
        return "benjamini_hochberg_adjusted: q-values were computed from supplied p-values."
    return f"{code}: multiplicity QC issue."


def _multiplicity_messages(
    *,
    workflow_validation_rows: Sequence[AssociationValidationRow],
    family_plan_rows: Sequence[AssociationMultiplicityFamilyPlanRow],
    qc_rows: Sequence[AssociationMultiplicityQcRow],
    result_rows: Sequence[AssociationMultiplicityResultRow],
    method_summary_rows: Sequence[AssociationMultiplicityMethodSummaryRow],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    errors: list[str] = []

    def add_status(status: str, message: str) -> None:
        if not message:
            return
        if status == "error":
            errors.append(message)
        elif status in {"warning", "deferred"}:
            warnings.append(message)

    for row in workflow_validation_rows:
        add_status(row.status, row.message)
    for row in family_plan_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in (*qc_rows, *result_rows, *method_summary_rows):
        warnings.extend(row.warnings)
        errors.extend(row.errors)
        if getattr(row, "status", "") in {"error", "warning", "deferred"}:
            add_status(str(getattr(row, "status")), str(getattr(row, "message", "")))
    return _unique_texts(warnings), _unique_texts(errors)


def _multiplicity_correction_method_count(
    family_plan_rows: Sequence[AssociationMultiplicityFamilyPlanRow],
) -> int:
    return len({row.correction_method for row in family_plan_rows if row.correction_method})


def _multiplicity_provenance_rows(
    workflow: TabularAssociationWorkflowSpec,
    *,
    executed: bool,
    plan_only: bool,
    input_row_count: int,
    family_count: int,
    adjusted_row_count: int,
    missing_p_value_count: int,
    invalid_p_value_count: int,
    correction_method_count: int,
    qc_mode: str,
    p_value_field: str,
    p_value_policy: str,
) -> tuple[TabularAssociationMultiplicityProvenanceRow, ...]:
    return (
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="schema_version", value=SCHEMA_VERSION),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="workflow_id", value=workflow.workflow_id),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="requested_backend", value=workflow.backend),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="runtime_backend", value=RUNTIME_BACKEND_RECORDS),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="input_row_count", value=input_row_count),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="family_count", value=family_count),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="adjusted_row_count", value=adjusted_row_count),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="missing_p_value_count", value=missing_p_value_count),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="invalid_p_value_count", value=invalid_p_value_count),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="correction_method_count", value=correction_method_count),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="qc_mode", value=qc_mode),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="p_value_field", value=p_value_field),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="p_value_policy", value=p_value_policy),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="executed", value=executed),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="plan_only", value=plan_only),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="will_write", value=False),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="output_written", value=False),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="output_paths_written", value=()),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow.workflow_id, key="no_output_paths_written", value=True),
    )


def _multiplicity_provenance_rows_for_error(
    workflow_id: str,
    *,
    executed: bool,
    plan_only: bool,
    input_row_count: int,
    qc_mode: str,
) -> tuple[TabularAssociationMultiplicityProvenanceRow, ...]:
    return (
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="schema_version", value=SCHEMA_VERSION),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="workflow_id", value=workflow_id),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="requested_backend", value=BACKEND_RECORDS),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="runtime_backend", value=RUNTIME_BACKEND_RECORDS),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="input_row_count", value=input_row_count),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="family_count", value=0),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="adjusted_row_count", value=0),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="missing_p_value_count", value=0),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="invalid_p_value_count", value=0),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="correction_method_count", value=0),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="qc_mode", value=qc_mode),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="executed", value=executed),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="plan_only", value=plan_only),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="will_write", value=False),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="output_written", value=False),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="output_paths_written", value=()),
        TabularAssociationMultiplicityProvenanceRow(workflow_id=workflow_id, key="no_output_paths_written", value=True),
    )


def _repeated_measures_methods(workflow: TabularAssociationWorkflowSpec) -> tuple[AssociationMethodSpec, ...]:
    return tuple(method for method in workflow.methods if method.method_name in DEFERRED_ASSOCIATION_METHODS)


def _repeated_measures_method_count(workflow: TabularAssociationWorkflowSpec) -> int:
    return len(_repeated_measures_methods(workflow))


def _repeated_measures_model_plan_rows(workflow: TabularAssociationWorkflowSpec) -> tuple[RepeatedMeasuresModelPlanRow, ...]:
    outcome_by_id = {outcome.variable_id: outcome for outcome in workflow.outcomes}
    predictor_by_id = {predictor.variable_id: predictor for predictor in workflow.predictors}
    covariate_by_id = {covariate.variable_id: covariate for covariate in workflow.covariates}
    grouping_by_id = {grouping.variable_id: grouping for grouping in workflow.groupings}
    source_by_id = {source.source_id: source for source in workflow.sources}
    rows: list[RepeatedMeasuresModelPlanRow] = []

    for method in _repeated_measures_methods(workflow):
        outcomes = _resolved_method_variables(method.outcome_ids, tuple(workflow.outcomes), outcome_by_id)
        predictors = _resolved_method_variables(method.predictor_ids, tuple(workflow.predictors), predictor_by_id)
        covariates = _resolved_method_variables(method.covariate_ids, tuple(workflow.covariates), covariate_by_id) if method.covariate_ids else ()
        groupings = _resolved_method_variables(method.grouping_ids, tuple(workflow.groupings), grouping_by_id) if method.grouping_ids else ()
        for outcome in outcomes:
            for predictor in predictors:
                source_id, source_warnings = _repeated_measures_model_source_id(
                    workflow=workflow,
                    outcome=outcome,
                    predictor=predictor,
                    covariates=covariates,
                    groupings=groupings,
                )
                source = source_by_id.get(source_id or "")
                repeated_columns = _repeated_measures_columns_for_source(workflow, source)
                repeated_metadata = workflow.repeated_measures.metadata if workflow.repeated_measures is not None else {}
                model_design, model_design_errors = _safe_model_design_metadata_for_method(
                    method=method,
                    repeated_metadata=repeated_metadata,
                )
                repeated_factor_columns = _repeated_measures_factor_columns(
                    method=method,
                    repeated_metadata=repeated_metadata,
                    timepoint_column=repeated_columns["timepoint_column"],
                )
                repeated_factor_columns = _unique_texts(
                    (
                        *repeated_factor_columns,
                        *(factor.column_name for factor in model_design.repeated_factors if factor.column_name),
                        *(factor.column_name for factor in model_design.within_subject_factors if factor.column_name),
                    )
                )
                cluster_columns = _metadata_text_values(
                    method.metadata,
                    repeated_metadata,
                    keys=("cluster_columns", "cluster_column", "cluster", "clusters", "cluster_ids"),
                )
                cluster_columns = _unique_texts(
                    (*cluster_columns, *(cluster.column_name for cluster in model_design.cluster_terms if cluster.column_name))
                )
                fixed_effect_metadata = _metadata_subset(
                    method.metadata,
                    keys=("fixed_effects", "fixed_effect_terms", "fixed_effect_metadata", "fixed_effect_term_ids"),
                )
                if model_design.fixed_effect_terms:
                    fixed_effect_metadata = _model_design_metadata_mapping(
                        "fixed_effect_terms",
                        tuple(term.to_dict() for term in model_design.fixed_effect_terms),
                    )
                random_effect_metadata = _metadata_subset(
                    method.metadata,
                    keys=("random_effects", "random_effect", "random_effect_metadata", "random_intercepts", "random_slopes"),
                )
                if model_design.random_effect_terms or model_design.random_intercepts or model_design.random_slopes:
                    random_effect_metadata = {
                        "metadata_version": TABULAR_ASSOCIATION_REPEATED_MEASURES_METADATA_VERSION,
                        "random_effect_terms": tuple(term.to_dict() for term in model_design.random_effect_terms),
                        "random_intercepts": tuple(intercept.to_dict() for intercept in model_design.random_intercepts),
                        "random_slopes": tuple(slope.to_dict() for slope in model_design.random_slopes),
                        "metadata_only": True,
                        "model_fitting_deferred": True,
                    }
                formula_metadata = _metadata_subset(
                    method.metadata,
                    keys=("formula", "model_formula", "formula_like", "model_spec", "model_expression"),
                )
                formula_like = model_design.formula_like
                if model_design.formula_metadata is not None:
                    formula_metadata = model_design.formula_metadata.to_dict()
                    formula_like = formula_like or model_design.formula_metadata.formula_like
                elif model_design.formula_like:
                    formula_metadata = {"formula_like": model_design.formula_like}
                fixed_effect_term_ids = _metadata_text_values(
                    method.metadata,
                    keys=("fixed_effect_term_ids", "fixed_effect_terms", "fixed_effects"),
                )
                if model_design.fixed_effect_terms:
                    fixed_effect_term_ids = _model_design_metadata_id_tuple(
                        tuple(term.term_id for term in model_design.fixed_effect_terms)
                    )
                random_effect_term_ids = _model_design_metadata_id_tuple(
                    tuple(term.term_id for term in model_design.random_effect_terms)
                )
                random_intercept_ids = _model_design_metadata_id_tuple(
                    tuple(intercept.intercept_id for intercept in model_design.random_intercepts)
                )
                random_slope_ids = _model_design_metadata_id_tuple(tuple(slope.slope_id for slope in model_design.random_slopes))
                repeated_factor_ids = _model_design_metadata_id_tuple(
                    tuple(factor.factor_id for factor in model_design.repeated_factors)
                )
                within_subject_factor_ids = _model_design_metadata_id_tuple(
                    tuple(factor.factor_id for factor in model_design.within_subject_factors)
                )
                within_subject_factor_columns = _model_design_metadata_id_tuple(
                    tuple(factor.column_name for factor in model_design.within_subject_factors if factor.column_name)
                )
                between_subject_factor_ids = _model_design_metadata_id_tuple(
                    tuple(factor.factor_id for factor in model_design.between_subject_factors)
                )
                between_subject_factor_columns = _model_design_metadata_id_tuple(
                    tuple(factor.column_name for factor in model_design.between_subject_factors if factor.column_name)
                )
                grouping_factor_ids = _model_design_metadata_id_tuple(
                    tuple(grouping.grouping_id for grouping in model_design.grouping_factors)
                )
                grouping_factor_columns = _model_design_metadata_id_tuple(
                    tuple(grouping.column_name for grouping in model_design.grouping_factors if grouping.column_name)
                )
                cluster_term_ids = _model_design_metadata_id_tuple(
                    tuple(cluster.cluster_id for cluster in model_design.cluster_terms)
                )
                timepoint_role_ids = _model_design_metadata_id_tuple(tuple(role.role_id for role in model_design.timepoint_roles))
                timepoint_columns = _model_design_metadata_id_tuple(
                    tuple(role.column_name for role in model_design.timepoint_roles if role.column_name)
                )
                categorical_coding_ids = _model_design_metadata_id_tuple(
                    tuple(coding.coding_id for coding in model_design.categorical_coding)
                )
                planned_comparison_ids = _model_design_metadata_id_tuple(
                    tuple(comparison.comparison_id for comparison in model_design.planned_comparisons)
                )
                planned_comparison_metadata = (
                    _model_design_metadata_mapping(
                        "planned_comparisons",
                        tuple(comparison.to_dict() for comparison in model_design.planned_comparisons),
                    )
                    if model_design.planned_comparisons
                    else {}
                )
                contrast_metadata_ids = _model_design_metadata_id_tuple(
                    tuple(contrast.contrast_id for contrast in model_design.contrast_metadata)
                )
                contrast_metadata = (
                    _model_design_metadata_mapping(
                        "contrast_metadata",
                        tuple(contrast.to_dict() for contrast in model_design.contrast_metadata),
                    )
                    if model_design.contrast_metadata
                    else {}
                )
                metadata_validation_issues = _model_design_metadata_validation_issues(
                    workflow=workflow,
                    design=model_design,
                    default_source_id=source_id,
                )
                errors: list[str] = []
                warnings = list(source_warnings)
                errors.extend(model_design_errors)
                errors.extend(issue["message"] for issue in metadata_validation_issues if issue["status"] == "error")
                if workflow.repeated_measures is None:
                    errors.append("missing_repeated_measures_spec: repeated-measures methods require a declaration.")
                elif source_id and workflow.repeated_measures.source_id != source_id:
                    warnings.append(
                        "repeated_measures_source_mismatch: model variables and repeated-measures declaration use different sources."
                    )
                if not repeated_columns["repeated_unit_columns"]:
                    errors.append("missing_repeated_unit_declaration: no repeated-unit or timepoint columns are declared.")
                rows.append(
                    RepeatedMeasuresModelPlanRow(
                        workflow_id=workflow.workflow_id,
                        model_plan_id=_repeated_measures_model_plan_id(method, outcome, predictor),
                        source_id=source_id,
                        method_id=method.method_id,
                        method_name=method.method_name,
                        method_kind=method.method_name,
                        family_id=method.family_id,
                        outcome_id=outcome.variable_id,
                        outcome_source_id=outcome.source_id,
                        outcome_column=outcome.column_name,
                        predictor_id=predictor.variable_id,
                        predictor_source_id=predictor.source_id,
                        predictor_column=predictor.column_name,
                        covariate_ids=tuple(covariate.variable_id for covariate in covariates),
                        covariate_source_ids=tuple(covariate.source_id for covariate in covariates),
                        covariate_columns=tuple(covariate.column_name for covariate in covariates),
                        group_id=groupings[0].variable_id if groupings else None,
                        group_column=groupings[0].column_name if groupings else None,
                        group_ids=tuple(grouping.variable_id for grouping in groupings),
                        group_columns=tuple(grouping.column_name for grouping in groupings),
                        subject_id_column=repeated_columns["subject_id_column"],
                        session_column=repeated_columns["session_column"],
                        timepoint_column=repeated_columns["timepoint_column"],
                        repeated_unit_columns=repeated_columns["repeated_unit_columns"],
                        repeated_factor_columns=repeated_factor_columns,
                        cluster_columns=cluster_columns,
                        model_design_id=model_design.model_design_id,
                        fixed_effect_term_ids=fixed_effect_term_ids,
                        fixed_effect_metadata=fixed_effect_metadata,
                        random_effect_term_ids=random_effect_term_ids,
                        random_intercept_ids=random_intercept_ids,
                        random_slope_ids=random_slope_ids,
                        random_effect_metadata=random_effect_metadata,
                        repeated_factor_ids=repeated_factor_ids,
                        within_subject_factor_ids=within_subject_factor_ids,
                        within_subject_factor_columns=within_subject_factor_columns,
                        between_subject_factor_ids=between_subject_factor_ids,
                        between_subject_factor_columns=between_subject_factor_columns,
                        grouping_factor_ids=grouping_factor_ids,
                        grouping_factor_columns=grouping_factor_columns,
                        cluster_term_ids=cluster_term_ids,
                        timepoint_role_ids=timepoint_role_ids,
                        timepoint_columns=timepoint_columns,
                        categorical_coding_ids=categorical_coding_ids,
                        formula_metadata=formula_metadata,
                        formula_like=formula_like,
                        planned_comparison_ids=planned_comparison_ids,
                        planned_comparison_metadata=planned_comparison_metadata,
                        contrast_metadata_ids=contrast_metadata_ids,
                        contrast_metadata=contrast_metadata,
                        model_family=model_design.model_family,
                        link_function=model_design.link_function,
                        method_metadata=method.metadata,
                        repeated_measures_metadata=repeated_metadata,
                        metadata_only=True,
                        model_fitting_deferred=True,
                        runtime_backend=RUNTIME_BACKEND_RECORDS,
                        executable=False,
                        deferred=True,
                        status="error" if errors else "deferred",
                        code="model_fitting_deferred",
                        warnings=_unique_texts(warnings),
                        errors=_unique_texts(errors),
                    )
                )
    return tuple(rows)


def _repeated_measures_model_source_id(
    *,
    workflow: TabularAssociationWorkflowSpec,
    outcome: AssociationVariableSpec,
    predictor: AssociationVariableSpec,
    covariates: Sequence[AssociationVariableSpec],
    groupings: Sequence[AssociationVariableSpec],
) -> tuple[str | None, tuple[str, ...]]:
    source_ids = {outcome.source_id, predictor.source_id, *(covariate.source_id for covariate in covariates)}
    source_ids.update(grouping.source_id for grouping in groupings)
    warnings: list[str] = []
    if len(source_ids) == 1:
        return next(iter(source_ids)), ()
    warnings.append(
        "cross_source_repeated_measures_design_deferred: model variables come from different sources; joins are not planned here."
    )
    if workflow.repeated_measures is not None:
        return workflow.repeated_measures.source_id, tuple(warnings)
    return None, tuple(warnings)


def _repeated_measures_columns_for_source(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec | None,
) -> dict[str, Any]:
    repeated = workflow.repeated_measures
    if repeated is not None and (source is None or repeated.source_id == source.source_id):
        subject_id_column = repeated.subject_id_column
        session_column = repeated.session_column
        timepoint_column = repeated.timepoint_column
        repeated_unit_columns = repeated.unit_columns or tuple(
            column for column in (subject_id_column, session_column, timepoint_column) if column
        )
        return {
            "subject_id_column": subject_id_column,
            "session_column": session_column,
            "timepoint_column": timepoint_column,
            "repeated_unit_columns": repeated_unit_columns,
        }
    if source is None:
        return {
            "subject_id_column": None,
            "session_column": None,
            "timepoint_column": None,
            "repeated_unit_columns": (),
        }
    repeated_unit_columns = tuple(
        column for column in (source.schema.subject_id_column, source.schema.session_column, source.schema.timepoint_column) if column
    )
    return {
        "subject_id_column": source.schema.subject_id_column,
        "session_column": source.schema.session_column,
        "timepoint_column": source.schema.timepoint_column,
        "repeated_unit_columns": repeated_unit_columns,
    }


def _repeated_measures_factor_columns(
    *,
    method: AssociationMethodSpec,
    repeated_metadata: Mapping[str, Any],
    timepoint_column: str | None,
) -> tuple[str, ...]:
    factor_columns: list[str] = []
    if timepoint_column:
        factor_columns.append(timepoint_column)
    factor_columns.extend(
        _metadata_text_values(
            method.metadata,
            repeated_metadata,
            keys=(
                "repeated_factor_columns",
                "repeated_factor_column",
                "factor_columns",
                "factor_column",
                "within_subject_factor_columns",
                "within_subject_factors",
            ),
        )
    )
    return _unique_texts(factor_columns)


def _metadata_text_values(
    *metadata_maps: Mapping[str, Any],
    keys: Sequence[str],
) -> tuple[str, ...]:
    values: list[str] = []
    for metadata in metadata_maps:
        for key in keys:
            if key not in metadata:
                continue
            values.extend(_metadata_value_texts(metadata[key]))
    return _unique_texts(values)


def _metadata_value_texts(value: Any) -> tuple[str, ...]:
    if value is None or isinstance(value, bool):
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Mapping):
        nested_values: list[str] = []
        for key in ("column", "column_name", "columns", "column_names", "id", "ids", "term", "terms"):
            if key in value:
                nested_values.extend(_metadata_value_texts(value[key]))
        return tuple(nested_values)
    if isinstance(value, (bytes, bytearray)):
        return ()
    try:
        iterator = iter(value)
    except TypeError:
        text = str(value).strip()
        return (text,) if text else ()
    texts: list[str] = []
    for item in iterator:
        texts.extend(_metadata_value_texts(item))
    return tuple(texts)


def _metadata_subset(metadata: Mapping[str, Any], *, keys: Sequence[str]) -> dict[str, Any]:
    return {key: _json_safe(metadata[key]) for key in keys if key in metadata}


def _metadata_extra(mapping: Mapping[str, Any], *, known_keys: Sequence[str]) -> Mapping[str, Any]:
    metadata = dict(_as_metadata(mapping.get("metadata", {})))
    known = set(known_keys) | {"metadata", "id"}
    for key, value in mapping.items():
        if key not in known:
            metadata.setdefault(str(key), _json_safe(value))
    return metadata


def _metadata_declaration_items(value: Any, *, id_field: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (_non_empty_text(value, field_name=id_field),)
    if isinstance(value, Mapping):
        if any(key in value for key in (id_field, "id", "name", "column_name", "column")):
            return (value,)
        items: list[Any] = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                item_mapping = dict(item)
                item_mapping.setdefault(id_field, key)
                items.append(item_mapping)
            elif isinstance(item, (str, bytes)):
                items.append({id_field: key, "label": _non_empty_text(item, field_name=id_field)})
            else:
                items.append({id_field: key, "metadata": {"value": _json_safe(item)}})
        return tuple(items)
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{id_field} declarations must be a mapping, string, or sequence.") from exc


def _coerce_fixed_effect_term_spec(value: FixedEffectTermSpec | Mapping[str, Any] | str) -> FixedEffectTermSpec:
    if isinstance(value, FixedEffectTermSpec):
        return value
    if isinstance(value, str):
        return FixedEffectTermSpec(term_id=value)
    mapping = _as_mapping(value, field_name="fixed_effect_term")
    known = (
        "term_id",
        "id",
        "name",
        "variable_id",
        "variable_ids",
        "variables",
        "column_name",
        "column_names",
        "column",
        "columns",
        "factor_id",
        "factor_ids",
        "factors",
        "coding_id",
        "coding_ids",
        "codings",
        "source_id",
        "source",
        "label",
    )
    return FixedEffectTermSpec(
        term_id=_first_present(mapping, "term_id", "id", "name", "variable_id", default=""),
        variable_ids=_sequence_from_mapping(mapping, "variable_ids", "variables", "variable_id"),
        column_names=_sequence_from_mapping(mapping, "column_names", "columns", "column_name", "column"),
        factor_ids=_sequence_from_mapping(mapping, "factor_ids", "factors", "factor_id"),
        coding_ids=_sequence_from_mapping(mapping, "coding_ids", "codings", "coding_id"),
        source_id=_first_present(mapping, "source_id", "source"),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_random_intercept_spec(value: RandomInterceptSpec | Mapping[str, Any] | str) -> RandomInterceptSpec:
    if isinstance(value, RandomInterceptSpec):
        return value
    if isinstance(value, str):
        return RandomInterceptSpec(intercept_id=value)
    mapping = _as_mapping(value, field_name="random_intercept")
    known = (
        "intercept_id",
        "id",
        "name",
        "grouping_id",
        "grouping_ids",
        "groupings",
        "grouping_column",
        "grouping_columns",
        "cluster_id",
        "cluster_ids",
        "clusters",
        "source_id",
        "source",
        "label",
    )
    return RandomInterceptSpec(
        intercept_id=_first_present(mapping, "intercept_id", "id", "name", default=""),
        grouping_ids=_sequence_from_mapping(mapping, "grouping_ids", "groupings", "grouping_id"),
        grouping_columns=_sequence_from_mapping(mapping, "grouping_columns", "grouping_column"),
        cluster_ids=_sequence_from_mapping(mapping, "cluster_ids", "clusters", "cluster_id"),
        source_id=_first_present(mapping, "source_id", "source"),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_random_slope_spec(value: RandomSlopeSpec | Mapping[str, Any] | str) -> RandomSlopeSpec:
    if isinstance(value, RandomSlopeSpec):
        return value
    if isinstance(value, str):
        return RandomSlopeSpec(slope_id=value)
    mapping = _as_mapping(value, field_name="random_slope")
    known = (
        "slope_id",
        "id",
        "name",
        "variable_id",
        "variable_ids",
        "variables",
        "column_name",
        "column_names",
        "column",
        "columns",
        "factor_id",
        "factor_ids",
        "factors",
        "grouping_id",
        "grouping_ids",
        "groupings",
        "grouping_column",
        "grouping_columns",
        "cluster_id",
        "cluster_ids",
        "clusters",
        "source_id",
        "source",
        "label",
    )
    return RandomSlopeSpec(
        slope_id=_first_present(mapping, "slope_id", "id", "name", default=""),
        variable_ids=_sequence_from_mapping(mapping, "variable_ids", "variables", "variable_id"),
        column_names=_sequence_from_mapping(mapping, "column_names", "columns", "column_name", "column"),
        factor_ids=_sequence_from_mapping(mapping, "factor_ids", "factors", "factor_id"),
        grouping_ids=_sequence_from_mapping(mapping, "grouping_ids", "groupings", "grouping_id"),
        grouping_columns=_sequence_from_mapping(mapping, "grouping_columns", "grouping_column"),
        cluster_ids=_sequence_from_mapping(mapping, "cluster_ids", "clusters", "cluster_id"),
        source_id=_first_present(mapping, "source_id", "source"),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_random_effect_term_spec(value: RandomEffectTermSpec | Mapping[str, Any] | str) -> RandomEffectTermSpec:
    if isinstance(value, RandomEffectTermSpec):
        return value
    if isinstance(value, str):
        return RandomEffectTermSpec(term_id=value)
    mapping = _as_mapping(value, field_name="random_effect_term")
    known = (
        "term_id",
        "id",
        "name",
        "random_intercept_id",
        "random_intercept_ids",
        "intercepts",
        "random_slope_id",
        "random_slope_ids",
        "slopes",
        "variable_id",
        "variable_ids",
        "variables",
        "column_name",
        "column_names",
        "column",
        "columns",
        "factor_id",
        "factor_ids",
        "factors",
        "grouping_id",
        "grouping_ids",
        "groupings",
        "cluster_id",
        "cluster_ids",
        "clusters",
        "source_id",
        "source",
        "label",
    )
    return RandomEffectTermSpec(
        term_id=_first_present(mapping, "term_id", "id", "name", default=""),
        random_intercept_ids=_sequence_from_mapping(mapping, "random_intercept_ids", "intercepts", "random_intercept_id"),
        random_slope_ids=_sequence_from_mapping(mapping, "random_slope_ids", "slopes", "random_slope_id"),
        variable_ids=_sequence_from_mapping(mapping, "variable_ids", "variables", "variable_id"),
        column_names=_sequence_from_mapping(mapping, "column_names", "columns", "column_name", "column"),
        factor_ids=_sequence_from_mapping(mapping, "factor_ids", "factors", "factor_id"),
        grouping_ids=_sequence_from_mapping(mapping, "grouping_ids", "groupings", "grouping_id"),
        cluster_ids=_sequence_from_mapping(mapping, "cluster_ids", "clusters", "cluster_id"),
        source_id=_first_present(mapping, "source_id", "source"),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_repeated_factor_spec(value: RepeatedFactorSpec | Mapping[str, Any] | str) -> RepeatedFactorSpec:
    if isinstance(value, RepeatedFactorSpec):
        return value
    if isinstance(value, str):
        return RepeatedFactorSpec(factor_id=value)
    mapping = _as_mapping(value, field_name="repeated_factor")
    known = ("factor_id", "id", "name", "column_name", "column", "source_id", "source", "levels", "label")
    return RepeatedFactorSpec(
        factor_id=_first_present(mapping, "factor_id", "id", "name", "column_name", "column", default=""),
        column_name=_first_present(mapping, "column_name", "column"),
        source_id=_first_present(mapping, "source_id", "source"),
        levels=_sequence_from_mapping(mapping, "levels"),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_within_subject_factor_spec(value: WithinSubjectFactorSpec | Mapping[str, Any] | str) -> WithinSubjectFactorSpec:
    if isinstance(value, WithinSubjectFactorSpec):
        return value
    if isinstance(value, str):
        return WithinSubjectFactorSpec(factor_id=value)
    mapping = _as_mapping(value, field_name="within_subject_factor")
    known = (
        "factor_id",
        "id",
        "name",
        "column_name",
        "column",
        "source_id",
        "source",
        "repeated_factor_id",
        "levels",
        "label",
    )
    return WithinSubjectFactorSpec(
        factor_id=_first_present(mapping, "factor_id", "id", "name", "column_name", "column", default=""),
        column_name=_first_present(mapping, "column_name", "column"),
        source_id=_first_present(mapping, "source_id", "source"),
        repeated_factor_id=_first_present(mapping, "repeated_factor_id"),
        levels=_sequence_from_mapping(mapping, "levels"),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_between_subject_factor_spec(value: BetweenSubjectFactorSpec | Mapping[str, Any] | str) -> BetweenSubjectFactorSpec:
    if isinstance(value, BetweenSubjectFactorSpec):
        return value
    if isinstance(value, str):
        return BetweenSubjectFactorSpec(factor_id=value)
    mapping = _as_mapping(value, field_name="between_subject_factor")
    known = (
        "factor_id",
        "id",
        "name",
        "column_name",
        "column",
        "source_id",
        "source",
        "variable_id",
        "levels",
        "label",
    )
    return BetweenSubjectFactorSpec(
        factor_id=_first_present(mapping, "factor_id", "id", "name", "column_name", "column", default=""),
        column_name=_first_present(mapping, "column_name", "column"),
        source_id=_first_present(mapping, "source_id", "source"),
        variable_id=_first_present(mapping, "variable_id"),
        levels=_sequence_from_mapping(mapping, "levels"),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_grouping_factor_spec(value: GroupingFactorSpec | Mapping[str, Any] | str) -> GroupingFactorSpec:
    if isinstance(value, GroupingFactorSpec):
        return value
    if isinstance(value, str):
        return GroupingFactorSpec(grouping_id=value)
    mapping = _as_mapping(value, field_name="grouping_factor")
    known = ("grouping_id", "id", "name", "variable_id", "column_name", "column", "source_id", "source", "label")
    return GroupingFactorSpec(
        grouping_id=_first_present(mapping, "grouping_id", "id", "name", "variable_id", "column_name", "column", default=""),
        variable_id=_first_present(mapping, "variable_id"),
        column_name=_first_present(mapping, "column_name", "column"),
        source_id=_first_present(mapping, "source_id", "source"),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_cluster_term_spec(value: ClusterTermSpec | Mapping[str, Any] | str) -> ClusterTermSpec:
    if isinstance(value, ClusterTermSpec):
        return value
    if isinstance(value, str):
        return ClusterTermSpec(cluster_id=value)
    mapping = _as_mapping(value, field_name="cluster_term")
    known = ("cluster_id", "id", "name", "column_name", "column", "source_id", "source", "grouping_id", "label")
    return ClusterTermSpec(
        cluster_id=_first_present(mapping, "cluster_id", "id", "name", "column_name", "column", default=""),
        column_name=_first_present(mapping, "column_name", "column"),
        source_id=_first_present(mapping, "source_id", "source"),
        grouping_id=_first_present(mapping, "grouping_id"),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_timepoint_role_spec(value: TimepointRoleSpec | Mapping[str, Any] | str) -> TimepointRoleSpec:
    if isinstance(value, TimepointRoleSpec):
        return value
    if isinstance(value, str):
        return TimepointRoleSpec(role_id=value)
    mapping = _as_mapping(value, field_name="timepoint_role")
    known = ("role_id", "id", "name", "column_name", "column", "source_id", "source", "factor_id", "role", "label")
    return TimepointRoleSpec(
        role_id=_first_present(mapping, "role_id", "id", "name", "column_name", "column", default=""),
        column_name=_first_present(mapping, "column_name", "column"),
        source_id=_first_present(mapping, "source_id", "source"),
        factor_id=_first_present(mapping, "factor_id"),
        role=_first_present(mapping, "role"),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_categorical_coding_spec(value: CategoricalCodingSpec | Mapping[str, Any] | str) -> CategoricalCodingSpec:
    if isinstance(value, CategoricalCodingSpec):
        return value
    if isinstance(value, str):
        return CategoricalCodingSpec(coding_id=value)
    mapping = _as_mapping(value, field_name="categorical_coding")
    known = (
        "coding_id",
        "id",
        "name",
        "target_id",
        "target",
        "variable_id",
        "factor_id",
        "column_name",
        "column",
        "source_id",
        "source",
        "scheme",
        "reference_level",
        "levels",
        "label",
    )
    return CategoricalCodingSpec(
        coding_id=_first_present(mapping, "coding_id", "id", "name", default=""),
        target_id=_first_present(mapping, "target_id", "target"),
        variable_id=_first_present(mapping, "variable_id"),
        factor_id=_first_present(mapping, "factor_id"),
        column_name=_first_present(mapping, "column_name", "column"),
        source_id=_first_present(mapping, "source_id", "source"),
        scheme=_first_present(mapping, "scheme"),
        reference_level=_first_present(mapping, "reference_level"),
        levels=_sequence_from_mapping(mapping, "levels"),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_model_formula_metadata_spec(
    value: ModelFormulaMetadataSpec | Mapping[str, Any] | str | None,
) -> ModelFormulaMetadataSpec | None:
    if value is None:
        return None
    if isinstance(value, ModelFormulaMetadataSpec):
        return value
    if isinstance(value, str):
        return ModelFormulaMetadataSpec(formula_like=value)
    mapping = _as_mapping(value, field_name="formula_metadata")
    known = (
        "formula_id",
        "id",
        "name",
        "formula_like",
        "formula",
        "model_formula",
        "design_intent",
        "fixed_formula",
        "random_formula",
        "variable_id",
        "variable_ids",
        "variables",
        "factor_id",
        "factor_ids",
        "factors",
    )
    return ModelFormulaMetadataSpec(
        formula_id=_first_present(mapping, "formula_id", "id", "name"),
        formula_like=_first_present(mapping, "formula_like", "formula", "model_formula"),
        design_intent=_first_present(mapping, "design_intent"),
        fixed_formula=_first_present(mapping, "fixed_formula"),
        random_formula=_first_present(mapping, "random_formula"),
        variable_ids=_sequence_from_mapping(mapping, "variable_ids", "variables", "variable_id"),
        factor_ids=_sequence_from_mapping(mapping, "factor_ids", "factors", "factor_id"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_planned_comparison_spec(value: PlannedComparisonSpec | Mapping[str, Any] | str) -> PlannedComparisonSpec:
    if isinstance(value, PlannedComparisonSpec):
        return value
    if isinstance(value, str):
        return PlannedComparisonSpec(comparison_id=value)
    mapping = _as_mapping(value, field_name="planned_comparison")
    known = (
        "comparison_id",
        "id",
        "name",
        "factor_id",
        "factor_ids",
        "factors",
        "variable_id",
        "variable_ids",
        "variables",
        "grouping_id",
        "grouping_ids",
        "groupings",
        "cluster_id",
        "cluster_ids",
        "clusters",
        "coding_id",
        "coding_ids",
        "codings",
        "contrast_metadata_id",
        "contrast_metadata_ids",
        "contrasts",
        "label",
    )
    return PlannedComparisonSpec(
        comparison_id=_first_present(mapping, "comparison_id", "id", "name", default=""),
        factor_ids=_sequence_from_mapping(mapping, "factor_ids", "factors", "factor_id"),
        variable_ids=_sequence_from_mapping(mapping, "variable_ids", "variables", "variable_id"),
        grouping_ids=_sequence_from_mapping(mapping, "grouping_ids", "groupings", "grouping_id"),
        cluster_ids=_sequence_from_mapping(mapping, "cluster_ids", "clusters", "cluster_id"),
        coding_ids=_sequence_from_mapping(mapping, "coding_ids", "codings", "coding_id"),
        contrast_metadata_ids=_sequence_from_mapping(
            mapping,
            "contrast_metadata_ids",
            "contrasts",
            "contrast_metadata_id",
        ),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _coerce_contrast_metadata_spec(value: ContrastMetadataSpec | Mapping[str, Any] | str) -> ContrastMetadataSpec:
    if isinstance(value, ContrastMetadataSpec):
        return value
    if isinstance(value, str):
        return ContrastMetadataSpec(contrast_id=value)
    mapping = _as_mapping(value, field_name="contrast_metadata")
    known = (
        "contrast_id",
        "id",
        "name",
        "comparison_id",
        "comparison_ids",
        "comparisons",
        "factor_id",
        "factor_ids",
        "factors",
        "variable_id",
        "variable_ids",
        "variables",
        "coding_id",
        "coding_ids",
        "codings",
        "label",
    )
    return ContrastMetadataSpec(
        contrast_id=_first_present(mapping, "contrast_id", "id", "name", default=""),
        comparison_ids=_sequence_from_mapping(mapping, "comparison_ids", "comparisons", "comparison_id"),
        factor_ids=_sequence_from_mapping(mapping, "factor_ids", "factors", "factor_id"),
        variable_ids=_sequence_from_mapping(mapping, "variable_ids", "variables", "variable_id"),
        coding_ids=_sequence_from_mapping(mapping, "coding_ids", "codings", "coding_id"),
        label=_first_present(mapping, "label"),
        metadata=_metadata_extra(mapping, known_keys=known),
    )


def _safe_model_design_metadata_for_method(
    *,
    method: AssociationMethodSpec,
    repeated_metadata: Mapping[str, Any],
) -> tuple[ModelDesignMetadataSpec, tuple[str, ...]]:
    try:
        return _model_design_metadata_for_method(method=method, repeated_metadata=repeated_metadata), ()
    except (TypeError, ValueError) as exc:
        return ModelDesignMetadataSpec(), (f"invalid_model_design_metadata: {exc}",)


def _model_design_metadata_for_method(
    *,
    method: AssociationMethodSpec,
    repeated_metadata: Mapping[str, Any],
) -> ModelDesignMetadataSpec:
    sources = _model_design_metadata_sources(repeated_metadata, method.metadata)

    def sequence(key: str, *, id_field: str) -> tuple[Any, ...]:
        values: list[Any] = []
        for metadata in sources:
            if key in metadata:
                values.extend(_metadata_declaration_items(metadata[key], id_field=id_field))
        return tuple(values)

    formula_metadata_value = _last_present_metadata_value(sources, "formula_metadata")
    formula_like = _last_present_metadata_value(sources, "formula_like")
    if formula_metadata_value is None:
        formula_metadata_value = _last_present_metadata_value(sources, "formula", "model_formula")
    if formula_like is None and isinstance(formula_metadata_value, str):
        formula_like = formula_metadata_value

    model_design_id = _last_present_metadata_value(sources, "model_design_id", "design_id", "id")
    metadata = _model_design_extra_metadata(_model_design_nested_sources(repeated_metadata, method.metadata))
    return ModelDesignMetadataSpec(
        model_design_id=_optional_text(model_design_id),
        fixed_effect_terms=sequence("fixed_effect_terms", id_field="term_id"),
        random_effect_terms=sequence("random_effect_terms", id_field="term_id"),
        random_intercepts=sequence("random_intercepts", id_field="intercept_id"),
        random_slopes=sequence("random_slopes", id_field="slope_id"),
        repeated_factors=sequence("repeated_factors", id_field="factor_id"),
        within_subject_factors=sequence("within_subject_factors", id_field="factor_id"),
        between_subject_factors=sequence("between_subject_factors", id_field="factor_id"),
        grouping_factors=sequence("grouping_factors", id_field="grouping_id"),
        cluster_terms=sequence("cluster_terms", id_field="cluster_id"),
        timepoint_roles=sequence("timepoint_roles", id_field="role_id"),
        categorical_coding=sequence("categorical_coding", id_field="coding_id"),
        formula_metadata=formula_metadata_value,
        formula_like=_optional_text(formula_like),
        planned_comparisons=sequence("planned_comparisons", id_field="comparison_id"),
        contrast_metadata=sequence("contrast_metadata", id_field="contrast_id"),
        model_family=_optional_text(_last_present_metadata_value(sources, "model_family", "family")),
        link_function=_optional_text(_last_present_metadata_value(sources, "link_function", "link")),
        variable_ids=_metadata_text_values(*sources, keys=("variable_ids", "variables", "variable_id")),
        factor_ids=_metadata_text_values(*sources, keys=("factor_ids", "factors", "factor_id")),
        metadata=metadata,
    )


def _model_design_metadata_sources(*metadata_maps: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = []
    for metadata in metadata_maps:
        if not isinstance(metadata, Mapping):
            continue
        sources.append(metadata)
        model_design = metadata.get("model_design")
        if isinstance(model_design, Mapping):
            sources.append(model_design)
    return tuple(sources)


def _model_design_nested_sources(*metadata_maps: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = []
    for metadata in metadata_maps:
        if not isinstance(metadata, Mapping):
            continue
        model_design = metadata.get("model_design")
        if isinstance(model_design, Mapping):
            sources.append(model_design)
    return tuple(sources)


def _last_present_metadata_value(sources: Sequence[Mapping[str, Any]], *keys: str) -> Any:
    value: Any = None
    for metadata in sources:
        for key in keys:
            if key in metadata and metadata[key] is not None:
                value = metadata[key]
    return value


def _model_design_extra_metadata(sources: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    metadata: dict[str, Any] = {}
    known = set(_REPEATED_MEASURES_METADATA_KEYS) | {
        "model_design_id",
        "design_id",
        "id",
        "formula",
        "model_formula",
        "family",
        "link",
        "variable_ids",
        "variables",
        "variable_id",
        "factor_ids",
        "factors",
        "factor_id",
    }
    for source in sources:
        if isinstance(source.get("metadata"), Mapping):
            metadata.update(_json_safe_mapping(source["metadata"]))
        for key, value in source.items():
            if key not in known and key != "metadata":
                metadata.setdefault(str(key), _json_safe(value))
    return metadata


def _model_design_has_explicit_metadata(design: ModelDesignMetadataSpec) -> bool:
    return design.has_declarations()


def _model_design_metadata_validation_issues(
    *,
    workflow: TabularAssociationWorkflowSpec,
    design: ModelDesignMetadataSpec,
    default_source_id: str | None,
    loaded_sources_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    if not _model_design_has_explicit_metadata(design):
        return ()

    issues: list[dict[str, Any]] = []
    source_ids = {source.source_id for source in workflow.sources}
    declared_columns = {column for source in workflow.sources for column in source.schema.column_names()}
    variable_ids = {variable.variable_id for variable in (*workflow.outcomes, *workflow.predictors, *workflow.covariates, *workflow.groupings)}
    workflow_grouping_ids = {grouping.variable_id for grouping in workflow.groupings}

    fixed_ids = tuple(term.term_id for term in design.fixed_effect_terms)
    random_term_ids = tuple(term.term_id for term in design.random_effect_terms)
    random_intercept_ids = tuple(term.intercept_id for term in design.random_intercepts)
    random_slope_ids = tuple(term.slope_id for term in design.random_slopes)
    repeated_factor_ids = tuple(factor.factor_id for factor in design.repeated_factors)
    within_factor_ids = tuple(factor.factor_id for factor in design.within_subject_factors)
    between_factor_ids = tuple(factor.factor_id for factor in design.between_subject_factors)
    grouping_factor_ids = tuple(grouping.grouping_id for grouping in design.grouping_factors)
    cluster_ids = tuple(cluster.cluster_id for cluster in design.cluster_terms)
    timepoint_role_ids = tuple(role.role_id for role in design.timepoint_roles)
    coding_ids = tuple(coding.coding_id for coding in design.categorical_coding)
    comparison_ids = tuple(comparison.comparison_id for comparison in design.planned_comparisons)
    contrast_ids = tuple(contrast.contrast_id for contrast in design.contrast_metadata)

    for category, ids in (
        ("fixed_effect_terms", fixed_ids),
        ("random_effect_terms", random_term_ids),
        ("random_intercepts", random_intercept_ids),
        ("random_slopes", random_slope_ids),
        ("repeated_factors", repeated_factor_ids),
        ("within_subject_factors", within_factor_ids),
        ("between_subject_factors", between_factor_ids),
        ("grouping_factors", grouping_factor_ids),
        ("cluster_terms", cluster_ids),
        ("timepoint_roles", timepoint_role_ids),
        ("categorical_coding", coding_ids),
        ("planned_comparisons", comparison_ids),
        ("contrast_metadata", contrast_ids),
    ):
        for duplicate in _duplicates(list(ids)):
            issues.append(
                _model_design_metadata_issue(
                    code="duplicate_metadata_id",
                    message=f"duplicate_metadata_id: {category} contains duplicate id {duplicate!r}.",
                    metadata={"category": category, "metadata_id": duplicate},
                )
            )

    known_factor_ids = set(design.factor_ids) | set(repeated_factor_ids) | set(within_factor_ids) | set(between_factor_ids)
    known_grouping_ids = set(grouping_factor_ids) | workflow_grouping_ids
    known_cluster_ids = set(cluster_ids)
    known_coding_ids = set(coding_ids)
    known_comparison_ids = set(comparison_ids)
    known_contrast_ids = set(contrast_ids)
    known_timepoint_ids = set(timepoint_role_ids)

    def add_unknown(owner_category: str, owner_id: str, reference_kind: str, references: Sequence[str], known: set[str]) -> None:
        for reference in references:
            if reference not in known:
                issues.append(
                    _model_design_metadata_issue(
                        code="unknown_metadata_reference",
                        message=(
                            "unknown_metadata_reference: "
                            f"{owner_category} {owner_id!r} references unknown {reference_kind} {reference!r}."
                        ),
                        metadata={
                            "owner_category": owner_category,
                            "owner_id": owner_id,
                            "reference_kind": reference_kind,
                            "reference_id": reference,
                        },
                    )
                )

    for term in design.fixed_effect_terms:
        add_unknown("fixed_effect_terms", term.term_id, "variable_id", term.variable_ids, variable_ids)
        add_unknown("fixed_effect_terms", term.term_id, "factor_id", term.factor_ids, known_factor_ids)
        add_unknown("fixed_effect_terms", term.term_id, "coding_id", term.coding_ids, known_coding_ids)
    for term in design.random_effect_terms:
        add_unknown("random_effect_terms", term.term_id, "random_intercept_id", term.random_intercept_ids, set(random_intercept_ids))
        add_unknown("random_effect_terms", term.term_id, "random_slope_id", term.random_slope_ids, set(random_slope_ids))
        add_unknown("random_effect_terms", term.term_id, "variable_id", term.variable_ids, variable_ids)
        add_unknown("random_effect_terms", term.term_id, "factor_id", term.factor_ids, known_factor_ids)
        add_unknown("random_effect_terms", term.term_id, "grouping_id", term.grouping_ids, known_grouping_ids)
        add_unknown("random_effect_terms", term.term_id, "cluster_id", term.cluster_ids, known_cluster_ids)
    for intercept in design.random_intercepts:
        add_unknown("random_intercepts", intercept.intercept_id, "grouping_id", intercept.grouping_ids, known_grouping_ids)
        add_unknown("random_intercepts", intercept.intercept_id, "cluster_id", intercept.cluster_ids, known_cluster_ids)
    for slope in design.random_slopes:
        add_unknown("random_slopes", slope.slope_id, "variable_id", slope.variable_ids, variable_ids)
        add_unknown("random_slopes", slope.slope_id, "factor_id", slope.factor_ids, known_factor_ids)
        add_unknown("random_slopes", slope.slope_id, "grouping_id", slope.grouping_ids, known_grouping_ids)
        add_unknown("random_slopes", slope.slope_id, "cluster_id", slope.cluster_ids, known_cluster_ids)
    for factor in design.within_subject_factors:
        if factor.repeated_factor_id:
            add_unknown("within_subject_factors", factor.factor_id, "repeated_factor_id", (factor.repeated_factor_id,), set(repeated_factor_ids))
    for factor in design.between_subject_factors:
        if factor.variable_id:
            add_unknown("between_subject_factors", factor.factor_id, "variable_id", (factor.variable_id,), variable_ids)
    for grouping in design.grouping_factors:
        if grouping.variable_id:
            add_unknown("grouping_factors", grouping.grouping_id, "variable_id", (grouping.variable_id,), variable_ids)
    for cluster in design.cluster_terms:
        if cluster.grouping_id:
            add_unknown("cluster_terms", cluster.cluster_id, "grouping_id", (cluster.grouping_id,), known_grouping_ids)
    for role in design.timepoint_roles:
        if role.factor_id:
            add_unknown("timepoint_roles", role.role_id, "factor_id", (role.factor_id,), known_factor_ids)
    for coding in design.categorical_coding:
        if coding.variable_id:
            add_unknown("categorical_coding", coding.coding_id, "variable_id", (coding.variable_id,), variable_ids)
        if coding.factor_id:
            add_unknown("categorical_coding", coding.coding_id, "factor_id", (coding.factor_id,), known_factor_ids)
        if coding.target_id:
            known_targets = variable_ids | known_factor_ids | known_grouping_ids | known_cluster_ids | known_timepoint_ids | declared_columns
            add_unknown("categorical_coding", coding.coding_id, "coding_target", (coding.target_id,), known_targets)
    if design.formula_metadata is not None:
        formula_id = design.formula_metadata.formula_id or "formula_metadata"
        add_unknown("formula_metadata", formula_id, "variable_id", design.formula_metadata.variable_ids, variable_ids)
        add_unknown("formula_metadata", formula_id, "factor_id", design.formula_metadata.factor_ids, known_factor_ids)
    add_unknown("model_design", design.model_design_id or "model_design", "variable_id", design.variable_ids, variable_ids)
    add_unknown("model_design", design.model_design_id or "model_design", "factor_id", design.factor_ids, known_factor_ids)
    for comparison in design.planned_comparisons:
        add_unknown("planned_comparisons", comparison.comparison_id, "factor_id", comparison.factor_ids, known_factor_ids)
        add_unknown("planned_comparisons", comparison.comparison_id, "variable_id", comparison.variable_ids, variable_ids)
        add_unknown("planned_comparisons", comparison.comparison_id, "grouping_id", comparison.grouping_ids, known_grouping_ids)
        add_unknown("planned_comparisons", comparison.comparison_id, "cluster_id", comparison.cluster_ids, known_cluster_ids)
        add_unknown("planned_comparisons", comparison.comparison_id, "coding_id", comparison.coding_ids, known_coding_ids)
        add_unknown(
            "planned_comparisons",
            comparison.comparison_id,
            "contrast_metadata_id",
            comparison.contrast_metadata_ids,
            known_contrast_ids,
        )
    for contrast in design.contrast_metadata:
        add_unknown("contrast_metadata", contrast.contrast_id, "planned_comparison_id", contrast.comparison_ids, known_comparison_ids)
        add_unknown("contrast_metadata", contrast.contrast_id, "factor_id", contrast.factor_ids, known_factor_ids)
        add_unknown("contrast_metadata", contrast.contrast_id, "variable_id", contrast.variable_ids, variable_ids)
        add_unknown("contrast_metadata", contrast.contrast_id, "coding_id", contrast.coding_ids, known_coding_ids)

    for category, metadata_id, source_id in _model_design_metadata_source_references(design):
        if source_id not in source_ids:
            issues.append(
                _model_design_metadata_issue(
                    code="unknown_metadata_reference",
                    message=(
                        "unknown_metadata_reference: "
                        f"{category} {metadata_id!r} references unknown source_id {source_id!r}."
                    ),
                    metadata={
                        "owner_category": category,
                        "owner_id": metadata_id,
                        "reference_kind": "source_id",
                        "reference_id": source_id,
                    },
                )
            )

    if loaded_sources_by_id:
        for source_id, column_name, category, metadata_id in _model_design_metadata_columns(
            design,
            default_source_id=default_source_id,
        ):
            if not source_id:
                issues.append(
                    _model_design_metadata_issue(
                        code="unknown_metadata_reference",
                        message=(
                            "unknown_metadata_reference: "
                            f"{category} {metadata_id!r} declares metadata column {column_name!r} without a source_id."
                        ),
                        metadata={
                            "owner_category": category,
                            "owner_id": metadata_id,
                            "reference_kind": "source_id",
                            "column_name": column_name,
                        },
                    )
                )
                continue
            loaded_source = loaded_sources_by_id.get(source_id)
            if loaded_source is None or loaded_source.get("load_status") not in {"loaded", "empty"}:
                continue
            if not loaded_source.get("rows"):
                continue
            observed_columns = set(loaded_source.get("observed_columns", ()))
            if column_name not in observed_columns:
                issues.append(
                    _model_design_metadata_issue(
                        code="missing_metadata_column",
                        message=(
                            "missing_metadata_column: "
                            f"{category} {metadata_id!r} declares column {column_name!r}, "
                            f"but it was not observed in source {source_id!r}."
                        ),
                        metadata={
                            "owner_category": category,
                            "owner_id": metadata_id,
                            "source_id": source_id,
                            "column_name": column_name,
                        },
                    )
                )
    return tuple(issues)


def _model_design_metadata_issue(*, code: str, message: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "error",
        "code": code,
        "message": message,
        "warnings": (),
        "errors": (message,),
        "metadata": _json_safe_mapping(metadata),
    }


def _model_design_metadata_source_references(
    design: ModelDesignMetadataSpec,
) -> tuple[tuple[str, str, str], ...]:
    references: list[tuple[str, str, str]] = []

    def add(category: str, metadata_id: str, source_id: str | None) -> None:
        if source_id:
            references.append((category, metadata_id, source_id))

    for term in design.fixed_effect_terms:
        add("fixed_effect_terms", term.term_id, term.source_id)
    for term in design.random_effect_terms:
        add("random_effect_terms", term.term_id, term.source_id)
    for intercept in design.random_intercepts:
        add("random_intercepts", intercept.intercept_id, intercept.source_id)
    for slope in design.random_slopes:
        add("random_slopes", slope.slope_id, slope.source_id)
    for factor in design.repeated_factors:
        add("repeated_factors", factor.factor_id, factor.source_id)
    for factor in design.within_subject_factors:
        add("within_subject_factors", factor.factor_id, factor.source_id)
    for factor in design.between_subject_factors:
        add("between_subject_factors", factor.factor_id, factor.source_id)
    for grouping in design.grouping_factors:
        add("grouping_factors", grouping.grouping_id, grouping.source_id)
    for cluster in design.cluster_terms:
        add("cluster_terms", cluster.cluster_id, cluster.source_id)
    for role in design.timepoint_roles:
        add("timepoint_roles", role.role_id, role.source_id)
    for coding in design.categorical_coding:
        add("categorical_coding", coding.coding_id, coding.source_id)
    return tuple(references)


def _model_design_metadata_columns(
    design: ModelDesignMetadataSpec,
    *,
    default_source_id: str | None,
) -> tuple[tuple[str | None, str, str, str], ...]:
    columns: list[tuple[str | None, str, str, str]] = []

    def add(source_id: str | None, column_name: str | None, category: str, metadata_id: str) -> None:
        if column_name:
            columns.append((source_id or default_source_id, column_name, category, metadata_id))

    def add_many(source_id: str | None, column_names: Sequence[str], category: str, metadata_id: str) -> None:
        for column_name in column_names:
            add(source_id, column_name, category, metadata_id)

    for term in design.fixed_effect_terms:
        add_many(term.source_id, term.column_names, "fixed_effect_terms", term.term_id)
    for term in design.random_effect_terms:
        add_many(term.source_id, term.column_names, "random_effect_terms", term.term_id)
    for intercept in design.random_intercepts:
        add_many(intercept.source_id, intercept.grouping_columns, "random_intercepts", intercept.intercept_id)
    for slope in design.random_slopes:
        add_many(slope.source_id, slope.column_names, "random_slopes", slope.slope_id)
        add_many(slope.source_id, slope.grouping_columns, "random_slopes", slope.slope_id)
    for factor in design.repeated_factors:
        add(factor.source_id, factor.column_name, "repeated_factors", factor.factor_id)
    for factor in design.within_subject_factors:
        add(factor.source_id, factor.column_name, "within_subject_factors", factor.factor_id)
    for factor in design.between_subject_factors:
        add(factor.source_id, factor.column_name, "between_subject_factors", factor.factor_id)
    for grouping in design.grouping_factors:
        add(grouping.source_id, grouping.column_name, "grouping_factors", grouping.grouping_id)
    for cluster in design.cluster_terms:
        add(cluster.source_id, cluster.column_name, "cluster_terms", cluster.cluster_id)
    for role in design.timepoint_roles:
        add(role.source_id, role.column_name, "timepoint_roles", role.role_id)
    for coding in design.categorical_coding:
        add(coding.source_id, coding.column_name, "categorical_coding", coding.coding_id)
    return tuple(columns)


def _model_design_metadata_id_tuple(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values if str(value))


def _model_design_metadata_mapping(key: str, values: Sequence[Any]) -> dict[str, Any]:
    return {
        "metadata_version": TABULAR_ASSOCIATION_REPEATED_MEASURES_METADATA_VERSION,
        key: tuple(_json_safe(value) for value in values),
        "metadata_only": True,
        "model_fitting_deferred": True,
    }


def _repeated_measures_model_plan_id(
    method: AssociationMethodSpec,
    outcome: AssociationVariableSpec,
    predictor: AssociationVariableSpec,
) -> str:
    return f"{method.method_id}::{outcome.variable_id}::{predictor.variable_id}"


def _repeated_measures_source_ids_to_load(
    workflow: TabularAssociationWorkflowSpec,
    model_plan_rows: Sequence[RepeatedMeasuresModelPlanRow],
) -> set[str]:
    source_ids = {row.source_id for row in model_plan_rows if row.source_id}
    if source_ids:
        return set(source_ids)
    if _repeated_measures_methods(workflow) and workflow.repeated_measures is not None:
        return {workflow.repeated_measures.source_id}
    return set()


def _repeated_measures_method_source_ids(
    workflow: TabularAssociationWorkflowSpec,
    model_plan_rows: Sequence[RepeatedMeasuresModelPlanRow],
) -> tuple[str, ...]:
    source_ids = _unique_texts(tuple(row.source_id for row in model_plan_rows if row.source_id))
    if source_ids:
        return source_ids
    if workflow.repeated_measures is not None:
        return (workflow.repeated_measures.source_id,)
    return ()


def _repeated_measures_source_methods(
    model_plan_rows: Sequence[RepeatedMeasuresModelPlanRow],
) -> tuple[tuple[str | None, str | None], ...]:
    pairs: list[tuple[str | None, str | None]] = []
    seen: set[tuple[str | None, str | None]] = set()
    for row in model_plan_rows:
        pair = (row.source_id, row.method_id)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return tuple(pairs)


def _repeated_measures_design_rows_for_method_source(
    *,
    workflow: TabularAssociationWorkflowSpec,
    method: AssociationMethodSpec,
    source: TabularSourceSpec,
    loaded_source: Mapping[str, Any],
    model_plan_rows: Sequence[RepeatedMeasuresModelPlanRow],
) -> tuple[RepeatedMeasuresDesignSummaryRow, tuple[RepeatedMeasuresFactorSummaryRow, ...], tuple[RepeatedMeasuresDesignQcRow, ...]]:
    rows = tuple(loaded_source["rows"])
    observed_columns = set(loaded_source["observed_columns"])
    repeated_columns = _repeated_design_columns_from_plan_rows(workflow, source, model_plan_rows)
    subject_id_column = repeated_columns["subject_id_column"]
    repeated_unit_columns = repeated_columns["repeated_unit_columns"]
    repeated_factor_columns = repeated_columns["repeated_factor_columns"]
    cluster_columns = repeated_columns["cluster_columns"]
    required_columns_by_role = _repeated_required_columns_by_role(
        subject_id_column=subject_id_column,
        repeated_unit_columns=repeated_unit_columns,
        repeated_factor_columns=repeated_factor_columns,
        cluster_columns=cluster_columns,
        model_plan_rows=model_plan_rows,
    )

    qc_rows: list[RepeatedMeasuresDesignQcRow] = []
    missing_required_messages: list[str] = []
    if rows:
        for column_name, roles in required_columns_by_role.items():
            if column_name in observed_columns:
                continue
            role = _repeated_required_column_role(roles)
            code = _repeated_missing_column_code(role)
            message = f"{code}: required {role} column {column_name!r} was not observed."
            missing_required_messages.append(message)
            qc_rows.append(
                _repeated_design_qc_row(
                    workflow=workflow,
                    source_id=source.source_id,
                    method=method,
                    status="error",
                    code=code,
                    message=message,
                    row_count=len(rows),
                    errors=(message,),
                    metadata={"column_name": column_name, "roles": roles},
                )
            )

    participant_counts = _participant_observation_counts(rows, subject_id_column)
    participant_count = len(participant_counts)
    observation_counts = tuple(participant_counts.values())
    min_observations = min(observation_counts) if observation_counts else 0
    max_observations = max(observation_counts) if observation_counts else 0
    singleton_count = sum(1 for count in observation_counts if count == 1)
    insufficient_count = sum(1 for count in observation_counts if count < 2)
    missing_subject_count = _missing_column_value_count(rows, subject_id_column)
    missing_repeated_key_count = _missing_repeated_key_count(rows, repeated_unit_columns)
    duplicate_count, duplicate_keys = _duplicate_repeated_unit_count(rows, repeated_unit_columns)
    cluster_count = _cluster_count(rows, cluster_columns)
    balanced_design, imbalance_indicator = _repeated_design_balance(
        rows,
        subject_id_column=subject_id_column,
        factor_columns=repeated_factor_columns,
        participant_counts=participant_counts,
    )

    if not rows:
        message = "empty_source_rows: source has zero rows for repeated-measures design QC."
        qc_rows.append(
            _repeated_design_qc_row(
                workflow=workflow,
                source_id=source.source_id,
                method=method,
                status="warning",
                code="empty_source_rows",
                message=message,
                warnings=(message,),
            )
        )
    if missing_subject_count:
        message = f"missing_participant_ids: observed {missing_subject_count} rows with missing participant identifiers."
        qc_rows.append(
            _repeated_design_qc_row(
                workflow=workflow,
                source_id=source.source_id,
                method=method,
                status="error",
                code="missing_participant_ids",
                message=message,
                row_count=len(rows),
                participant_count=participant_count,
                missing_subject_id_count=missing_subject_count,
                errors=(message,),
            )
        )
    if missing_repeated_key_count:
        message = f"missing_repeated_keys: observed {missing_repeated_key_count} rows with missing repeated-unit keys."
        qc_rows.append(
            _repeated_design_qc_row(
                workflow=workflow,
                source_id=source.source_id,
                method=method,
                status="error",
                code="missing_repeated_keys",
                message=message,
                row_count=len(rows),
                participant_count=participant_count,
                missing_repeated_key_count=missing_repeated_key_count,
                errors=(message,),
                metadata={"repeated_unit_columns": repeated_unit_columns},
            )
        )
    if duplicate_count:
        message = f"duplicate_repeated_unit_rows: observed {duplicate_count} duplicate repeated-unit rows."
        qc_rows.append(
            _repeated_design_qc_row(
                workflow=workflow,
                source_id=source.source_id,
                method=method,
                status="error",
                code="duplicate_repeated_unit_rows",
                message=message,
                row_count=len(rows),
                participant_count=participant_count,
                duplicate_repeated_unit_count=duplicate_count,
                errors=(message,),
                metadata={"duplicate_keys": duplicate_keys, "repeated_unit_columns": repeated_unit_columns},
            )
        )
    if singleton_count:
        message = f"singleton_participants: observed {singleton_count} participants with one observation."
        qc_rows.append(
            _repeated_design_qc_row(
                workflow=workflow,
                source_id=source.source_id,
                method=method,
                status="warning",
                code="singleton_participants",
                message=message,
                row_count=len(rows),
                participant_count=participant_count,
                singleton_participant_count=singleton_count,
                warnings=(message,),
            )
        )
    if insufficient_count:
        message = f"insufficient_repeated_observations: observed {insufficient_count} participants with fewer than two observations."
        qc_rows.append(
            _repeated_design_qc_row(
                workflow=workflow,
                source_id=source.source_id,
                method=method,
                status="warning",
                code="insufficient_repeated_observations",
                message=message,
                row_count=len(rows),
                participant_count=participant_count,
                insufficient_repeat_participant_count=insufficient_count,
                warnings=(message,),
            )
        )
    if not balanced_design and rows:
        message = f"imbalanced_repeated_design: {imbalance_indicator or 'repeated observations are imbalanced'}."
        qc_rows.append(
            _repeated_design_qc_row(
                workflow=workflow,
                source_id=source.source_id,
                method=method,
                status="warning",
                code="imbalanced_repeated_design",
                message=message,
                row_count=len(rows),
                participant_count=participant_count,
                warnings=(message,),
                metadata={"imbalance_indicator": imbalance_indicator},
            )
        )

    factor_rows = _repeated_factor_summary_rows(
        workflow=workflow,
        source_id=source.source_id,
        method=method,
        rows=rows,
        subject_id_column=subject_id_column,
        factor_columns=repeated_factor_columns,
    )
    status = "error" if any(row.status == "error" for row in qc_rows) else (
        "warning" if any(row.status in {"warning", "deferred"} for row in qc_rows) else "ok"
    )
    if not rows:
        status = "warning"
    summary_warnings = _unique_texts([warning for row in qc_rows for warning in row.warnings])
    summary_errors = _unique_texts([error for row in qc_rows for error in row.errors] + missing_required_messages)
    if status == "ok":
        code = "repeated_measures_design_summarized"
        message = "Repeated-measures design was summarized without fitting a model."
    elif status == "warning":
        code = "repeated_measures_design_qc_warning"
        message = "Repeated-measures design QC completed with warnings and no model fitting."
    else:
        code = "repeated_measures_design_qc_error"
        message = "Repeated-measures design QC found errors and no model fitting was attempted."
    summary_row = RepeatedMeasuresDesignSummaryRow(
        workflow_id=workflow.workflow_id,
        source_id=source.source_id,
        method_id=method.method_id,
        method_name=method.method_name,
        row_count=len(rows),
        observation_count=len(rows),
        participant_count=participant_count,
        cluster_count=cluster_count,
        min_observations_per_participant=min_observations,
        max_observations_per_participant=max_observations,
        singleton_participant_count=singleton_count,
        insufficient_repeat_participant_count=insufficient_count,
        duplicate_repeated_unit_count=duplicate_count,
        missing_subject_id_count=missing_subject_count,
        missing_repeated_key_count=missing_repeated_key_count,
        balanced_design=balanced_design,
        imbalance_indicator=imbalance_indicator,
        subject_id_column=subject_id_column,
        repeated_unit_columns=repeated_unit_columns,
        repeated_factor_columns=repeated_factor_columns,
        cluster_columns=cluster_columns,
        runtime_backend=RUNTIME_BACKEND_RECORDS,
        status=status,
        code=code,
        message=message,
        warnings=summary_warnings,
        errors=summary_errors,
    )
    return summary_row, factor_rows, tuple(qc_rows)


def _repeated_design_columns_from_plan_rows(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    model_plan_rows: Sequence[RepeatedMeasuresModelPlanRow],
) -> dict[str, Any]:
    if model_plan_rows:
        first = model_plan_rows[0]
        return {
            "subject_id_column": first.subject_id_column or source.schema.subject_id_column,
            "repeated_unit_columns": _unique_texts(
                tuple(column for row in model_plan_rows for column in row.repeated_unit_columns)
                or (source.schema.subject_id_column,)
            ),
            "repeated_factor_columns": _unique_texts(tuple(column for row in model_plan_rows for column in row.repeated_factor_columns)),
            "cluster_columns": _unique_texts(tuple(column for row in model_plan_rows for column in row.cluster_columns)),
        }
    repeated_columns = _repeated_measures_columns_for_source(workflow, source)
    repeated_metadata = workflow.repeated_measures.metadata if workflow.repeated_measures is not None else {}
    return {
        "subject_id_column": repeated_columns["subject_id_column"],
        "repeated_unit_columns": repeated_columns["repeated_unit_columns"],
        "repeated_factor_columns": _repeated_measures_factor_columns(
            method=AssociationMethodSpec(method_id="method-placeholder", method_name=METHOD_REPEATED_MEASURES),
            repeated_metadata=repeated_metadata,
            timepoint_column=repeated_columns["timepoint_column"],
        ),
        "cluster_columns": _metadata_text_values(repeated_metadata, keys=("cluster_columns", "cluster_column", "cluster", "clusters")),
    }


def _repeated_required_columns_by_role(
    *,
    subject_id_column: str | None,
    repeated_unit_columns: Sequence[str],
    repeated_factor_columns: Sequence[str],
    cluster_columns: Sequence[str],
    model_plan_rows: Sequence[RepeatedMeasuresModelPlanRow],
) -> dict[str, tuple[str, ...]]:
    roles_by_column: dict[str, list[str]] = {}

    def add(column_name: str | None, role: str) -> None:
        if not column_name:
            return
        roles_by_column.setdefault(column_name, []).append(role)

    add(subject_id_column, "subject")
    for column in repeated_unit_columns:
        add(column, "repeated_unit")
    for column in repeated_factor_columns:
        add(column, "repeated_factor")
    for column in cluster_columns:
        add(column, "cluster")
    for row in model_plan_rows:
        add(row.outcome_column, "outcome")
        add(row.predictor_column, "predictor")
        for column in row.covariate_columns:
            add(column, "covariate")
        for column in row.group_columns:
            add(column, "group")
    return {column: _unique_texts(roles) for column, roles in roles_by_column.items()}


def _repeated_required_column_role(roles: Sequence[str]) -> str:
    for role in ("subject", "repeated_unit", "repeated_factor", "outcome", "predictor", "covariate", "group", "cluster"):
        if role in roles:
            return role
    return roles[0] if roles else "required"


def _repeated_missing_column_code(role: str) -> str:
    if role == "subject":
        return "missing_subject_id_column"
    if role == "repeated_unit":
        return "missing_repeated_unit_column"
    if role == "repeated_factor":
        return "missing_repeated_factor_column"
    return f"missing_required_{role}_column"


def _participant_observation_counts(
    rows: Sequence[Mapping[str, Any]],
    subject_id_column: str | None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not subject_id_column:
        return counts
    for row in rows:
        value = row.get(subject_id_column)
        if _is_missing_value(value):
            continue
        participant_id = _safe_value_repr(value)
        counts[participant_id] = counts.get(participant_id, 0) + 1
    return counts


def _missing_column_value_count(rows: Sequence[Mapping[str, Any]], column_name: str | None) -> int:
    if not column_name:
        return len(rows) if rows else 0
    return sum(1 for row in rows if _is_missing_value(row.get(column_name)))


def _missing_repeated_key_count(rows: Sequence[Mapping[str, Any]], repeated_unit_columns: Sequence[str]) -> int:
    if not repeated_unit_columns:
        return len(rows) if rows else 0
    return sum(1 for row in rows if any(_is_missing_value(row.get(column)) for column in repeated_unit_columns))


def _duplicate_repeated_unit_count(
    rows: Sequence[Mapping[str, Any]],
    repeated_unit_columns: Sequence[str],
) -> tuple[int, tuple[str, ...]]:
    if not repeated_unit_columns:
        return 0, ()
    counts: dict[str, int] = {}
    for row in rows:
        values = tuple(row.get(column) for column in repeated_unit_columns)
        if any(_is_missing_value(value) for value in values):
            continue
        key = _key_value_repr(values)
        counts[key] = counts.get(key, 0) + 1
    duplicate_keys = tuple(key for key, count in counts.items() if count > 1)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    return duplicate_count, duplicate_keys


def _cluster_count(rows: Sequence[Mapping[str, Any]], cluster_columns: Sequence[str]) -> int:
    if not cluster_columns:
        return 0
    clusters: set[str] = set()
    for row in rows:
        values = tuple(row.get(column) for column in cluster_columns)
        if any(_is_missing_value(value) for value in values):
            continue
        clusters.add(_key_value_repr(values))
    return len(clusters)


def _repeated_design_balance(
    rows: Sequence[Mapping[str, Any]],
    *,
    subject_id_column: str | None,
    factor_columns: Sequence[str],
    participant_counts: Mapping[str, int],
) -> tuple[bool, str | None]:
    if not rows or not participant_counts:
        return True, None
    indicators: list[str] = []
    if len(set(participant_counts.values())) > 1:
        indicators.append("observation_count_by_participant")
    if subject_id_column:
        for factor_column in factor_columns:
            level_sets: dict[str, tuple[str, ...]] = {}
            for row in rows:
                participant_value = row.get(subject_id_column)
                factor_value = row.get(factor_column)
                if _is_missing_value(participant_value) or _is_missing_value(factor_value):
                    continue
                participant_id = _safe_value_repr(participant_value)
                current = set(level_sets.get(participant_id, ()))
                current.add(_safe_value_repr(factor_value))
                level_sets[participant_id] = tuple(sorted(current))
            if level_sets and len({levels for levels in level_sets.values()}) > 1:
                indicators.append(f"levels_by_participant:{factor_column}")
    if indicators:
        return False, ",".join(indicators)
    return True, None


def _repeated_factor_summary_rows(
    *,
    workflow: TabularAssociationWorkflowSpec,
    source_id: str,
    method: AssociationMethodSpec,
    rows: Sequence[Mapping[str, Any]],
    subject_id_column: str | None,
    factor_columns: Sequence[str],
) -> tuple[RepeatedMeasuresFactorSummaryRow, ...]:
    summary_rows: list[RepeatedMeasuresFactorSummaryRow] = []
    for factor_column in factor_columns:
        observations_by_level: dict[str, int] = {}
        participants_by_level_sets: dict[str, set[str]] = {}
        missing_count = 0
        for row in rows:
            value = row.get(factor_column)
            if _is_missing_value(value):
                missing_count += 1
                continue
            level = _safe_value_repr(value)
            observations_by_level[level] = observations_by_level.get(level, 0) + 1
            if subject_id_column:
                participant_value = row.get(subject_id_column)
                if not _is_missing_value(participant_value):
                    participants_by_level_sets.setdefault(level, set()).add(_safe_value_repr(participant_value))
        levels = tuple(observations_by_level.keys())
        participants_by_level = {level: len(participants_by_level_sets.get(level, set())) for level in levels}
        warnings: list[str] = []
        errors: list[str] = []
        status = "ok"
        code = "repeated_factor_summarized"
        message = "Repeated-factor levels were summarized without fitting a model."
        if not rows:
            status = "warning"
            code = "empty_repeated_factor_source"
            message = f"Repeated-factor column {factor_column!r} has no rows to summarize."
            warnings.append(message)
        elif missing_count:
            status = "warning"
            code = "missing_repeated_factor_values"
            message = f"Repeated-factor column {factor_column!r} has {missing_count} missing values."
            warnings.append(message)
        summary_rows.append(
            RepeatedMeasuresFactorSummaryRow(
                workflow_id=workflow.workflow_id,
                source_id=source_id,
                method_id=method.method_id,
                method_name=method.method_name,
                factor_column=factor_column,
                level_count=len(levels),
                levels=levels,
                observations_by_level=observations_by_level,
                participants_by_level=participants_by_level,
                missing_count=missing_count,
                runtime_backend=RUNTIME_BACKEND_RECORDS,
                status=status,
                code=code,
                message=message,
                warnings=warnings,
                errors=errors,
            )
        )
    return tuple(summary_rows)


def _repeated_design_qc_row(
    *,
    workflow: TabularAssociationWorkflowSpec,
    source_id: str | None,
    method: AssociationMethodSpec,
    status: str,
    code: str,
    message: str,
    row_count: int = 0,
    participant_count: int = 0,
    duplicate_repeated_unit_count: int = 0,
    missing_subject_id_count: int = 0,
    missing_repeated_key_count: int = 0,
    singleton_participant_count: int = 0,
    insufficient_repeat_participant_count: int = 0,
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> RepeatedMeasuresDesignQcRow:
    return RepeatedMeasuresDesignQcRow(
        workflow_id=workflow.workflow_id,
        source_id=source_id,
        method_id=method.method_id,
        method_name=method.method_name,
        model_plan_id=None,
        runtime_backend=RUNTIME_BACKEND_RECORDS,
        status=status,
        code=code,
        message=message,
        row_count=row_count,
        participant_count=participant_count,
        duplicate_repeated_unit_count=duplicate_repeated_unit_count,
        missing_subject_id_count=missing_subject_id_count,
        missing_repeated_key_count=missing_repeated_key_count,
        singleton_participant_count=singleton_participant_count,
        insufficient_repeat_participant_count=insufficient_repeat_participant_count,
        warnings=warnings,
        errors=errors,
        metadata={} if metadata is None else metadata,
    )


def _repeated_measures_metadata_qc_rows(
    *,
    workflow: TabularAssociationWorkflowSpec,
    method: AssociationMethodSpec,
    model_plan_row: RepeatedMeasuresModelPlanRow,
    repeated_metadata: Mapping[str, Any],
    loaded_sources_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[RepeatedMeasuresDesignQcRow, ...]:
    design, parse_errors = _safe_model_design_metadata_for_method(
        method=method,
        repeated_metadata=repeated_metadata,
    )
    rows: list[RepeatedMeasuresDesignQcRow] = []
    for message in parse_errors:
        rows.append(
            RepeatedMeasuresDesignQcRow(
                workflow_id=workflow.workflow_id,
                source_id=model_plan_row.source_id,
                method_id=method.method_id,
                method_name=method.method_name,
                model_plan_id=model_plan_row.model_plan_id,
                runtime_backend=RUNTIME_BACKEND_RECORDS,
                status="error",
                code="invalid_model_design_metadata",
                message=message,
                errors=(message,),
                metadata={
                    "metadata_version": TABULAR_ASSOCIATION_REPEATED_MEASURES_METADATA_VERSION,
                    "metadata_only": True,
                    "model_fitting_deferred": True,
                },
            )
        )
    if _model_design_has_explicit_metadata(design):
        counts = _model_design_metadata_counts(design)
        rows.append(
            RepeatedMeasuresDesignQcRow(
                workflow_id=workflow.workflow_id,
                source_id=model_plan_row.source_id,
                method_id=method.method_id,
                method_name=method.method_name,
                model_plan_id=model_plan_row.model_plan_id,
                runtime_backend=RUNTIME_BACKEND_RECORDS,
                status="ok",
                code="model_design_metadata_only",
                message="Model-design declarations are metadata only; model fitting remains deferred.",
                metadata={
                    "metadata_version": TABULAR_ASSOCIATION_REPEATED_MEASURES_METADATA_VERSION,
                    "model_design_id": design.model_design_id,
                    "metadata_only": True,
                    "model_fitting_deferred": True,
                    **counts,
                },
            )
        )
    for issue in _model_design_metadata_validation_issues(
        workflow=workflow,
        design=design,
        default_source_id=model_plan_row.source_id,
        loaded_sources_by_id=loaded_sources_by_id,
    ):
        rows.append(
            RepeatedMeasuresDesignQcRow(
                workflow_id=workflow.workflow_id,
                source_id=model_plan_row.source_id,
                method_id=method.method_id,
                method_name=method.method_name,
                model_plan_id=model_plan_row.model_plan_id,
                runtime_backend=RUNTIME_BACKEND_RECORDS,
                status=str(issue["status"]),
                code=str(issue["code"]),
                message=str(issue["message"]),
                warnings=tuple(issue["warnings"]),
                errors=tuple(issue["errors"]),
                metadata={
                    "metadata_version": TABULAR_ASSOCIATION_REPEATED_MEASURES_METADATA_VERSION,
                    "metadata_only": True,
                    "model_fitting_deferred": True,
                    **dict(issue["metadata"]),
                },
            )
        )
    return tuple(rows)


def _model_design_metadata_counts(design: ModelDesignMetadataSpec) -> dict[str, int]:
    return {
        "fixed_effect_term_count": len(design.fixed_effect_terms),
        "random_effect_term_count": len(design.random_effect_terms),
        "random_intercept_count": len(design.random_intercepts),
        "random_slope_count": len(design.random_slopes),
        "repeated_factor_count": len(design.repeated_factors),
        "within_subject_factor_count": len(design.within_subject_factors),
        "between_subject_factor_count": len(design.between_subject_factors),
        "grouping_factor_count": len(design.grouping_factors),
        "cluster_term_count": len(design.cluster_terms),
        "timepoint_role_count": len(design.timepoint_roles),
        "categorical_coding_count": len(design.categorical_coding),
        "planned_comparison_count": len(design.planned_comparisons),
        "contrast_metadata_count": len(design.contrast_metadata),
    }


def _repeated_measures_metadata_provenance_summary(
    model_plan_rows: Sequence[RepeatedMeasuresModelPlanRow],
) -> dict[str, int]:
    fixed_effect_ids: set[str] = set()
    random_effect_ids: set[str] = set()
    repeated_factor_ids: set[str] = set()
    planned_comparison_ids: set[str] = set()
    contrast_metadata_ids: set[str] = set()
    for row in model_plan_rows:
        fixed_effect_ids.update(row.fixed_effect_term_ids)
        random_effect_ids.update(row.random_effect_term_ids)
        repeated_factor_ids.update(row.repeated_factor_ids)
        planned_comparison_ids.update(row.planned_comparison_ids)
        contrast_metadata_ids.update(row.contrast_metadata_ids)
    return {
        "fixed_effect_term_count": len(fixed_effect_ids),
        "random_effect_term_count": len(random_effect_ids),
        "repeated_factor_count": len(repeated_factor_ids),
        "planned_comparison_count": len(planned_comparison_ids),
        "contrast_metadata_count": len(contrast_metadata_ids),
    }


def _repeated_measures_messages(
    *,
    workflow_validation_rows: Sequence[AssociationValidationRow],
    model_plan_rows: Sequence[RepeatedMeasuresModelPlanRow],
    source_load_rows: Sequence[TabularSourceLoadRow],
    design_summary_rows: Sequence[RepeatedMeasuresDesignSummaryRow],
    factor_summary_rows: Sequence[RepeatedMeasuresFactorSummaryRow],
    qc_rows: Sequence[RepeatedMeasuresDesignQcRow],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    errors: list[str] = []

    def add_status(status: str, message: str) -> None:
        if not message:
            return
        if status == "error":
            errors.append(message)
        elif status in {"warning", "deferred"}:
            warnings.append(message)

    for row in workflow_validation_rows:
        add_status(row.status, row.message)
    for row in model_plan_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in source_load_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in (*design_summary_rows, *factor_summary_rows, *qc_rows):
        warnings.extend(row.warnings)
        errors.extend(row.errors)
        add_status(row.status, getattr(row, "message", ""))
    return _unique_texts(warnings), _unique_texts(errors)


def _repeated_measures_provenance_rows(
    workflow: TabularAssociationWorkflowSpec,
    *,
    executed: bool,
    plan_only: bool,
    source_count: int,
    loaded_source_count: int,
    method_count: int,
    repeated_method_count: int,
    model_plan_row_count: int,
    design_summary_row_count: int,
    factor_summary_row_count: int,
    qc_row_count: int,
    qc_mode: str,
    source_methods: Sequence[tuple[str | None, str | None]],
    metadata_summary: Mapping[str, Any] | None = None,
) -> tuple[TabularAssociationRepeatedMeasuresProvenanceRow, ...]:
    metadata_summary = {} if metadata_summary is None else metadata_summary
    values: list[tuple[str | None, str | None, str, Any]] = [
        (None, None, "schema_version", SCHEMA_VERSION),
        (None, None, "repeated_measures_plan_version", TABULAR_ASSOCIATION_REPEATED_MEASURES_PLAN_VERSION),
        (
            None,
            None,
            "repeated_measures_metadata_version",
            TABULAR_ASSOCIATION_REPEATED_MEASURES_METADATA_VERSION,
        ),
        (None, None, "workflow_id", workflow.workflow_id),
        (None, None, "requested_backend", workflow.backend),
        (None, None, "runtime_backend", RUNTIME_BACKEND_RECORDS),
        (None, None, "source_count", source_count),
        (None, None, "loaded_source_count", loaded_source_count),
        (None, None, "method_count", method_count),
        (None, None, "repeated_method_count", repeated_method_count),
        (None, None, "model_plan_row_count", model_plan_row_count),
        (None, None, "design_summary_row_count", design_summary_row_count),
        (None, None, "factor_summary_row_count", factor_summary_row_count),
        (None, None, "qc_row_count", qc_row_count),
        (None, None, "qc_mode", qc_mode),
        (None, None, "model_design_metadata_only", True),
        (None, None, "model_fitting_deferred", True),
        (None, None, "executed", executed),
        (None, None, "plan_only", plan_only),
        (None, None, "will_write", False),
        (None, None, "output_written", False),
        (None, None, "output_paths_written", ()),
        (None, None, "no_output_written", True),
    ]
    for key in (
        "fixed_effect_term_count",
        "random_effect_term_count",
        "repeated_factor_count",
        "planned_comparison_count",
        "contrast_metadata_count",
    ):
        values.append((None, None, key, int(metadata_summary.get(key, 0))))
    for source_id, method_id in source_methods:
        values.append((source_id, method_id, "source_method_model_fitting_deferred", True))
        values.append((source_id, method_id, "source_method_no_output_written", True))
    return tuple(
        TabularAssociationRepeatedMeasuresProvenanceRow(
            workflow_id=workflow.workflow_id,
            source_id=source_id,
            method_id=method_id,
            runtime_backend=RUNTIME_BACKEND_RECORDS,
            step_version=TABULAR_ASSOCIATION_REPEATED_MEASURES_PLAN_VERSION,
            model_fitting_deferred=True,
            will_write=False,
            output_written=False,
            no_output_written=True,
            output_paths_written=(),
            key=key,
            value=value,
        )
        for source_id, method_id, key, value in values
    )


def _repeated_measures_provenance_rows_for_error(
    workflow_id: str,
    *,
    executed: bool,
    plan_only: bool,
    qc_mode: str,
) -> tuple[TabularAssociationRepeatedMeasuresProvenanceRow, ...]:
    values = (
        ("schema_version", SCHEMA_VERSION),
        ("repeated_measures_plan_version", TABULAR_ASSOCIATION_REPEATED_MEASURES_PLAN_VERSION),
        ("repeated_measures_metadata_version", TABULAR_ASSOCIATION_REPEATED_MEASURES_METADATA_VERSION),
        ("workflow_id", workflow_id),
        ("runtime_backend", RUNTIME_BACKEND_RECORDS),
        ("qc_mode", qc_mode),
        ("model_design_metadata_only", True),
        ("model_fitting_deferred", True),
        ("executed", executed),
        ("plan_only", plan_only),
        ("will_write", False),
        ("output_written", False),
        ("output_paths_written", ()),
        ("no_output_written", True),
    )
    return tuple(
        TabularAssociationRepeatedMeasuresProvenanceRow(
            workflow_id=workflow_id,
            source_id=None,
            method_id=None,
            runtime_backend=RUNTIME_BACKEND_RECORDS,
            step_version=TABULAR_ASSOCIATION_REPEATED_MEASURES_PLAN_VERSION,
            model_fitting_deferred=True,
            will_write=False,
            output_written=False,
            no_output_written=True,
            output_paths_written=(),
            key=key,
            value=value,
        )
        for key, value in values
    )


def _adjusted_methods(workflow: TabularAssociationWorkflowSpec) -> tuple[AssociationMethodSpec, ...]:
    return tuple(method for method in workflow.methods if method.method_name in {METHOD_PARTIAL_CORRELATION, METHOD_REGRESSION})


def _adjusted_method_count(workflow: TabularAssociationWorkflowSpec) -> int:
    return len(_adjusted_methods(workflow))


def _adjusted_pair_plan_rows(workflow: TabularAssociationWorkflowSpec) -> tuple[AdjustedAssociationPairPlanRow, ...]:
    outcome_by_id = {outcome.variable_id: outcome for outcome in workflow.outcomes}
    predictor_by_id = {predictor.variable_id: predictor for predictor in workflow.predictors}
    covariate_by_id = {covariate.variable_id: covariate for covariate in workflow.covariates}
    rows: list[AdjustedAssociationPairPlanRow] = []
    for method in _adjusted_methods(workflow):
        outcomes = _resolved_method_variables(method.outcome_ids, tuple(workflow.outcomes), outcome_by_id)
        predictors = _resolved_method_variables(method.predictor_ids, tuple(workflow.predictors), predictor_by_id)
        covariates = _adjusted_method_covariates(method, workflow, covariate_by_id)
        method_deferred_codes = _adjusted_method_deferred_codes(method)
        for outcome in outcomes:
            for predictor in predictors:
                deferred_codes: list[str] = []
                source_ids = {outcome.source_id, predictor.source_id, *(covariate.source_id for covariate in covariates)}
                if len(source_ids) != 1:
                    deferred_codes.append("cross_source_adjusted_association_deferred")
                if method.method_name == METHOD_PARTIAL_CORRELATION and not covariates:
                    deferred_codes.append("missing_partial_covariates")
                deferred_codes.extend(method_deferred_codes)
                deferred = bool(deferred_codes)
                code = deferred_codes[0] if deferred else "adjusted_association_pair_planned"
                warnings = tuple(_adjusted_deferred_message(code, method, outcome, predictor) for code in deferred_codes)
                source_id = next(iter(source_ids)) if len(source_ids) == 1 else None
                rows.append(
                    AdjustedAssociationPairPlanRow(
                        workflow_id=workflow.workflow_id,
                        pair_id=_adjusted_pair_id(method, outcome, predictor),
                        method_id=method.method_id,
                        method_kind=_adjusted_method_kind(method),
                        method_name=method.method_name,
                        family_id=method.family_id,
                        source_id=source_id,
                        outcome_id=outcome.variable_id,
                        outcome_source_id=outcome.source_id,
                        outcome_column=outcome.column_name,
                        predictor_id=predictor.variable_id,
                        predictor_source_id=predictor.source_id,
                        predictor_column=predictor.column_name,
                        covariate_ids=tuple(covariate.variable_id for covariate in covariates),
                        covariate_source_ids=tuple(covariate.source_id for covariate in covariates),
                        covariate_columns=tuple(covariate.column_name for covariate in covariates),
                        covariate_count=len(covariates),
                        executable=not deferred,
                        deferred=deferred,
                        status="deferred" if deferred else "planned",
                        code=code,
                        warnings=warnings,
                        errors=(),
                    )
                )
    return tuple(rows)


def _adjusted_method_covariates(
    method: AssociationMethodSpec,
    workflow: TabularAssociationWorkflowSpec,
    covariate_by_id: Mapping[str, AssociationVariableSpec],
) -> tuple[AssociationVariableSpec, ...]:
    if method.method_name == METHOD_PARTIAL_CORRELATION:
        if method.covariate_ids:
            return _resolved_method_variables(method.covariate_ids, tuple(workflow.covariates), covariate_by_id)
        return tuple(workflow.covariates)
    return _resolved_method_variables(method.covariate_ids, tuple(workflow.covariates), covariate_by_id) if method.covariate_ids else ()


def _adjusted_method_kind(method: AssociationMethodSpec) -> str:
    return method.method_name


def _adjusted_method_deferred_codes(method: AssociationMethodSpec) -> tuple[str, ...]:
    codes: list[str] = []
    if method.grouping_ids:
        codes.append("grouped_adjusted_association_deferred")
    metadata_checks = (
        ("rank_adjusted_association_deferred", ("rank", "spearman", "rank_adjusted", "partial_spearman")),
        ("stratified_adjusted_association_deferred", ("stratification", "stratify", "strata", "stratification_ids")),
        ("repeated_measures_adjusted_association_deferred", ("repeated", "repeated_measures", "repeated_measures_ids")),
        ("mixed_model_adjusted_association_deferred", ("mixed", "mixed_model", "random_effects")),
        ("generalized_linear_model_deferred", ("glm", "generalized_linear_model", "link")),
        ("interaction_model_deferred", ("interaction", "interactions", "interaction_terms")),
        ("mediation_model_deferred", ("mediation", "mediator", "mediators")),
        ("moderation_model_deferred", ("moderation", "moderator", "moderators")),
        ("categorical_encoding_deferred", ("categorical_encoding", "dummy_encoding", "one_hot", "contrasts")),
        ("nonlinear_model_deferred", ("nonlinear", "non_linear", "spline", "polynomial")),
        ("machine_learning_model_deferred", ("machine_learning", "ml", "estimator")),
        ("robust_or_clustered_uncertainty_deferred", ("robust", "robust_se", "clustered_se", "cluster", "clusters")),
    )
    for code, keys in metadata_checks:
        if code in codes:
            continue
        if any(_metadata_truthy(method.metadata.get(key)) for key in keys):
            codes.append(code)
    model_type = method.metadata.get("model_type")
    if method.method_name == METHOD_REGRESSION and isinstance(model_type, str):
        normalized = model_type.strip().lower()
        if normalized and normalized not in {"ols", "linear", "ordinary_least_squares"}:
            codes.append("unsupported_regression_model_deferred")
    return tuple(codes)


def _adjusted_deferred_message(
    code: str,
    method: AssociationMethodSpec,
    outcome: AssociationVariableSpec,
    predictor: AssociationVariableSpec,
) -> str:
    if code == "cross_source_adjusted_association_deferred":
        return (
            f"{code}: outcome {outcome.variable_id!r}, predictor {predictor.variable_id!r}, "
            "or covariates come from different sources; general joins are not implemented."
        )
    if code == "missing_partial_covariates":
        return f"{code}: partial association method {method.method_id!r} requires at least one covariate."
    if code == "grouped_adjusted_association_deferred":
        return f"{code}: method {method.method_id!r} declares grouping behavior."
    if code == "rank_adjusted_association_deferred":
        return f"{code}: method {method.method_id!r} declares rank or partial-Spearman behavior."
    if code == "stratified_adjusted_association_deferred":
        return f"{code}: method {method.method_id!r} declares stratified behavior."
    if code == "repeated_measures_adjusted_association_deferred":
        return f"{code}: method {method.method_id!r} declares repeated-measures behavior."
    if code == "mixed_model_adjusted_association_deferred":
        return f"{code}: method {method.method_id!r} declares mixed-model behavior."
    if code == "generalized_linear_model_deferred":
        return f"{code}: method {method.method_id!r} declares generalized-linear-model behavior."
    if code == "interaction_model_deferred":
        return f"{code}: method {method.method_id!r} declares interaction behavior."
    if code == "mediation_model_deferred":
        return f"{code}: method {method.method_id!r} declares mediation behavior."
    if code == "moderation_model_deferred":
        return f"{code}: method {method.method_id!r} declares moderation behavior."
    if code == "categorical_encoding_deferred":
        return f"{code}: method {method.method_id!r} declares categorical encoding behavior."
    if code == "nonlinear_model_deferred":
        return f"{code}: method {method.method_id!r} declares non-linear behavior."
    if code == "machine_learning_model_deferred":
        return f"{code}: method {method.method_id!r} declares machine-learning behavior."
    if code == "robust_or_clustered_uncertainty_deferred":
        return f"{code}: method {method.method_id!r} declares robust or clustered uncertainty behavior."
    if code == "unsupported_regression_model_deferred":
        return f"{code}: method {method.method_id!r} declares a non-OLS regression model."
    return f"{code}: method {method.method_id!r} declares unsupported adjusted/regression behavior."


def _adjusted_pair_id(
    method: AssociationMethodSpec,
    outcome: AssociationVariableSpec,
    predictor: AssociationVariableSpec,
) -> str:
    return f"{method.method_id}::{outcome.variable_id}::{predictor.variable_id}"


def _deferred_adjusted_result_row(
    pair: AdjustedAssociationPairPlanRow,
) -> tuple[AdjustedAssociationResultRow | RegressionAssociationResultRow, AdjustedAssociationComputationQcRow]:
    result_row = _empty_adjusted_result_row(pair, status="deferred", warnings=pair.warnings, errors=pair.errors)
    qc_row = _empty_adjusted_qc_row(
        pair,
        status="deferred",
        code=pair.code,
        message="Adjusted/regression association pair is deferred.",
        warnings=pair.warnings,
        errors=pair.errors,
    )
    return result_row, qc_row


def _unloaded_source_adjusted_result_row(
    pair: AdjustedAssociationPairPlanRow,
    loaded_source: Mapping[str, Any] | None,
) -> tuple[AdjustedAssociationResultRow | RegressionAssociationResultRow, AdjustedAssociationComputationQcRow]:
    message = "adjusted_association_source_not_loaded: source rows are not available for this same-source pair."
    if loaded_source is not None:
        message = f"adjusted_association_source_not_loaded: {loaded_source['message']}"
    result_row = _empty_adjusted_result_row(pair, status="error", warnings=(), errors=(message,))
    qc_row = _empty_adjusted_qc_row(
        pair,
        status="error",
        code="adjusted_association_source_not_loaded",
        message=message,
        warnings=(),
        errors=(message,),
    )
    return result_row, qc_row


def _empty_adjusted_result_row(
    pair: AdjustedAssociationPairPlanRow,
    *,
    status: str,
    warnings: Sequence[str],
    errors: Sequence[str],
) -> AdjustedAssociationResultRow | RegressionAssociationResultRow:
    if pair.method_name == METHOD_REGRESSION:
        model_parameter_count = 2 + pair.covariate_count
        return RegressionAssociationResultRow(
            workflow_id=pair.workflow_id,
            pair_id=pair.pair_id,
            method_id=pair.method_id,
            method_kind=pair.method_kind,
            method_name=pair.method_name,
            family_id=pair.family_id,
            source_id=pair.source_id,
            outcome_id=pair.outcome_id,
            outcome_source_id=pair.outcome_source_id,
            outcome_column=pair.outcome_column,
            predictor_id=pair.predictor_id,
            predictor_source_id=pair.predictor_source_id,
            predictor_column=pair.predictor_column,
            covariate_ids=pair.covariate_ids,
            covariate_columns=pair.covariate_columns,
            covariate_count=pair.covariate_count,
            model_parameter_count=model_parameter_count,
            residual_degrees_of_freedom=0 - model_parameter_count,
            n_total=0,
            n_used=0,
            n_missing_outcome=0,
            n_missing_predictor=0,
            n_missing_covariates=0,
            n_missing_listwise=0,
            n_nonfinite=0,
            n_invalid_numeric=0,
            n_bool_numeric=0,
            statistic_name="regression_coefficient",
            statistic_value=None,
            status=status,
            warnings=warnings,
            errors=errors,
        )
    return AdjustedAssociationResultRow(
        workflow_id=pair.workflow_id,
        pair_id=pair.pair_id,
        method_id=pair.method_id,
        method_kind=pair.method_kind,
        method_name=pair.method_name,
        family_id=pair.family_id,
        source_id=pair.source_id,
        outcome_id=pair.outcome_id,
        outcome_source_id=pair.outcome_source_id,
        outcome_column=pair.outcome_column,
        predictor_id=pair.predictor_id,
        predictor_source_id=pair.predictor_source_id,
        predictor_column=pair.predictor_column,
        covariate_ids=pair.covariate_ids,
        covariate_columns=pair.covariate_columns,
        covariate_count=pair.covariate_count,
        n_total=0,
        n_used=0,
        n_missing_outcome=0,
        n_missing_predictor=0,
        n_missing_covariates=0,
        n_missing_listwise=0,
        n_nonfinite=0,
        n_invalid_numeric=0,
        n_bool_numeric=0,
        statistic_name="partial_r",
        statistic_value=None,
        status=status,
        warnings=warnings,
        errors=errors,
    )


def _empty_adjusted_qc_row(
    pair: AdjustedAssociationPairPlanRow,
    *,
    status: str,
    code: str,
    message: str,
    warnings: Sequence[str],
    errors: Sequence[str],
) -> AdjustedAssociationComputationQcRow:
    model_parameter_count = 2 + pair.covariate_count if pair.method_name == METHOD_REGRESSION else None
    residual_degrees_of_freedom = 0 - model_parameter_count if model_parameter_count is not None else None
    return AdjustedAssociationComputationQcRow(
        workflow_id=pair.workflow_id,
        pair_id=pair.pair_id,
        method_id=pair.method_id,
        method_kind=pair.method_kind,
        method_name=pair.method_name,
        source_id=pair.source_id,
        outcome_id=pair.outcome_id,
        outcome_column=pair.outcome_column,
        predictor_id=pair.predictor_id,
        predictor_column=pair.predictor_column,
        covariate_ids=pair.covariate_ids,
        covariate_columns=pair.covariate_columns,
        covariate_count=pair.covariate_count,
        n_total=0,
        n_used=0,
        n_missing_outcome=0,
        n_missing_predictor=0,
        n_missing_covariates=0,
        n_missing_listwise=0,
        n_nonfinite=0,
        n_invalid_numeric=0,
        n_bool_numeric=0,
        model_parameter_count=model_parameter_count,
        residual_degrees_of_freedom=residual_degrees_of_freedom,
        status=status,
        code=code,
        message=message,
        warnings=warnings,
        errors=errors,
    )


def _computed_adjusted_result_row(
    *,
    workflow: TabularAssociationWorkflowSpec,
    pair: AdjustedAssociationPairPlanRow,
    loaded_source: Mapping[str, Any],
) -> tuple[AdjustedAssociationResultRow | RegressionAssociationResultRow, AdjustedAssociationComputationQcRow]:
    source_rows = tuple(loaded_source["rows"])
    counts, outcome_values, predictor_values, covariate_rows = _adjusted_numeric_counts(
        source_rows,
        outcome_column=pair.outcome_column,
        predictor_column=pair.predictor_column,
        covariate_columns=pair.covariate_columns,
    )
    warnings, errors = _adjusted_exclusion_messages(workflow, counts, source_id=pair.source_id)
    observed_columns = set(loaded_source["observed_columns"])
    required_columns = (pair.outcome_column, pair.predictor_column, *pair.covariate_columns)
    missing_columns = tuple(column for column in required_columns if column not in observed_columns)
    if missing_columns:
        code = "adjusted_association_column_not_observed"
        errors.append(f"{code}: required association columns were not observed: {', '.join(missing_columns)}.")
        return _final_adjusted_rows(pair, counts, code, warnings, errors, statistic_value=None)
    if pair.method_name == METHOD_REGRESSION:
        return _computed_regression_result_row(
            pair=pair,
            counts=counts,
            outcome_values=outcome_values,
            predictor_values=predictor_values,
            covariate_rows=covariate_rows,
            warnings=warnings,
            errors=errors,
        )
    return _computed_partial_result_row(
        pair=pair,
        counts=counts,
        outcome_values=outcome_values,
        predictor_values=predictor_values,
        covariate_rows=covariate_rows,
        warnings=warnings,
        errors=errors,
    )


def _computed_partial_result_row(
    *,
    pair: AdjustedAssociationPairPlanRow,
    counts: Mapping[str, int],
    outcome_values: Sequence[float],
    predictor_values: Sequence[float],
    covariate_rows: Sequence[Sequence[float]],
    warnings: list[str],
    errors: list[str],
) -> tuple[AdjustedAssociationResultRow, AdjustedAssociationComputationQcRow]:
    code = "partial_association_computed"
    statistic_value: float | None = None
    parameter_count = 1 + pair.covariate_count
    if counts["n_used"] < 2:
        code = "too_few_valid_rows"
        errors.append(f"{code}: at least two complete finite numeric rows are required.")
    elif counts["n_used"] <= parameter_count:
        code = "underdetermined_covariate_design"
        errors.append(f"{code}: more complete rows than intercept-plus-covariate parameters are required.")
    else:
        design = _design_matrix(covariate_rows, include_predictor=None)
        outcome_residuals = _ols_residuals(design, outcome_values)
        predictor_residuals = _ols_residuals(design, predictor_values)
        if outcome_residuals is None or predictor_residuals is None:
            code = "singular_covariate_design"
            errors.append(f"{code}: covariate residualization design matrix is singular or collinear.")
        else:
            statistic_value, variance_code = _pearson_coefficient(
                outcome_residuals,
                predictor_residuals,
                zero_outcome_code="zero_residual_variance_outcome",
                zero_predictor_code="zero_residual_variance_predictor",
            )
            if variance_code is not None:
                code = variance_code
                errors.append(f"{variance_code}: partial association is undefined when one residual vector has zero variance.")
    result_row, qc_row = _final_adjusted_rows(pair, counts, code, warnings, errors, statistic_value=statistic_value)
    return result_row, qc_row


def _computed_regression_result_row(
    *,
    pair: AdjustedAssociationPairPlanRow,
    counts: Mapping[str, int],
    outcome_values: Sequence[float],
    predictor_values: Sequence[float],
    covariate_rows: Sequence[Sequence[float]],
    warnings: list[str],
    errors: list[str],
) -> tuple[RegressionAssociationResultRow, AdjustedAssociationComputationQcRow]:
    code = "regression_association_computed"
    statistic_value: float | None = None
    model_parameter_count = 2 + pair.covariate_count
    residual_degrees_of_freedom = counts["n_used"] - model_parameter_count
    if counts["n_used"] < 2:
        code = "too_few_valid_rows"
        errors.append(f"{code}: at least two complete finite numeric rows are required.")
    elif counts["n_used"] < model_parameter_count:
        code = "underdetermined_design_matrix"
        errors.append(f"{code}: complete rows are fewer than intercept, predictor, and covariate parameters.")
    else:
        outcome_variance_code = _zero_variance_code(outcome_values, "zero_variance_outcome")
        predictor_variance_code = _zero_variance_code(predictor_values, "zero_variance_predictor")
        if outcome_variance_code is not None:
            code = outcome_variance_code
            errors.append(f"{outcome_variance_code}: regression association is undefined when the outcome has zero variance.")
        elif predictor_variance_code is not None:
            code = predictor_variance_code
            errors.append(f"{predictor_variance_code}: regression association is undefined when the predictor has zero variance.")
        else:
            design = _design_matrix(covariate_rows, include_predictor=predictor_values)
            coefficients = _ols_coefficients(design, outcome_values)
            if coefficients is None:
                code = "singular_design_matrix"
                errors.append(f"{code}: regression design matrix is singular or collinear.")
            else:
                statistic_value = coefficients[1]
    result_row, qc_row = _final_adjusted_rows(
        pair,
        counts,
        code,
        warnings,
        errors,
        statistic_value=statistic_value,
        model_parameter_count=model_parameter_count,
        residual_degrees_of_freedom=residual_degrees_of_freedom,
    )
    return result_row, qc_row


def _final_adjusted_rows(
    pair: AdjustedAssociationPairPlanRow,
    counts: Mapping[str, int],
    code: str,
    warnings: Sequence[str],
    errors: Sequence[str],
    *,
    statistic_value: float | None,
    model_parameter_count: int | None = None,
    residual_degrees_of_freedom: int | None = None,
) -> tuple[AdjustedAssociationResultRow | RegressionAssociationResultRow, AdjustedAssociationComputationQcRow]:
    status = "error" if errors else ("warning" if warnings else "ok")
    message = _adjusted_qc_message(code, status, method_name=pair.method_name)
    qc_model_parameter_count = model_parameter_count
    qc_residual_degrees_of_freedom = residual_degrees_of_freedom
    if pair.method_name == METHOD_REGRESSION:
        parameter_count = model_parameter_count if model_parameter_count is not None else 2 + pair.covariate_count
        residual_df = residual_degrees_of_freedom if residual_degrees_of_freedom is not None else counts["n_used"] - parameter_count
        qc_model_parameter_count = parameter_count
        qc_residual_degrees_of_freedom = residual_df
        result_row: AdjustedAssociationResultRow | RegressionAssociationResultRow = RegressionAssociationResultRow(
            workflow_id=pair.workflow_id,
            pair_id=pair.pair_id,
            method_id=pair.method_id,
            method_kind=pair.method_kind,
            method_name=pair.method_name,
            family_id=pair.family_id,
            source_id=pair.source_id,
            outcome_id=pair.outcome_id,
            outcome_source_id=pair.outcome_source_id,
            outcome_column=pair.outcome_column,
            predictor_id=pair.predictor_id,
            predictor_source_id=pair.predictor_source_id,
            predictor_column=pair.predictor_column,
            covariate_ids=pair.covariate_ids,
            covariate_columns=pair.covariate_columns,
            covariate_count=pair.covariate_count,
            model_parameter_count=parameter_count,
            residual_degrees_of_freedom=residual_df,
            n_total=counts["n_total"],
            n_used=counts["n_used"],
            n_missing_outcome=counts["n_missing_outcome"],
            n_missing_predictor=counts["n_missing_predictor"],
            n_missing_covariates=counts["n_missing_covariates"],
            n_missing_listwise=counts["n_missing_listwise"],
            n_nonfinite=counts["n_nonfinite"],
            n_invalid_numeric=counts["n_invalid_numeric"],
            n_bool_numeric=counts["n_bool_numeric"],
            statistic_name="regression_coefficient",
            statistic_value=statistic_value,
            status=status,
            warnings=warnings,
            errors=errors,
        )
    else:
        result_row = AdjustedAssociationResultRow(
            workflow_id=pair.workflow_id,
            pair_id=pair.pair_id,
            method_id=pair.method_id,
            method_kind=pair.method_kind,
            method_name=pair.method_name,
            family_id=pair.family_id,
            source_id=pair.source_id,
            outcome_id=pair.outcome_id,
            outcome_source_id=pair.outcome_source_id,
            outcome_column=pair.outcome_column,
            predictor_id=pair.predictor_id,
            predictor_source_id=pair.predictor_source_id,
            predictor_column=pair.predictor_column,
            covariate_ids=pair.covariate_ids,
            covariate_columns=pair.covariate_columns,
            covariate_count=pair.covariate_count,
            n_total=counts["n_total"],
            n_used=counts["n_used"],
            n_missing_outcome=counts["n_missing_outcome"],
            n_missing_predictor=counts["n_missing_predictor"],
            n_missing_covariates=counts["n_missing_covariates"],
            n_missing_listwise=counts["n_missing_listwise"],
            n_nonfinite=counts["n_nonfinite"],
            n_invalid_numeric=counts["n_invalid_numeric"],
            n_bool_numeric=counts["n_bool_numeric"],
            statistic_name="partial_r",
            statistic_value=statistic_value,
            status=status,
            warnings=warnings,
            errors=errors,
        )
    qc_row = AdjustedAssociationComputationQcRow(
        workflow_id=pair.workflow_id,
        pair_id=pair.pair_id,
        method_id=pair.method_id,
        method_kind=pair.method_kind,
        method_name=pair.method_name,
        source_id=pair.source_id,
        outcome_id=pair.outcome_id,
        outcome_column=pair.outcome_column,
        predictor_id=pair.predictor_id,
        predictor_column=pair.predictor_column,
        covariate_ids=pair.covariate_ids,
        covariate_columns=pair.covariate_columns,
        covariate_count=pair.covariate_count,
        n_total=counts["n_total"],
        n_used=counts["n_used"],
        n_missing_outcome=counts["n_missing_outcome"],
        n_missing_predictor=counts["n_missing_predictor"],
        n_missing_covariates=counts["n_missing_covariates"],
        n_missing_listwise=counts["n_missing_listwise"],
        n_nonfinite=counts["n_nonfinite"],
        n_invalid_numeric=counts["n_invalid_numeric"],
        n_bool_numeric=counts["n_bool_numeric"],
        model_parameter_count=qc_model_parameter_count,
        residual_degrees_of_freedom=qc_residual_degrees_of_freedom,
        status=status,
        code=code,
        message=message,
        warnings=warnings,
        errors=errors,
    )
    return result_row, qc_row


def _adjusted_numeric_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome_column: str,
    predictor_column: str,
    covariate_columns: Sequence[str],
) -> tuple[dict[str, int], tuple[float, ...], tuple[float, ...], tuple[tuple[float, ...], ...]]:
    counts = {
        "n_total": len(rows),
        "n_used": 0,
        "n_missing_outcome": 0,
        "n_missing_predictor": 0,
        "n_missing_covariates": 0,
        "n_missing_listwise": 0,
        "n_nonfinite": 0,
        "n_invalid_numeric": 0,
        "n_bool_numeric": 0,
    }
    outcome_values: list[float] = []
    predictor_values: list[float] = []
    covariate_values: list[tuple[float, ...]] = []
    for row in rows:
        outcome_status, outcome_number = _finite_float_or_missing(row.get(outcome_column))
        predictor_status, predictor_number = _finite_float_or_missing(row.get(predictor_column))
        covariate_statuses: list[str] = []
        covariate_numbers: list[float | None] = []
        for column in covariate_columns:
            covariate_status, covariate_number = _finite_float_or_missing(row.get(column))
            covariate_statuses.append(covariate_status)
            covariate_numbers.append(covariate_number)
        statuses = (outcome_status, predictor_status, *covariate_statuses)
        if outcome_status == "missing":
            counts["n_missing_outcome"] += 1
        if predictor_status == "missing":
            counts["n_missing_predictor"] += 1
        if any(status == "missing" for status in covariate_statuses):
            counts["n_missing_covariates"] += 1
        if any(status == "missing" for status in statuses):
            counts["n_missing_listwise"] += 1
        for status in statuses:
            if status == "bool":
                counts["n_bool_numeric"] += 1
                counts["n_invalid_numeric"] += 1
            elif status == "nonfinite":
                counts["n_nonfinite"] += 1
                counts["n_invalid_numeric"] += 1
            elif status == "invalid":
                counts["n_invalid_numeric"] += 1
        if (
            outcome_status == "valid"
            and predictor_status == "valid"
            and all(status == "valid" for status in covariate_statuses)
            and outcome_number is not None
            and predictor_number is not None
            and all(number is not None for number in covariate_numbers)
        ):
            outcome_values.append(outcome_number)
            predictor_values.append(predictor_number)
            covariate_values.append(tuple(float(number) for number in covariate_numbers if number is not None))
            counts["n_used"] += 1
    return counts, tuple(outcome_values), tuple(predictor_values), tuple(covariate_values)


def _adjusted_exclusion_messages(
    workflow: TabularAssociationWorkflowSpec,
    counts: Mapping[str, int],
    *,
    source_id: str | None,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    source = next((candidate for candidate in workflow.sources if candidate.source_id == source_id), None)
    strict_numeric = source.schema.numeric_validation.policy == "strict" if source is not None else False
    missing_count = counts["n_missing_listwise"]
    if missing_count:
        strategy = workflow.missing_data_policy.strategy
        message = f"Observed {missing_count} rows with listwise missing values."
        if strategy == "error":
            errors.append(f"missing_data_policy_error: {message}")
        elif strategy in {"listwise", "pairwise", "drop_rows"}:
            warnings.append(f"missing_data_policy_deferred: {message} Listwise complete rows were used for this bounded run.")
        else:
            warnings.append(f"missing_values_allowed: {message}")

    nonfinite_count = counts["n_nonfinite"]
    if nonfinite_count:
        strategy = workflow.nonfinite_policy.strategy
        message = f"Observed {nonfinite_count} non-finite numeric tokens."
        if strategy == "error":
            errors.append(f"nonfinite_values_observed: {message}")
        elif strategy == "coerce_missing":
            warnings.append(f"nonfinite_values_coerced_missing: {message}")
        elif strategy == "drop_rows":
            warnings.append(f"nonfinite_values_dropped: {message}")
        else:
            warnings.append(f"nonfinite_values_excluded: {message}")

    invalid_count = counts["n_invalid_numeric"]
    bool_count = counts["n_bool_numeric"]
    if invalid_count:
        message = f"Observed {invalid_count} invalid numeric values."
        if strict_numeric:
            errors.append(f"numeric_values_invalid: {message}")
        else:
            warnings.append(f"numeric_values_invalid: {message}")
    if bool_count:
        message = f"Observed {bool_count} boolean values in numeric association columns."
        if strict_numeric:
            errors.append(f"bool_numeric_values_invalid: {message}")
        else:
            warnings.append(f"bool_numeric_values_invalid: {message}")
    return warnings, errors


def _design_matrix(
    covariate_rows: Sequence[Sequence[float]],
    *,
    include_predictor: Sequence[float] | None,
) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for index, covariates in enumerate(covariate_rows):
        values: list[float] = [1.0]
        if include_predictor is not None:
            values.append(float(include_predictor[index]))
        values.extend(float(value) for value in covariates)
        rows.append(tuple(values))
    return tuple(rows)


def _ols_residuals(
    design: Sequence[Sequence[float]],
    response: Sequence[float],
) -> tuple[float, ...] | None:
    coefficients = _ols_coefficients(design, response)
    if coefficients is None:
        return None
    residuals: list[float] = []
    for row, value in zip(design, response):
        fitted = sum(coefficient * predictor for coefficient, predictor in zip(coefficients, row))
        residuals.append(value - fitted)
    return tuple(residuals)


def _ols_coefficients(
    design: Sequence[Sequence[float]],
    response: Sequence[float],
) -> tuple[float, ...] | None:
    if not design:
        return None
    column_count = len(design[0])
    if column_count == 0 or any(len(row) != column_count for row in design):
        return None
    xtx = [[0.0 for _ in range(column_count)] for _ in range(column_count)]
    xty = [0.0 for _ in range(column_count)]
    for row, value in zip(design, response):
        for column_index in range(column_count):
            xty[column_index] += row[column_index] * value
            for other_index in range(column_count):
                xtx[column_index][other_index] += row[column_index] * row[other_index]
    return _solve_linear_system(xtx, xty)


def _solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, ...] | None:
    size = len(vector)
    if size == 0 or len(matrix) != size or any(len(row) != size for row in matrix):
        return None
    augmented = [list(row) + [float(vector[index])] for index, row in enumerate(matrix)]
    tolerance = 1e-12
    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
        pivot_value = augmented[pivot_row][pivot_index]
        if abs(pivot_value) <= tolerance:
            return None
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot_value = augmented[pivot_index][pivot_index]
        for column_index in range(pivot_index, size + 1):
            augmented[pivot_index][column_index] /= pivot_value
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0.0:
                continue
            for column_index in range(pivot_index, size + 1):
                augmented[row_index][column_index] -= factor * augmented[pivot_index][column_index]
    return tuple(augmented[index][size] for index in range(size))


def _zero_variance_code(values: Sequence[float], code: str) -> str | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return code if sum((value - mean) * (value - mean) for value in values) == 0.0 else None


def _adjusted_qc_message(code: str, status: str, *, method_name: str) -> str:
    label = "Regression association" if method_name == METHOD_REGRESSION else "Partial association"
    if status == "ok":
        return f"{label} was computed from complete finite numeric rows."
    if status == "warning":
        return f"{label} was computed with QC warnings: {code}."
    if code == "too_few_valid_rows":
        return f"{label} could not be computed because too few complete finite numeric rows were available."
    if code in {"underdetermined_covariate_design", "underdetermined_design_matrix"}:
        return f"{label} could not be computed because the design matrix is underdetermined."
    if code in {"singular_covariate_design", "singular_design_matrix"}:
        return f"{label} could not be computed because the design matrix is singular."
    if code.startswith("zero_"):
        return f"{label} could not be computed because a required variable has zero variance."
    if code == "adjusted_association_column_not_observed":
        return f"{label} could not be computed because an association column was not observed."
    return f"{label} computation reported {status}: {code}."


def _adjusted_method_summary_rows(
    workflow: TabularAssociationWorkflowSpec,
    *,
    pair_plan_rows: Sequence[AdjustedAssociationPairPlanRow],
    result_rows: Sequence[AdjustedAssociationResultRow | RegressionAssociationResultRow],
    executed: bool,
    plan_only: bool,
) -> tuple[AdjustedAssociationMethodSummaryRow, ...]:
    result_rows_by_method: dict[str, list[AdjustedAssociationResultRow | RegressionAssociationResultRow]] = {}
    for row in result_rows:
        result_rows_by_method.setdefault(row.method_id, []).append(row)
    pair_rows_by_method: dict[str, list[AdjustedAssociationPairPlanRow]] = {}
    for row in pair_plan_rows:
        pair_rows_by_method.setdefault(row.method_id, []).append(row)

    outcome_by_id = {outcome.variable_id: outcome for outcome in workflow.outcomes}
    predictor_by_id = {predictor.variable_id: predictor for predictor in workflow.predictors}
    covariate_by_id = {covariate.variable_id: covariate for covariate in workflow.covariates}
    rows: list[AdjustedAssociationMethodSummaryRow] = []
    for method in _adjusted_methods(workflow):
        pair_rows = tuple(pair_rows_by_method.get(method.method_id, ()))
        method_result_rows = tuple(result_rows_by_method.get(method.method_id, ()))
        warnings: list[str] = []
        errors: list[str] = []
        for row in pair_rows:
            warnings.extend(row.warnings)
            errors.extend(row.errors)
        for row in method_result_rows:
            warnings.extend(row.warnings)
            errors.extend(row.errors)
        status = "error" if errors else ("warning" if warnings else "ok")
        outcomes = _resolved_method_variables(method.outcome_ids, tuple(workflow.outcomes), outcome_by_id)
        predictors = _resolved_method_variables(method.predictor_ids, tuple(workflow.predictors), predictor_by_id)
        covariates = _adjusted_method_covariates(method, workflow, covariate_by_id)
        rows.append(
            AdjustedAssociationMethodSummaryRow(
                workflow_id=workflow.workflow_id,
                method_id=method.method_id,
                method_kind=_adjusted_method_kind(method),
                method_name=method.method_name,
                family_id=method.family_id,
                outcome_count=len(outcomes),
                predictor_count=len(predictors),
                covariate_count=len(covariates),
                pair_count=len(pair_rows),
                executable_pair_count=sum(1 for row in pair_rows if row.executable),
                deferred_pair_count=sum(1 for row in pair_rows if row.deferred),
                result_row_count=len(method_result_rows),
                status=status,
                warnings=_unique_texts(warnings),
                errors=_unique_texts(errors),
                executed=executed,
                plan_only=plan_only,
                will_write=False,
                output_written=False,
            )
        )
    return tuple(rows)


def _adjusted_messages(
    *,
    workflow_validation_rows: Sequence[AssociationValidationRow],
    pair_plan_rows: Sequence[AdjustedAssociationPairPlanRow],
    source_load_rows: Sequence[TabularSourceLoadRow],
    input_qc_summary_rows: Sequence[AssociationInputQcSummaryRow],
    computation_qc_rows: Sequence[AdjustedAssociationComputationQcRow],
    result_rows: Sequence[AdjustedAssociationResultRow | RegressionAssociationResultRow],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    errors: list[str] = []

    def add_status(status: str, message: str) -> None:
        if not message:
            return
        if status == "error":
            errors.append(message)
        elif status == "warning":
            warnings.append(message)

    for row in workflow_validation_rows:
        add_status(row.status, row.message)
    for row in pair_plan_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in source_load_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in input_qc_summary_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in computation_qc_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
        if row.status in {"error", "warning"}:
            add_status(row.status, row.message)
    for row in result_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    return _unique_texts(warnings), _unique_texts(errors)


def _adjusted_provenance_rows(
    workflow: TabularAssociationWorkflowSpec,
    *,
    executed: bool,
    plan_only: bool,
    source_count: int,
    loaded_source_count: int,
    method_count: int,
    plan_row_count: int,
    result_row_count: int,
    qc_mode: str,
) -> tuple[TabularAssociationAdjustedProvenanceRow, ...]:
    return (
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="schema_version", value=SCHEMA_VERSION),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="workflow_id", value=workflow.workflow_id),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="requested_backend", value=workflow.backend),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="runtime_backend", value=RUNTIME_BACKEND_RECORDS),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="source_count", value=source_count),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="loaded_source_count", value=loaded_source_count),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="method_count", value=method_count),
        TabularAssociationAdjustedProvenanceRow(
            workflow_id=workflow.workflow_id,
            key="adjusted_regression_method_count",
            value=_adjusted_method_count(workflow),
        ),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="plan_row_count", value=plan_row_count),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="result_row_count", value=result_row_count),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="qc_mode", value=qc_mode),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="executed", value=executed),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="plan_only", value=plan_only),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="will_write", value=False),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="output_written", value=False),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="output_paths_written", value=()),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow.workflow_id, key="no_output_paths_written", value=True),
    )


def _adjusted_provenance_rows_for_error(
    workflow_id: str,
    *,
    executed: bool,
    plan_only: bool,
    qc_mode: str,
) -> tuple[TabularAssociationAdjustedProvenanceRow, ...]:
    return (
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="schema_version", value=SCHEMA_VERSION),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="workflow_id", value=workflow_id),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="runtime_backend", value=RUNTIME_BACKEND_RECORDS),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="source_count", value=0),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="loaded_source_count", value=0),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="method_count", value=0),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="adjusted_regression_method_count", value=0),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="plan_row_count", value=0),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="result_row_count", value=0),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="qc_mode", value=qc_mode),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="executed", value=executed),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="plan_only", value=plan_only),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="will_write", value=False),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="output_written", value=False),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="output_paths_written", value=()),
        TabularAssociationAdjustedProvenanceRow(workflow_id=workflow_id, key="no_output_paths_written", value=True),
    )


def _correlation_methods(workflow: TabularAssociationWorkflowSpec) -> tuple[AssociationMethodSpec, ...]:
    return tuple(method for method in workflow.methods if method.method_name in {METHOD_PEARSON, METHOD_SPEARMAN})


def _correlation_method_count(workflow: TabularAssociationWorkflowSpec) -> int:
    return len(_correlation_methods(workflow))


def _correlation_pair_plan_rows(workflow: TabularAssociationWorkflowSpec) -> tuple[AssociationPairPlanRow, ...]:
    outcome_by_id = {outcome.variable_id: outcome for outcome in workflow.outcomes}
    predictor_by_id = {predictor.variable_id: predictor for predictor in workflow.predictors}
    rows: list[AssociationPairPlanRow] = []
    for method in _correlation_methods(workflow):
        outcomes = _resolved_method_variables(method.outcome_ids, tuple(workflow.outcomes), outcome_by_id)
        predictors = _resolved_method_variables(method.predictor_ids, tuple(workflow.predictors), predictor_by_id)
        method_deferred_codes = _correlation_method_deferred_codes(method)
        for outcome in outcomes:
            for predictor in predictors:
                deferred_codes: list[str] = []
                if outcome.source_id != predictor.source_id:
                    deferred_codes.append("cross_source_correlation_deferred")
                deferred_codes.extend(method_deferred_codes)
                deferred = bool(deferred_codes)
                code = deferred_codes[0] if deferred else "correlation_pair_planned"
                warnings = tuple(_correlation_deferred_message(code, method, outcome, predictor) for code in deferred_codes)
                source_id = outcome.source_id if outcome.source_id == predictor.source_id else None
                rows.append(
                    AssociationPairPlanRow(
                        workflow_id=workflow.workflow_id,
                        pair_id=_correlation_pair_id(method, outcome, predictor),
                        method_id=method.method_id,
                        method_kind="correlation",
                        method_name=method.method_name,
                        family_id=method.family_id,
                        source_id=source_id,
                        outcome_id=outcome.variable_id,
                        outcome_source_id=outcome.source_id,
                        outcome_column=outcome.column_name,
                        predictor_id=predictor.variable_id,
                        predictor_source_id=predictor.source_id,
                        predictor_column=predictor.column_name,
                        executable=not deferred,
                        deferred=deferred,
                        status="deferred" if deferred else "planned",
                        code=code,
                        warnings=warnings,
                        errors=(),
                    )
                )
    return tuple(rows)


def _resolved_method_variables(
    requested_ids: Sequence[str],
    default_variables: Sequence[AssociationVariableSpec],
    variable_by_id: Mapping[str, AssociationVariableSpec],
) -> tuple[AssociationVariableSpec, ...]:
    if not requested_ids:
        return tuple(default_variables)
    return tuple(variable_by_id[variable_id] for variable_id in requested_ids if variable_id in variable_by_id)


def _correlation_method_deferred_codes(method: AssociationMethodSpec) -> tuple[str, ...]:
    codes: list[str] = []
    if method.covariate_ids:
        codes.append("covariate_adjusted_correlation_deferred")
    if method.grouping_ids:
        codes.append("grouped_correlation_deferred")
    metadata_checks = (
        ("partial_correlation_deferred", ("partial", "partial_correlation")),
        ("adjusted_correlation_deferred", ("adjusted", "adjustment", "adjustment_ids")),
        ("covariate_adjusted_correlation_deferred", ("covariates", "covariate_ids")),
        ("grouped_correlation_deferred", ("grouping", "groupings", "group_by", "grouping_ids")),
        ("stratified_correlation_deferred", ("stratification", "stratify", "strata", "stratification_ids")),
        ("repeated_measures_correlation_deferred", ("repeated", "repeated_measures", "repeated_measures_ids")),
        ("mixed_model_correlation_deferred", ("mixed", "mixed_model", "random_effects")),
    )
    for code, keys in metadata_checks:
        if code in codes:
            continue
        if any(_metadata_truthy(method.metadata.get(key)) for key in keys):
            codes.append(code)
    return tuple(codes)


def _metadata_truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    if isinstance(value, (Sequence, Mapping)) and not isinstance(value, (str, bytes)):
        return bool(value)
    return bool(value)


def _correlation_deferred_message(
    code: str,
    method: AssociationMethodSpec,
    outcome: AssociationVariableSpec,
    predictor: AssociationVariableSpec,
) -> str:
    if code == "cross_source_correlation_deferred":
        return (
            f"{code}: outcome {outcome.variable_id!r} and predictor {predictor.variable_id!r} "
            "come from different sources; general joins are not implemented."
        )
    if code == "covariate_adjusted_correlation_deferred":
        return f"{code}: method {method.method_id!r} declares covariate-adjusted behavior."
    if code == "grouped_correlation_deferred":
        return f"{code}: method {method.method_id!r} declares grouping behavior."
    if code == "stratified_correlation_deferred":
        return f"{code}: method {method.method_id!r} declares stratified behavior."
    if code == "partial_correlation_deferred":
        return f"{code}: method {method.method_id!r} declares partial-correlation behavior."
    if code == "repeated_measures_correlation_deferred":
        return f"{code}: method {method.method_id!r} declares repeated-measures behavior."
    if code == "mixed_model_correlation_deferred":
        return f"{code}: method {method.method_id!r} declares mixed-model behavior."
    return f"{code}: method {method.method_id!r} declares unsupported correlation behavior."


def _correlation_pair_id(
    method: AssociationMethodSpec,
    outcome: AssociationVariableSpec,
    predictor: AssociationVariableSpec,
) -> str:
    return f"{method.method_id}::{outcome.variable_id}::{predictor.variable_id}"


def _association_input_qc_summary_row(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    loaded_source: Mapping[str, Any],
) -> AssociationInputQcSummaryRow:
    missingness_rows: tuple[TabularMissingnessRow, ...] = ()
    duplicate_rows: tuple[TabularDuplicateRow, ...] = ()
    nonfinite_rows: tuple[TabularNonFiniteRow, ...] = ()
    numeric_qc_rows: tuple[TabularNumericQcRow, ...] = ()
    if loaded_source["load_status"] in {"loaded", "empty"}:
        missingness_rows = _missingness_rows_for_source(workflow, source, loaded_source)
        duplicate_rows = _duplicate_rows_for_source(workflow, source, loaded_source)
        nonfinite_rows = _nonfinite_rows_for_source(workflow, source, loaded_source)
        numeric_qc_rows = _numeric_qc_rows_for_source(workflow, source, loaded_source)

    warnings = list(loaded_source["warnings"])
    errors = list(loaded_source["errors"])
    for row in (*missingness_rows, *duplicate_rows, *nonfinite_rows, *numeric_qc_rows):
        if row.status == "error":
            errors.append(row.message)
        elif row.status == "warning":
            warnings.append(row.message)
    status = "error" if errors else ("warning" if warnings else "ok")
    return AssociationInputQcSummaryRow(
        workflow_id=workflow.workflow_id,
        source_id=source.source_id,
        requested_backend=source.backend,
        runtime_backend=RUNTIME_BACKEND_RECORDS,
        source_kind=str(loaded_source["source_kind"]),
        load_status=str(loaded_source["load_status"]),
        row_count=len(loaded_source["rows"]),
        observed_column_count=len(loaded_source["observed_columns"]),
        missingness_error_count=sum(1 for row in missingness_rows if row.status == "error"),
        missingness_warning_count=sum(1 for row in missingness_rows if row.status == "warning"),
        numeric_error_count=sum(1 for row in numeric_qc_rows if row.status == "error"),
        numeric_warning_count=sum(1 for row in numeric_qc_rows if row.status == "warning"),
        nonfinite_error_count=sum(1 for row in nonfinite_rows if row.status == "error"),
        nonfinite_warning_count=sum(1 for row in nonfinite_rows if row.status == "warning"),
        duplicate_error_count=sum(1 for row in duplicate_rows if row.status == "error"),
        duplicate_warning_count=sum(1 for row in duplicate_rows if row.status == "warning"),
        status=status,
        warnings=_unique_texts(warnings),
        errors=_unique_texts(errors),
    )


def _deferred_correlation_result_row(
    pair: AssociationPairPlanRow,
) -> tuple[CorrelationAssociationResultRow, CorrelationComputationQcRow]:
    statistic_name = _correlation_statistic_name(pair.method_name)
    result_row = CorrelationAssociationResultRow(
        workflow_id=pair.workflow_id,
        pair_id=pair.pair_id,
        method_id=pair.method_id,
        method_kind=pair.method_kind,
        method_name=pair.method_name,
        correlation_method=pair.method_name,
        family_id=pair.family_id,
        source_id=pair.source_id,
        outcome_id=pair.outcome_id,
        outcome_source_id=pair.outcome_source_id,
        outcome_column=pair.outcome_column,
        predictor_id=pair.predictor_id,
        predictor_source_id=pair.predictor_source_id,
        predictor_column=pair.predictor_column,
        n_total=0,
        n_used=0,
        n_missing_outcome=0,
        n_missing_predictor=0,
        n_missing_pairwise=0,
        n_nonfinite=0,
        n_invalid_numeric=0,
        n_bool_numeric=0,
        statistic_name=statistic_name,
        statistic_value=None,
        status="deferred",
        warnings=pair.warnings,
        errors=pair.errors,
    )
    qc_row = CorrelationComputationQcRow(
        workflow_id=pair.workflow_id,
        pair_id=pair.pair_id,
        method_id=pair.method_id,
        method_name=pair.method_name,
        source_id=pair.source_id,
        outcome_id=pair.outcome_id,
        outcome_column=pair.outcome_column,
        predictor_id=pair.predictor_id,
        predictor_column=pair.predictor_column,
        n_total=0,
        n_used=0,
        n_missing_outcome=0,
        n_missing_predictor=0,
        n_missing_pairwise=0,
        n_nonfinite=0,
        n_invalid_numeric=0,
        n_bool_numeric=0,
        status="deferred",
        code=pair.code,
        message="Correlation pair is deferred.",
        warnings=pair.warnings,
        errors=pair.errors,
    )
    return result_row, qc_row


def _unloaded_source_correlation_result_row(
    pair: AssociationPairPlanRow,
    loaded_source: Mapping[str, Any] | None,
) -> tuple[CorrelationAssociationResultRow, CorrelationComputationQcRow]:
    statistic_name = _correlation_statistic_name(pair.method_name)
    message = "correlation_source_not_loaded: source rows are not available for this same-source pair."
    if loaded_source is not None:
        message = f"correlation_source_not_loaded: {loaded_source['message']}"
    result_row = CorrelationAssociationResultRow(
        workflow_id=pair.workflow_id,
        pair_id=pair.pair_id,
        method_id=pair.method_id,
        method_kind=pair.method_kind,
        method_name=pair.method_name,
        correlation_method=pair.method_name,
        family_id=pair.family_id,
        source_id=pair.source_id,
        outcome_id=pair.outcome_id,
        outcome_source_id=pair.outcome_source_id,
        outcome_column=pair.outcome_column,
        predictor_id=pair.predictor_id,
        predictor_source_id=pair.predictor_source_id,
        predictor_column=pair.predictor_column,
        n_total=0,
        n_used=0,
        n_missing_outcome=0,
        n_missing_predictor=0,
        n_missing_pairwise=0,
        n_nonfinite=0,
        n_invalid_numeric=0,
        n_bool_numeric=0,
        statistic_name=statistic_name,
        statistic_value=None,
        status="error",
        warnings=(),
        errors=(message,),
    )
    qc_row = CorrelationComputationQcRow(
        workflow_id=pair.workflow_id,
        pair_id=pair.pair_id,
        method_id=pair.method_id,
        method_name=pair.method_name,
        source_id=pair.source_id,
        outcome_id=pair.outcome_id,
        outcome_column=pair.outcome_column,
        predictor_id=pair.predictor_id,
        predictor_column=pair.predictor_column,
        n_total=0,
        n_used=0,
        n_missing_outcome=0,
        n_missing_predictor=0,
        n_missing_pairwise=0,
        n_nonfinite=0,
        n_invalid_numeric=0,
        n_bool_numeric=0,
        status="error",
        code="correlation_source_not_loaded",
        message=message,
        warnings=(),
        errors=(message,),
    )
    return result_row, qc_row


def _computed_correlation_result_row(
    *,
    workflow: TabularAssociationWorkflowSpec,
    pair: AssociationPairPlanRow,
    loaded_source: Mapping[str, Any],
) -> tuple[CorrelationAssociationResultRow, CorrelationComputationQcRow]:
    source_rows = tuple(loaded_source["rows"])
    counts, outcome_values, predictor_values = _correlation_numeric_pair_counts(
        source_rows,
        outcome_column=pair.outcome_column,
        predictor_column=pair.predictor_column,
    )
    warnings, errors = _correlation_exclusion_messages(workflow, counts, source_id=pair.source_id)
    code = "correlation_computed"
    statistic_name = _correlation_statistic_name(pair.method_name)
    statistic_value: float | None = None
    tie_count_outcome = 0
    tie_count_predictor = 0
    tie_group_count_outcome = 0
    tie_group_count_predictor = 0

    observed_columns = set(loaded_source["observed_columns"])
    missing_columns = tuple(
        column for column in (pair.outcome_column, pair.predictor_column) if column not in observed_columns
    )
    if missing_columns:
        code = "correlation_column_not_observed"
        errors.append(f"{code}: required association columns were not observed: {', '.join(missing_columns)}.")
    elif counts["n_used"] < 2:
        code = "too_few_valid_pairs"
        errors.append(f"{code}: at least two complete finite numeric pairs are required.")
    else:
        if pair.method_name == METHOD_SPEARMAN:
            outcome_ranks, tie_count_outcome, tie_group_count_outcome = _average_ranks(outcome_values)
            predictor_ranks, tie_count_predictor, tie_group_count_predictor = _average_ranks(predictor_values)
            statistic_value, variance_code = _pearson_coefficient(
                outcome_ranks,
                predictor_ranks,
                zero_outcome_code="zero_variance_outcome_ranks",
                zero_predictor_code="zero_variance_predictor_ranks",
            )
        else:
            statistic_value, variance_code = _pearson_coefficient(
                outcome_values,
                predictor_values,
                zero_outcome_code="zero_variance_outcome",
                zero_predictor_code="zero_variance_predictor",
            )
        if variance_code is not None:
            code = variance_code
            errors.append(f"{variance_code}: correlation is undefined when one variable has zero variance.")

    status = "error" if errors else ("warning" if warnings else "ok")
    message = _correlation_qc_message(code, status)
    result_row = CorrelationAssociationResultRow(
        workflow_id=pair.workflow_id,
        pair_id=pair.pair_id,
        method_id=pair.method_id,
        method_kind=pair.method_kind,
        method_name=pair.method_name,
        correlation_method=pair.method_name,
        family_id=pair.family_id,
        source_id=pair.source_id,
        outcome_id=pair.outcome_id,
        outcome_source_id=pair.outcome_source_id,
        outcome_column=pair.outcome_column,
        predictor_id=pair.predictor_id,
        predictor_source_id=pair.predictor_source_id,
        predictor_column=pair.predictor_column,
        n_total=counts["n_total"],
        n_used=counts["n_used"],
        n_missing_outcome=counts["n_missing_outcome"],
        n_missing_predictor=counts["n_missing_predictor"],
        n_missing_pairwise=counts["n_missing_pairwise"],
        n_nonfinite=counts["n_nonfinite"],
        n_invalid_numeric=counts["n_invalid_numeric"],
        n_bool_numeric=counts["n_bool_numeric"],
        statistic_name=statistic_name,
        statistic_value=statistic_value,
        tie_count_outcome=tie_count_outcome,
        tie_count_predictor=tie_count_predictor,
        tie_group_count_outcome=tie_group_count_outcome,
        tie_group_count_predictor=tie_group_count_predictor,
        status=status,
        warnings=warnings,
        errors=errors,
    )
    qc_row = CorrelationComputationQcRow(
        workflow_id=pair.workflow_id,
        pair_id=pair.pair_id,
        method_id=pair.method_id,
        method_name=pair.method_name,
        source_id=pair.source_id,
        outcome_id=pair.outcome_id,
        outcome_column=pair.outcome_column,
        predictor_id=pair.predictor_id,
        predictor_column=pair.predictor_column,
        n_total=counts["n_total"],
        n_used=counts["n_used"],
        n_missing_outcome=counts["n_missing_outcome"],
        n_missing_predictor=counts["n_missing_predictor"],
        n_missing_pairwise=counts["n_missing_pairwise"],
        n_nonfinite=counts["n_nonfinite"],
        n_invalid_numeric=counts["n_invalid_numeric"],
        n_bool_numeric=counts["n_bool_numeric"],
        tie_count_outcome=tie_count_outcome,
        tie_count_predictor=tie_count_predictor,
        tie_group_count_outcome=tie_group_count_outcome,
        tie_group_count_predictor=tie_group_count_predictor,
        status=status,
        code=code,
        message=message,
        warnings=warnings,
        errors=errors,
    )
    return result_row, qc_row


def _correlation_numeric_pair_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome_column: str,
    predictor_column: str,
) -> tuple[dict[str, int], tuple[float, ...], tuple[float, ...]]:
    counts = {
        "n_total": len(rows),
        "n_used": 0,
        "n_missing_outcome": 0,
        "n_missing_predictor": 0,
        "n_missing_pairwise": 0,
        "n_nonfinite": 0,
        "n_invalid_numeric": 0,
        "n_bool_numeric": 0,
    }
    outcome_values: list[float] = []
    predictor_values: list[float] = []
    for row in rows:
        outcome_status, outcome_number = _finite_float_or_missing(row.get(outcome_column))
        predictor_status, predictor_number = _finite_float_or_missing(row.get(predictor_column))
        if outcome_status == "missing":
            counts["n_missing_outcome"] += 1
        if predictor_status == "missing":
            counts["n_missing_predictor"] += 1
        if outcome_status == "missing" or predictor_status == "missing":
            counts["n_missing_pairwise"] += 1
        for status in (outcome_status, predictor_status):
            if status == "bool":
                counts["n_bool_numeric"] += 1
                counts["n_invalid_numeric"] += 1
            elif status == "nonfinite":
                counts["n_nonfinite"] += 1
                counts["n_invalid_numeric"] += 1
            elif status == "invalid":
                counts["n_invalid_numeric"] += 1
        if outcome_status == "valid" and predictor_status == "valid" and outcome_number is not None and predictor_number is not None:
            outcome_values.append(outcome_number)
            predictor_values.append(predictor_number)
            counts["n_used"] += 1
    return counts, tuple(outcome_values), tuple(predictor_values)


def _finite_float_or_missing(value: Any) -> tuple[str, float | None]:
    if _is_missing_value(value):
        return "missing", None
    return _finite_float(value)


def _correlation_exclusion_messages(
    workflow: TabularAssociationWorkflowSpec,
    counts: Mapping[str, int],
    *,
    source_id: str | None,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    source = next((candidate for candidate in workflow.sources if candidate.source_id == source_id), None)
    strict_numeric = source.schema.numeric_validation.policy == "strict" if source is not None else False
    missing_count = counts["n_missing_pairwise"]
    if missing_count:
        strategy = workflow.missing_data_policy.strategy
        message = f"Observed {missing_count} rows with pairwise missing values."
        if strategy == "error":
            errors.append(f"missing_data_policy_error: {message}")
        elif strategy == "pairwise":
            warnings.append(f"pairwise_missing_values_dropped: {message}")
        elif strategy in {"listwise", "drop_rows"}:
            warnings.append(f"missing_data_policy_deferred: {message} Pairwise complete rows were used for this bounded run.")
        else:
            warnings.append(f"missing_values_allowed: {message}")

    nonfinite_count = counts["n_nonfinite"]
    if nonfinite_count:
        strategy = workflow.nonfinite_policy.strategy
        message = f"Observed {nonfinite_count} non-finite numeric tokens."
        if strategy == "error":
            errors.append(f"nonfinite_values_observed: {message}")
        elif strategy == "coerce_missing":
            warnings.append(f"nonfinite_values_coerced_missing: {message}")
        elif strategy == "drop_rows":
            warnings.append(f"nonfinite_values_dropped: {message}")
        else:
            warnings.append(f"nonfinite_values_excluded: {message}")

    invalid_count = counts["n_invalid_numeric"]
    bool_count = counts["n_bool_numeric"]
    if invalid_count:
        message = f"Observed {invalid_count} invalid numeric values."
        if strict_numeric:
            errors.append(f"numeric_values_invalid: {message}")
        else:
            warnings.append(f"numeric_values_invalid: {message}")
    if bool_count:
        message = f"Observed {bool_count} boolean values in numeric association columns."
        if strict_numeric:
            errors.append(f"bool_numeric_values_invalid: {message}")
        else:
            warnings.append(f"bool_numeric_values_invalid: {message}")
    return warnings, errors


def _pearson_coefficient(
    outcome_values: Sequence[float],
    predictor_values: Sequence[float],
    *,
    zero_outcome_code: str,
    zero_predictor_code: str,
) -> tuple[float | None, str | None]:
    n = len(outcome_values)
    outcome_mean = sum(outcome_values) / n
    predictor_mean = sum(predictor_values) / n
    outcome_centered = tuple(value - outcome_mean for value in outcome_values)
    predictor_centered = tuple(value - predictor_mean for value in predictor_values)
    outcome_ss = sum(value * value for value in outcome_centered)
    predictor_ss = sum(value * value for value in predictor_centered)
    if outcome_ss == 0.0:
        return None, zero_outcome_code
    if predictor_ss == 0.0:
        return None, zero_predictor_code
    coefficient = sum(outcome * predictor for outcome, predictor in zip(outcome_centered, predictor_centered)) / math.sqrt(
        outcome_ss * predictor_ss
    )
    if coefficient > 1.0 and coefficient <= 1.0 + 1e-12:
        coefficient = 1.0
    elif coefficient < -1.0 and coefficient >= -1.0 - 1e-12:
        coefficient = -1.0
    return coefficient, None


def _average_ranks(values: Sequence[float]) -> tuple[tuple[float, ...], int, int]:
    indexed_values = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    tie_count = 0
    tie_group_count = 0
    index = 0
    while index < len(indexed_values):
        end = index + 1
        while end < len(indexed_values) and indexed_values[end][1] == indexed_values[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for original_index, _value in indexed_values[index:end]:
            ranks[original_index] = average_rank
        group_size = end - index
        if group_size > 1:
            tie_count += group_size
            tie_group_count += 1
        index = end
    return tuple(ranks), tie_count, tie_group_count


def _correlation_statistic_name(method_name: str) -> str:
    return "rho" if method_name == METHOD_SPEARMAN else "r"


def _correlation_qc_message(code: str, status: str) -> str:
    if status == "ok":
        return "Correlation was computed from complete finite numeric pairs."
    if status == "warning":
        return f"Correlation was computed with QC warnings: {code}."
    if code == "too_few_valid_pairs":
        return "Correlation could not be computed because too few complete finite numeric pairs were available."
    if code.startswith("zero_variance"):
        return "Correlation could not be computed because one variable has zero variance."
    if code == "correlation_column_not_observed":
        return "Correlation could not be computed because an association column was not observed."
    return f"Correlation computation reported {status}: {code}."


def _correlation_method_summary_rows(
    workflow: TabularAssociationWorkflowSpec,
    *,
    pair_plan_rows: Sequence[AssociationPairPlanRow],
    result_rows: Sequence[CorrelationAssociationResultRow],
    executed: bool,
    plan_only: bool,
) -> tuple[CorrelationMethodSummaryRow, ...]:
    result_rows_by_method: dict[str, list[CorrelationAssociationResultRow]] = {}
    for row in result_rows:
        result_rows_by_method.setdefault(row.method_id, []).append(row)
    pair_rows_by_method: dict[str, list[AssociationPairPlanRow]] = {}
    for row in pair_plan_rows:
        pair_rows_by_method.setdefault(row.method_id, []).append(row)

    rows: list[CorrelationMethodSummaryRow] = []
    for method in _correlation_methods(workflow):
        pair_rows = tuple(pair_rows_by_method.get(method.method_id, ()))
        method_result_rows = tuple(result_rows_by_method.get(method.method_id, ()))
        warnings: list[str] = []
        errors: list[str] = []
        for row in pair_rows:
            warnings.extend(row.warnings)
            errors.extend(row.errors)
        for row in method_result_rows:
            warnings.extend(row.warnings)
            errors.extend(row.errors)
        status = "error" if errors else ("warning" if warnings else "ok")
        outcomes = _resolved_method_variables(
            method.outcome_ids,
            tuple(workflow.outcomes),
            {outcome.variable_id: outcome for outcome in workflow.outcomes},
        )
        predictors = _resolved_method_variables(
            method.predictor_ids,
            tuple(workflow.predictors),
            {predictor.variable_id: predictor for predictor in workflow.predictors},
        )
        rows.append(
            CorrelationMethodSummaryRow(
                workflow_id=workflow.workflow_id,
                method_id=method.method_id,
                method_kind="correlation",
                method_name=method.method_name,
                family_id=method.family_id,
                outcome_count=len(outcomes),
                predictor_count=len(predictors),
                pair_count=len(pair_rows),
                executable_pair_count=sum(1 for row in pair_rows if row.executable),
                deferred_pair_count=sum(1 for row in pair_rows if row.deferred),
                result_row_count=len(method_result_rows),
                status=status,
                warnings=_unique_texts(warnings),
                errors=_unique_texts(errors),
                executed=executed,
                plan_only=plan_only,
                will_write=False,
                output_written=False,
            )
        )
    return tuple(rows)


def _correlation_messages(
    *,
    workflow_validation_rows: Sequence[AssociationValidationRow],
    pair_plan_rows: Sequence[AssociationPairPlanRow],
    source_load_rows: Sequence[TabularSourceLoadRow],
    input_qc_summary_rows: Sequence[AssociationInputQcSummaryRow],
    computation_qc_rows: Sequence[CorrelationComputationQcRow],
    result_rows: Sequence[CorrelationAssociationResultRow],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    errors: list[str] = []

    def add_status(status: str, message: str) -> None:
        if not message:
            return
        if status == "error":
            errors.append(message)
        elif status == "warning":
            warnings.append(message)

    for row in workflow_validation_rows:
        add_status(row.status, row.message)
    for row in pair_plan_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in source_load_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in input_qc_summary_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in computation_qc_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
        if row.status in {"error", "warning"}:
            add_status(row.status, row.message)
    for row in result_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    return _unique_texts(warnings), _unique_texts(errors)


def _correlation_provenance_rows(
    workflow: TabularAssociationWorkflowSpec,
    *,
    executed: bool,
    plan_only: bool,
    source_count: int,
    loaded_source_count: int,
    method_count: int,
    pair_count: int,
    result_row_count: int,
    qc_mode: str,
) -> tuple[TabularAssociationCorrelationProvenanceRow, ...]:
    return (
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="schema_version", value=SCHEMA_VERSION),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="workflow_id", value=workflow.workflow_id),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="requested_backend", value=workflow.backend),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="runtime_backend", value=RUNTIME_BACKEND_RECORDS),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="source_count", value=source_count),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="loaded_source_count", value=loaded_source_count),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="method_count", value=method_count),
        TabularAssociationCorrelationProvenanceRow(
            workflow_id=workflow.workflow_id,
            key="correlation_method_count",
            value=_correlation_method_count(workflow),
        ),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="pair_count", value=pair_count),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="result_row_count", value=result_row_count),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="qc_mode", value=qc_mode),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="executed", value=executed),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="plan_only", value=plan_only),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="will_write", value=False),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="output_written", value=False),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="output_paths_written", value=()),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow.workflow_id, key="no_output_paths_written", value=True),
    )


def _correlation_provenance_rows_for_error(
    workflow_id: str,
    *,
    executed: bool,
    plan_only: bool,
    qc_mode: str,
) -> tuple[TabularAssociationCorrelationProvenanceRow, ...]:
    return (
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="schema_version", value=SCHEMA_VERSION),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="workflow_id", value=workflow_id),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="runtime_backend", value=RUNTIME_BACKEND_RECORDS),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="source_count", value=0),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="loaded_source_count", value=0),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="method_count", value=0),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="correlation_method_count", value=0),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="pair_count", value=0),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="result_row_count", value=0),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="qc_mode", value=qc_mode),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="executed", value=executed),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="plan_only", value=plan_only),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="will_write", value=False),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="output_written", value=False),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="output_paths_written", value=()),
        TabularAssociationCorrelationProvenanceRow(workflow_id=workflow_id, key="no_output_paths_written", value=True),
    )


def _unique_texts(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return tuple(result)


def _coerce_workflow_spec(document: Mapping[str, Any] | TabularAssociationWorkflowSpec) -> TabularAssociationWorkflowSpec:
    if isinstance(document, TabularAssociationWorkflowSpec):
        return document
    mapping = _as_mapping(document, field_name="workflow document")
    workflow_id = _first_present(mapping, "workflow_id", "id", default=_first_present(mapping, "name", default=""))
    sources = tuple(_source_from_mapping(item) for item in _mapping_or_sequence_items(mapping.get("sources", ()), "source_id"))
    outcomes = tuple(_variable_from_mapping(item, OutcomeSpec) for item in _mapping_or_sequence_items(mapping.get("outcomes", ()), "variable_id"))
    predictors = tuple(
        _variable_from_mapping(item, PredictorSpec)
        for item in _mapping_or_sequence_items(mapping.get("predictors", ()), "variable_id")
    )
    covariates = tuple(
        _variable_from_mapping(item, CovariateSpec)
        for item in _mapping_or_sequence_items(mapping.get("covariates", ()), "variable_id")
    )
    groupings = tuple(
        _variable_from_mapping(item, GroupingSpec)
        for item in _mapping_or_sequence_items(mapping.get("groupings", ()), "variable_id")
    )
    repeated_measures_doc = _first_present(mapping, "repeated_measures", "repeated_measures_spec")
    repeated_measures = _repeated_measures_from_mapping(repeated_measures_doc) if repeated_measures_doc is not None else None

    handoff_docs: list[Mapping[str, Any]] = []
    handoff_docs.extend(_mapping_or_sequence_items(mapping.get("handoffs", ()), "handoff_id"))
    handoff_docs.extend(
        _with_default(mapping_item, "handoff_type", "publication")
        for mapping_item in _mapping_or_sequence_items(mapping.get("publication_handoffs", ()), "handoff_id")
    )
    handoff_docs.extend(
        _with_default(mapping_item, "handoff_type", "visualization")
        for mapping_item in _mapping_or_sequence_items(mapping.get("visualization_handoffs", ()), "handoff_id")
    )
    return TabularAssociationWorkflowSpec(
        workflow_id=workflow_id,
        name=_first_present(mapping, "name"),
        description=_first_present(mapping, "description"),
        sources=sources,
        outcomes=outcomes,
        predictors=predictors,
        covariates=covariates,
        groupings=groupings,
        repeated_measures=repeated_measures,
        missing_data_policy=_policy_from_mapping(
            _first_present(mapping, "missing_data_policy", "missing_data"),
            MissingDataPolicy,
        ),
        duplicate_subject_policy=_policy_from_mapping(
            _first_present(mapping, "duplicate_subject_policy", "duplicate_subjects"),
            DuplicateSubjectPolicy,
        ),
        nonfinite_policy=_policy_from_mapping(_first_present(mapping, "nonfinite_policy", "non_finite_policy"), NonFinitePolicy),
        standardization_policy=_policy_from_mapping(
            _first_present(mapping, "standardization_policy", "scaling_policy"),
            StandardizationPolicy,
        ),
        transformation_policy=_policy_from_mapping(_first_present(mapping, "transformation_policy"), TransformationPolicy),
        methods=tuple(
            _method_from_mapping(item)
            for item in _mapping_or_sequence_items(_first_present(mapping, "methods", "association_methods", default=()), "method_id")
        ),
        families=tuple(
            _family_from_mapping(item)
            for item in _mapping_or_sequence_items(_first_present(mapping, "families", "association_families", default=()), "family_id")
        ),
        multiple_testing=tuple(
            _multiple_testing_from_mapping(item)
            for item in _mapping_or_sequence_items(_first_present(mapping, "multiple_testing", "fdr_families", default=()), "family_id")
        ),
        outputs=tuple(
            _output_from_mapping(item)
            for item in _mapping_or_sequence_items(_first_present(mapping, "outputs", "planned_outputs", default=()), "output_id")
        ),
        handoffs=tuple(_handoff_from_mapping(item) for item in handoff_docs),
        backend=_first_present(mapping, "backend", default=BACKEND_RECORDS),
        metadata=_as_metadata(mapping.get("metadata", {})),
    )


def _validate_workflow(workflow: TabularAssociationWorkflowSpec) -> tuple[AssociationValidationRow, ...]:
    rows: list[AssociationValidationRow] = []

    def add(status: str, code: str, message: str, *, location: str | None = None, level: str | None = None) -> None:
        rows.append(
            AssociationValidationRow(
                level=level or ("error" if status == "error" else "info"),
                status=status,
                code=code,
                message=message,
                location=location,
            )
        )

    add("ok", "workflow_identity", "Workflow id and name are declared.", location="workflow")
    add("ok", "backend_declared", "Backend is a declared schema value only.", location="workflow.backend")

    if not workflow.sources:
        add("error", "missing_sources", "At least one source table declaration is required.", location="sources")
    source_ids = [source.source_id for source in workflow.sources]
    for duplicate in _duplicates(source_ids):
        add("error", "duplicate_source_id", f"Source ids must be unique: {duplicate}.", location="sources")

    source_by_id = {source.source_id: source for source in workflow.sources}
    source_columns = {source.source_id: set(source.schema.column_names()) for source in workflow.sources}

    for source in workflow.sources:
        add("ok", "source_declared", f"Source {source.source_id} is declared.", location=f"sources.{source.source_id}")
        if source.schema.subject_id_column not in source_columns[source.source_id] and source_columns[source.source_id]:
            add(
                "error",
                "missing_subject_column",
                f"Subject identifier column {source.schema.subject_id_column!r} is not declared for source {source.source_id!r}.",
                location=f"sources.{source.source_id}.schema.subject_id_column",
            )
        for optional_name, column_name in (
            ("session_column", source.schema.session_column),
            ("timepoint_column", source.schema.timepoint_column),
        ):
            if column_name and source_columns[source.source_id] and column_name not in source_columns[source.source_id]:
                add(
                    "error",
                    f"missing_{optional_name}",
                    f"{optional_name} {column_name!r} is not declared for source {source.source_id!r}.",
                    location=f"sources.{source.source_id}.schema.{optional_name}",
                )
        add("ok", "source_backend_declared", "Source backend is adapter metadata only.", location=f"sources.{source.source_id}.backend")

    all_variables: list[AssociationVariableSpec] = [
        *workflow.outcomes,
        *workflow.predictors,
        *workflow.covariates,
        *workflow.groupings,
    ]
    variable_ids = [variable.variable_id for variable in all_variables]
    for duplicate in _duplicates(variable_ids):
        add("error", "duplicate_variable_id", f"Variable ids must be unique: {duplicate}.", location="variables")

    if not workflow.outcomes:
        add("error", "missing_outcome", "At least one outcome declaration is required.", location="outcomes")
    else:
        add("ok", "outcomes_declared", "At least one outcome declaration is present.", location="outcomes")

    if not workflow.predictors:
        add(
            "error",
            "missing_predictor",
            "At least one predictor declaration is required for association planning.",
            location="predictors",
        )
    else:
        add("ok", "predictors_declared", "At least one predictor declaration is present.", location="predictors")

    for variable in all_variables:
        if variable.source_id not in source_by_id:
            add(
                "error",
                "unknown_variable_source",
                f"Variable {variable.variable_id!r} references unknown source {variable.source_id!r}.",
                location=f"variables.{variable.variable_id}.source_id",
            )
            continue
        declared_columns = source_columns[variable.source_id]
        if declared_columns and variable.column_name not in declared_columns:
            add(
                "error",
                "unknown_variable_column",
                f"Variable {variable.variable_id!r} references undeclared column {variable.column_name!r}.",
                location=f"variables.{variable.variable_id}.column_name",
            )
    if all_variables:
        add("ok", "variable_columns_declared", "Variable column references are declared metadata.", location="variables")

    if workflow.repeated_measures is not None:
        repeated = workflow.repeated_measures
        if repeated.source_id not in source_by_id:
            add(
                "error",
                "unknown_repeated_measures_source",
                f"Repeated-measures spec references unknown source {repeated.source_id!r}.",
                location="repeated_measures.source_id",
            )
        else:
            declared_columns = source_columns[repeated.source_id]
            for column_name in (
                repeated.subject_id_column,
                repeated.session_column,
                repeated.timepoint_column,
                *repeated.unit_columns,
            ):
                if column_name and declared_columns and column_name not in declared_columns:
                    add(
                        "error",
                        "unknown_repeated_measures_column",
                        f"Repeated-measures column {column_name!r} is not declared for source {repeated.source_id!r}.",
                        location="repeated_measures",
                    )
            add(
                "ok",
                "repeated_measures_declared",
                "Repeated-measures identifiers are declared as plan-only metadata.",
                location="repeated_measures",
            )

    add("ok", "missing_data_policy", "Missing-data policy is a valid declaration.", location="missing_data_policy")
    add("ok", "duplicate_subject_policy", "Duplicate-subject policy is a valid declaration.", location="duplicate_subject_policy")
    add("ok", "nonfinite_policy", "Non-finite value policy is a valid declaration.", location="nonfinite_policy")
    add("ok", "standardization_policy", "Standardization policy is a valid declaration.", location="standardization_policy")
    add("ok", "transformation_policy", "Transformation policy is a valid declaration.", location="transformation_policy")

    method_ids = [method.method_id for method in workflow.methods]
    for duplicate in _duplicates(method_ids):
        add("error", "duplicate_method_id", f"Method ids must be unique: {duplicate}.", location="methods")
    method_id_set = set(method_ids)
    variable_id_by_role = {
        "outcome": {variable.variable_id for variable in workflow.outcomes},
        "predictor": {variable.variable_id for variable in workflow.predictors},
        "covariate": {variable.variable_id for variable in workflow.covariates},
        "grouping": {variable.variable_id for variable in workflow.groupings},
    }

    family_ids = {family.family_id for family in workflow.families} | {spec.family_id for spec in workflow.multiple_testing}
    for method in workflow.methods:
        add("ok", "association_method_declared", f"Association method {method.method_name!r} is declared.", location=f"methods.{method.method_id}")
        if method.executable or not method.planned_only:
            add(
                "error",
                "method_not_plan_only",
                f"Method {method.method_id!r} must remain planned-only and non-executable.",
                location=f"methods.{method.method_id}",
            )
        if method.method_name in DEFERRED_ASSOCIATION_METHODS:
            add(
                "warning",
                "deferred_repeated_measures_method",
                f"Method {method.method_id!r} is deferred/planned only.",
                location=f"methods.{method.method_id}",
                level="warning",
            )
            if workflow.repeated_measures is None:
                add(
                    "error",
                    "missing_repeated_measures_spec",
                    f"Method {method.method_id!r} requires a repeated-measures declaration.",
                    location=f"methods.{method.method_id}",
                )
        _validate_method_variable_ids(method, variable_id_by_role, add)
        if method.method_name == METHOD_PARTIAL_CORRELATION and not (workflow.covariates or method.covariate_ids):
            add(
                "error",
                "missing_partial_covariates",
                f"Partial association method {method.method_id!r} requires at least one covariate declaration.",
                location=f"methods.{method.method_id}.covariate_ids",
            )
        if method.family_id and method.family_id not in family_ids:
            add(
                "error",
                "unknown_method_family",
                f"Method {method.method_id!r} references unknown family {method.family_id!r}.",
                location=f"methods.{method.method_id}.family_id",
            )

    if not workflow.methods:
        add(
            "warning",
            "no_methods_declared",
            "No association methods are declared; preview is schema inventory only.",
            location="methods",
            level="warning",
        )

    family_spec_ids = [family.family_id for family in workflow.families]
    for duplicate in _duplicates(family_spec_ids):
        add("error", "duplicate_family_id", f"Association family ids must be unique: {duplicate}.", location="families")
    multiple_testing_ids = [spec.family_id for spec in workflow.multiple_testing]
    for duplicate in _duplicates(multiple_testing_ids):
        add(
            "error",
            "duplicate_multiple_testing_family_id",
            f"Multiple-testing family ids must be unique: {duplicate}.",
            location="multiple_testing",
        )
    for family in workflow.families:
        for method_id in family.method_ids:
            if method_id not in method_id_set:
                add(
                    "error",
                    "unknown_family_method",
                    f"Family {family.family_id!r} references unknown method {method_id!r}.",
                    location=f"families.{family.family_id}.method_ids",
                )
    if workflow.families or workflow.multiple_testing:
        add("ok", "multiple_testing_families_declared", "Multiple-testing/FDR family declarations are metadata only.", location="families")

    output_ids = [output.output_id for output in workflow.outputs]
    for duplicate in _duplicates(output_ids):
        add("error", "duplicate_output_id", f"Output ids must be unique: {duplicate}.", location="outputs")
    output_id_set = set(output_ids) | {method.output_id for method in workflow.methods if method.output_id}
    for output in workflow.outputs:
        for method_id in output.source_method_ids:
            if method_id not in method_id_set:
                add(
                    "error",
                    "unknown_output_method",
                    f"Output {output.output_id!r} references unknown method {method_id!r}.",
                    location=f"outputs.{output.output_id}.source_method_ids",
                )
        for family_id in output.family_ids:
            if family_id not in family_ids:
                add(
                    "error",
                    "unknown_output_family",
                    f"Output {output.output_id!r} references unknown family {family_id!r}.",
                    location=f"outputs.{output.output_id}.family_ids",
                )
    if workflow.outputs:
        add("ok", "outputs_declared", "Planned output declarations are metadata only.", location="outputs")

    handoff_ids = [handoff.handoff_id for handoff in workflow.handoffs]
    for duplicate in _duplicates(handoff_ids):
        add("error", "duplicate_handoff_id", f"Handoff ids must be unique: {duplicate}.", location="handoffs")
    for handoff in workflow.handoffs:
        if not handoff.plan_only:
            add(
                "error",
                "handoff_not_metadata_only",
                f"Handoff {handoff.handoff_id!r} must remain metadata-only.",
                location=f"handoffs.{handoff.handoff_id}",
            )
        for output_id in handoff.output_ids:
            if output_id_set and output_id not in output_id_set:
                add(
                    "error",
                    "unknown_handoff_output",
                    f"Handoff {handoff.handoff_id!r} references unknown output {output_id!r}.",
                    location=f"handoffs.{handoff.handoff_id}.output_ids",
                )
    if any(handoff.handoff_type == "publication" for handoff in workflow.handoffs):
        add(
            "ok",
            "publication_handoff_metadata_only",
            "Publication handoff declarations are metadata only.",
            location="handoffs.publication",
        )
    if any(handoff.handoff_type in {"visualization", "report"} for handoff in workflow.handoffs):
        add(
            "ok",
            "visualization_handoff_metadata_only",
            "Visualization/report handoff declarations are metadata only.",
            location="handoffs.visualization",
        )

    return tuple(rows)


def _validate_method_variable_ids(
    method: AssociationMethodSpec,
    variable_id_by_role: Mapping[str, set[str]],
    add: Any,
) -> None:
    role_fields = (
        ("outcome", method.outcome_ids, "outcome_ids"),
        ("predictor", method.predictor_ids, "predictor_ids"),
        ("covariate", method.covariate_ids, "covariate_ids"),
        ("grouping", method.grouping_ids, "grouping_ids"),
    )
    for role, values, field_name in role_fields:
        known_ids = variable_id_by_role[role]
        for variable_id in values:
            if variable_id not in known_ids:
                add(
                    "error",
                    f"unknown_method_{role}",
                    f"Method {method.method_id!r} references unknown {role} variable {variable_id!r}.",
                    location=f"methods.{method.method_id}.{field_name}",
                )


def _source_rows(workflow: TabularAssociationWorkflowSpec) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "workflow_id": workflow.workflow_id,
            "source_id": source.source_id,
            "declared_format": source.format,
            "path": source.path,
            "root_ref": source.root_ref,
            "backend": source.backend,
            "subject_id_column": source.schema.subject_id_column,
            "session_column": source.schema.session_column,
            "timepoint_column": source.schema.timepoint_column,
            "categorical_validation": source.schema.categorical_validation.to_dict(),
            "numeric_validation": source.schema.numeric_validation.to_dict(),
            "plan_only": True,
            "will_write": False,
            "metadata": source.metadata,
        }
        for source in workflow.sources
    )


def _column_rows(workflow: TabularAssociationWorkflowSpec) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for source in workflow.sources:
        for column in source.schema.columns:
            rows.append(
                {
                    "workflow_id": workflow.workflow_id,
                    "source_id": source.source_id,
                    "column_name": column.column_name,
                    "value_type": column.value_type,
                    "role": column.role,
                    "required": column.required,
                    "metadata": column.metadata,
                }
            )
    return tuple(rows)


def _variable_rows(workflow: TabularAssociationWorkflowSpec) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "workflow_id": workflow.workflow_id,
            "variable_id": variable.variable_id,
            "variable_role": variable.role,
            "source_id": variable.source_id,
            "column_name": variable.column_name,
            "label": variable.label,
            "metadata": variable.metadata,
            "plan_only": True,
        }
        for variable in [*workflow.outcomes, *workflow.predictors, *workflow.covariates, *workflow.groupings]
    )


def _method_rows(workflow: TabularAssociationWorkflowSpec) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "workflow_id": workflow.workflow_id,
            "method_id": method.method_id,
            "method_name": method.method_name,
            "outcome_ids": method.outcome_ids,
            "predictor_ids": method.predictor_ids,
            "covariate_ids": method.covariate_ids,
            "grouping_ids": method.grouping_ids,
            "family_id": method.family_id,
            "output_id": method.output_id,
            "planned_only": True,
            "executable": False,
            "deferred": method.is_deferred(),
            "metadata": method.metadata,
        }
        for method in workflow.methods
    )


def _family_rows(workflow: TabularAssociationWorkflowSpec) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for family in workflow.families:
        rows.append(
            {
                "workflow_id": workflow.workflow_id,
                "row_type": "association_family",
                "family_id": family.family_id,
                "method_ids": family.method_ids,
                "description": family.description,
                "planned_only": True,
                "metadata": family.metadata,
            }
        )
    for spec in workflow.multiple_testing:
        rows.append(
            {
                "workflow_id": workflow.workflow_id,
                "row_type": "multiple_testing",
                "family_id": spec.family_id,
                "method": spec.method,
                "planned_only": True,
                "metadata": spec.metadata,
            }
        )
    return tuple(rows)


def _output_rows(workflow: TabularAssociationWorkflowSpec) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "workflow_id": workflow.workflow_id,
            "output_id": output.output_id,
            "output_type": output.output_type,
            "planned_fields": output.planned_fields,
            "source_method_ids": output.source_method_ids,
            "family_ids": output.family_ids,
            "plan_only": True,
            "will_write": False,
            "output_written": False,
            "metadata": output.metadata,
        }
        for output in workflow.outputs
    )


def _handoff_rows(workflow: TabularAssociationWorkflowSpec, *, handoff_types: set[str]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "workflow_id": workflow.workflow_id,
            "handoff_id": handoff.handoff_id,
            "handoff_type": handoff.handoff_type,
            "output_ids": handoff.output_ids,
            "target": handoff.target,
            "planned_fields": handoff.planned_fields,
            "plan_only": True,
            "will_write": False,
            "output_written": False,
            "metadata": handoff.metadata,
        }
        for handoff in workflow.handoffs
        if handoff.handoff_type in handoff_types
    )


def _provenance_rows(workflow: TabularAssociationWorkflowSpec) -> tuple[AssociationProvenanceRow, ...]:
    return (
        AssociationProvenanceRow(key="schema_version", value=SCHEMA_VERSION),
        AssociationProvenanceRow(key="workflow_id", value=workflow.workflow_id),
        AssociationProvenanceRow(key="backend", value=workflow.backend),
        AssociationProvenanceRow(key="source_count", value=len(workflow.sources)),
        AssociationProvenanceRow(key="method_count", value=len(workflow.methods)),
        AssociationProvenanceRow(key="executed", value=False),
        AssociationProvenanceRow(key="plan_only", value=True),
        AssociationProvenanceRow(key="will_write", value=False),
    )


def _provenance_rows_for_error(workflow_id: str) -> tuple[AssociationProvenanceRow, ...]:
    return (
        AssociationProvenanceRow(key="schema_version", value=SCHEMA_VERSION),
        AssociationProvenanceRow(key="workflow_id", value=workflow_id),
        AssociationProvenanceRow(key="executed", value=False),
        AssociationProvenanceRow(key="plan_only", value=True),
        AssociationProvenanceRow(key="will_write", value=False),
    )


def _unwrap_tabular_association_workflow_document(
    document: Mapping[str, Any] | TabularAssociationWorkflowSpec,
) -> Mapping[str, Any] | TabularAssociationWorkflowSpec:
    if isinstance(document, TabularAssociationWorkflowSpec):
        return document
    mapping = _as_mapping(document, field_name="workflow document")
    nested = mapping.get("tabular_association_workflow")
    if nested is None:
        return mapping
    if isinstance(nested, TabularAssociationWorkflowSpec):
        return nested
    return _as_mapping(nested, field_name="tabular_association_workflow")


def _best_effort_qc_workflow_id(document: Any) -> str:
    if isinstance(document, Mapping) and "tabular_association_workflow" in document:
        try:
            return _best_effort_workflow_id(_unwrap_tabular_association_workflow_document(document))
        except (TypeError, ValueError):
            return "unparsed-workflow"
    return _best_effort_workflow_id(document)


def _source_inventory_spec_from_mapping(value: Mapping[str, Any] | TabularSourceInventorySpec) -> TabularSourceInventorySpec:
    if isinstance(value, TabularSourceInventorySpec):
        return value
    mapping = _as_mapping(value, field_name="source inventory spec")
    return TabularSourceInventorySpec(
        source_id=_first_present(mapping, "source_id", "id", "source", default=""),
        source_format=_first_present(mapping, "source_format", "format"),
        path=_first_present(mapping, "path", "source_path"),
        required=bool(_first_present(mapping, "required", default=True)),
        row_key=_first_present(mapping, "row_key", "rows_key"),
        metadata=_as_metadata(mapping.get("metadata", {})),
    )


def _coerce_source_inventory_specs(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any] | TabularSourceInventorySpec],
) -> tuple[TabularSourceInventorySpec, ...]:
    if value is None:
        return ()
    if isinstance(value, TabularSourceInventorySpec):
        return (value,)
    if isinstance(value, Mapping):
        if any(key in value for key in ("source_id", "id", "source")):
            return (_source_inventory_spec_from_mapping(value),)
        specs: list[TabularSourceInventorySpec] = []
        for source_id, item in value.items():
            item_mapping = dict(_as_mapping(item, field_name="source inventory spec"))
            item_mapping.setdefault("source_id", source_id)
            specs.append(_source_inventory_spec_from_mapping(item_mapping))
        return tuple(specs)
    if isinstance(value, (str, bytes)):
        raise TypeError("source_inventory_specs must be mappings, not strings.")
    try:
        return tuple(_source_inventory_spec_from_mapping(item) for item in value)
    except TypeError as exc:
        raise TypeError("source_inventory_specs must be a mapping or sequence of mappings.") from exc


def _source_inventory_specs_by_id(
    specs: Sequence[TabularSourceInventorySpec],
) -> dict[str, TabularSourceInventorySpec]:
    by_id: dict[str, TabularSourceInventorySpec] = {}
    for spec in specs:
        by_id[spec.source_id] = spec
    return by_id


def _coerce_source_rows_by_id(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _as_mapping(value, field_name="source_rows_by_id")


def _coerce_row_source_adapter_spec(value: Any) -> TabularAssociationRowSourceAdapterSpec:
    if isinstance(value, TabularAssociationRowSourceAdapterSpec):
        return value
    mapping = _as_mapping(value, field_name="row_source_adapter_spec")
    return TabularAssociationRowSourceAdapterSpec(
        adapter_id=_first_present(
            mapping,
            "adapter_id",
            default=TABULAR_ASSOCIATION_ROW_SOURCE_ADAPTER_VERSION,
        ),
        requested_backend=_first_present(mapping, "requested_backend", "backend", default=BACKEND_RECORDS),
        runtime_backend=RUNTIME_BACKEND_RECORDS,
        row_source_kind=_first_present(mapping, "row_source_kind", default="uninspected"),
        include_input_row_index=bool(_first_present(mapping, "include_input_row_index", default=False)),
        input_row_index_field=_first_present(mapping, "input_row_index_field", default="input_row_index"),
        metadata=_as_metadata(mapping.get("metadata", {})),
    )


def _coerce_row_source_records(
    row_source: Any,
    *,
    spec: TabularAssociationRowSourceAdapterSpec,
) -> TabularAssociationRowSourceResult:
    row_source_kind, row_values, protocol_errors = _row_source_values(row_source)
    observed_metadata_columns = _row_source_columns(row_source)
    rows: tuple[dict[str, Any], ...] = ()
    row_errors: tuple[str, ...] = ()
    if not protocol_errors:
        rows, row_errors = _records_from_row_values(row_values, row_source_kind=row_source_kind)
    errors = (*protocol_errors, *row_errors)
    observed_columns = _observed_columns_from_rows(rows, initial_columns=observed_metadata_columns)
    if spec.include_input_row_index and not errors:
        collision = any(spec.input_row_index_field in row for row in rows)
        if collision:
            errors = (
                f"Input row index field {spec.input_row_index_field!r} already exists in at least one input record.",
            )
        else:
            indexed_rows: list[dict[str, Any]] = []
            for row_index, row in enumerate(rows):
                copied = dict(row)
                copied[spec.input_row_index_field] = row_index
                indexed_rows.append(copied)
            rows = tuple(indexed_rows)
            observed_columns = _observed_columns_from_rows(rows, initial_columns=observed_columns)
    valid = not errors
    status = "ok" if valid else "error"
    result_records: tuple[dict[str, Any], ...] = rows if valid else ()
    result_row_count = len(result_records)
    result_observed_columns = _observed_columns_from_rows(result_records, initial_columns=observed_columns)
    result_spec = TabularAssociationRowSourceAdapterSpec(
        adapter_id=spec.adapter_id,
        requested_backend=spec.requested_backend,
        runtime_backend=RUNTIME_BACKEND_RECORDS,
        row_source_kind=row_source_kind,
        include_input_row_index=spec.include_input_row_index,
        input_row_index_field=spec.input_row_index_field,
        metadata=spec.metadata,
        executed=False,
        plan_only=True,
        will_write=False,
        output_written=False,
        no_output_written=True,
        output_paths_written=(),
    )
    warnings: tuple[str, ...] = ()
    message = (
        f"Coerced {result_row_count} row-source records on the records runtime backend."
        if valid
        else "Could not coerce row source into mapping records."
    )
    qc_code = "row_source_records_coerced" if valid else "row_source_records_unsupported"
    if spec.include_input_row_index and errors and "already exists" in errors[0]:
        qc_code = "row_source_input_row_index_collision"
    qc_rows = (
        TabularAssociationRowSourceQcRow(
            adapter_id=result_spec.adapter_id,
            requested_backend=result_spec.requested_backend,
            runtime_backend=result_spec.runtime_backend,
            row_source_kind=result_spec.row_source_kind,
            status=status,
            code=qc_code,
            message=message,
            row_count=result_row_count,
            observed_column_count=len(result_observed_columns),
            include_input_row_index=result_spec.include_input_row_index,
            warnings=warnings,
            errors=errors,
            metadata={
                "input_row_index_field": result_spec.input_row_index_field,
                "no_output_written": True,
            },
        ),
    )
    provenance_rows = _row_source_provenance_rows(
        spec=result_spec,
        row_count=result_row_count,
        observed_column_count=len(result_observed_columns),
    )
    return TabularAssociationRowSourceResult(
        adapter_version=TABULAR_ASSOCIATION_ROW_SOURCE_ADAPTER_VERSION,
        spec=result_spec,
        valid=valid,
        status=status,
        records=result_records,
        observed_columns=result_observed_columns,
        warnings=warnings,
        errors=errors,
        qc_rows=qc_rows,
        provenance_rows=provenance_rows,
        executed=True,
        plan_only=False,
        will_write=False,
        output_written=False,
        no_output_written=True,
        output_paths_written=(),
    )


def _row_source_error_result(
    *,
    adapter_id: str,
    requested_backend: str,
    row_source_kind: str,
    include_input_row_index: bool,
    input_row_index_field: str,
    metadata: Mapping[str, Any],
    errors: Sequence[str],
) -> TabularAssociationRowSourceResult:
    safe_input_row_index_field = str(input_row_index_field).strip() or "input_row_index"
    spec = TabularAssociationRowSourceAdapterSpec(
        adapter_id=adapter_id,
        requested_backend=requested_backend if requested_backend in SUPPORTED_TABULAR_ASSOCIATION_BACKENDS else BACKEND_RECORDS,
        runtime_backend=RUNTIME_BACKEND_RECORDS,
        row_source_kind=row_source_kind,
        include_input_row_index=include_input_row_index,
        input_row_index_field=safe_input_row_index_field,
        metadata=metadata,
    )
    qc_rows = (
        TabularAssociationRowSourceQcRow(
            adapter_id=spec.adapter_id,
            requested_backend=spec.requested_backend,
            runtime_backend=spec.runtime_backend,
            row_source_kind=spec.row_source_kind,
            status="error",
            code="row_source_adapter_plan_error",
            message="Could not plan row-source adapter coercion.",
            row_count=0,
            observed_column_count=0,
            include_input_row_index=spec.include_input_row_index,
            errors=errors,
        ),
    )
    return TabularAssociationRowSourceResult(
        adapter_version=TABULAR_ASSOCIATION_ROW_SOURCE_ADAPTER_VERSION,
        spec=spec,
        valid=False,
        status="error",
        records=(),
        observed_columns=(),
        warnings=(),
        errors=errors,
        qc_rows=qc_rows,
        provenance_rows=_row_source_provenance_rows(spec=spec, row_count=0, observed_column_count=0),
    )


def _row_source_values(row_source: Any) -> tuple[str, Any, tuple[str, ...]]:
    if row_source is None:
        return SOURCE_KIND_UNSUPPORTED, (), ("Row source is required.",)
    if isinstance(row_source, (str, bytes)):
        return SOURCE_KIND_UNSUPPORTED, (), ("Row source must not be a string or bytes value.",)
    if isinstance(row_source, Mapping):
        return SOURCE_KIND_UNSUPPORTED, (), ("Row source must be a sequence of records, not a single mapping.",)
    if isinstance(row_source, Sequence):
        return "mapping_sequence", row_source, ()

    for method_name in ("to_dicts", "to_records"):
        method = _safe_row_source_attr(row_source, method_name)
        if method is None:
            continue
        if not callable(method):
            continue
        try:
            return method_name, method(), ()
        except Exception as exc:
            return method_name, (), (f"Row source method {method_name}() failed: {exc}",)

    iter_rows = _safe_row_source_attr(row_source, "iter_rows")
    if callable(iter_rows):
        try:
            return "iter_rows_named", iter_rows(named=True), ()
        except TypeError:
            try:
                return "iter_rows", iter_rows(), ()
            except Exception as exc:
                return "iter_rows", (), (f"Row source method iter_rows() failed: {exc}",)
        except Exception as exc:
            return "iter_rows_named", (), (f"Row source method iter_rows(named=True) failed: {exc}",)

    for attr_name in ("rows", "records"):
        value = _safe_row_source_attr(row_source, attr_name)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except Exception as exc:
                return attr_name, (), (f"Row source {attr_name}() failed: {exc}",)
        return attr_name, value, ()

    return SOURCE_KIND_UNSUPPORTED, (), ("Unsupported row source; expected mapping records or a generic record-producing protocol.",)


def _safe_row_source_attr(row_source: Any, attr_name: str) -> Any:
    try:
        return getattr(row_source, attr_name)
    except Exception:
        return None


def _records_from_row_values(row_values: Any, *, row_source_kind: str) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    if row_values is None:
        return (), (f"Row-source protocol {row_source_kind!r} returned None.",)
    if isinstance(row_values, (str, bytes)):
        return (), (f"Row-source protocol {row_source_kind!r} returned a string or bytes value.",)
    if isinstance(row_values, Mapping):
        return (), (f"Row-source protocol {row_source_kind!r} returned a single mapping instead of rows.",)
    try:
        iterator = iter(row_values)
    except TypeError:
        return (), (f"Row-source protocol {row_source_kind!r} did not return iterable rows.",)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    row_index = 0
    try:
        for row_index, row in enumerate(iterator, start=1):
            record, error = _record_from_row_object(row)
            if error:
                errors.append(f"Row-source protocol {row_source_kind!r} produced unsupported row {row_index}: {error}")
            else:
                rows.append(record)
    except Exception as exc:
        errors.append(f"Row-source protocol {row_source_kind!r} failed while yielding row {row_index + 1}: {exc}")
    return tuple(rows), tuple(errors)


def _record_from_row_object(row: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(row, Mapping):
        return _json_safe_record_mapping(row), None
    if is_dataclass(row) and not isinstance(row, type):
        return _json_safe_record_mapping({field_.name: getattr(row, field_.name) for field_ in fields(row)}), None
    row_to_dict = _safe_row_source_attr(row, "to_dict")
    if callable(row_to_dict):
        try:
            row_mapping = row_to_dict()
        except Exception as exc:
            return {}, f"to_dict() failed: {exc}"
        if not isinstance(row_mapping, Mapping):
            return {}, "to_dict() did not return a mapping"
        return _json_safe_record_mapping(row_mapping), None
    return {}, "row is not a mapping, dataclass, or to_dict()-mapping object"


def _json_safe_record_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _row_source_json_safe(item) for key, item in value.items()}


def _row_source_json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field_.name: _row_source_json_safe(getattr(value, field_.name)) for field_ in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _row_source_json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_row_source_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _row_source_columns(row_source: Any) -> tuple[str, ...]:
    columns = _safe_row_source_attr(row_source, "columns")
    if columns is None or callable(columns) or isinstance(columns, (str, bytes)):
        return ()
    try:
        return _unique_texts(tuple(str(column) for column in columns if str(column)))
    except TypeError:
        return ()


def _row_source_provenance_rows(
    *,
    spec: TabularAssociationRowSourceAdapterSpec,
    row_count: int,
    observed_column_count: int,
) -> tuple[TabularAssociationRowSourceProvenanceRow, ...]:
    return (
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="schema_version",
            value=SCHEMA_VERSION,
        ),
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="adapter_version",
            value=TABULAR_ASSOCIATION_ROW_SOURCE_ADAPTER_VERSION,
        ),
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="requested_backend",
            value=spec.requested_backend,
        ),
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="runtime_backend",
            value=RUNTIME_BACKEND_RECORDS,
        ),
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="row_source_kind",
            value=spec.row_source_kind,
        ),
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="adapter_kind",
            value="records_adapter",
        ),
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="row_count",
            value=row_count,
        ),
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="observed_column_count",
            value=observed_column_count,
        ),
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="include_input_row_index",
            value=spec.include_input_row_index,
        ),
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="will_write",
            value=False,
        ),
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="output_written",
            value=False,
        ),
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="output_paths_written",
            value=(),
        ),
        TabularAssociationRowSourceProvenanceRow(
            adapter_id=spec.adapter_id,
            key="no_output_written",
            value=True,
        ),
    )


def _planned_source_inventory_rows(
    workflow: TabularAssociationWorkflowSpec,
    *,
    spec_by_id: Mapping[str, TabularSourceInventorySpec],
) -> tuple[TabularSourceInventoryRow, ...]:
    rows: list[TabularSourceInventoryRow] = []
    for source in workflow.sources:
        inventory_spec = spec_by_id.get(source.source_id)
        source_format = _source_format(source, inventory_spec)
        path = _source_path(source, inventory_spec)
        declared_columns = source.schema.column_names()
        source_kind = _source_kind_from_format(source_format) if path else SOURCE_KIND_MISSING
        if source_kind == SOURCE_KIND_MISSING and path:
            source_kind = SOURCE_KIND_UNSUPPORTED
        rows.append(
            TabularSourceInventoryRow(
                workflow_id=workflow.workflow_id,
                source_id=source.source_id,
                source_kind=source_kind,
                source_format=source_format,
                path=path,
                requested_backend=source.backend,
                runtime_backend=RUNTIME_BACKEND_RECORDS,
                row_count=None,
                observed_column_count=0,
                observed_columns=(),
                declared_columns=declared_columns,
                declared_only_columns=declared_columns,
                observed_only_columns=(),
                declared_and_observed_columns=(),
                load_status="planned",
                warnings=(),
                errors=(),
                provenance={"plan_only": True, "will_write": False},
            )
        )
    return tuple(rows)


def _load_qc_source(
    *,
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    source_rows_by_id: Mapping[str, Any],
    inventory_spec: TabularSourceInventorySpec | None,
) -> dict[str, Any]:
    required = inventory_spec.required if inventory_spec is not None else True
    source_format = _source_format(source, inventory_spec)
    path = _source_path(source, inventory_spec)
    if source.source_id in source_rows_by_id:
        return _load_in_memory_qc_source(
            workflow=workflow,
            source=source,
            source_rows=source_rows_by_id[source.source_id],
            source_format=source_format,
            path=path,
            required=required,
        )
    if not path:
        message = f"Source {source.source_id!r} has no in-memory rows or configured path."
        warnings = (message,) if not required else ()
        errors = () if not required else (message,)
        return _loaded_source_payload(
            source_kind=SOURCE_KIND_MISSING,
            source_format=source_format,
            path=path,
            required=required,
            load_status="missing",
            rows=(),
            observed_columns=(),
            warnings=warnings,
            errors=errors,
            message=message,
            provenance={"root_ref_resolved": False},
        )
    if source_format not in SUPPORTED_SOURCE_INVENTORY_FORMATS:
        message = f"Source {source.source_id!r} has unsupported source format {source_format!r}."
        warnings = (message,) if not required else ()
        errors = () if not required else (message,)
        return _loaded_source_payload(
            source_kind=SOURCE_KIND_UNSUPPORTED,
            source_format=source_format,
            path=path,
            required=required,
            load_status="unsupported",
            rows=(),
            observed_columns=(),
            warnings=warnings,
            errors=errors,
            message=message,
            provenance={"root_ref_resolved": False},
        )
    return _load_file_qc_source(
        source=source,
        source_format=source_format,
        path=path,
        row_key=_source_row_key(source, inventory_spec),
        required=required,
    )


def _load_in_memory_qc_source(
    *,
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    source_rows: Any,
    source_format: str | None,
    path: str | None,
    required: bool,
) -> dict[str, Any]:
    del workflow
    adapter_result = coerce_tabular_association_records(
        source_rows,
        requested_backend=source.backend,
        metadata={"source_id": source.source_id, "row_source": "source_rows_by_id"},
    )
    errors = [f"In-memory rows for source {source.source_id!r}: {error}" for error in adapter_result.errors]
    rows = list(adapter_result.records)
    observed_columns = tuple(adapter_result.observed_columns)
    warnings: list[str] = []
    load_status = "loaded"
    message = f"Loaded {len(rows)} in-memory rows for source {source.source_id!r}."
    if errors:
        load_status = "error"
        message = f"Could not load in-memory rows for source {source.source_id!r}."
    elif not rows:
        load_status = "empty"
        warning = f"Source {source.source_id!r} has zero rows."
        warnings.append(warning)
        message = warning
    return _loaded_source_payload(
        source_kind=SOURCE_KIND_IN_MEMORY,
        source_format=source_format,
        path=path,
        required=required,
        load_status=load_status,
        rows=rows,
        observed_columns=observed_columns,
        warnings=warnings,
        errors=errors,
        message=message,
        provenance={
            "row_source": "source_rows_by_id",
            "root_ref_resolved": False,
            "row_source_adapter_version": adapter_result.adapter_version,
            "row_source_adapter_id": adapter_result.spec.adapter_id,
            "row_source_kind": adapter_result.spec.row_source_kind,
            "row_source_requested_backend": adapter_result.spec.requested_backend,
            "row_source_runtime_backend": adapter_result.spec.runtime_backend,
            "row_source_no_output_written": adapter_result.no_output_written,
            "row_source_output_paths_written": tuple(adapter_result.output_paths_written),
        },
    )


def _load_file_qc_source(
    *,
    source: TabularSourceSpec,
    source_format: str,
    path: str,
    row_key: str | None,
    required: bool,
) -> dict[str, Any]:
    source_path = Path(path)
    source_kind = _source_kind_from_format(source_format)
    provenance: dict[str, Any] = {"root_ref_resolved": False}
    if not source_path.exists():
        message = f"Source path for {source.source_id!r} does not exist."
        warnings = (message,) if not required else ()
        errors = () if not required else (message,)
        return _loaded_source_payload(
            source_kind=SOURCE_KIND_MISSING,
            source_format=source_format,
            path=path,
            required=required,
            load_status="missing",
            rows=(),
            observed_columns=(),
            warnings=warnings,
            errors=errors,
            message=message,
            provenance=provenance,
        )
    if source_path.is_dir():
        message = f"Source path for {source.source_id!r} is a directory, not a file."
        warnings = (message,) if not required else ()
        errors = () if not required else (message,)
        return _loaded_source_payload(
            source_kind=source_kind,
            source_format=source_format,
            path=path,
            required=required,
            load_status="error",
            rows=(),
            observed_columns=(),
            warnings=warnings,
            errors=errors,
            message=message,
            provenance=provenance,
        )
    file_hash = _file_sha256(source_path)
    if file_hash:
        provenance["sha256"] = file_hash
    try:
        if source_format == "tsv":
            rows, observed_columns = _read_delimited_rows(source_path, delimiter="\t")
        elif source_format == "csv":
            rows, observed_columns = _read_delimited_rows(source_path, delimiter=",")
        else:
            rows, observed_columns = _read_json_rows(source_path, row_key=row_key)
    except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError, TypeError, ValueError) as exc:
        message = f"Could not load source {source.source_id!r}: {exc}"
        warnings = (message,) if not required else ()
        errors = () if not required else (message,)
        return _loaded_source_payload(
            source_kind=source_kind,
            source_format=source_format,
            path=path,
            required=required,
            load_status="error",
            rows=(),
            observed_columns=(),
            warnings=warnings,
            errors=errors,
            message=message,
            provenance=provenance,
        )
    warnings: list[str] = []
    load_status = "loaded"
    message = f"Loaded {len(rows)} rows for source {source.source_id!r}."
    if not rows:
        load_status = "empty"
        message = f"Source {source.source_id!r} has zero rows."
        warnings.append(message)
    return _loaded_source_payload(
        source_kind=source_kind,
        source_format=source_format,
        path=path,
        required=required,
        load_status=load_status,
        rows=rows,
        observed_columns=observed_columns,
        warnings=warnings,
        errors=(),
        message=message,
        provenance=provenance,
    )


def _loaded_source_payload(
    *,
    source_kind: str,
    source_format: str | None,
    path: str | None,
    required: bool,
    load_status: str,
    rows: Sequence[Mapping[str, Any]],
    observed_columns: Sequence[str],
    warnings: Sequence[str],
    errors: Sequence[str],
    message: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_kind": source_kind,
        "source_format": source_format,
        "path": path,
        "required": required,
        "load_status": load_status,
        "rows": tuple(rows),
        "observed_columns": tuple(observed_columns),
        "warnings": tuple(warnings),
        "errors": tuple(errors),
        "message": message,
        "provenance": dict(provenance),
    }


def _read_delimited_rows(path: Path, *, delimiter: str) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        header_columns = tuple(str(column) for column in (reader.fieldnames or ()) if column is not None and str(column))
        for raw_row in reader:
            rows.append({str(key): value for key, value in raw_row.items() if key is not None})
    return tuple(rows), _observed_columns_from_rows(rows, initial_columns=header_columns)


def _read_json_rows(path: Path, *, row_key: str | None) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        row_values = payload
    elif isinstance(payload, Mapping):
        if not row_key:
            raise ValueError("JSON object sources require a configured row_key.")
        if row_key not in payload:
            raise ValueError(f"JSON object source does not contain row_key {row_key!r}.")
        row_values = payload[row_key]
    else:
        raise ValueError("JSON sources must be a list of row objects or an object containing row_key rows.")
    if not isinstance(row_values, list):
        raise ValueError("JSON row source must be a list of row objects.")
    rows: list[Mapping[str, Any]] = []
    for row_index, row in enumerate(row_values, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"JSON row {row_index} is not an object.")
        rows.append(row)
    return tuple(rows), _observed_columns_from_rows(rows)


def _file_sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _source_format(source: TabularSourceSpec, inventory_spec: TabularSourceInventorySpec | None) -> str | None:
    declared = inventory_spec.source_format if inventory_spec and inventory_spec.source_format else source.format
    if declared:
        return str(declared).strip().lower()
    path = _source_path(source, inventory_spec)
    if not path:
        return None
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or None


def _source_path(source: TabularSourceSpec, inventory_spec: TabularSourceInventorySpec | None) -> str | None:
    if inventory_spec is not None and inventory_spec.path:
        return inventory_spec.path
    return source.path


def _source_row_key(source: TabularSourceSpec, inventory_spec: TabularSourceInventorySpec | None) -> str | None:
    if inventory_spec is not None and inventory_spec.row_key:
        return inventory_spec.row_key
    row_key = source.metadata.get("row_key")
    if isinstance(row_key, str):
        return row_key.strip() or None
    return None


def _source_kind_from_format(source_format: str | None) -> str:
    if source_format == "tsv":
        return SOURCE_KIND_TSV
    if source_format == "csv":
        return SOURCE_KIND_CSV
    if source_format == "json":
        return SOURCE_KIND_JSON
    if source_format:
        return SOURCE_KIND_UNSUPPORTED
    return SOURCE_KIND_MISSING


def _source_inventory_row(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    loaded_source: Mapping[str, Any],
) -> TabularSourceInventoryRow:
    observed_columns = tuple(loaded_source["observed_columns"])
    declared_columns = source.schema.column_names()
    observed_set = set(observed_columns)
    declared_set = set(declared_columns)
    return TabularSourceInventoryRow(
        workflow_id=workflow.workflow_id,
        source_id=source.source_id,
        source_kind=str(loaded_source["source_kind"]),
        source_format=loaded_source["source_format"],
        path=loaded_source["path"],
        requested_backend=source.backend,
        runtime_backend=RUNTIME_BACKEND_RECORDS,
        row_count=len(loaded_source["rows"]),
        observed_column_count=len(observed_columns),
        observed_columns=observed_columns,
        declared_columns=declared_columns,
        declared_only_columns=tuple(column for column in declared_columns if column not in observed_set),
        observed_only_columns=tuple(column for column in observed_columns if column not in declared_set),
        declared_and_observed_columns=tuple(column for column in declared_columns if column in observed_set),
        load_status=str(loaded_source["load_status"]),
        warnings=loaded_source["warnings"],
        errors=loaded_source["errors"],
        provenance=loaded_source["provenance"],
    )


def _source_load_row(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    loaded_source: Mapping[str, Any],
) -> TabularSourceLoadRow:
    return TabularSourceLoadRow(
        workflow_id=workflow.workflow_id,
        source_id=source.source_id,
        source_kind=str(loaded_source["source_kind"]),
        source_format=loaded_source["source_format"],
        path=loaded_source["path"],
        required=bool(loaded_source["required"]),
        load_status=str(loaded_source["load_status"]),
        row_count=len(loaded_source["rows"]),
        observed_column_count=len(loaded_source["observed_columns"]),
        warning_count=len(loaded_source["warnings"]),
        error_count=len(loaded_source["errors"]),
        message=str(loaded_source["message"]),
        warnings=loaded_source["warnings"],
        errors=loaded_source["errors"],
        provenance=loaded_source["provenance"],
    )


def _column_inventory_rows_for_source(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    loaded_source: Mapping[str, Any],
) -> tuple[TabularColumnInventoryRow, ...]:
    observed_columns = tuple(loaded_source["observed_columns"])
    observed_set = set(observed_columns)
    declared_by_name = {column.column_name: column for column in source.schema.columns}
    rows: list[TabularColumnInventoryRow] = []
    for column in source.schema.columns:
        observed = column.column_name in observed_set
        errors = (f"Required declared column {column.column_name!r} was not observed.",) if column.required and not observed else ()
        rows.append(
            TabularColumnInventoryRow(
                workflow_id=workflow.workflow_id,
                source_id=source.source_id,
                column_name=column.column_name,
                declared=True,
                observed=observed,
                value_type=column.value_type,
                role=column.role,
                required=column.required,
                status="error" if errors else "ok",
                warnings=(),
                errors=errors,
                metadata=column.metadata,
            )
        )
    for column_name in observed_columns:
        if column_name in declared_by_name:
            continue
        warning = f"Observed column {column_name!r} is not declared in source {source.source_id!r}."
        rows.append(
            TabularColumnInventoryRow(
                workflow_id=workflow.workflow_id,
                source_id=source.source_id,
                column_name=column_name,
                declared=False,
                observed=True,
                value_type=None,
                role=None,
                required=None,
                status="warning",
                warnings=(warning,),
                errors=(),
            )
        )
    return tuple(rows)


def _schema_validation_rows_for_source(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    loaded_source: Mapping[str, Any],
) -> tuple[TabularSchemaValidationRow, ...]:
    if loaded_source["load_status"] not in {"loaded", "empty"}:
        status = "error" if loaded_source["errors"] else "warning"
        code = "source_not_loaded" if loaded_source["errors"] else "source_not_loaded_optional"
        return (
            TabularSchemaValidationRow(
                workflow_id=workflow.workflow_id,
                source_id=source.source_id,
                check_name="source_load",
                status=status,
                code=code,
                message=str(loaded_source["message"]),
                required=bool(loaded_source["required"]),
                observed=False,
            ),
        )

    observed_set = set(loaded_source["observed_columns"])
    rows: list[TabularSchemaValidationRow] = []

    def add_presence(
        *,
        check_name: str,
        code_prefix: str,
        column_name: str,
        role: str | None,
        required: bool,
    ) -> None:
        observed = column_name in observed_set
        status = "ok" if observed or not required else "error"
        code = f"{code_prefix}_present" if observed else f"{code_prefix}_missing"
        message = (
            f"Column {column_name!r} is present for {check_name}."
            if observed
            else f"Column {column_name!r} is missing for {check_name}."
        )
        rows.append(
            TabularSchemaValidationRow(
                workflow_id=workflow.workflow_id,
                source_id=source.source_id,
                check_name=check_name,
                status=status,
                code=code,
                message=message,
                column_name=column_name,
                role=role,
                required=required,
                observed=observed,
            )
        )

    add_presence(
        check_name="subject_identifier",
        code_prefix="subject_identifier_column",
        column_name=source.schema.subject_id_column,
        role="subject_identifier",
        required=True,
    )
    if source.schema.session_column:
        add_presence(
            check_name="session_identifier",
            code_prefix="session_column",
            column_name=source.schema.session_column,
            role="session_identifier",
            required=True,
        )
    if source.schema.timepoint_column:
        add_presence(
            check_name="timepoint_identifier",
            code_prefix="timepoint_column",
            column_name=source.schema.timepoint_column,
            role="timepoint_identifier",
            required=True,
        )
    for column in source.schema.columns:
        if column.required:
            add_presence(
                check_name="required_declared_column",
                code_prefix="required_declared_column",
                column_name=column.column_name,
                role=column.role,
                required=True,
            )

    for variable in _variables_for_source(workflow, source.source_id):
        add_presence(
            check_name=f"{variable.role}_variable",
            code_prefix=f"{variable.role}_column",
            column_name=variable.column_name,
            role=variable.role,
            required=True,
        )

    repeated = workflow.repeated_measures
    if repeated is not None and repeated.source_id == source.source_id:
        repeated_columns = repeated.unit_columns or tuple(
            column
            for column in (repeated.subject_id_column, repeated.session_column, repeated.timepoint_column)
            if column
        )
        for column_name in repeated_columns:
            add_presence(
                check_name="repeated_measures_identifier",
                code_prefix="repeated_measures_column",
                column_name=column_name,
                role="repeated_measures_identifier",
                required=True,
            )

    if not loaded_source["rows"]:
        rows.append(
            TabularSchemaValidationRow(
                workflow_id=workflow.workflow_id,
                source_id=source.source_id,
                check_name="row_count",
                status="warning",
                code="zero_row_source",
                message=f"Source {source.source_id!r} has zero rows for QC.",
                required=True,
                observed=False,
            )
        )
    return tuple(rows)


def _variable_qc_rows_for_source(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    loaded_source: Mapping[str, Any],
) -> tuple[TabularVariableQcRow, ...]:
    declared_by_name = {column.column_name: column for column in source.schema.columns}
    observed_set = set(loaded_source["observed_columns"])
    rows: list[TabularVariableQcRow] = []
    for variable in _variables_for_source(workflow, source.source_id):
        declared = variable.column_name in declared_by_name
        observed = variable.column_name in observed_set
        status = "ok" if declared and observed else "error"
        code = "variable_column_observed"
        if not declared:
            code = "variable_column_not_declared"
        elif not observed:
            code = "variable_column_not_observed"
        message = (
            f"Variable {variable.variable_id!r} column {variable.column_name!r} is observed."
            if status == "ok"
            else f"Variable {variable.variable_id!r} column {variable.column_name!r} is not available for QC."
        )
        rows.append(
            TabularVariableQcRow(
                workflow_id=workflow.workflow_id,
                variable_id=variable.variable_id,
                variable_role=variable.role,
                source_id=source.source_id,
                column_name=variable.column_name,
                declared_in_schema=declared,
                observed=observed,
                value_type=declared_by_name[variable.column_name].value_type if declared else None,
                status=status,
                code=code,
                message=message,
            )
        )
    return tuple(rows)


def _missingness_rows_for_source(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    loaded_source: Mapping[str, Any],
) -> tuple[TabularMissingnessRow, ...]:
    source_rows = tuple(loaded_source["rows"])
    total_count = len(source_rows)
    rows: list[TabularMissingnessRow] = []
    for column in source.schema.columns:
        missing_count = sum(1 for row in source_rows if _is_missing_value(row.get(column.column_name)))
        nonmissing_count = total_count - missing_count
        status, code, message = _missingness_status(
            column=column,
            missing_count=missing_count,
            total_count=total_count,
            policy=workflow.missing_data_policy,
        )
        rows.append(
            TabularMissingnessRow(
                workflow_id=workflow.workflow_id,
                source_id=source.source_id,
                column_name=column.column_name,
                role=column.role,
                required=column.required,
                missing_count=missing_count,
                nonmissing_count=nonmissing_count,
                total_count=total_count,
                policy_strategy=workflow.missing_data_policy.strategy,
                status=status,
                code=code,
                message=message,
            )
        )
    return tuple(rows)


def _duplicate_rows_for_source(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    loaded_source: Mapping[str, Any],
) -> tuple[TabularDuplicateRow, ...]:
    source_rows = tuple(loaded_source["rows"])
    observed_set = set(loaded_source["observed_columns"])
    rows: list[TabularDuplicateRow] = []
    subject_key_columns = workflow.duplicate_subject_policy.key_columns or (source.schema.subject_id_column,)
    if all(column in observed_set for column in subject_key_columns):
        rows.extend(
            _duplicate_rows_for_key(
                workflow=workflow,
                source=source,
                source_rows=source_rows,
                key_type="subject",
                key_columns=subject_key_columns,
            )
        )
    session_key_columns = tuple(
        column for column in (source.schema.subject_id_column, source.schema.session_column, source.schema.timepoint_column) if column
    )
    if len(session_key_columns) > 1 and all(column in observed_set for column in session_key_columns):
        rows.extend(
            _duplicate_rows_for_key(
                workflow=workflow,
                source=source,
                source_rows=source_rows,
                key_type="subject_session_timepoint",
                key_columns=session_key_columns,
            )
        )
    repeated = workflow.repeated_measures
    if repeated is not None and repeated.source_id == source.source_id:
        repeated_key_columns = repeated.unit_columns or tuple(
            column
            for column in (repeated.subject_id_column, repeated.session_column, repeated.timepoint_column)
            if column
        )
        if repeated_key_columns and all(column in observed_set for column in repeated_key_columns):
            rows.extend(
                _duplicate_rows_for_key(
                    workflow=workflow,
                    source=source,
                    source_rows=source_rows,
                    key_type="repeated_unit",
                    key_columns=repeated_key_columns,
                )
            )
    return tuple(rows)


def _duplicate_rows_for_key(
    *,
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    source_rows: Sequence[Mapping[str, Any]],
    key_type: str,
    key_columns: Sequence[str],
) -> tuple[TabularDuplicateRow, ...]:
    row_numbers_by_key: dict[str, list[int]] = {}
    for row_number, row in enumerate(source_rows, start=1):
        values = tuple(row.get(column) for column in key_columns)
        if any(_is_missing_value(value) for value in values):
            continue
        key = _key_value_repr(values)
        row_numbers_by_key.setdefault(key, []).append(row_number)
    rows: list[TabularDuplicateRow] = []
    for key, row_numbers in row_numbers_by_key.items():
        if len(row_numbers) <= 1:
            continue
        status = "error" if workflow.duplicate_subject_policy.strategy == "error" else "warning"
        if workflow.duplicate_subject_policy.strategy == "allow":
            status = "ok"
        rows.append(
            TabularDuplicateRow(
                workflow_id=workflow.workflow_id,
                source_id=source.source_id,
                key_type=key_type,
                key_columns=key_columns,
                duplicate_key=key,
                duplicate_count=len(row_numbers),
                row_numbers=tuple(row_numbers),
                policy_strategy=workflow.duplicate_subject_policy.strategy,
                status=status,
                code=f"duplicate_{key_type}_key",
                message=f"Duplicate {key_type} key observed in source {source.source_id!r}.",
            )
        )
    return tuple(rows)


def _nonfinite_rows_for_source(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    loaded_source: Mapping[str, Any],
) -> tuple[TabularNonFiniteRow, ...]:
    source_rows = tuple(loaded_source["rows"])
    rows: list[TabularNonFiniteRow] = []
    for column in source.schema.columns:
        tokens: list[str] = []
        count = 0
        for row in source_rows:
            token = _nonfinite_token(row.get(column.column_name))
            if token is None:
                continue
            count += 1
            if token not in tokens:
                tokens.append(token)
        status = "ok"
        code = "nonfinite_values_absent"
        message = f"No non-finite values observed in column {column.column_name!r}."
        if count:
            if workflow.nonfinite_policy.strategy == "error":
                status = "error"
            elif workflow.nonfinite_policy.strategy == "allow":
                status = "ok"
            else:
                status = "warning"
            code = "nonfinite_values_observed"
            message = f"Observed {count} non-finite values in column {column.column_name!r}."
        rows.append(
            TabularNonFiniteRow(
                workflow_id=workflow.workflow_id,
                source_id=source.source_id,
                column_name=column.column_name,
                role=column.role,
                nonfinite_count=count,
                tokens=tokens,
                policy_strategy=workflow.nonfinite_policy.strategy,
                status=status,
                code=code,
                message=message,
            )
        )
    return tuple(rows)


def _categorical_qc_rows_for_source(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    loaded_source: Mapping[str, Any],
) -> tuple[TabularCategoricalQcRow, ...]:
    source_rows = tuple(loaded_source["rows"])
    policy = source.schema.categorical_validation
    rows: list[TabularCategoricalQcRow] = []
    allowed_by_column = policy.allowed_values
    for column in source.schema.columns:
        if not _is_categorical_declared(column) and column.column_name not in allowed_by_column:
            continue
        allowed_values = tuple(allowed_by_column.get(column.column_name, ()))
        observed_levels = _observed_levels(source_rows, column.column_name)
        unknown_levels = _unknown_categorical_levels(
            observed_levels=observed_levels,
            allowed_values=allowed_values,
            case_sensitive=policy.case_sensitive,
        )
        status = "ok"
        code = "categorical_levels_declared"
        message = f"Categorical levels for column {column.column_name!r} are QC-compatible."
        if policy.policy != "none" and allowed_values and unknown_levels:
            code = "categorical_unknown_levels"
            message = f"Observed unlisted categorical levels in column {column.column_name!r}."
            if policy.policy == "strict":
                status = "error"
            elif not policy.allow_unlisted:
                status = "warning"
            else:
                status = "ok"
        rows.append(
            TabularCategoricalQcRow(
                workflow_id=workflow.workflow_id,
                source_id=source.source_id,
                column_name=column.column_name,
                role=column.role,
                policy=policy.policy,
                allowed_values=allowed_values,
                allow_unlisted=policy.allow_unlisted,
                case_sensitive=policy.case_sensitive,
                observed_level_count=len(observed_levels),
                unknown_level_count=len(unknown_levels),
                unknown_levels=unknown_levels,
                status=status,
                code=code,
                message=message,
            )
        )
    return tuple(rows)


def _numeric_qc_rows_for_source(
    workflow: TabularAssociationWorkflowSpec,
    source: TabularSourceSpec,
    loaded_source: Mapping[str, Any],
) -> tuple[TabularNumericQcRow, ...]:
    source_rows = tuple(loaded_source["rows"])
    policy = source.schema.numeric_validation
    rows: list[TabularNumericQcRow] = []
    for column in source.schema.columns:
        if not _is_numeric_declared(column):
            continue
        min_value = _column_policy_number(column, "min_value", policy.min_value)
        max_value = _column_policy_number(column, "max_value", policy.max_value)
        integer_only = bool(column.metadata.get("integer_only", policy.integer_only))
        counts = _numeric_counts(
            source_rows,
            column_name=column.column_name,
            min_value=min_value,
            max_value=max_value,
            integer_only=integer_only,
        )
        issue_count = (
            counts["invalid_numeric_count"]
            + counts["below_min_count"]
            + counts["above_max_count"]
            + counts["noninteger_count"]
        )
        status = "ok"
        code = "numeric_values_valid"
        message = f"Numeric values for column {column.column_name!r} are QC-compatible."
        if issue_count:
            code = "numeric_values_invalid"
            message = f"Observed numeric QC issues in column {column.column_name!r}."
            status = "error" if policy.policy == "strict" else "warning"
        rows.append(
            TabularNumericQcRow(
                workflow_id=workflow.workflow_id,
                source_id=source.source_id,
                column_name=column.column_name,
                role=column.role,
                policy=policy.policy,
                min_value=min_value,
                max_value=max_value,
                integer_only=integer_only,
                total_count=counts["total_count"],
                missing_count=counts["missing_count"],
                valid_numeric_count=counts["valid_numeric_count"],
                invalid_numeric_count=counts["invalid_numeric_count"],
                bool_count=counts["bool_count"],
                nonfinite_count=counts["nonfinite_count"],
                below_min_count=counts["below_min_count"],
                above_max_count=counts["above_max_count"],
                noninteger_count=counts["noninteger_count"],
                status=status,
                code=code,
                message=message,
            )
        )
    return tuple(rows)


def _variables_for_source(
    workflow: TabularAssociationWorkflowSpec,
    source_id: str,
) -> tuple[AssociationVariableSpec, ...]:
    return tuple(
        variable
        for variable in (*workflow.outcomes, *workflow.predictors, *workflow.covariates, *workflow.groupings)
        if variable.source_id == source_id
    )


def _missingness_status(
    *,
    column: ColumnSpec,
    missing_count: int,
    total_count: int,
    policy: MissingDataPolicy,
) -> tuple[str, str, str]:
    del total_count
    if missing_count == 0:
        return "ok", "missing_values_absent", f"No missing values observed in column {column.column_name!r}."
    if not column.required:
        return "ok", "nullable_missing_values", f"Missing values are allowed for nullable column {column.column_name!r}."
    role = column.role or ""
    identifier_roles = {"subject_identifier", "session_identifier", "timepoint_identifier", "repeated_measures_identifier"}
    if policy.strategy == "allow":
        return "warning", "required_missing_values_allowed", f"Required column {column.column_name!r} has missing values."
    if policy.strategy == "error" or role in set(policy.required_roles) | identifier_roles:
        return "error", "required_missing_values", f"Required column {column.column_name!r} has missing values."
    return "warning", "missing_values_policy_deferred", f"Column {column.column_name!r} has missing values."


def _observed_columns_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    initial_columns: Sequence[str] = (),
) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for column in initial_columns:
        text = str(column)
        if text and text not in seen:
            columns.append(text)
            seen.add(text)
    for row in rows:
        for key in row.keys():
            text = str(key)
            if text and text not in seen:
                columns.append(text)
                seen.add(text)
    return tuple(columns)


def _is_missing_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _nonfinite_token(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"nan", "+nan", "-nan"}:
            return "nan"
        if token in {"inf", "+inf", "infinity", "+infinity"}:
            return "inf"
        if token in {"-inf", "-infinity"}:
            return "-inf"
    return None


def _key_value_repr(values: Sequence[Any]) -> str:
    return "|".join(_safe_value_repr(value) for value in values)


def _safe_value_repr(value: Any) -> str:
    token = _nonfinite_token(value)
    if token is not None:
        return token
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _observed_levels(rows: Sequence[Mapping[str, Any]], column_name: str) -> tuple[str, ...]:
    levels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get(column_name)
        if _is_missing_value(value):
            continue
        text = _safe_value_repr(value)
        if text not in seen:
            levels.append(text)
            seen.add(text)
    return tuple(levels)


def _unknown_categorical_levels(
    *,
    observed_levels: Sequence[str],
    allowed_values: Sequence[str],
    case_sensitive: bool,
) -> tuple[str, ...]:
    if not allowed_values:
        return ()
    if case_sensitive:
        allowed = set(allowed_values)
        return tuple(level for level in observed_levels if level not in allowed)
    allowed = {value.lower() for value in allowed_values}
    return tuple(level for level in observed_levels if level.lower() not in allowed)


def _is_numeric_declared(column: ColumnSpec) -> bool:
    return column.value_type.strip().lower() in {"numeric", "number", "float", "double", "integer", "int"}


def _is_categorical_declared(column: ColumnSpec) -> bool:
    return column.value_type.strip().lower() in {"categorical", "category", "string", "text", "bool", "boolean"}


def _column_policy_number(
    column: ColumnSpec,
    key: str,
    default: float | int | None,
) -> float | int | None:
    value = column.metadata.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool):
        return default
    if isinstance(value, (float, int)) and math.isfinite(value):
        return value
    return default


def _numeric_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    column_name: str,
    min_value: float | int | None,
    max_value: float | int | None,
    integer_only: bool,
) -> dict[str, int]:
    counts = {
        "total_count": len(rows),
        "missing_count": 0,
        "valid_numeric_count": 0,
        "invalid_numeric_count": 0,
        "bool_count": 0,
        "nonfinite_count": 0,
        "below_min_count": 0,
        "above_max_count": 0,
        "noninteger_count": 0,
    }
    for row in rows:
        value = row.get(column_name)
        if _is_missing_value(value):
            counts["missing_count"] += 1
            continue
        number_status, number = _finite_float(value)
        if number_status == "bool":
            counts["bool_count"] += 1
            counts["invalid_numeric_count"] += 1
            continue
        if number_status == "nonfinite":
            counts["nonfinite_count"] += 1
            counts["invalid_numeric_count"] += 1
            continue
        if number_status == "invalid":
            counts["invalid_numeric_count"] += 1
            continue
        counts["valid_numeric_count"] += 1
        if min_value is not None and number is not None and number < min_value:
            counts["below_min_count"] += 1
        if max_value is not None and number is not None and number > max_value:
            counts["above_max_count"] += 1
        if integer_only and number is not None and not float(number).is_integer():
            counts["noninteger_count"] += 1
    return counts


def _finite_float(value: Any) -> tuple[str, float | None]:
    if isinstance(value, bool):
        return "bool", None
    if _nonfinite_token(value) is not None:
        return "nonfinite", None
    if isinstance(value, (float, int)):
        return ("valid", float(value)) if math.isfinite(float(value)) else ("nonfinite", None)
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return "invalid", None
        return ("valid", number) if math.isfinite(number) else ("nonfinite", None)
    return "invalid", None


def _qc_result_messages(
    *,
    workflow_validation_rows: Sequence[AssociationValidationRow],
    source_inventory_rows: Sequence[TabularSourceInventoryRow],
    source_load_rows: Sequence[TabularSourceLoadRow],
    column_inventory_rows: Sequence[TabularColumnInventoryRow],
    schema_validation_rows: Sequence[TabularSchemaValidationRow],
    variable_qc_rows: Sequence[TabularVariableQcRow],
    missingness_rows: Sequence[TabularMissingnessRow],
    duplicate_rows: Sequence[TabularDuplicateRow],
    nonfinite_rows: Sequence[TabularNonFiniteRow],
    categorical_qc_rows: Sequence[TabularCategoricalQcRow],
    numeric_qc_rows: Sequence[TabularNumericQcRow],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    errors: list[str] = []

    def add(status: str, message: str) -> None:
        if not message:
            return
        if status == "error":
            errors.append(message)
        elif status == "warning":
            warnings.append(message)

    for row in workflow_validation_rows:
        add(row.status, row.message)
    for row in source_inventory_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in source_load_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in column_inventory_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in (
        *schema_validation_rows,
        *variable_qc_rows,
        *missingness_rows,
        *duplicate_rows,
        *nonfinite_rows,
        *categorical_qc_rows,
        *numeric_qc_rows,
    ):
        add(row.status, row.message)
    return tuple(warnings), tuple(errors)


def _qc_provenance_rows(
    workflow: TabularAssociationWorkflowSpec,
    *,
    plan_only: bool,
    source_count: int,
    loaded_source_count: int,
) -> tuple[TabularAssociationQcProvenanceRow, ...]:
    return (
        TabularAssociationQcProvenanceRow(workflow_id=workflow.workflow_id, key="schema_version", value=SCHEMA_VERSION),
        TabularAssociationQcProvenanceRow(workflow_id=workflow.workflow_id, key="workflow_id", value=workflow.workflow_id),
        TabularAssociationQcProvenanceRow(workflow_id=workflow.workflow_id, key="backend", value=workflow.backend),
        TabularAssociationQcProvenanceRow(workflow_id=workflow.workflow_id, key="runtime_backend", value=RUNTIME_BACKEND_RECORDS),
        TabularAssociationQcProvenanceRow(workflow_id=workflow.workflow_id, key="source_count", value=source_count),
        TabularAssociationQcProvenanceRow(workflow_id=workflow.workflow_id, key="loaded_source_count", value=loaded_source_count),
        TabularAssociationQcProvenanceRow(workflow_id=workflow.workflow_id, key="executed", value=False),
        TabularAssociationQcProvenanceRow(workflow_id=workflow.workflow_id, key="plan_only", value=plan_only),
        TabularAssociationQcProvenanceRow(workflow_id=workflow.workflow_id, key="will_write", value=False),
        TabularAssociationQcProvenanceRow(workflow_id=workflow.workflow_id, key="output_written", value=False),
    )


def _qc_provenance_rows_for_error(workflow_id: str) -> tuple[TabularAssociationQcProvenanceRow, ...]:
    return (
        TabularAssociationQcProvenanceRow(workflow_id=workflow_id, key="schema_version", value=SCHEMA_VERSION),
        TabularAssociationQcProvenanceRow(workflow_id=workflow_id, key="workflow_id", value=workflow_id),
        TabularAssociationQcProvenanceRow(workflow_id=workflow_id, key="runtime_backend", value=RUNTIME_BACKEND_RECORDS),
        TabularAssociationQcProvenanceRow(workflow_id=workflow_id, key="executed", value=False),
        TabularAssociationQcProvenanceRow(workflow_id=workflow_id, key="will_write", value=False),
        TabularAssociationQcProvenanceRow(workflow_id=workflow_id, key="output_written", value=False),
    )


def _source_from_mapping(value: Mapping[str, Any]) -> TabularSourceSpec:
    mapping = _as_mapping(value, field_name="source")
    schema_doc = _as_mapping(mapping.get("schema", {}), field_name="schema")
    merged_schema = dict(schema_doc)
    for key in ("subject_id_column", "subject_column", "session_column", "timepoint_column", "columns"):
        if key in mapping and key not in merged_schema:
            merged_schema[key] = mapping[key]
    return TabularSourceSpec(
        source_id=_first_present(mapping, "source_id", "id", default=""),
        format=_first_present(mapping, "format", "source_format"),
        path=_first_present(mapping, "path", "source_path"),
        root_ref=_first_present(mapping, "root_ref", "source_root_ref"),
        backend=_first_present(mapping, "backend", default=BACKEND_RECORDS),
        schema=_schema_from_mapping(merged_schema),
        metadata=_as_metadata(mapping.get("metadata", {})),
    )


def _schema_from_mapping(value: Mapping[str, Any]) -> TabularSchemaSpec:
    mapping = _as_mapping(value, field_name="schema")
    return TabularSchemaSpec(
        subject_id_column=_first_present(mapping, "subject_id_column", "subject_column", default=""),
        session_column=_first_present(mapping, "session_column", "session_id_column"),
        timepoint_column=_first_present(mapping, "timepoint_column", "time_column"),
        columns=tuple(_coerce_column_spec(item) for item in _column_items(mapping.get("columns", ()))),
        categorical_validation=_policy_from_mapping(mapping.get("categorical_validation"), CategoricalValidationPolicy),
        numeric_validation=_policy_from_mapping(mapping.get("numeric_validation"), NumericValidationPolicy),
        metadata=_as_metadata(mapping.get("metadata", {})),
    )


def _coerce_schema_spec(value: Mapping[str, Any] | TabularSchemaSpec) -> TabularSchemaSpec:
    if isinstance(value, TabularSchemaSpec):
        return value
    return _schema_from_mapping(value)


def _coerce_column_spec(value: Mapping[str, Any] | ColumnSpec | str) -> ColumnSpec:
    if isinstance(value, ColumnSpec):
        return value
    if isinstance(value, str):
        return ColumnSpec(column_name=value)
    mapping = _as_mapping(value, field_name="column")
    return ColumnSpec(
        column_name=_first_present(mapping, "column_name", "name", "column", default=""),
        value_type=_first_present(mapping, "value_type", "type", "data_type", default="unspecified"),
        role=_first_present(mapping, "role"),
        required=bool(_first_present(mapping, "required", default=True)),
        metadata=_as_metadata(mapping.get("metadata", {})),
    )


def _variable_from_mapping(value: Mapping[str, Any], spec_type: type[AssociationVariableSpec]) -> AssociationVariableSpec:
    mapping = _as_mapping(value, field_name="variable")
    column_name = _first_present(mapping, "column_name", "column", default="")
    return spec_type(
        variable_id=_first_present(mapping, "variable_id", "id", default=column_name),
        source_id=_first_present(mapping, "source_id", "source", default=""),
        column_name=column_name,
        label=_first_present(mapping, "label", "name"),
        metadata=_as_metadata(mapping.get("metadata", {})),
    )


def _repeated_measures_from_mapping(value: Mapping[str, Any]) -> RepeatedMeasuresSpec:
    mapping = _as_mapping(value, field_name="repeated_measures")
    if _truthy_forbidden_execution(mapping):
        raise ValueError("Repeated-measures declarations must remain planned-only and non-executable.")
    metadata = dict(_as_metadata(mapping.get("metadata", {})))
    for metadata_key in _REPEATED_MEASURES_METADATA_KEYS:
        if metadata_key in mapping and metadata_key not in metadata:
            metadata[metadata_key] = _json_safe(mapping[metadata_key])
    return RepeatedMeasuresSpec(
        source_id=_first_present(mapping, "source_id", "source", default=""),
        subject_id_column=_first_present(mapping, "subject_id_column", "subject_column", default=""),
        session_column=_first_present(mapping, "session_column", "session_id_column"),
        timepoint_column=_first_present(mapping, "timepoint_column", "time_column"),
        unit_columns=_sequence_from_mapping(mapping, "unit_columns", "repeated_unit_columns", "identifier_columns"),
        metadata=metadata,
    )


def _method_from_mapping(value: Mapping[str, Any]) -> AssociationMethodSpec:
    mapping = _as_mapping(value, field_name="method")
    if _truthy_forbidden_execution(mapping):
        raise ValueError("Association method declarations must remain planned-only and non-executable.")
    method_name = _first_present(mapping, "method_name", "method", default=_first_present(mapping, "correlation", default=""))
    normalized = _normalized_choice(method_name, field_name="association method", supported=SUPPORTED_ASSOCIATION_METHODS)
    metadata = dict(_as_metadata(mapping.get("metadata", {})))
    for deferred_key in (
        "partial",
        "adjusted",
        "adjustment",
        "adjustment_ids",
        "stratification",
        "stratify",
        "strata",
        "stratification_ids",
        "repeated",
        "repeated_measures",
        "repeated_measures_ids",
        "mixed",
        "mixed_model",
        "random_effects",
        *_REPEATED_MEASURES_METADATA_KEYS,
    ):
        if deferred_key in mapping and deferred_key not in metadata:
            metadata[deferred_key] = _json_safe(mapping[deferred_key])
    kwargs = {
        "method_id": _first_present(mapping, "method_id", "id", default=""),
        "method_name": normalized,
        "outcome_ids": _sequence_from_mapping(mapping, "outcome_ids", "outcomes", "outcome_id"),
        "predictor_ids": _sequence_from_mapping(mapping, "predictor_ids", "predictors", "predictor_id"),
        "covariate_ids": _sequence_from_mapping(mapping, "covariate_ids", "covariates", "covariate_id"),
        "grouping_ids": _sequence_from_mapping(mapping, "grouping_ids", "groupings", "grouping_id"),
        "family_id": _first_present(mapping, "family_id", "family"),
        "output_id": _first_present(mapping, "output_id", "output"),
        "metadata": metadata,
    }
    if normalized in {METHOD_PEARSON, METHOD_SPEARMAN}:
        return CorrelationSpec(**kwargs)
    if normalized == METHOD_PARTIAL_CORRELATION:
        return PartialCorrelationSpec(**kwargs)
    if normalized == METHOD_REGRESSION:
        return RegressionAssociationSpec(**kwargs)
    return RepeatedMeasuresAssociationSpec(
        **kwargs,
        deferred_reason=_first_present(mapping, "deferred_reason", default="repeated-measures execution is deferred"),
    )


def _family_from_mapping(value: Mapping[str, Any]) -> AssociationFamilySpec:
    mapping = _as_mapping(value, field_name="family")
    return AssociationFamilySpec(
        family_id=_first_present(mapping, "family_id", "id", default=""),
        method_ids=_sequence_from_mapping(mapping, "method_ids", "methods", "method_id"),
        description=_first_present(mapping, "description"),
        metadata=_as_metadata(mapping.get("metadata", {})),
    )


def _multiple_testing_from_mapping(value: Mapping[str, Any]) -> MultipleTestingSpec:
    mapping = _as_mapping(value, field_name="multiple_testing")
    if _truthy_forbidden_execution(mapping):
        raise ValueError("Multiple-testing declarations must remain planned-only.")
    return MultipleTestingSpec(
        family_id=_first_present(mapping, "family_id", "id", default=""),
        method=_first_present(mapping, "method", "correction", default="benjamini_hochberg"),
        metadata=_as_metadata(mapping.get("metadata", {})),
    )


def _output_from_mapping(value: Mapping[str, Any]) -> AssociationOutputSpec:
    mapping = _as_mapping(value, field_name="output")
    if _truthy_forbidden_write(mapping):
        raise ValueError("Output declarations are planned metadata only and cannot request writes.")
    return AssociationOutputSpec(
        output_id=_first_present(mapping, "output_id", "id", default=""),
        output_type=_first_present(mapping, "output_type", "type", default="association_results"),
        planned_fields=_sequence_from_mapping(mapping, "planned_fields", "fields"),
        source_method_ids=_sequence_from_mapping(mapping, "source_method_ids", "method_ids", "methods"),
        family_ids=_sequence_from_mapping(mapping, "family_ids", "families", "family_id"),
        metadata=_as_metadata(mapping.get("metadata", {})),
    )


def _handoff_from_mapping(value: Mapping[str, Any]) -> AssociationHandoffSpec:
    mapping = _as_mapping(value, field_name="handoff")
    if _truthy_forbidden_execution(mapping) or _truthy_forbidden_write(mapping):
        raise ValueError("Handoff declarations are metadata-only and cannot request execution or writes.")
    return AssociationHandoffSpec(
        handoff_id=_first_present(mapping, "handoff_id", "id", default=""),
        handoff_type=_first_present(mapping, "handoff_type", "type", default=""),
        output_ids=_sequence_from_mapping(mapping, "output_ids", "outputs", "output_id"),
        target=_first_present(mapping, "target", "adapter"),
        planned_fields=_sequence_from_mapping(mapping, "planned_fields", "fields"),
        metadata=_as_metadata(mapping.get("metadata", {})),
    )


def _policy_from_mapping(value: Any, spec_type: type[Any]) -> Any:
    if value is None:
        return spec_type()
    if isinstance(value, spec_type):
        return value
    if isinstance(value, str):
        if spec_type in {MissingDataPolicy, DuplicateSubjectPolicy, NonFinitePolicy}:
            return spec_type(strategy=value)
        if spec_type in {CategoricalValidationPolicy, NumericValidationPolicy}:
            return spec_type(policy=value)
        return spec_type(method=value)
    mapping = _as_mapping(value, field_name=spec_type.__name__)
    return spec_type(**mapping)


def _column_items(value: Any) -> tuple[Mapping[str, Any] | ColumnSpec | str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, ColumnSpec)):
        return (value,)
    if isinstance(value, Mapping):
        if any(key in value for key in ("column_name", "name", "column")):
            return (value,)
        items: list[Mapping[str, Any]] = []
        for key, item in value.items():
            item_mapping = dict(_as_mapping(item, field_name="column"))
            item_mapping.setdefault("column_name", key)
            items.append(item_mapping)
        return tuple(items)
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError("columns must be a mapping or sequence of column declarations.") from exc


def _coerce_source_spec(value: Mapping[str, Any] | TabularSourceSpec) -> TabularSourceSpec:
    if isinstance(value, TabularSourceSpec):
        return value
    return _source_from_mapping(value)


def _coerce_variable_spec(
    value: Mapping[str, Any] | AssociationVariableSpec,
    spec_type: type[AssociationVariableSpec],
) -> AssociationVariableSpec:
    if isinstance(value, spec_type):
        return value
    return _variable_from_mapping(value, spec_type)


def _coerce_method_spec(value: Mapping[str, Any] | AssociationMethodSpec) -> AssociationMethodSpec:
    if isinstance(value, AssociationMethodSpec):
        return value
    return _method_from_mapping(value)


def _coerce_family_spec(value: Mapping[str, Any] | AssociationFamilySpec) -> AssociationFamilySpec:
    if isinstance(value, AssociationFamilySpec):
        return value
    return _family_from_mapping(value)


def _coerce_multiple_testing_spec(value: Mapping[str, Any] | MultipleTestingSpec) -> MultipleTestingSpec:
    if isinstance(value, MultipleTestingSpec):
        return value
    return _multiple_testing_from_mapping(value)


def _coerce_output_spec(value: Mapping[str, Any] | AssociationOutputSpec) -> AssociationOutputSpec:
    if isinstance(value, AssociationOutputSpec):
        return value
    return _output_from_mapping(value)


def _coerce_handoff_spec(value: Mapping[str, Any] | AssociationHandoffSpec) -> AssociationHandoffSpec:
    if isinstance(value, AssociationHandoffSpec):
        return value
    return _handoff_from_mapping(value)


def _mapping_or_sequence_items(value: Any, id_field: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        if any(key in value for key in (id_field, "id", "column_name", "method", "method_name")):
            return (_as_mapping(value, field_name=id_field),)
        items: list[Mapping[str, Any]] = []
        for key, item in value.items():
            item_mapping = dict(_as_mapping(item, field_name=id_field))
            item_mapping.setdefault(id_field, key)
            items.append(item_mapping)
        return tuple(items)
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{id_field} declarations must be mappings, not strings.")
    try:
        return tuple(_as_mapping(item, field_name=id_field) for item in value)
    except TypeError as exc:
        raise TypeError(f"{id_field} declarations must be a mapping or sequence of mappings.") from exc


def _sequence_from_mapping(mapping: Mapping[str, Any], *keys: str) -> tuple[str, ...]:
    value = _first_present(mapping, *keys)
    if value is None:
        return ()
    return _text_tuple(value, field_name=keys[0])


def _with_default(mapping: Mapping[str, Any], key: str, value: Any) -> Mapping[str, Any]:
    result = dict(mapping)
    result.setdefault(key, value)
    return result


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    return value


def _as_metadata(value: Any) -> Mapping[str, Any]:
    return _json_safe_mapping(_as_mapping(value, field_name="metadata"))


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _text_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (_non_empty_text(value, field_name=field_name),)
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a string or a sequence of strings.") from exc
    return tuple(_non_empty_text(item, field_name=field_name) for item in values)


def _normalized_choice(value: Any, *, field_name: str, supported: frozenset[str]) -> str:
    text = _non_empty_text(value, field_name=field_name).lower()
    if text not in supported:
        supported_values = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported {field_name} {text!r}. Use one of: {supported_values}.")
    return text


def _non_empty_text(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite_number(value: float | int, *, field_name: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite numeric value.")
    return value


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _truthy_forbidden_execution(mapping: Mapping[str, Any]) -> bool:
    return any(bool(mapping.get(key)) for key in ("execute", "executable", "run", "fit", "compute"))


def _truthy_forbidden_write(mapping: Mapping[str, Any]) -> bool:
    return any(bool(mapping.get(key)) for key in ("write", "will_write", "output_written", "render"))


def _best_effort_workflow_id(document: Any) -> str:
    if isinstance(document, TabularAssociationWorkflowSpec):
        return document.workflow_id
    if isinstance(document, Mapping):
        value = _first_present(document, "workflow_id", "id", "name")
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return "unparsed-workflow"


def _json_safe_dataclass(instance: object) -> dict[str, Any]:
    if not is_dataclass(instance):
        raise TypeError("_json_safe_dataclass requires a dataclass instance.")
    return {field_.name: _json_safe(getattr(instance, field_.name)) for field_ in fields(instance)}


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(item) for key, item in value.items()}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Tabular association schema values cannot contain non-finite floats.")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _tsv_safe_mapping(value: Mapping[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, item in value.items():
        if item is None:
            row[str(key)] = ""
        elif isinstance(item, bool):
            row[str(key)] = "true" if item else "false"
        elif isinstance(item, (str, int, float)):
            row[str(key)] = str(_json_safe(item))
        else:
            row[str(key)] = json.dumps(_json_safe(item), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return row


__all__ = [
    "AdjustedAssociationComputationQcRow",
    "AdjustedAssociationMethodSummaryRow",
    "AdjustedAssociationPairPlanRow",
    "AdjustedAssociationResultRow",
    "AssociationFamilySpec",
    "AssociationHandoffSpec",
    "AssociationInputQcSummaryRow",
    "AssociationMethodSpec",
    "AssociationPublicationInputSummaryRow",
    "AssociationPublicationManifestRow",
    "AssociationPublicationMissingnessTableRow",
    "AssociationPublicationMultiplicityTableRow",
    "AssociationPublicationProvenanceRow",
    "AssociationPublicationQcTableRow",
    "AssociationPublicationTableRow",
    "AssociationMultiplicityFamilyPlanRow",
    "AssociationMultiplicityInputRow",
    "AssociationMultiplicityMethodSummaryRow",
    "AssociationMultiplicityQcRow",
    "AssociationMultiplicityResultRow",
    "AssociationPairPlanRow",
    "AssociationOutputSpec",
    "AssociationPlanPreview",
    "AssociationProvenanceRow",
    "AssociationValidationRow",
    "AssociationVariableSpec",
    "BACKEND_PANDAS",
    "BACKEND_POLARS",
    "BACKEND_RECORDS",
    "BetweenSubjectFactorSpec",
    "CategoricalValidationPolicy",
    "CategoricalCodingSpec",
    "ColumnSpec",
    "ClusterTermSpec",
    "ContrastMetadataSpec",
    "CorrelationAssociationResultRow",
    "CorrelationComputationQcRow",
    "CorrelationMethodSummaryRow",
    "CorrelationSpec",
    "CovariateSpec",
    "DuplicateSubjectPolicy",
    "METHOD_MIXED_MODEL",
    "METHOD_PARTIAL_CORRELATION",
    "METHOD_PEARSON",
    "METHOD_REGRESSION",
    "METHOD_REPEATED_MEASURES",
    "METHOD_SPEARMAN",
    "MODEL_RESULT_KIND_CONTRAST",
    "MODEL_RESULT_KIND_FIXED_EFFECT",
    "MODEL_RESULT_KIND_MODEL_FIT_SUMMARY",
    "MODEL_RESULT_KIND_PLANNED_COMPARISON",
    "MODEL_RESULT_KIND_RANDOM_EFFECT",
    "MODEL_RESULT_KIND_VARIANCE_COMPONENT",
    "MissingDataPolicy",
    "FixedEffectTermSpec",
    "GroupingFactorSpec",
    "ModelContrastResultRow",
    "ModelDesignMetadataSpec",
    "ModelFitSummaryRow",
    "ModelFixedEffectResultRow",
    "ModelFormulaMetadataSpec",
    "ModelPlannedComparisonResultRow",
    "ModelRandomEffectResultRow",
    "ModelResultProvenanceRow",
    "ModelResultQcRow",
    "ModelVarianceComponentResultRow",
    "MultipleTestingSpec",
    "NonFinitePolicy",
    "NumericValidationPolicy",
    "OutcomeSpec",
    "PartialCorrelationSpec",
    "PlannedComparisonSpec",
    "PredictorSpec",
    "RandomEffectTermSpec",
    "RandomInterceptSpec",
    "RandomSlopeSpec",
    "RegressionAssociationSpec",
    "RegressionAssociationResultRow",
    "RepeatedFactorSpec",
    "RepeatedMeasuresDesignQcRow",
    "RepeatedMeasuresDesignSummaryRow",
    "RepeatedMeasuresFactorSummaryRow",
    "RepeatedMeasuresModelPlanRow",
    "RepeatedMeasuresAssociationSpec",
    "RepeatedMeasuresSpec",
    "SCHEMA_VERSION",
    "SUPPORTED_ASSOCIATION_METHODS",
    "SUPPORTED_MODEL_RESULT_KINDS",
    "SUPPORTED_MULTIPLE_TESTING_METHODS",
    "SUPPORTED_P_VALUE_POLICIES",
    "SUPPORTED_SOURCE_INVENTORY_FORMATS",
    "SUPPORTED_TABULAR_ASSOCIATION_BACKENDS",
    "SOURCE_KIND_CSV",
    "SOURCE_KIND_IN_MEMORY",
    "SOURCE_KIND_JSON",
    "SOURCE_KIND_MISSING",
    "SOURCE_KIND_TSV",
    "SOURCE_KIND_UNSUPPORTED",
    "StandardizationPolicy",
    "TabularAssociationAdjustedPlan",
    "TabularAssociationAdjustedProvenanceRow",
    "TabularAssociationAdjustedResult",
    "TabularAssociationMultiplicityPlan",
    "TabularAssociationMultiplicityProvenanceRow",
    "TabularAssociationMultiplicityResult",
    "TabularAssociationModelResultContract",
    "TabularAssociationModelResultPlan",
    "TabularAssociationModelResultValidationResult",
    "TabularAssociationPublicationPlan",
    "TabularAssociationPublicationResult",
    "TabularAssociationQcPlan",
    "TabularAssociationQcProvenanceRow",
    "TabularAssociationQcResult",
    "TabularAssociationRepeatedMeasuresDesignQcResult",
    "TabularAssociationRepeatedMeasuresPlan",
    "TabularAssociationRepeatedMeasuresProvenanceRow",
    "TabularAssociationRecordsAdapter",
    "TabularAssociationCorrelationPlan",
    "TabularAssociationCorrelationProvenanceRow",
    "TabularAssociationCorrelationResult",
    "TabularAssociationRowSourceAdapterSpec",
    "TabularAssociationRowSourceProvenanceRow",
    "TabularAssociationRowSourceQcRow",
    "TabularAssociationRowSourceResult",
    "TabularAssociationWorkflowSpec",
    "TabularCategoricalQcRow",
    "TabularColumnInventoryRow",
    "TabularDuplicateRow",
    "TabularMissingnessRow",
    "TabularNonFiniteRow",
    "TabularNumericQcRow",
    "TabularSchemaValidationRow",
    "TabularSchemaSpec",
    "TabularSourceInventoryRow",
    "TabularSourceInventorySpec",
    "TabularSourceLoadRow",
    "TabularSourceSpec",
    "TabularVariableQcRow",
    "TransformationPolicy",
    "TimepointRoleSpec",
    "GroupingSpec",
    "TABULAR_ASSOCIATION_PUBLICATION_HANDOFF_VERSION",
    "TABULAR_ASSOCIATION_MODEL_RESULTS_CONTRACT_VERSION",
    "TABULAR_ASSOCIATION_REPEATED_MEASURES_METADATA_VERSION",
    "TABULAR_ASSOCIATION_REPEATED_MEASURES_PLAN_VERSION",
    "TABULAR_ASSOCIATION_ROW_SOURCE_ADAPTER_VERSION",
    "WithinSubjectFactorSpec",
    "build_tabular_association_publication_tables",
    "coerce_tabular_association_records",
    "inspect_tabular_association_row_source",
    "iter_tabular_association_records",
    "normalize_tabular_association_model_result_rows",
    "parse_tabular_association_workflow_document",
    "plan_tabular_association_adjusted",
    "plan_tabular_association_correlations",
    "plan_tabular_association_multiplicity",
    "plan_tabular_association_model_results",
    "plan_tabular_association_publication_tables",
    "plan_tabular_association_qc",
    "plan_tabular_association_repeated_measures",
    "plan_tabular_association_row_source_adapter",
    "plan_tabular_association_workflow",
    "run_tabular_association_adjusted",
    "run_tabular_association_correlations",
    "run_tabular_association_multiplicity",
    "run_tabular_association_qc",
    "run_tabular_association_repeated_measures_design_qc",
    "validate_tabular_association_model_result_rows",
    "validate_tabular_association_workflow_document",
]
