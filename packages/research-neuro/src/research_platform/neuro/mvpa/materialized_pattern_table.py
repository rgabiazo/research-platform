"""Plan and load deterministic, pre-materialized MVPA feature tables.

Planning is deliberately lightweight: it streams TSV rows, validates scalar
metadata, joins them to exact analysis units, and hashes the source bytes
without decoding feature vectors.  The loader is a separate, digest-checked,
all-or-nothing in-memory boundary that decodes and validates vectors without
importing ``research-analysis`` or writing outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
import csv
import json
import math
import re

from research_platform.neuro._roi_path_safety import (
    configured_path_is_unsafe,
    published_text_contains_local_path_reference,
)

from .pattern_sources import (
    PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE,
    REPRESENTATION_PREPARED_FEATURES,
    PlannedPatternRow,
    ResolvedAnalysisUnit,
)


SCHEMA_VERSION = "research_platform.neuro.mvpa.materialized_pattern_table.v1"

REQUIRED_COLUMNS = (
    "schema_version",
    "pattern_id",
    "subject_id",
    "condition_id",
    "pattern_source_name",
    "roi_source_name",
    "roi_label",
    "feature_count",
    "voxel_order",
    "voxel_index_hash",
    "feature_space_id",
    "roi_definition_id",
    "feature_values",
    "usable",
    "status",
    "mean_centering_applied",
    "mean_centering_scope",
    "noise_status",
    "noise_usable",
)

OPTIONAL_COLUMNS = (
    "session_id",
    "task_id",
    "run_id",
    "cross_validation_label",
    "event_count",
    "qc_status",
    "qc_reason",
    "exclusion_id",
    "exclusion_reason",
    "grouping_values",
    "warnings",
    "errors",
    "roi_reference",
    "generator_version",
    "software_version",
    "derivation_id",
    "holdout_id",
    "noise_values",
    "noise_feature_count",
    "noise_voxel_order",
    "noise_voxel_index_hash",
    "noise_feature_space_id",
    "noise_roi_definition_id",
    "noise_value_kind",
    "noise_estimation_scope",
    "noise_source",
)

FIXED_COLUMNS = frozenset((*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS))
_VECTOR_COLUMNS = frozenset({"feature_values", "noise_values"})
_SAFE_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_PARENT_REFERENCE = re.compile(r"(^|[=:\s,;\"'\\/])\.\.[\\/]")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_TRUE_VALUES = frozenset({"true", "1"})
_FALSE_VALUES = frozenset({"false", "0"})
_USABLE_STATUSES = frozenset({"ok", "warning", "usable"})
_UNUSABLE_STATUSES = frozenset({"excluded", "error", "failed", "skipped", "unusable"})
_SOURCE_KEYS = frozenset({"name", "backend", "root_ref", "path", "schema_version"})
_DERIVED_RESERVED_COLUMNS = frozenset(
    {
        "backend",
        "backend_metadata",
        "backend_name",
        "data_row",
        "executed",
        "materialized",
        "noise_reference",
        "pattern_reference",
        "representation_kind",
        "source_name",
        "source_reference",
        "table_data_row",
        "table_sha256",
        "unit_id",
        "unit_metadata",
    }
)


@dataclass(frozen=True)
class MaterializedPatternScalarRow:
    """Vector-free scalar metadata retained by the streaming planner."""

    data_row: int
    pattern_id: str
    unit_id: str
    subject_id: str
    session_id: str | None
    task_id: str | None
    run_id: str | None
    cross_validation_label: str
    condition_id: str
    pattern_source_name: str
    roi_source_name: str
    roi_label: str
    feature_count: int
    voxel_order: str
    voxel_index_hash: str
    feature_space_id: str
    roi_definition_id: str
    usable: bool
    status: str
    mean_centering_applied: bool
    mean_centering_scope: str
    noise_status: str
    noise_usable: bool
    event_count: int | None
    qc_status: str | None
    qc_reason: str | None
    exclusion_id: str | None
    exclusion_reason: str | None
    roi_reference: str | None
    generator_version: str | None
    software_version: str | None
    derivation_id: str | None
    holdout_id: str | None
    unit_metadata: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializedPatternTablePlan:
    """Private load handle plus portable, JSON-safe table plan metadata."""

    source_name: str
    root_ref: str | None
    relative_path: str | None
    portable_reference: str | None
    source_sha256: str | None
    columns: tuple[str, ...] = ()
    key_columns: tuple[str, ...] = ()
    scalar_rows: tuple[MaterializedPatternScalarRow, ...] = ()
    pattern_rows: tuple[PlannedPatternRow, ...] = ()
    total_row_count: int = 0
    selected_row_count: int = 0
    unselected_row_count: int = 0
    unselected_pattern_ids: tuple[str, ...] = ()
    event_threshold_rows: tuple[Mapping[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    _local_path: Path | None = field(default=None, repr=False, compare=False)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def ready_for_materialization(self) -> bool:
        return self.valid and bool(self.scalar_rows) and self._local_path is not None

    @property
    def usable_selected_row_count(self) -> int:
        """Return the number of selected rows eligible for analysis."""

        return sum(row.usable for row in self.scalar_rows)

    @property
    def usable_unit_count(self) -> int:
        """Return the number of exact units represented by usable rows."""

        return len({row.unit_id for row in self.scalar_rows if row.usable})

    @property
    def usable_condition_count(self) -> int:
        """Return the number of configured conditions represented by usable rows."""

        return len({row.condition_id for row in self.scalar_rows if row.usable})

    @property
    def usable_roi_count(self) -> int:
        """Return the number of ROI identities represented by usable rows."""

        return len(
            {
                (row.roi_source_name, row.roi_label)
                for row in self.scalar_rows
                if row.usable
            }
        )

    @property
    def usable_coverage_complete(self) -> bool:
        """Return whether every required exact-unit/condition/ROI row is usable."""

        return bool(self.scalar_rows) and self.usable_selected_row_count == len(self.scalar_rows)

    @property
    def ready_for_execution(self) -> bool:
        """Return whether scalar planning and every configured threshold passed."""

        return self.ready_for_materialization and self.usable_coverage_complete and all(
            str(row.get("status")) == "passed" for row in self.event_threshold_rows
        )

    def public_summary(self) -> dict[str, Any]:
        """Return the portable plan summary without the runtime path."""

        return {
            "schema_version": SCHEMA_VERSION,
            "source_name": self.source_name,
            "root_ref": self.root_ref,
            "relative_path": self.relative_path,
            "portable_reference": self.portable_reference,
            "source_sha256": self.source_sha256,
            "columns": list(self.columns),
            "key_columns": list(self.key_columns),
            "counts": {
                "total_rows": self.total_row_count,
                "selected_rows": self.selected_row_count,
                "usable_selected_rows": self.usable_selected_row_count,
                "usable_units": self.usable_unit_count,
                "usable_conditions": self.usable_condition_count,
                "usable_rois": self.usable_roi_count,
                "unselected_rows": self.unselected_row_count,
            },
            "unselected_pattern_ids": list(self.unselected_pattern_ids),
            "event_threshold_rows": [dict(row) for row in self.event_threshold_rows],
            "usable_coverage_complete": self.usable_coverage_complete,
            "ready_for_materialization": self.ready_for_materialization,
            "ready_for_execution": self.ready_for_execution,
            "execution_reason": (
                "scalar_plan_usable_coverage_and_event_thresholds_ready"
                if self.ready_for_execution
                else "scalar_plan_usable_coverage_or_event_thresholds_not_ready"
            ),
            "executed": False,
        }

    def public_source_rows(self) -> tuple[Mapping[str, Any], ...]:
        """Return vector-free, portable scalar rows in resolved plan order."""

        return tuple(
            {
                "data_row": row.data_row,
                "pattern_id": row.pattern_id,
                "unit_id": row.unit_id,
                "subject_id": row.subject_id,
                "session_id": row.session_id,
                "task_id": row.task_id,
                "run_id": row.run_id,
                "cross_validation_label": row.cross_validation_label,
                "condition_id": row.condition_id,
                "pattern_source_name": row.pattern_source_name,
                "roi_source_name": row.roi_source_name,
                "roi_label": row.roi_label,
                "feature_count": row.feature_count,
                "voxel_order": row.voxel_order,
                "voxel_index_hash": row.voxel_index_hash,
                "feature_space_id": row.feature_space_id,
                "roi_definition_id": row.roi_definition_id,
                "usable": row.usable,
                "status": row.status,
                "mean_centering_applied": row.mean_centering_applied,
                "mean_centering_scope": row.mean_centering_scope,
                "noise_status": row.noise_status,
                "noise_usable": row.noise_usable,
                "event_count": row.event_count,
                "qc_status": row.qc_status,
                "qc_reason": row.qc_reason,
                "exclusion_id": row.exclusion_id,
                "exclusion_reason": row.exclusion_reason,
                "roi_reference": row.roi_reference,
                "generator_version": row.generator_version,
                "software_version": row.software_version,
                "derivation_id": row.derivation_id,
                "holdout_id": row.holdout_id,
                "unit_metadata": dict(row.unit_metadata),
                "source_reference": f"{self.portable_reference}#row={row.data_row}",
                **_unit_metadata_columns(row.unit_metadata),
                **dict(row.extras),
            }
            for row in self.scalar_rows
        )


@dataclass(frozen=True)
class MaterializedPatternTableLoadResult:
    """Strict all-or-nothing in-memory materialization result."""

    rows: tuple[Mapping[str, Any], ...] = ()
    qc_rows: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    materialized: bool = False
    executed: bool = False

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_materialized_pattern_source_fields(
    source: Mapping[str, Any],
    label: str,
) -> tuple[str, ...]:
    """Validate the fixed v1 materialized-table declaration without I/O."""

    errors: list[str] = []
    name = _text(source.get("name"))
    backend = _text(source.get("backend"))
    root_ref = _text(source.get("root_ref"))
    path = _text(source.get("path"))
    version = _text(source.get("schema_version"))
    if name is None:
        errors.append(f"{label}.name must be defined for a materialized pattern table.")
    elif not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        errors.append(f"{label}.name must be a safe identifier.")
    if backend != PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE:
        errors.append(
            f"{label}.backend must be {PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE!r}."
        )
    if root_ref is None:
        errors.append(f"{label}.root_ref must be defined for a materialized pattern table.")
    elif not re.fullmatch(r"[A-Za-z0-9_.-]+", root_ref):
        errors.append(f"{label}.root_ref must be a safe named-root identifier.")
    if path is None:
        errors.append(f"{label}.path must be defined for a materialized pattern table.")
    elif not _configured_source_path_is_safe(path):
        errors.append(f"{label}.path must be a safe relative path beneath root_ref.")
    elif Path(path).suffix.casefold() != ".tsv":
        errors.append(f"{label}.path must reference one fixed TSV file.")
    if version != SCHEMA_VERSION:
        errors.append(f"{label}.schema_version must be {SCHEMA_VERSION!r}.")
    unsupported = sorted(
        {
            str(key)
            for key in source
            if str(key) not in _SOURCE_KEYS
        }
    )
    if unsupported:
        errors.append(
            f"{label} uses unsupported materialized-table source options: "
            + ", ".join(unsupported)
            + "."
        )
    for key, value in source.items():
        if isinstance(value, str) and (
            published_text_contains_local_path_reference(value) or _PARENT_REFERENCE.search(value)
        ):
            errors.append(f"{label}.{key} contains a non-portable local path reference.")
    return tuple(errors)


def plan_materialized_pattern_table(
    config: Any,
    source: Any,
    units: Sequence[ResolvedAnalysisUnit],
    *,
    roots: Mapping[str, str | Path] | None,
) -> MaterializedPatternTablePlan:
    """Stream and plan one fixed-v1 TSV without decoding vector JSON."""

    source_fields = _source_fields(source)
    source_name = _text(getattr(source, "name", None) or source_fields.get("name")) or "<unnamed>"
    declaration_errors = list(
        validate_materialized_pattern_source_fields(source_fields, "pattern_source")
    )
    root_ref = _text(getattr(source, "root_ref", None) or source_fields.get("root_ref"))
    relative_path = _text(getattr(source, "path", None) or source_fields.get("path"))
    portable_reference = (
        f"root_ref:{root_ref}/{relative_path}" if root_ref is not None and relative_path is not None else None
    )
    if declaration_errors:
        public_name = source_name if _safe_identifier(source_name) else "<invalid-source>"
        public_root_ref = root_ref if root_ref is not None and _safe_identifier(root_ref) else None
        public_relative_path = (
            relative_path
            if relative_path is not None and _configured_source_path_is_safe(relative_path)
            else None
        )
        public_reference = (
            f"root_ref:{public_root_ref}/{public_relative_path}"
            if public_root_ref is not None and public_relative_path is not None
            else None
        )
        return _empty_plan(
            public_name,
            public_root_ref,
            public_relative_path,
            public_reference,
            errors=declaration_errors,
        )
    if not units:
        return _empty_plan(
            source_name,
            root_ref,
            relative_path,
            portable_reference,
            errors=("Materialized-table planning requires at least one resolved exact analysis unit.",),
        )
    key_columns = _unit_key_columns(units)
    if "subject_id" not in key_columns or any(tuple(unit.key_columns) != key_columns for unit in units):
        return _empty_plan(
            source_name,
            root_ref,
            relative_path,
            portable_reference,
            errors=(
                "Materialized-table units must share ordered key columns that include subject_id.",
            ),
        )
    duplicate_unit_rows = _duplicate_unit_key_rows(units, key_columns=key_columns)
    if duplicate_unit_rows:
        first, second = duplicate_unit_rows[0]
        return _empty_plan(
            source_name,
            root_ref,
            relative_path,
            portable_reference,
            errors=(
                "Materialized-table exact units repeat a configured unit key "
                f"at source rows {first} and {second}.",
            ),
        )
    unit_metadata_errors = tuple(
        error
        for unit in units
        for error in _unit_metadata_errors(unit)
    )
    if unit_metadata_errors:
        return _empty_plan(
            source_name,
            root_ref,
            relative_path,
            portable_reference,
            errors=unit_metadata_errors,
        )
    if roots is None or root_ref not in roots:
        return _empty_plan(
            source_name,
            root_ref,
            relative_path,
            portable_reference,
            errors=(f"Configured root_ref {root_ref!r} is unavailable for materialized-table planning.",),
        )

    assert relative_path is not None
    root_path = Path(roots[root_ref])
    candidate_path = root_path / Path(relative_path)
    try:
        resolved_root = root_path.resolve(strict=True)
        local_path = candidate_path.resolve(strict=True)
        local_path.relative_to(resolved_root)
    except (OSError, ValueError):
        return _empty_plan(
            source_name,
            root_ref,
            relative_path,
            portable_reference,
            errors=(
                "The configured materialized pattern table is missing, is not a regular file, "
                "or does not remain beneath root_ref.",
            ),
            local_path=candidate_path,
        )
    if not local_path.is_file():
        return _empty_plan(
            source_name,
            root_ref,
            relative_path,
            portable_reference,
            errors=("The configured materialized pattern table is not a regular file.",),
            local_path=local_path,
        )

    errors: list[str] = []
    warnings: list[str] = []
    columns, raw_rows, source_digest, read_errors = _stream_scalar_rows(
        local_path,
        retain_vectors=False,
    )
    errors.extend(read_errors)
    errors.extend(_header_errors(columns, key_columns=key_columns))
    if errors:
        return MaterializedPatternTablePlan(
            source_name=source_name,
            root_ref=root_ref,
            relative_path=relative_path,
            portable_reference=portable_reference,
            source_sha256=source_digest,
            columns=columns,
            key_columns=key_columns,
            total_row_count=len(raw_rows),
            errors=tuple(_unique(errors)),
            _local_path=local_path,
        )

    unit_by_key = {_unit_key(unit, key_columns): unit for unit in units}
    condition_ids = tuple(str(condition.id) for condition in getattr(config, "conditions", ()))
    condition_rank = {condition_id: index for index, condition_id in enumerate(condition_ids)}
    configured_roi_sources = tuple(str(item.name) for item in getattr(config, "roi_sources", ()))
    roi_source_rank = {name: index for index, name in enumerate(configured_roi_sources)}
    declared_roi_labels = _declared_roi_labels(config)
    observed_roi_labels: dict[str, list[str]] = {name: [] for name in configured_roi_sources}
    pattern_ids: dict[str, int] = {}
    selected: list[tuple[ResolvedAnalysisUnit, dict[str, str], int]] = []
    unselected_ids: list[str] = []

    for data_row, row in raw_rows:
        errors.extend(_all_row_scalar_errors(row, data_row=data_row, key_columns=key_columns))
        pattern_id = _required_cell(row, "pattern_id", data_row, errors)
        if pattern_id:
            previous = pattern_ids.get(pattern_id)
            if previous is not None:
                errors.append(
                    f"Materialized table pattern_id is duplicated at data rows {previous} and {data_row}."
                )
            else:
                pattern_ids[pattern_id] = data_row
        row_key = tuple(row.get(column, "") for column in key_columns)
        unit = unit_by_key.get(row_key)
        if unit is None:
            if pattern_id:
                unselected_ids.append(pattern_id)
            continue

        errors.extend(
            _selected_scalar_errors(
                config=config,
                source_name=source_name,
                row=row,
                data_row=data_row,
                unit=unit,
                condition_ids=condition_ids,
                configured_roi_sources=configured_roi_sources,
            )
        )
        roi_source_name = row.get("roi_source_name", "")
        roi_label = row.get("roi_label", "")
        if (
            roi_source_name in observed_roi_labels
            and roi_label
            and roi_label not in observed_roi_labels[roi_source_name]
        ):
            observed_roi_labels[roi_source_name].append(roi_label)
        selected.append((unit, row, data_row))

    roi_pairs, roi_errors = _expected_roi_pairs(
        configured_roi_sources,
        declared_roi_labels=declared_roi_labels,
        observed_roi_labels=observed_roi_labels,
    )
    errors.extend(roi_errors)
    roi_rank = {pair: index for index, pair in enumerate(roi_pairs)}
    errors.extend(
        _coverage_errors(
            units=units,
            condition_ids=condition_ids,
            roi_pairs=roi_pairs,
            selected=selected,
        )
    )
    errors.extend(_planning_group_consistency_errors(config=config, selected=selected))
    event_threshold_rows = _materialized_event_threshold_rows(config, selected=selected)
    for threshold in event_threshold_rows:
        status = str(threshold.get("status"))
        if status == "failed":
            errors.append(
                "Materialized pattern table does not satisfy configured "
                f"{threshold['threshold']}={threshold['value']}."
            )
        elif status == "not_evaluated":
            errors.append(
                "Materialized pattern table cannot evaluate configured "
                f"{threshold['threshold']}={threshold['value']}."
            )

    unit_rank = {unit.unit_id: index for index, unit in enumerate(units)}
    selected.sort(
        key=lambda item: (
            unit_rank.get(item[0].unit_id, len(unit_rank)),
            condition_rank.get(item[1].get("condition_id", ""), len(condition_rank)),
            roi_source_rank.get(item[1].get("roi_source_name", ""), len(roi_source_rank)),
            roi_rank.get((item[1].get("roi_source_name", ""), item[1].get("roi_label", "")), len(roi_rank)),
            item[2],
        )
    )

    scalar_rows: list[MaterializedPatternScalarRow] = []
    pattern_rows: list[PlannedPatternRow] = []
    if not errors:
        for unit, row, data_row in selected:
            scalar = _scalar_row(config, row, unit=unit, data_row=data_row, fixed_columns=columns)
            scalar_rows.append(scalar)
            row_reference = f"{portable_reference}#row={data_row}"
            pattern_rows.append(
                PlannedPatternRow(
                    unit_id=unit.unit_id,
                    subject_id=unit.subject_id,
                    session_id=unit.session_id,
                    task_id=unit.task_id,
                    run_id=unit.run_id,
                    cross_validation_label=scalar.cross_validation_label,
                    condition_id=scalar.condition_id,
                    source_name=source_name,
                    backend_name=PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE,
                    representation_kind=REPRESENTATION_PREPARED_FEATURES,
                    pattern_reference=row_reference,
                    noise_reference=(
                        f"{row_reference}&field=noise_values"
                        if _noise_method(config) == "diagonal" and row.get("noise_values", "").strip()
                        else None
                    ),
                    event_count=scalar.event_count,
                    qc_status=scalar.qc_status,
                    status=scalar.status,
                    unit_metadata=dict(unit.metadata),
                    backend_metadata={
                        "pattern_id": scalar.pattern_id,
                        "roi_source_name": scalar.roi_source_name,
                        "roi_label": scalar.roi_label,
                        "feature_count": scalar.feature_count,
                        "voxel_order": scalar.voxel_order,
                        "voxel_index_hash": scalar.voxel_index_hash,
                        "feature_space_id": scalar.feature_space_id,
                        "roi_definition_id": scalar.roi_definition_id,
                        "mean_centering_applied": scalar.mean_centering_applied,
                        "mean_centering_scope": scalar.mean_centering_scope,
                        "noise_status": scalar.noise_status,
                        "noise_usable": scalar.noise_usable,
                        "table_sha256": source_digest,
                        "table_data_row": data_row,
                    },
                )
            )

    if unselected_ids:
        warnings.append(
            f"Materialized table contains {len(unselected_ids)} row(s) outside the selected exact units."
        )
    return MaterializedPatternTablePlan(
        source_name=source_name,
        root_ref=root_ref,
        relative_path=relative_path,
        portable_reference=portable_reference,
        source_sha256=source_digest,
        columns=columns,
        key_columns=key_columns,
        scalar_rows=tuple(scalar_rows),
        pattern_rows=tuple(pattern_rows),
        total_row_count=len(raw_rows),
        selected_row_count=len(selected),
        unselected_row_count=len(unselected_ids),
        unselected_pattern_ids=tuple(unselected_ids),
        event_threshold_rows=event_threshold_rows,
        warnings=tuple(warnings),
        errors=tuple(_unique(errors)),
        _local_path=local_path,
    )


def load_materialized_pattern_table(
    plan: MaterializedPatternTablePlan,
    *,
    expected_sha256: str,
) -> MaterializedPatternTableLoadResult:
    """Digest-check, decode, and strictly validate selected feature rows."""

    if not plan.ready_for_materialization or plan._local_path is None:
        return _load_failure(plan, "Materialized-table plan is not ready for loading.")
    expected = str(expected_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        return _load_failure(plan, "Expected materialized-table SHA-256 must contain 64 hexadecimal characters.")
    if expected != plan.source_sha256:
        return _load_failure(plan, "Expected materialized-table SHA-256 does not match the planned digest.")
    columns, raw_rows, actual, read_errors = _stream_scalar_rows(
        plan._local_path,
        retain_vectors=True,
    )
    if read_errors:
        return _load_failure(plan, *read_errors)
    if actual != expected:
        return _load_failure(plan, "Materialized pattern table changed after planning; SHA-256 mismatch.")
    if columns != plan.columns:
        return _load_failure(plan, "Materialized pattern table header changed after planning.")
    by_data_row = {data_row: row for data_row, row in raw_rows}
    loaded_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    warnings: list[str] = list(plan.warnings)
    errors: list[str] = []

    for scalar in plan.scalar_rows:
        row = by_data_row.get(scalar.data_row)
        if row is None or row.get("pattern_id") != scalar.pattern_id:
            errors.append(
                f"Planned pattern {scalar.pattern_id!r} is absent or changed at data row {scalar.data_row}."
            )
            continue
        row_errors: list[str] = []
        feature_values = _numeric_json_vector(
            row.get("feature_values", ""),
            label="feature_values",
            data_row=scalar.data_row,
            strictly_positive=False,
            errors=row_errors,
        )
        if feature_values is not None and len(feature_values) != scalar.feature_count:
            row_errors.append(
                f"Materialized table data row {scalar.data_row} feature_count does not match feature_values."
            )
        grouping_values = _json_scalar_mapping(
            row.get("grouping_values", ""),
            label="grouping_values",
            data_row=scalar.data_row,
            errors=row_errors,
        )
        warning_values = _json_string_sequence(
            row.get("warnings", ""),
            label="warnings",
            data_row=scalar.data_row,
            errors=row_errors,
        )
        error_values = _json_string_sequence(
            row.get("errors", ""),
            label="errors",
            data_row=scalar.data_row,
            errors=row_errors,
        )
        noise_payload, noise_warnings = _loaded_noise_payload(
            plan,
            row,
            scalar=scalar,
            errors=row_errors,
        )
        warnings.extend(noise_warnings)
        if scalar.usable and error_values:
            row_errors.append(
                f"Materialized table data row {scalar.data_row} is usable but declares row errors."
            )
        if not scalar.usable and scalar.status not in _UNUSABLE_STATUSES:
            row_errors.append(
                f"Materialized table data row {scalar.data_row} is unusable but has an incompatible status."
            )

        if row_errors:
            errors.extend(row_errors)
            qc_rows.extend(
                _qc_failure(scalar.pattern_id, scalar.data_row, message) for message in row_errors
            )
            continue
        assert feature_values is not None
        output: dict[str, Any] = {
            "pattern_id": scalar.pattern_id,
            "unit_id": scalar.unit_id,
            "condition_id": scalar.condition_id,
            "cross_validation_label": scalar.cross_validation_label,
            "subject_id": scalar.subject_id,
            "session_id": scalar.session_id,
            "task_id": scalar.task_id,
            "run_id": scalar.run_id,
            "pattern_source_name": scalar.pattern_source_name,
            "roi_source_name": scalar.roi_source_name,
            "roi_label": scalar.roi_label,
            "feature_values": feature_values,
            "feature_count": scalar.feature_count,
            "voxel_order": scalar.voxel_order,
            "voxel_index_hash": scalar.voxel_index_hash,
            "feature_space_id": scalar.feature_space_id,
            "roi_definition_id": scalar.roi_definition_id,
            "usable": scalar.usable,
            "status": scalar.status,
            "mean_centering_applied": scalar.mean_centering_applied,
            "mean_centering_scope": scalar.mean_centering_scope,
            "event_count": scalar.event_count,
            "qc_status": scalar.qc_status,
            "qc_reason": scalar.qc_reason,
            "exclusion_id": scalar.exclusion_id,
            "exclusion_reason": scalar.exclusion_reason,
            "roi_reference": scalar.roi_reference,
            "generator_version": scalar.generator_version,
            "software_version": scalar.software_version,
            "derivation_id": scalar.derivation_id,
            "holdout_id": scalar.holdout_id,
            "unit_metadata": dict(scalar.unit_metadata),
            "grouping_values": grouping_values,
            "warnings": warning_values,
            "errors": error_values,
            **_unit_metadata_columns(scalar.unit_metadata),
            **dict(scalar.extras),
            **noise_payload,
        }
        loaded_rows.append(output)
        qc_rows.append(
            {
                "pattern_id": scalar.pattern_id,
                "data_row": scalar.data_row,
                "status": "pass" if scalar.usable else "excluded",
                "code": "materialized_row_validated",
                "message": "Materialized row passed strict vector and metadata validation.",
                "qc_status": scalar.qc_status,
                "qc_reason": scalar.qc_reason,
                "exclusion_id": scalar.exclusion_id,
                "exclusion_reason": scalar.exclusion_reason,
            }
        )

    errors.extend(_loaded_group_consistency_errors(loaded_rows, plan=plan))
    if errors:
        return MaterializedPatternTableLoadResult(
            rows=(),
            qc_rows=tuple(qc_rows),
            provenance=_loader_provenance(plan, digest=actual, loaded_rows=0, rows=loaded_rows),
            warnings=tuple(_unique(warnings)),
            errors=tuple(_unique(errors)),
            materialized=False,
            executed=False,
        )
    return MaterializedPatternTableLoadResult(
        rows=tuple(loaded_rows),
        qc_rows=tuple(qc_rows),
        provenance=_loader_provenance(plan, digest=actual, loaded_rows=len(loaded_rows), rows=loaded_rows),
        warnings=tuple(_unique(warnings)),
        errors=(),
        materialized=True,
        executed=False,
    )


def _empty_plan(
    source_name: str,
    root_ref: str | None,
    relative_path: str | None,
    portable_reference: str | None,
    *,
    errors: Sequence[str],
    local_path: Path | None = None,
) -> MaterializedPatternTablePlan:
    return MaterializedPatternTablePlan(
        source_name=source_name,
        root_ref=root_ref,
        relative_path=relative_path,
        portable_reference=portable_reference,
        source_sha256=None,
        errors=tuple(errors),
        _local_path=local_path,
    )


def _source_fields(source: Any) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    fields = getattr(source, "fields", None)
    return fields if isinstance(fields, Mapping) else {}


def _safe_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value))


def _configured_source_path_is_safe(value: str) -> bool:
    return not (
        configured_path_is_unsafe(value)
        or _URI_SCHEME.match(value)
        or "\\" in value
        or published_text_contains_local_path_reference(value)
        or _PARENT_REFERENCE.search(value)
    )


def _stream_scalar_rows(
    path: Path,
    *,
    retain_vectors: bool,
) -> tuple[tuple[str, ...], list[tuple[int, dict[str, str]]], str | None, list[str]]:
    """Read and hash the exact same bytes through one open file descriptor."""

    rows: list[tuple[int, dict[str, str]]] = []
    errors: list[str] = []
    digest = sha256()
    try:
        with path.open("rb") as handle:
            def decoded_lines():
                for raw_line in handle:
                    digest.update(raw_line)
                    yield raw_line.decode("utf-8")

            reader = csv.DictReader(decoded_lines(), delimiter="\t")
            columns = tuple(reader.fieldnames or ())
            for data_row, raw in enumerate(reader, start=1):
                if None in raw:
                    errors.append(f"Materialized table data row {data_row} has more cells than header columns.")
                    continue
                normalized = {str(key): "" if value is None else str(value) for key, value in raw.items()}
                errors.extend(_safe_row_errors(normalized, data_row=data_row))
                if not retain_vectors:
                    for column in _VECTOR_COLUMNS:
                        normalized[column] = "<present>" if normalized.get(column, "").strip() else ""
                rows.append((data_row, normalized))
    except (OSError, UnicodeError, csv.Error):
        return (), [], None, ["Materialized pattern table could not be read as UTF-8 TSV."]
    return columns, rows, digest.hexdigest(), errors


def _header_errors(columns: Sequence[str], *, key_columns: Sequence[str]) -> tuple[str, ...]:
    errors: list[str] = []
    if not columns:
        return ("Materialized pattern table must contain a TSV header.",)
    if len(set(columns)) != len(columns):
        errors.append("Materialized pattern table header contains duplicate columns.")
    if len({column.casefold() for column in columns}) != len(columns):
        errors.append("Materialized pattern table header contains case-insensitive column collisions.")
    for column in columns:
        if not _SAFE_COLUMN.fullmatch(column):
            errors.append("Materialized pattern table header contains an unsafe column name.")
        if column in _DERIVED_RESERVED_COLUMNS:
            errors.append(
                f"Materialized pattern table column {column!r} collides with a derived row field."
            )
    for column in (*REQUIRED_COLUMNS, *key_columns):
        if column not in columns:
            errors.append(f"Materialized pattern table is missing required column {column!r}.")
    return tuple(errors)


def _unit_metadata_errors(unit: ResolvedAnalysisUnit) -> tuple[str, ...]:
    """Reject metadata that cannot be emitted as a portable scalar plan value."""

    errors: list[str] = []
    for raw_column, value in unit.metadata.items():
        column = str(raw_column)
        if not _SAFE_COLUMN.fullmatch(column):
            errors.append(
                f"Exact unit source row {unit.source_row} contains an unsafe metadata column name."
            )
        if published_text_contains_local_path_reference(column):
            errors.append(
                f"Exact unit source row {unit.source_row} contains a non-portable metadata column name."
            )
        if isinstance(value, (Mapping, list, tuple, set)):
            errors.append(
                f"Exact unit source row {unit.source_row} metadata column {column!r} must be scalar."
            )
            continue
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(
                f"Exact unit source row {unit.source_row} metadata column {column!r} must be finite."
            )
            continue
        if value is not None and not isinstance(value, (str, int, float, bool)):
            errors.append(
                f"Exact unit source row {unit.source_row} metadata column {column!r} "
                "must be a JSON-safe scalar."
            )
            continue
        if isinstance(value, (str, Path)) and (
            published_text_contains_local_path_reference(str(value))
            or _PARENT_REFERENCE.search(str(value))
        ):
            errors.append(
                f"Exact unit source row {unit.source_row} metadata column {column!r} "
                "contains a non-portable local path reference."
            )
    return tuple(errors)


def _unit_key_columns(units: Sequence[ResolvedAnalysisUnit]) -> tuple[str, ...]:
    if not units:
        return ()
    return tuple(units[0].key_columns)


def _unit_key(unit: ResolvedAnalysisUnit, columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(unit.values.get(column, "")) for column in columns)


def _duplicate_unit_key_rows(
    units: Sequence[ResolvedAnalysisUnit],
    *,
    key_columns: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    seen: dict[tuple[str, ...], int] = {}
    duplicates: list[tuple[int, int]] = []
    for unit in units:
        key = _unit_key(unit, key_columns)
        previous = seen.get(key)
        if previous is not None:
            duplicates.append((previous, unit.source_row))
        else:
            seen[key] = unit.source_row
    return tuple(duplicates)


def _safe_row_errors(row: Mapping[str, str], *, data_row: int) -> tuple[str, ...]:
    errors: list[str] = []
    for column, value in row.items():
        if published_text_contains_local_path_reference(value) or _PARENT_REFERENCE.search(value):
            errors.append(
                f"Materialized table data row {data_row} column {column!r} "
                "contains a non-portable local path reference."
            )
    return tuple(errors)


def _all_row_scalar_errors(
    row: Mapping[str, str],
    *,
    data_row: int,
    key_columns: Sequence[str],
) -> tuple[str, ...]:
    errors: list[str] = []
    for column in REQUIRED_COLUMNS:
        _required_cell(row, column, data_row, errors)
    for column in key_columns:
        _required_cell(row, column, data_row, errors)
    if row.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"Materialized table data row {data_row} has an unsupported schema_version.")
    _positive_int(row.get("feature_count", ""), "feature_count", data_row, errors)
    _bool_value(row.get("usable", ""), "usable", data_row, errors)
    _bool_value(row.get("mean_centering_applied", ""), "mean_centering_applied", data_row, errors)
    _bool_value(row.get("noise_usable", ""), "noise_usable", data_row, errors)
    for column, value in row.items():
        if column in FIXED_COLUMNS or column in key_columns:
            continue
        text = value.strip()
        if text.startswith(("[", "{")):
            errors.append(
                f"Materialized table data row {data_row} extra column {column!r} must contain a scalar value."
            )
    return tuple(errors)


def _selected_scalar_errors(
    *,
    config: Any,
    source_name: str,
    row: Mapping[str, str],
    data_row: int,
    unit: ResolvedAnalysisUnit,
    condition_ids: Sequence[str],
    configured_roi_sources: Sequence[str],
) -> tuple[str, ...]:
    errors: list[str] = []
    if row.get("pattern_source_name") != source_name:
        errors.append(f"Materialized table data row {data_row} pattern_source_name does not match its source config.")
    for column in unit.key_columns:
        if row.get(column, "") != str(unit.values.get(column, "")):
            errors.append(f"Materialized table data row {data_row} does not exactly match unit key column {column!r}.")
    for column, value in row.items():
        if column in FIXED_COLUMNS or column in unit.key_columns or column not in unit.metadata:
            continue
        if value != str(unit.metadata[column]):
            errors.append(
                f"Materialized table data row {data_row} metadata column {column!r} "
                "does not match the authoritative exact unit."
            )
    for column, expected in (
        ("subject_id", unit.subject_id),
        ("session_id", unit.session_id),
        ("task_id", unit.task_id),
        ("run_id", unit.run_id),
    ):
        actual = row.get(column, "")
        if expected is not None and actual != expected:
            errors.append(f"Materialized table data row {data_row} canonical {column} does not match the exact unit.")
        if expected is None and actual:
            errors.append(
                f"Materialized table data row {data_row} invents optional {column} "
                "absent from the exact unit."
            )
    if row.get("condition_id") not in condition_ids:
        errors.append(f"Materialized table data row {data_row} references an unknown condition_id.")
    if row.get("roi_source_name") not in configured_roi_sources:
        errors.append(f"Materialized table data row {data_row} references an unknown roi_source_name.")
    derived_cv = _cross_validation_label(config, unit)
    if derived_cv is None:
        errors.append(f"Materialized table data row {data_row} cannot derive the configured cross-validation label.")
    elif row.get("cross_validation_label", "") and row["cross_validation_label"] != derived_cv:
        errors.append(f"Materialized table data row {data_row} cross_validation_label does not match the exact unit.")
    _optional_nonnegative_int(row.get("event_count", ""), "event_count", data_row, errors)
    usable = _bool_value(row.get("usable", ""), "usable", data_row, errors)
    status = row.get("status", "").strip().lower()
    if usable is True and status not in _USABLE_STATUSES:
        errors.append(f"Materialized table data row {data_row} usable/status fields are inconsistent.")
    if usable is False and status not in _UNUSABLE_STATUSES:
        errors.append(f"Materialized table data row {data_row} usable/status fields are inconsistent.")
    qc_status = row.get("qc_status", "").strip().lower()
    excluded = bool(row.get("exclusion_id", "").strip() or row.get("exclusion_reason", "").strip())
    if usable is True and (excluded or qc_status in {"fail", "failed", "error", "excluded"}):
        errors.append(f"Materialized table data row {data_row} QC/exclusion fields conflict with usable=true.")
    if usable is False and qc_status in {"pass", "passed", "ok"}:
        errors.append(f"Materialized table data row {data_row} QC status conflicts with usable=false.")
    if usable is False and not (
        row.get("qc_reason", "").strip()
        or row.get("exclusion_reason", "").strip()
        or row.get("errors", "").strip() not in {"", "[]"}
    ):
        errors.append(f"Materialized table data row {data_row} unusable row requires an auditable reason.")
    centered = _bool_value(row.get("mean_centering_applied", ""), "mean_centering_applied", data_row, errors)
    scope = row.get("mean_centering_scope", "").strip()
    expected_centered = bool(getattr(getattr(config, "mean_centering", None), "enabled", False))
    expected_scope = str(getattr(getattr(config, "mean_centering", None), "scope", "none"))
    if centered is not None and centered != expected_centered:
        errors.append(f"Materialized table data row {data_row} mean-centering state does not match configuration.")
    if scope != expected_scope:
        errors.append(f"Materialized table data row {data_row} mean-centering scope does not match configuration.")
    thresholds = getattr(config, "event_thresholds", None)
    min_events = getattr(thresholds, "min_events_per_condition_per_run", None)
    if min_events is not None and not row.get("event_count", "").strip():
        errors.append(f"Materialized table data row {data_row} event_count is required by configured thresholds.")
    event_count = _optional_nonnegative_int(row.get("event_count", ""), "event_count", data_row, [])
    if min_events is not None and event_count is not None and event_count < int(min_events) and usable is True:
        errors.append(f"Materialized table data row {data_row} is usable despite failing its event-count threshold.")
    noise_method = _noise_method(config)
    noise_status = row.get("noise_status", "").strip().lower()
    noise_usable = _bool_value(row.get("noise_usable", ""), "noise_usable", data_row, errors)
    if noise_method == "identity":
        if noise_status != "unused" or noise_usable is not False:
            errors.append(
                f"Materialized table data row {data_row} identity noise must be "
                "status=unused and usable=false."
            )
    elif noise_method == "diagonal":
        if noise_status not in {"ok", "warning", "usable"} or noise_usable is not True:
            errors.append(f"Materialized table data row {data_row} diagonal noise must be declared usable.")
        for column in (
            "noise_values",
            "noise_feature_count",
            "noise_voxel_order",
            "noise_voxel_index_hash",
            "noise_feature_space_id",
            "noise_roi_definition_id",
            "noise_value_kind",
            "noise_estimation_scope",
            "noise_source",
        ):
            _required_cell(row, column, data_row, errors)
        feature_count = _positive_int(row.get("feature_count", ""), "feature_count", data_row, [])
        noise_count = _positive_int(
            row.get("noise_feature_count", ""),
            "noise_feature_count",
            data_row,
            errors,
        )
        if feature_count is not None and noise_count is not None and feature_count != noise_count:
            errors.append(f"Materialized table data row {data_row} feature and noise counts differ.")
        for feature_column, noise_column in (
            ("voxel_order", "noise_voxel_order"),
            ("voxel_index_hash", "noise_voxel_index_hash"),
            ("feature_space_id", "noise_feature_space_id"),
            ("roi_definition_id", "noise_roi_definition_id"),
        ):
            if row.get(feature_column) != row.get(noise_column):
                errors.append(
                    f"Materialized table data row {data_row} {noise_column} "
                    "does not match its feature metadata."
                )
        if row.get("noise_value_kind", "").strip().lower() != "variance":
            errors.append(
                f"Materialized table data row {data_row} noise values must be declared as variances."
            )
    return tuple(errors)


def _materialized_event_threshold_rows(
    config: Any,
    *,
    selected: Sequence[tuple[ResolvedAnalysisUnit, Mapping[str, str], int]],
) -> tuple[Mapping[str, Any], ...]:
    """Evaluate configured thresholds from vector-free exact-unit row metadata."""

    thresholds = getattr(config, "event_thresholds", None)
    if thresholds is None:
        return ()
    fields = dict(getattr(thresholds, "fields", {}) or {})
    for name in ("min_events_per_condition_per_run", "min_runs_per_condition"):
        value = getattr(thresholds, name, None)
        if value is not None:
            fields[name] = value

    rows: list[Mapping[str, Any]] = []
    for name in ("min_events_per_condition_per_run", "min_runs_per_condition"):
        if name not in fields:
            continue
        value = fields[name]
        if not isinstance(value, int) or isinstance(value, bool):
            rows.append(
                {
                    "threshold": name,
                    "value": value,
                    "status": "not_evaluated",
                    "reason": "threshold_value_is_not_an_integer",
                }
            )
            continue
        if value <= 0:
            rows.append(
                {
                    "threshold": name,
                    "value": value,
                    "status": "passed",
                    "reason": "nonpositive_threshold_requires_no_observations",
                    "evaluated_row_count": len(selected),
                }
            )
            continue

        usable = [
            (unit, row, data_row)
            for unit, row, data_row in selected
            if row.get("usable", "").strip().lower() in _TRUE_VALUES
        ]
        if name == "min_events_per_condition_per_run":
            event_counts = [
                _optional_nonnegative_int(row.get("event_count", ""), "event_count", data_row, [])
                for _unit, row, data_row in usable
            ]
            missing = sum(count is None for count in event_counts)
            failing = sum(count is not None and count < value for count in event_counts)
            status = "not_evaluated" if missing else ("failed" if failing else "passed")
            rows.append(
                {
                    "threshold": name,
                    "value": value,
                    "status": status,
                    "reason": (
                        "event_count_missing"
                        if missing
                        else ("event_count_below_threshold" if failing else "all_usable_rows_passed")
                    ),
                    "evaluated_row_count": len(usable) - missing,
                    "missing_row_count": missing,
                    "failing_row_count": failing,
                }
            )
            continue

        run_groups: dict[tuple[str, ...], set[str]] = {}
        missing_run_count = 0
        for unit, row, _data_row in usable:
            group = (
                unit.subject_id,
                unit.session_id or "",
                unit.task_id or "",
                row.get("pattern_source_name", ""),
                row.get("roi_source_name", ""),
                row.get("roi_label", ""),
                row.get("condition_id", ""),
            )
            run_groups.setdefault(group, set())
            if unit.run_id is None or not str(unit.run_id).strip():
                missing_run_count += 1
            else:
                run_groups[group].add(str(unit.run_id))
        failing_groups = sum(1 for run_ids in run_groups.values() if len(run_ids) < value)
        status = (
            "not_evaluated"
            if missing_run_count or not run_groups
            else ("failed" if failing_groups else "passed")
        )
        rows.append(
            {
                "threshold": name,
                "value": value,
                "status": status,
                "reason": (
                    "run_identity_missing"
                    if missing_run_count or not run_groups
                    else ("run_count_below_threshold" if failing_groups else "all_condition_roi_groups_passed")
                ),
                "evaluated_group_count": len(run_groups),
                "failing_group_count": failing_groups,
                "missing_run_row_count": missing_run_count,
            }
        )

    for name, value in fields.items():
        if name in {"min_events_per_condition_per_run", "min_runs_per_condition"}:
            continue
        rows.append(
            {
                "threshold": str(name),
                "value": value,
                "status": "not_evaluated",
                "reason": "unsupported_materialized_threshold",
            }
        )
    return tuple(rows)


def _declared_roi_labels(config: Any) -> Mapping[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for source in getattr(config, "roi_sources", ()):
        labels: list[str] = []
        fields = getattr(source, "fields", {})
        raw_labels = fields.get("roi_labels", ()) if isinstance(fields, Mapping) else ()
        if isinstance(raw_labels, str):
            raw_labels = (raw_labels,)
        if isinstance(raw_labels, Sequence):
            labels.extend(str(value) for value in raw_labels if str(value).strip())
        for mask in getattr(source, "masks", ()):
            if not isinstance(mask, Mapping):
                continue
            label = _text(mask.get("label") or mask.get("roi_label"))
            if label is not None:
                labels.append(label)
        result[str(source.name)] = tuple(_unique(labels))
    return result


def _expected_roi_pairs(
    configured_sources: Sequence[str],
    *,
    declared_roi_labels: Mapping[str, Sequence[str]],
    observed_roi_labels: Mapping[str, Sequence[str]],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    pairs: list[tuple[str, str]] = []
    errors: list[str] = []
    for source_name in configured_sources:
        declared = tuple(declared_roi_labels.get(source_name, ()))
        observed = tuple(observed_roi_labels.get(source_name, ()))
        labels = declared or observed
        if not labels:
            errors.append(f"Configured ROI source {source_name!r} has no materialized ROI labels.")
            continue
        unknown = tuple(label for label in observed if declared and label not in declared)
        if unknown:
            errors.append(f"Materialized table contains ROI labels outside configured source {source_name!r}.")
        pairs.extend((source_name, label) for label in labels)
    return tuple(pairs), tuple(errors)


def _coverage_errors(
    *,
    units: Sequence[ResolvedAnalysisUnit],
    condition_ids: Sequence[str],
    roi_pairs: Sequence[tuple[str, str]],
    selected: Sequence[tuple[ResolvedAnalysisUnit, Mapping[str, str], int]],
) -> tuple[str, ...]:
    errors: list[str] = []
    seen: dict[tuple[str, str, str, str], int] = {}
    for unit, row, data_row in selected:
        key = (
            unit.unit_id,
            row.get("condition_id", ""),
            row.get("roi_source_name", ""),
            row.get("roi_label", ""),
        )
        previous = seen.get(key)
        if previous is not None:
            errors.append(
                f"Materialized table repeats one unit-condition-ROI key at data rows {previous} and {data_row}."
            )
        else:
            seen[key] = data_row
    for unit in units:
        for condition_id in condition_ids:
            for source_name, roi_label in roi_pairs:
                key = (unit.unit_id, condition_id, source_name, roi_label)
                if key not in seen:
                    errors.append(
                        "Materialized table does not fully cover every selected unit, configured condition, and ROI."
                    )
                    return tuple(errors)
    return tuple(errors)


def _planning_group_consistency_errors(
    *,
    config: Any,
    selected: Sequence[tuple[ResolvedAnalysisUnit, Mapping[str, str], int]],
) -> tuple[str, ...]:
    """Validate vector-free feature and noise identity across analysis groups."""

    errors: list[str] = []
    feature_signatures: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    noise_signatures: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    diagonal_noise = _noise_method(config) == "diagonal"
    for unit, row, _data_row in selected:
        feature_group_key = (
            unit.subject_id,
            unit.session_id,
            unit.task_id,
            row.get("pattern_source_name", ""),
            row.get("roi_source_name", ""),
            row.get("roi_label", ""),
        )
        feature_signature = (
            _integer_signature(row.get("feature_count", "")),
            row.get("voxel_order", ""),
            row.get("voxel_index_hash", ""),
            row.get("feature_space_id", ""),
            row.get("roi_definition_id", ""),
        )
        previous_feature = feature_signatures.setdefault(feature_group_key, feature_signature)
        if previous_feature != feature_signature:
            errors.append("Materialized feature metadata is inconsistent within an analysis/ROI group.")

        if not diagonal_noise:
            continue
        cross_validation_label = (
            row.get("cross_validation_label", "")
            or _cross_validation_label(config, unit)
            or ""
        )
        noise_group_key = (
            unit.unit_id,
            cross_validation_label,
            row.get("pattern_source_name", ""),
            row.get("roi_source_name", ""),
            row.get("roi_label", ""),
        )
        noise_signature = (
            row.get("noise_status", "").strip().lower(),
            row.get("noise_usable", "").strip().lower(),
            _integer_signature(row.get("noise_feature_count", "")),
            row.get("noise_voxel_order", ""),
            row.get("noise_voxel_index_hash", ""),
            row.get("noise_feature_space_id", ""),
            row.get("noise_roi_definition_id", ""),
            row.get("noise_value_kind", "").strip().lower(),
            row.get("noise_estimation_scope", ""),
            row.get("noise_source", ""),
        )
        previous_noise = noise_signatures.setdefault(noise_group_key, noise_signature)
        if previous_noise != noise_signature:
            errors.append(
                "Materialized noise identity metadata is inconsistent across conditions "
                "for one unit/ROI/CV group."
            )
    return tuple(_unique(errors))


def _integer_signature(value: str) -> int | str:
    """Normalize a valid integer cell without turning validation errors into exceptions."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _scalar_row(
    config: Any,
    row: Mapping[str, str],
    *,
    unit: ResolvedAnalysisUnit,
    data_row: int,
    fixed_columns: Sequence[str],
) -> MaterializedPatternScalarRow:
    known = FIXED_COLUMNS.union(str(column) for column in unit.metadata)
    extras = {
        column: row.get(column, "")
        for column in fixed_columns
        if column not in known and column not in _VECTOR_COLUMNS
    }
    return MaterializedPatternScalarRow(
        data_row=data_row,
        pattern_id=row["pattern_id"],
        unit_id=unit.unit_id,
        subject_id=unit.subject_id,
        session_id=unit.session_id,
        task_id=unit.task_id,
        run_id=unit.run_id,
        cross_validation_label=row.get("cross_validation_label", "") or _cross_validation_label(config, unit) or "",
        condition_id=row["condition_id"],
        pattern_source_name=row["pattern_source_name"],
        roi_source_name=row["roi_source_name"],
        roi_label=row["roi_label"],
        feature_count=int(row["feature_count"]),
        voxel_order=row["voxel_order"],
        voxel_index_hash=row["voxel_index_hash"],
        feature_space_id=row["feature_space_id"],
        roi_definition_id=row["roi_definition_id"],
        usable=_parse_bool(row["usable"]),
        status=row["status"].strip().lower(),
        mean_centering_applied=_parse_bool(row["mean_centering_applied"]),
        mean_centering_scope=row["mean_centering_scope"],
        noise_status=row["noise_status"].strip().lower(),
        noise_usable=_parse_bool(row["noise_usable"]),
        event_count=int(row["event_count"]) if row.get("event_count", "").strip() else None,
        qc_status=_text(row.get("qc_status")),
        qc_reason=_text(row.get("qc_reason")),
        exclusion_id=_text(row.get("exclusion_id")),
        exclusion_reason=_text(row.get("exclusion_reason")),
        roi_reference=_text(row.get("roi_reference")),
        generator_version=_text(row.get("generator_version")),
        software_version=_text(row.get("software_version")),
        derivation_id=_text(row.get("derivation_id")),
        holdout_id=_text(row.get("holdout_id")),
        unit_metadata=dict(unit.metadata),
        extras=extras,
    )


def _unit_metadata_columns(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten safe non-reserved unit metadata for existing row consumers."""

    return {
        str(column): value
        for column, value in metadata.items()
        if str(column) not in FIXED_COLUMNS
        and str(column) not in _DERIVED_RESERVED_COLUMNS
    }


def _loaded_noise_payload(
    plan: MaterializedPatternTablePlan,
    row: Mapping[str, str],
    *,
    scalar: MaterializedPatternScalarRow,
    errors: list[str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    # The planner records the configured method in every canonical row.
    diagonal = bool(plan.pattern_rows) and any(item.noise_reference is not None for item in plan.pattern_rows)
    if not diagonal:
        nonempty = any(
            row.get(column, "").strip()
            for column in OPTIONAL_COLUMNS
            if column.startswith("noise_") and column not in {"noise_status", "noise_usable"}
        )
        warnings = (
            (f"Materialized pattern {scalar.pattern_id!r} noise payload is unused under identity normalization.",)
            if nonempty
            else ()
        )
        return {
            "noise_status": "unused",
            "noise_usable": False,
            "noise_values": (),
            "noise_feature_count": 0,
            "noise_voxel_order": None,
            "noise_voxel_index_hash": None,
        }, warnings

    values = _numeric_json_vector(
        row.get("noise_values", ""),
        label="noise_values",
        data_row=scalar.data_row,
        strictly_positive=True,
        errors=errors,
    )
    count = _positive_int(row.get("noise_feature_count", ""), "noise_feature_count", scalar.data_row, errors)
    if values is not None and count is not None and len(values) != count:
        errors.append(f"Materialized table data row {scalar.data_row} noise_feature_count does not match noise_values.")
    if count is not None and count != scalar.feature_count:
        errors.append(f"Materialized table data row {scalar.data_row} feature and noise counts differ.")
    for feature_column, noise_column in (
        ("voxel_order", "noise_voxel_order"),
        ("voxel_index_hash", "noise_voxel_index_hash"),
        ("feature_space_id", "noise_feature_space_id"),
        ("roi_definition_id", "noise_roi_definition_id"),
    ):
        if row.get(feature_column) != row.get(noise_column):
            errors.append(
                f"Materialized table data row {scalar.data_row} {noise_column} "
                "does not match its feature metadata."
            )
    if row.get("noise_value_kind", "").strip().lower() != "variance":
        errors.append(f"Materialized table data row {scalar.data_row} noise values must be declared as variances.")
    if not row.get("noise_estimation_scope", "").strip() or not row.get("noise_source", "").strip():
        errors.append(f"Materialized table data row {scalar.data_row} noise estimation scope and source are required.")
    return {
        "noise_status": scalar.noise_status,
        "noise_usable": scalar.noise_usable,
        "noise_values": values or (),
        "noise_feature_count": count,
        "noise_voxel_order": row.get("noise_voxel_order"),
        "noise_voxel_index_hash": row.get("noise_voxel_index_hash"),
        "noise_feature_space_id": row.get("noise_feature_space_id"),
        "noise_roi_definition_id": row.get("noise_roi_definition_id"),
        "noise_value_kind": row.get("noise_value_kind"),
        "noise_estimation_scope": row.get("noise_estimation_scope"),
        "noise_source": row.get("noise_source"),
    }, ()


def _loaded_group_consistency_errors(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: MaterializedPatternTablePlan,
) -> tuple[str, ...]:
    errors: list[str] = []
    signatures: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    noise_signatures: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for row in rows:
        feature_group_key = (
            row.get("subject_id"),
            row.get("session_id"),
            row.get("task_id"),
            row.get("pattern_source_name"),
            row.get("roi_source_name"),
            row.get("roi_label"),
        )
        signature = (
            row.get("feature_count"),
            row.get("voxel_order"),
            row.get("voxel_index_hash"),
            row.get("feature_space_id"),
            row.get("roi_definition_id"),
        )
        previous = signatures.setdefault(feature_group_key, signature)
        if previous != signature:
            errors.append("Materialized feature metadata is inconsistent within an analysis/ROI group.")
            break
        noise_group_key = (
            row.get("unit_id"),
            row.get("cross_validation_label"),
            row.get("pattern_source_name"),
            row.get("roi_source_name"),
            row.get("roi_label"),
        )
        noise_signature = (
            row.get("noise_status"),
            row.get("noise_usable"),
            tuple(row.get("noise_values") or ()),
            row.get("noise_feature_count"),
            row.get("noise_voxel_order"),
            row.get("noise_voxel_index_hash"),
            row.get("noise_feature_space_id"),
            row.get("noise_roi_definition_id"),
            row.get("noise_value_kind"),
            row.get("noise_estimation_scope"),
            row.get("noise_source"),
        )
        previous_noise = noise_signatures.setdefault(noise_group_key, noise_signature)
        if previous_noise != noise_signature:
            errors.append(
                "Materialized noise metadata or variances differ across conditions "
                "for one unit/ROI/CV group."
            )
            break
    return tuple(errors)


def _numeric_json_vector(
    value: str,
    *,
    label: str,
    data_row: int,
    strictly_positive: bool,
    errors: list[str],
) -> tuple[float, ...] | None:
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        errors.append(f"Materialized table data row {data_row} {label} must be a JSON numeric array.")
        return None
    if not isinstance(decoded, list) or not decoded:
        errors.append(f"Materialized table data row {data_row} {label} must be a non-empty JSON numeric array.")
        return None
    values: list[float] = []
    for item in decoded:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            errors.append(f"Materialized table data row {data_row} {label} contains a non-numeric value.")
            return None
        number = float(item)
        if not math.isfinite(number) or (strictly_positive and number <= 0.0):
            errors.append(f"Materialized table data row {data_row} {label} contains an invalid numeric value.")
            return None
        values.append(number)
    return tuple(values)


def _json_scalar_mapping(
    value: str,
    *,
    label: str,
    data_row: int,
    errors: list[str],
) -> Mapping[str, Any]:
    if not value.strip():
        return {}
    try:
        decoded = json.loads(value)
    except ValueError:
        errors.append(f"Materialized table data row {data_row} {label} must be a JSON object.")
        return {}
    if not isinstance(decoded, dict) or any(isinstance(item, (dict, list)) for item in decoded.values()):
        errors.append(f"Materialized table data row {data_row} {label} must contain only scalar values.")
        return {}
    if any(
        isinstance(item, float) and not math.isfinite(item)
        for item in decoded.values()
    ):
        errors.append(f"Materialized table data row {data_row} {label} contains a non-finite value.")
        return {}
    return {str(key): item for key, item in decoded.items()}


def _json_string_sequence(
    value: str,
    *,
    label: str,
    data_row: int,
    errors: list[str],
) -> tuple[str, ...]:
    if not value.strip():
        return ()
    try:
        decoded = json.loads(value)
    except ValueError:
        errors.append(f"Materialized table data row {data_row} {label} must be a JSON string array.")
        return ()
    if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
        errors.append(f"Materialized table data row {data_row} {label} must be a JSON string array.")
        return ()
    return tuple(decoded)


def _cross_validation_label(config: Any, unit: ResolvedAnalysisUnit) -> str | None:
    cv_unit = str(getattr(getattr(config, "distance", None), "cv_unit", "run"))
    if cv_unit == "subject":
        return unit.subject_id
    if cv_unit == "session":
        return unit.session_id
    if cv_unit == "run":
        return unit.run_id
    columns = tuple(getattr(getattr(config, "distance", None), "grouping_columns", ()))
    values: list[str] = []
    for column in columns:
        value = unit.metadata.get(column)
        if value is None or not str(value).strip():
            return None
        values.append(f"{column}={value}")
    return "|".join(values) if values else None


def _noise_method(config: Any) -> str:
    distance = getattr(config, "distance", None)
    noise = getattr(distance, "noise_normalization", None)
    return str(getattr(noise, "method", "identity"))


def _required_cell(row: Mapping[str, str], column: str, data_row: int, errors: list[str]) -> str:
    value = row.get(column, "")
    if not value.strip():
        errors.append(f"Materialized table data row {data_row} requires non-empty {column}.")
    return value


def _positive_int(value: str, label: str, data_row: int, errors: list[str]) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"Materialized table data row {data_row} {label} must be a positive integer.")
        return None
    if parsed < 1 or str(parsed) != str(value).strip():
        errors.append(f"Materialized table data row {data_row} {label} must be a positive integer.")
        return None
    return parsed


def _optional_nonnegative_int(value: str, label: str, data_row: int, errors: list[str]) -> int | None:
    if not str(value).strip():
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"Materialized table data row {data_row} {label} must be a non-negative integer.")
        return None
    if parsed < 0 or str(parsed) != str(value).strip():
        errors.append(f"Materialized table data row {data_row} {label} must be a non-negative integer.")
        return None
    return parsed


def _bool_value(value: str, label: str, data_row: int, errors: list[str]) -> bool | None:
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    errors.append(f"Materialized table data row {data_row} {label} must be true or false.")
    return None


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in _TRUE_VALUES


def _qc_failure(pattern_id: str, data_row: int, message: str) -> dict[str, Any]:
    return {
        "pattern_id": pattern_id,
        "data_row": data_row,
        "status": "fail",
        "code": "materialized_row_invalid",
        "message": message,
    }


def _loader_provenance(
    plan: MaterializedPatternTablePlan,
    *,
    digest: str,
    loaded_rows: int,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_name": plan.source_name,
        "source_reference": plan.portable_reference,
        "source_sha256": digest,
        "planned_selected_rows": plan.selected_row_count,
        "loaded_rows": loaded_rows,
        "unselected_rows": plan.unselected_row_count,
        "executed": False,
    }
    for column in (
        "roi_reference",
        "generator_version",
        "software_version",
        "derivation_id",
        "holdout_id",
        "exclusion_id",
        "exclusion_reason",
    ):
        values = tuple(_unique([str(row[column]) for row in rows if row.get(column) not in {None, ""}]))
        if values:
            provenance[column + "s"] = values
    return provenance


def _load_failure(plan: MaterializedPatternTablePlan, *errors: str) -> MaterializedPatternTableLoadResult:
    return MaterializedPatternTableLoadResult(
        rows=(),
        qc_rows=(),
        provenance={
            "schema_version": SCHEMA_VERSION,
            "source_name": plan.source_name,
            "source_reference": plan.portable_reference,
            "source_sha256": plan.source_sha256,
            "loaded_rows": 0,
            "executed": False,
        },
        errors=tuple(errors),
        materialized=False,
        executed=False,
    )


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


__all__ = [
    "FIXED_COLUMNS",
    "MaterializedPatternScalarRow",
    "MaterializedPatternTableLoadResult",
    "MaterializedPatternTablePlan",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "load_materialized_pattern_table",
    "plan_materialized_pattern_table",
    "validate_materialized_pattern_source_fields",
]
