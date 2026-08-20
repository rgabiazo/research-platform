"""Representation-aware, in-memory MVPA pattern runtime dispatch.

This module owns only the pattern materialization boundary.  It does not
prepare analysis groups, compute distances, write outputs, or re-plan pattern
sources.  Prepared-feature execution consumes the private adapter handle kept
by the exact discovery plan so the planned source digest remains authoritative.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
import math

from .extraction import extract_mvpa_patterns_from_discovery_plan
from .materialized_pattern_table import (
    MaterializedPatternTablePlan,
    load_materialized_pattern_table,
)
from .pattern_sources import (
    PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE,
    REPRESENTATION_IMAGE,
    REPRESENTATION_PREPARED_FEATURES,
    PatternSourceExecutionHandle,
)


class MvpaRuntimeRepresentationError(ValueError):
    """Raised when a discovery plan has no safe implemented runtime route."""


@dataclass(frozen=True)
class MvpaPatternRuntimeResult:
    """Backend-neutral in-memory patterns at the analysis preparation boundary."""

    representation_kind: str
    source_audit_kind: str
    pattern_rows: tuple[Mapping[str, Any], ...] = ()
    qc_rows: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    executed: bool = True

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


def materialize_mvpa_patterns_from_plan(
    plan: Any,
    *,
    load_noise: bool = False,
) -> MvpaPatternRuntimeResult:
    """Materialize one exact discovery plan without re-planning its sources.

    Image plans retain the existing NIfTI/ROI extraction implementation.
    Prepared-feature plans use only opaque handles retained by this exact plan;
    callers cannot substitute a path or digest at execution time.
    """

    representations = _planned_representation_kinds(plan)
    if len(representations) != 1:
        if not representations:
            raise MvpaRuntimeRepresentationError(
                "MVPA execution requires one planned image or prepared_features representation."
            )
        raise MvpaRuntimeRepresentationError(
            "MVPA execution does not support mixed image and prepared_features sources in v1."
        )
    representation = representations[0]
    if representation == REPRESENTATION_IMAGE:
        return _materialize_image_plan(plan, load_noise=load_noise)
    if representation == REPRESENTATION_PREPARED_FEATURES:
        return _materialize_prepared_plan(plan)
    raise MvpaRuntimeRepresentationError(
        f"MVPA representation {representation!r} has no implemented local runtime adapter."
    )


def _materialize_image_plan(plan: Any, *, load_noise: bool) -> MvpaPatternRuntimeResult:
    extracted = extract_mvpa_patterns_from_discovery_plan(plan, load_noise=load_noise)
    provenance = {
        "representation_kind": REPRESENTATION_IMAGE,
        "source_audit_kind": "image_extraction",
        "runtime": _mapping_value(extracted, "provenance", {}),
    }
    return MvpaPatternRuntimeResult(
        representation_kind=REPRESENTATION_IMAGE,
        source_audit_kind="image_extraction",
        pattern_rows=_mapping_rows(_mapping_value(extracted, "pattern_rows", ())),
        qc_rows=_mapping_rows(_mapping_value(extracted, "qc_rows", ())),
        provenance=_json_safe(provenance),
        warnings=_text_sequence(_mapping_value(extracted, "warnings", ())),
        errors=_text_sequence(_mapping_value(extracted, "errors", ())),
        executed=bool(_mapping_value(extracted, "executed", True)),
    )


def _materialize_prepared_plan(plan: Any) -> MvpaPatternRuntimeResult:
    handles = _private_execution_handles(plan)
    planned_sources = _planned_prepared_source_names(plan)
    prepared_handles = tuple(
        handle
        for handle in handles
        if handle.representation_kind == REPRESENTATION_PREPARED_FEATURES
    )
    handle_sources = tuple(handle.source_name for handle in prepared_handles)
    if not prepared_handles or handle_sources != planned_sources:
        raise MvpaRuntimeRepresentationError(
            "Prepared-feature execution requires the private handles retained by the exact discovery plan."
        )

    rows: list[Mapping[str, Any]] = []
    qc_rows: list[Mapping[str, Any]] = []
    provenance_rows: list[Mapping[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    for handle in prepared_handles:
        if handle.backend_name != PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE:
            raise MvpaRuntimeRepresentationError(
                f"Prepared-feature backend {handle.backend_name!r} has no implemented local runtime adapter."
            )
        table_plan = handle.payload
        if not isinstance(table_plan, MaterializedPatternTablePlan):
            raise MvpaRuntimeRepresentationError(
                "The retained materialized-pattern execution handle has an invalid internal type."
            )
        if table_plan.source_sha256 is None:
            raise MvpaRuntimeRepresentationError(
                "The retained materialized-pattern execution handle has no planned source digest."
            )
        loaded = load_materialized_pattern_table(
            table_plan,
            expected_sha256=table_plan.source_sha256,
        )
        qc_rows.extend(_mapping_rows(loaded.qc_rows))
        warnings.extend(loaded.warnings)
        errors.extend(loaded.errors)
        provenance_rows.append(
            {
                "source_name": handle.source_name,
                "backend_name": handle.backend_name,
                **dict(loaded.provenance),
            }
        )
        if loaded.valid and loaded.materialized:
            rows.extend(_mapping_rows(loaded.rows))

    if errors:
        rows = []
    return MvpaPatternRuntimeResult(
        representation_kind=REPRESENTATION_PREPARED_FEATURES,
        source_audit_kind="pattern_materialization",
        pattern_rows=tuple(rows),
        qc_rows=tuple(qc_rows),
        provenance={
            "representation_kind": REPRESENTATION_PREPARED_FEATURES,
            "source_audit_kind": "pattern_materialization",
            "sources": tuple(provenance_rows),
        },
        warnings=tuple(dict.fromkeys(str(value) for value in warnings)),
        errors=tuple(dict.fromkeys(str(value) for value in errors)),
        executed=not errors,
    )


def _planned_representation_kinds(plan: Any) -> tuple[str, ...]:
    values: list[str] = []
    for row in _mapping_rows(_mapping_value(plan, "pattern_rows", ())):
        value = str(row.get("representation_kind") or "").strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


def _planned_prepared_source_names(plan: Any) -> tuple[str, ...]:
    names: list[str] = []
    for row in _mapping_rows(_mapping_value(plan, "pattern_rows", ())):
        if row.get("representation_kind") != REPRESENTATION_PREPARED_FEATURES:
            continue
        name = str(row.get("source_name") or "").strip()
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _private_execution_handles(plan: Any) -> tuple[PatternSourceExecutionHandle, ...]:
    raw = getattr(plan, "_execution_handles", ())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        handles = tuple(raw)
    else:
        handles = ()
    if any(not isinstance(handle, PatternSourceExecutionHandle) for handle in handles):
        raise MvpaRuntimeRepresentationError("The MVPA plan contains an invalid private execution handle.")
    return handles


def _mapping_rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return ()
    rows: list[Mapping[str, Any]] = []
    for row in value:
        mapped = _as_mapping(row)
        if mapped is None:
            raise MvpaRuntimeRepresentationError("MVPA runtime rows must be mappings or dataclasses.")
        rows.append(_json_safe_mapping(mapped))
    return tuple(rows)


def _mapping_value(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        return mapped if isinstance(mapped, Mapping) else None
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    return None


def _text_sequence(value: Any) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return () if value is None else (str(value),)
    return tuple(dict.fromkeys(str(item) for item in value))


def _json_safe_dataclass(value: Any) -> dict[str, Any]:
    return {item.name: _json_safe(getattr(value, item.name)) for item in fields(value)}


def _json_safe_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(item) for key, item in value.items()}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe_dataclass(value)
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("MVPA runtime results must not contain non-finite values.")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "MvpaPatternRuntimeResult",
    "MvpaRuntimeRepresentationError",
    "materialize_mvpa_patterns_from_plan",
]
