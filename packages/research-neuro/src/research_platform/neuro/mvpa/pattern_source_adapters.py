"""Explicit MVPA pattern-source adapter registry.

The registry is intentionally static for the public alpha.  Importing this
module does not discover plugins, inspect the filesystem, load images, or run
external software.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_platform.neuro._roi_path_safety import (
    UnmappedLocalPathError,
    portable_path_reference,
    published_value_local_path_fields,
)

from .config import validate_fsl_feat_pe_source_fields
from .fsl_feat import plan_fsl_feat_pattern_source
from .materialized_pattern_table import (
    plan_materialized_pattern_table,
    validate_materialized_pattern_source_fields,
)
from .pattern_sources import (
    PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE,
    PATTERN_BACKEND_FSL_FEAT_PE,
    PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE,
    PATTERN_BACKEND_NILEARN_GLM,
    PATTERN_BACKEND_SURFACE_CIFTI,
    REPRESENTATION_IMAGE,
    REPRESENTATION_PREPARED_FEATURES,
    PatternSourceAdapterCapabilities,
    PatternSourceAdapterPlan,
    PatternSourceAdapterRegistry,
    PlannedPatternRow,
    ResolvedAnalysisUnit,
)


_FSL_BACKEND_UNIT_METADATA_KEYS = frozenset(
    {
        "cope",
        "cope_number",
        "design_fsf",
        "design_row",
        "ev_index",
        "feat_dir",
        "fsl_ev_title",
        "pe_image",
        "pe_number",
    }
)


@dataclass(frozen=True)
class FslFeatPePatternSourceAdapter:
    """Plan FSL FEAT parameter-estimate images behind the shared contract."""

    name: str = PATTERN_BACKEND_FSL_FEAT_PE
    status: str = "available_external_runtime"
    capabilities: PatternSourceAdapterCapabilities = PatternSourceAdapterCapabilities(
        status="available_external_runtime",
        schema_supported=True,
        planning_supported=True,
        execution_ready=False,
        representation_kinds=(REPRESENTATION_IMAGE,),
        reason=(
            "FSL FEAT planning is integrated, but CLI execution is deferred until portable "
            "feature-space, ROI-definition, and diagonal-noise provenance identities are available."
        ),
    )

    def validate_source(self, source: Mapping[str, Any], label: str) -> tuple[str, ...]:
        return validate_fsl_feat_pe_source_fields(source, label)

    def plan_source(
        self,
        *,
        config: Any,
        source: Any,
        units: Sequence[ResolvedAnalysisUnit],
        roots: Mapping[str, str | Path] | None,
        context: Mapping[str, Any] | None,
        raise_on_fail_policy: bool = False,
    ) -> PatternSourceAdapterPlan:
        unit_contexts = runtime_unit_contexts(config, units, context=context)
        compatibility = plan_fsl_feat_pattern_source(
            config,
            pattern_source_name=source.name,
            roots=roots,
            context=context,
            unit_contexts=unit_contexts,
            raise_on_fail_policy=raise_on_fail_policy,
        )
        pattern_rows, portability_errors = _canonical_fsl_pattern_rows(
            config=config,
            source=source,
            units=units,
            compatibility_units=compatibility.units,
            condition_rows=compatibility.condition_pe_rows,
            roots=roots,
        )
        errors = _unique_text((*compatibility.errors, *portability_errors))
        status = "error" if errors else compatibility.status
        return PatternSourceAdapterPlan(
            adapter_name=self.name,
            status=status,
            ready_for_execution=False,
            pattern_rows=pattern_rows,
            compatibility_source_rows=compatibility.units,
            compatibility_condition_rows=compatibility.condition_pe_rows,
            input_checks=compatibility.input_checks,
            warnings=compatibility.warnings,
            errors=errors,
            context=compatibility.context,
            executed=False,
        )


@dataclass(frozen=True)
class DeferredPatternSourceAdapter:
    """Truthful registered placeholder for a structurally supported backend."""

    name: str
    representation_kind: str
    reason: str
    status: str = "deferred"

    @property
    def capabilities(self) -> PatternSourceAdapterCapabilities:
        return PatternSourceAdapterCapabilities(
            status=self.status,
            schema_supported=True,
            planning_supported=False,
            execution_ready=False,
            representation_kinds=(self.representation_kind,),
            reason=self.reason,
        )

    def validate_source(self, source: Mapping[str, Any], label: str) -> tuple[str, ...]:
        del source, label
        return ()

    def plan_source(
        self,
        *,
        config: Any,
        source: Any,
        units: Sequence[ResolvedAnalysisUnit],
        roots: Mapping[str, str | Path] | None,
        context: Mapping[str, Any] | None,
        raise_on_fail_policy: bool = False,
    ) -> PatternSourceAdapterPlan:
        del config, source, units, roots, raise_on_fail_policy
        return PatternSourceAdapterPlan(
            adapter_name=self.name,
            status="deferred",
            ready_for_execution=False,
            context=dict(context or {}),
            executed=False,
        )


@dataclass(frozen=True)
class MaterializedPatternTableAdapter:
    """Plan and expose digest-checked prepared-feature table materialization."""

    name: str = PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE
    status: str = "available_local_runtime"
    capabilities: PatternSourceAdapterCapabilities = PatternSourceAdapterCapabilities(
        status="available_local_runtime",
        schema_supported=True,
        planning_supported=True,
        execution_ready=True,
        materialization_supported=True,
        representation_kinds=(REPRESENTATION_PREPARED_FEATURES,),
        reason="Materialized table planning and digest-checked local materialization are available.",
    )

    def validate_source(self, source: Mapping[str, Any], label: str) -> tuple[str, ...]:
        return validate_materialized_pattern_source_fields(source, label)

    def plan_source(
        self,
        *,
        config: Any,
        source: Any,
        units: Sequence[ResolvedAnalysisUnit],
        roots: Mapping[str, str | Path] | None,
        context: Mapping[str, Any] | None,
        raise_on_fail_policy: bool = False,
    ) -> PatternSourceAdapterPlan:
        del raise_on_fail_policy
        table_plan = plan_materialized_pattern_table(
            config,
            source,
            units,
            roots=roots,
        )
        status = "valid" if table_plan.valid else "error"
        summary = {
            **table_plan.public_summary(),
            "ready_for_execution": table_plan.ready_for_execution,
            "execution_reason": (
                "digest_checked_local_runtime"
                if table_plan.ready_for_execution
                else "materialization_usable_coverage_or_event_thresholds_not_ready"
            ),
        }
        provenance = {
            "schema_version": summary["schema_version"],
            "source_name": summary["source_name"],
            "source_reference": summary["portable_reference"],
            "source_sha256": summary["source_sha256"],
            "columns": summary["columns"],
            "counts": summary["counts"],
            "unselected_pattern_ids": summary["unselected_pattern_ids"],
            "executed": False,
        }
        return PatternSourceAdapterPlan(
            adapter_name=self.name,
            status=status,
            ready_for_execution=table_plan.ready_for_execution,
            ready_for_materialization=table_plan.ready_for_materialization,
            pattern_rows=table_plan.pattern_rows,
            source_rows=table_plan.public_source_rows(),
            source_summary=summary,
            source_provenance=provenance,
            compatibility_source_rows=(),
            compatibility_condition_rows=(),
            input_checks=(),
            warnings=table_plan.warnings,
            errors=table_plan.errors,
            context=dict(context or {}),
            materialization_handle=table_plan,
            executed=False,
        )


def default_pattern_source_adapter_registry() -> PatternSourceAdapterRegistry:
    """Return the authoritative, explicitly ordered alpha adapter registry."""

    return PatternSourceAdapterRegistry(
        (
            FslFeatPePatternSourceAdapter(),
            MaterializedPatternTableAdapter(),
            DeferredPatternSourceAdapter(
                name=PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE,
                representation_kind=REPRESENTATION_PREPARED_FEATURES,
                reason="The BIDS-derivative pattern-table adapter is not implemented.",
            ),
            DeferredPatternSourceAdapter(
                name=PATTERN_BACKEND_NILEARN_GLM,
                representation_kind=REPRESENTATION_IMAGE,
                reason="The Nilearn GLM adapter is not implemented.",
            ),
            DeferredPatternSourceAdapter(
                name=PATTERN_BACKEND_SURFACE_CIFTI,
                representation_kind=REPRESENTATION_IMAGE,
                reason="The surface/CIFTI adapter is not implemented.",
            ),
        )
    )


def _canonical_fsl_pattern_rows(
    *,
    config: Any,
    source: Any,
    units: Sequence[ResolvedAnalysisUnit],
    compatibility_units: Sequence[Any],
    condition_rows: Sequence[Any],
    roots: Mapping[str, str | Path] | None,
) -> tuple[tuple[PlannedPatternRow, ...], tuple[str, ...]]:
    rows: list[PlannedPatternRow] = []
    errors: list[str] = []
    condition_count = len(config.conditions)
    if len(compatibility_units) != len(units):
        errors.append("FSL adapter returned a pattern-unit count that does not match the exact unit input.")
        return (), tuple(errors)
    if len(condition_rows) != len(units) * condition_count:
        errors.append("FSL adapter returned a condition-row count that does not match unit-by-condition planning.")
        return (), tuple(errors)

    for unit_index, unit in enumerate(units):
        unsafe_metadata = published_value_local_path_fields(unit.metadata, label="unit_metadata")
        if unsafe_metadata:
            errors.append(
                f"Pattern source {source.name!r} unit {unit.unit_id!r} contains a non-portable metadata reference."
            )
            continue
        compatibility_unit = compatibility_units[unit_index]
        shared_unit_metadata, backend_unit_metadata = _partition_fsl_unit_metadata(
            unit.metadata
        )
        cv_label = _cross_validation_label(config, unit)
        if cv_label is None:
            errors.append(
                f"Pattern source {source.name!r} unit {unit.unit_id!r} lacks the configured cross-validation value."
            )
            continue
        unit_condition_rows = condition_rows[
            unit_index * condition_count : (unit_index + 1) * condition_count
        ]
        for condition_row in unit_condition_rows:
            try:
                pattern_reference = _portable_reference(condition_row.pe_image, roots=roots)
                noise_reference = _portable_reference(condition_row.noise_image, roots=roots)
                feat_reference = _portable_reference(compatibility_unit.feat_dir, roots=roots)
                design_reference = _portable_reference(compatibility_unit.design_fsf, roots=roots)
                event_reference = _portable_reference(condition_row.event_file, roots=roots)
            except UnmappedLocalPathError:
                errors.append(
                    f"Pattern source {source.name!r} unit {unit.unit_id!r} has a local path outside configured roots."
                )
                continue
            if pattern_reference is None:
                if condition_row.status in {"ok", "valid"}:
                    errors.append(
                        f"Pattern source {source.name!r} unit {unit.unit_id!r} has no portable pattern reference."
                    )
                continue
            rows.append(
                PlannedPatternRow(
                    unit_id=unit.unit_id,
                    subject_id=unit.subject_id,
                    session_id=unit.session_id,
                    task_id=unit.task_id,
                    run_id=unit.run_id,
                    cross_validation_label=cv_label,
                    condition_id=condition_row.condition_id,
                    source_name=source.name,
                    backend_name=source.backend,
                    representation_kind=REPRESENTATION_IMAGE,
                    pattern_reference=pattern_reference,
                    noise_reference=noise_reference,
                    event_count=condition_row.event_count,
                    qc_status=compatibility_unit.status,
                    status=condition_row.status,
                    unit_metadata=shared_unit_metadata,
                    backend_metadata={
                        "requested_ev_title": condition_row.requested_ev_title,
                        "matched_ev_title": condition_row.matched_ev_title,
                        "matched_alias": condition_row.matched_alias,
                        "ev_index": condition_row.ev_index,
                        "pe_number": condition_row.pe_number,
                        "feat_directory_reference": feat_reference,
                        "design_reference": design_reference,
                        "event_reference": event_reference,
                        "unit_metadata": backend_unit_metadata,
                    },
                )
            )
    return tuple(rows), tuple(_unique_text(errors))


def runtime_unit_contexts(
    config: Any,
    units: Sequence[ResolvedAnalysisUnit],
    *,
    context: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    """Return ordered adapter aliases without changing canonical identities."""

    return tuple(_runtime_unit_context(config, unit, context=context) for unit in units)


def _runtime_unit_context(
    config: Any,
    unit: ResolvedAnalysisUnit,
    *,
    context: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    row: dict[str, Any] = dict(context or {})
    row.update(unit.metadata)
    row.update(_entity_context(config))
    subject_id, subject = _entity_alias(unit.subject_id, "sub")
    session_id, session = _optional_entity_alias(unit.session_id, "ses")
    run_id, run = _optional_entity_alias(unit.run_id, "run")
    task_value = unit.task_id or getattr(getattr(config, "entities", None), "task", None)
    task_id = _strip_entity_prefix(task_value, "task") if task_value is not None else None
    row.update(
        {
            "subject_id": subject_id,
            "subject": subject,
            "subject_dir": subject,
            "session_id": session_id,
            "session": session,
            "session_dir": session,
            "run_id": run_id,
            "run": run,
            "run_entity": run,
            "task_id": task_id,
            "task": task_id,
            "mvpa_set": config.name,
        }
    )
    return row


def _partition_fsl_unit_metadata(
    metadata: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    shared: dict[str, Any] = {}
    backend: dict[str, Any] = {}
    for key, value in metadata.items():
        target = backend if str(key).casefold() in _FSL_BACKEND_UNIT_METADATA_KEYS else shared
        target[str(key)] = value
    return shared, backend


def _entity_context(config: Any) -> Mapping[str, Any]:
    entities = getattr(config, "entities", None)
    direction = _strip_entity_prefix(getattr(entities, "direction", None), "dir")
    resolution = _strip_entity_prefix(getattr(entities, "resolution", None), "res")
    return {
        "direction": direction or None,
        "dir": direction or None,
        "model": getattr(entities, "model", None),
        "space": getattr(entities, "space", None),
        "resolution": resolution or None,
        "res": resolution or None,
    }


def _entity_alias(value: str, prefix: str) -> tuple[str, str]:
    entity_id = _strip_entity_prefix(value, prefix)
    entity = value if value.startswith(f"{prefix}-") else f"{prefix}-{entity_id}"
    return entity_id, entity


def _optional_entity_alias(value: str | None, prefix: str) -> tuple[str, str]:
    if value is None:
        return "", ""
    return _entity_alias(value, prefix)


def _strip_entity_prefix(value: Any, prefix: str) -> str:
    text = "" if value is None else str(value).strip()
    marker = f"{prefix}-"
    return text[len(marker) :] if text.startswith(marker) else text


def _cross_validation_label(config: Any, unit: ResolvedAnalysisUnit) -> str | None:
    cv_unit = str(config.distance.cv_unit)
    if cv_unit == "subject":
        return unit.subject_id
    if cv_unit == "session":
        return unit.session_id
    if cv_unit == "run":
        return unit.run_id
    columns = tuple(config.distance.grouping_columns)
    values: list[str] = []
    for column in columns:
        value = unit.metadata.get(column)
        if value is None or not str(value).strip():
            return None
        values.append(f"{column}={value}")
    return "|".join(values) if values else None


def _portable_reference(
    value: str | Path | None,
    *,
    roots: Mapping[str, str | Path] | None,
) -> str | None:
    if value is None:
        return None
    return portable_path_reference(value, named_roots=roots)


def _unique_text(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


__all__ = [
    "DeferredPatternSourceAdapter",
    "FslFeatPePatternSourceAdapter",
    "MaterializedPatternTableAdapter",
    "default_pattern_source_adapter_registry",
    "runtime_unit_contexts",
]
