"""Generic analysis workflow and exact-unit bundle contracts.

This module validates dictionary-shaped workflow recipes and resolves
configuration-owned bundle selections over caller-supplied manifest tables. It
does not discover inputs, execute stages, submit jobs, call domain packages, or
write outputs. Domain-specific extension payloads and component references are
carried for owner packages to validate or execute separately.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from numbers import Number
from typing import Any
import json
import re

from .manifests import (
    ManifestTable,
    manifest_row_matches,
    normalize_manifest_identity,
    normalized_manifest_filters,
    unknown_manifest_filter_columns,
)


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_SELECTOR_VALUE = re.compile(r"^[A-Za-z0-9_.{}<>:-]+$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(^|[=:\s])/[^\s]+"),
    re.compile(r"(^|[=:\s])~(?:/|$)"),
    re.compile(r"(^|[=:\s])[A-Za-z]:[\\/][^\s]+"),
)
_PATH_LIKE_KEYS = frozenset(
    {
        "path",
        "paths",
        "relative_path",
        "relative_paths",
        "path_template",
        "path_templates",
        "template",
        "templates",
        "pattern",
        "patterns",
        "root",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "command",
        "commands",
        "container",
        "executable",
        "execute",
        "execution",
        "run_command",
        "script",
        "scripts",
    }
)
_ANALYSIS_BUNDLE_COMPONENT_PATHS = {
    "roi_set": "roi_sets",
    "extraction_set": "extraction_sets",
    "mvpa_set": "mvpa",
}
_ANALYSIS_BUNDLE_STAGE_COMPONENTS = {
    "roi_build": ("roi_set",),
    "roi_extraction": ("roi_set", "extraction_set"),
    "mvpa": ("mvpa_set",),
}
_ANALYSIS_BUNDLE_CHECK_IDS = (
    "bundle_schema",
    "selection_exists",
    "required_and_filter_columns",
    "duplicate_unit_keys",
    "longitudinal_completeness",
    "component_configs",
    "stage_component_consistency",
)


@dataclass(frozen=True)
class AnalysisWorkflowSelector:
    """Subject/session/task/run selectors for a generic workflow recipe."""

    subjects: tuple[str, ...]
    sessions: tuple[str, ...]
    tasks: tuple[str, ...]
    runs: tuple[str, ...]
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisRootRef:
    """A named root reference that stages and extensions may point at."""

    name: str
    description: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisOutputRoot:
    """A named root-ref plus relative output path template."""

    name: str
    root_ref: str
    path: str
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisWorkflowStage:
    """One configured workflow stage declaration."""

    name: str
    kind: str
    extension: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisPublicationSettings:
    """Publication metadata common to analysis workflow recipes."""

    enabled: bool = False
    derivative_name: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisReportingSettings:
    """Reporting metadata common to analysis workflow recipes."""

    enabled: bool = False
    formats: tuple[str, ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisWorkflowConfig:
    """Validated generic analysis workflow recipe."""

    name: str
    selector: AnalysisWorkflowSelector
    stages: tuple[AnalysisWorkflowStage, ...]
    root_refs: tuple[AnalysisRootRef, ...] = ()
    output_roots: tuple[AnalysisOutputRoot, ...] = ()
    runtime_tag: str | None = None
    publication: AnalysisPublicationSettings = field(default_factory=AnalysisPublicationSettings)
    reporting: AnalysisReportingSettings = field(default_factory=AnalysisReportingSettings)
    extensions: Mapping[str, Any] = field(default_factory=dict)
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisWorkflowStagePlanRow:
    """Plan row for a configured workflow stage."""

    name: str
    kind: str
    extension: str | None = None
    status: str = "configured"
    reason: str = "plan_only_stage_declaration"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AnalysisRootRefPlanRow:
    """Plan row for a root reference declaration."""

    name: str
    status: str = "declared"
    reason: str = "root_resolution_deferred_to_caller"
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AnalysisOutputRootPlanRow:
    """Plan row for a root-ref plus relative output path template."""

    name: str
    root_ref: str
    relative_path_template: str
    status: str = "preview"
    reason: str = "output_materialization_not_implemented"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AnalysisExtensionPlanRow:
    """Plan row noting that an owner package should validate an extension."""

    name: str
    status: str = "deferred"
    reason: str = "extension_validation_deferred_to_owner_package"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AnalysisWorkflowPlan:
    """JSON-serializable generic analysis workflow plan preview."""

    workflow_name: str | None
    status: str
    errors: tuple[str, ...] = ()
    runtime_tag: str | None = None
    subjects: tuple[str, ...] = ()
    sessions: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    runs: tuple[str, ...] = ()
    stages: tuple[AnalysisWorkflowStagePlanRow, ...] = ()
    root_refs: tuple[AnalysisRootRefPlanRow, ...] = ()
    output_roots: tuple[AnalysisOutputRootPlanRow, ...] = ()
    publication: Mapping[str, Any] = field(default_factory=dict)
    reporting: Mapping[str, Any] = field(default_factory=dict)
    extensions: tuple[AnalysisExtensionPlanRow, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    executed: bool = False

    @property
    def valid(self) -> bool:
        """Return whether validation passed before plan assembly."""

        return self.status != "invalid"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        payload["plan_only"] = True
        return payload


@dataclass(frozen=True)
class AnalysisBundleSelection:
    """Exactly one named cohort or batch selected by an analysis bundle."""

    cohort: str | None = None
    batch: str | None = None


@dataclass(frozen=True)
class AnalysisBundleUnits:
    """Configuration-owned unit identity and optional longitudinal policy."""

    key_columns: tuple[str, ...]
    subject_column: str = "subject_id"
    occasion_column: str | None = None
    occasion_order_column: str | None = None
    required_occasions: tuple[str, ...] = ()
    incomplete: str = "allow"


@dataclass(frozen=True)
class AnalysisBundleConfig:
    """Validated plan-only analysis-bundle configuration."""

    name: str
    selection: AnalysisBundleSelection
    units: AnalysisBundleUnits
    components: Mapping[str, str]
    stages: tuple[str, ...]
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisCohortExclusion:
    """One stable, auditable exclusion rule in a named cohort view."""

    id: str
    filters: Mapping[str, tuple[str, ...]]
    reason: str | None = None
    reason_field: str | None = None


@dataclass(frozen=True)
class AnalysisCohortView:
    """A named filtered view over one canonical batch manifest."""

    name: str
    batch: str
    include: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    exclude: tuple[AnalysisCohortExclusion, ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisBundleCheck:
    """Stable doctor check emitted by the authoritative bundle resolver."""

    id: str
    status: str
    messages: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status in {"pass", "warning"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "ok": self.ok,
            "messages": list(self.messages),
        }


@dataclass(frozen=True)
class AnalysisBundleResolution:
    """Deterministic plan-only result for one exact-row analysis bundle."""

    bundle_name: str | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: tuple[AnalysisBundleCheck, ...] = ()
    selection: Mapping[str, Any] = field(default_factory=dict)
    source_batch: Mapping[str, Any] = field(default_factory=dict)
    key_columns: tuple[str, ...] = ()
    subject_column: str = "subject_id"
    occasion_column: str | None = None
    occasion_order_column: str | None = None
    required_occasions: tuple[str, ...] = ()
    incomplete_policy: str = "allow"
    included_units: tuple[Mapping[str, Any], ...] = ()
    excluded_units: tuple[Mapping[str, Any], ...] = ()
    not_included_units: tuple[Mapping[str, Any], ...] = ()
    dropped_units: tuple[Mapping[str, Any], ...] = ()
    incomplete_subjects: tuple[Mapping[str, Any], ...] = ()
    occasion_order: tuple[Mapping[str, Any], ...] = ()
    unmatched_exclusion_rules: tuple[str, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)
    components: Mapping[str, str] = field(default_factory=dict)
    stages: tuple[str, ...] = ()
    source_batch_sha256: str = ""
    effective_config_sha256: str = ""
    plan_digest: str = ""
    executed: bool = False

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def ready_for_planning(self) -> bool:
        return self.valid

    def to_dict(self) -> dict[str, Any]:
        """Return the complete host-independent plan payload."""

        return {
            "bundle_name": self.bundle_name,
            "valid": self.valid,
            "ready_for_planning": self.ready_for_planning,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": [check.to_dict() for check in self.checks],
            "selection": _json_safe(self.selection),
            "source_batch": _json_safe(self.source_batch),
            "units": {
                "key_columns": list(self.key_columns),
                "subject_column": self.subject_column,
                "occasion_column": self.occasion_column,
                "occasion_order_column": self.occasion_order_column,
                "required_occasions": list(self.required_occasions),
                "incomplete": self.incomplete_policy,
                "included": _json_safe(self.included_units),
                "excluded": _json_safe(self.excluded_units),
                "not_included": _json_safe(self.not_included_units),
                "dropped": _json_safe(self.dropped_units),
                "incomplete_subjects": _json_safe(self.incomplete_subjects),
                "occasion_order": _json_safe(self.occasion_order),
            },
            "counts": _json_safe(self.counts),
            "unmatched_exclusion_rules": list(self.unmatched_exclusion_rules),
            "components": _json_safe(self.components),
            "stages": list(self.stages),
            "digests": {
                "source_batch_sha256": self.source_batch_sha256,
                "effective_config_sha256": self.effective_config_sha256,
                "plan_sha256": self.plan_digest,
            },
            "plan_only": True,
            "executed": self.executed,
        }


def validate_analysis_bundle_document(document: Mapping[str, Any] | Any) -> list[str]:
    """Validate the pure, plan-only analysis-bundle YAML contract."""

    if not isinstance(document, Mapping):
        return ["analysis_bundle document must contain a mapping."]
    errors: list[str] = []
    if "analysis_bundle" in document:
        sibling_keys = _reject_unknown_keys(document, {"analysis_bundle"}, "document", errors)
        if set(sibling_keys) & {"subject", "subjects", "session", "sessions", "task", "tasks", "run", "runs"}:
            errors.append(
                "analysis_bundle document must not contain sibling inline subject, session, task, or run selectors."
            )
        if set(sibling_keys) & _EXECUTION_KEYS:
            errors.append("analysis_bundle document must not contain sibling execution declarations.")
    payload = _analysis_bundle_payload(document, errors=errors)
    has_payload = isinstance(document.get("analysis_bundle"), Mapping) or (
        "name" in document and "selection" in document and "units" in document
    )
    if not has_payload:
        return errors

    _reject_unknown_keys(
        payload,
        {"name", "selection", "units", "components", "stages"},
        "analysis_bundle",
        errors,
    )
    name = _optional_text(payload.get("name"))
    if name is None:
        errors.append("analysis_bundle.name must be defined.")
    else:
        _validate_safe_identifier(name, "analysis_bundle.name", errors)

    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        errors.append("analysis_bundle.selection must contain a mapping.")
    else:
        unknown = _reject_unknown_keys(
            selection,
            {"cohort", "batch"},
            "analysis_bundle.selection",
            errors,
        )
        if unknown:
            errors.append(
                "analysis_bundle selection must not contain inline subject, session, task, or run selectors; "
                "store exact units in manifests/batches and select them through a batch or cohort."
            )
        cohort = _optional_text(selection.get("cohort"))
        batch = _optional_text(selection.get("batch"))
        if (cohort is None) == (batch is None):
            errors.append("analysis_bundle.selection must define exactly one of cohort or batch.")
        for key, value in (("cohort", cohort), ("batch", batch)):
            if value is not None:
                _validate_safe_identifier(value, f"analysis_bundle.selection.{key}", errors)

    _validate_analysis_bundle_units(payload.get("units"), errors)
    _validate_analysis_bundle_components(payload.get("components"), errors)
    _validate_analysis_bundle_stages(payload.get("stages"), errors)
    _validate_no_execution_fields(payload, "analysis_bundle", errors)
    _validate_no_personal_paths(payload, "analysis_bundle", errors)
    return errors


def parse_analysis_bundle_document(document: Mapping[str, Any]) -> AnalysisBundleConfig:
    """Parse a schema-valid bundle document into frozen configuration types."""

    errors = validate_analysis_bundle_document(document)
    if errors:
        raise ValueError("; ".join(errors))
    payload = _analysis_bundle_payload(document)
    selection_payload = payload["selection"]
    units_payload = payload["units"]
    components_payload = payload["components"]
    return AnalysisBundleConfig(
        name=str(payload["name"]).strip(),
        selection=AnalysisBundleSelection(
            cohort=_optional_text(selection_payload.get("cohort")),
            batch=_optional_text(selection_payload.get("batch")),
        ),
        units=AnalysisBundleUnits(
            key_columns=tuple(_string_sequence(units_payload.get("key_columns"))),
            subject_column=_optional_text(units_payload.get("subject_column")) or "subject_id",
            occasion_column=_optional_text(units_payload.get("occasion_column")),
            occasion_order_column=_optional_text(units_payload.get("occasion_order_column")),
            required_occasions=tuple(_string_sequence(units_payload.get("required_occasions"))),
            incomplete=_optional_text(units_payload.get("incomplete")) or "allow",
        ),
        components={str(key): str(value).strip() for key, value in components_payload.items()},
        stages=tuple(_string_sequence(payload.get("stages"))),
        fields=dict(payload),
    )


def resolve_analysis_bundle(
    document: Mapping[str, Any] | Any,
    *,
    cohorts_document: Mapping[str, Any] | Any,
    batch_tables: Mapping[str, ManifestTable],
    available_components: Mapping[str, Collection[str]] | None = None,
    expected_name: str | None = None,
) -> AnalysisBundleResolution:
    """Resolve one bundle against canonical batches without creating units.

    This is the authoritative contextual validator and planner. Callers provide
    literal parsed configuration and raw manifest tables; the resolver performs
    no filesystem discovery, environment expansion, domain validation, or IO.
    """

    schema_errors = validate_analysis_bundle_document(document)
    original_payload = _analysis_bundle_payload(document) if isinstance(document, Mapping) else {}
    effective_config: dict[str, Any] = {"analysis_bundle": _json_safe(original_payload), "cohort": None}
    effective_config_sha256 = _canonical_sha256(effective_config)
    if schema_errors:
        return _finalize_analysis_bundle_resolution(
            AnalysisBundleResolution(
                bundle_name=_candidate_bundle_name(document),
                errors=tuple(schema_errors),
                checks=_bundle_checks(
                    {"bundle_schema": schema_errors},
                    not_evaluated=set(_ANALYSIS_BUNDLE_CHECK_IDS) - {"bundle_schema"},
                ),
                effective_config_sha256=effective_config_sha256,
            )
        )

    config = parse_analysis_bundle_document(document)
    name_errors: list[str] = []
    if expected_name is not None and config.name != expected_name:
        name_errors.append(
            f"analysis_bundle.name {config.name!r} must match its configuration filename {expected_name!r}."
        )

    selection_errors: list[str] = []
    cohort_errors: list[str] = []
    cohort: AnalysisCohortView | None = None
    batch_name = config.selection.batch
    if config.selection.cohort is not None:
        cohort, cohort_errors = _parse_analysis_cohort_view(
            cohorts_document,
            config.selection.cohort,
        )
        if cohort is not None:
            batch_name = cohort.batch
            effective_config["cohort"] = _effective_cohort_payload(cohort)
        else:
            selection_errors.extend(cohort_errors)
    effective_config_sha256 = _canonical_sha256(effective_config)

    table = batch_tables.get(batch_name or "")
    if batch_name is None:
        selection_errors.append("Analysis bundle selection did not resolve a batch name.")
    elif table is None:
        selection_errors.append(f"Selected batch {batch_name!r} was not found under manifests/batches.")

    stage_errors = _analysis_bundle_stage_component_errors(config)
    component_errors = _analysis_bundle_component_existence_errors(
        config,
        available_components=available_components,
    )
    errors_by_check: dict[str, list[str]] = {
        "bundle_schema": name_errors,
        "selection_exists": selection_errors,
        "component_configs": component_errors,
        "stage_component_consistency": stage_errors,
    }
    if table is None:
        all_errors = [*name_errors, *selection_errors, *component_errors, *stage_errors]
        return _finalize_analysis_bundle_resolution(
            AnalysisBundleResolution(
                bundle_name=config.name,
                errors=tuple(all_errors),
                checks=_bundle_checks(
                    errors_by_check,
                    not_evaluated={
                        "required_and_filter_columns",
                        "duplicate_unit_keys",
                        "longitudinal_completeness",
                    },
                ),
                selection=_analysis_bundle_selection_payload(config, cohort=cohort, batch_name=batch_name),
                components=dict(config.components),
                stages=config.stages,
                effective_config_sha256=effective_config_sha256,
            )
        )

    return _resolve_analysis_bundle_table(
        config,
        cohort=cohort,
        table=table,
        batch_name=str(batch_name),
        effective_config_sha256=effective_config_sha256,
        errors_by_check=errors_by_check,
    )


def _analysis_bundle_payload(
    document: Mapping[str, Any],
    *,
    errors: list[str] | None = None,
) -> Mapping[str, Any]:
    payload = document.get("analysis_bundle")
    if isinstance(payload, Mapping):
        return payload
    if "name" in document and "selection" in document and "units" in document:
        return document
    if errors is not None:
        errors.append("analysis_bundle must contain a mapping.")
    return {}


def _candidate_bundle_name(document: Any) -> str | None:
    if not isinstance(document, Mapping):
        return None
    return _optional_text(_analysis_bundle_payload(document).get("name"))


def _reject_unknown_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    label: str,
    errors: list[str],
) -> tuple[str, ...]:
    unknown = tuple(str(key) for key in payload if str(key) not in allowed)
    for key in unknown:
        errors.append(f"{label}.{key} is not supported.")
    return unknown


def _validate_analysis_bundle_units(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append("analysis_bundle.units must contain a mapping.")
        return
    _reject_unknown_keys(
        value,
        {
            "key_columns",
            "subject_column",
            "occasion_column",
            "occasion_order_column",
            "required_occasions",
            "incomplete",
        },
        "analysis_bundle.units",
        errors,
    )
    key_columns = _analysis_bundle_string_list(
        value.get("key_columns"),
        "analysis_bundle.units.key_columns",
        errors,
        required=True,
    )
    for index, column in enumerate(key_columns):
        _validate_safe_identifier(column, f"analysis_bundle.units.key_columns[{index}]", errors)
    for duplicate in _duplicates(key_columns):
        errors.append(f"analysis_bundle.units.key_columns contains duplicate column: {duplicate}.")

    subject_column = _optional_text(value.get("subject_column")) or "subject_id"
    _validate_safe_identifier(subject_column, "analysis_bundle.units.subject_column", errors)
    if subject_column != "subject_id":
        errors.append("analysis_bundle.units.subject_column must be subject_id for canonical neuro units.")
    if subject_column not in key_columns:
        errors.append("analysis_bundle.units.key_columns must include subject_id.")

    occasion_column = _optional_text(value.get("occasion_column"))
    occasion_order_column = _optional_text(value.get("occasion_order_column"))
    for key, column in (
        ("occasion_column", occasion_column),
        ("occasion_order_column", occasion_order_column),
    ):
        if column is not None:
            _validate_safe_identifier(column, f"analysis_bundle.units.{key}", errors)
    required_occasions = _analysis_bundle_string_list(
        value.get("required_occasions", []),
        "analysis_bundle.units.required_occasions",
        errors,
    )
    for duplicate in _duplicates(required_occasions):
        errors.append(f"analysis_bundle.units.required_occasions contains duplicate value: {duplicate}.")
    if required_occasions and occasion_column is None:
        errors.append("analysis_bundle.units.occasion_column is required when required_occasions are configured.")
    if occasion_order_column is not None and occasion_column is None:
        errors.append("analysis_bundle.units.occasion_column is required when occasion_order_column is configured.")
    if occasion_order_column is not None and occasion_order_column == occasion_column:
        errors.append(
            "analysis_bundle.units.occasion_order_column must be distinct from occasion_column; "
            "use explicit visit-order metadata rather than session-label ordering."
        )

    incomplete = _optional_text(value.get("incomplete")) or "allow"
    if incomplete not in {"fail", "drop", "allow"}:
        errors.append("analysis_bundle.units.incomplete must be one of: fail, drop, allow.")


def _validate_analysis_bundle_components(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append("analysis_bundle.components must contain a mapping.")
        return
    _reject_unknown_keys(
        value,
        set(_ANALYSIS_BUNDLE_COMPONENT_PATHS),
        "analysis_bundle.components",
        errors,
    )
    for name, reference in value.items():
        text = _optional_text(reference)
        if text is None:
            errors.append(f"analysis_bundle.components.{name} must be a non-empty configuration name.")
        else:
            _validate_safe_identifier(text, f"analysis_bundle.components.{name}", errors)


def _validate_analysis_bundle_stages(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("analysis_bundle.stages must contain an ordered list.")
        return
    stages = _analysis_bundle_string_list(
        value,
        "analysis_bundle.stages",
        errors,
    )
    for index, stage in enumerate(stages):
        if stage not in _ANALYSIS_BUNDLE_STAGE_COMPONENTS:
            supported = ", ".join(_ANALYSIS_BUNDLE_STAGE_COMPONENTS)
            errors.append(f"analysis_bundle.stages[{index}] must be one of: {supported}.")
    for duplicate in _duplicates(stages):
        errors.append(f"analysis_bundle.stages contains duplicate stage: {duplicate}.")


def _parse_analysis_cohort_view(
    document: Mapping[str, Any] | Any,
    name: str,
) -> tuple[AnalysisCohortView | None, list[str]]:
    errors: list[str] = []
    if not isinstance(document, Mapping):
        return None, ["config/cohorts.yaml must contain a mapping."]
    cohorts = document.get("cohorts")
    if not isinstance(cohorts, Mapping):
        return None, ["config/cohorts.yaml must define a cohorts mapping."]
    payload = cohorts.get(name)
    if not isinstance(payload, Mapping):
        return None, [f"Selected cohort {name!r} was not found in config/cohorts.yaml."]

    _reject_unknown_keys(payload, {"batch", "include", "exclude"}, f"cohorts.{name}", errors)
    batch = _optional_text(payload.get("batch"))
    if batch is None:
        errors.append(f"cohorts.{name}.batch must be defined.")
    else:
        _validate_safe_identifier(batch, f"cohorts.{name}.batch", errors)

    include = _parse_filter_mapping(payload.get("include", {}), f"cohorts.{name}.include", errors)
    exclusions: list[AnalysisCohortExclusion] = []
    raw_exclusions = payload.get("exclude", [])
    if raw_exclusions is None:
        raw_exclusions = []
    if not isinstance(raw_exclusions, list):
        errors.append(f"cohorts.{name}.exclude must contain a list of exclusion rules.")
    else:
        exclusion_ids: list[str] = []
        for index, raw_rule in enumerate(raw_exclusions):
            label = f"cohorts.{name}.exclude[{index}]"
            if not isinstance(raw_rule, Mapping):
                errors.append(f"{label} must contain a mapping.")
                continue
            _reject_unknown_keys(raw_rule, {"id", "filters", "reason", "reason_field"}, label, errors)
            rule_id = _optional_text(raw_rule.get("id"))
            if rule_id is None:
                errors.append(f"{label}.id must be defined.")
                continue
            _validate_safe_identifier(rule_id, f"{label}.id", errors)
            exclusion_ids.append(rule_id)
            filters = _parse_filter_mapping(raw_rule.get("filters"), f"{label}.filters", errors)
            if not filters:
                errors.append(f"{label}.filters must define at least one active filter.")
            reason = _optional_text(raw_rule.get("reason"))
            reason_field = _optional_text(raw_rule.get("reason_field"))
            if reason is not None and reason_field is not None:
                errors.append(f"{label} must define at most one of reason or reason_field.")
            if reason_field is not None:
                _validate_safe_identifier(reason_field, f"{label}.reason_field", errors)
            exclusions.append(
                AnalysisCohortExclusion(
                    id=rule_id,
                    filters=filters,
                    reason=reason,
                    reason_field=reason_field,
                )
            )
        for duplicate in _duplicates(exclusion_ids):
            errors.append(f"cohorts.{name}.exclude contains duplicate rule id: {duplicate}.")

    if errors or batch is None:
        return None, errors
    return (
        AnalysisCohortView(
            name=name,
            batch=batch,
            include=include,
            exclude=tuple(exclusions),
            fields=dict(payload),
        ),
        [],
    )


def _parse_filter_mapping(value: Any, label: str, errors: list[str]) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must contain a mapping of columns to accepted values.")
        return {}
    filters: dict[str, tuple[str, ...]] = {}
    for raw_column, raw_values in value.items():
        column = str(raw_column).strip()
        _validate_safe_identifier(column, f"{label}.{column}", errors)
        _validate_filter_alternatives(raw_values, f"{label}.{column}", errors)
        normalized = normalized_manifest_filters({column: raw_values}).get(column, ())
        if not normalized:
            errors.append(f"{label}.{column} must define at least one non-empty value.")
            continue
        filters[column] = normalized
    return filters


def _analysis_bundle_string_list(
    value: Any,
    label: str,
    errors: list[str],
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        errors.append(f"{label} must contain a list of strings.")
        return ()
    if required and not value:
        errors.append(f"{label} must define at least one value.")
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"{label}[{index}] must be a string.")
            continue
        text = item.strip()
        if not text:
            errors.append(f"{label}[{index}] must be a non-empty string.")
            continue
        normalized.append(text)
    return tuple(normalized)


def _validate_filter_alternatives(value: Any, label: str, errors: list[str]) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            if not _is_filter_scalar(item):
                errors.append(
                    f"{label}[{index}] must be a scalar string, boolean, or number; "
                    "nested collections are not supported."
                )
            elif isinstance(item, str) and not item.strip():
                errors.append(f"{label}[{index}] must not be blank.")
        return
    if not _is_filter_scalar(value):
        errors.append(
            f"{label} must be a scalar string, boolean, or number, or a flat list of those values."
        )


def _is_filter_scalar(value: Any) -> bool:
    return isinstance(value, (str, bool, Number))


def _resolve_analysis_bundle_table(
    config: AnalysisBundleConfig,
    *,
    cohort: AnalysisCohortView | None,
    table: ManifestTable,
    batch_name: str,
    effective_config_sha256: str,
    errors_by_check: dict[str, list[str]],
) -> AnalysisBundleResolution:
    column_errors: list[str] = []
    duplicate_errors: list[str] = []
    longitudinal_errors: list[str] = []
    selection_warnings: list[str] = []

    if not table.rows:
        column_errors.append(f"Selected batch {batch_name!r} must contain at least one analysis-unit row.")
    if "subject_id" not in table.columns:
        column_errors.append(f"Selected batch {batch_name!r} must define the required subject_id column.")

    required_columns = list(config.units.key_columns)
    for column in (config.units.occasion_column, config.units.occasion_order_column):
        if column is not None and column not in required_columns:
            required_columns.append(column)
    for column in required_columns:
        if column not in table.columns:
            column_errors.append(f"Selected batch {batch_name!r} is missing configured column {column!r}.")

    for row_index, row in enumerate(table.rows, start=1):
        if not str(row.get("subject_id", "")).strip():
            column_errors.append(f"Selected batch {batch_name!r} row {row_index} is missing required subject_id.")
        for column in required_columns:
            if column in table.columns and not str(row.get(column, "")).strip():
                column_errors.append(
                    f"Selected batch {batch_name!r} row {row_index} is missing required value for {column!r}."
                )

    include_filters = cohort.include if cohort is not None else {}
    exclusion_rules = cohort.exclude if cohort is not None else ()
    for column in unknown_manifest_filter_columns(table.columns, include_filters):
        column_errors.append(f"Cohort include filter references unknown batch column {column!r}.")
    for rule in exclusion_rules:
        for column in unknown_manifest_filter_columns(table.columns, rule.filters):
            column_errors.append(f"Cohort exclusion {rule.id!r} references unknown batch column {column!r}.")
        if rule.reason_field is not None and rule.reason_field not in table.columns:
            column_errors.append(
                f"Cohort exclusion {rule.id!r} reason_field references unknown batch column {rule.reason_field!r}."
            )

    if all(column in table.columns for column in config.units.key_columns):
        seen_keys: dict[tuple[str, ...], int] = {}
        for row_index, row in enumerate(table.rows, start=1):
            key = tuple(
                normalize_manifest_identity(row.get(column, ""))
                for column in config.units.key_columns
            )
            previous = seen_keys.get(key)
            if previous is not None:
                rendered = ", ".join(
                    f"{column}={value!r}" for column, value in zip(config.units.key_columns, key)
                )
                duplicate_errors.append(
                    f"Duplicate analysis-unit key at batch rows {previous} and {row_index}: {rendered}."
                )
            else:
                seen_keys[key] = row_index

    candidates: list[tuple[int, dict[str, str]]] = []
    not_included: list[Mapping[str, Any]] = []
    filters_are_known = not any("unknown batch column" in error for error in column_errors)
    for row_index, row in enumerate(table.rows, start=1):
        values = dict(row)
        if include_filters and filters_are_known and not manifest_row_matches(values, include_filters):
            not_included.append(
                {
                    "source_row": row_index,
                    "values": values,
                    "reason_id": "include_filter_mismatch",
                    "reason": "Row did not satisfy every configured cohort include column.",
                }
            )
        else:
            candidates.append((row_index, values))

    excluded: list[Mapping[str, Any]] = []
    retained: list[tuple[int, dict[str, str]]] = []
    matched_rule_ids: set[str] = set()
    for row_index, row in candidates:
        matched: list[AnalysisCohortExclusion] = []
        if filters_are_known:
            matched = [rule for rule in exclusion_rules if manifest_row_matches(row, rule.filters)]
        if not matched:
            retained.append((row_index, row))
            continue
        exclusion_ids: list[str] = []
        reasons: list[str] = []
        for rule in matched:
            matched_rule_ids.add(rule.id)
            exclusion_ids.append(rule.id)
            if rule.reason is not None:
                reasons.append(rule.reason)
            elif rule.reason_field is not None:
                reason_value = str(row.get(rule.reason_field, "")).strip()
                if reason_value:
                    reasons.append(reason_value)
                else:
                    reasons.append(
                        f"Excluded by cohort rule {rule.id}; configured reason field {rule.reason_field} was empty."
                    )
            else:
                reasons.append(f"Excluded by cohort rule {rule.id}.")
        excluded.append(
            {
                "source_row": row_index,
                "values": row,
                "exclusion_ids": exclusion_ids,
                "reasons": reasons,
            }
        )

    unmatched = tuple(rule.id for rule in exclusion_rules if rule.id not in matched_rule_ids)
    for rule_id in unmatched:
        selection_warnings.append(
            f"Cohort exclusion rule {rule_id!r} matched no rows after inclusion filtering."
        )

    included, dropped, incomplete, occasion_order, longitudinal_messages = _apply_longitudinal_policy(
        retained,
        units=config.units,
    )
    longitudinal_errors.extend(longitudinal_messages)
    longitudinal_warnings = _longitudinal_policy_warnings(
        included=included,
        dropped=dropped,
        incomplete_subjects=incomplete,
        units=config.units,
    )
    warnings = [*selection_warnings, *longitudinal_warnings]
    if not included:
        errors_by_check.setdefault("selection_exists", []).append(
            "Analysis bundle selection resolved zero included units; review cohort filters, "
            "exclusion rules, and the incomplete-case policy."
        )

    errors_by_check["required_and_filter_columns"] = column_errors
    errors_by_check["duplicate_unit_keys"] = duplicate_errors
    errors_by_check["longitudinal_completeness"] = longitudinal_errors
    all_errors = [
        message
        for check_id in _ANALYSIS_BUNDLE_CHECK_IDS
        for message in errors_by_check.get(check_id, ())
    ]
    counts = _analysis_bundle_counts(
        table=table,
        included=included,
        excluded=excluded,
        not_included=not_included,
        dropped=dropped,
        units=config.units,
    )
    resolution = AnalysisBundleResolution(
        bundle_name=config.name,
        errors=tuple(all_errors),
        warnings=tuple(warnings),
        checks=_bundle_checks(
            errors_by_check,
            warnings_by_check={
                "selection_exists": selection_warnings,
                "longitudinal_completeness": longitudinal_warnings,
            },
        ),
        selection=_analysis_bundle_selection_payload(config, cohort=cohort, batch_name=batch_name),
        source_batch={
            "name": batch_name,
            "columns": list(table.columns),
            "row_count": len(table.rows),
            "sha256": table.source_sha256,
        },
        key_columns=config.units.key_columns,
        subject_column=config.units.subject_column,
        occasion_column=config.units.occasion_column,
        occasion_order_column=config.units.occasion_order_column,
        required_occasions=config.units.required_occasions,
        incomplete_policy=config.units.incomplete,
        included_units=tuple(_unit_payload(row_index, row) for row_index, row in included),
        excluded_units=tuple(excluded),
        not_included_units=tuple(not_included),
        dropped_units=tuple(dropped),
        incomplete_subjects=tuple(incomplete),
        occasion_order=tuple(occasion_order),
        unmatched_exclusion_rules=unmatched,
        counts=counts,
        components=dict(config.components),
        stages=config.stages,
        source_batch_sha256=table.source_sha256,
        effective_config_sha256=effective_config_sha256,
        executed=False,
    )
    return _finalize_analysis_bundle_resolution(resolution)


def _apply_longitudinal_policy(
    rows: Sequence[tuple[int, dict[str, str]]],
    *,
    units: AnalysisBundleUnits,
) -> tuple[
    list[tuple[int, dict[str, str]]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[str],
]:
    if units.occasion_column is None:
        return list(rows), [], [], [], []

    subject_rows: dict[str, list[tuple[int, dict[str, str]]]] = {}
    subject_display_values: dict[str, str] = {}
    for row_index, row in rows:
        raw_subject = str(row.get(units.subject_column, ""))
        subject = normalize_manifest_identity(raw_subject)
        subject_rows.setdefault(subject, []).append((row_index, row))
        subject_display_values.setdefault(subject, raw_subject)

    incomplete_subjects: list[Mapping[str, Any]] = []
    order_summary: list[Mapping[str, Any]] = []
    errors: list[str] = []
    incomplete_ids: set[str] = set()
    for subject, grouped_rows in subject_rows.items():
        present_identities = _first_seen_values(
            normalize_manifest_identity(row.get(units.occasion_column, ""))
            for _, row in grouped_rows
        )
        present_values = _first_seen_values_by_identity(
            str(row.get(units.occasion_column, "")) for _, row in grouped_rows
        )
        present_identity_set = set(present_identities)
        missing = tuple(
            value
            for value in units.required_occasions
            if normalize_manifest_identity(value) not in present_identity_set
        )
        display_subject = subject_display_values[subject]
        if missing:
            incomplete_ids.add(subject)
            incomplete_subjects.append(
                {
                    "subject_id": display_subject,
                    "present_occasions": list(present_values),
                    "missing_occasions": list(missing),
                    "policy": units.incomplete,
                }
            )
        ordered, order_errors = _ordered_subject_occasions(
            display_subject,
            grouped_rows,
            units=units,
        )
        order_summary.append({"subject_id": display_subject, "occasions": ordered})
        errors.extend(order_errors)

    if incomplete_ids and units.incomplete == "fail":
        rendered = ", ".join(
            f"{entry['subject_id']} missing {', '.join(entry['missing_occasions'])}"
            for entry in incomplete_subjects
        )
        errors.append(f"Incomplete longitudinal units violate the fail policy: {rendered}.")

    dropped: list[Mapping[str, Any]] = []
    included: list[tuple[int, dict[str, str]]] = []
    missing_by_subject = {
        normalize_manifest_identity(entry["subject_id"]): entry["missing_occasions"]
        for entry in incomplete_subjects
    }
    for row_index, row in rows:
        subject = normalize_manifest_identity(row.get(units.subject_column, ""))
        if units.incomplete == "drop" and subject in incomplete_ids:
            missing = missing_by_subject[subject]
            dropped.append(
                {
                    "source_row": row_index,
                    "values": row,
                    "reason_id": "incomplete_required_occasions",
                    "reason": f"Subject is missing required occasions: {', '.join(missing)}.",
                }
            )
        else:
            included.append((row_index, row))
    return included, dropped, incomplete_subjects, order_summary, errors


def _longitudinal_policy_warnings(
    *,
    included: Sequence[tuple[int, Mapping[str, str]]],
    dropped: Sequence[Mapping[str, Any]],
    incomplete_subjects: Sequence[Mapping[str, Any]],
    units: AnalysisBundleUnits,
) -> list[str]:
    if not incomplete_subjects or units.incomplete == "fail":
        return []
    subject_count = len(incomplete_subjects)
    if units.incomplete == "drop":
        return [
            "Longitudinal completeness policy 'drop' removed "
            f"{len(dropped)} unit(s) across {subject_count} incomplete subject(s)."
        ]

    incomplete_ids = {
        normalize_manifest_identity(entry["subject_id"])
        for entry in incomplete_subjects
    }
    retained_count = sum(
        normalize_manifest_identity(row.get(units.subject_column, "")) in incomplete_ids
        for _, row in included
    )
    return [
        "Longitudinal completeness policy 'allow' retains "
        f"{retained_count} unit(s) across {subject_count} incomplete subject(s)."
    ]


def _ordered_subject_occasions(
    subject: str,
    rows: Sequence[tuple[int, dict[str, str]]],
    *,
    units: AnalysisBundleUnits,
) -> tuple[list[dict[str, Any]], list[str]]:
    assert units.occasion_column is not None
    first_by_occasion: dict[
        str,
        tuple[int, str, str | None, tuple[int, Decimal | str] | None],
    ] = {}
    errors: list[str] = []
    for row_index, row in rows:
        raw_occasion = str(row.get(units.occasion_column, ""))
        occasion = normalize_manifest_identity(raw_occasion)
        raw_order_value = (
            str(row.get(units.occasion_order_column, ""))
            if units.occasion_order_column is not None
            else None
        )
        order_identity = (
            _explicit_occasion_order_identity(raw_order_value)
            if raw_order_value is not None
            else None
        )
        previous = first_by_occasion.get(occasion)
        if previous is None:
            first_by_occasion[occasion] = (
                row_index,
                raw_occasion,
                raw_order_value,
                order_identity,
            )
        elif units.occasion_order_column is not None and previous[3] != order_identity:
            errors.append(
                f"Subject {subject!r} occasion {raw_occasion!r} has conflicting "
                f"{units.occasion_order_column!r} values."
            )

    if units.occasion_order_column is not None:
        occasion_by_order: dict[tuple[int, Decimal | str], str] = {}
        for occasion, (_, raw_occasion, _, order_identity) in first_by_occasion.items():
            assert order_identity is not None
            prior_occasion = occasion_by_order.get(order_identity)
            if prior_occasion is not None and prior_occasion != occasion:
                errors.append(
                    f"Subject {subject!r} has distinct occasions {prior_occasion!r} and "
                    f"{raw_occasion!r} with the same {units.occasion_order_column!r} value."
                )
            else:
                occasion_by_order[order_identity] = raw_occasion

    values = [
        {
            "occasion": raw_occasion,
            "order_value": raw_order_value,
            "source_row": row_index,
            "_occasion_identity": occasion,
            "_order_identity": order_identity,
        }
        for occasion, (row_index, raw_occasion, raw_order_value, order_identity) in first_by_occasion.items()
    ]
    if units.occasion_order_column is not None:
        values.sort(
            key=lambda item: (*item["_order_identity"], item["source_row"])
        )
    elif units.required_occasions:
        configured_order = {
            normalize_manifest_identity(value): index
            for index, value in enumerate(units.required_occasions)
        }
        values.sort(
            key=lambda item: (
                configured_order.get(item["_occasion_identity"], len(configured_order)),
                item["source_row"],
            )
        )
    for item in values:
        item.pop("_occasion_identity")
        item.pop("_order_identity")
    return values, errors


def _explicit_occasion_order_identity(value: Any) -> tuple[int, Decimal | str]:
    text = normalize_manifest_identity(value)
    try:
        numeric = Decimal(text)
    except InvalidOperation:
        return (1, text)
    if not numeric.is_finite():
        return (1, text)
    return (0, numeric)


def _analysis_bundle_stage_component_errors(config: AnalysisBundleConfig) -> list[str]:
    errors: list[str] = []
    if not config.stages:
        return [
            "analysis_bundle.stages must configure at least one stage before the bundle is ready for planning."
        ]
    for stage in config.stages:
        for component in _ANALYSIS_BUNDLE_STAGE_COMPONENTS.get(stage, ()):
            if component not in config.components:
                errors.append(f"Analysis bundle stage {stage!r} requires components.{component}.")
    if "roi_build" in config.stages and "roi_extraction" in config.stages:
        if config.stages.index("roi_build") > config.stages.index("roi_extraction"):
            errors.append("Analysis bundle stage roi_build must precede roi_extraction when both are configured.")
    return errors


def _analysis_bundle_component_existence_errors(
    config: AnalysisBundleConfig,
    *,
    available_components: Mapping[str, Collection[str]] | None,
) -> list[str]:
    if available_components is None:
        return []
    errors: list[str] = []
    for component, reference in config.components.items():
        available = {str(name) for name in available_components.get(component, ())}
        if reference not in available:
            location = _ANALYSIS_BUNDLE_COMPONENT_PATHS[component]
            errors.append(
                f"Referenced {component} {reference!r} was not found under config/analysis/{location}."
            )
    return errors


def _analysis_bundle_selection_payload(
    config: AnalysisBundleConfig,
    *,
    cohort: AnalysisCohortView | None,
    batch_name: str | None,
) -> dict[str, Any]:
    if config.selection.cohort is not None:
        return {
            "kind": "cohort",
            "name": config.selection.cohort,
            "batch": batch_name,
            "include": _json_safe(cohort.include) if cohort is not None else {},
            "exclude_rule_ids": [rule.id for rule in cohort.exclude] if cohort is not None else [],
        }
    return {"kind": "batch", "name": batch_name, "batch": batch_name}


def _effective_cohort_payload(cohort: AnalysisCohortView) -> dict[str, Any]:
    return {
        "name": cohort.name,
        "batch": cohort.batch,
        "include": _json_safe(cohort.include),
        "exclude": [
            {
                "id": rule.id,
                "filters": _json_safe(rule.filters),
                "reason": rule.reason,
                "reason_field": rule.reason_field,
            }
            for rule in cohort.exclude
        ],
    }


def _analysis_bundle_counts(
    *,
    table: ManifestTable,
    included: Sequence[tuple[int, Mapping[str, str]]],
    excluded: Sequence[Mapping[str, Any]],
    not_included: Sequence[Mapping[str, Any]],
    dropped: Sequence[Mapping[str, Any]],
    units: AnalysisBundleUnits,
) -> dict[str, int]:
    included_rows = [row for _, row in included]

    def distinct_nonempty(column: str | None) -> int:
        if column is None or column not in table.columns:
            return 0
        return len(
            {
                normalize_manifest_identity(row.get(column, ""))
                for row in included_rows
                if normalize_manifest_identity(row.get(column, ""))
            }
        )

    explicitly_excluded = len(excluded)
    excluded_total = explicitly_excluded + len(not_included) + len(dropped)
    return {
        "source_units": len(table.rows),
        "unit_count": len(included),
        "included_units": len(included),
        "excluded_units": excluded_total,
        "explicitly_excluded_units": explicitly_excluded,
        "not_included_units": len(not_included),
        "dropped_units": len(dropped),
        "subjects": distinct_nonempty(units.subject_column),
        "occasions": distinct_nonempty(units.occasion_column),
        "tasks": distinct_nonempty("task_id"),
        "runs": distinct_nonempty("run_id"),
    }


def _unit_payload(source_row: int, values: Mapping[str, str]) -> dict[str, Any]:
    return {"source_row": source_row, "values": dict(values)}


def _first_seen_values(values: Sequence[str] | Any) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _first_seen_values_by_identity(values: Sequence[str] | Any) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = str(raw_value)
        identity = normalize_manifest_identity(value)
        if identity not in seen:
            seen.add(identity)
            ordered.append(value)
    return tuple(ordered)


def _bundle_checks(
    errors_by_check: Mapping[str, Sequence[str]],
    *,
    warnings_by_check: Mapping[str, Sequence[str]] | None = None,
    not_evaluated: Collection[str] = (),
) -> tuple[AnalysisBundleCheck, ...]:
    warnings_by_check = warnings_by_check or {}
    checks: list[AnalysisBundleCheck] = []
    for check_id in _ANALYSIS_BUNDLE_CHECK_IDS:
        errors = tuple(errors_by_check.get(check_id, ()))
        warnings = tuple(warnings_by_check.get(check_id, ()))
        if errors:
            checks.append(AnalysisBundleCheck(id=check_id, status="fail", messages=errors))
        elif check_id in not_evaluated:
            checks.append(
                AnalysisBundleCheck(
                    id=check_id,
                    status="not_evaluated",
                    messages=("Check was not evaluated because an earlier contract failed.",),
                )
            )
        elif warnings:
            checks.append(AnalysisBundleCheck(id=check_id, status="warning", messages=warnings))
        else:
            checks.append(AnalysisBundleCheck(id=check_id, status="pass", messages=()))
    return tuple(checks)


def _finalize_analysis_bundle_resolution(resolution: AnalysisBundleResolution) -> AnalysisBundleResolution:
    digest_payload = resolution.to_dict()
    digest_payload["digests"]["plan_sha256"] = ""
    return replace(resolution, plan_digest=_canonical_sha256(digest_payload))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_analysis_workflow_document(document: Mapping[str, Any] | Any) -> list[str]:
    """Validate a generic analysis workflow recipe document."""

    if not isinstance(document, Mapping):
        return ["analysis_workflow document must contain a mapping."]

    errors: list[str] = []
    workflow = _payload_mapping(document, errors=errors)
    if not workflow:
        return errors

    name = _optional_text(workflow.get("name"))
    if name is None:
        errors.append("analysis_workflow.name must be defined.")
    else:
        _validate_safe_identifier(name, "analysis_workflow.name", errors)

    runtime_tag = _optional_text(workflow.get("runtime_tag"))
    if runtime_tag is not None:
        _validate_safe_identifier(runtime_tag, "analysis_workflow.runtime_tag", errors)

    _validate_no_execution_fields(workflow, "analysis_workflow", errors)
    _validate_no_personal_paths(workflow, "analysis_workflow", errors)
    _validate_relative_path_strings(workflow, "analysis_workflow", errors)
    _validate_selector(workflow, errors)
    _validate_root_refs(workflow, errors)
    _validate_stages(workflow, errors)
    _validate_output_roots(workflow, errors)
    _validate_publication(workflow, errors)
    _validate_reporting(workflow, errors)
    _validate_extensions(workflow, errors)
    return errors


def parse_analysis_workflow_document(document: Mapping[str, Any]) -> AnalysisWorkflowConfig:
    """Parse a validated workflow recipe into frozen config contracts."""

    errors = validate_analysis_workflow_document(document)
    if errors:
        raise ValueError("; ".join(errors))

    workflow = _payload_mapping(document)
    selector = _parse_selector(workflow)
    return AnalysisWorkflowConfig(
        name=str(workflow["name"]).strip(),
        runtime_tag=_optional_text(workflow.get("runtime_tag")),
        selector=selector,
        stages=tuple(_parse_stage(stage) for stage in _stage_mappings(workflow)),
        root_refs=tuple(_parse_root_ref(name, root) for name, root in _root_ref_mappings(workflow).items()),
        output_roots=tuple(_parse_output_root(name, output) for name, output in _output_root_mappings(workflow)),
        publication=_parse_publication(workflow),
        reporting=_parse_reporting(workflow),
        extensions=dict(_extension_payload(workflow)),
        fields=dict(workflow),
    )


def plan_analysis_workflow_recipe(
    document: Mapping[str, Any] | Any,
    context: Mapping[str, Any] | None = None,
) -> AnalysisWorkflowPlan:
    """Return a JSON-serializable plan-only preview for a workflow recipe."""

    errors = validate_analysis_workflow_document(document)
    if errors:
        return AnalysisWorkflowPlan(
            workflow_name=_candidate_workflow_name(document),
            status="invalid",
            errors=tuple(errors),
            context=dict(context or {}),
            executed=False,
        )

    config = parse_analysis_workflow_document(document)
    return AnalysisWorkflowPlan(
        workflow_name=config.name,
        status="deferred",
        runtime_tag=config.runtime_tag,
        subjects=config.selector.subjects,
        sessions=config.selector.sessions,
        tasks=config.selector.tasks,
        runs=config.selector.runs,
        stages=tuple(
            AnalysisWorkflowStagePlanRow(
                name=stage.name,
                kind=stage.kind,
                extension=stage.extension,
            )
            for stage in config.stages
        ),
        root_refs=tuple(
            AnalysisRootRefPlanRow(
                name=root.name,
                description=root.description,
            )
            for root in config.root_refs
        ),
        output_roots=tuple(
            AnalysisOutputRootPlanRow(
                name=output.name,
                root_ref=output.root_ref,
                relative_path_template=output.path,
            )
            for output in config.output_roots
        ),
        publication={
            "enabled": config.publication.enabled,
            "derivative_name": config.publication.derivative_name,
        },
        reporting={
            "enabled": config.reporting.enabled,
            "formats": config.reporting.formats,
        },
        extensions=tuple(AnalysisExtensionPlanRow(name=name) for name in sorted(config.extensions)),
        context=dict(context or {}),
        executed=False,
    )


def _payload_mapping(document: Mapping[str, Any], *, errors: list[str] | None = None) -> Mapping[str, Any]:
    if isinstance(document.get("analysis_workflow"), Mapping):
        return document["analysis_workflow"]  # type: ignore[index]
    if _looks_like_workflow_payload(document):
        return document
    if errors is not None:
        errors.append("analysis_workflow must contain a mapping.")
    return {}


def _looks_like_workflow_payload(document: Mapping[str, Any]) -> bool:
    return "name" in document and "stages" in document


def _candidate_workflow_name(document: Any) -> str | None:
    if not isinstance(document, Mapping):
        return None
    payload = document.get("analysis_workflow") if isinstance(document.get("analysis_workflow"), Mapping) else document
    if not isinstance(payload, Mapping):
        return None
    return _optional_text(payload.get("name"))


def _validate_selector(workflow: Mapping[str, Any], errors: list[str]) -> None:
    selector = _selector_payload(workflow)
    for plural, singular in (
        ("subjects", "subject"),
        ("sessions", "session"),
        ("tasks", "task"),
        ("runs", "run"),
    ):
        values = _selector_values(selector, plural, singular)
        label = f"analysis_workflow.{plural}"
        if not values:
            errors.append(f"{label} must define at least one value.")
            continue
        for index, value in enumerate(values):
            if not value:
                errors.append(f"{label}[{index}] must be a non-empty value.")
            elif not _SAFE_SELECTOR_VALUE.fullmatch(value):
                errors.append(f"{label}[{index}] must be a safe selector label or placeholder.")


def _parse_selector(workflow: Mapping[str, Any]) -> AnalysisWorkflowSelector:
    selector = _selector_payload(workflow)
    return AnalysisWorkflowSelector(
        subjects=tuple(_selector_values(selector, "subjects", "subject")),
        sessions=tuple(_selector_values(selector, "sessions", "session")),
        tasks=tuple(_selector_values(selector, "tasks", "task")),
        runs=tuple(_selector_values(selector, "runs", "run")),
        fields=dict(selector),
    )


def _selector_payload(workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("selector", "selectors"):
        if isinstance(workflow.get(key), Mapping):
            payload.update(workflow[key])  # type: ignore[arg-type]
    if isinstance(workflow.get("cohort"), Mapping):
        cohort = workflow["cohort"]  # type: ignore[index]
        if isinstance(cohort.get("subjects"), Mapping):
            payload["subjects"] = cohort["subjects"].get("include", cohort["subjects"].get("values"))  # type: ignore[index]
        elif "subjects" in cohort:
            payload["subjects"] = cohort.get("subjects")
        for key in ("sessions", "tasks", "runs"):
            if key in cohort:
                payload[key] = cohort.get(key)
    for key in ("subjects", "sessions", "tasks", "runs"):
        if key in workflow:
            payload[key] = workflow.get(key)
    return payload


def _selector_values(selector: Mapping[str, Any], plural: str, singular: str) -> tuple[str, ...]:
    raw = selector.get(plural, selector.get(singular))
    if isinstance(raw, Mapping):
        raw = raw.get("include", raw.get("values"))
    return tuple(_string_sequence(raw))


def _validate_root_refs(workflow: Mapping[str, Any], errors: list[str]) -> None:
    root_refs = workflow.get("root_refs", workflow.get("roots"))
    if root_refs is None:
        return
    if not isinstance(root_refs, Mapping):
        errors.append("analysis_workflow.root_refs must contain a mapping when declared.")
        return
    for name, root in root_refs.items():
        label = f"analysis_workflow.root_refs.{name}"
        _validate_safe_identifier(str(name), label, errors)
        if not isinstance(root, (Mapping, str)):
            errors.append(f"{label} must be a mapping or string root declaration.")


def _root_ref_mappings(workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    root_refs = workflow.get("root_refs", workflow.get("roots", {}))
    return root_refs if isinstance(root_refs, Mapping) else {}


def _parse_root_ref(name: str, root: Any) -> AnalysisRootRef:
    fields_value = dict(root) if isinstance(root, Mapping) else {"value": root}
    description = _optional_text(root.get("description")) if isinstance(root, Mapping) else None
    return AnalysisRootRef(name=str(name), description=description, fields=fields_value)


def _validate_stages(workflow: Mapping[str, Any], errors: list[str]) -> None:
    stages = _stage_mappings(workflow, errors=errors)
    names: list[str] = []
    if not stages:
        errors.append("analysis_workflow.stages must define at least one stage.")
        return
    for index, stage in enumerate(stages):
        label = f"analysis_workflow.stages[{index}]"
        name = _optional_text(stage.get("name") or stage.get("id"))
        if name is None:
            errors.append(f"{label}.name must be defined.")
        else:
            names.append(name)
            _validate_safe_identifier(name, f"{label}.name", errors)
        kind = _optional_text(stage.get("kind") or stage.get("type"))
        if kind is None:
            errors.append(f"{label}.kind must be defined.")
        else:
            _validate_safe_identifier(kind, f"{label}.kind", errors)
        extension = _optional_text(stage.get("extension"))
        if extension is not None:
            _validate_safe_identifier(extension, f"{label}.extension", errors)
    for duplicate in _duplicates(names):
        errors.append(f"analysis_workflow.stages contains duplicate stage name: {duplicate}.")


def _stage_mappings(workflow: Mapping[str, Any], *, errors: list[str] | None = None) -> list[Mapping[str, Any]]:
    return _mapping_list(workflow.get("stages"), "analysis_workflow.stages", errors, require_non_empty=True)


def _parse_stage(stage: Mapping[str, Any]) -> AnalysisWorkflowStage:
    return AnalysisWorkflowStage(
        name=str(stage.get("name") or stage.get("id")).strip(),
        kind=str(stage.get("kind") or stage.get("type")).strip(),
        extension=_optional_text(stage.get("extension")),
        fields=dict(stage),
    )


def _validate_output_roots(workflow: Mapping[str, Any], errors: list[str]) -> None:
    output_roots = workflow.get("output_roots", workflow.get("outputs"))
    if not isinstance(output_roots, Mapping):
        errors.append("analysis_workflow.output_roots must contain at least one root-ref output declaration.")
        return
    roots = _output_root_mappings(workflow, errors=errors)
    if not roots:
        errors.append("analysis_workflow.output_roots must contain at least one root-ref output declaration.")
        return
    for name, root in roots:
        label = f"analysis_workflow.output_roots.{name}"
        _validate_safe_identifier(name, label, errors)
        root_ref = _optional_text(root.get("root_ref"))
        if root_ref is None:
            errors.append(f"{label}.root_ref must be defined.")
        else:
            _validate_safe_identifier(root_ref, f"{label}.root_ref", errors)
        path = _optional_text(root.get("path") or root.get("relative_path"))
        if path is None:
            errors.append(f"{label}.path must be defined.")
        else:
            _validate_relative_path(path, f"{label}.path", errors)


def _output_root_mappings(
    workflow: Mapping[str, Any],
    *,
    errors: list[str] | None = None,
) -> list[tuple[str, Mapping[str, Any]]]:
    output_roots = workflow.get("output_roots", workflow.get("outputs"))
    if not isinstance(output_roots, Mapping):
        return []
    if "root_ref" in output_roots or "path" in output_roots or "relative_path" in output_roots:
        return [("root", output_roots)]
    roots: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in output_roots.items():
        if isinstance(value, Mapping):
            roots.append((str(key), value))
        elif errors is not None:
            errors.append(f"analysis_workflow.output_roots.{key} must contain a mapping.")
    return roots


def _parse_output_root(name: str, output: Mapping[str, Any]) -> AnalysisOutputRoot:
    return AnalysisOutputRoot(
        name=name,
        root_ref=str(output["root_ref"]).strip(),
        path=str(output.get("path") or output.get("relative_path")).strip(),
        fields=dict(output),
    )


def _validate_publication(workflow: Mapping[str, Any], errors: list[str]) -> None:
    publication = workflow.get("publication")
    if publication is None:
        return
    if not isinstance(publication, Mapping):
        errors.append("analysis_workflow.publication must contain a mapping when declared.")
        return
    derivative_name = _optional_text(publication.get("derivative_name"))
    if derivative_name is not None:
        _validate_safe_identifier(derivative_name, "analysis_workflow.publication.derivative_name", errors)


def _parse_publication(workflow: Mapping[str, Any]) -> AnalysisPublicationSettings:
    publication = workflow.get("publication")
    if not isinstance(publication, Mapping):
        return AnalysisPublicationSettings()
    return AnalysisPublicationSettings(
        enabled=bool(publication.get("enabled", False)),
        derivative_name=_optional_text(publication.get("derivative_name")),
        fields=dict(publication),
    )


def _validate_reporting(workflow: Mapping[str, Any], errors: list[str]) -> None:
    reporting = workflow.get("reporting")
    if reporting is None:
        return
    if not isinstance(reporting, Mapping):
        errors.append("analysis_workflow.reporting must contain a mapping when declared.")
        return
    for index, format_name in enumerate(_string_sequence(reporting.get("formats"))):
        _validate_safe_identifier(format_name, f"analysis_workflow.reporting.formats[{index}]", errors)


def _parse_reporting(workflow: Mapping[str, Any]) -> AnalysisReportingSettings:
    reporting = workflow.get("reporting")
    if not isinstance(reporting, Mapping):
        return AnalysisReportingSettings()
    return AnalysisReportingSettings(
        enabled=bool(reporting.get("enabled", False)),
        formats=tuple(_string_sequence(reporting.get("formats"))),
        fields=dict(reporting),
    )


def _validate_extensions(workflow: Mapping[str, Any], errors: list[str]) -> None:
    extensions = workflow.get("extensions")
    if extensions is None:
        return
    if not isinstance(extensions, Mapping):
        errors.append("analysis_workflow.extensions must contain a mapping when declared.")
        return
    for name in extensions:
        _validate_safe_identifier(str(name), f"analysis_workflow.extensions.{name}", errors)


def _extension_payload(workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    extensions = workflow.get("extensions", {})
    return extensions if isinstance(extensions, Mapping) else {}


def _validate_no_execution_fields(value: Any, label: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if str(key).lower() in _EXECUTION_KEYS:
                errors.append(f"{child_label} is not allowed in a schema/plan-only workflow recipe.")
            _validate_no_execution_fields(child, child_label, errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_execution_fields(child, f"{label}[{index}]", errors)


def _validate_no_personal_paths(value: Any, label: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_no_personal_paths(child, f"{label}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_personal_paths(child, f"{label}[{index}]", errors)
    elif isinstance(value, str):
        for pattern in _ABSOLUTE_PATH_PATTERNS:
            if pattern.search(value):
                errors.append(f"{label} must not contain a personal absolute path.")
                break


def _validate_relative_path_strings(value: Any, label: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if str(key).lower() in _PATH_LIKE_KEYS:
                for index, path_value in enumerate(_path_values(child)):
                    path_label = child_label if index == 0 else f"{child_label}[{index}]"
                    _validate_relative_path(path_value, path_label, errors)
            else:
                _validate_relative_path_strings(child, child_label, errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_relative_path_strings(child, f"{label}[{index}]", errors)


def _validate_relative_path(value: str, label: str, errors: list[str]) -> None:
    text = str(value).strip()
    if not text:
        errors.append(f"{label} must be a non-empty relative path.")
        return
    if text.startswith("/") or _WINDOWS_ABSOLUTE_PATH.match(text):
        errors.append(f"{label} must be a relative path or root-ref template, not an absolute path.")
        return
    if any(part == ".." for part in text.replace("\\", "/").split("/")):
        errors.append(f"{label} must not contain parent-directory traversal.")


def _path_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _validate_safe_identifier(value: str, label: str, errors: list[str]) -> None:
    if not _SAFE_IDENTIFIER.fullmatch(value):
        errors.append(f"{label} must be a safe identifier.")


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence):
        values: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                values.append(text)
        return tuple(values)
    text = str(value).strip()
    return (text,) if text else ()


def _mapping_list(
    value: Any,
    label: str,
    errors: list[str] | None,
    *,
    require_non_empty: bool,
) -> list[Mapping[str, Any]]:
    if value is None:
        if require_non_empty and errors is not None:
            errors.append(f"{label} must contain a list of mappings.")
        return []
    if isinstance(value, Mapping):
        return [value]
    if not isinstance(value, list):
        if errors is not None:
            errors.append(f"{label} must contain a list of mappings.")
        return []
    rows: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            rows.append(item)
        elif errors is not None:
            errors.append(f"{label}[{index}] must contain a mapping.")
    if require_non_empty and not rows and errors is not None:
        errors.append(f"{label} must contain at least one mapping.")
    return rows


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_safe_dataclass(value: Any) -> dict[str, Any]:
    return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child) for child in value]
    return str(value)


__all__ = [
    "AnalysisBundleCheck",
    "AnalysisBundleConfig",
    "AnalysisBundleResolution",
    "AnalysisBundleSelection",
    "AnalysisBundleUnits",
    "AnalysisCohortExclusion",
    "AnalysisCohortView",
    "AnalysisExtensionPlanRow",
    "AnalysisOutputRoot",
    "AnalysisOutputRootPlanRow",
    "AnalysisPublicationSettings",
    "AnalysisReportingSettings",
    "AnalysisRootRef",
    "AnalysisRootRefPlanRow",
    "AnalysisWorkflowConfig",
    "AnalysisWorkflowPlan",
    "AnalysisWorkflowSelector",
    "AnalysisWorkflowStage",
    "AnalysisWorkflowStagePlanRow",
    "parse_analysis_workflow_document",
    "parse_analysis_bundle_document",
    "plan_analysis_workflow_recipe",
    "resolve_analysis_bundle",
    "validate_analysis_bundle_document",
    "validate_analysis_workflow_document",
]
