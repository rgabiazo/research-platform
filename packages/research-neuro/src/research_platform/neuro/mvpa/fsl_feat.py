"""Plan-only FSL FEAT PE discovery for MVPA pattern sources.

This module previews configured ``fsl_feat_pe`` inputs and maps FEAT EV titles
to PE image paths. It does not run FSL, load NIfTI files, resolve ROI masks,
extract voxels, compute MVPA distances, or write outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Any
import re

from research_platform.neuro.fsl.feat_design import (
    map_conditions_to_pe_numbers,
    parse_fsl_design_ev_titles,
    parse_fsl_design_file,
)

from .config import (
    MISSING_INPUT_POLICY_FAIL,
    MISSING_INPUT_POLICY_SKIP,
    MISSING_INPUT_POLICY_WARN,
    NOISE_NORMALIZATION_DIAGONAL,
    PATTERN_BACKEND_FSL_FEAT_PE,
    ConditionDefinition,
    ExclusionRule,
    MissingInputPolicy,
    MvpaSetConfig,
    PatternSourceConfig,
    parse_mvpa_set_document,
    validate_mvpa_set_document,
)


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_UNRESOLVED_PLACEHOLDER = re.compile(r"\{[^{}]+\}")
_GLOB_CHARS = frozenset("*?[")
_POLICY_KEYS = frozenset(
    {
        "default",
        "root",
        "feat_dir",
        "design_fsf",
        "design_parse",
        "condition_ev_title",
        "ambiguous_ev_title",
        "pe_image",
        "noise_image",
        "event_file",
        "event_file_parse",
    }
)
_SUPPORTED_POLICIES = frozenset(
    {
        MISSING_INPUT_POLICY_WARN,
        MISSING_INPUT_POLICY_SKIP,
        MISSING_INPUT_POLICY_FAIL,
    }
)


@dataclass(frozen=True)
class FslFeatInputCheckRow:
    """One filesystem or metadata check performed in plan mode."""

    input_kind: str
    path: str | None
    exists: bool | None
    status: str
    policy: str | None = None
    message: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    condition_id: str | None = None
    event_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class FslFeatConditionPePlanRow:
    """One configured condition mapped to an expected FEAT PE image."""

    subject_id: str
    session_id: str
    run_id: str
    condition_id: str
    requested_ev_title: str | None = None
    matched_ev_title: str | None = None
    matched_alias: str | None = None
    ev_index: int | None = None
    pe_number: int | None = None
    pe_image: str | None = None
    noise_image: str | None = None
    event_file: str | None = None
    event_count: int | None = None
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class FslFeatPatternSourcePlanRow:
    """One subject/session/run FEAT pattern-source discovery unit."""

    subject_id: str
    session_id: str
    run_id: str
    task_id: str | None
    direction: str | None
    model: str | None
    pattern_source_name: str
    backend: str
    feat_dir: str | None
    design_fsf: str | None
    noise_image: str | None
    status: str = "ok"
    excluded: bool = False
    exclusion_reason: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class FslFeatPatternSourcePlan:
    """JSON-safe plan-only FEAT PE discovery result."""

    mvpa_set_name: str | None
    pattern_source_name: str | None
    backend: str = PATTERN_BACKEND_FSL_FEAT_PE
    status: str = "ok"
    units: tuple[FslFeatPatternSourcePlanRow, ...] = ()
    condition_pe_rows: tuple[FslFeatConditionPePlanRow, ...] = ()
    input_checks: tuple[FslFeatInputCheckRow, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    executed: bool = False

    @property
    def valid(self) -> bool:
        """Return whether config parsing succeeded for this plan."""

        return self.status != "invalid"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        return payload


def plan_fsl_feat_pattern_source(
    document_or_config: Mapping[str, Any] | MvpaSetConfig,
    *,
    pattern_source_name: str | None = None,
    roots: Mapping[str, str | Path] | None = None,
    context: Mapping[str, Any] | None = None,
    unit_contexts: Sequence[Mapping[str, Any]] | None = None,
    raise_on_fail_policy: bool = False,
) -> FslFeatPatternSourcePlan:
    """Plan FEAT PE inputs for a configured ``fsl_feat_pe`` pattern source.

    The returned plan is a preview/QC object only. Concrete filesystem checks
    are limited to paths that have a supplied root, no unresolved placeholders,
    and no glob-like characters.
    """

    config_result = _coerce_config(document_or_config, context=context)
    if isinstance(config_result, FslFeatPatternSourcePlan):
        return config_result
    config = config_result

    source_result = _select_pattern_source(config, pattern_source_name)
    if isinstance(source_result, FslFeatPatternSourcePlan):
        return source_result
    source = source_result

    policy_resolver = _PolicyResolver(source=source, top_level=config.missing_input_policy)
    root_state = _resolve_root(source, roots=roots)
    template_settings = _template_settings(source)
    plan_context = _plan_context(config=config, source=source, root_state=root_state, context=context)

    unit_rows: list[FslFeatPatternSourcePlanRow] = []
    condition_rows: list[FslFeatConditionPePlanRow] = []
    input_checks: list[FslFeatInputCheckRow] = []
    warnings: list[str] = []
    errors: list[str] = []

    planned_unit_contexts = (
        tuple(unit_contexts)
        if unit_contexts is not None
        else _unit_contexts(config, context=context)
    )
    for unit_context in planned_unit_contexts:
        unit_plan = _plan_unit(
            config=config,
            source=source,
            root_state=root_state,
            settings=template_settings,
            policy_resolver=policy_resolver,
            unit_context=unit_context,
            roots=roots,
        )
        unit_rows.append(unit_plan.unit)
        condition_rows.extend(unit_plan.conditions)
        input_checks.extend(unit_plan.input_checks)
        warnings.extend(unit_plan.warnings)
        errors.extend(unit_plan.errors)

    warnings = _unique_text(warnings)
    errors = _unique_text(errors)
    status = _aggregate_plan_status(unit_rows, condition_rows, input_checks, errors=errors, warnings=warnings)
    if raise_on_fail_policy and errors:
        raise ValueError("; ".join(errors))

    return FslFeatPatternSourcePlan(
        mvpa_set_name=config.name,
        pattern_source_name=source.name,
        backend=source.backend,
        status=status,
        units=tuple(unit_rows),
        condition_pe_rows=tuple(condition_rows),
        input_checks=tuple(input_checks),
        warnings=tuple(warnings),
        errors=tuple(errors),
        context=plan_context,
        executed=False,
    )


@dataclass(frozen=True)
class _RootState:
    root_ref: str | None
    root: Path | None
    path_preview: str | None
    status: str
    message: str | None = None
    checkable: bool = False
    exists: bool | None = None
    missing_policy_applies: bool = False


@dataclass(frozen=True)
class _EventFileSettings:
    path_template: str | None = None
    root_ref: str | None = None
    root: str | None = None


@dataclass(frozen=True)
class _TemplateSettings:
    feat_dir_template: str
    design_file: str
    pe_image_template: str
    noise_image_template: str
    case_sensitive: bool
    event_files: _EventFileSettings = field(default_factory=_EventFileSettings)


@dataclass(frozen=True)
class _UnitPlan:
    unit: FslFeatPatternSourcePlanRow
    conditions: tuple[FslFeatConditionPePlanRow, ...]
    input_checks: tuple[FslFeatInputCheckRow, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


class _PolicyResolver:
    def __init__(self, *, source: PatternSourceConfig, top_level: MissingInputPolicy) -> None:
        self._source_missing = _policy_mapping(source.fields.get("missing"))
        self._top_level_missing = _policy_mapping(top_level.fields)
        self._top_level_default = _normalize_policy(top_level.policy) or MISSING_INPUT_POLICY_WARN

    def for_kind(self, kind: str) -> str:
        source_value = self._source_missing.get(kind) or self._source_missing.get("default")
        if source_value is not None:
            return source_value
        top_level_value = self._top_level_missing.get(kind) or self._top_level_missing.get("default")
        if top_level_value is not None:
            return top_level_value
        return self._top_level_default


def _coerce_config(
    document_or_config: Mapping[str, Any] | MvpaSetConfig,
    *,
    context: Mapping[str, Any] | None,
) -> MvpaSetConfig | FslFeatPatternSourcePlan:
    if isinstance(document_or_config, MvpaSetConfig):
        return document_or_config

    errors = validate_mvpa_set_document(document_or_config)
    if errors:
        return FslFeatPatternSourcePlan(
            mvpa_set_name=_candidate_mvpa_set_name(document_or_config),
            pattern_source_name=None,
            status="invalid",
            errors=tuple(errors),
            context=dict(context or {}),
            executed=False,
        )
    return parse_mvpa_set_document(document_or_config)


def _select_pattern_source(
    config: MvpaSetConfig,
    pattern_source_name: str | None,
) -> PatternSourceConfig | FslFeatPatternSourcePlan:
    if pattern_source_name is not None:
        for source in config.pattern_sources:
            if source.name == pattern_source_name:
                if source.backend != PATTERN_BACKEND_FSL_FEAT_PE:
                    return FslFeatPatternSourcePlan(
                        mvpa_set_name=config.name,
                        pattern_source_name=source.name,
                        backend=source.backend,
                        status="invalid",
                        errors=(f"Pattern source {source.name!r} is not backend {PATTERN_BACKEND_FSL_FEAT_PE}.",),
                    )
                return source
        return FslFeatPatternSourcePlan(
            mvpa_set_name=config.name,
            pattern_source_name=pattern_source_name,
            status="invalid",
            errors=(f"Pattern source {pattern_source_name!r} was not found.",),
        )

    candidates = tuple(source for source in config.pattern_sources if source.backend == PATTERN_BACKEND_FSL_FEAT_PE)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return FslFeatPatternSourcePlan(
            mvpa_set_name=config.name,
            pattern_source_name=None,
            status="invalid",
            errors=(f"No {PATTERN_BACKEND_FSL_FEAT_PE} pattern source is configured.",),
        )
    return FslFeatPatternSourcePlan(
        mvpa_set_name=config.name,
        pattern_source_name=None,
        status="invalid",
        errors=("Multiple fsl_feat_pe pattern sources are configured; pass pattern_source_name.",),
    )


def _template_settings(source: PatternSourceConfig) -> _TemplateSettings:
    fields = source.fields
    return _TemplateSettings(
        feat_dir_template=_optional_text(fields.get("feat_dir_template"))
        or _optional_text(fields.get("feat_dir"))
        or _optional_text(source.path)
        or _optional_text(source.pattern)
        or ".",
        design_file=_optional_text(fields.get("design_file")) or "design.fsf",
        pe_image_template=_optional_text(fields.get("pe_image_template")) or "stats/pe{pe_number}.nii.gz",
        noise_image_template=_optional_text(fields.get("noise_image_template")) or "stats/sigmasquareds.nii.gz",
        case_sensitive=bool(fields.get("case_sensitive", True)),
        event_files=_event_file_settings(source),
    )


def _event_file_settings(source: PatternSourceConfig) -> _EventFileSettings:
    fields = source.fields
    events = fields.get("events") or fields.get("event_files") or fields.get("event_source")
    event_fields = dict(events) if isinstance(events, Mapping) else {}
    return _EventFileSettings(
        path_template=_optional_text(
            event_fields.get("path")
            or event_fields.get("path_template")
            or event_fields.get("template")
            or fields.get("event_file_template")
        ),
        root_ref=_optional_text(event_fields.get("root_ref") or fields.get("events_root_ref") or fields.get("event_root_ref")),
        root=_optional_text(event_fields.get("root") or fields.get("events_root") or fields.get("event_root")),
    )


def _resolve_root(source: PatternSourceConfig, *, roots: Mapping[str, str | Path] | None) -> _RootState:
    root_ref = source.root_ref or _optional_text(source.fields.get("root_ref"))
    literal_root = _optional_text(source.fields.get("root"))
    if root_ref and roots is not None and root_ref in roots:
        root = Path(roots[root_ref])
        preview = root.as_posix()
        checkable = _path_is_checkable(preview)
        exists = root.exists() if checkable else None
        return _RootState(
            root_ref=root_ref,
            root=root,
            path_preview=preview,
            status="ok" if exists else "missing",
            checkable=checkable,
            exists=exists,
            missing_policy_applies=exists is False,
        )

    if literal_root is not None:
        root = Path(literal_root)
        preview = root.as_posix()
        checkable = _path_is_checkable(preview)
        exists = root.exists() if checkable else None
        return _RootState(
            root_ref=root_ref,
            root=root,
            path_preview=preview,
            status="ok" if exists else "missing",
            checkable=checkable,
            exists=exists,
            missing_policy_applies=exists is False,
        )

    if root_ref and roots is None:
        return _RootState(
            root_ref=root_ref,
            root=None,
            path_preview=root_ref,
            status="preview_only",
            message=f"No concrete roots mapping supplied for root_ref {root_ref!r}.",
        )

    message = f"No concrete root was supplied for root_ref {root_ref!r}." if root_ref else "No root_ref or root was supplied."
    return _RootState(
        root_ref=root_ref,
        root=None,
        path_preview=root_ref,
        status="missing",
        message=message,
        missing_policy_applies=True,
    )


def _plan_unit(
    *,
    config: MvpaSetConfig,
    source: PatternSourceConfig,
    root_state: _RootState,
    settings: _TemplateSettings,
    policy_resolver: _PolicyResolver,
    unit_context: Mapping[str, Any],
    roots: Mapping[str, str | Path] | None,
) -> _UnitPlan:
    input_checks: list[FslFeatInputCheckRow] = []
    condition_rows: list[FslFeatConditionPePlanRow] = []
    warnings: list[str] = []
    errors: list[str] = []

    subject_id = str(unit_context["subject_id"])
    session_id = str(unit_context["session_id"])
    run_id = str(unit_context["run_id"])
    check_kwargs = {
        "subject_id": subject_id,
        "session_id": session_id,
        "run_id": run_id,
    }

    root_ok = _record_root_check(
        root_state,
        policy_resolver=policy_resolver,
        input_checks=input_checks,
        warnings=warnings,
        errors=errors,
        **check_kwargs,
    )
    feat_dir = _render_path(root_state.root, settings.feat_dir_template, unit_context)
    design_fsf = _render_child_path(feat_dir, settings.design_file, unit_context)
    noise_image = _render_child_path(feat_dir, settings.noise_image_template, unit_context)
    excluded, exclusion_reason = _exclusion_for_unit(config.exclusions, unit_context)

    if not root_ok:
        _add_dependent_not_checked(input_checks, "feat_dir", feat_dir, "root is unavailable", **check_kwargs)
        _add_dependent_not_checked(input_checks, "design_fsf", design_fsf, "root is unavailable", **check_kwargs)
        if _noise_required(config):
            _add_dependent_not_checked(input_checks, "noise_image", noise_image, "root is unavailable", **check_kwargs)
        condition_rows.extend(
            _not_checked_condition_rows(config.conditions, unit_context, noise_image, "root is unavailable")
        )
        return _finalize_unit_plan(
            config=config,
            source=source,
            unit_context=unit_context,
            feat_dir=feat_dir,
            design_fsf=design_fsf,
            noise_image=noise_image,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
            condition_rows=condition_rows,
            input_checks=input_checks,
            warnings=warnings,
            errors=errors,
        )

    feat_check = _check_path(
        input_kind="feat_dir",
        path=feat_dir,
        policy_key="feat_dir",
        missing_message=f"FEAT directory is missing: {feat_dir}",
        policy_resolver=policy_resolver,
        may_check=root_state.root is not None and root_state.exists is not False,
        input_checks=input_checks,
        warnings=warnings,
        errors=errors,
        **check_kwargs,
    )
    if feat_check.exists is not True:
        reason = _dependency_reason(feat_check)
        _add_dependent_not_checked(input_checks, "design_fsf", design_fsf, reason, **check_kwargs)
        if _noise_required(config):
            _add_dependent_not_checked(input_checks, "noise_image", noise_image, reason, **check_kwargs)
        condition_rows.extend(_not_checked_condition_rows(config.conditions, unit_context, noise_image, reason))
        return _finalize_unit_plan(
            config=config,
            source=source,
            unit_context=unit_context,
            feat_dir=feat_dir,
            design_fsf=design_fsf,
            noise_image=noise_image,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
            condition_rows=condition_rows,
            input_checks=input_checks,
            warnings=warnings,
            errors=errors,
        )

    design_check = _check_path(
        input_kind="design_fsf",
        path=design_fsf,
        policy_key="design_fsf",
        missing_message=f"FEAT design.fsf is missing: {design_fsf}",
        policy_resolver=policy_resolver,
        may_check=True,
        input_checks=input_checks,
        warnings=warnings,
        errors=errors,
        **check_kwargs,
    )
    if design_check.exists is not True:
        reason = _dependency_reason(design_check)
        if _noise_required(config):
            _record_noise_check(
                config=config,
                noise_image=noise_image,
                policy_resolver=policy_resolver,
                input_checks=input_checks,
                warnings=warnings,
                errors=errors,
                may_check=design_check.status != "preview_only",
                **check_kwargs,
            )
        condition_rows.extend(_not_checked_condition_rows(config.conditions, unit_context, noise_image, reason))
        return _finalize_unit_plan(
            config=config,
            source=source,
            unit_context=unit_context,
            feat_dir=feat_dir,
            design_fsf=design_fsf,
            noise_image=noise_image,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
            condition_rows=condition_rows,
            input_checks=input_checks,
            warnings=warnings,
            errors=errors,
        )

    design_result = parse_fsl_design_file(Path(str(design_fsf)))
    if design_result.status == "error":
        _record_policy_problem(
            input_kind="design_parse",
            path=design_fsf,
            policy_key="design_parse",
            message=f"FEAT design.fsf could not be parsed cleanly: {'; '.join(design_result.errors)}",
            policy_resolver=policy_resolver,
            input_checks=input_checks,
            warnings=warnings,
            errors=errors,
            **check_kwargs,
        )
        if _noise_required(config):
            _record_noise_check(
                config=config,
                noise_image=noise_image,
                policy_resolver=policy_resolver,
                input_checks=input_checks,
                warnings=warnings,
                errors=errors,
                may_check=True,
                **check_kwargs,
            )
        condition_rows.extend(_not_checked_condition_rows(config.conditions, unit_context, noise_image, "design parse failed"))
        return _finalize_unit_plan(
            config=config,
            source=source,
            unit_context=unit_context,
            feat_dir=feat_dir,
            design_fsf=design_fsf,
            noise_image=noise_image,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
            condition_rows=condition_rows,
            input_checks=input_checks,
            warnings=warnings,
            errors=errors,
        )

    if design_result.warnings:
        input_checks.append(
            FslFeatInputCheckRow(
                input_kind="design_parse",
                path=design_fsf,
                exists=None,
                status="warning",
                policy=None,
                message="; ".join(design_result.warnings),
                **check_kwargs,
            )
        )
        warnings.extend(design_result.warnings)
    else:
        input_checks.append(
            FslFeatInputCheckRow(
                input_kind="design_parse",
                path=design_fsf,
                exists=None,
                status="ok",
                policy=None,
                message="Parsed FEAT EV-title metadata.",
                **check_kwargs,
            )
        )

    noise_check = _record_noise_check(
        config=config,
        noise_image=noise_image,
        policy_resolver=policy_resolver,
        input_checks=input_checks,
        warnings=warnings,
        errors=errors,
        may_check=True,
        **check_kwargs,
    )
    mapping = map_conditions_to_pe_numbers(
        config.conditions,
        design_result,
        case_sensitive=settings.case_sensitive,
        missing_policy="error",
    )
    for mapping_row in mapping.mappings:
        condition = _condition_by_id(config.conditions).get(mapping_row.condition_id)
        condition_row = _condition_mapping_row(
            mapping_row=mapping_row,
            condition=condition,
            unit_context=unit_context,
            feat_dir=feat_dir,
            noise_image=noise_image,
            settings=settings,
            policy_resolver=policy_resolver,
            input_checks=input_checks,
            warnings=warnings,
            errors=errors,
            may_check=True,
            roots=roots,
        )
        condition_rows.append(condition_row)

    if noise_check.status == "skipped":
        condition_rows = [_with_condition_status(row, "skipped", "noise image is unavailable") for row in condition_rows]

    return _finalize_unit_plan(
        config=config,
        source=source,
        unit_context=unit_context,
        feat_dir=feat_dir,
        design_fsf=design_fsf,
        noise_image=noise_image,
        excluded=excluded,
        exclusion_reason=exclusion_reason,
        condition_rows=condition_rows,
        input_checks=input_checks,
        warnings=warnings,
        errors=errors,
    )


def _record_root_check(
    root_state: _RootState,
    *,
    policy_resolver: _PolicyResolver,
    input_checks: list[FslFeatInputCheckRow],
    warnings: list[str],
    errors: list[str],
    subject_id: str,
    session_id: str,
    run_id: str,
) -> bool:
    if root_state.status == "preview_only":
        input_checks.append(
            FslFeatInputCheckRow(
                input_kind="root",
                path=root_state.path_preview,
                exists=None,
                status="preview_only",
                policy=None,
                message=root_state.message,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
            )
        )
        return True
    if root_state.status == "ok":
        input_checks.append(
            FslFeatInputCheckRow(
                input_kind="root",
                path=root_state.path_preview,
                exists=root_state.exists,
                status="ok",
                policy=None,
                message=None,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
            )
        )
        return True
    if root_state.missing_policy_applies:
        _record_policy_problem(
            input_kind="root",
            path=root_state.path_preview,
            policy_key="root",
            message=root_state.message or f"Root is missing: {root_state.path_preview}",
            policy_resolver=policy_resolver,
            input_checks=input_checks,
            warnings=warnings,
            errors=errors,
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
        )
        return False
    return True


def _check_path(
    *,
    input_kind: str,
    path: str | None,
    policy_key: str,
    missing_message: str,
    policy_resolver: _PolicyResolver,
    may_check: bool,
    input_checks: list[FslFeatInputCheckRow],
    warnings: list[str],
    errors: list[str],
    subject_id: str,
    session_id: str,
    run_id: str,
    condition_id: str | None = None,
) -> FslFeatInputCheckRow:
    if path is None or not may_check or not _path_is_checkable(path):
        row = FslFeatInputCheckRow(
            input_kind=input_kind,
            path=path,
            exists=None,
            status="preview_only",
            policy=None,
            message="Path contains unresolved placeholders, glob-like characters, or lacks a concrete root.",
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            condition_id=condition_id,
        )
        input_checks.append(row)
        return row

    exists = Path(path).exists()
    if exists:
        row = FslFeatInputCheckRow(
            input_kind=input_kind,
            path=path,
            exists=True,
            status="ok",
            policy=None,
            message=None,
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            condition_id=condition_id,
        )
        input_checks.append(row)
        return row

    status = _record_policy_problem(
        input_kind=input_kind,
        path=path,
        policy_key=policy_key,
        message=missing_message,
        policy_resolver=policy_resolver,
        input_checks=input_checks,
        warnings=warnings,
        errors=errors,
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
        condition_id=condition_id,
    )
    return input_checks[-1] if status else input_checks[-1]


def _record_policy_problem(
    *,
    input_kind: str,
    path: str | None,
    policy_key: str,
    message: str,
    policy_resolver: _PolicyResolver,
    input_checks: list[FslFeatInputCheckRow],
    warnings: list[str],
    errors: list[str],
    subject_id: str,
    session_id: str,
    run_id: str,
    condition_id: str | None = None,
) -> str:
    policy = policy_resolver.for_kind(policy_key)
    status = _status_for_policy(policy)
    input_checks.append(
        FslFeatInputCheckRow(
            input_kind=input_kind,
            path=path,
            exists=False if input_kind != "design_parse" else None,
            status=status,
            policy=policy,
            message=message,
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            condition_id=condition_id,
        )
    )
    if policy == MISSING_INPUT_POLICY_FAIL:
        errors.append(message)
    else:
        warnings.append(message)
    return status


def _record_noise_check(
    *,
    config: MvpaSetConfig,
    noise_image: str | None,
    policy_resolver: _PolicyResolver,
    input_checks: list[FslFeatInputCheckRow],
    warnings: list[str],
    errors: list[str],
    may_check: bool,
    subject_id: str,
    session_id: str,
    run_id: str,
) -> FslFeatInputCheckRow:
    if not _noise_required(config):
        row = FslFeatInputCheckRow(
            input_kind="noise_image",
            path=noise_image,
            exists=None,
            status="not_required",
            policy=None,
            message="Noise image is not required by configured noise normalization.",
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
        )
        input_checks.append(row)
        return row

    return _check_path(
        input_kind="noise_image",
        path=noise_image,
        policy_key="noise_image",
        missing_message=f"Noise variance image is missing: {noise_image}",
        policy_resolver=policy_resolver,
        may_check=may_check,
        input_checks=input_checks,
        warnings=warnings,
        errors=errors,
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
    )


def _condition_mapping_row(
    *,
    mapping_row: Any,
    condition: ConditionDefinition | None,
    unit_context: Mapping[str, Any],
    feat_dir: str | None,
    noise_image: str | None,
    settings: _TemplateSettings,
    policy_resolver: _PolicyResolver,
    input_checks: list[FslFeatInputCheckRow],
    warnings: list[str],
    errors: list[str],
    may_check: bool,
    roots: Mapping[str, str | Path] | None,
) -> FslFeatConditionPePlanRow:
    subject_id = str(unit_context["subject_id"])
    session_id = str(unit_context["session_id"])
    run_id = str(unit_context["run_id"])
    condition_id = str(mapping_row.condition_id or "<missing-id>")
    condition_warnings = list(mapping_row.warnings)
    condition_errors = list(mapping_row.errors)
    status = mapping_row.status

    if mapping_row.errors:
        policy_key = _condition_error_policy_key(mapping_row.errors)
        message = "; ".join(mapping_row.errors)
        policy = policy_resolver.for_kind(policy_key)
        input_checks.append(
            FslFeatInputCheckRow(
                input_kind=policy_key,
                path=None,
                exists=None,
                status=_status_for_policy(policy),
                policy=policy,
                message=message,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                condition_id=condition_id,
            )
        )
        if policy == MISSING_INPUT_POLICY_FAIL:
            errors.append(message)
            status = "error"
        elif policy == MISSING_INPUT_POLICY_SKIP:
            warnings.append(message)
            status = "skipped"
            condition_errors = []
            condition_warnings.append(message)
        else:
            warnings.append(message)
            status = "warning"
            condition_errors = []
            condition_warnings.append(message)
        _add_dependent_not_checked(
            input_checks,
            "pe_image",
            None,
            f"{policy_key} is unresolved",
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            condition_id=condition_id,
        )
        return FslFeatConditionPePlanRow(
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            condition_id=condition_id,
            requested_ev_title=mapping_row.requested_ev_title,
            matched_ev_title=mapping_row.matched_ev_title,
            matched_alias=mapping_row.matched_ev_title if mapping_row.matched_by == "alias" else None,
            ev_index=mapping_row.ev_index,
            pe_number=mapping_row.pe_number,
            pe_image=None,
            noise_image=noise_image,
            event_file=None,
            event_count=None,
            status=status,
            warnings=tuple(condition_warnings),
            errors=tuple(condition_errors),
        )

    pe_image = _render_pe_image(feat_dir, settings.pe_image_template, unit_context, mapping_row.pe_number)
    pe_check = _check_path(
        input_kind="pe_image",
        path=pe_image,
        policy_key="pe_image",
        missing_message=f"PE image is missing for condition {condition_id}: {pe_image}",
        policy_resolver=policy_resolver,
        may_check=may_check,
        input_checks=input_checks,
        warnings=warnings,
        errors=errors,
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
        condition_id=condition_id,
    )
    if pe_check.status == "warning":
        condition_warnings.append(pe_check.message or "PE image check produced a warning.")
        status = "warning"
    elif pe_check.status == "skipped":
        condition_warnings.append(pe_check.message or "PE image is unavailable.")
        status = "skipped"
    elif pe_check.status == "error":
        condition_errors.append(pe_check.message or "PE image is unavailable.")
        status = "error"
    elif pe_check.status == "preview_only":
        status = "preview_only"

    event_file, event_count, event_status, event_warnings, event_errors = _event_file_plan(
        mapping_row=mapping_row,
        condition=condition,
        unit_context=unit_context,
        event_settings=settings.event_files,
        policy_resolver=policy_resolver,
        input_checks=input_checks,
        may_check=may_check,
        roots=roots,
    )
    warnings.extend(event_warnings)
    errors.extend(event_errors)
    condition_warnings.extend(event_warnings)
    condition_errors.extend(event_errors)
    if event_status == "warning" and status == "ok":
        status = "warning"
    elif event_status in {"skipped", "error"}:
        status = event_status

    return FslFeatConditionPePlanRow(
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
        condition_id=condition_id,
        requested_ev_title=mapping_row.requested_ev_title,
        matched_ev_title=mapping_row.matched_ev_title,
        matched_alias=mapping_row.matched_ev_title if mapping_row.matched_by == "alias" else None,
        ev_index=mapping_row.ev_index,
        pe_number=mapping_row.pe_number,
        pe_image=pe_image,
        noise_image=noise_image,
        event_file=event_file,
        event_count=event_count,
        status=status,
        warnings=tuple(condition_warnings),
        errors=tuple(condition_errors),
    )


def _condition_error_policy_key(errors: Sequence[str]) -> str:
    if any("ambiguous" in error.lower() for error in errors):
        return "ambiguous_ev_title"
    return "condition_ev_title"


def _event_file_plan(
    *,
    mapping_row: Any,
    condition: ConditionDefinition | None,
    unit_context: Mapping[str, Any],
    event_settings: _EventFileSettings,
    policy_resolver: _PolicyResolver,
    input_checks: list[FslFeatInputCheckRow],
    may_check: bool,
    roots: Mapping[str, str | Path] | None,
) -> tuple[str | None, int | None, str, list[str], list[str]]:
    if event_settings.path_template is None:
        return None, None, "not_configured", [], []

    condition_id = str(mapping_row.condition_id or "<missing-id>")
    check_kwargs = {
        "subject_id": str(unit_context["subject_id"]),
        "session_id": str(unit_context["session_id"]),
        "run_id": str(unit_context["run_id"]),
        "condition_id": condition_id,
    }
    event_path = _render_event_file_path(
        event_settings,
        context=_event_file_context(mapping_row=mapping_row, condition=condition, unit_context=unit_context),
        roots=roots,
    )
    event_check = _check_path(
        input_kind="event_file",
        path=event_path,
        policy_key="event_file",
        missing_message=f"Event file is missing for condition {condition_id}: {event_path}",
        policy_resolver=policy_resolver,
        may_check=may_check,
        input_checks=input_checks,
        warnings=[],
        errors=[],
        **check_kwargs,
    )
    warnings = [event_check.message] if event_check.status in {"warning", "skipped"} and event_check.message else []
    errors = [event_check.message] if event_check.status == "error" and event_check.message else []
    if event_check.exists is not True or event_path is None:
        return event_path, None, event_check.status, warnings, errors

    try:
        event_count = _count_event_file_rows(Path(event_path))
    except OSError as exc:
        message = f"Event file could not be read for condition {condition_id}: {exc}"
        status = _record_policy_problem(
            input_kind="event_file_parse",
            path=event_path,
            policy_key="event_file_parse",
            message=message,
            policy_resolver=policy_resolver,
            input_checks=input_checks,
            warnings=warnings,
            errors=errors,
            **check_kwargs,
        )
        return event_path, None, status, warnings, errors

    input_checks.append(
        FslFeatInputCheckRow(
            input_kind="event_file_parse",
            path=event_path,
            exists=True,
            status="ok",
            policy=None,
            message=f"Counted {event_count} event row(s).",
            event_count=event_count,
            **check_kwargs,
        )
    )
    return event_path, event_count, "ok", warnings, errors


def _event_file_context(
    *,
    mapping_row: Any,
    condition: ConditionDefinition | None,
    unit_context: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(unit_context)
    condition_id = str(mapping_row.condition_id or "")
    payload.update(
        {
            "condition_id": condition_id,
            "condition": condition_id,
            "requested_ev_title": mapping_row.requested_ev_title,
            "matched_ev_title": mapping_row.matched_ev_title,
            "matched_alias": mapping_row.matched_ev_title if mapping_row.matched_by == "alias" else None,
            "event_id": _condition_event_id(condition) or condition_id,
        }
    )
    return payload


def _condition_event_id(condition: ConditionDefinition | None) -> str | None:
    if condition is None:
        return None
    selector = condition.selector if isinstance(condition.selector, Mapping) else {}
    return _optional_text(
        condition.fields.get("event_id")
        or condition.fields.get("event_name")
        or condition.fields.get("ev_name")
        or selector.get("event_id")
        or selector.get("event_name")
        or selector.get("ev_name")
    )


def _render_event_file_path(
    settings: _EventFileSettings,
    *,
    context: Mapping[str, Any],
    roots: Mapping[str, str | Path] | None,
) -> str | None:
    template = settings.path_template
    if template is None:
        return None
    rendered = _render_template(template, context)
    candidate = Path(rendered)
    if candidate.is_absolute():
        return candidate.as_posix()
    if settings.root_ref and roots is not None and settings.root_ref in roots:
        return (Path(roots[settings.root_ref]) / rendered).as_posix()
    if settings.root is not None:
        return (Path(_render_template(settings.root, context)) / rendered).as_posix()
    if settings.root_ref:
        return f"{settings.root_ref}/{rendered}"
    return rendered


def _count_event_file_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#"))


def _with_condition_status(row: FslFeatConditionPePlanRow, status: str, message: str) -> FslFeatConditionPePlanRow:
    warnings = row.warnings
    if message not in warnings:
        warnings = (*warnings, message)
    return FslFeatConditionPePlanRow(
        subject_id=row.subject_id,
        session_id=row.session_id,
        run_id=row.run_id,
        condition_id=row.condition_id,
        requested_ev_title=row.requested_ev_title,
        matched_ev_title=row.matched_ev_title,
        matched_alias=row.matched_alias,
        ev_index=row.ev_index,
        pe_number=row.pe_number,
        pe_image=row.pe_image,
        noise_image=row.noise_image,
        event_file=row.event_file,
        event_count=row.event_count,
        status=status,
        warnings=warnings,
        errors=row.errors,
    )


def _not_checked_condition_rows(
    conditions: Sequence[ConditionDefinition],
    unit_context: Mapping[str, Any],
    noise_image: str | None,
    reason: str,
) -> tuple[FslFeatConditionPePlanRow, ...]:
    return tuple(
        FslFeatConditionPePlanRow(
            subject_id=str(unit_context["subject_id"]),
            session_id=str(unit_context["session_id"]),
            run_id=str(unit_context["run_id"]),
            condition_id=condition.id,
            requested_ev_title=_condition_ev_title(condition),
            noise_image=noise_image,
            status="not_checked",
            warnings=(f"Condition-to-PE mapping not checked because {reason}.",),
        )
        for condition in conditions
    )


def _add_dependent_not_checked(
    input_checks: list[FslFeatInputCheckRow],
    input_kind: str,
    path: str | None,
    reason: str,
    *,
    subject_id: str,
    session_id: str,
    run_id: str,
    condition_id: str | None = None,
) -> None:
    input_checks.append(
        FslFeatInputCheckRow(
            input_kind=input_kind,
            path=path,
            exists=None,
            status="not_checked",
            policy=None,
            message=f"Not checked because {reason}.",
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            condition_id=condition_id,
        )
    )


def _finalize_unit_plan(
    *,
    config: MvpaSetConfig,
    source: PatternSourceConfig,
    unit_context: Mapping[str, Any],
    feat_dir: str | None,
    design_fsf: str | None,
    noise_image: str | None,
    excluded: bool,
    exclusion_reason: str | None,
    condition_rows: Sequence[FslFeatConditionPePlanRow],
    input_checks: Sequence[FslFeatInputCheckRow],
    warnings: Sequence[str],
    errors: Sequence[str],
) -> _UnitPlan:
    unit_warnings = list(warnings)
    unit_errors = list(errors)
    if excluded and exclusion_reason:
        unit_warnings.append(f"Unit is excluded: {exclusion_reason}")
    status = _aggregate_unit_status(condition_rows, input_checks, warnings=unit_warnings, errors=unit_errors)
    if excluded and status == "ok":
        status = "excluded"
    unit = FslFeatPatternSourcePlanRow(
        subject_id=str(unit_context["subject_id"]),
        session_id=str(unit_context["session_id"]),
        run_id=str(unit_context["run_id"]),
        task_id=_optional_text(unit_context.get("task_id")),
        direction=_optional_text(unit_context.get("direction")),
        model=_optional_text(unit_context.get("model")),
        pattern_source_name=source.name,
        backend=source.backend,
        feat_dir=feat_dir,
        design_fsf=design_fsf,
        noise_image=noise_image if _noise_required(config) else noise_image,
        status=status,
        excluded=excluded,
        exclusion_reason=exclusion_reason,
        warnings=tuple(_unique_text(unit_warnings)),
        errors=tuple(_unique_text(unit_errors)),
    )
    return _UnitPlan(
        unit=unit,
        conditions=tuple(condition_rows),
        input_checks=tuple(input_checks),
        warnings=tuple(_unique_text(unit_warnings)),
        errors=tuple(_unique_text(unit_errors)),
    )


def _unit_contexts(config: MvpaSetConfig, *, context: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    base_context = dict(context or {})
    for subject in config.selector.subjects:
        subject_values = _subject_values(subject)
        for session in config.selector.sessions:
            session_values = _session_values(session)
            for run in config.selector.runs:
                run_values = _run_values(run)
                row: dict[str, Any] = dict(base_context)
                row.update(subject_values)
                row.update(session_values)
                row.update(run_values)
                row.update(_entity_values(config))
                row["mvpa_set"] = config.name
                rows.append(row)
    return tuple(rows)


def _condition_by_id(conditions: Sequence[ConditionDefinition]) -> dict[str, ConditionDefinition]:
    return {condition.id: condition for condition in conditions}


def _subject_values(value: str) -> dict[str, str]:
    subject_id = _strip_entity_prefix(value, "sub")
    subject = value if value.startswith("sub-") else f"sub-{subject_id}"
    return {"subject_id": subject_id, "subject": subject, "subject_dir": subject}


def _session_values(value: str) -> dict[str, str]:
    session_id = _strip_entity_prefix(value, "ses")
    session = value if value.startswith("ses-") else f"ses-{session_id}"
    return {"session_id": session_id, "session": session, "session_dir": session}


def _run_values(value: str) -> dict[str, str]:
    run_id = _strip_entity_prefix(value, "run")
    run = value if value.startswith("run-") else f"run-{run_id}"
    return {"run_id": run_id, "run": run, "run_entity": run}


def _entity_values(config: MvpaSetConfig) -> dict[str, str | None]:
    task_id = _strip_entity_prefix(config.entities.task, "task") if config.entities.task else None
    direction = _strip_entity_prefix(config.entities.direction, "dir") if config.entities.direction else None
    resolution = _strip_entity_prefix(config.entities.resolution, "res") if config.entities.resolution else None
    return {
        "task_id": task_id,
        "task": task_id,
        "direction": direction,
        "dir": direction,
        "model": config.entities.model,
        "space": config.entities.space,
        "resolution": resolution,
        "res": resolution,
    }


def _strip_entity_prefix(value: str | None, prefix: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    marker = f"{prefix}-"
    return text[len(marker) :] if text.startswith(marker) else text


def _exclusion_for_unit(
    exclusions: Sequence[ExclusionRule],
    unit_context: Mapping[str, Any],
) -> tuple[bool, str | None]:
    for rule in exclusions:
        payload = dict(rule.fields)
        match_payload = payload.get("match")
        if isinstance(match_payload, Mapping):
            payload.update(match_payload)
        subject = _first_text(payload, "subject_id", "subject")
        session = _first_text(payload, "session_id", "session")
        run = _first_text(payload, "run_id", "run")
        if subject is None or session is None or run is None:
            continue
        if (
            _strip_entity_prefix(subject, "sub") == unit_context["subject_id"]
            and _strip_entity_prefix(session, "ses") == unit_context["session_id"]
            and _strip_entity_prefix(run, "run") == unit_context["run_id"]
        ):
            return True, rule.reason or _optional_text(payload.get("reason")) or rule.id
    return False, None


def _plan_context(
    *,
    config: MvpaSetConfig,
    source: PatternSourceConfig,
    root_state: _RootState,
    context: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = dict(context or {})
    payload.update(
        {
            "mvpa_set_name": config.name,
            "pattern_source_name": source.name,
            "backend": source.backend,
            "root_ref": root_state.root_ref,
            "root": root_state.path_preview,
            "event_threshold_status": "not_evaluated",
            "exclusions": tuple(_exclusion_context(rule) for rule in config.exclusions),
        }
    )
    if config.event_thresholds is not None:
        payload["event_thresholds"] = dict(config.event_thresholds.fields)
    else:
        payload["event_thresholds"] = None
    return payload


def _exclusion_context(rule: ExclusionRule) -> Mapping[str, Any]:
    payload = dict(rule.fields)
    payload["id"] = rule.id
    payload["reason"] = rule.reason
    payload["subject_id"] = rule.subject_id
    payload["session_id"] = rule.session_id
    payload["run_id"] = rule.run_id
    payload["source_config_field"] = rule.source_config_field
    payload["status"] = "configured"
    return payload


def _render_path(root: Path | None, template: str, context: Mapping[str, Any]) -> str:
    rendered = _render_template(template, context)
    if root is None:
        return rendered
    rendered_path = Path(rendered)
    if rendered_path.is_absolute():
        return rendered_path.as_posix()
    return (root / rendered_path).as_posix()


def _render_child_path(parent: str | None, template: str, context: Mapping[str, Any]) -> str | None:
    rendered = _render_template(template, context)
    rendered_path = Path(rendered)
    if rendered_path.is_absolute() or parent is None:
        return rendered_path.as_posix()
    return (Path(parent) / rendered_path).as_posix()


def _render_pe_image(
    feat_dir: str | None,
    template: str,
    context: Mapping[str, Any],
    pe_number: int | None,
) -> str | None:
    pe_context = dict(context)
    if pe_number is not None:
        pe_context["pe_number"] = pe_number
    return _render_child_path(feat_dir, template, pe_context)


def _render_template(template: str, context: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context or context[key] is None:
            return match.group(0)
        return str(context[key])

    return _PLACEHOLDER.sub(replace, template)


def _path_is_checkable(path: str) -> bool:
    return not _UNRESOLVED_PLACEHOLDER.search(path) and not any(char in path for char in _GLOB_CHARS)


def _noise_required(config: MvpaSetConfig) -> bool:
    noise = config.distance.noise_normalization
    variance_source = (noise.variance_source or "").strip().lower()
    return noise.method == NOISE_NORMALIZATION_DIAGONAL or variance_source in {"sigmasquareds", "sigma_squareds"}


def _dependency_reason(row: FslFeatInputCheckRow) -> str:
    if row.status == "preview_only":
        return "upstream path was preview-only"
    return row.message or f"{row.input_kind} status is {row.status}"


def _status_for_policy(policy: str) -> str:
    if policy == MISSING_INPUT_POLICY_FAIL:
        return "error"
    if policy == MISSING_INPUT_POLICY_SKIP:
        return "skipped"
    return "warning"


def _aggregate_plan_status(
    units: Sequence[FslFeatPatternSourcePlanRow],
    conditions: Sequence[FslFeatConditionPePlanRow],
    input_checks: Sequence[FslFeatInputCheckRow],
    *,
    errors: Sequence[str],
    warnings: Sequence[str],
) -> str:
    if errors or any(row.status == "error" for row in units) or any(row.status == "error" for row in conditions):
        return "error"
    if any(row.status == "error" for row in input_checks):
        return "error"
    warning_statuses = {"warning", "skipped", "excluded"}
    if warnings or any(row.status in warning_statuses for row in units):
        return "warning"
    if any(row.status in warning_statuses for row in conditions):
        return "warning"
    if any(row.status in warning_statuses for row in input_checks):
        return "warning"
    return "ok"


def _aggregate_unit_status(
    conditions: Sequence[FslFeatConditionPePlanRow],
    input_checks: Sequence[FslFeatInputCheckRow],
    *,
    warnings: Sequence[str],
    errors: Sequence[str],
) -> str:
    if errors or any(row.status == "error" for row in conditions) or any(row.status == "error" for row in input_checks):
        return "error"
    if any(row.status == "skipped" for row in conditions) or any(row.status == "skipped" for row in input_checks):
        return "skipped"
    if warnings or any(row.status == "warning" for row in conditions) or any(row.status == "warning" for row in input_checks):
        return "warning"
    if conditions and all(row.status == "not_checked" for row in conditions):
        return "not_checked"
    if input_checks and all(row.status in {"preview_only", "not_checked"} for row in input_checks):
        return "preview_only"
    return "ok"


def _policy_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    policies: dict[str, str] = {}
    for key, raw_policy in value.items():
        key_text = str(key)
        if key_text not in _POLICY_KEYS:
            continue
        policy = _normalize_policy(raw_policy)
        if policy is not None:
            policies[key_text] = policy
    return policies


def _normalize_policy(value: Any) -> str | None:
    text = _optional_text(value)
    if text in _SUPPORTED_POLICIES:
        return text
    return None


def _condition_ev_title(condition: ConditionDefinition) -> str | None:
    return _optional_text(condition.fields.get("ev_title") or condition.fields.get("fsl_ev_title"))


def _candidate_mvpa_set_name(document: Any) -> str | None:
    if not isinstance(document, Mapping):
        return None
    payload = document.get("mvpa_set") if isinstance(document.get("mvpa_set"), Mapping) else document
    if not isinstance(payload, Mapping):
        return None
    return _optional_text(payload.get("name"))


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in mapping:
            return _optional_text(mapping.get(key))
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _unique_text(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            unique.append(text)
            seen.add(text)
    return unique


def _json_safe_dataclass(value: Any) -> dict[str, Any]:
    return {field.name: _json_safe(getattr(value, field.name)) for field in dataclass_fields(value)}


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
    "FslFeatConditionPePlanRow",
    "FslFeatInputCheckRow",
    "FslFeatPatternSourcePlan",
    "FslFeatPatternSourcePlanRow",
    "plan_fsl_feat_pattern_source",
]
