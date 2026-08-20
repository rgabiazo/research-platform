"""Plan-only MVPA ROI mask source lookup.

This module previews configured ROI mask and JSON sidecar paths for MVPA
workflows. It does not build ROI masks, read sidecar JSON, inspect NIfTI
geometry, run FSL, extract voxel patterns, compute MVPA distances, or write
outputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Any
import re

from .config import (
    MISSING_INPUT_POLICY_FAIL,
    MISSING_INPUT_POLICY_SKIP,
    MISSING_INPUT_POLICY_WARN,
    ROI_SOURCE_EXPLICIT_MASKS,
    ROI_SOURCE_ROI_SET,
    ROI_SOURCE_ROI_SET_PUBLICATION,
    ROI_SOURCE_ROI_SET_RUNTIME,
    ExclusionRule,
    MissingInputPolicy,
    MvpaSetConfig,
    RoiSourceConfig,
    parse_mvpa_set_document,
    validate_mvpa_set_document,
)


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_UNRESOLVED_PLACEHOLDER = re.compile(r"\{[^{}]+\}")
_GLOB_CHARS = frozenset("*?[")
_SUPPORTED_POLICIES = frozenset(
    {
        MISSING_INPUT_POLICY_WARN,
        MISSING_INPUT_POLICY_SKIP,
        MISSING_INPUT_POLICY_FAIL,
    }
)
_POLICY_KEYS = frozenset(
    {
        "default",
        "root",
        "roi_mask",
        "roi_sidecar",
        "roi_label",
        "roi_source",
        "provenance",
    }
)
_ROI_SET_SOURCES = frozenset({ROI_SOURCE_ROI_SET, ROI_SOURCE_ROI_SET_RUNTIME, ROI_SOURCE_ROI_SET_PUBLICATION})


@dataclass(frozen=True)
class MvpaRoiInputCheckRow:
    """One plan-only ROI source input check."""

    input_kind: str
    path: str | None
    exists: bool | None
    status: str
    policy: str | None = None
    message: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    roi_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MvpaRoiProvenancePlanRow:
    """One expected ROI sidecar/provenance preview row."""

    subject_id: str
    session_id: str
    run_id: str
    roi_label: str
    sidecar_path: str | None
    exists: bool | None
    status: str
    policy: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MvpaRoiSourcePlanRow:
    """One subject/session/run/ROI-label source preview row."""

    subject_id: str
    session_id: str
    run_id: str
    task_id: str | None
    direction: str | None
    model: str | None
    space: str | None
    resolution: str | None
    roi_source_name: str
    configured_source: str
    source: str
    roi_set_ref: str | None
    roi_label: str
    mask_path: str | None
    sidecar_path: str | None
    mask_exists: bool | None
    sidecar_exists: bool | None
    status: str
    excluded: bool = False
    exclusion_reason: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MvpaRoiSourcePlan:
    """JSON-safe plan-only MVPA ROI source result."""

    mvpa_set_name: str | None
    roi_source_name: str | None
    status: str
    rows: tuple[MvpaRoiSourcePlanRow, ...] = ()
    input_checks: tuple[MvpaRoiInputCheckRow, ...] = ()
    provenance_rows: tuple[MvpaRoiProvenancePlanRow, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    executed: bool = False

    @property
    def valid(self) -> bool:
        """Return whether config parsing and source selection succeeded."""

        return self.status != "invalid"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        return payload


def plan_mvpa_roi_sources(
    document_or_config: Mapping[str, Any] | MvpaSetConfig,
    *,
    roi_source_name: str | None = None,
    roots: Mapping[str, str | Path] | None = None,
    roi_sets: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    unit_contexts: Sequence[Mapping[str, Any]] | None = None,
    raise_on_fail_policy: bool = False,
) -> MvpaRoiSourcePlan:
    """Preview ROI mask and sidecar source paths for one MVPA ROI source.

    The returned plan is lookup/provenance planning only. Filesystem checks are
    limited to concrete paths with a supplied root, no unresolved placeholders,
    and no glob-like characters.
    """

    config_result = _coerce_config(document_or_config, context=context)
    if isinstance(config_result, MvpaRoiSourcePlan):
        return config_result
    config = config_result

    source_result = _select_roi_source(config, roi_source_name)
    if isinstance(source_result, MvpaRoiSourcePlan):
        return source_result
    source = source_result

    canonical_source = _canonical_source(source.source)
    if canonical_source not in {ROI_SOURCE_ROI_SET_RUNTIME, ROI_SOURCE_ROI_SET_PUBLICATION, ROI_SOURCE_EXPLICIT_MASKS}:
        return MvpaRoiSourcePlan(
            mvpa_set_name=config.name,
            roi_source_name=source.name,
            status="invalid",
            errors=(f"ROI source {source.name!r} has unsupported source {source.source!r}.",),
            context=dict(context or {}),
            executed=False,
        )

    policy_resolver = _PolicyResolver(source=source, top_level=config.missing_input_policy)
    roi_set_payload = _roi_set_payload(source, roi_sets=roi_sets)
    mask_specs = _mask_specs(config=config, source=source, canonical_source=canonical_source, roi_set_payload=roi_set_payload)
    plan_context = _plan_context(
        config=config,
        source=source,
        canonical_source=canonical_source,
        roi_set_payload=roi_set_payload,
        context=context,
    )

    rows: list[MvpaRoiSourcePlanRow] = []
    input_checks: list[MvpaRoiInputCheckRow] = []
    provenance_rows: list[MvpaRoiProvenancePlanRow] = []
    warnings: list[str] = []
    errors: list[str] = []

    if not mask_specs:
        message = f"ROI source {source.name!r} did not define any ROI labels to preview."
        _record_source_problem(
            message,
            policy_key="roi_label",
            policy_resolver=policy_resolver,
            input_checks=input_checks,
            warnings=warnings,
            errors=errors,
        )

    planned_unit_contexts = (
        tuple(unit_contexts)
        if unit_contexts is not None
        else _unit_contexts(config, context=context)
    )
    for unit_context in planned_unit_contexts:
        excluded, exclusion_reason = _exclusion_for_unit(config.exclusions, unit_context)
        for spec in mask_specs:
            row_plan = _plan_roi_row(
                config=config,
                source=source,
                canonical_source=canonical_source,
                spec=spec,
                roi_set_payload=roi_set_payload,
                roots=roots,
                unit_context=unit_context,
                excluded=excluded,
                exclusion_reason=exclusion_reason,
                policy_resolver=policy_resolver,
            )
            rows.append(row_plan.row)
            input_checks.extend(row_plan.input_checks)
            provenance_rows.extend(row_plan.provenance_rows)
            warnings.extend(row_plan.warnings)
            errors.extend(row_plan.errors)

    warnings = _unique_text(warnings)
    errors = _unique_text(errors)
    status = _aggregate_plan_status(rows, input_checks, provenance_rows, warnings=warnings, errors=errors)
    if raise_on_fail_policy and errors:
        raise ValueError("; ".join(errors))

    return MvpaRoiSourcePlan(
        mvpa_set_name=config.name,
        roi_source_name=source.name,
        status=status,
        rows=tuple(rows),
        input_checks=tuple(input_checks),
        provenance_rows=tuple(provenance_rows),
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
    exists: bool | None = None
    missing_policy_applies: bool = False


@dataclass(frozen=True)
class _MaskSpec:
    roi_label: str
    fields: Mapping[str, Any] = field(default_factory=dict)
    roi_set_fields: Mapping[str, Any] = field(default_factory=dict)
    missing_definition: bool = False


@dataclass(frozen=True)
class _RenderedPaths:
    root_state: _RootState | None
    mask_path: str | None
    sidecar_path: str | None


@dataclass(frozen=True)
class _RowPlan:
    row: MvpaRoiSourcePlanRow
    input_checks: tuple[MvpaRoiInputCheckRow, ...]
    provenance_rows: tuple[MvpaRoiProvenancePlanRow, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


class _PolicyResolver:
    def __init__(self, *, source: RoiSourceConfig, top_level: MissingInputPolicy) -> None:
        self._source_missing = _policy_mapping(source.fields.get("missing"))
        self._top_level_missing = _policy_mapping(top_level.fields)
        self._top_level_default = _normalize_policy(top_level.policy) or MISSING_INPUT_POLICY_WARN

    def for_kind(self, kind: str) -> str:
        for key in _policy_key_order(kind):
            source_value = self._source_missing.get(key)
            if source_value is not None:
                return source_value
        source_default = self._source_missing.get("default")
        if source_default is not None:
            return source_default
        for key in _policy_key_order(kind):
            top_level_value = self._top_level_missing.get(key)
            if top_level_value is not None:
                return top_level_value
        top_level_default = self._top_level_missing.get("default")
        if top_level_default is not None:
            return top_level_default
        return self._top_level_default


def _coerce_config(
    document_or_config: Mapping[str, Any] | MvpaSetConfig,
    *,
    context: Mapping[str, Any] | None,
) -> MvpaSetConfig | MvpaRoiSourcePlan:
    if isinstance(document_or_config, MvpaSetConfig):
        return document_or_config

    errors = validate_mvpa_set_document(document_or_config)
    if errors:
        return MvpaRoiSourcePlan(
            mvpa_set_name=_candidate_mvpa_set_name(document_or_config),
            roi_source_name=None,
            status="invalid",
            errors=tuple(errors),
            context=dict(context or {}),
            executed=False,
        )
    return parse_mvpa_set_document(document_or_config)


def _select_roi_source(config: MvpaSetConfig, roi_source_name: str | None) -> RoiSourceConfig | MvpaRoiSourcePlan:
    if roi_source_name is not None:
        for source in config.roi_sources:
            if source.name == roi_source_name:
                return source
        return MvpaRoiSourcePlan(
            mvpa_set_name=config.name,
            roi_source_name=roi_source_name,
            status="invalid",
            errors=(f"ROI source {roi_source_name!r} was not found.",),
        )

    if len(config.roi_sources) == 1:
        return config.roi_sources[0]
    if not config.roi_sources:
        return MvpaRoiSourcePlan(
            mvpa_set_name=config.name,
            roi_source_name=None,
            status="invalid",
            errors=("No ROI source is configured.",),
        )
    return MvpaRoiSourcePlan(
        mvpa_set_name=config.name,
        roi_source_name=None,
        status="invalid",
        errors=("Multiple ROI sources are configured; pass roi_source_name.",),
    )


def _plan_roi_row(
    *,
    config: MvpaSetConfig,
    source: RoiSourceConfig,
    canonical_source: str,
    spec: _MaskSpec,
    roi_set_payload: Mapping[str, Any],
    roots: Mapping[str, str | Path] | None,
    unit_context: Mapping[str, Any],
    excluded: bool,
    exclusion_reason: str | None,
    policy_resolver: _PolicyResolver,
) -> _RowPlan:
    row_context = {**unit_context, **_scalar_fields(spec.fields), "roi_label": spec.roi_label}
    paths = _render_source_paths(
        source=source,
        canonical_source=canonical_source,
        spec=spec,
        roi_set_payload=roi_set_payload,
        roots=roots,
        context=row_context,
    )
    input_checks: list[MvpaRoiInputCheckRow] = []
    provenance_rows: list[MvpaRoiProvenancePlanRow] = []
    warnings: list[str] = []
    errors: list[str] = []
    check_kwargs = _check_row_keys(row_context, spec.roi_label)

    if excluded and exclusion_reason:
        warnings.append(f"Unit is excluded: {exclusion_reason}")

    if spec.missing_definition and not excluded:
        _record_policy_problem(
            input_kind="roi_label",
            path=None,
            policy_key="roi_label",
            message=f"ROI label {spec.roi_label!r} was not found in provided ROI set metadata.",
            policy_resolver=policy_resolver,
            input_checks=input_checks,
            warnings=warnings,
            errors=errors,
            **check_kwargs,
        )

    _record_root_check(
        paths.root_state,
        policy_resolver=policy_resolver,
        input_checks=input_checks,
        warnings=warnings,
        errors=errors,
        excluded=excluded,
        **check_kwargs,
    )
    concrete_root = _has_concrete_root(paths.root_state)
    mask_check = _check_path(
        input_kind="roi_mask",
        path=paths.mask_path,
        policy_key="roi_mask",
        missing_message=f"ROI mask is missing: {paths.mask_path}",
        policy_resolver=policy_resolver,
        input_checks=input_checks,
        warnings=warnings,
        errors=errors,
        may_check=not excluded and concrete_root,
        preview_when_uncheckable=not excluded,
        **check_kwargs,
    )

    sidecar_check: MvpaRoiInputCheckRow
    if mask_check.status in {"skipped", "error"}:
        sidecar_check = _add_dependent_not_checked(
            input_checks,
            "roi_sidecar",
            paths.sidecar_path,
            f"ROI mask status is {mask_check.status}",
            **check_kwargs,
        )
    else:
        sidecar_check = _check_path(
            input_kind="roi_sidecar",
            path=paths.sidecar_path,
            policy_key="roi_sidecar",
            missing_message=f"ROI sidecar is missing: {paths.sidecar_path}",
            policy_resolver=policy_resolver,
            input_checks=input_checks,
            warnings=warnings,
            errors=errors,
            may_check=not excluded and concrete_root and mask_check.status != "preview_only",
            preview_when_uncheckable=not excluded,
            **check_kwargs,
        )

    provenance_rows.append(
        MvpaRoiProvenancePlanRow(
            subject_id=str(row_context["subject_id"]),
            session_id=str(row_context["session_id"]),
            run_id=str(row_context["run_id"]),
            roi_label=spec.roi_label,
            sidecar_path=paths.sidecar_path,
            exists=sidecar_check.exists,
            status=sidecar_check.status,
            policy=sidecar_check.policy,
            message=sidecar_check.message,
        )
    )

    row_status = _row_status(mask_check, sidecar_check, input_checks, excluded=excluded, warnings=warnings, errors=errors)
    row = MvpaRoiSourcePlanRow(
        subject_id=str(row_context["subject_id"]),
        session_id=str(row_context["session_id"]),
        run_id=str(row_context["run_id"]),
        task_id=_optional_text(row_context.get("task_id")),
        direction=_optional_text(row_context.get("direction")),
        model=_optional_text(row_context.get("model")),
        space=_optional_text(row_context.get("space")),
        resolution=_optional_text(row_context.get("resolution")),
        roi_source_name=source.name,
        configured_source=source.source,
        source=canonical_source,
        roi_set_ref=source.roi_set_ref,
        roi_label=spec.roi_label,
        mask_path=paths.mask_path,
        sidecar_path=paths.sidecar_path,
        mask_exists=mask_check.exists,
        sidecar_exists=sidecar_check.exists,
        status=row_status,
        excluded=excluded,
        exclusion_reason=exclusion_reason,
        warnings=tuple(_unique_text(warnings)),
        errors=tuple(_unique_text(errors)),
    )
    return _RowPlan(
        row=row,
        input_checks=tuple(input_checks),
        provenance_rows=tuple(provenance_rows),
        warnings=tuple(_unique_text(warnings)),
        errors=tuple(_unique_text(errors)),
    )


def _render_source_paths(
    *,
    source: RoiSourceConfig,
    canonical_source: str,
    spec: _MaskSpec,
    roi_set_payload: Mapping[str, Any],
    roots: Mapping[str, str | Path] | None,
    context: Mapping[str, Any],
) -> _RenderedPaths:
    if canonical_source == ROI_SOURCE_EXPLICIT_MASKS:
        return _render_explicit_paths(source=source, spec=spec, roots=roots, context=context)
    if canonical_source == ROI_SOURCE_ROI_SET_PUBLICATION:
        return _render_roi_set_publication_paths(
            source=source,
            spec=spec,
            roi_set_payload=roi_set_payload,
            roots=roots,
            context=context,
        )
    return _render_roi_set_runtime_paths(
        source=source,
        spec=spec,
        roi_set_payload=roi_set_payload,
        roots=roots,
        context=context,
    )


def _render_explicit_paths(
    *,
    source: RoiSourceConfig,
    spec: _MaskSpec,
    roots: Mapping[str, str | Path] | None,
    context: Mapping[str, Any],
) -> _RenderedPaths:
    fields = _merge_mappings(source.fields, spec.fields)
    template = _first_text(
        fields,
        "mask_template",
        "mask_path",
        "mask_pattern",
        "path",
        "relative_path",
        "pattern",
    )
    root_state = _resolve_root(
        root_ref=_first_text(fields, "root_ref") or source.root_ref,
        literal_root=_first_text(fields, "root"),
        subpath=None,
        roots=roots,
    )
    mask_path = _render_path(root_state, template or "{roi_label}_mask.nii.gz", context)
    sidecar_template = _first_text(fields, "sidecar_template", "sidecar_path", "sidecar_pattern")
    sidecar_path = _render_path(root_state, sidecar_template, context) if sidecar_template else _sidecar_path(mask_path)
    return _RenderedPaths(root_state=root_state, mask_path=mask_path, sidecar_path=sidecar_path)


def _render_roi_set_publication_paths(
    *,
    source: RoiSourceConfig,
    spec: _MaskSpec,
    roi_set_payload: Mapping[str, Any],
    roots: Mapping[str, str | Path] | None,
    context: Mapping[str, Any],
) -> _RenderedPaths:
    source_fields = dict(source.fields)
    publication = _publication_settings(source_fields, roi_set_payload)
    mask_template = _source_mask_template(source_fields, source=source, path_key="publication_path")
    root_state = _resolve_root(
        root_ref=_first_text(source_fields, "publication_root_ref")
        or _nested_text(publication, ("root", "root_ref"))
        or source.root_ref,
        literal_root=_first_text(source_fields, "root") or _root_literal(publication.get("root")),
        subpath=_publication_subpath(source_fields, publication),
        roots=roots,
    )
    if mask_template is not None:
        mask_path = _render_path(root_state, mask_template, context)
    else:
        mask_path = _build_publication_mask_path(
            root_state,
            context=context,
            roi_label=spec.roi_label,
            contrast_alias=_contrast_alias(source_fields, spec, roi_set_payload),
            mask_desc=_publication_mask_desc(source_fields, spec, publication, context),
            datatype=_datatype(source_fields, roi_set_payload),
        )
    sidecar_template = _first_text(source_fields, "sidecar_template", "sidecar_path", "sidecar_pattern")
    sidecar_path = _render_path(root_state, sidecar_template, context) if sidecar_template else _sidecar_path(mask_path)
    return _RenderedPaths(root_state=root_state, mask_path=mask_path, sidecar_path=sidecar_path)


def _render_roi_set_runtime_paths(
    *,
    source: RoiSourceConfig,
    spec: _MaskSpec,
    roi_set_payload: Mapping[str, Any],
    roots: Mapping[str, str | Path] | None,
    context: Mapping[str, Any],
) -> _RenderedPaths:
    source_fields = dict(source.fields)
    outputs = _outputs_settings(roi_set_payload)
    mask_template = _source_mask_template(source_fields, source=source, path_key="runtime_path")
    root_state = _resolve_root(
        root_ref=_first_text(source_fields, "runtime_root_ref")
        or _nested_text(outputs, ("root_ref",))
        or source.root_ref,
        literal_root=_first_text(source_fields, "root"),
        subpath=_runtime_subpath(source_fields, outputs),
        roots=roots,
    )
    if mask_template is not None:
        mask_path = _render_path(root_state, mask_template, context)
    else:
        mask_path = _build_runtime_mask_path(
            root_state,
            context=context,
            roi_set_ref=source.roi_set_ref or _optional_text(roi_set_payload.get("name")) or source.name,
            roi_label=spec.roi_label,
            method_desc=_runtime_method_desc(source_fields, spec, roi_set_payload),
            datatype=_datatype(source_fields, roi_set_payload),
        )
    sidecar_template = _first_text(source_fields, "sidecar_template", "sidecar_path", "sidecar_pattern")
    sidecar_path = _render_path(root_state, sidecar_template, context) if sidecar_template else _sidecar_path(mask_path)
    return _RenderedPaths(root_state=root_state, mask_path=mask_path, sidecar_path=sidecar_path)


def _record_root_check(
    root_state: _RootState | None,
    *,
    policy_resolver: _PolicyResolver,
    input_checks: list[MvpaRoiInputCheckRow],
    warnings: list[str],
    errors: list[str],
    excluded: bool,
    subject_id: str,
    session_id: str,
    run_id: str,
    roi_label: str,
) -> None:
    if root_state is None:
        return
    if excluded:
        input_checks.append(
            MvpaRoiInputCheckRow(
                input_kind="root",
                path=root_state.path_preview,
                exists=None,
                status="not_checked",
                message="Not checked because the unit is excluded.",
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                roi_label=roi_label,
            )
        )
        return
    if root_state.status == "preview_only":
        input_checks.append(
            MvpaRoiInputCheckRow(
                input_kind="root",
                path=root_state.path_preview,
                exists=None,
                status="preview_only",
                message=root_state.message,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                roi_label=roi_label,
            )
        )
        return
    if root_state.status == "ok":
        input_checks.append(
            MvpaRoiInputCheckRow(
                input_kind="root",
                path=root_state.path_preview,
                exists=root_state.exists,
                status="ok",
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                roi_label=roi_label,
            )
        )
        return
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
            roi_label=roi_label,
        )


def _check_path(
    *,
    input_kind: str,
    path: str | None,
    policy_key: str,
    missing_message: str,
    policy_resolver: _PolicyResolver,
    input_checks: list[MvpaRoiInputCheckRow],
    warnings: list[str],
    errors: list[str],
    may_check: bool,
    preview_when_uncheckable: bool = False,
    subject_id: str,
    session_id: str,
    run_id: str,
    roi_label: str,
) -> MvpaRoiInputCheckRow:
    if path is None:
        row = MvpaRoiInputCheckRow(
            input_kind=input_kind,
            path=None,
            exists=None,
            status="not_checked",
            message="No path template was available.",
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            roi_label=roi_label,
        )
        input_checks.append(row)
        return row
    if not may_check:
        status = "preview_only" if preview_when_uncheckable else "not_checked"
        message = (
            "Path contains unresolved placeholders, glob-like characters, or lacks a concrete root."
            if preview_when_uncheckable
            else "Not checked because the unit is excluded or a required input was skipped."
        )
        row = MvpaRoiInputCheckRow(
            input_kind=input_kind,
            path=path,
            exists=None,
            status=status,
            message=message,
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            roi_label=roi_label,
        )
        input_checks.append(row)
        return row
    if not _path_is_checkable(path):
        row = MvpaRoiInputCheckRow(
            input_kind=input_kind,
            path=path,
            exists=None,
            status="preview_only",
            message="Path contains unresolved placeholders, glob-like characters, or lacks a concrete root.",
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            roi_label=roi_label,
        )
        input_checks.append(row)
        return row

    exists = Path(path).exists()
    if exists:
        row = MvpaRoiInputCheckRow(
            input_kind=input_kind,
            path=path,
            exists=True,
            status="ok",
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            roi_label=roi_label,
        )
        input_checks.append(row)
        return row

    _record_policy_problem(
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
        roi_label=roi_label,
    )
    return input_checks[-1]


def _record_policy_problem(
    *,
    input_kind: str,
    path: str | None,
    policy_key: str,
    message: str,
    policy_resolver: _PolicyResolver,
    input_checks: list[MvpaRoiInputCheckRow],
    warnings: list[str],
    errors: list[str],
    subject_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    roi_label: str | None = None,
) -> None:
    policy = policy_resolver.for_kind(policy_key)
    input_checks.append(
        MvpaRoiInputCheckRow(
            input_kind=input_kind,
            path=path,
            exists=False,
            status=_status_for_policy(policy),
            policy=policy,
            message=message,
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            roi_label=roi_label,
        )
    )
    if policy == MISSING_INPUT_POLICY_FAIL:
        errors.append(message)
    else:
        warnings.append(message)


def _record_source_problem(
    message: str,
    *,
    policy_key: str,
    policy_resolver: _PolicyResolver,
    input_checks: list[MvpaRoiInputCheckRow],
    warnings: list[str],
    errors: list[str],
) -> None:
    _record_policy_problem(
        input_kind="roi_source",
        path=None,
        policy_key=policy_key,
        message=message,
        policy_resolver=policy_resolver,
        input_checks=input_checks,
        warnings=warnings,
        errors=errors,
    )


def _add_dependent_not_checked(
    input_checks: list[MvpaRoiInputCheckRow],
    input_kind: str,
    path: str | None,
    reason: str,
    *,
    subject_id: str,
    session_id: str,
    run_id: str,
    roi_label: str,
) -> MvpaRoiInputCheckRow:
    row = MvpaRoiInputCheckRow(
        input_kind=input_kind,
        path=path,
        exists=None,
        status="not_checked",
        policy=None,
        message=f"Not checked because {reason}.",
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
        roi_label=roi_label,
    )
    input_checks.append(row)
    return row


def _mask_specs(
    *,
    config: MvpaSetConfig,
    source: RoiSourceConfig,
    canonical_source: str,
    roi_set_payload: Mapping[str, Any],
) -> tuple[_MaskSpec, ...]:
    if canonical_source == ROI_SOURCE_EXPLICIT_MASKS:
        return _explicit_mask_specs(source)
    return _roi_set_mask_specs(config=config, source=source, roi_set_payload=roi_set_payload)


def _explicit_mask_specs(source: RoiSourceConfig) -> tuple[_MaskSpec, ...]:
    requested_labels = _label_values(source.fields)
    masks = source.masks
    if not masks:
        labels = requested_labels or _label_values(source.fields, allow_label=True)
        return tuple(_MaskSpec(roi_label=label, fields=dict(source.fields)) for label in labels)

    specs: list[_MaskSpec] = []
    requested = set(requested_labels) if requested_labels else None
    for mask in masks:
        label_values = _label_values(mask, allow_label=True)
        if not label_values and requested_labels:
            label_values = requested_labels
        for label in label_values:
            if requested is not None and label not in requested:
                continue
            specs.append(_MaskSpec(roi_label=label, fields=dict(mask)))
    return tuple(specs)


def _roi_set_mask_specs(
    *,
    config: MvpaSetConfig,
    source: RoiSourceConfig,
    roi_set_payload: Mapping[str, Any],
) -> tuple[_MaskSpec, ...]:
    requested_labels = _label_values(source.fields)
    roi_defs = _roi_definitions(roi_set_payload)
    roi_by_label = {label: fields for label, fields in roi_defs}
    labels = requested_labels or tuple(label for label, _fields in roi_defs)
    if not labels:
        labels = _label_values(source.fields, allow_label=True)
    specs: list[_MaskSpec] = []
    for label in labels:
        roi_fields = roi_by_label.get(label, {})
        specs.append(
            _MaskSpec(
                roi_label=label,
                fields=dict(source.fields),
                roi_set_fields=roi_fields,
                missing_definition=bool(roi_defs) and label not in roi_by_label,
            )
        )
    return tuple(specs)


def _roi_set_payload(source: RoiSourceConfig, *, roi_sets: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not source.roi_set_ref or roi_sets is None or source.roi_set_ref not in roi_sets:
        return {}
    payload = roi_sets[source.roi_set_ref]
    if isinstance(payload, Mapping):
        inner = payload.get("roi_set")
        return dict(inner) if isinstance(inner, Mapping) else dict(payload)
    fields = getattr(payload, "fields", None)
    if isinstance(fields, Mapping):
        result = dict(fields)
        name = getattr(payload, "name", None)
        if name is not None:
            result.setdefault("name", str(name))
        rois = getattr(payload, "rois", None)
        if isinstance(rois, Sequence) and not isinstance(rois, (str, bytes, bytearray)):
            result.setdefault("rois", tuple(_roi_definition_from_object(roi) for roi in rois))
        return result
    return {}


def _roi_definition_from_object(value: Any) -> Mapping[str, Any]:
    fields = getattr(value, "fields", None)
    result = dict(fields) if isinstance(fields, Mapping) else {}
    label = getattr(value, "label", None)
    if label is not None:
        result.setdefault("label", str(label))
    desc = getattr(value, "desc", None)
    if desc is not None:
        result.setdefault("desc", str(desc))
    family = getattr(value, "family", None)
    if family is not None:
        result.setdefault("family", str(family))
    return result


def _roi_definitions(roi_set_payload: Mapping[str, Any]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    raw = roi_set_payload.get("rois")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    definitions: list[tuple[str, Mapping[str, Any]]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        label = _optional_text(item.get("label") or item.get("roi_label"))
        if label is not None:
            definitions.append((label, dict(item)))
    return tuple(definitions)


def _resolve_root(
    *,
    root_ref: str | None,
    literal_root: str | None,
    subpath: str | None,
    roots: Mapping[str, str | Path] | None,
) -> _RootState:
    if root_ref and roots is not None and root_ref in roots:
        root = _append_subpath(Path(roots[root_ref]), subpath)
        preview = root.as_posix()
        if not _path_is_checkable(preview):
            return _RootState(root_ref=root_ref, root=None, path_preview=preview, status="preview_only")
        return _RootState(
            root_ref=root_ref,
            root=root,
            path_preview=preview,
            status="ok" if root.exists() else "missing",
            exists=root.exists(),
            missing_policy_applies=not root.exists(),
        )
    if literal_root is not None:
        root = _append_subpath(Path(_render_template(literal_root, {})), subpath)
        preview = root.as_posix()
        if not _path_is_checkable(preview):
            return _RootState(root_ref=root_ref, root=None, path_preview=preview, status="preview_only")
        return _RootState(
            root_ref=root_ref,
            root=root,
            path_preview=preview,
            status="ok" if root.exists() else "missing",
            exists=root.exists(),
            missing_policy_applies=not root.exists(),
        )
    if root_ref and roots is None:
        preview = _join_preview(root_ref, subpath)
        return _RootState(
            root_ref=root_ref,
            root=None,
            path_preview=preview,
            status="preview_only",
            message=f"No concrete roots mapping supplied for root_ref {root_ref!r}.",
        )
    if root_ref:
        preview = _join_preview(root_ref, subpath)
        return _RootState(
            root_ref=root_ref,
            root=None,
            path_preview=preview,
            status="missing",
            message=f"No concrete root was supplied for root_ref {root_ref!r}.",
            missing_policy_applies=True,
        )
    if subpath:
        return _RootState(
            root_ref=None,
            root=None,
            path_preview=subpath,
            status="preview_only",
            message="No concrete root was supplied for the configured subpath.",
        )
    return _RootState(
        root_ref=None,
        root=None,
        path_preview=None,
        status="preview_only",
        message="No root_ref or root was supplied.",
    )


def _append_subpath(root: Path, subpath: str | None) -> Path:
    if not subpath:
        return root
    rendered = Path(subpath)
    return rendered if rendered.is_absolute() else root / rendered


def _render_path(root_state: _RootState | None, template: str, context: Mapping[str, Any]) -> str:
    rendered = _render_template(template, context)
    rendered_path = Path(rendered)
    if rendered_path.is_absolute():
        return rendered_path.as_posix()
    if root_state is not None and root_state.root is not None:
        return (root_state.root / rendered_path).as_posix()
    if root_state is not None and root_state.path_preview:
        return _join_preview(root_state.path_preview, rendered)
    return rendered_path.as_posix()


def _build_runtime_mask_path(
    root_state: _RootState,
    *,
    context: Mapping[str, Any],
    roi_set_ref: str,
    roi_label: str,
    method_desc: str,
    datatype: str,
) -> str:
    filename = _roi_mask_filename(context=context, roi_label=roi_label, method_desc=method_desc)
    relative = _join_preview(
        _runtime_masks_relative_base(roi_set_ref),
        _join_preview(_entity_dirs(context, datatype=datatype), filename),
    )
    return _render_path(root_state, relative, context)


def _build_publication_mask_path(
    root_state: _RootState,
    *,
    context: Mapping[str, Any],
    roi_label: str,
    contrast_alias: str,
    mask_desc: str,
    datatype: str,
) -> str:
    filename = _published_loso_mask_filename(
        context=context,
        roi_label=roi_label,
        contrast_alias=contrast_alias,
        mask_desc=mask_desc,
    )
    relative = _join_preview("masks", _join_preview(_entity_dirs(context, datatype=datatype), filename))
    return _render_path(root_state, relative, context)


def _runtime_masks_relative_base(roi_set_ref: str) -> str:
    return _join_preview("rois", roi_set_ref)


def _entity_dirs(context: Mapping[str, Any], *, datatype: str) -> str:
    subject_dir = _subject_dir(_optional_text(context.get("subject_id")) or "{subject_id}")
    session_id = _optional_text(context.get("session_id"))
    if session_id:
        return _join_preview(subject_dir, _join_preview(_session_dir(session_id), datatype))
    return _join_preview(subject_dir, datatype)


def _roi_mask_filename(*, context: Mapping[str, Any], roi_label: str, method_desc: str) -> str:
    entities = (
        ("sub", _strip_entity_prefix(_optional_text(context.get("subject_id")) or "{subject_id}", "sub")),
        ("ses", _strip_entity_prefix(_optional_text(context.get("session_id")), "ses")),
        ("task", _label_or_placeholder(_optional_text(context.get("task_id")) or "{task_id}")),
        ("dir", _optional_label(_optional_text(context.get("direction")))),
        ("space", _label_or_placeholder(_optional_text(context.get("space")) or "{space}")),
        ("res", _optional_label(_optional_text(context.get("resolution")))),
        ("label", _label_or_placeholder(roi_label)),
        ("desc", _label_or_placeholder(method_desc)),
    )
    return f"{_entity_stem(entities)}_mask.nii.gz"


def _published_loso_mask_filename(
    *,
    context: Mapping[str, Any],
    roi_label: str,
    contrast_alias: str,
    mask_desc: str,
) -> str:
    entities = (
        ("sub", _strip_entity_prefix(_optional_text(context.get("subject_id")) or "{subject_id}", "sub")),
        ("ses", _strip_entity_prefix(_optional_text(context.get("session_id")), "ses")),
        ("task", _label_or_placeholder(_optional_text(context.get("task_id")) or "{task_id}")),
        ("dir", _optional_label(_optional_text(context.get("direction")))),
        ("space", _label_or_placeholder(_optional_text(context.get("space")) or "{space}")),
        ("res", _label_or_placeholder(_optional_text(context.get("resolution")) or "{resolution}")),
        ("label", _label_or_placeholder(roi_label)),
        ("contrast", _label_or_placeholder(contrast_alias)),
        ("desc", _label_or_placeholder(mask_desc)),
    )
    return f"{_entity_stem(entities)}_mask.nii.gz"


def _entity_stem(entities: Iterable[tuple[str, str | None]]) -> str:
    return "_".join(f"{key}-{value}" for key, value in entities if value not in (None, ""))


def _sidecar_path(mask_path: str | None) -> str | None:
    if mask_path is None:
        return None
    path = Path(mask_path)
    if path.name.endswith(".nii.gz"):
        return path.with_name(f"{path.name[:-7]}.json").as_posix()
    return path.with_suffix(".json").as_posix()


def _source_mask_template(source_fields: Mapping[str, Any], *, source: RoiSourceConfig, path_key: str) -> str | None:
    for key in ("mask_template", "mask_path", "mask_pattern"):
        value = _optional_text(source_fields.get(key))
        if value is not None:
            return value
    path_value = _optional_text(source_fields.get(path_key))
    if path_value is not None and _path_looks_like_mask_template(path_value):
        return path_value
    for value in (source.path, source.pattern):
        text = _optional_text(value)
        if text is not None:
            return text
    return None


def _publication_subpath(source_fields: Mapping[str, Any], publication: Mapping[str, Any]) -> str | None:
    path = _optional_text(source_fields.get("publication_path"))
    if path is not None and not _path_looks_like_mask_template(path):
        return path
    return _nested_text(publication, ("root", "path")) or _nested_text(publication, ("root", "subpath"))


def _runtime_subpath(source_fields: Mapping[str, Any], outputs: Mapping[str, Any]) -> str | None:
    path = _optional_text(source_fields.get("runtime_path"))
    if path is not None and not _path_looks_like_mask_template(path):
        return path
    return _optional_text(outputs.get("path") or outputs.get("relative_path") or outputs.get("subpath"))


def _path_looks_like_mask_template(value: str) -> bool:
    lower = value.lower()
    return lower.endswith((".nii", ".nii.gz")) or "_mask" in lower or "label-" in lower or "{roi_label}" in value


def _publication_settings(source_fields: Mapping[str, Any], roi_set_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    settings: dict[str, Any] = {}
    roi_publication = roi_set_payload.get("publication")
    if isinstance(roi_publication, Mapping):
        settings.update(dict(roi_publication))
    source_publication = source_fields.get("publication")
    if isinstance(source_publication, Mapping):
        settings.update(dict(source_publication))
    return settings


def _outputs_settings(roi_set_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    outputs = roi_set_payload.get("outputs")
    if not isinstance(outputs, Mapping):
        return {}
    if "root_ref" in outputs or "path" in outputs or "relative_path" in outputs:
        return dict(outputs)
    for key in ("runtime_root", "runtime", "root", "output_root", "derivative_root"):
        value = outputs.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _publication_mask_desc(
    source_fields: Mapping[str, Any],
    spec: _MaskSpec,
    publication: Mapping[str, Any],
    context: Mapping[str, Any],
) -> str:
    template = (
        _optional_text(source_fields.get("mask_desc"))
        or _optional_text(publication.get("mask_desc"))
        or "{model}LOSOFlame1Sphere{sphere_radius_mm}mm"
    )
    values = {
        **context,
        **_scalar_fields(spec.roi_set_fields),
        **_scalar_fields(source_fields),
        "roi_label": spec.roi_label,
    }
    return _label_or_placeholder(_render_template(template, values))


def _runtime_method_desc(source_fields: Mapping[str, Any], spec: _MaskSpec, roi_set_payload: Mapping[str, Any]) -> str:
    value = (
        _optional_text(source_fields.get("mask_desc"))
        or _optional_text(spec.roi_set_fields.get("desc"))
        or _contrast_desc(spec.roi_set_fields, roi_set_payload)
        or spec.roi_label
    )
    return _label_or_placeholder(value)


def _contrast_alias(source_fields: Mapping[str, Any], spec: _MaskSpec, roi_set_payload: Mapping[str, Any]) -> str:
    explicit = _optional_text(source_fields.get("contrast_alias"))
    if explicit is not None:
        return _label_or_placeholder(explicit)
    contrast_id = _contrast_id(source_fields, spec.roi_set_fields, roi_set_payload)
    contrast_desc = _contrast_desc(spec.roi_set_fields, roi_set_payload) or contrast_id
    aliases = _merge_mappings(
        _mapping(roi_set_payload.get("contrast_aliases")),
        _mapping(_mapping(roi_set_payload.get("publication")).get("contrast_aliases")),
        _mapping(_mapping(source_fields.get("publication")).get("contrast_aliases")),
        _mapping(source_fields.get("contrast_aliases")),
    )
    for key in (contrast_id, contrast_desc):
        if key is not None and aliases.get(key) is not None:
            return _label_or_placeholder(str(aliases[key]))
    if contrast_id is None:
        return "{contrast_alias}"
    return _safe_camel_alias(contrast_desc or contrast_id)


def _contrast_id(
    source_fields: Mapping[str, Any],
    roi_fields: Mapping[str, Any],
    roi_set_payload: Mapping[str, Any],
) -> str | None:
    explicit = _first_text(source_fields, "contrast_alias", "contrast_id", "contrast", "source_contrast")
    if explicit is not None and source_fields.get("contrast_alias") is None:
        return explicit
    roi_value = _first_text(roi_fields, "contrast_id", "contrast", "source_contrast")
    if roi_value is not None:
        return roi_value
    contrasts = _contrast_mappings(roi_set_payload)
    if len(contrasts) == 1:
        return _first_text(contrasts[0], "id", "name", "contrast_id", "contrast")
    return None


def _contrast_desc(roi_fields: Mapping[str, Any], roi_set_payload: Mapping[str, Any]) -> str | None:
    roi_contrast = _first_text(roi_fields, "contrast_id", "contrast", "source_contrast")
    for contrast in _contrast_mappings(roi_set_payload):
        contrast_id = _first_text(contrast, "id", "name", "contrast_id", "contrast")
        if roi_contrast is None or roi_contrast == contrast_id:
            return _first_text(contrast, "desc", "contrast_desc") or contrast_id
    return _first_text(roi_fields, "desc")


def _contrast_mappings(roi_set_payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = roi_set_payload.get("contrasts")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    return tuple(dict(item) for item in raw if isinstance(item, Mapping))


def _datatype(source_fields: Mapping[str, Any], roi_set_payload: Mapping[str, Any]) -> str:
    return _optional_text(source_fields.get("datatype") or roi_set_payload.get("datatype")) or "func"


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


def _subject_values(value: str) -> dict[str, str]:
    subject_id = _strip_entity_prefix(value, "sub") or value
    subject = value if value.startswith("sub-") else f"sub-{subject_id}"
    return {"subject_id": subject_id, "subject": subject, "subject_dir": subject}


def _session_values(value: str) -> dict[str, str]:
    session_id = _strip_entity_prefix(value, "ses") or value
    session = value if value.startswith("ses-") else f"ses-{session_id}"
    return {"session_id": session_id, "session": session, "session_dir": session}


def _run_values(value: str) -> dict[str, str]:
    run_id = _strip_entity_prefix(value, "run") or value
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
    source: RoiSourceConfig,
    canonical_source: str,
    roi_set_payload: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = dict(context or {})
    payload.update(
        {
            "mvpa_set_name": config.name,
            "roi_source_name": source.name,
            "configured_source": source.source,
            "source": canonical_source,
            "roi_set_ref": source.roi_set_ref,
            "roi_set_available": bool(roi_set_payload),
            "exclusions": tuple(_exclusion_context(rule) for rule in config.exclusions),
        }
    )
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


def _check_row_keys(context: Mapping[str, Any], roi_label: str) -> dict[str, str]:
    return {
        "subject_id": str(context["subject_id"]),
        "session_id": str(context["session_id"]),
        "run_id": str(context["run_id"]),
        "roi_label": roi_label,
    }


def _row_status(
    mask_check: MvpaRoiInputCheckRow,
    sidecar_check: MvpaRoiInputCheckRow,
    input_checks: Sequence[MvpaRoiInputCheckRow],
    *,
    excluded: bool,
    warnings: Sequence[str],
    errors: Sequence[str],
) -> str:
    if excluded:
        return "excluded"
    if errors or any(row.status == "error" for row in input_checks):
        return "error"
    if mask_check.status == "skipped" or sidecar_check.status == "skipped":
        return "skipped"
    if warnings or any(row.status == "warning" for row in input_checks):
        return "warning"
    if mask_check.status == "preview_only" or sidecar_check.status == "preview_only":
        return "preview_only"
    if mask_check.status == "not_checked" or sidecar_check.status == "not_checked":
        return "not_checked"
    return "ok"


def _has_concrete_root(root_state: _RootState | None) -> bool:
    return root_state is None or root_state.root is not None


def _aggregate_plan_status(
    rows: Sequence[MvpaRoiSourcePlanRow],
    input_checks: Sequence[MvpaRoiInputCheckRow],
    provenance_rows: Sequence[MvpaRoiProvenancePlanRow],
    *,
    warnings: Sequence[str],
    errors: Sequence[str],
) -> str:
    if errors or any(row.status == "error" for row in rows) or any(row.status == "error" for row in input_checks):
        return "error"
    if warnings or any(row.status in {"warning", "skipped", "excluded"} for row in rows):
        return "warning"
    if any(row.status in {"warning", "skipped"} for row in input_checks):
        return "warning"
    if any(row.status in {"warning", "skipped"} for row in provenance_rows):
        return "warning"
    if rows and all(row.status == "preview_only" for row in rows):
        return "preview_only"
    return "ok"


def _status_for_policy(policy: str) -> str:
    if policy == MISSING_INPUT_POLICY_FAIL:
        return "error"
    if policy == MISSING_INPUT_POLICY_SKIP:
        return "skipped"
    return "warning"


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


def _policy_key_order(kind: str) -> tuple[str, ...]:
    if kind == "roi_sidecar":
        return ("roi_sidecar", "provenance")
    if kind == "provenance":
        return ("provenance", "roi_sidecar")
    return (kind,)


def _normalize_policy(value: Any) -> str | None:
    text = _optional_text(value)
    if text in _SUPPORTED_POLICIES:
        return text
    return None


def _canonical_source(source: str) -> str:
    if source == ROI_SOURCE_ROI_SET:
        return ROI_SOURCE_ROI_SET_RUNTIME
    return source


def _label_values(mapping: Mapping[str, Any], *, allow_label: bool = False) -> tuple[str, ...]:
    for key in ("roi_labels", "labels"):
        values = _string_sequence(mapping.get(key))
        if values:
            return values
    if allow_label:
        return _string_sequence(mapping.get("roi_label") or mapping.get("label"))
    return ()


def _scalar_fields(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in mapping.items()
        if isinstance(value, (str, int, float, bool)) and value is not None
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _merge_mappings(*mappings: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        merged.update(dict(mapping))
    return merged


def _nested_text(mapping: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return _optional_text(value)


def _root_literal(value: Any) -> str | None:
    return _optional_text(value) if isinstance(value, (str, Path)) else None


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


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[str] = []
        for item in value:
            text = _optional_text(item)
            if text is not None:
                values.append(text)
        return tuple(values)
    text = _optional_text(value)
    return (text,) if text is not None else ()


def _render_template(template: str, context: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context or context[key] is None:
            return match.group(0)
        return str(context[key])

    return _PLACEHOLDER.sub(replace, template)


def _path_is_checkable(path: str) -> bool:
    return not _UNRESOLVED_PLACEHOLDER.search(path) and not any(char in path for char in _GLOB_CHARS)


def _join_preview(*parts: str | None) -> str:
    output = ""
    for part in parts:
        if part in (None, ""):
            continue
        text = str(part).strip("/")
        if not text:
            continue
        output = text if not output else f"{output}/{text}"
    return output


def _subject_dir(value: str) -> str:
    text = _strip_entity_prefix(value, "sub") or value
    return text if text.startswith("sub-") else f"sub-{text}"


def _session_dir(value: str) -> str:
    text = _strip_entity_prefix(value, "ses") or value
    return text if text.startswith("ses-") else f"ses-{text}"


def _strip_entity_prefix(value: str | None, prefix: str) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    marker = f"{prefix}-"
    return text[len(marker) :] if text.startswith(marker) else text


def _optional_label(value: str | None) -> str | None:
    if value is None:
        return None
    return _label_or_placeholder(value)


def _label_or_placeholder(value: str) -> str:
    if _UNRESOLVED_PLACEHOLDER.search(value):
        return value
    label = "".join(character for character in str(value) if character.isalnum())
    return label or value


def _safe_camel_alias(value: str) -> str:
    if _UNRESOLVED_PLACEHOLDER.search(value):
        return value
    words = re.findall(r"[A-Za-z0-9]+", str(value))
    alias = "".join(word[:1].upper() + word[1:] for word in words)
    return _label_or_placeholder(alias or "Contrast")


def _candidate_mvpa_set_name(document: Any) -> str | None:
    if not isinstance(document, Mapping):
        return None
    payload = document.get("mvpa_set") if isinstance(document.get("mvpa_set"), Mapping) else document
    if not isinstance(payload, Mapping):
        return None
    return _optional_text(payload.get("name"))


def _unique_text(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            output.append(text)
            seen.add(text)
    return output


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
    "MvpaRoiInputCheckRow",
    "MvpaRoiProvenancePlanRow",
    "MvpaRoiSourcePlan",
    "MvpaRoiSourcePlanRow",
    "plan_mvpa_roi_sources",
]
