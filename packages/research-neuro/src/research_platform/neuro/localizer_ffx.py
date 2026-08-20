"""Plan-only localizer fixed-effects inventory helpers.

This module inspects FEAT-like directory metadata and builds JSON-safe
subject-level fixed-effects plans. It does not run FSL, write outputs, create
ROIs, transform masks, load NIfTI files, or import optional neuroimaging
dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from itertools import product
from pathlib import Path
from typing import Any

from research_platform.neuro.fsl.feat_design import (
    ContrastAliasCopeMapping,
    parse_fsl_design_contrast_names,
    resolve_contrast_aliases_to_cope_numbers,
)
from research_platform.neuro.fsl.flame import build_fixed_effects_command_plan


@dataclass(frozen=True)
class LocalizerFeatSourceSpec:
    """A localizer run-level FEAT source declaration."""

    name: str
    root_ref: str
    feat_dir_template: str
    design_file: str = "design.fsf"
    mask_file: str = "mask.nii.gz"
    fields: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class LocalizerContrastAliasSpec:
    """A configured localizer contrast alias resolved through FEAT metadata."""

    contrast_id: str
    aliases: tuple[str, ...] = ()
    contrast_name: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RunExclusionSpec:
    """A run exclusion selector applied before inventory grouping."""

    id: str
    reason: str | None = None
    source_name: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class LocalizerRunSpec:
    """One expected localizer FEAT run."""

    source_name: str
    subject_id: str
    session_id: str | None
    task_id: str | None
    run_id: str
    fields: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RunLevelFeatInventoryRow:
    """Inventory state for one run-level FEAT and one configured contrast."""

    source_name: str
    subject_id: str
    session_id: str | None
    task_id: str | None
    run_id: str
    contrast_id: str
    feat_dir: str | None
    design_path: str | None
    mask_path: str | None
    contrast_name: str | None
    contrast_number: int | None
    cope_number: int | None
    varcope_number: int | None
    cope_path: str | None
    varcope_path: str | None
    complete: bool
    status: str
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MissingInputRow:
    """A missing or invalid plan prerequisite."""

    input_kind: str
    message: str
    source_name: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    contrast_id: str | None = None
    root_ref: str | None = None
    path: str | None = None
    severity: str = "error"
    status: str = "missing"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ExcludedRunRow:
    """A run removed from inventory by an explicit exclusion selector."""

    exclusion_id: str
    source_name: str
    subject_id: str
    session_id: str | None
    task_id: str | None
    run_id: str
    reason: str | None = None
    status: str = "excluded"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class SubjectLevelFfxJobPlanRow:
    """A subject-level fixed-effects job planned from complete run inputs."""

    job_id: str
    source_name: str
    subject_id: str
    session_id: str | None
    task_id: str | None
    contrast_id: str
    contrast_name: str | None
    contrast_number: int | None
    complete_run_count: int
    run_ids: tuple[str, ...]
    cope_inputs: tuple[str, ...]
    varcope_inputs: tuple[str, ...]
    mask_path: str
    output_dir: str
    work_dir: str
    commands: tuple[tuple[str, ...], ...]
    status: str = "planned"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PlannedOutputPathRow:
    """One output path that would be materialized by a future execution step."""

    job_id: str
    path_kind: str
    path: str
    source_name: str
    subject_id: str
    session_id: str | None
    task_id: str | None
    contrast_id: str
    status: str = "planned"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RootRefProvenanceRow:
    """Concrete caller-supplied root-ref state used during planning."""

    root_ref: str
    role: str
    path: str | None
    status: str
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class LocalizerFixedEffectsPlan:
    """Top-level JSON-safe localizer fixed-effects plan."""

    status: str
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    feat_sources: tuple[LocalizerFeatSourceSpec, ...] = ()
    contrast_aliases: tuple[LocalizerContrastAliasSpec, ...] = ()
    run_exclusions: tuple[RunExclusionSpec, ...] = ()
    inventory_rows: tuple[RunLevelFeatInventoryRow, ...] = ()
    excluded_run_rows: tuple[ExcludedRunRow, ...] = ()
    missing_input_rows: tuple[MissingInputRow, ...] = ()
    ffx_job_rows: tuple[SubjectLevelFfxJobPlanRow, ...] = ()
    output_path_rows: tuple[PlannedOutputPathRow, ...] = ()
    root_ref_rows: tuple[RootRefProvenanceRow, ...] = ()
    min_complete_runs: int = 1
    executed: bool = False

    @property
    def valid(self) -> bool:
        return self.status != "error"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        payload["plan_only"] = True
        return payload


@dataclass(frozen=True)
class _PlanConfig:
    workflow_name: str | None
    runtime_tag: str | None
    feat_sources: tuple[LocalizerFeatSourceSpec, ...]
    contrast_aliases: tuple[LocalizerContrastAliasSpec, ...]
    run_exclusions: tuple[RunExclusionSpec, ...]
    runs: tuple[LocalizerRunSpec, ...]
    output_root_ref: str | None
    output_subpath: str | None
    output_dir_template: str
    work_dir_template: str
    min_complete_runs: int
    case_sensitive: bool
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def plan_localizer_fixed_effects(
    document: Mapping[str, Any] | Any,
    *,
    roots: Mapping[str, str | Path | Mapping[str, Any]] | None = None,
    min_complete_runs: int | None = None,
    case_sensitive: bool | None = None,
) -> LocalizerFixedEffectsPlan:
    """Inventory run-level localizer FEATs and plan subject-level FFX jobs.

    Planning is read-only. The function reads only ``design.fsf`` plus the
    existence state of ``stats/cope<N>.nii.gz``, ``stats/varcope<N>.nii.gz``,
    and ``mask.nii.gz``-style files.
    """

    config = _parse_plan_config(
        document,
        min_complete_runs_override=min_complete_runs,
        case_sensitive_override=case_sensitive,
    )
    warnings = list(config.warnings)
    errors = list(config.errors)
    missing_inputs: list[MissingInputRow] = []
    root_ref_rows: list[RootRefProvenanceRow] = []
    inventory_rows: list[RunLevelFeatInventoryRow] = []
    excluded_rows: list[ExcludedRunRow] = []
    job_rows: list[SubjectLevelFfxJobPlanRow] = []
    output_rows: list[PlannedOutputPathRow] = []

    root_cache: dict[tuple[str, str], Path | None] = {}
    source_by_name = {source.name: source for source in config.feat_sources}
    for source in config.feat_sources:
        root_cache[("source", source.root_ref)] = _record_root_ref(
            root_ref_rows,
            roots,
            source.root_ref,
            role=f"feat_source:{source.name}",
            missing_inputs=missing_inputs,
        )

    output_root = None
    if config.output_root_ref is None:
        missing = MissingInputRow(
            input_kind="output_root_ref",
            message="No localizer fixed-effects output root_ref was configured.",
            status="missing",
        )
        missing_inputs.append(missing)
        errors.append(missing.message)
    else:
        output_root = _record_root_ref(
            root_ref_rows,
            roots,
            config.output_root_ref,
            role="fixed_effects_output",
            missing_inputs=missing_inputs,
        )
        if output_root is not None and config.output_subpath:
            output_context = _base_template_context(
                workflow_name=config.workflow_name,
                runtime_tag=config.runtime_tag,
            )
            rendered_subpath, subpath_messages = _render_template(config.output_subpath, output_context)
            warnings.extend(subpath_messages)
            output_root = output_root / rendered_subpath

    if not config.feat_sources:
        errors.append("At least one localizer FEAT source must be configured.")
    if not config.contrast_aliases:
        errors.append("At least one localizer contrast alias must be configured.")
    if not config.runs:
        errors.append("At least one localizer run must be configured or derivable from selectors.")

    for run in config.runs:
        exclusion_matches = [exclusion for exclusion in config.run_exclusions if _exclusion_matches(exclusion, run)]
        if exclusion_matches:
            for exclusion in exclusion_matches:
                excluded_rows.append(
                    ExcludedRunRow(
                        exclusion_id=exclusion.id,
                        source_name=run.source_name,
                        subject_id=run.subject_id,
                        session_id=run.session_id,
                        task_id=run.task_id,
                        run_id=run.run_id,
                        reason=exclusion.reason,
                    )
                )
            continue

        source = source_by_name.get(run.source_name)
        if source is None:
            message = f"No localizer FEAT source named {run.source_name!r} is configured."
            missing_inputs.append(_missing_for_run("feat_source", message, run))
            errors.append(message)
            continue

        source_root = root_cache.get(("source", source.root_ref))
        run_rows, run_missing = _inventory_run(
            source=source,
            run=run,
            source_root=source_root,
            aliases=config.contrast_aliases,
            workflow_name=config.workflow_name,
            runtime_tag=config.runtime_tag,
            case_sensitive=config.case_sensitive,
        )
        inventory_rows.extend(run_rows)
        missing_inputs.extend(run_missing)

    complete_rows = [row for row in inventory_rows if row.complete]
    groups: dict[tuple[str, str, str | None, str | None, str], list[RunLevelFeatInventoryRow]] = {}
    for row in complete_rows:
        key = (row.source_name, row.subject_id, row.session_id, row.task_id, row.contrast_id)
        groups.setdefault(key, []).append(row)

    for key, rows in sorted(groups.items(), key=lambda item: _group_sort_key(item[0])):
        source_name, subject_id, session_id, task_id, contrast_id = key
        rows = sorted(rows, key=lambda row: row.run_id)
        if len(rows) < config.min_complete_runs:
            message = (
                f"Insufficient complete runs for {source_name}/{subject_id}/"
                f"{_none_to_label(session_id)}/{_none_to_label(task_id)}/{contrast_id}: "
                f"{len(rows)} complete, requires {config.min_complete_runs}."
            )
            missing_inputs.append(
                MissingInputRow(
                    input_kind="insufficient_complete_runs",
                    message=message,
                    source_name=source_name,
                    subject_id=subject_id,
                    session_id=session_id,
                    task_id=task_id,
                    contrast_id=contrast_id,
                    status="insufficient",
                )
            )
            errors.append(message)
            continue
        if output_root is None:
            continue

        job, paths = _plan_job(
            source_name=source_name,
            subject_id=subject_id,
            session_id=session_id,
            task_id=task_id,
            contrast_id=contrast_id,
            rows=rows,
            output_root=output_root,
            output_dir_template=config.output_dir_template,
            work_dir_template=config.work_dir_template,
            workflow_name=config.workflow_name,
            runtime_tag=config.runtime_tag,
        )
        job_rows.append(job)
        output_rows.extend(paths)

    expected_groups = _expected_group_keys(config.runs, config.contrast_aliases, excluded_rows)
    observed_groups = set(groups)
    for key in sorted(expected_groups - observed_groups, key=_group_sort_key):
        source_name, subject_id, session_id, task_id, contrast_id = key
        message = (
            f"No complete localizer runs for {source_name}/{subject_id}/"
            f"{_none_to_label(session_id)}/{_none_to_label(task_id)}/{contrast_id}."
        )
        missing_inputs.append(
            MissingInputRow(
                input_kind="insufficient_complete_runs",
                message=message,
                source_name=source_name,
                subject_id=subject_id,
                session_id=session_id,
                task_id=task_id,
                contrast_id=contrast_id,
                status="insufficient",
            )
        )
        errors.append(message)

    for row in inventory_rows:
        warnings.extend(row.warnings)
        errors.extend(row.errors)
    for row in missing_inputs:
        if row.severity == "warning":
            warnings.append(row.message)
        else:
            errors.append(row.message)

    status = _status(warnings, errors)
    return LocalizerFixedEffectsPlan(
        status=status,
        warnings=tuple(_dedupe(warnings)),
        errors=tuple(_dedupe(errors)),
        feat_sources=config.feat_sources,
        contrast_aliases=config.contrast_aliases,
        run_exclusions=config.run_exclusions,
        inventory_rows=tuple(inventory_rows),
        excluded_run_rows=tuple(excluded_rows),
        missing_input_rows=tuple(missing_inputs),
        ffx_job_rows=tuple(job_rows),
        output_path_rows=tuple(output_rows),
        root_ref_rows=tuple(root_ref_rows),
        min_complete_runs=config.min_complete_runs,
        executed=False,
    )


def _parse_plan_config(
    document: Mapping[str, Any] | Any,
    *,
    min_complete_runs_override: int | None,
    case_sensitive_override: bool | None,
) -> _PlanConfig:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(document, Mapping):
        return _PlanConfig(
            workflow_name=None,
            runtime_tag=None,
            feat_sources=(),
            contrast_aliases=(),
            run_exclusions=(),
            runs=(),
            output_root_ref=None,
            output_subpath=None,
            output_dir_template=_DEFAULT_OUTPUT_DIR_TEMPLATE,
            work_dir_template=_DEFAULT_WORK_DIR_TEMPLATE,
            min_complete_runs=1,
            case_sensitive=True,
            errors=("localizer fixed-effects document must contain a mapping.",),
        )

    workflow = document.get("analysis_workflow") if isinstance(document.get("analysis_workflow"), Mapping) else None
    workflow_payload = workflow if isinstance(workflow, Mapping) else {}
    mvpa_payload = _mvpa_payload(document)
    ffx_payload = _ffx_payload(document, mvpa_payload)
    source_payload = {**dict(mvpa_payload), **dict(ffx_payload)} if (mvpa_payload or ffx_payload) else document

    workflow_name = _optional_text(workflow_payload.get("name"))
    runtime_tag = _optional_text(workflow_payload.get("runtime_tag"))
    feat_sources = tuple(_parse_feat_sources(source_payload, errors))
    contrast_aliases = tuple(_parse_contrast_aliases(source_payload, errors))
    run_exclusions = tuple(_parse_run_exclusions(source_payload, errors))
    selectors = _selector_payload(document, workflow_payload, ffx_payload)
    runs = tuple(_parse_runs(source_payload, selectors, feat_sources, errors))
    output_root_ref, output_subpath, output_dir_template, work_dir_template = _parse_output_settings(
        ffx_payload,
        workflow_payload,
        warnings,
    )
    configured_min = _optional_int(ffx_payload.get("min_complete_runs") if isinstance(ffx_payload, Mapping) else None)
    if min_complete_runs_override is not None:
        configured_min = min_complete_runs_override
    if configured_min is None:
        configured_min = 1
    if configured_min < 1:
        errors.append("min_complete_runs must be greater than or equal to 1.")
        configured_min = 1

    configured_case_sensitive = _optional_bool(
        ffx_payload.get("case_sensitive") if isinstance(ffx_payload, Mapping) else None
    )
    if case_sensitive_override is not None:
        configured_case_sensitive = case_sensitive_override
    if configured_case_sensitive is None:
        configured_case_sensitive = True

    return _PlanConfig(
        workflow_name=workflow_name,
        runtime_tag=runtime_tag,
        feat_sources=feat_sources,
        contrast_aliases=contrast_aliases,
        run_exclusions=run_exclusions,
        runs=runs,
        output_root_ref=output_root_ref,
        output_subpath=output_subpath,
        output_dir_template=output_dir_template,
        work_dir_template=work_dir_template,
        min_complete_runs=configured_min,
        case_sensitive=configured_case_sensitive,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _mvpa_payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    workflow = document.get("analysis_workflow")
    if isinstance(workflow, Mapping):
        extensions = workflow.get("extensions")
        if isinstance(extensions, Mapping) and isinstance(extensions.get("mvpa"), Mapping):
            return extensions["mvpa"]  # type: ignore[return-value]
        if isinstance(workflow.get("mvpa"), Mapping):
            return workflow["mvpa"]  # type: ignore[return-value]
    for key in ("mvpa", "mvpa_extension", "mvpa_workflow"):
        if isinstance(document.get(key), Mapping):
            return document[key]  # type: ignore[return-value]
    return {}


def _ffx_payload(document: Mapping[str, Any], mvpa_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for payload in (document, mvpa_payload):
        for key in ("localizer_fixed_effects", "localizer_ffx", "fixed_effects"):
            if isinstance(payload.get(key), Mapping):
                return payload[key]  # type: ignore[return-value]
    if _looks_like_direct_ffx(document):
        return document
    return {}


def _looks_like_direct_ffx(document: Mapping[str, Any]) -> bool:
    return any(
        key in document
        for key in (
            "contrast_aliases",
            "contrasts",
            "feat_sources",
            "localizer_feat_sources",
            "output_dir_template",
        )
    )


def _parse_feat_sources(payload: Mapping[str, Any], errors: list[str]) -> list[LocalizerFeatSourceSpec]:
    rows = _mapping_list(
        _first_present(payload, "feat_sources", "localizer_feat_sources", "sources"),
        "localizer_fixed_effects.feat_sources",
        errors,
    )
    sources: list[LocalizerFeatSourceSpec] = []
    names: list[str] = []
    for index, row in enumerate(rows):
        label = f"localizer_fixed_effects.feat_sources[{index}]"
        name = _optional_text(row.get("name") or row.get("id"))
        root_ref = _optional_text(row.get("root_ref"))
        template = _optional_text(
            row.get("feat_dir_template")
            or row.get("path_template")
            or row.get("template")
        )
        if name is None:
            errors.append(f"{label}.name must be defined.")
            name = f"source-{index}"
        if root_ref is None:
            errors.append(f"{label}.root_ref must be defined.")
            root_ref = ""
        if template is None:
            errors.append(f"{label}.feat_dir_template must be defined.")
            template = ""
        names.append(name)
        sources.append(
            LocalizerFeatSourceSpec(
                name=name,
                root_ref=root_ref,
                feat_dir_template=template,
                design_file=_optional_text(row.get("design_file")) or "design.fsf",
                mask_file=_optional_text(row.get("mask_file") or row.get("mask_path")) or "mask.nii.gz",
                fields=dict(row),
            )
        )
    for duplicate in _duplicates(names):
        errors.append(f"localizer_fixed_effects.feat_sources contains duplicate source name: {duplicate}.")
    return sources


def _parse_contrast_aliases(payload: Mapping[str, Any], errors: list[str]) -> list[LocalizerContrastAliasSpec]:
    raw = _first_present(
        payload,
        "contrast_aliases",
        "contrasts",
        "localizer_contrasts",
        "pe_mapping_aliases",
    )
    rows = _alias_rows(raw, "localizer_fixed_effects.contrast_aliases", errors)
    aliases: list[LocalizerContrastAliasSpec] = []
    ids: list[str] = []
    for index, row in enumerate(rows):
        label = f"localizer_fixed_effects.contrast_aliases[{index}]"
        contrast_id = _optional_text(
            row.get("id")
            or row.get("name")
            or row.get("condition")
            or row.get("contrast_id")
            or row.get("source_contrast")
        )
        if contrast_id is None:
            errors.append(f"{label}.id must be defined.")
            contrast_id = f"contrast-{index}"
        ids.append(contrast_id)
        alias_values = _string_sequence(
            row.get("aliases")
            or row.get("contrast_aliases")
            or row.get("names")
        )
        aliases.append(
            LocalizerContrastAliasSpec(
                contrast_id=contrast_id,
                aliases=tuple(alias_values),
                contrast_name=_optional_text(row.get("contrast_name") or row.get("fsl_contrast_name")),
                fields=dict(row),
            )
        )
    for duplicate in _duplicates(ids):
        errors.append(f"localizer_fixed_effects.contrast_aliases contains duplicate id: {duplicate}.")
    return aliases


def _parse_run_exclusions(payload: Mapping[str, Any], errors: list[str]) -> list[RunExclusionSpec]:
    raw = _first_present(payload, "run_exclusions", "excluded_runs")
    if raw is None and isinstance(payload.get("exclusions"), Mapping):
        raw = payload["exclusions"].get("runs")  # type: ignore[index]
    rows = _mapping_list(raw, "localizer_fixed_effects.run_exclusions", errors)
    exclusions: list[RunExclusionSpec] = []
    for index, row in enumerate(rows):
        selectors = row.get("selectors") if isinstance(row.get("selectors"), Mapping) else {}
        selector_payload = {**dict(selectors), **dict(row)}
        exclusion_id = _optional_text(row.get("id") or row.get("name")) or f"run-exclusion-{index}"
        exclusions.append(
            RunExclusionSpec(
                id=exclusion_id,
                reason=_optional_text(row.get("reason")),
                source_name=_optional_text(selector_payload.get("source_name") or selector_payload.get("source")),
                subject_id=_entity_value(selector_payload, "subject"),
                session_id=_entity_value(selector_payload, "session"),
                task_id=_entity_value(selector_payload, "task"),
                run_id=_entity_value(selector_payload, "run"),
                fields=dict(row),
            )
        )
    return exclusions


def _parse_runs(
    payload: Mapping[str, Any],
    selectors: Mapping[str, tuple[str, ...]],
    feat_sources: Sequence[LocalizerFeatSourceSpec],
    errors: list[str],
) -> list[LocalizerRunSpec]:
    explicit = _explicit_run_rows(payload)
    if explicit:
        runs: list[LocalizerRunSpec] = []
        source_names = tuple(source.name for source in feat_sources)
        for index, row in enumerate(explicit):
            subject_id = _entity_value(row, "subject")
            run_id = _entity_value(row, "run")
            if subject_id is None:
                errors.append(f"localizer_fixed_effects.runs[{index}].subject_id must be defined.")
                continue
            if run_id is None:
                errors.append(f"localizer_fixed_effects.runs[{index}].run_id must be defined.")
                continue
            requested_source = _optional_text(row.get("source_name") or row.get("source"))
            row_sources = (requested_source,) if requested_source is not None else source_names
            for source_name in row_sources:
                runs.append(
                    LocalizerRunSpec(
                        source_name=source_name,
                        subject_id=subject_id,
                        session_id=_entity_value(row, "session"),
                        task_id=_entity_value(row, "task"),
                        run_id=run_id,
                        fields=dict(row),
                    )
                )
        return runs

    subjects = selectors.get("subjects", ())
    sessions = selectors.get("sessions", ()) or (None,)
    tasks = selectors.get("tasks", ()) or (None,)
    run_ids = selectors.get("runs", ())
    runs = []
    for source, subject_id, session_id, task_id, run_id in product(feat_sources, subjects, sessions, tasks, run_ids):
        if subject_id is None or run_id is None:
            continue
        runs.append(
            LocalizerRunSpec(
                source_name=source.name,
                subject_id=subject_id,
                session_id=session_id,
                task_id=task_id,
                run_id=run_id,
            )
        )
    return runs


def _explicit_run_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("runs", "run_inventory", "localizer_runs"):
        raw = payload.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            rows = [item for item in raw if isinstance(item, Mapping)]
            if rows:
                return rows
    return []


def _selector_payload(
    document: Mapping[str, Any],
    workflow_payload: Mapping[str, Any],
    ffx_payload: Mapping[str, Any],
) -> Mapping[str, tuple[str, ...]]:
    source: dict[str, Any] = {}
    for payload in (workflow_payload, ffx_payload, document):
        for key in ("selector", "selectors"):
            if isinstance(payload.get(key), Mapping):
                source.update(payload[key])  # type: ignore[arg-type]
        if isinstance(payload.get("cohort"), Mapping):
            cohort = payload["cohort"]  # type: ignore[index]
            if isinstance(cohort.get("subjects"), Mapping):
                source["subjects"] = cohort["subjects"].get("include", cohort["subjects"].get("values"))  # type: ignore[index]
            elif "subjects" in cohort:
                source["subjects"] = cohort.get("subjects")
            for key in ("sessions", "tasks", "runs"):
                if key in cohort:
                    source[key] = cohort.get(key)
        for key in ("subjects", "sessions", "tasks", "runs"):
            if key in payload:
                source[key] = payload.get(key)
    return {
        "subjects": tuple(_string_sequence(source.get("subjects", source.get("subject")))),
        "sessions": tuple(_string_sequence(source.get("sessions", source.get("session")))),
        "tasks": tuple(_string_sequence(source.get("tasks", source.get("task")))),
        "runs": tuple(_string_sequence(source.get("runs", source.get("run")))),
    }


def _parse_output_settings(
    ffx_payload: Mapping[str, Any],
    workflow_payload: Mapping[str, Any],
    warnings: list[str],
) -> tuple[str | None, str | None, str, str]:
    output_payload = ffx_payload.get("outputs", ffx_payload.get("output", {}))
    if not isinstance(output_payload, Mapping):
        output_payload = {}
    root_ref = _optional_text(output_payload.get("root_ref") or output_payload.get("output_root_ref"))
    subpath = _optional_text(output_payload.get("path") or output_payload.get("relative_path"))
    output_roots = workflow_payload.get("output_roots", workflow_payload.get("outputs", {}))
    if root_ref is None and isinstance(output_roots, Mapping):
        root_decl = output_roots.get("runtime_root") or next(iter(output_roots.values()), None)
        if isinstance(root_decl, Mapping):
            root_ref = _optional_text(root_decl.get("root_ref"))
            subpath = subpath or _optional_text(root_decl.get("path") or root_decl.get("relative_path"))
    output_dir_template = _optional_text(
        output_payload.get("output_dir_template")
        or ffx_payload.get("output_dir_template")
    )
    work_dir_template = _optional_text(
        output_payload.get("work_dir_template")
        or ffx_payload.get("work_dir_template")
    )
    if output_dir_template is None:
        output_dir_template = _DEFAULT_OUTPUT_DIR_TEMPLATE
        warnings.append("No localizer fixed-effects output_dir_template configured; using a generic plan template.")
    if work_dir_template is None:
        work_dir_template = _DEFAULT_WORK_DIR_TEMPLATE
        warnings.append("No localizer fixed-effects work_dir_template configured; using a generic plan template.")
    return root_ref, subpath, output_dir_template, work_dir_template


def _record_root_ref(
    rows: list[RootRefProvenanceRow],
    roots: Mapping[str, str | Path | Mapping[str, Any]] | None,
    root_ref: str,
    *,
    role: str,
    missing_inputs: list[MissingInputRow],
) -> Path | None:
    existing = next((row for row in rows if row.root_ref == root_ref and row.role == role), None)
    if existing is not None:
        return Path(existing.path) if existing.path is not None else None
    root = _root_from_mapping(roots, root_ref)
    if root is None:
        message = f"No concrete root was supplied for root_ref {root_ref!r}."
        rows.append(RootRefProvenanceRow(root_ref=root_ref, role=role, path=None, status="missing", message=message))
        missing_inputs.append(
            MissingInputRow(
                input_kind="root_ref",
                message=message,
                root_ref=root_ref,
                status="missing",
            )
        )
        return None
    rows.append(RootRefProvenanceRow(root_ref=root_ref, role=role, path=str(root), status="resolved"))
    return root


def _root_from_mapping(
    roots: Mapping[str, str | Path | Mapping[str, Any]] | None,
    root_ref: str,
) -> Path | None:
    if roots is None or root_ref not in roots:
        return None
    raw = roots[root_ref]
    if isinstance(raw, Mapping):
        raw = raw.get("path", raw.get("root"))  # type: ignore[assignment]
    if raw is None:
        return None
    return Path(raw).resolve()


def _inventory_run(
    *,
    source: LocalizerFeatSourceSpec,
    run: LocalizerRunSpec,
    source_root: Path | None,
    aliases: Sequence[LocalizerContrastAliasSpec],
    workflow_name: str | None,
    runtime_tag: str | None,
    case_sensitive: bool,
) -> tuple[tuple[RunLevelFeatInventoryRow, ...], tuple[MissingInputRow, ...]]:
    missing: list[MissingInputRow] = []
    rows: list[RunLevelFeatInventoryRow] = []
    context = _run_template_context(
        source_name=source.name,
        subject_id=run.subject_id,
        session_id=run.session_id,
        task_id=run.task_id,
        run_id=run.run_id,
        workflow_name=workflow_name,
        runtime_tag=runtime_tag,
        extra={**dict(source.fields), **dict(run.fields)},
    )
    feat_dir: Path | None = None
    design_path: Path | None = None
    mask_path: Path | None = None
    base_errors: list[str] = []
    base_warnings: list[str] = []
    alias_mappings: Mapping[str, ContrastAliasCopeMapping] = {}

    if source_root is None:
        base_errors.append(f"Missing source root for root_ref {source.root_ref!r}.")
    else:
        rendered_template, template_warnings = _render_template(source.feat_dir_template, context)
        base_warnings.extend(template_warnings)
        feat_dir = source_root / rendered_template
        design_path = feat_dir / source.design_file
        mask_path = feat_dir / source.mask_file
        if not feat_dir.is_dir():
            message = f"Missing FEAT directory: {feat_dir}."
            missing.append(_missing_for_run("feat_dir", message, run, source=source, path=feat_dir))
            base_errors.append(message)
        elif not design_path.is_file():
            message = f"Missing design.fsf: {design_path}."
            missing.append(_missing_for_run("design_fsf", message, run, source=source, path=design_path))
            base_errors.append(message)
        else:
            design_result = parse_fsl_design_contrast_names(design_path.read_text(encoding="utf-8"))
            base_warnings.extend(design_result.warnings)
            base_errors.extend(design_result.errors)
            resolver = resolve_contrast_aliases_to_cope_numbers(
                [_alias_to_resolver_payload(alias) for alias in aliases],
                design_result,
                case_sensitive=case_sensitive,
            )
            alias_mappings = {
                mapping.contrast_id: mapping
                for mapping in resolver.mappings
                if mapping.contrast_id is not None
            }

    for alias in aliases:
        row_warnings = list(base_warnings)
        row_errors = list(base_errors)
        mapping = alias_mappings.get(alias.contrast_id)
        cope_path = None
        varcope_path = None
        complete = False
        if mapping is None and design_path is not None and design_path.is_file() and not base_errors:
            message = f"No contrast alias resolver row was produced for {alias.contrast_id!r}."
            row_errors.append(message)
            missing.append(_missing_for_run("contrast_alias", message, run, source=source, contrast_id=alias.contrast_id))
        elif mapping is not None:
            row_warnings.extend(mapping.warnings)
            row_errors.extend(mapping.errors)
            if mapping.status == "error":
                message = "; ".join(mapping.errors) or f"Contrast alias {alias.contrast_id!r} could not be resolved."
                missing.append(
                    _missing_for_run(
                        "contrast_alias",
                        message,
                        run,
                        source=source,
                        contrast_id=alias.contrast_id,
                    )
                )
            elif feat_dir is not None:
                cope_path = feat_dir / "stats" / str(mapping.cope_filename)
                varcope_path = feat_dir / "stats" / str(mapping.varcope_filename)
                if not cope_path.is_file():
                    message = f"Missing COPE file: {cope_path}."
                    row_errors.append(message)
                    missing.append(_missing_for_run("cope", message, run, source=source, contrast_id=alias.contrast_id, path=cope_path))
                if not varcope_path.is_file():
                    message = f"Missing VARCOPE file: {varcope_path}."
                    row_errors.append(message)
                    missing.append(_missing_for_run("varcope", message, run, source=source, contrast_id=alias.contrast_id, path=varcope_path))
                if mask_path is not None and not mask_path.is_file():
                    message = f"Missing mask file: {mask_path}."
                    row_errors.append(message)
                    missing.append(_missing_for_run("mask", message, run, source=source, contrast_id=alias.contrast_id, path=mask_path))
                complete = not row_errors and cope_path.is_file() and varcope_path.is_file() and mask_path is not None and mask_path.is_file()

        rows.append(
            RunLevelFeatInventoryRow(
                source_name=source.name,
                subject_id=run.subject_id,
                session_id=run.session_id,
                task_id=run.task_id,
                run_id=run.run_id,
                contrast_id=alias.contrast_id,
                feat_dir=str(feat_dir) if feat_dir is not None else None,
                design_path=str(design_path) if design_path is not None else None,
                mask_path=str(mask_path) if mask_path is not None else None,
                contrast_name=mapping.matched_contrast_name if mapping is not None else None,
                contrast_number=mapping.contrast_number if mapping is not None else None,
                cope_number=mapping.cope_number if mapping is not None else None,
                varcope_number=mapping.varcope_number if mapping is not None else None,
                cope_path=str(cope_path) if cope_path is not None else None,
                varcope_path=str(varcope_path) if varcope_path is not None else None,
                complete=complete,
                status=_inventory_status(complete, row_warnings, row_errors),
                warnings=tuple(row_warnings),
                errors=tuple(row_errors),
            )
        )
    return tuple(rows), tuple(missing)


def _plan_job(
    *,
    source_name: str,
    subject_id: str,
    session_id: str | None,
    task_id: str | None,
    contrast_id: str,
    rows: Sequence[RunLevelFeatInventoryRow],
    output_root: Path,
    output_dir_template: str,
    work_dir_template: str,
    workflow_name: str | None,
    runtime_tag: str | None,
) -> tuple[SubjectLevelFfxJobPlanRow, tuple[PlannedOutputPathRow, ...]]:
    first = rows[0]
    context = _run_template_context(
        source_name=source_name,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        run_id=None,
        workflow_name=workflow_name,
        runtime_tag=runtime_tag,
        extra={
            "contrast_id": contrast_id,
            "contrast": contrast_id,
            "contrast_name": first.contrast_name or "",
            "contrast_number": first.contrast_number or "",
            "cope_number": first.cope_number or "",
            "varcope_number": first.varcope_number or "",
        },
    )
    output_dir_text, output_warnings = _render_template(output_dir_template, context)
    work_dir_text, work_warnings = _render_template(work_dir_template, context)
    output_dir = (output_root / output_dir_text).resolve()
    work_dir = (output_root / work_dir_text).resolve()
    merged_cope = work_dir / "merged_cope.nii.gz"
    merged_varcope = work_dir / "merged_varcope.nii.gz"
    design_file = work_dir / "design.mat"
    t_contrast_file = work_dir / "design.con"
    plan = build_fixed_effects_command_plan(
        cope_inputs=[row.cope_path for row in rows if row.cope_path is not None],
        varcope_inputs=[row.varcope_path for row in rows if row.varcope_path is not None],
        mask_path=str(first.mask_path),
        work_dir=work_dir,
        output_dir=output_dir,
        design_file=design_file,
        t_contrast_file=t_contrast_file,
        merged_cope_path=merged_cope,
        merged_varcope_path=merged_varcope,
    )
    job_id = _job_id(source_name, subject_id, session_id, task_id, contrast_id)
    warnings = tuple(output_warnings + work_warnings)
    job = SubjectLevelFfxJobPlanRow(
        job_id=job_id,
        source_name=source_name,
        subject_id=subject_id,
        session_id=session_id,
        task_id=task_id,
        contrast_id=contrast_id,
        contrast_name=first.contrast_name,
        contrast_number=first.contrast_number,
        complete_run_count=len(rows),
        run_ids=tuple(row.run_id for row in rows),
        cope_inputs=tuple(str(row.cope_path) for row in rows if row.cope_path is not None),
        varcope_inputs=tuple(str(row.varcope_path) for row in rows if row.varcope_path is not None),
        mask_path=str(first.mask_path),
        output_dir=str(output_dir),
        work_dir=str(work_dir),
        commands=plan.commands,
        warnings=warnings,
    )
    path_specs = (
        ("output_dir", output_dir),
        ("work_dir", work_dir),
        ("merged_cope", merged_cope),
        ("merged_varcope", merged_varcope),
        ("design_file", design_file),
        ("t_contrast_file", t_contrast_file),
        ("output_cope", output_dir / "stats" / "cope1.nii.gz"),
        ("output_varcope", output_dir / "stats" / "varcope1.nii.gz"),
        ("output_mask", output_dir / "mask.nii.gz"),
    )
    paths = tuple(
        PlannedOutputPathRow(
            job_id=job_id,
            path_kind=kind,
            path=str(path),
            source_name=source_name,
            subject_id=subject_id,
            session_id=session_id,
            task_id=task_id,
            contrast_id=contrast_id,
        )
        for kind, path in path_specs
    )
    return job, paths


def _missing_for_run(
    input_kind: str,
    message: str,
    run: LocalizerRunSpec,
    *,
    source: LocalizerFeatSourceSpec | None = None,
    contrast_id: str | None = None,
    path: Path | None = None,
) -> MissingInputRow:
    return MissingInputRow(
        input_kind=input_kind,
        message=message,
        source_name=source.name if source is not None else run.source_name,
        subject_id=run.subject_id,
        session_id=run.session_id,
        task_id=run.task_id,
        run_id=run.run_id,
        contrast_id=contrast_id,
        root_ref=source.root_ref if source is not None else None,
        path=str(path) if path is not None else None,
        status="missing",
    )


def _alias_to_resolver_payload(alias: LocalizerContrastAliasSpec) -> dict[str, Any]:
    payload = {
        "id": alias.contrast_id,
        "aliases": list(alias.aliases),
    }
    if alias.contrast_name is not None:
        payload["contrast_name"] = alias.contrast_name
    return payload


def _expected_group_keys(
    runs: Sequence[LocalizerRunSpec],
    aliases: Sequence[LocalizerContrastAliasSpec],
    excluded_rows: Sequence[ExcludedRunRow],
) -> set[tuple[str, str, str | None, str | None, str]]:
    excluded_run_keys = {
        (row.source_name, row.subject_id, row.session_id, row.task_id, row.run_id)
        for row in excluded_rows
    }
    keys: set[tuple[str, str, str | None, str | None, str]] = set()
    for run in runs:
        if (run.source_name, run.subject_id, run.session_id, run.task_id, run.run_id) in excluded_run_keys:
            continue
        for alias in aliases:
            keys.add((run.source_name, run.subject_id, run.session_id, run.task_id, alias.contrast_id))
    return keys


def _exclusion_matches(exclusion: RunExclusionSpec, run: LocalizerRunSpec) -> bool:
    checks = (
        (exclusion.source_name, run.source_name),
        (exclusion.subject_id, run.subject_id),
        (exclusion.session_id, run.session_id),
        (exclusion.task_id, run.task_id),
        (exclusion.run_id, run.run_id),
    )
    return all(expected is None or expected == actual for expected, actual in checks)


def _run_template_context(
    *,
    source_name: str,
    subject_id: str,
    session_id: str | None,
    task_id: str | None,
    run_id: str | None,
    workflow_name: str | None,
    runtime_tag: str | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = _base_template_context(workflow_name=workflow_name, runtime_tag=runtime_tag)
    context.update(
        {
            "source": source_name,
            "source_name": source_name,
            "subject": subject_id,
            "subject_id": subject_id,
            "participant": subject_id,
            "participant_id": subject_id,
            "session": session_id or "",
            "session_id": session_id or "",
            "task": task_id or "",
            "task_id": task_id or "",
            "run": run_id or "",
            "run_id": run_id or "",
        }
    )
    if extra:
        context.update({str(key): "" if value is None else value for key, value in extra.items()})
    return context


def _base_template_context(*, workflow_name: str | None, runtime_tag: str | None) -> dict[str, Any]:
    return {
        "workflow": workflow_name or "",
        "workflow_name": workflow_name or "",
        "runtime_tag": runtime_tag or "",
    }


def _render_template(template: str, context: Mapping[str, Any]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    try:
        rendered = template.format_map(_TemplateContext(context, warnings))
    except ValueError as exc:
        warnings.append(f"Could not fully render template {template!r}: {exc}.")
        rendered = template
    return rendered.strip("/"), warnings


class _TemplateContext(dict[str, Any]):
    def __init__(self, context: Mapping[str, Any], warnings: list[str]) -> None:
        super().__init__(context)
        self._warnings = warnings

    def __missing__(self, key: str) -> str:
        self._warnings.append(f"Template placeholder {key!r} was not supplied.")
        return "{" + key + "}"


def _entity_value(row: Mapping[str, Any], entity: str) -> str | None:
    keys = {
        "subject": ("subject_id", "subject", "participant_id", "participant"),
        "session": ("session_id", "session"),
        "task": ("task_id", "task"),
        "run": ("run_id", "run"),
    }[entity]
    for key in keys:
        value = _optional_text(row.get(key))
        if value is not None:
            return value
    return None


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _alias_rows(raw: Any, label: str, errors: list[str]) -> list[Mapping[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        rows: list[Mapping[str, Any]] = []
        for key, value in raw.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("id", key)
                rows.append(row)
            elif isinstance(value, str):
                rows.append({"id": key, "aliases": [value]})
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                rows.append({"id": key, "aliases": list(value)})
            else:
                errors.append(f"{label}.{key} must be a mapping, string, or sequence of strings.")
        return rows
    return _mapping_list(raw, label, errors)


def _mapping_list(raw: Any, label: str, errors: list[str]) -> list[Mapping[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        errors.append(f"{label} must be a list of mappings.")
        return []
    rows: list[Mapping[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            errors.append(f"{label}[{index}] must be a mapping.")
            continue
        rows.append(item)
    return rows


def _string_sequence(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        values: list[str] = []
        for item in raw:
            text = _optional_text(item)
            if text is not None:
                values.append(text)
        return tuple(values)
    text = _optional_text(raw)
    return (text,) if text is not None else ()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _job_id(
    source_name: str,
    subject_id: str,
    session_id: str | None,
    task_id: str | None,
    contrast_id: str,
) -> str:
    parts = [source_name, subject_id]
    if session_id:
        parts.append(session_id)
    if task_id:
        parts.append(task_id)
    parts.append(contrast_id)
    return "_".join(part.replace("/", "_") for part in parts)


def _inventory_status(complete: bool, warnings: Sequence[str], errors: Sequence[str]) -> str:
    if errors:
        return "error"
    if warnings:
        return "warning" if complete else "missing"
    return "complete" if complete else "missing"


def _status(warnings: Sequence[str], errors: Sequence[str]) -> str:
    if errors:
        return "error"
    if warnings:
        return "warning"
    return "ok"


def _group_sort_key(key: tuple[str, str, str | None, str | None, str]) -> tuple[str, str, str, str, str]:
    source_name, subject_id, session_id, task_id, contrast_id = key
    return (source_name, subject_id, session_id or "", task_id or "", contrast_id)


def _none_to_label(value: str | None) -> str:
    return value if value is not None else "<none>"


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


def _json_safe_dataclass(value: Any) -> dict[str, Any]:
    return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}


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


_DEFAULT_OUTPUT_DIR_TEMPLATE = "{source_name}/{subject_id}/{session_id}/{task_id}/{contrast_id}/fixed-effects.feat"
_DEFAULT_WORK_DIR_TEMPLATE = "{source_name}/{subject_id}/{session_id}/{task_id}/{contrast_id}/work"


__all__ = [
    "ExcludedRunRow",
    "LocalizerContrastAliasSpec",
    "LocalizerFeatSourceSpec",
    "LocalizerFixedEffectsPlan",
    "LocalizerRunSpec",
    "MissingInputRow",
    "PlannedOutputPathRow",
    "RootRefProvenanceRow",
    "RunExclusionSpec",
    "RunLevelFeatInventoryRow",
    "SubjectLevelFfxJobPlanRow",
    "plan_localizer_fixed_effects",
]
