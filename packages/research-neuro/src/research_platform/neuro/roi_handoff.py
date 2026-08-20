"""Plan-only handoff from localizer FFX outputs into LOSO ROI planning.

The helpers in this module bridge completed subject-level localizer fixed
effects into the existing LOSO ROI workflow. They validate and preview the
handoff only: no ROI masks, group maps, extraction tables, distance tables, or
publication artifacts are created.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
import copy

from research_platform.neuro.localizer_ffx import LocalizerFixedEffectsPlan
from research_platform.neuro import roi_execution, roi_loso
from research_platform.neuro.roi_execution import RoiExecutionContext


_REQUIRED_FFX_OUTPUT_KINDS = ("output_cope", "output_varcope", "output_mask")
_CATALOG_METADATA_KEYS = frozenset(
    {
        "label",
        "roi_label",
        "family",
        "roi_family",
        "desc",
        "description",
        "group",
        "roi_group",
        "catalog",
        "category",
        "tags",
        "hemisphere",
        "network",
        "source",
        "source_contrast",
        "contrast",
        "contrast_id",
    }
)
_SEMANTIC_ROI_KEYS = frozenset(
    {
        "coordinate",
        "seed_coordinate",
        "full_sample_seed_coordinate",
        "loso_peak_coordinate",
        "search_radius_mm",
        "sphere_radius_mm",
        "radius_mm",
        "z_threshold",
        "exploratory_z_threshold",
        "allow_below_threshold_fallback",
        "min_voxels_warn",
        "min_voxels_fail",
        "mask_intersection_policy",
        "coverage_masks",
        "backend_config",
        "fsl_flame1",
    }
)


@dataclass(frozen=True)
class CompletedSubjectFfxOutputInventoryRow:
    """Inventory row for one planned subject-level FFX output bundle."""

    source_job_id: str
    source_name: str | None
    subject_id: str | None
    session_id: str | None
    task_id: str | None
    run_ids: tuple[str, ...]
    model: str | None
    contrast_id: str | None
    contrast_name: str | None
    contrast_number: int | None
    fixed_effects_cope_number: str | None
    output_dir: str | None
    cope_path: str | None
    varcope_path: str | None
    mask_path: str | None
    completion_source: str
    complete: bool
    status: str
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    missing_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MissingCompletedFfxOutputRow:
    """A missing planned subject-level FFX output needed by the handoff."""

    source_job_id: str | None
    path_kind: str
    path: str | None
    subject_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    contrast_id: str | None = None
    status: str = "missing"
    severity: str = "error"
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class LosoRoiHandoffSourceSpec:
    """Summary of the source documents used for one handoff plan."""

    localizer_plan_source: str
    completion_check: str
    roi_set_reference: str | None = None
    roi_set_name: str | None = None
    generated_plan_object: bool = False
    existing_roi_build_preview: bool = False
    fields: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class LosoRoiHandoffRow:
    """One completed localizer FFX bundle offered to the LOSO ROI workflow."""

    source_job_id: str
    subject_id: str | None
    session_id: str | None
    task_id: str | None
    model: str | None
    contrast_id: str | None
    output_dir: str | None
    cope_path: str | None
    varcope_path: str | None
    mask_path: str | None
    status: str
    matched_fixed_effects_input: bool = False
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class LosoFixedEffectsInputMappingRow:
    """Existing LOSO fixed-effects input row matched against FFX outputs."""

    subject_id: str | None
    subject_dir: str | None
    session_id: str | None
    task_id: str | None
    model: str | None
    contrast_id: str | None
    cope_number: str | None
    cope_dir: str | None
    cope_path: str | None
    varcope_path: str | None
    mask_path: str | None
    matched_source_job_id: str | None
    status: str
    complete: bool
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    missing_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiSetReferenceDocumentRow:
    """Preserved ROI set reference or document summary."""

    roi_set_reference: str | None
    roi_set_name: str | None
    mode: str
    preserved: bool
    status: str
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiCatalogMetadataRow:
    """ROI catalog fields copied only as grouping/reporting metadata."""

    roi_set_name: str | None
    roi_label: str | None
    roi_family: str | None
    roi_group: str | None = None
    roi_desc: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "metadata_only"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ContrastAliasHandoffRow:
    """Contrast alias copied for handoff reporting and matching diagnostics."""

    source_contrast_id: str | None
    target_contrast_id: str | None
    alias: str | None = None
    desc: str | None = None
    source: str = "metadata"
    status: str = "metadata_only"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ExistingRoiBuildPreviewRow:
    """A JSON-safe preview row returned from the existing ROI build planner."""

    roi_set_name: str | None
    action_index: int
    roi_label: str | None
    roi_family: str | None
    backend: str | None
    mask_path: str | None
    sidecar_path: str | None
    status: str
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class CommandPackagePlanPreviewRow:
    """Plan-only package or existing command lifecycle preview."""

    preview_kind: str
    name: str
    executed: bool = False
    status: str = "planned"
    command: tuple[str, ...] = ()
    package_api: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ProvenanceRootRefSummaryRow:
    """Root-ref provenance used while validating the handoff."""

    root_ref: str
    role: str
    path: str | None
    status: str
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class LosoRoiHandoffPlan:
    """Top-level JSON-safe, plan-only LOSO ROI handoff preview."""

    status: str
    source_spec: LosoRoiHandoffSourceSpec
    completed_ffx_output_rows: tuple[CompletedSubjectFfxOutputInventoryRow, ...] = ()
    missing_ffx_output_rows: tuple[MissingCompletedFfxOutputRow, ...] = ()
    handoff_rows: tuple[LosoRoiHandoffRow, ...] = ()
    fixed_effects_input_mapping_rows: tuple[LosoFixedEffectsInputMappingRow, ...] = ()
    roi_set_rows: tuple[RoiSetReferenceDocumentRow, ...] = ()
    roi_catalog_rows: tuple[RoiCatalogMetadataRow, ...] = ()
    contrast_alias_rows: tuple[ContrastAliasHandoffRow, ...] = ()
    existing_roi_build_preview_rows: tuple[ExistingRoiBuildPreviewRow, ...] = ()
    command_preview_rows: tuple[CommandPackagePlanPreviewRow, ...] = ()
    package_plan_preview_rows: tuple[CommandPackagePlanPreviewRow, ...] = ()
    provenance_root_rows: tuple[ProvenanceRootRefSummaryRow, ...] = ()
    generated_roi_set_plan_object: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    executed: bool = False
    plan_only: bool = True

    @property
    def valid(self) -> bool:
        return self.status != "error"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        return payload


def inventory_completed_subject_ffx_outputs(
    localizer_ffx_plan: LocalizerFixedEffectsPlan | Mapping[str, Any] | Any,
    *,
    execution_result: Mapping[str, Any] | Any | None = None,
    check_filesystem: bool = True,
) -> tuple[CompletedSubjectFfxOutputInventoryRow, ...]:
    """Normalize planned subject-level FFX outputs into completion rows."""

    plan_payload = _mapping_payload(localizer_ffx_plan)
    if plan_payload is None:
        return ()

    result_payload = _mapping_payload(execution_result) if execution_result is not None else None
    output_rows = _mapping_rows(plan_payload.get("output_path_rows"))
    outputs_by_job = _output_rows_by_job(output_rows)
    result_state = _execution_result_state(result_payload)
    inventory: list[CompletedSubjectFfxOutputInventoryRow] = []

    for job in _mapping_rows(plan_payload.get("ffx_job_rows")):
        job_id = _optional_text(job.get("job_id")) or ""
        rows_by_kind = outputs_by_job.get(job_id, {})
        paths = {kind: _optional_text(rows_by_kind.get(kind, {}).get("path")) for kind in _REQUIRED_FFX_OUTPUT_KINDS}
        output_dir = _optional_text(job.get("output_dir"))
        warnings: list[str] = []
        errors: list[str] = []
        missing_paths: list[str] = []
        completion_source = "filesystem" if result_payload is None else "execution_result+filesystem"

        missing_declared = [kind for kind, path in paths.items() if path is None]
        for kind in missing_declared:
            errors.append(f"Missing planned {kind} path for localizer FFX job {job_id!r}.")
            missing_paths.append(kind)

        filesystem_present = _path_presence(paths, enabled=bool(check_filesystem or result_payload is not None))
        for kind, present in filesystem_present.items():
            if not present:
                path = paths.get(kind)
                missing_paths.append(path or kind)
                errors.append(f"Planned {kind} does not exist: {path}.")

        if result_payload is not None:
            if job_id not in result_state["completed_job_ids"]:
                errors.append(f"Execution result does not mark localizer FFX job {job_id!r} completed.")
            for kind in _REQUIRED_FFX_OUTPUT_KINDS:
                status = result_state["expected_status_by_job_kind"].get((job_id, kind))
                if status != "present":
                    path = paths.get(kind)
                    missing_paths.append(path or kind)
                    errors.append(f"Execution result does not report {kind} present for localizer FFX job {job_id!r}.")

        complete = not errors and all(paths.values())
        if not check_filesystem and result_payload is None:
            complete = False
            warnings.append("Filesystem completion checks were disabled; planned outputs were not marked complete.")
            completion_source = "planned_unchecked"

        inventory.append(
            CompletedSubjectFfxOutputInventoryRow(
                source_job_id=job_id,
                source_name=_optional_text(job.get("source_name") or job.get("source")),
                subject_id=_optional_text(job.get("subject_id") or job.get("subject")),
                session_id=_optional_text(job.get("session_id") or job.get("session")),
                task_id=_optional_text(job.get("task_id") or job.get("task")),
                run_ids=tuple(_string_sequence(job.get("run_ids"))),
                model=_optional_text(job.get("model") or job.get("model_id")),
                contrast_id=_optional_text(job.get("contrast_id") or job.get("contrast")),
                contrast_name=_optional_text(job.get("contrast_name")),
                contrast_number=_optional_int(job.get("contrast_number")),
                fixed_effects_cope_number=_cope_number_from_path(paths.get("output_cope")),
                output_dir=output_dir,
                cope_path=paths.get("output_cope"),
                varcope_path=paths.get("output_varcope"),
                mask_path=paths.get("output_mask"),
                completion_source=completion_source,
                complete=complete,
                status="complete" if complete else "missing",
                warnings=tuple(_dedupe(warnings)),
                errors=tuple(_dedupe(errors)),
                missing_paths=tuple(_dedupe(missing_paths)),
            )
        )
    return tuple(inventory)


def map_ffx_outputs_to_loso_fixed_effects_inputs(
    completed_outputs: Sequence[CompletedSubjectFfxOutputInventoryRow | Mapping[str, Any]],
    *,
    roi_set_document: Mapping[str, Any] | None = None,
    context: RoiExecutionContext | None = None,
    contrast_aliases: Sequence[ContrastAliasHandoffRow | Mapping[str, Any]] = (),
    validate_personal_paths: bool = True,
) -> tuple[LosoFixedEffectsInputMappingRow, ...]:
    """Map completed FFX outputs to the existing LOSO fixed-effects input model."""

    completed_rows = tuple(_completed_output_payload(row) for row in completed_outputs)
    complete_rows = tuple(row for row in completed_rows if row.get("complete") is True)
    if roi_set_document is None or context is None:
        return tuple(_mapping_row_from_completed_output(row) for row in complete_rows)

    aliases = tuple(_contrast_alias_payload(row) for row in contrast_aliases)
    exact_index = {
        _path_triple(row.get("cope_path"), row.get("varcope_path"), row.get("mask_path")): row
        for row in complete_rows
        if _path_triple(row.get("cope_path"), row.get("varcope_path"), row.get("mask_path")) is not None
    }
    entity_index: dict[tuple[str | None, str | None, str | None, str | None], list[Mapping[str, Any]]] = {}
    for row in complete_rows:
        key = (
            _normalized_subject(row.get("subject_id")),
            _optional_text(row.get("session_id")),
            _optional_text(row.get("task_id")),
            _optional_text(row.get("contrast_id")),
        )
        entity_index.setdefault(key, []).append(row)

    mapped: list[LosoFixedEffectsInputMappingRow] = []
    for item in roi_loso.discover_subject_fixed_effects_inputs(
        roi_set_document,
        context=context,
        validate_personal_paths=validate_personal_paths,
    ):
        payload = item.to_dict()
        expected = _path_triple(payload.get("cope_path"), payload.get("varcope_path"), payload.get("mask_path"))
        matched = exact_index.get(expected)
        warnings: list[str] = []
        errors: list[str] = []
        missing_paths = tuple(
            _optional_text(payload.get(f"{kind}_path")) or str(kind)
            for kind in payload.get("missing", ())
            if kind
        )
        status = "matched"

        if missing_paths:
            status = "missing_expected_input"
            errors.append("Existing LOSO fixed-effects input discovery reports missing paths.")
        elif matched is None:
            candidates = _entity_candidates(entity_index, payload, aliases)
            if candidates:
                status = "path_mismatch"
                errors.append("Completed localizer FFX output exists for the same entity but paths do not match the ROI set expectations.")
            else:
                status = "unmatched"
                errors.append("No completed localizer FFX output matched the ROI set fixed-effects input expectation.")
        else:
            matched_contrast = _optional_text(matched.get("contrast_id"))
            expected_contrast = _optional_text(payload.get("contrast_id"))
            if matched_contrast and expected_contrast and not _contrast_equivalent(matched_contrast, expected_contrast, aliases):
                warnings.append("Matched by path, but source and ROI contrast identifiers differ.")

        mapped.append(
            LosoFixedEffectsInputMappingRow(
                subject_id=_optional_text(payload.get("subject_id")),
                subject_dir=_optional_text(payload.get("subject_dir")),
                session_id=_optional_text(payload.get("session_id")),
                task_id=_optional_text(payload.get("task_id")),
                model=_optional_text(payload.get("model")),
                contrast_id=_optional_text(payload.get("contrast_id")),
                cope_number=_optional_text(payload.get("cope_number")),
                cope_dir=_optional_text(payload.get("cope_dir")),
                cope_path=_optional_text(payload.get("cope_path")),
                varcope_path=_optional_text(payload.get("varcope_path")),
                mask_path=_optional_text(payload.get("mask_path")),
                matched_source_job_id=_optional_text(matched.get("source_job_id")) if matched is not None else None,
                status=status,
                complete=bool(payload.get("complete")) and matched is not None,
                warnings=tuple(_dedupe(warnings)),
                errors=tuple(_dedupe(errors)),
                missing_paths=missing_paths,
            )
        )
    return tuple(mapped)


def generate_loso_roi_set_plan_object(
    roi_set_document: Mapping[str, Any],
    *,
    fixed_effects_inputs: Mapping[str, Any] | None = None,
    validate_personal_paths: bool = False,
) -> dict[str, Any]:
    """Return an in-memory ROI-set-compatible planning document.

    The returned object is a deep copy of caller-supplied ROI semantics. The
    function only inserts caller-supplied ``fixed_effects_inputs`` when present;
    it never invents ROI definitions, seeds, thresholds, masks, or LOSO
    settings.
    """

    if not isinstance(roi_set_document, Mapping):
        raise ValueError("roi_set_document must contain a mapping.")
    generated = copy.deepcopy(dict(roi_set_document))
    raw_roi_set = generated.get("roi_set")
    if not isinstance(raw_roi_set, Mapping):
        raise ValueError("ROI set document must contain roi_set.")
    roi_set = dict(raw_roi_set)
    generated["roi_set"] = roi_set
    if fixed_effects_inputs is not None:
        roi_set["fixed_effects_inputs"] = copy.deepcopy(dict(fixed_effects_inputs))
    roi_loso.load_loso_roi_set_config(generated, validate_personal_paths=validate_personal_paths)
    return _json_safe(generated)


def plan_loso_roi_handoff(
    document: Mapping[str, Any] | Any | None = None,
    *,
    handoff_config: Mapping[str, Any] | None = None,
    roi_set_reference: str | Path | None = None,
    roi_set_document: Mapping[str, Any] | None = None,
    localizer_ffx_plan: LocalizerFixedEffectsPlan | Mapping[str, Any] | Any | None = None,
    localizer_ffx_execution_result: Mapping[str, Any] | Any | None = None,
    roots: Mapping[str, str | Path | Mapping[str, Any]] | None = None,
    context: RoiExecutionContext | None = None,
    check_filesystem: bool = True,
    preview_existing_roi_build: bool | None = None,
    generate_plan_object: bool | None = None,
    validate_personal_paths: bool = True,
) -> LosoRoiHandoffPlan:
    """Return a JSON-safe, no-write handoff plan for existing LOSO ROI builds."""

    document_payload = document if isinstance(document, Mapping) else {}
    config = _handoff_payload(document_payload)
    if handoff_config is not None:
        config = {**dict(config), **dict(handoff_config)}

    plan = localizer_ffx_plan or config.get("localizer_fixed_effects_plan") or config.get("ffx_plan")
    execution_result = (
        localizer_ffx_execution_result
        if localizer_ffx_execution_result is not None
        else config.get("localizer_fixed_effects_execution_result")
        or config.get("ffx_execution_result")
    )
    roi_doc = roi_set_document or _mapping_or_none(config.get("roi_set_document") or config.get("roi_set"))
    roi_reference = _optional_text(roi_set_reference) or _optional_text(
        config.get("roi_set_reference") or config.get("roi_set_ref") or config.get("roi_set_path")
    )
    preview_build = _optional_bool(config.get("preview_existing_roi_build"))
    if preview_existing_roi_build is not None:
        preview_build = preview_existing_roi_build
    if preview_build is None:
        preview_build = roi_doc is not None
    generate_object = _optional_bool(config.get("generate_plan_object"))
    if generate_plan_object is not None:
        generate_object = generate_plan_object
    if generate_object is None:
        generate_object = False

    warnings: list[str] = []
    errors: list[str] = []
    if plan is None:
        errors.append("A LocalizerFixedEffectsPlan object or compatible mapping is required.")

    completed_rows = inventory_completed_subject_ffx_outputs(
        plan,
        execution_result=execution_result,
        check_filesystem=check_filesystem,
    ) if plan is not None else ()
    missing_rows = _missing_rows_from_inventory(completed_rows)
    errors.extend(row.message for row in missing_rows if row.severity == "error" and row.message)
    warnings.extend(row.message for row in missing_rows if row.severity == "warning" and row.message)

    contrast_rows = _collect_contrast_alias_rows(config, _mapping_payload(plan), roi_doc)
    roi_set_rows, roi_catalog_rows = _roi_set_summary_rows(roi_doc, roi_reference)
    errors.extend(error for row in roi_set_rows for error in row.errors)
    warnings.extend(warning for row in roi_set_rows for warning in row.warnings)

    generated_doc: dict[str, Any] | None = None
    planning_doc = roi_doc
    if generate_object:
        try:
            if roi_doc is None:
                raise ValueError("Generated plan-object mode requires an existing ROI set document.")
            generated_doc = generate_loso_roi_set_plan_object(
                roi_doc,
                fixed_effects_inputs=_mapping_or_none(config.get("fixed_effects_inputs")),
                validate_personal_paths=validate_personal_paths,
            )
            planning_doc = generated_doc
        except ValueError as exc:
            errors.append(str(exc))

    roi_context = context or _context_from_roots(roots)
    mapping_rows: tuple[LosoFixedEffectsInputMappingRow, ...] = ()
    build_preview_rows: tuple[ExistingRoiBuildPreviewRow, ...] = ()
    package_rows: list[CommandPackagePlanPreviewRow] = []
    if planning_doc is not None:
        try:
            mapping_rows = map_ffx_outputs_to_loso_fixed_effects_inputs(
                completed_rows,
                roi_set_document=planning_doc,
                context=roi_context,
                contrast_aliases=contrast_rows,
                validate_personal_paths=validate_personal_paths,
            )
            errors.extend(error for row in mapping_rows for error in row.errors)
            warnings.extend(warning for row in mapping_rows for warning in row.warnings)
        except ValueError as exc:
            errors.append(str(exc))

        if preview_build:
            try:
                build_plan = roi_execution.plan_roi_build(
                    planning_doc,
                    context=roi_context,
                    validate_personal_paths=validate_personal_paths,
                )
                build_preview_rows = _roi_build_preview_rows(build_plan)
                package_rows.append(
                    CommandPackagePlanPreviewRow(
                        preview_kind="package_api",
                        name="existing_roi_build_plan",
                        executed=False,
                        status="planned",
                        package_api="research_platform.neuro.roi_execution.plan_roi_build",
                    )
                )
            except ValueError as exc:
                errors.append(str(exc))
                package_rows.append(
                    CommandPackagePlanPreviewRow(
                        preview_kind="package_api",
                        name="existing_roi_build_plan",
                        executed=False,
                        status="error",
                        package_api="research_platform.neuro.roi_execution.plan_roi_build",
                        errors=(str(exc),),
                    )
                )
    elif roi_reference is not None:
        warnings.append("ROI set reference was preserved, but no ROI set document was supplied for fixed-effects input validation.")

    matched_job_ids = {row.matched_source_job_id for row in mapping_rows if row.matched_source_job_id is not None}
    handoff_rows = tuple(_handoff_row(row, matched=row.source_job_id in matched_job_ids) for row in completed_rows if row.complete)
    root_rows = _root_ref_rows(_mapping_payload(plan), roots, roi_context)

    source_spec = LosoRoiHandoffSourceSpec(
        localizer_plan_source="provided_object" if plan is not None else "missing",
        completion_check="execution_result+filesystem" if execution_result is not None else "filesystem",
        roi_set_reference=roi_reference,
        roi_set_name=_roi_set_name(planning_doc),
        generated_plan_object=generated_doc is not None,
        existing_roi_build_preview=bool(build_preview_rows),
        fields=_metadata_only(config),
    )

    warnings.extend(warning for row in completed_rows for warning in row.warnings)
    errors.extend(error for row in completed_rows for error in row.errors)
    flat_warnings = _dedupe(_flatten_messages(warnings))
    flat_errors = _dedupe(_flatten_messages(errors))
    status = "error" if flat_errors else "warning" if flat_warnings else "ok"
    return LosoRoiHandoffPlan(
        status=status,
        source_spec=source_spec,
        completed_ffx_output_rows=completed_rows,
        missing_ffx_output_rows=missing_rows,
        handoff_rows=handoff_rows,
        fixed_effects_input_mapping_rows=mapping_rows,
        roi_set_rows=roi_set_rows,
        roi_catalog_rows=roi_catalog_rows,
        contrast_alias_rows=contrast_rows,
        existing_roi_build_preview_rows=build_preview_rows,
        command_preview_rows=(),
        package_plan_preview_rows=tuple(package_rows),
        provenance_root_rows=root_rows,
        generated_roi_set_plan_object=generated_doc,
        warnings=flat_warnings,
        errors=flat_errors,
        executed=False,
        plan_only=True,
    )


def _handoff_payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    workflow = document.get("analysis_workflow")
    if isinstance(workflow, Mapping):
        extensions = workflow.get("extensions")
        if isinstance(extensions, Mapping):
            mvpa = extensions.get("mvpa")
            if isinstance(mvpa, Mapping):
                for key in ("loso_roi_handoff", "roi_handoff", "step5_handoff"):
                    if isinstance(mvpa.get(key), Mapping):
                        return mvpa[key]  # type: ignore[return-value]
        if isinstance(workflow.get("mvpa"), Mapping):
            mvpa = workflow["mvpa"]  # type: ignore[index]
            if isinstance(mvpa, Mapping):
                for key in ("loso_roi_handoff", "roi_handoff", "step5_handoff"):
                    if isinstance(mvpa.get(key), Mapping):
                        return mvpa[key]  # type: ignore[return-value]
    for key in ("loso_roi_handoff", "roi_handoff", "step5_handoff"):
        if isinstance(document.get(key), Mapping):
            return document[key]  # type: ignore[return-value]
    return document if _looks_like_handoff_config(document) else {}


def _looks_like_handoff_config(document: Mapping[str, Any]) -> bool:
    return any(
        key in document
        for key in (
            "roi_set_document",
            "roi_set_reference",
            "localizer_fixed_effects_plan",
            "fixed_effects_inputs",
            "generate_plan_object",
        )
    )


def _mapping_payload(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, LocalizerFixedEffectsPlan):
        return value.to_dict()
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    return None


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _mapping_rows(raw: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    rows: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            rows.append(dict(item))
            continue
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return tuple(rows)


def _output_rows_by_job(output_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in output_rows:
        job_id = _optional_text(row.get("job_id"))
        path_kind = _optional_text(row.get("path_kind"))
        if job_id is None or path_kind is None:
            continue
        grouped.setdefault(job_id, {})[path_kind] = row
    return grouped


def _execution_result_state(result: Mapping[str, Any] | None) -> dict[str, Any]:
    completed = {
        str(row["job_id"])
        for row in _mapping_rows(result.get("completed_job_rows") if result is not None else None)
        if _optional_text(row.get("job_id")) is not None and _optional_text(row.get("status")) == "completed"
    }
    status_by_kind: dict[tuple[str, str], str] = {}
    if result is not None:
        for row in _mapping_rows(result.get("expected_output_check_rows")):
            job_id = _optional_text(row.get("job_id"))
            kind = _optional_text(row.get("path_kind"))
            status = _optional_text(row.get("status"))
            if job_id is not None and kind is not None and status is not None:
                status_by_kind[(job_id, kind)] = status
        for completed_row in _mapping_rows(result.get("completed_job_rows")):
            for row in _mapping_rows(completed_row.get("expected_outputs")):
                job_id = _optional_text(row.get("job_id") or completed_row.get("job_id"))
                kind = _optional_text(row.get("path_kind"))
                status = _optional_text(row.get("status"))
                if job_id is not None and kind is not None and status is not None:
                    status_by_kind[(job_id, kind)] = status
    return {"completed_job_ids": completed, "expected_status_by_job_kind": status_by_kind}


def _path_presence(paths: Mapping[str, str | None], *, enabled: bool) -> dict[str, bool]:
    if not enabled:
        return {}
    return {kind: bool(path and Path(path).is_file()) for kind, path in paths.items() if path is not None}


def _cope_number_from_path(path: str | None) -> str | None:
    if path is None:
        return None
    name = Path(path).name
    if name.startswith("cope") and name.endswith(".nii.gz"):
        number = name[len("cope") : -len(".nii.gz")]
        return number if number else None
    if name.startswith("cope") and name.endswith(".nii"):
        number = name[len("cope") : -len(".nii")]
        return number if number else None
    return None


def _missing_rows_from_inventory(
    inventory: Sequence[CompletedSubjectFfxOutputInventoryRow],
) -> tuple[MissingCompletedFfxOutputRow, ...]:
    rows: list[MissingCompletedFfxOutputRow] = []
    for item in inventory:
        if item.complete:
            continue
        for missing in item.missing_paths:
            rows.append(
                MissingCompletedFfxOutputRow(
                    source_job_id=item.source_job_id,
                    path_kind=_path_kind_for_missing(item, missing),
                    path=missing if "/" in missing or missing.endswith((".nii", ".nii.gz")) else None,
                    subject_id=item.subject_id,
                    session_id=item.session_id,
                    task_id=item.task_id,
                    contrast_id=item.contrast_id,
                    message=f"Localizer FFX output is not complete for job {item.source_job_id!r}: {missing}.",
                )
            )
    return tuple(rows)


def _path_kind_for_missing(item: CompletedSubjectFfxOutputInventoryRow, missing: str) -> str:
    for kind, path in (
        ("output_cope", item.cope_path),
        ("output_varcope", item.varcope_path),
        ("output_mask", item.mask_path),
    ):
        if missing == kind or missing == path:
            return kind
    return "output"


def _completed_output_payload(value: CompletedSubjectFfxOutputInventoryRow | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, CompletedSubjectFfxOutputInventoryRow):
        return value.to_dict()
    return value


def _contrast_alias_payload(value: ContrastAliasHandoffRow | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, ContrastAliasHandoffRow):
        return value.to_dict()
    return value


def _path_triple(cope_path: Any, varcope_path: Any, mask_path: Any) -> tuple[str, str, str] | None:
    paths = (_optional_text(cope_path), _optional_text(varcope_path), _optional_text(mask_path))
    if any(path is None for path in paths):
        return None
    return tuple(str(Path(str(path)).resolve()) for path in paths if path is not None)  # type: ignore[return-value]


def _mapping_row_from_completed_output(row: Mapping[str, Any]) -> LosoFixedEffectsInputMappingRow:
    return LosoFixedEffectsInputMappingRow(
        subject_id=_optional_text(row.get("subject_id")),
        subject_dir=None,
        session_id=_optional_text(row.get("session_id")),
        task_id=_optional_text(row.get("task_id")),
        model=_optional_text(row.get("model")),
        contrast_id=_optional_text(row.get("contrast_id")),
        cope_number=_optional_text(row.get("fixed_effects_cope_number")),
        cope_dir=_optional_text(row.get("output_dir")),
        cope_path=_optional_text(row.get("cope_path")),
        varcope_path=_optional_text(row.get("varcope_path")),
        mask_path=_optional_text(row.get("mask_path")),
        matched_source_job_id=_optional_text(row.get("source_job_id")),
        status="available_without_roi_set_validation",
        complete=True,
    )


def _entity_candidates(
    entity_index: Mapping[tuple[str | None, str | None, str | None, str | None], Sequence[Mapping[str, Any]]],
    payload: Mapping[str, Any],
    aliases: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    subject = _normalized_subject(payload.get("subject_id"))
    session = _optional_text(payload.get("session_id"))
    task = _optional_text(payload.get("task_id"))
    contrast = _optional_text(payload.get("contrast_id"))
    candidates: list[Mapping[str, Any]] = []
    for key, rows in entity_index.items():
        row_subject, row_session, row_task, row_contrast = key
        if row_subject != subject or row_session != session or row_task != task:
            continue
        if contrast is None or row_contrast is None or _contrast_equivalent(row_contrast, contrast, aliases):
            candidates.extend(rows)
    return tuple(candidates)


def _contrast_equivalent(left: str, right: str, aliases: Sequence[Mapping[str, Any]]) -> bool:
    if left == right:
        return True
    for row in aliases:
        source = _optional_text(row.get("source_contrast_id"))
        target = _optional_text(row.get("target_contrast_id"))
        alias = _optional_text(row.get("alias"))
        values = {value for value in (source, target, alias) if value is not None}
        if left in values and right in values:
            return True
    return False


def _collect_contrast_alias_rows(
    config: Mapping[str, Any],
    plan_payload: Mapping[str, Any] | None,
    roi_doc: Mapping[str, Any] | None,
) -> tuple[ContrastAliasHandoffRow, ...]:
    rows: list[ContrastAliasHandoffRow] = []
    for row in _mapping_rows(config.get("contrast_aliases") or config.get("contrast_handoffs") or config.get("contrast_mapping")):
        source = _optional_text(row.get("source_contrast_id") or row.get("source") or row.get("from"))
        target = _optional_text(row.get("target_contrast_id") or row.get("target") or row.get("to") or row.get("contrast_id"))
        aliases = _string_sequence(row.get("aliases") or row.get("alias"))
        if aliases:
            for alias in aliases:
                rows.append(
                    ContrastAliasHandoffRow(
                        source_contrast_id=source,
                        target_contrast_id=target,
                        alias=alias,
                        desc=_optional_text(row.get("desc")),
                        source="handoff_config",
                    )
                )
        else:
            rows.append(
                ContrastAliasHandoffRow(
                    source_contrast_id=source,
                    target_contrast_id=target,
                    alias=None,
                    desc=_optional_text(row.get("desc")),
                    source="handoff_config",
                )
            )

    if plan_payload is not None:
        for row in _mapping_rows(plan_payload.get("contrast_aliases")):
            contrast_id = _optional_text(row.get("contrast_id") or row.get("id"))
            for alias in _string_sequence(row.get("aliases")):
                rows.append(
                    ContrastAliasHandoffRow(
                        source_contrast_id=contrast_id,
                        target_contrast_id=contrast_id,
                        alias=alias,
                        desc=_optional_text(row.get("contrast_name")),
                        source="localizer_fixed_effects_plan",
                    )
                )

    if roi_doc is not None:
        roi_set = _roi_set_payload(roi_doc, required=False)
        for row in _mapping_rows(roi_set.get("contrasts") if isinstance(roi_set, Mapping) else None):
            contrast_id = _optional_text(row.get("id") or row.get("name") or row.get("contrast_id") or row.get("contrast"))
            rows.append(
                ContrastAliasHandoffRow(
                    source_contrast_id=contrast_id,
                    target_contrast_id=contrast_id,
                    alias=_optional_text(row.get("desc")),
                    desc=_optional_text(row.get("desc")),
                    source="roi_set_document",
                )
            )
        aliases = roi_set.get("contrast_aliases") if isinstance(roi_set, Mapping) else None
        if isinstance(aliases, Mapping):
            for key, value in aliases.items():
                for alias in _string_sequence(value):
                    rows.append(
                        ContrastAliasHandoffRow(
                            source_contrast_id=str(key),
                            target_contrast_id=str(key),
                            alias=alias,
                            source="roi_set_document",
                        )
                    )
    return tuple(rows)


def _roi_set_summary_rows(
    roi_doc: Mapping[str, Any] | None,
    roi_reference: str | None,
) -> tuple[tuple[RoiSetReferenceDocumentRow, ...], tuple[RoiCatalogMetadataRow, ...]]:
    if roi_doc is None:
        if roi_reference is None:
            return (), ()
        return (
            (
                RoiSetReferenceDocumentRow(
                    roi_set_reference=roi_reference,
                    roi_set_name=None,
                    mode="reference_only",
                    preserved=True,
                    status="warning",
                    warnings=("No ROI set document was supplied; only the reference was preserved.",),
                ),
            ),
            (),
        )

    try:
        roi_set = roi_loso.load_loso_roi_set_config(roi_doc, validate_personal_paths=False)
    except ValueError as exc:
        return (
            (
                RoiSetReferenceDocumentRow(
                    roi_set_reference=roi_reference,
                    roi_set_name=_roi_set_name(roi_doc),
                    mode="existing_document",
                    preserved=True,
                    status="error",
                    errors=(str(exc),),
                ),
            ),
            (),
        )

    catalog_rows = []
    for roi in roi_set.rois:
        fields_payload = dict(roi.fields)
        metadata = {
            str(key): value
            for key, value in fields_payload.items()
            if key in _CATALOG_METADATA_KEYS and key not in _SEMANTIC_ROI_KEYS
        }
        catalog_rows.append(
            RoiCatalogMetadataRow(
                roi_set_name=roi_set.name,
                roi_label=roi.label,
                roi_family=roi.family,
                roi_group=_optional_text(fields_payload.get("group") or fields_payload.get("roi_group")),
                roi_desc=roi.desc,
                metadata=_json_safe(metadata),
            )
        )
    return (
        (
            RoiSetReferenceDocumentRow(
                roi_set_reference=roi_reference,
                roi_set_name=roi_set.name,
                mode="existing_document",
                preserved=True,
                status="ok",
            ),
        ),
        tuple(catalog_rows),
    )


def _roi_set_payload(document: Mapping[str, Any], *, required: bool = True) -> dict[str, Any]:
    payload = document.get("roi_set") if isinstance(document, Mapping) else None
    if isinstance(payload, Mapping):
        return dict(payload)
    if required:
        raise ValueError("ROI set document must contain roi_set.")
    return {}


def _roi_set_name(document: Mapping[str, Any] | None) -> str | None:
    if document is None:
        return None
    roi_set = document.get("roi_set")
    if isinstance(roi_set, Mapping):
        return _optional_text(roi_set.get("name"))
    return None


def _context_from_roots(roots: Mapping[str, str | Path | Mapping[str, Any]] | None) -> RoiExecutionContext:
    root_refs = _root_ref_paths(roots)
    workspace = root_refs.get("workspace_root") or Path.cwd()
    project = root_refs.get("project_root") or workspace
    artifacts = root_refs.get("artifacts_root") or root_refs.get("artifact_root") or workspace
    return RoiExecutionContext(
        workspace_root=workspace,
        project_root=project,
        artifacts_root=artifacts,
        project_name=None,
        root_refs=root_refs,
    )


def _root_ref_paths(roots: Mapping[str, str | Path | Mapping[str, Any]] | None) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    if roots is None:
        return resolved
    for name, raw in roots.items():
        value: Any = raw
        if isinstance(raw, Mapping):
            value = raw.get("path", raw.get("root"))
        if value is None:
            continue
        resolved[str(name)] = Path(value).expanduser().resolve()
    return resolved


def _roi_build_preview_rows(build_plan: Any) -> tuple[ExistingRoiBuildPreviewRow, ...]:
    actions = tuple(getattr(build_plan, "actions", ()) or ())
    roi_set_name = _optional_text(getattr(build_plan, "roi_set_name", None))
    rows: list[ExistingRoiBuildPreviewRow] = []
    for index, action in enumerate(actions):
        metadata = getattr(action, "metadata", {})
        warnings = _action_warnings(metadata)
        rows.append(
            ExistingRoiBuildPreviewRow(
                roi_set_name=roi_set_name,
                action_index=index,
                roi_label=_optional_text(getattr(action, "roi_label", None)),
                roi_family=_optional_text(getattr(action, "family", None)),
                backend=_optional_text(getattr(action, "backend", None)),
                mask_path=_optional_text(getattr(action, "mask_path", None)),
                sidecar_path=_optional_text(getattr(action, "sidecar_path", None)),
                status="planned",
                warnings=warnings,
                metadata=_json_safe(metadata),
            )
        )
    return tuple(rows)


def _action_warnings(metadata: Any) -> tuple[str, ...]:
    if not isinstance(metadata, Mapping):
        return ()
    job = metadata.get("loso_group_job")
    if isinstance(job, Mapping):
        return tuple(_string_sequence(job.get("warnings")))
    return ()


def _handoff_row(row: CompletedSubjectFfxOutputInventoryRow, *, matched: bool) -> LosoRoiHandoffRow:
    return LosoRoiHandoffRow(
        source_job_id=row.source_job_id,
        subject_id=row.subject_id,
        session_id=row.session_id,
        task_id=row.task_id,
        model=row.model,
        contrast_id=row.contrast_id,
        output_dir=row.output_dir,
        cope_path=row.cope_path,
        varcope_path=row.varcope_path,
        mask_path=row.mask_path,
        status="matched" if matched else "available",
        matched_fixed_effects_input=matched,
        warnings=row.warnings,
        errors=row.errors,
    )


def _root_ref_rows(
    plan_payload: Mapping[str, Any] | None,
    roots: Mapping[str, str | Path | Mapping[str, Any]] | None,
    context: RoiExecutionContext,
) -> tuple[ProvenanceRootRefSummaryRow, ...]:
    rows: list[ProvenanceRootRefSummaryRow] = []
    if plan_payload is not None:
        for row in _mapping_rows(plan_payload.get("root_ref_rows")):
            root_ref = _optional_text(row.get("root_ref"))
            if root_ref is None:
                continue
            rows.append(
                ProvenanceRootRefSummaryRow(
                    root_ref=root_ref,
                    role=_optional_text(row.get("role")) or "localizer_fixed_effects",
                    path=_optional_text(row.get("path")),
                    status=_optional_text(row.get("status")) or "unknown",
                    message=_optional_text(row.get("message")),
                )
            )
    supplied_roots = _root_ref_paths(roots)
    for name, path in sorted({**supplied_roots, **dict(context.root_refs)}.items()):
        rows.append(
            ProvenanceRootRefSummaryRow(
                root_ref=str(name),
                role="roi_handoff_context",
                path=str(path),
                status="resolved",
            )
        )
    return tuple(rows)


def _metadata_only(config: Mapping[str, Any]) -> Mapping[str, Any]:
    allowed = {
        str(key): value
        for key, value in config.items()
        if key
        in {
            "name",
            "id",
            "desc",
            "description",
            "workflow",
            "workflow_name",
            "roi_set_reference",
            "roi_set_ref",
            "tags",
        }
    }
    return _json_safe(allowed)


def _normalized_subject(value: Any) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    return text[4:] if text.startswith("sub-") else text


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


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


def _flatten_messages(values: Sequence[Any]) -> tuple[str, ...]:
    messages: list[str] = []
    for value in values:
        if isinstance(value, str):
            if value:
                messages.append(value)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            messages.extend(str(item) for item in value if str(item))
    return tuple(messages)


def _json_safe_dataclass(value: Any) -> dict[str, Any]:
    return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items() if child is not None}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child) for child in value]
    return str(value)


__all__ = [
    "CommandPackagePlanPreviewRow",
    "CompletedSubjectFfxOutputInventoryRow",
    "ContrastAliasHandoffRow",
    "ExistingRoiBuildPreviewRow",
    "LosoFixedEffectsInputMappingRow",
    "LosoRoiHandoffPlan",
    "LosoRoiHandoffRow",
    "LosoRoiHandoffSourceSpec",
    "MissingCompletedFfxOutputRow",
    "ProvenanceRootRefSummaryRow",
    "RoiCatalogMetadataRow",
    "RoiSetReferenceDocumentRow",
    "generate_loso_roi_set_plan_object",
    "inventory_completed_subject_ffx_outputs",
    "map_ffx_outputs_to_loso_fixed_effects_inputs",
    "plan_loso_roi_handoff",
]
