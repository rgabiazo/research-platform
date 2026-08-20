"""LOSO group-map ROI planning and local execution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any
import hashlib
import json

from research_platform.neuro._roi_path_safety import configured_path_is_unsafe
from research_platform.neuro.roi import (
    RoiDefinition,
    RoiSet,
    normalize_sidecar_provenance,
    parse_roi_set_document,
    provenance_path_reference,
    validate_portable_provenance_paths,
)


FIXED_EFFECTS_LAYOUT_FSL_GFEAT_NESTED = "fsl_gfeat_nested"
GROUP_MASK_STRATEGY_FIXED_EFFECTS_INTERSECTION = "fixed_effects_mask_intersection"
_FIXED_EFFECTS_INPUT_LAYOUTS = frozenset({FIXED_EFFECTS_LAYOUT_FSL_GFEAT_NESTED})
_GENERATED_GROUP_MASK_STRATEGY_ALIASES = {
    GROUP_MASK_STRATEGY_FIXED_EFFECTS_INTERSECTION: GROUP_MASK_STRATEGY_FIXED_EFFECTS_INTERSECTION,
    "generated_intersection": GROUP_MASK_STRATEGY_FIXED_EFFECTS_INTERSECTION,
    "generated_fixed_effects_mask_intersection": GROUP_MASK_STRATEGY_FIXED_EFFECTS_INTERSECTION,
}
_GENERATED_GROUP_MASK_SCOPES = frozenset(
    {
        "loso_training_subjects",
        "training_subjects",
        "included_subjects_intersection",
    }
)
_FSL_GFEAT_NESTED_GFEAT_DIR = (
    "{subject_dir}/{session_dir}/"
    "{subject_dir}_{session_dir}_task-{task_id}_dir-{direction}_desc-{model}.gfeat"
)


@dataclass(frozen=True)
class LosoContrastSpec:
    """Configured group-map contrast for LOSO ROI construction."""

    contrast_id: str
    cope_number: str
    desc: str
    fields: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contrast_id": self.contrast_id,
            "cope_number": self.cope_number,
            "desc": self.desc,
        }


@dataclass(frozen=True)
class FixedEffectsInput:
    """Subject-level fixed-effects inputs for one subject and contrast."""

    subject_id: str
    session_id: str | None
    task_id: str
    model: str
    contrast_id: str
    cope_number: str
    cope_dir: Path
    cope_path: Path
    varcope_path: Path
    mask_path: Path
    missing: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing

    @property
    def subject_dir(self) -> str:
        return f"sub-{self.subject_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject_dir": self.subject_dir,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "model": self.model,
            "contrast_id": self.contrast_id,
            "cope_number": self.cope_number,
            "cope_dir": str(self.cope_dir),
            "cope_path": str(self.cope_path),
            "varcope_path": str(self.varcope_path),
            "mask_path": str(self.mask_path),
            "missing": list(self.missing),
            "complete": self.complete,
        }


@dataclass(frozen=True)
class LosoGroupMaskInput:
    """Configured group-level mask input for one LOSO entity group."""

    session_id: str | None
    task_id: str
    model: str
    contrast_id: str
    cope_number: str
    mask_path: Path
    missing: tuple[str, ...]
    generated: bool = False
    strategy: str | None = None
    scope: str | None = None
    heldout_subject: str | None = None
    included_subjects: tuple[str, ...] = ()
    excluded_subjects: tuple[str, ...] = ()
    source_mask_paths: tuple[Path, ...] = ()
    sidecar_path: Path | None = None

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "model": self.model,
            "contrast_id": self.contrast_id,
            "cope_number": self.cope_number,
            "mask_path": str(self.mask_path),
            "missing": list(self.missing),
            "complete": self.complete,
            "generated": self.generated,
            "strategy": self.strategy,
            "scope": self.scope,
            "heldout_subject": self.heldout_subject,
            "included_subjects": list(self.included_subjects),
            "excluded_subjects": list(self.excluded_subjects),
            "source_mask_paths": [str(path) for path in self.source_mask_paths],
            "sidecar_path": str(self.sidecar_path) if self.sidecar_path is not None else None,
        }


@dataclass(frozen=True)
class GeneratedGroupMaskPlan:
    """Planned LOSO group mask generated from subject fixed-effects masks."""

    strategy: str
    scope: str
    mask_path: Path
    sidecar_path: Path
    source_mask_paths: tuple[Path, ...]
    included_subjects: tuple[str, ...]
    excluded_subjects: tuple[str, ...]
    heldout_subject: str | None
    contrast_id: str
    cope_number: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "scope": self.scope,
            "mask_path": str(self.mask_path),
            "sidecar_path": str(self.sidecar_path),
            "source_mask_paths": [str(path) for path in self.source_mask_paths],
            "included_subjects": list(self.included_subjects),
            "excluded_subjects": list(self.excluded_subjects),
            "heldout_subject": self.heldout_subject,
            "contrast_id": self.contrast_id,
            "cope_number": self.cope_number,
        }


@dataclass(frozen=True)
class LosoGroupMapJob:
    """A unique held-out-subject/contrast LOSO group-map job."""

    roi_set_name: str
    heldout_subject: str
    session_id: str | None
    task_id: str
    model: str
    contrast: LosoContrastSpec
    training_inputs: tuple[FixedEffectsInput, ...]
    heldout_input: FixedEffectsInput | None
    group_mask_path: Path
    zstat_path: Path
    sidecar_path: Path
    work_dir: Path
    output_root: Path
    input_root: Path
    cache_key: str
    cache_reuse: bool
    generated_group_mask: GeneratedGroupMaskPlan | None
    backend_config: Mapping[str, Any]
    design_config: Mapping[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_set": self.roi_set_name,
            "heldout_subject": self.heldout_subject,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "model": self.model,
            "contrast": self.contrast.to_dict(),
            "training_subjects": [input_spec.subject_id for input_spec in self.training_inputs],
            "training_inputs": [input_spec.to_dict() for input_spec in self.training_inputs],
            "heldout_input": self.heldout_input.to_dict() if self.heldout_input is not None else None,
            "group_mask_path": str(self.group_mask_path),
            "zstat_path": str(self.zstat_path),
            "sidecar_path": str(self.sidecar_path),
            "work_dir": str(self.work_dir),
            "output_root": str(self.output_root),
            "input_root": str(self.input_root),
            "cache_key": self.cache_key,
            "cache_reuse": self.cache_reuse,
            "generated_group_mask": self.generated_group_mask.to_dict() if self.generated_group_mask is not None else None,
            "backend_config": dict(self.backend_config),
            "design_config": dict(self.design_config),
            "warnings": list(self.warnings),
        }


def load_loso_roi_set_config(document: Mapping[str, Any], *, validate_personal_paths: bool = True) -> RoiSet:
    """Validate and parse a LOSO ROI set config."""

    roi_set = parse_roi_set_document(document, validate_personal_paths=validate_personal_paths)
    loso_rois = [roi for roi in roi_set.rois if roi.family == "loso_group_map"]
    if not loso_rois:
        raise ValueError("LOSO ROI config must contain at least one loso_group_map ROI.")
    for roi in loso_rois:
        backend = roi.backend or roi_set.backend
        if backend != "fsl_flame1":
            raise ValueError("Phase 4 LOSO ROI execution supports only backend fsl_flame1.")
    return roi_set


def discover_subject_fixed_effects_inputs(
    document: Mapping[str, Any],
    *,
    context: Any,
    session_id: str | None = None,
    task_id: str | None = None,
    model: str | None = None,
    contrast_id: str | None = None,
    validate_personal_paths: bool = True,
) -> tuple[FixedEffectsInput, ...]:
    """Discover configured subject fixed-effects inputs for one LOSO entity group."""

    roi_set = load_loso_roi_set_config(document, validate_personal_paths=validate_personal_paths)
    common = _roi_set_common_fields(roi_set)
    subjects = _subjects(common)
    sessions = _entity_values(session_id or common.get("session_id") or common.get("session"), label="session")
    tasks = _entity_values(task_id or common.get("task_id") or common.get("task"), label="task")
    models = _entity_values(model or common.get("model"), label="model")
    contrasts = _contrast_specs(common)
    selected = _optional_text(contrast_id)
    if selected is not None:
        contrasts = tuple(contrast for contrast in contrasts if contrast.contrast_id == selected)
        if not contrasts:
            raise ValueError(f"Unknown LOSO contrast_id {selected!r}.")

    inputs: list[FixedEffectsInput] = []
    for session in sessions:
        for task in tasks:
            for model_name in models:
                for contrast in contrasts:
                    inputs.extend(
                        _discover_fixed_effects_inputs_for_group(
                            common,
                            context=context,
                            subjects=subjects,
                            session_id=session,
                            task_id=task,
                            model=model_name,
                            contrast=contrast,
                        )
                    )
    return tuple(inputs)


def discover_loso_group_mask_inputs(
    document: Mapping[str, Any],
    *,
    context: Any,
    session_id: str | None = None,
    task_id: str | None = None,
    model: str | None = None,
    contrast_id: str | None = None,
    validate_personal_paths: bool = True,
) -> tuple[LosoGroupMaskInput, ...]:
    """Resolve configured LOSO group masks without planning or running FLAME1."""

    roi_set = load_loso_roi_set_config(document, validate_personal_paths=validate_personal_paths)
    common = _roi_set_common_fields(roi_set)
    subjects = _subjects(common)
    heldouts = _heldout_subjects(common, default_subjects=subjects)
    sessions = _entity_values(session_id or common.get("session_id") or common.get("session"), label="session")
    tasks = _entity_values(task_id or common.get("task_id") or common.get("task"), label="task")
    models = _entity_values(model or common.get("model"), label="model")
    contrasts = _contrast_specs(common)
    selected = _optional_text(contrast_id)
    if selected is not None:
        contrasts = tuple(contrast for contrast in contrasts if contrast.contrast_id == selected)
        if not contrasts:
            raise ValueError(f"Unknown LOSO contrast_id {selected!r}.")

    masks: list[LosoGroupMaskInput] = []
    for session in sessions:
        for task in tasks:
            for model_name in models:
                for contrast in contrasts:
                    if _uses_generated_group_mask(common):
                        all_inputs = _discover_fixed_effects_inputs_for_group(
                            common,
                            context=context,
                            subjects=subjects,
                            session_id=session,
                            task_id=task,
                            model=model_name,
                            contrast=contrast,
                        )
                        for heldout in heldouts:
                            training_inputs = tuple(
                                item for item in all_inputs if item.subject_id != heldout and item.complete
                            )
                            generated = _plan_generated_group_mask(
                                common,
                                context=context,
                                session_id=session,
                                task_id=task,
                                model=model_name,
                                contrast=contrast,
                                heldout_subject=heldout,
                                training_inputs=training_inputs,
                            )
                            masks.append(
                                LosoGroupMaskInput(
                                    session_id=session,
                                    task_id=task,
                                    model=model_name,
                                    contrast_id=contrast.contrast_id,
                                    cope_number=contrast.cope_number,
                                    mask_path=generated.mask_path,
                                    missing=(),
                                    generated=True,
                                    strategy=generated.strategy,
                                    scope=generated.scope,
                                    heldout_subject=heldout,
                                    included_subjects=generated.included_subjects,
                                    excluded_subjects=generated.excluded_subjects,
                                    source_mask_paths=generated.source_mask_paths,
                                    sidecar_path=generated.sidecar_path,
                                )
                            )
                    else:
                        mask_path = _resolve_static_group_mask_path(
                            common,
                            context=context,
                            session_id=session,
                            task_id=task,
                            model=model_name,
                            contrast=contrast,
                        )
                        masks.append(
                            LosoGroupMaskInput(
                                session_id=session,
                                task_id=task,
                                model=model_name,
                                contrast_id=contrast.contrast_id,
                                cope_number=contrast.cope_number,
                                mask_path=mask_path,
                                missing=() if mask_path.exists() else ("group_mask",),
                            )
                        )
    return tuple(masks)


def expected_loso_group_map_build_action_count(document: Mapping[str, Any]) -> int:
    """Return the number of LOSO ROI mask build actions implied by config."""

    roi_set = load_loso_roi_set_config(document)
    common = _roi_set_common_fields(roi_set)
    subjects = _subjects(common)
    heldouts = _heldout_subjects(common, default_subjects=subjects)
    sessions = _entity_values(common.get("session_id") or common.get("session"), label="session")
    tasks = _entity_values(common.get("task_id") or common.get("task"), label="task")
    models = _entity_values(common.get("model"), label="model")
    count = 0
    for roi in tuple(roi for roi in roi_set.rois if roi.family == "loso_group_map"):
        fields = _merge_fields(common, roi.fields)
        _contrast_for_roi(fields, _contrast_specs(fields))
        count += len(heldouts) * len(sessions) * len(tasks) * len(models)
    return count


def plan_loso_group_map_jobs(document: Mapping[str, Any], *, context: Any) -> tuple[LosoGroupMapJob, ...]:
    """Plan unique LOSO group-map jobs from a config document."""

    roi_set = load_loso_roi_set_config(document)
    loso_rois = tuple(roi for roi in roi_set.rois if roi.family == "loso_group_map")
    return tuple(_plan_unique_group_jobs(roi_set, loso_rois, context=context).values())


def plan_loso_group_map_build_actions(
    roi_set: RoiSet,
    rois: Sequence[RoiDefinition],
    *,
    context: Any,
) -> tuple[Any, ...]:
    """Plan LOSO ROI mask build actions for the Phase 3/4 execution surface."""

    from research_platform.neuro.roi_execution import RoiBuildAction

    jobs = _plan_unique_group_jobs(roi_set, rois, context=context)
    actions: list[RoiBuildAction] = []
    output_root = _resolve_base_output_root(_roi_set_common_fields(roi_set), context=context)
    roi_output_root = output_root / "rois" / roi_set.name

    for roi in rois:
        fields = _merge_fields(_roi_set_common_fields(roi_set), roi.fields)
        contrast = _contrast_for_roi(fields, _contrast_specs(fields))
        method_desc = _method_desc(roi, contrast)
        parameters = _roi_loso_parameters(fields)
        policy = _mask_intersection_policy(fields)
        coverage_paths = _coverage_mask_paths(fields, context=context)
        for job in jobs.values():
            if job.contrast.contrast_id != contrast.contrast_id:
                continue
            entities = _entities(
                subject_id=job.heldout_subject,
                session_id=job.session_id,
                task_id=job.task_id,
                model=job.model,
                fields=fields,
                roi_set_name=roi_set.name,
            )
            mask_path = _build_roi_mask_path(
                roi_output_root,
                entities=entities,
                roi_label=roi.label,
                method_desc=method_desc,
            )
            sidecar_path = _build_roi_sidecar_path(mask_path)
            input_paths = _action_input_paths(job, coverage_paths)
            actions.append(
                RoiBuildAction(
                    roi_label=roi.label,
                    family=roi.family,
                    backend="fsl_flame1",
                    mask_path=mask_path,
                    sidecar_path=sidecar_path,
                    input_paths=input_paths,
                    metadata={
                        "desc": method_desc,
                        "entities": entities,
                        "loso_group_job": job.to_dict(),
                        "roi_parameters": parameters,
                        "mask_intersection_policy": policy,
                        "coverage_masks": {name: str(path) for name, path in coverage_paths.items()},
                    },
                )
            )
    return tuple(actions)


def execute_loso_build_action(
    action: Any,
    roi: RoiDefinition,
    *,
    context: Any,
    cache_state: dict[str, Mapping[str, Any]] | None = None,
    provenance_job: Mapping[str, Any] | None = None,
) -> Any:
    """Execute one LOSO ROI build action, reusing group-map cache state."""

    from research_platform.neuro.fsl import flame
    from research_platform.neuro.nifti import load_nifti_image, validate_compatible_geometry
    from research_platform.neuro.roi_builders import build_loso_group_map_roi

    job = _job_metadata(action)
    recorded_job = provenance_job or job
    parameters = dict(action.metadata.get("roi_parameters", {}))
    cache_key = str(job["cache_key"])
    cache = cache_state if cache_state is not None else {}
    zstat_path = Path(str(job["zstat_path"]))
    sidecar_path = Path(str(job["sidecar_path"]))
    cache_status = "reused_in_memory" if cache_key in cache else None

    if cache_status is None and _can_reuse_group_map(zstat_path, sidecar_path, cache_key, bool(job["cache_reuse"])):
        cache_status = "reused_existing"
        cache[cache_key] = {"zstat_path": str(zstat_path), "status": cache_status}

    if cache_status is None:
        zstat_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(job.get("generated_group_mask"), Mapping):
            materialize_generated_group_mask(
                job,
                root_refs=getattr(context, "root_refs", {}),
                provenance_job=recorded_job,
            )
        command_plan = flame.build_flame1_command_plan(
            cope_inputs=[item["cope_path"] for item in job["training_inputs"]],
            varcope_inputs=[item["varcope_path"] for item in job["training_inputs"]],
            mask_path=job["group_mask_path"],
            work_dir=job["work_dir"],
            output_zstat_path=job["zstat_path"],
            environment=dict(job.get("backend_config", {})).get("environment", {}),
        )
        recorded_command_plan = flame.build_flame1_command_plan(
            cope_inputs=[item["cope_path"] for item in recorded_job["training_inputs"]],
            varcope_inputs=[item["varcope_path"] for item in recorded_job["training_inputs"]],
            mask_path=recorded_job["group_mask_path"],
            work_dir=recorded_job["work_dir"],
            output_zstat_path=recorded_job["zstat_path"],
            environment=dict(recorded_job.get("backend_config", {})).get("environment", {}),
        )
        flame.execute_flame1_command_plan(command_plan)
        _write_group_map_sidecar(
            recorded_job,
            output_path=sidecar_path,
            command_plan=recorded_command_plan.to_dict(),
            cache_status="computed",
            root_refs=getattr(context, "root_refs", {}),
        )
        cache_status = "computed"
        cache[cache_key] = {"zstat_path": str(zstat_path), "status": cache_status}

    zstat_image = load_nifti_image(zstat_path)
    policy = str(action.metadata.get("mask_intersection_policy") or "intersection")
    coverage_masks: dict[str, Any] = {}
    if policy == "intersection":
        group_mask_image = load_nifti_image(job["group_mask_path"])
        validate_compatible_geometry(zstat_image, group_mask_image)
        coverage_masks["group"] = group_mask_image.get_fdata()
        heldout = job.get("heldout_input")
        if not isinstance(heldout, Mapping):
            raise ValueError("LOSO ROI execution requires a held-out subject fixed-effects mask.")
        heldout_mask_image = load_nifti_image(str(heldout["mask_path"]))
        validate_compatible_geometry(zstat_image, heldout_mask_image)
        coverage_masks["heldout_subject"] = heldout_mask_image.get_fdata()
        for name, raw_path in dict(action.metadata.get("coverage_masks", {})).items():
            coverage_image = load_nifti_image(raw_path)
            validate_compatible_geometry(zstat_image, coverage_image)
            coverage_masks[str(name)] = coverage_image.get_fdata()
    elif policy != "none":
        raise ValueError("Phase 4 LOSO mask execution supports mask_intersection_policy none or intersection.")

    provenance = {
        "roi_set": job["roi_set"],
        "project": context.project_name,
        "held_out_subject": f"sub-{job['heldout_subject']}",
        "session": f"ses-{job['session_id']}" if job.get("session_id") else None,
        "task": job["task_id"],
        "model": job["model"],
        "source_contrast": job["contrast"]["contrast_id"],
        "cope_number": job["contrast"]["cope_number"],
        "group_map_path": _provenance_path_reference(
            recorded_job["zstat_path"],
            job=recorded_job,
            context=context,
        ),
        "group_map_cache_key": cache_key,
        "group_map_cache_status": cache_status,
        "training_subjects": [f"sub-{item['subject_id']}" for item in job["training_inputs"]],
        "mask_intersection_policy": policy,
    }
    provenance = {key: value for key, value in provenance.items() if value is not None}
    return build_loso_group_map_roi(
        zstat_image=zstat_image,
        roi_label=action.roi_label,
        desc=_optional_text(action.metadata.get("desc")),
        output_mask_path=action.mask_path,
        sidecar_path=action.sidecar_path,
        seed_xyz_mm=_number_sequence(parameters["seed_coordinate"], label="seed_coordinate"),
        search_radius_mm=_number_value(parameters["search_radius_mm"], label="search_radius_mm"),
        sphere_radius_mm=_number_value(parameters["sphere_radius_mm"], label="sphere_radius_mm"),
        z_threshold=_number_value(parameters["z_threshold"], label="z_threshold"),
        exploratory_z_threshold=_optional_number(parameters.get("exploratory_z_threshold")),
        allow_below_threshold_fallback=bool(parameters.get("allow_below_threshold_fallback", False)),
        coverage_masks=coverage_masks,
        min_voxels_warn=_optional_int(parameters.get("min_voxels_warn")),
        min_voxels_fail=_optional_int(parameters.get("min_voxels_fail")),
        fail_on_qc=bool(parameters.get("fail_on_qc", True)),
        provenance=provenance,
    )


def build_loso_group_map_cache_key(
    *,
    input_paths: Sequence[str | Path],
    contrast_id: str,
    heldout_subject: str,
    design_config: Mapping[str, Any] | None = None,
    backend_config: Mapping[str, Any] | None = None,
) -> str:
    """Build a deterministic lightweight cache key for one LOSO group map."""

    payload = {
        "backend_config": dict(backend_config or {}),
        "contrast_id": str(contrast_id),
        "design_config": dict(design_config or {}),
        "heldout_subject": _strip_entity_prefix(str(heldout_subject), "sub"),
        "inputs": [_path_fingerprint(path) for path in input_paths],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def materialize_generated_group_mask(
    job: Mapping[str, Any],
    *,
    root_refs: Mapping[str, str | Path] | None = None,
    provenance_job: Mapping[str, Any] | None = None,
) -> Path:
    """Write a generated LOSO group mask by intersecting source subject masks."""

    generated = job.get("generated_group_mask")
    if not isinstance(generated, Mapping):
        raise ValueError("LOSO group-map job does not define a generated_group_mask plan.")
    if generated.get("strategy") != GROUP_MASK_STRATEGY_FIXED_EFFECTS_INTERSECTION:
        raise ValueError(f"Unsupported generated group mask strategy {generated.get('strategy')!r}.")

    import numpy as np
    from research_platform.neuro.nifti import (
        load_nifti_image,
        make_nifti_like,
        save_nifti_image,
        validate_compatible_geometry,
    )
    from research_platform.neuro.roi_masks import validate_binary_mask

    source_paths = tuple(Path(str(path)) for path in generated.get("source_mask_paths", ()))
    if not source_paths:
        raise ValueError("Generated LOSO group mask requires at least one source mask.")

    reference_image = load_nifti_image(source_paths[0])
    source_masks = []
    for index, source_path in enumerate(source_paths):
        image = load_nifti_image(source_path)
        validate_compatible_geometry(reference_image, image)
        source_masks.append(validate_binary_mask(image.get_fdata(), allow_empty=True, label=f"source_masks[{index}]"))

    intersection = np.logical_and.reduce(source_masks)
    mask_path = Path(str(generated["mask_path"]))
    mask_image = make_nifti_like(reference_image, intersection.astype(np.uint8), dtype=np.uint8)
    save_nifti_image(mask_image, mask_path)
    recorded_job = provenance_job or job
    recorded_generated = recorded_job.get("generated_group_mask")
    if not isinstance(recorded_generated, Mapping):
        raise ValueError("LOSO provenance job does not define a generated_group_mask plan.")
    _write_generated_group_mask_sidecar(
        recorded_job,
        recorded_generated,
        output_path=Path(str(generated["sidecar_path"])),
        voxel_count=int(np.count_nonzero(intersection)),
        root_refs=root_refs,
    )
    return mask_path


def _plan_unique_group_jobs(
    roi_set: RoiSet,
    rois: Sequence[RoiDefinition],
    *,
    context: Any,
) -> dict[tuple[str, str | None, str, str, str], LosoGroupMapJob]:
    common = _roi_set_common_fields(roi_set)
    subjects = _subjects(common)
    heldouts = _heldout_subjects(common, default_subjects=subjects)
    sessions = _entity_values(common.get("session_id") or common.get("session"), label="session")
    tasks = _entity_values(common.get("task_id") or common.get("task"), label="task")
    models = _entity_values(common.get("model"), label="model")
    output_root = _resolve_base_output_root(common, context=context)
    group_output_root = output_root / "loso_groupmaps" / roi_set.name
    jobs: dict[tuple[str, str | None, str, str, str], LosoGroupMapJob] = {}

    for roi in rois:
        fields = _merge_fields(common, roi.fields)
        contrast = _contrast_for_roi(fields, _contrast_specs(fields))
        for heldout in heldouts:
            for session in sessions:
                for task in tasks:
                    for model in models:
                        key = (heldout, session, task, model, contrast.contrast_id)
                        if key in jobs:
                            continue
                        jobs[key] = _plan_group_job(
                            roi_set,
                            fields,
                            context=context,
                            subjects=subjects,
                            heldout_subject=heldout,
                            session_id=session,
                            task_id=task,
                            model=model,
                            contrast=contrast,
                            group_output_root=group_output_root,
                            output_root=output_root,
                        )
    return jobs


def _plan_group_job(
    roi_set: RoiSet,
    fields: Mapping[str, Any],
    *,
    context: Any,
    subjects: Sequence[str],
    heldout_subject: str,
    session_id: str | None,
    task_id: str,
    model: str,
    contrast: LosoContrastSpec,
    group_output_root: Path,
    output_root: Path,
) -> LosoGroupMapJob:
    all_inputs = _discover_fixed_effects_inputs_for_group(
        fields,
        context=context,
        subjects=subjects,
        session_id=session_id,
        task_id=task_id,
        model=model,
        contrast=contrast,
    )
    input_config = _normalized_fixed_effects_input_config(_fixed_effects_input_config(fields))
    input_root = _input_root(input_config, context=context)
    heldout_input = next((item for item in all_inputs if item.subject_id == heldout_subject), None)
    training_inputs = tuple(item for item in all_inputs if item.subject_id != heldout_subject and item.complete)
    min_group_n = _positive_int(fields.get("min_group_n") or _loso_block(fields).get("min_group_n"), label="min_group_n")
    warnings = _missing_input_messages(all_inputs)
    if heldout_input is None:
        warnings.append(f"held-out subject sub-{heldout_subject} was not present in configured subjects")
    elif "mask" in heldout_input.missing:
        warnings.append(f"held-out subject sub-{heldout_subject} is missing fixed-effects mask: {heldout_input.mask_path}")
    if len(training_inputs) < min_group_n:
        raise ValueError(
            "LOSO group map has fewer than min_group_n complete training subjects "
            f"for held-out sub-{heldout_subject}, contrast {contrast.contrast_id}: "
            f"{len(training_inputs)} < {min_group_n}."
        )

    generated_group_mask: GeneratedGroupMaskPlan | None = None
    if _uses_generated_group_mask(fields):
        generated_group_mask = _plan_generated_group_mask(
            fields,
            context=context,
            session_id=session_id,
            task_id=task_id,
            model=model,
            contrast=contrast,
            heldout_subject=heldout_subject,
            training_inputs=training_inputs,
        )
        group_mask_path = generated_group_mask.mask_path
    else:
        group_mask_path = _resolve_static_group_mask_path(
            fields,
            context=context,
            session_id=session_id,
            task_id=task_id,
            model=model,
            contrast=contrast,
        )
    if generated_group_mask is None and not group_mask_path.exists():
        warnings.append(f"group mask is missing: {group_mask_path}")
    if _missing_policy(fields) == "error" and warnings:
        raise ValueError("LOSO input discovery failed: " + "; ".join(warnings))

    backend_config = _backend_config(fields)
    design_config = {
        "kind": "one_sample_group_mean",
        "n_subjects": len(training_inputs),
        "session_id": session_id,
        "task_id": task_id,
        "model": model,
    }
    if generated_group_mask is not None:
        design_config["group_mask"] = {
            "strategy": generated_group_mask.strategy,
            "scope": generated_group_mask.scope,
        }
    input_paths = [] if generated_group_mask is not None else [group_mask_path]
    for item in training_inputs:
        input_paths.extend([item.cope_path, item.varcope_path, item.mask_path])
    if heldout_input is not None:
        input_paths.append(heldout_input.mask_path)
    cache_key = build_loso_group_map_cache_key(
        input_paths=input_paths,
        contrast_id=contrast.contrast_id,
        heldout_subject=heldout_subject,
        design_config=design_config,
        backend_config=backend_config,
    )
    zstat_path = _build_loso_group_map_path(
        group_output_root,
        session_id=session_id,
        task_id=task_id,
        space=str(fields["space"]),
        direction=_optional_text(fields.get("direction") or fields.get("dir")),
        resolution=_optional_text(fields.get("resolution") or fields.get("res")),
        method_desc=_bids_label(f"{model}{contrast.desc}"),
        heldout_subject=heldout_subject,
    )
    return LosoGroupMapJob(
        roi_set_name=roi_set.name,
        heldout_subject=heldout_subject,
        session_id=session_id,
        task_id=task_id,
        model=model,
        contrast=contrast,
        training_inputs=training_inputs,
        heldout_input=heldout_input,
        group_mask_path=group_mask_path,
        zstat_path=zstat_path,
        sidecar_path=_nii_sidecar_path(zstat_path),
        work_dir=output_root / ".cache" / "loso_groupmaps" / roi_set.name / cache_key[:16],
        output_root=output_root,
        input_root=input_root,
        cache_key=cache_key,
        cache_reuse=_cache_reuse(fields),
        generated_group_mask=generated_group_mask,
        backend_config=backend_config,
        design_config=design_config,
        warnings=tuple(warnings),
    )


def _discover_fixed_effects_inputs_for_group(
    fields: Mapping[str, Any],
    *,
    context: Any,
    subjects: Sequence[str],
    session_id: str | None,
    task_id: str,
    model: str,
    contrast: LosoContrastSpec,
) -> tuple[FixedEffectsInput, ...]:
    input_config = _normalized_fixed_effects_input_config(_fixed_effects_input_config(fields))
    root = _input_root(input_config, context=context)
    inputs: list[FixedEffectsInput] = []
    for subject in subjects:
        values = _entities(
            subject_id=subject,
            session_id=session_id,
            task_id=task_id,
            model=model,
            fields={**fields, **contrast.fields},
            roi_set_name=_optional_text(fields.get("name")) or _optional_text(fields.get("roi_set")) or "roi_set",
        )
        values.update(
            {
                "contrast_id": contrast.contrast_id,
                "contrast": contrast.contrast_id,
                "contrast_desc": contrast.desc,
                "cope_number": contrast.cope_number,
                "cope": contrast.cope_number,
                "model": model,
            }
        )
        cope_dir = _resolve_input_path(root, _required_input_value(input_config, "cope_dir", "cope_dir_pattern"), values, label="cope_dir")
        cope_path = _resolve_relative_or_spec(
            input_config,
            root=root,
            base=cope_dir,
            values=values,
            keys=("cope_image", "cope_image_path", "cope_path", "cope_pattern"),
            label="cope_image",
        )
        varcope_path = _resolve_relative_or_spec(
            input_config,
            root=root,
            base=cope_dir,
            values=values,
            keys=("varcope_image", "varcope_image_path", "varcope_path", "varcope_pattern"),
            label="varcope_image",
        )
        mask_path = _resolve_relative_or_spec(
            input_config,
            root=root,
            base=cope_dir,
            values=values,
            keys=("mask_image", "mask_path", "mask", "subject_mask"),
            label="mask_image",
        )
        missing = tuple(
            name
            for name, path in (
                ("cope", cope_path),
                ("varcope", varcope_path),
                ("mask", mask_path),
            )
            if not path.exists()
        )
        inputs.append(
            FixedEffectsInput(
                subject_id=subject,
                session_id=session_id,
                task_id=task_id,
                model=model,
                contrast_id=contrast.contrast_id,
                cope_number=contrast.cope_number,
                cope_dir=cope_dir,
                cope_path=cope_path,
                varcope_path=varcope_path,
                mask_path=mask_path,
                missing=missing,
            )
        )
    return tuple(inputs)


def _roi_loso_parameters(fields: Mapping[str, Any]) -> dict[str, Any]:
    seed = fields.get("seed_coordinate", fields.get("coordinate"))
    if seed is None:
        raise ValueError("LOSO ROI config must define seed_coordinate or coordinate.")
    parameters = {
        "seed_coordinate": _number_sequence(seed, label="seed_coordinate"),
        "search_radius_mm": _number_value(_required_value(fields, "search_radius_mm"), label="search_radius_mm"),
        "sphere_radius_mm": _number_value(_required_value(fields, "sphere_radius_mm", "radius_mm"), label="sphere_radius_mm"),
        "z_threshold": _number_value(_required_value(fields, "z_threshold"), label="z_threshold"),
        "allow_below_threshold_fallback": _bool_value(
            fields.get("allow_below_threshold_fallback", fields.get("below_threshold_fallback")),
            default=False,
        ),
        "fail_on_qc": _bool_value(fields.get("fail_on_qc"), default=True),
    }
    exploratory = _optional_number(fields.get("exploratory_z_threshold"))
    if exploratory is not None:
        parameters["exploratory_z_threshold"] = exploratory
    for key in ("min_voxels_warn", "min_voxels_fail"):
        value = _optional_int(fields.get(key))
        if value is not None:
            parameters[key] = value
    return parameters


def _action_input_paths(job: LosoGroupMapJob, coverage_paths: Mapping[str, Path]) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "loso_zstat": job.zstat_path,
        "group_mask": job.group_mask_path,
    }
    if job.heldout_input is not None:
        paths["heldout_mask"] = job.heldout_input.mask_path
    for item in job.training_inputs:
        paths[f"cope:sub-{item.subject_id}"] = item.cope_path
        paths[f"varcope:sub-{item.subject_id}"] = item.varcope_path
        paths[f"mask:sub-{item.subject_id}"] = item.mask_path
    for name, path in coverage_paths.items():
        paths[f"coverage_mask:{name}"] = path
    return paths


def _write_generated_group_mask_sidecar(
    job: Mapping[str, Any],
    generated: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
    voxel_count: int,
    root_refs: Mapping[str, str | Path] | None = None,
) -> Path:
    sidecar_path = Path(output_path) if output_path is not None else Path(str(generated["sidecar_path"]))
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    payload = normalize_sidecar_provenance(
        {
            "backend": GROUP_MASK_STRATEGY_FIXED_EFFECTS_INTERSECTION,
            "generation_policy": generated["strategy"],
            "scope": generated.get("scope"),
            "group_mask_path": generated["mask_path"],
            "source_mask_paths": list(generated.get("source_mask_paths", ())),
            "source_mask_count": len(tuple(generated.get("source_mask_paths", ()))),
            "voxel_count": voxel_count,
            "included_subjects": [f"sub-{subject}" for subject in generated.get("included_subjects", ())],
            "excluded_subjects": [f"sub-{subject}" for subject in generated.get("excluded_subjects", ())],
            "held_out_subject": f"sub-{job['heldout_subject']}",
            "contrast": job["contrast"],
            "cope_number": job["contrast"]["cope_number"],
            "model": job["model"],
            "roi_set": job["roi_set"],
            "session": f"ses-{job['session_id']}" if job.get("session_id") else None,
            "task": job["task_id"],
        },
        env_roots=_sidecar_env_roots(job),
        root_refs=root_refs,
    )
    payload = {key: value for key, value in payload.items() if value is not None}
    errors = validate_portable_provenance_paths(payload, label="LOSO generated group-mask sidecar")
    if errors:
        raise ValueError("Invalid LOSO generated group-mask sidecar provenance: " + "; ".join(errors))
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar_path


def _write_group_map_sidecar(
    job: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
    command_plan: Mapping[str, Any],
    cache_status: str,
    root_refs: Mapping[str, str | Path] | None = None,
) -> Path:
    sidecar_path = Path(output_path) if output_path is not None else Path(str(job["sidecar_path"]))
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    payload = normalize_sidecar_provenance(
        {
            "backend": "fsl_flame1",
            "cache_key": job["cache_key"],
            "cache_status": cache_status,
            "contrast": job["contrast"],
            "group_mask_path": job["group_mask_path"],
            "held_out_subject": f"sub-{job['heldout_subject']}",
            "model": job["model"],
            "roi_set": job["roi_set"],
            "session": f"ses-{job['session_id']}" if job.get("session_id") else None,
            "task": job["task_id"],
            "training_subjects": [f"sub-{item['subject_id']}" for item in job["training_inputs"]],
            "zstat_path": job["zstat_path"],
            "command_plan": command_plan,
        },
        env_roots=_sidecar_env_roots(job),
        root_refs=root_refs,
    )
    payload = {key: value for key, value in payload.items() if value is not None}
    errors = validate_portable_provenance_paths(payload, label="LOSO group-map sidecar")
    if errors:
        raise ValueError("Invalid LOSO group-map sidecar provenance: " + "; ".join(errors))
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar_path


def _can_reuse_group_map(zstat_path: Path, sidecar_path: Path, cache_key: str, reuse: bool) -> bool:
    if not reuse or not zstat_path.exists() or not sidecar_path.exists():
        return False
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(payload, Mapping) and payload.get("cache_key") == cache_key


def _provenance_path_reference(path: str | Path, *, job: Mapping[str, Any], context: Any) -> str:
    env_roots: dict[str, str | Path] = {}
    output_root = job.get("output_root")
    if output_root is not None:
        env_roots["ROI_DERIV_ROOT"] = output_root
    return provenance_path_reference(path, env_roots=env_roots, root_refs=getattr(context, "root_refs", {}))


def _sidecar_env_roots(job: Mapping[str, Any]) -> dict[str, str | Path]:
    env_roots: dict[str, str | Path] = {}
    output_root = job.get("output_root")
    if output_root is not None:
        env_roots["ROI_DERIV_ROOT"] = output_root
    input_root = job.get("input_root")
    if input_root is not None:
        env_roots["ROI_FEAT_ROOT"] = input_root
    return env_roots


def _job_metadata(action: Any) -> Mapping[str, Any]:
    job = action.metadata.get("loso_group_job")
    if not isinstance(job, Mapping):
        raise ValueError("LOSO build action is missing group-map job metadata.")
    return job


def _roi_set_common_fields(roi_set: RoiSet) -> dict[str, Any]:
    return {
        **dict(roi_set.fields),
        "name": roi_set.name,
        "roi_set": roi_set.name,
        "desc": roi_set.desc,
        "backend": roi_set.backend,
        **dict(roi_set.provenance if isinstance(roi_set.provenance, Mapping) else {}),
    }


def _merge_fields(common: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(common)
    for key, value in fields.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = {**dict(merged[key]), **dict(value)}
        else:
            merged[key] = value
    return merged


def _loso_block(fields: Mapping[str, Any]) -> Mapping[str, Any]:
    block = fields.get("loso")
    return block if isinstance(block, Mapping) else {}


def _subjects(fields: Mapping[str, Any]) -> tuple[str, ...]:
    candidates = (
        fields.get("subjects"),
        fields.get("subject_ids"),
        _loso_block(fields).get("subjects"),
        _loso_block(fields).get("subject_ids"),
    )
    for candidate in candidates:
        values = _optional_string_list(candidate)
        if values:
            return tuple(_strip_entity_prefix(value, "sub") for value in values)
    raise ValueError("LOSO ROI config must define subjects or subject_ids.")


def _heldout_subjects(fields: Mapping[str, Any], *, default_subjects: Sequence[str]) -> tuple[str, ...]:
    candidates = (
        fields.get("heldout_subjects"),
        fields.get("held_out_subjects"),
        fields.get("heldout_subject"),
        fields.get("held_out_subject"),
        _loso_block(fields).get("heldout_subjects"),
        _loso_block(fields).get("held_out_subjects"),
    )
    for candidate in candidates:
        values = _optional_string_list(candidate)
        if values:
            return tuple(_strip_entity_prefix(value, "sub") for value in values)
    return tuple(default_subjects)


def _entity_values(value: Any, *, label: str) -> tuple[str | None, ...]:
    values = _optional_string_list(value)
    if not values:
        if label == "session":
            return (None,)
        raise ValueError(f"LOSO ROI config must define {label}.")
    if label == "session":
        return tuple(_strip_entity_prefix(item, "ses") for item in values)
    return tuple(values)


def _contrast_specs(fields: Mapping[str, Any]) -> tuple[LosoContrastSpec, ...]:
    raw = fields.get("contrasts") or _loso_block(fields).get("contrasts")
    if raw is None:
        contrast_id = _optional_text(fields.get("contrast_id") or fields.get("contrast") or fields.get("source_contrast"))
        cope_number = _optional_text(fields.get("cope_number") or fields.get("cope"))
        if contrast_id and cope_number:
            return (LosoContrastSpec(contrast_id=contrast_id, cope_number=cope_number, desc=_bids_label(contrast_id), fields={}),)
        raise ValueError("LOSO ROI config must define contrasts or contrast_id plus cope_number.")
    if not isinstance(raw, list) or not raw:
        raise ValueError("LOSO contrasts must contain a non-empty list.")
    contrasts: list[LosoContrastSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"LOSO contrasts[{index}] must contain a mapping.")
        contrast_id = _optional_text(item.get("id") or item.get("name") or item.get("contrast_id") or item.get("contrast"))
        cope_number = _optional_text(item.get("cope_number") or item.get("cope") or item.get("cope_id"))
        if contrast_id is None or cope_number is None:
            raise ValueError(f"LOSO contrasts[{index}] must define id and cope_number.")
        desc = _optional_text(item.get("desc") or item.get("method_desc")) or _bids_label(contrast_id)
        contrasts.append(LosoContrastSpec(contrast_id=contrast_id, cope_number=cope_number, desc=_bids_label(desc), fields=dict(item)))
    return tuple(contrasts)


def _contrast_for_roi(fields: Mapping[str, Any], contrasts: Sequence[LosoContrastSpec]) -> LosoContrastSpec:
    selected = _optional_text(fields.get("contrast_id") or fields.get("contrast") or fields.get("source_contrast"))
    if selected is None:
        if len(contrasts) == 1:
            return contrasts[0]
        raise ValueError("LOSO ROI must define contrast_id when multiple contrasts are configured.")
    for contrast in contrasts:
        if contrast.contrast_id == selected:
            return contrast
    raise ValueError(f"LOSO ROI references unknown contrast_id {selected!r}.")


def _method_desc(roi: RoiDefinition, contrast: LosoContrastSpec) -> str:
    return _bids_label(_optional_text(roi.desc) or contrast.desc)


def _fixed_effects_input_config(fields: Mapping[str, Any]) -> Mapping[str, Any]:
    for source in (
        fields.get("fixed_effects_inputs"),
        fields.get("subject_fixed_effects"),
        fields.get("subject_level_fixed_effects"),
        _loso_block(fields).get("fixed_effects_inputs"),
        _loso_block(fields).get("subject_fixed_effects"),
    ):
        if isinstance(source, Mapping):
            return source
    inputs = fields.get("inputs")
    if isinstance(inputs, Mapping):
        for key in ("fixed_effects", "subject_fixed_effects", "subject_level_fixed_effects"):
            if isinstance(inputs.get(key), Mapping):
                return inputs[key]
    raise ValueError("LOSO ROI config must define subject fixed-effects input discovery settings.")


def _normalized_fixed_effects_input_config(input_config: Mapping[str, Any]) -> Mapping[str, Any]:
    layout = _optional_text(input_config.get("layout") or input_config.get("source_layout"))
    if layout is None:
        return input_config
    if layout not in _FIXED_EFFECTS_INPUT_LAYOUTS:
        known = ", ".join(sorted(_FIXED_EFFECTS_INPUT_LAYOUTS))
        raise ValueError(f"Unsupported LOSO fixed-effects input layout {layout!r}. Known layouts: {known}.")
    if layout == FIXED_EFFECTS_LAYOUT_FSL_GFEAT_NESTED:
        normalized = dict(input_config)
        if _optional_text(normalized.get("cope_dir") or normalized.get("cope_dir_pattern")) is None:
            gfeat_dir = _optional_text(
                normalized.get("gfeat_dir")
                or normalized.get("gfeat_dir_template")
                or normalized.get("group_feat_dir")
                or normalized.get("group_feat_dir_template")
            )
            normalized["cope_dir"] = f"{gfeat_dir or _FSL_GFEAT_NESTED_GFEAT_DIR}/cope{{cope_number}}.feat"
        normalized.setdefault("cope_image", "stats/cope1.nii.gz")
        normalized.setdefault("varcope_image", "stats/varcope1.nii.gz")
        normalized.setdefault("mask_image", "mask.nii.gz")
        return normalized
    return input_config


def _input_root(input_config: Mapping[str, Any], *, context: Any) -> Path:
    root_ref = _optional_text(input_config.get("root_ref"))
    if root_ref is not None:
        return Path(context.resolve_root_ref(root_ref))
    root = input_config.get("root") or input_config.get("path_root")
    if root is not None:
        return _resolve_path(root, context=context, values={}, label="fixed_effects.root")
    return context.project_root


def _uses_generated_group_mask(fields: Mapping[str, Any]) -> bool:
    spec, _label = _group_mask_config(fields)
    return _generated_group_mask_strategy(spec) is not None


def _plan_generated_group_mask(
    fields: Mapping[str, Any],
    *,
    context: Any,
    session_id: str | None,
    task_id: str,
    model: str,
    contrast: LosoContrastSpec,
    heldout_subject: str,
    training_inputs: Sequence[FixedEffectsInput],
) -> GeneratedGroupMaskPlan:
    spec, label = _group_mask_config(fields)
    strategy = _generated_group_mask_strategy(spec)
    if strategy is None:
        raise ValueError(f"{label} must define strategy: {GROUP_MASK_STRATEGY_FIXED_EFFECTS_INTERSECTION}.")
    scope = _generated_group_mask_scope(spec)
    heldout = _strip_entity_prefix(heldout_subject, "sub")
    values = _group_mask_template_values(
        fields,
        session_id=session_id,
        task_id=task_id,
        model=model,
        contrast=contrast,
        heldout_subject=heldout,
    )
    path_spec = _generated_group_mask_path_spec(spec)
    mask_path = _resolve_path(path_spec, context=context, values=values, label=label)
    included = tuple(item.subject_id for item in training_inputs)
    return GeneratedGroupMaskPlan(
        strategy=strategy,
        scope=scope,
        mask_path=mask_path,
        sidecar_path=_nii_sidecar_path(mask_path),
        source_mask_paths=tuple(item.mask_path for item in training_inputs),
        included_subjects=included,
        excluded_subjects=(heldout,),
        heldout_subject=heldout,
        contrast_id=contrast.contrast_id,
        cope_number=contrast.cope_number,
    )


def _resolve_static_group_mask_path(
    fields: Mapping[str, Any],
    *,
    context: Any,
    session_id: str | None,
    task_id: str,
    model: str,
    contrast: LosoContrastSpec,
) -> Path:
    spec, label = _group_mask_config(fields)
    if _generated_group_mask_strategy(spec) is not None:
        raise ValueError(f"{label} defines a generated group mask and requires LOSO planning context.")
    values = _group_mask_template_values(
        fields,
        session_id=session_id,
        task_id=task_id,
        model=model,
        contrast=contrast,
        heldout_subject=None,
    )
    return _resolve_path(spec, context=context, values=values, label=label)


def _group_mask_config(fields: Mapping[str, Any]) -> tuple[Any, str]:
    for key in ("group_mask", "group_mask_path", "mask"):
        if fields.get(key) is not None:
            return fields[key], key
    loso_group_mask = _loso_block(fields).get("group_mask")
    if loso_group_mask is not None:
        return loso_group_mask, "loso.group_mask"
    raise ValueError("LOSO ROI config must define a group_mask or group_mask_path.")


def _group_mask_template_values(
    fields: Mapping[str, Any],
    *,
    session_id: str | None,
    task_id: str,
    model: str,
    contrast: LosoContrastSpec,
    heldout_subject: str | None,
) -> dict[str, Any]:
    values = _entities(
        subject_id="group",
        session_id=session_id,
        task_id=task_id,
        model=model,
        fields={**fields, **contrast.fields},
        roi_set_name=_optional_text(fields.get("name")) or _optional_text(fields.get("roi_set")) or "roi_set",
    )
    values.update(
        {
            "contrast_id": contrast.contrast_id,
            "contrast": contrast.contrast_id,
            "contrast_desc": contrast.desc,
            "cope_number": contrast.cope_number,
            "cope": contrast.cope_number,
        }
    )
    if heldout_subject is not None:
        heldout = _strip_entity_prefix(heldout_subject, "sub")
        values.update(
            {
                "heldout_subject": heldout,
                "heldout_subject_id": heldout,
                "heldout_subject_dir": f"sub-{heldout}",
                "held_out_subject": heldout,
                "held_out_subject_id": heldout,
                "held_out_subject_dir": f"sub-{heldout}",
            }
        )
    return values


def _generated_group_mask_strategy(spec: Any) -> str | None:
    if not isinstance(spec, Mapping):
        return None
    raw = _optional_text(spec.get("strategy") or spec.get("source") or spec.get("policy"))
    if raw is None:
        return None
    strategy = _GENERATED_GROUP_MASK_STRATEGY_ALIASES.get(raw)
    if strategy is None:
        known = ", ".join(sorted(_GENERATED_GROUP_MASK_STRATEGY_ALIASES))
        raise ValueError(f"Unsupported generated LOSO group mask strategy {raw!r}. Known strategies: {known}.")
    return strategy


def _generated_group_mask_scope(spec: Any) -> str:
    if not isinstance(spec, Mapping):
        return "loso_training_subjects"
    scope = _optional_text(spec.get("scope")) or "loso_training_subjects"
    if scope not in _GENERATED_GROUP_MASK_SCOPES:
        known = ", ".join(sorted(_GENERATED_GROUP_MASK_SCOPES))
        raise ValueError(f"Unsupported generated LOSO group mask scope {scope!r}. Known scopes: {known}.")
    return scope


def _generated_group_mask_path_spec(spec: Any) -> Any:
    if not isinstance(spec, Mapping):
        return spec
    normalized = dict(spec)
    if normalized.get("path") is None and normalized.get("pattern") is None:
        for key in ("output_template", "template", "subpath"):
            if normalized.get(key) is not None:
                normalized["path"] = normalized[key]
                break
    if normalized.get("root_ref") is None and normalized.get("output_root_ref") is not None:
        normalized["root_ref"] = normalized["output_root_ref"]
    if normalized.get("root") is None and normalized.get("output_root") is not None:
        normalized["root"] = normalized["output_root"]
    return normalized


def _coverage_mask_paths(fields: Mapping[str, Any], *, context: Any) -> dict[str, Path]:
    raw = fields.get("coverage_masks") or _loso_block(fields).get("coverage_masks")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("coverage_masks must contain a mapping.")
    values = _entities(
        subject_id=_strip_entity_prefix(_optional_text(fields.get("subject_id") or fields.get("subject")) or "group", "sub"),
        session_id=_strip_entity_prefix(_optional_text(fields.get("session_id") or fields.get("session")), "ses"),
        task_id=_optional_text(fields.get("task_id") or fields.get("task")) or "task",
        model=_optional_text(fields.get("model")) or "model",
        fields=fields,
        roi_set_name=_optional_text(fields.get("name")) or _optional_text(fields.get("roi_set")) or "roi_set",
    )
    return {str(name): _resolve_path(spec, context=context, values=values, label=f"coverage_masks.{name}") for name, spec in raw.items()}


def _resolve_base_output_root(fields: Mapping[str, Any], *, context: Any) -> Path:
    outputs = fields.get("outputs")
    if isinstance(outputs, Mapping):
        if outputs.get("root_ref") is not None:
            root = context.resolve_root_ref(str(outputs["root_ref"]))
            subpath = _optional_text(outputs.get("path") or outputs.get("subpath"))
            return root / _named_root_relative_path(subpath, label="outputs.path") if subpath else root
        if outputs.get("path") is not None:
            return context.project_root / _named_root_relative_path(outputs["path"], label="outputs.path")
        for key in ("root", "output_root", "derivative_root"):
            if outputs.get(key) is not None:
                return _resolve_path(outputs[key], context=context, values={}, label=f"outputs.{key}")
    for key in ("output_root", "derivative_root"):
        if fields.get(key) is not None:
            return _resolve_path(fields[key], context=context, values={}, label=key)
    project = context.project_name or "project"
    return context.artifacts_root / "roi" / project / "derivatives"


def _named_root_relative_path(value: Any, *, label: str) -> Path:
    text = str(value).strip()
    if not text or configured_path_is_unsafe(text):
        raise ValueError(
            f"{label} must be a relative path that remains beneath its configured root."
        )
    return Path(text)


def _build_roi_mask_path(
    derivative_root: Path,
    *,
    entities: Mapping[str, Any],
    roi_label: str,
    method_desc: str,
) -> Path:
    from research_platform.bids.roi import build_roi_mask_path

    return build_roi_mask_path(
        derivative_root,
        subject_id=str(entities["subject_id"]),
        session_id=_optional_text(entities.get("session_id")),
        task_id=str(entities["task_id"]),
        direction=_optional_text(entities.get("direction")),
        space=str(entities["space"]),
        resolution=_optional_text(entities.get("resolution")),
        roi_label=roi_label,
        method_desc=method_desc,
        pipeline_name=None,
        datatype=str(entities.get("datatype", "func")),
    )


def _build_loso_group_map_path(
    derivative_root: Path,
    *,
    session_id: str | None,
    task_id: str,
    space: str,
    direction: str | None,
    resolution: str | None,
    method_desc: str,
    heldout_subject: str,
) -> Path:
    from research_platform.bids.roi import build_loso_group_map_path

    return build_loso_group_map_path(
        derivative_root,
        session_id=session_id,
        task_id=task_id,
        direction=direction,
        space=space,
        resolution=resolution,
        method_desc=method_desc,
        heldout_subject=heldout_subject,
        statistic=None,
        suffix="zstat",
        pipeline_name=None,
    )


def _build_roi_sidecar_path(mask_path: Path) -> Path:
    from research_platform.bids.roi import build_roi_sidecar_path

    return build_roi_sidecar_path(mask_path)


def _nii_sidecar_path(path: Path) -> Path:
    if path.name.endswith(".nii.gz"):
        return path.with_name(f"{path.name[:-7]}.json")
    return path.with_suffix(".json")


def _resolve_relative_or_spec(
    input_config: Mapping[str, Any],
    *,
    root: Path,
    base: Path,
    values: Mapping[str, Any],
    keys: Sequence[str],
    label: str,
) -> Path:
    for key in keys:
        if input_config.get(key) is None:
            continue
        raw = input_config[key]
        if isinstance(raw, Mapping):
            return _resolve_path(raw, context=_PathContext(root), values=values, label=key)
        rendered = _render_template(str(raw), values, label=key)
        path = Path(rendered).expanduser()
        if path.is_absolute():
            return path.resolve()
        if key.endswith("pattern") or key.endswith("_path"):
            return (root / path).resolve()
        return (base / path).resolve()
    raise ValueError(f"LOSO fixed-effects input config must define {label}.")


def _resolve_input_path(root: Path, pattern: Any, values: Mapping[str, Any], *, label: str) -> Path:
    rendered = _render_template(str(pattern), values, label=label)
    path = Path(rendered).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_path(spec: Any, *, context: Any, values: Mapping[str, Any], label: str) -> Path:
    if isinstance(spec, (str, Path)):
        rendered = _render_template(str(spec), values, label=label)
        path = Path(rendered).expanduser()
        return path.resolve() if path.is_absolute() else (context.project_root / path).resolve()
    if not isinstance(spec, Mapping):
        raise ValueError(f"{label} must be a path string or mapping.")
    root_ref = _optional_text(spec.get("root_ref") or spec.get("output_root_ref"))
    if root_ref is not None:
        base = context.resolve_root_ref(root_ref)
    else:
        root_value = spec.get("root") or spec.get("output_root") or spec.get("path_root")
        base = _resolve_path_root(root_value, context=context, values=values, label=f"{label}.root") if root_value is not None else context.project_root
    raw = spec.get("pattern") if spec.get("pattern") is not None else spec.get("path")
    if raw is None:
        for key in ("output_template", "template", "subpath"):
            if spec.get(key) is not None:
                raw = spec[key]
                break
    if raw is None:
        return Path(base)
    rendered = _render_template(str(raw), values, label=label)
    if root_ref is not None:
        return Path(base) / _named_root_relative_path(rendered, label=label)
    path = Path(rendered).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def _resolve_path_root(root_value: Any, *, context: Any, values: Mapping[str, Any], label: str) -> Path:
    if isinstance(root_value, Mapping):
        return _resolve_path(root_value, context=context, values=values, label=label)
    rendered = _render_template(str(root_value), values, label=label)
    path = Path(rendered).expanduser()
    return path.resolve() if path.is_absolute() else (context.project_root / path).resolve()


class _PathContext:
    def __init__(self, root: Path) -> None:
        self.project_root = root

    def resolve_root_ref(self, name: str) -> Path:
        raise ValueError(f"Nested input path specs cannot use root_ref {name!r}.")


def _required_input_value(input_config: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if input_config.get(key) is not None:
            return input_config[key]
    raise ValueError(f"LOSO fixed-effects input config must define one of: {', '.join(keys)}.")


def _required_value(fields: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if fields.get(key) is not None:
            return fields[key]
    raise ValueError(f"LOSO ROI config must define one of: {', '.join(keys)}.")


def _entities(
    *,
    subject_id: str,
    session_id: str | None,
    task_id: str,
    model: str,
    fields: Mapping[str, Any],
    roi_set_name: str,
) -> dict[str, Any]:
    subject = _strip_entity_prefix(subject_id, "sub")
    session = _strip_entity_prefix(session_id, "ses") if session_id is not None else None
    direction = _optional_text(fields.get("direction") or fields.get("dir"))
    resolution = _optional_text(fields.get("resolution") or fields.get("res"))
    space = _optional_text(fields.get("space"))
    if space is None:
        raise ValueError("LOSO ROI config must define space.")
    return {
        "subject_id": subject,
        "subject": subject,
        "subject_dir": f"sub-{subject}" if subject != "group" else "group",
        "session_id": session,
        "session": session,
        "session_dir": f"ses-{session}" if session else "",
        "task_id": task_id,
        "task": task_id,
        "model": model,
        "space": space,
        "direction": direction,
        "dir": direction,
        "resolution": resolution,
        "res": resolution,
        "datatype": _optional_text(fields.get("datatype")) or "func",
        "roi_set": roi_set_name,
    }


def _missing_input_messages(inputs: Sequence[FixedEffectsInput]) -> list[str]:
    messages: list[str] = []
    for item in inputs:
        for missing in item.missing:
            path = getattr(item, f"{missing}_path")
            messages.append(f"sub-{item.subject_id} missing {missing}: {path}")
    return messages


def _missing_policy(fields: Mapping[str, Any]) -> str:
    policy = _optional_text(fields.get("missing_inputs") or fields.get("missing_input_policy") or _loso_block(fields).get("missing_input_policy"))
    if policy is None:
        return "warn"
    if policy not in {"warn", "error"}:
        raise ValueError("missing_input_policy must be warn or error.")
    return policy


def _mask_intersection_policy(fields: Mapping[str, Any]) -> str:
    policy = _optional_text(fields.get("mask_intersection_policy") or _loso_block(fields).get("mask_intersection_policy")) or "intersection"
    if policy not in {"none", "intersection"}:
        raise ValueError("Phase 4 LOSO mask execution supports mask_intersection_policy none or intersection.")
    return policy


def _cache_reuse(fields: Mapping[str, Any]) -> bool:
    cache = fields.get("cache") or _loso_block(fields).get("cache")
    if isinstance(cache, Mapping):
        return _bool_value(cache.get("reuse"), default=True)
    if fields.get("reuse_loso_group_maps") is not None:
        return _bool_value(fields.get("reuse_loso_group_maps"), default=True)
    return True


def _backend_config(fields: Mapping[str, Any]) -> dict[str, Any]:
    config = fields.get("backend_config") or fields.get("fsl_flame1") or _loso_block(fields).get("backend_config")
    if isinstance(config, Mapping):
        return dict(config)
    runtime = fields.get("runtime")
    if isinstance(runtime, Mapping):
        return dict(runtime)
    return {}


def _path_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    if resolved.exists():
        stat = resolved.stat()
        payload.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return payload


def _render_template(template: str, values: Mapping[str, Any], *, label: str) -> str:
    names = {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}
    missing = sorted(name for name in names if name not in values)
    if missing:
        raise ValueError(f"{label} references missing template field(s): {', '.join(missing)}.")
    return template.format(**values)


def _optional_string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _strip_entity_prefix(value: str | None, prefix: str) -> str:
    text = _optional_text(value)
    if text is None:
        return ""
    token = f"{prefix}-"
    return text[len(token) :] if text.startswith(token) else text


def _bids_label(value: str) -> str:
    cleaned = "".join(character for character in str(value) if character.isalnum())
    if not cleaned:
        raise ValueError(f"Cannot derive BIDS-like label from {value!r}.")
    return cleaned


def _number_sequence(value: Any, *, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three numeric values.")
    return tuple(_number_value(item, label=f"{label}[{index}]") for index, item in enumerate(value))


def _number_value(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    return _number_value(value, label="number")


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    return int(value)


def _positive_int(value: Any, *, label: str) -> int:
    if value is None:
        raise ValueError(f"LOSO ROI config must define {label}.")
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return number


def _bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Boolean LOSO ROI config fields must contain true or false.")
