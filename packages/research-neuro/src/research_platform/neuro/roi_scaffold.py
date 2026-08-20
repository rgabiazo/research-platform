"""Scaffold reusable ROI workflow configuration YAML.

The helpers in this module render starter config only. They do not import
neuroimaging runtimes, inspect images, create ROI masks, or run extraction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import copy
import re

from research_platform.neuro.roi import validate_extraction_set_document, validate_roi_set_document


ROI_SET_TEMPLATES = frozenset(
    {
        "coordinate_sphere",
        "manual_mask",
        "atlas_label",
        "functional_threshold_map",
        "loso_group_map",
        "data_driven_hook",
    }
)
EXTRACTION_SET_TEMPLATES = frozenset({"generic_nifti", "fsl_featquery"})
_SINGLE_ENTITY_ROI_SET_TEMPLATES = frozenset(
    {
        "coordinate_sphere",
        "manual_mask",
        "functional_threshold_map",
        "atlas_label",
        "data_driven_hook",
    }
)
_SINGLE_ENTITY_EXTRACTION_SET_TEMPLATES = frozenset({"generic_nifti"})
_MULTI_ENTITY_ROI_SET_TEMPLATES = frozenset({"loso_group_map"})
_MULTI_ENTITY_EXTRACTION_SET_TEMPLATES = frozenset({"fsl_featquery"})
if (
    _SINGLE_ENTITY_ROI_SET_TEMPLATES & _MULTI_ENTITY_ROI_SET_TEMPLATES
    or (_SINGLE_ENTITY_ROI_SET_TEMPLATES | _MULTI_ENTITY_ROI_SET_TEMPLATES) != ROI_SET_TEMPLATES
    or _SINGLE_ENTITY_EXTRACTION_SET_TEMPLATES & _MULTI_ENTITY_EXTRACTION_SET_TEMPLATES
    or (_SINGLE_ENTITY_EXTRACTION_SET_TEMPLATES | _MULTI_ENTITY_EXTRACTION_SET_TEMPLATES)
    != EXTRACTION_SET_TEMPLATES
):
    raise RuntimeError("Every ROI scaffold template must have exactly one subject-cardinality classification.")

ROI_SET_TEMPLATE_METADATA: dict[str, dict[str, str]] = {
    "coordinate_sphere": {
        "status": "local_nifti",
        "description": "Local NIfTI sphere build; requires a user-provided reference image.",
    },
    "manual_mask": {
        "status": "local_nifti",
        "description": "Local NIfTI mask copy; requires a user-provided binary mask.",
    },
    "functional_threshold_map": {
        "status": "local_nifti",
        "description": "Local NIfTI peak/sphere build; requires a user-provided statistical map.",
    },
    "loso_group_map": {
        "status": "external_runtime",
        "description": "LOSO group-map build; requires fixed-effects inputs and local FSL tools.",
    },
    "atlas_label": {
        "status": "deferred",
        "description": "Configuration scaffold only; atlas-label execution is not supported yet.",
    },
    "data_driven_hook": {
        "status": "deferred",
        "description": "Configuration scaffold only; custom-hook execution is not supported yet.",
    },
}
EXTRACTION_SET_TEMPLATE_METADATA: dict[str, dict[str, str]] = {
    "generic_nifti": {
        "status": "local_nifti",
        "description": "Local NIfTI extraction; requires user-provided value maps and ROI masks.",
    },
    "fsl_featquery": {
        "status": "external_runtime",
        "description": "FSL featquery extraction; requires existing FEAT inputs, ROI masks, and featquery.",
    },
}
DEFAULT_PATH_PROFILE = "generic"
LOSO_FLAME1_RUNTIME_ROOT_REF = "dataset_derivatives_root"
LOSO_FLAME1_RUNTIME_PATH = ".research-platform/roi-loso-flame1-runtime"
PATH_PROFILE_REGISTRY: dict[str, dict[str, Any]] = {
    "generic": {
        "roi_sets": {},
        "extraction_sets": {},
        "extraction_targets": {},
    },
    "research_platform_fsl_ffx": {
        "roi_sets": {
            "loso_group_map": {
                "fixed_effects_inputs": {
                    "root_ref": None,
                    "root": "${ROI_FEAT_ROOT:-}",
                    "cope_dir": "{subject_dir}/{session_dir}/{subject_dir}_{session_dir}_task-{task_id}_dir-{direction}_desc-{model}FFX.gfeat/cope{cope_number}.feat",
                    "cope_image": "stats/cope1.nii.gz",
                    "varcope_image": "stats/varcope1.nii.gz",
                    "mask_image": "mask.nii.gz",
                },
                "group_mask": {
                    "root_ref": None,
                    "path": "${ROI_FEAT_ROOT:-}/higher_level/group_{session_dir}_task-{task_id}_dir-{direction}_desc-{model}FFX_FLAME1/cope{cope_number}.gfeat/mask.nii.gz",
                },
                "outputs": {"root_ref": LOSO_FLAME1_RUNTIME_ROOT_REF, "path": LOSO_FLAME1_RUNTIME_PATH},
                "publication": {
                    "enabled": True,
                    "layout": "loso_flame1_bidslike",
                    "root": {"root_ref": "dataset_derivatives_root", "path": "roi-loso-flame1"},
                    "dataset_description": {
                        "name": "ROI LOSO FLAME1 outputs",
                        "generated_by_name": "roi-loso-flame1",
                    },
                    "map_desc": "{model}LOSOFlame1",
                    "mask_desc": "{model}LOSOFlame1Sphere{sphere_radius_mm}mm",
                },
                "missing_input_policy": "warn",
            },
        },
        "extraction_sets": {
            "fsl_featquery": {
                "outputs": {
                    "root_ref": LOSO_FLAME1_RUNTIME_ROOT_REF,
                    "path": LOSO_FLAME1_RUNTIME_PATH,
                    "format": "tsv",
                },
                "publication": {
                    "enabled": True,
                    "layout": "loso_flame1_bidslike",
                    "root": {"root_ref": "dataset_derivatives_root", "path": "roi-loso-flame1"},
                    "dataset_description": {
                        "name": "ROI LOSO FLAME1 outputs",
                        "generated_by_name": "roi-loso-flame1",
                    },
                    "table_desc": "{model}LOSOFlame1Featquery",
                },
            },
        },
        "extraction_targets": {
            "fsl_featquery": [
                {
                    "inputs": {
                        "root_ref": None,
                        "feat_dir": "${ROI_FEAT_ROOT:-}/{subject_dir}/{session_dir}/{subject_dir}_{session_dir}_task-{task_id}_dir-{direction}_desc-{model}FFX.gfeat/cope{cope}.feat",
                        "value_image": "stats/cope1.nii.gz",
                    },
                    "featquery_output_name": "fq_loso_{roi_label}_{source_contrast}_cope{cope}",
                    "missing": {"feat_dir": "warn", "roi_mask": "warn", "report_values": "warn"},
                }
            ],
        },
    },
}

_SAFE_SCALAR = re.compile(r"^[A-Za-z0-9_./${}{}-]+$")
_NUMERIC_SCALAR = re.compile(r"-?\d+(\.\d+)?")
_SUBJECT_TOKEN = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*?)(\d+)$")
_SIMPLE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_ROI_ENTITY_OVERRIDE_FIELDS = ("session", "task", "direction", "model", "space", "resolution")
_ROI_NUMERIC_OVERRIDE_FIELDS = ("search_radius_mm", "sphere_radius_mm", "z_threshold")
_ROI_INTEGER_OVERRIDE_FIELDS = ("min_voxels_warn", "min_voxels_fail")
_CONTRAST_ERROR = "--contrast must use id:cope_number[:desc], for example condition_a_gt_b:1:ConditionAGtB."
_ROI_ERROR = "--roi must use label:contrast_id:x,y,z[:desc], for example ExampleSeed:condition_a_gt_b:-2,-58,64:ConditionAGtB."
_SUBJECTS_ERROR = "--subjects must be a comma-separated list or inclusive range like sub-101:sub-103."
_HELD_OUT_SUBJECTS_ERROR = "--held-out-subjects must be same-as-subjects, a comma-separated list, or an inclusive range like sub-101:sub-103."
_METRIC_ERROR = "--metric must be a non-empty metric name; repeat --metric for multiple metrics."
_ROI_LABEL_ERROR = "--roi-label must be a non-empty BIDS-like label containing only letters and numbers."


def supported_path_profiles() -> tuple[str, ...]:
    """Return supported scaffold-time path profile names."""

    return tuple(sorted(PATH_PROFILE_REGISTRY))


def supported_roi_set_templates() -> tuple[str, ...]:
    """Return discoverable ROI-set scaffold template names."""

    return tuple(sorted(ROI_SET_TEMPLATES))


def supported_extraction_set_templates() -> tuple[str, ...]:
    """Return discoverable extraction-set scaffold template names."""

    return tuple(sorted(EXTRACTION_SET_TEMPLATES))


def roi_set_template_help() -> str:
    """Return concise runtime/prerequisite help for ROI-set templates."""

    return _template_help(ROI_SET_TEMPLATE_METADATA)


def extraction_set_template_help() -> str:
    """Return concise runtime/prerequisite help for extraction templates."""

    return _template_help(EXTRACTION_SET_TEMPLATE_METADATA)


def build_roi_set_document(
    name: str,
    template: str,
    overrides: Mapping[str, Any] | None = None,
    *,
    path_profile: str | None = DEFAULT_PATH_PROFILE,
) -> dict[str, Any]:
    """Return a generic ROI set document for a supported scaffold template."""

    if template not in ROI_SET_TEMPLATES:
        supported = ", ".join(sorted(ROI_SET_TEMPLATES))
        raise ValueError(f"Unsupported ROI scaffold template {template!r}. Supported templates: {supported}.")
    builder = {
        "coordinate_sphere": _coordinate_sphere_roi_set,
        "manual_mask": _manual_mask_roi_set,
        "atlas_label": _atlas_label_roi_set,
        "functional_threshold_map": _functional_threshold_map_roi_set,
        "loso_group_map": _loso_group_map_roi_set,
        "data_driven_hook": _data_driven_hook_roi_set,
    }[template]
    document = apply_path_profile(
        {"roi_set": builder(name)},
        scaffold_kind="roi_set",
        template=template,
        path_profile=path_profile,
    )
    if overrides:
        document = apply_roi_set_overrides(document, overrides)
    return document


def build_extraction_set_document(
    name: str,
    *,
    roi_set: str,
    template: str,
    overrides: Mapping[str, Any] | None = None,
    path_profile: str | None = DEFAULT_PATH_PROFILE,
) -> dict[str, Any]:
    """Return a generic ROI extraction set document for a scaffold template."""

    if template not in EXTRACTION_SET_TEMPLATES:
        supported = ", ".join(sorted(EXTRACTION_SET_TEMPLATES))
        raise ValueError(f"Unsupported ROI extraction scaffold template {template!r}. Supported templates: {supported}.")
    builder = {
        "generic_nifti": _generic_nifti_extraction_set,
        "fsl_featquery": _fsl_featquery_extraction_set,
    }[template]
    document = apply_path_profile(
        {"extraction_set": builder(name, roi_set=roi_set)},
        scaffold_kind="extraction_set",
        template=template,
        path_profile=path_profile,
    )
    if overrides:
        document = apply_extraction_set_overrides(document, overrides)
    return document


def validate_roi_set_scaffold(document: Mapping[str, Any]) -> list[str]:
    """Validate a scaffolded ROI set with the reusable ROI schema rules."""

    return validate_roi_set_document(document)


def validate_extraction_set_scaffold(document: Mapping[str, Any]) -> list[str]:
    """Validate a scaffolded extraction set with the reusable ROI schema rules."""

    return validate_extraction_set_document(document)


def render_yaml(document: Mapping[str, Any]) -> str:
    """Render a stable, dependency-free YAML subset for scaffold files."""

    return "\n".join(_dump_yaml_lines(document, indent=0)) + "\n"


def apply_path_profile(
    document: Mapping[str, Any],
    *,
    scaffold_kind: str,
    template: str,
    path_profile: str | None = DEFAULT_PATH_PROFILE,
) -> dict[str, Any]:
    """Return a scaffold document with reusable path-profile defaults merged in."""

    profile_name = _normalize_path_profile(path_profile)
    updated = copy.deepcopy(dict(document))
    if profile_name == DEFAULT_PATH_PROFILE:
        return updated

    profile = PATH_PROFILE_REGISTRY[profile_name]
    if scaffold_kind == "roi_set":
        defaults = profile.get("roi_sets", {}).get(template)
        if isinstance(defaults, Mapping):
            roi_set = _payload_mapping(updated, "roi_set")
            updated["roi_set"] = _merge_profile_defaults(roi_set, defaults)
            _ensure_named_runtime_output_path(updated["roi_set"])
        return updated

    if scaffold_kind == "extraction_set":
        extraction_set = _payload_mapping(updated, "extraction_set")
        defaults = profile.get("extraction_sets", {}).get(template)
        if isinstance(defaults, Mapping):
            extraction_set = _merge_profile_defaults(extraction_set, defaults)
            updated["extraction_set"] = extraction_set
            _ensure_named_runtime_output_path(extraction_set)
        target_defaults = profile.get("extraction_targets", {}).get(template)
        if isinstance(target_defaults, Sequence) and not isinstance(target_defaults, (str, bytes)):
            _merge_extraction_target_profile_defaults(extraction_set, target_defaults)
        return updated

    raise ValueError(f"Unsupported scaffold kind {scaffold_kind!r}.")


def _normalize_path_profile(path_profile: str | None) -> str:
    profile_name = _clean_text(path_profile) or DEFAULT_PATH_PROFILE
    if profile_name not in PATH_PROFILE_REGISTRY:
        supported = ", ".join(supported_path_profiles())
        raise ValueError(f"Unsupported ROI path profile {profile_name!r}. Supported profiles: {supported}.")
    return profile_name


def _merge_profile_defaults(base: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in defaults.items():
        if value is None:
            merged.pop(key, None)
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_profile_defaults(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _ensure_named_runtime_output_path(payload: dict[str, Any]) -> None:
    outputs = payload.get("outputs")
    name = _clean_text(payload.get("name"))
    if not isinstance(outputs, dict) or name is None:
        return
    path = _clean_text(outputs.get("path"))
    if path is None:
        return
    parts = path.rstrip("/").split("/")
    if parts[-1] != name:
        outputs["path"] = f"{path.rstrip('/')}/{name}"


def _template_help(metadata: Mapping[str, Mapping[str, str]]) -> str:
    return " ".join(
        f"{name}: {metadata[name]['description']}"
        for name in sorted(metadata)
    )


def _merge_extraction_target_profile_defaults(
    extraction_set: dict[str, Any],
    target_defaults: Sequence[Mapping[str, Any]],
) -> None:
    targets = extraction_set.get("targets")
    if not isinstance(targets, list):
        return
    for index, defaults in enumerate(target_defaults):
        if index >= len(targets) or not isinstance(defaults, Mapping):
            continue
        target = targets[index]
        if isinstance(target, Mapping):
            targets[index] = _merge_profile_defaults(target, defaults)


def apply_roi_set_overrides(document: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a scaffolded ROI set with user-friendly init overrides applied."""

    updated = copy.deepcopy(dict(document))
    roi_set = _payload_mapping(updated, "roi_set")
    normalized = _normalize_roi_set_overrides(overrides)

    subjects_supplied = "subjects" in normalized
    if subjects_supplied:
        _apply_subject_override(
            roi_set,
            normalized["subjects"],
            single_entity_templates=_SINGLE_ENTITY_ROI_SET_TEMPLATES,
            multi_entity_templates=_MULTI_ENTITY_ROI_SET_TEMPLATES,
            scaffold_kind="ROI",
        )

    for field in _ROI_ENTITY_OVERRIDE_FIELDS:
        if field in normalized:
            roi_set[field] = normalized[field]

    if "contrasts" in normalized:
        roi_set["contrasts"] = _contrast_dicts(normalized["contrasts"], cope_key="cope_number")

    if "held_out_subjects" in normalized:
        if normalized["held_out_subjects"] == "same-as-subjects":
            roi_set["held_out_subjects"] = _subjects_from_roi_set(roi_set)
        else:
            roi_set["held_out_subjects"] = list(normalized["held_out_subjects"])
    elif subjects_supplied and roi_set.get("provenance", {}).get("scaffold") == "loso_group_map":
        roi_set["held_out_subjects"] = list(normalized["subjects"])

    if subjects_supplied and roi_set.get("provenance", {}).get("scaffold") == "loso_group_map":
        roi_set["min_group_n"] = _infer_loso_min_group_n(
            subjects=_subjects_from_roi_set(roi_set),
            held_out_subjects=_held_out_subjects_from_roi_set(roi_set),
        )

    roi_numeric_defaults = {
        key: normalized[key]
        for key in (*_ROI_NUMERIC_OVERRIDE_FIELDS, *_ROI_INTEGER_OVERRIDE_FIELDS)
        if key in normalized
    }
    if "rois" in normalized:
        roi_set["rois"] = _build_override_rois(roi_set, normalized["rois"], roi_numeric_defaults)
    else:
        existing_rois = _roi_list(roi_set)
        if "contrasts" in normalized:
            _retarget_roi_contrasts(existing_rois, roi_set)
        if roi_numeric_defaults:
            _apply_roi_numeric_defaults(existing_rois, roi_numeric_defaults)

    return updated


def apply_extraction_set_overrides(document: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a scaffolded extraction set with user-friendly init overrides applied."""

    updated = copy.deepcopy(dict(document))
    extraction_set = _payload_mapping(updated, "extraction_set")
    normalized = _normalize_extraction_set_overrides(overrides)

    if "subjects" in normalized:
        _apply_subject_override(
            extraction_set,
            normalized["subjects"],
            single_entity_templates=_SINGLE_ENTITY_EXTRACTION_SET_TEMPLATES,
            multi_entity_templates=_MULTI_ENTITY_EXTRACTION_SET_TEMPLATES,
            scaffold_kind="extraction",
        )

    for field in _ROI_ENTITY_OVERRIDE_FIELDS:
        if field in normalized:
            extraction_set[field] = normalized[field]

    targets = _target_list(extraction_set)
    if "metrics" in normalized:
        for target in targets:
            target["metrics"] = list(normalized["metrics"])
            _sync_featquery_percent_signal_change_config(target, metrics=normalized["metrics"])

    if "contrasts" in normalized:
        contrast_dicts = _contrast_dicts(normalized["contrasts"], cope_key="cope")
        for target in targets:
            target["contrasts"] = copy.deepcopy(contrast_dicts)

    if "roi_labels" in normalized:
        for target in targets:
            target["roi_labels"] = list(normalized["roi_labels"])

    return updated


def _normalize_roi_set_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if "subjects" in overrides:
        normalized["subjects"] = _parse_subject_spec(overrides.get("subjects"), error_message=_SUBJECTS_ERROR)
    if "held_out_subjects" in overrides:
        held_out = _required_text(overrides.get("held_out_subjects"), error_message=_HELD_OUT_SUBJECTS_ERROR)
        if held_out == "same-as-subjects":
            normalized["held_out_subjects"] = held_out
        else:
            normalized["held_out_subjects"] = _parse_subject_spec(held_out, error_message=_HELD_OUT_SUBJECTS_ERROR)

    for field in _ROI_ENTITY_OVERRIDE_FIELDS:
        if field in overrides:
            normalized[field] = _required_text(overrides.get(field), error_message=f"--{field.replace('_', '-')} must be a non-empty value.")

    for field in _ROI_NUMERIC_OVERRIDE_FIELDS:
        if field in overrides:
            normalized[field] = _parse_number(overrides.get(field), flag=f"--{field.replace('_', '-')}")

    for field in _ROI_INTEGER_OVERRIDE_FIELDS:
        if field in overrides:
            normalized[field] = _parse_integer(overrides.get(field), flag=f"--{field.replace('_', '-')}")

    if "contrasts" in overrides:
        normalized["contrasts"] = _parse_contrasts(overrides.get("contrasts"))
    if "rois" in overrides:
        normalized["rois"] = _parse_rois(overrides.get("rois"))
    return normalized


def _normalize_extraction_set_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if "subjects" in overrides:
        normalized["subjects"] = _parse_subject_spec(overrides.get("subjects"), error_message=_SUBJECTS_ERROR)

    for field in _ROI_ENTITY_OVERRIDE_FIELDS:
        if field in overrides:
            normalized[field] = _required_text(overrides.get(field), error_message=f"--{field.replace('_', '-')} must be a non-empty value.")

    if "metrics" in overrides:
        normalized["metrics"] = _parse_metrics(overrides.get("metrics"))
    if "contrasts" in overrides:
        normalized["contrasts"] = _parse_contrasts(overrides.get("contrasts"))
    if "roi_labels" in overrides:
        normalized["roi_labels"] = _parse_roi_labels(overrides.get("roi_labels"))
    return normalized


def _parse_subject_spec(value: Any, *, error_message: str) -> list[str]:
    text = _required_text(value, error_message=error_message)
    tokens = text.split(",")
    if any(token.strip() == "" for token in tokens):
        raise ValueError(error_message)
    subjects: list[str] = []
    for token in tokens:
        token = token.strip()
        if ":" not in token:
            _prefix, digits = _parse_subject_token(token, error_message=error_message)
            subjects.append(token[: len(token) - len(digits)] + digits)
            continue
        if token.count(":") != 1:
            raise ValueError(error_message)
        start_text, end_text = (part.strip() for part in token.split(":", 1))
        start_prefix, start_digits = _parse_subject_token(start_text, error_message=error_message)
        end_prefix, end_digits = _parse_subject_token(end_text, error_message=error_message)
        if start_prefix != end_prefix or len(start_digits) != len(end_digits):
            raise ValueError(error_message)
        start = int(start_digits)
        end = int(end_digits)
        if start > end:
            raise ValueError(error_message)
        subjects.extend(f"{start_prefix}{index:0{len(start_digits)}d}" for index in range(start, end + 1))
    return subjects


def _parse_subject_token(value: str, *, error_message: str) -> tuple[str, str]:
    match = _SUBJECT_TOKEN.fullmatch(value)
    if match is None:
        raise ValueError(error_message)
    return match.group(1), match.group(2)


def _apply_subject_override(
    payload: dict[str, Any],
    subjects: Sequence[str],
    *,
    single_entity_templates: frozenset[str],
    multi_entity_templates: frozenset[str],
    scaffold_kind: str,
) -> None:
    provenance = payload.get("provenance")
    template = _clean_text(provenance.get("scaffold")) if isinstance(provenance, Mapping) else None
    if template in single_entity_templates:
        if len(subjects) != 1:
            raise ValueError(
                f"The {template} {scaffold_kind} scaffold represents one configured subject and "
                "does not perform multi-subject or Cartesian expansion; --subjects must resolve "
                "to exactly one value. Multi-value lists and ranges are supported only by "
                "loso_group_map ROI and fsl_featquery extraction scaffolds."
            )
        payload["subject"] = subjects[0]
        payload.pop("subjects", None)
        payload.pop("subject_ids", None)
        return
    if template in multi_entity_templates:
        payload["subjects"] = list(subjects)
        payload.pop("subject", None)
        payload.pop("subject_id", None)
        return
    expected = ", ".join(sorted(single_entity_templates | multi_entity_templates))
    raise ValueError(
        f"Cannot apply --subjects because the {scaffold_kind} scaffold identity is missing or "
        f"unsupported; expected one of: {expected}."
    )


def _parse_contrasts(value: Any) -> list[dict[str, Any]]:
    contrasts: list[dict[str, Any]] = []
    for raw in _repeat_values(value):
        text = _required_text(raw, error_message=_CONTRAST_ERROR)
        parts = text.split(":", 2)
        if len(parts) not in {2, 3}:
            raise ValueError(_CONTRAST_ERROR)
        contrast_id = _clean_text(parts[0])
        cope_text = _clean_text(parts[1])
        desc = _clean_text(parts[2]) if len(parts) == 3 else None
        if contrast_id is None or cope_text is None or (len(parts) == 3 and desc is None):
            raise ValueError(_CONTRAST_ERROR)
        if not _SIMPLE_ID.fullmatch(contrast_id):
            raise ValueError(_CONTRAST_ERROR)
        try:
            cope_number = int(cope_text)
        except ValueError as exc:
            raise ValueError(_CONTRAST_ERROR) from exc
        if cope_number <= 0:
            raise ValueError(_CONTRAST_ERROR)
        contrast: dict[str, Any] = {"id": contrast_id, "cope_number": cope_number}
        if desc is not None:
            contrast["desc"] = desc
        contrasts.append(contrast)
    _reject_duplicate_values([str(contrast["id"]) for contrast in contrasts], error_message="--contrast values must use unique ids.")
    return contrasts


def _parse_rois(value: Any) -> list[dict[str, Any]]:
    rois: list[dict[str, Any]] = []
    for raw in _repeat_values(value):
        text = _required_text(raw, error_message=_ROI_ERROR)
        parts = text.split(":", 3)
        if len(parts) not in {3, 4}:
            raise ValueError(_ROI_ERROR)
        label = _clean_text(parts[0])
        contrast_id = _clean_text(parts[1])
        coordinate_text = _clean_text(parts[2])
        desc = _clean_text(parts[3]) if len(parts) == 4 else None
        if label is None or contrast_id is None or coordinate_text is None or (len(parts) == 4 and desc is None):
            raise ValueError(_ROI_ERROR)
        if not _is_bids_label(label):
            raise ValueError("--roi label must be a non-empty BIDS-like label containing only letters and numbers.")
        if not _SIMPLE_ID.fullmatch(contrast_id):
            raise ValueError(_ROI_ERROR)
        coordinates = coordinate_text.split(",")
        if len(coordinates) != 3 or any(part.strip() == "" for part in coordinates):
            raise ValueError(_ROI_ERROR)
        try:
            parsed_coordinates = [_parse_coordinate_number(part.strip()) for part in coordinates]
        except ValueError as exc:
            raise ValueError(_ROI_ERROR) from exc
        if desc is not None and not _is_bids_label(desc):
            raise ValueError("--roi desc must be a BIDS-like label containing only letters and numbers.")
        rois.append({"label": label, "contrast_id": contrast_id, "coordinate": parsed_coordinates, "desc": desc})
    _reject_duplicate_values([str(roi["label"]) for roi in rois], error_message="--roi values must use unique labels.")
    return rois


def _parse_metrics(value: Any) -> list[str]:
    metrics: list[str] = []
    for raw in _repeat_values(value):
        metric = _clean_text(raw)
        if metric is None:
            raise ValueError(_METRIC_ERROR)
        metrics.append(metric)
    return metrics


def _sync_featquery_percent_signal_change_config(target: dict[str, Any], *, metrics: Sequence[str]) -> None:
    if _clean_text(target.get("backend")) != "fsl_featquery":
        return
    if "percent_signal_change" in set(metrics):
        featquery = target.get("featquery")
        if not isinstance(featquery, dict):
            featquery = {}
        featquery["include_percent_signal_change"] = True
        target["featquery"] = featquery
        target.pop("include_percent_signal_change", None)
        return
    target.pop("include_percent_signal_change", None)
    featquery = target.get("featquery")
    if isinstance(featquery, dict):
        featquery.pop("include_percent_signal_change", None)
        if not featquery:
            target.pop("featquery", None)


def _parse_roi_labels(value: Any) -> list[str]:
    labels: list[str] = []
    for raw in _repeat_values(value):
        label = _clean_text(raw)
        if label is None or not _is_bids_label(label):
            raise ValueError(_ROI_LABEL_ERROR)
        labels.append(label)
    _reject_duplicate_values(labels, error_message="--roi-label values must be unique.")
    return labels


def _contrast_dicts(contrasts: Sequence[Mapping[str, Any]], *, cope_key: str) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for contrast in contrasts:
        item: dict[str, Any] = {"id": contrast["id"], cope_key: contrast["cope_number"]}
        if contrast.get("desc") is not None:
            item["desc"] = contrast["desc"]
        rendered.append(item)
    return rendered


def _build_override_rois(
    roi_set: Mapping[str, Any],
    rois: Sequence[Mapping[str, Any]],
    numeric_defaults: Mapping[str, Any],
) -> list[dict[str, Any]]:
    contrast_descs = _contrast_descs(roi_set)
    contrast_ids = set(contrast_descs)
    rendered: list[dict[str, Any]] = []
    base_rois = _roi_list(roi_set)
    for index, roi in enumerate(rois):
        contrast_id = str(roi["contrast_id"])
        if contrast_id not in contrast_ids:
            raise ValueError(f"--roi references unknown contrast_id {contrast_id!r}.")
        base = base_rois[index] if index < len(base_rois) else (base_rois[0] if base_rois else {})
        rendered_roi = _roi_from_base(base, roi, contrast_desc=contrast_descs[contrast_id])
        rendered.append(rendered_roi)
    _apply_roi_numeric_defaults(rendered, numeric_defaults)
    return rendered


def _roi_from_base(base: Mapping[str, Any], override: Mapping[str, Any], *, contrast_desc: str | None) -> dict[str, Any]:
    roi = dict(base)
    label = str(override["label"])
    family = _clean_text(roi.get("family")) or "loso_group_map"
    roi["label"] = label
    desc = _clean_text(override.get("desc")) or (contrast_desc if contrast_desc and _is_bids_label(contrast_desc) else label)
    roi["desc"] = desc
    roi["contrast"] = override["contrast_id"]
    if family == "coordinate_sphere":
        roi.pop("seed_coordinate", None)
        roi["coordinate"] = list(override["coordinate"])
    else:
        roi.pop("coordinate", None)
        roi["seed_coordinate"] = list(override["coordinate"])
    return roi


def _apply_roi_numeric_defaults(rois: Sequence[dict[str, Any]], defaults: Mapping[str, Any]) -> None:
    for roi in rois:
        for key, value in defaults.items():
            roi[key] = value
            if key == "sphere_radius_mm" and roi.get("family") == "coordinate_sphere":
                roi["radius_mm"] = value


def _retarget_roi_contrasts(rois: Sequence[dict[str, Any]], roi_set: Mapping[str, Any]) -> None:
    contrast_descs = _contrast_descs(roi_set)
    contrast_ids = list(contrast_descs)
    if not contrast_ids:
        return
    for index, roi in enumerate(rois):
        if not any(key in roi for key in ("contrast", "contrast_id", "source_contrast")):
            continue
        contrast_id = contrast_ids[index] if index < len(contrast_ids) else contrast_ids[0]
        roi["contrast"] = contrast_id
        if "contrast_id" in roi:
            roi["contrast_id"] = contrast_id
        if "source_contrast" in roi:
            roi["source_contrast"] = contrast_id
        desc = contrast_descs.get(contrast_id)
        if desc and _is_bids_label(desc):
            roi["desc"] = desc


def _contrast_descs(roi_set: Mapping[str, Any]) -> dict[str, str | None]:
    raw = roi_set.get("contrasts")
    if not isinstance(raw, list) or not raw:
        return {}
    contrast_descs: dict[str, str | None] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        contrast_id = _clean_text(item.get("id") or item.get("name") or item.get("contrast_id") or item.get("contrast"))
        if contrast_id is None:
            continue
        contrast_descs[contrast_id] = _clean_text(item.get("desc") or item.get("contrast_desc"))
    return contrast_descs


def _subjects_from_roi_set(roi_set: Mapping[str, Any]) -> list[str]:
    subjects = roi_set.get("subjects") or roi_set.get("subject_ids")
    if isinstance(subjects, list) and subjects and all(_clean_text(subject) is not None for subject in subjects):
        return [str(subject).strip() for subject in subjects]
    raise ValueError("--held-out-subjects same-as-subjects requires subjects in the template or --subjects.")


def _held_out_subjects_from_roi_set(roi_set: Mapping[str, Any]) -> list[str]:
    held_out_subjects = roi_set.get("held_out_subjects")
    if (
        isinstance(held_out_subjects, list)
        and held_out_subjects
        and all(_clean_text(subject) is not None for subject in held_out_subjects)
    ):
        return [str(subject).strip() for subject in held_out_subjects]
    raise ValueError("LOSO scaffold min_group_n inference requires held_out_subjects.")


def _infer_loso_min_group_n(*, subjects: Sequence[str], held_out_subjects: Sequence[str]) -> int:
    subject_count = len(subjects)
    subject_set = set(subjects)
    training_counts = [
        subject_count - 1 if held_out_subject in subject_set else subject_count
        for held_out_subject in held_out_subjects
    ]
    return max(1, min(training_counts, default=subject_count))


def _roi_list(roi_set: Mapping[str, Any]) -> list[dict[str, Any]]:
    rois = roi_set.get("rois")
    if not isinstance(rois, list):
        return []
    return [roi for roi in rois if isinstance(roi, dict)]


def _target_list(extraction_set: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets = extraction_set.get("targets")
    if not isinstance(targets, list):
        return []
    return [target for target in targets if isinstance(target, dict)]


def _payload_mapping(document: Mapping[str, Any], key: str) -> dict[str, Any]:
    payload = document.get(key) if key in document else document
    if not isinstance(payload, dict):
        raise ValueError(f"{key} must contain a mapping.")
    return payload


def _repeat_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _required_text(value: Any, *, error_message: str) -> str:
    text = _clean_text(value)
    if text is None:
        raise ValueError(error_message)
    return text


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_number(value: Any, *, flag: str) -> int | float:
    text = _required_text(value, error_message=f"{flag} must be numeric.")
    if not re.fullmatch(r"-?\d+(\.\d+)?", text):
        raise ValueError(f"{flag} must be numeric.")
    number = float(text)
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    return number


def _parse_integer(value: Any, *, flag: str) -> int:
    text = _required_text(value, error_message=f"{flag} must be an integer.")
    if not re.fullmatch(r"-?\d+", text):
        raise ValueError(f"{flag} must be an integer.")
    return int(text)


def _parse_coordinate_number(value: str) -> int | float:
    if not re.fullmatch(r"-?\d+(\.\d+)?", value):
        raise ValueError(value)
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    return float(value)


def _is_bids_label(value: str) -> bool:
    from research_platform.neuro.roi import is_bids_label_value

    return is_bids_label_value(value)


def _reject_duplicate_values(values: Sequence[str], *, error_message: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(error_message)
        seen.add(value)


def _common_entities() -> dict[str, Any]:
    return {
        "subject": "sub-001",
        "session": "ses-01",
        "task": "exampletask",
        "direction": "AP",
        "space": "MNI152NLin6Asym",
        "resolution": "2",
    }


def _common_outputs(name: str, *, include_format: bool = False) -> dict[str, Any]:
    outputs: dict[str, Any] = {
        "root_ref": "artifacts_root",
        "path": f"roi-runtime/{name}",
    }
    if include_format:
        outputs["format"] = "tsv"
    return outputs


def _loso_flame1_runtime_outputs(name: str, *, include_format: bool = False) -> dict[str, Any]:
    outputs: dict[str, Any] = {
        "root_ref": LOSO_FLAME1_RUNTIME_ROOT_REF,
        "path": f"{LOSO_FLAME1_RUNTIME_PATH}/{name}",
    }
    if include_format:
        outputs["format"] = "tsv"
    return outputs


def _loso_flame1_roi_runtime_defaults() -> dict[str, Any]:
    return {
        "existing_output": "fail",
        "cleanup": {"after_roi_build": "roi_runtime", "after_extraction": "none"},
    }


def _fsl_featquery_runtime_defaults() -> dict[str, Any]:
    return {
        "existing_output": "fail",
        "cleanup": {"after_extraction": "extraction_runtime"},
    }


def _local_runtime_defaults() -> dict[str, Any]:
    return {"existing_output": "fail"}


def _project_input(path: str) -> dict[str, str]:
    return {"root_ref": "project_root", "path": path}


def _coordinate_sphere_roi_set(name: str) -> dict[str, Any]:
    return {
        "name": name,
        **_common_entities(),
        "outputs": _common_outputs(name),
        "runtime": _local_runtime_defaults(),
        "rois": [
            {
                "label": "SeedA",
                "family": "coordinate_sphere",
                "backend": "generic_nifti",
                "desc": "SeedA",
                "reference_image": _project_input("inputs/roi/example_reference.nii.gz"),
                "coordinate": [0, 0, 0],
                "sphere_radius_mm": 6,
                "radius_mm": 6,
            },
            {
                "label": "SeedB",
                "family": "coordinate_sphere",
                "backend": "generic_nifti",
                "desc": "SeedB",
                "reference_image": _project_input("inputs/roi/example_reference.nii.gz"),
                "coordinate": [6, 0, 0],
                "sphere_radius_mm": 6,
                "radius_mm": 6,
            },
        ],
        "provenance": {"schema_version": "1", "scaffold": "coordinate_sphere"},
    }


def _manual_mask_roi_set(name: str) -> dict[str, Any]:
    return {
        "name": name,
        **_common_entities(),
        "outputs": _common_outputs(name),
        "runtime": _local_runtime_defaults(),
        "rois": [
            {
                "label": "ManualMask",
                "family": "manual_mask",
                "backend": "manual",
                "desc": "ManualMask",
                "source": _project_input("inputs/roi/example_mask.nii.gz"),
                "reference_image": _project_input("inputs/roi/example_reference.nii.gz"),
            }
        ],
        "provenance": {"schema_version": "1", "scaffold": "manual_mask"},
    }


def _atlas_label_roi_set(name: str) -> dict[str, Any]:
    return {
        "name": name,
        **_common_entities(),
        "outputs": _common_outputs(name),
        "runtime": _local_runtime_defaults(),
        "rois": [
            {
                "label": "AtlasLabel",
                "family": "atlas_label",
                "desc": "AtlasLabel",
                "atlas": "ExampleAtlas",
                "atlas_image": _project_input("inputs/roi/example_atlas.nii.gz"),
                "atlas_space": "MNI152NLin6Asym",
                "labels": [1, 2],
            }
        ],
        "provenance": {"schema_version": "1", "scaffold": "atlas_label"},
    }


def _functional_threshold_map_roi_set(name: str) -> dict[str, Any]:
    return {
        "name": name,
        **_common_entities(),
        "outputs": _common_outputs(name),
        "runtime": _local_runtime_defaults(),
        "rois": [
            {
                "label": "FunctionalPeakSeed",
                "family": "functional_threshold_map",
                "backend": "generic_nifti",
                "desc": "ThresholdMap",
                "stat_map": _project_input("inputs/roi/example_stat_map.nii.gz"),
                "seed_coordinate": [0, 0, 0],
                "search_radius_mm": 12,
                "sphere_radius_mm": 6,
                "z_threshold": 3.1,
                "exploratory_z_threshold": 2.3,
                "allow_below_threshold_fallback": True,
                "min_voxels_warn": 12,
                "min_voxels_fail": 4,
            }
        ],
        "provenance": {"schema_version": "1", "scaffold": "functional_threshold_map"},
    }


def _loso_group_map_roi_set(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "backend": "fsl_flame1",
        "subjects": ["sub-001", "sub-002"],
        "held_out_subjects": ["sub-001", "sub-002"],
        "session": "ses-01",
        "task": "exampletask",
        "direction": "AP",
        "model": "ModelA",
        "space": "MNI152NLin6Asym",
        "resolution": "2",
        "min_group_n": 1,
        "outputs": _loso_flame1_runtime_outputs(name),
        "runtime": _loso_flame1_roi_runtime_defaults(),
        "publication": {
            "enabled": True,
            "layout": "loso_flame1_bidslike",
            "root": {"root_ref": "dataset_derivatives_root", "path": "roi-loso-flame1"},
            "dataset_description": {
                "name": "ROI LOSO FLAME1 outputs",
                "generated_by_name": "roi-loso-flame1",
            },
            "map_desc": "{model}LOSOFlame1",
            "mask_desc": "{model}LOSOFlame1Sphere{sphere_radius_mm}mm",
        },
        "fixed_effects_inputs": {
            "root_ref": "project_root",
            "cope_dir": "inputs/roi/fixed-effects/{subject_dir}/{session_dir}/func/task-{task_id}_model-{model}_contrast-{contrast_id}",
            "cope_image": "cope{cope_number}.nii.gz",
            "varcope_image": "varcope{cope_number}.nii.gz",
            "mask_image": "mask.nii.gz",
        },
        "group_mask": {
            "root_ref": "project_root",
            "path": "inputs/roi/group/{session_dir}/func/task-{task_id}_model-{model}/cope{cope_number}.gfeat/mask.nii.gz",
        },
        "contrasts": [
            {"id": "ContrastA", "cope_number": 1, "desc": "ContrastA"},
            {"id": "ContrastB", "cope_number": 2, "desc": "ContrastB"},
        ],
        "cache": {"reuse": True},
        "rois": [
            {
                "label": "SeedA",
                "family": "loso_group_map",
                "backend": "fsl_flame1",
                "desc": "ContrastA",
                "contrast": "ContrastA",
                "seed_coordinate": [0, 0, 0],
                "search_radius_mm": 12,
                "sphere_radius_mm": 6,
                "z_threshold": 3.1,
                "allow_below_threshold_fallback": True,
                "min_voxels_warn": 12,
                "min_voxels_fail": 4,
            },
            {
                "label": "SeedB",
                "family": "loso_group_map",
                "backend": "fsl_flame1",
                "desc": "ContrastB",
                "contrast": "ContrastB",
                "seed_coordinate": [6, 0, 0],
                "search_radius_mm": 12,
                "sphere_radius_mm": 6,
                "z_threshold": 3.1,
                "allow_below_threshold_fallback": True,
                "min_voxels_warn": 12,
                "min_voxels_fail": 4,
            },
        ],
        "provenance": {"schema_version": "1", "scaffold": "loso_group_map"},
    }


def _data_driven_hook_roi_set(name: str) -> dict[str, Any]:
    return {
        "name": name,
        **_common_entities(),
        "outputs": _common_outputs(name),
        "runtime": _local_runtime_defaults(),
        "rois": [
            {
                "label": "HookGeneratedSeed",
                "family": "data_driven_hook",
                "backend": "custom_hook",
                "desc": "HookSeed",
                "hook": "example_roi_hook",
                "hook_config": "example_roi_hook_config",
            }
        ],
        "provenance": {"schema_version": "1", "scaffold": "data_driven_hook"},
    }


def _generic_nifti_extraction_set(name: str, *, roi_set: str) -> dict[str, Any]:
    return {
        "name": name,
        "roi_set": roi_set,
        **_common_entities(),
        "outputs": _common_outputs(name, include_format=True),
        "runtime": _local_runtime_defaults(),
        "targets": [
            {
                "name": "GenericNiftiValues",
                "backend": "generic_nifti",
                "desc": "GenericValues",
                "metrics": ["mean", "median", "voxel_count"],
                "inputs": _project_input("inputs/roi/example_value_map.nii.gz"),
                "roi_labels": ["SeedA", "SeedB"],
            }
        ],
        "provenance": {"schema_version": "1", "scaffold": "generic_nifti"},
    }


def _fsl_featquery_extraction_set(name: str, *, roi_set: str) -> dict[str, Any]:
    return {
        "name": name,
        "roi_set": roi_set,
        "subjects": ["sub-001", "sub-002"],
        "session": "ses-01",
        "task": "exampletask",
        "direction": "AP",
        "model": "ModelA",
        "space": "MNI152NLin6Asym",
        "resolution": "2",
        "roi_mask_source": {"source": "roi_set_publication"},
        "outputs": _loso_flame1_runtime_outputs(name, include_format=True),
        "runtime": _fsl_featquery_runtime_defaults(),
        "targets": [
            {
                "name": "FslFeatqueryValues",
                "backend": "fsl_featquery",
                "desc": "ModelAFeatquery",
                "metrics": ["mean_cope", "roi_voxel_count"],
                "inputs": {
                    "root_ref": "project_root",
                    "feat_dir": "inputs/roi/feat/{subject_dir}/{session_dir}/func/sub-{subject_id}_{session_dir}_task-{task_id}_model-{model}.feat",
                    "value_image": "stats/cope{cope}.nii.gz",
                },
                "contrasts": [
                    {"id": "ContrastA", "cope": 1, "desc": "ContrastA"},
                    {"id": "ContrastB", "cope": 2, "desc": "ContrastB"},
                ],
                "roi_labels": ["SeedA", "SeedB"],
                "featquery_output_name": "fq_{roi_label}_{source_contrast}_cope{cope}",
                "missing": {"feat_dir": "warn", "roi_mask": "warn", "report_values": "warn"},
            }
        ],
        "provenance": {"schema_version": "1", "scaffold": "fsl_featquery"},
    }


def _dump_yaml_lines(value: Any, *, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(value, Mapping):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                if not item:
                    lines.append(f"{prefix}{key}: {{}}")
                    continue
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_yaml_lines(item, indent=indent + 2))
                continue
            if isinstance(item, list):
                if not item:
                    lines.append(f"{prefix}{key}: []")
                    continue
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_yaml_lines(item, indent=indent + 2))
                continue
            lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, Mapping):
                if not item:
                    lines.append(f"{prefix}- {{}}")
                    continue
                lines.append(f"{prefix}-")
                lines.extend(_dump_yaml_lines(item, indent=indent + 2))
                continue
            if isinstance(item, list):
                if not item:
                    lines.append(f"{prefix}- []")
                    continue
                lines.append(f"{prefix}-")
                lines.extend(_dump_yaml_lines(item, indent=indent + 2))
                continue
            lines.append(f"{prefix}- {_format_scalar(item)}")
        return lines
    return [f"{prefix}{_format_scalar(value)}"]


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text:
        return '""'
    if text.lower() in {"true", "false", "null", "~"} or _NUMERIC_SCALAR.fullmatch(text):
        return _quote(text)
    if _SAFE_SCALAR.fullmatch(text):
        return text
    return _quote(text)


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
