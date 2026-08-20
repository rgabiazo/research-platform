"""Reusable ROI workflow schemas and validation helpers.

Phase 1 defines configuration and provenance contracts only. The helpers in
this module do not import neuroimaging runtimes and do not create masks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import re

from research_platform.neuro._roi_path_safety import (
    configured_path_is_unsafe,
    published_text_contains_local_path_reference,
    published_value_local_path_fields,
)


ROI_FAMILIES = frozenset(
    {
        "manual_mask",
        "coordinate_sphere",
        "atlas_label",
        "functional_threshold_map",
        "loso_group_map",
        "data_driven_hook",
    }
)

ROI_BACKENDS = frozenset(
    {
        "manual",
        "generic_nifti",
        "fsl",
        "fsl_featquery",
        "fsl_flame1",
        "nilearn",
        "freesurfer",
        "ants",
        "custom",
        "custom_hook",
    }
)

ROI_FALLBACK_STATUSES = frozenset({"thresholded", "below_threshold_fallback", "not_applicable"})
MASK_INTERSECTION_POLICIES = frozenset({"none", "intersection", "union", "exclude", "custom"})
ROI_CLEANUP_AFTER_ROI_BUILD_VALUES = frozenset({"none", "cache_only", "roi_runtime"})
ROI_CLEANUP_AFTER_EXTRACTION_VALUES = frozenset({"none", "roi_runtime"})
EXTRACTION_CLEANUP_AFTER_EXTRACTION_VALUES = frozenset({"none", "extraction_runtime"})
ROI_MASK_SOURCE_VALUES = frozenset({"roi_set_runtime", "roi_set_publication"})
ROI_PUBLICATION_EXISTING_OUTPUT_VALUES = frozenset({"fail", "replace"})
ROI_RUNTIME_EXISTING_OUTPUT_VALUES = frozenset({"fail", "replace"})

_BIDS_LABEL_VALUE = re.compile(r"^[A-Za-z0-9]+$")
_COORDINATE_FIELDS = frozenset({"coordinate", "seed_coordinate", "full_sample_seed_coordinate", "loso_peak_coordinate"})
_RADIUS_FIELDS = frozenset({"radius", "radius_mm", "sphere_radius", "sphere_radius_mm", "search_radius", "search_radius_mm"})
_LABEL_FIELDS = frozenset({"label", "desc", "roi_label", "roi_desc", "output_desc"})
_NAMED_ROOT_FIELDS = frozenset({"root_ref", "output_root_ref", "feat_root_ref"})
_CONFIGURED_PATH_CONTAINER_FIELDS = frozenset(
    {
        "atlas_image",
        "coverage_masks",
        "fixed_effects_inputs",
        "group_mask",
        "inputs",
        "mask",
        "masks",
        "reference_image",
        "roi_masks",
        "search_mask",
        "search_mask_image",
        "source",
        "source_mask",
        "stat_map",
        "value_map",
        "value_maps",
    }
)


@dataclass(frozen=True)
class RoiDefinition:
    """A reusable ROI definition without execution behavior."""

    label: str
    family: str
    desc: str | None = None
    backend: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoiSet:
    """Named collection of ROI definitions."""

    name: str
    rois: tuple[RoiDefinition, ...]
    desc: str | None = None
    backend: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionTarget:
    """A configured target for ROI extraction."""

    name: str
    backend: str | None = None
    desc: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionSet:
    """Named ROI extraction configuration."""

    name: str
    roi_set: str | None
    targets: tuple[ExtractionTarget, ...]
    backend: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoiSidecarProvenance:
    """JSON sidecar payload describing a generated or planned ROI mask."""

    roi_label: str
    roi_family: str
    fields: Mapping[str, Any] = field(default_factory=dict)


def parse_roi_set_document(document: Mapping[str, Any], *, validate_personal_paths: bool = True) -> RoiSet:
    """Parse a validated ROI set document into a small data model."""

    errors = validate_roi_set_document(document, validate_personal_paths=validate_personal_paths)
    if errors:
        raise ValueError("; ".join(errors))
    roi_set = _payload_mapping(document, "roi_set")
    rois = tuple(
        RoiDefinition(
            label=str(roi["label"]).strip(),
            family=str(roi["family"]).strip(),
            desc=_optional_text(roi.get("desc")),
            backend=_optional_text(roi.get("backend")),
            fields=dict(roi),
        )
        for roi in _roi_definitions(roi_set)
    )
    return RoiSet(
        name=str(roi_set["name"]).strip(),
        desc=_optional_text(roi_set.get("desc")),
        backend=_optional_text(roi_set.get("backend")),
        rois=rois,
        provenance=roi_set.get("provenance", {}) if isinstance(roi_set.get("provenance"), Mapping) else {},
        fields=dict(roi_set),
    )


def parse_extraction_set_document(
    document: Mapping[str, Any],
    *,
    validate_personal_paths: bool = True,
) -> ExtractionSet:
    """Parse a validated extraction set document into a small data model."""

    errors = validate_extraction_set_document(document, validate_personal_paths=validate_personal_paths)
    if errors:
        raise ValueError("; ".join(errors))
    extraction_set = _payload_mapping(document, "extraction_set")
    targets = tuple(
        ExtractionTarget(
            name=str(target["name"]).strip(),
            backend=_optional_text(target.get("backend")),
            desc=_optional_text(target.get("desc")),
            fields=dict(target),
        )
        for target in _extraction_targets(extraction_set)
    )
    return ExtractionSet(
        name=str(extraction_set["name"]).strip(),
        roi_set=_optional_text(extraction_set.get("roi_set") or extraction_set.get("roi_set_ref")),
        backend=_optional_text(extraction_set.get("backend")),
        targets=targets,
        provenance=extraction_set.get("provenance", {}) if isinstance(extraction_set.get("provenance"), Mapping) else {},
        fields=dict(extraction_set),
    )


def parse_roi_sidecar_document(document: Mapping[str, Any]) -> RoiSidecarProvenance:
    """Parse a validated ROI sidecar/provenance document."""

    errors = validate_roi_sidecar_document(document)
    if errors:
        raise ValueError("; ".join(errors))
    sidecar = _sidecar_payload(document)
    return RoiSidecarProvenance(
        roi_label=str(sidecar["roi_label"]).strip(),
        roi_family=str(sidecar["roi_family"]).strip(),
        fields=dict(sidecar),
    )


def validate_roi_set_document(
    document: Mapping[str, Any] | Any,
    *,
    validate_personal_paths: bool = True,
    personal_path_document: Mapping[str, Any] | Any | None = None,
) -> list[str]:
    """Validate a reusable ROI set document.

    The validator checks config shape and constraints only. It intentionally
    does not check whether referenced files or neuroimaging tools exist.
    When a caller validates an env-resolved document, personal_path_document can
    carry the raw config so placeholder roots are checked before expansion.
    """

    if not isinstance(document, Mapping):
        return ["ROI set document must contain a mapping."]
    errors: list[str] = []
    roi_set = _payload_mapping(document, "roi_set", errors=errors)
    if not roi_set:
        return errors

    name = _optional_text(roi_set.get("name"))
    if name is None:
        errors.append("roi_set.name must be defined.")

    _validate_backend_value(roi_set.get("backend"), "roi_set.backend", errors)
    _validate_label_fields(roi_set, "roi_set", errors)

    provenance = roi_set.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        errors.append("roi_set.provenance must contain a mapping when declared.")

    rois = _roi_definitions(roi_set, errors=errors)
    labels: list[str] = []
    for index, roi in enumerate(rois):
        label = f"roi_set.rois[{index}]"
        roi_label = _optional_text(roi.get("label"))
        if roi_label is None:
            errors.append(f"{label}.label must be defined.")
        else:
            labels.append(roi_label)
            _validate_bids_label_value(roi_label, f"{label}.label", errors)
        family = _optional_text(roi.get("family"))
        if family is None:
            errors.append(f"{label}.family must be defined.")
        elif family not in ROI_FAMILIES:
            errors.append(f"{label}.family must be one of: {', '.join(sorted(ROI_FAMILIES))}.")
        _validate_backend_value(roi.get("backend"), f"{label}.backend", errors)
        _validate_label_fields(roi, label, errors)
        _validate_common_numeric_rules(roi, label, errors)

    for duplicate in _duplicates(labels):
        errors.append(f"roi_set.rois contains duplicate ROI label: {duplicate}.")

    _validate_common_numeric_rules(roi_set, "roi_set", errors)
    _validate_named_root_path_specs(roi_set, "roi_set", errors)
    _validate_roi_runtime_cleanup(roi_set, "roi_set", errors)
    _validate_publication(roi_set, "roi_set", errors)
    personal_path_payload = roi_set
    if personal_path_document is not None:
        if not isinstance(personal_path_document, Mapping):
            errors.append("ROI set personal-path document must contain a mapping.")
            personal_path_payload = {}
        else:
            personal_path_payload = _payload_mapping(personal_path_document, "roi_set", errors=errors)
    _validate_runtime_output_paths(
        personal_path_payload,
        "roi_set",
        errors,
        reject_absolute_roots=validate_personal_paths,
    )
    for index, roi in enumerate(_roi_definitions(personal_path_payload)):
        _validate_runtime_output_paths(
            roi,
            f"roi_set.rois[{index}]",
            errors,
            reject_absolute_roots=validate_personal_paths,
        )
    _validate_configured_path_containers(
        personal_path_payload,
        "roi_set",
        errors,
        reject_absolute=validate_personal_paths,
    )
    if validate_personal_paths:
        _validate_no_personal_paths(personal_path_payload, "roi_set", errors)
    return errors


def validate_extraction_set_document(
    document: Mapping[str, Any] | Any,
    *,
    validate_personal_paths: bool = True,
    personal_path_document: Mapping[str, Any] | Any | None = None,
) -> list[str]:
    """Validate an ROI extraction set document without running extraction.

    When a caller validates an env-resolved document, personal_path_document can
    carry the raw config so placeholder roots are checked before expansion.
    """

    if not isinstance(document, Mapping):
        return ["Extraction set document must contain a mapping."]
    errors: list[str] = []
    extraction_set = _payload_mapping(document, "extraction_set", errors=errors)
    if not extraction_set:
        return errors

    if _optional_text(extraction_set.get("name")) is None:
        errors.append("extraction_set.name must be defined.")
    has_explicit_masks = any(
        isinstance(target, Mapping) and (target.get("roi_masks") is not None or target.get("masks") is not None)
        for target in _extraction_targets(extraction_set)
    )
    if _optional_text(extraction_set.get("roi_set") or extraction_set.get("roi_set_ref")) is None and not has_explicit_masks:
        errors.append("extraction_set.roi_set or extraction_set.roi_set_ref must reference a ROI set name unless targets define explicit roi_masks.")
    _validate_backend_value(extraction_set.get("backend"), "extraction_set.backend", errors)
    _validate_label_fields(extraction_set, "extraction_set", errors)

    provenance = extraction_set.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        errors.append("extraction_set.provenance must contain a mapping when declared.")

    targets = _extraction_targets(extraction_set, errors=errors)
    for index, target in enumerate(targets):
        label = f"extraction_set.targets[{index}]"
        if _optional_text(target.get("name")) is None:
            errors.append(f"{label}.name must be defined.")
        _validate_backend_value(target.get("backend"), f"{label}.backend", errors)
        _validate_label_fields(target, label, errors)
        _validate_common_numeric_rules(target, label, errors)
        _validate_roi_mask_source(target, label, errors)

    _validate_common_numeric_rules(extraction_set, "extraction_set", errors)
    _validate_named_root_path_specs(extraction_set, "extraction_set", errors)
    _validate_roi_mask_source(extraction_set, "extraction_set", errors)
    _validate_extraction_runtime_cleanup(extraction_set, "extraction_set", errors)
    _validate_publication(extraction_set, "extraction_set", errors)
    personal_path_payload = extraction_set
    if personal_path_document is not None:
        if not isinstance(personal_path_document, Mapping):
            errors.append("Extraction set personal-path document must contain a mapping.")
            personal_path_payload = {}
        else:
            personal_path_payload = _payload_mapping(personal_path_document, "extraction_set", errors=errors)
    _validate_runtime_output_paths(
        personal_path_payload,
        "extraction_set",
        errors,
        reject_absolute_roots=validate_personal_paths,
    )
    for index, target in enumerate(_extraction_targets(personal_path_payload)):
        _validate_runtime_output_paths(
            target,
            f"extraction_set.targets[{index}]",
            errors,
            reject_absolute_roots=validate_personal_paths,
        )
    _validate_configured_path_containers(
        personal_path_payload,
        "extraction_set",
        errors,
        reject_absolute=validate_personal_paths,
    )
    if validate_personal_paths:
        _validate_no_personal_paths(personal_path_payload, "extraction_set", errors)
    return errors


def validate_roi_sidecar_document(document: Mapping[str, Any] | Any) -> list[str]:
    """Validate an ROI JSON sidecar/provenance payload."""

    if not isinstance(document, Mapping):
        return ["ROI sidecar document must contain a mapping."]
    errors: list[str] = []
    sidecar = _sidecar_payload(document)
    if not isinstance(sidecar, Mapping):
        return ["ROI sidecar provenance must contain a mapping."]

    roi_label = _optional_text(sidecar.get("roi_label"))
    if roi_label is None:
        errors.append("roi_label must be defined.")
    else:
        _validate_bids_label_value(roi_label, "roi_label", errors)

    roi_family = _optional_text(sidecar.get("roi_family"))
    if roi_family is None:
        errors.append("roi_family must be defined.")
    elif roi_family not in ROI_FAMILIES:
        errors.append(f"roi_family must be one of: {', '.join(sorted(ROI_FAMILIES))}.")

    _validate_backend_value(sidecar.get("backend"), "backend", errors)
    _validate_label_fields(sidecar, "sidecar", errors)
    _validate_common_numeric_rules(sidecar, "sidecar", errors)

    for field_name in _COORDINATE_FIELDS:
        if field_name in sidecar:
            _validate_coordinate(sidecar[field_name], field_name, errors)

    voxel_count = sidecar.get("voxel_count")
    if voxel_count is not None:
        try:
            normalized = int(voxel_count)
        except (TypeError, ValueError):
            errors.append("voxel_count must be an integer.")
        else:
            if normalized < 0:
                errors.append("voxel_count must be greater than or equal to zero.")

    fallback_status = _optional_text(sidecar.get("fallback_status"))
    if fallback_status is not None and fallback_status not in ROI_FALLBACK_STATUSES:
        errors.append(f"fallback_status must be one of: {', '.join(sorted(ROI_FALLBACK_STATUSES))}.")

    policy = _optional_text(sidecar.get("mask_intersection_policy"))
    if policy is not None and policy not in MASK_INTERSECTION_POLICIES:
        errors.append(f"mask_intersection_policy must be one of: {', '.join(sorted(MASK_INTERSECTION_POLICIES))}.")

    for list_field in ("warnings", "qc_flags"):
        if list_field in sidecar and not _is_string_list(sidecar[list_field]):
            errors.append(f"{list_field} must contain a list of strings.")

    _validate_no_personal_paths(sidecar, "sidecar", errors)
    return errors


def summarize_roi_set_document(document: Mapping[str, Any]) -> str:
    errors = validate_roi_set_document(document)
    if errors:
        return f"ROI set: invalid\nValidation issues: {len(errors)}"
    roi_set = parse_roi_set_document(document)
    families = sorted({roi.family for roi in roi_set.rois})
    return "\n".join(
        [
            f"ROI set: {roi_set.name}",
            f"ROIs ({len(roi_set.rois)}): {', '.join(roi.label for roi in roi_set.rois)}",
            f"Families: {', '.join(families)}",
        ]
    )


def summarize_extraction_set_document(document: Mapping[str, Any]) -> str:
    errors = validate_extraction_set_document(document)
    if errors:
        return f"ROI extraction set: invalid\nValidation issues: {len(errors)}"
    extraction_set = parse_extraction_set_document(document)
    backends = sorted({target.backend for target in extraction_set.targets if target.backend})
    return "\n".join(
        [
            f"ROI extraction set: {extraction_set.name}",
            f"ROI set: {extraction_set.roi_set or '<explicit masks>'}",
            f"Targets ({len(extraction_set.targets)}): {', '.join(target.name for target in extraction_set.targets)}",
            f"Backends: {', '.join(backends) if backends else '<unspecified>'}",
        ]
    )


def is_bids_label_value(value: str) -> bool:
    """Return whether a value is safe for BIDS-like label/desc entities."""

    return bool(_BIDS_LABEL_VALUE.fullmatch(value))


def provenance_path_reference(
    path: str | Path,
    *,
    env_roots: Mapping[str, str | Path] | None = None,
    root_refs: Mapping[str, str | Path] | None = None,
    hash_unmapped_personal: bool = True,
) -> str:
    """Return a personal-path-safe reference for ROI sidecar provenance."""

    resolved = Path(path).expanduser().resolve()
    candidates = _provenance_path_reference_candidates(resolved, env_roots=env_roots, root_refs=root_refs)

    if candidates:
        return max(candidates)[2]

    text = str(resolved)
    if hash_unmapped_personal and _contains_personal_path(text):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"unresolved_path:{resolved.name}:sha256-{digest}"
    return text


def normalize_sidecar_provenance(
    provenance: Mapping[str, Any],
    *,
    env_roots: Mapping[str, str | Path] | None = None,
    root_refs: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Recursively rewrite configured-root paths in generated sidecar provenance."""

    normalized = _normalize_provenance_value(provenance, env_roots=env_roots, root_refs=root_refs)
    return dict(normalized) if isinstance(normalized, Mapping) else {}


def validate_portable_provenance_paths(payload: Any, *, label: str = "provenance") -> list[str]:
    """Return path-portability errors for generated provenance payloads."""

    errors: list[str] = []
    _validate_no_personal_paths(payload, label, errors)
    return errors


def runtime_existing_output_policy(document: Mapping[str, Any], *, payload_key: str) -> str:
    """Return the configured runtime collision policy, defaulting to ``fail``."""

    if not isinstance(document, Mapping):
        raise ValueError("ROI runtime policy document must contain a mapping.")
    payload = document.get(payload_key, document)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{payload_key} must contain a mapping.")
    runtime = payload.get("runtime")
    if runtime is None:
        return "fail"
    if not isinstance(runtime, Mapping):
        raise ValueError(f"{payload_key}.runtime must contain a mapping when declared.")
    policy = _optional_text(runtime.get("existing_output")) or "fail"
    if policy not in ROI_RUNTIME_EXISTING_OUTPUT_VALUES:
        allowed = ", ".join(sorted(ROI_RUNTIME_EXISTING_OUTPUT_VALUES))
        raise ValueError(f"{payload_key}.runtime.existing_output must be one of: {allowed}.")
    return policy


def _payload_mapping(document: Mapping[str, Any], key: str, *, errors: list[str] | None = None) -> Mapping[str, Any]:
    if key in document:
        payload = document.get(key)
    else:
        payload = document
    if not isinstance(payload, Mapping):
        if errors is not None:
            errors.append(f"{key} must contain a mapping.")
        return {}
    return payload


def _sidecar_payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    if "roi_provenance" in document:
        payload = document.get("roi_provenance")
    elif "provenance" in document and isinstance(document.get("provenance"), Mapping):
        payload = document.get("provenance")
    else:
        payload = document
    return payload if isinstance(payload, Mapping) else {}


def _roi_definitions(roi_set: Mapping[str, Any], *, errors: list[str] | None = None) -> list[Mapping[str, Any]]:
    raw_rois = roi_set.get("rois", roi_set.get("roi_definitions"))
    if not isinstance(raw_rois, list) or not raw_rois:
        if errors is not None:
            errors.append("roi_set.rois must define at least one ROI mapping.")
        return []
    rois: list[Mapping[str, Any]] = []
    for index, raw_roi in enumerate(raw_rois):
        if not isinstance(raw_roi, Mapping):
            if errors is not None:
                errors.append(f"roi_set.rois[{index}] must contain a mapping.")
            continue
        rois.append(raw_roi)
    return rois


def _extraction_targets(extraction_set: Mapping[str, Any], *, errors: list[str] | None = None) -> list[Mapping[str, Any]]:
    raw_targets = extraction_set.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        if errors is not None:
            errors.append("extraction_set.targets must define at least one target mapping.")
        return []
    targets: list[Mapping[str, Any]] = []
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, Mapping):
            if errors is not None:
                errors.append(f"extraction_set.targets[{index}] must contain a mapping.")
            continue
        targets.append(raw_target)
    return targets


def _validate_backend_value(value: Any, label: str, errors: list[str]) -> None:
    backend = _optional_text(value)
    if backend is None:
        return
    if backend not in ROI_BACKENDS:
        errors.append(f"{label} must be one of: {', '.join(sorted(ROI_BACKENDS))}.")


def _validate_roi_runtime_cleanup(payload: Mapping[str, Any], label: str, errors: list[str]) -> None:
    runtime = _runtime_payload(payload, label, errors)
    if runtime is None:
        return
    _validate_runtime_existing_output(runtime, label, errors)
    cleanup = _runtime_cleanup_payload(runtime, label, errors)
    if cleanup is None:
        return
    _validate_choice(
        cleanup.get("after_roi_build", "none"),
        f"{label}.runtime.cleanup.after_roi_build",
        ROI_CLEANUP_AFTER_ROI_BUILD_VALUES,
        errors,
    )
    _validate_choice(
        cleanup.get("after_extraction", "none"),
        f"{label}.runtime.cleanup.after_extraction",
        ROI_CLEANUP_AFTER_EXTRACTION_VALUES,
        errors,
    )


def _validate_extraction_runtime_cleanup(payload: Mapping[str, Any], label: str, errors: list[str]) -> None:
    runtime = _runtime_payload(payload, label, errors)
    if runtime is None:
        return
    _validate_runtime_existing_output(runtime, label, errors)
    cleanup = _runtime_cleanup_payload(runtime, label, errors)
    if cleanup is None:
        return
    _validate_choice(
        cleanup.get("after_extraction", "none"),
        f"{label}.runtime.cleanup.after_extraction",
        EXTRACTION_CLEANUP_AFTER_EXTRACTION_VALUES,
        errors,
    )


def _validate_publication(payload: Mapping[str, Any], label: str, errors: list[str]) -> None:
    if "publication" not in payload:
        return
    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        errors.append(f"{label}.publication must contain a mapping when declared.")
        return
    if "existing_output" in publication:
        _validate_choice(
            publication.get("existing_output"),
            f"{label}.publication.existing_output",
            ROI_PUBLICATION_EXISTING_OUTPUT_VALUES,
            errors,
        )
    root = publication.get("root") or publication.get("output_root")
    if isinstance(root, Mapping):
        subpath = root.get("path") or root.get("subpath")
        if subpath is not None and configured_path_is_unsafe(str(subpath)):
            errors.append(
                f"{label}.publication.root.path must be relative and remain beneath its configured root."
            )


def _validate_runtime_output_paths(
    payload: Mapping[str, Any],
    label: str,
    errors: list[str],
    *,
    reject_absolute_roots: bool,
) -> None:
    for key in ("derivative_root", "output_root"):
        if key in payload:
            _validate_runtime_output_root_spec(
                payload.get(key),
                f"{label}.{key}",
                errors,
                reject_absolute=reject_absolute_roots,
            )

    if "outputs" not in payload:
        return
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        errors.append(f"{label}.outputs must contain a mapping when declared.")
        return

    for key in ("path", "subpath"):
        if key not in outputs:
            continue
        value = outputs.get(key)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            errors.append(f"{label}.outputs.{key} must be a non-empty relative path when declared.")
        elif configured_path_is_unsafe(str(value)):
            errors.append(
                f"{label}.outputs.{key} must be a relative path that remains beneath its configured root."
            )

    for key in ("derivative_root", "root", "output_root"):
        if key in outputs:
            _validate_runtime_output_root_spec(
                outputs.get(key),
                f"{label}.outputs.{key}",
                errors,
                reject_absolute=reject_absolute_roots,
            )


def _validate_runtime_output_root_spec(
    value: Any,
    label: str,
    errors: list[str],
    *,
    reject_absolute: bool,
) -> None:
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        if not text:
            errors.append(f"{label} must be a non-empty path string or mapping when declared.")
        elif _has_parent_path_component(text) or (reject_absolute and configured_path_is_unsafe(text)):
            errors.append(
                f"{label} must use a safe project-relative, environment-root, or named-root path reference."
            )
        return
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be a path string or mapping when declared.")
        return

    has_declared_root_ref = "root_ref" in value
    path_key: str | None = None
    if value.get("pattern") is not None:
        path_key = "pattern"
    elif "path" in value:
        path_key = "path"
    elif "pattern" in value:
        path_key = "pattern"
    if not has_declared_root_ref and path_key is None:
        errors.append(f"{label} must define path, pattern, or root_ref.")
        return
    if path_key is None:
        return

    path_value = value.get(path_key)
    path_label = f"{label}.{path_key}"
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        errors.append(f"{path_label} must be a non-empty path when declared.")
        return
    text = str(path_value)
    if has_declared_root_ref:
        if configured_path_is_unsafe(text):
            errors.append(
                f"{path_label} must be a relative path that remains beneath its configured root."
            )
    elif _has_parent_path_component(text) or (reject_absolute and configured_path_is_unsafe(text)):
        errors.append(
            f"{path_label} must use a safe project-relative, environment-root, or named-root path reference."
        )


def _validate_named_root_path_specs(value: Any, label: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key in _NAMED_ROOT_FIELDS:
            if key in value and (
                not isinstance(value.get(key), str) or not str(value.get(key)).strip()
            ):
                errors.append(f"{label}.{key} must be a non-empty string when declared.")
        has_named_root = any(
            _optional_text(value.get(key)) is not None
            for key in _NAMED_ROOT_FIELDS
        )
        if has_named_root:
            for key, child in value.items():
                if key in _NAMED_ROOT_FIELDS:
                    continue
                if label.endswith(".outputs") and key in {"path", "subpath"}:
                    continue
                _validate_named_root_field(child, f"{label}.{key}", errors)
        for key, child in value.items():
            _validate_named_root_path_specs(child, f"{label}.{key}", errors)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_named_root_path_specs(child, f"{label}[{index}]", errors)


def _validate_configured_path_containers(
    value: Any,
    label: str,
    errors: list[str],
    *,
    reject_absolute: bool,
) -> None:
    """Reject unsafe strings inside fields that are interpreted as paths.

    Raw YAML is used when available so environment-root placeholders stay
    portable even when the runtime document has expanded them to an absolute
    local path. Parent traversal remains invalid in either representation.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if key in _CONFIGURED_PATH_CONTAINER_FIELDS:
                _validate_configured_path_container(
                    child,
                    child_label,
                    errors,
                    reject_absolute=reject_absolute,
                )
            else:
                _validate_configured_path_containers(
                    child,
                    child_label,
                    errors,
                    reject_absolute=reject_absolute,
                )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_configured_path_containers(
                child,
                f"{label}[{index}]",
                errors,
                reject_absolute=reject_absolute,
            )


def _validate_configured_path_container(
    value: Any,
    label: str,
    errors: list[str],
    *,
    reject_absolute: bool,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _NAMED_ROOT_FIELDS:
                continue
            _validate_configured_path_container(
                child,
                f"{label}.{key}",
                errors,
                reject_absolute=reject_absolute,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_configured_path_container(
                child,
                f"{label}[{index}]",
                errors,
                reject_absolute=reject_absolute,
            )
        return
    if not isinstance(value, (str, Path)):
        return
    text = str(value)
    if _has_parent_path_component(text) or (reject_absolute and configured_path_is_unsafe(text)):
        errors.append(
            f"{label} must use a safe project-relative, environment-root, or named-root path reference."
        )


def _has_parent_path_component(value: str) -> bool:
    return any(component == ".." for component in re.split(r"[\\/]", str(value).strip()))


def _validate_named_root_field(value: Any, label: str, errors: list[str]) -> None:
    if isinstance(value, (str, Path)):
        if configured_path_is_unsafe(str(value)):
            errors.append(
                f"{label} must be a relative path that remains beneath its configured root."
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_named_root_field(child, f"{label}.{key}", errors)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_named_root_field(child, f"{label}[{index}]", errors)


def _validate_roi_mask_source(payload: Mapping[str, Any], label: str, errors: list[str]) -> None:
    if "roi_mask_source" not in payload or payload.get("roi_mask_source") is None:
        return
    source = payload.get("roi_mask_source")
    if isinstance(source, Mapping):
        if "source" not in source:
            errors.append(f"{label}.roi_mask_source.source must be one of: {', '.join(sorted(ROI_MASK_SOURCE_VALUES))}.")
            return
        _validate_choice(
            source.get("source"),
            f"{label}.roi_mask_source.source",
            ROI_MASK_SOURCE_VALUES,
            errors,
        )
        return
    _validate_choice(source, f"{label}.roi_mask_source", ROI_MASK_SOURCE_VALUES, errors)


def _runtime_payload(payload: Mapping[str, Any], label: str, errors: list[str]) -> Mapping[str, Any] | None:
    runtime = payload.get("runtime")
    if runtime is None:
        return None
    if not isinstance(runtime, Mapping):
        errors.append(f"{label}.runtime must contain a mapping when declared.")
        return None
    return runtime


def _runtime_cleanup_payload(runtime: Mapping[str, Any], label: str, errors: list[str]) -> Mapping[str, Any] | None:
    cleanup = runtime.get("cleanup")
    if cleanup is None:
        return None
    if not isinstance(cleanup, Mapping):
        errors.append(f"{label}.runtime.cleanup must contain a mapping when declared.")
        return None
    return cleanup


def _validate_runtime_existing_output(runtime: Mapping[str, Any], label: str, errors: list[str]) -> None:
    if "existing_output" not in runtime:
        return
    _validate_choice(
        runtime.get("existing_output"),
        f"{label}.runtime.existing_output",
        ROI_RUNTIME_EXISTING_OUTPUT_VALUES,
        errors,
    )


def _validate_choice(value: Any, label: str, allowed: frozenset[str], errors: list[str]) -> None:
    text = _optional_text(value)
    if text is None:
        errors.append(f"{label} must be one of: {', '.join(sorted(allowed))}.")
        return
    if text not in allowed:
        errors.append(f"{label} must be one of: {', '.join(sorted(allowed))}.")


def _validate_label_fields(payload: Mapping[str, Any], label: str, errors: list[str]) -> None:
    for key, value in payload.items():
        if key not in _LABEL_FIELDS:
            continue
        text = _optional_text(value)
        if text is None:
            continue
        _validate_bids_label_value(text, f"{label}.{key}", errors)


def _validate_bids_label_value(value: str, label: str, errors: list[str]) -> None:
    if not is_bids_label_value(value):
        errors.append(f"{label} must contain only letters and numbers for BIDS-like label/desc use.")


def _validate_common_numeric_rules(payload: Any, label: str, errors: list[str]) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child_label = f"{label}.{key}"
            if key in _RADIUS_FIELDS:
                _validate_positive_number(value, child_label, errors)
            _validate_common_numeric_rules(value, child_label, errors)

        z_threshold = _optional_number(payload.get("z_threshold"))
        exploratory = _optional_number(payload.get("exploratory_z_threshold"))
        if z_threshold is not None and exploratory is not None and z_threshold < exploratory:
            errors.append(f"{label}.z_threshold must be greater than or equal to exploratory_z_threshold.")

        min_fail = _optional_int(payload.get("min_voxels_fail"))
        min_warn = _optional_int(payload.get("min_voxels_warn"))
        if min_fail is not None and min_fail < 0:
            errors.append(f"{label}.min_voxels_fail must be greater than or equal to zero.")
        if min_warn is not None and min_warn < 0:
            errors.append(f"{label}.min_voxels_warn must be greater than or equal to zero.")
        if min_fail is not None and min_warn is not None and min_fail > min_warn:
            errors.append(f"{label}.min_voxels_fail must be less than or equal to min_voxels_warn.")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _validate_common_numeric_rules(value, f"{label}[{index}]", errors)


def _validate_positive_number(value: Any, label: str, errors: list[str]) -> None:
    number = _optional_number(value)
    if number is None:
        errors.append(f"{label} must be numeric.")
    elif number <= 0:
        errors.append(f"{label} must be greater than zero.")


def _validate_coordinate(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) != 3:
        errors.append(f"{label} must contain exactly three numeric coordinates.")
        return
    for index, coordinate in enumerate(value):
        if _optional_number(coordinate) is None:
            errors.append(f"{label}[{index}] must be numeric.")


def _validate_no_personal_paths(payload: Any, label: str, errors: list[str]) -> None:
    for field in published_value_local_path_fields(payload, label=label):
        errors.append(f"{field} contains a personal absolute path; use a project/env-root reference instead.")


def _normalize_provenance_value(
    value: Any,
    *,
    env_roots: Mapping[str, str | Path] | None,
    root_refs: Mapping[str, str | Path] | None,
) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _normalize_provenance_value(child, env_roots=env_roots, root_refs=root_refs)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_normalize_provenance_value(item, env_roots=env_roots, root_refs=root_refs) for item in value]
    if isinstance(value, tuple):
        return [_normalize_provenance_value(item, env_roots=env_roots, root_refs=root_refs) for item in value]
    if isinstance(value, Path):
        return provenance_path_reference(
            value,
            env_roots=env_roots,
            root_refs=root_refs,
            hash_unmapped_personal=False,
        )
    if isinstance(value, str):
        return _normalize_provenance_string(value, env_roots=env_roots, root_refs=root_refs)
    return value


def _normalize_provenance_string(
    value: str,
    *,
    env_roots: Mapping[str, str | Path] | None,
    root_refs: Mapping[str, str | Path] | None,
) -> str:
    normalized = value
    for _, _, root_text, prefix in _provenance_root_replacements(env_roots=env_roots, root_refs=root_refs):
        normalized = _replace_root_occurrences(normalized, root_text, prefix)
    if normalized != value:
        return normalized

    stripped = value.strip()
    if configured_path_is_unsafe(stripped):
        return provenance_path_reference(
            stripped,
            env_roots=env_roots,
            root_refs=root_refs,
            hash_unmapped_personal=False,
        )
    return value


def _provenance_path_reference_candidates(
    path: Path,
    *,
    env_roots: Mapping[str, str | Path] | None,
    root_refs: Mapping[str, str | Path] | None,
) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []
    for score, priority, root_text, prefix in _provenance_root_replacements(env_roots=env_roots, root_refs=root_refs):
        reference = _path_reference_under_root(path, Path(root_text), prefix)
        if reference is not None:
            candidates.append((score, priority, reference))
    return candidates


def _provenance_root_replacements(
    *,
    env_roots: Mapping[str, str | Path] | None,
    root_refs: Mapping[str, str | Path] | None,
) -> list[tuple[int, int, str, str]]:
    replacements: list[tuple[int, int, str, str]] = []
    for index, (env_name, root) in enumerate((env_roots or {}).items()):
        lexical_root = Path(root).expanduser()
        for root_path in dict.fromkeys((lexical_root, lexical_root.resolve(strict=False))):
            if len(root_path.parts) <= 1:
                continue
            replacements.append((len(root_path.parts), 1000 - index, root_path.as_posix(), f"${{{env_name}:-}}"))
    for index, (name, root) in enumerate((root_refs or {}).items()):
        lexical_root = Path(root).expanduser()
        for root_path in dict.fromkeys((lexical_root, lexical_root.resolve(strict=False))):
            if len(root_path.parts) <= 1:
                continue
            replacements.append((len(root_path.parts), -index, root_path.as_posix(), f"root_ref:{name}"))
    return sorted(replacements, reverse=True)


def _replace_root_occurrences(value: str, root_text: str, prefix: str) -> str:
    pattern = re.compile(rf"(?<![A-Za-z0-9_{{}}$]){re.escape(root_text)}(?=$|[/\s'\"),;\]])")
    return pattern.sub(prefix, value)


def _path_reference_under_root(path: Path, root: Path, prefix: str) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    relative_text = relative.as_posix()
    return prefix if relative_text in {"", "."} else f"{prefix}/{relative_text}"


def _contains_personal_path(value: str) -> bool:
    return published_text_contains_local_path_reference(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
