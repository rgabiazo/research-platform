"""MVPA extension contracts for generic analysis workflow recipes.

The contracts here validate and plan neuro/MVPA-specific recipe metadata only.
They do not discover FEAT directories, parse designs, load masks or NIfTI files,
transform ROIs, create ROIs, compute distances, run FSL/ANTs, or write outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any
import re

from .config import ALLOWED_CV_UNITS, ALLOWED_NOISE_NORMALIZATION_METHODS


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_LABEL_OR_PLACEHOLDER = re.compile(r"^[A-Za-z0-9_.{}<>:-]+$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(^|[=:\s])/[^\s]+"),
    re.compile(r"(^|[=:\s])~(?:/|$)"),
    re.compile(r"(^|[=:\s])[A-Za-z]:[\\/][^\s]+"),
)
_PATH_LIKE_KEYS = frozenset(
    {
        "design_file",
        "feat_dir_template",
        "mask_template",
        "noise_image_template",
        "path",
        "path_template",
        "pe_image_template",
        "pattern",
        "relative_path",
        "template",
        "transform_template",
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
_PE_COPE_NUMBER_FIELDS = frozenset(
    {
        "cope",
        "cope_index",
        "cope_number",
        "contrast_number",
        "design_pe",
        "feat_pe",
        "pe",
        "pe_index",
        "pe_number",
    }
)
_PE_COPE_NUMBER_PATH_PATTERN = re.compile(
    r"(?i)(^|[/_.-])(pe|cope|varcope|zstat|contrast)[_-]?\d+(?=($|[/_.-]))"
)
_ALLOWED_LOSO_UNITS = frozenset({"subject", "session", "run", "custom"})
_ALLOWED_MEAN_CENTERING_SCOPES = frozenset({"none", "run", "session", "subject", "roi", "condition", "custom"})


@dataclass(frozen=True)
class MvpaFeatSourceConfig:
    """A deferred FEAT source declaration for localizer or MVPA inputs."""

    name: str
    role: str
    root_ref: str
    feat_dir_template: str | None = None
    design_file: str | None = None
    pe_image_template: str | None = None
    noise_image_template: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoiCollectionConfig:
    """ROI collection or catalog metadata referenced by the MVPA workflow."""

    name: str
    catalog_ref: str | None = None
    roi_set_refs: tuple[str, ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LosoWorkflowSettings:
    """LOSO settings carried as plan metadata only."""

    enabled: bool = False
    heldout_unit: str = "subject"
    grouping: tuple[str, ...] = ()
    min_training_observations: int | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformSettings:
    """A deferred transform declaration."""

    name: str
    from_space: str | None = None
    to_space: str | None = None
    transform_ref: str | None = None
    apply_to: tuple[str, ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PeMappingAlias:
    """Condition-to-design alias metadata without numeric PE coupling."""

    condition: str
    aliases: tuple[str, ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionPair:
    """A configured condition pair for future MVPA distance rows."""

    left: str
    right: str
    name: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdSweep:
    """Minimum event/observation thresholds to evaluate later."""

    name: str | None = None
    min_events: int | None = None
    min_observations: int | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunExclusion:
    """A configured run exclusion rule."""

    id: str
    reason: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MvpaWorkflowExtensionConfig:
    """Validated MVPA extension payload for a generic workflow recipe."""

    localizer_feat_sources: tuple[MvpaFeatSourceConfig, ...]
    mvpa_feat_sources: tuple[MvpaFeatSourceConfig, ...]
    roi_collections: tuple[RoiCollectionConfig, ...] = ()
    loso: LosoWorkflowSettings = field(default_factory=LosoWorkflowSettings)
    transforms: tuple[TransformSettings, ...] = ()
    pe_mapping_aliases: tuple[PeMappingAlias, ...] = ()
    condition_pairs: tuple[ConditionPair, ...] = ()
    threshold_sweeps: tuple[ThresholdSweep, ...] = ()
    run_exclusions: tuple[RunExclusion, ...] = ()
    crossvalidation_unit: str = "run"
    noise_normalization: str = "identity"
    mean_centering: Mapping[str, Any] = field(default_factory=dict)
    publication: Mapping[str, Any] = field(default_factory=dict)
    reporting: Mapping[str, Any] = field(default_factory=dict)
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MvpaFeatSourcePlanRow:
    """Plan row for a deferred FEAT source."""

    name: str
    role: str
    root_ref: str
    feat_dir_template: str | None = None
    design_file: str | None = None
    pe_image_template: str | None = None
    noise_image_template: str | None = None
    status: str = "deferred"
    reason: str = "feat_source_discovery_not_implemented_schema_slice"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RoiCollectionPlanRow:
    """Plan row for ROI catalog/collection metadata."""

    name: str
    catalog_ref: str | None = None
    roi_set_refs: tuple[str, ...] = ()
    status: str = "configured"
    reason: str = "roi_resolution_not_implemented_schema_slice"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class TransformPlanRow:
    """Plan row for a deferred transform declaration."""

    name: str
    from_space: str | None = None
    to_space: str | None = None
    transform_ref: str | None = None
    apply_to: tuple[str, ...] = ()
    status: str = "deferred"
    reason: str = "mask_transform_execution_not_implemented_schema_slice"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PeMappingAliasPlanRow:
    """Plan row for a PE alias declaration."""

    condition: str
    aliases: tuple[str, ...] = ()
    status: str = "configured"
    reason: str = "numeric_pe_mapping_deferred_to_design_metadata"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ConditionPairPlanRow:
    """Plan row for a condition pair declaration."""

    left: str
    right: str
    name: str | None = None
    status: str = "configured"
    reason: str = "distance_computation_not_implemented_schema_slice"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ThresholdSweepPlanRow:
    """Plan row for a future minimum-events/observations check."""

    name: str | None = None
    min_events: int | None = None
    min_observations: int | None = None
    status: str = "not_evaluated"
    reason: str = "event_and_observation_counts_not_read_schema_slice"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class RunExclusionPlanRow:
    """Plan row for a configured run exclusion rule."""

    id: str
    reason: str | None = None
    status: str = "configured"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class MvpaWorkflowExtensionPlan:
    """JSON-serializable MVPA extension plan preview."""

    workflow_name: str | None
    status: str
    errors: tuple[str, ...] = ()
    localizer_feat_sources: tuple[MvpaFeatSourcePlanRow, ...] = ()
    mvpa_feat_sources: tuple[MvpaFeatSourcePlanRow, ...] = ()
    roi_collections: tuple[RoiCollectionPlanRow, ...] = ()
    loso: Mapping[str, Any] = field(default_factory=dict)
    transforms: tuple[TransformPlanRow, ...] = ()
    pe_mapping_aliases: tuple[PeMappingAliasPlanRow, ...] = ()
    condition_pairs: tuple[ConditionPairPlanRow, ...] = ()
    threshold_sweeps: tuple[ThresholdSweepPlanRow, ...] = ()
    run_exclusions: tuple[RunExclusionPlanRow, ...] = ()
    settings: Mapping[str, Any] = field(default_factory=dict)
    publication: Mapping[str, Any] = field(default_factory=dict)
    reporting: Mapping[str, Any] = field(default_factory=dict)
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


def validate_mvpa_workflow_extension_document(document: Mapping[str, Any] | Any) -> list[str]:
    """Validate an MVPA extension document without inspecting files."""

    if not isinstance(document, Mapping):
        return ["mvpa workflow extension document must contain a mapping."]

    errors: list[str] = []
    payload = _extension_payload(document, errors=errors)
    if not payload:
        return errors

    _validate_no_execution_fields(payload, "mvpa", errors)
    _validate_no_personal_paths(payload, "mvpa", errors)
    _validate_relative_path_strings(payload, "mvpa", errors)
    _validate_no_hard_coded_pe_cope_fields(payload, "mvpa", errors)
    _validate_feat_sources(payload, "localizer_feat_sources", "localizer", errors)
    _validate_feat_sources(payload, "mvpa_feat_sources", "mvpa", errors)
    _validate_roi_collections(payload, errors)
    _validate_loso(payload, errors)
    _validate_transforms(payload, errors)
    _validate_pe_mapping_aliases(payload, errors)
    _validate_condition_pairs(payload, errors)
    _validate_threshold_sweeps(payload, errors)
    _validate_run_exclusions(payload, errors)
    _validate_crossvalidation(payload, errors)
    _validate_noise_normalization(payload, errors)
    _validate_mean_centering(payload, errors)
    _validate_publication_and_reporting(payload, errors)
    return errors


def parse_mvpa_workflow_extension_document(document: Mapping[str, Any]) -> MvpaWorkflowExtensionConfig:
    """Parse a validated MVPA extension document into frozen contracts."""

    errors = validate_mvpa_workflow_extension_document(document)
    if errors:
        raise ValueError("; ".join(errors))

    payload = _extension_payload(document)
    return MvpaWorkflowExtensionConfig(
        localizer_feat_sources=tuple(_parse_feat_source(source, role="localizer") for source in _feat_source_mappings(payload, "localizer_feat_sources")),
        mvpa_feat_sources=tuple(_parse_feat_source(source, role="mvpa") for source in _feat_source_mappings(payload, "mvpa_feat_sources")),
        roi_collections=tuple(_parse_roi_collection(row) for row in _roi_collection_mappings(payload)),
        loso=_parse_loso(payload),
        transforms=tuple(_parse_transform(row) for row in _transform_mappings(payload)),
        pe_mapping_aliases=tuple(_parse_pe_mapping_alias(row) for row in _pe_mapping_alias_mappings(payload)),
        condition_pairs=tuple(_parse_condition_pair(row) for row in _condition_pair_mappings(payload)),
        threshold_sweeps=tuple(_parse_threshold_sweep(row) for row in _threshold_sweep_mappings(payload)),
        run_exclusions=tuple(_parse_run_exclusion(row) for row in _run_exclusion_mappings(payload)),
        crossvalidation_unit=_crossvalidation_unit(payload),
        noise_normalization=_noise_normalization_method(payload),
        mean_centering=_mean_centering_payload(payload),
        publication=dict(payload.get("publication")) if isinstance(payload.get("publication"), Mapping) else {},
        reporting=dict(payload.get("reporting")) if isinstance(payload.get("reporting"), Mapping) else {},
        fields=dict(payload),
    )


def plan_mvpa_workflow_extension(document: Mapping[str, Any] | Any) -> MvpaWorkflowExtensionPlan:
    """Return a plan-only MVPA extension preview."""

    errors = validate_mvpa_workflow_extension_document(document)
    if errors:
        return MvpaWorkflowExtensionPlan(
            workflow_name=_candidate_workflow_name(document),
            status="invalid",
            errors=tuple(errors),
            executed=False,
        )

    config = parse_mvpa_workflow_extension_document(document)
    return MvpaWorkflowExtensionPlan(
        workflow_name=_candidate_workflow_name(document),
        status="deferred",
        localizer_feat_sources=tuple(_feat_source_plan_row(source) for source in config.localizer_feat_sources),
        mvpa_feat_sources=tuple(_feat_source_plan_row(source) for source in config.mvpa_feat_sources),
        roi_collections=tuple(
            RoiCollectionPlanRow(
                name=collection.name,
                catalog_ref=collection.catalog_ref,
                roi_set_refs=collection.roi_set_refs,
            )
            for collection in config.roi_collections
        ),
        loso={
            "enabled": config.loso.enabled,
            "heldout_unit": config.loso.heldout_unit,
            "grouping": config.loso.grouping,
            "min_training_observations": config.loso.min_training_observations,
            "status": "configured",
            "reason": "loso_execution_not_implemented_schema_slice",
        },
        transforms=tuple(
            TransformPlanRow(
                name=transform.name,
                from_space=transform.from_space,
                to_space=transform.to_space,
                transform_ref=transform.transform_ref,
                apply_to=transform.apply_to,
            )
            for transform in config.transforms
        ),
        pe_mapping_aliases=tuple(
            PeMappingAliasPlanRow(condition=alias.condition, aliases=alias.aliases)
            for alias in config.pe_mapping_aliases
        ),
        condition_pairs=tuple(
            ConditionPairPlanRow(left=pair.left, right=pair.right, name=pair.name)
            for pair in config.condition_pairs
        ),
        threshold_sweeps=tuple(
            ThresholdSweepPlanRow(
                name=sweep.name,
                min_events=sweep.min_events,
                min_observations=sweep.min_observations,
            )
            for sweep in config.threshold_sweeps
        ),
        run_exclusions=tuple(
            RunExclusionPlanRow(id=exclusion.id, reason=exclusion.reason)
            for exclusion in config.run_exclusions
        ),
        settings={
            "crossvalidation_unit": config.crossvalidation_unit,
            "noise_normalization": config.noise_normalization,
            "mean_centering": config.mean_centering,
        },
        publication=config.publication,
        reporting=config.reporting,
        executed=False,
    )


def _extension_payload(document: Mapping[str, Any], *, errors: list[str] | None = None) -> Mapping[str, Any]:
    if isinstance(document.get("analysis_workflow"), Mapping):
        workflow = document["analysis_workflow"]  # type: ignore[index]
        extensions = workflow.get("extensions")
        if isinstance(extensions, Mapping) and isinstance(extensions.get("mvpa"), Mapping):
            return extensions["mvpa"]  # type: ignore[index]
        if isinstance(workflow.get("mvpa"), Mapping):
            return workflow["mvpa"]  # type: ignore[index]
        if errors is not None:
            errors.append("analysis_workflow.extensions.mvpa must contain a mapping.")
        return {}
    for key in ("mvpa_workflow", "mvpa_extension", "mvpa"):
        if isinstance(document.get(key), Mapping):
            return document[key]  # type: ignore[index]
    if _looks_like_extension_payload(document):
        return document
    if errors is not None:
        errors.append("mvpa workflow extension must contain a mapping.")
    return {}


def _looks_like_extension_payload(document: Mapping[str, Any]) -> bool:
    return any(
        key in document
        for key in (
            "condition_pairs",
            "localizer_feat_sources",
            "mvpa_feat_sources",
            "pe_mapping_aliases",
            "roi_collections",
        )
    )


def _candidate_workflow_name(document: Any) -> str | None:
    if not isinstance(document, Mapping):
        return None
    workflow = document.get("analysis_workflow")
    if isinstance(workflow, Mapping):
        return _optional_text(workflow.get("name"))
    return None


def _validate_feat_sources(payload: Mapping[str, Any], key: str, role: str, errors: list[str]) -> None:
    sources = _feat_source_mappings(payload, key, errors=errors)
    if not sources:
        errors.append(f"mvpa.{key} must define at least one {role} FEAT source.")
        return
    names: list[str] = []
    for index, source in enumerate(sources):
        label = f"mvpa.{key}[{index}]"
        name = _optional_text(source.get("name") or source.get("id"))
        if name is None:
            errors.append(f"{label}.name must be defined.")
        else:
            names.append(name)
            _validate_safe_identifier(name, f"{label}.name", errors)
        root_ref = _optional_text(source.get("root_ref"))
        if root_ref is None:
            errors.append(f"{label}.root_ref must be defined.")
        else:
            _validate_safe_identifier(root_ref, f"{label}.root_ref", errors)
    for duplicate in _duplicates(names):
        errors.append(f"mvpa.{key} contains duplicate source name: {duplicate}.")


def _feat_source_mappings(
    payload: Mapping[str, Any],
    key: str,
    *,
    errors: list[str] | None = None,
) -> list[Mapping[str, Any]]:
    return _mapping_list(payload.get(key), f"mvpa.{key}", errors, require_non_empty=True)


def _parse_feat_source(source: Mapping[str, Any], *, role: str) -> MvpaFeatSourceConfig:
    return MvpaFeatSourceConfig(
        name=str(source.get("name") or source.get("id")).strip(),
        role=role,
        root_ref=str(source["root_ref"]).strip(),
        feat_dir_template=_optional_text(source.get("feat_dir_template") or source.get("path_template")),
        design_file=_optional_text(source.get("design_file")),
        pe_image_template=_optional_text(source.get("pe_image_template")),
        noise_image_template=_optional_text(source.get("noise_image_template")),
        fields=dict(source),
    )


def _feat_source_plan_row(source: MvpaFeatSourceConfig) -> MvpaFeatSourcePlanRow:
    return MvpaFeatSourcePlanRow(
        name=source.name,
        role=source.role,
        root_ref=source.root_ref,
        feat_dir_template=source.feat_dir_template,
        design_file=source.design_file,
        pe_image_template=source.pe_image_template,
        noise_image_template=source.noise_image_template,
    )


def _validate_roi_collections(payload: Mapping[str, Any], errors: list[str]) -> None:
    rows = _roi_collection_mappings(payload)
    for index, row in enumerate(rows):
        label = f"mvpa.roi_collections[{index}]"
        name = _optional_text(row.get("name") or row.get("id"))
        if name is None:
            errors.append(f"{label}.name must be defined.")
        else:
            _validate_safe_identifier(name, f"{label}.name", errors)
        catalog_ref = _optional_text(row.get("catalog_ref"))
        if catalog_ref is not None:
            _validate_safe_identifier(catalog_ref, f"{label}.catalog_ref", errors)
        for ref_index, ref in enumerate(_string_sequence(row.get("roi_set_refs") or row.get("roi_sets"))):
            _validate_safe_identifier(ref, f"{label}.roi_set_refs[{ref_index}]", errors)


def _roi_collection_mappings(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _mapping_list(payload.get("roi_collections", payload.get("roi_catalogs")), "mvpa.roi_collections", None, require_non_empty=False)


def _parse_roi_collection(row: Mapping[str, Any]) -> RoiCollectionConfig:
    return RoiCollectionConfig(
        name=str(row.get("name") or row.get("id")).strip(),
        catalog_ref=_optional_text(row.get("catalog_ref")),
        roi_set_refs=tuple(_string_sequence(row.get("roi_set_refs") or row.get("roi_sets"))),
        fields=dict(row),
    )


def _validate_loso(payload: Mapping[str, Any], errors: list[str]) -> None:
    loso = payload.get("loso")
    if loso is None:
        return
    if not isinstance(loso, Mapping):
        errors.append("mvpa.loso must contain a mapping when declared.")
        return
    heldout_unit = _optional_text(loso.get("heldout_unit") or loso.get("unit")) or "subject"
    if heldout_unit not in _ALLOWED_LOSO_UNITS:
        errors.append(f"mvpa.loso.heldout_unit must be one of: {', '.join(sorted(_ALLOWED_LOSO_UNITS))}.")
    for index, value in enumerate(_string_sequence(loso.get("grouping") or loso.get("group_by"))):
        _validate_safe_label_or_placeholder(value, f"mvpa.loso.grouping[{index}]", errors)
    min_training = _optional_int(loso.get("min_training_observations"))
    if "min_training_observations" in loso and min_training is None:
        errors.append("mvpa.loso.min_training_observations must be an integer.")
    elif min_training is not None and min_training < 0:
        errors.append("mvpa.loso.min_training_observations must be non-negative.")


def _parse_loso(payload: Mapping[str, Any]) -> LosoWorkflowSettings:
    loso = payload.get("loso")
    if not isinstance(loso, Mapping):
        return LosoWorkflowSettings()
    return LosoWorkflowSettings(
        enabled=bool(loso.get("enabled", False)),
        heldout_unit=_optional_text(loso.get("heldout_unit") or loso.get("unit")) or "subject",
        grouping=tuple(_string_sequence(loso.get("grouping") or loso.get("group_by"))),
        min_training_observations=_optional_int(loso.get("min_training_observations")),
        fields=dict(loso),
    )


def _validate_transforms(payload: Mapping[str, Any], errors: list[str]) -> None:
    for index, row in enumerate(_transform_mappings(payload)):
        label = f"mvpa.transforms[{index}]"
        name = _optional_text(row.get("name") or row.get("id"))
        if name is None:
            errors.append(f"{label}.name must be defined.")
        else:
            _validate_safe_identifier(name, f"{label}.name", errors)
        for key in ("from_space", "to_space", "transform_ref"):
            value = _optional_text(row.get(key))
            if value is not None:
                _validate_safe_label_or_placeholder(value, f"{label}.{key}", errors)
        for apply_index, value in enumerate(_string_sequence(row.get("apply_to"))):
            _validate_safe_identifier(value, f"{label}.apply_to[{apply_index}]", errors)


def _transform_mappings(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _mapping_list(payload.get("transforms", payload.get("transform_settings")), "mvpa.transforms", None, require_non_empty=False)


def _parse_transform(row: Mapping[str, Any]) -> TransformSettings:
    return TransformSettings(
        name=str(row.get("name") or row.get("id")).strip(),
        from_space=_optional_text(row.get("from_space")),
        to_space=_optional_text(row.get("to_space")),
        transform_ref=_optional_text(row.get("transform_ref")),
        apply_to=tuple(_string_sequence(row.get("apply_to"))),
        fields=dict(row),
    )


def _validate_pe_mapping_aliases(payload: Mapping[str, Any], errors: list[str]) -> None:
    for index, row in enumerate(_pe_mapping_alias_mappings(payload)):
        label = f"mvpa.pe_mapping_aliases[{index}]"
        condition = _optional_text(row.get("condition") or row.get("condition_id") or row.get("id"))
        if condition is None:
            errors.append(f"{label}.condition must be defined.")
        else:
            _validate_safe_label_or_placeholder(condition, f"{label}.condition", errors)
        aliases = _string_sequence(row.get("aliases", row.get("alias")))
        if not aliases:
            errors.append(f"{label}.aliases must define at least one alias.")
        for alias_index, alias in enumerate(aliases):
            _validate_safe_label_or_placeholder(alias, f"{label}.aliases[{alias_index}]", errors)


def _pe_mapping_alias_mappings(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("pe_mapping_aliases", payload.get("pe_aliases"))
    if isinstance(raw, Mapping):
        rows: list[Mapping[str, Any]] = []
        for condition, aliases in raw.items():
            rows.append({"condition": condition, "aliases": aliases})
        return rows
    return _mapping_list(raw, "mvpa.pe_mapping_aliases", None, require_non_empty=False)


def _parse_pe_mapping_alias(row: Mapping[str, Any]) -> PeMappingAlias:
    return PeMappingAlias(
        condition=str(row.get("condition") or row.get("condition_id") or row.get("id")).strip(),
        aliases=tuple(_string_sequence(row.get("aliases", row.get("alias")))),
        fields=dict(row),
    )


def _validate_condition_pairs(payload: Mapping[str, Any], errors: list[str]) -> None:
    pairs = _condition_pair_mappings(payload)
    for index, row in enumerate(pairs):
        label = f"mvpa.condition_pairs[{index}]"
        for key in ("left", "right"):
            value = _optional_text(row.get(key))
            if value is None:
                errors.append(f"{label}.{key} must be defined.")
            else:
                _validate_safe_label_or_placeholder(value, f"{label}.{key}", errors)
        name = _optional_text(row.get("name") or row.get("id"))
        if name is not None:
            _validate_safe_identifier(name, f"{label}.name", errors)


def _condition_pair_mappings(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _mapping_list(payload.get("condition_pairs"), "mvpa.condition_pairs", None, require_non_empty=False)


def _parse_condition_pair(row: Mapping[str, Any]) -> ConditionPair:
    return ConditionPair(
        left=str(row.get("left")).strip(),
        right=str(row.get("right")).strip(),
        name=_optional_text(row.get("name") or row.get("id")),
        fields=dict(row),
    )


def _validate_threshold_sweeps(payload: Mapping[str, Any], errors: list[str]) -> None:
    for index, row in enumerate(_threshold_sweep_mappings(payload)):
        label = f"mvpa.threshold_sweeps[{index}]"
        name = _optional_text(row.get("name") or row.get("id"))
        if name is not None:
            _validate_safe_identifier(name, f"{label}.name", errors)
        min_events = _optional_int(row.get("min_events", row.get("min_events_per_condition")))
        min_observations = _optional_int(row.get("min_observations", row.get("min_observations_per_condition")))
        if min_events is None and min_observations is None:
            errors.append(f"{label} must define min_events or min_observations.")
        if min_events is not None and min_events < 0:
            errors.append(f"{label}.min_events must be non-negative.")
        if min_observations is not None and min_observations < 0:
            errors.append(f"{label}.min_observations must be non-negative.")


def _threshold_sweep_mappings(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("threshold_sweeps", payload.get("min_events_min_observations_sweeps"))
    return _mapping_list(raw, "mvpa.threshold_sweeps", None, require_non_empty=False)


def _parse_threshold_sweep(row: Mapping[str, Any]) -> ThresholdSweep:
    return ThresholdSweep(
        name=_optional_text(row.get("name") or row.get("id")),
        min_events=_optional_int(row.get("min_events", row.get("min_events_per_condition"))),
        min_observations=_optional_int(row.get("min_observations", row.get("min_observations_per_condition"))),
        fields=dict(row),
    )


def _validate_run_exclusions(payload: Mapping[str, Any], errors: list[str]) -> None:
    for index, row in enumerate(_run_exclusion_mappings(payload)):
        label = f"mvpa.run_exclusions[{index}]"
        exclusion_id = _optional_text(row.get("id") or row.get("name"))
        if exclusion_id is None:
            errors.append(f"{label}.id must be defined.")
        else:
            _validate_safe_identifier(exclusion_id, f"{label}.id", errors)
        selectors = row.get("selectors") if isinstance(row.get("selectors"), Mapping) else row
        if isinstance(selectors, Mapping):
            for key in ("subject", "subject_id", "session", "session_id", "task", "task_id", "run", "run_id"):
                value = _optional_text(selectors.get(key))
                if value is not None:
                    _validate_safe_label_or_placeholder(value, f"{label}.{key}", errors)


def _run_exclusion_mappings(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _mapping_list(payload.get("run_exclusions"), "mvpa.run_exclusions", None, require_non_empty=False)


def _parse_run_exclusion(row: Mapping[str, Any]) -> RunExclusion:
    return RunExclusion(
        id=str(row.get("id") or row.get("name")).strip(),
        reason=_optional_text(row.get("reason")),
        fields=dict(row),
    )


def _validate_crossvalidation(payload: Mapping[str, Any], errors: list[str]) -> None:
    unit = _crossvalidation_unit(payload)
    if unit not in ALLOWED_CV_UNITS:
        errors.append(f"mvpa.cross_validation.unit must be one of: {', '.join(sorted(ALLOWED_CV_UNITS))}.")


def _crossvalidation_unit(payload: Mapping[str, Any]) -> str:
    cross_validation = payload.get("cross_validation", payload.get("crossvalidation"))
    raw = payload.get("crossvalidation_unit", payload.get("cv_unit"))
    if raw is None and isinstance(cross_validation, Mapping):
        raw = cross_validation.get("unit")
    return _optional_text(raw) or "run"


def _validate_noise_normalization(payload: Mapping[str, Any], errors: list[str]) -> None:
    method = _noise_normalization_method(payload)
    if method not in ALLOWED_NOISE_NORMALIZATION_METHODS:
        errors.append(
            "mvpa.noise_normalization.method must be one of: "
            f"{', '.join(sorted(ALLOWED_NOISE_NORMALIZATION_METHODS))}."
        )


def _noise_normalization_method(payload: Mapping[str, Any]) -> str:
    raw = payload.get("noise_normalization")
    if isinstance(raw, Mapping):
        raw = raw.get("method")
    return _optional_text(raw) or "identity"


def _validate_mean_centering(payload: Mapping[str, Any], errors: list[str]) -> None:
    mean_centering = _mean_centering_payload(payload)
    scope = _optional_text(mean_centering.get("scope")) or "none"
    if scope not in _ALLOWED_MEAN_CENTERING_SCOPES:
        errors.append(f"mvpa.mean_centering.scope must be one of: {', '.join(sorted(_ALLOWED_MEAN_CENTERING_SCOPES))}.")


def _mean_centering_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = payload.get("mean_centering")
    if isinstance(raw, Mapping):
        return dict(raw)
    if raw is None:
        return {"enabled": False, "scope": "none"}
    return {"enabled": bool(raw), "scope": "run" if bool(raw) else "none"}


def _validate_publication_and_reporting(payload: Mapping[str, Any], errors: list[str]) -> None:
    for key in ("publication", "reporting"):
        raw = payload.get(key)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            errors.append(f"mvpa.{key} must contain a mapping when declared.")
            continue
        derivative_name = _optional_text(raw.get("derivative_name"))
        if derivative_name is not None:
            _validate_safe_identifier(derivative_name, f"mvpa.{key}.derivative_name", errors)
        for index, value in enumerate(_string_sequence(raw.get("formats") or raw.get("sections"))):
            _validate_safe_identifier(value, f"mvpa.{key}.formats[{index}]", errors)


def _validate_no_execution_fields(value: Any, label: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if str(key).lower() in _EXECUTION_KEYS:
                errors.append(f"{child_label} is not allowed in a schema/plan-only MVPA extension.")
            _validate_no_execution_fields(child, child_label, errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_execution_fields(child, f"{label}[{index}]", errors)


def _validate_no_hard_coded_pe_cope_fields(value: Any, label: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if str(key).lower() in _PE_COPE_NUMBER_FIELDS:
                errors.append(f"{child_label} must not hard-code PE/COPE/contrast numbers.")
            _validate_no_hard_coded_pe_cope_fields(child, child_label, errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_hard_coded_pe_cope_fields(child, f"{label}[{index}]", errors)


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
    if _PE_COPE_NUMBER_PATH_PATTERN.search(text):
        errors.append(f"{label} must use a placeholder instead of a hard-coded PE/COPE/contrast number.")


def _path_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _validate_safe_identifier(value: str, label: str, errors: list[str]) -> None:
    if not _SAFE_IDENTIFIER.fullmatch(value):
        errors.append(f"{label} must be a safe identifier.")


def _validate_safe_label_or_placeholder(value: str, label: str, errors: list[str]) -> None:
    if not _SAFE_LABEL_OR_PLACEHOLDER.fullmatch(value):
        errors.append(f"{label} must be a safe label or placeholder.")


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


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return None


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


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
    "ConditionPair",
    "ConditionPairPlanRow",
    "LosoWorkflowSettings",
    "MvpaFeatSourceConfig",
    "MvpaFeatSourcePlanRow",
    "MvpaWorkflowExtensionConfig",
    "MvpaWorkflowExtensionPlan",
    "PeMappingAlias",
    "PeMappingAliasPlanRow",
    "RoiCollectionConfig",
    "RoiCollectionPlanRow",
    "RunExclusion",
    "RunExclusionPlanRow",
    "ThresholdSweep",
    "ThresholdSweepPlanRow",
    "TransformPlanRow",
    "TransformSettings",
    "parse_mvpa_workflow_extension_document",
    "plan_mvpa_workflow_extension",
    "validate_mvpa_workflow_extension_document",
]
