"""Plan-only MVPA configuration contracts.

The helpers in this module validate dictionary-shaped MVPA set configs without
resolving files, discovering FEAT directories, parsing designs, loading masks,
or importing optional neuroimaging/MVPA dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
import math
import re

from .pattern_sources import (
    ALLOWED_PATTERN_BACKENDS,
    ALLOWED_UNIT_SELECTION_MODES,
    PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE,
    PATTERN_BACKEND_FSL_FEAT_PE,
    PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE,
    PATTERN_BACKEND_NILEARN_GLM,
    PATTERN_BACKEND_SURFACE_CIFTI,
    UNIT_SELECTION_EXACT,
    UNIT_SELECTION_LEGACY_CARTESIAN,
    PatternSourceAdapterRegistry,
    UnitSelectionConfig,
)


METRIC_CROSSNOBIS = "crossnobis"
METRIC_EUCLIDEAN = "euclidean"
METRIC_CORRELATION = "correlation"
ENGINE_NATIVE_REFERENCE = "native_reference"
ENGINE_RSATOOLBOX = "rsatoolbox"
ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1 = "manual_diagonal_crossnobis_v1"
NOISE_NORMALIZATION_IDENTITY = "identity"
NOISE_NORMALIZATION_DIAGONAL = "diagonal"
CV_UNIT_RUN = "run"
CV_UNIT_SESSION = "session"
CV_UNIT_SUBJECT = "subject"
CV_UNIT_CUSTOM = "custom"
MISSING_INPUT_POLICY_WARN = "warn"
MISSING_INPUT_POLICY_SKIP = "skip"
MISSING_INPUT_POLICY_FAIL = "fail"
ROI_SOURCE_ROI_SET = "roi_set"
ROI_SOURCE_ROI_SET_RUNTIME = "roi_set_runtime"
ROI_SOURCE_ROI_SET_PUBLICATION = "roi_set_publication"
ROI_SOURCE_EXPLICIT_MASKS = "explicit_masks"
ROI_SOURCE_MATERIALIZED_FEATURES = "materialized_features"
RUNTIME_EXISTING_OUTPUT_FAIL = "fail"

ALLOWED_METRICS = frozenset({METRIC_CROSSNOBIS, METRIC_EUCLIDEAN, METRIC_CORRELATION})
ALLOWED_ENGINES = frozenset({ENGINE_NATIVE_REFERENCE, ENGINE_RSATOOLBOX, ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1})
ALLOWED_NOISE_NORMALIZATION_METHODS = frozenset({NOISE_NORMALIZATION_IDENTITY, NOISE_NORMALIZATION_DIAGONAL})
ALLOWED_NOISE_NONPOSITIVE_POLICIES = frozenset(
    {"strict", "fail", "drop_pattern", "drop_features", "filter_nonpositive_features"}
)
CONDITION_PAIR_MODE_ALL_PAIRS = "all_pairs"
DEFAULT_NOISE_NONPOSITIVE_POLICY = "strict"
DEFAULT_MIN_RETAINED_FEATURES = 5
DEFAULT_WARN_DROPPED_FEATURE_FRACTION = 0.10
ALLOWED_CV_UNITS = frozenset({CV_UNIT_RUN, CV_UNIT_SESSION, CV_UNIT_SUBJECT, CV_UNIT_CUSTOM})
ALLOWED_MISSING_INPUT_POLICIES = frozenset(
    {MISSING_INPUT_POLICY_WARN, MISSING_INPUT_POLICY_SKIP, MISSING_INPUT_POLICY_FAIL}
)
ALLOWED_ROI_SOURCE_TYPES = frozenset(
    {
        ROI_SOURCE_ROI_SET,
        ROI_SOURCE_ROI_SET_RUNTIME,
        ROI_SOURCE_ROI_SET_PUBLICATION,
        ROI_SOURCE_EXPLICIT_MASKS,
        ROI_SOURCE_MATERIALIZED_FEATURES,
    }
)
_ROI_SET_SOURCE_TYPES = frozenset({ROI_SOURCE_ROI_SET, ROI_SOURCE_ROI_SET_RUNTIME, ROI_SOURCE_ROI_SET_PUBLICATION})

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_SELECTOR_VALUE = re.compile(r"^[A-Za-z0-9_.{}<>:-]+$")
_BIDS_LABEL_VALUE = re.compile(r"^[A-Za-z0-9]+$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(^|[=:\s])/[^\s]+"),
    re.compile(r"(^|[=:\s])~(?:/|$)"),
    re.compile(r"(^|[=:\s])[A-Za-z]:[\\/][^\s]+"),
)
_PE_COPE_SELECTOR_FIELDS = frozenset(
    {
        "pe",
        "pe_number",
        "pe_index",
        "cope",
        "cope_number",
        "design_pe",
        "feat_pe",
    }
)
_MVPA_PAYLOAD_KEYS = frozenset(
    {
        "condition_pairs",
        "conditions",
        "distance",
        "entities",
        "exclusion_rules",
        "grouping_columns",
        "mean_centering",
        "missing_input_policy",
        "outputs",
        "pattern_sources",
        "roi_sources",
        "runtime",
        "run_exclusions",
        "runs",
        "sessions",
        "subjects",
        "threshold_sweeps",
        "unit_selection",
        "within_roi_mean_centering",
    }
)
_PATH_LIKE_KEYS = frozenset(
    {
        "path",
        "paths",
        "relative_path",
        "relative_paths",
        "pattern",
        "patterns",
        "mask_path",
        "mask_paths",
        "mask_pattern",
        "mask_patterns",
        "template",
        "templates",
    }
)
_FSL_FEAT_PE_TEMPLATE_FIELDS = frozenset(
    {
        "feat_dir_template",
        "design_file",
        "pe_image_template",
        "noise_image_template",
    }
)
_FSL_FEAT_PE_MISSING_KEYS = frozenset(
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
_ROI_SOURCE_MISSING_KEYS = frozenset(
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
_ALLOWED_MEAN_CENTERING_SCOPES = frozenset({"none", "roi", "within_roi"})


@dataclass(frozen=True)
class SubjectSessionRunSelector:
    """Subject/session/run selectors for a plan-only MVPA set."""

    subjects: tuple[str, ...]
    sessions: tuple[str, ...]
    runs: tuple[str, ...]
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MvpaEntities:
    """BIDS-like entity selectors carried by MVPA plans."""

    task: str | None = None
    direction: str | None = None
    model: str | None = None
    space: str | None = None
    resolution: str | None = None
    acquisition: str | None = None
    datatype: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionDefinition:
    """A configured condition without PE/COPE-number coupling."""

    id: str
    aliases: tuple[str, ...] = ()
    selector: Mapping[str, Any] = field(default_factory=dict)
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionPairConfig:
    """A configured condition pair for runtime distance filtering."""

    id: str
    condition_id_a: str
    condition_id_b: str
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatternSourceConfig:
    """A deferred pattern source declaration."""

    name: str
    backend: str
    root_ref: str | None = None
    path: str | None = None
    pattern: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoiSourceConfig:
    """A deferred ROI source declaration."""

    name: str
    source: str
    roi_set_ref: str | None = None
    root_ref: str | None = None
    path: str | None = None
    pattern: str | None = None
    masks: tuple[Mapping[str, Any], ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventThresholdConfig:
    """Event-count thresholds carried forward for future checks."""

    min_events_per_condition_per_run: int | None = None
    min_runs_per_condition: int | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdSweepConfig:
    """Minimum event/observation thresholds to evaluate at runtime."""

    id: str
    min_events: int | None = None
    min_observations: int | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExclusionRule:
    """A configured exclusion rule for a deferred MVPA plan."""

    id: str
    reason: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    source_config_field: str = "mvpa_set.exclusions.rules"
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeanCenteringConfig:
    """Within-ROI pattern mean-centering runtime settings."""

    enabled: bool = False
    scope: str = "none"
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NoiseNormalizationConfig:
    """Noise-normalization settings for distance contracts."""

    method: str = "identity"
    variance_source: str | None = None
    nonpositive_policy: str = DEFAULT_NOISE_NONPOSITIVE_POLICY
    min_retained_features: int = DEFAULT_MIN_RETAINED_FEATURES
    warn_dropped_feature_fraction: float = DEFAULT_WARN_DROPPED_FEATURE_FRACTION
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DistanceConfig:
    """Distance/RDM settings without execution behavior."""

    metrics: tuple[str, ...] = ("crossnobis",)
    engines: tuple[str, ...] = ("native_reference",)
    cv_unit: str = "run"
    grouping_columns: tuple[str, ...] = ()
    noise_normalization: NoiseNormalizationConfig = field(default_factory=NoiseNormalizationConfig)
    fields: Mapping[str, Any] = field(default_factory=dict)

    @property
    def engine(self) -> str:
        """Return the preferred engine for callers that expect one value."""

        return self.engines[0]


@dataclass(frozen=True)
class OutputRootConfig:
    """A root-ref plus relative path template output declaration."""

    name: str
    root_ref: str
    path: str
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputConfig:
    """Configured plan output roots."""

    roots: tuple[OutputRootConfig, ...]
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    """Local runtime ownership and collision policy."""

    existing_output: str = RUNTIME_EXISTING_OUTPUT_FAIL
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicationConfig:
    """Publication settings for future MVPA derivatives."""

    enabled: bool = False
    derivative_name: str | None = None
    write_json_sidecars: bool = True
    write_provenance: bool = True
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MissingInputPolicy:
    """How future runtime phases should handle missing inputs."""

    policy: str = "warn"
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProvenanceConfig:
    """Provenance metadata supplied by config."""

    schema_version: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MvpaSetConfig:
    """Validated plan-only MVPA set config."""

    name: str
    unit_selection: UnitSelectionConfig
    selector: SubjectSessionRunSelector
    entities: MvpaEntities
    conditions: tuple[ConditionDefinition, ...]
    pattern_sources: tuple[PatternSourceConfig, ...]
    roi_sources: tuple[RoiSourceConfig, ...]
    distance: DistanceConfig
    outputs: OutputConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    event_thresholds: EventThresholdConfig | None = None
    condition_pairs: tuple[ConditionPairConfig, ...] = ()
    threshold_sweeps: tuple[ThresholdSweepConfig, ...] = ()
    exclusions: tuple[ExclusionRule, ...] = ()
    mean_centering: MeanCenteringConfig = field(default_factory=MeanCenteringConfig)
    publication: PublicationConfig = field(default_factory=PublicationConfig)
    missing_input_policy: MissingInputPolicy = field(default_factory=MissingInputPolicy)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    fields: Mapping[str, Any] = field(default_factory=dict)


def _resolved_pattern_source_adapter_registry(
    adapter_registry: PatternSourceAdapterRegistry | None,
) -> PatternSourceAdapterRegistry:
    if adapter_registry is not None:
        return adapter_registry
    from .pattern_source_adapters import default_pattern_source_adapter_registry

    return default_pattern_source_adapter_registry()


def validate_mvpa_set_document(
    document: Mapping[str, Any] | Any,
    *,
    adapter_registry: PatternSourceAdapterRegistry | None = None,
) -> list[str]:
    """Validate a plan-only MVPA set document.

    Validation checks dictionary shape and configured contract values only. It
    intentionally does not inspect the filesystem, discover FEAT outputs, parse
    ``design.fsf``, resolve ROI masks, extract images, or run commands.
    """

    if not isinstance(document, Mapping):
        return ["MVPA set document must contain a mapping."]

    errors: list[str] = []
    mvpa_set = _payload_mapping(document, errors=errors)
    if not mvpa_set:
        return errors

    name = _optional_text(mvpa_set.get("name"))
    if name is None:
        errors.append("mvpa_set.name must be defined.")
    else:
        _validate_safe_identifier(name, "mvpa_set.name", errors)

    _validate_no_personal_paths(mvpa_set, "mvpa_set", errors)
    _validate_relative_path_strings(mvpa_set, "mvpa_set", errors)

    _validate_unit_selection(mvpa_set, errors)
    _validate_selector(mvpa_set, errors)
    _validate_entities(_entities_payload(mvpa_set), errors)
    _validate_conditions(mvpa_set, errors)
    registry = _resolved_pattern_source_adapter_registry(adapter_registry)
    _validate_pattern_sources(
        mvpa_set,
        errors,
        adapter_registry=registry,
    )
    _validate_pattern_source_selection_contract(
        mvpa_set,
        errors,
        adapter_registry=registry,
    )
    _validate_roi_sources(mvpa_set, errors)
    _validate_event_thresholds(mvpa_set, errors)
    _validate_condition_pairs(mvpa_set, errors)
    _validate_threshold_sweeps(mvpa_set, errors)
    _validate_exclusions(mvpa_set, errors)
    _validate_mean_centering(mvpa_set, errors)
    _validate_distance(mvpa_set, errors)
    _validate_outputs(mvpa_set, errors)
    _validate_runtime(mvpa_set, errors)
    _validate_publication(mvpa_set, errors)
    _validate_missing_input_policy(mvpa_set, errors)
    _validate_provenance(mvpa_set, errors)
    return errors


def parse_mvpa_set_document(
    document: Mapping[str, Any],
    *,
    adapter_registry: PatternSourceAdapterRegistry | None = None,
) -> MvpaSetConfig:
    """Parse a validated MVPA set document into frozen config contracts."""

    errors = validate_mvpa_set_document(document, adapter_registry=adapter_registry)
    if errors:
        raise ValueError("; ".join(errors))

    mvpa_set = _payload_mapping(document)
    selector = _parse_selector(mvpa_set)
    entities = _parse_entities(mvpa_set)
    conditions = tuple(_parse_condition(condition) for condition in _condition_mappings(mvpa_set))
    pattern_sources = tuple(_parse_pattern_source(source) for source in _pattern_source_mappings(mvpa_set))
    roi_sources = tuple(_parse_roi_source(source) for source in _roi_source_mappings(mvpa_set))
    return MvpaSetConfig(
        name=str(mvpa_set["name"]).strip(),
        unit_selection=_parse_unit_selection(mvpa_set),
        selector=selector,
        entities=entities,
        conditions=conditions,
        pattern_sources=pattern_sources,
        roi_sources=roi_sources,
        event_thresholds=_parse_event_thresholds(mvpa_set),
        condition_pairs=tuple(_parse_condition_pair(pair) for pair in _condition_pair_mappings(mvpa_set)),
        threshold_sweeps=tuple(
            _parse_threshold_sweep(sweep, index=index)
            for index, sweep in enumerate(_threshold_sweep_mappings(mvpa_set), start=1)
        ),
        exclusions=tuple(_parse_exclusion(rule, source_config_field=source) for rule, source in _exclusion_mapping_rows(mvpa_set)),
        mean_centering=_parse_mean_centering(mvpa_set),
        distance=_parse_distance(mvpa_set),
        outputs=_parse_outputs(mvpa_set),
        runtime=_parse_runtime(mvpa_set),
        publication=_parse_publication(mvpa_set),
        missing_input_policy=_parse_missing_input_policy(mvpa_set),
        provenance=_parse_provenance(mvpa_set),
        fields=dict(mvpa_set),
    )


def _payload_mapping(document: Mapping[str, Any], *, errors: list[str] | None = None) -> Mapping[str, Any]:
    if "mvpa_set" in document:
        payload = document.get("mvpa_set")
    elif _looks_like_mvpa_set_payload(document):
        payload = document
    else:
        if errors is not None:
            errors.append("mvpa_set must contain a mapping.")
        return {}
    if not isinstance(payload, Mapping):
        if errors is not None:
            errors.append("mvpa_set must contain a mapping.")
        return {}
    return payload


def _looks_like_mvpa_set_payload(document: Mapping[str, Any]) -> bool:
    return "name" in document and bool(_MVPA_PAYLOAD_KEYS.intersection(document.keys()))


def _validate_unit_selection(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    raw = mvpa_set.get("unit_selection")
    if raw is not None and not isinstance(raw, Mapping):
        errors.append("mvpa_set.unit_selection must contain a mapping when configured.")
        return

    payload = raw if isinstance(raw, Mapping) else {}
    mode = _optional_text(payload.get("mode")) or UNIT_SELECTION_LEGACY_CARTESIAN
    if mode not in ALLOWED_UNIT_SELECTION_MODES:
        errors.append(
            "mvpa_set.unit_selection.mode must be one of: "
            f"{', '.join(sorted(ALLOWED_UNIT_SELECTION_MODES))}."
        )
        return

    raw_key_columns = payload.get("key_columns")
    key_columns = _string_sequence(raw_key_columns)
    if mode == UNIT_SELECTION_EXACT:
        if raw_key_columns is not None and (
            not isinstance(raw_key_columns, Sequence)
            or isinstance(raw_key_columns, (str, bytes, bytearray))
        ):
            errors.append("mvpa_set.unit_selection.key_columns must contain an ordered list.")
        if not key_columns:
            errors.append(
                "mvpa_set.unit_selection.key_columns must define at least one column in exact_units mode."
            )
        for index, column in enumerate(key_columns):
            _validate_safe_identifier(column, f"mvpa_set.unit_selection.key_columns[{index}]", errors)
        for duplicate in _duplicates(key_columns):
            errors.append(
                f"mvpa_set.unit_selection.key_columns contains duplicate column: {duplicate}."
            )
        if "subject_id" not in key_columns:
            errors.append(
                "mvpa_set.unit_selection.key_columns must include subject_id in exact_units mode."
            )
        if _has_inline_unit_selectors(mvpa_set):
            errors.append(
                "mvpa_set exact_units selection must not be mixed with inline subject, session, or run selectors."
            )
    elif raw_key_columns is not None:
        errors.append(
            "mvpa_set.unit_selection.key_columns is only valid when mode is exact_units."
        )


def _parse_unit_selection(mvpa_set: Mapping[str, Any]) -> UnitSelectionConfig:
    raw = mvpa_set.get("unit_selection")
    payload = dict(raw) if isinstance(raw, Mapping) else {}
    mode = _optional_text(payload.get("mode")) or UNIT_SELECTION_LEGACY_CARTESIAN
    return UnitSelectionConfig(
        mode=mode,
        key_columns=tuple(_string_sequence(payload.get("key_columns"))),
        fields=payload,
    )


def _unit_selection_mode(mvpa_set: Mapping[str, Any]) -> str:
    raw = mvpa_set.get("unit_selection")
    if not isinstance(raw, Mapping):
        return UNIT_SELECTION_LEGACY_CARTESIAN
    return _optional_text(raw.get("mode")) or UNIT_SELECTION_LEGACY_CARTESIAN


def _has_inline_unit_selectors(mvpa_set: Mapping[str, Any]) -> bool:
    selector_keys = {"subject", "subjects", "session", "sessions", "run", "runs"}
    if selector_keys.intersection(mvpa_set):
        return True
    selector = mvpa_set.get("selector")
    if isinstance(selector, Mapping) and selector_keys.intersection(selector):
        return True
    cohort = mvpa_set.get("cohort")
    return isinstance(cohort, Mapping) and bool(selector_keys.intersection(cohort))


def _validate_selector(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    if _unit_selection_mode(mvpa_set) == UNIT_SELECTION_EXACT:
        return
    selector = _selector_payload(mvpa_set)
    subjects = _selector_values(selector, "subjects", "subject")
    sessions = _selector_values(selector, "sessions", "session")
    runs = _selector_values(selector, "runs", "run")
    _validate_required_selector_values(subjects, "mvpa_set.subjects", errors)
    _validate_required_selector_values(sessions, "mvpa_set.sessions", errors)
    _validate_required_selector_values(runs, "mvpa_set.runs", errors)


def _parse_selector(mvpa_set: Mapping[str, Any]) -> SubjectSessionRunSelector:
    selector = _selector_payload(mvpa_set)
    fields = dict(selector)
    return SubjectSessionRunSelector(
        subjects=tuple(_selector_values(selector, "subjects", "subject")),
        sessions=tuple(_selector_values(selector, "sessions", "session")),
        runs=tuple(_selector_values(selector, "runs", "run")),
        fields=fields,
    )


def _selector_payload(mvpa_set: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(mvpa_set.get("selector"), Mapping):
        return mvpa_set["selector"]  # type: ignore[index]
    if isinstance(mvpa_set.get("cohort"), Mapping):
        cohort = mvpa_set["cohort"]  # type: ignore[index]
        payload: dict[str, Any] = {}
        if isinstance(cohort.get("subjects"), Mapping):
            payload["subjects"] = cohort["subjects"].get("include")  # type: ignore[index]
        elif "subjects" in cohort:
            payload["subjects"] = cohort.get("subjects")
        for key in ("sessions", "runs"):
            if key in cohort:
                payload[key] = cohort.get(key)
        for key in ("subjects", "sessions", "runs"):
            if key in mvpa_set:
                payload[key] = mvpa_set.get(key)
        return payload
    return mvpa_set


def _selector_values(selector: Mapping[str, Any], plural: str, singular: str) -> tuple[str, ...]:
    raw = selector.get(plural, selector.get(singular))
    if isinstance(raw, Mapping):
        raw = raw.get("include", raw.get("values"))
    return tuple(_string_sequence(raw))


def _validate_required_selector_values(values: Sequence[str], label: str, errors: list[str]) -> None:
    if not values:
        errors.append(f"{label} must define at least one value.")
        return
    for index, value in enumerate(values):
        item_label = f"{label}[{index}]"
        if not value:
            errors.append(f"{item_label} must be a non-empty value.")
        elif not _SAFE_SELECTOR_VALUE.fullmatch(value):
            errors.append(f"{item_label} must be a safe selector label or placeholder.")


def _validate_entities(entities: Mapping[str, Any], errors: list[str]) -> None:
    for canonical, aliases in _entity_aliases().items():
        value = _first_present(entities, aliases)
        text = _optional_text(value)
        if text is not None:
            _validate_bids_label_or_placeholder(text, f"mvpa_set.entities.{canonical}", errors)


def _parse_entities(mvpa_set: Mapping[str, Any]) -> MvpaEntities:
    entities = _entities_payload(mvpa_set)
    return MvpaEntities(
        task=_optional_text(_first_present(entities, ("task", "task_id"))),
        direction=_optional_text(_first_present(entities, ("direction", "dir"))),
        model=_optional_text(entities.get("model")),
        space=_optional_text(entities.get("space")),
        resolution=_optional_text(_first_present(entities, ("resolution", "res"))),
        acquisition=_optional_text(_first_present(entities, ("acquisition", "acq"))),
        datatype=_optional_text(entities.get("datatype")),
        fields=dict(entities),
    )


def _entities_payload(mvpa_set: Mapping[str, Any]) -> Mapping[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(mvpa_set.get("entities"), Mapping):
        payload.update(mvpa_set["entities"])  # type: ignore[arg-type]
    for alias_group in _entity_aliases().values():
        for alias in alias_group:
            if alias in mvpa_set:
                payload[alias] = mvpa_set.get(alias)
    return payload


def _entity_aliases() -> Mapping[str, tuple[str, ...]]:
    return {
        "task": ("task", "task_id"),
        "direction": ("direction", "dir"),
        "model": ("model",),
        "space": ("space",),
        "resolution": ("resolution", "res"),
        "acquisition": ("acquisition", "acq"),
        "datatype": ("datatype",),
    }


def _validate_conditions(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    conditions = _condition_mappings(mvpa_set, errors=errors)
    ids: list[str] = []
    aliases: list[str] = []
    for index, condition in enumerate(conditions):
        label = f"mvpa_set.conditions[{index}]"
        condition_id = _optional_text(condition.get("id") or condition.get("condition_id") or condition.get("name"))
        if condition_id is None:
            errors.append(f"{label}.id must be defined.")
        else:
            ids.append(condition_id)
            _validate_safe_identifier(condition_id, f"{label}.id", errors)
        for alias_index, alias in enumerate(_condition_aliases(condition)):
            if alias:
                aliases.append(alias)
            else:
                errors.append(f"{label}.aliases[{alias_index}] must be a non-empty value.")
        _validate_no_hard_coded_pe_cope_fields(condition, label, errors)
    for duplicate in _duplicates(ids):
        errors.append(f"mvpa_set.conditions contains duplicate condition id: {duplicate}.")
    for duplicate in _duplicates(aliases):
        errors.append(f"mvpa_set.conditions contains duplicate condition alias: {duplicate}.")


def _parse_condition(condition: Mapping[str, Any]) -> ConditionDefinition:
    selector = condition.get("selector", condition.get("pattern_selector", {}))
    return ConditionDefinition(
        id=str(condition.get("id") or condition.get("condition_id") or condition.get("name")).strip(),
        aliases=tuple(_condition_aliases(condition)),
        selector=dict(selector) if isinstance(selector, Mapping) else {},
        fields=dict(condition),
    )


def _condition_mappings(mvpa_set: Mapping[str, Any], *, errors: list[str] | None = None) -> list[Mapping[str, Any]]:
    return _mapping_list(mvpa_set.get("conditions"), "mvpa_set.conditions", errors, require_non_empty=True)


def _condition_aliases(condition: Mapping[str, Any]) -> tuple[str, ...]:
    aliases = _string_sequence(condition.get("aliases", condition.get("condition_aliases")))
    alias = _optional_text(condition.get("alias"))
    if alias is not None:
        aliases = (*aliases, alias)
    return tuple(aliases)


def _validate_no_hard_coded_pe_cope_fields(payload: Any, label: str, errors: list[str]) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child_label = f"{label}.{key}"
            if str(key).lower() in _PE_COPE_SELECTOR_FIELDS:
                errors.append(f"{child_label} must not hard-code PE/COPE selector numbers in condition definitions.")
            _validate_no_hard_coded_pe_cope_fields(value, child_label, errors)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _validate_no_hard_coded_pe_cope_fields(value, f"{label}[{index}]", errors)


def _validate_pattern_sources(
    mvpa_set: Mapping[str, Any],
    errors: list[str],
    *,
    adapter_registry: PatternSourceAdapterRegistry,
) -> None:
    sources = _pattern_source_mappings(mvpa_set, errors=errors)
    for index, source in enumerate(sources):
        label = f"mvpa_set.pattern_sources[{index}]"
        name = _optional_text(source.get("name") or source.get("id"))
        if name is None:
            errors.append(f"{label}.name must be defined.")
        else:
            _validate_safe_identifier(name, f"{label}.name", errors)
        backend = _optional_text(source.get("backend"))
        if backend is None:
            errors.append(f"{label}.backend must be defined.")
        else:
            adapter = adapter_registry.adapter(backend)
            if adapter is None:
                errors.append(
                    f"{label}.backend must be one of: {', '.join(sorted(adapter_registry.names))}."
                )
            else:
                errors.extend(adapter.validate_source(source, label))
        _validate_optional_root_ref(source.get("root_ref"), f"{label}.root_ref", errors)
        _validate_source_relative_path_fields(source, label, errors)


def _parse_pattern_source(source: Mapping[str, Any]) -> PatternSourceConfig:
    return PatternSourceConfig(
        name=str(source.get("name") or source.get("id")).strip(),
        backend=str(source.get("backend")).strip(),
        root_ref=_optional_text(source.get("root_ref")),
        path=_optional_text(source.get("path") or source.get("relative_path")),
        pattern=_optional_text(source.get("pattern")),
        fields=dict(source),
    )


def _validate_pattern_source_selection_contract(
    mvpa_set: Mapping[str, Any],
    errors: list[str],
    *,
    adapter_registry: PatternSourceAdapterRegistry,
) -> None:
    sources = _pattern_source_mappings(mvpa_set)
    materialized = tuple(
        source
        for source in sources
        if _optional_text(source.get("backend")) == PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE
    )
    if materialized and _unit_selection_mode(mvpa_set) != UNIT_SELECTION_EXACT:
        errors.append(
            "mvpa_set materialized_pattern_table sources require unit_selection.mode=exact_units."
        )

    representation_kinds: set[str] = set()
    for source in sources:
        backend = _optional_text(source.get("backend"))
        adapter = adapter_registry.adapter(backend) if backend is not None else None
        if adapter is not None:
            representation_kinds.update(adapter.capabilities.representation_kinds)
    if "image" in representation_kinds and "prepared_features" in representation_kinds:
        errors.append(
            "mvpa_set must not mix image and prepared_features pattern-source representations in v1."
        )


def _pattern_source_mappings(mvpa_set: Mapping[str, Any], *, errors: list[str] | None = None) -> list[Mapping[str, Any]]:
    return _mapping_list(mvpa_set.get("pattern_sources"), "mvpa_set.pattern_sources", errors, require_non_empty=True)


def _validate_roi_sources(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    sources = _roi_source_mappings(mvpa_set, errors=errors)
    for index, source in enumerate(sources):
        label = f"mvpa_set.roi_sources[{index}]"
        name = _optional_text(source.get("name") or source.get("id"))
        if name is None:
            errors.append(f"{label}.name must be defined.")
        else:
            _validate_safe_identifier(name, f"{label}.name", errors)
        kind = _roi_source_kind(source, label, errors)
        if kind in _ROI_SET_SOURCE_TYPES:
            roi_set_ref = _optional_text(source.get("roi_set_ref") or source.get("roi_set"))
            if roi_set_ref is None:
                errors.append(f"{label}.roi_set_ref or {label}.roi_set must be defined for source {kind}.")
            else:
                _validate_safe_identifier(roi_set_ref, f"{label}.roi_set_ref", errors)
        elif kind == "explicit_masks":
            _validate_explicit_masks(source, label, errors)
        elif kind == ROI_SOURCE_MATERIALIZED_FEATURES:
            _validate_materialized_feature_identity(source, label, errors)
        _validate_optional_root_ref(source.get("root_ref"), f"{label}.root_ref", errors)
        _validate_source_relative_path_fields(source, label, errors)
        _validate_roi_source_missing(source, label, errors)


def _parse_roi_source(source: Mapping[str, Any]) -> RoiSourceConfig:
    kind = _roi_source_kind(source, "mvpa_set.roi_sources[]", [])
    masks = tuple(dict(mask) for mask in _mask_mappings(source))
    return RoiSourceConfig(
        name=str(source.get("name") or source.get("id")).strip(),
        source=kind or str(source.get("source")).strip(),
        roi_set_ref=_optional_text(source.get("roi_set_ref") or source.get("roi_set")),
        root_ref=_optional_text(source.get("root_ref")),
        path=_optional_text(source.get("path") or source.get("relative_path")),
        pattern=_optional_text(source.get("pattern") or source.get("mask_pattern")),
        masks=masks,
        fields=dict(source),
    )


def _roi_source_mappings(mvpa_set: Mapping[str, Any], *, errors: list[str] | None = None) -> list[Mapping[str, Any]]:
    return _mapping_list(mvpa_set.get("roi_sources"), "mvpa_set.roi_sources", errors, require_non_empty=True)


def _roi_source_kind(source: Mapping[str, Any], label: str, errors: list[str]) -> str | None:
    source_text = _optional_text(source.get("source"))
    has_roi_ref = _optional_text(source.get("roi_set_ref") or source.get("roi_set")) is not None
    has_masks = _has_mask_declarations(source) or ("mask_template" in source and not has_roi_ref)
    if source_text is not None:
        if source_text not in ALLOWED_ROI_SOURCE_TYPES:
            errors.append(f"{label}.source must be one of: {', '.join(sorted(ALLOWED_ROI_SOURCE_TYPES))}.")
            return None
        if source_text in _ROI_SET_SOURCE_TYPES and has_masks:
            errors.append(f"{label} must not mix roi_set references with explicit mask declarations.")
        if source_text == "explicit_masks" and has_roi_ref:
            errors.append(f"{label} must not mix explicit mask declarations with roi_set references.")
        if source_text == ROI_SOURCE_MATERIALIZED_FEATURES and (has_roi_ref or has_masks):
            errors.append(
                f"{label} materialized feature identities must not declare ROI-set references or mask paths."
            )
        return source_text
    if has_roi_ref and has_masks:
        errors.append(f"{label} must not mix roi_set references with explicit mask declarations.")
        return None
    if has_roi_ref:
        return ROI_SOURCE_ROI_SET
    if has_masks:
        return ROI_SOURCE_EXPLICIT_MASKS
    errors.append(f"{label}.source must be one of: {', '.join(sorted(ALLOWED_ROI_SOURCE_TYPES))}.")
    return None


def _validate_roi_source_missing(source: Mapping[str, Any], label: str, errors: list[str]) -> None:
    missing = source.get("missing")
    if missing is None:
        return
    if not isinstance(missing, Mapping):
        errors.append(f"{label}.missing must contain a mapping when configured.")
        return
    for key, value in missing.items():
        key_text = str(key)
        if key_text not in _ROI_SOURCE_MISSING_KEYS:
            continue
        policy = _optional_text(value)
        if policy not in ALLOWED_MISSING_INPUT_POLICIES:
            errors.append(
                f"{label}.missing.{key_text} must be one of: "
                f"{', '.join(sorted(ALLOWED_MISSING_INPUT_POLICIES))}."
            )


def _validate_explicit_masks(source: Mapping[str, Any], label: str, errors: list[str]) -> None:
    masks = _mask_mappings(source, errors=errors, label=f"{label}.masks")
    if not masks:
        path_value = _source_path_value(source)
        if path_value is None:
            errors.append(f"{label}.masks or {label}.path must define at least one explicit mask reference.")
            return
        root_ref = _optional_text(source.get("root_ref"))
        root = _optional_text(source.get("root"))
        if root_ref is None and root is None:
            errors.append(f"{label}.root_ref must be defined for explicit mask references.")
        else:
            if root_ref is not None:
                _validate_safe_identifier(root_ref, f"{label}.root_ref", errors)
        _validate_relative_path(path_value, f"{label}.path", errors)
        return
    source_root_ref = _optional_text(source.get("root_ref"))
    source_root = _optional_text(source.get("root"))
    for index, mask in enumerate(masks):
        mask_label = f"{label}.masks[{index}]"
        roi_label = _optional_text(mask.get("label") or mask.get("roi_label"))
        if roi_label is not None:
            _validate_bids_label_or_placeholder(roi_label, f"{mask_label}.label", errors)
        root_ref = _optional_text(mask.get("root_ref")) or source_root_ref
        root = _optional_text(mask.get("root")) or source_root
        if root_ref is None and root is None:
            errors.append(f"{mask_label}.root_ref must be defined for explicit mask references.")
        else:
            if root_ref is not None:
                _validate_safe_identifier(root_ref, f"{mask_label}.root_ref", errors)
        path_value = _mask_path_value(mask)
        if path_value is None:
            errors.append(f"{mask_label}.path or {mask_label}.pattern must be defined for explicit mask references.")
        else:
            _validate_relative_path(path_value, f"{mask_label}.path", errors)


def _validate_materialized_feature_identity(
    source: Mapping[str, Any],
    label: str,
    errors: list[str],
) -> None:
    labels = _string_sequence(source.get("roi_labels"))
    if not labels:
        errors.append(f"{label}.roi_labels must define at least one prepared-vector ROI label.")
    for index, roi_label in enumerate(labels):
        _validate_safe_identifier(roi_label, f"{label}.roi_labels[{index}]", errors)
    for field_name in ("feature_space_id", "roi_definition_id"):
        value = _optional_text(source.get(field_name))
        if value is None:
            errors.append(f"{label}.{field_name} must be defined for materialized feature identities.")
        else:
            _validate_safe_identifier(value, f"{label}.{field_name}", errors)


def _has_mask_declarations(source: Mapping[str, Any]) -> bool:
    return any(
        key in source
        for key in (
            "masks",
            "mask",
            "roi_masks",
            "explicit_masks",
            "path",
            "relative_path",
            "pattern",
            "mask_path",
            "mask_pattern",
        )
    )


def _mask_mappings(
    source: Mapping[str, Any],
    *,
    errors: list[str] | None = None,
    label: str = "masks",
) -> list[Mapping[str, Any]]:
    raw = source.get("masks", source.get("roi_masks", source.get("explicit_masks", source.get("mask"))))
    if isinstance(raw, Mapping):
        return [raw]
    return _mapping_list(raw, label, errors, require_non_empty=False)


def _mask_path_value(mask: Mapping[str, Any]) -> str | None:
    return _optional_text(
        mask.get("path")
        or mask.get("relative_path")
        or mask.get("pattern")
        or mask.get("mask_path")
        or mask.get("mask_pattern")
        or mask.get("mask_template")
    )


def _source_path_value(source: Mapping[str, Any]) -> str | None:
    return _optional_text(
        source.get("path")
        or source.get("relative_path")
        or source.get("pattern")
        or source.get("mask_path")
        or source.get("mask_pattern")
        or source.get("mask_template")
    )


def _validate_event_thresholds(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    thresholds = _event_threshold_payload(mvpa_set)
    if thresholds is None:
        return
    for key in ("min_events_per_condition_per_run", "min_runs_per_condition"):
        if key in thresholds:
            _validate_non_negative_int(thresholds[key], f"mvpa_set.event_thresholds.{key}", errors)


def _parse_event_thresholds(mvpa_set: Mapping[str, Any]) -> EventThresholdConfig | None:
    thresholds = _event_threshold_payload(mvpa_set)
    if thresholds is None:
        return None
    return EventThresholdConfig(
        min_events_per_condition_per_run=_optional_int(thresholds.get("min_events_per_condition_per_run")),
        min_runs_per_condition=_optional_int(thresholds.get("min_runs_per_condition")),
        fields=dict(thresholds),
    )


def _event_threshold_payload(mvpa_set: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if isinstance(mvpa_set.get("event_thresholds"), Mapping):
        return mvpa_set["event_thresholds"]  # type: ignore[index]
    events = mvpa_set.get("events")
    if isinstance(events, Mapping) and isinstance(events.get("thresholds"), Mapping):
        return events["thresholds"]  # type: ignore[index]
    return None


def _validate_condition_pairs(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    pairs = _condition_pair_mappings(mvpa_set, errors=errors)
    condition_ids = set(_configured_condition_ids(mvpa_set))
    seen_pairs: list[frozenset[str]] = []
    seen_pair_ids: list[str] = []
    for index, pair in enumerate(pairs):
        label = f"mvpa_set.condition_pairs[{index}]"
        left = _optional_text(pair.get("left") or pair.get("condition_id_a") or pair.get("condition_a"))
        right = _optional_text(pair.get("right") or pair.get("condition_id_b") or pair.get("condition_b"))
        if left is None:
            errors.append(f"{label}.left must be defined.")
        else:
            _validate_selector_label(left, f"{label}.left", errors)
            if condition_ids and left not in condition_ids:
                errors.append(f"{label}.left references unknown condition id: {left}.")
        if right is None:
            errors.append(f"{label}.right must be defined.")
        else:
            _validate_selector_label(right, f"{label}.right", errors)
            if condition_ids and right not in condition_ids:
                errors.append(f"{label}.right references unknown condition id: {right}.")
        if left is not None and right is not None:
            if left == right:
                errors.append(f"{label} must contain two distinct condition ids.")
            pair_key = frozenset((left, right))
            if pair_key in seen_pairs:
                errors.append(f"mvpa_set.condition_pairs contains duplicate unordered pair: {left}/{right}.")
            seen_pairs.append(pair_key)
        pair_id = _optional_text(pair.get("id") or pair.get("name"))
        if pair_id is not None:
            _validate_safe_identifier(pair_id, f"{label}.id", errors)
            seen_pair_ids.append(pair_id)
    for duplicate in _duplicates(seen_pair_ids):
        errors.append(f"mvpa_set.condition_pairs contains duplicate pair id: {duplicate}.")


def _parse_condition_pair(pair: Mapping[str, Any]) -> ConditionPairConfig:
    left = str(pair.get("left") or pair.get("condition_id_a") or pair.get("condition_a")).strip()
    right = str(pair.get("right") or pair.get("condition_id_b") or pair.get("condition_b")).strip()
    return ConditionPairConfig(
        id=_optional_text(pair.get("id") or pair.get("name")) or _generated_pair_id(left, right),
        condition_id_a=left,
        condition_id_b=right,
        fields=dict(pair),
    )


def _condition_pair_mappings(
    mvpa_set: Mapping[str, Any],
    *,
    errors: list[str] | None = None,
) -> list[Mapping[str, Any]]:
    distance = mvpa_set.get("distance")
    raw = mvpa_set.get("condition_pairs")
    if raw is None and isinstance(distance, Mapping):
        raw = distance.get("condition_pairs")
    if isinstance(raw, Mapping):
        if _optional_text(raw.get("mode")) is not None:
            return _condition_pair_mappings_from_mode(mvpa_set, raw, errors=errors)
        rows: list[Mapping[str, Any]] = []
        for pair_id, value in raw.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("id", pair_id)
                rows.append(row)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                values = tuple(value)
                if len(values) == 2:
                    rows.append({"id": pair_id, "left": values[0], "right": values[1]})
                elif errors is not None:
                    errors.append(f"mvpa_set.condition_pairs.{pair_id} must contain exactly two condition ids.")
            elif errors is not None:
                errors.append(f"mvpa_set.condition_pairs.{pair_id} must contain a mapping or two-condition sequence.")
        return rows
    return _mapping_list(raw, "mvpa_set.condition_pairs", errors, require_non_empty=False)


def _condition_pair_mappings_from_mode(
    mvpa_set: Mapping[str, Any],
    raw: Mapping[str, Any],
    *,
    errors: list[str] | None,
) -> list[Mapping[str, Any]]:
    mode = _optional_text(raw.get("mode"))
    if mode != CONDITION_PAIR_MODE_ALL_PAIRS:
        if errors is not None:
            errors.append(f"mvpa_set.condition_pairs.mode must be {CONDITION_PAIR_MODE_ALL_PAIRS!r}.")
        return []
    configured_conditions = _configured_condition_ids(mvpa_set)
    selected_conditions = _string_sequence(raw.get("conditions", raw.get("condition_ids"))) or configured_conditions
    if len(selected_conditions) < 2:
        if errors is not None:
            errors.append("mvpa_set.condition_pairs.conditions must define at least two conditions for all_pairs mode.")
        return []
    for duplicate in _duplicates(selected_conditions):
        if errors is not None:
            errors.append(f"mvpa_set.condition_pairs.conditions contains duplicate condition id: {duplicate}.")
    known_conditions = set(configured_conditions)
    for condition_id in selected_conditions:
        if not _SAFE_SELECTOR_VALUE.fullmatch(condition_id):
            if errors is not None:
                errors.append(f"mvpa_set.condition_pairs.conditions contains unsafe condition id: {condition_id}.")
        elif known_conditions and condition_id not in known_conditions and errors is not None:
            errors.append(f"mvpa_set.condition_pairs.conditions references unknown condition id: {condition_id}.")
    id_template = _optional_text(raw.get("id_template") or raw.get("pair_id_template")) or "{condition_a}_minus_{condition_b}"
    rows: list[Mapping[str, Any]] = []
    for left_index, left in enumerate(selected_conditions):
        for right in selected_conditions[left_index + 1 :]:
            rows.append(
                {
                    "id": _render_condition_pair_id(id_template, left=left, right=right),
                    "left": left,
                    "right": right,
                    "mode": CONDITION_PAIR_MODE_ALL_PAIRS,
                }
            )
    return rows


def _configured_condition_ids(mvpa_set: Mapping[str, Any]) -> tuple[str, ...]:
    ids: list[str] = []
    for condition in _condition_mappings(mvpa_set):
        condition_id = _optional_text(condition.get("id") or condition.get("condition_id") or condition.get("name"))
        if condition_id is not None:
            ids.append(condition_id)
    return tuple(ids)


def _render_condition_pair_id(template: str, *, left: str, right: str) -> str:
    return (
        template.replace("{condition_a}", left)
        .replace("{condition_b}", right)
        .replace("{left}", left)
        .replace("{right}", right)
    )


def _validate_threshold_sweeps(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    for index, row in enumerate(_threshold_sweep_mappings(mvpa_set, errors=errors)):
        label = f"mvpa_set.threshold_sweeps[{index}]"
        threshold_id = _optional_text(row.get("id") or row.get("name"))
        if threshold_id is not None:
            _validate_safe_identifier(threshold_id, f"{label}.id", errors)
        min_events = _optional_strict_int(row.get("min_events", row.get("min_events_per_condition")))
        min_observations = _optional_strict_int(
            row.get("min_observations", row.get("min_observations_per_condition"))
        )
        if min_events is None and min_observations is None:
            errors.append(f"{label} must define min_events or min_observations.")
        if ("min_events" in row or "min_events_per_condition" in row) and min_events is None:
            errors.append(f"{label}.min_events must be an integer.")
        elif min_events is not None and min_events < 0:
            errors.append(f"{label}.min_events must be non-negative.")
        if ("min_observations" in row or "min_observations_per_condition" in row) and min_observations is None:
            errors.append(f"{label}.min_observations must be an integer.")
        elif min_observations is not None and min_observations < 0:
            errors.append(f"{label}.min_observations must be non-negative.")


def _parse_threshold_sweep(row: Mapping[str, Any], *, index: int) -> ThresholdSweepConfig:
    return ThresholdSweepConfig(
        id=_optional_text(row.get("id") or row.get("name")) or f"threshold-{index}",
        min_events=_optional_strict_int(row.get("min_events", row.get("min_events_per_condition"))),
        min_observations=_optional_strict_int(
            row.get("min_observations", row.get("min_observations_per_condition"))
        ),
        fields=dict(row),
    )


def _threshold_sweep_mappings(
    mvpa_set: Mapping[str, Any],
    *,
    errors: list[str] | None = None,
) -> list[Mapping[str, Any]]:
    raw = mvpa_set.get("threshold_sweeps")
    if raw is None:
        thresholds = mvpa_set.get("thresholds")
        if isinstance(thresholds, Mapping):
            raw = thresholds.get("sweeps")
    return _mapping_list(raw, "mvpa_set.threshold_sweeps", errors, require_non_empty=False)


def _validate_exclusions(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    rules = _exclusion_mapping_rows(mvpa_set)
    for index, (rule, source_config_field) in enumerate(rules):
        label = f"{source_config_field}[{index}]"
        rule_id = _optional_text(rule.get("id") or rule.get("name"))
        if rule_id is None:
            errors.append(f"{label}.id must be defined.")
        else:
            _validate_safe_identifier(rule_id, f"{label}.id", errors)
        selectors = rule.get("match") if isinstance(rule.get("match"), Mapping) else rule
        if isinstance(selectors, Mapping):
            for key in ("subject", "subject_id", "session", "session_id", "run", "run_id"):
                value = _optional_text(selectors.get(key))
                if value is not None:
                    _validate_selector_label(value, f"{label}.{key}", errors)


def _parse_exclusion(rule: Mapping[str, Any], *, source_config_field: str) -> ExclusionRule:
    payload = dict(rule)
    match = payload.get("match")
    if isinstance(match, Mapping):
        payload.update(match)
    return ExclusionRule(
        id=str(rule.get("id") or rule.get("name")).strip(),
        reason=_optional_text(rule.get("reason")),
        subject_id=_optional_text(_first_present(payload, ("subject_id", "subject"))),
        session_id=_optional_text(_first_present(payload, ("session_id", "session"))),
        run_id=_optional_text(_first_present(payload, ("run_id", "run"))),
        source_config_field=source_config_field,
        fields=dict(rule),
    )


def _exclusion_mappings(mvpa_set: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row, _source in _exclusion_mapping_rows(mvpa_set)]


def _exclusion_mapping_rows(mvpa_set: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], str]]:
    rows: list[tuple[Mapping[str, Any], str]] = []
    exclusions = mvpa_set.get("exclusions")
    if isinstance(exclusions, Mapping):
        rows.extend(
            (row, "mvpa_set.exclusions.rules")
            for row in _mapping_list(exclusions.get("rules"), "mvpa_set.exclusions.rules", None, require_non_empty=False)
        )
    rows.extend(
        (row, "mvpa_set.run_exclusions")
        for row in _mapping_list(mvpa_set.get("run_exclusions"), "mvpa_set.run_exclusions", None, require_non_empty=False)
    )
    rows.extend(
        (row, "mvpa_set.exclusion_rules")
        for row in _mapping_list(mvpa_set.get("exclusion_rules"), "mvpa_set.exclusion_rules", None, require_non_empty=False)
    )
    return rows


def _validate_mean_centering(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    payload = _mean_centering_payload(mvpa_set)
    scope = _optional_text(payload.get("scope")) or "none"
    if scope not in _ALLOWED_MEAN_CENTERING_SCOPES:
        errors.append(
            "mvpa_set.mean_centering.scope must be one of: "
            f"{', '.join(sorted(_ALLOWED_MEAN_CENTERING_SCOPES))}."
        )


def _parse_mean_centering(mvpa_set: Mapping[str, Any]) -> MeanCenteringConfig:
    payload = dict(_mean_centering_payload(mvpa_set))
    enabled = bool(payload.get("enabled", False))
    scope = _optional_text(payload.get("scope")) or ("roi" if enabled else "none")
    if scope == "within_roi":
        scope = "roi"
    if not enabled:
        scope = "none"
    return MeanCenteringConfig(enabled=enabled, scope=scope, fields=payload)


def _mean_centering_payload(mvpa_set: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = mvpa_set.get("within_roi_mean_centering")
    if raw is not None:
        if isinstance(raw, Mapping):
            payload = dict(raw)
            payload.setdefault("enabled", True)
            payload.setdefault("scope", "roi")
            return payload
        return {"enabled": bool(raw), "scope": "roi" if bool(raw) else "none"}

    raw = mvpa_set.get("mean_centering")
    if isinstance(raw, Mapping):
        payload = dict(raw)
        if "within_roi" in payload and "enabled" not in payload:
            payload["enabled"] = bool(payload["within_roi"])
        if payload.get("enabled") and "scope" not in payload:
            payload["scope"] = "roi"
        return payload
    if raw is None:
        return {"enabled": False, "scope": "none"}
    return {"enabled": bool(raw), "scope": "roi" if bool(raw) else "none"}


def _generated_pair_id(left: str, right: str) -> str:
    return f"{_safe_generated_label(left)}__{_safe_generated_label(right)}"


def _safe_generated_label(value: str) -> str:
    text = "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value)
    return text.strip("._-") or "condition"


def _validate_selector_label(value: str, label: str, errors: list[str]) -> None:
    if not _SAFE_SELECTOR_VALUE.fullmatch(value):
        errors.append(f"{label} must be a safe selector label or placeholder.")


def _validate_distance(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    distance = mvpa_set.get("distance")
    if distance is None:
        errors.append("mvpa_set.distance must contain a mapping.")
        return
    if not isinstance(distance, Mapping):
        errors.append("mvpa_set.distance must contain a mapping.")
        return
    metrics = _distance_metrics(distance)
    if not metrics:
        errors.append("mvpa_set.distance.metrics must define at least one metric.")
    for metric in metrics:
        if metric not in ALLOWED_METRICS:
            errors.append(f"mvpa_set.distance.metrics must contain only: {', '.join(sorted(ALLOWED_METRICS))}.")
            break
    engines = _distance_engines(distance)
    if not engines:
        errors.append("mvpa_set.distance.engine must define at least one engine.")
    for engine in engines:
        if engine not in ALLOWED_ENGINES:
            errors.append(f"mvpa_set.distance.engine must be one of: {', '.join(sorted(ALLOWED_ENGINES))}.")
            break
    cv_unit = _distance_cv_unit(distance)
    if cv_unit not in ALLOWED_CV_UNITS:
        errors.append(f"mvpa_set.distance.cross_validation.unit must be one of: {', '.join(sorted(ALLOWED_CV_UNITS))}.")
    grouping_columns = _distance_grouping_columns(mvpa_set, distance)
    for index, column in enumerate(grouping_columns):
        _validate_safe_identifier(column, f"mvpa_set.distance.grouping_columns[{index}]", errors)
    _validate_noise_normalization(_noise_normalization_payload(mvpa_set, distance), errors)


def _parse_distance(mvpa_set: Mapping[str, Any]) -> DistanceConfig:
    distance = mvpa_set["distance"]
    assert isinstance(distance, Mapping)
    noise = _parse_noise_normalization(_noise_normalization_payload(mvpa_set, distance))
    grouping_columns = _distance_grouping_columns(mvpa_set, distance)
    return DistanceConfig(
        metrics=tuple(_distance_metrics(distance) or ("crossnobis",)),
        engines=tuple(_distance_engines(distance) or ("native_reference",)),
        cv_unit=_distance_cv_unit(distance),
        grouping_columns=tuple(grouping_columns),
        noise_normalization=noise,
        fields=dict(distance),
    )


def _distance_metrics(distance: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(_string_sequence(distance.get("metrics", distance.get("metric", "crossnobis"))))


def _distance_engines(distance: Mapping[str, Any]) -> tuple[str, ...]:
    raw = distance.get("engine", distance.get("engines", distance.get("engine_name", "native_reference")))
    if isinstance(raw, Mapping):
        values: list[str] = []
        for key in ("preferred", "fallback", "name"):
            text = _optional_text(raw.get(key))
            if text is not None:
                values.append(text)
        return tuple(values)
    return tuple(_string_sequence(raw))


def _distance_cv_unit(distance: Mapping[str, Any]) -> str:
    raw = distance.get("cv_unit")
    cross_validation = distance.get("cross_validation", distance.get("cv"))
    if raw is None and isinstance(cross_validation, Mapping):
        raw = cross_validation.get("unit")
    return _optional_text(raw) or "run"


def _distance_grouping_columns(mvpa_set: Mapping[str, Any], distance: Mapping[str, Any]) -> tuple[str, ...]:
    raw = distance.get("grouping_columns")
    cross_validation = distance.get("cross_validation", distance.get("cv"))
    if raw is None and isinstance(cross_validation, Mapping):
        raw = cross_validation.get("grouping_columns") or cross_validation.get("group_by")
    if raw is None:
        raw = mvpa_set.get("grouping_columns") or mvpa_set.get("group_by")
    return tuple(_string_sequence(raw))


def _validate_noise_normalization(payload: Mapping[str, Any], errors: list[str]) -> None:
    method = _optional_text(payload.get("method")) or "identity"
    if method not in ALLOWED_NOISE_NORMALIZATION_METHODS:
        errors.append(
            "mvpa_set.distance.noise_normalization.method must be one of: "
            f"{', '.join(sorted(ALLOWED_NOISE_NORMALIZATION_METHODS))}."
        )
    policy = _optional_text(payload.get("nonpositive_policy") or payload.get("nonpositive"))
    if policy is not None and policy not in ALLOWED_NOISE_NONPOSITIVE_POLICIES:
        errors.append(
            "mvpa_set.distance.noise_normalization.nonpositive_policy must be one of: "
            f"{', '.join(sorted(ALLOWED_NOISE_NONPOSITIVE_POLICIES))}."
        )
    min_retained_features = payload.get("min_retained_features")
    if min_retained_features is not None and _optional_positive_int(min_retained_features) is None:
        errors.append("mvpa_set.distance.noise_normalization.min_retained_features must be an integer of at least 1.")
    warn_fraction = payload.get("warn_dropped_feature_fraction")
    if warn_fraction is not None and _optional_fraction(warn_fraction) is None:
        errors.append("mvpa_set.distance.noise_normalization.warn_dropped_feature_fraction must be between 0 and 1.")


def _parse_noise_normalization(payload: Mapping[str, Any]) -> NoiseNormalizationConfig:
    return NoiseNormalizationConfig(
        method=_optional_text(payload.get("method")) or "identity",
        variance_source=_optional_text(payload.get("variance_source")),
        nonpositive_policy=_normalized_noise_nonpositive_policy(
            _optional_text(payload.get("nonpositive_policy") or payload.get("nonpositive"))
            or DEFAULT_NOISE_NONPOSITIVE_POLICY
        ),
        min_retained_features=_optional_positive_int(payload.get("min_retained_features"))
        or DEFAULT_MIN_RETAINED_FEATURES,
        warn_dropped_feature_fraction=_optional_fraction(payload.get("warn_dropped_feature_fraction"))
        if payload.get("warn_dropped_feature_fraction") is not None
        else DEFAULT_WARN_DROPPED_FEATURE_FRACTION,
        fields=dict(payload),
    )


def _noise_normalization_payload(mvpa_set: Mapping[str, Any], distance: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = distance.get("noise_normalization", mvpa_set.get("noise_normalization"))
    if isinstance(raw, Mapping):
        return raw
    if raw is not None:
        return {"method": raw}
    return {}


def _validate_outputs(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    outputs = mvpa_set.get("outputs")
    if not isinstance(outputs, Mapping):
        errors.append("mvpa_set.outputs must contain at least one root-ref output declaration.")
        return
    roots = _output_root_mappings(outputs, errors=errors)
    if not roots:
        errors.append("mvpa_set.outputs must contain at least one root-ref output declaration.")
        return
    for name, root in roots:
        label = f"mvpa_set.outputs.{name}"
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


def _parse_outputs(mvpa_set: Mapping[str, Any]) -> OutputConfig:
    outputs = mvpa_set["outputs"]
    assert isinstance(outputs, Mapping)
    roots = tuple(
        OutputRootConfig(
            name=name,
            root_ref=str(root["root_ref"]).strip(),
            path=str(root.get("path") or root.get("relative_path")).strip(),
            fields=dict(root),
        )
        for name, root in _output_root_mappings(outputs)
    )
    return OutputConfig(roots=roots, fields=dict(outputs))


def _validate_runtime(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    runtime = mvpa_set.get("runtime")
    if runtime is None:
        return
    if not isinstance(runtime, Mapping):
        errors.append("mvpa_set.runtime must contain a mapping when declared.")
        return
    unsupported = sorted(str(key) for key in runtime if str(key) != "existing_output")
    if unsupported:
        errors.append(
            "mvpa_set.runtime supports only existing_output in v1; unsupported fields: "
            + ", ".join(unsupported)
            + "."
        )
    policy = _optional_text(runtime.get("existing_output")) or RUNTIME_EXISTING_OUTPUT_FAIL
    if policy != RUNTIME_EXISTING_OUTPUT_FAIL:
        errors.append("mvpa_set.runtime.existing_output must be 'fail' in v1.")


def _parse_runtime(mvpa_set: Mapping[str, Any]) -> RuntimeConfig:
    runtime = mvpa_set.get("runtime")
    if not isinstance(runtime, Mapping):
        return RuntimeConfig()
    return RuntimeConfig(
        existing_output=_optional_text(runtime.get("existing_output")) or RUNTIME_EXISTING_OUTPUT_FAIL,
        fields=dict(runtime),
    )


def _output_root_mappings(
    outputs: Mapping[str, Any],
    *,
    errors: list[str] | None = None,
) -> list[tuple[str, Mapping[str, Any]]]:
    if "root_ref" in outputs or "path" in outputs or "relative_path" in outputs:
        return [("root", outputs)]
    roots: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in outputs.items():
        if isinstance(value, Mapping):
            if _is_output_root_key(str(key)) or "root_ref" in value or "path" in value or "relative_path" in value:
                roots.append((str(key), value))
        elif _is_output_root_key(str(key)) and errors is not None:
            errors.append(f"mvpa_set.outputs.{key} must contain a mapping.")
    return roots


def _is_output_root_key(key: str) -> bool:
    return key.endswith("_root") or key in {"runtime", "published", "publication", "derivative", "report"}


def _validate_publication(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    publication = mvpa_set.get("publication")
    if publication is None:
        return
    if not isinstance(publication, Mapping):
        errors.append("mvpa_set.publication must contain a mapping when declared.")
        return
    derivative_name = _optional_text(publication.get("derivative_name"))
    if derivative_name is not None:
        _validate_path_segment(derivative_name, "mvpa_set.publication.derivative_name", errors)


def _parse_publication(mvpa_set: Mapping[str, Any]) -> PublicationConfig:
    publication = mvpa_set.get("publication")
    if not isinstance(publication, Mapping):
        return PublicationConfig()
    return PublicationConfig(
        enabled=bool(publication.get("enabled", False)),
        derivative_name=_optional_text(publication.get("derivative_name")),
        write_json_sidecars=bool(publication.get("write_json_sidecars", True)),
        write_provenance=bool(publication.get("write_provenance", True)),
        fields=dict(publication),
    )


def _validate_missing_input_policy(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    policy = _missing_input_policy_value(mvpa_set)
    if policy not in ALLOWED_MISSING_INPUT_POLICIES:
        errors.append(f"mvpa_set.missing_input_policy must be one of: {', '.join(sorted(ALLOWED_MISSING_INPUT_POLICIES))}.")


def _parse_missing_input_policy(mvpa_set: Mapping[str, Any]) -> MissingInputPolicy:
    raw = mvpa_set.get("missing_input_policy", mvpa_set.get("missing_inputs", mvpa_set.get("missing_input")))
    fields = dict(raw) if isinstance(raw, Mapping) else {}
    return MissingInputPolicy(policy=_missing_input_policy_value(mvpa_set), fields=fields)


def _missing_input_policy_value(mvpa_set: Mapping[str, Any]) -> str:
    raw = mvpa_set.get("missing_input_policy", mvpa_set.get("missing_inputs", mvpa_set.get("missing_input")))
    if isinstance(raw, Mapping):
        raw = raw.get("policy", raw.get("value"))
    return _optional_text(raw) or "warn"


def _validate_provenance(mvpa_set: Mapping[str, Any], errors: list[str]) -> None:
    provenance = mvpa_set.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        errors.append("mvpa_set.provenance must contain a mapping when declared.")


def _parse_provenance(mvpa_set: Mapping[str, Any]) -> ProvenanceConfig:
    provenance = mvpa_set.get("provenance")
    if not isinstance(provenance, Mapping):
        return ProvenanceConfig()
    return ProvenanceConfig(
        schema_version=_optional_text(provenance.get("schema_version")),
        fields=dict(provenance),
    )


def _validate_source_relative_path_fields(source: Mapping[str, Any], label: str, errors: list[str]) -> None:
    root_ref = _optional_text(source.get("root_ref"))
    root = _optional_text(source.get("root"))
    for key in ("path", "relative_path", "pattern", "mask_pattern"):
        value = _optional_text(source.get(key))
        if value is None:
            continue
        _validate_relative_path(value, f"{label}.{key}", errors)
        if root_ref is None and root is None:
            errors.append(f"{label}.root_ref must be defined when {key} is configured.")


def validate_fsl_feat_pe_source_fields(
    source: Mapping[str, Any],
    label: str,
) -> tuple[str, ...]:
    """Validate FSL FEAT source-owned fields without filesystem access."""

    errors: list[str] = []
    if "root" in source:
        root = source.get("root")
        if not isinstance(root, str):
            errors.append(f"{label}.root must be a string when configured.")
        elif _has_parent_traversal(root):
            errors.append(f"{label}.root must not contain parent-directory traversal.")

    for key in sorted(_FSL_FEAT_PE_TEMPLATE_FIELDS):
        if key not in source:
            continue
        value = source.get(key)
        if not isinstance(value, str):
            errors.append(f"{label}.{key} must be a string when configured.")
            continue
        _validate_relative_path(value, f"{label}.{key}", errors)

    if "case_sensitive" in source and not isinstance(source.get("case_sensitive"), bool):
        errors.append(f"{label}.case_sensitive must be a boolean when configured.")

    missing = source.get("missing")
    if missing is None:
        return tuple(errors)
    if not isinstance(missing, Mapping):
        errors.append(f"{label}.missing must contain a mapping when configured.")
        return tuple(errors)
    for key, value in missing.items():
        key_text = str(key)
        if key_text not in _FSL_FEAT_PE_MISSING_KEYS:
            continue
        policy = _optional_text(value)
        if policy not in ALLOWED_MISSING_INPUT_POLICIES:
            errors.append(
                f"{label}.missing.{key_text} must be one of: "
                f"{', '.join(sorted(ALLOWED_MISSING_INPUT_POLICIES))}."
            )
    return tuple(errors)


def _validate_optional_root_ref(value: Any, label: str, errors: list[str]) -> None:
    text = _optional_text(value)
    if text is not None:
        _validate_safe_identifier(text, label, errors)


def _validate_safe_identifier(value: str, label: str, errors: list[str]) -> None:
    if not _SAFE_IDENTIFIER.fullmatch(value):
        errors.append(f"{label} must be a safe identifier using letters, numbers, dots, underscores, or hyphens.")


def _validate_bids_label_or_placeholder(value: str, label: str, errors: list[str]) -> None:
    if _has_placeholder(value):
        return
    if not _BIDS_LABEL_VALUE.fullmatch(value):
        errors.append(f"{label} must contain only letters and numbers for BIDS-like label use.")


def _validate_path_segment(value: str, label: str, errors: list[str]) -> None:
    if not _SAFE_IDENTIFIER.fullmatch(value):
        errors.append(f"{label} must be a safe path segment.")


def _validate_relative_path(value: str, label: str, errors: list[str]) -> None:
    text = value.strip()
    if not text:
        errors.append(f"{label} must be a non-empty relative path.")
        return
    if text.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE_PATH.match(text):
        errors.append(f"{label} must be a relative path under a root_ref.")
    if _has_parent_traversal(text):
        errors.append(f"{label} must not contain parent-directory traversal.")


def _validate_relative_path_strings(payload: Any, label: str, errors: list[str]) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child_label = f"{label}.{key}"
            if _is_path_like_key(str(key)):
                for path_index, text in enumerate(_path_like_values(value)):
                    suffix = f"[{path_index}]" if path_index else ""
                    _validate_relative_path(text, f"{child_label}{suffix}", errors)
            else:
                _validate_relative_path_strings(value, child_label, errors)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _validate_relative_path_strings(value, f"{label}[{index}]", errors)


def _is_path_like_key(key: str) -> bool:
    lower = key.lower()
    return lower in _PATH_LIKE_KEYS or lower.endswith("_path") or lower.endswith("_pattern")


def _path_like_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _has_parent_traversal(value: str) -> bool:
    return any(part == ".." for part in re.split(r"[\\/]+", value))


def _validate_no_personal_paths(payload: Any, label: str, errors: list[str]) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            _validate_no_personal_paths(value, f"{label}.{key}", errors)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _validate_no_personal_paths(value, f"{label}[{index}]", errors)
    elif isinstance(payload, str):
        for pattern in _ABSOLUTE_PATH_PATTERNS:
            if pattern.search(payload):
                errors.append(f"{label} contains a personal absolute path; use a root_ref plus relative path instead.")
                break


def _validate_non_negative_int(value: Any, label: str, errors: list[str]) -> None:
    normalized = _optional_int(value)
    if normalized is None:
        errors.append(f"{label} must be an integer.")
    elif normalized < 0:
        errors.append(f"{label} must be greater than or equal to zero.")


def _mapping_list(
    value: Any,
    label: str,
    errors: list[str] | None,
    *,
    require_non_empty: bool,
) -> list[Mapping[str, Any]]:
    if value is None:
        if require_non_empty and errors is not None:
            errors.append(f"{label} must define at least one mapping.")
        return []
    if not isinstance(value, list) or (require_non_empty and not value):
        if errors is not None:
            errors.append(f"{label} must define at least one mapping.")
        return []
    mappings: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            if errors is not None:
                errors.append(f"{label}[{index}] must contain a mapping.")
            continue
        mappings.append(item)
    return mappings


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
            else:
                values.append("")
        return tuple(values)
    text = _optional_text(value)
    return (text,) if text is not None else ()


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_strict_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"-?\d+", text):
            return int(text)
    return None


def _optional_positive_int(value: Any) -> int | None:
    resolved = _optional_strict_int(value)
    if resolved is None or resolved < 1:
        return None
    return resolved


def _optional_fraction(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved) or resolved < 0.0 or resolved > 1.0:
        return None
    return resolved


def _normalized_noise_nonpositive_policy(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", "strict", "fail", "drop_pattern"}:
        return "strict"
    if normalized in {"drop_features", "filter_nonpositive_features"}:
        return "drop_features"
    return normalized


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _has_placeholder(value: str) -> bool:
    return ("{" in value and "}" in value) or ("<" in value and ">" in value)


__all__ = [
    "ALLOWED_CV_UNITS",
    "ALLOWED_ENGINES",
    "ALLOWED_METRICS",
    "ALLOWED_MISSING_INPUT_POLICIES",
    "ALLOWED_NOISE_NONPOSITIVE_POLICIES",
    "ALLOWED_NOISE_NORMALIZATION_METHODS",
    "ALLOWED_PATTERN_BACKENDS",
    "ALLOWED_ROI_SOURCE_TYPES",
    "ALLOWED_UNIT_SELECTION_MODES",
    "DEFAULT_MIN_RETAINED_FEATURES",
    "DEFAULT_NOISE_NONPOSITIVE_POLICY",
    "DEFAULT_WARN_DROPPED_FEATURE_FRACTION",
    "ConditionDefinition",
    "ConditionPairConfig",
    "CV_UNIT_CUSTOM",
    "CV_UNIT_RUN",
    "CV_UNIT_SESSION",
    "CV_UNIT_SUBJECT",
    "DistanceConfig",
    "ENGINE_MANUAL_DIAGONAL_CROSSNOBIS_V1",
    "ENGINE_NATIVE_REFERENCE",
    "ENGINE_RSATOOLBOX",
    "EventThresholdConfig",
    "ExclusionRule",
    "MeanCenteringConfig",
    "METRIC_CORRELATION",
    "METRIC_CROSSNOBIS",
    "METRIC_EUCLIDEAN",
    "MissingInputPolicy",
    "MvpaEntities",
    "MvpaSetConfig",
    "MISSING_INPUT_POLICY_FAIL",
    "MISSING_INPUT_POLICY_SKIP",
    "MISSING_INPUT_POLICY_WARN",
    "NOISE_NORMALIZATION_DIAGONAL",
    "NOISE_NORMALIZATION_IDENTITY",
    "NoiseNormalizationConfig",
    "OutputConfig",
    "OutputRootConfig",
    "PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE",
    "PATTERN_BACKEND_FSL_FEAT_PE",
    "PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE",
    "PATTERN_BACKEND_NILEARN_GLM",
    "PATTERN_BACKEND_SURFACE_CIFTI",
    "PatternSourceConfig",
    "ProvenanceConfig",
    "PublicationConfig",
    "ROI_SOURCE_EXPLICIT_MASKS",
    "ROI_SOURCE_MATERIALIZED_FEATURES",
    "ROI_SOURCE_ROI_SET",
    "ROI_SOURCE_ROI_SET_PUBLICATION",
    "ROI_SOURCE_ROI_SET_RUNTIME",
    "RoiSourceConfig",
    "RUNTIME_EXISTING_OUTPUT_FAIL",
    "RuntimeConfig",
    "SubjectSessionRunSelector",
    "ThresholdSweepConfig",
    "UNIT_SELECTION_EXACT",
    "UNIT_SELECTION_LEGACY_CARTESIAN",
    "UnitSelectionConfig",
    "parse_mvpa_set_document",
    "validate_fsl_feat_pe_source_fields",
    "validate_mvpa_set_document",
]
