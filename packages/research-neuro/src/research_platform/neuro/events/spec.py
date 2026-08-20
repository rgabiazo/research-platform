from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class ErrorRule:
    kind: str
    left: str
    right: str | None = None
    value: str | None = None


@dataclass(frozen=True)
class PhaseSpec:
    run: int
    condition: str
    phase: str
    image_prefix: str
    onset_column: str
    response_time_column: str
    response_column: str
    trial_type: str
    instruction_trial_type: str
    duration: str
    instruction_duration: str
    instruction_offset: str
    block_n: str
    instruction_probe_type: str
    error_rule: ErrorRule


@dataclass(frozen=True)
class CompiledTransform:
    target_column: str
    operator: str
    args: dict[str, Any]


@dataclass(frozen=True)
class CompiledLookup:
    name: str
    source_row_sets: list[str]
    key_columns: list[str]
    value_column: str
    when_column_equals: dict[str, str]


@dataclass(frozen=True)
class CompiledValidation:
    operator: str
    args: dict[str, Any]


@dataclass(frozen=True)
class CompiledRowSet:
    name: str
    run: int
    condition: str
    phase: str
    selectors: list[dict[str, Any]]
    instruction_transforms: list[CompiledTransform]
    trial_transforms: list[CompiledTransform]
    context: dict[str, str]
    onset_sort_column: str


@dataclass(frozen=True)
class CompiledPlan:
    row_sets: list[CompiledRowSet]
    lookups: list[CompiledLookup]
    validations: list[CompiledValidation]
    output_columns: list[str]
    missing_value: str
    required_source_columns: set[str]


@dataclass(frozen=True)
class BuildSpec:
    name: str
    source_path: Path
    ops_path: Path
    sidecar_path: Path
    spec_hash: str
    source_encoding: str
    subject_column: str
    subject_regex: str
    subject_width: int
    session_column: str | None
    session_regex: str | None
    session_width: int | None
    run_column: str
    stim_column: str
    task: str
    acq_label: str | None
    dir_label: str | None
    datatype: str
    suffix: str
    ped_fallback_enabled: bool
    ped_dir_map: dict[str, str]
    sidecar_writes: bool
    sidecar_columns: dict[str, dict[str, Any]]
    stimuli_enabled: bool
    stimuli_source_roots: list[Path]
    columns: list[str]
    missing_value: str
    required_source_columns: set[str]
    analysis_include_labels: set[str]
    probe_type_map: dict[str, str]
    recognition_acc_map: dict[str, dict[str, str]]
    phases: list[PhaseSpec]
    compiled_plan: CompiledPlan


def _read_json_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not valid Phase 1 config. Use JSON-compatible YAML for this phase."
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must decode to an object.")
    return payload


def _sibling_config_path(spec_path: Path, suffix: str) -> Path:
    stem = spec_path.name.removesuffix(".yaml")
    return spec_path.with_name(f"{stem}{suffix}")


def _title_case(token: str) -> str:
    return token[:1].upper() + token[1:]


def _hash_spec_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _literal(value: str) -> dict[str, Any]:
    return {"source": "literal", "value": value}


def _missing() -> dict[str, Any]:
    return {"source": "missing"}


def _source_field(column: str) -> dict[str, Any]:
    return {"source": "source_field", "column": column}


def _context_value(name: str) -> dict[str, Any]:
    return {"source": "context_value", "name": name}


def _derived_value(name: str) -> dict[str, Any]:
    return {"source": "derived_value", "name": name}


def _copy_transform(target_column: str, value: dict[str, Any]) -> CompiledTransform:
    return CompiledTransform(target_column=target_column, operator="concat", args={"values": [value]})


def _compile_recognition_acc_map(recognition_acc_map: dict[str, dict[str, str]]) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for probe_type, responses in recognition_acc_map.items():
        for response, acc_label in responses.items():
            flattened[f"{probe_type}::{response}"] = acc_label
    return flattened


def _compile_error_transform(*, error_rule: ErrorRule) -> CompiledTransform:
    if error_rule.kind == "equals_literal":
        return CompiledTransform(
            target_column="is_error",
            operator="if_else",
            args={
                "condition": {
                    "operator": "equals_value",
                    "left": _derived_value(error_rule.left),
                    "right": _literal(str(error_rule.value)),
                },
                "when_true": _literal("1"),
                "when_false": _literal("0"),
            },
        )
    if error_rule.kind == "not_equals_literal":
        return CompiledTransform(
            target_column="is_error",
            operator="if_else",
            args={
                "condition": {
                    "operator": "not_equals_value",
                    "left": _derived_value(error_rule.left),
                    "right": _literal(str(error_rule.value)),
                },
                "when_true": _literal("1"),
                "when_false": _literal("0"),
            },
        )
    raise ValueError(f"Unsupported legacy error_rule.kind {error_rule.kind!r} for Phase A compiler.")


def _compile_trial_transforms(
    *,
    phase: PhaseSpec,
    stim_column: str,
    analysis_include_labels: set[str],
    probe_type_map: dict[str, str],
    recognition_acc_map: dict[str, dict[str, str]],
) -> list[CompiledTransform]:
    response_source = _source_field(phase.response_column)
    response_time_source = _source_field(phase.response_time_column)
    onset_source = _source_field(phase.onset_column)
    stimulus_source = _source_field(stim_column)
    transforms = [
        _copy_transform("onset", onset_source),
        _copy_transform("duration", _literal(phase.duration)),
        _copy_transform("trial_type", _literal(phase.trial_type)),
        CompiledTransform(
            target_column="stimulus_identity",
            operator="strip_prefix",
            args={"value": stimulus_source, "prefix": _context_value("image_prefix")},
        ),
        CompiledTransform(
            target_column="stim_id",
            operator="basename",
            args={"value": _derived_value("stimulus_identity")},
        ),
        CompiledTransform(
            target_column="stim_file",
            operator="if_else",
            args={
                "condition": {
                    "operator": "equals_value",
                    "left": _context_value("preserve_source_stim_file"),
                    "right": _literal("1"),
                },
                "when_true": stimulus_source,
                "when_false": _derived_value("stim_id"),
            },
        ),
        CompiledTransform(
            target_column="response_time",
            operator="if_else",
            args={
                "condition": {"operator": "present", "value": response_time_source},
                "when_true": response_time_source,
                "when_false": _missing(),
            },
        ),
        CompiledTransform(
            target_column="response",
            operator="if_else",
            args={
                "condition": {"operator": "present", "value": response_source},
                "when_true": response_source,
                "when_false": _missing(),
            },
        ),
        _copy_transform("phase", _literal(phase.phase)),
        _copy_transform("condition", _literal(phase.condition)),
        _copy_transform("is_instruction", _missing()),
        _copy_transform("block_n", _literal(phase.block_n)),
        CompiledTransform(target_column="trial_n", operator="sequence_number", args={"start": 2}),
    ]
    if phase.phase == "encoding":
        transforms.extend(
            [
                CompiledTransform(
                    target_column="enc_later_outcome",
                    operator="lookup_value",
                    args={
                        "lookup": "recognition_targets",
                        "key": [_literal(phase.condition), _derived_value("stimulus_identity")],
                        "default": _literal("not_tested"),
                    },
                ),
                _copy_transform("acc_label", _missing()),
                _copy_transform("probe_type", _missing()),
                CompiledTransform(
                    target_column="enc_is_tested",
                    operator="if_else",
                    args={
                        "condition": {
                            "operator": "not_equals_value",
                            "left": _derived_value("enc_later_outcome"),
                            "right": _literal("not_tested"),
                        },
                        "when_true": _literal("1"),
                        "when_false": _literal("0"),
                    },
                ),
                _copy_transform("analysis_include", _literal("0")),
                _compile_error_transform(error_rule=phase.error_rule),
            ]
        )
        return transforms

    recognition_acc_by_key = _compile_recognition_acc_map(recognition_acc_map)
    transforms.extend(
        [
            CompiledTransform(
                target_column="probe_type",
                operator="map_value",
                args={
                    "value": _source_field("image_old_new"),
                    "mapping": probe_type_map,
                    "default": _missing(),
                },
            ),
            CompiledTransform(
                target_column="acc_lookup_key",
                operator="concat",
                args={"values": [_derived_value("probe_type"), _literal("::"), _derived_value("response")]},
            ),
            CompiledTransform(
                target_column="acc_label",
                operator="map_value",
                args={
                    "value": _derived_value("acc_lookup_key"),
                    "mapping": recognition_acc_by_key,
                    "default": _missing(),
                },
            ),
            _copy_transform("enc_is_tested", _missing()),
            _copy_transform("enc_later_outcome", _missing()),
            CompiledTransform(
                target_column="is_error",
                operator="if_else",
                args={
                    "condition": {
                        "operator": "in_set",
                        "value": _derived_value("acc_label"),
                        "values": ["miss", "false_alarm"],
                    },
                    "when_true": _literal("1"),
                    "when_false": _literal("0"),
                },
            ),
            CompiledTransform(
                target_column="analysis_include",
                operator="if_else",
                args={
                    "condition": {
                        "operator": "in_set",
                        "value": _derived_value("acc_label"),
                        "values": sorted(analysis_include_labels),
                    },
                    "when_true": _literal("1"),
                    "when_false": _literal("0"),
                },
            ),
        ]
    )
    return transforms


def _compile_instruction_transforms(phase: PhaseSpec) -> list[CompiledTransform]:
    onset_source = _source_field(phase.onset_column)
    return [
        CompiledTransform(
            target_column="onset",
            operator="subtract_value",
            args={
                "left": {"source": "first_group_value", "value": onset_source},
                "right": _literal(phase.instruction_offset),
            },
        ),
        _copy_transform("duration", _literal(phase.instruction_duration)),
        _copy_transform("trial_type", _literal(phase.instruction_trial_type)),
        _copy_transform("stim_file", _missing()),
        _copy_transform("response_time", _missing()),
        _copy_transform("response", _missing()),
        _copy_transform("phase", _literal(phase.phase)),
        _copy_transform("condition", _literal(phase.condition)),
        _copy_transform("stim_id", _missing()),
        _copy_transform("acc_label", _missing()),
        _copy_transform("probe_type", _literal(phase.instruction_probe_type)),
        _copy_transform("enc_is_tested", _literal("0") if phase.phase == "encoding" else _missing()),
        _copy_transform("enc_later_outcome", _literal("not_tested") if phase.phase == "encoding" else _missing()),
        _copy_transform("is_instruction", _literal("1")),
        _copy_transform("is_error", _literal("0")),
        _copy_transform("block_n", _literal(phase.block_n)),
        _copy_transform("trial_n", _literal("1")),
        _copy_transform("analysis_include", _literal("0")),
    ]


def _iter_value_specs(value: Any) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "source" in value:
            specs.append(value)
        for nested in value.values():
            specs.extend(_iter_value_specs(nested))
    elif isinstance(value, list):
        for item in value:
            specs.extend(_iter_value_specs(item))
    return specs


def _derive_required_source_columns(
    *,
    subject_column: str,
    session_column: str | None,
    run_column: str,
    stim_column: str,
    row_sets: list[CompiledRowSet],
    validations: list[CompiledValidation],
) -> set[str]:
    required = {subject_column, run_column, stim_column}
    if session_column:
        required.add(session_column)
    for row_set in row_sets:
        for selector in row_set.selectors:
            if selector["operator"] == "equals":
                required.add(str(selector["column"]))
            elif selector["operator"] == "starts_with":
                required.add(str(selector["column"]))
            elif selector["operator"] == "present":
                required.add(str(selector["column"]))
        for transform in [*row_set.instruction_transforms, *row_set.trial_transforms]:
            for value_spec in _iter_value_specs(transform.args):
                if value_spec.get("source") == "source_field":
                    required.add(str(value_spec["column"]))
        for validation in validations:
            for value_spec in _iter_value_specs(validation.args):
                if value_spec.get("source") == "source_field":
                    required.add(str(value_spec["column"]))
    return required


def _compile_legacy_plan(
    *,
    phases: list[PhaseSpec],
    subject_column: str,
    session_column: str | None,
    run_column: str,
    stim_column: str,
    output_columns: list[str],
    missing_value: str,
    analysis_include_labels: set[str],
    probe_type_map: dict[str, str],
    recognition_acc_map: dict[str, dict[str, str]],
) -> CompiledPlan:
    row_sets: list[CompiledRowSet] = []
    for phase in phases:
        row_sets.append(
            CompiledRowSet(
                name=f"run_{phase.run}_{phase.condition}_{phase.phase}",
                run=phase.run,
                condition=phase.condition,
                phase=phase.phase,
                selectors=[
                    {"operator": "equals", "column": run_column, "value": str(phase.run)},
                    {"operator": "starts_with", "column": stim_column, "value": f"{phase.image_prefix}/"},
                    {"operator": "present", "column": phase.onset_column},
                ],
                instruction_transforms=_compile_instruction_transforms(phase),
                trial_transforms=_compile_trial_transforms(
                    phase=phase,
                    stim_column=stim_column,
                    analysis_include_labels=analysis_include_labels,
                    probe_type_map=probe_type_map,
                    recognition_acc_map=recognition_acc_map,
                ),
                context={
                    "image_prefix": phase.image_prefix,
                    "preserve_source_stim_file": "0",
                },
                onset_sort_column=phase.onset_column,
            )
        )

    lookups = [
        CompiledLookup(
            name="recognition_targets",
            source_row_sets=[row_set.name for row_set in row_sets if row_set.phase == "recognition"],
            key_columns=["condition", "stimulus_identity"],
            value_column="acc_label",
            when_column_equals={"phase": "recognition", "probe_type": "target"},
        )
    ]
    validations = [
        CompiledValidation(operator="required_columns", args={"columns": sorted([])}),
        CompiledValidation(
            operator="distinct_key_values",
            args={
                "partition_columns": ["condition", "stim_id"],
                "value_column": "stimulus_identity",
                "message_prefix": "Distinct stimuli would collapse to the same emitted stim_id within one run",
            },
        ),
    ]
    required_source_columns = _derive_required_source_columns(
        subject_column=subject_column,
        session_column=session_column,
        run_column=run_column,
        stim_column=stim_column,
        row_sets=row_sets,
        validations=validations,
    )
    validations = [
        CompiledValidation(operator="required_columns", args={"columns": sorted(required_source_columns)}),
        CompiledValidation(
            operator="distinct_key_values",
            args={
                "partition_columns": ["condition", "stim_id"],
                "value_column": "stimulus_identity",
                "message_prefix": "Distinct stimuli would collapse to the same emitted stim_id within one run",
            },
        ),
        *[
            CompiledValidation(
                operator="lookup_unique_keys",
                args={"lookup_name": lookup.name},
            )
            for lookup in lookups
        ],
    ]
    return CompiledPlan(
        row_sets=row_sets,
        lookups=lookups,
        validations=validations,
        output_columns=output_columns,
        missing_value=missing_value,
        required_source_columns=required_source_columns,
    )


def _resolve_source_roots(base_path: Path, payload: dict[str, Any]) -> list[Path]:
    stimuli = payload.get("stimuli", {})
    return [
        (base_path.parent / Path(root)).resolve() if not Path(root).is_absolute() else Path(root)
        for root in stimuli.get("source_roots", [])
    ]


def _build_spec_from_compiled_plan(
    *,
    base_path: Path,
    ops_path: Path,
    sidecar_path: Path,
    base: dict[str, Any],
    sidecar: dict[str, Any],
    compiled_plan: CompiledPlan,
    analysis_include_labels: set[str] | None = None,
    probe_type_map: dict[str, str] | None = None,
    recognition_acc_map: dict[str, dict[str, str]] | None = None,
    phases: list[PhaseSpec] | None = None,
) -> BuildSpec:
    source = base["source"]
    entities = base["entities"]
    output = base["output"]
    session_column = source.get("session_column")
    session_regex = source.get("session_regex")
    session_width = source.get("session_width")
    ped_fallback = entities.get("ped_fallback", {})

    return BuildSpec(
        name=str(base["name"]),
        source_path=base_path,
        ops_path=ops_path,
        sidecar_path=sidecar_path,
        spec_hash=_hash_spec_files([base_path, ops_path, sidecar_path]),
        source_encoding=str(source["encoding"]),
        subject_column=str(source["subject_column"]),
        subject_regex=str(source["subject_regex"]),
        subject_width=int(source["subject_width"]),
        session_column=str(session_column) if session_column else None,
        session_regex=str(session_regex) if session_regex else None,
        session_width=int(session_width) if session_width is not None else None,
        run_column=str(source["run_column"]),
        stim_column=str(source["stim_column"]),
        task=str(entities["task"]),
        acq_label=str(entities["acq"]) if entities.get("acq") else None,
        dir_label=str(entities["dir"]) if entities.get("dir") else None,
        datatype=str(entities["datatype"]),
        suffix=str(entities["suffix"]),
        ped_fallback_enabled=bool(ped_fallback.get("enabled", False)),
        ped_dir_map={str(key): str(value) for key, value in ped_fallback.get("map", {}).items()},
        sidecar_writes=bool(sidecar.get("writes_sidecar", False)),
        sidecar_columns={
            str(key): {str(inner_key): inner_value for inner_key, inner_value in value.items()}
            for key, value in sidecar.get("columns", {}).items()
        },
        stimuli_enabled=bool(base.get("stimuli", {}).get("enabled", False)),
        stimuli_source_roots=_resolve_source_roots(base_path, base),
        columns=[str(column) for column in output["columns"]],
        missing_value=str(output["missing_value"]),
        required_source_columns=set(compiled_plan.required_source_columns),
        analysis_include_labels=analysis_include_labels or set(),
        probe_type_map=probe_type_map or {},
        recognition_acc_map=recognition_acc_map or {},
        phases=phases or [],
        compiled_plan=compiled_plan,
    )


def _format_template_value(value: Any, tokens: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**tokens)
    if isinstance(value, list):
        return [_format_template_value(item, tokens) for item in value]
    if isinstance(value, dict):
        return {str(key): _format_template_value(item, tokens) for key, item in value.items()}
    return value


def _parse_compiled_transform(payload: dict[str, Any]) -> CompiledTransform:
    return CompiledTransform(
        target_column=str(payload["target_column"]),
        operator=str(payload["operator"]),
        args={str(key): value for key, value in payload.get("args", {}).items()},
    )


def _parse_compiled_row_set(payload: dict[str, Any]) -> CompiledRowSet:
    return CompiledRowSet(
        name=str(payload["name"]),
        run=int(payload["run"]),
        condition=str(payload["condition"]),
        phase=str(payload["phase"]),
        selectors=[{str(key): value for key, value in selector.items()} for selector in payload["selectors"]],
        instruction_transforms=[_parse_compiled_transform(item) for item in payload["instruction_transforms"]],
        trial_transforms=[_parse_compiled_transform(item) for item in payload["trial_transforms"]],
        context={str(key): str(value) for key, value in payload.get("context", {}).items()},
        onset_sort_column=str(payload["onset_sort_column"]),
    )


def _validate_template_conditions(
    template: dict[str, Any],
    *,
    declared_conditions: set[str],
) -> None:
    only_conditions = template.get("only_conditions")
    if only_conditions is None:
        return
    normalized = {str(item) for item in only_conditions}
    unknown = sorted(normalized - declared_conditions)
    if not unknown:
        return
    template_name = str(template.get("name", "<unnamed>"))
    declared = ", ".join(sorted(declared_conditions))
    unknown_text = ", ".join(unknown)
    raise ValueError(
        "Generic row-set template "
        f"{template_name!r} references unknown only_conditions: {unknown_text}. "
        f"Declared conditions: {declared}."
    )


def _template_applies_to_condition(template: dict[str, Any], condition_name: str) -> bool:
    only_conditions = template.get("only_conditions")
    if only_conditions is None:
        return True
    return condition_name in {str(item) for item in only_conditions}


def _parse_compiled_lookup(payload: dict[str, Any], row_sets: list[CompiledRowSet]) -> CompiledLookup:
    source_row_sets = payload.get("source_row_sets")
    if source_row_sets is None:
        source_phase = payload.get("source_phase")
        if source_phase is None:
            raise ValueError("Generic lookup config must provide source_row_sets or source_phase.")
        source_row_sets = [row_set.name for row_set in row_sets if row_set.phase == str(source_phase)]
    return CompiledLookup(
        name=str(payload["name"]),
        source_row_sets=[str(name) for name in source_row_sets],
        key_columns=[str(column) for column in payload["key_columns"]],
        value_column=str(payload["value_column"]),
        when_column_equals={str(key): str(value) for key, value in payload.get("when_column_equals", {}).items()},
    )


def _parse_compiled_validation(payload: dict[str, Any]) -> CompiledValidation:
    return CompiledValidation(
        operator=str(payload["operator"]),
        args={str(key): value for key, value in payload.get("args", {}).items()},
    )


def _normalize_required_columns_validations(
    *,
    validations: list[CompiledValidation],
    required_source_columns: set[str],
) -> list[CompiledValidation]:
    normalized = [
        validation for validation in validations if validation.operator != "required_columns"
    ]
    return [
        CompiledValidation(operator="required_columns", args={"columns": sorted(required_source_columns)}),
        *normalized,
    ]


def _load_legacy_build_spec(
    *,
    base_path: Path,
    ops_path: Path,
    sidecar_path: Path,
    base: dict[str, Any],
    ops: dict[str, Any],
    sidecar: dict[str, Any],
) -> BuildSpec:
    phase_templates = base["phase_templates"]
    conditions = base["conditions"]
    ops_conditions = ops["conditions"]

    phases: list[PhaseSpec] = []
    for run in base["runs"]:
        for phase_template in phase_templates:
            for condition_name, condition_payload in conditions.items():
                ops_payload = ops_conditions[condition_name][phase_template["phase"]]
                tokens = {
                    "run": run,
                    "condition": condition_name,
                    "condition_title": _title_case(condition_payload["title"]),
                }
                phases.append(
                    PhaseSpec(
                        run=int(run),
                        condition=condition_name,
                        phase=phase_template["phase"],
                        image_prefix=phase_template["image_prefix_template"].format(**tokens),
                        onset_column=phase_template["onset_column_template"].format(**tokens),
                        response_time_column=phase_template["response_time_column_template"].format(**tokens),
                        response_column=ops_payload["response_column"],
                        trial_type=phase_template["trial_type_template"].format(**tokens),
                        instruction_trial_type=phase_template["instruction_trial_type_template"].format(**tokens),
                        duration=str(phase_template["duration"]),
                        instruction_duration=str(phase_template["instruction_duration"]),
                        instruction_offset=str(phase_template["instruction_offset"]),
                        block_n=str(phase_template["block_n"]),
                        instruction_probe_type=str(phase_template["instruction_probe_type"]),
                        error_rule=ErrorRule(
                            kind=ops_payload["error_rule"]["kind"],
                            left=ops_payload["error_rule"]["left"],
                            right=ops_payload["error_rule"].get("right"),
                            value=ops_payload["error_rule"].get("value"),
                        ),
                    )
                )

    analysis_include_labels = {str(label) for label in ops["recognition"]["analysis_include_labels"]}
    probe_type_map = {str(key): str(value) for key, value in ops["recognition"]["probe_type_map"].items()}
    recognition_acc_map = {
        str(key): {str(inner_key): str(inner_value) for inner_key, inner_value in value.items()}
        for key, value in ops["recognition"]["acc_label_map"].items()
    }
    source = base["source"]
    output = base["output"]
    compiled_plan = _compile_legacy_plan(
        phases=phases,
        subject_column=str(source["subject_column"]),
        session_column=str(source.get("session_column")) if source.get("session_column") else None,
        run_column=str(source["run_column"]),
        stim_column=str(source["stim_column"]),
        output_columns=[str(column) for column in output["columns"]],
        missing_value=str(output["missing_value"]),
        analysis_include_labels=analysis_include_labels,
        probe_type_map=probe_type_map,
        recognition_acc_map=recognition_acc_map,
    )
    return _build_spec_from_compiled_plan(
        base_path=base_path,
        ops_path=ops_path,
        sidecar_path=sidecar_path,
        base=base,
        sidecar=sidecar,
        compiled_plan=compiled_plan,
        analysis_include_labels=analysis_include_labels,
        probe_type_map=probe_type_map,
        recognition_acc_map=recognition_acc_map,
        phases=phases,
    )


def _load_v2_build_spec(
    *,
    base_path: Path,
    ops_path: Path,
    sidecar_path: Path,
    base: dict[str, Any],
    ops: dict[str, Any],
    sidecar: dict[str, Any],
) -> BuildSpec:
    runs = [int(run) for run in base["runs"]]
    conditions = {str(key): value for key, value in ops["conditions"].items()}
    declared_conditions = set(conditions)
    for template in ops["row_set_templates"]:
        _validate_template_conditions(template, declared_conditions=declared_conditions)
    row_sets: list[CompiledRowSet] = []
    for run in runs:
        for condition_name, condition_payload in conditions.items():
            base_tokens = {
                "run": run,
                "condition": condition_name,
                **{str(key): value for key, value in condition_payload.items()},
            }
            formatted_condition_payload = _format_template_value(condition_payload, base_tokens)
            tokens = {
                **base_tokens,
                **{str(key): value for key, value in formatted_condition_payload.items()},
            }
            for template in ops["row_set_templates"]:
                if not _template_applies_to_condition(template, condition_name):
                    continue
                formatted = _format_template_value(template, tokens)
                row_sets.append(_parse_compiled_row_set(formatted))

    lookup_templates = ops.get("lookup_templates", [])
    lookups = [_parse_compiled_lookup(template, row_sets) for template in lookup_templates]
    validations = [_parse_compiled_validation(template) for template in ops.get("validations", [])]
    source = base["source"]
    output = base["output"]
    required_source_columns = _derive_required_source_columns(
        subject_column=str(source["subject_column"]),
        session_column=str(source.get("session_column")) if source.get("session_column") else None,
        run_column=str(source["run_column"]),
        stim_column=str(source["stim_column"]),
        row_sets=row_sets,
        validations=validations,
    )
    compiled_plan = CompiledPlan(
        row_sets=row_sets,
        lookups=lookups,
        validations=_normalize_required_columns_validations(
            validations=validations,
            required_source_columns=required_source_columns,
        ),
        output_columns=[str(column) for column in output["columns"]],
        missing_value=str(output["missing_value"]),
        required_source_columns=required_source_columns,
    )
    return _build_spec_from_compiled_plan(
        base_path=base_path,
        ops_path=ops_path,
        sidecar_path=sidecar_path,
        base=base,
        sidecar=sidecar,
        compiled_plan=compiled_plan,
    )


def load_build_spec(spec_path: str | Path) -> BuildSpec:
    base_path = Path(spec_path)
    ops_path = _sibling_config_path(base_path, ".ops.yaml")
    sidecar_path = _sibling_config_path(base_path, ".events.json.yaml")

    base = _read_json_yaml(base_path)
    ops = _read_json_yaml(ops_path)
    sidecar = _read_json_yaml(sidecar_path)

    version = int(base.get("version", 1))
    if version == 2:
        return _load_v2_build_spec(
            base_path=base_path,
            ops_path=ops_path,
            sidecar_path=sidecar_path,
            base=base,
            ops=ops,
            sidecar=sidecar,
        )
    return _load_legacy_build_spec(
        base_path=base_path,
        ops_path=ops_path,
        sidecar_path=sidecar_path,
        base=base,
        ops=ops,
        sidecar=sidecar,
    )


def resolve_subject(value: str, regex: str, width: int) -> str:
    match = re.search(regex, value)
    if not match:
        raise ValueError(f"Unable to resolve subject from {value!r}.")
    return match.group(1).zfill(width)


def resolve_session(value: str, regex: str, width: int | None) -> str:
    match = re.search(regex, value)
    if not match:
        raise ValueError(f"Unable to resolve session from {value!r}.")
    return normalize_session_value(match.group(1), width)


def normalize_session_value(value: str, width: int | None) -> str:
    token = str(value).strip().removeprefix("ses-")
    if not token:
        raise ValueError("Session value must not be empty.")
    if token.isdigit():
        if width is None:
            return token
        normalized = token.lstrip("0") or "0"
        return normalized.zfill(width)
    return token
