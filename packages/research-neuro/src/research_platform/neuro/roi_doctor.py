"""Preflight checks for reusable ROI build and extraction workflows.

Doctor helpers validate configuration and inspect resolvable paths only. They
do not execute FLAME1, create ROI masks, run featquery, or write outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from string import Formatter
from typing import Any
import shutil

from research_platform.neuro.roi import (
    ExtractionSet,
    RoiSet,
    parse_extraction_set_document,
    parse_roi_set_document,
    validate_extraction_set_document,
    validate_roi_set_document,
)
from research_platform.neuro.roi_execution import (
    RoiExecutionContext,
    plan_roi_extraction,
)


CommandFinder = Callable[[str], str | None]


_CHECK_ACTIONS = {
    "configuration_valid": "Fix the reported configuration issue, then rerun validate and doctor.",
    "roi_family_supported": "Choose a locally supported ROI family or backend before execution.",
    "configured_root_available": "Configure or create the required named root before execution.",
    "input_exists": "Populate or correct the required input path before execution.",
    "python_dependency_available": "Use an installation profile that supplies the required Python dependency.",
    "external_tool_available": "Install, load, or configure the required external tool before execution.",
    "image_readable": "Replace or correct the input with a readable NIfTI image.",
    "image_geometry_compatible": "Use inputs with compatible image geometry before execution.",
    "output_collision": (
        "Choose an unused output location, or deliberately set runtime.existing_output to replace."
    ),
}


def doctor_roi_set(
    document: Mapping[str, Any],
    *,
    context: RoiExecutionContext,
    command_finder: CommandFinder | None = None,
    runtime_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready preflight report for an ROI set."""

    runtime_document = runtime_document or document
    schema_errors = list(
        validate_roi_set_document(runtime_document, personal_path_document=document)
    )
    compatibility_errors: list[str] = []
    compatibility_warnings: list[str] = []
    checked_inputs: list[dict[str, Any]] = []
    missing_inputs: list[dict[str, Any]] = []
    project_name = context.project_name or "<unknown>"
    roi_set_name = _payload_name(document, "roi_set")
    expected_actions = 0
    output_root = _output_root_report(None, configured=False)
    fsl_required = False

    roi_set: RoiSet | None = None
    if not schema_errors:
        try:
            validation_roi_set = parse_roi_set_document(document)
            roi_set = parse_roi_set_document(runtime_document, validate_personal_paths=False)
            roi_set_name = roi_set.name
            expected_actions = _expected_roi_build_action_count(document, validation_roi_set)
            output_path, configured_output = _resolve_roi_output_root(roi_set, context=context)
            output_root = _output_root_report(output_path, configured=configured_output)
            fsl_required = _roi_set_uses_backend(roi_set, "fsl_flame1")
        except (ValueError, RuntimeError) as exc:
            compatibility_errors.append(str(exc))

    if roi_set is not None and any(roi.family == "loso_group_map" for roi in roi_set.rois):
        _check_loso_inputs(
            runtime_document,
            context=context,
            checked_inputs=checked_inputs,
            missing_inputs=missing_inputs,
            warnings=compatibility_warnings,
            errors=compatibility_errors,
            validate_personal_paths=False,
        )

    finder = command_finder or shutil.which
    fsl_tools = _fsl_tool_report(required=fsl_required, command_finder=finder)
    if fsl_tools["required"]:
        missing_tools = [name for name, status in fsl_tools["tools"].items() if not status["available"]]
        if missing_tools:
            compatibility_warnings.append("FSL tool(s) not found for fsl_flame1 execution: " + ", ".join(missing_tools))

    if missing_inputs:
        compatibility_warnings.append(f"{len(missing_inputs)} required input path(s) are missing.")

    checks = [_schema_check("ROI", schema_errors)]
    preflight_ready = False
    if not schema_errors:
        try:
            preflight_roi_build, _preflight_roi_extraction = _execution_preflight_functions()
            preflight = preflight_roi_build(
                runtime_document,
                context=context,
                validate_personal_paths=False,
                command_finder=finder,
            )
            preflight_payload = preflight.to_dict()
            preflight_ready = bool(preflight_payload.get("ready_for_execution", False))
            checks.extend(_preflight_check_rows(preflight_payload, skip_configuration=True))
        except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
            checks.append(
                _actionable_check(
                    {
                        "check_id": "configuration_valid",
                        "status": "error",
                        "message": str(exc),
                        "category": "preflight",
                    }
                )
            )

    checks = _dedupe_checks(checks)
    _extend_input_compatibility(checks, checked_inputs=checked_inputs, missing_inputs=missing_inputs)
    schema_valid = not schema_errors
    ready_for_execution = schema_valid and preflight_ready and not compatibility_errors
    check_errors = [str(check["message"]) for check in checks if check.get("status") == "error"]
    check_warnings = [str(check["message"]) for check in checks if check.get("status") == "warning"]
    errors = _unique_text([*schema_errors, *compatibility_errors, *check_errors])
    warnings = _unique_text([*compatibility_warnings, *check_warnings])

    return {
        "valid": schema_valid,
        "schema_valid": schema_valid,
        "ready_for_execution": ready_for_execution,
        "project": project_name,
        "roi_set": roi_set_name,
        "expected_actions": expected_actions,
        "checked_inputs": checked_inputs,
        "missing_inputs": missing_inputs,
        "output_root": output_root,
        "fsl_tools": fsl_tools,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "next_steps": _unique_text(
            [
                *_failed_check_actions(checks),
                *_roi_next_steps(
                    project_name,
                    roi_set_name,
                    errors=[*schema_errors, *compatibility_errors],
                    missing_inputs=missing_inputs,
                    fsl_tools=fsl_tools,
                ),
            ]
        ),
    }


def doctor_extraction_set(
    extraction_document: Mapping[str, Any],
    *,
    roi_set_document: Mapping[str, Any] | None = None,
    context: RoiExecutionContext,
    runtime_document: Mapping[str, Any] | None = None,
    runtime_roi_set_document: Mapping[str, Any] | None = None,
    command_finder: CommandFinder | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready preflight report for an ROI extraction set."""

    runtime_document = runtime_document or extraction_document
    runtime_roi_set_document = runtime_roi_set_document or roi_set_document
    schema_errors = list(
        validate_extraction_set_document(
            runtime_document,
            personal_path_document=extraction_document,
        )
    )
    compatibility_errors: list[str] = []
    compatibility_warnings: list[str] = []
    checked_inputs: list[dict[str, Any]] = []
    missing_inputs: list[dict[str, Any]] = []
    command_checks: list[dict[str, Any]] = []
    project_name = context.project_name or "<unknown>"
    extraction_set_name = _payload_name(extraction_document, "extraction_set")
    roi_set_name = None
    expected_actions = 0

    extraction_set: ExtractionSet | None = None
    roi_set: RoiSet | None = None
    if not schema_errors:
        try:
            parse_extraction_set_document(extraction_document)
            extraction_set = parse_extraction_set_document(runtime_document, validate_personal_paths=False)
            extraction_set_name = extraction_set.name
            roi_set_name = extraction_set.roi_set
            if roi_set_document is not None:
                roi_errors = validate_roi_set_document(
                    runtime_roi_set_document or roi_set_document,
                    personal_path_document=roi_set_document,
                )
                if roi_errors:
                    schema_errors.extend(f"referenced ROI set: {error}" for error in roi_errors)
                else:
                    roi_set = parse_roi_set_document(
                        runtime_roi_set_document or roi_set_document,
                        validate_personal_paths=False,
                    )
                    roi_set_name = roi_set.name
            elif extraction_set.roi_set is not None:
                compatibility_errors.append(f"Referenced ROI set {extraction_set.roi_set!r} was not loaded.")
        except ValueError as exc:
            compatibility_errors.append(str(exc))

    if extraction_set is not None and not schema_errors and not compatibility_errors:
        try:
            plan = plan_roi_extraction(
                runtime_document,
                roi_set_document=runtime_roi_set_document,
                context=context,
                validate_personal_paths=False,
            )
            expected_actions = len(plan.actions)
            target_fields = {target.name: target.fields for target in extraction_set.targets}
            common_fields = _extraction_set_common_fields(extraction_set)
            for action in plan.actions:
                _check_extraction_action_inputs(action, checked_inputs=checked_inputs, missing_inputs=missing_inputs)
                if action.backend == "fsl_featquery":
                    check = _featquery_command_check(
                        action,
                        target_fields=target_fields.get(action.target_name, {}),
                        common_fields=common_fields,
                    )
                    command_checks.append(check)
                    compatibility_errors.extend(check["errors"])
                    compatibility_warnings.extend(check["warnings"])
        except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
            compatibility_errors.append(str(exc))

    if missing_inputs:
        compatibility_warnings.append(f"{len(missing_inputs)} required extraction input path(s) are missing.")

    checks = [_schema_check("ROI extraction", schema_errors)]
    preflight_ready = False
    if not schema_errors:
        try:
            _preflight_roi_build, preflight_roi_extraction = _execution_preflight_functions()
            preflight = preflight_roi_extraction(
                runtime_document,
                roi_set_document=runtime_roi_set_document,
                context=context,
                validate_personal_paths=False,
                command_finder=command_finder or shutil.which,
            )
            preflight_payload = preflight.to_dict()
            preflight_ready = bool(preflight_payload.get("ready_for_execution", False))
            checks.extend(_preflight_check_rows(preflight_payload, skip_configuration=True))
        except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
            checks.append(
                _actionable_check(
                    {
                        "check_id": "configuration_valid",
                        "status": "error",
                        "message": str(exc),
                        "category": "preflight",
                    }
                )
            )

    checks = _dedupe_checks(checks)
    _extend_input_compatibility(checks, checked_inputs=checked_inputs, missing_inputs=missing_inputs)
    schema_valid = not schema_errors
    ready_for_execution = schema_valid and preflight_ready and not compatibility_errors
    check_errors = [str(check["message"]) for check in checks if check.get("status") == "error"]
    check_warnings = [str(check["message"]) for check in checks if check.get("status") == "warning"]
    errors = _unique_text([*schema_errors, *compatibility_errors, *check_errors])
    warnings = _unique_text([*compatibility_warnings, *check_warnings])

    return {
        "valid": schema_valid,
        "schema_valid": schema_valid,
        "ready_for_execution": ready_for_execution,
        "project": project_name,
        "extraction_set": extraction_set_name,
        "roi_set": roi_set_name or "<explicit masks>",
        "expected_actions": expected_actions,
        "checked_inputs": checked_inputs,
        "missing_inputs": missing_inputs,
        "command_checks": command_checks,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "next_steps": _unique_text(
            [
                *_failed_check_actions(checks),
                *_extraction_next_steps(
                    project_name,
                    extraction_set_name,
                    errors=[*schema_errors, *compatibility_errors],
                    missing_inputs=missing_inputs,
                    command_checks=command_checks,
                ),
            ]
        ),
    }


def _execution_preflight_functions() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Import execution preflights lazily to keep doctor imports lightweight."""

    from research_platform.neuro.roi_execution import (
        preflight_roi_build,
        preflight_roi_extraction,
    )

    return preflight_roi_build, preflight_roi_extraction


def _schema_check(label: str, errors: Sequence[str]) -> dict[str, Any]:
    if errors:
        message = f"{label} configuration has {len(errors)} structural issue(s)."
        status = "error"
    else:
        message = f"{label} configuration is structurally valid."
        status = "ok"
    return _actionable_check(
        {
            "check_id": "configuration_valid",
            "status": status,
            "message": message,
            "category": "configuration",
        }
    )


def _preflight_check_rows(
    payload: Mapping[str, Any],
    *,
    skip_configuration: bool,
) -> list[dict[str, Any]]:
    raw_checks = payload.get("checks", ())
    if not isinstance(raw_checks, Sequence) or isinstance(raw_checks, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for raw in raw_checks:
        if not isinstance(raw, Mapping):
            continue
        check_id = str(raw.get("check_id", raw.get("id", "preflight")))
        if skip_configuration and check_id == "configuration_valid":
            continue
        rows.append(_actionable_check({**dict(raw), "check_id": check_id}))
    return rows


def _actionable_check(payload: Mapping[str, Any]) -> dict[str, Any]:
    check_id = str(payload.get("check_id", payload.get("id", "preflight")))
    status = str(payload.get("status", "error"))
    row = {
        "check_id": check_id,
        "status": status,
        "message": str(payload.get("message", "Execution readiness check failed.")),
    }
    for key in ("path", "category"):
        value = payload.get(key)
        if value is not None:
            row[key] = str(value)
    if status != "ok":
        row["action"] = str(
            payload.get("action")
            or _CHECK_ACTIONS.get(
                check_id,
                "Correct the reported readiness issue before execution.",
            )
        )
    return row


def _dedupe_checks(checks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for check in checks:
        row = dict(check)
        key = tuple(
            str(row.get(field, ""))
            for field in ("check_id", "status", "message", "path", "category")
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _extend_input_compatibility(
    checks: Sequence[Mapping[str, Any]],
    *,
    checked_inputs: list[dict[str, Any]],
    missing_inputs: list[dict[str, Any]],
) -> None:
    existing = {
        (str(item.get("kind", "input")), str(item.get("path", "")))
        for item in checked_inputs
    }
    for check in checks:
        if check.get("check_id") != "input_exists" or check.get("path") is None:
            continue
        kind = str(check.get("category") or "input")
        path = str(check["path"])
        if (kind, path) in existing:
            continue
        row = {
            "kind": kind,
            "path": path,
            "exists": check.get("status") == "ok",
        }
        checked_inputs.append(row)
        existing.add((kind, path))
        if check.get("status") == "error":
            missing_inputs.append(row)


def _failed_check_actions(checks: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(check["action"])
        for check in checks
        if check.get("status") != "ok" and check.get("action")
    ]


def _unique_text(values: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _check_loso_inputs(
    document: Mapping[str, Any],
    *,
    context: RoiExecutionContext,
    checked_inputs: list[dict[str, Any]],
    missing_inputs: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    validate_personal_paths: bool = True,
) -> None:
    from research_platform.neuro.roi_loso import discover_loso_group_mask_inputs, discover_subject_fixed_effects_inputs

    try:
        for item in discover_subject_fixed_effects_inputs(
            document,
            context=context,
            validate_personal_paths=validate_personal_paths,
        ):
            for kind, path in (
                ("fixed_effects_cope", item.cope_path),
                ("fixed_effects_varcope", item.varcope_path),
                ("fixed_effects_mask", item.mask_path),
            ):
                _record_input_check(
                    kind,
                    path,
                    checked_inputs=checked_inputs,
                    missing_inputs=missing_inputs,
                    subject_id=item.subject_id,
                    session_id=item.session_id,
                    task_id=item.task_id,
                    model=item.model,
                    contrast_id=item.contrast_id,
                    cope_number=item.cope_number,
                )
    except (ValueError, RuntimeError) as exc:
        errors.append(str(exc))

    try:
        for item in discover_loso_group_mask_inputs(
            document,
            context=context,
            validate_personal_paths=validate_personal_paths,
        ):
            if item.generated:
                checked_inputs.append(
                    {
                        "kind": "generated_group_mask",
                        "path": str(item.mask_path),
                        "exists": item.mask_path.exists(),
                        "status": "existing" if item.mask_path.exists() else "planned",
                        "session_id": item.session_id,
                        "task_id": item.task_id,
                        "model": item.model,
                        "contrast_id": item.contrast_id,
                        "cope_number": item.cope_number,
                        "strategy": item.strategy,
                        "scope": item.scope,
                        "heldout_subject": item.heldout_subject,
                        "included_subjects": list(item.included_subjects),
                        "excluded_subjects": list(item.excluded_subjects),
                        "source_mask_count": len(item.source_mask_paths),
                        "source_mask_paths": [str(path) for path in item.source_mask_paths],
                        "sidecar_path": str(item.sidecar_path) if item.sidecar_path is not None else None,
                    }
                )
            else:
                _record_input_check(
                    "group_mask",
                    item.mask_path,
                    checked_inputs=checked_inputs,
                    missing_inputs=missing_inputs,
                    session_id=item.session_id,
                    task_id=item.task_id,
                    model=item.model,
                    contrast_id=item.contrast_id,
                    cope_number=item.cope_number,
                )
    except (ValueError, RuntimeError) as exc:
        errors.append(str(exc))

    if not any(check["kind"] in {"group_mask", "generated_group_mask"} for check in checked_inputs):
        warnings.append("No group_mask paths were resolved for LOSO ROI checks.")


def _check_extraction_action_inputs(
    action: Any,
    *,
    checked_inputs: list[dict[str, Any]],
    missing_inputs: list[dict[str, Any]],
) -> None:
    metadata = action.metadata if isinstance(action.metadata, Mapping) else {}
    if action.backend == "fsl_featquery":
        feat_dir = _optional_path(metadata.get("feat_dir"))
        if feat_dir is not None:
            _record_input_check(
                "feat_dir",
                feat_dir,
                checked_inputs=checked_inputs,
                missing_inputs=missing_inputs,
                target_name=action.target_name,
                roi_label=action.roi_label,
                source_contrast=metadata.get("source_contrast"),
                cope=metadata.get("cope"),
            )
        _record_input_check(
            "value_image",
            action.value_map_path,
            checked_inputs=checked_inputs,
            missing_inputs=missing_inputs,
            target_name=action.target_name,
            roi_label=action.roi_label,
            source_contrast=metadata.get("source_contrast"),
            cope=metadata.get("cope"),
        )
        _record_input_check(
            "roi_mask",
            action.mask_path,
            checked_inputs=checked_inputs,
            missing_inputs=missing_inputs,
            target_name=action.target_name,
            roi_label=action.roi_label,
            source_contrast=metadata.get("source_contrast"),
            cope=metadata.get("cope"),
        )
        return

    _record_input_check(
        "value_image",
        action.value_map_path,
        checked_inputs=checked_inputs,
        missing_inputs=missing_inputs,
        target_name=action.target_name,
        roi_label=action.roi_label,
    )
    _record_input_check(
        "roi_mask",
        action.mask_path,
        checked_inputs=checked_inputs,
        missing_inputs=missing_inputs,
        target_name=action.target_name,
        roi_label=action.roi_label,
    )


def _featquery_command_check(
    action: Any,
    *,
    target_fields: Mapping[str, Any],
    common_fields: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = action.metadata if isinstance(action.metadata, Mapping) else {}
    command = tuple(str(part) for part in metadata.get("command", ()))
    roi_mask = str(action.mask_path)
    roi_positions = [index for index, part in enumerate(command) if part == roi_mask]
    extra_one = any(index > 0 and command[index - 1] == "1" for index in roi_positions)
    includes_percent_flag = "-p" in command
    explicit_percent_flag = _include_percent_signal_change_configured(common_fields, target_fields)
    metrics = tuple(str(metric) for metric in action.metrics)
    requests_percent_signal_change = explicit_percent_flag or "percent_signal_change" in metrics
    warnings: list[str] = []
    errors: list[str] = []

    if extra_one:
        errors.append(f"featquery command for {action.target_name}/{action.roi_label} has an extra '1' before the ROI mask.")
    if includes_percent_flag and not requests_percent_signal_change:
        errors.append(
            f"featquery command for {action.target_name}/{action.roi_label} includes -p without a percent_signal_change request."
        )
    if "mean_cope" in metrics and requests_percent_signal_change:
        errors.append(
            "FSL featquery raw COPE and percent signal change require separate extraction targets because PSC uses the -p conversion flag."
        )
    if "percent_signal_change" in metrics and not includes_percent_flag:
        errors.append(f"{action.target_name} requests percent_signal_change, but the featquery command does not include -p.")

    return {
        "target_name": action.target_name,
        "roi_label": action.roi_label,
        "command": list(command),
        "roi_mask_path": roi_mask,
        "has_extra_one_before_roi_mask": extra_one,
        "includes_percent_signal_change_flag": includes_percent_flag,
        "percent_signal_change_explicitly_configured": explicit_percent_flag,
        "percent_signal_change_requested": requests_percent_signal_change,
        "metrics": list(metrics),
        "warnings": warnings,
        "errors": errors,
    }


def _record_input_check(
    kind: str,
    path: str | Path,
    *,
    checked_inputs: list[dict[str, Any]],
    missing_inputs: list[dict[str, Any]],
    **metadata: Any,
) -> None:
    resolved = Path(path)
    exists = resolved.exists()
    payload = {
        "kind": kind,
        "path": str(resolved),
        "exists": exists,
        **{str(key): value for key, value in metadata.items() if value is not None},
    }
    checked_inputs.append(payload)
    if not exists:
        missing_inputs.append(payload)


def _expected_roi_build_action_count(document: Mapping[str, Any], roi_set: RoiSet) -> int:
    count = sum(1 for roi in roi_set.rois if roi.family != "loso_group_map")
    if any(roi.family == "loso_group_map" for roi in roi_set.rois):
        from research_platform.neuro.roi_loso import expected_loso_group_map_build_action_count

        count += expected_loso_group_map_build_action_count(document)
    return count


def _resolve_roi_output_root(roi_set: RoiSet, *, context: RoiExecutionContext) -> tuple[Path, bool]:
    common = _roi_set_common_fields(roi_set)
    return _resolve_output_root(common, context=context)


def _resolve_output_root(fields: Mapping[str, Any], *, context: RoiExecutionContext) -> tuple[Path, bool]:
    outputs = fields.get("outputs") if isinstance(fields.get("outputs"), Mapping) else {}
    if isinstance(outputs, Mapping):
        root_value = _optional_text(outputs.get("root") or outputs.get("derivative_root") or outputs.get("output_root"))
        if root_value is not None:
            return _resolve_path_value(root_value, context=context, values={}, label="outputs.root"), True
        root_ref = _optional_text(outputs.get("root_ref"))
        if root_ref is not None:
            root = context.resolve_root_ref(root_ref)
            subpath = _optional_text(outputs.get("path") or outputs.get("subpath"))
            return ((root / subpath).resolve() if subpath else root.resolve()), True
        path_value = _optional_text(outputs.get("path"))
        if path_value is not None:
            return _resolve_path_value(path_value, context=context, values={}, label="outputs.path"), True

    for key in ("output_root", "derivative_root"):
        value = _optional_text(fields.get(key))
        if value is not None:
            return _resolve_path_value(value, context=context, values={}, label=key), True

    project = context.project_name or "project"
    return context.artifacts_root / "roi" / project / "derivatives", False


def _resolve_path_value(value: str, *, context: RoiExecutionContext, values: Mapping[str, Any], label: str) -> Path:
    rendered = _render_template(value, values, label=label)
    path = Path(rendered).expanduser()
    return path.resolve() if path.is_absolute() else (context.project_root / path).resolve()


def _output_root_report(path: Path | None, *, configured: bool) -> dict[str, Any]:
    if path is None:
        return {"path": None, "configured": configured, "exists": False, "is_dir": False, "parent_exists": False}
    return {
        "path": str(path),
        "configured": configured,
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "parent_exists": path.parent.exists(),
    }


def _fsl_tool_report(*, required: bool, command_finder: CommandFinder) -> dict[str, Any]:
    tools: dict[str, dict[str, Any]] = {}
    if required:
        for name in ("fslmerge", "flameo"):
            path = command_finder(name)
            tools[name] = {"available": path is not None, "path": path}
    return {"required": required, "tools": tools}


def _roi_set_uses_backend(roi_set: RoiSet, backend: str) -> bool:
    if roi_set.backend == backend:
        return True
    return any((roi.backend or roi_set.backend) == backend for roi in roi_set.rois)


def _include_percent_signal_change_configured(*payloads: Mapping[str, Any]) -> bool:
    for payload in payloads:
        if _truthy(payload.get("include_percent_signal_change")):
            return True
        for key in ("featquery", "fsl", "backend_settings", "fsl_featquery"):
            block = payload.get(key)
            if isinstance(block, Mapping) and _truthy(block.get("include_percent_signal_change")):
                return True
    return False


def _roi_set_common_fields(roi_set: RoiSet) -> dict[str, Any]:
    return {
        **dict(roi_set.fields),
        "name": roi_set.name,
        "roi_set": roi_set.name,
        "desc": roi_set.desc,
        "backend": roi_set.backend,
        **dict(roi_set.provenance if isinstance(roi_set.provenance, Mapping) else {}),
    }


def _extraction_set_common_fields(extraction_set: ExtractionSet) -> dict[str, Any]:
    return {
        **dict(extraction_set.fields),
        "name": extraction_set.name,
        "extraction_set": extraction_set.name,
        "roi_set": extraction_set.roi_set,
        "backend": extraction_set.backend,
        **dict(extraction_set.provenance if isinstance(extraction_set.provenance, Mapping) else {}),
    }


def _payload_name(document: Mapping[str, Any], key: str) -> str:
    payload = document.get(key, document)
    if isinstance(payload, Mapping):
        name = _optional_text(payload.get("name"))
        if name is not None:
            return name
    return "<unknown>"


def _optional_path(value: Any) -> Path | None:
    text = _optional_text(value)
    return Path(text) if text is not None else None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _optional_text(value)
    return text is not None and text.lower() in {"1", "true", "yes", "on"}


def _render_template(template: str, values: Mapping[str, Any], *, label: str) -> str:
    names = {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}
    missing = sorted(name for name in names if name not in values)
    if missing:
        raise ValueError(f"{label} references missing template field(s): {', '.join(missing)}.")
    return template.format(**values)


def _roi_next_steps(
    project_name: str,
    roi_set_name: str,
    *,
    errors: Sequence[str],
    missing_inputs: Sequence[Mapping[str, Any]],
    fsl_tools: Mapping[str, Any],
) -> list[str]:
    if errors:
        return ["Fix the reported ROI config or path-resolution errors, then rerun the doctor command."]
    steps: list[str] = []
    if missing_inputs:
        steps.append("Populate or correct missing input paths before running ROI build with --execute.")
    missing_tools = [name for name, status in dict(fsl_tools.get("tools", {})).items() if not status.get("available")]
    if missing_tools:
        steps.append("Load or install FSL before executing fsl_flame1 ROI builds.")
    steps.extend(
        [
            f"Review the build plan: rp analysis roi build {roi_set_name} --project {project_name}",
            f"Execute only after reviewing the plan: rp analysis roi build {roi_set_name} --project {project_name} --execute",
        ]
    )
    return steps


def _extraction_next_steps(
    project_name: str,
    extraction_set_name: str,
    *,
    errors: Sequence[str],
    missing_inputs: Sequence[Mapping[str, Any]],
    command_checks: Sequence[Mapping[str, Any]],
) -> list[str]:
    if errors:
        return ["Fix the reported extraction config, command, or path-resolution errors, then rerun the doctor command."]
    steps: list[str] = []
    if missing_inputs:
        steps.append("Populate or correct missing FEAT, value image, or ROI mask paths before running extraction with --execute.")
    if any("-p" in check.get("command", ()) for check in command_checks):
        steps.append("Confirm the local FSL featquery version supports -p before executing percent-signal-change extraction.")
    steps.extend(
        [
            f"Review the extraction plan: rp analysis roi extraction run {extraction_set_name} --project {project_name}",
            f"Execute only after reviewing the plan: rp analysis roi extraction run {extraction_set_name} --project {project_name} --execute",
        ]
    )
    return steps
