"""Execute planned localizer subject-level fixed-effects jobs.

The only mutating public API in this module is
``execute_localizer_fixed_effects_plan``. Validation and selection helpers are
read-only and consume the plan emitted by ``research_platform.neuro.localizer_ffx``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
import json
import subprocess

from research_platform.neuro.fsl.flame import write_fixed_effects_design
from research_platform.neuro.localizer_ffx import LocalizerFixedEffectsPlan


CommandRunner = Callable[[Sequence[str]], Any]

_COMMAND_NAMES = ("fslmerge", "fslmerge", "flameo")
_DIR_PATH_KINDS = frozenset({"output_dir", "work_dir"})
_PLANNED_FILE_PATH_KINDS = frozenset(
    {
        "design_file",
        "t_contrast_file",
        "merged_cope",
        "merged_varcope",
        "output_cope",
        "output_varcope",
        "output_mask",
    }
)
_EXPECTED_OUTPUT_KINDS = frozenset({"output_cope", "output_varcope", "output_mask"})
_REQUIRED_OUTPUT_KINDS = _DIR_PATH_KINDS | _PLANNED_FILE_PATH_KINDS
_PROVENANCE_BASENAME = "localizer_ffx_execution_provenance.json"


def validate_localizer_fixed_effects_execution_plan(
    plan: LocalizerFixedEffectsPlan | Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Validate that a localizer FFX plan is structurally executable."""

    payload = _plan_payload(plan)
    errors: list[str] = []
    warnings: list[str] = []
    malformed_job_rows: list[dict[str, Any]] = []
    malformed_output_rows: list[dict[str, Any]] = []
    fatal_error_rows: list[dict[str, Any]] = []

    if payload is None:
        errors.append("localizer fixed-effects execution plan must be a mapping or LocalizerFixedEffectsPlan.")
        return _validation_result(
            errors=errors,
            warnings=warnings,
            malformed_job_rows=malformed_job_rows,
            malformed_output_rows=malformed_output_rows,
            fatal_error_rows=fatal_error_rows,
            job_count=0,
            output_path_count=0,
            root_ref_rows=(),
        )

    plan_status = _optional_text(payload.get("status"))
    plan_errors = _text_sequence(payload.get("errors"))
    if plan_status == "error" or payload.get("valid") is False or plan_errors:
        message = "Plan contains fatal errors and cannot be executed."
        errors.append(message)
        fatal_error_rows.append(
            {
                "status": "error",
                "message": message,
                "plan_status": plan_status,
                "plan_errors": list(plan_errors),
            }
        )

    job_rows = _mapping_rows(payload.get("ffx_job_rows"))
    output_rows = _mapping_rows(payload.get("output_path_rows"))
    output_rows_by_job = _output_rows_by_job(output_rows, malformed_output_rows)
    if payload.get("ffx_job_rows") is None:
        errors.append("Plan is missing ffx_job_rows.")
    if payload.get("output_path_rows") is None:
        errors.append("Plan is missing output_path_rows.")

    for index, job in enumerate(job_rows):
        row_errors = _job_row_validation_errors(job, index=index)
        if row_errors:
            malformed_job_rows.append(
                {
                    "row_index": index,
                    "job_id": _optional_text(job.get("job_id")),
                    "status": "error",
                    "errors": row_errors,
                }
            )
            errors.extend(row_errors)
            continue
        job_output_errors = _job_output_validation_errors(job, output_rows_by_job.get(str(job["job_id"]), ()))
        if job_output_errors:
            malformed_output_rows.append(
                {
                    "job_id": str(job["job_id"]),
                    "status": "error",
                    "errors": job_output_errors,
                }
            )
            errors.extend(job_output_errors)

    output_job_ids = set(output_rows_by_job)
    known_job_ids = {str(row.get("job_id")) for row in job_rows if _optional_text(row.get("job_id")) is not None}
    for unknown_job_id in sorted(output_job_ids - known_job_ids):
        message = f"Output path rows reference unknown job_id {unknown_job_id!r}."
        malformed_output_rows.append({"job_id": unknown_job_id, "status": "error", "errors": [message]})
        errors.append(message)

    return _validation_result(
        errors=errors,
        warnings=warnings,
        malformed_job_rows=malformed_job_rows,
        malformed_output_rows=malformed_output_rows,
        fatal_error_rows=fatal_error_rows,
        job_count=len(job_rows),
        output_path_count=len(output_rows),
        root_ref_rows=tuple(_json_rows(payload.get("root_ref_rows"))),
    )


def select_executable_localizer_ffx_jobs(
    plan: LocalizerFixedEffectsPlan | Mapping[str, Any] | Any,
) -> list[dict[str, Any]]:
    """Return structurally executable subject-level FFX job rows from ``plan``."""

    payload = _plan_payload(plan)
    if payload is None:
        return []

    selected: list[dict[str, Any]] = []
    for index, job in enumerate(_mapping_rows(payload.get("ffx_job_rows"))):
        if _job_row_validation_errors(job, index=index):
            continue
        if _text_sequence(job.get("errors")):
            continue
        status = _optional_text(job.get("status"))
        if status in {"error", "failed", "skipped"}:
            continue
        selected.append(_normalize_job_row(job))
    return selected


def execute_localizer_fixed_effects_plan(
    plan: LocalizerFixedEffectsPlan | Mapping[str, Any] | Any,
    *,
    runner: CommandRunner | None = None,
    missing_input_policy: str = "fail",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute planned subject-level fixed-effects jobs.

    ``runner`` receives each planned command as an argv list. The default runner
    uses ``subprocess.run`` with ``shell=False`` and captures stdout/stderr.
    """

    policy = str(missing_input_policy).strip().casefold()
    validation = validate_localizer_fixed_effects_execution_plan(plan)
    payload = _plan_payload(plan)
    result = _empty_execution_result(
        validation=validation,
        missing_input_policy=policy,
        overwrite=bool(overwrite),
        root_ref_rows=tuple(_json_rows(payload.get("root_ref_rows"))) if payload is not None else (),
    )

    if policy not in {"fail", "skip"}:
        result["status"] = "error"
        result["errors"].append("missing_input_policy must be 'fail' or 'skip'.")
        return _json_safe(result)

    if not validation["valid"] or payload is None:
        result["status"] = "error"
        result["errors"].extend(validation["errors"])
        return _json_safe(result)

    output_rows_by_job = _output_rows_by_job(_mapping_rows(payload.get("output_path_rows")), [])
    command_runner = runner or _subprocess_runner
    selected_jobs = select_executable_localizer_ffx_jobs(payload)
    result["executed"] = True
    result["selected_job_count"] = len(selected_jobs)

    for job in selected_jobs:
        job_id = str(job["job_id"])
        planned_output_rows = output_rows_by_job.get(job_id, ())
        work_dir = Path(str(job["work_dir"])).resolve()
        output_dir = Path(str(job["output_dir"])).resolve()
        job_command_records: list[dict[str, Any]] = []
        job_output_checks: list[dict[str, Any]] = []
        attempt_row = {
            "job_id": job_id,
            "status": "attempted",
            "work_dir": str(work_dir),
            "output_dir": str(output_dir),
            "overwrite": bool(overwrite),
        }
        result["execution_attempt_rows"].append(attempt_row)

        missing_rows = _missing_input_rows(job)
        if missing_rows:
            if policy == "skip":
                skipped = {
                    "job_id": job_id,
                    "status": "skipped",
                    "reason": "missing_inputs",
                    "missing_inputs": missing_rows,
                }
                result["skipped_job_rows"].append(skipped)
                _record_provenance(result, job, attempt_row, job_command_records, job_output_checks, skipped)
            else:
                failed = {
                    "job_id": job_id,
                    "status": "failed",
                    "reason": "missing_inputs",
                    "missing_inputs": missing_rows,
                }
                result["failed_job_rows"].append(failed)
                _record_provenance(result, job, attempt_row, job_command_records, job_output_checks, failed)
            continue

        existing_rows = _existing_planned_file_rows(job, planned_output_rows)
        if existing_rows and not overwrite:
            refusal = {
                "job_id": job_id,
                "status": "refused",
                "reason": "planned_outputs_exist",
                "paths": existing_rows,
            }
            failed = {
                "job_id": job_id,
                "status": "failed",
                "reason": "overwrite_refused",
                "overwrite_refusal": refusal,
            }
            result["overwrite_refusal_rows"].append(refusal)
            result["failed_job_rows"].append(failed)
            _record_provenance(result, job, attempt_row, job_command_records, job_output_checks, failed)
            continue

        if overwrite:
            removal_error = _remove_existing_planned_files(job, planned_output_rows)
            if removal_error is not None:
                failed = {
                    "job_id": job_id,
                    "status": "failed",
                    "reason": "overwrite_failed",
                    "error": removal_error,
                }
                result["failed_job_rows"].append(failed)
                _record_provenance(result, job, attempt_row, job_command_records, job_output_checks, failed)
                continue

        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_fixed_effects_design(work_dir, n_inputs=len(job["cope_inputs"]))
        except OSError as exc:
            failed = {
                "job_id": job_id,
                "status": "failed",
                "reason": "design_write_failed",
                "error": _exception_summary(exc),
            }
            result["failed_job_rows"].append(failed)
            _record_provenance(result, job, attempt_row, job_command_records, job_output_checks, failed)
            continue

        failed_command: dict[str, Any] | None = None
        commands = _command_vectors(job.get("commands"))
        for command_index, command in enumerate(commands):
            record = _run_command(
                command,
                command_index=command_index,
                job_id=job_id,
                runner=command_runner,
            )
            job_command_records.append(record)
            result["command_records"].append(record)
            if record["status"] != "ok":
                failed_command = record
                break

        if failed_command is not None:
            failed = {
                "job_id": job_id,
                "status": "failed",
                "reason": "command_failed",
                "failed_command_index": failed_command["command_index"],
                "failed_command": failed_command,
                "skipped_command_count": len(commands) - int(failed_command["command_index"]) - 1,
            }
            result["failed_job_rows"].append(failed)
            _record_provenance(result, job, attempt_row, job_command_records, job_output_checks, failed)
            continue

        job_output_checks = _expected_output_checks(job, planned_output_rows)
        result["expected_output_check_rows"].extend(job_output_checks)
        missing_expected = [row for row in job_output_checks if row["status"] != "present"]
        if missing_expected:
            failed = {
                "job_id": job_id,
                "status": "failed",
                "reason": "missing_expected_outputs",
                "missing_expected_outputs": missing_expected,
            }
            result["failed_job_rows"].append(failed)
            _record_provenance(result, job, attempt_row, job_command_records, job_output_checks, failed)
            continue

        completed = {
            "job_id": job_id,
            "status": "completed",
            "expected_outputs": job_output_checks,
        }
        result["completed_job_rows"].append(completed)
        _record_provenance(result, job, attempt_row, job_command_records, job_output_checks, completed)

    result["status"] = _execution_status(result)
    return _json_safe(result)


def _validation_result(
    *,
    errors: Sequence[str],
    warnings: Sequence[str],
    malformed_job_rows: Sequence[Mapping[str, Any]],
    malformed_output_rows: Sequence[Mapping[str, Any]],
    fatal_error_rows: Sequence[Mapping[str, Any]],
    job_count: int,
    output_path_count: int,
    root_ref_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _json_safe(
        {
            "status": "error" if errors else "warning" if warnings else "ok",
            "valid": not errors,
            "errors": list(dict.fromkeys(str(error) for error in errors)),
            "warnings": list(dict.fromkeys(str(warning) for warning in warnings)),
            "malformed_job_rows": list(malformed_job_rows),
            "malformed_output_rows": list(malformed_output_rows),
            "fatal_error_rows": list(fatal_error_rows),
            "job_count": int(job_count),
            "output_path_count": int(output_path_count),
            "root_ref_rows": list(root_ref_rows),
        }
    )


def _empty_execution_result(
    *,
    validation: Mapping[str, Any],
    missing_input_policy: str,
    overwrite: bool,
    root_ref_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "not_run",
        "executed": False,
        "missing_input_policy": missing_input_policy,
        "overwrite": overwrite,
        "selected_job_count": 0,
        "validation": dict(validation),
        "errors": [],
        "warnings": [],
        "execution_attempt_rows": [],
        "command_records": [],
        "skipped_job_rows": [],
        "failed_job_rows": [],
        "completed_job_rows": [],
        "expected_output_check_rows": [],
        "overwrite_refusal_rows": [],
        "provenance_rows": [],
        "root_ref_rows": list(root_ref_rows),
    }


def _plan_payload(plan: LocalizerFixedEffectsPlan | Mapping[str, Any] | Any) -> Mapping[str, Any] | None:
    if isinstance(plan, LocalizerFixedEffectsPlan):
        return plan.to_dict()
    if isinstance(plan, Mapping):
        return plan
    to_dict = getattr(plan, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    return None


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


def _json_rows(raw: Any) -> tuple[dict[str, Any], ...]:
    return tuple(_json_safe(row) for row in _mapping_rows(raw))


def _normalize_job_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["cope_inputs"] = list(_text_sequence(row.get("cope_inputs")))
    normalized["varcope_inputs"] = list(_text_sequence(row.get("varcope_inputs")))
    normalized["commands"] = [list(command) for command in _command_vectors(row.get("commands"))]
    return _json_safe(normalized)


def _job_row_validation_errors(row: Mapping[str, Any], *, index: int) -> list[str]:
    errors: list[str] = []
    label = f"ffx_job_rows[{index}]"
    job_id = _optional_text(row.get("job_id"))
    if job_id is None:
        errors.append(f"{label}.job_id must be defined.")
    for key in ("mask_path", "output_dir", "work_dir"):
        if _optional_text(row.get(key)) is None:
            errors.append(f"{label}.{key} must be defined.")

    row_errors = _text_sequence(row.get("errors"))
    if row_errors:
        errors.append(f"{label} contains planned job errors: {'; '.join(row_errors)}")

    cope_inputs = _text_sequence(row.get("cope_inputs"))
    varcope_inputs = _text_sequence(row.get("varcope_inputs"))
    if not cope_inputs:
        errors.append(f"{label}.cope_inputs must contain at least one path.")
    if not varcope_inputs:
        errors.append(f"{label}.varcope_inputs must contain at least one path.")
    if cope_inputs and varcope_inputs and len(cope_inputs) != len(varcope_inputs):
        errors.append(f"{label}.cope_inputs and varcope_inputs must have matching lengths.")

    commands = _command_vectors(row.get("commands"))
    if len(commands) != 3:
        errors.append(f"{label}.commands must contain fslmerge, fslmerge, and flameo argv vectors.")
        return errors
    for command_index, (command, expected_name) in enumerate(zip(commands, _COMMAND_NAMES, strict=True)):
        if not command:
            errors.append(f"{label}.commands[{command_index}] must not be empty.")
            continue
        if command[0] != expected_name:
            errors.append(f"{label}.commands[{command_index}] must start with {expected_name!r}.")
        if any(not isinstance(part, str) or part == "" for part in command):
            errors.append(f"{label}.commands[{command_index}] must contain non-empty string argv elements.")

    if len(commands[0]) < 4 or commands[0][1] != "-t":
        errors.append(f"{label}.commands[0] must be an fslmerge -t argv vector.")
    if len(commands[1]) < 4 or commands[1][1] != "-t":
        errors.append(f"{label}.commands[1] must be an fslmerge -t argv vector.")
    if not errors:
        errors.extend(_planned_command_boundary_errors(row, commands, label=label))
    return errors


def _planned_command_boundary_errors(
    row: Mapping[str, Any],
    commands: Sequence[Sequence[str]],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    work_dir = Path(str(row["work_dir"])).resolve()
    output_dir = Path(str(row["output_dir"])).resolve()
    mask_path = Path(str(row["mask_path"])).resolve()
    merged_cope = Path(commands[0][2]).resolve()
    merged_varcope = Path(commands[1][2]).resolve()
    if not _is_relative_to(merged_cope, work_dir):
        errors.append(f"{label}.commands[0] writes outside work_dir.")
    if not _is_relative_to(merged_varcope, work_dir):
        errors.append(f"{label}.commands[1] writes outside work_dir.")

    flameo = commands[2]
    flags = _flameo_flags(flameo)
    if flags.get("--runmode") != "fe":
        errors.append(f"{label}.commands[2] must use --runmode=fe.")
    path_expectations = {
        "--cope": merged_cope,
        "--vc": merged_varcope,
        "--mask": mask_path,
        "--dm": work_dir / "design.mat",
        "--tc": work_dir / "design.con",
        "--ld": output_dir,
    }
    for flag, expected in path_expectations.items():
        value = flags.get(flag)
        if value is None:
            errors.append(f"{label}.commands[2] is missing {flag}=.")
            continue
        if Path(value).resolve() != expected.resolve():
            errors.append(f"{label}.commands[2] {flag}= path does not match the planned path.")
    return errors


def _job_output_validation_errors(
    job: Mapping[str, Any],
    output_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    job_id = str(job["job_id"])
    rows_by_kind = _output_rows_by_kind(output_rows)
    missing_kinds = sorted(_REQUIRED_OUTPUT_KINDS - set(rows_by_kind))
    if missing_kinds:
        errors.append(f"output_path_rows for job_id {job_id!r} are missing: {', '.join(missing_kinds)}.")
        return errors

    work_dir = Path(str(job["work_dir"])).resolve()
    output_dir = Path(str(job["output_dir"])).resolve()
    expected_paths = {
        "output_dir": output_dir,
        "work_dir": work_dir,
        "merged_cope": work_dir / "merged_cope.nii.gz",
        "merged_varcope": work_dir / "merged_varcope.nii.gz",
        "design_file": work_dir / "design.mat",
        "t_contrast_file": work_dir / "design.con",
        "output_cope": output_dir / "stats" / "cope1.nii.gz",
        "output_varcope": output_dir / "stats" / "varcope1.nii.gz",
        "output_mask": output_dir / "mask.nii.gz",
    }
    for kind, expected in expected_paths.items():
        actual = Path(str(rows_by_kind[kind]["path"])).resolve()
        if actual != expected.resolve():
            errors.append(f"output_path_rows for job_id {job_id!r} have unexpected {kind} path.")
    for kind in _PLANNED_FILE_PATH_KINDS:
        path = Path(str(rows_by_kind[kind]["path"])).resolve()
        if not (_is_relative_to(path, work_dir) or _is_relative_to(path, output_dir)):
            errors.append(f"output_path_rows for job_id {job_id!r} include {kind} outside planned directories.")
    return errors


def _output_rows_by_job(
    output_rows: Sequence[Mapping[str, Any]],
    malformed_rows: list[dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(output_rows):
        job_id = _optional_text(row.get("job_id"))
        path_kind = _optional_text(row.get("path_kind"))
        path = _optional_text(row.get("path"))
        row_errors: list[str] = []
        if job_id is None:
            row_errors.append(f"output_path_rows[{index}].job_id must be defined.")
        if path_kind is None:
            row_errors.append(f"output_path_rows[{index}].path_kind must be defined.")
        if path is None:
            row_errors.append(f"output_path_rows[{index}].path must be defined.")
        if row_errors:
            malformed_rows.append({"row_index": index, "status": "error", "errors": row_errors})
            continue
        grouped.setdefault(str(job_id), []).append(dict(row))
    return {job_id: tuple(rows) for job_id, rows in grouped.items()}


def _output_rows_by_kind(output_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in output_rows:
        kind = _optional_text(row.get("path_kind"))
        if kind is not None:
            rows[kind] = dict(row)
    return rows


def _missing_input_rows(job: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for kind, values in (
        ("cope", _text_sequence(job.get("cope_inputs"))),
        ("varcope", _text_sequence(job.get("varcope_inputs"))),
        ("mask", (_optional_text(job.get("mask_path")) or "",)),
    ):
        for value in values:
            path = Path(value)
            if not path.is_file():
                rows.append({"input_kind": kind, "path": str(path), "status": "missing"})
    return rows


def _existing_planned_file_rows(
    job: Mapping[str, Any],
    output_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    work_dir = Path(str(job["work_dir"])).resolve()
    output_dir = Path(str(job["output_dir"])).resolve()
    existing: list[dict[str, str]] = []
    for row in output_rows:
        kind = _optional_text(row.get("path_kind"))
        path_text = _optional_text(row.get("path"))
        if kind not in _PLANNED_FILE_PATH_KINDS or path_text is None:
            continue
        path = Path(path_text).resolve()
        if not (_is_relative_to(path, work_dir) or _is_relative_to(path, output_dir)):
            existing.append({"path_kind": str(kind), "path": str(path), "status": "outside_planned_directories"})
        elif path.exists():
            existing.append({"path_kind": str(kind), "path": str(path), "status": "exists"})
    return existing


def _remove_existing_planned_files(
    job: Mapping[str, Any],
    output_rows: Sequence[Mapping[str, Any]],
) -> str | None:
    work_dir = Path(str(job["work_dir"])).resolve()
    output_dir = Path(str(job["output_dir"])).resolve()
    for row in output_rows:
        kind = _optional_text(row.get("path_kind"))
        path_text = _optional_text(row.get("path"))
        if kind not in _PLANNED_FILE_PATH_KINDS or path_text is None:
            continue
        path = Path(path_text).resolve()
        if not (_is_relative_to(path, work_dir) or _is_relative_to(path, output_dir)):
            return f"Refusing to overwrite {path}; it is outside planned work/output directories."
        if path.exists() and not path.is_file() and not path.is_symlink():
            return f"Refusing to overwrite non-file planned path: {path}."
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                return _exception_summary(exc)
    return None


def _run_command(
    command: Sequence[str],
    *,
    command_index: int,
    job_id: str,
    runner: CommandRunner,
) -> dict[str, Any]:
    argv = [str(part) for part in command]
    try:
        outcome = runner(argv)
    except Exception as exc:  # pragma: no cover - exercised through tests, not branch-specific.
        return {
            "job_id": job_id,
            "command_index": command_index,
            "argv": argv,
            "status": "failed",
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "stdout_summary": "",
            "stderr_summary": "",
            "error": _exception_summary(exc),
        }

    returncode = _returncode(outcome)
    stdout = _summary_text(getattr(outcome, "stdout", ""))
    stderr = _summary_text(getattr(outcome, "stderr", ""))
    return {
        "job_id": job_id,
        "command_index": command_index,
        "argv": argv,
        "status": "ok" if returncode == 0 else "failed",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_summary": stdout,
        "stderr_summary": stderr,
        "error": "" if returncode == 0 else f"Command returned non-zero exit status {returncode}.",
    }


def _expected_output_checks(
    job: Mapping[str, Any],
    output_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    rows_by_kind = _output_rows_by_kind(output_rows)
    checks: list[dict[str, str]] = []
    for kind in sorted(_EXPECTED_OUTPUT_KINDS):
        path = Path(str(rows_by_kind[kind]["path"])).resolve()
        checks.append(
            {
                "job_id": str(job["job_id"]),
                "path_kind": kind,
                "path": str(path),
                "status": "present" if path.is_file() else "missing",
            }
        )
    return checks


def _record_provenance(
    result: dict[str, Any],
    job: Mapping[str, Any],
    attempt_row: Mapping[str, Any],
    command_records: Sequence[Mapping[str, Any]],
    output_checks: Sequence[Mapping[str, Any]],
    outcome: Mapping[str, Any],
) -> None:
    work_dir = Path(str(job["work_dir"])).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    path = _next_provenance_path(work_dir)
    payload = _json_safe(
        {
            "job": _normalize_job_row(job),
            "attempt": dict(attempt_row),
            "commands": list(command_records),
            "expected_output_checks": list(output_checks),
            "outcome": dict(outcome),
        }
    )
    path.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["provenance_rows"].append(
        {
            "job_id": str(job["job_id"]),
            "status": "written",
            "path": str(path),
            "work_dir": str(work_dir),
        }
    )


def _next_provenance_path(work_dir: Path) -> Path:
    base = work_dir / _PROVENANCE_BASENAME
    if not base.exists():
        return base
    stem = base.name.removesuffix(".json")
    for index in range(1, 10000):
        candidate = work_dir / f"{stem}-{index}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate a provenance JSON path.")


def _execution_status(result: Mapping[str, Any]) -> str:
    if result["errors"] or result["failed_job_rows"] or result["overwrite_refusal_rows"]:
        return "error"
    if result["skipped_job_rows"] or result["warnings"]:
        return "warning"
    if result["completed_job_rows"]:
        return "ok"
    return "not_run"


def _subprocess_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in argv],
        text=True,
        capture_output=True,
        check=False,
    )


def _command_vectors(raw: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    commands: list[tuple[str, ...]] = []
    for command in raw:
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes, bytearray)):
            return ()
        parts: list[str] = []
        for part in command:
            if not isinstance(part, str):
                return ()
            parts.append(part)
        commands.append(tuple(parts))
    return tuple(commands)


def _flameo_flags(command: Sequence[str]) -> dict[str, str]:
    flags: dict[str, str] = {}
    for arg in command[1:]:
        if not arg.startswith("--") or "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        flags[key] = value
    return flags


def _returncode(outcome: Any) -> int:
    if outcome is None:
        return 0
    if isinstance(outcome, int):
        return outcome
    value = getattr(outcome, "returncode", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _summary_text(value: Any, *, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _text_sequence(raw: Any) -> tuple[str, ...]:
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return value


__all__ = [
    "CommandRunner",
    "execute_localizer_fixed_effects_plan",
    "select_executable_localizer_ffx_jobs",
    "validate_localizer_fixed_effects_execution_plan",
]
