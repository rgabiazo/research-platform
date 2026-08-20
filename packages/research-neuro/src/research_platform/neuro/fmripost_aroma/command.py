"""Container plan assembly helpers for fMRIPost-AROMA."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Mapping, Sequence

from .selection import (
    build_flat_bids_filter,
    discover_derivative_runs,
    group_runtime_plan_runs,
    normalize_entity_label,
)

DEFAULT_IMAGE_REPOSITORY = "nipreps/fmripost-aroma"
DEFAULT_IMAGE_TAG = "0.0.12"
DEFAULT_LOCAL_BACKEND = "docker"
DEFAULT_HPC_BACKEND = "apptainer"
DEFAULT_APPTAINER_PULL_MODE = "if_missing"
DEFAULT_APPTAINER_IMAGE_ROOT = "${RP_REMOTE_CONTAINER_ROOT:-$SCRATCH/containers/fmripost_aroma}"
DEFAULT_APPTAINER_CACHE_DIR = "${SCRATCH}/apptainer-cache"
DEFAULT_APPTAINER_TMPDIR = '${SLURM_TMPDIR:-$SCRATCH/apptainer-tmp}'
MIN_GB_PER_FMRIPOST_AROMA_PROCESS = 8
SUCCESS_MARKERS = ("fMRIPost-AROMA finished successfully!",)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
    "KMP_AFFINITY": "disabled",
    "KMP_BLOCKTIME": "0",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "MALLOC_ARENA_MAX": "1",
}
_SUPPORTED_RUNTIME_GROUPINGS = frozenset({"compatible", "row"})


def build_thread_environment(threads: int) -> dict[str, str]:
    thread_value = str(max(1, int(threads)))
    return {
        "OMP_NUM_THREADS": thread_value,
        "MKL_NUM_THREADS": thread_value,
        "OPENBLAS_NUM_THREADS": thread_value,
        "NUMEXPR_NUM_THREADS": thread_value,
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": thread_value,
        "KMP_AFFINITY": THREAD_ENVIRONMENT["KMP_AFFINITY"],
        "KMP_BLOCKTIME": THREAD_ENVIRONMENT["KMP_BLOCKTIME"],
        "OMP_DYNAMIC": THREAD_ENVIRONMENT["OMP_DYNAMIC"],
        "MKL_DYNAMIC": THREAD_ENVIRONMENT["MKL_DYNAMIC"],
        "MALLOC_ARENA_MAX": THREAD_ENVIRONMENT["MALLOC_ARENA_MAX"],
    }


def resolve_fmripost_aroma_runtime_resources(resources: Mapping[str, Any] | None) -> dict[str, int]:
    if not resources:
        return {"nprocs": 1, "omp_nthreads": 1}

    cpus = max(1, int(resources.get("cpus", 1)))
    threads = max(1, int(resources.get("threads", 1)))
    if threads > cpus:
        raise ValueError("fMRIPost-AROMA resources require threads <= cpus.")
    nprocs = cpus
    resolved = {
        "nprocs": nprocs,
        "omp_nthreads": threads,
    }
    if resources.get("ram_gb") not in (None, ""):
        ram_gb = max(1, int(float(resources["ram_gb"])))
        resolved["mem_mb"] = ram_gb * 1024
        resolved["nprocs"] = max(1, min(nprocs, ram_gb // MIN_GB_PER_FMRIPOST_AROMA_PROCESS))
    return resolved


def build_batch_runtime_plan(
    *,
    raw_bids_root: str | Path,
    derivative_root: str | Path,
    derivative_name: str,
    batch_rows: Sequence[Mapping[str, str]],
    output_root: str | Path,
    work_root: str | Path,
    plan_path: str | Path,
    command_script_path: str | Path,
    selection: Mapping[str, str | None] | None = None,
    backend: str | None = None,
    image_repository: str = DEFAULT_IMAGE_REPOSITORY,
    image_tag: str = DEFAULT_IMAGE_TAG,
    tool_options: Mapping[str, Any] | None = None,
    templateflow_home: str | Path | None = None,
    resources: Mapping[str, Any] | None = None,
    container_pull_mode: str | None = None,
    container_image_root: str | None = None,
    container_image_name: str | None = None,
) -> dict[str, Any]:
    raw_root_path = Path(raw_bids_root).resolve()
    derivative_root_path = Path(derivative_root).resolve()
    output_root_path = Path(output_root).resolve()
    work_root_path = Path(work_root).resolve()
    plan_path_obj = Path(plan_path).resolve()
    command_script_path_obj = Path(command_script_path).resolve()
    templateflow_root = _resolve_templateflow_home(plan_path_obj, templateflow_home)
    filter_root = output_root_path / "bids-filters"
    output_data_root = output_root_path / "fmripost_aroma"
    unit_marker_root = output_root_path / "runtime-plan-markers" / "fmripost_aroma"

    filter_root.mkdir(parents=True, exist_ok=True)
    output_data_root.mkdir(parents=True, exist_ok=True)
    work_root_path.mkdir(parents=True, exist_ok=True)
    templateflow_root.mkdir(parents=True, exist_ok=True)
    unit_marker_root.mkdir(parents=True, exist_ok=True)

    selected_steps: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    resolved_backend = backend or DEFAULT_LOCAL_BACKEND
    image_reference = f"{image_repository}:{image_tag}"
    resolved_container_pull_mode = _normalize_apptainer_pull_mode(container_pull_mode)
    resolved_container_image_root = _optional_value(container_image_root)
    resolved_container_image_name = _normalize_image_name(_optional_value(container_image_name))
    runtime_resources = resolve_fmripost_aroma_runtime_resources(resources)
    resolved_selection = _normalize_selectors(selection or {})
    resolved_tool_options = _normalize_tool_options(tool_options)
    runtime_grouping = str(resolved_tool_options["runtime_grouping"])
    container_prep = _build_container_prep_step(
        backend=resolved_backend,
        image_reference=image_reference,
        pull_mode=resolved_container_pull_mode,
        image_root=resolved_container_image_root,
        image_name=resolved_container_image_name,
        marker_root=unit_marker_root,
    )
    execution_image_reference = image_reference
    execution_container_pull_mode = resolved_container_pull_mode
    execution_container_image_root = resolved_container_image_root
    execution_container_image_name = resolved_container_image_name
    if container_prep is not None:
        execution_image_reference = str(container_prep["runtime_image"])
        execution_container_pull_mode = "never"
    grouped_matches: dict[tuple[str, ...], dict[str, Any]] = {}

    for row in batch_rows:
        selectors = _resolve_row_selectors(row, default_selection=resolved_selection)
        subject_id = selectors["subject_id"]
        session_id = selectors["session_id"]
        task_id = selectors["task_id"]
        run_id = selectors["run_id"]
        if not subject_id:
            raise ValueError("Batch rows must define subject_id.")
        derivative_runs = discover_derivative_runs(
            derivative_root_path,
            subject_id=subject_id,
            session_id=session_id,
            task_id=task_id,
            run_id=run_id,
        )
        if not derivative_runs:
            skipped_rows.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id or "",
                    "task_id": task_id or "",
                    "run_id": run_id or "",
                    "reason": "No matching derivative-backed BOLD runs.",
                }
            )
            continue

        for grouped_runs in group_runtime_plan_runs(
            derivative_runs,
            runtime_grouping=runtime_grouping,
        ):
            representative = grouped_runs[0]
            group_key = _runtime_step_group_key(
                runtime_grouping=runtime_grouping,
                subject_id=subject_id,
                run=representative,
            )
            group = grouped_matches.setdefault(
                group_key,
                {
                    "subject_id": subject_id,
                    "session_id": _prefixed_entity_label("ses", representative.session_id),
                    "explicit_task_ids": set(),
                    "runs": {},
                },
            )
            group["explicit_task_ids"].add(normalize_entity_label(task_id, "task"))
            for run in grouped_runs:
                group["runs"][_matched_run_key(run)] = run

    for group_key in sorted(grouped_matches):
        group = grouped_matches[group_key]
        grouped_runs = sorted(group["runs"].values(), key=_matched_run_sort_key)
        representative = grouped_runs[0]
        group_task_id = representative.task_id
        group_run_id = representative.run_id if len(grouped_runs) == 1 else None
        group_acquisition = representative.acquisition
        group_direction = representative.direction
        subject_id = str(group["subject_id"])
        session_id = _optional_value(group.get("session_id"))

        subject_label = normalize_entity_label(subject_id, "sub") or ""
        filter_payload = build_flat_bids_filter(
            grouped_runs,
            session_id=session_id,
            task_id=group_task_id,
            run_id=group_run_id,
        )
        filter_path = filter_root / _filter_filename(
            subject_id=subject_id,
            session_id=session_id,
            task_id=group_task_id,
            acquisition=group_acquisition,
            direction=group_direction,
            run_id=group_run_id,
        )
        filter_path.write_text(json.dumps(filter_payload, indent=2) + "\n", encoding="utf-8")

        step_work_root = work_root_path / _run_label(
            subject_id=subject_id,
            session_id=session_id,
            task_id=group_task_id,
            acquisition=group_acquisition,
            direction=group_direction,
            run_id=group_run_id,
        )
        step_work_root.mkdir(parents=True, exist_ok=True)
        env = build_thread_environment(runtime_resources["omp_nthreads"]) | {"TEMPLATEFLOW_HOME": "/templateflow"}
        command = _build_container_command(
            backend=resolved_backend,
            image_reference=execution_image_reference,
            env=env,
            raw_bids_root=raw_root_path,
            derivative_root=derivative_root_path,
            derivative_name=derivative_name,
            output_root=output_data_root,
            work_root=step_work_root,
            filter_path=filter_path,
            templateflow_home=templateflow_root,
            subject_label=subject_label,
            task_id=_command_task_id(group["explicit_task_ids"]),
            tool_options=resolved_tool_options,
            nprocs=runtime_resources["nprocs"],
            omp_nthreads=runtime_resources["omp_nthreads"],
            mem_mb=runtime_resources.get("mem_mb"),
            container_pull_mode=execution_container_pull_mode,
            container_image_root=execution_container_image_root,
            container_image_name=execution_container_image_name,
        )
        selected_steps.append(
            {
                "subject_id": subject_id,
                "session_id": session_id,
                "task_id": group_task_id,
                "run_id": group_run_id,
                "acquisition": group_acquisition,
                "direction": group_direction,
                "matched_runs": [run.to_dict() for run in grouped_runs],
                "bids_filter": filter_payload,
                "bids_filter_file": str(filter_path),
                "output_dir": str(output_data_root),
                "work_dir": str(step_work_root),
                "env": env,
                "command": command,
                "success_markers": list(SUCCESS_MARKERS),
            }
        )

    if not selected_steps:
        container_prep = None
    units = _build_runtime_units(selected_steps, marker_root=unit_marker_root)

    return {
        "tool": "fmripost_aroma",
        "backend": resolved_backend,
        "image": {
            "repository": image_repository,
            "tag": image_tag,
            "reference": image_reference,
            "runtime_image": str(container_prep["runtime_image"]) if container_prep is not None else image_reference,
            "pull_mode": resolved_container_pull_mode,
            "image_root": resolved_container_image_root or DEFAULT_APPTAINER_IMAGE_ROOT,
            "image_name": resolved_container_image_name
            or _derive_container_image_name(_normalize_apptainer_image_reference(image_reference)),
        },
        "container_prep": container_prep,
        "templateflow_home": str(templateflow_root),
        "raw_bids_root": str(raw_root_path),
        "derivative_root": str(derivative_root_path),
        "output_root": str(output_root_path),
        "command_script": str(command_script_path_obj),
        "plan_path": str(plan_path_obj),
        "unit_marker_root": str(unit_marker_root),
        "resources": runtime_resources,
        "tool_options": resolved_tool_options,
        "steps": selected_steps,
        "units": units,
        "skipped_rows": skipped_rows,
    }


def write_runtime_plan(plan: Mapping[str, Any], plan_path: str | Path) -> Path:
    path = Path(plan_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return path


def write_command_script(plan: Mapping[str, Any], command_script_path: str | Path) -> Path:
    path = Path(command_script_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    container_prep = plan.get("container_prep")
    if isinstance(container_prep, Mapping):
        prep_command = container_prep.get("command")
        if isinstance(prep_command, Sequence):
            lines.append(" ".join(shlex.quote(str(part)) for part in prep_command))
    for step in plan.get("steps", []):
        lines.append(" ".join(shlex.quote(part) for part in step["command"]))
    if len(lines) == 3:
        lines.append("printf 'No derivative-backed runs were selected.\\n' >&2")
        lines.append("exit 1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def execute_runtime_plan(plan: Mapping[str, Any], *, cwd: str | Path) -> int:
    container_prep = plan.get("container_prep")
    if isinstance(container_prep, Mapping):
        prep_command = container_prep.get("command")
        prep_marker_path = _optional_value(container_prep.get("marker_path"))
        prep_marker_exists = prep_marker_path is not None and Path(prep_marker_path).exists()
        if isinstance(prep_command, Sequence) and not prep_marker_exists:
            completed = subprocess.run(list(prep_command), cwd=Path(cwd), check=False)
            if completed.returncode != 0:
                return completed.returncode
    for step in plan.get("steps", []):
        completed = subprocess.run(step["command"], cwd=Path(cwd), check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0 if plan.get("steps") else 1


def _resolve_templateflow_home(plan_path: Path, override: str | Path | None) -> Path:
    if override is not None and str(override).strip():
        return Path(override).expanduser().resolve()
    return (plan_path.parents[3] / "templateflow").resolve()


def _filter_filename(
    *,
    subject_id: str,
    session_id: str | None,
    task_id: str | None,
    acquisition: str | None,
    direction: str | None,
    run_id: str | None,
) -> str:
    parts = [subject_id]
    if session_id:
        parts.append(session_id)
    if task_id:
        parts.append(f"task-{task_id}")
    if acquisition:
        parts.append(f"acq-{acquisition}")
    if direction:
        parts.append(f"dir-{direction}")
    if run_id:
        parts.append(f"run-{run_id}")
    return "_".join(parts) + "_bids-filter.json"


def _run_label(
    *,
    subject_id: str,
    session_id: str | None,
    task_id: str | None,
    acquisition: str | None,
    direction: str | None,
    run_id: str | None,
) -> str:
    parts = [subject_id]
    if session_id:
        parts.append(session_id)
    if task_id:
        parts.append(f"task-{task_id}")
    if acquisition:
        parts.append(f"acq-{acquisition}")
    if direction:
        parts.append(f"dir-{direction}")
    if run_id:
        parts.append(f"run-{run_id}")
    return "_".join(parts)


def _build_runtime_units(steps: Sequence[Mapping[str, Any]], *, marker_root: Path) -> list[dict[str, Any]]:
    grouped_steps: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        unit_id = _runtime_unit_id(step)
        grouped_steps.setdefault(unit_id, []).append(dict(step))

    units: list[dict[str, Any]] = []
    for unit_id in sorted(grouped_steps):
        unit_steps = grouped_steps[unit_id]
        units.append(
            {
                "unit_id": unit_id,
                "marker_path": str(marker_root / f"{unit_id}.txt"),
                "step_count": len(unit_steps),
                "steps": unit_steps,
            }
        )
    return units


def _build_container_prep_step(
    *,
    backend: str,
    image_reference: str,
    pull_mode: str | None,
    image_root: str | None,
    image_name: str | None,
    marker_root: Path,
) -> dict[str, Any] | None:
    if backend not in {"apptainer", "singularity"}:
        return None

    image_details = _resolve_apptainer_runtime_image(
        image_reference=image_reference,
        pull_mode=pull_mode,
        image_root=image_root,
        image_name=image_name,
    )
    if not image_details["requires_pull"]:
        return None

    marker_path = marker_root / "_container-ready.txt"
    command = [
        "bash",
        "-lc",
        _build_apptainer_prep_shell(
            backend=backend,
            runtime_image=str(image_details["runtime_image"]),
            image_source=str(image_details["image_source"]),
            image_root=str(image_details["image_root"]),
        ),
    ]
    return {
        "marker_path": str(marker_path),
        "runtime_image": str(image_details["runtime_image"]),
        "image_source": str(image_details["image_source"]),
        "image_root": str(image_details["image_root"]),
        "command": command,
    }


def _runtime_unit_id(step: Mapping[str, Any]) -> str:
    subject_id = _optional_value(step.get("subject_id"))
    if subject_id is None:
        raise ValueError("Runtime-plan steps must define subject_id for unit grouping.")
    return subject_id


def _build_container_command(
    *,
    backend: str,
    image_reference: str,
    env: Mapping[str, str],
    raw_bids_root: Path,
    derivative_root: Path,
    derivative_name: str,
    output_root: Path,
    work_root: Path,
    filter_path: Path,
    templateflow_home: Path,
    subject_label: str,
    task_id: str | None,
    tool_options: Mapping[str, Any],
    nprocs: int,
    omp_nthreads: int,
    mem_mb: int | None,
    container_pull_mode: str | None,
    container_image_root: str | None,
    container_image_name: str | None,
) -> list[str]:
    derivative_mount = f"/derivatives/{derivative_name}"
    container_args = [
        "/data",
        "/out",
        "participant",
        "--skip-bids-validation",
        "--participant-label",
        subject_label,
        "--bids-filter-file",
        f"/filters/{filter_path.name}",
        "--derivatives",
        f"{derivative_name}={derivative_mount}",
        "--work-dir",
        f"/work/{work_root.name}",
        "--nprocs",
        str(max(1, int(nprocs))),
        "--omp-nthreads",
        str(max(1, int(omp_nthreads))),
    ]
    if mem_mb is not None:
        container_args.extend(["--mem", str(max(1, int(mem_mb)))])
    container_args.append("--notrack")
    if task_id:
        container_args.extend(["--task-id", task_id])
    if tool_options.get("denoising_method"):
        container_args.extend(["--denoising-method", str(tool_options["denoising_method"])])
    if tool_options.get("melodic_dimensionality") not in (None, ""):
        container_args.extend(["--melodic-dimensionality", str(tool_options["melodic_dimensionality"])])
    if tool_options.get("melodic_seed") not in (None, ""):
        container_args.extend(["--random-seed", str(tool_options["melodic_seed"])])
    if tool_options.get("dummy_scans") not in (None, ""):
        container_args.extend(["--dummy-scans", str(tool_options["dummy_scans"])])
    if bool(tool_options.get("low_mem", False)):
        container_args.append("--low-mem")

    mounts = [
        (raw_bids_root, "/data", True),
        (derivative_root, derivative_mount, True),
        (output_root, "/out", False),
        (work_root.parent, "/work", False),
        (filter_path.parent, "/filters", True),
        (templateflow_home, "/templateflow", False),
    ]
    if backend == "docker":
        return _docker_command(image_reference=image_reference, env=env, mounts=mounts, container_args=container_args)
    if backend in {"apptainer", "singularity"}:
        return _apptainer_command(
            backend=backend,
            image_reference=image_reference,
            env=env,
            mounts=mounts,
            container_args=container_args,
            pull_mode=container_pull_mode,
            image_root=container_image_root,
            image_name=container_image_name,
        )
    raise ValueError(f"Unsupported container backend: {backend}")


def _docker_command(
    *,
    image_reference: str,
    env: Mapping[str, str],
    mounts: Sequence[tuple[Path, str, bool]],
    container_args: Sequence[str],
) -> list[str]:
    command = ["docker", "run", "--rm"]
    for key, value in env.items():
        command.extend(["-e", f"{key}={value}"])
    for host_path, container_path, read_only in mounts:
        suffix = ":ro" if read_only else ""
        command.extend(["-v", f"{host_path}:{container_path}{suffix}"])
    command.append(image_reference)
    command.extend(container_args)
    return command


def _apptainer_command(
    *,
    backend: str,
    image_reference: str,
    env: Mapping[str, str],
    mounts: Sequence[tuple[Path, str, bool]],
    container_args: Sequence[str],
    pull_mode: str | None,
    image_root: str | None,
    image_name: str | None,
) -> list[str]:
    image_details = _resolve_apptainer_runtime_image(
        image_reference=image_reference,
        pull_mode=pull_mode,
        image_root=image_root,
        image_name=image_name,
    )
    shell = _build_apptainer_run_shell(
        backend=backend,
        env=env,
        mounts=mounts,
        container_args=container_args,
        runtime_image=image_details["runtime_image"],
        image_source=image_details["image_source"],
        image_root=image_details["image_root"],
        requires_pull=image_details["requires_pull"],
    )
    return ["bash", "-lc", shell]


def _normalize_selectors(selection: Mapping[str, str | None]) -> dict[str, str | None]:
    return {
        "subject_id": _optional_value(selection.get("subject_id")),
        "session_id": _optional_value(selection.get("session_id")),
        "task_id": _optional_value(selection.get("task_id")),
        "run_id": _optional_value(selection.get("run_id")),
    }


def _resolve_row_selectors(
    row: Mapping[str, str],
    *,
    default_selection: Mapping[str, str | None],
) -> dict[str, str | None]:
    subject_id = _optional_value(row.get("subject_id"))
    if subject_id is None:
        raise ValueError("Batch rows must define subject_id.")
    return {
        "subject_id": subject_id,
        "session_id": _optional_value(row.get("session_id")) or default_selection.get("session_id"),
        "task_id": _optional_value(row.get("task_id")) or default_selection.get("task_id"),
        "run_id": _optional_value(row.get("run_id")) or default_selection.get("run_id"),
    }


def _normalize_tool_options(tool_options: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = tool_options or {}
    runtime_grouping = _optional_value(payload.get("runtime_grouping")) or "compatible"
    if runtime_grouping not in _SUPPORTED_RUNTIME_GROUPINGS:
        allowed_groupings = ", ".join(sorted(_SUPPORTED_RUNTIME_GROUPINGS))
        raise ValueError(f"preprocessing.tool_options.runtime_grouping must be one of: {allowed_groupings}.")
    return {
        "denoising_method": _optional_value(payload.get("denoising_method")),
        "melodic_dimensionality": _optional_value(payload.get("melodic_dimensionality")),
        "melodic_seed": _optional_value(payload.get("melodic_seed")),
        "dummy_scans": _optional_value(payload.get("dummy_scans")),
        "low_mem": _coerce_bool(payload.get("low_mem")),
        "runtime_grouping": runtime_grouping,
    }


def _optional_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _normalize_apptainer_pull_mode(value: str | None) -> str:
    pull_mode = _optional_value(value) or DEFAULT_APPTAINER_PULL_MODE
    if pull_mode not in {"never", "if_missing"}:
        supported = ", ".join(sorted({"never", "if_missing"}))
        raise ValueError(f"Unsupported fMRIPost-AROMA Apptainer pull_mode {pull_mode!r}. Expected one of: {supported}.")
    return pull_mode


def _resolve_apptainer_runtime_image(
    *,
    image_reference: str,
    pull_mode: str | None,
    image_root: str | None,
    image_name: str | None,
) -> dict[str, Any]:
    image_source = _normalize_apptainer_image_reference(image_reference)
    resolved_pull_mode = _normalize_apptainer_pull_mode(pull_mode)
    resolved_image_root = _optional_value(image_root) or DEFAULT_APPTAINER_IMAGE_ROOT
    resolved_image_name = _normalize_image_name(image_name) or _derive_container_image_name(image_source)

    runtime_image = image_source
    requires_pull = False
    if image_source.startswith("docker://") and resolved_pull_mode == "if_missing":
        runtime_image = _join_runtime_path(resolved_image_root, resolved_image_name)
        requires_pull = True

    return {
        "image_source": image_source,
        "runtime_image": runtime_image,
        "image_root": resolved_image_root,
        "requires_pull": requires_pull,
    }


def _normalize_apptainer_image_reference(image_reference: str) -> str:
    text = image_reference.strip()
    if "://" in text:
        return text
    if text.endswith(".sif") or text.startswith(("/", "./", "../", "$", "${")):
        return text
    return f"docker://{text}"


def _normalize_image_name(value: str | None) -> str | None:
    if value is None:
        return None
    return value if value.endswith(".sif") else f"{value}.sif"


def _derive_container_image_name(image_reference: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", image_reference.removeprefix("docker://")).strip("-")
    if not sanitized:
        sanitized = "fmripost-aroma"
    if not sanitized.endswith(".sif"):
        sanitized += ".sif"
    return sanitized


def _join_runtime_path(root: str, leaf: str) -> str:
    return f"{root.rstrip('/')}/{leaf}" if root.rstrip("/") else leaf


def _build_apptainer_run_shell(
    *,
    backend: str,
    env: Mapping[str, str],
    mounts: Sequence[tuple[Path, str, bool]],
    container_args: Sequence[str],
    runtime_image: str,
    image_source: str,
    image_root: str,
    requires_pull: bool,
) -> str:
    env_part = ""
    if env:
        env_part = " --env " + shlex.quote(",".join(f"{key}={env[key]}" for key in sorted(env)))
    bind_part = "".join(f" --bind {shlex.quote(f'{host_path}:{container_path}')}" for host_path, container_path, _ in mounts)
    container_args_part = " ".join(shlex.quote(str(part)) for part in container_args)
    exec_shell = f"exec {shlex.quote(backend)} run --cleanenv{env_part}{bind_part} \"$RUNTIME_IMAGE\" {container_args_part}"

    lines = [
        "set -euo pipefail",
        f"RUNTIME_IMAGE={_double_quoted_shell_value(runtime_image)}",
    ]
    if not requires_pull:
        lines.append(exec_shell)
        return "\n".join(lines)

    lines.extend(
        [
            f"IMAGE_ROOT={_double_quoted_shell_value(image_root)}",
            f"IMAGE_SOURCE={_double_quoted_shell_value(image_source)}",
            'LOCK_DIR="${RUNTIME_IMAGE}.lock.d"',
            'TMP_IMAGE="${RUNTIME_IMAGE}.tmp.$$"',
            'cleanup(){',
            '  if [ -n "${LOCK_DIR:-}" ]; then',
            '    rmdir "$LOCK_DIR" 2>/dev/null || true',
            "  fi",
            '  if [ -n "${TMP_IMAGE:-}" ] && [ -f "$TMP_IMAGE" ]; then',
            '    rm -f "$TMP_IMAGE"',
            "  fi",
            "}",
            "trap cleanup EXIT INT TERM",
            'mkdir -p "$IMAGE_ROOT"',
            'while ! mkdir "$LOCK_DIR" 2>/dev/null; do',
            "  sleep 1",
            "done",
            'if [ ! -s "$RUNTIME_IMAGE" ]; then',
            f"  {shlex.quote(backend)} pull \"$TMP_IMAGE\" \"$IMAGE_SOURCE\"",
            '  mv "$TMP_IMAGE" "$RUNTIME_IMAGE"',
            "fi",
            exec_shell,
        ]
    )
    return "\n".join(lines)


def _build_apptainer_prep_shell(
    *,
    backend: str,
    runtime_image: str,
    image_source: str,
    image_root: str,
) -> str:
    lines = [
        "set -euo pipefail",
        f'export APPTAINER_CACHEDIR="${{APPTAINER_CACHEDIR:-{DEFAULT_APPTAINER_CACHE_DIR}}}"',
        f'export APPTAINER_TMPDIR="${{APPTAINER_TMPDIR:-{DEFAULT_APPTAINER_TMPDIR}}}"',
        'export TMPDIR="${TMPDIR:-$APPTAINER_TMPDIR}"',
        f"RUNTIME_IMAGE={_double_quoted_shell_value(runtime_image)}",
        f"IMAGE_SOURCE={_double_quoted_shell_value(image_source)}",
        f"IMAGE_ROOT={_double_quoted_shell_value(image_root)}",
        'TMP_IMAGE="${RUNTIME_IMAGE}.tmp.$$"',
        'mkdir -p "$IMAGE_ROOT" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"',
        'if [ -s "$RUNTIME_IMAGE" ]; then',
        "  exit 0",
        "fi",
        'rm -rf "${RUNTIME_IMAGE}.lock.d"',
        'rm -f "${RUNTIME_IMAGE}"',
        'rm -f "${RUNTIME_IMAGE}.tmp."*',
        f'{shlex.quote(backend)} pull "$TMP_IMAGE" "$IMAGE_SOURCE"',
        'mv "$TMP_IMAGE" "$RUNTIME_IMAGE"',
    ]
    return "\n".join(lines)


def _double_quoted_shell_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{escaped}"'


def _runtime_step_group_key(
    *,
    runtime_grouping: str,
    subject_id: str,
    run: Any,
) -> tuple[str, ...]:
    if runtime_grouping == "compatible":
        return (
            runtime_grouping,
            subject_id,
            _prefixed_entity_label("ses", run.session_id) or "",
            str(run.task_id or ""),
            str(run.acquisition or ""),
            str(run.direction or ""),
        )
    if runtime_grouping == "row":
        return (runtime_grouping, *_matched_run_key(run))
    raise ValueError(f"Unsupported runtime grouping: {runtime_grouping}")


def _matched_run_key(run: Any) -> tuple[str, ...]:
    return (
        str(run.subject_id),
        str(run.session_id or ""),
        str(run.task_id or ""),
        str(run.run_id or ""),
        str(run.acquisition or ""),
        str(run.direction or ""),
        str(run.source_file),
    )


def _matched_run_sort_key(run: Any) -> tuple[str, ...]:
    return (
        str(run.subject_id),
        str(run.session_id or ""),
        str(run.task_id or ""),
        str(run.acquisition or ""),
        str(run.direction or ""),
        str(run.run_id or ""),
        str(run.source_file),
    )


def _prefixed_entity_label(prefix: str, value: str | None) -> str | None:
    normalized = _optional_value(value)
    if normalized is None:
        return None
    if normalized.startswith(f"{prefix}-"):
        return normalized
    return f"{prefix}-{normalized}"


def _command_task_id(explicit_task_ids: set[str | None]) -> str | None:
    normalized = {task_id for task_id in explicit_task_ids}
    if len(normalized) != 1:
        return None
    return next(iter(normalized))
