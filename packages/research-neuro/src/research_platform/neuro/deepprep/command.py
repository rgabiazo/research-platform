"""Runtime-plan assembly helpers for DeepPrep."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
from typing import Any, Mapping, Sequence

from research_platform.core.config import resolve_env_value

from .selection import normalize_entity_label

DEFAULT_IMAGE = "docker://pbfslab/deepprep:25.1.0"
DEFAULT_LOCAL_BACKEND = "docker"
DEFAULT_HPC_BACKEND = "apptainer"
DEFAULT_APPTAINER_PULL_MODE = "if_missing"
DEFAULT_APPTAINER_IMAGE_ROOT = "${RP_REMOTE_CONTAINER_ROOT:-$SCRATCH/containers/deepprep}"
DEFAULT_APPTAINER_IMAGE_NAME = "deepprep_25.1.0.sif"
DEFAULT_NEXTFLOW_VERSION = "24.10.3"
DEFAULT_NEXTFLOW_HOST_HOME = "${RP_DEEPPREP_NEXTFLOW_HOME:-$SCRATCH/deepprep/nextflow}"
DEFAULT_NEXTFLOW_CONTAINER_HOME = "/output/WorkDir/nextflow"
_SUPPORTED_BACKENDS = {"docker", "apptainer", "singularity"}
_BOOLEAN_FLAGS = (
    "anat_only",
    "bold_only",
    "bold_sdc",
    "bold_confounds",
    "bold_cifti",
    "ignore_error",
    "resume",
)
_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
}


def build_runtime_plan(
    *,
    manifest: Mapping[str, Any],
    workspace_root: str | Path,
    plan_path: str | Path,
    command_script_path: str | Path,
) -> dict[str, Any]:
    workspace_root_path = Path(workspace_root).resolve()
    output_root = _resolve_manifest_path(workspace_root_path, manifest["execution"]["output_dir"])
    work_root = _resolve_manifest_path(workspace_root_path, manifest["execution"]["work_dir"])
    plan_path_obj = Path(plan_path).resolve()
    command_script_path_obj = Path(command_script_path).resolve()
    output_data_dirname = (
        manifest.get("tool", {}).get("runtime_metadata", {}).get("output_data_dirname") or "deepprep_units"
    )
    output_data_root = output_root / str(output_data_dirname)
    unit_marker_root = output_root / "runtime-plan-markers" / "deepprep"
    output_data_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    unit_marker_root.mkdir(parents=True, exist_ok=True)

    raw_bids_root = _resolve_manifest_path(workspace_root_path, manifest["dataset"]["root"])
    fs_license_file = _resolve_fs_license_file(manifest=manifest, workspace_root=workspace_root_path)
    batch_rows = _read_batch_rows(_resolve_manifest_path(workspace_root_path, manifest["batch"]["path"]))
    runtime_resources = _resolve_runtime_resources(manifest.get("resources"))
    runtime_backend = _resolve_runtime_backend(manifest)
    resolved_tool_options = _normalize_tool_options(manifest.get("tool", {}).get("options"))
    selection = _normalize_selection(manifest.get("selection", {}))
    output_config = dict(manifest.get("tool", {}).get("output", {})) if isinstance(manifest.get("tool", {}).get("output"), dict) else {}
    container_prep = _build_container_prep_step(
        backend=runtime_backend["backend"],
        image_reference=runtime_backend["image"],
        pull_mode=runtime_backend["pull_mode"],
        image_root=runtime_backend["image_root"],
        image_name=runtime_backend["image_name"],
        marker_root=unit_marker_root,
    )
    execution_image = runtime_backend["image"]
    execution_pull_mode = runtime_backend["pull_mode"]
    execution_image_root = runtime_backend["image_root"]
    execution_image_name = runtime_backend["image_name"]
    if container_prep is not None:
        execution_image = str(container_prep["runtime_image"])
        execution_pull_mode = "never"

    steps: list[dict[str, Any]] = []
    for unit in _runtime_units_from_rows(batch_rows, selection=selection, tool_options=resolved_tool_options):
        unit_id = _sanitize_label(_unit_dir_name(unit=unit, output_config=output_config))
        unit_output_root = output_data_root / unit_id
        unit_work_root = work_root / unit_id
        unit_output_root.mkdir(parents=True, exist_ok=True)
        unit_work_root.mkdir(parents=True, exist_ok=True)
        env = _build_deepprep_environment(
            unit_output_root=unit_output_root,
            threads=runtime_resources["cpus"],
            runtime_environment=runtime_backend["environment"],
            set_home_env=runtime_backend["backend"] == "docker",
            nextflow=runtime_backend["nextflow"],
        )
        command = _build_container_command(
            backend=runtime_backend["backend"],
            image_reference=execution_image,
            env=env,
            raw_bids_root=raw_bids_root,
            output_root=unit_output_root,
            fs_license_file=fs_license_file,
            subject_label=unit["subject_label"],
            task_label=unit["task_label"],
            tool_options=resolved_tool_options,
            resources=runtime_resources,
            pull_mode=execution_pull_mode,
            image_root=execution_image_root,
            image_name=execution_image_name,
            nextflow=runtime_backend["nextflow"],
        )
        steps.append(
            {
                "unit_id": unit_id,
                "subject_id": f"sub-{unit['subject_label']}",
                "session_id": unit.get("session_id") or "",
                "task_id": f"task-{unit['task_label']}" if unit.get("task_label") else "",
                "run_id": unit.get("run_id") or "",
                "row_count": len(unit["rows"]),
                "rows": unit["rows"],
                "output_dir": str(unit_output_root),
                "work_dir": str(unit_work_root),
                "env": env,
                "command": command,
            }
        )

    if not steps:
        container_prep = None
    units = _build_plan_units(steps, marker_root=unit_marker_root)
    plan = {
        "tool": "deepprep",
        "backend": runtime_backend["backend"],
        "image": {
            "reference": runtime_backend["image"],
            "runtime_image": str(container_prep["runtime_image"]) if container_prep else runtime_backend["image"],
            "pull_mode": runtime_backend["pull_mode"],
            "image_root": runtime_backend["image_root"],
            "image_name": runtime_backend["image_name"],
        },
        "container_prep": container_prep,
        "raw_bids_root": str(raw_bids_root),
        "fs_license_file": str(fs_license_file),
        "output_root": str(output_root),
        "output_data_root": str(output_data_root),
        "work_root": str(work_root),
        "plan_path": str(plan_path_obj),
        "command_script": str(command_script_path_obj),
        "unit_marker_root": str(unit_marker_root),
        "resources": runtime_resources,
        "nextflow": runtime_backend["nextflow"],
        "tool_options": resolved_tool_options,
        "steps": steps,
        "units": units,
        "skipped_rows": [],
    }
    write_runtime_plan(plan, plan_path_obj)
    write_command_script(plan, command_script_path_obj)
    return plan


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
        if isinstance(prep_command, Sequence) and not isinstance(prep_command, (str, bytes)):
            lines.append(" ".join(shlex.quote(str(part)) for part in prep_command))
    for step in plan.get("steps", []):
        command = step.get("command", []) if isinstance(step, Mapping) else []
        if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
            lines.append(" ".join(shlex.quote(str(part)) for part in command))
    if len(lines) == 3:
        lines.append("printf 'No DeepPrep units were selected.\\n' >&2")
        lines.append("exit 1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _resolve_manifest_path(workspace_root: Path, value: Any) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root / candidate).resolve()


def _read_batch_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: str(value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle, delimiter="\t")]


def _resolve_fs_license_file(*, manifest: Mapping[str, Any], workspace_root: Path) -> Path:
    tool_inputs = manifest.get("tool", {}).get("inputs", {})
    if not isinstance(tool_inputs, Mapping):
        raise ValueError("DeepPrep requires tool.inputs.fs_license_file.")
    mode = str(manifest.get("execution", {}).get("mode", "local")).strip()
    raw_value = tool_inputs.get("remote_fs_license_file") if mode == "slurm" else tool_inputs.get("fs_license_file")
    value = resolve_env_value(raw_value)
    if value is None and mode == "slurm":
        value = resolve_env_value(tool_inputs.get("fs_license_file"))
    if value is None:
        raise ValueError("DeepPrep requires preprocessing.inputs.fs_license_file.")
    return _resolve_manifest_path(workspace_root, value)


def _resolve_runtime_resources(resources: Any) -> dict[str, int]:
    payload = resources if isinstance(resources, Mapping) else {}
    cpus = max(1, int(payload.get("cpus", 4)))
    ram_gb = max(1, int(float(payload.get("ram_gb", 32))))
    return {"cpus": cpus, "memory_gb": ram_gb}


def _resolve_runtime_backend(manifest: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(manifest.get("execution", {}).get("mode", "local")).strip()
    runtime_profile = manifest.get("tool", {}).get("runtime_profile", {})
    config = runtime_profile.get("config", {}) if isinstance(runtime_profile, Mapping) else {}
    profile_block = config.get("slurm" if mode == "slurm" else "local", {}) if isinstance(config, Mapping) else {}
    if not isinstance(profile_block, Mapping):
        profile_block = {}
    backend = _optional_value(profile_block.get("execution_backend"))
    if backend is None:
        backend = DEFAULT_HPC_BACKEND if mode == "slurm" else DEFAULT_LOCAL_BACKEND
    if backend not in _SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(_SUPPORTED_BACKENDS))
        raise ValueError(f"Unsupported DeepPrep execution backend {backend!r}. Expected one of: {supported}.")

    container_block = profile_block.get("container", {}) if isinstance(profile_block.get("container", {}), Mapping) else {}
    image = _resolve_config_text(container_block.get("image")) or _resolve_config_text(profile_block.get("image")) or DEFAULT_IMAGE
    pull_mode = _resolve_config_text(container_block.get("pull_mode")) or DEFAULT_APPTAINER_PULL_MODE
    image_root = _resolve_config_text(container_block.get("image_root")) or DEFAULT_APPTAINER_IMAGE_ROOT
    image_name = _normalize_image_name(_resolve_config_text(container_block.get("image_name")) or DEFAULT_APPTAINER_IMAGE_NAME)
    environment = _resolve_environment(profile_block.get("environment"))
    nextflow = _resolve_nextflow_config(profile_block.get("nextflow"), backend=backend)
    return {
        "backend": backend,
        "image": image,
        "pull_mode": pull_mode,
        "image_root": image_root,
        "image_name": image_name,
        "environment": environment,
        "nextflow": nextflow,
    }


def _resolve_nextflow_config(value: Any, *, backend: str) -> dict[str, str]:
    payload = value if isinstance(value, Mapping) else {}
    enabled = _coerce_bool(payload.get("enabled"), default=backend in {"apptainer", "singularity"})
    version = _resolve_config_text(payload.get("version")) or DEFAULT_NEXTFLOW_VERSION
    jar_name = _resolve_config_text(payload.get("jar_name")) or f"nextflow-{version}-one.jar"
    return {
        "enabled": "true" if enabled else "false",
        "version": version,
        "host_home": _resolve_config_text(payload.get("host_home") or payload.get("home")) or DEFAULT_NEXTFLOW_HOST_HOME,
        "container_home": _resolve_config_text(payload.get("container_home")) or DEFAULT_NEXTFLOW_CONTAINER_HOME,
        "jar_name": jar_name,
        "jar_url": _resolve_config_text(payload.get("jar_url")) or f"https://www.nextflow.io/releases/v{version}/{jar_name}",
    }


def _resolve_environment(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    environment: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name).strip()
        resolved = _resolve_config_text(raw_value)
        if name and resolved:
            environment[name] = resolved
    return environment


def _resolve_config_text(value: Any) -> str | None:
    return resolve_env_value(value)


def _normalize_tool_options(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, Mapping) else {}
    return {
        "bold_task_type": _optional_value(payload.get("bold_task_type")),
        "anat_only": _coerce_bool(payload.get("anat_only")),
        "bold_only": _coerce_bool(payload.get("bold_only")),
        "bold_sdc": _coerce_bool(payload.get("bold_sdc"), default=True),
        "bold_confounds": _coerce_bool(payload.get("bold_confounds"), default=True),
        "bold_skip_frame": _optional_value(payload.get("bold_skip_frame") or payload.get("bold_skip_frames") or 0) or "0",
        "bold_cifti": _coerce_bool(payload.get("bold_cifti")),
        "bold_surface_spaces": _optional_value(payload.get("bold_surface_spaces")) or "fsaverage6",
        "bold_volume_space": _optional_value(payload.get("bold_volume_space")) or "MNI152NLin6Asym",
        "bold_volume_res": _optional_value(payload.get("bold_volume_res")) or "02",
        "skip_bids_validation": _coerce_bool(payload.get("skip_bids_validation"), default=True),
        "ignore_error": _coerce_bool(payload.get("ignore_error")),
        "resume": _coerce_bool(payload.get("resume"), default=True),
        "device": _optional_value(payload.get("device")) or "cpu",
    }


def _normalize_selection(value: Any) -> dict[str, str | None]:
    payload = value if isinstance(value, Mapping) else {}
    return {
        "subject_id": _optional_value(payload.get("subject_id")),
        "session_id": _optional_value(payload.get("session_id")),
        "task_id": _optional_value(payload.get("task_id")),
        "run_id": _optional_value(payload.get("run_id")),
    }


def _runtime_units_from_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    selection: Mapping[str, str | None],
    tool_options: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    configured_task = normalize_entity_label(_optional_value(tool_options.get("bold_task_type")), "task")
    selected_task = normalize_entity_label(selection.get("task_id"), "task")
    selected_session = _prefixed_label("ses", selection.get("session_id"))
    selected_run = _prefixed_label("run", selection.get("run_id"))
    for row in rows:
        subject_label = normalize_entity_label(row.get("subject_id") or selection.get("subject_id"), "sub")
        if not subject_label:
            continue
        task_label = configured_task or normalize_entity_label(row.get("task_id"), "task") or selected_task
        session_id = row.get("session_id") or selected_session or ""
        run_id = row.get("run_id") or selected_run or ""
        key = (subject_label, task_label or "", session_id)
        group = grouped.setdefault(
            key,
            {
                "subject_label": subject_label,
                "task_label": task_label,
                "session_id": session_id,
                "run_id": run_id,
                "rows": [],
            },
        )
        group["rows"].append(dict(row))
    return [grouped[key] for key in sorted(grouped)]


def _build_deepprep_environment(
    *,
    unit_output_root: Path,
    threads: int,
    runtime_environment: Mapping[str, str],
    set_home_env: bool,
    nextflow: Mapping[str, str],
) -> dict[str, str]:
    thread_value = str(max(1, int(threads)))
    env = {key: thread_value for key in _THREAD_ENVIRONMENT}
    env.update(
        {
            "NXF_HOME": str(nextflow.get("container_home") or DEFAULT_NEXTFLOW_CONTAINER_HOME),
            "NXF_WORK": "/output/WorkDir/nextflow-work",
            "TMPDIR": "/output/WorkDir/tmp",
            "TEMP": "/output/WorkDir/tmp",
            "TMP": "/output/WorkDir/tmp",
        }
    )
    if set_home_env:
        env["HOME"] = "/output/WorkDir/home"
    env.update({str(key): str(value) for key, value in runtime_environment.items() if str(key).strip()})
    for child in ("home", "nextflow", "nextflow-work", "tmp"):
        (unit_output_root / "WorkDir" / child).mkdir(parents=True, exist_ok=True)
    return env


def _build_container_command(
    *,
    backend: str,
    image_reference: str,
    env: Mapping[str, str],
    raw_bids_root: Path,
    output_root: Path,
    fs_license_file: Path,
    subject_label: str,
    task_label: str | None,
    tool_options: Mapping[str, Any],
    resources: Mapping[str, int],
    pull_mode: str,
    image_root: str,
    image_name: str,
    nextflow: Mapping[str, str],
) -> list[str]:
    container_args = _deepprep_args(
        subject_label=subject_label,
        task_label=task_label,
        tool_options=tool_options,
        resources=resources,
    )
    mounts = [
        (raw_bids_root, "/input", True),
        (output_root, "/output", False),
        (fs_license_file, "/fs_license.txt", True),
    ]
    if backend == "docker":
        return _docker_command(image_reference=_docker_image_reference(image_reference), env=env, mounts=mounts, container_args=container_args)
    if backend in {"apptainer", "singularity"}:
        return _apptainer_command(
            backend=backend,
            image_reference=image_reference,
            env=env,
            mounts=mounts,
            container_args=container_args,
            pull_mode=pull_mode,
            image_root=image_root,
            image_name=image_name,
            home_host=output_root / "WorkDir" / "home",
            nextflow=_nextflow_runtime_paths(nextflow=nextflow, unit_output_root=output_root),
        )
    raise ValueError(f"Unsupported DeepPrep backend: {backend}")


def _deepprep_args(
    *,
    subject_label: str,
    task_label: str | None,
    tool_options: Mapping[str, Any],
    resources: Mapping[str, int],
) -> list[str]:
    args = [
        "/input",
        "/output",
        "participant",
        "--fs_license_file",
        "/fs_license.txt",
        "--participant_label",
        _deepprep_participant_label(subject_label),
        "--device",
        str(tool_options["device"]),
        "--cpus",
        str(resources["cpus"]),
        "--memory",
        str(resources["memory_gb"]),
    ]
    if task_label:
        args.extend(["--bold_task_type", task_label])
    if tool_options.get("skip_bids_validation"):
        args.append("--skip_bids_validation")
    for name in _BOOLEAN_FLAGS:
        if tool_options.get(name):
            args.append(f"--{name}")
    if tool_options.get("bold_skip_frame") not in (None, ""):
        args.extend(["--bold_skip_frame", str(tool_options["bold_skip_frame"])])
    for name in ("bold_surface_spaces", "bold_volume_space", "bold_volume_res"):
        value = _optional_value(tool_options.get(name))
        if value is not None:
            args.extend([f"--{name}", value])
    return args


def _deepprep_participant_label(subject_label: str) -> str:
    normalized = normalize_entity_label(subject_label, "sub")
    if not normalized:
        return subject_label
    if normalized.isdigit():
        return f"'{normalized}'"
    return normalized


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
    pull_mode: str,
    image_root: str,
    image_name: str,
    home_host: Path,
    nextflow: Mapping[str, str],
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
        runtime_image=str(image_details["runtime_image"]),
        home_host=home_host,
        nextflow_jar_source=nextflow.get("jar_source", ""),
        nextflow_jar_target=nextflow.get("jar_target", ""),
    )
    return ["bash", "-lc", shell]


def _nextflow_runtime_paths(*, nextflow: Mapping[str, str], unit_output_root: Path) -> dict[str, str]:
    if str(nextflow.get("enabled", "")).lower() != "true":
        return {}
    version = nextflow.get("version") or DEFAULT_NEXTFLOW_VERSION
    jar_name = nextflow.get("jar_name") or f"nextflow-{version}-one.jar"
    host_home = nextflow.get("host_home") or DEFAULT_NEXTFLOW_HOST_HOME
    return {
        "jar_source": _join_runtime_path(_join_runtime_path(host_home, f"framework/{version}"), jar_name),
        "jar_target": str(unit_output_root / "WorkDir" / "nextflow" / "framework" / version / jar_name),
    }


def _build_container_prep_step(
    *,
    backend: str,
    image_reference: str,
    pull_mode: str,
    image_root: str,
    image_name: str,
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
    return {
        "marker_path": str(marker_path),
        "runtime_image": str(image_details["runtime_image"]),
        "image_source": str(image_details["image_source"]),
        "image_root": str(image_details["image_root"]),
        "command": [
            "bash",
            "-lc",
            _build_apptainer_prep_shell(
                backend=backend,
                runtime_image=str(image_details["runtime_image"]),
                image_source=str(image_details["image_source"]),
                image_root=str(image_details["image_root"]),
            ),
        ],
    }


def _resolve_apptainer_runtime_image(
    *,
    image_reference: str,
    pull_mode: str,
    image_root: str,
    image_name: str,
) -> dict[str, Any]:
    image_source = _normalize_apptainer_image_reference(image_reference)
    normalized_pull_mode = _normalize_pull_mode(pull_mode)
    normalized_image_name = _normalize_image_name(image_name) or _derive_container_image_name(image_source)
    runtime_image = image_source
    requires_pull = False
    if image_source.startswith("docker://") and normalized_pull_mode == "if_missing":
        runtime_image = _join_runtime_path(image_root, normalized_image_name)
        requires_pull = True
    return {
        "image_source": image_source,
        "runtime_image": runtime_image,
        "image_root": image_root,
        "requires_pull": requires_pull,
    }


def _build_apptainer_run_shell(
    *,
    backend: str,
    env: Mapping[str, str],
    mounts: Sequence[tuple[Path, str, bool]],
    container_args: Sequence[str],
    runtime_image: str,
    home_host: Path | None = None,
    nextflow_jar_source: str = "",
    nextflow_jar_target: str = "",
) -> str:
    env_part = ""
    if env:
        env_part = " --env " + shlex.quote(",".join(f"{key}={env[key]}" for key in sorted(env)))
    bind_part = "".join(f" --bind {_double_quoted_shell_value(f'{host_path}:{container_path}')}" for host_path, container_path, _ in mounts)
    args_part = " ".join(shlex.quote(str(part)) for part in container_args)
    setup_lines: list[str] = []
    home_part = ""
    if home_host is not None:
        setup_lines.extend(
            [
                f"DEEPPREP_HOME_HOST={_double_quoted_shell_value(str(home_host))}",
                'DEEPPREP_HOME_DEST="/home/${USER:-deepprep}"',
                'mkdir -p "$DEEPPREP_HOME_HOST"',
            ]
        )
        home_part = ' --home "$DEEPPREP_HOME_HOST:$DEEPPREP_HOME_DEST"'
    if nextflow_jar_source and nextflow_jar_target:
        setup_lines.extend(
            [
                f"NEXTFLOW_JAR_SOURCE={_double_quoted_shell_value(nextflow_jar_source)}",
                f"NEXTFLOW_JAR_TARGET={_double_quoted_shell_value(nextflow_jar_target)}",
                'if [ ! -s "$NEXTFLOW_JAR_SOURCE" ]; then',
                '  echo "ERROR: Missing pre-staged Nextflow jar: $NEXTFLOW_JAR_SOURCE" >&2',
                "  exit 66",
                "fi",
                'mkdir -p "$(dirname "$NEXTFLOW_JAR_TARGET")"',
                'if [ ! -s "$NEXTFLOW_JAR_TARGET" ]; then',
                '  cp "$NEXTFLOW_JAR_SOURCE" "$NEXTFLOW_JAR_TARGET"',
                "fi",
            ]
        )
    return "\n".join(
        [
            "set -euo pipefail",
            f"RUNTIME_IMAGE={_double_quoted_shell_value(runtime_image)}",
            *setup_lines,
            f"exec {shlex.quote(backend)} run --cleanenv{home_part}{env_part}{bind_part} \"$RUNTIME_IMAGE\" {args_part}",
        ]
    )


def _build_apptainer_prep_shell(
    *,
    backend: str,
    runtime_image: str,
    image_source: str,
    image_root: str,
) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            f"RUNTIME_IMAGE={_double_quoted_shell_value(runtime_image)}",
            f"IMAGE_SOURCE={_double_quoted_shell_value(image_source)}",
            f"IMAGE_ROOT={_double_quoted_shell_value(image_root)}",
            'LOCK_DIR="${RUNTIME_IMAGE}.lock.d"',
            'TMP_IMAGE="${RUNTIME_IMAGE}.tmp.$$"',
            'cleanup(){ rmdir "$LOCK_DIR" 2>/dev/null || true; rm -f "$TMP_IMAGE" 2>/dev/null || true; }',
            "trap cleanup EXIT INT TERM",
            'mkdir -p "$IMAGE_ROOT"',
            'while ! mkdir "$LOCK_DIR" 2>/dev/null; do sleep 5; done',
            'if [ ! -s "$RUNTIME_IMAGE" ]; then',
            f"  {shlex.quote(backend)} pull \"$TMP_IMAGE\" \"$IMAGE_SOURCE\"",
            '  mv "$TMP_IMAGE" "$RUNTIME_IMAGE"',
            "fi",
        ]
    )


def _build_plan_units(steps: Sequence[Mapping[str, Any]], *, marker_root: Path) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for step in sorted(steps, key=lambda item: str(item["unit_id"])):
        unit_id = str(step["unit_id"])
        units.append(
            {
                "unit_id": unit_id,
                "marker_path": str(marker_root / f"{unit_id}.txt"),
                "step_count": 1,
                "steps": [dict(step)],
            }
        )
    return units


def _unit_dir_name(*, unit: Mapping[str, Any], output_config: Mapping[str, Any]) -> str:
    template = _optional_value(output_config.get("unit_dir_template"))
    subject_id = f"sub-{unit['subject_label']}"
    task_label = str(unit.get("task_label") or "").strip()
    if template:
        return template.format(subject_id=subject_id, task_id=task_label, task_label=task_label)
    return f"{subject_id}-{task_label}" if task_label else subject_id


def _prefixed_label(prefix: str, value: str | None) -> str | None:
    label = normalize_entity_label(value, prefix)
    return f"{prefix}-{label}" if label else None


def _optional_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_pull_mode(value: str) -> str:
    normalized = _optional_value(value) or DEFAULT_APPTAINER_PULL_MODE
    if normalized not in {"never", "if_missing"}:
        supported = ", ".join(sorted({"never", "if_missing"}))
        raise ValueError(f"Unsupported DeepPrep pull_mode {normalized!r}. Expected one of: {supported}.")
    return normalized


def _normalize_apptainer_image_reference(value: str) -> str:
    text = str(value).strip()
    if "://" in text:
        return text
    if text.endswith(".sif") or text.startswith(("/", "./", "../", "$", "${")):
        return text
    return f"docker://{text}"


def _docker_image_reference(value: str) -> str:
    return value.removeprefix("docker://")


def _normalize_image_name(value: str | None) -> str | None:
    if value is None:
        return None
    return value if value.endswith(".sif") else f"{value}.sif"


def _derive_container_image_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.removeprefix("docker://")).strip("-")
    return f"{sanitized or 'deepprep'}.sif"


def _join_runtime_path(root: str, leaf: str) -> str:
    return f"{root.rstrip('/')}/{leaf}" if root.rstrip("/") else leaf


def _double_quoted_shell_value(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "deepprep-unit"
