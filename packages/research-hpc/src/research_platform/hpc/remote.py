
"""Remote stage/status/pull helpers for already planned runs."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
from typing import Any, Callable

from ._yaml import write_yaml
from .connection import ResolvedHpcConnection, build_manifest_hpc_connection_hint, resolve_hpc_connection
from .ssh import build_ssh_command
from .sync import (
    build_rsync_pull_command as build_host_rsync_pull_command,
    build_rsync_push_command as build_host_rsync_push_command,
)
from .transfers import build_rsync_pull_command as build_profile_rsync_pull_command, build_rsync_push_command as build_profile_rsync_push_command

_SBATCH_SUBMITTED_JOB_PATTERN = re.compile(r"Submitted batch job (\d+)")
_REMOTE_PATH_STATUS_PATTERN = re.compile(r"^(present|missing)\t(.*)$")
Runner = Callable[..., subprocess.CompletedProcess[str]]


def resolve_remote_run_root(*, run_id: str, remote_workspace_root: str | None, remote_artifacts_root: str | None) -> str:
    if remote_artifacts_root:
        return str(PurePosixPath(remote_artifacts_root) / "runs" / run_id)
    if remote_workspace_root:
        return str(PurePosixPath(remote_workspace_root) / "artifacts" / "runs" / run_id)
    return ""


def build_stage_plan(
    *,
    workspace_root: str | Path,
    run_root: str | Path,
    manifest: dict[str, Any],
    status: dict[str, Any],
    exclude_file: str | Path | None,
    profile_name: str | None = None,
    role: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    workspace_root_path = Path(workspace_root).resolve()
    run_root_path = Path(run_root)
    stage_dir = run_root_path / "hpc" / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    connection = resolve_hpc_connection(
        manifest=manifest,
        profile_name=profile_name,
        role=role,
        config_path=config_path,
        workspace_root=workspace_root_path,
    )
    remote_run_root = _require_remote_run_root(manifest)
    remote_workspace_root = manifest.get("provision", {}).get("remote_workspace_root") or manifest.get("hpc", {}).get("remote_workspace_root", "")
    staged_files = _stage_scope_payload(workspace_root=workspace_root_path, run_root=run_root_path, manifest=manifest, stage_dir=stage_dir)
    prepare_commands: list[list[str]] = []
    push_commands: list[dict[str, Any]] = []
    prepare_directories = _required_remote_directories(
        manifest=manifest,
        remote_workspace_root=remote_workspace_root,
        remote_run_root=remote_run_root,
    )
    if prepare_directories:
        prepare_commands.append(_build_remote_command(connection, "mkdir -p " + " ".join(shlex.quote(path) for path in prepare_directories)))
    push_commands.append(
        {
            "scope": "run",
            "label": "run-bundle",
            "kind": "directory",
            "source": str(stage_dir),
            "destination": remote_run_root,
            "exclude_files": [str(exclude_file)] if exclude_file and Path(exclude_file).exists() else [],
            "command": _build_push_command(
                connection=connection,
                source=stage_dir,
                destination=remote_run_root,
                exclude_file=exclude_file,
            ),
        }
    )
    for scope in manifest.get("provision", {}).get("scopes", []):
        for entry in scope.get("entries", []):
            push_commands.append(
                _build_scope_push_command(
                    workspace_root=workspace_root_path,
                    connection=connection,
                    remote_workspace_root=remote_workspace_root,
                    scope_name=str(scope["name"]),
                    entry=entry,
                )
            )
    submit_command = _build_remote_command(connection, f"cd {remote_run_root} && sbatch submit.sbatch")
    plan_path = run_root_path / "hpc" / "stage-plan.yaml"

    plan = {
        "run_id": manifest["run_id"],
        "connection": _render_connection_report(connection),
        "stage_dir": str(stage_dir),
        "staged_files": staged_files,
        "prepare_commands": prepare_commands,
        "push_command": push_commands[0]["command"] if push_commands else [],
        "push_commands": push_commands,
        "submit_command": submit_command,
        "local_files_written": [*staged_files, str(plan_path)],
    }
    write_yaml(plan_path, plan)
    next_status = dict(status) | {"state": "stage-prepared", "last_updated": _timestamp()}
    return {"report": plan, "status": next_status}


def build_pull_plan(
    *,
    workspace_root: str | Path,
    run_root: str | Path,
    manifest: dict[str, Any],
    status: dict[str, Any],
    exclude_file: str | Path | None,
    subpath: str | None = None,
    destination: str | Path | None = None,
    profile_name: str | None = None,
    role: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    _ = workspace_root
    run_root_path = Path(run_root)
    pull_root = run_root_path / "hpc" / "pulled"
    normalized_subpath = _normalize_pull_subpath(subpath)
    remote_source = _resolve_pull_remote_source(manifest=manifest, subpath=normalized_subpath)
    pull_dir = _resolve_pull_destination(
        run_root=run_root_path,
        pull_root=pull_root,
        subpath=normalized_subpath,
        destination=destination,
    )
    connection = resolve_hpc_connection(
        manifest=manifest,
        profile_name=profile_name,
        role=role,
        config_path=config_path,
        workspace_root=workspace_root,
    )
    pull_command = _build_pull_command(
        connection=connection,
        source=remote_source,
        destination=pull_dir,
        exclude_file=exclude_file,
        progress=True,
    )
    plan_path = run_root_path / "hpc" / "pull-plan.yaml"
    plan = {
        "run_id": manifest["run_id"],
        "connection": _render_connection_report(connection),
        "pull_scope": "artifacts",
        "subpath": normalized_subpath,
        "remote_source": remote_source,
        "destination": str(pull_dir),
        "progress": True,
        "pull_dir": str(pull_dir),
        "pull_command": pull_command,
        "local_files_written": [str(plan_path)],
    }
    write_yaml(plan_path, plan)
    next_status = dict(status) | {"state": "pull-prepared", "last_updated": _timestamp()}
    return {"report": plan, "status": next_status}


def build_cancel_plan(*, manifest: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    job_id = status.get("job_id") or manifest.get("slurm", {}).get("job_id", "")
    cancel_command: list[str] = []
    if job_id:
        try:
            cancel_command = _build_remote_command(resolve_hpc_connection(manifest=manifest), f"scancel {job_id}")
        except ValueError:
            cancel_command = []
    report = {
        "run_id": manifest["run_id"],
        "job_id": job_id,
        "cancel_command": cancel_command,
    }
    next_status = dict(status) | {"state": "cancel-requested", "last_updated": _timestamp()}
    return {"report": report, "status": next_status}


def execute_stage_plan(plan: dict[str, Any]) -> dict[str, Any]:
    executed: list[dict[str, Any]] = []
    for command in plan.get("prepare_commands", []):
        result = _run_command(command)
        executed.append(result)
        if result["returncode"] != 0:
            return {"ok": False, "returncode": result["returncode"], "commands": executed}
    for entry in plan.get("push_commands", []):
        result = _run_command(entry.get("command", []), scope=entry.get("scope"), label=entry.get("label"))
        executed.append(result)
        if result["returncode"] != 0:
            return {"ok": False, "returncode": result["returncode"], "commands": executed}
    return {"ok": True, "returncode": 0, "commands": executed}


def execute_pull_plan(plan: dict[str, Any]) -> dict[str, Any]:
    destination = str(plan.get("destination") or plan.get("pull_dir") or "").strip()
    if destination:
        Path(destination).mkdir(parents=True, exist_ok=True)
    result = _run_command(plan.get("pull_command", []), scope=plan.get("pull_scope"), label="pull")
    return {
        "ok": result["returncode"] == 0,
        "returncode": result["returncode"],
        "commands": [result],
    }


def build_submit_plan(
    *,
    manifest: dict[str, Any],
    status: dict[str, Any],
    profile_name: str | None = None,
    role: str | None = None,
    config_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    _ = status
    submit_command = list(manifest.get("hpc", {}).get("submit_command", []))
    if not submit_command:
        submit_command = _build_remote_command(
            resolve_hpc_connection(
                manifest=manifest,
                profile_name=profile_name,
                role=role,
                config_path=config_path,
                workspace_root=workspace_root,
            ),
            f"cd {_require_remote_run_root(manifest)} && sbatch submit.sbatch",
        )
    return {
        "run_id": manifest["run_id"],
        "submit_command": submit_command,
    }


def verify_remote_paths(
    *,
    paths: list[str],
    profile_name: str | None = None,
    role: str | None = None,
    config_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    normalized_paths = _dedupe_remote_paths(paths)
    connection = resolve_hpc_connection(
        manifest=_verification_connection_manifest(
            profile_name=profile_name,
            role=role,
            config_path=config_path,
        ),
        profile_name=profile_name,
        role=role,
        config_path=config_path,
        workspace_root=workspace_root,
    )
    command = _build_remote_path_verify_command(connection, normalized_paths)
    execution = _run_runner_command(command, runner=runner or subprocess.run, capture_output=True)
    reported_paths = _parse_remote_path_status_output(normalized_paths, execution["stdout"])
    return {
        "connection": _render_connection_report(connection),
        "command": command,
        "paths": reported_paths,
        "ok": execution["returncode"] == 0 and all(item["exists"] for item in reported_paths),
        "returncode": execution["returncode"],
        "stdout": execution["stdout"],
        "stderr": execution["stderr"],
    }


def execute_submit_plan(plan: dict[str, Any]) -> dict[str, Any]:
    result = _run_command(plan.get("submit_command", []), capture_output=True)
    stdout = result.get("stdout", "")
    match = _SBATCH_SUBMITTED_JOB_PATTERN.search(stdout)
    return {
        "ok": result["returncode"] == 0,
        "returncode": result["returncode"],
        "command": result["command"],
        "stdout": stdout,
        "stderr": result.get("stderr", ""),
        "job_id": match.group(1) if match else "",
    }


def _timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stage_scope_payload(
    *,
    workspace_root: Path,
    run_root: Path,
    manifest: dict[str, Any],
    stage_dir: Path,
) -> list[str]:
    staged_files: list[str] = []
    run_root_resolved = run_root.resolve()
    copies = [run_root / "run-manifest.yaml"]
    execute_script = run_root / "execute.sh"
    if execute_script.exists():
        copies.append(execute_script)
    script_path = manifest.get("slurm", {}).get("script_path")
    if script_path:
        copies.append((workspace_root / script_path).resolve())
    for command_part in manifest.get("execution", {}).get("command", []):
        for candidate_text in _command_part_path_candidates(str(command_part)):
            candidate = Path(candidate_text)
            if not candidate.is_absolute():
                candidate = (workspace_root / candidate).resolve()
            if candidate.exists() and candidate.is_file():
                try:
                    candidate.relative_to(run_root_resolved)
                except ValueError:
                    continue
                copies.append(candidate)
    seen: set[Path] = set()
    for source_path in copies:
        if source_path in seen or not source_path.exists():
            continue
        seen.add(source_path)
        try:
            relative_path = source_path.resolve().relative_to(run_root_resolved)
        except ValueError:
            relative_path = Path(source_path.name)
        target_path = stage_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        staged_files.append(str(target_path))
    return staged_files


def _command_part_path_candidates(command_part: str) -> list[str]:
    candidates = [command_part]
    if "=" in command_part:
        _, value = command_part.split("=", 1)
        if value:
            candidates.append(value)
    return candidates


def _build_scope_push_command(
    *,
    workspace_root: Path,
    connection: ResolvedHpcConnection,
    remote_workspace_root: str,
    scope_name: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    source_path = (workspace_root / entry["source"]).resolve()
    destination_path = _resolve_remote_destination_path(
        remote_workspace_root=remote_workspace_root,
        destination=entry["destination"],
    )
    command = _build_push_command(
        connection=connection,
        source=source_path,
        destination=destination_path,
        exclude_files=[workspace_root / path for path in entry.get("exclude_files", [])],
        source_is_directory=entry.get("kind", "directory") == "directory",
    )
    return {
        "scope": scope_name,
        "label": entry["label"],
        "kind": entry["kind"],
        "source": entry["source"],
        "destination": entry["destination"],
        "exclude_files": list(entry.get("exclude_files", [])),
        "command": command,
    }


def _build_remote_command(connection: ResolvedHpcConnection, remote_command: str) -> list[str]:
    if connection.profile is not None:
        return build_ssh_command(connection.profile, mode=connection.mode, remote_command=remote_command, allocate_tty=False)
    return ["ssh", connection.ssh_target, remote_command]


def _build_push_command(
    *,
    connection: ResolvedHpcConnection,
    source: str | Path,
    destination: str,
    exclude_file: str | Path | None = None,
    exclude_files: list[str | Path] | None = None,
    source_is_directory: bool = True,
) -> list[str]:
    if connection.profile is not None:
        return build_profile_rsync_push_command(
            source=source,
            profile=connection.profile,
            destination=destination,
            mode=connection.mode,
            exclude_file=exclude_file,
            exclude_files=exclude_files,
            source_is_directory=source_is_directory,
        )
    return build_host_rsync_push_command(
        source=source,
        ssh_host=connection.ssh_target,
        destination=destination,
        exclude_file=exclude_file,
        exclude_files=exclude_files,
        source_is_directory=source_is_directory,
    )


def _build_pull_command(
    *,
    connection: ResolvedHpcConnection,
    source: str,
    destination: str | Path,
    exclude_file: str | Path | None = None,
    exclude_files: list[str | Path] | None = None,
    progress: bool = False,
    source_is_directory: bool = True,
) -> list[str]:
    if connection.profile is not None:
        return build_profile_rsync_pull_command(
            profile=connection.profile,
            source=source,
            destination=destination,
            mode=connection.mode,
            exclude_file=exclude_file,
            exclude_files=exclude_files,
            progress=progress,
            source_is_directory=source_is_directory,
        )
    return build_host_rsync_pull_command(
        ssh_host=connection.ssh_target,
        source=source,
        destination=destination,
        exclude_file=exclude_file,
        exclude_files=exclude_files,
        progress=progress,
        source_is_directory=source_is_directory,
    )


def _normalize_pull_subpath(subpath: str | None) -> str:
    if subpath is None:
        return ""
    normalized = str(subpath).strip()
    if not normalized:
        raise ValueError("Pull subpath must be a non-empty relative directory under the remote run root.")
    if "\\" in normalized:
        raise ValueError("Pull subpath must use forward slashes and stay relative to the remote run root.")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute():
        raise ValueError("Pull subpath must be relative to the remote run root.")
    parts = candidate.parts
    if not parts or str(candidate) in {".", ""}:
        raise ValueError("Pull subpath must be a non-empty relative directory under the remote run root.")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Pull subpath must stay within the remote run root and must not contain traversal segments.")
    rendered = str(candidate)
    if rendered != normalized:
        raise ValueError("Pull subpath must be normalized without redundant separators or dot segments.")
    return rendered


def _resolve_pull_remote_source(*, manifest: dict[str, Any], subpath: str) -> str:
    remote_run_root = PurePosixPath(_require_remote_run_root(manifest))
    if not subpath:
        return str(remote_run_root)
    return str(remote_run_root / PurePosixPath(subpath))


def _resolve_pull_destination(
    *,
    run_root: Path,
    pull_root: Path,
    subpath: str,
    destination: str | Path | None,
) -> Path:
    if destination is None:
        if not subpath:
            return pull_root
        return pull_root / Path(*PurePosixPath(subpath).parts)
    _ = run_root
    return Path(destination).expanduser().resolve()


def _render_connection_report(connection: ResolvedHpcConnection) -> dict[str, str]:
    report = {
        "kind": connection.kind,
        "target": connection.ssh_target,
        "mode": connection.mode,
    }
    if connection.profile_name:
        report["profile"] = connection.profile_name
    if connection.role:
        report["role"] = connection.role
    if connection.config_path is not None:
        report["config"] = str(connection.config_path)
    return report


def _verification_connection_manifest(
    *,
    profile_name: str | None,
    role: str | None,
    config_path: str | Path | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {"hpc": {}}
    connection_hint = build_manifest_hpc_connection_hint(
        profile_name=profile_name,
        role=role,
        config_path=config_path,
    )
    if connection_hint is not None:
        manifest["hpc"]["connection"] = connection_hint
    return manifest


def _build_remote_path_verify_command(connection: ResolvedHpcConnection, paths: list[str]) -> list[str]:
    if not paths:
        return _build_remote_command(connection, "true")
    lines = ["set -eu"]
    for path in paths:
        quoted = shlex.quote(path)
        lines.extend(
            [
                f"if [ -e {quoted} ]; then",
                f"  printf 'present\\t%s\\n' {quoted}",
                "else",
                f"  printf 'missing\\t%s\\n' {quoted}",
                "fi",
            ]
        )
    return _build_remote_command(connection, "\n".join(lines))


def _parse_remote_path_status_output(paths: list[str], stdout: str) -> list[dict[str, Any]]:
    parsed: dict[str, bool] = {}
    for raw_line in stdout.splitlines():
        match = _REMOTE_PATH_STATUS_PATTERN.match(raw_line.strip())
        if match is None:
            continue
        parsed[match.group(2)] = match.group(1) == "present"
    return [{"path": path, "exists": parsed.get(path, False)} for path in paths]


def _dedupe_remote_paths(paths: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        normalized = str(PurePosixPath(str(raw_path).strip()))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _required_remote_directories(
    *,
    manifest: dict[str, Any],
    remote_workspace_root: str,
    remote_run_root: str,
) -> list[str]:
    required: list[str] = []
    if remote_workspace_root:
        required.append(str(PurePosixPath(remote_workspace_root)))
    if remote_run_root:
        required.append(str(PurePosixPath(remote_run_root)))
    required.extend(_runtime_remote_directories(manifest))
    for scope in manifest.get("provision", {}).get("scopes", []):
        for entry in scope.get("entries", []):
            destination_path = _resolve_remote_destination_path(
                remote_workspace_root=remote_workspace_root,
                destination=str(entry["destination"]),
            )
            if not destination_path:
                continue
            destination = PurePosixPath(destination_path)
            required.append(str(destination if entry.get("kind", "directory") == "directory" else destination.parent))
    deduped: list[str] = []
    seen: set[str] = set()
    for path in required:
        normalized = str(PurePosixPath(path))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _runtime_remote_directories(manifest: dict[str, Any]) -> list[str]:
    required: list[str] = []
    execution = manifest.get("execution", {})
    for key in ("log_dir", "work_dir", "output_dir"):
        value = str(execution.get(key, "")).strip()
        if value:
            required.append(str(PurePosixPath(value)))
    jobspec = manifest.get("slurm", {}).get("jobspec", {})
    for key in ("log_out", "log_err"):
        value = str(jobspec.get(key, "")).strip()
        if not value:
            continue
        parent = PurePosixPath(value).parent
        if str(parent) != ".":
            required.append(str(parent))
    return required


def _resolve_remote_destination_path(*, remote_workspace_root: str, destination: str) -> str:
    if remote_workspace_root:
        return str(PurePosixPath(remote_workspace_root) / destination)
    return destination


def _require_remote_run_root(manifest: dict[str, Any]) -> str:
    remote_run_root = str(manifest.get("hpc", {}).get("remote_run_root", "")).strip()
    if remote_run_root:
        return remote_run_root
    raise ValueError("Remote run root is not configured for this run manifest.")


def _run_command(
    command: list[str],
    *,
    scope: str | None = None,
    label: str | None = None,
    capture_output: bool = False,
) -> dict[str, Any]:
    if not command:
        return {
            "scope": scope or "",
            "label": label or "",
            "command": [],
            "returncode": 1,
            "stdout": "",
            "stderr": "Missing command.",
        }
    completed = subprocess.run(command, check=False, text=True, capture_output=capture_output)
    return {
        "scope": scope or "",
        "label": label or "",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout if capture_output else "",
        "stderr": completed.stderr if capture_output else "",
    }


def _run_runner_command(
    command: list[str],
    *,
    runner: Runner,
    capture_output: bool,
) -> dict[str, Any]:
    completed = runner(command, check=False, text=True, capture_output=capture_output)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout if capture_output else "",
        "stderr": completed.stderr if capture_output else "",
    }
