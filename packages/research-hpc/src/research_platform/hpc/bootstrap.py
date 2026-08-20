"""Remote bootstrap planning and explicit execution helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
from typing import Any, Callable

from ._yaml import write_yaml
from .connection import ResolvedHpcConnection, resolve_hpc_connection
from .ssh import build_ssh_command

Runner = Callable[..., subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]]


def build_bootstrap_execution_plan(
    *,
    run_root: str | Path,
    manifest: dict[str, Any],
    status: dict[str, Any],
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    run_root_path = Path(run_root)
    bootstrap = manifest.get("bootstrap", {})
    enabled = bool(bootstrap.get("enabled"))
    connection: ResolvedHpcConnection | None = None
    connection_error = ""
    if enabled:
        try:
            connection = resolve_hpc_connection(manifest=manifest, workspace_root=workspace_root)
        except ValueError as exc:
            connection_error = str(exc)

    directory_commands = _build_directory_commands(connection=connection, bootstrap=bootstrap)
    hook_commands = _build_hook_commands(connection=connection, bootstrap=bootstrap)
    executable = enabled and connection is not None and bool(directory_commands or hook_commands)
    plan_path = run_root_path / "hpc" / "bootstrap-plan.yaml"
    report = {
        "run_id": manifest["run_id"],
        "enabled": enabled,
        "ssh_host": manifest.get("hpc", {}).get("ssh_host", ""),
        "connection": _render_connection_report(connection),
        "connection_error": connection_error,
        "directory_commands": directory_commands,
        "hook_commands": hook_commands,
        "command_count": len(directory_commands) + len(hook_commands),
        "executable": executable,
        "local_files_written": [str(plan_path)],
    }
    write_yaml(plan_path, report)
    next_status = dict(status)
    if enabled:
        next_status |= {"state": "bootstrap-planned", "last_updated": _timestamp()}
    return {"report": report, "status": next_status}


def execute_bootstrap_plan(plan: dict[str, Any], *, runner: Runner = subprocess.run) -> dict[str, Any]:
    if not plan.get("enabled"):
        return {"ok": True, "executed": False, "returncode": 0, "steps": []}
    if not plan.get("executable"):
        return {"ok": False, "executed": False, "returncode": 1, "steps": [], "error": "Bootstrap plan is not executable."}

    steps: list[dict[str, Any]] = []
    for entry in [*plan.get("directory_commands", []), *plan.get("hook_commands", [])]:
        completed = _run_ssh_command(entry["command"], runner=runner)
        step = {
            "name": entry["name"],
            "scope": entry.get("scope", ""),
            "kind": entry["kind"],
            "returncode": completed.returncode,
        }
        steps.append(step)
        if completed.returncode != 0:
            return {"ok": False, "executed": True, "returncode": completed.returncode, "steps": steps}
    return {"ok": True, "executed": True, "returncode": 0, "steps": steps}


def _build_directory_commands(*, connection: ResolvedHpcConnection | None, bootstrap: dict[str, Any]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for entry in bootstrap.get("remote_directories", []):
        path = str(entry.get("path", "")).strip()
        if not path:
            continue
        remote_command = f"mkdir -p {shlex.quote(path)}"
        commands.append(
            {
                "name": str(entry.get("label", "remote-directory")).strip() or "remote-directory",
                "scope": "common",
                "kind": "directory-preparation",
                "path": path,
                "remote_command": remote_command,
                "command": _build_remote_command(connection, remote_command),
            }
        )
    return commands


def _build_hook_commands(*, connection: ResolvedHpcConnection | None, bootstrap: dict[str, Any]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for scope in bootstrap.get("hook_scopes", []):
        scope_name = str(scope.get("name", "")).strip()
        for hook in scope.get("hooks", []):
            remote_command = str(hook.get("command", "")).strip()
            if not remote_command:
                continue
            commands.append(
                {
                    "name": str(hook.get("name", "bootstrap-hook")).strip() or "bootstrap-hook",
                    "scope": scope_name,
                    "kind": str(hook.get("kind", "shell")).strip() or "shell",
                    "remote_command": remote_command,
                    "command": _build_remote_command(connection, remote_command),
                }
            )
    return commands


def _build_remote_command(connection: ResolvedHpcConnection | None, remote_command: str) -> list[str]:
    if connection is None:
        return []
    if connection.profile is not None:
        return build_ssh_command(connection.profile, mode=connection.mode, remote_command=remote_command, allocate_tty=False)
    return ["ssh", connection.ssh_target, remote_command]


def _render_connection_report(connection: ResolvedHpcConnection | None) -> dict[str, str]:
    if connection is None:
        return {}
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


def _run_ssh_command(command: list[str], *, runner: Runner) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    return runner(command, capture_output=True, text=True, check=False)


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
