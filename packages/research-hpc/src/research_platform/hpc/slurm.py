
"""SLURM helpers for the minimal BIDS/HPC execution slice."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import re
import shlex
from typing import Any

_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SLURM_DIRECTIVE_FIELDS = {
    "account": "--account",
    "partition": "--partition",
    "qos": "--qos",
    "nodes": "--nodes",
    "ntasks": "--ntasks",
    "ntasks_per_node": "--ntasks-per-node",
    "mem_per_cpu": "--mem-per-cpu",
    "constraint": "--constraint",
    "export": "--export",
}


def normalize_jobspec(raw: dict[str, Any]) -> dict[str, Any]:
    omit_mem_directive = _normalize_bool(raw.get("omit_mem_directive", False))
    mem = str(raw.get("mem", "4G"))
    jobspec = {
        "job_name": str(raw.get("job_name", "rp-preprocess-bids")),
        "cpus": int(raw.get("cpus", 1)),
        "mem": mem,
        "mem_directive": "" if omit_mem_directive else f"#SBATCH --mem={mem}",
        "omit_mem_directive": omit_mem_directive,
        "time": str(raw.get("time", "01:00:00")),
        "log_out": str(raw.get("log_out", "artifacts/logs/slurm.out")),
        "log_err": str(raw.get("log_err", "artifacts/logs/slurm.err")),
        "command": str(raw.get("command", "true")),
    }
    for field in _SLURM_DIRECTIVE_FIELDS:
        value = _normalize_optional_directive_value(raw.get(field))
        if value is not None:
            jobspec[field] = value
    jobspec["optional_directives"] = _render_optional_directives(jobspec)
    return jobspec


def build_slurm_jobspec(
    *,
    resources: Mapping[str, Any],
    job_name: str,
    time: str,
    log_out: str,
    log_err: str,
    command: str,
    slurm_site: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    optional_directives = {
        field: slurm_site[field]
        for field in _SLURM_DIRECTIVE_FIELDS
        if isinstance(slurm_site, Mapping) and _normalize_optional_directive_value(slurm_site.get(field)) is not None
    }
    return normalize_jobspec(
        {
            "job_name": job_name,
            "cpus": int(resources.get("cpus", 1)),
            "mem": f"{int(resources.get('ram_gb', 4))}G",
            "time": time,
            "log_out": log_out,
            "log_err": log_err,
            "command": command,
            "omit_mem_directive": (
                slurm_site.get("omit_mem_directive", False) if isinstance(slurm_site, Mapping) else False
            ),
            **optional_directives,
        }
    )


def build_slurm_setup_commands(
    *,
    remote_workspace_root: str | None = None,
    modules: Any = None,
    environment: Any = None,
    pre_activate_commands: Any = None,
    prepare_directories: Any = None,
    bootstrap_command: str | None = None,
    activate_command: str | None = None,
) -> list[str]:
    commands: list[str] = []
    if remote_workspace_root:
        commands.append(f"cd {shlex.quote(str(remote_workspace_root))}")
    module_command = _build_module_load_command(modules)
    if module_command is not None:
        commands.append(module_command)
    commands.extend(_build_environment_export_commands(environment))
    commands.extend(normalize_shell_commands(pre_activate_commands))
    prepare_directories_command = _build_prepare_directories_command(prepare_directories)
    if prepare_directories_command is not None:
        commands.append(prepare_directories_command)
    if bootstrap_command:
        commands.append(bootstrap_command)
    if activate_command:
        commands.append(activate_command)
    return commands


def build_slurm_command_script(
    *,
    setup_commands: list[str],
    workflow_command: str,
    required_executables: Any = None,
) -> str:
    commands = [command.strip() for command in setup_commands if command and command.strip()]
    commands.extend(build_slurm_executable_guard_commands(required_executables=required_executables))
    normalized_workflow_command = workflow_command.strip()
    if normalized_workflow_command:
        commands.append(normalized_workflow_command)
    return "; ".join(commands)


def render_slurm_script(*, template_path: str | Path, jobspec: dict[str, Any]) -> str:
    text = Path(template_path).read_text(encoding="utf-8")
    rendered = text
    for key, value in normalize_jobspec(jobspec).items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
    return rendered


def normalize_slurm_batch_script(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    stripped = normalized.lstrip()
    if stripped.startswith("#!"):
        normalized = stripped
    elif not normalized.startswith("#!"):
        normalized = "#!/usr/bin/env bash\n" + normalized.lstrip("\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def write_slurm_script(path: str | Path, content: str) -> Path:
    script_path = Path(path)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(normalize_slurm_batch_script(content), encoding="utf-8")
    return script_path


def normalize_modules(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, (list, tuple)):
        modules: list[str] = []
        for item in value:
            normalized = str(item).strip()
            if normalized:
                modules.append(normalized)
        return modules
    normalized = str(value).strip()
    return [normalized] if normalized else []


def normalize_shell_commands(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, (list, tuple)):
        commands: list[str] = []
        for item in value:
            normalized = str(item).strip()
            if normalized:
                commands.append(normalized)
        return commands
    normalized = str(value).strip()
    return [normalized] if normalized else []


def normalize_prepare_directories(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        directories: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("SLURM prepare_directories entries must be strings.")
            normalized = item.strip()
            if not normalized:
                raise ValueError("SLURM prepare_directories entries must not be empty.")
            directories.append(normalized)
        return directories
    raise ValueError("SLURM prepare_directories must contain a list of non-empty strings when declared.")


def normalize_environment(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("SLURM environment must contain a mapping when declared.")

    environment: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name).strip()
        if not _ENVIRONMENT_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid SLURM environment variable name: {raw_name}")
        if not isinstance(raw_value, str):
            raise ValueError(f"SLURM environment value for {name} must be a string.")
        normalized_value = raw_value.strip()
        if not normalized_value:
            raise ValueError(f"SLURM environment value for {name} must not be empty.")
        environment[name] = normalized_value
    return environment


def build_slurm_executable_guard_commands(*, required_executables: Any = None) -> list[str]:
    guards: list[str] = []
    for executable in normalize_modules(required_executables):
        guards.append(
            " ".join(
                [
                    f"command -v {shlex.quote(executable)} >/dev/null 2>&1",
                    "||",
                    "{",
                    "printf '%s\\n'",
                    shlex.quote(
                        (
                            f"ERROR: Required executable '{executable}' was not found after SLURM environment setup. "
                            "Check compute.slurm.modules, compute.slurm.pre_activate_commands, and repo bootstrap dependencies."
                        )
                    ),
                    ">&2;",
                    "exit 127;",
                    "}",
                ]
            )
        )
    return guards


def _build_module_load_command(value: Any) -> str | None:
    modules = normalize_modules(value)
    if not modules:
        return None
    return f"module load {' '.join(shlex.quote(module) for module in modules)}"


def _build_environment_export_commands(value: Any) -> list[str]:
    commands: list[str] = []
    for name, raw_value in normalize_environment(value).items():
        escaped_value = raw_value.replace("\\", "\\\\").replace('"', '\\"')
        commands.append(f'export {name}="{escaped_value}"')
    return commands


def _build_prepare_directories_command(value: Any) -> str | None:
    directories = normalize_prepare_directories(value)
    if not directories:
        return None
    return "mkdir -p " + " ".join(_quote_shell_expandable_path(directory) for directory in directories)


def _quote_shell_expandable_path(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _normalize_optional_directive_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _render_optional_directives(jobspec: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for field, directive in _SLURM_DIRECTIVE_FIELDS.items():
        value = _normalize_optional_directive_value(jobspec.get(field))
        if value is None:
            continue
        lines.append(f"#SBATCH {directive}={value}")
    return "\n".join(lines)
