"""Pure, offline validation for local HPC target and SSH configuration.

This module intentionally contains no subprocess, socket, DNS, scheduler, or
remote-filesystem integration.  It validates local configuration documents and
returns a report; callers decide how to display that report.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
from typing import Any

from ._yaml import expand_env_placeholders, load_yaml
from .ssh_profiles import (
    validate_ssh_host_or_alias,
    validate_ssh_options,
    validate_ssh_path_reference,
    validate_ssh_port,
    validate_ssh_profile_name,
    validate_ssh_user,
)

_TARGET_DOCUMENT_KEYS = frozenset({"version", "default", "targets"})
_TARGET_KEYS = frozenset(
    {
        "ssh_profile",
        "role",
        "ssh_config",
        "env",
        "slurm",
        "projects",
        "promotion",
    }
)
_PROJECT_OVERRIDE_KEYS = frozenset({"env"})
_PROMOTION_KEYS = frozenset({"mode"})
_SSH_DOCUMENT_KEYS = frozenset({"defaults", "profiles"})
_SSH_PROFILE_KEYS = frozenset(
    {
        "host",
        "user",
        "port",
        "ssh_config_host",
        "identity_file",
        "known_hosts_file",
        "options",
        "defaults",
        "roles",
    }
)
_SSH_CONNECTION_KEYS = frozenset(
    {
        "host",
        "user",
        "port",
        "ssh_config_host",
        "identity_file",
        "known_hosts_file",
        "options",
    }
)
_SLURM_KEYS = frozenset(
    {
        "modules",
        "environment",
        "pre_activate_commands",
        "prepare_directories",
        "omit_mem_directive",
        "account",
        "partition",
        "qos",
        "nodes",
        "ntasks",
        "ntasks_per_node",
        "mem_per_cpu",
        "constraint",
        "export",
    }
)
_SLURM_LIST_KEYS = frozenset({"modules", "pre_activate_commands", "prepare_directories"})
_SLURM_POSITIVE_INTEGER_KEYS = frozenset({"nodes", "ntasks", "ntasks_per_node"})
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ANGLE_PLACEHOLDER_PATTERN = re.compile(r"<[^>\n]+>")
_ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{[^}\n]+\}")
_PRODUCTION_ENV_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}"
)
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_REMOTE_PATH_PATTERN = re.compile(r"""[;&|`$><#"'\\*?\[\](){}!]""")
_REMOTE_ROOT_ENV_NAMES = frozenset(
    {
        "RP_REMOTE_WORKSPACE_ROOT",
        "RP_REMOTE_ARTIFACTS_ROOT",
        "RP_REMOTE_CONTAINER_ROOT",
        "RP_REMOTE_PROJECT_DATA_ROOT",
        "RP_REMOTE_WORK_ROOT",
        "RP_REMOTE_CACHE_ROOT",
        "RP_REMOTE_TMP_ROOT",
        "RP_REMOTE_TEMP_ROOT",
        "TMPDIR",
        "SCRATCH",
        "APPTAINER_CACHEDIR",
        "APPTAINER_TMPDIR",
    }
)
_EXCLUSIVE_REMOTE_ROOT_SEMANTICS = frozenset(
    {
        "workspace",
        "artifacts",
        "container",
        "work",
        "cache",
        "temporary",
    }
)
_NOT_REMOTELY_CHECKED = (
    "hostname reachability",
    "authentication or MFA",
    "scheduler availability",
    "account or partition authorization",
    "remote path existence or permissions",
    "filesystem promotion capability",
    "runtime readiness",
    "installed software",
    "storage quota",
    "data readiness",
)
_LOCAL_SELECTION_ENV_NAMES = frozenset(
    {
        "ALLIANCE_USER",
        "RESEARCH_HPC_TARGET",
        "RP_HPC_TARGET",
        "RESEARCH_HPC_TARGETS_CONFIG",
        "RP_HPC_TARGETS_CONFIG",
        "RESEARCH_HPC_SSH_CONFIG",
        "RP_SSH_CONFIG",
        "RESEARCH_HPC_PROFILE",
        "RP_HPC_PROFILE",
        "RESEARCH_HPC_ROLE",
        "RP_HPC_ROLE",
        "RP_REMOTE_WORKSPACE_ROOT",
        "RP_REMOTE_ARTIFACTS_ROOT",
    }
)


def resolve_local_hpc_environment_defaults(
    *,
    workspace_root: str | Path,
    environment: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return effective and locally managed HPC defaults without mutation.

    Only the small documented HPC selection/root allowlist is read from the
    ignored ``secrets/.env`` file. Existing process values retain precedence.
    The second result records only values that production would mark as
    managed after loading that file, so later target defaults can replace them
    without replacing an unmanaged process value.
    """

    effective = dict(os.environ if environment is None else environment)
    managed: dict[str, str] = {}
    env_path = Path(workspace_root).expanduser().resolve() / "secrets" / ".env"
    unsafe_parent = _first_unsafe_local_parent(env_path)
    if unsafe_parent is not None:
        raise ValueError(
            f"Local HPC defaults parent must be a real directory, not a symlink or special entry: "
            f"{unsafe_parent}"
        )
    try:
        file_stat = env_path.lstat()
    except FileNotFoundError:
        return effective, managed
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(
            f"Local HPC defaults must be a real regular file, not a symlink or special file: {env_path}"
        )
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        raw_name, separator, raw_value = line.partition("=")
        name = raw_name.strip()
        if not separator or name not in _LOCAL_SELECTION_ENV_NAMES or name in effective:
            continue
        value = raw_value.strip()
        effective[name] = value
        managed[name] = value
    return effective, managed


def read_local_hpc_environment_defaults(
    *,
    workspace_root: str | Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return effective local HPC defaults without mutating ``os.environ``."""

    effective, _ = resolve_local_hpc_environment_defaults(
        workspace_root=workspace_root,
        environment=environment,
    )
    return effective


def _expand_target_document_like_production(
    value: Any,
    *,
    environment: Mapping[str, str],
) -> Any:
    """Mirror the single parse-time expansion used by core target loading."""

    if isinstance(value, str):
        return _PRODUCTION_ENV_PATTERN.sub(
            lambda match: environment.get(match.group(1), match.group(3) or ""),
            value,
        )
    if isinstance(value, list):
        return [
            _expand_target_document_like_production(item, environment=environment)
            for item in value
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _expand_target_document_like_production(
                item,
                environment=environment,
            )
            for key, item in value.items()
        }
    return value


def _apply_managed_environment_defaults(
    environment: Mapping[str, str],
    defaults: Mapping[str, str],
    *,
    managed_defaults: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Pure equivalent of core's managed target-default application."""

    effective = dict(environment)
    managed = dict(managed_defaults)
    for raw_name, raw_value in defaults.items():
        name = str(raw_name).strip()
        if not _ENVIRONMENT_NAME_PATTERN.fullmatch(name):
            continue
        value = str(raw_value).strip()
        current = effective.get(name)
        if current is not None and managed.get(name) != current:
            continue
        effective[name] = value
        managed[name] = value
    return effective, managed


def _target_environment_defaults(
    target: Mapping[str, Any] | None,
    *,
    project_name: str | None,
) -> dict[str, str]:
    """Collect the values production attempts to apply for one target pass."""

    if not isinstance(target, Mapping):
        return {}
    defaults: dict[str, str] = {}
    for field, name in (
        ("ssh_profile", "RESEARCH_HPC_PROFILE"),
        ("role", "RESEARCH_HPC_ROLE"),
        ("ssh_config", "RESEARCH_HPC_SSH_CONFIG"),
    ):
        value = _optional_text(target.get(field))
        if value is not None:
            defaults[name] = value
    target_environment = target.get("env")
    if isinstance(target_environment, Mapping):
        for raw_name, raw_value in target_environment.items():
            name = str(raw_name).strip()
            if (
                _ENVIRONMENT_NAME_PATTERN.fullmatch(name)
                and isinstance(raw_value, str)
                and raw_value.strip()
            ):
                defaults[name] = raw_value.strip()
    projects = target.get("projects")
    if project_name is not None and isinstance(projects, Mapping):
        project = projects.get(project_name)
        if isinstance(project, Mapping):
            project_environment = project.get("env")
            if isinstance(project_environment, Mapping):
                for raw_name, raw_value in project_environment.items():
                    name = str(raw_name).strip()
                    if (
                        _ENVIRONMENT_NAME_PATTERN.fullmatch(name)
                        and isinstance(raw_value, str)
                        and raw_value.strip()
                    ):
                        defaults[name] = raw_value.strip()
    return defaults


def _resolve_target_environment_like_production(
    raw_document: Mapping[str, Any],
    *,
    target_name: str | None,
    project_name: str | None,
    environment: Mapping[str, str],
    managed_defaults: Mapping[str, str],
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None, dict[str, str]]:
    """Resolve target interpolation/default precedence without global mutation.

    Production applies a target once before project dispatch. Project-aware
    handlers then load the target document again and apply its project
    override. Repeating that sequence matters because each target-document
    parse expands placeholders against the environment from the preceding
    pass.
    """

    baseline = dict(environment)
    managed = dict(managed_defaults)
    first_document = _expand_target_document_like_production(
        raw_document,
        environment=baseline,
    )
    if not isinstance(first_document, dict):
        return {}, None, None, baseline
    first_targets = first_document.get("targets")
    first_targets = first_targets if isinstance(first_targets, dict) else {}
    selected_name = (
        _optional_text(target_name)
        or _first_environment_value(
            baseline,
            ("RESEARCH_HPC_TARGET", "RP_HPC_TARGET"),
        )
        or _optional_text(first_document.get("default"))
    )
    first_target = first_targets.get(selected_name) if selected_name is not None else None
    first_target = first_target if isinstance(first_target, dict) else None
    effective, managed = _apply_managed_environment_defaults(
        baseline,
        _target_environment_defaults(first_target, project_name=None),
        managed_defaults=managed,
    )

    if project_name is None:
        return first_document, selected_name, first_target, effective

    final_document = _expand_target_document_like_production(
        raw_document,
        environment=effective,
    )
    if not isinstance(final_document, dict):
        return {}, selected_name, None, effective
    final_targets = final_document.get("targets")
    final_targets = final_targets if isinstance(final_targets, dict) else {}
    selected_name = (
        _optional_text(target_name)
        or _first_environment_value(
            effective,
            ("RESEARCH_HPC_TARGET", "RP_HPC_TARGET"),
        )
        or _optional_text(final_document.get("default"))
    )
    final_target = final_targets.get(selected_name) if selected_name is not None else None
    final_target = final_target if isinstance(final_target, dict) else None
    effective, _ = _apply_managed_environment_defaults(
        effective,
        _target_environment_defaults(final_target, project_name=project_name),
        managed_defaults=managed,
    )
    return final_document, selected_name, final_target, effective


def validate_hpc_configuration(
    *,
    workspace_root: str | Path,
    targets_config_path: str | Path,
    target_name: str | None = None,
    ssh_config_path: str | Path | None = None,
    profile_name: str | None = None,
    role: str | None = None,
    project_name: str | None = None,
    project_root: str | Path | None = None,
    targets_document: Mapping[str, Any] | None = None,
    ssh_document: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    managed_environment_defaults: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate one effective HPC target without writes or remote activity.

    ``targets_document`` and ``ssh_document`` allow setup code to validate a
    complete proposed state before either bounded local configuration write.
    Paths are still required so relative references have deterministic anchors.
    """

    root = Path(workspace_root).expanduser().resolve()
    targets_path = _resolve_local_path(targets_config_path, base=root)
    env = dict(os.environ if environment is None else environment)
    errors: list[str] = []
    warnings: list[str] = []
    notices = [
        "Offline validation writes nothing and contacts no remote system.",
        "Local configuration validity is not live provider readiness.",
    ]

    raw_targets = _load_document(
        path=targets_path,
        supplied=targets_document,
        label="HPC targets config",
        errors=errors,
    )
    selected_target_name: str | None = None
    target: dict[str, Any] | None = None
    raw_ssh_path: str | None = None
    effective_environment = dict(env)

    if raw_targets is not None:
        (
            raw_targets,
            selected_target_name,
            target,
            effective_environment,
        ) = _resolve_target_environment_like_production(
            raw_targets,
            target_name=target_name,
            project_name=project_name,
            environment=env,
            managed_defaults=managed_environment_defaults or {},
        )
        _reject_unknown_keys(raw_targets, _TARGET_DOCUMENT_KEYS, "HPC targets config", errors)
        version = raw_targets.get("version")
        if isinstance(version, bool) or version != 1:
            errors.append("HPC targets config version must be the supported integer schema version 1.")

        targets = raw_targets.get("targets")
        if not isinstance(targets, dict):
            errors.append("HPC targets config targets must contain a mapping.")
            targets = {}
        else:
            for candidate_name, candidate_target in targets.items():
                if not isinstance(candidate_name, str) or not _NAME_PATTERN.fullmatch(candidate_name):
                    errors.append(f"HPC target name {candidate_name!r} is invalid.")
                if not isinstance(candidate_target, dict):
                    errors.append(f"HPC target {candidate_name!r} must contain a mapping.")
                    continue
                _reject_unknown_keys(
                    candidate_target,
                    _TARGET_KEYS,
                    f"HPC target {candidate_name!r}",
                    errors,
                )
                _validate_slurm_document_shape(
                    candidate_target.get("slurm"),
                    label=f"HPC target {candidate_name!r} slurm",
                    errors=errors,
                )
                projects = candidate_target.get("projects", {})
                if projects is not None and not isinstance(projects, dict):
                    errors.append(f"HPC target {candidate_name!r} projects must contain a mapping.")
                elif isinstance(projects, dict):
                    for candidate_project, override in projects.items():
                        if not isinstance(candidate_project, str) or not _NAME_PATTERN.fullmatch(candidate_project):
                            errors.append(
                                f"HPC target {candidate_name!r} project override name "
                                f"{candidate_project!r} is invalid."
                            )
                        if not isinstance(override, dict):
                            errors.append(
                                f"HPC target {candidate_name!r} project override "
                                f"{candidate_project!r} must contain a mapping."
                            )
                            continue
                        _reject_unknown_keys(
                            override,
                            _PROJECT_OVERRIDE_KEYS,
                            (
                                f"HPC target {candidate_name!r} project override "
                                f"{candidate_project!r}"
                            ),
                            errors,
                        )

        default_target = _optional_text(raw_targets.get("default"))
        if raw_targets.get("default") is not None and default_target is None:
            errors.append("HPC targets config default must be a nonblank target name.")
        if default_target is not None and default_target not in targets:
            errors.append(
                f"HPC targets config default {default_target!r} does not refer to a declared target."
            )

        if selected_target_name is None:
            errors.append("Select an HPC target with --target or declare a valid targets default.")
        elif selected_target_name not in targets:
            errors.append(f"Selected HPC target {selected_target_name!r} is not declared.")
        elif isinstance(targets[selected_target_name], dict):
            target = deepcopy(targets[selected_target_name])

        for candidate_name, candidate_target in targets.items():
            if not isinstance(candidate_target, dict):
                continue
            if candidate_name != selected_target_name:
                _validate_environment_document_shape(
                    candidate_target.get("env"),
                    label=f"HPC target {candidate_name!r} env",
                    errors=errors,
                )
                if "promotion" in candidate_target:
                    _validate_promotion(
                        candidate_target.get("promotion"),
                        label=f"HPC target {candidate_name!r} promotion",
                        errors=errors,
                    )
            projects = candidate_target.get("projects")
            if not isinstance(projects, dict):
                continue
            for candidate_project, override in projects.items():
                if not isinstance(override, dict):
                    continue
                if (
                    candidate_name == selected_target_name
                    and candidate_project == project_name
                ):
                    continue
                _validate_environment_document_shape(
                    override.get("env"),
                    label=(
                        f"HPC target {candidate_name!r} project override "
                        f"{candidate_project!r} env"
                    ),
                    errors=errors,
                )

    selected_profile: str | None = None
    selected_role: str | None = None
    resolved_ssh_path: Path | None = None
    root_records: list[tuple[str, str, str]] = []

    if target is not None and selected_target_name is not None:
        declared_profile = _required_text(
            target.get("ssh_profile"),
            label=f"HPC target {selected_target_name!r} ssh_profile",
            errors=errors,
        )
        raw_ssh_path = _required_text(
            target.get("ssh_config"),
            label=f"HPC target {selected_target_name!r} ssh_config",
            errors=errors,
        )
        if raw_ssh_path is not None:
            _validate_nonplaceholder_text(
                raw_ssh_path,
                label=f"HPC target {selected_target_name!r} ssh_config",
                errors=errors,
            )
        declared_role: str | None = None
        if "role" in target:
            declared_role = _required_text(
                target.get("role"),
                label=f"HPC target {selected_target_name!r} role",
                errors=errors,
            )
            if declared_role is not None and not _valid_name(declared_role):
                errors.append(
                    f"HPC target {selected_target_name!r} role "
                    f"{declared_role!r} is invalid."
                )
        selected_profile = (
            _optional_text(profile_name)
            or _first_environment_value(
                effective_environment,
                ("RESEARCH_HPC_PROFILE", "RP_HPC_PROFILE"),
            )
            or declared_profile
        )
        selected_role = (
            _optional_text(role)
            or _first_environment_value(
                effective_environment,
                ("RESEARCH_HPC_ROLE", "RP_HPC_ROLE"),
            )
            or declared_role
            or "login"
        )
        if selected_profile is None:
            errors.append("The selected target does not resolve an SSH profile.")
        if not _valid_name(selected_role):
            errors.append(f"Selected SSH role {selected_role!r} is invalid.")

        env_mapping = _validate_environment_mapping(
            target.get("env"),
            label=f"HPC target {selected_target_name!r} env",
            environment=effective_environment,
            expand=False,
            errors=errors,
        )
        _collect_environment_roots(
            env_mapping,
            label=f"targets.{selected_target_name}.env",
            records=root_records,
            errors=errors,
        )
        if "RP_REMOTE_WORKSPACE_ROOT" not in env_mapping:
            errors.append(
                f"HPC target {selected_target_name!r} must declare "
                "env.RP_REMOTE_WORKSPACE_ROOT."
            )

        project_env: dict[str, str] = {}
        project_overrides = target.get("projects", {})
        if project_name is not None and isinstance(project_overrides, dict):
            override = project_overrides.get(project_name)
            if override is not None:
                if not isinstance(override, dict):
                    errors.append(
                        f"HPC target {selected_target_name!r} project override "
                        f"{project_name!r} must contain a mapping."
                    )
                else:
                    project_env = _validate_environment_mapping(
                        override.get("env"),
                        label=(
                            f"HPC target {selected_target_name!r} project override "
                            f"{project_name!r} env"
                        ),
                        environment=effective_environment,
                        expand=False,
                        errors=errors,
                    )
                    _collect_environment_roots(
                        project_env,
                        label=f"targets.{selected_target_name}.projects.{project_name}.env",
                        records=root_records,
                        errors=errors,
                    )

        effective_root_names = set(env_mapping) | set(project_env)
        effective_root_names.update(
            name
            for name in effective_environment
            if name.startswith("RP_REMOTE_")
        )
        _collect_environment_roots(
            {
                name: effective_environment.get(name, "")
                for name in effective_root_names
            },
            label="effective environment",
            records=root_records,
            errors=errors,
        )
        _validate_slurm(
            target.get("slurm"),
            label=f"HPC target {selected_target_name!r} slurm",
            environment=effective_environment,
            expand=False,
            root_records=root_records,
            errors=errors,
        )
        _validate_promotion(
            target.get("promotion"),
            label=f"HPC target {selected_target_name!r} promotion",
            errors=errors,
        )

        selected_ssh_path = _optional_text(ssh_config_path) or _first_environment_value(
            effective_environment,
            ("RESEARCH_HPC_SSH_CONFIG", "RP_SSH_CONFIG"),
        )
        if selected_ssh_path is None:
            selected_ssh_path = raw_ssh_path
        if selected_ssh_path is not None:
            resolved_ssh_path = _resolve_local_path(selected_ssh_path, base=root)

    raw_ssh = None
    if resolved_ssh_path is not None:
        raw_ssh = _load_document(
            path=resolved_ssh_path,
            supplied=ssh_document,
            label="SSH profile config",
            errors=errors,
        )
    elif target is not None:
        errors.append("The selected target does not resolve an SSH profile config path.")

    if raw_ssh is not None:
        _validate_ssh_document(
            raw_ssh,
            profile_name=selected_profile,
            role=selected_role,
            config_path=resolved_ssh_path,
            environment=effective_environment,
            errors=errors,
        )

    resolved_project_root: Path | None = None
    if project_name is not None:
        if not _valid_name(project_name):
            errors.append(f"Project name {project_name!r} is invalid.")
        elif project_root is None:
            errors.append("A project root is required when --project is supplied.")
        else:
            resolved_project_root = _resolve_local_path(project_root, base=root)
            _validate_project(
                project_name=project_name,
                project_root=resolved_project_root,
                environment=effective_environment,
                root_records=root_records,
                errors=errors,
            )

    _validate_root_conflicts(root_records, errors)

    report = {
        "configuration_valid": not errors,
        "offline": True,
        "network_contacted": False,
        "target": selected_target_name,
        "profile": selected_profile,
        "role": selected_role,
        "configuration_paths": {
            "targets": str(targets_path),
            "ssh": str(resolved_ssh_path) if resolved_ssh_path is not None else None,
            "local_defaults": str(root / "secrets" / ".env"),
            "project": str(resolved_project_root) if resolved_project_root is not None else None,
        },
        "project": project_name,
        "promotion_policy": {
            "mode": (
                target.get("promotion", {}).get("mode")
                if isinstance(target, dict) and isinstance(target.get("promotion"), dict)
                else None
            ),
            "verification": "declared, not remotely verified",
        },
        "errors": errors,
        "warnings": warnings,
        "notices": notices,
        "not_remotely_checked": list(_NOT_REMOTELY_CHECKED),
    }
    return report


def render_hpc_validation_report(report: Mapping[str, Any]) -> str:
    """Render the offline report without exposing host, user, or secret data."""

    valid = bool(report.get("configuration_valid"))
    paths = report.get("configuration_paths")
    paths = paths if isinstance(paths, Mapping) else {}
    promotion = report.get("promotion_policy")
    promotion = promotion if isinstance(promotion, Mapping) else {}
    lines = [
        "HPC offline configuration validation",
        f"Configuration valid: {'yes' if valid else 'no'}",
        "Offline: yes",
        "Network contacted: no",
        f"Target: {report.get('target') or 'not selected'}",
        f"Profile: {report.get('profile') or 'not selected'}",
        f"Role: {report.get('role') or 'not selected'}",
        f"Targets config: {paths.get('targets') or 'not resolved'}",
        f"SSH config: {paths.get('ssh') or 'not resolved'}",
        f"Local defaults: {paths.get('local_defaults') or 'not resolved'}",
        f"Project: {report.get('project') or 'not requested'}",
        f"Project config root: {paths.get('project') or 'not requested'}",
        (
            "Promotion policy: "
            f"{promotion.get('mode') or 'not declared'} "
            f"({promotion.get('verification') or 'unverified'})"
        ),
    ]
    for heading, key in (("Errors", "errors"), ("Warnings", "warnings"), ("Notices", "notices")):
        values = report.get(key)
        if isinstance(values, list) and values:
            lines.extend(["", heading])
            lines.extend(f"- {value}" for value in values)
    lines.extend(["", "Not remotely checked"])
    lines.extend(f"- {value}" for value in report.get("not_remotely_checked", []))
    return "\n".join(lines)


def _load_document(
    *,
    path: Path,
    supplied: Mapping[str, Any] | None,
    label: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            errors.append(f"{label} must contain a top-level mapping.")
            return None
        return deepcopy(dict(supplied))

    unsafe_parent = _first_unsafe_local_parent(path)
    if unsafe_parent is not None:
        errors.append(
            f"{label} parent must be a real directory, not a symlink or special entry: "
            f"{unsafe_parent}"
        )
        return None
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        errors.append(f"{label} was not found: {path}")
        return None
    except (OSError, ValueError) as exc:
        errors.append(f"{label} could not be inspected safely: {exc}")
        return None
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        errors.append(f"{label} must be a real regular file, not a symlink or special file: {path}")
        return None
    try:
        document = load_yaml(path)
    except (OSError, ValueError) as exc:
        errors.append(f"{label} could not be parsed: {exc}")
        return None
    if not isinstance(document, dict):
        errors.append(f"{label} must contain a top-level mapping.")
        return None
    return document


def _validate_ssh_document(
    document: dict[str, Any],
    *,
    profile_name: str | None,
    role: str | None,
    config_path: Path | None,
    environment: Mapping[str, str],
    errors: list[str],
) -> None:
    _reject_unknown_keys(document, _SSH_DOCUMENT_KEYS, "SSH profile config", errors)
    defaults = document.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        errors.append("SSH profile config defaults must contain a mapping.")
        defaults = {}
    else:
        _reject_unknown_keys(defaults, _SSH_CONNECTION_KEYS, "SSH profile config defaults", errors)

    profiles = document.get("profiles")
    if not isinstance(profiles, dict):
        errors.append("SSH profile config profiles must contain a mapping.")
        return
    for name, entry in profiles.items():
        try:
            validate_ssh_profile_name(name)
        except ValueError as exc:
            errors.append(str(exc))
        if not isinstance(entry, dict):
            errors.append(f"SSH profile {name!r} must contain a mapping.")
            continue
        _reject_unknown_keys(entry, _SSH_PROFILE_KEYS, f"SSH profile {name!r}", errors)
        profile_defaults = entry.get("defaults")
        if profile_defaults is not None:
            if not isinstance(profile_defaults, dict):
                errors.append(f"SSH profile {name!r} defaults must contain a mapping.")
            else:
                _reject_unknown_keys(
                    profile_defaults,
                    _SSH_CONNECTION_KEYS,
                    f"SSH profile {name!r} defaults",
                    errors,
                )
        roles = entry.get("roles")
        if roles is not None:
            if not isinstance(roles, dict):
                errors.append(f"SSH profile {name!r} roles must contain a mapping.")
            else:
                for role_name, role_entry in roles.items():
                    try:
                        validate_ssh_profile_name(role_name)
                    except ValueError as exc:
                        errors.append(
                            f"SSH profile {name!r} role name is invalid: {exc}"
                        )
                    if not isinstance(role_entry, dict):
                        errors.append(f"SSH profile {name!r} role {role_name!r} must contain a mapping.")
                    else:
                        _reject_unknown_keys(
                            role_entry,
                            _SSH_CONNECTION_KEYS,
                            f"SSH profile {name!r} role {role_name!r}",
                            errors,
                        )

    if profile_name is None:
        errors.append("No effective SSH profile was selected.")
        return
    entry = profiles.get(profile_name)
    if not isinstance(entry, dict):
        errors.append(f"Selected SSH profile {profile_name!r} is not declared.")
        return
    resolved_role = role or "login"
    roles = entry.get("roles")
    if roles is not None:
        if not isinstance(roles, dict) or resolved_role not in roles:
            errors.append(
                f"Selected SSH profile {profile_name!r} does not declare role {resolved_role!r}."
            )
            return
        role_entry = roles[resolved_role]
        if not isinstance(role_entry, dict):
            return
    elif resolved_role != "login":
        errors.append(
            f"Flat SSH profile {profile_name!r} supports the implicit login role only, "
            f"not {resolved_role!r}."
        )
        return
    else:
        role_entry = {}

    profile_defaults = entry.get("defaults", {})
    profile_defaults = profile_defaults if isinstance(profile_defaults, dict) else {}
    flat = {key: value for key, value in entry.items() if key in _SSH_CONNECTION_KEYS}
    payload = _merge_mappings(defaults, flat, profile_defaults, role_entry)
    expanded = _expand_environment_for_validation(
        payload,
        environment=environment,
        label=f"SSH profile {profile_name!r}",
        errors=errors,
    )
    if not isinstance(expanded, dict):
        errors.append(f"Selected SSH profile {profile_name!r} could not be resolved.")
        return
    _validate_effective_ssh_profile(
        expanded,
        profile_name=profile_name,
        role=resolved_role,
        config_path=config_path,
        errors=errors,
    )


def _expand_environment_for_validation(
    value: Any,
    *,
    environment: Mapping[str, str],
    label: str,
    errors: list[str],
) -> Any:
    try:
        return expand_env_placeholders(value, env=environment)
    except ValueError as exc:
        errors.append(f"{label} environment expansion failed: {exc}")
        return value


def _validate_effective_ssh_profile(
    payload: dict[str, Any],
    *,
    profile_name: str,
    role: str,
    config_path: Path | None,
    errors: list[str],
) -> None:
    host = _optional_text(payload.get("host"))
    alias = _optional_text(payload.get("ssh_config_host"))
    if (host is None) == (alias is None):
        errors.append(
            f"SSH profile {profile_name!r} role {role!r} must resolve exactly one "
            "of host or ssh_config_host."
        )
    for label, value in (("host", host), ("ssh_config_host", alias)):
        if value is None:
            continue
        try:
            validate_ssh_host_or_alias(
                value,
                label=f"SSH profile {profile_name!r} {label}",
            )
        except ValueError as exc:
            errors.append(str(exc))

    user = _required_text(
        payload.get("user"),
        label=f"SSH profile {profile_name!r} user",
        errors=errors,
    )
    if user is not None:
        try:
            validate_ssh_user(user)
        except ValueError as exc:
            errors.append(f"SSH profile {profile_name!r}: {exc}")

    port = payload.get("port")
    if port is not None:
        try:
            validate_ssh_port(port)
        except ValueError as exc:
            errors.append(f"SSH profile {profile_name!r}: {exc}")

    options = payload.get("options", {})
    if options is None:
        options = {}
    if not isinstance(options, dict):
        errors.append(f"SSH profile {profile_name!r} options must contain a mapping.")
    else:
        try:
            validate_ssh_options(options)
        except ValueError as exc:
            errors.append(f"SSH profile {profile_name!r}: {exc}")

    for field in ("identity_file", "known_hosts_file"):
        value = _optional_text(payload.get(field))
        if value is None:
            continue
        try:
            validate_ssh_path_reference(
                value,
                label=f"SSH profile {profile_name!r} {field}",
            )
        except ValueError as exc:
            errors.append(str(exc))


def _validate_environment_mapping(
    value: Any,
    *,
    label: str,
    environment: Mapping[str, str],
    expand: bool = True,
    errors: list[str],
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a mapping of environment names to strings.")
        return {}
    expanded = (
        _expand_environment_for_validation(
            value,
            environment=environment,
            label=label,
            errors=errors,
        )
        if expand
        else value
    )
    normalized: dict[str, str] = {}
    for raw_name, raw_value in expanded.items():
        name = str(raw_name)
        if not _ENVIRONMENT_NAME_PATTERN.fullmatch(name):
            errors.append(f"{label} contains invalid environment name {raw_name!r}.")
            continue
        if not isinstance(raw_value, str):
            errors.append(f"{label}.{name} must contain a string.")
            continue
        text = raw_value.strip()
        if not text:
            errors.append(f"{label}.{name} must contain a nonblank value.")
            continue
        _validate_nonplaceholder_text(text, label=f"{label}.{name}", errors=errors)
        normalized[name] = text
    return normalized


def _validate_environment_document_shape(
    value: Any,
    *,
    label: str,
    errors: list[str],
) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a mapping of environment names to strings.")
        return
    for raw_name, raw_value in value.items():
        name = str(raw_name)
        if not _ENVIRONMENT_NAME_PATTERN.fullmatch(name):
            errors.append(f"{label} contains invalid environment name {raw_name!r}.")
        if not isinstance(raw_value, str):
            errors.append(f"{label}.{name} must contain a string.")
        elif not raw_value.strip():
            errors.append(f"{label}.{name} must contain a nonblank value.")


def _collect_environment_roots(
    environment: Mapping[str, str],
    *,
    label: str,
    records: list[tuple[str, str, str]],
    errors: list[str],
) -> None:
    for name, value in environment.items():
        if name not in _REMOTE_ROOT_ENV_NAMES and not (
            name.endswith("_ROOT") and ("REMOTE" in name or name.startswith("RP_"))
        ):
            continue
        _validate_remote_root(
            value,
            label=f"{label}.{name}",
            semantic=_root_semantic(name),
            records=records,
            errors=errors,
        )


def _validate_remote_root(
    value: Any,
    *,
    label: str,
    semantic: str,
    records: list[tuple[str, str, str]],
    errors: list[str],
) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must contain a nonempty absolute POSIX path.")
        return
    text = value.strip()
    _validate_nonplaceholder_text(text, label=label, errors=errors)
    if _UNSAFE_REMOTE_PATH_PATTERN.search(text):
        errors.append(f"{label} contains an unsafe shell-control character.")
    if not text.startswith("/"):
        errors.append(f"{label} must be an absolute POSIX path.")
        return
    if text == "/":
        errors.append(f"{label} must not use the remote filesystem root '/'.")
        return
    components = text.split("/")
    if any(component in {".", ".."} for component in components):
        errors.append(f"{label} must not contain '.' or '..' traversal components.")
    normalized = posixpath.normpath(text)
    if normalized != text:
        errors.append(f"{label} must be normalized; expected {normalized!r}.")
    path = PurePosixPath(text)
    if not path.is_absolute():
        errors.append(f"{label} must be an absolute POSIX path.")
        return
    records.append((semantic, text, label))


def _validate_root_conflicts(
    records: list[tuple[str, str, str]],
    errors: list[str],
) -> None:
    by_path: dict[str, tuple[str, str]] = {}
    for semantic, path, label in records:
        existing = by_path.get(path)
        if existing is None:
            by_path[path] = (semantic, label)
            continue
        existing_semantic, existing_label = existing
        if semantic != existing_semantic and (
            semantic in _EXCLUSIVE_REMOTE_ROOT_SEMANTICS
            or existing_semantic in _EXCLUSIVE_REMOTE_ROOT_SEMANTICS
        ):
            errors.append(
                f"Remote roots {existing_label} and {label} reuse {path!r} for "
                "semantically distinct purposes."
            )


def _validate_slurm(
    value: Any,
    *,
    label: str,
    environment: Mapping[str, str],
    expand: bool = True,
    root_records: list[tuple[str, str, str]],
    errors: list[str],
) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a mapping.")
        return
    expanded = (
        _expand_environment_for_validation(
            value,
            environment=environment,
            label=label,
            errors=errors,
        )
        if expand
        else value
    )
    if not isinstance(expanded, dict):
        return
    for key, raw_value in expanded.items():
        field_label = f"{label}.{key}"
        if key in _SLURM_LIST_KEYS:
            if not isinstance(raw_value, list):
                errors.append(f"{field_label} must contain a list.")
                continue
            for index, item in enumerate(raw_value):
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"{field_label}[{index}] must contain a nonblank string.")
                    continue
                _validate_scheduler_text(item, label=f"{field_label}[{index}]", errors=errors)
                if key == "prepare_directories" and item.strip().startswith("/"):
                    _validate_remote_root(
                        item.strip(),
                        label=f"{field_label}[{index}]",
                        semantic=f"slurm-prepare-{index}",
                        records=root_records,
                        errors=errors,
                    )
        elif key == "environment":
            if not isinstance(raw_value, dict):
                errors.append(f"{field_label} must contain a mapping.")
                continue
            for raw_name, item in raw_value.items():
                name = str(raw_name)
                if not _ENVIRONMENT_NAME_PATTERN.fullmatch(name):
                    errors.append(f"{field_label} contains invalid environment name {raw_name!r}.")
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"{field_label}.{name} must contain a nonblank string.")
                    continue
                _validate_scheduler_text(item, label=f"{field_label}.{name}", errors=errors)
                if name in _REMOTE_ROOT_ENV_NAMES or (
                    name.endswith("_ROOT") and ("REMOTE" in name or name.startswith("RP_"))
                ):
                    _validate_remote_root(
                        item.strip(),
                        label=f"{field_label}.{name}",
                        semantic=_root_semantic(name),
                        records=root_records,
                        errors=errors,
                    )
        elif key == "omit_mem_directive":
            if not isinstance(raw_value, bool):
                errors.append(f"{field_label} must be a boolean.")
        elif key in _SLURM_POSITIVE_INTEGER_KEYS:
            if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < 1:
                errors.append(f"{field_label} must be a positive integer.")
        else:
            if not isinstance(raw_value, (str, int)) or isinstance(raw_value, bool):
                errors.append(f"{field_label} must contain a nonblank scalar.")
                continue
            text = str(raw_value).strip()
            if not text:
                errors.append(f"{field_label} must contain a nonblank scalar.")
                continue
            _validate_scheduler_text(text, label=field_label, errors=errors)


def _validate_slurm_document_shape(
    value: Any,
    *,
    label: str,
    errors: list[str],
) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a mapping.")
        return
    _reject_unknown_keys(value, _SLURM_KEYS, label, errors)


def _validate_scheduler_text(value: str, *, label: str, errors: list[str]) -> None:
    _validate_nonplaceholder_text(value, label=label, errors=errors)
    if "#SBATCH" in value.upper():
        errors.append(f"{label} must not embed an #SBATCH directive.")


def _validate_promotion(value: Any, *, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must declare mode: atomic_no_replace.")
        return
    _reject_unknown_keys(value, _PROMOTION_KEYS, label, errors)
    mode = _optional_text(value.get("mode"))
    if mode != "atomic_no_replace":
        errors.append(
            f"{label}.mode must be atomic_no_replace; merge, overwrite, replace, "
            "force, and unknown modes are unsupported."
        )


def _validate_project(
    *,
    project_name: str,
    project_root: Path,
    environment: Mapping[str, str],
    root_records: list[tuple[str, str, str]],
    errors: list[str],
) -> None:
    unsafe_parent = _first_unsafe_local_parent(project_root)
    if unsafe_parent is not None:
        errors.append(
            f"Project overlay {project_name!r} parent must be a real directory, "
            f"not a symlink or special entry: {unsafe_parent}"
        )
        return
    try:
        project_stat = project_root.lstat()
    except FileNotFoundError:
        errors.append(f"Project overlay {project_name!r} was not found: {project_root}")
        return
    except (OSError, ValueError) as exc:
        errors.append(f"Project overlay {project_name!r} could not be inspected safely: {exc}")
        return
    if stat.S_ISLNK(project_stat.st_mode) or not stat.S_ISDIR(project_stat.st_mode):
        errors.append(f"Project overlay {project_name!r} must be a real directory: {project_root}")
        return

    documents: dict[str, dict[str, Any]] = {}
    for relative in ("project.yaml", "config/compute.yaml", "config/dataset.yaml"):
        path = project_root / relative
        document = _load_document(path=path, supplied=None, label=relative, errors=errors)
        if document is not None:
            documents[relative] = document

    project_document = documents.get("project.yaml")
    if project_document is not None:
        declared_name = _required_text(
            project_document.get("name"),
            label="project.yaml name",
            errors=errors,
        )
        if declared_name is not None and declared_name != project_name:
            errors.append(
                f"Project overlay name {project_name!r} does not match project.yaml name "
                f"{declared_name!r}."
            )
        _validate_project_hpc_roots(
            project_document,
            environment=environment,
            root_records=root_records,
            errors=errors,
        )

    compute_document = documents.get("config/compute.yaml")
    if compute_document is not None:
        compute = compute_document.get("compute")
        if not isinstance(compute, dict):
            errors.append("config/compute.yaml must define a compute mapping.")
        else:
            slurm = compute.get("slurm")
            if slurm is not None and not isinstance(slurm, dict):
                errors.append("config/compute.yaml compute.slurm must contain a mapping.")
            elif isinstance(slurm, dict):
                expanded_slurm = _expand_environment_for_validation(
                    slurm,
                    environment=environment,
                    label="config/compute.yaml compute.slurm",
                    errors=errors,
                )
                for key in ("remote_workspace_root", "remote_artifacts_root"):
                    declared = slurm.get(key)
                    if declared is None:
                        continue
                    raw = expanded_slurm.get(key)
                    _validate_remote_root(
                        raw,
                        label=f"config/compute.yaml compute.slurm.{key}",
                        semantic=_root_semantic(key),
                        records=root_records,
                        errors=errors,
                    )

    dataset_document = documents.get("config/dataset.yaml")
    if dataset_document is not None:
        dataset = dataset_document.get("dataset")
        if not isinstance(dataset, dict):
            errors.append("config/dataset.yaml must define a dataset mapping.")
        else:
            expanded_dataset = _expand_environment_for_validation(
                dataset,
                environment=environment,
                label="config/dataset.yaml dataset",
                errors=errors,
            )
            if isinstance(expanded_dataset, dict):
                for key, declared in dataset.items():
                    if "remote" not in str(key).lower() or "root" not in str(key).lower():
                        continue
                    if declared is None:
                        continue
                    raw = expanded_dataset.get(key)
                    _validate_remote_root(
                        raw,
                        label=f"config/dataset.yaml dataset.{key}",
                        semantic=_root_semantic(str(key)),
                        records=root_records,
                        errors=errors,
                    )

    analysis_path = project_root / "config" / "analysis.yaml"
    analysis_document = _load_optional_document(
        path=analysis_path,
        label="config/analysis.yaml",
        errors=errors,
    )
    if analysis_document is not None:
        _validate_analysis_external_roots(
            analysis_document,
            environment=environment,
            root_records=root_records,
            errors=errors,
        )


def _load_optional_document(
    *,
    path: Path,
    label: str,
    errors: list[str],
) -> dict[str, Any] | None:
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        errors.append(f"{label} could not be inspected safely: {exc}")
        return None
    return _load_document(path=path, supplied=None, label=label, errors=errors)


def _validate_project_hpc_roots(
    document: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    root_records: list[tuple[str, str, str]],
    errors: list[str],
) -> None:
    hpc = document.get("hpc")
    if hpc is None:
        return
    if not isinstance(hpc, dict):
        errors.append("project.yaml hpc must contain a mapping when declared.")
        return
    declarations = hpc.get("data_roots")
    if declarations is None:
        return
    if not isinstance(declarations, list):
        errors.append("project.yaml hpc.data_roots must contain a list of mappings.")
        return
    for index, declaration in enumerate(declarations):
        label = f"project.yaml hpc.data_roots[{index}]"
        if not isinstance(declaration, dict):
            errors.append(f"{label} must contain a mapping.")
            continue
        _validate_local_path_reference(
            declaration.get("local_path"),
            label=f"{label}.local_path",
            environment=environment,
            errors=errors,
        )
        if "sync_enabled" in declaration and not isinstance(declaration.get("sync_enabled"), bool):
            errors.append(f"{label}.sync_enabled must be a boolean.")
        remote_root = declaration.get("remote_root")
        if remote_root is None:
            continue
        expanded = _expand_environment_for_validation(
            remote_root,
            environment=environment,
            label=f"{label}.remote_root",
            errors=errors,
        )
        _validate_remote_root(
            expanded,
            label=f"{label}.remote_root",
            semantic=f"project-data-{index}",
            records=root_records,
            errors=errors,
        )


def _validate_analysis_external_roots(
    document: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    root_records: list[tuple[str, str, str]],
    errors: list[str],
) -> None:
    analysis = document.get("analysis")
    if not isinstance(analysis, dict):
        errors.append("config/analysis.yaml must define an analysis mapping.")
        return
    declarations = analysis.get("external_input_roots")
    if declarations is None:
        return
    if not isinstance(declarations, dict):
        errors.append(
            "config/analysis.yaml analysis.external_input_roots must contain a mapping."
        )
        return
    for raw_name, declaration in declarations.items():
        name = str(raw_name)
        label = f"config/analysis.yaml analysis.external_input_roots.{name}"
        if not _valid_name(name):
            errors.append(f"{label} uses an invalid root name.")
        if not isinstance(declaration, dict):
            errors.append(f"{label} must contain a mapping.")
            continue
        _validate_local_path_reference(
            declaration.get("local_root"),
            label=f"{label}.local_root",
            environment=environment,
            errors=errors,
        )
        if "sync_enabled" in declaration and not isinstance(declaration.get("sync_enabled"), bool):
            errors.append(f"{label}.sync_enabled must be a boolean.")
        remote_root = declaration.get("remote_root")
        if remote_root is None:
            continue
        expanded = _expand_environment_for_validation(
            remote_root,
            environment=environment,
            label=f"{label}.remote_root",
            errors=errors,
        )
        _validate_remote_root(
            expanded,
            label=f"{label}.remote_root",
            semantic=f"analysis-input-{name}",
            records=root_records,
            errors=errors,
        )


def _validate_local_path_reference(
    value: Any,
    *,
    label: str,
    environment: Mapping[str, str],
    errors: list[str],
) -> None:
    if not isinstance(value, str):
        errors.append(f"{label} must contain a nonblank string.")
        return
    expanded = _expand_environment_for_validation(
        value,
        environment=environment,
        label=label,
        errors=errors,
    )
    if not isinstance(expanded, str) or not expanded.strip():
        errors.append(f"{label} must resolve to a nonblank path.")
        return
    _validate_nonplaceholder_text(expanded.strip(), label=label, errors=errors)


def _root_semantic(name: str) -> str:
    lowered = name.lower()
    for semantic, token in (
        ("workspace", "workspace"),
        ("artifacts", "artifact"),
        ("container", "container"),
        ("project-data", "project_data"),
        ("cache", "cache"),
        ("temporary", "tmp"),
        ("temporary", "temp"),
        ("work", "work"),
    ):
        if token in lowered:
            return semantic
    return lowered.removesuffix("_root").removesuffix("root").strip("_") or "remote-root"


def _validate_nonplaceholder_text(value: str, *, label: str, errors: list[str]) -> None:
    if _CONTROL_PATTERN.search(value):
        errors.append(f"{label} contains a control character.")
    if _ANGLE_PLACEHOLDER_PATTERN.search(value) or _ENV_PLACEHOLDER_PATTERN.search(value):
        errors.append(f"{label} contains an unresolved placeholder.")
    lowered = value.strip().lower()
    if (
        lowered in {"your-username", "your-user", "change-me", "changeme"}
        or lowered.endswith(".example")
        or lowered.endswith(".example.org")
    ):
        errors.append(f"{label} contains a starter placeholder value.")


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    label: str,
    errors: list[str],
) -> None:
    for key in value:
        if not isinstance(key, str) or key not in allowed:
            errors.append(f"{label} contains unknown key {key!r}.")


def _merge_mappings(*values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        for key, item in value.items():
            if isinstance(result.get(key), dict) and isinstance(item, Mapping):
                result[key] = _merge_mappings(result[key], item)
            else:
                result[key] = deepcopy(item)
    return result


def _required_text(value: Any, *, label: str, errors: list[str]) -> str | None:
    text = _optional_text(value)
    if text is None:
        errors.append(f"{label} must contain a nonblank string.")
    return text


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _valid_name(value: str | None) -> bool:
    return value is not None and bool(_NAME_PATTERN.fullmatch(value))


def _resolve_local_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(path))


def _first_unsafe_local_parent(path: Path) -> Path | None:
    """Return the first symlink or non-directory ancestor, if one exists."""

    absolute = Path(os.path.abspath(path))
    for parent in reversed(absolute.parents):
        if parent == Path(parent.anchor):
            continue
        try:
            parent_stat = parent.lstat()
        except FileNotFoundError:
            break
        except (OSError, ValueError):
            return parent
        if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
            return parent
    return None


def _first_environment_value(
    environment: Mapping[str, str],
    names: tuple[str, ...],
) -> str | None:
    for name in names:
        value = _optional_text(environment.get(name))
        if value is not None:
            return value
    return None


__all__ = [
    "read_local_hpc_environment_defaults",
    "render_hpc_validation_report",
    "resolve_local_hpc_environment_defaults",
    "validate_hpc_configuration",
]
