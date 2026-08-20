
"""Configuration loading helpers for the workspace CLI slice."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any
import json
import os
import re

from .paths import dataset_path, pipeline_path, project_path, resolve_path, workspace_paths
from .tool_adapters import (
    load_bids_analysis_tool_adapter,
    load_bids_tool_adapter,
    validate_tool_options_shape,
)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_SCALAR = re.compile(r"^[A-Za-z0-9_./${}-]+$")
_LOCAL_HPC_DEFAULT_KEYS = (
    "ALLIANCE_USER",
    "RESEARCH_HPC_TARGET",
    "RP_HPC_TARGET",
    "RESEARCH_HPC_TARGETS_CONFIG",
    "RP_HPC_TARGETS_CONFIG",
    "RP_REMOTE_WORKSPACE_ROOT",
    "RP_REMOTE_ARTIFACTS_ROOT",
    "RESEARCH_HPC_SSH_CONFIG",
    "RP_SSH_CONFIG",
    "RESEARCH_HPC_PROFILE",
    "RP_HPC_PROFILE",
    "RESEARCH_HPC_ROLE",
    "RP_HPC_ROLE",
)
_LOCAL_HPC_INIT_KEYS = (
    "ALLIANCE_USER",
    "RESEARCH_HPC_TARGET",
    "RESEARCH_HPC_TARGETS_CONFIG",
    "RP_REMOTE_WORKSPACE_ROOT",
    "RP_REMOTE_ARTIFACTS_ROOT",
    "RESEARCH_HPC_SSH_CONFIG",
    "RESEARCH_HPC_PROFILE",
    "RESEARCH_HPC_ROLE",
)
_LOCAL_HPC_ENV_DEFAULT_VALUES: dict[str, str] = {}
_HPC_TARGET_ENV_DEFAULT_VALUES: dict[str, str] = {}
_HPC_TARGET_CONFIG_ENV_KEYS = ("RESEARCH_HPC_TARGETS_CONFIG", "RP_HPC_TARGETS_CONFIG")
_HPC_TARGET_ENV_KEYS = ("RESEARCH_HPC_TARGET", "RP_HPC_TARGET")
_HPC_TARGET_DEFAULT_CONFIG = Path("secrets/hpc/targets.yaml")
_HPC_TARGET_SLURM_SITE_FIELDS = (
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
)
_HPC_TARGET_SLURM_DIRECTIVE_FIELDS = (
    "account",
    "partition",
    "qos",
    "nodes",
    "ntasks",
    "ntasks_per_node",
    "mem_per_cpu",
    "constraint",
    "export",
)
_PLACEHOLDER_PATTERN = re.compile(r"(<[^>\s]+>|\$\{[A-Za-z_][A-Za-z0-9_]*[^}]*\})")
_TABULAR_RESERVED_FEATURE_COLUMNS = frozenset({"split_set"})


class WorkspaceRootNotFoundError(FileNotFoundError):
    """Raised when the CLI cannot discover a workspace root."""


class ProjectOverlayNotFoundError(FileNotFoundError):
    """Raised when a requested project overlay is absent from the workspace."""


def validate_tabular_feature_columns(
    value: Any,
    *,
    label: str = "config/models.yaml models.default.feature_columns",
) -> list[str]:
    """Return a validated ordered tabular predictor contract."""

    if not isinstance(value, list):
        raise ValueError(f"{label} must be an ordered, nonempty YAML sequence of strings.")
    if not value:
        raise ValueError(f"{label} must contain at least one predictor column.")

    feature_columns: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label}[{index}] must be a nonblank string.")
        if item in seen:
            raise ValueError(f"{label} contains duplicate predictor column {item!r}.")
        if item in _TABULAR_RESERVED_FEATURE_COLUMNS:
            raise ValueError(f"{label} must not include reserved generated column {item!r}.")
        seen.add(item)
        feature_columns.append(item)
    return feature_columns


def workspace_root(default: str = ".") -> Path:
    anchor = os.environ.get("RESEARCH_PLATFORM_ROOT") if default == "." else default
    return discover_workspace_root(anchor)


def discover_workspace_root(anchor: str | Path | None = None) -> Path:
    start = _workspace_anchor(anchor)
    for candidate in (start, *start.parents):
        if (candidate / "WORKSPACE.yaml").exists():
            return candidate.resolve()
    raise WorkspaceRootNotFoundError(
        f"Could not find WORKSPACE.yaml from {start} or its parents. "
        "Run rp from inside a workspace or set RESEARCH_PLATFORM_ROOT to the workspace root or a child directory."
    )


def _workspace_anchor(anchor: str | Path | None) -> Path:
    if anchor is None:
        env_root = os.environ.get("RESEARCH_PLATFORM_ROOT")
        candidate = Path(env_root).expanduser() if env_root else Path.cwd()
    else:
        candidate = Path(anchor).expanduser()
    resolved = candidate.resolve()
    if resolved.name == "WORKSPACE.yaml" or resolved.is_file():
        return resolved.parent
    return resolved


def _resolve_workspace_root(root: str | Path | None = None) -> Path:
    if root is None:
        return discover_workspace_root()
    try:
        return discover_workspace_root(root)
    except WorkspaceRootNotFoundError:
        return _workspace_anchor(root)


def _resolve_local_hpc_root(root: str | Path | None = None) -> Path:
    try:
        return _resolve_workspace_root(root)
    except WorkspaceRootNotFoundError:
        return _workspace_anchor(root)


def load_local_hpc_env_defaults(root: str | Path | None = None) -> dict[str, str]:
    resolved_root = _resolve_local_hpc_root(root)
    env_path = resolved_root / "secrets" / ".env"
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        normalized_key = key.strip()
        if normalized_key not in _LOCAL_HPC_DEFAULT_KEYS or normalized_key in os.environ:
            if normalized_key in os.environ and _LOCAL_HPC_ENV_DEFAULT_VALUES.get(normalized_key) != os.environ.get(normalized_key):
                _LOCAL_HPC_ENV_DEFAULT_VALUES.pop(normalized_key, None)
            continue
        normalized_value = value.strip()
        os.environ[normalized_key] = normalized_value
        _LOCAL_HPC_ENV_DEFAULT_VALUES[normalized_key] = normalized_value
        loaded[normalized_key] = normalized_value
    return loaded


def write_local_hpc_env_defaults(values: dict[str, str | None], root: str | Path | None = None) -> Path:
    resolved_root = _resolve_local_hpc_root(root)
    env_path = resolved_root / "secrets" / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)

    updates = {
        key: _normalize_local_hpc_env_value(value)
        for key, value in values.items()
        if key in _LOCAL_HPC_INIT_KEYS
    }
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    rendered_lines: list[str] = []
    handled_keys: set[str] = set()

    for raw_line in existing_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            rendered_lines.append(raw_line)
            continue
        key, separator, _ = raw_line.partition("=")
        normalized_key = key.strip()
        if not separator or normalized_key not in updates:
            rendered_lines.append(raw_line)
            continue
        if normalized_key in handled_keys:
            continue
        handled_keys.add(normalized_key)
        replacement = updates[normalized_key]
        if replacement is not None:
            rendered_lines.append(f"{normalized_key}={replacement}")

    for key in _LOCAL_HPC_INIT_KEYS:
        if key not in handled_keys and key in updates and updates[key] is not None:
            rendered_lines.append(f"{key}={updates[key]}")

    content = "\n".join(rendered_lines)
    env_path.write_text(f"{content}\n" if content else "", encoding="utf-8")
    return env_path


def _normalize_local_hpc_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def resolve_hpc_targets_config_path(
    config_path: str | Path | None = None,
    *,
    root: str | Path | None = None,
) -> Path:
    resolved_root = _resolve_local_hpc_root(root)
    raw_path = _optional_text(config_path) or _first_env_value(_HPC_TARGET_CONFIG_ENV_KEYS)
    candidate = Path(raw_path).expanduser() if raw_path else _HPC_TARGET_DEFAULT_CONFIG
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    return candidate.resolve() if candidate.exists() else candidate


def load_hpc_targets_config(
    config_path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    require: bool = False,
) -> dict[str, Any]:
    resolved_path = resolve_hpc_targets_config_path(config_path, root=root)
    explicit_path = _optional_text(config_path) or _first_env_value(_HPC_TARGET_CONFIG_ENV_KEYS)
    if not resolved_path.exists():
        if require or explicit_path:
            raise FileNotFoundError(f"HPC targets config not found: {resolved_path}")
        return {}

    document = load_yaml(resolved_path)
    if not isinstance(document, dict):
        raise ValueError(f"{resolved_path} must contain a mapping.")
    targets = document.get("targets", {})
    if not isinstance(targets, dict):
        raise ValueError(f"{resolved_path} targets must contain a mapping.")
    return document


def list_hpc_targets(
    *,
    config_path: str | Path | None = None,
    root: str | Path | None = None,
) -> list[dict[str, Any]]:
    document = load_hpc_targets_config(config_path, root=root, require=True)
    default_name = _optional_text(document.get("default"))
    active_name = resolve_hpc_target_name(document, target_name=None)
    targets = document.get("targets", {})
    if not isinstance(targets, dict):
        return []
    return [
        {
            "name": str(name),
            "active": str(name) == active_name,
            "default": str(name) == default_name,
        }
        for name in sorted(targets)
    ]


def resolve_hpc_target_name(document: dict[str, Any], *, target_name: str | None = None) -> str | None:
    explicit_name = _optional_text(target_name)
    if explicit_name is not None:
        return explicit_name
    env_name = _first_env_value(_HPC_TARGET_ENV_KEYS)
    if env_name is not None:
        return env_name
    return _optional_text(document.get("default"))


def resolve_hpc_target(
    *,
    target_name: str | None = None,
    project_name: str | None = None,
    config_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any] | None:
    resolved_root = _resolve_local_hpc_root(root)
    resolved_config_path = resolve_hpc_targets_config_path(config_path, root=resolved_root)
    document = load_hpc_targets_config(config_path, root=resolved_root, require=False)
    if not document:
        return None

    selected_name = resolve_hpc_target_name(document, target_name=target_name)
    if selected_name is None:
        return None
    targets = document.get("targets", {})
    if not isinstance(targets, dict):
        raise ValueError(f"{resolved_config_path} targets must contain a mapping.")
    raw_target = targets.get(selected_name)
    if raw_target is None:
        raise ValueError(f"HPC target {selected_name!r} is not defined in {resolved_config_path}.")
    if not isinstance(raw_target, dict):
        raise ValueError(f"HPC target {selected_name!r} must contain a mapping.")

    env = _normalize_hpc_target_env_mapping(raw_target.get("env"), label=f"targets.{selected_name}.env")
    projects = raw_target.get("projects", {})
    if projects is None:
        projects = {}
    if not isinstance(projects, dict):
        raise ValueError(f"targets.{selected_name}.projects must contain a mapping when declared.")
    project_env: dict[str, str] = {}
    if project_name is not None and isinstance(projects.get(project_name), dict):
        project_target = projects[project_name]
        project_env = _normalize_hpc_target_env_mapping(
            project_target.get("env"),
            label=f"targets.{selected_name}.projects.{project_name}.env",
        )

    slurm = _normalize_hpc_target_slurm(raw_target.get("slurm"), label=f"targets.{selected_name}.slurm")
    target = {
        "name": selected_name,
        "config_path": str(resolved_config_path),
        "ssh_profile": _optional_text(raw_target.get("ssh_profile")),
        "role": _optional_text(raw_target.get("role")),
        "ssh_config": _optional_text(raw_target.get("ssh_config")),
        "env": env,
        "project_env": project_env,
        "projects": sorted(str(name) for name in projects.keys()),
        "slurm": slurm,
    }
    target["warnings"] = _hpc_target_warnings(target, workspace_root=resolved_root)
    return target


def apply_hpc_target_defaults(
    *,
    project_name: str | None = None,
    target_name: str | None = None,
    config_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any] | None:
    target = resolve_hpc_target(
        target_name=target_name,
        project_name=project_name,
        config_path=config_path,
        root=root,
    )
    if target is None:
        return None

    defaults: dict[str, str] = {}
    if target.get("ssh_profile"):
        defaults["RESEARCH_HPC_PROFILE"] = str(target["ssh_profile"])
    if target.get("role"):
        defaults["RESEARCH_HPC_ROLE"] = str(target["role"])
    if target.get("ssh_config"):
        defaults["RESEARCH_HPC_SSH_CONFIG"] = str(target["ssh_config"])
    defaults.update(target.get("env", {}))
    defaults.update(target.get("project_env", {}))

    applied: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for key, value in defaults.items():
        if _set_managed_hpc_default(key, value):
            applied[key] = value
        else:
            skipped[key] = os.environ[key]

    resolved = dict(target)
    resolved["applied_env"] = applied
    resolved["skipped_env"] = skipped
    return resolved


def merge_hpc_target_compute_defaults(
    compute_defaults: Any,
    target_default: dict[str, Any] | None,
) -> dict[str, Any]:
    target_slurm = target_default.get("slurm", {}) if isinstance(target_default, dict) else {}
    target_slurm = target_slurm if isinstance(target_slurm, dict) else {}
    merged_compute = merge_workspace_hpc_runtime_compute_defaults(compute_defaults, {"slurm": target_slurm})
    merged_slurm = dict(merged_compute.get("slurm", {})) if isinstance(merged_compute.get("slurm"), dict) else {}

    for field in _HPC_TARGET_SLURM_DIRECTIVE_FIELDS:
        if field not in target_slurm:
            continue
        if _has_slurm_field_value(merged_slurm.get(field)):
            continue
        merged_slurm[field] = target_slurm[field]
    if "omit_mem_directive" in target_slurm and "omit_mem_directive" not in merged_slurm:
        merged_slurm["omit_mem_directive"] = target_slurm["omit_mem_directive"]

    merged_compute["slurm"] = merged_slurm
    return merged_compute


def apply_hpc_target_defaults_to_compute_document(
    compute_document: Any,
    target_default: dict[str, Any] | None,
) -> Any:
    if target_default is None or not isinstance(compute_document, dict):
        return compute_document
    compute_config = compute_document.get("compute")
    if compute_config is not None and not isinstance(compute_config, dict):
        return compute_document

    merged_document = dict(compute_document)
    merged_document["compute"] = merge_hpc_target_compute_defaults(compute_config, target_default)
    return merged_document


def resolve_hpc_slurm_site_settings(slurm_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(slurm_config, dict):
        return {}
    return {
        field: slurm_config[field]
        for field in _HPC_TARGET_SLURM_SITE_FIELDS
        if _has_slurm_field_value(slurm_config.get(field))
    }


def _set_managed_hpc_default(key: str, value: str) -> bool:
    normalized_key = str(key).strip()
    if not _ENVIRONMENT_NAME_PATTERN.fullmatch(normalized_key):
        raise ValueError(f"Invalid environment variable name in HPC target: {key}")
    normalized_value = str(value).strip()
    current = os.environ.get(normalized_key)
    if current is not None and not _is_managed_hpc_default(normalized_key, current):
        return False
    os.environ[normalized_key] = normalized_value
    _HPC_TARGET_ENV_DEFAULT_VALUES[normalized_key] = normalized_value
    return True


def _is_managed_hpc_default(key: str, current_value: str) -> bool:
    return (
        _LOCAL_HPC_ENV_DEFAULT_VALUES.get(key) == current_value
        or _HPC_TARGET_ENV_DEFAULT_VALUES.get(key) == current_value
    )


def load_yaml(path: str | Path, *, resolve_env: bool = True) -> Any:
    file_path = Path(path)
    return parse_yaml(file_path.read_text(encoding="utf-8"), resolve_env=resolve_env)


def write_yaml(path: str | Path, data: Any) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(dump_yaml(data), encoding="utf-8")
    return file_path


def parse_yaml(text: str, *, resolve_env: bool = True) -> Any:
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        value = json.loads(stripped)
        return _resolve_env_in_data(value) if resolve_env else value
    lines = text.splitlines()
    value, index = _parse_block(lines, 0, 0, resolve_env=resolve_env)
    index = _skip_ignored(lines, index)
    if index != len(lines):
        raise ValueError(f"Unexpected trailing YAML content at line {index + 1}.")
    return value if value is not None else {}


def dump_yaml(data: Any) -> str:
    return "\n".join(_dump_yaml_lines(_normalize_data(data), indent=0)) + "\n"


def load_workspace_config(root: str | Path | None = None) -> dict[str, Any]:
    resolved_root = _resolve_workspace_root(root)
    config = load_yaml(resolved_root / "WORKSPACE.yaml")
    if not isinstance(config, dict):
        raise ValueError("WORKSPACE.yaml must contain a mapping at the top level.")
    return config


def resolve_workspace_hpc_runtime_default(workspace_config: dict[str, Any]) -> dict[str, Any] | None:
    hpc = workspace_config.get("hpc")
    if hpc is None:
        return None
    if not isinstance(hpc, dict):
        raise ValueError("WORKSPACE.yaml hpc must contain a mapping when declared.")

    runtime_defaults = hpc.get("runtime_defaults")
    if runtime_defaults is None:
        return None
    if not isinstance(runtime_defaults, dict):
        raise ValueError("WORKSPACE.yaml hpc.runtime_defaults must contain a mapping when declared.")

    default_name = _optional_text(runtime_defaults.get("default"))
    if default_name is None:
        return None

    catalog = runtime_defaults.get("catalog")
    if catalog is None:
        return None
    if not isinstance(catalog, dict):
        raise ValueError("WORKSPACE.yaml hpc.runtime_defaults.catalog must contain a mapping when declared.")

    preset = catalog.get(default_name)
    if preset is None:
        return None
    if not isinstance(preset, dict):
        raise ValueError(
            f"WORKSPACE.yaml hpc.runtime_defaults.catalog.{default_name} must contain a mapping."
        )

    slurm = preset.get("slurm")
    if slurm is None:
        return {"name": default_name, "slurm": {}}
    if not isinstance(slurm, dict):
        raise ValueError(
            f"WORKSPACE.yaml hpc.runtime_defaults.catalog.{default_name}.slurm must contain a mapping."
        )

    resolved_slurm: dict[str, Any] = {}
    if "modules" in slurm:
        resolved_slurm["modules"] = _normalize_string_list(
            slurm.get("modules"),
            label=f"hpc.runtime_defaults.catalog.{default_name}.slurm.modules",
        )
    if "pre_activate_commands" in slurm:
        resolved_slurm["pre_activate_commands"] = _normalize_string_list(
            slurm.get("pre_activate_commands"),
            label=f"hpc.runtime_defaults.catalog.{default_name}.slurm.pre_activate_commands",
        )
    if "prepare_directories" in slurm:
        resolved_slurm["prepare_directories"] = _normalize_raw_string_list(
            slurm.get("prepare_directories"),
            label=f"hpc.runtime_defaults.catalog.{default_name}.slurm.prepare_directories",
        )
    if "environment" in slurm:
        resolved_slurm["environment"] = _normalize_raw_string_mapping(
            slurm.get("environment"),
            label=f"hpc.runtime_defaults.catalog.{default_name}.slurm.environment",
        )
    if "remote_workspace_root" in slurm:
        remote_workspace_root = resolve_env_value(slurm.get("remote_workspace_root"))
        if remote_workspace_root is not None:
            resolved_slurm["remote_workspace_root"] = str(PurePosixPath(remote_workspace_root))
    if "remote_artifacts_root" in slurm:
        remote_artifacts_root = resolve_env_value(slurm.get("remote_artifacts_root"))
        if remote_artifacts_root is not None:
            resolved_slurm["remote_artifacts_root"] = str(PurePosixPath(remote_artifacts_root))
    return {"name": default_name, "slurm": resolved_slurm}


def _load_optional_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    data = load_yaml(file_path)
    if not isinstance(data, dict):
        raise ValueError(f"{file_path} must contain a mapping.")
    return data


def _load_named_config_dir(path: str | Path, *, child_key: str) -> dict[str, Any]:
    directory = Path(path)
    if not directory.exists():
        return {}

    loaded: dict[str, Any] = {}
    for file_path in sorted(directory.glob("*.yaml")):
        document = load_yaml(file_path)
        if not isinstance(document, dict):
            raise ValueError(f"{file_path} must contain a mapping.")
        payload = document.get(child_key, document)
        if not isinstance(payload, dict):
            raise ValueError(f"{file_path} {child_key} entry must contain a mapping.")
        name = _optional_text(payload.get("name")) or file_path.stem
        loaded[name] = payload
    return loaded


def merge_workspace_hpc_runtime_compute_defaults(
    compute_defaults: Any,
    runtime_default: dict[str, Any] | None,
) -> dict[str, Any]:
    merged_compute = dict(compute_defaults) if isinstance(compute_defaults, dict) else {}
    merged_slurm = dict(merged_compute.get("slurm")) if isinstance(merged_compute.get("slurm"), dict) else {}
    runtime_slurm = runtime_default.get("slurm", {}) if isinstance(runtime_default, dict) else {}
    runtime_slurm = runtime_slurm if isinstance(runtime_slurm, dict) else {}

    if "modules" in runtime_slurm or "modules" in merged_slurm:
        merged_slurm["modules"] = _merge_ordered_unique_string_lists(
            runtime_slurm.get("modules"),
            merged_slurm.get("modules"),
        )
    if "pre_activate_commands" in runtime_slurm or "pre_activate_commands" in merged_slurm:
        merged_slurm["pre_activate_commands"] = _merge_ordered_unique_string_lists(
            runtime_slurm.get("pre_activate_commands"),
            merged_slurm.get("pre_activate_commands"),
        )
    if "prepare_directories" in runtime_slurm or "prepare_directories" in merged_slurm:
        merged_slurm["prepare_directories"] = _merge_ordered_unique_prepare_directories(
            runtime_slurm.get("prepare_directories"),
            merged_slurm.get("prepare_directories"),
        )
    if "environment" in runtime_slurm or "environment" in merged_slurm:
        merged_slurm["environment"] = _merge_string_mapping(
            runtime_slurm.get("environment"),
            merged_slurm.get("environment"),
        )
    if "omit_mem_directive" in runtime_slurm and "omit_mem_directive" not in merged_slurm:
        merged_slurm["omit_mem_directive"] = runtime_slurm["omit_mem_directive"]
    if not _optional_text(merged_slurm.get("remote_workspace_root")):
        runtime_remote_workspace_root = _optional_text(runtime_slurm.get("remote_workspace_root"))
        if runtime_remote_workspace_root is not None:
            merged_slurm["remote_workspace_root"] = runtime_remote_workspace_root
    if not _optional_text(merged_slurm.get("remote_artifacts_root")):
        runtime_remote_artifacts_root = _optional_text(runtime_slurm.get("remote_artifacts_root"))
        if runtime_remote_artifacts_root is not None:
            merged_slurm["remote_artifacts_root"] = runtime_remote_artifacts_root

    merged_compute["slurm"] = merged_slurm
    return merged_compute


def apply_workspace_hpc_runtime_defaults_to_compute_document(
    compute_document: Any,
    runtime_default: dict[str, Any] | None,
) -> Any:
    if runtime_default is None or not isinstance(compute_document, dict):
        return compute_document

    compute_config = compute_document.get("compute")
    if compute_config is not None and not isinstance(compute_config, dict):
        return compute_document
    if not _can_merge_compute_runtime_defaults(compute_config):
        return compute_document

    merged_document = dict(compute_document)
    merged_document["compute"] = merge_workspace_hpc_runtime_compute_defaults(compute_config, runtime_default)
    return merged_document


def default_project_name(workspace_config: dict[str, Any]) -> str:
    projects = workspace_config.get("projects", {})
    project_name = projects.get("default")
    if not project_name:
        raise ValueError("WORKSPACE.yaml must define projects.default for the rp workspace CLI.")
    return str(project_name)


def resolve_env_value(value: Any) -> str | None:
    if value is None:
        return None
    text = _resolve_env(str(value)).strip()
    return text or None


def resolve_bids_dataset_root(bundle: dict[str, Any], *, root: str | Path | None = None) -> Path:
    resolved_root = _resolve_workspace_root(root)
    workspace = bundle["workspace"]
    dataset = bundle["dataset"]["dataset"]
    explicit_root = resolve_env_value(dataset.get("bids_root"))
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    return dataset_path(resolved_root, workspace, str(dataset["primary"]))


def resolve_bids_input_derivative_root(bundle: dict[str, Any], *, root: str | Path | None = None) -> Path:
    dataset = bundle["dataset"]["dataset"]
    explicit_root = resolve_env_value(dataset.get("input_derivative_root"))
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    return resolve_bids_dataset_root(bundle, root=root) / "derivatives" / str(dataset["input_derivative"])


def resolve_bids_remote_dataset_root(bundle: dict[str, Any]) -> str | None:
    dataset = bundle["dataset"]["dataset"]
    explicit_root = resolve_env_value(dataset.get("remote_bids_root"))
    if explicit_root:
        return str(PurePosixPath(explicit_root))
    return None


def resolve_bids_remote_input_derivative_root(bundle: dict[str, Any]) -> str | None:
    dataset = bundle["dataset"]["dataset"]
    explicit_root = resolve_env_value(dataset.get("remote_input_derivative_root"))
    if explicit_root:
        return str(PurePosixPath(explicit_root))
    remote_dataset_root = resolve_bids_remote_dataset_root(bundle)
    if remote_dataset_root:
        return str(PurePosixPath(remote_dataset_root) / "derivatives" / str(dataset["input_derivative"]))
    return None


def load_project_bundle(project_name: str | None = None, root: str | Path | None = None) -> dict[str, Any]:
    resolved_root = _resolve_workspace_root(root)
    target_default = apply_hpc_target_defaults(project_name=None, root=resolved_root)
    workspace = load_workspace_config(resolved_root)
    selected_project = project_name or default_project_name(workspace)
    target_default = apply_hpc_target_defaults(project_name=selected_project, root=resolved_root) or target_default
    runtime_default = resolve_workspace_hpc_runtime_default(workspace)
    project_root_path = project_path(resolved_root, workspace, selected_project)
    _require_project_overlay(selected_project, project_root_path, workspace_root=resolved_root)
    models_path = project_root_path / "config" / "models.yaml"
    analysis_root = project_root_path / "config" / "analysis"
    compute_document = load_yaml(project_root_path / "config" / "compute.yaml")
    compute_document = apply_workspace_hpc_runtime_defaults_to_compute_document(compute_document, runtime_default)
    compute_document = apply_hpc_target_defaults_to_compute_document(compute_document, target_default)
    bundle = {
        "workspace": workspace,
        "project_root": project_root_path,
        "project": load_yaml(project_root_path / "project.yaml"),
        "dataset": load_yaml(project_root_path / "config" / "dataset.yaml"),
        "compute": compute_document,
        "preprocessing": _load_optional_yaml(project_root_path / "config" / "preprocessing.yaml"),
        "analysis": _load_optional_yaml(project_root_path / "config" / "analysis.yaml"),
        "analysis_models": {"models": _load_named_config_dir(analysis_root / "models", child_key="model")},
        "analysis_groupings": {"groupings": _load_named_config_dir(analysis_root / "groupings", child_key="grouping")},
        "models": load_yaml(models_path) if models_path.exists() else {},
        "hpc_target": target_default,
    }
    return bundle


def load_project_record(project_name: str | None = None, root: str | Path | None = None) -> dict[str, Any]:
    resolved_root = _resolve_workspace_root(root)
    target_default = apply_hpc_target_defaults(project_name=None, root=resolved_root)
    workspace = load_workspace_config(resolved_root)
    selected_project = project_name or default_project_name(workspace)
    target_default = apply_hpc_target_defaults(project_name=selected_project, root=resolved_root) or target_default
    project_root_path = project_path(resolved_root, workspace, selected_project)
    _require_project_overlay(selected_project, project_root_path, workspace_root=resolved_root)
    return {
        "workspace_root": resolved_root,
        "workspace": workspace,
        "project_root": project_root_path,
        "project": load_yaml(project_root_path / "project.yaml"),
        "hpc_target": target_default,
    }


def _require_project_overlay(project_name: str, project_root: Path, *, workspace_root: Path) -> None:
    if project_root.is_dir() and (project_root / "project.yaml").is_file():
        return
    try:
        expected_path = project_root.relative_to(workspace_root).as_posix()
        projects_root = project_root.parent.relative_to(workspace_root).as_posix()
    except ValueError:
        expected_path = str(project_root)
        projects_root = str(project_root.parent)
    raise ProjectOverlayNotFoundError(
        f"Project overlay {project_name!r} was not found at {expected_path}. "
        f"Expected a project overlay under {projects_root}/. "
        f"Initialize it with `rp project init {project_name}`."
    )


def project_has_structured_config(project_root: str | Path) -> bool:
    project_root_path = Path(project_root)
    base_required_files = (
        project_root_path / "config" / "compute.yaml",
        project_root_path / "config" / "dataset.yaml",
    )
    action_files = (
        project_root_path / "config" / "preprocessing.yaml",
        project_root_path / "config" / "analysis.yaml",
    )
    return all(path.exists() for path in base_required_files) and any(path.exists() for path in action_files)


def resolve_project_notebook_path(
    project_config: dict[str, Any],
    *,
    workspace_root: str | Path,
    project_root: str | Path,
) -> Path | None:
    overlay = project_config.get("overlay")
    if isinstance(overlay, dict):
        notebook_value = overlay.get("notebook")
        if notebook_value:
            return _resolve_project_declared_path(notebook_value, workspace_root=workspace_root, project_root=project_root)
    notebook_value = project_config.get("notebook")
    if notebook_value:
        return _resolve_project_declared_path(notebook_value, workspace_root=workspace_root, project_root=project_root)
    return None


def resolve_project_overlay_data_roots(
    project_config: dict[str, Any],
    *,
    workspace_root: str | Path,
    project_root: str | Path,
    workspace_config: dict[str, Any],
) -> list[Path]:
    roots: list[Path] = []
    for dataset_name in project_config.get("datasets", []) or []:
        roots.append(dataset_path(workspace_root, workspace_config, str(dataset_name)))

    overlay = project_config.get("overlay")
    if not isinstance(overlay, dict):
        return _dedupe_paths(roots)

    private_data_root = overlay.get("private_data_root")
    if private_data_root:
        roots.append(_resolve_project_declared_path(private_data_root, workspace_root=workspace_root, project_root=project_root))

    raw_inputs = overlay.get("raw_inputs")
    if isinstance(raw_inputs, dict):
        for value in raw_inputs.values():
            if value:
                roots.append(_resolve_project_declared_path(value, workspace_root=workspace_root, project_root=project_root))
    return _dedupe_paths(roots)


def _analysis_root_label(value: object) -> str:
    label = str(value).replace("_", "-")
    if label.endswith("-root"):
        return label
    return f"{label}-root"


def build_data_root_spec(
    label: str,
    path: str | Path,
    *,
    remote_root: str | None = None,
    sync_enabled: bool = True,
    preserve_nested_sync_target: bool = False,
    name: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    resolved_path = Path(path).resolve()
    normalized_remote_root = resolve_env_value(remote_root)
    return {
        "label": label,
        "path": resolved_path,
        "exists": resolved_path.exists(),
        "remote_root": str(PurePosixPath(normalized_remote_root)) if normalized_remote_root else None,
        "sync_enabled": sync_enabled,
        "preserve_nested_sync_target": preserve_nested_sync_target,
        "name": name,
        "source": source,
    }


def merge_declared_data_roots(
    existing_roots: list[dict[str, Any]],
    declared_roots: list[dict[str, Any]],
    *,
    conflict_label: str = "declared data roots",
) -> tuple[list[dict[str, Any]], list[str]]:
    merged = [dict(root, path=Path(root["path"]).resolve()) for root in existing_roots]
    path_to_index = {Path(root["path"]).resolve(): index for index, root in enumerate(merged)}
    errors: list[str] = []

    for declaration in declared_roots:
        resolved_path = Path(declaration["path"]).resolve()
        existing_index = path_to_index.get(resolved_path)
        if existing_index is None:
            path_to_index[resolved_path] = len(merged)
            merged.append(
                build_data_root_spec(
                    str(declaration["label"]),
                    resolved_path,
                    remote_root=declaration.get("remote_root"),
                    sync_enabled=bool(declaration.get("sync_enabled", True)),
                    preserve_nested_sync_target=bool(declaration.get("preserve_nested_sync_target", False)),
                    name=_optional_text(declaration.get("name")),
                    source=_optional_text(declaration.get("source")),
                )
            )
            continue

        existing_root = dict(merged[existing_index])
        existing_remote_root = _normalize_optional_remote_root(existing_root.get("remote_root"))
        declared_remote_root = _normalize_optional_remote_root(declaration.get("remote_root"))
        if existing_remote_root and declared_remote_root and existing_remote_root != declared_remote_root:
            errors.append(
                f"{conflict_label} declare a conflicting remote_root for {resolved_path}: {declared_remote_root}"
            )
            continue
        if existing_remote_root is None and declared_remote_root is not None:
            existing_root["remote_root"] = declared_remote_root
        if declaration.get("sync_enabled") is False:
            existing_root["sync_enabled"] = False
        if declaration.get("preserve_nested_sync_target") is True:
            existing_root["preserve_nested_sync_target"] = True
        if existing_root.get("name") is None and declaration.get("name") is not None:
            existing_root["name"] = str(declaration["name"])
        if existing_root.get("source") is None and declaration.get("source") is not None:
            existing_root["source"] = str(declaration["source"])
        merged[existing_index] = existing_root
    return merged, errors


def resolve_project_hpc_data_root_declarations(
    project_config: dict[str, Any],
    *,
    workspace_root: str | Path,
    project_root: str | Path,
    require_remote_root: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    hpc_config = project_config.get("hpc")
    if hpc_config is None:
        return [], []
    if not isinstance(hpc_config, dict):
        return [], ["project.yaml hpc must contain a mapping when declared."]

    declarations = hpc_config.get("data_roots")
    if declarations is None:
        return [], []
    if not isinstance(declarations, list):
        return [], ["project.yaml hpc.data_roots must contain a list of mappings."]

    resolved: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, declaration in enumerate(declarations):
        label = f"project.yaml hpc.data_roots[{index}]"
        if not isinstance(declaration, dict):
            errors.append(f"{label} must contain a mapping.")
            continue

        local_path_value = _optional_text(declaration.get("local_path"))
        if local_path_value is None:
            errors.append(f"{label} must define local_path.")
            continue

        remote_root = resolve_env_value(declaration.get("remote_root"))
        if require_remote_root and remote_root is None:
            errors.append(f"{label} must define remote_root.")
            continue

        sync_enabled = declaration.get("sync_enabled", True)
        if not isinstance(sync_enabled, bool):
            errors.append(f"{label} sync_enabled must be true or false when declared.")
            continue

        try:
            local_path = _resolve_project_declared_path(
                local_path_value,
                workspace_root=workspace_root,
                project_root=project_root,
            )
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{label} local_path could not be resolved: {exc}")
            continue

        resolved.append(
            build_data_root_spec(
                _optional_text(declaration.get("label")) or "data-root",
                local_path,
                remote_root=remote_root,
                sync_enabled=sync_enabled,
                source="project.hpc.data_roots",
            )
        )
    return resolved, errors


def resolve_analysis_external_input_root_declarations(
    analysis_config: dict[str, Any],
    *,
    workspace_root: str | Path,
    project_root: str | Path,
    require_remote_root: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    external_roots = analysis_config.get("external_input_roots")
    if external_roots is None:
        return [], []
    if not isinstance(external_roots, dict):
        return [], ["config/analysis.yaml analysis.external_input_roots must contain a mapping when declared."]

    resolved: list[dict[str, Any]] = []
    errors: list[str] = []
    for name, declaration in external_roots.items():
        normalized_name = _optional_text(name)
        label = f"config/analysis.yaml analysis.external_input_roots.{name}"
        if normalized_name is None:
            errors.append(f"{label} must use a non-empty root name.")
            continue
        if not isinstance(declaration, dict):
            errors.append(f"{label} must contain a mapping.")
            continue

        local_root_value = _optional_text(declaration.get("local_root"))
        if local_root_value is None:
            errors.append(f"{label} must define local_root.")
            continue

        remote_root = resolve_env_value(declaration.get("remote_root"))
        if require_remote_root and remote_root is None:
            errors.append(f"{label} must define remote_root.")
            continue

        sync_enabled = declaration.get("sync_enabled", True)
        if not isinstance(sync_enabled, bool):
            errors.append(f"{label} sync_enabled must be true or false when declared.")
            continue

        try:
            local_root = _resolve_project_declared_path(
                local_root_value,
                workspace_root=workspace_root,
                project_root=project_root,
            )
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{label} local_root could not be resolved: {exc}")
            continue

        resolved.append(
            build_data_root_spec(
                _optional_text(declaration.get("label")) or _analysis_root_label(normalized_name),
                local_root,
                remote_root=remote_root,
                sync_enabled=sync_enabled,
                preserve_nested_sync_target=remote_root is not None,
                name=normalized_name,
                source="analysis.external_input_roots",
            )
        )
    return resolved, errors


def validate_project_bundle(
    bundle: dict[str, Any],
    *,
    root: str | Path | None = None,
    require_default_batch: bool = True,
) -> list[str]:
    resolved_root = _resolve_workspace_root(root)
    workspace = bundle["workspace"]
    project_root_path = Path(bundle["project_root"])
    errors: list[str] = []

    project = _expect_mapping(bundle.get("project"), "project.yaml", errors)
    dataset = _expect_mapping(bundle.get("dataset"), "config/dataset.yaml", errors).get("dataset", {})
    compute = _expect_mapping(bundle.get("compute"), "config/compute.yaml", errors).get("compute", {})
    preprocessing = _expect_mapping(bundle.get("preprocessing") or {}, "config/preprocessing.yaml", errors).get("preprocessing", {})
    analysis = _expect_mapping(bundle.get("analysis") or {}, "config/analysis.yaml", errors).get("analysis", {})
    analysis_models = _expect_mapping(bundle.get("analysis_models") or {}, "config/analysis/models", errors).get("models", {})
    models = _expect_mapping(bundle.get("models"), "config/models.yaml", errors).get("models", {})
    slice_name = project_slice(bundle)
    analysis_external_roots: list[dict[str, Any]] = []

    if project.get("name") is None:
        errors.append("project.yaml must define project name.")
    if dataset.get("primary") is None:
        errors.append("config/dataset.yaml must define dataset.primary.")
    if not preprocessing and not analysis:
        errors.append("Project config must define either config/preprocessing.yaml or config/analysis.yaml.")
    if preprocessing and preprocessing.get("default_batch") is None:
        errors.append("config/preprocessing.yaml must define preprocessing.default_batch.")

    if errors:
        return errors

    _, hpc_data_root_errors = resolve_project_hpc_data_root_declarations(
        project,
        workspace_root=resolved_root,
        project_root=project_root_path,
        require_remote_root=False,
    )
    errors.extend(hpc_data_root_errors)

    try:
        dataset_root_path = resolve_bids_dataset_root(bundle, root=resolved_root) if slice_name == "bids" else dataset_path(
            resolved_root,
            workspace,
            str(dataset["primary"]),
        )
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(str(exc))
        return errors
    required_paths: dict[str, Path] = {
        "project root": project_root_path,
        "dataset root": dataset_root_path,
    }
    bids_requires_input_derivative = slice_name == "bids"
    preprocessing_adapter = None
    if slice_name == "bids":
        if preprocessing and preprocessing.get("tool_adapter") is not None:
            try:
                preprocessing_adapter = load_bids_tool_adapter(preprocessing)
            except ValueError as exc:
                errors.append(str(exc))
            else:
                bids_requires_input_derivative = preprocessing_adapter.requires_input_derivative()
        if analysis:
            bids_requires_input_derivative = True
        if bids_requires_input_derivative:
            if dataset.get("input_derivative") is None:
                errors.append("config/dataset.yaml must define dataset.input_derivative.")
            else:
                derivative_root_path = resolve_bids_input_derivative_root(bundle, root=resolved_root)
                required_paths |= {
                    "input derivative root": derivative_root_path,
                }
        if preprocessing:
            if preprocessing.get("pipeline") is None:
                errors.append("config/preprocessing.yaml must define preprocessing.pipeline.")
            if preprocessing.get("tool") is None:
                errors.append("config/preprocessing.yaml must define preprocessing.tool.")
            if preprocessing.get("tool_adapter") is None:
                errors.append("config/preprocessing.yaml must define preprocessing.tool_adapter.")
            errors.extend(validate_tool_options_shape(preprocessing))
            if require_default_batch:
                batch_path = project_root_path / "manifests" / "batches" / f"{preprocessing['default_batch']}.tsv"
                required_paths["default batch manifest"] = batch_path
            if not errors:
                pipeline_root_path = pipeline_path(resolved_root, workspace, str(preprocessing["pipeline"]))
                required_paths |= {
                    "pipeline root": pipeline_root_path,
                    "pipeline defaults": pipeline_root_path / "config" / "defaults.yaml",
                    "local profile": pipeline_root_path / "profiles" / str(preprocessing.get("local_profile", "local")) / "config.yaml",
                    "slurm profile": pipeline_root_path / "profiles" / str(preprocessing.get("slurm_profile", "slurm")) / "config.yaml",
                }
                pipeline_defaults_path = pipeline_root_path / "config" / "defaults.yaml"
                if pipeline_defaults_path.exists():
                    pipeline_defaults = load_yaml(pipeline_defaults_path)
                    if preprocessing_adapter is not None:
                        errors.extend(
                            preprocessing_adapter.validate_project(
                                bundle=bundle,
                                pipeline_defaults=pipeline_defaults,
                                workspace_root=str(resolved_root),
                            )
                        )
        if analysis:
            analysis_defaults = _expect_mapping(analysis.get("defaults"), "config/analysis.yaml analysis.defaults", errors)
            analysis_tools = _expect_mapping(analysis.get("tools"), "config/analysis.yaml analysis.tools", errors)
            analysis_inputs = _expect_mapping(analysis.get("inputs"), "config/analysis.yaml analysis.inputs", errors)
            analysis_stages = _expect_mapping(analysis.get("stages"), "config/analysis.yaml analysis.stages", errors)
            analysis_external_roots, analysis_external_root_errors = resolve_analysis_external_input_root_declarations(
                analysis,
                workspace_root=resolved_root,
                project_root=project_root_path,
                require_remote_root=False,
            )
            errors.extend(analysis_external_root_errors)
            analysis_external_root_names = {
                str(root_spec.get("name"))
                for root_spec in analysis_external_roots
                if _optional_text(root_spec.get("name")) is not None
            }
            tool_profiles = compute.get("tool_profiles")
            if tool_profiles is not None and not isinstance(tool_profiles, dict):
                errors.append("config/compute.yaml compute.tool_profiles must contain a mapping when declared.")
            stage_name = _optional_text(analysis_defaults.get("stage"))
            tool_name = _optional_text(analysis_defaults.get("tool"))
            model_ref = _optional_text(analysis_defaults.get("model_ref"))
            if analysis.get("pipeline") is None:
                errors.append("config/analysis.yaml must define analysis.pipeline.")
            if analysis.get("local_profile") is None:
                errors.append("config/analysis.yaml must define analysis.local_profile.")
            if analysis.get("slurm_profile") is None:
                errors.append("config/analysis.yaml must define analysis.slurm_profile.")
            if stage_name is None:
                errors.append("config/analysis.yaml must define analysis.defaults.stage.")
            if tool_name is None:
                errors.append("config/analysis.yaml must define analysis.defaults.tool.")
            if model_ref is None:
                errors.append("config/analysis.yaml must define analysis.defaults.model_ref.")
            if not analysis_models:
                errors.append("config/analysis/models must define at least one reusable model spec.")
            if model_ref is not None and model_ref not in analysis_models:
                errors.append(f"analysis.defaults.model_ref {model_ref!r} was not found under config/analysis/models/*.yaml.")
            stage_config = analysis_stages.get(stage_name) if stage_name is not None else None
            if stage_name is not None and not isinstance(stage_config, dict):
                errors.append(f"config/analysis.yaml analysis.stages.{stage_name} must contain a mapping.")
                stage_config = None
            tool_entry = analysis_tools.get(tool_name) if tool_name is not None else None
            if tool_name is not None and not isinstance(tool_entry, dict):
                errors.append(f"config/analysis.yaml analysis.tools.{tool_name} must contain a mapping.")
                tool_entry = None
            bold_config = analysis_inputs.get("bold")
            if not isinstance(bold_config, dict):
                errors.append("config/analysis.yaml analysis.inputs.bold must contain a mapping.")
            else:
                try:
                    bold_patterns = _normalize_string_list(
                        bold_config.get("patterns"),
                        label="config/analysis.yaml analysis.inputs.bold.patterns",
                    )
                except ValueError as exc:
                    errors.append(str(exc))
                else:
                    if not bold_patterns:
                        errors.append("config/analysis.yaml analysis.inputs.bold.patterns must define at least one path pattern.")
            evs_config = analysis_inputs.get("evs")
            if not isinstance(evs_config, dict):
                errors.append("config/analysis.yaml analysis.inputs.evs must contain a mapping.")
            else:
                ev_root_ref = _optional_text(evs_config.get("root_ref"))
                if ev_root_ref is not None:
                    if ev_root_ref not in analysis_external_root_names:
                        errors.append(
                            f"config/analysis.yaml analysis.inputs.evs.root_ref {ev_root_ref!r} was not found under analysis.external_input_roots."
                        )
                elif _optional_text(evs_config.get("root")) is None:
                    errors.append("config/analysis.yaml analysis.inputs.evs must define either root_ref or root.")
                try:
                    ev_patterns = _normalize_string_list(
                        evs_config.get("patterns"),
                        label="config/analysis.yaml analysis.inputs.evs.patterns",
                    )
                except ValueError as exc:
                    errors.append(str(exc))
                else:
                    if not ev_patterns:
                        errors.append("config/analysis.yaml analysis.inputs.evs.patterns must define at least one path pattern.")
            for input_name, input_config in analysis_inputs.items():
                if not isinstance(input_config, dict):
                    continue
                root_ref = _optional_text(input_config.get("root_ref"))
                if root_ref is None:
                    continue
                if root_ref not in analysis_external_root_names:
                    errors.append(
                        f"config/analysis.yaml analysis.inputs.{input_name}.root_ref {root_ref!r} was not found under analysis.external_input_roots."
                    )
            if stage_config is not None and stage_config.get("default_batch") is None:
                errors.append(f"config/analysis.yaml analysis.stages.{stage_name}.default_batch must be defined.")
            if tool_entry is not None:
                if tool_entry.get("adapter") is None:
                    errors.append(f"config/analysis.yaml analysis.tools.{tool_name}.adapter must be defined.")
                runtime_profile_name = _optional_text(tool_entry.get("runtime_profile"))
                if runtime_profile_name is None:
                    errors.append(f"config/analysis.yaml analysis.tools.{tool_name}.runtime_profile must be defined.")
                elif isinstance(tool_profiles, dict) and runtime_profile_name not in tool_profiles:
                    errors.append(
                        f"config/compute.yaml compute.tool_profiles must define {runtime_profile_name!r} for analysis.tools.{tool_name}."
                    )
            if require_default_batch and isinstance(stage_config, dict) and stage_config.get("default_batch"):
                analysis_batch_path = project_root_path / "manifests" / "batches" / f"{stage_config['default_batch']}.tsv"
                required_paths["analysis default batch manifest"] = analysis_batch_path
            if not errors and isinstance(stage_config, dict) and isinstance(tool_entry, dict):
                analysis_pipeline_root_path = pipeline_path(resolved_root, workspace, str(analysis["pipeline"]))
                required_paths |= {
                    "analysis pipeline root": analysis_pipeline_root_path,
                    "analysis pipeline defaults": analysis_pipeline_root_path / "config" / "defaults.yaml",
                    "analysis local profile": analysis_pipeline_root_path / "profiles" / str(analysis.get("local_profile", "local")) / "config.yaml",
                    "analysis slurm profile": analysis_pipeline_root_path / "profiles" / str(analysis.get("slurm_profile", "slurm")) / "config.yaml",
                }
                analysis_pipeline_defaults_path = analysis_pipeline_root_path / "config" / "defaults.yaml"
                if analysis_pipeline_defaults_path.exists():
                    pipeline_defaults = load_yaml(analysis_pipeline_defaults_path)
                    try:
                        adapter = load_bids_analysis_tool_adapter(tool_entry)
                    except ValueError as exc:
                        errors.append(str(exc))
                    else:
                        errors.extend(
                            adapter.validate_project(
                                bundle=bundle,
                                pipeline_defaults=pipeline_defaults,
                                workspace_root=str(resolved_root),
                            )
                        )
    elif slice_name == "tabular":
        canonical_dataset = dataset.get("canonical_dataset")
        canonical_features_root = dataset.get("canonical_features_root")
        if canonical_dataset is None:
            errors.append("config/dataset.yaml must define dataset.canonical_dataset for the tabular slice.")
        if canonical_features_root is None:
            errors.append("config/dataset.yaml must define dataset.canonical_features_root for the tabular slice.")
        default_model = _expect_mapping(models.get("default"), "config/models.yaml models.default", errors)
        if default_model.get("kind") != "logistic_regression":
            errors.append("This slice supports only models.default.kind=logistic_regression.")
        try:
            validate_tabular_feature_columns(default_model.get("feature_columns"))
        except ValueError as exc:
            errors.append(str(exc))
        if errors:
            return errors
        canonical_dataset_root = dataset_path(resolved_root, workspace, str(canonical_dataset))
        required_paths |= {
            "canonical dataset root": canonical_dataset_root,
            "canonical features root": canonical_dataset_root / str(canonical_features_root),
        }
    else:
        errors.append(f"Unsupported project slice: {slice_name}")
        return errors

    merge_validation_roots: list[dict[str, Any]] = []
    if slice_name == "bids":
        merge_validation_roots.append(
            build_data_root_spec(
                "raw-dataset-root",
                required_paths.get("dataset root", dataset_root_path),
                remote_root=resolve_bids_remote_dataset_root(bundle),
            )
        )
        if bids_requires_input_derivative:
            merge_derivative_root = required_paths.get("input derivative root")
            if merge_derivative_root is None and dataset.get("input_derivative") is not None:
                merge_derivative_root = resolve_bids_input_derivative_root(bundle, root=resolved_root)
            merge_validation_roots.append(
                build_data_root_spec(
                    "input-derivative-root",
                    merge_derivative_root,
                    remote_root=resolve_bids_remote_input_derivative_root(bundle)
                    if dataset.get("input_derivative") is not None
                    else None,
                )
            )
    elif slice_name == "tabular":
        merge_validation_roots.extend(
            [
                build_data_root_spec("dataset-root", dataset_root_path),
            ]
        )
        canonical_features_root = required_paths.get("canonical features root")
        if canonical_features_root is not None:
            merge_validation_roots.append(build_data_root_spec("canonical-features-root", canonical_features_root))

    declared_data_roots, _ = resolve_project_hpc_data_root_declarations(
        project,
        workspace_root=resolved_root,
        project_root=project_root_path,
        require_remote_root=False,
    )
    _, data_root_merge_errors = merge_declared_data_roots(
        merge_validation_roots,
        [*declared_data_roots, *analysis_external_roots],
        conflict_label="Project data roots",
    )
    errors.extend(data_root_merge_errors)

    for label, path in required_paths.items():
        if not Path(path).exists():
            errors.append(f"Missing {label}: {path}")

    paths_config = workspace_paths(resolved_root, workspace)
    for required_root in ("artifacts_root", "datasets_root", "ops_root"):
        if required_root not in paths_config:
            errors.append(f"WORKSPACE.yaml must define paths.{required_root}.")

    if compute.get("default_profile") not in {"local", "slurm"}:
        errors.append("config/compute.yaml must define compute.default_profile as local or slurm.")
    slurm_config = compute.get("slurm")
    if slurm_config is not None and not isinstance(slurm_config, dict):
        errors.append("config/compute.yaml compute.slurm must contain a mapping when declared.")
        return errors
    if isinstance(slurm_config, dict) and "environment" in slurm_config:
        try:
            _normalize_raw_string_mapping(slurm_config.get("environment"), label="config/compute.yaml compute.slurm.environment")
        except ValueError as exc:
            errors.append(str(exc))
    if isinstance(slurm_config, dict) and "prepare_directories" in slurm_config:
        try:
            _normalize_raw_string_list(
                slurm_config.get("prepare_directories"),
                label="config/compute.yaml compute.slurm.prepare_directories",
            )
        except ValueError as exc:
            errors.append(str(exc))

    return errors


def summarize_bundle(bundle: dict[str, Any], *, root: str | Path | None = None) -> dict[str, Any]:
    resolved_root = _resolve_workspace_root(root)
    workspace = bundle["workspace"]
    paths_config = workspace_paths(resolved_root, workspace)
    project_root_path = Path(bundle["project_root"])
    dataset_config = bundle["dataset"]["dataset"]
    preprocessing = _expect_mapping(bundle.get("preprocessing") or {}, "config/preprocessing.yaml", []).get("preprocessing", {})
    analysis = _expect_mapping(bundle.get("analysis") or {}, "config/analysis.yaml", []).get("analysis", {})
    slice_name = project_slice(bundle)
    resolved_paths = {
        name: str(path) for name, path in paths_config.items()
    } | {
        "project_root": str(project_root_path),
        "dataset_root": str(resolve_bids_dataset_root(bundle, root=resolved_root))
        if slice_name == "bids"
        else str(dataset_path(resolved_root, workspace, dataset_config["primary"])),
    }
    if slice_name == "bids":
        requires_input_derivative = True
        if preprocessing.get("tool_adapter") is not None and not analysis:
            try:
                requires_input_derivative = load_bids_tool_adapter(preprocessing).requires_input_derivative()
            except ValueError:
                requires_input_derivative = True
        if requires_input_derivative:
            resolved_paths["input_derivative_root"] = str(resolve_bids_input_derivative_root(bundle, root=resolved_root))
        if preprocessing.get("pipeline"):
            resolved_paths["pipeline_root"] = str(pipeline_path(resolved_root, workspace, preprocessing["pipeline"]))
        if analysis.get("pipeline"):
            resolved_paths["analysis_pipeline_root"] = str(pipeline_path(resolved_root, workspace, analysis["pipeline"]))
        remote_dataset_root = resolve_bids_remote_dataset_root(bundle)
        remote_input_derivative_root = resolve_bids_remote_input_derivative_root(bundle) if requires_input_derivative else None
        if remote_dataset_root:
            resolved_paths["remote_dataset_root"] = remote_dataset_root
        if remote_input_derivative_root:
            resolved_paths["remote_input_derivative_root"] = remote_input_derivative_root
    elif slice_name == "tabular":
        canonical_dataset_root = dataset_path(resolved_root, workspace, dataset_config["canonical_dataset"])
        resolved_paths |= {
            "canonical_dataset_root": str(canonical_dataset_root),
            "canonical_features_root": str(canonical_dataset_root / dataset_config["canonical_features_root"]),
        }
    return {
        "workspace_root": str(resolved_root),
        "slice": slice_name,
        "project": bundle["project"],
        "dataset": bundle["dataset"],
        "compute": bundle["compute"],
        "preprocessing": bundle["preprocessing"],
        "analysis": bundle.get("analysis", {}),
        "analysis_models": bundle.get("analysis_models", {}),
        "analysis_groupings": bundle.get("analysis_groupings", {}),
        "models": bundle.get("models", {}),
        "resolved_paths": resolved_paths,
    }


def project_slice(bundle: dict[str, Any]) -> str:
    preprocessing = _expect_mapping(bundle.get("preprocessing") or {}, "config/preprocessing.yaml", []).get("preprocessing", {})
    analysis = _expect_mapping(bundle.get("analysis") or {}, "config/analysis.yaml", []).get("analysis", {})
    explicit = preprocessing.get("slice")
    if explicit:
        return str(explicit)
    explicit = analysis.get("slice")
    if explicit:
        return str(explicit)
    return "bids"


def _expect_mapping(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a mapping.")
        return {}
    return value


def _skip_ignored(lines: list[str], index: int) -> int:
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        return index
    return index


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_block(lines: list[str], index: int, indent: int, *, resolve_env: bool) -> tuple[Any, int]:
    index = _skip_ignored(lines, index)
    if index >= len(lines):
        return None, index

    current_indent = _indent_of(lines[index])
    if current_indent < indent:
        return None, index

    stripped = lines[index].strip()
    if stripped == "[]":
        return [], index + 1
    if stripped == "{}":
        return {}, index + 1
    if stripped == "-" or stripped.startswith("- "):
        return _parse_list(lines, index, indent, resolve_env=resolve_env)
    return _parse_mapping(lines, index, indent, resolve_env=resolve_env)


def _parse_mapping(lines: list[str], index: int, indent: int, *, resolve_env: bool) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while True:
        index = _skip_ignored(lines, index)
        if index >= len(lines):
            break

        line = lines[index]
        current_indent = _indent_of(line)
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation at line {index + 1}.")

        stripped = line.strip()
        if stripped == "-" or stripped.startswith("- "):
            break
        if ":" not in stripped:
            raise ValueError(f"Expected mapping entry at line {index + 1}.")

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            mapping[key] = _parse_scalar(raw_value, resolve_env=resolve_env)
            continue

        next_index = _skip_ignored(lines, index)
        if next_index >= len(lines) or _indent_of(lines[next_index]) <= indent:
            mapping[key] = None
            index = next_index
            continue

        nested, index = _parse_block(lines, index, indent + 2, resolve_env=resolve_env)
        mapping[key] = nested

    return mapping, index


def _parse_list(lines: list[str], index: int, indent: int, *, resolve_env: bool) -> tuple[list[Any], int]:
    items: list[Any] = []
    while True:
        index = _skip_ignored(lines, index)
        if index >= len(lines):
            break

        line = lines[index]
        current_indent = _indent_of(line)
        if current_indent < indent:
            break
        if current_indent != indent:
            raise ValueError(f"Unexpected list indentation at line {index + 1}.")

        stripped = line.strip()
        if stripped != "-" and not stripped.startswith("- "):
            break

        raw_value = "" if stripped == "-" else stripped[2:].strip()
        index += 1
        if not raw_value:
            nested, index = _parse_block(lines, index, indent + 2, resolve_env=resolve_env)
            items.append(nested)
            continue
        items.append(_parse_scalar(raw_value, resolve_env=resolve_env))

    return items, index


def _parse_scalar(value: str, *, resolve_env: bool) -> Any:
    if value.startswith('"') and value.endswith('"'):
        parsed = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        return _resolve_env(parsed) if resolve_env else parsed
    if value.startswith("'") and value.endswith("'"):
        parsed = value[1:-1]
        return _resolve_env(parsed) if resolve_env else parsed
    if value == "[]":
        return []
    if value == "{}":
        return {}

    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)

    return _resolve_env(value) if resolve_env else value


def _resolve_env(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        variable = match.group(1)
        default = match.group(3) or ""
        return os.environ.get(variable, default)

    return _ENV_PATTERN.sub(replace, value)


def _resolve_env_in_data(value: Any) -> Any:
    if isinstance(value, str):
        return _resolve_env(value)
    if isinstance(value, list):
        return [_resolve_env_in_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _resolve_env_in_data(item) for key, item in value.items()}
    return value


def _first_env_value(keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _optional_text(os.environ.get(key))
        if value is not None:
            return value
    return None


def _normalize_hpc_target_env_mapping(value: Any, *, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"HPC targets {label} must contain a mapping of environment variable names to strings.")

    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not _ENVIRONMENT_NAME_PATTERN.fullmatch(key):
            raise ValueError(f"HPC targets {label} contains an invalid environment variable name: {raw_key}")
        if not isinstance(raw_value, str):
            raise ValueError(f"HPC targets {label}.{key} must contain a string.")
        normalized_value = raw_value.strip()
        if normalized_value:
            normalized[key] = normalized_value
    return normalized


def _normalize_hpc_target_slurm(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"HPC targets {label} must contain a mapping when declared.")

    normalized: dict[str, Any] = {}
    if "modules" in value:
        normalized["modules"] = _normalize_string_list(value.get("modules"), label=f"{label}.modules")
    if "environment" in value:
        normalized["environment"] = _normalize_raw_string_mapping(value.get("environment"), label=f"{label}.environment")
    if "pre_activate_commands" in value:
        normalized["pre_activate_commands"] = _normalize_string_list(
            value.get("pre_activate_commands"),
            label=f"{label}.pre_activate_commands",
        )
    if "prepare_directories" in value:
        normalized["prepare_directories"] = _normalize_raw_string_list(
            value.get("prepare_directories"),
            label=f"{label}.prepare_directories",
        )
    if "omit_mem_directive" in value:
        normalized["omit_mem_directive"] = _normalize_bool_field(
            value.get("omit_mem_directive"),
            label=f"{label}.omit_mem_directive",
        )
    for field in _HPC_TARGET_SLURM_DIRECTIVE_FIELDS:
        if field not in value:
            continue
        normalized_value = _optional_text(value.get(field))
        if normalized_value is not None:
            normalized[field] = normalized_value
    return normalized


def _hpc_target_warnings(target: dict[str, Any], *, workspace_root: Path) -> list[str]:
    warnings: list[str] = []
    ssh_profile = _optional_text(target.get("ssh_profile"))
    ssh_config = _optional_text(target.get("ssh_config"))
    if ssh_profile is not None:
        if ssh_config is None:
            warnings.append(f"Target {target['name']} declares ssh_profile but no ssh_config.")
        else:
            ssh_config_path = Path(ssh_config).expanduser()
            if not ssh_config_path.is_absolute():
                ssh_config_path = workspace_root / ssh_config_path
            if not ssh_config_path.exists():
                warnings.append(f"SSH config is missing: {ssh_config_path}")
            elif not _ssh_profile_declared(ssh_config_path, ssh_profile):
                warnings.append(f"SSH profile {ssh_profile!r} is not declared in {ssh_config_path}.")

    for scope, values in (
        ("env", target.get("env", {})),
        ("project env", target.get("project_env", {})),
        ("slurm", target.get("slurm", {})),
    ):
        for placeholder in _iter_unresolved_placeholders(values):
            warnings.append(f"Target {target['name']} {scope} contains unresolved placeholder: {placeholder}")
    return warnings


def _ssh_profile_declared(config_path: Path, profile_name: str) -> bool:
    try:
        document = load_yaml(config_path)
    except Exception:
        return False
    profiles = document.get("profiles") if isinstance(document, dict) else None
    return isinstance(profiles, dict) and profile_name in profiles


def _iter_unresolved_placeholders(value: Any) -> list[str]:
    placeholders: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            placeholders.extend(_iter_unresolved_placeholders(item))
    elif isinstance(value, list):
        for item in value:
            placeholders.extend(_iter_unresolved_placeholders(item))
    elif isinstance(value, str):
        placeholders.extend(match.group(0) for match in _PLACEHOLDER_PATTERN.finditer(value))
    return placeholders


def _has_slurm_field_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return bool(value)
    return _optional_text(value) is not None


def _normalize_data(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_data(item) for item in value]
    return value


def _resolve_project_declared_path(value: str | Path, *, workspace_root: str | Path, project_root: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    workspace_root_path = Path(workspace_root).resolve()
    project_root_path = Path(project_root).resolve()
    workspace_candidate = (workspace_root_path / candidate).resolve()
    if workspace_candidate.exists():
        return workspace_candidate
    project_candidate = (project_root_path / candidate).resolve()
    if project_candidate.exists():
        return project_candidate

    try:
        project_root_relative = project_root_path.relative_to(workspace_root_path)
    except ValueError:
        project_root_relative = None

    if project_root_relative is not None:
        project_root_parts = project_root_relative.parts
        if candidate.parts[: len(project_root_parts)] == project_root_parts:
            return workspace_candidate

    return project_candidate


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_bool_field(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{label} must contain true or false when declared.")
    normalized = text.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{label} must contain true or false when declared.")


def _normalize_optional_remote_root(value: Any) -> str | None:
    resolved = resolve_env_value(value)
    if resolved is None:
        return None
    return str(PurePosixPath(resolved))


def _normalize_string_list(value: Any, *, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            normalized = _optional_text(item)
            if normalized is not None:
                values.append(normalized)
        return values
    raise ValueError(f"WORKSPACE.yaml {label} must contain a string or list of strings when declared.")


def _normalize_raw_string_mapping(value: Any, *, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"WORKSPACE.yaml {label} must contain a mapping of strings when declared.")

    normalized: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name).strip()
        if not _ENVIRONMENT_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"WORKSPACE.yaml {label} contains an invalid environment variable name: {raw_name}")
        if not isinstance(raw_value, str):
            raise ValueError(f"WORKSPACE.yaml {label}.{name} must contain a non-empty string.")
        normalized_value = raw_value.strip()
        if not normalized_value:
            raise ValueError(f"WORKSPACE.yaml {label}.{name} must contain a non-empty string.")
        normalized[name] = normalized_value
    return normalized


def _normalize_raw_string_list(value: Any, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"WORKSPACE.yaml {label} must contain a list of non-empty strings when declared.")

    normalized: list[str] = []
    for index, raw_item in enumerate(value):
        if not isinstance(raw_item, str):
            raise ValueError(f"WORKSPACE.yaml {label}[{index}] must contain a non-empty string.")
        normalized_item = raw_item.strip()
        if not normalized_item:
            raise ValueError(f"WORKSPACE.yaml {label}[{index}] must contain a non-empty string.")
        normalized.append(normalized_item)
    return normalized


def _dedupe_paths(values: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for value in values:
        resolved = value.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _can_merge_compute_runtime_defaults(compute_defaults: Any) -> bool:
    if compute_defaults is None:
        return True
    if not isinstance(compute_defaults, dict):
        return False

    slurm = compute_defaults.get("slurm")
    if slurm is None:
        return True
    if not isinstance(slurm, dict):
        return False

    if "environment" in slurm:
        try:
            _normalize_raw_string_mapping(slurm.get("environment"), label="config/compute.yaml compute.slurm.environment")
        except ValueError:
            return False
    if "prepare_directories" in slurm:
        try:
            _normalize_raw_string_list(
                slurm.get("prepare_directories"),
                label="config/compute.yaml compute.slurm.prepare_directories",
            )
        except ValueError:
            return False
    return True


def _merge_ordered_unique_string_lists(*values: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _normalize_mergeable_string_list(value):
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def _merge_ordered_unique_prepare_directories(*values: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _normalize_mergeable_prepare_directories(value):
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def _merge_string_mapping(*values: Any) -> dict[str, str]:
    merged: dict[str, str] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            if not key or not isinstance(raw_value, str):
                continue
            normalized_value = raw_value.strip()
            if normalized_value:
                merged[key] = normalized_value
    return merged


def _normalize_mergeable_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, (list, tuple)):
        normalized_values: list[str] = []
        for item in value:
            normalized = str(item).strip()
            if normalized:
                normalized_values.append(normalized)
        return normalized_values
    normalized = str(value).strip()
    return [normalized] if normalized else []


def _normalize_mergeable_prepare_directories(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        normalized_values: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if normalized:
                normalized_values.append(normalized)
        return normalized_values
    return []


def _dump_yaml_lines(value: Any, *, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                if not item:
                    lines.append(f"{prefix}{key}: {{}}")
                    continue
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_yaml_lines(item, indent=indent + 2))
                continue
            if isinstance(item, list):
                if not item:
                    lines.append(f"{prefix}{key}: []")
                    continue
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, dict):
                if not item:
                    lines.append(f"{prefix}- {{}}")
                    continue
                lines.append(f"{prefix}-")
                lines.extend(_dump_yaml_lines(item, indent=indent + 2))
                continue
            if isinstance(item, list):
                if not item:
                    lines.append(f"{prefix}- []")
                    continue
                lines.append(f"{prefix}-")
                lines.extend(_dump_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return lines
    return [f"{prefix}{_format_scalar(value)}"]


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text:
        return '""'
    if text.lower() in {"true", "false", "null", "~"}:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if re.fullmatch(r"-?\d+", text) or re.fullmatch(r"-?\d+\.\d+", text):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if _SAFE_SCALAR.fullmatch(text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
