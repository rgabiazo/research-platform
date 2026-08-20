"""Helpers for resolving profile-aware HPC connections from run manifests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from .ssh_profiles import SshProfile, load_ssh_profile, resolve_ssh_profile_config_path

_PROFILE_ENV_KEYS = ("RESEARCH_HPC_PROFILE", "RP_HPC_PROFILE")
_ROLE_ENV_KEYS = ("RESEARCH_HPC_ROLE", "RP_HPC_ROLE")
_CONFIG_ENV_KEYS = ("RESEARCH_HPC_SSH_CONFIG", "RP_SSH_CONFIG")
_MISSING_CONNECTION_ERROR = (
    "HPC connection is not configured. Provide --profile or set RESEARCH_HPC_PROFILE / RP_HPC_PROFILE."
)


@dataclass(frozen=True)
class ResolvedHpcConnection:
    """Connection settings ready for SSH and rsync command rendering."""

    kind: str
    ssh_target: str
    mode: str
    profile_name: str | None = None
    role: str | None = None
    config_path: Path | None = None
    profile: SshProfile | None = None


def build_manifest_hpc_connection_hint(
    *,
    profile_name: str | None = None,
    role: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, str] | None:
    resolved_profile = _normalize_optional(profile_name) or _env_value(_PROFILE_ENV_KEYS)
    if not resolved_profile:
        return None
    payload = {
        "kind": "ssh-profile",
        "profile": resolved_profile,
        "role": _normalize_optional(role) or _env_value(_ROLE_ENV_KEYS) or "login",
    }
    resolved_config = _normalize_optional(config_path) or _env_value(_CONFIG_ENV_KEYS)
    if resolved_config:
        payload["config"] = resolved_config
    return payload


def resolve_hpc_connection(
    *,
    manifest: dict[str, Any],
    profile_name: str | None = None,
    role: str | None = None,
    config_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> ResolvedHpcConnection:
    hpc = manifest.get("hpc", {}) if isinstance(manifest.get("hpc", {}), dict) else {}
    connection = hpc.get("connection", {}) if isinstance(hpc.get("connection", {}), dict) else {}
    ssh_host = _normalize_optional(hpc.get("ssh_host"))
    resolved_profile = _normalize_optional(profile_name) or _normalize_optional(connection.get("profile"))
    if resolved_profile is None and ssh_host is None:
        resolved_profile = _env_value(_PROFILE_ENV_KEYS)
    resolved_role = _normalize_optional(role) or _normalize_optional(connection.get("role")) or _env_value(_ROLE_ENV_KEYS) or "login"
    resolved_mode = "batch" if resolved_role == "robot" else "interactive"

    if resolved_profile:
        raw_config_path = _normalize_optional(config_path) or _normalize_optional(connection.get("config"))
        try:
            resolved_config_path = resolve_ssh_profile_config_path(raw_config_path, workspace_root=workspace_root)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        profile = load_ssh_profile(resolved_config_path, resolved_profile, role=resolved_role)
        return ResolvedHpcConnection(
            kind="ssh-profile",
            ssh_target=profile.target(),
            mode=resolved_mode,
            profile_name=profile.name,
            role=profile.role,
            config_path=resolved_config_path,
            profile=profile,
        )

    if ssh_host:
        return ResolvedHpcConnection(kind="ssh-host", ssh_target=ssh_host, mode=resolved_mode)
    raise ValueError(_MISSING_CONNECTION_ERROR)


def _env_value(keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _normalize_optional(os.environ.get(key))
        if value:
            return value
    return None


def _normalize_optional(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
