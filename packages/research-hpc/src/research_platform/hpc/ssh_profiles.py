"""Named SSH profile loading for reusable HPC access helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any

from ._yaml import expand_env_placeholders, load_yaml

_PROFILE_FIELDS = (
    "host",
    "user",
    "port",
    "ssh_config_host",
    "identity_file",
    "known_hosts_file",
    "options",
)
_DEFAULT_SSH_CONFIG_PATH = Path("secrets") / "hpc" / "ssh-profiles.yaml"
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SSH_ENDPOINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\[\]-]*$")
_SSH_USER = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_ALLOWED_SSH_OPTIONS = frozenset(
    {
        "connectionattempts",
        "connecttimeout",
        "controlmaster",
        "controlpath",
        "controlpersist",
        "preferredauthentications",
        "serveralivecountmax",
        "serveraliveinterval",
        "tcpkeepalive",
    }
)


@dataclass(frozen=True)
class SshProfile:
    """Normalized SSH profile suitable for command rendering."""

    name: str
    role: str = "login"
    host: str | None = None
    user: str | None = None
    port: int | None = None
    ssh_config_host: str | None = None
    identity_file: str | None = None
    known_hosts_file: str | None = None
    options: dict[str, str] = field(default_factory=dict)

    def target(self) -> str:
        if self.ssh_config_host:
            alias = _validate_ssh_endpoint_syntax(
                self.ssh_config_host,
                label="SSH config alias",
            )
            if self.user:
                return f"{_validate_ssh_user_syntax(self.user)}@{alias}"
            return alias
        if not self.host:
            raise ValueError(f"SSH profile {self.name!r} must define host or ssh_config_host.")
        host = _validate_ssh_endpoint_syntax(self.host, label="SSH host")
        if self.user:
            return f"{_validate_ssh_user_syntax(self.user)}@{host}"
        return host

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "host": self.host,
            "user": self.user,
            "port": self.port,
            "ssh_config_host": self.ssh_config_host,
            "identity_file": self.identity_file,
            "known_hosts_file": self.known_hosts_file,
            "options": dict(self.options),
        }


def load_ssh_profile(config_path: str | Path, profile_name: str, role: str = "login") -> SshProfile:
    config = _load_ssh_profile_config(config_path)
    profiles = _require_mapping(config.get("profiles"), message="SSH profile config must define profiles.")
    entry = profiles.get(profile_name)
    if not isinstance(entry, dict):
        raise ValueError(f"SSH profile {profile_name!r} was not found.")
    defaults = _require_mapping(config.get("defaults", {}), message="SSH profile defaults must be a mapping.")
    resolved = _resolve_profile_payload(profile_name, entry=entry, role=role, defaults=defaults)
    return _parse_profile(profile_name, resolved, role=role)


def list_ssh_profiles(config_path: str | Path) -> list[dict[str, Any]]:
    config = _load_ssh_profile_config(config_path)
    profiles = _require_mapping(config.get("profiles"), message="SSH profile config must define profiles.")
    catalog: list[dict[str, Any]] = []
    for name in sorted(profiles):
        entry = profiles[name]
        if not isinstance(entry, dict):
            raise ValueError(f"SSH profile {name!r} must be a mapping.")
        roles = entry.get("roles")
        available_roles = sorted(str(key) for key in roles) if isinstance(roles, dict) and roles else ["login"]
        catalog.append(
            {
                "name": str(name),
                "kind": "family" if "roles" in entry or "defaults" in entry else "flat",
                "roles": available_roles,
            }
        )
    return catalog


def build_ssh_profile_entry(
    *,
    host: str,
    user: str,
    port: int | None = None,
    identity_file: str | None = None,
    known_hosts_file: str | None = None,
    options: dict[str, str | int | bool] | None = None,
) -> dict[str, Any]:
    """Build one provider-neutral, self-contained SSH profile entry.

    Only explicitly supplied connection settings are emitted.  In particular,
    this helper does not choose an authentication method, key path, SSH
    multiplexing policy, provider, or role family.
    """

    normalized_host = validate_ssh_host_or_alias(host, label="SSH host")
    normalized_user = validate_ssh_user(user)
    entry: dict[str, Any] = {
        "host": normalized_host,
        "user": normalized_user,
    }
    if port is not None:
        entry["port"] = validate_ssh_port(port)
    normalized_identity_file = _optional_template_value(identity_file)
    if normalized_identity_file is not None:
        entry["identity_file"] = validate_ssh_path_reference(
            normalized_identity_file,
            label="SSH identity-file path",
        )
    normalized_known_hosts_file = _optional_template_value(known_hosts_file)
    if normalized_known_hosts_file is not None:
        entry["known_hosts_file"] = validate_ssh_path_reference(
            normalized_known_hosts_file,
            label="SSH known-hosts-file path",
        )
    if options is not None:
        entry["options"] = validate_ssh_options(options)
    return entry


def upsert_ssh_profile_document(
    document: dict[str, Any] | None,
    *,
    profile_name: str,
    profile: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Return a copy with one selected profile added or replaced.

    Existing unrelated defaults and profiles are preserved.  A selected
    profile with different content is a conflict unless ``force`` is true.
    """

    normalized_name = validate_ssh_profile_name(profile_name)
    if not isinstance(profile, dict):
        raise ValueError(f"SSH profile {normalized_name!r} must be a mapping.")
    if document is None:
        updated: dict[str, Any] = {}
    elif isinstance(document, dict):
        updated = deepcopy(document)
    else:
        raise ValueError("SSH profile config must contain a top-level mapping.")

    profiles = updated.get("profiles")
    if profiles is None:
        profiles = {}
        updated["profiles"] = profiles
    if not isinstance(profiles, dict):
        raise ValueError("SSH profile config must define profiles as a mapping.")

    existing = profiles.get(normalized_name)
    if existing is not None and existing != profile and not force:
        raise ValueError(
            f"SSH profile {normalized_name!r} already exists with different settings; "
            "use --force to replace only that profile."
        )
    profiles[normalized_name] = deepcopy(profile)
    return updated


def require_generic_profile_isolation(document: dict[str, Any] | None) -> None:
    """Reject generic insertion when document-wide connection defaults exist.

    The established loader merges top-level defaults into every profile.
    Keeping those defaults is necessary for unrelated profiles, but allowing a
    new generic profile to inherit them would silently add provider,
    authentication, identity, or multiplexing assumptions.  A separate
    destination is therefore required instead of rewriting unrelated defaults.
    """

    if document is None:
        return
    if not isinstance(document, dict):
        raise ValueError("SSH profile config must contain a top-level mapping.")
    defaults = document.get("defaults")
    if defaults in (None, {}):
        return
    raise ValueError(
        "Generic SSH setup cannot use a config with nonempty top-level defaults "
        "because they would be inherited implicitly; choose a separate SSH config "
        "path or remove those defaults deliberately before setup."
    )


def materialize_ssh_profile_entry(
    template: dict[str, Any],
    *,
    profile_name: str,
) -> dict[str, Any]:
    """Return one self-contained profile from a template document.

    Template-wide defaults are folded into the selected profile so callers can
    safely insert it into an existing document without changing defaults that
    may affect unrelated profiles.
    """

    normalized_name = validate_ssh_profile_name(profile_name)
    if not isinstance(template, dict):
        raise ValueError("SSH template must contain a top-level mapping.")
    profiles = template.get("profiles")
    if not isinstance(profiles, dict) or not isinstance(profiles.get(normalized_name), dict):
        raise ValueError(f"SSH template did not define selected profile {normalized_name!r}.")
    profile = deepcopy(profiles[normalized_name])
    shared_defaults = template.get("defaults")
    if shared_defaults is not None and not isinstance(shared_defaults, dict):
        raise ValueError("SSH template defaults must be a mapping.")
    if shared_defaults:
        profile_defaults = profile.get("defaults")
        if profile_defaults is not None and not isinstance(profile_defaults, dict):
            raise ValueError(f"SSH profile {normalized_name!r} defaults must be a mapping.")
        profile["defaults"] = _merge_mappings(shared_defaults, profile_defaults or {})
    return profile


def build_ssh_config_template(
    template_name: str = "generic",
    *,
    profile_name: str | None = None,
    host: str | None = None,
    user: str | None = None,
    port: int | None = None,
    identity_file: str | None = None,
    known_hosts_file: str | None = None,
    options: dict[str, str | int | bool] | None = None,
) -> dict[str, Any]:
    """Build a generic private config or an explicit Alliance starter.

    The generic template is intentionally concrete: callers must supply a
    real host and user instead of receiving an invented example hostname or
    unresolved provider placeholder.
    """

    normalized_template = str(template_name).strip().lower()
    if normalized_template == "generic":
        if host is None or user is None:
            raise ValueError("The generic SSH template requires explicit host and user values.")
        normalized_profile = validate_ssh_profile_name(_optional_template_value(profile_name) or "generic")
        return {
            "profiles": {
                normalized_profile: build_ssh_profile_entry(
                    host=host,
                    user=user,
                    port=port,
                    identity_file=identity_file,
                    known_hosts_file=known_hosts_file,
                    options=options,
                )
            }
        }
    if normalized_template != "alliance":
        raise ValueError(f"Unsupported SSH config template: {template_name!r}")

    alliance_profile_name = validate_ssh_profile_name(_optional_template_value(profile_name) or "alliance")
    explicit_user = _optional_template_value(user)
    if explicit_user is not None:
        explicit_user = validate_ssh_user(explicit_user)
    explicit_identity_file = _optional_template_value(identity_file)
    if explicit_identity_file is not None:
        explicit_identity_file = validate_ssh_path_reference(
            explicit_identity_file,
            label="SSH identity-file path",
        )
    defaults: dict[str, Any] = {
        "user": explicit_user or "${HPC_USER:-your-username}",
        "identity_file": explicit_identity_file or "${HPC_IDENTITY_FILE:-~/.ssh/id_ed25519}",
        "options": {
            "ServerAliveInterval": 30,
            "PreferredAuthentications": "publickey,keyboard-interactive",
            "ControlMaster": "auto",
            "ControlPath": "~/.ssh/cm-%C",
            "ControlPersist": "2h",
        },
    }
    if options:
        defaults["options"].update(validate_ssh_options(options))
    if known_hosts_file is not None:
        defaults["known_hosts_file"] = validate_ssh_path_reference(
            known_hosts_file,
            label="SSH known-hosts-file path",
        )
    explicit_host = _optional_template_value(host)
    if explicit_host is not None:
        explicit_host = validate_ssh_host_or_alias(explicit_host, label="SSH host")
    alliance_defaults: dict[str, Any] = {
        "host": explicit_host or "${ALLIANCE_LOGIN_HOST:-login.cluster.example}",
    }
    if port is not None:
        alliance_defaults["port"] = validate_ssh_port(port)
    return {
        "defaults": defaults,
        "profiles": {
            alliance_profile_name: {
                "defaults": alliance_defaults,
                "roles": {
                    "login": {},
                    "robot": {
                        "user": "${ALLIANCE_ROBOT_USER:-${HPC_USER:-automation}}",
                        "identity_file": "${ALLIANCE_ROBOT_IDENTITY_FILE:-${HPC_IDENTITY_FILE:-~/.ssh/id_ed25519}}",
                    },
                },
            }
        },
    }


def validate_ssh_profile_name(value: object) -> str:
    """Validate a profile selector without normalizing unsafe syntax."""

    normalized = _require_nonblank_template_value(value, label="SSH profile name")
    if not _PROFILE_NAME.fullmatch(normalized):
        raise ValueError(
            "SSH profile name must start with an alphanumeric character and contain "
            "only alphanumerics, '.', '_', or '-'."
        )
    _reject_starter_placeholder(normalized, label="SSH profile name")
    return normalized


def validate_ssh_host_or_alias(value: object, *, label: str = "SSH host or alias") -> str:
    """Validate a concrete direct host or SSH-config alias."""

    normalized = _require_nonblank_template_value(value, label=label)
    _reject_starter_placeholder(normalized, label=label)
    return _validate_ssh_endpoint_syntax(normalized, label=label)


def validate_ssh_user(value: object) -> str:
    """Validate one explicit SSH user identifier."""

    normalized = _require_nonblank_template_value(value, label="SSH user")
    _reject_starter_placeholder(normalized, label="SSH user")
    return _validate_ssh_user_syntax(normalized)


def _validate_ssh_endpoint_syntax(value: object, *, label: str) -> str:
    normalized = _require_nonblank_template_value(value, label=label)
    if not _SSH_ENDPOINT.fullmatch(normalized):
        raise ValueError(
            f"{label} must start with an alphanumeric character and contain only "
            "hostname, address, or alias characters."
        )
    return normalized


def _validate_ssh_user_syntax(value: object) -> str:
    normalized = _require_nonblank_template_value(value, label="SSH user")
    if not _SSH_USER.fullmatch(normalized):
        raise ValueError(
            "SSH user must start with an alphanumeric character or '_' and contain "
            "only alphanumerics, '.', '_', or '-'."
        )
    return normalized


def validate_ssh_port(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError("SSH port must be an integer from 1 through 65535.")
    return value


def validate_ssh_path_reference(value: object, *, label: str) -> str:
    """Validate path syntax only; this function never opens the referenced file."""

    normalized = _require_nonblank_template_value(value, label=label)
    _reject_starter_placeholder(normalized, label=label)
    if normalized.startswith("-"):
        raise ValueError(f"{label} must not look like a command-line option.")
    if "\x00" in normalized:
        raise ValueError(f"{label} must not contain NUL.")
    return normalized


def validate_ssh_options(options: object) -> dict[str, str | int | bool]:
    """Return reviewed, non-routing OpenSSH options with original key casing."""

    if not isinstance(options, dict):
        raise ValueError("SSH profile options must be a mapping.")
    validated: dict[str, str | int | bool] = {}
    seen: set[str] = set()
    for raw_key, raw_value in options.items():
        if not isinstance(raw_key, str):
            raise ValueError("SSH option names must be strings.")
        key = _require_nonblank_template_value(raw_key, label="SSH option name")
        lowered = key.casefold()
        if lowered in seen:
            raise ValueError(f"SSH option {key!r} is duplicated case-insensitively.")
        seen.add(lowered)
        if lowered not in _ALLOWED_SSH_OPTIONS:
            raise ValueError(
                f"SSH option {key!r} is not permitted by the H1 non-routing option allowlist."
            )
        if isinstance(raw_value, (dict, list, tuple, set)) or raw_value is None:
            raise ValueError(f"SSH option {key!r} must have a scalar value.")
        if not isinstance(raw_value, (str, int, bool, float)):
            raise ValueError(f"SSH option {key!r} must have a scalar value.")
        rendered = str(raw_value)
        if not rendered.strip():
            raise ValueError(f"SSH option {key!r} must have a nonblank value.")
        if any(ord(character) < 32 or ord(character) == 127 for character in rendered):
            raise ValueError(f"SSH option {key!r} must not contain control characters.")
        _reject_starter_placeholder(rendered, label=f"SSH option {key!r}")
        validated[key] = deepcopy(raw_value)
    return validated


def _reject_starter_placeholder(value: str, *, label: str) -> None:
    lowered = value.casefold()
    if (
        "${" in value
        or ("<" in value and ">" in value)
        or lowered in {"your-username", "change-me"}
        or lowered.endswith(".example")
        or lowered.endswith(".example.org")
    ):
        raise ValueError(f"{label} must be a concrete private value, not a starter placeholder.")


def _require_nonblank_template_value(value: object, *, label: str) -> str:
    normalized = _optional_template_value(value)
    if normalized is None:
        raise ValueError(f"{label} must be a nonblank value.")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{label} must not contain control characters.")
    return normalized


def _require_concrete_template_value(value: object, *, label: str) -> str:
    normalized = _require_nonblank_template_value(value, label=label)
    _reject_starter_placeholder(normalized, label=label)
    return normalized


def _optional_template_value(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def resolve_ssh_profile_config_path(
    config_path: str | Path | None,
    *,
    workspace_root: str | Path | None = None,
) -> Path:
    candidate = config_path or os.environ.get("RESEARCH_HPC_SSH_CONFIG") or os.environ.get("RP_SSH_CONFIG")
    if candidate:
        return _resolve_config_candidate(candidate, workspace_root=workspace_root)
    fallback_path = _find_default_ssh_config_path(workspace_root=workspace_root)
    if fallback_path is not None:
        return fallback_path
    raise ValueError(
        "Provide --config, set RESEARCH_HPC_SSH_CONFIG or RP_SSH_CONFIG, or create secrets/hpc/ssh-profiles.yaml."
    )


def _load_ssh_profile_config(config_path: str | Path) -> dict[str, Any]:
    resolved_config_path = Path(config_path).expanduser().resolve()
    try:
        raw = load_yaml(resolved_config_path)
    except FileNotFoundError as exc:
        raise ValueError(f"SSH profile config was not found: {resolved_config_path}") from exc
    expanded = expand_env_placeholders(raw)
    return _require_mapping(expanded, message="SSH profile config must contain a top-level mapping.")


def _resolve_config_candidate(candidate: str | Path, *, workspace_root: str | Path | None) -> Path:
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path.resolve()
    anchor = _discover_workspace_root(workspace_root)
    if anchor is not None:
        return (anchor / path).resolve()
    return path.resolve()


def _find_default_ssh_config_path(*, workspace_root: str | Path | None = None) -> Path | None:
    candidate_roots: list[Path] = []
    discovered_workspace_root = _discover_workspace_root(workspace_root)
    if discovered_workspace_root is not None:
        candidate_roots.append(discovered_workspace_root)
    candidate_roots.extend((Path.cwd(), *Path.cwd().parents))

    seen: set[Path] = set()
    for root in candidate_roots:
        resolved_root = root.resolve()
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        candidate = root / _DEFAULT_SSH_CONFIG_PATH
        if candidate.exists():
            return candidate.resolve()
    return None


def _discover_workspace_root(workspace_root: str | Path | None) -> Path | None:
    if workspace_root is not None:
        return Path(workspace_root).expanduser().resolve()

    env_root = os.environ.get("RESEARCH_PLATFORM_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    for root in (Path.cwd(), *Path.cwd().parents):
        if (root / "WORKSPACE.yaml").exists():
            return root.resolve()
    return None


def _resolve_profile_payload(
    profile_name: str,
    *,
    entry: dict[str, Any],
    role: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    base_payload = _merge_mappings(defaults, _extract_profile_fields(entry))
    if "defaults" in entry:
        base_payload = _merge_mappings(
            base_payload,
            _require_mapping(entry.get("defaults"), message=f"SSH profile {profile_name!r} defaults must be a mapping."),
        )

    roles = entry.get("roles")
    if roles is None:
        return base_payload

    role_mapping = _require_mapping(roles, message=f"SSH profile {profile_name!r} roles must be a mapping.")
    role_payload = role_mapping.get(role)
    if role_payload is None:
        if _has_connection_settings(base_payload):
            return base_payload
        available_roles = ", ".join(sorted(str(name) for name in role_mapping))
        raise ValueError(f"SSH profile {profile_name!r} does not define role {role!r}. Available roles: {available_roles}")

    resolved_role_payload = _require_mapping(
        role_payload,
        message=f"SSH profile {profile_name!r} role {role!r} must be a mapping.",
    )
    return _merge_mappings(base_payload, resolved_role_payload)


def _parse_profile(name: str, payload: dict[str, Any], *, role: str) -> SshProfile:
    host = _optional_string(payload.get("host"))
    ssh_config_host = _optional_string(payload.get("ssh_config_host"))
    if host is None and ssh_config_host is None:
        raise ValueError(f"SSH profile {name!r} must define host or ssh_config_host.")

    port = payload.get("port")
    if port is not None:
        try:
            port = int(port)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SSH profile {name!r} has invalid port: {port!r}") from exc
        port = validate_ssh_port(port)

    options_raw = validate_ssh_options(payload.get("options") or {})

    return SshProfile(
        name=name,
        role=role,
        host=host,
        user=_optional_string(payload.get("user")),
        port=port,
        ssh_config_host=ssh_config_host,
        identity_file=_expand_optional_path(payload.get("identity_file")),
        known_hosts_file=_expand_optional_path(payload.get("known_hosts_file")),
        options={str(key): str(value) for key, value in options_raw.items()},
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _expand_optional_path(value: object) -> str | None:
    text = _optional_string(value)
    if text is None:
        return None
    return str(Path(os.path.expandvars(text)).expanduser())


def _extract_profile_fields(payload: dict[str, Any]) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    for field_name in _PROFILE_FIELDS:
        if field_name in payload:
            extracted[field_name] = deepcopy(payload[field_name])
    return extracted


def _has_connection_settings(payload: dict[str, Any]) -> bool:
    for field_name in _PROFILE_FIELDS:
        value = payload.get(field_name)
        if value is None:
            continue
        if isinstance(value, dict) and not value:
            continue
        if isinstance(value, list) and not value:
            continue
        return True
    return False


def _merge_mappings(*mappings: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            existing = merged.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = _merge_mappings(existing, value)
            else:
                merged[key] = deepcopy(value)
    return merged


def _require_mapping(value: Any, *, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(message)
    return value
