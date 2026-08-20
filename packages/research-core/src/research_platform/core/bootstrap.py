"""Generic bootstrap manifest planning for opt-in HPC runs."""

from __future__ import annotations

from typing import Any


def build_bootstrap_manifest(*, context: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any] | None:
    bootstrap_config = context.get("compute", {}).get("slurm", {}).get("bootstrap", {})
    if not _bootstrap_enabled(bootstrap_config):
        return None

    hook_scopes = _build_scope_hooks(manifest=manifest, bootstrap_config=bootstrap_config)
    return {
        "enabled": True,
        "kind": "remote-bootstrap-plan",
        "remote_directories": _build_directory_preparation(manifest=manifest),
        "hook_scopes": hook_scopes,
        "selected_scopes": [scope["name"] for scope in hook_scopes if scope["hooks"]],
    }


def _bootstrap_enabled(bootstrap_config: Any) -> bool:
    return isinstance(bootstrap_config, dict) and bool(bootstrap_config.get("enabled"))


def _build_directory_preparation(*, manifest: dict[str, Any]) -> list[dict[str, str]]:
    hpc = manifest.get("hpc", {})
    ordered = (
        ("remote-workspace-root", hpc.get("remote_workspace_root", "")),
        ("remote-artifacts-root", hpc.get("remote_artifacts_root", "")),
        ("remote-run-root", hpc.get("remote_run_root", "")),
    )
    prepared: list[dict[str, str]] = []
    seen: set[str] = set()
    for label, path in ordered:
        if not path or path in seen:
            continue
        seen.add(str(path))
        prepared.append({"label": label, "path": str(path)})
    return prepared


def _build_scope_hooks(*, manifest: dict[str, Any], bootstrap_config: dict[str, Any]) -> list[dict[str, Any]]:
    scope_config = bootstrap_config.get("hooks", {})
    scope_names = list(dict.fromkeys(manifest.get("provision", {}).get("selected_scopes", [])))
    hook_scopes: list[dict[str, Any]] = []
    for scope_name in scope_names:
        raw_hooks = scope_config.get(scope_name, [])
        if not isinstance(raw_hooks, list):
            raw_hooks = []
        hooks = [
            normalized
            for index, raw_hook in enumerate(raw_hooks, start=1)
            if (normalized := _normalize_hook(raw_hook=raw_hook, scope_name=scope_name, index=index)) is not None
        ]
        hook_scopes.append({"name": scope_name, "hooks": hooks})
    return hook_scopes


def _normalize_hook(*, raw_hook: Any, scope_name: str, index: int) -> dict[str, str] | None:
    if not isinstance(raw_hook, dict):
        return None
    command = str(raw_hook.get("command", "")).strip()
    if not command:
        return None
    name = str(raw_hook.get("name", f"{scope_name}-hook-{index}")).strip()
    kind = str(raw_hook.get("kind", "shell")).strip() or "shell"
    return {"name": name, "kind": kind, "command": command}
