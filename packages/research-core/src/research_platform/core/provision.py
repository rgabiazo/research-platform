"""Slice-aware HPC provision planning helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import path_exists_within_workspace, to_workspace_relative
from .tool_adapters import load_bids_tool_adapter

_GLOBAL_EXCLUDE = "ops/sync/rsync/exclude.txt"
_COMMON_EXCLUDE = "ops/sync/rsync/exclude.common.txt"
_PROJECT_OVERLAY_EXCLUDE = "ops/sync/rsync/exclude.project-overlay.txt"
_NEURO_BIDS_EXCLUDE = "ops/sync/rsync/exclude.neuro-bids.txt"
_TABULAR_ML_EXCLUDE = "ops/sync/rsync/exclude.tabular-ml.txt"
_WORKSPACE_EXCLUDE = "ops/sync/rsync/exclude.workspace.txt"


def build_provision_plan(*, context: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    workspace_root = Path(context["workspace_root"]).resolve()
    scopes = [build_common_sync_scope(context=context, manifest=manifest)]
    if manifest["slice"] == "bids":
        scopes.append(build_bids_tool_sync_scope(context=context, manifest=manifest))
    elif manifest["slice"] == "tabular":
        scopes.append(build_tabular_ml_sync_scope(context=context, manifest=manifest))
    else:
        raise ValueError(f"Unsupported provision slice: {manifest['slice']}")
    selected_scopes = [scope for scope in scopes if scope["entries"]]
    return {
        "phase": "2A",
        "kind": "workspace-sync-plan",
        "remote_workspace_root": manifest.get("hpc", {}).get("remote_workspace_root", ""),
        "selected_scopes": [scope["name"] for scope in selected_scopes],
        "scopes": selected_scopes,
    }


def build_project_sync_plan(*, context: dict[str, Any]) -> dict[str, Any]:
    manifest = {"slice": context.get("slice", "overlay")}
    entries = list(build_common_sync_scope(context=context, manifest=manifest)["entries"])
    slice_name = context.get("slice")
    if slice_name == "bids":
        entries.extend(
            entry
            for entry in build_bids_tool_sync_scope(context=context, manifest=manifest)["entries"]
            if entry.get("sync_scope") == "project"
        )
    elif slice_name == "tabular":
        entries.extend(
            entry
            for entry in build_tabular_ml_sync_scope(context=context, manifest=manifest)["entries"]
            if entry.get("sync_scope") == "project"
        )
    return {
        "kind": "project-sync-plan",
        "entries": entries,
    }


def build_project_data_sync_plan(*, context: dict[str, Any]) -> dict[str, Any]:
    workspace_root = Path(context["workspace_root"]).resolve()
    entries: list[dict[str, Any]] = []
    for root_spec in _dedupe_nested_root_specs(context.get("data_roots", [])):
        if not root_spec.get("sync_enabled", True):
            continue
        entry = build_data_sync_entry(
            workspace_root=workspace_root,
            source=root_spec["path"],
            remote_root=root_spec.get("remote_root"),
            exclude_files=_data_sync_exclude_files(label=str(root_spec["label"]), workspace_root=workspace_root),
            label=str(root_spec["label"]),
        )
        if entry:
            entries.append(entry)
    return {
        "kind": "data-sync-plan",
        "entries": entries,
    }


def build_workspace_sync_plan(
    *,
    workspace_root: str | Path,
    remote_workspace_root: str,
    extra_exclude_file: str | Path | None = None,
) -> dict[str, Any]:
    workspace_root_path = Path(workspace_root).resolve()
    return {
        "kind": "workspace-sync-plan",
        "source": workspace_root_path,
        "destination": str(remote_workspace_root),
        "exclude_files": resolve_workspace_sync_exclude_files(
            workspace_root_path,
            extra_exclude_file=extra_exclude_file,
        ),
    }


def build_common_sync_scope(*, context: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    workspace_root = Path(context["workspace_root"]).resolve()
    common_excludes = exclude_file_paths(workspace_root, _GLOBAL_EXCLUDE, _COMMON_EXCLUDE)
    project_overlay_excludes = exclude_file_paths(workspace_root, _GLOBAL_EXCLUDE, _PROJECT_OVERLAY_EXCLUDE)
    entries = [
        build_scope_entry(
            workspace_root=workspace_root,
            source=workspace_root / "WORKSPACE.yaml",
            kind="file",
            exclude_files=exclude_file_paths(workspace_root, _GLOBAL_EXCLUDE),
            label="workspace-config",
        ),
        build_scope_entry(
            workspace_root=workspace_root,
            source=context["project_root"],
            exclude_files=project_overlay_excludes,
            label="project-overlay",
        ),
        build_scope_entry(
            workspace_root=workspace_root,
            source=workspace_root / "ops",
            exclude_files=common_excludes,
            label="ops-root",
        ),
        build_scope_entry(
            workspace_root=workspace_root,
            source=workspace_root / "packages" / "research-core",
            exclude_files=common_excludes,
            label="research-core",
        ),
        build_scope_entry(
            workspace_root=workspace_root,
            source=workspace_root / "packages" / "research-hpc",
            exclude_files=common_excludes,
            label="research-hpc",
        ),
    ]
    return {
        "name": "common",
        "description": "Shared workspace configuration and reusable core package payload.",
        "entries": [entry for entry in entries if entry],
    }


def build_bids_tool_sync_scope(*, context: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    workspace_root = Path(context["workspace_root"]).resolve()
    dataset_excludes = exclude_file_paths(workspace_root, _GLOBAL_EXCLUDE, _NEURO_BIDS_EXCLUDE)
    remote_dataset_root = context.get("remote_dataset_root")
    remote_input_derivative_root = context.get("remote_input_derivative_root")
    entries = [
        _build_declared_scope_entry(workspace_root=workspace_root, entry=entry)
        for entry in _bids_tool_sync_entries(context=context)
        if str(entry.get("sync_scope", "project")) != "data"
    ]
    if not remote_dataset_root:
        entries.append(
            build_scope_entry(
                workspace_root=workspace_root,
                source=context["dataset_root"],
                exclude_files=dataset_excludes,
                label="raw-dataset-root",
                sync_scope="data",
            )
        )
    if context.get("requires_input_derivative", True) and not remote_input_derivative_root:
        entries.append(
            build_scope_entry(
                workspace_root=workspace_root,
                source=context["input_derivative_root"],
                exclude_files=exclude_file_paths(workspace_root, _GLOBAL_EXCLUDE),
                label="input-derivative-root",
                sync_scope="data",
            )
        )
    if context.get("analysis"):
        entries.extend(_analysis_data_root_entries(context=context, workspace_root=workspace_root))
    return {
        "name": "neuro-bids",
        "description": "Configured BIDS tool code plus dataset and derivative payload.",
        "entries": [entry for entry in entries if entry],
    }


def build_tabular_ml_sync_scope(*, context: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    workspace_root = Path(context["workspace_root"]).resolve()
    package_excludes = exclude_file_paths(workspace_root, _GLOBAL_EXCLUDE, _TABULAR_ML_EXCLUDE)
    entries = [
        build_scope_entry(
            workspace_root=workspace_root,
            source=workspace_root / "packages" / "research-analysis",
            exclude_files=package_excludes,
            label="research-analysis",
        ),
        build_scope_entry(
            workspace_root=workspace_root,
            source=workspace_root / "packages" / "research-ml",
            exclude_files=package_excludes,
            label="research-ml",
        ),
        build_scope_entry(
            workspace_root=workspace_root,
            source=context["feature_table_path"],
            kind="file",
            exclude_files=exclude_file_paths(workspace_root, _GLOBAL_EXCLUDE),
            label="feature-table",
            sync_scope="data",
        ),
    ]
    return {
        "name": "tabular-ml",
        "description": "Slice-selected tabular feature table plus analysis and ML package payload.",
        "entries": [entry for entry in entries if entry],
    }


def build_scope_entry(
    *,
    workspace_root: str | Path,
    source: str | Path,
    kind: str = "directory",
    exclude_files: list[str] | None = None,
    label: str,
    destination: str | None = None,
    sync_scope: str = "project",
) -> dict[str, Any] | None:
    workspace_root_path = Path(workspace_root).resolve()
    source_path = Path(source).resolve()
    if not path_exists_within_workspace(source_path, workspace_root_path):
        return None
    if kind not in {"directory", "file"}:
        raise ValueError(f"Unsupported sync entry kind: {kind}")
    relative_source = to_workspace_relative(source_path, workspace_root_path)
    return {
        "label": label,
        "kind": kind,
        "source": relative_source,
        "destination": destination or relative_source,
        "exclude_files": list(exclude_files or []),
        "sync_scope": sync_scope,
    }


def build_data_sync_entry(
    *,
    workspace_root: str | Path,
    source: str | Path,
    remote_root: str | None,
    exclude_files: list[str] | None,
    label: str,
) -> dict[str, Any] | None:
    workspace_root_path = Path(workspace_root).resolve()
    source_path = Path(source).resolve()
    if not source_path.exists():
        return None

    destination = remote_root
    if path_exists_within_workspace(source_path, workspace_root_path):
        return build_scope_entry(
            workspace_root=workspace_root_path,
            source=source_path,
            exclude_files=exclude_files,
            label=label,
            destination=destination,
            sync_scope="data",
        )

    if not destination:
        return None

    return {
        "label": label,
        "kind": "directory",
        "source": str(source_path),
        "destination": str(destination),
        "exclude_files": list(exclude_files or []),
        "sync_scope": "data",
    }


def _bids_tool_sync_entries(*, context: dict[str, Any]) -> list[dict[str, Any]]:
    adapter = context.get("tool_adapter")
    if adapter is None:
        adapter = load_bids_tool_adapter(context["preprocessing"])
    return adapter.sync_entries(
        workspace_root=str(Path(context["workspace_root"]).resolve()),
        context=context,
    )


def _build_declared_scope_entry(*, workspace_root: Path, entry: dict[str, Any]) -> dict[str, Any] | None:
    exclude_files = [_resolve_workspace_path(workspace_root, value) for value in entry.get("exclude_files", [])]
    source = _resolve_workspace_path(workspace_root, entry["source"])
    kind = str(entry.get("kind", "directory"))
    return build_scope_entry(
        workspace_root=workspace_root,
        source=source,
        kind=kind,
        exclude_files=[to_workspace_relative(path, workspace_root) for path in exclude_files if path.exists()],
        label=str(entry["label"]),
        destination=entry.get("destination"),
        sync_scope=str(entry.get("sync_scope", "project")),
    )


def _resolve_workspace_path(workspace_root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root / candidate).resolve()


def _analysis_data_root_entries(*, context: dict[str, Any], workspace_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_paths = {
        Path(context["dataset_root"]).resolve(),
        Path(context["input_derivative_root"]).resolve(),
    }
    for root_spec in _dedupe_nested_root_specs(context.get("data_roots", [])):
        resolved_path = Path(root_spec["path"]).resolve()
        if resolved_path in seen_paths:
            continue
        if not root_spec.get("sync_enabled", True):
            continue
        entry = build_data_sync_entry(
            workspace_root=workspace_root,
            source=resolved_path,
            remote_root=root_spec.get("remote_root"),
            exclude_files=_data_sync_exclude_files(label=str(root_spec["label"]), workspace_root=workspace_root),
            label=str(root_spec["label"]),
        )
        if entry is not None:
            entries.append(entry)
    return entries


def exclude_file_paths(workspace_root: str | Path, *relative_paths: str) -> list[str]:
    workspace_root_path = Path(workspace_root).resolve()
    resolved: list[str] = []
    for relative_path in relative_paths:
        candidate = workspace_root_path / relative_path
        if candidate.exists():
            resolved.append(to_workspace_relative(candidate, workspace_root_path))
    return resolved


def _data_sync_exclude_files(*, label: str, workspace_root: Path) -> list[str]:
    if label == "raw-dataset-root":
        return exclude_file_paths(workspace_root, _GLOBAL_EXCLUDE, _NEURO_BIDS_EXCLUDE)
    return exclude_file_paths(workspace_root, _GLOBAL_EXCLUDE)


def resolve_workspace_sync_exclude_files(
    workspace_root: str | Path,
    *,
    extra_exclude_file: str | Path | None = None,
) -> list[Path]:
    workspace_root_path = Path(workspace_root).resolve()
    default_exclude_file = workspace_root_path / _WORKSPACE_EXCLUDE
    if not default_exclude_file.exists():
        raise FileNotFoundError(f"Missing tracked workspace exclude file: {default_exclude_file}")
    resolved = [default_exclude_file]
    if extra_exclude_file is not None:
        extra_path = Path(extra_exclude_file).expanduser()
        if not extra_path.is_absolute():
            extra_path = (workspace_root_path / extra_path).resolve()
        else:
            extra_path = extra_path.resolve()
        if not extra_path.exists():
            raise FileNotFoundError(f"Extra exclude file was not found: {extra_path}")
        resolved.append(extra_path)
    return resolved


def _dedupe_nested_root_specs(root_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for root_spec in root_specs:
        normalized_root_spec = dict(root_spec, path=Path(root_spec["path"]).resolve())
        if any(_same_root_sync_target(normalized_root_spec, selected_root_spec) for selected_root_spec in selected):
            continue
        if not _preserve_nested_root_spec(normalized_root_spec) and any(
            _is_relative_to(Path(normalized_root_spec["path"]), Path(selected_root_spec["path"]))
            for selected_root_spec in selected
        ):
            continue
        selected.append(normalized_root_spec)
    return selected


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _preserve_nested_root_spec(root_spec: dict[str, Any]) -> bool:
    return bool(root_spec.get("preserve_nested_sync_target"))


def _same_root_sync_target(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return Path(left["path"]) == Path(right["path"]) and str(left.get("remote_root") or "") == str(right.get("remote_root") or "")
