
"""Path resolution helpers."""

from __future__ import annotations

from pathlib import Path

from .run_lifecycle import resolved_run_path


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_path(base: str | Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(base) / path).resolve()


def workspace_paths(workspace_root: str | Path, workspace_config: dict) -> dict[str, Path]:
    raw_paths = workspace_config.get("paths", {})
    return {name: resolve_path(workspace_root, value) for name, value in raw_paths.items()}


def dataset_path(workspace_root: str | Path, workspace_config: dict, dataset_name: str) -> Path:
    configured = workspace_config.get("datasets", {}).get(dataset_name)
    if configured is not None:
        return resolve_path(workspace_root, configured)
    return workspace_paths(workspace_root, workspace_config)["datasets_root"] / dataset_name


def pipeline_path(workspace_root: str | Path, workspace_config: dict, pipeline_name: str) -> Path:
    pipelines_root = resolve_path(workspace_root, workspace_config.get("repos", {}).get("pipelines_root", "./pipelines"))
    return pipelines_root / pipeline_name


def project_path(workspace_root: str | Path, workspace_config: dict, project_name: str) -> Path:
    projects_root = resolve_path(workspace_root, workspace_config.get("repos", {}).get("project_root", "./project"))
    return projects_root / project_name


def run_path(artifacts_root: str | Path, run_id: str) -> Path:
    return resolved_run_path(artifacts_root, run_id)


def path_exists_within_workspace(path: str | Path, workspace_root: str | Path) -> bool:
    resolved_path = Path(path).resolve()
    resolved_root = Path(workspace_root).resolve()
    if not resolved_path.exists():
        return False
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return False
    return True


def to_workspace_relative(path: str | Path, workspace_root: str | Path) -> str:
    resolved_path = Path(path).resolve()
    resolved_root = Path(workspace_root).resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)
