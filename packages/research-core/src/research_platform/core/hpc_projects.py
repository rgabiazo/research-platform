"""Project-aware HPC resolution helpers for CLI wrappers."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

from .config import (
    apply_hpc_target_defaults,
    build_data_root_spec,
    default_project_name,
    merge_hpc_target_compute_defaults,
    load_yaml,
    load_project_bundle,
    load_project_record,
    load_workspace_config,
    merge_declared_data_roots,
    merge_workspace_hpc_runtime_compute_defaults,
    project_has_structured_config,
    project_slice,
    resolve_analysis_external_input_root_declarations,
    resolve_bids_dataset_root,
    resolve_bids_input_derivative_root,
    resolve_bids_remote_dataset_root,
    resolve_bids_remote_input_derivative_root,
    resolve_env_value,
    resolve_project_hpc_data_root_declarations,
    resolve_project_notebook_path,
    resolve_project_overlay_data_roots,
    resolve_workspace_hpc_runtime_default,
    validate_project_bundle,
    workspace_root,
)
from .paths import dataset_path, pipeline_path, to_workspace_relative, workspace_paths
from .tool_adapters import load_bids_analysis_tool_adapter, load_bids_tool_adapter


def build_workspace_hpc_context(project_name: str | None = None) -> dict[str, Any]:
    resolved_root = workspace_root()
    hpc_target = apply_hpc_target_defaults(project_name=None, root=resolved_root)
    workspace = load_workspace_config(resolved_root)
    resolved_project_name = project_name or default_project_name(workspace)
    hpc_target = apply_hpc_target_defaults(project_name=resolved_project_name, root=resolved_root) or hpc_target
    remote_workspace_root = _env_value("RP_REMOTE_WORKSPACE_ROOT")
    if remote_workspace_root is None:
        project_context = build_project_hpc_context(resolved_project_name)
        remote_workspace_root = project_context.get("remote_workspace_root")
    return {
        "workspace_root": resolved_root,
        "workspace": workspace,
        "paths": workspace_paths(resolved_root, workspace),
        "project_name": resolved_project_name,
        "remote_workspace_root": remote_workspace_root,
        "hpc_target": hpc_target,
    }


def build_project_hpc_context(project_name: str) -> dict[str, Any]:
    hpc_target = apply_hpc_target_defaults(project_name=project_name)
    record = load_project_record(project_name)
    hpc_target = record.get("hpc_target") or hpc_target
    workspace_root = Path(record["workspace_root"]).resolve()
    workspace = record["workspace"]
    project_root = Path(record["project_root"]).resolve()
    project_config = record["project"]
    runtime_default = resolve_workspace_hpc_runtime_default(workspace)
    notebook_path = resolve_project_notebook_path(project_config, workspace_root=workspace_root, project_root=project_root)
    context: dict[str, Any] = {
        "workspace_root": workspace_root,
        "workspace": workspace,
        "paths": workspace_paths(workspace_root, workspace),
        "project_root": project_root,
        "project": project_config,
        "project_kind": "overlay",
        "slice": "overlay",
        "compute": merge_hpc_target_compute_defaults(
            merge_workspace_hpc_runtime_compute_defaults({}, runtime_default),
            hpc_target,
        ),
        "hpc_target": hpc_target,
        "project_roots": [{"label": "project-root", "path": project_root}],
        "data_roots": _build_overlay_data_roots(
            project_config=project_config,
            workspace_root=workspace_root,
            workspace=workspace,
            project_root=project_root,
        ),
        "notebook_path": notebook_path,
        "validation_errors": [],
    }

    if project_has_structured_config(project_root):
        bundle = load_project_bundle(project_name, workspace_root)
        slice_name = project_slice(bundle)
        compute_config = bundle["compute"].get("compute", {})
        context.update(
            {
                "project_kind": "bundle",
                "slice": slice_name,
                "compute": compute_config,
                "dataset": bundle["dataset"].get("dataset", {}),
                "preprocessing": bundle["preprocessing"].get("preprocessing", {}),
                "analysis": bundle.get("analysis", {}).get("analysis", {}),
                "analysis_models": bundle.get("analysis_models", {}).get("models", {}),
                "models": bundle.get("models", {}).get("models", {}),
                "hpc_target": bundle.get("hpc_target") or hpc_target,
                "validation_errors": validate_project_bundle(bundle, root=workspace_root),
            }
        )
        if slice_name == "bids":
            validation_errors = list(context.get("validation_errors", []))
            pipeline_value = context["preprocessing"].get("pipeline") or context["analysis"].get("pipeline")
            pipeline_root = pipeline_path(workspace_root, workspace, str(pipeline_value)) if pipeline_value else None
            dataset_root = _safe_resolve_bids_path(resolve_bids_dataset_root, bundle, workspace_root)
            dataset_config = bundle["dataset"].get("dataset", {})
            remote_dataset_root = resolve_bids_remote_dataset_root(bundle)
            pipeline_defaults: dict[str, Any] = {}
            if pipeline_root is not None:
                pipeline_defaults_path = pipeline_root / "config" / "defaults.yaml"
                if pipeline_defaults_path.exists():
                    try:
                        pipeline_defaults = load_yaml(pipeline_defaults_path)
                    except Exception as exc:  # pragma: no cover - defensive
                        validation_errors.append(f"Unable to load pipeline defaults: {pipeline_defaults_path} ({exc})")
            tool_adapter = None
            requires_input_derivative = True
            if context["preprocessing"].get("tool_adapter") is not None:
                try:
                    tool_adapter = load_bids_tool_adapter(context["preprocessing"])
                except ValueError as exc:
                    validation_errors.append(str(exc))
                else:
                    requires_input_derivative = tool_adapter.requires_input_derivative()
            elif context["analysis"]:
                requires_input_derivative = True
                analysis_defaults = context["analysis"].get("defaults", {})
                analysis_stage_name = str(analysis_defaults.get("stage", "")).strip()
                analysis_stage = context["analysis"].get("stages", {}).get(analysis_stage_name, {})
                analysis_model_ref = str(
                    analysis_stage.get("model_ref") or analysis_defaults.get("model_ref") or ""
                ).strip()
                context["analysis_inputs"] = context["analysis"].get("inputs", {})
                context["analysis_stage_name"] = analysis_stage_name
                context["analysis_stage"] = analysis_stage if isinstance(analysis_stage, dict) else {}
                context["analysis_model_ref"] = analysis_model_ref
                context["analysis_model"] = (
                    context.get("analysis_models", {}).get(analysis_model_ref, {})
                    if analysis_model_ref
                    else {}
                )
                defaults = context["analysis"].get("defaults", {})
                tools = context["analysis"].get("tools", {})
                default_tool_name = str(defaults.get("tool", "")).strip()
                tool_entry = tools.get(default_tool_name)
                if isinstance(tool_entry, dict):
                    try:
                        tool_adapter = load_bids_analysis_tool_adapter(tool_entry)
                    except ValueError as exc:
                        validation_errors.append(str(exc))
            if context["analysis"]:
                requires_input_derivative = True
            if requires_input_derivative:
                input_derivative_root = _safe_resolve_bids_path(resolve_bids_input_derivative_root, bundle, workspace_root)
                remote_input_derivative_root = resolve_bids_remote_input_derivative_root(bundle)
            else:
                input_derivative_root = dataset_root
                remote_input_derivative_root = remote_dataset_root
            project_roots = [{"label": "project-root", "path": project_root}]
            if pipeline_root is not None:
                project_roots.append({"label": "pipeline-root", "path": pipeline_root})
            data_roots: list[dict[str, Any]] = []
            if dataset_root is not None:
                data_roots.append(
                    _root_spec(
                        "raw-dataset-root",
                        dataset_root,
                        remote_root=remote_dataset_root,
                        preserve_nested_sync_target=bool(resolve_env_value(dataset_config.get("remote_bids_root"))),
                    )
                )
            if requires_input_derivative and input_derivative_root is not None:
                data_roots.append(
                    _root_spec(
                        "input-derivative-root",
                        input_derivative_root,
                        remote_root=remote_input_derivative_root,
                        preserve_nested_sync_target=bool(
                            resolve_env_value(dataset_config.get("remote_input_derivative_root"))
                        ),
                    )
                )
            context.update(
                {
                    "pipeline_root": pipeline_root,
                    "pipeline_defaults": pipeline_defaults,
                    "tool_adapter": tool_adapter,
                    "dataset_root": dataset_root,
                    "input_derivative_root": input_derivative_root,
                    "remote_dataset_root": remote_dataset_root,
                    "remote_input_derivative_root": remote_input_derivative_root,
                    "requires_input_derivative": requires_input_derivative,
                    "project_roots": project_roots,
                    "data_roots": data_roots,
                    "validation_errors": _dedupe_validation_errors(validation_errors),
                }
            )
        elif slice_name == "tabular":
            tabular_context: dict[str, Any] = {"data_roots": []}
            primary_dataset = context["dataset"].get("primary")
            if primary_dataset is not None:
                dataset_root = dataset_path(workspace_root, workspace, str(primary_dataset))
                tabular_context["dataset_root"] = dataset_root
                tabular_context["data_roots"].append(_root_spec("dataset-root", dataset_root))

            canonical_dataset = context["dataset"].get("canonical_dataset")
            canonical_features_root_value = context["dataset"].get("canonical_features_root")
            if canonical_dataset is not None:
                canonical_dataset_root = dataset_path(workspace_root, workspace, str(canonical_dataset))
                tabular_context["canonical_dataset_root"] = canonical_dataset_root
                if canonical_features_root_value is not None:
                    canonical_features_root = canonical_dataset_root / str(canonical_features_root_value)
                    tabular_context["canonical_features_root"] = canonical_features_root
                    tabular_context["data_roots"].append(_root_spec("canonical-features-root", canonical_features_root))

            context.update(tabular_context)

    declared_data_roots, declaration_errors = resolve_project_hpc_data_root_declarations(
        project_config,
        workspace_root=workspace_root,
        project_root=project_root,
        require_remote_root=False,
    )
    analysis_external_roots: list[dict[str, Any]] = []
    analysis_external_root_errors: list[str] = []
    if isinstance(context.get("analysis"), dict) and context.get("slice") == "bids":
        analysis_external_roots, analysis_external_root_errors = resolve_analysis_external_input_root_declarations(
            context["analysis"],
            workspace_root=workspace_root,
            project_root=project_root,
            require_remote_root=False,
        )
        context["analysis_input_roots"] = {
            str(root_spec["name"]): dict(root_spec)
            for root_spec in analysis_external_roots
            if root_spec.get("name")
        }
    else:
        context["analysis_input_roots"] = {}

    merged_data_roots, merge_errors = merge_declared_data_roots(
        context.get("data_roots", []),
        [*declared_data_roots, *analysis_external_roots],
        conflict_label="Project data roots",
    )
    adapter_data_roots = adapter_data_root_declarations(context=context)
    merged_data_roots, adapter_merge_errors = merge_declared_data_roots(
        merged_data_roots,
        adapter_data_roots,
        conflict_label="Adapter data roots",
    )
    context["data_roots"] = merged_data_roots
    context["analysis_adapter_data_roots"] = adapter_data_roots
    context["validation_errors"] = _dedupe_validation_errors(
        [
            *context.get("validation_errors", []),
            *declaration_errors,
            *merge_errors,
            *analysis_external_root_errors,
            *adapter_merge_errors,
        ]
    )

    remote_workspace_root = _resolve_remote_setting(context.get("compute", {}), "remote_workspace_root") or _env_value(
        "RP_REMOTE_WORKSPACE_ROOT"
    )
    remote_artifacts_root = _resolve_remote_setting(context.get("compute", {}), "remote_artifacts_root") or _env_value(
        "RP_REMOTE_ARTIFACTS_ROOT"
    )
    context["remote_workspace_root"] = remote_workspace_root
    context["remote_artifacts_root"] = remote_artifacts_root
    context["notebook_remote_path"] = _resolve_remote_path(
        notebook_path,
        workspace_root=workspace_root,
        remote_workspace_root=remote_workspace_root,
    )
    return context


def default_notebook_launch_path(context: dict[str, Any]) -> Path:
    notebook_path = context.get("notebook_path")
    if notebook_path is not None:
        return Path(notebook_path)
    return Path(context["project_root"])


def notebook_launch_target(context: dict[str, Any]) -> str:
    path = default_notebook_launch_path(context)
    remote_path = _resolve_remote_path(
        path,
        workspace_root=Path(context["workspace_root"]),
        remote_workspace_root=context.get("remote_workspace_root"),
    )
    if remote_path:
        return remote_path
    return to_workspace_relative(path, context["workspace_root"])


def _build_overlay_data_roots(
    *,
    project_config: dict[str, Any],
    workspace_root: Path,
    workspace: dict[str, Any],
    project_root: Path,
) -> list[dict[str, Any]]:
    roots = resolve_project_overlay_data_roots(
        project_config,
        workspace_root=workspace_root,
        project_root=project_root,
        workspace_config=workspace,
    )
    labels = _overlay_labels(project_config, workspace_root=workspace_root, project_root=project_root)
    return [_root_spec(labels.get(path, "data-root"), path) for path in roots]


def _overlay_labels(
    project_config: dict[str, Any],
    *,
    workspace_root: Path,
    project_root: Path,
) -> dict[Path, str]:
    labels: dict[Path, str] = {}
    overlay = project_config.get("overlay")
    if not isinstance(overlay, dict):
        return labels
    private_data_root = overlay.get("private_data_root")
    if private_data_root:
        labels[_resolve_overlay_path(private_data_root, workspace_root=workspace_root, project_root=project_root)] = (
            "private-data-root"
        )
    raw_inputs = overlay.get("raw_inputs")
    if isinstance(raw_inputs, dict):
        for key, value in raw_inputs.items():
            if value:
                labels[_resolve_overlay_path(value, workspace_root=workspace_root, project_root=project_root)] = (
                    _overlay_root_label(key)
                )
    return labels


def _overlay_root_label(value: object) -> str:
    label = str(value).replace("_", "-")
    if label.endswith("-root"):
        return label
    return f"{label}-root"


def _resolve_overlay_path(value: str | Path, *, workspace_root: Path, project_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    workspace_candidate = (workspace_root / candidate).resolve()
    if workspace_candidate.exists():
        return workspace_candidate
    return (project_root / candidate).resolve()


def _root_spec(
    label: str,
    path: Path,
    *,
    remote_root: str | None = None,
    sync_enabled: bool = True,
    preserve_nested_sync_target: bool = False,
) -> dict[str, Any]:
    return build_data_root_spec(
        label,
        path,
        remote_root=remote_root,
        sync_enabled=sync_enabled,
        preserve_nested_sync_target=preserve_nested_sync_target,
    )


def _apply_declared_hpc_data_roots(
    existing_roots: list[dict[str, Any]],
    declared_roots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    return merge_declared_data_roots(
        existing_roots,
        declared_roots,
        conflict_label="Project data roots",
    )


def _resolve_remote_setting(compute_config: dict[str, Any], key: str) -> str | None:
    slurm_config = compute_config.get("slurm")
    if not isinstance(slurm_config, dict):
        return None
    return resolve_env_value(slurm_config.get(key))


def _env_value(name: str) -> str | None:
    return resolve_env_value(os.environ.get(name))


def _resolve_remote_path(
    path: Path | None,
    *,
    workspace_root: Path,
    remote_workspace_root: str | None,
) -> str | None:
    if path is None or not remote_workspace_root:
        return None
    try:
        relative = Path(path).resolve().relative_to(workspace_root)
    except ValueError:
        return None
    return str(PurePosixPath(remote_workspace_root) / relative.as_posix())


def _safe_resolve_bids_path(
    resolver: Any,
    bundle: dict[str, Any],
    workspace_root: Path,
) -> Path | None:
    try:
        resolved = resolver(bundle, root=workspace_root)
    except Exception:  # pragma: no cover - defensive
        return None
    return Path(resolved).resolve()


def _dedupe_validation_errors(errors: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for error in errors:
        normalized = str(error).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def adapter_data_root_declarations(*, context: dict[str, Any]) -> list[dict[str, Any]]:
    adapter = context.get("tool_adapter")
    if adapter is None:
        return []

    try:
        sync_entries = adapter.sync_entries(
            workspace_root=str(Path(context["workspace_root"]).resolve()),
            context=context,
        )
    except Exception:  # pragma: no cover - defensive
        return []

    workspace_root = Path(context["workspace_root"]).resolve()
    resolved: list[dict[str, Any]] = []
    for entry in sync_entries:
        if str(entry.get("sync_scope", "project")) != "data":
            continue
        kind = str(entry.get("kind", "directory")).strip() or "directory"
        if kind != "directory":
            continue
        source = entry.get("source")
        if source is None:
            continue
        source_path = _resolve_sync_entry_source_path(workspace_root, source)
        if source_path is None:
            continue
        resolved.append(
            build_data_root_spec(
                str(entry.get("label", "data-root")),
                source_path,
                remote_root=_optional_sync_entry_destination(entry.get("destination")),
                sync_enabled=bool(entry.get("sync_enabled", True)),
                preserve_nested_sync_target=bool(entry.get("preserve_nested_sync_target", False)),
                source="adapter.sync_entries",
            )
        )
    return resolved


def _resolve_sync_entry_source_path(workspace_root: Path, value: Any) -> Path | None:
    text = str(value).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root / candidate).resolve()


def _optional_sync_entry_destination(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
