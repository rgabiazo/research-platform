"""Selection helpers for first-level FEAT analysis."""

from __future__ import annotations

import glob
from pathlib import Path, PurePosixPath
from typing import Any

from ..common import (
    build_bids_base,
    normalize_entity_label,
    parse_bidsish_entities,
    render_path_pattern,
    resolve_reference_path,
)


def discover_batch_rows(
    derivative_root: str,
    *,
    selectors: dict[str, str | None],
    context: dict[str, Any],
) -> list[dict[str, str]]:
    bold_matches = _discover_bold_matches(Path(derivative_root), context=context)
    discovered: dict[tuple[str, str, str, str], dict[str, str]] = {}
    normalized_selectors = _normalize_selectors(selectors)
    for bold_path in bold_matches:
        entities = parse_bidsish_entities(bold_path)
        row = {
            "subject_id": entities.get("subject_id", ""),
            "session_id": entities.get("session_id", ""),
            "task_id": entities.get("task_id", ""),
            "run_id": entities.get("run_id", ""),
        }
        if not _row_matches_selectors(row, normalized_selectors):
            continue
        key = (row["subject_id"], row["session_id"], row["task_id"], row["run_id"])
        discovered[key] = row
    return [discovered[key] for key in sorted(discovered)]


def resolve_first_level_inputs(
    *,
    derivative_root: str | Path,
    batch_rows: list[dict[str, str]],
    context: dict[str, Any],
    workspace_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    derivative_root_path = Path(derivative_root).resolve()
    resolved: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []

    for row in batch_rows:
        bold_matches = _resolve_row_bold_matches(
            derivative_root_path=derivative_root_path,
            row=row,
            context=context,
        )
        if not bold_matches:
            skipped_rows.append(
                {
                    "subject_id": row.get("subject_id", ""),
                    "session_id": row.get("session_id", ""),
                    "task_id": row.get("task_id", ""),
                    "run_id": row.get("run_id", ""),
                    "reason": "No matching FEAT input BOLD runs were resolved from the configured patterns.",
                }
            )
            continue

        for bold_path in bold_matches:
            entities = parse_bidsish_entities(bold_path)
            bids_base = build_bids_base(entities)
            ev_paths_by_name, missing_evs = _resolve_ev_paths(
                workspace_root=workspace_root,
                entities=entities,
                bids_base=bids_base,
                context=context,
            )
            resolved.append(
                {
                    "row": {name: row.get(name, "") for name in ("subject_id", "session_id", "task_id", "run_id")},
                    "entities": entities,
                    "bids_base": bids_base,
                    "bold_path": bold_path,
                    "confounds_path": _resolve_confounds_path(
                        derivative_root_path=derivative_root_path,
                        workspace_root=workspace_root,
                        entities=entities,
                        bids_base=bids_base,
                        context=context,
                    ),
                    "ev_paths_by_name": ev_paths_by_name,
                    "missing_evs": missing_evs,
                }
            )
    return resolved, skipped_rows


def expected_remote_input_files(
    derivative_root: str,
    *,
    remote_derivative_root: str,
    row: dict[str, str],
    context: dict[str, Any],
) -> list[str]:
    workspace_root = Path(context["workspace_root"]).resolve()
    local_derivative_root = Path(derivative_root).resolve()
    remote_workspace_root = str(context.get("compute", {}).get("slurm", {}).get("remote_workspace_root", "")).strip()
    remote_dataset_root = str(context.get("remote_dataset_root", "")).strip()
    declared_data_roots = _declared_data_roots(context)
    resolved_inputs, _ = resolve_first_level_inputs(
        derivative_root=local_derivative_root,
        batch_rows=[row],
        context=context,
        workspace_root=workspace_root,
    )
    if not resolved_inputs:
        return []

    paths: list[str] = []
    for resolved in resolved_inputs:
        for path in (
            resolved["bold_path"],
            resolved.get("confounds_path"),
            *resolved.get("ev_paths_by_name", {}).values(),
        ):
            if path is None:
                continue
            remote_path = _remote_path_for_local(
                local_path=Path(path),
                workspace_root=workspace_root,
                local_derivative_root=local_derivative_root,
                remote_derivative_root=remote_derivative_root,
                remote_workspace_root=remote_workspace_root,
                remote_dataset_root=remote_dataset_root,
                local_dataset_root=Path(context["dataset_root"]).resolve() if context.get("dataset_root") else None,
                declared_data_roots=declared_data_roots,
            )
            if remote_path:
                paths.append(remote_path)
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _discover_bold_matches(derivative_root: Path, *, context: dict[str, Any]) -> list[Path]:
    inputs = context.get("analysis_inputs", {})
    bold_config = inputs.get("bold", {}) if isinstance(inputs, dict) else {}
    patterns = _string_list(bold_config.get("patterns"))
    if not patterns:
        return []
    render_values = {
        "derivative_root": str(derivative_root),
        "subject_dir": "sub-*",
        "session_dir": "ses-*",
        "bids_base": "*",
    }
    matches: set[Path] = set()
    for pattern in patterns:
        rendered = render_path_pattern(pattern, render_values)
        for match in glob.glob(rendered, recursive=True):
            path = Path(match)
            if path.exists():
                matches.add(path.resolve())
    return sorted(matches)


def _resolve_row_bold_matches(
    *,
    derivative_root_path: Path,
    row: dict[str, str],
    context: dict[str, Any],
) -> list[Path]:
    normalized_row = _normalize_selectors(row)
    matches = _discover_bold_matches(derivative_root_path, context=context)
    selected: list[Path] = []
    for path in matches:
        entities = parse_bidsish_entities(path)
        candidate = {
            "subject_id": entities.get("subject_id", ""),
            "session_id": entities.get("session_id", ""),
            "task_id": entities.get("task_id", ""),
            "run_id": entities.get("run_id", ""),
        }
        if _row_matches_selectors(candidate, normalized_row):
            selected.append(path)
    return selected


def _resolve_confounds_path(
    *,
    derivative_root_path: Path,
    workspace_root: str | Path,
    entities: dict[str, str],
    bids_base: str,
    context: dict[str, Any],
) -> Path | None:
    inputs = context.get("analysis_inputs", {})
    confounds_config = inputs.get("confounds", {}) if isinstance(inputs, dict) else {}
    patterns = _string_list(confounds_config.get("patterns"))
    if not patterns:
        return None
    input_root = _resolve_analysis_input_root(
        input_name="confounds",
        input_config=confounds_config,
        context=context,
        workspace_root=workspace_root,
    )
    render_values = _pattern_values(
        derivative_root=str(derivative_root_path),
        entities=entities,
        bids_base=bids_base,
        input_root=str(input_root) if input_root is not None else "",
    ) | {"confounds_root": str(input_root) if input_root is not None else ""}
    for pattern in patterns:
        rendered = render_path_pattern(pattern, render_values)
        for match in sorted(glob.glob(rendered, recursive=True)):
            path = Path(match)
            if path.exists():
                return path.resolve()
    return None


def _resolve_ev_paths(
    *,
    workspace_root: str | Path,
    entities: dict[str, str],
    bids_base: str,
    context: dict[str, Any],
) -> tuple[dict[str, Path], list[str]]:
    inputs = context.get("analysis_inputs", {})
    ev_config = inputs.get("evs", {}) if isinstance(inputs, dict) else {}
    ev_patterns = _string_list(ev_config.get("patterns"))
    if not ev_patterns:
        return {}, list(context.get("analysis_model", {}).get("ev_order", []))

    ev_root = _resolve_analysis_input_root(
        input_name="evs",
        input_config=ev_config,
        context=context,
        workspace_root=workspace_root,
    )
    if ev_root is None:
        return {}, list(context.get("analysis_model", {}).get("ev_order", []))

    model = context.get("analysis_model", {})
    ev_order = [str(name).strip() for name in model.get("ev_order", []) if str(name).strip()]
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    render_values = _pattern_values(
        derivative_root="",
        entities=entities,
        bids_base=bids_base,
        input_root=str(ev_root),
    ) | {"ev_root": str(ev_root)}
    for ev_name in ev_order:
        ev_path = None
        for pattern in ev_patterns:
            rendered = render_path_pattern(pattern, render_values | {"ev_name": ev_name})
            matches = [Path(match) for match in sorted(glob.glob(rendered, recursive=True))]
            existing = [match.resolve() for match in matches if match.exists()]
            if existing:
                ev_path = existing[0]
                break
        if ev_path is None:
            missing.append(ev_name)
        else:
            resolved[ev_name] = ev_path
    return resolved, missing


def _pattern_values(*, derivative_root: str, entities: dict[str, str], bids_base: str, input_root: str = "") -> dict[str, str]:
    return {
        "derivative_root": derivative_root,
        "input_root": input_root,
        "subject_dir": entities.get("subject_id", ""),
        "session_dir": entities.get("session_id", ""),
        "bids_base": bids_base,
    }


def _normalize_selectors(values: dict[str, str | None]) -> dict[str, str | None]:
    return {
        "subject_id": normalize_entity_label(values.get("subject_id"), prefix="sub"),
        "session_id": normalize_entity_label(values.get("session_id"), prefix="ses"),
        "task_id": normalize_entity_label(values.get("task_id"), prefix="task"),
        "run_id": normalize_entity_label(values.get("run_id"), prefix="run"),
    }


def _row_matches_selectors(row: dict[str, str], selectors: dict[str, str | None]) -> bool:
    for key in ("subject_id", "session_id", "task_id", "run_id"):
        expected = selectors.get(key)
        if expected is None:
            continue
        actual = normalize_entity_label(row.get(key), prefix=_selector_prefix(key))
        if actual != expected:
            return False
    return True


def _selector_prefix(key: str) -> str:
    return {
        "subject_id": "sub",
        "session_id": "ses",
        "task_id": "task",
        "run_id": "run",
    }[key]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _remote_path_for_local(
    *,
    local_path: Path,
    workspace_root: Path,
    local_derivative_root: Path,
    remote_derivative_root: str,
    remote_workspace_root: str,
    remote_dataset_root: str,
    local_dataset_root: Path | None,
    declared_data_roots: list[dict[str, Any]],
) -> str | None:
    for root_spec in declared_data_roots:
        root_path = Path(root_spec["path"]).resolve()
        remote_root = str(root_spec.get("remote_root", "")).strip()
        if not remote_root:
            continue
        try:
            relative_to_declared_root = local_path.resolve().relative_to(root_path)
        except ValueError:
            relative_to_declared_root = None
        if relative_to_declared_root is not None:
            return str(PurePosixPath(remote_root) / relative_to_declared_root.as_posix())

    try:
        relative_to_derivative = local_path.resolve().relative_to(local_derivative_root)
    except ValueError:
        relative_to_derivative = None
    if relative_to_derivative is not None:
        return str(PurePosixPath(remote_derivative_root) / relative_to_derivative.as_posix())

    if local_dataset_root is not None and remote_dataset_root:
        try:
            relative_to_dataset = local_path.resolve().relative_to(local_dataset_root)
        except ValueError:
            relative_to_dataset = None
        if relative_to_dataset is not None:
            return str(PurePosixPath(remote_dataset_root) / relative_to_dataset.as_posix())

    if remote_workspace_root:
        try:
            relative_to_workspace = local_path.resolve().relative_to(workspace_root)
        except ValueError:
            relative_to_workspace = None
        if relative_to_workspace is not None:
            return str(PurePosixPath(remote_workspace_root) / relative_to_workspace.as_posix())
    return None


def _resolve_analysis_input_root(
    *,
    input_name: str,
    input_config: dict[str, Any],
    context: dict[str, Any],
    workspace_root: str | Path,
) -> Path | None:
    input_roots = context.get("analysis_input_roots")
    if not isinstance(input_roots, dict):
        input_roots = context.get("input_roots", {})

    root_ref = str(input_config.get("root_ref", "")).strip()
    if root_ref and isinstance(input_roots, dict):
        root_spec = input_roots.get(root_ref)
        if isinstance(root_spec, dict):
            root_value = str(root_spec.get("path", "")).strip()
            if root_value:
                return resolve_reference_path(workspace_root, root_value)

    direct_root = str(input_config.get("root", "")).strip()
    if direct_root:
        return resolve_reference_path(workspace_root, direct_root)

    # Backward-compatible manifest/runtime support when direct roots are rendered separately.
    if isinstance(input_roots, dict):
        root_spec = input_roots.get(input_name)
        if isinstance(root_spec, dict):
            root_value = str(root_spec.get("path", "")).strip()
            if root_value:
                return resolve_reference_path(workspace_root, root_value)
    return None


def _declared_data_roots(context: dict[str, Any]) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    for root_spec in context.get("data_roots", []):
        if not isinstance(root_spec, dict):
            continue
        if root_spec.get("path") is None:
            continue
        roots.append(root_spec)

    analysis_input_roots = context.get("analysis_input_roots")
    if isinstance(analysis_input_roots, dict):
        for root_spec in analysis_input_roots.values():
            if not isinstance(root_spec, dict):
                continue
            if root_spec.get("path") is None:
                continue
            roots.append(root_spec)

    inputs = context.get("analysis_inputs", {})
    if isinstance(inputs, dict):
        for input_config in inputs.values():
            if not isinstance(input_config, dict):
                continue
            if str(input_config.get("root_ref", "")).strip():
                continue
            direct_root = str(input_config.get("root", "")).strip()
            if not direct_root:
                continue
            roots.append(
                {
                    "path": resolve_reference_path(context["workspace_root"], direct_root),
                    "remote_root": str(input_config.get("remote_root", "")).strip() or None,
                }
            )
    return roots
