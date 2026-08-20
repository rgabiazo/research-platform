"""Runtime-plan helpers for first-level FEAT."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import shlex
from typing import Any, Mapping

from ....core.runtime_plan import execute_runtime_plan_steps
from ...events.files import inspect_numeric_event_file
from .models import (
    FeatFirstLevelPlan,
    feat_results_complete,
    load_model_spec,
    preflight_feat_model,
    render_first_level_fsf,
    validate_confounds_rows,
    write_text_file,
)
from .selection import resolve_first_level_inputs
from ..common import (
    FslRuntimeBackendSpec,
    build_fsl_container_prepare_shell,
    build_fsl_container_prepare_and_exec_shell,
    build_fsl_headless_env,
    collect_fsl_bind_roots,
    infer_nvols_and_tr,
    resolve_fsl_container_runtime_image,
    resolve_fsl_runtime_backend,
)


def build_runtime_plan(
    *,
    manifest: Mapping[str, Any],
    workspace_root: str,
    plan_path: str,
    command_script_path: str,
) -> dict[str, Any]:
    analysis_block = manifest.get("analysis", {})
    stage_name = str(analysis_block.get("stage", "")).strip()
    if stage_name != "first_level":
        raise ValueError("Phase 1 FEAT runtime planning supports only analysis stage first_level.")

    model = load_model_spec(str(analysis_block.get("model_ref", "default")), dict(analysis_block.get("model", {})))
    stage_config = dict(analysis_block.get("stage_config", {}))
    validation_config = dict(stage_config.get("validation", {}))
    settings = dict(stage_config.get("settings", {}))
    output_config = dict(stage_config.get("outputs", {}))
    output_desc = _resolve_output_desc(output_config.get("desc"))

    batch_path = _resolve_manifest_path(workspace_root, manifest["batch"]["path"])
    batch_rows = _read_batch_rows(batch_path)
    dataset_root = _resolve_manifest_path(workspace_root, manifest["dataset"]["root"])
    resolution_context = dict(analysis_block)
    resolution_context["analysis_inputs"] = analysis_block.get("inputs", {})
    resolution_context["analysis_model"] = analysis_block.get("model", {})
    resolution_context["analysis_input_roots"] = analysis_block.get("input_roots", {})
    resolved_inputs, skipped_rows = resolve_first_level_inputs(
        derivative_root=_resolve_manifest_path(workspace_root, manifest["dataset"]["derivative_root"]),
        batch_rows=batch_rows,
        context=resolution_context,
        workspace_root=workspace_root,
    )

    output_root = _resolve_manifest_path(workspace_root, manifest["execution"]["output_dir"])
    output_data_dirname = (
        manifest.get("tool", {}).get("runtime_metadata", {}).get("output_data_dirname") or "fsl_feat"
    )
    output_data_root = (output_root / output_data_dirname).resolve()
    fsf_root = (output_root / "fsf").resolve()
    unit_marker_root = (output_root / "runtime-plan-markers" / "feat").resolve()
    output_data_root.mkdir(parents=True, exist_ok=True)
    fsf_root.mkdir(parents=True, exist_ok=True)
    unit_marker_root.mkdir(parents=True, exist_ok=True)

    backend_spec = _resolve_feat_runtime_backend(manifest)
    bind_roots = collect_fsl_bind_roots(
        manifest=manifest,
        output_root=output_root,
        workspace_root=workspace_root,
    )
    container_prep = _build_feat_container_prep_step(
        backend_spec=backend_spec,
        marker_root=unit_marker_root,
    )
    execution_backend_spec = _execution_backend_spec(backend_spec, container_prep=container_prep)
    steps: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    runtime_environment = build_fsl_headless_env(backend_spec.environment)
    allow_missing_evs = bool(validation_config.get("allow_missing_evs", False))
    require_confounds = bool(validation_config.get("require_confounds", False))
    empty_ev_policy = _resolve_empty_ev_policy(validation_config.get("empty_ev_policy"))
    overwrite_results = bool(stage_config.get("overwrite", {}).get("results", False))

    for resolved in resolved_inputs:
        output_stem = _feat_output_stem(str(resolved["bids_base"]), output_desc=output_desc)
        unit_label = _sanitize_unit_id(output_stem)
        output_parent = _feat_output_parent(
            row=resolved["row"],
            entities=resolved["entities"],
        )
        out_dir = (output_data_root / output_parent / f"{output_stem}.feat").resolve()
        fsf_path = (fsf_root / output_parent / f"{output_stem}.fsf").resolve()
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        fsf_path.parent.mkdir(parents=True, exist_ok=True)

        nvols, tr = infer_nvols_and_tr(
            resolved["bold_path"],
            override_tr=_coerce_optional_float(settings.get("tr")),
            metadata_search_roots=[dataset_root],
        )
        if nvols is None or tr is None:
            raise ValueError(f"Unable to infer TR/nvols for FEAT input: {resolved['bold_path']}")

        ev_paths_by_name = {name: Path(path).resolve() for name, path in resolved["ev_paths_by_name"].items()}
        empty_evs, ev_problems = _inspect_ev_inputs(
            ev_paths_by_name=ev_paths_by_name,
            ev_order=model.ev_order,
            empty_ev_policy=empty_ev_policy,
        )
        plan = FeatFirstLevelPlan(
            row=dict(resolved["row"]),
            entities=dict(resolved["entities"]),
            bids_base=str(resolved["bids_base"]),
            bold_path=Path(resolved["bold_path"]).resolve(),
            confounds_path=Path(resolved["confounds_path"]).resolve() if resolved.get("confounds_path") else None,
            ev_paths_by_name=ev_paths_by_name,
            missing_evs=list(resolved["missing_evs"]),
            out_dir=out_dir,
            fsf_path=fsf_path,
            empty_evs=empty_evs,
            nvols=nvols,
            tr=tr,
        )

        if plan.missing_evs and not allow_missing_evs:
            raise ValueError(f"Missing EV files for {plan.bids_base}: {', '.join(plan.missing_evs)}")

        if require_confounds and plan.confounds_path is None:
            raise ValueError(f"Confounds are required but were not resolved for {plan.bids_base}.")

        if ev_problems:
            raise ValueError(f"Invalid EV files for {plan.bids_base}: {'; '.join(ev_problems)}")

        confounds_ok, confounds_message = validate_confounds_rows(plan.confounds_path, npts=nvols)
        if not confounds_ok:
            raise ValueError(f"Invalid confounds for {plan.bids_base}: {confounds_message}")

        if feat_results_complete(out_dir) and not overwrite_results:
            skipped_rows.append(
                {
                    "subject_id": plan.row.get("subject_id", ""),
                    "session_id": plan.row.get("session_id", ""),
                    "task_id": plan.row.get("task_id", ""),
                    "run_id": plan.row.get("run_id", ""),
                    "reason": f"Existing FEAT outputs already look complete: {out_dir}",
                }
            )
            continue

        fsf_text = render_first_level_fsf(
            model=model,
            plan=plan,
            tr=tr,
            npts=nvols,
            settings=settings | {"overwrite_design": bool(stage_config.get("overwrite", {}).get("design", False))},
        )
        write_text_file(fsf_path, fsf_text)
        preflight_ok, preflight_message = preflight_feat_model(fsf_path)
        if not preflight_ok:
            raise ValueError(f"FEAT preflight failed for {plan.bids_base}: {preflight_message}")

        marker_path = unit_marker_root / f"{unit_label}.txt"
        command = _build_feat_runtime_command(
            backend_spec=execution_backend_spec,
            bind_roots=bind_roots,
            fsf_path=fsf_path,
            runtime_environment=runtime_environment,
        )
        step = {
            "unit_id": unit_label,
            "row": plan.row,
            "bids_base": plan.bids_base,
            "bold_path": str(plan.bold_path),
            "output_dir": str(out_dir),
            "fsf_path": str(fsf_path),
            "command": command,
            "env": runtime_environment,
            "preflight": preflight_message,
            "backend": backend_spec.execution_backend,
            "bind_roots": bind_roots,
        }
        if backend_spec.container is not None:
            image_details = {
                "enabled": backend_spec.container.enabled,
                "backend": backend_spec.container.backend,
                "image": backend_spec.container.image,
                "pull_mode": backend_spec.container.pull_mode,
                "image_name": backend_spec.container.image_name,
                "image_root": backend_spec.container.image_root,
            }
            step["container"] = image_details
        steps.append(step)
        units.append(
            {
                "unit_id": unit_label,
                "step_count": 1,
                "marker_path": str(marker_path),
                "steps": [step],
            }
        )

    if not steps:
        container_prep = None
    plan = {
        "tool": "feat",
        "stage": stage_name,
        "plan_path": str(Path(plan_path).resolve()),
        "command_script": str(Path(command_script_path).resolve()),
        "output_root": str(output_root),
        "output_data_root": str(output_data_root),
        "container_prep": container_prep,
        "steps": steps,
        "units": units,
        "skipped_rows": skipped_rows,
    }
    write_runtime_plan(plan, plan_path)
    write_command_script(plan, command_script_path)
    return plan


def write_runtime_plan(plan: Mapping[str, Any], plan_path: str | Path) -> Path:
    path = Path(plan_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return path


def write_command_script(plan: Mapping[str, Any], command_script_path: str | Path) -> Path:
    path = Path(command_script_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    container_prep = plan.get("container_prep")
    if isinstance(container_prep, Mapping):
        prep_command = container_prep.get("command")
        if isinstance(prep_command, list):
            lines.append(" ".join(shlex.quote(str(part)) for part in prep_command))
    for step in plan.get("steps", []):
        env_block = ""
        if isinstance(step.get("env"), dict) and step["env"]:
            env_block = "env " + " ".join(
                f"{name}={shlex.quote(str(value))}" for name, value in sorted(step["env"].items())
            ) + " "
        lines.append(env_block + " ".join(shlex.quote(part) for part in step["command"]))
    if len(lines) == 3:
        lines.append("printf 'No FEAT runs were selected.\\n' >&2")
        lines.append("exit 1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def execute_runtime_plan(plan: Mapping[str, Any], *, cwd: str | Path) -> int:
    container_prep = plan.get("container_prep")
    if isinstance(container_prep, Mapping):
        prep_marker_path = _optional_text(container_prep.get("marker_path"))
        prep_command = container_prep.get("command")
        prep_marker_exists = prep_marker_path is not None and Path(prep_marker_path).exists()
        if isinstance(prep_command, list) and not prep_marker_exists:
            exit_code = execute_runtime_plan_steps([{"command": prep_command}], cwd=cwd)
            if exit_code != 0:
                return exit_code
    return execute_runtime_plan_steps(plan.get("steps", []), cwd=cwd)


def _resolve_manifest_path(workspace_root: str | Path, value: Any) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(workspace_root).resolve() / candidate).resolve()


def _read_batch_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: str(value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle, delimiter="\t")]


def _runtime_environment(manifest: Mapping[str, Any]) -> dict[str, str]:
    tool = manifest.get("tool", {})
    runtime_profile = tool.get("runtime_profile", {})
    config = runtime_profile.get("config", {}) if isinstance(runtime_profile, dict) else {}
    mode = str(manifest.get("execution", {}).get("mode", "local")).strip()
    profile_block = config.get("slurm" if mode == "slurm" else "local", {})
    if not isinstance(profile_block, dict):
        return {}
    environment = profile_block.get("environment")
    if not isinstance(environment, dict):
        return {}
    return {
        str(name).strip(): str(value).strip()
        for name, value in environment.items()
        if str(name).strip() and str(value).strip()
    }


def _resolve_feat_runtime_backend(manifest: Mapping[str, Any]):
    tool = manifest.get("tool", {})
    runtime_profile = tool.get("runtime_profile", {})
    config = runtime_profile.get("config", {}) if isinstance(runtime_profile, dict) else {}
    mode = str(manifest.get("execution", {}).get("mode", "local")).strip()
    return resolve_fsl_runtime_backend(config if isinstance(config, Mapping) else {}, mode=mode)


def _build_feat_runtime_command(
    *,
    backend_spec,
    bind_roots: list[str],
    fsf_path: Path,
    runtime_environment: Mapping[str, str],
) -> list[str]:
    if backend_spec.execution_backend == "native":
        return ["feat", str(fsf_path)]
    if backend_spec.container is None:
        raise ValueError(f"FSL backend {backend_spec.execution_backend!r} requires a container specification.")
    shell_command = build_fsl_container_prepare_and_exec_shell(
        backend=backend_spec.execution_backend,
        container=backend_spec.container,
        bind_roots=bind_roots,
        env=runtime_environment,
        command=["feat", str(fsf_path)],
    )
    return ["bash", "-lc", shell_command]


def _build_feat_container_prep_step(
    *,
    backend_spec: FslRuntimeBackendSpec,
    marker_root: Path,
) -> dict[str, Any] | None:
    if backend_spec.execution_backend == "native" or backend_spec.container is None:
        return None
    image_details = resolve_fsl_container_runtime_image(backend_spec.container)
    if not image_details["requires_pull"]:
        return None

    marker_path = marker_root / "_container-ready.txt"
    return {
        "marker_path": str(marker_path),
        "runtime_image": str(image_details["runtime_image"]),
        "image_source": str(image_details["image_reference"]),
        "image_root": str(backend_spec.container.image_root or ""),
        "command": [
            "bash",
            "-lc",
            build_fsl_container_prepare_shell(
                backend=backend_spec.execution_backend,
                container=backend_spec.container,
            ),
        ],
    }


def _execution_backend_spec(
    backend_spec: FslRuntimeBackendSpec,
    *,
    container_prep: Mapping[str, Any] | None,
) -> FslRuntimeBackendSpec:
    if container_prep is None or backend_spec.container is None:
        return backend_spec

    runtime_image = str(container_prep["runtime_image"])
    prepared_container = replace(
        backend_spec.container,
        image=runtime_image,
        pull_mode="never",
    )
    return replace(backend_spec, container=prepared_container)


def _sanitize_unit_id(value: str) -> str:
    sanitized = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return sanitized or "feat"


def _feat_output_parent(*, row: Mapping[str, Any], entities: Mapping[str, str]) -> Path:
    parts: list[str] = []
    for key in ("subject_id", "session_id"):
        value = _optional_text(row.get(key)) or _optional_text(entities.get(key))
        if value is not None:
            parts.append(value)
    return Path(*parts) if parts else Path()


def _feat_output_stem(bids_base: str, *, output_desc: str | None) -> str:
    if output_desc is None:
        return bids_base
    return f"{bids_base}_desc-{output_desc}"


def _coerce_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _resolve_empty_ev_policy(value: Any) -> str:
    policy = str(value or "fail").strip()
    if policy not in {"as_zero", "fail"}:
        raise ValueError(f"Unsupported empty_ev_policy {policy!r}; expected 'as_zero' or 'fail'.")
    return policy


def _resolve_output_desc(value: Any) -> str | None:
    if value in (None, ""):
        return None
    desc = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9]+", desc):
        raise ValueError(f"Unsupported output desc {desc!r}; use only letters and numbers.")
    return desc


def _inspect_ev_inputs(
    *,
    ev_paths_by_name: dict[str, Path],
    ev_order: list[str],
    empty_ev_policy: str,
) -> tuple[list[str], list[str]]:
    empty_evs: list[str] = []
    problems: list[str] = []
    for ev_name in ev_order:
        path = ev_paths_by_name.get(ev_name)
        if path is None:
            problems.append(f"Missing EV file for {ev_name}.")
            continue
        inspection = inspect_numeric_event_file(path, min_columns=3)
        if inspection.status == "valid":
            continue
        if inspection.status == "empty" and empty_ev_policy == "as_zero":
            empty_evs.append(ev_name)
            continue
        if inspection.status == "empty":
            problems.append(f"EV file is empty: {path}")
            continue
        if inspection.status == "missing":
            problems.append(f"Missing EV file for {ev_name}: {path}")
            continue
        detail = f": {inspection.message}" if inspection.message else ""
        problems.append(f"EV file is not a valid numeric 3-column file: {path}{detail}")
    return empty_evs, problems


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
