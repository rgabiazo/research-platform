#!/usr/bin/env python3
"""Thin pipeline runner for derivative-first fMRIPost-AROMA execution."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
CORE_SRC = WORKSPACE_ROOT / "packages" / "research-core" / "src"
NEURO_SRC = WORKSPACE_ROOT / "packages" / "research-neuro" / "src"
for package_src in (CORE_SRC, NEURO_SRC):
    if str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))

from research_platform.core.config import load_yaml, resolve_env_value  # noqa: E402
from research_platform.core.runtime_plan import (  # noqa: E402
    execute_runtime_plan_unit,
    load_runtime_plan,
    runtime_plan_unit_ids,
    write_runtime_plan_marker,
)
from research_platform.neuro.fmripost_aroma import (  # noqa: E402
    DEFAULT_APPTAINER_IMAGE_ROOT,
    DEFAULT_APPTAINER_PULL_MODE,
    DEFAULT_HPC_BACKEND,
    DEFAULT_IMAGE_REPOSITORY,
    DEFAULT_IMAGE_TAG,
    DEFAULT_LOCAL_BACKEND,
    build_batch_runtime_plan,
    execute_runtime_plan,
    write_command_script,
    write_runtime_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or execute derivative-first fMRIPost-AROMA commands.")
    parser.add_argument("--run-manifest", required=True, help="Path to the rp-generated run manifest.")
    parser.add_argument("--marker", default=None, help="Marker file written on success.")
    parser.add_argument("--plan-only", action="store_true", help="Build and persist the runtime plan without executing it.")
    parser.add_argument("--aggregate-only", action="store_true", help="Write the final aggregate marker without executing runtime units.")
    parser.add_argument(
        "--prepare-container-only",
        action="store_true",
        help="Prepare any shared container runtime image needed by the current plan, then write a marker.",
    )
    parser.add_argument("--unit-id", default=None, help="Execute exactly one runtime-plan unit by id.")
    parser.add_argument("--plan-path", default=None, help="Optional existing runtime-plan path to load instead of rebuilding.")
    args = parser.parse_args(argv)
    if not args.plan_only and args.marker is None:
        parser.error("--marker is required unless --plan-only is used.")

    manifest_path = Path(args.run_manifest).resolve()
    manifest = load_yaml(manifest_path)
    _validate_local_plan_only_manifest(
        manifest_path=manifest_path,
        plan_only=args.plan_only,
        explicit_plan_path=args.plan_path is not None,
        mode=str(manifest["execution"]["mode"]),
        output_dir_value=manifest["execution"]["output_dir"],
        work_dir_value=manifest["execution"]["work_dir"],
    )
    defaults = load_yaml(Path(__file__).resolve().parents[1] / "config" / "defaults.yaml")
    batch_rows = _read_batch_rows(WORKSPACE_ROOT / manifest["batch"]["path"])
    mode = manifest["execution"]["mode"]
    output_dir = WORKSPACE_ROOT / manifest["execution"]["output_dir"]
    work_dir = WORKSPACE_ROOT / manifest["execution"]["work_dir"]
    outputs = manifest.get("outputs", {})
    tool_block = manifest.get("tool", {})

    local_backend = os.environ.get("RP_FMRIPOST_AROMA_LOCAL_BACKEND", defaults["tool"].get("local_backend", DEFAULT_LOCAL_BACKEND))
    slurm_backend = os.environ.get("RP_FMRIPOST_AROMA_SLURM_BACKEND", defaults["tool"].get("slurm_backend", DEFAULT_HPC_BACKEND))
    backend = slurm_backend if mode == "slurm" else local_backend
    image_repository = os.environ.get("RP_FMRIPOST_AROMA_IMAGE_REPOSITORY", defaults["tool"].get("image_repository", DEFAULT_IMAGE_REPOSITORY))
    image_tag = os.environ.get("RP_FMRIPOST_AROMA_IMAGE_TAG", defaults["tool"].get("image_tag", DEFAULT_IMAGE_TAG))
    container_pull_mode = os.environ.get("RP_FMRIPOST_AROMA_CONTAINER_PULL_MODE")
    if container_pull_mode is None and mode == "slurm" and backend in {"apptainer", "singularity"}:
        container_pull_mode = DEFAULT_APPTAINER_PULL_MODE
    container_image_root = os.environ.get("RP_FMRIPOST_AROMA_IMAGE_ROOT")
    if container_image_root is None and mode == "slurm" and backend in {"apptainer", "singularity"}:
        container_image_root = DEFAULT_APPTAINER_IMAGE_ROOT
    container_image_name = os.environ.get("RP_FMRIPOST_AROMA_IMAGE_NAME")
    templateflow_home = _optional_value(
        os.environ.get("RP_TEMPLATEFLOW_HOME")
        or os.environ.get("TEMPLATEFLOW_HOME")
        or defaults["tool"].get("templateflow_home")
    )

    plan_path = Path(args.plan_path).resolve() if args.plan_path else (WORKSPACE_ROOT / outputs["runtime_plan"]).resolve()
    plan = _load_or_build_plan(
        manifest=manifest,
        batch_rows=batch_rows,
        backend=backend,
        image_repository=image_repository,
        image_tag=image_tag,
        templateflow_home=templateflow_home,
        output_dir=output_dir,
        work_dir=work_dir,
        plan_path=plan_path,
        command_script_path=(WORKSPACE_ROOT / outputs["command_script"]).resolve(),
        container_pull_mode=container_pull_mode,
        container_image_root=container_image_root,
        container_image_name=container_image_name,
    )
    if not plan["steps"]:
        return 1
    if args.plan_only:
        return 0
    marker_path = Path(args.marker).resolve()
    if args.prepare_container_only:
        exit_code = _prepare_runtime_container(plan, cwd=WORKSPACE_ROOT)
        if exit_code == 0:
            write_runtime_plan_marker(marker_path, step_count=1)
        return exit_code
    if args.aggregate_only:
        write_runtime_plan_marker(marker_path, step_count=len(plan["steps"]), unit_count=len(runtime_plan_unit_ids(plan)))
        return 0
    if args.unit_id:
        prep_exit_code = _prepare_runtime_container(plan, cwd=WORKSPACE_ROOT)
        if prep_exit_code != 0:
            return prep_exit_code
        exit_code = execute_runtime_plan_unit(plan, args.unit_id, cwd=WORKSPACE_ROOT)
        if exit_code == 0:
            unit = next(unit for unit in plan.get("units", []) if unit.get("unit_id") == args.unit_id)
            write_runtime_plan_marker(marker_path, unit_id=args.unit_id, step_count=int(unit.get("step_count", 0)))
        return exit_code
    if manifest["execution"]["dry_run"]:
        write_runtime_plan_marker(marker_path, step_count=len(plan["steps"]), unit_count=len(runtime_plan_unit_ids(plan)))
        return 0
    prep_exit_code = _prepare_runtime_container(plan, cwd=WORKSPACE_ROOT)
    if prep_exit_code != 0:
        return prep_exit_code
    exit_code = execute_runtime_plan(plan, cwd=WORKSPACE_ROOT)
    if exit_code == 0:
        write_runtime_plan_marker(marker_path, step_count=len(plan["steps"]), unit_count=len(runtime_plan_unit_ids(plan)))
    return exit_code


def _read_batch_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: resolve_env_value(value) or "" for key, value in row.items()} for row in csv.DictReader(handle, delimiter="\t")]


def _load_or_build_plan(
    *,
    manifest: dict[str, object],
    batch_rows: list[dict[str, str]],
    backend: str,
    image_repository: str,
    image_tag: str,
    templateflow_home: str | None,
    output_dir: Path,
    work_dir: Path,
    plan_path: Path,
    command_script_path: Path,
    container_pull_mode: str | None,
    container_image_root: str | None,
    container_image_name: str | None,
) -> dict[str, object]:
    if plan_path.exists():
        return load_runtime_plan(plan_path)

    tool_block = manifest.get("tool", {})
    dataset_block = manifest["dataset"]
    plan = build_batch_runtime_plan(
        raw_bids_root=WORKSPACE_ROOT / dataset_block["root"],
        derivative_root=WORKSPACE_ROOT / dataset_block["derivative_root"],
        derivative_name=dataset_block["derivative_name"],
        batch_rows=batch_rows,
        output_root=output_dir,
        work_root=work_dir,
        plan_path=plan_path,
        command_script_path=command_script_path,
        selection=manifest.get("selection", {}),
        backend=backend,
        image_repository=image_repository,
        image_tag=image_tag,
        tool_options=tool_block.get("options"),
        templateflow_home=templateflow_home,
        resources=manifest.get("resources"),
        container_pull_mode=container_pull_mode,
        container_image_root=container_image_root,
        container_image_name=container_image_name,
    )
    write_runtime_plan(plan, plan_path)
    write_command_script(plan, command_script_path)
    return plan


def _optional_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _prepare_runtime_container(plan: dict[str, object], *, cwd: Path) -> int:
    raw_prep = plan.get("container_prep")
    if not isinstance(raw_prep, dict):
        return 0

    prep_marker_path = _optional_value(raw_prep.get("marker_path"))
    if prep_marker_path is not None and Path(prep_marker_path).exists():
        return 0

    command = raw_prep.get("command")
    if not isinstance(command, list) or not command:
        return 0

    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    if prep_marker_path is not None:
        write_runtime_plan_marker(Path(prep_marker_path), step_count=1)
    return 0


def _validate_local_plan_only_manifest(
    *,
    manifest_path: Path,
    plan_only: bool,
    explicit_plan_path: bool,
    mode: str,
    output_dir_value: object,
    work_dir_value: object,
) -> None:
    if not plan_only or mode != "slurm":
        return

    manifest_looks_local = _is_within_workspace(manifest_path)
    if not manifest_looks_local and explicit_plan_path:
        return

    invalid_paths: list[str] = []
    for label, value in (
        ("execution.output_dir", output_dir_value),
        ("execution.work_dir", work_dir_value),
    ):
        raw_path = Path(str(value))
        resolved_path = (WORKSPACE_ROOT / raw_path).resolve()
        if raw_path.is_absolute() and not _is_within_workspace(resolved_path):
            invalid_paths.append(f"{label}={resolved_path}")

    if not invalid_paths:
        return

    path_summary = "; ".join(invalid_paths)
    raise SystemExit(
        "This manifest is planned for remote/SLURM execution and cannot be used for local --plan-only validation "
        f"with remote execution paths ({path_summary}). "
        "For local plan validation, use `rp run plan preprocess bids` to create a local manifest."
    )


def _is_within_workspace(path: Path) -> bool:
    try:
        path.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
