#!/usr/bin/env python3
"""Thin pipeline runner for adapter-driven BIDS preprocessing execution."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
CORE_SRC = WORKSPACE_ROOT / "packages" / "research-core" / "src"
NEURO_SRC = WORKSPACE_ROOT / "packages" / "research-neuro" / "src"
for package_src in (CORE_SRC, NEURO_SRC):
    if str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))

from research_platform.core.config import load_yaml  # noqa: E402
from research_platform.core.runtime_plan import (  # noqa: E402
    execute_runtime_plan_unit,
    load_runtime_plan,
    runtime_plan_unit_ids,
    write_runtime_plan_marker,
)
from research_platform.core.tool_adapters import load_bids_tool_adapter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or execute adapter-driven BIDS preprocessing commands.")
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
    outputs = manifest.get("outputs", {})
    plan_path = Path(args.plan_path).resolve() if args.plan_path else (WORKSPACE_ROOT / str(outputs["runtime_plan"])).resolve()
    command_script_path = (WORKSPACE_ROOT / str(outputs["command_script"])).resolve()
    plan = _load_or_build_plan(
        manifest=manifest,
        plan_path=plan_path,
        command_script_path=command_script_path,
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
    exit_code = _execute_runtime_plan_steps(plan, cwd=WORKSPACE_ROOT)
    if exit_code == 0:
        write_runtime_plan_marker(marker_path, step_count=len(plan["steps"]), unit_count=len(runtime_plan_unit_ids(plan)))
    return exit_code


def _load_or_build_plan(
    *,
    manifest: dict[str, Any],
    plan_path: Path,
    command_script_path: Path,
) -> dict[str, Any]:
    if plan_path.exists():
        return load_runtime_plan(plan_path)

    adapter = load_bids_tool_adapter(
        {
            "tool": manifest.get("tool", {}).get("name"),
            "tool_adapter": manifest.get("tool", {}).get("adapter"),
        }
    )
    return adapter.build_runtime_plan(
        manifest=manifest,
        workspace_root=str(WORKSPACE_ROOT),
        plan_path=str(plan_path),
        command_script_path=str(command_script_path),
    )


def _prepare_runtime_container(plan: dict[str, Any], *, cwd: Path) -> int:
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


def _execute_runtime_plan_steps(plan: Mapping[str, Any], *, cwd: Path) -> int:
    for unit_id in runtime_plan_unit_ids(plan):
        exit_code = execute_runtime_plan_unit(plan, unit_id, cwd=cwd)
        if exit_code != 0:
            return exit_code
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
        path.resolve().relative_to(WORKSPACE_ROOT)
    except ValueError:
        return False
    return True


def _optional_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())
