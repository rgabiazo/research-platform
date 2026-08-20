"""Shared helpers for runtime-plan unit fan-out and execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

SUCCESS_MARKER_EXIT_GRACE_SECONDS = 10.0
SUCCESS_MARKER_TERMINATE_TIMEOUT_SECONDS = 5.0


def load_runtime_plan(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime_plan_units(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = plan.get("units", [])
    return [unit for unit in units if isinstance(unit, dict)]


def runtime_plan_unit_ids(plan: Mapping[str, Any]) -> list[str]:
    return [str(unit["unit_id"]) for unit in runtime_plan_units(plan) if str(unit.get("unit_id", "")).strip()]


def find_runtime_plan_unit(plan: Mapping[str, Any], unit_id: str) -> dict[str, Any]:
    normalized = str(unit_id).strip()
    for unit in runtime_plan_units(plan):
        if str(unit.get("unit_id", "")).strip() == normalized:
            return unit
    raise KeyError(f"Runtime plan does not define unit_id={normalized!r}.")


def execute_runtime_plan_steps(steps: Sequence[Mapping[str, Any]], *, cwd: str | Path) -> int:
    for step in steps:
        if _step_success_markers(step):
            exit_code = _execute_step_with_success_markers(step, cwd=cwd)
            if exit_code != 0:
                return exit_code
            continue
        completed = subprocess.run(
            step["command"],
            cwd=Path(cwd),
            check=False,
            capture_output=True,
            text=True,
            env=_step_environment(step),
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0 and not _step_reported_success(step, completed):
            return completed.returncode
    return 0 if steps else 1


def execute_runtime_plan_unit(plan: Mapping[str, Any], unit_id: str, *, cwd: str | Path) -> int:
    unit = find_runtime_plan_unit(plan, unit_id)
    return execute_runtime_plan_steps(unit.get("steps", []), cwd=cwd)


def write_runtime_plan_marker(
    path: str | Path,
    *,
    unit_id: str | None = None,
    step_count: int | None = None,
    unit_count: int | None = None,
) -> Path:
    marker_path = Path(path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if unit_id is not None:
        lines.append(f"unit_id={unit_id}")
    if step_count is not None:
        lines.append(f"steps={step_count}")
    if unit_count is not None:
        lines.append(f"units={unit_count}")
    marker_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return marker_path


def _execute_step_with_success_markers(step: Mapping[str, Any], *, cwd: str | Path) -> int:
    success_markers = _step_success_markers(step)
    if not success_markers:
        raise ValueError("Success-marker execution requires at least one success marker.")

    process = subprocess.Popen(
        step["command"],
        cwd=Path(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_step_environment(step),
    )
    if process.stdout is None:
        raise RuntimeError("Runtime plan success-marker execution requires stdout capture.")

    combined_output = ""
    try:
        for line in process.stdout:
            print(line, end="")
            sys.stdout.flush()
            combined_output += line
            if all(marker in combined_output for marker in success_markers):
                break
    finally:
        process.stdout.close()

    if all(marker in combined_output for marker in success_markers):
        _terminate_lingering_process_after_success(process)
        return 0
    return process.wait()


def _terminate_lingering_process_after_success(process: subprocess.Popen[str]) -> None:
    try:
        process.wait(timeout=SUCCESS_MARKER_EXIT_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        process.terminate()

    try:
        process.wait(timeout=SUCCESS_MARKER_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _step_reported_success(step: Mapping[str, Any], completed: subprocess.CompletedProcess[str]) -> bool:
    normalized_markers = _step_success_markers(step)
    if not normalized_markers:
        return False
    combined_output = "\n".join(
        part for part in (completed.stdout or "", completed.stderr or "") if part
    )
    return all(marker in combined_output for marker in normalized_markers)


def _step_success_markers(step: Mapping[str, Any]) -> tuple[str, ...]:
    success_markers = step.get("success_markers", [])
    if not isinstance(success_markers, Sequence) or isinstance(success_markers, (str, bytes)):
        return ()
    return tuple(str(marker).strip() for marker in success_markers if str(marker).strip())


def _step_environment(step: Mapping[str, Any]) -> dict[str, str] | None:
    raw_environment = step.get("env")
    if not isinstance(raw_environment, Mapping):
        return None

    environment = os.environ.copy()
    for raw_key, raw_value in raw_environment.items():
        key = str(raw_key).strip()
        value = str(raw_value).strip()
        if key and value:
            environment[key] = value
    return environment
