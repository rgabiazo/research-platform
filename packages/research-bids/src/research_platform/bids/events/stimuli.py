from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from research_platform.neuro.events import BuildSpec


def _resolve_stimulus_source(stim_file: str, source_roots: list[Path]) -> Path:
    matches: list[Path] = []
    relpath = Path(stim_file)
    for root in source_roots:
        candidate = root / relpath
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved not in matches:
                matches.append(resolved)
    if len(matches) > 1:
        raise ValueError(
            f"Duplicate stimulus basename collision while resolving {stim_file!r}: "
            + ", ".join(str(path) for path in matches)
        )
    if matches:
        return matches[0]
    roots = ", ".join(str(root) for root in source_roots)
    raise ValueError(f"Unable to resolve stimulus source for {stim_file!r} under source roots: {roots}.")


def stage_stimuli(
    *,
    artifact_root: str | Path,
    run_rows: dict[int, list[dict[str, Any]]],
    spec: BuildSpec,
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, str]]]:
    if not spec.stimuli_enabled or not spec.stimuli_source_roots:
        return run_rows, []

    staged: dict[str, dict[str, str]] = {}
    staged_sources: dict[str, str] = {}
    rewritten_rows: dict[int, list[dict[str, Any]]] = {}
    stimuli_root = Path(artifact_root) / "staged" / "stimuli"

    for run, rows in run_rows.items():
        rewritten: list[dict[str, Any]] = []
        for row in rows:
            stim_file = str(row.get("stim_file", "")).strip()
            if not stim_file or stim_file == spec.missing_value:
                rewritten.append(dict(row))
                continue
            source_path = _resolve_stimulus_source(stim_file, spec.stimuli_source_roots)
            destination = stimuli_root / Path(stim_file).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination_key = str(destination)
            source_key = str(source_path.resolve())
            if destination_key in staged_sources and staged_sources[destination_key] != source_key:
                raise ValueError(
                    "Duplicate stimulus basename collision for staged target "
                    f"{Path('stimuli') / destination.name}: {staged_sources[destination_key]}, {source_key}"
                )
            if destination_key not in staged:
                shutil.copy2(source_path, destination)
                staged_sources[destination_key] = source_key
                staged[destination_key] = {
                    "staged_path": destination_key,
                    "publish_relpath": str(Path("stimuli") / destination.name),
                }
            updated = dict(row)
            updated["stim_file"] = str(Path("stimuli") / destination.name)
            rewritten.append(updated)
        rewritten_rows[run] = rewritten
    return rewritten_rows, [staged[path] for path in sorted(staged)]
