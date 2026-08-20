from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from .anchors import resolve_bold_anchor
from .manifest import write_build_manifest
from .paths import build_staged_output_path
from .stimuli import stage_stimuli
from .writers import write_events_tsv, write_sidecar_json

from research_platform.neuro.events import EventsSemanticResult, events_semantic_api  # noqa: E402


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_manifest_payload(
    *,
    spec: Any,
    source_path: str | Path,
    artifact_root: str | Path,
    outputs: list[dict[str, Any]],
    warnings: list[str],
    counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "input_path": str(Path(source_path)),
        "input_sha256": _file_sha256(source_path),
        "artifact_root": str(Path(artifact_root)),
        "spec_files": {
            "base": str(spec.source_path),
            "ops": str(spec.ops_path),
            "sidecar": str(spec.sidecar_path),
        },
        "spec_hash": spec.spec_hash,
        "outputs": outputs,
        "counts": counts,
        "warnings": warnings,
    }


def _build_sidecar_payload(spec: Any) -> dict[str, Any]:
    return {column: spec.sidecar_columns[column] for column in spec.columns if column in spec.sidecar_columns}


def _resolve_output_contexts(
    *,
    semantic_result: EventsSemanticResult,
    artifact_root: str | Path,
    dataset_root: str | Path | None,
) -> tuple[dict[int, list[dict[str, str]]], list[dict[str, Any]]]:
    spec = semantic_result.spec
    subject = semantic_result.subject
    resolved_session = semantic_result.session
    run_rows = semantic_result.run_rows_by_run()

    outputs = []
    for run, event_rows in sorted(run_rows.items()):
        anchor = None
        output_subject = subject
        output_session = resolved_session
        output_acq = spec.acq_label
        output_dir = spec.dir_label
        if dataset_root is not None:
            anchor = resolve_bold_anchor(
                dataset_root=dataset_root,
                spec=spec,
                subject=subject,
                session=resolved_session,
                run=run,
            )
            output_subject = anchor.subject
            output_session = anchor.session
            output_acq = anchor.acq
            output_dir = anchor.dir_label
        output_path = build_staged_output_path(
            artifact_root=artifact_root,
            spec=spec,
            subject=output_subject,
            session=output_session,
            run=run,
            acq_label=output_acq,
            dir_label=output_dir,
        )
        output_payload: dict[str, Any] = {
            "run": run,
            "resolved_entities": {
                "subject": output_subject,
                "session": output_session,
                "task": spec.task,
                "run": f"{run:02d}",
                "acq": output_acq,
                "dir": output_dir,
            },
            "staged_path": str(output_path),
            "publish_path": str(output_path.relative_to(Path(artifact_root) / "staged")),
            "row_count": len(event_rows),
        }
        if anchor is not None:
            output_payload["matched_anchor"] = str(anchor.anchor_path)
        outputs.append(output_payload)
    return run_rows, outputs


def plan_events(
    *,
    spec_path: str | Path,
    source_path: str | Path,
    artifact_root: str | Path,
    backend: str = "polars",
    dataset_root: str | Path | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    del backend
    semantic_result = events_semantic_api.plan(
        spec_path=spec_path,
        source_path=source_path,
        session=session,
    )
    _, outputs = _resolve_output_contexts(
        semantic_result=semantic_result,
        artifact_root=artifact_root,
        dataset_root=dataset_root,
    )
    return _build_manifest_payload(
        spec=semantic_result.spec,
        source_path=source_path,
        artifact_root=artifact_root,
        outputs=outputs,
        warnings=list(semantic_result.warnings),
        counts={"outputs": len(outputs), "rows": sum(int(output["row_count"]) for output in outputs), "stimuli": 0},
    )


def build_events(
    *,
    spec_path: str | Path,
    source_path: str | Path,
    artifact_root: str | Path,
    backend: str = "polars",
    dataset_root: str | Path | None = None,
    session: str | None = None,
    write_sidecars: bool = False,
    copy_stimuli: bool = False,
) -> Path:
    semantic_result = events_semantic_api.build(
        spec_path=spec_path,
        source_path=source_path,
        session=session,
        preserve_source_stim_file=copy_stimuli,
    )
    spec = semantic_result.spec
    run_rows, outputs = _resolve_output_contexts(
        semantic_result=semantic_result,
        artifact_root=artifact_root,
        dataset_root=dataset_root,
    )
    stimuli_entries: list[dict[str, str]] = []
    warnings = list(semantic_result.warnings)
    if write_sidecars and not spec.sidecar_writes:
        warnings.append("Sidecar writing was requested but disabled in the loaded sidecar spec.")
    if copy_stimuli and not spec.stimuli_enabled:
        warnings.append("Stimuli copying was requested but disabled in the loaded events spec.")
    if copy_stimuli:
        run_rows, stimuli_entries = stage_stimuli(artifact_root=artifact_root, run_rows=run_rows, spec=spec)
    outputs_by_run = {int(output["run"]): output for output in outputs}
    for run, event_rows in sorted(run_rows.items()):
        output_path = Path(outputs_by_run[run]["staged_path"])
        write_events_tsv(output_path, event_rows, spec.columns, backend=backend)
        outputs_by_run[run]["staged_path"] = str(output_path)
        if write_sidecars and spec.sidecar_writes:
            sidecar_path = output_path.with_suffix(".json")
            write_sidecar_json(sidecar_path, _build_sidecar_payload(spec))
            outputs_by_run[run]["staged_sidecar_path"] = str(sidecar_path)
            outputs_by_run[run]["publish_sidecar_path"] = str(sidecar_path.relative_to(Path(artifact_root) / "staged"))

    manifest_path = Path(artifact_root) / "manifests" / "build-manifest.json"
    manifest_payload = _build_manifest_payload(
        spec=spec,
        source_path=source_path,
        artifact_root=artifact_root,
        outputs=list(outputs_by_run.values()),
        warnings=warnings,
        counts={
            "outputs": len(outputs_by_run),
            "rows": semantic_result.total_row_count(),
            "stimuli": len(stimuli_entries),
        },
    )
    manifest_payload["stimuli"] = stimuli_entries
    manifest_payload["options"] = {
        "backend": backend,
        "write_sidecars": write_sidecars and spec.sidecar_writes,
        "copy_stimuli": copy_stimuli and spec.stimuli_enabled,
    }
    return write_build_manifest(
        manifest_path,
        manifest_payload,
    )


def publish_events(*, dataset_root: str | Path, manifest_path: str | Path, overwrite: bool = False) -> None:
    import json

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    staged_root = Path(payload["artifact_root"]) / "staged"
    file_entries: list[tuple[Path, Path]] = []
    for output in payload.get("outputs", []):
        staged_tsv = Path(output["staged_path"])
        file_entries.append((staged_tsv, Path(dataset_root) / staged_tsv.relative_to(staged_root)))
        if output.get("staged_sidecar_path"):
            staged_json = Path(output["staged_sidecar_path"])
            file_entries.append((staged_json, Path(dataset_root) / staged_json.relative_to(staged_root)))
    for stimulus in payload.get("stimuli", []):
        staged_path = Path(stimulus["staged_path"])
        file_entries.append((staged_path, Path(dataset_root) / stimulus["publish_relpath"]))

    conflicts = [destination for _, destination in file_entries if destination.exists()]
    if conflicts and not overwrite:
        conflict_list = ", ".join(str(path) for path in conflicts)
        raise ValueError(f"Publish target conflicts detected. Re-run with --overwrite to replace: {conflict_list}")

    for staged_path, destination in file_entries:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged_path, destination)
