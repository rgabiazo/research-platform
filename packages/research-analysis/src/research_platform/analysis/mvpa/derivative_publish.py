from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .._version import DISTRIBUTION_NAME, package_version
from ._path_safety import (
    configured_path_is_unsafe,
    published_text_contains_local_path_reference,
    published_value_contains_local_path_reference,
)


DEFAULT_TARGET = "local_artifact"
DEFAULT_DERIVATIVE_NAME = "mvpa-crossnobis"
SUPPORTED_FIGURE_FORMATS = ("svg", "pdf", "png")


def validate_mvpa_derivative_publish_document(document: Mapping[str, Any] | Any) -> list[str]:
    if not isinstance(document, Mapping):
        return ["MVPA derivative publish config must contain a mapping."]
    payload = _payload(document)
    errors: list[str] = []
    publish_set = _optional_text(payload.get("name") or payload.get("id") or payload.get("publish_set"))
    if publish_set is None:
        errors.append("mvpa_derivative_publish.name must be defined.")
    elif not _safe_label(publish_set):
        errors.append("mvpa_derivative_publish.name must be a safe label.")

    derivative_name = _optional_text(payload.get("derivative_name")) or DEFAULT_DERIVATIVE_NAME
    if not _safe_label(derivative_name):
        errors.append("mvpa_derivative_publish.derivative_name must be a safe label.")

    targets = payload.get("targets")
    if not isinstance(targets, Mapping) or not targets:
        errors.append("mvpa_derivative_publish.targets must define at least one target mapping.")
    else:
        if DEFAULT_TARGET not in targets:
            errors.append(f"mvpa_derivative_publish.targets must define {DEFAULT_TARGET!r} as the default safe target.")
        for target_name, target_config in targets.items():
            target_label = str(target_name)
            if not _safe_label(target_label):
                errors.append(f"mvpa_derivative_publish.targets.{target_label} must be a safe label.")
            if not isinstance(target_config, Mapping):
                errors.append(f"mvpa_derivative_publish.targets.{target_label} must be a mapping.")
                continue
            root_ref = _optional_text(target_config.get("root_ref"))
            if root_ref is None:
                errors.append(f"mvpa_derivative_publish.targets.{target_label}.root_ref must be defined.")
            elif not _safe_label(root_ref):
                errors.append(f"mvpa_derivative_publish.targets.{target_label}.root_ref must be a safe label.")
            relative_path = _optional_text(target_config.get("relative_path") or target_config.get("path"))
            if relative_path is None:
                errors.append(f"mvpa_derivative_publish.targets.{target_label}.relative_path must be defined.")
            elif configured_path_is_unsafe(relative_path):
                errors.append(
                    f"mvpa_derivative_publish.targets.{target_label}.relative_path must be relative and stay under root_ref."
                )

    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        errors.append("mvpa_derivative_publish.inputs must be a mapping.")
    else:
        table_sets = inputs.get("table_sets")
        if not isinstance(table_sets, Sequence) or isinstance(table_sets, (str, bytes, bytearray)) or not table_sets:
            errors.append("mvpa_derivative_publish.inputs.table_sets must define at least one table set mapping.")
        else:
            for index, table_set in enumerate(table_sets, start=1):
                if not isinstance(table_set, Mapping):
                    errors.append(f"mvpa_derivative_publish.inputs.table_sets[{index}] must be a mapping.")
                    continue
                label = _optional_text(table_set.get("table_set") or table_set.get("name"))
                if label is None:
                    errors.append(f"mvpa_derivative_publish.inputs.table_sets[{index}].table_set must be defined.")
                elif not _safe_label(label):
                    errors.append(f"mvpa_derivative_publish.inputs.table_sets[{index}].table_set must be a safe label.")
                for key in ("distances", "audit"):
                    _validate_relative_path_value(
                        table_set.get(key),
                        f"mvpa_derivative_publish.inputs.table_sets[{index}].{key}",
                        errors,
                    )
                manifest = table_set.get("manifest")
                if manifest is not None:
                    _validate_relative_path_value(
                        manifest,
                        f"mvpa_derivative_publish.inputs.table_sets[{index}].manifest",
                        errors,
                    )

        rdm_set = inputs.get("rdm_set") or inputs.get("rdms")
        if not isinstance(rdm_set, Mapping):
            errors.append("mvpa_derivative_publish.inputs.rdm_set must be a mapping.")
        else:
            label = _optional_text(rdm_set.get("rdm_set") or rdm_set.get("name"))
            if label is None:
                errors.append("mvpa_derivative_publish.inputs.rdm_set.rdm_set must be defined.")
            elif not _safe_label(label):
                errors.append("mvpa_derivative_publish.inputs.rdm_set.rdm_set must be a safe label.")
            _validate_relative_path_value(rdm_set.get("root"), "mvpa_derivative_publish.inputs.rdm_set.root", errors)
            rdms = rdm_set.get("rdms")
            if not isinstance(rdms, Sequence) or isinstance(rdms, (str, bytes, bytearray)) or not rdms:
                errors.append("mvpa_derivative_publish.inputs.rdm_set.rdms must define at least one RDM mapping.")
            else:
                for index, rdm in enumerate(rdms, start=1):
                    if not isinstance(rdm, Mapping):
                        errors.append(f"mvpa_derivative_publish.inputs.rdm_set.rdms[{index}] must be a mapping.")
                        continue
                    rdm_id = _optional_text(rdm.get("rdm_id"))
                    if rdm_id is None:
                        errors.append(f"mvpa_derivative_publish.inputs.rdm_set.rdms[{index}].rdm_id must be defined.")
                    elif not _safe_label(rdm_id):
                        errors.append(f"mvpa_derivative_publish.inputs.rdm_set.rdms[{index}].rdm_id must be a safe label.")
                    basename = _optional_text(rdm.get("basename") or rdm.get("output_basename"))
                    if basename is None:
                        errors.append(f"mvpa_derivative_publish.inputs.rdm_set.rdms[{index}].basename must be defined.")
                    elif not _filename_stem(basename):
                        errors.append(f"mvpa_derivative_publish.inputs.rdm_set.rdms[{index}].basename must be a filename stem.")
                    publish_desc = _optional_text(rdm.get("publish_desc") or rdm.get("description_label"))
                    if publish_desc is not None and not _filename_stem(publish_desc):
                        errors.append(
                            f"mvpa_derivative_publish.inputs.rdm_set.rdms[{index}].publish_desc must be a filename stem."
                        )

        asset_groups = inputs.get("asset_groups") or inputs.get("assets")
        if asset_groups is not None:
            if not isinstance(asset_groups, Sequence) or isinstance(asset_groups, (str, bytes, bytearray)):
                errors.append("mvpa_derivative_publish.inputs.asset_groups must be a sequence when defined.")
            else:
                for index, asset_group in enumerate(asset_groups, start=1):
                    if not isinstance(asset_group, Mapping):
                        errors.append(f"mvpa_derivative_publish.inputs.asset_groups[{index}] must be a mapping.")
                        continue
                    group_id = _optional_text(asset_group.get("asset_group") or asset_group.get("name") or asset_group.get("id"))
                    if group_id is None:
                        errors.append(f"mvpa_derivative_publish.inputs.asset_groups[{index}].asset_group must be defined.")
                    elif not _safe_label(group_id):
                        errors.append(f"mvpa_derivative_publish.inputs.asset_groups[{index}].asset_group must be a safe label.")
                    root_ref = _optional_text(asset_group.get("root_ref"))
                    if root_ref is None:
                        errors.append(f"mvpa_derivative_publish.inputs.asset_groups[{index}].root_ref must be defined.")
                    elif not _safe_label(root_ref):
                        errors.append(f"mvpa_derivative_publish.inputs.asset_groups[{index}].root_ref must be a safe label.")
                    _validate_relative_path_value(
                        asset_group.get("source_glob"),
                        f"mvpa_derivative_publish.inputs.asset_groups[{index}].source_glob",
                        errors,
                    )
                    preserve_from = _optional_text(asset_group.get("preserve_from") or asset_group.get("source_base"))
                    if preserve_from is not None and configured_path_is_unsafe(preserve_from):
                        errors.append(
                            f"mvpa_derivative_publish.inputs.asset_groups[{index}].preserve_from must be relative and stay under root_ref."
                        )
                    destination_root = _optional_text(asset_group.get("destination_root") or asset_group.get("destination"))
                    if destination_root is not None and configured_path_is_unsafe(destination_root):
                        errors.append(
                            f"mvpa_derivative_publish.inputs.asset_groups[{index}].destination_root must be relative and stay under the publish root."
                        )

    for key in ("conditions", "contrasts", "rois"):
        values = payload.get(key, [])
        if values is not None and (
            not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray))
        ):
            errors.append(f"mvpa_derivative_publish.{key} must be a sequence when defined.")

    return errors


def plan_or_execute_mvpa_derivative_publish(
    document: Mapping[str, Any],
    *,
    workspace_root: str | Path,
    root_refs: Mapping[str, str | Path],
    target: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    payload = _payload(document)
    config_errors = validate_mvpa_derivative_publish_document(document)
    workspace = Path(workspace_root).resolve()
    roots = {str(key): Path(value).expanduser().resolve() for key, value in root_refs.items()}
    artifact_root = roots.get("artifact_root") or roots.get("artifacts_root") or workspace / "artifacts"
    publish_set = _optional_text(payload.get("name") or payload.get("id") or payload.get("publish_set")) or "mvpa_derivative_publish"
    selected_target, target_record, target_errors = _resolve_target(payload, roots=roots, target=target)
    target_root = target_record["path"] if target_record else None
    analysis_label = _optional_text(payload.get("analysis_label")) or _title_safe(publish_set)
    derivative_name = _optional_text(payload.get("derivative_name")) or DEFAULT_DERIVATIVE_NAME
    entities = _entities(payload)
    warnings: list[str] = []
    errors: list[str] = [*config_errors, *target_errors]

    source_records: list[dict[str, Any]] = []
    table_inputs: list[dict[str, Any]] = []
    rdm_inputs: list[dict[str, Any]] = []
    asset_inputs: list[dict[str, Any]] = []
    asset_outputs: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, Any]] = {}
    output_files: list[dict[str, Any]] = []
    rows: dict[str, list[dict[str, str]]] = {
        "group_distances": [],
        "group_audit": [],
        "group_summary": [],
        "inputs_manifest": [],
        "software_versions": [],
        "exclusions": [],
    }
    subject_outputs: dict[str, dict[str, list[dict[str, str]]]] = {}
    group_figures: list[dict[str, Any]] = []

    if target_root is not None:
        output_plan = _output_plan(payload, target_root=target_root, entities=entities, analysis_label=analysis_label)
        outputs = output_plan["outputs"]
        output_files = output_plan["output_files"]
    else:
        output_plan = {"outputs": {}, "output_files": []}

    if not errors:
        table_inputs, table_errors = _collect_table_inputs(payload, artifact_root=artifact_root, workspace_root=workspace)
        errors.extend(table_errors)
        source_records.extend(table_inputs)
        rdm_inputs, rdm_errors = _collect_rdm_inputs(payload, artifact_root=artifact_root, workspace_root=workspace)
        errors.extend(rdm_errors)
        source_records.extend(rdm_inputs)
        asset_inputs, asset_errors, asset_warnings = _collect_asset_inputs(
            payload,
            roots=roots,
            workspace_root=workspace,
        )
        errors.extend(asset_errors)
        warnings.extend(asset_warnings)
        source_records.extend(asset_inputs)

    if not errors:
        asset_outputs = _asset_output_records(asset_inputs, output_plan=output_plan)
        rows["group_distances"] = _concat_table_rows(table_inputs, key="distances_rows", source_key="table_set")
        rows["group_audit"] = _concat_table_rows(table_inputs, key="audit_rows", source_key="table_set")
        rows["group_summary"] = _concat_table_rows(rdm_inputs, key="summary_rows", source_key="rdm_id")
        rows["inputs_manifest"] = _source_manifest_rows(source_records)
        rows["software_versions"] = _software_version_rows(payload)
        rows["exclusions"] = _exclusion_rows(payload)
        subject_outputs = _subject_outputs(rdm_inputs)
        group_figures = _group_figure_records(rdm_inputs, output_plan=output_plan, entities=entities)
        _extend_subject_output_plan(output_plan, subject_outputs=subject_outputs, entities=entities, analysis_label=analysis_label)
        outputs = output_plan["outputs"]
        output_files = output_plan["output_files"]
        errors.extend(
            _published_text_leak_errors(
                payload,
                rows=rows,
                subject_outputs=subject_outputs,
                asset_outputs=asset_outputs,
                analysis_label=analysis_label,
                derivative_name=derivative_name,
            )
        )

    if execute and not errors:
        existing = [record["relative_path"] for record in output_files if Path(record["path"]).exists()]
        if existing:
            errors.append(f"MVPA derivative publish refuses to overwrite existing output(s): {', '.join(sorted(existing))}.")

    if execute and not errors:
        _write_publish_outputs(
            payload,
            output_plan=output_plan,
            rows=rows,
            source_records=source_records,
            subject_outputs=subject_outputs,
            rdm_inputs=rdm_inputs,
            group_figures=group_figures,
            asset_outputs=asset_outputs,
            analysis_label=analysis_label,
            derivative_name=derivative_name,
            entities=entities,
            selected_target=selected_target,
        )

    manifest = _publish_manifest(
        payload,
        publish_set=publish_set,
        analysis_label=analysis_label,
        derivative_name=derivative_name,
        selected_target=selected_target,
        target_record=target_record,
        outputs=outputs,
        source_records=source_records,
        rows=rows,
        subject_outputs=subject_outputs,
        warnings=warnings,
        errors=errors,
        executed=execute and not errors,
    )
    if execute and not errors:
        manifest_path = outputs.get("publish_manifest_json", {}).get("path")
        if manifest_path:
            _write_json_atomic(Path(manifest_path), manifest)

    return {
        "valid": not errors,
        "executed": execute and not errors,
        "publish_set": publish_set,
        "analysis_label": analysis_label,
        "derivative_name": derivative_name,
        "target": selected_target,
        "default_target": DEFAULT_TARGET,
        "dataset_derivatives_requires_explicit_target": selected_target != "dataset_derivatives" or target == "dataset_derivatives",
        "target_root": _target_root_record(target_record),
        "outputs": outputs,
        "output_count": len(output_files),
        "source_inputs": _source_input_records(source_records),
        "table_sets": [record["table_set"] for record in table_inputs],
        "rdm_set": _rdm_set(payload),
        "rdms": _rdm_summary_records(rdm_inputs),
        "asset_groups": _asset_group_summary_records(asset_inputs),
        "row_counts": _row_counts(rows, subject_outputs=subject_outputs),
        "published_absolute_paths_excluded": True,
        "preserves_editable_svg_pdf": True,
        "recomputed_values": False,
        "warnings": warnings,
        "errors": errors,
        "manifest": manifest,
    }


def _payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = document.get("mvpa_derivative_publish") or document.get("mvpa_derivatives") or document.get("mvpa_publish")
    return payload if isinstance(payload, Mapping) else document


def _resolve_target(
    payload: Mapping[str, Any],
    *,
    roots: Mapping[str, Path],
    target: str | None,
) -> tuple[str, dict[str, Any] | None, list[str]]:
    targets = payload.get("targets")
    if not isinstance(targets, Mapping):
        return target or DEFAULT_TARGET, None, ["mvpa_derivative_publish.targets must be a mapping."]
    selected = target or DEFAULT_TARGET
    if selected not in targets:
        return selected, None, [f"MVPA derivative publish target {selected!r} is not configured."]
    config = targets[selected]
    if not isinstance(config, Mapping):
        return selected, None, [f"MVPA derivative publish target {selected!r} must be a mapping."]
    root_ref = _optional_text(config.get("root_ref"))
    relative_path = _optional_text(config.get("relative_path") or config.get("path"))
    errors: list[str] = []
    if root_ref is None or root_ref not in roots:
        errors.append(f"MVPA derivative publish target {selected!r} root_ref {root_ref!r} is not known.")
    if relative_path is None or configured_path_is_unsafe(relative_path):
        errors.append(f"MVPA derivative publish target {selected!r} relative_path must be relative.")
    if errors:
        return selected, None, errors
    root = roots[root_ref]
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        errors.append(f"MVPA derivative publish target {selected!r} resolves outside root_ref {root_ref!r}.")
    if errors:
        return selected, None, errors
    return selected, {"root_ref": root_ref, "relative_path": relative_path, "path": path}, []


def _collect_table_inputs(
    payload: Mapping[str, Any],
    *,
    artifact_root: Path,
    workspace_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), Mapping) else {}
    table_sets = inputs.get("table_sets", []) if isinstance(inputs, Mapping) else []
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for table_config in table_sets:
        if not isinstance(table_config, Mapping):
            continue
        table_set = _optional_text(table_config.get("table_set") or table_config.get("name")) or "table_set"
        distances_path, distances_relpath = _source_path(table_config.get("distances"), artifact_root=artifact_root)
        audit_path, audit_relpath = _source_path(table_config.get("audit"), artifact_root=artifact_root)
        manifest_path: Path | None = None
        manifest_relpath: str | None = None
        if table_config.get("manifest") is not None:
            manifest_path, manifest_relpath = _source_path(table_config.get("manifest"), artifact_root=artifact_root)
        record = {
            "kind": "table_set",
            "table_set": table_set,
            "distances_path": distances_path,
            "distances_relpath": distances_relpath,
            "audit_path": audit_path,
            "audit_relpath": audit_relpath,
            "manifest_path": manifest_path,
            "manifest_relpath": manifest_relpath,
            "distances_rows": [],
            "audit_rows": [],
        }
        if not distances_path.is_file():
            errors.append(f"MVPA derivative publish source distances table is missing: {distances_relpath}.")
        else:
            record["distances_rows"] = _read_tsv(distances_path)
        if not audit_path.is_file():
            errors.append(f"MVPA derivative publish source audit table is missing: {audit_relpath}.")
        else:
            record["audit_rows"] = _read_tsv(audit_path)
        if manifest_path is not None and not manifest_path.is_file():
            errors.append(f"MVPA derivative publish source table manifest is missing: {manifest_relpath}.")
        record["source_files"] = [
            _source_file_record(distances_path, distances_relpath, workspace_root=workspace_root),
            _source_file_record(audit_path, audit_relpath, workspace_root=workspace_root),
        ]
        if manifest_path is not None and manifest_relpath is not None:
            record["source_files"].append(_source_file_record(manifest_path, manifest_relpath, workspace_root=workspace_root))
        records.append(record)
    return records, errors


def _collect_rdm_inputs(
    payload: Mapping[str, Any],
    *,
    artifact_root: Path,
    workspace_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), Mapping) else {}
    rdm_set = inputs.get("rdm_set") or inputs.get("rdms") if isinstance(inputs, Mapping) else {}
    if not isinstance(rdm_set, Mapping):
        return [], []
    root_path, root_relpath = _source_path(rdm_set.get("root"), artifact_root=artifact_root)
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for rdm_config in rdm_set.get("rdms", []):
        if not isinstance(rdm_config, Mapping):
            continue
        rdm_id = _optional_text(rdm_config.get("rdm_id")) or "rdm"
        basename = _optional_text(rdm_config.get("basename") or rdm_config.get("output_basename")) or rdm_id
        publish_desc = _optional_text(rdm_config.get("publish_desc") or rdm_config.get("description_label")) or _strip_mvpa_suffix(basename)
        source_files: list[dict[str, Any]] = []
        paths = {
            "matrix_tsv": root_path / f"{basename}_matrix.tsv",
            "long_tsv": root_path / f"{basename}_long.tsv",
            "subject_pairs_tsv": root_path / f"{basename}_subject-pairs.tsv",
            "summary_tsv": root_path / f"{basename}_summary.tsv",
            "manifest_json": root_path / f"{basename}_manifest.json",
        }
        for fmt in SUPPORTED_FIGURE_FORMATS:
            paths[f"figure_{fmt}"] = root_path / f"{basename}.{fmt}"
        relpaths = {key: f"{root_relpath}/{path.name}" for key, path in paths.items()}
        for key, path in paths.items():
            if not path.is_file():
                errors.append(f"MVPA derivative publish source RDM artifact is missing: {relpaths[key]}.")
            source_files.append(_source_file_record(path, relpaths[key], workspace_root=workspace_root))
        matrix_rows = _read_tsv(paths["matrix_tsv"]) if paths["matrix_tsv"].is_file() else []
        long_rows = _read_tsv(paths["long_tsv"]) if paths["long_tsv"].is_file() else []
        subject_pair_rows = _read_tsv(paths["subject_pairs_tsv"]) if paths["subject_pairs_tsv"].is_file() else []
        summary_rows = _read_tsv(paths["summary_tsv"]) if paths["summary_tsv"].is_file() else []
        records.append(
            {
                "kind": "rdm",
                "rdm_id": rdm_id,
                "basename": basename,
                "publish_desc": publish_desc,
                "config": dict(rdm_config),
                "paths": paths,
                "relpaths": relpaths,
                "source_files": source_files,
                "matrix_rows": matrix_rows,
                "long_rows": long_rows,
                "subject_pair_rows": subject_pair_rows,
                "summary_rows": summary_rows,
            }
        )
    return records, errors


def _collect_asset_inputs(
    payload: Mapping[str, Any],
    *,
    roots: Mapping[str, Path],
    workspace_root: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), Mapping) else {}
    asset_groups = inputs.get("asset_groups") or inputs.get("assets") if isinstance(inputs, Mapping) else []
    if not isinstance(asset_groups, Sequence) or isinstance(asset_groups, (str, bytes, bytearray)):
        return [], [], []
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for group_config in asset_groups:
        if not isinstance(group_config, Mapping):
            continue
        group_id = _optional_text(group_config.get("asset_group") or group_config.get("name") or group_config.get("id")) or "asset_group"
        root_ref = _optional_text(group_config.get("root_ref")) or ""
        source_glob = _optional_text(group_config.get("source_glob")) or ""
        preserve_from = _strip_slashes(_optional_text(group_config.get("preserve_from") or group_config.get("source_base")) or "")
        destination_root = _strip_slashes(_optional_text(group_config.get("destination_root") or group_config.get("destination")) or "")
        required = bool(group_config.get("required", False))
        source_files: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        if root_ref not in roots:
            errors.append(f"MVPA derivative publish asset group {group_id!r} root_ref {root_ref!r} is not known.")
            records.append(
                {
                    "kind": "asset_group",
                    "asset_group": group_id,
                    "root_ref": root_ref,
                    "source_glob": source_glob,
                    "preserve_from": preserve_from,
                    "destination_root": destination_root,
                    "file_count": 0,
                    "source_files": [],
                    "files": [],
                }
            )
            continue
        source_root = roots[root_ref]
        matched_paths = sorted(path for path in source_root.glob(source_glob) if path.is_file())
        if not matched_paths:
            message = f"MVPA derivative publish asset group {group_id!r} matched no files for {root_ref}/{source_glob}."
            if required:
                errors.append(message)
            else:
                warnings.append(message)
        for source_path in matched_paths:
            try:
                source_relpath = source_path.resolve().relative_to(source_root).as_posix()
            except ValueError:
                errors.append(f"MVPA derivative publish asset group {group_id!r} source escaped root_ref {root_ref!r}.")
                continue
            preserved_relpath = _preserved_asset_relpath(source_relpath, preserve_from)
            if preserved_relpath is None:
                errors.append(
                    f"MVPA derivative publish asset group {group_id!r} source {source_relpath!r} is not under preserve_from {preserve_from!r}."
                )
                continue
            destination_relpath = _join_relative(destination_root, preserved_relpath)
            source_record = _source_file_record(source_path, f"{root_ref}/{source_relpath}", workspace_root=workspace_root)
            sanitized_json: Any | None = None
            if source_path.name.endswith(".json"):
                sanitized_json, json_errors = _sanitized_asset_json(source_path, roots=roots)
                errors.extend(f"asset group {group_id!r}: {error}" for error in json_errors)
            source_files.append(source_record)
            file_record = {
                "source": source_path,
                "source_relpath": source_record["relative_path"],
                "destination_relpath": destination_relpath,
                "filename": source_path.name,
            }
            if sanitized_json is not None:
                file_record["sanitized_json"] = sanitized_json
            files.append(file_record)
        records.append(
            {
                "kind": "asset_group",
                "asset_group": group_id,
                "root_ref": root_ref,
                "source_glob": source_glob,
                "preserve_from": preserve_from,
                "destination_root": destination_root,
                "file_count": len(files),
                "source_files": source_files,
                "files": files,
            }
        )
    return records, errors, warnings


def _output_plan(
    payload: Mapping[str, Any],
    *,
    target_root: Path,
    entities: Mapping[str, str],
    analysis_label: str,
) -> dict[str, Any]:
    ses = entities.get("session_id", "ses-01")
    task = entities.get("task_id", "task")
    direction = entities.get("direction")
    prefix = _entity_prefix(session_id=ses, task_id=task, direction=direction)
    no_dir_prefix = _entity_prefix(session_id=ses, task_id=task, direction=None)
    outputs: dict[str, dict[str, Any]] = {}
    output_files: list[dict[str, Any]] = []

    def add(key: str, relative_path: str, *, copy_source: Path | None = None) -> None:
        path = target_root / relative_path
        record = {
            "relative_path": relative_path,
            "path": path.as_posix(),
            "filename": path.name,
            "status": "planned",
        }
        if copy_source is not None:
            record["copy_source"] = copy_source.as_posix()
        outputs[key] = record
        output_files.append(record)

    add("dataset_description_json", "dataset_description.json")
    add("readme_md", "README.md")
    add("config_yaml", f"config/task-{task}_desc-{analysis_label}_config.yaml")
    add("conditions_tsv", f"config/task-{task}_desc-{analysis_label}_conditions.tsv")
    add("contrasts_tsv", f"config/task-{task}_desc-{analysis_label}_contrasts.tsv")
    add("rois_tsv", f"config/task-{task}_desc-{analysis_label}ROIs.tsv")
    add("group_distances_tsv", f"group/{ses}/tables/{prefix}_desc-{analysis_label}_distances.tsv")
    add("group_distances_json", f"group/{ses}/tables/{prefix}_desc-{analysis_label}_distances.json")
    add("group_audit_tsv", f"group/{ses}/tables/{prefix}_desc-{analysis_label}_audit.tsv")
    add("group_audit_json", f"group/{ses}/tables/{prefix}_desc-{analysis_label}_audit.json")
    add("group_summary_tsv", f"group/{ses}/tables/{prefix}_desc-{analysis_label}_groupSummary.tsv")
    add("group_summary_json", f"group/{ses}/tables/{prefix}_desc-{analysis_label}_groupSummary.json")
    add("inputs_tsv", f"sourcedata/manifests/{no_dir_prefix}_desc-{analysis_label}_inputs.tsv")
    add("exclusions_tsv", f"sourcedata/manifests/{no_dir_prefix}_desc-{analysis_label}_exclusions.tsv")
    add("software_versions_tsv", f"sourcedata/manifests/{no_dir_prefix}_desc-{analysis_label}_softwareVersions.tsv")
    add("publish_manifest_json", f"sourcedata/manifests/{no_dir_prefix}_desc-{analysis_label}_publishManifest.json")
    return {"outputs": outputs, "output_files": output_files, "target_root": target_root}


def _asset_output_records(
    asset_inputs: Sequence[Mapping[str, Any]],
    *,
    output_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    target_root = Path(output_plan["target_root"])
    records: list[dict[str, Any]] = []
    for asset_group in asset_inputs:
        group_id = str(asset_group.get("asset_group", "asset_group"))
        for index, file_record in enumerate(asset_group.get("files", []), start=1):
            relative_path = str(file_record["destination_relpath"])
            target = target_root / relative_path
            key = f"asset_{group_id}_{index}"
            record = {
                "relative_path": relative_path,
                "path": target.as_posix(),
                "filename": target.name,
                "status": "planned",
                "asset_group": group_id,
                "copy_source_relpath": str(file_record["source_relpath"]),
            }
            output_plan["outputs"][key] = record
            output_plan["output_files"].append(record)
            records.append(
                {
                    "asset_group": group_id,
                    "source": file_record["source"],
                    "source_relpath": str(file_record["source_relpath"]),
                    "relative_path": relative_path,
                    "sanitized_json": file_record.get("sanitized_json"),
                }
            )
    return records


def _extend_subject_output_plan(
    output_plan: dict[str, Any],
    *,
    subject_outputs: Mapping[str, Mapping[str, list[dict[str, str]]]],
    entities: Mapping[str, str],
    analysis_label: str,
) -> None:
    target_root = Path(output_plan["target_root"])
    ses = entities.get("session_id", "ses-01")
    task = entities.get("task_id", "task")
    direction = entities.get("direction")
    for participant_id in sorted(subject_outputs):
        prefix = _entity_prefix(participant_id=participant_id, session_id=ses, task_id=task, direction=direction)
        for suffix in ("distances", "rdm", "qc"):
            for ext in ("tsv", "json"):
                key = f"subject_{participant_id}_{suffix}_{ext}"
                relative_path = f"{participant_id}/{ses}/rsa/{prefix}_desc-{analysis_label}_{suffix}.{ext}"
                path = target_root / relative_path
                record = {
                    "relative_path": relative_path,
                    "path": path.as_posix(),
                    "filename": path.name,
                    "status": "planned",
                }
                output_plan["outputs"][key] = record
                output_plan["output_files"].append(record)


def _group_figure_records(
    rdm_inputs: Sequence[Mapping[str, Any]],
    *,
    output_plan: Mapping[str, Any],
    entities: Mapping[str, str],
) -> list[dict[str, Any]]:
    target_root = Path(output_plan["target_root"])
    outputs = output_plan["outputs"]
    output_files = output_plan["output_files"]
    ses = entities.get("session_id", "ses-01")
    task = entities.get("task_id", "task")
    direction = entities.get("direction")
    prefix = _entity_prefix(session_id=ses, task_id=task, direction=direction)
    records: list[dict[str, Any]] = []
    for rdm in rdm_inputs:
        rdm_id = str(rdm["rdm_id"])
        publish_desc = str(rdm["publish_desc"])
        for fmt in SUPPORTED_FIGURE_FORMATS:
            source = Path(rdm["paths"][f"figure_{fmt}"])
            source_relpath = str(rdm["relpaths"][f"figure_{fmt}"])
            relative_path = f"group/{ses}/figures/{prefix}_desc-{publish_desc}.{fmt}"
            path = target_root / relative_path
            key = f"group_figure_{rdm_id}_{fmt}"
            record = {
                "relative_path": relative_path,
                "path": path.as_posix(),
                "filename": path.name,
                "status": "planned",
                "copy_source_relpath": source_relpath,
            }
            outputs[key] = record
            output_files.append(record)
            records.append(
                {
                    "rdm_id": rdm_id,
                    "format": fmt,
                    "source": source,
                    "source_relpath": source_relpath,
                    "relative_path": relative_path,
                }
            )
    return records


def _write_publish_outputs(
    payload: Mapping[str, Any],
    *,
    output_plan: Mapping[str, Any],
    rows: Mapping[str, list[dict[str, str]]],
    source_records: Sequence[Mapping[str, Any]],
    subject_outputs: Mapping[str, Mapping[str, list[dict[str, str]]]],
    rdm_inputs: Sequence[Mapping[str, Any]],
    group_figures: Sequence[Mapping[str, Any]],
    asset_outputs: Sequence[Mapping[str, Any]],
    analysis_label: str,
    derivative_name: str,
    entities: Mapping[str, str],
    selected_target: str,
) -> None:
    outputs = output_plan["outputs"]
    ses = entities.get("session_id", "ses-01")
    task = entities.get("task_id", "task")
    direction = entities.get("direction")
    prefix = _entity_prefix(session_id=ses, task_id=task, direction=direction)

    _write_json_atomic(Path(outputs["dataset_description_json"]["path"]), _dataset_description(payload, derivative_name=derivative_name))
    _write_text_atomic(Path(outputs["readme_md"]["path"]), _readme_text(analysis_label=analysis_label, derivative_name=derivative_name))
    _write_text_atomic(Path(outputs["config_yaml"]["path"]), json.dumps(_config_snapshot(payload), indent=2) + "\n")
    _write_tsv_atomic(Path(outputs["conditions_tsv"]["path"]), _condition_rows(payload))
    _write_tsv_atomic(Path(outputs["contrasts_tsv"]["path"]), _contrast_rows(payload))
    _write_tsv_atomic(Path(outputs["rois_tsv"]["path"]), _roi_rows(payload, rows["group_distances"]))
    _write_tsv_atomic(Path(outputs["group_distances_tsv"]["path"]), rows["group_distances"])
    _write_json_atomic(Path(outputs["group_distances_json"]["path"]), _sidecar("Subject-level MVPA crossnobis distances."))
    _write_tsv_atomic(Path(outputs["group_audit_tsv"]["path"]), rows["group_audit"])
    _write_json_atomic(Path(outputs["group_audit_json"]["path"]), _sidecar("Subject-level MVPA audit and provenance."))
    _write_tsv_atomic(Path(outputs["group_summary_tsv"]["path"]), rows["group_summary"])
    _write_json_atomic(Path(outputs["group_summary_json"]["path"]), _sidecar("Group RDM summary values."))
    _write_tsv_atomic(Path(outputs["inputs_tsv"]["path"]), rows["inputs_manifest"])
    _write_tsv_atomic(Path(outputs["exclusions_tsv"]["path"]), rows["exclusions"])
    _write_tsv_atomic(Path(outputs["software_versions_tsv"]["path"]), rows["software_versions"])

    for figure in group_figures:
        relative_path = str(figure["relative_path"])
        target = Path(output_plan["target_root"]) / relative_path
        _copy_file_atomic(Path(figure["source"]), target)

    for asset in asset_outputs:
        target = Path(output_plan["target_root"]) / str(asset["relative_path"])
        if asset.get("sanitized_json") is not None:
            _write_json_atomic(target, asset["sanitized_json"])
        else:
            _copy_file_atomic(Path(asset["source"]), target)

    for participant_id, participant_rows in subject_outputs.items():
        subject_prefix = _entity_prefix(participant_id=participant_id, session_id=ses, task_id=task, direction=direction)
        base = f"{participant_id}/{ses}/rsa/{subject_prefix}_desc-{analysis_label}"
        _write_tsv_atomic(Path(output_plan["target_root"]) / f"{base}_distances.tsv", participant_rows["distances"])
        _write_json_atomic(Path(output_plan["target_root"]) / f"{base}_distances.json", _sidecar("Subject RDM pair distances."))
        _write_tsv_atomic(Path(output_plan["target_root"]) / f"{base}_rdm.tsv", participant_rows["rdm"])
        _write_json_atomic(Path(output_plan["target_root"]) / f"{base}_rdm.json", _sidecar("Subject-level RDM matrices."))
        _write_tsv_atomic(Path(output_plan["target_root"]) / f"{base}_qc.tsv", participant_rows["qc"])
        _write_json_atomic(Path(output_plan["target_root"]) / f"{base}_qc.json", _sidecar("Subject-level MVPA derivative QC."))


def _subject_outputs(rdm_inputs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[dict[str, str]]]]:
    by_subject: dict[str, dict[str, list[dict[str, str]]]] = {}
    for rdm in rdm_inputs:
        subject_rows = rdm["subject_pair_rows"]
        rdm_id = str(rdm["rdm_id"])
        condition_ids = _condition_ids_for_rdm(rdm)
        pairs_by_subject: dict[str, list[dict[str, str]]] = {}
        for row in subject_rows:
            participant_id = row.get("participant_id", "")
            if not participant_id:
                continue
            pairs_by_subject.setdefault(participant_id, []).append(dict(row))
            by_subject.setdefault(participant_id, {"distances": [], "rdm": [], "qc": []})
            by_subject[participant_id]["distances"].append(
                {
                    "participant_id": participant_id,
                    "rdm_id": rdm_id,
                    "condition_a": row.get("condition_a", ""),
                    "condition_b": row.get("condition_b", ""),
                    "crossnobis": row.get("crossnobis", ""),
                    "source_contrast_id": row.get("source_contrast_id", ""),
                    "pooled_roi_count": row.get("pooled_roi_count", ""),
                    "pooled_row_count": row.get("pooled_row_count", ""),
                }
            )
        for participant_id, rows in pairs_by_subject.items():
            matrix_rows = _subject_matrix_rows(participant_id, rdm_id, condition_ids, rows)
            by_subject[participant_id]["rdm"].extend(matrix_rows)
            expected_pairs = len(condition_ids) * (len(condition_ids) - 1) // 2
            by_subject[participant_id]["qc"].append(
                {
                    "participant_id": participant_id,
                    "rdm_id": rdm_id,
                    "condition_count": str(len(condition_ids)),
                    "subject_pair_rows": str(len(rows)),
                    "expected_pair_count": str(expected_pairs),
                    "complete": str(len(rows) >= expected_pairs).lower(),
                }
            )
    return by_subject


def _subject_matrix_rows(
    participant_id: str,
    rdm_id: str,
    condition_ids: Sequence[str],
    pair_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    values: dict[tuple[str, str], str] = {}
    for row in pair_rows:
        a = row.get("condition_a", "")
        b = row.get("condition_b", "")
        value = row.get("crossnobis", "")
        if a and b:
            values[(a, b)] = value
            values[(b, a)] = value
    rows: list[dict[str, str]] = []
    for condition in condition_ids:
        matrix_row = {"participant_id": participant_id, "rdm_id": rdm_id, "condition_id": condition}
        for column in condition_ids:
            matrix_row[column] = "0.0" if column == condition else values.get((condition, column), "")
        rows.append(matrix_row)
    return rows


def _condition_ids_for_rdm(rdm: Mapping[str, Any]) -> list[str]:
    config = rdm.get("config", {})
    if isinstance(config, Mapping):
        conditions = config.get("conditions")
        if isinstance(conditions, Sequence) and not isinstance(conditions, (str, bytes, bytearray)):
            ids = [_optional_text(condition.get("condition_id")) for condition in conditions if isinstance(condition, Mapping)]
            ids = [condition_id for condition_id in ids if condition_id]
            if ids:
                return ids
    matrix_rows = rdm.get("matrix_rows", [])
    if isinstance(matrix_rows, Sequence) and matrix_rows:
        ids = [row.get("condition_id", "") for row in matrix_rows if isinstance(row, Mapping)]
        ids = [condition_id for condition_id in ids if condition_id]
        if ids:
            return ids
    ids = []
    for row in rdm.get("subject_pair_rows", []):
        if not isinstance(row, Mapping):
            continue
        for key in ("condition_a", "condition_b"):
            value = row.get(key, "")
            if value and value not in ids:
                ids.append(value)
    return ids


def _concat_table_rows(records: Sequence[Mapping[str, Any]], *, key: str, source_key: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        source_value = str(record.get(source_key, ""))
        for row in record.get(key, []):
            output_row = {str(column): str(value) for column, value in row.items()}
            if source_key == "rdm_id":
                output_row["source_rdm_id"] = source_value
            else:
                output_row["source_table_set"] = source_value
            rows.append(output_row)
    return rows


def _source_manifest_rows(source_records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in source_records:
        for source_file in record.get("source_files", []):
            rows.append(
                {
                    "source_kind": str(record.get("kind", "")),
                    "source_id": str(record.get("table_set") or record.get("rdm_id") or record.get("asset_group") or ""),
                    "source_relpath": str(source_file.get("relative_path", "")),
                    "exists": str(source_file.get("exists", False)).lower(),
                    "sha256": str(source_file.get("sha256", "")),
                    "size_bytes": str(source_file.get("size_bytes", "")),
                }
            )
    return rows


def _software_version_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    configured = payload.get("software_versions", [])
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes, bytearray)) and configured:
        return [{str(key): str(value) for key, value in row.items()} for row in configured if isinstance(row, Mapping)]
    return [{"name": DISTRIBUTION_NAME, "version": package_version(), "role": "publisher"}]


def _exclusion_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    configured = payload.get("exclusions", [])
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes, bytearray)) and configured:
        return [{str(key): str(value) for key, value in row.items()} for row in configured if isinstance(row, Mapping)]
    return [{"exclusion_id": "none", "description": "No derivative-publisher exclusions configured."}]


def _condition_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = payload.get("conditions", [])
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)) and rows:
        return [{str(key): str(value) for key, value in row.items()} for row in rows if isinstance(row, Mapping)]
    return []


def _contrast_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = payload.get("contrasts", [])
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)) and rows:
        return [{str(key): str(value) for key, value in row.items()} for row in rows if isinstance(row, Mapping)]
    return []


def _roi_rows(payload: Mapping[str, Any], group_distance_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = payload.get("rois", [])
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)) and rows:
        return [{str(key): str(value) for key, value in row.items()} for row in rows if isinstance(row, Mapping)]
    labels: list[str] = []
    for row in group_distance_rows:
        label = row.get("roi_label", "")
        if label and label not in labels:
            labels.append(label)
    return [{"roi_label": label} for label in labels]


def _publish_manifest(
    payload: Mapping[str, Any],
    *,
    publish_set: str,
    analysis_label: str,
    derivative_name: str,
    selected_target: str,
    target_record: Mapping[str, Any] | None,
    outputs: Mapping[str, Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    rows: Mapping[str, list[dict[str, str]]],
    subject_outputs: Mapping[str, Mapping[str, list[dict[str, str]]]],
    warnings: Sequence[str],
    errors: Sequence[str],
    executed: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "research_platform.analysis.mvpa.derivative_publish.v1",
        "publish_set": publish_set,
        "analysis_label": analysis_label,
        "derivative_name": derivative_name,
        "executed": executed,
        "target": selected_target,
        "target_relative_path": target_record.get("relative_path") if target_record else None,
        "outputs": {
            key: {"relative_path": value["relative_path"], "filename": value["filename"]}
            for key, value in sorted(outputs.items())
        },
        "source_inputs": _source_input_records(source_records),
        "row_counts": _row_counts(rows, subject_outputs=subject_outputs),
        "table_sets": [record["table_set"] for record in source_records if record.get("kind") == "table_set"],
        "rdm_set": _rdm_set(payload),
        "rdms": [record["rdm_id"] for record in source_records if record.get("kind") == "rdm"],
        "asset_groups": _asset_group_summary_records(source_records),
        "absolute_source_paths_excluded": True,
        "recomputed_values": False,
        "warnings": list(warnings),
        "errors": list(errors),
    }


def _row_counts(
    rows: Mapping[str, list[dict[str, str]]],
    *,
    subject_outputs: Mapping[str, Mapping[str, list[dict[str, str]]]],
) -> dict[str, Any]:
    return {
        "group_distances": len(rows.get("group_distances", [])),
        "group_audit": len(rows.get("group_audit", [])),
        "group_summary": len(rows.get("group_summary", [])),
        "subjects": len(subject_outputs),
        "subject_distance_rows": sum(len(record.get("distances", [])) for record in subject_outputs.values()),
        "subject_rdm_rows": sum(len(record.get("rdm", [])) for record in subject_outputs.values()),
        "subject_qc_rows": sum(len(record.get("qc", [])) for record in subject_outputs.values()),
    }


def _rdm_summary_records(rdm_inputs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rdm_id": str(record["rdm_id"]),
            "basename": str(record["basename"]),
            "publish_desc": str(record["publish_desc"]),
            "row_counts": {
                "matrix": len(record.get("matrix_rows", [])),
                "long": len(record.get("long_rows", [])),
                "subject_pairs": len(record.get("subject_pair_rows", [])),
                "summary": len(record.get("summary_rows", [])),
            },
        }
        for record in rdm_inputs
    ]


def _source_input_records(source_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in source_records:
        for source_file in record.get("source_files", []):
            rows.append(
                {
                    "source_kind": record.get("kind"),
                    "source_id": record.get("table_set") or record.get("rdm_id") or record.get("asset_group"),
                    "relative_path": source_file.get("relative_path"),
                    "exists": source_file.get("exists"),
                    "sha256": source_file.get("sha256"),
                    "size_bytes": source_file.get("size_bytes"),
                }
            )
    return rows


def _asset_group_summary_records(asset_inputs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in asset_inputs:
        if record.get("kind") != "asset_group":
            continue
        rows.append(
            {
                "asset_group": record.get("asset_group"),
                "root_ref": record.get("root_ref"),
                "source_glob": record.get("source_glob"),
                "preserve_from": record.get("preserve_from", ""),
                "destination_root": record.get("destination_root", ""),
                "file_count": int(record.get("file_count", 0)),
            }
        )
    return rows


def _sanitized_asset_json(path: Path, *, roots: Mapping[str, Path]) -> tuple[Any | None, list[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        return None, [f"JSON asset is not valid JSON: {path.name}: {exc}"]
    sanitized, errors = _sanitize_json_value(payload, roots=roots, context=path.name)
    errors.extend(_forbidden_text_errors(sanitized, label=f"sanitized asset JSON {path.name}"))
    return sanitized, errors


def _sanitize_json_value(value: Any, *, roots: Mapping[str, Path], context: str) -> tuple[Any, list[str]]:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        errors: list[str] = []
        for key, child in value.items():
            child_value, child_errors = _sanitize_json_value(child, roots=roots, context=f"{context}.{key}")
            sanitized[str(key)] = child_value
            errors.extend(child_errors)
        return sanitized, errors
    if isinstance(value, list):
        sanitized_list: list[Any] = []
        errors = []
        for index, child in enumerate(value):
            child_value, child_errors = _sanitize_json_value(child, roots=roots, context=f"{context}[{index}]")
            sanitized_list.append(child_value)
            errors.extend(child_errors)
        return sanitized_list, errors
    if isinstance(value, tuple):
        return _sanitize_json_value(list(value), roots=roots, context=context)
    if isinstance(value, str):
        return _sanitize_json_string(value, roots=roots, context=context)
    return value, []


def _sanitize_json_string(value: str, *, roots: Mapping[str, Path], context: str) -> tuple[Any, list[str]]:
    candidate = Path(value).expanduser()
    if not published_text_contains_local_path_reference(value):
        return value, []
    if not candidate.is_absolute():
        return value, [f"{context} contains an unmapped local absolute path: {value}"]
    mapped = _root_relative_reference(candidate, roots=roots)
    if mapped is not None:
        return mapped, []
    if _can_reduce_to_tool_name(candidate):
        return candidate.name, []
    return value, [f"{context} contains an unmapped local absolute path: {value}"]


def _root_relative_reference(path: Path, *, roots: Mapping[str, Path]) -> dict[str, str] | None:
    candidate = path.expanduser().resolve(strict=False)
    ordered_roots = sorted(
        ((root_ref, root.expanduser().resolve(strict=False)) for root_ref, root in roots.items()),
        key=lambda item: len(item[1].as_posix()),
        reverse=True,
    )
    for root_ref, root in ordered_roots:
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        return {"root_ref": str(root_ref), "relative_path": relative.as_posix()}
    return None


def _can_reduce_to_tool_name(path: Path) -> bool:
    name = path.name
    if not name or path.suffix:
        return False
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", name):
        return False
    return any(part in {"bin", "sbin", "Scripts"} for part in path.parts)


def _published_text_leak_errors(
    payload: Mapping[str, Any],
    *,
    rows: Mapping[str, list[dict[str, str]]],
    subject_outputs: Mapping[str, Mapping[str, list[dict[str, str]]]],
    asset_outputs: Sequence[Mapping[str, Any]],
    analysis_label: str,
    derivative_name: str,
) -> list[str]:
    values: list[tuple[str, Any]] = [
        ("dataset_description.json", _dataset_description(payload, derivative_name=derivative_name)),
        ("README.md", _readme_text(analysis_label=analysis_label, derivative_name=derivative_name)),
        ("config snapshot", _config_snapshot(payload)),
        ("conditions.tsv", _condition_rows(payload)),
        ("contrasts.tsv", _contrast_rows(payload)),
        ("rois.tsv", _roi_rows(payload, rows.get("group_distances", []))),
        ("runtime rows", rows),
        ("subject outputs", subject_outputs),
    ]
    for asset in asset_outputs:
        if asset.get("sanitized_json") is not None:
            values.append((f"asset JSON {asset.get('relative_path', '')}", asset["sanitized_json"]))
    errors: list[str] = []
    for label, value in values:
        errors.extend(_forbidden_text_errors(value, label=label))
    return errors


def _forbidden_text_errors(value: Any, *, label: str) -> list[str]:
    if published_value_contains_local_path_reference(value):
        return [f"Published text output {label} contains a local absolute path marker."]
    return []


def _source_file_record(path: Path, relpath: str, *, workspace_root: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "relative_path": relpath,
        "exists": exists,
        "sha256": _sha256(path) if exists else "",
        "size_bytes": str(path.stat().st_size) if exists else "",
    }


def _source_path(value: Any, *, artifact_root: Path) -> tuple[Path, str]:
    relative_path = _optional_text(value) or ""
    return (artifact_root / relative_path).resolve(), relative_path


def _preserved_asset_relpath(source_relpath: str, preserve_from: str) -> str | None:
    prefix = _strip_slashes(preserve_from)
    if not prefix:
        return source_relpath
    if source_relpath == prefix:
        return Path(source_relpath).name
    if source_relpath.startswith(f"{prefix}/"):
        return source_relpath[len(prefix) + 1 :]
    return None


def _join_relative(root: str, child: str) -> str:
    normalized_root = _strip_slashes(root)
    normalized_child = _strip_slashes(child)
    if not normalized_root:
        return normalized_child
    if not normalized_child:
        return normalized_root
    return f"{normalized_root}/{normalized_child}"


def _strip_slashes(value: str) -> str:
    normalized = value.strip().strip("/")
    return "" if normalized == "." else normalized


def _target_root_record(record: Mapping[str, Any] | None) -> dict[str, str] | None:
    if record is None:
        return None
    return {
        "root_ref": str(record["root_ref"]),
        "relative_path": str(record["relative_path"]),
        "path": Path(record["path"]).as_posix(),
    }


def _dataset_description(payload: Mapping[str, Any], *, derivative_name: str) -> dict[str, Any]:
    source_datasets = payload.get("source_datasets", [{"Description": "Source dataset is not included in this derivative."}])
    return {
        "Name": _optional_text(payload.get("dataset_description_name")) or derivative_name,
        "BIDSVersion": _optional_text(payload.get("bids_version")) or "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": DISTRIBUTION_NAME, "Version": package_version()}],
        "SourceDatasets": source_datasets,
    }


def _readme_text(*, analysis_label: str, derivative_name: str) -> str:
    return (
        f"# {derivative_name}\n\n"
        f"This derivative contains published MVPA crossnobis outputs for `{analysis_label}`.\n\n"
        "Crossnobis values are subject-level, model-derived representational distances. "
        "Group RDMs are group summaries after configured within-participant aggregation. "
        "When configured, copied ROI/localizer mask artifacts are stored under participant or group `func` directories. "
        "Raw BOLD data are not included. Source provenance is recorded under `sourcedata/manifests`.\n"
    )


def _config_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mvpa_derivative_publish": _json_safe(payload),
        "note": "This is a publish-time config snapshot. Absolute local source paths are intentionally excluded.",
    }


def _sidecar(description: str) -> dict[str, Any]:
    return {
        "Description": description,
        "GeneratedBy": "research-platform",
        "AbsoluteSourcePathsExcluded": True,
    }


def _entities(payload: Mapping[str, Any]) -> dict[str, str]:
    raw = payload.get("entities", {})
    if not isinstance(raw, Mapping):
        raw = {}
    session = _optional_text(payload.get("session_id") or raw.get("session_id")) or "ses-01"
    if session and not session.startswith("ses-"):
        session = f"ses-{session}"
    return {
        "session_id": session,
        "task_id": _optional_text(payload.get("task_id") or raw.get("task_id")) or "task",
        "direction": _optional_text(payload.get("direction") or raw.get("direction")) or "",
    }


def _entity_prefix(
    *,
    participant_id: str | None = None,
    session_id: str,
    task_id: str,
    direction: str | None,
) -> str:
    parts: list[str] = []
    if participant_id:
        parts.append(participant_id)
    parts.append(session_id)
    parts.append(f"task-{task_id}")
    if direction:
        parts.append(f"dir-{direction}")
    return "_".join(parts)


def _rdm_set(payload: Mapping[str, Any]) -> str | None:
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), Mapping) else {}
    rdm_set = inputs.get("rdm_set") or inputs.get("rdms") if isinstance(inputs, Mapping) else {}
    if not isinstance(rdm_set, Mapping):
        return None
    return _optional_text(rdm_set.get("rdm_set") or rdm_set.get("name"))


def _strip_mvpa_suffix(basename: str) -> str:
    text = basename
    for suffix in ("_mvpa", ".mvpa"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    match = re.search(r"_desc-([^_]+)", text)
    return match.group(1) if match else text


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _write_tsv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = _columns(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _copy_file_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
        with source.open("rb") as source_handle:
            shutil.copyfileobj(source_handle, handle)
        temp_path = Path(handle.name)
    temp_path.replace(target)


def _columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(str(column))
    return columns or ["empty"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _validate_relative_path_value(value: Any, label: str, errors: list[str]) -> None:
    text = _optional_text(value)
    if text is None:
        errors.append(f"{label} must be defined.")
    elif configured_path_is_unsafe(text):
        errors.append(f"{label} must be relative and stay under its source root.")


def _safe_label(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value))


def _filename_stem(value: str) -> bool:
    return bool(value) and "/" not in value and "\\" not in value and not value.startswith(".") and not Path(value).suffix


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _title_safe(value: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", value) if part) or value


__all__ = [
    "DEFAULT_TARGET",
    "plan_or_execute_mvpa_derivative_publish",
    "validate_mvpa_derivative_publish_document",
]
