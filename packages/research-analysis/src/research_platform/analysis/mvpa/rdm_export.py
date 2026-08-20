from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from ._path_safety import configured_path_is_unsafe, published_value_contains_local_path_reference


SUPPORTED_RDM_KINDS = ("rdm_heatmap",)
SUPPORTED_OUTPUT_FORMATS = ("svg", "pdf", "png")
DEFAULT_INPUT_RELATIVE_PATH = ".research-platform/mvpa/reports/{table_set}/{filename}"
DEFAULT_OUTPUT_RELATIVE_PATH = ".research-platform/mvpa/reports/{rdm_set}/rdms"
DEFAULT_DISTANCE_FILENAME = "{filename_prefix}_desc-SubjectLevelCrossnobisDistances_mvpa.tsv"
MATRIX_FIRST_COLUMN = "condition_id"
LONG_COLUMNS = (
    "rdm_id",
    "condition_a",
    "condition_b",
    "condition_a_label",
    "condition_b_label",
    "group_mean_crossnobis",
    "n",
    "sd",
    "sem",
    "ci_low",
    "ci_high",
    "source_contrast_id",
)
SUBJECT_PAIR_COLUMNS = (
    "participant_id",
    "rdm_id",
    "condition_a",
    "condition_b",
    "crossnobis",
    "source_contrast_id",
    "pooled_roi_count",
    "pooled_row_count",
)


def validate_mvpa_rdm_export_document(document: Mapping[str, Any] | Any) -> list[str]:
    if not isinstance(document, Mapping):
        return ["MVPA RDM export config must contain a mapping."]
    payload = _payload(document)
    errors: list[str] = []
    rdm_set = _optional_text(payload.get("name") or payload.get("id") or payload.get("rdm_set"))
    if rdm_set is None:
        errors.append("mvpa_rdm_export.name must be defined.")
    elif not _safe_label(rdm_set):
        errors.append("mvpa_rdm_export.name must be a safe label.")

    raw_input = payload.get("input") or payload.get("input_table")
    if raw_input is not None and not isinstance(raw_input, Mapping):
        errors.append("mvpa_rdm_export.input must be a mapping when defined.")
    elif isinstance(raw_input, Mapping):
        _validate_input_config(raw_input, label="mvpa_rdm_export.input", errors=errors)

    outputs = payload.get("outputs", {})
    if outputs is not None and not isinstance(outputs, Mapping):
        errors.append("mvpa_rdm_export.outputs must be a mapping when defined.")
    elif isinstance(outputs, Mapping):
        path = _optional_text(outputs.get("path") or outputs.get("relative_path"))
        if path is not None and configured_path_is_unsafe(path):
            errors.append("mvpa_rdm_export.outputs.path must be relative and stay under its root_ref.")
        root_ref = _optional_text(outputs.get("root_ref"))
        if root_ref is not None and not _safe_label(root_ref):
            errors.append("mvpa_rdm_export.outputs.root_ref must be a safe label.")

    rdms = payload.get("rdms")
    if not isinstance(rdms, Sequence) or isinstance(rdms, (str, bytes, bytearray)) or not rdms:
        errors.append("mvpa_rdm_export.rdms must define at least one RDM mapping.")
    else:
        seen: set[str] = set()
        for index, rdm in enumerate(rdms, start=1):
            if not isinstance(rdm, Mapping):
                errors.append(f"mvpa_rdm_export.rdms[{index}] must be a mapping.")
                continue
            errors.extend(_validate_rdm_mapping(rdm, index=index, seen_ids=seen))
    return errors


def plan_or_execute_mvpa_rdm_export(
    document: Mapping[str, Any],
    *,
    workspace_root: str | Path,
    root_refs: Mapping[str, str | Path],
    execute: bool = False,
) -> dict[str, Any]:
    payload = _payload(document)
    config_errors = validate_mvpa_rdm_export_document(document)
    rdm_set = _optional_text(payload.get("name") or payload.get("id") or payload.get("rdm_set")) or "mvpa_rdm_export"
    workspace = Path(workspace_root).resolve()
    roots = {str(key): Path(value).expanduser().resolve() for key, value in root_refs.items()}
    input_path, input_relative_path, input_errors = _input_table_path(payload, rdm_set=rdm_set, roots=roots)
    output_root, output_relative_root, output_errors = _output_root(payload, rdm_set=rdm_set, roots=roots)
    rdm_targets = {
        _rdm_id(rdm): _rdm_targets(rdm, output_root=output_root, output_relative_root=output_relative_root)
        for rdm in _rdm_mappings(payload)
    }
    errors: list[str] = [*config_errors, *output_errors]
    warnings: list[str] = []
    default_source_rows: list[dict[str, str]] = []
    default_input_columns: tuple[str, ...] = ()
    input_columns: tuple[str, ...] = ()
    if not errors and not input_errors and input_path.is_file():
        default_source_rows, default_input_columns = _read_subject_table(input_path)
        input_columns = default_input_columns

    rdm_results: list[dict[str, Any]] = []
    if not errors:
        for rdm in _rdm_mappings(payload):
            rdm_input_path, rdm_input_relative_path, rdm_input_errors = _rdm_input_table_path(
                payload,
                rdm,
                rdm_set=rdm_set,
                roots=roots,
            )
            rdm_input = {
                "relative_path": rdm_input_relative_path,
                "path": rdm_input_path.as_posix(),
                "exists": rdm_input_path.is_file(),
                "row_count": 0,
                "columns": [],
            }
            if rdm_input_errors:
                result = _empty_rdm_result(rdm, targets=rdm_targets[_rdm_id(rdm)], input_table=rdm_input)
                result["status"] = "error" if result["status"] != "disabled" else "disabled"
                result["errors"] = _unique([*result["errors"], *rdm_input_errors])
                rdm_results.append(result)
                errors.extend(result["errors"])
                continue
            if not rdm_input_path.is_file():
                result = _empty_rdm_result(rdm, targets=rdm_targets[_rdm_id(rdm)], input_table=rdm_input)
                if result["status"] != "disabled":
                    result["status"] = "error"
                    result["errors"] = [f"MVPA subject-level table is missing: {rdm_input_relative_path}."]
                    errors.extend(result["errors"])
                rdm_results.append(result)
                continue
            if rdm_input_path == input_path and default_source_rows:
                source_rows = default_source_rows
                input_columns = default_input_columns
            else:
                source_rows, input_columns = _read_subject_table(rdm_input_path)
            rdm_input["row_count"] = len(source_rows)
            rdm_input["columns"] = list(input_columns)
            result = _prepare_rdm(
                rdm,
                source_rows=source_rows,
                input_columns=input_columns,
                targets=rdm_targets[_rdm_id(rdm)],
                input_table=rdm_input,
            )
            rdm_results.append(result)
            warnings.extend(result["warnings"])
            errors.extend(result["errors"])
    else:
        for rdm in _rdm_mappings(payload):
            targets = rdm_targets[_rdm_id(rdm)]
            rdm_results.append(_empty_rdm_result(rdm, targets=targets))

    if execute and not errors:
        existing = [
            f"{rdm_result['rdm_id']}:{name}"
            for rdm_result in rdm_results
            if rdm_result["status"] != "disabled"
            for name, target in rdm_result["outputs"].items()
            if Path(target["path"]).exists()
        ]
        if existing:
            errors.append(f"MVPA RDM export refuses to overwrite existing output(s): {', '.join(sorted(existing))}.")

    if execute and not errors:
        try:
            plt = _load_matplotlib_pyplot()
        except ImportError as exc:
            errors.append(str(exc))
            plt = None
        if plt is not None:
            for rdm_result in rdm_results:
                if rdm_result["status"] == "disabled":
                    continue
                render_warnings = _render_and_write_rdm(
                    plt,
                    rdm_result,
                    input_relative_path=str(_mapping(rdm_result.get("input_table")).get("relative_path") or input_relative_path),
                    execute=execute,
                )
                rdm_result["layout_warnings"] = _unique([*rdm_result.get("layout_warnings", []), *render_warnings])
                rdm_result["warnings"] = _unique([*rdm_result["warnings"], *render_warnings])
                warnings.extend(render_warnings)
                if rdm_result["layout_warnings"] and bool(_mapping(rdm_result["config"]).get("fail_on_layout_warning")):
                    errors.append(f"RDM {rdm_result['rdm_id']} has layout warning(s).")

    valid = not errors
    return {
        "valid": valid,
        "executed": bool(execute and valid),
        "rdm_set": rdm_set,
        "input_table": {
            "relative_path": input_relative_path,
            "path": input_path.as_posix(),
            "exists": input_path.is_file(),
            "row_count": len(default_source_rows),
            "columns": list(input_columns),
            "errors": input_errors,
        },
        "output_root": {
            "relative_path": output_relative_root,
            "path": output_root.as_posix(),
        },
        "rdms": [
            _public_rdm_result(
                result,
                executed=bool(execute and valid),
                input_relative_path=str(_mapping(result.get("input_table")).get("relative_path") or input_relative_path),
            )
            for result in rdm_results
        ],
        "rdm_count": len(rdm_results),
        "enabled_rdm_count": sum(1 for result in rdm_results if result["status"] != "disabled"),
        "supported_rdm_kinds": list(SUPPORTED_RDM_KINDS),
        "warnings": _unique(warnings),
        "errors": _unique(errors),
    }


def _validate_rdm_mapping(rdm: Mapping[str, Any], *, index: int, seen_ids: set[str]) -> list[str]:
    errors: list[str] = []
    enabled = bool(rdm.get("enabled", True))
    rdm_id = _optional_text(rdm.get("rdm_id") or rdm.get("id"))
    if rdm_id is None:
        errors.append(f"mvpa_rdm_export.rdms[{index}].rdm_id must be defined.")
    elif not _safe_label(rdm_id):
        errors.append(f"mvpa_rdm_export.rdms[{index}].rdm_id must be a safe label.")
    elif rdm_id in seen_ids:
        errors.append(f"mvpa_rdm_export.rdms[{index}].rdm_id duplicates {rdm_id!r}.")
    if rdm_id is not None:
        seen_ids.add(rdm_id)
    raw_input = rdm.get("input") or rdm.get("input_table")
    if raw_input is not None and not isinstance(raw_input, Mapping):
        errors.append(f"mvpa_rdm_export.rdms[{index}].input must be a mapping when defined.")
    elif isinstance(raw_input, Mapping):
        _validate_input_config(raw_input, label=f"mvpa_rdm_export.rdms[{index}].input", errors=errors)
    kind = _optional_text(rdm.get("kind"))
    if kind not in SUPPORTED_RDM_KINDS:
        errors.append(f"mvpa_rdm_export.rdms[{index}].kind must be one of: {', '.join(SUPPORTED_RDM_KINDS)}.")
    value_column = _optional_text(rdm.get("value_column")) or "crossnobis"
    if not _safe_column(value_column):
        errors.append(f"mvpa_rdm_export.rdms[{index}].value_column must be a safe column name.")
    output_basename = _optional_text(rdm.get("output_basename"))
    if output_basename is None:
        errors.append(f"mvpa_rdm_export.rdms[{index}].output_basename must be defined.")
    elif configured_path_is_unsafe(output_basename) or "/" in output_basename:
        errors.append(f"mvpa_rdm_export.rdms[{index}].output_basename must be a filename stem.")
    formats = _output_formats(rdm)
    unknown_formats = sorted(set(formats).difference(SUPPORTED_OUTPUT_FORMATS))
    if unknown_formats:
        errors.append(
            f"mvpa_rdm_export.rdms[{index}].output_formats contains unsupported format(s): "
            f"{', '.join(unknown_formats)}."
        )

    conditions = _condition_mappings(rdm)
    if len(conditions) < 2:
        errors.append(f"mvpa_rdm_export.rdms[{index}].conditions must define at least two conditions.")
    condition_ids: list[str] = []
    for condition_index, condition in enumerate(conditions, start=1):
        condition_id = _optional_text(condition.get("condition_id") or condition.get("id"))
        if condition_id is None:
            errors.append(f"mvpa_rdm_export.rdms[{index}].conditions[{condition_index}].condition_id must be defined.")
        elif not _safe_label(condition_id):
            errors.append(f"mvpa_rdm_export.rdms[{index}].conditions[{condition_index}].condition_id must be a safe label.")
        else:
            condition_ids.append(condition_id)
    duplicates = sorted({condition_id for condition_id in condition_ids if condition_ids.count(condition_id) > 1})
    if duplicates:
        errors.append(f"mvpa_rdm_export.rdms[{index}].conditions contain duplicate condition_id(s): {', '.join(duplicates)}.")

    condition_set = set(condition_ids)
    pair_mappings = _pair_mappings(rdm)
    seen_pairs: set[tuple[str, str]] = set()
    seen_contrasts: set[str] = set()
    for pair_index, pair in enumerate(pair_mappings, start=1):
        contrast_id = _optional_text(pair.get("contrast_id") or pair.get("source_contrast_id"))
        condition_a = _optional_text(pair.get("condition_a"))
        condition_b = _optional_text(pair.get("condition_b"))
        if contrast_id is None:
            errors.append(f"mvpa_rdm_export.rdms[{index}].pair_mappings[{pair_index}].contrast_id must be defined.")
        elif not _safe_label(contrast_id):
            errors.append(f"mvpa_rdm_export.rdms[{index}].pair_mappings[{pair_index}].contrast_id must be a safe label.")
        elif contrast_id in seen_contrasts:
            errors.append(f"mvpa_rdm_export.rdms[{index}].pair_mappings[{pair_index}].contrast_id duplicates {contrast_id!r}.")
        if contrast_id is not None:
            seen_contrasts.add(contrast_id)
        for key, value in (("condition_a", condition_a), ("condition_b", condition_b)):
            if value is None:
                errors.append(f"mvpa_rdm_export.rdms[{index}].pair_mappings[{pair_index}].{key} must be defined.")
            elif value not in condition_set:
                errors.append(
                    f"mvpa_rdm_export.rdms[{index}].pair_mappings[{pair_index}].{key} "
                    f"{value!r} is not defined in conditions."
                )
        if condition_a and condition_b:
            if condition_a == condition_b:
                errors.append(f"mvpa_rdm_export.rdms[{index}].pair_mappings[{pair_index}] must map two different conditions.")
            pair_key = _pair_key(condition_a, condition_b)
            if pair_key in seen_pairs:
                errors.append(
                    f"mvpa_rdm_export.rdms[{index}].pair_mappings[{pair_index}] duplicates pair "
                    f"{pair_key[0]!r}/{pair_key[1]!r}."
                )
            seen_pairs.add(pair_key)

    if enabled and bool(rdm.get("strict_all_pairs", True)) and len(conditions) >= 2:
        missing_pairs = sorted(_required_pair_keys(condition_ids).difference(seen_pairs))
        if missing_pairs:
            formatted = ", ".join(f"{a}/{b}" for a, b in missing_pairs)
            errors.append(f"mvpa_rdm_export.rdms[{index}] strict_all_pairs is missing pair mapping(s): {formatted}.")

    aggregate = _mapping(rdm.get("aggregate_within_participant"))
    if aggregate:
        method = _optional_text(aggregate.get("method")) or "mean"
        if method != "mean":
            errors.append(f"mvpa_rdm_export.rdms[{index}].aggregate_within_participant.method must be mean.")
        if bool(aggregate.get("enabled", False)):
            across = _optional_text(aggregate.get("across"))
            if across is not None and not _safe_column(across):
                errors.append(f"mvpa_rdm_export.rdms[{index}].aggregate_within_participant.across must be a safe column name.")

    group_summary = _mapping(rdm.get("group_summary"))
    method = _optional_text(group_summary.get("method")) or "mean"
    if method != "mean":
        errors.append(f"mvpa_rdm_export.rdms[{index}].group_summary.method must be mean.")
    return errors


def _validate_input_config(config: Mapping[str, Any], *, label: str, errors: list[str]) -> None:
    path = _optional_text(config.get("path") or config.get("relative_path"))
    if path is not None and configured_path_is_unsafe(path):
        errors.append(f"{label}.path must be relative and stay under its root_ref.")
    root_ref = _optional_text(config.get("root_ref"))
    if root_ref is not None and not _safe_label(root_ref):
        errors.append(f"{label}.root_ref must be a safe label.")
    table_set = _optional_text(config.get("table_set"))
    if table_set is not None and not _safe_label(table_set):
        errors.append(f"{label}.table_set must be a safe label.")
    filename = _optional_text(config.get("filename"))
    if filename is not None and (configured_path_is_unsafe(filename) or "/" in filename or "\\" in filename):
        errors.append(f"{label}.filename must be a filename, not a path.")


def _payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = document.get("mvpa_rdm_export") or document.get("mvpa_rdm_set") or document
    return payload if isinstance(payload, Mapping) else {}


def _input_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    config = payload.get("input") or payload.get("input_table") or {}
    return config if isinstance(config, Mapping) else {}


def _input_config_for_rdm(payload: Mapping[str, Any], rdm: Mapping[str, Any]) -> Mapping[str, Any]:
    merged = dict(_input_config(payload))
    rdm_input = _input_config(rdm)
    if rdm_input:
        merged.update(rdm_input)
    return merged


def _rdm_mappings(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rdms = payload.get("rdms")
    if not isinstance(rdms, Sequence) or isinstance(rdms, (str, bytes, bytearray)):
        return ()
    return tuple(rdm for rdm in rdms if isinstance(rdm, Mapping))


def _condition_mappings(rdm: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    conditions = rdm.get("conditions")
    if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes, bytearray)):
        return ()
    return tuple(condition for condition in conditions if isinstance(condition, Mapping))


def _pair_mappings(rdm: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    pairs = rdm.get("pair_mappings") or rdm.get("pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes, bytearray)):
        return ()
    return tuple(pair for pair in pairs if isinstance(pair, Mapping))


def _input_table_path(
    payload: Mapping[str, Any],
    *,
    rdm_set: str,
    roots: Mapping[str, Path],
) -> tuple[Path, str, list[str]]:
    config = _input_config(payload)
    return _input_table_path_from_config(config, payload=payload, rdm_set=rdm_set, roots=roots, label="mvpa_rdm_export.input")


def _rdm_input_table_path(
    payload: Mapping[str, Any],
    rdm: Mapping[str, Any],
    *,
    rdm_set: str,
    roots: Mapping[str, Path],
) -> tuple[Path, str, list[str]]:
    config = _input_config_for_rdm(payload, rdm)
    label = f"mvpa_rdm_export.rdms.{_rdm_id(rdm)}.input"
    return _input_table_path_from_config(config, payload=payload, rdm_set=rdm_set, roots=roots, label=label)


def _input_table_path_from_config(
    config: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    rdm_set: str,
    roots: Mapping[str, Path],
    label: str,
) -> tuple[Path, str, list[str]]:
    root_ref = _optional_text(config.get("root_ref")) or "artifact_root"
    table_set = _optional_text(config.get("table_set")) or rdm_set
    filename_prefix = _optional_text(config.get("filename_prefix")) or _filename_prefix(payload)
    default_filename = DEFAULT_DISTANCE_FILENAME.format(filename_prefix=filename_prefix)
    filename = _optional_text(config.get("filename")) or default_filename
    relative_template = _optional_text(config.get("path") or config.get("relative_path")) or DEFAULT_INPUT_RELATIVE_PATH
    errors: list[str] = []
    if root_ref not in roots:
        errors.append(f"{label}.root_ref {root_ref!r} is not a known root_ref.")
        root = Path(".").resolve()
    else:
        root = roots[root_ref]
    if configured_path_is_unsafe(relative_template):
        errors.append(f"{label}.path must be relative and stay under its root_ref.")
        relative_path = DEFAULT_INPUT_RELATIVE_PATH.format(table_set=table_set, filename=filename)
    else:
        relative_path = _render_template(
            relative_template,
            rdm_set=rdm_set,
            table_set=table_set,
            filename=filename,
            filename_prefix=filename_prefix,
        )
    return (root / relative_path).resolve(), relative_path, errors


def _output_root(
    payload: Mapping[str, Any],
    *,
    rdm_set: str,
    roots: Mapping[str, Path],
) -> tuple[Path, str, list[str]]:
    outputs = _mapping(payload.get("outputs"))
    root_ref = _optional_text(outputs.get("root_ref")) or "artifact_root"
    relative_template = _optional_text(outputs.get("path") or outputs.get("relative_path")) or DEFAULT_OUTPUT_RELATIVE_PATH
    errors: list[str] = []
    if root_ref not in roots:
        errors.append(f"mvpa_rdm_export.outputs.root_ref {root_ref!r} is not a known root_ref.")
        root = Path(".").resolve()
    else:
        root = roots[root_ref]
    if configured_path_is_unsafe(relative_template):
        errors.append("mvpa_rdm_export.outputs.path must be relative and stay under its root_ref.")
        relative_path = DEFAULT_OUTPUT_RELATIVE_PATH.format(rdm_set=rdm_set)
    else:
        relative_path = _render_template(relative_template, rdm_set=rdm_set)
    return (root / relative_path).resolve(), relative_path, errors


def _rdm_targets(
    rdm: Mapping[str, Any],
    *,
    output_root: Path,
    output_relative_root: str,
) -> dict[str, dict[str, Any]]:
    basename = _optional_text(rdm.get("output_basename")) or _rdm_id(rdm)
    files: OrderedDict[str, str] = OrderedDict()
    for output_format in _output_formats(rdm):
        files[f"figure_{output_format}"] = f"{basename}.{output_format}"
    files["matrix_tsv"] = f"{basename}_matrix.tsv"
    files["long_tsv"] = f"{basename}_long.tsv"
    files["subject_pairs_tsv"] = f"{basename}_subject-pairs.tsv"
    files["summary_tsv"] = f"{basename}_summary.tsv"
    files["manifest_json"] = f"{basename}_manifest.json"
    return {
        key: {
            "path": output_root / filename,
            "relative_path": f"{output_relative_root}/{filename}",
            "filename": filename,
        }
        for key, filename in files.items()
    }


def _read_subject_table(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [dict(row) for row in reader]
        return rows, tuple(reader.fieldnames or ())


def _prepare_rdm(
    rdm: Mapping[str, Any],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    input_columns: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
    input_table: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not bool(rdm.get("enabled", True)):
        return _rdm_result(
            rdm,
            targets=targets,
            input_table=input_table,
            status="disabled",
            matrix_rows=[],
            long_rows=[],
            subject_pair_rows=[],
            summary_rows=[],
            warnings=[],
            errors=[],
        )
    rdm_id = _rdm_id(rdm)
    warnings: list[str] = []
    errors: list[str] = []
    required_columns = _required_columns(rdm)
    missing_columns = sorted(required_columns.difference(input_columns))
    if missing_columns:
        errors.append(f"RDM {rdm_id} input table is missing required column(s): {', '.join(missing_columns)}.")
        return _rdm_result(
            rdm,
            targets=targets,
            input_table=input_table,
            status="error",
            matrix_rows=[],
            long_rows=[],
            subject_pair_rows=[],
            summary_rows=[],
            warnings=warnings,
            errors=errors,
        )

    filtered_rows = _filter_rows(source_rows, _mapping(rdm.get("filters")))
    if not filtered_rows:
        errors.append(f"RDM {rdm_id} has no rows after filters.")

    conditions = _condition_records(rdm)
    pair_records = _pair_records(rdm, conditions)
    mapped_rows = _mapped_input_rows(rdm, filtered_rows, pair_records=pair_records, errors=errors)
    subject_pair_rows = _subject_pair_rows(rdm, mapped_rows=mapped_rows, pair_records=pair_records, errors=errors)
    long_rows = _long_summary_rows(rdm, subject_pair_rows=subject_pair_rows, pair_records=pair_records, conditions=conditions, errors=errors)
    matrix_rows = _matrix_rows(rdm, long_rows=long_rows, conditions=conditions, errors=errors)
    if published_value_contains_local_path_reference([*matrix_rows, *long_rows, *subject_pair_rows]):
        errors.append(f"RDM {rdm_id} export rows contain an absolute local path.")
    return _rdm_result(
        rdm,
        targets=targets,
        input_table=input_table,
        status="planned" if not errors else "error",
        matrix_rows=matrix_rows,
        long_rows=long_rows,
        subject_pair_rows=subject_pair_rows,
        summary_rows=long_rows,
        warnings=warnings,
        errors=errors,
    )


def _required_columns(rdm: Mapping[str, Any]) -> set[str]:
    columns = {"participant_id", "contrast_id", _value_column(rdm)}
    columns.update(str(column) for column in _mapping(rdm.get("filters")).keys())
    aggregate = _mapping(rdm.get("aggregate_within_participant"))
    across = _optional_text(aggregate.get("across"))
    if bool(aggregate.get("enabled", False)) and across:
        columns.add(across)
    return {column for column in columns if column}


def _condition_records(rdm: Mapping[str, Any]) -> OrderedDict[str, dict[str, str]]:
    records: OrderedDict[str, dict[str, str]] = OrderedDict()
    for condition in _condition_mappings(rdm):
        condition_id = str(condition.get("condition_id") or condition.get("id") or "")
        records[condition_id] = {
            "condition_id": condition_id,
            "label": _optional_text(condition.get("label")) or condition_id,
        }
    return records


def _pair_records(
    rdm: Mapping[str, Any],
    conditions: Mapping[str, Mapping[str, str]],
) -> OrderedDict[tuple[str, str], dict[str, str]]:
    records: OrderedDict[tuple[str, str], dict[str, str]] = OrderedDict()
    for pair in _pair_mappings(rdm):
        condition_a = str(pair.get("condition_a") or "")
        condition_b = str(pair.get("condition_b") or "")
        pair_key = _pair_key(condition_a, condition_b)
        records[pair_key] = {
            "condition_a": condition_a,
            "condition_b": condition_b,
            "condition_a_label": conditions.get(condition_a, {}).get("label", condition_a),
            "condition_b_label": conditions.get(condition_b, {}).get("label", condition_b),
            "source_contrast_id": str(pair.get("contrast_id") or pair.get("source_contrast_id") or ""),
        }
    return records


def _mapped_input_rows(
    rdm: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_records: Mapping[tuple[str, str], Mapping[str, str]],
    errors: list[str],
) -> list[dict[str, Any]]:
    rdm_id = _rdm_id(rdm)
    contrast_to_pair = {record["source_contrast_id"]: pair_key for pair_key, record in pair_records.items()}
    unmapped_contrasts: OrderedDict[str, None] = OrderedDict()
    mapped: list[dict[str, Any]] = []
    value_column = _value_column(rdm)
    for row in rows:
        contrast_id = str(row.get("contrast_id") or "")
        pair_key = contrast_to_pair.get(contrast_id)
        if pair_key is None:
            unmapped_contrasts[contrast_id] = None
            continue
        participant_id = _optional_text(row.get("participant_id"))
        if participant_id is None:
            errors.append(f"RDM {rdm_id} row has missing participant_id.")
            continue
        value = _finite_number(row.get(value_column), f"RDM {rdm_id} {value_column}", errors)
        if not math.isfinite(value):
            continue
        mapped.append(
            {
                "participant_id": participant_id,
                "pair_key": pair_key,
                "source_contrast_id": contrast_id,
                "value": value,
                "row": row,
            }
        )
    if unmapped_contrasts:
        formatted = ", ".join(str(contrast) for contrast in unmapped_contrasts if contrast)
        errors.append(f"RDM {rdm_id} has filtered rows with unmapped contrast_id(s): {formatted or '<blank>'}.")
    if bool(rdm.get("strict_all_pairs", True)):
        required_pairs = _required_pair_keys(list(_condition_records(rdm)))
        configured_pairs = set(pair_records)
        missing_config_pairs = sorted(required_pairs.difference(configured_pairs))
        if missing_config_pairs:
            errors.append(f"RDM {rdm_id} strict_all_pairs is missing configured pair(s): {_format_pairs(missing_config_pairs)}.")
        observed_pairs = {row["pair_key"] for row in mapped}
        missing_observed_pairs = sorted(required_pairs.difference(observed_pairs))
        if missing_observed_pairs:
            errors.append(f"RDM {rdm_id} strict_all_pairs has no rows for pair(s): {_format_pairs(missing_observed_pairs)}.")
    return mapped


def _subject_pair_rows(
    rdm: Mapping[str, Any],
    *,
    mapped_rows: Sequence[Mapping[str, Any]],
    pair_records: Mapping[tuple[str, str], Mapping[str, str]],
    errors: list[str],
) -> list[dict[str, Any]]:
    rdm_id = _rdm_id(rdm)
    aggregate = _mapping(rdm.get("aggregate_within_participant"))
    aggregate_enabled = bool(aggregate.get("enabled", False))
    across_column = _optional_text(aggregate.get("across"))
    groups: OrderedDict[tuple[str, tuple[str, str]], list[Mapping[str, Any]]] = OrderedDict()
    for row in mapped_rows:
        groups.setdefault((str(row["participant_id"]), row["pair_key"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for (participant_id, pair_key), group_rows in groups.items():
        if len(group_rows) > 1 and not aggregate_enabled:
            errors.append(
                f"RDM {rdm_id} has multiple rows for participant {participant_id!r} pair "
                f"{pair_key[0]!r}/{pair_key[1]!r}; configure aggregate_within_participant."
            )
            continue
        values = [float(row["value"]) for row in group_rows]
        value = sum(values) / len(values)
        pair = pair_records[pair_key]
        if across_column:
            pooled_roi_count = len({str(row["row"].get(across_column) or "") for row in group_rows if str(row["row"].get(across_column) or "")})
        else:
            pooled_roi_count = len(group_rows) if aggregate_enabled else 1
        rows.append(
            {
                "participant_id": participant_id,
                "rdm_id": rdm_id,
                "condition_a": pair["condition_a"],
                "condition_b": pair["condition_b"],
                "crossnobis": value,
                "source_contrast_id": pair["source_contrast_id"],
                "pooled_roi_count": pooled_roi_count,
                "pooled_row_count": len(group_rows),
            }
        )
    return rows


def _long_summary_rows(
    rdm: Mapping[str, Any],
    *,
    subject_pair_rows: Sequence[Mapping[str, Any]],
    pair_records: Mapping[tuple[str, str], Mapping[str, str]],
    conditions: Mapping[str, Mapping[str, str]],
    errors: list[str],
) -> list[dict[str, Any]]:
    rdm_id = _rdm_id(rdm)
    ci_level = float(_number_or_default(_mapping(rdm.get("group_summary")).get("ci_level"), 0.95))
    rows_by_pair: OrderedDict[tuple[str, str], list[Mapping[str, Any]]] = OrderedDict()
    for pair_key in _ordered_pair_keys(conditions):
        if pair_key in pair_records:
            rows_by_pair[pair_key] = []
    for row in subject_pair_rows:
        pair_key = _pair_key(str(row["condition_a"]), str(row["condition_b"]))
        rows_by_pair.setdefault(pair_key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for pair_key, rows in rows_by_pair.items():
        if not rows:
            continue
        participants = {str(row["participant_id"]) for row in rows if row.get("participant_id")}
        if len(participants) < 2:
            errors.append(f"RDM {rdm_id} pair {pair_key[0]!r}/{pair_key[1]!r} must have at least two participants.")
            continue
        values = [float(row["crossnobis"]) for row in rows]
        pair = pair_records[pair_key]
        sd = _sample_sd(values)
        sem = sd / math.sqrt(len(values)) if values else math.nan
        mean = sum(values) / len(values)
        margin = _t_critical(ci_level, len(values) - 1) * sem if len(values) > 1 else math.nan
        summaries.append(
            {
                "rdm_id": rdm_id,
                "condition_a": pair["condition_a"],
                "condition_b": pair["condition_b"],
                "condition_a_label": pair["condition_a_label"],
                "condition_b_label": pair["condition_b_label"],
                "group_mean_crossnobis": mean,
                "n": len(values),
                "sd": sd,
                "sem": sem,
                "ci_low": mean - margin,
                "ci_high": mean + margin,
                "source_contrast_id": pair["source_contrast_id"],
            }
        )
    return summaries


def _matrix_rows(
    rdm: Mapping[str, Any],
    *,
    long_rows: Sequence[Mapping[str, Any]],
    conditions: Mapping[str, Mapping[str, str]],
    errors: list[str],
) -> list[dict[str, Any]]:
    del errors
    diagonal_value = float(_number_or_default(rdm.get("diagonal_value"), 0.0))
    symmetric = bool(rdm.get("symmetric", True))
    labels = {condition_id: record["label"] for condition_id, record in conditions.items()}
    values: dict[tuple[str, str], float] = {}
    for row in long_rows:
        condition_a = str(row["condition_a"])
        condition_b = str(row["condition_b"])
        value = float(row["group_mean_crossnobis"])
        values[(condition_a, condition_b)] = value
        if symmetric:
            values[(condition_b, condition_a)] = value
    matrix: list[dict[str, Any]] = []
    for condition_a in conditions:
        row: OrderedDict[str, Any] = OrderedDict()
        row[MATRIX_FIRST_COLUMN] = condition_a
        for condition_b in conditions:
            label = labels[condition_b]
            if condition_a == condition_b:
                row[label] = diagonal_value
            else:
                row[label] = values.get((condition_a, condition_b), "")
        matrix.append(row)
    return [dict(row) for row in matrix]


def _filter_rows(rows: Sequence[Mapping[str, Any]], filters: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not filters:
        return list(rows)
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        keep = True
        for column, expected in filters.items():
            values = set(_text_sequence(expected))
            if not values:
                expected_text = _optional_text(expected)
                values = {expected_text} if expected_text is not None else set()
            if values and str(row.get(str(column)) or "") not in values:
                keep = False
                break
        if keep:
            selected.append(row)
    return selected


def _render_and_write_rdm(
    plt: Any,
    rdm_result: dict[str, Any],
    *,
    input_relative_path: str,
    execute: bool,
) -> list[str]:
    rdm = rdm_result["config"]
    targets = rdm_result["outputs"]
    fig, ax, annotations = _draw_rdm_heatmap(plt, rdm, matrix_rows=rdm_result["matrix_rows"], long_rows=rdm_result["long_rows"])
    layout_warnings = _layout_warnings(fig, ax, annotations=annotations, rdm=rdm)
    dpi = int(_number_or_default(rdm.get("dpi"), 300))
    matrix_columns = _matrix_columns(rdm_result["matrix_rows"])
    try:
        for output_format in _output_formats(rdm):
            target = Path(targets[f"figure_{output_format}"]["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, format=output_format, dpi=dpi)
        _write_tsv_atomic(Path(targets["matrix_tsv"]["path"]), rdm_result["matrix_rows"], matrix_columns)
        _write_tsv_atomic(Path(targets["long_tsv"]["path"]), rdm_result["long_rows"], LONG_COLUMNS)
        _write_tsv_atomic(Path(targets["subject_pairs_tsv"]["path"]), rdm_result["subject_pair_rows"], SUBJECT_PAIR_COLUMNS)
        _write_tsv_atomic(Path(targets["summary_tsv"]["path"]), rdm_result["summary_rows"], LONG_COLUMNS)
        manifest = _rdm_manifest(
            rdm_result,
            input_relative_path=input_relative_path,
            layout_warnings=layout_warnings,
            executed=execute,
        )
        _write_text_atomic(Path(targets["manifest_json"]["path"]), json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    finally:
        plt.close(fig)
    return layout_warnings


def _draw_rdm_heatmap(
    plt: Any,
    rdm: Mapping[str, Any],
    *,
    matrix_rows: Sequence[Mapping[str, Any]],
    long_rows: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, list[Any]]:
    del long_rows
    conditions = _condition_records(rdm)
    labels = [conditions[condition_id]["label"] for condition_id in conditions]
    values: list[list[float]] = []
    for row in matrix_rows:
        values.append([_matrix_float(row.get(label, math.nan)) for label in labels])
    size_config = _mapping(rdm.get("figure_size")) or _mapping(rdm.get("layout"))
    width = float(_number_or_default(size_config.get("width_inches"), max(4.8, len(labels) * 1.25 + 2.4)))
    height = float(_number_or_default(size_config.get("height_inches"), max(4.4, len(labels) * 1.05 + 1.8)))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        layout_engine.set(
            w_pad=float(_number_or_default(rdm.get("layout_w_pad"), 0.28)),
            h_pad=float(_number_or_default(rdm.get("layout_h_pad"), 0.24)),
            wspace=float(_number_or_default(rdm.get("layout_wspace"), 0.03)),
            hspace=float(_number_or_default(rdm.get("layout_hspace"), 0.03)),
        )
    cmap = str(rdm.get("cmap") or "viridis")
    vmin = _optional_number(rdm.get("vmin"))
    vmax = _optional_number(rdm.get("vmax"))
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    colorbar = fig.colorbar(
        image,
        ax=ax,
        shrink=float(_number_or_default(rdm.get("colorbar_shrink"), 0.80)),
        pad=float(_number_or_default(rdm.get("colorbar_pad"), 0.08)),
    )
    colorbar.set_label(str(rdm.get("colorbar_label") or _value_column(rdm)))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    rotation = int(_number_or_default(rdm.get("x_tick_rotation"), 35 if max((len(label) for label in labels), default=0) > 14 else 0))
    ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    ax.set_yticklabels(labels)
    ax.set_title(
        str(rdm.get("title") or _rdm_id(rdm)),
        pad=float(_number_or_default(rdm.get("title_pad"), 14)),
        fontsize=float(_number_or_default(rdm.get("title_fontsize"), 11)),
    )
    annotations: list[Any] = []
    if bool(rdm.get("annotate_cells", True)):
        decimals = int(_number_or_default(rdm.get("annotation_decimals"), 3))
        for y_index, row_values in enumerate(values):
            for x_index, value in enumerate(row_values):
                text_color = _annotation_text_color(value, image=image, rdm=rdm)
                text = "" if not math.isfinite(value) else f"{value:.{decimals}f}"
                annotations.append(ax.text(x_index, y_index, text, ha="center", va="center", color=text_color, fontsize=8))
    ax.tick_params(axis="both", pad=2.0)
    return fig, ax, annotations


def _layout_warnings(fig: Any, ax: Any, *, annotations: Sequence[Any], rdm: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    qc = _mapping(rdm.get("layout_qc"))
    max_title_chars = int(_number_or_default(qc.get("max_title_chars"), 120))
    title = _optional_text(rdm.get("title"))
    if title and len(title) > max_title_chars:
        warnings.append(f"title exceeds {max_title_chars} characters.")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    text_artists = [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *annotations]
    for artist in text_artists:
        text = _optional_text(artist.get_text()) if hasattr(artist, "get_text") else None
        if text is None:
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox.x0 < fig_bbox.x0 or bbox.x1 > fig_bbox.x1 or bbox.y0 < fig_bbox.y0 or bbox.y1 > fig_bbox.y1:
            warnings.append(f"text artist is outside figure bounds: {text[:40]}.")
    return _unique(warnings)


def _rdm_result(
    rdm: Mapping[str, Any],
    *,
    targets: Mapping[str, Mapping[str, Any]],
    input_table: Mapping[str, Any] | None = None,
    status: str,
    matrix_rows: Sequence[Mapping[str, Any]],
    long_rows: Sequence[Mapping[str, Any]],
    subject_pair_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
    errors: Sequence[str],
) -> dict[str, Any]:
    return {
        "rdm_id": _rdm_id(rdm),
        "kind": str(rdm.get("kind") or ""),
        "enabled": bool(rdm.get("enabled", True)),
        "status": status,
        "disabled_reason": _optional_text(rdm.get("disabled_reason")) if not bool(rdm.get("enabled", True)) else None,
        "config": dict(rdm),
        "input_table": dict(input_table or {}),
        "outputs": _json_targets(targets, executed=False, status="disabled" if status == "disabled" else "planned"),
        "matrix_rows": [dict(row) for row in matrix_rows],
        "long_rows": [dict(row) for row in long_rows],
        "subject_pair_rows": [dict(row) for row in subject_pair_rows],
        "summary_rows": [dict(row) for row in summary_rows],
        "row_counts": {
            "matrix": len(matrix_rows),
            "long": len(long_rows),
            "subject_pairs": len(subject_pair_rows),
            "summary": len(summary_rows),
        },
        "layout_warnings": [],
        "warnings": _unique(warnings),
        "errors": _unique(errors),
    }


def _empty_rdm_result(
    rdm: Mapping[str, Any],
    *,
    targets: Mapping[str, Mapping[str, Any]],
    input_table: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not bool(rdm.get("enabled", True)):
        return _rdm_result(
            rdm,
            targets=targets,
            input_table=input_table,
            status="disabled",
            matrix_rows=[],
            long_rows=[],
            subject_pair_rows=[],
            summary_rows=[],
            warnings=[],
            errors=[],
        )
    return _rdm_result(
        rdm,
        targets=targets,
        input_table=input_table,
        status="planned",
        matrix_rows=[],
        long_rows=[],
        subject_pair_rows=[],
        summary_rows=[],
        warnings=[],
        errors=[],
    )


def _public_rdm_result(result: Mapping[str, Any], *, executed: bool, input_relative_path: str) -> dict[str, Any]:
    public = {key: value for key, value in result.items() if key not in {"config"}}
    output_status = "disabled" if result["status"] == "disabled" else ("written" if executed else "planned")
    public["outputs"] = {
        key: {**dict(value), "status": output_status, "executed": bool(executed and result["status"] != "disabled")}
        for key, value in result["outputs"].items()
    }
    public["manifest"] = _rdm_manifest(
        result,
        input_relative_path=input_relative_path,
        layout_warnings=result.get("layout_warnings", []),
        executed=bool(executed and result["status"] != "disabled"),
    )
    return public


def _rdm_manifest(
    rdm_result: Mapping[str, Any],
    *,
    input_relative_path: str,
    layout_warnings: Sequence[str],
    executed: bool,
) -> dict[str, Any]:
    outputs = {
        key: {"relative_path": value["relative_path"], "filename": value["filename"]}
        for key, value in rdm_result["outputs"].items()
    }
    output_hashes = {
        key: _sha256(Path(value["path"]))
        for key, value in rdm_result["outputs"].items()
        if Path(value["path"]).is_file()
    }
    conditions = _condition_records(_mapping(rdm_result.get("config")))
    return {
        "schema_version": "research_platform.analysis.mvpa.rdm_export.v1",
        "rdm_id": rdm_result["rdm_id"],
        "kind": rdm_result["kind"],
        "enabled": rdm_result["enabled"],
        "status": rdm_result["status"],
        "executed": executed,
        "input_table_relpath": input_relative_path,
        "outputs": outputs,
        "output_hashes": output_hashes,
        "row_counts": rdm_result["row_counts"],
        "condition_count": len(conditions),
        "conditions": list(conditions.values()),
        "pair_count": len(rdm_result["long_rows"]),
        "strict_all_pairs": bool(_mapping(rdm_result.get("config")).get("strict_all_pairs", True)),
        "aggregation_within_participant_only": True,
        "matrix_columns": _matrix_columns(rdm_result["matrix_rows"]),
        "long_columns": list(LONG_COLUMNS),
        "subject_pair_columns": list(SUBJECT_PAIR_COLUMNS),
        "summary_columns": list(LONG_COLUMNS),
        "layout_qc": {
            "warnings": list(layout_warnings),
            "status": "warning" if layout_warnings else "ok",
        },
        "editable_text_settings": {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
        "absolute_source_paths_excluded": True,
        "warnings": _unique([*rdm_result["warnings"], *layout_warnings]),
        "errors": rdm_result["errors"],
        "disabled_reason": rdm_result.get("disabled_reason"),
    }


def _json_targets(targets: Mapping[str, Mapping[str, Any]], *, executed: bool, status: str | None = None) -> dict[str, dict[str, Any]]:
    resolved_status = status or ("written" if executed else "planned")
    return {
        key: {
            "relative_path": str(target["relative_path"]),
            "path": Path(target["path"]).as_posix(),
            "filename": str(target["filename"]),
            "status": resolved_status,
            "executed": executed,
        }
        for key, target in targets.items()
    }


def _load_matplotlib_pyplot() -> Any:
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "MVPA RDM exports require matplotlib. Install it locally with: uv add matplotlib "
            "or run with: uv run --with matplotlib ..."
        ) from exc
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    return plt


def _write_tsv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in columns})
    os.replace(tmp_path, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        handle.write(text)
    os.replace(tmp_path, path)


def _output_formats(rdm: Mapping[str, Any]) -> tuple[str, ...]:
    values = _text_sequence(rdm.get("output_formats"))
    return tuple(value.lower().lstrip(".") for value in values) or SUPPORTED_OUTPUT_FORMATS


def _rdm_id(rdm: Mapping[str, Any]) -> str:
    return str(rdm.get("rdm_id") or rdm.get("id") or "rdm")


def _filename_prefix(payload: Mapping[str, Any]) -> str:
    input_config = _input_config(payload)
    configured = _optional_text(input_config.get("filename_prefix"))
    if configured:
        return configured
    entities = _mapping(payload.get("entities"))
    session = _optional_text(entities.get("session_id") or entities.get("session"))
    task = _optional_text(entities.get("task_id") or entities.get("task"))
    pieces: list[str] = []
    if session:
        pieces.append(session)
    if task:
        pieces.append(f"task-{task.removeprefix('task-')}")
    return "_".join(pieces) if pieces else "desc-Crossnobis"


def _value_column(rdm: Mapping[str, Any]) -> str:
    return _optional_text(rdm.get("value_column")) or "crossnobis"


def _pair_key(condition_a: str, condition_b: str) -> tuple[str, str]:
    return tuple(sorted((condition_a, condition_b)))  # type: ignore[return-value]


def _required_pair_keys(condition_ids: Sequence[str]) -> set[tuple[str, str]]:
    return {_pair_key(a, b) for a, b in itertools.combinations(condition_ids, 2)}


def _ordered_pair_keys(conditions: Mapping[str, Mapping[str, str]]) -> list[tuple[str, str]]:
    return [_pair_key(a, b) for a, b in itertools.combinations(list(conditions), 2)]


def _format_pairs(pairs: Sequence[tuple[str, str]]) -> str:
    return ", ".join(f"{a}/{b}" for a, b in pairs)


def _matrix_columns(matrix_rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if not matrix_rows:
        return (MATRIX_FIRST_COLUMN,)
    return tuple(matrix_rows[0].keys())


def _annotation_text_color(value: float, *, image: Any, rdm: Mapping[str, Any]) -> str:
    configured = _optional_text(rdm.get("annotation_text_color"))
    if configured is None and not bool(rdm.get("annotation_text_contrast", True)):
        configured = "#111827"
    mode = (configured or "auto").lower()
    if mode in {"light", "white"}:
        return "#ffffff"
    if mode in {"dark", "black"}:
        return "#111827"
    if mode != "auto":
        return configured or "#111827"
    if not math.isfinite(value):
        return "#111827"
    rgba = image.cmap(image.norm(value))
    threshold = float(_number_or_default(rdm.get("annotation_luminance_threshold"), 0.52))
    return _annotation_text_color_for_rgba(rgba, threshold=threshold)


def _annotation_text_color_for_rgba(rgba: Sequence[float], *, threshold: float = 0.52) -> str:
    red, green, blue = (float(rgba[index]) for index in range(3))
    luminance = _relative_luminance(red, green, blue)
    return "#111827" if luminance >= threshold else "#ffffff"


def _relative_luminance(red: float, green: float, blue: float) -> float:
    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)


def _matrix_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _t_critical(ci_level: float, df: int) -> float:
    if df <= 0:
        return math.nan
    try:
        from scipy import stats

        return float(stats.t.ppf((1.0 + ci_level) / 2.0, df))
    except Exception:
        table_95 = {
            1: 12.7062047364,
            2: 4.3026527299,
            3: 3.1824463053,
            4: 2.7764451052,
            5: 2.5705818366,
            6: 2.4469118511,
            7: 2.3646242510,
            8: 2.3060041350,
            9: 2.2621571627,
            10: 2.2281388519,
            11: 2.2009851601,
            12: 2.1788128297,
            13: 2.1603686565,
            14: 2.1447866879,
            15: 2.1314495456,
            16: 2.1199052992,
            17: 2.1098155778,
            18: 2.1009220402,
            19: 2.0930240544,
            20: 2.0859634473,
            21: 2.0796138447,
            22: 2.0738730679,
            23: 2.0686576104,
            24: 2.0638985616,
            25: 2.0595385528,
            26: 2.0555294386,
            27: 2.0518305165,
            28: 2.0484071418,
            29: 2.0452296421,
            30: 2.0422724563,
        }
        if math.isclose(ci_level, 0.95) and df in table_95:
            return table_95[df]
        return 1.9599639845


def _finite_number(value: Any, label: str, errors: list[str]) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be finite numeric.")
        return math.nan
    if not math.isfinite(number):
        errors.append(f"{label} must be finite numeric.")
    return number


def _optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number_or_default(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(text for text in (_optional_text(item) for item in value) if text)
    text = _optional_text(value)
    return (text,) if text else ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_label(value: str) -> bool:
    return bool(value) and all(char.isalnum() or char in "._-" for char in value)


def _safe_column(value: str) -> bool:
    return bool(value) and all(char.isalnum() or char in "._-" for char in value)


def _render_template(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value.is_integer():
            return f"{value:.1f}"
        return format(value, ".17g")
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _unique(values: Sequence[str]) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for value in values:
        if value:
            seen[str(value)] = None
    return list(seen)


__all__ = [
    "LONG_COLUMNS",
    "MATRIX_FIRST_COLUMN",
    "SUBJECT_PAIR_COLUMNS",
    "SUPPORTED_OUTPUT_FORMATS",
    "SUPPORTED_RDM_KINDS",
    "plan_or_execute_mvpa_rdm_export",
    "validate_mvpa_rdm_export_document",
]
