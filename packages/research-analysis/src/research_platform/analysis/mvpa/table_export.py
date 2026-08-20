from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from ._path_safety import configured_path_is_unsafe, published_value_contains_local_path_reference


TABLE_A_BASE_COLUMNS = (
    "participant_id",
    "analysis_variant",
    "phase_id",
    "roi_label",
    "contrast_id",
    "crossnobis",
    "feature_count",
    "cv_unit_count",
    "observation_count",
)
AUDIT_COLUMNS = (
    "participant_id",
    "analysis_variant",
    "phase_id",
    "roi_label",
    "contrast_id",
    "subject_id",
    "session_id",
    "task_id",
    "mvpa_set",
    "family_id",
    "metric",
    "engine_name",
    "normalization_method",
    "source_distances_relpath",
)
REQUIRED_DISTANCE_COLUMNS = frozenset(
    {
        "subject_id",
        "session_id",
        "roi_label",
        "condition_pair_id",
        "distance",
        "metric",
        "engine_name",
        "normalization_method",
        "cv_unit_count",
        "feature_count",
        "observation_count",
    }
)
DEFAULT_SOURCE_RELATIVE_PATH = ".research-platform/mvpa/{mvpa_set}/analysis/prepared-distances/distances.tsv"
DEFAULT_OUTPUT_RELATIVE_PATH = ".research-platform/mvpa/reports/{table_set}"
OUTPUT_FILENAME_KEYS = frozenset({"subject_level_distances", "subject_level_audit", "manifest"})


def validate_mvpa_table_export_document(document: Mapping[str, Any] | Any) -> list[str]:
    if not isinstance(document, Mapping):
        return ["MVPA table export config must contain a mapping."]
    payload = _payload(document)
    errors: list[str] = []
    table_set = _optional_text(payload.get("name") or payload.get("id") or payload.get("table_set"))
    if table_set is None:
        errors.append("mvpa_table_export.name must be defined.")
    elif not _safe_label(table_set):
        errors.append("mvpa_table_export.name must be a safe label.")

    sources = payload.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes, bytearray)) or not sources:
        errors.append("mvpa_table_export.sources must define at least one source mapping.")
    else:
        seen: set[str] = set()
        for index, source in enumerate(sources, start=1):
            if not isinstance(source, Mapping):
                errors.append(f"mvpa_table_export.sources[{index}] must be a mapping.")
                continue
            mvpa_set = _optional_text(source.get("mvpa_set"))
            if mvpa_set is None:
                errors.append(f"mvpa_table_export.sources[{index}].mvpa_set must be defined.")
            elif not _safe_label(mvpa_set):
                errors.append(f"mvpa_table_export.sources[{index}].mvpa_set must be a safe label.")
            for key in ("phase_id", "analysis_variant", "family_id"):
                value = _optional_text(source.get(key))
                if value is None:
                    errors.append(f"mvpa_table_export.sources[{index}].{key} must be defined.")
                elif not _safe_label(value):
                    errors.append(f"mvpa_table_export.sources[{index}].{key} must be a safe label.")
            if mvpa_set is not None and mvpa_set in seen:
                errors.append(f"mvpa_table_export.sources[{index}].mvpa_set duplicates {mvpa_set!r}.")
            if mvpa_set is not None:
                seen.add(mvpa_set)
            expected_rows = source.get("expected_rows")
            if expected_rows is not None and _positive_or_zero_int(expected_rows) is None:
                errors.append(f"mvpa_table_export.sources[{index}].expected_rows must be a non-negative integer.")

    expected = payload.get("expected", {})
    if expected is not None and not isinstance(expected, Mapping):
        errors.append("mvpa_table_export.expected must be a mapping when defined.")
    elif isinstance(expected, Mapping):
        for key in ("total_rows", "subject_level_rows", "participant_count"):
            value = expected.get(key)
            if value is not None and _positive_or_zero_int(value) is None:
                errors.append(f"mvpa_table_export.expected.{key} must be a non-negative integer.")

    outputs = payload.get("outputs", {})
    if outputs is not None and not isinstance(outputs, Mapping):
        errors.append("mvpa_table_export.outputs must be a mapping when defined.")
    elif isinstance(outputs, Mapping):
        path = _optional_text(outputs.get("path") or outputs.get("relative_path"))
        if path is not None and configured_path_is_unsafe(path):
            errors.append("mvpa_table_export.outputs.path must be relative and stay under its root_ref.")
        root_ref = _optional_text(outputs.get("root_ref"))
        if root_ref is not None and not _safe_label(root_ref):
            errors.append("mvpa_table_export.outputs.root_ref must be a safe label.")
        filenames = outputs.get("filenames")
        if filenames is not None and not isinstance(filenames, Mapping):
            errors.append("mvpa_table_export.outputs.filenames must be a mapping when defined.")
        elif isinstance(filenames, Mapping):
            for key, value in filenames.items():
                key_text = str(key)
                if key_text not in OUTPUT_FILENAME_KEYS:
                    continue
                filename = _optional_text(value)
                if filename is None:
                    errors.append(f"mvpa_table_export.outputs.filenames.{key_text} must be non-empty.")
                elif configured_path_is_unsafe(filename) or "/" in filename or "\\" in filename:
                    errors.append(
                        f"mvpa_table_export.outputs.filenames.{key_text} must be a filename, not a path."
                    )

    return errors


def plan_or_execute_mvpa_table_export(
    document: Mapping[str, Any],
    *,
    workspace_root: str | Path,
    root_refs: Mapping[str, str | Path],
    execute: bool = False,
) -> dict[str, Any]:
    payload = _payload(document)
    config_errors = validate_mvpa_table_export_document(document)
    table_set = _optional_text(payload.get("name") or payload.get("id") or payload.get("table_set")) or "mvpa_table_export"
    workspace = Path(workspace_root).resolve()
    roots = {str(key): Path(value).expanduser().resolve() for key, value in root_refs.items()}
    artifact_root = roots.get("artifact_root") or roots.get("artifacts_root") or workspace / "artifacts"
    output_root, output_relative_root, root_errors = _output_root(payload, table_set=table_set, roots=roots)
    targets = _output_targets(payload, output_root=output_root, output_relative_root=output_relative_root)
    warnings: list[str] = []
    errors: list[str] = [*config_errors, *root_errors]

    source_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    if not errors:
        for source in _source_mappings(payload):
            source_result = _read_source_rows(
                source,
                payload,
                artifact_root=artifact_root,
                workspace_root=workspace,
            )
            source_records.append(source_result["record"])
            warnings.extend(source_result["warnings"])
            errors.extend(source_result["errors"])
            source_rows.extend(source_result["rows"])

    table_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    invariant_entities: dict[str, str] = {}
    table_a_columns = TABLE_A_BASE_COLUMNS
    audit_columns = AUDIT_COLUMNS
    manifest: dict[str, Any] | None = None

    if not errors:
        table_rows, audit_rows, invariant_entities, table_a_columns, build_warnings, build_errors = _build_export_rows(
            source_rows,
            payload,
        )
        warnings.extend(build_warnings)
        errors.extend(build_errors)

    if not errors:
        manifest = _manifest(
            table_set=table_set,
            targets=targets,
            source_records=source_records,
            table_rows=table_rows,
            audit_rows=audit_rows,
            table_a_columns=table_a_columns,
            audit_columns=audit_columns,
            invariant_entities=invariant_entities,
            warnings=warnings,
            executed=execute,
        )
        errors.extend(_expected_count_errors(payload, table_rows=table_rows, source_records=source_records))

    if execute and not errors:
        existing = [name for name, target in targets.items() if target["path"].exists()]
        if existing:
            errors.append(f"MVPA table export refuses to overwrite existing output(s): {', '.join(sorted(existing))}.")
        else:
            assert manifest is not None
            _write_export_targets(targets, table_rows=table_rows, audit_rows=audit_rows, manifest=manifest)

    if manifest is None:
        manifest = _manifest(
            table_set=table_set,
            targets=targets,
            source_records=source_records,
            table_rows=table_rows,
            audit_rows=audit_rows,
            table_a_columns=table_a_columns,
            audit_columns=audit_columns,
            invariant_entities=invariant_entities,
            warnings=warnings,
            executed=False,
        )

    valid = not errors
    return {
        "valid": valid,
        "executed": bool(execute and valid),
        "table_set": table_set,
        "output_root": {
            "relative_path": output_relative_root,
            "path": output_root.as_posix(),
        },
        "outputs": _json_targets(targets, executed=bool(execute and valid)),
        "source_mvpa_sets": [record["mvpa_set"] for record in source_records],
        "sources": source_records,
        "row_counts": {
            "table_a": len(table_rows),
            "audit": len(audit_rows),
            "participants": len({row["participant_id"] for row in table_rows if row.get("participant_id")}),
        },
        "table_a_columns": list(table_a_columns),
        "audit_table_columns": list(audit_columns),
        "manifest": manifest,
        "warnings": _unique(warnings),
        "errors": _unique(errors),
        "absolute_source_paths_excluded": True,
        "pooled_exports_implemented": False,
    }


def _payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = document.get("mvpa_table_export") or document.get("mvpa_table_set") or document
    return payload if isinstance(payload, Mapping) else {}


def _source_mappings(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    sources = payload.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes, bytearray)):
        return ()
    return tuple(source for source in sources if isinstance(source, Mapping))


def _output_root(
    payload: Mapping[str, Any],
    *,
    table_set: str,
    roots: Mapping[str, Path],
) -> tuple[Path, str, list[str]]:
    outputs = payload.get("outputs", {})
    output_mapping = outputs if isinstance(outputs, Mapping) else {}
    root_ref = _optional_text(output_mapping.get("root_ref")) or "artifact_root"
    relative_template = _optional_text(output_mapping.get("path") or output_mapping.get("relative_path")) or DEFAULT_OUTPUT_RELATIVE_PATH
    errors: list[str] = []
    if root_ref not in roots:
        errors.append(f"mvpa_table_export.outputs.root_ref {root_ref!r} is not a known root_ref.")
        root = Path(".").resolve()
    else:
        root = roots[root_ref]
    if configured_path_is_unsafe(relative_template):
        errors.append("mvpa_table_export.outputs.path must be relative and stay under its root_ref.")
        relative_path = DEFAULT_OUTPUT_RELATIVE_PATH.format(table_set=table_set)
    else:
        relative_path = _render_template(relative_template, table_set=table_set)
    return (root / relative_path).resolve(), relative_path, errors


def _output_targets(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    output_relative_root: str,
) -> dict[str, dict[str, Any]]:
    prefix = _filename_prefix(payload)
    files = {
        "subject_level_distances": f"{prefix}_desc-SubjectLevelCrossnobisDistances_mvpa.tsv",
        "subject_level_audit": f"{prefix}_desc-SubjectLevelCrossnobisAudit_mvpa.tsv",
        "manifest": f"{prefix}_desc-CrossnobisTables_manifest.json",
    }
    outputs = payload.get("outputs", {})
    output_mapping = outputs if isinstance(outputs, Mapping) else {}
    filenames = output_mapping.get("filenames", {})
    filename_mapping = filenames if isinstance(filenames, Mapping) else {}
    for key in tuple(files):
        override = _optional_text(filename_mapping.get(key))
        if override is not None:
            files[key] = override
    return {
        key: {
            "path": output_root / filename,
            "relative_path": f"{output_relative_root}/{filename}",
            "filename": filename,
        }
        for key, filename in files.items()
    }


def _filename_prefix(payload: Mapping[str, Any]) -> str:
    outputs = payload.get("outputs", {})
    output_mapping = outputs if isinstance(outputs, Mapping) else {}
    configured = _optional_text(output_mapping.get("filename_prefix"))
    if configured:
        return configured
    entities = payload.get("entities", {})
    entity_mapping = entities if isinstance(entities, Mapping) else {}
    session = _optional_text(entity_mapping.get("session_id") or entity_mapping.get("session"))
    task = _optional_text(entity_mapping.get("task_id") or entity_mapping.get("task"))
    pieces: list[str] = []
    if session:
        pieces.append(session)
    if task:
        pieces.append(f"task-{task.removeprefix('task-')}")
    return "_".join(pieces) if pieces else "desc-Crossnobis"


def _read_source_rows(
    source: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    artifact_root: Path,
    workspace_root: Path,
) -> dict[str, Any]:
    mvpa_set = str(source["mvpa_set"])
    relative_template = _optional_text(source.get("distances_path") or source.get("path")) or DEFAULT_SOURCE_RELATIVE_PATH
    relative_path = _render_template(relative_template, mvpa_set=mvpa_set)
    source_path = (artifact_root / relative_path).resolve()
    source_relpath = _relative_path(source_path, root=artifact_root, fallback_root=workspace_root)
    record = {
        "mvpa_set": mvpa_set,
        "phase_id": str(source["phase_id"]),
        "analysis_variant": str(source["analysis_variant"]),
        "family_id": str(source["family_id"]),
        "source_distances_relpath": source_relpath,
        "exists": source_path.is_file(),
        "expected_rows": _positive_or_zero_int(source.get("expected_rows")),
        "row_count": 0,
    }
    warnings: list[str] = []
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    if not source_path.is_file():
        return {"record": record, "rows": rows, "warnings": warnings, "errors": [f"Source distances TSV is missing for {mvpa_set}: {source_relpath}."]}

    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = tuple(reader.fieldnames or ())
        missing = sorted(REQUIRED_DISTANCE_COLUMNS.difference(fieldnames))
        if missing:
            return {
                "record": record,
                "rows": rows,
                "warnings": warnings,
                "errors": [f"Source {mvpa_set} distances TSV is missing required column(s): {', '.join(missing)}."],
            }
        for row_index, row in enumerate(reader, start=2):
            exported, row_errors = _source_row(source, payload, row, source_relpath=source_relpath, row_index=row_index)
            if row_errors:
                errors.extend(f"{mvpa_set} row {row_index}: {message}" for message in row_errors)
                continue
            rows.append(exported)
    record["row_count"] = len(rows)
    return {"record": record, "rows": rows, "warnings": warnings, "errors": errors}


def _source_row(
    source: Mapping[str, Any],
    payload: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    source_relpath: str,
    row_index: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    subject_id = _required_text(row.get("subject_id"), "subject_id", errors)
    session_id = _required_text(row.get("session_id") or _entity_value(payload, source, "session_id", "session"), "session_id", errors)
    task_id = _optional_text(row.get("task_id") or _entity_value(payload, source, "task_id", "task"))
    roi_label = _required_text(row.get("roi_label"), "roi_label", errors)
    contrast_id = _optional_text(row.get("condition_pair_id"))
    if contrast_id is None:
        contrast_id = _contrast_from_conditions(row)
    if contrast_id is None:
        errors.append("condition_pair_id must be non-empty or condition_id_a/condition_id_b must be available.")
    crossnobis = _finite_number(row.get("distance"), "distance", errors)
    feature_count = _integer_number(row.get("feature_count"), "feature_count", errors)
    cv_unit_count = _integer_number(row.get("cv_unit_count"), "cv_unit_count", errors)
    observation_count = _integer_number(row.get("observation_count"), "observation_count", errors)
    if errors:
        return {}, errors
    participant_id = _bids_participant_id(subject_id)
    return (
        {
            "participant_id": participant_id,
            "subject_id": subject_id,
            "session_id": session_id,
            "task_id": task_id or "",
            "mvpa_set": str(source["mvpa_set"]),
            "phase_id": str(source["phase_id"]),
            "analysis_variant": str(source["analysis_variant"]),
            "family_id": str(source["family_id"]),
            "roi_label": roi_label,
            "contrast_id": contrast_id,
            "crossnobis": crossnobis,
            "metric": str(row.get("metric") or ""),
            "engine_name": str(row.get("engine_name") or ""),
            "normalization_method": str(row.get("normalization_method") or ""),
            "feature_count": feature_count,
            "cv_unit_count": cv_unit_count,
            "observation_count": observation_count,
            "source_distances_relpath": source_relpath,
            "_source_row_index": row_index,
        },
        [],
    )


def _build_export_rows(
    rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], tuple[str, ...], list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    invariant_entities: dict[str, str] = {}
    variable_entity_columns: list[str] = []
    keep_entities = set(_text_sequence(_mapping(payload.get("table_a")).get("keep_entities")))
    for column in ("session_id", "task_id"):
        values = sorted({str(row.get(column) or "") for row in rows if str(row.get(column) or "")})
        if len(values) == 1 and column not in keep_entities:
            invariant_entities[column] = values[0]
        elif len(values) > 1 or column in keep_entities:
            variable_entity_columns.append(column)

    table_a_columns = (
        ("participant_id",)
        + tuple(variable_entity_columns)
        + tuple(column for column in TABLE_A_BASE_COLUMNS if column != "participant_id")
    )
    table_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen_keys: OrderedDict[tuple[Any, ...], int] = OrderedDict()
    duplicate_keys: list[tuple[Any, ...]] = []
    for row in rows:
        table_row = {column: row.get(column, "") for column in table_a_columns}
        audit_row = {column: row.get(column, "") for column in AUDIT_COLUMNS}
        table_rows.append(table_row)
        audit_rows.append(audit_row)
        key_columns = ("participant_id", *variable_entity_columns, "analysis_variant", "phase_id", "roi_label", "contrast_id")
        key = tuple(table_row.get(column) for column in key_columns)
        if key in seen_keys and key not in duplicate_keys:
            duplicate_keys.append(key)
        seen_keys[key] = seen_keys.get(key, 0) + 1
        if not table_row.get("participant_id"):
            errors.append("Table A contains a row with missing participant_id.")
        if table_row.get("crossnobis") is None:
            errors.append("Table A contains a row with missing crossnobis.")
    if duplicate_keys:
        errors.append(f"Table A contains duplicate participant/variant/phase/ROI/contrast key(s): {len(duplicate_keys)}.")
    if published_value_contains_local_path_reference([*table_rows, *audit_rows]):
        errors.append("Table A contains an absolute local path, which is not allowed.")
    return table_rows, audit_rows, invariant_entities, table_a_columns, warnings, errors


def _expected_count_errors(
    payload: Mapping[str, Any],
    *,
    table_rows: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    expected = _mapping(payload.get("expected"))
    total_expected = _positive_or_zero_int(expected.get("total_rows", expected.get("subject_level_rows")))
    if total_expected is not None and len(table_rows) != total_expected:
        errors.append(f"Expected {total_expected} subject-level row(s), found {len(table_rows)}.")
    participant_expected = _positive_or_zero_int(expected.get("participant_count"))
    participant_count = len({row.get("participant_id") for row in table_rows if row.get("participant_id")})
    if participant_expected is not None and participant_count != participant_expected:
        errors.append(f"Expected {participant_expected} participant(s), found {participant_count}.")
    for record in source_records:
        expected_rows = record.get("expected_rows")
        if expected_rows is not None and int(expected_rows) != int(record.get("row_count", 0)):
            errors.append(
                f"Source {record['mvpa_set']} expected {expected_rows} row(s), found {record.get('row_count', 0)}."
            )
    return errors


def _manifest(
    *,
    table_set: str,
    targets: Mapping[str, Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    table_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    table_a_columns: Sequence[str],
    audit_columns: Sequence[str],
    invariant_entities: Mapping[str, str],
    warnings: Sequence[str],
    executed: bool,
) -> dict[str, Any]:
    return {
        "table_set": table_set,
        "executed": executed,
        "outputs": {
            key: {"relative_path": str(target["relative_path"]), "filename": str(target["filename"])}
            for key, target in targets.items()
        },
        "source_mvpa_sets": [str(record["mvpa_set"]) for record in source_records],
        "row_counts": {
            "subject_level": len(table_rows),
            "audit": len(audit_rows),
            "participants": len({row.get("participant_id") for row in table_rows if row.get("participant_id")}),
        },
        "rows_by_source_mvpa_set": _count_by(audit_rows, "mvpa_set"),
        "rows_by_analysis_variant": _count_by(table_rows, "analysis_variant"),
        "rows_by_phase_id": _count_by(table_rows, "phase_id"),
        "rows_by_family_id": _count_by(audit_rows, "family_id"),
        "invariant_entities_moved_out_of_table_a": dict(invariant_entities),
        "table_a_columns": list(table_a_columns),
        "audit_table_columns": list(audit_columns),
        "absolute_source_paths_excluded": True,
        "pooled_exports_implemented": False,
        "warnings": list(_unique(warnings)),
    }


def _write_export_targets(
    targets: Mapping[str, Mapping[str, Any]],
    *,
    table_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    _write_tsv_atomic(targets["subject_level_distances"]["path"], table_rows, list(table_rows[0]) if table_rows else list(TABLE_A_BASE_COLUMNS))
    _write_tsv_atomic(targets["subject_level_audit"]["path"], audit_rows, list(AUDIT_COLUMNS))
    _write_text_atomic(targets["manifest"]["path"], json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")


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


def _json_targets(targets: Mapping[str, Mapping[str, Any]], *, executed: bool) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "relative_path": str(target["relative_path"]),
            "path": Path(target["path"]).as_posix(),
            "status": "written" if executed else "planned",
            "executed": executed,
        }
        for key, target in targets.items()
    }


def _count_by(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, int]:
    counts: OrderedDict[str, int] = OrderedDict()
    for row in rows:
        key = str(row.get(column) or "")
        counts[key] = counts.get(key, 0) + 1
    return dict(counts)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _entity_value(payload: Mapping[str, Any], source: Mapping[str, Any], *keys: str) -> str | None:
    for mapping in (source, _mapping(payload.get("entities"))):
        for key in keys:
            value = _optional_text(mapping.get(key))
            if value is not None:
                return value
    return None


def _contrast_from_conditions(row: Mapping[str, Any]) -> str | None:
    left = _optional_text(row.get("condition_id_a"))
    right = _optional_text(row.get("condition_id_b"))
    if left is None or right is None:
        return None
    return f"{left}_minus_{right}"


def _required_text(value: Any, label: str, errors: list[str]) -> str:
    text = _optional_text(value)
    if text is None:
        errors.append(f"{label} must be non-empty.")
        return ""
    return text


def _finite_number(value: Any, label: str, errors: list[str]) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be finite numeric.")
        return math.nan
    if not math.isfinite(number):
        errors.append(f"{label} must be finite numeric.")
    return number


def _integer_number(value: Any, label: str, errors: list[str]) -> int:
    number = _finite_number(value, label, errors)
    if not math.isfinite(number):
        return 0
    if not number.is_integer():
        errors.append(f"{label} must be an integer numeric value.")
        return 0
    return int(number)


def _positive_or_zero_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bids_participant_id(subject_id: str) -> str:
    text = subject_id.strip()
    if text.startswith("sub-"):
        return text
    if text.isdigit():
        return f"sub-{int(text):03d}"
    return f"sub-{text}"


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


def _safe_label(value: str) -> bool:
    return bool(value) and all(char.isalnum() or char in "._-" for char in value)


def _render_template(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _relative_path(path: Path, *, root: Path, fallback_root: Path) -> str:
    for candidate_root in (root, fallback_root):
        try:
            return path.relative_to(candidate_root).as_posix()
        except ValueError:
            continue
    return path.name


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return f"{value:.1f}"
        return format(value, ".17g")
    return str(value)


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
    "AUDIT_COLUMNS",
    "REQUIRED_DISTANCE_COLUMNS",
    "TABLE_A_BASE_COLUMNS",
    "plan_or_execute_mvpa_table_export",
    "validate_mvpa_table_export_document",
]
