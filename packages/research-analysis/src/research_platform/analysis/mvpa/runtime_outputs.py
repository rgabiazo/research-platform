from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = "research_platform.analysis.mvpa.runtime_outputs.v1"

_PATTERN_GROUP_OUTPUT_COLUMNS = (
    "group_id",
    "group_key",
    "group_by",
    "group_cv_unit",
    "group_cv_labels",
    "group_condition_ids",
    "group_feature_count",
    "group_voxel_order",
    "group_voxel_index_hash",
)
_PATTERN_PREPARED_ROW_COLUMNS = (
    "pattern_id",
    "condition_id",
    "cv_unit",
    "cv_label",
    "feature_values",
    "feature_count",
    "source_row_index",
    "voxel_order",
    "voxel_index_hash",
    "mean_centering_applied",
    "mean_centering_scope",
    "event_count",
    "noise_values",
    "noise_usable",
    "noise_feature_count",
    "noise_voxel_index_hash",
)
_PATTERN_ROW_COLUMNS = _PATTERN_GROUP_OUTPUT_COLUMNS + _PATTERN_PREPARED_ROW_COLUMNS
_PATTERN_QC_COLUMNS = (
    "level",
    "status",
    "code",
    "message",
    "source_row_index",
    "group_id",
    "group_key",
    "pattern_id",
    "condition_id",
    "cv_unit",
    "cv_label",
    "usable",
    "context",
)
_DISTANCE_ROW_COLUMNS = (
    "group_id",
    "group_key",
    "condition_id_a",
    "condition_id_b",
    "condition_pair_id",
    "distance",
    "metric",
    "engine_name",
    "normalization_method",
    "cv_unit_count",
    "feature_count",
    "observation_count",
    "context",
)
_DISTANCE_QC_COLUMNS = (
    "level",
    "status",
    "code",
    "message",
    "source",
    "source_row_index",
    "group_id",
    "group_key",
    "pattern_id",
    "condition_id",
    "condition_id_a",
    "condition_id_b",
    "condition_pair_id",
    "cv_unit",
    "cv_label",
    "usable",
    "context",
)
_SUMMARY_ROW_COLUMNS = (
    "group_id",
    "condition_id_a",
    "condition_id_b",
    "metric",
    "engine_name",
    "normalization_method",
    "n",
    "mean_distance",
    "std_distance",
    "sem_distance",
    "min_distance",
    "max_distance",
)
_SUMMARY_QC_COLUMNS = (
    "level",
    "status",
    "code",
    "message",
    "source",
    "source_row_index",
    "group_id",
    "group_key",
    "condition_id_a",
    "condition_id_b",
    "field_name",
    "context",
)


@dataclass(frozen=True)
class _ResolvedOutputPath:
    path: Path
    relative_path: str


def plan_prepared_mvpa_pattern_outputs(
    result: Any,
    *,
    output_root: str | Path,
    rows_path: str | Path,
    qc_path: str | Path,
    provenance_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Return a JSON-safe dry-run record for prepared MVPA pattern runtime outputs."""

    prepared = _prepared_pattern_output_payload(result)
    root, targets = _resolved_output_targets(
        output_root,
        {
            "rows": rows_path,
            "qc": qc_path,
            "provenance": provenance_path,
        },
    )
    return _outputs_record(
        artifact_kind="prepared_mvpa_pattern_outputs",
        output_root=root,
        targets=targets,
        row_counts={
            "rows": len(prepared["rows"]),
            "qc": len(prepared["qc_rows"]),
            "provenance": 1,
        },
        columns={
            "rows": prepared["row_columns"],
            "qc": prepared["qc_columns"],
            "provenance": (),
        },
        overwrite=overwrite,
        will_write=False,
        output_written=False,
    )


def write_prepared_mvpa_pattern_outputs(
    result: Any,
    *,
    output_root: str | Path,
    rows_path: str | Path,
    qc_path: str | Path,
    provenance_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write prepared MVPA pattern rows, QC rows, and provenance runtime outputs."""

    prepared = _prepared_pattern_output_payload(result)
    root, targets = _resolved_output_targets(
        output_root,
        {
            "rows": rows_path,
            "qc": qc_path,
            "provenance": provenance_path,
        },
    )
    _reject_existing_targets(targets, overwrite=overwrite)
    provenance = _pattern_provenance_payload(
        prepared,
        targets=targets,
        overwrite=overwrite,
    )
    rows_text = _tsv_text(prepared["rows"], prepared["row_columns"])
    qc_text = _tsv_text(prepared["qc_rows"], prepared["qc_columns"])
    provenance_text = _json_text(provenance)

    _write_text_atomic(targets["rows"].path, rows_text)
    _write_text_atomic(targets["qc"].path, qc_text)
    _write_text_atomic(targets["provenance"].path, provenance_text)

    return _outputs_record(
        artifact_kind="prepared_mvpa_pattern_outputs",
        output_root=root,
        targets=targets,
        row_counts={
            "rows": len(prepared["rows"]),
            "qc": len(prepared["qc_rows"]),
            "provenance": 1,
        },
        columns={
            "rows": prepared["row_columns"],
            "qc": prepared["qc_columns"],
            "provenance": (),
        },
        overwrite=overwrite,
        will_write=True,
        output_written=True,
    )


def plan_prepared_mvpa_distance_outputs(
    result: Any,
    *,
    output_root: str | Path,
    distances_path: str | Path,
    qc_path: str | Path,
    provenance_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Return a JSON-safe dry-run record for prepared MVPA distance runtime outputs."""

    prepared = _prepared_distance_output_payload(result)
    root, targets = _resolved_output_targets(
        output_root,
        {
            "distances": distances_path,
            "qc": qc_path,
            "provenance": provenance_path,
        },
    )
    return _outputs_record(
        artifact_kind="prepared_mvpa_distance_outputs",
        output_root=root,
        targets=targets,
        row_counts={
            "distances": len(prepared["distances"]),
            "qc": len(prepared["qc_rows"]),
            "provenance": 1,
        },
        columns={
            "distances": prepared["distance_columns"],
            "qc": prepared["qc_columns"],
            "provenance": (),
        },
        overwrite=overwrite,
        will_write=False,
        output_written=False,
    )


def write_prepared_mvpa_distance_outputs(
    result: Any,
    *,
    output_root: str | Path,
    distances_path: str | Path,
    qc_path: str | Path,
    provenance_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write prepared MVPA distance rows, QC rows, and provenance runtime outputs."""

    prepared = _prepared_distance_output_payload(result)
    root, targets = _resolved_output_targets(
        output_root,
        {
            "distances": distances_path,
            "qc": qc_path,
            "provenance": provenance_path,
        },
    )
    _reject_existing_targets(targets, overwrite=overwrite)
    provenance = _distance_provenance_payload(
        prepared,
        targets=targets,
        overwrite=overwrite,
    )
    distances_text = _tsv_text(prepared["distances"], prepared["distance_columns"])
    qc_text = _tsv_text(prepared["qc_rows"], prepared["qc_columns"])
    provenance_text = _json_text(provenance)

    _write_text_atomic(targets["distances"].path, distances_text)
    _write_text_atomic(targets["qc"].path, qc_text)
    _write_text_atomic(targets["provenance"].path, provenance_text)

    return _outputs_record(
        artifact_kind="prepared_mvpa_distance_outputs",
        output_root=root,
        targets=targets,
        row_counts={
            "distances": len(prepared["distances"]),
            "qc": len(prepared["qc_rows"]),
            "provenance": 1,
        },
        columns={
            "distances": prepared["distance_columns"],
            "qc": prepared["qc_columns"],
            "provenance": (),
        },
        overwrite=overwrite,
        will_write=True,
        output_written=True,
    )


def plan_prepared_mvpa_summary_outputs(
    result: Any,
    *,
    output_root: str | Path,
    summaries_path: str | Path,
    qc_path: str | Path,
    provenance_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Return a JSON-safe dry-run record for prepared MVPA summary runtime outputs."""

    prepared = _prepared_summary_output_payload(result)
    root, targets = _resolved_output_targets(
        output_root,
        {
            "summaries": summaries_path,
            "qc": qc_path,
            "provenance": provenance_path,
        },
    )
    return _outputs_record(
        artifact_kind="prepared_mvpa_distance_summary_outputs",
        output_root=root,
        targets=targets,
        row_counts={
            "summaries": len(prepared["summary_rows"]),
            "qc": len(prepared["qc_rows"]),
            "provenance": 1,
        },
        columns={
            "summaries": prepared["summary_columns"],
            "qc": prepared["qc_columns"],
            "provenance": (),
        },
        overwrite=overwrite,
        will_write=False,
        output_written=False,
    )


def write_prepared_mvpa_summary_outputs(
    result: Any,
    *,
    output_root: str | Path,
    summaries_path: str | Path,
    qc_path: str | Path,
    provenance_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write prepared MVPA summary rows, QC rows, and provenance runtime outputs."""

    prepared = _prepared_summary_output_payload(result)
    root, targets = _resolved_output_targets(
        output_root,
        {
            "summaries": summaries_path,
            "qc": qc_path,
            "provenance": provenance_path,
        },
    )
    _reject_existing_targets(targets, overwrite=overwrite)
    provenance = _summary_provenance_payload(
        prepared,
        targets=targets,
        overwrite=overwrite,
    )
    summaries_text = _tsv_text(prepared["summary_rows"], prepared["summary_columns"])
    qc_text = _tsv_text(prepared["qc_rows"], prepared["qc_columns"])
    provenance_text = _json_text(provenance)

    _write_text_atomic(targets["summaries"].path, summaries_text)
    _write_text_atomic(targets["qc"].path, qc_text)
    _write_text_atomic(targets["provenance"].path, provenance_text)

    return _outputs_record(
        artifact_kind="prepared_mvpa_distance_summary_outputs",
        output_root=root,
        targets=targets,
        row_counts={
            "summaries": len(prepared["summary_rows"]),
            "qc": len(prepared["qc_rows"]),
            "provenance": 1,
        },
        columns={
            "summaries": prepared["summary_columns"],
            "qc": prepared["qc_columns"],
            "provenance": (),
        },
        overwrite=overwrite,
        will_write=True,
        output_written=True,
    )


def write_mvpa_runtime_tsv(
    rows: Iterable[Mapping[str, Any]],
    *,
    columns: Sequence[str],
    output_root: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
    artifact_kind: str = "mvpa_runtime_tsv",
) -> dict[str, Any]:
    """Write a generic MVPA runtime TSV with the same path and serialization rules."""

    row_maps = tuple(_string_key_mapping(_as_mapping(row, field_name="rows item")) for row in rows)
    resolved_columns = tuple(str(column) for column in columns)
    root, targets = _resolved_output_targets(output_root, {"tsv": output_path})
    _reject_existing_targets(targets, overwrite=overwrite)
    text = _tsv_text(row_maps, resolved_columns)
    _write_text_atomic(targets["tsv"].path, text)
    return _outputs_record(
        artifact_kind=artifact_kind,
        output_root=root,
        targets=targets,
        row_counts={"tsv": len(row_maps)},
        columns={"tsv": resolved_columns},
        overwrite=overwrite,
        will_write=True,
        output_written=True,
    )


def write_mvpa_runtime_json(
    payload: Mapping[str, Any],
    *,
    output_root: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
    artifact_kind: str = "mvpa_runtime_json",
) -> dict[str, Any]:
    """Write a generic MVPA runtime JSON file with the same path and safety rules."""

    root, targets = _resolved_output_targets(output_root, {"json": output_path})
    _reject_existing_targets(targets, overwrite=overwrite)
    text = _json_text(payload)
    _write_text_atomic(targets["json"].path, text)
    return _outputs_record(
        artifact_kind=artifact_kind,
        output_root=root,
        targets=targets,
        row_counts={"json": 1},
        columns={"json": ()},
        overwrite=overwrite,
        will_write=True,
        output_written=True,
    )


def _prepared_pattern_output_payload(result: Any) -> dict[str, Any]:
    payload = _string_key_mapping(_as_mapping(result, field_name="result"))
    groups = _as_sequence(payload.get("groups", ()), field_name="groups")
    rows = _flatten_prepared_pattern_rows(groups)
    qc_rows = _known_rows(
        payload.get("qc_rows", ()),
        base_columns=_PATTERN_QC_COLUMNS,
        field_name="qc_rows",
    )
    return {
        "groups": groups,
        "rows": rows,
        "qc_rows": qc_rows,
        "row_columns": _columns_for_rows(_PATTERN_ROW_COLUMNS, rows),
        "qc_columns": _columns_for_rows(_PATTERN_QC_COLUMNS, qc_rows),
        "input_provenance": _input_provenance(payload.get("provenance", ())),
        "warnings": _json_safe_value(payload.get("warnings", ())),
        "errors": _json_safe_value(payload.get("errors", ())),
        "executed": _json_safe_value(payload.get("executed")),
    }


def _prepared_distance_output_payload(result: Any) -> dict[str, Any]:
    payload = _string_key_mapping(_as_mapping(result, field_name="result"))
    distances = _known_rows(
        payload.get("distances", ()),
        base_columns=_DISTANCE_ROW_COLUMNS,
        field_name="distances",
        flatten_group_key=True,
    )
    qc_rows = _known_rows(
        payload.get("qc_rows", ()),
        base_columns=_DISTANCE_QC_COLUMNS,
        field_name="qc_rows",
    )
    return {
        "distances": distances,
        "qc_rows": qc_rows,
        "distance_columns": _columns_for_rows(_DISTANCE_ROW_COLUMNS, distances),
        "qc_columns": _columns_for_rows(_DISTANCE_QC_COLUMNS, qc_rows),
        "input_provenance": _input_provenance(payload.get("provenance", ())),
        "warnings": _json_safe_value(payload.get("warnings", ())),
        "errors": _json_safe_value(payload.get("errors", ())),
        "executed": _json_safe_value(payload.get("executed")),
    }


def _prepared_summary_output_payload(result: Any) -> dict[str, Any]:
    payload = _string_key_mapping(_as_mapping(result, field_name="result"))
    summary_rows = _known_rows(
        payload.get("summary_rows", ()),
        base_columns=_SUMMARY_ROW_COLUMNS,
        field_name="summary_rows",
    )
    qc_rows = _known_rows(
        payload.get("qc_rows", ()),
        base_columns=_SUMMARY_QC_COLUMNS,
        field_name="qc_rows",
    )
    return {
        "summary_rows": summary_rows,
        "qc_rows": qc_rows,
        "summary_columns": _columns_for_rows(_SUMMARY_ROW_COLUMNS, summary_rows),
        "qc_columns": _columns_for_rows(_SUMMARY_QC_COLUMNS, qc_rows),
        "input_provenance": _input_provenance(payload.get("provenance", ())),
        "warnings": _json_safe_value(payload.get("warnings", ())),
        "errors": _json_safe_value(payload.get("errors", ())),
        "executed": _json_safe_value(payload.get("executed")),
    }


def _flatten_prepared_pattern_rows(groups: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    flattened: list[dict[str, Any]] = []
    for group_index, raw_group in enumerate(groups):
        group = _string_key_mapping(_as_mapping(raw_group, field_name=f"groups[{group_index}]"))
        group_values = {
            "group_id": group.get("group_id"),
            "group_key": group.get("group_key", {}),
            "group_by": group.get("group_by", ()),
            "group_cv_unit": group.get("cv_unit"),
            "group_cv_labels": group.get("cv_labels", ()),
            "group_condition_ids": group.get("condition_ids", ()),
            "group_feature_count": group.get("feature_count"),
            "group_voxel_order": group.get("voxel_order"),
            "group_voxel_index_hash": group.get("voxel_index_hash"),
        }
        group_key = group.get("group_key", {})
        if isinstance(group_key, Mapping):
            for key, value in group_key.items():
                column = str(key)
                if column not in group_values:
                    group_values[column] = value
        group_rows = _as_sequence(group.get("rows", ()), field_name=f"groups[{group_index}].rows")
        for row_index, raw_row in enumerate(group_rows):
            row = _string_key_mapping(
                _as_mapping(raw_row, field_name=f"groups[{group_index}].rows[{row_index}]")
            )
            flattened_row = dict(group_values)
            for column in _PATTERN_PREPARED_ROW_COLUMNS:
                flattened_row[column] = row.get(column)
            for column in sorted(row):
                if column not in flattened_row and column != "group_key":
                    flattened_row[column] = row[column]
            flattened.append(flattened_row)
    return tuple(flattened)


def _known_rows(
    value: Any,
    *,
    base_columns: Sequence[str],
    field_name: str,
    flatten_group_key: bool = False,
) -> tuple[dict[str, Any], ...]:
    rows = _as_sequence(value, field_name=field_name)
    normalized: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        row = _string_key_mapping(_as_mapping(raw_row, field_name=f"{field_name}[{index}]"))
        normalized_row = {column: row.get(column) for column in base_columns}
        if flatten_group_key and isinstance(row.get("group_key"), Mapping):
            for key, item in row["group_key"].items():
                column = str(key)
                if column not in normalized_row:
                    normalized_row[column] = item
        for column in sorted(row):
            if column not in normalized_row:
                normalized_row[column] = row[column]
        normalized.append(normalized_row)
    return tuple(normalized)


def _columns_for_rows(base_columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    extras = sorted({str(column) for row in rows for column in row if column not in base_columns})
    return tuple(base_columns) + tuple(column for column in extras if column not in base_columns)


def _pattern_provenance_payload(
    prepared: Mapping[str, Any],
    *,
    targets: Mapping[str, _ResolvedOutputPath],
    overwrite: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "prepared_mvpa_pattern_outputs",
        "writer_module": __name__,
        "output_written": True,
        "overwrite": overwrite,
        "overwrite_policy": _overwrite_policy(overwrite),
        "output_paths": _relative_output_paths(targets),
        "columns": {
            "rows": list(prepared["row_columns"]),
            "qc": list(prepared["qc_columns"]),
        },
        "row_counts": {
            "rows": len(prepared["rows"]),
            "qc": len(prepared["qc_rows"]),
            "groups": len(prepared["groups"]),
        },
        "prepared_row_count": len(prepared["rows"]),
        "qc_row_count": len(prepared["qc_rows"]),
        "group_count": len(prepared["groups"]),
        "input_provenance": prepared["input_provenance"],
        "warnings": prepared["warnings"],
        "errors": prepared["errors"],
        "executed": prepared["executed"],
    }


def _distance_provenance_payload(
    prepared: Mapping[str, Any],
    *,
    targets: Mapping[str, _ResolvedOutputPath],
    overwrite: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "prepared_mvpa_distance_outputs",
        "writer_module": __name__,
        "output_written": True,
        "overwrite": overwrite,
        "overwrite_policy": _overwrite_policy(overwrite),
        "output_paths": _relative_output_paths(targets),
        "columns": {
            "distances": list(prepared["distance_columns"]),
            "qc": list(prepared["qc_columns"]),
        },
        "row_counts": {
            "distances": len(prepared["distances"]),
            "qc": len(prepared["qc_rows"]),
        },
        "distance_row_count": len(prepared["distances"]),
        "qc_row_count": len(prepared["qc_rows"]),
        "input_provenance": prepared["input_provenance"],
        "warnings": prepared["warnings"],
        "errors": prepared["errors"],
        "executed": prepared["executed"],
    }


def _summary_provenance_payload(
    prepared: Mapping[str, Any],
    *,
    targets: Mapping[str, _ResolvedOutputPath],
    overwrite: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "prepared_mvpa_distance_summary_outputs",
        "writer_module": __name__,
        "output_written": True,
        "overwrite": overwrite,
        "overwrite_policy": _overwrite_policy(overwrite),
        "output_paths": _relative_output_paths(targets),
        "columns": {
            "summaries": list(prepared["summary_columns"]),
            "qc": list(prepared["qc_columns"]),
        },
        "row_counts": {
            "summaries": len(prepared["summary_rows"]),
            "qc": len(prepared["qc_rows"]),
        },
        "summary_row_count": len(prepared["summary_rows"]),
        "qc_row_count": len(prepared["qc_rows"]),
        "input_provenance": prepared["input_provenance"],
        "warnings": prepared["warnings"],
        "errors": prepared["errors"],
        "executed": prepared["executed"],
    }


def _outputs_record(
    *,
    artifact_kind: str,
    output_root: Path,
    targets: Mapping[str, _ResolvedOutputPath],
    row_counts: Mapping[str, int],
    columns: Mapping[str, Sequence[str]],
    overwrite: bool,
    will_write: bool,
    output_written: bool,
) -> dict[str, Any]:
    return _json_safe_value(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": artifact_kind,
            "writer_module": __name__,
            "output_root": str(output_root),
            "overwrite": overwrite,
            "overwrite_policy": _overwrite_policy(overwrite),
            "will_write": will_write,
            "output_written": output_written,
            "artifacts": [
                _artifact_record(
                    parent_artifact_kind=artifact_kind,
                    name=name,
                    target=target,
                    row_count=row_counts[name],
                    columns=columns[name],
                    overwrite=overwrite,
                    will_write=will_write,
                )
                for name, target in targets.items()
            ],
        }
    )


def _artifact_record(
    *,
    parent_artifact_kind: str,
    name: str,
    target: _ResolvedOutputPath,
    row_count: int,
    columns: Sequence[str],
    overwrite: bool,
    will_write: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "artifact_kind": _child_artifact_kind(parent_artifact_kind, name),
        "path": str(target.path),
        "relative_path": target.relative_path,
        "row_count": row_count,
        "columns": list(columns),
        "exists": target.path.exists(),
        "overwrite": overwrite,
        "overwrite_policy": _overwrite_policy(overwrite),
        "will_write": will_write,
    }


def _child_artifact_kind(parent_artifact_kind: str, name: str) -> str:
    if parent_artifact_kind == "prepared_mvpa_pattern_outputs":
        return {
            "rows": "prepared_mvpa_pattern_rows",
            "qc": "prepared_mvpa_pattern_qc",
            "provenance": "prepared_mvpa_pattern_provenance",
        }[name]
    if parent_artifact_kind == "prepared_mvpa_distance_outputs":
        return {
            "distances": "prepared_mvpa_distance_rows",
            "qc": "prepared_mvpa_distance_qc",
            "provenance": "prepared_mvpa_distance_provenance",
        }[name]
    if parent_artifact_kind == "prepared_mvpa_distance_summary_outputs":
        return {
            "summaries": "prepared_mvpa_distance_summary_rows",
            "qc": "prepared_mvpa_distance_summary_qc",
            "provenance": "prepared_mvpa_distance_summary_provenance",
        }[name]
    if name in {"tsv", "json"}:
        return parent_artifact_kind
    return f"{parent_artifact_kind}_{name}"


def _resolved_output_targets(
    output_root: str | Path,
    paths: Mapping[str, str | Path],
) -> tuple[Path, dict[str, _ResolvedOutputPath]]:
    root = _resolved_output_root(output_root)
    targets = {name: _resolved_output_path(root, path) for name, path in paths.items()}
    _reject_duplicate_targets(targets)
    return root, targets


def _resolved_output_root(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser()
    if _has_parent_traversal(root):
        raise ValueError("output_root must not contain parent traversal.")
    return root.resolve(strict=False)


def _resolved_output_path(root: Path, output_path: str | Path) -> _ResolvedOutputPath:
    raw_path = Path(output_path).expanduser()
    if _has_parent_traversal(raw_path):
        raise ValueError(f"Output path {output_path!s} contains parent traversal.")
    target = raw_path if raw_path.is_absolute() else root / raw_path
    resolved = target.resolve(strict=False)
    if resolved == root:
        raise ValueError("Output path must name a file inside output_root.")
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Output path {output_path!s} resolves outside output_root.") from exc
    return _ResolvedOutputPath(path=resolved, relative_path=relative.as_posix())


def _has_parent_traversal(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def _reject_duplicate_targets(targets: Mapping[str, _ResolvedOutputPath]) -> None:
    seen: dict[Path, str] = {}
    for name, target in targets.items():
        previous = seen.get(target.path)
        if previous is not None:
            raise ValueError(f"Output paths for {previous!r} and {name!r} must be distinct.")
        seen[target.path] = name


def _reject_existing_targets(targets: Mapping[str, _ResolvedOutputPath], *, overwrite: bool) -> None:
    for name, target in targets.items():
        if target.path.is_dir():
            raise IsADirectoryError(f"Output path for {name!r} is an existing directory: {target.path}")
        if target.path.exists() and not overwrite:
            raise FileExistsError(f"Output path for {name!r} already exists: {target.path}")


def _tsv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(tuple(columns))
    for row in rows:
        writer.writerow([_serialize_tsv_cell(row.get(column)) for column in columns])
    return output.getvalue()


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe_value(payload), allow_nan=False, indent=2, sort_keys=True) + "\n"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _serialize_tsv_cell(value: Any) -> str:
    safe_value = _json_safe_value(value)
    if safe_value is None:
        return ""
    if isinstance(safe_value, Mapping) or isinstance(safe_value, list):
        return json.dumps(safe_value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    if isinstance(safe_value, bool):
        return "true" if safe_value else "false"
    if isinstance(safe_value, float):
        if not math.isfinite(safe_value):
            raise ValueError("MVPA runtime TSV cells cannot contain non-finite floats.")
        return repr(safe_value)
    if isinstance(safe_value, int):
        return str(safe_value)
    return str(safe_value)


def _json_safe_value(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _json_safe_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("MVPA runtime outputs cannot contain non-finite floats.")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif is_dataclass(value):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or expose to_dict().")
    return value


def _string_key_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        text_key = str(key)
        if text_key in normalized:
            raise ValueError(f"Mapping contains duplicate key after string conversion: {text_key!r}")
        normalized[text_key] = item
    return normalized


def _as_sequence(value: Any, *, field_name: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a sequence, not a scalar or mapping.")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence.") from exc


def _input_provenance(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_safe_value(value)
    provenance: dict[str, Any] = {}
    for index, raw_row in enumerate(_as_sequence(value, field_name="provenance")):
        row = _string_key_mapping(_as_mapping(raw_row, field_name=f"provenance[{index}]"))
        if row.get("key") is None:
            continue
        provenance[str(row["key"])] = _json_safe_value(row.get("value"))
    return provenance


def _relative_output_paths(targets: Mapping[str, _ResolvedOutputPath]) -> dict[str, str]:
    return {name: target.relative_path for name, target in targets.items()}


def _overwrite_policy(overwrite: bool) -> str:
    return "replace_existing" if overwrite else "fail_if_exists"


__all__ = [
    "plan_prepared_mvpa_distance_outputs",
    "plan_prepared_mvpa_pattern_outputs",
    "plan_prepared_mvpa_summary_outputs",
    "write_mvpa_runtime_json",
    "write_mvpa_runtime_tsv",
    "write_prepared_mvpa_distance_outputs",
    "write_prepared_mvpa_pattern_outputs",
    "write_prepared_mvpa_summary_outputs",
]
