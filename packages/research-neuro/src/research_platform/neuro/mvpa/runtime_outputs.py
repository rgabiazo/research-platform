"""Stdlib-only runtime/cache writers for MVPA pattern extraction outputs."""

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

from .._version import DISTRIBUTION_NAME as WRITER_PACKAGE
from .._version import package_version

SCHEMA_VERSION = "research_platform.neuro.mvpa.runtime_outputs.v1"

_PATTERN_COLUMNS = (
    "pattern_id",
    "condition_id",
    "cv_unit",
    "subject_id",
    "session_id",
    "run_id",
    "task_id",
    "direction",
    "model",
    "pattern_source_name",
    "roi_source_name",
    "roi_label",
    "pe_image",
    "mask_path",
    "noise_image",
    "voxel_count",
    "valid_voxel_count",
    "feature_count",
    "voxel_order",
    "voxel_index_hash",
    "usable",
    "feature_values",
    "event_count",
    "mean_centering_applied",
    "mean_centering_scope",
    "grouping_values",
    "noise_loaded",
    "noise_status",
    "noise_usable",
    "noise_feature_count",
    "noise_voxel_order",
    "noise_voxel_index_hash",
    "noise_min",
    "noise_max",
    "noise_mean",
    "noise_nonfinite_count",
    "noise_nonpositive_count",
    "noise_values",
)
_QC_COLUMNS = (
    "subject_id",
    "session_id",
    "run_id",
    "condition_id",
    "roi_label",
    "pattern_source_name",
    "roi_source_name",
    "pe_image",
    "mask_path",
    "noise_image",
    "status",
    "usable",
    "reason",
    "excluded",
    "exclusion_reason",
    "exclusion_id",
    "exclusion_source_field",
    "skipped_stage",
    "pe_exists",
    "mask_exists",
    "noise_exists",
    "geometry_status",
    "mask_status",
    "voxel_count",
    "valid_voxel_count",
    "warnings",
    "errors",
    "event_threshold_status",
    "grouping_values",
    "noise_loaded",
    "noise_status",
    "noise_usable",
    "noise_feature_count",
    "noise_voxel_order",
    "noise_voxel_index_hash",
    "noise_min",
    "noise_max",
    "noise_mean",
    "noise_nonfinite_count",
    "noise_nonpositive_count",
)
_JSON_ARRAY_TSV_COLUMNS = frozenset({"feature_values", "noise_values", "warnings", "errors"})
_VECTOR_VALUE_COLUMNS = frozenset({"feature_values", "noise_values"})
_MATERIALIZED_PATTERN_COLUMNS = (
    "pattern_id",
    "unit_id",
    "condition_id",
    "cross_validation_label",
    "subject_id",
    "session_id",
    "task_id",
    "run_id",
    "pattern_source_name",
    "roi_source_name",
    "roi_label",
    "feature_count",
    "voxel_order",
    "voxel_index_hash",
    "feature_space_id",
    "roi_definition_id",
    "usable",
    "status",
    "feature_values",
    "event_count",
    "mean_centering_applied",
    "mean_centering_scope",
    "grouping_values",
    "qc_status",
    "qc_reason",
    "exclusion_id",
    "exclusion_reason",
    "roi_reference",
    "noise_status",
    "noise_usable",
    "noise_feature_count",
    "noise_voxel_order",
    "noise_voxel_index_hash",
    "noise_feature_space_id",
    "noise_roi_definition_id",
    "noise_value_kind",
    "noise_estimation_scope",
    "noise_source",
    "noise_values",
)
_MATERIALIZED_QC_COLUMNS = (
    "pattern_id",
    "data_row",
    "status",
    "code",
    "message",
    "qc_status",
    "qc_reason",
    "exclusion_id",
    "exclusion_reason",
)


@dataclass(frozen=True)
class _ResolvedOutputPath:
    path: Path
    relative_path: str


def plan_mvpa_pattern_extraction_outputs(
    result: Any,
    *,
    output_root: str | Path,
    patterns_path: str | Path,
    qc_path: str | Path,
    provenance_path: str | Path,
    vector_metadata_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Return a JSON-safe dry-run record for MVPA pattern extraction outputs."""

    prepared = _pattern_extraction_output_payload(result)
    root, targets = _resolved_pattern_extraction_targets(
        output_root=output_root,
        patterns_path=patterns_path,
        qc_path=qc_path,
        provenance_path=provenance_path,
        vector_metadata_path=vector_metadata_path,
    )
    return _outputs_record(
        artifact_kind="mvpa_pattern_extraction_outputs",
        output_root=root,
        targets=targets,
        row_counts=_pattern_extraction_row_counts(prepared, targets=targets),
        columns=_pattern_extraction_columns(prepared, targets=targets),
        overwrite=overwrite,
        will_write=False,
        output_written=False,
    )


def write_mvpa_pattern_extraction_outputs(
    result: Any,
    *,
    output_root: str | Path,
    patterns_path: str | Path,
    qc_path: str | Path,
    provenance_path: str | Path,
    vector_metadata_path: str | Path | None = None,
    overwrite: bool = False,
    append: bool = False,
) -> dict[str, Any]:
    """Write MVPA pattern rows, QC rows, and provenance runtime outputs."""

    _reject_append(append)
    prepared = _pattern_extraction_output_payload(result)
    root, targets = _resolved_pattern_extraction_targets(
        output_root=output_root,
        patterns_path=patterns_path,
        qc_path=qc_path,
        provenance_path=provenance_path,
        vector_metadata_path=vector_metadata_path,
    )
    _reject_existing_targets(targets, overwrite=overwrite)
    provenance = _pattern_extraction_provenance_payload(
        prepared,
        targets=targets,
        overwrite=overwrite,
    )

    texts = {
        "patterns": _tsv_text(prepared["pattern_rows"], prepared["pattern_columns"]),
        "qc": _tsv_text(prepared["qc_rows"], prepared["qc_columns"]),
        "provenance": _json_text(provenance),
    }
    if "vector_metadata" in targets:
        texts["vector_metadata"] = _json_text(
            _vector_metadata_payload(prepared, target=targets["vector_metadata"]),
            compact=True,
        )

    for name, text in texts.items():
        _write_text_atomic(targets[name].path, text)

    return _outputs_record(
        artifact_kind="mvpa_pattern_extraction_outputs",
        output_root=root,
        targets=targets,
        row_counts=_pattern_extraction_row_counts(prepared, targets=targets),
        columns=_pattern_extraction_columns(prepared, targets=targets),
        overwrite=overwrite,
        will_write=True,
        output_written=True,
    )


def plan_mvpa_pattern_materialization_outputs(
    result: Any,
    *,
    output_root: str | Path,
    patterns_path: str | Path,
    qc_path: str | Path,
    provenance_path: str | Path,
    vector_metadata_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Return a no-write record for prepared-feature materialization outputs."""

    prepared = _pattern_materialization_output_payload(result)
    root, targets = _resolved_pattern_extraction_targets(
        output_root=output_root,
        patterns_path=patterns_path,
        qc_path=qc_path,
        provenance_path=provenance_path,
        vector_metadata_path=vector_metadata_path,
    )
    return _outputs_record(
        artifact_kind="mvpa_pattern_materialization_outputs",
        output_root=root,
        targets=targets,
        row_counts=_pattern_extraction_row_counts(prepared, targets=targets),
        columns=_pattern_extraction_columns(prepared, targets=targets),
        overwrite=overwrite,
        will_write=False,
        output_written=False,
    )


def write_mvpa_pattern_materialization_outputs(
    result: Any,
    *,
    output_root: str | Path,
    patterns_path: str | Path,
    qc_path: str | Path,
    provenance_path: str | Path,
    vector_metadata_path: str | Path | None = None,
    overwrite: bool = False,
    append: bool = False,
) -> dict[str, Any]:
    """Write prepared-feature source audit records without image terminology."""

    _reject_append(append)
    prepared = _pattern_materialization_output_payload(result)
    root, targets = _resolved_pattern_extraction_targets(
        output_root=output_root,
        patterns_path=patterns_path,
        qc_path=qc_path,
        provenance_path=provenance_path,
        vector_metadata_path=vector_metadata_path,
    )
    _reject_existing_targets(targets, overwrite=overwrite)
    provenance = _pattern_extraction_provenance_payload(
        prepared,
        targets=targets,
        overwrite=overwrite,
        artifact_kind="mvpa_pattern_materialization_outputs",
    )
    texts = {
        "patterns": _tsv_text(prepared["pattern_rows"], prepared["pattern_columns"]),
        "qc": _tsv_text(prepared["qc_rows"], prepared["qc_columns"]),
        "provenance": _json_text(provenance),
    }
    if "vector_metadata" in targets:
        texts["vector_metadata"] = _json_text(
            _vector_metadata_payload(
                prepared,
                target=targets["vector_metadata"],
                artifact_kind="mvpa_pattern_materialization_vector_metadata",
            ),
            compact=True,
        )
    for name, text in texts.items():
        _write_text_atomic(targets[name].path, text)
    return _outputs_record(
        artifact_kind="mvpa_pattern_materialization_outputs",
        output_root=root,
        targets=targets,
        row_counts=_pattern_extraction_row_counts(prepared, targets=targets),
        columns=_pattern_extraction_columns(prepared, targets=targets),
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
    append: bool = False,
    artifact_kind: str = "mvpa_runtime_tsv",
) -> dict[str, Any]:
    """Write a generic MVPA runtime TSV with the same safety rules."""

    _reject_append(append)
    row_maps = tuple(
        _string_key_mapping(_as_mapping(row, field_name=f"rows[{index}]"))
        for index, row in enumerate(_as_sequence(rows, field_name="rows"))
    )
    resolved_columns = _resolved_columns(columns)
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
    append: bool = False,
    artifact_kind: str = "mvpa_runtime_json",
) -> dict[str, Any]:
    """Write a generic MVPA runtime JSON file with the same safety rules."""

    _reject_append(append)
    root, targets = _resolved_output_targets(output_root, {"json": output_path})
    _reject_existing_targets(targets, overwrite=overwrite)
    text = _json_text(_as_mapping(payload, field_name="payload"))
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


def _pattern_extraction_output_payload(result: Any) -> dict[str, Any]:
    payload = _string_key_mapping(_as_mapping(result, field_name="result"))
    pattern_rows = _known_rows(
        payload.get("pattern_rows", ()),
        base_columns=_PATTERN_COLUMNS,
        field_name="pattern_rows",
    )
    qc_rows = _known_rows(
        payload.get("qc_rows", ()),
        base_columns=_QC_COLUMNS,
        field_name="qc_rows",
    )
    pattern_columns = _columns_for_rows(_PATTERN_COLUMNS, pattern_rows)
    vector_metadata_columns = tuple(column for column in pattern_columns if column not in _VECTOR_VALUE_COLUMNS)
    return {
        "pattern_rows": pattern_rows,
        "qc_rows": qc_rows,
        "pattern_columns": pattern_columns,
        "qc_columns": _columns_for_rows(_QC_COLUMNS, qc_rows),
        "vector_metadata_rows": _vector_metadata_rows(pattern_rows, columns=vector_metadata_columns),
        "vector_metadata_columns": vector_metadata_columns,
        "input_provenance": _input_provenance(payload.get("provenance", {})),
        "warnings": _json_array_value(payload.get("warnings", ()), field_name="warnings"),
        "errors": _json_array_value(payload.get("errors", ()), field_name="errors"),
        "executed": _json_safe_value(payload.get("executed")),
    }


def _pattern_materialization_output_payload(result: Any) -> dict[str, Any]:
    payload = _string_key_mapping(_as_mapping(result, field_name="result"))
    pattern_rows = _known_rows(
        payload.get("pattern_rows", ()),
        base_columns=_MATERIALIZED_PATTERN_COLUMNS,
        field_name="pattern_rows",
    )
    qc_rows = _known_rows(
        payload.get("qc_rows", ()),
        base_columns=_MATERIALIZED_QC_COLUMNS,
        field_name="qc_rows",
    )
    pattern_columns = _columns_for_rows(_MATERIALIZED_PATTERN_COLUMNS, pattern_rows)
    vector_metadata_columns = tuple(
        column for column in pattern_columns if column not in _VECTOR_VALUE_COLUMNS
    )
    return {
        "pattern_rows": pattern_rows,
        "qc_rows": qc_rows,
        "pattern_columns": pattern_columns,
        "qc_columns": _columns_for_rows(_MATERIALIZED_QC_COLUMNS, qc_rows),
        "vector_metadata_rows": _vector_metadata_rows(
            pattern_rows,
            columns=vector_metadata_columns,
        ),
        "vector_metadata_columns": vector_metadata_columns,
        "input_provenance": _input_provenance(payload.get("provenance", {})),
        "warnings": _json_array_value(payload.get("warnings", ()), field_name="warnings"),
        "errors": _json_array_value(payload.get("errors", ()), field_name="errors"),
        "executed": _json_safe_value(payload.get("executed")),
    }


def _known_rows(value: Any, *, base_columns: Sequence[str], field_name: str) -> tuple[dict[str, Any], ...]:
    rows = _as_sequence(value, field_name=field_name)
    normalized: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        row = _string_key_mapping(_as_mapping(raw_row, field_name=f"{field_name}[{index}]"))
        normalized_row = {
            column: _row_cell_value(row.get(column), column=column, field_name=field_name)
            for column in base_columns
        }
        for column in sorted(str(key) for key in row):
            if column not in normalized_row:
                normalized_row[column] = _row_cell_value(row[column], column=column, field_name=field_name)
        normalized.append(normalized_row)
    return tuple(normalized)


def _row_cell_value(value: Any, *, column: str, field_name: str) -> Any:
    if column in _JSON_ARRAY_TSV_COLUMNS and value is not None:
        return _json_array_value(value, field_name=f"{field_name}.{column}")
    return value


def _columns_for_rows(base_columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    extras = sorted({str(column) for row in rows for column in row if column not in base_columns})
    return tuple(base_columns) + tuple(column for column in extras if column not in base_columns)


def _vector_metadata_rows(
    pattern_rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    return tuple({column: row.get(column) for column in columns} for row in pattern_rows)


def _pattern_extraction_provenance_payload(
    prepared: Mapping[str, Any],
    *,
    targets: Mapping[str, _ResolvedOutputPath],
    overwrite: bool,
    artifact_kind: str = "mvpa_pattern_extraction_outputs",
) -> dict[str, Any]:
    row_counts = {
        "patterns": len(prepared["pattern_rows"]),
        "qc": len(prepared["qc_rows"]),
    }
    if "vector_metadata" in targets:
        row_counts["vector_metadata"] = len(prepared["vector_metadata_rows"])
    columns = {
        "patterns": list(prepared["pattern_columns"]),
        "qc": list(prepared["qc_columns"]),
    }
    if "vector_metadata" in targets:
        columns["vector_metadata"] = list(prepared["vector_metadata_columns"])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        **_writer_metadata(),
        "output_written": True,
        "overwrite": overwrite,
        "overwrite_policy": _overwrite_policy(overwrite),
        "output_paths": _relative_output_paths(targets),
        "columns": columns,
        "row_counts": row_counts,
        "pattern_row_count": len(prepared["pattern_rows"]),
        "qc_row_count": len(prepared["qc_rows"]),
        "input_provenance": prepared["input_provenance"],
        "warnings": prepared["warnings"],
        "errors": prepared["errors"],
        "executed": prepared["executed"],
    }


def _vector_metadata_payload(
    prepared: Mapping[str, Any],
    *,
    target: _ResolvedOutputPath,
    artifact_kind: str = "mvpa_pattern_extraction_vector_metadata",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        **_writer_metadata(),
        "output_written": True,
        "output_path": target.relative_path,
        "columns": list(prepared["vector_metadata_columns"]),
        "row_count": len(prepared["vector_metadata_rows"]),
        "vectors": prepared["vector_metadata_rows"],
    }


def _pattern_extraction_row_counts(
    prepared: Mapping[str, Any],
    *,
    targets: Mapping[str, _ResolvedOutputPath],
) -> dict[str, int]:
    row_counts = {
        "patterns": len(prepared["pattern_rows"]),
        "qc": len(prepared["qc_rows"]),
        "provenance": 1,
    }
    if "vector_metadata" in targets:
        row_counts["vector_metadata"] = len(prepared["vector_metadata_rows"])
    return row_counts


def _pattern_extraction_columns(
    prepared: Mapping[str, Any],
    *,
    targets: Mapping[str, _ResolvedOutputPath],
) -> dict[str, Sequence[str]]:
    columns: dict[str, Sequence[str]] = {
        "patterns": prepared["pattern_columns"],
        "qc": prepared["qc_columns"],
        "provenance": (),
    }
    if "vector_metadata" in targets:
        columns["vector_metadata"] = prepared["vector_metadata_columns"]
    return columns


def _resolved_pattern_extraction_targets(
    *,
    output_root: str | Path,
    patterns_path: str | Path,
    qc_path: str | Path,
    provenance_path: str | Path,
    vector_metadata_path: str | Path | None,
) -> tuple[Path, dict[str, _ResolvedOutputPath]]:
    paths: dict[str, str | Path] = {
        "patterns": patterns_path,
        "qc": qc_path,
        "provenance": provenance_path,
    }
    if vector_metadata_path is not None:
        paths["vector_metadata"] = vector_metadata_path
    return _resolved_output_targets(output_root, paths)


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
            **_writer_metadata(),
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
    if parent_artifact_kind in {
        "mvpa_pattern_extraction_outputs",
        "mvpa_pattern_materialization_outputs",
    }:
        prefix = (
            "mvpa_pattern_extraction"
            if parent_artifact_kind == "mvpa_pattern_extraction_outputs"
            else "mvpa_pattern_materialization"
        )
        return {
            "patterns": f"{prefix}_patterns",
            "qc": f"{prefix}_qc",
            "provenance": f"{prefix}_provenance",
            "vector_metadata": f"{prefix}_vector_metadata",
        }[name]
    if name in {"tsv", "json"}:
        return parent_artifact_kind
    return f"{parent_artifact_kind}_{name}"


def _resolved_output_targets(
    output_root: str | Path,
    paths: Mapping[str, str | Path],
) -> tuple[Path, dict[str, _ResolvedOutputPath]]:
    root = _resolved_output_root(output_root)
    targets = {name: _resolved_output_path(root, path, label=name) for name, path in paths.items()}
    _reject_duplicate_targets(targets)
    return root, targets


def _resolved_output_root(output_root: str | Path) -> Path:
    if output_root is None:
        raise ValueError("output_root is required.")
    root = Path(output_root).expanduser()
    if _has_parent_traversal(root):
        raise ValueError("output_root must not contain parent traversal.")
    resolved = root.resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(f"output_root is not a directory: {resolved}")
    return resolved


def _resolved_output_path(root: Path, output_path: str | Path, *, label: str) -> _ResolvedOutputPath:
    if output_path is None:
        raise ValueError(f"Output path for {label!r} is required.")
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
        for parent in target.path.parents:
            if parent.exists() and not parent.is_dir():
                raise NotADirectoryError(f"Output parent for {name!r} is not a directory: {parent}")


def _reject_append(append: bool) -> None:
    if append:
        raise ValueError("Append mode is not supported for MVPA runtime outputs.")


def _tsv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(tuple(columns))
    for row in rows:
        writer.writerow([_serialize_tsv_cell(row.get(column)) for column in columns])
    return output.getvalue()


def _json_text(payload: Mapping[str, Any], *, compact: bool = False) -> str:
    safe_payload = _json_safe_value(payload)
    if compact:
        return json.dumps(safe_payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    return json.dumps(safe_payload, allow_nan=False, indent=2, sort_keys=True) + "\n"


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
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_safe_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (tuple, list)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("MVPA runtime outputs cannot contain non-finite floats.")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif is_dataclass(value) and not isinstance(value, type):
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
    if isinstance(value, (str, bytes, bytearray)) or isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a sequence, not a scalar or mapping.")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence.") from exc


def _json_array_value(value: Any, *, field_name: str) -> list[Any]:
    return _json_safe_value(_as_sequence(value, field_name=field_name))


def _resolved_columns(columns: Sequence[str]) -> tuple[str, ...]:
    raw_columns = _as_sequence(columns, field_name="columns")
    resolved: list[str] = []
    seen: set[str] = set()
    for raw_column in raw_columns:
        column = str(raw_column)
        if column in seen:
            raise ValueError(f"Column {column!r} is duplicated.")
        resolved.append(column)
        seen.add(column)
    return tuple(resolved)


def _input_provenance(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
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


def _writer_metadata() -> dict[str, str]:
    return {
        "writer_module": __name__,
        "writer_package": WRITER_PACKAGE,
        "writer_version": package_version(),
    }


__all__ = [
    "SCHEMA_VERSION",
    "plan_mvpa_pattern_extraction_outputs",
    "plan_mvpa_pattern_materialization_outputs",
    "write_mvpa_pattern_extraction_outputs",
    "write_mvpa_pattern_materialization_outputs",
    "write_mvpa_runtime_json",
    "write_mvpa_runtime_tsv",
]
