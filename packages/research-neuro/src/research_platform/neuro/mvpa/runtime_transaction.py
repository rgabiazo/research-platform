"""All-or-nothing local MVPA runtime-root transactions.

The transaction owns one complete runtime directory.  Scientific computation
is supplied by the caller and must finish before this module is invoked.  This
module then stages existing serializers on the destination filesystem,
validates the complete inventory, writes a portable success manifest, and
promotes the directory once.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
import ctypes
import csv
import errno
import json
import math
import os
import shutil
import sys
import tempfile

from research_platform.neuro._roi_path_safety import published_value_local_path_fields

from .pattern_sources import REPRESENTATION_IMAGE, REPRESENTATION_PREPARED_FEATURES


SCHEMA_VERSION = "research_platform.neuro.mvpa.runtime_transaction.v1"
MANIFEST_RELATIVE_PATH = "manifest.json"
EXISTING_OUTPUT_FAIL = "fail"


@dataclass(frozen=True)
class MvpaRuntimeOutputSpec:
    """One fixed v1 file in a complete runtime transaction."""

    name: str
    relative_path: str
    content_type: str


_IMAGE_SOURCE_OUTPUTS = (
    MvpaRuntimeOutputSpec("neuro_patterns_tsv", "neuro/pattern-extraction/patterns.tsv", "tsv"),
    MvpaRuntimeOutputSpec("neuro_pattern_qc_tsv", "neuro/pattern-extraction/qc.tsv", "tsv"),
    MvpaRuntimeOutputSpec(
        "neuro_pattern_provenance_json",
        "neuro/pattern-extraction/provenance.json",
        "json",
    ),
    MvpaRuntimeOutputSpec(
        "neuro_pattern_vector_metadata_json",
        "neuro/pattern-extraction/vector_metadata.json",
        "json",
    ),
)
_PREPARED_SOURCE_OUTPUTS = (
    MvpaRuntimeOutputSpec(
        "neuro_materialized_patterns_tsv",
        "neuro/pattern-materialization/patterns.tsv",
        "tsv",
    ),
    MvpaRuntimeOutputSpec(
        "neuro_materialized_pattern_qc_tsv",
        "neuro/pattern-materialization/qc.tsv",
        "tsv",
    ),
    MvpaRuntimeOutputSpec(
        "neuro_materialized_pattern_provenance_json",
        "neuro/pattern-materialization/provenance.json",
        "json",
    ),
    MvpaRuntimeOutputSpec(
        "neuro_materialized_pattern_vector_metadata_json",
        "neuro/pattern-materialization/vector_metadata.json",
        "json",
    ),
)
_ANALYSIS_OUTPUTS = (
    MvpaRuntimeOutputSpec(
        "analysis_prepared_pattern_rows_tsv",
        "analysis/prepared-patterns/rows.tsv",
        "tsv",
    ),
    MvpaRuntimeOutputSpec(
        "analysis_prepared_pattern_qc_tsv",
        "analysis/prepared-patterns/qc.tsv",
        "tsv",
    ),
    MvpaRuntimeOutputSpec(
        "analysis_prepared_pattern_provenance_json",
        "analysis/prepared-patterns/provenance.json",
        "json",
    ),
    MvpaRuntimeOutputSpec(
        "analysis_prepared_distance_rows_tsv",
        "analysis/prepared-distances/distances.tsv",
        "tsv",
    ),
    MvpaRuntimeOutputSpec(
        "analysis_prepared_distance_qc_tsv",
        "analysis/prepared-distances/qc.tsv",
        "tsv",
    ),
    MvpaRuntimeOutputSpec(
        "analysis_prepared_distance_provenance_json",
        "analysis/prepared-distances/provenance.json",
        "json",
    ),
    MvpaRuntimeOutputSpec(
        "analysis_prepared_summary_rows_tsv",
        "analysis/prepared-summaries/summaries.tsv",
        "tsv",
    ),
    MvpaRuntimeOutputSpec(
        "analysis_prepared_summary_qc_tsv",
        "analysis/prepared-summaries/qc.tsv",
        "tsv",
    ),
    MvpaRuntimeOutputSpec(
        "analysis_prepared_summary_provenance_json",
        "analysis/prepared-summaries/provenance.json",
        "json",
    ),
)
_MANIFEST_OUTPUT = MvpaRuntimeOutputSpec(
    "successful_run_manifest_json",
    MANIFEST_RELATIVE_PATH,
    "json",
)
_PORTABLE_PROVENANCE_PATHS = frozenset(
    {
        "neuro/pattern-materialization/provenance.json",
        "analysis/prepared-patterns/provenance.json",
        "analysis/prepared-distances/provenance.json",
        "analysis/prepared-summaries/provenance.json",
    }
)


class MvpaRuntimeTransactionError(RuntimeError):
    """A transaction failed, optionally leaving one recoverable owned path."""

    def __init__(self, message: str, *, recovery_path: str | Path | None = None) -> None:
        super().__init__(message)
        self.recovery_path = Path(recovery_path) if recovery_path is not None else None


@dataclass(frozen=True)
class MvpaRuntimeTransactionPlan:
    """Read-only preflight for one fixed runtime-root transaction."""

    representation_kind: str
    named_root: Path = field(repr=False, compare=False)
    final_root: Path = field(repr=False, compare=False)
    existing_output: str = EXISTING_OUTPUT_FAIL
    outputs: tuple[MvpaRuntimeOutputSpec, ...] = ()
    collision_paths: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "representation_kind": self.representation_kind,
            "existing_output": self.existing_output,
            "final_root": str(self.final_root),
            "outputs": [
                {
                    "name": output.name,
                    "relative_path": output.relative_path,
                    "content_type": output.content_type,
                    "path": str(self.final_root / output.relative_path),
                    "executed": False,
                }
                for output in self.outputs
            ],
            "collision_paths": list(self.collision_paths),
            "errors": list(self.errors),
            "executed": False,
        }


@dataclass(frozen=True)
class MvpaRuntimeTransactionResult:
    """Successful atomic promotion record."""

    final_root: Path
    representation_kind: str
    manifest: Mapping[str, Any]
    output_sha256: Mapping[str, str]
    writer_records: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    recovery_path: Path | None = None
    executed: bool = True


def runtime_output_specs(representation_kind: str) -> tuple[MvpaRuntimeOutputSpec, ...]:
    """Return the exact 13 runtime files plus the top-level manifest."""

    if representation_kind == REPRESENTATION_IMAGE:
        source = _IMAGE_SOURCE_OUTPUTS
    elif representation_kind == REPRESENTATION_PREPARED_FEATURES:
        source = _PREPARED_SOURCE_OUTPUTS
    else:
        raise ValueError(
            f"MVPA representation {representation_kind!r} has no fixed v1 runtime inventory."
        )
    return (*source, *_ANALYSIS_OUTPUTS, _MANIFEST_OUTPUT)


def plan_mvpa_runtime_transaction(
    *,
    named_root: str | Path,
    final_root: str | Path,
    representation_kind: str,
    existing_output: str = EXISTING_OUTPUT_FAIL,
) -> MvpaRuntimeTransactionPlan:
    """Inspect the complete destination set without creating or changing paths."""

    errors: list[str] = []
    collisions: list[str] = []
    try:
        outputs = runtime_output_specs(representation_kind)
    except ValueError as exc:
        outputs = ()
        errors.append(str(exc))
    root = _lexical_absolute(named_root)
    destination = _lexical_absolute(final_root)
    if existing_output != EXISTING_OUTPUT_FAIL:
        errors.append("MVPA runtime existing_output must be 'fail' in v1.")
    atomic_support_error = _atomic_no_replace_support_error()
    if atomic_support_error is not None:
        errors.append(atomic_support_error)
    try:
        relative_root = destination.relative_to(root)
    except ValueError:
        errors.append("MVPA runtime root must remain beneath its configured named root.")
    else:
        if not relative_root.parts:
            errors.append("MVPA runtime root must name a child directory beneath its configured named root.")
    if not _lexists(root):
        errors.append("The configured MVPA runtime named root does not exist.")
    elif root.is_symlink():
        errors.append("The configured MVPA runtime named root must not be a symbolic link.")
    elif not root.is_dir():
        errors.append("The configured MVPA runtime named root is not a directory.")

    errors.extend(_existing_parent_errors(root, destination))
    if _lexists(destination):
        collisions.append("runtime_root")
        errors.append("The final MVPA runtime root already exists; existing_output=fail.")
    target_keys: dict[str, str] = {}
    for output in outputs:
        target = destination / output.relative_path
        try:
            target.relative_to(destination)
        except ValueError:
            errors.append("A planned MVPA runtime output escapes the final runtime root.")
            continue
        alias_key = os.path.normcase(str(target)).casefold()
        previous = target_keys.get(alias_key)
        if previous is not None:
            errors.append(
                f"Planned MVPA runtime outputs {previous!r} and {output.name!r} alias one destination."
            )
        else:
            target_keys[alias_key] = output.name
        if _lexists(target):
            collisions.append(output.name)
    if collisions and "runtime_root" not in collisions:
        errors.append("One or more planned MVPA runtime destinations already exist.")
    nearest = _nearest_existing_parent(destination.parent)
    if nearest is not None and nearest.is_dir() and not os.access(nearest, os.W_OK | os.X_OK):
        errors.append("The nearest existing MVPA runtime destination parent is not writable.")
    return MvpaRuntimeTransactionPlan(
        representation_kind=representation_kind,
        named_root=root,
        final_root=destination,
        existing_output=existing_output,
        outputs=tuple(outputs),
        collision_paths=tuple(collisions),
        errors=tuple(dict.fromkeys(errors)),
    )


def execute_mvpa_runtime_transaction(
    plan: MvpaRuntimeTransactionPlan,
    *,
    write_outputs: Callable[[Path], Mapping[str, Any] | None],
    manifest_payload: Mapping[str, Any],
) -> MvpaRuntimeTransactionResult:
    """Stage, validate, and atomically promote one already-computed run."""

    current = plan_mvpa_runtime_transaction(
        named_root=plan.named_root,
        final_root=plan.final_root,
        representation_kind=plan.representation_kind,
        existing_output=plan.existing_output,
    )
    if not current.valid:
        raise MvpaRuntimeTransactionError("; ".join(current.errors))

    created_parents: list[Path] = []
    staging_root: Path | None = None
    claim_path: Path | None = None
    claim_owned = False
    promotion_identity: tuple[int, int] | None = None
    try:
        created_parents = _create_missing_parents(current.final_root.parent)
        staging_root = Path(
            tempfile.mkdtemp(
                dir=current.final_root.parent,
                prefix=f".{current.final_root.name}.",
                suffix=".tmp",
            )
        )
        raw_records = write_outputs(staging_root)
        writer_records = dict(raw_records or {})
        expectations = _writer_artifact_expectations(writer_records)
        data_specs = tuple(
            output for output in current.outputs if output.relative_path != MANIFEST_RELATIVE_PATH
        )
        output_sha256 = _validate_staged_outputs(
            staging_root,
            expected=data_specs,
            allow_manifest=False,
            expectations=expectations,
        )
        manifest = _successful_manifest(
            manifest_payload,
            representation_kind=current.representation_kind,
            outputs=data_specs,
            output_sha256=output_sha256,
            expectations=expectations,
        )
        _write_manifest(staging_root / MANIFEST_RELATIVE_PATH, manifest)
        final_sha256 = _validate_staged_outputs(
            staging_root,
            expected=current.outputs,
            allow_manifest=True,
            expectations=expectations,
        )
        _validate_manifest_hashes(manifest, final_sha256)

        claim_path = current.final_root.parent / f".{current.final_root.name}.claim"
        try:
            claim_path.mkdir()
        except FileExistsError as exc:
            raise MvpaRuntimeTransactionError(
                "Another MVPA runtime transaction already claims this destination."
            ) from exc
        claim_owned = True
        if _lexists(current.final_root):
            raise MvpaRuntimeTransactionError(
                "The final MVPA runtime root was claimed concurrently before promotion."
            )
        promotion_identity = _path_identity(staging_root)
        if promotion_identity is None:
            raise MvpaRuntimeTransactionError(
                "The owned MVPA runtime staging root disappeared before promotion."
            )
        _promote_staging_tree(staging_root, current.final_root)
        staging_root = None
        try:
            claim_path.rmdir()
            claim_owned = False
            claim_path = None
            claim_warning: tuple[str, ...] = ()
            claim_recovery: Path | None = None
        except OSError:
            claim_warning = (
                "MVPA runtime succeeded, but its transaction claim requires manual cleanup.",
            )
            claim_recovery = claim_path
        return MvpaRuntimeTransactionResult(
            final_root=current.final_root,
            representation_kind=current.representation_kind,
            manifest=manifest,
            output_sha256=final_sha256,
            writer_records=writer_records,
            warnings=claim_warning,
            recovery_path=claim_recovery,
            executed=True,
        )
    except BaseException as exc:
        recovery = _cleanup_transaction_paths(
            staging_root=staging_root,
            claim_path=claim_path if claim_owned else None,
            final_root=current.final_root,
            promotion_identity=promotion_identity,
            created_parents=created_parents,
        )
        if recovery is not None:
            raise MvpaRuntimeTransactionError(
                "MVPA runtime failed and its recoverable transaction path could not be removed.",
                recovery_path=recovery,
            ) from exc
        raise


def _successful_manifest(
    payload: Mapping[str, Any],
    *,
    representation_kind: str,
    outputs: Sequence[MvpaRuntimeOutputSpec],
    output_sha256: Mapping[str, str],
    expectations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = _json_safe_mapping(payload)
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "representation_kind": representation_kind,
            "outputs": [
                {
                    "name": output.name,
                    "relative_path": output.relative_path,
                    "sha256": output_sha256[output.relative_path],
                    "row_count": expectations[output.relative_path]["row_count"],
                }
                for output in outputs
            ],
            "errors": [],
        }
    )
    unsafe = published_value_local_path_fields(manifest, label="successful_run_manifest")
    if unsafe:
        raise ValueError(
            "Successful MVPA runtime manifest contains a non-portable local path reference."
        )
    return manifest


def _validate_staged_outputs(
    staging_root: Path,
    *,
    expected: Sequence[MvpaRuntimeOutputSpec],
    allow_manifest: bool,
    expectations: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    expected_paths = {output.relative_path: output for output in expected}
    observed: dict[str, Path] = {}
    for path in staging_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("MVPA runtime staging must not contain symbolic links.")
        if path.is_file():
            observed[path.relative_to(staging_root).as_posix()] = path
        elif not path.is_dir():
            raise ValueError("MVPA runtime staging contains a special filesystem entry.")
    if set(observed) != set(expected_paths):
        raise ValueError("MVPA runtime staging inventory does not match the fixed v1 output set.")
    digests: dict[str, str] = {}
    for relative_path, spec in expected_paths.items():
        path = observed[relative_path]
        if spec.content_type == "tsv":
            header, row_count = _validate_tsv(path)
            expectation = expectations.get(relative_path)
            if expectation is None:
                raise ValueError(
                    "MVPA runtime writer records do not cover the complete staged TSV inventory."
                )
            expected_columns = tuple(str(value) for value in expectation.get("columns", ()))
            if expected_columns and header != expected_columns:
                raise ValueError("MVPA runtime TSV header does not match its writer record.")
            if row_count != expectation.get("row_count"):
                raise ValueError("MVPA runtime TSV row count does not match its writer record.")
        elif spec.content_type == "json":
            payload = _read_json(path)
            if relative_path == MANIFEST_RELATIVE_PATH and not allow_manifest:
                raise ValueError("MVPA runtime manifest appeared before the commit-marker stage.")
            _validate_finite_json(payload)
            if relative_path != MANIFEST_RELATIVE_PATH:
                expectation = expectations.get(relative_path)
                if expectation is None:
                    raise ValueError(
                        "MVPA runtime writer records do not cover the complete staged JSON inventory."
                    )
                if relative_path.endswith("/provenance.json"):
                    _validate_provenance_relationships(
                        payload,
                        provenance_path=relative_path,
                        expected_paths=set(expected_paths),
                        expectations=expectations,
                    )
                if relative_path in _PORTABLE_PROVENANCE_PATHS:
                    unsafe = published_value_local_path_fields(
                        payload,
                        label="portable_runtime_provenance",
                    )
                    if unsafe:
                        raise ValueError(
                            "Portable MVPA runtime provenance contains a non-portable local path reference."
                        )
        else:
            raise ValueError(f"Unsupported MVPA runtime content type {spec.content_type!r}.")
        digests[relative_path] = _file_sha256(path)
    return digests


def _validate_manifest_hashes(manifest: Mapping[str, Any], hashes: Mapping[str, str]) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("Successful MVPA runtime manifest outputs must contain a list.")
    for output in outputs:
        if not isinstance(output, Mapping):
            raise ValueError("Successful MVPA runtime manifest output records must be mappings.")
        relative_path = str(output.get("relative_path") or "")
        if relative_path == MANIFEST_RELATIVE_PATH or output.get("sha256") != hashes.get(relative_path):
            raise ValueError("Successful MVPA runtime manifest output digest validation failed.")


def _writer_artifact_expectations(
    writer_records: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    expectations: dict[str, Mapping[str, Any]] = {}
    for writer_name, raw_record in writer_records.items():
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"MVPA runtime writer record {writer_name!r} must contain a mapping.")
        artifacts = raw_record.get("artifacts")
        if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes, bytearray)):
            raise ValueError(f"MVPA runtime writer record {writer_name!r} must list its artifacts.")
        for raw_artifact in artifacts:
            if not isinstance(raw_artifact, Mapping):
                raise TypeError("MVPA runtime artifact records must contain mappings.")
            relative_path = str(raw_artifact.get("relative_path") or "").strip()
            name = str(raw_artifact.get("name") or "").strip()
            row_count = raw_artifact.get("row_count")
            columns = raw_artifact.get("columns", ())
            if (
                not relative_path
                or not name
                or isinstance(row_count, bool)
                or not isinstance(row_count, int)
                or row_count < 0
                or not isinstance(columns, Sequence)
                or isinstance(columns, (str, bytes, bytearray))
            ):
                raise ValueError("MVPA runtime artifact record is incomplete or invalid.")
            if relative_path in expectations:
                raise ValueError("MVPA runtime writer records contain a duplicate output path.")
            expectations[relative_path] = {
                "writer": str(writer_name),
                "name": name,
                "row_count": row_count,
                "columns": tuple(str(column) for column in columns),
            }
    return expectations


def _validate_provenance_relationships(
    payload: Any,
    *,
    provenance_path: str,
    expected_paths: set[str],
    expectations: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("MVPA runtime provenance output must contain a mapping.")
    output_paths = payload.get("output_paths")
    row_counts = payload.get("row_counts")
    if not isinstance(output_paths, Mapping) or not isinstance(row_counts, Mapping):
        raise ValueError("MVPA runtime provenance must declare output paths and row counts.")
    provenance_expectation = expectations.get(provenance_path)
    if provenance_expectation is None:
        raise ValueError("MVPA runtime provenance has no matching writer artifact record.")
    writer_name = str(provenance_expectation.get("writer"))
    writer_artifacts = {
        relative_path: record
        for relative_path, record in expectations.items()
        if str(record.get("writer")) == writer_name
    }
    artifacts_by_name: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for relative_path, record in writer_artifacts.items():
        name = str(record.get("name"))
        if name in artifacts_by_name:
            raise ValueError("MVPA runtime writer artifact names must be unique within one writer.")
        artifacts_by_name[name] = (relative_path, record)

    declared_paths = {str(name): str(path) for name, path in output_paths.items()}
    expected_path_names = set(artifacts_by_name)
    if set(declared_paths) != expected_path_names:
        raise ValueError(
            "MVPA runtime provenance must declare the complete writer artifact set."
        )
    for name, relative_path in declared_paths.items():
        expected_relative_path, _record = artifacts_by_name[name]
        if relative_path != expected_relative_path or relative_path not in expected_paths:
            raise ValueError("MVPA runtime provenance output-path relationship is invalid.")

    required_count_names = {
        name
        for name, (relative_path, _record) in artifacts_by_name.items()
        if relative_path != provenance_path
    }
    declared_counts = {str(name): value for name, value in row_counts.items()}
    declared_count_names = set(declared_counts)
    if not required_count_names.issubset(declared_count_names):
        raise ValueError(
            "MVPA runtime provenance must declare row counts for every writer data artifact."
        )
    for name in required_count_names:
        _relative_path, expected = artifacts_by_name[name]
        if declared_counts[name] != expected["row_count"]:
            raise ValueError("MVPA runtime provenance row count does not match its artifact.")


def _validate_tsv(path: Path) -> tuple[tuple[str, ...], int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("MVPA runtime TSV output must contain a header.") from exc
        if not header or any(not column for column in header) or len(set(header)) != len(header):
            raise ValueError("MVPA runtime TSV output has an invalid header.")
        resolved_header = tuple(header)
        width = len(resolved_header)
        row_count = 0
        for row in reader:
            if len(row) != width:
                raise ValueError("MVPA runtime TSV output row width does not match its header.")
            for value in row:
                _validate_finite_text(value)
            row_count += 1
    return resolved_header, row_count


def _validate_finite_text(value: str) -> None:
    stripped = value.strip()
    if not stripped:
        return
    if stripped.casefold() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
        raise ValueError("MVPA runtime TSV output contains a non-finite value.")
    if stripped.startswith(("[", "{")):
        try:
            parsed = json.loads(stripped, parse_constant=_reject_json_constant)
        except json.JSONDecodeError:
            return
        _validate_finite_json(parsed)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_json_constant)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"MVPA runtime JSON contains forbidden numeric constant {value!r}.")


def _validate_finite_json(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_json(str(key))
            _validate_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite_json(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("MVPA runtime JSON contains a non-finite value.")


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
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
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _promote_staging_tree(staging_root: Path, final_root: Path) -> None:
    """Atomically rename one sibling tree without replacing any destination."""

    if staging_root.parent != final_root.parent:
        raise MvpaRuntimeTransactionError(
            "MVPA runtime staging and final roots must share one filesystem parent."
        )
    if os.name == "nt":
        os.rename(staging_root, final_root)
        return

    parent_fd = os.open(staging_root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        library = ctypes.CDLL(None, use_errno=True)
        old_name = os.fsencode(staging_root.name)
        new_name = os.fsencode(final_root.name)
        if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
            promote = library.renameatx_np
            promote.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            promote.restype = ctypes.c_int
            result = promote(parent_fd, old_name, parent_fd, new_name, 0x00000004)
        elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
            promote = library.renameat2
            promote.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            promote.restype = ctypes.c_int
            result = promote(parent_fd, old_name, parent_fd, new_name, 0x00000001)
        else:
            raise MvpaRuntimeTransactionError(
                "This platform does not expose an atomic no-replace directory promotion primitive."
            )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                raise MvpaRuntimeTransactionError(
                    "The final MVPA runtime root was claimed concurrently before promotion."
                )
            raise OSError(
                error_number,
                os.strerror(error_number),
                os.fspath(final_root),
            )
    finally:
        os.close(parent_fd)


def _atomic_no_replace_support_error() -> str | None:
    """Return a preflight error when atomic no-replace promotion is unavailable."""

    if os.name == "nt":
        return None
    try:
        library = ctypes.CDLL(None, use_errno=True)
    except OSError:
        return (
            "This platform cannot load the atomic no-replace directory promotion primitive."
        )
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        return None
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        return None
    return "This platform does not expose an atomic no-replace directory promotion primitive."


def _existing_parent_errors(root: Path, destination: Path) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        relative = destination.relative_to(root)
    except ValueError:
        return ()
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if not _lexists(current):
            continue
        if current.is_symlink():
            errors.append("An MVPA runtime destination parent is a symbolic link.")
        elif not current.is_dir():
            errors.append("An MVPA runtime destination parent is not a directory.")
    return tuple(errors)


def _create_missing_parents(parent: Path) -> list[Path]:
    missing: list[Path] = []
    current = parent
    while not _lexists(current):
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise MvpaRuntimeTransactionError("MVPA runtime destination parent became unsafe.")
    created: list[Path] = []
    try:
        for path in reversed(missing):
            try:
                path.mkdir()
                created.append(path)
            except FileExistsError:
                if path.is_symlink() or not path.is_dir():
                    raise MvpaRuntimeTransactionError(
                        "MVPA runtime destination parent was claimed by an unsafe filesystem entry."
                    )
    except BaseException as exc:
        recovery = _remove_created_parents(created)
        if recovery is not None:
            raise MvpaRuntimeTransactionError(
                "MVPA runtime parent creation failed and an owned path requires recovery.",
                recovery_path=recovery,
            ) from exc
        raise
    return created


def _cleanup_transaction_paths(
    *,
    staging_root: Path | None,
    claim_path: Path | None,
    final_root: Path,
    promotion_identity: tuple[int, int] | None,
    created_parents: Sequence[Path],
) -> Path | None:
    recovery: Path | None = None
    if (
        promotion_identity is not None
        and _path_identity(final_root) == promotion_identity
    ):
        try:
            shutil.rmtree(final_root)
        except OSError:
            recovery = final_root
    if staging_root is not None and _lexists(staging_root):
        try:
            shutil.rmtree(staging_root)
        except OSError:
            recovery = staging_root
    if claim_path is not None and _lexists(claim_path):
        try:
            claim_path.rmdir()
        except OSError:
            recovery = recovery or claim_path
    return recovery or _remove_created_parents(created_parents)


def _remove_created_parents(created_parents: Sequence[Path]) -> Path | None:
    for path in reversed(tuple(created_parents)):
        try:
            path.rmdir()
        except FileNotFoundError:
            continue
        except OSError:
            return path
    return None


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while current != current.parent and not _lexists(current):
        current = current.parent
    return current if _lexists(current) else None


def _lexical_absolute(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(value)))


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _path_identity(path: Path) -> tuple[int, int] | None:
    try:
        stat_result = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return stat_result.st_dev, stat_result.st_ino


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(item) for key, item in value.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Successful MVPA runtime manifest values must be finite.")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError("Successful MVPA runtime manifest values must be JSON-safe and portable.")


__all__ = [
    "EXISTING_OUTPUT_FAIL",
    "MANIFEST_RELATIVE_PATH",
    "MvpaRuntimeOutputSpec",
    "MvpaRuntimeTransactionError",
    "MvpaRuntimeTransactionPlan",
    "MvpaRuntimeTransactionResult",
    "SCHEMA_VERSION",
    "execute_mvpa_runtime_transaction",
    "plan_mvpa_runtime_transaction",
    "runtime_output_specs",
]
