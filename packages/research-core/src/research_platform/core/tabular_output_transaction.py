"""All-or-nothing local tabular output-directory transactions.

The caller owns scientific computation and the run-level execution claim.  This
module owns only one hidden staging directory, validates its fixed inventory,
writes a portable integrity manifest, flushes the tree, and promotes it to the
absent ``outputs/`` directory with an atomic no-replace rename.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
import csv
import ctypes
import errno
import io
import json
import math
import os
import shutil
import stat
import sys
import tempfile

from .run_lifecycle import RunLifecycleError, validate_run_id


SCHEMA_VERSION = "research_platform.core.tabular_output_transaction.v1"
TRANSACTION_MANIFEST_NAME = "transaction-manifest.json"
FINAL_OUTPUT_DIRECTORY = "outputs"
EXISTING_OUTPUT_FAIL = "fail"
STAGING_PREFIX = ".outputs."
STAGING_SUFFIX = ".staging"
_CONTENT_TYPES = frozenset({"json", "tsv"})
_DIGEST_RE = frozenset("0123456789abcdef")


class TabularOutputTransactionError(RuntimeError):
    """A local output transaction failed, possibly with recovery evidence."""

    def __init__(
        self,
        message: str,
        *,
        recovery_path: str | Path | None = None,
        promotion_committed: bool = False,
    ) -> None:
        super().__init__(message)
        self.recovery_path = Path(recovery_path) if recovery_path is not None else None
        self.promotion_committed = promotion_committed


@dataclass(frozen=True)
class OutputSpec:
    """One required scientific file in the fixed output inventory."""

    logical_name: str
    relative_path: str
    content_type: str
    expected_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputRecord:
    """Integrity and structural metadata for one validated output file."""

    logical_name: str
    relative_path: str
    content_type: str
    byte_size: int
    sha256: str
    row_count: int | None = None
    columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the portable transaction-manifest record."""

        record: dict[str, Any] = {
            "logical_name": self.logical_name,
            "relative_path": self.relative_path,
            "content_type": self.content_type,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }
        if self.content_type == "tsv":
            record["row_count"] = self.row_count
            record["columns"] = list(self.columns)
        return record


@dataclass(frozen=True)
class OwnedStaging:
    """One exclusively created staging directory and its filesystem identity."""

    path: Path
    device: int
    inode: int

    @property
    def filesystem_identity(self) -> tuple[int, int]:
        """Return the device/inode pair recorded at creation."""

        return self.device, self.inode


def build_transaction_plan(
    *,
    run_id: str,
    workflow_action: str,
    workflow_target: str,
    outputs: Sequence[OutputSpec],
) -> dict[str, Any]:
    """Return the stable, portable plan bound into reviewed plan identity."""

    specs = _validated_specs(outputs)
    try:
        validate_run_id(run_id)
    except RunLifecycleError as exc:
        raise TabularOutputTransactionError(str(exc)) from exc
    for label, value in {
        "workflow action": workflow_action,
        "workflow target": workflow_target,
    }.items():
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise TabularOutputTransactionError(f"Transaction {label} must be a nonempty string.")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "workflow": {
            "action": workflow_action,
            "target": workflow_target,
        },
        "final_output_directory": FINAL_OUTPUT_DIRECTORY,
        "outputs": [
            {
                "logical_name": spec.logical_name,
                "relative_path": spec.relative_path,
                "content_type": spec.content_type,
            }
            for spec in specs
        ],
        "transaction_manifest": (
            f"{FINAL_OUTPUT_DIRECTORY}/{TRANSACTION_MANIFEST_NAME}"
        ),
        "existing_output": EXISTING_OUTPUT_FAIL,
    }


def output_specs_from_plan(plan: Mapping[str, Any]) -> tuple[OutputSpec, ...]:
    """Parse and validate the fixed output specification stored in a plan."""

    if set(plan) != {
        "schema_version",
        "run_id",
        "workflow",
        "final_output_directory",
        "outputs",
        "transaction_manifest",
        "existing_output",
    }:
        raise TabularOutputTransactionError(
            "Tabular output transaction plan has an invalid structure."
        )
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise TabularOutputTransactionError(
            "Tabular output transaction plan uses an unsupported schema version."
        )
    if plan.get("final_output_directory") != FINAL_OUTPUT_DIRECTORY:
        raise TabularOutputTransactionError(
            "Tabular output transaction plan has an invalid final output directory."
        )
    if plan.get("transaction_manifest") != (
        f"{FINAL_OUTPUT_DIRECTORY}/{TRANSACTION_MANIFEST_NAME}"
    ):
        raise TabularOutputTransactionError(
            "Tabular output transaction plan has an invalid manifest path."
        )
    if plan.get("existing_output") != EXISTING_OUTPUT_FAIL:
        raise TabularOutputTransactionError(
            "Tabular output transaction existing_output must be 'fail'."
        )
    run_id = plan.get("run_id")
    try:
        validate_run_id(run_id)  # type: ignore[arg-type]
    except RunLifecycleError as exc:
        raise TabularOutputTransactionError(str(exc)) from exc
    workflow = plan.get("workflow")
    if not isinstance(workflow, Mapping) or set(workflow) != {"action", "target"}:
        raise TabularOutputTransactionError(
            "Tabular output transaction plan has an invalid workflow identity."
        )
    for value in workflow.values():
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise TabularOutputTransactionError(
                "Tabular output transaction workflow values must be nonempty strings."
            )
    raw_outputs = plan.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise TabularOutputTransactionError(
            "Tabular output transaction plan must list its scientific outputs."
        )
    specs: list[OutputSpec] = []
    for raw in raw_outputs:
        if not isinstance(raw, Mapping) or set(raw) != {
            "logical_name",
            "relative_path",
            "content_type",
        }:
            raise TabularOutputTransactionError(
                "Tabular output transaction plan contains an invalid output record."
            )
        specs.append(
            OutputSpec(
                logical_name=str(raw["logical_name"]),
                relative_path=str(raw["relative_path"]),
                content_type=str(raw["content_type"]),
            )
        )
    return _validated_specs(specs)


def preflight_transaction_root(run_root: str | Path) -> None:
    """Reject final output or transaction staging residue without writing."""

    root = Path(run_root)
    _require_real_directory(root, label="run root")
    final_root = root / FINAL_OUTPUT_DIRECTORY
    if _lexists(final_root):
        raise TabularOutputTransactionError(
            "The final tabular output directory already exists; existing_output=fail."
        )
    residue = transaction_staging_entries(root)
    if residue:
        raise TabularOutputTransactionError(
            "Tabular output staging residue is recovery evidence and prevents execution.",
            recovery_path=residue[0],
        )
    support_error = atomic_no_replace_support_error()
    if support_error is not None:
        raise TabularOutputTransactionError(support_error)


def transaction_staging_entries(run_root: str | Path) -> tuple[Path, ...]:
    """Return transaction staging entries, including symlinks and special files."""

    root = Path(run_root)
    if not _lexists(root) or not root.is_dir() or root.is_symlink():
        return ()
    return tuple(
        sorted(
            (
                entry
                for entry in root.iterdir()
                if entry.name.startswith(STAGING_PREFIX)
                and entry.name.endswith(STAGING_SUFFIX)
            ),
            key=lambda item: item.name,
        )
    )


def create_owned_staging(run_root: str | Path) -> OwnedStaging:
    """Exclusively create one hidden staging directory beneath the run root."""

    root = Path(run_root)
    preflight_transaction_root(root)
    root_identity = _path_identity(root)
    if root_identity is None:
        raise TabularOutputTransactionError("The tabular run root disappeared before staging.")
    path = Path(
        tempfile.mkdtemp(
            dir=root,
            prefix=STAGING_PREFIX,
            suffix=STAGING_SUFFIX,
        )
    )
    staging_identity: tuple[int, int] | None = None
    try:
        identity = os.lstat(path)
        if not stat.S_ISDIR(identity.st_mode) or stat.S_ISLNK(identity.st_mode):
            raise TabularOutputTransactionError(
                "The owned tabular staging path is not a real directory.",
                recovery_path=path,
            )
        if _path_identity(root) != root_identity:
            raise TabularOutputTransactionError(
                "The tabular run root changed while staging was created.",
                recovery_path=path,
            )
        staging_identity = (identity.st_dev, identity.st_ino)
        staging = OwnedStaging(path=path, device=identity.st_dev, inode=identity.st_ino)
        residue = transaction_staging_entries(root)
        if residue != (path,):
            raise TabularOutputTransactionError(
                "Foreign tabular staging residue appeared during staging creation.",
                recovery_path=path,
            )
        _fsync_directory(root)
        return staging
    except BaseException as exc:
        current_identity = _path_identity(path)
        if current_identity is None:
            raise
        if staging_identity is None or current_identity != staging_identity:
            raise TabularOutputTransactionError(
                "Tabular staging creation failed and an unowned recovery path remains.",
                recovery_path=path,
            ) from exc
        try:
            shutil.rmtree(path)
            _fsync_directory(root)
        except OSError as cleanup_error:
            raise TabularOutputTransactionError(
                "Tabular staging creation failed and owned cleanup requires recovery.",
                recovery_path=path,
            ) from cleanup_error
        raise


def validate_staged_outputs(
    staging: OwnedStaging,
    outputs: Sequence[OutputSpec],
) -> tuple[OutputRecord, ...]:
    """Validate and hash the exact pre-manifest scientific inventory."""

    _require_owned_staging(staging)
    specs = _validated_specs(outputs)
    expected_files = {spec.relative_path: spec for spec in specs}
    expected_directories = _required_directories(expected_files)
    observed_files, observed_directories = _observed_inventory(staging.path)
    if observed_files != set(expected_files) or observed_directories != expected_directories:
        raise TabularOutputTransactionError(
            "Tabular staging inventory does not match the planned scientific output set."
        )

    records: list[OutputRecord] = []
    for spec in specs:
        data = read_owned_regular_file(
            staging.path,
            spec.relative_path,
            expected_root_identity=staging.filesystem_identity,
        )
        if spec.content_type == "json":
            _validate_json_bytes(data)
            row_count: int | None = None
            columns: tuple[str, ...] = ()
        elif spec.content_type == "tsv":
            columns, row_count = _validate_tsv_bytes(data)
            if spec.expected_columns and columns != spec.expected_columns:
                raise TabularOutputTransactionError(
                    f"Tabular output {spec.logical_name!r} has an unexpected TSV header."
                )
        else:  # pragma: no cover - guarded by _validated_specs
            raise TabularOutputTransactionError(
                f"Unsupported tabular output content type {spec.content_type!r}."
            )
        records.append(
            OutputRecord(
                logical_name=spec.logical_name,
                relative_path=spec.relative_path,
                content_type=spec.content_type,
                byte_size=len(data),
                sha256=sha256(data).hexdigest(),
                row_count=row_count,
                columns=columns,
            )
        )
    return tuple(records)


def build_transaction_manifest(
    *,
    run_id: str,
    workflow_action: str,
    workflow_target: str,
    plan_identity_schema: str,
    plan_identity_sha256: str,
    outputs: Sequence[OutputRecord],
) -> dict[str, Any]:
    """Build the deterministic portable successful-output attestation."""

    try:
        validate_run_id(run_id)
    except RunLifecycleError as exc:
        raise TabularOutputTransactionError(str(exc)) from exc
    for label, value in {
        "workflow action": workflow_action,
        "workflow target": workflow_target,
    }.items():
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise TabularOutputTransactionError(
                f"Transaction {label} must be a nonempty string."
            )
    if not _is_sha256(plan_identity_sha256):
        raise TabularOutputTransactionError(
            "Reviewed plan identity must contain a lowercase SHA-256 digest."
        )
    if not isinstance(plan_identity_schema, str) or not plan_identity_schema.strip():
        raise TabularOutputTransactionError(
            "Reviewed plan identity schema must be a nonempty string."
        )
    records = tuple(outputs)
    if not records:
        raise TabularOutputTransactionError(
            "A successful tabular transaction must attest scientific outputs."
        )
    if len({record.logical_name for record in records}) != len(records):
        raise TabularOutputTransactionError(
            "Transaction output records contain duplicate logical names."
        )
    if len({record.relative_path for record in records}) != len(records):
        raise TabularOutputTransactionError(
            "Transaction output records contain duplicate relative paths."
        )
    for record in records:
        _validate_relative_path(record.relative_path)
        if record.content_type not in _CONTENT_TYPES:
            raise TabularOutputTransactionError("Transaction output content type is invalid.")
        if record.byte_size < 0 or not _is_sha256(record.sha256):
            raise TabularOutputTransactionError("Transaction output integrity record is invalid.")
        if record.content_type == "tsv" and (
            record.row_count is None
            or record.row_count < 0
            or not record.columns
            or len(set(record.columns)) != len(record.columns)
        ):
            raise TabularOutputTransactionError("Transaction TSV record is incomplete.")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "workflow": {
            "action": workflow_action,
            "target": workflow_target,
        },
        "plan_identity": {
            "schema_version": plan_identity_schema,
            "sha256": plan_identity_sha256,
        },
        "outputs": [record.to_dict() for record in records],
    }


def seal_staged_transaction(
    staging: OwnedStaging,
    *,
    outputs: Sequence[OutputSpec],
    run_id: str,
    workflow_action: str,
    workflow_target: str,
    plan_identity_schema: str,
    plan_identity_sha256: str,
    expected_records: Sequence[OutputRecord] | None = None,
) -> dict[str, Any]:
    """Validate outputs, write their manifest, revalidate, and flush the tree."""

    records = validate_staged_outputs(staging, outputs)
    if expected_records is not None and records != tuple(expected_records):
        raise TabularOutputTransactionError(
            "Staged tabular output bytes changed after cross-contract validation."
        )
    manifest = build_transaction_manifest(
        run_id=run_id,
        workflow_action=workflow_action,
        workflow_target=workflow_target,
        plan_identity_schema=plan_identity_schema,
        plan_identity_sha256=plan_identity_sha256,
        outputs=records,
    )
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _write_new_regular_file(staging, TRANSACTION_MANIFEST_NAME, manifest_bytes)
    validate_sealed_transaction(staging, outputs=outputs, expected_manifest=manifest)
    fsync_staging_tree(staging)
    validate_sealed_transaction(staging, outputs=outputs, expected_manifest=manifest)
    return manifest


def validate_sealed_transaction(
    staging: OwnedStaging,
    *,
    outputs: Sequence[OutputSpec],
    expected_manifest: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate an exact sealed tree and return its manifest and byte digest."""

    _require_owned_staging(staging)
    specs = _validated_specs(outputs)
    expected_files = {spec.relative_path for spec in specs} | {TRANSACTION_MANIFEST_NAME}
    observed_files, observed_directories = _observed_inventory(staging.path)
    if observed_files != expected_files or observed_directories != _required_directories(
        {path: None for path in expected_files}
    ):
        raise TabularOutputTransactionError(
            "Sealed tabular transaction inventory is incomplete or contains unexpected entries."
        )
    manifest_bytes = read_owned_regular_file(
        staging.path,
        TRANSACTION_MANIFEST_NAME,
        expected_root_identity=staging.filesystem_identity,
    )
    raw_manifest = _validate_json_bytes(manifest_bytes)
    if not isinstance(raw_manifest, dict):
        raise TabularOutputTransactionError(
            "Tabular transaction manifest must contain a JSON object."
        )
    if expected_manifest is not None and raw_manifest != dict(expected_manifest):
        raise TabularOutputTransactionError(
            "Tabular transaction manifest changed after it was written."
        )
    records = validate_staged_outputs_without_manifest(staging, specs)
    if raw_manifest.get("outputs") != [record.to_dict() for record in records]:
        raise TabularOutputTransactionError(
            "Tabular transaction manifest does not match the exact output bytes."
        )
    return raw_manifest, sha256(manifest_bytes).hexdigest()


def validate_committed_transaction(
    final_root: str | Path,
    *,
    outputs: Sequence[OutputSpec],
    expected_run_id: str,
    expected_workflow_action: str,
    expected_workflow_target: str,
    expected_plan_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], str, tuple[OutputRecord, ...]]:
    """Strictly verify one committed successful output transaction.

    Every scientific output is read once for both parsing and digest checks.
    The result is suitable for binding into a downstream reviewed plan.
    """

    root = Path(final_root)
    if root.name != FINAL_OUTPUT_DIRECTORY:
        raise TabularOutputTransactionError(
            "A committed tabular transaction directory must be named 'outputs'."
        )
    _require_real_directory(root, label="committed tabular output directory")
    identity = os.lstat(root)
    owned_view = OwnedStaging(path=root, device=identity.st_dev, inode=identity.st_ino)
    specs = _validated_specs(outputs)
    expected_files = {spec.relative_path for spec in specs} | {TRANSACTION_MANIFEST_NAME}
    observed_files, observed_directories = _observed_inventory(root)
    if observed_files != expected_files or observed_directories != _required_directories(
        {path: None for path in expected_files}
    ):
        raise TabularOutputTransactionError(
            "Committed tabular output inventory is incomplete or contains unexpected entries."
        )

    manifest_bytes = read_owned_regular_file(
        root,
        TRANSACTION_MANIFEST_NAME,
        expected_root_identity=owned_view.filesystem_identity,
    )
    raw_manifest = _validate_json_bytes(manifest_bytes)
    if not isinstance(raw_manifest, dict):
        raise TabularOutputTransactionError(
            "Tabular transaction manifest must contain a JSON object."
        )
    _validate_manifest_identity(
        raw_manifest,
        expected_run_id=expected_run_id,
        expected_workflow_action=expected_workflow_action,
        expected_workflow_target=expected_workflow_target,
        expected_plan_identity=expected_plan_identity,
    )
    records = validate_staged_outputs_without_manifest(owned_view, specs)
    if raw_manifest.get("outputs") != [record.to_dict() for record in records]:
        raise TabularOutputTransactionError(
            "Committed transaction manifest does not match the exact current output bytes."
        )
    return raw_manifest, sha256(manifest_bytes).hexdigest(), records


def validate_staged_outputs_without_manifest(
    staging: OwnedStaging,
    outputs: Sequence[OutputSpec],
) -> tuple[OutputRecord, ...]:
    """Validate outputs in a sealed tree while ignoring only its fixed manifest."""

    _require_owned_staging(staging)
    specs = _validated_specs(outputs)
    manifest_path = staging.path / TRANSACTION_MANIFEST_NAME
    if not _lexists(manifest_path):
        raise TabularOutputTransactionError("The sealed transaction manifest is missing.")
    manifest_identity = os.lstat(manifest_path)
    if not stat.S_ISREG(manifest_identity.st_mode) or stat.S_ISLNK(manifest_identity.st_mode):
        raise TabularOutputTransactionError("The sealed transaction manifest is not a regular file.")
    records: list[OutputRecord] = []
    for spec in specs:
        data = read_owned_regular_file(
            staging.path,
            spec.relative_path,
            expected_root_identity=staging.filesystem_identity,
        )
        if spec.content_type == "json":
            _validate_json_bytes(data)
            row_count = None
            columns: tuple[str, ...] = ()
        else:
            columns, row_count = _validate_tsv_bytes(data)
            if spec.expected_columns and columns != spec.expected_columns:
                raise TabularOutputTransactionError(
                    f"Tabular output {spec.logical_name!r} has an unexpected TSV header."
                )
        records.append(
            OutputRecord(
                logical_name=spec.logical_name,
                relative_path=spec.relative_path,
                content_type=spec.content_type,
                byte_size=len(data),
                sha256=sha256(data).hexdigest(),
                row_count=row_count,
                columns=columns,
            )
        )
    return tuple(records)


def read_owned_regular_file(
    root: str | Path,
    relative_path: str,
    *,
    expected_root_identity: tuple[int, int] | None = None,
) -> bytes:
    """Read and hash-safe-parse one confined nonsymlink file from one open."""

    base = Path(root)
    _require_real_directory(base, label="transaction root")
    initial_root_identity = _path_identity(base)
    if expected_root_identity is not None and initial_root_identity != expected_root_identity:
        raise TabularOutputTransactionError(
            "Transaction root changed filesystem identity before an output read."
        )
    relative = _validate_relative_path(relative_path)
    current = base
    for part in relative.parts[:-1]:
        current = current / part
        _require_real_directory(current, label="transaction output parent")
    path = base.joinpath(*relative.parts)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TabularOutputTransactionError(
            f"Transaction output {relative_path!r} cannot be opened safely."
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TabularOutputTransactionError(
                f"Transaction output {relative_path!r} must be one physically confined regular file."
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise TabularOutputTransactionError(
                f"Transaction output {relative_path!r} changed while it was read."
            )
        identity = os.lstat(path)
        if stat.S_ISLNK(identity.st_mode) or (
            identity.st_dev,
            identity.st_ino,
        ) != (after.st_dev, after.st_ino):
            raise TabularOutputTransactionError(
                f"Transaction output {relative_path!r} changed filesystem identity while read."
            )
        if _path_identity(base) != initial_root_identity:
            raise TabularOutputTransactionError(
                "Transaction root changed filesystem identity during an output read."
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def fsync_staging_tree(staging: OwnedStaging) -> None:
    """Flush every staged regular file and directory from leaves to root."""

    _require_owned_staging(staging)
    observed_files, observed_directories = _observed_inventory(staging.path)
    for relative_path in sorted(observed_files):
        path = staging.path / relative_path
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            identity = os.fstat(descriptor)
            if not stat.S_ISREG(identity.st_mode):
                raise TabularOutputTransactionError("Staged output changed before flush.")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for relative_path in sorted(
        observed_directories,
        key=lambda value: (value.count("/"), value),
        reverse=True,
    ):
        _fsync_directory(staging.path / relative_path)
    _fsync_directory(staging.path)


def promote_staging_no_replace(staging: OwnedStaging, final_root: str | Path) -> Path:
    """Atomically promote a flushed sibling staging tree without replacement."""

    _require_owned_staging(staging)
    final = Path(final_root)
    if staging.path.parent != final.parent:
        raise TabularOutputTransactionError(
            "Tabular staging and final output directories must be siblings."
        )
    if final.name != FINAL_OUTPUT_DIRECTORY:
        raise TabularOutputTransactionError(
            "The final tabular transaction directory must be named 'outputs'."
        )
    if _lexists(final):
        raise TabularOutputTransactionError(
            "The final tabular output directory was claimed before promotion."
        )
    support_error = atomic_no_replace_support_error()
    if support_error is not None:
        raise TabularOutputTransactionError(support_error)

    try:
        _atomic_no_replace_directory(staging.path, final)
    except BaseException as exc:
        if (
            _path_identity(staging.path) is None
            and _path_identity(final) == staging.filesystem_identity
        ):
            raise TabularOutputTransactionError(
                "Tabular output promotion completed but promotion cleanup is uncertain.",
                recovery_path=final,
                promotion_committed=True,
            ) from exc
        raise
    if _path_identity(final) != staging.filesystem_identity:
        raise TabularOutputTransactionError(
            "Tabular output promotion completed with an uncertain filesystem identity.",
            recovery_path=final,
            promotion_committed=True,
        )
    try:
        _fsync_directory(final.parent)
    except OSError as exc:
        raise TabularOutputTransactionError(
            "Tabular outputs were promoted but parent-directory durability is uncertain.",
            recovery_path=final,
            promotion_committed=True,
        ) from exc
    return final


def cleanup_owned_staging(staging: OwnedStaging) -> None:
    """Remove only the same staging directory object recorded at creation."""

    current = _path_identity(staging.path)
    if current is None:
        raise TabularOutputTransactionError(
            "Owned tabular staging disappeared before cleanup.",
            recovery_path=staging.path,
        )
    if current != staging.filesystem_identity:
        raise TabularOutputTransactionError(
            "Tabular staging was replaced and the foreign entry will not be removed.",
            recovery_path=staging.path,
        )
    identity = os.lstat(staging.path)
    if not stat.S_ISDIR(identity.st_mode) or stat.S_ISLNK(identity.st_mode):
        raise TabularOutputTransactionError(
            "Tabular staging became unsafe and will not be removed.",
            recovery_path=staging.path,
        )
    try:
        shutil.rmtree(staging.path)
        _fsync_directory(staging.path.parent)
    except OSError as exc:
        raise TabularOutputTransactionError(
            "Owned tabular staging could not be removed and requires recovery.",
            recovery_path=staging.path,
        ) from exc


def atomic_no_replace_support_error() -> str | None:
    """Return a stable fail-closed error when no safe directory rename exists."""

    try:
        library = ctypes.CDLL(None, use_errno=True)
    except OSError:
        return "This platform cannot load an atomic no-replace directory promotion primitive."
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        return None
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        return None
    return "This platform does not expose an atomic no-replace directory promotion primitive."


def _atomic_no_replace_directory(source: Path, destination: Path) -> None:
    parent_descriptor = os.open(
        source.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        library = ctypes.CDLL(None, use_errno=True)
        source_name = os.fsencode(source.name)
        destination_name = os.fsencode(destination.name)
        if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
            rename = library.renameatx_np
            rename.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            rename.restype = ctypes.c_int
            result = rename(
                parent_descriptor,
                source_name,
                parent_descriptor,
                destination_name,
                0x00000004,  # RENAME_EXCL
            )
        elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
            rename = library.renameat2
            rename.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            rename.restype = ctypes.c_int
            result = rename(
                parent_descriptor,
                source_name,
                parent_descriptor,
                destination_name,
                0x00000001,  # RENAME_NOREPLACE
            )
        else:
            raise TabularOutputTransactionError(
                "This platform does not expose an atomic no-replace directory promotion primitive."
            )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                raise TabularOutputTransactionError(
                    "The final tabular output directory was claimed concurrently before promotion."
                )
            raise OSError(error_number, os.strerror(error_number), os.fspath(destination))
    finally:
        os.close(parent_descriptor)


def _validated_specs(outputs: Sequence[OutputSpec]) -> tuple[OutputSpec, ...]:
    specs = tuple(outputs)
    if not specs:
        raise TabularOutputTransactionError(
            "A tabular output transaction must declare at least one scientific output."
        )
    names: set[str] = set()
    paths: set[str] = set()
    aliases: set[str] = set()
    for spec in specs:
        if not isinstance(spec, OutputSpec):
            raise TabularOutputTransactionError("Transaction outputs must be OutputSpec values.")
        if not spec.logical_name or spec.logical_name != spec.logical_name.strip():
            raise TabularOutputTransactionError("Transaction logical names must be nonempty.")
        relative = _validate_relative_path(spec.relative_path)
        if spec.relative_path == TRANSACTION_MANIFEST_NAME:
            raise TabularOutputTransactionError(
                "The transaction manifest is reserved and is not a scientific output."
            )
        if spec.content_type not in _CONTENT_TYPES:
            raise TabularOutputTransactionError(
                f"Unsupported tabular output content type {spec.content_type!r}."
            )
        if spec.expected_columns and (
            spec.content_type != "tsv"
            or any(not isinstance(column, str) or not column for column in spec.expected_columns)
            or len(set(spec.expected_columns)) != len(spec.expected_columns)
        ):
            raise TabularOutputTransactionError("Expected TSV columns are invalid.")
        alias = os.path.normcase(relative.as_posix()).casefold()
        if spec.logical_name in names or spec.relative_path in paths or alias in aliases:
            raise TabularOutputTransactionError(
                "Transaction outputs contain duplicate or aliased destinations."
            )
        names.add(spec.logical_name)
        paths.add(spec.relative_path)
        aliases.add(alias)
    return specs


def _validate_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_workflow_action: str,
    expected_workflow_target: str,
    expected_plan_identity: Mapping[str, Any],
) -> None:
    if set(manifest) != {
        "schema_version",
        "run_id",
        "workflow",
        "plan_identity",
        "outputs",
    }:
        raise TabularOutputTransactionError(
            "Tabular transaction manifest has an invalid structure."
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise TabularOutputTransactionError(
            "Tabular transaction manifest uses an unsupported schema version."
        )
    if manifest.get("run_id") != expected_run_id:
        raise TabularOutputTransactionError(
            "Tabular transaction manifest identifies a different source run."
        )
    if manifest.get("workflow") != {
        "action": expected_workflow_action,
        "target": expected_workflow_target,
    }:
        raise TabularOutputTransactionError(
            "Tabular transaction manifest identifies a different workflow."
        )
    if set(expected_plan_identity) != {"schema_version", "sha256"}:
        raise TabularOutputTransactionError(
            "Expected reviewed plan identity is incomplete."
        )
    plan_schema = expected_plan_identity.get("schema_version")
    plan_digest = expected_plan_identity.get("sha256")
    if not isinstance(plan_schema, str) or not plan_schema or not _is_sha256(plan_digest):
        raise TabularOutputTransactionError(
            "Expected reviewed plan identity is invalid."
        )
    if manifest.get("plan_identity") != dict(expected_plan_identity):
        raise TabularOutputTransactionError(
            "Tabular transaction manifest does not match the reviewed plan identity."
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise TabularOutputTransactionError(
            "Tabular transaction manifest must attest its scientific outputs."
        )


def _validate_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        raise TabularOutputTransactionError(
            "Transaction output paths must be nonempty portable relative paths."
        )
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise TabularOutputTransactionError(
            "Transaction output paths must remain beneath the output directory."
        )
    if relative.as_posix() != value:
        raise TabularOutputTransactionError(
            "Transaction output paths must use canonical POSIX separators."
        )
    return relative


def _required_directories(paths: Mapping[str, Any] | set[str]) -> set[str]:
    required: set[str] = set()
    for value in paths:
        relative = PurePosixPath(value)
        parent = relative.parent
        while parent != PurePosixPath("."):
            required.add(parent.as_posix())
            parent = parent.parent
    return required


def _observed_inventory(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        identity = os.lstat(path)
        if stat.S_ISLNK(identity.st_mode):
            raise TabularOutputTransactionError(
                "Tabular staging must not contain symbolic links."
            )
        if stat.S_ISREG(identity.st_mode):
            if identity.st_nlink != 1:
                raise TabularOutputTransactionError(
                    "Tabular staging must not contain hard-linked files."
                )
            files.add(relative)
        elif stat.S_ISDIR(identity.st_mode):
            directories.add(relative)
        else:
            raise TabularOutputTransactionError(
                "Tabular staging contains a special filesystem entry."
            )
    return files, directories


def _write_new_regular_file(staging: OwnedStaging, relative_path: str, data: bytes) -> None:
    _require_owned_staging(staging)
    relative = _validate_relative_path(relative_path)
    if len(relative.parts) != 1:
        raise TabularOutputTransactionError("Transaction manifest must be top-level.")
    path = staging.path / relative_path
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as exc:
        raise TabularOutputTransactionError(
            "The transaction manifest destination is not exclusively available."
        ) from exc
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(errno.EIO, "Short transaction manifest write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_json_bytes(data: bytes) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TabularOutputTransactionError("Tabular JSON output must be UTF-8.") from exc
    try:
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise TabularOutputTransactionError(
            "Tabular JSON output must contain strict finite JSON."
        ) from exc
    _validate_finite_json(payload)
    return payload


def _validate_tsv_bytes(data: bytes) -> tuple[tuple[str, ...], int]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TabularOutputTransactionError("Tabular TSV output must be UTF-8.") from exc
    reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise TabularOutputTransactionError("Tabular TSV output must contain a header.") from exc
    if not header or any(not column for column in header) or len(set(header)) != len(header):
        raise TabularOutputTransactionError(
            "Tabular TSV output must contain a nonempty unique header."
        )
    width = len(header)
    row_count = 0
    for row in reader:
        if len(row) != width:
            raise TabularOutputTransactionError(
                "Tabular TSV output row width does not match its header."
            )
        for value in row:
            _validate_finite_text(value)
        row_count += 1
    return tuple(header), row_count


def _validate_finite_text(value: str) -> None:
    stripped = value.strip()
    if not stripped:
        return
    if stripped.casefold() in {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }:
        raise TabularOutputTransactionError("Tabular TSV output contains a non-finite value.")
    if stripped.startswith(("[", "{")):
        try:
            parsed = json.loads(
                stripped,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
        except json.JSONDecodeError:
            return
        except ValueError as exc:
            raise TabularOutputTransactionError(
                "Tabular TSV output contains invalid embedded JSON."
            ) from exc
        _validate_finite_json(parsed)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in pairs:
        if key in resolved:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        resolved[key] = value
    return resolved


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Forbidden JSON numeric constant {value!r}.")


def _validate_finite_json(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_json(str(key))
            _validate_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite_json(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise TabularOutputTransactionError("Tabular JSON output contains a non-finite value.")


def _require_owned_staging(staging: OwnedStaging) -> None:
    identity = _path_identity(staging.path)
    if identity != staging.filesystem_identity:
        raise TabularOutputTransactionError(
            "Owned tabular staging changed filesystem identity.",
            recovery_path=staging.path,
        )
    _require_real_directory(staging.path, label="owned tabular staging")


def _require_real_directory(path: Path, *, label: str) -> None:
    try:
        identity = os.lstat(path)
    except OSError as exc:
        raise TabularOutputTransactionError(f"The {label} is unavailable.") from exc
    if not stat.S_ISDIR(identity.st_mode) or stat.S_ISLNK(identity.st_mode):
        raise TabularOutputTransactionError(f"The {label} must be a real directory.")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_identity(path: Path) -> tuple[int, int] | None:
    try:
        identity = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    return identity.st_dev, identity.st_ino


def _lexists(path: Path) -> bool:
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_DIGEST_RE)
    )
