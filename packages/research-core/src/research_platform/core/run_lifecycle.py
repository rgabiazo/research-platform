"""Safe run identities, reviewed-plan digests, and execution claims."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping


PLAN_IDENTITY_SCHEMA = "research_platform.core.run_plan_identity.v1"
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_PLAN_IDENTITY_KEYS = frozenset({"schema_version", "sha256", "files"})


class RunLifecycleError(ValueError):
    """A run path, reviewed plan, or execution claim is unsafe."""

    @classmethod
    def for_reuse(cls, run_id: str, reason: str) -> RunLifecycleError:
        """Build the stable, actionable error used for rejected run reuse."""

        return cls(
            f"Run {run_id!r} cannot be reused because {reason}. "
            "Inspect the existing run and choose a new run id; there is no overwrite, "
            "resume, retry, replace, or force option."
        )


def validate_run_id(run_id: str) -> str:
    """Return *run_id* unchanged when it is one portable path segment."""

    if not isinstance(run_id, str):
        raise RunLifecycleError("Run id must be a nonempty string.")
    if not run_id or not run_id.strip():
        raise RunLifecycleError("Run id must be a nonempty filesystem name.")
    if run_id != run_id.strip():
        raise RunLifecycleError("Run id must not have leading or trailing whitespace.")
    if run_id in {".", ".."}:
        raise RunLifecycleError("Run id must not be '.' or '..'.")
    if "/" in run_id or "\\" in run_id:
        raise RunLifecycleError("Run id must be one filesystem name, not a path.")
    if any(ord(character) < 32 or ord(character) == 127 for character in run_id):
        raise RunLifecycleError("Run id must not contain control characters.")
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise RunLifecycleError(
            "Run id may contain only ASCII letters, digits, '.', '_', and '-', "
            "and must begin with a letter or digit."
        )
    if run_id.endswith("."):
        raise RunLifecycleError("Run id must not end with '.'.")
    if len(run_id.encode("utf-8")) > 248:
        raise RunLifecycleError("Run id must not exceed 248 UTF-8 bytes.")
    return run_id


def resolved_run_path(artifacts_root: str | Path, run_id: str) -> Path:
    """Return the contained run path without following a run-root symlink."""

    safe_run_id = validate_run_id(run_id)
    runs_root = (Path(artifacts_root).expanduser() / "runs").resolve(strict=False)
    candidate = runs_root / safe_run_id
    if candidate.parent != runs_root:
        raise RunLifecycleError("Run id resolves outside the configured artifacts/runs directory.")
    return candidate


def claim_path(artifacts_root: str | Path, run_id: str) -> Path:
    """Return the hidden sibling execution-claim path for *run_id*."""

    run_root = resolved_run_path(artifacts_root, run_id)
    return run_root.parent / f".{run_root.name}.claim"


def path_entry_exists(path: str | Path) -> bool:
    """Return whether a filesystem entry exists, including a broken symlink."""

    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


class ExecutionClaim:
    """An exclusively created execution claim with recorded filesystem identity."""

    def __init__(self, path: Path, *, device: int, inode: int) -> None:
        self.path = path
        self.device = device
        self.inode = inode
        self._released = False

    @property
    def filesystem_identity(self) -> tuple[int, int]:
        """Return the device/inode pair recorded when the claim was acquired."""

        return self.device, self.inode

    @property
    def released(self) -> bool:
        """Return whether this instance successfully removed its owned claim."""

        return self._released

    def release(self) -> None:
        """Remove only the same, still-empty directory this instance created."""

        if self._released:
            return
        try:
            current = os.lstat(self.path)
        except FileNotFoundError as exc:
            raise RunLifecycleError(
                f"Execution claim disappeared before owned cleanup: {self.path}"
            ) from exc
        if not stat.S_ISDIR(current.st_mode):
            raise RunLifecycleError(
                f"Execution claim was replaced by an unsafe filesystem entry: {self.path}"
            )
        if (current.st_dev, current.st_ino) != self.filesystem_identity:
            raise RunLifecycleError(
                f"Execution claim was replaced and will not be removed: {self.path}"
            )
        try:
            self.path.rmdir()
        except OSError as exc:
            raise RunLifecycleError(
                f"Owned execution claim is not an empty removable directory: {self.path}"
            ) from exc
        self._released = True


def acquire_execution_claim(artifacts_root: str | Path, run_id: str) -> ExecutionClaim:
    """Atomically create and return the execution claim for *run_id*."""

    path = claim_path(artifacts_root, run_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RunLifecycleError(f"Could not prepare the run directory: {path.parent}") from exc
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise RunLifecycleError(
            f"Run {validate_run_id(run_id)!r} already has an execution claim. "
            "Inspect the existing run and choose a new run id."
        ) from exc
    except OSError as exc:
        raise RunLifecycleError(f"Could not acquire execution claim: {path}") from exc

    try:
        identity = os.lstat(path)
    except OSError as exc:
        raise RunLifecycleError(
            f"Execution claim could not be inspected after creation: {path}"
        ) from exc
    if not stat.S_ISDIR(identity.st_mode):
        raise RunLifecycleError(f"Execution claim is not a directory: {path}")
    return ExecutionClaim(path, device=identity.st_dev, inode=identity.st_ino)


def build_plan_identity(
    manifest: Mapping[str, Any],
    execute_script: bytes,
    slurm_script: bytes | None = None,
    *,
    extra_files: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    """Build a deterministic identity for stable plan semantics and reviewed bytes."""

    canonical_manifest = _canonical_manifest(manifest)
    files = _identity_file_records(
        execute_script=execute_script,
        slurm_script=slurm_script,
        extra_files=extra_files,
    )
    payload = {
        "schema_version": PLAN_IDENTITY_SCHEMA,
        "manifest": canonical_manifest,
        "files": files,
    }
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RunLifecycleError("Run plan identity contains a non-JSON-safe value.") from exc
    return {
        "schema_version": PLAN_IDENTITY_SCHEMA,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def verify_plan_identity(
    manifest: Mapping[str, Any],
    execute_script: bytes,
    slurm_script: bytes | None = None,
    *,
    extra_files: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    """Verify and return the reviewed identity stored in *manifest*."""

    stored = manifest.get("plan_identity")
    if not isinstance(stored, dict):
        raise RunLifecycleError("Run manifest has no complete plan_identity mapping.")
    if set(stored) != _PLAN_IDENTITY_KEYS:
        raise RunLifecycleError("Run manifest plan_identity has an invalid structure.")
    if stored.get("schema_version") != PLAN_IDENTITY_SCHEMA:
        raise RunLifecycleError("Run manifest plan_identity uses an unsupported schema version.")
    digest = stored.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RunLifecycleError("Run manifest plan_identity has an invalid SHA-256 digest.")
    if not isinstance(stored.get("files"), dict):
        raise RunLifecycleError("Run manifest plan_identity has invalid reviewed-file records.")

    expected = build_plan_identity(
        manifest,
        execute_script,
        slurm_script,
        extra_files=extra_files,
    )
    if not _constant_time_text_equal(str(stored["sha256"]), expected["sha256"]):
        raise RunLifecycleError("Run manifest plan identity does not match the reviewed plan.")
    if stored["files"] != expected["files"]:
        raise RunLifecycleError("Run manifest reviewed-file identity does not match stored files.")
    return copy.deepcopy(stored)


def _canonical_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise RunLifecycleError("Run plan identity requires a manifest mapping.")
    canonical = copy.deepcopy(dict(manifest))
    canonical.pop("created_at", None)
    canonical.pop("plan_identity", None)
    canonical.pop("submission_identity", None)
    execution = canonical.get("execution")
    if isinstance(execution, dict):
        execution = dict(execution)
        if execution.get("mode") in {"plan", "local"}:
            execution["mode"] = "local"
            execution["dry_run"] = False
        canonical["execution"] = execution
    return canonical


def _identity_file_records(
    *,
    execute_script: bytes,
    slurm_script: bytes | None,
    extra_files: Mapping[str, bytes] | None,
) -> dict[str, dict[str, Any]]:
    records: dict[str, bytes] = {"execute.sh": _require_bytes(execute_script, label="execute.sh")}
    if slurm_script is not None:
        records["submit.sbatch"] = _require_bytes(slurm_script, label="submit.sbatch")
    if extra_files is not None:
        if not isinstance(extra_files, Mapping):
            raise RunLifecycleError("extra_files must be a mapping of stable labels to bytes.")
        for raw_label, content in extra_files.items():
            if not isinstance(raw_label, str) or not raw_label:
                raise RunLifecycleError("Reviewed-file labels must be nonempty strings.")
            if raw_label in records:
                raise RunLifecycleError(f"Duplicate reviewed-file label: {raw_label}")
            records[raw_label] = _require_bytes(content, label=raw_label)
    return {
        label: {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
        for label, content in sorted(records.items())
    }


def _require_bytes(value: Any, *, label: str) -> bytes:
    if not isinstance(value, bytes):
        raise RunLifecycleError(f"Reviewed file {label!r} must be supplied as bytes.")
    return value


def _constant_time_text_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)
