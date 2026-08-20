"""Failure-safe staging and promotion for local ROI runtime outputs.

The helpers in this module operate only after a caller has explicitly entered
an execution path.  Preflight is read-only; creating staging directories is a
separate operation so planning and readiness checks cannot mutate the
filesystem accidentally.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os
import shutil
import tempfile


ExistingOutputPolicy = Literal["fail", "replace"]
OutputKind = Literal["file", "directory"]


class RoiRuntimeOutputError(RuntimeError):
    """Raised when a ROI runtime output set is unsafe or cannot be committed."""


@dataclass(frozen=True)
class RoiRuntimeOutput:
    """One declared destination in a local ROI runtime operation."""

    destination: Path
    category: str
    kind: OutputKind = "file"
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination", Path(self.destination))
        if self.kind not in {"file", "directory"}:
            raise ValueError("ROI runtime output kind must be file or directory.")


def preflight_runtime_outputs(
    outputs: Sequence[RoiRuntimeOutput],
    *,
    existing_output: str,
) -> tuple[RoiRuntimeOutput, ...]:
    """Validate a complete runtime destination set without writing anything."""

    policy = _existing_output_policy(existing_output)
    ordered = tuple(
        sorted(
            outputs,
            key=lambda item: (0 if item.kind == "directory" else 1, str(item.destination), item.category),
        )
    )
    duplicates = _duplicates(item.destination for item in ordered)
    if duplicates:
        raise RoiRuntimeOutputError(
            "ROI runtime output planning produced duplicate destinations: "
            + ", ".join(str(path) for path in duplicates)
            + "."
        )
    for output in ordered:
        _preflight_destination(output, existing_output=policy)
    return ordered


class RoiRuntimeOutputTransaction:
    """Stage outputs and promote the complete set with rollback.

    A transaction may span more than one filesystem.  Each output is staged
    beneath an existing directory on the same device as its destination.  The
    promotion sequence is therefore atomic per destination and rollback
    restores the complete pre-existing destination set after an ordinary
    failure. Tool-owned outputs that cannot be redirected may be prepared as
    guarded direct outputs: prior content is moved or copied into transaction
    storage before the tool runs and restored if the operation does not commit.
    """

    def __init__(
        self,
        outputs: Sequence[RoiRuntimeOutput],
        *,
        existing_output: str,
    ) -> None:
        self._policy = _existing_output_policy(existing_output)
        self._outputs = preflight_runtime_outputs(outputs, existing_output=self._policy)
        self._roots: dict[int, Path] = {}
        self._root_for_destination: dict[Path, Path] = {}
        self._candidates: dict[Path, Path] = {}
        self._sibling_candidates: set[Path] = set()
        self._direct_outputs: set[Path] = set()
        self._backups: dict[Path, Path] = {}
        self._promoted: list[Path] = []
        self._created_directories: list[Path] = []
        self._committed = False
        self._recovery_required = False
        self._closed = False

    @property
    def outputs(self) -> tuple[RoiRuntimeOutput, ...]:
        return self._outputs

    @property
    def temporary_roots(self) -> tuple[Path, ...]:
        return tuple(self._roots.values())

    def __enter__(self) -> "RoiRuntimeOutputTransaction":
        try:
            for index, output in enumerate(self._outputs):
                parent = _nearest_existing_directory(output.destination.parent)
                device = int(parent.stat().st_dev)
                transaction_root = self._roots.get(device)
                if transaction_root is None:
                    transaction_root = Path(tempfile.mkdtemp(prefix=".roi-runtime-", dir=parent))
                    self._roots[device] = transaction_root
                self._root_for_destination[output.destination] = transaction_root
                candidate = transaction_root / "candidate" / f"{index:06d}" / output.destination.name
                self._candidates[output.destination] = candidate
            return self
        except Exception:
            self.cleanup()
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if not self._committed and not self._recovery_required:
            try:
                self._rollback()
            except RoiRuntimeOutputError as rollback_error:
                self._recovery_required = True
                if exc is not None:
                    raise rollback_error from exc
                raise
        if not self._recovery_required:
            self.cleanup()
        return False

    def candidate_path(self, destination: str | Path) -> Path:
        """Return the same-filesystem staging path for a declared destination."""

        key = Path(destination)
        try:
            candidate = self._candidates[key]
        except KeyError as exc:
            raise RoiRuntimeOutputError(f"ROI runtime destination was not declared for staging: {key}.") from exc
        if key in self._direct_outputs:
            raise RoiRuntimeOutputError(f"ROI runtime destination is configured as a guarded direct output: {key}.")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def sibling_candidate_path(self, destination: str | Path) -> Path:
        """Return an absent hidden sibling path for a directory-writing tool."""

        if self._closed or self._committed:
            raise RoiRuntimeOutputError("ROI runtime output transaction is already closed.")
        key = Path(destination)
        output = self._output_for(key)
        if output.kind != "directory":
            raise RoiRuntimeOutputError(f"ROI runtime sibling candidate must be a directory destination: {key}.")
        if key in self._direct_outputs:
            raise RoiRuntimeOutputError(f"ROI runtime destination is configured as a guarded direct output: {key}.")
        root = self._root_for_destination[key]
        index = next(i for i, item in enumerate(self._outputs) if item.destination == key)
        candidate = key.parent / f"{root.name}-output-{index:06d}"
        if candidate.exists() or candidate.is_symlink():
            raise RoiRuntimeOutputError(f"ROI runtime sibling staging destination unexpectedly exists: {candidate}.")
        self._candidates[key] = candidate
        self._sibling_candidates.add(candidate)
        return candidate

    def prepare_direct_output(
        self,
        destination: str | Path,
        *,
        preserve_existing_for_read: bool = False,
    ) -> Path:
        """Guard a tool-owned output that cannot be redirected.

        This method must be called only after the complete read-only preflight.
        Existing content is moved into same-filesystem transaction storage by
        default. A cache that the tool may read can instead be copied into the
        backup while remaining in place. Failure or an uncommitted context
        restores the prior destination either way.
        """

        if self._closed or self._committed:
            raise RoiRuntimeOutputError("ROI runtime output transaction is already closed.")
        key = Path(destination)
        output = self._output_for(key)
        if key in self._direct_outputs:
            return key
        _preflight_destination(output, existing_output=self._policy)
        if key.exists():
            backup = self._backup_path(key)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if preserve_existing_for_read:
                _copy_existing_output(key, backup, kind=output.kind)
            else:
                os.replace(key, backup)
            self._backups[key] = backup
        _ensure_parent_directories(key.parent, self._created_directories)
        self._direct_outputs.add(key)
        return key

    def promote(self) -> None:
        """Validate staged outputs and atomically promote the complete set."""

        if self._closed:
            raise RoiRuntimeOutputError("ROI runtime output transaction is already closed.")
        self._restore_unwritten_optional_direct_outputs()
        active_outputs: list[RoiRuntimeOutput] = []
        for output in self._outputs:
            candidate = output.destination if output.destination in self._direct_outputs else self._candidates[output.destination]
            if not output.required and not candidate.exists() and not candidate.is_symlink():
                continue
            if candidate.is_symlink() or not _matches_kind(candidate, output.kind):
                raise RoiRuntimeOutputError(
                    f"ROI runtime staged {output.category} is missing or has the wrong type: {output.destination}."
                )
            active_outputs.append(output)

        # Recheck immediately before mutation to catch destination changes that
        # occurred after the read-only preflight.
        for output in active_outputs:
            if output.destination in self._direct_outputs:
                continue
            _preflight_destination(output, existing_output=self._policy)

        current = Path("<unknown>")
        try:
            for output in active_outputs:
                destination = output.destination
                if destination in self._direct_outputs or not destination.exists():
                    continue
                backup = self._backup_path(destination)
                backup.parent.mkdir(parents=True, exist_ok=True)
                if output.kind == "directory":
                    os.replace(destination, backup)
                else:
                    _stage_existing_file(destination, backup)
                self._backups[destination] = backup

            for output in active_outputs:
                current = output.destination
                if current in self._direct_outputs:
                    continue
                _ensure_parent_directories(current.parent, self._created_directories)
                os.replace(self._candidates[current], current)
                self._promoted.append(current)
        except Exception as exc:
            try:
                self._rollback()
            except RoiRuntimeOutputError as rollback_error:
                self._recovery_required = True
                raise rollback_error from exc
            raise RoiRuntimeOutputError(
                f"ROI runtime output promotion failed at {current}; the prior destination set was restored."
            ) from exc
        self._committed = True

    def _restore_unwritten_optional_direct_outputs(self) -> None:
        for output in self._outputs:
            destination = output.destination
            if output.required or destination not in self._direct_outputs or destination.exists():
                continue
            backup = self._backups.pop(destination, None)
            if backup is not None and backup.exists():
                _restore_backup(destination, backup)

    def _output_for(self, destination: Path) -> RoiRuntimeOutput:
        for output in self._outputs:
            if output.destination == destination:
                return output
        raise RoiRuntimeOutputError(f"ROI runtime destination was not declared for staging: {destination}.")

    def _backup_path(self, destination: Path) -> Path:
        transaction_root = self._root_for_destination[destination]
        position = next(i for i, output in enumerate(self._outputs) if output.destination == destination)
        return transaction_root / "backup" / f"{position:06d}" / destination.name

    def _rollback(self) -> None:
        restore = set(self._promoted) | set(self._direct_outputs) | {
            destination
            for destination, backup in self._backups.items()
            if backup.exists() and destination not in self._promoted
        }
        rollback_failed = False
        for destination in sorted(restore, key=lambda path: len(path.parts), reverse=True):
            backup = self._backups.get(destination)
            try:
                if backup is None:
                    _remove_output(destination)
                elif backup.exists():
                    _restore_backup(destination, backup)
            except Exception:
                rollback_failed = True
        for directory in sorted(set(self._created_directories), key=lambda item: len(item.parts), reverse=True):
            if directory.exists() and not _retry(directory.rmdir):
                rollback_failed = True
        if rollback_failed:
            raise RoiRuntimeOutputError(
                "ROI runtime output rollback could not restore the complete prior destination set; staging was retained for recovery."
            )
        self._promoted.clear()
        self._direct_outputs.clear()
        self._backups.clear()
        self._created_directories.clear()

    def cleanup(self) -> None:
        """Remove all transaction-owned staging trees, retrying once."""

        if self._closed:
            return
        if self._recovery_required:
            raise RoiRuntimeOutputError(
                "ROI runtime output staging was retained because manual recovery may be required."
            )
        failed: list[Path] = []
        for root in self._roots.values():
            if root.exists() and not _retry(lambda root=root: shutil.rmtree(root)):
                failed.append(root)
        for candidate in self._sibling_candidates:
            if (candidate.exists() or candidate.is_symlink()) and not _retry(
                lambda candidate=candidate: _remove_output(candidate)
            ):
                failed.append(candidate)
        self._closed = True
        if failed:
            raise RoiRuntimeOutputError(
                "ROI runtime output staging could not be removed: " + ", ".join(str(path) for path in failed) + "."
            )


def _existing_output_policy(value: str) -> ExistingOutputPolicy:
    policy = str(value).strip()
    if policy not in {"fail", "replace"}:
        raise RoiRuntimeOutputError("ROI runtime existing_output must be one of: fail, replace.")
    return policy  # type: ignore[return-value]


def _preflight_destination(output: RoiRuntimeOutput, *, existing_output: ExistingOutputPolicy) -> None:
    destination = output.destination
    if not destination.is_absolute():
        raise RoiRuntimeOutputError(f"ROI runtime {output.category} destination must be an absolute resolved path: {destination}.")
    _reject_symlink_ancestors(destination)

    parent = destination.parent
    nearest = _nearest_existing_path(parent)
    if nearest.exists() and not nearest.is_dir():
        raise RoiRuntimeOutputError(
            f"ROI runtime {output.category} destination parent is not a directory: {nearest}."
        )

    if destination.is_symlink():
        raise RoiRuntimeOutputError(
            f"ROI runtime {output.category} destination is a symbolic link and cannot be replaced safely: {destination}."
        )
    if not destination.exists():
        return
    if output.kind == "file" and not destination.is_file():
        raise RoiRuntimeOutputError(
            f"ROI runtime {output.category} destination is not a regular file: {destination}."
        )
    if output.kind == "directory" and not destination.is_dir():
        raise RoiRuntimeOutputError(
            f"ROI runtime {output.category} destination is not a directory: {destination}."
        )
    if existing_output == "fail":
        raise RoiRuntimeOutputError(
            f"ROI runtime {output.category} already exists: {destination}. Set runtime.existing_output to replace to authorize replacement."
        )


def _reject_symlink_ancestors(destination: Path) -> None:
    current = destination.parent
    while current != current.parent:
        if current.is_symlink():
            raise RoiRuntimeOutputError(f"ROI runtime destination parent is a symbolic link: {current}.")
        current = current.parent


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists() and not current.is_symlink() and current != current.parent:
        current = current.parent
    return current


def _nearest_existing_directory(path: Path) -> Path:
    current = _nearest_existing_path(path)
    if not current.is_dir():
        raise RoiRuntimeOutputError(f"ROI runtime output has no usable staging filesystem beneath {path}.")
    return current


def _duplicates(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    duplicates: list[Path] = []
    for path in paths:
        normalized = _lexically_normalized_absolute_path(path)
        if normalized in seen and normalized not in duplicates:
            duplicates.append(normalized)
        seen.add(normalized)
    return tuple(sorted(duplicates, key=str))


def _lexically_normalized_absolute_path(path: Path) -> Path:
    if not path.is_absolute():
        return path
    return Path(os.path.abspath(os.path.normpath(str(path))))


def _stage_existing_file(source: Path, backup: Path) -> None:
    try:
        os.link(source, backup)
    except OSError:
        shutil.copy2(source, backup)


def _copy_existing_output(source: Path, backup: Path, *, kind: OutputKind) -> None:
    if kind == "directory":
        shutil.copytree(source, backup, symlinks=True)
    else:
        shutil.copy2(source, backup)


def _restore_backup(destination: Path, backup: Path) -> None:
    _remove_output(destination)
    os.replace(backup, destination)


def _remove_output(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _matches_kind(path: Path, kind: OutputKind) -> bool:
    return path.is_file() if kind == "file" else path.is_dir()


def _ensure_parent_directories(path: Path, created: list[Path]) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists() and current != current.parent:
        if current.is_symlink():
            raise RoiRuntimeOutputError(f"ROI runtime destination parent is a symbolic link: {current}.")
        missing.append(current)
        current = current.parent
    if current.is_symlink():
        raise RoiRuntimeOutputError(f"ROI runtime destination parent is a symbolic link: {current}.")
    if not current.is_dir():
        raise RoiRuntimeOutputError(f"ROI runtime destination parent is not a directory: {current}.")
    for directory in reversed(missing):
        directory.mkdir()
        created.append(directory)


def _retry(operation: Callable[[], object]) -> bool:
    for _attempt in range(2):
        try:
            operation()
        except Exception:
            continue
        return True
    return False


__all__ = [
    "RoiRuntimeOutput",
    "RoiRuntimeOutputError",
    "RoiRuntimeOutputTransaction",
    "preflight_runtime_outputs",
]
