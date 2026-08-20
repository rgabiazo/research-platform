#!/usr/bin/env python3
"""Capture and verify a deterministic source-tree integrity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Iterable


SCHEMA_VERSION = "research-platform.ci.source-tree-integrity.v1"
_READ_SIZE = 1024 * 1024


class IntegrityError(ValueError):
    """Raised when an integrity manifest cannot be captured or verified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_READ_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(path: Path, relative_path: str) -> dict[str, Any]:
    metadata = path.lstat()
    mode = f"{stat.S_IMODE(metadata.st_mode):04o}"

    if stat.S_ISLNK(metadata.st_mode):
        return {
            "path": relative_path,
            "type": "symlink",
            "mode": mode,
            "target": os.readlink(path),
        }
    if stat.S_ISDIR(metadata.st_mode):
        return {"path": relative_path, "type": "directory", "mode": mode}
    if stat.S_ISREG(metadata.st_mode):
        return {
            "path": relative_path,
            "type": "file",
            "mode": mode,
            "size": metadata.st_size,
            "sha256": _sha256(path),
        }
    return {
        "path": relative_path,
        "type": "special",
        "mode": mode,
        "device": metadata.st_rdev,
    }


def _walk(root: Path) -> Iterable[tuple[Path, str]]:
    """Yield all entries below *root* in deterministic path order."""

    def visit(directory: Path, prefix: str) -> Iterable[tuple[Path, str]]:
        try:
            with os.scandir(directory) as scanner:
                children = sorted(scanner, key=lambda item: item.name)
        except OSError as error:
            raise IntegrityError(f"cannot read source directory: {directory}") from error

        for child in children:
            relative_path = f"{prefix}/{child.name}" if prefix else child.name
            if not prefix and child.name == ".git":
                continue
            path = Path(child.path)
            yield path, Path(relative_path).as_posix()
            try:
                is_directory = child.is_dir(follow_symlinks=False)
            except OSError as error:
                raise IntegrityError(f"cannot inspect source path: {relative_path}") from error
            if is_directory:
                yield from visit(path, relative_path)

    yield from visit(root, "")


def capture_tree(root: Path) -> dict[str, Any]:
    """Return a path-independent manifest for *root* without modifying it."""

    root = root.resolve()
    if not root.is_dir():
        raise IntegrityError(f"source root is not a directory: {root}")
    entries = sorted(
        (_entry(path, relative_path) for path, relative_path in _walk(root)),
        key=lambda item: item["path"],
    )
    return {"schema_version": SCHEMA_VERSION, "entries": entries}


def _assert_outside_root(root: Path, destination: Path, label: str) -> None:
    resolved_root = root.resolve()
    resolved_destination = destination.resolve(strict=False)
    try:
        resolved_destination.relative_to(resolved_root)
    except ValueError:
        return
    raise IntegrityError(f"{label} must be outside the measured source root")


def _write_manifest(manifest: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            temporary_path = Path(stream.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IntegrityError(f"baseline manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"baseline manifest is not valid JSON: {path}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise IntegrityError(f"baseline manifest has an unsupported schema: {path}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
        raise IntegrityError(f"baseline manifest has invalid entries: {path}")
    paths = [item.get("path") for item in entries]
    if any(not isinstance(value, str) or not value for value in paths):
        raise IntegrityError(f"baseline manifest has invalid entry paths: {path}")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise IntegrityError(f"baseline manifest entries are not uniquely sorted: {path}")
    return manifest


def _entry_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["path"]: entry for entry in manifest["entries"]}


def _describe_changes(
    baseline: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    expected = _entry_map(baseline)
    observed = _entry_map(current)
    lines: list[str] = []
    for path in sorted(observed.keys() - expected.keys()):
        lines.append(f"added: {path}")
    for path in sorted(expected.keys() - observed.keys()):
        lines.append(f"removed: {path}")
    for path in sorted(expected.keys() & observed.keys()):
        if expected[path] == observed[path]:
            continue
        if expected[path].get("type") != observed[path].get("type"):
            lines.append(
                "type changed: "
                f"{path} ({expected[path].get('type')} -> {observed[path].get('type')})"
            )
        else:
            lines.append(f"changed: {path}")
    return lines


def capture_command(root: Path, output: Path) -> None:
    _assert_outside_root(root, output, "baseline output")
    manifest = capture_tree(root)
    _write_manifest(manifest, output)
    print(f"Captured source-tree baseline: {output}")


def verify_command(root: Path, baseline_path: Path) -> None:
    _assert_outside_root(root, baseline_path, "baseline manifest")
    baseline = _load_manifest(baseline_path)
    current = capture_tree(root)
    changes = _describe_changes(baseline, current)
    if changes:
        details = "\n".join(f"  {line}" for line in changes)
        raise IntegrityError(f"tested source copy changed after baseline:\n{details}")
    print(f"Source-tree integrity verified: {root}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture or verify a deterministic CI source-tree manifest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="write a source-tree baseline")
    capture.add_argument("--root", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="compare a source tree with a baseline")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--baseline", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            capture_command(args.root, args.output)
        else:
            verify_command(args.root, args.baseline)
    except (IntegrityError, OSError) as error:
        print(f"source-tree integrity error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
