
"""Sync command builders for staged HPC flows."""

from __future__ import annotations

from pathlib import Path


def build_rsync_push_command(
    *,
    source: str | Path,
    ssh_host: str,
    destination: str,
    exclude_file: str | Path | None = None,
    exclude_files: list[str | Path] | None = None,
    source_is_directory: bool = True,
) -> list[str]:
    command = ["rsync", "-az"]
    for resolved_exclude in _normalize_exclude_files(exclude_file=exclude_file, exclude_files=exclude_files):
        command.extend(["--exclude-from", resolved_exclude])
    formatted_source = _format_local_path(source, is_directory=source_is_directory)
    formatted_destination = _format_remote_path(ssh_host=ssh_host, value=destination, is_directory=source_is_directory)
    command.extend([formatted_source, formatted_destination])
    return command


def build_rsync_pull_command(
    *,
    ssh_host: str,
    source: str,
    destination: str | Path,
    exclude_file: str | Path | None = None,
    exclude_files: list[str | Path] | None = None,
    progress: bool = False,
    source_is_directory: bool = True,
) -> list[str]:
    command = ["rsync", "-az"]
    if progress:
        command.append("--progress")
    for resolved_exclude in _normalize_exclude_files(exclude_file=exclude_file, exclude_files=exclude_files):
        command.extend(["--exclude-from", resolved_exclude])
    formatted_source = _format_remote_path(ssh_host=ssh_host, value=source, is_directory=source_is_directory)
    formatted_destination = _format_local_path(destination, is_directory=source_is_directory)
    command.extend([formatted_source, formatted_destination])
    return command


def _normalize_exclude_files(
    *,
    exclude_file: str | Path | None,
    exclude_files: list[str | Path] | None,
) -> list[str]:
    candidates: list[str | Path] = []
    if exclude_file is not None:
        candidates.append(exclude_file)
    candidates.extend(exclude_files or [])
    normalized: list[str] = []
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            rendered = str(path)
            if rendered not in normalized:
                normalized.append(rendered)
    return normalized


def _format_local_path(value: str | Path, *, is_directory: bool) -> str:
    text = str(value)
    if is_directory:
        return text.rstrip("/") + "/"
    return text


def _format_remote_path(*, ssh_host: str, value: str, is_directory: bool) -> str:
    rendered = value.rstrip("/") + "/" if is_directory else value
    return f"{ssh_host}:{rendered}"
