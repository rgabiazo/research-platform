"""Rsync command helpers for batch and interactive SSH modes."""

from __future__ import annotations

from pathlib import Path
import subprocess

from .ssh import render_ssh_shell
from .ssh_profiles import SshProfile


def build_rsync_push_command(
    *,
    source: str | Path,
    profile: SshProfile,
    destination: str,
    mode: str,
    exclude_file: str | Path | None = None,
    exclude_files: list[str | Path] | None = None,
    dry_run: bool = False,
    progress: bool = False,
    itemize_changes: bool = False,
    delete: bool = False,
    source_is_directory: bool = True,
) -> list[str]:
    command = _build_rsync_base_command(
        profile=profile,
        mode=mode,
        exclude_file=exclude_file,
        exclude_files=exclude_files,
        dry_run=dry_run,
        progress=progress,
        itemize_changes=itemize_changes,
        delete=delete,
    )
    command.extend(
        [
            _format_local_path(source, is_directory=source_is_directory),
            _format_remote_path(profile=profile, value=destination, is_directory=source_is_directory),
        ]
    )
    return command


def build_rsync_pull_command(
    *,
    profile: SshProfile,
    source: str,
    destination: str | Path,
    mode: str,
    exclude_file: str | Path | None = None,
    exclude_files: list[str | Path] | None = None,
    dry_run: bool = False,
    progress: bool = False,
    itemize_changes: bool = False,
    delete: bool = False,
    source_is_directory: bool = True,
) -> list[str]:
    command = _build_rsync_base_command(
        profile=profile,
        mode=mode,
        exclude_file=exclude_file,
        exclude_files=exclude_files,
        dry_run=dry_run,
        progress=progress,
        itemize_changes=itemize_changes,
        delete=delete,
    )
    command.extend(
        [
            _format_remote_path(profile=profile, value=source, is_directory=source_is_directory),
            _format_local_path(destination, is_directory=source_is_directory),
        ]
    )
    return command


def run_rsync_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True)


def _build_rsync_base_command(
    *,
    profile: SshProfile,
    mode: str,
    exclude_file: str | Path | None,
    exclude_files: list[str | Path] | None,
    dry_run: bool,
    progress: bool,
    itemize_changes: bool,
    delete: bool,
) -> list[str]:
    if mode not in {"batch", "interactive"}:
        raise ValueError(f"Unsupported rsync mode: {mode}")
    command = ["rsync", "-az", "-e", render_ssh_shell(profile, mode=mode)]
    if dry_run:
        command.append("--dry-run")
    if progress:
        command.append("--progress")
    if itemize_changes:
        command.append("--itemize-changes")
    if delete:
        command.append("--delete")
    for candidate in _normalize_exclude_files(exclude_file=exclude_file, exclude_files=exclude_files):
        command.extend(["--exclude-from", candidate])
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
    rendered = str(value)
    if is_directory:
        return rendered.rstrip("/") + "/"
    return rendered


def _format_remote_path(*, profile: SshProfile, value: str, is_directory: bool) -> str:
    rendered = value.rstrip("/") + "/" if is_directory else value
    return f"{profile.target()}:{rendered}"
