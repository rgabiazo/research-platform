
"""Explicit reusable CLI for HPC run-manifest and sync operations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Any

from .bootstrap import build_bootstrap_execution_plan, execute_bootstrap_plan
from .manifest import read_status, read_run_manifest
from .remote import (
    build_cancel_plan,
    build_pull_plan,
    build_stage_plan,
    build_submit_plan,
    execute_pull_plan,
    execute_stage_plan,
    execute_submit_plan,
)
from .ssh import build_ssh_command, render_ssh_shell, run_ssh_connectivity_check
from .ssh_profiles import (
    build_ssh_config_template,
    list_ssh_profiles,
    load_ssh_profile,
    materialize_ssh_profile_entry,
    require_generic_profile_isolation,
    resolve_ssh_profile_config_path,
    upsert_ssh_profile_document,
)
from .transfers import build_rsync_pull_command, build_rsync_push_command, run_rsync_command
from ._yaml import dump_yaml, load_yaml, parse_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reusable HPC helpers for planned research-platform runs and scoped sync plans.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("stage", "pull", "cancel", "submit"):
        command = subparsers.add_parser(name)
        command.add_argument("--run-root", required=True)
        if name in {"stage", "pull"}:
            command.add_argument("--workspace-root", required=True)
            command.add_argument("--execute", action="store_true")
            if name == "pull":
                command.add_argument("--subpath", default=None)
                command.add_argument("--destination", default=None)
            _add_optional_profile_args(command)
        elif name == "submit":
            _add_optional_profile_args(command)
            command.add_argument("--execute", action="store_true", help="Explicitly submit the rendered SLURM command.")

    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--run-root", required=True)
    bootstrap.add_argument("--execute", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", required=True)

    ssh = subparsers.add_parser("ssh", help="Named SSH profile helpers.")
    ssh_subparsers = ssh.add_subparsers(dest="ssh_command", required=True)

    ssh_check = ssh_subparsers.add_parser("check", help="Check SSH connectivity with batch-first fallback.")
    _add_profile_args(ssh_check)
    ssh_check.add_argument("--mode", choices=("auto", "batch", "interactive"), default="auto")
    ssh_check.add_argument("--remote-command", default="true")

    ssh_print = ssh_subparsers.add_parser("print", help="Render SSH commands and rsync shell strings.")
    _add_profile_args(ssh_print)
    ssh_print.add_argument("--mode", choices=("batch", "interactive"), default="batch")
    ssh_print.add_argument("--remote-command", default="true")

    ssh_list = ssh_subparsers.add_parser("list", help="List available SSH profiles and roles.")
    ssh_list.add_argument("--config", default=None, help="SSH profile config path.")

    ssh_show = ssh_subparsers.add_parser("show", help="Show the resolved SSH profile for a selected role.")
    _add_profile_args(ssh_show)

    ssh_init = ssh_subparsers.add_parser(
        "init-config",
        help="Write or update a local SSH profile config without contacting a host.",
    )
    ssh_init.add_argument(
        "--template",
        choices=("generic", "alliance"),
        default="generic",
        help="Provider-neutral generic profile by default; Alliance is an explicit site-reviewed integration.",
    )
    ssh_init.add_argument("--output", required=True, help="Where to write the starter config.")
    ssh_init.add_argument("--profile", default=None, help="Profile name. Defaults to generic or alliance.")
    ssh_init.add_argument("--host", default=None, help="Explicit SSH host. Required for the generic template.")
    ssh_init.add_argument("--user", default=None, help="Explicit SSH user. Required for the generic template.")
    ssh_init.add_argument("--port", type=int, default=None, help="Optional SSH port.")
    ssh_init.add_argument("--identity-file", default=None, help="Optional identity-file path reference; contents are never read.")
    ssh_init.add_argument(
        "--known-hosts-file",
        default=None,
        help="Optional known-hosts-file path reference; contents are never read.",
    )
    ssh_init.add_argument(
        "--force",
        action="store_true",
        help="Replace only the selected profile when it already has different settings.",
    )

    rsync = subparsers.add_parser("rsync", help="Named-profile rsync helpers.")
    rsync_subparsers = rsync.add_subparsers(dest="rsync_command", required=True)
    for name in ("push", "pull"):
        command = rsync_subparsers.add_parser(name)
        _add_profile_args(command)
        command.add_argument("--mode", choices=("batch", "interactive"), default="batch")
        command.add_argument("--source", required=True)
        command.add_argument("--destination", required=True)
        command.add_argument("--exclude-file", default=None)
        command.add_argument("--progress", action="store_true")
        command.add_argument("--itemize-changes", action="store_true")
        command.add_argument("--delete", action="store_true")
        authorization = command.add_mutually_exclusive_group()
        authorization.add_argument("--dry-run", action="store_true")
        authorization.add_argument("--execute", action="store_true")

    args = parser.parse_args(argv)
    if args.command in {"stage", "pull", "cancel", "bootstrap", "status", "submit"}:
        report = _handle_run_manifest_command(args)
    elif args.command == "ssh":
        report = _handle_ssh_command(args)
    else:
        report = _handle_rsync_command(args)

    print(json.dumps(report, indent=2))
    return int(report.get("returncode", 0))


def _add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True, help="Named SSH profile.")
    parser.add_argument("--role", choices=("login", "robot"), default="login", help="Named role within a profile family.")
    parser.add_argument("--config", default=None, help="SSH profile config path.")


def _add_optional_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=None, help="Named SSH profile.")
    parser.add_argument("--role", choices=("login", "robot"), default=None, help="Named role within a profile family.")
    parser.add_argument("--config", default=None, help="SSH profile config path.")


def _handle_run_manifest_command(args: argparse.Namespace) -> dict[str, Any]:
    workspace_root = Path(getattr(args, "workspace_root", ".")).resolve()
    run_root = Path(args.run_root).resolve()
    manifest = read_run_manifest(run_root)
    run_status = read_status(run_root)

    try:
        if args.command == "stage":
            report = build_stage_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest=manifest,
                status=run_status,
                exclude_file=workspace_root / "ops" / "sync" / "rsync" / "exclude.txt",
                profile_name=getattr(args, "profile", None),
                role=getattr(args, "role", None),
                config_path=getattr(args, "config", None),
            )["report"]
            if args.execute:
                report["execution"] = execute_stage_plan(report)
                report["returncode"] = report["execution"]["returncode"]
            return report
        if args.command == "pull":
            report = build_pull_plan(
                workspace_root=workspace_root,
                run_root=run_root,
                manifest=manifest,
                status=run_status,
                exclude_file=workspace_root / "ops" / "sync" / "rsync" / "exclude.txt",
                subpath=getattr(args, "subpath", None),
                destination=getattr(args, "destination", None),
                profile_name=getattr(args, "profile", None),
                role=getattr(args, "role", None),
                config_path=getattr(args, "config", None),
            )["report"]
            if args.execute:
                report["execution"] = execute_pull_plan(report)
                report["returncode"] = report["execution"]["returncode"]
            return report
        if args.command == "cancel":
            return build_cancel_plan(manifest=manifest, status=run_status)["report"]
        if args.command == "submit":
            report = build_submit_plan(
                manifest=manifest,
                status=run_status,
                profile_name=getattr(args, "profile", None),
                role=getattr(args, "role", None),
                config_path=getattr(args, "config", None),
                workspace_root=workspace_root,
            )
            report["executed"] = bool(getattr(args, "execute", False))
            if getattr(args, "execute", False):
                report["execution"] = execute_submit_plan(report)
                report["returncode"] = report["execution"]["returncode"]
            return report
        if args.command == "bootstrap":
            report = build_bootstrap_execution_plan(
                run_root=run_root,
                manifest=manifest,
                status=run_status,
                workspace_root=workspace_root,
            )["report"]
            if args.execute:
                report["execution"] = execute_bootstrap_plan(report)
                report["returncode"] = report["execution"]["returncode"]
            return report
        return {
            "run_id": manifest["run_id"],
            "state": run_status.get("state"),
            "last_updated": run_status.get("last_updated"),
            "mode": manifest["execution"]["mode"],
        }
    except ValueError as exc:
        return {"error": str(exc), "returncode": 1}


def _handle_ssh_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.ssh_command == "init-config":
        try:
            template = build_ssh_config_template(
                args.template,
                profile_name=args.profile,
                host=args.host,
                user=args.user,
                port=args.port,
                identity_file=args.identity_file,
                known_hosts_file=args.known_hosts_file,
            )
            profile_name = args.profile or args.template
            output_path = Path(os.path.abspath(Path(args.output).expanduser()))
            current, expected_identity = _read_existing_private_ssh_config(output_path)
            if current is not None:
                if args.template == "generic":
                    require_generic_profile_isolation(current)
                selected_profile = materialize_ssh_profile_entry(template, profile_name=profile_name)
                current_profiles = current.get("profiles")
                selected_exists = isinstance(current_profiles, dict) and profile_name in current_profiles
                existing_profile = (
                    materialize_ssh_profile_entry(current, profile_name=profile_name)
                    if selected_exists
                    else None
                )
                if selected_exists and existing_profile == selected_profile:
                    proposed = current
                else:
                    proposed = upsert_ssh_profile_document(
                        current,
                        profile_name=profile_name,
                        profile=selected_profile,
                        force=bool(args.force),
                    )
            else:
                proposed = template
            _write_private_ssh_config(
                output_path,
                proposed,
                expected_identity=expected_identity,
            )
        except (OSError, ValueError) as exc:
            return {
                "error": str(exc),
                "created": False,
                "returncode": 1,
            }
        return {
            "template": args.template,
            "profile": profile_name,
            "output": str(output_path),
            "created": True,
            "returncode": 0,
        }

    config_path = resolve_ssh_profile_config_path(args.config)
    if args.ssh_command == "list":
        return {
            "config": str(config_path),
            "profiles": list_ssh_profiles(config_path),
            "returncode": 0,
        }

    profile = load_ssh_profile(config_path, args.profile, role=args.role)
    if args.ssh_command == "show":
        return {
            "config": str(config_path),
            "profile": profile.name,
            "role": profile.role,
            "target": profile.target(),
            "resolved": profile.as_dict(),
            "returncode": 0,
        }
    if args.ssh_command == "print":
        return {
            "config": str(config_path),
            "profile": profile.name,
            "role": profile.role,
            "target": profile.target(),
            "mode": args.mode,
            "ssh_shell": render_ssh_shell(profile, mode=args.mode),
            "ssh_command": build_ssh_command(profile, mode=args.mode, remote_command=args.remote_command, allocate_tty=args.mode == "interactive"),
        }
    return run_ssh_connectivity_check(profile, mode=args.mode, remote_command=args.remote_command)


def _handle_rsync_command(args: argparse.Namespace) -> dict[str, Any]:
    config_path = resolve_ssh_profile_config_path(args.config)
    profile = load_ssh_profile(config_path, args.profile, role=args.role)
    if args.rsync_command == "push":
        command = build_rsync_push_command(
            source=args.source,
            profile=profile,
            destination=args.destination,
            mode=args.mode,
            exclude_file=args.exclude_file,
            dry_run=args.dry_run,
            progress=args.progress,
            itemize_changes=args.itemize_changes,
            delete=args.delete,
        )
    else:
        command = build_rsync_pull_command(
            profile=profile,
            source=args.source,
            destination=args.destination,
            mode=args.mode,
            exclude_file=args.exclude_file,
            dry_run=args.dry_run,
            progress=args.progress,
            itemize_changes=args.itemize_changes,
            delete=args.delete,
        )

    report: dict[str, Any] = {
        "config": str(config_path),
        "profile": profile.name,
        "role": profile.role,
        "target": profile.target(),
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "progress": bool(args.progress),
        "itemize_changes": bool(args.itemize_changes),
        "delete": bool(args.delete),
        "command": command,
        "executed": bool(args.execute),
        "returncode": 0,
    }
    if args.execute:
        completed = run_rsync_command(command)
        report["returncode"] = completed.returncode
        report["ok"] = completed.returncode == 0
    return report


def _read_existing_private_ssh_config(
    path: Path,
) -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
    """Read one lower-level private config after rejecting unsafe file types."""

    _validate_private_config_parent_chain(path.parent)
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return None, None
    _require_private_config_file(path, path_stat)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        descriptor_stat = os.fstat(descriptor)
        _require_private_config_file(path, descriptor_stat)
        expected_identity = (path_stat.st_dev, path_stat.st_ino)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != expected_identity:
            raise ValueError(f"SSH profile config destination changed while it was being validated: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        parsed = parse_yaml(b"".join(chunks).decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"SSH profile config must be UTF-8: {path}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("SSH profile config must contain a top-level mapping.")
    return parsed, expected_identity


def _write_private_ssh_config(
    path: Path,
    document: dict[str, Any],
    *,
    expected_identity: tuple[int, int] | None,
) -> None:
    """Write one SSH config with private POSIX modes and bounded checks."""

    payload = dump_yaml(document).encode("utf-8")
    _create_private_parent_directories(path.parent)
    if expected_identity is None:
        _create_private_config_file(path, payload)
        return

    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        descriptor_stat = os.fstat(descriptor)
        _require_private_config_file(path, descriptor_stat)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != expected_identity:
            raise ValueError(f"SSH profile config destination changed before it could be written: {path}")
        current_path_stat = path.lstat()
        if (current_path_stat.st_dev, current_path_stat.st_ino) != expected_identity:
            raise ValueError(f"SSH profile config destination changed before it could be written: {path}")
        _secure_private_file_descriptor(path, descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        current = bytearray()
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            current.extend(chunk)
        if bytes(current) == payload:
            return
        os.lseek(descriptor, 0, os.SEEK_SET)
        _write_all(descriptor, payload)
        os.ftruncate(descriptor, len(payload))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_private_config_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    identity: tuple[int, int] | None = None
    try:
        descriptor_stat = os.fstat(descriptor)
        identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        _require_private_config_file(path, descriptor_stat)
        _secure_private_file_descriptor(path, descriptor)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    except BaseException:
        if identity is not None:
            _unlink_owned_new_file(path, identity=identity)
        raise
    finally:
        os.close(descriptor)


def _create_private_parent_directories(parent: Path) -> None:
    _validate_private_config_parent_chain(parent)
    missing: list[Path] = []
    candidate = parent
    while True:
        try:
            candidate_stat = candidate.lstat()
        except FileNotFoundError:
            missing.append(candidate)
            next_candidate = candidate.parent
            if next_candidate == candidate:
                raise ValueError(f"Cannot locate an existing ancestor for private config directory: {parent}")
            candidate = next_candidate
            continue
        if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISDIR(candidate_stat.st_mode):
            raise ValueError(f"SSH profile config parent must be a real directory, not a symlink: {candidate}")
        break

    for directory in reversed(missing):
        os.mkdir(directory, 0o700)
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(directory, flags)
        try:
            os.fchmod(descriptor, 0o700)
            descriptor_stat = os.fstat(descriptor)
            if not stat.S_ISDIR(descriptor_stat.st_mode) or stat.S_IMODE(descriptor_stat.st_mode) != 0o700:
                raise ValueError(f"New private config directory does not have mode 0700: {directory}")
        finally:
            os.close(descriptor)


def _validate_private_config_parent_chain(parent: Path) -> None:
    """Reject symlinks and non-directories anywhere above a private output."""

    candidate = parent
    while True:
        try:
            candidate_stat = candidate.lstat()
        except FileNotFoundError:
            next_candidate = candidate.parent
            if next_candidate == candidate:
                raise ValueError(
                    f"Cannot locate an existing ancestor for private config directory: {parent}"
                )
            candidate = next_candidate
            continue
        if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISDIR(candidate_stat.st_mode):
            raise ValueError(
                f"SSH profile config parent must be a real directory, not a symlink: {candidate}"
            )
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _require_private_config_file(path: Path, path_stat: os.stat_result) -> None:
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(f"SSH profile config destination must be a regular file, not a symlink: {path}")
    if path_stat.st_nlink != 1:
        raise ValueError(f"SSH profile config destination must not be hard-linked: {path}")


def _secure_private_file_descriptor(path: Path, descriptor: int) -> None:
    os.fchmod(descriptor, 0o600)
    descriptor_stat = os.fstat(descriptor)
    _require_private_config_file(path, descriptor_stat)
    if stat.S_IMODE(descriptor_stat.st_mode) != 0o600:
        raise ValueError(f"SSH profile config destination does not have mode 0600: {path}")


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("Failed to write private SSH profile configuration.")
        offset += written


def _unlink_owned_new_file(path: Path, *, identity: tuple[int, int]) -> None:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == identity:
        path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
