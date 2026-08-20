"""SSH command rendering and connectivity checks."""

from __future__ import annotations

import shlex
import subprocess
import sys
from typing import Any, Callable

from .ssh_profiles import SshProfile, validate_ssh_options

Runner = Callable[..., subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]]


def build_ssh_command(
    profile: SshProfile,
    *,
    mode: str,
    remote_command: str | None = None,
    allocate_tty: bool = False,
) -> list[str]:
    command = ["ssh", *_build_ssh_option_args(profile, mode=mode)]
    if allocate_tty:
        command.append("-tt")
    command.append(profile.target())
    if remote_command:
        command.append(remote_command)
    return command


def render_ssh_shell(profile: SshProfile, *, mode: str) -> str:
    return " ".join(shlex.quote(part) for part in ["ssh", *_build_ssh_option_args(profile, mode=mode)])


def run_ssh_connectivity_check(
    profile: SshProfile,
    *,
    mode: str,
    remote_command: str = "true",
    batch_runner: Runner = subprocess.run,
    interactive_runner: Runner = subprocess.run,
    interactive_available: bool | None = None,
) -> dict[str, Any]:
    if mode not in {"auto", "batch", "interactive"}:
        raise ValueError(f"Unsupported SSH mode: {mode}")

    if mode == "interactive":
        return _run_interactive_check(profile, remote_command=remote_command, interactive_runner=interactive_runner)

    batch_command = build_ssh_command(profile, mode="batch", remote_command=remote_command, allocate_tty=False)
    batch_result = batch_runner(batch_command, capture_output=True, text=True, check=False)
    stdout = _coerce_text(getattr(batch_result, "stdout", ""))
    stderr = _coerce_text(getattr(batch_result, "stderr", ""))
    report = {
        "profile": profile.name,
        "role": profile.role,
        "target": profile.target(),
        "requested_mode": mode,
        "mode_used": "batch",
        "batch_command": batch_command,
        "ok": batch_result.returncode == 0,
        "returncode": batch_result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "fallback_to_interactive": False,
        "interactive_attempted": False,
    }
    report.update(analyze_ssh_failure(stderr))
    if batch_result.returncode == 0 or mode == "batch":
        return report

    if not report["should_retry_interactive"]:
        return report

    report["fallback_to_interactive"] = True
    interactive_command = build_ssh_command(profile, mode="interactive", remote_command=remote_command, allocate_tty=True)
    report["interactive_command"] = interactive_command
    if interactive_available is None:
        interactive_available = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive_available:
        report["interactive_available"] = False
        return report

    interactive_result = interactive_runner(interactive_command, check=False)
    report["interactive_available"] = True
    report["interactive_attempted"] = True
    report["mode_used"] = "interactive"
    report["ok"] = interactive_result.returncode == 0
    report["returncode"] = interactive_result.returncode
    return report


def analyze_ssh_failure(stderr: str) -> dict[str, Any]:
    normalized = stderr.lower()
    guidance = ""
    should_retry_interactive = False
    failure_type = "unknown"

    if "remote host identification has changed" in normalized or "offending key in" in normalized:
        failure_type = "host_key_mismatch"
        guidance = (
            "Host key mismatch detected. Verify the host key out-of-band, then remove or update the stale "
            "known_hosts entry before retrying."
        )
    elif "host key verification failed" in normalized or "the authenticity of host" in normalized:
        failure_type = "host_key_verification"
        guidance = (
            "Host key acceptance is still required. Retry in interactive mode to accept the host key, "
            "or pre-seed the correct key for automation."
        )
        should_retry_interactive = True
    elif "permission denied" in normalized or "keyboard-interactive" in normalized or "verification code" in normalized:
        failure_type = "authentication"
        guidance = "Interactive authentication or MFA may be required. Retry in interactive mode."
        should_retry_interactive = True
    elif "operation timed out" in normalized or "connection timed out" in normalized:
        failure_type = "timeout"
        guidance = "SSH timed out before authentication completed."
    elif "could not resolve hostname" in normalized:
        failure_type = "dns"
        guidance = "SSH could not resolve the configured hostname."

    return {
        "failure_type": failure_type,
        "host_key_fix_guidance": guidance,
        "should_retry_interactive": should_retry_interactive,
    }


def _run_interactive_check(
    profile: SshProfile,
    *,
    remote_command: str,
    interactive_runner: Runner,
) -> dict[str, Any]:
    interactive_command = build_ssh_command(profile, mode="interactive", remote_command=remote_command, allocate_tty=True)
    result = interactive_runner(interactive_command, check=False)
    return {
        "profile": profile.name,
        "role": profile.role,
        "target": profile.target(),
        "requested_mode": "interactive",
        "mode_used": "interactive",
        "interactive_command": interactive_command,
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "fallback_to_interactive": False,
        "interactive_attempted": True,
        "failure_type": "unknown",
        "host_key_fix_guidance": "",
        "should_retry_interactive": False,
    }


def _build_ssh_option_args(profile: SshProfile, *, mode: str) -> list[str]:
    profile_options = validate_ssh_options(profile.options)
    options = {
        "BatchMode": "yes" if mode == "batch" else "no",
        "StrictHostKeyChecking": "yes" if mode == "batch" else "ask",
    }
    if profile.port is not None:
        port_args = ["-p", str(profile.port)]
    else:
        port_args = []
    if profile.identity_file:
        identity_args = ["-i", profile.identity_file]
    else:
        identity_args = []
    if profile.known_hosts_file:
        options["UserKnownHostsFile"] = profile.known_hosts_file
    options.update(profile_options)

    option_args: list[str] = []
    for key, value in options.items():
        option_args.extend(["-o", f"{key}={value}"])
    return [*port_args, *identity_args, *option_args]


def _coerce_text(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
