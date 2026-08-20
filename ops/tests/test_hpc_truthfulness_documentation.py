from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

# Public-contract tests run from source checkouts as well as editable installs.
for _source_root in sorted((REPO_ROOT / "packages").glob("research-*/src")):
    sys.path.insert(0, str(_source_root))

from research_platform.core import cli as core_cli
from research_platform.hpc import cli as hpc_cli
from research_platform.hpc._yaml import parse_yaml
from research_platform.hpc.offline_validation import validate_hpc_configuration


DOCUMENTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "byod.md",
    REPO_ROOT / "docs" / "capabilities.md",
    REPO_ROOT / "docs" / "bids-hpc-slice.md",
    REPO_ROOT / "docs" / "bids-analysis-slice.md",
    REPO_ROOT / "docs" / "tabular-slice.md",
    REPO_ROOT / "docs" / "mvpa-crossnobis-command-runbook.md",
    REPO_ROOT / "docs" / "how-to" / "hpc-troubleshooting.md",
    REPO_ROOT / "docs" / "how-to" / "run-deepprep-on-slurm.md",
    REPO_ROOT / "docs" / "how-to" / "run-feat-first-level-on-slurm.md",
    REPO_ROOT / "packages" / "research-hpc" / "README.md",
    REPO_ROOT / "secrets" / "README.md",
    REPO_ROOT / "ops" / "envs" / "hpc" / "README.md",
    REPO_ROOT / "ops" / "sync" / "ssh" / "README.md",
)
BEGINNER_HPC_GUIDES = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "bids-hpc-slice.md",
    REPO_ROOT / "docs" / "how-to" / "run-deepprep-on-slurm.md",
    REPO_ROOT / "docs" / "how-to" / "run-feat-first-level-on-slurm.md",
    REPO_ROOT / "docs" / "mvpa-crossnobis-command-runbook.md",
    REPO_ROOT / "packages" / "research-hpc" / "README.md",
    REPO_ROOT / "secrets" / "README.md",
    REPO_ROOT / "ops" / "sync" / "ssh" / "README.md",
)
SSH_CONFIG_EXAMPLE = REPO_ROOT / "ops" / "sync" / "ssh" / "config.example"
TARGETS_EXAMPLE = REPO_ROOT / "ops" / "sync" / "ssh" / "targets.example.yaml"
LINK_PATTERN = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
FENCE_PATTERN = re.compile(r"^```([^\n]*)\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).split())


def _subparser(*names: str) -> argparse.ArgumentParser:
    parser = core_cli._build_parser()
    for name in names:
        action = next(
            item
            for item in parser._actions
            if isinstance(item, argparse._SubParsersAction)
        )
        parser = action.choices[name]
    return parser


def _help(*names: str) -> str:
    normalized = " ".join(_subparser(*names).format_help().split())
    # argparse may wrap immediately after a hyphen, producing text such as
    # ``not- found`` even though the configured help says ``not-found``.
    return re.sub(r"(?<=\w)-\s+(?=\w)", "-", normalized)


def _assert_local_links_resolve(document: Path) -> None:
    for raw_target in LINK_PATTERN.findall(_text(document)):
        target = raw_target.strip().strip("<>")
        path_text = target.split("#", 1)[0]
        if not path_text or "://" in path_text or path_text.startswith("mailto:"):
            continue
        resolved = (document.parent / path_text).resolve()
        resolved.relative_to(REPO_ROOT)
        assert resolved.is_file(), f"Broken link in {document.relative_to(REPO_ROOT)}: {raw_target}"


def _replace_shell_placeholder(token: str) -> str:
    if (
        "$" in token
        or "<" in token
        or ">" in token
        or token.startswith("{")
        or token.endswith("}")
    ):
        return "example"
    return token


def _required_command_tokens(tokens: list[str]) -> list[str]:
    """Drop documentation-only ``[optional arguments]`` from a command."""

    required: list[str] = []
    inside_optional_group = False
    for token in tokens:
        if inside_optional_group:
            if token.endswith("]"):
                inside_optional_group = False
            continue
        if token.startswith("["):
            inside_optional_group = not token.endswith("]")
            continue
        required.append(token)
    if inside_optional_group:
        raise AssertionError("Unclosed illustrative optional-argument group")
    return required


def _documented_cli_commands(document: Path, executable: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for info, body in FENCE_PATTERN.findall(_text(document)):
        if info.strip().casefold() not in {"bash", "sh", "shell"}:
            continue
        logical = body.replace("\\\n", " ")
        for line in logical.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                tokens = shlex.split(stripped)
            except ValueError as exc:
                raise AssertionError(
                    f"Invalid shell syntax in {document.relative_to(REPO_ROOT)}: {line}"
                ) from exc
            if executable not in tokens:
                continue
            executable_index = tokens.index(executable)
            argv: list[str] = []
            for token in _required_command_tokens(tokens[executable_index + 1 :]):
                if token in {"|", "||", "&&", ";"}:
                    break
                argv.append(_replace_shell_placeholder(token))
            commands.append(argv)
    return commands


def _documented_rp_commands(document: Path) -> list[list[str]]:
    return _documented_cli_commands(document, "rp")


class _ResearchHpcCommandParsed(Exception):
    """Stop low-level CLI dispatch immediately after successful parsing."""


def _parse_research_hpc_without_execution(command: list[str]) -> argparse.Namespace:
    original_parse_args = argparse.ArgumentParser.parse_args
    captured: dict[str, argparse.Namespace] = {}

    def capture_parse(
        parser: argparse.ArgumentParser,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = original_parse_args(parser, args, namespace)
        captured["namespace"] = parsed
        raise _ResearchHpcCommandParsed

    with mock.patch.object(argparse.ArgumentParser, "parse_args", new=capture_parse):
        with pytest.raises(_ResearchHpcCommandParsed):
            hpc_cli.main(command)
    return captured["namespace"]


def _assert_shell_fences_parse(document: Path) -> None:
    for info, body in FENCE_PATTERN.findall(_text(document)):
        if info.strip().casefold() not in {"bash", "sh", "shell"}:
            continue
        normalized = re.sub(r"<[A-Za-z0-9_.:/-]+>", "example", body)
        completed = subprocess.run(
            ["bash", "-n"],
            input=normalized,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, (
            f"Invalid shell fence in {document.relative_to(REPO_ROOT)}:\n"
            f"{completed.stderr}"
        )


def test_documented_command_parser_omits_illustrative_optional_groups() -> None:
    synopsis = shlex.split(
        "hpc verify data --project <project> "
        "[--batch <batch>] [--profile <profile>] [--role <role>]"
    )
    assert _required_command_tokens(synopsis) == [
        "hpc",
        "verify",
        "data",
        "--project",
        "<project>",
    ]


def test_hpc_help_exposes_local_remote_and_accounting_boundaries() -> None:
    setup = _help("hpc", "setup")
    assert "Canonical beginner setup" in setup
    assert "provider-neutral" in setup
    assert "generic" in setup
    assert "Alliance" in setup and "explicit" in setup
    assert "under secrets/" in setup
    assert "makes no network call" in setup
    assert "rp hpc validate" in setup

    validate = _help("hpc", "validate")
    assert "offline" in validate
    assert "subprocess" in validate
    assert "write" in validate
    assert "network" in validate
    assert "unverified" in validate

    legacy_init = _help("hpc", "init")
    assert "Legacy/backward-compatible Alliance-oriented" in legacy_init
    assert "rp hpc setup" in legacy_init
    assert "no connectivity" in legacy_init

    doctor = _help("hpc", "doctor")
    assert "immediately check connectivity" in doctor
    assert "configured host" in doctor
    assert "not a local-only validation" in doctor
    assert "authentication, or MFA" in doctor

    verify_data = _help("hpc", "verify", "data")
    assert "immediately contact" in verify_data
    assert "over SSH" in verify_data
    assert "read-only remote check" in verify_data

    status = _help("hpc", "status")
    assert "recorded local manifest/status state without a subprocess" in status
    assert "immediately contact" in status
    assert "one squeue query" in status
    assert "does not query sacct" in status
    assert "not-found-or-completed" in status

    cancel = _help("hpc", "cancel")
    assert "cancel-requested" in cancel
    assert "render a possible scancel command" in cancel
    assert "invokes no SSH or scheduler subprocess" in cancel
    assert "does not confirm remote cancellation" in cancel
    assert "--execute" not in cancel
    with pytest.raises(SystemExit) as exc_info:
        core_cli._build_parser().parse_args(
            ["hpc", "cancel", "--run-id", "example", "--execute"]
        )
    assert exc_info.value.code == 2

    pull = _help("hpc", "pull")
    assert "merge-oriented rsync -az" in pull
    assert "does not prove scheduler success" in pull
    assert "atomically publish" in pull
    assert "attest digests" in pull
    assert "interrupted-transfer recovery" in pull


def test_beginner_entry_point_and_legacy_init_are_labeled_truthfully() -> None:
    for document in BEGINNER_HPC_GUIDES:
        text = _text(document)
        assert "rp hpc setup" in text, document
        assert "rp hpc validate" in text, document
        assert "rp hpc doctor" in text, document
        assert (
            text.index("rp hpc setup")
            < text.index("rp hpc validate")
            < text.index("rp hpc doctor")
        ), document
        assert "--cluster" not in text, document
        assert "starter is currently Alliance/MFA-oriented" not in text, document
        assert "provider-neutral" in text.casefold(), document
        assert "alliance" in text.casefold() and "optional" in text.casefold(), document

    completed = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for relative in completed.stdout.splitlines():
        if relative.startswith("docs/decisions/"):
            continue
        path = REPO_ROOT / relative
        text = _text(path)
        for match in re.finditer(r"rp hpc init", text):
            context = text[max(0, match.start() - 500) : match.end() + 500].casefold()
            assert any(
                label in context
                for label in (
                    "legacy",
                    "backward-compatible",
                    "alliance-oriented",
                    "historical",
                )
            ), f"Unlabelled rp hpc init reference in {relative}"


def test_capability_and_byod_guides_separate_local_and_ssh_active_surfaces() -> None:
    capabilities = _normalized(REPO_ROOT / "docs" / "capabilities.md")
    byod = _normalized(REPO_ROOT / "docs" / "byod.md")

    for surface in (
        "Local HPC setup, target inspection, and plan rendering",
        "Local recorded HPC status",
        "Local cancellation-request rendering",
    ):
        assert f"| {surface} | Plan/validation only |" in capabilities
    for surface in (
        "SSH doctor and remote-data verification",
        "Live scheduler status",
        "Explicit HPC transfer, bootstrap, submission, and retrieval",
    ):
        assert f"| {surface} | Experimental or external-runtime |" in capabilities

    assert "local ROI, bundle, and MVPA doctor commands do not contact remote systems" in byod
    assert "rp hpc doctor" in byod and "immediately checks SSH connectivity" in byod
    assert "rp hpc verify data" in byod and "immediately contacts" in byod
    assert "rp hpc status --live" in byod and "immediately uses SSH" in byod
    assert "status` without `--live` reads only recorded local" in byod
    assert "have not been live-cluster validated" in byod


def test_status_cancellation_and_retrieval_limitations_are_explicit() -> None:
    troubleshooting = _normalized(REPO_ROOT / "docs" / "how-to" / "hpc-troubleshooting.md")
    combined = " ".join(_normalized(document) for document in DOCUMENTS)

    assert "not-found-or-completed" in troubleshooting
    assert "intentionally ambiguous" in troubleshooting
    assert "not evidence of successful completion" in troubleshooting
    assert "checked" in troubleshooting and "ok" in troubleshooting
    assert "does not run `sacct`" in troubleshooting
    assert "separate advanced operator command" in troubleshooting

    assert "cancel-requested" in combined
    assert "does not run `scancel`" in combined
    assert "not proof of remote cancellation" in combined

    for limitation in (
        "merge-oriented `rsync -az`",
        "does not prove that the scheduler job succeeded",
        "not atomically",
        "digest",
        "interrupted-transfer recovery",
    ):
        assert limitation in combined


def test_provider_adaptation_checklist_and_optional_examples_are_bounded() -> None:
    environment_guide = _normalized(REPO_ROOT / "ops" / "envs" / "hpc" / "README.md")
    for item in (
        "SSH host/profile",
        "authentication",
        "account and partition",
        "remote workspace",
        "artifact",
        "container",
        "temporary",
        "scratch",
        "module commands",
        "Python version",
        "virtual environment",
        "offline wheelhouse",
        "scheduler commands",
        "accounting",
        "container runtime",
        "outbound-network restrictions",
        "storage quotas",
        "data-governance",
    ):
        assert item.casefold() in environment_guide.casefold(), item
    assert "site review" in environment_guide.casefold()
    assert "not live-cluster validation" in environment_guide.casefold()

    combined = "\n".join(_text(document) for document in DOCUMENTS)
    assert "Alliance/MFA" in combined
    assert "optional provider integration" in combined.casefold()
    assert "provider-validated" not in combined.casefold()
    assert "Alliance has been live validated" not in combined
    assert "Nibi has been live validated" not in combined


def test_h1_generic_examples_are_provider_neutral_and_fail_closed_until_edited() -> None:
    ssh = _text(SSH_CONFIG_EXAMPLE)
    targets = _text(TARGETS_EXAMPLE)
    combined = f"{ssh}\n{targets}"

    assert "profiles:" in ssh
    assert "host:" in ssh and "user:" in ssh
    assert "<replace-with-real-ssh-host>" in ssh
    assert "<replace-with-real-ssh-user>" in ssh
    assert "identity_file:" not in ssh

    assert "version: 1" in targets
    assert "ssh_profile:" in targets
    assert "role: login" in targets
    assert "RP_REMOTE_WORKSPACE_ROOT:" in targets
    assert "RP_REMOTE_ARTIFACTS_ROOT:" in targets
    assert "mode: atomic_no_replace" in targets
    assert "Replace every placeholder" in targets

    for forbidden in (
        "ALLIANCE_USER",
        "Nibi",
        "ControlMaster",
        "ControlPath",
        "ControlPersist",
        "robot:",
        "RP_REMOTE_CONTAINER_ROOT",
        "account:",
        "partition:",
        "$SCRATCH",
        "/local/scratch",
        "module:",
        "apptainer",
    ):
        assert forbidden.casefold() not in combined.casefold(), forbidden

    report = validate_hpc_configuration(
        workspace_root="/private/tmp/synthetic-h1-example-workspace",
        targets_config_path="/private/tmp/synthetic-h1-example-workspace/secrets/hpc/targets.yaml",
        targets_document=parse_yaml(targets),
        ssh_document=parse_yaml(ssh),
        environment={},
    )
    assert report["configuration_valid"] is False
    assert any("placeholder" in error.casefold() for error in report["errors"])


def test_h1_docs_bound_offline_validation_and_promotion_truthfully() -> None:
    combined = " ".join(_normalized(document) for document in BEGINNER_HPC_GUIDES)

    for requirement in (
        "subprocess-free",
        "write-free",
        "network-free",
        "atomic_no_replace",
        "declared, not remotely verified",
        "host reachability",
        "scheduler",
        "runtime",
        "data readiness",
    ):
        assert requirement.casefold() in combined.casefold(), requirement

    capabilities = _normalized(REPO_ROOT / "docs" / "capabilities.md")
    assert "| Local HPC setup, target inspection, and plan rendering | Plan/validation only |" in capabilities
    assert "`rp hpc validate`" in capabilities
    assert "no provider has been live validated" in capabilities.casefold()


def test_h1_docs_distinguish_setup_surfaces_and_current_local_limits() -> None:
    high_and_low_level_guides = "\n".join(
        _normalized(document)
        for document in (
            REPO_ROOT / "packages" / "research-hpc" / "README.md",
            REPO_ROOT / "secrets" / "README.md",
            REPO_ROOT / "ops" / "sync" / "ssh" / "README.md",
        )
    )
    normalized = high_and_low_level_guides.casefold()

    assert "high-level `rp hpc setup`" in high_and_low_level_guides
    assert "not confined to `secrets/`" in high_and_low_level_guides
    assert "fails before content mutation" in normalized
    assert "hard-linked destinations" in normalized
    assert "public scaffold exceptions" in normalized
    assert "mode `0600`" in normalized
    assert "never reads identity" in normalized
    assert "path existence or regular-file type" in normalized
    assert "profile defaults to the target" in normalized
    assert "role defaults to `login`" in normalized
    assert "no role assumption" not in normalized

    for runbook in (
        REPO_ROOT / "docs" / "how-to" / "run-deepprep-on-slurm.md",
        REPO_ROOT / "docs" / "how-to" / "run-feat-first-level-on-slurm.md",
    ):
        assert '--remote-container-root "$RP_REMOTE_CONTAINER_ROOT"' in _text(runbook)


def test_changed_document_links_shell_and_cli_commands_are_valid() -> None:
    parsed: list[tuple[Path, list[str]]] = []
    parsed_low_level: list[tuple[Path, list[str]]] = []
    for document in DOCUMENTS:
        _assert_local_links_resolve(document)
        _assert_shell_fences_parse(document)
        for command in _documented_rp_commands(document):
            if command in (["--version"], ["--help"]) or core_cli._literal_project_init_name(command) is not None:
                parsed.append((document, command))
                continue
            try:
                core_cli._build_parser().parse_args(command)
            except SystemExit as exc:
                if exc.code != 0:
                    raise AssertionError(
                        f"Parser-invalid command in {document.relative_to(REPO_ROOT)}: rp {shlex.join(command)}"
                    ) from exc
            parsed.append((document, command))
        for command in _documented_cli_commands(document, "research-hpc"):
            try:
                _parse_research_hpc_without_execution(command)
            except SystemExit as exc:
                if exc.code != 0:
                    raise AssertionError(
                        "Parser-invalid command in "
                        f"{document.relative_to(REPO_ROOT)}: "
                        f"research-hpc {shlex.join(command)}"
                    ) from exc
            parsed_low_level.append((document, command))

    assert parsed
    assert parsed_low_level
    command_names = {tuple(command[:3]) for _, command in parsed}
    assert ("hpc", "setup") in {tuple(command[:2]) for _, command in parsed}
    assert ("hpc", "validate") in {tuple(command[:2]) for _, command in parsed}
    assert ("hpc", "doctor") in {tuple(command[:2]) for _, command in parsed}
    assert ("hpc", "verify", "data") in command_names
    assert ("hpc", "status") in {tuple(command[:2]) for _, command in parsed}
    assert ("hpc", "pull") in {tuple(command[:2]) for _, command in parsed}
    low_level_names = {tuple(command[:2]) for _, command in parsed_low_level}
    assert ("ssh", "init-config") in low_level_names
    assert ("ssh", "check") in low_level_names
