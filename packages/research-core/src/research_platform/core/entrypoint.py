"""Lightweight installed entry point for the public ``rp`` command."""

from __future__ import annotations

import sys

from .version import version_report


def main(argv: list[str] | None = None) -> int:
    """Handle version reporting before loading the full orchestration CLI."""

    command_args = list(sys.argv[1:] if argv is None else argv)
    if command_args == ["--version"]:
        print(version_report())
        return 0

    from .cli import main as cli_main

    return cli_main(argv)
