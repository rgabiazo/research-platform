from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "ops" / "scripts" / "detect_execution_profile.sh"


def _run_script(**env_overrides: str) -> str:
    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
    }
    env.update(env_overrides)
    completed = subprocess.run(
        ["sh", str(SCRIPT_PATH)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout.strip()


def test_detect_execution_profile_prefers_explicit_override() -> None:
    assert _run_script(RP_EXECUTION_PROFILE="notebook-local", CI="true", CODEX_HOME="/tmp/codex") == "notebook-local"


def test_detect_execution_profile_detects_ci() -> None:
    assert _run_script(CI="true") == "ci"


def test_detect_execution_profile_detects_codex() -> None:
    assert _run_script(CODEX_HOME="/tmp/codex") == "codex"


def test_detect_execution_profile_defaults_to_local() -> None:
    assert _run_script() == "local"
