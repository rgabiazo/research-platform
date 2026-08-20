from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CHECK = REPO_ROOT / "ops" / "envs" / "dev" / "smoke-check.sh"


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _run_smoke(venv: Path, *args: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = {
        "HOME": str(venv.parent),
        "PATH": os.environ.get("PATH", ""),
        "RP_DEV_VENV": str(venv),
    }
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(SMOKE_CHECK), *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_install_smoke_runs_the_documented_read_only_command_matrix(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    log_path = tmp_path / "commands.log"
    script = (
        "#!/usr/bin/env bash\n"
        f"printf '%s|%s|%s|%s\\n' \"$(basename -- \"$0\") $*\" "
        f"\"${{PYTHONDONTWRITEBYTECODE:-}}\" \"${{PIP_NO_INDEX:-}}\" \"${{PYTHONNOUSERSITE:-}}\" >> {log_path!s}\n"
    )
    for command_name in ("python", "rp", "research-io", "research-bids", "research-hpc"):
        _write_executable(venv / "bin" / command_name, script)

    completed = _run_smoke(venv, RP_DEV_VENV=os.path.relpath(venv, REPO_ROOT))
    calls = log_path.read_text(encoding="utf-8").splitlines()

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert [line.split("|", 1)[0] for line in calls] == [
        "rp --help",
        "rp --version",
        "rp setup",
        "rp config validate --project project-pilot-tabular",
        "research-io --help",
        "research-bids --help",
        "research-hpc --help",
        "python -m research_platform.analysis.cli --help",
    ]
    assert all(line.endswith("|1|1|1") for line in calls)
    assert "Installation smoke check passed." in completed.stdout
    assert not list(venv.rglob("*.pyc"))
    assert not list(venv.rglob("__pycache__"))
    assert not list(venv.rglob(".pytest_cache"))


def test_install_smoke_fails_cleanly_when_an_entrypoint_is_missing(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    for command_name in ("python", "rp", "research-io", "research-hpc"):
        _write_executable(venv / "bin" / command_name, "#!/usr/bin/env bash\nexit 0\n")

    completed = _run_smoke(venv)

    assert completed.returncode == 1
    assert "could not find research-bids" in completed.stderr


def test_install_smoke_requires_a_non_option_venv_path(tmp_path: Path) -> None:
    completed = _run_smoke(tmp_path / "venv", "--venv", "--quiet")

    assert completed.returncode == 2
    assert "--venv requires a path" in completed.stderr
