from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = REPO_ROOT / "ops" / "envs" / "dev" / "bootstrap.sh"
SYSTEM_BASH = Path("/bin/bash")


def _base_env(tmp_path: Path) -> dict[str, str]:
    return {
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", ""),
        "RP_DEV_VENV": str(tmp_path / "venv"),
    }


def _run_bootstrap(tmp_path: Path, *args: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = _base_env(tmp_path)
    env.update(env_overrides)
    return subprocess.run(
        [str(SYSTEM_BASH), str(BOOTSTRAP), *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _embedded_python(name: str) -> str:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    marker = f"{name}='"
    return source.split(marker, 1)[1].split("'\n", 1)[0]


def _write_fake_runtime(
    venv: Path,
    log_path: Path,
    *,
    include_snakemake: bool = False,
    isolated: bool = True,
) -> None:
    bin_dir = venv / "bin"
    venv.mkdir(parents=True, exist_ok=True)
    if isolated:
        (venv / "pyvenv.cfg").write_text("home = /synthetic/python\n", encoding="utf-8")
    command_script = (
        "#!/usr/bin/env bash\n"
        f"printf '%s|find_links=%s|config=%s|no_index=%s|requirement=%s|target=%s|require_venv=%s|no_cache=%s\\n' \"$(basename -- \"$0\") $*\" "
        f"\"${{PIP_FIND_LINKS-unset}}\" \"${{PIP_CONFIG_FILE-unset}}\" \"${{PIP_NO_INDEX-unset}}\" "
        f"\"${{PIP_REQUIREMENT-unset}}\" \"${{PIP_TARGET-unset}}\" \"${{PIP_REQUIRE_VIRTUALENV-unset}}\" "
        f"\"${{PIP_NO_CACHE_DIR-unset}}\" >> {log_path!s}\n"
    )
    for command_name in ("python", "rp", "research-io", "research-bids", "research-hpc"):
        _write_executable(bin_dir / command_name, command_script)
    if include_snakemake:
        _write_executable(bin_dir / "snakemake", command_script)


def test_bootstrap_help_is_non_mutating(tmp_path: Path) -> None:
    completed = _run_bootstrap(tmp_path, "--help", PYTHON_BIN=str(tmp_path / "missing-python"))

    assert completed.returncode == 0
    assert "--profile PROFILE" in completed.stdout
    assert "--print-plan" in completed.stdout
    assert not (tmp_path / "venv").exists()


@pytest.mark.parametrize("next_option", ["--print-plan", "--help"])
def test_profile_requires_a_non_option_value(tmp_path: Path, next_option: str) -> None:
    completed = _run_bootstrap(tmp_path, "--profile", next_option)

    assert completed.returncode == 2
    assert "--profile requires a value" in completed.stderr
    assert not (tmp_path / "venv").exists()


def test_hpc_dependency_check_uses_current_requirement_metadata(tmp_path: Path) -> None:
    code = _embedded_python("hpc_dependency_check_code")
    compile(code, "<hpc_dependency_check_code>", "exec")
    requirements = tmp_path / "requirements.txt"
    pyproject = tmp_path / "pyproject.toml"
    requirements.write_text("pip\n", encoding="utf-8")
    pyproject.write_text("[project]\nname = \"example\"\ndependencies = []\n", encoding="utf-8")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}

    satisfied = subprocess.run(
        [sys.executable, "-c", code, str(requirements), str(pyproject)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    requirements.write_text("package-that-is-intentionally-absent\n", encoding="utf-8")
    missing = subprocess.run(
        [sys.executable, "-c", code, str(requirements), str(pyproject)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert satisfied.returncode == 0, satisfied.stderr
    assert missing.returncode != 0


def test_omitted_local_profile_defaults_to_minimal(tmp_path: Path) -> None:
    planned_venv = tmp_path / "relative-venv"
    completed = _run_bootstrap(
        tmp_path,
        "--print-plan",
        RP_DEV_VENV=os.path.relpath(planned_venv, REPO_ROOT),
    )

    assert completed.returncode == 0, completed.stderr
    assert "profile: minimal" in completed.stdout
    assert "packages/research-viz" not in completed.stdout
    assert str(planned_venv) in completed.stdout
    assert not planned_venv.exists()


@pytest.mark.parametrize("profile", ["minimal", "dev", "full", "hpc"])
def test_print_plan_is_non_mutating_for_every_profile(tmp_path: Path, profile: str) -> None:
    marker = tmp_path / "python-invoked"
    fake_python = tmp_path / "python"
    _write_executable(fake_python, f"#!/usr/bin/env bash\ntouch {marker!s}\n")

    completed = _run_bootstrap(
        tmp_path,
        "--print-plan",
        "--profile",
        profile,
        PYTHON_BIN=str(fake_python),
    )

    assert completed.returncode == 0, completed.stderr
    assert f"profile: {profile}" in completed.stdout
    assert "sys.version_info" in completed.stdout
    if profile == "hpc":
        assert "research-hpc" in completed.stdout
        assert "snakemake" in completed.stdout
        assert "no installation will be attempted" in completed.stdout
        assert "--upgrade" not in completed.stdout
        assert " -m venv " not in completed.stdout
    else:
        assert "smoke-check.sh" in completed.stdout
        assert "pyvenv.cfg" in completed.stdout
    assert not marker.exists()
    assert not (tmp_path / "venv").exists()


def test_profile_plans_keep_optional_groups_separate(tmp_path: Path) -> None:
    minimal = _run_bootstrap(tmp_path, "--print-plan", "--profile", "minimal").stdout
    dev = _run_bootstrap(tmp_path, "--print-plan", "--profile", "dev").stdout
    full = _run_bootstrap(tmp_path, "--print-plan", "--profile", "full").stdout
    hpc = _run_bootstrap(tmp_path, "--print-plan", "--profile", "hpc").stdout

    assert "packages/research-viz" not in minimal
    assert "pytest>=8" not in minimal
    assert "requirements-notebook.txt" not in minimal
    assert "requirements-runtime.txt" not in minimal
    assert "rsatoolbox" not in minimal
    assert "pytest>=8" in dev
    assert "packages/research-viz" not in dev
    assert "requirements-notebook.txt" not in dev
    assert "pytest>=8" in full
    assert "packages/research-viz" in full
    assert "requirements-notebook.txt" in full
    assert "rsatoolbox" in full
    assert "research-bids[pandas]" in full
    assert "research-io[pandas]" in full
    assert "research-ml[xgboost]" in full
    assert "requirements-runtime.txt" in hpc
    assert "package index: disabled" in hpc
    assert "python -m pip install" not in hpc


def test_hpc_wheelhouse_plan_is_offline_and_uses_find_links(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()

    completed = _run_bootstrap(
        tmp_path,
        "--print-plan",
        "--profile",
        "hpc",
        RP_BOOTSTRAP_WHEELHOUSE=str(wheelhouse),
    )

    assert completed.returncode == 0, completed.stderr
    assert "package index: disabled" in completed.stdout
    assert "--no-index" in completed.stdout
    assert "--find-links" in completed.stdout
    assert str(wheelhouse) in completed.stdout


def test_minimal_plan_uses_a_configured_wheelhouse_without_disabling_the_index(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()

    completed = _run_bootstrap(
        tmp_path,
        "--print-plan",
        "--profile",
        "minimal",
        RP_BOOTSTRAP_WHEELHOUSE=str(wheelhouse),
    )

    assert completed.returncode == 0, completed.stderr
    assert "--find-links" in completed.stdout
    assert str(wheelhouse) in completed.stdout
    assert "--no-index" not in completed.stdout


@pytest.mark.parametrize("configure_wheelhouse", [False, True], ids=["empty", "nonempty"])
def test_system_bash_handles_optional_pip_source_args_in_plan_and_execution(
    tmp_path: Path,
    configure_wheelhouse: bool,
) -> None:
    venv = tmp_path / "venv"
    runtime_log = tmp_path / "runtime.log"
    _write_fake_runtime(venv, runtime_log)
    env: dict[str, str] = {}
    wheelhouse = tmp_path / "wheelhouse"
    if configure_wheelhouse:
        wheelhouse.mkdir()
        env["RP_BOOTSTRAP_WHEELHOUSE"] = str(wheelhouse)

    plan = _run_bootstrap(tmp_path, "--print-plan", "--profile", "minimal", **env)
    completed = _run_bootstrap(tmp_path, "--profile", "minimal", **env)
    pip_calls = [
        line
        for line in runtime_log.read_text(encoding="utf-8").splitlines()
        if "python -m pip" in line
    ]

    assert plan.returncode == 0, plan.stderr
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert pip_calls
    assert "install ''" not in plan.stdout
    assert all("install  " not in call for call in pip_calls)
    if configure_wheelhouse:
        assert "--find-links" in plan.stdout
        assert str(wheelhouse) in plan.stdout
        assert all(f"--find-links {wheelhouse}" in call for call in pip_calls)
    else:
        assert "--find-links" not in plan.stdout
        assert all("--find-links" not in call for call in pip_calls)


def test_hpc_rejects_remote_requirements_before_installing(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example-package @ https://example.org/example-package.whl\n", encoding="utf-8")
    marker = tmp_path / "python-invoked"
    fake_python = tmp_path / "python"
    _write_executable(fake_python, f"#!/usr/bin/env bash\ntouch {marker!s}\n")

    completed = _run_bootstrap(
        tmp_path,
        "--profile",
        "hpc",
        PYTHON_BIN=str(fake_python),
        RP_BOOTSTRAP_WHEELHOUSE=str(wheelhouse),
        RP_DEV_REQUIREMENTS=str(requirements),
    )

    assert completed.returncode == 2
    assert "only offline-resolvable package requirements" in completed.stderr
    assert not marker.exists()
    assert not (tmp_path / "venv").exists()


def test_hpc_rejects_a_late_remote_dependency_after_an_extras_requirement(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    copied_bootstrap = workspace / "ops" / "envs" / "dev" / "bootstrap.sh"
    copied_bootstrap.parent.mkdir(parents=True)
    shutil.copy2(BOOTSTRAP, copied_bootstrap)
    hpc_requirements = workspace / "ops" / "envs" / "hpc" / "requirements-runtime.txt"
    hpc_requirements.parent.mkdir(parents=True)
    hpc_requirements.write_text("snakemake\nsnakemake-executor-plugin-slurm\n", encoding="utf-8")
    package_root = workspace / "packages" / "research-hpc"
    package_root.mkdir(parents=True)
    package_root.joinpath("pyproject.toml").write_text(
        "[build-system]\nrequires = [\"setuptools>=69\", \"wheel\"]\n"
        "[project]\nname = \"research-hpc\"\nversion = \"0.1.0\"\n"
        "dependencies = [\n"
        "  \"safe-package[extra]>=1\",\n"
        "  \"remote-package @ https://example.org/remote-package.whl\",\n"
        "]\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(copied_bootstrap), "--print-plan", "--profile", "hpc"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        env={**_base_env(tmp_path), "RP_DEV_VENV": str(tmp_path / "partial-runtime")},
    )

    assert completed.returncode == 2
    assert "only offline-resolvable package requirements" in completed.stderr
    assert not (tmp_path / "partial-runtime").exists()


def test_slurm_forces_hpc_and_fails_before_install_without_local_source(tmp_path: Path) -> None:
    marker = tmp_path / "python-invoked"
    fake_python = tmp_path / "python"
    _write_executable(fake_python, f"#!/usr/bin/env bash\ntouch {marker!s}\n")

    completed = _run_bootstrap(
        tmp_path,
        PYTHON_BIN=str(fake_python),
        SLURM_JOB_ID="12345",
        CI="true",
        RP_EXECUTION_PROFILE="local",
    )

    assert completed.returncode == 2
    assert "cannot install packages without a local source" in completed.stderr
    assert "No installation was attempted" in completed.stderr
    assert not marker.exists()
    assert not (tmp_path / "venv").exists()


def test_slurm_rejects_an_explicit_non_hpc_profile(tmp_path: Path) -> None:
    completed = _run_bootstrap(tmp_path, "--profile", "minimal", SLURM_CLUSTER_NAME="cluster-a")

    assert completed.returncode == 2
    assert "requires --profile hpc" in completed.stderr
    assert not (tmp_path / "venv").exists()


def test_python_version_preflight_runs_before_environment_creation(tmp_path: Path) -> None:
    fake_python = tmp_path / "old-python"
    _write_executable(fake_python, "#!/usr/bin/env bash\nexit 1\n")

    completed = _run_bootstrap(tmp_path, "--profile", "minimal", PYTHON_BIN=str(fake_python))

    assert completed.returncode == 1
    assert "requires Python 3.11 or newer" in completed.stderr
    assert not (tmp_path / "venv").exists()


def test_bootstrap_refuses_to_modify_a_non_isolated_existing_python(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    marker = tmp_path / "python-invoked"
    _write_executable(venv / "bin" / "python", f"#!/usr/bin/env bash\ntouch {marker!s}\n")

    plan = _run_bootstrap(tmp_path, "--print-plan", "--profile", "minimal")
    completed = _run_bootstrap(tmp_path, "--profile", "minimal")

    assert plan.returncode == 0
    assert "pyvenv.cfg" in plan.stdout
    assert "sys.prefix" in plan.stdout
    assert completed.returncode == 2
    assert "Refusing to modify a non-isolated Python environment" in completed.stderr
    assert not marker.exists()


def test_bootstrap_rejects_a_wrong_prefix_even_with_a_pyvenv_marker(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /synthetic/python\n", encoding="utf-8")
    _write_executable(
        venv / "bin" / "python",
        f"#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} \"$@\"\n",
    )

    completed = _run_bootstrap(tmp_path, "--profile", "minimal")

    assert completed.returncode == 2
    assert "Refusing to modify a non-isolated Python environment" in completed.stderr


def test_bootstrap_reuses_an_existing_virtual_environment(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    runtime_log = tmp_path / "runtime.log"
    system_marker = tmp_path / "system-python-invoked"
    system_python = tmp_path / "system-python"
    _write_fake_runtime(venv, runtime_log)
    _write_executable(system_python, f"#!/usr/bin/env bash\ntouch {system_marker!s}\n")

    completed = _run_bootstrap(
        tmp_path,
        "--profile",
        "minimal",
        PYTHON_BIN=str(system_python),
        PIP_REQUIREMENT="https://example.org/requirements.txt",
        PIP_CONSTRAINT="https://example.org/constraints.txt",
        PIP_TARGET=str(tmp_path / "redirected-target"),
    )
    runtime_calls = runtime_log.read_text(encoding="utf-8")
    pip_calls = [line for line in runtime_calls.splitlines() if "python -m pip" in line]

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Reusing virtual environment" in completed.stdout
    assert not system_marker.exists()
    assert "python -m pip install" in runtime_calls
    assert pip_calls
    assert all("requirement=unset|target=unset|require_venv=1" in line for line in pip_calls)
    assert not (tmp_path / "redirected-target").exists()


def test_hpc_reuses_a_complete_site_managed_runtime_without_installing(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    runtime_log = tmp_path / "runtime.log"
    _write_fake_runtime(venv, runtime_log, include_snakemake=True, isolated=False)

    completed = _run_bootstrap(
        tmp_path,
        "--profile",
        "hpc",
        PIP_FIND_LINKS="https://example.org/packages",
        PIP_INDEX_URL="https://example.org/index",
        PIP_REQUIREMENT="https://example.org/requirements.txt",
    )
    runtime_calls = runtime_log.read_text(encoding="utf-8")
    pip_calls = [line for line in runtime_calls.splitlines() if "python -m pip" in line]

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Reusing usable offline HPC environment" in completed.stdout
    assert not (venv / "pyvenv.cfg").exists()
    assert not pip_calls
    assert "--upgrade" not in runtime_calls
    assert "snakemake --help" in runtime_calls
    assert "snakemake-executor-plugin-slurm" in runtime_calls
    assert "rp --help" in runtime_calls
    assert "research-bids --help" in runtime_calls
    assert "research-io --help" in runtime_calls
    assert "https://example.org" not in runtime_calls


def test_hpc_wheelhouse_execution_scrubs_sources_and_mutates_only_the_virtualenv(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    runtime_log = tmp_path / "runtime.log"
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _write_fake_runtime(venv, runtime_log, include_snakemake=True)

    completed = _run_bootstrap(
        tmp_path,
        "--profile",
        "hpc",
        RP_BOOTSTRAP_WHEELHOUSE=str(wheelhouse),
        PIP_FIND_LINKS="https://example.org/packages",
        PIP_INDEX_URL="https://example.org/index",
        PIP_REQUIREMENT="https://example.org/requirements.txt",
        PIP_TARGET=str(tmp_path / "redirected-target"),
    )
    runtime_calls = runtime_log.read_text(encoding="utf-8")
    pip_calls = [line for line in runtime_calls.splitlines() if "python -m pip" in line]
    mutating_calls = pip_calls

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert mutating_calls
    assert any("--upgrade" in line for line in mutating_calls)
    assert all("--no-index" in line and "--find-links" in line for line in mutating_calls)
    assert all("--no-cache-dir" in line for line in mutating_calls)
    assert all("find_links=unset|config=/dev/null|no_index=1|requirement=unset|target=unset|require_venv=1|no_cache=1" in line for line in mutating_calls)
    assert "https://example.org" not in "\n".join(pip_calls)
    assert not (tmp_path / "redirected-target").exists()


@pytest.mark.parametrize("missing_contract", ["snakemake-executor-plugin-slurm", "direct_url.json"])
def test_hpc_incomplete_runtime_stops_without_mutating(tmp_path: Path, missing_contract: str) -> None:
    venv = tmp_path / "venv"
    runtime_log = tmp_path / "runtime.log"
    _write_fake_runtime(venv, runtime_log, include_snakemake=True, isolated=False)
    python_path = venv / "bin" / "python"
    python_script = python_path.read_text(encoding="utf-8")
    python_path.write_text(
        python_script.replace(
            "#!/usr/bin/env bash\n",
            f"#!/usr/bin/env bash\nif [[ \"$*\" == *{missing_contract}* ]]; then exit 1; fi\n",
            1,
        ),
        encoding="utf-8",
    )

    completed = _run_bootstrap(tmp_path, "--profile", "hpc")
    runtime_calls = runtime_log.read_text(encoding="utf-8")

    assert completed.returncode == 2
    assert "cannot install packages without a local source" in completed.stderr
    assert "--upgrade" not in runtime_calls
    assert "python -m pip" not in runtime_calls


def test_hpc_plan_tracks_only_packages_in_a_partial_staged_checkout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    copied_bootstrap = workspace / "ops" / "envs" / "dev" / "bootstrap.sh"
    copied_bootstrap.parent.mkdir(parents=True)
    shutil.copy2(BOOTSTRAP, copied_bootstrap)
    hpc_requirements = workspace / "ops" / "envs" / "hpc" / "requirements-runtime.txt"
    hpc_requirements.parent.mkdir(parents=True)
    hpc_requirements.write_text("snakemake\nsnakemake-executor-plugin-slurm\n", encoding="utf-8")
    for package_name in ("research-core", "research-hpc"):
        package_root = workspace / "packages" / package_name
        package_root.mkdir(parents=True)
        package_root.joinpath("pyproject.toml").write_text(
            "[build-system]\nrequires = [\"setuptools>=69\", \"wheel\"]\n"
            f"[project]\nname = \"{package_name}\"\nversion = \"0.1.0\"\ndependencies = []\n"
            "[project.urls]\nHomepage = \"https://example.org/project\"\n",
            encoding="utf-8",
        )

    completed = subprocess.run(
        ["bash", str(copied_bootstrap), "--print-plan", "--profile", "hpc"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        env={**_base_env(tmp_path), "RP_DEV_VENV": str(tmp_path / "partial-runtime")},
    )

    assert completed.returncode == 0, completed.stderr
    assert "packages/research-core" in completed.stdout
    assert "packages/research-hpc" in completed.stdout
    assert "research_platform.core" in completed.stdout
    assert "research_platform.hpc" in completed.stdout
    assert "research_platform.analysis" not in completed.stdout


def test_quickstart_commands_match_the_core_parser_without_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "packages" / "research-neuro" / "src"))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "packages" / "research-hpc" / "src"))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "packages" / "research-core" / "src"))
    from research_platform.core.cli import _build_parser

    parser = _build_parser()
    commands = [
        [
            "batch",
            "show",
            "--project",
            "project-pilot-tabular",
            "--batch",
            "toy_binary_logreg",
        ],
        [
            "run",
            "plan",
            "preprocess",
            "tabular",
            "--project",
            "project-pilot-tabular",
            "--batch",
            "toy_binary_logreg",
            "--run-id",
            "quickstart-toy-preprocess",
        ],
        [
            "run",
            "local",
            "preprocess",
            "tabular",
            "--project",
            "project-pilot-tabular",
            "--batch",
            "toy_binary_logreg",
            "--run-id",
            "quickstart-toy-preprocess",
            "--dry-run",
        ],
        [
            "run",
            "local",
            "preprocess",
            "tabular",
            "--project",
            "project-pilot-tabular",
            "--batch",
            "toy_binary_logreg",
            "--run-id",
            "quickstart-toy-preprocess",
            "--execute",
        ],
    ]

    parsed = [parser.parse_args(command) for command in commands]

    assert parsed[0].batch_command == "show"
    assert parsed[1].run_command == "plan"
    assert parsed[2].dry_run is True and parsed[2].execute is False
    assert parsed[3].dry_run is False and parsed[3].execute is True
    assert not (tmp_path / "artifacts").exists()


def test_public_quickstarts_reference_the_implemented_bootstrap_and_smoke_commands() -> None:
    for path in (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "onboarding" / "quickstart.md"):
        text = path.read_text(encoding="utf-8")
        assert "bash ops/envs/dev/bootstrap.sh --profile minimal" in text
        assert "bash ops/envs/dev/smoke-check.sh" in text
        assert "rp batch show" in text
        assert "rp run local preprocess tabular" in text
        assert "--dry-run" in text
        assert "--execute" in text

    quickstart = (REPO_ROOT / "docs" / "onboarding" / "quickstart.md").read_text(encoding="utf-8")
    local_dev = (REPO_ROOT / "docs" / "onboarding" / "local-dev.md").read_text(encoding="utf-8")
    assert 'source "$RP_DEV_VENV/bin/activate"' in quickstart
    assert 'source "$RP_DEV_VENV/bin/activate"' in local_dev
