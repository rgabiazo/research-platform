"""Static contracts for the least-privilege public-alpha CI workflow."""

from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CI_README = REPO_ROOT / "ops" / "ci" / "README.md"

CHECKOUT_PIN = "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"
SETUP_PYTHON_PIN = (
    "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405"
)
EXPECTED_JOBS = (
    "linux-minimal",
    "linux-package-tests",
    "macos-arm-smoke",
    "public-contracts",
    "release-archives",
)
CI_SCRIPTS = (
    "run-package-tests.sh",
    "run-public-contracts.sh",
    "build-release-archives.sh",
    "inspect-release-archives.py",
    "source-tree-integrity.py",
    "check-clean-checkout.sh",
)
SOURCE_COPY_JOBS = (
    "linux-minimal",
    "linux-package-tests",
    "macos-arm-smoke",
    "public-contracts",
)
PACKAGE_TEST_PATHS = (
    "packages/research-analysis/tests",
    "packages/research-bids/tests",
    "packages/research-core/tests",
    "packages/research-hpc/tests",
    "packages/research-io/tests",
    "packages/research-ml/tests",
    "packages/research-neuro/tests",
    "packages/research-viz/tests",
)
FIXTURE_CHECKS = (
    "generate_toy_memory_fixtures.py --check",
    "generate_toy_tabular_fixtures.py --check",
    "generate_toy_roi_fixtures.py --check",
    "generate_toy_mvpa_fixtures.py --check",
)
PUBLIC_OVERLAYS = (
    "project-template",
    "project-example",
    "project-pilot-bids",
    "project-pilot-tabular",
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _indented_block(text: str, heading: str, *, indent: int = 0) -> str:
    """Return one simple YAML mapping block without requiring a YAML package."""

    prefix = " " * indent
    lines = text.splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line == f"{prefix}{heading}:"
    )
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line and len(line) - len(line.lstrip(" ")) <= indent:
            break
        body.append(line)
    return "\n".join(body)


def _job_blocks(text: str) -> dict[str, str]:
    jobs = _indented_block(text, "jobs")
    lines = jobs.splitlines()
    starts = [
        (index, match.group(1))
        for index, line in enumerate(lines)
        if (match := re.fullmatch(r"  ([a-z0-9-]+):", line))
    ]
    blocks: dict[str, str] = {}
    for position, (start, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        blocks[name] = "\n".join(lines[start:end])
    return blocks


def _multiline_run_bodies(text: str) -> tuple[str, ...]:
    lines = text.splitlines()
    bodies: list[str] = []
    for index, line in enumerate(lines):
        match = re.fullmatch(r"(\s*)run: \|", line)
        if not match:
            continue
        indent = len(match.group(1))
        body: list[str] = []
        for candidate in lines[index + 1 :]:
            candidate_indent = len(candidate) - len(candidate.lstrip(" "))
            if candidate and candidate_indent <= indent:
                break
            body.append(candidate)
        bodies.append("\n".join(body))
    return tuple(bodies)


def test_workflow_has_only_the_allowed_events_permissions_and_concurrency() -> None:
    text = _workflow_text()

    assert re.search(r"(?m)^name: CI$", text)
    event_block = _indented_block(text, "on")
    assert set(re.findall(r"(?m)^  ([a-z_]+):", event_block)) == {
        "push",
        "pull_request",
        "workflow_dispatch",
    }
    assert _indented_block(text, "permissions").strip() == "contents: read"
    assert len(re.findall(r"(?m)^permissions:$", text)) == 1
    concurrency = _indented_block(text, "concurrency")
    assert "group: ${{ github.workflow }}-${{ github.ref }}" in concurrency
    assert "cancel-in-progress: true" in concurrency


def test_every_inline_shell_block_uses_strict_failure_handling() -> None:
    bodies = _multiline_run_bodies(_workflow_text())

    assert bodies
    for body in bodies:
        first_command = next(line.strip() for line in body.splitlines() if line.strip())
        assert first_command == "set -euo pipefail", body


def test_workflow_uses_only_reviewed_immutable_official_actions() -> None:
    text = _workflow_text()
    uses = re.findall(r"(?m)^\s*-?\s*uses:\s*([^\s#]+)", text)

    assert uses
    assert set(uses) == {CHECKOUT_PIN, SETUP_PYTHON_PIN}
    assert re.search(
        rf"uses: {re.escape(CHECKOUT_PIN)}\s+# v6\.0\.2\n"
        r"(?:\s+.*\n)*?\s+persist-credentials: false",
        text,
    )
    assert f"{SETUP_PYTHON_PIN} # v6.2.0" in text
    assert not re.search(r"actions/(?:cache|upload-artifact|download-artifact)@", text)


def test_jobs_cover_supported_platforms_and_have_finite_boundaries() -> None:
    blocks = _job_blocks(_workflow_text())

    assert tuple(blocks) == EXPECTED_JOBS
    for name, block in blocks.items():
        assert re.search(r"(?m)^    timeout-minutes: [1-9][0-9]*$", block), name
        assert CHECKOUT_PIN in block, name
        assert SETUP_PYTHON_PIN in block, name
        assert "bash ops/ci/check-clean-checkout.sh" in block, name
        assert re.search(
            r"(?m)^      - name: Prove the checkout stayed clean\n"
            r"        if: \$\{\{ always\(\) \}\}$",
            block,
        ), name
        assert block.rstrip().endswith("bash ops/ci/check-clean-checkout.sh"), name

    linux_minimal = blocks["linux-minimal"]
    linux_tests = blocks["linux-package-tests"]
    for block in (linux_minimal, linux_tests):
        assert "runs-on: ubuntu-24.04" in block
        assert "fail-fast: false" in block
        assert re.search(
            r"python-version:\s*\[\s*[\"']3\.11[\"']\s*,\s*[\"']3\.12[\"']\s*\]",
            block,
        )

    macos = blocks["macos-arm-smoke"]
    assert "runs-on: macos-15" in macos
    assert re.search(r"python-version:\s*[\"']3\.12[\"']", macos)
    assert "platform.machine()" in macos
    assert "arm64" in macos.lower()

    for name in ("public-contracts", "release-archives"):
        assert "runs-on: ubuntu-24.04" in blocks[name]
        assert re.search(r"python-version:\s*[\"']3\.12[\"']", blocks[name])


def test_smoke_jobs_prove_installation_identity_and_isolation() -> None:
    blocks = _job_blocks(_workflow_text())

    for name in ("linux-minimal", "macos-arm-smoke"):
        block = blocks[name]
        assert "RP_DEV_VENV" in block
        assert "sys.prefix" in block and "sys.base_prefix" in block
        assert "RUNNER_TEMP" in block
        assert "site.ENABLE_USER_SITE is not False" in block
        assert '${GITHUB_WORKSPACE}/.venv' in block
        assert '"${RP_DEV_VENV}/bin/python" -m pip check' in block
        assert "ops/envs/dev/smoke-check.sh --venv" in block
        assert '"${RP_DEV_VENV}/bin/rp" --version' in block
        assert 'research-platform 0.1.0a1' in block

    linux = blocks["linux-minimal"]
    assert "platform.platform()" in linux
    assert "platform.machine()" in linux
    assert "platform.python_version()" in linux


def test_macos_runs_standard_library_hpc_safety_contracts_offline() -> None:
    macos = _job_blocks(_workflow_text())["macos-arm-smoke"]
    step = "Run standard-library HPC safety contracts"

    assert step in macos
    assert macos.index("Disable package-index access") < macos.index(step)
    assert macos.index("Capture the tested source-copy baseline") < macos.index(
        step
    )
    assert macos.index(step) < macos.index("Verify isolation and installation")
    safety = macos[
        macos.index(step) : macos.index("Verify isolation and installation")
    ]
    assert 'cd "${CI_WORKSPACE}"' in safety
    assert 'TMPDIR="${RUNNER_TEMP}"' in safety
    assert '"${RP_DEV_VENV}/bin/python" -m unittest discover' in safety
    assert "-s packages/research-hpc/tests/unit" in safety
    assert "-p 'test_safety_*.py'" in safety
    for forbidden in (
        "pip install",
        "continue-on-error",
        "actions/cache",
        "upload-artifact",
        "download-artifact",
    ):
        assert forbidden not in safety


def test_runner_local_paths_are_initialized_through_github_env() -> None:
    text = _workflow_text()
    blocks = _job_blocks(text)

    assert "${{ runner.temp }}" not in text
    for name in SOURCE_COPY_JOBS:
        block = blocks[name]
        initialization = block.index("Initialize runner-local paths")
        assert initialization < block.index("Copy the complete source checkout")
        assert "${RUNNER_TEMP}" in block
        assert '>> "${GITHUB_ENV}"' in block
        for variable in ("CI_WORKSPACE", "RP_DEV_VENV", "CI_SOURCE_BASELINE"):
            assert f"{variable}=%s" in block, (name, variable)
        assert "GITHUB_RUN_ID" in block and "GITHUB_RUN_ATTEMPT" in block

    release = blocks["release-archives"]
    assert release.index("Initialize runner-local paths") < release.index(
        "Install the CI-only build-tool closure"
    )
    assert "${RUNNER_TEMP}" in release
    assert "RP_RELEASE_DIST_DIR=%s" in release
    assert '>> "${GITHUB_ENV}"' in release
    assert "GITHUB_RUN_ID" in release and "GITHUB_RUN_ATTEMPT" in release


def test_tested_source_copy_and_original_checkout_are_both_verified() -> None:
    blocks = _job_blocks(_workflow_text())
    validation_steps = {
        "linux-minimal": "Verify virtual-environment isolation",
        "linux-package-tests": "Run each package test directory separately",
        "macos-arm-smoke": "Verify isolation and installation",
        "public-contracts": "Run public contracts, fixtures, and overlay validation",
    }

    for name in SOURCE_COPY_JOBS:
        block = blocks[name]
        capture = block.index("Capture the tested source-copy baseline")
        validation = block.index(validation_steps[name])
        verify = block.index("Prove the tested source copy stayed unchanged")
        checkout = block.index("Prove the checkout stayed clean")
        assert block.index("Disable package-index access") < capture < validation
        assert validation < verify < checkout
        assert "source-tree-integrity.py capture" in block
        assert '--root "${CI_WORKSPACE}"' in block
        assert '--output "${CI_SOURCE_BASELINE}"' in block
        assert "source-tree-integrity.py verify" in block
        assert '--baseline "${CI_SOURCE_BASELINE}"' in block
        assert "Source-integrity baseline was not created" in block
        assert len(re.findall(r"if: \$\{\{ always\(\) \}\}", block)) == 2
        assert "bash ops/ci/check-clean-checkout.sh" in block


def test_ci_environment_and_network_boundary_are_explicit() -> None:
    text = _workflow_text()
    for variable in (
        "PYTHONDONTWRITEBYTECODE: \"1\"",
        "PYTHONNOUSERSITE: \"1\"",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD: \"1\"",
        "PIP_DISABLE_PIP_VERSION_CHECK: \"1\"",
    ):
        assert variable in text

    blocks = _job_blocks(text)
    for name, block in blocks.items():
        assert "PIP_NO_INDEX=1" in block, name
        assert "GITHUB_ENV" in block, name

    post_install_markers = {
        "linux-minimal": "Run minimal installation checks",
        "linux-package-tests": "Run each package test directory separately",
        "macos-arm-smoke": "Verify isolation and installation",
        "public-contracts": "Run public contracts, fixtures, and overlay validation",
        "release-archives": "Build eight wheels and eight sdists outside the checkout",
    }
    for name, validation_marker in post_install_markers.items():
        block = blocks[name]
        assert block.index("Disable package-index access") < block.index(
            validation_marker
        ), name

    assert "bootstrap.sh --profile minimal" in blocks["linux-minimal"]
    assert "bootstrap.sh --profile minimal" in blocks["macos-arm-smoke"]
    assert "bootstrap.sh --profile dev" in blocks["linux-package-tests"]
    assert "bootstrap.sh --profile dev" in blocks["public-contracts"]
    assert re.search(
        r"pip install\s+(?:\\\n\s+)*--no-deps\s+(?:\\\n\s+)*--no-build-isolation"
        r"\s+(?:\\\n\s+)*-e packages/research-viz",
        blocks["linux-package-tests"],
    )

    release = blocks["release-archives"]
    for requirement in ('"build==1.5.0"', '"setuptools>=69"', "wheel", '"pytest>=8"'):
        assert requirement in release
    assert release.index("pip install") < release.index("PIP_NO_INDEX=1")


def test_workflow_invokes_package_public_and_archive_contracts() -> None:
    blocks = _job_blocks(_workflow_text())

    assert "bash ops/ci/run-package-tests.sh" in blocks["linux-package-tests"]
    assert "bash ops/ci/run-public-contracts.sh" in blocks["public-contracts"]
    release = blocks["release-archives"]
    assert "bash ops/ci/build-release-archives.sh" in release
    assert "ops/ci/inspect-release-archives.py" in release
    assert "--archive-dir" in release
    assert "RP_RELEASE_WHEEL_DIR" in release
    assert "RP_RELEASE_SDIST_DIR" in release
    assert "ops/tests/test_release_metadata.py" in release


def test_workflow_has_no_privileged_or_external_runtime_features() -> None:
    text = _workflow_text()
    normalized = text.lower()

    assert "continue-on-error" not in normalized
    assert "pull_request_target" not in normalized
    assert "workflow_run" not in normalized
    assert "schedule:" not in normalized
    assert "self-hosted" not in normalized
    assert "${{ secrets." not in normalized
    assert "permissions: write" not in normalized
    assert not re.search(r"(?m)^\s+[a-z-]+:\s+write$", normalized)
    assert "services:" not in normalized
    assert "container:" not in normalized
    forbidden_commands = (
        "sudo",
        "curl",
        "wget",
        "ssh",
        "rsync",
        "sbatch",
        "scancel",
        "slurm",
        "docker",
        "apptainer",
        "singularity",
        "twine",
        "gh release",
    )
    for command in forbidden_commands:
        assert not re.search(rf"(?<![a-z0-9_-]){re.escape(command)}(?![a-z0-9_-])", normalized)


def test_reusable_ci_scripts_are_strict_non_network_and_non_destructive() -> None:
    ci_root = REPO_ROOT / "ops" / "ci"
    for name in CI_SCRIPTS:
        path = ci_root / name
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".sh":
            assert re.search(r"(?m)^set -euo pipefail$", text), path
            assert not re.search(r"(?m)^\s*rm(?:\s|$)", text), path
        assert not re.search(r"(?m)^\s*(?:curl|wget|ssh|rsync)(?:\s|$)", text), path


def test_package_test_runner_uses_eight_separate_ordered_pytest_processes() -> None:
    text = (REPO_ROOT / "ops" / "ci" / "run-package-tests.sh").read_text(
        encoding="utf-8"
    )

    positions = [text.index(path) for path in PACKAGE_TEST_PATHS]
    assert positions == sorted(positions)
    assert len(re.findall(r"(?m)^  packages/research-[a-z-]+/tests$", text)) == len(
        PACKAGE_TEST_PATHS
    )
    assert 'for test_directory in "${test_directories[@]}"' in text
    assert (
        '"${python_bin}" -m pytest -p no:cacheprovider "${test_directory}"' in text
    )


def test_public_contract_runner_has_all_fixture_and_overlay_checks() -> None:
    text = (REPO_ROOT / "ops" / "ci" / "run-public-contracts.sh").read_text(
        encoding="utf-8"
    )

    assert "ops/tests" in text
    assert "-p no:cacheprovider" in text
    for command in FIXTURE_CHECKS:
        generator, _ = command.split()
        assert f"ops/scripts/{generator}" in text
    assert '"${python_bin}" "${generator}" --check' in text
    for overlay in PUBLIC_OVERLAYS:
        assert re.search(rf"(?m)^  {re.escape(overlay)}$", text)
    assert '"${rp_bin}" config validate --project "${overlay}"' in text


def test_archive_builder_uses_the_complete_staged_index_and_builds_both_formats() -> None:
    text = (REPO_ROOT / "ops" / "ci" / "build-release-archives.sh").read_text(
        encoding="utf-8"
    )

    assert "packages/research-*" in text or "research-analysis" in text
    assert 'git checkout-index --all --prefix="${snapshot_root%/}/"' in text
    assert '${snapshot_root}/packages/${package}' in text
    assert not re.search(r"(?m)^\s*cp(?:\s|$)", text)
    assert "-m build --no-isolation" in text
    assert "--sdist" not in text and "--wheel" not in text
    assert "RUNNER_TEMP" in text and "RP_RELEASE_DIST_DIR" in text
    assert "-mindepth" not in text and "-maxdepth" not in text
    assert "len(entries) != 16" in text
    assert "len(regular_files) != 16" in text
    assert "len(wheels) != 8" in text and "len(sdists) != 8" in text

    manifest = REPO_ROOT / "packages" / "research-neuro" / "MANIFEST.in"
    assert manifest.read_bytes() == b"prune tests\n"
    assert list((REPO_ROOT / "packages").glob("research-*/MANIFEST.in")) == [manifest]


def test_archive_inspector_requires_exact_wheel_and_sdist_sets() -> None:
    text = (REPO_ROOT / "ops" / "ci" / "inspect-release-archives.py").read_text(
        encoding="utf-8"
    )

    for package in (
        "research-analysis",
        "research-bids",
        "research-core",
        "research-hpc",
        "research-io",
        "research-ml",
        "research-neuro",
        "research-viz",
    ):
        assert f'"{package}"' in text
    assert 'len(list(archive_dir.glob("*.whl"))) == len(PACKAGE_NAMES)' in text
    assert 'len(list(archive_dir.glob("*.tar.gz"))) == len(PACKAGE_NAMES)' in text
    assert "sha256" in text.lower()
    assert "test_release_metadata.py" not in text
    for contract in (
        'message["Version"] == VERSION',
        'message["Requires-Python"] == ">=3.11"',
        'message["Author"] == AUTHOR',
        'message["Author-email"] is None',
        "PROJECT_URLS.items() <= project_urls(message).items()",
        "internal_requirements(message)",
        "CONSOLE_SCRIPTS[distribution]",
        "assert_safe_paths(names)",
        "license_bytes",
    ):
        assert contract in text
    for forbidden_part in (
        ".agents",
        ".claude",
        ".codex",
        ".cursor",
        ".pytest_cache",
        "__pycache__",
        "artifacts",
        "build",
        "datasets",
        "dist",
        "project",
        "secrets",
        "tests",
    ):
        assert f'"{forbidden_part}"' in text


def test_clean_checkout_script_checks_git_and_generated_residue() -> None:
    text = (REPO_ROOT / "ops" / "ci" / "check-clean-checkout.sh").read_text(
        encoding="utf-8"
    )

    assert "git diff --exit-code" in text
    assert "git diff --cached --exit-code" in text
    assert "git status --porcelain=v1 --untracked-files=all" in text
    for marker in (
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "*.egg-info",
        "build",
        "dist",
        ".coverage",
        "*.claim",
        ".*.publication-*",
        ".*.transaction-*",
        ".*.staging-*",
    ):
        assert marker in text
    assert "rm " not in text


def test_ci_documentation_maps_to_the_reusable_scripts() -> None:
    readme = CI_README.read_text(encoding="utf-8")
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    for command in (
        "bash ops/ci/run-package-tests.sh",
        "bash ops/ci/run-public-contracts.sh",
        "bash ops/ci/build-release-archives.sh",
        "bash ops/ci/check-clean-checkout.sh",
    ):
        assert command in readme
    assert "ops/ci/inspect-release-archives.py" in readme
    assert "ops/ci/source-tree-integrity.py" in readme
    assert "git checkout-index --all" in readme
    assert "--archive-dir" in readme
    assert "PIP_NO_INDEX=1" in readme
    assert "RP_RELEASE_WHEEL_DIR" in readme
    assert "RP_RELEASE_SDIST_DIR" in readme
    assert "-p no:cacheprovider" in readme
    assert "ops/ci/README.md" in contributing
    assert re.search(
        r"(?m)^test-ci-contract:\n\tPYTHONDONTWRITEBYTECODE=1 "
        r"PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
        r".*ops/tests/test_ci_contract\.py$",
        makefile,
    )
