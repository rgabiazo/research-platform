from __future__ import annotations

from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_EXCLUDES = REPO_ROOT / "ops" / "sync" / "rsync" / "exclude.workspace.txt"
SCOPED_MARKDOWN = (
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "onboarding" / "README.md",
    REPO_ROOT / "docs" / "onboarding" / "coding-agent-workflow.md",
    REPO_ROOT / "docs" / "mvpa-crossnobis.md",
    REPO_ROOT / "project" / "project-example" / "README.md",
)
LINK_PATTERN = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def _ignored(path: str) -> bool:
    completed = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", "--", path],
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        raise AssertionError(f"git check-ignore failed for {path}: {completed.returncode}")
    return completed.returncode == 0


def _tracked_paths() -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(completed.stdout.splitlines())


def test_local_agent_log_editor_and_scratch_probes_are_ignored() -> None:
    probes = (
        ".codex/session.json",
        ".agents/state.json",
        ".cursor/settings.json",
        ".claude/settings.json",
        "AGENTS.local.md",
        "packages/research-io/AGENTS.local.md",
        "task.log",
        "packages/research-core/debug.log",
        "notes.swp",
        "docs/notes.swo",
        "README.md~",
        "scratch/private-audit.md",
        "scratch/temporary-table.tsv",
    )

    for probe in probes:
        assert _ignored(probe), probe


def test_public_scaffolds_and_generator_owned_fixtures_remain_visible() -> None:
    visible = (
        "scratch/README.md",
        "scratch/cache/.gitkeep",
        "scratch/tmp/.gitkeep",
        "scratch/work/.gitkeep",
        "datasets/ds-tabular-example/toy_observations.csv",
        "datasets/ds-roi-example/images/toy_reference.nii",
        "datasets/ds-mvpa-example/patterns/toy_crossnobis_patterns.tsv",
        "packages/research-bids/tests/fixtures/toy-memory/raw/toy01_visit01_toymemory_2099-01-01.csv",
    )

    for path in visible:
        assert not _ignored(path), path


def test_private_data_artifacts_derivatives_and_overlays_remain_ignored() -> None:
    protected = (
        "secrets/hpc/targets.yaml",
        "project/project-private/project.yaml",
        "datasets/ds-private-example/data.tsv",
        "datasets/ds-tabular-example/rawdata/private.csv",
        "datasets/ds-other-example/images/private.nii",
        "datasets/ds-derivatives-example/derivatives/features/private-output.tsv",
        "artifacts/runs/private-run/manifest.json",
    )

    for path in protected:
        assert _ignored(path), path


def test_private_hpc_destinations_and_casefolded_public_scaffolds_match_git_boundary() -> None:
    private = (
        "secrets/hpc/ssh-profiles.yaml",
        "secrets/hpc/targets.yaml",
        "secrets/.env",
    )
    public = (
        "secrets/local/readme.md",
        "secrets/local/ReadMe.MD",
        "secrets/local/.GITKEEP",
        "secrets/local/.GitKeep",
        "secrets/local/config.EXAMPLE",
        "secrets/local/CONFIG.Example",
    )

    for path, expected_returncode in (
        *((path, 0) for path in private),
        *((path, 1) for path in public),
    ):
        completed = subprocess.run(
            [
                "git",
                "-c",
                "core.ignoreCase=true",
                "check-ignore",
                "--no-index",
                "--quiet",
                "--",
                path,
            ],
            cwd=REPO_ROOT,
            check=False,
        )
        assert completed.returncode == expected_returncode, path


def test_workspace_sync_excludes_all_local_agent_state() -> None:
    excludes = {
        line.strip()
        for line in WORKSPACE_EXCLUDES.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {".codex/", ".agents/", ".cursor/", ".claude/", "AGENTS.local.md"} <= excludes


def test_public_contributor_guidance_is_tracked_and_private_coordination_is_not() -> None:
    tracked = _tracked_paths()

    assert {
        "AGENTS.md",
        "packages/research-io/AGENTS.md",
        "CONTRIBUTING.md",
    } <= tracked
    assert not any(path.startswith(".codex/") for path in tracked)
    assert not any(path.startswith("docs/plans/2026-03-23-") for path in tracked)
    assert not any(
        path.startswith("docs/onboarding/")
        and Path(path).name.startswith("codex")
        for path in tracked
    )
    assert not any(path.endswith("AGENTS.local.md") for path in tracked)
    assert not any(".git/info/exclude" in path for path in tracked)


def test_generalized_coding_agent_links_resolve_and_old_redirects_are_absent() -> None:
    for document in SCOPED_MARKDOWN:
        text = document.read_text(encoding="utf-8")
        for target in LINK_PATTERN.findall(text):
            path_text = target.split("#", 1)[0]
            if not path_text or "://" in path_text:
                continue
            assert (document.parent / path_text).resolve().is_file(), (document, target)


def test_beginner_mvpa_temp_commands_are_portable_and_shell_valid() -> None:
    expected = '''TMP_BASE="${TMPDIR:-/tmp}"
export ARTIFACTS_ROOT="$(
  mktemp -d "${TMP_BASE%/}/research-platform-toy-mvpa.XXXXXX"
)"'''

    for document in SCOPED_MARKDOWN[-2:]:
        text = document.read_text(encoding="utf-8")
        assert expected in text
        assert "/private/tmp" not in text

    completed = subprocess.run(
        ["bash", "-n", "-c", expected],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_scoped_generic_hpc_examples_have_no_site_default_assumptions() -> None:
    scoped = (
        REPO_ROOT / "ops" / "sync" / "ssh" / "targets.example.yaml",
        REPO_ROOT / "docs" / "bids-hpc-slice.md",
        REPO_ROOT / "docs" / "bids-analysis-slice.md",
    )
    forbidden = ("$SCRATCH", "/local/scratch", "StdEnv/2023", "arrow/23.0.1", "apptainer/1.4.5")

    for document in scoped:
        text = document.read_text(encoding="utf-8")
        for assumption in forbidden:
            assert assumption not in text, (document, assumption)

    targets = scoped[0].read_text(encoding="utf-8")
    assert "Provider-neutral starter" in targets
    assert "module stack, scratch convention" in targets
    assert "mode: atomic_no_replace" in targets

    hpc_guide = scoped[1].read_text(encoding="utf-8")
    assert "`rp hpc setup` (canonical local starter)" in hpc_guide
    assert "`rp hpc init` (legacy Alliance-oriented local-default helper)" in hpc_guide
    assert "interactive-login" not in hpc_guide
    assert "--alliance-user" not in hpc_guide
    assert "--target target-a" in hpc_guide
    assert "rp hpc validate --target target-a" in hpc_guide

    analysis_guide = scoped[2].read_text(encoding="utf-8")
    assert "optional operator example is" in analysis_guide
    assert "provider-neutral" in analysis_guide
