from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "ops" / "ci" / "source-tree-integrity.py"


def _run(
    *arguments: str, expected_returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected_returncode, result.stdout + result.stderr
    return result


def _capture(root: Path, baseline: Path) -> None:
    _run("capture", "--root", str(root), "--output", str(baseline))


def _verify(
    root: Path, baseline: Path, *, expected_returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return _run(
        "verify",
        "--root",
        str(root),
        "--baseline",
        str(baseline),
        expected_returncode=expected_returncode,
    )


def _tree_state(root: Path) -> dict[str, tuple[object, ...]]:
    state: dict[str, tuple[object, ...]] = {}
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            state[relative_path] = ("symlink", mode, os.readlink(path))
        elif path.is_dir():
            state[relative_path] = ("directory", mode)
        else:
            state[relative_path] = (
                "file",
                mode,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    return state


@pytest.fixture
def source_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    root.mkdir()
    (root / "README.md").write_text("public source\n", encoding="utf-8")
    package = root / "package"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    return root, baseline


def test_unchanged_tree_verifies(source_tree: tuple[Path, Path]) -> None:
    root, baseline = source_tree
    _capture(root, baseline)

    result = _verify(root, baseline)

    assert "Source-tree integrity verified" in result.stdout


def test_capture_is_deterministic_and_does_not_write_measured_tree(
    source_tree: tuple[Path, Path], tmp_path: Path
) -> None:
    root, baseline = source_tree
    second_baseline = tmp_path / "second-baseline.json"
    before = _tree_state(root)

    _capture(root, baseline)
    after_first_capture = _tree_state(root)
    _capture(root, second_baseline)
    _verify(root, baseline)

    assert after_first_capture == before
    assert _tree_state(root) == before
    assert baseline.read_bytes() == second_baseline.read_bytes()


def test_git_metadata_is_excluded(source_tree: tuple[Path, Path]) -> None:
    root, baseline = source_tree
    git_directory = root / ".git"
    git_directory.mkdir()
    (git_directory / "index").write_bytes(b"private git state")
    _capture(root, baseline)
    manifest = json.loads(baseline.read_text(encoding="utf-8"))

    assert all(
        entry["path"] != ".git" and not entry["path"].startswith(".git/")
        for entry in manifest["entries"]
    )

    (git_directory / "index").write_bytes(b"changed git state")
    _verify(root, baseline)


def test_changed_file_bytes_fail(source_tree: tuple[Path, Path]) -> None:
    root, baseline = source_tree
    _capture(root, baseline)
    (root / "README.md").write_text("changed source\n", encoding="utf-8")

    result = _verify(root, baseline, expected_returncode=1)

    assert "changed: README.md" in result.stderr


@pytest.mark.parametrize("change", ["added", "removed"])
def test_added_and_removed_paths_fail(
    source_tree: tuple[Path, Path], change: str
) -> None:
    root, baseline = source_tree
    removable = root / "remove-me.txt"
    removable.write_text("present\n", encoding="utf-8")
    _capture(root, baseline)
    if change == "added":
        (root / "added.txt").write_text("new\n", encoding="utf-8")
        expected = "added: added.txt"
    else:
        removable.unlink()
        expected = "removed: remove-me.txt"

    result = _verify(root, baseline, expected_returncode=1)

    assert expected in result.stderr


def test_symbolic_link_target_change_fails(source_tree: tuple[Path, Path]) -> None:
    root, baseline = source_tree
    link = root / "current"
    try:
        link.symlink_to("README.md")
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")
    _capture(root, baseline)
    link.unlink()
    link.symlink_to("package/module.py")

    result = _verify(root, baseline, expected_returncode=1)

    assert "changed: current" in result.stderr


def test_ignored_installer_metadata_is_measured(
    source_tree: tuple[Path, Path]
) -> None:
    root, baseline = source_tree
    (root / ".gitignore").write_text("*.egg-info/\n", encoding="utf-8")
    metadata = root / "package.egg-info"
    metadata.mkdir()
    sources = metadata / "SOURCES.txt"
    sources.write_text("package/module.py\n", encoding="utf-8")
    _capture(root, baseline)
    sources.write_text("package/module.py\nadded.py\n", encoding="utf-8")

    result = _verify(root, baseline, expected_returncode=1)

    assert "changed: package.egg-info/SOURCES.txt" in result.stderr


def test_path_type_change_fails(source_tree: tuple[Path, Path]) -> None:
    root, baseline = source_tree
    path = root / "kind"
    path.write_text("file\n", encoding="utf-8")
    _capture(root, baseline)
    path.unlink()
    path.mkdir()

    result = _verify(root, baseline, expected_returncode=1)

    assert "type changed: kind (file -> directory)" in result.stderr


def test_file_mode_change_fails(source_tree: tuple[Path, Path]) -> None:
    root, baseline = source_tree
    path = root / "README.md"
    path.chmod(0o644)
    _capture(root, baseline)
    path.chmod(0o755)

    result = _verify(root, baseline, expected_returncode=1)

    assert "changed: README.md" in result.stderr


def test_missing_baseline_has_clear_error(source_tree: tuple[Path, Path]) -> None:
    root, baseline = source_tree

    result = _verify(root, baseline, expected_returncode=1)

    assert "baseline manifest does not exist" in result.stderr


def test_baseline_must_be_outside_measured_tree(
    source_tree: tuple[Path, Path]
) -> None:
    root, _ = source_tree
    baseline = root / "baseline.json"

    result = _run(
        "capture",
        "--root",
        str(root),
        "--output",
        str(baseline),
        expected_returncode=1,
    )

    assert "must be outside the measured source root" in result.stderr
    assert not baseline.exists()
