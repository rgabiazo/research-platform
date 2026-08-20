#!/usr/bin/env python3
"""Inspect coordinated package wheels and sdists without external dependencies."""

from __future__ import annotations

import argparse
import configparser
from email.parser import BytesParser
from email.policy import default
import hashlib
from pathlib import Path, PurePosixPath
import re
import tarfile
import tomllib
import zipfile


PACKAGE_NAMES = (
    "research-analysis",
    "research-bids",
    "research-core",
    "research-hpc",
    "research-io",
    "research-ml",
    "research-neuro",
    "research-viz",
)
VERSION = "0.1.0a1"
AUTHOR = "Raphael Gabiazon"
REPOSITORY_URL = "https://github.com/rgabiazo/research-platform"
PROJECT_URLS = {
    "Homepage": REPOSITORY_URL,
    "Repository": REPOSITORY_URL,
    "Documentation": f"{REPOSITORY_URL}/blob/main/docs/README.md",
    "Issues": f"{REPOSITORY_URL}/issues",
    "Changelog": f"{REPOSITORY_URL}/blob/main/CHANGELOG.md",
}
INTERNAL_REQUIREMENTS = {
    "research-analysis": {"research-ml>=0.1.0a1,<0.2"},
    "research-bids": {"research-neuro>=0.1.0a1,<0.2"},
    "research-core": {"research-hpc>=0.1.0a1,<0.2"},
}
CONSOLE_SCRIPTS = {
    "research-analysis": {},
    "research-bids": {"research-bids": "research_platform.bids.cli:main"},
    "research-core": {"rp": "research_platform.core.entrypoint:main"},
    "research-hpc": {"research-hpc": "research_platform.hpc.cli:main"},
    "research-io": {"research-io": "research_platform.io.cli:main"},
    "research-ml": {},
    "research-neuro": {},
    "research-viz": {},
}
FORBIDDEN_PARTS = {
    ".agents",
    ".claude",
    ".codex",
    ".cursor",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "build",
    "datasets",
    "dist",
    "project",
    "secrets",
    "tests",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    return parser.parse_args()


def assert_safe_paths(names: list[str]) -> None:
    for name in names:
        path = PurePosixPath(name)
        assert not path.is_absolute(), name
        assert ".." not in path.parts, name
        assert "\\" not in name, name
        assert not (set(path.parts) & FORBIDDEN_PARTS), name
        assert not name.endswith((".pyc", ".pyo", ".swp", ".swo", "~")), name


def project_urls(message: object) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in message.get_all("Project-URL", []):
        label, separator, url = value.partition(",")
        assert separator, value
        result[label.strip()] = url.strip()
    return result


def requirement_semantics(requirement: str) -> tuple[str, tuple[str, ...]]:
    requirement = requirement.split(";", 1)[0].strip()
    match = re.fullmatch(r"(research-[a-z]+)\s*(.*)", requirement)
    assert match, requirement
    name, specifiers = match.groups()
    return name, tuple(
        sorted(part.replace(" ", "") for part in specifiers.split(",") if part.strip())
    )


def internal_requirements(message: object) -> set[tuple[str, tuple[str, ...]]]:
    requirements = {
        value
        for value in message.get_all("Requires-Dist", [])
        if re.match(r"^research-[a-z]+(?:$|[<=>!~])", value)
    }
    return {requirement_semantics(value) for value in requirements}


def entry_points(content: bytes | None) -> dict[str, str]:
    if content is None:
        return {}
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(content.decode("utf-8"))
    if not parser.has_section("console_scripts"):
        return {}
    return dict(parser.items("console_scripts"))


def assert_metadata(message: object, distribution: str) -> None:
    assert message["Name"] == distribution
    assert message["Version"] == VERSION
    assert message["Requires-Python"] == ">=3.11"
    assert message["Author"] == AUTHOR
    assert message["Author-email"] is None
    assert PROJECT_URLS.items() <= project_urls(message).items()
    observed = internal_requirements(message)
    expected = {
        requirement_semantics(value)
        for value in INTERNAL_REQUIREMENTS.get(distribution, set())
    }
    assert observed == expected, (distribution, observed, expected)
    expression = message["License-Expression"]
    declaration = message["License"]
    assert expression == "MIT" or (
        declaration is not None and "MIT License" in declaration
    )


def expected_archive_paths(archive_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    wheels: dict[str, Path] = {}
    sdists: dict[str, Path] = {}
    for distribution in PACKAGE_NAMES:
        normalized = distribution.replace("-", "_")
        wheel_matches = sorted(archive_dir.glob(f"{normalized}-{VERSION}-*.whl"))
        assert len(wheel_matches) == 1, (distribution, wheel_matches)
        wheels[distribution] = wheel_matches[0]
        sdist_matches = {
            *archive_dir.glob(f"{distribution}-{VERSION}.tar.gz"),
            *archive_dir.glob(f"{normalized}-{VERSION}.tar.gz"),
        }
        assert len(sdist_matches) == 1, (distribution, sdist_matches)
        sdists[distribution] = next(iter(sdist_matches))
    assert len(list(archive_dir.glob("*.whl"))) == len(PACKAGE_NAMES)
    assert len(list(archive_dir.glob("*.tar.gz"))) == len(PACKAGE_NAMES)
    return wheels, sdists


def inspect_wheel(
    path: Path,
    distribution: str,
    license_bytes: bytes,
) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert_safe_paths(names)
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_names) == 1
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_names[0]))
        assert_metadata(metadata, distribution)
        license_names = [
            name
            for name in names
            if ".dist-info/licenses/" in name and PurePosixPath(name).name == "LICENSE"
        ]
        assert len(license_names) == 1
        assert archive.read(license_names[0]) == license_bytes
        entry_point_names = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        assert len(entry_point_names) <= 1
        content = archive.read(entry_point_names[0]) if entry_point_names else None
        assert entry_points(content) == CONSOLE_SCRIPTS[distribution]


def inspect_sdist(
    path: Path,
    distribution: str,
    license_bytes: bytes,
) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        assert all(member.isfile() or member.isdir() for member in members)
        assert_safe_paths([member.name for member in members])
        metadata_members = [
            member
            for member in members
            if PurePosixPath(member.name).name == "PKG-INFO"
            and len(PurePosixPath(member.name).parts) == 2
        ]
        assert len(metadata_members) == 1
        metadata_stream = archive.extractfile(metadata_members[0])
        assert metadata_stream is not None
        assert_metadata(
            BytesParser(policy=default).parsebytes(metadata_stream.read()), distribution
        )
        license_members = [
            member
            for member in members
            if PurePosixPath(member.name).name == "LICENSE"
            and len(PurePosixPath(member.name).parts) == 2
        ]
        assert len(license_members) == 1
        license_stream = archive.extractfile(license_members[0])
        assert license_stream is not None
        assert license_stream.read() == license_bytes


def main() -> int:
    args = parse_args()
    archive_dir = args.archive_dir.resolve()
    repo_root = args.repo_root.resolve()
    if not archive_dir.is_dir():
        raise SystemExit(f"archive directory was not found: {archive_dir}")
    with (repo_root / "packages" / "research-core" / "pyproject.toml").open(
        "rb"
    ) as stream:
        assert tomllib.load(stream)["project"]["version"] == VERSION
    license_bytes = (repo_root / "LICENSE").read_bytes()
    wheels, sdists = expected_archive_paths(archive_dir)
    for distribution in PACKAGE_NAMES:
        inspect_wheel(wheels[distribution], distribution, license_bytes)
        inspect_sdist(sdists[distribution], distribution, license_bytes)
    for archive in sorted((*wheels.values(), *sdists.values()), key=lambda item: item.name):
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        print(f"{digest}  {archive.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
