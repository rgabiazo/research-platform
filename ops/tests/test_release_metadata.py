"""Contracts for public-alpha identity, licensing, and built archives."""

from __future__ import annotations

import configparser
from email.parser import BytesParser
from email.policy import default
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tomllib
from urllib.parse import urlparse
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
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
PUBLIC_ALPHA_VERSION = "0.1.0a1"
FUTURE_GIT_TAG = "v0.1.0a1"
FUTURE_RELEASE_TITLE = "Research Platform 0.1.0 Alpha 1"
REPOSITORY_URL = "https://github.com/rgabiazo/research-platform"
PROJECT_URLS = {
    "Homepage": REPOSITORY_URL,
    "Repository": REPOSITORY_URL,
    "Documentation": f"{REPOSITORY_URL}/blob/main/docs/README.md",
    "Issues": f"{REPOSITORY_URL}/issues",
    "Changelog": f"{REPOSITORY_URL}/blob/main/CHANGELOG.md",
}
ORCID_URL = "https://orcid.org/0009-0008-6575-4993"
CITATION_EMAIL = "@".join(("rgabiazo", "uwo.ca"))
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
REQUIRED_CLASSIFIERS = {
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
}
MIT_LICENSE = """MIT License

Copyright (c) 2026 Raphael Gabiazon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _package_root(distribution: str) -> Path:
    return REPO_ROOT / "packages" / distribution


def _public_worktree_files() -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(
        REPO_ROOT / os.fsdecode(raw_path)
        for raw_path in completed.stdout.split(b"\0")
        if raw_path
    )


def _project_metadata(distribution: str) -> dict[str, object]:
    with (_package_root(distribution) / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _cff_structure(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Parse the deliberately small scalar/list subset used by CITATION.cff."""

    top_level: dict[str, str] = {}
    authors: list[dict[str, str]] = []
    in_authors = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        top_match = re.fullmatch(r"([a-z][a-z0-9-]*):(?:\s*(.*))?", line)
        if top_match:
            key, value = top_match.groups()
            assert key not in top_level, f"duplicate CFF key {key!r} at line {line_number}"
            top_level[key] = _scalar(value or "")
            in_authors = key == "authors"
            continue
        if in_authors:
            first_field = re.fullmatch(
                r"  - ([a-z][a-z0-9-]*):(?:\s*(.*))?", line
            )
            later_field = re.fullmatch(
                r"    ([a-z][a-z0-9-]*):(?:\s*(.*))?", line
            )
            if first_field:
                authors.append({first_field.group(1): _scalar(first_field.group(2) or "")})
                continue
            if later_field and authors:
                key, value = later_field.groups()
                assert key not in authors[-1], (
                    f"duplicate author key {key!r} at line {line_number}"
                )
                authors[-1][key] = _scalar(value or "")
                continue
        if line.startswith("  "):
            # Literal/folded content owned by a preceding scalar is valid YAML.
            continue
        raise AssertionError(f"unsupported or malformed CFF line {line_number}: {line!r}")
    return top_level, authors


def _orcid_checksum_is_valid(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "orcid.org":
        return False
    if parsed.params or parsed.query or parsed.fragment:
        return False
    match = re.fullmatch(r"/(\d{4})-(\d{4})-(\d{4})-(\d{3}[\dX])", parsed.path)
    if not match:
        return False
    identifier = "".join(match.groups())
    total = 0
    for character in identifier[:15]:
        total = (total + int(character)) * 2
    result = (12 - (total % 11)) % 11
    expected = "X" if result == 10 else str(result)
    return identifier[-1] == expected


def _project_urls_from_message(message: object) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in message.get_all("Project-URL", []):
        label, separator, url = item.partition(",")
        assert separator, item
        result[label.strip()] = url.strip()
    return result


def _internal_requirements(requirements: list[str]) -> set[str]:
    return {
        requirement
        for requirement in requirements
        if re.match(r"^research-[a-z]+(?:$|[<=>!~])", requirement)
    }


def _requirement_semantics(requirement: str) -> tuple[str, tuple[str, ...]]:
    """Compare simple internal requirements without depending on packaging."""

    requirement_without_marker = requirement.split(";", 1)[0].strip()
    match = re.fullmatch(r"(research-[a-z]+)\s*(.*)", requirement_without_marker)
    assert match, requirement
    name, specifiers = match.groups()
    normalized_specifiers = tuple(
        sorted(part.replace(" ", "") for part in specifiers.split(",") if part.strip())
    )
    return name, normalized_specifiers


def _entry_points(content: bytes | None) -> dict[str, str]:
    if content is None:
        return {}
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(content.decode("utf-8"))
    if not parser.has_section("console_scripts"):
        return {}
    return dict(parser.items("console_scripts"))


def _assert_safe_archive_paths(names: list[str]) -> None:
    forbidden_parts = {
        ".agents",
        ".claude",
        ".codex",
        ".cursor",
        ".git",
        ".pytest_cache",
        "__pycache__",
        "artifacts",
        "datasets",
        "project",
        "secrets",
        "tests",
    }
    for name in names:
        path = PurePosixPath(name)
        assert not path.is_absolute(), name
        assert ".." not in path.parts, name
        assert "\\" not in name, name
        assert not (set(path.parts) & forbidden_parts), name
        assert not name.endswith((".pyc", ".pyo", ".swp", ".swo", "~")), name


def _wheel_path(directory: Path, distribution: str) -> Path:
    normalized = distribution.replace("-", "_")
    matches = sorted(directory.glob(f"{normalized}-{PUBLIC_ALPHA_VERSION}-*.whl"))
    assert len(matches) == 1, (distribution, matches)
    return matches[0]


def _sdist_path(directory: Path, distribution: str) -> Path:
    variants = {distribution, distribution.replace("-", "_")}
    matches = sorted(
        path
        for variant in variants
        for path in directory.glob(f"{variant}-{PUBLIC_ALPHA_VERSION}.tar.gz")
    )
    assert len(set(matches)) == 1, (distribution, matches)
    return matches[0]


def _assert_archive_metadata(message: object, distribution: str) -> None:
    assert message["Name"] == distribution
    assert message["Version"] == PUBLIC_ALPHA_VERSION
    assert message["Requires-Python"] == ">=3.11"
    assert message["Author"] == "Raphael Gabiazon"
    assert message["Author-email"] is None
    assert REQUIRED_CLASSIFIERS <= set(message.get_all("Classifier", []))
    assert PROJECT_URLS.items() <= _project_urls_from_message(message).items()
    requirements = _internal_requirements(message.get_all("Requires-Dist", []))
    assert {_requirement_semantics(item) for item in requirements} == {
        _requirement_semantics(item)
        for item in INTERNAL_REQUIREMENTS.get(distribution, set())
    }
    license_expression = message["License-Expression"]
    license_declaration = message["License"]
    assert license_expression == "MIT" or (
        license_declaration is not None and "MIT License" in license_declaration
    )


def test_root_license_is_canonical_mit_text() -> None:
    assert (REPO_ROOT / "LICENSE").read_text(encoding="utf-8") == MIT_LICENSE


@pytest.mark.parametrize("distribution", PACKAGE_NAMES)
def test_package_metadata_has_coordinated_public_identity(distribution: str) -> None:
    project = _project_metadata(distribution)

    assert project["name"] == distribution
    assert project["version"] == PUBLIC_ALPHA_VERSION
    assert project["requires-python"] == ">=3.11"
    assert project["authors"] == [{"name": "Raphael Gabiazon"}]
    assert project["license"] == {"file": "LICENSE"}
    assert REQUIRED_CLASSIFIERS <= set(project["classifiers"])
    assert PROJECT_URLS.items() <= project["urls"].items()
    assert CITATION_EMAIL not in (_package_root(distribution) / "pyproject.toml").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("distribution", PACKAGE_NAMES)
def test_package_license_copy_matches_canonical_license(distribution: str) -> None:
    assert (_package_root(distribution) / "LICENSE").read_bytes() == (
        REPO_ROOT / "LICENSE"
    ).read_bytes()


def test_internal_dependency_constraints_remain_coordinated() -> None:
    observed: dict[str, set[str]] = {}
    for distribution in PACKAGE_NAMES:
        project = _project_metadata(distribution)
        requirements = list(project.get("dependencies", ()))
        for group in project.get("optional-dependencies", {}).values():
            requirements.extend(group)
        internal = _internal_requirements(requirements)
        if internal:
            observed[distribution] = internal
    assert observed == INTERNAL_REQUIREMENTS


def test_citation_cff_has_valid_public_identity() -> None:
    citation = REPO_ROOT / "CITATION.cff"
    top_level, authors = _cff_structure(citation)

    assert top_level["cff-version"] == "1.2.0"
    assert top_level["title"] == "Research Platform"
    assert top_level["type"] == "software"
    assert top_level["version"] == PUBLIC_ALPHA_VERSION
    assert top_level["repository-code"] == REPOSITORY_URL
    assert top_level["license"] == "MIT"
    assert top_level["message"]
    assert top_level["abstract"]
    assert len(authors) == 1
    assert authors[0] == {
        "family-names": "Gabiazon",
        "given-names": "Raphael",
        "affiliation": "Western University",
        "orcid": ORCID_URL,
        "email": CITATION_EMAIL,
    }
    assert _orcid_checksum_is_valid(authors[0]["orcid"])
    assert "doi" not in top_level
    assert "date-released" not in top_level


def test_citation_email_is_confined_to_citation_metadata() -> None:
    citation = REPO_ROOT / "CITATION.cff"
    email_bytes = CITATION_EMAIL.encode("utf-8")
    assert email_bytes in citation.read_bytes()
    for path in _public_worktree_files():
        if path == citation or not path.is_file():
            continue
        assert email_bytes not in path.read_bytes(), path


def test_security_policy_is_private_actionable_and_activation_aware() -> None:
    text = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())

    assert PUBLIC_ALPHA_VERSION in text
    assert f"{REPOSITORY_URL}/security/advisories/new" in text
    assert "private vulnerability reporting" in normalized
    assert "public issue" in normalized
    assert "do not" in normalized or "not" in normalized
    assert re.search(r"affected (?:component and )?version", normalized)
    for report_detail in ("reproduce", "impact"):
        assert report_detail in normalized
    assert "acknowledg" in normalized
    assert "update" in normalized
    assert "support" in normalized
    assert "installation" in normalized
    assert "scientific" in normalized
    future_reporting = re.search(r"future .*repository is public", normalized) or (
        "once the repository is public" in normalized
    )
    active_reporting = (
        "private vulnerability reporting is enabled for this repository" in normalized
        and "maintainer monitors github security-alert notifications" in normalized
    )
    assert future_reporting or active_reporting
    assert CITATION_EMAIL not in text
    assert not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)


def test_release_metadata_has_no_placeholders_or_unreleased_claims() -> None:
    scoped = (
        REPO_ROOT / "LICENSE",
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "SECURITY.md",
        *(_package_root(distribution) / "pyproject.toml" for distribution in PACKAGE_NAMES),
    )
    placeholder_pattern = re.compile(
        r"(?i)(?:PASTE_[A-Z0-9_]+_HERE|\bTODO\b|\bTBD\b|example\.com)"
    )
    for path in scoped:
        assert not placeholder_pattern.search(path.read_text(encoding="utf-8")), path

    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "date-released:" not in citation
    assert not re.search(r"(?im)^doi\s*:", citation)


def test_changelog_describes_planned_alpha_without_overclaiming() -> None:
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())

    assert "## [Unreleased]" in text
    assert FUTURE_GIT_TAG in text
    assert "docs/capabilities.md" in text
    for runnable_surface in (
        "tabular",
        "coordinate-sphere ROI",
        "generic-NIfTI extraction",
        "materialized-pattern crossnobis",
    ):
        assert runnable_surface.lower() in normalized
    for boundary in ("plan-only", "experimental", "external-runtime", "HPC"):
        assert boundary.lower() in normalized
    for completed_work in (
        r"deterministic,? generated-from-scratch public fixture",
        r"sanitation",
        r"portab(?:ility|le)",
        r"transaction(?: safety|al output)",
    ):
        assert re.search(completed_work, normalized)
    assert re.search(r"source[- ]checkout", normalized)
    assert "planned" in normalized
    assert re.search(r"not currently (?:available from|distributed through) pypi", normalized)
    assert not re.search(r"(?m)^## \[(?:v)?0\.1\.0a1\]\s+-\s+\d{4}-\d{2}-\d{2}$", text)
    assert "doi" not in normalized
    assert "live-cluster validation was performed" not in normalized
    assert "spm support" not in normalized
    assert "no live-cluster or public real-data validation is claimed" in normalized
    if "distances.tsv" in text:
        assert "RDM-ready pairwise-distance table" in text
        assert "exported RDM" not in text


def test_public_release_identity_and_policy_links_are_consistent() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    combined = f"{readme}\n{docs_index}\n{(REPO_ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')}"

    assert FUTURE_GIT_TAG in combined
    assert FUTURE_RELEASE_TITLE in combined
    assert PUBLIC_ALPHA_VERSION in readme
    assert "not currently available from pypi" in " ".join(readme.lower().split())
    for readme_target, docs_target in (
        ("LICENSE", "../LICENSE"),
        ("CHANGELOG.md", "../CHANGELOG.md"),
        ("CITATION.cff", "../CITATION.cff"),
        ("SECURITY.md", "../SECURITY.md"),
        ("CONTRIBUTING.md", "../CONTRIBUTING.md"),
        ("docs/capabilities.md", "capabilities.md"),
    ):
        assert re.search(rf"\[[^]]+\]\({re.escape(readme_target)}\)", readme)
        assert re.search(rf"\[[^]]+\]\({re.escape(docs_target)}\)", docs_index)

    stale_tag_pattern = re.compile(r"v0\.1\.0-alpha\.\d+")
    text_suffixes = {".cff", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
    for path in _public_worktree_files():
        if path.suffix.lower() not in text_suffixes or not path.is_file():
            continue
        assert not stale_tag_pattern.search(path.read_text(encoding="utf-8")), path


def test_release_metadata_markdown_links_resolve_locally() -> None:
    documents = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "SECURITY.md",
        REPO_ROOT / "docs" / "README.md",
    )
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for document in documents:
        for target in link_pattern.findall(document.read_text(encoding="utf-8")):
            target_path = target.split("#", 1)[0]
            if not target_path or "://" in target_path:
                continue
            assert (document.parent / target_path).resolve().is_file(), (document, target)


@pytest.mark.parametrize("distribution", PACKAGE_NAMES)
def test_built_wheel_has_public_metadata_and_license(distribution: str) -> None:
    directory_text = os.environ.get("RP_RELEASE_WHEEL_DIR")
    if not directory_text:
        pytest.skip("set RP_RELEASE_WHEEL_DIR to inspect offline release wheels")
    wheel = _wheel_path(Path(directory_text), distribution)

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        _assert_safe_archive_paths(names)
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_names) == 1
        message = BytesParser(policy=default).parsebytes(archive.read(metadata_names[0]))
        _assert_archive_metadata(message, distribution)

        license_names = [
            name
            for name in names
            if ".dist-info/licenses/" in name and PurePosixPath(name).name == "LICENSE"
        ]
        assert len(license_names) == 1
        assert archive.read(license_names[0]) == (REPO_ROOT / "LICENSE").read_bytes()

        entry_point_names = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        assert len(entry_point_names) <= 1
        entry_points = (
            archive.read(entry_point_names[0]) if entry_point_names else None
        )
        assert _entry_points(entry_points) == CONSOLE_SCRIPTS[distribution]

    assert hashlib.sha256(wheel.read_bytes()).hexdigest()


@pytest.mark.parametrize("distribution", PACKAGE_NAMES)
def test_built_sdist_has_public_metadata_and_license(distribution: str) -> None:
    directory_text = os.environ.get("RP_RELEASE_SDIST_DIR")
    if not directory_text:
        pytest.skip("set RP_RELEASE_SDIST_DIR to inspect offline release sdists")
    sdist = _sdist_path(Path(directory_text), distribution)

    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        _assert_safe_archive_paths(names)
        assert all(member.isfile() or member.isdir() for member in members)
        metadata_members = [
            member
            for member in members
            if PurePosixPath(member.name).name == "PKG-INFO"
            and len(PurePosixPath(member.name).parts) == 2
        ]
        assert len(metadata_members) == 1
        metadata_stream = archive.extractfile(metadata_members[0])
        assert metadata_stream is not None
        _assert_archive_metadata(
            BytesParser(policy=default).parsebytes(metadata_stream.read()), distribution
        )
        license_members = [
            member
            for member in members
            if PurePosixPath(member.name).name == "LICENSE" and len(PurePosixPath(member.name).parts) == 2
        ]
        assert len(license_members) == 1
        license_stream = archive.extractfile(license_members[0])
        assert license_stream is not None
        assert license_stream.read() == (REPO_ROOT / "LICENSE").read_bytes()

    assert hashlib.sha256(sdist.read_bytes()).hexdigest()
