"""Contract tests for the coordinated public-alpha distribution identity."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ALPHA_VERSION = "0.1.0a1"
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
SUBSTANTIVE_PIPELINE_MANIFESTS = frozenset(
    {
        "pipelines/analysis-bids/pipeline.yaml",
        "pipelines/preprocess-bids/pipeline.yaml",
    }
)
SCAFFOLD_PIPELINE_MANIFESTS = frozenset(
    {
        "pipelines/evaluate-model/pipeline.yaml",
        "pipelines/feature-engineering/pipeline.yaml",
        "pipelines/ingest/pipeline.yaml",
        "pipelines/publish-results/pipeline.yaml",
        "pipelines/qc-reporting/pipeline.yaml",
        "pipelines/train-model/pipeline.yaml",
    }
)
FULL_SOURCE_PIPELINE_MANIFESTS = (
    SUBSTANTIVE_PIPELINE_MANIFESTS | SCAFFOLD_PIPELINE_MANIFESTS
)
SUPPORTED_PIPELINE_MANIFEST_SETS = (
    SUBSTANTIVE_PIPELINE_MANIFESTS,
    FULL_SOURCE_PIPELINE_MANIFESTS,
)
INTERNAL_REQUIREMENTS = {
    "research-analysis": ("research-ml>=0.1.0a1,<0.2",),
    "research-bids": ("research-neuro>=0.1.0a1,<0.2",),
    "research-core": ("research-hpc>=0.1.0a1,<0.2",),
}


def _project_metadata(distribution: str) -> dict[str, object]:
    path = REPO_ROOT / "packages" / distribution / "pyproject.toml"
    with path.open("rb") as stream:
        return tomllib.load(stream)["project"]


@pytest.mark.parametrize("distribution", PACKAGE_NAMES)
def test_public_distributions_share_alpha_version(distribution: str) -> None:
    metadata = _project_metadata(distribution)

    assert metadata["name"] == distribution
    assert metadata["version"] == PUBLIC_ALPHA_VERSION


def test_every_internal_requirement_is_coordinated_and_accounted_for() -> None:
    observed: dict[str, tuple[str, ...]] = {}
    for distribution in PACKAGE_NAMES:
        metadata = _project_metadata(distribution)
        requirements = list(metadata.get("dependencies", ()))
        for optional_requirements in metadata.get("optional-dependencies", {}).values():
            requirements.extend(optional_requirements)
        internal = tuple(
            requirement
            for requirement in requirements
            if re.match(r"^research-[a-z]+(?:$|[<=>!~])", requirement)
        )
        if internal:
            observed[distribution] = internal

    assert observed == INTERNAL_REQUIREMENTS


def test_workspace_version_is_coordinated_platform_identity() -> None:
    workspace_text = (REPO_ROOT / "WORKSPACE.yaml").read_text(encoding="utf-8")

    assert re.search(
        rf"(?m)^workspace:\n  name: research-platform\n  version: {PUBLIC_ALPHA_VERSION}$",
        workspace_text,
    )


def test_independent_project_and_pipeline_versions_are_preserved() -> None:
    approved_overlays = (
        "project-template",
        "project-example",
        "project-pilot-bids",
        "project-pilot-tabular",
    )
    for overlay in approved_overlays:
        text = (REPO_ROOT / "project" / overlay / "project.yaml").read_text(
            encoding="utf-8"
        )
        assert re.search(r"(?m)^version: 0\.1\.0$", text)
        assert PUBLIC_ALPHA_VERSION not in text

    pipeline_manifests = frozenset(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "pipelines").glob("*/pipeline.yaml")
    )
    assert SUBSTANTIVE_PIPELINE_MANIFESTS <= pipeline_manifests
    assert pipeline_manifests in SUPPORTED_PIPELINE_MANIFEST_SETS

    if pipeline_manifests == SUBSTANTIVE_PIPELINE_MANIFESTS:
        for manifest in sorted(SCAFFOLD_PIPELINE_MANIFESTS):
            readme = REPO_ROOT / Path(manifest).parent / "README.md"
            assert readme.is_file()

    for relative_path in sorted(pipeline_manifests):
        path = REPO_ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        assert re.search(r"(?m)^version: 0\.1\.0$", text)
        assert PUBLIC_ALPHA_VERSION not in text


def test_independent_schema_dataset_and_generator_versions_are_preserved() -> None:
    bids_metadata = json.loads(
        (REPO_ROOT / "datasets/ds-bids-example/dataset_description.json").read_text(
            encoding="utf-8"
        )
    )
    mvpa_metadata = json.loads(
        (REPO_ROOT / "datasets/ds-mvpa-example/dataset_description.json").read_text(
            encoding="utf-8"
        )
    )
    mvpa_generator = (
        REPO_ROOT / "ops/scripts/generate_toy_mvpa_fixtures.py"
    ).read_text(encoding="utf-8")

    assert bids_metadata["BIDSVersion"] == "1.10.0"
    assert (
        mvpa_metadata["SchemaVersion"]
        == "research_platform.neuro.mvpa.materialized_pattern_table.v1"
    )
    assert '"generator_version": "toy-mvpa-generator-v1"' in mvpa_generator
    assert PUBLIC_ALPHA_VERSION not in bids_metadata.values()
    assert PUBLIC_ALPHA_VERSION not in mvpa_metadata.values()
