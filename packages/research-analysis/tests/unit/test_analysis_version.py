from __future__ import annotations

from importlib import metadata

from research_platform.analysis import _version


def test_package_version_queries_research_analysis_distribution(monkeypatch) -> None:
    requested: list[str] = []

    def installed_version(distribution_name: str) -> str:
        requested.append(distribution_name)
        return "0.1.0a1"

    monkeypatch.setattr(_version.metadata, "version", installed_version)

    assert _version.package_version() == "0.1.0a1"
    assert requested == ["research-analysis"]


def test_package_version_uses_source_checkout_fallback_when_metadata_is_unavailable(monkeypatch) -> None:
    def unavailable(distribution_name: str) -> str:
        raise metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(_version.metadata, "version", unavailable)

    assert _version.DISTRIBUTION_NAME == "research-analysis"
    assert _version.package_version() == "0.1.0a1"


def test_package_version_is_unknown_when_distribution_and_source_metadata_are_unavailable(monkeypatch) -> None:
    def unavailable(distribution_name: str) -> str:
        raise metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(_version.metadata, "version", unavailable)
    monkeypatch.setattr(_version, "_source_tree_version", lambda: _version.UNKNOWN_VERSION)

    assert _version.package_version() == "unknown"
