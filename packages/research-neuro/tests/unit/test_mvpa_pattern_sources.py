from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.mvpa.config import (
    parse_mvpa_set_document,
    validate_mvpa_set_document,
)
from research_platform.neuro.mvpa.pattern_sources import (
    ALLOWED_PATTERN_BACKENDS,
    PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE,
    PATTERN_BACKEND_FSL_FEAT_PE,
    PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE,
    PATTERN_BACKEND_NILEARN_GLM,
    PATTERN_BACKEND_SURFACE_CIFTI,
    AnalysisUnitResolution,
    PatternSourceAdapterCapabilities,
    PatternSourceAdapterPlan,
    PatternSourceAdapterRegistry,
    UNIT_SELECTION_EXACT,
    UNIT_SELECTION_LEGACY_CARTESIAN,
    resolve_analysis_units,
)


@dataclass(frozen=True)
class _FakeAdapter:
    name: str = PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE
    status: str = "test-only"
    capabilities: PatternSourceAdapterCapabilities = PatternSourceAdapterCapabilities(
        status="test-only",
        schema_supported=True,
        planning_supported=True,
        execution_ready=True,
        representation_kinds=("prepared_features",),
    )

    def validate_source(self, source: dict[str, Any], label: str) -> tuple[str, ...]:
        marker = source.get("adapter_marker")
        return () if marker == "accepted" else (f"{label}.adapter_marker must be accepted.",)

    def plan_source(self, **_: Any) -> PatternSourceAdapterPlan:
        return PatternSourceAdapterPlan(
            adapter_name=self.name,
            status="valid",
            ready_for_execution=True,
        )


def _registry() -> PatternSourceAdapterRegistry:
    return PatternSourceAdapterRegistry((_FakeAdapter(),))


def _document(*, exact: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "neutral_mvpa",
        "conditions": [{"id": "condition_a"}, {"id": "condition_b"}],
        "pattern_sources": [
            {
                "name": "patterns",
                "backend": PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE,
                "adapter_marker": "accepted",
            }
        ],
        "roi_sources": [
            {"name": "rois", "source": "roi_set", "roi_set_ref": "neutral_rois"}
        ],
        "distance": {
            "metrics": ["crossnobis"],
            "engine": "native_reference",
            "cross_validation": {"unit": "run"},
        },
        "outputs": {
            "runtime_root": {
                "root_ref": "artifact_root",
                "path": ".research-platform/mvpa/{mvpa_set}",
            }
        },
    }
    if exact:
        payload["unit_selection"] = {
            "mode": UNIT_SELECTION_EXACT,
            "key_columns": ["subject_id", "session_id", "run_id"],
        }
    else:
        payload.update(
            {
                "subjects": ["sub-alpha", "sub-beta"],
                "sessions": ["ses-early"],
                "runs": ["run-01", "run-02"],
                "entities": {"task": "exampletask"},
            }
        )
    return {"mvpa_set": payload}


def test_registry_is_explicit_ordered_and_immutable() -> None:
    adapter = _FakeAdapter()
    registry = PatternSourceAdapterRegistry((adapter,))

    assert registry.names == (PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE,)
    assert registry.adapter(PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE) is adapter
    assert registry.adapter("not_registered") is None
    assert ALLOWED_PATTERN_BACKENDS == {
        PATTERN_BACKEND_FSL_FEAT_PE,
        PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE,
        PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE,
        PATTERN_BACKEND_NILEARN_GLM,
        PATTERN_BACKEND_SURFACE_CIFTI,
    }
    assert all("spm" not in backend for backend in ALLOWED_PATTERN_BACKENDS)

    with pytest.raises(ValueError, match="duplicate names"):
        registry.with_adapter(_FakeAdapter())
    with pytest.raises(AttributeError):
        registry._adapters = ()  # type: ignore[misc]


def test_config_validation_dispatches_to_injected_adapter() -> None:
    document = _document(exact=False)
    assert validate_mvpa_set_document(document, adapter_registry=_registry()) == []

    document["mvpa_set"]["pattern_sources"][0]["adapter_marker"] = "rejected"
    errors = validate_mvpa_set_document(document, adapter_registry=_registry())

    assert errors == [
        "mvpa_set.pattern_sources[0].adapter_marker must be accepted."
    ]


def test_legacy_mode_remains_explicit_cartesian_compatibility() -> None:
    config = parse_mvpa_set_document(_document(exact=False), adapter_registry=_registry())
    resolution = resolve_analysis_units(config)

    assert config.unit_selection.mode == UNIT_SELECTION_LEGACY_CARTESIAN
    assert resolution.valid
    assert resolution.mode == UNIT_SELECTION_LEGACY_CARTESIAN
    assert [
        (unit.subject_id, unit.session_id, unit.task_id, unit.run_id)
        for unit in resolution.units
    ] == [
        ("sub-alpha", "ses-early", "exampletask", "run-01"),
        ("sub-alpha", "ses-early", "exampletask", "run-02"),
        ("sub-beta", "ses-early", "exampletask", "run-01"),
        ("sub-beta", "ses-early", "exampletask", "run-02"),
    ]


def test_cross_sectional_exact_units_preserve_order_values_and_metadata() -> None:
    document = _document(exact=True)
    document["mvpa_set"]["unit_selection"]["key_columns"] = ["subject_id"]
    config = parse_mvpa_set_document(document, adapter_registry=_registry())
    rows = [
        {
            "source_row": 8,
            "values": {
                "subject_id": "sub-zeta",
                "cohort_id": "group-b",
                "qc_status": "pass",
            },
        },
        {
            "source_row": 3,
            "values": {
                "subject_id": "sub-alpha",
                "cohort_id": "group-a",
                "qc_status": "review",
            },
        },
    ]

    resolution = resolve_analysis_units(
        config,
        exact_units=rows,
        unit_key_columns=("subject_id",),
    )

    assert isinstance(resolution, AnalysisUnitResolution)
    assert resolution.valid
    assert resolution.mode == UNIT_SELECTION_EXACT
    assert [unit.source_row for unit in resolution.units] == [8, 3]
    assert [unit.subject_id for unit in resolution.units] == ["sub-zeta", "sub-alpha"]
    assert all(unit.session_id is None for unit in resolution.units)
    assert all(unit.task_id is None for unit in resolution.units)
    assert all(unit.run_id is None for unit in resolution.units)
    assert resolution.units[0].metadata["cohort_id"] == "group-b"
    assert resolution.units[1].metadata["qc_status"] == "review"


def test_irregular_longitudinal_units_are_not_cartesian_expanded() -> None:
    config = parse_mvpa_set_document(_document(exact=True), adapter_registry=_registry())
    rows = [
        {
            "subject_id": "sub-alpha",
            "session_id": "ses-late",
            "task_id": "exampletask",
            "run_id": "run-03",
            "visit_index": "2",
            "acquisition_id": "acq-b",
        },
        {
            "subject_id": "sub-alpha",
            "session_id": "ses-early",
            "task_id": "exampletask",
            "run_id": "run-01",
            "visit_index": "1",
            "acquisition_id": "acq-a",
        },
        {
            "subject_id": "sub-alpha",
            "session_id": "ses-early",
            "task_id": "exampletask",
            "run_id": "run-02",
            "visit_index": "1",
            "acquisition_id": "acq-b",
        },
        {
            "subject_id": "sub-beta",
            "session_id": "ses-only",
            "task_id": "exampletask",
            "run_id": "run-01",
            "visit_index": "1",
            "acquisition_id": "acq-a",
        },
    ]

    resolution = resolve_analysis_units(
        config,
        exact_units=rows,
        unit_key_columns=("subject_id", "session_id", "run_id"),
    )

    assert resolution.valid
    assert len(resolution.units) == len(rows)
    assert [
        (unit.subject_id, unit.session_id, unit.run_id)
        for unit in resolution.units
    ] == [
        ("sub-alpha", "ses-late", "run-03"),
        ("sub-alpha", "ses-early", "run-01"),
        ("sub-alpha", "ses-early", "run-02"),
        ("sub-beta", "ses-only", "run-01"),
    ]
    assert resolution.units[0].metadata["visit_index"] == "2"
    assert resolution.units[2].metadata["acquisition_id"] == "acq-b"


def test_exact_unit_duplicates_and_key_contract_mismatches_are_rejected() -> None:
    config = parse_mvpa_set_document(_document(exact=True), adapter_registry=_registry())
    repeated = {
        "subject_id": "sub-alpha",
        "session_id": "ses-01",
        "run_id": "run-01",
    }

    resolution = resolve_analysis_units(
        config,
        exact_units=(repeated, dict(repeated)),
        unit_key_columns=("subject_id", "run_id"),
    )

    assert not resolution.valid
    assert any("must match" in error for error in resolution.errors)
    assert any("Duplicate exact-unit key" in error for error in resolution.errors)


def test_exact_config_rejects_inline_selectors_and_invalid_keys() -> None:
    document = _document(exact=True)
    document["mvpa_set"]["subjects"] = ["sub-alpha"]
    document["mvpa_set"]["unit_selection"]["key_columns"] = ["run_id", "run_id"]

    errors = validate_mvpa_set_document(document, adapter_registry=_registry())

    assert any("must not be mixed" in error for error in errors)
    assert any("duplicate column" in error for error in errors)
    assert any("must include subject_id" in error for error in errors)


def test_legacy_config_rejects_caller_supplied_exact_rows() -> None:
    config = parse_mvpa_set_document(_document(exact=False), adapter_registry=_registry())

    resolution = resolve_analysis_units(
        config,
        exact_units=({"subject_id": "sub-alpha"},),
        unit_key_columns=("subject_id",),
    )

    assert not resolution.valid
    assert resolution.units == ()
    assert any("must not be mixed" in error for error in resolution.errors)
