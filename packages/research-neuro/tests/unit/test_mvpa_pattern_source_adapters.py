from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import copy
import subprocess
import sys
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro._roi_path_safety import published_value_local_path_fields
from research_platform.neuro.mvpa.config import parse_mvpa_set_document
from research_platform.neuro.mvpa.fsl_feat import plan_fsl_feat_pattern_source
from research_platform.neuro.mvpa.pattern_source_adapters import (
    default_pattern_source_adapter_registry,
    runtime_unit_contexts,
)
from research_platform.neuro.mvpa.pattern_sources import (
    PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE,
    PATTERN_BACKEND_FSL_FEAT_PE,
    PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE,
    PATTERN_BACKEND_NILEARN_GLM,
    PATTERN_BACKEND_SURFACE_CIFTI,
    REPRESENTATION_IMAGE,
    UNIT_SELECTION_EXACT,
    UNIT_SELECTION_LEGACY_CARTESIAN,
    PatternSourceAdapterCapabilities,
    PatternSourceAdapterPlan,
    PatternSourceAdapterRegistry,
    PlannedPatternRow,
    ResolvedAnalysisUnit,
    resolve_analysis_units,
)
from research_platform.neuro.mvpa.plan import plan_mvpa_discovery


_DEFERRED_BACKENDS = (
    PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE,
    PATTERN_BACKEND_NILEARN_GLM,
    PATTERN_BACKEND_SURFACE_CIFTI,
)


class _RecordingAdapter:
    name = PATTERN_BACKEND_NILEARN_GLM
    status = "test_available"
    capabilities = PatternSourceAdapterCapabilities(
        status="test_available",
        schema_supported=True,
        planning_supported=True,
        execution_ready=True,
        representation_kinds=(REPRESENTATION_IMAGE,),
        reason="Test-only adapter for registry dispatch.",
    )

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[ResolvedAnalysisUnit, ...]]] = []

    def validate_source(self, source: Mapping[str, Any], label: str) -> tuple[str, ...]:
        del source, label
        return ()

    def plan_source(
        self,
        *,
        config: Any,
        source: Any,
        units: Sequence[ResolvedAnalysisUnit],
        roots: Mapping[str, str | Path] | None,
        context: Mapping[str, Any] | None,
        raise_on_fail_policy: bool = False,
    ) -> PatternSourceAdapterPlan:
        del roots, context, raise_on_fail_policy
        ordered_units = tuple(units)
        self.calls.append((source.name, ordered_units))
        rows = tuple(
            PlannedPatternRow(
                unit_id=unit.unit_id,
                subject_id=unit.subject_id,
                session_id=unit.session_id,
                task_id=unit.task_id,
                run_id=unit.run_id,
                cross_validation_label=unit.run_id or unit.session_id or unit.subject_id,
                condition_id=condition.id,
                source_name=source.name,
                backend_name=self.name,
                representation_kind=REPRESENTATION_IMAGE,
                pattern_reference=(
                    f"root_ref:pattern_root/{unit.unit_id}/condition-{condition.id}.tsv"
                ),
                unit_metadata=dict(unit.metadata),
                backend_metadata={"adapter_marker": "recording"},
                status="valid",
            )
            for unit in ordered_units
            for condition in config.conditions
        )
        return PatternSourceAdapterPlan(
            adapter_name=self.name,
            status="valid",
            ready_for_execution=True,
            pattern_rows=rows,
            executed=False,
        )


def _exact_document(
    *,
    backend: str = PATTERN_BACKEND_NILEARN_GLM,
    key_columns: Sequence[str] = ("subject_id",),
    conditions: Sequence[tuple[str, str]] = (("condition_a", "Condition A"),),
    cv_unit: str = "subject",
    fsl: bool = False,
    roi_template: str = "{subject_dir}/label-{roi_label}_mask.nii",
) -> dict[str, Any]:
    if fsl:
        pattern_source: dict[str, Any] = {
            "name": "first_level_patterns",
            "backend": PATTERN_BACKEND_FSL_FEAT_PE,
            "root_ref": "feat_root",
            "feat_dir_template": (
                "{subject_dir}/{session_dir}/func/"
                "{subject_dir}_{session_dir}_task-{task_id}_{run_entity}_model-{model}.feat"
            ),
            "design_file": "design.fsf",
            "pe_image_template": "stats/pe{pe_number}.nii.gz",
            "noise_image_template": "stats/sigmasquareds.nii.gz",
        }
    else:
        pattern_source = {
            "name": "prepared_patterns",
            "backend": backend,
            "root_ref": "pattern_root",
            "path": "tables/patterns.tsv",
        }

    return {
        "mvpa_set": {
            "name": "neutral_mvpa",
            "unit_selection": {
                "mode": UNIT_SELECTION_EXACT,
                "key_columns": list(key_columns),
            },
            "entities": {"model": "modelA"},
            "conditions": [
                {"id": condition_id, "fsl_ev_title": title}
                for condition_id, title in conditions
            ],
            "pattern_sources": [pattern_source],
            "roi_sources": [
                {
                    "name": "explicit_rois",
                    "source": "explicit_masks",
                    "root_ref": "roi_root",
                    "roi_labels": ["SeedA"],
                    "mask_template": roi_template,
                }
            ],
            "distance": {
                "metrics": ["crossnobis"],
                "engine": "native_reference",
                "cross_validation": {"unit": cv_unit},
                "noise_normalization": {
                    "method": "diagonal",
                    "variance_source": "sigmasquareds",
                },
            },
            "outputs": {
                "runtime_root": {
                    "root_ref": "artifact_root",
                    "path": ".research-platform/mvpa/{mvpa_set}",
                }
            },
            "missing_input_policy": "warn",
        }
    }


def _legacy_fsl_document() -> dict[str, Any]:
    document = _exact_document(
        fsl=True,
        key_columns=("subject_id", "session_id", "run_id"),
        conditions=(("condition_a", "Condition A"), ("condition_b", "Condition B")),
        cv_unit="run",
        roi_template=(
            "{subject_dir}/{session_dir}/"
            "{subject_dir}_{session_dir}_task-{task_id}_{run_entity}_label-{roi_label}_mask.nii"
        ),
    )
    payload = document["mvpa_set"]
    payload.pop("unit_selection")
    payload["subjects"] = ["sub-toy01"]
    payload["sessions"] = ["ses-02"]
    payload["runs"] = ["run-03"]
    payload["entities"]["task"] = "exampletask"
    return document


def _adapter_registry(adapter: _RecordingAdapter) -> PatternSourceAdapterRegistry:
    return PatternSourceAdapterRegistry((adapter,))


def _fsl_feat_dir(root: Path, unit: Mapping[str, str]) -> Path:
    subject = unit["subject_id"]
    session = unit["session_id"]
    task = unit["task_id"]
    run = unit["run_id"]
    return (
        root
        / subject
        / session
        / "func"
        / f"{subject}_{session}_task-{task}_{run}_model-modelA.feat"
    )


def _write_fsl_inputs(
    root: Path,
    unit: Mapping[str, str],
    *,
    conditions: Sequence[tuple[str, str]],
) -> None:
    feat_dir = _fsl_feat_dir(root, unit)
    stats_dir = feat_dir / "stats"
    stats_dir.mkdir(parents=True)
    design = "".join(
        f'set fmri(evtitle{index}) "{title}"\n'
        for index, (_condition_id, title) in enumerate(conditions, start=1)
    )
    (feat_dir / "design.fsf").write_text(design, encoding="utf-8", newline="\n")
    for index in range(1, len(conditions) + 1):
        (stats_dir / f"pe{index}.nii.gz").write_bytes(b"")
    (stats_dir / "sigmasquareds.nii.gz").write_bytes(b"")


def _write_roi_input(root: Path, unit: Mapping[str, str]) -> None:
    subject = unit["subject_id"]
    session = unit.get("session_id")
    task = unit.get("task_id")
    run = unit.get("run_id")
    if session is None:
        acquisition = unit["acquisition_id"]
        mask = root / subject / acquisition / "label-SeedA_mask.nii"
    else:
        mask = (
            root
            / subject
            / session
            / f"{subject}_{session}_task-{task}_{run}_label-SeedA_mask.nii"
        )
    mask.parent.mkdir(parents=True, exist_ok=True)
    mask.write_bytes(b"")
    mask.with_suffix(".json").write_text("{}\n", encoding="utf-8", newline="\n")


def _relative_files(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    )


def _without_annotations(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"source_type", "source_name", "planner"}
    }


def test_default_registry_is_authoritative_and_does_not_claim_spm() -> None:
    registry = default_pattern_source_adapter_registry()

    assert registry.names == (
        PATTERN_BACKEND_FSL_FEAT_PE,
        PATTERN_BACKEND_MATERIALIZED_PATTERN_TABLE,
        PATTERN_BACKEND_BIDS_DERIVATIVE_PATTERN_TABLE,
        PATTERN_BACKEND_NILEARN_GLM,
        PATTERN_BACKEND_SURFACE_CIFTI,
    )
    assert len(registry.adapters) == 5
    assert registry.adapter("spm_beta_images") is None
    assert all("spm" not in name.casefold() for name in registry.names)


def test_injected_adapter_dispatches_without_a_core_backend_branch() -> None:
    adapter = _RecordingAdapter()
    registry = _adapter_registry(adapter)
    units = (
        {"subject_id": "sub-toy02", "qc_status": "pass"},
        {"subject_id": "sub-toy01", "qc_status": "review"},
    )

    plan = plan_mvpa_discovery(
        _exact_document(),
        roots={},
        exact_units=units,
        unit_key_columns=("subject_id",),
        adapter_registry=registry,
    )

    assert len(adapter.calls) == 1
    assert adapter.calls[0][0] == "prepared_patterns"
    assert [unit.subject_id for unit in adapter.calls[0][1]] == ["sub-toy02", "sub-toy01"]
    assert [row["subject_id"] for row in plan.pattern_rows] == ["sub-toy02", "sub-toy01"]
    assert [row["unit_metadata"]["qc_status"] for row in plan.pattern_rows] == [
        "pass",
        "review",
    ]


@pytest.mark.parametrize("backend", _DEFERRED_BACKENDS)
def test_deferred_backends_validate_but_are_not_execution_ready(backend: str) -> None:
    plan = plan_mvpa_discovery(
        _exact_document(backend=backend),
        roots={},
        exact_units=({"subject_id": "sub-toy01"},),
        unit_key_columns=("subject_id",),
    )

    assert plan.schema_valid
    assert not plan.ready_for_execution
    assert plan.pattern_sources[0].status == "deferred"
    assert plan.pattern_rows == ()
    assert len(plan.adapter_availability) == 1
    availability = plan.adapter_availability[0]
    assert availability["status"] == "deferred"
    assert availability["available"] is False
    assert availability["planning_supported"] is False
    assert availability["execution_available"] is False
    assert availability["ready_for_execution"] is False


def test_integrated_fsl_rows_are_canonical_portable_and_compatibility_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conditions = (("condition_a", "Condition A"), ("condition_b", "Condition B"))
    unit = {
        "subject_id": "sub-toy01",
        "session_id": "ses-02",
        "task_id": "exampletask",
        "run_id": "run-03",
        "visit_index": "2",
        "acquisition_id": "acq-b",
        "pe_number": "source-metadata",
    }
    document = _exact_document(
        fsl=True,
        key_columns=("subject_id", "session_id", "run_id"),
        conditions=conditions,
        cv_unit="run",
        roi_template=(
            "{subject_dir}/{session_dir}/"
            "{subject_dir}_{session_dir}_task-{task_id}_{run_entity}_label-{roi_label}_mask.nii"
        ),
    )
    feat_root = tmp_path / "feat"
    roi_root = tmp_path / "roi"
    _write_fsl_inputs(feat_root, unit, conditions=conditions)
    _write_roi_input(roi_root, unit)
    before = _relative_files(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("plan mode invoked subprocess.run"),
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("plan mode invoked subprocess.Popen"),
    )

    plan = plan_mvpa_discovery(
        document,
        roots={"feat_root": feat_root, "roi_root": roi_root},
        exact_units=(unit,),
        unit_key_columns=("subject_id", "session_id", "run_id"),
    )

    assert _relative_files(tmp_path) == before
    assert plan.schema_valid
    assert not plan.ready_for_execution
    assert plan.adapter_availability[0]["planning_supported"] is True
    assert plan.adapter_availability[0]["execution_available"] is False
    assert "portable" in str(plan.adapter_availability[0]["reason"]).lower()
    assert not plan.executed
    assert len(plan.analysis_units) == 1
    assert len(plan.pattern_rows) == len(conditions)
    assert [row["condition_id"] for row in plan.pattern_rows] == [
        "condition_a",
        "condition_b",
    ]
    for index, row in enumerate(plan.pattern_rows, start=1):
        assert row["subject_id"] == "sub-toy01"
        assert row["session_id"] == "ses-02"
        assert row["task_id"] == "exampletask"
        assert row["run_id"] == "run-03"
        assert row["cross_validation_label"] == "run-03"
        assert row["representation_kind"] == "image"
        assert row["pattern_reference"].startswith("root_ref:feat_root/")
        assert row["pattern_reference"].endswith(f"stats/pe{index}.nii.gz")
        assert row["noise_reference"].startswith("root_ref:feat_root/")
        assert row["unit_metadata"]["visit_index"] == "2"
        assert "pe_number" not in row["unit_metadata"]
        assert published_value_local_path_fields(row, label="pattern_row") == ()
        assert not {
            "ev_index",
            "pe_number",
            "pe_image",
            "feat_dir",
            "design_fsf",
        }.intersection(row)
        assert row["backend_metadata"]["ev_index"] == index
        assert row["backend_metadata"]["pe_number"] == index
        assert row["backend_metadata"]["unit_metadata"]["pe_number"] == "source-metadata"
        assert row["backend_metadata"]["feat_directory_reference"].startswith(
            "root_ref:feat_root/"
        )
        assert row["backend_metadata"]["design_reference"].startswith(
            "root_ref:feat_root/"
        )

    config = parse_mvpa_set_document(document)
    resolution = resolve_analysis_units(
        config,
        exact_units=(unit,),
        unit_key_columns=("subject_id", "session_id", "run_id"),
    )
    direct = plan_fsl_feat_pattern_source(
        config,
        roots={"feat_root": feat_root, "roi_root": roi_root},
        unit_contexts=runtime_unit_contexts(config, resolution.units, context=None),
    )
    assert [_without_annotations(row) for row in plan.pattern_source_rows] == [
        row.to_dict() for row in direct.units
    ]
    assert [_without_annotations(row) for row in plan.condition_pe_rows] == [
        row.to_dict() for row in direct.condition_pe_rows
    ]


def test_exact_cross_sectional_units_reach_pattern_and_roi_planners_in_source_order(
    tmp_path: Path,
) -> None:
    adapter = _RecordingAdapter()
    units = (
        {
            "source_row": 9,
            "values": {
                "subject_id": "sub-toy02",
                "acquisition_id": "acq-b",
                "qc_status": "review",
            },
        },
        {
            "source_row": 4,
            "values": {
                "subject_id": "sub-toy01",
                "acquisition_id": "acq-a",
                "qc_status": "pass",
            },
        },
    )
    for row in units:
        _write_roi_input(tmp_path / "roi", row["values"])

    plan = plan_mvpa_discovery(
        _exact_document(
            roi_template="{subject_dir}/{acquisition_id}/label-{roi_label}_mask.nii"
        ),
        roots={"roi_root": tmp_path / "roi"},
        exact_units=units,
        unit_key_columns=("subject_id",),
        adapter_registry=_adapter_registry(adapter),
    )

    assert plan.unit_selection_mode == UNIT_SELECTION_EXACT
    assert [row["source_row"] for row in plan.analysis_units] == [9, 4]
    assert [row["subject_id"] for row in plan.pattern_rows] == ["sub-toy02", "sub-toy01"]
    assert all(row["session_id"] is None for row in plan.pattern_rows)
    assert all(row["task_id"] is None for row in plan.pattern_rows)
    assert all(row["run_id"] is None for row in plan.pattern_rows)
    assert [row["subject_id"] for row in plan.roi_source_rows] == ["toy02", "toy01"]
    assert all(row["session_id"] == "" for row in plan.roi_source_rows)
    assert all(row["run_id"] == "" for row in plan.roi_source_rows)
    assert [Path(str(row["mask_path"])).parent.name for row in plan.roi_source_rows] == [
        "acq-b",
        "acq-a",
    ]


def test_irregular_exact_units_with_multiple_runs_are_never_cartesian_expanded() -> None:
    adapter = _RecordingAdapter()
    document = _exact_document(
        key_columns=("subject_id", "session_id", "run_id"),
        conditions=(("condition_a", "Condition A"), ("condition_b", "Condition B")),
        cv_unit="run",
    )
    units = (
        {
            "subject_id": "sub-toy02",
            "session_id": "ses-late",
            "task_id": "exampletask",
            "run_id": "run-03",
            "visit_index": "2",
        },
        {
            "subject_id": "sub-toy01",
            "session_id": "ses-early",
            "task_id": "exampletask",
            "run_id": "run-01",
            "visit_index": "1",
        },
        {
            "subject_id": "sub-toy01",
            "session_id": "ses-early",
            "task_id": "exampletask",
            "run_id": "run-02",
            "visit_index": "1",
        },
    )

    plan = plan_mvpa_discovery(
        document,
        roots={},
        exact_units=units,
        unit_key_columns=("subject_id", "session_id", "run_id"),
        adapter_registry=_adapter_registry(adapter),
    )

    assert len(adapter.calls[0][1]) == len(units)
    assert len(plan.analysis_units) == len(units)
    assert len(plan.pattern_rows) == len(units) * 2
    assert [
        (row["subject_id"], row["session_id"], row["run_id"], row["condition_id"])
        for row in plan.pattern_rows
    ] == [
        (unit["subject_id"], unit["session_id"], unit["run_id"], condition_id)
        for unit in units
        for condition_id in ("condition_a", "condition_b")
    ]
    assert [row["unit_metadata"]["visit_index"] for row in plan.pattern_rows] == [
        "2",
        "2",
        "1",
        "1",
        "1",
        "1",
    ]


def test_duplicate_and_mixed_unit_selection_fail_before_adapter_planning() -> None:
    adapter = _RecordingAdapter()
    registry = _adapter_registry(adapter)
    duplicate = {"subject_id": "sub-toy01"}

    duplicate_plan = plan_mvpa_discovery(
        _exact_document(),
        roots={},
        exact_units=(duplicate, copy.deepcopy(duplicate)),
        unit_key_columns=("subject_id",),
        adapter_registry=registry,
    )
    assert duplicate_plan.schema_valid
    assert duplicate_plan.status == "error"
    assert any("Duplicate exact-unit key" in error for error in duplicate_plan.errors)
    assert adapter.calls == []

    mixed_exact = _exact_document()
    mixed_exact["mvpa_set"]["subjects"] = ["sub-toy01"]
    mixed_plan = plan_mvpa_discovery(
        mixed_exact,
        roots={},
        exact_units=(duplicate,),
        unit_key_columns=("subject_id",),
        adapter_registry=registry,
    )
    assert not mixed_plan.schema_valid
    assert any("must not be mixed" in error for error in mixed_plan.errors)
    assert adapter.calls == []

    legacy_document = _legacy_fsl_document()
    legacy_mixed = plan_mvpa_discovery(
        legacy_document,
        roots={},
        exact_units=(duplicate,),
        unit_key_columns=("subject_id",),
    )
    assert legacy_mixed.schema_valid
    assert legacy_mixed.unit_selection_mode == UNIT_SELECTION_LEGACY_CARTESIAN
    assert any("must not be mixed" in error for error in legacy_mixed.errors)


def test_legacy_fsl_compatibility_rows_and_selector_fields_are_unchanged(tmp_path: Path) -> None:
    document = _legacy_fsl_document()
    unit = {
        "subject_id": "sub-toy01",
        "session_id": "ses-02",
        "task_id": "exampletask",
        "run_id": "run-03",
    }
    conditions = (("condition_a", "Condition A"), ("condition_b", "Condition B"))
    feat_root = tmp_path / "feat"
    roi_root = tmp_path / "roi"
    _write_fsl_inputs(feat_root, unit, conditions=conditions)
    _write_roi_input(roi_root, unit)

    direct = plan_fsl_feat_pattern_source(document, roots={"feat_root": feat_root})
    integrated = plan_mvpa_discovery(
        document,
        roots={"feat_root": feat_root, "roi_root": roi_root},
    )

    assert integrated.unit_selection_mode == UNIT_SELECTION_LEGACY_CARTESIAN
    assert integrated.subjects == ("sub-toy01",)
    assert integrated.sessions == ("ses-02",)
    assert integrated.runs == ("run-03",)
    assert [_without_annotations(row) for row in integrated.pattern_source_rows] == [
        row.to_dict() for row in direct.units
    ]
    assert [_without_annotations(row) for row in integrated.condition_pe_rows] == [
        row.to_dict() for row in direct.condition_pe_rows
    ]
