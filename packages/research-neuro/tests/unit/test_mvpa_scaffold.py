from __future__ import annotations

import csv
from io import StringIO

from research_platform.neuro.mvpa.config import validate_mvpa_set_document
from research_platform.neuro.mvpa.scaffold import (
    COMPARISONS_COLUMNS,
    CONDITIONS_COLUMNS,
    ROIS_COLUMNS,
    SUPPORTED_TEMPLATES,
    build_mvpa_config_scaffold,
    normalize_components,
)


def _tsv_rows(content: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(content), delimiter="\t"))


def test_scaffold_plan_builds_generic_runtime_and_specs() -> None:
    plan = build_mvpa_config_scaffold(
        analysis_id="baseline_crossnobis",
        analysis_label="BaselineCrossnobis",
        task="memory",
        session="01",
        direction="AP",
        template="distance-rdm",
        components="specs,runtime,tables,rdms,derivatives",
        condition_specs=["face:Face", "place:Place"],
        comparison_specs=["face_minus_place:face:place:Face vs place"],
        roi_specs=["visual:VisualROI"],
    )

    assert plan["valid"] is True
    assert plan["analysis_label"] == "BaselineCrossnobis"
    assert plan["comparison_mode"] == "explicit"
    assert plan["components"] == ["specs", "runtime", "tables", "rdms", "derivatives"]
    runtime = next(file for file in plan["files"] if file["component"] == "runtime")
    assert runtime["relative_path"] == "config/analysis/mvpa/baseline_crossnobis_distance_rdm.yaml"
    assert runtime["document"]["mvpa_set"]["scaffold_status"] == "not_ready"
    assert validate_mvpa_set_document(runtime["document"]) == []
    comparisons = _tsv_rows(next(file for file in plan["files"] if file["relative_path"].endswith("_comparisons.tsv"))["content"])
    assert list(comparisons[0]) == list(COMPARISONS_COLUMNS)
    assert comparisons[0]["comparison_id"] == "face_minus_place"
    assert comparisons[0]["notes"].endswith("exported contrast_id.")


def test_complete_comparison_mode_generates_all_condition_comparisons() -> None:
    plan = build_mvpa_config_scaffold(
        analysis_id="three_condition",
        analysis_label="ThreeCondition",
        comparison_mode="complete",
        template="distance-rdm",
        components="specs,runtime,rdms",
        condition_specs=["cond_a:Condition A", "cond_b:Condition B", "cond_c:Condition C"],
    )

    assert plan["valid"] is True
    assert [row["comparison_id"] for row in plan["condition_comparisons"]] == [
        "cond_a_minus_cond_b",
        "cond_a_minus_cond_c",
        "cond_b_minus_cond_c",
    ]
    runtime = next(file for file in plan["files"] if file["component"] == "runtime")
    assert runtime["document"]["mvpa_set"]["condition_pairs"]["mode"] == "all_pairs"
    rdm = next(file for file in plan["files"] if file["component"] == "rdms")
    assert rdm["document"]["mvpa_rdm_export"]["rdms"][0]["strict_all_pairs"] is True


def test_component_selection_can_emit_runtime_only() -> None:
    plan = build_mvpa_config_scaffold(
        analysis_id="runtime_only",
        analysis_label="RuntimeOnly",
        components="runtime",
    )

    assert plan["valid"] is True
    assert [file["component"] for file in plan["files"]] == ["runtime"]
    assert plan["dependencies"] == []
    assert plan["files"][0]["relative_path"] == "config/analysis/mvpa/runtime_only.yaml"


def test_generated_spec_tsv_columns_are_stable() -> None:
    plan = build_mvpa_config_scaffold(
        analysis_id="demo",
        analysis_label="Demo",
        template="distance-rdm",
    )
    specs = {file["relative_path"].split("_")[-1]: file for file in plan["files"] if file["component"] == "specs"}

    assert list(_tsv_rows(specs["conditions.tsv"]["content"])[0]) == list(CONDITIONS_COLUMNS)
    assert list(_tsv_rows(specs["comparisons.tsv"]["content"])[0]) == list(COMPARISONS_COLUMNS)
    assert list(_tsv_rows(specs["rois.tsv"]["content"])[0]) == list(ROIS_COLUMNS)


def test_invalid_labels_fail_cleanly() -> None:
    plan = build_mvpa_config_scaffold(analysis_id="bad path", analysis_label="Bad_Label")

    assert plan["valid"] is False
    assert "analysis_id must be a safe label." in plan["errors"]
    assert "analysis_label must contain only letters and digits for BIDS desc fields." in plan["errors"]


def test_component_aliases_remain_generic() -> None:
    errors: list[str] = []
    assert normalize_components("runtime,derivative-publisher", errors=errors) == ("runtime", "derivatives")
    assert errors == []


def test_materialized_crossnobis_is_default_one_yaml_scaffold() -> None:
    plan = build_mvpa_config_scaffold(analysis_id="prepared_demo")

    assert plan["valid"] is True
    assert plan["template"] == "materialized-crossnobis"
    assert plan["components"] == ["runtime"]
    assert [record["relative_path"] for record in plan["files"]] == [
        "config/analysis/mvpa/prepared_demo.yaml"
    ]
    document = plan["files"][0]["document"]
    payload = document["mvpa_set"]
    assert validate_mvpa_set_document(document) == []
    assert payload["name"] == "prepared_demo"
    assert payload["unit_selection"] == {
        "mode": "exact_units",
        "key_columns": ["subject_id", "run_id"],
    }
    assert "subjects" not in payload
    assert "sessions" not in payload
    assert "runs" not in payload
    assert payload["pattern_sources"] == [
        {
            "name": "prepared-patterns",
            "backend": "materialized_pattern_table",
            "root_ref": "mvpa_inputs",
            "path": "patterns.tsv",
            "schema_version": "research_platform.neuro.mvpa.materialized_pattern_table.v1",
        }
    ]
    assert payload["roi_sources"] == [
        {
            "name": "prepared-rois",
            "source": "materialized_features",
            "roi_labels": ["SeedA"],
            "feature_space_id": "example-feature-space",
            "roi_definition_id": "example-roi-definition",
        }
    ]
    assert payload["runtime"] == {"existing_output": "fail"}
    assert all("mask" not in key for key in payload["roi_sources"][0])


def test_all_templates_are_discoverable_and_structurally_valid() -> None:
    assert SUPPORTED_TEMPLATES == (
        "materialized-crossnobis",
        "fsl-feat-crossnobis",
        "distance-rdm",
    )
    for template in SUPPORTED_TEMPLATES:
        plan = build_mvpa_config_scaffold(analysis_id="template_demo", template=template)
        runtime = next(record for record in plan["files"] if record["component"] == "runtime")
        assert plan["valid"] is True
        assert validate_mvpa_set_document(runtime["document"]) == []


def test_materialized_template_advanced_condition_options_remain_compatible() -> None:
    plan = build_mvpa_config_scaffold(
        analysis_id="custom_demo",
        condition_specs=["condition-x:Condition X", "condition-y:Condition Y"],
        comparison_specs=["x_minus_y:condition-x:condition-y"],
        roi_specs=["unused:AdvancedSpecOnly"],
    )

    payload = plan["files"][0]["document"]["mvpa_set"]
    assert [condition["id"] for condition in payload["conditions"]] == [
        "condition-x",
        "condition-y",
    ]
    assert payload["condition_pairs"][0]["id"] == "x_minus_y"


def test_not_ready_materialized_scaffold_still_enforces_structural_contract() -> None:
    plan = build_mvpa_config_scaffold(analysis_id="prepared_demo")
    document = plan["files"][0]["document"]
    payload = document["mvpa_set"]

    del payload["pattern_sources"][0]["schema_version"]
    payload["unit_selection"] = {"mode": "legacy_cartesian"}

    errors = validate_mvpa_set_document(document)

    assert any("schema_version" in error for error in errors)
    assert any("unit_selection.mode=exact_units" in error for error in errors)
