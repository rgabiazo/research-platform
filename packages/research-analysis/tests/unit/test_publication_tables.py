from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

import pytest

import research_platform.analysis.publication_tables as publication_tables
from research_platform.analysis.publication_tables import (
    ConfidenceIntervalFormatSpec,
    NumericFormatSpec,
    PValueFormatSpec,
    PublicationColumnSpec,
    PublicationFormatSpec,
    PublicationOutputSpec,
    PublicationSourceSpec,
    PublicationTableSpec,
    build_publication_table_rows,
    plan_publication_table_outputs,
    write_publication_table_outputs,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "group_label": "roi-beta",
            "effect_label": "contrast-beta",
            "measure": "measure-alpha",
            "n": 8,
            "mean": 0.12345,
            "ci_low": 0.01234,
            "ci_high": 0.23456,
            "sd": 0.5,
            "se": 0.1,
            "p_method": "sign_flip",
            "p_value": 0.0004,
            "q_method": "benjamini_hochberg",
            "q_value": 0.0456,
            "effect_size": 0.789,
            "effect_size_type": "dz",
            "percent_positive": 87.5,
            "loo_min": 0.01,
            "loo_max": 0.22,
            "status": "ok",
            "warnings": [],
        },
        {
            "group_label": "roi-alpha",
            "effect_label": "contrast-alpha",
            "measure": "measure-alpha",
            "n": 4,
            "mean": 1.234,
            "ci_low": 0.9,
            "ci_high": 1.5,
            "sd": 0.2,
            "se": 0.05,
            "p_method": "sign_flip",
            "p_value": 0.045,
            "q_method": "benjamini_hochberg",
            "q_value": 0.06,
            "effect_size": 0.5,
            "effect_size_type": "dz",
            "percent_positive": 75.0,
            "loo_min": 1.0,
            "loo_max": 1.4,
            "status": "ok",
            "warnings": [],
        },
        {
            "group_label": "roi-gamma",
            "effect_label": "contrast-gamma",
            "measure": "measure-alpha",
            "n": 2,
            "mean": -1.0,
            "ci_low": -2.0,
            "ci_high": 0.0,
            "sd": 0.1,
            "p_method": "sign_flip",
            "p_value": 1.0,
            "q_method": "benjamini_hochberg",
            "q_value": 1.0,
            "effect_size": -0.1,
            "effect_size_type": "dz",
            "percent_positive": 0.0,
            "status": "failed",
            "warnings": ["synthetic failure"],
        },
    ]


def _mvpa_style_spec() -> PublicationTableSpec:
    return PublicationTableSpec(
        table_id="generic-main-table",
        columns=(
            PublicationColumnSpec(output_name="ROI", source="group_label"),
            PublicationColumnSpec(output_name="Contrast", source="effect_label"),
            PublicationColumnSpec(
                output_name="N",
                source="n",
                column_type="numeric",
                numeric_format=NumericFormatSpec(precision=0),
            ),
            PublicationColumnSpec(
                output_name="Crossnobis M [95% CI]",
                source="mean",
                column_type="confidence_interval",
                ci_low_source="ci_low",
                ci_high_source="ci_high",
                confidence_interval_format=ConfidenceIntervalFormatSpec(
                    estimate=NumericFormatSpec(precision=2),
                    bounds=NumericFormatSpec(precision=2),
                ),
            ),
            PublicationColumnSpec(
                output_name="SD",
                source="sd",
                column_type="numeric",
                numeric_format=NumericFormatSpec(precision=2),
            ),
            PublicationColumnSpec(
                output_name="p_signflip",
                source="p_value",
                column_type="p_value",
                value_filter={"p_method": "sign_flip"},
                p_value_format=PValueFormatSpec(precision=3, threshold=0.001),
            ),
            PublicationColumnSpec(
                output_name="q_FDR",
                source="q_value",
                column_type="q_value",
                value_filter={"q_method": "benjamini_hochberg"},
                q_value_format=PValueFormatSpec(precision=3, threshold=0.001),
            ),
            PublicationColumnSpec(
                output_name="dz",
                source="effect_size",
                column_type="numeric",
                value_filter={"effect_size_type": "dz"},
                numeric_format=NumericFormatSpec(precision=2),
            ),
        ),
        format=PublicationFormatSpec(missing_value="NA"),
        status_values=("ok",),
        sort_by=("group_label", "effect_label"),
        metadata_columns=("measure", "percent_positive"),
    )


def _read_delimited(path: Path, *, delimiter: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_in_memory_rows_build_display_and_machine_tables() -> None:
    result = build_publication_table_rows(_rows(), table_spec=_mvpa_style_spec())

    assert result["status"] == "ok"
    assert result["source_row_count"] == 3
    assert result["display_rows"] == [
        {
            "ROI": "roi-alpha",
            "Contrast": "contrast-alpha",
            "N": "4",
            "Crossnobis M [95% CI]": "1.23 [0.90, 1.50]",
            "SD": "0.20",
            "p_signflip": "0.045",
            "q_FDR": "0.060",
            "dz": "0.50",
            "measure": "measure-alpha",
            "percent_positive": "75.0",
        },
        {
            "ROI": "roi-beta",
            "Contrast": "contrast-beta",
            "N": "8",
            "Crossnobis M [95% CI]": "0.12 [0.01, 0.23]",
            "SD": "0.50",
            "p_signflip": "<0.001",
            "q_FDR": "0.046",
            "dz": "0.79",
            "measure": "measure-alpha",
            "percent_positive": "87.5",
        },
    ]
    assert result["machine_rows"][0]["N"] == 4
    assert result["machine_rows"][0]["Crossnobis M [95% CI]"] == {
        "mean": 1.234,
        "ci_low": 0.9,
        "ci_high": 1.5,
    }
    assert result["machine_rows"][0]["p_signflip"] == 0.045
    assert result["machine_rows"][0]["q_FDR"] == 0.06
    assert result["column_mappings"][0]["source"] == "group_label"
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_source_tsv_csv_json_and_json_row_key_inputs(tmp_path: Path) -> None:
    rows = _rows()
    tsv_path = tmp_path / "source.tsv"
    csv_path = tmp_path / "source.csv"
    json_list_path = tmp_path / "source-list.json"
    json_key_path = tmp_path / "source-key.json"
    _write_tsv(tsv_path, rows)
    _write_csv(csv_path, rows)
    json_list_path.write_text(json.dumps(rows), encoding="utf-8")
    json_key_path.write_text(json.dumps({"summary_rows": rows}), encoding="utf-8")

    for source in (
        PublicationSourceSpec(source_id="tsv-source", path=tsv_path),
        PublicationSourceSpec(source_id="csv-source", path=csv_path),
        PublicationSourceSpec(source_id="json-list-source", path=json_list_path),
        PublicationSourceSpec(source_id="json-key-source", path=json_key_path, json_rows_key="summary_rows"),
    ):
        result = build_publication_table_rows(source_spec=source, table_spec=_mvpa_style_spec())

        assert [row["ROI"] for row in result["display_rows"]] == ["roi-alpha", "roi-beta"]
        assert result["display_rows"][0]["Crossnobis M [95% CI]"] == "1.23 [0.90, 1.50]"
        assert result["machine_rows"][1]["q_FDR"] == pytest.approx(0.0456)


def test_plan_mode_previews_manifest_and_writes_nothing(tmp_path: Path) -> None:
    source_path = tmp_path / "source.tsv"
    output_root = tmp_path / "published"
    _write_tsv(source_path, _rows())

    plan = plan_publication_table_outputs(
        source_spec=PublicationSourceSpec(source_id="summary", path=source_path),
        table_spec=_mvpa_style_spec(),
        output_spec=PublicationOutputSpec(
            output_root=output_root,
            display_tsv_path="tables/display.tsv",
            display_csv_path="tables/display.csv",
            display_markdown_path="tables/display.md",
            machine_tsv_path="tables/machine.tsv",
            machine_csv_path="tables/machine.csv",
            machine_json_path="tables/machine.json",
            manifest_path="tables/manifest.json",
        ),
        provenance={"runtime_stage": "step-9"},
    )
    payload = plan.to_dict()

    assert payload["status"] == "ok"
    assert payload["will_write"] is False
    assert payload["output_written"] is False
    assert not output_root.exists()
    assert payload["output_paths"]["display_tsv"].endswith("tables/display.tsv")
    assert payload["manifest"]["source_paths"] == [str(source_path.resolve(strict=False))]
    assert payload["manifest"]["source_hashes"]["summary"]
    assert payload["manifest"]["source_row_counts"] == {"summary": 3}
    assert payload["manifest"]["output_row_counts"]["display_tsv"] == 2
    assert payload["manifest"]["column_mappings"][0]["output_name"] == "ROI"
    assert payload["manifest"]["filters"] == {"filters": {}, "status_values": ["ok"]}
    assert payload["manifest"]["sort_settings"] == ["group_label", "effect_label"]
    assert payload["manifest"]["format_settings"]["missing_value"] == "NA"
    assert payload["manifest"]["provenance"]["runtime_stage"] == "step-9"
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_write_mode_writes_configured_files_and_machine_json_preserves_numbers(tmp_path: Path) -> None:
    output_root = tmp_path / "published"
    unplanned_dir = output_root / "unplanned"
    unplanned_dir.mkdir(parents=True)
    keep_path = unplanned_dir / "keep.txt"
    keep_path.write_text("keep\n", encoding="utf-8")

    result = write_publication_table_outputs(
        _rows(),
        table_spec=_mvpa_style_spec(),
        output_spec=PublicationOutputSpec(
            output_root=output_root,
            display_tsv_path="tables/display.tsv",
            display_csv_path="tables/display.csv",
            display_markdown_path="tables/display.md",
            machine_tsv_path="tables/machine.tsv",
            machine_csv_path="tables/machine.csv",
            machine_json_path="tables/machine.json",
            manifest_path="tables/manifest.json",
        ),
        provenance={"runtime_stage": "step-9"},
    )

    assert result.output_written is True
    assert keep_path.read_text(encoding="utf-8") == "keep\n"
    assert sorted(path.name for path in (output_root / "tables").iterdir()) == [
        "display.csv",
        "display.md",
        "display.tsv",
        "machine.csv",
        "machine.json",
        "machine.tsv",
        "manifest.json",
    ]
    assert _read_delimited(output_root / "tables/display.tsv", delimiter="\t")[0]["N"] == "4"
    assert _read_delimited(output_root / "tables/display.csv", delimiter=",")[1]["p_signflip"] == "<0.001"
    machine_json = json.loads((output_root / "tables/machine.json").read_text(encoding="utf-8"))
    assert machine_json["rows"][0]["N"] == 4
    assert machine_json["rows"][0]["Crossnobis M [95% CI]"]["mean"] == 1.234
    manifest = json.loads((output_root / "tables/manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_hashes"]["display_tsv"]
    assert manifest["output_hashes"]["machine_json"]
    assert manifest["output_row_counts"]["machine_json"] == 2
    assert result.output_hashes["manifest"]


def test_overwrite_refusal_and_overwrite_true_replace_only_planned_files(tmp_path: Path) -> None:
    output_root = tmp_path / "published"
    output_spec = PublicationOutputSpec(
        output_root=output_root,
        display_tsv_path="tables/display.tsv",
        machine_json_path="tables/machine.json",
        manifest_path="tables/manifest.json",
    )
    write_publication_table_outputs(_rows(), table_spec=_mvpa_style_spec(), output_spec=output_spec)
    unplanned_path = output_root / "tables/unplanned.txt"
    unplanned_path.write_text("do not touch\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_publication_table_outputs(_rows(), table_spec=_mvpa_style_spec(), output_spec=output_spec)

    replacement = _rows()
    replacement[1]["mean"] = 2.0
    write_publication_table_outputs(replacement, table_spec=_mvpa_style_spec(), output_spec=output_spec, overwrite=True)

    assert unplanned_path.read_text(encoding="utf-8") == "do not touch\n"
    display_rows = _read_delimited(output_root / "tables/display.tsv", delimiter="\t")
    assert display_rows[0]["Crossnobis M [95% CI]"] == "2.00 [0.90, 1.50]"


def test_missing_source_path_returns_plan_error(tmp_path: Path) -> None:
    plan = plan_publication_table_outputs(
        source_spec=PublicationSourceSpec(source_id="missing", path=tmp_path / "missing.tsv"),
        table_spec=_mvpa_style_spec(),
        output_spec=PublicationOutputSpec(output_root=tmp_path / "published", display_tsv_path="display.tsv"),
    )

    assert plan.status == "error"
    assert "does not exist" in plan.errors[0]
    assert plan.display_rows == ()
    assert not (tmp_path / "published").exists()


def test_tsv_csv_and_markdown_outputs_are_safe_for_table_control_characters(tmp_path: Path) -> None:
    rows = [
        {
            **_rows()[0],
            "group_label": "roi|alpha\nline",
            "effect_label": "contrast\talpha",
        }
    ]
    output_root = tmp_path / "published"

    write_publication_table_outputs(
        rows,
        table_spec=_mvpa_style_spec(),
        output_spec=PublicationOutputSpec(
            output_root=output_root,
            display_tsv_path="display.tsv",
            display_csv_path="display.csv",
            display_markdown_path="display.md",
        ),
    )

    assert _read_delimited(output_root / "display.tsv", delimiter="\t")[0]["ROI"] == "roi|alpha line"
    assert _read_delimited(output_root / "display.csv", delimiter=",")[0]["Contrast"] == "contrast alpha"
    markdown_text = (output_root / "display.md").read_text(encoding="utf-8")
    assert "roi\\|alpha line" in markdown_text
    assert "\t" not in markdown_text


def test_generic_mvpa_style_table_uses_only_generic_column_mapping() -> None:
    spec = _mvpa_style_spec()
    result = build_publication_table_rows(_rows(), table_spec=spec)

    assert [column.output_name for column in spec.columns] == [
        "ROI",
        "Contrast",
        "N",
        "Crossnobis M [95% CI]",
        "SD",
        "p_signflip",
        "q_FDR",
        "dz",
    ]
    assert result["display_rows"][0]["ROI"] == "roi-alpha"
    assert result["column_mappings"][3]["source"] == "mean"
    assert result["column_mappings"][3]["ci_low_source"] == "ci_low"
    assert result["column_mappings"][5]["value_filter"] == {"p_method": "sign_flip"}
    assert result["column_mappings"][6]["value_filter"] == {"q_method": "benjamini_hochberg"}


def test_forbidden_imports_and_study_specific_constants_are_absent() -> None:
    forbidden_modules = (
        "research_platform.neuro",
        "research_platform.bids",
        "research_platform.core",
        "research_platform.viz",
        "research_platform.io",
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "sklearn",
        "nilearn",
        "rsatoolbox",
        "pipelines",
        "ops",
    )
    forbidden_production_text = (
        "confidential-study-marker",
        "private-task-marker",
        "private-cohort-marker",
        "vendor-tool-a",
        "vendor-tool-b",
        "participant-alpha",
        "participant-beta",
    )
    imported_modules: list[str] = []
    combined_text = ""
    production_text = Path(publication_tables.__file__).read_text(encoding="utf-8")
    for path in (Path(publication_tables.__file__), Path(__file__)):
        source_text = path.read_text(encoding="utf-8")
        combined_text += source_text
        tree = ast.parse(source_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)

    for imported_module in imported_modules:
        assert not any(
            imported_module == forbidden or imported_module.startswith(f"{forbidden}.")
            for forbidden in forbidden_modules
        )
    for text in forbidden_production_text:
        assert text not in production_text
    assert re.search(r"sub-\d{3}", combined_text) is None
