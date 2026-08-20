from __future__ import annotations

import ast
import csv
import io
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BYOD_GUIDE = REPO_ROOT / "docs" / "byod.md"
MATERIALIZED_GUIDE = REPO_ROOT / "docs" / "materialized-pattern-table-v1.md"
ROI_GUIDE = REPO_ROOT / "docs" / "roi-workflows.md"
CAPABILITIES = REPO_ROOT / "docs" / "capabilities.md"
BOOTSTRAP = REPO_ROOT / "ops" / "envs" / "dev" / "bootstrap.sh"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CHANGED_DOCUMENTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "datasets/ds-mvpa-example/README.md",
    REPO_ROOT / "docs/README.md",
    BYOD_GUIDE,
    CAPABILITIES,
    REPO_ROOT / "docs/conventions/configuration.md",
    REPO_ROOT / "docs/decisions/ADR-0017-materialized-mvpa-pattern-tables.md",
    MATERIALIZED_GUIDE,
    REPO_ROOT / "docs/mvpa-crossnobis.md",
    REPO_ROOT / "docs/onboarding/README.md",
    REPO_ROOT / "docs/onboarding/add-a-project.md",
    REPO_ROOT / "docs/onboarding/quickstart.md",
    ROI_GUIDE,
    REPO_ROOT / "docs/tabular-slice.md",
)

# Public-contract tests run against the source checkout as well as editable
# installations. Keep this explicit list aligned with bootstrap's workspace
# package discovery rather than relying on a developer's active environment.
for _source_root in sorted((REPO_ROOT / "packages").glob("research-*/src")):
    sys.path.insert(0, str(_source_root))

from research_platform.core import cli as core_cli
from research_platform.core.config import parse_yaml, validate_tabular_feature_columns
from research_platform.core.run_lifecycle import RunLifecycleError
from research_platform.neuro.mvpa.config import parse_mvpa_set_document
from research_platform.neuro.mvpa.materialized_pattern_table import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    load_materialized_pattern_table,
    plan_materialized_pattern_table,
    validate_materialized_pattern_source_fields,
)
from research_platform.neuro.mvpa.pattern_sources import ResolvedAnalysisUnit
from research_platform.neuro.roi_execution import (
    DEFERRED_BUILD_FAMILIES,
    SUPPORTED_BUILD_FAMILIES,
    RoiExecutionContext,
    plan_roi_build,
    plan_roi_extraction,
)


LINK_PATTERN = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
FENCE_PATTERN = re.compile(r"^```([^\n]*)\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    start = text.index(heading) + len(heading)
    match = re.search(r"(?m)^##?\s+", text[start:])
    return text[start:] if match is None else text[start : start + match.start()]


def _fenced_blocks(text: str, language: str) -> list[str]:
    return [
        body
        for info, body in FENCE_PATTERN.findall(text)
        if info.strip().casefold() == language.casefold()
    ]


def _markdown_column_names(text: str, heading: str) -> tuple[str, ...]:
    rows = []
    for line in _section(text, heading).splitlines():
        match = re.match(r"^\|\s*`([^`]+)`\s*\|", line)
        if match:
            rows.append(match.group(1))
    return tuple(rows)


def _parse_tsv_block(block: str) -> tuple[tuple[str, ...], dict[str, str]]:
    reader = csv.DictReader(io.StringIO(block), delimiter="\t")
    rows = list(reader)
    assert reader.fieldnames is not None
    assert len(rows) == 1, "A complete teaching TSV block must contain exactly one row."
    assert None not in rows[0], "The teaching TSV row has more cells than its header."
    assert all(value is not None for value in rows[0].values())
    return tuple(reader.fieldnames), rows[0]


def _materialized_config(*, cv_unit: str, noise_method: str) -> SimpleNamespace:
    return SimpleNamespace(
        conditions=(SimpleNamespace(id="condition_a"),),
        roi_sources=(
            SimpleNamespace(
                name="prepared_rois",
                fields={"roi_labels": ["SeedA"]},
                masks=(),
            ),
        ),
        distance=SimpleNamespace(
            cv_unit=cv_unit,
            grouping_columns=(),
            noise_normalization=SimpleNamespace(method=noise_method),
        ),
        mean_centering=SimpleNamespace(enabled=False, scope="none"),
    )


def _materialized_source() -> SimpleNamespace:
    fields = {
        "name": "prepared_patterns",
        "backend": "materialized_pattern_table",
        "root_ref": "private_inputs",
        "path": "patterns/patterns.tsv",
        "schema_version": SCHEMA_VERSION,
    }
    return SimpleNamespace(
        name=fields["name"],
        root_ref=fields["root_ref"],
        path=fields["path"],
        fields=fields,
    )


def _materialized_unit(*, with_run: bool) -> ResolvedAnalysisUnit:
    values = {"subject_id": "sub-example01"}
    key_columns = ("subject_id",)
    run_id = None
    if with_run:
        values["run_id"] = "run-01"
        key_columns = ("subject_id", "run_id")
        run_id = "run-01"
    return ResolvedAnalysisUnit(
        unit_id="unit-example01" + ("-run-01" if with_run else ""),
        source_row=1,
        key_columns=key_columns,
        subject_id="sub-example01",
        run_id=run_id,
        values=values,
    )


def _plan_and_load_documented_row(
    tmp_path: Path,
    block: str,
    *,
    cv_unit: str,
    noise_method: str,
    with_run: bool,
) -> tuple[object, object]:
    table = tmp_path / "patterns" / "patterns.tsv"
    table.parent.mkdir(parents=True)
    table.write_text(block + ("" if block.endswith("\n") else "\n"), encoding="utf-8", newline="")
    return _plan_and_load_table(
        tmp_path,
        cv_unit=cv_unit,
        noise_method=noise_method,
        with_run=with_run,
    )


def _plan_and_load_table(
    root: Path,
    *,
    cv_unit: str,
    noise_method: str,
    with_run: bool,
) -> tuple[object, object]:
    plan = plan_materialized_pattern_table(
        _materialized_config(cv_unit=cv_unit, noise_method=noise_method),
        _materialized_source(),
        (_materialized_unit(with_run=with_run),),
        roots={"private_inputs": root},
    )
    assert plan.valid, plan.errors
    assert plan.ready_for_materialization
    assert plan.source_sha256 is not None
    loaded = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256)
    assert loaded.valid, loaded.errors
    assert loaded.materialized
    assert len(loaded.rows) == 1
    return plan, loaded


def _assert_local_links_resolve(document: Path) -> None:
    for raw_target in LINK_PATTERN.findall(_text(document)):
        target = raw_target.strip().strip("<>")
        path_text = target.split("#", 1)[0]
        if not path_text or "://" in path_text or path_text.startswith("mailto:"):
            continue
        resolved = (document.parent / path_text).resolve()
        resolved.relative_to(REPO_ROOT)
        assert resolved.is_file(), f"Broken link in {document.relative_to(REPO_ROOT)}: {raw_target}"


def _shell_rp_commands(block: str) -> list[list[str]]:
    logical = block.replace("\\\n", " ")
    commands: list[list[str]] = []
    for line in logical.splitlines():
        stripped = line.strip()
        if not stripped.startswith("rp "):
            continue
        commands.append(shlex.split(stripped))
    return commands


def _parse_documented_rp_command(command: list[str]) -> None:
    argv = ["example" if "$" in token else token for token in command[1:]]
    if argv == ["--version"] or core_cli._literal_project_init_name(argv) is not None:
        return
    core_cli._build_parser().parse_args(argv)


def _capability_statuses() -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in _text(CAPABILITIES).splitlines():
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        cells = [cell.strip().replace("`", "") for cell in line.strip("|").split("|")]
        if len(cells) >= 2:
            statuses[cells[0]] = cells[1]
    return statuses


def test_materialized_column_tables_and_producer_constants_match_runtime() -> None:
    text = _text(MATERIALIZED_GUIDE)

    assert _markdown_column_names(text, "## Required columns (19)") == REQUIRED_COLUMNS
    assert _markdown_column_names(text, "## Optional columns (26)") == OPTIONAL_COLUMNS
    assert len(REQUIRED_COLUMNS) == 19
    assert len(OPTIONAL_COLUMNS) == 26

    producer_blocks = _fenced_blocks(text, "python")
    producer = next(block for block in producer_blocks if "writer.writeheader()" in block)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(producer).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"SCHEMA_VERSION", "REQUIRED_COLUMNS", "OPTIONAL_COLUMNS"}
    }
    assert assignments == {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "REQUIRED_COLUMNS": REQUIRED_COLUMNS,
        "OPTIONAL_COLUMNS": OPTIONAL_COLUMNS,
    }


def test_standard_library_producer_executes_and_its_output_passes_plan_and_load(
    tmp_path: Path,
) -> None:
    producer = next(
        block
        for block in _fenced_blocks(_text(MATERIALIZED_GUIDE), "python")
        if "writer.writeheader()" in block
    )
    root = tmp_path / "producer"
    table = root / "patterns" / "patterns.tsv"
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", producer, str(table)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert table.is_file()
    assert b"\r" not in table.read_bytes()

    plan, loaded = _plan_and_load_table(
        root,
        cv_unit="run",
        noise_method="diagonal",
        with_run=True,
    )
    assert plan.columns == (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)
    assert loaded.rows[0]["pattern_id"] == "pattern-sub-example01-run-01-condition_a-SeedA"
    assert list(loaded.rows[0]["feature_values"]) == [1.0, 2.5, -0.5]
    assert list(loaded.rows[0]["noise_values"]) == [1.0, 1.5, 2.0]


def test_both_complete_materialized_rows_pass_the_real_planner_and_loader(tmp_path: Path) -> None:
    blocks = _fenced_blocks(_section(_text(MATERIALIZED_GUIDE), "## Complete example rows"), "tsv")
    assert len(blocks) == 2

    identity_columns, identity_row = _parse_tsv_block(blocks[0])
    assert identity_columns == REQUIRED_COLUMNS
    identity_plan, identity_load = _plan_and_load_documented_row(
        tmp_path / "identity",
        blocks[0],
        cv_unit="subject",
        noise_method="identity",
        with_run=False,
    )
    assert identity_plan.key_columns == ("subject_id",)
    assert identity_row["noise_status"] == "unused"
    assert identity_row["noise_usable"] == "false"
    assert list(identity_load.rows[0]["feature_values"]) == [1.0, 2.5, -0.5]

    diagonal_columns, diagonal_row = _parse_tsv_block(blocks[1])
    assert diagonal_columns[: len(REQUIRED_COLUMNS)] == REQUIRED_COLUMNS
    assert set(diagonal_columns[len(REQUIRED_COLUMNS) :]).issubset(OPTIONAL_COLUMNS)
    assert len(diagonal_columns) == len(set(diagonal_columns))
    diagonal_plan, diagonal_load = _plan_and_load_documented_row(
        tmp_path / "diagonal",
        blocks[1],
        cv_unit="run",
        noise_method="diagonal",
        with_run=True,
    )
    assert diagonal_plan.key_columns == ("subject_id", "run_id")
    assert diagonal_row["noise_value_kind"] == "variance"
    assert list(diagonal_load.rows[0]["noise_values"]) == [1.0, 1.5, 2.0]


def test_checked_in_toy_materialized_table_passes_real_config_plan_and_load() -> None:
    mvpa_document = parse_yaml(
        _text(REPO_ROOT / "project/project-example/config/analysis/mvpa/toy-crossnobis.yaml"),
        resolve_env=False,
    )
    config = parse_mvpa_set_document(mvpa_document)
    assert config.unit_selection.key_columns == ("subject_id", "task_id", "run_id")
    assert len(config.pattern_sources) == 1

    units_path = REPO_ROOT / "project/project-example/manifests/batches/toy_mvpa_units.tsv"
    with units_path.open("r", encoding="utf-8", newline="") as handle:
        unit_rows = list(csv.DictReader(handle, delimiter="\t"))
    units = tuple(
        ResolvedAnalysisUnit(
            unit_id=f"unit-{source_row}",
            source_row=source_row,
            key_columns=config.unit_selection.key_columns,
            subject_id=row["subject_id"],
            task_id=row["task_id"],
            run_id=row["run_id"],
            values=row,
        )
        for source_row, row in enumerate(unit_rows, start=1)
    )
    assert len(units) == 4

    plan = plan_materialized_pattern_table(
        config,
        config.pattern_sources[0],
        units,
        roots={"mvpa_example": REPO_ROOT / "datasets/ds-mvpa-example"},
    )
    assert plan.valid, plan.errors
    assert plan.ready_for_execution
    assert plan.total_row_count == plan.selected_row_count == 16
    assert plan.source_sha256 is not None
    loaded = load_materialized_pattern_table(plan, expected_sha256=plan.source_sha256)
    assert loaded.valid, loaded.errors
    assert loaded.materialized
    assert len(loaded.rows) == 16
    assert {row["subject_id"] for row in loaded.rows} == {"sub-toy01", "sub-toy02"}
    assert {row["roi_label"] for row in loaded.rows} == {"SeedA", "SeedB"}


def test_source_yaml_and_all_byod_yaml_examples_parse_and_validate() -> None:
    byod_blocks = _fenced_blocks(_text(BYOD_GUIDE), "yaml")
    assert len(byod_blocks) >= 2
    documents = [parse_yaml(block, resolve_env=False) for block in byod_blocks]
    assert all(isinstance(document, dict) for document in documents)

    models = next(document for document in documents if "models" in document)
    predictors = models["models"]["default"]["feature_columns"]
    assert validate_tabular_feature_columns(predictors) == ["predictor_1", "predictor_2"]

    source_blocks = _fenced_blocks(_section(_text(MATERIALIZED_GUIDE), "## Source declaration"), "yaml")
    assert len(source_blocks) == 1
    source = parse_yaml(source_blocks[0], resolve_env=False)
    assert validate_materialized_pattern_source_fields(source, "pattern_source") == ()
    assert source["schema_version"] == SCHEMA_VERSION


def test_documentation_links_shell_syntax_and_rp_commands_are_valid() -> None:
    for document in CHANGED_DOCUMENTS:
        _assert_local_links_resolve(document)

    shell_blocks = _fenced_blocks(_text(BYOD_GUIDE), "bash")
    roi_recipe = _section(_text(ROI_GUIDE), "## Bring your own 3D NIfTI pair")
    shell_blocks.extend(_fenced_blocks(roi_recipe, "bash"))
    tabular_recipe = _section(
        _text(REPO_ROOT / "docs/tabular-slice.md"),
        "## Bring your own table",
    )
    shell_blocks.extend(_fenced_blocks(tabular_recipe, "bash"))
    for document in (
        REPO_ROOT / "docs/onboarding/quickstart.md",
        REPO_ROOT / "docs/onboarding/add-a-project.md",
    ):
        shell_blocks.extend(_fenced_blocks(_text(document), "bash"))
    assert shell_blocks
    for block in shell_blocks:
        checked = subprocess.run(
            ["bash", "-n"],
            input=block,
            text=True,
            capture_output=True,
            check=False,
        )
        assert checked.returncode == 0, checked.stderr
        for command in _shell_rp_commands(block):
            _parse_documented_rp_command(command)


def test_installation_claims_match_bootstrap_profiles_and_ci_matrix() -> None:
    guide = " ".join(_text(BYOD_GUIDE).split())
    bootstrap = _text(BOOTSTRAP)
    ci = _text(CI_WORKFLOW)
    package_block = re.search(r"minimal_package_names=\(\n(.*?)\n\)", bootstrap, re.DOTALL)
    assert package_block is not None
    minimal_packages = tuple(re.findall(r"(?m)^\s+(research-[a-z-]+)\s*$", package_block.group(1)))

    assert len(minimal_packages) == 7
    assert "the seven runtime packages" in guide
    assert "minimal|dev|full|hpc" in bootstrap
    assert 'dev_requirements=("pytest>=8")' in bootstrap
    assert 'full_package_names=("${minimal_package_names[@]}" research-viz)' in bootstrap
    assert "requirements-notebook.txt" in bootstrap
    assert "packages are not currently available from PyPI" in guide
    assert "Python 3.11 or 3.12" in guide
    assert "python -m pip check" in guide
    assert "Ubuntu 24.04 x86_64 with Python 3.11 and 3.12" in guide
    assert "macOS 15 ARM64 with Python 3.12" in guide
    assert "Python 3.13 and newer are outside the verified contract" in guide
    assert 'runs-on: ubuntu-24.04' in ci
    assert 'python-version: ["3.11", "3.12"]' in ci
    assert 'runs-on: macos-15' in ci
    assert 'python-version: "3.12"' in ci
    assert 'machine != "arm64"' in ci

    public_install_docs = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs/onboarding/quickstart.md",
        CAPABILITIES,
        BYOD_GUIDE,
    )
    for document in public_install_docs:
        assert "Python 3.11 or newer" not in _text(document), document
    assert (
        "They do not mean that arbitrary files under every `datasets/<name>/` directory are ignored"
        in guide
    )


def test_tabular_examples_preserve_one_row_predictor_order_and_train_attestation() -> None:
    section = _section(_text(BYOD_GUIDE), "## 4. Tabular route")
    batch_blocks = _fenced_blocks(section, "text")
    assert len(batch_blocks) == 1
    batch_text = batch_blocks[0].replace("\\t", "\t")
    batch_reader = csv.DictReader(io.StringIO(batch_text), delimiter="\t")
    batch_rows = list(batch_reader)
    assert batch_reader.fieldnames == ["feature_table", "target_column"]
    assert batch_rows == [{"feature_table": "features.tsv", "target_column": "binary_target"}]

    models = parse_yaml(_fenced_blocks(section, "yaml")[0], resolve_env=False)
    documented_predictors = models["models"]["default"]["feature_columns"]
    assert validate_tabular_feature_columns(documented_predictors) == documented_predictors
    assert documented_predictors == ["predictor_1", "predictor_2"]

    args = SimpleNamespace()
    with pytest.raises(RunLifecycleError, match="exactly one data row"):
        core_cli._plan_tabular_run(
            args,
            context={"batch": {"name": "main", "row_count": 2}},
            mode="local",
            execute=False,
        )

    # The public parser requires an input training run before evaluation can
    # reach any filesystem or execution behavior.
    with pytest.raises(SystemExit):
        core_cli._build_parser().parse_args(
            [
                "run",
                "local",
                "evaluate",
                "model",
                "--project",
                "private-tabular",
                "--batch",
                "main",
                "--run-id",
                "private-tabular-evaluate-001",
                "--dry-run",
            ]
        )

    commands = [command for block in _fenced_blocks(section, "bash") for command in _shell_rp_commands(block)]
    workflow_commands = [command for command in commands if command[1:3] == ["run", "local"]]
    actions = [(command[3], "--dry-run" in command, "--execute" in command) for command in workflow_commands]
    assert actions == [
        ("preprocess", True, False),
        ("preprocess", False, True),
        ("train", True, False),
        ("train", False, True),
        ("evaluate", True, False),
        ("evaluate", False, True),
    ]
    evaluate_commands = [command for command in workflow_commands if command[3] == "evaluate"]
    assert all("--input-run" in command for command in evaluate_commands)
    assert all(
        command[command.index("--input-run") + 1] == "private-tabular-train-001"
        for command in evaluate_commands
    )
    assert "Training must succeed before evaluation can even be planned" in section


def test_linked_roi_recipe_has_one_shared_identity_and_passes_real_planners(tmp_path: Path) -> None:
    recipe = _section(_text(ROI_GUIDE), "## Bring your own 3D NIfTI pair")
    yaml_blocks = _fenced_blocks(recipe, "yaml")
    assert len(yaml_blocks) == 3
    analysis_document, roi_document, extraction_document = (
        parse_yaml(block, resolve_env=False) for block in yaml_blocks
    )
    assert "external_input_roots" in analysis_document["analysis"]

    roi = roi_document["roi_set"]
    extraction = extraction_document["extraction_set"]
    for document in (roi, extraction):
        assert "subjects" not in document
        assert document.get("session") is None
    assert (roi["subject"], roi["task"], roi["space"]) == (
        extraction["subject"],
        extraction["task"],
        extraction["space"],
    )

    private_inputs = tmp_path / "private-inputs"
    project_root = tmp_path / "project"
    artifacts_root = tmp_path / "artifacts"
    for path in (private_inputs, project_root, artifacts_root):
        path.mkdir()
    context = RoiExecutionContext(
        workspace_root=tmp_path,
        project_root=project_root,
        artifacts_root=artifacts_root,
        project_name="private-analysis",
        root_refs={"private_inputs": private_inputs},
    )
    build = plan_roi_build(roi_document, context=context)
    extraction_plan = plan_roi_extraction(
        extraction_document,
        roi_set_document=roi_document,
        context=context,
    )
    assert len(build.actions) == len(extraction_plan.actions) == 1
    build_entities = build.actions[0].metadata["entities"]
    extraction_entities = extraction_plan.actions[0].metadata
    assert build_entities["subject_id"] == extraction_entities["subject_id"] == "example01"
    assert build_entities["task_id"] == extraction_entities["task_id"] == "exampletask"
    assert build_entities["space"] == extraction_entities["space"] == "ExampleNative"
    assert build_entities.get("session_id") is None
    assert extraction_entities.get("session_id") is None

    byod_roi_section = _section(_text(BYOD_GUIDE), "## 5. One-entity 3D-NIfTI ROI route")
    assert f"roi validate {roi['name']} " in byod_roi_section
    assert f"roi extraction validate {extraction['name']} " in byod_roi_section


def test_guides_keep_unsupported_claims_inside_runtime_and_capability_boundaries() -> None:
    byod = " ".join(_text(BYOD_GUIDE).split())
    materialized = " ".join(_text(MATERIALIZED_GUIDE).split())
    statuses = _capability_statuses()

    assert statuses["Local materialized-pattern crossnobis"] == "Runnable locally"
    assert statuses["Advanced and external MVPA inputs and execution"] == "Experimental or external-runtime"
    assert statuses["MVPA RDM, figure, table, derivative, and publication exports"] == (
        "Experimental or external-runtime"
    )
    assert statuses["Advanced and externally backed ROI workflows"] == "Experimental or external-runtime"
    assert statuses["Atlas-label and data-driven ROI families"] == "Scaffold only"

    assert {"atlas_label", "data_driven_hook"}.issubset(DEFERRED_BUILD_FAMILIES)
    assert "coordinate_sphere" in SUPPORTED_BUILD_FAMILIES
    for boundary in (
        "FSL/image MVPA and real-data neuroimaging pipelines remain experimental or external-runtime",
        "SPM is unsupported",
        "RDM/report export is deferred",
        "have not been live-cluster validated",
        "There is no automatic resampling, transform, or 4D time-series support",
        "does not establish public real-data validation",
    ):
        assert boundary in byod
    for hpc_boundary in (
        "local ROI, bundle, and MVPA doctor commands do not contact remote systems",
        "`rp hpc doctor` immediately checks SSH connectivity",
        "`rp hpc verify data` immediately contacts the configured host",
        "check remote paths over SSH",
        "`rp hpc status --live` immediately uses SSH",
        "repository coverage mocks their remote boundaries",
    ):
        assert hpc_boundary in byod
    for boundary in (
        "support only ROI-final `prepared_features`",
        "do not provide an image, FSL, SPM, Nilearn, CIFTI, or mixed-representation producer",
        "not evidence of public real-data validation",
    ):
        assert boundary in materialized
