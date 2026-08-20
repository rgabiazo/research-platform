from __future__ import annotations

import re
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES = REPO_ROOT / "docs" / "capabilities.md"
ARCHITECTURE = REPO_ROOT / "ARCHITECTURE.md"
ROADMAP = REPO_ROOT / "ROADMAP.md"
BLUEPRINT = REPO_ROOT / "BLUEPRINT.md"
ADD_PROJECT_GUIDE = REPO_ROOT / "docs" / "onboarding" / "add-a-project.md"
ONBOARDING = REPO_ROOT / "docs" / "onboarding"
ONBOARDING_INDEX = ONBOARDING / "README.md"
CODING_AGENT_GUIDE = ONBOARDING / "coding-agent-workflow.md"
MVPA_TRANSACTION_ADR = (
    REPO_ROOT / "docs" / "decisions" / "ADR-0018-mvpa-exact-unit-runtime-transactions.md"
)
RESEARCH_ANALYSIS_README = REPO_ROOT / "packages" / "research-analysis" / "README.md"
RESEARCH_NEURO_README = REPO_ROOT / "packages" / "research-neuro" / "README.md"
ALLOWED_STATUSES = {
    "Runnable locally",
    "Plan/validation only",
    "Experimental or external-runtime",
    "Scaffold only",
}
START_MARKER = "<!-- capability-matrix:start -->"
END_MARKER = "<!-- capability-matrix:end -->"
LINK_PATTERN = re.compile(r"\[[^]]+\]\(([^)]+)\)")
EXPECTED_HEADER = (
    "Surface",
    "Status",
    "Public example or configuration",
    "Verification evidence",
    "External prerequisites",
    "Guide",
    "Known alpha limitation",
)
EXPECTED_SURFACES_BY_STATUS = {
    "Runnable locally": frozenset(
        {
            "rp project init and project validation",
            "project/project-example",
            "project/project-pilot-tabular",
            "Tabular preview, inspection, and keyed merge",
            "Tabular split and preprocessing",
            "Tabular classification and evaluation",
            "Continuous-target regression",
            "BIDS traversal, anchor discovery, and deterministic toy event construction",
            "Generic local coordinate-sphere ROI building",
            "Generic local NIfTI ROI extraction",
            "Local materialized-pattern crossnobis",
        }
    ),
    "Plan/validation only": frozenset(
        {
            "project/project-template",
            "project/project-pilot-bids",
            "Analysis-bundle validation and exact-unit planning",
            "datasets/ds-bids-example",
            "Generic local run planning",
            "Generic SLURM planning",
            "Local HPC setup, target inspection, and plan rendering",
            "Remote-operation planning for synchronization, staging, submission, bootstrap, and retrieval",
            "Local recorded HPC status",
            "Local cancellation-request rendering",
            "Publish-back",
        }
    ),
    "Experimental or external-runtime": frozenset(
        {
            "Publication tables and visualization/report rendering",
            "SSH doctor and remote-data verification",
            "Explicit HPC transfer, bootstrap, submission, and retrieval",
            "Live scheduler status",
            "DeepPrep",
            "fMRIPost-AROMA",
            "First-level FSL FEAT",
            "Advanced and externally backed ROI workflows",
            "Advanced and external MVPA inputs and execution",
            "MVPA RDM, figure, table, derivative, and publication exports",
            "Remote notebook helpers",
            "pipelines/preprocess-bids",
            "pipelines/analysis-bids",
        }
    ),
    "Scaffold only": frozenset(
        {
            "Atlas-label and data-driven ROI families",
            "Checked-in local notebook content",
            "pipelines/ingest",
            "pipelines/feature-engineering",
            "pipelines/train-model",
            "pipelines/evaluate-model",
            "pipelines/qc-reporting",
            "pipelines/publish-results",
            "apps/api",
            "apps/dashboard",
            "apps/streamlit-app",
        }
    ),
}
SCAFFOLD_BOUNDARIES = {
    "apps/api": ("future optional API surface", "app"),
    "apps/dashboard": ("future dashboard surface", "app"),
    "apps/streamlit-app": ("future Streamlit-style exploratory application", "app"),
    "pipelines/evaluate-model": ("future workflow that evaluates models", "pipeline"),
    "pipelines/feature-engineering": ("future workflow that produces feature", "pipeline"),
    "pipelines/ingest": ("future workflow that brings external", "pipeline"),
    "pipelines/publish-results": ("future workflow that packages or publishes", "pipeline"),
    "pipelines/qc-reporting": ("future workflow that generates", "pipeline"),
    "pipelines/train-model": ("future workflow that trains models", "pipeline"),
}


def _matrix_rows() -> list[dict[str, str]]:
    text = CAPABILITIES.read_text(encoding="utf-8")
    table = text.split(START_MARKER, 1)[1].split(END_MARKER, 1)[0]
    lines = [line for line in table.splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    if tuple(header) != EXPECTED_HEADER:
        raise AssertionError(f"Unexpected capability matrix columns: {header}")
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            raise AssertionError(f"Capability row has {len(cells)} cells instead of {len(header)}: {line}")
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


def _surface_key(value: str) -> str:
    return value.replace("`", "").strip()


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _indexed_top_level_directories(parent: str) -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "--", f"{parent}/*"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "/".join(path.split("/", 2)[:2])
        for path in completed.stdout.splitlines()
        if path.startswith(f"{parent}/") and len(path.split("/")) >= 3
    }


class CapabilityDocumentationContractTests(unittest.TestCase):
    def test_installation_evidence_is_durable_and_bounded(self) -> None:
        text = _normalize_whitespace(CAPABILITIES.read_text(encoding="utf-8"))

        self.assertNotRegex(text, r"(?i)\b(?:commit|revision)\s+`?[0-9a-f]{7,40}`?")
        for evidence in (
            "Python 3.11 and 3.12",
            "Ubuntu 24.04 x86_64",
            "macOS 15 ARM64",
            "Python 3.12",
            'bash ops/envs/dev/smoke-check.sh --venv "$VIRTUAL_ENV"',
            "python -m pip check",
            "rp --version",
            "ops/tests/test_bootstrap.py",
            "ops/tests/test_install_smoke.py",
            "ops/tests/test_distribution_versions.py",
            "ops/tests/test_ci_contract.py",
            "`project-pilot-tabular`",
            "24-row synthetic `toy_features.tsv`",
        ):
            self.assertIn(evidence, text)

        for unverified_boundary in (
            "Python 3.13 or newer",
            "another operating system or architecture",
            "complete optional `full` profile",
            "later full-profile release validation",
            "offline HPC installation",
            "a live cluster",
            "an external neuroimaging runtime",
        ):
            self.assertIn(unverified_boundary, text)

    def test_scaffold_versions_are_not_coordinated_release_claims(self) -> None:
        text = _normalize_whitespace(CAPABILITIES.read_text(encoding="utf-8"))

        for identity in (
            "coordinated source-package version is `0.1.0a1`",
            "eight `packages/research-*` distributions",
            "`project-example`",
            "`project-template`",
            "independent scaffold metadata",
            "not part of that coordinated source-package version",
            "three app roots remain **Scaffold only** README-level ownership and extension boundaries",
            "private or development checkout may contain placeholder app metadata or files",
            "curated public export may retain only each root README",
            "Neither directory presence nor placeholder content establishes implementation",
            "implemented runtime product",
        ):
            self.assertIn(identity, text)

        self.assertIn("guided command interface", text)
        self.assertNotIn("beginner-facing command surface", text)

        rows = {_surface_key(row["Surface"]): row for row in _matrix_rows()}
        for app in ("apps/api", "apps/dashboard", "apps/streamlit-app"):
            self.assertEqual(rows[app]["Status"], "Scaffold only")

    def test_onboarding_uses_the_tooling_neutral_coding_agent_guide(self) -> None:
        self.assertTrue(CODING_AGENT_GUIDE.is_file())
        index = ONBOARDING_INDEX.read_text(encoding="utf-8")
        self.assertIn("coding-agent-workflow.md", index)
        self.assertFalse(any(path.name.startswith("codex") for path in ONBOARDING.glob("*.md")))

    def test_matrix_rows_use_allowed_statuses_and_valid_local_guides(self) -> None:
        rows = _matrix_rows()

        self.assertEqual(len(rows), 46)
        surfaces = [_surface_key(row["Surface"]) for row in rows]
        self.assertEqual(len(surfaces), len(set(surfaces)), "Capability surfaces must be unique")
        for row in rows:
            self.assertIn(row["Status"], ALLOWED_STATUSES, row["Surface"])

        observed_by_status = {
            status: frozenset(
                _surface_key(row["Surface"])
                for row in rows
                if row["Status"] == status
            )
            for status in ALLOWED_STATUSES
        }
        self.assertEqual(observed_by_status, EXPECTED_SURFACES_BY_STATUS)

        for row in rows:
            self.assertTrue(LINK_PATTERN.findall(row["Guide"]), row["Surface"])

        for target in LINK_PATTERN.findall(CAPABILITIES.read_text(encoding="utf-8")):
            path_text = target.split("#", 1)[0]
            if "://" in path_text:
                continue
            self.assertTrue(path_text, target)
            self.assertTrue((CAPABILITIES.parent / path_text).resolve().is_file(), target)

    def test_scaffold_readmes_and_matrix_rows_define_readme_only_boundaries(self) -> None:
        rows = {_surface_key(row["Surface"]): row for row in _matrix_rows()}
        common_verification = (
            "Static documentation contract verifies the README-only public boundary; "
            "development placeholders are not implementation evidence"
        )
        forbidden_positive_evidence = (
            "pipeline.yaml",
            "snakefile",
            "ruleset",
            "results/.keep",
            "source module",
            "implementation module",
            "app package metadata",
            "pyproject.toml",
            "placeholder function",
            "placeholder package",
            "placeholder source",
            "application runtime",
        )

        for surface, (role_fragment, boundary_kind) in SCAFFOLD_BOUNDARIES.items():
            readme = REPO_ROOT / surface / "README.md"
            text = _normalize_whitespace(readme.read_text(encoding="utf-8"))
            for required in (
                "Alpha status: Scaffold only",
                "retained to document the intended ownership and extension boundary",
                role_fragment,
                "private or development checkout may contain placeholder metadata or files",
                "their presence does not establish implementation or support",
                "curated public export may retain only this README",
                "docs/capabilities.md",
                "owning implementation",
                "tests",
                "documentation",
                "capability review",
                "release review",
            ):
                self.assertIn(required, text, surface)

            if boundary_kind == "pipeline":
                for absent_claim in (
                    "does not establish a runnable workflow",
                    "Snakefile",
                    "ruleset",
                    "implementation module",
                    "deployment artifact",
                    "supported execution path",
                ):
                    self.assertIn(absent_claim, text, surface)
            elif surface == "apps/api":
                for absent_claim in (
                    "does not establish a runnable application runtime",
                    "API service",
                    "endpoint",
                    "public API",
                    "implementation module",
                    "deployment artifact",
                    "supported execution path",
                ):
                    self.assertIn(absent_claim, text, surface)
            elif surface == "apps/dashboard":
                self.assertIn(
                    "does not establish a runnable application runtime, dashboard",
                    text,
                    surface,
                )
            else:
                self.assertIn(
                    "does not establish a runnable application runtime, Streamlit application",
                    text,
                    surface,
                )

            row = rows[surface]
            self.assertEqual(row["Status"], "Scaffold only", surface)
            self.assertIn(
                "README-level ownership and extension boundary",
                row["Public example or configuration"],
                surface,
            )
            self.assertEqual(row["Verification evidence"], common_verification, surface)
            for requirement in (
                "owning",
                "implementation",
                "tests",
                "documentation",
                "capability review",
                "release review",
            ):
                self.assertIn(requirement, row["External prerequisites"], surface)
            self.assertIn(
                "curated public export may contain only the README",
                row["Known alpha limitation"],
                surface,
            )

            positive_evidence = " ".join(
                (
                    row["Public example or configuration"],
                    row["Verification evidence"],
                )
            ).casefold()
            for forbidden in forbidden_positive_evidence:
                self.assertNotIn(forbidden.casefold(), positive_evidence, surface)

        capabilities_text = CAPABILITIES.read_text(encoding="utf-8")
        for stale_claim in (
            "Placeholder Snakefile and empty declared runtime contract",
            "Repository inventory shows only a `results/.keep` placeholder rule",
            "Placeholder function returning a placeholder string",
            "Repository inventory and placeholder source",
            "Placeholder package only",
        ):
            self.assertNotIn(stale_claim, capabilities_text)

    def test_every_top_level_app_and_pipeline_has_exactly_one_row(self) -> None:
        surfaces = [_surface_key(row["Surface"]) for row in _matrix_rows()]

        expected_statuses = {
            "apps/api": "Scaffold only",
            "apps/dashboard": "Scaffold only",
            "apps/streamlit-app": "Scaffold only",
            "pipelines/analysis-bids": "Experimental or external-runtime",
            "pipelines/evaluate-model": "Scaffold only",
            "pipelines/feature-engineering": "Scaffold only",
            "pipelines/ingest": "Scaffold only",
            "pipelines/preprocess-bids": "Experimental or external-runtime",
            "pipelines/publish-results": "Scaffold only",
            "pipelines/qc-reporting": "Scaffold only",
            "pipelines/train-model": "Scaffold only",
        }
        indexed = _indexed_top_level_directories("apps") | _indexed_top_level_directories("pipelines")

        self.assertEqual(indexed, set(expected_statuses))
        rows_by_surface = {_surface_key(row["Surface"]): row for row in _matrix_rows()}
        for expected, status in expected_statuses.items():
            self.assertEqual(surfaces.count(expected), 1, expected)
            self.assertEqual(rows_by_surface[expected]["Status"], status, expected)

    def test_public_overlays_have_accurate_roles(self) -> None:
        rows = {_surface_key(row["Surface"]): row for row in _matrix_rows()}
        expected = {
            "project/project-template": ("Plan/validation only", "schema"),
            "project/project-example": ("Runnable locally", "toy ROI"),
            "project/project-pilot-bids": ("Plan/validation only", "planning"),
            "project/project-pilot-tabular": ("Runnable locally", "Primary"),
        }

        self.assertEqual(set(expected), {name for name in rows if name.startswith("project/")})
        for name, (status, role_fragment) in expected.items():
            row = rows[name]
            self.assertEqual(row["Status"], status)
            role_text = " ".join(row.values())
            self.assertIn(role_fragment.lower(), role_text.lower())
        self.assertIn("toy-crossnobis", " ".join(rows["project/project-example"].values()))

    def test_roi_rows_preserve_the_public_example_boundary(self) -> None:
        rows = {_surface_key(row["Surface"]): row for row in _matrix_rows()}
        expected = {
            "Generic local coordinate-sphere ROI building": "Runnable locally",
            "Generic local NIfTI ROI extraction": "Runnable locally",
            "Advanced and externally backed ROI workflows": "Experimental or external-runtime",
            "Atlas-label and data-driven ROI families": "Scaffold only",
        }

        for surface, status in expected.items():
            self.assertIn(surface, rows)
            self.assertEqual(rows[surface]["Status"], status)
        for surface in tuple(expected)[:2]:
            self.assertIn("project-example", rows[surface]["Public example or configuration"])

    def test_analysis_bundle_row_preserves_plan_only_boundary(self) -> None:
        rows = {_surface_key(row["Surface"]): row for row in _matrix_rows()}
        surface = "Analysis-bundle validation and exact-unit planning"

        self.assertIn(surface, rows)
        self.assertEqual(rows[surface]["Status"], "Plan/validation only")
        self.assertIn("toy-roi", rows[surface]["Public example or configuration"])
        self.assertIn("toy-crossnobis", rows[surface]["Public example or configuration"])
        self.assertIn("no bundle `run` command", rows[surface]["Known alpha limitation"])

    def test_hpc_rows_separate_local_planning_from_ssh_active_operations(self) -> None:
        rows = {_surface_key(row["Surface"]): row for row in _matrix_rows()}
        expected = {
            "Local HPC setup, target inspection, and plan rendering": "Plan/validation only",
            "SSH doctor and remote-data verification": "Experimental or external-runtime",
            "Local recorded HPC status": "Plan/validation only",
            "Live scheduler status": "Experimental or external-runtime",
            "Local cancellation-request rendering": "Plan/validation only",
            "Explicit HPC transfer, bootstrap, submission, and retrieval": (
                "Experimental or external-runtime"
            ),
        }

        for surface, status in expected.items():
            self.assertIn(surface, rows)
            self.assertEqual(rows[surface]["Status"], status)

        doctor = " ".join(rows["SSH doctor and remote-data verification"].values())
        self.assertIn("contact the configured host immediately", doctor)
        self.assertIn("neither has live-cluster validation", doctor)

        live = " ".join(rows["Live scheduler status"].values())
        live_lower = live.casefold()
        self.assertIn("squeue", live_lower)
        self.assertIn("no `sacct`", live_lower)
        self.assertIn("not-found-or-completed", live_lower)

        cancellation = " ".join(rows["Local cancellation-request rendering"].values())
        self.assertIn("does not run `scancel` or confirm cancellation", cancellation)

        retrieval = " ".join(
            rows["Explicit HPC transfer, bootstrap, submission, and retrieval"].values()
        )
        self.assertIn("merge-oriented `rsync -az`", retrieval)
        self.assertIn("atomic-publication", retrieval)

    def test_mvpa_rows_preserve_the_public_example_boundary(self) -> None:
        rows = {_surface_key(row["Surface"]): row for row in _matrix_rows()}
        expected = {
            "Local materialized-pattern crossnobis": "Runnable locally",
            "Advanced and external MVPA inputs and execution": "Experimental or external-runtime",
            "MVPA RDM, figure, table, derivative, and publication exports": "Experimental or external-runtime",
        }

        for surface, status in expected.items():
            self.assertIn(surface, rows)
            self.assertEqual(rows[surface]["Status"], status)

        local = rows["Local materialized-pattern crossnobis"]
        local_text = " ".join(local.values())
        self.assertIn("project-example", local_text)
        self.assertIn("toy-crossnobis", local_text)
        self.assertIn("materialized-pattern", local_text)
        self.assertIn("test_toy_mvpa_project_cli.py", local["Verification evidence"])
        self.assertIn("distances.tsv", local["Known alpha limitation"])
        self.assertIn("RDM-ready", local["Known alpha limitation"])
        self.assertIn("not an exported RDM", local["Known alpha limitation"])

        advanced_text = " ".join(rows["Advanced and external MVPA inputs and execution"].values())
        self.assertIn("FSL/image CLI execution", advanced_text)
        self.assertIn("real-data MVPA", advanced_text)
        self.assertIn("HPC", advanced_text)
        self.assertIn("deferred adapters", advanced_text)
        self.assertIn("no SPM adapter", advanced_text)

        exports = rows["MVPA RDM, figure, table, derivative, and publication exports"]
        self.assertIn("does not execute RDM/report export", exports["Known alpha limitation"])
        self.assertIn("publication", " ".join(exports.values()))

    def test_current_mvpa_and_roadmap_docs_preserve_the_public_boundary(self) -> None:
        current_status_docs = {
            "ARCHITECTURE.md": _normalize_whitespace(ARCHITECTURE.read_text(encoding="utf-8")),
            "ROADMAP.md": _normalize_whitespace(ROADMAP.read_text(encoding="utf-8")),
            "add-a-project guide": _normalize_whitespace(
                ADD_PROJECT_GUIDE.read_text(encoding="utf-8")
            ),
            "research-analysis README": _normalize_whitespace(
                RESEARCH_ANALYSIS_README.read_text(encoding="utf-8")
            ),
            "research-neuro README": _normalize_whitespace(
                RESEARCH_NEURO_README.read_text(encoding="utf-8")
            ),
        }
        stale_claims = (
            "MVPA still lacks a checked-in public project-level executable happy path.",
            "a public project-level executable path remains roadmap work",
            "Complete a checked-in public MVPA/crossnobis happy path",
            "no checked-in public project-level executable MVPA happy path",
            "MVPA plan-only config/contract foundations",
            "future MVPA components",
        )

        for label, text in current_status_docs.items():
            for claim in stale_claims:
                self.assertNotIn(claim, text, f"{label}: {claim}")

        combined = "\n".join(current_status_docs.values())
        self.assertIn("local materialized-pattern crossnobis", combined)
        for boundary in (
            "FSL/image execution",
            "real-data MVPA",
            "HPC",
            "RDM/report exports",
            "publication",
            "deferred adapters",
        ):
            self.assertIn(boundary, combined)
        self.assertIn("`distances.tsv` is an RDM-ready", combined)
        self.assertIn("not an exported RDM", combined)

    def test_public_direction_model_is_consolidated(self) -> None:
        self.assertTrue(ROADMAP.is_file())
        roadmap = ROADMAP.read_text(encoding="utf-8")
        headings = [line for line in roadmap.splitlines() if line.startswith("#")]
        self.assertEqual(
            headings,
            [
                "# Roadmap",
                "## Current status",
                "## Completed foundations",
                "## Now",
                "## Next public-alpha milestones",
                "## After the first public alpha",
                "## Longer-term possibilities",
                "## Explicitly deferred or unsupported scope",
                "## How roadmap items become supported capabilities",
            ],
        )
        normalized_roadmap = _normalize_whitespace(roadmap)
        for boundary in (
            "describes project direction rather than current support",
            "makes no calendar commitment",
            "documented local ROI examples",
            "materialized-pattern crossnobis",
        ):
            self.assertIn(boundary, normalized_roadmap)
        normalized_architecture = _normalize_whitespace(
            ARCHITECTURE.read_text(encoding="utf-8")
        )
        self.assertIn("complete transactional remote lifecycle", normalized_architecture)
        self.assertIn("not a supported or released public workflow", normalized_architecture)

        blueprint = _normalize_whitespace(BLUEPRINT.read_text(encoding="utf-8"))
        self.assertIn("remains only to preserve historical references", blueprint)
        self.assertIn(
            "not the authority for current architecture, present support, future commitments, or release requirements",
            blueprint,
        )
        for link in (
            "[ARCHITECTURE.md](ARCHITECTURE.md)",
            "[ROADMAP.md](ROADMAP.md)",
            "[docs/capabilities.md](docs/capabilities.md)",
            "[accepted decision records](docs/decisions/README.md)",
        ):
            self.assertIn(link, BLUEPRINT.read_text(encoding="utf-8"))
        self.assertNotIn("## Phase", BLUEPRINT.read_text(encoding="utf-8"))

        retired = (
            REPO_ROOT / "USES.md",
            REPO_ROOT / "PLANNED_USE_CASES.md",
            REPO_ROOT / "docs" / "architecture" / "overview.md",
            REPO_ROOT / "docs" / "architecture" / "directory-map.md",
            REPO_ROOT / "docs" / "architecture" / "repo-boundaries.md",
            REPO_ROOT / "docs" / "release" / "public-release-checklist.md",
        )
        for path in retired:
            self.assertFalse(path.exists(), path.relative_to(REPO_ROOT))

    def test_mvpa_transaction_adr_preserves_history_and_adds_current_status(self) -> None:
        text = _normalize_whitespace(MVPA_TRANSACTION_ADR.read_text(encoding="utf-8"))
        historical_claims = (
            "There is still no checked-in public project-level MVPA execution example.",
            "no public MVPA project example is checked in",
        )
        for claim in historical_claims:
            self.assertIn(claim, text)

        heading = "## 2026-07-20 Implementation Status"
        self.assertIn(heading, text)
        addendum = text.split(heading, 1)[1]
        self.assertGreater(text.index(heading), max(text.index(claim) for claim in historical_claims))
        for current_fact in (
            "`project-example`",
            "`toy-crossnobis`",
            "local materialized-pattern crossnobis",
            "**Runnable locally**",
            "FSL/image execution",
            "real-data MVPA",
            "HPC",
            "deferred adapters",
            "RDM/report export",
            "publication",
            "`distances.tsv`",
            "RDM-ready",
            "not an exported RDM",
        ):
            self.assertIn(current_fact, addendum)

    def test_readme_links_to_matrix_and_runnable_rows_name_evidence(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertRegex(readme, r"\[[^]]+\]\(docs/capabilities\.md\)")
        for row in _matrix_rows():
            if row["Status"] != "Runnable locally":
                continue
            for field in ("Public example or configuration", "Verification evidence"):
                value = row[field]
                self.assertNotIn(value, {"", "—", "None", "N/A"}, row["Surface"])
                self.assertNotRegex(value, r"(?i)\b(?:TBD|TODO|placeholder|future)\b", row["Surface"])
            self.assertRegex(row["Verification evidence"], r"(?i)\b(?:test|check|execut)", row["Surface"])


if __name__ == "__main__":
    unittest.main()
