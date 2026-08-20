from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.localizer_ffx import plan_localizer_fixed_effects


def _document(*, min_complete_runs: int = 1) -> dict[str, object]:
    return {
        "localizer_fixed_effects": {
            "feat_sources": [
                {
                    "name": "localizer-source",
                    "root_ref": "feat-root",
                    "feat_dir_template": "{participant_id}/{session_id}/{task_id}/{run_id}.feat",
                }
            ],
            "contrast_aliases": [
                {"id": "localizer-alpha", "aliases": ["contrast-alpha"]},
            ],
            "runs": [
                {
                    "participant_id": "participant-a",
                    "session_id": "session-a",
                    "task_id": "task-alpha",
                    "run_id": "run-a",
                },
                {
                    "participant_id": "participant-a",
                    "session_id": "session-a",
                    "task_id": "task-alpha",
                    "run_id": "run-b",
                },
            ],
            "outputs": {
                "root_ref": "output-root",
                "output_dir_template": "{participant_id}/{session_id}/{task_id}/{contrast_id}/ffx.feat",
                "work_dir_template": "work/{participant_id}/{session_id}/{task_id}/{contrast_id}",
            },
            "min_complete_runs": min_complete_runs,
        }
    }


def _generic_workflow_document() -> dict[str, object]:
    return {
        "analysis_workflow": {
            "name": "workflow-alpha",
            "runtime_tag": "runtime-alpha",
            "cohort": {
                "subjects": {"include": ["participant-a"]},
                "sessions": ["session-a"],
                "tasks": ["task-alpha"],
                "runs": ["run-a"],
            },
            "output_roots": {
                "runtime_root": {
                    "root_ref": "output-root",
                    "path": "{workflow}/{runtime_tag}",
                }
            },
            "extensions": {
                "mvpa": {
                    "localizer_feat_sources": [
                        {
                            "name": "localizer-source",
                            "root_ref": "feat-root",
                            "feat_dir_template": "{participant_id}/{session_id}/{task_id}/{run_id}.feat",
                        }
                    ],
                    "pe_mapping_aliases": {
                        "localizer-alpha": ["contrast-alpha"],
                    },
                    "localizer_fixed_effects": {
                        "outputs": {
                            "output_dir_template": "{participant_id}/{session_id}/{task_id}/{contrast_id}/ffx.feat",
                            "work_dir_template": "work/{participant_id}/{session_id}/{task_id}/{contrast_id}",
                        }
                    },
                }
            },
        }
    }


def _feat_dir(root: Path, run_id: str) -> Path:
    return root / "participant-a" / "session-a" / "task-alpha" / f"{run_id}.feat"


def _write_feat(
    root: Path,
    run_id: str,
    *,
    design: str = 'set fmri(conname_real.1) "contrast-alpha"\n',
    cope: bool = True,
    varcope: bool = True,
    mask: bool = True,
) -> Path:
    feat_dir = _feat_dir(root, run_id)
    stats_dir = feat_dir / "stats"
    stats_dir.mkdir(parents=True)
    (feat_dir / "design.fsf").write_text(design, encoding="utf-8")
    if cope:
        (stats_dir / "cope1.nii.gz").write_text("synthetic cope\n", encoding="utf-8")
    if varcope:
        (stats_dir / "varcope1.nii.gz").write_text("synthetic varcope\n", encoding="utf-8")
    if mask:
        (feat_dir / "mask.nii.gz").write_text("synthetic mask\n", encoding="utf-8")
    return feat_dir


def _snapshot(root: Path) -> tuple[tuple[str, str, int | None], ...]:
    rows: list[tuple[str, str, int | None]] = []
    for path in sorted(root.rglob("*")):
        kind = "file" if path.is_file() else "dir"
        size = path.stat().st_size if path.is_file() else None
        rows.append((path.relative_to(root).as_posix(), kind, size))
    return tuple(rows)


class LocalizerFfxPlanTests(unittest.TestCase):
    def test_complete_run_inventory_plans_subject_level_ffx_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat root with spaces"
            output_root = root / "output root with spaces"
            _write_feat(feat_root, "run-a")
            _write_feat(feat_root, "run-b")
            before = _snapshot(root)

            plan = plan_localizer_fixed_effects(
                _document(),
                roots={"feat-root": feat_root, "output-root": output_root},
            )
            after = _snapshot(root)

        payload = plan.to_dict()
        self.assertEqual(plan.status, "ok")
        self.assertFalse(plan.executed)
        self.assertTrue(payload["plan_only"])
        self.assertEqual(len(plan.inventory_rows), 2)
        self.assertTrue(all(row.complete for row in plan.inventory_rows))
        self.assertEqual(len(plan.ffx_job_rows), 1)
        self.assertEqual(plan.ffx_job_rows[0].complete_run_count, 2)
        self.assertEqual(plan.ffx_job_rows[0].run_ids, ("run-a", "run-b"))
        self.assertEqual(before, after)
        json.dumps(payload, allow_nan=False, sort_keys=True)

    def test_raw_generic_analysis_workflow_document_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            output_root = root / "output"
            _write_feat(feat_root, "run-a")

            plan = plan_localizer_fixed_effects(
                _generic_workflow_document(),
                roots={"feat-root": feat_root, "output-root": output_root},
            )

        self.assertEqual(plan.status, "ok")
        self.assertEqual(len(plan.inventory_rows), 1)
        self.assertIn("workflow-alpha/runtime-alpha", plan.ffx_job_rows[0].output_dir)

    def test_missing_feat_directory_records_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = plan_localizer_fixed_effects(
                _document(),
                roots={"feat-root": root / "feat", "output-root": root / "output"},
            )

        self.assertEqual(plan.status, "error")
        self.assertTrue(any(row.input_kind == "feat_dir" for row in plan.missing_input_rows))
        self.assertFalse(plan.ffx_job_rows)

    def test_missing_design_fsf_records_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            (_feat_dir(feat_root, "run-a") / "stats").mkdir(parents=True)
            _write_feat(feat_root, "run-b")

            plan = plan_localizer_fixed_effects(
                _document(),
                roots={"feat-root": feat_root, "output-root": root / "output"},
            )

        self.assertEqual(plan.status, "error")
        self.assertTrue(any(row.input_kind == "design_fsf" for row in plan.missing_input_rows))

    def test_missing_cope_file_records_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            _write_feat(feat_root, "run-a", cope=False)
            _write_feat(feat_root, "run-b")

            plan = plan_localizer_fixed_effects(
                _document(),
                roots={"feat-root": feat_root, "output-root": root / "output"},
            )

        self.assertEqual(plan.status, "error")
        self.assertTrue(any(row.input_kind == "cope" for row in plan.missing_input_rows))

    def test_missing_varcope_file_records_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            _write_feat(feat_root, "run-a", varcope=False)
            _write_feat(feat_root, "run-b")

            plan = plan_localizer_fixed_effects(
                _document(),
                roots={"feat-root": feat_root, "output-root": root / "output"},
            )

        self.assertEqual(plan.status, "error")
        self.assertTrue(any(row.input_kind == "varcope" for row in plan.missing_input_rows))

    def test_missing_mask_file_records_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            _write_feat(feat_root, "run-a", mask=False)
            _write_feat(feat_root, "run-b")

            plan = plan_localizer_fixed_effects(
                _document(),
                roots={"feat-root": feat_root, "output-root": root / "output"},
            )

        self.assertEqual(plan.status, "error")
        self.assertTrue(any(row.input_kind == "mask" for row in plan.missing_input_rows))

    def test_missing_root_ref_records_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = plan_localizer_fixed_effects(
                _document(),
                roots={"output-root": root / "output"},
            )

        self.assertEqual(plan.status, "error")
        self.assertTrue(any(row.input_kind == "root_ref" and row.root_ref == "feat-root" for row in plan.missing_input_rows))

    def test_missing_contrast_alias_records_missing_input(self) -> None:
        document = _document()
        localizer = document["localizer_fixed_effects"]  # type: ignore[index]
        localizer["contrast_aliases"] = [{"id": "localizer-missing", "aliases": ["contrast-missing"]}]  # type: ignore[index]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            _write_feat(feat_root, "run-a")
            _write_feat(feat_root, "run-b")

            plan = plan_localizer_fixed_effects(
                document,
                roots={"feat-root": feat_root, "output-root": root / "output"},
            )

        self.assertEqual(plan.status, "error")
        self.assertTrue(any(row.input_kind == "contrast_alias" for row in plan.missing_input_rows))
        self.assertTrue(any("No contrast name match" in error for error in plan.errors))

    def test_ambiguous_contrast_alias_records_missing_input(self) -> None:
        design = textwrap.dedent(
            """
            set fmri(conname_real.1) "contrast-alpha"
            set fmri(conname_real.2) "contrast-alpha"
            """
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            _write_feat(feat_root, "run-a", design=design)
            _write_feat(feat_root, "run-b", design=design)

            plan = plan_localizer_fixed_effects(
                _document(),
                roots={"feat-root": feat_root, "output-root": root / "output"},
            )

        self.assertEqual(plan.status, "error")
        self.assertTrue(any("ambiguous" in error for error in plan.errors))
        self.assertTrue(any(row.input_kind == "contrast_alias" for row in plan.missing_input_rows))

    def test_explicit_run_exclusion_is_not_used_as_ffx_input(self) -> None:
        document = _document()
        localizer = document["localizer_fixed_effects"]  # type: ignore[index]
        localizer["run_exclusions"] = [  # type: ignore[index]
            {
                "id": "exclude-run-b",
                "reason": "synthetic exclusion",
                "selectors": {"run_id": "run-b"},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            run_a = _write_feat(feat_root, "run-a")
            run_b = _write_feat(feat_root, "run-b")

            plan = plan_localizer_fixed_effects(
                document,
                roots={"feat-root": feat_root, "output-root": root / "output"},
            )

        self.assertEqual(plan.status, "ok")
        self.assertEqual(len(plan.excluded_run_rows), 1)
        self.assertEqual(plan.excluded_run_rows[0].run_id, "run-b")
        self.assertEqual(plan.ffx_job_rows[0].run_ids, ("run-a",))
        self.assertIn(str((run_a / "stats" / "cope1.nii.gz").resolve()), plan.ffx_job_rows[0].commands[0])
        self.assertNotIn(str((run_b / "stats" / "cope1.nii.gz").resolve()), plan.ffx_job_rows[0].commands[0])

    def test_insufficient_complete_runs_records_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            _write_feat(feat_root, "run-a")

            plan = plan_localizer_fixed_effects(
                _document(min_complete_runs=2),
                roots={"feat-root": feat_root, "output-root": root / "output"},
            )

        self.assertEqual(plan.status, "error")
        self.assertTrue(any(row.input_kind == "insufficient_complete_runs" for row in plan.missing_input_rows))
        self.assertFalse(plan.ffx_job_rows)

    def test_planned_output_paths_include_expected_fixed_effects_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            _write_feat(feat_root, "run-a")
            _write_feat(feat_root, "run-b")

            plan = plan_localizer_fixed_effects(
                _document(),
                roots={"feat-root": feat_root, "output-root": root / "output"},
            )

        kinds = {row.path_kind for row in plan.output_path_rows}
        self.assertEqual(
            kinds,
            {
                "design_file",
                "merged_cope",
                "merged_varcope",
                "output_cope",
                "output_dir",
                "output_mask",
                "output_varcope",
                "t_contrast_file",
                "work_dir",
            },
        )
        output_paths = {row.path_kind: row.path for row in plan.output_path_rows}
        self.assertTrue(output_paths["output_cope"].endswith("stats/cope1.nii.gz"))
        self.assertTrue(output_paths["output_varcope"].endswith("stats/varcope1.nii.gz"))
        self.assertTrue(output_paths["output_mask"].endswith("mask.nii.gz"))

    def test_shell_safe_command_vectors_keep_paths_with_spaces_single_argv_elements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat root with spaces"
            output_root = root / "output root with spaces"
            run_a = _write_feat(feat_root, "run-a")
            _write_feat(feat_root, "run-b")

            plan = plan_localizer_fixed_effects(
                _document(),
                roots={"feat-root": feat_root, "output-root": output_root},
            )

        commands = plan.ffx_job_rows[0].commands
        cope_path = str((run_a / "stats" / "cope1.nii.gz").resolve())
        self.assertIsInstance(commands[0], tuple)
        self.assertEqual(commands[0][0:2], ("fslmerge", "-t"))
        self.assertIn(cope_path, commands[0])
        self.assertTrue(any(arg == "--runmode=fe" for arg in commands[2]))
        self.assertTrue(any(arg.startswith("--cope=") and "output root with spaces" in arg for arg in commands[2]))

    def test_plan_module_imports_no_forbidden_packages(self) -> None:
        script = textwrap.dedent(
            """
            import builtins
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path("packages/research-neuro/src").resolve()))
            forbidden = {
                "fsl",
                "fslpy",
                "nibabel",
                "nilearn",
                "rsatoolbox",
                "numpy",
                "pandas",
                "polars",
                "scipy",
                "mvpa2",
                "sklearn",
                "research_platform.core",
                "research_platform.bids",
                "research_platform.analysis",
                "research_platform.viz",
                "research_platform.ml",
                "pipelines",
                "ops",
            }
            real_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name in forbidden or any(name.startswith(prefix + ".") for prefix in forbidden):
                    raise RuntimeError(f"forbidden import: {name}")
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = guarded_import
            import research_platform.neuro.localizer_ffx  # noqa: F401
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=WORKSPACE_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
