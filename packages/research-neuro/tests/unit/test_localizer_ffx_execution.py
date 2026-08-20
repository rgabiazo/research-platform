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
from research_platform.neuro.localizer_ffx_execution import (
    execute_localizer_fixed_effects_plan,
    select_executable_localizer_ffx_jobs,
    validate_localizer_fixed_effects_execution_plan,
)


def _document() -> dict[str, object]:
    return {
        "localizer_fixed_effects": {
            "feat_sources": [
                {
                    "name": "localizer-source",
                    "root_ref": "feat-root",
                    "feat_dir_template": "{participant_id}/{session_id}/{task_id}/{run_id}.feat",
                }
            ],
            "contrast_aliases": [{"id": "localizer-alpha", "aliases": ["contrast-alpha"]}],
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
        }
    }


def _feat_dir(root: Path, run_id: str) -> Path:
    return root / "participant-a" / "session-a" / "task-alpha" / f"{run_id}.feat"


def _write_feat(root: Path, run_id: str) -> Path:
    feat_dir = _feat_dir(root, run_id)
    stats_dir = feat_dir / "stats"
    stats_dir.mkdir(parents=True)
    (feat_dir / "design.fsf").write_text('set fmri(conname_real.1) "contrast-alpha"\n', encoding="utf-8")
    (stats_dir / "cope1.nii.gz").write_text("synthetic cope\n", encoding="utf-8")
    (stats_dir / "varcope1.nii.gz").write_text("synthetic varcope\n", encoding="utf-8")
    (feat_dir / "mask.nii.gz").write_text("synthetic mask\n", encoding="utf-8")
    return feat_dir


def _plan(root: Path, *, spaces: bool = False):
    feat_root = root / ("feat root with spaces" if spaces else "feat")
    output_root = root / ("output root with spaces" if spaces else "output")
    _write_feat(feat_root, "run-a")
    _write_feat(feat_root, "run-b")
    return plan_localizer_fixed_effects(
        _document(),
        roots={"feat-root": feat_root, "output-root": output_root},
    )


def _snapshot(root: Path) -> tuple[tuple[str, str, int | None], ...]:
    rows: list[tuple[str, str, int | None]] = []
    for path in sorted(root.rglob("*")):
        kind = "file" if path.is_file() else "dir"
        size = path.stat().st_size if path.is_file() else None
        rows.append((path.relative_to(root).as_posix(), kind, size))
    return tuple(rows)


def _path_by_kind(plan: object, kind: str) -> Path:
    for row in plan.output_path_rows:
        if row.path_kind == kind:
            return Path(row.path)
    raise AssertionError(f"No planned path kind {kind!r}")


class FakeRunner:
    def __init__(
        self,
        *,
        fail_at: int | None = None,
        fail_returncode: int = 17,
        write_expected_outputs: bool = True,
    ) -> None:
        self.fail_at = fail_at
        self.fail_returncode = fail_returncode
        self.write_expected_outputs = write_expected_outputs
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        self.assert_argv(argv)
        command_index = len(self.calls)
        self.calls.append(list(argv))
        if self.fail_at == command_index:
            return subprocess.CompletedProcess(argv, self.fail_returncode, stdout="failed stdout", stderr="failed stderr")
        if argv[0] == "fslmerge":
            output_path = Path(argv[2])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"merged {command_index}\n", encoding="utf-8")
        if argv[0] == "flameo" and self.write_expected_outputs:
            output_dir = _flag_path(argv, "--ld")
            stats_dir = output_dir / "stats"
            stats_dir.mkdir(parents=True, exist_ok=True)
            (stats_dir / "cope1.nii.gz").write_text("fixed effects cope\n", encoding="utf-8")
            (stats_dir / "varcope1.nii.gz").write_text("fixed effects varcope\n", encoding="utf-8")
            (output_dir / "mask.nii.gz").write_text("fixed effects mask\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="ok stdout", stderr="")

    @staticmethod
    def assert_argv(argv: object) -> None:
        if not isinstance(argv, list):
            raise AssertionError("runner received a non-list argv")
        if any(not isinstance(part, str) for part in argv):
            raise AssertionError("runner received non-string argv element")


def _flag_path(argv: list[str], name: str) -> Path:
    prefix = f"{name}="
    for arg in argv:
        if arg.startswith(prefix):
            return Path(arg.split("=", 1)[1])
    raise AssertionError(f"Missing {name}= flag")


class LocalizerFfxExecutionTests(unittest.TestCase):
    def test_existing_plan_mode_still_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feat_root = root / "feat"
            output_root = root / "output"
            _write_feat(feat_root, "run-a")
            _write_feat(feat_root, "run-b")
            before = _snapshot(root)

            plan = plan_localizer_fixed_effects(
                _document(),
                roots={"feat-root": feat_root, "output-root": output_root},
            )
            after = _snapshot(root)

        self.assertEqual(plan.status, "ok")
        self.assertEqual(before, after)

    def test_successful_mocked_execution_from_plan_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            runner = FakeRunner()

            result = execute_localizer_fixed_effects_plan(plan.to_dict(), runner=runner)

            output_cope = _path_by_kind(plan, "output_cope")
            output_varcope = _path_by_kind(plan, "output_varcope")
            output_mask = _path_by_kind(plan, "output_mask")
            design_mat = _path_by_kind(plan, "design_file")
            design_con = _path_by_kind(plan, "t_contrast_file")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["completed_job_rows"]), 1)
            self.assertEqual([call[0] for call in runner.calls], ["fslmerge", "fslmerge", "flameo"])
            self.assertTrue(output_cope.is_file())
            self.assertTrue(output_varcope.is_file())
            self.assertTrue(output_mask.is_file())
            self.assertIn("/NumPoints 2", design_mat.read_text(encoding="utf-8"))
            self.assertIn("/NumContrasts 1", design_con.read_text(encoding="utf-8"))
            self.assertFalse((design_mat.parent / "design.grp").exists())
            json.dumps(result, allow_nan=False, sort_keys=True)

    def test_command_vectors_are_passed_exactly_as_argv_lists_with_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root, spaces=True)
            runner = FakeRunner()

            execute_localizer_fixed_effects_plan(plan, runner=runner)

        expected = [list(command) for command in plan.ffx_job_rows[0].commands]
        self.assertEqual(runner.calls, expected)
        self.assertTrue(any("output root with spaces" in arg for call in runner.calls for arg in call))
        self.assertTrue(any("feat root with spaces" in arg for call in runner.calls for arg in call))

    def test_design_files_are_written_only_during_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            design_mat = _path_by_kind(plan, "design_file")
            design_con = _path_by_kind(plan, "t_contrast_file")

            self.assertFalse(design_mat.exists())
            self.assertFalse(design_con.exists())

            execute_localizer_fixed_effects_plan(plan, runner=FakeRunner())

            self.assertTrue(design_mat.is_file())
            self.assertTrue(design_con.is_file())

    def test_existing_outputs_refuse_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            output_cope = _path_by_kind(plan, "output_cope")
            output_cope.parent.mkdir(parents=True)
            output_cope.write_text("old cope\n", encoding="utf-8")
            runner = FakeRunner()

            result = execute_localizer_fixed_effects_plan(plan, runner=runner)

            self.assertEqual(result["status"], "error")
            self.assertEqual(runner.calls, [])
            self.assertEqual(result["failed_job_rows"][0]["reason"], "overwrite_refused")
            self.assertEqual(result["overwrite_refusal_rows"][0]["paths"][0]["path_kind"], "output_cope")
            self.assertEqual(output_cope.read_text(encoding="utf-8"), "old cope\n")

    def test_overwrite_replaces_only_planned_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            output_cope = _path_by_kind(plan, "output_cope")
            output_dir = _path_by_kind(plan, "output_dir")
            work_dir = _path_by_kind(plan, "work_dir")
            output_cope.parent.mkdir(parents=True)
            work_dir.mkdir(parents=True)
            output_cope.write_text("old cope\n", encoding="utf-8")
            extra_output = output_dir / "keep-output.txt"
            extra_work = work_dir / "keep-work.txt"
            extra_output.write_text("keep output\n", encoding="utf-8")
            extra_work.write_text("keep work\n", encoding="utf-8")

            result = execute_localizer_fixed_effects_plan(plan, runner=FakeRunner(), overwrite=True)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(output_cope.read_text(encoding="utf-8"), "fixed effects cope\n")
            self.assertEqual(extra_output.read_text(encoding="utf-8"), "keep output\n")
            self.assertEqual(extra_work.read_text(encoding="utf-8"), "keep work\n")

    def test_missing_inputs_fail_or_skip_according_to_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            Path(plan.ffx_job_rows[0].cope_inputs[0]).unlink()
            fail_runner = FakeRunner()
            skip_runner = FakeRunner()

            failed = execute_localizer_fixed_effects_plan(plan, runner=fail_runner, missing_input_policy="fail")
            skipped = execute_localizer_fixed_effects_plan(plan, runner=skip_runner, missing_input_policy="skip")

        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["failed_job_rows"][0]["reason"], "missing_inputs")
        self.assertEqual(fail_runner.calls, [])
        self.assertEqual(skipped["status"], "warning")
        self.assertEqual(skipped["skipped_job_rows"][0]["reason"], "missing_inputs")
        self.assertEqual(skip_runner.calls, [])

    def test_failed_first_fslmerge_records_failure_and_skips_downstream(self) -> None:
        result, runner = self._execute_with_failed_command(fail_at=0)

        self.assertEqual(result["status"], "error")
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(result["failed_job_rows"][0]["failed_command_index"], 0)
        self.assertEqual(result["failed_job_rows"][0]["skipped_command_count"], 2)
        self.assertEqual(result["command_records"][0]["returncode"], 17)

    def test_failed_second_fslmerge_records_failure_and_skips_flameo(self) -> None:
        result, runner = self._execute_with_failed_command(fail_at=1)

        self.assertEqual(result["status"], "error")
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(result["failed_job_rows"][0]["failed_command_index"], 1)
        self.assertEqual(result["failed_job_rows"][0]["skipped_command_count"], 1)

    def test_failed_flameo_records_failure(self) -> None:
        result, runner = self._execute_with_failed_command(fail_at=2)

        self.assertEqual(result["status"], "error")
        self.assertEqual(len(runner.calls), 3)
        self.assertEqual(result["failed_job_rows"][0]["failed_command_index"], 2)
        self.assertEqual(result["failed_job_rows"][0]["failed_command"]["argv"][0], "flameo")

    def test_successful_commands_with_missing_expected_outputs_mark_job_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            runner = FakeRunner(write_expected_outputs=False)

            result = execute_localizer_fixed_effects_plan(plan, runner=runner)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["failed_job_rows"][0]["reason"], "missing_expected_outputs")
        self.assertTrue(all(row["status"] == "missing" for row in result["expected_output_check_rows"]))

    def test_execution_writes_files_only_under_planned_work_and_output_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            before = {path.resolve() for path in root.rglob("*") if path.is_file()}
            work_dir = _path_by_kind(plan, "work_dir").resolve()
            output_dir = _path_by_kind(plan, "output_dir").resolve()

            execute_localizer_fixed_effects_plan(plan, runner=FakeRunner())

            after = {path.resolve() for path in root.rglob("*") if path.is_file()}
            created = after - before

        self.assertTrue(created)
        for path in created:
            self.assertTrue(_is_under(path, work_dir) or _is_under(path, output_dir), str(path))

    def test_validation_and_selection_are_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            before = _snapshot(root)

            validation = validate_localizer_fixed_effects_execution_plan(plan)
            selected = select_executable_localizer_ffx_jobs(plan)
            after = _snapshot(root)

        self.assertTrue(validation["valid"])
        self.assertEqual(len(selected), 1)
        self.assertEqual(before, after)

    def test_plan_with_fatal_errors_refuses_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = plan_localizer_fixed_effects(
                _document(),
                roots={"feat-root": root / "missing-feat", "output-root": root / "output"},
            )
            runner = FakeRunner()

            result = execute_localizer_fixed_effects_plan(plan, runner=runner)

        self.assertEqual(plan.status, "error")
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["executed"])
        self.assertEqual(result["execution_attempt_rows"], [])
        self.assertEqual(runner.calls, [])

    def test_malformed_job_rows_refuse_execution(self) -> None:
        payload = {
            "status": "ok",
            "errors": [],
            "ffx_job_rows": [
                {
                    "job_id": "job-alpha",
                    "cope_inputs": ["cope-alpha.nii.gz"],
                    "varcope_inputs": ["varcope-alpha.nii.gz"],
                    "mask_path": "mask-alpha.nii.gz",
                    "output_dir": "output-alpha",
                    "work_dir": "work-alpha",
                    "commands": [["fslmerge"]],
                }
            ],
            "output_path_rows": [],
            "root_ref_rows": [],
        }
        runner = FakeRunner()

        result = execute_localizer_fixed_effects_plan(payload, runner=runner)

        self.assertEqual(result["status"], "error")
        self.assertFalse(result["executed"])
        self.assertTrue(result["validation"]["malformed_job_rows"])
        self.assertEqual(runner.calls, [])

    def test_execution_module_does_not_import_forbidden_packages(self) -> None:
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
            import research_platform.neuro.localizer_ffx_execution  # noqa: F401
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

    def test_execution_module_does_not_reference_loso_roi_or_mvpa_apis(self) -> None:
        source = (
            PACKAGE_ROOT
            / "src"
            / "research_platform"
            / "neuro"
            / "localizer_ffx_execution.py"
        ).read_text(encoding="utf-8")

        forbidden_fragments = (
            "roi_loso",
            "execute_loso",
            "plan_loso",
            "roi_build",
            "roi_execution",
            "research_platform.neuro.mvpa",
            "extract_patterns",
            "compute_distances",
            "crossnobis",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, source)

    def test_execution_module_has_no_study_specific_constants(self) -> None:
        source = (
            PACKAGE_ROOT
            / "src"
            / "research_platform"
            / "neuro"
            / "localizer_ffx_execution.py"
        ).read_text(encoding="utf-8")

        for fragment in (
            "confidential-study-marker",
            "private-task-marker",
            "private-cohort-marker",
            "participant-alpha",
            "participant-beta",
        ):
            self.assertNotIn(fragment, source)

    def _execute_with_failed_command(self, *, fail_at: int) -> tuple[dict[str, object], FakeRunner]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            plan = _plan(root)
            runner = FakeRunner(fail_at=fail_at)

            result = execute_localizer_fixed_effects_plan(plan, runner=runner)

        return result, runner


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    unittest.main()
