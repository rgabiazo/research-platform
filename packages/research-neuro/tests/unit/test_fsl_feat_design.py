from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from research_platform.neuro.fsl.feat_design import (
    map_contrast_names_to_cope_numbers,
    parse_fsl_design_ev_titles,
    parse_fsl_design_contrast_file,
    parse_fsl_design_contrast_names,
    parse_fsl_design_file,
    resolve_contrast_aliases_to_cope_numbers,
    map_conditions_to_pe_numbers,
    map_ev_titles_to_pe_numbers,
)


@dataclass(frozen=True)
class ConditionLike:
    id: str
    aliases: tuple[str, ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)


class FslFeatDesignTests(unittest.TestCase):
    def test_quoted_evtitle_parsing(self) -> None:
        result = parse_fsl_design_ev_titles('set fmri(evtitle1) "Condition A"\n')

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.ev_titles[0].ev_index, 1)
        self.assertEqual(result.ev_titles[0].ev_title, "Condition A")

    def test_unquoted_evtitle_parsing(self) -> None:
        result = parse_fsl_design_ev_titles("set fmri(evtitle2) ConditionB\n")

        self.assertEqual(result.ev_titles[0].ev_index, 2)
        self.assertEqual(result.ev_titles[0].ev_title, "ConditionB")

    def test_whitespace_and_tabs_are_supported(self) -> None:
        result = parse_fsl_design_ev_titles('\tset\tfmri(evtitle2)\t"Condition B"\t\n')

        self.assertEqual(result.ev_titles[0].ev_index, 2)
        self.assertEqual(result.ev_titles[0].ev_title, "Condition B")

    def test_comments_and_blank_lines_are_ignored(self) -> None:
        result = parse_fsl_design_ev_titles(
            textwrap.dedent(
                """

                # set fmri(evtitle1) "Commented"
                set fmri(evtitle1) "Condition A" # inline comment
                """
            )
        )

        self.assertEqual(len(result.ev_titles), 1)
        self.assertEqual(result.ev_titles[0].ev_title, "Condition A")

    def test_ignored_non_evtitle_fmri_lines(self) -> None:
        result = parse_fsl_design_ev_titles(
            textwrap.dedent(
                """
                set fmri(evs_orig) 2
                set fmri(evtitle1) "Condition A"
                set fmri(shape1) 3
                """
            )
        )

        self.assertEqual(len(result.ev_titles), 1)
        self.assertEqual(result.ev_titles[0].ev_index, 1)

    def test_source_line_and_line_number_are_preserved(self) -> None:
        source_line = '\tset fmri(evtitle7) "Condition A" # keep source'
        result = parse_fsl_design_ev_titles(f"\n# comment\n{source_line}\n")

        row = result.ev_titles[0]
        self.assertEqual(row.line_number, 3)
        self.assertEqual(row.source_line, source_line)

    def test_out_of_order_ev_indexes_are_preserved_in_source_order(self) -> None:
        result = parse_fsl_design_ev_titles(
            textwrap.dedent(
                """
                set fmri(evtitle2) "Condition B"
                set fmri(evtitle1) "Condition A"
                """
            )
        )

        self.assertEqual([row.ev_index for row in result.ev_titles], [2, 1])
        self.assertEqual(result.status, "ok")

    def test_missing_ev_indexes_warn_on_result(self) -> None:
        result = parse_fsl_design_ev_titles(
            textwrap.dedent(
                """
                set fmri(evtitle1) "Condition A"
                set fmri(evtitle3) "Condition C"
                """
            )
        )

        self.assertEqual(result.status, "warning")
        self.assertTrue(any("Missing EV index(es): 2" in warning for warning in result.warnings))

    def test_duplicate_ev_indexes_error_on_rows(self) -> None:
        result = parse_fsl_design_ev_titles(
            textwrap.dedent(
                """
                set fmri(evtitle1) "Condition A"
                set fmri(evtitle1) "Condition B"
                """
            )
        )

        self.assertEqual(result.status, "error")
        self.assertTrue(all(row.status == "error" for row in result.ev_titles))
        self.assertTrue(any("Duplicate EV index: 1" in error for error in result.errors))

    def test_duplicate_ev_titles_warn_and_are_ambiguous_for_conditions(self) -> None:
        result = parse_fsl_design_ev_titles(
            textwrap.dedent(
                """
                set fmri(evtitle1) "Condition A"
                set fmri(evtitle2) "Condition A"
                """
            )
        )

        self.assertEqual(result.status, "warning")
        self.assertTrue(all(row.status == "warning" for row in result.ev_titles))
        mapping = map_conditions_to_pe_numbers([{"id": "condition_a", "ev_title": "Condition A"}], result)
        self.assertEqual(mapping.status, "error")
        self.assertTrue(any("ambiguous" in error for error in mapping.mappings[0].errors))

    def test_empty_title_errors(self) -> None:
        result = parse_fsl_design_ev_titles('set fmri(evtitle1) ""\n')

        self.assertEqual(result.ev_titles[0].status, "error")
        self.assertTrue(any("empty" in error for error in result.ev_titles[0].errors))

    def test_malformed_evtitle_line_errors(self) -> None:
        result = parse_fsl_design_ev_titles("set fmri(evtitle) Broken\n")

        self.assertEqual(result.status, "error")
        self.assertEqual(result.ev_titles[0].ev_index, None)
        self.assertTrue(any("Malformed evtitle line" in error for error in result.ev_titles[0].errors))

    def test_ev_to_pe_mapping_uses_ev_index_as_pe_number(self) -> None:
        result = parse_fsl_design_ev_titles(
            textwrap.dedent(
                """
                set fmri(evtitle3) "Condition C"
                set fmri(evtitle1) "Condition A"
                """
            )
        )
        mapping = map_ev_titles_to_pe_numbers(result)

        self.assertEqual([(row.ev_index, row.pe_number, row.pe_filename) for row in mapping.mappings], [(3, 3, "pe3.nii.gz"), (1, 1, "pe1.nii.gz")])

    def test_ev_to_pe_mapping_accounts_for_temporal_derivatives(self) -> None:
        result = parse_fsl_design_ev_titles(
            textwrap.dedent(
                """
                set fmri(evtitle1) "Condition A"
                set fmri(deriv_yn1) 1
                set fmri(evtitle2) "Condition B"
                set fmri(deriv_yn2) 0
                set fmri(evtitle3) "Condition C"
                """
            )
        )
        mapping = map_conditions_to_pe_numbers(
            [
                {"id": "condition_a", "ev_title": "Condition A"},
                {"id": "condition_b", "ev_title": "Condition B"},
                {"id": "condition_c", "ev_title": "Condition C"},
            ],
            result,
        )

        self.assertEqual(result.ev_column_counts, {1: 2, 2: 1, 3: 1})
        self.assertEqual(result.ev_main_pe_numbers, {1: 1, 2: 3, 3: 4})
        self.assertEqual(
            [(row.ev_index, row.pe_number, row.pe_filename) for row in mapping.mappings],
            [(1, 1, "pe1.nii.gz"), (2, 3, "pe3.nii.gz"), (3, 4, "pe4.nii.gz")],
        )

    def test_condition_mapping_by_exact_ev_title(self) -> None:
        design = parse_fsl_design_ev_titles('set fmri(evtitle1) "Condition A"\n')

        mapping = map_conditions_to_pe_numbers([{"id": "condition_a", "fsl_ev_title": "Condition A"}], design)

        row = mapping.mappings[0]
        self.assertEqual(row.status, "ok")
        self.assertEqual(row.pe_number, 1)
        self.assertEqual(row.pe_filename, "pe1.nii.gz")
        self.assertEqual(row.matched_by, "ev_title")

    def test_condition_mapping_by_alias_with_condition_like_object(self) -> None:
        design = parse_fsl_design_ev_titles('set fmri(evtitle2) "faces"\n')

        mapping = map_conditions_to_pe_numbers([ConditionLike(id="encode_faces", aliases=("face_trials", "faces"))], design)

        row = mapping.mappings[0]
        self.assertEqual(row.status, "ok")
        self.assertEqual(row.pe_number, 2)
        self.assertEqual(row.matched_by, "alias")

    def test_explicit_ev_title_is_matched_before_aliases(self) -> None:
        design = parse_fsl_design_ev_titles(
            textwrap.dedent(
                """
                set fmri(evtitle1) "Condition A"
                set fmri(evtitle2) "Condition B"
                """
            )
        )

        mapping = map_conditions_to_pe_numbers(
            [{"id": "condition_b", "ev_title": "Condition B", "aliases": ["Condition A"]}],
            design,
        )

        self.assertEqual(mapping.mappings[0].pe_number, 2)
        self.assertEqual(mapping.mappings[0].matched_by, "ev_title")

    def test_missing_condition_match_errors_by_default(self) -> None:
        design = parse_fsl_design_ev_titles('set fmri(evtitle1) "Condition A"\n')

        mapping = map_conditions_to_pe_numbers([{"id": "missing", "ev_title": "Missing"}], design)

        self.assertEqual(mapping.status, "error")
        self.assertTrue(any("No EV title match" in error for error in mapping.mappings[0].errors))

    def test_missing_condition_match_can_warn(self) -> None:
        design = parse_fsl_design_ev_titles('set fmri(evtitle1) "Condition A"\n')

        mapping = map_conditions_to_pe_numbers(
            [{"id": "missing", "ev_title": "Missing"}],
            design,
            missing_policy="warn",
        )

        self.assertEqual(mapping.status, "warning")
        self.assertTrue(any("No EV title match" in warning for warning in mapping.mappings[0].warnings))

    def test_case_insensitive_matching_only_when_unique(self) -> None:
        design = parse_fsl_design_ev_titles('set fmri(evtitle1) "Condition A"\n')

        mapping = map_conditions_to_pe_numbers(
            [{"id": "condition_a", "ev_title": "condition a"}],
            design,
            case_sensitive=False,
        )

        self.assertEqual(mapping.mappings[0].status, "ok")
        self.assertEqual(mapping.mappings[0].pe_number, 1)

    def test_case_insensitive_collision_is_ambiguous(self) -> None:
        design = parse_fsl_design_ev_titles(
            textwrap.dedent(
                """
                set fmri(evtitle1) "Condition A"
                set fmri(evtitle2) "condition a"
                """
            )
        )

        mapping = map_conditions_to_pe_numbers(
            [{"id": "condition_a", "ev_title": "CONDITION A"}],
            design,
            case_sensitive=False,
        )

        self.assertEqual(mapping.status, "error")
        self.assertTrue(any("ambiguous" in error for error in mapping.mappings[0].errors))

    def test_hard_coded_pe_cope_fields_are_rejected(self) -> None:
        design = parse_fsl_design_ev_titles('set fmri(evtitle1) "Condition A"\n')

        mapping = map_conditions_to_pe_numbers(
            [{"id": "condition_a", "ev_title": "Condition A", "selector": {"cope": 2}, "pe_number": 1}],
            design,
        )

        row = mapping.mappings[0]
        self.assertEqual(row.status, "error")
        self.assertIsNone(row.pe_number)
        self.assertTrue(any("PE/COPE" in error and "pe_number" in error for error in row.errors))
        self.assertTrue(any("PE/COPE" in error and "selector.cope" in error for error in row.errors))

    def test_unsafe_condition_ids_are_flagged(self) -> None:
        design = parse_fsl_design_ev_titles('set fmri(evtitle1) "Condition A"\n')

        mapping = map_conditions_to_pe_numbers([{"id": "condition a", "ev_title": "Condition A"}], design)

        self.assertEqual(mapping.status, "error")
        self.assertTrue(any("not safe" in error for error in mapping.mappings[0].errors))

    def test_duplicate_pe_assignment_is_flagged(self) -> None:
        design = parse_fsl_design_ev_titles('set fmri(evtitle1) "Condition A"\n')

        mapping = map_conditions_to_pe_numbers(
            [
                {"id": "condition_a", "ev_title": "Condition A"},
                {"id": "condition_a_copy", "ev_title": "Condition A"},
            ],
            design,
        )

        self.assertEqual(mapping.status, "error")
        self.assertTrue(all(any("Duplicate PE assignment" in error for error in row.errors) for row in mapping.mappings))

    def test_to_dict_outputs_json_safe_payloads(self) -> None:
        design = parse_fsl_design_ev_titles('set fmri(evtitle1) "Condition A"\n')
        ev_mapping = map_ev_titles_to_pe_numbers(design)
        condition_mapping = map_conditions_to_pe_numbers([{"id": "condition_a", "ev_title": "Condition A"}], design)

        json.dumps(design.to_dict())
        json.dumps(ev_mapping.to_dict())
        json.dumps(condition_mapping.to_dict())
        self.assertEqual(design.to_dict()["ev_titles"][0]["warnings"], [])
        self.assertTrue(condition_mapping.to_dict()["valid"])

    def test_parse_fsl_design_file_reads_synthetic_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "design.fsf"
            path.write_text('set fmri(evtitle1) "Condition A"\n', encoding="utf-8")

            result = parse_fsl_design_file(path)

        self.assertEqual(result.ev_titles[0].ev_title, "Condition A")

    def test_contrast_name_parsing_from_design_fsf(self) -> None:
        result = parse_fsl_design_contrast_names(
            textwrap.dedent(
                """
                set fmri(conname_real.1) "contrast-alpha"
                set fmri(conname_real.2) contrast-beta
                """
            )
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual([(row.contrast_number, row.contrast_name) for row in result.contrasts], [(1, "contrast-alpha"), (2, "contrast-beta")])

    def test_contrast_name_parse_preserves_source_location(self) -> None:
        source_line = '\tset fmri(conname_real.3) "contrast-gamma" # comment'
        result = parse_fsl_design_contrast_names(f"\n{source_line}\n")

        row = result.contrasts[0]
        self.assertEqual(row.line_number, 2)
        self.assertEqual(row.source_line, source_line)

    def test_contrast_to_cope_mapping_uses_design_numbers(self) -> None:
        design = parse_fsl_design_contrast_names('set fmri(conname_real.4) "contrast-delta"\n')

        mapping = map_contrast_names_to_cope_numbers(design)

        row = mapping.mappings[0]
        self.assertEqual(row.contrast_number, 4)
        self.assertEqual(row.cope_number, 4)
        self.assertEqual(row.cope_filename, "cope4.nii.gz")
        self.assertEqual(row.varcope_number, 4)
        self.assertEqual(row.varcope_filename, "varcope4.nii.gz")

    def test_contrast_alias_to_cope_mapping(self) -> None:
        design = parse_fsl_design_contrast_names('set fmri(conname_real.2) "contrast-alpha"\n')

        mapping = resolve_contrast_aliases_to_cope_numbers(
            [{"id": "localizer-alpha", "aliases": ["contrast-alpha"]}],
            design,
        )

        row = mapping.mappings[0]
        self.assertEqual(row.status, "ok")
        self.assertEqual(row.matched_contrast_name, "contrast-alpha")
        self.assertEqual(row.cope_number, 2)
        self.assertEqual(row.varcope_number, 2)

    def test_contrast_alias_exact_name_precedes_aliases(self) -> None:
        design = parse_fsl_design_contrast_names(
            textwrap.dedent(
                """
                set fmri(conname_real.1) "contrast-alpha"
                set fmri(conname_real.2) "contrast-beta"
                """
            )
        )

        mapping = resolve_contrast_aliases_to_cope_numbers(
            [{"id": "localizer-beta", "contrast_name": "contrast-beta", "aliases": ["contrast-alpha"]}],
            design,
        )

        self.assertEqual(mapping.mappings[0].cope_number, 2)
        self.assertEqual(mapping.mappings[0].matched_by, "contrast_name")

    def test_duplicate_contrast_names_are_ambiguous_for_aliases(self) -> None:
        design = parse_fsl_design_contrast_names(
            textwrap.dedent(
                """
                set fmri(conname_real.1) "contrast-alpha"
                set fmri(conname_real.2) "contrast-alpha"
                """
            )
        )

        mapping = resolve_contrast_aliases_to_cope_numbers(
            [{"id": "localizer-alpha", "aliases": ["contrast-alpha"]}],
            design,
        )

        self.assertEqual(design.status, "warning")
        self.assertEqual(mapping.status, "error")
        self.assertTrue(any("ambiguous" in error for error in mapping.mappings[0].errors))

    def test_missing_contrast_alias_errors(self) -> None:
        design = parse_fsl_design_contrast_names('set fmri(conname_real.1) "contrast-alpha"\n')

        mapping = resolve_contrast_aliases_to_cope_numbers(
            [{"id": "localizer-missing", "aliases": ["contrast-missing"]}],
            design,
        )

        self.assertEqual(mapping.status, "error")
        self.assertTrue(any("No contrast name match" in error for error in mapping.mappings[0].errors))

    def test_contrast_alias_rejects_hard_coded_cope_numbers(self) -> None:
        design = parse_fsl_design_contrast_names('set fmri(conname_real.1) "contrast-alpha"\n')

        mapping = resolve_contrast_aliases_to_cope_numbers(
            [{"id": "localizer-alpha", "aliases": ["contrast-alpha"], "cope_number": 1}],
            design,
        )

        row = mapping.mappings[0]
        self.assertEqual(row.status, "error")
        self.assertIsNone(row.cope_number)
        self.assertTrue(any("hard-code" in error and "cope_number" in error for error in row.errors))

    def test_parse_fsl_design_contrast_file_reads_synthetic_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "design.fsf"
            path.write_text('set fmri(conname_real.1) "contrast-alpha"\n', encoding="utf-8")

            result = parse_fsl_design_contrast_file(path)

        self.assertEqual(result.contrasts[0].contrast_name, "contrast-alpha")

    def test_contrast_to_dict_outputs_json_safe_payloads(self) -> None:
        design = parse_fsl_design_contrast_names('set fmri(conname_real.1) "contrast-alpha"\n')
        cope_mapping = map_contrast_names_to_cope_numbers(design)
        alias_mapping = resolve_contrast_aliases_to_cope_numbers(
            [{"id": "localizer-alpha", "aliases": ["contrast-alpha"]}],
            design,
        )

        json.dumps(design.to_dict(), allow_nan=False)
        json.dumps(cope_mapping.to_dict(), allow_nan=False)
        json.dumps(alias_mapping.to_dict(), allow_nan=False)
        self.assertTrue(alias_mapping.to_dict()["valid"])

    def test_importing_feat_design_does_not_require_forbidden_dependencies(self) -> None:
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
            import research_platform.neuro.fsl.feat_design  # noqa: F401
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PACKAGE_ROOT.parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
