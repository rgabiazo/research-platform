"""Metadata-only parsing helpers for FSL FEAT ``design.fsf`` files.

The helpers in this module intentionally parse only FEAT design metadata. They
do not run FSL, discover FEAT directories, check PE images, inspect NIfTI data,
or import optional neuroimaging dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
import re
import shlex


_EVTITLE_VARIABLE = re.compile(r"^fmri\(evtitle([0-9]+)\)$")
_EVTITLE_HINT = re.compile(r"\bfmri\s*\(\s*evtitle", re.IGNORECASE)
_BEST_EFFORT_EV_INDEX = re.compile(r"evtitle\s*([0-9]+)", re.IGNORECASE)
_FMRI_VARIABLE = re.compile(r"^fmri\(([A-Za-z0-9_.]+)\)$")
_CONNAME_REAL_VARIABLE = re.compile(r"^fmri\(conname_real\.([0-9]+)\)$")
_CONNAME_REAL_HINT = re.compile(r"\bfmri\s*\(\s*conname_real\s*\.", re.IGNORECASE)
_BEST_EFFORT_CONTRAST_NUMBER = re.compile(r"conname_real\s*\.\s*([0-9]+)", re.IGNORECASE)
_SAFE_CONDITION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FORBIDDEN_CONDITION_FIELDS = frozenset(
    {
        "pe",
        "pe_number",
        "pe_index",
        "cope",
        "cope_number",
        "contrast_number",
        "design_pe",
        "feat_pe",
        "varcope",
        "varcope_number",
    }
)
_MISSING_POLICY_ERROR = frozenset({"error", "fail"})
_MISSING_POLICY_WARNING = frozenset({"skip", "warn", "warning"})


@dataclass(frozen=True)
class FslEvTitle:
    """One parsed FEAT EV-title row."""

    ev_index: int | None
    ev_title: str | None
    source_line: str
    line_number: int
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class FslDesignParseResult:
    """Parsed FEAT EV-title metadata from a ``design.fsf`` payload."""

    ev_titles: tuple[FslEvTitle, ...]
    ev_main_pe_numbers: Mapping[int, int] = field(default_factory=dict)
    ev_column_counts: Mapping[int, int] = field(default_factory=dict)
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status != "error"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        return payload


@dataclass(frozen=True)
class FslContrastName:
    """One parsed FEAT contrast-name row."""

    contrast_number: int | None
    contrast_name: str | None
    source_line: str
    line_number: int
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class FslContrastNameParseResult:
    """Parsed FEAT contrast-name metadata from a ``design.fsf`` payload."""

    contrasts: tuple[FslContrastName, ...]
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status != "error"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        return payload


@dataclass(frozen=True)
class EvTitlePeMapping:
    """A metadata-only EV-title to PE image-name mapping row."""

    ev_index: int | None
    ev_title: str | None
    pe_number: int | None
    pe_filename: str | None
    source_line: str
    line_number: int
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class EvTitlePeMappingResult:
    """Result rows for deriving PE numbers from FEAT EV indexes."""

    mappings: tuple[EvTitlePeMapping, ...]
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status != "error"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        return payload


@dataclass(frozen=True)
class ContrastCopeMapping:
    """A metadata-only contrast-name to COPE/VARCOPE image-name mapping row."""

    contrast_number: int | None
    contrast_name: str | None
    cope_number: int | None
    cope_filename: str | None
    varcope_number: int | None
    varcope_filename: str | None
    source_line: str
    line_number: int
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ContrastCopeMappingResult:
    """Result rows for deriving COPE/VARCOPE numbers from FEAT contrasts."""

    mappings: tuple[ContrastCopeMapping, ...]
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status != "error"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        return payload


@dataclass(frozen=True)
class ConditionPeMapping:
    """A condition-to-PE mapping row derived from EV-title metadata."""

    condition_id: str | None
    requested_ev_title: str | None
    matched_ev_title: str | None
    matched_by: str | None
    ev_index: int | None
    pe_number: int | None
    pe_filename: str | None
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ConditionPeMappingResult:
    """Result rows for matching configured conditions to FEAT PE metadata."""

    mappings: tuple[ConditionPeMapping, ...]
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status != "error"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        return payload


@dataclass(frozen=True)
class ContrastAliasCopeMapping:
    """A configured contrast alias resolved to derived COPE/VARCOPE numbers."""

    contrast_id: str | None
    requested_contrast_name: str | None
    matched_contrast_name: str | None
    matched_by: str | None
    contrast_number: int | None
    cope_number: int | None
    cope_filename: str | None
    varcope_number: int | None
    varcope_filename: str | None
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class ContrastAliasCopeMappingResult:
    """Result rows for matching configured contrast aliases to FEAT metadata."""

    mappings: tuple[ContrastAliasCopeMapping, ...]
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status != "error"

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["valid"] = self.valid
        return payload


def parse_fsl_design_ev_titles(text: str) -> FslDesignParseResult:
    """Parse FEAT EV-title lines from a ``design.fsf`` text payload.

    Content problems are reported on result rows and on the aggregate result;
    malformed design content does not raise.
    """

    drafts: list[dict[str, Any]] = []
    fmri_settings: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        source_line = raw_line.rstrip("\r\n")
        stripped = source_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        try:
            tokens = shlex.split(stripped, comments=True, posix=True)
        except ValueError as exc:
            if _EVTITLE_HINT.search(stripped):
                drafts.append(
                    _ev_title_draft(
                        ev_index=_best_effort_ev_index(stripped),
                        ev_title=None,
                        source_line=source_line,
                        line_number=line_number,
                        errors=(f"Malformed evtitle line: {exc}",),
                    )
                )
            continue

        if not tokens:
            continue

        if len(tokens) == 3 and tokens[0] == "set":
            variable_match = _FMRI_VARIABLE.fullmatch(tokens[1])
            if variable_match is not None:
                fmri_settings[variable_match.group(1)] = tokens[2]

        if not _EVTITLE_HINT.search(stripped):
            continue

        if len(tokens) != 3 or tokens[0] != "set":
            drafts.append(
                _ev_title_draft(
                    ev_index=_best_effort_ev_index(stripped),
                    ev_title=_best_effort_ev_title(tokens),
                    source_line=source_line,
                    line_number=line_number,
                    errors=("Malformed evtitle line: expected `set fmri(evtitleN) <title>`.",),
                )
            )
            continue

        variable_match = _EVTITLE_VARIABLE.fullmatch(tokens[1])
        if variable_match is None:
            drafts.append(
                _ev_title_draft(
                    ev_index=_best_effort_ev_index(tokens[1]),
                    ev_title=tokens[2],
                    source_line=source_line,
                    line_number=line_number,
                    errors=("Malformed evtitle line: expected FEAT variable `fmri(evtitleN)`.",),
                )
            )
            continue

        ev_index = int(variable_match.group(1))
        ev_title = tokens[2]
        errors: list[str] = []
        if ev_index <= 0:
            errors.append("EV index must be greater than zero.")
        if ev_title == "":
            errors.append("EV title is empty.")
        drafts.append(
            _ev_title_draft(
                ev_index=ev_index,
                ev_title=ev_title,
                source_line=source_line,
                line_number=line_number,
                errors=tuple(errors),
            )
        )

    _annotate_duplicate_ev_indexes(drafts)
    _annotate_duplicate_ev_titles(drafts)
    warnings = _missing_ev_index_warnings(drafts)
    rows = tuple(_finalize_ev_title(draft) for draft in drafts)
    ev_column_counts = _ev_column_counts(rows, fmri_settings)
    return FslDesignParseResult(
        ev_titles=rows,
        ev_main_pe_numbers=_ev_main_pe_numbers(rows, ev_column_counts),
        ev_column_counts=ev_column_counts,
        status=_aggregate_status(rows, warnings=warnings),
        warnings=warnings,
        errors=_aggregate_row_errors(rows),
    )


def parse_fsl_design_file(path: str | Path) -> FslDesignParseResult:
    """Read and parse a FEAT ``design.fsf`` file."""

    return parse_fsl_design_ev_titles(Path(path).read_text(encoding="utf-8"))


def parse_fsl_design_contrast_names(text: str) -> FslContrastNameParseResult:
    """Parse FEAT contrast-name lines from a ``design.fsf`` text payload.

    Only ``fmri(conname_real.N)`` metadata is parsed. COPE/VARCOPE numbers are
    later derived from the parsed contrast number; configured numeric selectors
    are intentionally not accepted.
    """

    drafts: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        source_line = raw_line.rstrip("\r\n")
        stripped = source_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not _CONNAME_REAL_HINT.search(stripped):
            continue

        try:
            tokens = shlex.split(stripped, comments=True, posix=True)
        except ValueError as exc:
            drafts.append(
                _contrast_name_draft(
                    contrast_number=_best_effort_contrast_number(stripped),
                    contrast_name=None,
                    source_line=source_line,
                    line_number=line_number,
                    errors=(f"Malformed contrast-name line: {exc}",),
                )
            )
            continue

        if not tokens:
            continue

        if len(tokens) != 3 or tokens[0] != "set":
            drafts.append(
                _contrast_name_draft(
                    contrast_number=_best_effort_contrast_number(stripped),
                    contrast_name=_best_effort_contrast_name(tokens),
                    source_line=source_line,
                    line_number=line_number,
                    errors=("Malformed contrast-name line: expected `set fmri(conname_real.N) <name>`.",),
                )
            )
            continue

        variable_match = _CONNAME_REAL_VARIABLE.fullmatch(tokens[1])
        if variable_match is None:
            drafts.append(
                _contrast_name_draft(
                    contrast_number=_best_effort_contrast_number(tokens[1]),
                    contrast_name=tokens[2],
                    source_line=source_line,
                    line_number=line_number,
                    errors=("Malformed contrast-name line: expected FEAT variable `fmri(conname_real.N)`.",),
                )
            )
            continue

        contrast_number = int(variable_match.group(1))
        contrast_name = tokens[2]
        errors: list[str] = []
        if contrast_number <= 0:
            errors.append("Contrast number must be greater than zero.")
        if contrast_name == "":
            errors.append("Contrast name is empty.")
        drafts.append(
            _contrast_name_draft(
                contrast_number=contrast_number,
                contrast_name=contrast_name,
                source_line=source_line,
                line_number=line_number,
                errors=tuple(errors),
            )
        )

    _annotate_duplicate_contrast_numbers(drafts)
    _annotate_duplicate_contrast_names(drafts)
    warnings = _missing_contrast_number_warnings(drafts)
    rows = tuple(_finalize_contrast_name(draft) for draft in drafts)
    return FslContrastNameParseResult(
        contrasts=rows,
        status=_aggregate_status(rows, warnings=warnings),
        warnings=warnings,
        errors=_aggregate_row_errors(rows),
    )


def parse_fsl_design_contrast_file(path: str | Path) -> FslContrastNameParseResult:
    """Read and parse FEAT contrast names from a ``design.fsf`` file."""

    return parse_fsl_design_contrast_names(Path(path).read_text(encoding="utf-8"))


def map_contrast_names_to_cope_numbers(
    contrasts: FslContrastNameParseResult | Sequence[FslContrastName],
) -> ContrastCopeMappingResult:
    """Derive COPE/VARCOPE mappings from FEAT contrast-name rows.

    COPE and VARCOPE numbers are derived only from ``contrast_number``. This
    function does not accept configured COPE, VARCOPE, or contrast numbers and
    does not check image existence.
    """

    rows = _contrast_name_rows(contrasts)
    drafts: list[dict[str, Any]] = []
    for row in rows:
        errors = list(row.errors)
        warnings = list(row.warnings)
        cope_number = row.contrast_number if row.contrast_number is not None else None
        varcope_number = row.contrast_number if row.contrast_number is not None else None
        drafts.append(
            {
                "contrast_number": row.contrast_number,
                "contrast_name": row.contrast_name,
                "cope_number": cope_number,
                "cope_filename": f"cope{cope_number}.nii.gz" if cope_number is not None else None,
                "varcope_number": varcope_number,
                "varcope_filename": f"varcope{varcope_number}.nii.gz" if varcope_number is not None else None,
                "source_line": row.source_line,
                "line_number": row.line_number,
                "warnings": warnings,
                "errors": errors,
            }
        )

    mappings = tuple(
        ContrastCopeMapping(
            contrast_number=draft["contrast_number"],
            contrast_name=draft["contrast_name"],
            cope_number=draft["cope_number"],
            cope_filename=draft["cope_filename"],
            varcope_number=draft["varcope_number"],
            varcope_filename=draft["varcope_filename"],
            source_line=draft["source_line"],
            line_number=draft["line_number"],
            status=_row_status(draft["warnings"], draft["errors"]),
            warnings=tuple(draft["warnings"]),
            errors=tuple(draft["errors"]),
        )
        for draft in drafts
    )
    return ContrastCopeMappingResult(
        mappings=mappings,
        status=_aggregate_status(mappings),
        warnings=_aggregate_row_warnings(mappings),
        errors=_aggregate_row_errors(mappings),
    )


def resolve_contrast_aliases_to_cope_numbers(
    aliases: Sequence[Any],
    contrasts: FslContrastNameParseResult | ContrastCopeMappingResult | Sequence[FslContrastName] | Sequence[ContrastCopeMapping],
    *,
    case_sensitive: bool = True,
    missing_policy: str = "error",
) -> ContrastAliasCopeMappingResult:
    """Match configured contrast aliases to derived COPE/VARCOPE numbers.

    Alias objects are read by duck typing. Mappings and ordinary objects may
    provide ``id``, ``name``, ``condition``, ``contrast_id`` or
    ``source_contrast`` for the alias identifier, optional ``contrast_name`` or
    ``fsl_contrast_name`` for an exact FEAT contrast name, and optional
    ``aliases``. Numeric contrast, COPE, and VARCOPE fields are rejected.
    """

    missing_policy = _normalize_missing_policy(missing_policy)
    contrast_rows = _contrast_mapping_rows(contrasts)
    name_index = _contrast_name_index(contrast_rows)
    folded_name_index = _folded_contrast_name_index(contrast_rows)

    drafts: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    for alias in aliases:
        warnings: list[str] = []
        errors: list[str] = []
        contrast_id = _contrast_alias_id(alias)
        if contrast_id is None:
            errors.append("Contrast alias id must be defined from `id`, `name`, `condition`, `contrast_id`, or `source_contrast`.")
        elif not _SAFE_CONDITION_ID.fullmatch(contrast_id):
            errors.append(f"Contrast alias id is not safe: {contrast_id}.")
        else:
            seen_ids[contrast_id] = seen_ids.get(contrast_id, 0) + 1

        for field_path in _forbidden_contrast_alias_field_paths(alias):
            errors.append(f"Contrast alias payload must not hard-code contrast/COPE selector field `{field_path}`.")

        requested_contrast_name = _contrast_requested_name(alias)
        candidates: list[tuple[str, str]] = []
        if requested_contrast_name is not None:
            candidates.append(("contrast_name", requested_contrast_name))
        else:
            candidates.extend(("alias", name) for name in _contrast_aliases(alias))

        matched_by: str | None = None
        matched_row: ContrastCopeMapping | None = None
        if not errors:
            for match_kind, candidate_name in candidates:
                matches = _match_contrast_name(
                    candidate_name,
                    name_index=name_index,
                    folded_name_index=folded_name_index,
                    case_sensitive=case_sensitive,
                )
                if len(matches) == 1:
                    matched_by = match_kind
                    matched_row = matches[0]
                    break
                if len(matches) > 1:
                    errors.append(f"Contrast name match is ambiguous for alias {contrast_id}: {candidate_name}.")
                    matched_by = match_kind
                    break
            if matched_row is None and not errors:
                message = _missing_contrast_alias_message(contrast_id, candidates)
                if missing_policy in _MISSING_POLICY_WARNING:
                    warnings.append(message)
                else:
                    errors.append(message)

        drafts.append(
            {
                "contrast_id": contrast_id,
                "requested_contrast_name": requested_contrast_name,
                "matched_contrast_name": matched_row.contrast_name if matched_row is not None else None,
                "matched_by": matched_by,
                "contrast_number": matched_row.contrast_number if matched_row is not None else None,
                "cope_number": matched_row.cope_number if matched_row is not None else None,
                "cope_filename": matched_row.cope_filename if matched_row is not None else None,
                "varcope_number": matched_row.varcope_number if matched_row is not None else None,
                "varcope_filename": matched_row.varcope_filename if matched_row is not None else None,
                "warnings": warnings,
                "errors": errors,
            }
        )

    for duplicate in sorted(identifier for identifier, count in seen_ids.items() if count > 1):
        for draft in drafts:
            if draft["contrast_id"] == duplicate:
                draft["errors"].append(f"Duplicate contrast alias id: {duplicate}.")
    _annotate_duplicate_cope_assignments(drafts)

    mappings = tuple(
        ContrastAliasCopeMapping(
            contrast_id=draft["contrast_id"],
            requested_contrast_name=draft["requested_contrast_name"],
            matched_contrast_name=draft["matched_contrast_name"],
            matched_by=draft["matched_by"],
            contrast_number=draft["contrast_number"],
            cope_number=draft["cope_number"],
            cope_filename=draft["cope_filename"],
            varcope_number=draft["varcope_number"],
            varcope_filename=draft["varcope_filename"],
            status=_row_status(draft["warnings"], draft["errors"]),
            warnings=tuple(draft["warnings"]),
            errors=tuple(draft["errors"]),
        )
        for draft in drafts
    )
    return ContrastAliasCopeMappingResult(
        mappings=mappings,
        status=_aggregate_status(mappings),
        warnings=_aggregate_row_warnings(mappings),
        errors=_aggregate_row_errors(mappings),
    )


def map_ev_titles_to_pe_numbers(ev_titles: FslDesignParseResult | Sequence[FslEvTitle]) -> EvTitlePeMappingResult:
    """Derive metadata-only PE mappings from FEAT EV-title rows.

    PE numbers are derived from the main-effect design column for each EV.
    Temporal derivatives and extra basis-function columns on preceding EVs are
    counted, so PE numbers are not assumed to equal EV indexes. This function
    does not accept configured PE numbers and does not check image existence.
    """

    rows = _ev_title_rows(ev_titles)
    main_pe_numbers = ev_titles.ev_main_pe_numbers if isinstance(ev_titles, FslDesignParseResult) else {}
    drafts: list[dict[str, Any]] = []
    for row in rows:
        errors = list(row.errors)
        warnings = list(row.warnings)
        pe_number = main_pe_numbers.get(row.ev_index, row.ev_index) if row.ev_index is not None else None
        pe_filename = f"pe{pe_number}.nii.gz" if pe_number is not None else None
        drafts.append(
            {
                "ev_index": row.ev_index,
                "ev_title": row.ev_title,
                "pe_number": pe_number,
                "pe_filename": pe_filename,
                "source_line": row.source_line,
                "line_number": row.line_number,
                "warnings": warnings,
                "errors": errors,
            }
        )

    _annotate_duplicate_pe_assignments(drafts, label_key="ev_title")
    mappings = tuple(
        EvTitlePeMapping(
            ev_index=draft["ev_index"],
            ev_title=draft["ev_title"],
            pe_number=draft["pe_number"],
            pe_filename=draft["pe_filename"],
            source_line=draft["source_line"],
            line_number=draft["line_number"],
            status=_row_status(draft["warnings"], draft["errors"]),
            warnings=tuple(draft["warnings"]),
            errors=tuple(draft["errors"]),
        )
        for draft in drafts
    )
    return EvTitlePeMappingResult(
        mappings=mappings,
        status=_aggregate_status(mappings),
        errors=_aggregate_row_errors(mappings),
        warnings=_aggregate_row_warnings(mappings),
    )


def map_conditions_to_pe_numbers(
    conditions: Sequence[Any],
    ev_titles: FslDesignParseResult | EvTitlePeMappingResult | Sequence[FslEvTitle] | Sequence[EvTitlePeMapping],
    *,
    case_sensitive: bool = True,
    missing_policy: str = "error",
) -> ConditionPeMappingResult:
    """Match condition declarations to FEAT PE numbers using EV titles.

    Condition objects are read by duck typing. Mappings and ordinary objects may
    provide ``id`` or ``condition_id``, optional ``ev_title`` or
    ``fsl_ev_title``, and optional ``aliases``. No PE or COPE fields are
    accepted from condition payloads.
    """

    missing_policy = _normalize_missing_policy(missing_policy)
    ev_rows = _condition_ev_rows(ev_titles)
    title_index = _title_index(ev_rows)
    folded_title_index = _folded_title_index(ev_rows)

    drafts: list[dict[str, Any]] = []
    for condition in conditions:
        warnings: list[str] = []
        errors: list[str] = []
        condition_id = _condition_id(condition)
        if condition_id is None:
            errors.append("Condition id must be defined from `id` or `condition_id`.")
        elif not _SAFE_CONDITION_ID.fullmatch(condition_id):
            errors.append(f"Condition id is not safe: {condition_id}.")

        for field_path in _forbidden_condition_field_paths(condition):
            errors.append(f"Condition payload must not hard-code PE/COPE selector field `{field_path}`.")

        requested_ev_title = _condition_requested_ev_title(condition)
        candidates: list[tuple[str, str]] = []
        if requested_ev_title is not None:
            candidates.append(("ev_title", requested_ev_title))
        else:
            candidates.extend(("alias", alias) for alias in _condition_aliases(condition))

        matched_by: str | None = None
        matched_row: EvTitlePeMapping | None = None
        if not errors:
            for match_kind, candidate_title in candidates:
                matches = _match_title(
                    candidate_title,
                    title_index=title_index,
                    folded_title_index=folded_title_index,
                    case_sensitive=case_sensitive,
                )
                if len(matches) == 1:
                    matched_by = match_kind
                    matched_row = matches[0]
                    break
                if len(matches) > 1:
                    errors.append(f"EV title match is ambiguous for condition {condition_id}: {candidate_title}.")
                    matched_by = match_kind
                    break
            if matched_row is None and not errors:
                message = _missing_condition_message(condition_id, candidates)
                if missing_policy in _MISSING_POLICY_WARNING:
                    warnings.append(message)
                else:
                    errors.append(message)

        drafts.append(
            {
                "condition_id": condition_id,
                "requested_ev_title": requested_ev_title,
                "matched_ev_title": matched_row.ev_title if matched_row is not None else None,
                "matched_by": matched_by,
                "ev_index": matched_row.ev_index if matched_row is not None else None,
                "pe_number": matched_row.pe_number if matched_row is not None else None,
                "pe_filename": matched_row.pe_filename if matched_row is not None else None,
                "warnings": warnings,
                "errors": errors,
            }
        )

    _annotate_duplicate_pe_assignments(drafts, label_key="condition_id")
    mappings = tuple(
        ConditionPeMapping(
            condition_id=draft["condition_id"],
            requested_ev_title=draft["requested_ev_title"],
            matched_ev_title=draft["matched_ev_title"],
            matched_by=draft["matched_by"],
            ev_index=draft["ev_index"],
            pe_number=draft["pe_number"],
            pe_filename=draft["pe_filename"],
            status=_row_status(draft["warnings"], draft["errors"]),
            warnings=tuple(draft["warnings"]),
            errors=tuple(draft["errors"]),
        )
        for draft in drafts
    )
    return ConditionPeMappingResult(
        mappings=mappings,
        status=_aggregate_status(mappings),
        warnings=_aggregate_row_warnings(mappings),
        errors=_aggregate_row_errors(mappings),
    )


def _contrast_name_draft(
    *,
    contrast_number: int | None,
    contrast_name: str | None,
    source_line: str,
    line_number: int,
    warnings: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "contrast_number": contrast_number,
        "contrast_name": contrast_name,
        "source_line": source_line,
        "line_number": line_number,
        "warnings": list(warnings),
        "errors": list(errors),
    }


def _finalize_contrast_name(draft: Mapping[str, Any]) -> FslContrastName:
    warnings = tuple(draft["warnings"])
    errors = tuple(draft["errors"])
    return FslContrastName(
        contrast_number=draft["contrast_number"],
        contrast_name=draft["contrast_name"],
        source_line=draft["source_line"],
        line_number=draft["line_number"],
        status=_row_status(warnings, errors),
        warnings=warnings,
        errors=errors,
    )


def _ev_title_draft(
    *,
    ev_index: int | None,
    ev_title: str | None,
    source_line: str,
    line_number: int,
    warnings: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "ev_index": ev_index,
        "ev_title": ev_title,
        "source_line": source_line,
        "line_number": line_number,
        "warnings": list(warnings),
        "errors": list(errors),
    }


def _finalize_ev_title(draft: Mapping[str, Any]) -> FslEvTitle:
    warnings = tuple(draft["warnings"])
    errors = tuple(draft["errors"])
    return FslEvTitle(
        ev_index=draft["ev_index"],
        ev_title=draft["ev_title"],
        source_line=draft["source_line"],
        line_number=draft["line_number"],
        status=_row_status(warnings, errors),
        warnings=warnings,
        errors=errors,
    )


def _best_effort_ev_index(text: str) -> int | None:
    match = _BEST_EFFORT_EV_INDEX.search(text)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _best_effort_ev_title(tokens: Sequence[str]) -> str | None:
    if len(tokens) < 3:
        return None
    return " ".join(tokens[2:])


def _best_effort_contrast_number(text: str) -> int | None:
    match = _BEST_EFFORT_CONTRAST_NUMBER.search(text)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _best_effort_contrast_name(tokens: Sequence[str]) -> str | None:
    if len(tokens) < 3:
        return None
    return " ".join(tokens[2:])


def _annotate_duplicate_ev_indexes(drafts: list[dict[str, Any]]) -> None:
    by_index: dict[int, list[dict[str, Any]]] = {}
    for draft in drafts:
        ev_index = draft["ev_index"]
        if ev_index is not None:
            by_index.setdefault(ev_index, []).append(draft)
    for ev_index, duplicate_rows in by_index.items():
        if len(duplicate_rows) > 1:
            for draft in duplicate_rows:
                draft["errors"].append(f"Duplicate EV index: {ev_index}.")


def _annotate_duplicate_ev_titles(drafts: list[dict[str, Any]]) -> None:
    by_title: dict[str, list[dict[str, Any]]] = {}
    for draft in drafts:
        ev_title = draft["ev_title"]
        if ev_title:
            by_title.setdefault(ev_title, []).append(draft)
    for ev_title, duplicate_rows in by_title.items():
        if len(duplicate_rows) > 1:
            for draft in duplicate_rows:
                draft["warnings"].append(f"Duplicate EV title: {ev_title}.")


def _annotate_duplicate_contrast_numbers(drafts: list[dict[str, Any]]) -> None:
    by_number: dict[int, list[dict[str, Any]]] = {}
    for draft in drafts:
        contrast_number = draft["contrast_number"]
        if contrast_number is not None:
            by_number.setdefault(contrast_number, []).append(draft)
    for contrast_number, duplicate_rows in by_number.items():
        if len(duplicate_rows) > 1:
            for draft in duplicate_rows:
                draft["errors"].append(f"Duplicate contrast number: {contrast_number}.")


def _annotate_duplicate_contrast_names(drafts: list[dict[str, Any]]) -> None:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for draft in drafts:
        contrast_name = draft["contrast_name"]
        if contrast_name:
            by_name.setdefault(contrast_name, []).append(draft)
    for contrast_name, duplicate_rows in by_name.items():
        if len(duplicate_rows) > 1:
            for draft in duplicate_rows:
                draft["warnings"].append(f"Duplicate contrast name: {contrast_name}.")


def _missing_ev_index_warnings(drafts: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    indexes = sorted(
        {
            draft["ev_index"]
            for draft in drafts
            if isinstance(draft.get("ev_index"), int) and draft["ev_index"] > 0
        }
    )
    if not indexes:
        return ()
    missing = [index for index in range(1, indexes[-1] + 1) if index not in indexes]
    if not missing:
        return ()
    return (f"Missing EV index(es): {', '.join(str(index) for index in missing)}.",)


def _ev_column_counts(
    rows: Sequence[FslEvTitle],
    fmri_settings: Mapping[str, str],
) -> dict[int, int]:
    indexes = sorted({row.ev_index for row in rows if row.ev_index is not None and row.ev_index > 0})
    if not indexes:
        return {}
    return {index: _ev_column_count(index, fmri_settings) for index in range(1, indexes[-1] + 1)}


def _ev_column_count(ev_index: int, fmri_settings: Mapping[str, str]) -> int:
    basis_count = _positive_int(fmri_settings.get(f"basisfnum{ev_index}")) or 1
    derivative_count = 1 if _truthy_fsf_value(fmri_settings.get(f"deriv_yn{ev_index}")) else 0
    return max(1, basis_count) + derivative_count


def _ev_main_pe_numbers(
    rows: Sequence[FslEvTitle],
    ev_column_counts: Mapping[int, int],
) -> dict[int, int]:
    indexes = sorted({row.ev_index for row in rows if row.ev_index is not None and row.ev_index > 0})
    if not indexes:
        return {}
    pe_numbers: dict[int, int] = {}
    next_column = 1
    for ev_index in range(1, indexes[-1] + 1):
        if ev_index in indexes:
            pe_numbers[ev_index] = next_column
        next_column += ev_column_counts.get(ev_index, 1)
    return pe_numbers


def _positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _truthy_fsf_value(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _missing_contrast_number_warnings(drafts: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    numbers = sorted(
        {
            draft["contrast_number"]
            for draft in drafts
            if isinstance(draft.get("contrast_number"), int) and draft["contrast_number"] > 0
        }
    )
    if not numbers:
        return ()
    missing = [number for number in range(1, numbers[-1] + 1) if number not in numbers]
    if not missing:
        return ()
    return (f"Missing contrast number(s): {', '.join(str(number) for number in missing)}.",)


def _ev_title_rows(ev_titles: FslDesignParseResult | Sequence[FslEvTitle]) -> tuple[FslEvTitle, ...]:
    if isinstance(ev_titles, FslDesignParseResult):
        return ev_titles.ev_titles
    return tuple(ev_titles)


def _contrast_name_rows(contrasts: FslContrastNameParseResult | Sequence[FslContrastName]) -> tuple[FslContrastName, ...]:
    if isinstance(contrasts, FslContrastNameParseResult):
        return contrasts.contrasts
    return tuple(contrasts)


def _condition_ev_rows(
    ev_titles: FslDesignParseResult | EvTitlePeMappingResult | Sequence[FslEvTitle] | Sequence[EvTitlePeMapping],
) -> tuple[EvTitlePeMapping, ...]:
    if isinstance(ev_titles, EvTitlePeMappingResult):
        return ev_titles.mappings
    if isinstance(ev_titles, FslDesignParseResult):
        return map_ev_titles_to_pe_numbers(ev_titles).mappings

    rows = tuple(ev_titles)
    if not rows:
        return ()
    if isinstance(rows[0], EvTitlePeMapping):
        return rows  # type: ignore[return-value]
    return map_ev_titles_to_pe_numbers(rows).mappings  # type: ignore[arg-type]


def _contrast_mapping_rows(
    contrasts: FslContrastNameParseResult | ContrastCopeMappingResult | Sequence[FslContrastName] | Sequence[ContrastCopeMapping],
) -> tuple[ContrastCopeMapping, ...]:
    if isinstance(contrasts, ContrastCopeMappingResult):
        return contrasts.mappings
    if isinstance(contrasts, FslContrastNameParseResult):
        return map_contrast_names_to_cope_numbers(contrasts).mappings

    rows = tuple(contrasts)
    if not rows:
        return ()
    if isinstance(rows[0], ContrastCopeMapping):
        return rows  # type: ignore[return-value]
    return map_contrast_names_to_cope_numbers(rows).mappings  # type: ignore[arg-type]


def _title_index(rows: Sequence[EvTitlePeMapping]) -> dict[str, tuple[EvTitlePeMapping, ...]]:
    by_title: dict[str, list[EvTitlePeMapping]] = {}
    for row in rows:
        if row.ev_title is not None and not row.errors:
            by_title.setdefault(row.ev_title, []).append(row)
    return {title: tuple(title_rows) for title, title_rows in by_title.items()}


def _folded_title_index(rows: Sequence[EvTitlePeMapping]) -> dict[str, tuple[EvTitlePeMapping, ...]]:
    by_title: dict[str, list[EvTitlePeMapping]] = {}
    for row in rows:
        if row.ev_title is not None and not row.errors:
            by_title.setdefault(row.ev_title.casefold(), []).append(row)
    return {title: tuple(title_rows) for title, title_rows in by_title.items()}


def _contrast_name_index(rows: Sequence[ContrastCopeMapping]) -> dict[str, tuple[ContrastCopeMapping, ...]]:
    by_name: dict[str, list[ContrastCopeMapping]] = {}
    for row in rows:
        if row.contrast_name is not None and not row.errors:
            by_name.setdefault(row.contrast_name, []).append(row)
    return {name: tuple(name_rows) for name, name_rows in by_name.items()}


def _folded_contrast_name_index(rows: Sequence[ContrastCopeMapping]) -> dict[str, tuple[ContrastCopeMapping, ...]]:
    by_name: dict[str, list[ContrastCopeMapping]] = {}
    for row in rows:
        if row.contrast_name is not None and not row.errors:
            by_name.setdefault(row.contrast_name.casefold(), []).append(row)
    return {name: tuple(name_rows) for name, name_rows in by_name.items()}


def _match_title(
    title: str,
    *,
    title_index: Mapping[str, tuple[EvTitlePeMapping, ...]],
    folded_title_index: Mapping[str, tuple[EvTitlePeMapping, ...]],
    case_sensitive: bool,
) -> tuple[EvTitlePeMapping, ...]:
    if case_sensitive:
        return title_index.get(title, ())
    return folded_title_index.get(title.casefold(), ())


def _match_contrast_name(
    name: str,
    *,
    name_index: Mapping[str, tuple[ContrastCopeMapping, ...]],
    folded_name_index: Mapping[str, tuple[ContrastCopeMapping, ...]],
    case_sensitive: bool,
) -> tuple[ContrastCopeMapping, ...]:
    if case_sensitive:
        return name_index.get(name, ())
    return folded_name_index.get(name.casefold(), ())


def _condition_id(condition: Any) -> str | None:
    return _optional_text(_condition_value(condition, "id", "condition_id"))


def _condition_requested_ev_title(condition: Any) -> str | None:
    return _optional_text(_condition_value(condition, "ev_title", "fsl_ev_title"))


def _condition_aliases(condition: Any) -> tuple[str, ...]:
    aliases = _condition_value(condition, "aliases")
    if aliases is None:
        return ()
    if isinstance(aliases, str):
        text = aliases.strip()
        return (text,) if text else ()
    if isinstance(aliases, Sequence):
        return tuple(text for item in aliases if (text := _optional_text(item)) is not None)
    return ()


def _contrast_alias_id(alias: Any) -> str | None:
    return _optional_text(_condition_value(alias, "id", "name", "condition", "contrast_id", "source_contrast"))


def _contrast_requested_name(alias: Any) -> str | None:
    return _optional_text(_condition_value(alias, "contrast_name", "fsl_contrast_name", "con_name"))


def _contrast_aliases(alias: Any) -> tuple[str, ...]:
    aliases = _condition_value(alias, "aliases", "contrast_aliases", "names")
    if aliases is None:
        return ()
    if isinstance(aliases, str):
        text = aliases.strip()
        return (text,) if text else ()
    if isinstance(aliases, Sequence):
        return tuple(text for item in aliases if (text := _optional_text(item)) is not None)
    return ()


def _condition_value(condition: Any, *names: str) -> Any:
    if isinstance(condition, Mapping):
        for name in names:
            if name in condition:
                return condition[name]
        return None

    for name in names:
        if hasattr(condition, name):
            return getattr(condition, name)

    fields_payload = getattr(condition, "fields", None)
    if isinstance(fields_payload, Mapping):
        for name in names:
            if name in fields_payload:
                return fields_payload[name]
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _forbidden_condition_field_paths(condition: Any) -> tuple[str, ...]:
    seen: set[int] = set()
    paths = _forbidden_field_paths(condition, path="", seen=seen)
    return tuple(paths)


def _forbidden_contrast_alias_field_paths(alias: Any) -> tuple[str, ...]:
    forbidden = {
        "cope",
        "cope_index",
        "cope_number",
        "contrast_index",
        "contrast_number",
        "varcope",
        "varcope_index",
        "varcope_number",
    }
    seen: set[int] = set()
    return tuple(_forbidden_field_paths(alias, path="", seen=seen, forbidden=forbidden))


def _forbidden_field_paths(
    value: Any,
    *,
    path: str,
    seen: set[int],
    forbidden: frozenset[str] | set[str] = _FORBIDDEN_CONDITION_FIELDS,
) -> list[str]:
    value_id = id(value)
    if value_id in seen:
        return []
    if isinstance(value, (Mapping, list, tuple)) or (is_dataclass(value) and not isinstance(value, type)):
        seen.add(value_id)

    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text.lower() in forbidden:
                paths.append(child_path)
            paths.extend(_forbidden_field_paths(child, path=child_path, seen=seen, forbidden=forbidden))
        return paths

    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_forbidden_field_paths(child, path=f"{path}[{index}]", seen=seen, forbidden=forbidden))
        return paths

    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            child_path = f"{path}.{field.name}" if path else field.name
            child = getattr(value, field.name)
            if field.name.lower() in forbidden:
                paths.append(child_path)
            paths.extend(_forbidden_field_paths(child, path=child_path, seen=seen, forbidden=forbidden))
        return paths

    for field_name in forbidden:
        if hasattr(value, field_name):
            paths.append(f"{path}.{field_name}" if path else field_name)
    for field_name in ("selector", "pattern_selector", "fields"):
        child = getattr(value, field_name, None)
        if isinstance(child, Mapping):
            child_path = f"{path}.{field_name}" if path else field_name
            paths.extend(_forbidden_field_paths(child, path=child_path, seen=seen, forbidden=forbidden))
    return paths


def _missing_condition_message(condition_id: str | None, candidates: Sequence[tuple[str, str]]) -> str:
    label = condition_id or "<missing-id>"
    if candidates:
        requested = ", ".join(title for _, title in candidates)
        return f"No EV title match for condition {label}: {requested}."
    return f"No EV title or aliases available for condition {label}."


def _missing_contrast_alias_message(contrast_id: str | None, candidates: Sequence[tuple[str, str]]) -> str:
    label = contrast_id or "<missing-id>"
    if candidates:
        requested = ", ".join(name for _, name in candidates)
        return f"No contrast name match for alias {label}: {requested}."
    return f"No contrast name or aliases available for alias {label}."


def _normalize_missing_policy(value: str) -> str:
    policy = str(value).strip().lower()
    if policy in _MISSING_POLICY_ERROR or policy in _MISSING_POLICY_WARNING:
        return policy
    supported = ", ".join(sorted(_MISSING_POLICY_ERROR | _MISSING_POLICY_WARNING))
    raise ValueError(f"missing_policy must be one of: {supported}.")


def _annotate_duplicate_pe_assignments(drafts: list[dict[str, Any]], *, label_key: str) -> None:
    by_pe: dict[int, list[dict[str, Any]]] = {}
    for draft in drafts:
        pe_number = draft.get("pe_number")
        if pe_number is not None:
            by_pe.setdefault(pe_number, []).append(draft)
    for pe_number, duplicate_rows in by_pe.items():
        if len(duplicate_rows) > 1:
            labels = ", ".join(str(row.get(label_key) or "<unknown>") for row in duplicate_rows)
            for draft in duplicate_rows:
                draft["errors"].append(f"Duplicate PE assignment for pe{pe_number}: {labels}.")


def _annotate_duplicate_cope_assignments(drafts: list[dict[str, Any]]) -> None:
    by_cope: dict[int, list[dict[str, Any]]] = {}
    for draft in drafts:
        cope_number = draft.get("cope_number")
        if cope_number is not None:
            by_cope.setdefault(cope_number, []).append(draft)
    for cope_number, duplicate_rows in by_cope.items():
        if len(duplicate_rows) > 1:
            labels = ", ".join(str(row.get("contrast_id") or "<unknown>") for row in duplicate_rows)
            for draft in duplicate_rows:
                draft["errors"].append(f"Duplicate COPE assignment for cope{cope_number}: {labels}.")


def _row_status(warnings: Sequence[str], errors: Sequence[str]) -> str:
    if errors:
        return "error"
    if warnings:
        return "warning"
    return "ok"


def _aggregate_status(rows: Sequence[Any], *, warnings: Sequence[str] = ()) -> str:
    if any(row.errors for row in rows):
        return "error"
    if warnings or any(row.warnings for row in rows):
        return "warning"
    return "ok"


def _aggregate_row_errors(rows: Sequence[Any]) -> tuple[str, ...]:
    errors: list[str] = []
    for row in rows:
        errors.extend(row.errors)
    return tuple(errors)


def _aggregate_row_warnings(rows: Sequence[Any]) -> tuple[str, ...]:
    warnings: list[str] = []
    for row in rows:
        warnings.extend(row.warnings)
    return tuple(warnings)


def _json_safe_dataclass(value: Any) -> dict[str, Any]:
    return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child) for child in value]
    return str(value)
