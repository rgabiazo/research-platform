
"""Dependency-free report document builders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
import html
import json
import math
from pathlib import Path
from string import Formatter
from typing import Any


@dataclass(frozen=True)
class ReportSectionSpec:
    """A generic text section in a report."""

    section_id: str
    heading: str | None = None
    text: str | None = None
    level: int = 2
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "section_id", _non_empty_text(self.section_id, field_name="section_id"))
        object.__setattr__(self, "level", _bounded_heading_level(self.level))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["section_type"] = "text"
        return payload


@dataclass(frozen=True)
class TableSectionSpec:
    """A table section rendered from already-computed rows."""

    section_id: str = "table"
    heading: str | None = None
    rows: Sequence[Mapping[str, Any]] = ()
    columns: Sequence[str] = ()
    required_columns: Sequence[str] = ()
    caption: str | None = None
    max_rows: int | None = None
    level: int = 2
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "section_id", _non_empty_text(self.section_id, field_name="section_id"))
        object.__setattr__(self, "rows", tuple(_json_safe_mapping(_as_mapping(row, field_name="rows")) for row in self.rows))
        object.__setattr__(self, "columns", tuple(str(column) for column in self.columns))
        object.__setattr__(self, "required_columns", tuple(str(column) for column in self.required_columns))
        if self.max_rows is not None and self.max_rows < 0:
            raise ValueError("max_rows must be non-negative when provided.")
        object.__setattr__(self, "level", _bounded_heading_level(self.level))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["section_type"] = "table"
        return payload


@dataclass(frozen=True)
class FigureSectionSpec:
    """A report section that references a pre-rendered figure artifact."""

    section_id: str = "figure"
    heading: str | None = None
    figure_path: str | Path | None = None
    plot_id: str | None = None
    caption: str | None = None
    alt_text: str | None = None
    level: int = 2
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "section_id", _non_empty_text(self.section_id, field_name="section_id"))
        if self.figure_path is not None:
            object.__setattr__(self, "figure_path", str(self.figure_path))
        if self.plot_id is not None:
            object.__setattr__(self, "plot_id", str(self.plot_id))
        object.__setattr__(self, "level", _bounded_heading_level(self.level))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload = _json_safe_dataclass(self)
        payload["section_type"] = "figure"
        return payload


@dataclass(frozen=True)
class ReportSpec:
    """Generic report document specification."""

    report_id: str = "report"
    title: str | None = None
    subtitle: str | None = None
    caption: str | None = None
    footnote: str | None = None
    alt_text: str | None = None
    methods_note: str | None = None
    sections: Sequence[ReportSectionSpec | TableSectionSpec | FigureSectionSpec | Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    include_artifact_list: bool = True
    include_visual_qc: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "report_id", _non_empty_text(self.report_id, field_name="report_id"))
        object.__setattr__(self, "sections", tuple(_coerce_section(section) for section in self.sections))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


def build_table_section(
    rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    section_id: str = "table",
    heading: str | None = None,
    columns: Sequence[str] = (),
    required_columns: Sequence[str] = (),
    caption: str | None = None,
    max_rows: int | None = None,
    level: int = 2,
    metadata: Mapping[str, Any] | None = None,
) -> TableSectionSpec:
    """Build a table section from caller-provided rows."""

    return TableSectionSpec(
        section_id=section_id,
        heading=heading,
        rows=tuple(rows or ()),
        columns=columns,
        required_columns=required_columns,
        caption=caption,
        max_rows=max_rows,
        level=level,
        metadata=metadata or {},
    )


def build_report_document(
    rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    report_spec: ReportSpec | None = None,
    format: str = "markdown",
    metadata: Mapping[str, Any] | None = None,
    figure_paths: Mapping[str, str] | None = None,
    visual_qc_rows: Sequence[Mapping[str, Any] | Any] = (),
    artifact_rows: Sequence[Mapping[str, Any] | Any] = (),
    warnings: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
) -> str:
    """Render a report document as Markdown, static HTML, or plain text."""

    normalized_format = str(format).lower()
    if normalized_format not in {"markdown", "html", "text"}:
        raise ValueError("Report format must be one of: markdown, html, text.")
    source_rows = _normalize_rows(rows or ())
    spec = report_spec or ReportSpec()
    merged_metadata = _merged_metadata(metadata, spec.metadata)
    spec = _expanded_report_spec(spec, merged_metadata)
    sections = _default_sections(spec, source_rows, figure_paths or {})
    if normalized_format == "markdown":
        return _markdown_report(
            source_rows,
            spec=spec,
            sections=sections,
            visual_qc_rows=visual_qc_rows,
            artifact_rows=artifact_rows,
            warnings=warnings,
            provenance=provenance or {},
        )
    if normalized_format == "html":
        return _html_report(
            source_rows,
            spec=spec,
            sections=sections,
            visual_qc_rows=visual_qc_rows,
            artifact_rows=artifact_rows,
            warnings=warnings,
            provenance=provenance or {},
        )
    return _text_report(
        source_rows,
        spec=spec,
        sections=sections,
        visual_qc_rows=visual_qc_rows,
        artifact_rows=artifact_rows,
        warnings=warnings,
        provenance=provenance or {},
    )


def _markdown_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: ReportSpec,
    sections: Sequence[ReportSectionSpec | TableSectionSpec | FigureSectionSpec],
    visual_qc_rows: Sequence[Mapping[str, Any] | Any],
    artifact_rows: Sequence[Mapping[str, Any] | Any],
    warnings: Sequence[str],
    provenance: Mapping[str, Any],
) -> str:
    lines: list[str] = []
    if spec.title:
        lines.extend((f"# {spec.title}", ""))
    if spec.subtitle:
        lines.extend((spec.subtitle, ""))
    for section in sections:
        lines.extend(_markdown_section(section, rows))
    lines.extend(_markdown_common_sections(spec, visual_qc_rows, artifact_rows, warnings, provenance))
    if spec.caption:
        lines.extend((spec.caption, ""))
    if spec.footnote:
        lines.extend((spec.footnote, ""))
    return "\n".join(lines).rstrip() + "\n"


def _html_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: ReportSpec,
    sections: Sequence[ReportSectionSpec | TableSectionSpec | FigureSectionSpec],
    visual_qc_rows: Sequence[Mapping[str, Any] | Any],
    artifact_rows: Sequence[Mapping[str, Any] | Any],
    warnings: Sequence[str],
    provenance: Mapping[str, Any],
) -> str:
    body: list[str] = []
    if spec.title:
        body.append(f"<h1>{_html(spec.title)}</h1>")
    if spec.subtitle:
        body.append(f"<p class=\"subtitle\">{_html(spec.subtitle)}</p>")
    for section in sections:
        body.append(_html_section(section, rows))
    body.extend(_html_common_sections(spec, visual_qc_rows, artifact_rows, warnings, provenance))
    if spec.caption:
        body.append(f"<p>{_html(spec.caption)}</p>")
    if spec.footnote:
        body.append(f"<p class=\"footnote\">{_html(spec.footnote)}</p>")
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{_html(spec.title or spec.report_id)}</title>\n"
        "<style>"
        "body{font-family:Arial,Helvetica,sans-serif;line-height:1.45;margin:32px;color:#222;}"
        "table{border-collapse:collapse;width:100%;margin:12px 0;}"
        "th,td{border:1px solid #d0d0d0;padding:6px 8px;text-align:left;vertical-align:top;}"
        "th{background:#f1f3f4;}"
        ".subtitle,.footnote,figcaption{color:#555;}"
        "img{max-width:100%;height:auto;}"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )


def _text_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: ReportSpec,
    sections: Sequence[ReportSectionSpec | TableSectionSpec | FigureSectionSpec],
    visual_qc_rows: Sequence[Mapping[str, Any] | Any],
    artifact_rows: Sequence[Mapping[str, Any] | Any],
    warnings: Sequence[str],
    provenance: Mapping[str, Any],
) -> str:
    lines: list[str] = []
    if spec.title:
        lines.extend((spec.title, "=" * len(spec.title), ""))
    if spec.subtitle:
        lines.extend((spec.subtitle, ""))
    for section in sections:
        lines.extend(_text_section(section, rows))
    lines.extend(_text_common_sections(spec, visual_qc_rows, artifact_rows, warnings, provenance))
    if spec.caption:
        lines.extend((spec.caption, ""))
    if spec.footnote:
        lines.extend((spec.footnote, ""))
    return "\n".join(lines).rstrip() + "\n"


def _markdown_section(section: ReportSectionSpec | TableSectionSpec | FigureSectionSpec, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    heading = getattr(section, "heading", None)
    if heading:
        lines.extend((f"{'#' * section.level} {heading}", ""))
    if isinstance(section, ReportSectionSpec):
        if section.text:
            lines.extend((section.text, ""))
    elif isinstance(section, TableSectionSpec):
        table_rows = _section_rows(section, rows)
        lines.append(_markdown_table(table_rows, _section_columns(section, table_rows)).rstrip())
        lines.append("")
        if section.caption:
            lines.extend((section.caption, ""))
    elif isinstance(section, FigureSectionSpec):
        alt = section.alt_text or section.heading or section.plot_id or "figure"
        path = _display_scalar(section.figure_path or "")
        if path:
            lines.extend((f"![{_markdown_inline(alt)}]({path})", ""))
        if section.caption:
            lines.extend((section.caption, ""))
    return lines


def _html_section(section: ReportSectionSpec | TableSectionSpec | FigureSectionSpec, rows: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    heading = getattr(section, "heading", None)
    if heading:
        parts.append(f"<h{section.level}>{_html(heading)}</h{section.level}>")
    if isinstance(section, ReportSectionSpec):
        if section.text:
            parts.append(f"<p>{_html(section.text)}</p>")
    elif isinstance(section, TableSectionSpec):
        table_rows = _section_rows(section, rows)
        parts.append(_html_table(table_rows, _section_columns(section, table_rows)))
        if section.caption:
            parts.append(f"<p>{_html(section.caption)}</p>")
    elif isinstance(section, FigureSectionSpec):
        path = _display_scalar(section.figure_path or "")
        if path:
            alt = section.alt_text or section.heading or section.plot_id or "figure"
            figure = [f"<figure><img src=\"{_html_attr(path)}\" alt=\"{_html_attr(alt)}\">"]
            if section.caption:
                figure.append(f"<figcaption>{_html(section.caption)}</figcaption>")
            figure.append("</figure>")
            parts.append("".join(figure))
    return "\n".join(parts)


def _text_section(section: ReportSectionSpec | TableSectionSpec | FigureSectionSpec, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    heading = getattr(section, "heading", None)
    if heading:
        lines.extend((heading, "-" * len(heading), ""))
    if isinstance(section, ReportSectionSpec):
        if section.text:
            lines.extend((section.text, ""))
    elif isinstance(section, TableSectionSpec):
        table_rows = _section_rows(section, rows)
        lines.append(_text_table(table_rows, _section_columns(section, table_rows)).rstrip())
        lines.append("")
        if section.caption:
            lines.extend((section.caption, ""))
    elif isinstance(section, FigureSectionSpec):
        if section.figure_path:
            lines.extend((f"Figure: {section.figure_path}", ""))
        if section.caption:
            lines.extend((section.caption, ""))
    return lines


def _markdown_common_sections(
    spec: ReportSpec,
    visual_qc_rows: Sequence[Mapping[str, Any] | Any],
    artifact_rows: Sequence[Mapping[str, Any] | Any],
    warnings: Sequence[str],
    provenance: Mapping[str, Any],
) -> list[str]:
    lines: list[str] = []
    if spec.methods_note:
        lines.extend((f"## Methods", "", spec.methods_note, ""))
    if spec.include_visual_qc and visual_qc_rows:
        rows = [_row_from_any(row) for row in visual_qc_rows]
        lines.extend(("## Visual QC", "", _markdown_table(rows, ("artifact_id", "check_id", "severity", "message")).rstrip(), ""))
    if spec.include_artifact_list and artifact_rows:
        rows = [_row_from_any(row) for row in artifact_rows]
        lines.extend(("## Generated Artifacts", "", _markdown_table(rows, ("role", "name", "path", "file_format", "written")).rstrip(), ""))
    if warnings:
        lines.extend(("## Warnings", ""))
        lines.extend(f"- {_markdown_inline(warning)}" for warning in warnings)
        lines.append("")
    if provenance:
        lines.extend(("## Provenance", "", _markdown_table([provenance], _columns_for_rows([provenance])).rstrip(), ""))
    return lines


def _html_common_sections(
    spec: ReportSpec,
    visual_qc_rows: Sequence[Mapping[str, Any] | Any],
    artifact_rows: Sequence[Mapping[str, Any] | Any],
    warnings: Sequence[str],
    provenance: Mapping[str, Any],
) -> list[str]:
    parts: list[str] = []
    if spec.methods_note:
        parts.append(f"<h2>Methods</h2><p>{_html(spec.methods_note)}</p>")
    if spec.include_visual_qc and visual_qc_rows:
        rows = [_row_from_any(row) for row in visual_qc_rows]
        parts.append("<h2>Visual QC</h2>" + _html_table(rows, ("artifact_id", "check_id", "severity", "message")))
    if spec.include_artifact_list and artifact_rows:
        rows = [_row_from_any(row) for row in artifact_rows]
        parts.append("<h2>Generated Artifacts</h2>" + _html_table(rows, ("role", "name", "path", "file_format", "written")))
    if warnings:
        parts.append("<h2>Warnings</h2><ul>" + "".join(f"<li>{_html(warning)}</li>" for warning in warnings) + "</ul>")
    if provenance:
        parts.append("<h2>Provenance</h2>" + _html_table([provenance], _columns_for_rows([provenance])))
    return parts


def _text_common_sections(
    spec: ReportSpec,
    visual_qc_rows: Sequence[Mapping[str, Any] | Any],
    artifact_rows: Sequence[Mapping[str, Any] | Any],
    warnings: Sequence[str],
    provenance: Mapping[str, Any],
) -> list[str]:
    lines: list[str] = []
    if spec.methods_note:
        lines.extend(("Methods", "-------", spec.methods_note, ""))
    if spec.include_visual_qc and visual_qc_rows:
        rows = [_row_from_any(row) for row in visual_qc_rows]
        lines.extend(("Visual QC", "---------", _text_table(rows, ("artifact_id", "check_id", "severity", "message")).rstrip(), ""))
    if spec.include_artifact_list and artifact_rows:
        rows = [_row_from_any(row) for row in artifact_rows]
        lines.extend(("Generated Artifacts", "-------------------", _text_table(rows, ("role", "name", "path", "file_format", "written")).rstrip(), ""))
    if warnings:
        lines.extend(("Warnings", "--------"))
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    if provenance:
        lines.extend(("Provenance", "----------", _text_table([provenance], _columns_for_rows([provenance])).rstrip(), ""))
    return lines


def _default_sections(
    spec: ReportSpec,
    rows: Sequence[Mapping[str, Any]],
    figure_paths: Mapping[str, str],
) -> tuple[ReportSectionSpec | TableSectionSpec | FigureSectionSpec, ...]:
    if spec.sections:
        sections: list[ReportSectionSpec | TableSectionSpec | FigureSectionSpec] = []
        for section in spec.sections:
            if isinstance(section, FigureSectionSpec) and section.figure_path is None and section.plot_id:
                sections.append(replace(section, figure_path=figure_paths.get(section.plot_id)))
            else:
                sections.append(section)
        return tuple(sections)
    if rows:
        return (TableSectionSpec(section_id="rows", heading="Rows"),)
    return ()


def _section_rows(section: TableSectionSpec, rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    section_rows = tuple(section.rows) if section.rows else tuple(rows)
    if section.max_rows is not None:
        section_rows = section_rows[: section.max_rows]
    return section_rows


def _section_columns(section: TableSectionSpec, rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(section.columns) if section.columns else _columns_for_rows(rows)


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not columns:
        return "\n"
    lines = [
        "| " + " | ".join(_markdown_inline(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_inline(_display_scalar(row.get(column))) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _html_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not columns:
        return "<table></table>"
    header = "".join(f"<th>{_html(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_html(_display_scalar(row.get(column)))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _text_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not columns:
        return "\n"
    lines = ["\t".join(columns)]
    lines.extend("\t".join(_display_scalar(row.get(column)) for column in columns) for row in rows)
    return "\n".join(lines) + "\n"


def _expanded_report_spec(spec: ReportSpec, metadata: Mapping[str, Any]) -> ReportSpec:
    def expand(value: str | None) -> str | None:
        if value is None:
            return None
        expanded, _ = _expand_template(value, metadata)
        return expanded

    sections: list[ReportSectionSpec | TableSectionSpec | FigureSectionSpec] = []
    for section in spec.sections:
        section_metadata = _merged_metadata(metadata, getattr(section, "metadata", {}))
        section_expand = lambda value, section_metadata=section_metadata: _expand_template(value, section_metadata)[0] if value is not None else None
        if isinstance(section, ReportSectionSpec):
            sections.append(replace(section, heading=section_expand(section.heading), text=section_expand(section.text)))
        elif isinstance(section, TableSectionSpec):
            sections.append(replace(section, heading=section_expand(section.heading), caption=section_expand(section.caption)))
        elif isinstance(section, FigureSectionSpec):
            sections.append(
                replace(
                    section,
                    heading=section_expand(section.heading),
                    caption=section_expand(section.caption),
                    alt_text=section_expand(section.alt_text),
                )
            )
    return replace(
        spec,
        title=expand(spec.title),
        subtitle=expand(spec.subtitle),
        caption=expand(spec.caption),
        footnote=expand(spec.footnote),
        alt_text=expand(spec.alt_text),
        methods_note=expand(spec.methods_note),
        sections=tuple(sections),
    )


def _coerce_section(section: ReportSectionSpec | TableSectionSpec | FigureSectionSpec | Mapping[str, Any]) -> ReportSectionSpec | TableSectionSpec | FigureSectionSpec:
    if isinstance(section, ReportSectionSpec | TableSectionSpec | FigureSectionSpec):
        return section
    mapping = _as_mapping(section, field_name="section")
    section_type = str(mapping.get("section_type", mapping.get("type", "text"))).lower()
    if section_type == "table":
        return TableSectionSpec(
            section_id=str(mapping.get("section_id", "table")),
            heading=mapping.get("heading"),
            rows=tuple(mapping.get("rows", ())),
            columns=tuple(mapping.get("columns", ())),
            required_columns=tuple(mapping.get("required_columns", ())),
            caption=mapping.get("caption"),
            max_rows=mapping.get("max_rows"),
            level=int(mapping.get("level", 2)),
            metadata=mapping.get("metadata", {}),
        )
    if section_type == "figure":
        return FigureSectionSpec(
            section_id=str(mapping.get("section_id", "figure")),
            heading=mapping.get("heading"),
            figure_path=mapping.get("figure_path"),
            plot_id=mapping.get("plot_id"),
            caption=mapping.get("caption"),
            alt_text=mapping.get("alt_text"),
            level=int(mapping.get("level", 2)),
            metadata=mapping.get("metadata", {}),
        )
    return ReportSectionSpec(
        section_id=str(mapping.get("section_id", "section")),
        heading=mapping.get("heading"),
        text=mapping.get("text"),
        level=int(mapping.get("level", 2)),
        metadata=mapping.get("metadata", {}),
    )


def _expand_template(template: str, metadata: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    formatter = Formatter()
    errors: list[str] = []
    parts: list[str] = []
    for literal, field_name_raw, format_spec, conversion in formatter.parse(template):
        parts.append(literal)
        if field_name_raw is None:
            continue
        root_name = field_name_raw
        if "." in root_name or "[" in root_name or "]" in root_name:
            errors.append(f"Template field {root_name!r} is not a simple metadata key.")
        elif root_name not in metadata:
            errors.append(f"Template field {root_name!r} is missing from metadata.")
        elif "{" in format_spec or "}" in format_spec:
            errors.append(f"Template format spec for {root_name!r} must not contain nested fields.")
        else:
            value = metadata[root_name]
            if conversion == "r":
                text = repr(value)
            elif conversion == "s":
                text = str(value)
            else:
                text = _display_scalar(value)
            if format_spec:
                try:
                    text = format(value, format_spec)
                except (TypeError, ValueError) as exc:
                    errors.append(f"Template format spec for {root_name!r} failed: {exc}")
                    text = _display_scalar(value)
            parts.append(text)
            continue
        parts.append("")
    return "".join(parts), tuple(errors)


def _row_from_any(row: Mapping[str, Any] | Any) -> dict[str, Any]:
    return _json_safe_mapping(_as_mapping(row, field_name="row"))


def _normalize_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        normalized.append(_json_safe_mapping(_as_mapping(row, field_name=f"rows[{index}]")))
    return tuple(normalized)


def _columns_for_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    columns: list[str] = []
    for row in rows:
        for column in row:
            text_column = str(column)
            if text_column not in columns:
                columns.append(text_column)
    return tuple(columns)


def _bounded_heading_level(value: int) -> int:
    level = int(value)
    if level < 1:
        return 1
    if level > 6:
        return 6
    return level


def _merged_metadata(*mappings: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        if mapping:
            merged.update(_json_safe_mapping(mapping))
    return merged


def _html(value: Any) -> str:
    return html.escape(_display_scalar(value), quote=False)


def _html_attr(value: Any) -> str:
    return html.escape(_display_scalar(value), quote=True)


def _markdown_inline(value: Any) -> str:
    return _display_scalar(value).replace("|", "\\|")


def _display_scalar(value: Any) -> str:
    safe_value = _json_safe(value)
    if safe_value is None:
        return ""
    if isinstance(safe_value, bool):
        return "true" if safe_value else "false"
    if isinstance(safe_value, int | float):
        return repr(safe_value)
    if isinstance(safe_value, str):
        return safe_value.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return json.dumps(safe_value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif is_dataclass(value):
        value = {field_.name: getattr(value, field_.name) for field_ in fields(value)}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or expose to_dict().")
    return value


def _json_safe_dataclass(instance: object) -> dict[str, Any]:
    if not is_dataclass(instance):
        raise TypeError("_json_safe_dataclass requires a dataclass instance.")
    return {field_.name: _json_safe(getattr(instance, field_.name)) for field_ in fields(instance)}


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(item) for key, item in value.items()}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe_dataclass(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, int | str | bool) or value is None:
        return value
    return str(value)


def _non_empty_text(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


__all__ = [
    "FigureSectionSpec",
    "ReportSectionSpec",
    "ReportSpec",
    "TableSectionSpec",
    "build_report_document",
    "build_table_section",
]
