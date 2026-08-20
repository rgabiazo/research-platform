
"""Dependency-free plot specifications and lightweight renderers.

The helpers in this module consume already-computed rectangular rows. They do
not calculate statistics, infer labels from project conventions, or import
plotting libraries at module import time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
import html
import json
import math
from pathlib import Path
from string import Formatter
from typing import Any


DEFAULT_COLOR_PALETTE = (
    "#2f6f9f",
    "#c44536",
    "#4f7d3f",
    "#8a5a9c",
    "#8c6d31",
    "#2c7a68",
)


@dataclass(frozen=True)
class FigureTextSpec:
    """Configurable text attached to a figure."""

    title: str | None = None
    subtitle: str | None = None
    footnote: str | None = None
    alt_text: str | None = None
    methods_note: str | None = None
    panel_labels: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_labels", {str(key): str(value) for key, value in self.panel_labels.items()})
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class AxisTextSpec:
    """Axis labels and deterministic tick-label controls."""

    x_label: str | None = None
    y_label: str | None = None
    x_tick_label_rotation: int = 0
    y_tick_label_rotation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class LegendSpec:
    """Legend title and optional caller-provided label mapping."""

    title: str | None = None
    label_mapping: Mapping[str, str] = field(default_factory=dict)
    show: bool = True
    required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "label_mapping", {str(key): str(value) for key, value in self.label_mapping.items()})

    def label_for(self, value: Any) -> str:
        key = _display_scalar(value)
        return self.label_mapping.get(key, key)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class CaptionSpec:
    """Figure caption and footnote text."""

    text: str | None = None
    footnote: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class LayoutSpec:
    """Deterministic layout policy for dependency-free figure renderers."""

    width: int = 720
    height: int = 480
    margin_top: int = 76
    margin_right: int = 156
    margin_bottom: int = 78
    margin_left: int = 148
    base_font_size: int = 12
    title_font_size: int = 18
    subtitle_font_size: int = 13
    axis_font_size: int = 12
    tick_font_size: int = 10
    legend_font_size: int = 11
    caption_font_size: int = 10
    data_label_font_size: int = 10
    point_radius: float = 4.0
    interval_stroke_width: float = 2.0
    color_palette: Sequence[str] = DEFAULT_COLOR_PALETTE

    def __post_init__(self) -> None:
        positive_fields = (
            "width",
            "height",
            "base_font_size",
            "title_font_size",
            "axis_font_size",
            "tick_font_size",
            "legend_font_size",
            "caption_font_size",
            "data_label_font_size",
        )
        for field_name in positive_fields:
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        margins = self.margin_left + self.margin_right
        if margins >= self.width:
            raise ValueError("Horizontal margins must leave a positive plotting area.")
        vertical_margins = self.margin_top + self.margin_bottom
        if vertical_margins >= self.height:
            raise ValueError("Vertical margins must leave a positive plotting area.")
        object.__setattr__(self, "color_palette", tuple(str(color) for color in self.color_palette) or DEFAULT_COLOR_PALETTE)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class VisualQcSpec:
    """Heuristic visual QC thresholds for publication-ready figure layout."""

    min_width: int = 480
    min_height: int = 320
    min_font_size: int = 9
    max_title_chars: int = 90
    max_subtitle_chars: int = 120
    max_axis_label_chars: int = 64
    max_caption_chars: int = 180
    max_tick_label_chars: int = 24
    max_tick_count: int = 14
    max_categories: int = 32
    max_data_labels: int = 24
    min_row_gap_px: float = 14.0
    require_title: bool = False
    require_axis_labels: bool = False
    require_legend_labels: bool = False
    require_caption: bool = False
    require_alt_text: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class VisualQcRow:
    """One deterministic visual layout QC finding."""

    artifact_id: str
    check_id: str
    severity: str
    status: str
    scope: str
    message: str
    metric: str | None = None
    value: Any = None
    threshold: Any = None
    row_index: int | None = None
    column: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _non_empty_text(self.artifact_id, field_name="artifact_id"))
        object.__setattr__(self, "check_id", _non_empty_text(self.check_id, field_name="check_id"))
        severity = str(self.severity or "warning").lower()
        if severity not in {"info", "warning", "error"}:
            raise ValueError("VisualQcRow.severity must be one of: info, warning, error.")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "status", str(self.status or severity).lower())
        object.__setattr__(self, "scope", str(self.scope or "figure"))
        object.__setattr__(self, "message", _non_empty_text(self.message, field_name="message"))
        object.__setattr__(self, "value", _json_safe(self.value))
        object.__setattr__(self, "threshold", _json_safe(self.threshold))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PlotSpec:
    """Base reusable plot metadata."""

    plot_id: str = "figure"
    plot_type: str = "point_interval"
    text: FigureTextSpec = field(default_factory=FigureTextSpec)
    axes: AxisTextSpec = field(default_factory=AxisTextSpec)
    legend: LegendSpec = field(default_factory=LegendSpec)
    caption: CaptionSpec = field(default_factory=CaptionSpec)
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    visual_qc: VisualQcSpec = field(default_factory=VisualQcSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    required_columns: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plot_id", _non_empty_text(self.plot_id, field_name="plot_id"))
        object.__setattr__(self, "plot_type", _non_empty_text(self.plot_type, field_name="plot_type"))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))
        object.__setattr__(self, "required_columns", tuple(str(column) for column in self.required_columns))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


@dataclass(frozen=True)
class PointIntervalPlotSpec:
    """Generic point/interval plot from already-computed row values."""

    plot_id: str = "point_interval"
    label_column: str = "label"
    estimate_column: str = "estimate"
    lower_column: str | None = None
    upper_column: str | None = None
    group_column: str | None = None
    data_label_column: str | None = None
    sort_by: Sequence[str] = ()
    text: FigureTextSpec = field(default_factory=FigureTextSpec)
    axes: AxisTextSpec = field(default_factory=AxisTextSpec)
    legend: LegendSpec = field(default_factory=LegendSpec)
    caption: CaptionSpec = field(default_factory=CaptionSpec)
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    visual_qc: VisualQcSpec = field(default_factory=VisualQcSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    required_columns: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plot_id", _non_empty_text(self.plot_id, field_name="plot_id"))
        object.__setattr__(self, "label_column", _non_empty_text(self.label_column, field_name="label_column"))
        object.__setattr__(self, "estimate_column", _non_empty_text(self.estimate_column, field_name="estimate_column"))
        for field_name in ("lower_column", "upper_column", "group_column", "data_label_column"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _non_empty_text(value, field_name=field_name))
        object.__setattr__(self, "sort_by", tuple(str(column) for column in self.sort_by))
        object.__setattr__(self, "metadata", _json_safe_mapping(self.metadata))
        object.__setattr__(self, "required_columns", tuple(str(column) for column in self.required_columns))

    def source_columns(self) -> tuple[str, ...]:
        columns = [self.label_column, self.estimate_column]
        for column in (
            self.lower_column,
            self.upper_column,
            self.group_column,
            self.data_label_column,
            *self.sort_by,
            *self.required_columns,
        ):
            if column:
                columns.append(str(column))
        return tuple(dict.fromkeys(columns))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dataclass(self)


def build_point_interval_plot_spec(
    *,
    label_column: str,
    estimate_column: str,
    lower_column: str | None = None,
    upper_column: str | None = None,
    group_column: str | None = None,
    data_label_column: str | None = None,
    plot_id: str = "point_interval",
    sort_by: Sequence[str] = (),
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    footnote: str | None = None,
    alt_text: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    legend_title: str | None = None,
    legend_label_mapping: Mapping[str, str] | None = None,
    layout: LayoutSpec | None = None,
    visual_qc: VisualQcSpec | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PointIntervalPlotSpec:
    """Build a generic point/interval plot spec without domain assumptions."""

    return PointIntervalPlotSpec(
        plot_id=plot_id,
        label_column=label_column,
        estimate_column=estimate_column,
        lower_column=lower_column,
        upper_column=upper_column,
        group_column=group_column,
        data_label_column=data_label_column,
        sort_by=sort_by,
        text=FigureTextSpec(title=title, subtitle=subtitle, footnote=footnote, alt_text=alt_text),
        axes=AxisTextSpec(x_label=x_label, y_label=y_label),
        legend=LegendSpec(title=legend_title, label_mapping=legend_label_mapping or {}),
        caption=CaptionSpec(text=caption, footnote=footnote),
        layout=layout or LayoutSpec(),
        visual_qc=visual_qc or VisualQcSpec(),
        metadata=metadata or {},
    )


def build_visual_layout_qc_rows(
    rows: Iterable[Mapping[str, Any]],
    plot_spec: PointIntervalPlotSpec,
    *,
    metadata: Mapping[str, Any] | None = None,
    visual_qc_spec: VisualQcSpec | None = None,
) -> tuple[VisualQcRow, ...]:
    """Return deterministic visual QC rows for a point/interval figure."""

    normalized_rows = _normalize_rows(rows)
    merged_metadata = _merged_metadata(metadata, plot_spec.metadata, plot_spec.text.metadata)
    spec, template_qc_rows = _expanded_point_interval_spec(plot_spec, merged_metadata)
    qc = visual_qc_spec or spec.visual_qc
    qc_rows: list[VisualQcRow] = list(template_qc_rows)
    layout = spec.layout
    artifact_id = spec.plot_id

    if layout.width < qc.min_width:
        qc_rows.append(
            _qc_row(
                artifact_id,
                "figure_width_too_small",
                f"Figure width {layout.width}px is below the minimum policy of {qc.min_width}px.",
                metric="width_px",
                value=layout.width,
                threshold=qc.min_width,
            )
        )
    if layout.height < qc.min_height:
        qc_rows.append(
            _qc_row(
                artifact_id,
                "figure_height_too_small",
                f"Figure height {layout.height}px is below the minimum policy of {qc.min_height}px.",
                metric="height_px",
                value=layout.height,
                threshold=qc.min_height,
            )
        )
    for font_field in (
        "base_font_size",
        "title_font_size",
        "subtitle_font_size",
        "axis_font_size",
        "tick_font_size",
        "legend_font_size",
        "caption_font_size",
        "data_label_font_size",
    ):
        value = int(getattr(layout, font_field))
        if value < qc.min_font_size:
            qc_rows.append(
                _qc_row(
                    artifact_id,
                    "font_size_below_minimum",
                    f"{font_field}={value}px is below the minimum policy of {qc.min_font_size}px.",
                    metric=font_field,
                    value=value,
                    threshold=qc.min_font_size,
                )
            )

    _append_required_text_qc(qc_rows, spec, qc)
    _append_text_clip_qc(qc_rows, spec, qc)
    _append_tick_density_qc(qc_rows, normalized_rows, spec, qc)
    _append_value_qc(qc_rows, normalized_rows, spec)
    _append_legend_qc(qc_rows, normalized_rows, spec, qc)
    _append_data_label_qc(qc_rows, normalized_rows, spec, qc)
    return tuple(qc_rows)


def render_point_interval_svg(
    rows: Iterable[Mapping[str, Any]],
    plot_spec: PointIntervalPlotSpec,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Render a dependency-free SVG point/interval figure.

    Rendering uses provided estimates and interval bounds as-is. It does not
    compute p values, confidence intervals, bootstrap estimates, or any other
    analysis statistics.
    """

    normalized_rows = _sort_rows(_normalize_rows(rows), plot_spec.sort_by)
    spec, _ = _expanded_point_interval_spec(
        plot_spec,
        _merged_metadata(metadata, plot_spec.metadata, plot_spec.text.metadata),
    )
    points = _point_rows(normalized_rows, spec)
    layout = spec.layout
    width = layout.width
    height = layout.height
    plot_left = layout.margin_left
    plot_right = width - layout.margin_right
    plot_top = layout.margin_top
    plot_bottom = height - layout.margin_bottom
    plot_width = max(1, plot_right - plot_left)
    plot_height = max(1, plot_bottom - plot_top)
    values = [value for point in points for value in (point["estimate"], point["low"], point["high"])]
    if values:
        minimum = min(values)
        maximum = max(values)
    else:
        minimum = -1.0
        maximum = 1.0
    if math.isclose(minimum, maximum):
        minimum -= 1.0
        maximum += 1.0
    pad = (maximum - minimum) * 0.08
    minimum -= pad
    maximum += pad

    def x_scale(value: float) -> float:
        return plot_left + ((value - minimum) / (maximum - minimum)) * plot_width

    row_gap = plot_height / max(1, len(points))
    color_for_group = _color_lookup(points, spec)
    title_y = 28
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        "<style>",
        ".rp-axis{stroke:#333;stroke-width:1}",
        ".rp-grid{stroke:#d8d8d8;stroke-width:1}",
        ".rp-interval{stroke-width:2;stroke-linecap:round}",
        ".rp-point{stroke:#ffffff;stroke-width:1.5}",
        ".rp-text{font-family:Arial, Helvetica, sans-serif;fill:#222}",
        "</style>",
        "</defs>",
    ]
    if spec.text.alt_text:
        lines.append(f"<title>{_xml(spec.text.alt_text)}</title>")
    if spec.text.title:
        lines.append(
            f'<text class="rp-text" x="{plot_left}" y="{title_y}" font-size="{layout.title_font_size}" '
            f'font-weight="700">{_xml(spec.text.title)}</text>'
        )
    if spec.text.subtitle:
        lines.append(
            f'<text class="rp-text" x="{plot_left}" y="{title_y + 22}" font-size="{layout.subtitle_font_size}">'
            f"{_xml(spec.text.subtitle)}</text>"
        )

    lines.extend(_x_grid_lines(minimum, maximum, x_scale, plot_top, plot_bottom, layout))
    if minimum < 0 < maximum:
        zero_x = x_scale(0.0)
        lines.append(f'<line class="rp-axis" x1="{zero_x:.2f}" y1="{plot_top}" x2="{zero_x:.2f}" y2="{plot_bottom}"/>')
    lines.append(f'<line class="rp-axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>')
    lines.append(f'<line class="rp-axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>')

    for index, point in enumerate(points):
        y = plot_top + row_gap * (index + 0.5)
        color = color_for_group[point["group"]]
        low_x = x_scale(point["low"])
        high_x = x_scale(point["high"])
        estimate_x = x_scale(point["estimate"])
        lines.append(
            f'<text class="rp-text" x="{plot_left - 8}" y="{y + 4:.2f}" font-size="{layout.tick_font_size}" '
            f'text-anchor="end">{_xml(point["label"])}</text>'
        )
        lines.append(
            f'<line class="rp-interval" x1="{low_x:.2f}" y1="{y:.2f}" x2="{high_x:.2f}" y2="{y:.2f}" '
            f'stroke="{_xml(color)}"/>'
        )
        lines.append(
            f'<circle class="rp-point" cx="{estimate_x:.2f}" cy="{y:.2f}" r="{layout.point_radius}" '
            f'fill="{_xml(color)}"/>'
        )
        if spec.data_label_column and point["data_label"]:
            label_x = min(plot_right + 8, estimate_x + 8)
            lines.append(
                f'<text class="rp-text" x="{label_x:.2f}" y="{y + 4:.2f}" font-size="{layout.data_label_font_size}">'
                f'{_xml(point["data_label"])}</text>'
            )

    if spec.axes.x_label:
        lines.append(
            f'<text class="rp-text" x="{(plot_left + plot_right) / 2:.2f}" y="{height - 30}" '
            f'font-size="{layout.axis_font_size}" text-anchor="middle">{_xml(spec.axes.x_label)}</text>'
        )
    if spec.axes.y_label:
        y_mid = (plot_top + plot_bottom) / 2
        lines.append(
            f'<text class="rp-text" transform="translate(22 {y_mid:.2f}) rotate(-90)" '
            f'font-size="{layout.axis_font_size}" text-anchor="middle">{_xml(spec.axes.y_label)}</text>'
        )

    lines.extend(_legend_lines(points, spec, color_for_group, x=plot_right + 24, y=plot_top))
    caption_y = height - 12
    if spec.caption.text:
        lines.append(
            f'<text class="rp-text" x="{plot_left}" y="{caption_y}" font-size="{layout.caption_font_size}">'
            f"{_xml(spec.caption.text)}</text>"
        )
        caption_y -= layout.caption_font_size + 4
    if spec.caption.footnote or spec.text.footnote:
        footnote = spec.caption.footnote or spec.text.footnote
        lines.append(
            f'<text class="rp-text" x="{plot_left}" y="{caption_y}" font-size="{layout.caption_font_size}">'
            f"{_xml(footnote)}</text>"
        )
    if not points:
        lines.append(
            f'<text class="rp-text" x="{(plot_left + plot_right) / 2:.2f}" y="{(plot_top + plot_bottom) / 2:.2f}" '
            f'font-size="{layout.base_font_size}" text-anchor="middle">No finite point values</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _expanded_point_interval_spec(
    spec: PointIntervalPlotSpec,
    metadata: Mapping[str, Any],
) -> tuple[PointIntervalPlotSpec, tuple[VisualQcRow, ...]]:
    qc_rows: list[VisualQcRow] = []

    def expand(value: str | None, field_name: str) -> str | None:
        if value is None:
            return None
        expanded, errors = _expand_template(value, metadata, field_name=field_name)
        for error in errors:
            qc_rows.append(
                _qc_row(
                    spec.plot_id,
                    "missing_template_field",
                    error,
                    severity="error",
                    scope="text",
                    metric=field_name,
                )
            )
        return expanded

    text = replace(
        spec.text,
        title=expand(spec.text.title, "title"),
        subtitle=expand(spec.text.subtitle, "subtitle"),
        footnote=expand(spec.text.footnote, "footnote"),
        alt_text=expand(spec.text.alt_text, "alt_text"),
        methods_note=expand(spec.text.methods_note, "methods_note"),
        panel_labels={key: expand(value, f"panel_labels.{key}") or "" for key, value in spec.text.panel_labels.items()},
    )
    axes = replace(
        spec.axes,
        x_label=expand(spec.axes.x_label, "x_label"),
        y_label=expand(spec.axes.y_label, "y_label"),
    )
    legend = replace(
        spec.legend,
        title=expand(spec.legend.title, "legend_title"),
        label_mapping={key: expand(value, f"legend_label_mapping.{key}") or "" for key, value in spec.legend.label_mapping.items()},
    )
    caption = replace(
        spec.caption,
        text=expand(spec.caption.text, "caption"),
        footnote=expand(spec.caption.footnote, "caption_footnote"),
    )
    return replace(spec, text=text, axes=axes, legend=legend, caption=caption), tuple(qc_rows)


def _append_required_text_qc(
    qc_rows: list[VisualQcRow],
    spec: PointIntervalPlotSpec,
    qc: VisualQcSpec,
) -> None:
    if qc.require_title and not _has_text(spec.text.title):
        qc_rows.append(_missing_text_row(spec.plot_id, "missing_title", "Figure title is required by visual QC policy."))
    if qc.require_axis_labels:
        if not _has_text(spec.axes.x_label):
            qc_rows.append(_missing_text_row(spec.plot_id, "missing_x_label", "X-axis label is required by visual QC policy."))
        if not _has_text(spec.axes.y_label):
            qc_rows.append(_missing_text_row(spec.plot_id, "missing_y_label", "Y-axis label is required by visual QC policy."))
    if qc.require_caption and not _has_text(spec.caption.text):
        qc_rows.append(_missing_text_row(spec.plot_id, "missing_caption", "Figure caption is required by visual QC policy."))
    if qc.require_alt_text and not _has_text(spec.text.alt_text):
        qc_rows.append(_missing_text_row(spec.plot_id, "missing_alt_text", "Figure alt text is required by visual QC policy."))


def _append_text_clip_qc(
    qc_rows: list[VisualQcRow],
    spec: PointIntervalPlotSpec,
    qc: VisualQcSpec,
) -> None:
    layout = spec.layout
    available_width = layout.width - layout.margin_left - 20
    text_checks = (
        ("title_clip_risk", "title", spec.text.title, layout.title_font_size, available_width, qc.max_title_chars),
        ("subtitle_clip_risk", "subtitle", spec.text.subtitle, layout.subtitle_font_size, available_width, qc.max_subtitle_chars),
        ("x_axis_label_clip_risk", "x_label", spec.axes.x_label, layout.axis_font_size, available_width, qc.max_axis_label_chars),
        ("y_axis_label_clip_risk", "y_label", spec.axes.y_label, layout.axis_font_size, layout.height - layout.margin_top - layout.margin_bottom, qc.max_axis_label_chars),
        ("caption_clip_risk", "caption", spec.caption.text, layout.caption_font_size, available_width, qc.max_caption_chars),
        ("legend_title_clip_risk", "legend_title", spec.legend.title, layout.legend_font_size, layout.margin_right - 18, qc.max_axis_label_chars),
    )
    for check_id, field_name, text, font_size, available, max_chars in text_checks:
        if not _has_text(text):
            continue
        text_value = str(text)
        width_estimate = _text_width_estimate(text_value, font_size)
        if len(text_value) > max_chars or width_estimate > available:
            qc_rows.append(
                _qc_row(
                    spec.plot_id,
                    check_id,
                    f"{field_name} text may clip or need wrapping.",
                    metric=f"{field_name}_chars",
                    value=len(text_value),
                    threshold=max_chars,
                    scope="text",
                )
            )


def _append_tick_density_qc(
    qc_rows: list[VisualQcRow],
    rows: Sequence[Mapping[str, Any]],
    spec: PointIntervalPlotSpec,
    qc: VisualQcSpec,
) -> None:
    labels = [_display_scalar(row.get(spec.label_column)) for row in rows if _display_scalar(row.get(spec.label_column))]
    unique_labels = tuple(dict.fromkeys(labels))
    if len(labels) > qc.max_tick_count:
        qc_rows.append(
            _qc_row(
                spec.plot_id,
                "dense_tick_labels",
                "Categorical tick labels may be too dense for the figure height.",
                metric="tick_label_count",
                value=len(labels),
                threshold=qc.max_tick_count,
            )
        )
    if len(unique_labels) > qc.max_categories:
        qc_rows.append(
            _qc_row(
                spec.plot_id,
                "too_many_categorical_labels",
                "The number of categorical labels may be too high for a compact static figure.",
                metric="category_count",
                value=len(unique_labels),
                threshold=qc.max_categories,
            )
        )
    longest = max((len(label) for label in unique_labels), default=0)
    if longest > qc.max_tick_label_chars:
        qc_rows.append(
            _qc_row(
                spec.plot_id,
                "long_tick_labels",
                "Long categorical labels may need rotation, wrapping, or a wider left margin.",
                metric="max_tick_label_chars",
                value=longest,
                threshold=qc.max_tick_label_chars,
            )
        )
    plot_height = spec.layout.height - spec.layout.margin_top - spec.layout.margin_bottom
    if labels and plot_height / len(labels) < qc.min_row_gap_px:
        qc_rows.append(
            _qc_row(
                spec.plot_id,
                "label_collision_risk",
                "Categorical labels may collide vertically.",
                metric="row_gap_px",
                value=round(plot_height / len(labels), 3),
                threshold=qc.min_row_gap_px,
            )
        )


def _append_value_qc(
    qc_rows: list[VisualQcRow],
    rows: Sequence[Mapping[str, Any]],
    spec: PointIntervalPlotSpec,
) -> None:
    value_columns = tuple(column for column in (spec.estimate_column, spec.lower_column, spec.upper_column) if column)
    for row_index, row in enumerate(rows):
        parsed = {column: _maybe_finite_number(row.get(column)) for column in value_columns}
        for column, value in parsed.items():
            if value is None:
                qc_rows.append(
                    _qc_row(
                        spec.plot_id,
                        "non_finite_plot_value",
                        f"Plot value in column {column!r} is missing or non-finite.",
                        severity="warning",
                        metric=column,
                        value=_display_scalar(row.get(column)),
                        row_index=row_index,
                        column=column,
                    )
                )
        estimate = parsed.get(spec.estimate_column)
        low = parsed.get(spec.lower_column or "")
        high = parsed.get(spec.upper_column or "")
        if low is not None and high is not None and low > high:
            qc_rows.append(
                _qc_row(
                    spec.plot_id,
                    "interval_bounds_invalid",
                    "Interval lower bound is greater than the upper bound.",
                    metric="interval_bounds",
                    value={"low": low, "high": high},
                    row_index=row_index,
                )
            )
        if estimate is not None and low is not None and high is not None:
            lower = min(low, high)
            upper = max(low, high)
            if estimate < lower or estimate > upper:
                qc_rows.append(
                    _qc_row(
                        spec.plot_id,
                        "estimate_outside_interval",
                        "Estimate falls outside the provided interval bounds.",
                        metric="estimate_interval_position",
                        value={"estimate": estimate, "low": low, "high": high},
                        row_index=row_index,
                    )
                )


def _append_legend_qc(
    qc_rows: list[VisualQcRow],
    rows: Sequence[Mapping[str, Any]],
    spec: PointIntervalPlotSpec,
    qc: VisualQcSpec,
) -> None:
    if not spec.group_column or not spec.legend.show:
        return
    groups = tuple(dict.fromkeys(_display_scalar(row.get(spec.group_column)) for row in rows))
    groups = tuple(group for group in groups if group)
    if qc.require_legend_labels and not _has_text(spec.legend.title):
        qc_rows.append(_missing_text_row(spec.plot_id, "missing_legend_title", "Legend title is required by visual QC policy."))
    if qc.require_legend_labels:
        missing = [group for group in groups if group not in spec.legend.label_mapping]
        if missing:
            qc_rows.append(
                _missing_text_row(
                    spec.plot_id,
                    "missing_legend_label_mapping",
                    "Legend label mapping is required for all observed groups.",
                    value=missing,
                )
            )
    legend_height = (len(groups) + (1 if spec.legend.title else 0)) * (spec.layout.legend_font_size + 6)
    plot_height = spec.layout.height - spec.layout.margin_top - spec.layout.margin_bottom
    if legend_height > plot_height:
        qc_rows.append(
            _qc_row(
                spec.plot_id,
                "legend_collision_risk",
                "Legend entries may collide with the title, caption, or plotting region.",
                metric="legend_height_px",
                value=legend_height,
                threshold=plot_height,
            )
        )


def _append_data_label_qc(
    qc_rows: list[VisualQcRow],
    rows: Sequence[Mapping[str, Any]],
    spec: PointIntervalPlotSpec,
    qc: VisualQcSpec,
) -> None:
    if not spec.data_label_column:
        return
    labels = [_display_scalar(row.get(spec.data_label_column)) for row in rows if _display_scalar(row.get(spec.data_label_column))]
    if len(labels) > qc.max_data_labels:
        qc_rows.append(
            _qc_row(
                spec.plot_id,
                "data_label_overlap_risk",
                "Requested data labels may overlap in a static figure.",
                metric="data_label_count",
                value=len(labels),
                threshold=qc.max_data_labels,
            )
        )
    longest = max((len(label) for label in labels), default=0)
    plot_width = spec.layout.width - spec.layout.margin_left - spec.layout.margin_right
    if longest and _text_width_estimate("X" * longest, spec.layout.data_label_font_size) > plot_width * 0.3:
        qc_rows.append(
            _qc_row(
                spec.plot_id,
                "data_label_width_risk",
                "Requested data labels may collide with intervals or the legend.",
                metric="max_data_label_chars",
                value=longest,
                threshold=max(1, int((plot_width * 0.3) / (spec.layout.data_label_font_size * 0.58))),
            )
        )


def _point_rows(rows: Sequence[Mapping[str, Any]], spec: PointIntervalPlotSpec) -> tuple[dict[str, Any], ...]:
    points: list[dict[str, Any]] = []
    for row in rows:
        estimate = _maybe_finite_number(row.get(spec.estimate_column))
        if estimate is None:
            continue
        low = _maybe_finite_number(row.get(spec.lower_column)) if spec.lower_column else estimate
        high = _maybe_finite_number(row.get(spec.upper_column)) if spec.upper_column else estimate
        if low is None or high is None:
            continue
        points.append(
            {
                "label": _display_scalar(row.get(spec.label_column)) or "(missing)",
                "estimate": estimate,
                "low": low,
                "high": high,
                "group": _display_scalar(row.get(spec.group_column)) if spec.group_column else "",
                "data_label": _display_scalar(row.get(spec.data_label_column)) if spec.data_label_column else "",
            }
        )
    return tuple(points)


def _x_grid_lines(
    minimum: float,
    maximum: float,
    x_scale: Any,
    plot_top: float,
    plot_bottom: float,
    layout: LayoutSpec,
) -> list[str]:
    lines: list[str] = []
    ticks = _nice_ticks(minimum, maximum, count=5)
    for tick in ticks:
        x = x_scale(tick)
        lines.append(f'<line class="rp-grid" x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}"/>')
        lines.append(
            f'<text class="rp-text" x="{x:.2f}" y="{plot_bottom + 18}" font-size="{layout.tick_font_size}" '
            f'text-anchor="middle">{_xml(_format_tick(tick))}</text>'
        )
    return lines


def _legend_lines(
    points: Sequence[Mapping[str, Any]],
    spec: PointIntervalPlotSpec,
    color_for_group: Mapping[str, str],
    *,
    x: float,
    y: float,
) -> list[str]:
    if not spec.group_column or not spec.legend.show:
        return []
    groups = tuple(color_for_group)
    lines: list[str] = []
    cursor = y
    if spec.legend.title:
        lines.append(
            f'<text class="rp-text" x="{x}" y="{cursor}" font-size="{spec.layout.legend_font_size}" '
            f'font-weight="700">{_xml(spec.legend.title)}</text>'
        )
        cursor += spec.layout.legend_font_size + 8
    for group in groups:
        label = spec.legend.label_for(group)
        color = color_for_group[group]
        lines.append(f'<rect x="{x}" y="{cursor - 9}" width="10" height="10" fill="{_xml(color)}"/>')
        lines.append(
            f'<text class="rp-text" x="{x + 16}" y="{cursor}" font-size="{spec.layout.legend_font_size}">'
            f"{_xml(label)}</text>"
        )
        cursor += spec.layout.legend_font_size + 8
    return lines


def _color_lookup(points: Sequence[Mapping[str, Any]], spec: PointIntervalPlotSpec) -> dict[str, str]:
    groups = tuple(dict.fromkeys(str(point["group"]) for point in points))
    if not groups:
        groups = ("",)
    return {group: spec.layout.color_palette[index % len(spec.layout.color_palette)] for index, group in enumerate(groups)}


def _sort_rows(rows: Sequence[Mapping[str, Any]], sort_by: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    sorted_rows = list(rows)
    for sort_key in reversed(tuple(sort_by)):
        descending = sort_key.startswith("-")
        column = sort_key[1:] if descending else sort_key
        sorted_rows.sort(key=lambda row: _sort_value(row.get(column)), reverse=descending)
    return tuple(sorted_rows)


def _sort_value(value: Any) -> tuple[int, int, Any]:
    number = _maybe_finite_number(value)
    if number is not None:
        return (0, 0, number)
    text = _display_scalar(value)
    if not text:
        return (1, 0, "")
    return (0, 1, text)


def _nice_ticks(minimum: float, maximum: float, *, count: int) -> tuple[float, ...]:
    if count <= 1:
        return (minimum, maximum)
    span = maximum - minimum
    if span <= 0:
        return (minimum,)
    step = span / (count - 1)
    return tuple(minimum + step * index for index in range(count))


def _format_tick(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        value = 0.0
    rendered = f"{value:.3f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _missing_text_row(
    artifact_id: str,
    check_id: str,
    message: str,
    *,
    value: Any = None,
) -> VisualQcRow:
    return _qc_row(
        artifact_id,
        check_id,
        message,
        severity="error",
        scope="text",
        value=value,
    )


def _qc_row(
    artifact_id: str,
    check_id: str,
    message: str,
    *,
    severity: str = "warning",
    scope: str = "figure",
    metric: str | None = None,
    value: Any = None,
    threshold: Any = None,
    row_index: int | None = None,
    column: str | None = None,
) -> VisualQcRow:
    return VisualQcRow(
        artifact_id=artifact_id,
        check_id=check_id,
        severity=severity,
        status=severity,
        scope=scope,
        message=message,
        metric=metric,
        value=value,
        threshold=threshold,
        row_index=row_index,
        column=column,
    )


def _expand_template(template: str, metadata: Mapping[str, Any], *, field_name: str) -> tuple[str, tuple[str, ...]]:
    formatter = Formatter()
    errors: list[str] = []
    parts: list[str] = []
    for literal, field_name_raw, format_spec, conversion in formatter.parse(template):
        parts.append(literal)
        if field_name_raw is None:
            continue
        root_name = field_name_raw
        if "." in root_name or "[" in root_name or "]" in root_name:
            errors.append(f"Template field {root_name!r} in {field_name!r} is not a simple metadata key.")
        elif root_name not in metadata:
            errors.append(f"Template field {root_name!r} in {field_name!r} is missing from metadata.")
        elif "{" in format_spec or "}" in format_spec:
            errors.append(f"Template format spec for {root_name!r} in {field_name!r} must not contain nested fields.")
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
                    errors.append(f"Template format spec for {root_name!r} in {field_name!r} failed: {exc}")
                    text = _display_scalar(value)
            parts.append(text)
            continue
        parts.append("")
    return "".join(parts), tuple(errors)


def _merged_metadata(*mappings: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        if mapping:
            merged.update(_json_safe_mapping(mapping))
    return merged


def _text_width_estimate(text: str, font_size: int | float) -> float:
    return len(text) * float(font_size) * 0.58


def _has_text(value: str | None) -> bool:
    return value is not None and str(value).strip() != ""


def _maybe_finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            number = float(stripped)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _normalize_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        mapping = _as_mapping(row, field_name=f"rows[{index}]")
        normalized.append({str(key): _json_safe(value) for key, value in mapping.items()})
    return tuple(normalized)


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif is_dataclass(value):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or expose to_dict().")
    return value


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


def _xml(value: Any) -> str:
    return html.escape(_display_scalar(value), quote=True)


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
    "AxisTextSpec",
    "CaptionSpec",
    "FigureTextSpec",
    "LayoutSpec",
    "LegendSpec",
    "PlotSpec",
    "PointIntervalPlotSpec",
    "VisualQcRow",
    "VisualQcSpec",
    "build_point_interval_plot_spec",
    "build_visual_layout_qc_rows",
    "render_point_interval_svg",
]
