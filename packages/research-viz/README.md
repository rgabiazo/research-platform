# research-viz

> **Alpha status — experimental integration.** Reusable visualization and
> report specifications, dependency-free renderers, and synthetic API tests
> exist. The capability remains **Experimental or external-runtime** because no
> checked-in project report configuration or guided `rp` workflow connects
> these interfaces into a supported end-to-end analysis.

`research-viz` turns already-computed rectangular rows into reviewable figure,
table, report, visual-QC, and manifest artifacts. It does not calculate the
statistics represented by those rows.

## Ownership and boundaries

This package owns:

- reusable plot, axis, legend, caption, layout, table, and report
  specifications;
- deterministic point/interval SVG rendering;
- Markdown, static HTML, and plain-text report rendering;
- configured output planning, explicit rendering, hashes, provenance, warning
  rows, error rows, and visual-layout QC;
- pathless, no-write visualization handoffs for already-computed generic
  tabular association rows.

It does not own:

- statistical estimation, confidence intervals, p-values, resampling,
  crossnobis mathematics, or other analysis;
- project-specific figure meaning, scientific labels, reporting policy, or
  publication approval;
- a dashboard product, web application, Streamlit application, desktop client,
  or notebook UI;
- a general project-level report workflow or an integrated analysis-to-report
  pipeline;
- BIDS, neuroimaging, HPC, or remote-runtime behaviour.

`research-analysis` computes generic statistics and publication-table rows.
`research-viz` consumes those completed rows and caller-supplied labels,
intervals, metadata, and output specifications. It does not recompute or infer
the scientific result.

## Relationship to `rp`

The top-level `rp` command is the integrated workspace interface, but there is
currently no guided `rp` visualization lifecycle. `research-viz` provides
package-level Python interfaces for advanced consumers and package
integration. Interface availability does not promote the incomplete project
workflow to supported status.

## Install from a source checkout

The alpha packages are not published on PyPI. The repository's `full` profile
currently includes `research-viz`:

```bash
bash ops/envs/dev/bootstrap.sh --profile full
source .venv/bin/activate
python -c "import research_platform.viz"
```

Use Python 3.11 or 3.12 and follow the
[source-checkout quickstart](../../docs/onboarding/quickstart.md). The `full`
profile contains additional optional workspace dependencies; their
installation does not expand this package's support claims.

## Plan-first example

This synthetic example previews an SVG point/interval figure without writing a
file:

```python
from research_platform.viz import (
    VisualizationOutputSpec,
    build_point_interval_plot_spec,
    plan_visualization_outputs,
)

rows = [
    {"label": "effect-a", "estimate": 0.20, "low": 0.10, "high": 0.30},
    {"label": "effect-b", "estimate": 0.35, "low": 0.22, "high": 0.48},
]

plot_spec = build_point_interval_plot_spec(
    label_column="label",
    estimate_column="estimate",
    lower_column="low",
    upper_column="high",
    title="Synthetic effects",
    x_label="Caller-provided estimate",
    y_label="Effect",
)

plan = plan_visualization_outputs(
    rows,
    output_spec=VisualizationOutputSpec(
        output_root="artifacts/viz-example",
        figure_svg_path="effects.svg",
    ),
    plot_spec=plot_spec,
)

assert plan.status == "ok"
assert plan.output_written is False
```

`plan_visualization_outputs(...)` validates sources, required columns, output
paths, format availability, templates, and visual layout. It returns previews,
structured QC, warnings, errors, provenance, and a manifest without writing.

`render_visualization_outputs(...)` is the separate write gate. It writes only
the configured report, figure, and manifest paths, refuses existing outputs by
default, and hashes the artifacts it creates.

## Inputs and outputs

| Surface | Inputs | Result |
| --- | --- | --- |
| Plot/report specification | Caller-selected columns, text, layout, sections, metadata, and already-computed values | Immutable reusable specification objects |
| Output planning | In-memory rows or TSV/CSV/JSON sources plus output paths | No-write previews, planned artifacts, visual-QC rows, warnings/errors, provenance, and manifest data |
| Explicit rendering | A valid plan-equivalent request and configured destinations | Markdown, HTML, text, SVG, and JSON artifacts only where requested |
| Tabular association handoff | Already-computed association and publication rows | Pathless in-memory datasets, specs, QC, manifest, and provenance rows |

The renderer uses supplied estimates and interval bounds as-is. It does not
derive intervals, calculate statistics, select scientific labels, or decide
which result is publishable.

## Formats and dependencies

`research-viz` declares no third-party runtime dependency. Built-in rendering
uses the Python standard library:

- reports: Markdown, static HTML, and plain text;
- figures: SVG point/interval output;
- manifests: JSON.

PNG and PDF figure requests do not silently import or assume a plotting stack.
They produce explicit unavailable-renderer warnings or errors according to the
configured policy. Supporting another renderer requires a separately reviewed
dependency and integration; it is not implied by the output specification.

The checked-in `dashboards.py` module is a placeholder. There is no supported
dashboard or application surface.

## Evidence and limitations

Focused tests verify:

- plan mode writes nothing;
- render mode writes only configured files;
- existing outputs are refused by default;
- in-memory, TSV, CSV, and JSON sources;
- Markdown, HTML, text, SVG, and JSON behaviour;
- caller-supplied labels, templates, intervals, tables, and figure references;
- deterministic visual-QC findings, warnings, errors, hashes, and provenance;
- pathless tabular-association handoff behaviour.

Those tests use synthetic rows. They do not establish a project report
configuration, publication workflow, dashboard, integrated `rp` lifecycle, or
scientific validity of user-supplied estimates.

Further reading:

- [Capability matrix](../../docs/capabilities.md)
- [Architecture and package ownership](../../ARCHITECTURE.md)
- [Tabular analysis slice](../../docs/tabular-slice.md)
- [ADR-0012: MVPA crossnobis foundation](../../docs/decisions/ADR-0012-mvpa-crossnobis-foundation.md)
- [ADR-0014: generic tabular association workflow](../../docs/decisions/ADR-0014-generic-tabular-association-workflow-roadmap.md)
