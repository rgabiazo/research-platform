# ADR-0014: Generic Tabular Association Workflow Roadmap

## Status

Accepted

## Context

The platform has reusable MVPA and subject-level inference foundations, but it
also needs a generic workflow contract for subject-level and other rectangular
tabular association analyses beyond MVPA. Examples include correlations,
partial correlations, covariate-adjusted associations, regression-style
associations, repeated-measures or mixed-model-style planning, group
comparisons, behavioral-neuroimaging association tables, ROI-to-behavior
associations, biomarker-to-outcome associations, and other subject-level or
tabular analyses.

This ADR began as a docs-only roadmap and audit. Step 11B now adds
standard-library-only schema and plan-preview contracts in `research-analysis`,
Step 11C adds no-write source inventory/QC, and Step 11D adds bounded
same-source Pearson/Spearman association rows with QC/provenance. Step 11E adds
bounded same-source residualized partial/covariate-adjusted and OLS
primary-predictor regression-style association rows. Step 11F adds no-write
Benjamini-Hochberg q-value support from already-supplied valid p-value fields
in generic association result rows. Step 11G adds no-write publication-table
handoff rows through the existing `publication_tables.py` helpers for supplied
in-memory association, multiplicity, QC, missingness, and provenance rows. Step
11H adds pathless, no-write visualization/report handoff rows and specs in
`research-viz` for supplied in-memory association, publication, multiplicity,
QC, missingness, and provenance rows. Step 11I-A is a docs-only policy decision
for optional dataframe backend dependency/extras and ownership boundaries; it
does not implement dataframe adapters or change runtime behavior. Step 11I-B
adds an available standard-library-only row-source/records coercion layer in
`research-analysis` with fake dataframe-like protocol tests while keeping
concrete pandas/Polars adapters outside `research-analysis`. Step 11I-C adds
available standard-library-only dataframe-like-to-records helpers in `research-io` with
fake dataframe-like protocol tests, no output writes, and no pandas, Polars, or
NumPy imports. Step 11I-D adds available explicit optional pandas
DataFrame-to-records helpers in `research-io` with lazy pandas imports only
inside pandas adapter calls; it does not make pandas an import-time dependency
or change `research-analysis`. Step 11I-E adds available explicit optional
Polars DataFrame-to-records helpers in `research-io` with lazy Polars imports
only inside Polars backend or records adapter calls; it does not make Polars an
import-time dependency or change `research-analysis`. Step 11I-F adds an
available `research-io` convenience helper that converts explicit source-ID
mappings to no-write `source_rows_by_id` records for existing
`research-analysis` Step 11C-G APIs without importing or wrapping
`research-analysis`. Step 11J-A adds available no-write repeated-measures and
mixed-model design/QC planning in `research-analysis`, and Step 11J-B adds
available metadata-only fixed/random effect, repeated/within/between factor,
grouping/cluster, timepoint, categorical-coding, formula/design-intent,
planned-comparison, contrast, family, and link declarations plus reference QC.
Step 11J-C adds available supplied-only model-result row contracts,
normalization, validation, QC, and provenance for externally computed future
repeated-measures or mixed-model outputs without adding model fitting or
statistical computation.
The roadmap still does not add dependencies,
dependency extras, project overlays, pipeline or ops behavior, visualization
implementation, publication file writing, mixed-model, p-value,
confidence-interval, effect-size, or non-BH FDR execution.

## Decision

Generic tabular association workflows will be config-driven, row-oriented, and
package-owned by responsibility. `research-analysis` owns the association
schema, validation, statistical planning, generic statistics, row shaping, QC,
provenance, and handoff rows. `research-viz` owns reusable rendering only.
`research-core` stays thin and orchestration-only.

Association outputs should remain generic, JSON/TSV/CSV-safe rows that existing
publication table and visualization/report helpers can consume. The platform
will not create separate publication or plotting systems for association
workflows. Generic association contracts must not bake in MVPA-specific,
neuro-specific, study-specific, task-specific, subject-specific,
project-specific, or cluster-specific labels.

## Existing Support Found

The repository already has useful pieces, but not a complete association
workflow contract.

In `research-analysis`:

- generic subject-level inference summaries with result, multiplicity,
  leave-one-subject-out sensitivity, missingness, QC, provenance, and
  JSON/TSV-safe row shapes;
- generic publication table and manifest helpers that consume already-computed
  rectangular rows and write display tables, machine-readable tables, and
  manifests without recomputing analysis outputs;
- lightweight statistics helpers for Pearson and Spearman correlations,
  summary tables, small ordinary least-squares summaries, one-way group
  summaries, and mixed-effects-ready grouped summaries;
- lightweight CSV/TSV/JSON tabular helpers used by analysis commands.

In `research-viz`:

- reusable visualization and report planning/rendering that consumes
  already-computed rows or simple source files;
- dependency-free SVG point/interval rendering and Markdown, HTML, and text
  report rendering;
- visual QC rows for figure dimensions, fonts, label density, long labels,
  data-label crowding, legends, captions, and required text fields;
- configurable text controls for titles, subtitles, captions, footnotes, axis
  labels, legend titles and labels, panel labels, alt text, and methods notes;
- plan/render separation through preview and write result objects.

In `research-io`:

- optional table read/write helpers for CSV, TSV, and related formats;
- a dataframe backend protocol and adapter pattern with Polars and pandas
  backends available in the IO layer;
- backend discovery and lazy-import guards around optional dataframe behavior.

In `research-core`:

- schema-only generic analysis workflow recipe contracts and plan previews;
- orchestration-only validation of workflow declarations, root references,
  output roots, publication settings, reporting settings, and package extension
  handoffs;
- no domain execution or statistical ownership in the core workflow schema.

## Missing Support

After Step 11H, `research-analysis` has schema-only declarations, plan preview
rows, no-write source QC, bounded same-source Pearson/Spearman association
rows, residualized partial/covariate-adjusted association rows, and OLS
primary-predictor regression-style association rows for generic tabular
association workflows, plus no-write Benjamini-Hochberg q-values from
already-present valid p-value fields in supplied in-memory result rows, and
no-write publication-table handoff rows for supplied in-memory rowsets.
`research-viz` has pathless, no-write visualization/report handoff rows and
specs for those supplied in-memory rowsets. Runtime execution is still missing
for:

- repeated-measures or mixed-model fitting;
- non-BH multiple-comparison and FDR correction;
- computed effect-size, confidence-interval, p-value, diagnostic, and
  model-estimate fields;
- publication table file writing;
- visualization and report rendering from association handoff bundles.

The existing lightweight statistics helpers are useful precedents, but they are
not yet a full association workflow specification with validated input schemas,
standardized result rows, missingness/duplicate policies, multiple-testing
families, publication handoff rows, visualization handoff rows, and provenance.

## Ownership Boundaries

- `research-analysis` owns generic tabular association schemas, validation,
  statistical planning, generic statistics, subject-level summaries, table
  shaping, QC/provenance rows, and publication table/manifest handoff.
- `research-viz` owns reusable visualization and report rendering only.
- `research-neuro` owns neuro, MVPA, fMRI, ROI, and extraction semantics only.
- `research-core` stays thin and orchestration-only.
- `research-bids` owns BIDS-like path helpers only when needed.
- `research-io` may provide optional file and table IO helpers where
  appropriate.
- `research-ml` is not the default owner for generic statistical association
  summaries unless a future slice introduces true ML or modeling components.
- Project overlays stay limited to config and manifests.
- `pipelines` and `ops` remain later orchestration and HPC layers.

## Planned Roadmap and API Concepts

Future implementation slices should add small, composable contracts in
`research-analysis`. Candidate concepts include:

- `TabularAssociationWorkflowSpec`;
- `TabularSourceSpec`;
- `TabularSchemaSpec`;
- `AssociationVariableSpec`;
- `PredictorSpec`;
- `OutcomeSpec`;
- `CovariateSpec`;
- `GroupingSpec`;
- `MissingDataPolicy`;
- `DuplicateSubjectPolicy`;
- `AssociationMethodSpec`;
- `CorrelationSpec`;
- `PartialCorrelationSpec`;
- `RegressionAssociationSpec`;
- `RepeatedMeasuresAssociationSpec`, as a later or deferred slice if needed;
- `MultipleTestingSpec`;
- `AssociationResultRow`;
- `AssociationQcRow`;
- `AssociationMissingnessRow`;
- `AssociationProvenanceRow`;
- `AssociationSummaryResult`;
- publication-table handoff rows;
- visualization/report handoff rows;
- optional dataframe backend adapter specs.

These concepts should accept explicit column names and metadata. They should not
infer study conventions, task labels, subject labels, ROI labels, contrast
labels, or project-specific identifiers from names.

## Step 9 Compatibility

Association outputs should be generic rows that the existing
`research-analysis` publication table and manifest helpers can consume. The
workflow should not create a separate publication table system.

Result rows should expose stable generic fields for labels, estimates,
effect-size values, confidence intervals, p-values, q-values, family IDs,
status, warnings, QC, missingness, and provenance. Publication-specific display
formatting belongs in the existing publication table helpers.

Generic association workflows must not hard-code MVPA, ROI, Crossnobis,
sign-flip, FDR, dz, or other standardized-effect labels into publication table
behavior.

Multiple-testing methods may be represented as configured metadata and generic
q-value fields, not as domain-specific publication assumptions.

## Step 10 Compatibility and Publication-Ready Visualization

Association workflows should not create a parallel plotting or reporting
system. Association modules should emit generic result, QC, provenance, and
handoff rows. `research-viz` should own:

- forest and point/interval plots;
- future scatter or association plots;
- future residual and diagnostic plots;
- Markdown, HTML, and text reports;
- visual QC;
- captions;
- legends;
- axes;
- labels;
- reusable text controls.

Publication-ready visual and report requirements include:

- layout and QC preflight for overlapping text;
- layout and QC preflight for overlapping or crowded data labels;
- readable fonts;
- adequate figure dimensions;
- appropriate spacing;
- long-label wrapping or rotation warnings;
- dense tick-label warnings;
- legend, title, and caption collision warnings;
- missing title, axis, legend, caption, or alt-text warnings when required;
- configurable reusable text controls for titles, subtitles, captions,
  footnotes, x-axis labels, y-axis labels, legend titles, legend labels, panel
  labels, alt text, and methods/provenance notes;
- safe metadata-template expansion for labels and captions;
- no hard-coded MVPA, ROI, task, study, subject, or project labels.

Association result rows should provide enough generic columns for
`research-viz` to render figures and reports without learning association
statistics or domain extraction semantics.

## Pandas, Polars, and Backend Policy

Step 11I-A is a docs-only dependency/extras and ownership policy decision. It
adds no dependency or dependency-extras metadata changes, no dataframe adapter
implementation, no package runtime behavior, and no package import changes.
Step 11I-B is now available as the first bounded runtime slice: it adds only a
standard-library row-source protocol and records coercion helper in
`research-analysis`.
Step 11I-C is now available in `research-io` as a reusable,
standard-library-only dataframe-like-to-records helper layer. It produces
copied ordered records, JSON/TSV-safe QC and provenance rows, no-write flags,
and optional deterministic `input_row_index` fields from generic fake
dataframe-like protocols only.
Step 11I-D is now available in `research-io` as explicit optional pandas
DataFrame-to-records adapter support. The helpers live in
`research_platform.io.dataframe.pandas_ops`, lazy-import pandas only inside
explicit pandas adapter calls, accept real pandas DataFrames only, keep
`runtime_backend` as `records`, report `requested_backend` as pandas metadata,
and produce the same no-write `DataframeRecordsResult` shape.
Step 11I-E is now available in `research-io` as explicit optional Polars
DataFrame-to-records adapter support. The helpers live in
`research_platform.io.dataframe.polars_ops`, lazy-import Polars only inside
explicit Polars backend or records adapter calls, accept real Polars
DataFrames only, reject Series and LazyFrames without collecting them, keep
`runtime_backend` as `records`, report `requested_backend` as Polars metadata,
and produce the same no-write `DataframeRecordsResult` shape.
Step 11I-F is now available in `research-io` as explicit source-ID mapping
conversion for association workflows. The helper lives in
`research_platform.io.dataframe.association_records`, converts each selected
records, dataframe-like, pandas, or Polars source to copied records, aggregates
existing dataframe conversion QC/provenance rows, and returns a no-write
`source_rows_by_id` mapping ready for existing `research-analysis` Step 11C-G
functions. It does not import `research-analysis`, auto-detect pandas or
Polars objects, compute statistics, compute multiplicity, render
visualizations or reports, or write outputs.

Records and mapping rows remain the default and canonical runtime input for
generic tabular association workflows in `research-analysis`. Lightweight
`research-analysis` workflows must remain dependency-free at import time, and
association workflows in `research-analysis` must not import pandas, Polars, or
`research-io` at import time.

Concrete dataframe-to-record adapters should live in `research-io` or in
caller-side utilities, not in core association execution logic.
The Step 11I-B `research-analysis` adapter accepts only copied records and
generic record-producing protocols such as mapping-row sequences, row-level
dataclasses or `to_dict()` mappings, fake `to_dicts()`, mapping-like
`to_records()`, named `iter_rows(...)`, `rows`, and `records`. These fake
dataframe-like tests validate duck typing only; they are not real pandas or
Polars support.

Dataframe backend adapters follow these constraints:

- no hard Polars default is added to `research-analysis`;
- pandas remains an optional interoperability backend for notebooks,
  ecosystem compatibility, and user-provided dataframe inputs;
- Polars remains an optional backend for large, repetitive,
  transformation-heavy tabular workflows;
- pandas and Polars adapters must use lazy imports;
- optional pandas and Polars tests must skip when those packages are absent;
- concrete pandas and Polars records adapters remain explicit opt-in helpers;
- dataframe adapters should convert dataframe-like inputs to records or
  mappings before the existing Step 11C-G logic runs;
- dataframe adapters must not duplicate QC, correlation, adjusted/regression,
  multiplicity, publication, or visualization logic;
- outputs should remain JSON/TSV/CSV safe.

The existing `research-io` dataframe adapter pattern is a useful precedent, but
association workflow contracts should remain usable without importing
`research-io` dataframe backends unless a caller explicitly chooses that path.
Dataframe-adapter slices should preserve row order, produce deterministic
`input_row_index` fields, and avoid mutating user-supplied dataframe objects.

## Step 11I Sub-Roadmap

Step 11I should remain PR-sized and adapter-oriented:

1. Step 11I-A: docs-only dependency/extras and ownership policy.
2. Step 11I-B: standard-library records/row-source protocol in
   `research-analysis` with fake dataframe-like tests only.
3. Step 11I-C: `research-io` dataframe-to-records helpers with fake
   dataframe-like tests and no pandas or Polars import at package import time.
   Available via `research_platform.io.dataframe.records`.
4. Step 11I-D: optional pandas adapter with lazy imports and skipped tests.
   Available via `research_platform.io.dataframe.pandas_ops` as explicit
   pandas DataFrame-to-records support only.
5. Step 11I-E: optional Polars adapter with lazy imports and skipped tests.
   Available via `research_platform.io.dataframe.polars_ops` as explicit
   Polars DataFrame-to-records support only.
6. Step 11I-F: optional convenience integration that feeds converted records
   into existing Step 11C-G functions without duplicating analysis logic.
   Available via `research_platform.io.dataframe.association_records` as an
   explicit no-write `source_rows_by_id` preparation helper in `research-io`.

## Future Test Roadmap

Step 11I-C adds synthetic fake dataframe-like tests in `research-io`. Future
tests should continue to use synthetic tabular rows only and should not require
real fMRI data, NIfTI files, FSL, ANTs, rsatoolbox, pandas, Polars, scipy, or
new dependencies.

Future tests should cover:

- schema validation;
- Pearson and Spearman associations;
- partial and covariate-adjusted associations;
- missingness policies;
- duplicate policies;
- grouping and stratification;
- FDR families;
- publication table handoff;
- visualization handoff and text controls;
- backend adapter behavior with lazy imports;
- JSON/TSV safety;
- absence of study-specific constants.

## PR-Sized Roadmap

1. Step 11A: docs-only ADR and ADR index update.
2. Step 11B: schema-only association specs in `research-analysis`, standard
   library only. Available via
   `research_platform.analysis.tabular_associations`.
3. Step 11C: source inventory, schema validation, missingness, duplicate,
   non-finite, categorical/numeric QC. Available in `research-analysis` via
   `plan_tabular_association_qc` and `run_tabular_association_qc` with
   standard-library records/TSV/CSV/JSON inspection only.
4. Step 11D: Pearson/Spearman association rows with QC/provenance and no new
   dependencies unless separately approved. Available in `research-analysis`
   via `plan_tabular_association_correlations` and
   `run_tabular_association_correlations` for same-source, records-backed
   Pearson/Spearman rows only.
5. Step 11E: partial/covariate-adjusted and regression-style association
   planning/results. Available in `research-analysis` via
   `plan_tabular_association_adjusted` and
   `run_tabular_association_adjusted` for same-source, records-backed numeric
   residualized partial associations and OLS primary-predictor coefficients
   only.
6. Step 11F: multiple-comparison/FDR family support for association result
   rows. Available in `research-analysis` via
   `plan_tabular_association_multiplicity` and
   `run_tabular_association_multiplicity` for no-write Benjamini-Hochberg
   q-values from already-supplied valid p-value fields in in-memory generic
   result rows only.
7. Step 11G: publication-table handoff through existing
   `publication_tables.py`. Available in `research-analysis` via
   `plan_tabular_association_publication_tables` and
   `build_tabular_association_publication_tables` for in-memory no-write
   display/machine row handoff, manifest rows, and provenance rows.
8. Step 11H: visualization/report handoff through existing `research-viz`.
   Available in `research-viz` via
   `plan_tabular_association_visualization_handoff` and
   `build_tabular_association_visualization_handoff` for pathless, no-write,
   records-backed point/interval dataset rows, visual QC rows, report handoff
   specs, manifest rows, and provenance rows.
9. Step 11I: optional dataframe backend policy and adapters, split into the
   scoped Step 11I-A through Step 11I-F sub-roadmap above. Step 11I-A is
   docs-only and adds no dependency or extras metadata changes; Step 11I-C is
   available in `research-io` for dependency-free fake dataframe-like
   conversion to records; Step 11I-D is available in `research-io` for
   explicit optional pandas DataFrame-to-records conversion; Step 11I-E is
   available in `research-io` for explicit optional Polars DataFrame-to-records
   conversion; Step 11I-F is available in `research-io` for explicit no-write
   source-ID mapping conversion to `source_rows_by_id` for downstream
   `research-analysis` Step 11C-G calls.
10. Step 11J-A: repeated-measures/mixed-model design and QC planning.
   Available in `research-analysis` via
   `plan_tabular_association_repeated_measures` and
   `run_tabular_association_repeated_measures_design_qc` for no-write,
   records-backed model-plan rows, long-format design summaries, factor
   summaries, QC rows, and provenance rows. This is design/QC planning only;
   mixed-model fitting, repeated-measures ANOVA, GLMM fitting, coefficients,
   p-values, q-values, confidence intervals, effect sizes, residuals,
   diagnostics, visualization, reports, publication writing, dataframe
   support, CLI behavior, and new dependencies remain out of scope.
11. Step 11J-B: richer repeated-measures/mixed-model metadata declarations.
   Available in `research-analysis` through the same Step 11J-A plan/QC APIs
   and public metadata-only spec objects. This normalizes fixed-effect terms,
   random-effect terms, random intercepts/slopes, repeated factors,
   within-subject factors, between-subject factors, grouping factors, cluster
   terms, timepoint roles, categorical coding, formula/design intent, planned
   comparisons, contrast metadata, model family, and link metadata into
   model-plan rows, and adds duplicate-ID, unknown-reference, malformed
   metadata, and supplied-record missing-column QC. This remains metadata and
   reference QC only; mixed-model fitting, repeated-measures ANOVA, GLMM
   fitting, coefficients, p-values, q-values, confidence intervals, effect
   sizes, residuals, diagnostics, visualization, reports, publication writing,
   dataframe support, CLI behavior, and new dependencies remain out of scope.
12. Step 11J-C: supplied-only model-result row contracts. Available in
   `research-analysis` through
   `plan_tabular_association_model_results`,
   `validate_tabular_association_model_result_rows`, and
   `normalize_tabular_association_model_result_rows` for in-memory externally
   supplied model-fit summary, fixed-effect, random-effect,
   variance-component, planned-comparison, and contrast rows. The slice adds
   JSON/TSV-safe normalized rows, validation/QC for identifiers, result kinds,
   numeric fields, p/q values, confidence interval bounds, optional model-plan
   and design metadata references, plus provenance documenting
   `runtime_backend="records"`, `supplied_only=True`,
   `computed_by_research_analysis=False`, no model fitting, and no output
   writes. It does not fit mixed models, repeated-measures ANOVA, GLMMs, or
   regressions, and does not compute coefficients, standard errors, p-values,
   q-values, confidence intervals, FDR, effect sizes, residuals, diagnostics,
   visualizations, reports, publication files, dataframe support, CLI
   behavior, or dependencies.
13. Later: core CLI, pipeline, and HPC orchestration only after package APIs
    stabilize.

## Consequences

Positive:

- Generic association work can build on existing subject-level inference,
  publication, visualization, IO, and orchestration foundations.
- Project overlays remain thin and declarative.
- Publication and visualization handoffs stay reusable instead of creating
  duplicate systems.
- Dataframe dependencies remain optional and adapter-driven.

Tradeoffs:

- Association execution remains intentionally narrow: only same-source
  Pearson/Spearman, residualized partial/covariate-adjusted, OLS
  primary-predictor regression-style rows, and Benjamini-Hochberg q-values from
  supplied p-values are executable after Step 11H; publication and
  visualization/report support are in-memory no-write handoffs only.
- Repeated-measures and mixed-model behavior is now limited to Step 11J-A
  design/QC planning, Step 11J-B metadata-only declarations and reference QC,
  and Step 11J-C supplied-only externally computed model-result row contracts;
  actual model fitting still needs a later, larger slice.
- Statistical semantics, QC fields, and publication/visualization handoff rows
  must be specified carefully before runtime behavior is added.
