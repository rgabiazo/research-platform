# ADR-0014: Generic Tabular Association Workflow

## Status

Accepted

## Context

The platform has reusable MVPA and subject-level inference foundations, but it
also needs a generic workflow contract for subject-level and other rectangular
tabular association analyses beyond MVPA: correlations, partial correlations,
covariate-adjusted associations, regression-style associations,
repeated-measures or mixed-model design planning, group comparisons,
behavioral-neuroimaging association tables, ROI-to-behavior associations, and
biomarker-to-outcome associations.

Before this decision the repository had generic subject-level inference
summaries and publication table helpers in `research-analysis`, visualization
and report rendering with visual QC in `research-viz`, optional table IO and a
dataframe backend protocol in `research-io`, and schema-only workflow recipe
contracts in `research-core`. None of them formed a complete association
workflow with validated input schemas, standardized result rows, missingness
and duplicate policies, multiple-testing families, handoff rows, and
provenance.

## Decision

Generic tabular association workflows are config-driven, row-oriented, and
package-owned by responsibility:

- `research-analysis` owns the association schema, validation, statistical
  planning, generic statistics, row shaping, QC, provenance, and publication
  table handoff.
- `research-viz` owns reusable visualization and report rendering only.
- `research-io` owns the optional dataframe-to-records adapters.
- `research-neuro` owns neuro, MVPA, fMRI, ROI, and extraction semantics only.
- `research-core` stays thin and orchestration-only, and `research-ml` is not
  the owner of generic statistical association summaries.
- Project overlays stay limited to config and manifests; `pipelines` and `ops`
  remain the orchestration and HPC layers.

Association outputs are generic JSON/TSV/CSV-safe rows that the existing
publication table and visualization/report helpers consume; the platform does
not create separate publication or plotting systems for association
workflows. Association contracts accept explicit column names and metadata and
do not infer study, task, subject, ROI, contrast, or project conventions from
names. They do not bake in MVPA-specific, neuro-specific, study-specific,
task-specific, subject-specific, project-specific, or cluster-specific labels.
Multiple-testing methods are configured metadata and generic q-value fields,
not domain-specific publication assumptions.

## What the contract covers

In `research-analysis` (`research_platform.analysis.tabular_associations`),
standard library only and without output writes:

- source, schema, variable (outcome, predictor, covariate, grouping), policy
  (missing data, duplicate subject, non-finite, standardization,
  transformation), method (correlation, partial correlation, regression,
  repeated measures), family, multiple-testing, output, and handoff specs;
  `validate_tabular_association_workflow_document` and
  `plan_tabular_association_workflow` validate a workflow document and preview
  its plan;
- `plan_tabular_association_row_source_adapter` coerces copied records and
  generic record-producing protocols (mapping-row sequences, row-level
  dataclasses or `to_dict()` mappings, `to_dicts()`, mapping-like
  `to_records()`, named `iter_rows(...)`, `rows`, `records`); this is duck
  typing, not pandas or Polars support;
- `plan_tabular_association_qc` and `run_tabular_association_qc`: source
  inventory, schema validation, missingness, duplicate, non-finite, and
  categorical/numeric QC;
- `plan_tabular_association_correlations` and
  `run_tabular_association_correlations`: same-source, records-backed
  Pearson/Spearman rows with QC and provenance;
- `plan_tabular_association_adjusted` and `run_tabular_association_adjusted`:
  same-source residualized partial/covariate-adjusted rows and OLS
  primary-predictor coefficients;
- `plan_tabular_association_multiplicity` and
  `run_tabular_association_multiplicity`: Benjamini-Hochberg q-values from
  already-supplied valid p-value fields in in-memory result rows;
- `plan_tabular_association_publication_tables` and
  `build_tabular_association_publication_tables`: in-memory display and
  machine row handoff, manifest rows, and provenance rows through the existing
  `publication_tables.py` helpers;
- `plan_tabular_association_repeated_measures` and
  `run_tabular_association_repeated_measures_design_qc`: records-backed
  model-plan rows, long-format design summaries, factor summaries, QC rows, and
  provenance rows, with metadata-only declarations for fixed-effect terms,
  random intercepts and slopes, repeated, within-subject, between-subject and
  grouping factors, cluster terms, timepoint roles, categorical coding,
  formula and design intent, planned comparisons, contrasts, model family, and
  link, plus duplicate-ID, unknown-reference, malformed-metadata, and
  missing-column QC;
- `plan_tabular_association_model_results`,
  `validate_tabular_association_model_result_rows`, and
  `normalize_tabular_association_model_result_rows`: supplied-only,
  externally computed model-fit summary, fixed-effect, random-effect,
  variance-component, planned-comparison, and contrast rows, normalized and
  validated (identifiers, result kinds, numeric fields, p and q values,
  confidence-interval bounds, optional model-plan references), with provenance
  recording `runtime_backend="records"`, `supplied_only=True`, and
  `computed_by_research_analysis=False`.

In `research-viz` (`research_platform.viz.tabular_associations`):
`plan_tabular_association_visualization_handoff` and
`build_tabular_association_visualization_handoff` produce pathless, no-write,
records-backed point/interval dataset rows, visual QC rows, report handoff
specs, manifest rows, and provenance rows. `research-viz` owns forest and
point/interval plots, Markdown, HTML, and text reports, visual QC (overlapping
or crowded text and data labels, readable fonts, figure dimensions, spacing,
long-label and dense-tick warnings, legend, title and caption collisions,
missing required text), and reusable text controls for titles, subtitles,
captions, footnotes, axis labels, legend titles and labels, panel labels, alt
text, and methods notes, with safe metadata-template expansion.

In `research-io` (`research_platform.io.dataframe`): `records` converts
generic dataframe-like objects to copied ordered records with QC and
provenance rows and deterministic `input_row_index` fields, standard library
only; `pandas_ops` and `polars_ops` convert real pandas and Polars DataFrames
to the same `DataframeRecordsResult` shape with lazy imports inside the
adapter calls (Polars Series and LazyFrames are rejected without collecting);
`association_records` converts a source-ID mapping of records, dataframe-like,
pandas, or Polars sources into a no-write `source_rows_by_id` mapping for the
`research-analysis` functions above without importing `research-analysis` or
auto-detecting dataframe objects.

## Dataframe backend policy

- Records and mapping rows are the canonical runtime input. Association
  workflows in `research-analysis` do not import pandas, Polars, or
  `research-io` at import time.
- No hard Polars default is added to `research-analysis`. pandas is an
  optional interoperability backend for notebooks and user-provided dataframe
  inputs; Polars is an optional backend for large, transformation-heavy
  tabular work.
- Adapters use lazy imports, their tests skip when the package is absent, and
  they convert inputs to records before the analysis logic runs. They
  preserve row order, produce deterministic `input_row_index` fields, never
  mutate user-supplied dataframe objects, and do not duplicate QC,
  correlation, adjusted/regression, multiplicity, publication, or
  visualization logic.
- Outputs stay JSON/TSV/CSV safe.

## Not covered

No runtime exists for repeated-measures or mixed-model fitting
(repeated-measures ANOVA, GLMM, regression fits), non-Benjamini-Hochberg
multiple-comparison correction, computed effect sizes, confidence intervals,
p-values, residuals, diagnostics, or model estimates, publication table file
writing, visualization or report rendering from association handoff bundles,
CLI commands, pipelines, or HPC orchestration. The decision adds no
dependencies or dependency extras. Tests use synthetic tabular rows only and
need no real fMRI data, NIfTI files, FSL, ANTs, rsatoolbox, pandas, Polars, or
scipy.

## Consequences

Positive:

- Generic association work builds on the existing subject-level inference,
  publication, visualization, IO, and orchestration foundations.
- Project overlays remain thin and declarative.
- Publication and visualization handoffs stay reusable instead of creating
  duplicate systems.
- Dataframe dependencies remain optional and adapter-driven.

Tradeoffs:

- Association execution is narrow: same-source Pearson/Spearman, residualized
  partial/covariate-adjusted, and OLS primary-predictor rows, plus
  Benjamini-Hochberg q-values from supplied p-values. Publication and
  visualization/report support are in-memory no-write handoffs only.
- Repeated-measures and mixed-model behavior is limited to design/QC planning,
  metadata-only declarations with reference QC, and supplied-only externally
  computed model-result rows.
- Statistical semantics, QC fields, and handoff rows must be specified
  carefully before runtime behavior is added.
