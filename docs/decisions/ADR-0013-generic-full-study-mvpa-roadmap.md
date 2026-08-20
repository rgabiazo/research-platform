# ADR-0013: Generic Full-Study Analysis Workflow Roadmap

## Status

Accepted

## Context

The platform needs a generic, full-study analysis workflow that can move from
run-level model outputs through subject-level summaries, leakage-aware ROI
definition, MVPA extraction, table-based inference, and later reporting without
turning project overlays into executable code.

MVPA is the motivating example, but the workflow should remain useful for other
full-study analyses that need inventory, planning, execution boundaries,
provenance, and publication-ready outputs.

## Decision

Full-study workflows will be config-driven, plan-first, and package-owned by
domain boundary. `research-core` stays thin and orchestration-only. Project
overlays stay config and manifests only. `pipelines` and `ops` remain later
layers for orchestration, HPC, and remote execution after package behavior is
stable.

## Current Support

The repository already has reusable foundations for:

- thin analysis lifecycle commands that locate project config and delegate work
  to packages;
- generic ROI build and extraction planning, with local execution for supported
  ROI families and existing FEAT-style extraction sources;
- local LOSO ROI generation from existing subject-level fixed-effects inputs;
- MVPA/crossnobis configuration, plan previews, neuro pattern extraction
  runtime outputs, prepared distance summaries, runtime MVPA refinements, and
  publication previews;
- plan-only localizer FEAT-like inventory and subject-level fixed-effects
  command planning in `research-neuro`, deriving COPE/VARCOPE numbers from
  parsed FEAT contrast names;
- explicit subject-level fixed-effects execution from existing localizer FFX
  plans in `research-neuro`, with injectable command runners and per-job
  provenance under planned work directories;
- plan-only handoff from completed subject-level localizer FFX outputs into
  existing LOSO ROI fixed-effects input planning in `research-neuro`;
- plan-only MNI-space ROI mask transform planning in `research-neuro`, including
  target T1w or subject-reference inventory, ordered transform-chain previews,
  `antsApplyTransforms` availability preflight, argv rendering, and ROI QC
  preview rows;
- explicit MNI-to-T1w ROI transform execution/QC package APIs in
  `research-neuro`, running already-planned ANTs argv vectors with injectable
  runners and writing only planned transformed-mask, QC JSON, and provenance
  JSON artifacts;
- generic subject-level table inference summaries in `research-analysis`,
  including one-sample summaries, deterministic sign-flip inference,
  leave-one-subject-out sensitivity rows, missingness and duplicate QC, and
  Benjamini-Hochberg correction within configured families;
- generic publication table and manifest helpers in `research-analysis`, able
  to shape already-computed rows into display tables, machine-readable
  companion tables, and manifest JSON without recomputing analysis outputs;
- reusable visualization/report outputs in `research-viz`, able to plan and
  render generic reports, SVG point/interval figures, visual QC rows, and JSON
  manifests from already-computed rows or simple source files;
- schema-only generic analysis workflow recipe contracts in `research-core`,
  with MVPA-specific extension metadata contracts in `research-neuro`;
- BIDS-like derivative naming helpers for reusable published outputs.

## Missing Support

The remaining upstream work after the plan-only handoff is execution lifecycle
integration across ROI generation, MVPA extraction, table-based inference, and
publication stages without adding duplicate LOSO ROI logic inside MVPA. Step 6B
ROI transform execution/QC is available as package API support, but it is not
yet wired into a broader CLI or pipeline lifecycle.

The current docs and runtime boundaries still do not provide a complete
full-study workflow that inventories run-level localizers, creates
subject-level fixed-effects outputs, feeds LOSO ROI generation, extracts MVPA
patterns, runs table-based inference, and publishes final review artifacts as a
single coherent plan.

## Ownership Boundaries

- `research-core` owns only thin orchestration surfaces: config lookup, root
  resolution, command lifecycle, plan printing, and package delegation.
- `research-neuro` owns FEAT and design parsing, localizer inventory,
  subject-level FFX and LOSO planning/execution, NIfTI and ROI operations,
  transforms, ROI QC, and MVPA extraction semantics.
- `research-analysis` owns generic table-based inference, statistics, prepared
  pattern and distance tables, summary models, and adapter-neutral analysis
  contracts.
- `research-bids` owns BIDS-like naming and path helpers.
- `research-viz` owns reusable figures and reports.
- Project overlays stay limited to config and manifests for this workflow.
- `pipelines` and `ops` stay reserved for later orchestration and HPC behavior.

ROI family and catalog metadata is not required for generic subject-level
inference. ROI catalogs are metadata for grouping and reporting ROI labels, not
hard-coded MVPA logic.

## Generic Config Model

A future full-study config should describe stages, not executable project code:

- study inputs: roots, participants, sessions, runs, model output inventories,
  events, exclusions, and optional manifests;
- localizer inventory: run-level model sources, design metadata, quality checks,
  and expected subject-level outputs;
- subject-level fixed effects: inputs, transforms, masks, merge policies,
  output roots, and provenance requirements;
- ROI sources: existing masks, LOSO ROI configs, or published ROI derivatives
  produced by the reusable ROI workflows;
- MVPA sources: ROI references, pattern extraction settings, condition
  selectors, cross-validation units, and noise-normalization settings;
- inference: rectangular input tables, grouping columns, estimands, statistics,
  contrasts by configured identifiers, and multiple-comparison policy;
- outputs: runtime/cache roots, published derivative roots, QC tables, and
  provenance sidecars.

The config should reference ROI sets and catalogs by name when helpful, but
subject-level inference should operate on explicit tables and columns rather
than requiring ROI catalog semantics.

## Plan-By-Default Behavior

Full-study commands should print a JSON plan by default. Plan mode may validate
config, resolve roots, inspect available inputs, preview outputs, and summarize
missing prerequisites. It must not write outputs, mutate project overlays,
launch pipelines, run HPC jobs, create figures, or execute analysis stages.

Execution should remain explicit through a future `--execute` style path for
each implemented stage. Expensive or state-changing behavior should be
reviewable before it runs.

## Execute Behavior by Future Stage

Future execution should be added in small stages:

- Stage 1: localizer inventory and subject-level FFX plan-only previews from
  run-level FEAT outputs. Available in `research-neuro`.
- Stage 2: local subject-level FFX execution in `research-neuro`, writing
  runtime/cache outputs and provenance only. Available in `research-neuro`.
- Stage 3: hand off generated subject-level FFX outputs to existing LOSO ROI
  generation, reusing current ROI workflows instead of duplicating them.
  Plan-only handoff is available in `research-neuro`.
- Stage 3A: plan MNI-space ROI mask transforms into subject T1w or
  subject-specific reference space, with ANTs `antsApplyTransforms` preflight
  and ROI QC previews only. Available in `research-neuro`.
- Stage 3B: execute already-planned MNI-to-T1w ROI transforms with ANTs
  `antsApplyTransforms`, then run actual transformed ROI QC and optional planned
  JSON writes. Available in `research-neuro`; users still provide ANTs on
  `PATH` or as an explicit executable path. Conda or micromamba can be one setup
  option, but no local absolute paths should be committed.
- Stage 4: MVPA extraction and prepared table generation using existing
  `research-neuro` and `research-analysis` responsibilities.
- Stage 5: generic table-based inference in `research-analysis`.
- Stage 6: generic publication table/manifest writing in `research-analysis`
  is available; reusable visualization/report outputs in `research-viz` are
  available.
- Stage 7: pipeline and HPC orchestration in `pipelines` and `ops`.

## Output and Provenance Boundaries

Runtime/cache outputs should remain separate from published derivatives.
Generated subject-level FFX outputs should be explicit inputs to downstream
LOSO ROI workflows. MVPA extraction outputs should record ROI source identity,
pattern source metadata, condition selectors, cross-validation units,
noise-normalization settings, exclusions, and software versions.

Published derivatives should use `research-bids` naming helpers and should not
depend on project overlay paths. Project overlays should not be rewritten by
execution. Visualization/reporting outputs are later work and belong in
`research-viz`.

## Test Strategy

The initial ADR was docs-only. Implementation slices should add focused tests
in the owning packages:

- `research-neuro`: FEAT/design parsing, localizer inventory, subject-level FFX
  planning/execution wrappers, transforms, ROI handoff, ROI QC, and MVPA
  extraction semantics with synthetic fixtures and mocked external tools.
- `research-analysis`: table schemas, prepared rows, estimands, statistics,
  grouping behavior, and provenance columns.
- `research-bids`: path helper coverage for new published derivatives.
- `research-core`: plan-only command behavior, no-write guarantees, and
  delegation boundaries.

End-to-end tests should stay small and fixture-driven until pipeline or HPC
layers are intentionally added.

## PR-Sized Roadmap

1. Docs-only roadmap ADR and MVPA doc pointer.
2. Schema-only generic analysis workflow recipe contracts plus an MVPA
   extension contract, with no execution.
3. `research-neuro` localizer inventory schema and plan-only APIs. Available.
4. Thin `research-core` plan command that delegates inventory/FFX planning.
5. Subject-level FFX execution wrappers and provenance in `research-neuro`.
   Available.
6. Handoff from generated subject-level FFX outputs into existing LOSO ROI
   workflows. Plan-only support available.
6A. MNI-to-T1w or subject-reference ROI transform planning and ROI QC preview
    in `research-neuro`. Plan-only support available.
6B. Execute already-planned MNI-to-T1w ROI transforms and actual post-transform
    ROI QC in `research-neuro`. Available; no CLI behavior, LOSO ROI
    generation, MVPA extraction, distance computation, inference, publication,
    project overlay mutation, or pipeline/ops behavior is included.
7. MVPA extraction integration from LOSO ROI outputs into prepared tables.
   Available with config-driven runtime refinements for run exclusions,
   condition pairs, min-events/min-observations threshold sweeps, within-ROI
   mean-centering, grouping-column preservation, and runtime/cache provenance;
   this does not add ROI creation, mask transforms, inference/statistics,
   publication outputs, visualization/report outputs, project overlays, or
   pipeline/ops/HPC behavior.
8. Generic table-based inference in `research-analysis`. Step 8A generic
   subject-level inference summaries are available; this slice adds no
   publication outputs, visualization/report outputs, project overlays,
   pipeline/ops/HPC behavior, ROI creation, mask transforms, MVPA extraction, or
   MVPA distance computation.
9. Publication helpers and derivative metadata. Generic publication
   table/manifest helpers are available in `research-analysis`.
10. Reusable visualization and reports in `research-viz`. Available.
11. Optional pipeline and HPC orchestration.

## First Implementation Slice After This Docs PR

The first code slice is schema/plan-only: reusable workflow recipe contracts
live in `research-core`, while MVPA-specific extension fields live in
`research-neuro`. It adds no CLI execution, FSL/ANTs execution, NIfTI loading,
MVPA distance computation, ROI creation, mask transforms, project overlays,
dependencies, or pipeline/ops behavior.

The next implementation slice should add a thin `research-core` plan command
that delegates to the `research-neuro` localizer FFX planner and prints JSON. It
should not execute FFX, modify ROI workflows, write project overlays, add
visualization, or touch pipeline/ops behavior.
