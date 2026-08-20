# ADR-0012: MVPA Crossnobis Foundation

## Status

Accepted

## Context

The platform needs a reusable MVPA foundation for crossnobis distance and RDM
workflows that can support neuroimaging ROI analyses without becoming tied to
one study, one file layout, or one third-party object model.

Phase 0 is documentation-only. It should define the architecture, package
boundaries, configuration shape, output conventions, and dependency policy
before any runtime code, CLI implementation, tests, project overlays, pipelines,
or new dependencies are added.

## Decision

MVPA/crossnobis workflows will be study-agnostic and config-driven.

Generic matrix, distance, RDM, cross-validation, and statistics contracts belong
in `research-analysis`. The stable public API should expose research-platform
config, table, and data contracts. It should not expose `rsatoolbox`, PyMVPA,
Nilearn, scikit-learn, or any other third-party object model as the stable API.

The preferred v1 external distance engine for crossnobis/RDM computation is
`rsatoolbox` through an optional RSA adapter. `rsatoolbox` is not a hard
required dependency for the platform in Phase 0. A lightweight
`native_reference` implementation remains planned for tests, validation,
numerical checks, and fallback.

Package ownership is:

- `research-analysis`: adapter-neutral MVPA contracts, distance/RDM table
  schemas, cross-validation units, statistics, optional distance-engine adapter
  contracts, and native reference validation plans.
- `research-neuro`: neuro-specific NIfTI, ROI, FEAT, `design.fsf`,
  `sigmasquareds`, pattern-source discovery, and pattern extraction logic. FSL
  FEAT is the first neuro pattern-source backend, not the core architecture.
- `research-bids`: BIDS-like derivative naming helpers for MVPA pattern,
  distance/RDM, QC, provenance, figure, and report outputs.
- `research-viz`: reusable MVPA plots, report tables, and rendering primitives
  without project-specific styling.
- `research-ml`: future decoding estimators such as LDA, SVM, and nonlinear
  models. scikit-learn belongs primarily to decoding/modeling, not ownership of
  crossnobis math.
- `research-core`: future thin lifecycle CLI surfaces only, delegating MVPA
  semantics to the packages above.
- `pipelines` and `ops`: orchestration only after package behavior is stable.
- `project` overlays: thin YAML/config only.

MVPA workflows should consume existing ROI outputs by default. Data-driven ROI
definitions must avoid circular analysis and leakage; leave-one-subject-out
definitions are the expected pattern when ROIs are learned from the same study
population.

Optional adapters remain adapter-based:

- `rsatoolbox`: preferred/default v1 external distance engine for crossnobis/RDM
  computation when the optional adapter is installed.
- PyMVPA: legacy/reference MVPA adapter to evaluate cautiously, not the default
  v1 backend.
- Nilearn: future neuroimaging pattern extraction, GLM, searchlight, and
  decoding adapter candidate.
- scikit-learn: future decoding/modeling dependency under `research-ml`, not
  the owner of crossnobis math.

Optional third-party integrations should be enabled through config and optional
extras later. They should not be hard-coded imports in core contracts.

## Non-Decisions

Phase 0 does not implement:

- MVPA runtime code
- CLI commands
- package modules
- tests
- project overlay changes
- pipeline or ops changes
- dependency metadata changes
- `rsatoolbox`, PyMVPA, Nilearn, scikit-learn, or other MVPA/RSA/decoding
  library installation

## Consequences

Positive:

- The platform can define stable MVPA contracts before committing to runtime
  details.
- `rsatoolbox` can be the preferred v1 distance engine without becoming a core
  dependency or public API model.
- A native reference path keeps validation and fallback possible.
- Neuro-specific source discovery stays separate from generic distance math.
- Future decoding and visualization work has clear package ownership.

Tradeoffs:

- Users cannot run MVPA from this phase alone.
- Adapter behavior, defaults, and provenance must be specified before execution
  phases can be considered stable.
- Published derivative naming is BIDS-like and reusable, but not a guarantee of
  full BIDS-validator compliance.
