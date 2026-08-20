# ADR-0011: Reusable ROI Workflow Foundations

## Status

Accepted

## Context

The platform needs reusable ROI workflows for BIDS-like neuroimaging analyses,
including manual masks, coordinate spheres, atlas labels, functional threshold
maps, LOSO group maps, and data-driven hooks.

The first step should establish portable schemas and naming conventions without
adding neuroimaging runtime behavior. In particular, the foundation must not
require FSL, nibabel, nilearn, FreeSurfer, ANTs, image IO, peak finding, mask
creation, or HPC execution changes.

## Decision

We added Phase 1 ROI foundations:

- ROI set, ROI definition, extraction set, extraction target, and sidecar
  provenance validation in `research-neuro`
- BIDS-like ROI mask, sidecar, LOSO/group-map, and extraction table path helpers
  in `research-bids`
- thin `rp analysis roi ...` lifecycle commands in `research-core` that locate
  project YAML and delegate validation
- generic documentation and examples under `docs/roi-workflows.md`

The default ROI mask name uses `label-<roiLabel>` plus
`desc-<methodContrast>` rather than desc-only ROI naming.

## Non-Decisions

Phase 1 does not implement:

- LOSO execution
- FSL FLAME1
- featquery
- NIfTI image math or image IO
- nibabel or nilearn runtime behavior
- peak finding or mask creation
- subject-level FFX or group FEAT
- HPC staging, SLURM runtime changes, or real ROI generation

Backend names such as `fsl_featquery`, `fsl_flame1`, `nilearn`,
`freesurfer`, and `ants` are schema values only at this stage.

## Consequences

Positive:

- ROI configs can be reviewed and tested before execution semantics are added.
- FSL-related ROI plans can be represented without requiring FSL locally.
- BIDS-like derivative names are centralized and reusable.
- Future ROI execution can build on validated contracts instead of inventing
  study-specific YAML shapes.

Tradeoffs:

- The path helpers are BIDS-like derivative conventions and should not be
  treated as a full BIDS-validator compliance guarantee.
- Project overlays can carry ROI YAML, but execution-ready ROI stages remain
  intentionally deferred.

## Phase 2 Clarification

Phase 2 adds generic NIfTI ROI foundations while preserving the Phase 1 package
boundaries:

- `research-neuro` owns nibabel-backed NIfTI IO, geometry conversion,
  coordinate-sphere masks, manual-mask validation/copying, functional-map peak
  selection from existing maps, sidecar writing, voxel-count QC, and generic
  extraction metrics.
- `research-bids` remains the owner of BIDS-like ROI mask, sidecar, and
  extraction-table path naming.
- `research-core` remains orchestration-only; Phase 2 does not add analysis-BIDS
  runtime stages or broad CLI execution wiring.

Phase 2 deliberately remains non-FSL. It does not run FLAME1, call featquery,
estimate subject-level FFX, run group FEAT, fetch atlases, add SLURM/HPC staging
changes, or depend on Nilearn, ANTs, FreeSurfer, FSL, surface, or CIFTI tooling.

## Phase 3 Clarification

Phase 3 adds local, user-facing execution wiring for the generic NIfTI subset
without changing the package boundaries:

- `research-neuro` owns config-driven local builds for `coordinate_sphere`,
  `manual_mask`, and `functional_threshold_map`, plus `generic_nifti`
  extraction table execution.
- `research-bids` remains the owner of BIDS-like ROI mask, sidecar, and
  extraction-table path naming.
- `research-core` remains thin: it resolves project config locations and named
  roots, prints plans by default, and delegates execution only when users pass
  `--execute`.

Phase 3 remains non-FSL and local-only. It still does not run LOSO group maps,
FLAME1, featquery, subject-level FFX, group FEAT, atlas fetching, analysis-bids
runtime stages, SLURM/HPC staging, or remote execution.

## Phase 4 Clarification

Phase 4 adds local FSL-backed LOSO group-map and ROI mask execution while
preserving the same package boundaries:

- `research-neuro` owns subject fixed-effects input discovery, held-out subject
  exclusion, one-sample FLAME1 command planning/execution wrappers, LOSO group
  map caching, peak selection, mask intersection, and sidecar provenance.
- `research-bids` remains the owner of BIDS-like LOSO zstat and ROI mask path
  naming.
- `research-core` remains thin: `rp analysis roi build <roi-set>` still
  resolves project config and delegates plan or execute behavior to
  `research-neuro`.

Phase 4 assumes subject-level fixed-effects COPE/VARCOPE/mask inputs already
exist. It deliberately does not implement FSL featquery, percent signal change
extraction, subject-level fixed-effects generation, general group FEAT
orchestration, analysis-bids runtime stages, SLURM/HPC staging, or remote
execution.

## Phase 5 Clarification

Phase 5 adds local FSL `featquery` extraction while preserving the same package
boundaries:

- `research-neuro` owns featquery command planning, local execution wrappers,
  report parsing, extraction row assembly, and QC/warning behavior.
- `research-bids` remains the owner of BIDS-like ROI extraction summary table
  path helpers.
- `research-core` remains thin: `rp analysis roi extraction run
  <extraction-set>` resolves project config and delegates plan or execute
  behavior to `research-neuro`.

Phase 5 consumes existing FEAT or fixed-effects COPE-style directories and
existing ROI masks, including LOSO masks from Phase 4. It deliberately does not
implement subject-level fixed-effects generation, general group FEAT
orchestration, LOSO FLAME1 changes beyond using existing masks,
analysis-bids runtime stages, SLURM/HPC staging, or remote execution.

## Phase 6 Clarification

Phase 6 adds automatic publication of LOSO FLAME1 ROI outputs into a polished
BIDS-like derivative layout while preserving the existing runtime/cache layout
and command surface:

- `research-neuro` owns internal publication helpers that hardlink or copy
  executed LOSO group maps, ROI masks, featquery tables, and JSON sidecars.
- `research-bids` owns the BIDS-like published path builders for `maps/`,
  `masks/`, and `tables/`.
- `research-core` keeps the same ROI commands and does not add a public
  `publish` command or publication-specific flags.

Publication is enabled by generated YAML defaults for LOSO FLAME1 ROI and FSL
featquery templates. Existing configs without `publication:` remain
backward-compatible and do not publish automatically.
New LOSO FLAME1 ROI and FSL featquery templates keep runtime/cache outputs
separate from the published derivative view by defaulting them under
`dataset_derivatives_root/.research-platform/roi-loso-flame1-runtime`, while
publication remains rooted at `dataset_derivatives_root/roi-loso-flame1`.
New FSL featquery templates also declare `roi_mask_source.source:
roi_set_publication`, so extraction reads masks from the referenced ROI set's
published `masks/` directory. Absence of `roi_mask_source` preserves the legacy
runtime-mask lookup, and explicit `roi_masks` still override referenced ROI-set
lookup.

Runtime cleanup is YAML-configured only. New LOSO FLAME1 scaffolds opt into
workflow-scoped cleanup after successful ROI publication with
`after_roi_build: roi_runtime`, which removes only the current ROI set's
`.cache/loso_groupmaps/`, `loso_groupmaps/`, and `rois/` subdirectories.
Featquery cleanup removes only `roi_extract/<extraction_set>/`. Existing configs
without `runtime.cleanup` remain backward-compatible and do not delete runtime
files.
