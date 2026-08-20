# ADR-0005: BIDS Tool Adapter Seam

## Status

Accepted

## Context

The BIDS preprocessing slice needs a beginner-friendly discovery and submit flow for fMRIPost-AROMA without pushing tool-specific logic into `research-core`.

The previous slice hard-coded:

- the tool name
- supported derivatives
- Snakemake rule and target metadata
- neuro package sync payloads
- derivative filename assumptions

That violated the intended repo boundaries and made `research-core` responsible for behavior that belongs in `research-neuro`.

## Decision

- Keep `research-core` orchestration-only and load BIDS tool behavior from `preprocessing.tool_adapter`.
- Define a small adapter contract for:
  - project validation
  - batch row discovery
  - runtime metadata
  - sync payload declarations
  - publish-back scaffold suggestions
- Implement the approved fMRIPost-AROMA adapter in `packages/research-neuro`.
- Keep fMRIPost-AROMA-specific discovery, row selectors, runtime option mapping, and publish-back suggestions in `research-neuro`.
- Keep stage, pull, and submit execution helpers in `packages/research-hpc`.
- Keep `rp run slurm ...` planning-only.
- Keep publish-back execution out of scope in this first pass.

## Consequences

- `research-core` no longer hard-codes tool names, rule names, supported derivatives, or derivative filename predicates for the BIDS slice.
- Beginner-facing commands such as `rp batch discover bids` and `rp run submit preprocess bids` can stay generic while still using tool-specific behavior.
- New BIDS tools can be added by implementing the adapter contract rather than editing orchestration code.
- The repo boundaries stay explicit:
  - `research-core`: manifests, orchestration, generic wrappers
  - `research-neuro`: BIDS/fMRIPost-AROMA logic
  - `research-hpc`: SSH/rsync/SLURM execution helpers

## 2026-05 Update: Raw-BIDS Preprocessing Adapters

DeepPrep adds the first raw-BIDS preprocessing adapter. The adapter contract now lets a tool declare whether it
requires an input derivative. Derivative-backed tools such as fMRIPost-AROMA still resolve and verify
`dataset.input_derivative_root`; raw-BIDS tools such as DeepPrep use the raw BIDS root directly.

The `preprocess-bids` Snakemake workflow now uses generic `bids_preprocess_*` rules and delegates runtime-plan
construction to the configured adapter. This keeps subject fan-out, container preparation, SLURM resources,
verification, sync, status, and pull behavior shared across preprocessing tools while leaving command rendering in
`research-neuro`.

DeepPrep pins `docker://pbfslab/deepprep:25.1.0` by default for reproducibility and makes the image configurable
through `compute.tool_profiles.deepprep`. Projects can override the image to `24.1.2` for CBRAIN parity or to a
site-specific CUDA/Apptainer image without adding CLI flags.
