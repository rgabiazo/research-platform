# ADR-0008: Analysis Action And BIDS Analysis Slice

## Status

Accepted

## Context

The platform already had:

- `preprocess bids`
- tabular preprocess/train/evaluate flows
- generic run manifests
- runtime-plan helpers
- generic HPC stage and submit helpers

We needed to add first-level FSL FEAT in a way that:

- does not overload `preprocess bids`
- keeps `research-core` generic to analysis
- keeps FEAT logic in `research-neuro`
- preserves the current run-manifest and SLURM flow
- remains reusable for future FSL and non-FSL analysis tools

## Decision

We added:

- `analysis` as a first-class action in `research-core`
- a phase 1 `analysis bids` target
- a generic `BidsAnalysisToolAdapter` seam in `research-core`
- a generic `pipelines/analysis-bids` orchestration repo
- a nested FSL namespace in `research-neuro`
- first-level FEAT as the first `analysis bids` tool
- generic external analysis input roots under `analysis.external_input_roots`
- reusable `analysis.inputs.*.root_ref` support for named external analysis roots
- analysis-scoped external root reuse of the existing generic `data_roots` / sync / verify / provision path

We did not:

- add FEAT-specific orchestration logic to `research-core`
- overload `preprocess bids`
- add higher-level FEAT or featquery in phase 1
- enable published-derivative fallback by default

## Consequences

Positive:

- FEAT can use the same local and HPC planning path as the existing platform
- external EV/confound-style inputs can be staged for HPC without manual rsync and without FEAT-specific core hooks
- future FSL tools can reuse `compute.tool_profiles.fsl`
- future analysis tools can reuse the `analysis` action without changing the public CLI shape

Tradeoffs:

- projects now may carry both `config/preprocessing.yaml` and `config/analysis.yaml`
- validation and bundle loading need to tolerate multiple action configs
- local validation and SLURM planning now intentionally have different strictness around unresolved remote env vars for analysis external roots
- phase 1 keeps the analysis slice intentionally narrow until higher-level contracts are proven
