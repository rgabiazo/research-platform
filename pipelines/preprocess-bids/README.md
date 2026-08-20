# preprocess-bids

> **Alpha status — Experimental or external-runtime.** This is substantive
> orchestration with automated planning and adapter coverage, not a supported
> end-to-end public preprocessing workflow. The checked-in public inputs are
> insufficient for imaging execution.

This pipeline coordinates adapter-driven BIDS preprocessing. Its Snakemake
workflow loads a reviewed run manifest, delegates runtime-plan construction to
the selected adapter, fans out plan units, and aggregates completion markers.
It does not own tool-specific scientific or command semantics:

- `research-core` owns generic manifests and orchestration contracts;
- `research-neuro` owns the DeepPrep and fMRIPost-AROMA adapters, discovery,
  validation, and command rendering;
- `research-hpc` owns generic remote transfer and scheduler mechanics; and
- this directory owns workflow definitions, profiles, and thin runners.

Execution requires user BIDS data or an appropriate input derivative,
Snakemake, the selected tool or container runtime, any tool-specific assets such
as a FreeSurfer license, and site configuration for remote use. No checked-in
project supplies a complete executable imaging dataset, and no live-cluster
validation has been accepted. Adapter presence, a rendered plan, or mocked
execution evidence does not change that support boundary.

See the [capability matrix](../../docs/capabilities.md), the
[BIDS and HPC guide](../../docs/bids-hpc-slice.md), and
[ADR-0005](../../docs/decisions/ADR-0005-bids-tool-adapters.md).
