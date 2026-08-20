# analysis-bids

> **Alpha status — Experimental or external-runtime.** This pipeline has real
> planning and orchestration code, but the public repository does not contain
> an executable end-to-end BIDS analysis dataset or accepted live-runtime
> evidence.

This directory provides generic BIDS analysis orchestration. The current
implemented path is first-level FSL FEAT: the pipeline loads a reviewed run
manifest, asks the configured adapter to build a runtime plan, fans out plan
units, and aggregates completion markers.

Ownership remains separated:

- `research-core` owns the generic `analysis` action, manifests, and CLI;
- `research-neuro` owns FEAT selection, validation, FSF authoring, and command
  rendering;
- `research-hpc` owns generic remote transfer and scheduler mechanics; and
- this directory owns the Snakemake workflow, profiles, and thin runner.

Actual execution requires user BOLD data, the configured EV inputs and any
required confounds, Snakemake, FSL or a reviewed container runtime, and site
configuration for SLURM use. The public BIDS fixture contains metadata and
placeholders rather than those executable inputs. Planning and automated
adapter tests therefore do not establish supported local or remote FEAT
execution.

See the [capability matrix](../../docs/capabilities.md), the
[BIDS analysis guide](../../docs/bids-analysis-slice.md), and
[ADR-0008](../../docs/decisions/ADR-0008-analysis-action-and-bids-analysis-slice.md).
