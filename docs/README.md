# Documentation

Pick the section for your role; the
[capability and support matrix](capabilities.md) says what runs today.

## For researchers

Getting started:

- [Quickstart](onboarding/quickstart.md): install the source checkout and run
  the synthetic tabular example.
- [Bring your own data](byod.md): connect a private project overlay and an
  external data root.
- [Add a project](onboarding/add-a-project.md): create a thin project overlay.

Workflows:

- [Analysis workflows](analysis-workflows.md): the tabular workflow and
  first-level FEAT.
- [ROI workflows](roi-workflows.md): build and extract synthetic
  coordinate-sphere ROIs.
- [MVPA and crossnobis](mvpa-crossnobis.md) and its
  [command runbook](mvpa-crossnobis-command-runbook.md).
- [BIDS events builder](bids-events-builder.md): build deterministic event
  tables.

Running on a cluster:

- [HPC setup](how-to/hpc-setup.md)
- [HPC troubleshooting](how-to/hpc-troubleshooting.md)
- [DeepPrep on SLURM](how-to/run-deepprep-on-slurm.md)
- [First-level FEAT on SLURM](how-to/run-feat-first-level-on-slurm.md)

The cluster guides cover SLURM and rsync over SSH only, driven from a local
workstation. None has been validated on a live cluster.

## For developers

- [Architecture overview](../ARCHITECTURE.md)
- [Data lifecycle](architecture/data-lifecycle.md)
- [Conventions](conventions/README.md)
- [Architecture Decision Records](decisions/README.md)
- [Contributing guide](../CONTRIBUTING.md)
- [Continuous integration](../ops/ci/README.md)
- [Local development](onboarding/local-dev.md)
- [Add a package](onboarding/add-a-package.md)
- [Add a pipeline](onboarding/add-a-pipeline.md)
- [Materialized pattern table v1](materialized-pattern-table-v1.md): the
  producer contract for prepared-vector inputs.

## Reference

- [Capability and support matrix](capabilities.md)
- [Roadmap](../ROADMAP.md)
- [Changelog](../CHANGELOG.md)
- [Citation metadata](../CITATION.cff)
- [MIT License](../LICENSE)
- [Security policy](../SECURITY.md)
