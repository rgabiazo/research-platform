# Research Platform

Research Platform is a command-line toolkit for reproducible research
workflows. We built it for labs that run neuroimaging (BIDS) and tabular
statistics or machine-learning analyses and want to move them between a
workstation and an HPC cluster without rewriting them. Reusable code lives in
packages, study-specific choices live in a small project configuration, and the
`rp` command plans each step and shows the plan for review before anything
executes.

Research Platform is an unreleased source-checkout alpha, version `0.1.0a1`.
Its packages are not currently available from PyPI. The checkout supports
Python 3.11 and 3.12.

## What runs today

These checked-in synthetic examples run locally:

- tabular preprocessing, logistic classification, and evaluation in the
  `project-pilot-tabular` overlay, plus package-level table inspection, keyed
  merge, and ElasticNet regression;
- project initialization and validation;
- BIDS traversal and toy event-table construction;
- coordinate-sphere ROI construction and NIfTI value extraction on small
  synthetic images, which checks the software on a regular grid, not
  anatomical validity;
- materialized-pattern crossnobis on prepared synthetic vectors, which
  produces a pairwise-distance table and does not run an image pipeline.

Everything else either plans without executing, needs an external runtime, or
is a scaffold. The [capability and support matrix](docs/capabilities.md) lists
the status, evidence, and prerequisites of each workflow.

## Quick start

Create the minimal environment and check the installation:

```bash
bash ops/envs/dev/bootstrap.sh --profile minimal
source .venv/bin/activate
bash ops/envs/dev/smoke-check.sh
python -m pip check
rp --version
rp config validate --project project-pilot-tabular
```

Show the synthetic batch and write a plan without running it:

```bash
rp batch show \
  --project project-pilot-tabular \
  --batch toy_binary_logreg

rp run local preprocess tabular \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id quickstart-toy-preprocess \
  --dry-run
```

The plan is written under `artifacts/runs/quickstart-toy-preprocess/`. Review
it, then run the same request once:

```bash
rp run local preprocess tabular \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id quickstart-toy-preprocess \
  --execute
```

A run ID belongs to one request: the dry run and its execute form share it,
and a second run under the same ID is rejected. After a completed or failed
run, or after changing the request, choose a new ID; existing runs are kept.
The [quickstart](docs/onboarding/quickstart.md) covers installation profiles,
run identity, recovery, and evaluation. [Analysis workflows](docs/analysis-workflows.md)
describes the tabular and first-level FEAT workflows in full.

## Running on a SLURM cluster

Describe a cluster with `rp hpc setup`, check the result with
`rp hpc validate`, then run `rp hpc doctor`. Setup and validation run locally;
doctor is the first command that contacts the configured host. The
configuration is provider-neutral; the Alliance/MFA integration is optional and
needs site review. The platform supports SLURM and rsync over SSH only, driven
from a local workstation, and has not been validated on a live cluster. The
[HPC setup guide](docs/how-to/hpc-setup.md) walks through the sequence.

## Bring your own data

Private study overlays and data roots stay outside the public tree and are
reached through configuration and environment variables. See
[bring your own data](docs/byod.md) and
[add a project](docs/onboarding/add-a-project.md).

## Repository layout

| Path | Responsibility |
| --- | --- |
| `packages/` | Reusable installable code with explicit domain ownership |
| `pipelines/` | Workflow orchestration, external tools, and execution profiles |
| `project/` | Thin project overlays containing configuration, manifests, notebooks, reports, and light glue |
| `datasets/` | Canonical public inputs and reusable derivatives |
| `artifacts/` | Ephemeral run outputs, logs, checkpoints, generated tables, and figures |
| `ops/` | Development environments, CI, containers, synchronization, and remote-execution support |
| `secrets/` | Ignored local credentials and connection settings; never tracked |
| `docs/` | User, contributor, architecture, capability, and decision documentation |

## Documentation

For researchers:

- [Documentation index](docs/README.md)
- [Quickstart](docs/onboarding/quickstart.md)
- [Analysis workflows](docs/analysis-workflows.md)
- [HPC setup](docs/how-to/hpc-setup.md)
- [ROI workflows](docs/roi-workflows.md)
- [MVPA and crossnobis](docs/mvpa-crossnobis.md)

For developers:

- [Architecture](ARCHITECTURE.md)
- [Contributing](CONTRIBUTING.md)
- [Architecture Decision Records](docs/decisions/README.md)
- [Continuous integration](ops/ci/README.md)

Reference:

- [Capability and support matrix](docs/capabilities.md)
- [Roadmap](ROADMAP.md)
- [Changelog](CHANGELOG.md)

## License, citation, and security

Research Platform is available under the [MIT License](LICENSE). Use
[CITATION.cff](CITATION.cff) to cite it and [SECURITY.md](SECURITY.md) to
report a security issue.
