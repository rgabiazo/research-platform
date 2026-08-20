# Research Platform

Research Platform is a modular Python workspace for reproducible tabular and
BIDS-oriented research workflows. It separates reusable code, workflow
orchestration, project configuration, canonical data, and run-specific outputs
so that research methods can be reused without embedding study assumptions or
machine-specific paths.

## Development status

Research Platform is an unreleased source-checkout alpha at version `0.1.0a1`.
Its packages are not currently available from PyPI, and it is not
production-ready. The source checkout supports Python 3.11 and 3.12.

The [capability and support matrix](docs/capabilities.md) is the detailed
reference for current workflow status, evidence, prerequisites, and
limitations. A directory or command may exist even when its workflow is
limited to planning, an external runtime, or scaffolding.

## Why this project exists

Research code often mixes reusable methods with study configuration, local
paths, scheduler details, and notebook state. That makes a workflow difficult
to review, reproduce, or move to another project.

Research Platform is for researchers, research software engineers, and
contributors who want a clearer separation between:

- reusable package code;
- workflow orchestration and external tools;
- project-level scientific choices;
- canonical datasets and derivatives;
- generated run outputs;
- local credentials and infrastructure settings.

The aim is a workspace that supports both tabular and BIDS-oriented research
without treating one study, machine, or execution environment as the default.

## What works today

The following features are **Runnable locally** from the source checkout. The
scientific examples use checked-in synthetic inputs:

- the `project-pilot-tabular` workflow for numeric preprocessing, logistic
  classification, and evaluation;
- package-level tabular inspection, keyed merge, and continuous-target
  ElasticNet regression;
- project initialization and validation;
- deterministic BIDS traversal and toy event construction;
- coordinate-sphere ROI construction and generic NIfTI value extraction on
  small synthetic images;
- materialized-pattern crossnobis analysis on prepared synthetic vectors.

The ROI example verifies regular-grid software behavior, not anatomical or
real-study validity. The crossnobis example consumes invented ROI-final vectors
and produces an RDM-ready pairwise-distance table; it does not execute an image
pipeline or export an RDM.

## What remains limited

The repository also contains useful configuration, planning, and implementation
work that is not a supported end-to-end local workflow:

- The checked-in `project-pilot-bids` workflow, generic run planning, and SLURM
  rendering are **Plan/validation only**.
- SSH-backed checks, execute-mode transfers, submission and retrieval, live
  scheduler status, DeepPrep, fMRIPost-AROMA, first-level FSL FEAT, advanced
  ROI paths, image-backed MVPA, and remote notebooks are **Experimental or
  external-runtime**. They require user data, external software, credentials,
  and reviewed infrastructure.
- The complete transactional HPC/fMRIPost lifecycle is under active development
  and validation. It is not yet an implemented or accepted end-to-end public
  workflow, has not been validated on a live cluster, and is not supported or
  released.
- The checked-in apps, notebook directories, atlas/custom ROI families, and six
  placeholder pipelines are **Scaffold only**.

Research Platform does not currently claim arbitrary pipeline support,
compatibility with arbitrary clusters, or a complete raw-BIDS preprocessing
chain. See the [capability matrix](docs/capabilities.md) before relying on any
workflow beyond the checked-in local examples.

For HPC configuration, use the provider-neutral sequence `rp hpc setup`,
`rp hpc validate`, then `rp hpc doctor`. Setup and validation are local;
doctor is the first step that may contact the configured host. The Alliance/MFA
integration is optional and requires separate site review.

## Try the synthetic tabular example

From a fresh source checkout, create the minimal editable environment and check
the installation:

```bash
bash ops/envs/dev/bootstrap.sh --profile minimal
source .venv/bin/activate
bash ops/envs/dev/smoke-check.sh
python -m pip check
rp --version
rp config validate --project project-pilot-tabular
```

Inspect the synthetic batch and create a local plan:

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

The plan is written beneath `artifacts/runs/quickstart-toy-preprocess/` without
executing the workflow. Review it, then authorize that same request once:

```bash
rp run local preprocess tabular \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id quickstart-toy-preprocess \
  --execute
```

The documented local tabular transaction publishes its validated output set
together and rejects reuse of an existing run ID. Executing an unchanged
reviewed plan uses its planned run ID. After a failed or completed execution,
or if the reviewed request changes, preserve the existing run and choose a new
ID. The [full quickstart](docs/onboarding/quickstart.md) explains
installation profiles, plan identity, recovery, and evaluation; the
[tabular guide](docs/tabular-slice.md) contains the complete integrity
contract.

`rp` offers guided setup for common tasks while keeping the lower-level
commands available when you need more control:

```bash
rp setup
rp onboard
rp onboard tabular
rp onboard preprocess
rp onboard analysis
rp onboard notebook
rp onboard custom
```

Guided setup does not change a workflow's support status. Check the capability
matrix before choosing a workflow family.

## Repository organization

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
| `apps/` | Scaffold-only extension points in this alpha |

The checkout includes four public project overlays: `project-template`,
`project-example`, `project-pilot-bids`, and `project-pilot-tabular`. They are
templates or synthetic examples. Real study configurations belong in a
separate private repository or another explicit private boundary, not in the
public `project/` tree.

`datasets/` and `artifacts/` have different roles: datasets are canonical
inputs or reusable derivatives, while artifacts are generated for a particular
run and may be discarded or regenerated.

## Engineering approach

- [Architecture](ARCHITECTURE.md) assigns ownership across packages,
  orchestration, projects, data, outputs, and operations.
- The documented local tabular workflow uses a reviewable plan, explicit
  execution, provenance, and collision-safe publication.
- The local ROI and materialized-pattern MVPA examples are deliberately narrow
  and document the limits of their scientific interpretation.
- The [capability matrix](docs/capabilities.md) separates implementation
  evidence from public support claims.
- [Architecture Decision Records](docs/decisions/README.md) preserve reviewed
  technical decisions, while the [CI guide](ops/ci/README.md) explains package,
  public-example, and clean-checkout checks.

These local guarantees do not automatically extend to external tools, remote
storage, schedulers, or live neuroimaging workflows.

## Documentation

- [Documentation index](docs/README.md) — choose a path by task and audience.
- [Source-checkout quickstart](docs/onboarding/quickstart.md) — install and run
  the primary synthetic tabular walkthrough.
- [Bring your own data](docs/byod.md) — connect private overlays and external
  data without placing them in the public repository.
- [Capability and support matrix](docs/capabilities.md) — check current status,
  prerequisites, evidence, and limitations.
- [BIDS and HPC planning](docs/bids-hpc-slice.md) — configure and review
  plan-only or external-runtime paths.
- [ROI workflows](docs/roi-workflows.md) and
  [MVPA and crossnobis](docs/mvpa-crossnobis.md) — use the checked-in local
  examples and understand the advanced boundaries.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change. Keep reusable
logic in the owning package, update focused tests and documentation together,
and keep private, participant-derived, credential, and machine-local material
out of tracked files. [AGENTS.md](AGENTS.md) contains additional instructions
for contributors and automated coding tools.

## Development and releases

This is a pre-release development repository, not a finished product. Near-term
work focuses on polishing the source-checkout experience and synthetic examples
and completing HPC safety work before remote execution is presented as
supported.

The [changelog](CHANGELOG.md) records completed changes, while the
[capability matrix](docs/capabilities.md) tracks current support and
limitations. The [roadmap](ROADMAP.md) describes future direction without
changing current capability classifications. Version `0.1.0a1` identifies the
coordinated source package version; it is not a PyPI publication or a
production release.

## License, citation, and security

Research Platform is available under the [MIT License](LICENSE). Use
[CITATION.cff](CITATION.cff) for citation metadata and
[SECURITY.md](SECURITY.md) for security-reporting guidance.
