
# Contributing

Research Platform accepts human-authored and coding-agent-assisted changes
under the same review and safety requirements. Contributors remain responsible
for understanding, validating, and reviewing everything they submit; generated
changes are not accepted without appropriate tests, documentation, privacy
checks, and safety checks.

## Repository boundaries

- `packages/` contains reusable package APIs and implementations.
- `pipelines/` contains reusable workflow orchestration and pipeline profiles.
- `project/` contains thin configuration, manifests, notebooks, reports, and
  light project glue.
- `datasets/` contains canonical public data and derivative metadata.
- `artifacts/` contains ignored, ephemeral run outputs.
- `ops/` contains environment, synchronization, scheduler, and operational
  infrastructure.

Keep BIDS and neuroimaging behavior in `research-bids` or `research-neuro`, not
in `research-core`. Keep HPC and remote-operation behavior in `research-hpc`
and `ops/`. Within `research-io`, only the dedicated backend modules named in
`packages/research-io/AGENTS.md` may import `polars` or `pandas` directly.

The four checked-in overlays—`project-template`, `project-example`,
`project-pilot-bids`, and `project-pilot-tabular`—are public examples. Real
study overlays under `project/` are intentionally ignored and belong in a
separate private repository or another explicit private boundary. Do not
weaken the public overlay allowlist to store private work here.

## Public and local content

- Never hard-code personal paths, usernames, sites, private study names,
  credentials, or infrastructure assumptions.
- Never add participant-derived fixtures. Public fixtures must be explicitly
  synthetic and document their provenance.
- Keep secrets and private data outside tracked files.
- Keep local instruction overrides, coding-agent state, editor state, logs,
  and scratch notes untracked. Use ignored `AGENTS.local.md` for local-only
  instructions; repository `AGENTS.md` files contain the public contribution
  contract.
- Keep reusable logic out of notebooks and project overlays.

## Change requirements

- Add or update focused tests when behavior changes.
- Update the directly affected documentation in the same change.
- Confirm documented commands match implemented entry points.
- For structural changes, update `ARCHITECTURE.md` when stable boundaries
  change, update `ROADMAP.md` when public future direction changes, and add or
  revise an ADR under `docs/decisions/` when a durable decision changes.
- Prefer configuration and small reusable functions over duplicated or
  project-coupled logic.

See the
[coding-agent contribution workflow](docs/onboarding/coding-agent-workflow.md)
for a concise workflow that applies when coding agents assist with a
contribution.

## Source-checkout checks

This alpha is developed and tested from an editable source checkout. After
activating the repository environment, use the focused bootstrap checks when
changing installation or entry-point behavior:

```bash
make bootstrap-plan
make install-smoke
make test-bootstrap
make test-ci-contract
```

`make bootstrap-plan` is non-mutating. `make install-smoke` checks the populated
environment selected by `RP_DEV_VENV` (default `.venv`). `make test-bootstrap`
uses the active `python3` by default; set `PYTHON` to choose another populated
development interpreter.

The [continuous-integration guide](ops/ci/README.md) documents the exact local
equivalents for package-by-package tests, public contracts, offline validation,
release-archive construction and inspection, and clean-checkout verification.
`make test-ci-contract` checks the workflow's least-privilege static contract;
it does not claim that a hosted GitHub Actions run has passed.

The analysis package intentionally uses its module entry point in commands and
tests:

```bash
python -m research_platform.analysis.cli --help
```

Do not document an analysis console alias unless one is actually added and
tested.

## Pull request checklist

- [ ] Boundaries still make sense
- [ ] No private, participant-derived, credential, or machine-local content
- [ ] Tests added or updated where applicable
- [ ] Docs updated if conventions changed
- [ ] Documented commands match implemented entry points
- [ ] Outputs are written to the correct layer
- [ ] Generated changes were understood and reviewed by the contributor
