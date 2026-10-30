
# Contributing

Every change goes through the same review and safety requirements, however it
was produced. Contributors remain responsible for understanding, validating,
and reviewing everything they submit; generated changes are not accepted
without appropriate tests, documentation, privacy checks, and safety checks.

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
and `ops/`. Within `research-io`, only the dedicated backend modules
`dataframe/polars_ops.py` and `dataframe/pandas_ops.py` import `polars` or
`pandas` directly.

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
- Keep local tool configuration, editor state, logs, and scratch notes
  untracked.
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

## Repository conventions

1. Do not hard-code absolute paths, usernames, cluster names, or study names.
2. Put reusable logic in `packages/`, not in notebooks and not in project
   overlays.
3. Keep `project/` repositories thin: configuration, manifests, notebooks,
   reports, and light glue only.
4. Treat `datasets/` as canonical data and `artifacts/` as ephemeral outputs.
5. Never place secrets in tracked files. Use `secrets/` locally and examples
   elsewhere.
6. Keep BIDS-specific logic in `research-bids` or `research-neuro`, not in
   `research-core`.
7. Keep HPC and remote execution logic in `research-hpc` and `ops/`.
8. When adding code, add or update tests and documentation in the same change.
9. Prefer small, composable functions and configuration-driven behavior.
10. Avoid silent coupling across repositories.

### Where changes belong

- Package API or reusable helper: `packages/`
- Workflow orchestration, job templates, profiles: `pipelines/` or `ops/`
- Dataset metadata or reusable derivative metadata: `datasets/`
- Run outputs: `artifacts/`
- Local-only credentials: `secrets/`
- Project-specific cohort, model, and report configuration: `project/`

### Implementation style

- Use typed, well-named functions.
- Separate IO, transforms, and orchestration.
- Use environment variables and configuration files instead of hard-coded
  literals.
- Keep notebooks as consumers of package APIs.
- Document assumptions in Markdown when structure changes.

### Before making large structural changes

- Update `ARCHITECTURE.md` when stable structural boundaries change.
- Update `ROADMAP.md` when public future direction changes.
- Add or update an ADR under `docs/decisions/` when a durable decision changes.
- Make sure the new structure still supports BIDS and non-BIDS use cases.

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
