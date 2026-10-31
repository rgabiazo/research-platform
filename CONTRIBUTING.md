# Contributing

Every change goes through the same review and safety requirements, however it
was produced. Contributors remain responsible for understanding, validating,
and reviewing everything they submit; generated changes are not accepted
without appropriate tests, documentation, privacy checks, and safety checks.

## Package boundaries

Where code, configuration, data and outputs belong is described in
[ARCHITECTURE.md, Package boundaries](ARCHITECTURE.md#package-boundaries).

## Public and local content

- Never hard-code absolute or personal paths, usernames, cluster names,
  sites, private study names, credentials, or infrastructure assumptions.
- Never add participant-derived fixtures. Public fixtures must be explicitly
  synthetic and document their provenance.
- Keep secrets and private data outside tracked files. Use `secrets/` locally
  and `*.example` files elsewhere.
- Keep local tool configuration, editor state, logs, and scratch notes
  untracked.
- Keep reusable logic out of notebooks and project overlays.

The four checked-in overlays—`project-template`, `project-example`,
`project-pilot-bids`, and `project-pilot-tabular`—are public examples. Real
study overlays under `project/` are intentionally ignored and belong in a
separate private repository or another explicit private boundary. Do not
weaken the public overlay allowlist to store private work here.

## Change requirements

- When adding code or changing behavior, add or update focused tests and the
  directly affected documentation in the same change.
- Confirm documented commands match implemented entry points.
- Prefer small, composable functions and configuration-driven behavior over
  duplicated or project-coupled logic.

## Implementation style

- Use typed, well-named functions.
- Separate IO, transforms, and orchestration.
- Use environment variables and configuration files instead of hard-coded
  literals.
- Keep notebooks as consumers of package APIs.
- Document assumptions in Markdown when structure changes.

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
