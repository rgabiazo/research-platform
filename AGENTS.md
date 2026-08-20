
# Instructions for coding agents

This repository is a scaffold for a federated research platform. Respect the boundaries below.

## Golden rules

1. Do not hard-code absolute paths, usernames, cluster names, or study names.
2. Put reusable logic in `packages/`, not in notebooks and not in project overlays.
3. Keep `project/` repos thin: config, manifests, notebooks, reports, and light glue only.
4. Treat `datasets/` as canonical data and `artifacts/` as ephemeral outputs.
5. Never place secrets in tracked files. Use `secrets/` locally and examples elsewhere.
6. Keep BIDS-specific logic in `research-bids` or `research-neuro`, not in `research-core`.
7. Keep HPC and remote execution logic in `research-hpc` and `ops/`.
8. If adding code, add or update tests and docs in the same change.
9. Prefer small, composable functions and configuration-driven behavior.
10. Avoid silent coupling across repos.

## Where changes belong

- Package API or reusable helper: `packages/`
- Workflow orchestration, job templates, profiles: `pipelines/` or `ops/`
- Dataset metadata or reusable derivative metadata: `datasets/`
- Run outputs: `artifacts/`
- Local-only credentials: `secrets/`
- Project-specific cohort/model/report configuration: `project/`

## Preferred implementation style

- Use typed, well-named functions
- Separate IO, transforms, and orchestration
- Use environment variables and config files instead of hard-coded literals
- Keep notebooks as consumers of package APIs
- Document assumptions in markdown when structure changes

## Before making large structural changes

- Update `ARCHITECTURE.md` when stable structural boundaries change
- Update `ROADMAP.md` when public future direction changes
- Add or update an ADR under `docs/decisions/` when a durable decision changes
- Make sure the new structure still supports BIDS and non-BIDS use cases
