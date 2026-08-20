# Coding-agent contribution workflow

Coding agents may assist with work in this repository. Human-authored and
coding-agent-assisted contributions follow the same review requirements, and
contributors remain responsible for understanding and reviewing every change
they submit.

## Start with the public repository contract

Read the applicable `AGENTS.md` files before changing code. They document
public engineering boundaries for both humans and automated tools. Also review
`CONTRIBUTING.md` and the documentation for the area being changed.

Local instruction overrides belong in ignored `AGENTS.local.md` files. Coding-
agent state, prompts, transcripts, task logs, editor state, scratch notes, and
other machine-local coordination material must remain untracked.

## Work in bounded changes

- Keep reusable logic in `packages/` and project-specific configuration and
  light glue in `project/`.
- Decide whether data are canonical public inputs under `datasets/` or
  ephemeral outputs under `artifacts/` before adding files.
- Keep secrets, private data, machine-local settings, personal paths, site
  assumptions, and participant-derived material out of tracked content.
- Update documentation when setup, behavior, or structure changes.
- Add or update focused tests when reusable behavior changes.
- Update `ARCHITECTURE.md` when stable structural boundaries change,
  `ROADMAP.md` when public future direction changes, and an ADR when a durable
  decision changes materially.

Useful coding-agent tasks include adding validation, extracting reusable logic
from notebooks, improving public scaffolds, and maintaining tests and
documentation. Tasks that mix canonical inputs with run outputs, conceal logic
in notebooks, or introduce private assumptions should not be accepted.

## Validate before review

Generated changes are not accepted without validation. Run the focused tests
and documented checks for the affected area, verify privacy and path safety,
inspect the complete diff, and confirm the working tree contains no unexpected
files. Do not treat a coding agent's report as a substitute for reviewing the
actual change.

The contributor who submits the change is accountable for its correctness,
scientific and operational safety, documentation, and public-release hygiene.
