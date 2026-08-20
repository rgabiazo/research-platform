
# ADR-0002: Separate canonical data from ephemeral outputs

## Status
Accepted

## Decision
Keep reusable dataset-level derivatives in `datasets/` and run-specific outputs in `artifacts/`.

## Consequences
- Cleaner data lifecycle
- Easier reproducibility and auditing
- Fewer accidental commits of transient material
