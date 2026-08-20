# BIDS Events Builder

> **Alpha status — Runnable locally for synthetic event construction.** The
> deterministic toy-memory example provides exact plan/build commands and
> verifies event output. Separate checked-in synthetic tests cover traversal,
> anchor discovery, and publish mechanics. This advanced package-specific
> interface does not prove BOLD preprocessing or another imaging workflow. The
> default Polars path is in the minimal environment; pandas-compatible golden
> bytes use the optional pandas extra.

## Where It Lives

- reusable BIDS-specific logic: `packages/research-bids`
- study-specific event mappings and sidecar content: `project/*/config/events/`
- staged previews, manifests, sidecars, copied stimuli: `artifacts/...`
- canonical published outputs: `datasets/...`

## Current Scope

The current implementation is a reusable BIDS events builder specialized for encoding/recognition-style tasks.

Reusable package behavior includes:
- BIDS path/entity handling
- anchor discovery and inherited entities
- staged build outputs and manifest-driven publish
- deterministic sidecars
- optional stimuli staging/publish with strict source resolution and collision checks

The current row-semantics layer is narrower:
- rows are grouped by configured phase prefixes
- recognition rows derive probe type and outcome labels
- encoding rows derive later outcomes from linked recognition rows
- the current source-schema contract expects a recognition label column such as `image_old_new`
- the emitted schema assumes fields such as `acc_label`, `probe_type`, `enc_is_tested`, and `enc_later_outcome`

That is sufficient for the current encoding/recognition examples, but it is not yet a fully generic config-driven events engine for arbitrary task families. The public `toy-memory` example and its goldens are deterministically generated and contain no participant-derived data.

## Commands

```bash
python -m research_platform.bids.cli events plan --spec ... --source ... --artifact-root ...
python -m research_platform.bids.cli events build --spec ... --source ... --artifact-root ... --write-sidecars --copy-stimuli
python -m research_platform.bids.cli events publish --dataset-root ... --manifest artifacts/.../manifests/build-manifest.json
```

## Contract

- canonical events semantics live at `research_platform.neuro.events`
- `research-bids` consumes that root semantic surface as the stable BIDS facade for plan/build/publish
- `plan` resolves staged output paths and manifest entries
- `build` writes staged `_events.tsv`, optional `_events.json`, optional `stimuli/`, and a deterministic build manifest under `artifacts/...`
- `publish` copies only the staged files listed in that manifest into `datasets/...`

## Backend Compatibility

- `polars` is the default backend
- `pandas` is the compatibility backend for the established encoding/recognition numeric output
- in the current pandas path, `onset` is preserved exactly as emitted by the row builder while selected numeric columns are normalized before `DataFrame.to_csv(...)`
