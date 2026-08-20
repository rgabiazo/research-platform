# research-bids

> **Alpha status — bounded local support.** BIDS traversal, anchor discovery,
> portable naming, and deterministic synthetic event construction are
> **Runnable locally**. This package is not a universal BIDS validator and does
> not make raw-BIDS preprocessing or neuroimaging-tool execution supported.

`research-bids` provides the reusable BIDS-facing contracts in Research
Platform. It turns configured entities and source event rows into reviewable
paths, staged artifacts, and manifest-governed dataset publication without
embedding project-specific scientific choices in package code.

## Ownership and boundaries

This package owns:

- parsing and canonical ordering for the supported BIDS entities `sub`, `ses`,
  `task`, `acq`, `ce`, `dir`, `rec`, and `run`;
- caller-rooted, portable BIDS-like paths for events and reusable ROI/MVPA
  derivative names;
- raw/derivative traversal, anchor matching, and inherited `acq` and `dir`
  entities;
- BIDS events `plan`, `build`, and manifest-driven `publish` mechanics;
- deterministic TSV and JSON sidecars, optional stimuli staging, and collision
  checks.

It does not own:

- universal BIDS validation or arbitrary BIDS extension semantics;
- raw-BIDS preprocessing, NIfTI processing, FEAT, DeepPrep, fMRIPost-AROMA, or
  another neuroimaging-tool runtime;
- generic table cleaning, leakage-safe preprocessing, statistics, or
  crossnobis mathematics;
- SSH, scheduler, or remote-execution behaviour;
- project-specific task mappings, trial interpretation, sidecar content, or
  stimuli policy.

Generic table behaviour belongs in `research-io`, analysis behaviour in
`research-analysis`, neuroimaging adapters in `research-neuro`, and remote
mechanics in `research-hpc`. Project-specific event choices belong in a private
overlay or under `project/*/config/events/`, not in reusable package defaults.

Ownership of a BIDS-like name does not guarantee that a derivative satisfies
every BIDS validator or extension rule. Naming helpers validate their supported
components and keep returned paths below a caller-supplied root; they do not
inspect image content or confer scientific validity.

## Relationship to `rp`

The top-level `rp` command is the integrated workspace interface.
`research-bids` is the lower-level package command for the advanced events
surface. The BIDS facade owns pathing, anchors, manifests, writers, stimuli,
and publication. It currently imports configured row-building semantics from
`research_platform.neuro.events`; this is transitional dependency debt, not
the intended ownership direction.

## Install from a source checkout

The alpha packages are not published on PyPI. From the repository root, use a
supported Python 3.11 or 3.12 interpreter and the repository bootstrap:

```bash
bash ops/envs/dev/bootstrap.sh --profile minimal
source .venv/bin/activate
research-bids --help
```

See the [source-checkout quickstart](../../docs/onboarding/quickstart.md) for
environment and verification details.

## Checked-in synthetic events example

The `toy-memory` fixture contains invented rows and identifiers. From the
repository root, preview its three event-table outputs without writing, then
stage them beneath the ignored artifact boundary:

```bash
research-bids events plan \
  --spec project/project-example/config/events/toy-memory.v2.yaml \
  --source packages/research-bids/tests/fixtures/toy-memory/raw/toy01_visit01_toymemory_2099-01-01.csv \
  --artifact-root artifacts/bids-events/toy-memory

research-bids events build \
  --spec project/project-example/config/events/toy-memory.v2.yaml \
  --source packages/research-bids/tests/fixtures/toy-memory/raw/toy01_visit01_toymemory_2099-01-01.csv \
  --artifact-root artifacts/bids-events/toy-memory
```

`plan` returns a JSON preview and writes nothing. `build` writes staged
`_events.tsv` files and `manifests/build-manifest.json`. Sidecars and stimuli
are added only when their configuration enables them and the corresponding
flags are supplied. `publish` is a separate, explicit operation that copies
only manifest-listed files into a caller-selected dataset root; it refuses
existing destinations unless overwrite is explicitly authorized.

## Inputs and outputs

| Surface | Inputs | Result |
| --- | --- | --- |
| Entity and naming helpers | Supported entity mappings, suffixes, and a caller-owned root | Ordered names or paths; no discovery, image loading, or analysis |
| Traversal and anchor matching | A dataset root plus configured entity constraints | Deterministic candidate/anchor resolution, including supported inherited entities |
| `events plan` | Event specification, source table, artifact root, and optional dataset anchors | In-memory/JSON output and manifest preview; no files written |
| `events build` | The reviewed plan inputs | Staged TSVs, optional JSON/stimuli, and a build manifest under `artifacts/...` |
| `events publish` | Dataset root and completed build manifest | Only the manifest-listed files under `datasets/...` or another caller-owned dataset root |

The current event semantics support the checked-in generic cue examples and
the encoding/recognition-style toy contract. They do not establish arbitrary
task-family support. A task whose row semantics do not fit those contracts
needs a reviewed semantic extension, not only a new configuration file.

## Dependencies and evidence

The declared runtime dependency currently points to the coordinated
`research-neuro` package because of that shared event implementation. The
intended direction allows Neuro to depend on BIDS-owned entities, portable
paths, and naming contracts; BIDS must not depend on Neuro-specific types or
scientific semantics. The present dependency is tracked implementation debt,
not a claim that the source graph already satisfies that rule.
The default `polars` event backend is part of the minimal environment and does
not require the pandas extra. The optional `pandas` extra exists only for
pandas-compatible numeric serialization and established golden bytes.

The public evidence includes deterministic toy-memory goldens, generic cue
fixtures, traversal and anchor tests, publish and stimuli-collision tests, and
ROI/MVPA path-safety tests. These verify software behaviour on synthetic
inputs, not imaging preprocessing, real-study validity, or universal BIDS
compliance.

Further reading:

- [BIDS events builder](../../docs/bids-events-builder.md)
- [Capability matrix](../../docs/capabilities.md)
- [Architecture and package ownership](../../ARCHITECTURE.md)
- [ADR-0004: BIDS events builder boundaries](../../docs/decisions/ADR-0004-bids-events-builder.md)
- [ADR-0012: MVPA crossnobis foundation](../../docs/decisions/ADR-0012-mvpa-crossnobis-foundation.md)
