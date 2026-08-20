# research-neuro

> **Alpha status — mixed support.** The checked-in coordinate-sphere ROI,
> generic NIfTI extraction, and prepared-pattern crossnobis examples are
> **Runnable locally** within their documented synthetic contracts. Advanced
> ROI paths, image-backed MVPA, FEAT, DeepPrep, fMRIPost-AROMA, and external
> neuroimaging-tool execution remain **Experimental or external-runtime**.

`research-neuro` owns reusable neuroimaging semantics and adapters. Package
ownership identifies where behaviour belongs; it is not evidence that every
adapter or execution path is currently supported.

## Ownership and boundaries

This package owns:

- generic NIfTI utilities and neuroimaging metadata inspection;
- coordinate-sphere and other ROI contracts, mask construction, value
  extraction, readiness checks, and neuroimaging-specific runtime safety;
- neuroimaging pattern-source discovery and extraction contracts;
- the prepared-vector `materialized_pattern_table` adapter and its
  neuroimaging-specific exact-unit runtime handoff;
- DeepPrep, fMRIPost-AROMA, FSL FEAT, Featquery, and related tool-specific
  adapters and command construction;
- the current semantic implementation shared by the BIDS events facade.

It does not own:

- BIDS entity ordering or reusable derivative naming, which belong in
  `research-bids`;
- generic crossnobis/RDM mathematics, statistics, or tabular association
  methods, which belong in `research-analysis`;
- top-level workspace orchestration, which belongs in `research-core` and the
  `rp` CLI;
- SSH, rsync, SLURM, or remote-safety mechanics, which belong in
  `research-hpc` and `ops/`;
- project-specific contrasts, cohorts, ROI definitions, task mappings, or
  scientific validity decisions.

Private or study-specific configuration belongs in a private project overlay.
Do not add participant identifiers, local paths, site details, or scientific
defaults to reusable package code or public examples.

## Relationship to `rp`

The supported local examples are exposed through the integrated `rp analysis`
lifecycles. Package modules remain available for advanced Python consumers, but
there is no separate `research-neuro` console command and no promise of a
single stable high-level Python SDK.

`rp` resolves project configuration and delegates neuroimaging behaviour to
this package. It does not make an experimental adapter supported merely by
exposing a command or configuration schema.

## Install from a source checkout

The alpha packages are not published on PyPI. From the repository root, use a
supported Python 3.11 or 3.12 interpreter:

```bash
bash ops/envs/dev/bootstrap.sh --profile minimal
source .venv/bin/activate
python -c "import research_platform.neuro"
```

See the [source-checkout quickstart](../../docs/onboarding/quickstart.md) for
the complete environment and smoke-check procedure.

## Checked-in local ROI example

The `project-example` ROI path uses deterministic regular-grid NIfTI fixtures.
It builds two coordinate-sphere masks and extracts mean, median, and voxel
count from one synthetic value image:

```bash
rp analysis roi validate toy-spheres --project project-example
rp analysis roi doctor toy-spheres --project project-example
rp analysis roi build toy-spheres --project project-example
rp analysis roi build toy-spheres --project project-example --execute

rp analysis roi extraction validate toy-values --project project-example
rp analysis roi extraction doctor toy-values --project project-example
rp analysis roi extraction run toy-values --project project-example
rp analysis roi extraction run toy-values --project project-example --execute
```

The commands without `--execute` validate readiness or render a plan. Execution
writes masks, portable sidecars, extraction rows, and run-specific QC beneath
the ignored artifact root. It does not alter the input dataset or publish a
canonical derivative. Reusing an occupied configured destination fails before
changing the existing result.

The evidence verifies two nonempty seven-voxel masks on a synthetic regular
grid. It does not establish anatomical validity, atlas validity, registration
quality, or suitability for real imaging data.

## Checked-in prepared-pattern crossnobis example

The `toy-crossnobis` example consumes invented ROI-final vectors from the fixed
`materialized_pattern_table` v1 contract. It does not load images or apply ROI
masks:

```bash
rp analysis bundle validate toy-crossnobis \
  --project project-example

rp analysis bundle doctor toy-crossnobis \
  --project project-example

rp analysis bundle plan toy-crossnobis \
  --project project-example

rp analysis mvpa validate toy-crossnobis \
  --project project-example

rp analysis mvpa doctor toy-crossnobis \
  --project project-example \
  --bundle toy-crossnobis

rp analysis mvpa plan toy-crossnobis \
  --project project-example \
  --bundle toy-crossnobis

rp analysis mvpa run toy-crossnobis \
  --project project-example \
  --bundle toy-crossnobis

rp analysis mvpa run toy-crossnobis \
  --project project-example \
  --bundle toy-crossnobis \
  --execute
```

Validation, doctor, planning, and the first `run` preview are non-mutating. The
final command authorizes one failure-safe 14-file runtime transaction. The
runtime `distances.tsv` is an RDM-ready pairwise-distance table, not an exported
RDM. It is stored at `analysis/prepared-distances/distances.tsv`. RDM, figure,
report, derivative, and publication exports remain separate advanced surfaces.

The fixture contains two invented subjects, two runs, two conditions, and two
ROIs represented by 16 prepared-vector rows. This verifies bounded software
behaviour and exact-unit handling, not image-backed extraction or real-data
MVPA. FSL/image execution, real-data MVPA, HPC, RDM/report exports,
publication, and deferred adapters remain Experimental or external-runtime.

## Inputs and outputs

| Surface | Inputs | Result |
| --- | --- | --- |
| Coordinate-sphere ROI | Reviewed sphere definitions and a compatible reference NIfTI | Binary NIfTI masks and portable JSON sidecars under the configured artifact root |
| Generic NIfTI extraction | Existing masks, a value NIfTI, and configured statistics | Primary values TSV plus run-specific QC/audit output |
| Prepared-pattern MVPA | Fixed v1 ROI-final vector table, exact bundle units, condition/ROI/CV identities | Failure-safe runtime records and an RDM-ready pairwise distance table |
| Advanced ROI and image MVPA | User imaging/FEAT inputs, reviewed geometry and scientific metadata, and optional tools | Experimental planning or external-runtime behaviour only |
| Tool adapters | User-owned BIDS/derivative inputs and tool/runtime configuration | Discovery, selection, or command/runtime plans; not general execution support |

Generic crossnobis computation remains an analysis-package responsibility.
`research-neuro` owns the neuroimaging input adapter, prepared-pattern
validation, and runtime handoff around that mathematics.

## External dependencies and limitations

The package declares `nibabel>=5.0`. The repository bootstrap installs the
coordinated package closure for the bounded local examples.

FSL, ANTs, Nextflow, Apptainer or another container runtime, DeepPrep images,
fMRIPost-AROMA images, scheduler access, and real imaging inputs are not
bundled. Their presence in adapters or tests does not establish live execution,
scientific validity, or support on an arbitrary workstation or cluster.

Advanced ROI surfaces include LOSO/group-map planning and execution, FSL
Featquery, FSL/ANTs transforms, publication helpers, and real FEAT/BOLD inputs.
They remain experimental or external-runtime despite unit tests around
synthetic or mocked boundaries.

No percent-signal-change or PSC workflow is claimed as runnable in this alpha.
The existence of Featquery planning/parsing code and a
`percent_signal_change` metric does not promote PSC extraction to supported
local behaviour.

Likewise, image-backed MVPA, FEAT-derived PE discovery, real-data MVPA, HPC
execution, and deferred BIDS-derivative, Nilearn, or surface adapters are not
covered by the prepared-vector example. No SPM adapter is implemented or
claimed.

## Evidence and authority

The support boundary comes from the accepted architecture, capability matrix,
and ADRs rather than package ownership or API presence. Focused tests verify
synthetic NIfTI geometry and extraction, collision refusal, portable paths,
prepared-vector schema and source digests, exact-unit crossnobis output, and
mocked advanced adapters.

Further reading:

- [ROI workflows](../../docs/roi-workflows.md)
- [MVPA and crossnobis](../../docs/mvpa-crossnobis.md)
- [Capability matrix](../../docs/capabilities.md)
- [Architecture and package ownership](../../ARCHITECTURE.md)
- [ADR-0005: BIDS tool adapter seam](../../docs/decisions/ADR-0005-bids-tool-adapters.md)
- [ADR-0008: analysis action and BIDS analysis slice](../../docs/decisions/ADR-0008-analysis-action-and-bids-analysis-slice.md)
- [ADR-0011: reusable ROI foundations](../../docs/decisions/ADR-0011-reusable-roi-workflow-foundations.md)
- [ADR-0012: MVPA crossnobis foundation](../../docs/decisions/ADR-0012-mvpa-crossnobis-foundation.md)
- [ADR-0017: materialized MVPA pattern tables](../../docs/decisions/ADR-0017-materialized-mvpa-pattern-tables.md)
- [ADR-0018: exact-unit MVPA transactions](../../docs/decisions/ADR-0018-mvpa-exact-unit-runtime-transactions.md)
