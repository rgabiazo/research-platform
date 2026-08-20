# research-core

`research-core` provides the integrated `rp` command, workspace and project
configuration, orchestration, and generic run-lifecycle services.

## Current status

This is an unreleased `0.1.0a1` source-checkout alpha for Python 3.11 and 3.12.
`rp` is the primary integrated user interface. The underlying Python modules are
package-level interfaces, not a stable unified Research Platform SDK.

Support is bounded by the
[capability matrix](../../docs/capabilities.md). Command presence or plan
rendering does not establish that a workflow is runnable.

## Responsibilities

`research-core` owns:

- workspace and project configuration;
- generic path and reference resolution;
- the top-level `rp` CLI;
- reviewed-plan composition and identity;
- generic manifests, provenance, and run-state concepts;
- the bounded local tabular transaction;
- thin orchestration over behaviour owned by other packages.

It does not own:

- table backend behaviour, which belongs in `research-io`;
- preprocessing, statistics, or MVPA mathematics, which belong in
  `research-analysis`;
- estimators and metrics, which belong in `research-ml`;
- BIDS or neuroimaging semantics;
- low-level SSH, transfer, scheduler, or remote-safety mechanics.

The direct `research-hpc` dependency supports CLI composition. It does not make
the complete remote lifecycle supported.

## Source-checkout use

Follow the repository [quickstart](../../docs/onboarding/quickstart.md) to build
a coordinated environment. The packages are not currently published on PyPI
and should not be treated as independently versioned public services.

After activating the source-checkout environment:

```bash
rp --version
rp config validate --project project-pilot-tabular
```

## Tested local tabular transaction

The checked-in synthetic tabular workflow is the primary runnable `rp` example:

```bash
rp batch show \
  --project project-pilot-tabular \
  --batch toy_binary_logreg

rp run local preprocess tabular \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id package-doc-example \
  --dry-run
```

The dry run reserves the run identity and writes reviewable planning material
beneath `artifacts/runs/package-doc-example/`; it does not execute the
preprocessing workflow. After reviewing the unchanged plan, the same request
can make its one allowed transition to execution:

```bash
rp run local preprocess tabular \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id package-doc-example \
  --execute
```

The transaction consumes the selected project and batch configuration plus the
checked-in synthetic feature table. A successful execution atomically publishes
this exact output set beneath the run:

- `outputs/split.json`;
- `outputs/prep.json`;
- `outputs/features.tsv`;
- `outputs/transaction-manifest.json`.

Existing, completed, failed, changed, or malformed run identities are preserved,
not overwritten or resumed. Generated run material remains noncanonical under
`artifacts/`; canonical publication requires a separate workflow-specific
policy.

## Evidence

The bounded contracts are exercised by:

- [`test_tabular_transaction_cli.py`](tests/unit/test_tabular_transaction_cli.py);
- [`test_tabular_output_transaction.py`](tests/unit/test_tabular_output_transaction.py);
- [`test_tabular_transaction_review_regressions.py`](tests/unit/test_tabular_transaction_review_regressions.py);
- [`test_cli_slice.py`](tests/unit/test_cli_slice.py);
- [`test_runtime_plan.py`](tests/unit/test_runtime_plan.py).

## Limitations

- Only the capability-matrix examples have their stated support status.
- Generic workflow recipes and many command families are validation or planning
  surfaces, not execution claims.
- Core does not add BIDS, neuroimaging, scientific-model, or HPC semantics.
- Local tabular transaction guarantees do not extend to remote execution.
- No stable high-level Python SDK, HTTP API, or supported graphical application
  is claimed.

See the [repository overview](../../README.md),
[architecture](../../ARCHITECTURE.md),
[quickstart](../../docs/onboarding/quickstart.md), and
[tabular guide](../../docs/tabular-slice.md).
