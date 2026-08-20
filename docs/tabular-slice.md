# Minimal Tabular Slice

> **Alpha status — Runnable locally.** `project-pilot-tabular` is the primary
> public walkthrough. CI verifies the supported source-checkout contract on
> Ubuntu 24.04 with Python 3.11 and 3.12 and on macOS 15 ARM64 with Python 3.12.
> Modeling and regression also have executable automated coverage; optional
> XGBoost and polished reporting are outside the verified minimal path.

This repository now includes a smallest safe non-BIDS/tabular execution slice for one concrete path:

- project overlay: `project/project-pilot-tabular`
- canonical feature root: `datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/`
- batch manifests: `project/project-pilot-tabular/manifests/batches/*.tsv`
- preprocessing/model path: split manifest, numeric standardization, logistic regression

For public tabular preprocess, train, and evaluate workflows, `--batch` must
name a batch manifest containing exactly one data row. These commands do not
silently select the first row, iterate rows, or create Cartesian expansions;
split distinct intentions into named one-row batch manifests and select the
intended one with `--batch`. The referenced feature table may contain many data
records. This admission rule does not apply to BIDS batches, analysis-unit
bundles or cohorts, or configuration-owned `analysis tabular` runs.

The shared analysis package now also supports a minimal continuous-target regression path:

- split strategies: `random`, `stratified_binary`, `stratified_binned`
- regression commands: `regression train`, `regression evaluate`
- initial regression model kinds: `elastic_net_regression`, optional `xgboost_regression`

The workspace CLI stays orchestration-only:

- `rp run plan preprocess tabular`
- `rp run local preprocess tabular`
- `rp run slurm preprocess tabular`
- `rp run plan train model`
- `rp run local train model`
- `rp run slurm train model`
- `rp run plan evaluate model`
- `rp run local evaluate model`
- `rp run slurm evaluate model`
- `rp hpc stage --run-id <run_id>`
- `rp hpc status --run-id <run_id>`
- `rp hpc pull --run-id <run_id>`
- `rp hpc cancel --run-id <run_id>`

## Run identity, review, and recovery

A run ID is a durable scientific execution identity in the workspace-global
`artifacts/runs/<run-id>/` namespace shared by tabular, BIDS, and other run
workflows. An explicit or generated ID must be one safe, nonempty filesystem
name. It cannot be `.`, `..`, an absolute or nested path, contain `/` or `\\`,
contain control characters, or escape the configured runs directory. Unsafe IDs
are rejected rather than normalized.

A fresh plan atomically reserves its absent run root, writes the complete
control-plane plan, and finishes with `state: planned`. It creates no execution
claim and invokes no scientific or remote subprocess. The manifest stores a
versioned SHA-256 plan identity that binds the run, project, workflow, selected
batch and table, ordered predictors, scientific settings, upstream run where
applicable, normalized resources, output contract, complete rendered commands,
and the exact reviewed `execute.sh` bytes. SLURM and submission material are also
bound when it belongs to that plan. Timestamps, authorization state, and future
scheduler observations do not change this scientific identity.

An intact plan may transition once to local execution under the same ID. Both
documented review forms are supported:

```bash
rp run plan preprocess tabular \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id tabular-preprocess-reviewed

rp run local preprocess tabular \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id tabular-preprocess-reviewed \
  --execute
```

or:

```bash
rp run local preprocess tabular \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id tabular-preprocess-reviewed \
  --dry-run

rp run local preprocess tabular \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id tabular-preprocess-reviewed \
  --execute
```

The same contract applies to `train model`, `evaluate model`, and local
`analysis tabular` runs. Evaluation must repeat its original `--input-run`; that
upstream identity is part of the reviewed plan. The allowed transition preserves
the original creation time, identity digest, and reviewed script. Any scientific,
configuration, resource, input, output-contract, or script drift requires a new
run ID. `rp run slurm ...` remains plan-only and always requires a fresh run ID;
it is not an execution transition from an earlier local plan.

Remote tabular analysis uses its own exact plan-to-authorized-submit pair:

```bash
rp run submit analysis tabular \
  --project <project> \
  --analysis <analysis-name> \
  --run-id <run-id>

rp run submit analysis tabular \
  --project <project> \
  --analysis <analysis-name> \
  --run-id <run-id> \
  --execute
```

The first command plans without SSH, transfer, or scheduler execution. The
second is accepted only for that exact reviewed submission plan. Remote staging
and submission remain external-runtime functionality and have not been validated
on a live cluster for this alpha.

Every other existing-root use fails closed before a subprocess or mutation. A
second plan, second execution, cross-workflow collision, changed request,
foreign or unexpected payload, malformed control file, modified script, or a
root in `running`, `failed`, `succeeded`, `completed`, `staged`, `submitted`,
stage/submit-failed, cancellation-related, or unknown state is never reused.
There is no run overwrite, resume, retry, replace, force, or automatic takeover
option.

Before execution, an exclusive hidden sibling claim such as
`artifacts/runs/.<run-id>.claim` establishes one process as the owner. The owner
rechecks the plan and root after claiming it. A concurrent request or stale or
foreign claim is preserved and rejected; do not delete such a claim casually,
because it is recovery evidence. The platform removes only a claim it created
and can still identify as the same filesystem object.

Inspect a preserved run with:

```bash
rp hpc status --run-id <run-id>
```

That status form reads recorded local manifest/status state and invokes no
subprocess. It does not prove a scheduler job's current or terminal state.
`rp hpc status --live` is different: it immediately loads the SSH profile and
runs one remote `squeue` query. It does not run `sacct` or reconcile terminal
accounting; empty `squeue` is the ambiguous `not-found-or-completed`, not proof
of success. `rp hpc cancel` likewise does not cancel a remote job: it has no
`--execute` option, records local `cancel-requested`, and may render a proposed
`scancel` command without running it. Non-execute stage and pull forms can
write local plan/status material, while their execute forms remain
external-runtime behavior. `rp hpc pull --execute` uses merge-oriented
`rsync -az` and provides no terminal-state, atomic-publication, digest, or
interrupted-transfer recovery guarantee.

Also inspect `run-manifest.yaml`, `status.yaml`, `execute.sh`, and `logs/` before
choosing a new run ID. Admission failures leave the existing tree unchanged.
The atomic claim and one-shot state machine protect the control plane and prevent
duplicate launch. For the supported local tabular workflows, the output
transaction below additionally prevents ordinary pre-promotion failures from
exposing a partial final scientific output tree. SLURM and remote execution do
not receive that guarantee in this alpha.

## Local output transactions

A local plan or dry-run records a stable, versioned transaction plan in
`run-manifest.yaml`. It names the absent final `outputs/` directory, the exact
logical scientific inventory and relative filenames, content types,
`outputs/transaction-manifest.json`, and the fail-if-existing collision policy.
Planning creates neither `outputs/` nor a staging directory. An older planned
root that already contains an output directory is not adopted; preserve it and
choose a new run ID.

After acquiring and revalidating the execution claim, local execution records
`state: running`, creates one exclusively owned hidden staging directory beside
`outputs/`, and directs every reviewed producer command there. The platform
validates the complete inventory, JSON and TSV structure, predictor and target
provenance, row contracts, finite values, confinement, and source stability.
It then writes the transaction manifest, flushes the staged tree, and uses a
supported atomic no-replace directory rename to publish the absent `outputs/`
directory. It never replaces or merges with an existing final directory.
`state: succeeded` is durable only after promotion.

The exact committed inventories are:

| Local workflow | Scientific files | Attestation |
| --- | --- | --- |
| `preprocess tabular` | `split.json`, `prep.json`, `features.tsv` | `transaction-manifest.json` |
| `train model` | `split.json`, `prep.json`, `features.tsv`, `model.json` | `transaction-manifest.json` |
| `evaluate model` | `evaluation.json` | `transaction-manifest.json` |
| `analysis tabular` | `<analysis-name>.json` | `transaction-manifest.json` |

No other files, directories, links, devices, caches, or temporary entries are
valid in a committed output tree. Each transaction-manifest record contains the
logical output name, portable path beneath `outputs/`, content type, byte size,
and SHA-256 of the exact validated bytes. TSV records also contain row count and
ordered columns. The manifest binds the run, workflow, and reviewed plan
identity; it contains no host or staging path and does not recursively attest
itself. No committed scientific file or manifest may contain the transient
staging path. Training records the portable final `outputs/features.tsv`
reference in model provenance even though fitting reads the owned staged file.

The batch and selected input-table bytes are checked against the reviewed
digests before computation, after staged validation, and immediately before
promotion. A changed source fails without publication. Ordinary producer,
validation, cleanup-safe, or collision failures remove only the staging tree
owned by the current process, record `failed`, leave final `outputs/` absent,
and release the owned claim after durable status. A foreign directory introduced
at the destination remains untouched.

An interruption, cleanup failure, uncertain promotion, or terminal-status
write failure is deliberately fail-closed. Recovery paths and the execution
claim remain evidence; do not delete or adopt them. If promotion completed but
the success-status write failed, the committed output tree remains and the last
confirmed durable state is `running`. If replacement occurred, `status.yaml`
may already read `succeeded`, but its durability is uncertain; the retained
claim causes evaluation to reject the source. This contract does not
claim power-loss rollback and provides no resume, retry, force, replacement,
automatic recovery, output adoption, or stale-claim deletion.

### Verified training inputs for evaluation

Evaluation can be planned only from an unclaimed, `succeeded`, local
`train model` run for the same project and batch. Its reviewed plan and
`execute.sh` must still validate; `outputs/` must contain exactly the train
inventory; and every regular nonsymlink output must match the portable path,
size, and SHA-256 in its valid transaction manifest. Planned, running, failed,
submitted, remote-only, legacy, malformed, wrong-project, wrong-batch, claimed,
extra-file, missing-file, symlinked, escaped, or modified sources are rejected
before the evaluation root is created.

The evaluation plan binds the upstream plan identity, the SHA-256 of the source
transaction manifest, every attested output record, and the expected split,
feature-table, and model digests. The complete upstream transaction is checked
again after the evaluation claim is acquired, immediately before the consumer,
and after output validation before promotion. The low-level consumer reads each
input once, checks the expected digest over those bytes, and parses those same
bytes; it does not validate a path and reopen it. Source drift after planning or
during execution therefore publishes no evaluation output. Keep the rejected
plan intact and choose a new run ID.

Remote and SLURM tabular plans retain their existing experimental or plan-only
boundary. They are not covered by local staging, output attestation, downstream
digest verification, or live-cluster validation.

## Bring your own table

The public `rp` route accepts a UTF-8 `.csv`, `.tsv`, or tab-delimited `.txt`
feature table with one nonempty header and stable row ordering. A selected
batch manifest contains exactly one data row that names `feature_table` and
`target_column`; the referenced table may contain many records. The resolved
input is:

```text
DATASETS_ROOT / canonical_dataset / canonical_features_root / feature_table
```

Review that path with `rp config paths` and the batch row with `rp batch show`.
Do not change the table between reviewed planning and execution: the plan binds
its bytes by SHA-256 and execution fails closed on drift. For the public binary
logistic path, encode the target as `0` and `1`.

The ordered, nonempty `models.default.feature_columns` list must name finite
numeric predictors. Do not include identifiers, the selected target, alternate
outcomes, group variables, or other leakage-prone fields. This bounded path
does not automatically encode categorical values, impute missing or nonfinite
predictors, convert multiclass targets, merge tables, iterate batch rows, or
construct Cartesian combinations. Prepare those choices explicitly before
running the platform.

After registering the private overlay and its data root as described in the
[bring-your-own-data guide](byod.md), use one fresh run ID per stage:

```bash
PROJECT_NAME=my-private-project
BATCH_NAME=my-binary-table
PREPROCESS_RUN_ID=byod-preprocess-001
TRAIN_RUN_ID=byod-train-001
EVALUATE_RUN_ID=byod-evaluate-001

rp config validate --project "$PROJECT_NAME"
rp config paths --project "$PROJECT_NAME"
rp batch show --project "$PROJECT_NAME" --batch "$BATCH_NAME"

rp run local preprocess tabular \
  --project "$PROJECT_NAME" --batch "$BATCH_NAME" \
  --run-id "$PREPROCESS_RUN_ID" --dry-run
rp run local preprocess tabular \
  --project "$PROJECT_NAME" --batch "$BATCH_NAME" \
  --run-id "$PREPROCESS_RUN_ID" --execute

rp run local train model \
  --project "$PROJECT_NAME" --batch "$BATCH_NAME" \
  --run-id "$TRAIN_RUN_ID" --dry-run
rp run local train model \
  --project "$PROJECT_NAME" --batch "$BATCH_NAME" \
  --run-id "$TRAIN_RUN_ID" --execute
rp hpc status --run-id "$TRAIN_RUN_ID"

rp run local evaluate model \
  --project "$PROJECT_NAME" --batch "$BATCH_NAME" \
  --run-id "$EVALUATE_RUN_ID" --input-run "$TRAIN_RUN_ID" --dry-run
rp run local evaluate model \
  --project "$PROJECT_NAME" --batch "$BATCH_NAME" \
  --run-id "$EVALUATE_RUN_ID" --input-run "$TRAIN_RUN_ID" --execute
```

Do not plan evaluation until the training status is `succeeded`. The training
transaction must still have its exact reviewed script, inventory, manifest,
sizes, and digests. Preprocess commits `split.json`, `prep.json`, and
`features.tsv`; train adds `model.json`; evaluation commits
`evaluation.json`. Each committed `outputs/` tree also contains exactly
`transaction-manifest.json`. Review the run manifest, status, logs, predictor
provenance, transaction manifest, and evaluation metrics. A failed,
interrupted, running, or completed run root is never reused. Preserve its QC,
claim, staging, and recovery evidence, then choose a new run ID.

## Predictor contract

Public `rp` preprocessing and model-training workflows take their ordered
predictor contract only from `models.default.feature_columns` in
`config/models.yaml`. The selected batch row continues to own `feature_table`
and `target_column`; predictor names do not belong in the batch TSV or in
`config/preprocessing.yaml`.

The list must be nonempty, unique, known to the selected table, distinct from
the target and generated columns, and numeric. Exclude identifiers, alternate
outcomes, group variables, and any other leakage-prone columns. Predictor order
is scientifically meaningful and is preserved in preprocessing, model, run,
and evaluation provenance. Invalid contracts fail before a run directory or
scientific output is written.

The public synthetic binary model uses this exact order:

```yaml
models:
  default:
    feature_columns:
      - feature_a
      - feature_b
      - feature_c
      - measure_x
      - measure_y
      - feature_d
```

Package-local commands stay explicit:

```bash
python -m research_platform.io.cli merge \
  datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/sources/toy_core.tsv \
  datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/sources/toy_measurements.tsv \
  --on record_id \
  --format tsv \
  --backend polars \
  --output datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/toy_features.tsv

python -m research_platform.analysis.cli split create \
  --table datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/toy_features.tsv \
  --target-column binary_target \
  --output artifacts/tabular-slice/split.json

python -m research_platform.analysis.cli prep fit \
  --table datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/toy_features.tsv \
  --split artifacts/tabular-slice/split.json \
  --target-column binary_target \
  --feature-columns feature_a feature_b feature_c measure_x measure_y feature_d \
  --output artifacts/tabular-slice/prep.json

python -m research_platform.analysis.cli prep apply \
  --table datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/toy_features.tsv \
  --plan artifacts/tabular-slice/prep.json \
  --split artifacts/tabular-slice/split.json \
  --output artifacts/tabular-slice/features.tsv

python -m research_platform.analysis.cli model train \
  --table artifacts/tabular-slice/features.tsv \
  --split artifacts/tabular-slice/split.json \
  --target-column binary_target \
  --feature-columns feature_a feature_b feature_c measure_x measure_y feature_d \
  --output artifacts/tabular-slice/model.json

python -m research_platform.analysis.cli model evaluate \
  --table artifacts/tabular-slice/features.tsv \
  --split artifacts/tabular-slice/split.json \
  --target-column binary_target \
  --model artifacts/tabular-slice/model.json \
  --output artifacts/tabular-slice/evaluation.json
```

Regression commands are additive and do not replace the logistic path:

```bash
python -m research_platform.analysis.cli split create \
  --table datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/toy_features.tsv \
  --target-column continuous_target \
  --strategy stratified_binned \
  --stratify-bin-count 4 \
  --output artifacts/tabular-slice/regression-split.json

python -m research_platform.analysis.cli prep fit \
  --table datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/toy_features.tsv \
  --split artifacts/tabular-slice/regression-split.json \
  --target-column continuous_target \
  --feature-columns feature_a feature_b feature_c measure_x measure_y feature_d \
  --output artifacts/tabular-slice/regression-prep.json

python -m research_platform.analysis.cli prep apply \
  --table datasets/ds-derivatives-example/derivatives/features/project-pilot-tabular/toy_features.tsv \
  --plan artifacts/tabular-slice/regression-prep.json \
  --split artifacts/tabular-slice/regression-split.json \
  --output artifacts/tabular-slice/regression-features.tsv

python -m research_platform.analysis.cli regression train \
  --table artifacts/tabular-slice/regression-features.tsv \
  --split artifacts/tabular-slice/regression-split.json \
  --target-column continuous_target \
  --feature-columns feature_a feature_b feature_c measure_x measure_y feature_d \
  --kind elastic_net_regression \
  --output artifacts/tabular-slice/regression-model.json

python -m research_platform.analysis.cli regression evaluate \
  --table artifacts/tabular-slice/regression-features.tsv \
  --split artifacts/tabular-slice/regression-split.json \
  --target-column continuous_target \
  --model artifacts/tabular-slice/regression-model.json \
  --output artifacts/tabular-slice/regression-evaluation.json
```

The package-level analysis CLI retains predictor inference when
`--feature-columns` is omitted for backward compatibility with direct callers.
That compatibility behavior is not the public `rp` bring-your-own-data
contract; public `rp` runs require the reviewed YAML list above.

Stage separation is enforced at the `rp` layer:

- `rp run preprocess tabular` only creates split/preprocess outputs
- `rp run train model` creates train-stage outputs, including the model artifact
- `rp run evaluate model` must consume an existing train run via `--input-run <train_run_id>` and only runs `model evaluate`

The primary public sequence must complete training before evaluation can be
planned. For example:

```bash
rp run local train model \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id tabular-train \
  --dry-run
rp run local train model \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id tabular-train \
  --execute
rp run local evaluate model \
  --project project-pilot-tabular \
  --batch toy_binary_logreg \
  --run-id tabular-eval \
  --input-run tabular-train \
  --dry-run
```

`rp run preprocess tabular` consumes an already-materialized canonical feature table and does not own merge semantics.

The canonical input is `datasets/ds-tabular-example/toy_observations.csv`;
`toy_core.tsv`, `toy_measurements.tsv`, and `toy_features.tsv` are validated
example derivatives. Every identifier and value is an algorithmic invention.
No participant, patient, health, demographic, or other human data were used,
and no external dataset was used. Regenerate or verify all four files with:

```bash
python3 ops/scripts/generate_toy_tabular_fixtures.py
python3 ops/scripts/generate_toy_tabular_fixtures.py --check
```
