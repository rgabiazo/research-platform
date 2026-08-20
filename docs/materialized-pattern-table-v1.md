# Materialized Pattern Table v1

This is the normative producer-facing reference for
`research_platform.neuro.mvpa.materialized_pattern_table.v1`. The format is a
BOM-free UTF-8, tab-delimited table of ROI-final feature vectors. It is not an
image manifest: every row already contains the values that will enter pattern
preparation.

The authoritative implementation constants are `SCHEMA_VERSION`,
`REQUIRED_COLUMNS`, and `OPTIONAL_COLUMNS` in
`research_platform.neuro.mvpa.materialized_pattern_table`. Documentation tests
compare the lists below directly with those constants.

## Source declaration

This mapping is one entry beneath `mvpa_set.pattern_sources`:

```yaml
name: prepared_patterns
backend: materialized_pattern_table
root_ref: private_inputs
path: patterns/patterns.tsv
schema_version: research_platform.neuro.mvpa.materialized_pattern_table.v1
```

All five keys are fixed. `name` and `root_ref` must match
`[A-Za-z0-9_.-]+`. `path` is one safe relative `.tsv` path beneath the named
root. Absolute paths, `~`, URI schemes, parent traversal, and backslashes are
rejected. The value is treated as one literal path: there is no glob, shard,
delimiter/column-mapping, or discovery-plugin behavior. Register the named
root with `analysis.external_input_roots`; do not put a personal path in the
MVPA configuration.

## Row meaning and table rules

One selected row represents exactly:

```text
one resolved exact analysis unit x one configured condition x one configured ROI
```

The batch and bundle—not this table—select exact units. The join uses the
bundle's ordered `key_columns` and exact stored values. `subject_id` must be
one key. Any other configured key column must also be present in the header and
nonempty on every row, even when it is listed as optional below. Prefixes such
as `sub-`, `ses-`, and `run-` are preserved; there is no filename inference,
prefix stripping, fuzzy matching, row iteration, or Cartesian expansion.

The header must have unique, case-distinct column names matching
`[A-Za-z_][A-Za-z0-9_.-]*`. All 19 required columns must be present. The 26
optional columns may be omitted from the header. Extra columns are permitted
only as portable scalar metadata: names must follow that same grammar and must
not collide with derived runtime fields, values must not be JSON containers or
local-path references, and a column also present in the authoritative exact
unit must match it exactly.

Every row, including an unselected row, must have the fixed schema version,
all required scalar cells, all configured unit-key cells, a globally unique
`pattern_id`, portable values, canonical booleans and a positive
`feature_count`. Selection-specific condition, ROI, CV, centering, QC, and
noise rules apply to rows joined to selected units. Unselected rows are
reported by count and stable `pattern_id`; they never enter analysis and their
vectors are not decoded by the loader.

Selected rows must cover every resolved unit, configured condition, and
configured ROI exactly once. Repeated unit-condition-ROI rows and missing
coverage are errors. Output order is resolved bundle-unit order, configured
condition order, configured ROI-source and ROI-label order, then source-row
order as a final tie-breaker.

The field tables below describe the full contract for **selected** rows unless
they explicitly say otherwise. Planning applies a smaller all-row audit before
selection: every row needs nonempty required and exact-unit-key cells, the
fixed schema version, a positive `feature_count`, canonical boolean cells,
portable scalar metadata, and a globally unique `pattern_id`. For an
unselected row, the planner does not decode JSON/vector cells or apply the
selected condition, ROI, CV, centering, QC, feature-width, or conditional-noise
rules. Its cells therefore are not evidence that its scientific payload would
be loadable if selected later.

## Required columns (19)

An empty cell is invalid for every required field.

| Field | Type and encoding | Meaning and validation |
| --- | --- | --- |
| `schema_version` | String; exact literal `research_platform.neuro.mvpa.materialized_pattern_table.v1` | Must agree with the source declaration on every row. |
| `pattern_id` | Nonempty portable string | Globally unique row identity, including across selected and unselected rows. |
| `subject_id` | Nonempty canonical string | Required exact-unit identity; must equal the resolved unit value exactly. |
| `condition_id` | Nonempty identifier | Must name a configured condition for a selected row. |
| `pattern_source_name` | Nonempty identifier | For a selected row, must equal the containing source declaration's `name`. |
| `roi_source_name` | Nonempty identifier | Must name a configured materialized-feature ROI source. |
| `roi_label` | Nonempty identifier | Identifies one ROI within `roi_source_name`; selected coverage is exact. |
| `feature_count` | Canonical base-10 positive integer | Validated as positive on every row. For a selected row, it must equal the number of values decoded from `feature_values`. Values such as `03`, zero, booleans, and fractions are invalid. |
| `voxel_order` | Nonempty producer-owned string | Stable feature-order identity. Despite the historical name, features need not be image voxels. It must remain consistent within a subject/session/task/source/ROI analysis group. |
| `voxel_index_hash` | Nonempty producer-owned string | Stable identity for the ordered feature identifiers. Runtime requires presence and group consistency but does not recompute a hash recipe. |
| `feature_space_id` | Nonempty producer-owned string | Identifies the coordinate or feature space so equal flat indices in different spaces are not treated as equal. Must remain group-consistent. |
| `roi_definition_id` | Nonempty producer-owned string | Stable identity of the ROI definition used to produce the vector. Must remain group-consistent. |
| `feature_values` | Nonempty JSON array of JSON numbers | Decoded only during loading. Booleans, strings, empty arrays, `NaN`, infinities, and nonfinite values are rejected. Length must equal `feature_count`. Values are not reordered, centered, normalized, truncated, padded, or repaired. |
| `usable` | Case-insensitive `true` or `1`; `false` or `0` | Selected execution requires complete usable coverage. False rows remain in QC and require an auditable reason. |
| `status` | Nonempty status string | With `usable=true`: `ok`, `warning`, or `usable`. With `usable=false`: `excluded`, `error`, `failed`, `skipped`, or `unusable`. Comparison is case-insensitive. |
| `mean_centering_applied` | Case-insensitive boolean using the forms above | Must equal the configured centering-enabled state. The adapter does not apply or repair centering. |
| `mean_centering_scope` | Nonempty string | Must equal the normalized configuration scope: currently `none` or `roi` (`within_roi` is normalized to `roi` by configuration parsing). |
| `noise_status` | Nonempty status string | Identity normalization requires `unused`. Diagonal normalization requires `ok`, `warning`, or `usable`. Comparison is case-insensitive. |
| `noise_usable` | Case-insensitive boolean using the forms above | Identity normalization requires false; diagonal normalization requires true. |

## Optional columns (26)

An optional column may be absent from the header. Where an empty cell has a
defined meaning, it is stated below.

| Field | Type and encoding | Requirement, empty behavior, and meaning |
| --- | --- | --- |
| `session_id` | Canonical string | Conditionally required and nonempty when it is an exact-unit key. Otherwise it must match the resolved session exactly, or remain empty when that unit has no session. |
| `task_id` | Canonical string | Conditionally required and nonempty when it is an exact-unit key. Otherwise it must match the resolved task exactly, or remain empty when absent. |
| `run_id` | Canonical BIDS run string | Conditionally required and nonempty when it is an exact-unit key. It remains distinct from the platform execution/run ID. It must match exactly or be empty when absent. |
| `cross_validation_label` | String | Empty means derive from the configured CV unit. A nonempty value must equal that derivation exactly. `subject`, `session`, and `run` derive to `subject_id`, `session_id`, and `run_id`, respectively. A custom CV unit uses the configured grouping-column order and joins `column=value` parts with `|`. |
| `event_count` | Canonical nonnegative integer | Empty means unavailable. It becomes required when `min_events_per_condition_per_run` is configured; a usable row may not fail that threshold. |
| `qc_status` | String | Empty means no separate QC status. `fail`, `failed`, `error`, or `excluded` conflicts with `usable=true`; pass-like values conflict with `usable=false`. |
| `qc_reason` | Portable string | Empty means no QC reason. An unusable row needs this, `exclusion_reason`, or a nonempty `errors` array. |
| `exclusion_id` | Portable stable string | Empty means no exclusion rule. Any exclusion identifier or reason conflicts with `usable=true`. |
| `exclusion_reason` | Portable string | Empty means no exclusion reason; otherwise provides an auditable explanation. |
| `grouping_values` | JSON object with scalar values | Empty means `{}`. Nested arrays/objects and nonfinite float values are rejected. This is metadata, not a second unit selector. |
| `warnings` | JSON array of strings | Empty means `[]`. Preserved as row audit information. |
| `errors` | JSON array of strings | Empty means `[]`. A usable row cannot declare errors. |
| `roi_reference` | Portable string | Empty means no reference. Use a named-root or repository-relative identity, never an absolute, home, parent-traversal, or `file:` path. |
| `generator_version` | Portable string | Empty means unspecified producer version. |
| `software_version` | Portable string | Empty means unspecified software version. |
| `derivation_id` | Portable stable string | Empty means unspecified derivation identity. |
| `holdout_id` | Portable stable string | Empty means unspecified holdout identity. |
| `noise_values` | Nonempty JSON numeric array when diagonal noise is configured | Diagonal-only conditional field. Values must be finite and strictly positive variances. Under identity normalization a supplied payload is ignored with an auditable warning. |
| `noise_feature_count` | Canonical positive integer | Required for diagonal noise; must equal both `feature_count` and the noise-vector width. Empty for identity noise. |
| `noise_voxel_order` | String | Required for diagonal noise and must equal `voxel_order`; empty for identity noise. |
| `noise_voxel_index_hash` | String | Required for diagonal noise and must equal `voxel_index_hash`; empty for identity noise. |
| `noise_feature_space_id` | String | Required for diagonal noise and must equal `feature_space_id`; empty for identity noise. |
| `noise_roi_definition_id` | String | Required for diagonal noise and must equal `roi_definition_id`; empty for identity noise. |
| `noise_value_kind` | String | Required for diagonal noise and must be `variance` case-insensitively. Standard deviations are not accepted. Empty for identity noise. |
| `noise_estimation_scope` | Nonempty portable string | Required for diagonal noise; identifies the estimation population/scope. Empty for identity noise. |
| `noise_source` | Nonempty portable string | Required for diagonal noise; identifies the variance estimator/source. Empty for identity noise. |

For diagonal normalization, all noise identity metadata and the actual variance
vector must be identical across conditions for the same exact unit, ROI, and
CV partition. Feature and noise identities and widths must match. Identity
normalization must declare `noise_status=unused` and `noise_usable=false`; omit
or leave empty all diagonal-only cells rather than supplying unused payloads.

## Identity and feature identity

Canonical subject, optional session, task, and BIDS run values come from the
exact resolved bundle units. The table cannot invent an optional dimension
that is absent from a unit. The platform execution ID is separate from BIDS
`run_id`.

The producer owns `voxel_order`, `voxel_index_hash`, `feature_space_id`, and
`roi_definition_id`. Choose identities that change whenever the ordered
features, feature space, or ROI definition changes. A recommended deterministic
recipe, used by the checked-in toy generator, hashes ordered feature IDs
separated by LF and ending in a final LF:

```python
from hashlib import sha256

feature_ids = ["SeedA:feature-01", "SeedA:feature-02", "SeedA:feature-03"]
payload = ("\n".join(feature_ids) + "\n").encode("utf-8")
voxel_index_hash = sha256(payload).hexdigest()
```

This is a producer recommendation, not a runtime hash algorithm. Runtime
requires a nonempty value and consistent identities within analysis groups; it
does not reconstruct feature IDs or recompute this recipe. Similarly, current
configuration validation requires ROI-collection `feature_space_id` and
`roi_definition_id` identifiers, but the materialized-table planner does not
compare those collection-level strings with every row. Do not treat the YAML
values as substitutes for reviewed row-level identities.

## Complete example rows

The identity-normalization example uses only the 19 required columns. It is a
complete cross-sectional row when the exact-unit key is `subject_id`, the CV
unit is `subject`, one condition and ROI are configured, centering is disabled,
and noise normalization is `identity`:

```tsv
schema_version	pattern_id	subject_id	condition_id	pattern_source_name	roi_source_name	roi_label	feature_count	voxel_order	voxel_index_hash	feature_space_id	roi_definition_id	feature_values	usable	status	mean_centering_applied	mean_centering_scope	noise_status	noise_usable
research_platform.neuro.mvpa.materialized_pattern_table.v1	pattern-sub-example01-condition_a-SeedA	sub-example01	condition_a	prepared_patterns	prepared_rois	SeedA	3	feature_id_ascending	7404f5c58f99830916346c351f9afcc88c5ee38dce890927a6fbfa76762a45c7	feature-space:SeedA:v1	roi-definition:SeedA:v1	[1,2.5,-0.5]	true	ok	false	none	unused	false
```

The diagonal-normalization example adds the run-derived CV label and complete
variance contract. It is complete when the exact-unit keys are `subject_id`
and `run_id`, the CV unit is `run`, one condition and ROI are configured,
centering is disabled, and noise normalization is `diagonal`:

```tsv
schema_version	pattern_id	subject_id	condition_id	pattern_source_name	roi_source_name	roi_label	feature_count	voxel_order	voxel_index_hash	feature_space_id	roi_definition_id	feature_values	usable	status	mean_centering_applied	mean_centering_scope	noise_status	noise_usable	run_id	cross_validation_label	event_count	noise_values	noise_feature_count	noise_voxel_order	noise_voxel_index_hash	noise_feature_space_id	noise_roi_definition_id	noise_value_kind	noise_estimation_scope	noise_source
research_platform.neuro.mvpa.materialized_pattern_table.v1	pattern-sub-example01-run-01-condition_a-SeedA	sub-example01	condition_a	prepared_patterns	prepared_rois	SeedA	3	feature_id_ascending	7404f5c58f99830916346c351f9afcc88c5ee38dce890927a6fbfa76762a45c7	feature-space:SeedA:v1	roi-definition:SeedA:v1	[1,2.5,-0.5]	true	ok	false	none	ok	true	run-01	run-01	12	[1,1.5,2]	3	feature_id_ascending	7404f5c58f99830916346c351f9afcc88c5ee38dce890927a6fbfa76762a45c7	feature-space:SeedA:v1	roi-definition:SeedA:v1	variance	exact_unit_roi	residual_variance_v1
```

If the configured analysis has multiple conditions, emit one row per
condition and repeat the exact same diagonal noise vector and identity metadata
for each condition in the same unit/ROI/CV partition.

## Standard-library producer example

This standalone example writes the diagonal row above with deterministic
UTF-8/LF TSV bytes. Supply an output path under your private input root:

```python
from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

SCHEMA_VERSION = "research_platform.neuro.mvpa.materialized_pattern_table.v1"
REQUIRED_COLUMNS = (
    "schema_version", "pattern_id", "subject_id", "condition_id",
    "pattern_source_name", "roi_source_name", "roi_label", "feature_count",
    "voxel_order", "voxel_index_hash", "feature_space_id",
    "roi_definition_id", "feature_values", "usable", "status",
    "mean_centering_applied", "mean_centering_scope", "noise_status",
    "noise_usable",
)
OPTIONAL_COLUMNS = (
    "session_id", "task_id", "run_id", "cross_validation_label",
    "event_count", "qc_status", "qc_reason", "exclusion_id",
    "exclusion_reason", "grouping_values", "warnings", "errors",
    "roi_reference", "generator_version", "software_version",
    "derivation_id", "holdout_id", "noise_values", "noise_feature_count",
    "noise_voxel_order", "noise_voxel_index_hash", "noise_feature_space_id",
    "noise_roi_definition_id", "noise_value_kind",
    "noise_estimation_scope", "noise_source",
)

feature_ids = ["SeedA:feature-01", "SeedA:feature-02", "SeedA:feature-03"]
index_hash = sha256(("\n".join(feature_ids) + "\n").encode("utf-8")).hexdigest()
row = {column: "" for column in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)}
row.update(
    schema_version=SCHEMA_VERSION,
    pattern_id="pattern-sub-example01-run-01-condition_a-SeedA",
    subject_id="sub-example01",
    run_id="run-01",
    cross_validation_label="run-01",
    condition_id="condition_a",
    pattern_source_name="prepared_patterns",
    roi_source_name="prepared_rois",
    roi_label="SeedA",
    feature_count="3",
    voxel_order="feature_id_ascending",
    voxel_index_hash=index_hash,
    feature_space_id="feature-space:SeedA:v1",
    roi_definition_id="roi-definition:SeedA:v1",
    feature_values=json.dumps([1, 2.5, -0.5], separators=(",", ":")),
    usable="true",
    status="ok",
    mean_centering_applied="false",
    mean_centering_scope="none",
    noise_status="ok",
    noise_usable="true",
    event_count="12",
    noise_values=json.dumps([1, 1.5, 2], separators=(",", ":")),
    noise_feature_count="3",
    noise_voxel_order="feature_id_ascending",
    noise_voxel_index_hash=index_hash,
    noise_feature_space_id="feature-space:SeedA:v1",
    noise_roi_definition_id="roi-definition:SeedA:v1",
    noise_value_kind="variance",
    noise_estimation_scope="exact_unit_roi",
    noise_source="residual_variance_v1",
)

destination = Path(sys.argv[1])
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=(*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerow(row)
```

A real producer must emit all selected units, configured conditions, and
configured ROIs, not just this one-row teaching example.

## Planning, loading, and digest protection

Planning resolves `root_ref` plus `path`, confines the table beneath the named
root, streams UTF-8 TSV rows, validates the header and scalar metadata, joins
exact units, and records SHA-256 over the exact source bytes. It does not
JSON-decode or retain `feature_values` or `noise_values`, load images, invoke
external tools, compute distances, or write files.

Loading accepts the private exact plan and its expected digest, re-reads and
rehashes the table, rejects header or byte drift, then decodes and validates
only the selected vectors. Loading is all-or-nothing: one invalid selected row
returns no analysis rows. It preserves unusable rows in QC rather than silently
repairing them. Runtime execution then requires complete usable condition/ROI
coverage and configured event thresholds.

## Local execution and output interpretation

Use the bundle and MVPA lifecycle in [MVPA crossnobis workflows](mvpa-crossnobis.md).
The materialized transaction contains these 14 regular files:

```text
neuro/pattern-materialization/patterns.tsv
neuro/pattern-materialization/qc.tsv
neuro/pattern-materialization/provenance.json
neuro/pattern-materialization/vector_metadata.json
analysis/prepared-patterns/rows.tsv
analysis/prepared-patterns/qc.tsv
analysis/prepared-patterns/provenance.json
analysis/prepared-distances/distances.tsv
analysis/prepared-distances/qc.tsv
analysis/prepared-distances/provenance.json
analysis/prepared-summaries/summaries.tsv
analysis/prepared-summaries/qc.tsv
analysis/prepared-summaries/provenance.json
manifest.json
```

Review materialization QC before interpreting prepared rows, distances, or
summaries. The successful `manifest.json` binds the planned and loaded table
digest and every output digest. `analysis/prepared-distances/distances.tsv` is
RDM-ready pairwise-distance data; it is not an exported RDM table or figure.

The v1 producer and local runtime support only ROI-final `prepared_features`.
They do not provide an image, FSL, SPM, Nilearn, CIFTI, or mixed-representation
producer. The deterministic example validates a synthetic local path; it is
not evidence of public real-data validation.

## Checked-in evidence

- [16-row deterministic table](../datasets/ds-mvpa-example/patterns/toy_crossnobis_patterns.tsv)
- [Exact-unit batch](../project/project-example/manifests/batches/toy_mvpa_units.tsv)
- [Analysis bundle](../project/project-example/config/analysis/bundles/toy-crossnobis.yaml)
- [MVPA configuration](../project/project-example/config/analysis/mvpa/toy-crossnobis.yaml)
- [Synthetic dataset provenance](../datasets/ds-mvpa-example/README.md)
- [ADR-0017](decisions/ADR-0017-materialized-mvpa-pattern-tables.md)
