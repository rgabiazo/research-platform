# Reusable ROI Workflows

> **Alpha status — Runnable locally for the checked-in generic path.**
> `project-example` now provides deterministic synthetic NIfTI inputs and a
> verified project-level coordinate-sphere build and generic-NIfTI extraction
> path. LOSO, featquery, transforms, real BOLD/FEAT inputs, and HPC execution
> remain experimental or external-runtime; atlas-label and data-driven hooks
> remain scaffolds.

This document describes the reusable ROI workflow foundation.

Phase 1 added validation, provenance, BIDS-like derivative paths, and thin
lifecycle CLI commands. Phase 2 added generic NIfTI ROI geometry, mask writing,
peak selection from existing statistical maps, and non-FSL extraction helpers.
Phase 3 wires those foundations into local, config-driven commands for generic
ROI mask builds and generic extraction tables. Phase 4 adds local FSL-backed
leave-one-subject-out (LOSO) group-map planning and ROI mask creation from
existing subject-level fixed-effects COPE/VARCOPE inputs. Phase 5 adds local
FSL `featquery` extraction from existing FEAT or fixed-effects COPE-style
directories using configured ROI masks, including LOSO masks produced by Phase
4.

Phase 5 still does not estimate subject-level fixed effects, run general group
FEAT orchestration, change LOSO FLAME1 generation beyond consuming existing
masks, add analysis-bids runtime stages, or stage/submit any HPC ROI jobs.

Completed subject-level localizer FFX outputs can now be handed off in
plan-only mode from `research-neuro` into this existing LOSO ROI planning
surface. The handoff validates that configured fixed-effects input paths are
satisfied by completed FFX COPE, VARCOPE, and mask outputs; it does not create
ROI masks, run FLAME1, transform masks, or run extraction.

MNI-to-T1w ROI transform planning is also available in `research-neuro` as a
plan-only preflight surface. It inventories existing MNI-space ROI masks,
target T1w or subject-specific FEAT/MVPA reference images, ordered transform
chains, planned transformed-mask paths, and QC checks to perform during explicit
execution. It renders shell-safe `antsApplyTransforms` argv vectors and checks
whether `antsApplyTransforms` is available on `PATH` or supplied as an explicit
executable path. The planner remains no-write.

Step 6B adds explicit package APIs in `research_platform.neuro.roi_transforms`
to validate, select, and execute already-planned MNI-to-T1w ROI transform jobs.
Execution runs exactly the planned ANTs argv vector, refuses unplanned or
unsafe writes by default, then performs actual transformed-mask QC: output
existence, reference geometry match, binary mask validation, voxel count and
empty/small-mask thresholds, optional coverage-mask overlap, nearest-neighbor
interpolation provenance, and ordered transform-chain provenance. ANTs remains
an external runtime dependency; the package does not install it. This support
now has a thin CLI lifecycle under `rp analysis roi transform`. It does not
create LOSO masks, run FSL, run MVPA extraction, or write visualization/report
outputs.

## Package Boundaries

- `research-neuro` owns reusable ROI set, extraction set, and sidecar validation,
  plus generic NIfTI image IO, geometry, masks, peak selection, sidecar writing,
  extraction metrics, and local generic ROI execution.
- `research-bids` owns BIDS-like ROI derivative path naming helpers.
- `research-core` exposes thin lifecycle and local execution commands that
  locate project config, resolve project roots, and delegate validation/runtime
  behavior.
- Project overlays stay thin and may carry ROI YAML under `config/analysis/`.

Suggested project config locations:

```text
project/<project-name>/config/analysis/roi_sets/<roi-set>.yaml
project/<project-name>/config/analysis/roi_transforms/<transform-set>.yaml
project/<project-name>/config/analysis/extraction_sets/<extraction-set>.yaml
```

## Public Toy ROI Quickstart

The checked-in `project-example` overlay provides a complete local example. It
uses the small regular-grid images in `datasets/ds-roi-example`, the named
`roi_example` root, two 2.1 mm spheres named `SeedA` and `SeedB`, and the
synthetic `sub-toy01`/`ses-01`/`exampletask` entities. Every input value is an
algorithmic invention; the images are visibly non-anatomical and exist only to
verify generic ROI mechanics.

From the repository root, run the lifecycle in this order:

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

The commands without `--execute` only validate or print a plan. The checked-in
configuration directs runtime masks, sidecars, extraction tables, and QC output
beneath the ignored `artifacts/project-example/toy-roi/` root; it does not
modify the canonical input dataset and does not publish a canonical derivative.
Set `ARTIFACTS_ROOT` to a fresh disposable directory before the sequence if you
want an isolated run. A repeated execution into an occupied destination fails
before changing the first result because `runtime.existing_output` is `fail`.
The mask sidecars and primary values table use portable identifiers. The QC
table is a run-specific execution audit and may retain resolved local input and
mask paths, so do not publish the artifact tree as a canonical derivative.

Regenerate or non-mutatingly verify the input fixtures with:

```bash
python3 ops/scripts/generate_toy_roi_fixtures.py
python3 ops/scripts/generate_toy_roi_fixtures.py --check
```

See the [dataset provenance](../datasets/ds-roi-example/README.md) for the exact
voxel and affine formulas. This bounded example verifies local
`coordinate_sphere` building and `generic_nifti` mean, median, and voxel-count
extraction only. It is not evidence for ROI analysis on real imaging data or
for any external-runtime backend.

## Beginner Local Lifecycle

The beginner contract is config-first. Start with the dependency-light
`coordinate_sphere` template, then review the generated YAML before validating,
checking readiness, reviewing the plan, and executing only after review:

```bash
rp analysis roi init <name> \
  --project <project> \
  --template coordinate_sphere

rp analysis roi validate <name> \
  --project <project>

rp analysis roi doctor <name> \
  --project <project>

rp analysis roi build <name> \
  --project <project>

rp analysis roi build <name> \
  --project <project> \
  --execute
```

Generic NIfTI extraction follows the same review boundary:

```bash
rp analysis roi extraction init <extraction-set> \
  --project <project> \
  --roi-set <roi-set> \
  --template generic_nifti

rp analysis roi extraction validate <extraction-set> \
  --project <project>

rp analysis roi extraction doctor <extraction-set> \
  --project <project>

rp analysis roi extraction run <extraction-set> \
  --project <project>

rp analysis roi extraction run <extraction-set> \
  --project <project> \
  --execute
```

Generic ROI construction and `generic_nifti` extraction each plan one configured
entity. Keep the same singular `subject` identity in the ROI and extraction
YAML; a session remains optional, while the existing subject, task, and space
requirements still apply. A single `--subjects` value can replace the scaffold
subject, but generic execution does not expand a list or create Cartesian
combinations. Multi-subject lists and inclusive ranges belong to the advanced
`loso_group_map` ROI and FSL `fsl_featquery` extraction scaffolds.

`validate` checks the YAML schema and portable configuration structure. A
fresh scaffold can therefore be schema-valid before its placeholder inputs
exist. `doctor` performs the read-only execution-readiness checks and reports
separate `schema_valid` and `ready_for_execution` fields. A not-ready doctor
report exits nonzero and names the input, dependency, tool, geometry, output,
or deferred-runtime finding that needs attention.

`build` and `extraction run` print JSON plans by default. Planning creates no
outputs, temporary files, publication artifacts, or external-tool processes.
Adding `--execute` is the only authorization for local ROI output mutation.
`list` and `show` are supporting inspection commands; see `rp analysis roi
--help` and the subcommand help for their concise interfaces.

## Bring your own 3D NIfTI pair

The runnable generic route builds one coordinate-sphere mask from one 3D
reference NIfTI, then summarizes one 3D value NIfTI inside that mask. Register
the private input directory in `config/analysis.yaml`; this declaration expands
only the named environment variable and neither copies nor publishes data:

```yaml
analysis:
  external_input_roots:
    private_inputs:
      label: private-inputs
      local_root: ${RP_PRIVATE_INPUT_ROOT}
      sync_enabled: false
```

The ROI document at `config/analysis/roi_sets/private-sphere.yaml` is one
configured entity. This complete example deliberately omits the optional
session; if a session is supplied, use the same singular identity in both
documents and filenames.

```yaml
roi_set:
  name: private-sphere
  subject: sub-example01
  task: exampletask
  space: ExampleNative
  outputs:
    root_ref: artifacts_root
    path: private-roi/build
  runtime:
    existing_output: fail
  rois:
    -
      label: SeedA
      family: coordinate_sphere
      backend: generic_nifti
      desc: CoordinateSphere
      reference_image:
        root_ref: private_inputs
        path: images/sub-example01_task-exampletask_space-ExampleNative_desc-reference.nii.gz
      coordinate:
        - 0.0
        - 0.0
        - 0.0
      radius_mm: 4.0
```

The matching extraction document at
`config/analysis/extraction_sets/private-values.yaml` uses the mask planned by
that ROI set and the same subject, task, and space:

```yaml
extraction_set:
  name: private-values
  roi_set: private-sphere
  subject: sub-example01
  task: exampletask
  space: ExampleNative
  outputs:
    root_ref: artifacts_root
    path: private-roi/extraction
    format: tsv
  runtime:
    existing_output: fail
  targets:
    -
      name: PrivateValues
      backend: generic_nifti
      desc: PrivateValues
      metrics:
        - mean
        - median
        - voxel_count
        - valid_voxel_count
      inputs:
        root_ref: private_inputs
        path: images/sub-example01_task-exampletask_space-ExampleNative_desc-values.nii.gz
      roi_labels:
        - SeedA
```

Set and inspect the root, then validate, check readiness, review each plan, and
authorize each execution separately:

```bash
export RP_PRIVATE_INPUT_ROOT=/path/to/private-inputs
PROJECT_NAME=my-private-project

rp config validate --project "$PROJECT_NAME"
rp config paths --project "$PROJECT_NAME"
rp analysis roi validate private-sphere --project "$PROJECT_NAME"
rp analysis roi doctor private-sphere --project "$PROJECT_NAME"
rp analysis roi build private-sphere --project "$PROJECT_NAME"
rp analysis roi build private-sphere --project "$PROJECT_NAME" --execute
rp analysis roi extraction validate private-values --project "$PROJECT_NAME"
rp analysis roi extraction doctor private-values --project "$PROJECT_NAME"
rp analysis roi extraction run private-values --project "$PROJECT_NAME"
rp analysis roi extraction run private-values --project "$PROJECT_NAME" --execute
```

Replace `/path/to/private-inputs` only in the local shell; do not commit that
literal path. Generic ROI construction and extraction do not expand subject
lists or create Cartesian combinations. The reference and value images must
each be 3D and expose a finite 4x4 affine. The configured xyz coordinate is a
world-millimetre coordinate, the radius must be positive, and the resulting
sphere must contain at least one in-grid voxel. The value image and mask must
have equal first-three-dimensional shapes and affines equal with relative
tolerance zero and absolute tolerance `1e-5`. There is no resampling,
registration, transform, or 4D time-series behavior in this path.

Only finite selected values contribute to numeric summaries. `voxel_count`
counts every voxel selected by the binary mask; `valid_voxel_count` counts the
selected voxels whose value is finite. Review a difference between those counts
before interpreting a summary. A successful one-ROI build produces one binary
`*_mask.nii.gz` and one matching `*_mask.json` sidecar. Extraction produces one
`*_roiextract.tsv` values table and one `*_roiextract_qc.tsv` audit table. With
multiple configured ROI labels, the build produces a mask/sidecar pair per ROI
and the extraction tables contain one row per planned ROI action.

`runtime.existing_output: fail` preserves existing outputs. A collision or
failed transaction is not a request to delete the old tree: inspect the mask
sidecars, values/QC tables, and any transaction recovery material, then choose
a new configured artifacts path. QC can contain identifiers and resolved local
input paths, so treat the artifact tree as private. This recipe validates file
geometry and software contracts, not anatomy or scientific appropriateness.
FSL, ANTs, SPM, real-data pipelines, and multi-entity execution are outside
this local generic route.

## Advanced and External-Runtime Interfaces

MNI-to-T1w transforms are an advanced phase-specific interface. They require
user transform/reference inputs and ANTs for execution:

```bash
rp analysis roi transform validate <transform-set> --project <project>
rp analysis roi transform doctor <transform-set> --project <project>
rp analysis roi transform plan <transform-set> --project <project>
rp analysis roi transform run <transform-set> --project <project>
rp analysis roi transform run <transform-set> --project <project> --execute
```

The `loso_group_map` and `fsl_featquery` templates are also advanced,
external-runtime surfaces. They consume existing fixed-effects/FEAT inputs and
require FSL; they are not the beginner local path. The optional
`research_platform_fsl_ffx` scaffold profile expresses the platform's reusable
FSL fixed-effects layout, but does not install FSL or supply imaging inputs.

Scaffolded YAML is project configuration only. If a YAML file is for one
person's local study run, it can stay local and untracked unless the team
intentionally wants that project overlay committed. Use a distinct named-root
output path for each project or run so independent experiments do not target
the same ROI outputs.
Tracked ROI and extraction YAML should not contain literal personal absolute
paths. It is safe for raw YAML to use environment-root placeholders that expand
to local absolute paths at runtime; validation checks the raw configured strings
for the personal-path policy, while doctor and plan-only commands use the
expanded paths for existence checks and runtime planning.

## Scaffold ROI Configs

Use scaffold commands to create editable starter YAML under a project overlay:

```bash
rp analysis roi init example_rois --project project-demo --template coordinate_sphere
rp analysis roi extraction init example_values --project project-demo --roi-set example_rois --template generic_nifti
```

These commands scaffold configs only. They do not create ROI masks, run FSL,
run featquery, or extract values. The generated YAML is the reproducible source
of truth. Review and edit it for project-specific images, coordinates,
entities, output policy, masks, contrasts, thresholds, metrics, and labels
before planning or executing a build or extraction.

Template choices and their runtime requirements are listed by each init
command's `--help`; invalid-template errors list the same choices. The local
NIfTI templates still require user-provided image inputs. Atlas and custom-hook
templates are deferred scaffolds, while LOSO and featquery require FSL.

`--dry-run` previews the destination and complete YAML without writing.
`--force` replaces only an existing scaffold configuration; it never authorizes
runtime output replacement. Advanced scaffold override options remain
available for compatibility and are grouped separately in command help. Keep
path patterns, subject selectors, contrasts, thresholds, metrics, external
runtime settings, and other analysis choices in YAML rather than turning the
init command into a long invocation.

## ROI Families

Phase 1 recognizes these ROI families:

- `manual_mask`
- `coordinate_sphere`
- `atlas_label`
- `functional_threshold_map`
- `loso_group_map`
- `data_driven_hook`

Recognized backend names include `generic_nifti`, `fsl_featquery`,
`fsl_flame1`, `nilearn`, `freesurfer`, and `ants`. Phase 5 executes the local
`generic_nifti`, `fsl_flame1` LOSO build, and `fsl_featquery` extraction paths.
Plan-only validation does not import or require FSL, Nilearn, FreeSurfer, ANTs,
or surface/CIFTI tooling.

Phase 3 can build locally:

- `coordinate_sphere` masks from xyz millimeter coordinates and a radius.
- `manual_mask` outputs by validating and copying an existing binary NIfTI mask.
- `functional_threshold_map` masks by selecting a peak from an already-existing
  statistical NIfTI map and building a sphere around that peak.

Phase 4 can also build `loso_group_map` masks locally when the backend is
`fsl_flame1`. The FSL commands are planned as shell-safe argument vectors and
can be mocked in tests; real execution requires a local FSL install.

Phase 5 can run `fsl_featquery` extraction locally when the source FEAT
directories and ROI masks already exist. Plan-only mode builds the featquery
argument vectors and reports expected `report.txt` paths without running FSL.
Execution mode calls local `featquery`, parses the resulting reports, and writes
a configured summary table.

`atlas_label` and `data_driven_hook` remain schema/deferred families for now.

## Naming

Default ROI masks use `label-<roiLabel>` plus `desc-<methodContrast>`:

```text
sub-001_ses-01_task-exampletask_dir-AP_space-MNI152NLin2009cAsym_res-2_label-SeedSphere_desc-ModelAContrast_mask.nii.gz
```

The JSON sidecar uses the same stem:

```text
sub-001_ses-01_task-exampletask_dir-AP_space-MNI152NLin2009cAsym_res-2_label-SeedSphere_desc-ModelAContrast_mask.json
```

LOSO/group maps use BIDS-like ordering and may include the platform convention
`heldout-<subject>` for leave-one-subject-out maps. Treat that as derivative
metadata, not a guarantee of full BIDS-validator compliance.

Phase 4 writes LOSO group maps under the configured output root:

```text
<output_root>/loso_groupmaps/<roi_set>/group/<session>/func/<group_entities>_desc-<modelContrast>_heldout-sub<id>_zstat.nii.gz
```

LOSO ROI masks use:

```text
<output_root>/rois/<roi_set>/sub-<id>/<session>/func/sub-<id>_<session>_task-<task>_space-<space>_label-<roiLabel>_desc-<contrast>_mask.nii.gz
```

New LOSO FLAME1 scaffolds default their runtime/cache outputs to:

```text
<dataset_derivatives_root>/.research-platform/roi-loso-flame1-runtime/<roi-set>/
```

They also include a separate `publication:` block. When enabled,
`rp analysis roi build ... --execute` still writes the runtime/cache outputs
above and then automatically publishes a polished derivative view:

```text
<dataset_derivatives_root>/roi-loso-flame1/
  dataset_description.json
  README.md
  maps/group/<session>/func/*_contrast-<alias>_stat-z_heldout-sub<id>_desc-<model>LOSOFlame1_statmap.nii.gz
  maps/group/<session>/func/*_statmap.json
  masks/sub-<id>/<session>/func/*_label-<roiLabel>_contrast-<alias>_desc-<model>LOSOFlame1Sphere<radius>mm_mask.nii.gz
  masks/sub-<id>/<session>/func/*_mask.json
```

Likewise, `rp analysis roi extraction run ... --execute` publishes featquery
tables when the extraction config has compatible publication settings:

```text
<dataset_derivatives_root>/roi-loso-flame1/tables/group/<session>/func/
  <session>_task-<task>_dir-<dir>_desc-<model>LOSOFlame1Featquery_roistats.tsv
  <session>_task-<task>_dir-<dir>_desc-<model>LOSOFlame1FeatqueryQC_roistats.tsv
  <matching table sidecars>.json
```

The canonical publication set therefore consists of dataset metadata, the
copied maps and masks with JSON sidecars, analysis and QC tables with JSON data
dictionaries, and their portable source, contrast, coordinate, voxel-count,
warning, and QC metadata. Commands and exact runtime output/report paths are
run-audit fields, not canonical publication fields.

Publication is controlled by YAML, not a separate command or extra CLI flags.
Old configs without `publication:` keep the previous behavior and do not
publish. Contrast aliases are read from `publication.contrast_aliases` or
per-contrast alias fields when present; otherwise the workflow derives a stable
BIDS-safe CamelCase alias from the configured contrast id or description.

Runtime/cache outputs and the canonical published derivative have different
provenance boundaries. Run-specific artifacts may retain exact local paths and
commands for a private execution audit. The canonical derivative does not copy
those values verbatim. Published inputs use a path relative to the derivative
dataset or a named reference such as `root_ref:dataset_derivatives_root/...`
when the configured root makes that mapping truthful. Publication fails when a
local absolute or user-root reference cannot be mapped; it does not invent,
hash, or silently sanitize a replacement. This check covers filenames and all
text-bearing content, including nested JSON keys and values, TSV headers and
cells, serialized command data, and Markdown.

Existing destinations are also controlled in the YAML publication block:

```yaml
publication:
  enabled: true
  layout: loso_flame1_bidslike
  existing_output: fail
```

`existing_output` defaults to `fail`. In that mode, any conflicting destination
stops the complete publication operation before an existing file is changed.
Set `existing_output: replace` deliberately when the complete planned output
set has been reviewed and replacement is intended; there is no additional CLI
flag. The `publication.root.path` value must be relative and remain beneath its
named root. Unrelated files already present in the derivative remain outside
the replacement set. Directory/special-file collisions and symbolic-link
parents are rejected; published maps and masks are independent copies rather
than links back to mutable runtime files.

Before promotion, the publisher preflights every source and destination, renders
the complete output set into a temporary staging directory on the same
filesystem, and validates the staged tree. It then promotes the completed tree
transactionally. A validation, serialization, copy, or promotion failure rolls
back the operation, removes publication staging data, and leaves the previous
derivative tree unchanged. If the underlying filesystem rejects both the
primary rollback and its independent-copy fallback, or repeatedly refuses
staging cleanup, the publisher raises an explicit recovery error and retains the hidden
`.<publication-root-name>.publication-*` recovery directory beside the nearest
existing publication parent instead of deleting the last backup.

Runtime build and extraction outputs use a separate YAML-owned collision
policy:

```yaml
runtime:
  existing_output: fail
```

The safe default is `fail`, including when `existing_output` is omitted. Set it
to `replace` only after reviewing the complete plan and deliberately choosing
to replace that runtime output set. This policy is independent of
`publication.existing_output`, and scaffold `--force` never changes it.

Before execution writes anything, the runtime preflight checks every input and
planned destination, including duplicate paths, unexpected file types,
symbolic-link parents, image readability, and geometry. Generic NIfTI
mask/sidecar and values/QC sets are rendered under same-filesystem temporary
roots, validated, and promoted transactionally. Featquery runs into a hidden
same-filesystem output name and promotes the completed directory with its
tables. LOSO zstats, sidecars, generated masks, work trees, and final ROI files
also use same-filesystem candidates; a complete reusable cache set is copied
into candidates before validation rather than modified in place. FSL receives
only candidate paths, while recorded provenance retains the reviewed logical
destinations. Temporary runtime trees are removed after success or a
successfully rolled-back failure. FSL and ANTs remain advanced
external-runtime integrations.

Runtime execution and canonical publication are separate transaction
boundaries. A publication failure leaves the already completed runtime set
intact for inspection or retry; the publisher's own preflight and rollback
leave canonical public destinations unchanged. It never leaves a half-written
runtime set or a half-written canonical derivative after an ordinary
validation, tool, or promotion failure.

Runtime/cache folders are hidden by default and remain separate from the
published derivative. New LOSO FLAME1 scaffolds also include config-driven
cleanup defaults:

```yaml
runtime:
  existing_output: fail
  cleanup:
    after_roi_build: roi_runtime
    after_extraction: none
```

For FSL featquery extraction scaffolds:

```yaml
roi_mask_source:
  source: roi_set_publication
runtime:
  existing_output: fail
  cleanup:
    after_extraction: extraction_runtime
```

New featquery configs use published ROI masks by default. The supported
`roi_mask_source.source` values are `roi_set_publication` and
`roi_set_runtime`; absence means the legacy `roi_set_runtime` behavior. Explicit
target-level `roi_masks` always override referenced ROI-set mask lookup.
Published-mask lookup computes the expected BIDS-like mask path from the
referenced ROI set's `publication.root`, contrast alias, mask description, and
subject/session/task/direction/space/resolution/ROI-label entities; it does not
glob the filesystem.

Cleanup runs only after successful, complete publication. After ROI build
publication, `roi_runtime` removes only the current ROI set's
`.cache/loso_groupmaps/<roi_set>/`, `loso_groupmaps/<roi_set>/`, and
`rois/<roi_set>/` directories. `cache_only` remains available for legacy
runtime-mask workflows and removes only `.cache/loso_groupmaps/<roi_set>/`.
After successful featquery table publication, `extraction_runtime` removes only
`roi_extract/<extraction_set>/`. Old configs without `runtime.cleanup` do not
clean.

Published maps and masks are independent copies, so later runtime changes do
not alter the canonical derivative. Cleanup can reclaim the separate runtime
copies after successful publication. Visible old-run folders such as
`<dataset_derivatives_root>/roi-loso-flame1-runtime/` are legacy artifacts; only
remove them manually after verifying the published
`<dataset_derivatives_root>/roi-loso-flame1/` output.

## Example: LOSO ModelA-Style ROI Set

```yaml
roi_set:
  name: loso_demo
  backend: fsl_flame1
  space: MNI152NLin2009cAsym
  session: ses-01
  task: exampletask
  model: ModelA
  subjects: [sub-001, sub-002, sub-003]
  held_out_subjects: [sub-001]
  min_group_n: 2
  outputs:
    root_ref: dataset_derivatives_root
    path: .research-platform/roi-loso-flame1-runtime/loso_demo
  runtime:
    existing_output: fail
    cleanup:
      after_roi_build: roi_runtime
      after_extraction: none
  publication:
    enabled: true
    layout: loso_flame1_bidslike
    root:
      root_ref: dataset_derivatives_root
      path: roi-loso-flame1
    dataset_description:
      name: ROI LOSO FLAME1 outputs
      generated_by_name: roi-loso-flame1
    map_desc: "{model}LOSOFlame1"
    mask_desc: "{model}LOSOFlame1Sphere{sphere_radius_mm}mm"
  fixed_effects_inputs:
    root: "${ROI_FEAT_ROOT:-datasets/example/derivatives/fixed-effects}"
    cope_dir: "{subject_dir}/{session_dir}/func/task-{task_id}_model-{model}_contrast-{contrast_id}"
    cope_image: "cope{cope_number}.nii.gz"
    varcope_image: "varcope{cope_number}.nii.gz"
    mask_image: "mask.nii.gz"
  group_mask:
    path: "${ROI_FEAT_ROOT:-datasets/example/derivatives/fixed-effects}/group/{session_dir}/func/task-{task_id}_model-{model}/cope{cope_number}.gfeat/mask.nii.gz"
  contrasts:
    - id: CondA
      cope_number: 1
      desc: CondA
  cache:
    reuse: true
  provenance:
    schema_version: "1"
    generated_by:
      - name: research-platform
        description: Phase 4 local LOSO ROI workflow
  rois:
    - label: SeedA
      family: loso_group_map
      backend: fsl_flame1
      desc: CondA
      contrast: CondA
      seed_coordinate: [0, -52, 26]
      search_radius_mm: 12
      sphere_radius_mm: 6
      z_threshold: 3.1
      exploratory_z_threshold: 2.3
      allow_below_threshold_fallback: true
      min_voxels_fail: 4
      min_voxels_warn: 12
      mask_intersection_policy: intersection
```

When a LOSO workflow should create a fresh group mask instead of reading a
pre-existing static mask, configure the group mask as a generated fixed-effects
intersection:

```yaml
group_mask:
  strategy: fixed_effects_mask_intersection
  scope: loso_training_subjects
  root_ref: dataset_derivatives_root
  path: roi-loso-flame1/group_masks/{subject_dir}/{session_dir}/func/{subject_dir}_{session_dir}_task-{task_id}_space-{space}_desc-{contrast_desc}_heldout-sub{heldout_subject_id}_mask.nii.gz
```

Doctor and plan commands report these as planned `generated_group_mask` outputs
and still report missing subject fixed-effects masks as required inputs.
Execution writes a binary mask and JSON sidecar for each held-out
subject/contrast.

Plan without running FSL or writing masks:

```bash
rp analysis roi build loso_demo --project project-demo
```

Run locally after reviewing the plan:

```bash
rp analysis roi build loso_demo --project project-demo --execute
```

For each held-out subject and contrast, Phase 4 excludes that subject from the
training COPE/VARCOPE list before running the one-sample FLAME1 group map. This
keeps the ROI-defining map independent of the held-out subject and avoids the
circularity that would come from defining a subject's ROI with that subject's
own effect estimate.

Peak selection first searches for a local maximum inside the configured sphere
around `seed_coordinate` using `z_threshold`. If no voxel reaches the threshold
and `allow_below_threshold_fallback` is true, the strongest searched voxel is
used and the sidecar records `fallback_status: below_threshold_fallback`.

The final sphere mask is intersected with the configured group mask, the
held-out subject's fixed-effects mask, and any optional coverage masks when
`mask_intersection_policy: intersection` is used. `min_voxels_warn` records a
QC warning; `min_voxels_fail` fails the build unless `fail_on_qc: false`.

LOSO group maps are cached per held-out subject, session, task, model, contrast,
backend/design config, and lightweight input fingerprints. Multiple ROI labels
that use the same held-out subject and contrast reuse the same group zstat map.
The local FLAME1 runner accepts FSL's direct `flame1/zstat1.nii.gz` output and
also tolerates an existing wrapper-style `flame1/stats/zstat1.nii.gz`. ROI
sidecars record cached group maps with root/env references such as
`${ROI_DERIV_ROOT:-}/loso_groupmaps/...` rather than local absolute paths.

Subject-level fixed-effects planning/execution helpers exist in
`research-neuro`, but thin `research-core` CLI orchestration for generating
those FFX outputs from first-level localizer FEAT directories remains deferred.
General group FEAT orchestration, SLURM/HPC staging, remote execution, and
analysis-bids runtime stages remain deferred.

## Example: Coordinate Sphere ROI

```yaml
roi_set:
  name: coordinate_examples
  subject: sub-001
  session: ses-01
  task: exampletask
  space: MNI152NLin2009cAsym
  outputs:
    root_ref: artifacts_root
    path: roi-runtime/coordinate_examples
  runtime:
    existing_output: fail
  rois:
    - label: SeedSphere
      family: coordinate_sphere
      backend: generic_nifti
      desc: CoordinateSphere
      reference_image:
        root_ref: project_roi_root
        pattern: fixtures/reference/sub-{subject}_ses-{session}_task-exampletask_space-MNI152NLin2009cAsym_desc-reference.nii.gz
      coordinate: [0, -52, 26]
      radius_mm: 6
```

```bash
rp analysis roi build coordinate_examples --project project-demo-roi
rp analysis roi build coordinate_examples --project project-demo-roi --execute
```

Callable Phase 2 usage:

```python
from research_platform.bids.roi import build_roi_mask_path, build_roi_sidecar_path
from research_platform.neuro.nifti import load_nifti_image
from research_platform.neuro.roi_builders import build_coordinate_sphere_roi

reference = load_nifti_image("datasets/example/derivatives/preproc/sub-001/func/reference.nii.gz")
mask_path = build_roi_mask_path(
    "datasets/example/derivatives",
    subject_id="001",
    task_id="exampletask",
    space="MNI152NLin2009cAsym",
    roi_label="SeedSphere",
    method_desc="CoordinateSphere",
)

build_coordinate_sphere_roi(
    reference_image=reference,
    center_xyz_mm=[0, -52, 26],
    radius_mm=6,
    roi_label="SeedSphere",
    desc="CoordinateSphere",
    output_mask_path=mask_path,
    sidecar_path=build_roi_sidecar_path(mask_path),
)
```

## Example: Atlas Label ROI

```yaml
roi_set:
  name: atlas_examples
  rois:
    - label: AtlasLabel
      family: atlas_label
      desc: ExampleAtlas
      atlas: ExampleAtlas
      atlas_space: MNI152NLin2009cAsym
      labels: [17, 53]
```

## Example: Manual Mask ROI

```yaml
roi_set:
  name: manual_examples
  rois:
    - label: ManualMask
      family: manual_mask
      desc: CuratedMask
      source:
        root_ref: project_roi_root
        pattern: manual_masks/sub-{subject}_space-MNI152NLin2009cAsym_label-ManualMask_mask.nii.gz
```

Use project or environment root references instead of personal absolute paths.

Phase 3 can run the same build from project config:

```yaml
roi_set:
  name: manual_examples
  subject: sub-001
  session: ses-01
  task: exampletask
  space: MNI152NLin2009cAsym
  outputs:
    root_ref: artifacts_root
    path: roi-runtime/manual_examples
  runtime:
    existing_output: fail
  rois:
    - label: ManualMask
      family: manual_mask
      backend: manual
      desc: CuratedMask
      source:
        root_ref: project_roi_root
        pattern: fixtures/manual_masks/sub-{subject}_space-MNI152NLin2009cAsym_label-ManualMask_mask.nii.gz
```

```bash
rp analysis roi build manual_examples --project project-demo-roi
rp analysis roi build manual_examples --project project-demo-roi --execute
```

The execution path validates that a manual mask is binary and
geometry-compatible with a reference image when one is supplied, then writes a
normalized uint8 mask plus a JSON sidecar:

```python
from research_platform.neuro.roi_builders import copy_manual_mask_roi

copy_manual_mask_roi(
    source_mask_path="project/example/config/analysis/manual_masks/sub-001_label-ManualMask_mask.nii.gz",
    output_mask_path=mask_path,
    sidecar_path=build_roi_sidecar_path(mask_path),
    reference_image=reference,
    roi_label="ManualMask",
    desc="CuratedMask",
    source_ref="config/analysis/manual_masks/sub-001_label-ManualMask_mask.nii.gz",
)
```

## Example: Functional Threshold Map From Existing NIfTI

Phase 2 consumes an already-existing statistical map. It does not estimate a
group model or run FLAME1. Peak selection can be constrained by a search mask or
by a spherical search region around a seed coordinate. If configured, the helper
can fall back to the strongest below-threshold local maximum and records that
status in the sidecar.

```yaml
roi_set:
  name: functional_examples
  subject: sub-001
  session: ses-01
  task: exampletask
  space: MNI152NLin2009cAsym
  outputs:
    root_ref: artifacts_root
    path: roi-runtime/functional_examples
  runtime:
    existing_output: fail
  rois:
    - label: PeakSphere
      family: functional_threshold_map
      backend: generic_nifti
      desc: ExistingMap
      stat_map:
        root_ref: derivative_root
        pattern: group/task-exampletask_space-MNI152NLin2009cAsym_desc-existing_stat-z_map.nii.gz
      seed_coordinate: [0, -52, 26]
      search_radius_mm: 12
      sphere_radius_mm: 6
      z_threshold: 3.1
      allow_below_threshold_fallback: true
```

Phase 3 can execute that local generic build from config:

```bash
rp analysis roi build functional_examples --project project-demo-roi
rp analysis roi build functional_examples --project project-demo-roi --execute
```

Callable usage:

```python
from research_platform.neuro.roi_builders import build_functional_threshold_map_roi

stat_map = load_nifti_image("datasets/example/derivatives/group/task-exampletask_desc-existing_stat-z_map.nii.gz")

build_functional_threshold_map_roi(
    stat_image=stat_map,
    roi_label="PeakSphere",
    desc="ExistingMap",
    output_mask_path=mask_path,
    sidecar_path=build_roi_sidecar_path(mask_path),
    sphere_radius_mm=6,
    threshold=3.1,
    seed_xyz_mm=[0, -52, 26],
    search_radius_mm=12,
    allow_below_threshold_fallback=True,
)
```

## Example: Generic NIfTI Extraction

```yaml
extraction_set:
  name: modelA_timeseries
  roi_set: modelA
  targets:
    - name: niftiTimeseries
      backend: generic_nifti
      desc: ModelATimeseries
      inputs:
        root_ref: derivative_root
        pattern: sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-exampletask_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
      outputs:
        table_desc: ModelATimeseries
```

Phase 2 extraction is generic NIfTI summary extraction from a 3D value map and a
3D ROI mask. It does not parse featquery output and does not run FSL.

Supported metrics:

- `mean`
- `median`
- `sum`
- `std`
- `min`
- `max`
- `voxel_count`
- `valid_voxel_count`

Phase 3 can run generic extraction from config. If `roi_masks` are not supplied
on the target, mask paths are inferred from the referenced `roi_set` and the
same BIDS-like mask naming helper used during build.

```yaml
extraction_set:
  name: generic_values
  roi_set: coordinate_examples
  subject: sub-001
  session: ses-01
  task: exampletask
  space: MNI152NLin2009cAsym
  outputs:
    root_ref: artifacts_root
    path: roi-runtime/generic_values
    format: tsv
  runtime:
    existing_output: fail
  targets:
    - name: GenericValues
      backend: generic_nifti
      desc: GenericValues
      metrics: [mean, median, std, voxel_count, valid_voxel_count]
      inputs:
        root_ref: project_roi_root
        pattern: fixtures/value_maps/sub-{subject}_ses-{session}_task-exampletask_desc-value_map.nii.gz
      roi_labels: [SeedSphere]
```

```bash
rp analysis roi extraction run generic_values --project project-demo-roi
rp analysis roi extraction run generic_values --project project-demo-roi --execute
```

To extract from explicitly configured masks instead, add `roi_masks` to the
target:

```yaml
      roi_masks:
        - label: ManualMask
          path: "${ROI_DERIV_ROOT:-artifacts/roi/project-demo-roi/derivatives}/roi/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-exampletask_space-MNI152NLin2009cAsym_label-ManualMask_desc-CuratedMask_mask.nii.gz"
```

Callable usage:

```python
from research_platform.neuro.roi_extraction import (
    build_extraction_result,
    extraction_result_to_row,
    write_extraction_table,
)

result = build_extraction_result(
    value_image=load_nifti_image("datasets/example/derivatives/sub-001/func/value_map.nii.gz"),
    roi_mask_image=load_nifti_image(str(mask_path)),
    roi_label="SeedSphere",
    value_desc="ExampleValueMap",
)

write_extraction_table(
    [extraction_result_to_row(result)],
    "datasets/example/derivatives/roi/sub-001/func/sub-001_task-exampletask_desc-ExampleValueMap_roiextract.tsv",
)
```

## Example: FSL Featquery Extraction

```yaml
extraction_set:
  name: modelA_featquery
  roi_set_ref: loso_demo
  subjects: [sub-001]
  session: ses-01
  task: exampletask
  model: ModelA
  space: MNI152NLin2009cAsym
  roi_mask_source:
    source: roi_set_publication
  outputs:
    root_ref: dataset_derivatives_root
    path: .research-platform/roi-loso-flame1-runtime/modelA_featquery
    format: tsv
  runtime:
    existing_output: fail
    cleanup:
      after_extraction: extraction_runtime
  targets:
    - name: FeatqueryValues
      backend: fsl_featquery
      desc: ModelAFeatquery
      metrics: [mean_cope, roi_voxel_count]
      inputs:
        feat_root_ref: first_level_feat_root
        feat_dir: "{subject_dir}/{session_dir}/func/sub-{subject_id}_{session_dir}_task-{task_id}_model-{model}.feat"
        value_image: "stats/cope{cope}"
      contrasts:
        - id: CondA
          cope: 1
          desc: CondA
      roi_labels: [SeedA]
      featquery_output_name: "fq_{roi_label}_{source_contrast}_cope{cope}"
      missing:
        feat_dir: warn
        roi_mask: warn
        report_values: warn
```

Plan without running FSL:

```bash
rp analysis roi extraction run modelA_featquery --project project-demo
```

Run locally after reviewing the plan:

```bash
rp analysis roi extraction run modelA_featquery --project project-demo --execute
```

When ROI masks come from an ROI set, `roi_set_ref` points at
`config/analysis/roi_sets/<roi-set>.yaml`. New generated featquery configs use
published masks from the referenced ROI set's `publication.root/masks/`; old
configs without `roi_mask_source` continue to use runtime masks from
`outputs.root/rois/<roi_set>/`. For explicit masks, omit `roi_set_ref` or keep
it only for metadata, and configure `roi_masks` on the target:

```yaml
      roi_masks:
        - label: SeedA
          family: manual_mask
          root_ref: project_roi_root
          pattern: "masks/{subject_dir}/{session_dir}/func/sub-{subject_id}_{session_dir}_task-{task_id}_label-SeedA_mask.nii.gz"
```

Plan-only output includes missing-input warnings, expected featquery output
directories, report paths, and command argument vectors such as:

```text
featquery 1 <feat_dir> 1 stats/cope1 fq_SeedA_CondA_cope1 <roi_mask>
```

The command is stored as an argument list in the JSON plan so paths with spaces
remain shell-safe.

### Percent Signal Change with FSL featquery

`mean_cope` is the raw mean COPE value reported by FSL `featquery` without
percent-signal-change conversion. Existing `mean_cope` configurations keep this
behavior and render commands without `-p`.

`percent_signal_change` requests FSL `featquery` percent signal change. The
platform does not manually compute PSC; it asks `featquery` to do the conversion
with `-p`, and records the resulting value in the `mean_psc` output column.

Use a PSC-only featquery target when PSC is needed:

```yaml
      metrics: [percent_signal_change, roi_voxel_count]
      featquery:
        include_percent_signal_change: true
```

That target renders command argument vectors such as:

```text
featquery 1 <feat_dir> 1 stats/cope1 fq_SeedA_CondA_cope1 -p <roi_mask>
```

Do not request `mean_cope` and `percent_signal_change` in the same featquery
target. Raw COPE and PSC use different featquery command semantics because PSC
uses `-p`; configure separate extraction targets or extraction sets if both are
needed.

By default, extraction writes BIDS-like group-level derivatives under the
configured output root:

```text
<output_root>/roi_extract/<extraction_set>/group/<session>/group_<session>_task-<task>_desc-<targetDesc>_values.tsv
<output_root>/roi_extract/<extraction_set>/group/<session>/group_<session>_task-<task>_desc-<targetDesc>_qc.tsv
```

Use the values TSV for statistics. It is analysis-ready: path, provenance,
runtime, command, usability, `qc_flags`, and `warnings` columns are omitted, and
metric columns that are completely empty, such as an unreported `mean_psc`, are
omitted as well. The `subject_id` and `session_id` columns use BIDS entity values
without `sub-` or `ses-` prefixes, for example `002` and `01`.

Example values columns include:

```text
subject_id  session_id  task_id  model  roi_set  roi_label  roi_desc  roi_family  source_contrast  cope  mean_cope  mean_psc  roi_voxel_count  thresholded_peak  below_threshold_fallback  peak_x_mm  peak_y_mm  peak_z_mm  z_at_peak
```

Use the runtime QC TSV for the private execution audit. It includes all rows,
including rows excluded from the values TSV, and may retain exact paths,
commands, backend, usability, `qc_flags`, and `warnings`. Its canonical
published counterpart keeps the reusable diagnostics and portable source
references, but omits exact runtime commands and rejects unmapped local paths.
The QC table also records whether each row was included in the values table and,
when excluded, the reason.

The report parser tolerates colon, tab, and multi-space label/value variants.
It prefers explicit labels such as `Mean % signal change`, `Mean cope`, and
`Voxels`. If requested values are missing or a report line contains ambiguous
numeric values, the QC row records flags/warnings instead of inventing a value;
unusable rows are omitted from the values TSV.

Phase 5 consumes existing FEAT directories and existing ROI masks only. It does
not create subject-level fixed effects, run group FEAT, change LOSO FLAME1 map
generation, stage SLURM/HPC jobs, run remotely, or wire ROI extraction into the
analysis-bids runtime pipeline.

## Sidecar Fields

ROI JSON sidecars can record:

- held-out subject, session, task, and model
- ROI label and family
- source contrast
- full-sample seed coordinate
- search radius
- LOSO peak coordinate
- z value at the selected peak
- selected peak coordinate and voxel index for generic functional maps
- sphere radius
- z threshold
- thresholded versus below-threshold fallback status
- mask intersection policy
- voxel count
- warnings and QC flags

Example:

```yaml
roi_label: LosoExample
roi_family: loso_group_map
held_out_subject: sub-001
session: ses-01
task: exampletask
model: ModelA
source_contrast: condition_a_gt_baseline
full_sample_seed_coordinate: [0, -52, 26]
search_radius_mm: 12
loso_peak_coordinate: [2, -50, 24]
selected_peak_z: 4.2
sphere_radius_mm: 6
z_threshold: 3.1
fallback_status: thresholded
mask_intersection_policy: intersection
voxel_count: 104
warnings: []
qc_flags: [pass]
```

## Current ROI Lifecycle Surface

- Local plan-only validation for generic ROI build and extraction configs.
- Local `--execute` builds for `coordinate_sphere`, `manual_mask`, and
  `functional_threshold_map`.
- Local FSL-backed LOSO ROI mask builds from existing fixed-effects inputs.
- JSON sidecars beside generated masks.
- BIDS-like ROI mask paths using `label-<roi>` and `desc-<methodContrast>`.
- Local `generic_nifti` extraction tables in TSV or CSV format.
- Local `fsl_featquery` plan-only command reporting and optional execution from
  existing FEAT directories or fixed-effects COPE-style directories.
- Safe parsing of synthetic or real `featquery` `report.txt` outputs into group
  summary TSV/CSV tables.
- Advanced MNI-to-T1w or subject-reference ROI transform planning and explicit
  execution with `antsApplyTransforms` preflight and post-execution ROI QC.
- Project-relative and named-root paths such as `project_roi_root`,
  `artifacts_root`, `dataset_root`, `derivative_root`, and
  `analysis.external_input_roots` entries when configured.

## Future Phases

Future phases can add atlas fetching or label extraction, FreeSurfer,
additional ANTs-backed ROI builders, Nilearn, surface and CIFTI support,
subject-level FFX, group FEAT,
analysis-bids runtime stages, and HPC staging/runtime changes. Those phases
should keep execution backends optional and continue to separate schemas, path
naming, IO, transforms, and orchestration.
