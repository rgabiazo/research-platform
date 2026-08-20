# ADR-0009: Run-Local Filtered Batches

## Status

Accepted

## Context

Project batch manifests are reusable inputs. They often represent a complete cohort or a stable
analysis scope, such as all discovered BIDS first-level runs. Users still need to submit smaller
subsets during development, smoke tests, reruns, or partial data availability checks.

Before this decision, users could create subset TSVs manually. That works, but it has tradeoffs:

- manual subset files can accumulate in `project/`
- repeated shell filtering is easy to mistype
- the relationship between the full batch and the submitted subset is not always recorded
- BIDS users naturally want to type shorthand subject labels such as `001`, while manifests store
  full labels such as `sub-001`

The platform also needs to stay useful for non-BIDS workflows, so filtering should not be hard-coded
to FEAT or neuro-specific concepts.

## Decision

We keep project-level batch manifests canonical and unmodified during run submission.

When a run command receives BIDS selectors such as:

```bash
--subject-id 001
```

the platform:

- filters the selected project batch into a run-local TSV under `artifacts/runs/<run-id>/inputs/`
- points the run manifest and Snakemake command at that filtered TSV
- records the source batch path, source row count, filtered row count, and normalized filters in the
  run manifest
- treats repeated values for the same column as OR
- treats filters across different columns as AND
- normalizes BIDS shorthand labels in the BIDS CLI layer, so `001` matches `sub-001`

Generic tabular filtering lives in `research-core`. BIDS entity normalization belongs in the BIDS
command layer. FEAT does not own subject filtering.

For HPC runs, run-local input files under `artifacts/runs/<run-id>/inputs/` are staged to the remote
run directory with their relative directory structure preserved.

## Consequences

Positive:

- users can submit subject subsets without creating project-level subset batches
- run manifests preserve the exact submitted subset
- the original project batch remains stable and reusable
- BIDS shorthand labels are user-friendly without leaking BIDS behavior into generic core filtering
- the same pattern can support other manifest selectors later

Tradeoffs:

- a run may use a batch file that lives under `artifacts/`, not only under `project/`
- HPC staging must include run-local inputs, not only `run-manifest.yaml` and `submit.sbatch`
- operators need to know that failed runs with old filtered TSVs should be resubmitted as fresh runs
  after staging fixes

Rejected alternatives:

- mutate the project batch in place: too surprising and not reproducible
- require users to create subset batches manually: workable but unnecessarily error-prone
- add FEAT-specific subject filtering: violates the package boundary and would not help future tools
