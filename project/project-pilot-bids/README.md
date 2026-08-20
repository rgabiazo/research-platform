# project-pilot-bids

> **Alpha status — Plan/validation only.** This public synthetic overlay can be
> validated and used to review BIDS/HPC plans. It does not contain executable
> BOLD data or establish local, external-tool, scheduler, or cluster support.

This is the smallest checked-in BIDS/HPC planning overlay. It includes:

- a `preprocess-bids` configuration for the external-runtime
  `fmripost_aroma` adapter;
- an `analysis-bids` first-level FEAT configuration whose execution is also
  external-runtime;
- a placeholder `deepprep-bold` derivative reference; and
- one tiny synthetic batch manifest with invented identifiers.

Plans and local manifests from this slice may be written under `artifacts/runs/`,
not under canonical derivatives. DeepPrep, fMRIPost-AROMA, FEAT, and remote
execution require separate user data, tools or containers, credentials, and
reviewed site configuration. The presence of adapters and plans does not make
this an executed imaging workflow.

See the [BIDS and HPC guide](../../docs/bids-hpc-slice.md), the
[source-checkout quickstart](../../docs/onboarding/quickstart.md), and the
[capability matrix](../../docs/capabilities.md).

## Privacy boundary

This is one of four checked-in public overlays only. Do not copy active-study
configuration, participant manifests, exclusion notes, or data into it.
Real-study configuration must live in a separate private repository or another
explicit private boundary, outside the public `project/` tree. Do not weaken
the root project allowlist.
