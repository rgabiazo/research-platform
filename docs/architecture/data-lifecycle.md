
# Data lifecycle

The root [architecture overview](../../ARCHITECTURE.md) explains how data fits
into the wider system. [ADR-0002](../decisions/ADR-0002-canonical-vs-ephemeral-data.md)
records the accepted distinction between canonical and run-specific material.

## Canonical datasets

`datasets/` contains approved canonical inputs and reusable derivatives. A
project or workflow may reference this material without making a private or
run-specific copy canonical.

## Run-scoped artifacts

`artifacts/` contains noncanonical material produced for a particular run. It
can include plans, logs, diagnostics, generated tables and figures, staging
material, and evidence needed to understand failure, uncertainty, or recovery.
Some artifacts may need to be retained even though they are not canonical
datasets.

An artifact becomes a canonical derivative only where an explicit, supported,
workflow-specific validation and publication policy exists. File presence or a
successful command is not sufficient, and this repository does not claim one
generic publication operation for every workflow.
