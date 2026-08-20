# ADR-0019: Explicit tabular predictor contract

## Status

Accepted

## Context

The low-level tabular analysis interface historically inferred numeric
predictors when callers omitted an explicit list. That behavior is useful for
backward compatibility, but it is unsafe as the public project workflow: an
identifier, alternate outcome, group variable, or other numeric leakage column
can enter a scientific model merely because it appears in a table.

The project batch already owns the canonical feature-table reference and target
column. Predictor selection needs one separate, reviewed, configuration-owned
source of truth that is preserved through planning and execution.

## Decision

1. Public `rp` tabular workflows own the ordered predictor list at
   `models.default.feature_columns` in `config/models.yaml`.
2. The selected batch row continues to own `feature_table` and
   `target_column`. Predictor lists are not copied into batch manifests or
   preprocessing configuration.
3. Public planning and execution fail closed when the predictor list is
   missing, malformed, empty, duplicated, unknown, nonnumeric, equal to the
   target, or reserved for generated workflow data. Validation occurs before a
   run tree or scientific output is created.
4. Predictor order is part of the scientific contract. The same order is
   passed to preprocessing and training. Preparation and model records retain
   the ordered names and target; each run manifest records those values plus
   the feature count, and evaluation provenance identifies its source run and
   recorded contract.
5. Split creation does not select predictors. Preparation application consumes
   the fitted preparation plan, and evaluation consumes the trained model and
   source-run predictor contract.
6. The low-level analysis CLI retains numeric inference only when an explicit
   list is omitted, preserving backward compatibility for direct callers. An
   explicit list is validated strictly.
7. The public `rp` interface does not add a predictor-selector flag. Scientific
   predictor selection remains reviewable YAML rather than an invocation-time
   override.

## Consequences

- Public project runs cannot silently absorb newly added numeric outcomes or
  identifiers.
- Predictor order is reproducible across local and SLURM plans and can be
  audited from run artifacts.
- Contributors must review `config/models.yaml` when the feature-table schema
  or scientific model changes.
- Direct low-level callers that rely on omitted-list inference remain
  supported, but that mode is not the public bring-your-own-data contract.
