# ADR-0025: research-utils leaf package

## Status

Accepted.

## Context

Several packages carry AST-identical copies of small stdlib-only helpers:
atomic text writes, SHA-256 file hashing, output-path checks, and the YAML
writer. Each copy is maintained separately, and no existing package is a
suitable shared home without adding a dependency between packages that do not
otherwise need one.

## Decision

- `research-utils` (`research_platform.utils`) is a dependency-free leaf
  package that every other package may depend on.
- It imports no other research package and no third-party library.
- Admission rule: a helper is admitted only as a body that is AST-identical in
  every package it is copied from, and nothing numerical is admitted. A
  same-named helper whose behaviour differs (for example an `_optional_text`
  that drops non-strings, or a `_write_text_atomic` without `fsync`) is not a
  copy: it stays in its own package until a separate decision changes it.

## Consequences

- New shared stdlib helpers go into `research-utils`.
- The packages that carried the AST-identical copies import them from
  `research-utils` and declare it as a dependency. Same-named variants stay in
  their own packages.
- The package graph stays acyclic: `research-utils` sits below every other
  package and depends on none of them.
