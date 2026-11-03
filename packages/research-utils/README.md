# research-utils

`research-utils` holds stdlib-only helpers for writing files safely (atomic text writes, SHA-256 file hashing, output-path checks, and a YAML writer); it imports no other research package and no third-party library. Every other package may depend on it.

## Modules

- `text`: `optional_text` and `non_empty_text` normalise text values.
- `fileio`: `write_text_atomic` writes a file through a temporary file and `os.replace`.
- `hashing`: `file_sha256` returns the SHA-256 hex digest of a file.
- `output_targets`: `resolved_output_root`, `has_parent_traversal`,
  `reject_duplicate_targets` and `reject_existing_targets` check output paths.
- `yaml_writer`: `dump_yaml` and `write_yaml` render mappings, lists and scalars as YAML.
- `_version`: `package_version` returns the installed distribution version.

## Examples

```python
from research_platform.utils.text import optional_text
from research_platform.utils.fileio import write_text_atomic
from research_platform.utils.hashing import file_sha256
from research_platform.utils.output_targets import resolved_output_root
from research_platform.utils.yaml_writer import dump_yaml
```

## What belongs here

A helper is admitted only as a body that is AST-identical in every package it
is copied from. A same-named helper whose behaviour differs stays in its own
package, and nothing numerical goes here.
