# Instructions for agents in research-io

Keep `research-io` generic and reusable.

- Do not place core logic in notebooks.
- Do not hard-code paths, usernames, cluster names, or study names.
- Default tabular backend is `polars`; `pandas` is optional.
- Keep backend coupling isolated: only `src/research_platform/io/dataframe/polars_ops.py` and
  `src/research_platform/io/dataframe/pandas_ops.py` may import `polars`/`pandas` directly.
- Reader/writer packages should only re-export and orchestrate, not implement backend details.
- Keep changes inside `packages/research-io/`.
- Add/refresh tests and docs for any new behavior.
