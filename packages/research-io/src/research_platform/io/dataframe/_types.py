from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

SupportedFormat = Literal["csv", "tsv", "txt", "parquet", "feather"]
BackendName = Literal["polars", "pandas"]
PathLike = str | Path
PathInput = PathLike | list[PathLike] | tuple[PathLike, ...]
ReadKwargs = dict[str, Any]
BackendKwargs = dict[str, Any]
FrameLike = Any
