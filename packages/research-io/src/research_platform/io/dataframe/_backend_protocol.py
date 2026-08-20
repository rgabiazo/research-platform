from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from ._types import FrameLike, SupportedFormat

MergeHow = Literal["inner", "left", "right", "outer", "full", "cross"]


class DataBackendError(RuntimeError):
    """Raised when backend operations cannot be completed."""


class UnsupportedBackendError(DataBackendError, ValueError):
    """Raised when a backend is unavailable or unknown."""


class LazyBackendUnsupportedError(DataBackendError):
    """Raised when lazy execution is requested on a backend that does not support it."""


class BackendProtocol(ABC):
    name: str
    supports_lazy: bool
    supported_formats: set[str]

    @abstractmethod
    def read(
        self,
        paths: Sequence[Path],
        *,
        format: SupportedFormat,
        lazy: bool = False,
        read_kwargs: Mapping[str, Any] | None = None,
    ) -> FrameLike:
        """Read one or more paths."""

    @abstractmethod
    def collect(self, table: FrameLike) -> FrameLike:
        """Materialize lazy tables when supported."""

    @abstractmethod
    def head(self, table: FrameLike, n: int) -> FrameLike:
        """Return a terminal-safe head view."""

    @abstractmethod
    def is_lazy(self, table: FrameLike) -> bool:
        """True when table is a lazy query object."""

    @abstractmethod
    def format_supported(self, format: str) -> bool:
        """Return True if format is supported by backend."""

    @abstractmethod
    def concat(self, tables: Sequence[FrameLike]) -> FrameLike:
        """Concatenate tables."""

    @abstractmethod
    def merge(
        self,
        left: FrameLike,
        right: FrameLike,
        *,
        on: str | None,
        left_on: str | None,
        right_on: str | None,
        how: MergeHow,
    ) -> FrameLike:
        """Merge two tables."""

    @abstractmethod
    def dtypes(self, table: FrameLike) -> dict[str, str]:
        """Return a normalized mapping of column names to string dtype names."""

    @abstractmethod
    def describe(self, table: FrameLike) -> dict[str, dict[str, Any]]:
        """Return a backend-neutral description of table values."""

    @abstractmethod
    def null_summary(self, table: FrameLike) -> dict[str, int]:
        """Return per-column null counts."""

    @abstractmethod
    def replace_invalid_values(
        self,
        table: FrameLike,
        columns: Sequence[str],
        invalid_values: Sequence[Any],
        replacement: Any = None,
    ) -> FrameLike:
        """Replace explicit invalid/sentinel values in selected columns."""

    @abstractmethod
    def fill_missing_median(
        self,
        table: FrameLike,
        columns: Sequence[str] | None = None,
    ) -> FrameLike:
        """Fill null/NA values in numeric-like columns with median."""

    @abstractmethod
    def fill_missing_mode(
        self,
        table: FrameLike,
        columns: Sequence[str] | None = None,
    ) -> FrameLike:
        """Fill null/NA values in categorical-like columns with mode."""

    @abstractmethod
    def drop_rows_by_numbers(
        self,
        table: FrameLike,
        row_numbers: Sequence[int],
    ) -> FrameLike:
        """Drop rows by explicit positional row numbers."""

    @abstractmethod
    def drop_rows_by_id_values(
        self,
        table: FrameLike,
        id_column: str,
        id_values: Sequence[Any],
    ) -> FrameLike:
        """Drop rows by explicit values in an ID column."""

    @abstractmethod
    def write(
        self,
        table: FrameLike,
        path: str | Path,
        format: str,
        write_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Write a table to disk."""
