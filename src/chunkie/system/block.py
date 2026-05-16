"""Block layout records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockLayout:
    row_count: int
    column_count: int
