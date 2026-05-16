"""RCIP compression state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RCIPState:
    metadata: dict[str, Any] = field(default_factory=dict)
