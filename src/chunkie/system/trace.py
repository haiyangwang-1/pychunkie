"""Boundary traces and jump terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .layer import LayerPotential

Side = Literal["left", "right", "interior", "exterior", "plus", "minus"]


@dataclass(frozen=True)
class JumpTerm:
    coefficient: complex
    density: str


@dataclass(frozen=True)
class BoundaryTrace:
    layer: LayerPotential
    target: object
    side: Side
    jump: JumpTerm | None = None
