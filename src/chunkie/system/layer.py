"""Layer-potential terms."""

from __future__ import annotations

from dataclasses import dataclass

from chunkie.kernels import Kernel


@dataclass(frozen=True)
class LayerPotential:
    name: str
    source: object
    kernel: Kernel
    density: str
    coefficient: complex = 1.0
