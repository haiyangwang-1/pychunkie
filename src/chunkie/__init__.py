"""Python-first boundary integral equation toolkit."""

from . import geometry, kernels, quadrature, rcip, system
from .geometry import Chunker, ChunkGraph
from .kernels import Kernel, kernel
from .system import (
    BoundaryEquation,
    BoundaryTrace,
    Density,
    IntegralSystem,
    LaplaceExteriorDirichletSystem,
    LayerPotential,
    SystemMatrix,
    SystemSolution,
)

__all__ = [
    "BoundaryEquation",
    "BoundaryTrace",
    "ChunkGraph",
    "Chunker",
    "Density",
    "IntegralSystem",
    "Kernel",
    "LaplaceExteriorDirichletSystem",
    "LayerPotential",
    "SystemMatrix",
    "SystemSolution",
    "geometry",
    "kernel",
    "kernels",
    "quadrature",
    "rcip",
    "system",
]
