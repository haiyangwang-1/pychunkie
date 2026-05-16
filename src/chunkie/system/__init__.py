"""Integral-equation system assembly, solve, and evaluation."""

from .config import SystemConfig
from .density import Density, DensityLayout, DensitySpace
from .equation import BoundaryEquation, IntegralSystem
from .layer import LayerPotential
from .matrix import SystemMatrix
from .solution import SystemSolution
from .trace import BoundaryTrace, JumpTerm

__all__ = [
    "BoundaryEquation",
    "BoundaryTrace",
    "Density",
    "DensityLayout",
    "DensitySpace",
    "IntegralSystem",
    "JumpTerm",
    "LayerPotential",
    "SystemConfig",
    "SystemMatrix",
    "SystemSolution",
]
