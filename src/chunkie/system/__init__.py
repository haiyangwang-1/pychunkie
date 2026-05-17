"""Integral-equation system assembly, solve, and evaluation."""

from .config import SystemConfig
from .corrections import PanelCorrection, build_panel_correction
from .density import Density, DensityLayout, DensitySpace
from .equation import BoundaryEquation, IntegralSystem
from .laplace import LaplaceExteriorDirichletSystem
from .layer import LayerPotential
from .matrix import SystemMatrix
from .matvec import fmm_matvec
from .nonsmooth import build_rcip_state
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
    "LaplaceExteriorDirichletSystem",
    "LayerPotential",
    "PanelCorrection",
    "SystemConfig",
    "SystemMatrix",
    "SystemSolution",
    "build_panel_correction",
    "build_rcip_state",
    "fmm_matvec",
]
