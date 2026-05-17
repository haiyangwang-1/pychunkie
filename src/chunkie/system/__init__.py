"""Integral-equation system assembly, solve, and evaluation."""

from .config import SystemConfig
from .corrections import PanelCorrection, build_corrections, build_panel_correction
from .density import Density, DensityLayout, DensitySpace
from .equation import BoundaryEquation, Constraint, ConstraintTerm, IntegralSystem
from .laplace import LaplaceExteriorDirichletSystem
from .layer import LayerPotential
from .matrix import SystemMatrix
from .matvec import SystemOperator, fmm_matvec, matrix_free_matvec
from .nonsmooth import build_rcip_state
from .solution import SystemSolution
from .trace import BoundaryTrace, JumpTerm

__all__ = [
    "BoundaryEquation",
    "BoundaryTrace",
    "Constraint",
    "ConstraintTerm",
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
    "SystemOperator",
    "SystemSolution",
    "build_panel_correction",
    "build_corrections",
    "build_rcip_state",
    "fmm_matvec",
    "matrix_free_matvec",
]
