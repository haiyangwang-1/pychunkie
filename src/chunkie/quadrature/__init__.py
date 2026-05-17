"""Local panel quadrature routines."""

from .adaptive import adaptive_panel_matrix, build_adaptive_panel_matrix
from .legendre import legendre_rule
from .panel import apply_panel_potential, dense_panel_matrix, dense_panel_operator_matrix

__all__ = [
    "adaptive_panel_matrix",
    "apply_panel_potential",
    "build_adaptive_panel_matrix",
    "dense_panel_matrix",
    "dense_panel_operator_matrix",
    "legendre_rule",
]
