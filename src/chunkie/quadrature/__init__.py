"""Local panel quadrature routines."""

from .adaptive import adaptive_panel_matrix, build_adaptive_panel_matrix
from .helsing_ojala import (
    build_helsing_ojala_panel_matrix,
    helsing_ojala_log_singular_matrix,
    helsing_ojala_weights,
)
from .legendre import legendre_rule
from .panel import apply_panel_potential, dense_panel_matrix, dense_panel_operator_matrix

__all__ = [
    "adaptive_panel_matrix",
    "apply_panel_potential",
    "build_adaptive_panel_matrix",
    "build_helsing_ojala_panel_matrix",
    "dense_panel_matrix",
    "dense_panel_operator_matrix",
    "helsing_ojala_log_singular_matrix",
    "helsing_ojala_weights",
    "legendre_rule",
]
