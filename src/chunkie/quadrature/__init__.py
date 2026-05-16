"""Local panel quadrature routines."""

from .legendre import legendre_rule
from .panel import apply_panel_potential, dense_panel_matrix

__all__ = ["apply_panel_potential", "dense_panel_matrix", "legendre_rule"]
