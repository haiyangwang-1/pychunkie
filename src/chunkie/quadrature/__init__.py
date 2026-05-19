"""Local panel quadrature routines."""

from .adaptive import adaptive_panel_matrix, build_adaptive_panel_matrix
from .ggq import (
    GGQRuleSet,
    build_ggq_panel_matrix,
    build_ggq_self_panel_matrix,
)
from .ggq import (
    getremovablequad as ggq_removable_rules,
)
from .ggq import (
    setup as setup_ggq,
)
from .helsing_ojala import (
    build_helsing_ojala_panel_matrix,
    helsing_ojala_log_singular_matrix,
    helsing_ojala_singular_matrix,
    helsing_ojala_weights,
)
from .legendre import bary_weights, barycentric_weights, interpolation_matrix, legendre_rule
from .panel import (
    apply_panel_potential,
    dense_panel_matrix,
    dense_panel_operator_matrix,
    operator_matrix_from_weighted_kernel,
)

__all__ = [
    "adaptive_panel_matrix",
    "apply_panel_potential",
    "build_adaptive_panel_matrix",
    "build_ggq_panel_matrix",
    "build_ggq_self_panel_matrix",
    "build_helsing_ojala_panel_matrix",
    "dense_panel_matrix",
    "dense_panel_operator_matrix",
    "GGQRuleSet",
    "bary_weights",
    "barycentric_weights",
    "ggq_removable_rules",
    "helsing_ojala_log_singular_matrix",
    "helsing_ojala_singular_matrix",
    "helsing_ojala_weights",
    "interpolation_matrix",
    "legendre_rule",
    "operator_matrix_from_weighted_kernel",
    "setup_ggq",
]
