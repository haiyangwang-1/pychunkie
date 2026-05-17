"""Shared square graph for nonsmooth Laplace examples."""

from __future__ import annotations

from _nonsmooth_laplace_common import DEFAULT_QUADRATURE_ORDER, make_square_graph


def square_graph(quadrature_order: int = DEFAULT_QUADRATURE_ORDER):
    """Build the four-edge square graph used by the corner examples."""

    return make_square_graph(quadrature_order=quadrature_order)
