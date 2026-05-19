"""Adaptive panel quadrature fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chunkie.geometry import PanelView
from chunkie.geometry.chunker import right_normals
from chunkie.kernels import Kernel

from .legendre import interpolation_matrix, legendre_rule


@dataclass(frozen=True)
class _SubpanelSourceView:
    positions: NDArray[np.floating]
    derivatives: NDArray[np.floating]
    second_derivatives: NDArray[np.floating]
    normals: NDArray[np.floating]
    weights: NDArray[np.floating]

    @property
    def flat_positions(self) -> NDArray[np.floating]:
        return self.positions[:, :, 0]

    @property
    def flat_normals(self) -> NDArray[np.floating]:
        return self.normals[:, :, 0]

    @property
    def flat_weights(self) -> NDArray[np.floating]:
        return self.weights[:, 0]


def build_adaptive_panel_matrix(
    panel: PanelView,
    target,
    kernel: Kernel,
    *,
    tolerance: float = 1.0e-12,
    max_depth: int = 20,
    quadrature_order: int | None = None,
) -> NDArray[np.generic]:
    return adaptive_panel_matrix(
        panel,
        target,
        kernel,
        tolerance=tolerance,
        max_depth=max_depth,
        quadrature_order=quadrature_order,
    )


def adaptive_panel_matrix(
    panel: PanelView,
    target,
    kernel: Kernel,
    *,
    tolerance: float = 1.0e-12,
    max_depth: int = 20,
    quadrature_order: int | None = None,
) -> NDArray[np.generic]:
    """Integrate one source panel adaptively against original panel nodes.

    The returned tensor maps density values on the original Legendre nodes to
    field values at the targets. Geometry and density are interpolated on each
    accepted subpanel, so this is a local correction/reference path rather than
    a replacement for global dense assembly.
    """

    order = int(quadrature_order or max(panel.nodes.size + 4, 8))
    low_order = max(4, order)
    high_order = low_order + 4
    return _adaptive_interval(
        panel,
        target,
        kernel,
        interval=(-1.0, 1.0),
        low_order=low_order,
        high_order=high_order,
        tolerance=float(tolerance),
        max_depth=int(max_depth),
        depth=0,
    )


def _adaptive_interval(
    panel: PanelView,
    target,
    kernel: Kernel,
    *,
    interval: tuple[float, float],
    low_order: int,
    high_order: int,
    tolerance: float,
    max_depth: int,
    depth: int,
) -> NDArray[np.generic]:
    low = _panel_interval_matrix(
        panel, target, kernel, interval=interval, quadrature_order=low_order
    )
    high = _panel_interval_matrix(
        panel, target, kernel, interval=interval, quadrature_order=high_order
    )

    # The absolute-plus-relative check prevents endless refinement for tiny
    # blocks while still forcing close-panel singular structure to subdivide.
    error = np.max(np.abs(high - low))
    scale = 1.0 + np.max(np.abs(high))
    if error <= tolerance * scale or depth >= max_depth:
        return high

    a, b = interval
    midpoint = 0.5 * (a + b)
    left = _adaptive_interval(
        panel,
        target,
        kernel,
        interval=(a, midpoint),
        low_order=low_order,
        high_order=high_order,
        tolerance=tolerance,
        max_depth=max_depth,
        depth=depth + 1,
    )
    right = _adaptive_interval(
        panel,
        target,
        kernel,
        interval=(midpoint, b),
        low_order=low_order,
        high_order=high_order,
        tolerance=tolerance,
        max_depth=max_depth,
        depth=depth + 1,
    )
    return left + right


def _panel_interval_matrix(
    panel: PanelView,
    target,
    kernel: Kernel,
    *,
    interval: tuple[float, float],
    quadrature_order: int,
) -> NDArray[np.generic]:
    nodes, reference_weights = legendre_rule(quadrature_order)
    a, b = interval
    half_width = 0.5 * (b - a)
    midpoint = 0.5 * (a + b)
    local_nodes = midpoint + half_width * nodes
    interpolation = interpolation_matrix(panel.nodes, local_nodes)

    positions = np.einsum("ql,rl->rq", interpolation, panel.positions)
    derivatives = np.einsum("ql,rl->rq", interpolation, panel.derivatives)
    second_derivatives = np.einsum("ql,rl->rq", interpolation, panel.second_derivatives)
    speed = np.linalg.norm(derivatives, axis=0)
    weights = half_width * reference_weights * speed
    normals = (
        right_normals(derivatives[:, :, None])[:, :, 0]
        if derivatives.shape[0] == 2
        else panel.normals
    )

    source = _SubpanelSourceView(
        positions=positions[:, :, None],
        derivatives=derivatives[:, :, None],
        second_derivatives=second_derivatives[:, :, None],
        normals=normals[:, :, None],
        weights=weights[:, None],
    )
    values = kernel(source, target)
    return np.einsum("oitq,qk,q->oitk", values, interpolation, weights)
