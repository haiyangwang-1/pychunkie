"""Bernstein ellipse helpers for close-panel detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .chunker import Chunker


@dataclass(frozen=True)
class BernsteinPanelImage:
    """Complex image of a panel-local Bernstein ellipse."""

    references: NDArray[np.complexfloating]
    positions: NDArray[np.complexfloating]
    center: NDArray[np.complexfloating]
    radius: float


def bernstein_ellipse(point_count: int, rho: float) -> NDArray[np.complexfloating]:
    """Return reference-plane points on the Bernstein ellipse with parameter ``rho``."""

    if point_count < 3:
        raise ValueError("point_count must be at least 3")
    if rho <= 1.0:
        raise ValueError("rho must be greater than 1")
    theta = np.linspace(0.0, 2.0 * np.pi, int(point_count), endpoint=False)
    unit = np.exp(1j * theta)
    rho_value = float(rho)
    return 0.5 * (rho_value * unit + rho_value**-1 * unit.conjugate())


def bernstein_panel_image(
    chunker: Chunker,
    panel_id: int,
    *,
    rho: float,
    point_count: int = 64,
) -> BernsteinPanelImage:
    """Map a reference Bernstein ellipse through one panel polynomial."""

    if not 0 <= panel_id < chunker.panel_count:
        raise IndexError("panel_id out of range")
    references = bernstein_ellipse(point_count, rho)
    interpolation = _complex_interpolation_matrix(chunker.nodes, references)
    positions = np.einsum("ql,Rl->Rq", interpolation, chunker.positions[:, :, panel_id])
    center = np.mean(positions, axis=1)
    return BernsteinPanelImage(
        references=references,
        positions=positions,
        center=center,
        radius=bernstein_radius(positions),
    )


def bernstein_radius(points: ArrayLike) -> float:
    """Return a conservative radius for a point cloud.

    Complex coordinates are accepted because panel Bernstein images live in the
    complexified reference plane before close-evaluation rules consume them.
    """

    arr = np.asarray(points)
    if arr.size == 0:
        return 0.0
    center = np.mean(arr, axis=-1, keepdims=True)
    distances = np.sqrt(np.sum(np.abs(arr - center) ** 2, axis=0))
    return float(np.max(distances))


def _complex_interpolation_matrix(
    nodes: NDArray[np.floating],
    targets: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    barycentric_weights = _barycentric_weights(nodes)
    matrix = np.empty((targets.size, nodes.size), dtype=complex)
    for row, target in enumerate(targets):
        differences = target - nodes
        exact = np.isclose(differences, 0.0, atol=1.0e-15, rtol=0.0)
        if np.any(exact):
            matrix[row] = 0.0
            matrix[row, np.argmax(exact)] = 1.0
            continue
        # Barycentric interpolation is stable for real and complex targets and
        # keeps the analytic-continuation boundary separate from quadrature.
        scaled = barycentric_weights / differences
        matrix[row] = scaled / np.sum(scaled)
    return matrix


def _barycentric_weights(nodes: NDArray[np.floating]) -> NDArray[np.floating]:
    differences = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(differences, 1.0)
    return 1.0 / np.prod(differences, axis=1)
