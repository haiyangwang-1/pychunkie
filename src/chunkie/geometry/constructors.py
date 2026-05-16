"""User-facing geometry constructors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from .chunker import Chunker, right_normals

Curve = Callable[[NDArray[np.floating]], Any]


def circle(
    *,
    radius: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
    quadrature_order: int = 16,
    panel_count: int = 16,
) -> Chunker:
    def curve(theta: NDArray[np.floating]):
        c = np.asarray(center, dtype=float).reshape(2, 1)
        r = float(radius)
        positions = c + r * np.vstack((np.cos(theta), np.sin(theta)))
        derivatives = r * np.vstack((-np.sin(theta), np.cos(theta)))
        second = -r * np.vstack((np.cos(theta), np.sin(theta)))
        return positions, derivatives, second

    return _chunker_from_parameter_curve(
        curve,
        quadrature_order=quadrature_order,
        panel_count=panel_count,
        start=0.0,
        stop=2.0 * np.pi,
        closed=True,
        orientation="ccw",
        metadata={"constructor": "circle", "radius": radius, "center": center},
    )


def ellipse(
    *,
    axes: tuple[float, float] = (1.0, 1.0),
    center: tuple[float, float] = (0.0, 0.0),
    quadrature_order: int = 16,
    panel_count: int = 16,
) -> Chunker:
    def curve(theta: NDArray[np.floating]):
        c = np.asarray(center, dtype=float).reshape(2, 1)
        a, b = axes
        positions = c + np.vstack((a * np.cos(theta), b * np.sin(theta)))
        derivatives = np.vstack((-a * np.sin(theta), b * np.cos(theta)))
        second = np.vstack((-a * np.cos(theta), -b * np.sin(theta)))
        return positions, derivatives, second

    return _chunker_from_parameter_curve(
        curve,
        quadrature_order=quadrature_order,
        panel_count=panel_count,
        start=0.0,
        stop=2.0 * np.pi,
        closed=True,
        orientation="ccw",
        metadata={"constructor": "ellipse", "axes": axes, "center": center},
    )


def chunker_from_curve(
    curve: Curve,
    *,
    quadrature_order: int = 16,
    closed: bool = True,
    tolerance: float = 1.0e-10,
    min_panel_count: int = 8,
    max_panel_count: int | None = None,
) -> Chunker:
    panel_count = min_panel_count if max_panel_count is None else min(min_panel_count, max_panel_count)
    start, stop = (0.0, 2.0 * np.pi) if closed else (0.0, 1.0)
    return _chunker_from_parameter_curve(
        curve,
        quadrature_order=quadrature_order,
        panel_count=panel_count,
        start=start,
        stop=stop,
        closed=closed,
        orientation="ccw" if closed else "open",
        metadata={"constructor": "chunker_from_curve", "tolerance": tolerance},
    )


def chunker_from_polygon(
    vertices: ArrayLike,
    *,
    quadrature_order: int = 16,
    closed: bool = True,
    corner_refinement: str = "dyadic",
    refinement_depth: int = 20,
) -> Chunker:
    verts = _as_vertices(vertices)
    edge_count = verts.shape[1] if closed else verts.shape[1] - 1
    nodes, reference_weights = leggauss(quadrature_order)
    positions = np.empty((2, quadrature_order, edge_count), dtype=float)
    derivatives = np.empty_like(positions)
    second = np.zeros_like(positions)
    weights = np.empty((quadrature_order, edge_count), dtype=float)

    for edge_id in range(edge_count):
        start = verts[:, edge_id]
        end = verts[:, (edge_id + 1) % verts.shape[1]]
        midpoint = 0.5 * (start + end)
        half_edge = 0.5 * (end - start)
        positions[:, :, edge_id] = midpoint[:, None] + half_edge[:, None] * nodes[None, :]
        derivatives[:, :, edge_id] = half_edge[:, None]
        weights[:, edge_id] = reference_weights * np.linalg.norm(half_edge)

    normals = right_normals(derivatives)
    return Chunker(
        positions=positions,
        derivatives=derivatives,
        second_derivatives=second,
        normals=normals,
        weights=weights,
        nodes=nodes,
        reference_weights=reference_weights,
        adjacency=_adjacency(edge_count, closed),
        closed=closed,
        orientation="ccw" if closed else "open",
        vertices=verts,
        metadata={
            "constructor": "chunker_from_polygon",
            "corner_refinement": corner_refinement,
            "refinement_depth": refinement_depth,
        },
    )


def _chunker_from_parameter_curve(
    curve: Curve,
    *,
    quadrature_order: int,
    panel_count: int,
    start: float,
    stop: float,
    closed: bool,
    orientation: str,
    metadata: dict[str, Any],
) -> Chunker:
    nodes, reference_weights = leggauss(quadrature_order)
    breaks = np.linspace(start, stop, panel_count + 1)
    first_positions, _, _ = _curve_outputs(curve, np.array([breaks[0]]))
    coordinate_dim = first_positions.shape[0]

    positions = np.empty((coordinate_dim, quadrature_order, panel_count), dtype=float)
    derivatives = np.empty_like(positions)
    second = np.empty_like(positions)
    weights = np.empty((quadrature_order, panel_count), dtype=float)

    for panel_id in range(panel_count):
        half_width = 0.5 * (breaks[panel_id + 1] - breaks[panel_id])
        midpoint = 0.5 * (breaks[panel_id + 1] + breaks[panel_id])
        parameters = midpoint + half_width * nodes
        r, d, d2 = _curve_outputs(curve, parameters)
        positions[:, :, panel_id] = r
        derivatives[:, :, panel_id] = d * half_width
        second[:, :, panel_id] = d2 * half_width**2
        weights[:, panel_id] = reference_weights * np.linalg.norm(derivatives[:, :, panel_id], axis=0)

    normals = right_normals(derivatives)
    return Chunker(
        positions=positions,
        derivatives=derivatives,
        second_derivatives=second,
        normals=normals,
        weights=weights,
        nodes=nodes,
        reference_weights=reference_weights,
        adjacency=_adjacency(panel_count, closed),
        closed=closed,
        orientation=orientation,  # type: ignore[arg-type]
        metadata=metadata,
    )


def _curve_outputs(curve: Curve, parameters: NDArray[np.floating]):
    output = curve(parameters)
    if isinstance(output, tuple) and len(output) >= 3:
        return tuple(np.asarray(item, dtype=float).reshape(np.asarray(item).shape[0], -1) for item in output[:3])

    # Position-only callbacks are supported by finite differences so simple
    # user curves can be introduced before adaptive construction is implemented.
    h = 1.0e-6
    r = np.asarray(output, dtype=float).reshape(np.asarray(output).shape[0], -1)
    rp = np.asarray(curve(parameters + h), dtype=float).reshape(r.shape)
    rm = np.asarray(curve(parameters - h), dtype=float).reshape(r.shape)
    d = (rp - rm) / (2.0 * h)
    d2 = (rp - 2.0 * r + rm) / h**2
    return r, d, d2


def _adjacency(panel_count: int, closed: bool) -> NDArray[np.integer]:
    adjacency = np.full((2, panel_count), -1, dtype=np.int64)
    for panel_id in range(panel_count):
        if panel_id > 0:
            adjacency[0, panel_id] = panel_id - 1
        elif closed:
            adjacency[0, panel_id] = panel_count - 1
        if panel_id + 1 < panel_count:
            adjacency[1, panel_id] = panel_id + 1
        elif closed:
            adjacency[1, panel_id] = 0
    return adjacency


def _as_vertices(vertices: ArrayLike) -> NDArray[np.floating]:
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2:
        raise ValueError("vertices must be a two-dimensional array")
    if verts.shape[0] != 2 and verts.shape[1] == 2:
        verts = verts.T
    if verts.shape[0] != 2 or verts.shape[1] < 2:
        raise ValueError("vertices must have shape (2, n) or (n, 2)")
    return verts
