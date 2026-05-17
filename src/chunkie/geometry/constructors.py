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
    start, stop = (0.0, 2.0 * np.pi) if closed else (0.0, 1.0)
    panel_limit = max(int(min_panel_count), int(max_panel_count) if max_panel_count is not None else 4096)
    breaks = _adaptive_parameter_breaks(
        curve,
        start=start,
        stop=stop,
        quadrature_order=quadrature_order,
        tolerance=tolerance,
        min_panel_count=min_panel_count,
        max_panel_count=panel_limit,
    )
    return _chunker_from_parameter_curve(
        curve,
        quadrature_order=quadrature_order,
        breaks=breaks,
        closed=closed,
        orientation="ccw" if closed else "open",
        metadata={
            "constructor": "chunker_from_curve",
            "tolerance": tolerance,
            "parameter_intervals": np.vstack((breaks[:-1], breaks[1:])),
        },
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
    breaks: NDArray[np.floating] | None = None,
    panel_count: int | None = None,
    start: float | None = None,
    stop: float | None = None,
    closed: bool,
    orientation: str,
    metadata: dict[str, Any],
) -> Chunker:
    nodes, reference_weights = leggauss(quadrature_order)
    if breaks is None:
        if panel_count is None or start is None or stop is None:
            raise ValueError("uniform curve construction requires panel_count, start, and stop")
        breaks = np.linspace(start, stop, panel_count + 1)
    else:
        breaks = np.asarray(breaks, dtype=float)
    panel_total = breaks.size - 1
    first_positions, _, _ = _curve_outputs(curve, np.array([breaks[0]]))
    coordinate_dim = first_positions.shape[0]

    positions = np.empty((coordinate_dim, quadrature_order, panel_total), dtype=float)
    derivatives = np.empty_like(positions)
    second = np.empty_like(positions)
    weights = np.empty((quadrature_order, panel_total), dtype=float)

    for panel_id in range(panel_total):
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
        adjacency=_adjacency(panel_total, closed),
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


def _adaptive_parameter_breaks(
    curve: Curve,
    *,
    start: float,
    stop: float,
    quadrature_order: int,
    tolerance: float,
    min_panel_count: int,
    max_panel_count: int,
) -> NDArray[np.floating]:
    initial = np.linspace(start, stop, int(min_panel_count) + 1)
    accepted: list[tuple[float, float]] = []
    stack = [(float(initial[index]), float(initial[index + 1]), 0) for index in range(initial.size - 2, -1, -1)]
    total_width = max(abs(stop - start), np.finfo(float).eps)
    while stack:
        left, right, depth = stack.pop()
        error, scale = _arclength_resolution_error(curve, left, right, quadrature_order)
        interval_budget = max(abs(right - left) / total_width, np.finfo(float).eps)
        would_exceed_limit = len(accepted) + len(stack) + 2 > max_panel_count
        if error <= tolerance * scale * interval_budget or would_exceed_limit or depth >= 30:
            accepted.append((left, right))
            continue
        midpoint = 0.5 * (left + right)
        stack.append((midpoint, right, depth + 1))
        stack.append((left, midpoint, depth + 1))

    accepted.sort(key=lambda interval: interval[0])
    breaks = [accepted[0][0]]
    breaks.extend(right for _, right in accepted)
    return np.asarray(breaks, dtype=float)


def _arclength_resolution_error(
    curve: Curve,
    left: float,
    right: float,
    quadrature_order: int,
) -> tuple[float, float]:
    low_nodes, low_weights = leggauss(max(4, int(quadrature_order)))
    high_nodes, high_weights = leggauss(max(2 * int(quadrature_order) + 8, 32))
    low = _interval_length(curve, left, right, low_nodes, low_weights)
    high = _interval_length(curve, left, right, high_nodes, high_weights)
    return abs(high - low), 1.0 + abs(high)


def _interval_length(
    curve: Curve,
    left: float,
    right: float,
    nodes: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> float:
    half_width = 0.5 * (right - left)
    midpoint = 0.5 * (right + left)
    parameters = midpoint + half_width * nodes
    _, derivatives, _ = _curve_outputs(curve, parameters)
    return float(abs(half_width) * np.sum(weights * np.linalg.norm(derivatives, axis=0)))


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
