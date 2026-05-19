"""Near-panel geometric queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .chunker import Chunker
from .points import PointInfoView


@dataclass(frozen=True)
class NearestPoint:
    positions: NDArray[np.floating]
    derivatives: NDArray[np.floating]
    second_derivatives: NDArray[np.floating]
    distances: NDArray[np.floating]
    reference_coordinates: NDArray[np.floating]
    panel_ids: NDArray[np.integer]


def flagnear(
    geometry: Chunker | PointInfoView,
    points: ArrayLike,
    *,
    rho: float = 1.8,
) -> NDArray[np.bool_]:
    """Flag targets inside MATLAB-style Bernstein-rectangle near regions."""

    return _flagnear_bernstein_rectangles(geometry, points, rho=rho)


def flagnear_rectangle(
    geometry: Chunker | PointInfoView,
    points: ArrayLike,
    *,
    rho: float = 1.8,
) -> NDArray[np.bool_]:
    """Compatibility alias for :func:`flagnear`."""

    return flagnear(geometry, points, rho=rho)


def flagnear_rectangle_grid(
    geometry: Chunker | PointInfoView,
    x: ArrayLike,
    y: ArrayLike,
    *,
    rho: float = 1.8,
) -> NDArray[np.bool_]:
    """Evaluate :func:`flagnear_rectangle` on a meshgrid in row-major grid order."""

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    xx, yy = np.meshgrid(x_array, y_array)
    points = np.vstack((xx.ravel(), yy.ravel()))
    panel_count = _near_geometry_arrays(geometry)[0].shape[2]
    return flagnear(geometry, points, rho=rho).reshape(y_array.size, x_array.size, panel_count)


def _flagnear_bernstein_rectangles(
    geometry: Chunker | PointInfoView,
    points: ArrayLike,
    *,
    rho: float,
) -> NDArray[np.bool_]:
    positions, derivatives, nodes = _near_geometry_arrays(geometry)
    if positions.shape[0] != 2:
        raise ValueError("Bernstein-rectangle near flags are implemented for 2D geometry")
    rho_value = float(rho)
    if rho_value <= 1.0:
        raise ValueError("rho must be greater than 1 for Bernstein-rectangle near flags")

    targets = np.asarray(points, dtype=float).reshape(2, -1)
    axes1, axes2, lower1, upper1, lower2, upper2 = _bernstein_rectangle_bounds(
        positions,
        derivatives,
        nodes,
        rho=rho_value,
    )
    target_columns = targets.T
    d1 = target_columns @ axes1
    d2 = target_columns @ axes2
    return (d1 >= lower1) & (d1 <= upper1) & (d2 >= lower2) & (d2 <= upper2)


def _near_geometry_arrays(
    geometry: Chunker | PointInfoView,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    if isinstance(geometry, Chunker):
        return geometry.positions, geometry.derivatives, geometry._legendre_nodes
    if isinstance(geometry, PointInfoView):
        return geometry.positions, geometry.derivatives, geometry.nodes
    raise TypeError("near flags require a Chunker or PointInfoView")


def _bernstein_rectangle_bounds(
    positions: NDArray[np.floating],
    derivatives: NDArray[np.floating],
    nodes: NDArray[np.floating],
    *,
    rho: float,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    from chunkie.geometry.bernstein import bernstein_ellipse
    from chunkie.quadrature.legendre import interpolation_matrix

    panel_count = positions.shape[2]
    ellipse_points = bernstein_ellipse(max(2 * panel_count, 20), rho)
    interpolation = interpolation_matrix(nodes, ellipse_points)
    complex_positions = np.einsum("qs,RsS->RqS", interpolation, positions)
    ellipse_images = np.stack(
        (
            (complex_positions[0] + 1j * complex_positions[1]).real,
            (complex_positions[0] + 1j * complex_positions[1]).imag,
        ),
        axis=0,
    )

    center_interpolation = interpolation_matrix(nodes, np.array([0.0]))[0]
    center_derivatives = np.einsum("s,RsS->RS", center_interpolation, derivatives)
    speeds = np.linalg.norm(center_derivatives, axis=0)
    if np.any(speeds <= np.finfo(float).eps):
        raise ValueError("panel center derivative must be nonzero for near-rectangle flags")

    axes1 = center_derivatives / speeds[None, :]
    axes2 = np.vstack((axes1[1], -axes1[0]))
    d1 = np.einsum("RqS,RS->qS", ellipse_images, axes1)
    d2 = np.einsum("RqS,RS->qS", ellipse_images, axes2)
    return (
        axes1,
        axes2,
        np.min(d1, axis=0),
        np.max(d1, axis=0),
        np.min(d2, axis=0),
        np.max(
            d2,
            axis=0,
        ),
    )


def nearest_point(chunker: Chunker, points: ArrayLike) -> NearestPoint:
    """Project target points to the nearest polynomial panel location."""

    targets = np.asarray(points, dtype=float).reshape(chunker.coordinate_dim, -1)
    positions = np.empty_like(targets)
    derivatives = np.empty_like(targets)
    second_derivatives = np.empty_like(targets)
    distances = np.empty(targets.shape[1], dtype=float)
    reference_coordinates = np.empty(targets.shape[1], dtype=float)
    panel_ids = np.empty(targets.shape[1], dtype=np.int64)
    panel_data = [
        _panel_legendre_data(chunker, panel_id) for panel_id in range(chunker.panel_count)
    ]

    for target_id in range(targets.shape[1]):
        best = None
        target = targets[:, target_id]
        for panel_id, data in enumerate(panel_data):
            reference_coordinate = _nearest_reference_coordinate(chunker, panel_id, target, data)
            position, derivative, second = _evaluate_panel_data(data, reference_coordinate)
            distance = float(np.linalg.norm(position - target))
            if best is None or distance < best[0]:
                best = (distance, reference_coordinate, panel_id, position, derivative, second)
        assert best is not None
        distance, reference_coordinate, panel_id, position, derivative, second = best
        positions[:, target_id] = position
        derivatives[:, target_id] = derivative
        second_derivatives[:, target_id] = second
        distances[target_id] = distance
        reference_coordinates[target_id] = reference_coordinate
        panel_ids[target_id] = panel_id

    return NearestPoint(
        positions=positions,
        derivatives=derivatives,
        second_derivatives=second_derivatives,
        distances=distances,
        reference_coordinates=reference_coordinates,
        panel_ids=panel_ids,
    )


def _panel_legendre_data(chunker: Chunker, panel_id: int):
    from chunkie.quadrature import legendre

    coefficient_transform = legendre.exps(chunker.quadrature_order)[2]
    coefficients = chunker.positions[:, :, panel_id] @ coefficient_transform.T
    derivative_coefficients = legendre.derpol(coefficients.T).T
    second_derivative_coefficients = legendre.derpol(derivative_coefficients.T).T
    return coefficients, derivative_coefficients, second_derivative_coefficients


def _nearest_reference_coordinate(
    chunker: Chunker, panel_id: int, target: NDArray[np.floating], data
) -> float:
    panel_positions = chunker.positions[:, :, panel_id]
    nearest_node = int(np.argmin(np.linalg.norm(panel_positions - target[:, None], axis=0)))
    reference_coordinate = float(chunker._legendre_nodes[nearest_node])
    for _ in range(20):
        position, derivative, second = _evaluate_panel_data(data, reference_coordinate)
        residual = position - target
        first = float(np.dot(residual, derivative))
        second_variation = float(np.dot(derivative, derivative) + np.dot(residual, second))
        if abs(second_variation) < 1.0e-15:
            break
        step = first / second_variation
        reference_coordinate = float(np.clip(reference_coordinate - step, -1.0, 1.0))
        if abs(step) < 1.0e-14:
            break
    return reference_coordinate


def _evaluate_panel_data(data, reference_coordinate: float):
    from chunkie.quadrature import legendre

    coefficients, derivative_coefficients, second_derivative_coefficients = data
    values, _ = legendre.pols(np.array([reference_coordinate]), coefficients.shape[1] - 1)
    dvalues, _ = legendre.pols(
        np.array([reference_coordinate]), derivative_coefficients.shape[1] - 1
    )
    ddvalues, _ = legendre.pols(
        np.array([reference_coordinate]), second_derivative_coefficients.shape[1] - 1
    )
    position = coefficients @ values[:, 0]
    derivative = derivative_coefficients @ dvalues[:, 0]
    second = second_derivative_coefficients @ ddvalues[:, 0]
    return position, derivative, second
