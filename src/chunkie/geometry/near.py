"""Near-panel geometric queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .chunker import Chunker


@dataclass(frozen=True)
class NearestPoint:
    positions: NDArray[np.floating]
    derivatives: NDArray[np.floating]
    second_derivatives: NDArray[np.floating]
    distances: NDArray[np.floating]
    reference_coordinates: NDArray[np.floating]
    panel_ids: NDArray[np.integer]


def flagnear(chunker: Chunker, points: ArrayLike, *, near_factor: float = 1.0) -> NDArray[np.bool_]:
    """Flag panels whose nodes are close to target points.

    This first implementation is intentionally direct: it compares target
    points with panel nodes and uses the panel arclength as the length scale.
    """

    targets = np.asarray(points, dtype=float).reshape(chunker.coordinate_dim, -1)
    flags = np.zeros((targets.shape[1], chunker.panel_count), dtype=bool)
    panel_lengths = np.sum(chunker.weights, axis=0)
    for panel_id in range(chunker.panel_count):
        source = chunker.positions[:, :, panel_id]
        distances = np.linalg.norm(targets[:, :, None] - source[:, None, :], axis=0)
        flags[:, panel_id] = np.min(distances, axis=1) <= near_factor * panel_lengths[panel_id]
    return flags


def flagnear_rectangle(chunker: Chunker, points: ArrayLike, *, rho: float = 1.0) -> NDArray[np.bool_]:
    """Flag targets inside per-panel padded bounding boxes."""

    targets = np.asarray(points, dtype=float).reshape(chunker.coordinate_dim, -1)
    flags = np.zeros((targets.shape[1], chunker.panel_count), dtype=bool)
    panel_lengths = chunker.panel_lengths
    for panel_id in range(chunker.panel_count):
        panel_positions = chunker.positions[:, :, panel_id]
        padding = float(rho) * panel_lengths[panel_id]
        lower = np.min(panel_positions, axis=1) - padding
        upper = np.max(panel_positions, axis=1) + padding
        # The rectangle test is deliberately axis-aligned. It is a conservative
        # candidate filter before more expensive distance or special-quadrature
        # logic decides whether a panel truly needs local treatment.
        flags[:, panel_id] = np.all((targets.T >= lower[None, :]) & (targets.T <= upper[None, :]), axis=1)
    return flags


def flagnear_rectangle_grid(
    chunker: Chunker,
    x: ArrayLike,
    y: ArrayLike,
    *,
    rho: float = 1.0,
) -> NDArray[np.bool_]:
    """Evaluate :func:`flagnear_rectangle` on a meshgrid in row-major grid order."""

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    xx, yy = np.meshgrid(x_array, y_array)
    points = np.vstack((xx.ravel(), yy.ravel()))
    return flagnear_rectangle(chunker, points, rho=rho).reshape(y_array.size, x_array.size, chunker.panel_count)


def nearest_point(chunker: Chunker, points: ArrayLike) -> NearestPoint:
    """Project target points to the nearest polynomial panel location."""

    targets = np.asarray(points, dtype=float).reshape(chunker.coordinate_dim, -1)
    positions = np.empty_like(targets)
    derivatives = np.empty_like(targets)
    second_derivatives = np.empty_like(targets)
    distances = np.empty(targets.shape[1], dtype=float)
    reference_coordinates = np.empty(targets.shape[1], dtype=float)
    panel_ids = np.empty(targets.shape[1], dtype=np.int64)
    panel_data = [_panel_legendre_data(chunker, panel_id) for panel_id in range(chunker.panel_count)]

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


def _nearest_reference_coordinate(chunker: Chunker, panel_id: int, target: NDArray[np.floating], data) -> float:
    panel_positions = chunker.positions[:, :, panel_id]
    nearest_node = int(np.argmin(np.linalg.norm(panel_positions - target[:, None], axis=0)))
    reference_coordinate = float(chunker.nodes[nearest_node])
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
    dvalues, _ = legendre.pols(np.array([reference_coordinate]), derivative_coefficients.shape[1] - 1)
    ddvalues, _ = legendre.pols(np.array([reference_coordinate]), second_derivative_coefficients.shape[1] - 1)
    position = coefficients @ values[:, 0]
    derivative = derivative_coefficients @ dvalues[:, 0]
    second = second_derivative_coefficients @ ddvalues[:, 0]
    return position, derivative, second
