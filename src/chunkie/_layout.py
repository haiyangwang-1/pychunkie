"""Explicit adapter helpers for flat solver/backend boundary layouts."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def as_boundary_vector(
    values: ArrayLike,
    *,
    expected_size: int | None = None,
    name: str = "values",
) -> np.ndarray:
    """Return the flat vector used at solver/backend adapter boundaries.

    This preserves the existing chunkIE component/node/chunk ordering while
    making the conversion explicit at call sites.
    """

    vector = np.asarray(values).reshape(-1, order="F")
    if expected_size is not None and vector.size != int(expected_size):
        raise ValueError(f"{name} has size {vector.size}, expected {int(expected_size)}")
    return vector


def as_boundary_tensor(
    values: ArrayLike, shape: tuple[int, ...], *, name: str = "values"
) -> np.ndarray:
    """Return an adapter-boundary tensor with an explicit target shape."""

    shape = tuple(int(dim) for dim in shape)
    expected_size = int(np.prod(shape, dtype=np.int64))
    arr = np.asarray(values)
    if arr.size != expected_size:
        raise ValueError(f"{name} has size {arr.size}, expected {expected_size} for shape {shape}")
    return arr.reshape(shape, order="F")


def as_boundary_point_matrix(
    values: ArrayLike,
    coordinate_dim: int,
    point_count: int,
    *,
    name: str = "values",
) -> np.ndarray:
    """Return a ``(coordinate_dim, point_count)`` matrix for point-data adapters."""

    coordinate_dim = int(coordinate_dim)
    point_count = int(point_count)
    return as_boundary_tensor(values, (coordinate_dim, point_count), name=name)


def as_boundary_chunk_tensor(
    values: ArrayLike,
    leading_dim: int,
    quadrature_order: int,
    chunk_count: int,
    *,
    name: str = "values",
) -> np.ndarray:
    """Return a ``(leading_dim, quadrature_order, chunk_count)`` chunk tensor."""

    return as_boundary_tensor(
        values,
        (int(leading_dim), int(quadrature_order), int(chunk_count)),
        name=name,
    )


def as_boundary_weight_matrix(
    values: ArrayLike,
    quadrature_order: int,
    chunk_count: int,
    *,
    name: str = "weights",
) -> np.ndarray:
    """Return a ``(quadrature_order, chunk_count)`` chunk weight matrix."""

    return as_boundary_tensor(values, (int(quadrature_order), int(chunk_count)), name=name)


def as_boundary_field_matrix(
    values: ArrayLike,
    component_count: int,
    point_count: int,
    *,
    name: str = "values",
) -> np.ndarray:
    """Return a ``(component_count, point_count)`` field matrix from boundary data."""

    return as_boundary_point_matrix(values, component_count, point_count, name=name)


def boundary_matrix_from_kernel_tensor(
    values: ArrayLike, *, name: str = "kernel values"
) -> np.ndarray:
    """Materialize ``kernel_values[f, t, d, s]`` as a boundary matrix."""

    arr = np.asarray(values)
    if arr.ndim != 4:
        raise ValueError(f"{name} must have shape (f, t, d, s)")
    output_components, target_count, input_components, source_count = arr.shape
    return as_boundary_tensor(
        arr,
        (output_components * target_count, input_components * source_count),
        name=name,
    )


def boundary_component_weights(weights: ArrayLike, component_count: int) -> np.ndarray:
    """Repeat point weights for component-interleaved boundary vectors."""

    component_count = int(component_count)
    if component_count < 1:
        raise ValueError("component_count must be positive")
    return np.repeat(as_boundary_vector(weights, name="weights"), component_count)


def weighted_density_for_boundary(
    density: ArrayLike,
    weights: ArrayLike,
    point_count: int,
    *,
    name: str = "density",
) -> np.ndarray:
    """Return a flat solver/backend density vector multiplied by point weights."""

    point_count = int(point_count)
    density_vector = as_boundary_vector(density, name=name)
    if point_count < 1:
        raise ValueError("point_count must be positive")
    if density_vector.size % point_count != 0:
        raise ValueError(f"{name} has incompatible size")
    component_count = density_vector.size // point_count
    if component_count == 1:
        return density_vector * as_boundary_vector(
            weights, expected_size=point_count, name="weights"
        )
    return density_vector * boundary_component_weights(weights, component_count)


def density_matmul_argument(
    values: ArrayLike, expected_rows: int, *, name: str = "density"
) -> np.ndarray:
    """Return a vector or RHS matrix suitable for ``LinearOperator`` matmul."""

    expected_rows = int(expected_rows)
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[0] == expected_rows:
        return arr
    return as_boundary_vector(arr, expected_size=expected_rows, name=name)
