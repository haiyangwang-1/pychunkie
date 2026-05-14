from __future__ import annotations

import numpy as np
import pytest

from chunkie._layout import (
    as_boundary_chunk_tensor,
    as_boundary_field_matrix,
    as_boundary_point_matrix,
    as_boundary_tensor,
    as_boundary_vector,
    as_boundary_weight_matrix,
    boundary_matrix_from_kernel_tensor,
    density_matmul_argument,
    weighted_density_for_boundary,
)


def test_boundary_vector_makes_flat_adapter_order_explicit():
    tensor = np.arange(12).reshape(2, 3, 2)

    actual = as_boundary_vector(tensor)

    np.testing.assert_array_equal(actual, np.array([0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11]))


def test_boundary_point_and_field_matrices_restore_point_axes():
    tensor = np.arange(12).reshape(2, 3, 2)

    point_matrix = as_boundary_point_matrix(tensor, 2, 6)
    field_matrix = as_boundary_field_matrix(tensor, 2, 6)

    expected = np.array([[0, 2, 4, 1, 3, 5], [6, 8, 10, 7, 9, 11]])
    np.testing.assert_array_equal(point_matrix, expected)
    np.testing.assert_array_equal(field_matrix, expected)


def test_boundary_tensor_helpers_restore_chunk_axes():
    flat = np.array([0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11])

    chunk_tensor = as_boundary_chunk_tensor(flat, leading_dim=2, quadrature_order=3, chunk_count=2)
    weights = as_boundary_weight_matrix(np.arange(6), quadrature_order=3, chunk_count=2)

    np.testing.assert_array_equal(chunk_tensor, np.arange(12).reshape(2, 3, 2))
    np.testing.assert_array_equal(weights, np.array([[0, 3], [1, 4], [2, 5]]))
    np.testing.assert_array_equal(as_boundary_tensor(flat, (2, 3, 2)), chunk_tensor)


def test_weighted_density_repeats_point_weights_for_components():
    density = np.arange(12).reshape(2, 3, 2)
    weights = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])

    actual = weighted_density_for_boundary(density, weights, point_count=6)

    expected_weights = np.array([10, 10, 20, 20, 30, 30, 11, 11, 21, 21, 31, 31])
    np.testing.assert_array_equal(actual, as_boundary_vector(density) * expected_weights)


def test_kernel_tensor_materializes_component_interleaved_matrix():
    kxx = np.array([[1.0, 2.0, 3.0]])
    kxy = np.array([[4.0, 5.0, 6.0]])
    kyx = np.array([[7.0, 8.0, 9.0]])
    kyy = np.array([[10.0, 11.0, 12.0]])
    kernel_values = np.empty((2, 1, 2, 3))
    kernel_values[0, :, 0, :] = kxx
    kernel_values[0, :, 1, :] = kxy
    kernel_values[1, :, 0, :] = kyx
    kernel_values[1, :, 1, :] = kyy

    actual = boundary_matrix_from_kernel_tensor(kernel_values)

    expected = np.array([[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], [7.0, 10.0, 8.0, 11.0, 9.0, 12.0]])
    np.testing.assert_array_equal(actual, expected)


def test_boundary_helpers_validate_sizes():
    with pytest.raises(ValueError, match="expected 4"):
        as_boundary_vector([1, 2, 3], expected_size=4, name="density")
    with pytest.raises(ValueError, match="expected 8"):
        as_boundary_point_matrix(np.arange(6), 2, 4, name="positions")
    with pytest.raises(ValueError, match="incompatible size"):
        weighted_density_for_boundary(np.arange(5), np.ones(3), point_count=3)
    with pytest.raises(ValueError, match="must have shape"):
        boundary_matrix_from_kernel_tensor(np.ones((2, 3)))


def test_density_matmul_argument_preserves_multiple_rhs_matrix():
    rhs = np.arange(12).reshape(6, 2)

    assert density_matmul_argument(rhs, 6) is rhs
    np.testing.assert_array_equal(density_matmul_argument(rhs.T, 12), as_boundary_vector(rhs.T))
