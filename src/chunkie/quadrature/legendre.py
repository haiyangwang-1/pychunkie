"""Legendre polynomial, interpolation, and panel-node utilities."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray


@cache
def legendre_rule(quadrature_order: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    nodes, weights = leggauss(int(quadrature_order))
    return nodes, weights


def pol(points: ArrayLike, degree: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Evaluate one Legendre polynomial and derivative."""

    if degree < 0:
        raise ValueError("degree must be non-negative")
    points_array = np.asarray(points, dtype=float)
    if degree == 0:
        return np.ones_like(points_array), np.zeros_like(points_array)
    if degree == 1:
        return points_array.copy(), np.ones_like(points_array)

    previous = np.ones_like(points_array)
    current = points_array.copy()
    for order in range(1, degree):
        next_value = ((2 * order + 1) * points_array * current - order * previous) / (order + 1)
        previous, current = current, next_value
    derivative = _legendre_derivative(points_array, current, previous, degree)
    return current, derivative


def pols(points: ArrayLike, max_degree: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Evaluate Legendre polynomials through ``max_degree`` and derivatives."""

    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    points_array = np.asarray(points, dtype=float)
    flat = points_array.reshape(-1)
    values = np.zeros((max_degree + 1, flat.size), dtype=float)
    derivatives = np.zeros_like(values)
    values[0] = 1.0
    if max_degree >= 1:
        values[1] = flat
        derivatives[1] = 1.0
    for degree in range(1, max_degree):
        values[degree + 1] = (
            (2 * degree + 1) * flat * values[degree] - degree * values[degree - 1]
        ) / (degree + 1)
    for degree in range(2, max_degree + 1):
        derivatives[degree] = _legendre_derivative(flat, values[degree], values[degree - 1], degree)
    out_shape = (max_degree + 1,) + points_array.shape
    return values.reshape(out_shape), derivatives.reshape(out_shape)


def exps(
    quadrature_order: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return Gaussian nodes, weights, and coefficient/value transforms."""

    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive")
    nodes, weights = legendre_rule(int(quadrature_order))
    values, _ = pols(nodes, quadrature_order - 1)
    value_transform = values.T
    scales = (2.0 * np.arange(1, quadrature_order + 1) - 1.0) / 2.0
    coefficient_transform = (value_transform * (weights[:, None] * scales[None, :])).T
    return nodes, weights, coefficient_transform, value_transform


def rts(quadrature_order: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return Legendre-Gauss nodes and weights."""

    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive")
    return legendre_rule(int(quadrature_order))


def rts_stab(quadrature_order: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Stable Legendre-Gauss nodes and weights."""

    return rts(quadrature_order)


def exev(points: ArrayLike, coefficients: ArrayLike) -> NDArray[np.generic]:
    """Evaluate one or more Legendre expansions."""

    points_array = np.asarray(points, dtype=float)
    coefficient_array = np.asarray(coefficients)
    degree = coefficient_array.shape[0] - 1
    values, _ = pols(points_array, degree)
    return np.moveaxis(values, 0, -1) @ coefficient_array


def derpol(coefficients: ArrayLike) -> NDArray[np.generic]:
    """Return Legendre coefficients of the derivative expansion."""

    coefficient_array = np.asarray(coefficients)
    if coefficient_array.shape[0] == 0:
        return np.zeros_like(coefficient_array)
    out = np.zeros(
        (max(coefficient_array.shape[0] - 1, 0),) + coefficient_array.shape[1:],
        dtype=coefficient_array.dtype,
    )
    for degree in range(1, coefficient_array.shape[0]):
        for target in range(degree - 1, -1, -2):
            out[target] += (2 * target + 1) * coefficient_array[degree]
    return out


def dermat(
    quadrature_order: int,
    coefficient_transform: ArrayLike | None = None,
    value_transform: ArrayLike | None = None,
) -> NDArray[np.generic]:
    """Return the spectral differentiation matrix on Legendre nodes."""

    if coefficient_transform is None or value_transform is None:
        _, _, coefficient_array, value_array = exps(quadrature_order)
    else:
        coefficient_array = np.asarray(coefficient_transform)
        value_array = np.asarray(value_transform)
    return value_array[:, :-1] @ derpol(coefficient_array)


def intpol(coefficients: ArrayLike, const_option: str = "true") -> NDArray[np.generic]:
    """Return Legendre coefficients of the integral from the left endpoint."""

    coefficient_array = np.asarray(coefficients)
    out = np.zeros(
        (coefficient_array.shape[0] + 1,) + coefficient_array.shape[1:],
        dtype=coefficient_array.dtype,
    )
    option = const_option.lower()
    if option == "true":
        constant_count = out.shape[0]
    elif option == "original":
        constant_count = out.shape[0] - 1
    else:
        raise ValueError("unknown option for integration constant")

    for index in range(1, coefficient_array.shape[0]):
        out[index + 1] = coefficient_array[index] / (2 * index + 1)
        out[index - 1] += -coefficient_array[index] / (2 * index + 1)
    out[1] += coefficient_array[0]

    sign = -1.0
    accum = np.zeros_like(out[-1])
    for index in range(1, constant_count):
        accum += out[index] * sign
        sign = -sign
    out[0] = -accum
    return out


def intmat(
    quadrature_order: int,
    coefficient_transform: ArrayLike | None = None,
    value_transform: ArrayLike | None = None,
) -> tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.generic]]:
    """Return the spectral integration matrix on Legendre nodes."""

    if coefficient_transform is None or value_transform is None:
        _, _, coefficient_array, value_array = exps(quadrature_order)
    else:
        coefficient_array = np.asarray(coefficient_transform)
        value_array = np.asarray(value_transform)
    integral_coefficients = intpol(coefficient_array, "original")
    return value_array @ integral_coefficients[:-1], coefficient_array, value_array


def matrin(
    quadrature_order: int,
    points: ArrayLike,
    coefficient_transform: ArrayLike | None = None,
) -> tuple[
    NDArray[np.generic],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.generic],
    NDArray[np.floating],
]:
    """Return the interpolation matrix from Legendre nodes to ``points``."""

    nodes, weights, default_coefficient_transform, value_transform = exps(quadrature_order)
    coefficient_array = (
        default_coefficient_transform
        if coefficient_transform is None
        else np.asarray(coefficient_transform)
    )
    values, _ = pols(points, quadrature_order - 1)
    matrix = np.moveaxis(values, 0, -1) @ coefficient_array
    return matrix, nodes, weights, coefficient_array, value_transform


def barycentric_weights(nodes: ArrayLike) -> NDArray[np.floating]:
    """Return barycentric Lagrange weights for interpolation at ``nodes``."""

    node_array = np.asarray(nodes, dtype=float).reshape(-1)
    differences = node_array[:, None] - node_array[None, :]
    np.fill_diagonal(differences, 1.0)
    return 1.0 / np.prod(differences, axis=1)


def interpolation_matrix(nodes: ArrayLike, targets: ArrayLike) -> NDArray[np.generic]:
    """Return the barycentric interpolation matrix from ``nodes`` to ``targets``."""

    node_array = np.asarray(nodes, dtype=float).reshape(-1)
    raw_targets = np.asarray(targets).reshape(-1)
    target_array = raw_targets.astype(np.result_type(raw_targets.dtype, float), copy=False)
    weights = barycentric_weights(node_array)
    matrix = np.empty(
        (target_array.size, node_array.size), dtype=np.result_type(target_array.dtype, float)
    )
    for row, target in enumerate(target_array):
        differences = target - node_array
        exact = np.isclose(differences, 0.0, atol=1.0e-15, rtol=0.0)
        if np.any(exact):
            matrix[row] = 0.0
            matrix[row, np.argmax(exact)] = 1.0
            continue
        scaled = weights / differences
        matrix[row] = scaled / np.sum(scaled)
    return matrix


bary_weights = barycentric_weights


def barywts(quadrature_order: int, points: ArrayLike | None = None) -> NDArray[np.floating]:
    """Return barycentric Lagrange weights normalized by the first node."""

    nodes = (
        legendre_rule(quadrature_order)[0] if points is None else np.asarray(points, dtype=float)
    )
    differences = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(differences, 1.0)
    weights = np.prod(differences, axis=0)
    return weights[0] / weights


def bernstein_ellipse(point_count: int, rho: float) -> NDArray[np.complexfloating]:
    """Return points on the Bernstein ellipse with parameter ``rho``."""

    if point_count <= 0:
        raise ValueError("point_count must be positive")
    if rho <= 0:
        raise ValueError("rho must be positive")
    theta = np.linspace(0.0, 2.0 * np.pi, point_count + 1)[:-1]
    z = rho * np.exp(1j * theta)
    return 0.5 * (z + 1.0 / z)


def polsum(
    points: ArrayLike, degree: int
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return ``P_degree``, derivative, and the weighted recurrence sum."""

    values, derivatives = pols(points, degree)
    weights = np.arange(degree + 1, dtype=float)[:, None] + 0.5
    flat_total = np.sum(values.reshape(degree + 1, -1) ** 2 * weights, axis=0)
    return values[degree], derivatives[degree], flat_total.reshape(np.asarray(points).shape)


def tayl(
    polynomial_value: ArrayLike,
    derivative_value: ArrayLike,
    points: ArrayLike,
    step: ArrayLike,
    degree: int,
    taylor_order: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Evaluate a Legendre polynomial Taylor step from ``points`` to ``points+step``."""

    if degree < 0:
        raise ValueError("degree must be non-negative")
    if taylor_order < 0:
        raise ValueError("taylor_order must be non-negative")
    polynomial_array, derivative_array, points_array, step_array = np.broadcast_arrays(
        np.asarray(polynomial_value, dtype=float),
        np.asarray(derivative_value, dtype=float),
        np.asarray(points, dtype=float),
        np.asarray(step, dtype=float),
    )
    out_polynomial = np.empty_like(polynomial_array, dtype=float)
    out_derivative = np.empty_like(polynomial_array, dtype=float)
    zero_step = step_array == 0.0
    if np.any(zero_step):
        out_polynomial[zero_step] = polynomial_array[zero_step]
        out_derivative[zero_step] = derivative_array[zero_step]
    if not np.all(zero_step):
        active = ~zero_step
        new_polynomial, new_derivative = _tayl_nonzero_step(
            polynomial_array[active],
            derivative_array[active],
            points_array[active],
            step_array[active],
            degree,
            taylor_order,
        )
        out_polynomial[active] = new_polynomial
        out_derivative[active] = new_derivative
    return out_polynomial, out_derivative


def adapgauss(
    function: Callable[[NDArray[np.floating]], ArrayLike],
    left: float,
    right: float,
    nodes: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> tuple[NDArray[np.generic], int, int, int]:
    """Adaptive Gauss-Legendre integration for scalar or vector functions."""

    if nodes is None or weights is None:
        node_array, weight_array = exps(16)[:2]
    else:
        node_array = np.asarray(nodes, dtype=float).reshape(-1)
        weight_array = np.asarray(weights, dtype=float).reshape(-1)
    if node_array.shape != weight_array.shape:
        raise ValueError("nodes and weights must have the same shape")

    tolerance = 1.0e-12
    max_intervals = 100000
    max_depth = 200
    stack = np.zeros((max_depth, 2), dtype=float)
    values: list[NDArray[np.generic] | None] = [None] * max_depth
    stack[0] = (float(left), float(right))
    values[0] = _one_interval(function, float(left), float(right), node_array, weight_array)
    total = np.zeros_like(values[0])
    depth = 0
    max_recursion = 0

    for interval_count in range(1, max_intervals + 1):
        max_recursion = max(max_recursion, depth + 1)
        interval_left, interval_right = stack[depth]
        midpoint = 0.5 * (interval_left + interval_right)
        left_value = _one_interval(function, interval_left, midpoint, node_array, weight_array)
        right_value = _one_interval(function, midpoint, interval_right, node_array, weight_array)
        current = values[depth]
        assert current is not None
        if np.all(np.abs(left_value + right_value - current) <= tolerance):
            total = total + left_value + right_value
            values[depth] = None
            depth -= 1
            if depth < 0:
                return total, max_recursion, interval_count, 0
            continue
        if depth + 1 >= max_depth:
            return total, max_recursion, interval_count, 8
        stack[depth + 1] = (interval_left, midpoint)
        values[depth + 1] = left_value
        stack[depth] = (midpoint, interval_right)
        values[depth] = right_value
        depth += 1
    return total, max_recursion, max_intervals, 16


def _legendre_derivative(
    points: NDArray[np.floating],
    current: NDArray[np.floating],
    previous: NDArray[np.floating],
    degree: int,
) -> NDArray[np.floating]:
    # The stable interior formula divides by x^2 - 1, so endpoint values use
    # the closed-form derivative limit P_n'(+-1)= (+-1)^(n+1) n(n+1)/2.
    derivative = np.empty_like(points, dtype=float)
    endpoint = np.isclose(np.abs(points), 1.0)
    derivative[~endpoint] = (
        degree
        * (points[~endpoint] * current[~endpoint] - previous[~endpoint])
        / (points[~endpoint] ** 2 - 1.0)
    )
    if np.any(endpoint):
        signs = np.where(points[endpoint] >= 0.0, 1.0, (-1.0) ** (degree + 1))
        derivative[endpoint] = signs * degree * (degree + 1) / 2.0
    return derivative


def _tayl_nonzero_step(
    polynomial_array: NDArray[np.floating],
    derivative_array: NDArray[np.floating],
    points_array: NDArray[np.floating],
    step_array: NDArray[np.floating],
    degree: int,
    taylor_order: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    # Taylor coefficients are generated from Legendre's differential equation;
    # this keeps high-order stepping local without converting through monomials.
    q0 = polynomial_array
    q1 = derivative_array * step_array
    q2 = (2 * points_array * derivative_array - degree * (degree + 1) * polynomial_array) / (
        1 - points_array**2
    )
    q2 = q2 * step_array**2 / 2.0
    polynomial_new = q0 + q1 + q2
    derivative_new = q1 / step_array + (2.0 * q2) / step_array
    if taylor_order <= 2:
        return polynomial_new, derivative_new

    qi = q1
    qip1 = q2
    for order in range(1, taylor_order - 1):
        delta = 2 * (points_array * (order + 1) ** 2) / step_array * qip1
        delta = delta - (degree * (degree + 1) - order * (order + 1)) * qi
        delta = delta / (order + 1) / (order + 2) * step_array**2 / (1 - points_array**2)
        polynomial_new = polynomial_new + delta
        derivative_new = derivative_new + delta * (order + 2) / step_array
        qi = qip1
        qip1 = delta
    return polynomial_new, derivative_new


def _one_interval(
    function: Callable[[NDArray[np.floating]], ArrayLike],
    left: float,
    right: float,
    nodes: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> NDArray[np.generic]:
    scale = (right - left) / 2.0
    shift = (right + left) / 2.0
    mapped_nodes = scale * nodes + shift
    values = np.asarray(function(mapped_nodes))
    if values.shape == ():
        return values * np.sum(weights) * scale
    if values.shape[0] == nodes.size:
        return np.tensordot(weights, values, axes=(0, 0)) * scale
    if values.shape[-1] == nodes.size:
        return np.tensordot(values, weights, axes=(-1, 0)) * scale
    raise ValueError("integrand output must have one axis matching nodes")
