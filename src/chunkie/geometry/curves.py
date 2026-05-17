"""Small reusable curve callbacks."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def unit_circle(theta: NDArray[np.floating]):
    positions = np.vstack((np.cos(theta), np.sin(theta)))
    derivatives = np.vstack((-np.sin(theta), np.cos(theta)))
    second = -positions
    return positions, derivatives, second


def line_segment(
    parameter: ArrayLike,
    start: ArrayLike,
    end: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Line segment parameterized by ``parameter in [0, 1]``."""

    parameter_array = np.asarray(parameter, dtype=float)
    flat = parameter_array.reshape(-1)
    start_array = np.asarray(start, dtype=float).reshape(2, 1)
    end_array = np.asarray(end, dtype=float).reshape(2, 1)
    delta = end_array - start_array
    positions = start_array + delta * flat[None, :]
    derivatives = delta * np.ones((1, flat.size))
    second_derivatives = np.zeros_like(derivatives)
    return _pack_from_flat(
        parameter_array,
        positions[0],
        positions[1],
        derivatives[0],
        derivatives[1],
        second_derivatives[0],
        second_derivatives[1],
    )


def parabola(
    parameter: ArrayLike,
    curvature: float,
    center: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Graph ``y = curvature * (x - center)**2`` with ``x = parameter``."""

    parameter_array = np.asarray(parameter, dtype=float)
    flat = parameter_array.reshape(-1)
    x = flat
    y = float(curvature) * (flat - float(center)) ** 2
    dx = np.ones_like(flat)
    dy = 2.0 * float(curvature) * (flat - float(center))
    d2x = np.zeros_like(flat)
    d2y = np.full_like(flat, 2.0 * float(curvature))
    return _pack_from_flat(parameter_array, x, y, dx, dy, d2x, d2y)


def sine_graph(
    parameter: ArrayLike,
    amplitude: float,
    wavenumber: float,
    phase: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Graph ``y = amplitude * sin(wavenumber*x + phase)``."""

    parameter_array = np.asarray(parameter, dtype=float)
    flat = parameter_array.reshape(-1)
    phase_values = float(wavenumber) * flat + float(phase)
    x = flat
    y = float(amplitude) * np.sin(phase_values)
    dx = np.ones_like(flat)
    dy = float(amplitude) * float(wavenumber) * np.cos(phase_values)
    d2x = np.zeros_like(flat)
    d2y = -float(amplitude) * float(wavenumber) ** 2 * np.sin(phase_values)
    return _pack_from_flat(parameter_array, x, y, dx, dy, d2x, d2y)


def starfish(
    theta: ArrayLike,
    arm_count: int = 5,
    amplitude: float = 0.3,
    center: ArrayLike | None = None,
    phase: float = 0.0,
    scale: float = 1.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Starfish polar curve and its first two parameter derivatives."""

    theta_array = np.asarray(theta, dtype=float)
    flat = theta_array.reshape(-1)
    center_array = np.zeros(2) if center is None else np.asarray(center, dtype=float).reshape(2)
    arms = int(arm_count)
    amp = float(amplitude)
    scale_value = float(scale)

    cos_theta = np.cos(flat)
    sin_theta = np.sin(flat)
    cos_arms = np.cos(arms * (flat + float(phase)))
    sin_arms = np.sin(arms * (flat + float(phase)))
    radius = 1.0 + amp * cos_arms
    radius_prime = -arms * amp * sin_arms
    radius_second = -(arms**2) * amp * cos_arms

    # The Cartesian derivatives are the product rule applied to
    # scale * radius(theta) * [cos(theta), sin(theta)].
    x = center_array[0] + scale_value * radius * cos_theta
    y = center_array[1] + scale_value * radius * sin_theta
    dx = scale_value * (radius_prime * cos_theta - radius * sin_theta)
    dy = scale_value * (radius_prime * sin_theta + radius * cos_theta)
    d2x = scale_value * (
        radius_second * cos_theta - 2.0 * radius_prime * sin_theta - radius * cos_theta
    )
    d2y = scale_value * (
        radius_second * sin_theta + 2.0 * radius_prime * cos_theta - radius * sin_theta
    )
    return _pack_from_flat(theta_array, x, y, dx, dy, d2x, d2y)


def fourier_radius(
    theta: ArrayLike,
    modes: ArrayLike,
    center: ArrayLike | None = None,
    scale: ArrayLike | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Polar curve with real Fourier coefficients for the radius."""

    theta_array = np.asarray(theta, dtype=float)
    flat = theta_array.reshape(-1)
    modes_array = np.asarray(modes, dtype=float).reshape(-1)
    center_array = np.zeros(2) if center is None else np.asarray(center, dtype=float).reshape(2)
    scale_array = np.ones(2) if scale is None else np.asarray(scale, dtype=float).reshape(2)

    radius = modes_array[0] * np.ones_like(flat)
    radius_prime = np.zeros_like(flat)
    radius_second = np.zeros_like(flat)
    for cosine_index in range(1, len(modes_array), 2):
        harmonic = (cosine_index + 1) // 2
        angle = harmonic * flat
        cos_h = np.cos(angle)
        sin_h = np.sin(angle)
        radius += modes_array[cosine_index] * cos_h
        radius_prime -= harmonic * modes_array[cosine_index] * sin_h
        radius_second -= harmonic**2 * modes_array[cosine_index] * cos_h
        sine_index = cosine_index + 1
        if sine_index < len(modes_array):
            radius += modes_array[sine_index] * sin_h
            radius_prime += harmonic * modes_array[sine_index] * cos_h
            radius_second -= harmonic**2 * modes_array[sine_index] * sin_h

    cos_theta = np.cos(flat)
    sin_theta = np.sin(flat)
    x = center_array[0] + scale_array[0] * radius * cos_theta
    y = center_array[1] + scale_array[1] * radius * sin_theta
    dx = scale_array[0] * (radius_prime * cos_theta - radius * sin_theta)
    dy = scale_array[1] * (radius_prime * sin_theta + radius * cos_theta)
    d2x = scale_array[0] * (
        radius_second * cos_theta - 2.0 * radius_prime * sin_theta - radius * cos_theta
    )
    d2y = scale_array[1] * (
        radius_second * sin_theta + 2.0 * radius_prime * cos_theta - radius * sin_theta
    )
    return _pack_from_flat(theta_array, x, y, dx, dy, d2x, d2y)


# Legacy MATLAB-inspired names stay local to this utility module so archived
# fixtures can be migrated without reintroducing old names at the package facade.
linefunc = line_segment
fpara = parabola
fsine = sine_graph
bymode = fourier_radius


def _pack_from_flat(
    parameter: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    dx: NDArray[np.floating],
    dy: NDArray[np.floating],
    d2x: NDArray[np.floating],
    d2y: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    shape = (2,) + parameter.shape
    positions = np.vstack((x, y)).reshape(shape)
    derivatives = np.vstack((dx, dy)).reshape(shape)
    second_derivatives = np.vstack((d2x, d2y)).reshape(shape)
    return positions, derivatives, second_derivatives
