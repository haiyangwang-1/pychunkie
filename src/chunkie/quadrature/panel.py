"""Dense reference panel quadrature."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chunkie.geometry import PointInfoView
from chunkie.kernels import Kernel


def dense_panel_matrix(source: PointInfoView, target, kernel: Kernel) -> NDArray[np.generic]:
    values = kernel(source, target)
    weights = source.flat_weights
    # Kernel values are unweighted. Matrix assembly applies quadrature weights
    # only at this boundary so kernel formulas stay reusable for diagnostics.
    return values * weights[None, None, None, :]


def dense_panel_operator_matrix(source: PointInfoView, target, kernel: Kernel) -> NDArray[np.generic]:
    values = dense_panel_matrix(source, target, kernel)
    return operator_matrix_from_weighted_kernel(values)


def operator_matrix_from_weighted_kernel(values: NDArray[np.generic]) -> NDArray[np.generic]:
    """Materialize ``values[o, i, t, s]`` into solver matrix layout.

    Solver vectors are component-major over panel-major points, so row
    ``o*target_count + target`` and column ``i*source_count + source`` are the
    explicit dense adapter boundary.
    """

    arr = np.asarray(values)
    if arr.ndim != 4:
        raise ValueError("weighted kernel values must have shape (output, input, target, source)")
    output_dim, input_dim, target_count, source_count = arr.shape
    return arr.transpose(0, 2, 1, 3).reshape(output_dim * target_count, input_dim * source_count)


def apply_panel_potential(
    source: PointInfoView,
    target,
    kernel: Kernel,
    density: NDArray[np.generic],
) -> NDArray[np.generic]:
    values = kernel(source, target)
    density_array = np.asarray(density)
    if density_array.ndim == 3:
        flat_density = density_array.swapaxes(1, 2).reshape(density_array.shape[0], -1)
    elif density_array.ndim == 2:
        flat_density = density_array
    else:
        raise ValueError("density must have shape (component, point) or (component, local_node, panel)")
    return np.einsum("oits,is,s->ot", values, flat_density, source.flat_weights)
