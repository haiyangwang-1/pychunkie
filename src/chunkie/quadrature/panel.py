"""Dense reference panel quadrature."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chunkie.geometry import PanelView, PointInfoView, flagnear
from chunkie.kernels import Kernel

from .adaptive import adaptive_panel_matrix


def dense_panel_matrix(source: PointInfoView, target, kernel: Kernel) -> NDArray[np.generic]:
    values = kernel(source, target)
    weights = source.flat_weights
    # Kernel values are unweighted. Matrix assembly applies quadrature weights
    # only at this boundary so kernel formulas stay reusable for diagnostics.
    return values * weights[None, None, None, :]


def dense_panel_operator_matrix(
    source: PointInfoView, target, kernel: Kernel
) -> NDArray[np.generic]:
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
    *,
    close_correction: bool = False,
    near_rho: float = 1.8,
    tolerance: float = 1.0e-12,
) -> NDArray[np.generic]:
    values = kernel(source, target)
    density_array = np.asarray(density)
    if density_array.ndim == 3:
        flat_density = density_array.swapaxes(1, 2).reshape(density_array.shape[0], -1)
    elif density_array.ndim == 2:
        flat_density = density_array
    else:
        raise ValueError(
            "density must have shape (component, point) or (component, local_node, panel)"
        )
    out = np.einsum("oits,is,s->ot", values, flat_density, source.flat_weights)
    if not close_correction:
        return out
    if density_array.ndim != 3:
        raise ValueError("close correction requires panel-major density values")
    return _apply_close_panel_corrections(
        out,
        source,
        target,
        kernel,
        density_array,
        near_rho=float(near_rho),
        tolerance=float(tolerance),
    )


def _apply_close_panel_corrections(
    values: NDArray[np.generic],
    source: PointInfoView,
    target,
    kernel: Kernel,
    density: NDArray[np.generic],
    *,
    near_rho: float,
    tolerance: float,
) -> NDArray[np.generic]:
    target_points = _flat_target_points(target)
    near_flags = flagnear(source, target_points, rho=near_rho)
    corrected = np.array(values, copy=True)
    for panel_id in range(source.positions.shape[2]):
        target_ids = np.flatnonzero(near_flags[:, panel_id])
        if target_ids.size == 0:
            continue

        panel_source = _panel_source(source, panel_id)
        panel_target = target_points[:, target_ids]
        direct_values = kernel(panel_source, panel_target)
        direct = np.einsum(
            "oits,is,s->ot",
            direct_values,
            density[:, :, panel_id],
            source.weights[:, panel_id],
        )
        replacement_tensor = adaptive_panel_matrix(
            _panel_view(source, panel_id),
            panel_target,
            kernel,
            tolerance=tolerance,
        )
        replacement = np.einsum("oits,is->ot", replacement_tensor, density[:, :, panel_id])
        corrected[:, target_ids] += replacement - direct
    return corrected


def _flat_target_points(target) -> NDArray[np.floating]:
    if hasattr(target, "flat_positions"):
        points = np.asarray(target.flat_positions, dtype=float)
    elif hasattr(target, "positions"):
        positions = np.asarray(target.positions, dtype=float)
        points = (
            positions.swapaxes(1, 2).reshape(positions.shape[0], -1)
            if positions.ndim == 3
            else positions
        )
    else:
        points = np.asarray(target, dtype=float)
    return points.reshape(points.shape[0], -1)


def _panel_view(source: PointInfoView, panel_id: int) -> PanelView:
    return PanelView(
        parent=source,
        panel_id=panel_id,
        positions=source.positions[:, :, panel_id],
        derivatives=source.derivatives[:, :, panel_id],
        second_derivatives=source.second_derivatives[:, :, panel_id],
        normals=source.normals[:, :, panel_id],
        weights=source.weights[:, panel_id],
        nodes=source.nodes,
    )


class _SinglePanelSource:
    def __init__(self, source: PointInfoView, panel_id: int) -> None:
        self.positions = source.positions[:, :, panel_id : panel_id + 1]
        self.derivatives = source.derivatives[:, :, panel_id : panel_id + 1]
        self.second_derivatives = source.second_derivatives[:, :, panel_id : panel_id + 1]
        self.normals = source.normals[:, :, panel_id : panel_id + 1]
        self.weights = source.weights[:, panel_id : panel_id + 1]

    @property
    def flat_positions(self) -> NDArray[np.floating]:
        return self.positions[:, :, 0]

    @property
    def flat_normals(self) -> NDArray[np.floating]:
        return self.normals[:, :, 0]

    @property
    def flat_weights(self) -> NDArray[np.floating]:
        return self.weights[:, 0]


def _panel_source(source: PointInfoView, panel_id: int) -> _SinglePanelSource:
    return _SinglePanelSource(source, panel_id)
