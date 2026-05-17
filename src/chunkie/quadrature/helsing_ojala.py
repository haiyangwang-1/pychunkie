"""Helsing-Ojala product quadrature integration."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chunkie.geometry import PanelView
from chunkie.kernels import Kernel


def build_helsing_ojala_panel_matrix(
    panel: PanelView,
    target,
    kernel: Kernel,
    *,
    side: str,
) -> NDArray[np.generic]:
    """Build a Helsing-Ojala corrected panel block from singular metadata."""

    return _singular_panel_matrix(panel, target, kernel, side=side)


def _corrected_singular_panel_matrix(
    panel: PanelView,
    target,
    kernel: Kernel,
    *,
    side: str,
) -> NDArray[np.generic]:
    singular_special = helsing_ojala_singular_matrix(panel, target, kernel, side=side)
    dense_values = kernel(panel, target) * panel.weights[None, None, None, :]
    singular_dense = kernel.singularity.expansion.evaluate(panel, target) * panel.weights[None, None, None, :]
    # The singular amplitudes may be target/source dependent. Helsing-Ojala
    # handles that declared singular part, while the finite remainder stays on
    # ordinary Gauss weights in the original panel basis.
    return singular_special + dense_values - singular_dense


def _singular_panel_matrix(
    panel: PanelView,
    target,
    kernel: Kernel,
    *,
    side: str,
) -> NDArray[np.generic]:
    if not kernel.singularity.expansion.terms:
        return kernel(panel, target) * panel.weights[None, None, None, :]
    return _corrected_singular_panel_matrix(panel, target, kernel, side=side)


def helsing_ojala_log_singular_matrix(
    panel: PanelView,
    target,
    kernel: Kernel,
    *,
    side: str,
) -> NDArray[np.generic]:
    """Build the log-basis singular part declared by ``kernel.singularity``."""

    if any(term.basis.derivative for term in kernel.singularity.expansion.terms):
        raise NotImplementedError("helsing_ojala_log_singular_matrix only accepts log-basis terms")
    return helsing_ojala_singular_matrix(panel, target, kernel, side=side)


def helsing_ojala_singular_matrix(
    panel: PanelView,
    target,
    kernel: Kernel,
    *,
    side: str,
) -> NDArray[np.generic]:
    """Build the declared Laplace-basis singular part with product weights."""

    source = _panel_complex_points(panel)
    source_normal = _panel_complex_normals(panel)
    source_wxp = _panel_complex_speed_weights(panel)
    start, end = _panel_endpoints(panel)
    target_points = _target_complex_points(target)
    max_derivative = max(len(term.basis.derivative) for term in kernel.singularity.expansion.terms)
    special_weights = helsing_ojala_weights(
        target_points,
        source,
        source_normal,
        source_wxp,
        start,
        end,
        side,
        nout=min(max_derivative + 1, 4),
    )
    out = np.zeros((kernel.output_dim, kernel.input_dim, target_points.size, source.size), dtype=complex)
    for term in kernel.singularity.expansion.terms:
        basis_weights = _laplace_basis_product_weights(term.basis.derivative, special_weights)
        amplitude = term.coefficient_values(
            panel,
            target,
            output_dim=kernel.output_dim,
            input_dim=kernel.input_dim,
        )
        out = out + amplitude * basis_weights[None, None, :, :]
    return np.real_if_close(out)


def _laplace_basis_product_weights(
    derivative: tuple[int, ...],
    special_weights: tuple[NDArray[np.complexfloating], ...],
) -> NDArray[np.generic]:
    if len(derivative) == 0:
        return special_weights[0]
    if len(derivative) == 1:
        # For C = i/(2*pi) int ds_y/(y-z), the target-gradient components of
        # G=-(1/2*pi)log|z-y| are (G_x, G_y) = (imag(C), real(C)).
        cauchy = special_weights[1]
        return np.imag(cauchy) if derivative[0] == 0 else np.real(cauchy)
    if len(derivative) == 2:
        # The differentiated Cauchy weight D = i/(2*pi) int ds_y/(y-z)^2
        # encodes the symmetric Hessian as
        # G_xx=imag(D), G_xy=G_yx=real(D), and G_yy=-imag(D).
        first_derivative = special_weights[2]
        a, b = derivative
        if a == 0 and b == 0:
            return np.imag(first_derivative)
        if a == 1 and b == 1:
            return -np.imag(first_derivative)
        return np.real(first_derivative)
    raise NotImplementedError("Helsing-Ojala dispatch supports Laplace bases through second derivatives")


def helsing_ojala_weights(
    target: ArrayLike,
    source: ArrayLike,
    source_normal: ArrayLike,
    source_wxp: ArrayLike,
    start: complex,
    end: complex,
    side: str,
    *,
    nout: int = 4,
) -> tuple[NDArray[np.complexfloating], ...]:
    """Return Helsing-Ojala log, Cauchy, and derivative product weights.

    ``source_wxp`` is the source quadrature weight multiplied by the complex
    panel derivative ``z'(t)``. The returned first matrix integrates the 2D
    Laplace log basis ``-log(|x-y|)/(2*pi)`` against smooth nodal data.
    """

    if nout < 0 or nout > 4:
        raise ValueError("nout must be between 0 and 4")
    x_targ = np.asarray(target, dtype=complex).reshape(-1)
    y_src = np.asarray(source, dtype=complex).reshape(-1)
    normal = np.asarray(source_normal, dtype=complex).reshape(-1)
    wxp_src = np.asarray(source_wxp, dtype=complex).reshape(-1)
    if y_src.size != normal.size or y_src.size != wxp_src.size:
        raise ValueError("source, source_normal, and source_wxp sizes must agree")
    if y_src.size == 0 or x_targ.size == 0:
        return tuple(np.zeros((x_targ.size, y_src.size), dtype=complex) for _ in range(nout))

    zscale = (complex(end) - complex(start)) / 2.0
    zmid = (complex(end) + complex(start)) / 2.0
    source_scaled = (y_src - zmid) / zscale
    target_scaled = (x_targ - zmid) / zscale
    source_count = y_src.size
    target_count = x_targ.size

    cauchy_moments = (1.0 - (-1.0) ** np.arange(1, source_count + 1)) / np.arange(
        1,
        source_count + 1,
    )
    vander = np.ones((source_count, source_count), dtype=complex)
    for index in range(1, source_count):
        vander[:, index] = vander[:, index - 1] * source_scaled

    pvals = np.zeros((source_count + 1, target_count), dtype=complex)
    near = np.abs(target_scaled) <= 1.1
    far = ~near
    gamma = np.exp(1j * np.pi / 4.0)
    if side.lower() == "e":
        gamma = np.conj(gamma)
    elif side.lower() != "i":
        raise ValueError("side must be 'i' or 'e'")

    pvals[0] = np.log(gamma) + np.log((1.0 - target_scaled) / (gamma * (-1.0 - target_scaled)))
    if np.any(near):
        for index in range(source_count):
            pvals[index + 1, near] = target_scaled[near] * pvals[index, near] + cauchy_moments[index]
    if np.any(far):
        xfar = target_scaled[far]
        wxp_scaled = wxp_src / zscale
        pvals[source_count, far] = np.sum(
            (wxp_scaled * source_scaled**source_count)[:, None]
            / (source_scaled[:, None] - xfar[None, :]),
            axis=0,
        )
        for matlab_index in range(source_count, 1, -1):
            pvals[matlab_index - 1, far] = (
                pvals[matlab_index, far] - cauchy_moments[matlab_index - 1]
            ) / xfar

    qvals = np.zeros((source_count, target_count), dtype=complex)
    qvals[0::2] = pvals[1::2] - np.log((1.0 - target_scaled) * (-1.0 - target_scaled))[None, :]
    qvals[1::2] = pvals[2::2] - (
        np.log(gamma) + np.log((1.0 - target_scaled) / (gamma * (-1.0 - target_scaled)))
    )[None, :]
    qvals *= (1.0 / np.arange(1, source_count + 1))[:, None]

    solve_q = np.linalg.solve(vander.T, qvals).T
    log_weights = np.real(solve_q * np.conj(1j * normal)[None, :] * zscale) / (
        2.0 * np.pi * abs(zscale)
    )
    log_weights = log_weights * abs(zscale) - np.log(abs(zscale)) / (2.0 * np.pi) * np.abs(
        wxp_src,
    )[None, :]
    out: list[NDArray[np.complexfloating]] = [log_weights.astype(complex)]
    if nout == 1:
        return tuple(out)

    cauchy = np.linalg.solve(vander.T, pvals[:source_count]).T * (1j / (2.0 * np.pi))
    out.append(cauchy)
    if nout == 2:
        return tuple(out)

    kidx = np.arange(source_count, dtype=float)[:, None]
    signs = (-1.0) ** np.arange(source_count, dtype=float)[:, None]
    rvals = -(1.0 / (1.0 - target_scaled)[None, :] + signs / (1.0 + target_scaled)[None, :])
    rvals += kidx * np.vstack((np.zeros((1, target_count), dtype=complex), pvals[: source_count - 1]))
    first_derivative = np.linalg.solve(vander.T, rvals).T * (1j / (2.0 * np.pi * zscale))
    out.append(first_derivative)
    if nout == 3:
        return tuple(out)

    svals = -(1.0 / (1.0 - target_scaled)[None, :] ** 2 - signs / (1.0 + target_scaled)[None, :] ** 2) / 2.0
    svals += kidx * np.vstack((np.zeros((1, target_count), dtype=complex), rvals[: source_count - 1])) / 2.0
    second_derivative = np.linalg.solve(vander.T, svals).T * (1j / (2.0 * np.pi * zscale**2))
    out.append(second_derivative)
    return tuple(out)


def _panel_complex_points(panel: PanelView) -> NDArray[np.complexfloating]:
    return panel.positions[0] + 1j * panel.positions[1]


def _panel_complex_normals(panel: PanelView) -> NDArray[np.complexfloating]:
    return panel.normals[0] + 1j * panel.normals[1]


def _panel_complex_speed_weights(panel: PanelView) -> NDArray[np.complexfloating]:
    speed = np.linalg.norm(panel.derivatives, axis=0)
    reference_weights = panel.weights / speed
    complex_derivative = panel.derivatives[0] + 1j * panel.derivatives[1]
    return reference_weights * complex_derivative


def _panel_endpoints(panel: PanelView) -> tuple[complex, complex]:
    interpolation = _lagrange_matrix(panel.nodes, np.array([-1.0, 1.0]))
    endpoints = np.einsum("ql,rl->rq", interpolation, panel.positions)
    complex_endpoints = endpoints[0] + 1j * endpoints[1]
    return complex(complex_endpoints[0]), complex(complex_endpoints[1])


def _target_complex_points(target) -> NDArray[np.complexfloating]:
    if hasattr(target, "flat_positions"):
        points = np.asarray(target.flat_positions, dtype=float)
    elif hasattr(target, "positions"):
        positions = np.asarray(target.positions, dtype=float)
        points = positions.swapaxes(1, 2).reshape(positions.shape[0], -1) if positions.ndim == 3 else positions
    else:
        points = np.asarray(target, dtype=float)
    points = points.reshape(points.shape[0], -1)
    if points.shape[0] != 2:
        raise ValueError("Helsing-Ojala quadrature currently supports 2D targets")
    return points[0] + 1j * points[1]


def _lagrange_matrix(nodes: NDArray[np.floating], evaluation_nodes: NDArray[np.floating]) -> NDArray[np.floating]:
    barycentric_weights = _barycentric_weights(nodes)
    matrix = np.empty((evaluation_nodes.size, nodes.size), dtype=float)
    for row, value in enumerate(evaluation_nodes):
        difference = value - nodes
        exact = np.where(np.abs(difference) <= 10.0 * np.finfo(float).eps)[0]
        if exact.size:
            matrix[row] = 0.0
            matrix[row, exact[0]] = 1.0
            continue
        terms = barycentric_weights / difference
        matrix[row] = terms / np.sum(terms)
    return matrix


def _barycentric_weights(nodes: NDArray[np.floating]) -> NDArray[np.floating]:
    weights = np.ones(nodes.size, dtype=float)
    for index, node in enumerate(nodes):
        weights[index] = 1.0 / np.prod(node - np.delete(nodes, index))
    return weights
