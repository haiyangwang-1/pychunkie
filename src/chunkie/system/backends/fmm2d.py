"""FMM2D backend adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from chunkie.kernels import Kernel


def apply_fmm(
    source: Any,
    target: Any,
    kernel: Kernel,
    density: NDArray[np.generic],
    *,
    eps: float = 1.0e-12,
) -> NDArray[np.generic]:
    """Evaluate supported layer potentials with FMM2D.

    The adapter consumes weighted scalar densities in the project layout and
    maps them to FMM2D charge or dipole strengths. Unsupported kernels fail
    clearly instead of silently falling back to dense evaluation.
    """

    if kernel.family != "laplace" or kernel.selector not in {"s", "d"}:
        raise NotImplementedError("FMM backend currently supports Laplace single and double layers")
    if kernel.input_dim != 1 or kernel.output_dim != 1:
        raise NotImplementedError("FMM backend currently supports scalar Laplace kernels")

    try:
        import fmm2dpy
    except ImportError as exc:  # pragma: no cover - dependency is installed in the project env.
        raise RuntimeError("fmm2dpy is required for FMM evaluation") from exc

    source_positions = _flat_positions(source)
    target_positions = _flat_positions(target)
    weighted_density = _flat_scalar_density(density, source) * _flat_weights(source)

    if kernel.selector == "s":
        result = fmm2dpy.lfmm2d(
            eps=float(eps),
            sources=source_positions,
            charges=weighted_density,
            targets=target_positions,
            pgt=1,
        )
    else:
        result = fmm2dpy.lfmm2d(
            eps=float(eps),
            sources=source_positions,
            dipstr=weighted_density,
            dipvec=_flat_normals(source),
            targets=target_positions,
            pgt=1,
        )

    # FMM2D's Laplace convention returns log(r); chunkie kernels use
    # -log(r)/(2*pi), so the scale is shared by charges and dipoles.
    values = -np.asarray(result.pottarg).reshape(1, -1) / (2.0 * np.pi)
    return np.real_if_close(values)


def _flat_scalar_density(density: NDArray[np.generic], source: Any) -> NDArray[np.generic]:
    arr = np.asarray(density)
    point_count = _flat_positions(source).shape[1]
    if arr.ndim == 1:
        flat = arr.reshape(-1)
    elif arr.ndim == 2:
        flat = arr.T.reshape(-1)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        flat = arr[0].T.reshape(-1)
    else:
        raise ValueError("FMM density must be scalar with shape (point,), (local, panel), or (1, local, panel)")
    if flat.size != point_count:
        raise ValueError("FMM density size does not match source point count")
    return flat


def _flat_positions(points: Any) -> NDArray[np.floating]:
    if hasattr(points, "flat_positions"):
        return np.asarray(points.flat_positions, dtype=float)
    if hasattr(points, "positions"):
        positions = np.asarray(points.positions, dtype=float)
        if positions.ndim == 3:
            return positions.swapaxes(1, 2).reshape(positions.shape[0], -1)
        return positions.reshape(positions.shape[0], -1)
    arr = np.asarray(points, dtype=float)
    return arr.reshape(arr.shape[0], -1)


def _flat_normals(points: Any) -> NDArray[np.floating]:
    if hasattr(points, "flat_normals"):
        return np.asarray(points.flat_normals, dtype=float)
    if hasattr(points, "normals"):
        normals = np.asarray(points.normals, dtype=float)
        if normals.ndim == 3:
            return normals.swapaxes(1, 2).reshape(normals.shape[0], -1)
        return normals.reshape(normals.shape[0], -1)
    raise ValueError("source normals are required for FMM double-layer evaluation")


def _flat_weights(points: Any) -> NDArray[np.floating]:
    if hasattr(points, "flat_weights"):
        return np.asarray(points.flat_weights, dtype=float)
    if hasattr(points, "weights"):
        weights = np.asarray(points.weights, dtype=float)
        return weights.T.reshape(-1) if weights.ndim == 2 else weights.reshape(-1)
    raise ValueError("source weights are required for FMM evaluation")
