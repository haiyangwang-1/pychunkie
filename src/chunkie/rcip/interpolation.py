"""RCIP density interpolation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def interpolate_density(values: NDArray[np.generic], prolongation: NDArray[np.floating]) -> NDArray[np.generic]:
    """Interpolate scalar or component density values with a prolongation matrix."""

    density = np.asarray(values)
    matrix = np.asarray(prolongation, dtype=float)
    if density.ndim == 1:
        return matrix @ density
    if density.ndim == 2:
        return np.einsum("ts,cs->ct", matrix, density)
    raise ValueError("density values must be rank 1 or rank 2")
