"""Special scalar helper functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erf


def absconvgauss(
    x: ArrayLike, a: float, b: float, h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth ``a*abs(x)+b`` by convolution with a Gaussian of width ``h``."""

    x_arr = np.asarray(x, dtype=float)
    x2 = x_arr / (np.sqrt(2.0) * h)
    expfac = np.exp(-(x_arr * x_arr) / (2.0 * h * h))
    val = a * x_arr * erf(x2) + np.sqrt(2.0 / np.pi) * a * h * expfac + b
    der = a * erf(x2)
    der2 = a * np.sqrt(2.0 / np.pi) / h * expfac
    return val, der, der2
