"""Legendre panel nodes and weights."""

from __future__ import annotations

from functools import cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray


@cache
def legendre_rule(quadrature_order: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    nodes, weights = leggauss(int(quadrature_order))
    return nodes, weights
