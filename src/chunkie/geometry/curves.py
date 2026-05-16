"""Small reusable curve callbacks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def unit_circle(theta: NDArray[np.floating]):
    positions = np.vstack((np.cos(theta), np.sin(theta)))
    derivatives = np.vstack((-np.sin(theta), np.cos(theta)))
    second = -positions
    return positions, derivatives, second
