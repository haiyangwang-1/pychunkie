"""FLAM backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FLAMFactor:
    factor: Any
    shape: tuple[int, int]
    diagnostics: dict[str, object]

    def apply(self, vector: NDArray[np.generic]) -> NDArray[np.generic]:
        import pyflam

        return pyflam.rskelf_mv(self.factor, vector)

    def solve(self, rhs: NDArray[np.generic]) -> NDArray[np.generic]:
        import pyflam

        return pyflam.rskelf_sv(self.factor, rhs)

    def logdet(self) -> complex:
        import pyflam

        return complex(pyflam.rskelf_logdet(self.factor))


def factor_system(
    matrix,
    points,
    *,
    occupancy: int = 64,
    tolerance: float = 1.0e-10,
    options: dict[str, Any] | None = None,
) -> FLAMFactor:
    """Build a FLAM recursive-skeletonization factor from a dense reference.

    ``points`` are the geometric coordinates used by FLAM to build its tree.
    The adapter starts from dense matrices so apply/solve behavior can be tested
    against the current reference path before callback assembly is introduced.
    """

    try:
        import pyflam
    except ImportError as exc:  # pragma: no cover - dependency is installed in the project env.
        raise RuntimeError("pyflam is required for FLAM factorization") from exc

    dense = np.asarray(matrix.to_dense() if hasattr(matrix, "to_dense") else matrix)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError("FLAM factorization requires a square dense matrix")
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != dense.shape[0]:
        raise ValueError("FLAM points must have shape (dimension, matrix_size)")

    opts = {"verb": False}
    if options is not None:
        opts.update(options)
    factor = pyflam.rskelf(
        dense,
        coords,
        int(occupancy),
        float(tolerance),
        opts=opts,
    )
    return FLAMFactor(
        factor=factor,
        shape=dense.shape,
        diagnostics={
            "backend": "pyflam.rskelf",
            "occupancy": int(occupancy),
            "tolerance": float(tolerance),
        },
    )
