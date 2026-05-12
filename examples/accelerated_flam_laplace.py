"""FLAM matrix application and solve for a shifted Laplace operator.

Run from the repository root:

    uv run python examples/accelerated_flam_laplace.py

The dense shifted matrix is used only as a reference check. The FLAM object is
the matrix-free/compressed object a larger solve would use.
"""

from __future__ import annotations

import numpy as np

from chunkie import chunkermat, kernel

from _accelerated_common import make_circle, relerr


def main() -> None:
    chnkr = make_circle()
    lap_s = kernel("lap", "s")
    rhs = np.cos(np.arange(chnkr.npt))

    dense_shifted = chunkermat(chnkr, lap_s) + np.eye(chnkr.npt)
    flam = chunkermat(
        chnkr,
        lap_s,
        {"acceleration": "flam", "dval": 1.0, "occ": 8, "rank_or_tol": 1e-9, "useproxy": False},
    )

    applied = flam @ rhs
    solved = flam.solve(rhs)
    print(f"FLAM shifted Laplace matvec relative error: {relerr(applied, dense_shifted @ rhs):.3e}")
    print(f"FLAM shifted Laplace solve residual: {relerr(dense_shifted @ solved, rhs):.3e}")


if __name__ == "__main__":
    main()
