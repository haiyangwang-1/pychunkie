"""Survey FMM and FLAM acceleration across several physics kernels.

Run from the repository root:

    uv run python examples/accelerated_physics_kernels.py

The FMM checks compare target evaluations against the dense/direct path. The
FLAM check builds a shifted single-layer matrix, applies it, and solves with
the compressed `rskelf` factor.
"""

from __future__ import annotations

import numpy as np

from chunkie import chunkerfunc, chunkerkerneval, chunkermat, kernel


def circle(t: np.ndarray):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def relerr(actual, expected):
    return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0)


def main() -> None:
    chnkr, _ = chunkerfunc(circle, {"nchmin": 6, "eps": 1e-8}, {"k": 8})
    pts = chnkr.r.reshape(2, chnkr.npt, order="F")
    targets = np.array([[0.1, 1.5, -0.7], [0.2, 0.3, 1.4]])
    scalar_density = np.cos(pts[0])

    scalar_cases = [
        ("Laplace single layer", kernel("lap", "s")),
        ("Helmholtz single layer", kernel("helm", "s", 1.4 + 0.1j)),
        ("Biharmonic single layer", kernel("biharm", "s")),
    ]
    for label, kern in scalar_cases:
        direct = chunkerkerneval(chnkr, kern, scalar_density, targets).reshape(-1, order="F")
        fmm = chunkerkerneval(
            chnkr,
            kern,
            scalar_density,
            targets,
            {"acceleration": "fmm", "eps": 1e-11},
        ).reshape(-1, order="F")
        print(f"{label} FMM relative error: {relerr(fmm, direct):.3e}")

    stokes_density = np.vstack((np.cos(pts[0]), np.sin(pts[1]))).reshape(-1, order="F")
    stokes = kernel("stok", "s", 1.0)
    stokes_direct = chunkerkerneval(chnkr, stokes, stokes_density, targets).reshape(-1, order="F")
    stokes_fmm = chunkerkerneval(
        chnkr,
        stokes,
        stokes_density,
        targets,
        {"acceleration": "fmm", "eps": 1e-11},
    ).reshape(-1, order="F")
    print(f"Stokes velocity FMM relative error: {relerr(stokes_fmm, stokes_direct):.3e}")

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
