"""FLAM matrix application and solve for a shifted Laplace operator."""

import numpy as np

from chunkie import chunkerfunc, chunkermat, ellipse, kernel


chnkr = chunkerfunc(ellipse, {"nchmin": 6, "eps": 1e-8}, {"k": 8})[0]
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
matvec_error = np.linalg.norm(applied - dense_shifted @ rhs) / max(
    np.linalg.norm(dense_shifted @ rhs), 1.0
)
solve_error = np.linalg.norm(dense_shifted @ solved - rhs) / max(np.linalg.norm(rhs), 1.0)

print(f"FLAM shifted Laplace matvec relative error: {matvec_error:.3e}")
print(f"FLAM shifted Laplace solve residual: {solve_error:.3e}")
