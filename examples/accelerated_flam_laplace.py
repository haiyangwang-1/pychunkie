"""FLAM matrix application and solve for a Laplace boundary system."""

import numpy as np

from chunkie.geometry import circle
from chunkie.system import LaplaceExteriorDirichletSystem
from chunkie.system.backends.flam import factor_system


def main() -> None:
    boundary = circle(quadrature_order=8, panel_count=12)
    system = LaplaceExteriorDirichletSystem(boundary, boundary.positions[0])
    matrix = system.assemble()
    # BoundaryEquation accepts panel-major data; FLAM sees the flattened
    # component-major solver vector used by dense assembly.
    rhs = boundary.positions[0].T.reshape(-1)
    # FLAM currently factors a dense reference matrix. Callback assembly will
    # replace this boundary while preserving the apply/solve contract below.
    flam = factor_system(
        matrix,
        boundary.pointinfo.flat_positions,
        occupancy=16,
        tolerance=1.0e-10,
    )

    dense = matrix.to_dense()
    applied = flam.apply(rhs)
    solved = flam.solve(rhs)
    matvec_error = np.linalg.norm(applied - dense @ rhs) / max(
        np.linalg.norm(dense @ rhs),
        1.0,
    )
    solve_error = np.linalg.norm(dense @ solved - rhs) / max(np.linalg.norm(rhs), 1.0)

    print(f"FLAM Laplace system matvec relative error: {matvec_error:.3e}")
    print(f"FLAM Laplace system solve residual: {solve_error:.3e}")


if __name__ == "__main__":
    main()
