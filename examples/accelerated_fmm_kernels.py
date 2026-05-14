"""FMM target evaluation for several physics kernels.

Run from the repository root:

    uv run python examples/accelerated_fmm_kernels.py

Each check compares accelerated target evaluation against the direct dense
target-evaluation path.
"""

from __future__ import annotations

import numpy as np
from _accelerated_common import TARGETS, boundary_nodes, component_vector, make_circle, relerr

from chunkie import chunkerkerneval, kernel


def compare_scalar_kernel(boundary, label: str, kernel_obj, density: np.ndarray) -> None:
    direct = chunkerkerneval(boundary, kernel_obj, density, TARGETS).reshape(-1)
    fmm = chunkerkerneval(
        boundary,
        kernel_obj,
        density,
        TARGETS,
        acceleration="fmm",
        tol=1e-11,
    ).reshape(-1)
    print(f"{label} FMM relative error: {relerr(fmm, direct):.3e}")


def main() -> None:
    boundary = make_circle()
    nodes = boundary_nodes(boundary)
    scalar_density = np.cos(nodes[0])

    compare_scalar_kernel(boundary, "Laplace single layer", kernel("lap", "s"), scalar_density)
    compare_scalar_kernel(
        boundary, "Helmholtz single layer", kernel("helm", "s", 1.4 + 0.1j), scalar_density
    )
    compare_scalar_kernel(
        boundary, "Biharmonic single layer", kernel("biharm", "s"), scalar_density
    )

    stokes_density = component_vector(np.vstack((np.cos(nodes[0]), np.sin(nodes[1]))))
    stokes = kernel("stok", "s", 1.0)
    stokes_direct = component_vector(chunkerkerneval(boundary, stokes, stokes_density, TARGETS))
    stokes_fmm = chunkerkerneval(
        boundary,
        stokes,
        stokes_density,
        TARGETS,
        acceleration="fmm",
        tol=1e-11,
    )
    print(
        f"Stokes velocity FMM relative error: {relerr(component_vector(stokes_fmm), stokes_direct):.3e}"
    )


if __name__ == "__main__":
    main()
