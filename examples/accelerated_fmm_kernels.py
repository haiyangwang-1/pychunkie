"""FMM target evaluation for several physics kernels.

Run from the repository root:

    uv run python examples/accelerated_fmm_kernels.py

Each check compares accelerated target evaluation against the direct dense
target-evaluation path.
"""

from __future__ import annotations

import numpy as np

from chunkie import chunkerkerneval, kernel

from _accelerated_common import TARGETS, boundary_nodes, make_circle, relerr


def compare_scalar_kernel(chnkr, label: str, kern, density: np.ndarray) -> None:
    direct = chunkerkerneval(chnkr, kern, density, TARGETS).reshape(-1, order="F")
    fmm = chunkerkerneval(
        chnkr,
        kern,
        density,
        TARGETS,
        {"acceleration": "fmm", "eps": 1e-11},
    ).reshape(-1, order="F")
    print(f"{label} FMM relative error: {relerr(fmm, direct):.3e}")


def main() -> None:
    chnkr = make_circle()
    nodes = boundary_nodes(chnkr)
    scalar_density = np.cos(nodes[0])

    compare_scalar_kernel(chnkr, "Laplace single layer", kernel("lap", "s"), scalar_density)
    compare_scalar_kernel(chnkr, "Helmholtz single layer", kernel("helm", "s", 1.4 + 0.1j), scalar_density)
    compare_scalar_kernel(chnkr, "Biharmonic single layer", kernel("biharm", "s"), scalar_density)

    stokes_density = np.vstack((np.cos(nodes[0]), np.sin(nodes[1]))).reshape(-1, order="F")
    stokes = kernel("stok", "s", 1.0)
    stokes_direct = chunkerkerneval(chnkr, stokes, stokes_density, TARGETS).reshape(-1, order="F")
    stokes_fmm = chunkerkerneval(
        chnkr,
        stokes,
        stokes_density,
        TARGETS,
        {"acceleration": "fmm", "eps": 1e-11},
    ).reshape(-1, order="F")
    print(f"Stokes velocity FMM relative error: {relerr(stokes_fmm, stokes_direct):.3e}")


if __name__ == "__main__":
    main()
