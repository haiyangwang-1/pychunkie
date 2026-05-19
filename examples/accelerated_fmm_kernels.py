"""FMM target evaluation for supported physics kernels."""

from __future__ import annotations

import numpy as np

from chunkie.geometry import circle
from chunkie.kernels import Kernel, kernel
from chunkie.quadrature import apply_panel_potential
from chunkie.system.backends.fmm2d import apply_fmm

TARGETS = np.array([[0.1, 1.5, -0.7], [0.2, 0.3, 1.4]])


def scalar_density(boundary):
    return np.cos(boundary.positions[0])[None, :, :]


def stokes_density(boundary):
    return np.stack((np.cos(boundary.positions[0]), np.sin(boundary.positions[1])), axis=0)


def relerr(actual, expected) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0))


def compare_kernel(label: str, kernel_obj: Kernel, density) -> None:
    boundary = circle(quadrature_order=12, panel_count=20)
    direct = apply_panel_potential(boundary.pointinfo, TARGETS, kernel_obj, density(boundary))
    fmm = apply_fmm(boundary.pointinfo, TARGETS, kernel_obj, density(boundary), eps=1.0e-11)
    print(f"{label} FMM relative error: {relerr(fmm, direct):.3e}")


def main() -> None:
    compare_kernel("Laplace single layer", kernel("laplace", selector="s"), scalar_density)
    compare_kernel("Laplace double layer", kernel("laplace", selector="d"), scalar_density)
    compare_kernel(
        "Helmholtz single layer",
        kernel("helmholtz", selector="s", wavenumber=1.4 + 0.1j),
        scalar_density,
    )
    compare_kernel(
        "Helmholtz double layer",
        kernel("helmholtz", selector="d", wavenumber=1.4 + 0.1j),
        scalar_density,
    )
    compare_kernel("Stokes single layer", kernel("stokes", selector="s"), stokes_density)


if __name__ == "__main__":
    main()
