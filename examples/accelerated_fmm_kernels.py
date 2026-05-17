"""FMM target evaluation for supported physics kernels."""

from __future__ import annotations

from _accelerated_common import TARGETS, make_circle, relerr, scalar_density, stokes_density

from chunkie.kernels import Kernel, kernel
from chunkie.quadrature import apply_panel_potential
from chunkie.system.backends.fmm2d import apply_fmm


def compare_kernel(label: str, kernel_obj: Kernel, density) -> None:
    boundary = make_circle()
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
