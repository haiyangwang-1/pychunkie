"""FMM target evaluation for a Stokes single-layer velocity."""

from _accelerated_common import TARGETS, make_circle, relerr, stokes_density

from chunkie.kernels import kernel
from chunkie.quadrature import apply_panel_potential
from chunkie.system.backends.fmm2d import apply_fmm


def main() -> None:
    boundary = make_circle()
    density = stokes_density(boundary)
    stokes_s = kernel("stokes", selector="s", viscosity=1.0)

    direct = apply_panel_potential(boundary.pointinfo, TARGETS, stokes_s, density)
    fmm = apply_fmm(boundary.pointinfo, TARGETS, stokes_s, density, eps=1.0e-11)
    print(f"Stokes single layer FMM relative error: {relerr(fmm, direct):.3e}")


if __name__ == "__main__":
    main()
