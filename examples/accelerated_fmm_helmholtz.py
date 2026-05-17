"""FMM target evaluation for a Helmholtz single-layer potential."""

from _accelerated_common import TARGETS, make_circle, relerr, scalar_density

from chunkie.kernels import kernel
from chunkie.quadrature import apply_panel_potential
from chunkie.system.backends.fmm2d import apply_fmm


def main() -> None:
    boundary = make_circle()
    density = scalar_density(boundary)
    helmholtz_s = kernel("helmholtz", selector="s", wavenumber=1.4 + 0.1j)

    direct = apply_panel_potential(boundary.pointinfo, TARGETS, helmholtz_s, density)
    fmm = apply_fmm(boundary.pointinfo, TARGETS, helmholtz_s, density, eps=1.0e-11)
    print(f"Helmholtz single layer FMM relative error: {relerr(fmm, direct):.3e}")


if __name__ == "__main__":
    main()
