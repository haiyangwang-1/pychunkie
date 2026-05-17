"""FMM target evaluation for a Laplace single-layer potential."""

from _accelerated_common import TARGETS, make_circle, relerr, scalar_density

from chunkie.kernels import kernel
from chunkie.quadrature import apply_panel_potential
from chunkie.system.backends.fmm2d import apply_fmm


def main() -> None:
    boundary = make_circle()
    density = scalar_density(boundary)
    laplace_s = kernel("laplace", selector="s")

    direct = apply_panel_potential(boundary.pointinfo, TARGETS, laplace_s, density)
    fmm = apply_fmm(boundary.pointinfo, TARGETS, laplace_s, density, eps=1.0e-11)
    print(f"Laplace single layer FMM relative error: {relerr(fmm, direct):.3e}")


if __name__ == "__main__":
    main()
