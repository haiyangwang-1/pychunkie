"""Biharmonic dense reference and current FMM support check."""

from _accelerated_common import TARGETS, make_circle, scalar_density

from chunkie.kernels import kernel
from chunkie.quadrature import apply_panel_potential
from chunkie.system.backends.fmm2d import apply_fmm


def main() -> None:
    boundary = make_circle()
    density = scalar_density(boundary)
    biharmonic_s = kernel("biharmonic", selector="s")
    direct = apply_panel_potential(boundary.pointinfo, TARGETS, biharmonic_s, density)

    print(f"Biharmonic dense single layer values: {direct.reshape(-1)}")
    try:
        apply_fmm(boundary.pointinfo, TARGETS, biharmonic_s, density, eps=1.0e-11)
    except NotImplementedError as exc:
        print(f"Biharmonic FMM backend unavailable: {exc}")


if __name__ == "__main__":
    main()
