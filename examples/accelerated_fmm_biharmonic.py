"""Biharmonic dense reference and current FMM support check."""

import numpy as np

from chunkie.geometry import circle
from chunkie.kernels import kernel
from chunkie.quadrature import apply_panel_potential
from chunkie.system.backends.fmm2d import apply_fmm

TARGETS = np.array([[0.1, 1.5, -0.7], [0.2, 0.3, 1.4]])


def main() -> None:
    boundary = circle(quadrature_order=12, panel_count=20)
    density = np.cos(boundary.positions[0])[None, :, :]
    biharmonic_s = kernel("biharmonic", selector="s")
    direct = apply_panel_potential(boundary.pointinfo, TARGETS, biharmonic_s, density)

    print(f"Biharmonic dense single layer values: {direct.reshape(-1)}")
    try:
        apply_fmm(boundary.pointinfo, TARGETS, biharmonic_s, density, eps=1.0e-11)
    except NotImplementedError as exc:
        print(f"Biharmonic FMM backend unavailable: {exc}")


if __name__ == "__main__":
    main()
