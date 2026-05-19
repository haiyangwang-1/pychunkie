"""FMM target evaluation for a Helmholtz single-layer potential."""

import numpy as np

from chunkie.geometry import circle
from chunkie.kernels import kernel
from chunkie.quadrature import apply_panel_potential
from chunkie.system.backends.fmm2d import apply_fmm

TARGETS = np.array([[0.1, 1.5, -0.7], [0.2, 0.3, 1.4]])


def relerr(actual, expected) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0))


def main() -> None:
    boundary = circle(quadrature_order=12, panel_count=20)
    density = np.cos(boundary.positions[0])[None, :, :]
    helmholtz_s = kernel("helmholtz", selector="s", wavenumber=1.4 + 0.1j)

    direct = apply_panel_potential(boundary.pointinfo, TARGETS, helmholtz_s, density)
    fmm = apply_fmm(boundary.pointinfo, TARGETS, helmholtz_s, density, eps=1.0e-11)
    print(f"Helmholtz single layer FMM relative error: {relerr(fmm, direct):.3e}")


if __name__ == "__main__":
    main()
