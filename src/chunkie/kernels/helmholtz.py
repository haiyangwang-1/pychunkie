"""Two-dimensional Helmholtz kernels."""

from __future__ import annotations

import numpy as np
from scipy.special import hankel1

from . import laplace
from .base import Kernel, flat_normals, flat_positions
from .singularities import SingularityInfo


def kernel(selector: str = "s", *, wavenumber: complex) -> Kernel:
    selector = laplace._canonical_selector(selector)
    input_dim, output_dim = laplace._dimensions(selector)
    return Kernel(
        family="helmholtz",
        selector=selector,
        params={"wavenumber": wavenumber},
        input_dim=input_dim,
        output_dim=output_dim,
        singularity=_singularity(selector, wavenumber),
        evaluator=lambda source, target: evaluate(selector, source, target, wavenumber=wavenumber),
    )


def evaluate(selector: str, source, target, *, wavenumber: complex):
    selector = laplace._canonical_selector(selector)
    source_positions = flat_positions(source)
    target_positions = flat_positions(target)
    dx = target_positions[0, :, None] - source_positions[0, None, :]
    dy = target_positions[1, :, None] - source_positions[1, None, :]
    r2 = dx**2 + dy**2
    rho = np.sqrt(r2)
    zk = complex(wavenumber)

    with np.errstate(divide="ignore", invalid="ignore"):
        h0 = hankel1(0, zk * rho)
        h1 = hankel1(1, zk * rho)
        value = 0.25j * h0
        gradient = np.stack((-0.25j * zk * h1 * dx / rho, -0.25j * zk * h1 * dy / rho), axis=0)
        h2 = 2.0 * h1 / (zk * rho) - h0
        hessian = np.empty((2, 2, target_positions.shape[1], source_positions.shape[1]), dtype=complex)
        hessian[0, 0] = 0.25j * zk * (((dx - dy) * (dx + dy) * h1 / rho**3) - zk * dx**2 * h0 / r2)
        hessian[0, 1] = 0.25j * zk**2 * dx * dy * h2 / r2
        hessian[1, 0] = hessian[0, 1]
        hessian[1, 1] = 0.25j * zk * (((dy - dx) * (dx + dy) * h1 / rho**3) - zk * dy**2 * h0 / r2)

    if selector == "s":
        return value[None, None, :, :]
    if selector == "sg":
        return gradient[:, None, :, :]
    if selector == "sp":
        target_normals = flat_normals(target, label="target")
        return np.einsum("rts,rt->ts", gradient, target_normals)[None, None, :, :]
    if selector == "d":
        source_normals = flat_normals(source, label="source")
        return -np.einsum("rts,rs->ts", gradient, source_normals)[None, None, :, :]
    if selector == "dg":
        source_normals = flat_normals(source, label="source")
        return -np.einsum("abts,bs->ats", hessian, source_normals)[:, None, :, :]
    if selector == "dp":
        source_normals = flat_normals(source, label="source")
        target_normals = flat_normals(target, label="target")
        return -np.einsum("abts,bs,at->ts", hessian, source_normals, target_normals)[
            None, None, :, :
        ]
    raise ValueError(f"unknown Helmholtz selector {selector!r}")


def _singularity(selector: str, wavenumber: complex) -> SingularityInfo:
    base = laplace.kernel(selector).singularity
    # The exact Helmholtz split multiplies the Laplace log basis by J0(k*rho).
    # This placeholder preserves the basis shape while the coefficiented-log
    # product-rule implementation lands with its smooth-remainder tests.
    return SingularityInfo(
        family="helmholtz",
        selector=selector,
        input_dim=base.input_dim,
        output_dim=base.output_dim,
        expansion=base.expansion,
        boundary_limit=base.boundary_limit,
        remainder_regular="unknown",
        requirements=base.requirements,
        side_sensitive=base.side_sensitive,
        notes=f"Local split follows Laplace basis with J0(k*rho), k={wavenumber!r}; coefficiented split pending.",
    )
