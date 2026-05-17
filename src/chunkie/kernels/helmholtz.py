"""Two-dimensional Helmholtz kernels."""

from __future__ import annotations

import numpy as np
from scipy.special import hankel1, jv

from . import laplace
from .base import Kernel, flat_normals, flat_positions
from .singularities import (
    LaplaceBasis,
    LaplaceSingularExpansion,
    LaplaceSingularTerm,
    SingularityInfo,
)


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
    return SingularityInfo(
        family="helmholtz",
        selector=selector,
        input_dim=base.input_dim,
        output_dim=base.output_dim,
        expansion=LaplaceSingularExpansion(
            input_dim=base.input_dim,
            output_dim=base.output_dim,
            terms=_singular_terms(selector, wavenumber),
        ),
        boundary_limit=base.boundary_limit,
        remainder_regular="smooth",
        requirements=base.requirements,
        side_sensitive=base.side_sensitive,
        notes=f"Exact local split differentiates J0(k*rho) * G with k={wavenumber!r}.",
    )


def _singular_terms(selector: str, wavenumber: complex) -> tuple[LaplaceSingularTerm, ...]:
    terms: list[LaplaceSingularTerm] = []
    if selector == "s":
        terms.append(
            LaplaceSingularTerm(
                LaplaceBasis(()),
                _component_coefficient(
                    lambda source, target: _j0_values(wavenumber, source, target),
                    output=0,
                    input_=0,
                    output_dim=1,
                ),
                "J0(k*rho) * G",
            )
        )
    elif selector == "sg":
        for a in range(2):
            terms.extend(_gradient_terms(wavenumber, output=a))
    elif selector == "sp":
        for a in range(2):
            terms.extend(_normal_gradient_terms(wavenumber, normal="target", component=a, sign=1.0))
    elif selector == "d":
        for a in range(2):
            terms.extend(_normal_gradient_terms(wavenumber, normal="source", component=a, sign=-1.0))
    elif selector == "dg":
        for a in range(2):
            for b in range(2):
                terms.extend(
                    _source_normal_hessian_terms(
                        wavenumber,
                        source_component=b,
                        sign=-1.0,
                        output=a,
                    )
                )
    elif selector == "dp":
        for a in range(2):
            for b in range(2):
                terms.extend(
                    _normal_hessian_terms(
                        wavenumber,
                        source_component=b,
                        target_component=a,
                        sign=-1.0,
                    )
                )
    return tuple(terms)


def _gradient_terms(wavenumber: complex, *, output: int = 0, input_: int = 0):
    return (
        LaplaceSingularTerm(
            LaplaceBasis((output,)),
            _component_coefficient(
                lambda source, target: _j0_values(wavenumber, source, target),
                output=output,
                input_=input_,
                output_dim=2,
            ),
            "J0(k*rho) * G_a",
        ),
        LaplaceSingularTerm(
            LaplaceBasis(()),
            _component_coefficient(
                lambda source, target: _j0_gradient(wavenumber, source, target)[output],
                output=output,
                input_=input_,
                output_dim=2,
            ),
            "d_xa J0(k*rho) * G",
        ),
    )


def _normal_gradient_terms(wavenumber: complex, *, normal: str, component: int, sign: float):
    return (
        LaplaceSingularTerm(
            LaplaceBasis((component,)),
            _normal_weighted_coefficient(
                lambda source, target: _j0_values(wavenumber, source, target),
                normal=normal,
                component=component,
                sign=sign,
            ),
            "normal-contracted J0(k*rho) * G_a",
        ),
        LaplaceSingularTerm(
            LaplaceBasis(()),
            _normal_weighted_coefficient(
                lambda source, target: _j0_gradient(wavenumber, source, target)[component],
                normal=normal,
                component=component,
                sign=sign,
            ),
            "normal-contracted d_xa J0(k*rho) * G",
        ),
    )


def _normal_hessian_terms(
    wavenumber: complex,
    *,
    source_component: int,
    target_component: int,
    sign: float,
    output: int = 0,
):
    # Product rule for d_ab(J0(k*rho) * G). The normal contractions are
    # coefficient functions so quadrature still sees only Laplace basis terms.
    output_dim = 2 if output != 0 else 1
    return (
        LaplaceSingularTerm(
            LaplaceBasis((target_component, source_component)),
            _paired_normal_coefficient(
                lambda source, target: _j0_values(wavenumber, source, target),
                target_component=target_component,
                source_component=source_component,
                sign=sign,
                output=output,
                output_dim=output_dim,
            ),
            "J0(k*rho) * G_ab",
        ),
        LaplaceSingularTerm(
            LaplaceBasis((source_component,)),
            _paired_normal_coefficient(
                lambda source, target: _j0_gradient(wavenumber, source, target)[target_component],
                target_component=target_component,
                source_component=source_component,
                sign=sign,
                output=output,
                output_dim=output_dim,
            ),
            "d_xa J0(k*rho) * G_b",
        ),
        LaplaceSingularTerm(
            LaplaceBasis((target_component,)),
            _paired_normal_coefficient(
                lambda source, target: _j0_gradient(wavenumber, source, target)[source_component],
                target_component=target_component,
                source_component=source_component,
                sign=sign,
                output=output,
                output_dim=output_dim,
            ),
            "d_xb J0(k*rho) * G_a",
        ),
        LaplaceSingularTerm(
            LaplaceBasis(()),
            _paired_normal_coefficient(
                lambda source, target: _j0_hessian(wavenumber, source, target)[
                    target_component, source_component
                ],
                target_component=target_component,
                source_component=source_component,
                sign=sign,
                output=output,
                output_dim=output_dim,
            ),
            "d_xa d_xb J0(k*rho) * G",
        ),
    )


def _source_normal_hessian_terms(
    wavenumber: complex,
    *,
    source_component: int,
    sign: float,
    output: int,
):
    return (
        LaplaceSingularTerm(
            LaplaceBasis((output, source_component)),
            _source_normal_hessian_coefficient(
                lambda source, target: _j0_values(wavenumber, source, target),
                source_component=source_component,
                sign=sign,
                output=output,
            ),
            "source-normal J0(k*rho) * G_ab",
        ),
        LaplaceSingularTerm(
            LaplaceBasis((source_component,)),
            _source_normal_hessian_coefficient(
                lambda source, target: _j0_gradient(wavenumber, source, target)[output],
                source_component=source_component,
                sign=sign,
                output=output,
            ),
            "source-normal d_xa J0(k*rho) * G_b",
        ),
        LaplaceSingularTerm(
            LaplaceBasis((output,)),
            _source_normal_hessian_coefficient(
                lambda source, target: _j0_gradient(wavenumber, source, target)[source_component],
                source_component=source_component,
                sign=sign,
                output=output,
            ),
            "source-normal d_xb J0(k*rho) * G_a",
        ),
        LaplaceSingularTerm(
            LaplaceBasis(()),
            _source_normal_hessian_coefficient(
                lambda source, target: _j0_hessian(wavenumber, source, target)[
                    output, source_component
                ],
                source_component=source_component,
                sign=sign,
                output=output,
            ),
            "source-normal d_xa d_xb J0(k*rho) * G",
        ),
    )


def _component_coefficient(values, *, output: int, input_: int, output_dim: int):
    def coefficient(source, target):
        base = np.asarray(values(source, target))
        out = np.zeros((output_dim, 1, *base.shape), dtype=complex)
        out[output, input_] = base
        return out

    return coefficient


def _normal_weighted_coefficient(values, *, normal: str, component: int, sign: float):
    def coefficient(source, target):
        base = np.asarray(values(source, target))
        normals = flat_normals(source if normal == "source" else target, label=normal)
        weight = normals[component][None, :] if normal == "source" else normals[component][:, None]
        return (sign * base * weight)[None, None, :, :]

    return coefficient


def _paired_normal_coefficient(
    values,
    *,
    target_component: int,
    source_component: int,
    sign: float,
    output: int,
    output_dim: int,
):
    def coefficient(source, target):
        base = np.asarray(values(source, target))
        source_normals = flat_normals(source, label="source")
        target_normals = flat_normals(target, label="target")
        weight = source_normals[source_component][None, :] * target_normals[target_component][:, None]
        out = np.zeros((output_dim, 1, *base.shape), dtype=complex)
        out[output, 0] = sign * base * weight
        return out

    return coefficient


def _source_normal_hessian_coefficient(values, *, source_component: int, sign: float, output: int):
    def coefficient(source, target):
        base = np.asarray(values(source, target))
        source_normals = flat_normals(source, label="source")
        weight = source_normals[source_component][None, :]
        out = np.zeros((2, 1, *base.shape), dtype=complex)
        out[output, 0] = sign * base * weight
        return out

    return coefficient


def _j0_values(wavenumber: complex, source, target):
    rho = _rho(source, target)
    return jv(0, complex(wavenumber) * rho)


def _j0_gradient(wavenumber: complex, source, target):
    displacement, rho = _displacement_and_rho(source, target)
    z = complex(wavenumber) * rho
    with np.errstate(divide="ignore", invalid="ignore"):
        radial = -complex(wavenumber) * jv(1, z) / rho
        gradient = radial[None, :, :] * displacement
    return np.where(rho[None, :, :] == 0.0, 0.0, gradient)


def _j0_hessian(wavenumber: complex, source, target):
    displacement, rho = _displacement_and_rho(source, target)
    z = complex(wavenumber) * rho
    with np.errstate(divide="ignore", invalid="ignore"):
        first = -complex(wavenumber) * jv(1, z)
        second = -0.5 * complex(wavenumber) ** 2 * (jv(0, z) - jv(2, z))
        radial_identity = first / rho
        radial_outer = (second - first / rho) / rho**2

    hessian = np.empty((2, 2, *rho.shape), dtype=complex)
    for a in range(2):
        for b in range(2):
            hessian[a, b] = radial_outer * displacement[a] * displacement[b]
            if a == b:
                hessian[a, b] += radial_identity

    zero = rho == 0.0
    if np.any(zero):
        for a in range(2):
            for b in range(2):
                hessian[a, b, zero] = -0.5 * complex(wavenumber) ** 2 if a == b else 0.0
    return hessian


def _rho(source, target):
    return _displacement_and_rho(source, target)[1]


def _displacement_and_rho(source, target):
    source_positions = flat_positions(source)
    target_positions = flat_positions(target)
    displacement = target_positions[:, :, None] - source_positions[:, None, :]
    rho = np.linalg.norm(displacement, axis=0)
    return displacement, rho
