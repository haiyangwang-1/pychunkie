"""Two-dimensional Stokes kernels."""

from __future__ import annotations

import numpy as np

from .base import Kernel, flat_normals, flat_positions
from .singularities import (
    GeometryRequirements,
    LaplaceBasis,
    LaplaceSingularExpansion,
    LaplaceSingularTerm,
    SingularityInfo,
    matrix_coefficient,
)


def kernel(selector: str = "s", *, viscosity: float = 1.0) -> Kernel:
    selector = _canonical_selector(selector)
    input_dim, output_dim = _dimensions(selector)
    return Kernel(
        family="stokes",
        selector=selector,
        params={"viscosity": viscosity},
        input_dim=input_dim,
        output_dim=output_dim,
        singularity=_singularity(selector, viscosity),
        evaluator=lambda source, target: evaluate(selector, source, target, viscosity=viscosity),
    )


def evaluate(selector: str, source, target, *, viscosity: float = 1.0):
    selector = _canonical_selector(selector)
    source_positions = flat_positions(source)
    target_positions = flat_positions(target)
    displacement = target_positions[:, :, None] - source_positions[:, None, :]
    dx, dy = displacement
    rho2 = dx**2 + dy**2
    ntarget, nsource = rho2.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        if selector == "s":
            log_inverse_rho = -0.5 * np.log(rho2)
            values = np.empty((2, 2, ntarget, nsource), dtype=float)
            values[0, 0] = (dx**2 / rho2 + log_inverse_rho) / (4.0 * np.pi * viscosity)
            values[1, 1] = (dy**2 / rho2 + log_inverse_rho) / (4.0 * np.pi * viscosity)
            values[0, 1] = dx * dy / rho2 / (4.0 * np.pi * viscosity)
            values[1, 0] = values[0, 1]
            return values
        if selector == "d":
            source_normals = flat_normals(source, label="source")
            rn = dx * source_normals[0, None, :] + dy * source_normals[1, None, :]
            rho4 = rho2**2
            values = np.empty((2, 2, ntarget, nsource), dtype=float)
            values[0, 0] = dx**2 * rn / (np.pi * rho4)
            values[1, 1] = dy**2 * rn / (np.pi * rho4)
            values[0, 1] = dx * dy * rn / (np.pi * rho4)
            values[1, 0] = values[0, 1]
            return values

    raise ValueError(f"unknown Stokes selector {selector!r}")


def _canonical_selector(selector: str) -> str:
    aliases = {"single": "s", "svel": "s", "double": "d", "dvel": "d"}
    return aliases.get(selector.lower(), selector.lower())


def _dimensions(selector: str) -> tuple[int, int]:
    if selector in {"s", "d"}:
        return 2, 2
    raise ValueError(f"unknown Stokes selector {selector!r}")


def _singularity(selector: str, viscosity: float) -> SingularityInfo:
    if selector == "s":
        terms = _single_velocity_terms(viscosity)
        requirements = GeometryRequirements()
        boundary_limit = "removable"
        remainder = "smooth"
    elif selector == "d":
        terms = _double_velocity_terms()
        requirements = GeometryRequirements(source_normals=True)
        boundary_limit = "pv"
        remainder = "smooth"
    else:
        terms = ()
        requirements = GeometryRequirements()
        boundary_limit = "smooth"
        remainder = "smooth"

    return SingularityInfo(
        family="stokes",
        selector=selector,
        input_dim=2,
        output_dim=2,
        expansion=LaplaceSingularExpansion(input_dim=2, output_dim=2, terms=terms),
        boundary_limit=boundary_limit,
        remainder_regular=remainder,
        requirements=requirements,
        side_sensitive=selector == "d",
    )


def _single_velocity_terms(viscosity: float) -> tuple[LaplaceSingularTerm, ...]:
    terms: list[LaplaceSingularTerm] = []
    inv_mu = 1.0 / float(viscosity)
    for i in range(2):
        for j in range(2):
            if i == j:
                terms.append(
                    LaplaceSingularTerm(
                        LaplaceBasis(()),
                        matrix_coefficient(2, 2, i, j, inv_mu),
                        "delta_ij G / mu",
                    )
                )
            for term in _b_hessian_terms(i, j, scale=inv_mu):
                terms.append(term)
    return tuple(terms)


def _double_velocity_terms() -> tuple[LaplaceSingularTerm, ...]:
    terms: list[LaplaceSingularTerm] = []
    for i in range(2):
        for j in range(2):
            terms.append(
                LaplaceSingularTerm(
                    LaplaceBasis((i, j)),
                    _rn_coefficient(output=i, input_=j),
                    "n_s[k] r_k G_ij",
                )
            )
            if i == j:
                for k in range(2):
                    terms.append(
                        LaplaceSingularTerm(
                            LaplaceBasis((k,)),
                            _source_normal_component(k, output=i, input_=j, sign=-1.0),
                            "-delta_ij n_s[k] G_k",
                        )
                    )
    return tuple(terms)


def _b_hessian_terms(a: int, b: int, *, scale: float):
    # B_ab = c_ab G + c_a G_b + c_b G_a + c G_ab, where B is the
    # biharmonic fundamental solution. This is the Stokeslet tensor split.
    return (
        LaplaceSingularTerm(
            LaplaceBasis(()),
            _matrix_component(_c_hessian(a, b), output=a, input_=b, scale=scale),
            "B_ab c_ab G",
        ),
        LaplaceSingularTerm(
            LaplaceBasis((b,)),
            _matrix_component(_c_gradient(a), output=a, input_=b, scale=scale),
            "B_ab c_a G_b",
        ),
        LaplaceSingularTerm(
            LaplaceBasis((a,)),
            _matrix_component(_c_gradient(b), output=a, input_=b, scale=scale),
            "B_ab c_b G_a",
        ),
        LaplaceSingularTerm(
            LaplaceBasis((a, b)),
            _matrix_component(_c_coefficient(), output=a, input_=b, scale=scale),
            "B_ab c G_ab",
        ),
    )


def _matrix_component(values, *, output: int, input_: int, scale: float = 1.0):
    def coefficient(source, target):
        base = np.asarray(values(source, target))
        out = np.zeros((2, 2, *base.shape), dtype=float)
        out[output, input_] = scale * base
        return out

    return coefficient


def _rn_coefficient(*, output: int, input_: int):
    def coefficient(source, target):
        displacement, _ = _displacement_and_rho2(source, target)
        source_normals = flat_normals(source, label="source")
        rn = displacement[0] * source_normals[0, None, :] + displacement[1] * source_normals[1, None, :]
        out = np.zeros((2, 2, *rn.shape), dtype=float)
        out[output, input_] = rn
        return out

    return coefficient


def _source_normal_component(component: int, *, output: int, input_: int, sign: float):
    def coefficient(source, target):
        source_normals = flat_normals(source, label="source")
        target_count = flat_positions(target).shape[1]
        out = np.zeros((2, 2, target_count, source_normals.shape[1]), dtype=float)
        out[output, input_] = sign * source_normals[component][None, :]
        return out

    return coefficient


def _c_coefficient():
    def coefficient(source, target):
        _, rho2 = _displacement_and_rho2(source, target)
        return -rho2 / 4.0

    return coefficient


def _c_gradient(component: int):
    def coefficient(source, target):
        displacement, _ = _displacement_and_rho2(source, target)
        return -displacement[component] / 2.0

    return coefficient


def _c_hessian(a: int, b: int):
    def coefficient(source, target):
        _, rho2 = _displacement_and_rho2(source, target)
        return np.full_like(rho2, -0.5 if a == b else 0.0)

    return coefficient


def _displacement_and_rho2(source, target):
    source_positions = flat_positions(source)
    target_positions = flat_positions(target)
    displacement = target_positions[:, :, None] - source_positions[:, None, :]
    return displacement, np.sum(displacement**2, axis=0)
