"""Two-dimensional linear-elasticity kernels."""

from __future__ import annotations

import numpy as np

from .base import Kernel, flat_positions
from .singularities import (
    GeometryRequirements,
    LaplaceBasis,
    LaplaceSingularExpansion,
    LaplaceSingularTerm,
    SingularityInfo,
    matrix_coefficient,
)


def kernel(selector: str = "s", *, lame_lambda: float, lame_mu: float) -> Kernel:
    selector = _canonical_selector(selector)
    input_dim, output_dim = _dimensions(selector)
    return Kernel(
        family="elasticity",
        selector=selector,
        params={"lame_lambda": lame_lambda, "lame_mu": lame_mu},
        input_dim=input_dim,
        output_dim=output_dim,
        singularity=_singularity(selector, lame_lambda, lame_mu),
        evaluator=lambda source, target: evaluate(
            selector,
            source,
            target,
            lame_lambda=lame_lambda,
            lame_mu=lame_mu,
        ),
    )


def evaluate(selector: str, source, target, *, lame_lambda: float, lame_mu: float):
    selector = _canonical_selector(selector)
    beta, gamma = _elasticity_constants(lame_lambda, lame_mu)
    source_positions = flat_positions(source)
    target_positions = flat_positions(target)
    displacement = target_positions[:, :, None] - source_positions[:, None, :]
    dx, dy = displacement
    rho2 = dx**2 + dy**2
    ntarget, nsource = rho2.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        if selector == "s":
            log_rho = 0.5 * np.log(rho2)
            values = np.empty((2, 2, ntarget, nsource), dtype=float)
            values[0, 0] = beta * log_rho + gamma / 2.0 + gamma * dx**2 / rho2
            values[1, 1] = beta * log_rho + gamma / 2.0 + gamma * dy**2 / rho2
            values[0, 1] = gamma * dx * dy / rho2
            values[1, 0] = values[0, 1]
            return values

    raise ValueError(f"unknown elasticity selector {selector!r}")


def _canonical_selector(selector: str) -> str:
    aliases = {"single": "s"}
    return aliases.get(selector.lower(), selector.lower())


def _dimensions(selector: str) -> tuple[int, int]:
    if selector == "s":
        return 2, 2
    raise ValueError(f"unknown elasticity selector {selector!r}")


def _singularity(selector: str, lame_lambda: float, lame_mu: float) -> SingularityInfo:
    beta, gamma = _elasticity_constants(lame_lambda, lame_mu)
    if selector != "s":
        return SingularityInfo.smooth(family="elasticity", selector=selector, input_dim=2, output_dim=2)

    terms: list[LaplaceSingularTerm] = []
    for i in range(2):
        for j in range(2):
            if i == j:
                terms.append(
                    LaplaceSingularTerm(
                        LaplaceBasis(()),
                        matrix_coefficient(2, 2, i, j, 2.0 * np.pi * (gamma - beta)),
                        "2*pi*(gamma-beta)*delta_ij*G",
                    )
                )
            terms.extend(_b_hessian_terms(i, j, scale=4.0 * np.pi * gamma))

    return SingularityInfo(
        family="elasticity",
        selector=selector,
        input_dim=2,
        output_dim=2,
        expansion=LaplaceSingularExpansion(input_dim=2, output_dim=2, terms=tuple(terms)),
        boundary_limit="removable",
        remainder_regular="smooth",
        requirements=GeometryRequirements(),
    )


def _elasticity_constants(lame_lambda: float, lame_mu: float) -> tuple[float, float]:
    beta = (lame_lambda + 3.0 * lame_mu) / (
        4.0 * np.pi * lame_mu * (lame_lambda + 2.0 * lame_mu)
    )
    gamma = -(lame_lambda + lame_mu) / (
        4.0 * np.pi * lame_mu * (lame_lambda + 2.0 * lame_mu)
    )
    return beta, gamma


def _b_hessian_terms(a: int, b: int, *, scale: float):
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
