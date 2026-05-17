"""Two-dimensional biharmonic kernels."""

from __future__ import annotations

import numpy as np

from .base import Kernel, flat_normals, flat_positions
from .singularities import (
    GeometryRequirements,
    LaplaceBasis,
    LaplaceSingularExpansion,
    LaplaceSingularTerm,
    SingularityInfo,
)


def kernel(selector: str = "s") -> Kernel:
    selector = _canonical_selector(selector)
    input_dim, output_dim = _dimensions(selector)
    return Kernel(
        family="biharmonic",
        selector=selector,
        params={},
        input_dim=input_dim,
        output_dim=output_dim,
        singularity=_singularity(selector),
        evaluator=lambda source, target: evaluate(selector, source, target),
    )


def evaluate(selector: str, source, target):
    selector = _canonical_selector(selector)
    source_positions = flat_positions(source)
    target_positions = flat_positions(target)
    displacement = target_positions[:, :, None] - source_positions[:, None, :]
    dx, dy = displacement
    rho2 = dx**2 + dy**2

    with np.errstate(divide="ignore", invalid="ignore"):
        log_rho = 0.5 * np.log(rho2)
        value = rho2 * log_rho / (8.0 * np.pi)
        gradient = displacement * (2.0 * log_rho[None, :, :] + 1.0) / (8.0 * np.pi)
        hessian = np.empty((2, 2, *rho2.shape), dtype=float)
        for a in range(2):
            for b in range(2):
                hessian[a, b] = displacement[a] * displacement[b] / (4.0 * np.pi * rho2)
                if a == b:
                    hessian[a, b] += (2.0 * log_rho + 1.0) / (8.0 * np.pi)

    value = np.where(rho2 == 0.0, 0.0, value)

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
    if selector == "hessian":
        return hessian.reshape(4, 1, *rho2.shape)
    raise ValueError(f"unknown biharmonic selector {selector!r}")


def _canonical_selector(selector: str) -> str:
    aliases = {"single": "s", "sgrad": "sg", "sprime": "sp", "double": "d", "hess": "hessian"}
    return aliases.get(selector.lower(), selector.lower())


def _dimensions(selector: str) -> tuple[int, int]:
    if selector == "sg":
        return 1, 2
    if selector == "hessian":
        return 1, 4
    return 1, 1


def _singularity(selector: str) -> SingularityInfo:
    input_dim, output_dim = _dimensions(selector)
    terms: list[LaplaceSingularTerm] = []
    requirements = GeometryRequirements(
        source_normals=selector == "d",
        target_normals=selector == "sp",
    )

    if selector == "s":
        terms.append(
            LaplaceSingularTerm(
                LaplaceBasis(()),
                _component_coefficient(_c_coefficient(), output=0, output_dim=1),
                "c * G",
            )
        )
        boundary_limit = "removable"
    elif selector == "sg":
        for a in range(2):
            terms.extend(_gradient_terms(a))
        boundary_limit = "removable"
    elif selector == "sp":
        for a in range(2):
            terms.extend(_normal_gradient_terms(a, normal="target", sign=1.0))
        boundary_limit = "removable"
    elif selector == "d":
        for a in range(2):
            terms.extend(_normal_gradient_terms(a, normal="source", sign=-1.0))
        boundary_limit = "removable"
    elif selector == "hessian":
        for a in range(2):
            for b in range(2):
                terms.extend(_hessian_terms(a, b))
        boundary_limit = "removable"
    else:
        boundary_limit = "smooth"

    return SingularityInfo(
        family="biharmonic",
        selector=selector,
        input_dim=input_dim,
        output_dim=output_dim,
        expansion=LaplaceSingularExpansion(input_dim=input_dim, output_dim=output_dim, terms=tuple(terms)),
        boundary_limit=boundary_limit,
        remainder_regular="smooth",
        requirements=requirements,
        side_sensitive=selector in {"d", "sp"},
    )


def _gradient_terms(component: int):
    return (
        LaplaceSingularTerm(
            LaplaceBasis(()),
            _component_coefficient(_c_gradient(component), output=component, output_dim=2),
            "c_a * G",
        ),
        LaplaceSingularTerm(
            LaplaceBasis((component,)),
            _component_coefficient(_c_coefficient(), output=component, output_dim=2),
            "c * G_a",
        ),
    )


def _hessian_terms(a: int, b: int):
    output = 2 * a + b
    return (
        LaplaceSingularTerm(
            LaplaceBasis(()),
            _component_coefficient(_c_hessian(a, b), output=output, output_dim=4),
            "c_ab * G",
        ),
        LaplaceSingularTerm(
            LaplaceBasis((b,)),
            _component_coefficient(_c_gradient(a), output=output, output_dim=4),
            "c_a * G_b",
        ),
        LaplaceSingularTerm(
            LaplaceBasis((a,)),
            _component_coefficient(_c_gradient(b), output=output, output_dim=4),
            "c_b * G_a",
        ),
        LaplaceSingularTerm(
            LaplaceBasis((a, b)),
            _component_coefficient(_c_coefficient(), output=output, output_dim=4),
            "c * G_ab",
        ),
    )


def _normal_gradient_terms(component: int, *, normal: str, sign: float):
    return (
        LaplaceSingularTerm(
            LaplaceBasis(()),
            _normal_coefficient(_c_gradient(component), normal=normal, component=component, sign=sign),
            "normal-contracted c_a * G",
        ),
        LaplaceSingularTerm(
            LaplaceBasis((component,)),
            _normal_coefficient(_c_coefficient(), normal=normal, component=component, sign=sign),
            "normal-contracted c * G_a",
        ),
    )


def _component_coefficient(values, *, output: int, output_dim: int):
    def coefficient(source, target):
        base = np.asarray(values(source, target))
        out = np.zeros((output_dim, 1, *base.shape), dtype=float)
        out[output, 0] = base
        return out

    return coefficient


def _normal_coefficient(values, *, normal: str, component: int, sign: float):
    def coefficient(source, target):
        base = np.asarray(values(source, target))
        normals = flat_normals(source if normal == "source" else target, label=normal)
        weight = normals[component][None, :] if normal == "source" else normals[component][:, None]
        return (sign * base * weight)[None, None, :, :]

    return coefficient


def _c_coefficient():
    def coefficient(source, target):
        displacement, rho2 = _displacement_and_rho2(source, target)
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
