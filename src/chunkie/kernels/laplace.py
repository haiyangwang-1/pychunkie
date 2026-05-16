"""Two-dimensional Laplace kernels."""

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


def kernel(selector: str = "s") -> Kernel:
    selector = _canonical_selector(selector)
    input_dim, output_dim = _dimensions(selector)
    return Kernel(
        family="laplace",
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
    dx = target_positions[0, :, None] - source_positions[0, None, :]
    dy = target_positions[1, :, None] - source_positions[1, None, :]
    r = (dx, dy)
    r2 = dx**2 + dy**2

    with np.errstate(divide="ignore", invalid="ignore"):
        value = -np.log(r2) / (4.0 * np.pi)
        gradient = np.stack((-dx / (2.0 * np.pi * r2), -dy / (2.0 * np.pi * r2)), axis=0)
        hessian = np.empty((2, 2, target_positions.shape[1], source_positions.shape[1]), dtype=float)
        for a in range(2):
            for b in range(2):
                delta = 1.0 if a == b else 0.0
                hessian[a, b] = (2.0 * r[a] * r[b] - delta * r2) / (2.0 * np.pi * r2**2)

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
    raise ValueError(f"unknown Laplace selector {selector!r}")


def _canonical_selector(selector: str) -> str:
    aliases = {
        "single": "s",
        "sgrad": "sg",
        "sprime": "sp",
        "double": "d",
        "dgrad": "dg",
        "dprime": "dp",
    }
    return aliases.get(selector.lower(), selector.lower())


def _dimensions(selector: str) -> tuple[int, int]:
    if selector in {"sg", "dg"}:
        return 1, 2
    return 1, 1


def _singularity(selector: str) -> SingularityInfo:
    input_dim, output_dim = _dimensions(selector)
    requirements = GeometryRequirements(
        source_normals=selector in {"d", "dg", "dp"},
        target_normals=selector in {"sp", "dp"},
    )
    terms: list[LaplaceSingularTerm] = []

    if selector == "s":
        terms.append(LaplaceSingularTerm(LaplaceBasis(()), 1.0, "Laplace single-layer log"))
        boundary_limit = "removable"
    elif selector == "sg":
        for a in range(2):
            terms.append(
                LaplaceSingularTerm(
                    LaplaceBasis((a,)),
                    matrix_coefficient(2, 1, a),
                    f"Cartesian gradient component {a}",
                )
            )
        boundary_limit = "pv"
    elif selector == "sp":
        for a in range(2):
            terms.append(LaplaceSingularTerm(LaplaceBasis((a,)), _target_normal_coefficient(a)))
        boundary_limit = "pv"
    elif selector == "d":
        for a in range(2):
            terms.append(LaplaceSingularTerm(LaplaceBasis((a,)), _source_normal_coefficient(a, sign=-1.0)))
        boundary_limit = "pv"
    elif selector == "dg":
        for a in range(2):
            for b in range(2):
                terms.append(
                    LaplaceSingularTerm(
                        LaplaceBasis((a, b)),
                        _source_normal_coefficient(b, sign=-1.0, output_dim=2, output=a),
                    )
                )
        boundary_limit = "hs"
    elif selector == "dp":
        for a in range(2):
            for b in range(2):
                terms.append(LaplaceSingularTerm(LaplaceBasis((a, b)), _normal_pair_coefficient(a, b, -1.0)))
        boundary_limit = "hs"
    else:
        boundary_limit = "smooth"

    return SingularityInfo(
        family="laplace",
        selector=selector,
        input_dim=input_dim,
        output_dim=output_dim,
        expansion=LaplaceSingularExpansion(input_dim=input_dim, output_dim=output_dim, terms=tuple(terms)),
        boundary_limit=boundary_limit,
        remainder_regular="smooth",
        requirements=requirements,
        side_sensitive=selector in {"d", "sp", "dp"},
    )


def _source_normal_coefficient(component: int, *, sign: float, output_dim: int = 1, output: int = 0):
    def coefficient(source, target):
        source_normals = flat_normals(source, label="source")
        target_count = flat_positions(target).shape[1]
        values = np.zeros((output_dim, 1, target_count, source_normals.shape[1]))
        values[output, 0] = sign * source_normals[component][None, :]
        return values

    return coefficient


def _target_normal_coefficient(component: int):
    def coefficient(source, target):
        target_normals = flat_normals(target, label="target")
        source_count = flat_positions(source).shape[1]
        values = np.zeros((1, 1, target_normals.shape[1], source_count))
        values[0, 0] = target_normals[component][:, None]
        return values

    return coefficient


def _normal_pair_coefficient(target_component: int, source_component: int, sign: float):
    def coefficient(source, target):
        source_normals = flat_normals(source, label="source")
        target_normals = flat_normals(target, label="target")
        values = np.zeros((1, 1, target_normals.shape[1], source_normals.shape[1]))
        values[0, 0] = sign * target_normals[target_component][:, None] * source_normals[source_component][None, :]
        return values

    return coefficient
