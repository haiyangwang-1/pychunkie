"""Kernel algebra helpers."""

from __future__ import annotations

from dataclasses import replace

from .base import Kernel
from .singularities import SingularityInfo


def scale(kernel: Kernel, factor: complex) -> Kernel:
    singularity = replace(
        kernel.singularity,
        expansion=kernel.singularity.expansion.scaled(factor),
        notes=(kernel.singularity.notes + f" scaled by {factor!r}").strip(),
    )

    def evaluator(source, target):
        return factor * kernel(source, target)

    return replace(kernel, singularity=singularity, evaluator=evaluator)


def add(left: Kernel, right: Kernel) -> Kernel:
    if left.input_dim != right.input_dim or left.output_dim != right.output_dim:
        raise ValueError("kernels must have matching input/output dimensions")
    expansion = left.singularity.expansion.combined(right.singularity.expansion)
    strength = expansion.legacy_strength
    singularity = SingularityInfo(
        family=f"{left.family}+{right.family}",
        selector=f"{left.selector}+{right.selector}",
        input_dim=left.input_dim,
        output_dim=left.output_dim,
        expansion=expansion,
        boundary_limit=_boundary_limit(strength),
        remainder_regular=(
            "smooth"
            if strength == "smooth"
            and left.singularity.remainder_regular == "smooth"
            and right.singularity.remainder_regular == "smooth"
            else "unknown"
        ),
        notes="Algebraic singularity combination with exact scalar/matrix coefficient canonicalization.",
    )

    def evaluator(source, target):
        return left(source, target) + right(source, target)

    return Kernel(
        family=singularity.family,
        selector=singularity.selector,
        params={},
        input_dim=left.input_dim,
        output_dim=left.output_dim,
        singularity=singularity,
        evaluator=evaluator,
    )


def _boundary_limit(strength: str):
    if strength == "smooth":
        return "smooth"
    if strength == "log":
        return "removable"
    if strength == "pv":
        return "pv"
    return "hs"
