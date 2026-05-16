"""Kernel algebra helpers."""

from __future__ import annotations

from dataclasses import replace

from .base import Kernel
from .singularities import LaplaceSingularExpansion, LaplaceSingularTerm, SingularityInfo


def scale(kernel: Kernel, factor: complex) -> Kernel:
    terms = tuple(
        LaplaceSingularTerm(term.basis, factor * term.coefficient, term.meaning)
        for term in kernel.singularity.expansion.terms
        if not callable(term.coefficient)
    )
    singularity = replace(
        kernel.singularity,
        expansion=LaplaceSingularExpansion(
            input_dim=kernel.input_dim,
            output_dim=kernel.output_dim,
            terms=terms if len(terms) == len(kernel.singularity.expansion.terms) else kernel.singularity.expansion.terms,
        ),
        notes=(kernel.singularity.notes + f" scaled by {factor!r}").strip(),
    )

    def evaluator(source, target):
        return factor * kernel(source, target)

    return replace(kernel, singularity=singularity, evaluator=evaluator)


def add(left: Kernel, right: Kernel) -> Kernel:
    if left.input_dim != right.input_dim or left.output_dim != right.output_dim:
        raise ValueError("kernels must have matching input/output dimensions")
    expansion = LaplaceSingularExpansion(
        input_dim=left.input_dim,
        output_dim=left.output_dim,
        terms=left.singularity.expansion.terms + right.singularity.expansion.terms,
    )
    singularity = SingularityInfo(
        family=f"{left.family}+{right.family}",
        selector=f"{left.selector}+{right.selector}",
        input_dim=left.input_dim,
        output_dim=left.output_dim,
        expansion=expansion,
        boundary_limit="supersingular" if expansion.legacy_strength == "hs" else "pv",
        remainder_regular="unknown",
        notes="Algebraic singularity combination; cancellation canonicalization is upcoming work.",
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
