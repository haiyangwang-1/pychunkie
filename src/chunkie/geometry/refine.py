"""Refinement entry points.

Panel splitting and adaptive refinement are required rewrite work. The active
bootstrap keeps this module as the ownership location so later implementation
does not leak refinement policy into storage classes.
"""

from __future__ import annotations

from .chunker import Chunker


def refine(chunker: Chunker, *, levels: int = 1) -> Chunker:
    raise NotImplementedError("Chunker refinement is a required upcoming rewrite milestone")
