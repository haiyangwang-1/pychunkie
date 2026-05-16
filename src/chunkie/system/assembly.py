"""Dense system assembly."""

from __future__ import annotations

import numpy as np

from chunkie.geometry import Chunker

from .config import SystemConfig
from .matrix import SystemMatrix
from .trace import BoundaryTrace


def assemble_system_matrix(system, *, config: SystemConfig) -> SystemMatrix:
    if len(system.unknowns) != 1 or len(system.equations) != 1:
        raise NotImplementedError("dense assembly currently supports one scalar unknown and one equation")

    unknown = system.unknowns[0]
    equation = system.equations[0]
    row_count = _point_count(equation.target)
    column_count = _point_count(unknown.geometry)
    matrix = np.zeros((row_count, column_count), dtype=complex)

    for term in equation.terms:
        if not isinstance(term, BoundaryTrace):
            raise NotImplementedError("only BoundaryTrace terms are supported in dense bootstrap assembly")
        if term.layer.density != unknown.name:
            raise NotImplementedError("dense bootstrap assembly supports one matching density")
        block = _trace_matrix(term)
        matrix += term.layer.coefficient * block
        if term.jump is not None:
            matrix += term.jump.coefficient * np.eye(row_count, column_count)

    return SystemMatrix(
        matrix,
        config,
        diagnostics={"assembly": "dense", "unknown": unknown.name, "equation": equation.name},
    )


def rhs_vector(system) -> np.ndarray:
    if len(system.equations) != 1:
        raise NotImplementedError("dense bootstrap RHS supports one equation")
    equation = system.equations[0]
    rhs = equation.rhs(equation.target.pointinfo) if callable(equation.rhs) else equation.rhs
    arr = np.asarray(rhs, dtype=complex)
    if arr.ndim == 2:
        arr = arr.T.reshape(-1)
    elif arr.ndim == 3:
        arr = arr.swapaxes(1, 2).reshape(-1)
    if arr.size != _point_count(equation.target):
        raise ValueError("boundary data has incompatible size")
    return arr.reshape(-1)


def identity_system_matrix(size: int, *, config: SystemConfig | None = None) -> SystemMatrix:
    return SystemMatrix(np.eye(size), SystemConfig() if config is None else config)


def _trace_matrix(trace: BoundaryTrace) -> np.ndarray:
    source = trace.layer.source
    target = trace.target
    if not isinstance(source, Chunker) or not isinstance(target, Chunker):
        raise NotImplementedError("dense bootstrap assembly currently supports Chunker sources and targets")

    values = trace.layer.kernel(source.pointinfo, target.pointinfo)
    if values.shape[:2] != (1, 1):
        raise NotImplementedError("dense bootstrap assembly currently supports scalar kernels")
    block = values[0, 0] * source.pointinfo.flat_weights[None, :]

    # Smooth closed-curve double-layer self blocks have a finite diagonal limit.
    # We insert the local limit here so the dense reference path is usable before
    # special quadrature owns same-panel correction.
    if source is target and trace.layer.kernel.family == "laplace" and trace.layer.kernel.selector == "d":
        diagonal = (-source.signed_curvature / (4.0 * np.pi)).T.reshape(-1)
        weights = source.pointinfo.flat_weights
        np.fill_diagonal(block, diagonal * weights)
    return block


def _point_count(geometry: object) -> int:
    if not hasattr(geometry, "point_count"):
        raise TypeError("geometry must expose point_count")
    return int(geometry.point_count)
