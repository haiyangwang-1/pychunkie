"""Dense system assembly."""

from __future__ import annotations

import numpy as np

from chunkie.geometry import Chunker
from chunkie.quadrature import dense_panel_operator_matrix

from .config import SystemConfig
from .matrix import SystemMatrix
from .trace import BoundaryTrace


def assemble_system_matrix(system, *, config: SystemConfig) -> SystemMatrix:
    if len(system.unknowns) != 1 or len(system.equations) != 1:
        raise NotImplementedError("dense assembly currently supports one unknown and one equation")

    unknown = system.unknowns[0]
    equation = system.equations[0]
    row_count = _equation_row_count(equation)
    column_count = _point_count(unknown.geometry) * unknown.component_count
    matrix = np.zeros((row_count, column_count), dtype=complex)

    for term in equation.terms:
        if not isinstance(term, BoundaryTrace):
            raise NotImplementedError("only BoundaryTrace terms are supported in dense bootstrap assembly")
        if term.layer.density != unknown.name:
            raise NotImplementedError("dense bootstrap assembly supports one matching density")
        block = _trace_matrix(term, unknown)
        if block.shape != matrix.shape:
            raise ValueError("trace block shape does not match dense system layout")
        matrix += term.layer.coefficient * block
        if term.jump is not None:
            if term.jump.density != unknown.name:
                raise NotImplementedError("dense bootstrap assembly supports jumps on the matching density")
            if row_count != column_count:
                raise ValueError("jump terms require matching row and column layout")
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
    output_dim = _equation_output_dim(equation)
    point_count = _point_count(equation.target)
    rhs = equation.rhs(equation.target.pointinfo) if callable(equation.rhs) else equation.rhs
    arr = np.asarray(rhs, dtype=complex)
    if arr.ndim == 1:
        vector = arr.reshape(-1)
    elif arr.ndim == 2:
        if output_dim == 1 and arr.shape == _panel_shape(equation.target):
            vector = arr.T.reshape(-1)
        elif arr.shape == (output_dim, point_count):
            vector = arr.reshape(-1)
        elif arr.shape == (point_count, output_dim):
            vector = arr.T.reshape(-1)
        else:
            raise ValueError("rank-2 boundary data has incompatible shape")
    elif arr.ndim == 3:
        if arr.shape[0] != output_dim or arr.shape[1:] != _panel_shape(equation.target):
            raise ValueError("rank-3 boundary data must have shape (component, local_node, panel)")
        vector = arr.swapaxes(1, 2).reshape(-1)
    else:
        raise ValueError("boundary data must be a vector, panel scalar field, or component panel field")
    if vector.size != output_dim * point_count:
        raise ValueError("boundary data has incompatible size")
    return vector


def identity_system_matrix(size: int, *, config: SystemConfig | None = None) -> SystemMatrix:
    return SystemMatrix(np.eye(size), SystemConfig() if config is None else config)


def _trace_matrix(trace: BoundaryTrace, unknown) -> np.ndarray:
    source = trace.layer.source
    target = trace.target
    if not isinstance(source, Chunker) or not isinstance(target, Chunker):
        raise NotImplementedError("dense bootstrap assembly currently supports Chunker sources and targets")
    if trace.layer.kernel.input_dim != unknown.component_count:
        raise ValueError("kernel input dimension must match density component count")

    block = dense_panel_operator_matrix(source.pointinfo, target.pointinfo, trace.layer.kernel)

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


def _panel_shape(geometry: object) -> tuple[int, int]:
    if not hasattr(geometry, "quadrature_order") or not hasattr(geometry, "panel_count"):
        raise TypeError("geometry must expose quadrature_order and panel_count")
    return int(geometry.quadrature_order), int(geometry.panel_count)


def _equation_row_count(equation) -> int:
    return _point_count(equation.target) * _equation_output_dim(equation)


def _equation_output_dim(equation) -> int:
    output_dim: int | None = None
    for term in equation.terms:
        if not isinstance(term, BoundaryTrace):
            raise NotImplementedError("only BoundaryTrace terms are supported in dense bootstrap assembly")
        term_output_dim = term.layer.kernel.output_dim
        if output_dim is None:
            output_dim = term_output_dim
        elif output_dim != term_output_dim:
            raise ValueError("all dense bootstrap terms in an equation must share output dimension")
    if output_dim is None:
        raise ValueError("boundary equation must contain at least one trace term")
    return output_dim
