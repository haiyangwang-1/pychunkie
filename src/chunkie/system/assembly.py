"""Dense system assembly."""

from __future__ import annotations

import numpy as np

from chunkie.quadrature import dense_panel_operator_matrix

from .config import SystemConfig
from .matrix import SystemMatrix
from .trace import BoundaryTrace


def assemble_system_matrix(system, *, config: SystemConfig) -> SystemMatrix:
    # Dense assembly keeps named equation and density blocks until this
    # explicit row/column-slice boundary, where the solver-facing matrix is
    # flattened into component-major linear algebra layout.
    row_slices = _equation_row_slices(system.equations)
    column_slices = _unknown_column_slices(system.unknowns)
    unknowns = {unknown.name: unknown for unknown in system.unknowns}
    row_count = sum(row_slice.stop - row_slice.start for row_slice in row_slices.values())
    column_count = sum(column_slice.stop - column_slice.start for column_slice in column_slices.values())
    matrix = np.zeros((row_count, column_count), dtype=complex)

    for equation in system.equations:
        row_slice = row_slices[equation.name]
        for term in equation.terms:
            if not isinstance(term, BoundaryTrace):
                raise NotImplementedError("only BoundaryTrace terms are supported in dense bootstrap assembly")
            try:
                unknown = unknowns[term.layer.density]
            except KeyError as exc:
                raise ValueError(f"trace term references unknown density {term.layer.density!r}") from exc
            column_slice = column_slices[unknown.name]
            expected_shape = (
                row_slice.stop - row_slice.start,
                column_slice.stop - column_slice.start,
            )
            if term.layer.coefficient != 0:
                block = _trace_matrix(term, unknown)
                if block.shape != expected_shape:
                    raise ValueError("trace block shape does not match dense system layout")
                matrix[row_slice, column_slice] += term.layer.coefficient * block
            if term.jump is not None:
                try:
                    jump_unknown = unknowns[term.jump.density]
                except KeyError as exc:
                    raise ValueError(f"jump term references unknown density {term.jump.density!r}") from exc
                jump_slice = column_slices[jump_unknown.name]
                jump_shape = (row_slice.stop - row_slice.start, jump_slice.stop - jump_slice.start)
                if jump_shape[0] != jump_shape[1]:
                    raise ValueError("jump terms require matching row and column layout")
                matrix[row_slice, jump_slice] += term.jump.coefficient * np.eye(*jump_shape)

    return SystemMatrix(
        matrix,
        config,
        diagnostics={
            "assembly": "dense",
            "unknowns": tuple(unknown.name for unknown in system.unknowns),
            "equations": tuple(equation.name for equation in system.equations),
        },
    )


def rhs_vector(system) -> np.ndarray:
    return np.concatenate([_equation_rhs_vector(equation) for equation in system.equations])


def _equation_rhs_vector(equation) -> np.ndarray:
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


def unknown_column_slices(unknowns) -> dict[str, slice]:
    return _unknown_column_slices(unknowns)


def equation_row_slices(equations) -> dict[str, slice]:
    return _equation_row_slices(equations)


def _equation_row_slices(equations) -> dict[str, slice]:
    offset = 0
    out: dict[str, slice] = {}
    for equation in equations:
        if equation.name in out:
            raise ValueError(f"duplicate equation name {equation.name!r}")
        row_count = _equation_row_count(equation)
        out[equation.name] = slice(offset, offset + row_count)
        offset += row_count
    return out


def _unknown_column_slices(unknowns) -> dict[str, slice]:
    offset = 0
    out: dict[str, slice] = {}
    for unknown in unknowns:
        if unknown.name in out:
            raise ValueError(f"duplicate unknown density name {unknown.name!r}")
        column_count = _point_count(unknown.geometry) * unknown.component_count
        out[unknown.name] = slice(offset, offset + column_count)
        offset += column_count
    return out


def identity_system_matrix(size: int, *, config: SystemConfig | None = None) -> SystemMatrix:
    return SystemMatrix(np.eye(size), SystemConfig() if config is None else config)


def _trace_matrix(trace: BoundaryTrace, unknown) -> np.ndarray:
    source = trace.layer.source
    target = trace.target
    if not hasattr(source, "pointinfo") or not hasattr(target, "pointinfo"):
        raise NotImplementedError("dense bootstrap assembly requires source and target pointinfo views")
    if trace.layer.kernel.input_dim != unknown.component_count:
        raise ValueError("kernel input dimension must match density component count")

    block = dense_panel_operator_matrix(source.pointinfo, target.pointinfo, trace.layer.kernel)

    # Smooth closed-curve double-layer self blocks have a finite diagonal limit.
    # We insert the local limit here so the dense reference path is usable before
    # special quadrature owns same-panel correction.
    if (
        source is target
        and hasattr(source, "signed_curvature")
        and trace.layer.kernel.family == "laplace"
        and trace.layer.kernel.selector == "d"
    ):
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
