"""Matrix-free system operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chunkie.quadrature import apply_panel_potential

from .assembly import equation_row_slices, unknown_column_slices
from .density import Density
from .trace import BoundaryTrace


@dataclass(frozen=True)
class SystemOperator:
    system: object

    def matvec(self, vector: NDArray[np.generic]) -> NDArray[np.generic]:
        return matrix_free_matvec(self.system, vector)


def matrix_free_matvec(system, vector: NDArray[np.generic]) -> NDArray[np.generic]:
    """Apply trace terms directly without materializing the dense matrix."""

    vector_array = np.asarray(vector)
    row_slices = equation_row_slices(system.equations)
    column_slices = unknown_column_slices(system.unknowns)
    unknowns = {unknown.name: unknown for unknown in system.unknowns}
    out = np.zeros(sum(row.stop - row.start for row in row_slices.values()), dtype=complex)

    for equation in system.equations:
        row_slice = row_slices[equation.name]
        for term in equation.terms:
            if not isinstance(term, BoundaryTrace):
                raise NotImplementedError(
                    "matrix-free matvec currently supports BoundaryTrace terms"
                )
            unknown = unknowns[term.layer.density]
            column_slice = column_slices[unknown.name]
            if term.layer.coefficient != 0:
                density = Density.from_vector(
                    unknown.name,
                    unknown.geometry,
                    vector_array[column_slice],
                    component_count=unknown.component_count,
                )
                # This is matrix-free only at the system level: kernels are
                # applied to physical point views and contracted with
                # density/weights here instead of first forming the global
                # component-major matrix.
                values = apply_panel_potential(
                    term.layer.source.pointinfo,
                    equation.target.pointinfo,
                    term.layer.kernel,
                    density.component_values,
                )
                out[row_slice] += term.layer.coefficient * values.reshape(-1)
            if term.jump is not None:
                jump_slice = column_slices[term.jump.density]
                out[row_slice] += term.jump.coefficient * vector_array[jump_slice]
    return np.real_if_close(out)
