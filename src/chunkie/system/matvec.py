"""Matrix-free system operators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .assembly import equation_row_slices, unknown_column_slices
from .backends.fmm2d import apply_fmm
from .density import Density
from .trace import BoundaryTrace


class SystemOperator:
    def matvec(self, vector):
        raise NotImplementedError("Matrix-free matvecs follow dense reference assembly")


def fmm_matvec(system, vector: NDArray[np.generic], *, eps: float = 1.0e-12) -> NDArray[np.generic]:
    """Apply supported system trace terms with the FMM backend."""

    vector_array = np.asarray(vector)
    row_slices = equation_row_slices(system.equations)
    column_slices = unknown_column_slices(system.unknowns)
    unknowns = {unknown.name: unknown for unknown in system.unknowns}
    out = np.zeros(sum(row.stop - row.start for row in row_slices.values()), dtype=complex)

    for equation in system.equations:
        row_slice = row_slices[equation.name]
        for term in equation.terms:
            if not isinstance(term, BoundaryTrace):
                raise NotImplementedError("FMM matvec currently supports BoundaryTrace terms")
            unknown = unknowns[term.layer.density]
            column_slice = column_slices[unknown.name]
            density = Density.from_vector(
                unknown.name,
                unknown.geometry,
                vector_array[column_slice],
                component_count=unknown.component_count,
            )
            # FMM is a backend boundary: it consumes physical source/target
            # point views and weighted scalar densities, while the system layer
            # handles named block slicing and vector reconstruction.
            values = apply_fmm(
                term.layer.source.pointinfo,
                equation.target.pointinfo,
                term.layer.kernel,
                density.values,
                eps=eps,
            )
            out[row_slice] += term.layer.coefficient * values.reshape(-1)
            if term.jump is not None:
                jump_slice = column_slices[term.jump.density]
                out[row_slice] += term.jump.coefficient * vector_array[jump_slice]
    return np.real_if_close(out)
