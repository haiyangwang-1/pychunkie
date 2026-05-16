"""Matrix-free operator wrapper classes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator

from .._legacy import warn_legacy_options
from ..geometry.chunker import Chunker
from ._blocks import _block_kernel_layout, _block_operator_dtype, _is_block_kernel_matrix
from ._common import (
    _flamtype,
    _kernel_opdims,
    _operator_dtype,
    _require_chunker,
    _require_fmm,
    _require_pyflam,
    _uses_special_quadrature,
)
from ._flam import _rskelf_logdet_complete, chunkerflam
from ._fmm import _block_chunkermatapply_fmm, _chunkermatapply_fmm
from ._special import (
    _block_special_correction_matrix,
    _block_uses_special_quadrature,
    _special_correction_matrix,
)
from .options import _NORMALIZED_OPTIONS_MARKER


class ChunkerFMMMatrix(LinearOperator):
    """Matrix-free ``chunkermat`` operator using kernel FMM application."""

    def __init__(
        self,
        chunker: Chunker,
        kernel: Callable[[Any, Any], np.ndarray],
        options: dict[str, Any] | None = None,
    ):
        self.kernel = kernel
        self.options = warn_legacy_options(options, "ChunkerFMMMatrix")
        self.options[_NORMALIZED_OPTIONS_MARKER] = True
        self._correction_mat: spmatrix | None = None
        if _is_block_kernel_matrix(kernel):
            self.chunker = chunker
            self.block_layout = _block_kernel_layout(chunker, kernel)
            self.opdims = None
            for item in self.block_layout.kernels.flat:
                _require_fmm(item)
            dtype = _block_operator_dtype(self.block_layout)
            shape = (int(self.block_layout.row_offsets[-1]), int(self.block_layout.col_offsets[-1]))
        else:
            self.chunker = _require_chunker(chunker)
            self.block_layout = None
            self.opdims = _kernel_opdims(self.chunker, kernel)
            _require_fmm(kernel)
            dtype = _operator_dtype(self.chunker, kernel)
            shape = (
                self.chunker.npt * int(self.opdims[0]),
                self.chunker.npt * int(self.opdims[1]),
            )
        super().__init__(dtype=dtype, shape=shape)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        if x.size != self.shape[1]:
            raise ValueError("density has incompatible size")
        if self.block_layout is not None:
            return _block_chunkermatapply_fmm(
                self.block_layout, x, self.options, self._correction()
            )
        return _chunkermatapply_fmm(self.chunker, self.kernel, x, self.options, self._correction())

    def _matmat(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.shape[1]:
            raise ValueError("density matrix has incompatible row count")
        if x.shape[1] == 0:
            return np.empty((self.shape[0], 0), dtype=np.result_type(self.dtype, x.dtype))
        return np.column_stack([self._matvec(x[:, col]) for col in range(x.shape[1])])

    def toarray(self) -> np.ndarray:
        """Materialize the operator by applying it to basis vectors."""

        return self._matmat(np.eye(self.shape[1], dtype=self.dtype))

    def todense(self) -> np.ndarray:
        return self.toarray()

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        arr = self.toarray()
        if dtype is not None:
            return np.array(arr, dtype=dtype, copy=True if copy is None else copy)
        if copy:
            return arr.copy()
        return arr

    def _correction(self) -> spmatrix | None:
        if self.block_layout is not None:
            if not _block_uses_special_quadrature(self.block_layout, self.options):
                return None
            if self._correction_mat is None:
                self._correction_mat = _block_special_correction_matrix(
                    self.block_layout, self.options
                )
            return self._correction_mat
        if not _uses_special_quadrature(self.kernel, self.options):
            return None
        if self._correction_mat is None:
            self._correction_mat = _special_correction_matrix(
                self.chunker, self.kernel, self.options
            )
        return self._correction_mat


class ChunkerFLAMMatrix(LinearOperator):
    """Matrix-free ``chunkermat`` operator backed by a PyFLAM factorization."""

    def __init__(
        self,
        chunker: Chunker,
        kernel: Callable[[Any, Any], np.ndarray],
        dval: ArrayLike | float | complex = 0.0,
        options: dict[str, Any] | None = None,
    ):
        self.chunker = _require_chunker(chunker) if not _is_block_kernel_matrix(kernel) else chunker
        self.kernel = kernel
        self.options = warn_legacy_options(options, "ChunkerFLAMMatrix")
        self.options[_NORMALIZED_OPTIONS_MARKER] = True
        self.factor = chunkerflam(self.chunker, self.kernel, dval, self.options)
        self.flamtype = _flamtype(self.options)
        if _is_block_kernel_matrix(kernel):
            self.block_layout = _block_kernel_layout(chunker, kernel)
            self.opdims = None
            dtype = _block_operator_dtype(self.block_layout)
            shape = (int(self.block_layout.row_offsets[-1]), int(self.block_layout.col_offsets[-1]))
        else:
            self.block_layout = None
            self.opdims = _kernel_opdims(self.chunker, kernel)
            dtype = _operator_dtype(self.chunker, kernel)
            shape = (
                self.chunker.npt * int(self.opdims[0]),
                self.chunker.npt * int(self.opdims[1]),
            )
        if np.asarray(dval).size:
            dtype = np.result_type(dtype, np.asarray(dval).dtype)
        super().__init__(dtype=dtype, shape=shape)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        if x.size != self.shape[1]:
            raise ValueError("density has incompatible size")
        return self._apply(x)

    def _matmat(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.shape[1]:
            raise ValueError("density matrix has incompatible row count")
        if x.shape[1] == 0:
            return np.empty((self.shape[0], 0), dtype=np.result_type(self.dtype, x.dtype))
        return self._apply(x)

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        if x.size != self.shape[0]:
            raise ValueError("density has incompatible size")
        return self._apply(x, trans="c")

    def _rmatmat(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.shape[0]:
            raise ValueError("density matrix has incompatible row count")
        if x.shape[1] == 0:
            return np.empty((self.shape[1], 0), dtype=np.result_type(self.dtype, x.dtype))
        return self._apply(x, trans="c")

    def solve(self, rhs: ArrayLike, trans: str = "n") -> np.ndarray:
        """Apply the FLAM approximate inverse to one or more right-hand sides."""

        if self.flamtype != "rskelf":
            raise NotImplementedError("solve is only available for rskelf FLAM factors")
        pyflam = _require_pyflam()
        arr = np.asarray(rhs)
        one_dim = arr.ndim == 1
        if one_dim:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != self.shape[0]:
            raise ValueError("right-hand side has incompatible row count")
        out = pyflam.rskelf_partial_sv(self.factor, arr, trans=trans)
        return out[:, 0] if one_dim else out

    def logdet(self):
        """Return the FLAM log-determinant, including any residual skeleton block."""

        if self.flamtype != "rskelf":
            raise NotImplementedError("logdet is only available for rskelf FLAM factors")
        return _rskelf_logdet_complete(self.factor)

    def toarray(self) -> np.ndarray:
        return self._matmat(np.eye(self.shape[1], dtype=self.dtype))

    def todense(self) -> np.ndarray:
        return self.toarray()

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        arr = self.toarray()
        if dtype is not None:
            return np.array(arr, dtype=dtype, copy=True if copy is None else copy)
        if copy:
            return arr.copy()
        return arr

    def _apply(self, x: ArrayLike, trans: str = "n") -> np.ndarray:
        pyflam = _require_pyflam()
        arr = np.asarray(x)
        one_dim = arr.ndim == 1
        if one_dim:
            arr = arr.reshape(-1, 1)
        if self.flamtype == "rskelf":
            out = pyflam.rskelf_partial_mv(self.factor, arr, trans=trans)
        elif self.flamtype == "rskel":
            out = pyflam.rskel_mv(self.factor, arr, trans=trans)
        else:
            raise NotImplementedError(f"unsupported FLAM factor type {self.flamtype!r}")
        return out[:, 0] if one_dim else out
