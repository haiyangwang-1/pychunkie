"""FMM-backed kernel wrappers and selector exports."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._fmm_biharmonic import _biharm2d_fmm
from ._fmm_common import (
    _conj_fmm,
    _derived_fmm,
    _direct_fmm,
    _interleave_dtype,
    _interleave_fmm,
    _interleave_indices,
    _scale_fmm,
    _sum_fmm,
    _target_count,
)
from ._fmm_elasticity import _elast2d_fmm
from ._fmm_helmholtz import _helm2d_fmm
from ._fmm_laplace import _lap2d_fmm
from ._fmm_stokes import _stok2d_fmm


class FmmKernel:
    """Callable FMM adapter built from a kernel object with an ``fmm`` evaluator."""

    def __init__(self, kernel: Any) -> None:
        if getattr(kernel, "fmm", None) is None:
            raise ValueError("kernel does not define an FMM evaluator")
        self.kernel = kernel
        self.name = getattr(kernel, "name", "custom")
        self.type = getattr(kernel, "type", "custom")
        self.opdims = getattr(kernel, "opdims", (0, 0))
        self.sing = getattr(kernel, "sing", "")
        self.params = dict(getattr(kernel, "params", {}))

    def __call__(self, eps: float, source: Any, target: Any, density: np.ndarray) -> Any:
        return self.kernel.fmm(eps, source, target, density)


__all__ = [
    "FmmKernel",
    "_biharm2d_fmm",
    "_conj_fmm",
    "_derived_fmm",
    "_direct_fmm",
    "_elast2d_fmm",
    "_helm2d_fmm",
    "_interleave_dtype",
    "_interleave_fmm",
    "_interleave_indices",
    "_lap2d_fmm",
    "_scale_fmm",
    "_stok2d_fmm",
    "_sum_fmm",
    "_target_count",
]
