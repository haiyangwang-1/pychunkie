"""Matrix-free system operators."""

from __future__ import annotations


class SystemOperator:
    def matvec(self, vector):
        raise NotImplementedError("Matrix-free matvecs follow dense reference assembly")
