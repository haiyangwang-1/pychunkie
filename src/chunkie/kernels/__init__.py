"""PDE kernel formulas and singularity metadata."""

from .base import Kernel
from .registry import kernel
from .singularities import (
    GeometryRequirements,
    LaplaceBasis,
    LaplaceSingularExpansion,
    LaplaceSingularTerm,
    SingularityInfo,
)

__all__ = [
    "GeometryRequirements",
    "Kernel",
    "LaplaceBasis",
    "LaplaceSingularExpansion",
    "LaplaceSingularTerm",
    "SingularityInfo",
    "kernel",
]
