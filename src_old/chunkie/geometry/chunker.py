"""Public chunker facade.

Implementation lives in smaller private modules split by responsibility.
"""

from __future__ import annotations

from ._chunker_class import Chunker, ChunkerPref
from ._chunker_fit import chunkerfit, chunkerfuncuni
from ._chunker_func import chunkerfunc
from ._chunker_options import (
    _LEGACY_OPTIONS_MARKER as _LEGACY_OPTIONS_MARKER,
)
from ._chunker_options import (
    _legacy_options as _legacy_options,
)
from ._chunker_options import (
    _set_option as _set_option,
)
from ._chunker_points import chunkerpoints, merge
from ._chunker_poly import chunkerpoly

__all__ = [
    "Chunker",
    "ChunkerPref",
    "chunkerfit",
    "chunkerfunc",
    "chunkerfuncuni",
    "chunkerpoints",
    "chunkerpoly",
    "merge",
]
