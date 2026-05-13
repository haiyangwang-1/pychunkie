"""Python port of the MATLAB ``chunkIE`` boundary-integral toolkit.

The public facade is intentionally small and mirrors the MATLAB package names
where practical:

``Chunker`` and the ``chunker*`` constructors build Legendre-panel boundary
discretizations for smooth curves, fitted point clouds, true polygons, and
sampled node arrays. ``ChunkGraph`` and the graph helpers collect multiple
chunker edges into region-aware interfaces for multiply connected or
multi-region boundary value problems.

``Kernel`` and :func:`kernel` wrap the supported physics kernels: Laplace,
Helmholtz, Stokes, biharmonic, elasticity, zero/NaN placeholders, and
interleaved block systems. ``chunkermat``, ``chunkermatapply``,
``chunkerkerneval``, and ``chunkerkernevalmat`` provide dense, FMM-backed, and
FLAM-backed operator assembly/application paths.

Most low-level arrays follow MATLAB's ``dim x k x nch`` geometry layout and
Fortran-order flattening of panel nodes. Boundary-integral jump terms such as
``+/- 1/2 I`` are not hidden in kernel objects; callers add them explicitly
when forming a particular interior or exterior BVP equation.
"""

from . import lege
from .chunker import (
    Chunker,
    ChunkerPref,
    chunker,
    chunkerfit,
    chunkerfunc,
    chunkerfuncuni,
    chunkerpoints,
    chunkerpoly,
    chunkerpref,
    merge,
)
from .chunkgraph import ChunkGraph, chunkgraph, chunkgraphinregion, find_edge_regions, tochunkgraph
from .domain import (
    HypOctNode,
    HypOctTree,
    checkcurveparam,
    ellipse,
    hypoct_uni,
    mergeregions,
    nonflatinterface,
    pointinregion,
    redblue,
    regioninside,
    starfish,
)
from .kernel import Kernel, kernel
from .operators import (
    ChunkerFLAMMatrix,
    ChunkerFMMMatrix,
    ChunkerRCIPMatrix,
    PointInfo,
    RCIPContext,
    chunkerinterior,
    chunkerintegral,
    chunkerflam,
    chunkerkerneval,
    chunkerkernevalmat,
    chunkermat,
    chunkermatapply,
    pointinfo,
)

__all__ = [
    "Chunker",
    "ChunkerPref",
    "ChunkGraph",
    "Kernel",
    "ChunkerFLAMMatrix",
    "ChunkerFMMMatrix",
    "ChunkerRCIPMatrix",
    "PointInfo",
    "RCIPContext",
    "HypOctNode",
    "HypOctTree",
    "checkcurveparam",
    "chunker",
    "chunkerfit",
    "chunkerflam",
    "chunkerfunc",
    "chunkerfuncuni",
    "chunkerpoints",
    "chunkerpoly",
    "chunkgraph",
    "chunkgraphinregion",
    "find_edge_regions",
    "chunkerinterior",
    "chunkerintegral",
    "chunkerkerneval",
    "chunkerkernevalmat",
    "chunkermat",
    "chunkermatapply",
    "chunkerpref",
    "ellipse",
    "hypoct_uni",
    "kernel",
    "mergeregions",
    "merge",
    "nonflatinterface",
    "lege",
    "pointinregion",
    "pointinfo",
    "redblue",
    "regioninside",
    "starfish",
    "tochunkgraph",
]
