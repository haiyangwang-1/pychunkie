"""Python-first boundary-integral toolkit inspired by MATLAB ``chunkIE``.

The public facade is intentionally small:

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

Geometry storage uses chunk tensors such as ``positions[R, s, S]``. Flat
vectors and interleaved matrices are adapter formats for solvers, sparse/FMM
backends, FLAM callbacks, and fixture comparisons. Boundary-integral jump terms
such as ``+/- 1/2 I`` are not hidden in kernel objects; callers add them
explicitly when forming a particular interior or exterior BVP equation.
"""

from . import lege
from .geometry import PointInfo
from .geometry.chunker import (
    Chunker,
    ChunkerPref,
    chunkerfit,
    chunkerfunc,
    chunkerfuncuni,
    chunkerpoints,
    chunkerpoly,
    merge,
)
from .geometry.chunkgraph import ChunkGraph, chunkgraphinregion, find_edge_regions, tochunkgraph
from .geometry.domain import (
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
from .kernels import Kernel, kernel
from .operators import (
    ChunkerFLAMMatrix,
    ChunkerFMMMatrix,
    ChunkerRCIPMatrix,
    RCIPContext,
    chunkerflam,
    chunkerintegral,
    chunkerinterior,
    chunkerkerneval,
    chunkerkernevalmat,
    chunkermat,
    chunkermatapply,
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
    "chunkerfit",
    "chunkerflam",
    "chunkerfunc",
    "chunkerfuncuni",
    "chunkerpoints",
    "chunkerpoly",
    "chunkgraphinregion",
    "find_edge_regions",
    "chunkerinterior",
    "chunkerintegral",
    "chunkerkerneval",
    "chunkerkernevalmat",
    "chunkermat",
    "chunkermatapply",
    "ellipse",
    "hypoct_uni",
    "kernel",
    "mergeregions",
    "merge",
    "nonflatinterface",
    "lege",
    "pointinregion",
    "redblue",
    "regioninside",
    "starfish",
    "tochunkgraph",
]
