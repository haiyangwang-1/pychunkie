# pychunkie Implementation Map

This map is a working guide for porting MATLAB `chunkIE` into Python. It maps the current Python implementation to the MATLAB reference checkout under `external/chunkie-matlab/chunkie`.

## Legend

- ✅ implemented in Python
- 🧪 covered by the Python test suite
- 🎯 compared against MATLAB-generated golden data; same behavior within the checked tolerances
- ⚠️ implemented with a known limitation, reduced scope, or intentional approximation
- 🚧 not implemented or explicitly deferred
- 🚫 explicit non-goal; do not implement
- 🧩 private/internal helper
- 🧭 support/reference file rather than package API

Verification snapshot: `uv run pytest` on 2026-05-12 with Python 3.11.9 collected 327 tests: `327 passed`. Full MATLAB parity runs generate ignored `tests/golden/*.mat` files on demand and require a populated `external/chunkie-matlab` checkout.

Updated for commits after `2568a934c759aaf614c48f428678da8f6bbcb39f`:

- `fbca951` Add rounded polygon smoother workflow
- `fd768bb` Add PV HS GGQ table support
- `3c1cb43` Add interleaved kernel blocks
- `0bee9f2` Add biharmonic Green kernels

The ignored directories `.venv/`, `.pytest_cache/`, and `.git/` are not expanded as repo structure here. Third-party/reference checkouts under `external/` are pinned as git submodules and summarized in the support tree; the relevant MATLAB reference paths are listed beside the Python nodes.

## Source Tree

```text
src/
└── chunkie/
    ├── __init__.py
    ├── chunker.py
    │   ├── class ChunkerPref
    │   │   └── from_any
    │   ├── class Chunker
    │   │   ├── __init__
    │   │   ├── properties: k, dim, npt, datadim, nvert, vertdeg
    │   │   ├── storage properties: r, d, d2, n, wts, adj, data
    │   │   ├── copy, addchunk, resize, makedatarows, cleardata
    │   │   ├── weights, normals, tangents
    │   │   ├── arclengthdens, arclengthder, arclengthfun
    │   │   ├── chunklen, chunkends, area, signed_curvature
    │   │   ├── exps, diffmat, intmat, onesmat, normonesmat
    │   │   ├── centroids, datares, sortinfo, checkadjinfo, sort
    │   │   ├── flagnear, flagnear_rectangle, flagnear_rectangle_grid, nearest
    │   │   ├── min, max, recompute_geometry, upsample, split, refine, arcresample
    │   │   ├── translate, transform, reverse, move, rotate, reflect
    │   │   └── __add__, __radd__, __mul__, __rmul__, __rmatmul__
    │   ├── chunker, chunkerpref
    │   ├── chunkerfunc, chunkerfuncuni, chunkerfit, chunkerpoly, chunkerpoints
    │   ├── merge
    │   └── private helpers
    ├── chunkgraph.py
    │   ├── class SourceInfo
    │   ├── class ChunkGraph
    │   │   ├── __init__
    │   │   ├── properties: npt, k, dim, datadim, r, d, d2, n, wts, data, adj
    │   │   ├── sourceinfo, merged, build_v2emat, procverts, findregions
    │   │   ├── slicegraph, edgeids, refine, copy
    │   │   ├── translate, transform, rotate, reflect
    │   │   ├── min, max, onesmat, normonesmat
    │   │   ├── flagnear, flagnear_rectangle, flagnear_rectangle_grid
    │   │   └── __add__, __radd__, __mul__, __rmul__, __rmatmul__
    │   ├── chunkgraph, tochunkgraph, chunkgraphinregion
    │   └── private graph/geometry helpers
    ├── domain.py
    │   ├── checkcurveparam, ellipse, starfish, nonflatinterface
    │   ├── redblue, hypoct_uni
    │   ├── pointinregion, regioninside, mergeregions
    │   ├── class HypOctNode, class HypOctTree
    │   └── private helpers
    ├── kernel.py
    │   ├── class Kernel
    │   │   ├── __call__
    │   │   ├── __add__, __sub__, __neg__, __mul__, __rmul__, __truediv__
    │   │   ├── conj, conjugate
    │   │   └── zeros, nans
    │   ├── kernel
    │   ├── lap2d_kernel, helm2d_kernel, helm1d_kernel, biharm2d_kernel
    │   ├── stok2d_kernel, elast2d_kernel
    │   ├── zeros, nans, interleave
    │   └── private helpers
    ├── operators.py
    │   ├── class PointInfo
    │   ├── class ChunkerFMMMatrix
    │   ├── class ChunkerFLAMMatrix
    │   ├── pointinfo
    │   ├── chunkermat, chunkermatapply, chunkerflam
    │   ├── chunkerintegral, chunkerinterior
    │   ├── chunkerkerneval, chunkerkernevalmat
    │   └── private helpers
    ├── chnk/
    │   ├── __init__.py
    │   ├── arcparam.py
    │   │   ├── class ArcParamData
    │   │   ├── init
    │   │   └── eval
    │   ├── curves.py
    │   │   ├── linefunc, fpara, fsine, bymode
    │   │   └── _pack
    │   ├── flam.py
    │   │   ├── kernbyindex, kernbyindexr
    │   │   ├── proxy_square_pts, proxy_circ_pts, proxy_rect_pts, nproxy_square
    │   │   └── proxyfun, proxyfunr
    │   ├── biharm2d.py
    │   │   ├── green, kern
    │   │   └── _require
    │   ├── elast2d.py
    │   │   ├── kern
    │   │   └── private helpers
    │   ├── geometry.py
    │   │   ├── perp, normal2d, curvature2d
    │   │   ├── flagnear, flagnear_rectangle, flagnear_rectangle_grid, flagself
    │   │   ├── chunk_nearparam
    │   │   └── _ptinfo_field
    │   ├── helm1d.py
    │   │   ├── green, kern, sweep
    │   │   └── private helpers
    │   ├── helm2d.py
    │   │   ├── green, kern
    │   │   └── _require
    │   ├── lap2d.py
    │   │   ├── green, kern
    │   │   └── _require
    │   ├── quadggq.py
    │   │   ├── class AuxQuad
    │   │   ├── setup, getlogquad, logavail, hqsuppavail
    │   │   ├── gethqsuppquad, getremovablequad, getpvquad, gethsquad
    │   │   ├── buildmat, buildmattd, diagbuildmat, nearbuildmat
    │   │   └── private helpers
    │   ├── quadadap.py
    │   │   └── buildmat
    │   ├── quadnative.py
    │   │   ├── buildmat
    │   │   └── _pointinfo_for_chunks
    │   ├── rcip.py
    │   │   ├── class RCIPSaved, class RCIPChunkGraphResult
    │   │   ├── IPinit, Pbcinit, setup, SchurBana
    │   │   ├── Rcompchunk, rhohatInterp, corner_refine, chunkgraph_rcip
    │   │   └── lowercase MATLAB-style aliases
    │   ├── smoother.py
    │   │   ├── class UniformMesh, class SmoothMesh
    │   │   ├── get_umesh, get_mesh, smooth
    │   │   ├── smooth_curve, smooth_curve2, smooth_curve3
    │   │   └── _panel_nodes
    │   ├── spcl.py
    │   │   └── absconvgauss
    │   └── stok2d.py
    │       ├── kern
    │       └── private helpers
    ├── data/
    │   └── quadggq/
    │       ├── metadata.npz
    │       ├── ggqnear*.npz
    │       ├── ggqself_*.npz
    │       ├── hsupp_*.npz
    │       └── hqsupp_*.npz
    └── lege/
        ├── __init__.py
        └── core.py
            ├── pol, pols, exps, rts, rts_stab
            ├── exev, derpol, dermat
            ├── intpol, intmat, matrin
            └── barywts, adapgauss, bernstein_ellipse, polsum, tayl
```

## Package Exports

- ✅ [src/chunkie/__init__.py](src/chunkie/__init__.py) exports the public Python API. MATLAB has no direct single-file equivalent; it is a Python package facade over MATLAB class folders and package folders.
- ✅ 🧪 [src/chunkie/domain.py](src/chunkie/domain.py) implements top-level MATLAB geometry/domain helpers exported from the Python package facade.
- ✅ [src/chunkie/chnk/__init__.py](src/chunkie/chnk/__init__.py) mirrors MATLAB `+chnk` package exports, including the newer `biharm2d`, `flam`, `quadadap`, `rcip`, and `smoother` modules.
- ✅ [src/chunkie/lege/__init__.py](src/chunkie/lege/__init__.py) mirrors MATLAB `+lege` package exports.


## Python To MATLAB Map

### I GEOMETRY
#### `chunker.py`

- ✅ 🧪 🎯 [src/chunkie/chunker.py](src/chunkie/chunker.py) is the main Python home for MATLAB `@chunker`, top-level chunker constructors, and related factory functions.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `chunkerfit` | ⚠️ 🧪 | `chunkerfit.m` | Spline/open-line/circle paths tested; remaining MATLAB fitting modes are deferred. |
| `ChunkerPref` | ✅ 🧪 🎯 | `@chunkerpref/chunkerpref.m` | Python dataclass-like preference holder; MATLAB fixture checks explicit preference fields. |
| `ChunkerPref.from_any` | ✅ 🧪 🎯 | `@chunkerpref/chunkerpref.m` | Python adapter for dict/None/preference inputs; covered through the `chunkerpref` wrapper fixture. |
| `_curve_outputs`, `_remap_adjacency` | 🧩 ✅ | Internal Python helpers | No direct MATLAB file. |
| `copy` | ✅ 🧪 🎯 | MATLAB value-copy behavior | Python explicit copy helper mirrors MATLAB value-object copy behavior under mutation. |
| `resize` | ✅ 🧪 🎯 | `@chunker/chunker.m` storage behavior | Storage growth and preserved live data are MATLAB-fixture tested. |
| `makedatarows` | ✅ 🧪 🎯 | `@chunker/makedatarows.m` | Data row allocation and preservation across expansion are MATLAB-fixture tested. |
| `cleardata` | ✅ 🧪 🎯 | `@chunker/chunker.m` data field behavior | Data-row reset behavior is MATLAB-fixture tested. |
| `checkadjinfo` | ✅ 🧪 🎯 | `@chunker/checkadjinfo.m` | Python tested through adjacency checks and MATLAB fixture parity. |
| `sort` | ✅ 🧪 🎯 | `@chunker/sort.m` | Python tested on open segments; strict geometry fixture compares sorted chunker fields. |
| `chunkerpref` | ✅ 🧪 🎯 | `@chunkerpref/chunkerpref.m` | Python preference wrapper has MATLAB fixture coverage for explicit field overrides. |
| `chunkerpoly` | ✅ 🧪 | `chunkerpoly.m`, `+chnk/+smoother/*` workflows | Straight-edge and rounded-corner polygon paths are implemented and tested. |
| `_rounded_chunkerpoly`, `_polygon_widths`, `_fill_line_chunk`, `_fill_quadratic_chunk` | 🧩 ✅ 🧪 | `chunkerpoly.m`, `+chnk/+smoother/*` concepts | Internal rounded polygon construction helpers. |
| `flagnear` | ✅ 🧪 🎯 | `@chunker/flagnear.m`, `+chnk/flagnear*` helpers | Python tested against brute-force distances and MATLAB fixture flags. |
| `flagnear_rectangle` | ✅ 🧪 🎯 | `@chunker/flagnear_rectangle.m` | 2D rectangle flags covered by direct MATLAB fixture values. |
| `flagnear_rectangle_grid` | ✅ 🧪 🎯 | `@chunker/flagnear_rectangle_grid.m` | Python meshgrid ordering and MATLAB fixture values covered. |
| `nearest` | ✅ 🧪 🎯 | `@chunker/nearest.m` | Python vectorized nearest results match MATLAB scalar-reference fixture columns. |
| `recompute_geometry` | ✅ 🧪 🎯 | MATLAB geometry recomputation inside constructors/transforms | Python tested after transforms/refinement and MATLAB fixture recomputation. |
| `translate` | ✅ 🧪 🎯 | `@chunker/plus.m` | Python operator helper covered against MATLAB translated chunkers. |
| `__add__`, `__radd__` | ✅ 🧪 🎯 | `@chunker/plus.m` | Left/right translation operators covered by MATLAB fixture values. |
| `chunkerfuncuni` | ✅ 🧪 🎯 | `chunkerfuncuni.m` | Uniform panel count tested; fixture covers MATLAB-compatible uniform geometry and spectral second derivatives. |
| `weights` | ✅ 🧪 🎯 | `@chunker/weights.m`, `@chunker/whts.m` | Returns quadrature weights; `geometry_core.mat` covers a noncircular chunker. |
| `normals` | ✅ 🧪 🎯 | `@chunker/normals.m` | 2D only; raises for non-2D; strict geometry fixture covers stored and recomputed normals. |
| `tangents` | ✅ 🧪 🎯 | `@chunker/tangents.m`, `@chunker/taus.m` | Tangent vectors from derivatives. |
| `arclengthdens` | ✅ 🧪 🎯 | `@chunker/arclengthdens.m` | Tested directly, through arc parameterization, and against MATLAB fixture data. |
| `arclengthder` | ✅ 🧪 🎯 | `@chunker/arclengthder.m` | Tested on circle data and MATLAB fixture values. |
| `arclengthfun` | ✅ 🧪 🎯 | `@chunker/arclengthfun.m` | Tested on circle data and MATLAB fixture values. |
| `chunkends` | ✅ 🧪 🎯 | `@chunker/chunkends.m` | Endpoint and tangent normalization behavior parity-tested. |
| `signed_curvature` | ✅ 🧪 🎯 | `@chunker/signed_curvature.m` | Strict geometry fixture covers noncircular curvature values. |
| `datares` | ✅ 🧪 🎯 | `@chunker/datares.m` | Python tested for high-order data flags and MATLAB fixture parity. |
| `sortinfo` | ✅ 🧪 🎯 | `@chunker/sortinfo.m` | Strict geometry fixture covers sorted indices, adjacency, and info fields. |
| `min`, `max` | ✅ 🧪 🎯 | `@chunker/min.m`, `@chunker/max.m` | Tested against nodewise extrema and MATLAB fixture values. |
| `upsample` | ✅ 🧪 🎯 | `@chunker/upsample.m` | Python tested with density transfer and MATLAB fixture parity. |
| `split` | ✅ 🧪 🎯 | `@chunker/split.m` | Focused parameter-space split parity covers geometry, weights, and adjacency. |
| `refine` | ✅ 🧪 🎯 | `@chunker/refine.m` | Splits selected chunks, enforces max chunk length, arc-length level restriction, and oversampling; fixture covers selected split plus oversampling. |
| `arcresample` | ✅ 🧪 🎯 | `@chunker/arcresample.m`, `+chnk/+arcparam/*` | Python tested for near-constant panel speed and MATLAB geometry parity. |
| `reverse` | ✅ 🧪 🎯 | `@chunker/reverse.m` | Python tested with polygon helpers and MATLAB fixture parity. |
| `rotate` | ✅ 🧪 🎯 | `@chunker/rotate.m` | Strict geometry fixture covers rotation about source/destination centers. |
| `reflect` | ✅ 🧪 🎯 | `@chunker/reflect.m` | Strict geometry fixture covers reflection about shifted lines. |
| `chunkerpoints` | ✅ 🧪 🎯 | `chunkerpoints.m`, `@chunker/chunkerpoints.m` | Python tested with optional derivatives and MATLAB fixture parity. |
| `merge` | ✅ 🧪 🎯 | `@chunker/merge.m` | Python tested for chunker/data row padding and MATLAB fixture parity. |
| `Chunker` | ✅ 🧪 🎯 | `@chunker/chunker.m` | Core storage and geometry object; MATLAB parity fixtures cover construction fields and many geometry transforms. |
| `Chunker.__init__` | ✅ 🧪 🎯 | `@chunker/chunker.m` | Constructor defaults and validation covered by tests. |
| `Chunker.k`, `dim`, `npt`, `datadim`, `nvert`, `vertdeg` | ✅ 🧪 🎯 | `@chunker/chunker.m` fields/properties | `k`, `dim`, geometry fields parity-tested through fixtures. |
| `Chunker.r`, `d`, `d2`, `n`, `wts`, `adj`, `data` | ✅ 🧪 🎯 | `@chunker/chunker.m` fields | `r`, `d`, `d2`, `n`, `wts`, `adj` checked against MATLAB fixtures. |
| `addchunk` | ✅ 🧪 🎯 | `@chunker/chunker.m` storage behavior | Used by MATLAB fixture reconstruction. |
| `chunklen` | ✅ 🧪 🎯 | `@chunker/chunklen.m` | MATLAB parity fixture checks values. |
| `area` | ✅ 🧪 🎯 | `@chunker/area.m` | MATLAB parity fixture checks values. |
| `exps` | ✅ 🧪 🎯 | `@chunker/exps.m`, `+lege/exps.m` | MATLAB parity via Legendre fixture and chunker fixture. |
| `diffmat` | ✅ 🧪 🎯 | `@chunker/diffmat.m` | MATLAB parity fixture checks first and second derivative matrices. |
| `intmat` | ✅ 🧪 🎯 | `@chunker/intmat.m` | MATLAB parity fixture checks chunk-order matrix. |
| `onesmat` | ✅ 🧪 🎯 | `@chunker/onesmat.m` | MATLAB parity fixture checks values. |
| `normonesmat` | ✅ 🧪 🎯 | `@chunker/normonesmat.m` | MATLAB parity fixture checks values. |
| `centroids` | ✅ 🧪 🎯 | `@chunker/centroids.m` | MATLAB parity fixture checks values. |
| `transform` | ✅ 🧪 🎯 | `@chunker/mtimes.m` | MATLAB parity fixture checks matrix transform. |
| `move` | ✅ 🧪 🎯 | `@chunker/move.m` | MATLAB parity fixture checks move/rotate/scale composition. |
| `__mul__`, `__rmul__`, `__rmatmul__` | ✅ 🧪 🎯 | `@chunker/mtimes.m` | Scalar and matrix transform behavior. |
| `chunker` | ✅ 🧪 🎯 | `@chunker/chunker.m` | Python constructor wrapper. |
| `chunkerfunc` | ✅ 🧪 🎯 | `chunkerfunc.m` | Circle fixture parity plus adaptive curve/speed resolution, level restriction, max-length splitting, and oversampling. |

#### `chunkgraph.py`

- ✅ 🧪 🎯 [src/chunkie/chunkgraph.py](src/chunkie/chunkgraph.py) maps the Python chunk graph object to MATLAB `@chunkgraph` plus top-level graph helpers.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `findregions` | ✅ 🧪 🎯 ⚠️ | `@chunkgraph/findregions.m` | MATLAB-style oriented face walking is fixture-tested for bounded cycles, adjacent regions, nested regions, bridge/loop/multiply connected counts, and region-id behavior; Python still keeps the outside region as an empty sentinel instead of MATLAB's signed unbounded loop. |
| private graph helpers | 🧩 ✅ | Internal Python helpers | Include edge normalization, subchunking, oriented face walks, polygon tests, and nesting-aware bounded-face ordering. |
| `copy` | ✅ 🧪 🎯 | MATLAB value-copy behavior | Fixture checks graph and edge-chunker storage mutation isolation against MATLAB value-object behavior. |
| `procverts` | ✅ 🧪 🎯 | `@chunkgraph/procverts.m` | MATLAB fixture covers counterclockwise tangent ordering and incident-edge signs. |
| `refine` | ✅ 🧪 🎯 | `@chunkgraph/refine.m` | Fixture covers per-edge refinement with sorted refined chunks and refreshed graph metadata. |
| `flagnear*` | ✅ 🧪 🎯 | `@chunkgraph/flagnear*.m` | Merged chunker near flags, rectangle flags, and grid flags are MATLAB-fixture tested. |
| operator overloads | ✅ 🧪 🎯 | `@chunkgraph/plus.m`, `mtimes.m` | Left/right translation, scalar scaling, and NumPy-left matrix transforms are MATLAB-fixture tested. |
| `tochunkgraph` | ✅ 🧪 🎯 | `@chunker/tochunkgraph.m` | Closed-loop and open-line conversions are MATLAB-fixture tested. |
| `chunkgraphinregion` | ✅ 🧪 🎯 | `chunkgraphinregion.m` | Point, meshgrid, transformed adjacent-region, and nested-region ids are MATLAB-fixture tested. |
| `merged` | ✅ 🧪 🎯 | `@chunkgraph/*` merged geometry behavior | MATLAB fixture checks merged field access through `r`, `d`, `d2`, `n`, `wts`, and `adj`. |
| `min`, `max` | ✅ 🧪 🎯 | `@chunkgraph/min.m`, `@chunkgraph/max.m` | MATLAB fixture covers nodewise extrema. |
| `SourceInfo` | ✅ 🧪 🎯 | MATLAB source-info structs | Python dataclass used by dense operators; fixture checks flattened source fields. |
| `ChunkGraph` | ✅ 🧪 🎯 | `@chunkgraph/chunkgraph.m` | Core graph container. |
| `ChunkGraph.__init__` | ✅ 🧪 🎯 | `@chunkgraph/chunkgraph.m` | Edge/vertex construction parity-tested on square graphs plus devtools pentagon and basic legacy-incidence versus `edgesendverts` workflows with curved sine-arc edge chunkers. |
| properties `npt`, `k`, `dim`, `datadim`, `r`, `d`, `d2`, `n`, `wts`, `data`, `adj` | ✅ 🧪 🎯 | `@chunkgraph/chunkgraph.m` fields | Exposes merged edge chunker fields; core fields are fixture-tested. |
| `sourceinfo` | ✅ 🧪 🎯 | MATLAB source-info structs | Used by dense operator tests and checked against geometry fixture fields. |
| `build_v2emat` | ✅ 🧪 🎯 | `@chunkgraph/build_v2emat.m` | Vertex-to-edge incidence construction parity-tested. |
| `slicegraph` | ✅ 🧪 🎯 | `@chunkgraph/slicegraph.m` | Python tested and MATLAB fixture parity-tested. |
| `edgeids` | ✅ 🧪 🎯 | `@chunkgraph/edgeids.m` | Python tested and MATLAB fixture parity-tested. |
| `translate`, `transform`, `rotate`, `reflect` | ✅ 🧪 🎯 | `@chunkgraph/plus.m`, `mtimes.m`, `rotate.m`, `reflect.m` | Translation/transform graph workflows have MATLAB fixture parity. |
| `onesmat`, `normonesmat` | ✅ 🧪 🎯 | `@chunkgraph/onesmat.m`, `normonesmat.m` | Dense helper tests plus MATLAB fixture parity. |
| `chunkgraph` | ✅ 🧪 🎯 | `@chunkgraph/chunkgraph.m`, `chunkgraphinit.m` | Python constructor wrapper with square-graph fixture parity and devtools legacy/new connectivity constructor parity. |

#### `domain.py`

- ✅ 🧪 🎯 [src/chunkie/domain.py](src/chunkie/domain.py) maps top-level MATLAB geometry/domain helpers.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| private helpers | 🧩 ✅ | Internal Python helpers | Edge decoding, polygon tests, hyperoctree neighbor construction. |
| `checkcurveparam` | ✅ 🧪 🎯 | `checkcurveparam.m` | Validates callback output dimensions and input-size compatibility. |
| `ellipse` | ✅ 🧪 🎯 | `ellipse.m` | Ellipse position, first derivative, and second derivative helper. |
| `starfish` | ✅ 🧪 🎯 | `starfish.m` | Top-level export matching the existing `chnk.curves.starfish` implementation. |
| `nonflatinterface` | ✅ 🧪 🎯 | `nonflatinterface.m` | Perturbed interface graph with analytic first and second derivatives. |
| `redblue` | ✅ 🧪 🎯 | `redblue.m` | MATLAB-style red-white-blue colormap. |
| `HypOctNode`, `HypOctTree`, `hypoct_uni` | ✅ 🧪 🎯 | `hypoct_uni.m` | Zero-based Python hyperoctree dataclasses and uniform tree builder. |
| `pointinregion` | ✅ 🧪 🎯 | `pointinregion.m` | Counts region loops containing a point using chunkgraph edge geometry. |
| `regioninside` | ✅ 🧪 🎯 | `regioninside.m` | Tests nested chunkgraph regions via representative edge endpoints. |
| `mergeregions` | ✅ 🧪 🎯 | `mergeregions.m` | Merges nested/disjoint chunkgraph region lists. |


#### `chnk/arcparam.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `ArcParamData` | ✅ 🧪 🎯 | `+chnk/+arcparam/init.m` output struct | Python dataclass; MATLAB fixture checks lengths, coefficients, diagnostics, and selected-panel metadata. |
| `init` | ✅ 🧪 🎯 | `+chnk/+arcparam/init.m` | Full and selected-panel initialization parity-tested. |
| `eval` | ✅ 🧪 🎯 | `+chnk/+arcparam/eval.m` | Original-node and sample arclength evaluation parity-tested. |

#### `chnk/curves.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `_pack` | 🧩 ✅ | Internal Python helper | No direct MATLAB file. |
| `fpara` | ✅ 🧪 🎯 | `+chnk/+curves/fpara.m` | MATLAB fixture covers positions and first/second derivatives. |
| `bymode` | ✅ 🧪 🎯 | `+chnk/+curves/bymode.m` | MATLAB fixture covers modes, center, and anisotropic scaling. |
| `linefunc` | ✅ 🧪 🎯 | `+chnk/+curves/linefunc.m` | Tested and MATLAB fixture parity-tested. |
| `fsine` | ✅ 🧪 🎯 | `+chnk/+curves/fsine.m` | Tested and MATLAB fixture parity-tested. |

#### `chnk/flam.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `kernbyindex`, `kernbyindexr` | ✅ 🧪 🎯 | `+chnk/+flam/kernbyindex.m`, `kernbyindexr.m` | Python uses 0-based row/column DOF indices, applies source weights, lets sparse special-quadrature entries overwrite smooth blocks, and accepts explicit chunker sequences by merging them for square and rectangular callbacks; MATLAB fixture parity covers square/rectangular Laplace entries and sparse overwrite precedence. |
| `proxy_square_pts`, `proxy_circ_pts`, `proxy_rect_pts`, `nproxy_square` | ✅ 🧪 🎯 | `+chnk/+flam/proxy_square_pts.m`, `proxy_circ_pts.m`, `proxy_rect_pts.m`, `nproxy_square.m` | Proxy geometry, normals, the square inside predicate, and deterministic Laplace adaptive proxy-order selection are Python-tested and MATLAB-fixture tested. |
| `proxyfun`, `proxyfunr` | ✅ 🧪 🎯 | `+chnk/+flam/proxyfun.m`, `proxyfunr.m` | 0-based callback helpers for PyFLAM compression; Python tests cover neighbor filtering, callback shapes, and integrated default/level-dependent rectangular proxy target evaluation, while MATLAB fixture parity covers Laplace proxy matrices and filtered neighbor indices. |

#### `chnk/geometry.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `_ptinfo_field` | 🧩 ✅ | Internal Python helper | No direct MATLAB file. |
| `perp` | ✅ 🧪 🎯 | `+chnk/perp.m` | Tested and MATLAB fixture parity-tested. |
| `normal2d` | ✅ 🧪 🎯 | `+chnk/normal2d.m` | Tested and MATLAB fixture parity-tested. |
| `curvature2d` | ✅ 🧪 🎯 | `+chnk/curvature2d.m` | Tested and MATLAB fixture parity-tested. |
| `flagself` | ✅ 🧪 🎯 | `+chnk/flagself.m` | Tested and MATLAB fixture parity-tested with one-based MATLAB pair conversion. |
| `chunk_nearparam` | ✅ 🧪 🎯 | `+chnk/chunk_nearparam.m` | Tested on a line segment and MATLAB fixture parity-tested on a curved panel. |
| `flagnear` | ✅ 🧪 🎯 | `@chunker/flagnear.m` / `@chunkgraph/flagnear.m` behavior | Delegates to chunker implementation; chunker wrapper path now has MATLAB fixture parity. |
| `flagnear_rectangle` | ✅ 🧪 🎯 | `@chunker/flagnear_rectangle.m` | Delegates to chunker implementation; MATLAB fixture covers direct rectangle flags. |
| `flagnear_rectangle_grid` | ✅ 🧪 🎯 | `@chunker/flagnear_rectangle_grid.m` | Delegates to chunker implementation; MATLAB fixture covers grid flattening order. |


### II KERNEL AND OPERATORS
#### `kernel.py`

- ✅ 🧪 🎯 [src/chunkie/kernel.py](src/chunkie/kernel.py) maps MATLAB `@kernel` composition and kernel factory behavior.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `_infer_opdims`, `_worst_sing`, `_worst_many` | 🧩 ✅ | Internal Python helpers | No direct MATLAB file. |
| `Kernel` | ✅ 🧪 🎯 | `@kernel/kernel.m` | Callable wrapper with op dimensions and singularity metadata; direct object metadata and values are fixture-tested. |
| `Kernel.__call__` | ✅ 🧪 🎯 | `@kernel/kernel.m` | Direct evaluation tested through operator/kernel tests and MATLAB `@kernel` fixtures. |
| `Kernel.__add__`, `__sub__`, `__neg__` | ✅ 🧪 🎯 | `@kernel/plus.m`, `minus.m`, `uminus.m` | Arithmetic behavior compared against MATLAB object algebra; FMM algebra remains Python direct/FMM-tested. |
| `Kernel.__mul__`, `__rmul__`, `__truediv__` | ✅ 🧪 🎯 | `@kernel/times.m`, `mtimes.m`, `rdivide.m`, `mrdivide.m` | Scalar composition compared against MATLAB object algebra; FMM scaling remains Python direct/FMM-tested. |
| `Kernel.conj`, `conjugate` | ✅ 🧪 🎯 | `@kernel/conj.m` | Direct conjugation compared against MATLAB object algebra; FMM conjugation remains Python direct/FMM-tested. |
| `Kernel.zeros`, `Kernel.nans`, module `zeros`, `nans` | ✅ 🧪 🎯 | `@kernel/zeros.m`, `@kernel/nans.m` | MATLAB fixture checks metadata and direct zero/NaN block values. |
| `kernel` | ✅ 🧪 🎯 | `@kernel/kernel.m` | Dispatches strings, callables, existing kernels, Helmholtz-difference kernels, and block arrays for `interleave`; MATLAB fixture covers factory/direct values. |
| `lap2d_kernel` | ✅ 🧪 🎯 | `@kernel/lap2d.m`, `+chnk/+lap2d/kern.m`, `+chnk/+lap2d/fmm.m` concepts | String dispatch plus `fmm2dpy` single, double, target-normal/tangential derivatives, Hilbert, double-prime, combined-prime, and gradient paths tested against dense direct evaluation; MATLAB fixture checks `@kernel` metadata/eval. |
| `helm2d_kernel` | ✅ 🧪 🎯 | `@kernel/helm2d.m`, `+chnk/+helm2d/kern.m`, `+chnk/+helm2d/fmm.m` concepts | String dispatch plus `fmm2dpy` single, double, target-normal/tangential derivatives, double-prime, combined-prime, combined-gradient, transmission-representation, single-gradient, and double-gradient paths tested against dense direct evaluation or FMM wiring tests; MATLAB fixture checks `@kernel` metadata/eval for the MATLAB factory-supported selectors and direct `+chnk/+helm2d/kern` fixture values for the extended selectors. |
| `helm2ddiff_kernel` | ✅ 🧪 🎯 | `@kernel/helm2ddiff.m`, `+chnk/+helm2d/kern.m` `_diff` selectors | Direct Helmholtz-difference kernel factory for scalar, combined, transmission, and interleaved selector families; exact devtools interleave identities are MATLAB-fixture tested. |
| `helm1d_kernel` | ✅ 🧪 🎯 | `@kernel/helm1d.m`, `+chnk/+helm1d/kern.m` | MATLAB fixture checks `@kernel` metadata/eval for the supported single-layer factory. |
| `biharm2d_kernel` | ✅ 🧪 🎯 | `fmm2d/src/biharmonic/*`, `+chnk/+flex2d/bhgreen.m` concepts | Biharmonic Green-kernel factory and selectors are compared against MATLAB `bhgreen`-derived fixture data; FMM wiring remains Python direct/FMM-tested. |
| `stok2d_kernel` | ✅ 🧪 🎯 | `@kernel/stok2d.m`, `+chnk/+stok2d/kern.m`, `+chnk/+stok2d/fmm.m` concepts | String dispatch plus `fmm2dpy` velocity, pressure, gradient, traction, and combined Stokes paths tested against dense direct evaluation or FMM wiring tests; MATLAB fixture checks `@kernel` metadata/eval. |
| `elast2d_kernel` | ✅ 🧪 🎯 | `@kernel/elast2d.m`, `+chnk/+elast2d/kern.m` | Elasticity single, gradient, traction, double, alternate double, alternate gradient, and alternate traction selectors have FMM wiring through Laplace/Stokes decompositions and MATLAB fixture eval parity; Python keeps correct `sgrad` opdims where MATLAB `@kernel` metadata omits the gradient row count. |
| `interleave` | ✅ 🧪 🎯 | MATLAB block kernel composition patterns | Builds mixed block systems from kernel arrays; direct interleaved metadata/eval is MATLAB-fixture tested and FMM paths remain Python direct/FMM-tested. |
| `_lap2d_fmm`, `_helm2d_fmm`, `_biharm2d_fmm`, `_stok2d_fmm`, `_elast2d_fmm`, `_direct_fmm`, `_sum_fmm`, `_interleave_fmm`, `_interleave_indices`, `_target_count` | 🧩 ✅ 🧪 | FMM-backed MATLAB kernel conventions | Implemented Laplace/Helmholtz/Biharmonic/Stokes/Elasticity selectors call `fmm2dpy` or algebraic combinations of `fmm2dpy` outputs; dense-direct fallback callables remain available for custom or unsupported kernels. |

Scope note: FMM integration for the currently implemented 2D kernel families is
wired where the existing kernel selector surface applies. Axisymmetric,
quasiperiodic, and flexural kernel families are explicit non-goals for this
port: `axissymhelm2d`, `axissymhelm2ddiff`, `helm2dquas`, most of `flex2d`, and
their matching `@kernel` factories.

#### `operators.py`

- ✅ 🧪 🎯 [src/chunkie/operators.py](src/chunkie/operators.py) maps dense/direct operator assembly and evaluation helpers.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `chunkermat` | ⚠️ 🧪 🎯 | `chunkermat.m` | Default `acceleration="dense"` native/special matrix path parity-tested, including a custom data-bearing Hilbert/cotangent PV kernel; `acceleration="fmm"` returns `ChunkerFMMMatrix` with MATLAB forced-FMM matvec parity; `acceleration="flam"` returns `ChunkerFLAMMatrix` backed by PyFLAM with sparse special-quadrature overwrites and MATLAB FLAM matvec/solve parity. |
| private helpers | 🧩 ✅ | Internal Python helpers | Chunker/chunker-sequence coercion, weighted density flattening, kernel evaluation, special-quadrature dispatch, and canonical `opts["acceleration"]` parsing with MATLAB-style boolean aliases intentionally ignored. |
| `PointInfo` | ✅ 🧪 🎯 | MATLAB `srcinfo`/`targinfo` structs | Python dataclass for point info; chunker-flattened fields are fixture-tested. |
| `ChunkerFMMMatrix` | ✅ 🧪 🎯 | `chunkermatapply.m`, `+chnk/chunkerkerneval_smooth.m` FMM concepts | Matrix-free `scipy.sparse.linalg.LinearOperator` returned by `chunkermat(..., {"acceleration": "fmm"})`; caches sparse special-quadrature corrections, supports vector/multiple-RHS products, and has deterministic RHS matvec parity against MATLAB forced-FMM output. |
| `ChunkerFLAMMatrix` | ⚠️ ✅ 🧪 🎯 | `chunkerflam.m`, `+chnk/+flam/*` concepts | Matrix-free `LinearOperator` returned by `chunkermat(..., {"acceleration": "flam"})`; supports vector, multiple-RHS, and adjoint products, exposes `.factor`, `.solve(rhs, trans="n")` including adjoint solves, `.logdet()`, and dense materialization helpers when backed by PyFLAM `rskelf`; Laplace `rskelf` matvec/solve are fixture-tested against MATLAB FLAM on a deterministic random RHS. Full multi-chunker block-kernel parity remains pending. |
| `pointinfo` | ✅ 🧪 🎯 | MATLAB point-info structs | Converts chunkers/dicts/arrays; chunker flattening is fixture-tested. |
| `chunkerflam` | ⚠️ ✅ 🧪 🎯 | `chunkerflam.m` | Builds PyFLAM `rskelf`/`rskel` factors using 0-based matrix callbacks and optional proxy compression for both factor types; scalar and vector-opdim explicit chunker sequences, smooth interleaved block-kernel matrix/evaluation, proxy-enabled, smooth/special l2scale, point-data, real/complex smooth diagonal-shift paths, and MATLAB FLAM `rskelf` matvec/solve parity are tested. |
| `chunkermatapply` | ✅ 🧪 🎯 | `chunkermatapply.m` | Smooth dense application is MATLAB-fixture tested; FMM/FLAM acceleration, shape-preserving single-column and multiple-RHS products, and sparse special-quadrature corrections remain Python-tested. |
| `chunkerintegral` | ✅ 🧪 🎯 | `chunkerintegral.m` | Smooth value and callable integration routes are MATLAB-fixture tested. |
| `chunkerinterior` | ✅ 🧪 🎯 | `chunkerinterior.m` | Direct point/grid classification is MATLAB-fixture tested; optional Laplace double-layer FMM and FLAM classification use direct close-boundary correction and are Python/devtools-tested. |
| `chunkerkerneval` | ✅ 🧪 🎯 | `chunkerkerneval.m` | MATLAB parity fixture checks dense target evaluation, including `forceadap` close-target replacement for Laplace Green-identity devtools targets, full-grid Gauss identity values/classification, and target-data directional-derivative direct/adaptive/FLAM parity; FLAM target evaluation is Python-tested and partially MATLAB-fixture-tested, with default/level-dependent rectangular proxies, real/complex smooth kernels, same-source special-quadrature handling, `forceadap=True` using PyFLAM smooth evaluation plus sparse adaptive near-target corrections, and data-bearing targets falling back to non-proxy compression. Adaptive target evaluation recomputes source normals from interpolated derivatives to match MATLAB `chunkerkerneval_adap`. |
| `chunkerkernevalmat` | ✅ 🧪 🎯 | `chunkerkernevalmat.m` | MATLAB parity fixture checks eval matrices, including adaptive close-target replacement through `forceadap`; FLAM eval-matrix materialization is Python-tested, with default/level-dependent rectangular proxies, real/complex smooth kernels, same-source special-quadrature handling, `forceadap=True` using PyFLAM smooth materialization plus sparse adaptive near-target corrections, and data-bearing targets falling back to non-proxy compression. |



#### `chnk/biharm2d.py`, `lap2d.py`, `helm2d.py`, `helm1d.py`, `stok2d.py`, `elast2d.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| private kernel helpers | 🧩 ✅ | Internal Python helpers | Interleaving and validation helpers. |
| `biharm2d.green` | ✅ 🧪 🎯 | `fmm2d/src/biharmonic/bhkernels2d.f`, `+chnk/+flex2d/bhgreen.m` concepts | Biharmonic Green value, gradient, Hessian, and Laplacian are compared against MATLAB `bhgreen`; gradient also checked by finite differences. |
| `biharm2d.kern` | ✅ 🧪 🎯 | Biharmonic/flex kernel concepts in MATLAB reference | Selectors for single, double, normal derivative, gradient, Hessian, and Laplacian are compared against MATLAB `bhgreen`-derived fixture data. |
| `lap2d.green` | ✅ 🧪 🎯 | `+chnk/+lap2d/green.m` | Direct formula and derivatives are MATLAB-fixture tested. |
| `helm2d.green`, `helm2d.helmdiffgreen` | ✅ 🧪 🎯 | `+chnk/+helm2d/green.m`, `helmdiffgreen.m` | Helmholtz value/gradient/Hessian and nonsingular Helmholtz-minus-Laplace Green data are MATLAB-fixture tested. |
| `helm1d.green` | ✅ 🧪 🎯 | `+chnk/+helm1d/green.m` | Value, gradient, and Hessian are MATLAB-fixture tested. |
| `helm1d.sweep` | ✅ 🧪 🎯 | `+chnk/+helm1d/sweep.m` | Direct causal sums are MATLAB-fixture tested. |
| `lap2d.kern` | ✅ 🧪 🎯 | `+chnk/+lap2d/kern.m` | Point kernels, including gradient row ordering, are parity-tested. |
| `helm2d.kern` | ✅ 🧪 🎯 | `+chnk/+helm2d/kern.m` | Point kernels, including gradient row ordering, combined-gradient, Helmholtz-difference `_diff` selectors, `c2trans`, `all`, and transmission-representation blocks, are parity-tested. |
| `helm1d.kern` | ✅ 🧪 🎯 | `+chnk/+helm1d/kern.m` | Many scalar/combined/transmission variants parity-tested. |
| `stok2d.kern` | ✅ 🧪 🎯 | `+chnk/+stok2d/kern.m` | Stokes variants parity-tested, including pressure/traction/gradient combined paths; `cgrad` parity uses MATLAB's saved `dgrad`/`sgrad` component blocks because the saved MATLAB combined `cgrad` value combines `sgrad` twice. |
| `elast2d.kern` | ✅ 🧪 🎯 | `+chnk/+elast2d/kern.m` | Elasticity variants parity-tested, including `sgrad`, `dalttrac`, and `daltgrad`. |

✅ External FMM acceleration is wired through `fmm2dpy` for the implemented 2D selector surface: Laplace single/double/normal/tangential/Hilbert/prime/gradient/combined paths; Helmholtz single/double/normal/tangential/prime/gradient/combined-prime/combined-gradient paths, with dense-direct fallback for Helmholtz transmission-representation selectors; biharmonic single/double/normal derivative/gradient/Hessian/Laplacian paths via Laplace moment decompositions; Stokes velocity/pressure/gradient/traction/combined paths; and elasticity single/gradient/traction/double/alternate-double workflows via Laplace/Stokes decompositions. Dense-direct fallbacks remain available for custom or unsupported kernels, optional dependency absence, and compatibility tests.


### III QUADRATURES
#### `chnk/quadnative.py`
| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `_pointinfo_for_chunks` | 🧩 ✅ | Internal Python helper | No direct MATLAB file. |
| `buildmat` | ✅ 🧪 🎯 | `+chnk/+quadnative/buildmat.m` | Dense native operator path parity-tested. |

#### `chnk/quadadap.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `buildmat` | ✅ 🧪 🎯 | `+chnk/+quadadap/buildmat.m` | MATLAB fixture checks log self blocks, adaptive Gauss neighbor blocks, and robust close non-neighbor replacement; adaptive weights use MATLAB's translation-invariant recentering by default, and other singularity types delegate to `quadggq`. |
| `adapgausswts` | ✅ 🧪 🎯 | `+chnk/adapgausswts.m` | Direct adaptive Gauss weight construction is devtools-fixture tested on the starfish Helmholtz double-layer neighbor block, including recursion metadata and agreement with the GGQ reference matrix block. |

#### `chnk/quadggq.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| other private helpers | 🧩 ✅ | Internal Python helpers | Interpolation/block slicing/dtype helpers. |
| `_load_npz_table`, `_load_near_table`, `_load_cell_table` | 🧩 ✅ 🧪 | MATLAB generated quadrature table files | Internal readers for packaged NumPy GGQ data under `src/chunkie/data/quadggq`. |
| `AuxQuad` | ✅ 🧪 🎯 | MATLAB aux quadrature structs/tables | Python holder fields are checked through MATLAB `setup` fixture outputs. |
| `logavail` | ✅ 🧪 🎯 | `+chnk/+quadggq/logavail.m` | Matches MATLAB near-rule availability orders. |
| `hqsuppavail` | ✅ 🧪 🎯 | `+chnk/+quadggq/gethqsuppquad.m`, `hqsupp_*`, `hsupp_*` tables | Fixture records MATLAB table orders for PV/HS support rules. |
| `gethqsuppquad` | ✅ 🧪 🎯 | `+chnk/+quadggq/gethqsuppquad.m`, generated support table files | Reads packaged NumPy support tables, with removable-rule fallback for unavailable orders. |
| `getpvquad`, `gethsquad` | ✅ 🧪 🎯 | `hsupp_*`, `hqsupp_*` support tables | Convenience wrappers for PV and hypersingular support rules, fixture-tested through saved support tables. |
| `getremovablequad` | ✅ 🧪 🎯 | `+chnk/+quadggq/getremovablequad.m` | MATLAB fixture checks removable split rules. |
| `buildmattd` | ✅ 🧪 🎯 | `+chnk/+quadggq/buildmattd.m` | Sparse special-block matrix, correction matrix, and `ilist` skipping are MATLAB-fixture tested. |
| `setup` | ✅ 🧪 🎯 | `+chnk/+quadggq/setup.m` | Supports `log`, `removable`, `pv`, and `hs`; aux tables are MATLAB-fixture tested. |
| `getlogquad` | ✅ 🧪 🎯 | `+chnk/+quadggq/getlogquad.m`, `ggqnear*`, `ggqself_*` | Reads packaged NumPy log near/self tables, with generated fallback for unavailable orders. |
| `buildmat` | ✅ 🧪 🎯 | `+chnk/+quadggq/buildmat.m` | Log/PV/HS matrix assembly and `ilist` skipping are MATLAB-fixture tested. |
| `diagbuildmat` | ✅ 🧪 🎯 | `+chnk/+quadggq/diagbuildmat.m` | Self-block and correction-block outputs are MATLAB-fixture tested. |
| `nearbuildmat` | ✅ 🧪 🎯 | `+chnk/+quadggq/nearbuildmat.m` | Oversampled neighbor block and MATLAB-style correction subtraction are fixture-tested. |

Log/PV/HS support tables are packaged as `.npz` assets and loaded with `importlib.resources`; runtime no longer depends on a MATLAB reference checkout for GGQ tables. `quadadap` covers MATLAB-style log self, neighbor, and robust close replacement. `quadba` is an explicit non-goal for this port.

#### `chnk/rcip.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `corner_refine` | ⚠️ 🧪 🎯 | `+chnk/+rcip/chunkerfunclocal.m` and corner workflows | Convenience helper, not a direct MATLAB API match; fixture checks MATLAB chunkgraph vertex-edge topology and endpoint refinement counts. |
| `RCIPChunkGraphResult` | ✅ 🧪 🎯 | Chunkgraph RCIP workflow result struct | Fixture checks result fields, selected vertices, incident edge lists, compression matrices, and saved recursion metadata. |
| `IPinit` / `ipinit` | ✅ 🧪 🎯 | `+chnk/+rcip/IPinit.m` | Interpolation and weighted prolongation matrices are MATLAB-fixture tested, including alias coverage. |
| `Pbcinit` / `pbcinit` | ✅ 🧪 🎯 | `+chnk/+rcip/Pbcinit.m` | Block-diagonal prolongation matrix is MATLAB-fixture tested, including alias coverage. |
| `setup` | ✅ 🧪 🎯 | `+chnk/+rcip/setup.m` | Prolongation blocks plus zero-based translations of MATLAB index arrays are fixture-tested. |
| `SchurBana` / `schurbana` | ✅ 🧪 🎯 | `+chnk/+rcip/SchurBana.m` | Schur-Banachiewicz update is MATLAB-fixture tested on a deterministic well-conditioned block system. |
| `chunkgraph_rcip` / `chunkgraphrcip` / `rcipchunkgraph` | ✅ 🧪 🎯 | `chunkgrphrcip*` workflow concepts | Fixture checks selected-vertex compression over a two-edge graph against MATLAB `Rcompchunk`; all public aliases are covered. |
| `RCIPSaved` | ✅ 🧪 🎯 | `+chnk/+rcip/*` saved structs | Metadata holder populated by recursive compression and checked through MATLAB RCIP fixture fields. |
| `shiftedlegbasismats`, `chunkerfunclocal` | ✅ 🧪 🎯 | `+chnk/+rcip/shiftedlegbasismats.m`, `chunkerfunclocal.m` | Ported helpers are exercised through recursive RCIP and MATLAB fixtures. |
| `Rcompchunk` / `rcompchunk` | ✅ 🧪 🎯 | `+chnk/+rcip/Rcompchunk.m` | Recursive local compression solver implemented and tested against MATLAB fixture for a two-edge corner. |
| `rhohatInterp` / `rhohatinterp` | ✅ 🧪 🎯 | `+chnk/+rcip/rhohatInterp.m` | Saved-level backward density interpolation implemented and MATLAB-fixture tested. |


### IV LEGENDRE
#### `lege/core.py`

- ✅ 🧪 🎯 [src/chunkie/lege/core.py](src/chunkie/lege/core.py) maps MATLAB `+lege`.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `pol` | ✅ | `+lege/pol.m` | Implemented; indirectly exercised by `pols`. |
| `rts` | ✅ 🧪 🎯 | `+lege/rts.m` | Alias behavior tested and MATLAB fixture checks nodes/weights. |
| `rts_stab` | ✅ 🧪 🎯 | `+lege/rts_stab.m` | Alias behavior tested and MATLAB fixture checks stable nodes/weights. |
| `adapgauss` | ✅ 🧪 🎯 | `+lege/adapgauss.m` | Adaptive Gauss-Legendre scalar/vector integration tested; scalar polynomial output and status metadata have MATLAB fixture parity. |
| `pols` | ✅ 🧪 🎯 | `+lege/pols.m` | MATLAB parity fixture checks polynomials and derivatives. |
| `exps` | ✅ 🧪 🎯 | `+lege/exps.m` | MATLAB parity fixture checks nodes, weights, transforms. |
| `exev` | ✅ 🧪 🎯 | `+lege/exev.m` | MATLAB parity fixture checks expansion evaluation. |
| `derpol` | ✅ 🧪 🎯 | `+lege/derpol.m` | MATLAB parity fixture checks coefficients. |
| `dermat` | ✅ 🧪 🎯 | `+lege/dermat.m` | Tested and basic MATLAB fixture checked. |
| `intpol` | ✅ 🧪 🎯 | `+lege/intpol.m` | MATLAB parity fixture checks `true` and `original` options. |
| `intmat` | ✅ 🧪 🎯 | `+lege/intmat.m` | MATLAB parity fixture checks matrix values. |
| `matrin` | ✅ 🧪 🎯 | `+lege/matrin.m` | MATLAB parity fixture checks interpolation matrix. |
| `barywts` | ✅ 🧪 🎯 | `+lege/barywts.m` | MATLAB parity fixture checks weights. |
| `bernstein_ellipse` | ✅ 🧪 🎯 | `+lege/bernstein_ellipse.m` | Conformal-map ellipse nodes tested and MATLAB parity fixture checked. |
| `polsum` | ✅ 🧪 🎯 | `+lege/polsum.m` | Recurrence value, derivative, and normalization total tested; MATLAB parity fixture checked. |
| `tayl` | ✅ 🧪 🎯 | `+lege/tayl.m` | Taylor stepping tested against direct Legendre evaluation and scalar-call MATLAB parity fixture outputs. |

### V SMOOTH
#### `chnk/smoother.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `smooth` | ⚠️ 🧪 | `+chnk/+smoother/smooth.m`, `smooth_curve*.m` | Lightweight rounded-`chunkerpoly` workflow with optional zero error outputs; full MATLAB smoothing/Newton workflow is a non-goal. |
| `smooth_curve`, `smooth_curve2`, `smooth_curve3` | ⚠️ 🧪 | `+chnk/+smoother/smooth_curve*.m` | Aliases to the lightweight `smooth` baseline; full MATLAB smoothing/Newton behavior is a non-goal. |
| `UniformMesh` | ✅ 🧪 | `+chnk/+smoother/get_umesh.m` output structs | Dataclass for polygon edge mesh metadata. |
| `SmoothMesh` | ✅ 🧪 | `+chnk/+smoother/get_mesh.m` output structs | Dataclass for sampled smoother mesh metadata. |
| `get_umesh` | ✅ 🧪 | `+chnk/+smoother/get_umesh.m` | Uniform polygon edge mesh tested. |
| `get_mesh` | ✅ 🧪 | `+chnk/+smoother/get_mesh.m` | Legendre-panel mesh expansion tested. |
| `_panel_nodes` | 🧩 ✅ 🧪 | Internal Python helper | Builds panel-local Legendre nodes and weights. |

#### `chnk/spcl.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `absconvgauss` | ✅ 🧪 🎯 | `+chnk/+spcl/absconvgauss.m` | Derivatives are tested against finite differences and MATLAB devtools fixture outputs. |


## Support Tree

```text
external/
├── FLAM/  # submodule fastalgorithms/FLAM @ 73b7accda7c1a933517b008831d8404d8d3cc764
├── chunkie-matlab/  # submodule fastalgorithms/chunkie @ af34cc41c81114e693b515066e4d308067bf7e63
└── fmm2d/  # submodule flatironinstitute/fmm2d @ 550dae5b77b1e006c8ffae37fc832f8c2b536871

docs/
├── matlab-reference-setup.md
└── special-quadrature.md

scripts/
├── clean_test_data.py
├── generate_quadggq_package_data.py
└── matlab/
    ├── fixture_context.m
    ├── fixture_circle_curve.m
    ├── fixture_circle_unit.m
    ├── fixture_make_ptinfo.m
    ├── fixture_pack_chunker.m
    ├── fixture_wobbly_curve.m
    ├── generate_basic_fixtures.m
    ├── generate_chunker_circle_fixture.m
    ├── generate_chunker_ops_fixture.m
    ├── generate_geometry_core_fixture.m
    ├── generate_devtools_easy_fixture.m
    ├── generate_kernel_pointinfo_fixture.m
    ├── generate_lege_basic_fixture.m
    ├── generate_lege_extended_fixture.m
    ├── generate_operator_parity_fixture.m
    ├── generate_quadggq_fixture.m
    └── generate_rcip_fixture.m

tests/
├── golden/
│   └── README.md
├── _fixture_generation.py
├── test_arcparam.py
├── test_biharm2d.py
├── test_chunker.py
├── test_chunkerfit.py
├── test_chunkerfunc.py
├── test_chunkerpoly.py
├── test_chunkgraph.py
├── test_domain.py
├── test_devtools_parity.py
├── test_easy_parity_stress.py
├── test_elast2d.py
├── test_flam.py
├── test_geometry.py
├── test_geometry_parity.py
├── test_helm1d.py
├── test_kernel.py
├── test_kernels.py
├── test_lege.py
├── test_matlab_fixtures.py
├── test_matlab_parity.py
├── test_operators.py
├── test_quadggq.py
├── test_rcip.py
├── test_rcip_parity.py
├── test_smoother.py
├── test_sortinfo.py
├── test_spcl.py
└── test_stok2d.py

root files:
├── .gitmodules
├── .gitignore
├── .python-version
├── README.md
├── easy-test.md
├── pyproject.toml
└── uv.lock
```

Support file roles:

- 🧭 `.gitmodules`: pins external test/parity reference dependencies for CI and local parity setup; these are not Python package runtime or build dependencies.
- 🧭 `external/chunkie-matlab`: MATLAB `chunkIE` reference checkout used by fixture-generation scripts.
- 🧭 `external/FLAM`: MATLAB FLAM reference checkout for parity work; runtime Python FLAM acceleration uses the `pyflam` git dependency pinned in `pyproject.toml` / `uv.lock`.
- 🧭 `external/fmm2d`: Flatiron FMM2D checkout pinned for MATLAB-side FMM2D reference and MEX parity setup; the runtime `fmm2dpy` package is pinned separately through `pyproject.toml` / `uv.lock`.
- 🧭 `docs/matlab-reference-setup.md`: local MATLAB checkout / fixture setup notes.
- 🧭 `docs/special-quadrature.md`: special quadrature implementation notes.
- 🧭 `src/chunkie/data/quadggq/*.npz`: packaged NumPy copies of upstream MATLAB GGQ near, log self, PV support, and HS support tables used at runtime.
- 🧭 `scripts/clean_test_data.py`: removes local generated `.mat`/`.npz` parity fixture files under `tests/golden`.
- 🧭 `scripts/generate_quadggq_package_data.py`: converts upstream MATLAB `+chnk/+quadggq` table files into the package `.npz` data assets.
- 🧭 `scripts/matlab/*.m`: MATLAB fixture-generation scripts; these are the source of the `.mat` golden data used for 🎯 flags.
- 🧪 `tests/_fixture_generation.py`: ensures missing MATLAB parity fixture files are generated on demand before tests load them; generation failure is a test failure.
- 🧪 `tests/golden/*.mat`: ignored MATLAB-generated parity fixture files created on demand by tests. `devtools_easy.mat` covers the low/mid devtools track through adaptive `chunkerfunc`, `chunkerarcparam`, chunkgraph constructor/basic region parity, partial `slicegraph`, direct `adapgausswts`, `chunkermat_quadadap`, Laplace/Helmholtz Green-identity and Gauss-identity target-evaluation parity, and converted `datafieldTest.m` Hilbert/cotangent plus target-data directional-derivative slices. `geometry_core.mat` covers compact I GEOMETRY parity excluding `chunkerfit` and smoother workflows. `quadggq.mat` covers Section III native, GGQ, and adaptive quadrature behavior; `rcip.mat` covers Section III RCIP helpers, Schur updates, chunkgraph driver metadata, and recursive compression.
- 🧪 `tests/test_matlab_parity.py`: main exact-behavior comparison suite against golden data.
- 🧪 `tests/test_geometry_parity.py`: focused I GEOMETRY comparison suite against `geometry_core.mat`.
- 🧪 `tests/test_matlab_fixtures.py`: basic fixture comparison suite.
- 🧪 `tests/test_easy_parity_stress.py`: focused hardening coverage for weak parity-style areas tracked in `easy-test.md`.
- 🧪 `tests/test_flam.py`: focused PyFLAM integration coverage for FLAM helper callbacks, proxy geometry, sparse special-block overwrites, FLAM-backed matrix application/solve/logdet, target evaluation, eval-matrix materialization, and interior classification.
- 🧪 Other `tests/test_*.py`: Python behavioral/unit coverage.
- 🧭 `easy-test.md`: living tracker for parity tests that are too easy, fixture-gated, or intentionally ignored during the current hardening push.

## Major Unported MATLAB Areas

Scope triage for MATLAB areas with no full Python equivalent yet:

Should implement:

- No active items remain from the current triage. FMM integration, advanced RCIP workflows, and the first PyFLAM-backed operator paths now have Python implementations and focused tests.

Deferred implementation:

- ⚠️ Remaining FLAM parity beyond the first PyFLAM-backed pass: strict MATLAB devtools FLAM fixtures beyond the converted Green-identity and datafield diagnostics, full block-kernel multi-chunker workflows, and larger proxy-by-level stress coverage.
- ⚠️ Remaining `chunkerfit` modes beyond the implemented spline/open-line/circle paths.

Do not implement:

- 🚫 Trapper family: `@trapper/*`, `@trapperpref/*`, `trapperfunc.m`, `trapperkerneval.m`, `trappermat.m`.
- 🚫 Axisymmetric, quasiperiodic, and flexural kernel families: `+chnk/+axissymhelm2d/*`, `axissymhelm2ddiff`, `+chnk/+helm2dquas/*`, most of `+chnk/+flex2d/*`, and matching `@kernel` factories.
- 🚫 `+chnk/+quadba/*`.
- 🚫 Full nonlinear MATLAB smoother workflow and `+chnk/+intchunk/*`; the lightweight rounded-polygon smoother remains the supported Python path.
- 🚫 MATLAB plotting/visualization methods: `plot`, `plot3`, `scatter`, `quiver`, `plot_regions` on MATLAB classes.

## Recommended Next Flags To Upgrade

- Promote more optional devtools parity into generated fixtures where runtime cost allows; current generated coverage includes `chunkerfunc`, `chunkerarcparam`, chunkgraph constructor/basic region parity, `slicegraph`, direct `adapgausswts`, `chunkermat_quadadap`, and Laplace/Helmholtz Green-identity plus Gauss-identity target-evaluation paths.
- Add focused tests for remaining implemented but currently lightly tested methods that are outside the compact I GEOMETRY and devtools fixtures.
- Add stricter MATLAB fixtures for full devtools solve/evaluation workflows around adaptive close quadrature; `smoother.py` remains a lightweight rounded-polygon path and full MATLAB smoothing/Newton behavior is a non-goal.
- Add stricter MATLAB fixtures for FMM-heavy solve/evaluation workflows and implemented selector families, especially direct/FMM layer-potential Green identity paths.
