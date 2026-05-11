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

Verification snapshot: `uv run pytest` on 2026-05-11 with Python 3.11.9 collected 211 tests: `210 passed, 1 failed` (`tests/test_devtools_parity.py::test_smoother_devtools_output_matches_matlab_thresholds`, current optional fixture lacks `fixture.chunker.npt`). Targeted geometry parity/domain run `uv run pytest tests/test_geometry_parity.py tests/test_domain.py tests/test_geometry.py tests/test_chunkgraph.py tests/test_chunker.py tests/test_chunkerfunc.py`: `49 passed`; targeted operator run `uv run pytest tests/test_operators.py`: `10 passed`; targeted quadrature run `uv run pytest tests/test_quadggq.py`: `12 passed`; targeted kernel run `uv run pytest tests/test_kernel.py`: `15 passed`; targeted RCIP run `uv run pytest tests/test_rcip.py`: `7 passed`; targeted Legendre run `uv run pytest tests/test_lege.py`: `12 passed`.

Updated for commits after `2568a934c759aaf614c48f428678da8f6bbcb39f`:

- `fbca951` Add rounded polygon smoother workflow
- `fd768bb` Add PV HS GGQ table support
- `3c1cb43` Add interleaved kernel blocks
- `0bee9f2` Add biharmonic Green kernels

The ignored directories `.venv/`, `.pytest_cache/`, `.git/`, and third-party/reference checkouts under `external/` are not expanded as repo structure here. The relevant MATLAB reference paths are listed beside the Python nodes.

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
    │   ├── pointinfo
    │   ├── chunkermat, chunkermatapply
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
    │   │   ├── buildmat, diagbuildmat, nearbuildmat
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
- ✅ [src/chunkie/chnk/__init__.py](src/chunkie/chnk/__init__.py) mirrors MATLAB `+chnk` package exports, including the newer `biharm2d`, `quadadap`, `rcip`, and `smoother` modules.
- ✅ [src/chunkie/lege/__init__.py](src/chunkie/lege/__init__.py) mirrors MATLAB `+lege` package exports.


## Python To MATLAB Map

### I GEOMETRY
#### `chunker.py`

- ✅ 🧪 🎯 [src/chunkie/chunker.py](src/chunkie/chunker.py) is the main Python home for MATLAB `@chunker`, top-level chunker constructors, and related factory functions.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `chunkerfit` | ⚠️ 🧪 | `chunkerfit.m` | Spline/open-line/circle paths tested; remaining MATLAB fitting modes are deferred. |
| `ChunkerPref` | ✅ | `@chunkerpref/chunkerpref.m` | Python dataclass-like preference holder. |
| `ChunkerPref.from_any` | ✅ | `@chunkerpref/chunkerpref.m` | Python adapter for dict/None/preference inputs. |
| `_curve_outputs`, `_remap_adjacency` | 🧩 ✅ | Internal Python helpers | No direct MATLAB file. |
| `copy` | ✅ 🧪 | MATLAB value-copy behavior | Python explicit copy helper. |
| `resize` | ✅ 🧪 | `@chunker/chunker.m` storage behavior | Focused storage-growth test covers live storage preservation. |
| `makedatarows` | ✅ 🧪 | `@chunker/makedatarows.m` | Tested for data row allocation. |
| `cleardata` | ✅ 🧪 | `@chunker/makedatarows.m` / data field behavior | Tested for data-row reset behavior. |
| `checkadjinfo` | ✅ 🧪 | `@chunker/checkadjinfo.m` | Python tested through adjacency checks. |
| `sort` | ✅ 🧪 | `@chunker/sort.m` | Python tested on open segments. |
| `flagnear` | ✅ 🧪 | `@chunker/flagnear.m`, `+chnk/flagnear*` helpers | Python tested against brute-force distances. |
| `flagnear_rectangle` | ✅ 🧪 | `@chunker/flagnear_rectangle.m` | 2D only. |
| `flagnear_rectangle_grid` | ✅ 🧪 | `@chunker/flagnear_rectangle_grid.m` | Python tested on meshgrid order. |
| `nearest` | ✅ 🧪 | `@chunker/nearest.m` | Python tested for point/chunk selection. |
| `recompute_geometry` | ✅ 🧪 | MATLAB geometry recomputation inside constructors/transforms | Python tested after transforms/refinement. |
| `translate` | ✅ 🧪 | `@chunker/plus.m` | Python operator helper. |
| `__add__`, `__radd__` | ✅ 🧪 | `@chunker/plus.m` | Translation operator. |
| `chunkerpref` | ✅ 🧪 | `@chunkerpref/chunkerpref.m` | Python preference wrapper. |
| `chunkerfuncuni` | ✅ 🧪 | `chunkerfuncuni.m` | Uniform panel count tested. |
| `chunkerpoly` | ✅ 🧪 | `chunkerpoly.m`, `+chnk/+smoother/*` workflows | Straight-edge and rounded-corner polygon paths are implemented and tested. |
| `_rounded_chunkerpoly`, `_polygon_widths`, `_fill_line_chunk`, `_fill_quadratic_chunk` | 🧩 ✅ 🧪 | `chunkerpoly.m`, `+chnk/+smoother/*` concepts | Internal rounded polygon construction helpers. |
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
| private graph helpers | 🧩 ✅ | Internal Python helpers | Include edge normalization, subchunking, simple cycles, polygon tests. |
| `procverts` | ✅ 🧪 | `@chunkgraph/procverts.m` | Vertex incidence structure. |
| `findregions` | ✅ 🧪 | `@chunkgraph/findregions.m` | Region/cycle discovery baseline. |
| `refine` | ✅ 🧪 | `@chunkgraph/refine.m` | Delegates to edge chunker refinement. |
| `copy` | ✅ 🧪 | MATLAB value-copy behavior | Python helper. |
| `flagnear*` | ✅ 🧪 | `@chunkgraph/flagnear*.m` | Delegates to merged chunker helpers. |
| operator overloads | ✅ 🧪 | `@chunkgraph/plus.m`, `mtimes.m` | Scalar/matrix/translation helpers. |
| `tochunkgraph` | ✅ 🧪 | `@chunker/tochunkgraph.m` | Preserves closed/open components. |
| `chunkgraphinregion` | ✅ 🧪 | `chunkgraphinregion.m` | Point-in-region baseline. |
| `merged` | ✅ 🧪 🎯 | `@chunkgraph/*` merged geometry behavior | MATLAB fixture checks merged field access through `r`, `d`, `d2`, `n`, `wts`, and `adj`. |
| `min`, `max` | ✅ 🧪 🎯 | `@chunkgraph/min.m`, `@chunkgraph/max.m` | MATLAB fixture covers nodewise extrema. |
| `SourceInfo` | ✅ 🧪 🎯 | MATLAB source-info structs | Python dataclass used by dense operators; fixture checks flattened source fields. |
| `ChunkGraph` | ✅ 🧪 🎯 | `@chunkgraph/chunkgraph.m` | Core graph container. |
| `ChunkGraph.__init__` | ✅ 🧪 🎯 | `@chunkgraph/chunkgraph.m` | Edge/vertex construction parity-tested on a square graph. |
| properties `npt`, `k`, `dim`, `datadim`, `r`, `d`, `d2`, `n`, `wts`, `data`, `adj` | ✅ 🧪 🎯 | `@chunkgraph/chunkgraph.m` fields | Exposes merged edge chunker fields; core fields are fixture-tested. |
| `sourceinfo` | ✅ 🧪 🎯 | MATLAB source-info structs | Used by dense operator tests and checked against geometry fixture fields. |
| `build_v2emat` | ✅ 🧪 🎯 | `@chunkgraph/build_v2emat.m` | Vertex-to-edge incidence construction parity-tested. |
| `slicegraph` | ✅ 🧪 🎯 | `@chunkgraph/slicegraph.m` | Python tested and MATLAB fixture parity-tested. |
| `edgeids` | ✅ 🧪 🎯 | `@chunkgraph/edgeids.m` | Python tested and MATLAB fixture parity-tested. |
| `translate`, `transform`, `rotate`, `reflect` | ✅ 🧪 🎯 | `@chunkgraph/plus.m`, `mtimes.m`, `rotate.m`, `reflect.m` | Translation/transform graph workflows have MATLAB fixture parity. |
| `onesmat`, `normonesmat` | ✅ 🧪 🎯 | `@chunkgraph/onesmat.m`, `normonesmat.m` | Dense helper tests plus MATLAB fixture parity. |
| `chunkgraph` | ✅ 🧪 🎯 | `@chunkgraph/chunkgraph.m`, `chunkgraphinit.m` | Python constructor wrapper with square-graph fixture parity. |

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
| `ArcParamData` | ✅ 🧪 | `+chnk/+arcparam/init.m` output struct | Python dataclass. |
| `init` | ✅ 🧪 | `+chnk/+arcparam/init.m` | Tested on chunk nodes. |
| `eval` | ✅ 🧪 | `+chnk/+arcparam/eval.m` | Tested for derivative consistency on a circle. |

#### `chnk/curves.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `_pack` | 🧩 ✅ | Internal Python helper | No direct MATLAB file. |
| `fpara` | ✅ 🧪 🎯 | `+chnk/+curves/fpara.m` | MATLAB fixture covers positions and first/second derivatives. |
| `bymode` | ✅ 🧪 🎯 | `+chnk/+curves/bymode.m` | MATLAB fixture covers modes, center, and anisotropic scaling. |
| `linefunc` | ✅ 🧪 🎯 | `+chnk/+curves/linefunc.m` | Tested and MATLAB fixture parity-tested. |
| `fsine` | ✅ 🧪 🎯 | `+chnk/+curves/fsine.m` | Tested and MATLAB fixture parity-tested. |

#### `chnk/geometry.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `_ptinfo_field` | 🧩 ✅ | Internal Python helper | No direct MATLAB file. |
| `flagnear` | ✅ 🧪 | `@chunker/flagnear.m` / `@chunkgraph/flagnear.m` behavior | Delegates to chunker implementation; chunker and chunkgraph paths tested. |
| `flagnear_rectangle` | ✅ 🧪 | `@chunker/flagnear_rectangle.m` | Delegates to chunker implementation; per-chunk padding and chunkgraph delegation tested. |
| `flagnear_rectangle_grid` | ✅ 🧪 | `@chunker/flagnear_rectangle_grid.m` | Delegates to chunker implementation. |
| `perp` | ✅ 🧪 🎯 | `+chnk/perp.m` | Tested and MATLAB fixture parity-tested. |
| `normal2d` | ✅ 🧪 🎯 | `+chnk/normal2d.m` | Tested and MATLAB fixture parity-tested. |
| `curvature2d` | ✅ 🧪 🎯 | `+chnk/curvature2d.m` | Tested and MATLAB fixture parity-tested. |
| `flagself` | ✅ 🧪 🎯 | `+chnk/flagself.m` | Tested and MATLAB fixture parity-tested with one-based MATLAB pair conversion. |
| `chunk_nearparam` | ✅ 🧪 🎯 | `+chnk/chunk_nearparam.m` | Tested on a line segment and MATLAB fixture parity-tested on a curved panel. |


### II KERNEL AND OPERATORS
#### `kernel.py`

- ✅ 🧪 [src/chunkie/kernel.py](src/chunkie/kernel.py) maps MATLAB `@kernel` composition and kernel factory behavior.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `_infer_opdims`, `_worst_sing`, `_worst_many` | 🧩 ✅ | Internal Python helpers | No direct MATLAB file. |
| `Kernel` | ✅ 🧪 | `@kernel/kernel.m` | Callable wrapper with op dimensions, singularity metadata, and `fmm2dpy`/dense-direct FMM hooks. |
| `Kernel.__call__` | ✅ 🧪 | `@kernel/kernel.m` | Direct evaluation tested through operator/kernel tests. |
| `Kernel.__add__`, `__sub__`, `__neg__` | ✅ 🧪 | `@kernel/plus.m`, `minus.m`, `uminus.m` | Arithmetic behavior and FMM algebra tested. |
| `Kernel.__mul__`, `__rmul__`, `__truediv__` | ✅ 🧪 | `@kernel/times.m`, `mtimes.m`, `rdivide.m`, `mrdivide.m` | Scalar composition and FMM scaling tested. |
| `Kernel.conj`, `conjugate` | ✅ 🧪 | `@kernel/conj.m` | Conjugation and FMM conjugation tested. |
| `Kernel.zeros`, `Kernel.nans`, module `zeros`, `nans` | ✅ 🧪 | `@kernel/zeros.m`, `@kernel/nans.m` | Tested. |
| `kernel` | ✅ 🧪 | `@kernel/kernel.m` | Dispatches strings, callables, existing kernels, and block arrays for `interleave`. |
| `lap2d_kernel` | ✅ 🧪 | `@kernel/lap2d.m`, `+chnk/+lap2d/kern.m`, `+chnk/+lap2d/fmm.m` concepts | String dispatch plus `fmm2dpy` single, double, target-normal/tangential derivatives, Hilbert, double-prime, combined-prime, and gradient paths tested against dense direct evaluation. |
| `helm2d_kernel` | ✅ 🧪 | `@kernel/helm2d.m`, `+chnk/+helm2d/kern.m`, `+chnk/+helm2d/fmm.m` concepts | String dispatch plus `fmm2dpy` single, double, target-normal/tangential derivatives, double-prime, combined-prime, single-gradient, and double-gradient paths tested against dense direct evaluation or FMM wiring tests. |
| `helm1d_kernel` | ✅ 🧪 | `@kernel/helm1d.m`, `+chnk/+helm1d/kern.m` | Tested through string dispatch. |
| `biharm2d_kernel` | ✅ 🧪 | `fmm2d/src/biharmonic/*`, `+chnk/+flex2d/bhgreen.m` concepts | Biharmonic Green-kernel factory and selectors tested; single, double, target-normal derivative, gradient, Hessian, and Laplacian selectors have FMM wiring through Laplace moment decompositions. |
| `stok2d_kernel` | ✅ 🧪 | `@kernel/stok2d.m`, `+chnk/+stok2d/kern.m`, `+chnk/+stok2d/fmm.m` concepts | String dispatch plus `fmm2dpy` velocity, pressure, gradient, traction, and combined Stokes paths tested against dense direct evaluation or FMM wiring tests. |
| `elast2d_kernel` | ✅ 🧪 | `@kernel/elast2d.m`, `+chnk/+elast2d/kern.m` | Elasticity single, gradient, traction, double, alternate double, alternate gradient, and alternate traction selectors have FMM wiring through Laplace/Stokes decompositions and are tested against dense direct evaluation. |
| `interleave` | ✅ 🧪 | MATLAB block kernel composition patterns | Builds mixed block systems from kernel arrays; direct and FMM paths tested. |
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
| `chunkermat` | ⚠️ 🧪 🎯 | `chunkermat.m` | Default `acceleration="dense"` native/special matrix path parity-tested; `acceleration="fmm"` returns `ChunkerFMMMatrix` for matrix-free FMM products with special-quadrature corrections; `acceleration="flam"` is recognized but deferred. |
| private helpers | 🧩 ✅ | Internal Python helpers | Chunker coercion, weighted density flattening, kernel evaluation, special-quadrature dispatch. |
| `PointInfo` | ✅ 🧪 | MATLAB `srcinfo`/`targinfo` structs | Python dataclass for point info. |
| `ChunkerFMMMatrix` | ✅ 🧪 | `chunkermatapply.m`, `+chnk/chunkerkerneval_smooth.m` FMM concepts | Matrix-free `scipy.sparse.linalg.LinearOperator` returned by `chunkermat(..., {"acceleration": "fmm"})`; caches sparse special-quadrature corrections and supports vector/multiple-RHS products. |
| `pointinfo` | ✅ 🧪 | MATLAB point-info structs | Converts chunkers/dicts/arrays. |
| `chunkermatapply` | ✅ 🧪 | `chunkermatapply.m` | Matrix application helper with FMM acceleration plus sparse special-quadrature corrections for singular kernels. |
| `chunkerintegral` | ✅ 🧪 | `chunkerintegral.m` | Values and callables tested. |
| `chunkerinterior` | ✅ 🧪 | `chunkerinterior.m` | Direct polygon/ray classifier plus optional Laplace double-layer FMM classification with direct close-boundary correction; FLAM interior acceleration is deferred. |
| `chunkerkerneval` | ✅ 🧪 🎯 | `chunkerkerneval.m` | MATLAB parity fixture checks dense target evaluation. |
| `chunkerkernevalmat` | ✅ 🧪 🎯 | `chunkerkernevalmat.m` | MATLAB parity fixture checks eval matrix. |



#### `chnk/biharm2d.py`, `lap2d.py`, `helm2d.py`, `helm1d.py`, `stok2d.py`, `elast2d.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| private kernel helpers | 🧩 ✅ | Internal Python helpers | Interleaving and validation helpers. |
| `biharm2d.green` | ✅ 🧪 | `fmm2d/src/biharmonic/bhkernels2d.f`, `+chnk/+flex2d/bhgreen.m` concepts | Biharmonic Green value, gradient, Hessian, and Laplacian; gradient checked by finite differences. |
| `biharm2d.kern` | ✅ 🧪 | Biharmonic/flex kernel concepts in MATLAB reference | Selectors for single, double, normal derivative, gradient, Hessian, and Laplacian are tested. |
| `lap2d.green` | ✅ 🧪 | `+chnk/+lap2d/green.m` | Direct formula tested. |
| `helm2d.green` | ✅ 🧪 | `+chnk/+helm2d/green.m` | Gradient finite-difference tested. |
| `helm1d.green` | ✅ 🧪 | `+chnk/+helm1d/green.m` | Gradient finite-difference tested. |
| `helm1d.sweep` | ✅ 🧪 | `+chnk/+helm1d/sweep.m` | Direct causal sums tested. |
| `lap2d.kern` | ✅ 🧪 🎯 | `+chnk/+lap2d/kern.m` | Point kernels, including gradient row ordering, are parity-tested. |
| `helm2d.kern` | ✅ 🧪 🎯 | `+chnk/+helm2d/kern.m` | Point kernels, including gradient row ordering, are parity-tested. |
| `helm1d.kern` | ✅ 🧪 🎯 | `+chnk/+helm1d/kern.m` | Many scalar/combined/transmission variants parity-tested. |
| `stok2d.kern` | ✅ 🧪 🎯 | `+chnk/+stok2d/kern.m` | Stokes variants parity-tested. |
| `elast2d.kern` | ✅ 🧪 🎯 | `+chnk/+elast2d/kern.m` | Elasticity variants parity-tested. |

✅ External FMM acceleration is wired through `fmm2dpy` for the implemented 2D selector surface: Laplace single/double/normal/tangential/Hilbert/prime/gradient/combined paths; Helmholtz single/double/normal/tangential/prime/gradient/combined-prime paths; biharmonic single/double/normal derivative/gradient/Hessian/Laplacian paths via Laplace moment decompositions; Stokes velocity/pressure/gradient/traction/combined paths; and elasticity single/gradient/traction/double/alternate-double workflows via Laplace/Stokes decompositions. Dense-direct fallbacks remain available for custom kernels, optional dependency absence, and compatibility tests.


### III QUADRATURES
#### `chnk/quadnative.py`
| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `_pointinfo_for_chunks` | 🧩 ✅ | Internal Python helper | No direct MATLAB file. |
| `buildmat` | ✅ 🧪 🎯 | `+chnk/+quadnative/buildmat.m` | Dense native operator path parity-tested. |

#### `chnk/quadadap.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `buildmat` | ✅ 🧪 | `+chnk/+quadadap/buildmat.m` | Uses GGQ self blocks, adaptive Gauss neighbor blocks, and optional robust close non-neighbor replacement for log kernels; other singularity types delegate to `quadggq`. |

#### `chnk/quadggq.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| other private helpers | 🧩 ✅ | Internal Python helpers | Interpolation/block slicing/dtype helpers. |
| `AuxQuad` | ✅ 🧪 | MATLAB aux quadrature structs/tables | Python dataclass-like table holder. |
| `logavail` | ✅ 🧪 | `+chnk/+quadggq/logavail.m` | Tested against MATLAB-supported table orders. |
| `hqsuppavail` | ✅ 🧪 | `+chnk/+quadggq/gethqsuppquad.m`, `hqsupp_*`, `hsupp_*` tables | MATLAB table orders for PV/HS support rules. |
| `gethqsuppquad` | ✅ 🧪 | `+chnk/+quadggq/gethqsuppquad.m`, generated support table files | Reads MATLAB support table files when present, with removable-rule fallback. |
| `getpvquad`, `gethsquad` | ✅ 🧪 | `hsupp_*`, `hqsupp_*` support tables | Convenience wrappers for PV and hypersingular support rules. |
| `getremovablequad` | ✅ 🧪 | `+chnk/+quadggq/getremovablequad.m` | Tested through setup/build paths. |
| `buildmattd` | ✅ 🧪 | `+chnk/+quadggq/buildmattd.m` | Sparse special-block matrix for self/neighbor interactions with `ilist` skipping tested. |
| `_matlab_quadggq_dir`, `_parse_matlab_cell_table`, `_parse_cells`, `_parse_matlab_vector_assignment` | 🧩 ✅ 🧪 | MATLAB generated quadrature table files | Internal readers for reference `.m` table files. |
| `setup` | ✅ 🧪 🎯 | `+chnk/+quadggq/setup.m` | Supports `log`, `removable`, `pv`, and `hs`; log/PV/HS aux tables are MATLAB-fixture tested. |
| `getlogquad` | ✅ 🧪 🎯 | `+chnk/+quadggq/getlogquad.m`, `ggqnear*`, `ggqself_*` | Reads MATLAB log near/self tables when available, with generated fallback. |
| `buildmat` | ✅ 🧪 🎯 | `+chnk/+quadggq/buildmat.m` | Log/PV/HS matrix assembly and `ilist` skipping are MATLAB-fixture tested. |
| `diagbuildmat` | ✅ 🧪 🎯 | `+chnk/+quadggq/diagbuildmat.m` | Tested through MATLAB-parity `buildmat` special-quadrature paths. |
| `nearbuildmat` | ✅ 🧪 🎯 | `+chnk/+quadggq/nearbuildmat.m` | Oversampled neighbor block, MATLAB-style correction subtraction, and full matrix near blocks are tested. |

Log/PV/HS support tables are consumed when the MATLAB reference checkout is available. `quadadap` covers MATLAB-style log self, neighbor, and robust close replacement. `quadba` is an explicit non-goal for this port.

#### `chnk/rcip.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `corner_refine` | ⚠️ 🧪 | `+chnk/+rcip/chunkerfunclocal.m` and corner workflows | Convenience helper, not a direct MATLAB API match; tested on chunkgraph corner refinement. |
| `RCIPChunkGraphResult` | ✅ 🧪 | Chunkgraph RCIP workflow result struct | Holds per-vertex compression matrices, saved recursion metadata, incident edge lists, and selected local kernels. |
| `IPinit` / `ipinit` | ✅ 🧪 | `+chnk/+rcip/IPinit.m` | Interpolation and weighted preservation tested. |
| `Pbcinit` / `pbcinit` | ✅ 🧪 | `+chnk/+rcip/Pbcinit.m` | Tested through `setup` block-shape checks. |
| `setup` | ✅ 🧪 | `+chnk/+rcip/setup.m` | Zero-based Python index translation and block shapes tested. |
| `SchurBana` / `schurbana` | ✅ 🧪 | `+chnk/+rcip/SchurBana.m` | Block formula shape path tested. |
| `chunkgraph_rcip` / `chunkgraphrcip` / `rcipchunkgraph` | ✅ 🧪 | `chunkgrphrcip*` workflow concepts | Runs RCIP compression over selected chunkgraph vertices, supports ignored vertices, and subselects global edge-by-edge block kernels for local corner solves. |
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
| `rts` | ✅ 🧪 | `+lege/rts.m` | Alias behavior tested. |
| `rts_stab` | ✅ 🧪 | `+lege/rts_stab.m` | Alias behavior tested. |
| `adapgauss` | ✅ 🧪 | `+lege/adapgauss.m` | Adaptive Gauss-Legendre scalar/vector integration tested. |
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
| `absconvgauss` | ✅ 🧪 | `+chnk/+spcl/absconvgauss.m` | Derivatives tested against finite differences. |


## Support Tree

```text
docs/
├── matlab-reference-setup.md
└── special-quadrature.md

scripts/
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
│   ├── README.md
│   ├── chunker_circle.mat
│   ├── geometry_core.mat
│   ├── kernel_pointinfo.mat
│   ├── lege_basic.mat
│   ├── lege_extended.mat
│   ├── operator_parity.mat
│   ├── quadggq.mat
│   └── rcip.mat
├── test_arcparam.py
├── test_biharm2d.py
├── test_chunker.py
├── test_chunkerfit.py
├── test_chunkerfunc.py
├── test_chunkerpoly.py
├── test_chunkgraph.py
├── test_domain.py
├── test_elast2d.py
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
├── test_smoother.py
├── test_sortinfo.py
├── test_spcl.py
└── test_stok2d.py

root files:
├── .gitignore
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

Support file roles:

- 🧭 `docs/matlab-reference-setup.md`: local MATLAB checkout / fixture setup notes.
- 🧭 `docs/special-quadrature.md`: special quadrature implementation notes.
- 🧭 `scripts/matlab/*.m`: MATLAB fixture-generation scripts; these are the source of the `.mat` golden data used for 🎯 flags.
- 🧪 `tests/golden/*.mat`: small MATLAB-generated parity fixtures. Large generated parity snapshots such as `devtools_easy.mat`, `devtools_easy_python.npz`, and `chunker_ops.mat` are ignored and optional; tests skip cleanly when an optional fixture is absent. `geometry_core.mat` covers compact I GEOMETRY parity excluding `chunkerfit` and smoother workflows. `quadggq.mat` remains tracked because it covers singular quadrature behavior.
- 🧪 `tests/test_matlab_parity.py`: main exact-behavior comparison suite against golden data.
- 🧪 `tests/test_geometry_parity.py`: focused I GEOMETRY comparison suite against `geometry_core.mat`.
- 🧪 `tests/test_matlab_fixtures.py`: basic fixture comparison suite.
- 🧪 Other `tests/test_*.py`: Python behavioral/unit coverage.

## Major Unported MATLAB Areas

Scope triage for MATLAB areas with no full Python equivalent yet:

Should implement:

- No active items remain from the current triage. FMM integration and advanced RCIP workflows now have Python implementations and focused tests.

Deferred implementation:

- ⚠️ FLAM-backed acceleration: `chunkerflam.m`, `+chnk/+flam/*`, and MATLAB-style FLAM integrations are deferred until `pyflam` exists; once `pyflam` lands, pychunkie should be revisited for integration.
- ⚠️ Remaining `chunkerfit` modes beyond the implemented spline/open-line/circle paths.
- ⚠️ FLAM acceleration in `chunkermat`.
- ⚠️ FLAM interior acceleration in `chunkerinterior`.

Do not implement:

- 🚫 Trapper family: `@trapper/*`, `@trapperpref/*`, `trapperfunc.m`, `trapperkerneval.m`, `trappermat.m`.
- 🚫 Axisymmetric, quasiperiodic, and flexural kernel families: `+chnk/+axissymhelm2d/*`, `axissymhelm2ddiff`, `+chnk/+helm2dquas/*`, most of `+chnk/+flex2d/*`, and matching `@kernel` factories.
- 🚫 `+chnk/+quadba/*`.
- 🚫 Full nonlinear MATLAB smoother workflow and `+chnk/+intchunk/*`; the lightweight rounded-polygon smoother remains the supported Python path.
- 🚫 MATLAB plotting/visualization methods: `plot`, `plot3`, `scatter`, `quiver`, `plot_regions` on MATLAB classes.

## Recommended Next Flags To Upgrade

- Add golden MATLAB fixtures for `chunkerpoly` rounded paths, `chunkerfit`, `arcparam`, `spcl`, `smoother`, `rcip`, and `biharm2d` to upgrade many ✅ 🧪 nodes to 🎯.
- Add focused tests for remaining implemented but currently lightly tested methods that are outside the compact I GEOMETRY fixture.
- Add stricter MATLAB fixtures for `quadadap.py`; `smoother.py` remains a lightweight rounded-polygon path and full MATLAB smoothing/Newton behavior is a non-goal.
- Add stricter MATLAB fixtures for the newly wired FMM selector families, especially biharmonic and elasticity selectors, while keeping dense-direct fallbacks available for compatibility and tests.
