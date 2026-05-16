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

## Current Snapshot

- Verification snapshot: `uv run pytest -q --no-test-log` on 2026-05-14 with
  Python 3.11.9 collected 396 tests: `396 passed` in 234.91 seconds. Full
  MATLAB parity runs generate ignored
  `tests/golden/*.mat` files on demand and require a populated
  `external/chunkie-matlab` checkout.
- The implemented surface covers core chunkers/chunkgraphs, domain helpers, kernel factories, dense/FMM/FLAM operator paths, GGQ/adaptive quadrature, RCIP helpers, Legendre utilities, and the lightweight rounded-polygon smoother.
- No active `should implement` items remain from the current MATLAB scope triage. Deferred work is concentrated in stricter FLAM devtools parity, remaining `chunkerfit` modes, and full solve parity for the hard devtools cases listed in `devtools_coverage.md`.
- Use `docs/python-test-suite-summary.md` for the per-test index and `devtools_coverage.md` for the MATLAB devtools inventory.

The ignored directories `.venv/`, `.pytest_cache/`, and `.git/` are not expanded as repo structure here. Third-party/reference checkouts under `external/` are pinned as git submodules and summarized in the support tree; the relevant MATLAB reference paths are listed beside the Python nodes.

## Source Tree

```text
src/
└── chunkie/
    ├── __init__.py
    ├── _legacy.py
    │   └── shared deprecation helper for legacy public option dictionaries
    ├── _layout.py
    │   └── private boundary vector/tensor adapter helpers
    ├── operators/
    │   ├── __init__.py
    │   ├── core.py
    │   │   └── public operator facade
    │   ├── _assembly.py, _evaluation.py, _matrices.py
    │   │   └── public assembly/evaluation implementations and matrix wrappers
    │   ├── _common.py, _blocks.py, _special.py
    │   │   └── shared coercion, block layout, scaling, and quadrature helpers
    │   ├── _fmm.py, _flam.py, _rcip.py, _interior.py
    │   │   └── acceleration, RCIP, and direct-interior implementation helpers
    │   ├── options.py
    │   │   └── public keyword-option normalizer and typed internal accessors
    │   └── types.py
    │       └── class ChunkerRCIPMatrix, class RCIPContext
    ├── rcip/
    │   ├── __init__.py
    │   ├── algebra.py
    │   │   └── IPinit, Pbcinit, setup, SchurBana
    │   ├── core.py
    │   │   └── public RCIP facade
    │   ├── _matrix.py, _interp.py, _local.py, _graph.py
    │   │   └── private RCIP compression/interpolation/local/graph helpers
    │   └── types.py
    │       └── class RCIPSaved, class RCIPChunkGraphResult
    ├── acceleration/
    │   ├── __init__.py
    │   ├── fmm.py
    │   │   └── optional fmm2dpy loader
    │   ├── fmm_kernel.py
    │   │   └── FmmKernel adapter plus FMM selector exports
    │   ├── _fmm_common.py, _fmm_laplace.py, _fmm_helmholtz.py
    │   ├── _fmm_biharmonic.py, _fmm_stokes.py, _fmm_elasticity.py
    │   │   └── private per-family fmm2dpy selector adapters
    │   ├── _flam_common.py, _flam_index.py, _flam_proxy.py
    │   │   └── private FLAM layout, index, and proxy helpers
    │   └── flam.py
    │       └── public FLAM callback facade
    ├── geometry/
    │   ├── __init__.py
    │   ├── domain.py
    │   │   ├── checkcurveparam, ellipse, starfish, nonflatinterface
    │   │   ├── redblue, hypoct_uni
    │   │   ├── pointinregion, regioninside, mergeregions
    │   │   └── region wrappers over chunkgraph helpers
    │   ├── _hypoctree.py
    │   │   └── class HypOctNode, class HypOctTree, hypoct_uni
    │   ├── chunker.py
    │   │   └── public facade for Chunker, ChunkerPref, constructors, and merge
    │   ├── _chunker_class.py, _chunker_storage.py, _chunker_geometry.py
    │   ├── _chunker_near.py, _chunker_refine.py, _chunker_transform.py
    │   ├── _chunker_func.py, _chunker_fit.py, _chunker_poly.py
    │   ├── _chunker_points.py, _chunker_pref.py, _chunker_options.py
    │   │   └── private chunker responsibilities split out of the facade
    │   ├── chunkgraph.py
    │   │   ├── class SourceInfo, class ChunkGraph
    │   │   └── public graph helper facade
    │   ├── _chunkgraph_build.py, _chunkgraph_refine.py, _chunkgraph_regions.py
    │   │   └── private chunkgraph construction/refinement/region helpers
    │   ├── pointinfo.py
    │   │   └── class PointInfo
    │   ├── curves.py
    │   │   ├── linefunc, fpara, fsine, bymode
    │   │   └── _pack
    │   ├── _chunker_polygon.py
    │   │   └── private polygon construction helpers used by chunkerpoly
    │   └── _nearest.py
    │       └── private nearest-panel helper used by Chunker.nearest
    ├── kernels/
    │   ├── __init__.py
    │   ├── factory.py
    │   │   ├── class Kernel
    │   │   ├── kernel
    │   │   ├── lap2d_kernel, helm2d_kernel, helm1d_kernel, biharm2d_kernel
    │   │   ├── stok2d_kernel, elast2d_kernel
    │   │   ├── zeros, nans, interleave
    │   │   └── kernel algebra and dense metadata helpers
    │   ├── biharmonic.py, elasticity.py, helmholtz.py, helmholtz_1d.py
    │   ├── laplace.py
    │   └── stokes.py
    ├── misc/
    │   ├── __init__.py
    │   ├── arcparam.py
    │   │   ├── class ArcParamData
    │   │   ├── init
    │   │   └── eval
    │   ├── smoother.py
    │   │   ├── class UniformMesh, class SmoothMesh
    │   │   ├── get_umesh, get_mesh, smooth
    │   │   ├── smooth_curve, smooth_curve2, smooth_curve3
    │   │   └── _panel_nodes
    │   └── absconvgauss.py
    │       └── absconvgauss
    ├── quadrature/
    │   ├── __init__.py
    │   ├── adaptive.py
    │   │   ├── buildmat, adapgausswts
    │   │   └── warn_adaptive_failures
    │   ├── ggq.py
    │   │   ├── class AuxQuad
    │   │   ├── setup, getlogquad, logavail, hqsuppavail
    │   │   ├── gethqsuppquad, getremovablequad, getpvquad, gethsquad
    │   │   ├── buildmat, buildmattd, diagbuildmat, nearbuildmat
    │   │   └── private helpers
    │   ├── native.py
    │   │   ├── buildmat
    │   │   └── _pointinfo_for_chunks
    │   ├── panel.py
    │   │   ├── class SplitInfo
    │   │   ├── pquadwts, panel_pquadwts, panel_matrix, panel_matrix_auto_side
    │   │   ├── sd_special_quad, splitinfo_for_kernel
    │   │   └── private helpers
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

- ✅ [src/chunkie/__init__.py](src/chunkie/__init__.py) exports the public Python API. MATLAB has no direct single-file equivalent; it is a Python package facade over MATLAB class folders and package folders, with Python-first shape/naming guidance documented in `CONTRIBUTING.md`.
- 🧩 ✅ 🧪 [src/chunkie/_layout.py](src/chunkie/_layout.py) centralizes flat boundary-vector, point-matrix, chunk-tensor, weighted-density, and component-interleaved kernel-matrix adapter conversions used at solver/backend, sparse, FMM, FLAM, and fixture boundaries.
- ✅ 🧪 [src/chunkie/acceleration/__init__.py](src/chunkie/acceleration/__init__.py) exposes FMM optional-backend loading, `FmmKernel`, FMM selector adapters, and FLAM callback/proxy helper modules.
- ✅ 🧪 [src/chunkie/geometry/domain.py](src/chunkie/geometry/domain.py) implements top-level MATLAB geometry/domain wrappers exported from the Python package facade and the geometry package; [src/chunkie/geometry/_hypoctree.py](src/chunkie/geometry/_hypoctree.py) owns the hyperoctree data structures and builder.
- ✅ 🧪 [src/chunkie/geometry/__init__.py](src/chunkie/geometry/__init__.py) exposes `Chunker`, `ChunkerPref`, `ChunkGraph`, chunker constructors, graph helpers, `PointInfo`, curve helpers, domain helpers, and the geometry submodules. Pure MATLAB-style class-constructor aliases (`chunker`, `chunkerpref`, `chunkgraph`) have been removed from the package facade; use `Chunker`, `ChunkerPref.from_any`, and `ChunkGraph`.
- ✅ 🧪 [src/chunkie/kernels/__init__.py](src/chunkie/kernels/__init__.py) exposes the `Kernel` factory/algebra layer plus concrete Laplace, Helmholtz, Stokes, biharmonic, and elasticity kernel-family modules.
- ✅ 🧪 [src/chunkie/misc/__init__.py](src/chunkie/misc/__init__.py) exposes arclength parametrization, lightweight smoother, and `absconvgauss` helper modules.
- ✅ 🧪 [src/chunkie/operators/__init__.py](src/chunkie/operators/__init__.py) exposes dense, FMM, FLAM, and RCIP-aware operator assembly/application helpers through a package facade.
- ✅ 🧪 [src/chunkie/quadrature/__init__.py](src/chunkie/quadrature/__init__.py) exposes Python-native quadrature modules for native, GGQ, adaptive, and panel-product workflows.
- ✅ 🧪 [src/chunkie/rcip/__init__.py](src/chunkie/rcip/__init__.py) exposes RCIP corner-compression workflows as their own responsibility package.
- ✅ [src/chunkie/lege/__init__.py](src/chunkie/lege/__init__.py) mirrors MATLAB `+lege` package exports.


## Python To MATLAB Map

### I GEOMETRY
#### `geometry/chunker.py`

- ✅ 🧪 🎯 [src/chunkie/geometry/chunker.py](src/chunkie/geometry/chunker.py) is the public Python facade for MATLAB `@chunker`; implementation is split across private `_chunker_*` responsibility modules for storage, analysis, nearest/rectangle checks, refinement, transforms, constructors, and merge helpers.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `chunkerfit` | ⚠️ 🧪 🎯 | `chunkerfit.m` | Python-first signature uses `points` and `options`; spline/open-line/circle paths tested; devtools parity compares closed/open fitted fields including `r`, `d`, MATLAB-compatible zero `d2`, normals, weights, adjacency, chunk lengths, and area against MATLAB. Remaining MATLAB fitting modes are deferred. |
| `ChunkerPref` | ✅ 🧪 🎯 | `@chunkerpref/chunkerpref.m` | Python dataclass-like preference holder; MATLAB fixture checks explicit preference fields. |
| `ChunkerPref.from_any` | ✅ 🧪 🎯 | `@chunkerpref/chunkerpref.m` | Python adapter for dict/None/preference inputs; covered by explicit MATLAB preference-field fixture checks. |
| `_curve_outputs`, `_remap_adjacency` | 🧩 ✅ | Internal Python helpers | No direct MATLAB file. |
| `copy` | ✅ 🧪 🎯 | MATLAB value-copy behavior | Python explicit copy helper mirrors MATLAB value-object copy behavior under mutation. |
| `resize` | ✅ 🧪 🎯 | `@chunker/chunker.m` storage behavior | Storage growth and preserved live data are MATLAB-fixture tested. |
| `makedatarows` | ✅ 🧪 🎯 | `@chunker/makedatarows.m` | Data row allocation and preservation across expansion are MATLAB-fixture tested. |
| `cleardata` | ✅ 🧪 🎯 | `@chunker/chunker.m` data field behavior | Data-row reset behavior is MATLAB-fixture tested. |
| `checkadjinfo` | ✅ 🧪 🎯 | `@chunker/checkadjinfo.m` | Python tested through adjacency checks and MATLAB fixture parity. |
| `sort` | ✅ 🧪 🎯 | `@chunker/sort.m` | Python tested on open segments; strict geometry fixture compares sorted chunker fields. |
| `chunkerpoly` | ✅ 🧪 🎯 | `chunkerpoly.m`, `+chnk/+smoother/*` workflows | Straight-edge, MATLAB-style dyadic true-polygon refinement, and rounded-corner polygon paths are implemented and tested; devtools parity compares the non-smooth true-polygon chunk geometry as an unordered/orientation-aware panel set, while exact rounded smoother geometry remains the supported lightweight path. Keyword-style migration can pass edge values positionally while geometry controls are supplied as keywords. |
| `_dyadic_chunkerpoly`, `_rounded_chunkerpoly`, `_polygon_widths`, `_fill_line_chunk`, `_fill_quadratic_chunk` | 🧩 ✅ 🧪 | `chunkerpoly.m`, `+chnk/+smoother/*` concepts | Internal polygon construction helpers live in `src/chunkie/geometry/_chunker_polygon.py`; touched helpers use `chunker` and `chunk_index` names for readability without changing public constructors. |
| `flagnear` | ✅ 🧪 🎯 | `@chunker/flagnear.m`, `+chnk/flagnear*` helpers | Python-first signature uses `points` and `options`; Python tested against brute-force distances and MATLAB fixture flags. |
| `flagnear_rectangle` | ✅ 🧪 🎯 | `@chunker/flagnear_rectangle.m` | Python-first signature uses `points` and `options`; 2D rectangle flags covered by direct MATLAB fixture values. |
| `flagnear_rectangle_grid` | ✅ 🧪 🎯 | `@chunker/flagnear_rectangle_grid.m` | Python-first signature uses `options`; Python returns a regular meshgrid-shaped boolean tensor `(len(y), len(x), nch)`, and MATLAB flat fixture values are converted at the fixture boundary for parity. |
| `nearest` | ✅ 🧪 🎯 | `@chunker/nearest.m` | Python-first signature uses `points`, `chunks`, `options`, and `node_parameters`; vectorized nearest results match MATLAB scalar-reference fixture columns. |
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
| `datares` | ✅ 🧪 🎯 | `@chunker/datares.m` | Python-first signature uses keyword arguments such as `data_indices`, `coefficient_count`, and `tolerance`; legacy option dictionaries warn. Python tested for high-order data flags and MATLAB fixture parity. |
| `sortinfo` | ✅ 🧪 🎯 | `@chunker/sortinfo.m` | Strict geometry fixture covers sorted indices, adjacency, and info fields. |
| `min`, `max` | ✅ 🧪 🎯 | `@chunker/min.m`, `@chunker/max.m` | Tested against nodewise extrema and MATLAB fixture values. |
| `upsample` | ✅ 🧪 🎯 | `@chunker/upsample.m` | Python tested with density transfer and MATLAB fixture parity. |
| `split` | ✅ 🧪 🎯 | `@chunker/split.m` | Focused parameter-space and arclength split parity covers geometry, weights, adjacency, and curved-panel scalar Newton updates. |
| `refine` | ✅ 🧪 🎯 | `@chunker/refine.m` | Splits selected chunks, enforces max chunk length, arc-length level restriction, and oversampling; fixture covers selected split plus oversampling. |
| `arcresample` | ✅ 🧪 🎯 | `@chunker/arcresample.m`, `+chnk/+arcparam/*` | Python-first signature uses `move_boundaries`; legacy option dictionaries warn. Python tested for near-constant panel speed and MATLAB geometry parity, including fixed-boundary and boundary-moving modes. |
| `reverse` | ✅ 🧪 🎯 | `@chunker/reverse.m` | Python tested with polygon helpers and MATLAB fixture parity. |
| `rotate` | ✅ 🧪 🎯 | `@chunker/rotate.m` | Strict geometry fixture covers rotation about source/destination centers. |
| `reflect` | ✅ 🧪 🎯 | `@chunker/reflect.m` | Strict geometry fixture covers reflection about shifted lines. |
| `chunkerpoints` | ✅ 🧪 🎯 | `chunkerpoints.m`, `@chunker/chunkerpoints.m` | Python-first signature uses `source` and `options`; Python tested with optional derivatives and MATLAB fixture parity. |
| `merge` | ✅ 🧪 🎯 | `@chunker/merge.m` | Python tested for chunker/data row padding and MATLAB fixture parity. |
| `Chunker` | ✅ 🧪 🎯 | `@chunker/chunker.m` | Core storage and geometry object; MATLAB parity fixtures cover construction fields and many geometry transforms. |
| `Chunker.__init__` | ✅ 🧪 🎯 | `@chunker/chunker.m` | Constructor defaults and validation covered by tests. |
| `Chunker.k`, `dim`, `npt`, `datadim`, `nvert`, `vertdeg` | ✅ 🧪 🎯 | `@chunker/chunker.m` fields/properties | `k`, `dim`, geometry fields parity-tested through fixtures; Python-first aliases include `quadrature_order`, `coordinate_dim`, and `point_count`. |
| `Chunker.r`, `d`, `d2`, `n`, `wts`, `adj`, `data` | ✅ 🧪 🎯 | `@chunker/chunker.m` fields | `r`, `d`, `d2`, `n`, `wts`, `adj` checked against MATLAB fixtures; clearer aliases include `positions`, `derivatives`, `second_derivatives`, `normal_vectors`, `quadrature_weights`, and `adjacency`. |
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
| `move` | ⚠️ ✅ 🧪 🎯 | `@chunker/move.m` | MATLAB parity fixture checks positive-scale move/rotate/scale composition; Python also corrects normals for negative scale so `move(scale=-s)` agrees with scalar transformation and preserves positive-determinant area scaling. |
| `__mul__`, `__rmul__`, `__rmatmul__` | ✅ 🧪 🎯 | `@chunker/mtimes.m` | Scalar and matrix transform behavior. |
| `chunker` | ✅ 🧪 🎯 | `@chunker/chunker.m` | Python constructor wrapper. |
| `chunkerfunc` | ✅ 🧪 🎯 | `chunkerfunc.m` | Circle fixture parity plus adaptive curve/speed resolution, level restriction, max-length splitting, oversampling, and MATLAB-style open/closed endpoint warnings. |

#### `geometry/chunkgraph.py`

- ✅ 🧪 🎯 [src/chunkie/geometry/chunkgraph.py](src/chunkie/geometry/chunkgraph.py) maps the Python chunk graph object to MATLAB `@chunkgraph` plus top-level graph helpers; construction, refinement, and region traversal helpers live in private `_chunkgraph_*` modules.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `findregions` | ✅ 🧪 🎯 | `@chunkgraph/findregions.m` | MATLAB-style oriented face walking, signed unbounded regions, nested/disjoint component merging, adjacent regions, bridge/loop/multiply connected counts, and region-id behavior are fixture-tested. Python exposes positive edges as zero-based ids and reversed edges as `-(edge + 1)`. |
| `find_edge_regions` | ✅ 🧪 🎯 | `@chunkgraph/find_edge_regions.m` | Python-first signature uses `graph`; returns the one-based MATLAB-style region id on the positive/negative-normal side of each edge; strict `chunkgrphregionTest.m` fixture parity covers edge-to-region maps. |
| private graph helpers | 🧩 ✅ | Internal Python helpers | Include edge normalization, subchunking, oriented face walks, polygon tests, and nesting-aware bounded-face ordering. |
| `copy` | ✅ 🧪 🎯 | MATLAB value-copy behavior | Fixture checks graph and edge-chunker storage mutation isolation against MATLAB value-object behavior. |
| `procverts` | ✅ 🧪 🎯 | `@chunkgraph/procverts.m` | MATLAB fixture covers counterclockwise tangent ordering and incident-edge signs. |
| `refine` | ✅ 🧪 🎯 | `@chunkgraph/refine.m` | Fixture covers selected-edge refinement, per-edge split chunks, graph endpoint balancing, `last_len` endpoint-panel matching, sorted refined chunks, and refreshed graph metadata. |
| `flagnear*` | ✅ 🧪 🎯 | `@chunkgraph/flagnear*.m` | Python-first signatures use `points` and `options`; merged chunker near flags, rectangle flags, and grid flags are MATLAB-fixture tested. |
| operator overloads | ✅ 🧪 🎯 | `@chunkgraph/plus.m`, `mtimes.m` | Left/right translation, scalar scaling, and NumPy-left matrix transforms are MATLAB-fixture tested. |
| `tochunkgraph` | ✅ 🧪 🎯 | `@chunker/tochunkgraph.m` | Python-first signature uses `chunker`; closed-loop and open-line conversions are MATLAB-fixture tested. |
| `chunkgraphinregion` | ✅ 🧪 🎯 | `chunkgraphinregion.m` | Python-first signature uses `graph` and `points`; point, meshgrid, transformed adjacent-region, and nested-region ids are MATLAB-fixture tested. |
| `merged` | ✅ 🧪 🎯 | `@chunkgraph/*` merged geometry behavior | MATLAB fixture checks merged field access through `r`, `d`, `d2`, `n`, `wts`, and `adj`. |
| `min`, `max` | ✅ 🧪 🎯 | `@chunkgraph/min.m`, `@chunkgraph/max.m` | MATLAB fixture covers nodewise extrema. |
| `SourceInfo` | ✅ 🧪 🎯 | MATLAB source-info structs | Python dataclass used by dense operators; fixture checks flattened source fields. |
| `ChunkGraph` | ✅ 🧪 🎯 | `@chunkgraph/chunkgraph.m` | Core graph container. |
| `ChunkGraph.__init__` | ✅ 🧪 🎯 | `@chunkgraph/chunkgraph.m` | Edge/vertex construction parity-tested on square graphs plus devtools pentagon/basic legacy-incidence, `edgesendverts`, and MATLAB-style `NaN` closed-edge workflows with curved sine-arc edge chunkers. |
| properties `npt`, `k`, `dim`, `datadim`, `r`, `d`, `d2`, `n`, `wts`, `data`, `adj` | ✅ 🧪 🎯 | `@chunkgraph/chunkgraph.m` fields | Exposes merged edge chunker fields; core fields are fixture-tested. Python-first aliases mirror `Chunker`: `point_count`, `quadrature_order`, `coordinate_dim`, `positions`, `derivatives`, `second_derivatives`, `normal_vectors`, `quadrature_weights`, and `adjacency`. |
| `sourceinfo` | ✅ 🧪 🎯 | MATLAB source-info structs | Used by dense operator tests and checked against geometry fixture fields. |
| `build_v2emat` | ✅ 🧪 🎯 | `@chunkgraph/build_v2emat.m` | Vertex-to-edge incidence construction parity-tested. |
| `slicegraph` | ✅ 🧪 🎯 | `@chunkgraph/slicegraph.m` | Python tested and MATLAB fixture parity-tested. |
| `edgeids` | ✅ 🧪 🎯 | `@chunkgraph/edgeids.m` | Python tested and MATLAB fixture parity-tested. |
| `translate`, `transform`, `rotate`, `reflect` | ✅ 🧪 🎯 | `@chunkgraph/plus.m`, `mtimes.m`, `rotate.m`, `reflect.m` | Translation/transform graph workflows have MATLAB fixture parity. |
| `onesmat`, `normonesmat` | ✅ 🧪 🎯 | `@chunkgraph/onesmat.m`, `normonesmat.m` | Dense helper tests plus MATLAB fixture parity. |

#### `geometry/domain.py`

- ✅ 🧪 🎯 [src/chunkie/geometry/domain.py](src/chunkie/geometry/domain.py) maps top-level MATLAB geometry/domain helpers and delegates hyperoctree and chunkgraph-region internals to dedicated private modules.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| private helpers | 🧩 ✅ | Internal Python helpers | Edge decoding, polygon tests, graph balancing/refinement, hyperoctree neighbor construction; touched helpers use `graph`, `chunker`, `options`, and `chunk_count` names where they clarify helper boundaries. |
| `checkcurveparam` | ✅ 🧪 🎯 | `checkcurveparam.m` | Validates callback output dimensions and input-size compatibility. |
| `ellipse` | ✅ 🧪 🎯 | `ellipse.m` | Ellipse position, first derivative, and second derivative helper. |
| `starfish` | ⚠️ ✅ 🧪 🎯 | `starfish.m` | Top-level export matching `geometry.curves.starfish`; position and first derivative retain MATLAB fixture parity, while scaled second derivatives intentionally use the analytic linearly scaled formula instead of MATLAB's legacy double-scale expression. |
| `nonflatinterface` | ✅ 🧪 🎯 | `nonflatinterface.m` | Perturbed interface graph with analytic first and second derivatives. |
| `redblue` | ✅ 🧪 🎯 | `redblue.m` | MATLAB-style red-white-blue colormap. |
| `HypOctNode`, `HypOctTree`, `hypoct_uni` | ✅ 🧪 🎯 | `hypoct_uni.m` | Zero-based Python hyperoctree dataclasses and uniform tree builder; public signature uses `points`, `box_size`, `max_level`, and `extent`. |
| `pointinregion` | ✅ 🧪 🎯 | `pointinregion.m` | Python-first signature uses `graph`, `region`, and `point`; counts region loops containing a point using chunkgraph edge geometry. |
| `regioninside` | ✅ 🧪 🎯 | `regioninside.m` | Python-first signature uses `graph`, `containing_region`, and `candidate_region`; tests nested chunkgraph regions via representative edge endpoints. |
| `mergeregions` | ✅ 🧪 🎯 | `mergeregions.m` | Python-first signature uses `graph`, `first_region`, and `second_region`; merges nested/disjoint chunkgraph region lists. |


#### `misc/arcparam.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `ArcParamData` | ✅ 🧪 🎯 | `+chnk/+arcparam/init.m` output struct | Python dataclass; MATLAB fixture checks lengths, coefficients, diagnostics, and selected-panel metadata. |
| `init` | ✅ 🧪 🎯 | `+chnk/+arcparam/init.m` | Python-first signature uses `chunker` and `chunks`; full and selected-panel initialization parity-tested. |
| `eval` | ✅ 🧪 🎯 | `+chnk/+arcparam/eval.m` | Original-node and sample arclength evaluation parity-tested. |

#### `geometry/curves.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `_pack` | 🧩 ✅ | Internal Python helper | No direct MATLAB file. |
| `fpara` | ✅ 🧪 🎯 | `+chnk/+curves/fpara.m` | MATLAB fixture covers positions and first/second derivatives. |
| `bymode` | ✅ 🧪 🎯 | `+chnk/+curves/bymode.m` | MATLAB fixture covers modes, center, and anisotropic scaling. |
| `linefunc` | ✅ 🧪 🎯 | `+chnk/+curves/linefunc.m` | Tested and MATLAB fixture parity-tested. |
| `fsine` | ✅ 🧪 🎯 | `+chnk/+curves/fsine.m` | Tested and MATLAB fixture parity-tested. |

#### `acceleration/flam.py`

- ✅ 🧪 🎯 [src/chunkie/acceleration/flam.py](src/chunkie/acceleration/flam.py) is the public facade for FLAM callback helpers; private `_flam_index.py`, `_flam_proxy.py`, and `_flam_common.py` own index callbacks, proxy construction/callbacks, and shared layout adapters.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `kernbyindex`, `kernbyindexr` | ✅ 🧪 🎯 | `+chnk/+flam/kernbyindex.m`, `kernbyindexr.m` | Python-first signatures use `chunker`, `kernel`, and `target`; callbacks use 0-based row/column DOF indices, apply source weights, let sparse special-quadrature entries overwrite smooth blocks, and accept explicit chunker sequences by merging them for square and rectangular callbacks. MATLAB fixture parity covers square/rectangular Laplace entries and sparse overwrite precedence. |
| `proxy_square_pts`, `proxy_circ_pts`, `proxy_rect_pts`, `nproxy_square` | ✅ 🧪 🎯 | `+chnk/+flam/proxy_square_pts.m`, `proxy_circ_pts.m`, `proxy_rect_pts.m`, `nproxy_square.m` | Python-first signatures use names such as `proxy_order`, `half_lengths`, `counts`, `point_count`, `use_legendre`, `source_count`, and `rank_or_tol`; legacy option dictionaries warn. Proxy geometry, normals, the square inside predicate, and deterministic Laplace adaptive proxy-order selection are Python-tested and MATLAB-fixture tested. |
| `proxyfun`, `proxyfunr` | ✅ 🧪 🎯 | `+chnk/+flam/proxyfun.m`, `proxyfunr.m` | 0-based callback helpers for PyFLAM compression use Python-first `chunker`, `kernel`, and `target` adapter names around pyflam's positional callback arguments; Python tests cover neighbor filtering, callback shapes, and integrated default/level-dependent rectangular proxy target evaluation, while MATLAB fixture parity covers Laplace proxy matrices and filtered neighbor indices. |

#### `geometry/pointinfo.py` and private nearest helpers

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `PointInfo` | ✅ 🧪 🎯 | MATLAB `srcinfo`/`targinfo` structs | Structured point data now lives in geometry and is constructed through `from_chunker(chunker)`, `from_points(points)`, and `from_mapping(mapping)`; chunker-flattened fields are fixture-tested. |
| `PointInfo.from_any` | 🧩 ✅ 🧪 🎯 | MATLAB point-info interchange structs | Internal legacy normalizer used by operators, FLAM callbacks, tests, and MATLAB fixture interchange. |
| `_nearest.chunk_nearparam` | 🧩 ✅ 🧪 🎯 | `+chnk/chunk_nearparam.m` | Private helper used by `Chunker.nearest`; public code should call `Chunker.nearest(...)`. |
| deleted public predicate wrappers | 🚫 | `+chnk/perp.m`, `+chnk/normal2d.m`, `+chnk/flagself.m` | Removed from the public API because they were dead or duplicative in Python; near-flagging is available as `Chunker`/`ChunkGraph` methods. |


### II KERNEL AND OPERATORS
#### `kernels/factory.py`

- ✅ 🧪 🎯 [src/chunkie/kernels/factory.py](src/chunkie/kernels/factory.py) maps MATLAB `@kernel` composition and kernel factory behavior.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `_infer_opdims`, `_worst_sing`, `_worst_many` | 🧩 ✅ | Internal Python helpers | No direct MATLAB file. |
| `Kernel` | ✅ 🧪 🎯 | `@kernel/kernel.m` | Callable wrapper with op dimensions, singularity metadata, and fully spelled selector `type` metadata; direct object metadata and values are fixture-tested. |
| `Kernel.__call__` | ✅ 🧪 🎯 | `@kernel/kernel.m` | Python-first signature uses `source` and `target`; direct evaluation is tested through operator/kernel tests and MATLAB `@kernel` fixtures. |
| `Kernel.__add__`, `__sub__`, `__neg__` | ✅ 🧪 🎯 | `@kernel/plus.m`, `minus.m`, `uminus.m` | Arithmetic behavior compared against MATLAB object algebra; FMM algebra remains Python direct/FMM-tested. |
| `Kernel.__mul__`, `__rmul__`, `__truediv__` | ✅ 🧪 🎯 | `@kernel/times.m`, `mtimes.m`, `rdivide.m`, `mrdivide.m` | Scalar composition compared against MATLAB object algebra; FMM scaling remains Python direct/FMM-tested, and zero scaling returns an explicit zero kernel instead of evaluating singular or NaN operands. |
| `Kernel.conj` / `Kernel.conjugate` | ✅ 🧪 🎯 | `@kernel/conj.m` | Both names share one implementation; direct conjugation is compared against MATLAB object algebra, and FMM conjugation remains Python direct/FMM-tested. |
| `Kernel.zeros`, `Kernel.nans`, module `zeros`, `nans` | ✅ 🧪 🎯 | `@kernel/zeros.m`, `@kernel/nans.m` | MATLAB fixture checks metadata and direct zero/NaN block values. |
| `kernel` | ✅ 🧪 🎯 | `@kernel/kernel.m` | Python-first factory signature uses `spec` for the family/callable/kernel/block input; dispatches strings, callables, existing kernels, Helmholtz-difference kernels, and block arrays for `interleave`; legacy short selector strings still dispatch, while factory metadata records canonical spelled selector names. MATLAB fixture covers factory/direct values. |
| `lap2d_kernel` | ✅ 🧪 🎯 | `@kernel/lap2d.m`, `+chnk/+lap2d/kern.m`, `+chnk/+lap2d/fmm.m` concepts | String dispatch plus `fmm2dpy` single, double, target-normal/tangential derivatives, Hilbert, double-prime, combined-prime, and gradient paths tested against dense direct evaluation; MATLAB fixture checks `@kernel` metadata/eval. |
| `helm2d_kernel` | ✅ 🧪 🎯 | `@kernel/helm2d.m`, `+chnk/+helm2d/kern.m`, `+chnk/+helm2d/fmm.m` concepts | String dispatch plus `fmm2dpy` single, double, target-normal/tangential derivatives, double-prime, combined-prime, combined-gradient, transmission-representation, single-gradient, and double-gradient paths tested against dense direct evaluation or FMM wiring tests; MATLAB fixture checks `@kernel` metadata/eval for the MATLAB factory-supported selectors and direct `+chnk/+helm2d/kern` fixture values for the extended selectors. |
| `helm2ddiff_kernel` | ✅ 🧪 🎯 | `@kernel/helm2ddiff.m`, `+chnk/+helm2d/kern.m` `_diff` selectors | Direct Helmholtz-difference kernel factory for scalar, combined, transmission, and interleaved selector families; exact devtools interleave identities are MATLAB-fixture tested, and same-node single-layer smooth diagonals are Python-tested for finite removable limits. |
| `helm1d_kernel` | ✅ 🧪 🎯 | `@kernel/helm1d.m`, `+chnk/+helm1d/kern.m` | MATLAB fixture checks `@kernel` metadata/eval for the supported single-layer factory. |
| `biharm2d_kernel` | ✅ 🧪 🎯 | `fmm2d/src/biharmonic/*`, `+chnk/+flex2d/bhgreen.m` concepts | Biharmonic Green-kernel factory and selectors are compared against MATLAB `bhgreen`-derived fixture data; FMM wiring remains Python direct/FMM-tested. |
| `stok2d_kernel` | ✅ 🧪 🎯 | `@kernel/stok2d.m`, `+chnk/+stok2d/kern.m`, `+chnk/+stok2d/fmm.m` concepts | String dispatch plus `fmm2dpy` velocity, pressure, gradient, traction, and combined Stokes paths tested against dense direct evaluation or FMM wiring tests; MATLAB fixture checks `@kernel` metadata/eval. |
| `elast2d_kernel` | ✅ 🧪 🎯 | `@kernel/elast2d.m`, `+chnk/+elast2d/kern.m` | Elasticity single, gradient, traction, double, alternate double, alternate gradient, and alternate traction selectors have FMM wiring through Laplace/Stokes decompositions and MATLAB fixture eval parity; Python keeps correct `sgrad` opdims where MATLAB `@kernel` metadata omits the gradient row count. |
| `interleave` | ✅ 🧪 🎯 | MATLAB block kernel composition patterns | Builds mixed block systems from kernel arrays; direct interleaved metadata/eval is MATLAB-fixture tested, compact Helmholtz block-system dense solve/evaluation parity covers `kernel_interleaveTest.m`, and FMM paths remain Python direct/FMM-tested, including complex Helmholtz block output with real densities. |
| `_lap2d_fmm`, `_helm2d_fmm`, `_biharm2d_fmm`, `_stok2d_fmm`, `_elast2d_fmm`, `_direct_fmm`, `_sum_fmm`, `_interleave_fmm`, `_interleave_indices`, `_target_count`, `FmmKernel` | 🧩 ✅ 🧪 | FMM-backed MATLAB kernel conventions | Implemented Laplace/Helmholtz/Biharmonic/Stokes/Elasticity selectors call `fmm2dpy` or algebraic combinations of `fmm2dpy` outputs; optional `fmm2dpy` loading, selector-specific FMM formulas, dense-direct fallback marking, FMM algebra, interleaved FMM, and the `FmmKernel` adapter now live under `chunkie.acceleration`; the kernel factory imports these adapters while keeping dense kernel metadata/algebra in `kernels.factory`; FMM callback boundaries use `source_info` and `target_info`; dense-direct fallback callables remain available for custom or unsupported kernels and are marked so explicit FMM requests warn before using the O(NM) fallback; interleaved FMM widens output dtype when block outputs are complex. |

Scope note: FMM integration for the currently implemented 2D kernel families is
wired where the existing kernel selector surface applies. Axisymmetric,
quasiperiodic, and flexural kernel families are explicit non-goals for this
port: `axissymhelm2d`, `axissymhelm2ddiff`, `helm2dquas`, most of `flex2d`, and
their matching `@kernel` factories.

#### `operators/core.py`

- ✅ 🧪 🎯 [src/chunkie/operators/core.py](src/chunkie/operators/core.py) is the public facade for dense/direct operator assembly and evaluation helpers behind the [src/chunkie/operators/__init__.py](src/chunkie/operators/__init__.py) package facade. Public operator signatures are Python-first (`chunker`, `kernel`, `density`, `target`/`points`, and `options`), while backend adapter internals may still use legacy dictionary keys at explicit boundaries. Implementation is split across private `_assembly`, `_evaluation`, `_matrices`, `_common`, `_blocks`, `_special`, `_fmm`, `_flam`, `_rcip`, and `_interior` modules. [src/chunkie/operators/options.py](src/chunkie/operators/options.py) owns option normalization/accessors, and [src/chunkie/operators/types.py](src/chunkie/operators/types.py) owns small operator data wrappers.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `chunkermat` | ⚠️ 🧪 🎯 | `chunkermat.m` | Default `acceleration="dense"` native/special matrix path parity-tested, including dense l2 scaling, Laplace/Helmholtz starfish Dirichlet solve/target evaluation, Stokes combined-velocity and traction-system solve/target diagnostics, close-touching robust adaptive correction via `adaptive_correction`, Laplace `sprime` removable self limits plus Stokes single-layer traction self limits plus PV/HS singular diagnostics, chunker-sequence/chunkgraph block-kernel opdim assembly with finite special-quadrature self blocks, and a custom data-bearing Hilbert/cotangent PV kernel; nonsmooth scalar chunkgraphs now default to MATLAB-style RCIP corner compression for second-kind Laplace/Helmholtz kernels and attach `RCIPContext` metadata for postprocessed evaluation; `acceleration="fmm"` returns `ChunkerFMMMatrix` for scalar and block-kernel matrices whose blocks expose FMM evaluators, with MATLAB forced-FMM scalar matvec parity and Python dense cross-checks for block products/l2 scaling and diagonal special-correction self blocks; `acceleration="flam"` returns `ChunkerFLAMMatrix` backed by PyFLAM with sparse special-quadrature overwrites and MATLAB FLAM matvec/solve parity for scalar, level-dependent proxy, and smooth multi-chunker block kernels; Python-first option keywords include `flam_occupancy`, `rcip_subdivisions`, and `rcip_save_depth` for backend knobs that still map to legacy internal names. |
| private helpers | 🧩 ✅ | Internal Python helpers | Chunker/chunker-sequence coercion, explicit boundary layout adapters, weighted density conversion, kernel evaluation, special-quadrature dispatch, dense l2 matrix scaling, RCIP context handling, FMM/FLAM dispatch helpers, direct polygon interior tests, and normalized keyword-only option handling with temporary dict deprecation warnings. Helper boundaries are split by responsibility across private operator modules. The normalizer maps Python-first keywords such as `correction_matrix`, `flam_occupancy`, `rcip_subdivisions`, `rcip_save_depth`, and `rcip_eval_depth` onto internal adapter keys. |
| `ChunkerRCIPMatrix`, `RCIPContext` | ✅ 🧪 | MATLAB RCIP workflow metadata | Dense `ndarray` subclass and context holder returned/attached by default chunkgraph RCIP assembly; Python tests verify compressed solve metadata caching and corner-aware target reconstruction. |
| `ChunkerFMMMatrix` | ✅ 🧪 🎯 | `chunkermatapply.m`, `+chnk/chunkerkerneval_smooth.m` FMM concepts | Matrix-free `scipy.sparse.linalg.LinearOperator` returned by `chunkermat(..., acceleration="fmm")`; constructor and public attributes use `chunker`, `kernel`, and `options`; caches sparse special-quadrature corrections, supports vector/multiple-RHS products, single-chunker and block-kernel matrix application, and l2-scaled FMM products. Scalar deterministic RHS matvecs have MATLAB forced-FMM parity; block-kernel FMM products, including singular diagonal self blocks corrected exactly once, are Python-tested against dense block matrices. |
| `ChunkerFLAMMatrix` | ⚠️ ✅ 🧪 🎯 | `chunkerflam.m`, `+chnk/+flam/*` concepts | Matrix-free `LinearOperator` returned by `chunkermat(..., acceleration="flam")`; constructor and public attributes use `chunker`, `kernel`, and `options`; supports vector, multiple-RHS, adjoint products, multi-chunker block kernels, exposes `.factor`, `.solve(rhs, trans="n")` including adjoint solves, `.logdet()`, and dense materialization helpers when backed by PyFLAM `rskelf`; scalar Laplace no-proxy and level-dependent proxy `rskelf` matvec/solve plus smooth block-kernel `rskelf` matvec/solve are fixture-tested against MATLAB FLAM on deterministic random RHS vectors. |
| `chunkerflam` | ⚠️ ✅ 🧪 🎯 | `chunkerflam.m` | Builds PyFLAM `rskelf`/`rskel` factors using 0-based matrix callbacks and optional proxy compression for scalar factors; scalar and vector-opdim explicit chunker sequences, smooth multi-chunker block-kernel factors, smooth interleaved block-kernel matrix/evaluation, proxy-enabled and level-dependent-proxy scalar paths, smooth/special l2scale, point-data, real/complex smooth diagonal-shift paths, and MATLAB FLAM `rskelf` matvec/solve parity are tested. |
| `chunkermatapply` | ✅ 🧪 🎯 | `chunkermatapply.m` | Smooth dense application, scalar Laplace matrix-free system apply/solve diagnostics, vector-valued Helmholtz-transmission matrix-free apply diagnostics, and diagnostic scalar/vector chunkgraph apply paths are MATLAB-fixture tested; dense nonsmooth chunkgraph application now follows `chunkermat`'s default RCIP compression; FMM/FLAM acceleration, block-kernel FMM application, shape-preserving single-column and multiple-RHS products, and sparse special-quadrature corrections remain Python-tested. |
| `chunkerintegral` | ✅ 🧪 🎯 | `chunkerintegral.m` | Smooth value and callable integration routes are MATLAB-fixture tested. |
| `chunkerinterior` | ✅ 🧪 🎯 | `chunkerinterior.m` | Direct point/grid classification is MATLAB-fixture tested; optional Laplace double-layer FMM and FLAM classification use direct close-boundary correction and are Python/devtools-tested, including forwarding of FLAM keyword options. |
| `chunkerkerneval` | ✅ 🧪 🎯 | `chunkerkerneval.m` | MATLAB parity fixture checks dense target evaluation, including `force_adaptive=True` close-target replacement for Laplace Green-identity devtools targets, FMM smooth evaluation plus sparse close-panel correction for the `kernelclassTest.m` Green-identity path, smooth evaluation plus explicit `correction_matrix` application for Helmholtz near-target corrections, full-grid Gauss identity values/classification, target-data directional-derivative direct/adaptive/FLAM parity, and a compact FLAM level-dependent-proxy target evaluation within MATLAB ifmm approximation tolerance; default chunkgraph evaluation reuses cached `RCIPContext` metadata to zero coarse corner nodes and add `rhohatInterp` local corner panels, with close-target correction applied on both coarse and local RCIP panels and `use_panel_quadrature=True` selecting Helsing-Ojala product quadrature where side inference succeeds; same-source FMM boundary evaluation of singular kernels now uses smooth FMM plus one sparse special-quadrature correction; FLAM target evaluation is Python-tested with default/level-dependent rectangular proxies, real/complex smooth kernels, same-source special-quadrature handling, `force_adaptive=True`/`near_factor` keyword migration, PyFLAM smooth evaluation plus sparse near-target corrections, and data-bearing targets falling back to non-proxy compression. Close-target evaluation now prefers Helsing-Ojala pquad for kernels with split metadata and classifiable side, then falls back to adaptive Gauss. |
| `chunkerkernevalmat` | ✅ 🧪 🎯 | `chunkerkernevalmat.m` | MATLAB parity fixture checks eval matrices, including close-target replacement through `force_adaptive=True` and sparse correction-only matrices through `corrections=True`; FMM eval-matrix materialization is Python-tested by basis application for off-boundary targets and same-source special-quadrature matrices; FLAM eval-matrix materialization is Python-tested, with default/level-dependent rectangular proxies, real/complex smooth kernels, same-source special-quadrature handling, `force_adaptive=True` using PyFLAM smooth materialization plus sparse near-target corrections, and data-bearing targets falling back to non-proxy compression. Off-boundary target matrices and correction matrices now prefer pquad for eligible close panels, with adaptive fallback; same-source boundary matrices require explicit side/opt-in before using pquad. |



#### `kernels/biharmonic.py`, `laplace.py`, `helmholtz.py`, `helmholtz_1d.py`, `stokes.py`, `elasticity.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| private kernel helpers | 🧩 ✅ | Internal Python helpers | Interleaving and validation helpers. |
| `biharm2d.green` | ✅ 🧪 🎯 | `fmm2d/src/biharmonic/bhkernels2d.f`, `+chnk/+flex2d/bhgreen.m` concepts | Python-first signature uses `source` and `target`; biharmonic Green value, gradient, Hessian, and Laplacian are compared against MATLAB `bhgreen`; gradient also checked by finite differences, and coincident points fill only removable value/gradient limits while preserving singular second-derivative nonfinite values. |
| `biharm2d.kernel` | ✅ 🧪 🎯 | Biharmonic/flex kernel concepts in MATLAB reference | Python-first signature uses `source` and `target`; selectors for single, double, normal derivative, gradient, Hessian, and Laplacian are compared against MATLAB `bhgreen`-derived fixture data. |
| `lap2d.green` | ✅ 🧪 🎯 | `+chnk/+lap2d/green.m` | Python-first signature uses `source` and `target`; direct formula and derivatives are MATLAB-fixture tested. |
| `helm2d.green`, `helm2d.helmdiffgreen` | ✅ 🧪 🎯 | `+chnk/+helm2d/green.m`, `helmdiffgreen.m` | Python-first signatures use `source` and `target`; Helmholtz value/gradient/Hessian and nonsingular Helmholtz-minus-Laplace Green data are MATLAB-fixture tested; coincident value and gradient limits for the Helmholtz-minus-Laplace kernel are filled analytically. |
| `helm1d.green` | ✅ 🧪 🎯 | `+chnk/+helm1d/green.m` | Python-first signature uses `source` and `target`; value, gradient, and Hessian are MATLAB-fixture tested. |
| `helm1d.sweep` | ✅ 🧪 🎯 | `+chnk/+helm1d/sweep.m` | Direct causal sums are MATLAB-fixture tested. |
| `lap2d.kernel` | ✅ 🧪 🎯 | `+chnk/+lap2d/kern.m` | Python-first signature uses `source` and `target`; point kernels, including gradient row ordering, are parity-tested. |
| `helm2d.kernel` | ✅ 🧪 🎯 | `+chnk/+helm2d/kern.m` | Python-first signature uses `source` and `target`; point kernels, including gradient row ordering, combined-gradient, Helmholtz-difference `_diff` selectors, `c2trans`, `all`, and transmission-representation blocks, are parity-tested. |
| `helm1d.kernel` | ✅ 🧪 🎯 | `+chnk/+helm1d/kern.m` | Python-first signature uses `source` and `target`; many scalar/combined/transmission variants are parity-tested. |
| `stok2d.kernel` | ✅ 🧪 🎯 | `+chnk/+stok2d/kern.m` | Python-first signature uses `source` and `target`; Stokes variants are parity-tested, including pressure/traction/gradient combined paths. `cgrad` parity uses MATLAB's saved `dgrad`/`sgrad` component blocks because the saved MATLAB combined `cgrad` value combines `sgrad` twice. |
| `elast2d.kernel` | ✅ 🧪 🎯 | `+chnk/+elast2d/kern.m` | Python-first signature uses `source` and `target`; elasticity variants are parity-tested, including `sgrad`, `dalttrac`, and `daltgrad`. |

✅ External FMM acceleration is wired through `fmm2dpy` for the implemented 2D selector surface: Laplace single/double/normal/tangential/Hilbert/prime/gradient/combined paths; Helmholtz single/double/normal/tangential/prime/gradient/combined-prime/combined-gradient paths, with dense-direct fallback for Helmholtz transmission-representation selectors; biharmonic single/double/normal derivative/gradient/Hessian/Laplacian paths via Laplace moment decompositions; Stokes velocity/pressure/gradient/traction/combined paths; and elasticity single/gradient/traction/double/alternate-double workflows via Laplace/Stokes decompositions. Dense-direct fallbacks remain available for custom or unsupported kernels, optional dependency absence, and compatibility tests; explicit `acceleration="fmm"` calls now warn when they use that fallback.


### III QUADRATURES
#### `quadrature/native.py`
| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `_pointinfo_for_chunks` | 🧩 ✅ | Internal Python helper | No direct MATLAB file. |
| `buildmat` | ✅ 🧪 🎯 | `+chnk/+quadnative/buildmat.m` | Python-first signature uses `chunker`, `kernel`, `target_chunks`, `source_chunks`, and `weights`; dense native operator path parity-tested. |

#### `quadrature/adaptive.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `buildmat` | ✅ 🧪 🎯 | `+chnk/+quadadap/buildmat.m` | Python-first signature uses `chunker`, `kernel`, and keyword controls such as `singularity`, `robust`, `eps`, `use_panel_quadrature`, and `side`; legacy option dictionaries warn. MATLAB fixture checks log self blocks, adaptive Gauss neighbor blocks, and robust close non-neighbor replacement; eligible close blocks can use Helsing-Ojala pquad when requested by the operator path or keyword controls, while adaptive weights remain the low-level default/fallback and use MATLAB's translation-invariant recentering by default. Public close-panel assembly warns when adaptive recursion returns max-depth or max-interval failure statuses. Other singularity types delegate to `quadggq`. |
| `adapgausswts` | ✅ 🧪 🎯 | `+chnk/adapgausswts.m` | Python-first signature uses `chunker`, `source_chunk`, `target`, `kernel`, plus keywords such as `eps`, `max_intervals`, and `max_depth`; legacy option dictionaries warn. Direct adaptive Gauss weight construction is devtools-fixture tested on the starfish Helmholtz double-layer neighbor block, including recursion metadata and agreement with the GGQ reference matrix block; callers now preserve and report nonzero status through warning helpers instead of silently discarding it. |

#### `quadrature/panel.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `SplitInfo`, split constants, private helpers | 🧩 ✅ 🧪 | `kernel.splitinfo`, `+chnk/pquadwts.m` internals | Kernel-split metadata and side-classification helpers support product-quadrature tests and the public close-panel replacement path. |
| `sd_special_quad` | ✅ 🧪 🎯 | nested `SDspecialquad` in `+chnk/pquadwts.m` | Helsing-Ojala smooth/log/Cauchy/hypersingular/supersingular close-panel weights are Python-tested against high-order Legendre moment references and exercised through MATLAB `pquadwts` fixture parity. |
| `pquadwts`, `panel_pquadwts` | ✅ 🧪 🎯 | `+chnk/pquadwts.m` | Python-first signatures use `chunker`, `source_chunk`, `target`, `types`, `side`, `endpoint_interpolator`, `interpolator`, and `upsample`; product-quadrature weights for one target-panel set support original-node and upsampled-node forms; Python tests verify interpolation composition, and devtools fixture parity compares compact MATLAB exterior/interior log/Cauchy plus hypersingular/supersingular weights. |
| `panel_matrix`, `panel_matrix_auto_side`, `splitinfo_for_kernel` | ✅ 🧪 | `chunkerkerneval.m` pquad branch and built-in `kernel.splitinfo` | Python-first signatures use `chunker`, `source_chunk`, `target`, `split_info`, and `kernel`; panel-matrix assembly is Python-tested for Laplace and Helmholtz scalar single/double layer kernels against high-order oversampled Legendre matrices and through public `force_adaptive`/correction dispatch. Combined-kernel migration remains pending. |

#### `quadrature/ggq.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| other private helpers | 🧩 ✅ | Internal Python helpers | Interpolation/block slicing/dtype helpers. |
| `_load_npz_table`, `_load_near_table`, `_load_cell_table` | 🧩 ✅ 🧪 | MATLAB generated quadrature table files | Internal readers for packaged NumPy GGQ data under `src/chunkie/data/quadggq`. |
| `AuxQuad` | ✅ 🧪 🎯 | MATLAB aux quadrature structs/tables | Python holder fields are checked through MATLAB `setup` fixture outputs. |
| `logavail` | ✅ 🧪 🎯 | `+chnk/+quadggq/logavail.m` | Matches MATLAB near-rule availability orders. |
| `hqsuppavail` | ✅ 🧪 🎯 | `+chnk/+quadggq/gethqsuppquad.m`, `hqsupp_*`, `hsupp_*` tables | Fixture records MATLAB table orders for PV/HS support rules. |
| `gethqsuppquad` | ✅ 🧪 🎯 | `+chnk/+quadggq/gethqsuppquad.m`, generated support table files | Python-first signature uses `quadrature_order` and `singularity_code`; reads packaged NumPy support tables, with removable-rule fallback for unavailable orders. |
| `getpvquad`, `gethsquad` | ✅ 🧪 🎯 | `hsupp_*`, `hqsupp_*` support tables | Convenience wrappers for PV and hypersingular support rules, fixture-tested through saved support tables. |
| `getremovablequad` | ✅ 🧪 🎯 | `+chnk/+quadggq/getremovablequad.m` | Python-first signature uses `quadrature_order`; MATLAB fixture checks removable split rules. |
| `buildmattd` | ✅ 🧪 🎯 | `+chnk/+quadggq/buildmattd.m` | Python-first signature uses `chunker`, `kernel`, and `singularity`; sparse special-block matrix, correction matrix, and `ilist` skipping are MATLAB-fixture tested. |
| `setup` | ✅ 🧪 🎯 | `+chnk/+quadggq/setup.m` | Python-first signature uses `quadrature_order` and `singularity`; supports `log`, `removable`, `pv`, and `hs`; aux tables are MATLAB-fixture tested. |
| `getlogquad` | ✅ 🧪 🎯 | `+chnk/+quadggq/getlogquad.m`, `ggqnear*`, `ggqself_*` | Python-first signature uses `quadrature_order`; reads packaged NumPy log near/self tables, with generated fallback for unavailable orders. |
| `buildmat` | ✅ 🧪 🎯 | `+chnk/+quadggq/buildmat.m` | Python-first signature uses `chunker`, `kernel`, and `singularity`; Log/PV/HS matrix assembly and `ilist` skipping are MATLAB-fixture tested; unexpected nonfinite kernel output away from coincident source/target points is rejected instead of zero-filled. |
| `diagbuildmat` | ✅ 🧪 🎯 | `+chnk/+quadggq/diagbuildmat.m` | Self-block and correction-block outputs are MATLAB-fixture tested; only coincident singular samples are zeroed during special assembly. |
| `nearbuildmat` | ✅ 🧪 🎯 | `+chnk/+quadggq/nearbuildmat.m` | Oversampled neighbor block and MATLAB-style correction subtraction are fixture-tested; operator dispatch can request pquad replacement for eligible split kernels, with oversampled GGQ/Gauss fallback, and noncoincident nonfinite kernel values are rejected. |

Log/PV/HS support tables are packaged as `.npz` assets and loaded with `importlib.resources`; runtime no longer depends on a MATLAB reference checkout for GGQ tables. `quadrature.adaptive` covers MATLAB-style log self, neighbor, and robust close replacement, and the operator path can request `quadrature.panel` for eligible close panels while retaining adaptive Gauss fallback. `quadrature.panel` now backs eligible close-target/correction matrices and opt-in GGQ/adaptive neighbor blocks; lower-level quadrature adapter helpers touched by the Python-first pass use `chunker`, `kernel`, `source`, `target`, and `quadrature_order` names where they are not compact formula locals. `quadba` is an explicit non-goal for this port.

#### `rcip/core.py`

- ✅ 🧪 🎯 [src/chunkie/rcip/core.py](src/chunkie/rcip/core.py) is the public facade for RCIP corner-compression helpers in a dedicated responsibility package rather than generic quadrature. Private `_matrix.py`, `_interp.py`, `_local.py`, and `_graph.py` own compression construction, saved-density interpolation, local curve helpers, and chunkgraph drivers. [src/chunkie/rcip/algebra.py](src/chunkie/rcip/algebra.py) owns the prolongation and Schur-Banachiewicz setup algebra, and [src/chunkie/rcip/types.py](src/chunkie/rcip/types.py) owns the small saved-data containers.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `corner_refine` | ⚠️ 🧪 🎯 | `+chnk/+rcip/chunkerfunclocal.m` and corner workflows | Convenience helper, not a direct MATLAB API match; fixture checks MATLAB chunkgraph vertex-edge topology and endpoint refinement counts. |
| `RCIPChunkGraphResult` | ✅ 🧪 🎯 | Chunkgraph RCIP workflow result struct | Fixture checks result fields, selected vertices, incident edge lists, compression matrices, and saved recursion metadata. |
| `IPinit` | ✅ 🧪 🎯 | `+chnk/+rcip/IPinit.m` | Python-first parameter names use `nodes` and `weights`; interpolation and weighted prolongation matrices are MATLAB-fixture tested. Lowercase MATLAB-compatible aliases were removed from the Python-first API. |
| `Pbcinit` | ✅ 🧪 🎯 | `+chnk/+rcip/Pbcinit.m` | Python-first parameter names use `interpolation`, `edge_count`, and `dimension`; block-diagonal prolongation matrix is MATLAB-fixture tested. Lowercase MATLAB-compatible aliases were removed from the Python-first API. |
| `setup` | ✅ 🧪 🎯 | `+chnk/+rcip/setup.m` | Python-first parameter names use `quadrature_order`, `dimension`, `edge_count`, and `starts_at_corner`; prolongation blocks plus zero-based translations of MATLAB index arrays are fixture-tested. |
| `SchurBana` | ✅ 🧪 🎯 | `+chnk/+rcip/SchurBana.m` | Schur-Banachiewicz update is MATLAB-fixture tested on a deterministic well-conditioned block system. Lowercase MATLAB-compatible aliases were removed from the Python-first API. |
| `chunkgraph_rcip` | ✅ 🧪 🎯 | `chunkgrphrcip*` workflow concepts | Python-first signature uses `subdivisions` and `save_depth`; legacy option dictionaries warn. Fixture checks selected-vertex compression over a two-edge graph against MATLAB `Rcompchunk`; redundant MATLAB-style aliases were removed. |
| `RCIPSaved` | ✅ 🧪 🎯 | `+chnk/+rcip/*` saved structs | Metadata holder populated by recursive compression and checked through MATLAB RCIP fixture fields. |
| `shiftedlegbasismats`, `chunkerfunclocal` | ✅ 🧪 🎯 | `+chnk/+rcip/shiftedlegbasismats.m`, `chunkerfunclocal.m` | `shiftedlegbasismats` uses `quadrature_order`; ported helpers are exercised through recursive RCIP and MATLAB fixtures. |
| `Rcompchunk` | ✅ 🧪 🎯 | `+chnk/+rcip/Rcompchunk.m` | Python-first signature uses `subdivisions` and `save_depth`; legacy option dictionaries warn. Recursive local compression solver implemented and tested against MATLAB fixture for a two-edge corner; local blocks with nonfinite values are rejected instead of zero-filled. The lowercase alias was removed. |
| `rhohatInterp` | ✅ 🧪 🎯 | `+chnk/+rcip/rhohatInterp.m` | Saved-level backward density interpolation implemented and MATLAB-fixture tested. The lowercase alias was removed. |


### IV LEGENDRE
#### `lege/core.py`

- ✅ 🧪 🎯 [src/chunkie/lege/core.py](src/chunkie/lege/core.py) maps MATLAB `+lege`.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `pol` | ✅ 🧪 🎯 | `+lege/pol.m` | Degree-`n` polynomial and derivative values are compared against MATLAB fixture outputs. |
| `rts` | ✅ 🧪 🎯 | `+lege/rts.m` | Alias behavior tested and MATLAB fixture checks nodes/weights. |
| `rts_stab` | ✅ 🧪 🎯 | `+lege/rts_stab.m` | Alias behavior tested and MATLAB fixture checks stable nodes/weights. |
| `adapgauss` | ✅ 🧪 🎯 | `+lege/adapgauss.m` | Adaptive Gauss-Legendre scalar/vector integration tested; scalar polynomial output and status metadata have MATLAB fixture parity. |
| `pols` | ✅ 🧪 🎯 | `+lege/pols.m` | MATLAB parity fixture checks polynomials and derivatives. |
| `exps` | ✅ 🧪 🎯 | `+lege/exps.m` | Python-first signature uses `quadrature_order`; MATLAB parity fixture checks nodes, weights, transforms. |
| `exev` | ✅ 🧪 🎯 | `+lege/exev.m` | MATLAB parity fixture checks expansion evaluation. |
| `derpol` | ✅ 🧪 🎯 | `+lege/derpol.m` | MATLAB parity fixture checks coefficients. |
| `dermat` | ✅ 🧪 🎯 | `+lege/dermat.m` | Python-first signature uses `quadrature_order`; tested and basic MATLAB fixture checked. |
| `intpol` | ✅ 🧪 🎯 | `+lege/intpol.m` | MATLAB parity fixture checks `true` and `original` options. |
| `intmat` | ✅ 🧪 🎯 | `+lege/intmat.m` | MATLAB parity fixture checks matrix values. |
| `matrin` | ✅ 🧪 🎯 | `+lege/matrin.m` | MATLAB parity fixture checks interpolation matrix. |
| `barywts` | ✅ 🧪 🎯 | `+lege/barywts.m` | Python-first signature uses `quadrature_order`; MATLAB parity fixture checks weights. |
| `bernstein_ellipse` | ✅ 🧪 🎯 | `+lege/bernstein_ellipse.m` | Conformal-map ellipse nodes tested and MATLAB parity fixture checked. |
| `polsum` | ✅ 🧪 🎯 | `+lege/polsum.m` | Recurrence value, derivative, and normalization total tested; MATLAB parity fixture checked. |
| `tayl` | ✅ 🧪 🎯 | `+lege/tayl.m` | Taylor stepping tested against direct Legendre evaluation and scalar-call MATLAB parity fixture outputs. |

### V SMOOTH
#### `misc/smoother.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `smooth` | ⚠️ 🧪 | `+chnk/+smoother/smooth.m`, `smooth_curve*.m` | Python-first signature uses `vertices`, `quadrature_order`, `widths`, and `return_error`; legacy option dictionaries warn. Lightweight rounded-`chunkerpoly` workflow with optional zero error outputs; full MATLAB smoothing/Newton workflow is a non-goal. |
| `smooth_curve`, `smooth_curve2`, `smooth_curve3` | ⚠️ 🧪 | `+chnk/+smoother/smooth_curve*.m` | Aliases to the lightweight `smooth` baseline; full MATLAB smoothing/Newton behavior is a non-goal. |
| `UniformMesh` | ✅ 🧪 | `+chnk/+smoother/get_umesh.m` output structs | Dataclass for polygon edge mesh metadata. |
| `SmoothMesh` | ✅ 🧪 | `+chnk/+smoother/get_mesh.m` output structs | Dataclass for sampled smoother mesh metadata. |
| `get_umesh` | ✅ 🧪 | `+chnk/+smoother/get_umesh.m` | Python-first signature uses `vertices`; uniform polygon edge mesh tested. |
| `get_mesh` | ✅ 🧪 | `+chnk/+smoother/get_mesh.m` | Python-first signature uses `chunk_counts` and `quadrature_order`; Legendre-panel mesh expansion tested. |
| `_panel_nodes` | 🧩 ✅ 🧪 | Internal Python helper | Builds panel-local Legendre nodes and weights. |

#### `misc/absconvgauss.py`

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
├── bie-overview.md
├── fmm2dpy-install.md
├── matlab-reference-setup.md
└── special-quadrature.md

examples/
├── accelerated_flam_laplace.py
├── accelerated_fmm_biharmonic.py
├── accelerated_fmm_helmholtz.py
├── accelerated_fmm_laplace.py
├── accelerated_fmm_stokes.py
├── chunkgraph_annular_dirichlet.py
├── chunkgraph_region_classification.py
├── nonsmooth_laplace_exterior_dirichlet.py
├── nonsmooth_laplace_exterior_neumann.py
├── nonsmooth_laplace_interior_dirichlet.py
├── nonsmooth_laplace_interior_neumann.py
├── smooth_laplace_exterior_dirichlet.py
├── smooth_laplace_exterior_neumann.py
├── smooth_laplace_interior_dirichlet.py
└── smooth_laplace_interior_neumann.py

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
├── _numerical.py
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
├── test_kernel_algebra.py
├── test_keyword_options.py
├── test_layout_adapters.py
├── test_kernels.py
├── test_lege.py
├── test_matlab_fixtures.py
├── test_matlab_parity.py
├── test_numerical_assertions.py
├── test_operators.py
├── test_quadggq.py
├── test_rcip.py
├── test_rcip_parity.py
├── test_smoother.py
├── test_sortinfo.py
├── test_spcl.py
└── test_stok2d.py

root files:
├── AGENTS.md
├── CONTRIBUTING.md
├── .gitmodules
├── .gitignore
├── .python-version
├── README.md
├── easy-test.md
├── new_structure_map.md
├── refactor-progress.md
├── pyproject.toml
└── uv.lock
```

Support file roles:

- 🧭 `.gitmodules`: pins external test/parity reference dependencies for CI and local parity setup; these are not Python package runtime or build dependencies.
- 🧭 `AGENTS.md`: agent workflow rules, including the `CONTRIBUTING.md` reading requirement and living-doc commit gate.
- 🧭 `CONTRIBUTING.md`: living Python-first design, naming, tensor, testing, tooling, documentation, and commit guideline.
- 🧭 `external/chunkie-matlab`: MATLAB `chunkIE` reference checkout used by fixture-generation scripts.
- 🧭 `external/FLAM`: MATLAB FLAM reference checkout for parity work; runtime Python FLAM acceleration uses the `pyflam` git dependency pinned in `pyproject.toml` / `uv.lock`.
- 🧭 `external/fmm2d`: Flatiron FMM2D checkout pinned for MATLAB-side FMM2D reference and MEX parity setup; the runtime `fmm2dpy` package is pinned separately through `pyproject.toml` / `uv.lock`.
- 🧭 `docs/bie-overview.md`: high-level guide to the chunker/chunkgraph/kernel/operator API, BIE workflows, supported physics kernels, and dense/FMM/FLAM acceleration options.
- 🧭 `docs/fmm2dpy-install.md`: platform-specific notes for building the pinned `fmm2dpy` dependency.
- 🧭 `docs/matlab-reference-setup.md`: local MATLAB checkout / fixture setup notes.
- 🧭 `docs/special-quadrature.md`: special quadrature implementation notes.
- 🧭 `examples/*.py`: runnable documentation demos for smooth Laplace
  interior/exterior Dirichlet and Neumann systems, default-operator
  RCIP-compressed non-smooth polygon solves with PNG error output,
  chunkgraph multi-region workflows, and one-backend/one-physics accelerated
  Laplace/Helmholtz/Stokes/biharmonic kernel evaluation. Examples use
  Python-first variable names and keyword options, with flat adapter layout
  hidden behind helper functions.
- 🧭 `src/chunkie/data/quadggq/*.npz`: packaged NumPy copies of upstream MATLAB GGQ near, log self, PV support, and HS support tables used at runtime.
- 🧭 `scripts/clean_test_data.py`: removes local generated `.mat`/`.npz` parity fixture files under `tests/golden`.
- 🧭 `scripts/generate_quadggq_package_data.py`: converts upstream MATLAB `+chnk/+quadggq` table files into the package `.npz` data assets.
- 🧭 `scripts/matlab/*.m`: MATLAB fixture-generation scripts; these are the source of the `.mat` golden data used for 🎯 flags.
- 🧪 `tests/_fixture_generation.py`: ensures missing MATLAB parity fixture files are generated on demand before tests load them; generation failure is a test failure.
- 🧪 `tests/_numerical.py`: shared absolute-or-relative numerical assertion helper that logs maximum absolute, relative, and normalized accepted errors.
- 🧪 `tests/golden/*.mat`: ignored MATLAB-generated parity fixture files created on demand by tests. `devtools_easy.mat` covers the low/mid devtools track through adaptive `chunkerfunc`, `chunkerarcparam`, non-smooth dyadic `chunkerpoly`, chunkgraph constructor/basic region/full signed-region/opdim/last-length refinement parity, partial `slicegraph`, direct `adapgausswts`, compact product-quadrature weights, `chunkermat_l2scale`, `chunkermat_quadadap`, close-touching adaptive `chunkermat` solve/evaluation diagnostics, scalar/vector chunker and diagnostic chunkgraph `chunkermatapply` paths, interleaved Helmholtz block-system dense solve/evaluation diagnostics, singular PV/HS diagnostics, Laplace/Helmholtz/Stokes dense `chunkermat` solve/target evaluation, Laplace/Helmholtz Green-identity, explicit near-target correction, elasticity direct-kernel diagnostics, Helmholtz 1D direct Green/kernel/sweep diagnostics, and Gauss-identity target-evaluation parity, and converted `datafieldTest.m` Hilbert/cotangent plus target-data directional-derivative slices. `operator_parity.mat` covers dense/FMM/FLAM operator parity, including level-dependent proxy FLAM matvec/solve/evaluation and smooth multi-chunker block-kernel FLAM matvec/solve. `geometry_core.mat` covers compact I GEOMETRY parity excluding `chunkerfit` and smoother workflows. `quadggq.mat` covers Section III native, GGQ, and adaptive quadrature behavior; `rcip.mat` covers Section III RCIP helpers, Schur updates, chunkgraph driver metadata, and recursive compression.
- 🧪 `tests/test_matlab_parity.py`: main exact-behavior comparison suite against golden data.
- 🧪 `tests/test_geometry_parity.py`: focused I GEOMETRY comparison suite against `geometry_core.mat`.
- 🧪 `tests/test_matlab_fixtures.py`: basic fixture comparison suite.
- 🧪 `tests/test_easy_parity_stress.py`: focused hardening coverage for weak parity-style areas tracked in `easy-test.md`.
- 🧪 `tests/test_flam.py`: focused PyFLAM integration coverage for FLAM helper callbacks, proxy geometry, sparse special-block overwrites, FLAM-backed matrix application/solve/logdet, multi-chunker block kernels, target evaluation, eval-matrix materialization, and interior classification.
- 🧪 `tests/test_layout_adapters.py`: focused coverage for explicit boundary layout adapters and component-interleaved kernel tensor materialization.
- 🧪 `tests/test_numerical_assertions.py`: focused coverage for the absolute-or-relative numerical assertion helper, including near-zero references and matching NaNs.
- 🧪 Other `tests/test_*.py`: Python behavioral/unit coverage.
- 🧭 `easy-test.md`: living tracker for parity tests that are too easy, fixture-gated, or intentionally ignored during the current hardening push.
- 🧭 `new_structure_map.md`: living import/path migration map from older MATLAB-shaped Python modules to the current responsibility-based package layout.
- 🧭 `refactor-progress.md`: living staged tracker for the Python-first refactor.

## Major Unported MATLAB Areas

Scope triage for MATLAB areas with no full Python equivalent yet:

Should implement:

- No active items remain from the current triage. FMM integration, block-kernel FMM application/materialization paths, advanced RCIP workflows, and the first PyFLAM-backed operator paths now have Python implementations and focused tests.

Deferred implementation:

- ⚠️ Remaining FLAM parity beyond the first PyFLAM-backed pass: strict MATLAB devtools FLAM fixtures beyond the converted Green-identity/datafield diagnostics.
- ⚠️ Remaining `chunkerfit` modes beyond the implemented spline/open-line/circle paths.

Do not implement:

- 🚫 Trapper family: `@trapper/*`, `@trapperpref/*`, `trapperfunc.m`, `trapperkerneval.m`, `trappermat.m`.
- 🚫 Axisymmetric, quasiperiodic, and flexural kernel families: `+chnk/+axissymhelm2d/*`, `axissymhelm2ddiff`, `+chnk/+helm2dquas/*`, most of `+chnk/+flex2d/*`, and matching `@kernel` factories.
- 🚫 `+chnk/+quadba/*`.
- 🚫 Full nonlinear MATLAB smoother workflow and `+chnk/+intchunk/*`; the lightweight rounded-polygon smoother remains the supported Python path.
- 🚫 MATLAB plotting/visualization methods: `plot`, `plot3`, `scatter`, `quiver`, `plot_regions` on MATLAB classes.

## Recommended Next Flags To Upgrade

- Promote more optional devtools parity into generated fixtures where runtime cost allows; current generated coverage includes `chunkerfunc`, `chunkerarcparam`, chunkgraph constructor/basic region/full signed-region/opdim/last-length refinement parity, `slicegraph`, direct `adapgausswts`, `chunkermat_l2scale`, `chunkermat_quadadap`, close-touching adaptive `chunkermat` solve/evaluation diagnostics, scalar/vector chunker and diagnostic chunkgraph `chunkermatapply` paths, interleaved Helmholtz block-system dense solve/evaluation diagnostics, singular PV/HS diagnostics, Laplace/Helmholtz/Stokes dense `chunkermat` solve/target evaluation, and Laplace/Helmholtz Green-identity, explicit near-target correction, plus Gauss-identity target-evaluation paths.
- Add focused tests for remaining implemented but currently lightly tested methods that are outside the compact I GEOMETRY and devtools fixtures.
- Add stricter MATLAB fixtures for full devtools solve/evaluation workflows around adaptive close quadrature; `smoother.py` remains a lightweight rounded-polygon path and full MATLAB smoothing/Newton behavior is a non-goal.
- Add stricter MATLAB fixtures for FMM-heavy solve/evaluation workflows and implemented selector families, especially direct/FMM layer-potential Green identity paths.
