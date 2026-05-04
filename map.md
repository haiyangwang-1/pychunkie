# pychunkie Implementation Map

This map is a working guide for porting MATLAB `chunkIE` into Python. It maps the current Python implementation to the MATLAB reference checkout under `external/chunkie-matlab/chunkie`.

## Legend

- ✅ implemented in Python
- 🧪 covered by the Python test suite
- 🎯 compared against MATLAB-generated golden data; same behavior within the checked tolerances
- ⚠️ implemented with a known limitation, reduced scope, or intentional approximation
- 🚧 not implemented or explicitly deferred
- 🧩 private/internal helper
- 🧭 support/reference file rather than package API

Verification snapshot: `uv run pytest` on 2026-05-04 with Python 3.11.9: `126 passed, 5 xfailed`.

Known MATLAB-parity xfails:

- `chnk.lap2d.kern`: `sgrad`, `dgrad`, `cgrad` row interleaving differs from MATLAB.
- `chnk.helm2d.kern`: `sgrad`, `dgrad` row interleaving differs from MATLAB.

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
    ├── kernel.py
    │   ├── class Kernel
    │   │   ├── __call__
    │   │   ├── __add__, __sub__, __neg__, __mul__, __rmul__, __truediv__
    │   │   ├── conj, conjugate
    │   │   └── zeros, nans
    │   ├── kernel
    │   ├── lap2d_kernel, helm2d_kernel, helm1d_kernel, stok2d_kernel, elast2d_kernel
    │   ├── zeros, nans
    │   └── private helpers
    ├── operators.py
    │   ├── class PointInfo
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
    │   │   ├── setup, getlogquad, logavail, getremovablequad
    │   │   ├── buildmat, diagbuildmat, nearbuildmat
    │   │   └── private helpers
    │   ├── quadnative.py
    │   │   ├── buildmat
    │   │   └── _pointinfo_for_chunks
    │   ├── rcip.py
    │   │   ├── class RCIPSaved
    │   │   ├── IPinit, Pbcinit, setup, SchurBana
    │   │   ├── Rcompchunk, rhohatInterp, corner_refine
    │   │   └── lowercase MATLAB-style aliases
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
            └── barywts
```

## Python To MATLAB Map

### Package Exports

- ✅ [src/chunkie/__init__.py](src/chunkie/__init__.py) exports the public Python API. MATLAB has no direct single-file equivalent; it is a Python package facade over MATLAB class folders and package folders.
- ✅ [src/chunkie/chnk/__init__.py](src/chunkie/chnk/__init__.py) mirrors MATLAB `+chnk` package exports.
- ✅ [src/chunkie/lege/__init__.py](src/chunkie/lege/__init__.py) mirrors MATLAB `+lege` package exports.

### `chunker.py`

- ✅ 🧪 🎯 [src/chunkie/chunker.py](src/chunkie/chunker.py) is the main Python home for MATLAB `@chunker`, top-level chunker constructors, and related factory functions.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `ChunkerPref` | ✅ | `@chunkerpref/chunkerpref.m` | Python dataclass-like preference holder. |
| `ChunkerPref.from_any` | ✅ | `@chunkerpref/chunkerpref.m` | Python adapter for dict/None/preference inputs. |
| `Chunker` | ✅ 🧪 🎯 | `@chunker/chunker.m` | Core storage and geometry object; MATLAB parity fixtures cover construction fields and many geometry transforms. |
| `Chunker.__init__` | ✅ 🧪 🎯 | `@chunker/chunker.m` | Constructor defaults and validation covered by tests. |
| `Chunker.k`, `dim`, `npt`, `datadim`, `nvert`, `vertdeg` | ✅ 🧪 🎯 | `@chunker/chunker.m` fields/properties | `k`, `dim`, geometry fields parity-tested through fixtures. |
| `Chunker.r`, `d`, `d2`, `n`, `wts`, `adj`, `data` | ✅ 🧪 🎯 | `@chunker/chunker.m` fields | `r`, `d`, `d2`, `n`, `wts`, `adj` checked against MATLAB fixtures. |
| `copy` | ✅ 🧪 | MATLAB value-copy behavior | Python explicit copy helper. |
| `addchunk` | ✅ 🧪 🎯 | `@chunker/chunker.m` storage behavior | Used by MATLAB fixture reconstruction. |
| `resize` | ✅ | `@chunker/chunker.m` storage behavior | No direct focused test yet. |
| `makedatarows` | ✅ 🧪 | `@chunker/makedatarows.m` | Tested for data row allocation. |
| `cleardata` | ✅ | `@chunker/makedatarows.m` / data field behavior | No focused test yet. |
| `weights` | ✅ 🧪 | `@chunker/weights.m`, `@chunker/whts.m` | Returns quadrature weights. |
| `normals` | ✅ 🧪 | `@chunker/normals.m` | 2D only; raises for non-2D. |
| `tangents` | ✅ 🧪 | `@chunker/tangents.m`, `@chunker/taus.m` | Tangent vectors from derivatives. |
| `arclengthdens` | ✅ 🧪 | `@chunker/arclengthdens.m` | Tested directly and through arc parameterization. |
| `arclengthder` | ✅ 🧪 | `@chunker/arclengthder.m` | Tested on circle data. |
| `arclengthfun` | ✅ 🧪 | `@chunker/arclengthfun.m` | Tested on circle data. |
| `chunklen` | ✅ 🧪 🎯 | `@chunker/chunklen.m` | MATLAB parity fixture checks values. |
| `chunkends` | ✅ | `@chunker/chunkends.m` | Implemented; no focused test yet. |
| `area` | ✅ 🧪 🎯 | `@chunker/area.m` | MATLAB parity fixture checks values. |
| `signed_curvature` | ✅ 🧪 | `@chunker/signed_curvature.m` | Python tested; not yet golden-parity tested. |
| `exps` | ✅ 🧪 🎯 | `@chunker/exps.m`, `+lege/exps.m` | MATLAB parity via Legendre fixture and chunker fixture. |
| `diffmat` | ✅ 🧪 🎯 | `@chunker/diffmat.m` | MATLAB parity fixture checks first and second derivative matrices. |
| `intmat` | ✅ 🧪 🎯 | `@chunker/intmat.m` | MATLAB parity fixture checks chunk-order matrix. |
| `onesmat` | ✅ 🧪 🎯 | `@chunker/onesmat.m` | MATLAB parity fixture checks values. |
| `normonesmat` | ✅ 🧪 🎯 | `@chunker/normonesmat.m` | MATLAB parity fixture checks values. |
| `centroids` | ✅ 🧪 🎯 | `@chunker/centroids.m` | MATLAB parity fixture checks values. |
| `datares` | ✅ 🧪 | `@chunker/datares.m` | Python tested for high-order data flags. |
| `sortinfo` | ✅ 🧪 | `@chunker/sortinfo.m` | Python tested; no golden fixture yet. |
| `checkadjinfo` | ✅ 🧪 | `@chunker/checkadjinfo.m` | Python tested through adjacency checks. |
| `sort` | ✅ 🧪 | `@chunker/sort.m` | Python tested on open segments. |
| `flagnear` | ✅ 🧪 | `@chunker/flagnear.m`, `+chnk/flagnear*` helpers | Python tested against brute-force distances. |
| `flagnear_rectangle` | ✅ 🧪 | `@chunker/flagnear_rectangle.m` | 2D only. |
| `flagnear_rectangle_grid` | ✅ 🧪 | `@chunker/flagnear_rectangle_grid.m` | Python tested on meshgrid order. |
| `nearest` | ✅ 🧪 | `@chunker/nearest.m` | Python tested for point/chunk selection. |
| `min`, `max` | ✅ | `@chunker/min.m`, `@chunker/max.m` | Implemented; no focused test yet. |
| `recompute_geometry` | ✅ 🧪 | MATLAB geometry recomputation inside constructors/transforms | Python tested after transforms/refinement. |
| `upsample` | ✅ 🧪 | `@chunker/upsample.m` | Python tested with density transfer. |
| `split` | ✅ 🧪 | `@chunker/split.m` | Tested indirectly through `refine`; add focused parity later. |
| `refine` | ⚠️ 🧪 | `@chunker/refine.m` | Current baseline splits chunks; adaptive refinement options are deferred. |
| `arcresample` | ✅ 🧪 | `@chunker/arcresample.m`, `+chnk/+arcparam/*` | Python tested for near-constant panel speed. |
| `translate` | ✅ 🧪 | `@chunker/plus.m` | Python operator helper. |
| `transform` | ✅ 🧪 🎯 | `@chunker/mtimes.m` | MATLAB parity fixture checks matrix transform. |
| `reverse` | ✅ 🧪 | `@chunker/reverse.m` | Python tested with polygon helpers. |
| `move` | ✅ 🧪 🎯 | `@chunker/move.m` | MATLAB parity fixture checks move/rotate/scale composition. |
| `rotate` | ✅ 🧪 | `@chunker/rotate.m` | Python tested; no golden fixture yet. |
| `reflect` | ✅ 🧪 | `@chunker/reflect.m` | Python tested; no golden fixture yet. |
| `__add__`, `__radd__` | ✅ 🧪 | `@chunker/plus.m` | Translation operator. |
| `__mul__`, `__rmul__`, `__rmatmul__` | ✅ 🧪 🎯 | `@chunker/mtimes.m` | Scalar and matrix transform behavior. |
| `chunker` | ✅ 🧪 🎯 | `@chunker/chunker.m` | Python constructor wrapper. |
| `chunkerpref` | ✅ 🧪 | `@chunkerpref/chunkerpref.m` | Python preference wrapper. |
| `chunkerfunc` | ⚠️ 🧪 🎯 | `chunkerfunc.m` | Circle fixture parity; adaptive refinement options intentionally deferred. |
| `chunkerfuncuni` | ✅ 🧪 | `chunkerfuncuni.m` | Uniform panel count tested. |
| `chunkerfit` | ⚠️ 🧪 | `chunkerfit.m` | Spline/open-line/circle paths tested; unsupported methods raise. |
| `chunkerpoly` | ⚠️ 🧪 | `chunkerpoly.m` | Baseline one panel per edge; rounded corners explicitly 🚧. |
| `chunkerpoints` | ✅ 🧪 | `chunkerpoints.m`, `@chunker/chunkerpoints.m` | Python tested with optional derivatives. |
| `merge` | ✅ 🧪 | `@chunker/merge.m` | Python tested for chunker/data row padding. |
| `_curve_outputs`, `_remap_adjacency` | 🧩 ✅ | Internal Python helpers | No direct MATLAB file. |

### `chunkgraph.py`

- ✅ 🧪 [src/chunkie/chunkgraph.py](src/chunkie/chunkgraph.py) maps the Python chunk graph object to MATLAB `@chunkgraph` plus top-level graph helpers.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `SourceInfo` | ✅ 🧪 | MATLAB source-info structs | Python dataclass used by dense operators. |
| `ChunkGraph` | ✅ 🧪 | `@chunkgraph/chunkgraph.m` | Core graph container. |
| `ChunkGraph.__init__` | ✅ 🧪 | `@chunkgraph/chunkgraph.m` | Edge/vertex construction tested. |
| properties `npt`, `k`, `dim`, `datadim`, `r`, `d`, `d2`, `n`, `wts`, `data`, `adj` | ✅ 🧪 | `@chunkgraph/chunkgraph.m` fields | Exposes merged edge chunker fields. |
| `sourceinfo` | ✅ 🧪 | MATLAB source-info structs | Used by dense operator tests. |
| `merged` | ✅ | `@chunkgraph/*` merged geometry behavior | No focused test yet. |
| `build_v2emat` | ✅ 🧪 | `@chunkgraph/build_v2emat.m` | Vertex-to-edge incidence construction. |
| `procverts` | ✅ 🧪 | `@chunkgraph/procverts.m` | Vertex incidence structure. |
| `findregions` | ✅ 🧪 | `@chunkgraph/findregions.m` | Region/cycle discovery baseline. |
| `slicegraph` | ✅ 🧪 | `@chunkgraph/slicegraph.m` | Python tested. |
| `edgeids` | ✅ 🧪 | `@chunkgraph/edgeids.m` | Python tested. |
| `refine` | ✅ 🧪 | `@chunkgraph/refine.m` | Delegates to edge chunker refinement. |
| `copy` | ✅ 🧪 | MATLAB value-copy behavior | Python helper. |
| `translate`, `transform`, `rotate`, `reflect` | ✅ 🧪 | `@chunkgraph/plus.m`, `mtimes.m`, `rotate.m`, `reflect.m` | Translation/transform tested through graph workflows. |
| `min`, `max` | ✅ | `@chunkgraph/min.m`, `@chunkgraph/max.m` | Implemented; no focused test yet. |
| `onesmat`, `normonesmat` | ✅ 🧪 | `@chunkgraph/onesmat.m`, `normonesmat.m` | Dense helper tests. |
| `flagnear*` | ✅ 🧪 | `@chunkgraph/flagnear*.m` | Delegates to merged chunker helpers. |
| operator overloads | ✅ 🧪 | `@chunkgraph/plus.m`, `mtimes.m` | Scalar/matrix/translation helpers. |
| `chunkgraph` | ✅ 🧪 | `@chunkgraph/chunkgraph.m`, `chunkgraphinit.m` | Python constructor wrapper. |
| `tochunkgraph` | ✅ 🧪 | `@chunker/tochunkgraph.m` | Preserves closed/open components. |
| `chunkgraphinregion` | ✅ 🧪 | `chunkgraphinregion.m` | Point-in-region baseline. |
| private graph helpers | 🧩 ✅ | Internal Python helpers | Include edge normalization, subchunking, simple cycles, polygon tests. |

### `kernel.py`

- ✅ 🧪 [src/chunkie/kernel.py](src/chunkie/kernel.py) maps MATLAB `@kernel` composition and kernel factory behavior.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `Kernel` | ✅ 🧪 | `@kernel/kernel.m` | Callable wrapper with op dimensions and singularity metadata. |
| `Kernel.__call__` | ✅ 🧪 | `@kernel/kernel.m` | Direct evaluation tested through operator/kernel tests. |
| `Kernel.__add__`, `__sub__`, `__neg__` | ✅ 🧪 | `@kernel/plus.m`, `minus.m`, `uminus.m` | Arithmetic behavior tested. |
| `Kernel.__mul__`, `__rmul__`, `__truediv__` | ✅ 🧪 | `@kernel/times.m`, `mtimes.m`, `rdivide.m`, `mrdivide.m` | Scalar composition tested. |
| `Kernel.conj`, `conjugate` | ✅ 🧪 | `@kernel/conj.m` | Conjugation tested. |
| `Kernel.zeros`, `Kernel.nans`, module `zeros`, `nans` | ✅ 🧪 | `@kernel/zeros.m`, `@kernel/nans.m` | Tested. |
| `kernel` | ✅ 🧪 | `@kernel/kernel.m` | Dispatches strings/callables/existing kernels. |
| `lap2d_kernel` | ✅ 🧪 | `@kernel/lap2d.m`, `+chnk/+lap2d/kern.m` | Tested through string dispatch. |
| `helm2d_kernel` | ✅ 🧪 | `@kernel/helm2d.m`, `+chnk/+helm2d/kern.m` | Tested through string dispatch. |
| `helm1d_kernel` | ✅ 🧪 | `@kernel/helm1d.m`, `+chnk/+helm1d/kern.m` | Tested through string dispatch. |
| `stok2d_kernel` | ✅ 🧪 | `@kernel/stok2d.m`, `+chnk/+stok2d/kern.m` | Tested through string dispatch. |
| `elast2d_kernel` | ✅ 🧪 | `@kernel/elast2d.m`, `+chnk/+elast2d/kern.m` | Tested through string dispatch. |
| `_infer_opdims`, `_worst_sing` | 🧩 ✅ | Internal Python helpers | No direct MATLAB file. |

🚧 MATLAB kernel factories not yet represented in Python: `axissymhelm2d`, `axissymhelm2ddiff`, `helm2ddiff`, `helm2dquas`.

### `operators.py`

- ✅ 🧪 🎯 [src/chunkie/operators.py](src/chunkie/operators.py) maps dense/direct operator assembly and evaluation helpers.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `PointInfo` | ✅ 🧪 | MATLAB `srcinfo`/`targinfo` structs | Python dataclass for point info. |
| `pointinfo` | ✅ 🧪 | MATLAB point-info structs | Converts chunkers/dicts/arrays. |
| `chunkermat` | ⚠️ 🧪 🎯 | `chunkermat.m` | Dense native path parity-tested; special quadrature for log kernels exists; FMM/FLAM acceleration deferred. |
| `chunkermatapply` | ✅ 🧪 | `chunkermatapply.m` | Matrix application helper. |
| `chunkerintegral` | ✅ 🧪 | `chunkerintegral.m` | Values and callables tested. |
| `chunkerinterior` | ⚠️ 🧪 | `chunkerinterior.m` | Polygon/ray baseline; close-boundary correction deferred. |
| `chunkerkerneval` | ✅ 🧪 🎯 | `chunkerkerneval.m` | MATLAB parity fixture checks dense target evaluation. |
| `chunkerkernevalmat` | ✅ 🧪 🎯 | `chunkerkernevalmat.m` | MATLAB parity fixture checks eval matrix. |
| private helpers | 🧩 ✅ | Internal Python helpers | Chunker coercion, kernel evaluation, special-quadrature dispatch. |

### `chnk/arcparam.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `ArcParamData` | ✅ 🧪 | `+chnk/+arcparam/init.m` output struct | Python dataclass. |
| `init` | ✅ 🧪 | `+chnk/+arcparam/init.m` | Tested on chunk nodes. |
| `eval` | ✅ 🧪 | `+chnk/+arcparam/eval.m` | Tested for derivative consistency on a circle. |

### `chnk/curves.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `linefunc` | ✅ 🧪 | `+chnk/+curves/linefunc.m` | Tested. |
| `fpara` | ✅ | `+chnk/+curves/fpara.m` | Implemented; no focused test yet. |
| `fsine` | ✅ 🧪 | `+chnk/+curves/fsine.m` | Tested. |
| `bymode` | ✅ | `+chnk/+curves/bymode.m` | Implemented; no focused test yet. |
| `_pack` | 🧩 ✅ | Internal Python helper | No direct MATLAB file. |

### `chnk/lap2d.py`, `helm2d.py`, `helm1d.py`, `stok2d.py`, `elast2d.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `lap2d.green` | ✅ 🧪 | `+chnk/+lap2d/green.m` | Direct formula tested. |
| `lap2d.kern` | ⚠️ 🧪 🎯 | `+chnk/+lap2d/kern.m` | Most point kernels parity-tested; gradient row ordering xfails noted above. |
| `helm2d.green` | ✅ 🧪 | `+chnk/+helm2d/green.m` | Gradient finite-difference tested. |
| `helm2d.kern` | ⚠️ 🧪 🎯 | `+chnk/+helm2d/kern.m` | Most point kernels parity-tested; gradient row ordering xfails noted above. |
| `helm1d.green` | ✅ 🧪 | `+chnk/+helm1d/green.m` | Gradient finite-difference tested. |
| `helm1d.kern` | ✅ 🧪 🎯 | `+chnk/+helm1d/kern.m` | Many scalar/combined/transmission variants parity-tested. |
| `helm1d.sweep` | ✅ 🧪 | `+chnk/+helm1d/sweep.m` | Direct causal sums tested. |
| `stok2d.kern` | ✅ 🧪 🎯 | `+chnk/+stok2d/kern.m` | Stokes variants parity-tested. |
| `elast2d.kern` | ✅ 🧪 🎯 | `+chnk/+elast2d/kern.m` | Elasticity variants parity-tested. |
| private kernel helpers | 🧩 ✅ | Internal Python helpers | Interleaving and validation helpers. |

🚧 FMM entry points are intentionally absent for now: `+chnk/+lap2d/fmm.m`, `+chnk/+helm2d/fmm.m`, `+chnk/+stok2d/fmm.m`.

### `chnk/geometry.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `perp` | ✅ 🧪 | `+chnk/perp.m` | Tested. |
| `normal2d` | ✅ 🧪 | `+chnk/normal2d.m` | Tested. |
| `curvature2d` | ✅ 🧪 | `+chnk/curvature2d.m` | Tested. |
| `flagnear` | ✅ 🧪 | `@chunker/flagnear.m` / `@chunkgraph/flagnear.m` behavior | Delegates to chunker implementation. |
| `flagnear_rectangle` | ✅ 🧪 | `@chunker/flagnear_rectangle.m` | Delegates to chunker implementation. |
| `flagnear_rectangle_grid` | ✅ 🧪 | `@chunker/flagnear_rectangle_grid.m` | Delegates to chunker implementation. |
| `flagself` | ✅ 🧪 | `+chnk/flagself.m` | Tested. |
| `chunk_nearparam` | ✅ 🧪 | `+chnk/chunk_nearparam.m` | Tested on a line segment. |
| `_ptinfo_field` | 🧩 ✅ | Internal Python helper | No direct MATLAB file. |

### `chnk/quadnative.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `buildmat` | ✅ 🧪 🎯 | `+chnk/+quadnative/buildmat.m` | Dense native operator path parity-tested. |
| `_pointinfo_for_chunks` | 🧩 ✅ | Internal Python helper | No direct MATLAB file. |

### `chnk/quadggq.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `AuxQuad` | ✅ 🧪 | MATLAB aux quadrature structs/tables | Python dataclass-like table holder. |
| `setup` | ⚠️ 🧪 | `+chnk/+quadggq/setup.m` | Supports `log`/`removable`; other generated rule families are 🚧. |
| `getlogquad` | ✅ 🧪 | `+chnk/+quadggq/getlogquad.m`, `logwhts.mat` | Generated self rules tested. |
| `logavail` | ✅ 🧪 | `+chnk/+quadggq/logavail.m` | Tested. |
| `getremovablequad` | ✅ | `+chnk/+quadggq/getremovablequad.m` | Implemented; no focused test yet. |
| `buildmat` | ⚠️ 🧪 🎯 | `+chnk/+quadggq/buildmat.m` | Removes log singular diagonal issues; dense parity covered through `chunkermat`. |
| `diagbuildmat` | ✅ | `+chnk/+quadggq/diagbuildmat.m` | Implemented; no focused test yet. |
| `nearbuildmat` | ⚠️ | `+chnk/+quadggq/nearbuildmat.m` | Present baseline; add near-field parity tests later. |
| private helpers | 🧩 ✅ | Internal Python helpers | Interpolation/block slicing/dtype helpers. |

🚧 MATLAB files not yet ported here include `buildmattd.m`, `gethqsuppquad.m`, `setuplogquad.m`, and the large generated `ggqnear*`, `ggqself_*`, `hsupp_*`, `hqsupp_*` rule files beyond the compact Python baseline.

### `chnk/rcip.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `RCIPSaved` | ⚠️ | `+chnk/+rcip/*` saved structs | Metadata holder. |
| `IPinit` / `ipinit` | ✅ | `+chnk/+rcip/IPinit.m` | Implemented; no tests yet. |
| `Pbcinit` / `pbcinit` | ✅ | `+chnk/+rcip/Pbcinit.m` | Implemented; no tests yet. |
| `setup` | ✅ | `+chnk/+rcip/setup.m` | Zero-based Python index translation; no tests yet. |
| `SchurBana` / `schurbana` | ✅ | `+chnk/+rcip/SchurBana.m` | Implemented; no tests yet. |
| `Rcompchunk` / `rcompchunk` | ⚠️ | `+chnk/+rcip/Rcompchunk.m` | Returns identity compression and metadata; full recursive local compression solver is 🚧. |
| `rhohatInterp` / `rhohatinterp` | ⚠️ | `+chnk/+rcip/rhohatInterp.m` | Baseline saved-level interpolation; no tests yet. |
| `corner_refine` | ⚠️ | `+chnk/+rcip/chunkerfunclocal.m` and corner workflows | Convenience helper, not a direct MATLAB API match. |

### `chnk/spcl.py`

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `absconvgauss` | ✅ 🧪 | `+chnk/+spcl/absconvgauss.m` | Derivatives tested against finite differences. |

### `lege/core.py`

- ✅ 🧪 🎯 [src/chunkie/lege/core.py](src/chunkie/lege/core.py) maps most of MATLAB `+lege`.

| Python node | Flags | MATLAB reference | Notes |
| --- | --- | --- | --- |
| `pol` | ✅ | `+lege/pol.m` | Implemented; indirectly exercised by `pols`. |
| `pols` | ✅ 🧪 🎯 | `+lege/pols.m` | MATLAB parity fixture checks polynomials and derivatives. |
| `exps` | ✅ 🧪 🎯 | `+lege/exps.m` | MATLAB parity fixture checks nodes, weights, transforms. |
| `rts` | ✅ 🧪 | `+lege/rts.m` | Alias behavior tested. |
| `rts_stab` | ✅ 🧪 | `+lege/rts_stab.m` | Alias behavior tested. |
| `exev` | ✅ 🧪 🎯 | `+lege/exev.m` | MATLAB parity fixture checks expansion evaluation. |
| `derpol` | ✅ 🧪 🎯 | `+lege/derpol.m` | MATLAB parity fixture checks coefficients. |
| `dermat` | ✅ 🧪 🎯 | `+lege/dermat.m` | Tested and basic MATLAB fixture checked. |
| `intpol` | ✅ 🧪 🎯 | `+lege/intpol.m` | MATLAB parity fixture checks `true` and `original` options. |
| `intmat` | ✅ 🧪 🎯 | `+lege/intmat.m` | MATLAB parity fixture checks matrix values. |
| `matrin` | ✅ 🧪 🎯 | `+lege/matrin.m` | MATLAB parity fixture checks interpolation matrix. |
| `barywts` | ✅ 🧪 🎯 | `+lege/barywts.m` | MATLAB parity fixture checks weights. |

🚧 MATLAB `+lege` helpers not yet ported: `adapgauss.m`, `bernstein_ellipse.m`, `polsum.m`, `tayl.m`.

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
    ├── generate_kernel_pointinfo_fixture.m
    ├── generate_lege_basic_fixture.m
    ├── generate_lege_extended_fixture.m
    └── generate_operator_parity_fixture.m

tests/
├── golden/
│   ├── README.md
│   ├── chunker_circle.mat
│   ├── chunker_ops.mat
│   ├── kernel_pointinfo.mat
│   ├── lege_basic.mat
│   ├── lege_extended.mat
│   └── operator_parity.mat
├── test_arcparam.py
├── test_chunker.py
├── test_chunkerfit.py
├── test_chunkerfunc.py
├── test_chunkerpoly.py
├── test_chunkgraph.py
├── test_elast2d.py
├── test_geometry.py
├── test_helm1d.py
├── test_kernel.py
├── test_kernels.py
├── test_lege.py
├── test_matlab_fixtures.py
├── test_matlab_parity.py
├── test_operators.py
├── test_quadggq.py
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
- 🧪 `tests/golden/*.mat`: MATLAB-generated parity fixtures.
- 🧪 `tests/test_matlab_parity.py`: main exact-behavior comparison suite against golden data.
- 🧪 `tests/test_matlab_fixtures.py`: basic fixture comparison suite.
- 🧪 Other `tests/test_*.py`: Python behavioral/unit coverage.

## Major Unported MATLAB Areas

These are useful future implementation targets from `external/chunkie-matlab/chunkie` with no full Python equivalent yet.

- 🚧 Top-level geometry/domain helpers: `checkcurveparam.m`, `ellipse.m`, `hypoct_uni.m`, `mergeregions.m`, `nonflatinterface.m`, `pointinregion.m`, `redblue.m`, `regioninside.m`, `starfish.m`.
- 🚧 Trapper family: `@trapper/*`, `@trapperpref/*`, `trapperfunc.m`, `trapperkerneval.m`, `trappermat.m`.
- 🚧 FLAM/FMM accelerated paths: `chunkerflam.m`, `@kernel/*` FMM-backed workflows, `+chnk/+flam/*`, bundled FLAM and FMM2D integrations.
- 🚧 Axisymmetric / quasiperiodic / flex kernels: `+chnk/+axissymhelm2d/*`, `+chnk/+helm2dquas/*`, `+chnk/+flex2d/*`, plus matching `@kernel` factories.
- 🚧 Adaptive and close quadrature packages beyond the current baseline: `+chnk/+quadadap/*`, `+chnk/+quadba/*`, portions of `+chnk/+quadggq/*`.
- 🚧 Full RCIP recursive compression solver: `+chnk/+rcip/Rcompchunk.m` and related local recursion workflows.
- 🚧 Smoother and intchunk packages: `+chnk/+smoother/*`, `+chnk/+intchunk/*`.
- 🚧 Plotting/visualization methods: `plot`, `plot3`, `scatter`, `quiver`, `plot_regions` on MATLAB classes.

## Recommended Next Flags To Upgrade

- Add golden MATLAB fixtures for `chunkerpoly`, `chunkerfit`, `sortinfo`, `chunkgraph`, `arcparam`, `geometry`, and `spcl` to upgrade many ✅ 🧪 nodes to 🎯.
- Resolve the `lap2d`/`helm2d` gradient row interleaving xfails to turn those kernel rows from ⚠️ 🎯 to ✅ 🎯.
- Add focused tests for implemented but currently untested methods such as `min`, `max`, `chunkends`, `resize`, `cleardata`, `getremovablequad`, `diagbuildmat`, and `nearbuildmat`.
- Decide how far `rcip.py` should go in the near term: it is present and mapped, but much of the full MATLAB recursive compression workflow is still marked ⚠️/🚧.
