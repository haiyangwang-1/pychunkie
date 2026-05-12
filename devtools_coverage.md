# MATLAB Devtools Coverage Map

This file inventories `external/chunkie-matlab/devtools/test` and ranks the
MATLAB tests from easiest to hardest to port into Python parity tests.

## Legend

- ✅ Python package surface exists for the behavior the MATLAB test exercises.
- 🧪 There is Python test coverage for this slice.
- 🎯 The slice is compared against MATLAB-generated devtools golden data.
- ⚠️ Coverage is partial, has a known limitation, or the MATLAB test itself is diagnostic-only.
- 🚧 Devtools parity is not implemented yet or is intentionally deferred.
- 🚫 Explicit non-goal; do not port.
- 🧩 Helper/support file rather than package behavior.
- 🧭 Planning/reference item.

## Parity Workflow

The parity rule for this repo is:

1. Do not edit `external/chunkie-matlab/devtools/test`.
2. Add MATLAB wrapper scripts under `scripts/matlab` when fixture data is needed.
3. Save MATLAB outputs under ignored `tests/golden` fixture files. Tests
   generate missing fixture files on demand and fail if generation fails.
4. Recompute the same quantities from Python in `tests`, then compare. Remove
   local generated fixture files with `uv run python scripts/clean_test_data.py`.

## Current Snapshot

- MATLAB fixture generator: `scripts/matlab/generate_devtools_easy_fixture.m`
- MATLAB fixture data: generated-on-demand local `tests/golden/devtools_easy.mat`
- Python snapshot generator: `scripts/generate_devtools_easy_python_fixture.py`
- Python comparisons: `tests/test_devtools_parity.py`
- Python verification: `uv run pytest tests/test_devtools_parity.py` on
  2026-05-12 with the regenerated local fixture: `40 passed`
- Covered now: `absconvgaussTest.m`, `adapgausswtsTest.m`, `legeexpsunitTest.m`,
  `arclengthfunTest.m`, `chunker_diffintmatTest.m`,
  `chunker_nearestTest.m`, `chunkerclassunitTest.m`, `chunkerfitTest.m`,
  `chunkerfuncuniTest.m`, `chunkerfuncTest.m`, `chunkgraph_basicTest.m`,
  `chunkgraph_lastlengthTest.m`, `chunkgrphconstructTest.m`,
  `chunkgrphregionTest.m`, `chunkrgrphOpdimTest.m`,
  `chunkerintegralTest.m`,
  `chunkerinteriorTest.m`, `chunkerpolyTest.m`, `chunkerarcparamTest.m`,
  `chunkermat_l2scaleTest.m`, `chunkermat_quadadapTest.m`, `flagrectTest.m`, `flagselfTest.m`,
  `flagnearTest.m`, `flamutilitiesTest.m`, `helm2d_greenTest.m`, `kernelopTest.m`,
  `KernDerInterleaveTest.m`, `kernelclassTest.m`, `chunkerkerneval_greenlapTest.m`,
  `chunkerkerneval_gaussidTest.m`, `chunkerkerneval_greenhelmTest.m`,
  `chunkerkernevalmat_greenlapTest.m`, `slicegraphTest.m`, `smootherTest.m`,
  `stokes_dtracTest.m`, and `tochunkgraphTest.m`

## Status At A Glance

| Status | Count | Tests |
| --- | ---: | --- |
| ✅ 🧪 🎯 fully covered by devtools parity | 31 | `absconvgaussTest.m`, `adapgausswtsTest.m`, `legeexpsunitTest.m`, `arclengthfunTest.m`, `chunker_diffintmatTest.m`, `chunker_nearestTest.m`, `chunkerclassunitTest.m`, `chunkerfitTest.m`, `chunkerfuncuniTest.m`, `chunkerfuncTest.m`, `chunkgraph_basicTest.m`, `chunkgraph_lastlengthTest.m`, `chunkgrphconstructTest.m`, `chunkgrphregionTest.m`, `chunkrgrphOpdimTest.m`, `chunkerintegralTest.m`, `chunkerinteriorTest.m`, `chunkerkerneval_gaussidTest.m`, `chunkerkerneval_greenhelmTest.m`, `chunkerkernevalmat_greenlapTest.m`, `chunkermat_l2scaleTest.m`, `chunkermat_quadadapTest.m`, `chunkerpolyTest.m`, `flagrectTest.m`, `flagselfTest.m`, `flagnearTest.m`, `helm2d_greenTest.m`, `KernDerInterleaveTest.m`, `kernelopTest.m`, `stokes_dtracTest.m`, `tochunkgraphTest.m` |
| ✅ 🧪 🎯 ⚠️ partial/diagnostic devtools parity | 7 | `chunkerarcparamTest.m`, `chunkerkerneval_greenlapTest.m`, `datafieldTest.m`, `flamutilitiesTest.m`, `kernelclassTest.m`, `slicegraphTest.m`, `smootherTest.m` |
| 🧩 🧭 helper/reference | 1 | `gradient_check.m` |
| 🚧 should/pending parity not yet converted | 24 | Remaining ranked entries below, excluding explicit non-goals. |
| 🚫 explicit non-goal; do not port | 11 | `trappermatTest.m`, `quasiperiodicTest.m`, axisymmetric tests 63-68, and flexural tests 72-74. |

## Scope Triage For Remaining Work

Should implement:

- No active items remain from the current triage.

Implemented from this scope:

- Top-level geometry/domain helpers: `checkcurveparam`, `ellipse`,
  `hypoct_uni`, `mergeregions`, `nonflatinterface`, `pointinregion`, `redblue`,
  `regioninside`, and `starfish`; compact non-devtools fixture parity in
  `geometry_core.mat` now covers these helpers plus core chunker/chunkgraph
  geometry surfaces, chunker near-flag/nearest/translation helpers, and
  `chunkerfuncuni` uniform geometry, arcparam init/eval, plus chunkgraph
  procverts/refine/near-flag wrappers, conversion, operators, and region
  classification.
- Helmholtz double-gradient FMM selector wiring.
- Stokes traction FMM selector wiring.
- FMM integration across the implemented 2D kernel selector surface, including
  Laplace derived selectors, Helmholtz target-derivative selectors, full
  biharmonic scalar selector wiring, Stokes traction/combined paths, and
  elasticity single/traction/double/alternate-double workflows.
- Remaining `+lege` helpers: `rts`, `rts_stab`, `adapgauss`,
  `bernstein_ellipse`, `polsum`, and `tayl`; strict MATLAB fixture parity now
  covers `rts`, `rts_stab`, scalar `adapgauss`, `bernstein_ellipse`,
  `polsum`, and scalar-call `tayl` outputs.
- Adaptive refinement in `chunker.refine` and `chunkerfunc`.
- `quadggq/buildmattd` sparse special-block assembly.
- Full adaptive/close quadrature for `quadadap`: GGQ self blocks, adaptive
  neighbor blocks, direct `adapgausswts` neighbor-block parity, and robust
  close non-neighbor replacement for log kernels.
- Optional devtools parity fixtures for adaptive `chunkerfunc`,
  `chunkerarcparam`, `chunkgrphconstruct`, `slicegraph`, and the starfish
  `chunkermat_quadadap` matrix comparison.
- `chunkgrphregionTest.m`: full signed region construction now saves MATLAB's
  manually specified nested/disjoint region loops plus `find_edge_regions`
  side maps and compares them against Python's zero-based signed loops.
- `chunkrgrphOpdimTest.m`: chunkgraph edge-by-edge block kernels now assemble
  dense matrices with variable row/column operator dimensions; the fixture
  compares MATLAB shape, per-edge point counts, and representative off-diagonal
  blocks.
- `chunkermat(..., acceleration="fmm")` matrix-free FMM operators and
  `chunkermatapply` FMM acceleration with sparse special-quadrature
  corrections for singular kernels.
- `chunkerinterior` FMM classification with direct close-boundary correction.
- First-pass PyFLAM integration: `chunkerflam`, `chnk.flam` helper callbacks,
  `ChunkerFLAMMatrix`, FLAM-backed `chunkermat`/`chunkermatapply`, target
  evaluation/materialization, and `chunkerinterior` FLAM classification. These
  are covered by focused Python tests; the Green-identity target-evaluation
  devtools fixture now also saves MATLAB FLAM diagnostics, and the devtools
  fixture has strict MATLAB parity for square/circular/rectangular FLAM proxy
  geometry plus Hilbert/cotangent and target-data directional-derivative
  datafield slices.
- Advanced RCIP chunkgraph workflows: selected vertices, ignored vertices, and
  global block-kernel subselection for local corner compression.
- Laplace and Helmholtz Green-identity devtools target-evaluation fixtures:
  `kernelclass`, `chunkerkerneval_greenlap`, `chunkerkerneval_greenhelm`, and
  `chunkerkernevalmat_greenlap` now compare saved sources, boundary densities,
  target truth, adaptive close-target evaluation outputs, eval matrices, and
  NaN-kernel diagnostics. Python `forceadap` covers close-target corrections;
  Python FLAM force-adaptive target evaluation is checked against saved direct
  and MATLAB FLAM fixture values for the Laplace Green-identity workflow.
  MATLAB FMM values are retained as fixture diagnostics where the original
  devtools test used them.
- `chunkerkerneval_gaussidTest.m`: the fixture saves the full `40 x 40`
  target grid, unit double-layer density, MATLAB adaptive double-layer values,
  Gauss-identity residuals, and inside/outside labels; Python compares
  adaptive target evaluation values and direct interior classification.

Deferred implementation:

- Remaining FLAM parity beyond the first PyFLAM-backed pass: strict MATLAB
  devtools fixtures beyond the converted Green-identity target-evaluation and
  datafield diagnostics, full block-kernel multi-chunker workflows, and larger
  proxy-by-level stress coverage.
- Remaining `chunkerfit` modes beyond the implemented spline/open-line/circle
  paths.

Do not implement:

- The `trapper` family.
- Axisymmetric, quasiperiodic, and flexural kernel families, including
  `axissymhelm2d`, `axissymhelm2ddiff`, `helm2dquas`, most of `flex2d`, and
  matching kernel factories.
- `quadba`.
- Full nonlinear MATLAB smoother/Newton workflow and `+chnk/+intchunk`; the
  lightweight rounded-polygon smoother remains the supported path.
- MATLAB plotting/visualization methods: `plot`, `plot3`, `scatter`, `quiver`,
  and `plot_regions`.

## Ranked Test Inventory

| Rank | MATLAB file | Flags | Difficulty | What the MATLAB test covers | Python parity target |
| ---: | --- | --- | --- | --- | --- |
| 1 | `gradient_check.m` | 🧩 🧭 | Easy | Helper routine for finite-difference gradient convergence checks; it perturbs inputs, compares function value changes to returned gradients, and reports convergence errors. | Port only as shared test utility if needed; no standalone Python package API is tested. |
| 2 | `absconvgaussTest.m` | ✅ 🧪 🎯 | Easy | Checks `chnk.spcl.absconvgauss` value derivatives by running `gradient_check` on the smoothed absolute-value function and on its first derivative. | Covered in `devtools_easy.mat`: compare Python `spcl.absconvgauss` value, first derivative, second derivative, and saved MATLAB gradient-check thresholds. |
| 3 | `legeexpsunitTest.m` | ✅ 🧪 🎯 | Easy | Verifies Legendre nodes/weights, derivative matrix, integration matrix, derivative of `sin(x)`, antiderivative of `sin(x)`, and coefficient-space integration via `lege.intpol`/`lege.exev`. | Covered in `devtools_easy.mat`: compare `lege.exps`, `dermat`, `intmat`, `intpol`, and `exev` outputs. |
| 4 | `smootherTest.m` | ✅ 🧪 🎯 ⚠️ | Easy | Builds a smoothed chunker from a triangular polygon with `chnk.smoother.smooth` and asserts the returned smoothing error is below `1e-6`. | Covered in `devtools_easy.mat`: compare triangle vertices plus MATLAB/Python smoother error thresholds, then verify Python lightweight rounded-polygon geometry invariants: zero per-node errors, `2*nv` chunks, adjacency, unit normals, and positive chunk lengths. Exact MATLAB smoother geometry is still not covered. |
| 5 | `arclengthfunTest.m` | ✅ 🧪 🎯 | Easy | Computes arclength coordinates on a circle and on two merged circles, comparing each component to analytic polar angle, with the second circle scaled by `1.1`. | Covered in `devtools_easy.mat`: compare single-component and merged-component arclength coordinates against Python `Chunker.arclengthfun`. |
| 6 | `chunker_diffintmatTest.m` | ✅ 🧪 🎯 | Easy | Builds ellipse/circle chunkers, checks `diffmat` produces unit arclength tangents, and checks `intmat` inverts differentiation up to a constant. | Covered in `devtools_easy.mat`: compare MATLAB `D`, `C`, tangent derivatives, integrated coordinates, and Python `Chunker.diffmat`/`intmat`. |
| 7 | `chunker_nearestTest.m` | ✅ 🧪 🎯 | Easy | For 1000 random radial targets around a circle, checks `nearest` returns the closest circle angle to within `1e-12`. | Covered in `devtools_easy.mat`: compare saved targets, closest points, derivatives, second derivatives, distances, panel ids, local parameters, and angle error against Python `Chunker.nearest`. |
| 8 | `flagselfTest.m` | ✅ 🧪 🎯 | Easy | Flags overlapping source and target point sets and checks the reported source permutation matches duplicated target coordinates. | Covered in `devtools_easy.mat`: compare deterministic source/target arrays and expected overlap index pairs against Python `geometry.flagself`. |
| 9 | `flagrectTest.m` | ✅ 🧪 🎯 | Easy | Tests rectangle-based near-flagging by comparing direct target flags to tensor-grid flags on a refined starfish chunker. | Covered in `devtools_easy.mat`: compare saved refined starfish chunker, target grid, direct flags, and grid flags against Python `flagnear_rectangle`/grid behavior. |
| 10 | `flagnearTest.m` | ✅ 🧪 🎯 | Easy | Tests near-point flagging against a brute-force distance check for targets scaled radially around a starfish curve. | Covered in `devtools_easy.mat`: compare saved starfish chunker, targets, MATLAB near flags, and brute-force flags against Python `geometry.flagnear`. |
| 11 | `kernelopTest.m` | ✅ 🧪 🎯 | Easy | Verifies kernel algebra: interleave, scalar multiply/divide, negation, addition, subtraction, and conjugation on Laplace/Helmholtz kernels. | Covered in `devtools_easy.mat`: compare saved kernel matrices from deterministic source/target normals. |
| 12 | `KernDerInterleaveTest.m` | ✅ 🧪 🎯 | Easy-Medium | Verifies algebraic relationships among interleaved Helmholtz, Helmholtz-difference, and Laplace kernels, including combined kernels, transmission blocks, gradients, and normal derivative as gradient dot normal. | Covered in `devtools_easy.mat`: compare the exact source/target/coefficient cases for Laplace, Helmholtz, and Helmholtz-difference combined kernels, transmission blocks, `all` interleaving, gradients, and normal-derivative identities. |
| 13 | `helm2d_greenTest.m` | ✅ 🧪 🎯 | Easy-Medium | Uses `gradient_check` to verify `chnk.helm2d.green` potential and gradient components against finite differences. | Covered in `devtools_easy.mat`: compare source/target, MATLAB Green value/gradient/Hessian, and saved finite-difference thresholds against Python `helm2d.green`. |
| 14 | `stokes_dtracTest.m` | ✅ 🧪 🎯 | Easy-Medium | Compares Stokes double-layer traction values against applying the double-layer gradient kernel and contracting with target normals. The MATLAB file prints the norm; Python asserts the saved relation. | Covered in `devtools_easy.mat`: compare saved Stokes `dtrac`, `dgrad`, `dpres`, reconstructed stress traction, and residual norm against Python kernels. |
| 15 | `chunkerclassunitTest.m` | ✅ 🧪 🎯 | Medium | Exercises `chunker` constructor failures, chunk allocation/resizing, adjacency links, translations, rotations, affine transforms, and area scaling. | Covered in `devtools_easy.mat`: compare constructor failure flags, adjacency reciprocity, translated/transformed/scaled chunker fields, centroids, and area scaling against Python `Chunker`. |
| 16 | `chunkerfitTest.m` | ✅ 🧪 🎯 | Medium | Samples a smooth curve at random points, fits a chunker, and checks adjacency after fitting and after an open-curve fit. | Covered in `devtools_easy.mat`: compare saved random sample points, MATLAB status, and Python `chunkerfit` closed/open fields (`r`, `d`, normals, weights, adjacency, chunk lengths, and area) against the MATLAB fixture. Second-derivative fields and remaining MATLAB fitting modes are deferred. |
| 17 | `chunkerfuncuniTest.m` | ✅ 🧪 🎯 | Medium | Builds uniformly chunked starfish/random-mode/circle curves and checks adjacency plus circle area. Also exercises plot/quiver/sort/reverse utilities lightly. | Covered in `devtools_easy.mat`: compare uniform starfish, reversed random-mode, and circle chunker fields, adjacency, and area against Python `chunkerfuncuni`. |
| 18 | `chunkerfuncTest.m` | ✅ 🧪 🎯 | Medium | Tests adaptive `chunkerfunc` on starfish, random Fourier-mode curves, reversal, circle area, and expected warnings for open/closed flags. | Covered in `devtools_easy.mat`: compare adaptive starfish, `nout=3`, random-mode, reversed random-mode, circle, refined circle chunkers, adjacency, area diagnostics, and matching Python open/closed endpoint warning behavior. |
| 19 | `chunkerpolyTest.m` | ✅ 🧪 🎯 | Medium | Builds rounded and adaptively refined polygon chunkers for a barbell-like polygon and checks adjacency. | Covered in `devtools_easy.mat`: compare saved vertices, edge data, MATLAB status, true-polygon area/length diagnostics, Python rounded chunk count/data dimensions/positive lengths, and open lightweight adjacency/positive lengths. Exact MATLAB rounded/open geometry remains partial because Python uses the supported lightweight path. |
| 20 | `chunkerintegralTest.m` | ✅ 🧪 🎯 | Medium | Integrates a scalar function over a starfish chunker several ways and checks all routes agree to `1e-9`. | Covered in `devtools_easy.mat`: compare saved starfish chunker, scalar function values, and MATLAB integral variants against Python `chunkerintegral`. |
| 21 | `chunkerinteriorTest.m` | ✅ 🧪 🎯 | Medium | Classifies targets inside/outside starfish domains, including targets passed as arrays/chunkers, axisymmetric option, boundary convention, and a stress case against `inpolygon`. | Covered in `devtools_easy.mat`: compare saved starfish targets, chunker targets, MATLAB FLAM/FMM classifications, axisymmetric targets, and multiply connected stress grid against Python `chunkerinterior`, including the PyFLAM-backed classifier. |
| 22 | `chunkerarcparamTest.m` | ✅ 🧪 🎯 ⚠️ | Medium | Tests arc-length parameterization initialization/evaluation, derivatives, reparameterized chunker area/length, boundary moving, and unit-speed condition. | Covered in `devtools_easy.mat`: compare Python `chnk.arcparam.init/eval`, node/sample evaluations, derivative residuals, `arcresample(..., mv_bdries=0)`, area/length preservation, and unit-speed diagnostics. The MATLAB `mv_bdries=1` branch is not separately implemented in Python. |
| 23 | `tochunkgraphTest.m` | ✅ 🧪 🎯 | Medium | Converts merged circle/open-arc chunkers to a chunkgraph, checks vertices, edge count, point count, edge chunker preservation, and endpoint alignment after shift/scale. | Covered in `devtools_easy.mat`: compare merged chunker graph fields and manual `chunkgraph` endpoint alignment against Python `tochunkgraph`/`chunkgraph`. |
| 24 | `slicegraphTest.m` | ✅ 🧪 🎯 ⚠️ | Medium | Builds concentric-square chunkgraphs, slices selected edges, checks sliced geometry, compares sliced system matrix to full submatrix, and verifies edge id ordering. | Covered in `devtools_easy.mat`: compare sliced geometry and `edgeids`, and verify MATLAB and Python each preserve the inner sliced/full-submatrix relation. Direct dense matrix-value parity remains partial because current Python graph double-layer self blocks produce NaNs where MATLAB stores finite special values. |
| 25 | `chunkgraph_basicTest.m` | ✅ 🧪 🎯 | Medium | Tests chunkgraph constructors, legacy/new graph formats, multiply connected regions, bridge edges, loops, nested regions, dyadic refinement, and region id queries. | Covered in `devtools_easy.mat`: compare legacy/new incidence matrices, region counts for multiply connected/bridge/loop/nested graphs, adjacent and nested point-region ids, affine/scale/rotate/reflect region-query invariance, and graph refinement chunk counts against Python `chunkgraph` APIs. |
| 26 | `chunkgraph_lastlengthTest.m` | ✅ 🧪 🎯 | Medium | Checks chunkgraph refinement near vertices so all adjacent edge arclengths agree and are negative powers of two times `last_len`. | Covered in `devtools_easy.mat`: compare MATLAB selected-edge refinement, per-edge split-chunk routing, `NaN` closed-edge construction, graph balancing counts, and `last_len` endpoint-panel arclengths/degrees against Python `ChunkGraph.refine`. |
| 27 | `chunkgrphconstructTest.m` | ✅ 🧪 🎯 | Medium | Builds a pentagonal chunkgraph from circular/sine arcs and compares legacy connectivity construction against newer `edgesendverts` construction. | Covered in `devtools_easy.mat`: compare MATLAB balanced vertices, incidence matrices, endpoint indices, and first-edge sine-arc chunker geometry against Python legacy-incidence and `edgesendverts` construction. |
| 28 | `chunkgrphregionTest.m` | ✅ 🧪 🎯 | Medium | Builds several graph regions, checks region edge orientation, edge-to-region map, and region numbering against manually specified truth. | Covered in `devtools_easy.mat`: compare MATLAB signed region loops, Python zero-based signed region conversion, region count, and `find_edge_regions` side maps. |
| 29 | `chunkrgrphOpdimTest.m` | ✅ 🧪 🎯 | Medium | Smoke-tests operator dimensions for chunkgraph kernels and block kernels. The file is short and mainly validates dimension plumbing. | Covered in `devtools_easy.mat`: compare graph edge point counts, variable row/column opdim matrix shape, and representative off-diagonal blocks for Helmholtz transmission block kernels. |
| 30 | `datafieldTest.m` | ✅ 🧪 🎯 ⚠️ | Medium-Hard | Tests chunker data fields used by custom kernels, including Hilbert/cotangent-style kernels, directional derivative single-layer kernels, FLAM comparison, and data propagation to target info. | Covered in `devtools_easy.mat` for Hilbert/cotangent source-data dense/FLAM products and the directional-derivative single-layer target-data slice: compare saved MATLAB density, direct/adaptive/FLAM target evaluations, and directional-gradient truth. Wider custom-kernel matrix stress cases remain pending. |
| 31 | `kernelclassTest.m` | ✅ 🧪 🎯 ⚠️ | Medium-Hard | Tests `kernel` objects as carriers of FMM/singularity metadata through `chunkerkerneval`, verifies Green's identity, and checks NaN-kernel propagation. | Covered in `devtools_easy.mat`: compare saved sources, boundary densities, close-corrected Python `forceadap` Green identity, and NaN-kernel propagation. MATLAB direct/FMM equality is retained as fixture diagnostics; Python FMM close-target correction remains partial. |
| 32 | `elastickernelsTest.m` | ✅ 🧪 🚧 | Medium-Hard | Validates elasticity kernels by checking PDE residuals, alternative double-layer residuals, divergence, traction, Green's identity, gradient consistency, and direct kernel values. | Existing point-kernel fixtures cover some elasticity blocks; add saved PDE residual arrays and boundary identity values. |
| 33 | `helm1d_greenTest.m` | ✅ 🧪 🚧 ⚠️ | Medium-Hard | Tests 1D/interface Helmholtz Green functions and a fast solve wrapper for a layered/flat interface setup, ending with a relative error below `1e-5`. | Start with Green/sweep outputs before attempting the full GMRES/interface solve. |
| 34 | `complexificationTest.m` | 🚧 | Medium-Hard | Compares Sommerfeld-density solve/evaluation to a complexification solution for a flat interface point-charge setup. | Requires Python Sommerfeld/complexification support; fixture should save interface parameters, densities, potential, and exact solution. |
| 35 | `adapgausswtsTest.m` | ✅ 🧪 🎯 | Hard | Compares adaptive Gaussian quadrature weights for a near interaction against a GGQ-built reference matrix block on a starfish geometry. | Covered in `devtools_easy.mat`: compare the saved starfish geometry, Helmholtz double-layer kernel, source chunk, neighboring target chunk, MATLAB adaptive weights, recursion metadata, and GGQ reference block against Python `quadadap.adapgausswts`. |
| 36 | `chunkermat_quadadapTest.m` | ✅ 🧪 🎯 | Hard | Builds dense matrices with standard and adaptive quadrature for a starfish near-interaction case and asserts relative Frobenius agreement below `1e-9`. | Covered in `devtools_easy.mat`: compare saved starfish geometry, Helmholtz double-layer GGQ matrix, adaptive-neighbor matrix, and MATLAB/Python GGQ-vs-adaptive Frobenius agreement. |
| 37 | `chunkerkerneval_correctionsTest.m` | ✅ 🚧 ⚠️ | Hard | Checks special near-target correction in `chunkerkerneval`: corrected evaluation matches truth while uncorrected smooth evaluation is measurably wrong. | Save source geometry, targets, density, corrected/unfixed values, and compare Python near-correction behavior. |
| 38 | `chunkerkerneval_gaussidTest.m` | ✅ 🧪 🎯 | Hard | Uses Gauss' identity for the Laplace double-layer potential over grids and checks values are near `0` or `-1` depending on inside/outside classification. | Covered in `devtools_easy.mat`: compare the full saved `40 x 40` target grid, unit density, MATLAB adaptive double-layer values, Gauss-identity residuals, and inside/outside labels against Python `chunkerkerneval(..., forceadap=True)` and direct `chunkerinterior`. |
| 39 | `chunkerkerneval_greenlapTest.m` | ✅ 🧪 🎯 ⚠️ | Hard | Tests Laplace Green's identity using direct, FMM, and FLAM/smooth-work paths for layer potential evaluation at targets. | Covered in `devtools_easy.mat`: compare saved source strengths, boundary densities, target truth, close-corrected Python `forceadap` layer evaluations, and PyFLAM force-adaptive layer evaluations against direct and MATLAB FLAM values. MATLAB direct/FMM/FLAM equality is retained as fixture diagnostics. |
| 40 | `chunkerkerneval_greenhelmTest.m` | ✅ 🧪 🎯 | Hard | Tests Helmholtz Green's identity for a starfish geometry by comparing layer-potential evaluation to known field values at targets. | Covered in `devtools_easy.mat`: compare saved Helmholtz wave number, sources, boundary densities, targets, close-corrected Python `forceadap` layer evaluations, and Green-identity target values. |
| 41 | `chunkerkernevalmat_greenlapTest.m` | ✅ 🧪 🎯 | Hard | Builds Laplace target-evaluation matrices and checks applying them reproduces Green's identity target values. | Covered in `devtools_easy.mat`: compare saved sources, densities, targets, single-layer eval matrix, adaptive double-layer eval matrix, and the Green-identity target values against Python `chunkerkernevalmat(..., forceadap=True)`. |
| 42 | `chunkermatTest.m` | ✅ 🧪 🚧 ⚠️ | Hard | Builds a Laplace Dirichlet dense system matrix on a starfish, solves it, evaluates at targets, and checks exterior solution accuracy. | Existing operator parity covers smaller dense paths; add full starfish solve fixture when native/special quadrature is ready. |
| 43 | `chunkermat_helm2dTest.m` | ✅ 🧪 🚧 ⚠️ | Hard | Builds and solves a Helmholtz combined-field Dirichlet system on a starfish and checks target accuracy below `1e-10`. | Save system matrix, RHS, solution, target values, and compare Python `chunkermat`/`chunkerkerneval`. |
| 44 | `chunkermat_stok2dTest.m` | ✅ 🧪 🚧 ⚠️ | Hard | Validates Stokes single/double/pressure/traction/gradient kernels, then builds and solves a Stokes boundary integral problem with target accuracy checks. | Existing point-kernel fixtures cover Stokes blocks; add full Stokes matrix/solve fixture later. |
| 45 | `chunkermat_stok_tractiontest.m` | ✅ 🚧 ⚠️ | Hard | Builds a Stokes traction system and compares target velocity/traction evaluation to analytic values. | Save traction matrix, RHS, solution, and target diagnostics; compare Python Stokes traction support. |
| 46 | `chunkermat_l2scaleTest.m` | ✅ 🧪 🎯 | Hard | Constructs a Helmholtz transmission matrix manually and with `chunkermat` `l2scale`, then compares scaled matrix and density solution. | Covered in `devtools_easy.mat`: compare MATLAB and Python manual l2-scaled Helmholtz transmission-style block matrices against `chunkermat(..., {"l2scale": "true"})` and verify the scaled/unscaled density solves agree to the diagnostic threshold. |
| 47 | `chunkermatapplyTest.m` | ✅ 🧪 🚧 | Hard | Tests matrix-free `chunkermatapply` against dense matrices for scalar, vector-valued, multi-boundary, and block-kernel cases; includes solve comparisons. | Python covers dense products, explicit `chunkermat` FMM LinearOperator products, and FMM-accelerated special-corrected application; save dense/matrix-free outputs by case for strict MATLAB parity. |
| 48 | `chunkermat_quadadap_closetotouchingTest.m` | ✅ 🧪 🚧 ⚠️ | Hard | Solves an exterior Laplace Dirichlet problem on two nearly touching disks and demonstrates adaptive quadrature accuracy near close interactions. | Compact non-devtools `quadggq.mat` now covers robust close matrix replacement on two disks; still save the full solve, targets, and adaptive/smooth diagnostics for devtools parity. |
| 49 | `chunkermat_truepolygonTest.m` | ✅ 🚧 ⚠️ | Hard | Solves a true-corner polygon Neumann problem using adaptive refinement/RCIP-style machinery and checks target accuracy. | Save refined polygon graph/chunker, system, RHS, solution, and targets; compare after Python true-corner support matures. |
| 50 | `pquadTest.m` | 🚧 | Hard | Tests product quadrature for an interior Helmholtz problem by comparing product-quadrature layer-potential values to a reference solution. | Save product-quadrature matrices/values and compare Python pquad implementation when available. |
| 51 | `singularkernelTest.m` | ✅ 🧪 🚧 ⚠️ | Hard | Checks principal-value and hypersingular quadratures by comparing boundary tangential and normal derivatives from singular kernels to analytic boundary data. | Save boundary fields, PV/HS values, and relative errors; compare Python singular GGQ paths. |
| 52 | `trappermatTest.m` | 🚫 | Hard | Builds a trapper discretization, assembles a Laplace Dirichlet matrix, compares GMRES/backslash solutions, and evaluates target accuracy. The file prints diagnostics but has no final assert. | Do not port; the trapper family is an explicit non-goal. |
| 53 | `rcipTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Tests RCIP for an exterior Dirichlet problem on two circular arcs meeting at corners, comparing system matrices, solutions, interpolation to fine grid, and target accuracy. | Python has some RCIP helpers; save coarse/fine operators and densities before attempting full solve parity. |
| 54 | `chunkgrphrcip_ignoreTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Tests chunkgraph RCIP with artificial vertices ignored, verifying solution accuracy improves when ignored vertices are treated correctly. | Python now covers selected/ignored vertex RCIP compression; save graph, ignored-vertex metadata, solutions with/without ignoring, and target values for strict solve parity. |
| 55 | `chunkgrphrcipTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Builds pentagonal chunkgraphs and solves an interior Helmholtz Dirichlet problem with RCIP refinement. | Python now has a chunkgraph-level RCIP compression driver; save graph, RCIP setup, system, RHS, solution, targets for full solve parity. |
| 56 | `chunkgrphrcipTransmissionTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Solves a Helmholtz transmission problem on a chunkgraph with RCIP refinement and block kernels. | Python now subselects global edge-by-edge block kernels for local RCIP compression; save block system, material parameters, graph, densities, and interior/exterior target values for full solve parity. |
| 57 | `mixedbcTest.m` | ✅ 🚧 ⚠️ | Very Hard | Tests mixed boundary conditions: Dirichlet/Neumann and Dirichlet/transmission, variable operator dimensions, RCIP scaling, target evaluation, and correction matrices. | Save the two mixed systems separately; port block dimensions and direct evaluation before RCIP-corrected solve parity. |
| 58 | `kernel_interleaveTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Tests invalid interleave detection, then solves an exterior Neumann problem with a block interleaved Helmholtz representation and fast-direct interfaces. | Python now covers smooth interleaved block-kernel FLAM matrix and target-evaluation application against dense products. Port remaining stages: invalid interleave fixture, Helmholtz block matrix entries, dense solve, FMM paths where applicable, and full fast-direct solve parity. |
| 59 | `quasiperiodicTest.m` | 🚫 | Very Hard | Tests quasi-periodic Helmholtz kernels, shifted phase relations, combined/transmission/all/gradient kernels, and an integral-equation solve for a periodic scattering setup. | Do not port; quasiperiodic kernels are explicit non-goals. |
| 60 | `flamutilitiesTest.m` | ✅ 🧪 🎯 ⚠️ | Very Hard | Tests FLAM matrix builder utilities, dense-vs-FLAM matrix entry reconstruction, fast-direct solves, and target accuracy for Laplace problems. | Strict MATLAB fixture parity covers square/circular/rectangular FLAM proxy geometry, the square inside predicate, deterministic Laplace `nproxy_square`, square/rectangular `kernbyindex` callbacks with sparse overwrite precedence, square/rectangular `proxyfun` matrices with filtered neighbor indices, plus a compact Laplace `chunkerflam` `rskelf_mv`/`rskelf_sv` comparison against the PyFLAM-backed operator on the same deterministic RHS. First PyFLAM-backed Python coverage also exists for square/rectangular `chnk.flam` callbacks, sparse overwrite precedence, smooth diagonal shifts, smooth/special l2 scaling, data-field callbacks, `ChunkerFLAMMatrix` apply/adjoint/solve/adjoint-solve/logdet, target evaluation with adaptive correction, and interior classification. Full devtools matrix-entry and target-evaluation parity remains pending. |
| 61 | `flamproxybylevelTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Tests FLAM matrix building with level-dependent proxy points and checks solve/evaluation errors against tolerances. | Proxy helpers plus default and level-dependent PyFLAM proxy application are implemented and Python-tested for square matrix compression and rectangular target evaluation. MATLAB tolerance comparisons and larger stress fixtures remain pending. |
| 62 | `flamopdimsTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Tests FLAM with multi-operator-dimension Helmholtz systems across two chunkers and compares analytic solution accuracy. | Scalar explicit chunker-sequence, smooth interleaved block-kernel matrix/evaluation, and simple vector-opdim PyFLAM wiring is active through the generic operator surface, including smooth diagonal shifts and target evaluation/materialization. Full two-chunker Helmholtz/block-opdim devtools parity remains pending. |
| 63 | `axissymkernTest.m` | 🚫 ⚠️ | Very Hard | Compares axisymmetric Helmholtz kernel evaluations against explicit azimuthal integral reference kernels, including shifted-kernel behavior. The file prints errors rather than asserting. | Do not port; axisymmetric kernels are explicit non-goals. |
| 64 | `axissymkernel_sphereTest.m` | 🚫 ⚠️ | Very Hard | Builds axisymmetric sphere geometry, solves/evaluates a modal layer potential, and compares against spherical Bessel/Hankel analytic values. It prints the error. | Do not port; axisymmetric modal kernels are explicit non-goals. |
| 65 | `chunkermat_axissymhelm2dTest.m` | 🚫 | Very Hard | Builds and solves an axisymmetric Helmholtz 2D matrix problem on a starfish-like generating curve and checks target accuracy. | Do not port; axisymmetric matrix builders are explicit non-goals. |
| 66 | `chunkermat_axissymhelm_neumannTest.m` | 🚫 ⚠️ | Very Hard | Solves an axisymmetric Helmholtz exterior Neumann problem, compares adaptive target evaluation, matrix-entry reconstruction, and fast-direct solution diagnostics. It prints errors rather than asserting. | Do not port; axisymmetric kernels and matrix builders are explicit non-goals. |
| 67 | `chunkermat_axissymhelm_transmissionTest.m` | 🚫 ⚠️ | Very Hard | Solves an axisymmetric Helmholtz transmission problem with interior/exterior target checks, matrix-entry reconstruction, and fast-direct diagnostics. | Do not port; axisymmetric transmission kernels are explicit non-goals. |
| 68 | `chunkermat_axissymhelm_transmissionTest_re.m` | 🚫 ⚠️ | Very Hard | Variant of the axisymmetric Helmholtz transmission test, using a related representation/scaling path and the same style of target/matrix/fast-direct diagnostics. | Do not port; axisymmetric transmission variants are explicit non-goals. |
| 69 | `chunkermat_biharmonic_interiorsupportedTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Solves an interior biharmonic supported-plate boundary value problem and checks target accuracy below `1e-9`. | Python has biharmonic point kernels; save full supported-plate matrix/RHS/solution/targets for later operator parity. |
| 70 | `chunkermat_biharmonic_interiorclampedTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Solves an interior biharmonic clamped-plate problem and checks target accuracy below `1e-9`. | Save matrix/RHS/solution/targets; port once biharmonic boundary operators are implemented. |
| 71 | `chunkermat_biharmonic_interiorfreeTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Solves an interior biharmonic free-plate problem, including multiple boundary operators, and checks target accuracy below `1e-9`. | Save all operator blocks and solution diagnostics; likely harder than clamped/supported due free boundary conditions. |
| 72 | `chunkermat_flex2d_exteriorsupportedTest.m` | 🚫 | Very Hard | Solves an exterior flexural-wave supported boundary problem with plate PDE coefficients and checks target accuracy below `1e-9`. | Do not port; flexural kernel/operator families are explicit non-goals. |
| 73 | `chunkermat_flex2d_exteriorclampedTest.m` | 🚫 | Very Hard | Solves an exterior flexural-wave clamped boundary problem and checks target accuracy below `1e-9`. | Do not port; flexural kernel/operator families are explicit non-goals. |
| 74 | `chunkermat_flex2d_exteriorfreeTest.m` | 🚫 | Very Hard | Solves an exterior flexural-wave free boundary problem with multiple boundary operators and checks target accuracy below `1e-9`. | Do not port; flexural kernel/operator families are explicit non-goals. |

## Suggested Next Ports

1. `chunkermat_quadadap_closetotouchingTest.m`: lower-level robust close
   replacement is already covered in `quadggq.mat`; the remaining devtools
   work is the full two-disk solve/evaluation diagnostic.
2. `chunkerkerneval_correctionsTest.m`: save the corrected and uncorrected
   near-target values plus the source geometry and density.
