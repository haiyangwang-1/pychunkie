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
  2026-05-12 with the regenerated local fixture: `28 passed`
- Covered now: `absconvgaussTest.m`, `legeexpsunitTest.m`,
  `arclengthfunTest.m`, `chunker_diffintmatTest.m`,
  `chunker_nearestTest.m`, `chunkerclassunitTest.m`, `chunkerfitTest.m`,
  `chunkerfuncuniTest.m`, `chunkerfuncTest.m`, `chunkerintegralTest.m`,
  `chunkerinteriorTest.m`, `chunkerpolyTest.m`, `chunkerarcparamTest.m`,
  `chunkermat_quadadapTest.m`, `flagrectTest.m`, `flagselfTest.m`,
  `flagnearTest.m`, `helm2d_greenTest.m`, `kernelopTest.m`,
  `kernelclassTest.m`, `chunkerkerneval_greenlapTest.m`,
  `chunkerkerneval_greenhelmTest.m`, `chunkerkernevalmat_greenlapTest.m`,
  `slicegraphTest.m`, `smootherTest.m`, `stokes_dtracTest.m`, and
  `tochunkgraphTest.m`

## Status At A Glance

| Status | Count | Tests |
| --- | ---: | --- |
| ✅ 🧪 🎯 fully covered by devtools parity | 20 | `absconvgaussTest.m`, `legeexpsunitTest.m`, `arclengthfunTest.m`, `chunker_diffintmatTest.m`, `chunker_nearestTest.m`, `chunkerclassunitTest.m`, `chunkerfitTest.m`, `chunkerfuncuniTest.m`, `chunkerintegralTest.m`, `chunkerinteriorTest.m`, `chunkerkerneval_greenhelmTest.m`, `chunkerkernevalmat_greenlapTest.m`, `chunkermat_quadadapTest.m`, `chunkerpolyTest.m`, `flagrectTest.m`, `flagselfTest.m`, `flagnearTest.m`, `helm2d_greenTest.m`, `kernelopTest.m`, `tochunkgraphTest.m` |
| ✅ 🧪 🎯 ⚠️ partial/diagnostic devtools parity | 7 | `chunkerarcparamTest.m`, `chunkerfuncTest.m`, `chunkerkerneval_greenlapTest.m`, `kernelclassTest.m`, `slicegraphTest.m`, `smootherTest.m`, `stokes_dtracTest.m` |
| 🧩 🧭 helper/reference | 1 | `gradient_check.m` |
| 🚧 should/pending parity not yet converted | 35 | Remaining ranked entries below, excluding explicit non-goals. |
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
- Remaining `+lege` helpers: `adapgauss`, `bernstein_ellipse`, `polsum`, and
  `tayl`; strict MATLAB fixture parity now covers `bernstein_ellipse`,
  `polsum`, and scalar-call `tayl` outputs.
- Adaptive refinement in `chunker.refine` and `chunkerfunc`.
- `quadggq/buildmattd` sparse special-block assembly.
- Full adaptive/close quadrature for `quadadap`: GGQ self blocks, adaptive
  neighbor blocks, and robust close non-neighbor replacement for log kernels.
- Optional devtools parity fixtures for adaptive `chunkerfunc`,
  `chunkerarcparam`, `slicegraph`, and the starfish `chunkermat_quadadap`
  matrix comparison.
- `chunkermat(..., acceleration="fmm")` matrix-free FMM operators and
  `chunkermatapply` FMM acceleration with sparse special-quadrature
  corrections for singular kernels.
- `chunkerinterior` FMM classification with direct close-boundary correction.
- First-pass PyFLAM integration: `chunkerflam`, `chnk.flam` helper callbacks,
  `ChunkerFLAMMatrix`, FLAM-backed `chunkermat`/`chunkermatapply`, target
  evaluation/materialization, and `chunkerinterior` FLAM classification. These
  are covered by focused Python tests; the Green-identity target-evaluation
  devtools fixture now also saves MATLAB FLAM diagnostics.
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

Deferred implementation:

- Remaining FLAM parity beyond the first PyFLAM-backed pass: strict MATLAB
  devtools fixtures beyond the converted Green-identity target-evaluation
  diagnostic, full multi-chunker block-kernel workflows, and larger
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
| 4 | `smootherTest.m` | ✅ 🧪 🎯 ⚠️ | Easy | Builds a smoothed chunker from a triangular polygon with `chnk.smoother.smooth` and asserts the returned smoothing error is below `1e-6`. | Covered in `devtools_easy.mat`: compare triangle vertices plus MATLAB/Python smoother error thresholds. Python remains a lightweight rounded-polygon path, so exact geometry is not covered yet. |
| 5 | `arclengthfunTest.m` | ✅ 🧪 🎯 | Easy | Computes arclength coordinates on a circle and on two merged circles, comparing each component to analytic polar angle, with the second circle scaled by `1.1`. | Covered in `devtools_easy.mat`: compare single-component and merged-component arclength coordinates against Python `Chunker.arclengthfun`. |
| 6 | `chunker_diffintmatTest.m` | ✅ 🧪 🎯 | Easy | Builds ellipse/circle chunkers, checks `diffmat` produces unit arclength tangents, and checks `intmat` inverts differentiation up to a constant. | Covered in `devtools_easy.mat`: compare MATLAB `D`, `C`, tangent derivatives, integrated coordinates, and Python `Chunker.diffmat`/`intmat`. |
| 7 | `chunker_nearestTest.m` | ✅ 🧪 🎯 | Easy | For 1000 random radial targets around a circle, checks `nearest` returns the closest circle angle to within `1e-12`. | Covered in `devtools_easy.mat`: compare saved targets, closest points, derivatives, second derivatives, distances, panel ids, local parameters, and angle error against Python `Chunker.nearest`. |
| 8 | `flagselfTest.m` | ✅ 🧪 🎯 | Easy | Flags overlapping source and target point sets and checks the reported source permutation matches duplicated target coordinates. | Covered in `devtools_easy.mat`: compare deterministic source/target arrays and expected overlap index pairs against Python `geometry.flagself`. |
| 9 | `flagrectTest.m` | ✅ 🧪 🎯 | Easy | Tests rectangle-based near-flagging by comparing direct target flags to tensor-grid flags on a refined starfish chunker. | Covered in `devtools_easy.mat`: compare saved refined starfish chunker, target grid, direct flags, and grid flags against Python `flagnear_rectangle`/grid behavior. |
| 10 | `flagnearTest.m` | ✅ 🧪 🎯 | Easy | Tests near-point flagging against a brute-force distance check for targets scaled radially around a starfish curve. | Covered in `devtools_easy.mat`: compare saved starfish chunker, targets, MATLAB near flags, and brute-force flags against Python `geometry.flagnear`. |
| 11 | `kernelopTest.m` | ✅ 🧪 🎯 | Easy | Verifies kernel algebra: interleave, scalar multiply/divide, negation, addition, subtraction, and conjugation on Laplace/Helmholtz kernels. | Covered in `devtools_easy.mat`: compare saved kernel matrices from deterministic source/target normals. |
| 12 | `KernDerInterleaveTest.m` | ✅ 🧪 🚧 | Easy-Medium | Verifies algebraic relationships among interleaved Helmholtz, Helmholtz-difference, and Laplace kernels, including combined kernels, transmission blocks, gradients, and normal derivative as gradient dot normal. | Existing point-kernel fixtures cover much of this; add devtools fixture with the exact source/target/coefficient cases and compare all interleaved blocks. |
| 13 | `helm2d_greenTest.m` | ✅ 🧪 🎯 | Easy-Medium | Uses `gradient_check` to verify `chnk.helm2d.green` potential and gradient components against finite differences. | Covered in `devtools_easy.mat`: compare source/target, MATLAB Green value/gradient/Hessian, and saved finite-difference thresholds against Python `helm2d.green`. |
| 14 | `stokes_dtracTest.m` | ✅ 🧪 🎯 ⚠️ | Easy-Medium | Compares Stokes double-layer traction values against applying the double-layer gradient kernel and contracting with target normals. The MATLAB file prints the norm but does not assert. | Covered in `devtools_easy.mat`: compare saved Stokes `dtrac`, `dgrad`, `dpres`, reconstructed stress traction, and residual norm against Python kernels. |
| 15 | `chunkerclassunitTest.m` | ✅ 🧪 🎯 | Medium | Exercises `chunker` constructor failures, chunk allocation/resizing, adjacency links, translations, rotations, affine transforms, and area scaling. | Covered in `devtools_easy.mat`: compare constructor failure flags, adjacency reciprocity, translated/transformed/scaled chunker fields, centroids, and area scaling against Python `Chunker`. |
| 16 | `chunkerfitTest.m` | ✅ 🧪 🎯 | Medium | Samples a smooth curve at random points, fits a chunker, and checks adjacency after fitting and after an open-curve fit. | Covered in `devtools_easy.mat`: compare saved random sample points and MATLAB adjacency status against Python `chunkerfit` on the same closed/open inputs; remaining MATLAB fitting modes are deferred. |
| 17 | `chunkerfuncuniTest.m` | ✅ 🧪 🎯 | Medium | Builds uniformly chunked starfish/random-mode/circle curves and checks adjacency plus circle area. Also exercises plot/quiver/sort/reverse utilities lightly. | Covered in `devtools_easy.mat`: compare uniform starfish, reversed random-mode, and circle chunker fields, adjacency, and area against Python `chunkerfuncuni`. |
| 18 | `chunkerfuncTest.m` | ✅ 🧪 🎯 ⚠️ | Medium | Tests adaptive `chunkerfunc` on starfish, random Fourier-mode curves, reversal, circle area, and expected warnings for open/closed flags. | Covered in `devtools_easy.mat`: compare adaptive starfish, `nout=3`, random-mode, reversed random-mode, circle, and refined circle chunkers plus adjacency/area diagnostics. MATLAB warning flags are saved as fixture diagnostics; Python does not yet emit matching open/closed warnings. |
| 19 | `chunkerpolyTest.m` | ✅ 🧪 🎯 | Medium | Builds rounded and adaptively refined polygon chunkers for a barbell-like polygon and checks adjacency. | Covered in `devtools_easy.mat`: compare saved vertices, edge data, MATLAB adjacency status, and area/length diagnostics against Python `chunkerpoly` adjacency behavior. |
| 20 | `chunkerintegralTest.m` | ✅ 🧪 🎯 | Medium | Integrates a scalar function over a starfish chunker several ways and checks all routes agree to `1e-9`. | Covered in `devtools_easy.mat`: compare saved starfish chunker, scalar function values, and MATLAB integral variants against Python `chunkerintegral`. |
| 21 | `chunkerinteriorTest.m` | ✅ 🧪 🎯 | Medium | Classifies targets inside/outside starfish domains, including targets passed as arrays/chunkers, axisymmetric option, boundary convention, and a stress case against `inpolygon`. | Covered in `devtools_easy.mat`: compare saved starfish targets, chunker targets, MATLAB FLAM/FMM classifications, axisymmetric targets, and multiply connected stress grid against Python `chunkerinterior`, including the PyFLAM-backed classifier. |
| 22 | `chunkerarcparamTest.m` | ✅ 🧪 🎯 ⚠️ | Medium | Tests arc-length parameterization initialization/evaluation, derivatives, reparameterized chunker area/length, boundary moving, and unit-speed condition. | Covered in `devtools_easy.mat`: compare Python `chnk.arcparam.init/eval`, node/sample evaluations, derivative residuals, `arcresample(..., mv_bdries=0)`, area/length preservation, and unit-speed diagnostics. The MATLAB `mv_bdries=1` branch is not separately implemented in Python. |
| 23 | `tochunkgraphTest.m` | ✅ 🧪 🎯 | Medium | Converts merged circle/open-arc chunkers to a chunkgraph, checks vertices, edge count, point count, edge chunker preservation, and endpoint alignment after shift/scale. | Covered in `devtools_easy.mat`: compare merged chunker graph fields and manual `chunkgraph` endpoint alignment against Python `tochunkgraph`/`chunkgraph`. |
| 24 | `slicegraphTest.m` | ✅ 🧪 🎯 ⚠️ | Medium | Builds concentric-square chunkgraphs, slices selected edges, checks sliced geometry, compares sliced system matrix to full submatrix, and verifies edge id ordering. | Covered in `devtools_easy.mat`: compare sliced geometry and `edgeids`, and verify MATLAB and Python each preserve the inner sliced/full-submatrix relation. Direct dense matrix-value parity remains partial because current Python graph double-layer self blocks produce NaNs where MATLAB stores finite special values. |
| 25 | `chunkgraph_basicTest.m` | ✅ 🧪 🚧 | Medium | Tests chunkgraph constructors, legacy/new graph formats, multiply connected regions, bridge edges, loops, nested regions, dyadic refinement, and region id queries. | Save graph topology, region structures, and point-region ids; compare Python `chunkgraph` and region APIs. |
| 26 | `chunkgraph_lastlengthTest.m` | ✅ 🧪 🚧 | Medium | Checks chunkgraph refinement near vertices so all adjacent edge arclengths agree and are negative powers of two times `last_len`. | Save refined edge lengths and expected dyadic ratios; compare Python graph refinement when available. |
| 27 | `chunkgrphconstructTest.m` | ✅ 🧪 🚧 | Medium | Builds a pentagonal chunkgraph from circular/sine arcs and compares legacy connectivity construction against newer `edgesendverts` construction. | Save both graph forms and compare Python graph construction equivalence. |
| 28 | `chunkgrphregionTest.m` | ✅ 🧪 🚧 | Medium | Builds several graph regions, checks region edge orientation, edge-to-region map, and region numbering against manually specified truth. | Save expected region lists and edge-region maps; compare Python region construction. |
| 29 | `chunkrgrphOpdimTest.m` | ✅ 🚧 | Medium | Smoke-tests operator dimensions for chunkgraph kernels and block kernels. The file is short and mainly validates dimension plumbing. | Save kernel/operator dimension metadata and assembled shape outputs; compare Python once chunkgraph operator dimensions are supported. |
| 30 | `datafieldTest.m` | ✅ 🧪 🚧 ⚠️ | Medium-Hard | Tests chunker data fields used by custom kernels, including Hilbert/cotangent-style kernels, directional derivative single-layer kernels, FLAM comparison, and data propagation to target info. | Focused Python tests now cover PyFLAM callbacks for a data-dependent kernel with proxy disabled. Strict MATLAB data-field fixtures, custom-kernel matrices, and directional derivative diagnostics remain pending. |
| 31 | `kernelclassTest.m` | ✅ 🧪 🎯 ⚠️ | Medium-Hard | Tests `kernel` objects as carriers of FMM/singularity metadata through `chunkerkerneval`, verifies Green's identity, and checks NaN-kernel propagation. | Covered in `devtools_easy.mat`: compare saved sources, boundary densities, close-corrected Python `forceadap` Green identity, and NaN-kernel propagation. MATLAB direct/FMM equality is retained as fixture diagnostics; Python FMM close-target correction remains partial. |
| 32 | `elastickernelsTest.m` | ✅ 🧪 🚧 | Medium-Hard | Validates elasticity kernels by checking PDE residuals, alternative double-layer residuals, divergence, traction, Green's identity, gradient consistency, and direct kernel values. | Existing point-kernel fixtures cover some elasticity blocks; add saved PDE residual arrays and boundary identity values. |
| 33 | `helm1d_greenTest.m` | ✅ 🧪 🚧 ⚠️ | Medium-Hard | Tests 1D/interface Helmholtz Green functions and a fast solve wrapper for a layered/flat interface setup, ending with a relative error below `1e-5`. | Start with Green/sweep outputs before attempting the full GMRES/interface solve. |
| 34 | `complexificationTest.m` | 🚧 | Medium-Hard | Compares Sommerfeld-density solve/evaluation to a complexification solution for a flat interface point-charge setup. | Requires Python Sommerfeld/complexification support; fixture should save interface parameters, densities, potential, and exact solution. |
| 35 | `adapgausswtsTest.m` | ✅ 🧪 🚧 ⚠️ | Hard | Compares adaptive Gaussian quadrature weights for a near interaction against a GGQ-built reference matrix block on a starfish geometry. | Save the exact chunk, target, adaptive weights, reference block, and compare Python adaptive quadrature internals. |
| 36 | `chunkermat_quadadapTest.m` | ✅ 🧪 🎯 | Hard | Builds dense matrices with standard and adaptive quadrature for a starfish near-interaction case and asserts relative Frobenius agreement below `1e-9`. | Covered in `devtools_easy.mat`: compare saved starfish geometry, Helmholtz double-layer GGQ matrix, adaptive-neighbor matrix, and MATLAB/Python GGQ-vs-adaptive Frobenius agreement. |
| 37 | `chunkerkerneval_correctionsTest.m` | ✅ 🚧 ⚠️ | Hard | Checks special near-target correction in `chunkerkerneval`: corrected evaluation matches truth while uncorrected smooth evaluation is measurably wrong. | Save source geometry, targets, density, corrected/unfixed values, and compare Python near-correction behavior. |
| 38 | `chunkerkerneval_gaussidTest.m` | ✅ 🧪 🚧 ⚠️ | Hard | Uses Gauss' identity for the Laplace double-layer potential over grids and checks values are near `0` or `-1` depending on inside/outside classification. | Save grid/targets, classifications, and double-layer values; compare Python evaluation and interior classification. |
| 39 | `chunkerkerneval_greenlapTest.m` | ✅ 🧪 🎯 ⚠️ | Hard | Tests Laplace Green's identity using direct, FMM, and FLAM/smooth-work paths for layer potential evaluation at targets. | Covered in `devtools_easy.mat`: compare saved source strengths, boundary densities, target truth, close-corrected Python `forceadap` layer evaluations, and PyFLAM force-adaptive layer evaluations against direct and MATLAB FLAM values. MATLAB direct/FMM/FLAM equality is retained as fixture diagnostics. |
| 40 | `chunkerkerneval_greenhelmTest.m` | ✅ 🧪 🎯 | Hard | Tests Helmholtz Green's identity for a starfish geometry by comparing layer-potential evaluation to known field values at targets. | Covered in `devtools_easy.mat`: compare saved Helmholtz wave number, sources, boundary densities, targets, close-corrected Python `forceadap` layer evaluations, and Green-identity target values. |
| 41 | `chunkerkernevalmat_greenlapTest.m` | ✅ 🧪 🎯 | Hard | Builds Laplace target-evaluation matrices and checks applying them reproduces Green's identity target values. | Covered in `devtools_easy.mat`: compare saved sources, densities, targets, single-layer eval matrix, adaptive double-layer eval matrix, and the Green-identity target values against Python `chunkerkernevalmat(..., forceadap=True)`. |
| 42 | `chunkermatTest.m` | ✅ 🧪 🚧 ⚠️ | Hard | Builds a Laplace Dirichlet dense system matrix on a starfish, solves it, evaluates at targets, and checks exterior solution accuracy. | Existing operator parity covers smaller dense paths; add full starfish solve fixture when native/special quadrature is ready. |
| 43 | `chunkermat_helm2dTest.m` | ✅ 🧪 🚧 ⚠️ | Hard | Builds and solves a Helmholtz combined-field Dirichlet system on a starfish and checks target accuracy below `1e-10`. | Save system matrix, RHS, solution, target values, and compare Python `chunkermat`/`chunkerkerneval`. |
| 44 | `chunkermat_stok2dTest.m` | ✅ 🧪 🚧 ⚠️ | Hard | Validates Stokes single/double/pressure/traction/gradient kernels, then builds and solves a Stokes boundary integral problem with target accuracy checks. | Existing point-kernel fixtures cover Stokes blocks; add full Stokes matrix/solve fixture later. |
| 45 | `chunkermat_stok_tractiontest.m` | ✅ 🚧 ⚠️ | Hard | Builds a Stokes traction system and compares target velocity/traction evaluation to analytic values. | Save traction matrix, RHS, solution, and target diagnostics; compare Python Stokes traction support. |
| 46 | `chunkermat_l2scaleTest.m` | 🚧 | Hard | Constructs a Helmholtz transmission matrix manually and with `chunkermat` `l2scale`, then compares scaled matrix and density solution. | Save manual/scaled matrices, right-hand side, and solution; compare Python l2 scaling once implemented. |
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
| 58 | `kernel_interleaveTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Tests invalid interleave detection, then solves an exterior Neumann problem with a block interleaved Helmholtz representation and fast-direct interfaces. | Port in stages: invalid interleave, block matrix entries, dense solve, and FMM paths where applicable; FLAM/fast-direct parity can build on the new PyFLAM path once block-kernel workflows are converted. |
| 59 | `quasiperiodicTest.m` | 🚫 | Very Hard | Tests quasi-periodic Helmholtz kernels, shifted phase relations, combined/transmission/all/gradient kernels, and an integral-equation solve for a periodic scattering setup. | Do not port; quasiperiodic kernels are explicit non-goals. |
| 60 | `flamutilitiesTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Tests FLAM matrix builder utilities, dense-vs-FLAM matrix entry reconstruction, fast-direct solves, and target accuracy for Laplace problems. | First PyFLAM-backed Python coverage exists for square/rectangular `chnk.flam` callbacks, proxy geometry, sparse overwrite precedence, smooth diagonal shifts, smooth/special l2 scaling, data-field callbacks, `ChunkerFLAMMatrix` apply/adjoint/solve/logdet, target evaluation with adaptive correction, and interior classification. Strict MATLAB utility fixtures and full solve/evaluation devtools parity remain pending. |
| 61 | `flamproxybylevelTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Tests FLAM matrix building with level-dependent proxy points and checks solve/evaluation errors against tolerances. | Proxy helpers plus default and level-dependent PyFLAM proxy application are implemented and Python-tested for square matrix compression and rectangular target evaluation. MATLAB tolerance comparisons and larger stress fixtures remain pending. |
| 62 | `flamopdimsTest.m` | ✅ 🧪 🚧 ⚠️ | Very Hard | Tests FLAM with multi-operator-dimension Helmholtz systems across two chunkers and compares analytic solution accuracy. | Scalar and simple vector-opdim PyFLAM wiring is active through the generic operator surface, including smooth diagonal shifts and target evaluation/materialization. Full two-chunker Helmholtz/block-opdim devtools parity remains pending. |
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

1. `chunkerkerneval_gaussidTest.m`: coverable next with saved target grids,
   classifications, and close-corrected double-layer outputs.
2. `chunkermat_quadadap_closetotouchingTest.m`: lower-level robust close
   replacement is already covered in `quadggq.mat`; the remaining devtools
   work is the full two-disk solve/evaluation diagnostic.
3. `KernDerInterleaveTest.m`: Laplace and implemented Helmholtz point-kernel
   algebra is covered elsewhere, but the exact devtools test still needs
   Python support for the MATLAB 2D Helmholtz `all`/transmission/difference
   selectors before it can be strict parity without source changes.
