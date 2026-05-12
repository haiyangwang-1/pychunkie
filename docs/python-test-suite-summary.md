# Python Test Suite Summary

This document summarizes the Python tests under `tests/test_*.py`. The current
collection expands to 331 pytest cases because several MATLAB parity tests are
parametrized; those parametrized functions are described once, with the covered
selector list called out explicitly.
MATLAB parity fixture files under `tests/golden` are ignored and generated on
demand during tests, so the full suite requires a populated
`external/chunkie-matlab` checkout.

## What The Suite Is Checking

The suite is a mix of analytic unit tests, dense direct boundary-integral tests,
FMM cross-checks, special-quadrature tests, RCIP corner-compression tests, and
MATLAB golden-fixture parity tests. Most tests use Legendre panels: a `Chunker`
stores geometry at Gauss-Legendre nodes on each panel, with dense/native
operator matrices using weights in the source dimension:

```text
u(x_i) = sum_j K(x_i, y_j) sigma_j w_j
M_ij = K(x_i, y_j) w_j
```

Several tests reuse the unit-circle parameterization
`r(t) = (cos(t), sin(t))`, `r'(t) = (-sin(t), cos(t))`, and
`r''(t) = (-cos(t), -sin(t))`; radius and translation variants scale this
formula. The exact circle ground truths are area `pi R^2`, perimeter `2 pi R`,
unit tangent speed after arclength normalization, and curvature `1/R`.

The main PDE kernels are:

```text
Laplace 2D:      G(x,y) = -log(|x-y|^2) / (4 pi)
Helmholtz 2D:   G_k(x,y) = i H_0^(1)(k |x-y|) / 4
Helmholtz 1D:   G_k(x,y) = exp(i k |x-y|)
Biharmonic 2D:  G(x,y) = |x-y|^2 log(|x-y|) / (8 pi)
```

For layer kernels, the common selectors are single layer `s`, double layer `d`,
target-normal derivative `sp`, double-prime/normal-normal derivative `dp`,
gradients such as `sgrad` and `dgrad`, and combined kernels such as `c`.
Stokes tests cover velocity, pressure, traction, and gradient blocks for the
2D Stokeslet/stresslet family. Elasticity tests cover the Kelvin single layer,
traction, double layer, alternate double layer, and derivative/traction forms
for Lame parameters `lam` and `mu`.

Special quadrature is provided by `chnk.quadggq`. Logarithmic kernels use log
GGQ rules; principal-value kernels use PV support tables; hypersingular kernels
use HS support tables. Accelerated tests request
`{"acceleration": "fmm"}` and compare against the dense direct path. FLAM
tests request `{"acceleration": "flam"}` and compare PyFLAM-backed matrix,
target-evaluation, and interior-classification paths against dense/FMM
references. RCIP tests exercise recursive compressed inverse preconditioning
for corner edges.

Ground truth comes from four places:

- Closed-form geometry and PDE identities, such as circle area, line-segment
  nearest points, Green's functions, and traction formulas.
- Finite differences, used for derivative and gradient validation.
- Dense direct/native computation, used as the reference for FMM, special
  dispatch, and operator wrapper tests.
- MATLAB-generated golden fixtures in `tests/golden`, used for strict parity
  with the MATLAB `chunkIE` implementation. The `.mat` fixture files are
  ignored and generated on demand by `tests/_fixture_generation.py`; if MATLAB
  setup or fixture generation fails, the requesting test fails.

## Implementation Scope Tracked By Tests

The living docs split remaining MATLAB parity work into three buckets.

Should implement:

- No active items remain from the current triage.

Implemented from this scope:

- Top-level geometry/domain helpers: `checkcurveparam`, `ellipse`,
  `hypoct_uni`, `mergeregions`, `nonflatinterface`, `pointinregion`, `redblue`,
  `regioninside`, and `starfish`; these now have compact MATLAB fixture parity
  in `tests/golden/geometry_core.mat`.
- Chunker preference, storage-growth, data-row allocation/reset, explicit copy,
  `checkadjinfo`, and sorted-field behavior now have compact MATLAB fixture
  parity in `tests/golden/chunker_ops.mat` / `geometry_core.mat`.
- Helmholtz double-gradient FMM selector wiring.
- Stokes traction FMM selector wiring.
- FMM integration across the implemented 2D kernel selector surface, including
  Laplace derived selectors, Helmholtz target-derivative selectors, full
  biharmonic scalar selector wiring, Stokes traction/combined paths, and
  elasticity single/traction/double/alternate-double workflows.
- Remaining `+lege` helpers: `rts`, `rts_stab`, `adapgauss`,
  `bernstein_ellipse`, `polsum`, and `tayl`.
- Adaptive refinement in `chunker.refine` and `chunkerfunc`.
- Section III quadrature parity: `quadggq/buildmattd` sparse special-block
  assembly, direct `adapgausswts` neighbor-block parity, and `quadadap`
  log-kernel self, neighbor, and robust close replacement are now covered by
  compact MATLAB fixtures.
- `chunkermat(..., acceleration="fmm")` matrix-free FMM operators and
  `chunkermatapply` FMM acceleration with sparse special-quadrature
  corrections for singular kernels, including deterministic RHS matvec parity
  against MATLAB forced-FMM output.
- `chunkerinterior` FMM and FLAM classification with direct close-boundary
  correction.
- Advanced RCIP chunkgraph workflows: selected vertices, ignored vertices, and
  global block-kernel subselection for local corner compression.
- Section II kernel/operator parity: MATLAB `@kernel` factory metadata and
  direct evaluations, kernel algebra/interleave, Green helpers, biharmonic
  `bhgreen`-derived selectors, Helmholtz-difference interleave identities, and
  smooth dense operator helper routes.
- Chunkgraph constructor parity now includes the devtools
  `chunkgrphconstructTest.m` legacy incidence versus `edgesendverts` workflow
  with matching curved edge chunkers.
- Chunkgraph basic devtools parity now covers legacy/new graph formats,
  multiply connected, bridge, loop, nested, and adjacent-triangle region
  cases, graph-region id queries, affine/scale/rotate/reflect transforms, and
  per-edge dyadic refinement counts.
- Chunkgraph signed-region devtools parity now covers the manual
  `chunkgrphregionTest.m` nested/disjoint region ordering and edge-side
  region maps through `findregions` and `find_edge_regions`.
- Chunkgraph operator-dimension devtools parity now covers dense assembly of
  edge-by-edge block kernels with variable row/column dimensions from
  `chunkrgrphOpdimTest.m`.
- Chunkgraph refinement parity now includes graph-level selected-edge
  refinement, per-edge split-chunk routing, vertex endpoint balancing,
  MATLAB-style `NaN` closed-edge construction, and `last_len` endpoint-panel
  matching from `chunkgraph_lastlengthTest.m`.
- Laplace and Helmholtz Green-identity target-evaluation parity for the devtools
  `kernelclass`, `chunkerkerneval_greenlap`, and
  `chunkerkerneval_greenhelm`, `chunkerkerneval_gaussid`, and
  `chunkerkernevalmat_greenlap` paths, including Python `forceadap`
  close-target replacement in `chunkerkerneval` and `chunkerkernevalmat`.
- Dense `chunkermat` l2 scaling now has devtools parity for the Helmholtz
  transmission-style matrix relation from `chunkermat_l2scaleTest.m`,
  including MATLAB-style string truth parsing for `opts.l2scale`.
- Data-field parity for converted slices of the devtools `datafieldTest.m`
  workflow, including Hilbert/cotangent source data through dense and PyFLAM
  matrix products plus directional-derivative target data through direct,
  `forceadap`, and PyFLAM target evaluation against a MATLAB-solved density.
- PyFLAM-backed acceleration: `chunkerflam`, `chnk.flam` callback/proxy
  helpers, `ChunkerFLAMMatrix`, `chunkermat`/`chunkermatapply`
  `acceleration="flam"`, FLAM target evaluation/materialization, FLAM
  interior classification, adaptive near-target correction, explicit
  chunker-sequence coercion, smooth interleaved block-kernel matrix and target
  evaluation paths, shape-preserving single-column and multiple-RHS
  application, adjoint application/solve helpers, l2 scaling, and
  source/target point-data callbacks. Square, circular, and rectangular FLAM
  proxy geometry, plus Hilbert/cotangent and target-data directional-derivative
  devtools datafield slices, now also have strict MATLAB fixture parity; a
  Laplace `rskelf` matrix product and solve are compared against MATLAB FLAM on
  a deterministic random RHS.

Deferred implementation:

- Remaining FLAM parity beyond the first PyFLAM-backed pass: strict MATLAB
  devtools FLAM fixtures beyond the converted Green-identity and datafield
  diagnostics, full block-kernel multi-chunker workflows, and larger
  proxy-by-level stress coverage.
- Remaining `chunkerfit` modes beyond the implemented spline/open-line/circle
  paths.

Do not implement:

- The `trapper` family.
- Axisymmetric, quasiperiodic, and flexural kernel families, including
  `axissymhelm2d`, `axissymhelm2ddiff`, `helm2dquas`, most of `flex2d`, and
  matching kernel factories.
- `quadba`.
- The full nonlinear MATLAB smoother/Newton workflow and `+chnk/+intchunk`;
  the lightweight rounded-polygon smoother remains the supported path.
- MATLAB plotting/visualization methods: `plot`, `plot3`, `scatter`, `quiver`,
  and `plot_regions`.

## `tests/test_arcparam.py`

`test_arcparam_evaluates_original_chunk_nodes` checks arclength
parameterization on a merged geometry made from two circles. The equation is
`s = integral |dr/dt| dt`, with the second component's arclength shifted by the
first component length. The method is `arcparam.init` followed by
`arcparam.eval`, using Legendre-panel interpolation in arclength. Ground truth
is the original chunker node coordinates, explicit arclength first and second
derivatives transformed from the source parameterization, unit-speed
derivatives `|dr/ds| = 1`, and orthogonality `dr/ds dot d2r/ds2 = 0`.

`test_arcparam_derivatives_are_consistent_on_circle` checks the same
arclength evaluator away from the original nodes on a radius-2 circle. The
circle equations imply `r dot dr/ds = 0`, `|dr/ds| = 1`, and
`dr/ds dot d2r/ds2 = 0`. The method is spectral interpolation from the chunker
arclength data. Ground truth is the analytic circle position, tangent,
curvature vector, and arclength differential identities.

`test_arcresample_makes_panel_speed_constant` checks that `Chunker.arcresample`
reparameterizes panels by arclength. The invariant is that each panel has
constant speed density `chunklen / 2` on the reference interval `[-1,1]`.
The method is arclength resampling of a Legendre chunker. Ground truth is
preservation of area and total length, nonnegative reported error, and constant
panel speed after resampling, plus preserved circle radius, tangent/radius
orthogonality, normals, and curvature.

## `tests/test_biharm2d.py`

`test_biharmonic_green_gradient_matches_finite_difference` checks the
biharmonic Green's function `G = r^2 log(r) / (8 pi)`, implemented as
`r2 log(r2) / (16 pi)`. The method evaluates `green` and compares the target
`x` derivative to a centered finite difference with step `1e-6`. Ground truth
is the finite-difference gradient and the identity `lap(G) = G_xx + G_yy`.

`test_biharmonic_kernel_selectors_and_factory_shapes` checks the selector
surface for biharmonic single layer, double layer, target-normal derivative,
single-layer gradient, and Hessian kernels. The equation content is the same
biharmonic Green's function and its target derivatives, projected onto source
or target normals where needed. The method is direct point-kernel evaluation
through `biharm2d.kern` and the generic `kernel("biharm", ...)` factory.
Ground truth is `biharm2d.green` value, gradient, and Hessian data projected
into the expected selector blocks, plus `opdims` metadata.

`test_biharmonic_layer_evaluation_uses_special_quadrature` checks the
self/near singular dispatch for a biharmonic single-layer potential on a circle.
The kernel is marked as logarithmic, so `chunkerkerneval` on a chunker target
uses GGQ special quadrature instead of naive coincident-node quadrature. Ground
truth is the analytic unit-density circle identity `S[1](x) = |x|^2/4` at an
interior target.

## `tests/test_chunker.py`

`test_chunker_constructor_defaults_and_validation` checks the basic `Chunker`
constructor and `chunkerpref` plumbing. There is no PDE equation here; the
invariants are `k`, dimension, chunk count, and point count initialization. The
method is direct construction plus an invalid low-order constructor call.
Ground truth is the expected default state and a `ValueError` when `k` is too
small.

`test_addchunk_resizes_storage_and_exposes_live_slices` checks allocation and
storage-view behavior. The invariant is that adding chunks updates `nch`,
resizes backing arrays, and exposes live slices so assigning through the public
geometry properties changes the matching backing stores. The method is direct
chunk storage manipulation. Ground truth is array shape and equality of the
public slices with the backing coordinate, derivative, normal, and weight
storage.

`test_resize_chunkends_min_max_and_cleardata_helpers` checks helper methods on
a one-panel circle chunker. The endpoint equations give both endpoints at
`(1,0)` for a closed panel running around the circle, with unit endpoint
tangents after normalization. The method calls `resize`, `chunkends`, `min`,
`max`, `makedatarows`, and `cleardata`. Ground truth is analytic endpoint
geometry, NumPy min/max over stored nodes, and empty data storage after
clearing.

`test_circle_weights_normals_tangents_area_and_length` checks geometric
quantities on a circle. The equations are `length = 2 pi`, `area = pi`,
`|r'(t)| = pi` for the reference-panel map used in the helper, and signed
curvature `1`. The method is the chunker's native Legendre weights, normals,
tangents, arclength density, curvature, area, and chunk-length routines. Ground
truth is exact circle geometry and tangent-normal orthogonality.

`test_translation_and_scaling_match_matlab_style_operations` checks chunker
operator overloads for translation and scalar scaling. The equations are
`center(r + a) = center(r) + a`, `area(alpha r) = alpha^2 area(r)`, and
`length(alpha r) = alpha length(r)`. The method uses `__radd__` and scalar
`__mul__`. Ground truth is the weighted center shift, exact translated and
scaled coordinate/derivative/normal/weight storage, and exact area and length
scaling.

`test_matrix_transform_updates_derivatives_normals_and_weights` checks affine
matrix transforms. The equations are `r_new = A r`, `d_new = A d`,
`d2_new = A d2`, `n_new = (dy,-dx)/|d_new|`, `w_new = |d_new| w_leg`, and
`area_new = det(A) area`. The method uses matrix multiplication `A @ chnkr`,
recomputing normals and weights. Ground truth is explicit matrix application,
recomputed normals and weights, determinant-based area scaling, and
`TypeError` for matrix right-multiplication with `chnkr * A`.

`test_rotate_and_reflect_match_matlab_transform_formulas` checks rotation and
reflection helpers with source and destination centers. The equations are
`r_new = R(theta)(r-r0)+r1` for rotation and the standard reflection matrix
with angle `angle` for reflection. The method is `rotate` and `reflect`, which
transform coordinates, derivatives, and normals. Ground truth is explicit
matrix application to position, first and second derivatives, normals, and
unchanged quadrature weights.

`test_chunker_spectral_helpers_on_circle` checks the spectral expansion,
arclength, differentiation, and differentiation-matrix helpers. The equation is
`s = pi(t+1)` on the one-panel circle helper and `d/ds sin(s) = cos(s)`.
The method uses `exps`, `arclengthfun`, `arclengthder`, and `diffmat`.
Ground truth is reconstruction of the position/derivative/second-derivative
Legendre coefficients, analytic sine differentiation, and matching
matrix-based differentiation.

`test_intmat_integrates_in_chunk_order` checks cumulative arclength integration
after refining a circle chunker. The equation is `integral_0^s 1 ds = s`.
The method builds `intmat` and applies it to an all-ones vector in Fortran
chunk order. Ground truth is `arclengthfun()` flattened in the same order.

`test_onesmat_and_normonesmat_shapes` checks block matrix helpers used by
scalar and vector-valued operators. There is no PDE equation; the invariant is
that scalar ones are weighted by source quadrature weights and normal-vector
ones repeat the outer products of target/source normals with those weights. The
method directly calls `onesmat` and `normonesmat`. Ground truth is explicit
matrix entries assembled from `wts` and `n`.

`test_centroids_and_adjacency_info` checks centroids, sort metadata, component
count, closedness, and adjacency validity on a two-panel refined circle. The
method is `centroids`, `sortinfo`, and `checkadjinfo`. Ground truth is one
closed component, two chunks, identity chunk ordering, adjacency equal to the
stored adjacency, no adjacency errors, and centroid coordinates equal to
weighted panel means.

`test_upsample_preserves_circle_geometry_and_density_values` checks spectral
upsampling from `k=16` to `k=24`. The scalar data equation is
`sigma(t) = 1 + t - 2 t^3`, a polynomial exactly representable by the panel
basis. The method is `Chunker.upsample` for geometry and attached density.
Ground truth is analytic circle positions, first and second derivatives,
normals, quadrature weights, preserved area, and exact polynomial values at
the upsampled Legendre nodes.

`test_refine_oversamples_by_splitting_chunks` checks chunk refinement by panel
splitting. The invariant is that one circle panel split with `nover=1` becomes
two panels with closed adjacency, while area and total length are preserved.
The method is `Chunker.refine`. Ground truth is expected adjacency, analytic
half-circle positions, derivatives, normals, and weights on both split panels,
unchanged area, and unchanged summed chunk lengths.

`test_refine_enforces_arc_length_level_restriction` checks adaptive refinement
of an existing uneven chunker. The method builds an open line with two short
panels followed by a long panel, calls `Chunker.refine` with `lvlr="a"`, and
verifies every adjacent chunk-length ratio is within the requested factor.
Ground truth also includes straight-line coordinates, derivatives, normals,
second derivatives, and weights after refinement.

`test_chunkerpoints_builds_from_nodes_and_optional_derivatives` checks
reconstruction of a chunker from node positions alone and from explicit
derivative fields. The equation source is the circle helper, where spectral
differentiation can recover `d` and `d2` from positions. The method is
`chunkerpoints`. Ground truth is equality to the original circle fields when
derivatives are inferred, and exact preservation of explicitly supplied
`2*d` and `3*d2`.

`test_datares_flags_high_order_data_coefficients` checks data-resolution
detection in Legendre coefficient space. The low-order row `1+t^2` should be
resolved, while a row equal to the highest Legendre mode should not. The method
uses `lege.exps` to get the expansion matrix and `Chunker.datares` with a
tolerance. Ground truth is the expected boolean flags for all data rows and for
an explicitly selected row.

`test_merge_combines_chunkers_and_pads_data_rows` checks merging open
chunkers with different data dimensions. The invariant is that geometry and
data are concatenated chunk by chunk, adjacency for unrelated open components
is free-ended, and missing data rows are zero-padded. The method is `merge`.
Ground truth is original coordinate, derivative, normal, and weight storage in
each output chunk, datadim equal to the maximum input datadim, and the expected
padded data arrays.

## `tests/test_chunkerfit.py`

`test_chunkerfit_open_line_with_split_points` checks spline fitting through
collinear points with splits at the input points. The geometry equation is an
open line from `x=0` to `x=3`, total length `3`, with `y=0`. The method is
`chunkerfit(..., splitatpoints=True)` with `ifclosed=False`. Ground truth is
three panels, free-ended adjacency, total length `3`, and exact line-panel
positions, derivatives, normals, and weights.

`test_chunkerfit_closed_circle_spline_area` checks closed spline fitting of
16 samples from the unit circle. The equations are circle area `pi`, perimeter
`2 pi`, unit node radius, and radial/tangent orthogonality. The method is
`chunkerfit` with a closed periodic fit and split points. Ground truth is one
panel per input interval, closed adjacency, circle geometry, and length/area
within spline tolerances.

`test_chunkerfit_rejects_unsupported_methods` checks API validation. There is
no numerical equation; the invariant is that unsupported fit methods are
rejected instead of silently using a fallback. The method requests
`method="linear"`. Ground truth is a `ValueError`.

## `tests/test_chunkerfunc.py`

`test_chunkerfunc_builds_closed_circle_with_area_and_adjacency` checks
construction from an analytic radius-2.5 circle function. The equations are
`area = pi R^2`, `length = 2 pi R`, and four uniform parameter intervals of
length `pi/2`. The method is `chunkerfunc` with `nchmin=4` and Legendre order
16. Ground truth is four chunks, closed adjacency, expected interval endpoints,
exact circle area and length, and pointwise radius-scaled positions,
derivatives, normals, and quadrature weights on every panel.

`test_chunkerfunc_open_curve_marks_free_ends` checks open-curve construction
for a line segment from `(0,0)` to `(2,0)`. The equation is total length `2`.
The method is `chunkerfunc` on `curves.linefunc` with `ifclosed=False` and two
chunks. Ground truth is free-ended adjacency at both ends, the first interval
`[0, 0.5]`, total length `2`, and exact straight-line positions,
derivatives, normals, and quadrature weights on each panel.

`test_chunkerfuncuni_builds_requested_uniform_panel_count` checks uniform
panel construction independent of adaptivity. The equation is area
`pi * 1.5^2` for a radius-1.5 circle. The method is `chunkerfuncuni` with six
panels and Legendre order 10. Ground truth is exactly six chunks, `k=10`, and
the analytic area plus pointwise uniform-panel circle positions, derivatives,
normals, and weights.

`test_chunkerfunc_can_spectrally_differentiate_position_only_curve` checks that
the constructor can infer derivatives when a curve callback returns only
positions. The equations are the unit-circle identities for `r`, `d`, `d2`,
normals, weights, and area. The method is spectral differentiation of position
data inside `chunkerfunc`. Ground truth is analytic circle geometry after
derivative inference.

`test_chunkerfunc_adaptively_refines_unresolved_curve` checks adaptive
parameter-interval refinement. The method constructs a high-frequency open
curve once with `ifrefine=False` and once with adaptive refinement enabled.
Ground truth is that the unresolved one-panel curve is split into multiple
contiguous panels preserving interval coverage from `0` to `1`, and that the
refined arclength matches an independent high-order Gauss-Legendre reference
for `sqrt(1 + y'(t)^2)`.

`test_basic_curve_helpers_match_expected_derivatives` checks the canned
`curves.fsine` helper. The equations are `x(t)=t`, `x'(t)=1`, `x''(t)=0`, and
`y(t)=amp sin(freq t + phase)` plus its first two derivatives. The method
directly evaluates the helper. Ground truth is the known coordinate and
derivative formulas.

## `tests/test_chunkerpoly.py`

`test_chunkerpoly_closed_square_area_length_and_adjacency` checks a closed,
unrounded square polygon. The equations are area `1` and perimeter `4`. The
method is `chunkerpoly` with one straight Legendre panel per edge. Ground truth
is four chunks, closed cyclic adjacency, exact area, exact summed length, and
straight-edge positions, derivatives, normals, and weights.

`test_chunkerpoly_open_polyline_and_edge_data` checks an open two-edge
polyline with per-edge data. The equation is total length `2 + 3 = 5`. The
method is `chunkerpoly(..., ifclosed=False)` with `edgevals`. Ground truth is
open adjacency, two data rows, constant data on each corresponding edge, and
the exact total length plus straight-edge positions, derivatives, normals, and
weights.

`test_chunkerpoly_rounded_builds_trimmed_edges_and_corner_panels` checks
rounded-polygon construction for a square. The method trims the straight edges
and inserts rounded corner panels of width `0.1`. The geometric invariant is
that the rounded shape remains valid, has positive panel lengths, and has area
between `0.9` and the original square area `1`. Ground truth is eight chunks,
valid adjacency, area/length inequalities, trimmed edge node coordinates, and
rounded-corner coordinate/derivative/second-derivative values.

`test_chunkerpoly_rounded_open_polyline_and_edge_data` checks rounded
construction for an open L-shaped polyline with data interpolation through the
rounded corner. The method uses widths `[0, 0.2, 0]` and scalar edge values
`2` and `4`. Ground truth is three chunks, free-ended adjacency, open rounded
corner geometry, preserved endpoint-edge data, and middle-panel data bounded
between the adjacent edge values.

`test_reverse_and_move_preserve_expected_geometry` checks orientation reversal
and rigid/scale movement on a square. The equations are `area(reverse) = -area`
and `area(scale * r) = scale^2 area`, with scale `3`. The method is
`reverse` and `move` with translation, rotation, and scaling. Ground truth is
reversed coordinate/derivative/normal/weight storage, area `-1` after
reversal, transformed coordinate/derivative/normal/weight storage, and area
`9` after scaling by `3`.

## `tests/test_chunkgraph.py`

`test_chunkgraph_constructs_edges_and_vertex_incidence` checks conversion of a
square graph from vertices and directed edge endpoints. The invariant is the
edge-to-vertex incidence matrix with `-1` at the start and `+1` at the end.
The method is `chunkgraph` construction plus source-info assembly. Ground truth
is four edge chunkers, the explicit `v2emat`, total point count equal to the
sum over edges, straight-edge node/derivative/normal/weight formulas,
source-info flattening, exact vertex-edge structure, the signed unbounded and
bounded square region loops, and the edge-side region map.

`test_chunkgraph_accepts_incidence_matrix_edges` checks the alternate graph
constructor format where edges are supplied as an incidence matrix. The method
converts incidence columns back to endpoint pairs. Ground truth is the same
square endpoint matrix `[[0,1,2,3],[1,2,3,0]]`, the original incidence matrix,
exact vertex-edge structure, and the signed unbounded and bounded square
region loops.

`test_chunkgraph_slice_and_edgeids_match_selected_edges` checks subgraph
slicing and global point-index selection. The method is `slicegraph([0,1])`
and `edgeids([0,1])`. Ground truth is a two-edge subgraph with the expected
endpoint matrix, exact selected global point ids, subgraph coordinates,
derivatives, and weights matching the selected edge chunkers.

`test_chunkgraph_region_ids_survive_translation` checks region classification
on a square graph and after translation. The geometric invariant is that
inside/outside region ids are translation-invariant when targets translate by
the same vector. The method is `chunkgraphinregion` before and after `cg + a`.
Ground truth is region ids `[2, 1]` for the sample inside/outside targets in
both coordinate systems.

`test_chunkgraph_works_with_dense_operator_helpers` checks that chunkgraphs can
be passed into dense operator assembly/evaluation APIs. The kernel is the
smooth polynomial `1 + dx^2 + dy^2`. The method is `chunkermat` and
`chunkerkerneval` on a graph with a sinusoidal density. Ground truth is an
explicit raw-kernel matrix weighted by graph quadrature weights and the
corresponding manual matrix-vector target evaluation.

`test_tochunkgraph_preserves_closed_and_open_components` checks conversion from
ordinary chunkers to graph form. The invariant is that a closed component
becomes a loop edge whose start and end vertex are the same, while an open line
becomes an edge between distinct vertices. The method is `tochunkgraph`.
Ground truth is the loop endpoint condition for the closed square and
`[[0],[1]]` endpoints for the open line, plus preservation of merged graph
coordinates, weights, edge chunker storage, point counts, and vertex
coordinates.

## `tests/test_domain.py`

`test_ellipse_and_starfish_helpers_match_expected_formulas` checks top-level
curve helpers. The method evaluates `ellipse` and `starfish`; ground truth is
the analytic ellipse formula and equality with the existing
`chnk.curves.starfish` implementation for a shifted/scaled starfish.

`test_checkcurveparam_validates_dimension_and_output_shapes` checks the
MATLAB-style curve callback validator. The method calls `checkcurveparam` on a
valid ellipse callback and on callbacks with wrong sample count or inconsistent
output dimension. Ground truth is the returned ambient dimension and
`ValueError` for invalid outputs.

`test_nonflatinterface_derivatives_and_redblue_colormap` checks the perturbed
interface helper and the red-white-blue colormap. The method compares the
interface against the closed-form y-coordinate plus first and second
derivatives, finite-differences the y-coordinate, and samples a five-color
map. Ground truth is the analytic formulas and blue/white/red endpoint colors.

`test_hypoct_uni_builds_zero_based_uniform_tree` checks the top-level
hyperoctree helper on four quadrant points. The method builds `hypoct_uni`
with an explicit square extent. Ground truth is zero-based tree indexing,
two-level structure, exact child centers, one point per child, and exact
sibling neighbor connectivity.

`test_chunkgraph_region_helpers_count_inside_and_merge_nested_regions` checks
top-level region helpers for a nested square graph. The method uses
`pointinregion`, `regioninside`, and `mergeregions`. Ground truth is
inside/outside loop counts, detection that the inner region lies inside the
outer region, and a merged region list containing both loops.

## `tests/test_devtools_parity.py`

`test_absconvgauss_devtools_outputs_match_matlab` checks
`spcl.absconvgauss`, a smoothed absolute-value convolution used by the
smoother/devtools track. The method evaluates value, first derivative, and
second derivative in Python. Ground truth is the MATLAB `devtools_easy.mat`
fixture, including saved gradient-check error thresholds.

`test_legeexpsunit_devtools_outputs_match_matlab` checks Legendre nodes,
weights, value/coefficient transforms, derivative matrix, integration matrix,
and coefficient-space integration against MATLAB. The equations include
`d/dx sin(x)` and antiderivative evaluation through `lege.intpol` and
`lege.exev`. The method is the Python `lege` package. Ground truth is the
MATLAB devtools fixture for every array.

`test_arclengthfun_single_component_devtools_output_matches_matlab` checks
`Chunker.arclengthfun` and `chunklen` for a single MATLAB-saved chunker. The
equation is cumulative arclength along one connected component. The method is
reconstructing a Python `Chunker` from fixture fields and running the Python
helpers. Ground truth is the MATLAB arclength vector, expected parameter-space
truth, and MATLAB chunk lengths.

`test_arclengthfun_merged_components_devtools_output_matches_matlab` checks
the same cumulative arclength calculation on a merged two-component chunker.
Ground truth is MATLAB's merged-component fixture, including the second
component scaled by `1.1`.

`test_chunkerarcparam_devtools_outputs_match_matlab` checks arc-length
parameterization against MATLAB. The method reconstructs the saved merged
chunker, runs `arcparam.init/eval`, compares original-node and sample-point
evaluations, verifies derivative residual diagnostics, and checks
`arcresample` area/length preservation plus unit-speed panel output. Ground
truth is `devtools_easy.mat`.

`test_chunker_diffintmat_devtools_outputs_match_matlab` checks spectral
differentiation and integration matrices on saved ellipse and circle chunkers.
The equations are unit arclength tangents after differentiation and integration
as the inverse of differentiation up to an additive constant. The method is
`diffmat` and `intmat` applied to saved geometry. Ground truth is MATLAB's
saved matrices, differentiated coordinate vectors, integrated coordinate
vectors, and residual arrays.

`test_chunker_nearest_devtools_output_matches_matlab` checks closest-point
search on many targets around a circle. The equation compares target polar
angle to nearest boundary polar angle, with wrapped angle error. The method is
`Chunker.nearest` on a chunker reconstructed from MATLAB fields. Ground truth
is MATLAB's nearest point, derivative, second derivative, distance, local
parameter, chunk index, and angle-error fixture.

`test_chunkerfit_devtools_outputs_match_matlab` checks the saved
`chunkerfitTest.m` closed and open curve fits. The method rebuilds the same
random-mode sample points in Python, fits closed and open chunkers, and
compares geometry fields, normals, weights, adjacency, chunk lengths, and area
against the MATLAB devtools fixture.

`test_flagself_devtools_output_matches_matlab` checks source-target duplicate
pair detection. There is no PDE equation; the invariant is exact coordinate
coincidence within tolerance. The method is `flagself` and a sorted-pair
comparison. Ground truth is the MATLAB pair list, adjusted from 1-based to
0-based indexing, plus the fixture's zero error count.

`test_flagnear_devtools_output_matches_matlab` checks chunk-near target flags
on a saved chunker. The equation is a per-target/per-chunk distance threshold
controlled by `fac`. The method is `flagnear` using chunker geometry and
sortinfo-style spatial filtering. Ground truth is MATLAB's near-flag matrix
and its brute-force verification matrix.

`test_flagrect_devtools_output_matches_matlab` checks rectangle-based near
flagging in both direct point-list and tensor-grid modes. The method is
`flagnear_rectangle` and `flagnear_rectangle_grid`. Ground truth is MATLAB's
direct and grid flag matrices, with the fixture asserting no mismatch between
the two routes.

`test_helm2d_green_devtools_output_matches_matlab` checks the 2D Helmholtz
Green's function `i H_0^(1)(k r) / 4`, its gradient, and Hessian. The method is
direct `helm2d.green` evaluation. Ground truth is MATLAB's saved value,
gradient, Hessian, and finite-difference error thresholds.

`test_kernelop_devtools_outputs_match_matlab` checks kernel algebra and
interleaving with deterministic source/target point-info. The equations are
linear algebra identities: scalar multiply/divide, negation, addition,
subtraction, conjugation, and block interleave preserve the underlying kernel
values. The method uses Laplace single layer, Helmholtz double layer, and the
generic `kernel` wrapper. Ground truth is the MATLAB fixture for each algebraic
combination.

`test_kernderinterleave_devtools_outputs_match_matlab` checks the exact
`KernDerInterleaveTest.m` source/target/coefficient cases. The equations are
combined-kernel assembly, transmission blocks, `all` interleaving, gradients,
and target-normal derivatives as gradient-dot-normal contractions for Laplace,
Helmholtz, and Helmholtz-difference kernels. Ground truth is the MATLAB fixture.

`test_chunkerfunc_devtools_outputs_match_matlab` checks adaptive
`chunkerfunc` cases from the MATLAB devtools test: starfish construction,
`nout=3`, random Fourier-mode construction, reversal, circle area, refinement,
and open/closed endpoint warning behavior. Ground truth is MATLAB-saved
chunker fields plus adjacency, area, and warning diagnostics.

`test_chunkerpoly_devtools_outputs_match_matlab` checks the barbell-like
polygon cases from `chunkerpolyTest.m`. The method builds Python rounded,
true-polygon, and open polygon chunkers from the fixture inputs. Ground truth
is MATLAB status/diagnostics, true-polygon area and length, data dimensions,
rounded chunk count and positive lengths, and open lightweight adjacency and
length invariants.

`test_smoother_devtools_output_matches_matlab_thresholds` checks the
lightweight smoother path against the MATLAB smoother diagnostic fixture. The
method verifies the MATLAB and Python error thresholds and then checks the
supported Python rounded-polygon chunker for zero reported errors, `2*nv`
chunks, clean adjacency, unit normals, and positive chunk lengths.

`test_chunkgrphconstruct_devtools_outputs_match_matlab` checks the constructor
equivalence from `chunkgrphconstructTest.m`. The method builds the pentagonal
graph with both MATLAB's legacy incidence matrix and the newer `edgesendverts`
format, using the same sine-arc edge callbacks. Ground truth is MATLAB's
balanced vertices, incidence matrices, endpoint indices, and first-edge
chunker geometry.

`test_chunkgraph_basic_devtools_outputs_match_matlab` checks the graph
constructor, region, transform, region-query, and dyadic-refinement cases from
`chunkgraph_basicTest.m`. The method compares MATLAB legacy/new incidence
matrices, region counts for multiply connected/bridge/loop/nested graphs,
adjacent and nested point-region ids, transformed graph ids after affine,
scaling, rotation, and reflection operations, and per-edge refinement counts.

`test_chunkgrphregion_devtools_outputs_match_matlab` checks the full
signed-region workflow from `chunkgrphregionTest.m`. The method reconstructs
the nested/disjoint graph with a closed starfish edge, converts MATLAB's
one-based signed region loops to Python's zero-based signed convention, and
compares region ordering plus `find_edge_regions` side maps.

`test_chunkrgrph_opdim_devtools_outputs_match_matlab` checks the chunkgraph
block-kernel operator-dimension workflow from `chunkrgrphOpdimTest.m`. The
method assembles a dense edge-by-edge Helmholtz transmission block matrix with
mixed one- and two-component row/column dimensions. Ground truth is MATLAB's
edge point counts, global matrix shape, and two representative off-diagonal
blocks.

`test_chunkgraph_lastlength_devtools_outputs_match_matlab` checks graph-level
refinement behavior from `chunkgraph_lastlengthTest.m`. The method compares
MATLAB and Python chunk counts for selected-edge refinement and per-edge
split-chunk routing, verifies `NaN` closed-edge construction maps to a closed
Python edge, and compares `last_len` endpoint-panel arclengths and vertex
degrees against the MATLAB fixture.

`test_slicegraph_devtools_outputs_match_matlab` checks the concentric-square
`slicegraph` workflow. The method compares sliced geometry and edge id
ordering against MATLAB, and verifies that both MATLAB and Python preserve the
inner-slice/full-submatrix relation. Direct matrix values remain partial
because current Python graph double-layer self blocks produce NaNs where the
MATLAB fixture stores finite special entries.

`test_chunkermat_quadadap_devtools_outputs_match_matlab` checks the starfish
adaptive-neighbor matrix comparison from `chunkermat_quadadapTest.m`. The
method compares MATLAB and Python Helmholtz double-layer GGQ and adaptive
matrices and verifies both routes agree to the devtools Frobenius threshold.

`test_chunkermat_l2scale_devtools_outputs_match_matlab` checks the
transmission-style Helmholtz l2-scaling relation from
`chunkermat_l2scaleTest.m`. The method builds the manual
`diag(sqrt(w))*A*diag(1/sqrt(w))` block matrix and compares it to Python
`chunkermat(..., {"l2scale": "true"})`, then verifies both solve the same RHS
to the MATLAB diagnostic threshold.

`test_adapgausswts_devtools_neighbor_block_matches_matlab` checks the direct
adaptive weight routine from `adapgausswtsTest.m`. The method reconstructs the
saved starfish chunker, Helmholtz double-layer kernel, source chunk, and
neighbor target chunk, then compares Python `quadadap.adapgausswts` output,
recursion metadata, and GGQ reference block against MATLAB.

`test_datafield_devtools_target_data_flam_matches_matlab` checks the
directional-derivative single-layer target-data slice from MATLAB
`datafieldTest.m`. The method uses the saved MATLAB boundary density, evaluates
the custom target-data kernel through direct, `forceadap`, and
`acceleration="flam"` paths, and compares against MATLAB direct/adaptive/FLAM
outputs plus the saved single-layer-gradient directional truth.

`test_datafield_devtools_hilbert_data_flam_matches_matlab` checks the
Hilbert/cotangent source-data slice from the same MATLAB datafield test. The
method rebuilds the saved data-bearing starfish chunker, evaluates the custom
PV cotangent kernel through dense special quadrature and PyFLAM application, and
compares against MATLAB's dense and FLAM products.

`test_flam_proxy_geometry_helpers_match_matlab_fixture` checks deterministic
FLAM helper geometry against the MATLAB fixture. The method compares
`proxy_square_pts(64)` points, tangents, weights, and inside predicate samples,
`proxy_circ_pts(16)` points, normals, and weights, and
`proxy_rect_pts([2, 3], [4, 6])` points, tangents, and weights. It also
compares deterministic Laplace `nproxy_square` proxy-order selection and the
0-based `kernbyindex`/`kernbyindexr` callback entries against MATLAB's 1-based
fixture output, including sparse overwrite precedence. The same fixture also
checks `proxyfun` and `proxyfunr` proxy matrices plus filtered neighbor lists.

`test_stokes_dtrac_devtools_output_matches_matlab` checks the Stokes double
layer traction relation against MATLAB. The continuum equation is the traction
stress identity `t = -p n + mu (grad u + grad u^T) n`, reconstructed from
`dgrad` and `dpres`. The method evaluates `dtrac`, `dgrad`, and `dpres` kernel
blocks through the generic `kernel` wrapper and contracts with saved
strengths. Ground truth is MATLAB's `Kt`, `Kg`, `Kp`, reconstructed traction,
and residual norm below `1e-13`.

`test_kernelclass_devtools_green_identity_matches_matlab` checks the Laplace
Green-identity portion of MATLAB `kernelclassTest.m` plus NaN-kernel
propagation. The method reconstructs the saved starfish chunker, recomputes
boundary `u` and normal-derivative densities from exterior point sources, then
uses `chunkerkerneval(..., forceadap=True)` for close-corrected target layer
evaluation. Ground truth is MATLAB's boundary data, target truth, layer
potentials, Green-identity residual, and NaN-kernel diagnostics.

`test_chunkerkerneval_greenlap_devtools_outputs_match_matlab` checks the
Laplace Green-identity target-evaluation workflow from
`chunkerkerneval_greenlapTest.m`. The method compares saved point-source
fields, boundary densities, and close-corrected single/double-layer target
evaluations through `forceadap`. Ground truth is MATLAB's direct outputs and
diagnostic FMM equality plus saved MATLAB FLAM outputs. Python FLAM
force-adaptive target evaluation is checked against both direct and MATLAB
FLAM fixture values.

`test_chunkerkernevalmat_greenlap_devtools_outputs_match_matlab` checks the
matrix form of the same Laplace Green identity. The method builds target
evaluation matrices with `chunkerkernevalmat(..., forceadap=True)`, applies
them to saved boundary densities, and verifies the reconstructed target field.
Ground truth is MATLAB's single-layer matrix, adaptive double-layer matrix,
applied layer potentials, and relative identity residual.

`test_chunkerkerneval_greenhelm_devtools_outputs_match_matlab` checks the
Helmholtz Green-identity workflow from `chunkerkerneval_greenhelmTest.m`. The
method uses the saved complex wave number, source strengths, boundary
densities, and targets, then evaluates single and double Helmholtz layers with
`chunkerkerneval(..., forceadap=True)`. Ground truth is MATLAB's layer
potentials and relative identity residual.

`test_chunkerkerneval_gaussid_devtools_outputs_match_matlab` checks the
adaptive assertion from `chunkerkerneval_gaussidTest.m`. The method uses the
saved starfish geometry, full `40 x 40` target grid, and unit double-layer
density, then compares Python `chunkerkerneval(..., forceadap=True)` values to
MATLAB and verifies the Gauss identity values classify as either `0` outside
or `-1` inside. Ground truth is MATLAB's target values, identity residuals, and
inside/outside labels.

## `tests/test_easy_parity_stress.py`

`test_point_kernels_stress_combined_selectors_and_green_gradients` hardens the
previous small point-kernel parity surface. It uses multiple non-axis-aligned
source and target points with varied source/target normals and tangents. The
method checks Laplace combined, combined-prime, and combined-gradient selectors
against their component kernels, checks Helmholtz combined and combined-prime
selectors at a complex wavenumber, and validates Laplace/Helmholtz Green
gradients by centered finite differences.

`test_dense_native_operator_stress_on_wobbly_curve_matches_manual_weighting`
checks dense/native operator assembly on a noncircular wobbly curve. The method
uses a custom two-output smooth kernel, off-boundary targets, and a nonconstant
density. Ground truth is the raw kernel matrix multiplied by Legendre source
weights, plus a direct manual target-evaluation matrix-vector product.

`test_quadggq_stress_noncircle_complex_special_blocks_and_robust_close_eval`
checks Section III special quadrature on harder geometry. The method compares
complex Helmholtz log-GGQ self and neighbor topological blocks on a wobbly
curve with the full special matrix, verifies far blocks are zero in the
topological matrix, checks ignored-source behavior, and confirms robust close
replacement changes the Laplace log matrix for nearby wobbly curves while
remaining finite.

`test_schurbana_stress_matches_independent_block_update` checks RCIP
Schur-Banachiewicz compression with a larger rectangular block setup. Ground
truth is an independently assembled block-update formula matching the public
algorithm, and the test also verifies the input matrix is not mutated.

`test_chunkgraph_rcip_stress_nonorthogonal_vertex_and_global_blocks` checks
advanced RCIP chunkgraph behavior on a nonorthogonal closed graph. The method
uses a global block-kernel array, selected vertices, ignored vertices,
recursive compression, and `rhohatInterp`. Ground truth is correct vertex
filtering, incident-edge subselection, block identity preservation, finite
nonidentity compression, and interpolation output shapes.

`test_interleaved_fmm_stress_matches_direct_on_wobbly_curve` compares FMM and
dense direct target evaluation for an interleaved 2x2 kernel block mixing
Helmholtz and Laplace single/double layers. The source geometry is a wobbly
curve and the density is complex and vector-valued. Ground truth is the dense
direct evaluation at off-boundary targets.

`test_smoother_stress_returns_valid_rounded_asymmetric_polygon` checks the
supported lightweight smoother workflow on an asymmetric polygon with varying
corner widths. Ground truth is valid rounded chunker structure: chunk count,
zero reported error, finite geometry, positive chunk lengths, unit normals,
clean adjacency, and positive area below the original polygon area.

## `tests/test_elast2d.py`

`test_elasticity_kernel_shapes_and_factory` checks the 2D linear elasticity
kernel selector and factory metadata. The equations are the Kelvin fundamental
solution and its traction/gradient/double-layer variants for Lame parameters
`lam=1.5`, `mu=2.1`. The method is direct point-kernel evaluation plus
`kernel("elast", ...)`. Ground truth is explicit Kelvin, double-layer, and
alternate-double formulas, finite-difference single-layer gradients,
stress-traction contractions for `strac`/`dalttrac`, and operator-dimension
metadata for vector-valued densities and outputs.

`test_elasticity_single_layer_is_symmetric_in_components` checks symmetry of
the Kelvin single-layer matrix block. The equation is the symmetric displacement
Green's tensor `G_ij = G_ji`. The method evaluates a one-source, one-target
single-layer matrix. Ground truth is equality of the off-diagonal entries.

`test_elasticity_single_gradient_matches_target_finite_difference` checks
target derivatives of the elasticity single layer. The method compares
`elast2d.kern(..., "sgrad")` to one-sided finite differences of
`elast2d.kern(..., "s")` in target `x` and `y`. Ground truth is the finite
difference gradient block with `rtol=1e-5`.

`test_elasticity_dalt_traction_is_gradient_traction` checks that alternate
double-layer traction equals the stress tensor built from the alternate
double-layer gradient. The equation is
`t = lam n div(u) + mu(grad u + grad u^T)n`, expanded componentwise. The method
evaluates `daltgrad`, contracts with a two-component density, then reconstructs
traction and compares to `dalttrac`. Ground truth is the stress formula.

## `tests/test_flam.py`

`test_acceleration_option_uses_single_key_without_boolean_aliases` checks that
the canonical `opts["acceleration"]` key is the only acceleration selector. The
method verifies MATLAB-style boolean aliases such as `{"flam": True}` and
`{"fmm": True}` leave `chunkermat` on the dense path, while
`{"acceleration": "flam"}` returns a `ChunkerFLAMMatrix` and invalid
acceleration strings raise `ValueError`.

`test_flam_kernbyindex_matches_dense_and_sparse_overwrites` checks 0-based
FLAM matrix callbacks. The method requests selected row/column DOFs from
`chnk.flam.kernbyindex`, compares them to the dense weighted matrix, and
verifies sparse special-quadrature entries overwrite smooth entries.

`test_flam_accepts_explicit_chunker_sequences` checks scalar FLAM behavior for
explicit sequences of chunkers. The method compares `chnk.flam.kernbyindex`,
`chunkermat(..., {"acceleration": "flam"})`, and
`chunkermatapply(..., {"acceleration": "flam"})` on a two-chunker list against
the same dense operations on `merge(chunkers)`.

`test_flam_accepts_vector_opdim_chunker_sequences` checks that explicit
chunker sequences and vector operator dimensions compose. The method builds a
two-component smooth kernel over a two-chunker list, applies both
`chunkermat(..., {"acceleration": "flam"})` and
`chunkermatapply(..., {"acceleration": "flam"})`, and compares against the
dense matrix on `merge(chunkers)`.

`test_flam_kernbyindexr_matches_dense_and_sparse_overwrites` checks the
rectangular target/source FLAM callback. The method selects off-boundary target
rows and boundary source columns, compares to `chunkerkernevalmat`, and
verifies sparse target-evaluation entries have overwrite precedence.

`test_flam_proxy_square_geometry_and_proxyfun_shapes` checks proxy helper
geometry and callback behavior. The method validates square proxy weights,
interior flags, and `proxyfun` output/neighbor shapes for a small circle
chunker. Ground truth is stable geometry, 0-based callback sizing, and direct
proxy/source kernel blocks for the returned `Kpxy` values.

`test_flam_proxyfunr_column_and_row_shapes` checks rectangular proxy callbacks
for both column and row compression sides. The method verifies 0-based
neighbor filtering and output shapes for off-boundary target evaluation.
Ground truth is direct proxy-target and source-proxy kernel blocks for both
callback orientations.

`test_chunkermat_flam_applies_solves_and_logdet_against_dense` checks the
PyFLAM-backed boundary matrix wrapper. The method builds
`chunkermat(..., {"acceleration": "flam"})`, compares matrix-vector products
against the dense special matrix, solves a shifted Laplace system, and compares
`logdet()` to NumPy's dense determinant calculation.

`test_chunkermat_flam_proxy_paths_match_dense_application` checks the default
proxy-enabled and level-dependent proxy paths. The method applies both FLAM
`rskelf` operators plus the proxy-enabled `rskel` operator to a shifted
Laplace single-layer system, compares against the dense special matrix product,
and verifies `rskel` does not expose the `rskelf` solve helper.

`test_chunkermat_flam_adjoint_products_match_dense` checks SciPy adjoint
products for PyFLAM-backed boundary matrices. The method builds
complex-shifted `rskelf` and `rskel` operators, applies `.H` to vector and
multiple-RHS inputs, and compares with the conjugate-transpose dense matrix
product.

`test_chunkermat_flam_adjoint_solve_matches_dense` checks adjoint fast-direct
solve support for `rskelf`-backed FLAM matrices. The method calls
`solve(..., trans="c")` for vector and multiple-RHS right-hand sides and
verifies the dense conjugate-transpose system residual.

`test_chunkermat_flam_adds_dval_without_replacing_smooth_diagonal` checks
diagonal shifts for smooth kernels. The method compares scalar, complex-shift,
and vector-opdim smooth FLAM applications with `dval` against dense matrices
with an additive diagonal, ensuring the shift does not overwrite the native
smooth diagonal.

`test_chunkermat_flam_interleaved_block_kernel_matches_dense` checks smooth
2-by-2 interleaved block-kernel FLAM matrix application. The method builds a
mixed kernel with off-diagonal signs, applies a shifted PyFLAM operator, and
compares against the dense interleaved matrix product.

`test_chunkermat_flam_l2scale_matches_scaled_dense_matrix` checks FLAM l2
scaling. The method compares shifted smooth and singular-special FLAM
operators with `l2scale=True` against dense weighted matrices transformed by
the square-root source and target quadrature weights.

`test_chunkermat_flam_preserves_point_data_without_proxy` checks that point
data fields are passed through FLAM matrix callbacks. The method allocates a
chunker data row, evaluates a custom data-dependent kernel, and compares the
PyFLAM operator with the dense matrix. Ground truth is dense evaluation with
the proxy path disabled automatically for data-bearing chunkers.

`test_chunkermatapply_flam_accepts_multiple_rhs` checks direct
`chunkermatapply(..., {"acceleration": "flam"})` application on multiple
right-hand sides. The method applies the PyFLAM-backed path to two density
columns and compares the result to dense special-quadrature matrix products.

`test_chunkermatapply_flam_preserves_single_column_rhs` checks that direct
`chunkermatapply(..., {"acceleration": "flam"})` preserves an explicit
single-column density shape. The method applies the PyFLAM-backed path to an
`(n, 1)` density and compares against the dense matrix product with the same
shape.

`test_chunkerkerneval_flam_preserves_target_data_without_proxy` checks FLAM
target-evaluation callbacks when target `PointInfo` carries a `data` field.
The method leaves proxy use enabled in options, verifies the implementation
falls back to the non-proxy PyFLAM path for target-data kernels, and compares
both materialized and applied outputs against dense target evaluation.

`test_chunkerkerneval_flam_matches_eval_matrix_and_dense` checks off-boundary
target FLAM evaluation. The method compares `chunkerkerneval` and
`chunkerkernevalmat` with `acceleration="flam"` against dense target
evaluation for scalar and vector-opdim smooth kernels.

`test_chunkerkerneval_flam_complex_target_eval_matches_dense` checks complex
rectangular target FLAM evaluation. The method materializes and applies a
complex-valued smooth target-evaluation factor and compares both outputs
against dense target evaluation.

`test_chunkerkerneval_flam_interleaved_block_kernel_matches_dense` checks
smooth 2-by-2 interleaved block-kernel FLAM target evaluation. The method
materializes the rectangular PyFLAM eval matrix, applies it to a two-component
density, and compares both outputs against dense target evaluation.

`test_chunkerkerneval_flam_default_proxy_matches_dense` checks the integrated
rectangular FLAM proxy path for target evaluation. The method leaves
`useproxy=True`, materializes the PyFLAM eval matrix, applies the same factor
to a density, and compares both outputs with dense direct evaluation.

`test_chunkerkerneval_flam_proxy_by_level_matches_dense` checks
level-dependent rectangular FLAM proxy target evaluation. The method requests
`proxybylevel=True`, materializes the PyFLAM eval matrix, applies the same
factor to a density, and compares both outputs with dense direct evaluation.

`test_chunkerkerneval_flam_forceadap_matches_dense_adaptive_corrections`
checks close-target corrections in the FLAM target-evaluation path. The method
combines PyFLAM smooth evaluation with sparse adaptive correction blocks and
compares both materialized matrices and applied values against the dense
adaptive reference.

`test_chunkerkerneval_flam_same_source_special_quadrature_matches_dense`
checks FLAM self-target evaluation for singular kernels. The method requests
`chunkerkernevalmat(..., targobj=chnkr, {"acceleration": "flam"})` and
`chunkerkerneval(..., targobj=chnkr, {"acceleration": "flam"})` for a Laplace
single-layer kernel, then compares both routes against the dense GGQ
same-source matrix.

`test_chunkerinterior_flam_matches_direct_classification` checks FLAM interior
classification. The method evaluates inside, outside, and near-boundary sample
targets with `chunkerinterior(..., {"acceleration": "flam"})` and compares to
the direct classifier.

## `tests/test_geometry.py`

`test_flagnear_matches_bruteforce_chunk_node_distance` checks chunk-near flags
against an explicit brute-force loop. The equation is
`min_node_distance(target, chunk) < fac * chunk_length`. The method is
`flagnear` on a four-panel circle. Ground truth is the manually computed
boolean target-by-chunk matrix.

`test_flagnear_rectangle_grid_matches_direct_meshgrid_order` checks that
rectangular near-flagging preserves Fortran-style meshgrid ordering. The method
compares `flagnear_rectangle` on a flattened target grid to
`flagnear_rectangle_grid` on separate `x` and `y` vectors. Ground truth is
exact equality of the two boolean arrays.

`test_flagnear_rectangle_uses_per_chunk_padding_and_chunkgraph_delegates`
checks rectangular near-flag padding and chunkgraph delegation. The invariant
is that the chosen targets produce known tight and padded per-chunk masks.
The method compares tight and padded rectangle flags on an open polyline, then
checks `tochunkgraph` delegates to chunker implementations. Ground truth is
the exact boolean masks and equality between graph and chunker results.

`test_flagself_reports_close_source_target_pairs` checks duplicate coordinate
pair reporting for small arrays. The method is `flagself`. Ground truth is the
explicit index-pair matrix identifying target coordinates that coincide with
source coordinates.

`test_flagself_handles_grid_sized_inputs_and_strict_tolerance` stress-tests
`flagself` on an 80-by-80 grid. The equation is coordinate equality within a
strict `1e-14` tolerance; an additional target shifted by `1e-14` should not
create an extra reported pair. The method is `flagself` on large point arrays.
Ground truth is exactly one pair per source point in identity order.

`test_basic_2d_geometry_helpers` checks elementary differential-geometry
helpers. The equations are `perp([x,y]) = [y,-x]`, outward normals from
tangents, and curvature
`kappa = (x' y'' - y' x'') / |r'|^3`, which equals `1` on the unit circle.
The method is `perp`, `normal2d`, and `curvature2d`. Ground truth is explicit
perp values, chunker normals, and unit curvature.

`test_chunk_nearparam_on_line_segment` checks nearest-parameter recovery on a
straight line segment from `(0,0)` to `(2,0)`. The equation is the affine map
`x = t + 1` from reference coordinate `t in [-1,1]`, so targets project to
`t=-0.5` and `t=0.75`. The method is `chunk_nearparam` on one panel. Ground
truth is projected points, constant derivative `(1,0)`, zero second
derivative, and squared distances `1` and `0.0625`.

`test_chunker_nearest_selects_point_and_chunk` checks full chunker nearest
search on an open L-shaped polyline. The target `(1.25,0.6)` is closest to the
horizontal segment at `(1.25,0)`. The method is `Chunker.nearest`, which
chooses both a closest point and a source chunk. Ground truth is projected
point, derivative, zero second derivative, distance `0.6`, local parameter
`0.25`, and chunk index `0`.

## `tests/test_geometry_parity.py`

These tests compare the implemented I GEOMETRY surface against
`tests/golden/geometry_core.mat`, generated by
`scripts/matlab/generate_geometry_core_fixture.m`. The fixture intentionally
excludes `chunkerfit` and smoother workflows.

`test_top_level_domain_helpers_match_matlab_fixture` checks `ellipse`,
`starfish`, `nonflatinterface`, `redblue`, and `checkcurveparam` against saved
MATLAB outputs, including validation failure cases for bad callback shapes and
inconsistent dimensions.

`test_chnk_curve_helpers_match_matlab_fixture` checks `chnk.curves.linefunc`,
`fpara`, `fsine`, and `bymode`, including center and anisotropic scaling for
the Fourier-mode curve helper.

`test_domain_tree_and_region_helpers_match_matlab_fixture` checks
`hypoct_uni`, `pointinregion`, `regioninside`, and `mergeregions`. The method
converts Python's zero-based tree and graph indices to MATLAB's one-based
fixture indices before comparison.

`test_chunker_core_geometry_helpers_match_matlab_fixture` reconstructs a
MATLAB-saved noncircular chunker and compares weights, normals, tangents,
arclength density, curvature, arclength coordinates and derivatives,
endpoints, extrema, `sortinfo`, `sort`, and `datares`.

`test_chunker_flag_nearest_translate_and_uniform_helpers_match_matlab_fixture`
checks chunker and `chnk.geometry` near-flag wrappers, rectangle/grid near
flags, vectorized nearest-point results against MATLAB scalar-reference calls,
geometry cache recomputation, left/right translation operators, and
`chunkerfuncuni` uniform geometry including MATLAB-compatible spectral second
derivatives.

`test_chunker_refinement_and_reconstruction_helpers_match_matlab_fixture`
checks `split`, `refine`, `upsample`, `arcresample`, `rotate`, `reflect`,
`reverse`, `chunkerpoints`, and `merge`. The split/refine fixture stores
MATLAB geometry after recomputing normals and weights so Python's live geometry
caches are compared to the same state.

`test_chnk_geometry_helpers_match_matlab_fixture` checks `perp`, `normal2d`,
`curvature2d`, `chunk_nearparam`, and `flagself` against MATLAB outputs, with
one-based MATLAB pair indices converted at assertion time.

`test_arcparam_helpers_match_matlab_fixture` checks `chnk.arcparam.init` and
`chnk.arcparam.eval` against MATLAB fixture data for the full chunker and a
selected-panel subset, including stored coefficients, panel lengths,
condition/error diagnostics, original-node evaluation, and sample arclength
evaluation.

`test_chunkgraph_helpers_match_matlab_fixture` checks square-graph
construction, merged field access, source info, incidence matrices, extrema,
dense helper matrices, `edgeids`, `slicegraph`, translation, matrix transform,
rotation, reflection, and `ChunkGraph.copy()` mutation isolation for graph
vertices and edge chunk storage.

`test_chunkgraph_region_flag_operator_and_conversion_helpers_match_matlab_fixture`
checks `procverts`, `findregions`, graph `refine`, graph near-flag wrappers,
left/right translation and scalar/matrix operator overloads, `tochunkgraph`
for closed and open components, and `chunkgraphinregion` point/grid ids
against MATLAB fixture data. Region assertions compare Python's zero-based
signed loop convention to MATLAB's one-based signed loops.

## `tests/test_helm1d.py`

`test_helm1d_green_gradient_matches_finite_difference` checks the flat-interface
Helmholtz helper `G = exp(i k |x-y|)`. The method evaluates `helm1d.green` and
compares target `x`/`y` gradients and Hessian components to centered finite
differences. Ground truth is the finite-difference derivative and Hessian data.

`test_helm1d_kernel_selectors_and_kernel_wrapper` checks 1D Helmholtz selector
plumbing for single layer, double layer, target-normal derivative,
transmission block, and full 2-by-2 transmission system. The method is direct
point-kernel evaluation and generic `kernel("helm1d", ...)`. Ground truth is
direct formulas from `helm1d.green` for `s`, `d`, `dp`, `c2trans`, and `all`
blocks, plus singularity metadata `removable` for the single layer.

`test_helm1d_sweep_matches_direct_causal_sums` checks the sweeping convolution
helper. The equation is the direct weighted sum
`u_i = sum_j exp(i k |t_i-t_j|) q_j`, where `q_j = u_j w_j` and only selected
indices have nonzero input. The method is `helm1d.sweep`, which uses forward
and backward recurrences. Ground truth is an explicit all-pairs direct sum.

## `tests/test_kernel.py`

`test_laplace_kernel_wrapper_evaluates_directly` checks the generic kernel
factory for the Laplace double-layer potential on a circle. The boundary
identity for unit density on a closed unit circle gives the interior value
`-1` for this sign convention. The method is `kernel("laplace","d")` through
`chunkerkerneval`. Ground truth is `opdims=(1,1)`, smooth singularity metadata,
and value `-1` at the origin.

`test_kernel_add_scale_zero_and_nan_behaviors` checks zero/nan kernels and
scalar algebra. The equation is linearity: `2 * LaplaceD + 0` applied to unit
density at the origin should be `-2`. The method uses `kernel("zero")`,
`kernel("nan")`, scalar multiplication, and `chunkerkerneval`. Ground truth is
zero output, doubled Laplace output, NaN propagation, and `iszero`/`isnan`
flags.

`test_kernel_subtract_negate_divide_and_conjugate` checks subtraction,
negation, division, and complex conjugation of kernels. The equations are
`K-K=0`, `-K` flips sign, `K/2` halves the result, and conjugation maps
`1+2i` to `1-2i`. The method uses dense layer evaluation with a custom complex
kernel. Ground truth is the corresponding algebraic output, including
`(1-2i) * sum(weights)` for the custom conjugated kernel.

`test_kernel_fmm_fallback_matches_direct_layer_evaluation` checks the FMM path
for a Laplace single-layer kernel. The method evaluates the same circle density
with dense direct summation and with `{"acceleration": "fmm"}`. Ground truth is
exact agreement with the dense direct reference and non-null FMM metadata.

`test_fmm2dpy_laplace_gradient_and_helmholtz_layers_match_direct` checks
`fmm2dpy` acceleration for Laplace gradients/double-layer variants and
Helmholtz single, double, and gradient layers. The equations are the Laplace
and Helmholtz layer potentials applied to `cos(x)` density on a circle. The
method compares FMM evaluation at several targets against dense direct
evaluation with tolerance `1e-9`. Ground truth is the direct dense path.

`test_fmm2dpy_laplace_derived_selectors_match_direct` checks FMM wiring for
Laplace target-normal derivative, target-tangential derivative, Hilbert,
double-prime, combined-prime, and combined-gradient selectors. The method
compares FMM evaluation against dense direct evaluation on the same circle
density. Ground truth is the direct dense path.

`test_fmm2dpy_helmholtz_derived_selectors_match_direct` checks FMM wiring for
Helmholtz target-normal derivative, target-tangential derivative,
double-prime, and combined-prime selectors. The method uses the existing
Helmholtz gradient FMM output and compares against dense direct evaluation.
Ground truth is the direct dense path.

`test_helmholtz_factory_transmission_selectors_match_direct_kernel` checks
Python factory dispatch for 2D Helmholtz combined-gradient and transmission
representation selectors. The method evaluates `kernel("helm", selector, ...)`
for `cgrad`, `c2trans`, `all`, `trans_rep`, `trans_rep_prime`, and
`trans_rep_grad`, and compares each result to direct `chnk.helm2d.kern`.

`test_helmholtz_double_gradient_fmm_requests_dipole_gradients` checks the
new Helmholtz double-gradient FMM wiring with a fake `fmm2dpy` module. The
method builds `kernel("helm","dgrad",zk)`, calls its FMM evaluator, and asserts
that `hfmm2d` receives dipole strengths, source normals, and `pgt=2`. Ground
truth is the fake module's `gradtarg` returned in Fortran target ordering.

`test_biharmonic_laplacian_fmm_reuses_laplace_single_layer` checks the
biharmonic Laplacian FMM wiring with a fake `fmm2dpy` module. The method builds
`biharm2d_kernel("lap")`, calls its FMM evaluator, and asserts that the path
uses the Laplace single-layer FMM with the analytic constant correction. Ground
truth is the fake module's potential output combined with `sum(sigma)/(2 pi)`.

`test_fmm2dpy_biharmonic_selectors_match_direct` checks FMM wiring for
biharmonic single, double, target-normal derivative, gradient, Hessian, and
Laplacian selectors. The implementation uses Laplace log-moment decompositions;
the test compares against dense direct evaluation. Ground truth is the direct
dense path.

`test_fmm2dpy_stokes_layers_match_direct` checks FMM acceleration for Stokes
velocity, pressure, gradient, and combined kernels. The vector density is
`(cos(x), sin(y))` interleaved by point. The method compares `fmm2dpy` Stokes
results with dense direct results for single, double, pressure, gradient, and
combined selectors. Ground truth is the direct dense path.

`test_stokes_traction_fmm_reconstructs_stress_from_pressure_and_gradient`
checks the Stokes traction FMM path with a fake `fmm2dpy` module. The method
evaluates `kernel("stok","strac",mu)` through its FMM callback and verifies
that pressure and gradient FMM calls are combined as
`-p n + mu (grad u + grad u^T) n`. Ground truth is the explicit stress
contraction for two target normals.

`test_fmm2dpy_elasticity_selectors_match_direct` checks FMM wiring for
elasticity single, gradient, traction, double, alternate double, alternate
gradient, and alternate traction selectors. The implementation decomposes the
elasticity kernels into Laplace and Stokes FMM calls; the test compares against
dense direct evaluation. Ground truth is the direct dense path.

`test_kernel_fmm_fallback_tracks_kernel_algebra` checks that composed kernels
preserve a usable FMM evaluator. The equation is the algebraic combination
`2 * LaplaceS - LaplaceD`. The method compares dense direct and FMM evaluation
for the composed kernel. Ground truth is the dense direct result and non-null
combined FMM metadata.

`test_kernel_interleave_builds_mixed_block_systems` checks block/interleaved
kernel assembly. The block equation is
`[[D, -S], [S, 0]]` applied to a two-component density. The method builds a
mixed kernel, assembles its dense evaluation matrix, and compares matrix-vector
application to `chunkerkerneval`; it also checks the FMM path. Ground truth is
the explicit weighted matrix product and equality with the non-FMM evaluation.

## `tests/test_kernels.py`

`test_laplace_green_matches_direct_formula` checks the 2D Laplace Green's
function formula. For targets and sources with squared distances `4` and `5`,
the value must be `-log(r2)/(4 pi)`. The method is direct `lap2d.green`.
Ground truth is the closed-form value, gradient, and Hessian for the same
source-target offsets.

`test_laplace_direct_layer_evaluation_on_circle` checks direct Laplace
single- and double-layer evaluation on a radius-2 circle at the center. The
equations are `S[1](0) = -R log(R)` for this kernel normalization and
`D[1](0) = -1` for the double layer. The method is `chunkerkerneval` with
lambda kernels calling `lap2d.kern`. Ground truth is the analytic circle
potential values.

`test_laplace_kernel_selectors_have_expected_shapes` checks point-kernel
selectors for Laplace single, double, combined, combined-prime, and gradient
forms. The equation content is the Laplace Green's function and its normal or
gradient projections. The method is direct `lap2d.kern` evaluation with
`pointinfo`. Ground truth is direct `lap2d.green` value, gradient, and Hessian
data projected into each selector block.

`test_helmholtz_green_gradient_matches_finite_difference` checks the 2D
Helmholtz Green's function derivatives. The method evaluates `helm2d.green`
at complex wavenumber `1.2+0.4i` and compares target `x`/`y` gradients and
Hessian components to centered finite differences. Ground truth is the
finite-difference derivative and Hessian data.

`test_helmholtz_kernel_selectors_have_expected_shapes` checks 2D Helmholtz
single, double, target-normal derivative, and combined selectors. The method is
direct `helm2d.kern` on point-info derived from a circle. Ground truth is
direct `helm2d.green` value, gradient, and Hessian data projected into the
single, double, double-prime, and combined selectors.

## `tests/test_lege.py`

`test_exps_round_trips_values_and_coefficients` checks Legendre expansion
matrices. The equation is `coeffs -> values = V coeffs -> U values = coeffs`.
The method is `lege.exps(12)`. Ground truth is round-trip coefficient equality,
quadrature weights summing to `2`, and symmetric Gauss-Legendre nodes.

`test_rts_aliases_match_exps_nodes_weights` checks that `rts` and `rts_stab`
are aliases for the nodes and weights returned by `exps`. The method evaluates
all three functions at order 9. Ground truth is exact equality of nodes and
weights.

`test_pols_matches_known_low_order_polynomials` checks Legendre polynomial
values and derivatives through degree 4. The equations are the standard
closed forms for `P_0` through `P_4` and `P_3' = (15x^2-3)/2`. The method is
`lege.pols`. Ground truth is direct polynomial evaluation.

`test_dermat_differentiates_node_values` checks the Legendre differentiation
matrix. The equation is that differentiating nodal values of a pure Legendre
mode equals evaluating the derivative coefficients from `derpol`. The method
uses `lege.dermat`, `lege.derpol`, and the expansion matrix. Ground truth is
coefficient-space differentiation and zero derivative for constants.

`test_matrin_interpolates_legendre_node_values` checks interpolation from
Gauss-Legendre nodes to arbitrary targets. The equation is evaluating a known
Legendre coefficient vector at target points. The method is `lege.matrin`.
Ground truth is direct polynomial basis evaluation through `lege.pols`, plus
identity behavior when targets are the original nodes.

`test_exev_evaluates_single_and_multiple_expansions` checks coefficient
evaluation for one and multiple Legendre expansions. The equation is
`1 + 2x + 3 P_2(x)` for the first expansion and twice that for the second
column. The method is `lege.exev`. Ground truth is direct closed-form
polynomial evaluation.

`test_intpol_and_intmat_integrate_constants_from_left_endpoint` checks
coefficient-space and nodal integration from `x=-1`. The equation is
`integral_-1^x 2 dt = 2(x+1)` and `integral_-1^x 1 dt = x+1`. The method is
`lege.intpol`, `lege.pols`, and `lege.intmat`. Ground truth is those exact
antiderivatives.

`test_barywts_reproduce_lagrange_basis_sign_pattern` checks barycentric
weights for Gauss-Legendre nodes. The method is `lege.barywts`. Ground truth
is an independent product formula for normalized barycentric weights plus
exact interpolation of a low-order polynomial at off-node target points.

`test_bernstein_ellipse_matches_conformal_map` checks Bernstein ellipse point
generation. The method is `lege.bernstein_ellipse(8, 2.0)`. Ground truth is the
conformal map `(z + 1/z)/2` applied to equally spaced circle nodes.

`test_polsum_matches_pols_and_recurrence_total` checks the `polsum` recurrence.
The method is `lege.polsum(xs, 5)`. Ground truth is direct `lege.pols`
evaluation plus the weighted normalization sum used by MATLAB root helpers.

`test_tayl_matches_direct_legendre_step` checks Taylor stepping for Legendre
polynomials. The method advances `P_6` and its derivative from `x` to `x+h`
with `lege.tayl`. Ground truth is direct `lege.pol(x+h, 6)` evaluation,
including the zero-step case.

`test_adapgauss_integrates_scalar_and_vector_functions` checks adaptive
Gauss-Legendre integration. The method is `lege.adapgauss` on a polynomial
scalar integrand and a two-column vector integrand. Ground truth is the exact
polynomial integrals and successful MATLAB-style status metadata.

## `tests/test_matlab_fixtures.py`

`test_legendre_basic_fixture_matches_matlab` checks the compact MATLAB fixture
for core Legendre data. The method recomputes `lege.exps` and `lege.dermat` in
Python. Ground truth is `tests/golden/lege_basic.mat`, including nodes,
weights, transform matrices, and derivative matrix.

`test_circle_chunker_fixture_matches_matlab` checks that Python `chunkerfunc`
matches MATLAB fields for a circle chunker. The equations are the circle
position, first derivative, second derivative, normals, weights, area, and
chunk lengths. The method builds the circle from an analytic callback and
compares stored fields. Ground truth is `tests/golden/chunker_circle.mat`.

## `tests/test_matlab_parity.py`

`test_extended_legendre_helpers_match_matlab_fixture` checks the broader
Legendre helper surface against MATLAB. The equations include polynomial values
and derivatives, interpolation matrices, integration matrices, expansion
evaluation, coefficient antiderivatives, coefficient derivatives, and
barycentric weights. It also checks `rts`, `rts_stab`, `bernstein_ellipse`,
`polsum`, scalar MATLAB `tayl` calls, and scalar `adapgauss` output/status
metadata against saved fixture outputs. The method recomputes every object in
Python. Ground truth is `tests/golden/lege_extended.mat`.

`test_chunker_geometry_and_transforms_match_matlab_fixture` checks MATLAB
parity for chunker geometry, move/transform operations, and dense helper
matrices. The equations are affine geometry transforms, determinant area
scaling, spectral differentiation, cumulative integration, scalar/vector ones
matrices, and centroids. The method reconstructs a MATLAB-saved chunker,
applies Python transforms, and compares helper matrices. Ground truth is
`tests/golden/chunker_ops.mat`, generated on demand when missing.

`test_chunker_storage_and_data_helpers_match_matlab_fixture` checks MATLAB
parity for `chunkerpref`, storage resizing, data-row allocation and clearing,
`checkadjinfo`, and explicit `Chunker.copy()` value behavior. The method
replays the same storage/data operations in Python and compares full backing
arrays where MATLAB exposes them. Ground truth is `tests/golden/chunker_ops.mat`.

`test_laplace_point_kernels_match_matlab_fixture` is parametrized over
`s`, `d`, `sp`, `stau`, `hilb`, `sgrad`, `dgrad`, `dp`, `c`, `cp`, and
`cgrad`. The equations are the 2D Laplace Green's function, normal/tangent
projections, Hessian projections, Hilbert-style kernel, and combined-layer
linear combinations. The method is direct point-kernel evaluation through
`lap2d.kern`. Ground truth is `tests/golden/kernel_pointinfo.mat`.

`test_helmholtz_2d_point_kernels_match_matlab_fixture` is parametrized over
`s`, `d`, `sp`, `stau`, `sgrad`, `dgrad`, `dp`, `c`, `cp`, `cgrad`,
`c2trans`, `all`, `trans_rep`, `trans_rep_prime`, and `trans_rep_grad`. The
equations are the 2D Helmholtz Green's function and the same normal, tangent,
gradient, combined, and transmission-representation blocks; devtools parity
also covers the corresponding Helmholtz-difference identities. The method is
direct `helm2d.kern` with saved wavenumber and coefficient data. Ground truth
is the MATLAB point-kernel fixture and `devtools_easy.mat`.

`test_helmholtz_1d_point_kernels_match_matlab_fixture` is parametrized over
`s`, `d`, `sp`, `stau`, `dp`, `c`, `cp`, `c2trans`, `all`, `trans_rep`,
`trans_rep_prime`, and `trans_rep_grad`. The equation is
`exp(i k |x-y|)` and transmission-system blocks assembled from its values,
normal derivatives, and gradients. The method is direct `helm1d.kern` with
selector-specific coefficients. Ground truth is the MATLAB point-kernel
fixture.

`test_stokes_point_kernels_match_matlab_fixture` is parametrized over `s`,
`spres`, `strac`, `d`, `dpres`, `dtrac`, `sgrad`, `dgrad`, `c`, `cpres`,
`ctrac`, and `cgrad`. The equations are 2D Stokes velocity, pressure,
traction, gradient, and combined layer blocks at viscosity `mu`. The method is
direct `stok2d.kern`. Ground truth is the MATLAB point-kernel fixture; direct
lower-level `cgrad` uses MATLAB's saved `dgrad` and `sgrad` component blocks as
the reference because the saved MATLAB combined `cgrad` value combines `sgrad`
twice.

`test_elasticity_point_kernels_match_matlab_fixture` is parametrized over
`s`, `sgrad`, `strac`, `d`, `dalt`, `dalttrac`, and `daltgrad`. The equations
are 2D linear elasticity single-layer, gradient, traction, double-layer,
alternate double-layer, alternate traction, and alternate gradient blocks for
saved Lame parameters. The method is direct `elast2d.kern`. Ground truth is
the MATLAB point-kernel fixture.

`test_kernel_objects_match_matlab_fixture` is parametrized over MATLAB
`@kernel` factories for Laplace, Helmholtz 2D, Helmholtz 1D, Stokes,
elasticity, zero/NaN kernels, and a custom callable. The method compares
Python factory metadata and direct evaluations against saved MATLAB
`@kernel` object outputs. The elasticity `sgrad` value is compared while
Python keeps the correct gradient operator dimensions that MATLAB metadata
omits.

`test_kernel_algebra_and_interleave_match_matlab_fixture` checks scalar
kernel algebra, addition/subtraction/negation/division, conjugation, and a
2-by-2 interleaved Laplace block kernel. Ground truth is saved MATLAB
`@kernel` algebra and interleave output from `kernel_pointinfo.mat`.

`test_green_helpers_match_matlab_fixture` checks `lap2d.green`,
`helm2d.green`, `helm1d.green`, and `helm1d.sweep` against MATLAB values,
gradients, Hessians, and sweep sums stored in `kernel_pointinfo.mat`.

`test_biharmonic_helpers_match_matlab_bhgreen_fixture` checks
`biharm2d.green`, `biharm2d.kern`, and the `kernel("biharm", ...)` factory
against MATLAB `chnk.flex2d.bhgreen`-derived value, gradient, Hessian,
Laplacian, double-layer, target-normal, gradient, and Hessian selector data.

`test_dense_native_operator_paths_match_matlab_fixture` checks dense operator
assembly and evaluation for Laplace and Stokes, point-info flattening, smooth
matrix application, zero-kernel matrices, smooth scalar integration, callable
integration, and direct point/grid interior classification. The matrix
equation is `M_ij = K(x_i,y_j) w_j`; application is `M sigma`. The method uses
`pointinfo`, `chunkermat`, `chunkermatapply`, `chunkerintegral`,
`chunkerinterior`, `chunkerkernevalmat`, `chunkerkerneval`, and
`quadnative.buildmat`. Ground truth is `tests/golden/operator_parity.mat`,
including matrices, applied values, integrals, and classifications.

`test_accelerated_operator_paths_match_matlab_fixture` checks accelerated
operator parity for a Laplace single-layer boundary matrix. The method compares
Python `ChunkerFMMMatrix` and `chunkermatapply(..., {"acceleration": "fmm"})`
products against MATLAB forced-FMM output, then compares Python
`ChunkerFLAMMatrix` matvec and `.solve()` results against MATLAB
`chunkerflam`/`rskelf_mv`/`rskelf_sv` output on the same deterministic random
right-hand side. Ground truth is `tests/golden/operator_parity.mat`, including
the dense special-quadrature matrix, FMM product, FLAM product, and FLAM solve.

`test_section_iii_quadratures_match_matlab_fixture` checks Section III
quadrature parity. The method compares native quadrature, log/removable/PV/HS
auxiliary tables, `quadggq` self/near/sparse special blocks with correction and
`ilist` variants, and `quadadap` log matrices with robust close replacement.
Ground truth is `tests/golden/quadggq.mat`, including exact table entries,
matrix entries, direct block outputs, and adaptive close-interaction matrices.

`test_rcip_recursive_compression_matches_matlab_fixture` checks RCIP parity for
two corner-adjacent edges. The equations are the shifted Legendre basis
matrices and the recursive Schur-compressed inverse operator `R` for a Laplace
double-layer corner problem. The method runs `shiftedlegbasismats`,
`Rcompchunk`, and `rhohatInterp`. Ground truth is `tests/golden/rcip.mat`,
including `R`, saved intermediate matrices, local refined chunker fields,
interpolated densities, source info, and weights.

## `tests/test_operators.py`

`test_chunkermat_matches_chunkerkerneval_on_boundary_for_smooth_kernel` checks
that dense boundary matrix application matches direct kernel evaluation at the
same boundary nodes for a smooth kernel. The equation is the weighted discrete
sum with `K = 1 + dx^2 + 0.5 dy^2`. The method compares `chunkermatapply` with
`chunkerkerneval(..., targ=chnkr)` for smooth non-singular data. Ground truth
is the raw kernel matrix applied to the density after source quadrature
weighting, plus equality between the two direct routes.

`test_chunkermatapply_fmm_matches_special_matrix_application` checks FMM
matrix application for a singular boundary operator. The method applies a
Laplace single-layer kernel through
`chunkermatapply(..., {"acceleration": "fmm"})`, then compares against the
dense special-quadrature matrix product. Ground truth is agreement after sparse
self/neighbor GGQ corrections are added to the FMM result.

`test_chunkermat_fmm_returns_matrix_free_operator_matching_dense_application`
checks the explicit FMM return path on `chunkermat`. The method requests
`chunkermat(..., {"acceleration": "fmm"})` for a Laplace single-layer kernel,
verifies that the result is a `ChunkerFMMMatrix`, and compares both vector and
multiple-right-hand-side products against the dense special-quadrature matrix.
Ground truth is agreement with the dense product after cached sparse GGQ
corrections are applied.

`test_pointinfo_uses_matlab_chunk_contiguous_ordering` checks point ordering
when flattening chunker fields. The invariant is MATLAB/Fortran chunk-contiguous
ordering: all nodes of chunk 0, then all nodes of chunk 1, and so on. The
method is `pointinfo(chnkr)`. Ground truth is equality of the full flattened
`r`, `d`, `d2`, and normal arrays with the original chunk arrays.

`test_chunkerkernevalmat_matches_direct_target_evaluation` checks target
evaluation matrix assembly. The equation is `values = EvalMat sigma` for the
same smooth polynomial kernel. The method compares `chunkerkernevalmat @ dens`
to `chunkerkerneval`. Ground truth is the explicit target/source kernel block
weighted by source quadrature weights, plus direct evaluation equality.

`test_chunkermat_accepts_kernel_objects` checks that `chunkermat` accepts a
`Kernel` object, not only callables. The method uses `kernel("zero")`.
Ground truth is an `npt x npt` zero matrix.

`test_quadnative_buildmat_matches_dense_chunkermat` checks that the lower-level
native quadrature builder agrees with the public dense matrix builder for a
smooth scalar kernel. The equation is the same weighted direct matrix
`K w`. The method compares `quadnative.buildmat` to `chunkermat`. Ground truth
is the explicit raw smooth kernel matrix multiplied by source weights, plus
equality of the two dense matrices.

`test_chunkerintegral_accepts_values_and_callables` checks scalar integration
over a circle. The equations are `integral_0^(2pi) 1 ds = 2 pi` and
`integral_circle x^2 ds = pi` on the unit circle. The method is
`chunkerintegral` with explicit values and with a callable. Ground truth is
the analytic circle integrals.

`test_chunkerinterior_classifies_points_and_grids` checks inside/outside
classification for a closed square. The method is the direct node-polygon
fallback in `chunkerinterior`, applied to both a point list and an `(x,y)`
grid specification. Ground truth is the obvious square membership of the
sample targets.

`test_chunkerinterior_fmm_matches_direct_with_close_correction` checks the
accelerated interior classifier. The method monkeypatches `chunkerkerneval` to
verify the FMM path is used, evaluates points inside, outside, and very close
to a circle boundary, and compares against the direct classifier. Ground truth
is the exact expected circle membership `[inside, outside, just-inside,
just-outside]` plus direct/FMM classification agreement after close-boundary
correction.

## `tests/test_quadggq.py`

`test_quadggq_tables_are_loaded_from_packaged_numpy_data` checks that a
representative GGQ near table is available as package-owned `.npz` data and
that `quadggq` no longer exposes the old MATLAB checkout path helper. The
method opens `chunkie/data/quadggq/ggqnear16.npz` through
`importlib.resources` and verifies known first node/weight entries.

`test_matlab_log_quadrature_tables_load_for_each_legendre_node` checks log GGQ
table loading/generation for Legendre order 8. The method is `quadggq.setup`,
`getlogquad`, and MATLAB-style `logavail` near-rule availability. Ground truth
is known packaged table entries, one self rule per Legendre node,
interpolation matrix shapes, no quadrature node exactly at the singular target
node, and self-rule weights summing to `2`.

`test_quadggq_buildmat_removes_laplace_single_layer_diagonal_infinities` checks
that log special quadrature replaces singular diagonal entries for the Laplace
single layer. The equation is the self-boundary single-layer potential of unit
density on the unit circle, which should be approximately zero for this
normalization. The method compares force-smooth native assembly, which has
infinite diagonal entries, to `quadggq.buildmat`, which uses special
quadrature. Ground truth is finite special matrix entries and near-zero
application to ones.

`test_chunkermat_uses_special_quadrature_for_log_kernels_by_default` checks
public dispatch to log GGQ for singular kernels. The method calls
`chunkermat` and `chunkerkerneval` with a Laplace single-layer `Kernel` on the
same circle boundary. Ground truth is finite results, equality between matrix
and evaluation routes, and near-zero unit-density boundary value.

`test_quadggq_handles_complex_helmholtz_single_layer_blocks` checks that GGQ
assembly supports complex kernels. The equation is the Helmholtz single-layer
kernel at wavenumber `1.3+0.2i`. The method is `quadggq.buildmat`. Ground
truth is complex-valued GGQ assembly matching adaptive log quadrature.

`test_nearbuildmat_matches_buildmat_neighbor_block_and_correction` checks
neighbor-block construction and correction output. The method builds the full
special matrix, extracts a neighboring target/source block, and compares it to
`nearbuildmat`; with `corrections=True`, it subtracts the native smooth block.
Ground truth is equality to the full special block and to `near - native` for
the correction.

`test_buildmat_ilist_skips_bad_neighbor_and_self_special_blocks` checks the
`ilist` skip mechanism for selected self/neighbor blocks. The method compares
blocks from a fully special matrix, a skipped matrix, and a force-smooth matrix.
Ground truth is that skipped blocks use smooth/native assembly, while unrelated
blocks still use special quadrature.

`test_buildmattd_returns_sparse_special_blocks_only` checks sparse
special-block assembly. The method calls `quadggq.buildmattd` and compares
self and neighbor blocks against `quadggq.buildmat`, verifies a far block is
zero, and checks `ilist` skip behavior. Ground truth is exact block equality
and a SciPy sparse return type.

`test_pv_and_hs_ggq_tables_are_available_for_matlab_orders` checks PV and
hypersingular table availability. The method is `hqsuppavail`, `getpvquad`,
and `gethsquad`. Ground truth is order-8 support and known first table entries
for PV and HS rules.

`test_setup_accepts_pv_and_hs_singularities` checks the `setup` dispatcher for
principal-value and hypersingular auxiliary quadrature. The method calls
`quadggq.setup(8, "pv")` and `quadggq.setup(8, "hs")`. Ground truth is stored
type metadata, one self rule per Legendre node, representative PV/HS rule
nodes and weights, and constant-preserving interpolation matrices.

`test_chunkermat_uses_special_quadrature_for_pv_and_hs_kernels` checks public
dispatch for Laplace `sgrad` and `dgrad`, which are marked PV and HS
respectively. The method is `chunkermat` on a circle. Ground truth is finite
matrices with shape `2*npt x npt`, self blocks matching the PV/HS sparse
special-block builder, far blocks matching force-smooth assembly, and nonzero
special self blocks.

`test_quadadap_buildmat_uses_adaptive_neighbor_blocks` checks adaptive
near-neighbor assembly. The method monkeypatches `quadadap.adapgausswts`,
builds a Laplace single-layer matrix, and verifies two adaptive neighbor calls
per chunk while preserving equality with the GGQ special matrix on a circle.

`test_quadadap_robust_mode_repairs_non_neighbor_close_blocks` checks robust
close-interaction replacement. The method merges two nearly touching circles,
enables `quadadap.buildmat(..., robust=True)`, and verifies adaptive correction
calls for target subsets outside the self/neighbor blocks. Ground truth is a
finite matrix, at least one non-panel-sized adaptive target set, and a
measurable difference from the non-robust matrix.

## `tests/test_rcip.py`

`test_ipinit_interpolates_to_half_panels_and_preserves_weights` checks RCIP
interpolation from a parent Legendre panel to two half panels. The equation
uses a polynomial `t^5 - 0.2 t^3 + 0.7`, which is exactly represented at order
8. The method is `rcip.IPinit`. Ground truth is exact interpolation to the two
half-panel nodes and equality of weighted integrals before and after
interpolation.

`test_setup_returns_zero_based_rcip_indices_and_block_shapes` checks the index
sets and prolongation matrices produced by RCIP setup. There is no PDE
equation; the invariant is array shape, zero-based indexing, and expected edge
pair list. The method is `rcip.setup(4,2,3, ...)`. Ground truth is block
matrix dimensions, index bounds, `ilist` columns, half-sized first-level index
sets, and explicit Kronecker constructions of `Pbc` and `PWbc` from `IPinit`.

`test_schurbana_matches_direct_block_formula_shapes` checks the Schur
complement assembly helper used inside RCIP. The equation is the block
compression/Schur update applied to randomly generated matrices with a
well-conditioned good block. The method is `rcip.SchurBana`. Ground truth is a
direct independent block-update formula for every output subblock.

`test_rcompchunk_identity_baseline_and_corner_refine` checks corner refinement
and the identity-kernel baseline for recursive compression. The method builds
a two-edge corner graph, applies `corner_refine`, then calls `Rcompchunk` with
an identity kernel. Ground truth is two added chunks per adjacent edge, an
identity `R` matrix of the expected size, saved metadata, and identity
`rhohatInterp` fallback output.

`test_rcompchunk_runs_recursive_compression_for_corner_edges` checks a
nontrivial RCIP solve for a Laplace double-layer corner problem. The method is
`Rcompchunk` with `kernel("lap","d")`, two recursive subdivisions, and saved
depth 2, followed by `rhohatInterp`. Ground truth is a finite non-identity
compression matrix, deterministic trace/Frobenius/leading-row values, expected
saved-array counts, and interpolated density, source-info, and weight shapes.

`test_chunkgraph_rcip_runs_selected_vertices_and_ignores_marked_vertices`
checks the chunkgraph-level RCIP driver. The method runs `chunkgraph_rcip` on a
square graph, skips one vertex, and applies a Laplace double-layer kernel at
the remaining corners. Ground truth is the selected vertex list, two incident
edges per corner, compression matrices matching direct per-vertex
`Rcompchunk` calls, and saved recursion metadata.

`test_chunkgraph_rcip_subselects_global_block_kernels` checks local
subselection from a global edge-by-edge block-kernel matrix. The method creates
distinct zero kernels for each global block and runs RCIP at one square-graph
vertex. Ground truth is that only the incident-edge submatrix is used and the
resulting zero-kernel compression matrix is identity.

## `tests/test_rcip_parity.py`

`test_rcip_setup_helpers_match_matlab_fixture` checks strict MATLAB parity for
the RCIP setup helpers. The method compares `IPinit`, `Pbcinit`, and `setup`
against `tests/golden/rcip.mat`, including zero-based translations of MATLAB's
index arrays and alias coverage for `ipinit` and `pbcinit`.

`test_rcip_schurbana_matches_matlab_fixture` checks the RCIP Schur-Banachiewicz
block update. Ground truth is a deterministic MATLAB-saved block system and
the resulting updated compression matrix; both `SchurBana` and `schurbana` are
compared.

`test_corner_refine_matches_matlab_corner_topology_fixture` checks the Python
corner refinement convenience helper against MATLAB chunkgraph topology saved
in the fixture. The method verifies the incident edge/sign ordering and the
expected endpoint chunk-count increments after two refinement passes.

`test_chunkgraph_rcip_driver_matches_matlab_fixture` is parametrized over
`chunkgraph_rcip`, `chunkgraphrcip`, and `rcipchunkgraph`. The method rebuilds
a two-edge corner graph from MATLAB-saved edge chunkers, runs selected-vertex
RCIP compression, and compares the returned `RCIPChunkGraphResult` fields,
compression matrix, and saved recursion blocks to the MATLAB fixture.

## `tests/test_smoother.py`

`test_smoother_uniform_mesh_matches_polygon_edges` checks the uniform mesh
preprocessing used by the rounded polygon smoother. The equations are unit edge
lengths and outward normals for a unit square; the first edge centroid is
`(0.5,0)` with normal `(0,-1)`. The method is `smoother.get_umesh`. Ground
truth is edge lengths, centroid, face normal, and unit pseudo-normal magnitude.

`test_smoother_get_mesh_expands_legendre_panels` checks expansion of a uniform
polygon mesh into Legendre panel nodes. The method is `smoother.get_mesh` with
two panels per edge and order 5. Ground truth is the expected flattened shapes
for points, normals, pseudo-normals, and weights, first-panel coordinates and
normal fields, and weight sum equal to the original edge-length sum.

`test_smoother_smooth_returns_rounded_chunker_and_error_outputs` checks the
high-level smoother path. The method is `smoother.smooth` on a square with
width `0.1` and `return_error=True`, internally producing a rounded chunker.
Ground truth is eight chunks, zero reported smoothing error in the current
implementation, per-point error shape, rounded edge geometry, clean adjacency,
unit normals, positive chunk lengths, and area between `0.9` and `1.0`.

## `tests/test_sortinfo.py`

`test_sort_reorders_two_open_segments_and_remaps_adjacency` checks sorting of
open chunks after their storage order has been manually reversed. The invariant
is that sorting restores geometric order and remaps adjacency consistently.
The method is `Chunker.sort`. Ground truth is success code `ier=0`, adjacency
equal to the original chunker, and restored coordinate, derivative, normal, and
weight storage.

## `tests/test_spcl.py`

`test_absconvgauss_derivatives_match_finite_differences` checks the smoothed
absolute-value convolution helper. The method evaluates value, first
derivative, and second derivative, compares all three against the closed-form
Gaussian-convolution formulas, then compares derivatives against centered
finite differences of the lower-order outputs.

## `tests/test_stok2d.py`

`test_stokes_kernel_shapes` checks Stokes selector and factory dimensions. The
equations are the 2D Stokes single-layer velocity, single-layer pressure,
single-layer gradient, double-layer traction, combined pressure, and combined
gradient blocks. The method is direct `stok2d.kern` evaluation and
`kernel("stok", ...)` metadata inspection. Ground truth is explicit Stokeslet
velocity and pressure formulas, finite-difference single-layer gradients, and
`opdims`.

`test_stokes_dtrac_matches_pressure_gradient_stress_identity` checks the
Stokes double-layer traction formula. The equation is
`t = -p n + mu (grad u + grad u^T) n`. The method evaluates `dtrac`, `dgrad`,
and `dpres` for one source/target pair and random strengths, reconstructing
traction from pressure and velocity gradient. Ground truth is the stress
identity.

`test_stokes_combined_variants_match_linear_combinations` checks Stokes
combined selectors. The equations are `cvel = a*d + b*s`,
`cpres = a*dpres + b*spres`, `ctrac = a*dtrac + b*strac`, and
`cgrad = a*dgrad + b*sgrad`. The method evaluates each combined selector and
its explicit linear combination. Ground truth is equality for the selected
coefficient vector.
