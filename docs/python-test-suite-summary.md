# Python Test Suite Summary

This document summarizes the Python tests under `tests/test_*.py`. The current
collection expands to 186 pytest cases because several MATLAB parity tests are
parametrized; those parametrized functions are described once, with the covered
selector list called out explicitly.

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
use HS support tables. FMM tests request `{"usefmm": True}` and compare the
accelerated path against the dense direct path. RCIP tests exercise recursive
compressed inverse preconditioning for corner edges. There is no active Python
test that exercises FLAM compression; `chunkerinterior` explicitly uses a
direct node-polygon fallback. Close-boundary correction and FMM interior
acceleration should be implemented; FLAM interior acceleration is deferred.

Ground truth comes from four places:

- Closed-form geometry and PDE identities, such as circle area, line-segment
  nearest points, Green's functions, and traction formulas.
- Finite differences, used for derivative and gradient validation.
- Dense direct/native computation, used as the reference for FMM, special
  dispatch, and operator wrapper tests.
- MATLAB-generated golden fixtures in `tests/golden`, used for strict parity
  with the MATLAB `chunkIE` implementation.

## Implementation Scope Tracked By Tests

The living docs split remaining MATLAB parity work into three buckets.

Should implement:

- FMM integration everywhere it applies to implemented kernel/operator
  families, including `chunkermat` FMM acceleration, `chunkerinterior` FMM
  acceleration, and missing selector wiring for biharmonic, elasticity, and
  other unsupported selectors.
- Full adaptive/close quadrature: complete `quadadap` and
  `quadggq/buildmattd`.
- Advanced RCIP workflows beyond the current two-edge corner fixture.
- Remaining `+lege` helpers: `adapgauss`, `bernstein_ellipse`, `polsum`, and
  `tayl`.
- Adaptive refinement in `chunker.refine` and `chunkerfunc`.
- `chunkerinterior` close-boundary correction.

Implemented from this scope:

- Top-level geometry/domain helpers: `checkcurveparam`, `ellipse`,
  `hypoct_uni`, `mergeregions`, `nonflatinterface`, `pointinregion`, `redblue`,
  `regioninside`, and `starfish`.
- Helmholtz double-gradient FMM selector wiring.
- Stokes traction FMM selector wiring.

Deferred implementation:

- FLAM-backed acceleration, including `chunkerflam`, `+chnk/+flam`, full
  MATLAB-style FLAM integrations, `chunkermat` FLAM acceleration, and
  `chunkerinterior` FLAM acceleration. This should be revisited after `pyflam`
  exists.
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
is the original chunker node coordinates, unit-speed derivatives
`|dr/ds| = 1`, and orthogonality `dr/ds dot d2r/ds2 = 0`.

`test_arcparam_derivatives_are_consistent_on_circle` checks the same
arclength evaluator away from the original nodes on a radius-2 circle. The
circle equations imply `r dot dr/ds = 0`, `|dr/ds| = 1`, and
`dr/ds dot d2r/ds2 = 0`. The method is spectral interpolation from the chunker
arclength data. Ground truth is the analytic circle geometry and arclength
differential identities.

`test_arcresample_makes_panel_speed_constant` checks that `Chunker.arcresample`
reparameterizes panels by arclength. The invariant is that each panel has
constant speed density `chunklen / 2` on the reference interval `[-1,1]`.
The method is arclength resampling of a Legendre chunker. Ground truth is
preservation of area and total length, nonnegative reported error, and constant
panel speed after resampling.

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
Ground truth is the expected block shape and `opdims` metadata for each
selector.

`test_biharmonic_layer_evaluation_uses_special_quadrature` checks the
self/near singular dispatch for a biharmonic single-layer potential on a circle.
The kernel is marked as logarithmic, so `chunkerkerneval` on a chunker target
uses GGQ special quadrature instead of naive coincident-node quadrature. Ground
truth is not an analytic value; it is that the evaluation returns a finite
scalar of the correct shape for a smooth unit density at an interior target.

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
`r` property changes `rstor`. The method is direct chunk storage manipulation.
Ground truth is array shape and equality of the public slice with the backing
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
`__mul__`. Ground truth is the weighted center shift plus exact area and length
scaling.

`test_matrix_transform_updates_derivatives_normals_and_weights` checks affine
matrix transforms. The equations are `r_new = A r`, `d_new = A d`, and
`area_new = det(A) area`. The method uses matrix multiplication
`A @ chnkr`, recomputing normals and weights. Ground truth is `np.einsum`
application of the matrix and determinant-based area scaling; multiplication
by a matrix with `chnkr * A` is expected to raise `TypeError`.

`test_rotate_and_reflect_match_matlab_transform_formulas` checks rotation and
reflection helpers with source and destination centers. The equations are
`r_new = R(theta)(r-r0)+r1` for rotation and the standard reflection matrix
with angle `angle` for reflection. The method is `rotate` and `reflect`, which
transform coordinates, derivatives, and normals. Ground truth is explicit
matrix application to each stored field.

`test_chunker_spectral_helpers_on_circle` checks the spectral expansion,
arclength, differentiation, and differentiation-matrix helpers. The equation is
`s = pi(t+1)` on the one-panel circle helper and `d/ds sin(s) = cos(s)`.
The method uses `exps`, `arclengthfun`, `arclengthder`, and `diffmat`.
Ground truth is analytic sine differentiation and matching matrix-based
differentiation.

`test_intmat_integrates_in_chunk_order` checks cumulative arclength integration
after refining a circle chunker. The equation is `integral_0^s 1 ds = s`.
The method builds `intmat` and applies it to an all-ones vector in Fortran
chunk order. Ground truth is `arclengthfun()` flattened in the same order.

`test_onesmat_and_normonesmat_shapes` checks block matrix helpers used by
scalar and vector-valued operators. There is no PDE equation; the invariant is
that scalar ones are `npt x npt` and normal-vector ones are
`2 npt x 2 npt`. The method directly calls `onesmat` and `normonesmat`.
Ground truth is expected matrix shape.

`test_centroids_and_adjacency_info` checks centroids, sort metadata, component
count, closedness, and adjacency validity on a two-panel refined circle. The
method is `centroids`, `sortinfo`, and `checkadjinfo`. Ground truth is one
closed component, two chunks, identity chunk ordering, adjacency equal to the
stored adjacency, and no adjacency errors.

`test_upsample_preserves_circle_geometry_and_density_values` checks spectral
upsampling from `k=8` to `k=16`. The scalar data equation is
`sigma(t) = 1 + t - 2 t^3`, a polynomial exactly representable by the panel
basis. The method is `Chunker.upsample` for geometry and attached density.
Ground truth is preserved area and exact polynomial values at the upsampled
Legendre nodes.

`test_refine_oversamples_by_splitting_chunks` checks chunk refinement by panel
splitting. The invariant is that one circle panel split with `nover=1` becomes
two panels with closed adjacency, while area and total length are preserved.
The method is `Chunker.refine`. Ground truth is expected adjacency, unchanged
area, and unchanged summed chunk lengths.

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
Ground truth is original geometry in each output chunk, datadim equal to the
maximum input datadim, and the expected padded data arrays.

## `tests/test_chunkerfit.py`

`test_chunkerfit_open_line_with_split_points` checks spline fitting through
collinear points with splits at the input points. The geometry equation is an
open line from `x=0` to `x=3`, total length `3`, with `y=0`. The method is
`chunkerfit(..., splitatpoints=True)` with `ifclosed=False`. Ground truth is
three panels, free-ended adjacency, total length `3`, and zero y-coordinates.

`test_chunkerfit_closed_circle_spline_area` checks closed spline fitting of
16 samples from the unit circle. The equation is the circle area `pi`. The
method is `chunkerfit` with a closed periodic fit and split points. Ground
truth is one panel per input interval and area within a loose spline tolerance
of `2e-3` relative error.

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
and exact circle area and length.

`test_chunkerfunc_open_curve_marks_free_ends` checks open-curve construction
for a line segment from `(0,0)` to `(2,0)`. The equation is total length `2`.
The method is `chunkerfunc` on `curves.linefunc` with `ifclosed=False` and two
chunks. Ground truth is free-ended adjacency at both ends, the first interval
`[0, 0.5]`, and total length `2`.

`test_chunkerfuncuni_builds_requested_uniform_panel_count` checks uniform
panel construction independent of adaptivity. The equation is area
`pi * 1.5^2` for a radius-1.5 circle. The method is `chunkerfuncuni` with six
panels and Legendre order 10. Ground truth is exactly six chunks, `k=10`, and
the analytic area.

`test_chunkerfunc_can_spectrally_differentiate_position_only_curve` checks that
the constructor can infer derivatives when a curve callback returns only
positions. The equation is the unit-circle area `pi`. The method is spectral
differentiation of position data inside `chunkerfunc`. Ground truth is the
correct area after derivative inference.

`test_basic_curve_helpers_match_expected_derivatives` checks the canned
`curves.fsine` helper. The equations are `x(t)=t`, `x'(t)=1`, `x''(t)=0`, and
`y(t)=amp sin(freq t + phase)`. The method directly evaluates the helper.
Ground truth is the known coordinate and derivative formulas.

## `tests/test_chunkerpoly.py`

`test_chunkerpoly_closed_square_area_length_and_adjacency` checks a closed,
unrounded square polygon. The equations are area `1` and perimeter `4`. The
method is `chunkerpoly` with one straight Legendre panel per edge. Ground truth
is four chunks, closed cyclic adjacency, exact area, and exact summed length.

`test_chunkerpoly_open_polyline_and_edge_data` checks an open two-edge
polyline with per-edge data. The equation is total length `2 + 3 = 5`. The
method is `chunkerpoly(..., ifclosed=False)` with `edgevals`. Ground truth is
open adjacency, two data rows, constant data on each corresponding edge, and
the exact total length.

`test_chunkerpoly_rounded_builds_trimmed_edges_and_corner_panels` checks
rounded-polygon construction for a square. The method trims the straight edges
and inserts rounded corner panels of width `0.1`. The geometric invariant is
that the rounded shape remains valid, has positive panel lengths, and has area
between `0.9` and the original square area `1`. Ground truth is eight chunks,
valid adjacency, and the area/length inequalities.

`test_chunkerpoly_rounded_open_polyline_and_edge_data` checks rounded
construction for an open L-shaped polyline with data interpolation through the
rounded corner. The method uses widths `[0, 0.2, 0]` and scalar edge values
`2` and `4`. Ground truth is three chunks, free-ended adjacency, preserved
endpoint-edge data, and middle-panel data bounded between the adjacent edge
values.

`test_reverse_and_move_preserve_expected_geometry` checks orientation reversal
and rigid/scale movement on a square. The equations are `area(reverse) = -area`
and `area(scale * r) = scale^2 area`, with scale `3`. The method is
`reverse` and `move` with translation, rotation, and scaling. Ground truth is
area `-1` after reversal and area `9` after scaling by `3`.

## `tests/test_chunkgraph.py`

`test_chunkgraph_constructs_edges_and_vertex_incidence` checks conversion of a
square graph from vertices and directed edge endpoints. The invariant is the
edge-to-vertex incidence matrix with `-1` at the start and `+1` at the end.
The method is `chunkgraph` construction plus source-info assembly. Ground truth
is four edge chunkers, the explicit `v2emat`, total point count equal to the
sum over edges, source-info shape, and at least two detected regions.

`test_chunkgraph_accepts_incidence_matrix_edges` checks the alternate graph
constructor format where edges are supplied as an incidence matrix. The method
converts incidence columns back to endpoint pairs. Ground truth is the same
square endpoint matrix `[[0,1,2,3],[1,2,3,0]]`.

`test_chunkgraph_slice_and_edgeids_match_selected_edges` checks subgraph
slicing and global point-index selection. The method is `slicegraph([0,1])`
and `edgeids([0,1])`. Ground truth is a two-edge subgraph with the expected
endpoint matrix, and point ids whose global coordinates match the concatenated
coordinates of the selected edge chunkers.

`test_chunkgraph_region_ids_survive_translation` checks region classification
on a square graph and after translation. The geometric invariant is that
inside/outside region ids are translation-invariant when targets translate by
the same vector. The method is `chunkgraphinregion` before and after `cg + a`.
Ground truth is region ids `[2, 1]` for the sample inside/outside targets in
both coordinate systems.

`test_chunkgraph_works_with_dense_operator_helpers` checks that chunkgraphs can
be passed into dense operator assembly/evaluation APIs. The kernel is the
smooth polynomial `1 + dx^2 + dy^2`. The method is `chunkermat` and
`chunkerkerneval` on a graph with a sinusoidal density. Ground truth is the
matrix and value shapes, not an external analytic value.

`test_tochunkgraph_preserves_closed_and_open_components` checks conversion from
ordinary chunkers to graph form. The invariant is that a closed component
becomes a loop edge whose start and end vertex are the same, while an open line
becomes an edge between distinct vertices. The method is `tochunkgraph`.
Ground truth is the loop endpoint condition for the closed square and
`[[0],[1]]` endpoints for the open line.

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
interface helper and the red-white-blue colormap. The method finite-differences
the interface y-coordinate and samples a five-color map. Ground truth is the
analytic first derivative and blue/white/red endpoint colors.

`test_hypoct_uni_builds_zero_based_uniform_tree` checks the top-level
hyperoctree helper on four quadrant points. The method builds `hypoct_uni`
with an explicit square extent. Ground truth is zero-based tree indexing,
two-level structure, one point per child, and sibling neighbor connectivity.

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

`test_arclengthfun_merged_components_devtools_output_matches_matlab` is marked
strict `xfail`. It documents a known parity gap: MATLAB resets arclength per
merged connected component, while Python currently accumulates through the
whole merged chunker. The method and equation are the same cumulative
arclength calculation as the single-component test. Ground truth is the MATLAB
merged-component fixture, and the expected outcome today is failure.

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

`test_stokes_dtrac_devtools_output_matches_matlab` checks the Stokes double
layer traction relation against MATLAB. The continuum equation is the traction
stress identity `t = -p n + mu (grad u + grad u^T) n`, reconstructed from
`dgrad` and `dpres`. The method evaluates `dtrac`, `dgrad`, and `dpres` kernel
blocks through the generic `kernel` wrapper and contracts with saved
strengths. Ground truth is MATLAB's `Kt`, `Kg`, `Kp`, reconstructed traction,
and residual norm below `1e-13`.

## `tests/test_elast2d.py`

`test_elasticity_kernel_shapes_and_factory` checks the 2D linear elasticity
kernel selector and factory metadata. The equations are the Kelvin fundamental
solution and its traction/gradient/double-layer variants for Lame parameters
`lam=1.5`, `mu=2.1`. The method is direct point-kernel evaluation plus
`kernel("elast", ...)`. Ground truth is expected block shape and operator
dimension metadata for vector-valued densities and outputs.

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
is monotonicity: increasing `rho` should not reduce the number of near flags.
The method compares tight and padded rectangle flags on an open polyline, then
checks `tochunkgraph` delegates to chunker implementations. Ground truth is
shape, monotonic flag count, and equality between graph and chunker results.

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

## `tests/test_helm1d.py`

`test_helm1d_green_gradient_matches_finite_difference` checks the flat-interface
Helmholtz helper `G = exp(i k |x-y|)`. The method evaluates `helm1d.green` and
compares the target `x` gradient to a centered finite difference. Ground truth
is the finite-difference derivative plus value and Hessian shape.

`test_helm1d_kernel_selectors_and_kernel_wrapper` checks 1D Helmholtz selector
plumbing for single layer, double layer, target-normal derivative,
transmission block, and full 2-by-2 transmission system. The method is direct
point-kernel evaluation and generic `kernel("helm1d", ...)`. Ground truth is
expected matrix shapes and singularity metadata `removable` for the single
layer.

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
with dense direct summation and with `{"usefmm": True}`. Ground truth is exact
agreement with the dense direct reference and non-null FMM metadata.

`test_fmm2dpy_laplace_gradient_and_helmholtz_layers_match_direct` checks
`fmm2dpy` acceleration for Laplace gradients/double-layer variants and
Helmholtz single, double, and gradient layers. The equations are the Laplace
and Helmholtz layer potentials applied to `cos(x)` density on a circle. The
method compares FMM evaluation at several targets against dense direct
evaluation with tolerance `1e-9`. Ground truth is the direct dense path.

`test_helmholtz_double_gradient_fmm_requests_dipole_gradients` checks the
new Helmholtz double-gradient FMM wiring with a fake `fmm2dpy` module. The
method builds `kernel("helm","dgrad",zk)`, calls its FMM evaluator, and asserts
that `hfmm2d` receives dipole strengths, source normals, and `pgt=2`. Ground
truth is the fake module's `gradtarg` returned in Fortran target ordering.

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
Ground truth is the closed-form value plus expected gradient and Hessian
shapes.

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
`pointinfo`. Ground truth is expected matrix shapes for two targets and all
source nodes.

`test_helmholtz_green_gradient_matches_finite_difference` checks the 2D
Helmholtz Green's function derivatives. The method evaluates `helm2d.green`
at complex wavenumber `1.2+0.4i` and compares the target `x` derivative to a
centered finite difference. Ground truth is the finite-difference gradient
plus expected value and Hessian shapes.

`test_helmholtz_kernel_selectors_have_expected_shapes` checks 2D Helmholtz
single, double, target-normal derivative, and combined selectors. The method is
direct `helm2d.kern` on point-info derived from a circle. Ground truth is the
expected target-by-source matrix shapes.

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
is the expected length and alternating sign pattern of barycentric weights on
ordered Legendre nodes.

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
barycentric weights. The method recomputes every object in Python. Ground truth
is `tests/golden/lege_extended.mat`.

`test_chunker_geometry_and_transforms_match_matlab_fixture` checks MATLAB
parity for chunker geometry, move/transform operations, and dense helper
matrices. The equations are affine geometry transforms, determinant area
scaling, spectral differentiation, cumulative integration, scalar/vector ones
matrices, and centroids. The method reconstructs a MATLAB-saved chunker,
applies Python transforms, and compares helper matrices. Ground truth is
`tests/golden/chunker_ops.mat`.

`test_laplace_point_kernels_match_matlab_fixture` is parametrized over
`s`, `d`, `sp`, `stau`, `hilb`, `sgrad`, `dgrad`, `dp`, `c`, `cp`, and
`cgrad`. The equations are the 2D Laplace Green's function, normal/tangent
projections, Hessian projections, Hilbert-style kernel, and combined-layer
linear combinations. The method is direct point-kernel evaluation through
`lap2d.kern`. Ground truth is `tests/golden/kernel_pointinfo.mat`.

`test_helmholtz_2d_point_kernels_match_matlab_fixture` is parametrized over
`s`, `d`, `sp`, `stau`, `sgrad`, `dgrad`, `dp`, `c`, and `cp`. The equations
are the 2D Helmholtz Green's function and the same normal, tangent, gradient,
and combined projections. The method is direct `helm2d.kern` with saved
wavenumber and coefficient data. Ground truth is the MATLAB point-kernel
fixture.

`test_helmholtz_1d_point_kernels_match_matlab_fixture` is parametrized over
`s`, `d`, `sp`, `stau`, `dp`, `c`, `cp`, `c2trans`, `all`, `trans_rep`,
`trans_rep_prime`, and `trans_rep_grad`. The equation is
`exp(i k |x-y|)` and transmission-system blocks assembled from its values,
normal derivatives, and gradients. The method is direct `helm1d.kern` with
selector-specific coefficients. Ground truth is the MATLAB point-kernel
fixture.

`test_stokes_point_kernels_match_matlab_fixture` is parametrized over `s`,
`spres`, `strac`, `d`, `dpres`, `dtrac`, `sgrad`, `dgrad`, and `c`. The
equations are 2D Stokes velocity, pressure, traction, gradient, and combined
layer blocks at viscosity `mu`. The method is direct `stok2d.kern`. Ground
truth is the MATLAB point-kernel fixture.

`test_elasticity_point_kernels_match_matlab_fixture` is parametrized over
`s`, `strac`, `d`, and `dalt`. The equations are 2D linear elasticity
single-layer, traction, double-layer, and alternate double-layer blocks for
saved Lame parameters. The method is direct `elast2d.kern`. Ground truth is
the MATLAB point-kernel fixture.

`test_dense_native_operator_paths_match_matlab_fixture` checks dense operator
assembly and evaluation for Laplace and Stokes. The matrix equation is
`M_ij = K(x_i,y_j) w_j`; application is `M sigma`. The method uses
`chunkermat`, `chunkerkernevalmat`, `chunkerkerneval`, and
`quadnative.buildmat`. Ground truth is `tests/golden/operator_parity.mat`,
including matrices and applied values.

`test_quadggq_special_quadrature_matches_matlab_fixture` checks GGQ special
quadrature parity. The method compares log, PV, and HS auxiliary quadrature
tables and special matrices for Laplace single layer, single-layer gradient,
and double-layer gradient. It also checks the `ilist` path that intentionally
skips selected special blocks. Ground truth is `tests/golden/quadggq.mat`,
including exact table entries, matrix entries, and the skipped-block pattern.

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
is equality between the two direct routes.

`test_pointinfo_uses_matlab_chunk_contiguous_ordering` checks point ordering
when flattening chunker fields. The invariant is MATLAB/Fortran chunk-contiguous
ordering: all nodes of chunk 0, then all nodes of chunk 1, and so on. The
method is `pointinfo(chnkr)`. Ground truth is equality of the first two blocks
with the original chunk arrays.

`test_chunkerkernevalmat_matches_direct_target_evaluation` checks target
evaluation matrix assembly. The equation is `values = EvalMat sigma` for the
same smooth polynomial kernel. The method compares `chunkerkernevalmat @ dens`
to `chunkerkerneval`. Ground truth is direct evaluation equality.

`test_chunkermat_accepts_kernel_objects` checks that `chunkermat` accepts a
`Kernel` object, not only callables. The method uses `kernel("zero")`.
Ground truth is an `npt x npt` zero matrix.

`test_quadnative_buildmat_matches_dense_chunkermat` checks that the lower-level
native quadrature builder agrees with the public dense matrix builder for a
smooth scalar kernel. The equation is the same weighted direct matrix
`K w`. The method compares `quadnative.buildmat` to `chunkermat`. Ground truth
is equality of the two dense matrices.

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

## `tests/test_quadggq.py`

`test_matlab_log_quadrature_tables_load_for_each_legendre_node` checks log GGQ
table loading/generation for Legendre order 8. The method is `quadggq.setup`,
`getlogquad`, and `logavail`. Ground truth is known table entries, one self
rule per Legendre node, interpolation matrix shapes, no quadrature node exactly
at the singular target node, and self-rule weights summing to `2`.

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
kernel at wavenumber `1.3+0.2i`. The method is `quadggq.buildmat`.
Ground truth is that the matrix is complex-valued and finite.

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

`test_pv_and_hs_ggq_tables_are_available_for_matlab_orders` checks PV and
hypersingular table availability. The method is `hqsuppavail`, `getpvquad`,
and `gethsquad`. Ground truth is order-8 support and known first table entries
for PV and HS rules.

`test_setup_accepts_pv_and_hs_singularities` checks the `setup` dispatcher for
principal-value and hypersingular auxiliary quadrature. The method calls
`quadggq.setup(8, "pv")` and `quadggq.setup(8, "hs")`. Ground truth is stored
type metadata and one self rule per Legendre node.

`test_chunkermat_uses_special_quadrature_for_pv_and_hs_kernels` checks public
dispatch for Laplace `sgrad` and `dgrad`, which are marked PV and HS
respectively. The method is `chunkermat` on a circle. Ground truth is finite
matrices with shape `2*npt x npt`.

`test_quadadap_buildmat_delegates_to_special_quadrature` checks that the
adaptive quadrature facade currently delegates to GGQ special quadrature for
log kernels. The method compares `quadadap.buildmat(..., sing="log")` to
`quadggq.buildmat(..., type="log")`. Ground truth is matrix equality.

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
matrix dimensions, index bounds, `ilist` columns, and half-sized first-level
index sets.

`test_schurbana_matches_direct_block_formula_shapes` checks the Schur
complement assembly helper used inside RCIP. The equation is the block
compression/Schur update applied to randomly generated matrices with a
well-conditioned good block. The method is `rcip.SchurBana`. Ground truth is a
finite output of expected `nbad x nbad` shape; this is mostly a shape and
smoke test.

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
compression matrix, expected saved-array counts, and interpolated density,
source-info, and weight shapes.

## `tests/test_smoother.py`

`test_smoother_uniform_mesh_matches_polygon_edges` checks the uniform mesh
preprocessing used by the rounded polygon smoother. The equations are unit edge
lengths and outward normals for a unit square; the first edge centroid is
`(0.5,0)` with normal `(0,-1)`. The method is `smoother.get_umesh`. Ground
truth is edge lengths, centroid, face normal, and unit pseudo-normal magnitude.

`test_smoother_get_mesh_expands_legendre_panels` checks expansion of a uniform
polygon mesh into Legendre panel nodes. The method is `smoother.get_mesh` with
two panels per edge and order 5. Ground truth is the expected flattened shapes
for points, normals, pseudo-normals, and weights, plus weight sum equal to the
original edge-length sum.

`test_smoother_smooth_returns_rounded_chunker_and_error_outputs` checks the
high-level smoother path. The method is `smoother.smooth` on a square with
width `0.1` and `return_error=True`, internally producing a rounded chunker.
Ground truth is eight chunks, zero reported smoothing error in the current
implementation, per-point error shape, and area greater than `0.9`.

## `tests/test_sortinfo.py`

`test_sort_reorders_two_open_segments_and_remaps_adjacency` checks sorting of
open chunks after their storage order has been manually reversed. The invariant
is that sorting restores geometric order and remaps adjacency consistently.
The method is `Chunker.sort`. Ground truth is success code `ier=0`, adjacency
equal to the original chunker, and restored coordinates.

## `tests/test_spcl.py`

`test_absconvgauss_derivatives_match_finite_differences` checks the smoothed
absolute-value convolution helper. The method evaluates value, first
derivative, and second derivative, then compares derivatives against centered
finite differences of the lower-order outputs. Ground truth is finite
difference agreement and value shape equal to the input grid.

## `tests/test_stok2d.py`

`test_stokes_kernel_shapes` checks Stokes selector and factory dimensions. The
equations are the 2D Stokes single-layer velocity, single-layer pressure,
single-layer gradient, double-layer traction, combined pressure, and combined
gradient blocks. The method is direct `stok2d.kern` evaluation and
`kernel("stok", ...)` metadata inspection. Ground truth is expected block
shapes and `opdims`.

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
