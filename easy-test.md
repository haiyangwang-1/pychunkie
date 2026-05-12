# Easy Parity Test Tracker

This file tracks parity tests that are currently too easy, too dependent on
local fixture files, or broad by test count but narrow by behavior.

Status legend:

- `failing`: should fail until the listed gap is resolved.
- `todo`: active parity-hardening work.
- `ignored`: intentionally not part of the current parity-hardening push.
- `done`: fixed or covered by a stronger test.

## Active Too-Easy Tests

No active too-easy rows remain from the 2026-05-12 audit. Keep new findings in
this table until they are fixed, then move them to the completed table below.

| Test or area | Status | Current weakness | Hardening direction |
| --- | --- | --- | --- |

## Fixture Availability Gate

Large `.mat` parity snapshots can stay ignored by git. Missing fixture data
should still be a test failure, not a skip. The current fixture helper
generates ignored fixtures on demand and fails the requesting test if MATLAB
setup or fixture generation fails.

| Item | Status | Current behavior | Follow-up |
| --- | --- | --- | --- |
| `tests/test_devtools_parity.py` using `tests/golden/devtools_easy.mat` | done | The ignored fixture is generated on demand by `tests/_fixture_generation.py`; generation failure is reported as a test failure. | Split compact high-value cases into tracked fixtures when practical. |
| `tests/test_matlab_parity.py::test_chunker_geometry_and_transforms_match_matlab_fixture` using `tests/golden/chunker_ops.mat` | done | The ignored fixture is generated on demand by `tests/_fixture_generation.py`; missing or failed generation is not skipped. | Consider a smaller tracked fixture for core transform coverage. |
| Other tracked parity fixtures under `tests/golden` | done | Tracked fixture files are expected repository test data and missing data fails with a direct message. | Keep missing tracked fixtures as direct failures. |

## Completed Too-Easy Test Fixes

These rows were previously marked `done` in the active parity table. They are
moved here so active triage stays focused on unfinished work.

| Area | Status | Previous weakness | Completed hardening |
| --- | --- | --- | --- |
| Point-kernel MATLAB parity in `tests/test_matlab_parity.py` | done | Many selectors were covered, but by one small fixed source/target fixture. | Added `tests/test_easy_parity_stress.py::test_point_kernels_stress_combined_selectors_and_green_gradients` with multi-point, non-axis-aligned source/target data, varied normals/tangents, combined Laplace/Helmholtz selectors, and finite-difference Green-gradient checks. |
| Dense/native operator parity in `tests/test_matlab_parity.py` | done | Used a small unit-circle fixture and limited targets. | Added `tests/test_easy_parity_stress.py::test_dense_native_operator_stress_on_wobbly_curve_matches_manual_weighting` with a noncircular wobbly curve, vector-valued smooth kernel, target evaluations, matrix assembly, and weighted manual references. |
| Section III quadrature parity in `tests/test_matlab_parity.py` and `tests/test_quadggq.py` | done | Some Python tests only checked finite outputs, shape, or equality to another Python path. | Added `tests/test_easy_parity_stress.py::test_quadggq_stress_noncircle_complex_special_blocks_and_robust_close_eval` with noncircle complex Helmholtz self/neighbor/far block checks, ignored-source behavior, and robust close replacement on nearby wobbly curves. |
| RCIP unit coverage in `tests/test_rcip.py` | done | `test_schurbana_matches_direct_block_formula_shapes` was mostly a shape test and had a tautological `out == out` assertion. | Added `tests/test_easy_parity_stress.py::test_schurbana_stress_matches_independent_block_update` with a rectangular-block Schur-Banachiewicz reference formula and input immutability check. |
| RCIP parity in `tests/test_matlab_parity.py` / `tests/test_rcip_parity.py` | done | Covered helper/compression outputs for a two-edge Laplace corner, not full solve parity. | Added `tests/test_easy_parity_stress.py::test_chunkgraph_rcip_stress_nonorthogonal_vertex_and_global_blocks` with a nonorthogonal graph, global block-kernel subselection, ignored-vertex handling, recursive compression, and interpolation checks. |
| FMM parity-style tests in `tests/test_kernel.py` and `tests/test_operators.py` | done | Mostly compared FMM paths to Python dense/direct paths on simple circle geometries. | Added `tests/test_easy_parity_stress.py::test_interleaved_fmm_stress_matches_direct_on_wobbly_curve` with a wobbly source curve, off-boundary targets, complex/vector densities, and interleaved Laplace/Helmholtz kernel blocks. |
| Smoother devtools parity | done | The check was diagnostic: error thresholds and output shapes, not MATLAB geometry equivalence. | Added `tests/test_easy_parity_stress.py::test_smoother_stress_returns_valid_rounded_asymmetric_polygon` with asymmetric widths and polygon geometry, area, adjacency, normals, chunk lengths, and per-node error checks for the supported lightweight smoother. |

## Completed Follow-up Quality Pass - 2026-05-12

| Test | Status | Previous weak assertion pattern | Completed hardening |
| --- | --- | --- | --- |
| `tests/test_operators.py::test_chunkermat_matches_chunkerkerneval_on_boundary_for_smooth_kernel` | done | Compared two public direct routes without an independent weighted-kernel value. | Adds an explicit raw smooth-kernel matrix-vector reference using source quadrature weights. |
| `tests/test_operators.py::test_chunkerkernevalmat_matches_direct_target_evaluation` | done | Compared evaluation matrix application to direct evaluation without checking matrix entries. | Compares the evaluation matrix to the raw target/source kernel block weighted by source quadrature weights. |
| `tests/test_operators.py::test_quadnative_buildmat_matches_dense_chunkermat` | done | Compared lower-level and public dense assembly paths only. | Compares native assembly to the explicit raw smooth-kernel matrix weighted by source quadrature weights. |
| `tests/test_operators.py::test_chunkerinterior_fmm_matches_direct_with_close_correction` | done | Verified FMM/direct agreement without asserting the expected inside/outside truth. | Asserts exact circle membership for inside, outside, just-inside, and just-outside targets before comparing the FMM route. |
| `tests/test_chunkerfunc.py::test_chunkerfunc_adaptively_refines_unresolved_curve` | done | Checked that refinement happened and endpoints survived, but not that the refined geometry was numerically accurate. | Adds contiguous interval checks and compares refined arclength to a high-order independent Gauss-Legendre reference. |
| `tests/test_chunkgraph.py::test_tochunkgraph_preserves_closed_and_open_components` | done | Checked graph endpoint topology without proving the converted graph retained source geometry values. | Adds point-count, merged-coordinate, weight, edge-chunker storage, and vertex-coordinate preservation checks. |
| `tests/test_domain.py::test_hypoct_uni_builds_zero_based_uniform_tree` | done | Checked tree levels and neighbor counts, but not exact child geometry or connectivity. | Adds exact child-center, point-assignment, and sibling-neighbor-list assertions. |
| `tests/test_kernels.py::test_laplace_green_matches_direct_formula` | done | Checked the Green value but only derivative tensor shapes. | Adds closed-form Laplace gradient and Hessian assertions. |
| `tests/test_kernels.py::test_helmholtz_green_gradient_matches_finite_difference` | done | Checked only target-x gradient and Hessian shape. | Adds target-y gradient and Hessian finite-difference checks. |
| `tests/test_helm1d.py::test_helm1d_green_gradient_matches_finite_difference` | done | Checked only target-x gradient and Hessian shape. | Adds target-y gradient and Hessian finite-difference checks. |
| `tests/test_operators.py::test_pointinfo_uses_matlab_chunk_contiguous_ordering` | done | Checked only the first two flattened position blocks. | Compares full flattened `r`, `d`, `d2`, and normal arrays against chunk-contiguous storage. |
| `tests/test_spcl.py::test_absconvgauss_derivatives_match_finite_differences` | done | Checked derivative consistency but left the smoothed value as a shape-only assertion. | Compares value, first derivative, and second derivative to the closed-form Gaussian-convolution formulas while retaining finite-difference checks. |
| `tests/test_chunkgraph.py::test_chunkgraph_constructs_edges_and_vertex_incidence` | done | Checked incidence, counts, source-info shape, and a loose region count without proving the graph retained the expected edge geometry. | Adds straight-edge node, derivative, normal, weight, source-info flattening, vertex-structure, and exact region-loop assertions. |
| `tests/test_chunker.py::test_matrix_transform_updates_derivatives_normals_and_weights` | done | The test name covered normals and weights, but assertions only checked transformed coordinates, first derivatives, and area. | Adds transformed second derivatives plus recomputed normal and quadrature-weight assertions. |
| `tests/test_chunkerfunc.py` circle/line construction smoke tests | done | Several constructor tests checked only panel counts, adjacency, area, or length, leaving pointwise geometry and derivative scaling unchecked. | Adds exact per-panel positions, derivatives, normals, and quadrature weights for analytic circle, uniform circle, and open line cases. |
| `tests/test_chunkerfunc.py::test_basic_curve_helpers_match_expected_derivatives` | done | Checked the sine helper's x-coordinate derivatives and y-coordinate value, but not the y-derivatives. | Adds first and second y-derivative formula checks. |
| `tests/test_chunkerfit.py::test_chunkerfit_open_line_with_split_points` | done | Checked split count, adjacency, total length, and zero y-coordinate, but not the fitted x-nodes or differential geometry. | Adds exact per-panel line positions, derivatives, normals, and quadrature weights. |
| `tests/test_chunkerpoly.py` unrounded polygon and transform checks | done | Plain polygon tests relied on area, length, adjacency, and data checks without pointwise geometry; reverse/move only checked area. | Adds straight-edge positions, derivatives, normals, weights, plus reversed and moved storage assertions. |
| `tests/test_chunker.py` translation/scaling/rotation/reflection checks | done | Transform tests did not cover all stored derivative, normal, and weight fields. | Adds translated, scaled, rotated, and reflected storage assertions across coordinates, derivatives, normals, and weights. |
| `tests/test_chunker.py::test_upsample_preserves_circle_geometry_and_density_values` | done | Checked upsampled area and attached density but not the upsampled geometry fields. | Uses higher panel orders and compares upsampled positions, derivatives, normals, and weights to analytic circle formulas. |
| `tests/test_chunker.py::test_chunker_spectral_helpers_on_circle` | done | Verified expansion output shapes without proving the coefficients reconstructed the stored geometry. | Reconstructs position, derivative, and second-derivative arrays from the Legendre coefficients. |
| `tests/test_chunker.py::test_merge_combines_chunkers_and_pads_data_rows` | done | Checked merged coordinates and data padding, but not derivative, normal, or weight fields. | Adds per-chunk derivative, second-derivative, normal, and weight preservation checks. |
| `tests/test_chunker.py::test_refine_oversamples_by_splitting_chunks` | done | Checked split count, adjacency, area, and length, but not the geometry of the split panels. | Adds analytic half-circle position, derivative, normal, and weight checks for both refined panels. |
| `tests/test_domain.py::test_nonflatinterface_derivatives_and_redblue_colormap` | done | Used finite differences for the y-derivative but did not check the closed-form y-coordinate or second derivative. | Adds analytic y-coordinate, first-derivative, and second-derivative assertions. |
| `tests/test_chunkgraph.py::test_chunkgraph_accepts_incidence_matrix_edges` | done | Checked only endpoint conversion from incidence-matrix input. | Adds incidence preservation, vertex-structure, and exact region-loop assertions. |
| `tests/test_chunkgraph.py::test_chunkgraph_slice_and_edgeids_match_selected_edges` | done | Checked selected coordinate lookup but not exact edge IDs or subgraph differential fields. | Adds exact global ID, subgraph coordinate, derivative, and weight assertions. |
| `tests/test_sortinfo.py::test_sort_reorders_two_open_segments_and_remaps_adjacency` | done | Checked restored adjacency and coordinates only. | Adds restored derivative, second-derivative, normal, and weight checks. |

## Completed Full Test Ease Audit - 2026-05-12

Scope: project-owned `tests/test_*.py` files only. Vendored tests under
`external/` are out of scope. `uv run pytest --collect-only -q` collected 322
pytest cases; parametrized parity tests were audited once per test function.

Marking rule: a test was marked `too easy` when the behavior under test was
numerical or geometric, but the assertions would still pass with wrong values
because they only checked shape, finiteness, coarse bounds, dispatch, or a
tautology. Narrow API/validation tests were left unmarked when that narrow
contract was the point of the test.

| Test | Status | Previous weak assertion pattern | Completed hardening |
| --- | --- | --- | --- |
| `tests/test_biharm2d.py::test_biharmonic_kernel_selectors_and_factory_shapes` | done | Only selector output shapes and one `opdims` value were checked. | Compares `s`, `d`, `sp`, `sgrad`, and `shess` against `biharm2d.green` projections. |
| `tests/test_biharm2d.py::test_biharmonic_layer_evaluation_uses_special_quadrature` | done | Only `(1, 1)` shape and finite output were checked. | Checks the unit-density circle value at an interior target against the analytic `|x|^2/4` identity. |
| `tests/test_chunker.py::test_onesmat_and_normonesmat_shapes` | done | Only matrix dimensions were checked. | Asserts scalar entries and normal-weighted block entries from chunker weights and normals. |
| `tests/test_chunker.py::test_centroids_and_adjacency_info` | done | Centroids were only checked for shape; adjacency/sort metadata was checked. | Compares centroid coordinates to weighted panel means. |
| `tests/test_chunkerfit.py::test_chunkerfit_closed_circle_spline_area` | done | Panel count plus loose area tolerance could pass with distorted geometry. | Checks total length, node radii, tangent/radius orthogonality, area, and closed adjacency. |
| `tests/test_chunkerfunc.py::test_chunkerfunc_can_spectrally_differentiate_position_only_curve` | done | The test claimed derivative inference but only checked area. | Asserts inferred derivatives, second derivatives, normals, weights, and area against circle identities. |
| `tests/test_chunkerpoly.py::test_chunkerpoly_rounded_builds_trimmed_edges_and_corner_panels` | done | Checked chunk count, adjacency status, area bounds, and positive lengths only. | Verifies trimmed straight-edge nodes plus rounded-corner geometry, derivatives, and second derivatives. |
| `tests/test_chunkerpoly.py::test_chunkerpoly_rounded_open_polyline_and_edge_data` | done | Edge data and free-end adjacency were checked, but rounded geometry was not. | Checks open rounded-panel geometry and interpolated edge data. |
| `tests/test_chunkgraph.py::test_chunkgraph_works_with_dense_operator_helpers` | done | Dense matrix and evaluation outputs were checked only for shape. | Compares graph matrix/evaluation values to manual weighted references. |
| `tests/test_devtools_parity.py::test_chunkerfit_devtools_outputs_match_matlab` | done | Reconstructed `chunkerfit` outputs were not compared to MATLAB fields. | Compares closed/open fitted chunker fields, weights, normals, adjacency, chunk lengths, and area to the devtools fixture. |
| `tests/test_devtools_parity.py::test_chunkerpoly_devtools_outputs_match_matlab` | done | Only status, adjacency status, and true-polygon scalar errors were checked. | Checks true-polygon area/length, data dimensions, rounded chunk count and lengths, and open lightweight adjacency/length invariants. |
| `tests/test_devtools_parity.py::test_smoother_devtools_output_matches_matlab_thresholds` | done | Only error thresholds, fixture size arithmetic, and error-vector shapes were checked. | Checks supported lightweight smoother geometry invariants: zero errors, `2*nv` chunks, adjacency, unit normals, and positive chunk lengths. |
| `tests/test_elast2d.py::test_elasticity_kernel_shapes_and_factory` | done | Elasticity selector coverage was shape and metadata only. | Adds explicit formula checks for `s`, `d`, and `dalt`, finite-difference `sgrad`, and stress-traction contractions for `strac` and `dalttrac`. |
| `tests/test_flam.py::test_flam_proxy_square_geometry_and_proxyfun_shapes` | done | Proxy geometry had value checks, but `proxyfun` only checked result shape and neighbor echo. | Compares proxy/source matrix blocks to direct kernel evaluations. |
| `tests/test_flam.py::test_flam_proxyfunr_column_and_row_shapes` | done | Only column/row result shapes and neighbor arrays were checked. | Compares both rectangular proxy callback modes to direct proxy-target/source blocks. |
| `tests/test_geometry.py::test_flagnear_rectangle_uses_per_chunk_padding_and_chunkgraph_delegates` | done | Padding was checked by monotonic count and delegation, not by expected flags. | Asserts exact tight and padded flag masks for chosen points. |
| `tests/test_helm1d.py::test_helm1d_kernel_selectors_and_kernel_wrapper` | done | Selector outputs were checked only for shape and one singularity string. | Compares `s`, `d`, `dp`, `c2trans`, and `all` blocks against direct `helm1d.green` formulas. |
| `tests/test_kernels.py::test_laplace_kernel_selectors_have_expected_shapes` | done | Laplace selector outputs were shape-only. | Compares selector values against Green-function values, gradients, Hessians, and normal projections. |
| `tests/test_kernels.py::test_helmholtz_kernel_selectors_have_expected_shapes` | done | Helmholtz selector outputs were shape-only. | Compares selector values against Helmholtz Green values, gradients, Hessians, and normal projections. |
| `tests/test_lege.py::test_barywts_reproduce_lagrange_basis_sign_pattern` | done | Shape and alternating signs did not validate barycentric weights. | Compares normalized weights to an independent product formula and verifies polynomial interpolation. |
| `tests/test_quadggq.py::test_quadggq_handles_complex_helmholtz_single_layer_blocks` | done | Only complex dtype and finite entries were checked. | Compares complex GGQ assembly to adaptive quadrature. |
| `tests/test_quadggq.py::test_setup_accepts_pv_and_hs_singularities` | done | Only auxiliary type and rule count were checked. | Checks representative PV/HS nodes, weights, and constant-preserving interpolation matrices. |
| `tests/test_quadggq.py::test_chunkermat_uses_special_quadrature_for_pv_and_hs_kernels` | done | Only matrix shapes and finiteness were checked. | Compares self special blocks to PV/HS topological matrices, far blocks to smooth assembly, and confirms nonzero special blocks. |
| `tests/test_quadggq.py::test_quadadap_robust_mode_repairs_non_neighbor_close_blocks` | done | Only finite output and the presence of a non-panel-sized adaptive call were checked. | Compares robust and standard matrices and asserts a nontrivial close-block replacement. |
| `tests/test_rcip.py::test_setup_returns_zero_based_rcip_indices_and_block_shapes` | done | Mostly shape, range, and two `ilist` columns; `Pbc`/`PWbc` values were unchecked. | Compares prolongation matrices to explicit `IPinit` Kronecker constructions. |
| `tests/test_rcip.py::test_schurbana_matches_direct_block_formula_shapes` | done | Ended with `np.testing.assert_allclose(out, out)`, a tautology. | Replaces it with an independent Schur-Banachiewicz block formula. |
| `tests/test_rcip.py::test_rcompchunk_runs_recursive_compression_for_corner_edges` | done | Nontrivial RCIP output was checked only for shape, finite entries, saved counts, and non-identity. | Adds deterministic trace, Frobenius norm, and leading-row value checks. |
| `tests/test_rcip.py::test_chunkgraph_rcip_runs_selected_vertices_and_ignores_marked_vertices` | done | Selected vertices and finite compression matrices were checked, but matrix values were not. | Compares each selected vertex result against direct `Rcompchunk` output. |
| `tests/test_rcip.py::test_chunkgraph_rcip_subselects_global_block_kernels` | done | Global block-kernel subselection shape was checked without a value invariant. | Asserts zero-kernel compression returns identity and only incident block kernels are used. |
| `tests/test_smoother.py::test_smoother_get_mesh_expands_legendre_panels` | done | Only array shapes and total weight sum were checked. | Checks first-panel Legendre coordinates, normals, pseudo-normals, and weights. |
| `tests/test_smoother.py::test_smoother_smooth_returns_rounded_chunker_and_error_outputs` | done | Checked chunk count, zero error, error-vector shape, and coarse area lower bound. | Asserts rounded-polygon geometry, adjacency, normals, chunk lengths, and area bounds. |
| `tests/test_stok2d.py::test_stokes_kernel_shapes` | done | Stokes selector coverage was shape and metadata only. | Compares single-layer velocity and pressure formulas plus finite-difference `sgrad`. |

Reviewed but not newly marked:

- `tests/test_matlab_parity.py`, `tests/test_matlab_fixtures.py`,
  `tests/test_geometry_parity.py`, and `tests/test_rcip_parity.py` are mostly
  strict fixture/value comparisons. Some are narrow fixtures, but they are not
  shape-only smoke tests.
- `tests/test_easy_parity_stress.py` exists specifically to harden several
  previously easy parity paths. It still leaves exact smoother/RCIP solve
  parity room for future work, but its assertions are not merely shape or
  finiteness checks.
- Dense-vs-FMM and dense-vs-FLAM cross-checks are not independent analytical
  proofs, but they compare numerical values across different execution paths
  and were not marked by this audit.
