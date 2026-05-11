# Easy Parity Test Tracker

This file tracks parity tests that are currently too easy, too dependent on
local fixture files, or broad by test count but narrow by behavior.

Status legend:

- `failing`: should fail until the listed gap is resolved.
- `todo`: active parity-hardening work.
- `ignored`: intentionally not part of the current parity-hardening push.
- `done`: fixed or covered by a stronger test.

## Fixture Availability Gate

Large `.mat` parity snapshots can stay ignored by git. Missing fixture data
should still be a test failure, not a skip, so a clean test run reports the
missing data file as the actionable reason.

| Item | Status | Current problem | Next step |
| --- | --- | --- | --- |
| `tests/test_devtools_parity.py` using `tests/golden/devtools_easy.mat` | failing | The fixture is large and git-ignored, so clean checkouts do not have the data needed for the devtools parity tests. | Keep the file ignored, but require local regeneration before running these tests; split compact high-value cases into tracked fixtures when practical. |
| `tests/test_matlab_parity.py::test_chunker_geometry_and_transforms_match_matlab_fixture` using `tests/golden/chunker_ops.mat` | failing | The fixture is large and git-ignored, so the chunker ops parity check can be missing on clean checkouts. | Keep the file ignored, but fail directly when missing; consider a smaller tracked fixture for core transform coverage. |
| Other tracked parity fixtures under `tests/golden` | failing if missing | These files are expected repository test data. | Missing tracked fixtures should fail with a direct message naming the missing file. |

## Too-Easy Parity Tests

| Area | Status | Current weakness | Stress-test direction |
| --- | --- | --- | --- |
| Point-kernel MATLAB parity in `tests/test_matlab_parity.py` | done | Many selectors are covered, but by one small fixed source/target fixture. | Added `tests/test_easy_parity_stress.py::test_point_kernels_stress_combined_selectors_and_green_gradients` with multi-point, non-axis-aligned source/target data, varied normals/tangents, combined Laplace/Helmholtz selectors, and finite-difference Green-gradient checks. |
| Dense/native operator parity in `tests/test_matlab_parity.py` | done | Uses a small unit-circle fixture and limited targets. | Added `tests/test_easy_parity_stress.py::test_dense_native_operator_stress_on_wobbly_curve_matches_manual_weighting` with a noncircular wobbly curve, vector-valued smooth kernel, target evaluations, matrix assembly, and weighted manual references. |
| Section III quadrature parity in `tests/test_matlab_parity.py` and `tests/test_quadggq.py` | done | Some Python tests only check finite outputs, shape, or equality to another Python path. | Added `tests/test_easy_parity_stress.py::test_quadggq_stress_noncircle_complex_special_blocks_and_robust_close_eval` with noncircle complex Helmholtz self/neighbor/far block checks, ignored-source behavior, and robust close replacement on nearby wobbly curves. |
| RCIP unit coverage in `tests/test_rcip.py` | done | `test_schurbana_matches_direct_block_formula_shapes` is mostly a shape test and has a tautological `out == out` assertion. | Added `tests/test_easy_parity_stress.py::test_schurbana_stress_matches_independent_block_update` with a rectangular-block Schur-Banachiewicz reference formula and input immutability check. |
| RCIP parity in `tests/test_matlab_parity.py` / `tests/test_rcip_parity.py` | done | Covers helper/compression outputs for a two-edge Laplace corner, not full solve parity. | Added `tests/test_easy_parity_stress.py::test_chunkgraph_rcip_stress_nonorthogonal_vertex_and_global_blocks` with a nonorthogonal graph, global block-kernel subselection, ignored-vertex handling, recursive compression, and interpolation checks. |
| FMM parity-style tests in `tests/test_kernel.py` and `tests/test_operators.py` | done | Mostly compare FMM paths to Python dense/direct paths on simple circle geometries. | Added `tests/test_easy_parity_stress.py::test_interleaved_fmm_stress_matches_direct_on_wobbly_curve` with a wobbly source curve, off-boundary targets, complex/vector densities, and interleaved Laplace/Helmholtz kernel blocks. |
| Smoother devtools parity | done | Current check is diagnostic: error thresholds and output shapes, not MATLAB geometry equivalence. | Added `tests/test_easy_parity_stress.py::test_smoother_stress_returns_valid_rounded_asymmetric_polygon` with asymmetric widths and polygon geometry, area, adjacency, normals, chunk lengths, and per-node error checks for the supported lightweight smoother. |

## Ignored For Now

| Area | Status | Reason |
| --- | --- | --- |
| `chunkerfit` devtools parity | ignored | Not part of the current hardening push. Existing coverage only checks adjacency/status and is not worth upgrading now. |
| `chunkerpoly` devtools parity | ignored | Not part of the current hardening push. Existing coverage checks adjacency/diagnostics more than full MATLAB geometry. |
