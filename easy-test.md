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
| Point-kernel MATLAB parity in `tests/test_matlab_parity.py` | todo | Many selectors are covered, but by one small fixed source/target fixture. | Add fixtures with more sources/targets, near-singular separations, varied normals/tangents, multiple wavenumbers/coefficient sets, and non-axis-aligned geometry. |
| Dense/native operator parity in `tests/test_matlab_parity.py` | todo | Uses a small unit-circle fixture and limited targets. | Add noncircular starfish or wobbly-curve chunkers, close targets, multiple densities, and vector-valued matrix applications. |
| Section III quadrature parity in `tests/test_matlab_parity.py` and `tests/test_quadggq.py` | todo | Some Python tests only check finite outputs, shape, or equality to another Python path. | Add MATLAB fixture comparisons for complex Helmholtz, PV, HS, robust close replacement, and noncircle near/self blocks. |
| RCIP unit coverage in `tests/test_rcip.py` | todo | `test_schurbana_matches_direct_block_formula_shapes` is mostly a shape test and has a tautological `out == out` assertion. | Replace with an independently computed Schur update or MATLAB fixture comparison. |
| RCIP parity in `tests/test_matlab_parity.py` / `tests/test_rcip_parity.py` | todo | Covers helper/compression outputs for a two-edge Laplace corner, not full solve parity. | Add more corner angles, block kernels, ignored-vertex workflows, and eventually a full RCIP solve fixture. |
| FMM parity-style tests in `tests/test_kernel.py` and `tests/test_operators.py` | todo | Mostly compare FMM paths to Python dense/direct paths on simple circle geometries. | Add MATLAB-backed or analytic stress fixtures for starfish geometry, close targets, vector densities, and combined/interleaved kernels. |
| Smoother devtools parity | todo | Current check is diagnostic: error thresholds and output shapes, not MATLAB geometry equivalence. | Either document it as diagnostic-only or add a compact geometry fixture for the supported lightweight smoother path. |

## Ignored For Now

| Area | Status | Reason |
| --- | --- | --- |
| `chunkerfit` devtools parity | ignored | Not part of the current hardening push. Existing coverage only checks adjacency/status and is not worth upgrading now. |
| `chunkerpoly` devtools parity | ignored | Not part of the current hardening push. Existing coverage checks adjacency/diagnostics more than full MATLAB geometry. |
