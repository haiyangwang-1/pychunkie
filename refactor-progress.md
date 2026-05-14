# Python-First Refactor Progress

This document tracks the staged refactor from MATLAB-layout-oriented internals
to the Python-first conventions in `CONTRIBUTING.md`. Keep it current whenever a
stage starts, finishes, changes scope, or leaves known follow-up work.

## Current Status

- Overall status: planned Python-first refactor complete
- Current stage: complete; future behavior ports continue through the normal living docs
- Last updated: 2026-05-14
- Latest verification: `uv run pytest -q --no-test-log` on 2026-05-14,
  `396 passed` in 234.91 seconds.
- Tooling status: `uv run ruff check .` and `uv run mypy` both pass. Mypy is a
  pragmatic first gate over `src/chunkie` with noisy NumPy/dynamic-kernel error
  families disabled while type precision is improved incrementally.

## Stage 0: Guidelines And Tooling

Status: complete

Goals:

- Add `CONTRIBUTING.md` with Python-first design, naming, tensor, testing,
  performance, tooling, and living-doc rules.
- Update `AGENTS.md` so future agent work reads and follows the guideline.
- Add `pytest-timeout`, `ruff`, and pragmatic `mypy` config through `uv`.
- Keep formatting/lint cleanup out of the guideline change.

Acceptance:

- `uv lock` resolves the new tool dependencies.
- `uv run pytest --collect-only -q --no-test-log` reads the pytest config.
- `uv run ruff --version` and `uv run mypy --version` work.

Completed notes:

- Tooling is installed and runnable through `uv`; the later dedicated
  mechanical pass made whole-repo ruff and package-level mypy pass.

## Stage 1: Numerical Test Guardrails

Status: complete

Goals:

- Add a shared pointwise absolute-or-relative assertion helper for numerical
  tests.
- Record maximum absolute error, maximum relative error, and the normalized
  accepted error used by the assertion.
- Make the helper handle zero and near-zero references without `rtol` ambiguity.
- Start migrating representative tests from raw `np.testing.assert_allclose` to
  the helper where the absolute-or-relative rule matters.

Acceptance:

- Focused tests cover the helper's pass/fail behavior, including near-zero
  references.
- Existing automatic allclose metric logging remains intact for tests that still
  use `np.testing.assert_allclose`.
- Migrated tests remain numerically equivalent to their previous checks or
  stricter in the intended absolute-or-relative sense.

## Stage 2: Layout Adapter Boundary

Status: complete

Goals:

- Add explicit helpers for boundary conversions between tensor-shaped data and
  flat solver/backend vectors.
- Centralize Fortran-order flattening that is still required by solver, sparse,
  FMM, FLAM, MATLAB fixture, or legacy public API boundaries.
- Give helpers names that state the boundary role instead of hiding it in
  repeated `reshape(..., order="F")` calls.
- Add focused tests for scalar and vector-component density packing,
  point-cloud packing, and output field unpacking.

Acceptance:

- New helper tests cover 1D vector, 2D matrix, and component-by-node cases.
- Low-risk operator and geometry call sites use the helpers.
- Remaining direct Fortran-order reshapes are either in tests/fixtures or marked
  as adapter-boundary follow-up work.

## Stage 3: Core Shape Migration

Status: complete

Goals:

- Make 1D arrays behave as ordinary vectors and 2D arrays as ordinary matrices
  in new and touched public APIs.
- Keep chunked geometry tensors in their natural axis order, using documented
  axis symbols in nearby code and comments when helpful.
- Remove internal flatten/unflatten round trips where a tensor operation is
  clearer and no backend boundary is involved.

Acceptance:

- Geometry helpers and point information conversions have explicit shape
  contracts.
- Tests cover ordinary C-order 2D matrices and generic tensor inputs where
  public APIs accept them.
- Behavior changes are reflected in `map.md` and test-summary docs.

## Stage 4: Kernel And Operator Tensorization

Status: complete

Goals:

- Represent vector-valued kernels internally as component-aware tensors where
  that exposes the formula better than interleaved flat matrices.
- Use `np.einsum` for kernel application and quadrature contractions with the
  symbols from `CONTRIBUTING.md`.
- Materialize flat/interleaved matrices only at explicit solver/backend
  boundaries.

Acceptance:

- Dense/direct operator tests pass against prior dense behavior.
- FMM and FLAM paths agree with dense/direct references within logged
  tolerances.
- Vector-valued kernel tests cover both tensor formula correctness and boundary
  materialization.

Completed notes:

- Source-level Fortran-order flatten/unflatten calls are centralized in
  `src/chunkie/_layout.py`; remaining source call sites use named adapter
  helpers.
- Stokes and elasticity component block materialization now builds
  `kernel_values[f, t, d, s]` tensors before crossing into flat boundary-matrix
  adapter layout.
- FMM/FLAM density and field conversions use explicit boundary adapter helpers
  instead of inline reshape calls.

## Stage 5: Naming Migration

Status: complete

Goals:

- Rename new and touched public parameters and helpers toward clear domain names.
- Prefer `quadrature_order`, `chunker`, `kernel`, `options`, `source`, and
  `target` outside tiny scopes.
- Keep `ggq`, `rcip`, and `flam` as concise specialized algorithm vocabulary.
- Wrap or comment FLAM callback shorthands at callback boundaries.

Acceptance:

- Public docs and examples use Python-first names.
- Tests cover renamed public entry points and removed MATLAB-style behavior.
- Remaining short names are either local formula names or tracked follow-up
  items.

Completed notes:

- `Chunker` and `ChunkGraph` expose Python-first aliases for point count,
  quadrature order, coordinate dimension, geometry tensors, normal vectors,
  quadrature weights, and adjacency while preserving existing storage fields.
- FLAM proxy callback boundaries now spell out `box_size` and `center` in the
  Python wrappers, with docstrings explaining the positional pyflam callback
  contract.
- Internal RCIP helper names touched by this pass prefer `quadrature_order` and
  `center` where the longer name improves readability.
- Public operator option keywords now cover backend names that used to leak as
  dictionary keys: `flam_occupancy`, `correction_matrix`,
  `rcip_subdivisions`, `rcip_save_depth`, and `rcip_eval_depth`.
- Public operator signatures and matrix wrapper constructors now use
  Python-first names (`chunker`, `kernel`, `density`, `target`, `points`,
  `integrand`, and `options`) instead of `chnkr`, `kern`, `dens`, `targobj`,
  `ptsobj`, `f`, and `opts`.
- Public geometry/chunkgraph/domain helpers touched by this pass now expose
  Python-first names: `chunkerfit(points, options=...)`,
  `chunkerpoints(source, options=...)`, `tochunkgraph(chunker)`,
  `find_edge_regions(graph)`, `chunkgraphinregion(graph, points)`,
  `hypoct_uni(points, box_size, max_level, extent)`, `pointinregion(graph,
  region, point)`, `regioninside(graph, containing_region, candidate_region)`,
  and `mergeregions(graph, first_region, second_region)`.
- Lower-level quadrature entry points touched by this pass now expose
  Python-first public names: `quadnative.buildmat(chunker, kernel,
  target_chunks=..., source_chunks=..., weights=...)`,
  `quadadap.buildmat(chunker, kernel, options=...)`,
  `quadadap.adapgausswts(chunker, source_chunk, target, kernel, options=...)`,
  and GGQ helpers using `quadrature_order` and `singularity` instead of `k`
  and `type`.
- FLAM callback and proxy helper entry points touched by this pass now expose
  Python-first public names such as `chunker`, `kernel`, `target`,
  `proxy_order`, `half_lengths`, `counts`, `point_count`, and `options`, while
  leaving pyflam-imposed positional callback names documented at the boundary.
- Direct point-kernel helpers touched by this pass now expose Python-first
  `source` and `target` parameter names across Laplace, Helmholtz,
  Helmholtz-difference, 1D Helmholtz, biharmonic, Stokes, and elasticity
  Green/kernel functions.
- Point-kernel module helpers are now named `kernel(...)` instead of `kern(...)`
  across Laplace, Helmholtz, 1D Helmholtz, biharmonic, Stokes, and elasticity;
  the kernel factory and tests no longer use the old module-level `kern` API.
- Pure MATLAB-style class-constructor aliases `chunker`, `chunkerpref`, and
  `chunkgraph` were removed from the top-level package facade; use `Chunker`,
  `ChunkerPref.from_any`, and `ChunkGraph`. The geometry subpackage still
  exposes `chunker` and `chunkgraph` as importable submodules.
- Geometry and misc helper entry points touched by this pass now expose
  Python-first names such as `points`, `options`, `chunks`,
  `node_parameters`, `vertices`, `chunk_counts`, and `quadrature_order` for
  near-flagging, nearest-point, data-resolution, arclength, point-info, and
  smoother workflows.
- Kernel factory and panel-quadrature entry points touched by this pass now
  expose Python-first names such as `spec`, `source`, `target`, `chunker`,
  `source_chunk`, `split_info`, `endpoint_interpolator`, `interpolator`, and
  `upsample`; the old `intp`, `intp_ab`, and `ifup` pquad keyword spellings
  have been removed.
- Legendre and RCIP helper entry points touched by this pass now expose
  Python-first parameter names such as `quadrature_order`, `nodes`, `weights`,
  `interpolation`, `edge_count`, `dimension`, and `starts_at_corner`, while
  keeping specialized RCIP algorithm names as domain vocabulary.
- RCIP public helper boundaries now use `chunker`, `edge_chunks`, `kernel`,
  `dimension`, `vertex`, `graph`, and `options`; redundant lowercase/MATLAB-style
  aliases such as `ipinit`, `pbcinit`, `schurbana`, `rcompchunk`,
  `chunkgraphrcip`, and `rcipchunkgraph` were removed.
- Lower-level quadrature adapter helpers touched by this pass now use
  `chunker`, `kernel`, `source`, `target`, `target_info`, `source_chunk`, and
  `quadrature_order` names where they clarify adaptive, GGQ, and RCIP adapter
  boundaries.
- Kernel-factory FMM callback boundaries touched by this pass now use
  `source_info` and `target_info`; compact `src`/`targ` locals remain only
  inside small algebraic formulas.
- Operator helper boundaries touched by this pass now use `chunker`, `kernel`,
  `options`, `density`, `source`, `target`, `source_info`, and `target_info`
  names for block-kernel layout, special-quadrature dispatch, l2 scaling,
  dtype probing, and point-in-region fallback helpers.
- Geometry helper boundaries touched by this pass now use `chunker`, `graph`,
  `options`, `chunk_index`, and `chunk_count` names for polygon fill, graph
  balance/refinement, subchunking, legacy-option normalization, and Bernstein
  ellipse helpers.
- `Chunker.refine`, `ChunkGraph.refine`, `chunk_nearparam`, kernel-factory FMM
  callbacks, and Legendre Taylor stepping helpers now use Python-first
  boundary names including `options`, `source`, `target`, and
  `taylor_order`.
- A source AST audit for function arguments named `chnkr`, `kern`, `opts`,
  `src`, `targ`, or ambiguous quadrature-order `k` returns no remaining hits
  under `src/chunkie`.
- Final small helper cleanup renamed remaining ambiguous callback parameters in
  GGQ block appending and Helmholtz 1D normal-Hessian helpers.
- Runnable examples now use `boundary`, `nodes`, `kernel_obj`, and keyword
  options instead of `chnkr`, inline Fortran-order reshapes, `cormat`, `occ`,
  `useproxy`, or `nsub` option dictionaries.
- Focused verification for the latest operator naming pass:
  `uv run pytest tests/test_keyword_options.py tests/test_operators.py
  tests/test_flam.py tests/test_pquad.py -q --no-test-log`,
  `65 passed` in 7.86 seconds.
- Focused verification for the geometry/domain naming pass:
  `uv run pytest tests/test_domain.py tests/test_geometry.py
  tests/test_geometry_parity.py tests/test_chunker.py tests/test_chunkerfit.py
  tests/test_chunkgraph.py -q --no-test-log`, `45 passed` in 2.65 seconds.
- Focused verification for the quadrature naming pass:
  `uv run pytest tests/test_quadggq.py tests/test_pquad.py
  tests/test_operators.py -q --no-test-log`, `48 passed` in 10.19 seconds.
- Focused verification for the FLAM helper naming pass:
  `uv run pytest tests/test_flam.py tests/test_operators.py -q
  --no-test-log`, `48 passed` in 5.65 seconds.
- Focused verification for the point-kernel naming pass:
  `uv run pytest tests/test_kernels.py tests/test_biharm2d.py
  tests/test_stok2d.py tests/test_elast2d.py tests/test_kernel.py
  tests/test_kernel_algebra.py -q --no-test-log`, `37 passed` in
  2.08 seconds, and `uv run pytest tests/test_matlab_parity.py
  tests/test_easy_parity_stress.py -q --no-test-log`, `113 passed` in
  8.95 seconds.
- Focused verification for the geometry/misc naming pass:
  `uv run pytest tests/test_geometry.py tests/test_geometry_parity.py
  tests/test_arcparam.py tests/test_smoother.py tests/test_chunker.py -q
  --no-test-log`, `36 passed` in 2.11 seconds, and targeted devtools parity
  checks for arcparam, nearest, smoother, and near-flagging, `6 passed` in
  16.37 seconds.
- Focused verification for the kernel-factory and panel-quadrature naming pass:
  `uv run pytest tests/test_pquad.py tests/test_rcip.py
  tests/test_devtools_parity.py::test_pquad_devtools_low_level_weights_match_matlab
  -q --no-test-log`, `23 passed` in 5.53 seconds, and
  `uv run pytest tests/test_kernel.py tests/test_kernel_algebra.py
  tests/test_kernels.py tests/test_easy_parity_stress.py tests/test_operators.py
  -q --no-test-log`, `52 passed` in 5.32 seconds.
- Focused verification for the Legendre/RCIP helper naming pass:
  `uv run pytest tests/test_lege.py tests/test_rcip.py
  tests/test_rcip_parity.py -q --no-test-log`, `27 passed` in 1.52 seconds.
- Focused verification for the lower-level quadrature helper naming pass:
  `uv run pytest tests/test_quadggq.py tests/test_pquad.py tests/test_rcip.py
  tests/test_rcip_parity.py -q --no-test-log`, `44 passed` in 8.36 seconds,
  `uv run pytest tests/test_easy_parity_stress.py tests/test_operators.py -q
  --no-test-log`, `26 passed` in 4.23 seconds, and targeted adaptive devtools
  parity, `2 passed` in 33.00 seconds.
- Focused verification for the kernel-factory callback naming pass:
  `uv run pytest tests/test_kernel.py tests/test_kernel_algebra.py
  tests/test_kernels.py tests/test_operators.py -q --no-test-log`,
  `45 passed` in 2.71 seconds.
- Focused verification for the operator helper naming pass:
  `uv run pytest tests/test_operators.py tests/test_rcip.py
  tests/test_keyword_options.py -q --no-test-log`, `32 passed` in
  2.67 seconds.
- Focused verification for the geometry helper naming pass:
  `uv run pytest tests/test_chunkerpoly.py tests/test_chunkgraph.py
  tests/test_geometry.py tests/test_geometry_parity.py tests/test_chunker.py -q
  --no-test-log`, `41 passed` in 1.83 seconds.
- Focused verification for the final small helper cleanup:
  `uv run pytest tests/test_quadggq.py tests/test_kernels.py tests/test_kernel.py
  tests/test_matlab_parity.py -q --no-test-log`, `147 passed` in
  10.75 seconds.
- Focused verification for the RCIP public-boundary alias removal:
  `uv run pytest tests/test_rcip.py tests/test_rcip_parity.py
  tests/test_matlab_parity.py::test_rcip_recursive_compression_matches_matlab_fixture
  tests/test_easy_parity_stress.py::test_chunkgraph_rcip_stress_nonorthogonal_vertex_and_global_blocks
  -q --no-test-log`, `15 passed` in 1.94 seconds.
- Focused verification for the point-kernel module rename:
  `uv run pytest tests/test_api_contract.py tests/test_kernels.py
  tests/test_kernel.py tests/test_biharm2d.py tests/test_stok2d.py
  tests/test_elast2d.py tests/test_helm1d.py
  tests/test_matlab_parity.py::test_laplace_point_kernels_match_matlab_fixture
  tests/test_matlab_parity.py::test_helmholtz_2d_point_kernels_match_matlab_fixture
  tests/test_matlab_parity.py::test_helmholtz_1d_point_kernels_match_matlab_fixture
  tests/test_matlab_parity.py::test_stokes_point_kernels_match_matlab_fixture
  tests/test_matlab_parity.py::test_elasticity_point_kernels_match_matlab_fixture
  tests/test_matlab_parity.py::test_biharmonic_helpers_match_matlab_bhgreen_fixture
  tests/test_easy_parity_stress.py::test_point_kernels_stress_combined_selectors_and_green_gradients
  -q --no-test-log`, `104 passed` in 2.58 seconds.
- Focused verification for the final source helper signature cleanup:
  `uv run pytest tests/test_chunker.py tests/test_chunkerfunc.py
  tests/test_lege.py
  tests/test_matlab_parity.py::test_extended_legendre_helpers_match_matlab_fixture
  tests/test_kernel.py tests/test_kernels.py -q --no-test-log`, `61 passed`
  in 3.35 seconds.
- Focused verification for constructor-alias removal:
  `uv run pytest tests/test_api_contract.py tests/test_chunker.py
  tests/test_chunkgraph.py tests/test_domain.py tests/test_geometry.py
  tests/test_geometry_parity.py tests/test_rcip.py tests/test_rcip_parity.py
  tests/test_matlab_parity.py::test_chunker_storage_and_data_helpers_match_matlab_fixture
  -q --no-test-log`, `62 passed` in 2.71 seconds; targeted chunkgraph
  devtools parity, `4 passed` in 12.00 seconds.

Remaining work:

- None for public/source naming. Continue broad verification and dedicated
  mechanical lint/type cleanup in later stages.

## Stage 6: Performance And Backend Test Policy

Status: complete

Goals:

- Mark slow, performance, and MATLAB-dependent tests consistently.
- Add backend-selection checks where acceleration behavior is part of the test.
- Add or strengthen timing metrics for acceleration tests without hard
  environment-sensitive speed gates in routine CI.

Acceptance:

- `uv run pytest --collect-only -q --no-test-log` shows no unknown marker
  warnings.
- Acceleration tests record backend, size, tolerance, elapsed time, and numerical
  error where relevant.
- Any hard performance threshold is justified by a controlled environment or
  moved behind the `performance` marker.

Completed notes:

- `pytest-timeout` is configured with a generous global timeout.
- `slow`, `performance`, and `requires_matlab` markers are registered in
  `pyproject.toml`.
- MATLAB fixture-backed parity modules are marked `slow` and
  `requires_matlab`; `uv run pytest -m requires_matlab --collect-only -q
  --no-test-log` collects 175 of 396 tests without unknown marker warnings.
- Focused verification for the fixture-marker pass:
  `uv run pytest tests/test_matlab_fixtures.py tests/test_geometry_parity.py
  tests/test_rcip_parity.py -q --no-test-log`, `17 passed` in 1.55 seconds;
  `uv run pytest tests/test_matlab_parity.py -q --no-test-log`,
  `106 passed` in 5.31 seconds; and
  `uv run pytest tests/test_devtools_parity.py -q --no-test-log`,
  `54 passed` in 275.27 seconds.
- Representative FMM/FLAM acceleration tests now record backend, problem size,
  elapsed time, absolute/relative tolerances, numerical error, and diagnostic
  speed ratios through `test_metrics`, without hard speed-ratio gates.
- Focused verification for the backend-metric pass:
  `uv run pytest tests/test_operators.py::test_chunkermatapply_fmm_matches_special_matrix_application
  tests/test_operators.py::test_chunkermat_fmm_returns_matrix_free_operator_matching_dense_application
  tests/test_flam.py::test_chunkermat_flam_applies_solves_and_logdet_against_dense
  tests/test_easy_parity_stress.py::test_interleaved_fmm_stress_matches_direct_on_wobbly_curve
  -q --no-test-log`, `4 passed` in 1.49 seconds.
- The first numerical tests migrated to the absolute-or-relative helper record
  both absolute and relative error metrics.
- Full-suite verification after the completed refactor is `396 passed` with the
  global timeout active.
- Whole-repo `uv run ruff check .` passes after the dedicated mechanical cleanup
  and format pass.
- `uv run mypy` passes over `src/chunkie` using the pragmatic first-pass config
  in `pyproject.toml`.

Remaining work:

- Extend structured backend/timing metrics opportunistically when acceleration
  tests are touched for future behavior work.

## Stage 7: Documentation Cleanup

Status: complete

Goals:

- Update living docs for the new Python-first APIs and shape behavior.
- Delete or rewrite docs that only describe obsolete MATLAB-layout behavior.
- Keep `map.md`, `devtools_coverage.md`, and
  `docs/python-test-suite-summary.md` aligned with the final implementation.

Acceptance:

- Every tracked Markdown doc is current or intentionally removed.
- Public examples match the refactored API and layout conventions.
- Verification snapshots are updated after the final focused and broad test
  runs.

Completed notes:

- `CONTRIBUTING.md`, `AGENTS.md`, `map.md`, `devtools_coverage.md`, and
  `docs/python-test-suite-summary.md` now describe the Python-first guideline,
  layout adapter boundary, regular grid-shaped near flags, and the current
  verification snapshot.
- `README.md`, `docs/bie-overview.md`, and `new_structure_map.md` have been
  moved away from MATLAB-layout-first language in the public API overview.
- Public examples have been reformatted and updated to use Python-first naming
  and explicit adapter/helper calls; representative smooth, nonsmooth, FMM,
  FLAM, and chunkgraph examples run successfully.
- The project-owned Markdown scan after public alias cleanup found no stale
  `kern`, `opts`, `chunkgraph(...)`, `chunker(...)`, or Fortran-layout guidance.
- `CONTRIBUTING.md` and `AGENTS.md` now make the living-doc rule explicit for
  project-owned Markdown while excluding vendored/reference `external/`
  Markdown from the pychunkie living-doc gate.
- Verification snapshots in `map.md`, `devtools_coverage.md`, and
  `docs/python-test-suite-summary.md` reflect the final 396-test suite.

Remaining work:

- None for this refactor. Delete or rewrite future docs when their subject no
  longer has a clear living-doc purpose.

## Stage 8: Responsibility Package Split

Status: complete

Goals:

- Move domain helpers under `chunkie.geometry` while preserving the top-level
  facade names.
- Move RCIP out of `chunkie.quadrature` into its own responsibility package.
- Convert the large operator coordinator into a package and peel off stable
  support modules without changing numerical behavior.
- Decide whether FMM adapter code has a clean `chunkie.acceleration` boundary.
- Avoid adding a kernel ABC unless it removes real complexity beyond the
  existing `Kernel` wrapper.

Acceptance:

- Public tests import the new responsibility packages directly.
- Existing top-level facade APIs continue to serve the documented Python-first
  public surface.
- `uv run ruff check .`, `uv run mypy`, and relevant pytest runs pass after
  each structural step.

Completed notes:

- Domain helpers moved from `src/chunkie/domain.py` to
  `src/chunkie/geometry/domain.py`; `chunkie.geometry` now exports the domain
  helper names, and the top-level facade imports them from the geometry package.
- RCIP moved from `src/chunkie/quadrature/rcip.py` to
  `src/chunkie/rcip/core.py` with `chunkie.rcip` as the public
  corner-compression package. `src/chunkie/rcip/algebra.py` now owns the
  prolongation/Schur setup helpers, and `src/chunkie/rcip/types.py` owns the
  saved-data containers. Quadrature now exports only generic quadrature modules.
- Operators moved from `src/chunkie/operators.py` to the `src/chunkie/operators/`
  package. `core.py` keeps the heavy dense/FMM/FLAM assembly and evaluation
  coordinator, `options.py` owns keyword-option normalization/accessors, and
  `types.py` owns small operator wrappers/context containers.
- The FMM boundary is split conservatively: optional `fmm2dpy` loading now lives
  in `src/chunkie/acceleration/fmm.py`, while physics-specific FMM selector
  formulas remain in `src/chunkie/kernels/factory.py` with the kernel metadata.

Remaining work:

- None for this structural pass. Future work can further split
  `operators/core.py` or kernel FMM dispatch when a stable ownership boundary
  becomes clearer.
