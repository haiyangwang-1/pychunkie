# Rewrite Implementation Map

Status: active clean rewrite.

Update trigger: update this file when public APIs, source-tree structure,
implemented behavior, known limitations, MATLAB-reference mapping, or
verification snapshots change.

This map is the active rewrite tracker. The older, MATLAB-shaped port map lives
in `docs_old/map.md` and is useful as a historical checklist, but its module
paths do not describe the current clean rewrite.

## Legend

| Flag | Meaning |
| --- | --- |
| ✅ | Implemented active behavior in the clean rewrite. |
| 🧪 | Covered by active Python tests. |
| 🎯 | Compared against MATLAB-generated golden fixture data. |
| 🟡 | Partial, Python-first, reduced-scope, or adapter-only equivalent. |
| 🚧 | Not implemented yet and still a possible rewrite target. |
| 💤 | Intentionally deferred for now. |
| 🚫 | Explicit non-goal for this Python rewrite. |
| 🧩 | Internal helper, not a public API commitment. |
| 🧭 | Reference, demo, fixture, or support material rather than package API. |

Status flags are behavior-level flags. The rewrite does not try to preserve
MATLAB names, MATLAB storage layout, plotting methods, or option dictionaries
unless a compatibility boundary needs them.

## Current Snapshot

- Source rewrite bootstrapped from `design.md`.
- Previous implementation, tests, docs, and status notes are archived in
  `src_old/`, `tests_old/`, and `docs_old/`.
- Active MATLAB fixture parity is currently narrow: `tests/golden/quadggq.mat`
  plus `tests/quadrature/test_ggq.py` cover generated removable-rule GGQ
  behavior. Most other MATLAB rows below are Python behavioral ports rather
  than MATLAB-fixture parity.
- Smooth Laplace circle examples and nonsmooth square Laplace examples are
  standalone scripts using the rewrite system API and generate solution/error
  figures for interior/exterior Dirichlet and Neumann cases.
- Accelerated and `ChunkGraph` examples are standalone scripts using rewrite
  APIs; active examples no longer depend on removed `chunkermat`/
  `chunkerkerneval` facade names or private example helpers.
- Verification during this map update:
  - Focused `uv run ruff check ...`: passes for the touched geometry,
    quadrature, RCIP, and test files.
  - `uv run pytest --collect-only -q`: 142 tests collected.
  - `uv run pytest -q`: `142 passed, 2 skipped` in 34.57 seconds.
- FLAM strategy: defer new FLAM parity and integration work until the upgraded
  FLAM package is ready. The existing `src/chunkie/system/backends/flam.py`
  dense-reference adapter remains documented as temporary coverage, not a
  milestone driver.

## Source Tree

| Area | Status | Notes |
| --- | --- | --- |
| `chunkie.geometry` | ✅ 🧪 in progress | Panel-major `Chunker` with private Gauss-Legendre nodes/weights, public physical quadrature weights, direct point-id arithmetic, pointinfo views, endpoint/bounds diagnostics, basic/adaptive constructors, metrics, Bernstein reference ellipses/panel images, quadrature-order interpolation, selected/max-length/level-restricted/oversampling panel refinement, affine/rotate/reflect transforms, near-panel node-distance flags, rectangle flags, nearest-point projection, orientation-aware selected-edge `BoundaryPart` views with curvature diagnostics, and operational single- and multi-edge `ChunkGraph` records/views/nested classification are active. Standalone curve callback helpers and arclength-resampling helpers are inactive to keep the geometry surface focused. |
| `chunkie.kernels` | ✅ 🧪 in progress | Kernel object, Laplace/Helmholtz/biharmonic/Stokes/elasticity formulas, registry, algebra, and singularity metadata foundation are active. Laplace metadata uses `G`, `G_a`, and `G_ab`; smooth amplitudes are allowed on those bases for special-quadrature consumption; Helmholtz metadata differentiates `J0(k*rho) * G`; biharmonic metadata uses `B=-(rho^2/4)G`; Stokes velocity metadata covers `s` and `d`; elasticity single-displacement metadata covers `s`; algebra scales and cancels exact scalar/matrix singular terms. |
| `chunkie.quadrature` | ✅ 🧪 🎯 in progress | Legendre utilities, dense panel helpers, component-major dense operator materialization, adaptive source-panel fallback, adaptive close-panel replacement for field evaluation, Helsing-Ojala log/Cauchy/derivative product weights, generated GGQ-style split rules/panel matrices, fixture-backed GGQ removable-rule parity, log/PV/HS `SingularityInfo` smooth-amplitude dispatch, and Helmholtz single-layer HO panel correction are active; broader MATLAB GGQ table parity remains a required milestone. |
| `chunkie.rcip` | ✅ 🧪 in progress | Dyadic local corner geometry with pointinfo-compatible derivatives/normals/weights, barycentric and split-panel prolongation matrices, edge/component block prolongation, dense local trace-operator records, dense single-level and recursive Schur compression updates, corner state records, saved recursion records, and density interpolation are active. |
| `chunkie.system` | ✅ 🧪 in progress | Density layout, dense multi-unknown/multi-equation `Chunker` and `BoundaryPart` trace assembly, explicit dense constraint rows, finite dense self diagonals for Laplace double-layer and adjoint double-layer traces, dense/direct/GMRES solve reconstruction, dense evaluation with adaptive close-panel replacement, FMM evaluation, dense-reference matrix-free and scalar Laplace FMM matvecs, automatic adaptive/Helsing-Ojala/GGQ panel replacement selection, old-style RCIP star-block insertion/evaluation for eligible scalar second-kind graph systems, graph-corner RCIP diagnostics with local trace operators, and `LaplaceExteriorDirichletSystem` are active. |
| `chunkie.system.backends` | ✅ 🧪 / 💤 FLAM | FMM2D evaluates scalar Laplace and Helmholtz single-/double-layer potentials, gradients, and target-normal derivatives plus Stokes single-layer velocity, and drives scalar off-boundary Laplace system matvecs against dense references. The current pyFLAM dense-reference adapter has apply/solve/logdet tests but is deferred pending the upcoming FLAM package upgrade. |

## Current Python Source Index

| Python file | Public classes / functions | MATLAB reference responsibility |
| --- | --- | --- |
| `src/chunkie/__init__.py` | top-level exports | Python package facade for the active public API. |
| `src/chunkie/geometry/chunker.py` | `Chunker`, `right_normals` | MATLAB `@chunker` core geometry fields, private Legendre reference data, physical weights, metrics, normals, transforms, and pointinfo. |
| `src/chunkie/geometry/chunkgraph.py` | `ChunkGraph`, `BoundaryPart`, `GraphVertex`, `GraphEdge`, `GraphRegion`, `RegionCycle`, `SignedEdge` | MATLAB `@chunkgraph`, `chunkgraphinit`, selected-edge boundary views, and region classification. |
| `src/chunkie/geometry/constructors.py` | `circle`, `ellipse`, `chunker_from_curve`, `chunker_from_polygon` | MATLAB `chunkerfunc`, `chunkerfuncuni`, `chunkerpoly`, `ellipse`, and constructor-family behavior. |
| `src/chunkie/geometry/refine.py` | `change_quadrature_order`, `refine` | MATLAB `@chunker/refine`, `@chunker/split`, and `@chunker/upsample` behavior at Python-first scope. |
| `src/chunkie/geometry/near.py` | `NearestPoint`, `flagnear`, `flagnear_rectangle`, `flagnear_rectangle_grid`, `nearest_point` | MATLAB near-panel and nearest-point helpers. |
| `src/chunkie/geometry/bernstein.py` | `BernsteinPanelImage`, `bernstein_ellipse`, `bernstein_panel_image`, `bernstein_radius` | MATLAB `@chunker/ellipses`, `+lege/bernstein_ellipse`, and analytic-continuation diagnostics. |
| `src/chunkie/geometry/points.py` | `PointInfoView`, `PanelView` | Pointinfo and panel/node adapter responsibilities. |
| `src/chunkie/geometry/regions.py` | `winding_number` | MATLAB `pointinregion`, `regioninside`, and graph region classification primitives. |
| `src/chunkie/geometry/transforms.py` | `translate`, `scale`, `affine`, `rotate`, `reflect` | MATLAB `plus`, `mtimes`, `move`, `rotate`, and `reflect` style transforms. |
| `src/chunkie/kernels/base.py` | `Kernel`, `flat_positions`, `flat_normals` | MATLAB `@kernel/kernel.m` object behavior and geometry adapters. |
| `src/chunkie/kernels/registry.py` | `kernel` | Python-first kernel factory equivalent for MATLAB `@kernel/*` constructors. |
| `src/chunkie/kernels/laplace.py` | `kernel`, `evaluate` | MATLAB `+chnk/+lap2d/kern.m`, `green.m`, and `@kernel/lap2d.m`. |
| `src/chunkie/kernels/helmholtz.py` | `kernel`, `evaluate` | MATLAB `+chnk/+helm2d/kern.m`, `green.m`, and `@kernel/helm2d.m`. |
| `src/chunkie/kernels/biharmonic.py` | `kernel`, `evaluate` | Biharmonic/flexural kernel formula subset, without full MATLAB `+chnk/+flex2d` workflow. |
| `src/chunkie/kernels/stokes.py` | `kernel`, `evaluate` | MATLAB `+chnk/+stok2d/kern.m` and `@kernel/stok2d.m`. |
| `src/chunkie/kernels/elasticity.py` | `kernel`, `evaluate` | MATLAB `+chnk/+elast2d/kern.m` and `@kernel/elast2d.m`. |
| `src/chunkie/kernels/algebra.py` | `scale`, `add` | MATLAB kernel algebra operators at Python-first scope. |
| `src/chunkie/kernels/singularities.py` | `GeometryRequirements`, `LaplaceBasis`, `LaplaceSingularTerm`, `LaplaceSingularExpansion`, `SingularityInfo`, `matrix_coefficient` | Operational replacement for MATLAB string/selector singularity metadata. |
| `src/chunkie/quadrature/legendre.py` | `legendre_rule`, `pol`, `pols`, `exps`, `rts`, `rts_stab`, `exev`, `derpol`, `dermat`, `intpol`, `intmat`, `matrin`, `barywts`, `barycentric_weights`, `interpolation_matrix`, `bernstein_ellipse`, `polsum`, `tayl`, `adapgauss` | MATLAB `+lege` package. |
| `src/chunkie/quadrature/panel.py` | `dense_panel_matrix`, `dense_panel_operator_matrix`, `operator_matrix_from_weighted_kernel`, `apply_panel_potential` | MATLAB `+chnk/+quadnative`, `pquadwts`, panel-product quadrature, and close-panel field replacement. |
| `src/chunkie/quadrature/adaptive.py` | `build_adaptive_panel_matrix`, `adaptive_panel_matrix` | MATLAB `+chnk/+quadadap/buildmat.m` and `+chnk/adapgausswts.m` style fallback. |
| `src/chunkie/quadrature/helsing_ojala.py` | `build_helsing_ojala_panel_matrix`, `helsing_ojala_log_singular_matrix`, `helsing_ojala_singular_matrix`, `helsing_ojala_weights` | MATLAB special local log/PV/HS panel correction responsibilities, implemented with Python-first singularity metadata. |
| `src/chunkie/quadrature/ggq.py` | `GGQRuleSet`, `setup`, `getremovablequad`, `build_ggq_panel_matrix`, `build_ggq_self_panel_matrix` | MATLAB `+chnk/+quadggq` generated rules and first fixture-backed removable parity. |
| `src/chunkie/rcip/algebra.py` | `split_panel_interpolation`, `block_interpolation`, `setup`, `schur_banachiewicz` | MATLAB `+chnk/+rcip/IPinit.m`, `Pbcinit.m`, `setup.m`, and `SchurBana.m` algebra. |
| `src/chunkie/rcip/local_geometry.py` | `LocalCornerGeometry`, `build_local_corner_geometry` | MATLAB `+chnk/+rcip/chunkerfunclocal.m` behavior. |
| `src/chunkie/rcip/prolongation.py` | `build_prolongation`, `build_split_panel_prolongation`, `build_block_prolongation` | RCIP prolongation/interpolation matrices. |
| `src/chunkie/rcip/compression.py` | `RCIPCornerState`, `RCIPState`, `RCIPSaved`, `RCIPSchurLevel`, `RecursiveCompressionResult`, `schur_compress_block`, `recursive_schur_compress` | MATLAB `Rcompchunk.m`, saved recursion records, and Schur compression concepts. |
| `src/chunkie/rcip/interpolation.py` | `interpolate_density` | MATLAB `rhohatInterp.m` density reconstruction primitive. |
| `src/chunkie/system/density.py` | `DensitySpace`, `Density` | Python replacement for MATLAB flat density vector conventions. |
| `src/chunkie/system/layer.py` | `LayerPotential` | MATLAB kernel/layer-potential term records. |
| `src/chunkie/system/trace.py` | `BoundaryTrace`, `JumpTerm` | MATLAB boundary trace/jump behavior in BIE assembly. |
| `src/chunkie/system/equation.py` | `BoundaryEquation`, `ConstraintTerm`, `Constraint`, `IntegralSystem` | MATLAB `chunkermat`/solve workflows recast as explicit equations. |
| `src/chunkie/system/assembly.py` | `assemble_system_matrix`, `rhs_vector`, `identity_system_matrix`, row/column slice helpers | MATLAB `chunkermat`, `chunkermatapply`, and dense block assembly. |
| `src/chunkie/system/corrections.py` | `PanelCorrection`, `build_corrections`, `build_panel_correction`, `panel_block_indices` | MATLAB correction insertion for native/adaptive/GGQ panel blocks. |
| `src/chunkie/system/matrix.py` | `SystemMatrix` | Dense reference matrix wrapper and solver adapter boundary. |
| `src/chunkie/system/solution.py` | `SystemSolution` | Reconstructed density plus evaluation object. |
| `src/chunkie/system/evaluation.py` | `FieldResult`, `evaluate_solution` | MATLAB `chunkerkerneval`/`chunkerkernevalmat` field evaluation workflows. |
| `src/chunkie/system/matvec.py` | `SystemOperator`, `matrix_free_matvec`, `fmm_matvec` | Matrix-free and FMM-backed application paths. |
| `src/chunkie/system/nonsmooth.py` | `build_rcip_state`, `apply_rcip_to_matrix`, `evaluate_rcip_layer` plus RCIP internals | MATLAB nonsmooth/RCIP system insertion for scalar second-kind graph systems, including old-style star-block replacement and coarse-plus-local field evaluation. |
| `src/chunkie/system/laplace.py` | `LaplaceExteriorDirichletSystem` | First high-level BIE convenience system. |
| `src/chunkie/system/config.py` | `SystemConfig` | Python solver/evaluation/correction policy object. |
| `src/chunkie/system/block.py` | `BlockLayout` | Dense block metadata. |
| `src/chunkie/system/solvers.py` | `solve_system` | Dense, GMRES, and temporary FLAM solve routing. |
| `src/chunkie/system/backends/fmm2d.py` | `apply_fmm` | Runtime FMM2D Python backend. |
| `src/chunkie/system/backends/flam.py` | `FLAMFactor`, `factor_system` | 💤 temporary dense-reference pyFLAM adapter; new FLAM integration deferred. |

## MATLAB Reference Index

The reference scope here is `external/chunkie-matlab/chunkie`. Demos, guides,
the vendored `fmm2d/` checkout, and the embedded MATLAB `FLAM/` checkout are
listed as support/deferred groups rather than expanded file-by-file.

### Root-Level MATLAB Files

| MATLAB item | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `startup.m` | none | 🧭 | MATLAB path setup only. |
| `checkcurveparam.m` | `src/chunkie/geometry/constructors.py` | 🟡 | Constructor validation is local and Python-first; no public checker. |
| `chunkerfit.m` | target: `src/chunkie/geometry/constructors.py` | 🚧 | No active fitting API beyond curve/polygon constructors. |
| `chunkerflam.m` | future FLAM package integration | 💤 | Deferred until the upgraded FLAM package is available. |
| `chunkerfunc.m` | `chunker_from_curve` | ✅ 🧪 | Full callback and position-only callback constructors are active. |
| `chunkerfuncuni.m`, `+chnk/funcuni.m` | `circle`, `ellipse`, `chunker_from_curve` | ✅ 🧪 | Uniform panel construction through Python constructors. |
| `chunkerintegral.m` | target: `src/chunkie/system/evaluation.py` or quadrature helper | 🚧 | No public high-level integral helper yet; weights/densities exist. |
| `chunkerinterior.m` | `ChunkGraph.classify_points`, `winding_number` | 🟡 🧪 | Region classification exists; MATLAB-style chunker-only helper is absent. |
| `chunkerkerneval.m` | `SystemSolution.evaluate`, `evaluate_solution`, `apply_panel_potential` | ✅ 🧪 | Dense and supported FMM off-boundary evaluation are active. |
| `chunkerkernevalmat.m` | `dense_panel_operator_matrix`, `assemble_system_matrix` | 🟡 🧪 | Matrix materialization exists through system/quadrature adapters, not MATLAB facade. |
| `chunkermat.m` | `IntegralSystem`, `assemble_system_matrix` | ✅ 🧪 | Dense block assembly and trace equations are active. |
| `chunkermatapply.m` | `SystemOperator`, `matrix_free_matvec`, `fmm_matvec` | ✅ 🧪 | Dense-reference matrix-free and scalar Laplace FMM matvecs are active. |
| `chunkerpoints.m` | `Chunker` constructor / target constructor helper | 🟡 🧪 | Direct tensor construction is available; MATLAB-style points helper is not public. |
| `chunkerpoly.m` | `chunker_from_polygon` | 🟡 🧪 | Straight polygon construction is active; full smoother-driven MATLAB polygon workflow is not. |
| `chunkgraphinit.m` | `ChunkGraph.from_vertices` | ✅ 🧪 | Multi-edge graph construction is active. |
| `chunkgraphinregion.m` | `ChunkGraph.classify_points`, `boundary_part` | ✅ 🧪 | Nested-cycle region classification and boundary-part views are active. |
| `ellipse.m` | `ellipse` | ✅ 🧪 | Active analytic ellipse constructor. |
| `starfish.m`, `+chnk/+curves/bymode.m` | none | 🚫 | Standalone curve callbacks are inactive; use ordinary Python callables with constructors instead. |
| `nonflatinterface.m` | none | 🚫 | Standalone curve callback ports are inactive. |
| `hypoct_uni.m` | target: future spatial indexing | 🚧 | No active hyperoctree implementation in the clean rewrite. |
| `pointinregion.m`, `regioninside.m` | `winding_number`, `ChunkGraph.classify_points` | ✅ 🧪 | Region membership behavior exists for current graph cases. |
| `mergeregions.m`, `redblue.m` | target: graph region utilities | 🚧 | No active public merge/color-region helper. |
| `trapperfunc.m`, `trapperkerneval.m`, `trappermat.m` | none | 🚫 | Trapper discretization family is an explicit non-goal. |

### MATLAB `@chunker`

| MATLAB method | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `centroids.m` | target: future diagnostics | 🚧 | No active public centroid helper. |
| `arclengthder.m`, `arclengthfun.m` | none | 🚫 | Standalone arclength parameterization/evaluation helpers are inactive. |
| `arcresample.m` | none | 🚫 | Arclength resampling is inactive. |
| `chunker.m` | `Chunker` | ✅ 🧪 | Panel-major storage with Python aliases; MATLAB storage layout is not preserved. |
| `area.m` | `Chunker.area` | ✅ 🧪 | Signed area for closed curves. |
| `arclengthdens.m` | `Chunker.arclength_density` | ✅ 🧪 | Active panel speed helper. |
| `chunkends.m` | `Chunker.panel_endpoints`, `panel_endpoint_tangents` | ✅ 🧪 | Endpoint diagnostics active. |
| `chunkerpoints.m` | `PointInfoView`, direct `Chunker` construction | 🟡 🧪 | Pointinfo adapters active; no MATLAB facade. |
| `chunklen.m` | `Chunker.panel_lengths` | ✅ 🧪 | Active panel lengths. |
| `diffmat.m` | `dermat` | 🟡 🧪 | Legendre derivative matrix exists; no chunker method wrapper. |
| `ellipses.m` | `bernstein_panel_image`, `bernstein_radius` | ✅ 🧪 | Analytic-continuation diagnostics active. |
| `exps.m` | `exps` | ✅ 🧪 | Legendre expansion utilities active. |
| `flagnear.m` | `flagnear` | ✅ 🧪 | Node-distance near flags active. |
| `flagnear_rectangle.m` | `flagnear_rectangle` | ✅ 🧪 | Padded rectangle flags active. |
| `flagnear_rectangle_grid.m` | `flagnear_rectangle_grid` | ✅ 🧪 | Meshgrid-shaped wrapper active. |
| `intmat.m` | `intmat` | ✅ 🧪 | Legendre integration matrix active. |
| `max.m`, `min.m` | `Chunker.bounds` | 🟡 🧪 | Bounds are active; MATLAB method names are not public. |
| `move.m`, `plus.m`, `mtimes.m` | `translated`, `scaled`, `affine`, transform functions | ✅ 🧪 | Python returns transformed copies. |
| `nearest.m` | `nearest_point` | ✅ 🧪 | Newton projection to panel reference coordinate active. |
| `normals.m` | `Chunker.normal_vectors`, `PointInfoView.flat_normals` | ✅ 🧪 | Right normals active. |
| `reflect.m`, `rotate.m` | `reflected`, `rotated` | ✅ 🧪 | Geometry transforms recompute normals/weights. |
| `refine.m` | `refine` | ✅ 🧪 | Selected-panel splitting, maximum panel length, level restriction, repeated oversampling, and arclength/parameter split modes are active with ordered Python panel storage. |
| `signed_curvature.m` | `Chunker.signed_curvature` | ✅ 🧪 | Active smooth-panel curvature. |
| `split.m` | `refine` | 🟡 🧪 | Split behavior is available through `refine`; Python keeps panels in traversal order instead of MATLAB append-new-panel order. |
| `tangents.m`, `taus.m` | `Chunker.tangents` | ✅ 🧪 | Unit tangents active. |
| `tochunkgraph.m` | `ChunkGraph.from_chunker` | ✅ 🧪 | Single-boundary graph conversion active. |
| `upsample.m` | `change_quadrature_order` | ✅ 🧪 | Geometry and panel-data interpolation active. |
| `weights.m`, `whts.m` | `Chunker.weights`, pointinfo weights | ✅ 🧪 | Physical quadrature weights active. |
| `checkadjinfo.m`, `sort.m`, `sortinfo.m` | target: geometry topology helpers | 🚧 | No public parity helper yet. |
| `datares.m`, `makedatarows.m`, `cleardata` behavior | none | 🚧 | MATLAB data-row storage is not part of current public API. |
| `merge.m` | target: public geometry merge helper | 🚧 | Private merge helpers exist for RCIP local assembly only. |
| `reverse.m` | target: orientation transform helper | 🚧 | Reversed `BoundaryPart` tensors exist; general chunker reverse is not public. |
| `onesmat.m`, `normonesmat.m` | target: system constraints/helpers | 🚧 | Constraint rows exist, but these MATLAB helper matrices do not. |
| `plot.m`, `plot3.m`, `scatter.m`, `quiver.m` | none | 🚫 | MATLAB visualization methods are non-goals. |

### MATLAB `@chunkgraph`

| MATLAB method | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `chunkgraph.m` | `ChunkGraph`, `from_vertices`, `from_chunker` | ✅ 🧪 | Graph records, edges, cycles, and regions are active. |
| `build_v2emat.m`, `procverts.m`, `edgeids.m` | graph construction records and `edge` access | 🟡 🧪 | Python stores explicit vertices/edges rather than exposing MATLAB helper matrices. |
| `findregions.m`, `find_edge_regions.m`, `findunbounded.m` | `classify_points`, `region`, `boundary_part` | ✅ 🧪 | Nested oriented-cycle classification active. |
| `slicegraph.m`, `vertextract.m` | `boundary_part`, `edge_points`, graph records | 🟡 🧪 | Selected-edge views active; no exact MATLAB slice facade. |
| `flagnear*.m` | `flagnear*` on boundary part / pointinfo views | 🟡 🧪 | Near helpers operate on current geometry views. |
| `signed_curvature.m` | `BoundaryPart.signed_curvature` | ✅ 🧪 | Orientation-aware boundary curvature active. |
| `normals.m`, `tangents.m`, `weights.m` | `pointinfo`, `BoundaryPart` tensors | ✅ 🧪 | Active through graph point views. |
| `min.m`, `max.m` | target: graph bounds helper | 🚧 | Bounds can be derived from point views, but no public graph method. |
| `refine.m`, `balance.m` | target: graph refinement | 🚧 | Adaptive/full graph refinement remains upcoming. |
| `move.m`, `plus.m`, `mtimes.m`, `reflect.m`, `rotate.m` | target: graph transform helpers | 🚧 | Chunker transforms are active; graph transforms are not public. |
| `datares.m`, `makedatarows.m` | none | 🚧 | MATLAB graph data rows are not in the clean API. |
| `onesmat.m`, `normonesmat.m` | target: system constraints/helpers | 🚧 | Not implemented as graph helpers. |
| `plot.m`, `plot_regions.m`, `quiver.m`, `scatter.m` | none | 🚫 | MATLAB visualization methods are non-goals. |

### MATLAB `@chunkerpref`

| MATLAB item | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `chunkerpref.m` | `SystemConfig` plus constructor keyword arguments | 🟡 🧪 | Preferences are split into explicit Python options instead of a global MATLAB-style preference object. |

### MATLAB `@kernel`

| MATLAB method | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `kernel.m` | `Kernel` | ✅ 🧪 | Dataclass-like callable kernel record. |
| `lap2d.m` | `kernels.laplace.kernel` | ✅ 🧪 | Selectors and singular metadata active. |
| `helm2d.m`, `helm2ddiff.m` | `kernels.helmholtz.kernel` | ✅ 🧪 | Helmholtz value/derivative selector support active; `diff` is behavior, not API name. |
| `stok2d.m` | `kernels.stokes.kernel` | ✅ 🧪 | Velocity single/double-layer formulas and metadata active. |
| `elast2d.m` | `kernels.elasticity.kernel` | ✅ 🧪 | Single-displacement formulas and metadata active. |
| Biharmonic/flexural formulas | `kernels.biharmonic.kernel` | 🟡 🧪 | Biharmonic formulas active; full MATLAB flexural package is not. |
| `plus.m`, `minus.m`, `uminus.m`, `mtimes.m`, `times.m`, `rdivide.m`, `mrdivide.m` | `Kernel.__add__`, `__sub__`, `__neg__`, `__mul__`, `scale`, `add` | 🟡 🧪 | Scalar scaling/addition/cancellation active; full MATLAB operator surface not promised. |
| `zeros.m`, `nans.m`, `conj.m` | target: kernel utility constructors | 🚧 | No active public equivalents. |
| `helm1d.m` | none | 🚧 | Helmholtz 1D is not active in the clean rewrite. |
| `axissymhelm2d.m`, `axissymhelm2ddiff.m`, `helm2dquas.m` | none | 🚫 | Axisymmetric and quasiperiodic kernel families are non-goals for now. |

### MATLAB `@trapper` and `@trapperpref`

| MATLAB item | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `@trapper/*`, `@trapperpref/trapperpref.m`, root `trapper*.m` | none | 🚫 | The trapper discretization family is outside the clean rewrite scope. |

### MATLAB `+chnk` Root Helpers

| MATLAB item | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `adapgausswts.m` | `adaptive_panel_matrix`, `build_adaptive_panel_matrix` | ✅ 🧪 | Adaptive close-panel fallback active. |
| `adapgausskerneval.m` | `apply_panel_potential`, `evaluate_solution` | 🟡 🧪 | Close-panel replacement active for field evaluation. |
| `chunk_nearparam.m` | `nearest_point`, `_near_target_ids` | 🟡 🧪 | Near-parameter behavior is internal. |
| `chunkerkerneval_smooth.m` | `evaluate_solution`, FMM backend | ✅ 🧪 | Smooth off-boundary evaluation paths active. |
| `curvature2d.m` | `Chunker.signed_curvature`, `BoundaryPart.signed_curvature` | ✅ 🧪 | Curvature diagnostics active. |
| `ellipse_oversample.m` | `bernstein_panel_image`, `bernstein_radius` | 🟡 🧪 | Analytic continuation support active; exact MATLAB helper absent. |
| `flagself.m` | target: self/near panel correction diagnostics | 🚧 | Same-panel selection exists internally but no public helper. |
| `normal2d.m`, `perp.m` | `right_normals` | ✅ 🧪 | Right-normal construction active. |
| `pquadwts.m` | `helsing_ojala_weights`, `build_helsing_ojala_panel_matrix` | 🟡 🧪 | Product weights exist through Helsing-Ojala-style local correction, not MATLAB facade. |

### MATLAB `+chnk/+arcparam`

| MATLAB item | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `init.m` | none | 🚫 | Arcparam helpers are inactive. |
| `eval.m` | none | 🚫 | Arcparam helpers are inactive. |

### MATLAB `+chnk/+curves`

| MATLAB item | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `linefunc.m` | none | 🚫 | Standalone curve callback ports are inactive. |
| `fpara.m` | none | 🚫 | Standalone curve callback ports are inactive. |
| `fsine.m` | none | 🚫 | Standalone curve callback ports are inactive. |
| `bymode.m` | none | 🚫 | Standalone curve callback ports are inactive. |

### MATLAB Physics Packages

| MATLAB package/file | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `+chnk/+lap2d/kern.m`, `green.m` | `src/chunkie/kernels/laplace.py` | ✅ 🧪 | Laplace direct formulas and singular metadata active. |
| `+chnk/+lap2d/fmm.m` | `src/chunkie/system/backends/fmm2d.py` | ✅ 🧪 | Supported Laplace FMM selectors active. |
| `+chnk/+helm2d/kern.m`, `green.m`, `helmdiffgreen.m` | `src/chunkie/kernels/helmholtz.py` | ✅ 🧪 | Helmholtz direct formulas and metadata active. |
| `+chnk/+helm2d/fmm.m` | `src/chunkie/system/backends/fmm2d.py` | ✅ 🧪 | Supported Helmholtz FMM selectors active. |
| `+chnk/+helm2d/besseldiff_etc_pscoefs.m`, `even_pseval.m` | `src/chunkie/kernels/helmholtz.py` | 🧩 ✅ 🧪 | Product-rule/smooth-remainder behavior is encoded directly, not exposed as helper files. |
| `+chnk/+helm2d/transmission_helper.m` | `docs/structured-rskelf-transmission.md`, future multi-region systems | 🚧 | Transmission-system target documented; implementation upcoming. |
| `+chnk/+stok2d/kern.m` | `src/chunkie/kernels/stokes.py` | ✅ 🧪 | Stokes velocity formulas and metadata active. |
| `+chnk/+stok2d/fmm.m` | `src/chunkie/system/backends/fmm2d.py` | ✅ 🧪 | Stokes single-layer velocity FMM active. |
| `+chnk/+elast2d/kern.m` | `src/chunkie/kernels/elasticity.py` | ✅ 🧪 | Elasticity single-displacement active. |
| `+chnk/+helm1d/green.m`, `kern.m`, `sweep.m` | none | 🚧 | Helmholtz 1D not active. |
| `+chnk/+flex2d/*` | `src/chunkie/kernels/biharmonic.py` for subset | 🟡 / 🚫 | Biharmonic pieces active; full flexural plate workflows are non-goals for now. |
| `+chnk/+axissymhelm2d/*` | none | 🚫 | Axisymmetric Helmholtz is a non-goal. |
| `+chnk/+helm2dquas/*` | none | 🚫 | Quasiperiodic Helmholtz is a non-goal. |
| `+chnk/+intchunk/*` | none | 🚫 | Interior chunk special machinery is not part of the clean rewrite. |

### MATLAB Quadrature Packages

| MATLAB package/file | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `+chnk/+quadnative/buildmat.m` | `dense_panel_matrix`, `dense_panel_operator_matrix` | ✅ 🧪 | Dense reference panel matrices active. |
| `+chnk/+quadadap/buildmat.m` | `adaptive_panel_matrix` | ✅ 🧪 | Adaptive close-panel source replacement active. |
| `+chnk/+quadggq/setup.m` | `setup`, `GGQRuleSet` | ✅ 🧪 | Generated split rules active. |
| `+chnk/+quadggq/getremovablequad.m` | `getremovablequad` | ✅ 🧪 🎯 | Compared with archived MATLAB `quadggq.mat` fixture. |
| `+chnk/+quadggq/buildmat.m`, `nearbuildmat.m`, `diagbuildmat.m`, `buildmattd.m` | `build_ggq_panel_matrix`, `build_ggq_self_panel_matrix` | 🟡 🧪 | First generated self/near panel matrices active; broader MATLAB table parity upcoming. |
| `+chnk/+quadggq/getlogquad.m`, `gethqsuppquad.m`, `logavail.m`, `setuplogquad.m` | `helsing_ojala_singular_matrix`, `helsing_ojala_log_singular_matrix` and future GGQ table loaders | 🟡 🧪 | Metadata-driven local singular correction active; full MATLAB table loading upcoming. |
| `+chnk/+quadggq/ggqnear*.m` | generated rules in `setup` | 🟡 🧪 | Generated Python rules replace direct MATLAB tables for now. |
| `+chnk/+quadggq/ggqself_*.m` | `build_ggq_self_panel_matrix` | 🟡 🧪 | Generated self split rules active; table parity broader than removable fixture upcoming. |
| `+chnk/+quadggq/hsupp_*.m`, `hqsupp_*.m` | future GGQ/PV/HS table support | 🚧 | Required special-quadrature milestone. |
| `+chnk/+quadba/*` | none | 🚫 | Bremer-Alpert quadrature package is a non-goal for now. |

### MATLAB RCIP Package

| MATLAB item | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `chunkerfunclocal.m` | `build_local_corner_geometry` | ✅ 🧪 | Dyadic local corner geometry active. |
| `IPinit.m` | `build_prolongation`, `split_panel_interpolation` | ✅ 🧪 | Barycentric interpolation/prolongation active. |
| `Pbcinit.m` | `build_block_prolongation`, `block_interpolation` | ✅ 🧪 | Edge/component block prolongation active. |
| `setup.m` | `rcip.algebra.setup` | ✅ 🧪 | Dense local setup algebra active. |
| `SchurBana.m` | `schur_banachiewicz`, `schur_compress_block` | ✅ 🧪 | Dense Schur update active. |
| `Rcompchunk.m` | `recursive_schur_compress`, `build_rcip_state`, `apply_rcip_to_matrix` | 🟡 🧪 | Recursive compression records and scalar second-kind system insertion are active for graph examples; broader vector/multi-density parity remains upcoming. |
| `rhohatInterp.m` | `interpolate_density`, `evaluate_rcip_layer` | 🟡 🧪 | Scalar solved-density reconstruction is active for RCIP field evaluation; broader public reconstruction APIs remain upcoming. |
| `shiftedlegbasismats.m` | `split_panel_interpolation`, Legendre utilities | 🟡 🧪 | Covered as interpolation/prolongation internals. |

### MATLAB Smoother and Special Packages

| MATLAB item | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `+chnk/+smoother/smooth*.m`, `get_mesh.m`, `get_umesh.m`, `get_sigs.m`, `psi_eval.m`, `newt_step.m`, `green.m`, `merge.m`, `expeval.m`, `gpsi_*.m` | `chunker_from_polygon` only | 🚫 | Full nonlinear rounded-polygon smoother is not part of the clean rewrite; straight polygon construction remains active. |
| `+chnk/+spcl/absconvgauss.m` | target: special-function helper if needed | 🚧 | No active clean-rewrite equivalent. |

### MATLAB `+chnk/+flam` and Embedded `FLAM/`

| MATLAB item | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `+chnk/+flam/kernbyindex*.m`, `proxy*.m`, `nproxy_square.m` | future FLAM package integration | 💤 | Deferred until the upgraded FLAM package is ready. |
| `chunkie/FLAM/**` vendored MATLAB package | future external package boundary | 💤 | Do not port file-by-file now. Existing Python `system.backends.flam` is temporary dense-reference coverage. |

### MATLAB `+lege`

| MATLAB item | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `pol.m` | `pol` | ✅ 🧪 | Legendre polynomial and derivative values active. |
| `pols.m` | `pols` | ✅ 🧪 | Batched polynomial values active. |
| `exps.m` | `exps` | ✅ 🧪 | Nodes, weights, value-to-coefficient transforms active. |
| `rts.m`, `rts_stab.m` | `rts`, `rts_stab` | ✅ 🧪 | Rule aliases active. |
| `exev.m` | `exev` | ✅ 🧪 | Expansion evaluation active. |
| `derpol.m`, `dermat.m` | `derpol`, `dermat` | ✅ 🧪 | Coefficient and nodal differentiation active. |
| `intpol.m`, `intmat.m` | `intpol`, `intmat` | ✅ 🧪 | Polynomial integration utilities active. |
| `matrin.m` | `matrin` | ✅ 🧪 | Interpolation matrix active. |
| `barywts.m` | `barywts`, `barycentric_weights` | ✅ 🧪 | Barycentric weights active. |
| `bernstein_ellipse.m` | `bernstein_ellipse` | ✅ 🧪 | Bernstein ellipse points active. |
| `polsum.m` | `polsum` | ✅ 🧪 | Legendre series summation active. |
| `tayl.m` | `tayl` | ✅ 🧪 | Taylor stepping active. |
| `adapgauss.m` | `adapgauss` | ✅ 🧪 | Scalar/vector adaptive Gauss integration active. |

### MATLAB Demos, Guides, and Vendored FMM2D

| MATLAB item | Python implementation / target | Status | Notes |
| --- | --- | --- | --- |
| `chunkie/demo/*.m`, `chunkie/guide/*.m`, `+chnk/+demo/*.m` | `examples/*.py`, docs | 🧭 / 🟡 | Active Python examples cover smooth/nonsmooth Laplace, ChunkGraph, FMM, and current accelerated examples; one-to-one demo parity is not required. |
| `chunkie/fmm2d/**` | `fmm2dpy` runtime dependency plus `system.backends.fmm2d` | 🧭 / ✅ 🧪 | Vendored MATLAB/C FMM2D checkout is reference/support material; Python calls the package-level runtime backend. |

## Public API Target

The top-level package should expose stable user-facing objects:

- `Chunker`
- `ChunkGraph`
- `Constraint`
- `ConstraintTerm`
- `Density`
- `Kernel`
- `kernel`
- `LayerPotential`
- `BoundaryTrace`
- `BoundaryEquation`
- `IntegralSystem`
- `SystemMatrix`
- `SystemSolution`

## Required Milestones

These are required rewrite work, not optional deferrals:

- Broader Helsing-Ojala panel quadrature and global correction insertion.
- Broader GGQ fixture-backed parity beyond the current removable-rule fixture.
- Broader RCIP parity for vector/multi-density systems, curved graph edges, and
  accelerated/matrix-free insertion.
- `ChunkGraph` adaptive refinement, graph transforms, and broader multi-region
  systems.
- Structured transmission-system assembly for multiple boundaries and
  densities.
- Broader FMM-backed matvec/evaluation coverage.
- Exact singularity expansion metadata where special quadrature consumes it,
  with smooth-remainder tests.

Deferred:

- FLAM parity and callback/structured integration until the upgraded FLAM
  package is available.
