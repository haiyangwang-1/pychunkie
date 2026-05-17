# RCIP

`chunkie.rcip` owns local corner compression primitives and the density
reconstruction tools needed by nonsmooth systems.

Current implementation status:

- `build_local_corner_geometry` creates dyadically refined ray panels around a
  corner vertex, including derivative, right-normal, quadrature-weight, and
  `pointinfo` views compatible with the rest of the panel-major adapters.
- `build_prolongation` builds barycentric interpolation matrices between local
  node sets.
- `build_split_panel_prolongation` builds the two-half-panel interpolation and
  weighted transfer matrices used when RCIP recursively refines a corner panel.
- `build_block_prolongation` lifts scalar prolongation matrices into the local
  edge/component block layout used by corner compression.
- `schur_compress_block` applies the dense reference Schur/Banachiewicz update
  for one RCIP local elimination step.
- `recursive_schur_compress` applies a sequence of dense Schur levels and
  retains each intermediate inverse for diagnostics and later system
  integration.
- `interpolate_density` applies prolongation matrices to scalar or component
  density values.
- `build_rcip_state` attaches finite dense local trace operators for graph
  corner systems when a boundary trace is available, including the one-sided
  jump in the local refined-panel ordering.

Required upcoming work:

- interpolation of reconstructed corner densities,
- recursive local compression from the attached operators,
- and system-level insertion/evaluation through `chunkie.system.nonsmooth`.
