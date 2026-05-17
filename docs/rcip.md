# RCIP

`chunkie.rcip` owns local corner compression primitives and the density
reconstruction tools needed by nonsmooth systems.

Current implementation status:

- `build_local_corner_geometry` creates dyadically refined ray panels around a
  corner vertex.
- `build_prolongation` builds barycentric interpolation matrices between local
  node sets.
- `build_split_panel_prolongation` builds the two-half-panel interpolation and
  weighted transfer matrices used when RCIP recursively refines a corner panel.
- `build_block_prolongation` lifts scalar prolongation matrices into the local
  edge/component block layout used by corner compression.
- `schur_compress_block` applies the dense reference Schur/Banachiewicz update
  for one RCIP local elimination step.
- `interpolate_density` applies prolongation matrices to scalar or component
  density values.

Required upcoming work:

- local corner operator assembly,
- recursive compression drivers,
- interpolation of reconstructed corner densities,
- and system-level insertion/evaluation through `chunkie.system.nonsmooth`.
