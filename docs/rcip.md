# RCIP

`chunkie.rcip` owns local corner compression primitives and the density
reconstruction tools needed by nonsmooth systems.

Current implementation status:

- `build_local_corner_geometry` creates dyadically refined ray panels around a
  corner vertex.
- `build_prolongation` builds barycentric interpolation matrices between local
  node sets.
- `interpolate_density` applies prolongation matrices to scalar or component
  density values.

Required upcoming work:

- local corner operator assembly,
- compressed inverse recursion,
- interpolation of reconstructed corner densities,
- and system-level insertion/evaluation through `chunkie.system.nonsmooth`.
