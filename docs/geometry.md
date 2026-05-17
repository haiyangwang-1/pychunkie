# Geometry

Active geometry code follows the panel-major storage contract from `design.md`.

Canonical arrays:

- `positions[R, s, S]`
- `derivatives[R, s, S]`
- `second_derivatives[R, s, S]`
- `normals[R, s, S]`
- `weights[s, S]`

The canonical boundary point id is:

```text
point_id = panel_id * quadrature_order + local_node_id
```

Normals are right normals of the oriented curve. For a counter-clockwise outer
boundary in two dimensions, the right normal points exterior.

Current implementation status:

- Basic `Chunker` storage and point views are active.
- `Chunker` exposes panel lengths, total length, signed area, arclength
  density, unit tangents, and signed curvature using the panel-major tensors.
- `arclength_parameterization`, `evaluate_arclength`, and
  `resample_by_arclength` are active for evaluating a discretized curve by
  physical arclength and rebuilding panels with constant arclength speed.
- `change_quadrature_order` interpolates panel geometry, and optional data with
  trailing `(quadrature_order, panel_count)` axes, to a new Legendre order
  without changing panel topology.
- `refine` uniformly splits panels by powers of two, preserving panel-major
  storage while rescaling first and second derivatives by the child reference
  coordinate map.
- Translation, uniform scaling, affine transforms, rotations, and reflections
  are active. Transforms recompute normals and quadrature weights from the
  transformed tangent instead of carrying stale geometric metadata.
- Near-panel helpers include direct node-distance flags and padded
  axis-aligned rectangle flags, including a grid wrapper that preserves
  `np.meshgrid` row-major ordering. `nearest_point` projects targets to the
  nearest Legendre-panel location using Newton iteration in panel reference
  coordinates.
- Circle, ellipse, curve, and polygon constructors are active first-pass
  implementations. Curve constructors accept full `(positions, derivatives,
  second_derivatives)` callbacks and position-only callbacks, with finite
  differences used only at that adapter boundary. `chunker_from_curve` now
  adaptively splits unresolved parameter intervals with a high-vs-low Gauss
  arclength estimate and records accepted intervals in metadata.
- `ChunkGraph.from_vertices` builds multi-edge graphs from vertices and
  directed edge indices. `ChunkGraph` records, merged point views, edge point
  views, region boundary parts, selected-edge `BoundaryPart` views, and
  oriented nested-cycle region classification are active. Reversed
  `BoundaryPart` orientations reverse point order and weights, flip first
  derivatives and right normals by the reference-coordinate chain rule, and
  keep second derivatives unchanged. Adaptive refinement and full multi-region
  system integration remain required upcoming work.
