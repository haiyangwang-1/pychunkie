# Geometry

Active geometry code follows the panel-major storage contract from `design.md`.

Canonical arrays:

- `positions[R, s, S]`
- `derivatives[R, s, S]`
- `second_derivatives[R, s, S]`
- `normals[R, s, S]`
- `weights[s, S]`

`Chunker` stores Gauss-Legendre reference nodes as `_legendre_nodes[s]` and
reference weights as `_legendre_weights[s]`. The public `weights[s, S]` tensor
contains physical quadrature weights:

```text
weights[s, S] = ||derivatives[:, s, S]|| * _legendre_weights[s]
```

For a smooth panel, `sum(weights[:, S])` is the Gauss-Legendre approximation to
that panel's arclength.

The canonical boundary point id is:

```text
point_id = panel_id * quadrature_order + local_node_id
```

Normals are right normals of the oriented curve. For a counter-clockwise outer
boundary in two dimensions, the right normal points exterior.

Current implementation status:

- Basic `Chunker` storage and point views are active. Point ids use direct
  panel-major arithmetic; there is no separate point-map object.
- `Chunker` exposes panel lengths, total length, signed area, arclength
  density, unit tangents, and signed curvature using the panel-major tensors.
- `Chunker` exposes panel endpoint positions, endpoint unit tangents, and
  coordinate bounds for diagnostics and graph/RCIP bookkeeping.
- `refine` supports selected panel splits, maximum panel length enforcement,
  adjacent-panel level restriction, repeated oversampling, and arclength or
  parameter-space split points while preserving panel-major storage.
- Translation, uniform scaling, affine transforms, rotations, and reflections
  are active. Transforms recompute normals and quadrature weights from the
  transformed tangent instead of carrying stale geometric metadata.
- Near-panel helpers include direct node-distance flags and padded
  axis-aligned rectangle flags, including a grid wrapper that preserves
  `np.meshgrid` row-major ordering. `nearest_point` projects targets to the
  nearest Legendre-panel location using Newton iteration in panel reference
  coordinates. Bernstein helpers build reference ellipses and complex panel
  images for future analytic-continuation close-panel tests.
- Circle, ellipse, curve, and polygon constructors are active first-pass
  implementations and default to `quadrature_order=16`. Existing chunkers are
  not upsampled to a different quadrature order. Curve constructors accept full
  `(positions, derivatives, second_derivatives)` callbacks and position-only
  callbacks, with finite differences used only at that adapter boundary.
  `chunker_from_curve` now adaptively splits unresolved parameter intervals
  with a high-vs-low Gauss arclength estimate and records accepted intervals in
  metadata.
- `ChunkGraph.from_vertices` builds multi-edge graphs from vertices and
  directed edge indices. `ChunkGraph` records, merged point views, edge point
  views, region boundary parts, selected-edge `BoundaryPart` views, and
  oriented nested-cycle region classification are active. `BoundaryPart`
  exposes signed curvature from its oriented point tensors so graph-boundary
  trace assembly can use the same smooth-panel diagonal convention as
  `Chunker`. Reversed
  `BoundaryPart` orientations reverse point order and weights, flip first
  derivatives and right normals by the reference-coordinate chain rule, and
  keep second derivatives unchanged. Adaptive refinement and full multi-region
  system integration remain required upcoming work.
