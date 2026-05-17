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
- Circle, ellipse, curve, and polygon constructors are active first-pass
  implementations.
- `ChunkGraph` records, merged point views, edge point views, region boundary
  parts, and single-cycle region classification are active. Multi-edge
  refinement, orientation-reversed boundary views, and full multi-region
  system integration remain required upcoming work.
