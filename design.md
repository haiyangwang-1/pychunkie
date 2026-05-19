# pychunkie Design

Status: draft rewrite architecture for discussion.

Update trigger: revise this document whenever the desired package architecture,
public abstractions, module boundaries, kernel metadata model, or system
workflow changes.

This document is a design target for a clean rewrite. It is not a refactoring
checklist for the current codebase, and it is not an implementation status
tracker. Current implementation and test coverage remain documented in
`map.md`, `devtools_coverage.md`, and
`docs/python-test-suite-summary.md`.

## Purpose

`pychunkie` should be a Python-first boundary integral equation toolkit. The
code should reflect the mathematical objects a user thinks about:

- geometry,
- PDE kernels,
- local quadrature rules,
- global integral-equation systems,
- linear solves,
- and field evaluation.

The design goal is a clean package that can solve boundary value problems while
keeping the low-level numerical pieces inspectable. Long mixed-purpose files
and modules that combine geometry, kernels, quadrature correction, system
assembly, matvec acceleration, and solver policy should be avoided.

## Desired Source Tree

The desired rewrite layout is:

```text
src/
└── chunkie/
    ├── __init__.py
    ├── geometry/
    │   ├── __init__.py
    │   ├── points.py
    │   ├── chunker.py
    │   ├── chunkgraph.py
    │   ├── constructors.py
    │   ├── refine.py
    │   ├── transforms.py
    │   ├── near.py
    │   ├── bernstein.py
    │   └── regions.py
    ├── kernels/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── registry.py
    │   ├── singularities.py
    │   ├── algebra.py
    │   ├── laplace.py
    │   ├── helmholtz.py
    │   ├── stokes.py
    │   ├── biharmonic.py
    │   └── elasticity.py
    ├── quadrature/
    │   ├── __init__.py
    │   ├── legendre.py
    │   ├── ggq.py
    │   ├── helsing_ojala.py
    │   ├── adaptive.py
    │   └── panel.py
    ├── rcip/
    │   ├── __init__.py
    │   ├── local_geometry.py
    │   ├── prolongation.py
    │   ├── compression.py
    │   └── interpolation.py
    ├── system/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── density.py
    │   ├── layer.py
    │   ├── trace.py
    │   ├── equation.py
    │   ├── block.py
    │   ├── assembly.py
    │   ├── corrections.py
    │   ├── nonsmooth.py
    │   ├── matrix.py
    │   ├── matvec.py
    │   ├── solvers.py
    │   ├── solution.py
    │   ├── evaluation.py
    │   └── backends/
    │       ├── __init__.py
    │       ├── fmm2d.py
    │       └── flam.py
    └── examples/
        ├── laplace_smooth.py
        ├── laplace_nonsmooth.py
        ├── chunkgraph_multiregion.py
        └── accelerated_solve.py
```

There should be no `operators` package in the rewrite. The old ideas behind
`chunkermat`, `chunkermatapply`, `chunkerkerneval`, and
`chunkerkernevalmat` belong in the `system` package:

- boundary matrix assembly,
- matrix-free matvec,
- solve,
- and off-boundary field evaluation.

The name `options` should be avoided in new APIs. Use `config` for durable
configuration objects and explicit keyword parameters for simple public
functions.

## Top-Level Public API

The top-level package should expose only stable user-facing objects:

```python
from chunkie import (
    Chunker,
    ChunkGraph,
    Density,
    Kernel,
    kernel,
    LayerPotential,
    BoundaryTrace,
    BoundaryEquation,
    IntegralSystem,
    SystemMatrix,
    SystemSolution,
)
```

Low-level subpackages remain importable for advanced use:

```python
from chunkie import geometry, kernels, quadrature, rcip, system
```

The top-level API should not expose MATLAB-shaped helper names. It should also
not expose old operator names as primary concepts. If compatibility is ever
needed, it should be isolated in a legacy module and clearly marked as such.

## Dependency Direction

The desired dependency direction is:

```text
geometry
   |
kernels
   |
quadrature
   |
rcip
   |
system
```

Acceleration lives under `system.backends`. It is not a top-level public
package, because acceleration acts on assembled systems, matrix-vector products,
solves, or evaluated fields. Advanced users may still import
`chunkie.system.backends.fmm2d` or `chunkie.system.backends.flam` directly when
they need backend-level control.

Dependency rules:

- `geometry` imports only NumPy/SciPy-light numerical helpers and local geometry
  code.
- `kernels` imports geometry data containers but does not import quadrature,
  RCIP, FMM, FLAM, or system code.
- `quadrature` imports geometry and kernels to build local panel rules only.
- `rcip` imports geometry, kernels, and local quadrature helpers for corner
  compression primitives.
- `system` imports all lower layers and owns global assembly, correction
  insertion, solver policy, matvec acceleration, and field evaluation.

## Geometry

### Responsibility

`chunkie.geometry` owns all geometric data and geometric algorithms:

- point storage,
- panel indexing,
- chunker construction,
- chunk graph topology,
- refinement,
- transforms,
- normals and weights,
- near-panel flags,
- Bernstein ellipse helpers,
- and region classification.

Geometry should not know about BVPs, layer densities, system matrices, FMM,
FLAM, or solver choices.

### Boundary Storage

`Chunker` should own the canonical storage for boundary geometry directly.
There should not be a separate `PointInfo` plus `PanelLayout` storage model
that has to stay synchronized with `Chunker`. This is a reasonable simplifier:
boundary discretizations almost always need positions, derivatives, second
derivatives, normals, and weights, so the rewrite should require those fields
to exist on every constructed `Chunker`.

There should also not be a public standalone `PointInfo` storage class.
Boundary point data should live directly on `Chunker` and `ChunkGraph`.
`pointinfo` can exist as a read-only property/view/protocol over that storage,
but users should not need to construct or synchronize a separate object.

### Panel-Major Point Storage

Boundary point storage should be panel-major:

```text
positions[R, s, S]
derivatives[R, s, S]
second_derivatives[R, s, S]
normals[R, s, S]
weights[s, S]
```

where:

- `R` is coordinate dimension,
- `s` is the local node index on one panel,
- `S` is the panel index.

The canonical boundary point id is:

```text
point_id = panel_id * quadrature_order + local_node_id
```

So all nodes of panel 0 come first, then all nodes of panel 1, and so on.
This convention should be used consistently by geometry views, local
quadrature, dense assembly adapters, and system vector layouts.

`chunker.pointinfo` and `chunkgraph.pointinfo` should be views exposing the
same panel-major attributes. They may be implemented as a small internal
`PointInfoView` protocol if useful, but this is not a public user-facing data
model and it should not own arrays.

Expected view attributes:

```python
class PointInfoView(Protocol):
    positions: ndarray              # positions[R, s, S]
    derivatives: ndarray            # derivatives[R, s, S]
    second_derivatives: ndarray     # second_derivatives[R, s, S]
    normals: ndarray                # right/exterior normals[R, s, S]
    weights: ndarray                # weights[s, S]
    nodes: ndarray                  # Legendre nodes[s]
    panel_ids: ndarray              # panel_ids[S] or selected panel ids
```

For arbitrary off-boundary target clouds, public APIs should accept coordinate
arrays directly, usually `targets[R, n]`. Internally, system evaluation may
wrap them in a minimal target view with positions only. That target view is an
adapter, not a geometry storage class.

`pointinfo` views should not store layer-potential densities. Densities are
unknowns or solved quantities and belong in `system.Density`.

### `Density`

`Density` should live in `chunkie.system`, not in `geometry`.

It should store density values associated with a source geometry:

```python
@dataclass
class Density:
    name: str
    geometry: Chunker | ChunkGraph | BoundaryPart
    values: ndarray             # values[d, s, S] or values[s, S] for scalar
    component_count: int
    layout: DensityLayout
```

It should know how to:

- validate point counts against its source geometry,
- return scalar or component views,
- convert to solver vectors at system boundaries,
- apply quadrature weights when requested by assembly/evaluation,
- and provide values to system-level RCIP reconstruction when evaluating
  nonsmooth solutions.

### Tensor And Solver-Vector Layout

Mathematical tensors should keep component axes explicit:

```text
density[i, s, S]              # input component, local node, panel
kernel_values[o, i, t, s]     # output component, input component, target point, source point
field[o, t]                   # output component, target point
```

Boundary point order is panel-major:

```text
point_id = panel_id * quadrature_order + local_node_id
```

Flat solver vectors should use the panel-major point order implied by the
canonical point id:

```text
flat_density[(i * point_count) + (panel * k + local_node)]
    == density[i, local_node, panel]
```

Because the stored density tensor is `density[i, local_node, panel]`, this is
not the same as blindly calling C-order `density.reshape(-1)`. The conversion
should be an explicit layout adapter:

```python
flat_density = density.swapaxes(1, 2).reshape(-1)
```

For scalar density values stored as `density[s, S]`, the analogous adapter is:

```python
flat_density = density.T.reshape(-1)
```

Geometry arrays should use the same point order when flattened for kernels or
backends:

```python
flat_positions = positions.swapaxes(1, 2).reshape(R, -1)
```

The inverse adapter should restore:

```python
density = flat_density.reshape(component_count, panel_count, k).swapaxes(1, 2)
```

Dense matrix rows and columns should use the analogous component-major layout:

```text
row = o * target_point_count + target_point
col = i * source_point_count + source_point
```

The implementation should use explicit layout adapters for these conversions.
Do not rely on memory-order tricks in public-facing code paths.
Interleaved point-major vectors are not the canonical project layout. If a
backend such as PyFLAM, FMM code, or an external solver needs a different
layout, the backend adapter should convert at the boundary.

### `Chunker`

`Chunker` represents one oriented curve discretized into Legendre panels.

Proposed fields:

```python
@dataclass
class Chunker:
    positions: ndarray              # positions[R, s, S]
    derivatives: ndarray            # derivatives[R, s, S]
    second_derivatives: ndarray     # second_derivatives[R, s, S]
    normals: ndarray                # right/exterior normals[R, s, S]
    weights: ndarray                # physical weights[s, S] = |dr/du| * Legendre weights[s]
    _legendre_nodes: ndarray        # private Gauss-Legendre nodes[s]
    _legendre_weights: ndarray      # private Gauss-Legendre weights[s]
    adjacency: ndarray              # adjacency[2, S]
    closed: bool
    orientation: Literal["ccw", "cw", "open"] | None
    vertices: ndarray | None
    metadata: dict[str, Any]
```

Required behavior:

- construct from smooth parametric curves,
- construct from polygon vertices,
- refine panels,
- split panels,
- compute normals, weights, curvature, arclength, and panel lengths,
- translate, rotate, scale, reflect, and apply affine transforms,
- expose `point_count`, `panel_count`, `quadrature_order`, and
  `coordinate_dim`,
- expose `pointinfo` views for all points and selected panels,
- map between panel-major point ids and `(panel, local_node)` ids,
- and support near-panel flagging through geometry helpers.

Normal convention:

- for a two-dimensional oriented curve with unit tangent `t = (tx, ty)`,
  the stored normal is the right normal `n = (ty, -tx)`;
- for a counter-clockwise oriented outer boundary, this normal points to the
  exterior;
- for a clockwise oriented boundary, the same formula gives the opposite
  side, so region-aware code should use `left_region`/`right_region` rather
  than infer physics only from orientation labels.

`chunkerfit` and `chunkerpoints` should not be part of the rewrite for now.
They can be reconsidered later if there is a clear user need.

### Constructors

The initial constructor surface should be small:

```python
chunker_from_curve(
    curve,
    *,
    quadrature_order: int = 16,
    closed: bool = True,
    tolerance: float = 1.0e-10,
    min_panel_count: int = 8,
    max_panel_count: int | None = None,
) -> Chunker

chunker_from_polygon(
    vertices,
    *,
    quadrature_order: int = 16,
    closed: bool = True,
    corner_refinement: str = "dyadic",
    refinement_depth: int = 20,
) -> Chunker

circle(*, radius=1.0, center=(0.0, 0.0), quadrature_order=16) -> Chunker
ellipse(*, axes=(1.0, 1.0), center=(0.0, 0.0), quadrature_order=16) -> Chunker
```

The old `chunkerfunc` and `chunkerpoly` names can be avoided in a rewrite.
If aliases are kept, they should be thin compatibility aliases, not the primary
API.

### Near Geometry

The geometry package owns all near-panel and Bernstein ellipse helpers.

Suggested modules:

- `geometry.near`: `flagnear`, `flagnear_grid`, nearest panel queries, close
  panel candidate selection.
- `geometry.bernstein`: Bernstein ellipse images, Bernstein rectangles,
  geometric admissibility information, and panel-local analytic continuation
  geometry.

These helpers should return geometric information only. They should not decide
which quadrature correction or solver backend to use.

### `ChunkGraph`

`ChunkGraph` is the geometry model for multiple edges, corners, interfaces, and
regions. It should be detailed enough to support multi-boundary and
multi-region systems without forcing users to manually manage global edge
indices.

Proposed fields:

```python
@dataclass
class SignedEdge:
    edge_id: int
    orientation: int            # +1 follows stored edge direction, -1 reverses it

@dataclass
class RegionCycle:
    edges: tuple[SignedEdge, ...]

@dataclass
class GraphVertex:
    id: int
    position: ndarray           # position[R]
    incident_edges: tuple[int, ...]

@dataclass
class GraphEdge:
    id: int
    chunker: Chunker
    start_vertex: int
    end_vertex: int
    orientation: int            # +1 if chunker follows start->end
    left_region: int | None
    right_region: int | None
    label: str | None = None

@dataclass
class GraphRegion:
    id: int
    boundary_cycles: tuple[RegionCycle, ...]
    bounded: bool
    label: str | None = None

@dataclass
class ChunkGraph:
    vertices: list[GraphVertex]
    edges: list[GraphEdge]
    regions: list[GraphRegion]
```

This keeps the MATLAB-style idea of regions as signed edge loops, but avoids
one-based signed integer conventions and the ambiguity of signed zero. A region
is a collection of cycles, and each cycle is an ordered sequence of `SignedEdge`
records.

Region and side convention:

- every edge has a canonical orientation from `start_vertex` to `end_vertex`;
- `left_region` and `right_region` are defined relative to that oriented edge;
- the unbounded exterior region has id `0`;
- bounded physical regions have positive integer ids;
- reversing an edge reverses the side interpretation;
- for a counter-clockwise oriented outer boundary, the right side of the curve
  is the exterior region under the convention used by this project;
- transmission systems should specify traces by region side, not by relying on
  ambiguous words like "inside" when an edge is shared.

Required behavior:

- construct from vertices, edge endpoints, and edge curve specs,
- expose edge `Chunker` objects,
- expose a global merged `pointinfo` view with a reversible point-to-edge map,
- classify target points by region,
- determine left/right region of each oriented edge,
- return boundary cycles for a region,
- return a `BoundaryPart` view for an edge, region boundary, or selected cycle,
- refine selected edges while preserving graph topology,
- balance panel lengths across vertices when needed,
- and provide corner/star data for RCIP.

Important `ChunkGraph` views:

```python
graph.edge(edge_id) -> GraphEdge
graph.edge_points(edge_id) -> PointInfoView
graph.region(region_id) -> GraphRegion
graph.boundary(region_id, side="interior") -> BoundaryPart
graph.merged_points() -> PointInfoView
```

`BoundaryPart` should be the object passed to systems when a BVP applies on a
particular side of an interface:

```python
@dataclass
class BoundaryPart:
    graph: ChunkGraph | None
    edges: tuple[int, ...]
    point_indices: ndarray
    side: Literal["left", "right", "interior", "exterior"] | int | None
    orientation: ndarray
    points: PointInfoView
```

This makes multi-region systems explicit: a trace is applied to a boundary
part, not to an ambiguous global merged curve.

### `PanelView`

`PanelView` should be a lightweight property/view produced by `Chunker` or
`ChunkGraph`, not a standalone public storage object. It exposes all geometric
data for one source panel in the same conventions as `pointinfo`.

```python
class PanelView(Protocol):
    parent: Chunker | ChunkGraph
    panel_id: int
    positions: ndarray              # positions[R, s]
    derivatives: ndarray            # derivatives[R, s]
    second_derivatives: ndarray     # second_derivatives[R, s]
    normals: ndarray                # right/exterior normals[R, s]
    weights: ndarray                # weights[s]
    nodes: ndarray                  # nodes[s]
```

Expected accessors:

```python
chunker.panel(panel_id) -> PanelView
graph.edge_panel(edge_id, panel_id) -> PanelView
```

## Kernels

### Responsibility

`chunkie.kernels` owns pure PDE kernel formulas and metadata. A kernel should
not call FMM, assemble matrices, apply quadrature corrections, or compose block
systems.

The kernel layer should answer:

- What PDE family is this?
- What layer or derivative selector is this?
- What are the input and output component dimensions?
- What singularity does it have?
- What canonical Laplace-basis singular expansion should be subtracted to leave
  a smooth remainder?

### Selector Vocabulary

Kernel selectors should be consistent across families where mathematically
meaningful.

Core selector names:

| Selector | Meaning |
| --- | --- |
| `s` | single-layer value |
| `sp` | target-normal derivative of single layer |
| `sg` | spatial gradient of single layer |
| `d` | double-layer value |
| `dp` | target-normal derivative of double layer |
| `dg` | spatial gradient of double layer |

The public selector vocabulary should avoid names such as `p`, `trac`, and
`lap`. Pressure, traction, and Laplacian-like outputs are not uniformly defined
across PDE families and can be assembled by system field definitions from the
primitive layer selectors and geometric contractions. Family-specific helper
functions can exist internally, but the public kernel factory should stay close
to the compact selector set above.

### `Kernel`

Proposed kernel object:

```python
@dataclass(frozen=True)
class Kernel:
    family: str
    selector: str
    params: Mapping[str, Any]
    input_dim: int
    output_dim: int
    singularity: SingularityInfo
    evaluator: Callable[[PointInfoView | ndarray, PointInfoView | ndarray], ndarray]
```

`Kernel.__call__(source, target)` should evaluate direct kernel values only.
It should return tensor-shaped values. Flattening is a system/backend adapter
responsibility.

```text
kernel_values[out_component, in_component, target_point, source_point]
```

Short axis notation:

```text
kernel_values[o, i, t, s]
```

This component-first layout is the canonical kernel layout. Scalar kernels
still return a four-dimensional array with shape `(1, 1, ntarget, nsource)`.
If user-facing convenience helpers want a squeezed scalar matrix, they should
provide that explicitly outside `Kernel.__call__`; the kernel core should not
change rank based on component count.

Kernel values are unweighted. Quadrature weights are applied by assembly and
evaluation code, not by kernel formulas.

The `target_point` and `source_point` axes are flattened point axes. For
boundary views, they use the canonical panel-major point order:

```text
point_id = panel_id * quadrature_order + local_node_id
```

For raw coordinate targets `targets[R, n]`, the target point order is the input
column order.

The corresponding mathematical contraction is:

```python
field = np.einsum("oits,is,s->ot", kernel_values, density, source_weights)
```

where `density[i, s]` is the source density flattened to panel-major source
point order. Dense matrix adapters should perform the transpose/reshape
explicitly rather than relying on implicit NumPy memory-order tricks.

Kernels should validate the geometric data needed by their selector and raise
a clear error when it is missing. In particular, passing a raw coordinate array
is valid only for selectors that need positions alone. Selectors such as `sp`
and `dp` require target normals, and `d`, `dg`, and `dp` require source
normals. Calling those selectors with only `targets[R, n]` should fail fast
instead of silently manufacturing geometry.

The implementation should use `np.einsum` when it improves readability,
especially for contractions involving normals, tangents, densities, stress
tensors, traction, or vector-component mixing.

### Singularity Metadata

Kernel singularity metadata should be detailed and operational. It should not
just say `"log"` or `"pv"`.

The final coding schema for singularity metadata should be decided after the
relevant scalar and vector kernel singular expansions have been derived and
numerically verified. The design principle is clear, but the exact data model
should follow the actual mathematical forms rather than force premature named
axes.

The likely abstraction is a canonical local singular expansion over Laplace
singular basis kernels. Conceptually, it should behave like a dictionary:

```python
{
    LaplaceBasis("s"): coefficient_0,
    LaplaceBasis("d"): coefficient_1,
    LaplaceBasis("sg", components=(0,)): coefficient_2,
    LaplaceBasis("hessian", components=(0, 1)): coefficient_3,
}
```

The implementation may store the expansion as immutable terms rather than a
literal `dict`, because vector PDEs often need tensor coefficients, component
labels, contractions with normals/tangents, or repeated basis pieces. For a
kernel returning `values[out, in, target, source]`, each singular expansion
term should have the mathematical form:

```python
basis_values = laplace_basis(source, target)       # basis_axes..., target, source
coef = coefficient(kernel, source, target)         # out, in, basis_axes..., target, source
term = np.einsum("oi...ts,...ts->oits", coef, basis_values)
```

Constant or geometry-only coefficients may omit trailing axes as long as they
are broadcastable to this convention. Scalar kernels are the degenerate case
where `out == in == 1` and `basis_axes` is empty.

The mathematical meaning should still be dictionary-like:

```text
kernel - sum(coefficient_i * laplace_basis_i)
```

should be smooth, or at least less singular in a declared way.

Sketch of one possible model, not a final API:

```python
SingularityCoefficient = (
    complex
    | ndarray
    | Callable[[Kernel, PointInfoView | ndarray, PointInfoView | ndarray], ndarray]
)

@dataclass(frozen=True)
class SingularityInfo:
    input_dim: int
    output_dim: int
    local: bool
    laplace_expansion: LaplaceSingularExpansion
    remainder_regular: Literal["smooth", "continuous", "bounded", "unknown"]
    removable_part: Kernel | None
    notes: str = ""

@dataclass(frozen=True)
class LaplaceSingularExpansion:
    input_dim: int
    output_dim: int
    terms: tuple[LaplaceSingularTerm, ...]

@dataclass(frozen=True)
class LaplaceSingularTerm:
    basis: LaplaceBasis
    coefficient: SingularityCoefficient
    coefficient_axes: tuple[str, ...] = ("out", "in", "basis", "target", "source")
    meaning: str = ""

@dataclass(frozen=True)
class LaplaceBasis:
    selector: Literal["s", "d", "sg", "sp", "dg", "dp", "hessian"]
    components: tuple[int, ...] = ()
    derivative_order: int = 0
```

`LaplaceSingularExpansion` should support canonicalization: combine terms with
the same basis and compatible coefficient axes, simplify zero coefficients,
and expose a dictionary-like view for algebra and diagnostics.

There should not be an independent `principal_order` field in the final design
unless implementation experience proves it useful as a cached derived property.
The principal singular order should be derivable from which Laplace-basis
coefficients are nonzero.

This makes vector kernels first-class without changing the core idea. A Stokes
velocity kernel, for example, can be represented as a sum of scalar Laplace
`s` pieces and derivative-basis pieces whose coefficients carry the output and
input velocity-component indices. Elasticity can use the same representation
with Lamé-parameter-dependent tensor coefficients. The kernel family owns the
asymptotic derivation; the system layer only sees canonical Laplace-basis
singular terms.

Use `pv` for Cauchy-principal-value singular behavior. The kernel may have a
Cauchy-type local analytic form, but the quadrature and operator semantics are
principal-value semantics, so a separate public `cauchy` singularity class is
unnecessary unless a later implementation needs a more refined basis label.

This metadata is needed by:

- singular quadrature,
- smooth remainder evaluation,
- pquad split construction,
- adaptive fallback decisions,
- FMM acceleration selection,
- and future analytic correction formulas.

Examples:

- Helmholtz `s` has the same logarithmic singularity as Laplace `s`, with a
  coefficient determined by the Green-function convention. The Helmholtz minus
  Laplace singular part is smooth at coincidence.
- Helmholtz `d`, `sp`, `dp`, `sg`, and `dg` should each identify the
  corresponding Laplace-basis expansion and coefficient that kills the
  singularity.
- Stokes single-layer velocity can be represented through Laplace logarithmic
  pieces plus derivative-basis pieces with smooth/tensor coefficients; this
  should be recorded componentwise.
- Stokes pressure, traction, and gradient kernels are not public selectors, but
  any internal field definitions that need their singular behavior should
  record the Laplace-basis derivative pieces needed for correction and
  acceleration.

The basis should be broad enough for common two-dimensional elliptic PDE
kernels: logarithmic Laplace single-layer singularities, principal-value
normal or Cartesian derivatives, hypersingular second derivatives, and smooth
geometry/tensor coefficients multiplying those basis pieces. The exact entries
should come from asymptotic expansions of the relevant Green functions as
`|x-y| -> 0`, not from numerical fitting.

This design intentionally pushes kernel singularity knowledge into kernel
metadata rather than scattering special cases throughout system assembly.

### Kernel Algebra

Kernel algebra should be limited and simple:

- scalar multiplication,
- addition/subtraction of kernels with the same input/output dimensions,
- negation,
- conjugation if needed.

Block composition should not be a kernel responsibility. A block operator is a
system-level concept because it depends on which equation row, unknown density,
boundary part, and constraint the block represents.

The algebra should propagate singularity metadata conservatively:

- sum singular parts when possible,
- detect exact cancellation of singular parts derived from
  `laplace_expansion`,
- downgrade the resulting `SingularityInfo` to `smooth` when all singular
  expansion terms cancel and the declared remainder is smooth,
- mark unknown singularity combinations explicitly when metadata cannot be
  simplified,
- and avoid pretending a block system is a single kernel.

## Quadrature

### Responsibility

`chunkie.quadrature` owns local quadrature rules only. It should not own global
sparse correction matrices, system assembly, block insertion, density layout,
or backend acceleration.

Quadrature routines should answer:

- Given one source panel and some target points, what weights or local matrix
  approximate the layer potential accurately?
- Given a singularity model, can GGQ or Helsing-Ojala provide a local
  correction rule?
- If not, can adaptive Gauss provide a robust fallback?

### Local Panel API

Local quadrature should work at panel granularity:

```python
build_smooth_panel_matrix(
    source_panel: PanelView,
    targets: PointInfoView | ndarray,
    kernel: Kernel,
    *,
    oversample: int = 1,
) -> LocalPanelMatrix

build_ggq_self_matrix(
    source_panel: PanelView,
    kernel: Kernel,
    *,
    singularity: SingularityInfo,
    table_order: int | None = None,
) -> LocalPanelMatrix

build_ggq_neighbor_matrix(
    source_panel: PanelView,
    target_panel: PanelView,
    kernel: Kernel,
    *,
    singularity: SingularityInfo,
    table_order: int | None = None,
) -> LocalPanelMatrix

build_helsing_ojala_matrix(
    source_panel: PanelView,
    targets: PointInfoView | ndarray,
    kernel: Kernel,
    *,
    side: Side,
    oversample: int = 2,
) -> LocalPanelMatrix

build_adaptive_panel_matrix(
    source_panel: PanelView,
    targets: PointInfoView | ndarray,
    kernel: Kernel,
    *,
    tolerance: float = 1.0e-12,
    max_depth: int = 52,
    max_intervals: int = 100000,
) -> LocalPanelMatrix
```

`LocalPanelMatrix` should contain:

```python
@dataclass
class LocalPanelMatrix:
    values: ndarray             # values[o, i, t, s] before system flattening
    source_point_indices: ndarray
    target_point_indices: ndarray | None
    source_panel: int
    target_panel: int | None
    diagnostics: QuadratureDiagnostics
```

The global system layer decides where this local block goes.

### GGQ

`quadrature.ggq` should provide:

- loading tabulated rules,
- generated fallback rules,
- local self-panel matrix construction,
- local neighbor-panel matrix construction,
- and diagnostics for missing tables or unsupported singularity classes.

It should not build global matrices.

### Helsing-Ojala

`quadrature.helsing_ojala` should provide:

- split-kernel local quadrature weights,
- side-aware panel evaluation,
- support for log, PV, hypersingular, and supersingular basis pieces where
  implemented,
- and local fallback diagnostics when a target side cannot be classified.

It should use kernel `SingularityInfo` and smooth-remainder metadata rather than
hard-coded family-specific cases wherever possible.

### Adaptive Gauss

`quadrature.adaptive` should provide:

- recursive local panel integration,
- error/status metadata,
- maximum interval/depth diagnostics,
- and robust fallback behavior for close targets not handled by GGQ or
  Helsing-Ojala.

It should not silently discard recursion failures. Diagnostics should flow up
to `system.corrections`.

## RCIP

`chunkie.rcip` should remain a specialized numerical package, not generic
quadrature.

Responsibility:

- build local corner refinement geometry,
- build prolongation/interpolation operators,
- perform recursive compressed inverse construction,
- return saved metadata needed to reconstruct or evaluate corner densities.

RCIP should not decide the global BVP. The system layer decides:

- which vertices are nonsmooth,
- which equation blocks need RCIP,
- how RCIP-modified local blocks are inserted into the global matrix,
- and how solved coarse densities are interpolated during field evaluation.

Suggested objects:

```python
@dataclass
class CornerPatch:
    vertex_id: int
    edge_ids: tuple[int, ...]
    local_geometry: Chunker | ChunkGraph

@dataclass
class RCIPCompression:
    corner: CornerPatch
    compressed_matrix: ndarray
    saved_levels: tuple[RCIPLevelData, ...]
    interpolation: Callable[[Density], Density]
```

### RCIP Metadata Ownership

Final design decision: RCIP metadata is system-level state. It is independent
of any particular solved density and should be reusable across different
right-hand sides and densities for the same assembled system.

In practice, the primary owner should be the assembled system artifact:
`SystemMatrix` for dense/direct assembly, or an analogous matrix-free system
operator for iterative/FMM assembly. `SystemSolution` and `Density` should only
hold references to this state when they need RCIP-aware evaluation.

Rationale:

- RCIP compression is tied to a specific assembled operator, block layout,
  corner patch, quadrature policy, and density layout.
- Repeated solves with the same matrix should not duplicate identical corner
  recursion data.
- Multiple densities solved against the same system should reuse the same
  RCIP state.
- A solved `Density` should remain the coarse system unknown, not a container
  for hidden refined-grid state.
- During field evaluation, refined corner densities can be reconstructed by
  applying the system-owned saved prolongation/interpolation data to the coarse
  density.
- The interpolation is local to each corner patch and uses already-saved RCIP
  recursion data, so it should be cheap compared with solve time and ordinary
  target evaluation.

Suggested shape:

```python
@dataclass
class RCIPState:
    compressions: dict[RCIPKey, RCIPCompression]

    def interpolate_density(
        self,
        density: Density,
        *,
        key: RCIPKey | None = None,
    ) -> Density:
        ...

@dataclass
class SystemMatrix:
    ...
    rcip_state: RCIPState | None
```

`SystemSolution` may keep a reference to the assembled `SystemMatrix` or
directly to its `RCIPState`, but it should not copy the RCIP metadata by
default. Evaluation can ask the system-owned `RCIPState` for the refined local
density values it needs.

This design assumes RCIP density interpolation is cheap compared with the
overall solve and field evaluation. That should usually be true because the
operation is local to corner patches and applies already-saved recursion data.
Do not add an interpolation cache in the initial rewrite. RCIP reconstruction
is a small local matvec-style operation, and caching would complicate ownership
without clear benefit.

The validity key for `RCIPState` should include the pieces that determine the
local compressed inverse:

- source geometry and graph/corner topology,
- incident boundary parts and orientation,
- equation block layout,
- system kernel or block kernel definition,
- singular/near quadrature policy used for local blocks,
- RCIP subdivision parameters,
- and density component layout.

It should not include the solved density values or the right-hand side.

The initial rewrite should save all RCIP recursion levels in memory. There is
no `rcip_save_depth` option and no disk persistence. A bare density alone is
not enough to perform RCIP-aware evaluation; the compatible in-memory
system-level `RCIPState` must still be available or rebuilt from the same
geometry, system definition, and config.

### RCIP Field Evaluation

RCIP-aware field evaluation should follow the same conceptual structure as the
current implementation:

1. Start from the solved coarse density.
2. Remove or zero the coarse corner/star contribution so it is not counted
   twice.
3. Evaluate the layer potential from the remaining coarse non-corner boundary.
4. For each RCIP corner patch, use the system-owned `RCIPState` to interpolate
   the solved coarse corner density back through the saved recursion levels.
5. Evaluate the layer potential from those reconstructed local refined corner
   panels.
6. Add the coarse and local-corner contributions.

So yes: for field evaluation, RCIP is essentially "interpolate the solved
corner density to the saved finer local grids, then evaluate from those local
panels", plus the important bookkeeping that the corresponding coarse corner
contribution must be removed from the ordinary coarse-boundary evaluation.

The interpolation should return source geometry, local weights, and local
density values:

```python
@dataclass
class RCIPReconstructedPatch:
    key: RCIPKey
    source: PointInfoView | Chunker
    weights: ndarray
    density: Density

@dataclass
class RCIPReconstructedDensity:
    coarse_density_without_star: Density
    patches: tuple[RCIPReconstructedPatch, ...]
```

`system.evaluation` should then evaluate:

```text
field =
    evaluate(coarse_geometry, coarse_density_without_star)
  + sum(evaluate(patch.source, patch.density) for patch in patches)
```

Near-target correction must be applied to both pieces. Targets near ordinary
coarse panels use the usual near/singular evaluation logic. Targets near an
RCIP corner patch use the local reconstructed corner panels as the source
geometry, with the same GGQ/Helsing-Ojala/adaptive fallback policy as ordinary
panel evaluation.

For dense evaluation, the local patch contribution can be evaluated directly.
For FMM evaluation, the reconstructed local sources can be appended to the
ordinary source list after the coarse star contribution is removed. A first
implementation may use dense local patch evaluation even when the far-field
coarse contribution is FMM-accelerated, because the number of reconstructed
corner points is small.

This confirms the ownership decision: `Density` should not own RCIP metadata.
A nonsmooth `SystemSolution` should keep enough reference to its assembled
system operator or `RCIPState` that evaluation can reconstruct the local corner
source representation on demand. A bare density can still be saved as the
coarse unknown, but it is not by itself a complete RCIP-aware solution.

## System

### Responsibility

`chunkie.system` is the central high-level module. It replaces the old
`operators` concept.

It owns:

- layer-potential representations,
- unknown densities,
- boundary equations,
- block systems,
- global matrix assembly,
- near/self quadrature correction insertion,
- corner singularity resolution via RCIP,
- dense and matrix-free matvecs,
- direct and iterative solves,
- FLAM fast direct solves,
- FMM-accelerated matvecs,
- and off-boundary solution evaluation.

### Submodules

#### `system.config`

One system-wide configuration object. Public APIs should use `config`, not
`options`, but simple geometry, kernel, and quadrature functions should prefer
plain keyword arguments instead of small dedicated config classes.

Suggested object:

```python
@dataclass
class SystemConfig:
    assembly_method: Literal["dense", "matrix_free"] = "dense"
    solve_method: Literal["dense", "gmres", "flam"] = "dense"
    evaluation_method: Literal["dense", "fmm"] = "dense"
    tolerance: float = 1.0e-12
    max_iterations: int | None = None
    close_correction: bool = True
    near_factor: float = 1.0
    singular_quadrature: Literal["auto", "ggq", "helsing_ojala", "adaptive"] = "auto"
    prefer_helsing_ojala: bool = True
    use_rcip: bool = True
    rcip_subdivisions: int = 20
    fmm_tolerance: float = 1.0e-12
    flam_tolerance: float = 1.0e-12
    flam_occupancy: int = 200
    flam_proxy: bool = True
```

The exact field list can evolve, but the design should avoid a proliferation
of `ChunkerConfig`, `QuadratureConfig`, `SolveConfig`, `EvaluationConfig`, and
backend config classes. The system is complicated enough to deserve a config
object; simpler layers are not.

#### `system.density`

Owns density storage and solver-vector conversion.

Objects:

- `Density`,
- `DensitySpace`,
- `DensityLayout`,
- `DensityVector`,
- component labels and block offsets.

Responsibilities:

- map unknown names to geometry point sets,
- manage scalar/vector density components,
- convert to/from solver vectors,
- apply source weights when forming layer potentials,
- and expose density views for field evaluation.

#### `system.layer`

Owns layer-potential terms:

```python
@dataclass
class LayerPotential:
    name: str
    source: Chunker | BoundaryPart
    kernel: Kernel
    density: str
    coefficient: complex = 1.0
```

Responsibilities:

- validate kernel input dimensions against density dimensions,
- identify source geometry requirements,
- provide source-panel iteration for assembly/evaluation.

#### `system.trace`

Owns boundary traces and jumps:

```python
@dataclass
class BoundaryTrace:
    layer: LayerPotential
    target: Chunker | BoundaryPart
    side: Side
    jump: JumpTerm | None = None
```

Examples:

- interior Dirichlet trace of a double layer: `-0.5 I + D`,
- exterior Dirichlet trace of a double layer: `+0.5 I + D`,
- Neumann trace of a single layer: `+/- 0.5 I + S'`,
- transmission equations with distinct traces from left and right regions.

The trace layer should be explicit. Kernels should not hide jumps. The sign
convention is:

- geometry normals are the right normals of oriented boundary curves;
- for a counter-clockwise outer boundary, the right normal is the exterior
  normal;
- Dirichlet double-layer jumps and Neumann single-layer jumps follow the
  standard limiting values with respect to that normal orientation;
- region-aware traces should specify the limiting side by `left_region` or
  `right_region`.

#### `system.equation`

Owns equations and constraints:

```python
@dataclass
class BoundaryEquation:
    name: str
    target: Chunker | BoundaryPart
    terms: tuple[BoundaryTrace | AlgebraicTerm, ...]
    rhs: BoundaryData

@dataclass
class Constraint:
    name: str
    terms: tuple[ConstraintTerm, ...]
    value: complex
```

Responsibilities:

- validate target point counts and component counts,
- evaluate callable boundary data on `pointinfo` views,
- represent nullspace constraints and compatibility conditions.

#### `system.block`

Owns global block layout. This replaces kernel block composition.

Objects:

- `BlockLayout`,
- `BlockRow`,
- `BlockColumn`,
- `BlockTerm`,
- `BlockOffsets`.

Responsibilities:

- map equation rows to target geometry/component slices,
- map unknown density columns to source geometry/component slices,
- insert layer-potential blocks,
- insert jump terms,
- insert constraints,
- and expose a global linear algebra shape.

Block composition belongs here because block layout is about equations and
unknowns, not about the mathematical kernel alone.

#### `system.assembly`

Owns global system matrix assembly.

Public function:

```python
assemble_system_matrix(
    system: IntegralSystem,
    *,
    config: SystemConfig,
) -> SystemMatrix
```

Responsibilities:

- build smooth far-field blocks,
- request local singular/near quadrature from `quadrature`,
- insert sparse or dense correction blocks,
- apply jump terms,
- add constraints,
- apply RCIP compression where configured,
- preserve diagnostic metadata.

This is the conceptual replacement for `chunkermat`.

#### `system.corrections`

Owns global correction construction and insertion.

Responsibilities:

- identify self, neighbor, and near-panel target/source pairs,
- use geometry near flags and Bernstein helpers to find candidate panels,
- request local GGQ, Helsing-Ojala, or adaptive panel matrices,
- subtract the smooth/native local block when building a correction,
- assemble global sparse correction matrices or directly overwrite dense
  matrix blocks,
- and report quadrature diagnostics.

Quadrature only builds local rules. Sparse correction matrices are a system
responsibility.

#### `system.nonsmooth`

Owns nonsmooth corner logic.

Responsibilities:

- identify graph vertices needing RCIP,
- build local corner systems,
- call `rcip` primitives,
- insert compressed corner updates into the global system,
- attach reusable system-level `RCIPState` metadata for evaluation.

#### `system.matrix`

Owns the assembled system matrix object.

```python
class SystemMatrix:
    shape: tuple[int, int]
    layout: BlockLayout
    config: SystemConfig
    diagnostics: AssemblyDiagnostics

    def to_dense(self) -> ndarray: ...
    def matvec(self, x: ndarray) -> ndarray: ...
    def solve(self, rhs: ndarray, *, config: SystemConfig) -> ndarray: ...
    def inverse(self, *, config: SystemConfig) -> SystemInverse: ...
```

This object replaces the conceptual role of `chunkermat`. It should not be just
a raw array because the matrix has metadata needed for matvecs, solves,
corrections, and post-solve density reconstruction.

#### `system.matvec`

Owns matrix-free application.

Suggested object:

```python
class SystemOperator:
    shape: tuple[int, int]
    layout: BlockLayout
    config: SystemConfig
    rcip_state: RCIPState | None

    def matvec(self, x: ndarray) -> ndarray: ...
```

Responsibilities:

- dense matvec from assembled matrix,
- FMM-accelerated matvec for supported far-field terms,
- near/self correction application,
- RCIP-aware matvecs where applicable,
- and shape-safe conversion between `DensityVector` and flat solver vectors.

GMRES should use this layer.

#### `system.solvers`

Owns solve policy:

- dense direct solve,
- dense least-squares or constrained solve where needed,
- GMRES with FMM matvec,
- optional preconditioners,
- FLAM fast direct solve,
- residual computation,
- and failure diagnostics.

Suggested API:

```python
solve_system(
    system: IntegralSystem,
    rhs: BoundaryData | None = None,
    *,
    config: SystemConfig,
) -> SystemSolution
```

#### `system.solution`

Owns solved densities and constants.

```python
@dataclass
class SystemSolution:
    system: IntegralSystem
    operator: SystemMatrix | SystemOperator | None
    rcip_state: RCIPState | None
    densities: Mapping[str, Density]
    constants: Mapping[str, complex]
    residual: ndarray
    diagnostics: SolveDiagnostics

    def evaluate(self, targets, *, config: SystemConfig) -> FieldResult: ...
```

The solution should be the user entry point for field evaluation.

#### `system.evaluation`

Owns off-boundary and on-boundary solution evaluation.

Responsibilities:

- evaluate layer potentials at `pointinfo` views or coordinate arrays,
- use dense direct local evaluation,
- use FMM for supported kernel families,
- apply near-target correction,
- interpolate RCIP corner densities,
- evaluate named fields such as value, gradient, pressure, traction, or stress,
- and mask grid outputs by `ChunkGraph` region.

Evaluation should be separate from system matrix assembly. Solving a boundary
system and evaluating fields off the boundary are related but different
operations.

#### `system.backends.fmm2d`

Owns FMM acceleration routines if acceleration is kept under `system`.

Responsibilities:

- map kernel metadata to fmm2dpy calls,
- choose source strengths from weighted densities,
- compose supported scalar/vector PDE calls,
- reject unsupported kernels with clear messages,
- and compare against dense/direct tests.

Kernels provide metadata; FMM routines decide how to use it.

#### `system.backends.flam`

Owns FLAM acceleration routines if acceleration is kept under `system`.

Responsibilities:

- build pyFLAM callbacks from `SystemMatrix` or `BlockLayout`,
- insert correction blocks when callbacks touch near/self interactions,
- expose fast apply and solve,
- expose logdet or diagnostics where meaningful,
- and leave IFMM-style target evaluation unimplemented until there is a clear,
  tested design for it.

## Integral System Object

`IntegralSystem` is the main high-level BVP object.

```python
@dataclass
class IntegralSystem:
    name: str
    geometry: Chunker | ChunkGraph | tuple[BoundaryPart, ...]
    unknowns: tuple[DensitySpace, ...]
    equations: tuple[BoundaryEquation, ...]
    constraints: tuple[Constraint, ...] = ()
    fields: Mapping[str, FieldDefinition] = field(default_factory=dict)

    def assemble(self, *, config: SystemConfig) -> SystemMatrix: ...
    def solve(self, *, config: SystemConfig) -> SystemSolution: ...
```

The base object should be generic and explicit. It should not infer jump terms,
constraints, or formulations unless the user asks for a known convenience
system. Users should be able to try unusual integral equation formulations
without fighting hidden assumptions.

The generic object should be explicit enough to express:

- first-kind single-layer equations,
- second-kind double-layer equations,
- Neumann systems with compatibility constraints,
- combined-field equations,
- multi-boundary chunkgraph systems,
- transmission conditions,
- vector PDE systems,
- and user-defined layer combinations.

The initial rewrite should include only one convenience system:

- `LaplaceExteriorDirichletSystem`.

Initial public constructor:

```python
LaplaceExteriorDirichletSystem(
    geometry: Chunker | ChunkGraph,
    boundary_data: BoundaryData | ndarray | Callable,
    *,
    name: str = "laplace_exterior_dirichlet",
    density_name: str = "sigma",
    field_name: str = "u",
) -> IntegralSystem
```

The first implementation should use the exterior double-layer formulation with
the project normal convention:

```text
(+0.5 I + D) sigma = boundary_data
u(x) = D[sigma](x)
u(x) -> 0 as |x| -> infinity
```

Nonzero far-field constants, charge constraints, multiply connected exterior
variants, and alternative formulations should be deferred until the generic
system path is stable.

All other BVP types, interior/exterior variants, physical systems, and
transmission formulations should initially be expressed through the generic
`IntegralSystem`. More convenience classes can be added later after the core
architecture, singularity metadata, quadrature corrections, RCIP, solve, and
evaluation paths are stable.

## Field Evaluation Model

Field evaluation should be based on named field definitions:

```python
@dataclass
class FieldDefinition:
    name: str
    terms: tuple[LayerPotential, ...]
    target_kernel_selector: str | None
```

Examples:

- `u`: potential value.
- `grad_u`: scalar potential gradient.
- `pressure`: Stokes pressure.
- `traction`: boundary traction or target traction.
- `stress`: vector PDE stress tensor.

User-facing calls:

```python
solution.evaluate(points, field="u", config=SystemConfig(evaluation_method="dense"))
solution.evaluate(points, field="u", config=SystemConfig(evaluation_method="fmm"))
solution.evaluate_grid(x, y, field="u", region=1)
```

Evaluation should return:

```python
@dataclass
class FieldResult:
    points: PointInfoView | ndarray
    values: ndarray             # values[o, p] or grid-shaped field
    field: str
    diagnostics: EvaluationDiagnostics
```

## Acceleration Placement

Acceleration should live under `system.backends`.

Rationale:

- FMM and FLAM operate on assembled systems, densities, matvecs, and field
  evaluation.
- Kernel objects should not call FMM.
- Sparse corrections and RCIP metadata are system-level concepts.
- Backend config naturally belongs next to solve/evaluation config.

The backend modules are not part of the top-level public API, but they should
remain importable by advanced users:

```python
from chunkie.system.backends import fmm2d, flam
```

Kernels must only provide metadata. Backends consume metadata and decide
whether acceleration is possible.

## Configuration Naming

Use one main `SystemConfig` object for system assembly, correction, solve, and
field evaluation. Do not introduce many small config classes for simple
geometry, kernel, or local quadrature helpers.

Simple functions should use explicit keyword arguments:

```python
chunker_from_curve(curve, quadrature_order=16, tolerance=1.0e-10)
build_adaptive_panel_matrix(panel, targets, kernel, tolerance=1.0e-12)
```

System workflows use the single config object:

```python
config = SystemConfig(solve_method="gmres", evaluation_method="fmm")
solution = integral_system.solve(config=config)
```

## Numerical Implementation Style

Use `np.einsum` where it clarifies the mathematics.

Good uses:

- contracting gradients with normals,
- applying tensor-valued kernels to vector densities,
- forming traction from stress tensors,
- applying quadrature weights to component densities,
- accumulating `kernel_values[o, i, t, s]` against `density[i, s]`.

Example:

```python
field = np.einsum("oits,is,s->ot", kernel_values, density, source_weights)
normal_derivative = np.einsum("rts,rt->ts", gradient, target_normals)
traction = np.einsum("abts,bt->ats", stress_tensor, target_normals)
```

Avoid using `einsum` for genuinely sequential algorithms:

- graph traversal,
- adaptive recursion,
- Newton iteration,
- panel splitting,
- RCIP recursion.

## Examples To Support

The rewrite should make these examples simple and idiomatic:

1. Smooth Laplace interior Dirichlet on a circle.
2. Smooth Laplace exterior Dirichlet with charge/constant handling.
3. Smooth Laplace interior and exterior Neumann with explicit compatibility
   constraints.
4. Nonsmooth square Laplace solve with RCIP corner treatment.
5. Chunkgraph annulus or multi-region solve with region-specific evaluation.
6. Stokes velocity/traction system on a smooth boundary.
7. Helmholtz combined-field solve with FMM matvec.
8. FLAM fast direct solve for a second-kind system.

The examples should be BVP examples built from `IntegralSystem`, not manual
matrix construction scripts.

## Singularity Metadata Development

The exact `SingularityInfo` and `LaplaceSingularExpansion` entries should be
derived from local asymptotic expansions of each PDE Green function and
verified numerically.

The verification workflow should be:

1. For a kernel selector, derive the local Laplace-basis singular expansion.
2. Build the smooth remainder predicted by metadata.
3. Evaluate source/target pairs approaching coincidence along several
   directions.
4. Verify the remainder is finite and has the expected smoothness.
5. Add a regression test before using that metadata in system correction or FMM
   acceleration logic.

This is especially important for Helmholtz and Stokes. Helmholtz should be
straightforward because each selector has a Laplace counterpart with the same
local singularity. Stokes needs componentwise metadata because the logarithmic,
principal-value, and derivative-basis pieces appear through tensor contractions.

## Testing Expectations For The Rewrite

Tests should be organized around the new architecture:

- `tests/geometry/`: `Chunker`, `pointinfo` views, `ChunkGraph`, flagnear,
  Bernstein helpers, and regions.
- `tests/kernels/`: family formulas, selector consistency, singularity
  metadata, smooth-remainder checks.
- `tests/quadrature/`: one-panel GGQ, Helsing-Ojala, and adaptive rules.
- `tests/rcip/`: local corner compression and density interpolation.
- `tests/system/`: assembly, corrections, nonsmooth systems, block layout,
  solves, and field evaluation.
- `tests/backends/`: FMM and FLAM comparisons against dense/direct system
  behavior.

Important singularity metadata tests:

- For each singular Helmholtz selector, subtract the declared Laplace
  singular expansion and verify finite/smooth same-point or near-point
  behavior.
- For Stokes singular selectors, verify the declared componentwise singular
  Laplace-basis expansion reproduces the leading singular behavior.
- For algebraic sums, verify singularity metadata combines conservatively.

## Documentation Expectations

Documentation should follow the rewrite architecture:

- `design.md`: desired architecture.
- `docs/geometry.md`: geometry and point storage.
- `docs/kernels.md`: kernel families, selectors, singularity metadata.
- `docs/quadrature.md`: local panel quadrature routines.
- `docs/system.md`: BVP assembly, solve, and field evaluation.
- `docs/backends.md`: FMM/FLAM backend behavior and limitations.

Status docs such as `map.md`, `devtools_coverage.md`, and
`docs/python-test-suite-summary.md` remain implementation trackers, not design
documents.

## Deferred Implementation Details

These points are intentionally deferred until the relevant implementation work:

1. The final singularity metadata schema. Do this after deriving and
   numerically verifying the Laplace-basis singular components for the scalar
   and vector kernels of interest.
2. The first transmission examples to use as generic `IntegralSystem`
   regression tests, even though transmission convenience classes are deferred.

## Non-Goals In The Initial Rewrite

- No `operators` package.
- No primary `chunkermat`/`chunkerkerneval` API.
- No `chunkerfit` or `chunkerpoints`.
- No kernel-owned FMM calls.
- No kernel block composition.
- No global sparse correction matrices in `quadrature`.
- No dictionary-first `options` API.
- No FLAM IFMM field-evaluation support in the initial rewrite.
- No axisymmetric, quasiperiodic, or flexural kernel families unless they are
  explicitly reprioritized.
