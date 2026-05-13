# pychunkie BIE Overview

`pychunkie` is a Python port of MATLAB `chunkIE` focused on 2D
boundary-integral equation work. It provides the low-level pieces needed to
build and apply BIE discretizations:

- geometry discretization on Legendre panels,
- physics kernels for layer potentials,
- dense and accelerated operator assembly/application,
- graph-based interfaces for multiple boundaries and multiple regions.

It is not a one-call PDE solver. A user still chooses the representation,
adds the correct interior/exterior jump terms, applies compatibility
constraints where the PDE requires them, and calls a dense or iterative linear
solver.

## Core Objects

`Chunker` is the basic boundary object. It stores a curve as `nch` chunks with
`k` Gauss-Legendre nodes per chunk. Geometry arrays use MATLAB layout:

```text
r, d, d2, n:  dim x k x nch
wts:          k x nch
```

Operator inputs and densities flatten nodes in Fortran order, so the first
`k` entries belong to chunk 0, the next `k` to chunk 1, and so on.

Common constructors:

- `chunkerfunc(fcurve, order=16, closed=True, tol=1e-6)` for smooth
  parametric curves.
- `chunkerpoly(verts, order=16, closed=True, dyadic=True)` for true polygons,
  rounded polygons, and dyadically refined non-smooth corners.
- `chunkerfit(points, order=16, closed=True)` for spline-fitted boundaries.
- `tochunkgraph(chnkr)` when a chunker component should be handled as a graph.

`ChunkGraph` stores a set of chunker edges and graph vertices. It is useful for
multiply connected domains, non-smooth corners, and multi-region interface
systems. It exposes merged geometry for scalar operators and edge lists for
edge-by-edge block systems.

`PointInfo` is the flattened point struct passed to kernels. It can carry
positions `r`, normals `n`, derivatives `d`/`d2`, and user data rows.

## Kernels

Use `kernel(family, selector, ...)` to build callable kernel objects. The
kernel object records:

- `opdims`: value and density components per node,
- `sing`: singularity class such as `log`, `pv`, or `hs`,
- `fmm`: an optional FMM evaluator.

Implemented families include:

| Family | Examples | Notes |
| --- | --- | --- |
| Laplace | `kernel("lap", "s")`, `kernel("lap", "d")`, `kernel("lap", "dp")` | Single, double, normal/tangential derivative, gradient, and combined forms. |
| Helmholtz | `kernel("helm", "s", zk)`, `kernel("helm", "c", zk)` | Scalar and transmission-style block selectors. |
| Stokes | `kernel("stok", "s", mu)`, `kernel("stok", "strac", mu)` | Velocity, pressure, traction, gradient, and combined forms. |
| Biharmonic | `kernel("biharm", "s")`, `kernel("biharm", "shess")` | Green, gradient, Hessian, Laplacian, and normal derivative forms. |
| Elasticity | `kernel("elast", "s", lam, mu)` | Kelvin single layer and implemented double/traction variants. |

Block kernels can be made with `kernel([[k11, k12], [k21, k22]])` for
node-interleaved vector systems.

## Operator Workflow

A typical BIE workflow is:

```python
chnkr, _ = chunkerfunc(curve, order=16, closed=True)
lap_s = kernel("lap", "s")
mat = chunkermat(chnkr, lap_s)
sigma = np.linalg.solve(mat, boundary_data)
values = chunkerkerneval(chnkr, lap_s, sigma, targets)
```

Important conventions:

- `chunkermat` includes quadrature weights in the source dimension.
- `chunkerkerneval` expects an unweighted density and applies weights
  internally.
- Boundary jump terms such as `+/- 0.5 * I` are not included automatically.
- For close target evaluation, pass `force_adaptive=True` when adaptive
  correction is needed.

## Smooth Single-Domain BVPs

For a smooth closed boundary, common Laplace second-kind and first-kind systems
are formed directly from the kernels:

- Dirichlet by single layer: solve `S sigma = g`.
- Interior Neumann by single layer: solve `(0.5 I + K') sigma = h`.
- Exterior Neumann by single layer: solve `(-0.5 I + K') sigma = h`.
- Combined-field Helmholtz systems use `kernel("helm", "c", zk, coefs)`.

For 2D Laplace single-layer Dirichlet systems, add the usual zero-net-charge or
constant-potential constraint when solving exterior or mean-sensitive problems.

## Non-Smooth Domains

Use `chunkerpoly(..., dyadic=True, depth=n)` for true corners. This
creates smaller panels near each corner and preserves a non-rounded geometry.
The package also includes RCIP utilities in `chunkie.quadrature.rcip` for local corner
compression on chunkgraphs. Current high-level examples keep the BVP solve
explicit so users can see where jump terms, compatibility constraints, and
corner refinement enter.

## Chunkgraphs And Multi-Region Problems

`ChunkGraph` represents a network of boundary edges. Region helpers include:

- `find_edge_regions(cg)`: one-based region id on each side of every edge.
- `chunkgraphinregion(cg, pts)`: classify target points by graph region.
- `cg.echnks`: edge chunkers for edge-wise physics or interface blocks.
- `cg.regions`: signed loops describing the graph regions.

For scalar operators, pass the graph directly to `chunkermat` or
`chunkerkerneval`; the graph is treated as merged geometry. For coupled
interface systems, pass an edge-by-edge kernel matrix with shape
`(len(cg.echnks), len(cg.echnks))`.

## Acceleration

Dense assembly is the reference path:

```python
mat = chunkermat(chnkr, kern)
```

FMM acceleration uses `fmm2dpy` for supported 2D kernels:

```python
values = chunkerkerneval(chnkr, kern, sigma, targets, acceleration="fmm", tol=1e-12)
op = chunkermat(chnkr, kern, acceleration="fmm")
y = op @ sigma
```

FLAM acceleration uses `pyflam` for compressed matrix application and, for
`rskelf`, approximate solves:

```python
op = chunkermat(
    chnkr,
    kern,
    acceleration="flam",
    dval=1.0,
    rank_or_tol=1e-10,
)
y = op @ sigma
sigma = op.solve(rhs)
```

Use `dval` to build shifted or second-kind systems. Use `l2scale=True` when
working with L2-scaled operator forms.

## Demos

Run examples with `uv` from the repository root:

```powershell
uv run python examples/smooth_laplace_interior_dirichlet.py
uv run python examples/smooth_laplace_exterior_dirichlet.py
uv run python examples/smooth_laplace_interior_neumann.py
uv run python examples/smooth_laplace_exterior_neumann.py
uv run python examples/nonsmooth_laplace_interior_dirichlet.py
uv run python examples/nonsmooth_laplace_exterior_dirichlet.py
uv run python examples/nonsmooth_laplace_interior_neumann.py
uv run python examples/nonsmooth_laplace_exterior_neumann.py
uv run python examples/nonsmooth_laplace_rcip.py
uv run python examples/chunkgraph_region_classification.py
uv run python examples/chunkgraph_annular_dirichlet.py
uv run python examples/accelerated_fmm_laplace.py
uv run python examples/accelerated_fmm_helmholtz.py
uv run python examples/accelerated_fmm_biharmonic.py
uv run python examples/accelerated_fmm_stokes.py
uv run python examples/accelerated_flam_laplace.py
```

The demos print relative or absolute errors against manufactured solutions.
The nonsmooth BVP demos also write solution and log-error PNG files next to the
script, using sparse corrected quadrature matrices for near-boundary target
evaluation. Each script is self-contained and covers one case.
