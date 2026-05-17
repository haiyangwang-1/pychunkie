# Structured RSKELF for Transmission Systems

Status: research/design note. This file records the formulation that should
drive the pyFLAM/RSKELF integration for multi-boundary systems with more than
one density per boundary. It is not an implemented fast-direct backend.

## Literature Anchors

The most relevant examples are transmission formulations with two interface
densities and different Helmholtz wavenumbers in adjacent regions.

- Borges, Rachh, and Greengard, "On the robustness of inverse scattering for
  penetrable, homogeneous objects with complicated boundary", arXiv:2210.11607.
  Section 2.1 uses the two-density Helmholtz transmission representation
  already mirrored by legacy pychunkie `helmdiff` transmission kernels:
  <https://arxiv.org/abs/2210.11607>.
- Boubendir, Bruno, Levadoux, and Turc, "Regularized combined field integral
  equations for acoustic transmission problems", arXiv:1312.6598. This gives a
  combined-source formulation and makes the two Cauchy-data rows explicit:
  <https://arxiv.org/abs/1312.6598>.
- Greengard and Lee, "Stable and accurate integral equation methods for
  scattering problems with multiple material interfaces in two dimensions",
  Journal of Computational Physics 231 (2012), 2389-2395. This is the important
  multiple-interface/triple-junction reference: it uses global charge/dipole
  densities on the whole interface so only difference kernels appear at
  junctions: <https://doi.org/10.1016/j.jcp.2011.11.034>.
- Greengard, Ho, and Lee, "A fast direct solver for scattering from periodic
  structures with multiple material interfaces in two dimensions", Journal of
  Computational Physics 258 (2014), 738-751. This is the fast-direct companion
  for fixed multi-material geometries: <https://doi.org/10.1016/j.jcp.2013.11.011>.
- Ho and Greengard, "A fast direct solver for structured linear systems by
  recursive skeletonization", SIAM Journal on Scientific Computing 34 (2012),
  A2507-A2532. This is the base RSKELF algorithm:
  <https://doi.org/10.1137/120866683>.

## Two-Region Transmission Model

Let an interior penetrable region have wavenumber `k_i`; let the exterior have
wavenumber `k_e`. The exterior total field is `u_inc + u_e`, while the interior
field is `u_i`. In the equal-density case used in Borges-Rachh-Greengard, the
boundary value problem is

```text
(Delta + k_i^2) u_i = 0                 in Omega_i
(Delta + k_e^2) u_e = 0                 in Omega_e
u_inc + u_e - u_i = 0                   on Gamma
d_n u_inc + d_n u_e - d_n u_i = 0       on Gamma
u_e satisfies the Sommerfeld radiation condition.
```

With single- and double-layer potentials

```text
S_k[mu](x) = integral_Gamma G_k(x,y) mu(y) ds_y
D_k[sigma](x) = integral_Gamma d_{n_y} G_k(x,y) sigma(y) ds_y,
```

the scattered and interior fields are represented by the same two densities:

```text
u_e = D_{k_e}[sigma] - S_{k_e}[mu]
u_i = D_{k_i}[sigma] - S_{k_i}[mu].
```

Taking Dirichlet and Neumann traces gives the second-kind block system

```text
[ I + D_{k_e} - D_{k_i}        S_{k_i} - S_{k_e}      ] [sigma] = [-u_inc]
[ D'_{k_e} - D'_{k_i}          I + S'_{k_i} - S'_{k_e}] [mu   ]   [-d_n u_inc].
```

This is the minimal clean example for pychunkie:

- one boundary curve;
- two density spaces, `sigma` and `mu`;
- two equation rows, Dirichlet and Neumann;
- four operator blocks;
- each block is a kernel expression involving one or more Helmholtz kernels;
- the diagonal identity/jump terms belong to the trace/equation layer, not the
  kernel object.

For a material ratio, the Neumann row changes by the normal-flux coefficients.
Using the Boubendir-Bruno-Levadoux-Turc convention,

```text
gamma_D^1 u^1 + gamma_D^1 u_inc = gamma_D^2 u^2
gamma_N^1 u^1 + gamma_N^1 u_inc = nu gamma_N^2 u^2,
```

so the same block layout remains, but the Neumann row coefficients are region
dependent. That coefficient choice is formulation data and should be expressed
as `BlockTerm` coefficients rather than hidden in Helmholtz kernels.

## Multiple Interfaces

Let `Omega_r` be regions with wavenumbers `k_r` and flux coefficients `c_r`.
Let each oriented edge `Gamma_e` separate two regions `r_minus(e)` and
`r_plus(e)`. Greengard-Lee's robust multi-material representation uses two
global densities on the total interface

```text
Gamma_all = union_e Gamma_e
u_r(x) = S_{k_r}(Gamma_all, sigma)(x) + c_r D_{k_r}(Gamma_all, mu)(x).
```

For a target point on an interface edge between `r_minus` and `r_plus`, the
Dirichlet and flux rows compare the traces from those two regions. In schematic
block form, every source edge contributes:

```text
A_{e,rowD; f,sigma} = S_{k_plus}[Gamma_e, Gamma_f]
                    - S_{k_minus}[Gamma_e, Gamma_f]

A_{e,rowD; f,mu}    = c_plus D_{k_plus}[Gamma_e, Gamma_f]
                    - c_minus D_{k_minus}[Gamma_e, Gamma_f]
                    + local jump identity if e == f

A_{e,rowN; f,sigma} = c_plus^{-1} S'_{k_plus}[Gamma_e, Gamma_f]
                    - c_minus^{-1} S'_{k_minus}[Gamma_e, Gamma_f]
                    + local jump identity if e == f

A_{e,rowN; f,mu}    = D'_{k_plus}[Gamma_e, Gamma_f]
                    - D'_{k_minus}[Gamma_e, Gamma_f].
```

The exact signs depend on the edge normal convention. In pychunkie this must be
resolved through `ChunkGraph` side metadata: equations should specify the
left/right region traces explicitly instead of relying on words such as
"inside" or "outside".

The important structural point is that each `(target edge, equation row,
source edge, density)` pair is its own operator block. For vector or tensor
kernels, each block also has output and input component axes:

```text
A[row_space, col_space][o, i, target_point, source_point].
```

Flattening this matrix too early destroys the information needed by RSKELF to
sample the correct kernel and proxy surface.

## Structured Operator Terms

The system layer should expose a structured operator as a sum of explicit terms:

```text
BlockTerm:
    row_space        equation row, target boundary part, output component count
    col_space        density name, source boundary part, input component count
    kernel_expr      scalar/vector kernel or algebraic kernel expression
    coefficient      complex scalar or region/material coefficient
    trace_side       left/right/plus/minus side for jump signs
    locality         far-compressible, near-exact, jump, correction
```

`kernel_expr` may be a fused expression such as
`D'_{k_plus} - D'_{k_minus}`. That matters: the integral equation is often
well-conditioned because singular pieces cancel in the block expression. The
fast-direct backend should sample the fused expression, not sample two
hypersingular terms as unrelated blocks and hope cancellation is recovered after
flattening.

## RSKELF Sampling Rule

The geometric tree should be built over physical source points on the relevant
interface geometry. Algebraic density/component axes remain metadata attached
to those points.

For a tree node `B` and a candidate column group `C_B`, the ID sample must be
assembled by stacking every far-field row family that can see that column group:

```text
Y_B(C_B) =
    stack over far target clusters T
    stack over boundary equations q
    stack over output components o
        A[q, C_B][o, :, T, B]

P_B(C_B) =
    stack over far-compressible terms touching C_B
    stack over proxy surfaces for the term's target region/kernel
        proxy_kernel_expr(term, proxy_points, source_points_B)
```

For nonsymmetric systems the reverse direction must also be sampled:

```text
Z_B(R_B) =
    stack over far source clusters S
    stack over density families p
    stack over input components i
        A[R_B, p][:, i, B, S]^*
```

The ID for a source block is then built from `[Y_B; P_B]`, and in the
nonsymmetric case from the corresponding incoming/outgoing samples. This is the
behavior we need from pyFLAM: proxy sampling is per operator term/channel, not
one scalar kernel reused for a flattened matrix.

## Skeleton Grouping Policy

There is not one universally correct grouping choice. The backend should make
the grouping explicit and testable.

```text
dof
    ID may select individual scalar DOFs. This is closest to a flat matrix and
    can be efficient, but it loses physical grouping.

point_family
    ID selects all components of one density family at a source point. This is
    the conservative first target for `sigma` and `mu` transmission systems.

point_all_densities
    ID selects all density families/components attached to a source point. This
    is more expensive, but best preserves physical point grouping for strongly
    coupled vector systems.
```

The first production target should be `point_family`, with tests comparing it
against dense assembly for the two-density Helmholtz transmission system.
`point_all_densities` should be kept available for vector PDEs and strongly
coupled formulations.

## Near, Jump, and Correction Blocks

RSKELF compression should only approximate far-field interactions. The backend
must keep these contributions exact in the local blocks:

- identity and jump terms;
- same-panel singular blocks;
- adjacent/near-panel quadrature corrections;
- RCIP-compressed local corner blocks;
- sparse replacement blocks from special quadrature.

The local block seen during elimination is therefore

```text
A_local = A_smooth_near + A_jump + A_special_corrections + Schur_updates.
```

Proxy samples are for the complement of the near set. This is the clean way to
respect both recursive skeletonization and pychunkie's quadrature model.

## Code Design Target

The pychunkie side should provide a backend-facing object with methods like:

```python
class StructuredSystemOperator:
    row_spaces: tuple[RowSpace, ...]
    col_spaces: tuple[ColSpace, ...]
    terms: tuple[BlockTerm, ...]

    def dense_block(self, rows: DofSelection, cols: DofSelection) -> np.ndarray:
        ...

    def far_sample(self, node: TreeNode, cols: DofSelection) -> np.ndarray:
        ...

    def proxy_sample(self, level: int, node: TreeNode, cols: DofSelection) -> np.ndarray:
        ...

    def local_block(self, rows: DofSelection, cols: DofSelection) -> np.ndarray:
        ...
```

The pyFLAM side should consume this structured operator directly. It should not
ask pychunkie to flatten first and then reconstruct meaning from integer row
indices.

## Verification Plan

1. Dense reference: assemble the two-density Helmholtz transmission system above
   as a generic `IntegralSystem` and compare every block against a hand-built
   dense matrix.
2. Legacy parity: port compact fixtures from `tests_old/test_devtools_parity.py`
   around the `chunkermat_l2scaleTest` transmission block.
3. ChunkGraph reference: build a multi-edge `ChunkGraph` with different
   wavenumbers and coefficients per region; compare dense apply against explicit
   block sums.
4. Proxy instrumentation: run structured pyFLAM on a tiny system and assert that
   each expected `(equation row, density, selector, wavenumber, output, input)`
   channel is sampled.
5. Compression correctness: compare structured RSKELF apply/solve with dense
   assembly on smooth single-interface and multi-interface systems.
6. Stress correctness: add corners/triple-junction examples only after RCIP or
   the relevant local correction path is active, because far-field compression
   cannot fix an incorrect local discretization.
