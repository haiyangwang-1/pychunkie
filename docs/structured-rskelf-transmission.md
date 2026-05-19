# Structured RSKELF for Multi-Material Junction Systems

Status: research/design note. This file records the formulation that should
drive the pyFLAM/RSKELF integration for multi-boundary systems with more than
one density per boundary. It is not an implemented fast-direct backend.

## Literature Anchors

The operator-block formulation in this note is based on:

- Greengard and Lee, "Stable and accurate integral equation methods for
  scattering problems with multiple material interfaces in two dimensions",
  Journal of Computational Physics 231 (2012), 2389-2395:
  <http://math.ewha.ac.kr/~jylee/Paper/junction-jcp12.pdf>.

The fast-direct compression target is recursive skeletonization:

- Ho and Greengard, "A fast direct solver for structured linear systems by
  recursive skeletonization", SIAM Journal on Scientific Computing 34 (2012),
  A2507-A2532: <https://doi.org/10.1137/120866683>.

The key modeling decision is to follow Greengard-Lee equation (9), not a local
two-region representation. Equation (9) uses two global densities on the whole
material interface. The derived equation (13) is the well-conditioned block BIE
that avoids the unmatched hypersingular junction terms which break the local
Muller/Rokhlin representation near triple junctions.

## Physical Model

Let the plane be partitioned into regions

$$
\Omega_0, \Omega_1, \ldots, \Omega_M,
$$

where $\Omega_0$ is the exterior region. Each region has constant Helmholtz
wavenumber $k_r$ and material coefficient $c_r$. The total field satisfies

$$
(\Delta + k_r^2)U^{tot}_r = 0
\qquad \text{in } \Omega_r.
$$

The total field is decomposed as

$$
U^{tot} = U^{in} + U,
$$

where $U^{in}$ is known and $U$ is the unknown scattered field. Across each
material interface, Greengard-Lee use the jump conditions

$$
\left[U^{tot}\right] = 0,
\qquad
\left[\frac{1}{c}\frac{\partial U^{tot}}{\partial n}\right] = 0.
$$

Equivalently, the scattered field satisfies

$$
\left[U\right] = -\left[U^{in}\right],
\qquad
\left[\frac{1}{c}\frac{\partial U}{\partial n}\right]
=
-\left[\frac{1}{c}\frac{\partial U^{in}}{\partial n}\right].
$$

For an oriented interface edge $\Gamma_e$, let the normal point from region
$r_-(e)$ to region $r_+(e)$. On that edge,

$$
[f]_e = f_{+}|_{\Gamma_e} - f_{-}|_{\Gamma_e},
$$

with $f_\pm$ denoting traces from $\Omega_{r_\pm(e)}$.

The total interface is

$$
\Gamma = \bigcup_e \Gamma_e.
$$

## Layer Operators

For a curve $C$, Greengard-Lee define

$$
S_k(C,\sigma;x)
=
\int_C G_k(|x-y|)\,\sigma(y)\,ds_y,
$$

and

$$
D_k(C,\mu;x)
=
\int_C \frac{\partial G_k(|x-y|)}{\partial n_y}\,\mu(y)\,ds_y.
$$

The normal derivatives are

$$
S'_k(C,\sigma;x)
=
\int_C \frac{\partial G_k(|x-y|)}{\partial n_x}\,\sigma(y)\,ds_y,
$$

and

$$
D'_k(C,\mu;x)
=
\int_C
\frac{\partial^2 G_k(|x-y|)}{\partial n_x\,\partial n_y}
\,\mu(y)\,ds_y.
$$

Here

$$
G_k(r) = \frac{i}{4}H_0^{(1)}(kr)
$$

is the outgoing two-dimensional Helmholtz Green function. The operator $S'_k$
is interpreted in principal value sense; $D'_k$ is hypersingular and interpreted
as a finite-part operator.

## Equation (9): Global Representation

Greengard-Lee equation (9) represents the scattered field in every region
$\Omega_r$ using the same two global densities $\sigma$ and $\mu$ on the total
interface $\Gamma$:

$$
U_r(x)
=
S_{k_r}(\Gamma,\sigma;x)
+ c_r D_{k_r}(\Gamma,\mu;x),
\qquad x \in \Omega_r.
$$

The important point is that the source curve is the total interface $\Gamma$,
not only the boundary of $\Omega_r$. Thus every density value participates in
the representation of every region. This is the feature that removes unmatched
hypersingular terms at material junctions.

## Derived BIE: Equation (13)

Let $x \in \Gamma_e$ and write

$$
r_+ = r_+(e), \qquad r_- = r_-(e),
$$

with

$$
k_\pm = k_{r_\pm},
\qquad
c_\pm = c_{r_\pm}.
$$

Taking traces of equation (9) in the jump conditions gives the Dirichlet row

$$
\frac{c_+ + c_-}{2}\,\mu(x)
+ S_{k_+}(\Gamma,\sigma;x)
- S_{k_-}(\Gamma,\sigma;x)
+ c_+D_{k_+}(\Gamma,\mu;x)
- c_-D_{k_-}(\Gamma,\mu;x)
=
-[U^{in}]_e(x),
$$

and the flux row

$$
-\left(\frac{1}{2c_+}+\frac{1}{2c_-}\right)\sigma(x)
+ \frac{1}{c_+}S'_{k_+}(\Gamma,\sigma;x)
- \frac{1}{c_-}S'_{k_-}(\Gamma,\sigma;x)
+ D'_{k_+}(\Gamma,\mu;x)
- D'_{k_-}(\Gamma,\mu;x)
=
-\left[\frac{1}{c}\partial_n U^{in}\right]_e(x).
$$

This is the BIE formulation pychunkie should use as the canonical
multi-material junction example. The unknowns are one global charge density
$\sigma$ and one global dipole density $\mu$ over $\Gamma$.

The stabilizing feature is visible in the last term of the flux row:

$$
D'_{k_+}(\Gamma,\mu;x) - D'_{k_-}(\Gamma,\mu;x).
$$

Even at a triple junction, the hypersingular contribution appears as a
difference of global-interface kernels, not as separate unmatched local edge
terms.

## Operator Block Layout

Discretize $\Gamma$ by edge views $\Gamma_f$. The row spaces are

$$
(e,D) \quad \text{and} \quad (e,N),
$$

where $D$ is the Dirichlet jump equation and $N$ is the scaled normal-flux jump
equation on target edge $\Gamma_e$. The column spaces are

$$
(f,\sigma) \quad \text{and} \quad (f,\mu),
$$

where $f$ is the source edge.

For each target edge $\Gamma_e$ and source edge $\Gamma_f$, equation (13)
induces four operator blocks:

$$
\begin{aligned}
A_{e,D;\,f,\sigma}
&=
S_{k_+}(\Gamma_f,\cdot;\Gamma_e)
-
S_{k_-}(\Gamma_f,\cdot;\Gamma_e), \\
A_{e,D;\,f,\mu}
&=
c_+D_{k_+}(\Gamma_f,\cdot;\Gamma_e)
-
c_-D_{k_-}(\Gamma_f,\cdot;\Gamma_e)
+ \delta_{ef}\,\frac{c_+ + c_-}{2}I_e, \\
A_{e,N;\,f,\sigma}
&=
\frac{1}{c_+}S'_{k_+}(\Gamma_f,\cdot;\Gamma_e)
-
\frac{1}{c_-}S'_{k_-}(\Gamma_f,\cdot;\Gamma_e)
-
\delta_{ef}\left(\frac{1}{2c_+}+\frac{1}{2c_-}\right)I_e, \\
A_{e,N;\,f,\mu}
&=
D'_{k_+}(\Gamma_f,\cdot;\Gamma_e)
-
D'_{k_-}(\Gamma_f,\cdot;\Gamma_e).
\end{aligned}
$$

Here $\delta_{ef}$ means the jump term is present only when the source and
target edge views coincide, away from endpoint duplication. The signs assume the
normal on $\Gamma_e$ points from $r_-(e)$ to $r_+(e)$.

The discrete system has the block form

$$
\begin{bmatrix}
A_{D,\sigma} & A_{D,\mu} \\
A_{N,\sigma} & A_{N,\mu}
\end{bmatrix}
\begin{bmatrix}
\sigma \\
\mu
\end{bmatrix}
=
\begin{bmatrix}
-[U^{in}] \\
-[(1/c)\partial_n U^{in}]
\end{bmatrix}.
$$

For vector or tensor kernels, each scalar block above generalizes to

$$
A_{\alpha,\beta}^{o,i}(t,s),
$$

where $\alpha$ is a row space, $\beta$ is a column space, $o$ is the output
component, $i$ is the input component, $t$ is the target point, and $s$ is the
source point. For this scalar Helmholtz junction model, $o=i=1$.

## Structured Operator Terms

The system layer should expose the operator as a sum of explicit terms. Each
term $\tau$ should carry the following metadata.

| Field | Meaning |
| --- | --- |
| $\alpha_\tau$ | row space, such as $(e,D)$ or $(e,N)$ |
| $\beta_\tau$ | column space, such as $(f,\sigma)$ or $(f,\mu)$ |
| $K_\tau$ | one fused kernel expression from equation (13) |
| $a_\tau$ | material coefficient, such as $c_+$, $-c_-$, $1/c_+$, or $-1/c_-$ |
| $s_\tau$ | trace side and edge-normal convention |
| $\ell_\tau$ | locality class: far-compressible, near-exact, jump, or correction |

The term contribution has the form

$$
A_\tau(t,s)
=
a_\tau K_\tau(x_t,x_s).
$$

For example, the far-compressible part of the flux-dipole block is the fused
difference

$$
K_{\tau}(x_t,x_s)
=
D'_{k_+}(x_t,x_s) - D'_{k_-}(x_t,x_s).
$$

This fused expression is the unit of compression. The backend should not sample
$D'_{k_+}$ and $D'_{k_-}$ as unrelated hypersingular operators and rely on
flattened-matrix subtraction to recover the cancellation later.

## RSKELF Sampling Rule

The geometric tree should be built over physical source points on the total
interface $\Gamma$. Algebraic density axes remain metadata attached to those
points.

For a tree node $B$ and a candidate column group $C_B$, the ID sample must stack
all far-field row families that can see that column group:

$$
Y_B(C_B)
=
\operatorname{stack}_{T,q,o}
\left(
A[q,C_B][o,:,T,B]
\right),
$$

where $T$ ranges over far target clusters, $q$ ranges over row equations
$(e,D)$ and $(e,N)$, and $o$ ranges over output components.

The proxy part should be assembled from the same fused block kernels:

$$
P_B(C_B)
=
\operatorname{stack}_{\tau}
\left(
K_{\tau}^{proxy}
    (\Xi_B,\ x_s \in B)
\right),
$$

where $\Xi_B$ is the proxy surface for node $B$ and $\tau$ ranges over
far-compressible equation (13) terms touching $C_B$.

For a charge column group, the proxy samples include the fused row kernels

$$
S_{k_+}^{proxy} - S_{k_-}^{proxy}
\qquad \text{and} \qquad
\frac{1}{c_+}(S'_{k_+})^{proxy}
- \frac{1}{c_-}(S'_{k_-})^{proxy}.
$$

For a dipole column group, the proxy samples include

$$
c_+(D_{k_+})^{proxy}
- c_-(D_{k_-})^{proxy}
\qquad \text{and} \qquad
(D'_{k_+})^{proxy}
- (D'_{k_-})^{proxy}.
$$

Jump terms, same-panel special quadrature, adjacent-panel corrections, and RCIP
corner corrections are not proxy terms. They remain in the near/local blocks.

For nonsymmetric systems the reverse direction must also be sampled:

$$
Z_B(R_B)
=
\operatorname{stack}_{S,p,i}
\left(
A[R_B,p][:,i,B,S]^*
\right),
$$

where $S$ ranges over far source clusters, $p$ ranges over density families
$\sigma$ and $\mu$, and $i$ ranges over input components.

The ID for a source block is then built from

$$
\begin{bmatrix}
Y_B(C_B) \\
P_B(C_B)
\end{bmatrix},
$$

and in the nonsymmetric case from the corresponding incoming/outgoing samples.
This is the behavior we need from pyFLAM: proxy sampling is per row equation,
per density, and per fused equation (13) block, not one scalar kernel reused for
a flattened matrix.

## Skeleton Grouping Policy

There is not one universally correct grouping choice. The backend should make
the grouping explicit and testable.

- `dof`: ID may select individual scalar DOFs. This is closest to a flat matrix
  and can be efficient, but it loses physical grouping.
- `point_family`: ID selects one density family at a source point, either
  $\sigma$ or $\mu$. This is the conservative first target for equation (13).
- `point_all_densities`: ID selects both $\sigma$ and $\mu$ attached to a source
  point. This is more expensive, but best preserves physical point grouping for
  strongly coupled systems.

The first production target should be `point_family`, with tests comparing it
against dense assembly for the Greengard-Lee multi-material junction system.
`point_all_densities` should remain available as a stability experiment.

## Near, Jump, and Correction Blocks

RSKELF compression should only approximate far-field interactions. The backend
must keep these contributions exact in the local blocks:

- the identity/jump terms in equation (13);
- same-panel singular and hypersingular blocks;
- adjacent/near-panel quadrature corrections;
- RCIP-compressed local corner or junction blocks;
- sparse replacement blocks from special quadrature.

The local block seen during elimination is therefore

$$
A_{\mathrm{local}}
= A_{\mathrm{smooth\ near}}
  + A_{\mathrm{jump}}
  + A_{\mathrm{special\ corrections}}
  + A_{\mathrm{Schur\ updates}}.
$$

Proxy samples are for the complement of the near set. This is the clean way to
respect both recursive skeletonization and the junction-stable structure of
Greengard-Lee equation (13).

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

1. Dense reference: assemble Greengard-Lee equation (13) for a smooth
   two-region interface and compare every block against a hand-built dense
   matrix.
2. Junction reference: assemble equation (13) on a multi-edge `ChunkGraph` with
   a triple junction and verify that the operator uses global-interface
   difference kernels, not local-region-boundary kernels.
3. Legacy parity: port compact fixtures from `tests_old/test_devtools_parity.py`
   around the Helmholtz transmission block, but reinterpret the target operator
   in the global equation (9)/(13) language.
4. Proxy instrumentation: run structured pyFLAM on a tiny system and assert that
   each expected $(\text{row equation}, \text{density}, \text{selector},
   \text{wavenumber}, \text{material coefficient})$ channel is sampled.
5. Compression correctness: compare structured RSKELF apply/solve with dense
   assembly on smooth single-interface and multi-material junction systems.
6. Stress correctness: add corners/triple-junction examples only after RCIP or
   the relevant local correction path is active, because far-field compression
   cannot fix an incorrect local discretization.
