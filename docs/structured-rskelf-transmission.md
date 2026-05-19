# Structured RSKELF for the Figure 4 Junction Example

Status: research/design note. This file records the concrete operator
formulation that should drive the pyFLAM/RSKELF integration. It is deliberately
limited to the three-region, triple-junction example in Greengard-Lee Figure 4
before any general multi-boundary API is designed.

## Literature Anchor

The formulation in this note follows Greengard and Lee, "Stable and accurate
integral equation methods for scattering problems with multiple material
interfaces in two dimensions", Journal of Computational Physics 231 (2012),
2389-2395: <http://math.ewha.ac.kr/~jylee/Paper/junction-jcp12.pdf>.

The two equations that matter here are the global representation in equation
(9) and the junction-stable BIE in equation (13). The compression target is
recursive skeletonization as in Ho and Greengard, "A fast direct solver for
structured linear systems by recursive skeletonization", SIAM Journal on
Scientific Computing 34 (2012), A2507-A2532:
<https://doi.org/10.1137/120866683>.

## Figure 4 Geometry

Greengard-Lee Figure 4 uses three constant-coefficient regions:

| Region | Meaning | $k_r$ | $c_r$ |
| --- | --- | ---: | ---: |
| $\Omega_0$ | exterior | $1$ | $1$ |
| $\Omega_1$ | trapezoidal subdomain | $6$ | $9$ |
| $\Omega_2$ | rectangular subdomain | $10$ | $20$ |

The paper gives the material values as $\epsilon_0=1$, $\epsilon_1=9$, and
$\epsilon_2=20$ and uses $c=\epsilon$ for this transmission condition. The
total interface is the graph with outer traversal

$$
A \to B \to C \to D \to E \to F \to G \to H \to A
$$

plus the internal material interface

$$
A \to D.
$$

The vertices $A$ and $D$ are triple junctions where
$\Omega_0$, $\Omega_1$, and $\Omega_2$ meet.

<figure>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 360" role="img" aria-label="Figure 4 topology with exterior, trapezoid, rectangle, and two triple junctions">
  <rect x="0" y="0" width="760" height="360" fill="#f7f7f4"/>
  <text x="34" y="42" font-size="18" font-family="Arial, sans-serif">&Omega;_0 exterior: k_0=1, c_0=1</text>
  <polygon points="150,122 234,60 352,76 460,122 530,218 420,306 256,304 92,220" fill="#f3f8ff" stroke="#1f2937" stroke-width="3"/>
  <polygon points="150,122 234,60 352,76 460,122" fill="#fff3d8" stroke="#b45309" stroke-width="2"/>
  <polygon points="150,122 460,122 530,218 420,306 256,304 92,220" fill="#e9f7ef" stroke="#047857" stroke-width="2"/>
  <line x1="150" y1="122" x2="460" y2="122" stroke="#7c3aed" stroke-width="4"/>
  <text x="265" y="102" font-size="18" font-family="Arial, sans-serif">&Omega;_1 trapezoid: k_1=6, c_1=9</text>
  <text x="250" y="222" font-size="18" font-family="Arial, sans-serif">&Omega;_2 rectangle: k_2=10, c_2=20</text>
  <text x="276" y="147" font-size="16" font-family="Arial, sans-serif" fill="#5b21b6">Gamma_12</text>
  <text x="222" y="44" font-size="16" font-family="Arial, sans-serif" fill="#92400e">Gamma_01</text>
  <text x="414" y="285" font-size="16" font-family="Arial, sans-serif" fill="#065f46">Gamma_02</text>
  <g font-size="16" font-family="Arial, sans-serif" font-weight="700" fill="#111827">
    <text x="137" y="114">A</text>
    <text x="226" y="52">B</text>
    <text x="350" y="67">C</text>
    <text x="466" y="118">D</text>
    <text x="536" y="224">E</text>
    <text x="417" y="330">F</text>
    <text x="247" y="328">G</text>
    <text x="73" y="224">H</text>
  </g>
  <circle cx="150" cy="122" r="7" fill="#dc2626"/>
  <circle cx="460" cy="122" r="7" fill="#dc2626"/>
  <text x="42" y="338" font-size="15" font-family="Arial, sans-serif">Red vertices A and D are the triple junctions. The sketch preserves the interface graph, not the paper's exact coordinates.</text>
</svg>
</figure>

For the algebra below, use the interface classes

$$
\Gamma_{01} = \Gamma_{AB} \cup \Gamma_{BC} \cup \Gamma_{CD},
$$

$$
\Gamma_{02} =
\Gamma_{DE} \cup \Gamma_{EF} \cup \Gamma_{FG}
\cup \Gamma_{GH} \cup \Gamma_{HA},
$$

and

$$
\Gamma_{12} = \Gamma_{AD}.
$$

Thus

$$
\Gamma = \Gamma_{01} \cup \Gamma_{02} \cup \Gamma_{12}.
$$

The normal convention is metadata. In this note, use

$$
(r_+(g),r_-(g)) =
\begin{cases}
(0,1), & g=01, \\
(0,2), & g=02, \\
(1,2), & g=12.
\end{cases}
$$

Changing an edge orientation only changes the recorded plus/minus side for that
edge; the block formulas are the same once that metadata is fixed.

## Global Representation

Let $\sigma$ be the global charge density and $\mu$ the global dipole density
on the full graph $\Gamma$. Greengard-Lee equation (9) represents the scattered
field in each region by

$$
U_r(x)
=
S_{k_r}(\Gamma,\sigma;x)
+ c_r D_{k_r}(\Gamma,\mu;x),
\qquad x \in \Omega_r.
$$

The critical point for the Figure 4 topology is that $\Gamma$ is the full
interface graph. A source density on $\Gamma_{02}$ contributes to the field in
$\Omega_1$, and a source density on $\Gamma_{01}$ contributes to the field in
$\Omega_2$. This global coupling is what makes the triple-junction equation
well behaved.

The layer potentials are

$$
S_k(C,\sigma;x)
=
\int_C G_k(|x-y|)\sigma(y)\,ds_y,
$$

and

$$
D_k(C,\mu;x)
=
\int_C \frac{\partial G_k(|x-y|)}{\partial n_y}\mu(y)\,ds_y,
$$

with

$$
G_k(r) = \frac{i}{4}H_0^{(1)}(kr).
$$

Their target-normal derivatives are

$$
S'_k(C,\sigma;x)
=
\int_C \frac{\partial G_k(|x-y|)}{\partial n_x}\sigma(y)\,ds_y,
$$

and

$$
D'_k(C,\mu;x)
=
\int_C
\frac{\partial^2 G_k(|x-y|)}{\partial n_x\,\partial n_y}
\mu(y)\,ds_y.
$$

The jump conditions are

$$
[U^{tot}] = 0,
\qquad
\left[\frac{1}{c}\partial_n U^{tot}\right] = 0.
$$

After subtracting the incident field, the right-hand sides are

$$
b_D = -[U^{in}],
\qquad
b_N = -\left[\frac{1}{c}\partial_n U^{in}\right].
$$

## Equation (13) On One Target Interface

For a target point $x \in \Gamma_g$, define

$$
p = r_+(g),
\qquad
m = r_-(g),
$$

so that

$$
k_p = k_{r_+(g)},
\qquad
k_m = k_{r_-(g)},
\qquad
c_p = c_{r_+(g)},
\qquad
c_m = c_{r_-(g)}.
$$

Equation (13a) gives the Dirichlet jump row

$$
\frac{c_p+c_m}{2}\mu(x)
+ S_{k_p}(\Gamma,\sigma;x)
- S_{k_m}(\Gamma,\sigma;x)
+ c_pD_{k_p}(\Gamma,\mu;x)
- c_mD_{k_m}(\Gamma,\mu;x)
= b_D(x).
$$

Equation (13b) gives the scaled normal-flux jump row

$$
-\left(\frac{1}{2c_p}+\frac{1}{2c_m}\right)\sigma(x)
+ \frac{1}{c_p}S'_{k_p}(\Gamma,\sigma;x)
- \frac{1}{c_m}S'_{k_m}(\Gamma,\sigma;x)
+ D'_{k_p}(\Gamma,\mu;x)
- D'_{k_m}(\Gamma,\mu;x)
= b_N(x).
$$

The stabilizing term at the triple junctions is the fused difference

$$
D'_{k_p}(\Gamma,\mu;x) - D'_{k_m}(\Gamma,\mu;x).
$$

For Figure 4, this means the hypersingular row kernels are

$$
D'_1 - D'_6,
\qquad
D'_1 - D'_{10},
\qquad
D'_6 - D'_{10},
$$

on $\Gamma_{01}$, $\Gamma_{02}$, and $\Gamma_{12}$ respectively. They must be
assembled and compressed as difference kernels, not as unrelated flat-matrix
pieces whose cancellation is expected to happen later.

## Concrete Global Block Matrix

Use row classes

$$
g \in \{01,02,12\}
$$

and source classes

$$
h \in \{01,02,12\}.
$$

For each pair $(g,h)$, define a $2 \times 2$ operator block from the source
densities $(\sigma_h,\mu_h)$ to the target equations $(D_g,N_g)$:

$$
B_{g,h}
=
\begin{bmatrix}
B^{D,\sigma}_{g,h} & B^{D,\mu}_{g,h} \\
B^{N,\sigma}_{g,h} & B^{N,\mu}_{g,h}
\end{bmatrix}.
$$

Let $p=r_+(g)$ and $m=r_-(g)$. Then

$$
B^{D,\sigma}_{g,h}
=
S_{k_p}(\Gamma_h,\cdot;\Gamma_g)
- S_{k_m}(\Gamma_h,\cdot;\Gamma_g),
$$

$$
B^{D,\mu}_{g,h}
=
c_pD_{k_p}(\Gamma_h,\cdot;\Gamma_g)
- c_mD_{k_m}(\Gamma_h,\cdot;\Gamma_g)
+ J^{D,\mu}_{g,h},
$$

$$
B^{N,\sigma}_{g,h}
=
\frac{1}{c_p}S'_{k_p}(\Gamma_h,\cdot;\Gamma_g)
- \frac{1}{c_m}S'_{k_m}(\Gamma_h,\cdot;\Gamma_g)
+ J^{N,\sigma}_{g,h},
$$

and

$$
B^{N,\mu}_{g,h}
=
D'_{k_p}(\Gamma_h,\cdot;\Gamma_g)
- D'_{k_m}(\Gamma_h,\cdot;\Gamma_g).
$$

The jump blocks are edgewise diagonal:

$$
J^{D,\mu}_{g,h}
=
\delta_{g,h}\frac{c_p+c_m}{2}I_g,
$$

and

$$
J^{N,\sigma}_{g,h}
=
-\delta_{g,h}
\left(\frac{1}{2c_p}+\frac{1}{2c_m}\right)I_g.
$$

Here $\delta_{g,h}I_g$ means the identity applies only when the same physical
edge point is both target and source. It is not a proxy term.

Define the grouped unknown and right-hand side vectors

$$
\mathbf u_h =
\begin{bmatrix}
\sigma_h \\
\mu_h
\end{bmatrix},
\qquad
\mathbf b_g =
\begin{bmatrix}
b_{D,g} \\
b_{N,g}
\end{bmatrix}.
$$

Then the full dense reference system is the $3 \times 3$ interface-class block
matrix

$$
\begin{bmatrix}
B_{01,01} & B_{01,02} & B_{01,12} \\
B_{02,01} & B_{02,02} & B_{02,12} \\
B_{12,01} & B_{12,02} & B_{12,12}
\end{bmatrix}
\begin{bmatrix}
\mathbf u_{01} \\
\mathbf u_{02} \\
\mathbf u_{12}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf b_{01} \\
\mathbf b_{02} \\
\mathbf b_{12}
\end{bmatrix}.
$$

Expanded, the unknown ordering is

$$
\begin{bmatrix}
\sigma_{01} \\
\mu_{01} \\
\sigma_{02} \\
\mu_{02} \\
\sigma_{12} \\
\mu_{12}
\end{bmatrix}.
$$

The implementation should keep these axes explicit rather than flattening them
before the compression layer.

## Numeric Kernel Channels

In this section, the subscript on an operator is the wavenumber. For example,
$S_6$ means $S_{k=6}$ and $D'_{10}$ means $D'_{k=10}$.

With the Figure 4 coefficients inserted, a target row on $\Gamma_{01}$ uses

$$
B^{D,\sigma}_{01,h}
=
S_1(\Gamma_h,\cdot;\Gamma_{01})
- S_6(\Gamma_h,\cdot;\Gamma_{01}),
$$

$$
B^{D,\mu}_{01,h}
=
D_1(\Gamma_h,\cdot;\Gamma_{01})
-9D_6(\Gamma_h,\cdot;\Gamma_{01})
+\delta_{01,h}5I_{01},
$$

$$
B^{N,\sigma}_{01,h}
=
S'_1(\Gamma_h,\cdot;\Gamma_{01})
-\frac{1}{9}S'_6(\Gamma_h,\cdot;\Gamma_{01})
-\delta_{01,h}\frac{5}{9}I_{01},
$$

and

$$
B^{N,\mu}_{01,h}
=
D'_1(\Gamma_h,\cdot;\Gamma_{01})
-D'_6(\Gamma_h,\cdot;\Gamma_{01}).
$$

A target row on $\Gamma_{02}$ uses

$$
B^{D,\sigma}_{02,h}
=
S_1(\Gamma_h,\cdot;\Gamma_{02})
- S_{10}(\Gamma_h,\cdot;\Gamma_{02}),
$$

$$
B^{D,\mu}_{02,h}
=
D_1(\Gamma_h,\cdot;\Gamma_{02})
-20D_{10}(\Gamma_h,\cdot;\Gamma_{02})
+\delta_{02,h}\frac{21}{2}I_{02},
$$

$$
B^{N,\sigma}_{02,h}
=
S'_1(\Gamma_h,\cdot;\Gamma_{02})
-\frac{1}{20}S'_{10}(\Gamma_h,\cdot;\Gamma_{02})
-\delta_{02,h}\frac{21}{40}I_{02},
$$

and

$$
B^{N,\mu}_{02,h}
=
D'_1(\Gamma_h,\cdot;\Gamma_{02})
-D'_{10}(\Gamma_h,\cdot;\Gamma_{02}).
$$

A target row on the internal interface $\Gamma_{12}$ uses

$$
B^{D,\sigma}_{12,h}
=
S_6(\Gamma_h,\cdot;\Gamma_{12})
- S_{10}(\Gamma_h,\cdot;\Gamma_{12}),
$$

$$
B^{D,\mu}_{12,h}
=
9D_6(\Gamma_h,\cdot;\Gamma_{12})
-20D_{10}(\Gamma_h,\cdot;\Gamma_{12})
+\delta_{12,h}\frac{29}{2}I_{12},
$$

$$
B^{N,\sigma}_{12,h}
=
\frac{1}{9}S'_6(\Gamma_h,\cdot;\Gamma_{12})
-\frac{1}{20}S'_{10}(\Gamma_h,\cdot;\Gamma_{12})
-\delta_{12,h}\frac{29}{360}I_{12},
$$

and

$$
B^{N,\mu}_{12,h}
=
D'_6(\Gamma_h,\cdot;\Gamma_{12})
-D'_{10}(\Gamma_h,\cdot;\Gamma_{12}).
$$

The source class $h$ is arbitrary in every formula above. For example,
$B^{N,\mu}_{12,02}$ maps dipoles on the exterior-rectangle boundary
$\Gamma_{02}$ into the internal-interface flux row by the fused kernel

$$
D'_6(\Gamma_{02},\cdot;\Gamma_{12})
-D'_{10}(\Gamma_{02},\cdot;\Gamma_{12}).
$$

That is the behavior pyFLAM has to see during compression.

## Structured RSKELF Compression

The RSKELF tree should be geometric, built on physical source points in

$$
\Gamma_{01} \cup \Gamma_{02} \cup \Gamma_{12}.
$$

The algebraic axes remain explicit metadata:

$$
\text{source family} \in \{\sigma,\mu\},
\qquad
\text{row equation} \in \{D,N\},
\qquad
\text{interface class} \in \{01,02,12\}.
$$

For a source node $B$ and a candidate column family $C_B$, the outgoing ID
sample must stack all far target rows that see that source family:

$$
Y_B(C_B)
=
\operatorname{stack}_{g,q}
\left(
B^{q,\beta}_{g,h(B)}[\text{far rows of } \Gamma_g,\ C_B]
\right),
$$

where $q \in \{D,N\}$, $\beta \in \{\sigma,\mu\}$ is the selected source
density, and $h(B)$ is the source interface class for the selected columns. If
a geometric box contains panels from more than one interface class, $C_B$ must
be split by source class before the ID or the class axis must be retained inside
the sample.

The proxy sample for that same column family must use the exact fused channel
for each target class $g$. For a charge column, the proxy rows are

$$
\begin{array}{c|c|c}
\text{target class } g
& \text{Dirichlet proxy}
& \text{flux proxy} \\
\hline
01 & S_1-S_6 & S'_1-\frac{1}{9}S'_6 \\
02 & S_1-S_{10} & S'_1-\frac{1}{20}S'_{10} \\
12 & S_6-S_{10} & \frac{1}{9}S'_6-\frac{1}{20}S'_{10}
\end{array}
$$

For a dipole column, the proxy rows are

$$
\begin{array}{c|c|c}
\text{target class } g
& \text{Dirichlet proxy}
& \text{flux proxy} \\
\hline
01 & D_1-9D_6 & D'_1-D'_6 \\
02 & D_1-20D_{10} & D'_1-D'_{10} \\
12 & 9D_6-20D_{10} & D'_6-D'_{10}
\end{array}
$$

This is the main pyFLAM requirement. The ID is not built from one scalar kernel
after flattening. It is built from the stacked response of the selected source
family under every fused Figure 4 row kernel that can observe it.

Equivalently, for each output channel $q$ and input density $\beta$, pyFLAM
needs a proxy callback for the corresponding fused kernel:

$$
K^{q,\beta}_g
\in
\left\{
S_1-S_6,\,
S_1-S_{10},\,
S_6-S_{10},\,
D_1-9D_6,\,
D_1-20D_{10},\,
9D_6-20D_{10},\,
S'_1-\frac{1}{9}S'_6,\,
S'_1-\frac{1}{20}S'_{10},\,
\frac{1}{9}S'_6-\frac{1}{20}S'_{10},\,
D'_1-D'_6,\,
D'_1-D'_{10},\,
D'_6-D'_{10}
\right\}.
$$

The callback must also know the source interface class and target row class so
that special quadrature, orientation, and near/far exclusions are applied to
the right physical panels.

## Near, Jump, and Triple-Junction Blocks

Only far interactions are proxy compressed. The following are local algebra and
must stay out of the proxy sample:

- the identity terms $5I_{01}$, $\frac{21}{2}I_{02}$, and
  $\frac{29}{2}I_{12}$ in the Dirichlet-dipole rows;
- the identity terms $-\frac{5}{9}I_{01}$, $-\frac{21}{40}I_{02}$, and
  $-\frac{29}{360}I_{12}$ in the flux-charge rows;
- same-panel singular, principal-value, and finite-part quadrature;
- adjacent-panel corrections across ordinary vertices;
- all panels incident on the triple junctions $A$ and $D$ until the local
  junction correction, RCIP path, or equivalent exact local discretization is
  active;
- Schur-complement updates created during elimination.

The local block has the form

$$
A_{\mathrm{local}}
=
A_{\mathrm{smooth\ near}}
+ A_{\mathrm{jump}}
+ A_{\mathrm{special}}
+ A_{\mathrm{junction}}
+ A_{\mathrm{Schur}}.
$$

Far compression can preserve the well-conditioned equation only if this local
block already represents equation (13) correctly near $A$ and $D$.

## Design Consequence

The pychunkie-to-pyFLAM boundary should pass a structured operator, not a flat
matrix plus a single kernel. For the Figure 4 example, the minimum structured
metadata is:

- physical points and panels grouped by $\Gamma_{01}$, $\Gamma_{02}$, and
  $\Gamma_{12}$;
- density families $\sigma$ and $\mu$ attached to each source point;
- row equations $D$ and $N$ attached to each target point;
- target-side material pairs $(0,1)$, $(0,2)$, and $(1,2)$;
- fused far kernels $K^{q,\beta}_g$ with their numeric coefficients;
- local block callbacks for jumps, special quadrature, and triple-junction
  corrections.

The first implementation target should be this exact Figure 4 operator. A
dense reference assembly should expose all nine interface-class blocks
$B_{g,h}$, and the structured RSKELF path should prove that every ID samples
the relevant fused proxy channels instead of reusing one flattened scalar
kernel.

## Verification Targets

1. Assemble the dense Figure 4 block matrix and verify every $B_{g,h}$ block
   against the formulas above.
2. Instrument proxy sampling and assert that charge columns sample the three
   charge proxy row pairs and dipole columns sample the three dipole proxy row
   pairs.
3. Verify that identity and special-quadrature blocks never appear in proxy
   samples.
4. Compare structured RSKELF apply/solve results against dense assembly on a
   small Figure 4 discretization with triple-junction panels kept local.
5. Add the triple-junction local correction path before using this as a
   convergence test near vertices $A$ and $D$.
