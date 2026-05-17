# Singularity Extraction Reference

Update trigger: revise this document whenever the singularity metadata basis,
kernel convention, product-rule extraction, or family-specific local expansion
changes.

This document records the mathematical formulas used to extract singular
kernel parts into `SingularityInfo`. The implementation target is:

$$
K(x,y) = K_{\mathrm{sing}}(x,y) + K_{\mathrm{smooth}}(x,y),
$$

where $K_{\mathrm{sing}}$ is represented as smooth-amplitude 2D Laplace basis
terms and $K_{\mathrm{smooth}}$ is verified numerically before the metadata is
used by quadrature or backend dispatch.

## Conventions

Let:

$$
r = x - y,\qquad \rho = |r|,\qquad r_a = x_a - y_a.
$$

Target derivatives are written as $\partial_a = \partial / \partial x_a$.
Source and target normals are $n_s$ and $n_t$. Repeated Cartesian indices are
summed over $a,b,k \in \{0,1\}$.

The canonical 2D Laplace basis is:

$$
G = -\frac{\log \rho}{2\pi}
    = -\frac{\log \rho^2}{4\pi},
$$

$$
G_a = \partial_a G
    = -\frac{r_a}{2\pi\rho^2},
$$

$$
G_{ab} = \partial_a\partial_b G
       = \frac{2r_a r_b - \delta_{ab}\rho^2}{2\pi\rho^4}.
$$

Kernel values use component-first tensor layout:

$$
K_{oits} = K_{oi}(x_t,y_s).
$$

A singular metadata term has the form:

$$
T_{oi}(x,y) = c_{oi\alpha}(x,y)\,B_\alpha(x,y),
$$

where $B_\alpha$ is one of $G$, $G_a$, $G_{ab}$, or a future higher derivative
basis.

## Smooth Amplitudes

The multiplier $c_{oi\alpha}(x,y)$ is a smooth amplitude, not necessarily a
constant. This is different from the leading singular coefficient:

$$
c_{oi\alpha}^{0}(y) = \lim_{x\to y} c_{oi\alpha}(x,y),
$$

which is often constant or a geometry-dependent tensor such as a normal
component. The metadata keeps the full smooth amplitude when that makes the
subtracted remainder smoother.

This is compatible with Helsing-Ojala and other special quadrature rules. On a
source panel, the singular contribution has the local form:

$$
\int_{\Gamma_p} c(x,y)B(x,y)\sigma(y)\,ds_y.
$$

For a fixed target $x$, define the panel density seen by the special rule as:

$$
q_x(y)=c(x,y)\sigma(y)J(y),
$$

where $J(y)$ is the source speed or quadrature Jacobian. If $c(x,y)$,
$\sigma(y)$, and $J(y)$ are smooth on the panel, then $q_x(y)$ is a smooth
panel function. The special rule only needs to know which singular basis
$B(x,y)$ is present; the smooth amplitude is absorbed into the interpolated
panel data.

For example, Helmholtz uses:

$$
H_{k,\mathrm{sing}}(x,y)=J_0(k|x-y|)G(x,y),
$$

with smooth amplitude $J_0(k|x-y|)$ and leading coefficient $1$. Likewise,
Cartesian Laplace metadata writes:

$$
D(x,y)=-n_s[a](y)G_a(x,y),
$$

so $-n_s[a](y)$ is stored as an amplitude because the canonical basis is
geometry-free $G_a$. If the basis were instead the source-normal Laplace
derivative, the corresponding multiplier would be the constant $-1$.

## Product Rule

For coefficiented logarithmic singularities, write:

$$
F(x,y) = a(x,y)\,G(x,y).
$$

Then:

$$
\partial_a F = a\,G_a + a_a\,G,
$$

and:

$$
\partial_a\partial_b F
= a\,G_{ab} + a_a\,G_b + a_b\,G_a + a_{ab}\,G.
$$

This product-rule form is the preferred way to generate derivative selectors.
It avoids duplicating selector-specific singular formulas and lets quadrature
see one canonical basis.

Normal contractions are applied to the coefficient side:

$$
\partial_{n_t} F = n_t[a]\partial_a F,
$$

$$
\partial_{n_s} F = -n_s[a]\partial_a F
$$

when the layer convention uses target derivatives of $G(x,y)$ and source
normal derivatives through $-\partial_{n_s}$.

## Laplace

The scalar Laplace single-layer singularity is:

$$
S = G.
$$

The spatial gradient selector is:

$$
SG_a = G_a.
$$

The target-normal derivative of the single layer is:

$$
S' = n_t[a]G_a.
$$

The double-layer value, using the project convention, is:

$$
D = -n_s[a]G_a.
$$

The double-layer gradient is:

$$
DG_a = -n_s[b]G_{ab}.
$$

The hypersingular normal-normal derivative is:

$$
D' = -n_t[a]n_s[b]G_{ab}.
$$

Boundary jump terms such as $\pm \frac{1}{2}I$ are not kernel singularities.
They belong to `BoundaryTrace`.

## Helmholtz

The outgoing Helmholtz Green function is:

$$
H_k(\rho) = \frac{i}{4}H_0^{(1)}(k\rho).
$$

Its logarithmic singular part is:

$$
H_{k,\mathrm{sing}} = J_0(k\rho)\,G.
$$

Set:

$$
a(\rho) = J_0(k\rho).
$$

Then:

$$
H_{k,\mathrm{sing}} = aG,
$$

$$
\partial_a H_{k,\mathrm{sing}} = aG_a + a_aG,
$$

and:

$$
\partial_a\partial_b H_{k,\mathrm{sing}}
= aG_{ab} + a_aG_b + a_bG_a + a_{ab}G.
$$

The coefficient derivatives are:

$$
a_a = -kJ_1(k\rho)\frac{r_a}{\rho},
$$

$$
a_{ab}
= F''(\rho)\frac{r_a r_b}{\rho^2}
 + \frac{F'(\rho)}{\rho}
   \left(\delta_{ab} - \frac{r_a r_b}{\rho^2}\right),
$$

where:

$$
F(\rho)=J_0(k\rho),\qquad
F'(\rho)=-kJ_1(k\rho),
$$

and:

$$
F''(\rho)=-\frac{k^2}{2}\left(J_0(k\rho)-J_2(k\rho)\right).
$$

At coincidence, use the stable limits:

$$
a(0)=1,\qquad a_a(0)=0,\qquad a_{ab}(0)=-\frac{k^2}{2}\delta_{ab}.
$$

The Helmholtz selectors follow by applying the Laplace contractions to
$aG$:

$$
S = aG,
$$

$$
SG_a = aG_a + a_aG,
$$

$$
S' = n_t[a]\left(aG_a + a_aG\right),
$$

$$
D = -n_s[a]\left(aG_a + a_aG\right),
$$

$$
DG_a = -n_s[b]\left(aG_{ab} + a_aG_b + a_bG_a + a_{ab}G\right),
$$

and:

$$
D' = -n_t[a]n_s[b]
     \left(aG_{ab} + a_aG_b + a_bG_a + a_{ab}G\right).
$$

## Biharmonic

The 2D biharmonic fundamental solution is:

$$
B = \frac{\rho^2\log\rho}{8\pi}.
$$

In the Laplace basis:

$$
B = -\frac{\rho^2}{4}G.
$$

Let:

$$
c = -\frac{\rho^2}{4}.
$$

Then:

$$
c_a = -\frac{r_a}{2},
$$

and:

$$
c_{ab} = -\frac{\delta_{ab}}{2}.
$$

The first derivatives are:

$$
B_a = c_aG + cG_a
    = -\frac{r_a}{2}G - \frac{\rho^2}{4}G_a.
$$

The second derivatives are:

$$
B_{ab}
= c_{ab}G + c_aG_b + c_bG_a + cG_{ab},
$$

or:

$$
B_{ab}
= -\frac{\delta_{ab}}{2}G
   -\frac{r_a}{2}G_b
   -\frac{r_b}{2}G_a
   -\frac{\rho^2}{4}G_{ab}.
$$

Higher biharmonic selectors should be generated by continuing the same
product-rule machinery, not by hand-writing one-off split formulas.

## Stokes

For 2D Stokes with viscosity $\mu$, the single-layer velocity singularity can
be represented through $G$ and $B_{ij}$:

$$
S_{ij}
= \frac{1}{\mu}\left(\delta_{ij}G + B_{ij}\right)
  + K_{ij,\mathrm{smooth}}.
$$

The single-layer pressure singularity is:

$$
p_j = -G_j.
$$

The double-layer velocity singularity is:

$$
D_{ij}
= n_s[k]\,r_k\,G_{ij} - \delta_{ij}n_s[k]G_k.
$$

The single-layer traction singularity has the same tensor structure with the
target normal replacing the source normal and the appropriate traction sign:

$$
T^{S}_{ij}
\sim -n_t[k]\,r_k\,G_{ij} + \delta_{ij}n_t[k]G_k.
$$

The double-layer pressure singularity is:

$$
p^{D}_j = 2\mu\,n_s[k]G_{jk}.
$$

Stokes double-layer gradients and double-layer traction require higher
derivative product-rule terms before special quadrature should consume their
metadata.

## Elasticity

For 2D isotropic elasticity, the single-displacement singularity can be written
with $G$ and $B_{ij}$:

$$
E_{ij}
= 2\pi(\gamma-\beta)\delta_{ij}G
 + 4\pi\gamma B_{ij}.
$$

Single-gradient metadata follows by differentiating:

$$
\partial_a E_{ij}
= 2\pi(\gamma-\beta)\delta_{ij}G_a
 + 4\pi\gamma B_{ij,a}.
$$

Traction, double-layer, and stress forms should be derived through the
elasticity stress tensor and reduced to $G_a$, $G_{ab}$, and higher derivative
bases only after smooth-remainder tests are added.

## Verification Requirements

For each selector with declared metadata, test:

$$
R(x,y) = K(x,y) - K_{\mathrm{sing}}(x,y).
$$

As $\rho \to 0$ along multiple directions, verify:

$$
|R(x,y)| < C
$$

for bounded remainders, and verify direction-independent limits when the
metadata declares:

$$
R \in C^\infty.
$$

Derivative metadata should also satisfy finite-difference consistency. For
example:

$$
\frac{K_s(x+h e_a,y)-K_s(x-h e_a,y)}{2h}
\approx K_{sg,a}(x,y),
$$

after subtracting their corresponding singular expansions.

Quadrature code may consume singular metadata only after these remainder tests
exist for the relevant family and selector.
