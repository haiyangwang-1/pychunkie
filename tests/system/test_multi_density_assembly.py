import numpy as np

from chunkie.geometry import circle
from chunkie.kernels import kernel
from chunkie.quadrature import dense_panel_operator_matrix
from chunkie.system import (
    BoundaryEquation,
    BoundaryTrace,
    DensitySpace,
    IntegralSystem,
    JumpTerm,
    LayerPotential,
)


def test_dense_assembly_builds_two_density_two_equation_blocks():
    boundary = circle(quadrature_order=6, panel_count=5)
    single = kernel("laplace", selector="s")
    base = dense_panel_operator_matrix(boundary.pointinfo, boundary.pointinfo, single)
    identity = np.eye(boundary.point_count)

    sigma = DensitySpace("sigma", boundary)
    mu = DensitySpace("mu", boundary)
    sigma_layer = LayerPotential("sigma_single", boundary, single, "sigma", coefficient=2.0)
    mu_layer = LayerPotential("mu_single", boundary, single, "mu", coefficient=-3.0)
    sigma_layer_row2 = LayerPotential(
        "sigma_single_row2", boundary, single, "sigma", coefficient=0.5
    )
    mu_layer_row2 = LayerPotential("mu_single_row2", boundary, single, "mu", coefficient=4.0)

    system = IntegralSystem(
        name="two_density_block_reference",
        geometry=boundary,
        unknowns=(sigma, mu),
        equations=(
            BoundaryEquation(
                "dirichlet",
                boundary,
                (
                    BoundaryTrace(sigma_layer, boundary, "exterior"),
                    BoundaryTrace(mu_layer, boundary, "exterior", jump=JumpTerm(0.25, "mu")),
                ),
                np.zeros(boundary.point_count),
            ),
            BoundaryEquation(
                "neumann",
                boundary,
                (
                    BoundaryTrace(
                        sigma_layer_row2, boundary, "exterior", jump=JumpTerm(-0.5, "sigma")
                    ),
                    BoundaryTrace(mu_layer_row2, boundary, "exterior"),
                ),
                np.zeros(boundary.point_count),
            ),
        ),
    )

    matrix = system.assemble().to_dense()
    n = boundary.point_count

    np.testing.assert_allclose(matrix[:n, :n], 2.0 * base)
    np.testing.assert_allclose(matrix[:n, n:], -3.0 * base + 0.25 * identity)
    np.testing.assert_allclose(matrix[n:, :n], 0.5 * base - 0.5 * identity)
    np.testing.assert_allclose(matrix[n:, n:], 4.0 * base)


def test_dense_solve_reconstructs_multiple_density_vectors():
    boundary = circle(quadrature_order=4, panel_count=4)
    single = kernel("laplace", selector="s")
    sigma = DensitySpace("sigma", boundary)
    mu = DensitySpace("mu", boundary)
    sigma_layer = LayerPotential("sigma_zero", boundary, single, "sigma", coefficient=0.0)
    mu_layer = LayerPotential("mu_zero", boundary, single, "mu", coefficient=0.0)
    sigma_rhs = boundary.positions[0]
    mu_rhs = 2.0 * boundary.positions[1]

    system = IntegralSystem(
        name="two_density_identity",
        geometry=boundary,
        unknowns=(sigma, mu),
        equations=(
            BoundaryEquation(
                "sigma_eq",
                boundary,
                (BoundaryTrace(sigma_layer, boundary, "exterior", jump=JumpTerm(1.0, "sigma")),),
                sigma_rhs,
            ),
            BoundaryEquation(
                "mu_eq",
                boundary,
                (BoundaryTrace(mu_layer, boundary, "exterior", jump=JumpTerm(1.0, "mu")),),
                mu_rhs,
            ),
        ),
    )

    solution = system.solve()

    np.testing.assert_allclose(solution.densities["sigma"].values, sigma_rhs)
    np.testing.assert_allclose(solution.densities["mu"].values, mu_rhs)
    assert np.linalg.norm(solution.residual) < 1.0e-12
