import numpy as np

from chunkie.geometry import circle
from chunkie.kernels import kernel
from chunkie.system import (
    BoundaryEquation,
    BoundaryTrace,
    DensitySpace,
    IntegralSystem,
    JumpTerm,
    LayerPotential,
)


def _laplace_neumann_circle_system(side: str):
    boundary = circle(quadrature_order=16, panel_count=24)
    density = DensitySpace("sigma", boundary)
    trace_layer = LayerPotential(
        "normal_single", boundary, kernel("laplace", selector="sp"), "sigma"
    )
    field_layer = LayerPotential("single", boundary, kernel("laplace", selector="s"), "sigma")
    jump = 0.5 if side == "interior" else -0.5
    rhs = boundary.positions[0] if side == "interior" else -boundary.positions[0]
    trace = BoundaryTrace(trace_layer, boundary, side, jump=JumpTerm(jump, "sigma"))
    system = IntegralSystem(
        f"laplace_{side}_neumann_circle",
        boundary,
        (density,),
        (BoundaryEquation("neumann", boundary, (trace,), rhs),),
        fields={"u": (field_layer,)},
    )
    return boundary, system


def test_laplace_single_layer_normal_derivative_self_diagonal_is_finite():
    boundary, system = _laplace_neumann_circle_system("interior")

    matrix = system.assemble().to_dense()
    diagonal_without_jump = np.diag(matrix) - 0.5
    expected = (-boundary.signed_curvature / (4.0 * np.pi)).T.reshape(-1)
    expected *= boundary.pointinfo.flat_weights

    assert np.all(np.isfinite(matrix))
    np.testing.assert_allclose(diagonal_without_jump, expected, atol=1.0e-14)


def test_laplace_interior_neumann_circle_matches_linear_solution_up_to_constant():
    _, system = _laplace_neumann_circle_system("interior")
    solution = system.solve()
    targets = np.array([[0.0, 0.3, -0.2], [0.0, 0.2, 0.4]])

    values = solution.evaluate(targets).values[0]
    truth = targets[0]
    values = values + np.mean(truth - values)

    assert np.linalg.norm(solution.residual) < 1.0e-10
    np.testing.assert_allclose(values, truth, atol=2.0e-3)


def test_laplace_exterior_neumann_circle_matches_dipole_solution_up_to_constant():
    _, system = _laplace_neumann_circle_system("exterior")
    solution = system.solve()
    targets = np.array([[2.0, 1.4, -1.6], [0.0, 0.8, 0.3]])

    values = solution.evaluate(targets).values[0]
    truth = targets[0] / np.sum(targets * targets, axis=0)
    values = values + np.mean(truth - values)

    assert np.linalg.norm(solution.residual) < 1.0e-10
    np.testing.assert_allclose(values, truth, atol=2.0e-3)
