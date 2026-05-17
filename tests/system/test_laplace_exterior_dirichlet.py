import numpy as np

from chunkie import LaplaceExteriorDirichletSystem
from chunkie.geometry import circle
from chunkie.system import SystemConfig


def test_laplace_exterior_dirichlet_solves_circle_cosine_mode():
    boundary = circle(quadrature_order=16, panel_count=24)
    boundary_data = boundary.positions[0]
    system = LaplaceExteriorDirichletSystem(boundary, boundary_data)

    solution = system.solve()
    target = np.array([[2.0], [0.0]])
    field = solution.evaluate(target)

    assert np.linalg.norm(solution.residual) < 1.0e-10
    np.testing.assert_allclose(field.values[0, 0], 0.5, atol=2.0e-3)


def test_laplace_exterior_dirichlet_fmm_evaluation_matches_dense():
    boundary = circle(quadrature_order=12, panel_count=16)
    system = LaplaceExteriorDirichletSystem(boundary, boundary.positions[0])
    solution = system.solve()
    targets = np.array([[2.0, -1.8, 0.3], [0.0, 0.2, 1.7]])

    dense = solution.evaluate(targets, config=SystemConfig(evaluation_method="dense"))
    fmm = solution.evaluate(targets, config=SystemConfig(evaluation_method="fmm"))

    assert fmm.diagnostics["evaluation"] == "fmm"
    np.testing.assert_allclose(fmm.values, dense.values, rtol=2.0e-10, atol=2.0e-11)
