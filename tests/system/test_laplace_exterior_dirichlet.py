import numpy as np

from chunkie import LaplaceExteriorDirichletSystem
from chunkie.geometry import circle


def test_laplace_exterior_dirichlet_solves_circle_cosine_mode():
    boundary = circle(quadrature_order=16, panel_count=24)
    boundary_data = boundary.positions[0]
    system = LaplaceExteriorDirichletSystem(boundary, boundary_data)

    solution = system.solve()
    target = np.array([[2.0], [0.0]])
    field = solution.evaluate(target)

    assert np.linalg.norm(solution.residual) < 1.0e-10
    np.testing.assert_allclose(field.values[0, 0], 0.5, atol=2.0e-3)
