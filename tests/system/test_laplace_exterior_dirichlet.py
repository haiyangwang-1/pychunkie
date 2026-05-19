import numpy as np
import pytest

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


def test_acceleration_config_values_are_not_supported():
    with pytest.raises(ValueError, match="solve_method"):
        SystemConfig(solve_method="flam")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="evaluation_method"):
        SystemConfig(evaluation_method="fmm")  # type: ignore[arg-type]


def test_laplace_exterior_dirichlet_gmres_solve_matches_dense():
    boundary = circle(quadrature_order=8, panel_count=12)
    system = LaplaceExteriorDirichletSystem(boundary, boundary.positions[0])
    targets = np.array([[2.0, -1.8], [0.0, 0.3]])

    dense = system.solve()
    gmres = system.solve(
        config=SystemConfig(solve_method="gmres", tolerance=1.0e-12, max_iterations=80)
    )

    assert gmres.diagnostics["backend"] == "scipy.sparse.linalg.gmres"
    assert gmres.diagnostics["gmres_info"] == 0
    assert np.linalg.norm(gmres.residual) < 1.0e-9
    np.testing.assert_allclose(
        gmres.evaluate(targets).values, dense.evaluate(targets).values, atol=1.0e-10
    )
