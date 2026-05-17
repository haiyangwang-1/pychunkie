from types import SimpleNamespace

import numpy as np

from chunkie.kernels import kernel


def test_laplace_selectors_match_closed_form_components():
    source = SimpleNamespace(
        positions=np.array([[0.0, 1.0], [0.0, -0.25]]),
        normals=np.array([[1.0, -0.6], [0.0, 0.8]]),
    )
    target = SimpleNamespace(
        positions=np.array([[0.3, -0.75, 1.4], [1.1, 0.2, -0.4]]),
        normals=np.array([[0.0, 0.8, -0.3], [1.0, 0.6, 0.95]]),
    )
    dx = target.positions[0, :, None] - source.positions[0, None, :]
    dy = target.positions[1, :, None] - source.positions[1, None, :]
    r2 = dx**2 + dy**2
    r4 = r2**2
    gradient = np.stack((-dx / (2.0 * np.pi * r2), -dy / (2.0 * np.pi * r2)), axis=0)
    hessian = np.empty((2, 2, target.positions.shape[1], source.positions.shape[1]))
    hessian[0, 0] = (2.0 * dx**2 - r2) / (2.0 * np.pi * r4)
    hessian[0, 1] = 2.0 * dx * dy / (2.0 * np.pi * r4)
    hessian[1, 0] = hessian[0, 1]
    hessian[1, 1] = (2.0 * dy**2 - r2) / (2.0 * np.pi * r4)

    expected_single = -np.log(r2) / (4.0 * np.pi)
    expected_source_normal = -np.einsum("rts,rs->ts", gradient, source.normals)
    expected_target_normal = np.einsum("rts,rt->ts", gradient, target.normals)
    expected_source_gradient = -np.einsum("abts,bs->ats", hessian, source.normals)
    expected_normal_derivative = -np.einsum("abts,bs,at->ts", hessian, source.normals, target.normals)

    np.testing.assert_allclose(kernel("laplace", selector="s")(source, target)[0, 0], expected_single)
    np.testing.assert_allclose(kernel("laplace", selector="sg")(source, target)[:, 0], gradient)
    np.testing.assert_allclose(kernel("laplace", selector="d")(source, target)[0, 0], expected_source_normal)
    np.testing.assert_allclose(kernel("laplace", selector="sp")(source, target)[0, 0], expected_target_normal)
    np.testing.assert_allclose(kernel("laplace", selector="dg")(source, target)[:, 0], expected_source_gradient)
    np.testing.assert_allclose(kernel("laplace", selector="dp")(source, target)[0, 0], expected_normal_derivative)


def test_helmholtz_single_gradient_matches_centered_difference():
    source = np.array([[0.3], [-0.2]])
    target = np.array([[1.1], [0.7]])
    wavenumber = 1.2 + 0.4j
    eps = 1.0e-6
    helmholtz_single = kernel("helmholtz", selector="s", wavenumber=wavenumber)
    helmholtz_gradient = kernel("helmholtz", selector="sg", wavenumber=wavenumber)

    value_xp = helmholtz_single(source, target + np.array([[eps], [0.0]]))[0, 0]
    value_xm = helmholtz_single(source, target - np.array([[eps], [0.0]]))[0, 0]
    value_yp = helmholtz_single(source, target + np.array([[0.0], [eps]]))[0, 0]
    value_ym = helmholtz_single(source, target - np.array([[0.0], [eps]]))[0, 0]
    expected = np.array(
        [
            ((value_xp - value_xm) / (2.0 * eps)).item(),
            ((value_yp - value_ym) / (2.0 * eps)).item(),
        ],
    )

    actual = helmholtz_gradient(source, target)[:, 0, 0, 0]

    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-7)


def test_biharmonic_value_gradient_and_hessian_match_closed_forms():
    source = np.array([[0.0, 0.7], [-0.3, 0.4]])
    target = np.array([[1.2, -0.4], [0.9, 1.1]])
    displacement = target[:, :, None] - source[:, None, :]
    dx, dy = displacement
    rho2 = dx**2 + dy**2
    log_rho = 0.5 * np.log(rho2)
    expected_value = rho2 * log_rho / (8.0 * np.pi)
    expected_gradient = displacement * (2.0 * log_rho[None, :, :] + 1.0) / (8.0 * np.pi)
    expected_hessian = np.empty((2, 2, *rho2.shape))
    expected_hessian[0, 0] = dx**2 / (4.0 * np.pi * rho2) + (2.0 * log_rho + 1.0) / (8.0 * np.pi)
    expected_hessian[0, 1] = dx * dy / (4.0 * np.pi * rho2)
    expected_hessian[1, 0] = expected_hessian[0, 1]
    expected_hessian[1, 1] = dy**2 / (4.0 * np.pi * rho2) + (2.0 * log_rho + 1.0) / (8.0 * np.pi)

    np.testing.assert_allclose(kernel("biharmonic", selector="s")(source, target)[0, 0], expected_value)
    np.testing.assert_allclose(kernel("biharmonic", selector="sg")(source, target)[:, 0], expected_gradient)
    np.testing.assert_allclose(
        kernel("biharmonic", selector="hessian")(source, target)[:, 0],
        expected_hessian.reshape(4, *rho2.shape),
    )


def test_stokes_single_velocity_matches_closed_form_tensor():
    source = np.array([[-0.2, 0.9], [0.5, -0.1]])
    target = np.array([[0.8, -0.6, 1.4], [1.2, 0.2, -0.7]])
    viscosity = 1.7
    dx = target[0, :, None] - source[0, None, :]
    dy = target[1, :, None] - source[1, None, :]
    rho2 = dx**2 + dy**2
    log_inverse_rho = -0.5 * np.log(rho2)
    expected = np.empty((2, 2, target.shape[1], source.shape[1]))
    expected[0, 0] = (dx**2 / rho2 + log_inverse_rho) / (4.0 * np.pi * viscosity)
    expected[1, 1] = (dy**2 / rho2 + log_inverse_rho) / (4.0 * np.pi * viscosity)
    expected[0, 1] = dx * dy / rho2 / (4.0 * np.pi * viscosity)
    expected[1, 0] = expected[0, 1]

    actual = kernel("stokes", selector="s", viscosity=viscosity)(source, target)

    np.testing.assert_allclose(actual, expected)


def test_elasticity_single_displacement_matches_closed_form_tensor():
    source = np.array([[0.1, -0.5], [0.3, -0.2]])
    target = np.array([[1.1, -0.7, 0.4], [0.8, 0.9, -0.6]])
    lame_lambda = 1.5
    lame_mu = 2.1
    beta = (lame_lambda + 3.0 * lame_mu) / (
        4.0 * np.pi * lame_mu * (lame_lambda + 2.0 * lame_mu)
    )
    gamma = -(lame_lambda + lame_mu) / (4.0 * np.pi * lame_mu * (lame_lambda + 2.0 * lame_mu))
    dx = target[0, :, None] - source[0, None, :]
    dy = target[1, :, None] - source[1, None, :]
    rho2 = dx**2 + dy**2
    log_rho = 0.5 * np.log(rho2)
    expected = np.empty((2, 2, target.shape[1], source.shape[1]))
    expected[0, 0] = beta * log_rho + gamma / 2.0 + gamma * dx**2 / rho2
    expected[1, 1] = beta * log_rho + gamma / 2.0 + gamma * dy**2 / rho2
    expected[0, 1] = gamma * dx * dy / rho2
    expected[1, 0] = expected[0, 1]

    actual = kernel(
        "elasticity",
        selector="s",
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
    )(source, target)

    np.testing.assert_allclose(actual, expected)
