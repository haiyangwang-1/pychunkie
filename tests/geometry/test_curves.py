import numpy as np

from chunkie.geometry import curves


def test_line_parabola_and_sine_graph_curve_helpers_preserve_input_shape():
    parameters = np.array([[0.0, 0.25], [0.5, 1.0]])

    line_positions, line_derivatives, line_second = curves.line_segment(
        parameters, [1.0, -1.0], [3.0, 2.0]
    )
    np.testing.assert_allclose(line_positions[0], 1.0 + 2.0 * parameters)
    np.testing.assert_allclose(line_positions[1], -1.0 + 3.0 * parameters)
    np.testing.assert_allclose(line_derivatives[0], 2.0)
    np.testing.assert_allclose(line_derivatives[1], 3.0)
    np.testing.assert_allclose(line_second, 0.0)

    parabola_positions, parabola_derivatives, parabola_second = curves.parabola(
        parameters, 1.5, -0.25
    )
    np.testing.assert_allclose(parabola_positions[0], parameters)
    np.testing.assert_allclose(parabola_positions[1], 1.5 * (parameters + 0.25) ** 2)
    np.testing.assert_allclose(parabola_derivatives[0], 1.0)
    np.testing.assert_allclose(parabola_derivatives[1], 3.0 * (parameters + 0.25))
    np.testing.assert_allclose(parabola_second[0], 0.0)
    np.testing.assert_allclose(parabola_second[1], 3.0)

    sine_positions, sine_derivatives, sine_second = curves.sine_graph(parameters, 2.0, 3.0, 0.1)
    phase = 3.0 * parameters + 0.1
    np.testing.assert_allclose(sine_positions[0], parameters)
    np.testing.assert_allclose(sine_positions[1], 2.0 * np.sin(phase))
    np.testing.assert_allclose(sine_derivatives[0], 1.0)
    np.testing.assert_allclose(sine_derivatives[1], 6.0 * np.cos(phase))
    np.testing.assert_allclose(sine_second[0], 0.0)
    np.testing.assert_allclose(sine_second[1], -18.0 * np.sin(phase))


def test_starfish_derivatives_match_centered_finite_differences():
    theta = np.array([-0.7, 0.2, 1.3])
    eps = 1.0e-6

    _, derivatives, second_derivatives = curves.starfish(
        theta,
        arm_count=4,
        amplitude=0.25,
        center=[0.1, -0.2],
        phase=0.31,
        scale=1.8,
    )
    positions_plus = curves.starfish(theta + eps, 4, 0.25, [0.1, -0.2], 0.31, 1.8)[0]
    positions_minus = curves.starfish(theta - eps, 4, 0.25, [0.1, -0.2], 0.31, 1.8)[0]
    derivatives_plus = curves.starfish(theta + eps, 4, 0.25, [0.1, -0.2], 0.31, 1.8)[1]
    derivatives_minus = curves.starfish(theta - eps, 4, 0.25, [0.1, -0.2], 0.31, 1.8)[1]

    np.testing.assert_allclose(
        (positions_plus - positions_minus) / (2.0 * eps),
        derivatives,
        rtol=1.0e-10,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        (derivatives_plus - derivatives_minus) / (2.0 * eps),
        second_derivatives,
        rtol=2.0e-9,
        atol=1.0e-9,
    )


def test_fourier_radius_curve_matches_finite_difference_derivatives_and_aliases():
    theta = np.array([-0.4, 0.3, 1.1])
    modes = np.array([1.0, 0.2, -0.05, 0.03])
    eps = 1.0e-6

    positions, derivatives, second_derivatives = curves.fourier_radius(
        theta,
        modes,
        center=[0.1, -0.25],
        scale=[1.2, 0.9],
    )
    positions_plus = curves.fourier_radius(theta + eps, modes, [0.1, -0.25], [1.2, 0.9])[0]
    positions_minus = curves.fourier_radius(theta - eps, modes, [0.1, -0.25], [1.2, 0.9])[0]
    derivatives_plus = curves.fourier_radius(theta + eps, modes, [0.1, -0.25], [1.2, 0.9])[1]
    derivatives_minus = curves.fourier_radius(theta - eps, modes, [0.1, -0.25], [1.2, 0.9])[1]

    np.testing.assert_allclose(
        (positions_plus - positions_minus) / (2.0 * eps),
        derivatives,
        rtol=2.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        (derivatives_plus - derivatives_minus) / (2.0 * eps),
        second_derivatives,
        rtol=2.0e-9,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(curves.bymode(theta, modes, [0.1, -0.25], [1.2, 0.9])[0], positions)
