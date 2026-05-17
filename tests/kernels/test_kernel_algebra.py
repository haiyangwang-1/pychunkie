import numpy as np

from chunkie.kernels import kernel


def test_scaled_kernel_scales_singular_expansion_and_values():
    source = np.array([[0.0], [0.0]])
    target = np.array([[2.0], [0.0]])
    base = kernel("laplace", selector="s")
    scaled = (2.0 - 1.0j) * base

    np.testing.assert_allclose(scaled(source, target), (2.0 - 1.0j) * base(source, target))
    np.testing.assert_allclose(
        scaled.singularity.expansion.evaluate(source, target),
        (2.0 - 1.0j) * base.singularity.expansion.evaluate(source, target),
    )
    assert scaled.singularity.legacy_strength == "log"


def test_kernel_subtraction_cancels_exact_scalar_singularity_metadata():
    source = np.array([[0.0], [0.0]])
    target = np.array([[2.0], [0.0]])
    base = kernel("laplace", selector="s")
    canceled = base - base

    np.testing.assert_allclose(canceled(source, target), 0.0)
    np.testing.assert_allclose(canceled.singularity.expansion.evaluate(source, target), 0.0)
    assert canceled.singularity.legacy_strength == "smooth"
    assert canceled.singularity.boundary_limit == "smooth"
    assert canceled.singularity.remainder_regular == "smooth"


def test_kernel_addition_preserves_strongest_laplace_basis_strength():
    combined = kernel("laplace", selector="s") + kernel("laplace", selector="d")

    assert combined.singularity.legacy_strength == "mixed"
    assert combined.singularity.boundary_limit == "hs"
    assert len(combined.singularity.expansion.terms) == 3


def test_scaled_callable_helmholtz_metadata_evaluates_scaled_expansion():
    source = np.array([[0.0], [0.0]])
    target = np.array([[0.01], [0.004]])
    base = kernel("helmholtz", selector="s", wavenumber=1.4)
    scaled = -3.0 * base

    np.testing.assert_allclose(
        scaled.singularity.expansion.evaluate(source, target),
        -3.0 * base.singularity.expansion.evaluate(source, target),
    )
