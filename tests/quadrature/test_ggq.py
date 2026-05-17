import numpy as np
from numpy.polynomial.legendre import leggauss

from chunkie.quadrature import ggq_removable_rules, setup_ggq


def test_generated_ggq_self_rules_split_around_each_legendre_node():
    order = 8
    base_nodes, _ = leggauss(order)
    nodes_by_source, weights_by_source = ggq_removable_rules(order, nfac=2)

    assert len(nodes_by_source) == order
    assert len(weights_by_source) == order
    for source_node, nodes, weights in zip(base_nodes, nodes_by_source, weights_by_source, strict=True):
        assert nodes.shape == weights.shape
        assert np.all(nodes >= -1.0)
        assert np.all(nodes <= 1.0)
        assert not np.any(np.isclose(nodes, source_node))
        np.testing.assert_allclose(np.sum(weights), 2.0, atol=1.0e-14)
        np.testing.assert_allclose(np.sum(weights * nodes), 0.0, atol=1.0e-14)


def test_ggq_setup_builds_interpolators_for_neighbor_and_self_rules():
    order = 6
    rules = setup_ggq(order, nfac_self=2, nfac_near=3)
    base_nodes, _ = leggauss(order)
    polynomial_values = 1.0 + base_nodes - 2.0 * base_nodes**3

    assert rules.neighbor_interpolator.shape == (18, order)
    assert len(rules.self_interpolators) == order
    np.testing.assert_allclose(
        rules.neighbor_interpolator @ polynomial_values,
        1.0 + rules.neighbor_nodes - 2.0 * rules.neighbor_nodes**3,
        atol=1.0e-13,
    )
    for nodes, interpolator in zip(rules.self_nodes, rules.self_interpolators, strict=True):
        np.testing.assert_allclose(
            interpolator @ polynomial_values,
            1.0 + nodes - 2.0 * nodes**3,
            atol=1.0e-13,
        )
