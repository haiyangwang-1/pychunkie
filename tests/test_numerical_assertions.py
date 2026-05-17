from __future__ import annotations

import math

import numpy as np
import pytest

from _numerical import assert_abs_or_rel_close


def test_abs_or_rel_helper_accepts_absolute_error_near_zero():
    actual = np.array([1.0e-14, 1.0 + 1.0e-8])
    expected = np.array([0.0, 1.0])

    metrics = assert_abs_or_rel_close(
        actual,
        expected,
        abs_tol=1.0e-13,
        rel_tol=1.0e-7,
        label="near_zero",
    )

    assert metrics.max_abs_error == pytest.approx(1.0e-8)
    assert math.isinf(metrics.max_rel_error)
    assert metrics.max_scaled_error == pytest.approx(0.1)


def test_abs_or_rel_helper_accepts_relative_error_for_large_values():
    assert_abs_or_rel_close(
        np.array([1000.0 + 1.0e-5]),
        np.array([1000.0]),
        abs_tol=1.0e-8,
        rel_tol=1.0e-7,
    )


def test_abs_or_rel_helper_rejects_points_failing_both_thresholds():
    with pytest.raises(AssertionError, match=r"first failing index=\(1,\)"):
        assert_abs_or_rel_close(
            np.array([0.0, 1.0e-4]),
            np.array([0.0, 0.0]),
            abs_tol=1.0e-5,
            rel_tol=1.0e-3,
            label="zero_reference",
        )


def test_abs_or_rel_helper_can_treat_matching_nans_as_exact():
    assert_abs_or_rel_close(
        np.array([np.nan, 1.0]),
        np.array([np.nan, 1.0]),
        abs_tol=0.0,
        rel_tol=0.0,
        equal_nan=True,
    )
