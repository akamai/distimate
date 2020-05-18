import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from distimate.stats import make_cdf

EDGES = [1, 10, 100]


def assert_func_values(f, x, y):
    for xv, yv in zip(x, y):
        actual = f(xv)
        assert actual == pytest.approx(
            yv, nan_ok=True
        ), f"Function value at {xv!r} should be {yv!r}, not {actual!r}."
    assert_allclose(f(x), y)


class TestCDF:
    def test_cdf_of_empty(self):
        cdf = make_cdf(EDGES, [0, 0, 0, 0])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [np.nan, np.nan, np.nan])
        assert_func_values(
            cdf,
            [0.9, 1, 5.5, 10, 55, 100, 101],
            [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        )

    def test_cdf_of_first_bin(self):
        cdf = make_cdf(EDGES, [7, 0, 0, 0])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [1, 1, 1])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 1, 1, 1, 1, 1, 1],
        )

    def test_cdf_of_last_bin(self):
        cdf = make_cdf(EDGES, [0, 0, 0, 7])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0, 0, 0])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 0, 0, 0, 0, 0, np.nan],
        )

    def test_cdf_of_inner_bin(self):
        cdf = make_cdf(EDGES, [0, 7, 0, 0])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0, 1, 1])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 0, 1 / 2, 1, 1, 1, 1],
        )

    def test_cdf_of_first_and_next_bin(self):
        cdf = make_cdf(EDGES, [3, 1, 0, 0])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0.75, 1, 1])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 3 / 4, 7 / 8, 1, 1, 1, 1],
        )

    def test_cdf_of_first_and_inner_bin(self):
        cdf = make_cdf(EDGES, [3, 0, 1, 0])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0.75, 0.75, 1])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 3 / 4, 3 / 4, 3 / 4, 7 / 8, 1, 1],
        )

    def test_cdf_of_first_and_last_bin(self):
        cdf = make_cdf(EDGES, [3, 0, 0, 1])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0.75, 0.75, 0.75])
        assert_func_values(
            cdf,
            [0.9, 1, 5.5, 10, 55, 100, 101],
            [0, 3 / 4, 3 / 4, 3 / 4, 3 / 4, 3 / 4, np.nan],
        )

    def test_cdf_of_last_and_inner_bin(self):
        cdf = make_cdf(EDGES, [0, 3, 0, 1])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0, 0.75, 0.75])
        assert_func_values(
            cdf,
            [0.9, 1, 5.5, 10, 55, 100, 101],
            [0, 0, 3 / 8, 3 / 4, 3 / 4, 3 / 4, np.nan],
        )

    def test_cdf_of_last_and_prev_bin(self):
        cdf = make_cdf(EDGES, [0, 0, 3, 1])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0, 0, 0.75])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 0, 0, 0, 3 / 8, 3 / 4, np.nan],
        )

    def test_cdf_of_inner_bins(self):
        cdf = make_cdf(EDGES, [0, 3, 1, 0])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0, 0.75, 1])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101],
            [0, 0, 3/8, 3/4, 7/8, 1, 1],
        )
