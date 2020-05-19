import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from distimate.stats import make_cdf, make_pdf, make_quantile


def assert_func_values(f, x, y):
    for xv, yv in zip(x, y):
        actual = f(xv)
        assert actual == pytest.approx(
            yv, nan_ok=True
        ), f"Function value at {xv!r} should be {yv!r}, not {actual!r}."
    assert_allclose(f(x), y)


class TestPDF:

    edges = [1, 10, 100, 1000]

    def test_pdf_of_empty(self):
        pdf = make_pdf([1, 10, 100], [0, 0, 0, 0])
        assert_array_equal(pdf.x, [1, 10, 10, 100])
        assert_array_equal(pdf.y, [np.nan, np.nan, np.nan, np.nan])
        assert_func_values(
            pdf,
            [0.9, 1, 2, 10, 20, 100, 101],
            [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        )

    def test_pdf_of_first_bin(self):
        pdf = make_pdf([1, 10, 100], [7, 0, 0, 0])
        assert_array_equal(pdf.x, [1, 10, 10, 100])
        assert_array_equal(pdf.y, [0, 0, 0, 0])
        assert_func_values(
            pdf, [0.9, 1, 2, 10, 20, 100, 101], [0, np.nan, 0, 0, 0, 0, 0],
        )

    def test_pdf_of_last_bin(self):
        pdf = make_pdf([1, 10, 100], [0, 0, 0, 7])
        assert_array_equal(pdf.x, [1, 10, 10, 100])
        assert_array_equal(pdf.y, [0, 0, 0, 0])
        assert_func_values(
            pdf, [0.9, 1, 2, 10, 20, 100, 101], [0, 0, 0, 0, 0, 0, np.nan],
        )

    def test_pdf_of_inner_bin(self):
        pdf = make_pdf([1, 10, 100, 1000], [0, 0, 7, 0, 0])
        assert_array_equal(pdf.x, [1, 10, 10, 100, 100, 1000])
        assert_array_equal(pdf.y, [0, 0, 1 / 90, 1 / 90, 0, 0])
        assert_func_values(
            pdf,
            [0.9, 1, 2, 10, 20, 100, 200, 1000, 1001],
            [0, 0, 0, 0, 1 / 90, 1 / 90, 0, 0, 0],
        )

    def test_pdf_of_first_inner_bin(self):
        pdf = make_pdf([1, 10, 100], [0, 7, 0, 0])
        assert_array_equal(pdf.x, [1, 10, 10, 100])
        assert_allclose(pdf.y, [1 / 9, 1 / 9, 0, 0])
        assert_func_values(
            pdf, [0.9, 1, 2, 10, 20, 100, 101], [0, 0, 1 / 9, 1 / 9, 0, 0, 0],
        )

    def test_pdf_of_last_inner_bin(self):
        pdf = make_pdf([1, 10, 100], [0, 0, 7, 0])
        assert_array_equal(pdf.x, [1, 10, 10, 100])
        assert_allclose(pdf.y, [0, 0, 1 / 90, 1 / 90])
        assert_func_values(
            pdf, [0.9, 1, 2, 10, 20, 100, 101], [0, 0, 0, 0, 1 / 90, 1 / 90, 0],
        )

    def test_pdf_of_first_and_next_bin(self):
        pdf = make_pdf([1, 10, 100], [3, 1, 0, 0])
        assert_array_equal(pdf.x, [1, 10, 10, 100])
        assert_array_equal(pdf.y, [1 / 4 / 9, 1 / 4 / 9, 0, 0])
        assert_func_values(
            pdf,
            [0.9, 1, 2, 10, 20, 100, 101],
            [0, np.nan, 1 / 4 / 9, 1 / 4 / 9, 0, 0, 0],
        )

    def test_pdf_of_first_and_inner_bin(self):
        pdf = make_pdf([1, 10, 100], [3, 0, 1, 0])
        assert_array_equal(pdf.x, [1, 10, 10, 100])
        assert_array_equal(pdf.y, [0, 0, 1 / 4 / 90, 1 / 4 / 90])
        assert_func_values(
            pdf,
            [0.9, 1, 2, 10, 20, 100, 101],
            [0, np.nan, 0, 0, 1 / 4 / 90, 1 / 4 / 90, 0],
        )

    def test_pdf_of_first_and_last_bin(self):
        pdf = make_pdf([1, 10, 100], [3, 0, 0, 1])
        assert_array_equal(pdf.x, [1, 10, 10, 100])
        assert_array_equal(pdf.y, [0, 0, 0, 0])
        assert_func_values(
            pdf, [0.9, 1, 2, 10, 20, 100, 101], [0, np.nan, 0, 0, 0, 0, np.nan],
        )

    def test_pdf_of_last_and_prev_bin(self):
        pdf = make_pdf([1, 10, 100], [0, 0, 3, 1])
        assert_array_equal(pdf.x, [1, 10, 10, 100])
        assert_array_equal(pdf.y, [0, 0, 3 / 4 / 90, 3 / 4 / 90])
        assert_func_values(
            pdf,
            [0.9, 1, 2, 10, 20, 100, 101],
            [0, 0, 0, 0, 3 / 4 / 90, 3 / 4 / 90, np.nan],
        )

    def test_pdf_of_last_and_inner_bin(self):
        pdf = make_pdf([1, 10, 100], [0, 3, 0, 1])
        assert_array_equal(pdf.x, [1, 10, 10, 100])
        assert_array_equal(pdf.y, [3 / 4 / 9, 3 / 4 / 9, 0, 0])
        assert_func_values(
            pdf,
            [0.9, 1, 2, 10, 20, 100, 101],
            [0, 0, 3 / 4 / 9, 3 / 4 / 9, 0, 0, np.nan],
        )

    def test_pdf_of_inner_bins(self):
        pdf = make_pdf([1, 10, 100], [0, 3, 1, 0])
        assert_array_equal(pdf.x, [1, 10, 10, 100])
        assert_array_equal(pdf.y, [3 / 4 / 9, 3 / 4 / 9, 1 / 4 / 90, 1 / 4 / 90])
        assert_func_values(
            pdf,
            [0.9, 1, 2, 10, 20, 100, 101],
            [0, 0, 3 / 4 / 9, 3 / 4 / 9, 1 / 4 / 90, 1 / 4 / 90, 0],
        )

    def test_pdf_of_inner_bins_with_gap(self):
        pdf = make_pdf([1, 10, 100, 1000], [0, 3, 0, 1, 0])
        assert_array_equal(pdf.x, [1, 10, 10, 100, 100, 1000])
        assert_array_equal(
            pdf.y, [3 / 4 / 9, 3 / 4 / 9, 0, 0, 1 / 4 / 900, 1 / 4 / 900]
        )
        assert_func_values(
            pdf,
            [0.9, 1, 2, 10, 20, 100, 200, 1000, 1001],
            [0, 0, 3 / 4 / 9, 3 / 4 / 9, 0, 0, 1 / 4 / 900, 1 / 4 / 900, 0],
        )


class TestCDF:
    def test_cdf_of_empty(self):
        cdf = make_cdf([1, 10, 100], [0, 0, 0, 0])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [np.nan, np.nan, np.nan])
        assert_func_values(
            cdf,
            [0.9, 1, 5.5, 10, 55, 100, 101],
            [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        )

    def test_cdf_of_first_bin(self):
        cdf = make_cdf([1, 10, 100], [7, 0, 0, 0])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [1, 1, 1])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 1, 1, 1, 1, 1, 1],
        )

    def test_cdf_of_last_bin(self):
        cdf = make_cdf([1, 10, 100], [0, 0, 0, 7])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0, 0, 0])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 0, 0, 0, 0, 0, np.nan],
        )

    def test_cdf_of_inner_bin(self):
        cdf = make_cdf([1, 10, 100], [0, 7, 0, 0])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0, 1, 1])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 0, 1 / 2, 1, 1, 1, 1],
        )

    def test_cdf_of_first_and_next_bin(self):
        cdf = make_cdf([1, 10, 100], [3, 1, 0, 0])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0.75, 1, 1])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 3 / 4, 7 / 8, 1, 1, 1, 1],
        )

    def test_cdf_of_first_and_inner_bin(self):
        cdf = make_cdf([1, 10, 100], [3, 0, 1, 0])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0.75, 0.75, 1])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 3 / 4, 3 / 4, 3 / 4, 7 / 8, 1, 1],
        )

    def test_cdf_of_first_and_last_bin(self):
        cdf = make_cdf([1, 10, 100], [3, 0, 0, 1])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0.75, 0.75, 0.75])
        assert_func_values(
            cdf,
            [0.9, 1, 5.5, 10, 55, 100, 101],
            [0, 3 / 4, 3 / 4, 3 / 4, 3 / 4, 3 / 4, np.nan],
        )

    def test_cdf_of_last_and_inner_bin(self):
        cdf = make_cdf([1, 10, 100], [0, 3, 0, 1])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0, 0.75, 0.75])
        assert_func_values(
            cdf,
            [0.9, 1, 5.5, 10, 55, 100, 101],
            [0, 0, 3 / 8, 3 / 4, 3 / 4, 3 / 4, np.nan],
        )

    def test_cdf_of_last_and_prev_bin(self):
        cdf = make_cdf([1, 10, 100], [0, 0, 3, 1])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0, 0, 0.75])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 0, 0, 0, 3 / 8, 3 / 4, np.nan],
        )

    def test_cdf_of_inner_bins(self):
        cdf = make_cdf([1, 10, 100], [0, 3, 1, 0])
        assert_array_equal(cdf.x, [1, 10, 100])
        assert_array_equal(cdf.y, [0, 0.75, 1])
        assert_func_values(
            cdf, [0.9, 1, 5.5, 10, 55, 100, 101], [0, 0, 3 / 8, 3 / 4, 7 / 8, 1, 1],
        )

    def test_cdf_of_inner_bins_with_gap(self):
        cdf = make_cdf([1, 10, 100, 1000], [0, 3, 0, 1, 0])
        assert_array_equal(cdf.x, [1, 10, 100, 1000])
        assert_array_equal(cdf.y, [0, 0.75, 0.75, 1])
        assert_func_values(
            cdf,
            [0.9, 1, 5.5, 10, 55, 100, 550, 1000, 1001],
            [0, 0, 3 / 8, 3 / 4, 3 / 4, 3 / 4, 7 / 8, 1, 1],
        )


class TestQuantileFunction:
    def test_quantile_of_empty(self):
        quantile = make_quantile([1, 10, 100], [0, 0, 0, 0, 0])
        assert_array_equal(quantile.x, [0, 1])
        assert_array_equal(quantile.y, [np.nan, np.nan])
        assert_func_values(
            quantile, [-1, 0, 1 / 2, 1, 2], [np.nan, np.nan, np.nan, np.nan, np.nan],
        )

    def test_quantile_of_first_bin(self):
        quantile = make_quantile([1, 10, 100], [7, 0, 0, 0])
        assert_array_equal(quantile.x, [0, 1])
        assert_array_equal(quantile.y, [1, 1])
        assert_func_values(
            quantile, [-1, 0, 1 / 2, 1, 2], [np.nan, 1, 1, 1, np.nan],
        )

    def test_quantile_of_last_bin(self):
        quantile = make_quantile([1, 10, 100], [0, 0, 0, 7])
        assert_array_equal(quantile.x, [0, 1])
        assert_array_equal(quantile.y, [np.nan, np.nan])
        assert_func_values(
            quantile, [-1, 0, 1 / 2, 1, 2], [np.nan, np.nan, np.nan, np.nan, np.nan],
        )

    def test_quantile_of_inner_bin(self):
        quantile = make_quantile([1, 10, 100, 1000], [0, 0, 7, 0, 0])
        assert_array_equal(quantile.x, [0, 1])
        assert_array_equal(quantile.y, [10, 100])
        assert_func_values(
            quantile, [-1, 0, 1 / 2, 1, 2], [np.nan, 10, 55, 100, np.nan],
        )

    def test_quantile_of_first_inner_bin(self):
        quantile = make_quantile([1, 10, 100], [0, 7, 0, 0])
        assert_array_equal(quantile.x, [0, 1])
        assert_array_equal(quantile.y, [1, 10])
        assert_func_values(
            quantile, [-1, 0, 1 / 2, 1, 2], [np.nan, 1, 5.5, 10, np.nan],
        )

    def test_quantile_of_last_inner_bin(self):
        quantile = make_quantile([1, 10, 100], [0, 0, 7, 0])
        assert_array_equal(quantile.x, [0, 1])
        assert_array_equal(quantile.y, [10, 100])
        assert_func_values(
            quantile, [-1, 0, 1 / 2, 1, 2], [np.nan, 10, 55, 100, np.nan],
        )

    def test_quantile_of_first_and_next_bin(self):
        quantile = make_quantile([1, 10, 100], [3, 1, 0, 0])
        assert_array_equal(quantile.x, [0, 0.75, 1])
        assert_array_equal(quantile.y, [1, 1, 10])
        assert_func_values(
            quantile, [0, 3 / 8, 3 / 4, 7 / 8, 1], [1, 1, 1, 5.5, 10],
        )

    def test_quantile_of_first_and_inner_bin(self):
        quantile = make_quantile([1, 10, 100], [3, 0, 1, 0])
        assert_array_equal(quantile.x, [0, 0.75, 0.75, 1])
        assert_array_equal(quantile.y, [1, 1, 10, 100])
        assert_func_values(
            quantile, [0, 3 / 8, 3 / 4, 7 / 8, 1], [1, 1, 1, 55, 100],
        )

    def test_quantile_of_first_and_last_bin(self):
        quantile = make_quantile([1, 10, 100], [3, 0, 0, 1])
        assert_array_equal(quantile.x, [0, 0.75, 0.75, 1])
        assert_array_equal(quantile.y, [1, 1, 100, np.nan])
        assert_func_values(
            quantile, [0, 3 / 8, 3 / 4, 7 / 8, 1], [1, 1, 1, np.nan, np.nan],
        )

    def test_quantile_of_last_and_inner_bin(self):
        quantile = make_quantile([1, 10, 100], [0, 3, 0, 1])
        assert_array_equal(quantile.x, [0, 0.75, 0.75, 1])
        assert_array_equal(quantile.y, [1, 10, 100, np.nan])
        assert_func_values(
            quantile, [0, 3 / 8, 3 / 4, 7 / 8, 1], [1, 5.5, 10, np.nan, np.nan],
        )

    def test_quantile_of_last_and_prev_bin(self):
        quantile = make_quantile([1, 10, 100], [0, 0, 3, 1])
        assert_array_equal(quantile.x, [0, 0.75, 1])
        assert_array_equal(quantile.y, [10, 100, np.nan])
        assert_func_values(
            quantile, [0, 3 / 8, 3 / 4, 7 / 8, 1], [10, 55, 100, np.nan, np.nan],
        )

    def test_quantile_of_inner_bins(self):
        quantile = make_quantile([1, 10, 100], [0, 3, 1, 0])
        assert_array_equal(quantile.x, [0, 0.75, 1])
        assert_array_equal(quantile.y, [1, 10, 100])
        assert_func_values(
            quantile, [0, 3 / 8, 3 / 4, 7 / 8, 1], [1, 5.5, 10, 55, 100],
        )

    def test_quantile_of_inner_bins_with_gap(self):
        quantile = make_quantile([1, 10, 100, 1000], [0, 3, 0, 1, 0])
        assert_array_equal(quantile.x, [0, 0.75, 0.75, 1])
        assert_array_equal(quantile.y, [1, 10, 100, 1000])
        assert_func_values(
            quantile, [0, 3 / 8, 3 / 4, 7 / 8, 1], [1, 5.5, 10, 550, 1000],
        )
