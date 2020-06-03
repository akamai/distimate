import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from distimate.distributions import Distribution

EDGES = [1, 10, 100]


class TestDistribution:
    def test_invalid_shape_0d(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution(EDGES, 3)
        assert str(exc_info.value) == "Histogram must be 1-D array-like."

    def test_invalid_shape_2d(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution(EDGES, [[1, 0, 2, 0]])
        assert str(exc_info.value) == "Histogram must be 1-D array-like."

    def test_invalid_length(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution(EDGES, [])
        assert str(exc_info.value) == "Histogram must have len(edges) + 1 items."

    def test_negative_hist_value(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution(EDGES, [0, 0, -1, 0])
        assert str(exc_info.value) == "Histogram values must not be negative."

    def test_from_samples_list(self):
        dist = Distribution.from_samples(EDGES, [0, 42, 47])
        assert_array_equal(dist.values, [1, 0, 2, 0])

    def test_from_samples_lists_w_weights(self):
        dist = Distribution.from_samples(EDGES, [0, 42, 47], [5, 1, 2])
        assert_array_equal(dist.values, [5, 0, 3, 0])

    def test_from_samples_array(self):
        dist = Distribution.from_samples(EDGES, np.array([0, 42, 47]))
        assert_array_equal(dist.values, [1, 0, 2, 0])

    def test_from_samples_array_w_weights(self):
        dist = Distribution.from_samples(
            EDGES, np.array([0, 42, 47]), np.array([5, 1, 2])
        )
        assert_array_equal(dist.values, [5, 0, 3, 0])

    def test_from_samples_series(self):
        dist = Distribution.from_samples(EDGES, pd.Series([0, 42, 47]))
        assert_array_equal(dist.values, [1, 0, 2, 0])

    def test_from_samples_series_w_weights(self):
        dist = Distribution.from_samples(
            EDGES, pd.Series([0, 42, 47]), pd.Series([5, 1, 2])
        )
        assert_array_equal(dist.values, [5, 0, 3, 0])

    def test_from_samples_frame(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution.from_samples(EDGES, pd.DataFrame())
        assert str(exc_info.value) == "Values must be 1-D array-like."

    def test_from_samples_frame_column(self):
        df = pd.DataFrame({"x": [0, 42, 47]})
        dist = Distribution.from_samples(EDGES, df["x"])
        assert_array_equal(dist.values, [1, 0, 2, 0])

    def test_from_histogram_list(self):
        dist = Distribution.from_histogram(EDGES, [2, 0, 1, 0])
        assert_array_equal(dist.values, [2, 0, 1, 0])

    def test_from_histogram_array(self):
        dist = Distribution.from_histogram(EDGES, np.array([2, 0, 1, 0]))
        assert_array_equal(dist.values, [2, 0, 1, 0])

    def test_from_histogram_invalid_length(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution.from_histogram(EDGES, [2, 0, 1])
        assert str(exc_info.value) == "Histogram must have len(edges) + 1 items."

    def test_from_cumulative_list(self):
        dist = Distribution.from_cumulative(EDGES, [2, 2, 3, 3])
        assert_array_equal(dist.values, [2, 0, 1, 0])

    def test_from_cumulative_array(self):
        dist = Distribution.from_cumulative(EDGES, np.array([2, 2, 3, 3]))
        assert_array_equal(dist.values, [2, 0, 1, 0])

    def test_from_cumulative_invalid_length(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution.from_cumulative(EDGES, [2, 2, 3])
        assert str(exc_info.value) == "Histogram must have len(edges) + 1 items."

    def test_empty(self):
        dist = Distribution(EDGES)
        assert_array_equal(dist.values, [0, 0, 0, 0])
        assert_array_equal(dist.to_histogram(), [0, 0, 0, 0])
        assert_array_equal(dist.to_cumulative(), [0, 0, 0, 0])
        assert dist.values.dtype == np.float64

    def test_full(self):
        dist = Distribution(EDGES, [1, 2, 0, 4])
        assert_array_equal(dist.values, [1, 2, 0, 4])
        assert_array_equal(dist.to_histogram(), [1, 2, 0, 4])
        assert_array_equal(dist.to_cumulative(), [1, 3, 3, 7])
        assert dist.values.dtype == np.float64

    def test_repr_of_empty(self):
        dist = Distribution(EDGES)
        assert repr(dist) == str(dist) == "<Distribution: weight=0, mean=nan>"

    def test_repr_of_full(self):
        dist = Distribution(EDGES, [1, 2, 0, 0])
        assert repr(dist) == str(dist) == "<Distribution: weight=3, mean=4.00>"

    def test_empty_equal(self):
        assert Distribution(EDGES) == Distribution(EDGES)

    def test_empty_not_equal(self):
        assert Distribution(EDGES) != Distribution(EDGES, [1, 2, 0, 4])

    def test_equal(self):
        assert Distribution(EDGES, [1, 2, 0, 4]) == Distribution(EDGES, [1, 2, 0, 4])

    def test_not_equal(self):
        assert Distribution(EDGES, [1, 2, 0, 4]) != Distribution(EDGES, [1, 2, 0, 5])

    def test_equal_different_edges(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution(EDGES, [1, 2, 0, 4]) == Distribution([0, 1, 2], [1, 2, 0, 4])
        assert str(exc_info.value) == "Distributions have different edges."

    def test_equal_different_edges_count(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution(EDGES, [1, 2, 0, 4]) == Distribution([1, 10], [1, 2, 0])
        assert str(exc_info.value) == "Distributions have different edges."

    def test_add_distribution(self):
        dist = Distribution(EDGES, [1, 2, 0, 0]) + Distribution(EDGES, [0, 2, 0, 4])
        assert_array_equal(dist.values, [1, 4, 0, 4])

    def test_add_distribution_in_place(self):
        dist = Distribution(EDGES, [1, 2, 0, 0])
        hist = dist.values
        dist += Distribution(EDGES, [0, 2, 0, 4])
        assert_array_equal(hist, [1, 4, 0, 4])

    def test_add_distribution_different_edges(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution(EDGES, [1, 2, 0, 0]) + Distribution([0, 1, 2], [0, 2, 0, 4])
        assert str(exc_info.value) == "Distributions have different edges."

    def test_add_distribution_different_edges_count(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution(EDGES, [1, 2, 0, 0]) + Distribution([1, 10], [0, 2, 0])
        assert str(exc_info.value) == "Distributions have different edges."

    def test_add_distribution_in_place_different_edges(self):
        dist = Distribution(EDGES, [1, 2, 0, 0])
        with pytest.raises(ValueError) as exc_info:
            dist += Distribution([0, 1, 2], [0, 2, 0, 4])
        assert str(exc_info.value) == "Distributions have different edges."

    def test_add_distribution_in_place_different_edges_count(self):
        dist = Distribution(EDGES, [1, 2, 0, 0])
        with pytest.raises(ValueError) as exc_info:
            dist += Distribution([1, 10], [0, 2, 0])
        assert str(exc_info.value) == "Distributions have different edges."

    def test_weight_of_empty(self):
        dist = Distribution(EDGES)
        assert dist.weight == 0

    def test_weight_of_full(self):
        dist = Distribution(EDGES, [1, 2, 0, 4])
        assert dist.weight == 7

    def test_add_lt_first_edge(self):
        dist = Distribution(EDGES)
        dist.add(0.9)
        assert_array_equal(dist.values, [1, 0, 0, 0])

    def test_add_eq_first_edge(self):
        dist = Distribution(EDGES)
        dist.add(1)
        assert_array_equal(dist.values, [1, 0, 0, 0])

    def test_add_gt_first_edge(self):
        dist = Distribution(EDGES)
        dist.add(1.1)
        assert_array_equal(dist.values, [0, 1, 0, 0])

    def test_add_lt_last_edge(self):
        dist = Distribution(EDGES)
        dist.add(99)
        assert_array_equal(dist.values, [0, 0, 1, 0])

    def test_add_eq_last_edge(self):
        dist = Distribution(EDGES)
        dist.add(100)
        assert_array_equal(dist.values, [0, 0, 1, 0])

    def test_add_gt_last_edge(self):
        dist = Distribution(EDGES)
        dist.add(101)
        assert_array_equal(dist.values, [0, 0, 0, 1])

    def test_add_with_weight(self):
        dist = Distribution(EDGES)
        dist.add(17, 7)
        assert_array_equal(dist.values, [0, 0, 7, 0])

    def test_add_not_scalar(self):
        dist = Distribution(EDGES)
        with pytest.raises(ValueError) as exc_info:
            dist.add([1, 1, 17])
        assert str(exc_info.value) == "Value must be a scalar."

    def test_update(self):
        dist = Distribution(EDGES)
        dist.update([1, 1, 17])
        assert_array_equal(dist.values, [2, 0, 1, 0])

    def test_update_with_weight(self):
        dist = Distribution(EDGES)
        dist.update([1, 1, 17], 10)
        assert_array_equal(dist.values, [20, 0, 10, 0])

    def test_update_with_multiple_weights(self):
        dist = Distribution(EDGES)
        dist.update([1, 1, 17], [3, 2, 1])
        assert_array_equal(dist.values, [5, 0, 1, 0])

    def test_update_not_1d(self):
        dist = Distribution(EDGES)
        with pytest.raises(ValueError) as exc_info:
            dist.update([[1, 1, 17]])
        assert str(exc_info.value) == "Values must be 1-D array-like."

    def test_mean_of_empty(self):
        dist = Distribution(EDGES, [0, 0, 0, 0])
        assert np.isnan(dist.mean)

    def test_mean_of_full(self):
        dist = Distribution(EDGES, [3, 0, 1, 0])
        assert dist.mean == (3 * 1 + 55) / 4

    def test_pdf_of_empty(self):
        dist = Distribution(EDGES, [0, 0, 0, 0])
        pdf = dist.pdf
        assert_array_equal(pdf.x, [1, 100])
        assert_array_equal(pdf.y, [np.nan, np.nan])
        assert_array_equal(pdf([0, 1, 55]), [0, np.nan, np.nan])

    def test_pdf_of_full(self):
        dist = Distribution(EDGES, [3, 0, 1, 0])
        pdf = dist.pdf
        assert_array_equal(pdf.x, [1, 1, 10, 10, 100])
        assert_array_equal(pdf.y, [np.nan, 0, 0, 1 / 4 / 90, 1 / 4 / 90])
        assert_array_equal(pdf([0, 1, 55]), [0, np.nan, 1 / 4 / 90])

    def test_cdf_of_empty(self):
        dist = Distribution(EDGES, [0, 0, 0, 0])
        cdf = dist.cdf
        assert_array_equal(cdf.x, EDGES)
        assert_array_equal(cdf.y, [np.nan, np.nan, np.nan])
        assert_array_equal(cdf([0, 1, 55]), [0, np.nan, np.nan])

    def test_cdf_of_full(self):
        dist = Distribution(EDGES, [3, 0, 1, 0])
        cdf = dist.cdf
        assert_array_equal(cdf.x, EDGES)
        assert_array_equal(cdf.y, [3 / 4, 3 / 4, 1])
        assert_array_equal(cdf([0, 1, 55]), [0, 3 / 4, 7 / 8])

    def test_quantile_of_empty(self):
        dist = Distribution(EDGES, [0, 0, 0, 0])
        quantile = dist.quantile
        assert_array_equal(quantile.x, [0, 1])
        assert_array_equal(quantile.y, [np.nan, np.nan])
        assert_array_equal(
            quantile([-1, 0, 1 / 2, 7 / 8]), [np.nan, np.nan, np.nan, np.nan]
        )

    def test_quantile_of_full(self):
        dist = Distribution(EDGES, [3, 0, 1, 0])
        quantile = dist.quantile
        assert_array_equal(quantile.x, [0, 3 / 4, 3 / 4, 1])
        assert_array_equal(quantile.y, [1, 1, 10, 100])
        assert_array_equal(quantile([-1, 0, 1 / 2, 7 / 8]), [np.nan, 1, 1, 55])
