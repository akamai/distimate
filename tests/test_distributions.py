import numpy as np
import pytest
from numpy.testing import assert_array_equal

from distimate import Distribution

EDGES = [1, 10, 100]


class TestDistribution:
    def test_invalid_shape(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution(EDGES, [[0, 0, 0, 0]])
        assert str(exc_info.value) == "Histogram must be 1-D array-like."

    def test_invalid_length(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution(EDGES, [])
        assert str(exc_info.value) == "Histogram must have len(edges) + 1 items."

    def test_negative_hist_value(self):
        with pytest.raises(ValueError) as exc_info:
            Distribution(EDGES, [0, 0, -1, 0])
        assert str(exc_info.value) == "Histogram values must not be negative."

    def test_empty(self):
        dist = Distribution(EDGES)
        assert_array_equal(dist.hist, [0, 0, 0, 0])
        assert_array_equal(dist.to_hist(), [0, 0, 0, 0])
        assert_array_equal(dist.to_cumulative(), [0, 0, 0, 0])
        assert dist.hist.dtype == np.float64

    def test_full(self):
        dist = Distribution(EDGES, [1, 2, 0, 4])
        assert_array_equal(dist.hist, [1, 2, 0, 4])
        assert_array_equal(dist.to_hist(), [1, 2, 0, 4])
        assert_array_equal(dist.to_cumulative(), [1, 3, 3, 7])
        assert dist.hist.dtype == np.float64

    def test_repr_of_empty(self):
        dist = Distribution(EDGES)
        assert repr(dist) == str(dist) == "<Distribution: size=0, mean=nan>"

    def test_repr_of_full(self):
        dist = Distribution(EDGES, [1, 2, 0, 0])
        assert repr(dist) == str(dist) == "<Distribution: size=3, mean=4.00>"

    def test_empty_equal(self):
        assert Distribution(EDGES) == Distribution(EDGES)

    def test_empty_not_equal(self):
        assert Distribution(EDGES) != Distribution(EDGES, [1, 2, 0, 4])

    def test_equal(self):
        assert Distribution(EDGES, [1, 2, 0, 4]) == Distribution(EDGES, [1, 2, 0, 4])

    def test_not_equal(self):
        assert Distribution(EDGES, [1, 2, 0, 4]) != Distribution(EDGES, [1, 2, 0, 5])

    def test_add_distribution(self):
        dist = Distribution(EDGES, [1, 2, 0, 0]) + Distribution(EDGES, [0, 2, 0, 4])
        assert_array_equal(dist.hist, [1, 4, 0, 4])

    def test_add_distribution_in_place(self):
        dist = Distribution(EDGES, [1, 2, 0, 0])
        hist = dist.hist
        dist += Distribution(EDGES, [0, 2, 0, 4])
        assert_array_equal(hist, [1, 4, 0, 4])

    def test_size_of_empty(self):
        dist = Distribution(EDGES)
        assert dist.size() == 0

    def test_size_of_full(self):
        dist = Distribution(EDGES, [1, 2, 0, 4])
        assert dist.size() == 7

    def test_mean_of_empty(self):
        dist = Distribution(EDGES, [0, 0, 0, 0])
        assert np.isnan(dist.mean())

    def test_mean_of_first_bin(self):
        dist = Distribution(EDGES, [7, 0, 0, 0])
        assert dist.mean() == 1

    def test_mean_of_inner_bin(self):
        dist = Distribution(EDGES, [0, 7, 0, 0])
        assert dist.mean() == 5.5

    def test_mean_of_last_bin(self):
        dist = Distribution(EDGES, [0, 0, 0, 7])
        assert np.isnan(dist.mean())

    def test_mean_of_multiple_bins(self):
        dist = Distribution(EDGES, [3, 1, 0, 0])
        assert dist.mean() == pytest.approx((3 * 1 + 1 * 5.5) / 4)

    def test_add_lt_first_edge(self):
        dist = Distribution(EDGES)
        dist.add(0.9)
        assert_array_equal(dist.hist, [1, 0, 0, 0])

    def test_add_eq_first_edge(self):
        dist = Distribution(EDGES)
        dist.add(1)
        assert_array_equal(dist.hist, [1, 0, 0, 0])

    def test_add_gt_first_edge(self):
        dist = Distribution(EDGES)
        dist.add(1.1)
        assert_array_equal(dist.hist, [0, 1, 0, 0])

    def test_add_lt_last_edge(self):
        dist = Distribution(EDGES)
        dist.add(99)
        assert_array_equal(dist.hist, [0, 0, 1, 0])

    def test_add_eq_last_edge(self):
        dist = Distribution(EDGES)
        dist.add(100)
        assert_array_equal(dist.hist, [0, 0, 1, 0])

    def test_add_gt_last_edge(self):
        dist = Distribution(EDGES)
        dist.add(101)
        assert_array_equal(dist.hist, [0, 0, 0, 1])

    def test_add_with_weight(self):
        dist = Distribution(EDGES)
        dist.add(17, 7)
        assert_array_equal(dist.hist, [0, 0, 7, 0])

    def test_add_not_scalar(self):
        dist = Distribution(EDGES)
        with pytest.raises(ValueError) as exc_info:
            dist.add([1, 1, 17])
        assert str(exc_info.value) == "Value must be a scalar."

    def test_update(self):
        dist = Distribution(EDGES)
        dist.update([1, 1, 17])
        assert_array_equal(dist.hist, [2, 0, 1, 0])

    def test_update_with_weight(self):
        dist = Distribution(EDGES)
        dist.update([1, 1, 17], weights=10)
        assert_array_equal(dist.hist, [20, 0, 10, 0])

    def test_update_with_multiple_weights(self):
        dist = Distribution(EDGES)
        dist.update([1, 1, 17], weights=[3, 2, 1])
        assert_array_equal(dist.hist, [5, 0, 1, 0])

    def test_update_not_1d(self):
        dist = Distribution(EDGES)
        with pytest.raises(ValueError) as exc_info:
            dist.update([[1, 1, 17]])
        assert str(exc_info.value) == "Values must be 1-D array-like."
