import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from distimate.types import DistributionType


class TestDistributionConversions:
    """Test ``DistributionType.from_*`` and ``Distribution.to_*`` methods."""

    dist_type = DistributionType([1, 10, 100])

    def test_empty(self):
        dist = self.dist_type.empty()
        assert_array_equal(dist.values, [0, 0, 0, 0])

    def test_from_samples_list(self):
        dist = self.dist_type.from_samples([0, 42, 47])
        assert_array_equal(dist.values, [1, 0, 2, 0])

    def test_from_samples_lists_w_weights(self):
        dist = self.dist_type.from_samples([0, 42, 47], [5, 1, 2])
        assert_array_equal(dist.values, [5, 0, 3, 0])

    def test_from_samples_array(self):
        dist = self.dist_type.from_samples(np.array([0, 42, 47]))
        assert_array_equal(dist.values, [1, 0, 2, 0])

    def test_from_samples_array_w_weights(self):
        dist = self.dist_type.from_samples(np.array([0, 42, 47]), np.array([5, 1, 2]))
        assert_array_equal(dist.values, [5, 0, 3, 0])

    def test_from_samples_series(self):
        dist = self.dist_type.from_samples(pd.Series([0, 42, 47]))
        assert_array_equal(dist.values, [1, 0, 2, 0])

    def test_from_samples_series_w_weights(self):
        dist = self.dist_type.from_samples(pd.Series([0, 42, 47]), pd.Series([5, 1, 2]))
        assert_array_equal(dist.values, [5, 0, 3, 0])

    def test_from_samples_frame(self):
        with pytest.raises(ValueError) as exc_info:
            self.dist_type.from_samples(pd.DataFrame())
        assert str(exc_info.value) == "Values must be 1-D array-like."

    def test_from_samples_frame_column(self):
        df = pd.DataFrame({"x": [0, 42, 47]})
        dist = self.dist_type.from_samples(df["x"])
        assert_array_equal(dist.values, [1, 0, 2, 0])

    def test_from_histogram_list(self):
        dist = self.dist_type.from_histogram([2, 0, 1, 0])
        assert_array_equal(dist.values, [2, 0, 1, 0])

    def test_from_histogram_array(self):
        dist = self.dist_type.from_histogram(np.array([2, 0, 1, 0]))
        assert_array_equal(dist.values, [2, 0, 1, 0])

    def test_from_histogram_invalid_length(self):
        with pytest.raises(ValueError) as exc_info:
            self.dist_type.from_histogram([2, 0, 1])
        assert str(exc_info.value) == "Histogram must have len(edges) + 1 items."

    def test_from_cumulative_list(self):
        dist = self.dist_type.from_cumulative([2, 2, 3, 3])
        assert_array_equal(dist.values, [2, 0, 1, 0])

    def test_from_cumulative_array(self):
        dist = self.dist_type.from_cumulative(np.array([2, 2, 3, 3]))
        assert_array_equal(dist.values, [2, 0, 1, 0])

    def test_from_cumulative_invalid_length(self):
        with pytest.raises(ValueError) as exc_info:
            self.dist_type.from_cumulative([2, 2, 3])
        assert str(exc_info.value) == "Histogram must have len(edges) + 1 items."
