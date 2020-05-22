from numpy.testing import assert_array_equal

from distimate.types import DistributionType


class TestDistributionConversions:
    """Test ``DistributionType.from_*`` and ``Distribution.to_*`` methods."""

    dist_type = DistributionType([1, 10, 100])

    def test_empty(self):
        dist = self.dist_type.empty()
        assert_array_equal(dist.values, [0, 0, 0, 0])

    def test_from_samples(self):
        dist = self.dist_type.from_samples([0, 42, 47])
        assert_array_equal(dist.values, [1, 0, 2, 0])

    def test_from_samples_w_weights(self):
        dist = self.dist_type.from_samples([0, 42, 47], [5, 1, 2])
        assert_array_equal(dist.values, [5, 0, 3, 0])

    def test_from_histogram(self):
        dist = self.dist_type.from_histogram([2, 0, 1, 0])
        assert_array_equal(dist.values, [2, 0, 1, 0])

    def test_from_cumulative(self):
        dist = self.dist_type.from_cumulative([2, 2, 3, 3])
        assert_array_equal(dist.values, [2, 0, 1, 0])
