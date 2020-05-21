import numpy as np

from distimate.distributions import Distribution


class DistributionType:
    """
    Factory for creating distributions with constant histogram edges.

    :param edges: 1-D array-like, list of histogram edges
    """

    __slots__ = ("edges",)

    _dist_cls = Distribution

    def __init__(self, edges):
        self.edges = np.asarray(edges)

    def empty(self):
        """
        Create an empty distribution.

        :return: :class:`Distribution`
        """
        return self._dist_cls(self.edges)

    def from_samples(self, samples, weights=None):
        """
        Create a distribution from a list of values.

        :param samples: 1-D array-like
        :param weights: 1-D array-like
        :return: :class:`Distribution`
        """
        dist = self.empty()
        dist.update(samples, weights)
        return dist

    def from_histogram(self, hist):
        """
        Create a distribution from a histogram.

        :param hist: 1-D array-like
        :return: :class:`Distribution`
        """
        hist = self._array_from_seq(hist)
        return self._dist_cls(self.edges, hist)

    def from_cumulative(self, cumulative):
        """
        Create a distribution from a cumulative histogram.

        :param cumulative: 1-D array-like
        :return: :class:`Distribution`
        """
        cumulative = self._array_from_seq(cumulative)
        hist = np.ediff1d(cumulative, to_begin=cumulative[0])
        return self.from_histogram(hist)

    def _array_from_seq(self, data):
        if np.ndim(data) != 1:
            raise ValueError("Histogram must be 1-D array-like.")
        array = np.asarray(data, dtype=self._dist_cls.dtype)
        if len(array) != len(self.edges) + 1:
            raise ValueError("Histogram must have len(edges) + 1 items.")
        return array
