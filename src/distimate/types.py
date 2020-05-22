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

        :return: a new :class:`Distribution`
        """
        return self._dist_cls(self.edges)

    def from_samples(self, samples, weights=None):
        """
        Create a distribution from a list of values.

        :param samples: 1-D array-like
        :param weights: optional 1-D array-like
        :return: a new :class:`Distribution`
        """
        return self._dist_cls.from_samples(self.edges, samples, weights)

    def from_histogram(self, histogram):
        """
        Create a distribution from a histogram.

        :param histogram: 1-D array-like
        :return: a new :class:`Distribution`
        """
        return self._dist_cls.from_histogram(self.edges, histogram)

    def from_cumulative(self, cumulative):
        """
        Create a distribution from a cumulative histogram.

        :param cumulative: 1-D array-like
        :return: a new :class:`Distribution`
        """
        return self._dist_cls.from_cumulative(self.edges, cumulative)
