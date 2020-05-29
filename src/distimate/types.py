import numpy as np

from distimate.distributions import Distribution


class DistributionType:
    """
    Factory for creating distributions with constant histogram edges.

    :param edges: 1-D array-like, ordered histogram edges
    """

    __slots__ = ("_edges",)

    _dist_cls = Distribution

    def __init__(self, edges):
        self._edges = np.asarray(edges)

    @property
    def edges(self):
        """
        Edges of the underlying histogram

        :return: :class: 1-D `numpy.array`, ordered histogram edges
        """
        return self._edges

    def empty(self):
        """
        Create an empty distribution.

        :return: a new :class:`Distribution`
        """
        return self._dist_cls(self._edges)

    def from_samples(self, samples, weights=None):
        """
        Create a distribution from a list of values.

        :param samples: 1-D array-like
        :param weights: optional 1-D array-like
        :return: a new :class:`Distribution`
        """
        return self._dist_cls.from_samples(self._edges, samples, weights)

    def from_histogram(self, histogram):
        """
        Create a distribution from a histogram.

        :param histogram: 1-D array-like
        :return: a new :class:`Distribution`
        """
        return self._dist_cls.from_histogram(self._edges, histogram)

    def from_cumulative(self, cumulative):
        """
        Create a distribution from a cumulative histogram.

        :param cumulative: 1-D array-like
        :return: a new :class:`Distribution`
        """
        return self._dist_cls.from_cumulative(self._edges, cumulative)
