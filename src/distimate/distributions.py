import numpy as np

from distimate.stats import make_cdf, make_pdf, make_quantile


class Distribution:
    """
    Statistical distribution represented by its histogram.

    Provides an object interface on top of a histogram array.
    Supports distribution merging and comparison.
    Implements approximation of common statistical functions.

    :param edges: 1-D array-like, ordered histogram edges
    :param values: 1-D array-like, histogram, one item longer than *edges*
    """

    __slots__ = ("edges", "values")

    dtype = np.float64

    def __init__(self, edges, values=None):
        #: Edges of the underlying histogram.
        self.edges = np.asarray(edges)
        size = len(self.edges) + 1
        if values is None:
            values = np.zeros(size, dtype=self.dtype)
        else:
            values = np.asarray(values, dtype=self.dtype)
            if values.ndim != 1:
                raise ValueError("Histogram must be 1-D array-like.")
            if len(values) != size:
                raise ValueError("Histogram must have len(edges) + 1 items.")
            if not np.all(values >= 0):
                raise ValueError("Histogram values must not be negative.")
        #: Values of the underlying histogram.
        self.values = values

    def __repr__(self):
        name = type(self).__name__
        return f"<{name}: weight={self.weight():.0f}, mean={self.mean():.2f}>"

    def __eq__(self, other):
        """Return whether distribution histograms are equal."""
        if isinstance(other, Distribution):
            return np.array_equal(self.values, other.values)
        return NotImplemented

    def __add__(self, other):
        """Combine this distribution with other distribution."""
        if isinstance(other, Distribution):
            values = self.values + other.values
            return Distribution(self.edges, values)
        return NotImplemented

    def __iadd__(self, other):
        """Combine this distribution with other distribution inplace."""
        if isinstance(other, Distribution):
            self.values += other.values
            return self
        return NotImplemented

    @classmethod
    def from_samples(cls, edges, samples, weights=None):
        """
        Create a distribution from a list of values.

        :param edges: 1-D array-like, ordered histogram edges
        :param samples: 1-D array-like
        :param weights: optional scalar
            or 1-D array-like with same length as samples.
        :return: a new :class:`Distribution`
        """
        dist = cls(edges)
        dist.update(samples, weights)
        return dist

    @classmethod
    def from_histogram(cls, edges, histogram):
        """
        Create a distribution from a histogram.

        :param edges: 1-D array-like, ordered histogram edges
        :param histogram: 1-D array-like, one item longer than edges
        :return: a new :class:`Distribution`
        """
        return cls(edges, histogram)

    @classmethod
    def from_cumulative(cls, edges, cumulative):
        """
        Create a distribution from a cumulative histogram.

        :param edges: 1-D array-like, ordered histogram edges
        :param cumulative: 1-D array-like, one item longer than edges
        :return: a new :class:`Distribution`
        """
        values = np.diff(cumulative, prepend=0)
        return cls(edges, values)

    def to_histogram(self):
        """
        Return a histogram of this distribution as a NumPy array.

        :return: 1-D :class:`numpy.array`
        """
        return self.values.copy()

    def to_cumulative(self):
        """
        Return a cumulative histogram of this distribution as a NumPy array.

        :return: 1-D :class:`numpy.array`
        """
        return np.cumsum(self.values)

    def weight(self):
        """
        Return a total weight of samples in this distribution.

        :return: float number
        """
        return self.values.sum()

    def add(self, value, weight=None):
        """
        Add a new item to this distribution.

        :param value: item to add
        :param weight: optional item weight
        """
        if np.ndim(value) != 0:
            raise ValueError("Value must be a scalar.")
        if weight is None:
            weight = 1
        index = self.edges.searchsorted(value)
        self.values[index] += weight

    def update(self, values, weights=None):
        """
        Add multiple items to this distribution.

        :param values: items to add, 1-D array-like
        :param weights: optional scalar or 1-D array-like
            with same length as samples.
        """
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError("Values must be 1-D array-like.")
        if weights is None:
            weights = 1
        index = self.edges.searchsorted(values)
        # Cannot use self._hist[index] += weights because it does
        # not accumulate if index contains duplicate values.
        np.add.at(self.values, index, weights)

    def mean(self):
        """
        Estimate mean of this distribution.

        The approximated mean is for sanity checks only,
        it is ineffective and imprecise to estimate mean from a histogram.

        Return NaN for distributions with no samples.

        - Inner bins are represented by their midpoint
          (assume that samples are evenly distributed in bins).
        - The left outer bin is represented by the leftmost edge
          (assume that there are no samples bellow the supported range).
        - Return NaN if the rightmost bin is not empty
          (because we cannot approximate outliers).
        """
        total = self.values.sum()
        if total == 0 or self.values[-1] != 0:
            return np.nan
        # For example, if edges are 0, 10, 100
        # then buckets are [0, 0], (0, 10], (10, 100].
        # So left = [0, 0, 10] and right = [0, 10, 100].
        left = np.r_[self.edges[0], self.edges[:-1]]
        right = self.edges
        middle = (left + right) / 2
        return np.sum(self.values[:-1] * middle) / total

    @property
    def pdf(self):
        """
        Probability density function (PDF) of this distribution.

        See :func:`.make_pdf` for details.

        :return: :class:`.StatsFunction`
        """
        return make_pdf(self.edges, self.values)

    @property
    def cdf(self):
        """
        Cumulative distribution function (CDF) of this distribution.

        See :func:`.make_cdf` for details.

        :return: :class:`.StatsFunction`
        """
        return make_cdf(self.edges, self.values)

    @property
    def quantile(self):
        """
        Quantile function of this distribution.

        See :func:`.make_quantile` for details.

        :return: :class:`.StatsFunction`
        """
        return make_quantile(self.edges, self.values)
