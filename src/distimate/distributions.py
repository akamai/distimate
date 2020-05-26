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
        return f"<{name}: size={self.size():.0f}, mean={self.mean():.2f}>"

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

    def size(self):
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

        Return NaN for distributions with no samples.

        - Inner bins are represented by their midpoint (assume that
          samples are evenly distributed in bins).
        - The left outer bin is represented by the leftmost edge
          (assume that there are no samples bellow the supported range).
        - Return NaN if the rightmost bin is not empty.
        """
        total = self.values.sum()
        if total == 0 or self.values[-1] != 0:
            return np.nan
        left = np.r_[self.edges[0], self.edges[:-1]]
        right = self.edges
        middle = (left + right) / 2
        return np.sum(self.values[:-1] * middle) / total

    @property
    def pdf(self):
        """
        Probability density function (PDF) of this distribution.

        Returns a callable object with ``.x`` and ``.y`` attributes.
        The attributes can be used for plotting, or the function
        can be called to estimate a PDF value at arbitrary point.
        The callable accepts a single value or an array-like.

        The returned callable takes inputs from a distribution domain
        and returns outputs between 0 and 1 (inclusive).

        The PDF values provides relative likelihoods of various
        distribution values. It is computed from a histogram
        by normalizing relative frequencies by bucket widths.

        - For inputs lesser than the first edges,
          the PDF will always return zero.
        - For inputs equal to the first edge (typically zero),
          the PDF function will return zero or NaN,
          depending on whether the first histogram bucket is empty.
          This is because the PDF is not defined for discrete distributions.
        - For inputs in each of inner histogram buckets (which are left-open),
          one value is returned. On a plot, this will form a staircase.
          To plot a non-continuous distribution, x-values are duplicated.
        - For inputs greater than the last edge,
          the PDF returns either zero or NaN,
          depending on whether the last histogram bucket is empty.

        :return: :class:`.StatsFunction`
        """
        return make_pdf(self.edges, self.values)

    @property
    def cdf(self):
        """
        Cumulative distribution function (CDF) of this distribution.

        Returns a callable object with ``.x`` and ``.y`` attributes.
        The attributes can be used for plotting, or the function
        can be called to estimate a CDF value at arbitrary point.
        The callable accepts a single value or an array-like.

        The returned callable takes inputs from a distribution domain
        and returns outputs between 0 and 1 (inclusive).

        ``cdf(x)`` returns a probability that a distribution
        value will be lesser than or equal to ``.x``.

        - For inputs lesser than the first edge,
          the CDF will always return zero.
        - Function return exact values for inputs equal to histogram edges.
          Values inside histogram buckets are interpolated.
        - CDF of the first edge can be used to obtain how many
          samples were equal to that edge (typically zero)
        - For inputs greater than the last edge,
          the PDF returns either one or NaN,
          depending on whether the last histogram bucket is empty.

        :return: :class:`.StatsFunction`
        """
        return make_cdf(self.edges, self.values)

    @property
    def quantile(self):
        """
        Quantile function of this distribution.

        Returns a callable object with ``.x`` and ``.y`` attributes.
        The attributes can be used for plotting, or the function
        can be called to estimate a quantile value at arbitrary point.
        The function accepts a single value or an array-like.

        The returned callable takes inputs from a range between 0 and 1
        (inclusive) and returns outputs from a distribution domain.

        ``quantile(q)`` returns the smallest ``.x`` for which ``cdf(x) >= q``.

        - If the first histogram bucket is not empty,
          the quantile value can return the first edge for many inputs.
        - If an inner histogram bucket is empty,
          then the quantile value can be ambiguous.
          In that case, duplicate x-values will be plotted. When called,
          the quantile function will return the lowest of possible results.
        - The function returns NaN for values outside of the <0, 1> range.
        - When called with zero,
          returns the left edge of the smallest non-empty bucket.
          If the first bucket is not empty, returns the first edge.
        - When called with one,
          returns the right edge of the greatest non-empty bucket.
          If the last bucket is not empty, returns NaN.

        :return: :class:`.StatsFunction`
        """
        return make_quantile(self.edges, self.values)
