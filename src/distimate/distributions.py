import numpy as np


class Distribution:
    """
    Statistical distribution represented by its histogram.

    :param edges: 1-D array-like
    :param hist: 1-D array-like, one item longer than *edges*
    """

    __slots__ = ("edges", "hist")

    dtype = np.float64

    def __init__(self, edges, hist=None):
        self.edges = np.asarray(edges)
        size = len(self.edges) + 1
        if hist is None:
            self.hist = np.zeros(size, dtype=self.dtype)
        else:
            self.hist = np.asarray(hist, dtype=self.dtype)
            if self.hist.ndim != 1:
                raise ValueError("Histogram must be 1-D array-like.")
            if len(self.hist) != size:
                raise ValueError("Histogram must have len(edges) + 1 items.")
            if not np.all(self.hist >= 0):
                raise ValueError("Histogram values must not be negative.")

    def __repr__(self):
        name = type(self).__name__
        return f"<{name}: size={self.size():.0f}, mean={self.mean():.2f}>"

    def __eq__(self, other):
        """Return whether distribution histograms are equal."""
        if isinstance(other, Distribution):
            return np.array_equal(self.hist, other.hist)
        return NotImplemented

    def __add__(self, other):
        """Combine this distribution with other distribution."""
        if isinstance(other, Distribution):
            values = self.hist + other.hist
            return Distribution(self.edges, values)
        return NotImplemented

    def __iadd__(self, other):
        """Combine this distribution with other distribution inplace."""
        if isinstance(other, Distribution):
            self.hist += other.hist
            return self
        return NotImplemented

    def to_hist(self):
        """
        Return a histogram of this distribution as a NumPy array.

        :return: :class:`numpy.array`
        """
        return self.hist.copy()

    def to_cumulative(self):
        """
        Return a cumulative histogram of this distribution as a NumPy array.

        :return: :class:`numpy.array`
        """
        return np.cumsum(self.hist)

    def size(self):
        """Return a total weight of samples in this distribution."""
        return self.hist.sum()

    def mean(self):
        """
        Estimate mean of this distribution.

        Return NaN for distributions with no samples.

        - Inner bins are represented by their midpoint (assume that
          samples are evenly distributed in bins).
        - The left outer bin is represented by the leftmost edge
          (assume that there are no samples bellow the supported range).
        - Return NaN if the rightmost bin is non-empty (assume that
          samples above the supported range can have any value).
        """
        total = self.hist.sum()
        if total == 0 or self.hist[-1] != 0:
            return np.nan
        left = np.r_[self.edges[0], self.edges[:-1]]
        right = self.edges
        middle = (left + right) / 2
        return np.sum(self.hist[:-1] * middle) / total

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
        self.hist[index] += weight

    def update(self, values, weights=None):
        """
        Add multiple items to this distribution.

        :param values: items to add, 1-D array_like
        :param weights: optional item weights
        """
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError("Values must be 1-D array-like.")
        if weights is None:
            weights = 1
        index = self.edges.searchsorted(values)
        # Cannot use self._hist[index] += weights because it does
        # not accumulate if index contains duplicate values.
        np.add.at(self.hist, index, weights)
