import numpy as np


def make_pdf(edges, hist):
    """Create a probability distribution function."""
    # Ignore the first bucket because PDF function is not
    # defined for non-continuous functions.
    edges = np.asarray(edges)
    hist = np.asarray(hist)
    total = np.sum(hist)
    if total == 0:
        total = np.nan
    values = hist[1:-1] / np.diff(edges) / total
    # PDF of last bin is nan if non-empty because of its infinite width.
    right = 0 if (hist[-1] == 0 and total > 0) else np.nan
    return ProbabilityDensityFunction(edges, values, left=0, right=right)


class ProbabilityDensityFunction:
    """Function defined by enumerated sample values."""

    def __init__(self, edges, values, *, left=np.nan, right=np.nan):
        self._edges = edges
        self._values = values
        self._left = left
        self._right = right

    @property
    def x(self):
        """X coordinate for plotting."""
        return np.r_[self._edges[0], np.repeat(self._edges[1:-1], 2), self._edges[-1]]

    @property
    def y(self):
        """Y coordinate for plotting."""
        return np.repeat(self._values, 2)

    def __call__(self, x):
        """
        Compute function value at the given point.

        :param x: scalar value or Numpy array-like
        :return: scalar value or Numpy array depending on *x*
        """
        indices = np.searchsorted(self._edges, x)
        return np.r_[self._left, self._values, self._right][indices]


def make_cdf(edges, hist):
    """Create a cumulative distribution function (CDF)."""
    edges = np.asarray(edges)
    hist = np.asarray(hist)
    cumulative = np.cumsum(hist, dtype=np.float64)
    if cumulative[-1] == 0:
        # When we have no samples then the function is undefined.
        y = np.full_like(edges, np.nan, dtype=float)
        right = np.nan
    else:
        y = cumulative[:-1] / cumulative[-1]
        right = 1 if hist[-1] == 0 else np.nan
    return InterpolatingFunction(edges, y, left=0, right=right)


def make_quantile(edges, hist):
    """Create a quantile function."""
    edges = np.asarray(edges)
    hist = np.asarray(hist)
    cumulative = np.cumsum(hist, dtype=np.float64)
    if cumulative[-2] == 0:
        # When we have no samples then the function is undefined.
        # In addition to that, the function is also undefined
        # if all samples are in the last bucket (greater than the last edge).
        x = np.array([0, 1], dtype=np.float64)
        y = np.array([np.nan, np.nan], dtype=np.float64)
    else:
        normed = cumulative[:-1] / cumulative[-1]
        # Make sure that x values start at zero and end at one.
        x_candidates = np.r_[0, normed, 1]
        y_candidates = np.r_[edges[0], edges, np.nan]
        # If there are vertical chains of points with same x value,
        # keep only the first and the last point of each chain.
        # Heading or trailing chains are removed completely.
        non_zero = np.r_[False, hist != 0, False]
        mask = non_zero[:-1] | non_zero[1:]
        x = x_candidates[mask]
        y = y_candidates[mask]
    return InterpolatingFunction(x, y, interp=interp_left)


def interp_left(v, xp, fp, left=None, right=None):
    """
    Like ``np.interp`` but uses lowest of equal xp values.

    >>> np.interp(10, [0, 10, 10, 20], [1, 2, 3, 4])
    3.0
    >>> interp_left(10, [0, 10, 10, 20], [1, 2, 3, 4])
    2.0
    """
    v = np.asarray(v)
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    return np.interp(-v, -xp[::-1], fp[::-1], left=right, right=left)


class InterpolatingFunction:
    """Function interpolating between sample values."""

    __slots__ = ("x", "y", "_left", "_right", "_interp")

    def __init__(self, x, y, *, left=np.nan, right=np.nan, interp=np.interp):
        self.x = x
        self.y = y
        self._left = left
        self._right = right
        self._interp = interp

    def __call__(self, v):
        """
        Compute function value at the given point.

        :param x: scalar value or Numpy array-like
        :return: scalar value or Numpy array depending on *x*
        """
        return self._interp(v, self.x, self.y, left=self._left, right=self._right)
