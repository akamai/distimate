import numpy as np


class StatsFunction:
    """
    Statistical function.

    Provides :attr:`x` and :attr:`y` attributes for plotting,
    or can be called for approximating value at an arbitrary point.
    """

    __slots__ = ("x", "y", "_left", "_right", "_interp")

    def __init__(self, x, y, *, left=np.nan, right=np.nan, interp=np.interp):
        #: :class:`numpy.array` of x-values for plotting
        self.x = x
        #: :class:`numpy.array` of y-values for plotting
        self.y = y
        self._left = left
        self._right = right
        self._interp = interp

    def __call__(self, v):
        """
        Compute function value at the given point.

        :param v: scalar value or Numpy array-like
        :return: scalar value or Numpy array depending on *x*
        """
        return self._interp(v, self.x, self.y, left=self._left, right=self._right)


def make_pdf(edges, hist):
    """
    Create a probability density function (PDF).

    This is an internal implementation of :meth:`.Distributon.pdf`.
    """
    edges = np.asarray(edges)
    hist = np.asarray(hist)
    total = np.sum(hist)
    if total == 0:
        # When we have no samples then the function is undefined.
        x = edges[[0, -1]]
        y = np.full_like(x, np.nan, dtype=np.float64)
        return StatsFunction(x, y, left=0)
    # Values in the first bucket should be equal to the first edge.
    # Because PDF is not defined for discrete distributions,
    # the value at the first edge is undefined if nonzero.
    head = 0 if hist[0] == 0 else np.nan
    # PDF values are relative frequencies normalized by bucket width.
    body = hist[1:-1] / np.diff(edges) / total
    # Because we cannot create a continuous PDF function from a histogram,
    # we have to repeat all values twice to plot staircase.
    x_all = np.repeat(edges, 2)[:-1]
    y_all = np.r_[head, np.repeat(body, 2)]
    # Remove unnecessary points: A step does not change height can be removed.
    dups = (y_all[:-2] == y_all[1:-1]) & (y_all[1:-1] == y_all[2:])
    mask = np.r_[True, ~dups, True]
    x = x_all[mask]
    y = y_all[mask]
    # If the last bucket is nonempty, we cannot compute its PDF
    # because it has unknown (infinite) width.
    right = 0 if hist[-1] == 0 else np.nan
    return StatsFunction(x, y, left=0, right=right, interp=interp_left)


def make_cdf(edges, hist):
    """
    Create a cumulative distribution function (CDF).

    This is an internal implementation of :meth:`.Distributon.cdf`.
    """
    edges = np.asarray(edges)
    hist = np.asarray(hist)
    cumulative = np.cumsum(hist, dtype=np.float64)
    if cumulative[-1] == 0:
        # When we have no samples then the function is undefined.
        y = np.full_like(edges, np.nan, dtype=np.float64)
        return StatsFunction(edges, y, left=0)
    y = cumulative[:-1] / cumulative[-1]
    right = 1 if hist[-1] == 0 else np.nan
    return StatsFunction(edges, y, left=0, right=right)


def make_quantile(edges, hist):
    """
    Create a quantile function.

    This is an internal implementation of :meth:`.Distributon.quantile`.
    """
    edges = np.asarray(edges)
    hist = np.asarray(hist)
    cumulative = np.cumsum(hist, dtype=np.float64)
    if cumulative[-2] == 0:
        # When we have no samples then the function is undefined.
        # In addition to that, the function is also undefined
        # if all samples are in the last bucket (greater than the last edge).
        x = np.array([0, 1], dtype=np.float64)
        y = np.array([np.nan, np.nan], dtype=np.float64)
        return StatsFunction(x, y)
    cdf = cumulative[:-1] / cumulative[-1]
    # Make sure that x values start at zero and end at one.
    x_all = np.r_[0, cdf, 1]
    y_all = np.r_[edges[0], edges, np.nan]
    # If there are vertical chains of points with same x value,
    # keep only the first and the last point of each chain.
    # Heading or trailing chains are removed completely.
    diffs = np.r_[False, hist != 0, False]
    mask = diffs[:-1] | diffs[1:]
    x = x_all[mask]
    y = y_all[mask]
    return StatsFunction(x, y, interp=interp_left)


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
