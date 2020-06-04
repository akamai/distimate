import numpy as np

interp_right = np.interp


def interp_left(v, xp, fp, left=None, right=None):
    """
    Like :func:`numpy.interp` but uses lowest of equal xp values.

    >>> np.interp(10, [0, 10, 10, 20], [1, 2, 3, 4])
    3.0
    >>> interp_left(10, [0, 10, 10, 20], [1, 2, 3, 4])
    2.0
    """
    v = np.asarray(v)
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    return np.interp(-v, -xp[::-1], fp[::-1], left=right, right=left)


def interp_middle(v, xp, fp, left=None, right=None):
    """
    Return a midpoint between ``interp_left`` and ``interp_right``.
    """
    low = interp_left(v, xp, fp, left=left, right=right)
    high = np.interp(v, xp, fp, left=left, right=right)
    return (low + high) / 2


class _StatsFunction:
    """
    Statistical function.

    Provides :attr:`x` and :attr:`y` attributes for plotting,
    or can be called for approximating value at an arbitrary point.
    """

    __slots__ = ("_x", "_y", "_left", "_right", "_interp")

    def __init__(self, x, y, *, left=np.nan, right=np.nan, interp=interp_right):
        self._x = x
        self._y = y
        self._left = left
        self._right = right
        self._interp = interp

    def __call__(self, v):
        """
        Compute function value at the given point.

        :param v: scalar value or Numpy array-like
        :return: scalar value or Numpy array depending on *x*
        """
        return self._interp(v, self._x, self._y, left=self._left, right=self._right)

    @property
    def x(self):
        """Return :class:`numpy.array` of x-values for plotting"""
        return self._x

    @property
    def y(self):
        """Return :class:`numpy.array` of y-values for plotting"""
        return self._y


def mean(edges, hist):
    """
    Estimate mean from a histogram.

    The approximated mean is for sanity checks only,
    it is ineffective and imprecise to estimate mean from a histogram.

    Return NaN for distributions with no samples.

    - Inner bins are represented by their midpoint
      (assume that samples are evenly distributed in bins).
    - The left outer bin is represented by the leftmost edge
      (assume that there are no samples bellow the supported range).
    - Return NaN if the rightmost bin is not empty
      (because we cannot approximate outliers).

    :param edges: 1-D array-like, ordered histogram edges
    :param hist: 1-D array-like, one item longer than edges
    :return: float number
    """
    edges = np.asarray(edges)
    hist = np.asarray(hist)
    total = hist.sum()
    if total == 0 or hist[-1] != 0:
        return np.nan
    # For example, if edges are 0, 10, 100
    # then buckets are [0, 0], (0, 10], (10, 100].
    # So left = [0, 0, 10] and right = [0, 10, 100].
    left = np.r_[edges[0], edges[:-1]]
    right = edges
    middle = (left + right) / 2
    return np.sum(hist[:-1] * middle) / total


class PDF(_StatsFunction):
    """
    Probability density function (PDF).

    Callable object with ``.x`` and ``.y`` attributes.
    The attributes can be used for plotting, or the function
    can be called to estimate a PDF value at arbitrary point.
    The callable accepts a single value or an array-like.

    The returned callable takes inputs from a distribution domain
    and returns outputs between 0 and 1 (inclusive).

    The PDF values provides relative likelihoods of various
    distribution values. It is computed from a histogram
    by normalizing relative frequencies by bucket widths.

    - For inputs less than the first edges,
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

    :param edges: 1-D array-like, ordered histogram edges
    :param hist: 1-D array-like, one item longer than edges`
    """

    __slots__ = ()

    def __init__(self, edges, hist):
        edges = np.asarray(edges)
        hist = np.asarray(hist)
        total = np.sum(hist)
        if total == 0:
            # When we have no samples then the function is undefined.
            x = edges[[0, -1]]
            y = np.full_like(x, np.nan, dtype=np.float64)
            right = np.nan
        else:
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
        super().__init__(x, y, left=0, right=right, interp=interp_left)


class CDF(_StatsFunction):
    """
    Cumulative distribution function (CDF).

    Callable object with ``.x`` and ``.y`` attributes.
    The attributes can be used for plotting, or the function
    can be called to estimate a CDF value at arbitrary point.
    The callable accepts a single value or an array-like.

    The returned callable takes inputs from a distribution domain
    and returns outputs between 0 and 1 (inclusive).

    ``cdf(x)`` returns a probability that a distribution
    value will be less than or equal to ``.x``.

    - For inputs less than the first edge,
      the CDF will always return zero.
    - Function return exact values for inputs equal to histogram edges.
      Values inside histogram buckets are interpolated.
    - CDF of the first edge can be used to obtain how many
      samples were equal to that edge (typically zero)
    - For inputs greater than the last edge,
      the PDF returns either one or NaN,
      depending on whether the last histogram bucket is empty.

    :param edges: 1-D array-like, ordered histogram edges
    :param hist: 1-D array-like, one item longer than edges
    """

    __slots__ = ()

    def __init__(self, edges, hist):
        edges = np.asarray(edges)
        hist = np.asarray(hist)
        cumulative = np.cumsum(hist, dtype=np.float64)
        if cumulative[-1] == 0:
            # When we have no samples then the function is undefined.
            y = np.full_like(edges, np.nan, dtype=np.float64)
            right = np.nan
        else:
            y = cumulative[:-1] / cumulative[-1]
            right = 1 if hist[-1] == 0 else np.nan
        super().__init__(edges, y, left=0, right=right)


class Quantile(_StatsFunction):
    """
   Create a quantile function.

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
     In that case, duplicate x-values will be plotted.
     When called, the quantile function will a middle of possible values.
   - The function returns NaN for values outside of the <0, 1> range.
   - When called with zero,
     returns the left edge of the smallest non-empty bucket.
     If the first bucket is not empty, returns the first edge.
   - When called with one,
     returns the right edge of the greatest non-empty bucket.
     If the last bucket is not empty, returns NaN.
   """

    __slots__ = ()

    def __init__(self, edges, hist):
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
        super().__init__(x, y, interp=interp_middle)
