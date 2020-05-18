import numpy as np


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


class InterpolatingFunction:
    """Function interpolating between sample values."""

    __slots__ = ("x", "y", "left", "right")

    def __init__(self, x, y, left=None, right=None):
        self.x = x
        self.y = y
        self.left = left
        self.right = right

    def __call__(self, v):
        """
        Compute function value at the given point.

        :param x: scalar value or Numpy array-like
        :return: scalar value or Numpy array depending on *x*
        """
        return self._interp(v, self.x, self.y, left=self.left, right=self.right)

    _interp = staticmethod(np.interp)
