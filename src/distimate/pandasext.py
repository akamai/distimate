import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


def _format_number(v):
    if round(v) == v:
        return str(int(v))
    return str(v)


class DistributionAccessor(object):
    """
    Implements ``.dist`` accessor on :class:`pandas.Series`.

    Allows to easily call :class:`.Distribution` methods
    on all instances in Pandas Series:

    .. code-block:: python

        df[col] = pd.Series.dist.from_histogram(histograms)
        median = df[col].dist.quantile(0.5)

    """

    def __init__(self, series):
        self._series = series

    @staticmethod
    def from_histogram(dist_type, histograms, *, name=None):
        """
        Construct a new :class:`pandas.Series` from histograms.

        This is a static method that can be accessed
        as ``pd.Series.dist.from_histogram()``.

        :param dist_type: :class:`.DistributionType` defining histogram buckets
        :param histograms: :class:`pandas.DataFrame` or 2-D array-like
        :param name: Optional name of the series.
        :return: :class:`pandas.Series`
        """
        index = None
        if isinstance(histograms, pd.DataFrame):
            index = histograms.index
            histograms = histograms.values
        dists = [dist_type.from_histogram(histogram) for histogram in histograms]
        return pd.Series(dists, index=index, name=name)

    @staticmethod
    def from_cumulative(dist_type, cumulatives, *, name=None):
        """
        Construct a new :class:`pandas.Series` from cumulative histograms.

        This is a static method that can be accessed
        as ``pd.Series.dist.from_cumulative()``.

        :param dist_type: :class:`.DistributionType` defining histogram buckets
        :param histograms: :class:`pandas.DataFrame` or 2-D array-like
        :param name: Optional name of the series.
        :return: :class:`pandas.Series`
        """
        index = None
        if isinstance(cumulatives, pd.DataFrame):
            index = cumulatives.index
            cumulatives = cumulatives.values
        histograms = np.diff(cumulatives, prepend=0)
        dists = [dist_type.from_histogram(histogram) for histogram in histograms]
        return pd.Series(dists, index=index, name=name)

    def to_histogram(self):
        """
        Convert :class:`pandas.Series` of :class:`.Distribution`
        instances to histograms.

        :return: :class:`pandas.DataFrame` with histogram values.
        """

        data = self.values
        columns = [self._get_name(f"histogram{i}") for i in range(data.shape[-1])]
        return pd.DataFrame(data, index=self._series.index, columns=columns)

    def to_cumulative(self):
        """
        Convert :class:`pandas.Series` of :class:`.Distribution` instances
        to cumulative histograms.

        :return: :class:`pandas.DataFrame` with cumulative values
        """
        data = np.cumsum(self.values, axis=1)
        columns = [self._get_name(f"cumulative{i}") for i in range(data.shape[-1])]
        return pd.DataFrame(data, index=self._series.index, columns=columns)

    def pdf(self, v):
        """
        Compute PDF for :class:`pandas.Series` of :class:`.Distribution` instances.

        :param v: input value, or list of them
        :return: :class:`pandas.Series`
        """
        return self._compute(self._pdf, v)

    def cdf(self, v):
        """
        Compute CDF for series of distribution instances.

        :param v: input value, or list of them
        :return: :class:`pandas.Series`
        """
        return self._compute(self._cdf, v)

    def quantile(self, v):
        """
        Compute quantile function :class:`pandas.Series`
        of :class:`.Distribution` intances.

        :param v: input value, or list of them
        :return: :class:`pandas.Series`
        """
        return self._compute(self._quantile, v)

    @property
    def values(self):
        """
        Values of the underlying histograms.

        :return: 2-D :class:`numpy.array`
        """
        if self._series.empty:
            return np.zeros((0, 0))
        return np.array([dist.values for dist in self._series])

    def _compute(self, meth, v):
        if isinstance(v, (tuple, list)):
            columns = [meth(i) for i in v]
            return pd.concat(columns, axis=1)
        return meth(v)

    def _pdf(self, v):
        name = self._get_name(f"pdf{_format_number(v)}")
        data = [dist.pdf(v) if pd.notna(dist) else np.nan for dist in self._series]
        return pd.Series(data, index=self._series.index, name=name)

    def _cdf(self, v):
        name = self._get_name(f"cdf{_format_number(v)}")
        data = [dist.cdf(v) if pd.notna(dist) else np.nan for dist in self._series]
        return pd.Series(data, index=self._series.index, name=name)

    def _quantile(self, v):
        name = self._get_name(f"q{_format_number(100 * v).rjust(2, '0')}")
        data = [dist.quantile(v) if pd.notna(dist) else np.nan for dist in self._series]
        return pd.Series(data, index=self._series.index, name=name)

    def _get_name(self, name):
        if self._series.name is None:
            return name
        return f"{self._series.name}_{name}"


def register_to_pandas():
    if pd is None:
        return  # Pandas are not installed
    pd.api.extensions.register_series_accessor("dist")(DistributionAccessor)
