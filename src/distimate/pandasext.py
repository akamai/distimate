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
    """Implements ``.dist`` attribute on Pandas series."""

    def __init__(self, series):
        self._series = series

    @staticmethod
    def from_histogram(dist_type, histograms, *, name=None):
        index = None
        if isinstance(histograms, pd.DataFrame):
            index = histograms.index
            histograms = histograms.values
        dists = [dist_type.from_histogram(histogram) for histogram in histograms]
        return pd.Series(dists, index=index, name=name)

    @staticmethod
    def from_cumulative(dist_type, cumulatives, *, name=None):
        index = None
        if isinstance(cumulatives, pd.DataFrame):
            index = cumulatives.index
            cumulatives = cumulatives.values
        histograms = np.diff(cumulatives, prepend=0)
        dists = [dist_type.from_histogram(histogram) for histogram in histograms]
        return pd.Series(dists, index=index, name=name)

    def to_histogram(self):
        data = self.values
        columns = [self._get_name(f"histogram{i}") for i in range(data.shape[-1])]
        return pd.DataFrame(data, index=self._series.index, columns=columns)

    def to_cumulative(self):
        data = np.cumsum(self.values, axis=1)
        columns = [self._get_name(f"cumulative{i}") for i in range(data.shape[-1])]
        return pd.DataFrame(data, index=self._series.index, columns=columns)

    @property
    def values(self):
        if self._series.empty:
            return np.zeros((0, 0))
        return np.array([dist.values for dist in self._series])

    def pdf(self, v):
        """
        Compute PDF for series of distribution instances.

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
        Compute the given quantile for all distributions in the series.

        :param v: input value, or list of them
        :return: :class:`pandas.Series`
        """
        return self._compute(self._quantile, v)

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
