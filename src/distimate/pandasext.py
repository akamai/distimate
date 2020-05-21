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
        rv = self._series.map(lambda dist: dist.pdf(v))
        return self._rename(rv, f"pdf{_format_number(v)}")

    def _cdf(self, v):
        rv = self._series.map(lambda dist: dist.cdf(v))
        return self._rename(rv, f"cdf{_format_number(v)}")

    def _quantile(self, v):
        rv = self._series.map(lambda dist: dist.quantile(v))
        return self._rename(rv, f"q{_format_number(100 * v).rjust(2, '0')}")

    def _rename(self, s, name):
        if self._series.name is not None:
            name = f"{self._series.name}_{name}"
        return s.rename(name, inplace=True)


def register_to_pandas():
    if pd is None:
        return  # Pandas are not installed
    pd.api.extensions.register_series_accessor("dist")(DistributionAccessor)
