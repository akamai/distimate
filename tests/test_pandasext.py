import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from distimate.types import DistributionType

dist_type = DistributionType([0, 10, 100])


class TestDistributionAccessor:

    dists = [
        dist_type.from_samples([0, 5]),
        dist_type.from_samples([10, 20]),
    ]

    def test_pdf_of_anonymous_series(self):
        series = pd.Series(self.dists)
        assert_series_equal(series.dist.pdf(10), pd.Series([0.05, 0.05], name="pdf10"))

    def test_pdf_of_named_series(self):
        series = pd.Series(self.dists, name="price")
        assert_series_equal(
            series.dist.pdf(10), pd.Series([0.05, 0.05], name="price_pdf10")
        )

    def test_pdf_of_dataframe_column(self):
        df = pd.DataFrame({"price": self.dists})
        assert_series_equal(
            df["price"].dist.pdf(10), pd.Series([0.05, 0.05], name="price_pdf10")
        )

    def test_multiple_pdf(self):
        series = pd.Series(self.dists)
        assert_frame_equal(
            series.dist.pdf([0, 10]),
            pd.DataFrame({"pdf0": [np.nan, 0], "pdf10": [0.05, 0.05]}),
        )

    def test_cdf_of_anonymous_series(self):
        series = pd.Series(self.dists)
        assert_series_equal(series.dist.cdf(10), pd.Series([1.0, 0.5], name="cdf10"))

    def test_cdf_of_named_series(self):
        series = pd.Series(self.dists, name="price")
        assert_series_equal(
            series.dist.cdf(10), pd.Series([1.0, 0.5], name="price_cdf10")
        )

    def test_cdf_of_dataframe_column(self):
        df = pd.DataFrame({"price": self.dists})
        assert_series_equal(
            df["price"].dist.cdf(10), pd.Series([1.0, 0.5], name="price_cdf10")
        )

    def test_multiple_cdf(self):
        series = pd.Series(self.dists)
        assert_frame_equal(
            series.dist.cdf([0, 10]),
            pd.DataFrame({"cdf0": [0.5, 0], "cdf10": [1.0, 0.5]}),
        )

    def test_quantile_of_anonymous_series(self):
        series = pd.Series(self.dists)
        assert_series_equal(
            series.dist.quantile(0.5), pd.Series([0.0, 10.0], name="q50")
        )

    def test_quantile_of_named_series(self):
        series = pd.Series(self.dists, name="price")
        assert_series_equal(
            series.dist.quantile(0.5), pd.Series([0.0, 10.0], name="price_q50")
        )

    def test_quantile_of_dataframe_column(self):
        df = pd.DataFrame({"price": self.dists})
        assert_series_equal(
            df["price"].dist.quantile(0.5), pd.Series([0.0, 10.0], name="price_q50"),
        )

    def test_multiple_quantile_values(self):
        series = pd.Series(self.dists)
        assert_frame_equal(
            series.dist.quantile([0.05, 0.5]),
            pd.DataFrame({"q05": [0.0, 1.0], "q50": [0.0, 10.0]}),
        )
