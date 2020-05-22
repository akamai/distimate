import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from distimate.types import DistributionType

dist_type = DistributionType([0, 10, 100])


class TestDistributionAccessor:

    dist1 = dist_type.from_samples([0, 5])
    dist2 = dist_type.from_samples([10, 20])

    dists = [dist1, dist2]

    def test_from_histogram_array(self):
        histograms = [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
        assert_series_equal(
            pd.Series.dist.from_histogram(dist_type, histograms, name="price"),
            pd.Series(self.dists, name="price"),
        )

    def test_from_empty_histogram_array(self):
        assert_series_equal(
            pd.Series.dist.from_histogram(dist_type, [], name="price"),
            pd.Series([], name="price"),
        )

    def test_from_histogram_frame(self):
        index = pd.Index(["a", "b"])
        histograms = pd.DataFrame(
            [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]], index=index
        )
        assert_series_equal(
            pd.Series.dist.from_histogram(dist_type, histograms, name="price"),
            pd.Series(self.dists, index=index, name="price"),
        )

    def test_from_empty_histogram_frame(self):
        index = pd.Index([])
        histograms = pd.DataFrame(index=index, columns=range(4))
        assert_series_equal(
            pd.Series.dist.from_histogram(dist_type, histograms, name="price"),
            pd.Series([], index=index, name="price"),
        )

    def test_from_cumulative_array(self):
        cumulatives = [[1, 2, 2.0, 2.0], [0.0, 1.0, 2.0, 2.0]]
        assert_series_equal(
            pd.Series.dist.from_cumulative(dist_type, cumulatives, name="price"),
            pd.Series(self.dists, name="price"),
        )

    def test_from_empty_cumulative_array(self):
        assert_series_equal(
            pd.Series.dist.from_cumulative(dist_type, [], name="price"),
            pd.Series([], name="price"),
        )

    def test_from_cumulative_frame(self):
        index = pd.Index(["a", "b"])
        cumulatives = pd.DataFrame(
            [[1, 2, 2.0, 2.0], [0.0, 1.0, 2.0, 2.0]], index=index
        )
        assert_series_equal(
            pd.Series.dist.from_cumulative(dist_type, cumulatives, name="price"),
            pd.Series(self.dists, index=index, name="price"),
        )

    def test_from_empty_cumulative_frame(self):
        index = pd.Index([])
        cumulatives = pd.DataFrame(index=index, columns=range(4))
        assert_series_equal(
            pd.Series.dist.from_cumulative(dist_type, cumulatives, name="price"),
            pd.Series([], index=index, name="price"),
        )

    def test_to_histogram_of_anonymous_series(self):
        series = pd.Series(self.dists)
        assert_frame_equal(
            series.dist.to_histogram(),
            pd.DataFrame(
                [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]],
                columns=["histogram0", "histogram1", "histogram2", "histogram3"],
            ),
        )

    def test_to_histogram_of_named_series(self):
        series = pd.Series(self.dists, name="price")
        assert_frame_equal(
            series.dist.to_histogram(),
            pd.DataFrame(
                [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]],
                columns=[
                    "price_histogram0",
                    "price_histogram1",
                    "price_histogram2",
                    "price_histogram3",
                ],
            ),
        )

    def test_to_cumulative_of_anonymous_series(self):
        series = pd.Series(self.dists)
        assert_frame_equal(
            series.dist.to_cumulative(),
            pd.DataFrame(
                [[1, 2, 2.0, 2.0], [0.0, 1.0, 2.0, 2.0]],
                columns=["cumulative0", "cumulative1", "cumulative2", "cumulative3"],
            ),
        )

    def test_to_cumulative_of_named_series(self):
        series = pd.Series(self.dists, name="price")
        assert_frame_equal(
            series.dist.to_cumulative(),
            pd.DataFrame(
                [[1, 2, 2.0, 2.0], [0.0, 1.0, 2.0, 2.0]],
                columns=[
                    "price_cumulative0",
                    "price_cumulative1",
                    "price_cumulative2",
                    "price_cumulative3",
                ],
            ),
        )

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

    def test_pdf_with_missing_data(self):
        series = pd.Series([None, pd.NA, self.dist1])
        assert_series_equal(
            series.dist.pdf(10), pd.Series([np.nan, np.nan, 0.05], name="pdf10")
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

    def test_cdf_with_missing_data(self):
        series = pd.Series([None, pd.NA, self.dist1])
        assert_series_equal(
            series.dist.cdf(10), pd.Series([np.nan, np.nan, 1.0], name="cdf10")
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

    def test_quantile_with_missing_data(self):
        series = pd.Series([None, pd.NA, self.dist1])
        assert_series_equal(
            series.dist.quantile(0.5), pd.Series([np.nan, np.nan, 0.0], name="q50")
        )

    def test_sum(self):
        series = pd.Series(self.dists)
        assert series.sum() == dist_type.from_samples([0, 5, 10, 20])

    def test_groupby_sum(self):
        df = pd.DataFrame({"cat": ["a", "a"], "price": self.dists})
        assert_frame_equal(
            df.groupby("cat").sum(),
            pd.DataFrame(
                {"price": [dist_type.from_samples([0, 5, 10, 20])]},
                index=pd.Index(["a"], name="cat"),
            ),
        )
