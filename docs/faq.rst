
FAQ
===

What is Distimate useful for?
    Distimate started as a library
    for plotting empirical cumulative distribution functions (ECDF).

    It can plot CDF not only from an array of samples
    (similar to `statsmodels ECDF <https://www.statsmodels.org/stable/generated/statsmodels.distributions.empirical_distribution.ECDF.html>`_),
    but also from a histogram.
    The advantage of histograms is that they can be computed by a database,
    aggregating millions of rows to a few dozens of buckets.

    Having CDFs, we can ask for median or other quantiles.
    Histograms can be easily merged, allowing to estimate quantiles of grouped data.

    Distimate helps to write cleaner code.
    The :class:`.Distribution` class wraps two arrays into a simple object with a nice interface.
    The Pandas :class:`.DistributionAccessor` helps to remove unnecessary boilerplate.


How are approximations lossy?
    CDF values at histogram edges are exact.
    If you choose the edges wisely at round values, the user may never ask for CDF at other points.

    When plotted using Matplotlib,
    you will hardly notice a difference between a plot with 100 and 1000 points.
    It is important to use similar scales for both histogram edges and plot axis.

    In case of quantile estimates, the error will never be larger than a width of a bucket.
    If samples are evenly distributed in the bucket, the error will be much smaller.

    You can look at a `paper <https://arxiv.org/abs/2001.06561>`_
    explaining a similar approach.


Why does Distimate use more buckets than NumPy?
    If you define 10 edges, :func:`numpy.histogram` will create a histogram with 9 buckets.
    Distimate will need a histogram with 11 buckets,
    because it also counts items bellow the left-most edge and above the right-most edge.

    Counting the out-of-range items is important for plotting CDFs or computing quantiles.
    It is necessary to know a total count to get relative a density right.

    Distimate assumes that edges are defined only once and then reused.
    With the constant edge, risk the out-of-range items cannot be eliminated.


Why does Distimate use left-open intervals (unlike NumPy)?
    If you define your buckets as ``[0, 10, 100]``,
    :func:`numpy.histogram` will insert the number 10 to the ``[10, 100)`` bucket.
    Distimate puts 10 to the ``(0, 10]`` bucket.

    The left-open interval correspond to definition of CDF,
    where ``cdf(x)`` includes all samples lesser than or equal to ``x``.

    This is especially important for ``cdf(0)``
    that should include all samples equal to zero.


Why does Distimate approximate left and right out-of-range values differently?

    The first bucket contains items lesser than or equal to the left-most edge.
    For approximations, we assume that the items in this bucket are equal to the left-most edge.

    The last bucket contains items greater that the right-most edge.
    We have no approximation for items in this bucket.

    For metrics that do not have negative values, the left-most edge is typically set to zero.
    The first bucket then counts samples that are exactly zero.
    Because the zero is a prominent value,
    Distimate is designed to return exact estimates for it.

    For example, if most samples are zero, median should return zero.
    This would not be possible if the first bucket contained negative values.

    Sample in the last bucket can be arbitrary high, so no approximation is possible.
