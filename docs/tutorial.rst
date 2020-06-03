
Tutorial
========

Statistical functions
---------------------

Distimate can approximate common statistical functions from a histogram.

.. note::

    Distimate is most useful in situations
    when it would be ineffective to retrieve a full dataset.

    For example, we can easily aggregate millions of database rows
    to 100 histogram buckets using SQL.
    The selected 100 points will provide enough detail for smooth CDF plots.


Distimate supports the following functions:

* :func:`.mean` (ineffective and imprecise, for sanity checks only)
* Probability density function (PDF) - :class:`.PDF`
* Cumulative distribution function (CDF) - :class:`.CDF`
* Quantile (percentile) function - :class:`.Quantile`

.. note::

    In many contexts, distribution functions approximated by Distimate
    should be called *empirical* distribution functions.
    They usually aggregate an empirical measure of a random sample.

    For example, many libraries implement
    an empirical cumulative distribution function (ECDF).
    Distimate calls that function CDF for brevity.


Each of the above functions can be either plotted as an object with ``.x`` and ``.y`` attributes,
or it can be called to approximate a function value at arbitrary point.

.. testcode::

    import distimate

    edges = [0, 10, 50, 100]

    cdf = distimate.CDF(edges, [4, 3, 1, 0, 2])
    print(cdf.x)
    print(cdf.y)

.. testoutput::

    [  0  10  50 100]
    [0.4 0.7 0.8 0.8]


The functions accept a number or a NumPy array-like.

.. testcode::

    print(cdf(-7))
    print(cdf(0))
    print(cdf(5))
    print(cdf(107))
    print(cdf([-7, 0, 5, 107]))

.. testoutput::

    0.0
    0.4
    0.55
    nan
    [0.   0.4  0.55  nan]


Functions are approximated from histograms.

- The first bucket is represented by the first edge.
- We assume that samples are uniformly distributed in inner buckets.
- Outliers in the last bucket cannot be approximated.

.. testcode::

    # The first bucket counts zeros.
    mean = distimate.mean(edges, [3, 0, 0, 0, 0])
    print(mean)

.. testoutput::

    0.0

.. testcode::

    # The midpoint of the (0, 10] bucket is 5.
    mean = distimate.mean(edges, [0, 7, 0, 0, 0])
    print(mean)

.. testoutput::

    5.0

.. testcode::

    # The last bucket cannot be approximated.
    mean = distimate.mean(edges, [0, 0, 0, 0, 13])
    print(mean)

.. testoutput::

    nan


The implementation intelligently handles various corner cases.
In the following example, a distribution median can be anything between 10 and 50.

.. testcode::

    quantile = distimate.Quantile(edges, [0, 5, 0, 5, 0])

    print(quantile.x, quantile.y)
    print(quantile(0.5))

.. testoutput::

    [0.  0.5 0.5 1. ] [  0.  10.  50. 100.]
    10.0

A plot will contain a vertical line,
but a function call returns the lowest of possible values, as stated in the method documentation.




Distributions
-------------

All approximations from histograms require histogram edges and values.
The :class:`.Distribution` class is a wrapper that holds both.
It provides methods for updating or combining distributions:

.. testcode::

    dist1 = distimate.Distribution(edges)
    dist1.add(7)
    print(dist1.to_histogram())

    dist2 = distimate.Distribution(edges)
    dist2.update([0, 1, 1])
    print(dist2.to_histogram())

    print("----------------")
    print((dist1 + dist2).to_histogram())

.. testoutput::

    [0. 1. 0. 0. 0.]
    [1. 2. 0. 0. 0.]
    ----------------
    [1. 3. 0. 0. 0.]


- The first histogram bucket counts items lesser than or equal to the left-most edge.
- The inner buckets count items between two edges.
  Intervals are left-open, the inner buckets count items
  greater than their left edge and lesser than or equal to their right edge.
- The last bucket counts items greater than the right-most edge.

.. note::

    The bucketing implemented in Distimate works best with non-negative metrics.

    - The left-most edge should be zero in most cases.
    - The right-most edge should be set to highest expected value.

    With this setup, the first bucket counts zeros and the last bucket counts outliers.


Optional weights are supported:

.. testcode::

    dist = distimate.Distribution(edges)
    dist.update([0, 7, 13], [1, 2, 3])
    print(dist.to_histogram())

.. testoutput::

    [1. 2. 3. 0. 0.]


It is common to define histogram edges once and reuse them between distributions.
The :class:`.DistributionType` class can remember the histogram edges.
It can be used as a factory for creating distributions:

.. testcode::

    dist_type = distimate.DistributionType([0, 10, 50, 100])
    print(dist_type.edges)

    dist = dist_type.from_samples([0, 7, 10, 107])
    print(dist.edges, dist.values)

.. testoutput::

    [  0  10  50 100]
    [  0  10  50 100] [1. 2. 0. 0. 1.]


Pandas integration
------------------

Consider that you load :class:`pandas.DataFrame` with histogram values:

.. testcode::

    import pandas as pd

    columns = ["color", "size", "hist0", "hist1", "hist2", "hist3", "hist4"]
    data = [
        (  "red", "M", 0, 1, 0, 0, 0),
        ( "blue", "L", 1, 2, 0, 0, 0),
        ( "blue", "M", 3, 2, 1, 0, 1),
    ]
    df = pd.DataFrame(data, columns=columns)
    print(df)

.. testoutput::

      color size  hist0  hist1  hist2  hist3  hist4
    0   red    M      0      1      0      0      0
    1  blue    L      1      2      0      0      0
    2  blue    M      3      2      1      0      1


The histogram data can be converted to :class:`pandas.Series`
with :class:`.Distribution` instances:

.. testcode::

    hist_columns = df.columns[2:]
    dists = pd.Series.dist.from_histogram(edges, df[hist_columns])
    print(dists)

.. testoutput::

    0    <Distribution: weight=1, mean=5.00>
    1    <Distribution: weight=3, mean=3.33>
    2     <Distribution: weight=7, mean=nan>
    dtype: object


We can replace histograms in the original DataFrame by the distributions:

.. testcode::

    df["qty"] = dists
    df.drop(columns=hist_columns, inplace=True)
    print(df)

.. testoutput::

      color size                                  qty
    0   red    M  <Distribution: weight=1, mean=5.00>
    1  blue    L  <Distribution: weight=3, mean=3.33>
    2  blue    M   <Distribution: weight=7, mean=nan>


The advantage of the new column is that we can use it with the ``dist`` accessor
to compute statistical functions for all DataFrame rows using a simple expression.

.. testcode::

    median = df["qty"].dist.quantile(0.5)
    print(median)

.. testoutput::

    0    5.0
    1    2.5
    2    2.5
    Name: qty_q50, dtype: float64


See :class:`.DistributionAccessor` for all methods available via the  ``dist`` accessor.


Series of :class:`Distribution` instances can be aggregated:

.. testcode::

    agg = df.groupby("color")["qty"].sum()
    print(agg)

.. testoutput::

    color
    blue    <Distribution: weight=10, mean=nan>
    red     <Distribution: weight=1, mean=5.00>
    Name: qty, dtype: object
