
Tutorial
========

Distributions
-------------

Distimate stores distributions as histograms with constant edges.
The :class:`.DistributionType` class can remember the histogram edges.

.. testcode::

    from distimate import DistributionType

    dist_type = DistributionType([0, 10, 50, 100])
    print(dist_type.edges)

.. testoutput::

    [  0  10  50 100]


Once we defined the histogram edges, we can create a :class:`.Distribution` instance.
Each distribution instance stores a histogram with one more bucket than it has edges.

.. testcode::

    dist = dist_type.from_samples([0, 7, 10, 107])
    print(dist.to_histogram())

.. testoutput::

    [1. 2. 0. 0. 1.]

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


Distributions can be updated or combined:

.. testcode::

    dist1 = dist_type.empty()
    dist1.add(7)
    print(dist1.to_histogram())

    dist2 = dist_type.empty()
    dist2.update([0, 1, 1])
    print(dist2.to_histogram())

    print("----------------")
    print((dist1 + dist2).to_histogram())

.. testoutput::

    [0. 1. 0. 0. 0.]
    [1. 2. 0. 0. 0.]
    ----------------
    [1. 3. 0. 0. 0.]


Optional weights are supported:

.. testcode::

    dist = dist_type.from_samples([0, 7, 13], [1, 2, 3])
    print(dist.to_histogram())

.. testoutput::

    [1. 2. 3. 0. 0.]


Statistics
----------

:class:`.Distribution` instances implement common statistical functions.
All functions are approximated from underlying histograms.

- The first bucket is represented by the first edge.
- We assume that samples are uniformly distributed in inner buckets.
- Outliers in the last bucket cannot be approximated.

.. testcode::

    # The first bucket counts zeros.
    dist = dist_type.from_histogram([3, 0, 0, 0, 0])
    print(dist)

.. testoutput::

    <Distribution: size=3, mean=0.00>

.. testcode::

    # The midpoint of the (0, 10] bucket is 5.
    dist = dist_type.from_histogram([0, 7, 0, 0, 0])
    print(dist)

.. testoutput::

    <Distribution: size=7, mean=5.00>

.. testcode::

    # The last bucket cannot be approximated.
    dist = dist_type.from_histogram([0, 0, 0, 0, 13])
    print(dist)

.. testoutput::

    <Distribution: size=13, mean=nan>


The main feature of Distimate is the ability to estimate common statistical functions:

 - probability density function (:attr:`.Distribution.pdf`),
 - cumulative distribution function (:attr:`.Distribution.cdf`),
 - quantile (percentile) function (:attr:`.Distribution.quantile`).

Each of the above functions can be either plotted as an object with ``.x`` and ``.y`` attributes,
or it can be called to approximate a function value at arbitrary point.

.. testcode::

    dist = dist_type.from_histogram([4, 3, 1, 0, 2])
    print(dist.cdf.x)
    print(dist.cdf.y)

.. testoutput::

    [  0  10  50 100]
    [0.4 0.7 0.8 0.8]


The functions accept a number or a NumPy array-like.

.. testcode::

    print(dist.cdf(-7))
    print(dist.cdf(0))
    print(dist.cdf(5))
    print(dist.cdf(107))
    print(dist.cdf([-7, 0, 5, 107]))

.. testoutput::

    0.0
    0.4
    0.55
    nan
    [0.   0.4  0.55  nan]


The implementation intelligently handles various corner cases.
In the following example, a distribution median can be anything between 10 and 50.

.. testcode::

    dist = dist_type.from_histogram([0, 5, 0, 5, 0])

    print(dist.quantile.x, dist.quantile.y)
    print(dist.quantile(0.5))

.. testoutput::

    [0.  0.5 0.5 1. ] [  0.  10.  50. 100.]
    10.0

A plot will contain a vertical line,
but a function call returns the lowest of possible values, as stated in the method documentation.


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
    dists = pd.Series.dist.from_histogram(dist_type, df[hist_columns])
    print(dists)

.. testoutput::

    0    <Distribution: size=1, mean=5.00>
    1    <Distribution: size=3, mean=3.33>
    2     <Distribution: size=7, mean=nan>
    dtype: object


We can replace histograms in the original DataFrame by the distributions:

.. testcode::

    df["qty"] = dists
    df.drop(columns=hist_columns, inplace=True)
    print(df)

.. testoutput::

      color size                                qty
    0   red    M  <Distribution: size=1, mean=5.00>
    1  blue    L  <Distribution: size=3, mean=3.33>
    2  blue    M   <Distribution: size=7, mean=nan>


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
    blue    <Distribution: size=10, mean=nan>
    red     <Distribution: size=1, mean=5.00>
    Name: qty, dtype: object
