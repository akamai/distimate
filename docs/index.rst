

Distimate documentation
========================

Distimate can approximate and plot common statistical functions
that describe distributions of random variables (PDF, CDF, quantile).

Distributions are represented as histograms with constant bucket edges.
This is especially useful when working with large datasets
that can be aggregated to histograms at database level.

.. plot::

    import distimate
    import matplotlib.pyplot as plt

    edges = [0, 1, 2, 5, 10, 15, 20, 50]
    values = [291, 10, 143, 190, 155, 60, 90, 34, 27]
    dist = distimate.Distribution.from_histogram(edges, values)

    plt.title(f"x̃={dist.quantile(0.5):.2f}")
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.plot(dist.cdf.x, dist.cdf.y, label="CDF")
    plt.plot(dist.pdf.x, dist.pdf.y, label="PDF")
    plt.legend(loc="lower right")


Features:

* Histogram creation and merging
* Probability density function (PDF)
* Cumulative distribution function (CDF)
* Quantile (percentile) function
* Pandas integration.


Table of Contents
-----------------

.. toctree::

    install
    tutorial
    api
    develop
    faq


Indices and tables
..................

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
