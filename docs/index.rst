

Distimate documentation
========================

Distimate allows you to analyze arbitrary large populations using constant memory.

Internally, distributions are approximated using histograms with predefined bucket edges.
This library can estimate common statistical functions (PDF, CDF, quantile) from the histograms.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from distimate import DistributionType

    edges = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    dist_type = DistributionType(edges)

    dist = dist_type.from_samples(np.random.lognormal(size=10**6))
    plt.plot(dist.cdf.x, dist.cdf.y)


.. toctree::

    install
    intro
    api




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
