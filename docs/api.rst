
API
===

Distributions
-------------

.. module:: distimate.distributions

.. autoclass:: Distribution
    :members:
    :special-members: __eq__, __add__


.. module:: distimate.types

.. autoclass:: DistributionType
    :members:


Statistics
----------

.. module:: distimate.stats

.. autofunction:: make_pdf()
.. autofunction:: make_cdf()
.. autofunction:: make_quantile()

.. autoclass:: StatsFunction()
    :members:
    :special-members: __call__


Pandas integration
------------------

    .. module:: distimate.pandasext

    .. autoclass:: DistributionAccessor
        :members:
