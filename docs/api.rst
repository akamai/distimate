
API
===

Statistical functions
---------------------

.. module:: distimate.stats

.. autofunction:: mean()

.. autoclass:: PDF()
    :members:
    :inherited-members:
    :special-members: __call__

.. autoclass:: CDF()
    :members:
    :inherited-members:
    :special-members: __call__

.. autoclass:: Quantile()
    :members:
    :inherited-members:
    :special-members: __call__


Distributions
-------------

.. module:: distimate.distributions

.. autoclass:: Distribution
    :members:
    :special-members: __eq__, __add__


.. module:: distimate.types

.. autoclass:: DistributionType
    :members:


Pandas integration
------------------

    .. module:: distimate.pandasext

    .. autoclass:: DistributionAccessor
        :members:
