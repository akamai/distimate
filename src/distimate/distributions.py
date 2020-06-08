# Copyright 2020 Akamai Technologies, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from distimate.stats import CDF, PDF, Quantile, mean


class Distribution:
    """
    Statistical distribution represented by its histogram.

    Provides an object interface on top of a histogram array.
    Supports distribution merging and comparison.
    Implements approximation of common statistical functions.

    :param edges: 1-D array-like, ordered histogram edges
    :param values: 1-D array-like, histogram, one item longer than *edges*
    """

    __slots__ = ("_edges", "_values")

    _dtype = np.float64

    def __init__(self, edges, values=None):
        self._edges = np.asarray(edges)
        size = len(self._edges) + 1
        if values is None:
            values = np.zeros(size, dtype=self._dtype)
        else:
            values = np.asarray(values, dtype=self._dtype)
            if values.ndim != 1:
                raise ValueError("Histogram must be 1-D array-like.")
            if len(values) != size:
                raise ValueError("Histogram must have len(edges) + 1 items.")
            if not np.all(values >= 0):
                raise ValueError("Histogram values must not be negative.")
        self._values = values

    def __repr__(self):
        name = type(self).__name__
        return f"<{name}: weight={self.weight:.0f}, mean={self.mean:.2f}>"

    def __eq__(self, other):
        """Return whether distribution histograms are equal."""
        if isinstance(other, Distribution):
            self._check_compatibility(other)
            return np.array_equal(self._values, other._values)
        return NotImplemented

    def __add__(self, other):
        """Combine this distribution with other distribution."""
        if isinstance(other, Distribution):
            self._check_compatibility(other)
            values = self._values + other._values
            return Distribution(self.edges, values)
        return NotImplemented

    def __iadd__(self, other):
        """Combine this distribution with other distribution inplace."""
        if isinstance(other, Distribution):
            self._check_compatibility(other)
            self._values += other._values
            return self
        return NotImplemented

    @property
    def edges(self):
        """
        Edges of the underlying histogram

        :return: :class: 1-D `numpy.array`, ordered histogram edges
        """
        return self._edges

    @property
    def values(self):
        """
        Values of the underlying histogram.

        :return: 1-D `numpy.array`, histogram values
        """
        return self._values

    @classmethod
    def from_samples(cls, edges, samples, weights=None):
        """
        Create a distribution from a list of values.

        :param edges: 1-D array-like, ordered histogram edges
        :param samples: 1-D array-like
        :param weights: optional scalar
            or 1-D array-like with same length as samples.
        :return: a new :class:`Distribution`
        """
        dist = cls(edges)
        dist.update(samples, weights)
        return dist

    @classmethod
    def from_histogram(cls, edges, histogram):
        """
        Create a distribution from a histogram.

        :param edges: 1-D array-like, ordered histogram edges
        :param histogram: 1-D array-like, one item longer than edges
        :return: a new :class:`Distribution`
        """
        return cls(edges, histogram)

    @classmethod
    def from_cumulative(cls, edges, cumulative):
        """
        Create a distribution from a cumulative histogram.

        :param edges: 1-D array-like, ordered histogram edges
        :param cumulative: 1-D array-like, one item longer than edges
        :return: a new :class:`Distribution`
        """
        values = np.diff(cumulative, prepend=0)
        return cls(edges, values)

    def to_histogram(self):
        """
        Return a histogram of this distribution as a NumPy array.

        :return: 1-D :class:`numpy.array`
        """
        return self._values.copy()

    def to_cumulative(self):
        """
        Return a cumulative histogram of this distribution as a NumPy array.

        :return: 1-D :class:`numpy.array`
        """
        return np.cumsum(self._values)

    def add(self, value, weight=None):
        """
        Add a new item to this distribution.

        :param value: item to add
        :param weight: optional item weight
        """
        if np.ndim(value) != 0:
            raise ValueError("Value must be a scalar.")
        if weight is None:
            weight = 1
        index = self._edges.searchsorted(value)
        self._values[index] += weight

    def update(self, values, weights=None):
        """
        Add multiple items to this distribution.

        :param values: items to add, 1-D array-like
        :param weights: optional scalar or 1-D array-like
            with same length as samples.
        """
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError("Values must be 1-D array-like.")
        if weights is None:
            weights = 1
        index = self._edges.searchsorted(values)
        # Cannot use self._hist[index] += weights because it does
        # not accumulate if index contains duplicate values.
        np.add.at(self._values, index, weights)

    @property
    def weight(self):
        """
        Return a total weight of samples in this distribution.

        :return: float number
        """
        return self._values.sum()

    @property
    def mean(self):
        """
        Estimate mean of this distribution.

        The approximated mean is for sanity checks only,
        it is ineffective and imprecise to estimate mean from a histogram.

        See :func:`.mean` for details.

        :return: float number
        """
        return mean(self._edges, self._values)

    @property
    def pdf(self):
        """
        Probability density function (PDF) of this distribution.

        See :class:`.PDF` for details.

        :return: a :class:`.PDF` instance
        """
        return PDF(self._edges, self._values)

    @property
    def cdf(self):
        """
        Cumulative distribution function (CDF) of this distribution.

        See :class:`.CDF` for details.

        :return: a :class:`.CDF` instance
        """
        return CDF(self._edges, self._values)

    @property
    def quantile(self):
        """
        Quantile function of this distribution.

        See :class:`.Quantile` for details.

        :return: a :class:`.Quantile` instance
        """
        return Quantile(self._edges, self._values)

    def _check_compatibility(self, dist):
        if not np.array_equal(dist._edges, self._edges):
            raise ValueError("Distributions have different edges.")
