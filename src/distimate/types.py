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

from distimate.distributions import Distribution


class DistributionType:
    """
    Factory for creating distributions with constant histogram edges.

    :param edges: 1-D array-like, ordered histogram edges
    """

    __slots__ = ("_edges",)

    _dist_cls = Distribution

    def __init__(self, edges):
        self._edges = np.asarray(edges)

    @property
    def edges(self):
        """
        Edges of the underlying histogram

        :return: :class: 1-D `numpy.array`, ordered histogram edges
        """
        return self._edges

    def empty(self):
        """
        Create an empty distribution.

        :return: a new :class:`Distribution`
        """
        return self._dist_cls(self._edges)

    def from_samples(self, samples, weights=None):
        """
        Create a distribution from a list of values.

        :param samples: 1-D array-like
        :param weights: optional 1-D array-like
        :return: a new :class:`Distribution`
        """
        return self._dist_cls.from_samples(self._edges, samples, weights)

    def from_histogram(self, histogram):
        """
        Create a distribution from a histogram.

        :param histogram: 1-D array-like
        :return: a new :class:`Distribution`
        """
        return self._dist_cls.from_histogram(self._edges, histogram)

    def from_cumulative(self, cumulative):
        """
        Create a distribution from a cumulative histogram.

        :param cumulative: 1-D array-like
        :return: a new :class:`Distribution`
        """
        return self._dist_cls.from_cumulative(self._edges, cumulative)
