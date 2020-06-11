
# Distimate - Distributions visualized

Distimate approximates and plots common statistical functions from histograms.

Distimate can aggregate empirical distributions of random variables.
The distributions are represented as histograms with user-defined bucket edges.
This is especially useful when working with large datasets
that can be aggregated to histograms at database level.

```python
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
```

Features:

* Histogram creation and merging
* Probability density function (PDF)
* Cumulative distribution function (CDF or ECDF)
* Quantile (percentile) function
* Pandas integration.

Distimate is hosted at [GitHub](http://github.com/akamai/distimate) and
it can be installed from [PyPI](https://pypi.org/project/distimate/).


## Documentation

Distimate documentation from the `docs/` directory can be read
online at [Read the Docs](https://distimate.readthedocs.io/).


## License

```
Copyright 2020 Akamai Technologies, Inc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```


## Contributing

```
By submitting a contribution (the “Contribution”) to this project,
and for good and valuable consideration, the receipt and sufficiency of which
are hereby acknowledged, you (the “Assignor”) irrevocably convey, transfer,
and assign the Contribution to the owner of the repository (the “Assignee”),
and the Assignee hereby accepts, all of your right, title, and interest in and
to the Contribution along with all associated copyrights, copyright
registrations, and/or applications for registration and all issuances,
extensions and renewals thereof (collectively, the “Assigned Copyrights”).
You also assign all of your rights of any kind whatsoever accruing under
the Assigned Copyrights provided by applicable law of any jurisdiction,
by international treaties and conventions and otherwise throughout the world.
```
