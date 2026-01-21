# pyhist

`pyhist` is a lightweight Python library for histogram-valued data analysis, inspired by
Irpino & Verde (2015) *Basic Statistics for distributional symbolic variables* and the
R package [HistDAWass](https://cran.r-project.org/web/packages/HistDAWass/index.html).

## Features

- Histogram representation with normalized bin masses.
- Basic statistics for histogram-valued data (mean, variance, quantiles).
- Wasserstein-2 distance for histogram-valued observations.
- Empirical barycenter (Fréchet mean) of histogram-valued samples.

## Quick start

```python
from pyhist import Histogram, wasserstein_distance, barycenter

h1 = Histogram(edges=[0, 1, 2], masses=[0.3, 0.7])
h2 = Histogram(edges=[0, 1, 2], masses=[0.6, 0.4])

print(h1.mean())
print(wasserstein_distance(h1, h2))

center = barycenter([h1, h2])
print(center.mean())
```

## Notes

This library follows the histogram-valued data modeling and Wasserstein-based metrics
proposed for distributional symbolic variables. It provides building blocks rather than
an exhaustive port of HistDAWass.
