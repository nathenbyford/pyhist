"""Core histogram-valued data structures and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Histogram:
    """Histogram-valued observation with bin edges and masses.

    The representation aligns with distributional symbolic variables: each observation
    is a histogram with contiguous bins and probability masses summing to 1.
    """

    edges: Sequence[float]
    masses: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.edges) < 2:
            raise ValueError("Histogram requires at least two bin edges.")
        if len(self.masses) != len(self.edges) - 1:
            raise ValueError("Masses length must be one less than edges length.")
        if any(self.edges[i] >= self.edges[i + 1] for i in range(len(self.edges) - 1)):
            raise ValueError("Edges must be strictly increasing.")
        if any(m < 0 for m in self.masses):
            raise ValueError("Masses must be non-negative.")
        total = sum(self.masses)
        if total <= 0:
            raise ValueError("Total mass must be positive.")
        if abs(total - 1.0) > 1e-9:
            object.__setattr__(self, "masses", [m / total for m in self.masses])

    @property
    def bins(self) -> List[tuple[float, float]]:
        return list(zip(self.edges[:-1], self.edges[1:]))

    @property
    def midpoints(self) -> List[float]:
        return [(low + high) / 2.0 for low, high in self.bins]

    def mean(self) -> float:
        return sum(m * midpoint for m, midpoint in zip(self.masses, self.midpoints))

    def variance(self) -> float:
        mean = self.mean()
        return sum(m * (midpoint - mean) ** 2 for m, midpoint in zip(self.masses, self.midpoints))

    def cdf(self) -> List[float]:
        cumulative = []
        running = 0.0
        for mass in self.masses:
            running += mass
            cumulative.append(running)
        return cumulative

    def quantile(self, q: float) -> float:
        if not 0 <= q <= 1:
            raise ValueError("Quantile must be between 0 and 1.")
        cumulative = 0.0
        for (low, high), mass in zip(self.bins, self.masses):
            if mass == 0:
                continue
            next_cum = cumulative + mass
            if q <= next_cum:
                fraction = (q - cumulative) / mass
                return low + fraction * (high - low)
            cumulative = next_cum
        return self.edges[-1]

    def support(self) -> tuple[float, float]:
        return (self.edges[0], self.edges[-1])

    def to_piecewise_linear_cdf(self) -> List[tuple[float, float]]:
        """Return (x, cdf(x)) points at bin edges for linear interpolation."""
        points = [(self.edges[0], 0.0)]
        cumulative = 0.0
        for edge, mass in zip(self.edges[1:], self.masses):
            cumulative += mass
            points.append((edge, cumulative))
        return points


def wasserstein_distance(h1: Histogram, h2: Histogram) -> float:
    """Compute the 2-Wasserstein distance between two histograms.

    Implements the squared L2 distance between quantile functions approximated
    on a fine grid, consistent with Irpino & Verde's approach for histogram data.
    """

    grid = _merge_edges(h1, h2)
    if len(grid) < 2:
        return 0.0
    distance_sq = 0.0
    for i in range(len(grid) - 1):
        left = grid[i]
        right = grid[i + 1]
        mid_q = (left + right) / 2.0
        q1 = h1.quantile(mid_q)
        q2 = h2.quantile(mid_q)
        distance_sq += (q1 - q2) ** 2 * (right - left)
    return distance_sq ** 0.5


def barycenter(histograms: Iterable[Histogram]) -> Histogram:
    """Compute the Wasserstein barycenter (Fréchet mean) of histogram-valued data."""

    hist_list = list(histograms)
    if not hist_list:
        raise ValueError("At least one histogram is required.")
    grid = _merge_edges(*hist_list)
    if len(grid) < 2:
        raise ValueError("Invalid grid for barycenter.")
    quantiles_at_grid = [
        sum(h.quantile(p) for h in hist_list) / len(hist_list) for p in grid
    ]
    masses = [grid[i + 1] - grid[i] for i in range(len(grid) - 1)]
    return Histogram(edges=quantiles_at_grid, masses=masses)


def _merge_edges(*histograms: Histogram) -> List[float]:
    points = sorted({0.0, 1.0})
    for hist in histograms:
        points.extend(hist.cdf())
    unique = sorted(set(points))
    if unique[0] != 0.0:
        unique.insert(0, 0.0)
    if unique[-1] != 1.0:
        unique.append(1.0)
    return unique
