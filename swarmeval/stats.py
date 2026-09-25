"""Small, dependency-free descriptive statistics.

Phase 1 reports means and spreads over repeated runs.  Confidence intervals
and paired comparisons are Phase 3 work (see ROADMAP.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: Sequence[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def median(xs: Sequence[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def percentile(xs: Sequence[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


@dataclass
class Summary:
    """Descriptive summary of one metric over a set of runs."""
    n: int
    mean: float
    std: float
    minimum: float
    maximum: float
    total: float

    def to_dict(self) -> Dict[str, float]:
        return {"n": self.n, "mean": self.mean, "std": self.std,
                "min": self.minimum, "max": self.maximum, "total": self.total}

    def fmt(self, unit: str = "", pct: bool = False, digits: int = 1) -> str:
        if pct:
            s = f"{self.mean*100:.{digits}f}%"
        else:
            s = f"{self.mean:.{digits}f}{unit}"
        if self.n > 1 and self.std > 0:
            s += f" ±{self.std*100:.{digits}f}" if pct else f" ±{self.std:.{digits}f}"
        return s


def summarize(values: Sequence[float]) -> Summary:
    values = [float(v) for v in values]
    if not values:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return Summary(len(values), mean(values), stdev(values), min(values), max(values), sum(values))
