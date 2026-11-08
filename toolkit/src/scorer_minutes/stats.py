"""Small statistics helpers (standard library only, deterministic with a seed)."""
from __future__ import annotations

import math
import random
import statistics
from typing import Callable, Sequence

NORMAL = statistics.NormalDist()


def quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default)."""
    s = sorted(xs)
    if not s:
        raise ValueError("empty")
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def bootstrap_ci(xs: Sequence[float], stat: Callable[[Sequence[float]], float], n_boot: int = 2000,
                 level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of stat(xs), resampling the items of xs."""
    rng = random.Random(seed)
    n = len(xs)
    vals = sorted(stat([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    a = (1 - level) / 2
    return quantile(vals, a), quantile(vals, 1 - a)


def lstsq(rows: Sequence[Sequence[float]], y: Sequence[float]) -> list[float]:
    """Ordinary least squares via the normal equations (fine for a handful of well-scaled columns)."""
    k = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    aty = [sum(r[i] * yy for r, yy in zip(rows, y)) for i in range(k)]
    m = [row[:] + [b] for row, b in zip(ata, aty)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular design: a regressor is constant or collinear")
        m[col], m[piv] = m[piv], m[col]
        for r in range(k):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return [m[i][k] / m[i][i] for i in range(k)]


def r_squared(rows: Sequence[Sequence[float]], y: Sequence[float], coef: Sequence[float]) -> float:
    pred = [sum(c * x for c, x in zip(coef, r)) for r in rows]
    mean = sum(y) / len(y)
    ss_tot = sum((v - mean) ** 2 for v in y)
    ss_res = sum((v - p) ** 2 for v, p in zip(y, pred))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def runs_needed(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Runs per agent so a two-sided test on mean solved counts detects a true difference delta (equal run SD)."""
    if delta <= 0:
        raise ValueError("delta must be > 0")
    z = NORMAL.inv_cdf(1 - alpha / 2) + NORMAL.inv_cdf(power)
    return max(1, math.ceil(2 * (z * sd / delta) ** 2))


def p_wrong_order(delta: float, sd: float, runs: int = 1) -> float:
    """Chance that the better agent (true margin delta) shows the lower mean over `runs` runs each."""
    if sd <= 0:
        return 0.0
    return NORMAL.cdf(-delta / (sd * math.sqrt(2 / runs)))


def chi2_cdf(x: float, k: int) -> float:
    """Chi-square CDF with k degrees of freedom: regularized lower incomplete gamma P(k/2, x/2), series form."""
    if x <= 0:
        return 0.0
    a, z = k / 2, x / 2
    term = total = 1 / a
    n = 0
    while abs(term) > 1e-15 * abs(total) and n < 10000:
        n += 1
        term *= z / (a + n)
        total += term
    return min(1.0, math.exp(-z + a * math.log(z) - math.lgamma(a)) * total)


def chi2_quantile(p: float, k: int) -> float:
    lo, hi = 0.0, max(10.0, 10.0 * k)
    while chi2_cdf(hi, k) < p:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if chi2_cdf(mid, k) < p else (lo, mid)
    return (lo + hi) / 2


def sd_ci(sd: float, dof: int, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval of a (pooled) normal SD from its degrees of freedom."""
    a = (1 - level) / 2
    return (math.sqrt(dof * sd ** 2 / chi2_quantile(1 - a, dof)), math.sqrt(dof * sd ** 2 / chi2_quantile(a, dof)))
