"""Speed factor of a run against the scorer, and the wall-clock cap that equals a scorer-minute budget.

    factor f = scorer tok/s / median session tok/s       (f > 1: this setup is slower than the scorer)
    scorer minutes = wall minutes / f
    wall cap = scorer-minute cap x f

Session tok/s = completion tokens / time inside model calls (prefill included: it is the throughput an agent actually
gets per second of model time; `timing` separates prefill from decode). The median over sessions is used because a
few stalled or very short sessions skew the mean.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .logs import Session
from .stats import bootstrap_ci, quantile


@dataclass
class Calibration:
    sessions: int
    median_tok_s: float
    p25_tok_s: float
    p75_tok_s: float
    scorer_tok_s: float
    factor: float
    factor_ci95: tuple[float, float]
    cap_scorer_min: float | None
    cap_wall_min: float | None

    def as_dict(self) -> dict:
        d = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}
        d["factor_ci95"] = [round(x, 3) for x in self.factor_ci95]
        return d


def session_rates(sessions: list[Session]) -> list[float]:
    return [s.tok_s for s in sessions if s.tok_s is not None]


def reference_tok_s(sessions: list[Session]) -> float:
    """Scorer speed measured from a reference run's own logs (median session tok/s)."""
    rates = session_rates(sessions)
    if not rates:
        raise ValueError("reference run has no usable sessions")
    return statistics.median(rates)


def calibrate(sessions: list[Session], scorer_tok_s: float, cap_scorer_min: float | None = None,
              n_boot: int = 2000, seed: int = 0) -> Calibration:
    rates = session_rates(sessions)
    if not rates:
        raise ValueError("no sessions with model time")
    med = statistics.median(rates)
    lo, hi = bootstrap_ci(rates, statistics.median, n_boot=n_boot, seed=seed)
    factor = scorer_tok_s / med
    return Calibration(
        sessions=len(rates), median_tok_s=med, p25_tok_s=quantile(rates, 0.25), p75_tok_s=quantile(rates, 0.75),
        scorer_tok_s=scorer_tok_s, factor=factor, factor_ci95=(scorer_tok_s / hi, scorer_tok_s / lo),
        cap_scorer_min=cap_scorer_min,
        cap_wall_min=None if cap_scorer_min is None else cap_scorer_min * factor)


def session_rows(sessions: list[Session], factor: float, cap_scorer_min: float | None = None) -> list[dict]:
    rows = []
    for s in sessions:
        wall = s.wall_seconds / 60
        rows.append({
            "session": "/".join(map(str, s.key)), "calls": len(s.calls), "gen_tokens": s.gen_tokens,
            "tok_s": None if s.tok_s is None else round(s.tok_s, 2), "wall_min": round(wall, 3),
            "model_share": round(s.model_seconds / s.wall_seconds, 4) if s.wall_seconds > 0 else None,
            "scorer_min": round(wall / factor, 3),
            "at_cap": None if cap_scorer_min is None else wall / factor >= 0.97 * cap_scorer_min,
        })
    return rows


def ceil_minutes(x: float) -> int:
    return math.ceil(x - 1e-9)
