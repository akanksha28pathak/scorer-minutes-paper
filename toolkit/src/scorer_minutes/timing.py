"""Where a session's time goes, and whether one speed factor is enough (the decode-bound assumption).

1. Model vs other time per session: other = wall - time inside model calls (tools, sandbox, harness).
2. Per-call cost model, least squares over all calls (tokens in thousands):

       seconds = c + a * new_prompt_k + b * gen_k + d * gen_k * context_k

   c = per-call overhead, a = prefill cost, b = decode cost at empty context, d = how decode slows as the context grows
   (attention over a longer KV cache). Prefill and decode are not timed separately in non-streaming logs; the fit
   splits them. Calls from parallel sessions share the GPU, so the fit is an average over the run's load.
3. Model-free check: completion tok/s per context-length bin, and early vs late calls of each session.

One factor is enough when model time dominates, prefill is a small share, and tok/s changes little from early to late
calls. Sessions failing a threshold are flagged.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .logs import Session
from .stats import lstsq, r_squared


@dataclass
class Fit:
    terms: list[str]
    coef: dict[str, float]
    r2: float
    calls: int

    def decode_tok_s(self, context_k: float) -> float | None:
        b = self.coef.get("gen_k", 0.0) + self.coef.get("gen_x_ctx_k", 0.0) * context_k
        return 1000 / b if b > 0 else None

    @property
    def prefill_tok_s(self) -> float | None:
        a = self.coef.get("new_prompt_k")
        return 1000 / a if a and a > 0 else None


def fit_calls(sessions: list[Session]) -> Fit | None:
    calls = [c for s in sessions for c in s.calls if c.seconds > 0]
    with_prompt = [c for c in calls if c.new_prompt_tokens is not None and c.prompt_tokens is not None]
    if len(with_prompt) >= max(20, 0.8 * len(calls)):
        terms = ["overhead", "new_prompt_k", "gen_k", "gen_x_ctx_k"]
        rows = [[1.0, c.new_prompt_tokens / 1000, c.completion_tokens / 1000,
                 c.completion_tokens / 1000 * c.prompt_tokens / 1000] for c in with_prompt]
        used = with_prompt
    else:
        terms = ["overhead", "gen_k"]
        rows = [[1.0, c.completion_tokens / 1000] for c in calls]
        used = calls
    if len(used) < len(terms) + 2:
        return None
    y = [c.seconds for c in used]
    try:
        coef = lstsq(rows, y)
    except ValueError:
        return None
    return Fit(terms, dict(zip(terms, coef)), r_squared(rows, y, coef), len(used))


def split_seconds(fit: Fit, s: Session) -> dict[str, float]:
    """Fitted overhead / prefill / decode seconds of one session (clipped at 0)."""
    out = {"overhead": 0.0, "prefill": 0.0, "decode": 0.0}
    for c in s.calls:
        out["overhead"] += max(fit.coef["overhead"], 0.0)
        if "new_prompt_k" in fit.coef and c.new_prompt_tokens is not None:
            out["prefill"] += max(fit.coef["new_prompt_k"], 0.0) * c.new_prompt_tokens / 1000
        dec = fit.coef["gen_k"] * c.completion_tokens / 1000
        if "gen_x_ctx_k" in fit.coef and c.prompt_tokens is not None:
            dec += fit.coef["gen_x_ctx_k"] * c.completion_tokens / 1000 * c.prompt_tokens / 1000
        out["decode"] += max(dec, 0.0)
    return out


def context_bins(sessions: list[Session], edges_k=(0, 8, 16, 24, 32, 1e9)) -> list[dict]:
    rows = []
    for lo, hi in zip(edges_k, edges_k[1:]):
        cs = [c for s in sessions for c in s.calls
              if c.prompt_tokens is not None and lo * 1000 <= c.prompt_tokens < hi * 1000 and c.seconds > 0]
        t = sum(c.seconds for c in cs)
        rows.append({"context_k": f"{lo:g}-{hi:g}" if hi < 1e9 else f">={lo:g}", "calls": len(cs),
                     "tok_s": round(sum(c.completion_tokens for c in cs) / t, 2) if t > 0 else None})
    return rows


def early_late(sessions: list[Session]) -> dict:
    """Median over sessions of tok/s in the first vs last third of each session's calls (sessions with >= 3 calls)."""
    early, late = [], []
    for s in sessions:
        n = len(s.calls)
        if n < 3:
            continue
        k = n // 3
        for part, out in ((s.calls[:k], early), (s.calls[-k:], late)):
            t = sum(c.seconds for c in part)
            if t > 0:
                out.append(sum(c.completion_tokens for c in part) / t)
    if not early or not late:
        return {"sessions": 0}
    e, l = statistics.median(early), statistics.median(late)
    return {"sessions": min(len(early), len(late)), "early_tok_s": round(e, 2), "late_tok_s": round(l, 2),
            "late_over_early": round(l / e, 3)}


@dataclass
class TimingReport:
    sessions: int
    model_share_median: float | None
    other_seconds_median: float | None
    fit: Fit | None
    shares: dict = field(default_factory=dict)
    context: list = field(default_factory=list)
    drift: dict = field(default_factory=dict)
    flagged: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {"sessions": self.sessions, "model_share_median": self.model_share_median,
             "other_seconds_median": self.other_seconds_median, "shares_of_model_time": self.shares,
             "tok_s_by_context": self.context, "early_vs_late": self.drift, "flagged_sessions": self.flagged}
        if self.fit:
            f = self.fit
            d["fit"] = {"calls": f.calls, "r2": round(f.r2, 3),
                        "coef": {k: round(v, 5) for k, v in f.coef.items()},
                        "prefill_tok_s": None if f.prefill_tok_s is None else round(f.prefill_tok_s, 1),
                        "decode_tok_s_at_context_k": {str(k): (None if f.decode_tok_s(k) is None
                                                               else round(f.decode_tok_s(k), 2))
                                                      for k in (0, 8, 16, 24, 32)}}
        return d


def timing_report(sessions: list[Session], min_model_share: float = 0.9,
                  max_prefill_share: float = 0.2) -> TimingReport:
    usable = [s for s in sessions if s.wall_seconds > 0]
    shares = [s.model_seconds / s.wall_seconds for s in usable]
    other = [s.wall_seconds - s.model_seconds for s in usable]
    fit = fit_calls(usable)
    rep = TimingReport(len(usable), round(statistics.median(shares), 4) if shares else None,
                       round(statistics.median(other), 2) if other else None, fit,
                       context=context_bins(usable), drift=early_late(usable))
    if fit:
        tot = {"overhead": 0.0, "prefill": 0.0, "decode": 0.0}
        for s in usable:
            for k, v in split_seconds(fit, s).items():
                tot[k] += v
        model_t = sum(c.seconds for s in usable for c in s.calls)
        rep.shares = {k: round(v / model_t, 4) for k, v in tot.items()} if model_t else {}
    for s, share in zip(usable, shares):
        reasons = []
        if share < min_model_share:
            reasons.append(f"model share {share:.2f} < {min_model_share}")
        if fit and "new_prompt_k" in fit.coef:
            t = sum(c.seconds for c in s.calls)
            pre = split_seconds(fit, s)["prefill"] / t if t else 0
            if pre > max_prefill_share:
                reasons.append(f"prefill share {pre:.2f} > {max_prefill_share}")
        if reasons:
            rep.flagged.append({"session": "/".join(map(str, s.key)), "reasons": reasons})
    return rep
