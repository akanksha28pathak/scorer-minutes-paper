"""Run-to-run noise of agent scores, and what it means for comparisons and single leaderboard draws.

Input: a CSV in one of two shapes.
  long (preferred):  config, split, run, task, solved      (solved = 0/1, one row per task per run)
  summary:           config, split, run, solved, n         (solved = count of n tasks)

"config" is the agent setup (same files, same budget), "run" one full pass. Repeated runs of one config on one split
measure run noise: sampling temperature, scheduling and timing make the same agent solve different tasks.

Reported per split:
* per config: solved counts per run, mean, sample SD;
* pooled within-config SD (over every config with >= 2 runs) with its degrees of freedom and chi-square 95% CI, also
  pooled over all splits of the same size;
* with task-level data, the SD a run would have if each task were an independent coin with the task's observed solve
  rate: sqrt(sum p_i (1 - p_i)) (k/(k-1) bias-corrected), which uses every task, not just the run totals;
* runs per agent needed to detect a true difference of D tasks (alpha 0.05 two-sided, power 0.8);
* chance that a single draw (or the mean of r draws) ranks two agents in the wrong order.
"""
from __future__ import annotations

import csv
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .stats import NORMAL, p_wrong_order, quantile, runs_needed, sd_ci


@dataclass
class Run:
    config: str
    split: str
    run: str
    solved: int
    n: int
    tasks: dict[str, int] | None = None


@dataclass
class Table:
    runs: list[Run] = field(default_factory=list)

    def groups(self) -> dict[tuple[str, str], list[Run]]:
        g: dict[tuple[str, str], list[Run]] = defaultdict(list)
        for r in self.runs:
            g[(r.config, r.split)].append(r)
        return dict(g)

    def splits(self) -> list[str]:
        return sorted({r.split for r in self.runs})


def _truthy(v: str) -> int:
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y", "solved", "pass", "passed", "✅"):
        return 1
    if v in ("0", "false", "no", "n", "unsolved", "fail", "failed", "❌", ""):
        return 0
    raise ValueError(f"cannot read solved value {v!r}")


def from_rows(rows: Iterable[dict]) -> Table:
    rows = list(rows)
    if not rows:
        return Table()
    cols = set(rows[0])
    if {"config", "split", "run", "task", "solved"} <= cols:
        acc: dict[tuple, dict[str, int]] = defaultdict(dict)
        for r in rows:
            acc[(r["config"], r["split"], r["run"])][r["task"]] = _truthy(str(r["solved"]))
        return Table([Run(c, s, run, sum(t.values()), len(t), dict(t)) for (c, s, run), t in acc.items()])
    if {"config", "split", "run", "solved", "n"} <= cols:
        return Table([Run(r["config"], r["split"], r["run"], int(r["solved"]), int(r["n"])) for r in rows])
    raise ValueError("CSV needs columns config,split,run,task,solved (long) or config,split,run,solved,n (summary)")


def read_csv(paths: Iterable[str | Path]) -> Table:
    t = Table()
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            t.runs += from_rows(csv.DictReader(f)).runs
    return t


def _sd(xs: list[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) >= 2 else None


def task_level_sd(runs: list[Run]) -> float | None:
    """SD of a run total if each task is an independent Bernoulli(p_i), p_i = the task's solve rate over the runs."""
    if len(runs) < 2 or any(r.tasks is None for r in runs):
        return None
    tasks = set.intersection(*(set(r.tasks) for r in runs))
    k = len(runs)
    var = 0.0
    for t in tasks:
        p = sum(r.tasks[t] for r in runs) / k
        var += p * (1 - p) * k / (k - 1)
    return math.sqrt(var)


@dataclass
class ConfigSummary:
    config: str
    split: str
    n: int
    solved: list[int]
    mean: float
    sd: float | None
    task_sd: float | None
    flaky_tasks: int | None


def summarize(table: Table) -> list[ConfigSummary]:
    out = []
    for (c, s), runs in sorted(table.groups().items()):
        totals = [r.solved for r in runs]
        flaky = None
        if all(r.tasks is not None for r in runs):
            common = set.intersection(*(set(r.tasks) for r in runs))
            flaky = sum(1 for t in common if 0 < sum(r.tasks[t] for r in runs) < len(runs))
        out.append(ConfigSummary(c, s, runs[0].n, totals, statistics.mean(totals), _sd(totals),
                                 task_level_sd(runs), flaky))
    return out


@dataclass
class Pooled:
    split: str
    n: int
    sd: float | None
    dof: int
    configs: int
    task_sd: float | None  # pooled task-level SD (same configs)

    @property
    def sd_ci95(self) -> tuple[float, float] | None:
        return sd_ci(self.sd, self.dof) if self.sd and self.dof else None


def pooled(table: Table, split: str | list[str]) -> Pooled:
    """Pooled run noise over the configs with >= 2 runs on one split, or on several splits of the same task count."""
    want = [split] if isinstance(split, str) else list(split)
    ss, dof, cfgs, tvar, tcfg, n = 0.0, 0, 0, 0.0, 0, None
    for (c, s), runs in table.groups().items():
        if s not in want or len(runs) < 2:
            continue
        n = runs[0].n
        totals = [r.solved for r in runs]
        m = statistics.mean(totals)
        ss += sum((x - m) ** 2 for x in totals)
        dof += len(totals) - 1
        cfgs += 1
        tsd = task_level_sd(runs)
        if tsd is not None:
            tvar += tsd ** 2
            tcfg += 1
    return Pooled(split if isinstance(split, str) else "+".join(want), n or 0, math.sqrt(ss / dof) if dof else None,
                  dof, cfgs,
                  math.sqrt(tvar / tcfg) if tcfg else None)


@dataclass
class Comparison:
    a: str
    b: str
    split: str
    runs_a: int
    runs_b: int
    mean_diff: float
    run_noise_se: float | None
    run_noise_ci95: tuple[float, float] | None
    p_value: float | None
    task_bootstrap_ci95: tuple[float, float] | None


def compare(table: Table, a: str, b: str, split: str, sd: float | None = None, n_boot: int = 5000,
            seed: int = 0) -> Comparison:
    g = table.groups()
    ra, rb = g.get((a, split)), g.get((b, split))
    if not ra or not rb:
        raise KeyError(f"need runs of both {a!r} and {b!r} on {split!r}")
    diff = statistics.mean(r.solved for r in ra) - statistics.mean(r.solved for r in rb)
    sd = sd if sd is not None else pooled(table, split).sd
    se = ci = p = None
    if sd:
        se = sd * math.sqrt(1 / len(ra) + 1 / len(rb))
        ci = (diff - 1.96 * se, diff + 1.96 * se)
        p = 2 * NORMAL.cdf(-abs(diff) / se)
    boot = None
    if all(r.tasks is not None for r in ra + rb):
        tasks = sorted(set.intersection(*(set(r.tasks) for r in ra + rb)))
        d = [sum(r.tasks[t] for r in ra) / len(ra) - sum(r.tasks[t] for r in rb) / len(rb) for t in tasks]
        rng = random.Random(seed)
        scale = ra[0].n / len(tasks) if tasks else 1
        vals = [sum(d[rng.randrange(len(d))] for _ in d) * scale for _ in range(n_boot)]
        boot = (quantile(vals, 0.025), quantile(vals, 0.975))
    return Comparison(a, b, split, len(ra), len(rb), diff, se, ci, p, boot)


def draw_table(sd: float, deltas=(1, 2, 3, 4, 5), runs=(1, 2, 3, 5)) -> list[dict]:
    """Per true margin D: chance of the wrong order with r runs each, and runs needed for 80% power."""
    return [{"delta": d, **{f"p_wrong_r{r}": round(p_wrong_order(d, sd, r), 3) for r in runs},
             "runs_needed": runs_needed(sd, d)} for d in deltas]


def single_draw_interval(score: float, sd: float, level: float = 0.95) -> tuple[float, float]:
    z = NORMAL.inv_cdf(0.5 + level / 2)
    return score - z * sd, score + z * sd


def report_markdown(table: Table, deltas=(1, 2, 3, 4, 5)) -> str:
    lines = ["## Runs per config", "",
             "| Split | Config | n | Runs | Solved per run | Mean | SD | Task-level SD | Flaky tasks |",
             "|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.2f}"
    for s in summarize(table):
        lines.append(f"| {s.split} | {s.config} | {s.n} | {len(s.solved)} | {', '.join(map(str, s.solved))} | "
                     f"{s.mean:.2f} | {f(s.sd)} | {f(s.task_sd)} | {'-' if s.flaky_tasks is None else s.flaky_tasks} |")
    lines += ["", "## Pooled run noise per split", "",
              "| Split | n | Configs with >= 2 runs | dof | Pooled SD (tasks) | 95% CI | Pooled task-level SD |",
              "|---|---|---|---|---|---|---|"]
    pools = [pooled(table, sp) for sp in table.splits()]
    by_n: dict[int, list[str]] = defaultdict(list)
    for p in pools:
        if p.dof:
            by_n[p.n].append(p.split)
    combined = [pooled(table, sps) for sps in by_n.values() if len(sps) > 1]
    for p in pools + combined:
        ci = p.sd_ci95
        lines.append(f"| {p.split} | {p.n} | {p.configs} | {p.dof} | {f(p.sd)} | "
                     f"{'-' if ci is None else f'{ci[0]:.2f}-{ci[1]:.2f}'} | {f(p.task_sd)} |")
    for p in pools:
        cands = {k: v for k, v in (("pooled run SD", p.sd), ("task-level SD", p.task_sd)) if v}
        if not cands:
            continue
        which, sd = max(cands.items(), key=lambda kv: kv[1])  # the larger estimate: conservative
        lines += ["", f"## {p.split}: reading comparisons (SD {sd:.2f} tasks of {p.n}, {which}, the larger estimate)",
                  "",
                  "| True margin (tasks) | P(wrong order), 1 run each | 2 runs | 3 runs | 5 runs | Runs each for 80% power |",
                  "|---|---|---|---|---|---|"]
        for r in draw_table(sd, deltas):
            lines.append(f"| {r['delta']} | {r['p_wrong_r1']:.2f} | {r['p_wrong_r2']:.2f} | {r['p_wrong_r3']:.2f} | "
                         f"{r['p_wrong_r5']:.2f} | {r['runs_needed']} |")
    return "\n".join(lines) + "\n"
