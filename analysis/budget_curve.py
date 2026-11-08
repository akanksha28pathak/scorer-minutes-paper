"""Budget-response curve of the scorer-minutes paper (section 4.2): one agent (g31_v12e), the 55 train tasks, one setup
(RTX PRO 6000, 5 sessions), wall caps converted for 3.5, 7 and 10 scorer minutes, plus the earlier converted control run
on the same GPU type. Shows how many tasks a factor error of a given size costs.

  python3 analysis/budget_curve.py [--out docs/paper/aggregates/budget_curve.csv]

Reads results/kaggle_runs/<run>/run/**/rows.jsonl (solved, harness timeout) and docs/paper/aggregates/speed_table.csv
(measured factor). Scorer minutes = wall cap / measured factor. Prints the slope (tasks per scorer minute, least squares).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [  # (run, wall cap in minutes, label)
    ("gemma4-eval-g31-v12e-rtx-b35", 3.1, "3.5 scorer min"),
    ("gemma4-eval-g31-v12e-train55-ctl", 6.0, "7 scorer min (control run, table 4.2)"),
    ("gemma4-eval-g31-v12e-rtx-b7r", 6.3, "7 scorer min (repeat)"),
    ("gemma4-eval-g31-v12e-rtx-b10", 9.0, "10 scorer min"),
]


def rows(run: str) -> list[dict]:
    out = {}
    for f in glob.glob(str(ROOT / "results/kaggle_runs" / run / "run" / "**" / "rows.jsonl"), recursive=True):
        for line in open(f):
            if line.startswith("{"):
                r = json.loads(line)
                out[r["id"]] = r
    return list(out.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs/paper/aggregates/budget_curve.csv"))
    a = ap.parse_args()
    speed = {r["run"]: r for r in csv.DictReader(open(ROOT / "docs/paper/aggregates/speed_table.csv"))}
    out = []
    for run, cap, label in RUNS:
        rs, f = rows(run), float(speed[run]["factor"])
        out.append({"run": run, "label": label, "wall_cap_min": cap, "factor": round(f, 2),
                    "scorer_min": round(cap / f, 1), "tasks": len(rs), "solved": sum(bool(r["resolved"]) for r in rs),
                    "timed_out": sum("exceeded session timeout" in (r.get("error") or "") for r in rs)})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    xs, ys = [r["scorer_min"] for r in out], [r["solved"] for r in out]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    for r in out:
        print(r)
    print(f"slope {slope:.2f} tasks per scorer minute; a 10% factor error at 7 scorer minutes = {0.7 * slope:.2f} tasks")


if __name__ == "__main__":
    main()
