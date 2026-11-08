"""Section 4.4 of the scorer-minutes paper: mean session wall time with and without LoRA adapter B (same agent, same
held-out tasks, same RTX PRO 6000 setup, two runs each), and the projected total time scaled from the base agent.

  python3 analysis/lora_sessions.py [--base-hours 10.3 10.8] [--out docs/paper/aggregates/lora_sessions.csv]

Reads results/kaggle_runs/<run>/run/**/rows.jsonl (duration per session). --base-hours is the base agent's projected
scorer time on the hidden set (docs/DECISIONS_31B.md); the adapter's projection is that range times the session-time
ratio. Counts and seconds only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"base (g31_v15)": ["gemma4-eval-g31-v15-val55-p1", "gemma4-eval-g31-v15-val55-p2"],
          "adapter B (g31_v15_sft)": ["gemma4-eval-g31-v15-sft-val55-p1", "gemma4-eval-g31-v15-sft-val55-p2"]}


def durations(runs: list[str]) -> list[float]:
    out = []
    for run in runs:
        for f in glob.glob(str(ROOT / "results/kaggle_runs" / run / "run" / "**" / "rows.jsonl"), recursive=True):
            out += [(lambda r: r.get("duration") or r.get("duration_s"))(json.loads(line)) for line in open(f) if line.startswith("{")]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-hours", nargs=2, type=float, default=[10.3, 10.8])
    ap.add_argument("--out", default=str(ROOT / "docs/paper/aggregates/lora_sessions.csv"))
    a = ap.parse_args()
    rows = []
    for name, runs in GROUPS.items():
        d = durations(runs)
        rows.append({"setup": name, "runs": "+".join(runs), "sessions": len(d), "mean_s": round(st.mean(d), 1),
                     "median_s": round(st.median(d), 1)})
    ratio = rows[1]["mean_s"] / rows[0]["mean_s"]
    for r in rows:
        k = 1.0 if r is rows[0] else ratio
        r["ratio_vs_base"] = round(k, 3)
        r["projected_h_lo"], r["projected_h_hi"] = round(a.base_hours[0] * k, 1), round(a.base_hours[1] * k, 1)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
