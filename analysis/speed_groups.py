"""Table 4.1 of the scorer-minutes paper: every full run (>= 20 sessions) of docs/paper/aggregates/speed_table.csv,
grouped by setup (GPU, concurrency, model, adapter), so that the rows add up to all full runs and sessions.

  python3 analysis/speed_groups.py [--out docs/paper/aggregates/speed_groups.csv]

Per group: runs, sessions, median and range of the per-run median tokens/s, and the factor against the scorer anchor
(the speed_table reference). Runs that ran several attempts of each task in parallel are their own group (their concurrency counts every attempt). Prints a markdown table.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPUS = [("4xL4", "Kaggle 4x L4"), ("H100", "H100 SXM 80 GB"), ("L40S", "L40S 48 GB"), ("RTX PRO 6000", "RTX PRO 6000 96 GB"),
        ("A100", "A100 80 GB"), ("RTX 5080", "RTX 5080 16 GB")]


def group(label: str) -> tuple[str, str, int, str]:
    gpu = next((name for key, name in GPUS if key in label), "other")
    m = re.search(r"(\d+) workers", label)
    workers = int(m.group(1)) if m else 0
    model = "E4B" if "E4B" in label else "26B-A4B" if "26B" in label else "31B"
    extra = "+ LoRA adapter" if "LoRA" in label else "parallel attempts" if "attempts x" in label else ""
    if gpu == "Kaggle 4x L4" and "TP1 x4" in label:
        extra = (extra + " 4 servers TP1").strip()
    elif gpu == "Kaggle 4x L4" and "TP2 x2" in label:
        extra = (extra + " 2 servers TP2").strip()
    return gpu, model, workers, extra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs/paper/aggregates/speed_groups.csv"))
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(ROOT / "docs/paper/aggregates/speed_table.csv")) if int(r["sessions"]) >= 20]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(group(r["label"]), []).append(r)
    out = []
    for (gpu, model, workers, extra), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2], kv[0][3])):
        tok = [float(r["median_tok_s"]) for r in rs]
        fac = [float(r["factor"]) for r in rs]
        out.append({"gpu": gpu, "model": model, "sessions_in_parallel": workers, "note": extra, "runs": len(rs),
                    "sessions": sum(int(r["sessions"]) for r in rs), "tok_s_median": round(st.median(tok), 1),
                    "tok_s_min": round(min(tok), 1), "tok_s_max": round(max(tok), 1),
                    "factor_median": round(st.median(fac), 2), "factor_min": round(min(fac), 2),
                    "factor_max": round(max(fac), 2)})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("| Setup | Model | Sessions in parallel | Runs (sessions) | Tokens/s per session, median (range) | Factor vs scorer |")
    print("|---|---|---|---|---|---|")
    for r in out:
        rng = f" ({r['tok_s_min']}-{r['tok_s_max']})" if r["runs"] > 1 else ""
        frng = f"{r['factor_median']}" if r["runs"] == 1 else f"{r['factor_min']}-{r['factor_max']}"
        print(f"| {r['gpu']}{', ' + r['note'] if r['note'] else ''} | {r['model']} | {r['sessions_in_parallel']} | "
              f"{r['runs']} ({r['sessions']:,}) | {r['tok_s_median']}{rng} | {frng} |")
    print(f"\nTotal: {sum(r['runs'] for r in out)} full runs, {sum(r['sessions'] for r in out):,} sessions")


if __name__ == "__main__":
    main()
